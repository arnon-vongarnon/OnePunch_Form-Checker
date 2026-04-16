import cv2
import numpy as np
import time
import random
import math
import argparse
from collections import deque

try:
    from ultralytics import YOLO
except ImportError:
    print("install ultralytics first: pip install ultralytics")
    exit()

# ------- config -------
POSE_CONF = 0.4
HIT_RADIUS = 55
REF_SPEED = 1.2
REF_SPAWN_SEC = 3.0
MAX_REFS = 5
LIVES = 3
COMBO_TIMEOUT = 3.5
VERDICT_FRAMES = 80

L_ELBOW = 7
R_ELBOW = 8

PUNCH_VEL_THRESH = 40
WRIST_HISTORY = 5

# yolo keypoint indices
NOSE, L_SHLDR, R_SHLDR, L_WRIST, R_WRIST = 0, 5, 6, 9, 10

REFEREES = [
    ("M. Atkinson", (30, 30, 180)),
    ("M. Oliver",   (20, 100, 200)),
    ("K. Friend",   (160, 30, 30)),
    ("A. Taylor",   (180, 80, 20)),
]


# ------- helpers -------

def get_kp(kpts, idx):
    if kpts is None or idx >= len(kpts):
        return None
    x, y, c = float(kpts[idx][0]), float(kpts[idx][1]), float(kpts[idx][2])
    return (x, y) if c >= POSE_CONF else None


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ------- throw form checker -------


class FormChecker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.verdict = "UNKNOWN"
        self._wrist_hist = deque(maxlen=WRIST_HISTORY)
        self._cooldown = 0   # frames before next punch can register
        # for drawing
        self.l_wrist = None
        self.r_wrist = None
        self.l_elbow = None
        self.r_elbow = None
        self.arm_extended = False
        self.fast_enough = False

    def update(self, pose_results):
        # cooldown so one punch doesn't trigger 10 times
        if self._cooldown > 0:
            self._cooldown -= 1
            return self.verdict

        r = pose_results[0]
        if r.keypoints is None or len(r.keypoints.data) == 0:
            return self.verdict

        kpts = r.keypoints.data[0]

        # person faces left, so use LEFT side keypoints
        # (their left arm is the punching arm from this view)
        l_shldr = get_kp(kpts, L_SHLDR)
        l_elbow = get_kp(kpts, L_ELBOW)
        l_wrist = get_kp(kpts, L_WRIST)
        r_wrist = get_kp(kpts, R_WRIST)
        r_elbow = get_kp(kpts, R_ELBOW)

        self.l_wrist = l_wrist
        self.r_wrist = r_wrist
        self.l_elbow = l_elbow
        self.r_elbow = r_elbow

        if not (l_shldr and l_elbow and l_wrist):
            return self.verdict

        # track wrist position history
        self._wrist_hist.append(l_wrist[0])

        # rule 1: arm is extended (wrist further left than elbow, elbow further left than shoulder)
        # facing left = smaller x values are further forward
        self.arm_extended = l_wrist[0] < l_elbow[0] < l_shldr[0]

        # rule 2: wrist moved fast enough to count as a punch
        self.fast_enough = False
        if len(self._wrist_hist) >= 2:
            vel = abs(self._wrist_hist[-1] - self._wrist_hist[-2])
            self.fast_enough = vel > PUNCH_VEL_THRESH

        if self.arm_extended and self.fast_enough:
            self.verdict = "VALID"
            self._cooldown = 15   # ~0.5s cooldown at 30fps
        else:
            self.verdict = "FOUL" if not self.arm_extended else "UNKNOWN"

        return self.verdict


# ------- referee -------

class Ref:
    def __init__(self, x, fw):
        self.x = x
        self.y = -50
        self.fw = fw
        name, color = random.choice(REFEREES)
        self.name = name
        self.color = color
        self.alive = True
        self.flash = 0
        self.dodge_dx = 0.0
        self.w = 70
        self.h = 80

    def update(self):
        self.y += REF_SPEED
        if self.dodge_dx:
            self.x = max(40, min(self.fw - 40, self.x + self.dodge_dx))
            self.dodge_dx *= 0.85
            if abs(self.dodge_dx) < 0.3:
                self.dodge_dx = 0.0
        if self.flash > 0:
            self.flash -= 1

    def draw(self, frame):
        x, y = int(self.x), int(self.y)
        hw, hh = self.w // 2, self.h // 2
        color = (255, 255, 255) if self.flash > 0 else self.color

        cv2.rectangle(frame, (x-hw, y-hh//2), (x+hw, y+hh), color, -1)
        cv2.rectangle(frame, (x-hw, y-hh//2), (x+hw, y+hh), (0, 0, 0), 1)
        for sx in range(x - hw + 8, x + hw, 14):
            cv2.line(frame, (sx, y-hh//2), (sx, y+hh), (0, 0, 0), 1)

        # head
        cv2.circle(frame, (x, y-hh//2-18), 18, (180, 140, 110), -1)
        cv2.circle(frame, (x, y-hh//2-18), 18, (0, 0, 0), 1)

        # red card
        cv2.rectangle(frame, (x+hw-5, y-hh//2-10),
                      (x+hw+8, y-hh//2+10), (40, 40, 220), -1)

        # name
        (tw, _), _ = cv2.getTextSize(self.name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(frame, self.name, (x - tw//2, y+hh+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    def hits_ball(self, bx, by):
        return (abs(bx - self.x) < self.w//2 + HIT_RADIUS and
                abs(by - self.y) < self.h//2 + HIT_RADIUS)


# ------- game -------

class Game:
    def __init__(self, fw, fh):
        self.fw, self.fh = fw, fh
        self.reset()

    def reset(self):
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.lives = LIVES
        self.hits = 0
        self.shots = 0
        self.fouls = 0
        self.over = False
        self.last_hit_t = 0.0
        self.popups = []
        self.verdict_txt = ""
        self.verdict_col = (255, 255, 255)
        self.verdict_left = 0
        self.refs = []
        self.last_spawn = time.time()

    def spawn_ref(self):
        alive = [r for r in self.refs if r.alive]
        if len(alive) >= MAX_REFS:
            return
        x = random.randint(80, self.fw - 80)
        self.refs.append(Ref(x, self.fw))
        self.last_spawn = time.time()

    def update_refs(self):
        escaped = []
        for r in self.refs:
            if not r.alive:
                continue
            r.update()
            if r.y > self.fh + r.h:
                escaped.append(r)

        self.refs = [r for r in self.refs if r.alive and r.y <= self.fh + r.h]

        for _ in escaped:
            self.lives -= 1
            if self.lives <= 0:
                self.over = True

        if time.time() - self.last_spawn > REF_SPAWN_SEC:
            self.spawn_ref()

    def register_hit(self, ref):
        now = time.time()
        self.combo = self.combo + 1 if now - self.last_hit_t < COMBO_TIMEOUT else 1
        self.last_hit_t = now
        self.max_combo = max(self.max_combo, self.combo)
        pts = 100 * self.combo
        self.score += pts
        self.hits += 1
        label = f"+{pts}  x{self.combo} COMBO!" if self.combo > 1 else f"+{pts}"
        self.popups.append(
            {"text": label, "x": int(ref.x), "y": int(ref.y), "t": now})
        self._verdict("VALID  HIT!", (40, 200, 40))

    def register_foul(self):
        self.fouls += 1
        self.combo = 0
        for r in self.refs:
            if r.alive:
                r.dodge_dx = random.choice([-3.5, 3.5])
        self._verdict("FOUL THROW!", (40, 40, 220))

    def _verdict(self, txt, col):
        self.verdict_txt = txt
        self.verdict_col = col
        self.verdict_left = VERDICT_FRAMES

    def draw(self, frame, form):
        h, w = frame.shape[:2]

        # refs
        for r in self.refs:
            if r.alive:
                r.draw(frame)

        # pose panel
        self._draw_pose_panel(frame, form)

        # top hud
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 52), (8, 8, 8), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, f"SCORE  {self.score:06d}", (12, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (220, 200, 0), 2)
        if self.combo > 1:
            col = (0, 140, 255) if self.combo >= 3 else (0, 210, 255)
            cv2.putText(frame, f"x{self.combo} COMBO", (w//2-65, 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, col, 2)
        for i in range(LIVES):
            col = (40, 40, 220) if i < self.lives else (140, 140, 140)
            cv2.circle(frame, (w - 25 - i*32, 26), 11, col, -1)

        # bottom stats
        cv2.putText(frame,
                    f"Shots:{self.shots}  Hits:{self.hits}  Fouls:{self.fouls}  MaxCombo:x{self.max_combo}",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

        # popups
        now = time.time()
        self.popups = [p for p in self.popups if now - p["t"] < 1.5]
        for p in self.popups:
            age = now - p["t"]
            cv2.putText(frame, p["text"], (p["x"]-50, int(p["y"] - age*55)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 210, 255), 2)

        # verdict banner
        if self.verdict_left > 0:
            self.verdict_left -= 1
            alpha = min(1.0, self.verdict_left / 20)
            tw = cv2.getTextSize(
                self.verdict_txt, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0][0]
            tx, ty = w//2 - tw//2, h//2 + 20
            ov2 = frame.copy()
            cv2.rectangle(ov2, (tx-20, ty-45),
                          (tx+tw+20, ty+15), (10, 10, 10), -1)
            cv2.addWeighted(ov2, 0.7*alpha, frame, 1-0.7*alpha, 0, frame)
            cv2.putText(frame, self.verdict_txt, (tx+2, ty+2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 4)
            cv2.putText(frame, self.verdict_txt, (tx, ty),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, self.verdict_col, 2)

    def _draw_pose_panel(self, frame, form):
        h, w = frame.shape[:2]
        px, py = w - 220, 65
        ov = frame.copy()
        cv2.rectangle(ov, (px-5, py-5), (px+210, py+75), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (px-5, py-5), (px+210, py+75), (220, 200, 0), 1)
        cv2.putText(frame, "PUNCH FORM", (px, py+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 200, 0), 1)

        for i, (label, ok) in enumerate([
            ("Arm extended", form.arm_extended),
            ("Fast enough",  form.fast_enough),
        ]):
            col = (40, 200, 40) if ok else (
                40, 40, 220)  # ← col defined here now
            icon = "v" if ok else "x"
            cv2.putText(frame, f"{icon} {label}", (px, py+36+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        for wrist, label in [(form.l_wrist, "LW"), (form.r_wrist, "RW")]:
            if wrist is None:
                continue
            col = (40, 200, 40) if form.arm_extended else (0, 140, 255)
            cv2.circle(frame, (int(wrist[0]), int(wrist[1])), 9, col, -1)
            cv2.putText(frame, label, (int(wrist[0])+10, int(wrist[1])-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    def draw_gameover(self, frame):
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (w//4, h//4), (3*w//4, 3*h//4), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
        cv2.putText(frame, "FINAL WHISTLE", (w//2-145, h//2-55),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (40, 40, 220), 3)
        for i, (txt, col) in enumerate([
            (f"Score:      {self.score}",      (255, 255, 255)),
            (f"Hits:       {self.hits}",        (40, 200, 40)),
            (f"Foul Throws:{self.fouls}",       (40, 40, 220)),
            (f"Max Combo:  x{self.max_combo}",  (0, 210, 255)),
            ("Press R to restart",              (140, 140, 140)),
        ]):
            cv2.putText(frame, txt, (w//2-100, h//2-10+i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 1 if i == 4 else 2)


# ------- main -------

def run(source):
    print("loading models...")
    pose_model = YOLO("yolov8n-pose.pt")

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"can't open: {source}")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera: {fw}x{fh}")

    form = FormChecker()
    game = Game(fw, fh)

    prev_verdict = "UNKNOWN"   # ← ADD HERE
    frame_idx = 0

    win = "Referee Shooter  [R=restart  P=pause  Q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if not str(source).isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            frame_idx += 1

        display = frame.copy()

        if game.over:
            game.draw_gameover(display)
            cv2.imshow(win, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                game.reset()
                form.reset()
                prev_verdict = "UNKNOWN"   # ← ADD HERE
            continue

        # pose
        pose_res = pose_model(frame, verbose=False)
        if len(pose_res) > 0:
            form_now = form.update(pose_res)
        else:
            form_now = form.verdict

        # punch detection — fires once per punch   ← ADD THIS WHOLE BLOCK
        if form_now == "VALID" and prev_verdict != "VALID":
            game.shots += 1
            hit_ref = None
            if form.l_wrist is not None:
                wx, wy = form.l_wrist[0], form.l_wrist[1]
                closest = 999999
                for ref in [r for r in game.refs if r.alive]:
                    d = math.hypot(wx - ref.x, wy - ref.y)
                    if d < closest:
                        closest = d
                        hit_ref = ref
                if hit_ref and closest < 200:
                    hit_ref.alive = False
                    hit_ref.flash = 10
                    game.register_hit(hit_ref)
                else:
                    game.register_foul()

        prev_verdict = form_now   # ← ADD HERE

        game.update_refs()

        annotated = pose_res[0].plot(img=display) if len(
            pose_res) > 0 else display
        game.draw(annotated, form)

        if frame_idx < 120:
            h2, w2 = annotated.shape[:2]
            cv2.putText(annotated, "punch the referees!",
                        (w2//2-120, h2//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 200, 0), 2)

        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('r'):
            game.reset()
            form.reset()
            prev_verdict = "UNKNOWN"   # ← ADD HERE

    cap.release()
    cv2.destroyAllWindows()
    print(f"done. score={game.score}  max combo=x{game.max_combo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    args = parser.parse_args()
    run(args.source)
