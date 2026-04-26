# Made by Xiaoke Yu g6836915 AND Arnon Vongarnon 6838864
# One Punch Form Checker
import cv2
import numpy as np
import time
import random
import math
import argparse
import os
from collections import deque
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("install ultralytics first: pip install ultralytics")
    exit()

# config
POSE_CONF = 0.4
HIT_RADIUS = 70
ENEMY_SPEED = 6
ENEMY_SPAWN_SEC = 3.0
MAX_ENEMIES = 5
LIVES = 3
COMBO_TIMEOUT = 3.5
VERDICT_FRAMES = 10
PUNCH_VEL_THRESH = 18
WRIST_HISTORY = 6

# keypoint indices
NOSE, L_SHLDR, R_SHLDR, L_WRIST, R_WRIST, L_ELBOW, R_ELBOW = 0, 5, 6, 9, 10, 7, 8


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


ENEMY_ROSTER = [
    "Deep Sea King",
    "Orochi",
    "Psykos",
    "Tatsumaki",
    "Black Sperm",
]

C_YELLOW = (0, 215, 255)
C_WHITE = (255, 255, 255)
C_RED = (30, 30, 220)
C_GREEN = (40, 200, 40)
C_BLACK = (0, 0, 0)
C_GOLD = (30, 185, 255)
C_ORANGE = (20, 140, 255)




def shadowed_text(frame, text, pos, scale=1.0, color=C_WHITE, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x+2, y+2), cv2.FONT_HERSHEY_DUPLEX,
                scale, C_BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)


# helpers

def get_kp(kpts, idx):
    if kpts is None or idx >= len(kpts):
        return None
    x, y, c = float(kpts[idx][0]), float(kpts[idx][1]), float(kpts[idx][2])
    return (x, y) if c >= POSE_CONF else None


def calculate_angle(a, b, c):
    """Calculates the angle at point b given points a, b, c."""
    if not (a and b and c):
        return 0
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def dist_2d(p1, p2):
    if not (p1 and p2):
        return 0
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


# form checker

class FormChecker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.verdict = "UNKNOWN"
        self._hist_l = deque(maxlen=WRIST_HISTORY)
        self._hist_r = deque(maxlen=WRIST_HISTORY)
        self._cooldown = 0
        self.l_wrist = None
        self.r_wrist = None
        self.arm_extended = False
        self.fast_enough = False
        self.shoulder_width = 100  # reference unit
        self.smooth_kpts = {}
        self.v_ratio = 0.0 # for UI bar

    def _update_smooth_kpts(self, pose_results):
        r = pose_results[0]
        if r.keypoints is None or len(r.keypoints.data) == 0:
            return
        kpts = r.keypoints.data[0]
        alpha = 0.5  # smoothing factor
        needed = [L_SHLDR, R_SHLDR, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, NOSE]
        for idx in needed:
            kp = get_kp(kpts, idx)
            if kp:
                if idx in self.smooth_kpts:
                    prev = self.smooth_kpts[idx]
                    self.smooth_kpts[idx] = (
                        prev[0]*(1-alpha) + kp[0]*alpha,
                        prev[1]*(1-alpha) + kp[1]*alpha
                    )
                else:
                    self.smooth_kpts[idx] = kp

    def update(self, pose_results):
        self._update_smooth_kpts(pose_results)

        if self._cooldown > 0:
            self._cooldown -= 1
            return self.verdict

        l_sh = self.smooth_kpts.get(L_SHLDR)
        l_el = self.smooth_kpts.get(L_ELBOW)
        l_wr = self.smooth_kpts.get(L_WRIST)
        r_sh = self.smooth_kpts.get(R_SHLDR)
        r_el = self.smooth_kpts.get(R_ELBOW)
        r_wr = self.smooth_kpts.get(R_WRIST)

        self.l_wrist = l_wr
        self.r_wrist = r_wr

        if l_sh and r_sh:
            self.shoulder_width = max(20, dist_2d(l_sh, r_sh))

        # normalize velocity threshold by player distance (torso size)
        norm_v_thresh = (PUNCH_VEL_THRESH / 100.0) * self.shoulder_width

        # LEFT ARM
        left_valid = False
        left_extended = False
        left_fast = False
        if l_sh and l_el and l_wr:
            self._hist_l.append(l_wr)
            angle = calculate_angle(l_sh, l_el, l_wr)
            # Extension: straight arm OR wrist significantly far from shoulder
            left_extended = (angle > 158) or (dist_2d(l_wr, l_sh) > 1.45 * dist_2d(l_el, l_sh))
            if len(self._hist_l) >= 3:
                # 2D velocity vector
                v = dist_2d(self._hist_l[-1], self._hist_l[0])
                left_fast = v > norm_v_thresh
            left_valid = left_extended and left_fast

        # RIGHT ARM
        right_valid = False
        right_extended = False
        right_fast = False
        if r_sh and r_el and r_wr:
            self._hist_r.append(r_wr)
            angle = calculate_angle(r_sh, r_el, r_wr)
            right_extended = (angle > 158) or (dist_2d(r_wr, r_sh) > 1.45 * dist_2d(r_el, r_sh))
            if len(self._hist_r) >= 3:
                v = dist_2d(self._hist_r[-1], self._hist_r[0])
                right_fast = v > norm_v_thresh
            right_valid = right_extended and right_fast

        self.arm_extended = left_extended or right_extended
        self.fast_enough = left_fast or right_fast
        
        # calculate max velocity ratio for UI
        v_l = dist_2d(self._hist_l[-1], self._hist_l[0]) if len(self._hist_l) >= 3 else 0
        v_r = dist_2d(self._hist_r[-1], self._hist_r[0]) if len(self._hist_r) >= 3 else 0
        self.v_ratio = max(v_l, v_r) / (norm_v_thresh + 1e-6)

        if left_valid or right_valid:
            self.verdict = "VALID"
            self._cooldown = 7
        elif self.arm_extended and not self.fast_enough:
            self.verdict = "UNKNOWN"
        elif not self.arm_extended and (left_fast or right_fast):
            self.verdict = "FOUL"
        else:
            self.verdict = "UNKNOWN"

        return self.verdict


# enemy

class Enemy:
    TARGET_H = 110

    def __init__(self, x, fw, name):
        self.x = float(x)
        self.y = -60.0
        self.fw = fw
        self.name = name
        self.alive = True
        self.flash = 0
        self.dodge_dx = 0.0
        self.bob_t = random.uniform(0, math.tau)
        self.w, self.h = 70, 90

    def update(self):
        self.y += ENEMY_SPEED
        self.bob_t += 0.08
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
        
        # Color: Flash white on hit, otherwise red
        col = C_WHITE if self.flash > 0 else C_RED
        
        # Draw the hitbox circle
        radius = self.w // 2
        cv2.circle(frame, (x, y), radius, col, 2)
        
        # Subtle fill
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), radius, col, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    def near_fist(self, fx, fy):
        return (abs(fx - self.x) < self.w // 2 + HIT_RADIUS and
                abs(fy - self.y) < self.h // 2 + HIT_RADIUS)




# game

class Game:
    def __init__(self, fw, fh, enemy_pool):
        self.fw, self.fh = fw, fh
        self.pool = enemy_pool
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
        self.verdict_col = C_WHITE
        self.verdict_left = 0
        self.enemies = []
        self.last_spawn = time.time()
        self.start_time = time.time()

    def spawn_enemy(self):
        if len([e for e in self.enemies if e.alive]) >= MAX_ENEMIES:
            return
        x = random.randint(80, self.fw - 80)
        name = random.choice(self.pool)
        self.enemies.append(Enemy(x, self.fw, name))
        self.last_spawn = time.time()

    def update_enemies(self):
        for e in self.enemies:
            if e.alive:
                e.update()
                if e.y > self.fh + e.h + 20:
                    e.alive = False
                    self.lives -= 1
                    if self.lives <= 0:
                        self.over = True

        self.enemies = [e for e in self.enemies
                        if e.alive or e.y <= self.fh + e.h + 20]
        self.enemies = [e for e in self.enemies if e.y <= self.fh + e.h + 20]

        if time.time() - self.last_spawn > ENEMY_SPAWN_SEC:
            self.spawn_enemy()

    def register_hit(self, enemy):
        now = time.time()
        self.combo = self.combo + 1 if now - self.last_hit_t < COMBO_TIMEOUT else 1
        self.last_hit_t = now
        self.max_combo = max(self.max_combo, self.combo)
        pts = 100 * self.combo
        self.score += pts
        self.hits += 1
        label = f"+{pts}  x{self.combo} COMBO!" if self.combo > 1 else f"+{pts}"
        self.popups.append({"text": label, "x": int(
            enemy.x), "y": int(enemy.y), "t": now})
        self._verdict("ONE PUNCH!", C_GOLD)

    def register_foul(self):
        self.fouls += 1
        self.combo = 0
        for e in self.enemies:
            if e.alive:
                e.dodge_dx = random.choice([-4.0, 4.0])
        self._verdict("SLOPPY FORM!", C_RED)

    def _verdict(self, txt, col):
        self.verdict_txt = txt
        self.verdict_col = col
        self.verdict_left = VERDICT_FRAMES

    def draw(self, frame, form):
        h, w = frame.shape[:2]

        for e in self.enemies:
            if e.alive:
                e.draw(frame)

        # top bar
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 58), (8, 8, 20), -1)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (0, 0), (5, 58), C_YELLOW, -1)
        shadowed_text(frame, f"SCORE  {self.score:07d}", (18, 38),
                      scale=1.0, color=C_YELLOW)
        if self.combo > 1:
            col = C_RED if self.combo >= 5 else C_ORANGE
            shadowed_text(frame, f"x{self.combo} COMBO!",
                          (w//2-75, 100), color=col)
        elapsed = int(time.time() - self.start_time)
        shadowed_text(frame, f"{elapsed:04d}s", (w-90, 38), scale=0.75)

        # lives
        for i in range(LIVES):
            cx = w - 22 - i * 30
            col = C_RED if i < self.lives else (60, 60, 60)
            cv2.circle(frame, (cx, 75), 11, col, -1)
            cv2.circle(frame, (cx, 75), 11, C_GOLD if i <
                       self.lives else (40, 40, 40), 1)

        # punch panel
        px, py = w - 215, 100
        ov2 = frame.copy()
        cv2.rectangle(ov2, (px-8, py-8), (px+205, py+98), (8, 8, 20), -1)
        cv2.addWeighted(ov2, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (px-8, py-8), (px+205, py+98), C_YELLOW, 1)
        shadowed_text(frame, "PUNCH ANALYSIS", (px, py+15), scale=0.48,
                      color=C_YELLOW, thickness=1)

        # Speed bar
        bar_w = 190
        bar_x, bar_y = px, py + 28
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6), (40, 40, 40), -1)
        fill_w = int(bar_w * min(1.0, form.v_ratio))
        bar_col = C_GREEN if form.v_ratio >= 1.0 else C_ORANGE
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 6), bar_col, -1)
        cv2.putText(frame, "SPEED", (px, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_WHITE, 1)

        for i, (label, ok) in enumerate([("Extension", form.arm_extended),
                                         ("Velocity",  form.fast_enough)]):
            col = C_GREEN if ok else (100, 100, 100)
            dot_y = py + 52 + i*22
            cv2.circle(frame, (px + 6, dot_y), 4, col, -1)
            cv2.putText(frame, label, (px + 18, dot_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE if ok else (160, 160, 160), 1, cv2.LINE_AA)

        for wrist, label in [(form.l_wrist, "L"), (form.r_wrist, "R")]:
            if wrist is None:
                continue
            col = C_GREEN if form.arm_extended else C_ORANGE
            cv2.circle(frame, (int(wrist[0]), int(wrist[1])), 10, col, -1)
            cv2.circle(frame, (int(wrist[0]), int(wrist[1])), 10, C_WHITE, 1)
            cv2.putText(frame, label, (int(wrist[0])+12, int(wrist[1])-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        # popups
        now = time.time()
        self.popups = [p for p in self.popups if now - p["t"] < 0.8]
        for p in self.popups:
            age = now - p["t"]
            fade = max(0, 1 - age / 0.5)
            color = tuple(int(c * fade) for c in C_GOLD)
            shadowed_text(frame, p["text"], (p["x"]-55, int(p["y"] - age*80)),
                          scale=0.85, color=C_GOLD)

        # verdict banner
        if self.verdict_left > 0:
            self.verdict_left -= 1
            alpha = min(1.0, self.verdict_left / 20.0)
            tw = cv2.getTextSize(
                self.verdict_txt, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)[0][0]
            tx, ty = w//2 - tw//2, h//2 + 25
            ov3 = frame.copy()
            cv2.rectangle(ov3, (tx-24, ty-52),
                          (tx+tw+24, ty+18), (8, 8, 20), -1)
            cv2.addWeighted(ov3, 0.75*alpha, frame, 1-0.75*alpha, 0, frame)
            cv2.putText(frame, self.verdict_txt, (tx+3, ty+3),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, C_BLACK, 6, cv2.LINE_AA)
            cv2.putText(frame, self.verdict_txt, (tx, ty),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, self.verdict_col, 3, cv2.LINE_AA)

        # bottom stats
        cv2.putText(frame,
                    f"Shots:{self.shots}  Hits:{self.hits}  Fouls:{self.fouls}  MaxCombo:x{self.max_combo}",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    def draw_gameover(self, frame):
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (w//5, h//5), (4*w//5, 4*h//5), (8, 8, 20), -1)
        cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (w//5, h//5), (4*w//5, 4*h//5), C_YELLOW, 2)
        shadowed_text(frame, "ONE PUNCH MAN", (w//2-165, h//2-80),
                      scale=1.3, color=C_YELLOW, thickness=3)
        shadowed_text(frame, "GAME OVER", (w//2-105, h//2-38),
                      scale=1.1, color=C_RED, thickness=2)
        for i, (txt, col) in enumerate([
            (f"Score:       {self.score:07d}", C_WHITE),
            (f"Enemies hit: {self.hits}",       C_GREEN),
            (f"Fouls:       {self.fouls}",       C_RED),
            (f"Max Combo:   x{self.max_combo}",  C_ORANGE),
            ("Press  R  to restart",             (140, 140, 140)),
        ]):
            shadowed_text(frame, txt, (w//2-120, h//2+5+i*32),
                          scale=0.65, color=col, thickness=1)


# Main

def run(source, show_skeleton=True):
    enemy_pool = ENEMY_ROSTER


    print("loading YOLO pose model...")
    pose_model = YOLO(resource_path("yolov8n-pose.pt"))

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    
    if not cap.isOpened():
        print(f"can't open: {source}")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera: {fw}x{fh}")

    form = FormChecker()
    game = Game(fw, fh, enemy_pool)
    prev_verdict = "UNKNOWN"
    frame_idx = 0
    paused = False

    win = "ONE PUNCH MAN  [R=restart  P=pause  S=skeleton  Q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
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
                prev_verdict = "UNKNOWN"
            continue

        pose_res = pose_model(frame, verbose=False)


        form_now = form.update(pose_res) if len(pose_res) > 0 else form.verdict

        # punch detection
        if form_now == "VALID" and prev_verdict != "VALID":
            game.shots += 1

            wrists = []
            if form.l_wrist:
                wrists.append(form.l_wrist)
            if form.r_wrist:
                wrists.append(form.r_wrist)

            hit_e = None
            closest = 999999

            for wx, wy in wrists:
                for e in [e for e in game.enemies if e.alive]:
                    d = math.hypot(wx - e.x, wy - e.y)
                    if d < closest:
                        closest = d
                        hit_e = e

            if hit_e and closest < 220:
                hit_e.alive = False
                hit_e.flash = 12
                game.register_hit(hit_e)
            else:
                game.register_foul()

        prev_verdict = form_now
        game.update_enemies()

        annotated = pose_res[0].plot(img=display) if show_skeleton and len(
            pose_res) > 0 else display
        game.draw(annotated, form)

        # intro hint
        if frame_idx < 50:
            alpha = max(0.0, 1.0 - frame_idx / 100.0)
            h2, w2 = annotated.shape[:2]
            msg = "PUNCH THE MONSTERS!"
            tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)[0][0]
            tx, ty = w2//2 - tw//2, h2//2
            ov = annotated.copy()
            cv2.rectangle(ov, (tx-20, ty-50),
                          (tx+tw+20, ty+18), (8, 8, 20), -1)
            cv2.addWeighted(ov, 0.7*alpha, annotated,
                            1-0.7*alpha, 0, annotated)
            col = tuple(int(c * alpha) for c in C_YELLOW)
            shadowed_text(annotated, msg, (tx, ty),
                          scale=1.1, color=col, thickness=3)

        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):
            show_skeleton = not show_skeleton
        elif key == ord('r'):
            game.reset()
            form.reset()
            prev_verdict = "UNKNOWN"

    cap.release()
    cv2.destroyAllWindows()
    print(f"done. score={game.score}  max combo=x{game.max_combo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",      default="0")
    parser.add_argument("--no-skeleton", action="store_true")
    args = parser.parse_args()
    run(args.source, show_skeleton=not args.no_skeleton)
