"""
AR Referee Shooter  —  "Give 'Em What They Deserve!"
======================================================
Throw a real ball at controversial Premier League referees falling from the sky.
BUT — your throw must follow FIFA throw-in rules (upper body only):
  Rule 1: Both hands must be above shoulder level
  Rule 2: Both hands must be above head (pass over the head)

If your form is illegal → FOUL THROW → ball is invalid, referee dodges.
If your form is correct → trajectory is calculated → hit = score!

YOLO models used:
  yolov8n.pt       — detects the ball (class 32 = sports ball)
  yolov8n-pose.pt  — detects player upper body pose (17 keypoints)

Usage:
    python ar_referee.py              # webcam
    python ar_referee.py --source 1   # webcam index 1
    python ar_referee.py --source vid.mp4

Controls:
    Q / ESC  — quit
    R        — restart
    P        — pause
    S        — screenshot
"""

import cv2
import numpy as np
import argparse
import time
import random
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[ERROR] ultralytics not found.  pip install ultralytics")
    YOLO_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # Models
    "ball_model":        "yolov8n.pt",
    "pose_model":        "yolov8n-pose.pt",
    "ball_class":        32,          # COCO sports ball
    "ball_conf":         0.20,        # lower = more sensitive
    "pose_conf":         0.4,         # keypoint confidence threshold

    # Ball tracking
    "history_len":       14,
    "fit_min_points":    5,
    "hit_radius":        55,          # px tolerance for hitting a referee

    # Throw-in form (upper body rules only)
    "hands_above_head_px":       -20, # wrists must be above (nose_y + offset)
    "violation_min_frames":       3,  # consecutive frames to confirm violation
    "throw_settle_frames":        3,  # ignore form for first N frames of throw

    # Game
    "referee_speed":     1.2,         # px/frame downward
    "referee_spawn_sec": 3.0,
    "max_referees":      5,
    "lives":             3,
    "combo_timeout_sec": 3.5,
    "verdict_hold_frames": 80,        # frames to show FOUL/VALID verdict
}

# YOLOv8 Pose keypoint indices
KP_NOSE          = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER= 6
KP_LEFT_WRIST    = 9
KP_RIGHT_WRIST   = 10

# Colours (BGR)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,   0,   0)
C_RED    = (40,  40,  220)
C_GREEN  = (40,  200, 40)
C_CYAN   = (220, 200, 0)
C_YELLOW = (0,   210, 255)
C_ORANGE = (0,   140, 255)
C_PURPLE = (200, 50,  150)
C_GRAY   = (140, 140, 140)
C_PINK   = (180, 100, 220)

# Controversial Premier League referees
REFEREES = [
    {"name": "M. Atkinson",    "color": (30,  30,  180), "accent": C_YELLOW},
    {"name": "M. Oliver",      "color": (20,  100, 200), "accent": C_CYAN},
    {"name": "K. Friend",      "color": (160, 30,  30),  "accent": C_ORANGE},
    {"name": "C. Kavanagh",    "color": (30,  140, 30),  "accent": C_WHITE},
    {"name": "S. Attwell",     "color": (100, 30,  160), "accent": C_YELLOW},
    {"name": "A. Taylor",      "color": (180, 80,  20),  "accent": C_WHITE},
    {"name": "P. Tierney",     "color": (20,  80,  160), "accent": C_ORANGE},
    {"name": "D. Coote",       "color": (60,  60,  60),  "accent": C_CYAN},
]

# Throw form verdict states
FORM_UNKNOWN = "UNKNOWN"
FORM_VALID   = "VALID"
FORM_FOUL    = "FOUL THROW"


# ─────────────────────────────────────────────────────────────────────────────
#  KEYPOINT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_kp(kpts_data, idx, conf_thresh):
    """Return (x, y, conf) or None."""
    if kpts_data is None or idx >= len(kpts_data):
        return None
    kp = kpts_data[idx]
    try:
        x, y, c = float(kp[0]), float(kp[1]), float(kp[2])
    except Exception:
        return None
    return (x, y, c) if c >= conf_thresh else None


# ─────────────────────────────────────────────────────────────────────────────
#  THROW FORM CHECKER  (upper body only — no feet/sideline)
# ─────────────────────────────────────────────────────────────────────────────

class ThrowFormChecker:
    """
    Uses pose keypoints to verify FIFA throw-in upper-body rules:
      Rule 1 — Both hands above shoulder level
      Rule 2 — Both hands above head (above nose)

    Violations must persist for `violation_min_frames` consecutive frames
    to avoid single-frame noise false positives.
    """

    def __init__(self, cfg):
        self.cfg    = cfg
        self._consec_violation = 0   # consecutive frames with bad form
        self._consec_valid     = 0
        self.current_form      = FORM_UNKNOWN
        self.throw_frame       = 0   # frames since throw started

        # Live values (updated each frame, used for HUD)
        self.hands_above_shoulders = False
        self.hands_above_head      = False
        self.nose_y                = None
        self.l_wrist               = None
        self.r_wrist               = None

    def reset(self):
        self.__init__(self.cfg)

    def update(self, pose_results) -> str:
        """
        Feed one frame of pose results.
        Returns current form verdict: FORM_UNKNOWN / FORM_VALID / FORM_FOUL
        """
        self.throw_frame += 1
        t = self.cfg["pose_conf"]

        # Reset live values
        self.hands_above_shoulders = False
        self.hands_above_head      = False
        self.nose_y                = None
        self.l_wrist               = None
        self.r_wrist               = None

        # Extract keypoints from first detected person
        r = pose_results[0]
        if r.keypoints is None or len(r.keypoints.data) == 0:
            return self.current_form

        kpts = r.keypoints.data[0]

        nose    = get_kp(kpts, KP_NOSE,           t)
        l_shldr = get_kp(kpts, KP_LEFT_SHOULDER,  t)
        r_shldr = get_kp(kpts, KP_RIGHT_SHOULDER, t)
        l_wrist = get_kp(kpts, KP_LEFT_WRIST,     t)
        r_wrist = get_kp(kpts, KP_RIGHT_WRIST,    t)

        self.l_wrist = l_wrist
        self.r_wrist = r_wrist
        if nose:
            self.nose_y = nose[1]

        # ── Rule 1: Both hands above shoulders ───────────────────────────────
        r1 = False
        if l_wrist and r_wrist and l_shldr and r_shldr:
            if l_wrist[1] < l_shldr[1] and r_wrist[1] < r_shldr[1]:
                r1 = True
        self.hands_above_shoulders = r1

        # ── Rule 2: Both hands above head (nose) ─────────────────────────────
        r2 = False
        offset = self.cfg["hands_above_head_px"]
        if l_wrist and r_wrist and nose:
            if l_wrist[1] < nose[1] + offset and r_wrist[1] < nose[1] + offset:
                r2 = True
        self.hands_above_head = r2

        # ── During settle frames: don't judge yet ─────────────────────────────
        settle = self.cfg["throw_settle_frames"]
        if self.throw_frame <= settle:
            return FORM_UNKNOWN

        # ── Determine if form is good this frame ─────────────────────────────
        good_form_this_frame = r1 and r2
        min_f = self.cfg["violation_min_frames"]

        if good_form_this_frame:
            self._consec_valid     += 1
            self._consec_violation  = 0
            if self._consec_valid >= min_f:
                self.current_form = FORM_VALID
        else:
            self._consec_violation += 1
            self._consec_valid      = 0
            if self._consec_violation >= min_f:
                self.current_form = FORM_FOUL

        return self.current_form


# ─────────────────────────────────────────────────────────────────────────────
#  BALL TRACKER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BallDetection:
    x: float
    y: float
    radius: float
    conf: float
    timestamp: float


class BallTracker:
    STATE_ABSENT  = "ABSENT"
    STATE_IN_HAND = "HOLDING"
    STATE_FLYING  = "FLYING"

    def __init__(self, cfg):
        self.cfg          = cfg
        self.history      = deque(maxlen=cfg["history_len"])
        self.state        = self.STATE_ABSENT
        self._still_count = 0

    def update(self, frame, model) -> Optional[BallDetection]:
        results = model(frame, verbose=False)
        det     = self._best_ball(results)
        now     = time.time()

        if det:
            self.history.append(BallDetection(
                x=det[0], y=det[1], radius=det[2], conf=det[3], timestamp=now
            ))
            self._update_state()
        return self.history[-1] if self.history else None

    def _best_ball(self, results):
        best_conf, best = 0, None
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if int(box.cls[0]) != CFG["ball_class"]:
                    continue
                conf = float(box.conf[0])
                if conf < self.cfg["ball_conf"] or conf <= best_conf:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                rad = float(max(x2 - x1, y2 - y1) / 2)
                best, best_conf = (cx, cy, rad, conf), conf
        return best

    def _update_state(self):
        if len(self.history) < 3:
            self.state = self.STATE_IN_HAND
            return
        recent = list(self.history)[-3:]
        dist   = math.hypot(recent[-1].x - recent[0].x,
                            recent[-1].y - recent[0].y)
        if dist < 15:
            self._still_count += 1
            if self._still_count > 5:
                self.state = self.STATE_IN_HAND
        else:
            self._still_count = 0
            self.state        = self.STATE_FLYING

    def clear(self):
        self.history.clear()
        self.state        = self.STATE_ABSENT
        self._still_count = 0


# ─────────────────────────────────────────────────────────────────────────────
#  TRAJECTORY FITTER
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryFitter:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit_and_predict(self, history, predict_frames=50):
        if len(history) < self.cfg["fit_min_points"]:
            return []
        pts = list(history)
        n   = len(pts)
        ts  = np.arange(n, dtype=float)
        xs  = np.array([p.x for p in pts])
        ys  = np.array([p.y for p in pts])
        try:
            px = np.polyfit(ts, xs, 1)
            py = np.polyfit(ts, ys, 2)
        except np.linalg.LinAlgError:
            return []
        predicted = []
        for i in range(1, predict_frames + 1):
            t = n - 1 + i
            predicted.append((float(np.polyval(px, t)),
                               float(np.polyval(py, t))))
        return predicted


# ─────────────────────────────────────────────────────────────────────────────
#  REFEREE  (the "monster")
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Referee:
    x: float
    y: float
    ref_data: dict
    width: int  = 70
    height: int = 80
    alive: bool = True
    hit_flash: int = 0
    dodge_dx: float = 0.0   # when foul throw, ref dodges sideways


class RefereeManager:
    def __init__(self, cfg, fw, fh):
        self.cfg       = cfg
        self.fw        = fw
        self.fh        = fh
        self.referees  = []
        self.last_spawn= time.time()

    def update(self, fw, fh):
        self.fw, self.fh = fw, fh
        escaped = []
        for ref in self.referees:
            if not ref.alive:
                continue
            ref.y += self.cfg["referee_speed"]
            if ref.dodge_dx != 0:
                ref.x = max(40, min(fw - 40, ref.x + ref.dodge_dx))
                ref.dodge_dx *= 0.85   # decelerate
                if abs(ref.dodge_dx) < 0.3:
                    ref.dodge_dx = 0.0
            if ref.hit_flash > 0:
                ref.hit_flash -= 1
            if ref.y > fh + ref.height:
                escaped.append(ref)

        self.referees = [r for r in self.referees
                         if r.alive and r.y <= fh + r.height]

        now = time.time()
        if (now - self.last_spawn > self.cfg["referee_spawn_sec"] and
                len(self.get_alive()) < self.cfg["max_referees"]):
            self._spawn()
            self.last_spawn = now

        return escaped

    def _spawn(self):
        margin = 80
        x      = random.randint(margin, self.fw - margin)
        ref    = Referee(x=x, y=-50, ref_data=random.choice(REFEREES))
        self.referees.append(ref)

    def get_alive(self):
        return [r for r in self.referees if r.alive]

    def dodge_all(self):
        """Called on foul throw — referees dodge sideways smugly."""
        for ref in self.get_alive():
            ref.dodge_dx = random.choice([-3.5, 3.5])

    def reset(self, fw, fh):
        self.referees.clear()
        self.fw, self.fh  = fw, fh
        self.last_spawn   = time.time()


# ─────────────────────────────────────────────────────────────────────────────
#  HIT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class HitDetector:
    def __init__(self, cfg):
        self.cfg = cfg

    def check_trajectory_hits(self, predicted, referees):
        hits   = []
        radius = self.cfg["hit_radius"]
        for ref in referees:
            if not ref.alive:
                continue
            hw, hh = ref.width // 2, ref.height // 2
            for px, py in predicted:
                if (ref.x - hw - radius < px < ref.x + hw + radius and
                        ref.y - hh - radius < py < ref.y + hh + radius):
                    hits.append(ref)
                    break
        return hits

    def check_realtime_hit(self, ball_history, referees):
        if not ball_history:
            return []
        latest = ball_history[-1]
        hits   = []
        radius = self.cfg["hit_radius"]
        for ref in referees:
            if not ref.alive:
                continue
            dist = math.hypot(latest.x - ref.x, latest.y - ref.y)
            hw   = max(ref.width, ref.height) // 2
            if dist < radius + hw:
                hits.append(ref)
        return hits


# ─────────────────────────────────────────────────────────────────────────────
#  GAME STATE
# ─────────────────────────────────────────────────────────────────────────────

class GameState:
    def __init__(self, cfg):
        self.cfg          = cfg
        self.score        = 0
        self.combo        = 0
        self.max_combo    = 0
        self.lives        = cfg["lives"]
        self.last_hit_t   = 0.0
        self.total_hits   = 0
        self.total_shots  = 0
        self.foul_throws  = 0
        self.game_over    = False
        self.popups       = []   # floating text effects

        # Verdict display (VALID THROW / FOUL THROW banner)
        self.verdict_text   = ""
        self.verdict_color  = C_WHITE
        self.verdict_frames = 0

    def register_hit(self, ref, hx, hy):
        now = time.time()
        if now - self.last_hit_t < self.cfg["combo_timeout_sec"]:
            self.combo += 1
        else:
            self.combo  = 1
        self.last_hit_t = now
        self.max_combo  = max(self.max_combo, self.combo)

        pts = 100 * self.combo
        self.score      += pts
        self.total_hits += 1

        text = (f"+{pts}  x{self.combo} COMBO!"
                if self.combo > 1 else f"+{pts}  HIT!")
        self.popups.append({
            "text":  text,
            "x": int(hx), "y": int(hy),
            "t": now,
            "color": C_ORANGE if self.combo >= 3 else C_YELLOW,
        })
        self.show_verdict("VALID THROW  ✓  HIT!", C_GREEN)

    def register_foul(self):
        self.foul_throws += 1
        self.combo        = 0   # foul throw resets combo
        self.show_verdict("FOUL THROW  ✗  INVALID!", C_RED)

    def register_shot(self):
        self.total_shots += 1

    def show_verdict(self, text, color):
        self.verdict_text   = text
        self.verdict_color  = color
        self.verdict_frames = CFG["verdict_hold_frames"]

    def lose_life(self):
        self.lives -= 1
        if self.lives <= 0:
            self.game_over = True

    def tick(self):
        now = time.time()
        self.popups = [p for p in self.popups if now - p["t"] < 1.5]
        if self.verdict_frames > 0:
            self.verdict_frames -= 1

    def reset(self):
        self.__init__(self.cfg)


# ─────────────────────────────────────────────────────────────────────────────
#  RENDERER
# ─────────────────────────────────────────────────────────────────────────────

class Renderer:

    def draw_referee(self, frame, ref):
        x, y   = int(ref.x), int(ref.y)
        hw, hh = ref.width // 2, ref.height // 2
        rc     = ref.ref_data
        body   = rc["color"]
        accent = rc["accent"]

        # Flash white on hit
        if ref.hit_flash > 0:
            body   = C_WHITE
            accent = C_BLACK

        # Body (referee shirt — black/white stripes suggestion)
        cv2.rectangle(frame, (x - hw, y - hh // 2),
                      (x + hw, y + hh), body, -1)
        cv2.rectangle(frame, (x - hw, y - hh // 2),
                      (x + hw, y + hh), accent, 1)

        # Stripes on shirt
        for sx in range(x - hw + 8, x + hw, 14):
            cv2.line(frame, (sx, y - hh // 2), (sx, y + hh), accent, 1)

        # Head
        cv2.circle(frame, (x, y - hh // 2 - 18), 18, (180, 140, 110), -1)
        cv2.circle(frame, (x, y - hh // 2 - 18), 18, accent, 1)

        # Whistle
        cv2.circle(frame, (x + 12, y - hh // 2 + 5), 5, C_YELLOW, -1)

        # Red card (held up smugly)
        cv2.rectangle(frame, (x + hw - 5, y - hh // 2 - 10),
                      (x + hw + 8, y - hh // 2 + 10), C_RED, -1)

        # Name label
        name  = rc["name"]
        scale = 0.38
        (tw, _), __ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(frame, name, (x - tw // 2, y + hh + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, accent, 1)

    def draw_ball_trail(self, frame, history):
        pts   = [(int(p.x), int(p.y)) for p in history]
        color = (0, 220, 255)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            t     = max(1, int(3 * alpha))
            c     = tuple(int(ch * alpha) for ch in color)
            cv2.line(frame, pts[i-1], pts[i], c, t)
        if pts:
            cv2.circle(frame, pts[-1], 13, color, 2)
            cv2.circle(frame, pts[-1],  5, C_WHITE, -1)

    def draw_predicted_path(self, frame, predicted, valid_form):
        color = C_GREEN if valid_form else C_RED
        for i in range(1, len(predicted)):
            if i % 2 == 0:
                x1, y1 = int(predicted[i-1][0]), int(predicted[i-1][1])
                x2, y2 = int(predicted[i][0]),   int(predicted[i][1])
                cv2.line(frame, (x1, y1), (x2, y2), color, 1)
            if i % 5 == 0:
                cv2.circle(frame,
                           (int(predicted[i][0]), int(predicted[i][1])),
                           2, color, -1)

    def draw_pose_indicators(self, frame, form_checker):
        """Small pose check panel — top right corner."""
        h, w   = frame.shape[:2]
        px, py = w - 220, 65
        panel_w, panel_h = 210, 75

        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 5, py - 5),
                      (px + panel_w, py + panel_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (px - 5, py - 5),
                      (px + panel_w, py + panel_h), C_CYAN, 1)

        cv2.putText(frame, "THROW FORM", (px, py + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1)

        checks = [
            ("Hands > Shoulders", form_checker.hands_above_shoulders),
            ("Hands > Head",      form_checker.hands_above_head),
        ]
        for i, (label, ok) in enumerate(checks):
            col  = C_GREEN if ok else C_RED
            icon = "v" if ok else "x"
            cv2.putText(frame, f"{icon} {label}",
                        (px, py + 36 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        # Draw wrist markers on frame
        t = CFG["pose_conf"]
        for wrist, label in [(form_checker.l_wrist, "LW"),
                              (form_checker.r_wrist, "RW")]:
            if wrist is None:
                continue
            ok  = form_checker.hands_above_head
            col = C_GREEN if ok else C_ORANGE
            wx, wy = int(wrist[0]), int(wrist[1])
            cv2.circle(frame, (wx, wy), 9, col, -1)
            cv2.putText(frame, label, (wx + 10, wy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        # Nose level line
        if form_checker.nose_y is not None:
            ny = int(form_checker.nose_y + CFG["hands_above_head_px"])
            cv2.line(frame, (0, ny), (w, ny), (80, 80, 80), 1)
            cv2.putText(frame, "head level", (5, ny - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GRAY, 1)

    def draw_verdict_banner(self, frame, game):
        if game.verdict_frames <= 0 or not game.verdict_text:
            return
        h, w   = frame.shape[:2]
        alpha  = min(1.0, game.verdict_frames / 20)   # fade out
        text   = game.verdict_text
        color  = game.verdict_color

        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0][0]
        tx = w // 2 - tw // 2
        ty = h // 2 + 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (tx - 20, ty - 45),
                      (tx + tw + 20, ty + 15), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7 * alpha, frame, 1 - 0.7 * alpha, 0, frame)
        # Shadow
        cv2.putText(frame, text, (tx + 2, ty + 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, C_BLACK, 4)
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2)

    def draw_hud(self, frame, game, ball_tracker):
        h, w = frame.shape[:2]

        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), (8, 8, 8), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Score
        cv2.putText(frame, f"SCORE  {game.score:06d}", (12, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, C_CYAN, 2)

        # Combo
        if game.combo > 1:
            col = C_ORANGE if game.combo >= 3 else C_YELLOW
            cv2.putText(frame, f"x{game.combo} COMBO",
                        (w // 2 - 65, 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, col, 2)

        # Lives
        for i in range(CFG["lives"]):
            col = C_RED if i < game.lives else C_GRAY
            cx  = w - 25 - i * 32
            cv2.circle(frame, (cx, 26), 11, col, -1)
            cv2.putText(frame, "H", (cx - 5, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_WHITE, 1)

        # Stats row (bottom)
        stats = (f"Shots:{game.total_shots}  "
                 f"Hits:{game.total_hits}  "
                 f"Fouls:{game.foul_throws}  "
                 f"MaxCombo:x{game.max_combo}")
        cv2.putText(frame, stats, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)

        # Ball state
        sc = {BallTracker.STATE_ABSENT:  C_GRAY,
              BallTracker.STATE_IN_HAND: C_GREEN,
              BallTracker.STATE_FLYING:  C_YELLOW}.get(ball_tracker.state, C_GRAY)
        cv2.putText(frame, f"Ball: {ball_tracker.state}",
                    (10, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, sc, 1)

    def draw_popups(self, frame, game):
        now = time.time()
        for p in game.popups:
            age = now - p["t"]
            if age > 1.5:
                continue
            y = int(p["y"] - age * 55)
            cv2.putText(frame, p["text"], (p["x"] - 50, y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, p["color"], 2)

    def draw_game_over(self, frame, game):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//4, h//4), (3*w//4, 3*h//4),
                      (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        cv2.putText(frame, "FINAL WHISTLE", (w//2 - 145, h//2 - 55),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, C_RED, 3)
        cv2.putText(frame, f"Score:     {game.score}", (w//2 - 100, h//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_WHITE, 2)
        cv2.putText(frame, f"Hits:      {game.total_hits}", (w//2 - 100, h//2 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_GREEN, 1)
        cv2.putText(frame, f"Foul Throws: {game.foul_throws}", (w//2 - 100, h//2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_RED, 1)
        cv2.putText(frame, f"Max Combo: x{game.max_combo}", (w//2 - 100, h//2 + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_YELLOW, 1)
        cv2.putText(frame, "Press R to restart", (w//2 - 105, h//2 + 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_GRAY, 1)

    def draw_intro(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(frame, "RAISE BOTH HANDS ABOVE YOUR HEAD",
                    (w//2 - 230, h//2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, C_CYAN, 2)
        cv2.putText(frame, "then throw the ball sideways to hit the referees!",
                    (w//2 - 250, h//2 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run(source):
    if not YOLO_AVAILABLE:
        return

    print(f"[INFO] Loading ball model  : {CFG['ball_model']}")
    ball_model = YOLO(CFG["ball_model"])
    print(f"[INFO] Loading pose model  : {CFG['pose_model']}")
    pose_model = YOLO(CFG["pose_model"])

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] {fw}x{fh} @ {fps:.1f}fps")

    # Init subsystems
    tracker   = BallTracker(CFG)
    fitter    = TrajectoryFitter(CFG)
    form      = ThrowFormChecker(CFG)
    refs      = RefereeManager(CFG, fw, fh)
    hit_det   = HitDetector(CFG)
    game      = GameState(CFG)
    renderer  = Renderer()

    paused          = False
    frame_idx       = 0
    screenshot_n    = 0
    prev_ball_state = BallTracker.STATE_ABSENT
    current_shot_id = 0
    shots_processed = set()

    # Form state per shot
    shot_form_verdict = FORM_UNKNOWN

    win = "AR Referee Shooter  [R=restart  P=pause  S=screenshot  Q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

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

        # ── GAME OVER screen ─────────────────────────────────────────────────
        if game.game_over:
            renderer.draw_game_over(display, game)
            cv2.imshow(win, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                _restart(game, tracker, form, refs, fw, fh,
                         shots_processed)
                current_shot_id   = 0
                shot_form_verdict = FORM_UNKNOWN
            continue

        # ── Pose detection ────────────────────────────────────────────────────
        pose_results = pose_model(frame, verbose=False)
        form_verdict = form.update(pose_results)

        # ── Ball detection ────────────────────────────────────────────────────
        tracker.update(display, ball_model)
        cur_state = tracker.state

        # ── Detect new throw ──────────────────────────────────────────────────
        if (cur_state == BallTracker.STATE_FLYING and
                prev_ball_state != BallTracker.STATE_FLYING):
            current_shot_id  += 1
            shot_form_verdict = form_verdict   # lock form verdict at throw moment
            game.register_shot()
            form.reset()   # reset form checker for next throw
            print(f"[GAME] Shot #{game.total_shots}  form={shot_form_verdict}")

        prev_ball_state = cur_state

        # ── Trajectory ───────────────────────────────────────────────────────
        predicted = []
        if (cur_state == BallTracker.STATE_FLYING and
                len(tracker.history) >= CFG["fit_min_points"]):
            predicted = fitter.fit_and_predict(tracker.history)

        # ── Hit detection (only if form was valid) ────────────────────────────
        alive_refs = refs.get_alive()

        if current_shot_id not in shots_processed:
            if shot_form_verdict == FORM_FOUL:
                # Foul throw: referees dodge, no hits possible
                if (cur_state == BallTracker.STATE_FLYING and
                        prev_ball_state == BallTracker.STATE_FLYING):
                    pass  # already flying, keep showing foul
                # Register foul once per shot
                if (cur_state == BallTracker.STATE_FLYING and
                        game.verdict_frames == 0):
                    game.register_foul()
                    refs.dodge_all()
                    shots_processed.add(current_shot_id)

            else:
                # Valid or unknown form — check hits
                traj_hits = hit_det.check_trajectory_hits(predicted, alive_refs)
                rt_hits   = hit_det.check_realtime_hit(
                    list(tracker.history), alive_refs)
                all_hits  = list({id(r): r for r in traj_hits + rt_hits}.values())

                for ref in all_hits:
                    refs.referees[refs.referees.index(ref)].alive = False
                    ref.hit_flash = 10
                    game.register_hit(ref, ref.x, ref.y)
                    print(f"[GAME] HIT {ref.ref_data['name']}! "
                          f"Score={game.score} Combo=x{game.combo}")
                if all_hits:
                    shots_processed.add(current_shot_id)

        # ── Update referees ───────────────────────────────────────────────────
        escaped = refs.update(fw, fh)
        for _ in escaped:
            game.lose_life()
            print(f"[GAME] Referee escaped! Lives={game.lives}")

        # ── Draw ─────────────────────────────────────────────────────────────

        # 1. Skeleton overlay from pose model
        annotated = pose_results[0].plot(img=display)

        # 2. Referees
        for ref in refs.get_alive():
            renderer.draw_referee(annotated, ref)

        # 3. Ball trail
        if tracker.history:
            renderer.draw_ball_trail(annotated, list(tracker.history))

        # 4. Predicted path (green=valid, red=foul)
        valid_form = (shot_form_verdict != FORM_FOUL)
        if predicted:
            renderer.draw_predicted_path(annotated, predicted, valid_form)

        # 5. Pose form panel
        renderer.draw_pose_indicators(annotated, form)

        # 6. HUD + popups + verdict
        game.tick()
        renderer.draw_hud(annotated, game, tracker)
        renderer.draw_popups(annotated, game)
        renderer.draw_verdict_banner(annotated, game)

        # 7. Intro hint for first few seconds
        if frame_idx < 120 and cur_state == BallTracker.STATE_ABSENT:
            renderer.draw_intro(annotated)

        # ── Show ─────────────────────────────────────────────────────────────
        cv2.imshow(win, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('r'):
            _restart(game, tracker, form, refs, fw, fh, shots_processed)
            current_shot_id   = 0
            shot_form_verdict = FORM_UNKNOWN
        elif key == ord('s'):
            name = f"screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(name, annotated)
            screenshot_n += 1
            print(f"[INFO] Saved {name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Score={game.score} MaxCombo=x{game.max_combo}")


def _restart(game, tracker, form, refs, fw, fh, shots_processed):
    game.reset()
    tracker.clear()
    form.reset()
    refs.reset(fw, fh)
    shots_processed.clear()
    print("[INFO] Game restarted.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AR Referee Shooter")
    parser.add_argument("--source", default="0",
                        help="Camera index or video file (default: 0)")
    args = parser.parse_args()
    run(args.source)
