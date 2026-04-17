import cv2
import numpy as np
import time
import random
import math
import argparse
import os
from collections import deque

try:
    from ultralytics import YOLO
except ImportError:
    print("install ultralytics first: pip install ultralytics")
    exit()

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
POSE_CONF        = 0.4
HIT_RADIUS       = 60
ENEMY_SPEED      = 1.4          # px / frame
ENEMY_SPAWN_SEC  = 3.0
MAX_ENEMIES      = 5
LIVES            = 3
COMBO_TIMEOUT    = 3.5
VERDICT_FRAMES   = 80

L_ELBOW  = 7
R_ELBOW  = 8
PUNCH_VEL_THRESH = 40
WRIST_HISTORY    = 5

# YOLOv8 keypoint indices
NOSE, L_SHLDR, R_SHLDR, L_WRIST, R_WRIST = 0, 5, 6, 9, 10

# OPM enemy roster  (name, power_label)
ENEMY_ROSTER = [
    ("Deep Sea King",  "DRAGON"),
    ("Orochi",         "DRAGON"),
    ("Psykos",         "DRAGON"),
    ("Tatsumaki",      "DRAGON"),
    ("Black Sperm",    "DRAGON"),
]

# Assets folder – same directory as this script
ASSET_DIR = os.path.dirname(os.path.abspath(__file__))

# OPM colour palette
C_YELLOW  = (0,   215, 255)
C_WHITE   = (255, 255, 255)
C_RED     = (30,   30, 220)
C_ORANGE  = (20,  140, 255)
C_BLACK   = (0,     0,   0)
C_GREEN   = (40,  200,  40)
C_DARK    = (12,   12,  12)
C_GOLD    = (30,  185, 255)

# ─────────────────────────────────────────────
#  IMAGE HELPERS
# ─────────────────────────────────────────────

def load_png(filename):
    """Load a PNG with alpha channel.  Returns None on failure."""
    path = os.path.join(ASSET_DIR, filename)
    img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[warn] could not load: {path}")
        return None
    if img.shape[2] == 3:                          # add alpha if missing
        alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255
        img   = np.dstack([img, alpha])
    return img


def overlay_png(bg, overlay, cx, cy, scale=1.0, alpha_mult=1.0):
    """
    Paste `overlay` (RGBA) onto `bg` (BGR) centred at (cx, cy).
    scale      : resize factor applied to overlay before pasting
    alpha_mult : global opacity multiplier 0-1
    """
    if overlay is None:
        return bg

    oh, ow = overlay.shape[:2]
    nw = max(1, int(ow * scale))
    nh = max(1, int(oh * scale))
    resized = cv2.resize(overlay, (nw, nh), interpolation=cv2.INTER_AREA)

    x1 = cx - nw // 2;  y1 = cy - nh // 2
    x2 = x1 + nw;       y2 = y1 + nh

    bh, bw = bg.shape[:2]
    ox1 = max(0, -x1);  oy1 = max(0, -y1)
    ox2 = nw - max(0, x2 - bw)
    oy2 = nh - max(0, y2 - bh)
    x1  = max(0, x1);   y1 = max(0, y1)
    x2  = min(bw, x2);  y2 = min(bh, y2)

    if x2 <= x1 or y2 <= y1:
        return bg

    roi = resized[oy1:oy2, ox1:ox2]
    if roi.size == 0:
        return bg

    alpha = roi[:, :, 3:4].astype(np.float32) / 255.0 * alpha_mult
    rgb   = roi[:, :, :3].astype(np.float32)
    bg_roi = bg[y1:y2, x1:x2].astype(np.float32)
    blended = rgb * alpha + bg_roi * (1.0 - alpha)
    bg[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return bg


def flash_white(img):
    """Return a fully-white copy of img (preserving alpha)."""
    out = img.copy()
    out[:, :, :3] = 255
    return out

# ─────────────────────────────────────────────
#  LOAD ASSETS  (called once at startup)
# ─────────────────────────────────────────────

def load_assets():
    enemies = []
    files   = [
        "DeepSeaKing.png",
        "Orochi.png",
        "Psykos.png",
        "Tatsumaki.png",
        "BlackSperm.png",
    ]
    for i, f in enumerate(files):
        img = load_png(f)
        name, power = ENEMY_ROSTER[i % len(ENEMY_ROSTER)]
        enemies.append({"img": img, "name": name, "power": power})

    suit = load_png("saitama_suit.png")
    head = load_png("saitama_head.png")
    return enemies, suit, head

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def get_kp(kpts, idx):
    if kpts is None or idx >= len(kpts):
        return None
    x, y, c = float(kpts[idx][0]), float(kpts[idx][1]), float(kpts[idx][2])
    return (x, y) if c >= POSE_CONF else None


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def draw_text_shadowed(frame, text, pos, font=cv2.FONT_HERSHEY_DUPLEX,
                       scale=1.0, color=C_WHITE, thickness=2, shadow_offset=2):
    x, y = pos
    cv2.putText(frame, text, (x + shadow_offset, y + shadow_offset),
                font, scale, C_BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_outlined_rect(frame, pt1, pt2, fill_color, border_color, alpha=0.75):
    ov = frame.copy()
    cv2.rectangle(ov, pt1, pt2, fill_color, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, pt1, pt2, border_color, 1)

# ─────────────────────────────────────────────
#  FORM CHECKER
# ─────────────────────────────────────────────

class FormChecker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.verdict      = "UNKNOWN"
        self._wrist_hist  = deque(maxlen=WRIST_HISTORY)
        self._cooldown    = 0
        self.l_wrist      = None
        self.r_wrist      = None
        self.l_elbow      = None
        self.r_elbow      = None
        self.arm_extended = False
        self.fast_enough  = False

    def update(self, pose_results):
        if self._cooldown > 0:
            self._cooldown -= 1
            return self.verdict

        r = pose_results[0]
        if r.keypoints is None or len(r.keypoints.data) == 0:
            return self.verdict

        kpts = r.keypoints.data[0]
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

        self._wrist_hist.append(l_wrist[0])
        self.arm_extended = l_wrist[0] < l_elbow[0] < l_shldr[0]

        self.fast_enough = False
        if len(self._wrist_hist) >= 2:
            vel = abs(self._wrist_hist[-1] - self._wrist_hist[-2])
            self.fast_enough = vel > PUNCH_VEL_THRESH

        if self.arm_extended and self.fast_enough:
            self.verdict   = "VALID"
            self._cooldown = 15
        else:
            self.verdict = "FOUL" if not self.arm_extended else "UNKNOWN"

        return self.verdict

# ─────────────────────────────────────────────
#  ENEMY  (replaces Ref)
# ─────────────────────────────────────────────

class Enemy:
    TARGET_H = 110          # display height in pixels

    def __init__(self, x, fw, enemy_data):
        self.x        = float(x)
        self.y        = -60.0
        self.fw       = fw
        self.data     = enemy_data          # dict with img / name / power
        self.alive    = True
        self.flash    = 0
        self.dodge_dx = 0.0
        self.bob_t    = random.uniform(0, math.tau)   # bobbing phase

        img = enemy_data["img"]
        if img is not None:
            ratio     = self.TARGET_H / img.shape[0]
            self.w    = int(img.shape[1] * ratio)
            self.h    = self.TARGET_H
            self.scale = ratio
        else:
            self.w = 70;  self.h = 90;  self.scale = 1.0

    def update(self):
        self.y     += ENEMY_SPEED
        self.bob_t += 0.08
        if self.dodge_dx:
            self.x  = max(40, min(self.fw - 40, self.x + self.dodge_dx))
            self.dodge_dx *= 0.85
            if abs(self.dodge_dx) < 0.3:
                self.dodge_dx = 0.0
        if self.flash > 0:
            self.flash -= 1

    def draw(self, frame):
        x  = int(self.x)
        y  = int(self.y + math.sin(self.bob_t) * 4)    # gentle bob

        img = self.data["img"]
        if img is not None:
            draw_img = flash_white(img) if self.flash > 0 else img
            overlay_png(frame, draw_img, x, y, self.scale)
        else:
            # fallback rectangle
            hw, hh = self.w // 2, self.h // 2
            col = C_WHITE if self.flash > 0 else (40, 40, 180)
            cv2.rectangle(frame, (x - hw, y - hh), (x + hw, y + hh), col, -1)
            cv2.rectangle(frame, (x - hw, y - hh), (x + hw, y + hh), C_BLACK, 1)

        # name tag below enemy
        name  = self.data["name"]
        power = self.data["power"]
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tx = x - tw // 2
        ty = y + self.h // 2 + 16
        # power tier badge
        badge_col = C_RED if power == "DRAGON" else C_ORANGE
        cv2.rectangle(frame, (tx - 4, ty - th - 2), (tx + tw + 4, ty + 4), badge_col, -1)
        cv2.putText(frame, name, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1, cv2.LINE_AA)

    def hits_fist(self, fx, fy):
        return (abs(fx - self.x) < self.w // 2 + HIT_RADIUS and
                abs(fy - self.y) < self.h // 2 + HIT_RADIUS)

# ─────────────────────────────────────────────
#  SUIT OVERLAY  (Saitama costume on player)
# ─────────────────────────────────────────────

def draw_saitama_suit(frame, pose_results, suit_img,
                      width_factor=2.4, vertical_offset=0.38):
    """
    Overlay saitama_suit.png aligned to the detected person's torso.
    width_factor    : suit width = shoulder_width * width_factor  (tune 2.0-3.0)
    vertical_offset : suit_center_y offset as fraction of scaled suit height (0.2-0.5)
    """
    if suit_img is None or len(pose_results) == 0:
        return frame

    r = pose_results[0]
    if r.keypoints is None or len(r.keypoints.data) == 0:
        return frame

    kpts    = r.keypoints.data[0]
    l_sh    = get_kp(kpts, L_SHLDR)
    r_sh    = get_kp(kpts, R_SHLDR)

    if l_sh is None or r_sh is None:
        return frame

    shoulder_w = dist(l_sh, r_sh)
    if shoulder_w < 10:
        return frame

    scale  = (shoulder_w * width_factor) / suit_img.shape[1]
    cx     = int((l_sh[0] + r_sh[0]) / 2)
    sh_y   = int((l_sh[1] + r_sh[1]) / 2)
    offset = int(suit_img.shape[0] * scale * vertical_offset)
    cy     = sh_y + offset

    overlay_png(frame, suit_img, cx, cy, scale, alpha_mult=0.92)
    return frame

# ─────────────────────────────────────────────
#  HEAD OVERLAY  (Saitama bald head on player)
# ─────────────────────────────────────────────

def draw_saitama_head(frame, pose_results, head_img,
                      size_factor=2.2, vertical_offset=-0.45):
    """
    Overlay saitama_head.png centred on the detected person's head.
    size_factor     : head_width = shoulder_width * size_factor  (tune 1.8-2.8)
    vertical_offset : head centre Y offset as fraction of scaled head height,
                      negative = upward  (tune -0.6 ~ -0.2)
    """
    if head_img is None or len(pose_results) == 0:
        return frame

    r = pose_results[0]
    if r.keypoints is None or len(r.keypoints.data) == 0:
        return frame

    kpts = r.keypoints.data[0]
    nose = get_kp(kpts, NOSE)
    l_sh = get_kp(kpts, L_SHLDR)
    r_sh = get_kp(kpts, R_SHLDR)

    if nose is None or l_sh is None or r_sh is None:
        return frame

    shoulder_w = dist(l_sh, r_sh)
    if shoulder_w < 10:
        return frame

    scale          = (shoulder_w * size_factor) / head_img.shape[1]
    head_h_scaled  = int(head_img.shape[0] * scale)
    cx             = int(nose[0])
    cy             = int(nose[1]) + int(head_h_scaled * vertical_offset)

    overlay_png(frame, head_img, cx, cy, scale, alpha_mult=0.95)
    return frame


# ─────────────────────────────────────────────
#  GAME
# ─────────────────────────────────────────────

class Game:
    def __init__(self, fw, fh, enemy_pool):
        self.fw   = fw
        self.fh   = fh
        self.pool = enemy_pool      # list of enemy_data dicts
        self.reset()

    def reset(self):
        self.score      = 0
        self.combo      = 0
        self.max_combo  = 0
        self.lives      = LIVES
        self.hits       = 0
        self.shots      = 0
        self.fouls      = 0
        self.over       = False
        self.last_hit_t = 0.0
        self.popups     = []
        self.verdict_txt   = ""
        self.verdict_col   = C_WHITE
        self.verdict_left  = 0
        self.enemies    = []
        self.last_spawn = time.time()
        self.start_time = time.time()

    # ── spawning ─────────────────────────────

    def spawn_enemy(self):
        alive = [e for e in self.enemies if e.alive]
        if len(alive) >= MAX_ENEMIES:
            return
        x    = random.randint(80, self.fw - 80)
        data = random.choice(self.pool)
        self.enemies.append(Enemy(x, self.fw, data))
        self.last_spawn = time.time()

    def update_enemies(self):
        escaped = []
        for e in self.enemies:
            if not e.alive:
                continue
            e.update()
            if e.y > self.fh + e.h + 20:
                escaped.append(e)

        self.enemies = [e for e in self.enemies
                        if e.alive and e.y <= self.fh + e.h + 20]

        for _ in escaped:
            self.lives -= 1
            if self.lives <= 0:
                self.over = True

        if time.time() - self.last_spawn > ENEMY_SPAWN_SEC:
            self.spawn_enemy()

    # ── hit / foul ───────────────────────────

    def register_hit(self, enemy):
        now         = time.time()
        self.combo  = (self.combo + 1
                       if now - self.last_hit_t < COMBO_TIMEOUT else 1)
        self.last_hit_t = now
        self.max_combo  = max(self.max_combo, self.combo)
        pts          = 100 * self.combo
        self.score  += pts
        self.hits   += 1
        label = (f"+{pts}  x{self.combo} COMBO!" if self.combo > 1
                 else f"+{pts}")
        self.popups.append({
            "text": label,
            "x": int(enemy.x),
            "y": int(enemy.y),
            "t": now
        })
        self._set_verdict("ONE PUNCH!", C_GOLD)

    def register_foul(self):
        self.fouls += 1
        self.combo  = 0
        for e in self.enemies:
            if e.alive:
                e.dodge_dx = random.choice([-4.0, 4.0])
        self._set_verdict("SLOPPY FORM!", C_RED)

    def _set_verdict(self, txt, col):
        self.verdict_txt  = txt
        self.verdict_col  = col
        self.verdict_left = VERDICT_FRAMES

    # ── draw ─────────────────────────────────

    def draw(self, frame, form):
        h, w = frame.shape[:2]

        # enemies
        for e in self.enemies:
            if e.alive:
                e.draw(frame)

        self._draw_top_hud(frame, w)
        self._draw_lives(frame, w)
        self._draw_pose_panel(frame, form, w)
        self._draw_popups(frame)
        self._draw_verdict_banner(frame, w, h)
        self._draw_bottom_stats(frame, h)

    # ── HUD sub-draws ─────────────────────────

    def _draw_top_hud(self, frame, w):
        ov = frame.copy()
        # gradient-ish bar
        cv2.rectangle(ov, (0, 0), (w, 58), (8, 8, 20), -1)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
        # decorative left stripe
        cv2.rectangle(frame, (0, 0), (5, 58), C_YELLOW, -1)

        draw_text_shadowed(frame, f"SCORE  {self.score:07d}",
                           (18, 38), scale=1.0, color=C_YELLOW, thickness=2)

        if self.combo > 1:
            col = C_RED if self.combo >= 5 else C_ORANGE
            label = f"x{self.combo} COMBO!"
            draw_text_shadowed(frame, label, (w // 2 - 75, 38),
                               scale=1.0, color=col, thickness=2)

        elapsed = int(time.time() - self.start_time)
        draw_text_shadowed(frame, f"{elapsed:04d}s",
                           (w - 90, 38), scale=0.75, color=C_WHITE)

    def _draw_lives(self, frame, w):
        # Draw OPM-style red "hearts" (circles)
        for i in range(LIVES):
            cx = w - 22 - i * 30
            cy = 75
            col    = C_RED   if i < self.lives else (60, 60, 60)
            border = C_GOLD  if i < self.lives else (40, 40, 40)
            cv2.circle(frame, (cx, cy), 11, col,    -1)
            cv2.circle(frame, (cx, cy), 11, border,  1)
            cv2.putText(frame, "!", (cx - 3, cy + 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, C_WHITE, 1)

    def _draw_pose_panel(self, frame, form, w):
        h = frame.shape[0]
        px, py = w - 215, 100
        draw_outlined_rect(frame, (px - 8, py - 8),
                           (px + 205, py + 80), (8, 8, 20), C_YELLOW, 0.65)
        draw_text_shadowed(frame, "PUNCH ANALYSIS",
                           (px, py + 15), scale=0.48,
                           color=C_YELLOW, thickness=1)

        checks = [
            ("Arm extended", form.arm_extended),
            ("Fast enough",  form.fast_enough),
        ]
        for i, (label, ok) in enumerate(checks):
            col  = C_GREEN if ok else C_RED
            icon = "OK" if ok else "--"
            cv2.putText(frame, f"[{icon}] {label}",
                        (px, py + 38 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        # draw wrist dots
        for wrist, label in [(form.l_wrist, "L"), (form.r_wrist, "R")]:
            if wrist is None:
                continue
            col = C_GREEN if form.arm_extended else C_ORANGE
            cv2.circle(frame, (int(wrist[0]), int(wrist[1])), 10, col, -1)
            cv2.circle(frame, (int(wrist[0]), int(wrist[1])), 10, C_WHITE, 1)
            cv2.putText(frame, label,
                        (int(wrist[0]) + 12, int(wrist[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    def _draw_popups(self, frame):
        now = time.time()
        self.popups = [p for p in self.popups if now - p["t"] < 1.5]
        for p in self.popups:
            age = now - p["t"]
            y   = int(p["y"] - age * 60)
            alpha = max(0.0, 1.0 - age / 1.5)
            draw_text_shadowed(frame, p["text"], (p["x"] - 55, y),
                               scale=0.85, color=C_GOLD, thickness=2)

    def _draw_verdict_banner(self, frame, w, h):
        if self.verdict_left <= 0:
            return
        self.verdict_left -= 1
        alpha = min(1.0, self.verdict_left / 20.0)

        tw = cv2.getTextSize(self.verdict_txt,
                             cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)[0][0]
        tx = w // 2 - tw // 2
        ty = h // 2 + 25

        ov = frame.copy()
        cv2.rectangle(ov, (tx - 24, ty - 52),
                      (tx + tw + 24, ty + 18), (8, 8, 20), -1)
        cv2.addWeighted(ov, 0.75 * alpha, frame, 1 - 0.75 * alpha, 0, frame)

        # outline text
        cv2.putText(frame, self.verdict_txt, (tx + 3, ty + 3),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, C_BLACK, 6, cv2.LINE_AA)
        col = tuple(int(c * alpha + 255 * (1 - alpha)) for c in self.verdict_col)
        cv2.putText(frame, self.verdict_txt, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, col, 3, cv2.LINE_AA)

    def _draw_bottom_stats(self, frame, h):
        txt = (f"Shots:{self.shots}   Hits:{self.hits}   "
               f"Fouls:{self.fouls}   MaxCombo:x{self.max_combo}")
        cv2.putText(frame, txt, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # ── game over screen ──────────────────────

    def draw_gameover(self, frame):
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (w // 5, h // 5), (4 * w // 5, 4 * h // 5),
                      (8, 8, 20), -1)
        cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (w // 5, h // 5), (4 * w // 5, 4 * h // 5),
                      C_YELLOW, 2)

        draw_text_shadowed(frame, "ONE PUNCH MAN",
                           (w // 2 - 165, h // 2 - 80),
                           scale=1.3, color=C_YELLOW, thickness=3)
        draw_text_shadowed(frame, "GAME OVER",
                           (w // 2 - 105, h // 2 - 38),
                           scale=1.1, color=C_RED, thickness=2)

        rows = [
            (f"Score:       {self.score:07d}", C_WHITE),
            (f"Enemies hit: {self.hits}",       C_GREEN),
            (f"Fouls:       {self.fouls}",       C_RED),
            (f"Max Combo:   x{self.max_combo}",  C_ORANGE),
            ("Press  R  to restart",             (140, 140, 140)),
        ]
        for i, (txt, col) in enumerate(rows):
            draw_text_shadowed(frame, txt,
                               (w // 2 - 120, h // 2 + 5 + i * 32),
                               scale=0.65, color=col, thickness=1)

# ─────────────────────────────────────────────
#  INTRO BANNER
# ─────────────────────────────────────────────

def draw_intro(frame, frame_idx):
    if frame_idx >= 150:
        return
    h, w = frame.shape[:2]
    alpha = max(0.0, 1.0 - frame_idx / 100.0)
    msg   = "PUNCH THE MONSTERS!"
    tw    = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)[0][0]
    tx    = w // 2 - tw // 2
    ty    = h // 2

    ov = frame.copy()
    cv2.rectangle(ov, (tx - 20, ty - 50), (tx + tw + 20, ty + 18),
                  (8, 8, 20), -1)
    cv2.addWeighted(ov, 0.7 * alpha, frame, 1 - 0.7 * alpha, 0, frame)
    col = tuple(int(c * alpha) for c in C_YELLOW)
    cv2.putText(frame, msg, (tx + 2, ty + 2),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, C_BLACK, 5, cv2.LINE_AA)
    cv2.putText(frame, msg, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, col, 3, cv2.LINE_AA)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run(source,
        suit_width_factor=2.4,
        suit_vertical_offset=0.38,
        head_size_factor=2.2,
        head_vertical_offset=-0.45,
        show_skeleton=True):

    print("Loading assets...")
    enemy_pool, suit_img, head_img = load_assets()
    if not enemy_pool:
        print("[error] No enemy images loaded.  Put PNGs next to this script.")
        return

    print("Loading YOLO pose model...")
    pose_model = YOLO("yolov8n-pose.pt")

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open: {source}")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {fw}x{fh}")

    form         = FormChecker()
    game         = Game(fw, fh, enemy_pool)
    prev_verdict = "UNKNOWN"
    frame_idx    = 0

    win = "ONE PUNCH MAN  [R=restart  P=pause  Q=quit]"
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

        # ── game over screen ──────────────────
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

        # ── pose inference ────────────────────
        pose_res = pose_model(frame, verbose=False)

        # ── saitama suit overlay ──────────────
        if len(pose_res) > 0:
            display = draw_saitama_suit(
                display, pose_res, suit_img,
                width_factor=suit_width_factor,
                vertical_offset=suit_vertical_offset
            )
            # head drawn AFTER suit so it sits on top of the collar
            display = draw_saitama_head(
                display, pose_res, head_img,
                size_factor=head_size_factor,
                vertical_offset=head_vertical_offset
            )

        # ── form check ────────────────────────
        form_now = form.update(pose_res) if len(pose_res) > 0 else form.verdict

        # ── punch edge detection ──────────────
        if form_now == "VALID" and prev_verdict != "VALID":
            game.shots += 1
            hit_enemy  = None
            if form.l_wrist is not None:
                wx, wy   = form.l_wrist
                closest  = 999999
                for e in [en for en in game.enemies if en.alive]:
                    d = math.hypot(wx - e.x, wy - e.y)
                    if d < closest:
                        closest  = d
                        hit_enemy = e
                if hit_enemy and closest < 220:
                    hit_enemy.alive = False
                    hit_enemy.flash = 12
                    game.register_hit(hit_enemy)
                else:
                    game.register_foul()

        prev_verdict = form_now

        # ── update enemies ────────────────────
        game.update_enemies()

        # ── annotate skeleton (optional) ──────
        if show_skeleton and len(pose_res) > 0:
            annotated = pose_res[0].plot(img=display)
        else:
            annotated = display

        # ── draw game UI ──────────────────────
        game.draw(annotated, form)
        draw_intro(annotated, frame_idx)

        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('r'):
            game.reset()
            form.reset()
            prev_verdict = "UNKNOWN"
        elif key == ord('s'):           # toggle skeleton
            show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done.  Score={game.score}  MaxCombo=x{game.max_combo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",   default="0",
                        help="camera index or video file path")
    parser.add_argument("--no-skeleton", action="store_true",
                        help="hide YOLO skeleton overlay")
    # Fine-tune the suit alignment without editing code
    parser.add_argument("--suit-width",   type=float, default=2.8,
                        help="suit width factor (default 2.8, tune 1.8-3.0)")
    parser.add_argument("--suit-voffset", type=float, default=0.38,
                        help="suit vertical offset fraction (default 0.38, tune 0.2-0.5)")
    # Fine-tune the head alignment
    parser.add_argument("--head-size",    type=float, default=1.8,
                        help="head size factor (default 1.8, tune 1.8-2.8)")
    parser.add_argument("--head-voffset", type=float, default=-0.45,
                        help="head vertical offset fraction (default -0.45, tune -0.6 ~ -0.2)")
    args = parser.parse_args()

    run(
        source=args.source,
        suit_width_factor=args.suit_width,
        suit_vertical_offset=args.suit_voffset,
        head_size_factor=args.head_size,
        head_vertical_offset=args.head_voffset,
        show_skeleton=not args.no_skeleton,
    )
