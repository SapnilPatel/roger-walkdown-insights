"""Read an analog gauge from a single camera frame.

Classical computer vision, no training data needed:
  1. Normalise contrast (CLAHE) so dim, hazy and washed-out frames look alike.
  2. Find the dial face (Hough circle; falls back to ellipse fitting when the
     gauge is seen off-axis).
  3. Cast rays from the centre across the dial's sweep. The needle is the
     darkest ray. A parabola through the best ray and its neighbours gives
     sub-degree precision.
  4. Convert the needle angle to a value using the gauge's known calibration
     (min/max and sweep), which comes from the asset register.
  5. Score confidence from how clearly the needle stands out. Low-confidence
     reads are routed to the human operator instead of being trusted blindly.
"""
from dataclasses import dataclass
import math
import numpy as np
import cv2

from .route import GaugeSpec


@dataclass
class Reading:
    value: float
    angle_deg: float
    confidence: float          # 0..1
    center: tuple
    radius: float
    method: str                # how the dial was located
    contrast: float            # raw needle-vs-face contrast, kept for debugging
    transform: object = None   # 2x3 affine applied before reading (off-axis correction)
    dip_width_deg: float = 0.0 # angular width of the needle's shadow; wide = smeared/blurred

    def to_dict(self):
        return {
            "value": round(self.value, 2),
            "angle_deg": round(self.angle_deg, 2),
            "confidence": round(self.confidence, 3),
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "radius": round(self.radius, 1),
            "method": self.method,
            "contrast": round(self.contrast, 4),
            "dip_width_deg": round(self.dip_width_deg, 1),
        }


def _normalise(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _find_dial(gray: np.ndarray):
    h, w = gray.shape
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w,
                               param1=120, param2=40,
                               minRadius=int(w * 0.28), maxRadius=int(w * 0.46))
    if circles is not None:
        x, y, r = circles[0][0]
        # Hough often locks onto the outer bezel; the white face is ~0.89 of that.
        # Check brightness just inside the found radius to decide.
        face_r = _refine_face_radius(gray, (x, y), r)
        return (float(x), float(y)), float(face_r), "hough"

    # off-axis view: the face is an ellipse; fit it from the bright region
    _, th = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    for c in contours:
        if len(c) < 50:
            continue
        area = cv2.contourArea(c)
        if area < 0.1 * w * h:
            continue
        if best is None or area > cv2.contourArea(best):
            best = c
    if best is not None:
        (x, y), (ma, mb), _ = cv2.fitEllipse(best)
        return (float(x), float(y)), float(min(ma, mb) / 2), "ellipse"

    return (w / 2.0, h / 2.0), w * 0.36, "fallback"


def _refine_face_radius(gray, center, r):
    """Shrink a bezel-sized circle down to the light face edge."""
    cx, cy = center
    best_r = r
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    for frac in np.linspace(1.0, 0.75, 26):
        rr = r * frac
        xs = np.clip((cx + rr * np.cos(angles)).astype(int), 0, gray.shape[1] - 1)
        ys = np.clip((cy - rr * np.sin(angles)).astype(int), 0, gray.shape[0] - 1)
        if np.median(gray[ys, xs]) > np.median(gray) + 5:
            best_r = rr
            break
    return best_r


def _hub_center(gray, center, radius):
    """The needle hub is the darkest blob near the middle; it pins the
    rotation centre more precisely than the circle fit."""
    cx, cy = center
    r = int(radius * 0.25)
    x0, y0 = int(max(cx - r, 0)), int(max(cy - r, 0))
    roi = gray[y0:int(cy + r), x0:int(cx + r)]
    if roi.size == 0:
        return center
    roi = cv2.GaussianBlur(roi, (0, 0), radius * 0.03 + 1)
    _, _, min_loc, _ = cv2.minMaxLoc(roi)
    hx, hy = x0 + min_loc[0], y0 + min_loc[1]
    if math.hypot(hx - cx, hy - cy) < radius * 0.15:
        return (float(hx), float(hy))
    return center


def _rectify(gray: np.ndarray):
    """If the dial face appears as an ellipse (gauge seen off-axis), stretch
    the image along the ellipse's short axis so the face becomes a circle again.
    Returns (image, affine matrix or None)."""
    h, w = gray.shape
    _, th = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cands = [c for c in contours if len(c) >= 50 and cv2.contourArea(c) > 0.1 * w * h]
    if not cands:
        return gray, None
    c = max(cands, key=cv2.contourArea)
    # the face must be roughly elliptical, otherwise this is not a dial outline
    (x, y), (d1, d2), ang = cv2.fitEllipse(c)
    ell_area = math.pi * d1 * d2 / 4
    if abs(cv2.contourArea(c) - ell_area) / ell_area > 0.08:
        return gray, None
    ratio = min(d1, d2) / max(d1, d2)
    if ratio > 0.985 or ratio < 0.6:
        return gray, None
    # axis of the short diameter, in image coordinates
    theta = math.radians(ang if d1 < d2 else ang + 90)
    u = np.array([math.cos(theta), math.sin(theta)])
    s = max(d1, d2) / min(d1, d2)
    # stretch by s along u, about the ellipse centre
    A = np.eye(2) + (s - 1) * np.outer(u, u)
    t = np.array([x, y]) - A @ np.array([x, y])
    M = np.hstack([A, t[:, None]]).astype(np.float32)
    return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE), M


def read_gauge(img_bgr: np.ndarray, spec: GaugeSpec) -> Reading:
    gray = _normalise(img_bgr)
    gray, M = _rectify(gray)
    center, radius, method = _find_dial(gray)
    if M is not None:
        method += "+rectified"
    center = _hub_center(gray, center, radius)
    cx, cy = center

    smooth = cv2.GaussianBlur(gray, (0, 0), 1.2).astype(np.float32)
    h, w = smooth.shape

    # rays across the dial sweep only (the dead zone at the bottom holds the
    # unit label, which would otherwise look like a needle)
    step = 0.5
    start = spec.start_angle_deg + 8
    angles = np.arange(start, start - spec.sweep_deg - 16, -step)
    radii = np.linspace(radius * 0.25, radius * 0.60, 40)
    a = np.radians(angles)[:, None]
    xs = cx + radii[None, :] * np.cos(a)
    ys = cy - radii[None, :] * np.sin(a)
    samples = cv2.remap(smooth, xs.astype(np.float32), ys.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    profile = samples.mean(axis=1)

    i = int(np.argmin(profile))
    # sub-sample refinement
    if 0 < i < len(profile) - 1:
        y0, y1, y2 = profile[i - 1], profile[i], profile[i + 1]
        denom = y0 - 2 * y1 + y2
        offset = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        offset = 0.0
    needle_angle = float(angles[i] - offset * step)

    # confidence: how far the needle ray sits below the typical face brightness,
    # and how unique that dip is (no rival dip elsewhere on the dial)
    face = float(np.median(profile))
    dip = face - float(profile[i])
    contrast = dip / (face + 1e-6)
    mask = np.abs(np.arange(len(profile)) - i) > int(12 / step)
    rival = face - float(profile[mask].min()) if mask.any() else 0.0
    uniqueness = 1.0 - (rival / dip) if dip > 0 else 0.0
    # needle width at half depth: a sharp needle is a few degrees wide,
    # motion blur or glare smears it sideways and makes the angle ambiguous
    half = face - dip / 2
    lo = i
    while lo > 0 and profile[lo - 1] < half:
        lo -= 1
    hi = i
    while hi < len(profile) - 1 and profile[hi + 1] < half:
        hi += 1
    dip_width = (hi - lo + 1) * step

    conf_contrast = np.clip((contrast - 0.05) / 0.35, 0, 1)
    conf_sharp = np.clip(1.0 - (dip_width - 8.0) / 10.0, 0.1, 1.0)
    confidence = float(np.clip(conf_contrast * (0.4 + 0.6 * max(uniqueness, 0.0)) * conf_sharp, 0, 1))
    if method == "fallback":
        confidence *= 0.5

    value = spec.angle_to_value(needle_angle)
    return Reading(value, needle_angle, confidence, center, radius, method, contrast, M, dip_width)


def annotate(img_bgr: np.ndarray, reading: Reading, spec: GaugeSpec, status_color=(0, 200, 255)):
    """Overlay what the pipeline 'saw': dial outline, detected needle, value.
    Geometry is found in the corrected (rectified) frame, so it is mapped back
    onto the original camera frame before drawing."""
    out = img_bgr.copy()
    cx, cy = reading.center
    r = reading.radius
    inv = cv2.invertAffineTransform(np.asarray(reading.transform, np.float32)) \
        if reading.transform is not None else None

    def to_img(pts):
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        if inv is not None:
            pts = cv2.transform(pts, inv)
        return pts.reshape(-1, 2)

    # HUD-style overlay, drawn at 2x and downsampled for clean anti-aliasing.
    # No burned-in text: the dashboard shows the numbers.
    k = 2
    big = cv2.resize(out, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    layer = big.copy()
    S = lambda pts: (to_img(pts) * k).astype(np.int32)

    # dial outline
    ring = [(cx + r * math.cos(t), cy - r * math.sin(t)) for t in np.linspace(0, 2 * np.pi, 160)]
    cv2.polylines(layer, [S(ring)], True, (255, 255, 255), 2 * k, cv2.LINE_AA)
    # sweep from the gauge minimum to the detected value, in the status colour
    a0, a1 = spec.start_angle_deg, reading.angle_deg
    arc = [(cx + r * 1.0 * math.cos(math.radians(t)), cy - r * 1.0 * math.sin(math.radians(t)))
           for t in np.linspace(a0, a1, max(8, int(abs(a0 - a1))))]
    cv2.polylines(layer, [S(arc)], False, status_color, 5 * k, cv2.LINE_AA)
    out_big = cv2.addWeighted(layer, 0.85, big, 0.15, 0)

    # detected needle + hub
    a = math.radians(reading.angle_deg)
    c0, tip = S([(cx, cy), (cx + r * 0.92 * math.cos(a), cy - r * 0.92 * math.sin(a))])
    cv2.line(out_big, tuple(c0), tuple(tip), (20, 20, 20), 7 * k, cv2.LINE_AA)
    cv2.line(out_big, tuple(c0), tuple(tip), status_color, 3 * k, cv2.LINE_AA)
    cv2.circle(out_big, tuple(tip), 5 * k, status_color, -1, cv2.LINE_AA)
    cv2.circle(out_big, tuple(c0), 6 * k, status_color, -1, cv2.LINE_AA)

    # corner brackets around the detected dial
    box = S([(cx - r * 1.12, cy - r * 1.12), (cx + r * 1.12, cy + r * 1.12)])
    (x0, y0), (x1, y1) = box[0], box[1]
    L = int(r * 0.22 * k)
    for (x, y, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        cv2.line(out_big, (x, y), (x + dx * L, y), (255, 255, 255), 2 * k, cv2.LINE_AA)
        cv2.line(out_big, (x, y), (x, y + dy * L), (255, 255, 255), 2 * k, cv2.LINE_AA)

    out = cv2.resize(out_big, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    return out
