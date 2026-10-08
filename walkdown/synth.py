"""Render simulated camera frames of analog gauges.

Real pilot footage is not public, so frames are synthesised with a known
ground-truth value. That gives an exact answer key for every frame, which is
what makes the accuracy evaluation in evaluate.py possible.

Conditions mirror what a robot actually sees on a platform:
  clean      - good light, head-on view
  glare      - specular reflection on the gauge glass
  low_light  - dim module at night, sensor noise
  blur       - motion blur from the robot's head or a vibrating pipe
  smoke      - haze from a leak, steam or smoke
  tilt       - gauge viewed off-axis (robot couldn't stand square to it)
  occluded   - fogged or dirty glass covering most of the face
"""
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .route import GaugeSpec

SIZE = 480
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(px):
    try:
        return ImageFont.truetype(FONT_PATH, px)
    except OSError:
        return ImageFont.load_default(size=px)


def _polar(cx, cy, r, angle_deg):
    a = math.radians(angle_deg)
    return cx + r * math.cos(a), cy - r * math.sin(a)


def render_gauge(spec: GaugeSpec, value: float, tag: str, seed: int = 0) -> np.ndarray:
    """Draw a clean gauge face (BGR uint8)."""
    rng = np.random.default_rng(seed)
    scale = 3  # supersample for anti-aliasing
    S = SIZE * scale
    # industrial background: painted steel / pipework
    bg = tuple(int(v) for v in rng.integers(70, 110, 3))
    img = Image.new("RGB", (S, S), bg)
    d = ImageDraw.Draw(img)

    # a pipe behind the gauge
    d.rectangle([0, S * 0.78, S, S * 0.9], fill=tuple(max(0, c - 25) for c in bg))

    cx, cy = S / 2 + rng.integers(-15, 15) * scale, S / 2 + rng.integers(-15, 15) * scale
    R = S * 0.38
    # bezel and face
    d.ellipse([cx - R * 1.12, cy - R * 1.12, cx + R * 1.12, cy + R * 1.12], fill=(40, 40, 44))
    d.ellipse([cx - R * 1.04, cy - R * 1.04, cx + R * 1.04, cy + R * 1.04], fill=(150, 150, 155))
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(238, 236, 228))

    # red band for the upper 15% of scale (typical on process gauges)
    red_start = spec.value_to_angle(spec.min_value + 0.85 * (spec.max_value - spec.min_value))
    end = spec.value_to_angle(spec.max_value)
    d.arc([cx - R * 0.9, cy - R * 0.9, cx + R * 0.9, cy + R * 0.9],
          start=-red_start, end=-end, fill=(200, 40, 40), width=int(R * 0.06))

    # ticks
    span = spec.max_value - spec.min_value
    n_major = int(round(span / spec.major_step))
    for i in range(n_major * 5 + 1):
        v = spec.min_value + i * spec.major_step / 5
        a = spec.value_to_angle(v)
        major = i % 5 == 0
        r0 = R * (0.80 if major else 0.86)
        p0, p1 = _polar(cx, cy, r0, a), _polar(cx, cy, R * 0.93, a)
        d.line([p0, p1], fill=(20, 20, 20), width=int(scale * (4 if major else 2)))
        if major:
            tx, ty = _polar(cx, cy, R * 0.68, a)
            label = f"{v:g}"
            f = _font(int(R * 0.11))
            w = d.textlength(label, font=f)
            d.text((tx - w / 2, ty - R * 0.06), label, fill=(20, 20, 20), font=f)

    # unit + tag text in the lower half (below the hub, outside the needle sweep path)
    f = _font(int(R * 0.12))
    w = d.textlength(spec.unit, font=f)
    d.text((cx - w / 2, cy + R * 0.30), spec.unit, fill=(40, 40, 40), font=f)
    f2 = _font(int(R * 0.08))
    w = d.textlength(tag, font=f2)
    d.text((cx - w / 2, cy + R * 0.48), tag, fill=(90, 90, 90), font=f2)

    # needle (black, tapered) with a short tail
    a = spec.value_to_angle(value)
    tip = _polar(cx, cy, R * 0.82, a)
    tail = _polar(cx, cy, R * 0.18, a + 180)
    perp = a + 90
    b1, b2 = _polar(cx, cy, R * 0.035, perp), _polar(cx, cy, R * 0.035, perp + 180)
    d.polygon([tip, b1, tail, b2], fill=(15, 15, 15))
    d.ellipse([cx - R * 0.07, cy - R * 0.07, cx + R * 0.07, cy + R * 0.07], fill=(25, 25, 25))

    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def degrade(img: np.ndarray, condition: str, seed: int = 0, severity: float = None) -> np.ndarray:
    """Apply a camera condition. severity in [0, 1]: 0 = mild, 1 = extreme.
    If not given it is drawn at random, so evaluation covers the whole range,
    including frames no reader should trust."""
    rng = np.random.default_rng(seed + 1000)
    s = float(rng.uniform(0, 1)) if severity is None else float(severity)
    out = img.astype(np.float32)
    h, w = out.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]

    if condition == "glare":
        cx, cy = w * rng.uniform(0.32, 0.68), h * rng.uniform(0.28, 0.55)
        sx, sy = w * (0.08 + 0.12 * s), h * (0.05 + 0.08 * s)
        blob = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        out = out + (140 + 220 * s) * blob[..., None]
    elif condition == "low_light":
        out = out * (0.40 - 0.30 * s) + rng.normal(0, 6 + 16 * s, out.shape)
    elif condition == "blur":
        k = int(7 + 22 * s) | 1
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2, k / 2), float(rng.uniform(0, 180)), 1)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        kernel /= kernel.sum()
        out = cv2.filter2D(out, -1, kernel)
    elif condition == "smoke":
        haze = cv2.GaussianBlur(rng.uniform(0, 1, (h // 8, w // 8)).astype(np.float32), (0, 0), 3)
        haze = cv2.resize(haze, (w, h))
        alpha = (0.30 + 0.55 * s + 0.15 * haze)[..., None].clip(0, 0.95)
        out = out * (1 - alpha) + 175 * alpha
    elif condition == "tilt":
        m = 0.03 + 0.14 * s
        sign = 1 if rng.uniform() < 0.5 else -1
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        if sign > 0:
            dst = np.float32([[w * m, h * m / 2], [w, 0], [w * m / 2, h], [w * (1 - m / 4), h * (1 - m / 2)]])
        else:
            dst = np.float32([[0, 0], [w * (1 - m), h * m / 2], [w * m / 4, h * (1 - m / 2)], [w * (1 - m / 2), h]])
        M = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif condition == "occluded":
        # condensation / grime: a bright smear over part or most of the face
        cx, cy = w * rng.uniform(0.40, 0.60), h * rng.uniform(0.36, 0.50)
        blob = np.exp(-(((xx - cx) / (w * (0.18 + 0.16 * s))) ** 2 + ((yy - cy) / (h * (0.15 + 0.14 * s))) ** 2))
        k = 0.65 + 0.33 * s
        out = out * (1 - k * blob[..., None]) + 215 * k * blob[..., None]
        out = cv2.GaussianBlur(out, (0, 0), 2.5)

    return np.clip(out, 0, 255).astype(np.uint8)


def capture(spec: GaugeSpec, value: float, tag: str, condition: str, seed: int = 0,
            severity: float = None) -> np.ndarray:
    """Simulate Roger's head camera capturing one gauge."""
    return degrade(render_gauge(spec, value, tag, seed), condition, seed, severity)
