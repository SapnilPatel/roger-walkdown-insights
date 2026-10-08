"""Measure reader accuracy against ground truth, per camera condition.

Method (kept deliberately honest):
  * Frames are generated with random true values on every gauge type in the
    route, under every condition.
  * A CALIBRATION set picks the confidence threshold: the lowest threshold at
    which >= 99% of auto-accepted reads fall within tolerance.
  * A separate HELD-OUT set (different random seeds) reports the numbers.
    The threshold never sees these frames.

Tolerance: 2% of full scale, which is about the accuracy class of a typical
industrial process gauge (class 1.0-1.6). Reading better than the gauge
itself is not meaningful.
"""
import json
import time
import numpy as np

from .route import ROUTE
from .synth import capture
from .reader import read_gauge

CONDITIONS = ["clean", "glare", "low_light", "blur", "smoke", "tilt", "occluded"]
TOLERANCE = 0.02  # fraction of full scale


def _run(n_per_condition: int, seed_offset: int):
    rng = np.random.default_rng(seed_offset)
    rows = []
    for cond in CONDITIONS:
        for k in range(n_per_condition):
            p = ROUTE[rng.integers(len(ROUTE))]
            g = p.gauge
            span = g.max_value - g.min_value
            true = float(rng.uniform(g.min_value + 0.03 * span, g.max_value - 0.03 * span))
            seed = seed_offset * 100000 + len(rows)
            t0 = time.perf_counter()
            r = read_gauge(capture(g, true, p.tag, cond, seed), g)
            ms = (time.perf_counter() - t0) * 1000
            rows.append({
                "condition": cond, "tag": p.tag, "true": true, "read": r.value,
                "abs_err_pct": abs(r.value - true) / span * 100,
                "confidence": r.confidence, "ms": ms,
            })
    return rows


def pick_threshold(rows, target=0.99):
    """Lowest threshold where EVERY condition meets the target on its own.
    Averaging across conditions would let easy clean frames hide failures in
    glare or blur, so the worst condition decides."""
    for t in np.arange(0.0, 1.0001, 0.01):
        worst = 1.0
        for cond in CONDITIONS:
            acc = [r for r in rows if r["condition"] == cond and r["confidence"] >= t]
            if acc:
                worst = min(worst, np.mean([r["abs_err_pct"] <= TOLERANCE * 100 for r in acc]))
        if worst >= target:
            return round(float(t), 2)
    return 1.0


def summarise(rows, threshold):
    out = {}
    for cond in CONDITIONS + ["all"]:
        sub = rows if cond == "all" else [r for r in rows if r["condition"] == cond]
        err = np.array([r["abs_err_pct"] for r in sub])
        within = err <= TOLERANCE * 100
        auto = np.array([r["confidence"] >= threshold for r in sub])
        out[cond] = {
            "n": len(sub),
            "median_err_pct_fs": round(float(np.median(err)), 3),
            "within_tol_pct": round(100 * float(within.mean()), 1),
            "auto_accepted_pct": round(100 * float(auto.mean()), 1),
            "auto_accepted_within_tol_pct": round(100 * float(within[auto].mean()), 1) if auto.any() else None,
            "sent_to_operator_pct": round(100 * float((~auto).mean()), 1),
            "bad_reads_caught_pct": round(100 * float((~auto[~within]).mean()), 1) if (~within).any() else None,
            "median_ms": round(float(np.median([r["ms"] for r in sub])), 1),
        }
    return out


def evaluate(n_per_condition=150):
    calib = _run(n_per_condition, seed_offset=1)
    threshold = pick_threshold(calib)
    held = _run(n_per_condition, seed_offset=2)
    return {
        "tolerance_pct_full_scale": TOLERANCE * 100,
        "confidence_threshold": threshold,
        "n_calibration": len(calib),
        "n_held_out": len(held),
        "held_out": summarise(held, threshold),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
