"""Run one simulated walkdown end to end.

    python -m walkdown.run            # walkdown only (uses saved evaluation)
    python -m walkdown.run --eval     # also re-run the accuracy evaluation

Writes everything the dashboard needs into docs/ (served by GitHub Pages):
    docs/frames/<tag>.jpg             what Roger's camera captured
    docs/frames/<tag>_seen.jpg        what the pipeline detected
    docs/data/walkdown.json           readings, statuses, trends, work orders
    docs/data/evaluation.json         accuracy on held-out simulated frames
    docs/data/work_orders.csv         ready to import into a maintenance system
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2

from .route import ROUTE, PLATFORM
from .synth import capture
from .reader import read_gauge, annotate
from .analyze import analyze_point, work_orders
from . import evaluate as ev

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
STATUS_BGR = {"OK": (90, 200, 90), "WATCH": (0, 190, 255), "ALARM": (60, 60, 230), "REVIEW": (230, 160, 60)}


def main(run_eval: bool = False):
    (DOCS / "frames").mkdir(parents=True, exist_ok=True)
    (DOCS / "data").mkdir(parents=True, exist_ok=True)

    eval_path = DOCS / "data" / "evaluation.json"
    if run_eval or not eval_path.exists():
        print("Running accuracy evaluation (a few minutes)...")
        evaluation = ev.evaluate()
        eval_path.write_text(json.dumps(evaluation, indent=2))
    evaluation = json.loads(eval_path.read_text())
    threshold = evaluation["confidence_threshold"]

    points = []
    t0 = time.perf_counter()
    for p in ROUTE:
        frame = capture(p.gauge, p.true_value, p.tag, p.condition, p.seed, p.severity)
        reading = read_gauge(frame, p.gauge)
        d = p.to_dict()
        d["reading"] = reading.to_dict()
        d["frame"] = f"frames/{p.tag}.jpg"
        d["frame_annotated"] = f"frames/{p.tag}_seen.jpg"
        d = analyze_point(d, threshold)
        cv2.imwrite(str(DOCS / d["frame"]), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        seen = annotate(frame, reading, p.gauge, STATUS_BGR[d["status"]])
        cv2.imwrite(str(DOCS / d["frame_annotated"]), seen, [cv2.IMWRITE_JPEG_QUALITY, 85])
        # ground truth is kept for the demo's honesty panel only, never used by the pipeline
        d["ground_truth"] = d.pop("true_value")
        points.append(d)
    elapsed = time.perf_counter() - t0

    orders = work_orders(points, PLATFORM["walkdown_id"])
    hazardous = [p for p in points if not p["hazard"].startswith("Low hazard")]
    summary = {
        "stops": len(points),
        "ok": sum(p["status"] == "OK" for p in points),
        "watch": sum(p["status"] == "WATCH" for p in points),
        "alarm": sum(p["status"] == "ALARM" for p in points),
        "review": sum(p["status"] == "REVIEW" for p in points),
        "trend_flags": sum(1 for p in points if p["trend"] and p["trend"].get("flag")),
        "hazardous_stops": len(hazardous),
        "processing_seconds": round(elapsed, 2),
    }

    out = {
        "platform": PLATFORM,
        "summary": summary,
        "confidence_threshold": threshold,
        "points": points,
        "work_orders": orders,
    }
    (DOCS / "data" / "walkdown.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    with open(DOCS / "data" / "work_orders.csv", "w", newline="") as f:
        if orders:
            w = csv.DictWriter(f, fieldnames=list(orders[0].keys()))
            w.writeheader()
            w.writerows(orders)

    render_dashboard(out, evaluation)

    print(f"Walkdown {PLATFORM['walkdown_id']}: {summary}")
    for o in orders:
        print(f"  {o['priority']} {o['asset_tag']:8} {o['type']:12} {o['reading']:>10}  {o['action']}")
    return out


REPO_URL = os.environ.get("REPO_URL", "https://github.com/SapnilPatel/roger-walkdown-insights")


def render_dashboard(walkdown: dict, evaluation: dict):
    """Bake the results into docs/index.html so the page works as a plain
    static file (GitHub Pages, any host, or opened locally) with no fetches."""
    def as_script(obj):
        return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")
    html = (DOCS / "template.html").read_text()
    html = html.replace("/*__DATA__*/", as_script(walkdown))
    html = html.replace("/*__EVAL__*/", as_script(evaluation))
    html = html.replace("__REPO_URL__", REPO_URL)
    html = html.replace("__REPO_LABEL__", REPO_URL.replace("https://", ""))
    # GitHub Pages / any static host needs a full document
    head, body = html.split("<style>", 1)
    page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            + head + "<style>\nhtml{-webkit-text-size-adjust:100%}body{margin:0}img{max-width:100%}[hidden]{display:none!important}\n"
            + body.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n")
    (DOCS / "index.html").write_text(page)
    # the claude.ai artifact host adds its own document skeleton, so it gets the fragment
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "artifact.html").write_text(html)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="re-run the accuracy evaluation")
    main(ap.parse_args().eval)
