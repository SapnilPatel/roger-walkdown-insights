"""HTTP service so the reader can sit next to the teleop stack.

    uvicorn walkdown.api:app --reload

POST /read  (multipart)  image=<frame>  tag=PI-2350
    -> reading, status, whether the operator must confirm

The gauge calibration and limits come from the asset register (route.py
here), keyed by tag, so the robot only has to send the frame and the tag of
the gauge it is looking at.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .route import ROUTE
from .reader import read_gauge
from .analyze import classify

app = FastAPI(title="Roger walkdown insights", version="0.1.0")
REGISTER = {p.tag: p for p in ROUTE}
EVAL = Path(__file__).resolve().parent.parent / "docs" / "data" / "evaluation.json"
THRESHOLD = json.loads(EVAL.read_text())["confidence_threshold"] if EVAL.exists() else 0.2


@app.get("/health")
def health():
    return {"ok": True, "gauges_registered": len(REGISTER), "confidence_threshold": THRESHOLD}


@app.get("/assets")
def assets():
    return [{"tag": p.tag, "name": p.name, "area": p.area, "unit": p.gauge.unit,
             "normal": p.normal, "alarm": p.alarm} for p in ROUTE]


@app.post("/read")
async def read(image: UploadFile = File(...), tag: str = Form(...)):
    if tag not in REGISTER:
        raise HTTPException(404, f"Unknown asset tag {tag}")
    buf = np.frombuffer(await image.read(), np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")
    p = REGISTER[tag]
    r = read_gauge(frame, p.gauge)
    trusted = r.confidence >= THRESHOLD
    return {
        "tag": tag,
        "reading": r.to_dict(),
        "unit": p.gauge.unit,
        "status": classify(r.value, p.normal, p.alarm) if trusted else "REVIEW",
        "operator_confirmation_required": not trusted,
    }
