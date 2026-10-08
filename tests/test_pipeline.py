"""Fast checks that run on every commit (the full evaluation is in walkdown.evaluate)."""
import json
from pathlib import Path

import numpy as np
import pytest

from walkdown.route import ROUTE, GaugeSpec
from walkdown.synth import capture
from walkdown.reader import read_gauge
from walkdown.analyze import classify, trend

ROOT = Path(__file__).resolve().parent.parent


def test_calibration_round_trip():
    g = GaugeSpec(0, 160, "bar", 20)
    for v in np.linspace(0, 160, 17):
        assert g.angle_to_value(g.value_to_angle(v)) == pytest.approx(v, abs=1e-6)


@pytest.mark.parametrize("value", [3, 25, 50, 77, 97])
def test_reads_clean_gauge_within_one_percent(value):
    g = GaugeSpec(0, 100, "bar", 10)
    r = read_gauge(capture(g, value, "PI-TEST", "clean", seed=value), g)
    assert abs(r.value - value) <= 1.0
    assert r.confidence > 0.8


@pytest.mark.parametrize("condition", ["glare", "low_light", "blur", "smoke", "tilt", "occluded"])
def test_mild_conditions_read_within_tolerance(condition):
    g = GaugeSpec(0, 100, "bar", 10)
    r = read_gauge(capture(g, 62, "PI-TEST", condition, seed=3, severity=0.2), g)
    assert abs(r.value - 62) <= 2.0


def test_low_confidence_on_destroyed_frame():
    p = next(x for x in ROUTE if x.tag == "PI-5310")
    r = read_gauge(capture(p.gauge, p.true_value, p.tag, p.condition, p.seed, p.severity), p.gauge)
    threshold = json.loads((ROOT / "docs/data/evaluation.json").read_text())["confidence_threshold"]
    assert r.confidence < threshold


def test_classify():
    assert classify(60, (55, 70), (40, 85)) == "OK"
    assert classify(72, (55, 70), (40, 85)) == "WATCH"
    assert classify(30, (55, 70), (40, 85)) == "ALARM"


def test_drift_and_step_change():
    drift = trend([112, 115, 118, 121, 124], 128, (80, 135), (95, 120))
    assert drift["flag"] == "DRIFT" and drift["walkdowns_to_alarm"] <= 4
    step = trend([72, 71, 73, 72, 71], 38, (45, 90), (60, 80))
    assert step["flag"] == "STEP_CHANGE"
    steady = trend([62, 64, 63, 62, 63], 63, (40, 85), (55, 70))
    assert "flag" not in steady


def test_api_reads_frame():
    from fastapi.testclient import TestClient
    import cv2
    from walkdown.api import app
    p = next(x for x in ROUTE if x.tag == "PI-2101")
    ok, buf = cv2.imencode(".jpg", capture(p.gauge, 63, p.tag, "clean", seed=1))
    res = TestClient(app).post("/read", files={"image": ("f.jpg", buf.tobytes(), "image/jpeg")},
                               data={"tag": "PI-2101"})
    body = res.json()
    assert res.status_code == 200
    assert abs(body["reading"]["value"] - 63) < 1.5 and body["status"] == "OK"
