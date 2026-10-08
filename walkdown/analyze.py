"""Turn raw gauge reads into decisions a plant team can act on.

For each inspection point:
  REVIEW  - the reader isn't confident; the teleoperator confirms the value
            from the live view before anything is trusted
  ALARM   - outside alarm limits -> priority work order
  WATCH   - outside the normal window -> routine work order
  OK      - inside the normal window

On top of single readings, the history from earlier walkdowns is checked
for slow drift (projected to hit a limit soon) and sudden step changes.
These are things a person doing one walkdown with a clipboard rarely catches.
"""
import numpy as np


def classify(value, normal, alarm):
    lo_a, hi_a = alarm
    lo_n, hi_n = normal
    if value < lo_a or value > hi_a:
        return "ALARM"
    if value < lo_n or value > hi_n:
        return "WATCH"
    return "OK"


def trend(history, current, alarm, normal):
    """Linear trend over previous walkdowns + today's reading."""
    series = list(history) + [current]
    if len(series) < 4:
        return None
    x = np.arange(len(series), dtype=float)
    slope, _ = np.polyfit(x, series, 1)
    prev = float(np.mean(history[-3:]))
    band = (normal[1] - normal[0]) or 1.0
    out = {"slope_per_walkdown": round(float(slope), 3), "previous_avg": round(prev, 2)}

    # sudden step change vs recent average (> 50% of the normal band)
    if abs(current - prev) > 0.5 * band:
        out["flag"] = "STEP_CHANGE"
        out["note"] = (f"Jumped from about {prev:.1f} to {current:.1f} since the last walkdowns. "
                       "Check for a process upset, a stuck valve or a failing transmitter.")
        return out

    # slow drift: is a limit reached within the next few walkdowns?
    if abs(slope) > 0.05 * band:
        limit = alarm[1] if slope > 0 else alarm[0]
        n = (limit - current) / slope
        if 0 < n <= 4:
            out["flag"] = "DRIFT"
            out["walkdowns_to_alarm"] = round(float(n), 1)
            out["note"] = (f"Rising steadily about {abs(slope):.1f} per walkdown. At this rate it reaches the "
                           f"alarm limit ({limit:g}) in about {n:.0f} walkdown{'s' if round(n) != 1 else ''}."
                           if slope > 0 else
                           f"Falling steadily about {abs(slope):.1f} per walkdown. At this rate it reaches the "
                           f"alarm limit ({limit:g}) in about {n:.0f} walkdown{'s' if round(n) != 1 else ''}.")
    return out


PRIORITY = {"ALARM": "P1", "WATCH": "P2", "DRIFT": "P2", "STEP_CHANGE": "P1", "REVIEW": "P3"}


def recommend(point, status, tr):
    unit = point["gauge"]["unit"]
    v = point["reading"]["value"]
    lo, hi = point["normal"]
    if status == "REVIEW":
        return "Reading not trusted automatically. Operator to confirm from live view; clean or re-shoot the gauge."
    if status == "ALARM":
        side = "below" if v < lo else "above"
        msg = (f"Reading {v:.1f} {unit} is {side} the alarm limit. Notify the control room now "
               "and verify against the control-system transmitter.")
        if tr and tr.get("flag") == "STEP_CHANGE":
            msg += f" It was steady around {tr['previous_avg']:g} {unit} on previous walkdowns, so this is a sudden change."
        return msg
    if tr and tr.get("flag") == "STEP_CHANGE":
        return "Verify against the control-system transmitter. If confirmed, raise with operations before the next shift." 
    if tr and tr.get("flag") == "DRIFT":
        return "Schedule inspection before the next walkdown; review trend with process engineering."
    if status == "WATCH":
        return f"Outside the normal window ({lo:g}-{hi:g} {unit}). Monitor; check on next round."
    return ""


def analyze_point(point: dict, threshold: float) -> dict:
    r = point["reading"]
    status = classify(r["value"], point["normal"], point["alarm"])
    point["raw_status"] = status  # what an unchecked reader would have reported
    if r["confidence"] < threshold:
        status = "REVIEW"
    tr = None if status == "REVIEW" else trend(point["history"], r["value"], point["alarm"], point["normal"])
    point["status"] = status
    point["trend"] = tr
    flags = [status] if status != "OK" else []
    if tr and tr.get("flag"):
        flags.append(tr["flag"])
    point["flags"] = flags
    point["recommendation"] = recommend(point, status, tr)
    return point


def work_orders(points, walkdown_id):
    orders = []
    for p in points:
        if not p["flags"]:
            continue
        top = min(p["flags"], key=lambda f: PRIORITY[f])
        orders.append({
            "id": f"{walkdown_id}-{len(orders) + 1:02d}",
            "asset_tag": p["tag"],
            "asset": p["name"],
            "area": p["area"],
            "priority": PRIORITY[top],
            "type": top,
            "reading": f'{p["reading"]["value"]:.1f} {p["gauge"]["unit"]}',
            "normal_range": f'{p["normal"][0]:g}-{p["normal"][1]:g} {p["gauge"]["unit"]}',
            "action": p["recommendation"],
            "evidence": p["frame_annotated"],
        })
    orders.sort(key=lambda o: o["priority"])
    for i, o in enumerate(orders, 1):   # number in priority order
        o["id"] = f"{walkdown_id}-{i:02d}"
    return orders
