"""Inspection route for a simulated walkdown.

Everything here is fictional: "Demo Platform Alpha" is a made-up offshore
production platform used to exercise the pipeline. Tags, limits and areas are
illustrative, chosen to look like a real topsides inspection round.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class GaugeSpec:
    """How a physical analog gauge is calibrated.

    The dial sweeps 270 degrees: the minimum sits at the 7-8 o'clock
    position, the maximum at the 4-5 o'clock position (standard for
    industrial bourdon-tube gauges).
    """
    min_value: float
    max_value: float
    unit: str
    major_step: float
    start_angle_deg: float = 225.0   # angle of min value, math convention (0 = 3 o'clock, CCW+)
    sweep_deg: float = 270.0         # clockwise sweep from min to max

    def value_to_angle(self, value: float) -> float:
        frac = (value - self.min_value) / (self.max_value - self.min_value)
        return self.start_angle_deg - self.sweep_deg * frac

    def angle_to_value(self, angle_deg: float) -> float:
        # normalise so the result is measured clockwise from the start angle
        delta = (self.start_angle_deg - angle_deg) % 360.0
        if delta > self.sweep_deg + 20:   # in the dead zone below the dial: snap to nearest end
            delta = 0.0 if delta > (self.sweep_deg + 360) / 2 else self.sweep_deg
        delta = min(delta, self.sweep_deg)
        return self.min_value + (delta / self.sweep_deg) * (self.max_value - self.min_value)


@dataclass
class InspectionPoint:
    tag: str                  # asset tag, e.g. PI-2101 (pressure indicator)
    name: str
    area: str
    hazard: str               # why a person entering this area is at risk
    gauge: GaugeSpec
    normal: tuple             # (low, high) normal operating window
    alarm: tuple              # (low, high) beyond which a work order is raised
    true_value: float         # ground truth used to render the simulated frame
    condition: str = "clean"  # camera condition when Roger captured the frame
    history: list = field(default_factory=list)  # readings from previous walkdowns
    severity: float = 0.6     # how bad the camera condition is (0 mild .. 1 extreme)
    seed: int = 0             # fixes the random details of the simulated frame

    def to_dict(self):
        d = asdict(self)
        d["normal"] = list(self.normal)
        d["alarm"] = list(self.alarm)
        return d


def _g(lo, hi, unit, step):
    return GaugeSpec(min_value=lo, max_value=hi, unit=unit, major_step=step)


PLATFORM = {
    "name": "Demo Platform Alpha",
    "type": "Offshore production platform (fictional)",
    "walkdown_id": "WD-0417",
    "operator_station": "Control room, Deck A (safe area)",
    "robot": "Humanoid inspection unit, teleoperated",
    "date": "2026-10-07",
    "shift": "Day shift",
}


# Previous five walkdowns per point (oldest -> newest). Hand-tuned so the
# demo contains one slow drift that only shows up across walkdowns.
ROUTE = [
    InspectionPoint("PI-2101", "HP separator pressure", "Separation module, Deck B",
                    "Hydrocarbon release zone", _g(0, 100, "bar", 10),
                    (55, 70), (40, 85), 63.0, "clean", [62, 64, 63, 62, 63]),
    InspectionPoint("PI-2104", "LP separator pressure", "Separation module, Deck B",
                    "Hydrocarbon release zone", _g(0, 25, "bar", 5),
                    (6, 12), (3, 18), 9.2, "glare", [9.0, 9.4, 9.1, 9.3, 9.2]),
    InspectionPoint("TI-2210", "Glycol contactor outlet temp", "Gas dehydration, Deck B",
                    "Hot surfaces, hydrocarbon zone", _g(0, 150, "°C", 25),
                    (35, 55), (20, 75), 47.0, "clean", [44, 45, 46, 46, 47]),
    InspectionPoint("PI-2350", "Compressor discharge pressure", "Gas compression, Deck C",
                    "High-pressure gas, noise > 105 dB", _g(0, 160, "bar", 20),
                    (95, 120), (80, 135), 128.0, "low_light", [112, 115, 118, 121, 124]),
    InspectionPoint("PI-2352", "Compressor seal gas pressure", "Gas compression, Deck C",
                    "High-pressure gas, noise > 105 dB", _g(0, 40, "bar", 5),
                    (18, 26), (14, 30), 21.5, "blur", [21, 22, 21.5, 21, 22]),
    InspectionPoint("PI-3101", "Crude export pump suction", "Export pumps, Deck C",
                    "Rotating equipment", _g(0, 16, "bar", 2),
                    (3, 8), (2, 11), 5.4, "clean", [5.5, 5.2, 5.6, 5.3, 5.5]),
    InspectionPoint("PI-3105", "Crude export pump discharge", "Export pumps, Deck C",
                    "Rotating equipment", _g(0, 100, "bar", 10),
                    (60, 80), (45, 90), 38.0, "smoke", [72, 71, 73, 72, 71]),
    InspectionPoint("PI-4401", "Wellhead A-3 tubing pressure", "Wellbay, Deck D",
                    "H2S area, breathing apparatus required", _g(0, 250, "bar", 50),
                    (140, 190), (110, 215), 168.0, "clean", [170, 169, 171, 168, 169]),
    InspectionPoint("PI-4402", "Wellhead A-3 annulus (A) pressure", "Wellbay, Deck D",
                    "H2S area, breathing apparatus required", _g(0, 100, "bar", 10),
                    (0, 20), (0, 35), 12.0, "tilt", [10, 11, 11, 12, 12]),
    InspectionPoint("PI-5201", "Firewater ring main pressure", "Utilities, Deck A",
                    "Low hazard (reference point)", _g(0, 25, "bar", 5),
                    (10, 14), (8, 16), 12.1, "clean", [12.0, 12.2, 12.1, 12.0, 12.1]),
    InspectionPoint("PI-5310", "Fuel gas supply pressure", "Utilities, Deck A",
                    "Hydrocarbon release zone", _g(0, 60, "bar", 10),
                    (25, 35), (18, 42), 29.0, "glare", [29, 30, 29, 30, 29], 1.0, 1),
    InspectionPoint("TI-5402", "Lube oil cooler outlet temp", "Utilities, Deck A",
                    "Hot surfaces", _g(0, 120, "°C", 20),
                    (40, 60), (30, 75), 52.0, "occluded", [50, 51, 52, 51, 52]),
]


# give each stop its own (but reproducible) random frame details
for _i, _p in enumerate(ROUTE):
    if _p.seed == 0:
        _p.seed = 10 + _i
