"""
fuzzy_mood.py - SOFT COMPUTING: a Mamdani fuzzy inference system, written from scratch.

Turns two vague human inputs  ->  concrete musical parameters.
    inputs : energy   in [0, 1]   ("how intense?")
             darkness in [0, 1]   ("how ominous?")
    outputs: tempo (BPM), density (notes per beat), leapiness, register, brightness -> mode

Pipeline (exactly the syllabus steps):
    1. Fuzzification     - triangular membership functions
    2. Rule evaluation   - AND = min (fuzzy intersection)
    3. Implication       - clip each consequent set at the rule's firing strength
    4. Aggregation       - OR = max (fuzzy union) of all clipped sets
    5. Defuzzification   - centroid of the aggregated set
"""
from dataclasses import dataclass, field
import numpy as np


def trimf(x, a, b, c):
    """Triangular membership. a==b or b==c gives a left/right shoulder."""
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if b == a else np.clip((x - a) / (b - a), 0, 1)
    right = np.ones_like(x) if c == b else np.clip((c - x) / (c - b), 0, 1)
    y = np.minimum(left, right)
    y[(x < a) | (x > c)] = 0.0
    return y


INPUT_SETS = {
    "energy":   {"low": (0, 0, 0.5), "mid": (0.2, 0.5, 0.8), "high": (0.5, 1, 1)},
    "darkness": {"low": (0, 0, 0.5), "mid": (0.2, 0.5, 0.8), "high": (0.5, 1, 1)},
}

OUTPUT_SETS = {
    "tempo":      ((60, 180),  {"slow": (60, 60, 100), "medium": (80, 115, 150), "fast": (130, 180, 180)}),
    "density":    ((0.5, 2.0), {"sparse": (0.5, 0.5, 1.0), "moderate": (0.7, 1.1, 1.5), "dense": (1.2, 2.0, 2.0)}),
    "leapiness":  ((0, 1),     {"smooth": (0, 0, 0.45), "mixed": (0.2, 0.5, 0.8), "wild": (0.55, 1, 1)}),
    "register":   ((-2, 9),    {"low": (-2, -2, 3), "mid": (1, 3.5, 6), "high": (4, 9, 9)}),
    "brightness": ((0, 1),     {"dark": (0, 0, 0.45), "neutral": (0.25, 0.5, 0.75), "bright": (0.55, 1, 1)}),
}

# (antecedents joined by AND) -> consequents
RULES = [
    ([("energy", "high")],                       [("tempo", "fast"), ("density", "dense")]),
    ([("energy", "mid")],                        [("tempo", "medium"), ("density", "moderate"), ("leapiness", "mixed")]),
    ([("energy", "low")],                        [("tempo", "slow"), ("density", "sparse")]),
    ([("darkness", "high")],                     [("brightness", "dark")]),
    ([("darkness", "mid")],                      [("brightness", "neutral"), ("register", "mid")]),
    ([("darkness", "low")],                      [("brightness", "bright")]),
    ([("energy", "high"), ("darkness", "high")], [("leapiness", "wild"), ("register", "low")]),     # aggressive growl
    ([("energy", "high"), ("darkness", "low")],  [("leapiness", "mixed"), ("register", "high")]),   # euphoric
    ([("energy", "low"), ("darkness", "low")],   [("leapiness", "smooth"), ("register", "high")]),  # dreamy
    ([("energy", "low"), ("darkness", "high")],  [("leapiness", "smooth"), ("register", "low")]),   # brooding
]


@dataclass
class MoodParams:
    energy: float
    darkness: float
    tempo: float
    density: float
    leapiness: float
    register: float
    brightness: float
    mode: str
    firing: list = field(default_factory=list)


def mode_from_brightness(b):
    if b < 0.3:
        return "phrygian"
    if b < 0.5:
        return "minor"
    if b < 0.7:
        return "dorian"
    return "major"


def infer(energy, darkness, resolution=401):
    inputs = {"energy": float(np.clip(energy, 0, 1)), "darkness": float(np.clip(darkness, 0, 1))}

    # 1. fuzzification
    degree = {(var, term): float(trimf([val], *INPUT_SETS[var][term])[0])
              for var, val in inputs.items() for term in INPUT_SETS[var]}

    universes = {o: np.linspace(lo, hi, resolution) for o, ((lo, hi), _) in OUTPUT_SETS.items()}
    aggregated = {o: np.zeros(resolution) for o in OUTPUT_SETS}
    firing = []

    for antecedents, consequents in RULES:
        strength = min(degree[a] for a in antecedents)                       # 2. AND = min
        firing.append((antecedents, consequents, strength))
        if strength <= 0:
            continue
        for out, term in consequents:
            mf = trimf(universes[out], *OUTPUT_SETS[out][1][term])
            clipped = np.minimum(mf, strength)                               # 3. implication
            aggregated[out] = np.maximum(aggregated[out], clipped)           # 4. aggregation

    crisp = {}
    for out, agg in aggregated.items():                                     # 5. centroid
        lo, hi = OUTPUT_SETS[out][0]
        crisp[out] = float((universes[out] * agg).sum() / agg.sum()) if agg.sum() > 0 else (lo + hi) / 2

    return MoodParams(energy=inputs["energy"], darkness=inputs["darkness"],
                      tempo=crisp["tempo"], density=crisp["density"], leapiness=crisp["leapiness"],
                      register=crisp["register"], brightness=crisp["brightness"],
                      mode=mode_from_brightness(crisp["brightness"]), firing=firing)


def explain(m):
    lines = [f"Fuzzy inputs: energy={m.energy:.2f}, darkness={m.darkness:.2f}", "Rule firing strengths:"]
    for ants, cons, s in m.firing:
        if s > 0:
            a = " AND ".join(f"{v} is {t}" for v, t in ants)
            c = ", ".join(f"{v} is {t}" for v, t in cons)
            lines.append(f"  {s:4.2f}  IF {a} THEN {c}")
    lines.append(f"Defuzzified -> tempo {m.tempo:.0f} BPM | density {m.density:.2f} notes/beat | "
                 f"leapiness {m.leapiness:.2f} | register {m.register:+.1f} | "
                 f"brightness {m.brightness:.2f} -> {m.mode} mode")
    return "\n".join(lines)
