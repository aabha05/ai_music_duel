"""
judge.py - the HEURISTIC EVALUATION FUNCTION used by the GA (fitness) and minimax (utility).

score(phrase) in [0, 10] = weighted sum of:
    critic     - neural network: does it sound like a real melody?
    harmony    - strong-beat notes are chord tones; phrase ends on a chord tone
    mood       - matches the fuzzy controller's density / register / leapiness targets
    coherence  - a good *answer*: starts near where the call ended, echoes its rhythm
                 (but not too much), and is not a copy

The coherence term is what makes it a real game: the phrase I play becomes the
opponent's "call", so an awkward ending or a strange rhythm makes their good replies harder.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from music import triad, onsets, note_at
from critic import melodic_features

WEIGHTS = {"critic": 0.30, "harmony": 0.30, "mood": 0.20, "coherence": 0.20}


@dataclass(frozen=True)
class Context:
    turn: int
    chords: tuple
    call: Optional[tuple]


def _rhythm_slots(phrase):
    return {int(round(t * 2)) for t in onsets(phrase)}


class Judge:
    def __init__(self, critic, mood, weights=None):
        self.critic = critic
        self.mood = mood
        w = dict(WEIGHTS, **(weights or {}))
        total = sum(w.values()) or 1.0
        self.weights = {k: v / total for k, v in w.items()}      # always sums to 1, so scores stay out of 10
        self._critic_cache = {}
        self._cache = {}
        self.evaluations = 0

    def critic_score(self, phrase):
        if phrase not in self._critic_cache:
            self._critic_cache[phrase] = float(self.critic.predict(melodic_features(phrase))[0])
        return self._critic_cache[phrase]

    def breakdown(self, phrase, ctx):
        degs = np.array([g for g, _ in phrase])
        iv = np.abs(np.diff(degs)) if len(degs) > 1 else np.zeros(1)

        # harmony: notes sounding on strong beats (1 and 3 of each bar) are chord tones
        strong = [note_at(phrase, t) % 7 in triad(ctx.chords[0] if t < 4 else ctx.chords[1]) for t in (0, 2, 4, 6)]
        harmony = 0.7 * np.mean(strong) + 0.3 * float(degs[-1] % 7 in triad(ctx.chords[1]))

        m = self.mood
        density_fit = np.exp(-((len(phrase) / 8 - m.density) / 0.5) ** 2)
        register_fit = np.exp(-((degs.mean() - m.register) / 3.0) ** 2)
        leap_fit = np.exp(-((np.mean(iv >= 3) - 0.5 * m.leapiness) / 0.25) ** 2)
        range_fit = np.exp(-max(0, degs.max() - degs.min() - 8) / 3.0)      # stay within ~an octave
        mood = (density_fit + register_fit + leap_fit + range_fit) / 4

        if ctx.call is None:
            coherence = 0.7
        else:
            start_fit = np.exp(-(abs(phrase[0][0] - ctx.call[-1][0]) / 2.5) ** 2)
            a, b = _rhythm_slots(phrase), _rhythm_slots(ctx.call)
            sim = len(a & b) / len(a | b)
            echo_fit = max(0.0, 1 - abs(sim - 0.6) / 0.6)
            same = sum(1 for x, y in zip(phrase, ctx.call) if x == y) / max(len(phrase), len(ctx.call))
            coherence = 0.4 * start_fit + 0.3 * echo_fit + 0.3 * (1 - same)

        parts = {"critic": self.critic_score(phrase), "harmony": float(harmony),
                 "mood": float(mood), "coherence": float(coherence)}
        parts["total"] = 10 * sum(self.weights[k] * parts[k] for k in self.weights)
        return parts

    def score(self, phrase, ctx):
        key = (phrase, ctx.chords, ctx.call)
        if key not in self._cache:
            self.evaluations += 1
            self._cache[key] = self.breakdown(phrase, ctx)["total"]
        return self._cache[key]
