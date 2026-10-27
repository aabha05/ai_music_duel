"""
agents.py - the two INTELLIGENT AGENTS.

PEAS description (same for both, personalities differ):
    Performance : judge score of its phrases minus the rival's, summed over the duel
    Environment : the shared musical state - chord progression, the rival's last phrase,
                  the mood set by the fuzzy controller. Multi-agent, competitive,
                  turn-based (sequential), fully observable, deterministic.
    Actuators   : plays a 2-bar phrase each turn (written to audio / MIDI)
    Sensors     : hears the rival's phrase, knows the current chords and mood

Agent type: a utility-based agent - it searches ahead (minimax) to maximise expected utility.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Musician:
    name: str
    index: int          # 0 = maximiser in minimax, 1 = minimiser
    voice: str
    personality: str
    motifs: tuple

    def style_bias(self, phrase):
        """Personality nudges the GA's search (not the judge) - up to ~1 point of fitness."""
        degs = np.array([g for g, _ in phrase])
        durs = np.array([d for _, d in phrase])
        iv = np.abs(np.diff(degs)) if len(degs) > 1 else np.zeros(1)
        if self.personality == "aggressive":      # many notes, big jumps
            return 0.6 * min(len(phrase) / 14, 1) + 0.4 * np.mean(iv >= 3)
        if self.personality == "playful":         # repeated notes, off-beat rhythms
            t = np.cumsum(np.r_[0, durs[:-1]])
            return 0.5 * np.mean(iv == 0) + 0.5 * np.mean(t % 1 != 0)
        return 0.6 * np.mean(iv <= 2) + 0.4 * np.mean(durs >= 1.5)   # lyrical: steps, long notes

    def peas(self):
        return {"Performance": "own judge score - rival's judge score",
                "Environment": "chords, rival's last phrase, fuzzy mood (multi-agent, sequential, deterministic)",
                "Actuators": "plays a 2-bar phrase",
                "Sensors": "rival's phrase, current chords, mood"}


PERSONALITY_MOTIFS = {
    "aggressive": (
        ((0, .5), (0, .5), (2, .5), (4, .5), (7, 1), (4, .5), (2, .5), (3, 1), (4, 1), (0, 2)),
        ((4, .5), (3, .5), (2, .5), (1, .5), (0, 1), (4, 1), (-1, .5), (0, .5), (2, .5), (4, .5), (7, 2)),
        ((0, 1), (7, .5), (6, .5), (4, 1), (2, .5), (4, .5), (5, 1.5), (4, .5), (2, 2)),
    ),
    "lyrical": (
        ((2, 1.5), (1, .5), (0, 2), (4, 1), (5, 1), (4, 2)),
        ((0, 2), (1, 1), (2, 1), (4, 1.5), (3, .5), (2, 2)),
        ((4, 1), (5, 1), (4, 1), (2, 1), (1, 2), (0, 2)),
    ),
    "playful": (
        ((0, .5), (0, .5), (0, 1), (2, .5), (2, .5), (4, 1), (4, .5), (5, .5), (4, 1), (2, 2)),
        ((4, 1.5), (4, .5), (2, .5), (4, .5), (5, 1), (4, .5), (4, .5), (2, 1), (0, 2)),
        ((2, .5), (2, .5), (4, .5), (2, .5), (0, 1), (0, .5), (1, .5), (2, 1.5), (2, .5), (0, 2)),
    ),
}
PERSONALITIES = tuple(PERSONALITY_MOTIFS)


def make_musician(name, index, voice, personality):
    return Musician(name=name, index=index, voice=voice, personality=personality,
                    motifs=PERSONALITY_MOTIFS[personality])


EMBER = make_musician("Ember", 0, "ember", "aggressive")
TIDE = make_musician("Tide", 1, "tide", "lyrical")
