"""
music.py - the shared musical vocabulary of the duel.

Representation (this is the "state" the AI reasons about):
  * A NOTE is a tuple (degree, duration)
      - degree   : scale degree relative to the tonic (0 = tonic, 7 = tonic an octave up,
                   -3 = the 5th below). Working in degrees keeps every note "in key".
      - duration : length in beats, one of ALLOWED_DURS.
  * A PHRASE is a tuple of notes lasting exactly PHRASE_BEATS (2 bars of 4/4).
    Tuples are hashable, so phrases can be cached and used as search-tree keys.
"""

SCALES = {
    "major":    [0, 2, 4, 5, 7, 9, 11],
    "dorian":   [0, 2, 3, 5, 7, 9, 10],
    "minor":    [0, 2, 3, 5, 7, 8, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
}

ALLOWED_DURS = (0.5, 1.0, 1.5, 2.0)
PHRASE_BEATS = 8.0
BEATS_PER_BAR = 4.0
DEG_MIN, DEG_MAX = -7, 14

# Chord progressions as scale-degree roots. Each turn (2 bars) is played over two consecutive chords.
PROGRESSIONS = {
    "epic":   (0, 5, 3, 4),   # I - vi - IV - V   (i - VI - iv - v in minor)
    "pop":    (0, 4, 5, 3),   # I - V - vi - IV   (the "four chord song")
    "jazz":   (1, 4, 0, 5),   # ii - V - I - vi
    "blues":  (0, 3, 0, 4),   # I - IV - I - V
    "dreamy": (0, 3, 5, 3),   # I - IV - vi - IV
}
PROGRESSION = PROGRESSIONS["epic"]


def set_progression(name):
    """Choose the chord progression for the next duel."""
    global PROGRESSION
    PROGRESSION = PROGRESSIONS[name]

KEYS = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
        "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]


def triad(root):
    """Scale degrees (mod 7) of the triad built on `root`."""
    return {root % 7, (root + 2) % 7, (root + 4) % 7}


def chords_for_turn(turn):
    """The two chord roots underneath a given turn."""
    n = len(PROGRESSION)
    return PROGRESSION[(2 * turn) % n], PROGRESSION[(2 * turn + 1) % n]


def chord_name(root, mode):
    scale = SCALES[mode]
    third = (scale[(root + 2) % 7] + 12 * ((root + 2) // 7) - scale[root % 7]) % 12
    name = ROMAN[root % 7]
    return name if third == 4 else name.lower()


def degree_to_midi(degree, tonic_midi, mode):
    scale = SCALES[mode]
    octave, idx = divmod(degree, 7)
    return tonic_midi + 12 * octave + scale[idx]


def midi_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def onsets(phrase):
    t, out = 0.0, []
    for _, d in phrase:
        out.append(t)
        t += d
    return out


def note_at(phrase, t):
    """Degree of the note sounding at beat t (within the phrase)."""
    start = 0.0
    for deg, d in phrase:
        if start <= t < start + d:
            return deg
        start += d
    return phrase[-1][0]


def is_valid(phrase):
    return (len(phrase) > 0
            and abs(sum(d for _, d in phrase) - PHRASE_BEATS) < 1e-9
            and all(d in ALLOWED_DURS for _, d in phrase)
            and all(DEG_MIN <= g <= DEG_MAX for g, _ in phrase))


def phrase_to_str(phrase, tonic_midi, mode):
    return " ".join(f"{midi_name(degree_to_midi(g, tonic_midi, mode))}/{d:g}" for g, d in phrase)
