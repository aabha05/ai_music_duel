"""
moods.py - named moods, placed on a two-axis MOOD MAP.

Psychology describes emotions with two axes (Russell's circumplex model of affect):
    energy   (arousal): calm  0 ......... 1 intense
    darkness (valence): bright 0 ......... 1 dark
Every named mood is just a point on that map. The fuzzy controller only ever sees the two
numbers, so any point in between - a mood with no name - works too.

A mood preset also picks things fuzzy logic does not decide: the chord progression and
the two musicians' personalities.
"""

MOODS = {
    "calm":       {"name": "Calm",       "energy": 0.15, "darkness": 0.30, "progression": "dreamy",
                   "p1": "lyrical", "p2": "lyrical", "about": "slow, gentle, unhurried"},
    "dreamy":     {"name": "Dreamy",     "energy": 0.20, "darkness": 0.10, "progression": "dreamy",
                   "p1": "playful", "p2": "lyrical", "about": "floating, bright and soft"},
    "romantic":   {"name": "Romantic",   "energy": 0.35, "darkness": 0.35, "progression": "jazz",
                   "p1": "lyrical", "p2": "lyrical", "about": "warm and tender"},
    "happy":      {"name": "Happy",      "energy": 0.60, "darkness": 0.05, "progression": "pop",
                   "p1": "playful", "p2": "lyrical", "about": "sunny and bouncy"},
    "playful":    {"name": "Playful",    "energy": 0.70, "darkness": 0.25, "progression": "blues",
                   "p1": "playful", "p2": "playful", "about": "cheeky, skipping rhythms"},
    "energetic":  {"name": "Energetic",  "energy": 0.90, "darkness": 0.30, "progression": "pop",
                   "p1": "aggressive", "p2": "playful", "about": "fast and full of drive"},
    "epic":       {"name": "Epic",       "energy": 0.80, "darkness": 0.55, "progression": "epic",
                   "p1": "aggressive", "p2": "lyrical", "about": "big, heroic, cinematic"},
    "tense":      {"name": "Tense",      "energy": 0.70, "darkness": 0.75, "progression": "epic",
                   "p1": "aggressive", "p2": "lyrical", "about": "on edge, something is coming"},
    "angry":      {"name": "Angry",      "energy": 0.95, "darkness": 0.90, "progression": "epic",
                   "p1": "aggressive", "p2": "aggressive", "about": "furious and jagged"},
    "mysterious": {"name": "Mysterious", "energy": 0.40, "darkness": 0.70, "progression": "jazz",
                   "p1": "playful", "p2": "lyrical", "about": "curious, shadowy"},
    "spooky":     {"name": "Spooky",     "energy": 0.30, "darkness": 0.95, "progression": "epic",
                   "p1": "playful", "p2": "lyrical", "about": "haunted and eerie"},
    "sad":        {"name": "Sad",        "energy": 0.15, "darkness": 0.75, "progression": "epic",
                   "p1": "lyrical", "p2": "lyrical", "about": "slow, heavy, longing"},
}


def settings_for(mood_id):
    """Engine settings for a named mood."""
    m = MOODS[mood_id]
    return {k: m[k] for k in ("energy", "darkness", "progression", "p1", "p2")}


def nearest(energy, darkness):
    """Name of the closest named mood on the map (used to label custom points)."""
    return min(MOODS, key=lambda k: (MOODS[k]["energy"] - energy) ** 2 + (MOODS[k]["darkness"] - darkness) ** 2)
