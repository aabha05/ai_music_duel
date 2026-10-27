"""
web_export.py - turns a duel result into data the demo website can replay.

    web/index.html        the demo page (never changes)
    web/data.js           your saved duels                        <- written here
    web/trees/<name>.png  the search tree picture for each duel   <- copied here
"""
import json
import os
import shutil

from music import PHRASE_BEATS, chords_for_turn, degree_to_midi, midi_name
from agents import PERSONALITY_MOTIFS
from moods import MOODS

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
DATA_FILE = os.path.join(WEB_DIR, "data.js")
PREFIX = "window.DUEL_DATA = "


def _notes(phrase, start, tonic, mode):
    out, t = [], start
    for g, d in phrase:
        m = degree_to_midi(g, tonic, mode)
        out.append([t, d, m, midi_name(m)])
        t += d
    return out


def motifs():
    return {p: _notes(ms[0], 0, 57, "minor") for p, ms in PERSONALITY_MOTIFS.items()}


def mood_list():
    return [{"id": k, "name": m["name"], "energy": m["energy"], "darkness": m["darkness"], "about": m["about"]}
            for k, m in MOODS.items()]


def build_run(r, label="Custom mood"):
    """Everything the page needs about one duel, as plain JSON-friendly data."""
    c, mood, tonic = r["config"], r["mood"], r["tonic"]
    turns = []
    for i, (phrase, entry) in enumerate(zip(r["history"], r["log"])):
        s = entry["score"]
        turns.append({
            "p": entry["player"], "chords": entry["chords"],
            "notes": _notes(phrase, i * PHRASE_BEATS, tonic, mood.mode),
            "bass": [degree_to_midi(root, tonic - 24, mood.mode) for root in chords_for_turn(i)],
            "score": round(s["total"], 2), "critic": round(s["critic"], 2), "harm": round(s["harmony"], 2),
            "mood": round(s["mood"], 2), "coh": round(s["coherence"], 2),
            "nodes": entry["nodes"], "pruned": entry["pruned"],
        })
    rules = [{"if": " and ".join(f"{v} is {t}" for v, t in ants),
              "then": ", ".join(f"{v} is {t}" for v, t in cons), "s": round(st, 2)}
             for ants, cons, st in mood.firing]
    return {
        "label": label,
        "energy": c["energy"], "darkness": c["darkness"], "key": c["key"], "seed": c["seed"],
        "progression": c["progression"],
        "players": [{"name": p.name, "personality": p.personality} for p in r["players"]],
        "mode": mood.mode, "tempo": round(mood.tempo), "density": round(mood.density, 2),
        "leap": round(mood.leapiness, 2), "register": round(mood.register, 1),
        "rules": rules, "turns": turns, "totals": [round(x, 1) for x in r["totals"]], "winner": r["winner"].name,
        "coda": _notes(r["coda"], len(r["history"]) * PHRASE_BEATS, tonic, mood.mode),
        "codaBass": [degree_to_midi(4, tonic - 24, mood.mode), tonic - 24],
        "astar": {"expanded": r["astar"]["expanded"], "cost": round(r["astar"]["cost"], 2),
                  "ucs_expanded": r["astar"]["ucs_expanded"]},
        "ab": r["first"]["alphabeta_nodes"], "mm": r["first"]["minimax_nodes"],
        "seconds": round(r["seconds"], 1),
    }


def load():
    if os.path.exists(DATA_FILE):
        text = open(DATA_FILE, encoding="utf-8").read().strip()
        if text.startswith(PREFIX):
            return json.loads(text[len(PREFIX):].rstrip(";"))
    return {"runs": {}, "latest": None}


def save_run(name, run, tree_png):
    """Adds (or replaces) a duel in web/data.js and copies its search-tree picture."""
    data = load()
    data["runs"][name] = run
    data["latest"] = name
    data["motifs"] = motifs()
    data["moods"] = mood_list()
    os.makedirs(os.path.join(WEB_DIR, "trees"), exist_ok=True)
    shutil.copyfile(tree_png, os.path.join(WEB_DIR, "trees", f"{name}.png"))
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        f.write(PREFIX + json.dumps(data, separators=(",", ":")) + ";\n")
    return os.path.join(WEB_DIR, "index.html")
