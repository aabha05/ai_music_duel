"""
engine.py - runs one complete duel from a dictionary of settings.

Both front doors use it:
    duel.py  (command line)         python duel.py --energy 0.3 --depth 2
    app.py   (live demo website)    sliders on the page -> the same settings

The big problem, broken into five sub-problems, each solved by one technique:
    1. mood words  -> exact settings          FUZZY LOGIC            fuzzy_mood.py
    2. what sounds like real music?           NEURAL NETWORK         critic.py (+ judge.py)
    3. invent melody options                  GENETIC ALGORITHM      genetic.py
    4. choose the best one against a rival    MINIMAX + ALPHA-BETA   adversarial.py
    5. end the piece smoothly                 A* SEARCH              cadence.py
"""
import json
import os
import time

import music
from music import KEYS, PHRASE_BEATS, chords_for_turn, chord_name, degree_to_midi, phrase_to_str
import fuzzy_mood
from critic import get_critic
from judge import Judge
from agents import make_musician
from genetic import PhraseGA
from adversarial import DuelSearch, count_tree
from cadence import compare_searches, CODA_BEATS

DEFAULTS = {
    # sub-problem 1: fuzzy logic
    "energy": 0.8, "darkness": 0.7,
    # music
    "key": "A", "turns": 8, "progression": "epic", "p1": "aggressive", "p2": "lyrical", "seed": 7,
    # sub-problem 2: what the judge (and its neural network) cares about
    "w_critic": 0.30, "w_harmony": 0.30, "w_mood": 0.20, "w_coherence": 0.20,
    # sub-problem 3: genetic algorithm
    "population": 60, "generations": 40, "mutation": 0.9, "crossover": 0.6,
    # sub-problem 4: minimax
    "depth": 3, "branching": 5, "pruning": True, "ordering": "best",
    # sub-problem 5: A*
    "astar_weight": 1.0,
}

LIMITS = {   # keeps a live demo from asking for something that would take hours
    "energy": (0, 1), "darkness": (0, 1), "turns": (2, 16), "seed": (0, 2**31 - 1),
    "w_critic": (0, 1), "w_harmony": (0, 1), "w_mood": (0, 1), "w_coherence": (0, 1),
    "population": (10, 120), "generations": (5, 80), "mutation": (0, 1), "crossover": (0, 1),
    "depth": (1, 4), "branching": (2, 8), "astar_weight": (0, 5),
}


def clean_config(cfg):
    c = dict(DEFAULTS)
    for k, v in (cfg or {}).items():
        if k not in DEFAULTS or v is None:
            continue
        default = DEFAULTS[k]
        if isinstance(default, bool):
            v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
        elif isinstance(default, int):
            v = int(float(v))
        elif isinstance(default, float):
            v = float(v)
        else:
            v = str(v)
        if k in LIMITS:
            lo, hi = LIMITS[k]
            v = type(v)(min(max(v, lo), hi))
        c[k] = v
    if c["key"] not in KEYS:
        c["key"] = "A"
    if c["progression"] not in music.PROGRESSIONS:
        c["progression"] = "epic"
    for p in ("p1", "p2"):
        if c[p] not in ("aggressive", "lyrical", "playful"):
            c[p] = DEFAULTS[p]
    if c["ordering"] not in ("best", "random", "worst"):
        c["ordering"] = "best"
    if c["turns"] % 2:
        c["turns"] += 1
    return c


def tonic_for(key):
    k = KEYS[key]
    return 60 + k if k <= 4 else 48 + k            # melody's home note sits between F3 and E4


def run_duel(cfg=None, say=print):
    """Runs a whole duel. `say(text)` receives progress messages. Returns a result dictionary."""
    c = clean_config(cfg)
    t_start = time.time()
    music.set_progression(c["progression"])

    # 1. FUZZY LOGIC: mood -> settings
    mood = fuzzy_mood.infer(c["energy"], c["darkness"])
    say(fuzzy_mood.explain(mood))
    tonic = tonic_for(c["key"])

    # 2. NEURAL NETWORK + JUDGE
    critic = get_critic(verbose=False)
    judge = Judge(critic, mood, weights={"critic": c["w_critic"], "harmony": c["w_harmony"],
                                         "mood": c["w_mood"], "coherence": c["w_coherence"]})
    players = (make_musician("Ember", 0, "ember", c["p1"]), make_musician("Tide", 1, "tide", c["p2"]))

    # 3 + 4. GENETIC ALGORITHM inside MINIMAX
    ga = PhraseGA(judge, mood, mutation_rate=c["mutation"], crossover_rate=c["crossover"])
    search = DuelSearch(ga, judge, players, depth=c["depth"], branching=c["branching"], total_turns=c["turns"],
                        seed=c["seed"], pruning=c["pruning"], ordering=c["ordering"],
                        population=c["population"], generations=c["generations"])

    history, log, totals, first = (), [], [0.0, 0.0], None
    for turn in range(c["turns"]):
        player = players[turn % 2]
        say(f"Turn {turn + 1} of {c['turns']}: {player.name} is thinking...")
        phrase, info = search.best_move(history)
        ctx = search.context(history)
        parts = judge.breakdown(phrase, ctx)
        totals[turn % 2] += parts["total"]
        visited, pruned = count_tree(info["tree"])
        if turn == 0:
            first = info
            mm_nodes, _ = search.compare_with_minimax(history)
            first["minimax_nodes"] = mm_nodes
            first["alphabeta_nodes"] = visited
        chords = chords_for_turn(turn)
        chord_txt = f"{chord_name(chords[0], mood.mode)}-{chord_name(chords[1], mood.mode)}"
        log.append({"turn": turn + 1, "player": player.name, "chords": chord_txt, "phrase": phrase,
                    "score": parts, "nodes": visited, "pruned": pruned, "seconds": round(info["seconds"], 2)})
        say(f"Turn {turn + 1:>2} | {player.name:<5} over {chord_txt:<8} | score {parts['total']:4.1f} "
            f"(critic {parts['critic']:.2f} harm {parts['harmony']:.2f} mood {parts['mood']:.2f} "
            f"coh {parts['coherence']:.2f})\n        {phrase_to_str(phrase, tonic, mood.mode)}\n"
            f"        searched {visited} nodes, pruned {pruned} branches, {info['seconds']:.1f}s")
        history += (phrase,)

    # 5. A* SEARCH: the winner's ending
    winner = players[0] if totals[0] >= totals[1] else players[1]
    coda, astar = compare_searches(history[-1][-1][0], c["astar_weight"])
    say(f"Final score: Ember {totals[0]:.1f} - Tide {totals[1]:.1f}  ->  {winner.name} wins!")
    say(f"Ending for {winner.name}: {phrase_to_str(tuple(coda), tonic, mood.mode)} "
        f"(cost {astar['cost']:.1f}, {astar['expanded']} states expanded; "
        f"uniform-cost search needs {astar['ucs_expanded']})")
    if c["pruning"]:
        say(f"Alpha-beta vs plain minimax on move 1: {first['alphabeta_nodes']} vs {first['minimax_nodes']} nodes "
            f"({100 * (1 - first['alphabeta_nodes'] / first['minimax_nodes']):.0f}% fewer)")

    return {"config": c, "mood": mood, "tonic": tonic, "players": players, "history": history, "log": log,
            "totals": totals, "winner": winner, "coda": coda, "astar": astar, "first": first,
            "evaluations": judge.evaluations, "seconds": time.time() - t_start}


def build_events(r):
    """All notes (melody, bass, ending) as timed events, for audio, MIDI and plots."""
    mood, tonic, players = r["mood"], r["tonic"], r["players"]
    events, chord_labels = [], []
    for turn, phrase in enumerate(r["history"]):
        voice = players[turn % 2].voice
        t = turn * PHRASE_BEATS
        for g, d in phrase:
            events.append({"start": t, "dur": d, "midi": degree_to_midi(g, tonic, mood.mode),
                           "voice": voice, "vel": 0.8 if voice == "ember" else 0.7})
            t += d
        for bar, root in enumerate(chords_for_turn(turn)):
            start = turn * PHRASE_BEATS + 4 * bar
            chord_labels.append((start, chord_name(root, mood.mode)))
            for half in (0, 2):
                events.append({"start": start + half, "dur": 2, "midi": degree_to_midi(root, tonic - 24, mood.mode),
                               "voice": "bass", "vel": 0.55})
    end = len(r["history"]) * PHRASE_BEATS
    t = end
    for g, d in r["coda"]:
        events.append({"start": t, "dur": d, "midi": degree_to_midi(g, tonic, mood.mode),
                       "voice": r["winner"].voice, "vel": 0.85, "coda": True})
        t += d
    chord_labels += [(end, chord_name(4, mood.mode)), (end + 2, chord_name(0, mood.mode))]
    events += [{"start": end, "dur": 2, "midi": degree_to_midi(4, tonic - 24, mood.mode), "voice": "bass", "vel": 0.55},
               {"start": end + 2, "dur": CODA_BEATS, "midi": tonic - 24, "voice": "bass", "vel": 0.6}]
    return events, chord_labels


def save_outputs(r, out_dir, audio=True):
    """Writes duel.wav, duel.mid, piano_roll.png, search_tree.png and duel_log.json into out_dir."""
    import render
    os.makedirs(out_dir, exist_ok=True)
    c, mood, totals = r["config"], r["mood"], r["totals"]
    events, chord_labels = build_events(r)
    if audio:
        render.render_wav(events, mood.tempo, os.path.join(out_dir, "duel.wav"))
        render.render_midi(events, mood.tempo, os.path.join(out_dir, "duel.mid"))
    title = (f"Ember {totals[0]:.1f} vs Tide {totals[1]:.1f} | {c['key']} {mood.mode}, {mood.tempo:.0f} BPM "
             f"| energy {c['energy']}, darkness {c['darkness']}")
    render.plot_piano_roll(events, chord_labels, title, os.path.join(out_dir, "piano_roll.png"))
    render.plot_search_tree(r["first"]["tree"], r["first"]["chosen"], r["players"],
                            f"Ember's first move: {'alpha-beta' if c['pruning'] else 'plain minimax'} "
                            f"(depth {c['depth']}, {c['branching']} options per move, {c['ordering']}-first ordering)",
                            os.path.join(out_dir, "search_tree.png"))
    with open(os.path.join(out_dir, "duel_log.json"), "w") as f:
        json.dump({"config": c, "mood": {k: v for k, v in vars(mood).items() if k != "firing"},
                   "turns": r["log"], "totals": totals, "winner": r["winner"].name, "coda": r["coda"],
                   "astar": r["astar"], "alphabeta_nodes": r["first"]["alphabeta_nodes"],
                   "minimax_nodes": r["first"]["minimax_nodes"]}, f, indent=1)
    return os.path.join(out_dir, "search_tree.png")
