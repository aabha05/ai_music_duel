"""
AI MUSIC DUEL - two AI musicians battle in call-and-response.

    python duel.py --mood calm                     # pick a named mood
    python duel.py --mood spooky --seed 42         # same mood, a brand-new variation
    python duel.py --energy 0.3 --darkness 0.6     # any point on the mood map
    python duel.py --list-moods

Every run saves its files in its own folder and adds the duel to the demo website (web/).
"""
import argparse
import os
import time

import engine
import web_export
from moods import MOODS, settings_for, nearest
from music import KEYS, PROGRESSIONS


def main():
    ap = argparse.ArgumentParser(description="AI Music Duel")
    ap.add_argument("--mood", choices=sorted(MOODS), help="a named mood (see --list-moods)")
    ap.add_argument("--list-moods", action="store_true", help="show the named moods and exit")
    ap.add_argument("--energy", type=float, help="0 = calm, 1 = intense (overrides the mood)")
    ap.add_argument("--darkness", type=float, help="0 = bright, 1 = dark (overrides the mood)")
    ap.add_argument("--seed", type=int, default=7, help="change it for a new variation of the same mood")
    ap.add_argument("--key", default="A", choices=sorted(KEYS))
    ap.add_argument("--progression", choices=sorted(PROGRESSIONS), help="chord progression (overrides the mood)")
    ap.add_argument("--turns", type=int, default=8, help="total phrases (even number)")
    ap.add_argument("--depth", type=int, default=3, help="minimax look-ahead in phrases (2 = faster)")
    ap.add_argument("--branching", type=int, default=5, help="candidate phrases per position")
    ap.add_argument("--out", help="output folder (default: output/<mood>)")
    args = ap.parse_args()

    if args.list_moods:
        print("Named moods on the mood map (energy, darkness):\n")
        for k, m in MOODS.items():
            print(f"  {k:<11} energy {m['energy']:.2f}  darkness {m['darkness']:.2f}   {m['about']}")
        return

    cfg = settings_for(args.mood) if args.mood else {}
    if args.energy is not None:
        cfg["energy"] = args.energy
    if args.darkness is not None:
        cfg["darkness"] = args.darkness
    if args.progression:
        cfg["progression"] = args.progression
    cfg.update(seed=args.seed, key=args.key, turns=args.turns, depth=args.depth, branching=args.branching)

    if args.mood:
        name, label = args.mood, MOODS[args.mood]["name"]
    else:
        e = cfg.get("energy", engine.DEFAULTS["energy"])
        d = cfg.get("darkness", engine.DEFAULTS["darkness"])
        name = f"custom_{int(e * 100)}_{int(d * 100)}"
        label = f"Custom (near {MOODS[nearest(e, d)]['name'].lower()})"
    if args.seed != 7:
        name, label = f"{name}_seed{args.seed}", f"{label}, variation {args.seed}"
    out = args.out or os.path.join("output", name)

    print("=" * 78 + f"\n  AI MUSIC DUEL  -  mood: {label}\n" + "=" * 78)
    result = engine.run_duel(cfg)
    tree = engine.save_outputs(result, out)
    page = web_export.save_run(name, web_export.build_run(result, label), tree)
    print(f"\nSaved to {out}: duel.wav, duel.mid, piano_roll.png, search_tree.png, duel_log.json")
    print(f"Added to the demo website as '{label}'. Open {os.path.relpath(page)} (or run python app.py).")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Done in {time.time() - t0:.1f}s")
