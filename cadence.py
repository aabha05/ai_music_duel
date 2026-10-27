"""
cadence.py - HEURISTIC SEARCH: A* plans the winner's final cadence (the "coda").

Problem: from the last note of the duel, reach the tonic in exactly CODA_BEATS beats,
over a V -> I chord change, with the smoothest possible voice-leading.

    state     : (current degree, beats used)
    actions   : play next note = (degree within +-4, duration in {0.5, 1, 2})
    step cost : |interval| + 1.5 * (extra size of any leap beyond a 3rd)
                + 0.5 if the note is not in the current chord + 0.3 per note
    goal      : beats used == CODA_BEATS and last note == tonic, held >= 1 beat
    h(n)      : |degree - tonic|

h is admissible AND consistent: every step costs at least |interval|, and by the
triangle inequality |a - t| <= |a - b| + |b - t|. So A* returns the optimal cadence.
"""
import heapq
import itertools
from music import triad

CODA_BEATS = 4.0
CODA_CHORDS = ((0.0, 4), (2.0, 0))      # (from beat, chord root): V then I


def chord_at(t):
    root = CODA_CHORDS[0][1]
    for start, r in CODA_CHORDS:
        if t >= start:
            root = r
    return root


def plan_cadence(start_degree, weight=1.0, durations=(0.5, 1.0, 2.0)):
    """weight = 0 -> uniform-cost search (no heuristic)
       weight = 1 -> A* (optimal, because h is admissible)
       weight > 1 -> weighted A*: faster, but h now overestimates, so the answer may not be optimal"""
    target = 7 * round(start_degree / 7)                     # nearest tonic
    h = lambda g: weight * abs(g - target)
    tie = itertools.count()
    start = (start_degree, 0.0)
    frontier = [(h(start_degree), next(tie), 0.0, start, [])]
    best_g = {start: 0.0}
    expanded = 0

    while frontier:
        f, _, g_cost, (deg, used), path = heapq.heappop(frontier)
        if used == CODA_BEATS and deg == target and path:
            return path, {"expanded": expanded, "cost": g_cost, "target": target}
        if g_cost > best_g.get((deg, used), float("inf")):
            continue
        expanded += 1
        for d in durations:
            nu = used + d
            if nu > CODA_BEATS:
                continue
            for nxt in range(deg - 4, deg + 5):
                if nu == CODA_BEATS and (nxt != target or d < 1.0):
                    continue                                      # must END on a held tonic
                step = abs(nxt - deg)
                cost = step + 1.5 * max(0, step - 2) + (0.5 if nxt % 7 not in triad(chord_at(used)) else 0) + 0.3
                ng = g_cost + cost
                state = (nxt, nu)
                if ng < best_g.get(state, float("inf")):
                    best_g[state] = ng
                    heapq.heappush(frontier, (ng + h(nxt), next(tie), ng, state, path + [(nxt, d)]))
    return [(target, CODA_BEATS)], {"expanded": expanded, "cost": float("nan"), "target": target}


def compare_searches(start_degree, weight):
    """Run the chosen search plus the two reference points, for the report and the demo."""
    path, info = plan_cadence(start_degree, weight)
    _, ucs = plan_cadence(start_degree, 0.0)
    _, astar = plan_cadence(start_degree, 1.0)
    info.update(weight=weight, ucs_expanded=ucs["expanded"], astar_expanded=astar["expanded"],
                optimal_cost=astar["cost"])
    return path, info
