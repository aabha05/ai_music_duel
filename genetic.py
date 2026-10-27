"""
genetic.py - EVOLUTIONARY TECHNIQUE: a genetic algorithm that breeds candidate phrases,
plus HILL CLIMBING (heuristic local search) to polish the best ones.

Why a GA? The space of 2-bar phrases is astronomically large (~10^15+), so the minimax
tree cannot branch over every possible phrase. The GA turns that huge space into a
handful of strong, *diverse* candidate moves -> the branching factor of the game tree.

    chromosome : a phrase  ((degree, duration), ...)
    fitness    : judge score + the musician's personality bias
    selection  : tournament (size 3) + elitism
    crossover  : one-point, cut at a beat both parents share
    mutation   : pitch nudge, transpose, split/merge notes, swap rhythms, invert a fragment
"""
import numpy as np
from music import ALLOWED_DURS, PHRASE_BEATS, DEG_MIN, DEG_MAX, onsets, is_valid


def _clamp(g):
    return int(min(max(g, DEG_MIN), DEG_MAX))


class PhraseGA:
    def __init__(self, judge, mood, mutation_rate=0.9, crossover_rate=0.6):
        self.judge = judge
        self.mood = mood
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    # ------------------------------------------------------------ creation
    def random_phrase(self, rng, start):
        weights = np.array([np.exp(-((1 / d - self.mood.density) / 0.6) ** 2) for d in ALLOWED_DURS])
        weights /= weights.sum()
        t, g, out = 0.0, start, []
        while t < PHRASE_BEATS:
            d = float(rng.choice(ALLOWED_DURS, p=weights))
            d = min(d, PHRASE_BEATS - t)
            if d not in ALLOWED_DURS:
                d = 0.5
            out.append((_clamp(g), d))
            t += d
            step = rng.integers(3, 6) if rng.random() < 0.5 * self.mood.leapiness else rng.integers(0, 3)
            g += int(step) * int(rng.choice([-1, 1]))
            g += int(np.sign(self.mood.register - g)) if rng.random() < 0.3 else 0   # drift toward register
        return tuple(out)

    def from_motif(self, motif, rng, target_start):
        shift = target_start - motif[0][0] + int(rng.integers(-2, 3))
        phrase = tuple((_clamp(g + shift), d) for g, d in motif)
        for _ in range(int(rng.integers(1, 3))):
            phrase = self.mutate(phrase, rng) or phrase
        return phrase

    # ------------------------------------------------------------ operators
    def mutate(self, p, rng):
        p = list(p)
        op = rng.integers(6)
        i = int(rng.integers(len(p)))
        if op == 0:                                                     # nudge one pitch
            p[i] = (_clamp(p[i][0] + int(rng.choice([-2, -1, 1, 2]))), p[i][1])
        elif op == 1:                                                   # transpose
            k = int(rng.choice([-3, -2, -1, 1, 2, 3]))
            p = [(_clamp(g + k), d) for g, d in p]
        elif op == 2 and p[i][1] >= 1.0 and len(p) < 16:                # split a note
            g, d = p[i]
            a = 0.5 if d == 1.5 else d / 2
            p[i:i + 1] = [(g, a), (_clamp(g + int(rng.choice([-1, 1, 2]))), d - a)]
        elif op == 3 and i + 1 < len(p) and p[i][1] + p[i + 1][1] in ALLOWED_DURS:   # merge two notes
            p[i:i + 2] = [(p[i][0], p[i][1] + p[i + 1][1])]
        elif op == 4 and len(p) > 1:                                    # swap two durations
            j = int(rng.integers(len(p)))
            p[i], p[j] = (p[i][0], p[j][1]), (p[j][0], p[i][1])
        elif op == 5 and len(p) > 3:                                    # invert a fragment's contour
            j = min(len(p), i + 4)
            pivot = p[i][0]
            p[i:j] = [(_clamp(2 * pivot - g), d) for g, d in p[i:j]]
        out = tuple(p)
        return out if is_valid(out) else None          # invalid offspring are discarded

    def crossover(self, a, b, rng):
        common = (set(onsets(a)) & set(onsets(b))) - {0.0}
        if not common:
            return a
        cut = float(rng.choice(sorted(common)))
        child = tuple(n for n, t in zip(a, onsets(a)) if t < cut) + \
                tuple(n for n, t in zip(b, onsets(b)) if t >= cut)
        return child if is_valid(child) else a

    # ------------------------------------------------------------ main loop
    def fitness(self, phrase, ctx, player):
        return self.judge.score(phrase, ctx) + player.style_bias(phrase)

    def evolve(self, ctx, player, rng, k=5, pop_size=40, generations=25):
        start = ctx.call[-1][0] if ctx.call else int(round(self.mood.register))
        pop = [self.from_motif(player.motifs[i % len(player.motifs)], rng, start) for i in range(pop_size // 2)]
        pop += [self.random_phrase(rng, start + int(rng.integers(-2, 3))) for _ in range(pop_size - len(pop))]
        pop = [p for p in pop if p is not None and is_valid(p)]
        memo = {}

        def fit(p):                                                    # remember scores within this run
            if p not in memo:
                memo[p] = self.fitness(p, ctx, player)
            return memo[p]

        for _ in range(generations):
            scored = sorted(pop, key=fit, reverse=True)
            nxt = scored[:2]                                           # elitism
            while len(nxt) < pop_size:
                parents = []
                for _ in range(2):                                     # tournament selection
                    contenders = [scored[int(j)] for j in rng.integers(len(scored), size=3)]
                    parents.append(max(contenders, key=fit))
                child = self.crossover(parents[0], parents[1], rng) if rng.random() < self.crossover_rate else parents[0]
                if rng.random() < self.mutation_rate:
                    child = self.mutate(child, rng) or child
                nxt.append(child)
            pop = nxt

        unique = []
        for p in sorted(set(pop), key=fit, reverse=True):
            if all(sum(x != y for x, y in zip(p, q)) + abs(len(p) - len(q)) >= 2 for q in unique):  # keep diverse
                unique.append(p)
            if len(unique) == k:
                break
        return unique

    # ------------------------------------------------------------ hill climbing
    def hill_climb(self, phrase, ctx, player, max_iter=25):
        """Steepest-ascent hill climbing over single-note pitch changes."""
        best, best_f = phrase, self.fitness(phrase, ctx, player)
        for _ in range(max_iter):
            neighbours = [best[:i] + ((_clamp(g + s), d),) + best[i + 1:]
                          for i, (g, d) in enumerate(best) for s in (-2, -1, 1, 2)]
            cand = max(neighbours, key=lambda p: self.fitness(p, ctx, player))
            f = self.fitness(cand, ctx, player)
            if f <= best_f + 1e-9:
                break                                                  # local maximum reached
            best, best_f = cand, f
        return best
