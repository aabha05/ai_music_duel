"""
adversarial.py - GAME PLAYING: minimax with alpha-beta pruning.

Game formulation
    state      : history of phrases played so far (+ whose turn it is)
    moves      : candidate phrases produced by the GA for that state
    players    : Ember = MAX, Tide = MIN
    utility    : sum(judge scores of Ember's phrases) - sum(judge scores of Tide's phrases)
                 accumulated along the path (a zero-sum duel)
    depth      : how many phrases ahead the agent imagines (default 3 = me, you, me)

Alpha-beta never changes the chosen move - it only skips branches that cannot affect it.
`compare_with_minimax()` proves this by running plain minimax on the same tree.
"""
import math
import time
from dataclasses import dataclass, field
import numpy as np
from music import chords_for_turn
from judge import Context


@dataclass
class TreeNode:
    player: int
    phrase: tuple = None
    score: float = 0.0
    value: float = None
    children: list = field(default_factory=list)
    pruned: int = 0                # children never explored because of a cutoff


class DuelSearch:
    def __init__(self, ga, judge, players, depth=3, branching=5, total_turns=8, seed=0,
                 pruning=True, ordering="best", population=60, generations=40):
        self.ga, self.judge, self.players = ga, judge, players
        self.depth, self.branching, self.total_turns, self.seed = depth, branching, total_turns, seed
        self.pruning = pruning          # False = plain minimax (every branch examined)
        self.ordering = ordering        # "best" / "random" / "worst": the order moves are examined in
        self.population, self.generations = population, generations
        self._cand_cache = {}
        self.nodes = 0

    def context(self, history):
        turn = len(history)
        return Context(turn=turn, chords=chords_for_turn(turn), call=history[-1] if history else None)

    def candidates(self, history, root=False):
        """GA-generated moves, sorted best-first (good ordering = more pruning). Deterministic per state."""
        key = (history[-1] if history else None, len(history), root)
        if key not in self._cand_cache:
            ctx = self.context(history)
            player = self.players[len(history) % 2]
            rng = np.random.default_rng(abs(hash((key[0], key[1], self.seed))) % (2 ** 32))
            if root:
                moves = self.ga.evolve(ctx, player, rng, k=self.branching,
                                       pop_size=self.population, generations=self.generations)
                moves = [self.ga.hill_climb(m, ctx, player) for m in moves]
                moves = list(dict.fromkeys(moves))
            else:
                moves = self.ga.evolve(ctx, player, rng, k=self.branching,
                                       pop_size=max(12, self.population * 2 // 5), generations=max(4, self.generations // 5))
            scored = sorted(((m, self.judge.score(m, ctx)) for m in moves), key=lambda x: -x[1])
            self._cand_cache[key] = scored
        return self._cand_cache[key]

    def ordered(self, moves, history):
        """Move ordering does not change the answer, only how much alpha-beta can prune.
        Candidates are cached sorted by the mover's own judge score, which is good for the mover
        whether it is MAX (adds the score) or MIN (subtracts it), so that order is best-first."""
        if self.ordering == "random":
            rng = np.random.default_rng(len(history) * 7919 + self.seed)
            return [moves[i] for i in rng.permutation(len(moves))]
        return list(moves) if self.ordering == "best" else list(moves)[::-1]

    # ------------------------------------------------------------ alpha-beta
    def _alphabeta(self, history, depth, alpha, beta, acc, node, root=False):
        self.nodes += 1
        if depth == 0 or len(history) >= self.total_turns:
            node.value = acc
            return acc
        maximizing = len(history) % 2 == 0
        moves = self.ordered(self.candidates(history, root=root), history)
        best = -math.inf if maximizing else math.inf
        for i, (phrase, sc) in enumerate(moves):
            child = TreeNode(player=len(history) % 2, phrase=phrase, score=sc)
            node.children.append(child)
            v = self._alphabeta(history + (phrase,), depth - 1, alpha, beta,
                                acc + sc if maximizing else acc - sc, child)
            child.value = v
            if maximizing:
                best, alpha = max(best, v), max(alpha, v)
            else:
                best, beta = min(best, v), min(beta, v)
            if self.pruning and alpha >= beta:      # CUTOFF: rival would never allow this line
                node.pruned = len(moves) - i - 1
                break
        node.value = best
        return best

    def _minimax(self, history, depth, acc, root=False):
        self.nodes += 1
        if depth == 0 or len(history) >= self.total_turns:
            return acc
        maximizing = len(history) % 2 == 0
        vals = [self._minimax(history + (p,), depth - 1, acc + s if maximizing else acc - s)
                for p, s in self.candidates(history, root=root)]
        return max(vals) if maximizing else min(vals)

    def best_move(self, history):
        """Returns (phrase, info) for the player to move."""
        t0 = time.time()
        self.nodes = 0
        root = TreeNode(player=1 - len(history) % 2)
        self._alphabeta(history, self.depth, -math.inf, math.inf, 0.0, root, root=True)
        maximizing = len(history) % 2 == 0
        pick = (max if maximizing else min)(root.children, key=lambda c: c.value)
        info = {"nodes": self.nodes, "seconds": time.time() - t0, "value": pick.value,
                "tree": root, "chosen": pick, "candidates": len(root.children)}
        return pick.phrase, info

    def compare_with_minimax(self, history):
        self.nodes = 0
        value = self._minimax(history, self.depth, 0.0, root=True)
        return self.nodes, value


def count_tree(node):
    """(visited nodes, pruned branches) in a recorded tree."""
    v, p = 1, node.pruned
    for c in node.children:
        cv, cp = count_tree(c)
        v, p = v + cv, p + cp
    return v, p
