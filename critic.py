"""
critic.py - SOFT COMPUTING: a multilayer perceptron trained with back-propagation (from scratch).

Job: given a phrase, output P(phrase sounds like a real human melody).

Training data:
  positives = 2-bar windows from public-domain melodies (below)
  negatives = random note sequences, shuffled real melodies, and "jittered" real melodies
So the network learns what natural melodic motion looks like, it is not just copying our rules.
(This generator-vs-critic idea is the same one behind GANs from the Emerging Trends unit.)

Upgrade path: replace CORPUS with windows extracted from a MIDI dataset (e.g. Lakh MIDI).
"""
import os
import numpy as np
from music import ALLOWED_DURS

# Public-domain melodies as (scale degree, beats). Degree 0 = tonic.
CORPUS = {
    "ode_to_joy": [(2,1),(2,1),(3,1),(4,1),(4,1),(3,1),(2,1),(1,1),(0,1),(0,1),(1,1),(2,1),(2,1.5),(1,.5),(1,2),
                   (2,1),(2,1),(3,1),(4,1),(4,1),(3,1),(2,1),(1,1),(0,1),(0,1),(1,1),(2,1),(1,1.5),(0,.5),(0,2)],
    "twinkle": [(0,1),(0,1),(4,1),(4,1),(5,1),(5,1),(4,2),(3,1),(3,1),(2,1),(2,1),(1,1),(1,1),(0,2),
                (4,1),(4,1),(3,1),(3,1),(2,1),(2,1),(1,2),(4,1),(4,1),(3,1),(3,1),(2,1),(2,1),(1,2)],
    "frere_jacques": [(0,1),(1,1),(2,1),(0,1),(0,1),(1,1),(2,1),(0,1),(2,1),(3,1),(4,2),(2,1),(3,1),(4,2),
                      (4,.5),(5,.5),(4,.5),(3,.5),(2,1),(0,1),(4,.5),(5,.5),(4,.5),(3,.5),(2,1),(0,1),
                      (0,1),(-3,1),(0,2),(0,1),(-3,1),(0,2)],
    "mary_lamb": [(2,1),(1,1),(0,1),(1,1),(2,1),(2,1),(2,2),(1,1),(1,1),(1,2),(2,1),(4,1),(4,2),
                  (2,1),(1,1),(0,1),(1,1),(2,1),(2,1),(2,1),(2,1),(1,1),(1,1),(2,1),(1,1),(0,4)],
    "london_bridge": [(4,1.5),(5,.5),(4,1),(3,1),(2,1),(3,1),(4,2),(1,1),(2,1),(3,2),(2,1),(3,1),(4,2),
                      (4,1.5),(5,.5),(4,1),(3,1),(2,1),(3,1),(4,2),(1,2),(4,2),(2,1),(0,3)],
    "jingle_bells": [(2,1),(2,1),(2,2),(2,1),(2,1),(2,2),(2,1),(4,1),(0,1.5),(1,.5),(2,4),
                     (3,1),(3,1),(3,1.5),(3,.5),(3,1),(2,1),(2,1),(2,.5),(2,.5),(2,1),(1,1),(1,1),(2,1),(1,2),(4,2)],
    "amazing_grace": [(-3,1),(0,2),(2,.5),(0,.5),(2,2),(1,1),(0,2),(-2,1),(-3,2),(-3,1),(0,2),(2,.5),(0,.5),
                      (2,2),(1,1),(4,3),(2,1),(4,1.5),(2,.5),(4,.5),(2,.5),(0,2),(-3,1),(-2,1.5),(0,.5),(0,.5),(-2,.5),(-3,2)],
    "greensleeves": [(0,1),(2,2),(3,1),(4,1.5),(5,.5),(4,1),(3,2),(1,1),(-1,1.5),(0,.5),(1,1),(2,2),(0,1),
                     (0,1.5),(-1,.5),(0,1),(1,2),(-1,1),(-3,2),(0,1),(2,2),(3,1),(4,1.5),(5,.5),(4,1),(3,2),(1,1)],
    "scarborough": [(0,2),(0,1),(4,2),(4,1),(1,1.5),(2,.5),(1,1),(0,3),(4,1),(7,2),(8,1),(7,2),(4,1),
                    (5,1),(3,1),(4,1),(2,3),(4,3)],
    "bach_minuet": [(4,1),(0,.5),(1,.5),(2,.5),(3,.5),(4,1),(0,1),(0,1),(5,1),(3,.5),(4,.5),(5,.5),(6,.5),
                    (7,1),(0,1),(0,1),(3,1),(4,.5),(3,.5),(2,.5),(1,.5),(2,1),(3,.5),(2,.5),(1,.5),(0,.5),
                    (-1,1),(0,.5),(1,.5),(2,.5),(0,.5),(1,3)],
    "auld_lang_syne": [(-3,1),(0,1.5),(0,.5),(0,1),(2,1),(1,1.5),(0,.5),(1,1),(2,.5),(1,.5),(0,1.5),(0,.5),
                       (2,1),(4,1),(5,3),(5,1),(4,1.5),(2,.5),(2,1),(0,1),(1,1.5),(0,.5),(1,1),(2,.5),(1,.5),(0,1.5),(-2,.5),(-2,1),(-3,1),(0,3)],
}

N_FEATURES = 20


def melodic_features(phrase):
    degs = np.array([n[0] for n in phrase], dtype=float)
    durs = np.clip(np.array([n[1] for n in phrase], dtype=float), 0.5, 2.0)
    iv = np.diff(degs) if len(degs) > 1 else np.zeros(1)
    a = np.abs(iv)

    interval_bins = [np.mean(a == 0), np.mean(a == 1), np.mean(a == 2),
                     np.mean(a == 3), np.mean(a == 4), np.mean(a >= 5)]
    up_ratio = np.mean(iv > 0)
    signs = np.sign(iv[iv != 0])
    contour_changes = np.mean(signs[1:] != signs[:-1]) if len(signs) > 1 else 0.0

    big = a >= 3
    if len(iv) > 1:
        same_dir_leaps = np.mean(big[1:] & big[:-1] & (np.sign(iv[1:]) == np.sign(iv[:-1])))
        leaps = np.where(big[:-1])[0]
        recovered = [a[i + 1] <= 2 and np.sign(iv[i + 1]) == -np.sign(iv[i]) for i in leaps]
        leap_recovery = np.mean(recovered) if recovered else 1.0
    else:
        same_dir_leaps, leap_recovery = 0.0, 1.0

    dur_hist = [np.mean(np.isclose(durs, d)) for d in ALLOWED_DURS]
    return np.array(interval_bins + [
        up_ratio, contour_changes, same_dir_leaps, leap_recovery,
        (degs.max() - degs.min()) / 7, a.mean() / 4, len(degs) / 16, a.max() / 7,
        float(durs[-1] >= 1.0), float(degs[-1] % 7 in (0, 2, 4)),
    ] + dur_hist, dtype=float)


# ---------------------------------------------------------------- dataset
def _windows(melody, beats=8.0):
    out = []
    for i in range(len(melody)):
        t, w = 0.0, []
        for note in melody[i:]:
            if t >= beats:
                break
            w.append(note)
            t += note[1]
        if 6.0 <= t <= 10.0 and len(w) >= 3:
            out.append(tuple(w))
    return out


def _random_phrase(rng):
    t, w, g = 0.0, [], int(rng.integers(-3, 8))
    while t < 8:
        d = float(rng.choice(ALLOWED_DURS))
        d = min(d, 8 - t)
        g = int(np.clip(g + rng.integers(-7, 8), -7, 14))
        w.append((g, d))
        t += d
    return tuple(w)


def build_dataset(seed=0):
    rng = np.random.default_rng(seed)
    pos = [w for mel in CORPUS.values() for w in _windows(mel)]
    neg = []
    for w in pos:
        neg.append(_random_phrase(rng))
        degs = [g for g, _ in w]
        rng.shuffle(degs)
        neg.append(tuple((g, d) for g, (_, d) in zip(degs, w)))
        neg.append(tuple((int(np.clip(g + (rng.choice([-1, 1]) * rng.integers(3, 7) if rng.random() < 0.45 else 0), -7, 14)), d)
                         for g, d in w))
    X = np.array([melodic_features(p) for p in pos + neg])
    y = np.array([1.0] * len(pos) + [0.0] * len(neg))
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ---------------------------------------------------------------- MLP from scratch
class MLP:
    def __init__(self, sizes=(N_FEATURES, 16, 8, 1), seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.mu = np.zeros(sizes[0])
        self.sd = np.ones(sizes[0])

    def _forward(self, X):
        acts = [(X - self.mu) / self.sd]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(1 / (1 + np.exp(-z)) if i == len(self.W) - 1 else np.maximum(z, 0))   # ReLU / sigmoid
        return acts

    def predict(self, X):
        return self._forward(np.atleast_2d(X))[-1][:, 0]

    def train(self, X, y, epochs=250, lr=0.05, batch=32, l2=1e-3, smoothing=0.1, seed=0):
        rng = np.random.default_rng(seed)
        y = y * (1 - smoothing) + smoothing / 2          # label smoothing: keeps the critic from being overconfident
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        for _ in range(epochs):
            order = rng.permutation(len(y))
            for s in range(0, len(y), batch):
                bi = order[s:s + batch]
                acts = self._forward(X[bi])
                delta = (acts[-1][:, 0] - y[bi])[:, None] / len(bi)      # dLoss/dz for sigmoid + BCE
                for i in reversed(range(len(self.W))):                   # back-propagation
                    gW = acts[i].T @ delta + l2 * self.W[i]
                    gb = delta.sum(0)
                    if i > 0:
                        delta = (delta @ self.W[i].T) * (acts[i] > 0)    # ReLU derivative
                    self.W[i] -= lr * gW
                    self.b[i] -= lr * gb

    def save(self, path):
        np.savez(path, mu=self.mu, sd=self.sd, **{f"W{i}": W for i, W in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    def load(self, path):
        d = np.load(path)
        self.mu, self.sd = d["mu"], d["sd"]
        self.W = [d[f"W{i}"] for i in range(len(self.W))]
        self.b = [d[f"b{i}"] for i in range(len(self.b))]


def get_critic(path=None, verbose=True):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "critic_weights.npz")
    net = MLP()
    if os.path.exists(path):
        net.load(path)
        if verbose:
            print("Critic: loaded trained weights from", os.path.basename(path))
        return net
    X, y = build_dataset()
    split = int(0.8 * len(y))
    net.train(X[:split], y[:split])
    acc_tr = np.mean((net.predict(X[:split]) > 0.5) == y[:split])
    acc_va = np.mean((net.predict(X[split:]) > 0.5) == y[split:])
    net.save(path)
    if verbose:
        print(f"Critic: trained MLP {N_FEATURES}-16-8-1 on {len(y)} phrases "
              f"(train acc {acc_tr:.1%}, validation acc {acc_va:.1%}) -> saved {os.path.basename(path)}")
    return net
