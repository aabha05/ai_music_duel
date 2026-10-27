"""
render.py - turns the duel into things you can hear and see:
    duel.wav          stereo audio (Ember left, Tide right), synthesised with numpy only
    duel.mid          MIDI file (open in any DAW / MuseScore for nicer instruments)
    piano_roll.png    who played what
    search_tree.png   the alpha-beta tree for one move, with pruned branches marked
"""
import struct
import wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COLORS = {"ember": "#E8590C", "tide": "#1098AD", "bass": "#868E96"}
SR = 22050


# ------------------------------------------------------------------ audio
def _tone(freq, seconds, voice, vel):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    if voice == "ember":          # bright, buzzy lead (additive sawtooth, slightly detuned)
        wave_ = sum(np.sin(2 * np.pi * freq * k * t * (1 + 0.002 * (k % 2))) / k for k in range(1, 9))
        a, r = 0.01, 0.08
    elif voice == "tide":         # soft, breathy lead
        wave_ = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t) + 0.1 * np.sin(6 * np.pi * freq * t)
        wave_ *= 1 + 0.03 * np.sin(2 * np.pi * 5.5 * t)          # vibrato-ish shimmer
        a, r = 0.05, 0.25
    else:                         # bass
        wave_ = np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(4 * np.pi * freq * t)
        a, r = 0.01, 0.15
    env = np.ones(n)
    na, nr = min(int(a * SR), n), min(int(r * SR), n)
    env[:na] = np.linspace(0, 1, na)
    env[n - nr:] *= np.linspace(1, 0, nr)
    env *= np.exp(-1.2 * t)
    return vel * wave_ * env


def render_wav(events, tempo, path):
    spb = 60.0 / tempo
    total = max(e["start"] + e["dur"] for e in events) * spb + 1.5
    out = np.zeros((int(total * SR), 2))
    pan = {"ember": (0.85, 0.35), "tide": (0.35, 0.85), "bass": (0.6, 0.6)}
    for e in events:
        freq = 440.0 * 2 ** ((e["midi"] - 69) / 12)
        sig = _tone(freq, e["dur"] * spb + 0.2, e["voice"], e["vel"])
        s = int(e["start"] * spb * SR)
        seg = out[s:s + len(sig)]
        seg += sig[:len(seg), None] * np.array(pan[e["voice"]])
    delay = int(0.375 * spb * SR)                                   # dotted-eighth echo
    for _ in range(3):
        out[delay:] += 0.25 * out[:-delay]
    out /= np.abs(out).max() + 1e-9
    pcm = (out * 0.9 * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ------------------------------------------------------------------ MIDI (written by hand, no libraries)
def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def render_midi(events, tempo, path, tpb=480):
    tracks = []
    setup = {"ember": (0, 81), "tide": (1, 73), "bass": (2, 33)}       # GM: saw lead, flute, finger bass
    for voice, (ch, prog) in setup.items():
        msgs = []
        for e in (e for e in events if e["voice"] == voice):
            on, off = int(e["start"] * tpb), int((e["start"] + e["dur"]) * tpb)
            v = int(40 + 80 * e["vel"])
            msgs += [(on, 1, bytes([0x90 | ch, e["midi"], v])), (off, 0, bytes([0x80 | ch, e["midi"], 0]))]
        msgs.sort(key=lambda m: (m[0], m[1]))
        data = bytearray()
        if voice == "ember":
            data += _vlq(0) + b"\xFF\x51\x03" + int(60_000_000 / tempo).to_bytes(3, "big")
        data += _vlq(0) + bytes([0xC0 | ch, prog])
        last = 0
        for tick, _, msg in msgs:
            data += _vlq(tick - last) + msg
            last = tick
        data += _vlq(0) + b"\xFF\x2F\x00"
        tracks.append(b"MTrk" + struct.pack(">I", len(data)) + bytes(data))
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), tpb) + b"".join(tracks))


# ------------------------------------------------------------------ plots
def plot_piano_roll(events, chord_labels, title, path):
    fig, ax = plt.subplots(figsize=(14, 5))
    mel = [e for e in events if e["voice"] != "bass"]
    for e in mel:
        c = "#F59F00" if e.get("coda") else COLORS[e["voice"]]
        ax.add_patch(Rectangle((e["start"], e["midi"] - 0.4), e["dur"] * 0.95, 0.8, color=c))
    lo, hi = min(e["midi"] for e in mel) - 2, max(e["midi"] for e in mel) + 3
    end = max(e["start"] + e["dur"] for e in events)
    for b in range(0, int(end) + 1, 4):
        ax.axvline(b, color="#ccc", lw=0.6, zorder=0)
    for b in range(0, int(end) + 1, 8):
        ax.axvline(b, color="#888", lw=1.0, zorder=0)
    for beat, label in chord_labels:
        ax.text(beat + 0.1, hi - 0.5, label, fontsize=8, color="#555")
    ax.set_xlim(0, end)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("beat")
    ax.set_ylabel("MIDI pitch")
    ax.set_title(title)
    handles = [Rectangle((0, 0), 1, 1, color=COLORS["ember"]), Rectangle((0, 0), 1, 1, color=COLORS["tide"]),
               Rectangle((0, 0), 1, 1, color="#F59F00")]
    ax.legend(handles, ["Ember (aggressive)", "Tide (lyrical)", "A* cadence"], loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_search_tree(root, chosen, players, title, path):
    pos, leaves = {}, [0]

    def layout(node, depth):
        slots = []
        for c in node.children:
            slots.append(layout(c, depth + 1))
        for _ in range(node.pruned):
            slots.append(leaves[0])
            leaves[0] += 1
        x = (min(slots) + max(slots)) / 2 if slots else leaves[0]
        if not slots:
            leaves[0] += 1
        pos[id(node)] = (x, -depth)
        return x

    layout(root, 0)
    width = max(8, leaves[0] * 0.18)
    fig, ax = plt.subplots(figsize=(min(width, 30), 6))

    def draw(node, depth):
        x, y = pos[id(node)]
        for c in node.children:
            cx, cy = pos[id(c)]
            on_path = c is chosen
            ax.plot([x, cx], [y, cy], color="#F59F00" if on_path else "#bbb", lw=3 if on_path else 0.8, zorder=1)
            draw(c, depth + 1)
        # pruned stubs
        visited_x = [pos[id(c)][0] for c in node.children]
        start = (max(visited_x) if visited_x else x)
        for k in range(node.pruned):
            px = start + (k + 1) * 0.9
            ax.plot([x, px], [y, y - 0.6], color="#E03131", lw=0.9, ls="--", zorder=1)
            ax.scatter([px], [y - 0.6], marker="x", color="#E03131", s=25, zorder=2)
        color = COLORS[players[node.player].voice] if depth > 0 else "#343A40"
        ax.scatter([x], [y], s=40 if depth else 120, color=color, zorder=3, edgecolor="white")
        if node.value is not None and (depth <= 1):
            bound = "\u2264" if node.pruned else ""          # cut-off nodes only know an upper bound
            ax.text(x, y + 0.12, f"{bound}{node.value:+.1f}", ha="center", fontsize=7)

    draw(root, 0)
    ax.set_yticks([0, -1, -2, -3][:max(1, int(-min(p[1] for p in pos.values())) + 1)])
    ax.set_yticklabels(["now", "move", "rival reply", "counter-reply"][:len(ax.get_yticks())])
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(title)
    ax.text(0.01, -0.08, "orange = chosen line   red dashed x = branch pruned by alpha-beta   "
            "numbers = minimax value (Ember's lead)", transform=ax.transAxes, fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
