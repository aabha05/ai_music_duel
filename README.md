---
title: AI Music Duel
emoji: 🎹
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AI Music Duel

**Live demo:** https://ai-music-duel.onrender.com  |  **Code:** https://github.com/aabha05/ai_music_duel

Two AI musicians, **Ember** (aggressive) and **Tide** (lyrical), battle in call-and-response.
Each turn, one plays a 2-bar phrase and the other must answer it. Each one thinks
ahead with minimax to play phrases its rival will struggle to answer well.

Everything (fuzzy system, neural network, genetic algorithm, search, synthesiser, MIDI writer)
is written from scratch in Python. The only libraries are NumPy and Matplotlib.

## Run it

```bash
pip install -r requirements.txt

python app.py                          # the live demo: opens the website, pick any mood, press Compose
python duel.py --list-moods            # the 12 named moods
python duel.py --mood spooky           # compose from the terminal
python duel.py --mood spooky --seed 42 # same mood, a brand-new variation
python duel.py --energy 0.3 --darkness 0.6   # any point on the mood map
```

### The mood map

Moods sit on two axes, energy (calm to intense) and darkness (bright to dark), the
two-axis model psychologists use for emotions. `moods.py` places 12 named moods on it:
calm, dreamy, romantic, happy, playful, energetic, epic, tense, angry, mysterious,
spooky and sad. Fuzzy logic only ever sees the two numbers, so any point between them
works too. Combined with the variation number (`--seed`), that gives millions of different duels.

### Files each run produces

Terminal runs save into `output/<mood>/`:

| File | What it is |
|------|------------|
| `duel.wav` | The duel as audio. Ember is panned left, Tide right. |
| `duel.mid` | The same duel as MIDI. Open it in MuseScore or a DAW for real instruments. |
| `piano_roll.png` | Who played what, over the chord progression. |
| `search_tree.png` | The alpha-beta tree for Ember's first move, with pruned branches marked. |
| `duel_log.json` | Every move, score breakdown, and search statistic, for your report. |

### Putting it online

- **Live demo:** https://ai-music-duel.onrender.com (hosted on Render's free tier)
- **Source code:** https://github.com/aabha05/ai_music_duel

Render settings: language Python 3, build command `pip install -r requirements.txt`,
start command `python app.py`, environment variables `HOST=0.0.0.0` and `LIGHT_MODE=1`.
Light mode looks 2 phrases ahead with a smaller genetic algorithm (population 30,
15 generations) so each duel stays quick on the small free server. The free server
sleeps after 15 minutes without visitors and takes about a minute to wake.

Every run is also added to the demo website (`web/data.js` and `web/trees/`). The website
works on its own too (open `web/index.html`, or host the `web` folder on Vercel); it then
plays the saved duels, and composing new ones needs `python app.py` running.

## How one turn works

```
 energy, darkness ──► FUZZY CONTROLLER ──► tempo, density, leapiness, register, mode
                                                    │
 rival's last phrase + chords ─────────────────────►│
                                                    ▼
                        GENETIC ALGORITHM breeds candidate phrases
                                                    │
                        HILL CLIMBING polishes the top ones
                                                    ▼
                  MINIMAX + ALPHA-BETA looks 3 phrases ahead
          (leaf utility = JUDGE: music-theory heuristics + NEURAL CRITIC)
                                                    ▼
                                         phrase is played
 after the last turn:  A* SEARCH plans the winner's cadence home to the tonic
```

## Syllabus mapping (CSE3121)

| Syllabus unit | Where | What it does |
|---|---|---|
| Intelligent agents, PEAS (CO1) | `agents.py` | Two utility-based agents in a multi-agent, sequential, deterministic, fully observable environment. |
| Problem formulation, state space (CO2) | `music.py`, `adversarial.py` | A state is the phrase history, a move is a phrase, and the utility is a score difference. |
| Heuristic search: A\* (CO2) | `cadence.py` | Finds the optimal smooth cadence. The heuristic is admissible and consistent. |
| Heuristic search: hill climbing (CO2) | `genetic.py` | Steepest-ascent polishing that stops at a local maximum. |
| Game playing: minimax, alpha-beta (CO2) | `adversarial.py` | Chooses each move. Plain minimax is also run to prove pruning gives the same answer. |
| Neural networks: perceptron, back-propagation (CO4) | `critic.py` | A 20-16-8-1 MLP with manual backprop, trained on real melodies versus fakes. |
| Fuzzy sets, rules, inference (CO4) | `fuzzy_mood.py` | Mamdani inference: triangular membership functions, min/max operations, centroid defuzzification. |
| Genetic algorithms (CO5) | `genetic.py` | Tournament selection, elitism, crossover, six mutation operators. |
| Emerging trends: GANs (CO5) | `genetic.py` + `critic.py` | The GA is the generator and the MLP is the discriminator, the same idea behind GANs. |
| Explainable AI (CO5) | Console output, `search_tree.png` | Shows rule firing strengths, score breakdowns, and the visible search tree. |

## Design decisions worth explaining in the viva

**Why a genetic algorithm inside minimax?** There are far too many possible 2-bar phrases
(about 10^15) to branch over. The GA reduces that space to 5 strong, diverse candidate moves,
which become the branching factor of the game tree.

**Why is it really a game?** The judge rewards a reply for starting near where the call ended,
echoing the call's rhythm without copying it, and landing on chord tones. So the phrase I play
sets the conditions for my rival's reply. An awkward ending or a strange rhythm makes their
good replies scarcer. Minimax captures exactly this.

**Why is the alpha-beta comparison important?** The console prints, for example,
`68 vs 156 nodes (56% fewer), same decision: True`. That shows pruning saved work without
changing the decision. Ordering moves best-first, from the GA's ranking, makes the cutoffs
happen earlier.

**Why is A\* optimal here?** Each step costs at least |interval|, and h = |degree − tonic|.
By the triangle inequality, h never overestimates and is consistent, so the first goal state
A\* pops is the cheapest cadence.

**Why does fuzzy logic fit the mood input?** "Fairly dark, quite energetic" is vague human
input. Fuzzy rules blend several partially true rules smoothly instead of forcing hard thresholds.

**How was the critic trained?** It sees 260 real 2-bar windows from 11 public-domain melodies
as positives. The negatives are random sequences, the same melodies with their notes shuffled,
and the melodies with random jumps added. It reaches about 83% validation accuracy, and label
smoothing stops it from becoming overconfident.

## Known limitations (good for the report)

- **The GA can fool the critic.** Critic scores often reach 0.99, because a GA optimising
  against a fixed network finds inputs it rates highly. This is an adversarial example. A real
  GAN fixes it by retraining the discriminator on the generator's output.
- **The corpus is small.** Training on a MIDI dataset such as Lakh MIDI would give the critic
  much better taste.
- **The judge is shared and hand-weighted.** Its weights in `judge.py` could themselves be tuned
  by the GA or learned from listener ratings.
- **Bias and ethics.** A critic trained on Western folk tunes encodes Western taste. Raga-based
  melodies would score poorly. This is a concrete example for the "bias in AI" unit.

## Ideas to extend it

- A Streamlit interface with live sliders.
- A human-versus-AI mode where you type or play a phrase and the AI answers.
- A raga mode, with scales and a corpus from Indian classical music.
- Neuroevolution, where the GA evolves the critic's weights.
- A third player.