# Catalonyst — Ball Possession (More than a Hack 2027, Phase 1)

**Team:** Catalonyst · **File:** `Catalonyst.csv` · **Code:** `possession/` in this repository ·
**Reproduction:** [section 6](#6-reproduction)

## 1. Approach

There is no learned model and no labels. The pipeline turns the tracking, homography and Smart
Tagging into two per-frame scores, **dead-ball evidence** and **team evidence**. It then decodes
each half with one **3-state Viterbi over {H, A, D}**, in which a label changes only when the
accumulated evidence pays a switch cost.

Why a decoder rather than a frame classifier: `spellF1` scores possessions, so every spurious flip
costs a whole possession. A brief opposing touch (a pass flying past an opponent) cannot pay for
two switches, while a real interception, followed by seconds of opposing control, does. The same
decode settles the stoppages, the turnover points and who wins each restart.

**Dead-ball evidence**
- ball more than 0.5 m outside the pitch (strongest cue)
- squad walking: mean player speed below 1.6 m/s; above it, the cue counts as *alive*
- placed ball: still, with nobody within 2 m
- ball lost by the tracker
- a Smart Tagging set-piece window
- a parked camera (from the homography)
- a ball travelling faster than 8 m/s inside the pitch counts *against* dead (only a ball in play
  is struck that hard)

**Team evidence**
- an uncontested touch: nearest player within 2 m and at least 0.75 m closer than any opponent
- a weak Smart Tagging phase vote

**Clean-up after decoding**
- a stoppage between two spells of the same team is removed if nothing physical supports it (ball
  never out, never placed, never lost for ≥ 2 s, no set piece)
- each change of possession is moved to the new team's first touch

**How the thresholds were set.** Every threshold is a football or physics prior, checked against
the *unlabelled* tracking in `possession/notebooks/threshold_analysis.ipynb`. The main example is
the walking threshold. We compared squad speed while the ball is out of the pitch (dead by the
Laws) with squad speed while the ball travels faster than any player can run (alive). The two
distributions cross at 1.6 m/s, stable across anchor definitions. All weights are in
`possession/config.py`, each with its justification.

## 2. Results

| | spellF1 | macroF1 | possessions |
|---|---|---|---|
| **Catalonyst** | **0.463** | 0.724 | 15 |
| baseline_smooth50 (organisers' floor) | 0.326 | 0.621 | 26 |
| ground truth | | | 17 |

These are scores on the public 5-minute window, used only as a pipeline check. The submission was
fixed before we scored it, and nothing was tuned on it. The window did prompt one design choice: an
earlier version's boundaries were poorly timed there, which led to the "first touch" rule. That
rule's only parameter (a ±2 s search window) was never evaluated on the window.

Full-match checks that need no labels:
- predicted dead-ball share 47.7%, inside the 43–51% that ~52–60 min of effective playing time
  implies for this match;
- 228 possessions, with a mean of 14.6 s.

## 3. Assets and training data

- **Used:** the tactical tracking (ATD), the tactical homography, and the tactical Smart Tagging.
- **Not used:**
  - the videos (the tracking already gives pitch coordinates);
  - the broadcast assets (not synchronised with the tactical video).
- **Tried and dropped:**
  - line detection, because a coordinate test on the tracking is more accurate;
  - interpolating the ball through tracking gaps, because the gaps are long (median 10.7 s);
  - camera position as team evidence, because it is ambiguous;
  - a vision model, because it was not feasible without labels before the deadline.
- **Training data: none.** We used no labelled data of any kind, no event data and no corrected
  tracking. No ground truth was used for tuning or selection.

## 4. Possession rules implemented

- Every graded frame is H, A or D. We never abstain inside graded time.
- **Control:** only an uncontested touch counts. Contested and loose balls carry no team evidence,
  so possession stays with the last team, as CHALLENGE.md specifies.
- **Not a change of possession:** deflections, blocks and a pass passing an opponent. These are too
  short to pay for two switches.
- **Change of possession:** sustained opposing control, timed at the new team's first touch.
- **Restarts:** the first team to touch the ball after a stoppage has it.

## 5. Limitations

- The ball is untracked on ~41% of graded frames. Possession through those gaps is inferred, not
  observed. A homography-guided ball detector on the video is the next step.
- Many stoppages rest on squad speed alone.
- The 2 m touch radius counts a player running past the ball.
- The weights are reasoned priors. With labels from *other* matches, they could be fitted.

## 6. Reproduction

Requirements: Python ≥ 3.11, `numpy==2.4.4`, `pandas==3.0.3`. No GPU. A run takes under a minute.

```bash
git clone https://github.com/ismailkattar/catalonyst-mtah2027-possession.git
cd catalonyst-mtah2027-possession
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r possession/requirements.txt
# data bundle in data/tactical/ (lower-case on Linux/macOS), or: export MTAH_DATA_DIR=/path/to/bundle
python possession/run.py --name Catalonyst
python evaluation/validate_submission.py submissions/Catalonyst.csv
```

The command writes `submissions/Catalonyst.csv`, which should match
`sha256 9fe4283781134f062535b762b6ce05a537333b6e35d80e92336f94403b29585e`. We verified this from
fresh clones on Windows (Python 3.14) and Ubuntu (Python 3.12). No match data or frame-level
output is committed.
