# Catalonyst — Ball Possession, More than a Hack 2027 (Phase 1)

**Team:** Catalonyst · **Submission file:** `Catalonyst.csv` (= pipeline version **v2**) ·
**Code:** this repository, folder `possession/`

The method has no learned model and uses no labels. It turns the provided tracking, homography and
Smart Tagging into per-frame evidence of "dead ball" and of "which team has it". It then decodes
the whole match with one three-state Viterbi over {H, A, D}, in which a label may change only when
the accumulated evidence pays a switch cost. v2 recalibrates one threshold from physical anchors in
the unlabelled tracking and adds two rule-based clean-ups that follow from the possession rules.

---

## 1. Approach

### Why segmentation, not frame classification
`spellF1` scores **possessions**, not frames. A per-frame classifier, even an accurate one, turns
every brief opposing touch into an extra possession: the organisers' raw nearest-player baseline
has 131 possessions on the public 5 minutes, against 17 real ones. We therefore decode the whole
half at once. The restart owner after a stoppage, the turnover point and the stoppage itself are
the same decision: where does the label change? One best path with switch costs makes each change
only when the evidence pays for it. A two-frame opposing touch inside a long possession (a pass
flying past an opponent) cannot pay for two switches. A real interception, followed by seconds of
opposing control, does.

### Pipeline (`possession/`)
| file | role |
|---|---|
| `parse_atd.py` | one streaming pass over the raw tactical ATD → arrays in metres (ball, 46 player slots, per-player speed, home/away slot map), cached to `cache/atd_tactical.npz` |
| `features.py` | per-frame features: ball out of pitch, nearest home/away distance, ball speed, squad speed, camera pan (homography), Smart Tagging phase vote and set-piece windows |
| `model.py` | dead evidence and team evidence → Viterbi per graded half → v2 post-processing |
| `config.py` | every threshold and weight, each with its football justification; `VERSIONS` holds v1 and v2 |
| `run.py` | end to end: data → `submissions/<name>.csv`, then the organisers' validator |
| `notebooks/threshold_analysis.ipynb` | where each threshold comes from, checked on the unlabelled tracking |

The organisers' own loaders (`games.py`, `starter/baseline.py`, `starter/load_homography.py`,
`starter/load_smart_tagging.py`) are imported unchanged.

### Evidence (per frame, 25 per second)
**Dead evidence** `e_dead` (positive = dead):

| cue | weight | justification |
|---|---|---|
| ball more than 0.5 m outside the pitch | +3.0 | out of play by Law 9. The strongest single cue |
| squad mean speed below `WALK_SPEED_MS` (2 s window) | +1.5 per m/s, clipped ±1 | players walk during stoppages and run while the ball is alive. Above the threshold it counts as *alive* |
| ball still (< 1 m/s) and nobody within 2 m | +0.5 | a placed ball: a restart is coming |
| ball not tracked | +0.3 | the tracker loses the ball more often during stoppages |
| first half of a Smart Tagging SET PIECES window | +0.5 | the restart is approaching |
| camera barely moving (< 0.5 m/s) | +0.2 | play is not travelling (weak: the camera also parks during slow build-up) |
| ball > 8 m/s inside the pitch | −1.0 | struck harder than anyone can carry it, so it is in play |

**Team evidence** `e_team` (positive = home):

| cue | weight | justification |
|---|---|---|
| uncontested touch: nearest player within 2 m **and** 0.75 m closer than the nearest opponent | ±1.0 | that player can play the ball. Contested balls carry no evidence (possession stays with the last team) |
| the same with the ball > 8 m/s | ±0.3 | the ball is probably passing by, not controlled |
| Smart Tagging phase vote (attack codes name the team in possession, defend codes the one without) | ±0.05 per tag, max 2 | long, coarse windows. Deliberately too weak to overrule touches |

**Decoder.** State scores are H = `e_team/2`, A = `−e_team/2`, D = `e_dead`. Switch costs are
H↔A = 10 and alive↔dead = 20, in the same units. So an opposing "possession" shorter than about
20 frames of clear touches never pays for itself. One Viterbi runs per graded half. Frames outside
the graded blocks are `X`.

### v2: what changed and why
v1 predicted **55.1 %** dead ball. Our label-free diagnosis (no ground truth involved) found the
slow-squad cue was driving it: 94 % of v1's D frames had squad speed < 1.8 m/s, but the ball was out
on only 4 % of them. Elite football has about 52–60 min of effective playing time. The graded frames
here contain 103.9 min of clock time between kickoffs and whistles, plus 2,934 pre-kickoff frames
that are all dead. So a plausible D share is about **43–51 %**, and v1 was outside it.

v2 makes four changes. Each follows from a principle, and none was tuned on a score:

1. **`WALK_SPEED_MS` 1.8 → 1.6 m/s, calibrated without labels.** We compared squad speed under two
   *physical anchors*, frames whose state is fixed by the Laws of the Game rather than by anyone's
   judgement:
   - **dead:** 3–15 s after the ball left the pitch, before it is back in;
   - **alive:** the ball inside the pitch moving faster than 8 m/s, above the 99.9th percentile of
     player speed.

   Their log-likelihood ratio crosses zero at **1.57–1.60 m/s** for every anchor definition that
   excludes the first 3 s after the ball goes out (when players are still decelerating). With
   those seconds included, it crosses at ~1.78 m/s, which is where v1's 1.8 came from. v2 uses 1.6.
   Details and plots are in `possession/notebooks/threshold_analysis.ipynb` §4.
2. **Gate the slow-squad cue while the ball is visibly in play** (`SLOW_IN_PLAY_FACTOR = 0.5`).
   25 % of slow-squad frames have the ball tracked, inside the pitch and moving: slow circulation,
   not a stoppage. The cue keeps full weight when the ball is missing, still or out.
3. **Merge unsupported stoppages.** An `H D H` (or `A D A`) whose D run has *no* hard stoppage cue
   becomes one possession. The hard cues are: ball out, a set-piece tag, a placed ball, or the
   tracker losing the ball for ≥ 2 s (the squad inside such gaps moves at dead-anchor speed). The
   reason is that a stoppage leaves a trace, and slow play alone is not one.
4. **Snap turnovers to touches.** Each H↔A boundary moves to the first touch of the new team after
   the last touch of the old one, searched ±2 s and never past a neighbouring boundary. CHALLENGE.md
   defines the change of possession at the moment the opponent gains control. The decoder's cost
   balance only puts the boundary somewhere near it.

### v1 vs v2

Label-free statistics, full match, graded frames:

| | v1 | +1.6 m/s | +gate | +merge | **v2** (+snap) |
|---|---|---|---|---|---|
| dead-ball share | 55.1 % | 50.5 % | 48.3 % | 47.7 % | **47.7 %** |
| possessions | 232 | 236 | 232 | 228 | **228** |
| mean possession | 12.3 s | 13.3 s | 14.2 s | 14.6 s | **14.6 s** |
| possessions < 2 s | 14 | 18 | 17 | 17 | **12** |
| D runs | 135 | 126 | 119 | 115 | **115** |
| D runs without any hard cue | 16 | 10 | 6 | 2 | **2** |

Public 5-minute window. This is a **pipeline sanity check only**: v2 was frozen before this was
computed, and nothing was selected or tuned on it. It holds 17 possessions, so it is far too small
to rank variants, and EVALUATION.md shows it can order the baselines the wrong way round.

| | macroF1 | spellF1 | possessions (GT 17) | mean duration (GT 14.2 s) |
|---|---|---|---|---|
| baseline_smooth50 (organisers' floor) | 0.621 | 0.326 | 26 | 10.8 s |
| v1 | **0.746** | 0.388 | 17 | 12.5 s |
| **v2 (submitted)** | 0.724 | **0.463** | 15 | 17.9 s |

v2 trades a little per-frame accuracy (fewer D frames, and some slow phases now called alive) for
possession structure, which is what `spellF1` ranks. We submit v2 because of the label-free
evidence above: a D share inside the plausible band, fewer unsupported stoppages and fewer
sub-2-second possessions. The public score agreeing is welcome, but it is not the reason.

---

## 2. Assets: used, tried and dropped

| asset | status | how |
|---|---|---|
| tactical ATD (tracking) | **used** | ball and player positions, per-player speed, home/away slot map from the ATD XML |
| tactical homography | **used** | pitch point under the image centre, as a camera-pan speed (weak still-camera cue) |
| tactical Smart Tagging | **used** | phase codes as a weak team vote; SET PIECES windows as a restart prior |
| tactical video | not used | the tracking already gives pitch coordinates; see "dropped" below |
| broadcast video, ATD and homography | not used | not synchronised with the tactical timeline; aligning them was not worth the risk before the deadline |
| public 5-minute ground truth | sanity check only | format and pipeline check, as the rules allow. Never used to train, tune or select |

**Tried or considered, then dropped:**
- **Line detection on the video** to detect out-of-play: dropped. The ATD is already in pitch
  coordinates, and a coordinate test is more accurate than detecting lines in the image.
- **Camera half-of-pitch as team evidence:** dropped as ambiguous (build-up happens in your own
  half).
- **Interpolating the ball across short tracking gaps:** dropped after measuring. The ball is lost in
  150 long gaps (median 10.7 s); only 17 are ≤ 1 s, so interpolation would change almost nothing.
- **A vision model (ball/player detector on video crops):** deferred. It is not feasible to train
  and validate without labels on this game before the deadline.
- **A data-driven squad-speed threshold from the histogram's shape** (quantile or Otsu): rejected.
  The distribution has a single broad mode with no valley, so any such cut would be arbitrary. The
  physical anchors replace it.

---

## 3. Training data

**None.** There is no trained model and no labelled dataset of any kind, from this game or any
other. Every weight and threshold is a football or physics prior written in `possession/config.py`.
One of them (`WALK_SPEED_MS` in v2) is checked against the unlabelled tracking, through the physical
anchors described above. No hand labels, no event data and no corrected tracking were used, and the
private ground truth was never seen.

---

## 4. Possession rules implemented

- **Three states only.** Every graded frame is H, A or D. We never abstain on graded frames (X only
  outside the graded blocks), so no frame depends on the scorer's forward-fill.
- **Dead ball:** the ball is out of the pitch by more than 0.5 m, or there is an accumulation of
  stoppage evidence (walking squad, placed or lost ball, set-piece window, parked camera) strong
  enough to pay for entering *and* leaving D (cost 20 each way). In v2, a stoppage between two
  spells of the same team must also show at least one hard cue.
- **Controlled possession:** only an uncontested touch is evidence. The nearest player must be
  within 2 m and at least 0.75 m closer than the nearest opponent.
- **Loose balls and challenges:** contested frames and fast balls carry no (or little) team
  evidence, so the decoder keeps the last team. This matches CHALLENGE.md: "possession stays with
  the last team that had it".
- **Deflections, blocks, a pass passing an opponent:** a short opposing touch cannot pay the two
  H↔A switches (cost 10 each), so it does not change possession. A change needs sustained opposing
  control.
- **Restarts:** after D, the team that touches the ball first collects the evidence and so wins the
  restart. The set-piece Team tag is not used for this, because throw-ins are tagged `N/A`.
- **Turnover instant (v2):** the first touch of the new team after the old team's last touch.

---

## 5. Limitations and what we would do next

- **The dead-ball share is still near the top of the plausible band (47.7 %).** The squad-speed cue
  remains the main D driver, and the ball is untracked on ~41 % of graded frames, so many stoppage
  decisions rest on squad motion alone.
- **Ball missing for long stretches.** Possession inside a long gap is carried by the decoder, not
  observed. A ball detector on the tactical video (guided by the homography) is the obvious next
  step, and the single biggest expected gain.
- **Touch ≠ control.** The 2 m radius uses body-centre positions, so a player running past the ball
  counts as a touch. Dwell time or a player-ball velocity match would be stricter.
- **Fixed weights.** The weights are reasoned priors, not estimates. With labels from *other*
  matches (allowed by the rules), they could be fitted as an HMM or CRF.
- **No uncertainty output.** We never abstain. Abstaining with `X` inside long untracked stretches
  could help `macroF1`, but it does not help `spellF1`.
- **Smart Tagging is used only weakly.** Its windows are coarse, and its team tag is `N/A` on every
  throw-in.

---

## 6. Reproduction

The same data bundle and this repository give **the same file, byte for byte**. Both versions are
checked by SHA-256 below.

**Requirements:** Python ≥ 3.11 (produced with CPython 3.14.0 on Windows 11), plus the pinned
`numpy==2.4.4` and `pandas==3.0.3`. No GPU is needed. A first run takes about 15–60 s on a laptop (parsing
the 196 MB tracking file dominates). We checked the steps below from a fresh clone of this
repository in a new virtual environment: both checksums matched.

```bash
# 1. get the code
git clone https://github.com/ismailkattar/catalonyst-mtah2027-possession.git
cd catalonyst-mtah2027-possession

# 2. environment
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r possession/requirements.txt

# 3. data -- the gated bundle, as DATA.md describes. Only these four files are read:
#      data/tactical/BAR-ATH_tactical_atd.txt
#      data/tactical/BAR-ATH_tactical_atd.xml
#      data/tactical/BAR-ATH_tactical_homography.csv
#      data/tactical/BAR-ATH_tactical_smart_tagging.xml
#    (folder name "tactical" in lower case on Linux/macOS), or point MTAH_DATA_DIR at the bundle:
#      export MTAH_DATA_DIR=/path/to/bundle   # Windows PowerShell: $env:MTAH_DATA_DIR = "D:\bundle"

# 4. regenerate the submitted file (v2) and validate it
python possession/run.py --version v2 --name Catalonyst
python evaluation/validate_submission.py submissions/Catalonyst.csv

# 5. optional: regenerate v1, and score both on the public 5 minutes
python possession/run.py --version v1
python evaluation/run_evaluation.py
```

Expected checksums (`sha256sum submissions/*.csv`, or `Get-FileHash` on Windows):

| file | SHA-256 |
|---|---|
| `submissions/Catalonyst.csv` (= `v2.csv`) | `7be418e044c6356f85b3cf17b321dca8f57f652fdd70a5eb0dd4c613daca560a` |
| `submissions/v1.csv` | `a185b8b826d0ce75c4d647b4da5129f82eac695a2296fb26dc8371989462aa6b` |

Expected console summary: v2 `graded share: H 33.3% A 19.1% D 47.7%`; v1 `H 28.6% A 16.3% D 55.1%`.

Notes:
- The first run parses the raw ATD into `cache/atd_tactical.npz` (about 110 MB, git-ignored). Later
  runs reuse it; `--reparse` forces a rebuild.
- `run.py` also copies the CSV into `evaluation/results/`, so `run_evaluation.py` scores it.
  `cache/`, `submissions/` and `evaluation/results/*.csv` are git-ignored. No match data or derived
  frame-level output is in this repository.
- Threshold notebook (optional): `pip install -r possession/requirements-notebook.txt`, then
  `jupyter nbconvert --to notebook --execute --inplace possession/notebooks/threshold_analysis.ipynb`.
  It reads no ground truth.
