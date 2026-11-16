"""Every threshold and weight in the pipeline, in one place, each with its football justification.

None of these were fitted to the public ground truth. They are priors from how football is played
and how the tracking behaves, checked against the unlabelled tracking itself in
notebooks/threshold_analysis.ipynb; the technical document reproduces this file.

The module-level values below are version 1. Later versions are a dict of overrides
(VERSIONS["v2"]) applied by select(version) before the pipeline runs, so v1 stays reproducible
byte for byte.

Evidence is accumulated per frame (25 per second) as log-score-like numbers, so a weight of 1.0
held for one second contributes 25. The Viterbi switch costs below are on the same scale.
"""

FPS = 25
PITCH = (105.3, 68.0)          # metres, games.py (measured from corrected tracking)

# --- geometry -------------------------------------------------------------------------------
OUT_MARGIN_M = 0.5             # ball radius + tracking noise before a ball is called out of play
TOUCH_RADIUS_M = 2.0           # a player within ~2 m of the ball can play it in one step
TOUCH_MARGIN_M = 0.75          # nearest player must beat the nearest opponent by this, else contested

# --- ball / player kinematics ---------------------------------------------------------------
BALL_SPEED_SMOOTH_FRAMES = 5   # 0.2 s smoothing of a noisy position derivative
FAST_BALL_MS = 8.0             # above this the ball is travelling (pass, shot), not under control
STILL_BALL_MS = 1.0            # below this the ball is (near) stationary: placed for a restart
PLAYER_SPEED_WINDOW_S = 2.0    # squad speed is judged over 2 s, a stoppage lasts longer than that
WALK_SPEED_MS = 1.8            # squad mean speed below this ~ walking: typical of a stoppage
CAMERA_PAN_WINDOW_S = 1.0
STILL_CAMERA_MS = 0.5          # tactical camera barely moving: play is not travelling

# --- dead-ball evidence (positive = dead, negative = alive) ---------------------------------
W_OUT = 3.0                    # ball outside the pitch: by the laws, dead. Strongest single cue
W_SLOW = 1.5                   # per m/s of squad speed below WALK_SPEED (clipped to +-1 m/s)
W_STILL_BALL = 0.5             # stationary ball not at a player's feet: placed for a restart
W_MISSING = 0.3                # tracker lost the ball: weakly more common at stoppages
W_SET_PIECE = 0.5              # first half of a tagged set-piece window: the restart is coming
W_STILL_CAMERA = 0.2
W_FAST_BALL_ALIVE = 1.0        # a ball travelling fast inside the pitch is in play

# --- team evidence (positive = home, negative = away) ---------------------------------------
W_TOUCH = 1.0                  # uncontested player within touch radius, ball slow enough to control
W_TOUCH_FAST = 0.3             # same, but ball travelling fast: probably passing by, not controlled
W_PHASE = 0.05                 # Smart Tagging phase vote: weak, long, noisy; must lose to touches

# --- segmentation (Viterbi switch costs, in evidence units) ---------------------------------
C_TURN = 10.0                  # H<->A: an opposing blip shorter than ~2*C_TURN frames is ignored
C_DEAD = 20.0                  # alive<->dead: a stoppage must carry real evidence to be declared

# Smart Tagging codes that name the team IN possession, and those that name the team WITHOUT it.
ATTACK_CODES = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD",
                "BALL IN THE BOX", "LONG BALL", "ATTACKING TRANSITION", "PLAYERS IN THE BOX"}
DEFEND_CODES = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}


# --- version 2 switches (all off in v1) ------------------------------------------------------
GATE_SLOW_IN_PLAY = False      # damp the slow-squad cue while the ball is visibly in play
SLOW_IN_PLAY_FACTOR = 1.0      # multiplier on the slow-squad dead evidence in that case
MERGE_UNSUPPORTED_DEAD = False # H-D-H / A-D-A with no hard stoppage cue in the D run -> one spell
MISSING_HARD_S = 2.0           # a ball lost by the tracker for this long counts as a hard cue
SNAP_TURNOVERS = False         # put each H<->A boundary on the first touch of the new team
SNAP_WINDOW_S = 2.0            # ...searched this far either side of the decoded boundary

VERSIONS = {
    "v1": {},
    "v2": {
        # Self-supervised calibration (notebooks/threshold_analysis.ipynb): squad speed while the
        # ball is out of the pitch (dead by law) vs while it travels fast inside it (alive). The
        # log-likelihood ratio of the two crosses zero at ~1.55-1.6 m/s, not 1.8: 1.8 m/s called
        # ordinary slow build-up "dead". 1.6 is the crossing rounded up to one decimal.
        "WALK_SPEED_MS": 1.6,
        # A slow squad is ambiguous when the ball is visibly moving inside the pitch (slow
        # circulation at the back looks like a stoppage to the squad-speed cue alone). Halve it
        # there; it keeps full weight when the ball is missing, still or out.
        "GATE_SLOW_IN_PLAY": True,
        "SLOW_IN_PLAY_FACTOR": 0.5,
        # Laws of the game: the ball is dead only when it leaves the pitch or the referee stops
        # play, and a stoppage leaves a trace (ball out, placed ball, set-piece tag, or the
        # tracker losing the ball for seconds as players crowd or the ball is retrieved).
        "MERGE_UNSUPPORTED_DEAD": True,
        # CHALLENGE.md: possession changes when the opponent gains control, i.e. at their first
        # touch after the last touch of the old team -- not wherever the decoder's cost balance
        # happens to land.
        "SNAP_TURNOVERS": True,
    },
}


_V1 = {key: value for key, value in globals().items() if key.isupper() and key != "VERSIONS"}


def select(version):
    """Reset to v1, then apply a version's overrides (call before building features)."""
    if version not in VERSIONS:
        raise SystemExit("unknown version {!r}; known: {}".format(version, sorted(VERSIONS)))
    globals().update(_V1)
    for key, value in VERSIONS[version].items():
        if key not in globals():
            raise KeyError(key)
        globals()[key] = value
