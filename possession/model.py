"""Per-frame evidence -> one 3-state Viterbi over {H, A, D} per graded half.

Why one decoder for everything: the restart owner after a stoppage, the turnover point, and the
stoppage itself are all the same decision -- where does the label change -- and a single best path
with switch costs makes each of them only when the accumulated evidence pays for the switch. A
two-frame opposing touch inside a long possession (a pass flying past an opponent) never pays for
two switches, so it is ignored; a real interception is followed by seconds of opposing control
and does.
"""

import numpy as np

import config

H, A, D = 0, 1, 2


def dead_evidence(f):
    e = np.zeros(len(f["seen"]))
    e += config.W_OUT * f["out"]
    squad = np.nan_to_num(f["squad_speed"], nan=config.WALK_SPEED_MS)
    slow = config.W_SLOW * np.clip(config.WALK_SPEED_MS - squad, -1.0, 1.0)
    near = np.minimum(f["d_home"], f["d_away"]) < config.TOUCH_RADIUS_M
    speed = np.nan_to_num(f["ball_speed"], nan=0.0)
    if config.GATE_SLOW_IN_PLAY:  # v2: a slow squad around a moving ball is slow play, not a stoppage
        in_play = f["seen"] & ~f["out"] & (speed >= config.STILL_BALL_MS)
        slow = np.where(in_play & (slow > 0), slow * config.SLOW_IN_PLAY_FACTOR, slow)
    e += slow
    e += config.W_STILL_BALL * (f["seen"] & (speed < config.STILL_BALL_MS) & ~near)
    e += config.W_MISSING * ~f["seen"]
    e += config.W_SET_PIECE * f["set_piece"]
    pan = np.nan_to_num(f["camera_pan"], nan=10.0)
    e += config.W_STILL_CAMERA * (pan < config.STILL_CAMERA_MS)
    e -= config.W_FAST_BALL_ALIVE * (f["seen"] & ~f["out"] & (speed > config.FAST_BALL_MS))
    return e


def touches(f):
    """Per-frame touch sign: +1 home, -1 away, 0 none (uncontested player within touch radius)."""
    d_home, d_away = f["d_home"], f["d_away"]
    touch = f["seen"] & ~f["out"] & (np.minimum(d_home, d_away) < config.TOUCH_RADIUS_M)
    with np.errstate(invalid="ignore"):  # inf - inf where nobody is tracked; touch is False there
        clear = np.abs(d_home - d_away) > config.TOUCH_MARGIN_M
    return np.where(touch & clear, np.where(d_home < d_away, 1, -1), 0)


def team_evidence(f):
    fast = np.nan_to_num(f["ball_speed"], nan=0.0) > config.FAST_BALL_MS
    weight = np.where(fast, config.W_TOUCH_FAST, config.W_TOUCH)
    e = touches(f) * weight
    e += config.W_PHASE * np.clip(f["phase_vote"], -2, 2)
    return e


def viterbi(scores, transition):
    n = len(scores)
    back = np.empty((n, 3), dtype=np.int8)
    current = scores[0].copy()
    for t in range(1, n):
        candidate = current[:, None] + transition      # [from, to]
        back[t] = candidate.argmax(axis=0)
        current = candidate.max(axis=0) + scores[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = current.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(f, blocks):
    """Labels for the whole video: H/A/D inside each (half, first, last) block, X elsewhere."""
    e_dead, e_team = dead_evidence(f), team_evidence(f)
    scores = np.stack([e_team / 2, -e_team / 2, e_dead], axis=1)
    t, d = config.C_TURN, config.C_DEAD
    transition = -np.array([[0, t, d], [t, 0, d], [d, d, 0]], dtype=float)

    labels = np.full(len(e_dead), "X", dtype="<U1")
    for _, first, last in blocks:
        path = viterbi(scores[first:last + 1], transition)
        labels[first:last + 1] = np.array(["H", "A", "D"])[path]

    if config.MERGE_UNSUPPORTED_DEAD:
        labels = merge_unsupported_dead(labels, f, blocks)
    if config.SNAP_TURNOVERS:
        labels = snap_turnovers(labels, touches(f), blocks)
    return labels


# --- version 2 post-processing --------------------------------------------------------------

def runs(values):
    """(start, stop_exclusive, value) of every maximal constant run."""
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(values)]))
    return [(int(a), int(b), values[a]) for a, b in zip(starts, stops)]


def hard_dead_cue(f):
    """Frames carrying direct evidence of a stoppage, independent of the squad-speed cue."""
    speed = np.nan_to_num(f["ball_speed"], nan=0.0)
    near = np.minimum(f["d_home"], f["d_away"]) < config.TOUCH_RADIUS_M
    placed = f["seen"] & (speed < config.STILL_BALL_MS) & ~near
    missing_long = np.zeros(len(speed), dtype=bool)
    min_len = int(config.MISSING_HARD_S * config.FPS)
    for a, b, lost in runs(~f["seen"]):
        if lost and b - a >= min_len:
            missing_long[a:b] = True
    return f["out"] | f["set_piece"] | placed | missing_long


def merge_unsupported_dead(labels, f, blocks):
    """H-D-H (or A-D-A) where the D run has no hard cue -> the whole stretch is one possession.

    A stoppage inferred from a slow squad alone -- ball never out, never placed, never lost by the
    tracker, no set piece tagged -- is far more often a slow spell of circulation by the team that
    had the ball before and after it."""
    out = labels.copy()
    cue = hard_dead_cue(f)
    for _, first, last in blocks:
        segment = runs(labels[first:last + 1])
        for i in range(1, len(segment) - 1):
            a, b, label = segment[i]
            before, after = segment[i - 1][2], segment[i + 1][2]
            if (label == "D" and before == after and before in ("H", "A")
                    and not cue[first + a:first + b].any()):
                out[first + a:first + b] = before
    return out


def snap_turnovers(labels, touch, blocks):
    """Move each H<->A boundary to the first touch of the new team after the old team's last.

    Searched within SNAP_WINDOW_S of the decoded boundary and never past a neighbouring boundary,
    so the number and order of spells is unchanged; only where they meet moves."""
    out = labels.copy()
    window = int(config.SNAP_WINDOW_S * config.FPS)
    for _, first, last in blocks:
        segment = runs(labels[first:last + 1])
        for i in range(1, len(segment)):
            (pa, _, old), (b, nb, new) = segment[i - 1], segment[i]
            if {old, new} != {"H", "A"}:
                continue
            lo = first + max(pa + 1, b - window)          # stay inside the old spell
            hi = first + min(nb - 1, b + window)          # ...and inside the new one
            sign_old, sign_new = (1, -1) if old == "H" else (-1, 1)
            span = touch[lo:hi + 1]
            old_hits = np.flatnonzero(span == sign_old)
            start = old_hits[-1] + 1 if len(old_hits) else 0
            new_hits = np.flatnonzero(span[start:] == sign_new)
            if not len(new_hits):
                continue
            boundary = lo + start + new_hits[0]
            out[lo:boundary] = old
            out[boundary:hi + 1] = new
    return out
