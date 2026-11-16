"""Per-frame features from the tracking, the homography and the Smart Tagging. All vectorised."""

import warnings

import numpy as np

from common import games
import load_homography
import load_smart_tagging

import config


def rolling_nanmean(x, window):
    """Centred mean over +-window//2 frames, ignoring NaN. NaN where the window has no data."""
    half = max(1, window // 2)
    valid = ~np.isnan(x)
    csum = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(x))
    lo, hi = np.maximum(0, idx - half), np.minimum(len(x), idx + half + 1)
    count = ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (csum[hi] - csum[lo]) / np.maximum(count, 1), np.nan)


def tracking_features(atd):
    length, width = config.PITCH
    ball, players, home = atd["ball"], atd["players"], atd["home_slot"]
    seen = ~np.isnan(ball[:, 0])

    m = config.OUT_MARGIN_M
    out = seen & ((ball[:, 0] < -m) | (ball[:, 0] > length + m)
                  | (ball[:, 1] < -m) | (ball[:, 1] > width + m))

    dist = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_home, d_away = dist[:, home].min(axis=1), dist[:, ~home].min(axis=1)

    step = np.full(len(ball), np.nan)
    step[1:] = np.hypot(*(ball[1:] - ball[:-1]).T) * config.FPS
    ball_speed = rolling_nanmean(step, config.BALL_SPEED_SMOOTH_FRAMES)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        squad = np.nanmean(atd["speed"], axis=1)
    squad = rolling_nanmean(squad, int(config.PLAYER_SPEED_WINDOW_S * config.FPS))
    return dict(seen=seen, out=out, d_home=d_home, d_away=d_away,
                ball_speed=ball_speed, squad_speed=squad)


def camera_features():
    """Pitch point under the image centre and how fast it moves (m/s)."""
    matrices, _, valid = load_homography.load()
    inverse = np.linalg.inv(np.where(valid[:, None, None], matrices, np.eye(3)))
    centre = inverse @ np.array([960.0, 540.0, 1.0])
    xy = centre[:, :2] / centre[:, 2:3] * np.array(config.PITCH)
    pan = np.full(len(xy), np.nan)
    pan[1:] = np.hypot(*(xy[1:] - xy[:-1]).T) * config.FPS
    pan = rolling_nanmean(np.clip(pan, 0, 60), int(config.CAMERA_PAN_WINDOW_S * config.FPS))
    return dict(camera_xy=xy, camera_pan=pan)


def tagging_features(n_frames, home_name, away_name):
    """phase_vote: +1 per tag saying home has the ball, -1 per tag saying away has it.
    set_piece: True on the first half of each tagged set-piece window (restart approaching)."""
    side = {home_name.strip().lower(): 1, away_name.strip().lower(): -1}
    vote = np.zeros(n_frames + 1)
    set_piece = np.zeros(n_frames + 1)
    for event in load_smart_tagging.load():
        code = (event["code"] or "").strip().upper()
        start = max(0, min(n_frames, event["start_frame"]))
        stop = max(0, min(n_frames, event["end_frame"]))
        if code == "SET PIECES":
            mid = (start + stop) // 2
            set_piece[start] += 1
            set_piece[mid] -= 1
            continue
        team = side.get((event["tags"].get("Team") or "").strip().lower())
        if team is None:
            continue
        sign = team if code in config.ATTACK_CODES else -team if code in config.DEFEND_CODES else 0
        vote[start] += sign
        vote[stop] -= sign
    return dict(phase_vote=np.cumsum(vote)[:n_frames],
                set_piece=np.cumsum(set_piece)[:n_frames] > 0)


def build(atd):
    n = len(atd["ball"])
    feats = tracking_features(atd)
    feats.update(camera_features())
    feats.update(tagging_features(n, str(atd["names"][0]), str(atd["names"][1])))
    return feats
