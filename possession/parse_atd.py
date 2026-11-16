"""Raw tactical ATD -> cached numpy arrays, in metres, indexed by video frame (= submission row).

One streaming pass over the 199,319-line text file. Each line is
    frame_id:x,y,s;x,y,s;...(46 slots):x,y,s(ball)
so swapping the two outer separators for commas gives 1 + 46*3 + 3 numbers, and float() reads
'NaN' natively.
"""

import os

import numpy as np

from common import CACHE_DIR, games
import baseline  # organisers' starter: team_of_each_track, team_names

import config

CACHE = os.path.join(CACHE_DIR, "atd_tactical.npz")


def parse(raw_path, n_frames, n_slots):
    width = 1 + n_slots * 3 + 3
    table = np.full((n_frames, width), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            values = np.array(line.replace(":", ",").replace(";", ",").split(","),
                              dtype=np.float32)
            if len(values) != width:
                continue
            table[int(values[0]) - 1] = values
    return table


def load(force=False):
    """dict(ball[n,2] m, players[n,S,2] m, speed[n,S] m/s, home_slot[S] bool, names)."""
    if os.path.isfile(CACHE) and not force:
        data = np.load(CACHE, allow_pickle=False)
        return {k: data[k] for k in data.files}

    meta_path, raw_path = games.require(["atd_meta", "atd_raw"])
    team_of_slot = baseline.team_of_each_track(meta_path)
    n_frames = games.source_frames("tactical")
    length, width = config.PITCH

    table = parse(raw_path, n_frames, len(team_of_slot))
    slots = table[:, 1:1 + 3 * len(team_of_slot)].reshape(n_frames, -1, 3)
    out = {
        "ball": np.stack([table[:, -3] * length, table[:, -2] * width], axis=1),
        "players": np.stack([slots[:, :, 0] * length, slots[:, :, 1] * width], axis=2),
        "speed": slots[:, :, 2],
        "home_slot": team_of_slot == baseline.HOME,
        "names": np.array(baseline.team_names(meta_path)),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(CACHE, **out)
    return out


if __name__ == "__main__":
    data = load(force=True)
    seen = ~np.isnan(data["ball"][:, 0])
    print("frames {}  slots {} (home {})  ball seen {:.1f}%  teams {}".format(
        len(seen), data["speed"].shape[1], int(data["home_slot"].sum()), 100 * seen.mean(),
        list(data["names"])))
