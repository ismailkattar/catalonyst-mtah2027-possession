"""End to end: provided data -> possession CSV, validated.

    python possession/run.py --version v2 [--name Catalonyst] [--reparse]

--version picks the parameter set in config.VERSIONS; --name defaults to the version.
Writes submissions/<name>.csv and a copy into the organisers' evaluation/results/ so
`python evaluation/run_evaluation.py` picks it up (that folder is gitignored there).
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

from common import ORG_REPO, SUBMISSION_DIR, games
import baseline

import config
import features
import model
import parse_atd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2", help="parameter set in config.VERSIONS")
    parser.add_argument("--name", default=None, help="output file stem (default: the version)")
    parser.add_argument("--reparse", action="store_true")
    args = parser.parse_args()
    args.name = args.name or args.version
    config.select(args.version)

    atd = parse_atd.load(force=args.reparse)
    feats = features.build(atd)
    labels = model.decode(feats, games.half_video_blocks(games.GAME))

    path = baseline.write_submission(os.path.join(SUBMISSION_DIR, args.name + ".csv"), labels)
    shutil.copy(path, os.path.join(ORG_REPO, "evaluation", "results", args.name + ".csv"))

    graded = labels != "X"
    spells = np.sum((labels[1:] != labels[:-1]) & np.isin(labels[1:], ["H", "A"]))
    print("wrote {}  H {}  A {}  D {}  X {}  ~{} possessions".format(
        path, *(int(np.sum(labels == k)) for k in "HADX"), int(spells)))
    print("graded share: H {:.1%} A {:.1%} D {:.1%}".format(
        *(np.mean(labels[graded] == k) for k in "HAD")))
    return subprocess.call([sys.executable,
                            os.path.join(ORG_REPO, "evaluation", "validate_submission.py"), path])


if __name__ == "__main__":
    sys.exit(main())
