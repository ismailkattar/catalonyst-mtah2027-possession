"""Paths and imports shared by the pipeline. Reuses the organisers' loaders rather than copying them.

Layout (this package lives inside the organisers' repository):

    <repo>/
    |-- games.py, starter/, evaluation/     organisers' code, imported as-is
    |-- data/                               gated match data (never committed)
    |-- possession/                         this pipeline
    |-- cache/                              parsed tracking (generated, never committed)
    `-- submissions/                        output CSVs (generated, never committed)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORG_REPO = ROOT
CACHE_DIR = os.path.join(ROOT, "cache")
SUBMISSION_DIR = os.path.join(ROOT, "submissions")

for path in (ORG_REPO, os.path.join(ORG_REPO, "starter"), os.path.join(ORG_REPO, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import games  # noqa: E402,F401
