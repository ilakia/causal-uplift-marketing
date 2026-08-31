# Data

## Getting the Criteo-Uplift dataset

This project's sandbox environment has restricted network access, so the
dataset needs to be downloaded from **your own Claude Code environment**
(on your machine) where you have full internet access:

1. Go to https://ailab.criteo.com/criteo-uplift-prediction-dataset/
2. Download `criteo-uplift-v2.1.csv.gz` (or current version listed there)
3. Place it here as `data/raw/criteo_uplift.csv.gz`

Alternatively, once downloaded once, it's also mirrored on Kaggle
("Criteo Uplift Modeling Dataset") if the direct link is slow.

## Columns (reference)

- `f0`–`f11`: anonymized user features (12 numerical features)
- `treatment`: 1 if user was randomly exposed to advertising, 0 otherwise
- `conversion`: 1 if user converted
- `visit`: 1 if user visited
- `exposure`: whether treatment was actually delivered

## Why not commit this to the repo

At ~25M rows / several GB, this isn't something to commit to git — treat it
as a local data dependency. Add `data/raw/` to `.gitignore` (already done).
