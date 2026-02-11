#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?}"
: "${PY:?}"

CONFIG="${CONFIG:-configs/default.yml}"
THEME="${THEME:-drugs}"
SEED="${SEED:-13}"
INCLUDE_SWAP="${INCLUDE_SWAP:-0}"
BASE_LIMIT="${BASE_LIMIT:-0}"
RAW_DS="${RAW_DS:-$ROOT/data/occlusion_suite_raw}"
OUT_DS="${OUT_DS:-$ROOT/data/occlusion_suite_methodC}"

cmd=(
  "$PY" scripts/make_occlusion_suite_dataset.py
  --config "$CONFIG"
  --out "$RAW_DS"
  --theme "$THEME"
  --seed "$SEED"
)
if [[ "$INCLUDE_SWAP" == "1" ]]; then
  cmd+=(--include-swap)
fi

echo "Generating raw occlusion suite into: $RAW_DS"
"${cmd[@]}"

echo "Filtering to tradeoff-only matched families in B3..."
ROOT="$ROOT" RAW_DS="$RAW_DS" OUT_DS="$OUT_DS" SEED="$SEED" BASE_LIMIT="$BASE_LIMIT" "$PY" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(os.environ["ROOT"])
raw_ds = Path(os.environ.get("RAW_DS", str(root / "data/occlusion_suite_raw")))
out_ds = Path(os.environ.get("OUT_DS", str(root / "data/occlusion_suite_methodC")))
seed = int(os.environ.get("SEED", "13"))
base_limit = int(os.environ.get("BASE_LIMIT", "0"))

trials = pd.read_parquet(raw_ds / "dataset_trials.parquet")
configs = pd.read_parquet(raw_ds / "dataset_configs.parquet")

attrs = ["E", "A", "S", "D"]
delta_cols = [f"delta_{a}" for a in attrs]

baseline = trials[(trials["manipulation"] == "short_reason") & (trials["block"] == "B3")].copy()
tradeoff = (baseline[delta_cols] > 0).any(axis=1) & (baseline[delta_cols] < 0).any(axis=1)
eligible_base_ids = baseline.loc[tradeoff, "base_trial_id"].astype(str).unique()

if base_limit > 0 and len(eligible_base_ids) > base_limit:
    rng = np.random.default_rng(seed)
    eligible_base_ids = rng.choice(eligible_base_ids, size=base_limit, replace=False)

allowed_manips = {"short_reason", "occlude_equalize", "occlude_drop", "occlude_swap"}
subset = trials[
    trials["base_trial_id"].astype(str).isin(set(eligible_base_ids))
    & trials["manipulation"].isin(allowed_manips)
].copy()
subset = subset.sort_values("trial_id").reset_index(drop=True)

cfg_ids = set(subset["config_id"].astype(str).unique())
subset_configs = configs[configs["config_id"].astype(str).isin(cfg_ids)].copy()
subset_configs = subset_configs.sort_values("config_id").reset_index(drop=True)

out_ds.mkdir(parents=True, exist_ok=True)
subset_configs.to_parquet(out_ds / "dataset_configs.parquet", index=False)
subset.to_parquet(out_ds / "dataset_trials.parquet", index=False)

manifest = {}
manifest_path = raw_ds / "MANIFEST.json"
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        manifest = {}

manifest.update(
    {
        "source_dataset": str(raw_ds),
        "n_configs": int(subset_configs["config_id"].nunique()),
        "n_trials": int(len(subset)),
        "actual_total": int(len(subset)),
        "target_total": int(len(subset)),
        "blocks": {k: int(v) for k, v in subset["block"].value_counts().to_dict().items()},
        "filter": {
            "type": "matched_tradeoff_families",
            "block": "B3",
            "baseline_manipulation": "short_reason",
            "tradeoff_rule": "any(delta_*)>0 AND any(delta_*)<0",
            "base_limit": base_limit,
            "paired_key": "base_trial_id",
            "allowed_manipulations": sorted(allowed_manips),
            "seed": seed,
        },
    }
)
(out_ds / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

print(f"✅ Wrote method-C dataset: {out_ds}")
print(f"   base families: {subset['base_trial_id'].nunique()}")
print(f"   trials: {len(subset)}")
print(f"   manipulations: {subset['manipulation'].value_counts().to_dict()}")
PY
