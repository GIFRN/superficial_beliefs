# Dataset Cleaning: Removal of Identical Profiles

## Issue Identified

Some trials in the datasets had **identical profiles** for both options (A and B), making them uninformative for the study since:
- There's no meaningful choice to make
- Model responses were generic ("Identical profiles; no expected difference")
- Cannot estimate attribute weights from non-decisions
- Add noise without signal

## Datasets Cleaned

### 1. `v1_short_no_b1`

**Before:**
- 428 trials
- 261 configs
- B2: 136 trials
- B3: 292 trials

**Removed:**
- 19 trials from 10 configs with identical profiles

**After:**
- 409 trials
- 251 configs
- B2: 128 trials
- B3: 281 trials

### 2. `v1_short_400`

**Before:**
- 400 trials
- 244 configs
- B2: 136 trials
- B3: 264 trials

**Removed:**
- 18 trials from 9 configs with identical profiles

**After:**
- 382 trials
- 235 configs
- B2: 128 trials
- B3: 254 trials

## Configs Removed

### Complete List of Identical Profile Configs

| Config ID | Profile (Both A and B) | Block | Notes |
|-----------|------------------------|-------|-------|
| B2-0005 | E:Medium, A:Medium, S:Medium, D:Medium | B2 | All attributes at medium level |
| B3-0021 | E:Medium, A:Low, S:High, D:Medium | B3 | |
| B3-0131 | E:High, A:Medium, S:Low, D:Medium | B3 | |
| B3-0141 | E:Low, A:Medium, S:High, D:High | B3 | |
| B3-0169 | E:Low, A:High, S:Medium, D:Low | B3 | |
| B3-0187 | E:High, A:High, S:High, D:High | B3 | All attributes at high level |
| B3-0191 | E:High, A:Low, S:Medium, D:High | B3 | |
| B3-0197 | E:Medium, A:Medium, S:High, D:High | B3 | |
| B3-0337 | E:Low, A:Medium, S:Low, D:Low | B3 | |
| B3-0399 | E:High, A:High, S:Low, D:Medium | B3 | Only in v1_short_no_b1 |

### Notable Patterns

**Extreme cases:**
- **B2-0005**: All Medium - literally no difference at all
- **B3-0187**: All High - best on everything
- **B3-0337**: Nearly all Low (except Medium adherence) - worst on most things

**Why B3 had more:** 
- B3 uses correlation screening for orthogonality
- Random generation can occasionally produce identical profiles
- These slipped through the generation filters

## Implementation

### Detection Method

```python
for config in configs_df:
    profile_a = json.loads(config['levels_left'])
    profile_b = json.loads(config['levels_right'])
    if profile_a == profile_b:
        # Flag for removal
```

### Removal Process

1. Identified all configs with identical profiles
2. Removed trials using those config_ids
3. Removed the config records themselves
4. Updated trial/config counts in MANIFEST.json
5. Documented removed config_ids in `cleaning_notes`

### Provenance Tracking

Both MANIFEST.json files now include a `cleaning_notes` section documenting:
- Action taken: `removed_identical_profiles`
- Number of configs removed
- Number of trials removed
- Specific config_ids that were removed

## Impact on Analysis

### Minimal Impact
- Represents only 4.4% of trials in v1_short_no_b1
- Represents only 4.5% of trials in v1_short_400
- Improves data quality by removing uninformative cases

### Benefits
- **Cleaner signal**: All remaining trials have meaningful tradeoffs
- **Better estimation**: No cases where model can't distinguish options
- **Clearer analysis**: Won't need to explain "identical profile" edge cases

### Dataset Recommendations

Moving forward, recommend using:
- **`v1_short_no_b1`** (409 trials): For analysis excluding B1 rationality checks
- **`v1_short_400`** (382 trials): For optimized subset with balanced B2/B3 coverage

Both are now clean of identical profile cases.

## Verification

To verify no identical profiles remain:

```python
import pandas as pd
import json

configs = pd.read_parquet('dataset_configs.parquet')

identical_count = 0
for _, config in configs.iterrows():
    profile_a = json.loads(config['levels_left'])
    profile_b = json.loads(config['levels_right'])
    if profile_a == profile_b:
        identical_count += 1

print(f"Identical profiles remaining: {identical_count}")
# Should output: 0
```

## Date

Cleaning performed: 2024-11-10

