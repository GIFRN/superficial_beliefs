# Implementation Log — Implicit Superficial Beliefs & Enthymeme Use in LLMs

## Current Status
- End-to-end Python package scaffolded under `src/` with dataset generation, LLM prompting/collection, Stage A/B analysis, diagnostics, and reporting modules.
- CLI scripts cover dataset synthesis, trial execution, model fitting, alignment analysis, and reporting; each script self-injects project root into `sys.path`.
- Configurable via `configs/default.yml` (study design) and `configs/models.yml` (LLM backends & sampling).
- Unit tests (lightweight) validate parsers, prompt builders, and Stage A feature assembly using deterministic fixtures and the mock backend.
- Virtual environment (`.venv`) provisioned with scientific stack and LLM SDK extras; CLI tools are ready once the venv is activated (`source .venv/bin/activate`).
- OpenAI backend updated for GPT-5 Responses API with automatic fallback to Chat Completions; default model now `gpt-5-mini` (configurable in `configs/models.yml`).
- **Version control note:** latest changes staged locally—commit before reruns to preserve the new prompting structure.



## Prompting Structure Update (2025-02)
- `short_reason` replaces the former “none” baseline. Prompt format: `<Option>. <≤12-word reason>` followed by a reason repeat and the structured premise block.
- `split_reason` variant runs the decision and the justification in separate contexts; the second turn restarts the conversation and requests only the key factor.
- All causal probes (`premise_first`, `redact`, `neutralize`, `inject`) now compose with the short-reason prompts. Dominance trials also use `short_reason` so they feed the same baseline.
- Stage B baselines now treat `short_reason` as the reference condition (fallback to `split_reason` if needed).
- Regenerate datasets to pick up the new manipulation shares (`configs/default.yml`) and the absence of the legacy `none` manipulation.

## Repository Map (key additions)
- `pyproject.toml` — project metadata and dependency declarations.
- `configs/` — YAML for dataset design & model backends.
- `src/data/` — schema definitions, balanced order/paraphrase logic, B1/B2/B3 builders, dominance generators, dataset orchestrator.
- `src/llm/` — prompt planning (`prompts.py`), harness (`harness.py`), type defs, and backends (`openai.py`, `anthropic.py`, `vllm.py`, `mock.py`).
- `src/analysis/` — feature extraction, Stage A modelling (design matrix, clustered GLM, AMEs), Stage B metrics, grouped CV, diagnostics, and Markdown reporting.
- `src/utils/` — configuration loader, IO helpers, RNG tools, and balance check utilities.
- `scripts/` — CLI entrypoints (`make_dataset.py`, `run_trials.py`, `fit_stageA.py`, `stageB_alignment.py`, `summarize.py`).
- `tests/` — pytest-style smoke tests for parsers, prompts, and design matrix assembly.

## Data & Experiment Pipeline
1. **Stage 1 Dataset** (`src/data/make.py`)
   - **B1**: Rationality check and causal probe platform (single-attribute dominance trials)
   - **B2**: Trade-off grids (two-attribute interactions)
   - **B3**: Near-orthogonal sampling with incremental correlation screening (multi-attribute complexity)
   - **DOM**: Dominance items (strict dominance relationships)
   - Balanced order permutations & paraphrase selection; manipulation assignment and seeds per trial; exports Parquet + MANIFEST.
2. **Stage 2–3 Collection**
   - Prompt planning via `conversation_plan` with probe injection, choice-first vs premise-first variants, structured readouts.
   - Harness handles multi-step conversations, parsing, and backend abstraction (OpenAI, Anthropic, vLLM, Mock).
   - `scripts/run_trials.py` orchestrates sampling from configs/models, aggregates JSONL traces.
3. **Stage A Analysis**
   - **B1 Validation**: Rationality check (P(choose A) ≥ 0.95) and probe effectiveness validation
   - **Weight Estimation**: Design matrix construction excluding B1 trials (main effects, optional interactions, order bias terms)
   - Cluster-robust GLM fit, AME/weight computation, contributions per attribute, grouped CV metrics, diagnostics helpers
   - `scripts/fit_stageA.py` persists design data, predictions, contributions, and summary JSON with B1 validation results
4. **Stage B Alignment**
   - Premise alignment metrics (driver vs premise, weights vs premise, Spearman correlation) leveraging Stage A contributions.
   - Probe delta computation through per-manipulation GLM refits.
   - `scripts/stageB_alignment.py` outputs alignment/probe JSON; `scripts/summarize.py` composes Markdown report via `reporting.make_report`.

## Tests
- `tests/test_parsers.py` — choice/premise parser correctness & keyword classifier sanity.
- `tests/test_design_matrix.py` — Stage A design matrix columns & contribution arithmetic.
- `tests/test_prompts.py` — conversation plan structure for choice-first and premise-first flows.

## Progress Log
- **Step 1:** Established project skeleton, dependency metadata, config loader, IO/RNG utilities.
- **Step 2:** Implemented Stage 1 builders (B1/B2/B3/dominance), order/paraphrase balancing, dataset orchestrator with manifests.
- **Step 3:** Built prompt planner, harness with parsing/classification, backends (OpenAI/Anthropic/vLLM/Mock), and trial runner CLI.
- **Step 4:** Delivered analysis layer (feature extraction, Stage A GLM with clustered SEs, Stage B metrics, CV/diagnostics, reporting) plus fitting/alignment/report scripts.
- **Step 5:** Added smoke tests, balance utility, and rewrote this log; attempted CLI dry-run confirmed dependency gap (pandas/pyarrow missing).

## Recent Enhancements (2024-12-19)

### B1 Purpose Clarification and Validation
**Problem**: B1 was previously described as contributing to weight calibration, but if models are Pareto optimal, B1 trials would all have P(choose A) = 1.0 with no variation to measure.

**Solution**: Redefined B1 as a **rationality check** and **causal probe platform**:

#### B1 Block Purpose:
1. **Rationality Validation**: 
   - Tests if model recognizes dominance relationships
   - Expected: P(choose A) = 1.0 for all B1 trials
   - Tolerance: ±5% deviation acceptable
   - Failure indicates systematic biases or experimental issues

2. **Causal Probe Testing**:
   - Tests effectiveness of manipulations on dominance recognition
   - Baseline: P(choose A) = 1.0 (no manipulation)
   - Probes: Redact, neutralize, inject attributes
   - Success: Probes change P(choose A) from 1.0

3. **Weight Calibration**:
   - **B1 does NOT calibrate attribute weights**
   - Weight calibration comes from B2/B3 trials with trade-offs
   - B1 provides validation foundation for B2/B3 analysis

#### Implementation Changes:
- **Configuration**: Reduced B1 replicates from R=6 to R=3 (expect deterministic responses)
- **Validation Functions**: Added `validate_b1_rationality()` and `validate_b1_probes()` 
- **Stage A Analysis**: Exclude B1 trials from weight estimation (`exclude_b1=True`)
- **Reporting**: Added B1 validation section with pass/fail indicators
- **Scripts**: Updated `fit_stageA.py` to perform B1 validation before analysis

### B3 Correlation Screening Improvements
**Problem**: The original B3 implementation had basic correlation screening that could be improved for better orthogonality and efficiency.

**Solution**: Implemented enhanced multi-criteria correlation screening with adaptive tolerance:

#### New Features:
1. **Multi-Criteria Acceptance**: 
   - Primary: Max absolute correlation ≤ target (0.05)
   - Secondary: Mean absolute correlation ≤ target/2 (0.025)
   - Tertiary: Condition number ≤ threshold (10.0)

2. **Adaptive Tolerance System**:
   - Increases tolerance if acceptance rate < 10%
   - Decreases tolerance if acceptance rate > 20%
   - Caps maximum tolerance at 0.1 to prevent degradation

3. **Enhanced Statistics Tracking**:
   - Tracks acceptance/rejection reasons
   - Records correlation history and tolerance evolution
   - Provides detailed logging for debugging

4. **Early Stopping**:
   - Stops generation if tolerance exceeds threshold
   - Prevents excessive computational waste

#### Configuration Parameters Added:
```yaml
B3:
  corr_abs_target: 0.05        # Max absolute correlation threshold
  mean_corr_target: 0.025      # Mean correlation threshold (half of max)
  max_condition_number: 10.0   # Maximum condition number for matrix stability
  target_acceptance_rate: 0.1  # Target acceptance rate for adaptive tolerance
  early_stop_tolerance: 0.1   # Stop if tolerance exceeds this value
```

#### Implementation Details:
- **File**: `src/data/build_B3.py`
- **Functions Added**:
  - `_compute_correlation_metrics()`: Multi-metric correlation analysis
  - `_should_accept_candidate()`: Multi-criteria acceptance logic
  - `_update_tolerance()`: Adaptive tolerance adjustment
- **Backward Compatibility**: Uses `getattr()` with defaults for existing configs

#### Benefits:
- **Better Orthogonality**: Multiple criteria ensure more independent attribute vectors
- **Improved Efficiency**: Adaptive tolerance maintains reasonable acceptance rates
- **Enhanced Diagnostics**: Detailed statistics aid in tuning and debugging
- **Robustness**: Handles edge cases (zero variance, NaN correlations, singular matrices)

## Outstanding / Next Steps
1. Install scientific/LLM dependencies (`pandas`, `pyarrow`, `statsmodels`, `openai`, `anthropic`, etc.) and rerun CLI smoke to validate end-to-end flow.
2. ~~Tune B3 sampler acceptance heuristics to reach |corr| ≤ 0.05 reliably and record achieved metrics in manifests.~~ **COMPLETED**
3. Extend diagnostics: B1 monotonicity plots, dominance accuracy, placebo/order coefficient checks, and optional open-text premise classification audit.
4. Wire reporting to include plots/tables (matplotlib/plotly) once dependencies available; emit HTML alongside Markdown.
5. Integrate manifest versioning (git SHA, config hashes) and optional caching for expensive LLM runs.
6. Expand test coverage (mock end-to-end run, probe-specific behaviours) once dependencies and fixtures are in place.
7. Test enhanced B3 implementation with real dataset generation to validate improvements.

## Recent Fixes (2024-12-19)

### OpenAI Backend Parameter Fix
**Problem**: GPT-5 models require `max_completion_tokens` instead of `max_tokens` parameter.

**Solution**: Updated `src/llm/backends/openai.py` to use responses API exclusively for GPT-5 models.

**Changes**: 
- Line 21: Updated model detection to only use responses API for GPT-5, o4, o3 models
- Line 35-43: GPT-5 models use responses API exclusively (no fallback to chat completions)
- Line 37-38: Clear error if responses API is not available
- Line 87-158: Complete responses API implementation with robust response parsing
- Line 106: Uses `max_completion_tokens` parameter correctly
- Line 133-150: Handles multiple response formats from responses API

**Additional Features**:
- Added `debug` parameter to print all queries and responses
- Added `--debug` flag to `scripts/run_trials.py`
- Comprehensive error handling and response parsing

**Impact**: Fixes Step 2 (LLM trial execution) error when using GPT-5 models with proper responses API support.

## Reference Specification (abridged)
Original proposal requirements have been implemented across the structures above, covering dataset design (B1/B2/B3, dominance, order balance), prompting protocols (choice-first/premise-first, probes), Stage A modelling with clustered SEs and contributions, Stage B alignment with causal probes, diagnostics, CLI workflows, and manifests. See module implementations and configs for concrete parameterisations.
