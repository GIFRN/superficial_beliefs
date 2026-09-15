# Superficial Beliefs in LLM Decision-Making

Gabriel Freedman and Francesca Toni · COLM 2026

Code, synthetic datasets, and saved analyses for studying how LLM choices and stated reasons relate to a fitted behavioural model. This is the paper submission snapshot, previously distributed as `COLM.zip`.

## Results

No installation or model calls are needed to view these saved results.

- **Main benchmark (Table 2):** [aggregate results](outputs/final_same_order/reports/PAPER_TABLE1_SUMMARY.md) and [results by model and setting](outputs/final_same_order/reports/PAPER_APPENDIX_SUBSTANTIVE_GRID.md).
- **Perturbations (Figure 3):** [figure](outputs/final_same_order/reports/Figure1_reproducibility_recovery.png).
- **Targeted occlusion (Figure 4):** [figure](outputs/final_same_order/reports/Figure2_occlusion_validation.png).
- **Alternative behavioural models:** [summary](outputs/final_same_order/reports/m0_m1_m2_choice_attribute_nll_driver_summary.csv).
- **Hospital benchmark:** [results](outputs/hospital_cyber_nt/reports/hospital_cyber_nt_summary.md) and [occlusion analysis](outputs/hospital_cyber_nt/reports/hospital_cyber_nt_equalise_occlusion.md).
- **Additional occlusions:** [Qwen / policy](outputs/occlusion_suite_policy_qwen_min/reports/occlusion_suite_current_run/methodc_current_run_analysis.md) and [Ministral / software](outputs/occlusion_suite_software_ministral_min/reports/occlusion_suite_current_run/methodc_current_run_analysis.md).

Some filenames retain earlier manuscript numbering. Other files in `outputs/` contain intermediate fits and diagnostic analyses.

**Metric conventions:** the saved main grid compares judge choices with observed LLM choices, and attribute reports with the behavioural model's predicted-side driver. These conventions differ from the corresponding descriptions in the paper and require reconciliation.

## Contents

- [`data/`](data/README.md): base datasets, themes, and model configurations.
- [`src/`](src/): data preparation, model interfaces, and analysis code.
- [`scripts/`](scripts/): benchmark and analysis entry points.
- [`outputs/`](outputs/): themed datasets, fitted results, and reports.
- [`tests/`](tests/): unit tests and checks requiring original response logs.

## Setup

Python 3.10 or newer. Run commands from the repository root.

```bash
git clone https://github.com/GIFRN/superficial_beliefs.git
cd superficial_beliefs
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For new API runs, install the clients and export the key:

```bash
python -m pip install -e '.[llm]'
export OPENAI_API_KEY="your-key"
```

Local Qwen and Ministral runs require model servers at the endpoints in [`data/models/`](data/models/).

## Entry points

The snapshot includes derived results but omits raw `runs/…/responses.jsonl` logs. Full regeneration requires those logs or new inference.

- [`run_final_benchmark.py`](scripts/run_final_benchmark.py): collect main-benchmark responses.
- [`analyze_final_benchmark.py`](scripts/analyze_final_benchmark.py): fit behavioural models and evaluate completed runs.
- [`generate_paper_results_bundle.py`](scripts/generate_paper_results_bundle.py): generate reports from completed runs and fits.

Each script accepts `--help`. Dataset and model configurations are under `data/`.
