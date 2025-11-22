# Superficial Beliefs Research Project

Research project investigating implicit superficial beliefs and enthymeme use in Large Language Models.

## Setup

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd superficial_beliefs
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package with dependencies:
```bash
# Core dependencies only
pip install -e .

# With LLM support (OpenAI, Anthropic)
pip install -e .[llm]
```

### Environment Setup

If using LLM APIs, create a `.env` file in the project root with your API keys:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## Project Structure

```
├── configs/          # Configuration files for models and themes
├── data/            # Data storage (generated datasets, runs)
├── results/         # Experimental results and visualizations
├── scripts/         # Executable analysis and processing scripts
├── src/             # Source code
│   ├── analysis/   # Statistical analysis modules
│   ├── data/       # Data generation and processing
│   ├── llm/        # LLM interaction utilities
│   └── utils/      # Utility functions
└── tests/          # Unit tests
```

## Key Scripts

- `scripts/make_dataset.py` - Generate experimental datasets
- `scripts/run_trials.py` - Run LLM trials
- `scripts/fit_stageA.py` - Stage A statistical analysis
- `scripts/stageB_alignment.py` - Stage B alignment analysis
- `scripts/compare_reasoning_efforts.py` - Compare reasoning effort conditions
- `scripts/visualize_reasoning_effort.py` - Create visualizations

## Usage

See individual script documentation and the various `.md` files in the root directory for detailed usage instructions:

- `QUICK_REFERENCE.md` - Quick reference guide
- `REASONING_EFFORT_COMPARISON_GUIDE.md` - Reasoning effort analysis guide
- `THEME_SYSTEM.md` - Theme application documentation

## Dependencies

Core dependencies include:
- numpy, pandas, scipy - Data processing and analysis
- statsmodels, scikit-learn - Statistical modeling
- matplotlib, plotly - Visualization
- pyyaml, pydantic - Configuration management
- pyarrow - Data serialization

Optional LLM dependencies:
- openai - OpenAI API client
- anthropic - Anthropic API client

## Development

Run tests:
```bash
pytest tests/
```

## License

[Add your license information here]

