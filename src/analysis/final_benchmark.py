from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
OUTPUT_ROOT = ROOT / "outputs/final_same_order"

MAIN_THEMES = ["drugs", "policy", "software"]
PLACEBO_THEMES = ["placebo_packaging", "placebo_label_border"]
ALL_THEMES = MAIN_THEMES + PLACEBO_THEMES


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    family: str
    effort: str
    provider: str
    config_path: Path


MODEL_SPECS = [
    ModelSpec("mini_min", "GPT-5-mini", "minimal", "openai", DATA_ROOT / "models/mini_min.yml"),
    ModelSpec("mini_low", "GPT-5-mini", "low", "openai", DATA_ROOT / "models/mini_low.yml"),
    ModelSpec("nano_min", "GPT-5-nano", "minimal", "openai", DATA_ROOT / "models/nano_min.yml"),
    ModelSpec("nano_low", "GPT-5-nano", "low", "openai", DATA_ROOT / "models/nano_low.yml"),
    ModelSpec("qwen_min", "Qwen3.5-14B", "minimal", "qwen", DATA_ROOT / "models/qwen_min.yml"),
    ModelSpec("qwen_low", "Qwen3.5-14B", "low", "qwen", DATA_ROOT / "models/qwen_low.yml"),
    ModelSpec("ministral_min", "Ministral-3-14B", "minimal", "ministral", DATA_ROOT / "models/ministral_min.yml"),
    ModelSpec("ministral_low", "Ministral-3-14B", "low", "ministral", DATA_ROOT / "models/ministral_low.yml"),
]

MODEL_TAGS = [spec.tag for spec in MODEL_SPECS]
MODEL_BY_TAG = {spec.tag: spec for spec in MODEL_SPECS}

THEME_CONFIGS = {
    "drugs": DATA_ROOT / "themes/drugs.yml",
    "policy": DATA_ROOT / "themes/policy.yml",
    "software": DATA_ROOT / "themes/software.yml",
    "placebo_packaging": DATA_ROOT / "themes/placebo_packaging.yml",
    "placebo_label_border": DATA_ROOT / "themes/placebo_label_border.yml",
}


def output_root(path: str | Path | None = None) -> Path:
    return Path(path).resolve() if path else OUTPUT_ROOT


def datasets_root(base: str | Path | None = None) -> Path:
    return output_root(base) / "datasets"


def runs_root(base: str | Path | None = None) -> Path:
    return output_root(base) / "runs"


def results_root(base: str | Path | None = None) -> Path:
    return output_root(base) / "results"


def reports_root(base: str | Path | None = None) -> Path:
    return output_root(base) / "reports"


def logs_root(base: str | Path | None = None) -> Path:
    return output_root(base) / "logs"


def dataset_dir(theme: str, split: str, *, base: str | Path | None = None) -> Path:
    return datasets_root(base) / theme / split


def run_prefix(theme: str, split: str, model_tag: str, kind: str, *, base: str | Path | None = None) -> Path:
    return runs_root(base) / theme / split / f"{model_tag}_{kind}"


def stagea_dir(theme: str, model_tag: str, *, base: str | Path | None = None) -> Path:
    return results_root(base) / f"stageA_{theme}_{model_tag}"


def judge_dir(theme: str, model_tag: str, *, base: str | Path | None = None) -> Path:
    return reports_root(base) / f"judge_{theme}_{model_tag}"


def resolve_run_dir(prefix: str | Path) -> Path | None:
    prefix_path = Path(prefix)
    if prefix_path.is_dir():
        return prefix_path
    matches = sorted(prefix_path.parent.glob(f"{prefix_path.name}__*"))
    if not matches:
        return None
    matches.sort(key=lambda path: (1 if path.name.endswith("_batch") else 0, path.name))
    return matches[0]
