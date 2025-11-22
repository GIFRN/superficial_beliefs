from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Union

from pydantic import BaseModel, Field, validator

from .io import read_yaml


class BlockB1Config(BaseModel):
    enable: bool = True
    R: int = 6


class BlockB2Config(BaseModel):
    enable: bool = True
    R: int = 4


class BlockB3Config(BaseModel):
    enable: bool = True
    candidates_per_batch: int = 2000
    max_batches: int = 200
    corr_abs_target: float = 0.4
    mean_corr_target: float = 0.25
    max_condition_number: float = 20.0
    target_acceptance_rate: float = 0.3
    early_stop_tolerance: float = 0.2
    initial_tolerance: float = 0.1
    randomization_factor: float = 0.3


class BlocksConfig(BaseModel):
    B1: BlockB1Config
    B2: BlockB2Config
    B3: BlockB3Config


class ManipulationShareConfig(BaseModel):
    short_reason: float = 0.5
    split_reason: float = 0.25
    premise_first: float = 0.15
    redact: float = 0.05
    neutralize: float = 0.05
    inject: float = 0.0

    @validator(
        "short_reason",
        "split_reason",
        "premise_first",
        "redact",
        "neutralize",
        "inject",
    )
    def shares_between_zero_one(cls, value: float) -> float:
        if not (0 <= value <= 1):
            raise ValueError("manipulation shares must lie between 0 and 1")
        return value

    @property
    def normalized(self) -> Dict[str, float]:
        weights = {
            "short_reason": self.short_reason,
            "split_reason": self.split_reason,
            "premise_first": self.premise_first,
            "redact": self.redact,
            "neutralize": self.neutralize,
            "inject": self.inject,
        }
        total = sum(weights.values())
        if total == 0:
            raise ValueError("manipulation shares sum to zero")
        return {name: value / total for name, value in weights.items()}


class ManipulationConfig(BaseModel):
    share: ManipulationShareConfig
    injection: Dict[str, object] = Field(default_factory=dict)


class ReplicatesConfig(BaseModel):
    S: int = 20
    temperature: float = 0.8


class DominanceConfig(BaseModel):
    n: int = 24


class ProfilesConfig(BaseModel):
    attributes: List[Literal["E", "A", "S", "D"]]
    levels: List[Literal["Low", "Medium", "High"]]
    level_scores: Dict[str, int]

    @property
    def level_order(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.levels)}


class PathsConfig(BaseModel):
    generated_dir: Path
    runs_dir: Path
    reports_dir: Path


class OrdersConfig(BaseModel):
    balance_pairwise_precedence: bool = True


class Config(BaseModel):
    seed_global: int = 13
    blocks: BlocksConfig
    orders: OrdersConfig
    paraphrases: Dict[str, int]
    manipulations: ManipulationConfig
    replicates: ReplicatesConfig
    dominance_items: DominanceConfig
    profiles: ProfilesConfig
    orders_permutations: List[List[Literal["E", "A", "S", "D"]]]
    probe_targets: List[Literal["E", "A", "S", "D"]]
    paths: PathsConfig

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        data = read_yaml(path)
        return cls.parse_obj(data)


def load_config(path: Union[str, Path]) -> Config:
    return Config.from_yaml(path)
