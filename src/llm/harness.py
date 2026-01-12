from __future__ import annotations

import json
from dataclasses import dataclass
import re
from typing import Any, Sequence

import numpy as np

from src.data.schema import ATTR_LABELS, Attribute, Profile
from src.utils.config import Config

from .prompts import conversation_plan
from .types import ConversationStep, TrialSpec


class LLMBackend:
    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> str:
        raise NotImplementedError


@dataclass
class StepResult:
    name: str
    content: str
    parsed: dict[str, Any]


@dataclass
class SingleRunResult:
    seed: int
    steps: list[StepResult]
    conversation: list[dict[str, str]]


def build_trial_specs(cfg: Config, configs_df, trials_df) -> list[TrialSpec]:
    attributes: list[Attribute] = cfg.profiles.attributes
    orders = cfg.orders_permutations
    config_lookup = {}
    for row in configs_df.to_dict("records"):
        left_levels = _normalize_levels(row["levels_left"], attributes)
        right_levels = _normalize_levels(row["levels_right"], attributes)
        config_lookup[row["config_id"]] = (
            Profile(left_levels),
            Profile(right_levels),
        )
    specs: list[TrialSpec] = []
    variant_map = {
        "premise_first": "premise_first",
        "short_reason": "short_reason",
        "split_reason": "split_reason",
    }
    for row in trials_df.to_dict("records"):
        left, right = config_lookup[row["config_id"]]
        label_of_left = row.get("labelA", "A")
        if label_of_left == "A":
            profile_a, profile_b = left, right
        else:
            profile_a, profile_b = right, left
        order_a = tuple(orders[int(row["order_id_A"])])
        order_b = tuple(orders[int(row["order_id_B"])])
        variant = variant_map.get(row["manipulation"], "short_reason")
        metadata = {
            "delta": {attr: row.get(f"delta_{attr}") for attr in attributes},
            "positions": {
                attr: {
                    "posA": row.get(f"posA_{attr}"),
                    "posB": row.get(f"posB_{attr}"),
                    "delta": row.get(f"delta_pos_{attr}"),
                }
                for attr in attributes
            },
            "labelA": label_of_left,
            "manipulation": row["manipulation"],
        }
        spec = TrialSpec(
            trial_id=row["trial_id"],
            config_id=row["config_id"],
            block=row.get("block", ""),
            profile_a=profile_a,
            profile_b=profile_b,
            order_a=order_a,
            order_b=order_b,
            paraphrase_id=int(row["paraphrase_id"]),
            manipulation=row["manipulation"],
            attribute_target=row.get("attribute_target"),
            inject_offset=int(row.get("inject_offset", 0)),
            variant=variant,
            seed=int(row.get("seed", cfg.seed_global)),
            metadata=metadata,
        )
        specs.append(spec)
    return specs


def run_trial(
    trial: TrialSpec,
    backend: LLMBackend,
    S: int,
    temperature: float,
    seed: int,
    *,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed ^ trial.seed)
    plan = conversation_plan(trial)
    runs: list[SingleRunResult] = []
    for _ in range(S):
        system_message = {"role": "system", "content": plan.system_prompt}
        conversation: list[dict[str, str]] = [system_message]
        replicate_seed = int(rng.integers(0, 2**32 - 1))
        steps_results: list[StepResult] = []
        history = [system_message.copy()]
        context: dict[str, Any] = {"trial": trial, "steps": {}}
        for idx, step in enumerate(plan.steps):
            prompt = step.prompt(context) if callable(step.prompt) else step.prompt
            if step.reset_context:
                history = [system_message.copy()]
                conversation.append(system_message.copy())
            history.append({"role": "user", "content": prompt})
            conversation.append({"role": "user", "content": prompt})
            step_seed = (replicate_seed + idx) % (2**32)
            # Let exceptions propagate to stop the program
            response = backend.complete(history, temperature=temperature, max_tokens=max_tokens, seed=step_seed)
            history.append({"role": "assistant", "content": response})
            conversation.append({"role": "assistant", "content": response})
            parsed = _parse_step(step, response)
            result = StepResult(name=step.name, content=response, parsed=parsed)
            steps_results.append(result)
            context["steps"][step.name] = result
            if not parsed.get("ok"):
                break
        runs.append(SingleRunResult(seed=replicate_seed, steps=steps_results, conversation=conversation))
    return {
        "trial_id": trial.trial_id,
        "config_id": trial.config_id,
        "block": trial.block,
        "manipulation": trial.manipulation,
        "variant": trial.variant,
        "responses": [
            {
                "seed": run.seed,
                "steps": [
                    {
                        "name": step.name,
                        "content": step.content,
                        "parsed": step.parsed,
                    }
                    for step in run.steps
                ],
                "conversation": run.conversation,
            }
            for run in runs
        ],
    }


def parse_choice(text: str) -> dict[str, Any]:
    """
    Parse choice from model response. Supports various formats:
    - "A" or "B" at start
    - "Drug A" or "Drug B"
    - "<Option A>" or "<Option B>"
    - "<Drug A>" or "<Drug B>"
    - "**Option A.**" (with markdown)
    - Any text containing "Option A/B" or "Drug A/B"
    """
    cleaned = text.strip().upper()
    if not cleaned:
        return {"ok": False, "choice": None}
    
    # Try strict match at start first: "Drug A" or just "A"
    match = re.match(r'^(?:DRUG\s+)?([AB])\b', cleaned)
    if match:
        return {"ok": True, "choice": match.group(1)}
    
    # Try matching common patterns with prefixes/brackets
    # Matches: <Option A>, <Drug A>, **Option A**, Option A, Drug A
    patterns = [
        r'<\s*(?:OPTION|DRUG)\s*>\s*(?:DRUG\s+)?([AB])\b',  # <Option>Drug A or <Option>A
        r'<\s*(?:OPTION|DRUG)\s+([AB])\s*>',  # <Option A> or <Drug A>
        r'\*\*\s*(?:OPTION|DRUG)\s+([AB])\b',  # **Option A or **Drug A
        r'\b(?:OPTION|DRUG)\s+([AB])\b',  # Option A or Drug A anywhere
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return {"ok": True, "choice": match.group(1)}
    
    # Last resort: find any standalone A or B near the start (first 50 chars)
    # This catches edge cases but avoids matching A/B in the middle of explanations
    start_section = cleaned[:50] if len(cleaned) > 50 else cleaned
    match = re.search(r'\b([AB])\b', start_section)
    if match:
        return {"ok": True, "choice": match.group(1)}
    
    return {"ok": False, "choice": None}


def parse_structured_premise(text: str) -> dict[str, Any]:
    attr_code = "UNK"
    attr_ok = False
    text_ok = False
    body = text.strip()
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    attr_map = {
        "E": "E",
        "EFFICACY": "E",
        "A": "A",
        "ADHERENCE": "A",
        "S": "S",
        "SAFETY": "S",
        "D": "D",
        "DURABILITY": "D",
    }
    premise_text = ""
    for line in lines:
        if line.lower().startswith("premiseattribute"):
            _, _, value = line.partition("=")
            token = value.strip().strip("[] ")
            token_upper = token.upper()
            if token_upper in attr_map:
                attr_code = attr_map[token_upper]
                attr_ok = True
            else:
                cleaned_token = token_upper.replace("]", "").replace("[", "")
                pieces = [part.strip() for part in re.split(r"[|,/]+", cleaned_token) if part.strip()]
                # Only accept if exactly one recognizable attribute token is present
                matched = {attr_map[part] for part in pieces if part in attr_map}
                if len(matched) == 1:
                    attr_code = matched.pop()
                    attr_ok = True
        if line.lower().startswith("premisetext"):
            _, _, value = line.partition("=")
            value = value.strip()
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                premise_text = value[1:-1].strip()
                text_ok = True
            else:
                premise_text = value.strip('"')
                text_ok = bool(premise_text)
    if not attr_ok and premise_text:
        inferred = classify_premise_open_text(premise_text)
        if inferred.get("conf", 0.0) > 0:
            attr_code = inferred.get("attr", attr_code)
            attr_ok = True
    return {"ok": attr_ok and text_ok, "attr": attr_code if attr_ok else "UNK", "text": premise_text}


def classify_premise_open_text(text: str) -> dict[str, Any]:
    cleaned = text.lower()
    keywords = {
        "E": ["efficacy", "effective", "response", "outcome", "benefit"],
        "A": ["adherence", "compliance", "follow", "consistent", "pills"],
        "S": ["safety", "risk", "side effect", "adverse", "harm"],
        "D": ["durability", "lasting", "maintain", "long-term", "sustain"],
    }
    scores = {attr: 0 for attr in keywords}
    for attr, terms in keywords.items():
        for term in terms:
            if term in cleaned:
                scores[attr] += 1
    best_attr = max(scores.items(), key=lambda kv: kv[1])[0]
    total = sum(scores.values())
    confidence = scores[best_attr] / total if total else 0.0
    return {"attr": best_attr, "conf": confidence}


def _normalize_levels(levels_obj: Any, attributes: Sequence[Attribute]) -> dict[Attribute, str]:
    if isinstance(levels_obj, dict):
        return {attr: str(levels_obj[attr]) for attr in attributes}
    if isinstance(levels_obj, str):
        data = json.loads(levels_obj)
        return {attr: str(data[attr]) for attr in attributes}
    if isinstance(levels_obj, (list, tuple)):
        return {attr: str(value) for attr, value in zip(attributes, levels_obj)}
    raise TypeError(f"Unsupported levels representation: {type(levels_obj)}")


def _parse_step(step: ConversationStep, response: str) -> dict[str, Any]:
    if step.expects == "choice":
        return parse_choice(response)
    if step.expects == "premise":
        return parse_structured_premise(response)
    if step.expects == "sentence":
        return {"ok": bool(response.strip()), "text": response.strip()}
    return {"ok": False, "unknown_step": step.expects}
