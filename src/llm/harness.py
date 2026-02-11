from __future__ import annotations

import json
from dataclasses import dataclass
import re
from typing import Any, Sequence

import numpy as np

from src.data.schema import ATTR_LABELS, Attribute, Profile
from src.data.themes import ThemeConfig, DRUGS_THEME, theme_from_dict
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


def build_trial_specs(
    cfg: Config, 
    configs_df, 
    trials_df,
    theme_config: ThemeConfig | None = None,
    variant_override: str | None = None,
) -> list[TrialSpec]:
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
        if variant_override:
            variant = variant_override
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
            theme_config=theme_config,
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
            parsed = _parse_step(step, response, trial.theme_config)
            result = StepResult(name=step.name, content=response, parsed=parsed)
            steps_results.append(result)
            context["steps"][step.name] = result
            if not parsed.get("ok") and step.stop_on_fail:
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


def parse_scores4(text: str) -> dict[str, Any]:
    """Parse four tau scores for E/A/S/D from JSON or key=value lines."""
    keys = ["E", "A", "S", "D"]
    values: dict[str, float] = {}
    body = text.strip()
    if not body:
        return {"ok": False, "tau": {}, "missing": keys}

    json_match = re.search(r"\{.*\}", body, flags=re.S)
    if json_match:
        snippet = json_match.group(0)
        try:
            payload = json.loads(snippet)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key, value in payload.items():
                key_upper = str(key).strip().upper()
                if key_upper in keys:
                    try:
                        val = float(value)
                    except (TypeError, ValueError):
                        continue
                    if 0.0 <= val <= 1.0:
                        values[key_upper] = val

    pattern = re.compile(r"\b([EASD])\s*[:=]\s*([0-9]*\.?[0-9]+)\b", re.I)
    for match in pattern.finditer(body):
        key = match.group(1).upper()
        if key in values:
            continue
        try:
            val = float(match.group(2))
        except ValueError:
            continue
        if 0.0 <= val <= 1.0:
            values[key] = val

    missing = [key for key in keys if key not in values]
    ok = len(values) >= 2
    return {"ok": ok, "tau": values, "missing": missing}


def parse_score1(text: str) -> dict[str, Any]:
    """Parse a single tau score in [0,1]."""
    body = text.strip()
    if not body:
        return {"ok": False, "tau": None}
    match = re.search(r"\btau\s*[:=]\s*([0-9]*\.?[0-9]+)\b", body, re.I)
    candidates = []
    if match:
        candidates.append(match.group(1))
    candidates.extend(re.findall(r"\b([0-9]*\.?[0-9]+)\b", body))
    for token in candidates:
        try:
            val = float(token)
        except ValueError:
            continue
        if 0.0 <= val <= 1.0:
            return {"ok": True, "tau": val}
    return {"ok": False, "tau": None}


def parse_pairwise6(text: str) -> dict[str, Any]:
    """Parse six pairwise winners for EA, ES, ED, AS, AD, SD."""
    keys = ["EA", "ES", "ED", "AS", "AD", "SD"]
    order = {"E": 0, "A": 1, "S": 2, "D": 3}
    winners: dict[str, str] = {}
    body = text.strip()
    if not body:
        return {"ok": False, "pairs": {}, "missing": keys}

    def canonical_pair(pair: str) -> str | None:
        pair = pair.upper()
        if len(pair) != 2 or pair[0] == pair[1]:
            return None
        if pair[0] not in order or pair[1] not in order:
            return None
        ordered = sorted(pair, key=lambda k: order[k])
        return "".join(ordered)

    def normalize_winner(token: str) -> str | None:
        token = token.strip().upper()
        if token in order:
            return token
        if token == "TIE":
            return "tie"
        return None

    json_match = re.search(r"\{.*\}", body, flags=re.S)
    if json_match:
        snippet = json_match.group(0)
        try:
            payload = json.loads(snippet)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key, value in payload.items():
                pair = canonical_pair(str(key))
                winner = normalize_winner(str(value))
                if pair in keys and winner:
                    winners[pair] = winner

    pattern = re.compile(r"\b(EA|AE|ES|SE|ED|DE|AS|SA|AD|DA|SD|DS)\s*[:=]\s*(E|A|S|D|TIE)\b", re.I)
    for match in pattern.finditer(body):
        pair = canonical_pair(match.group(1))
        if not pair or pair in winners or pair not in keys:
            continue
        winner = normalize_winner(match.group(2))
        if winner:
            winners[pair] = winner

    missing = [key for key in keys if key not in winners]
    ok = len(winners) >= 2
    return {"ok": ok, "pairs": winners, "missing": missing}


def parse_pairwise1(text: str) -> dict[str, Any]:
    """Parse a single pairwise winner token."""
    body = text.strip()
    if not body:
        return {"ok": False, "winner": None}
    match = re.search(r"\b(?:winner|choice)\s*[:=]\s*(E|A|S|D|TIE)\b", body, re.I)
    if not match:
        match = re.search(r"\b(E|A|S|D|TIE)\b", body, re.I)
    if not match:
        return {"ok": False, "winner": None}
    token = match.group(1).upper()
    winner = "tie" if token == "TIE" else token
    return {"ok": True, "winner": winner}


def parse_structured_premise(text: str, theme_config: ThemeConfig | None = None) -> dict[str, Any]:
    attr_code = "UNK"
    attr_ok = False
    text_ok = False
    body = text.strip()
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    
    # Build attr_map dynamically from theme
    # Maps theme labels (e.g., "Experience", "X") back to source attributes (E, A, S, D)
    attr_map = _build_attr_map(theme_config)
    
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
        inferred = classify_premise_open_text(premise_text, theme_config)
        if inferred.get("conf", 0.0) > 0:
            attr_code = inferred.get("attr", attr_code)
            attr_ok = True
    return {"ok": attr_ok and text_ok, "attr": attr_code if attr_ok else "UNK", "text": premise_text}


def _build_attr_map(theme_config: ThemeConfig | None) -> dict[str, str]:
    """Build a mapping from theme labels/names to source attributes (E, A, S, D).
    
    This allows parsing of themed responses back to canonical attribute codes.
    For example, with candidates theme:
        "Experience" -> "E", "X" -> "E"
        "Culture Fit" -> "A", "F" -> "A"
    """
    # Always include base attribute mappings
    attr_map = {
        "E": "E", "A": "A", "S": "S", "D": "D",
    }
    
    # Add default drug theme labels as fallback
    attr_map.update({
        "EFFICACY": "E",
        "ADHERENCE": "A",
        "SAFETY": "S",
        "DURABILITY": "D",
    })
    
    # If theme provided, add its specific mappings
    if theme_config:
        for attr, mapping in theme_config.attributes.items():
            # Map the themed label (e.g., "Experience") to source attr (e.g., "E")
            if mapping.label:
                attr_map[mapping.label.upper()] = attr
                # Also handle partial matches (e.g., "FOOD QUALITY" -> "FOOD" or "QUALITY")
                for word in mapping.label.upper().split():
                    if word not in attr_map:
                        attr_map[word] = attr
            # Map the themed name/code (e.g., "X") to source attr (e.g., "E")
            if mapping.name:
                attr_map[mapping.name.upper()] = attr
    
    return attr_map


def classify_premise_open_text(text: str, theme_config: ThemeConfig | None = None) -> dict[str, Any]:
    cleaned = text.lower()
    
    # Build theme-aware keywords
    keywords = _build_classification_keywords(theme_config)
    
    scores = {attr: 0 for attr in keywords}
    for attr, terms in keywords.items():
        for term in terms:
            if term in cleaned:
                scores[attr] += 1
    best_attr = max(scores.items(), key=lambda kv: kv[1])[0]
    total = sum(scores.values())
    confidence = scores[best_attr] / total if total else 0.0
    return {"attr": best_attr, "conf": confidence}


def _build_classification_keywords(theme_config: ThemeConfig | None) -> dict[str, list[str]]:
    """Build theme-aware keyword lists for open-text classification.
    
    Maps semantic keywords to source attributes (E, A, S, D).
    """
    # Base drug theme keywords (always included as fallback)
    keywords = {
        "E": ["efficacy", "effective", "response", "outcome", "benefit"],
        "A": ["adherence", "compliance", "follow", "consistent", "pills"],
        "S": ["safety", "risk", "side effect", "adverse", "harm"],
        "D": ["durability", "lasting", "maintain", "long-term", "sustain"],
    }
    
    if theme_config:
        # Add theme-specific keywords based on attribute labels
        theme_keywords = {
            "candidates": {
                "E": ["experience", "experienced", "years", "background", "worked"],
                "A": ["culture", "fit", "team", "values", "collaborative"],
                "S": ["technical", "skills", "coding", "programming", "expertise"],
                "D": ["communication", "communicate", "articulate", "present", "express"],
            },
            "restaurants": {
                "E": ["quality", "food", "taste", "flavor", "fresh", "delicious"],
                "A": ["value", "price", "affordable", "worth", "money", "cost"],
                "S": ["service", "staff", "attentive", "friendly", "waiter"],
                "D": ["ambiance", "atmosphere", "decor", "vibe", "setting"],
            },
        }
        
        if theme_config.name in theme_keywords:
            # Extend (don't replace) base keywords with theme-specific ones
            for attr, terms in theme_keywords[theme_config.name].items():
                keywords[attr] = list(set(keywords[attr] + terms))
        else:
            # For custom themes, add label words as keywords
            for attr, mapping in theme_config.attributes.items():
                if mapping.label:
                    label_words = [w.lower() for w in mapping.label.split()]
                    keywords[attr] = list(set(keywords.get(attr, []) + label_words))
    
    return keywords


def _normalize_levels(levels_obj: Any, attributes: Sequence[Attribute]) -> dict[Attribute, str]:
    if isinstance(levels_obj, dict):
        return {attr: str(levels_obj[attr]) for attr in attributes}
    if isinstance(levels_obj, str):
        data = json.loads(levels_obj)
        return {attr: str(data[attr]) for attr in attributes}
    if isinstance(levels_obj, (list, tuple)):
        return {attr: str(value) for attr, value in zip(attributes, levels_obj)}
    raise TypeError(f"Unsupported levels representation: {type(levels_obj)}")


def _parse_step(step: ConversationStep, response: str, theme_config: ThemeConfig | None = None) -> dict[str, Any]:
    if step.expects == "choice":
        return parse_choice(response)
    if step.expects == "premise":
        return parse_structured_premise(response, theme_config)
    if step.expects == "sentence":
        return {"ok": bool(response.strip()), "text": response.strip()}
    if step.expects == "scores4":
        return parse_scores4(response)
    if step.expects == "score1":
        return parse_score1(response)
    if step.expects == "pairwise6":
        return parse_pairwise6(response)
    if step.expects == "pairwise1":
        return parse_pairwise1(response)
    return {"ok": False, "unknown_step": step.expects}
