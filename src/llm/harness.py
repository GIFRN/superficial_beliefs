from __future__ import annotations

import asyncio
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

    async def complete_async(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self.complete,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )


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
    return asyncio.run(
        run_trial_async(
            trial=trial,
            backend=backend,
            S=S,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )
    )


async def run_trial_async(
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

    replicate_seeds = [
        int(rng.integers(0, 2**32 - 1))
        for _ in range(S)
    ]

    async def run_replicate(replicate_seed: int) -> SingleRunResult:
        system_message = {"role": "system", "content": plan.system_prompt}
        conversation: list[dict[str, str]] = [system_message]
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
            response = await backend.complete_async(
                history,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=step_seed,
            )
            history.append({"role": "assistant", "content": response})
            conversation.append({"role": "assistant", "content": response})
            parsed = parse_step_response(step, response, trial.theme_config)
            result = StepResult(name=step.name, content=response, parsed=parsed)
            steps_results.append(result)
            context["steps"][step.name] = result
            if not parsed.get("ok") and step.stop_on_fail:
                break
        return SingleRunResult(seed=replicate_seed, steps=steps_results, conversation=conversation)

    runs = await asyncio.gather(*(run_replicate(rep_seed) for rep_seed in replicate_seeds))

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


def _match_choice_prefix(text: str) -> tuple[str | None, int]:
    body = text.strip()
    if not body:
        return None, 0

    patterns = [
        r"^\s*(?:choice|answer|selected|pick)\s*[:=\-]?\s*([AB])\b",
        r"^\s*(?:option|entity|drug|policy|library|candidate|restaurant)\s+([AB])\b",
        r"^\s*([AB])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, body, re.I)
        if match:
            return match.group(1).upper(), match.end()

    start_section = body[:40]
    match = re.search(r"\b([AB])\b", start_section, re.I)
    if match:
        return match.group(1).upper(), match.end()
    return None, 0


def _extract_attr_from_tail(text: str, theme_config: ThemeConfig | None = None) -> tuple[str | None, str]:
    attr_map = _build_attr_map(theme_config)
    body = text.strip()
    if not body:
        return None, ""

    # Remove leading scaffolding words the model may add before the attribute.
    body = re.sub(
        r"^(?:attribute|attr|factor|feature|reason|premise)\s*[:=\-]?\s*",
        "",
        body,
        flags=re.I,
    ).strip()
    if not body:
        return None, ""

    upper_body = body.upper()
    compact_body = re.sub(r"[^A-Z0-9]+", "", upper_body)
    aliases = sorted(set(attr_map.keys()), key=lambda alias: (-len(alias.replace(" ", "")), -len(alias), alias))

    for alias in aliases:
        alias_upper = alias.upper()
        pattern = r"(?<![A-Z0-9])" + re.escape(alias_upper) + r"(?![A-Z0-9])"
        if re.search(pattern, upper_body):
            return attr_map[alias], alias_upper
        alias_compact = re.sub(r"[^A-Z0-9]+", "", alias_upper)
        if alias_compact and compact_body.startswith(alias_compact):
            return attr_map[alias], alias_upper

    def _common_prefix_len(a: str, b: str) -> int:
        n = min(len(a), len(b))
        idx = 0
        while idx < n and a[idx] == b[idx]:
            idx += 1
        return idx

    fuzzy_matches: list[tuple[int, str, str]] = []
    for alias in aliases:
        alias_upper = alias.upper()
        alias_compact = re.sub(r"[^A-Z0-9]+", "", alias_upper)
        if not alias_compact or len(alias_compact) < 6 or len(compact_body) < 6:
            continue
        prefix_len = _common_prefix_len(compact_body, alias_compact)
        if prefix_len >= 6:
            fuzzy_matches.append((prefix_len, attr_map[alias], alias_upper))
    if fuzzy_matches:
        fuzzy_matches.sort(key=lambda row: (-row[0], row[1], row[2]))
        best = fuzzy_matches[0]
        tied = [row for row in fuzzy_matches if row[0] == best[0]]
        if len({row[1] for row in tied}) == 1:
            return best[1], body

    inferred = classify_premise_open_text(body, theme_config)
    if inferred.get("conf", 0.0) > 0:
        return inferred.get("attr"), body
    return None, ""


def parse_choice_attr(text: str, theme_config: ThemeConfig | None = None) -> dict[str, Any]:
    body = text.strip()
    if not body:
        return {
            "ok": False,
            "choice_ok": False,
            "choice": None,
            "premise_ok": False,
            "attr": None,
            "text": "",
        }

    choice, choice_end = _match_choice_prefix(body)
    choice_ok = choice in {"A", "B"}

    attr = None
    attr_text = ""
    if choice_ok:
        tail = body[choice_end:]
        tail = tail.lstrip(" \t\r\n.:;,-")
        attr, attr_text = _extract_attr_from_tail(tail, theme_config)
    else:
        attr, attr_text = _extract_attr_from_tail(body, theme_config)

    premise_ok = attr in {"E", "A", "S", "D"}
    return {
        # Keep step success keyed to whether a choice was recovered, so judge
        # steps still run even if the attribute token is malformed.
        "ok": choice_ok,
        "choice_ok": choice_ok,
        "choice": choice,
        "premise_ok": premise_ok,
        "attr": attr,
        "text": attr_text.strip(),
    }


def _normalize_attr_key(token: Any, theme_config: ThemeConfig | None = None) -> str | None:
    if token is None:
        return None
    attr_map = _build_attr_map(theme_config)
    raw = str(token).strip()
    if not raw:
        return None
    raw = re.sub(r"[\[\](){}\"']", "", raw).strip()
    raw_upper = raw.upper()
    if raw_upper in attr_map:
        return attr_map[raw_upper]
    raw_compact = re.sub(r"[^A-Z0-9]+", "", raw_upper)
    if raw_compact in attr_map:
        return attr_map[raw_compact]
    return None


def _visible_attr_keys_from_prompt(prompt: str, theme_config: ThemeConfig | None = None) -> list[str]:
    keys = ["E", "A", "S", "D"]
    if not prompt:
        return keys
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("attributes:"):
            continue
        _, _, payload = line.partition(":")
        visible: list[str] = []
        for token in payload.split(","):
            key = _normalize_attr_key(token.strip(), theme_config)
            if key in keys and key not in visible:
                visible.append(key)
        if visible:
            return visible
    return keys


def parse_scores4(
    text: str,
    theme_config: ThemeConfig | None = None,
    visible_attrs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Parse four tau scores for E/A/S/D from JSON or key=value lines."""
    keys = ["E", "A", "S", "D"]
    visible_keys = [key for key in (visible_attrs or keys) if key in keys]
    if not visible_keys:
        visible_keys = keys[:]
    values: dict[str, float] = {}
    body = text.strip()
    if not body:
        return {"ok": False, "tau": {}, "missing": keys}

    def _coerce_tau_value(raw: Any) -> float | None:
        if isinstance(raw, (int, float)):
            val = float(raw)
            return val if 0.0 <= val <= 1.0 else None
        if isinstance(raw, str):
            cleaned = raw.strip().strip('"').strip("'")
            number_tokens = re.findall(r"([0-9]*\.?[0-9]+)", cleaned)
            if not number_tokens:
                return None
            if len(number_tokens) > 1:
                try:
                    vals = [float(token) for token in number_tokens]
                except ValueError:
                    return None
                if all(0.0 <= val <= 1.0 for val in vals) and max(vals) - min(vals) <= 1e-9:
                    return vals[0]
                return None
            try:
                val = float(number_tokens[0])
            except ValueError:
                return None
            return val if 0.0 <= val <= 1.0 else None
        return None

    def _split_top_level_segments(raw: str) -> list[str]:
        segments: list[str] = []
        current: list[str] = []
        depth = 0
        for char in raw:
            if char in "[({<":
                depth += 1
            elif char in "])}>":
                depth = max(0, depth - 1)
            if depth == 0 and char in ",\n":
                segment = "".join(current).strip()
                if segment:
                    segments.append(segment)
                current = []
                continue
            current.append(char)
        tail = "".join(current).strip()
        if tail:
            segments.append(tail)
        return segments

    def _consume_kv_segment(segment: str) -> None:
        if "=" in segment:
            key_text, value_text = segment.split("=", 1)
        elif ":" in segment:
            key_text, value_text = segment.split(":", 1)
        else:
            return
        key = _normalize_attr_key(key_text.strip().lstrip("-*0123456789. )("), theme_config)
        if key not in keys or key in values:
            return
        val = _coerce_tau_value(value_text)
        if val is not None:
            values[key] = val

    json_match = re.search(r"\{.*\}", body, flags=re.S)
    if json_match:
        snippet = json_match.group(0)
        try:
            payload = json.loads(snippet)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key, value in payload.items():
                norm_key = _normalize_attr_key(key, theme_config)
                if norm_key in keys:
                    val = _coerce_tau_value(value)
                    if val is not None:
                        values[norm_key] = val

    for segment in _split_top_level_segments(body):
        _consume_kv_segment(segment)

    # Canonical compact forms, tolerant to quoted numeric values.
    pattern = re.compile(r"\b([EASD])\s*[:=]\s*['\"]?\s*([0-9]*\.?[0-9]+)\s*['\"]?", re.I)
    for match in pattern.finditer(body):
        key = match.group(1).upper()
        if key in values:
            continue
        val = _coerce_tau_value(match.group(2))
        if val is not None:
            values[key] = val

    line_pattern = re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9 _-]{0,40})\s*[:=]\s*['\"]?\s*([0-9]*\.?[0-9]+)\s*[\])'\"]?\s*$",
        re.I,
    )
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        _consume_kv_segment(line)
        match = line_pattern.match(line)
        if not match:
            continue
        key = _normalize_attr_key(match.group(1), theme_config)
        if key not in keys or key in values:
            continue
        val = _coerce_tau_value(match.group(2))
        if val is not None:
            values[key] = val

    generic_pattern = re.compile(
        r"([A-Za-z][A-Za-z0-9 _-]{0,40})\s*[:=]\s*['\"]?\s*([0-9]*\.?[0-9]+)\s*[\])'\"]?",
        re.I,
    )
    for match in generic_pattern.finditer(body):
        key = _normalize_attr_key(match.group(1), theme_config)
        if key not in keys or key in values:
            continue
        val = _coerce_tau_value(match.group(2))
        if val is not None:
            values[key] = val

    # Narrative forms, e.g. "(E): Tau ≈ 0.8".
    narrative = re.compile(
        r"(?:\b([EASD])\b|\(([EASD])\))\s*[:=]\s*(?:tau\s*(?:≈|~|:|=)\s*)?['\"]?\s*([0-9]*\.?[0-9]+)\s*['\"]?",
        re.I,
    )
    for match in narrative.finditer(body):
        key = (match.group(1) or match.group(2) or "").upper()
        if key not in keys or key in values:
            continue
        val = _coerce_tau_value(match.group(3))
        if val is not None:
            values[key] = val

    for key in keys:
        if key not in visible_keys:
            values.setdefault(key, 0.0)

    missing = [key for key in keys if key not in values]
    ok = len(values) == len(keys)
    return {"ok": ok, "tau": values, "missing": missing}


def parse_score1(text: str) -> dict[str, Any]:
    """Parse a single tau score in [0,1]."""
    body = text.strip()
    if not body:
        return {"ok": False, "tau": None}
    match = re.search(r"\btau\s*[:=]\s*['\"]?\s*([0-9]*\.?[0-9]+)\s*['\"]?", body, re.I)
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
    
    def _de_md(value: str) -> str:
        # Remove light markdown wrappers often produced by models.
        value = re.sub(r"[*_`]", "", value)
        return value.strip()

    def _clean_attr_token(value: str) -> str:
        value = _de_md(value)
        value = re.sub(r"^[\[\](){}<>\"'=\s:.-]+", "", value)
        value = re.sub(r"[\[\](){}<>\"'\s:.;,!?-]+$", "", value)
        return value.strip().upper()

    def _clean_text_value(value: str) -> str:
        value = _de_md(value)
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        return value.strip()

    premise_text = ""
    attr_line_idx = -1
    for idx, raw_line in enumerate(lines):
        line = _de_md(raw_line)
        lower = line.lower()
        if lower.startswith("premiseattribute") or lower.startswith("premise attribute"):
            attr_line_idx = idx
            if "=" in line:
                _, _, value = line.partition("=")
            elif ":" in line:
                _, _, value = line.partition(":")
            else:
                parts = line.split(maxsplit=1)
                value = parts[1] if len(parts) > 1 else ""
            token_upper = _clean_attr_token(value)
            if token_upper in attr_map:
                attr_code = attr_map[token_upper]
                attr_ok = True
            else:
                token_no_space = token_upper.replace(" ", "")
                if token_no_space in attr_map:
                    attr_code = attr_map[token_no_space]
                    attr_ok = True
                else:
                    cleaned_token = re.sub(r"[\[\](){}<>]", "", token_upper)
                    pieces = [part.strip() for part in re.split(r"[|,/]+", cleaned_token) if part.strip()]
                    # Only accept if exactly one recognizable attribute token is present
                    matched = {attr_map[part] for part in pieces if part in attr_map}
                    if len(matched) == 1:
                        attr_code = matched.pop()
                        attr_ok = True
                    else:
                        words = [w.strip() for w in cleaned_token.split() if w.strip()]
                        matched_words = {attr_map[w] for w in words if w in attr_map}
                        if len(matched_words) == 1:
                            attr_code = matched_words.pop()
                            attr_ok = True
        if (
            lower.startswith("premisetext")
            or lower.startswith("premise text")
            or lower.startswith("preminetext")
            or lower.startswith("premine text")
        ):
            if "=" in line:
                _, _, value = line.partition("=")
            elif ":" in line:
                _, _, value = line.partition(":")
            else:
                parts = line.split(maxsplit=1)
                value = parts[1] if len(parts) > 1 else ""
            premise_text = _clean_text_value(value)
            text_ok = bool(premise_text)

    # Common fallback: attribute line present but text provided on the next line.
    if not text_ok and attr_line_idx >= 0 and attr_line_idx + 1 < len(lines):
        next_line = _de_md(lines[attr_line_idx + 1])
        if not next_line.lower().startswith("premise"):
            if "=" in next_line and "text" in next_line.lower():
                _, _, value = next_line.partition("=")
                premise_text = _clean_text_value(value)
            elif ":" in next_line and "text" in next_line.lower():
                _, _, value = next_line.partition(":")
                premise_text = _clean_text_value(value)
            else:
                premise_text = _clean_text_value(next_line)
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
                label_upper = mapping.label.upper()
                attr_map[label_upper] = attr
                attr_map[label_upper.replace(" ", "")] = attr
                if label_upper == "MAINTAINABILITY":
                    attr_map["MAINTABILITY"] = attr
                # Also handle partial matches (e.g., "FOOD QUALITY" -> "FOOD" or "QUALITY")
                for word in label_upper.split():
                    if word not in attr_map:
                        attr_map[word] = attr
            # Map the themed name/code (e.g., "X") to source attr (e.g., "E")
            if mapping.name:
                name_upper = mapping.name.upper()
                attr_map[name_upper] = attr
                attr_map[name_upper.replace(" ", "")] = attr
    
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


def parse_step_response(step: ConversationStep, response: str, theme_config: ThemeConfig | None = None) -> dict[str, Any]:
    if step.expects == "choice":
        return parse_choice(response)
    if step.expects == "choice_attr":
        return parse_choice_attr(response, theme_config)
    if step.expects == "premise":
        return parse_structured_premise(response, theme_config)
    if step.expects == "sentence":
        return {"ok": bool(response.strip()), "text": response.strip()}
    if step.expects == "scores4":
        visible_attrs = _visible_attr_keys_from_prompt(step.prompt, theme_config)
        return parse_scores4(response, theme_config, visible_attrs=visible_attrs)
    if step.expects == "score1":
        return parse_score1(response)
    if step.expects == "pairwise6":
        return parse_pairwise6(response)
    if step.expects == "pairwise1":
        return parse_pairwise1(response)
    return {"ok": False, "unknown_step": step.expects}


def _parse_step(step: ConversationStep, response: str, theme_config: ThemeConfig | None = None) -> dict[str, Any]:
    """Backward-compatible alias for older imports."""
    return parse_step_response(step, response, theme_config)
