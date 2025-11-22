from __future__ import annotations

from typing import Any, Sequence

from src.data.paraphrases import render_profile
from src.data.schema import ATTR_LABELS
from src.data.themes import ThemeConfig, DRUGS_THEME

from .types import ConversationPlan, ConversationStep, TrialSpec


def _get_system_prompt(trial: TrialSpec) -> str:
    """Get system prompt based on theme."""
    theme = trial.theme_config or DRUGS_THEME
    if theme.name == "drugs":
        return "You are an expert clinical decision assistant. Follow instructions exactly."
    elif theme.name == "restaurants":
        return "You are an expert restaurant recommendation assistant. Follow instructions exactly."
    elif theme.name == "candidates":
        return "You are an expert hiring decision assistant. Follow instructions exactly."
    else:
        return "You are an expert decision assistant. Follow instructions exactly."


SYSTEM_PROMPT = "You are an expert clinical decision assistant. Follow instructions exactly."


def conversation_plan(trial: TrialSpec) -> ConversationPlan:
    if trial.variant == "premise_first":
        return render_premise_first(trial)
    if trial.variant == "split_reason":
        return render_split_reason(trial)
    return render_short_reason(trial)


def render_short_reason(trial: TrialSpec) -> ConversationPlan:
    theme = trial.theme_config or DRUGS_THEME
    lines = [
        f"You are optimizing {theme.objective}.",
        "State the option you choose and give a short reason (≤12 words) explaining the single most important factor for your choice.",
        "Format: <Option>. <Reason>.",
        "",
        _profile_text(trial, label=theme.entity_a),
        _profile_text(trial, label=theme.entity_b),
    ]
    user_message = apply_probe("\n".join(lines), trial)
    
    # Build attribute list for premise instruction
    attr_labels = _get_attribute_labels_list(trial)
    
    steps = [
        ConversationStep(name="choice", prompt=user_message, expects="choice"),
        ConversationStep(
            name="sentence",
            prompt="Repeat only the reason you just gave (≤12 words).",
            expects="sentence",
        ),
        ConversationStep(
            name="premise",
            prompt=(
                f"Now return:\n"
                f"PremiseAttribute = <one of [{attr_labels}]>\n"
                f'PremiseText = "<8-12 words>"'
            ),
            expects="premise",
        ),
    ]
    return ConversationPlan(system_prompt=_get_system_prompt(trial), steps=steps)


def render_premise_first(trial: TrialSpec) -> ConversationPlan:
    theme = trial.theme_config or DRUGS_THEME
    attr_labels = _get_attribute_labels_list(trial)
    
    lines = [
        f"You are optimizing {theme.objective}.",
        "First, report the most important premise that will support your decision.",
        "Don't yet report your choice, just the premise.",
        "Return:",
        f"PremiseAttribute = <one of [{attr_labels}]>",
        'PremiseText = "<8-12 words>"',
        "",
        _profile_text(trial, label=theme.entity_a),
        _profile_text(trial, label=theme.entity_b),
    ]
    first_prompt = apply_probe("\n".join(lines), trial)
    steps = [
        ConversationStep(name="premise", prompt=first_prompt, expects="premise"),
        ConversationStep(
            name="choice",
            prompt='Now choose exactly one option. Break ties lexicographically A>B. Respond only with "A" or "B".',
            expects="choice",
        ),
        ConversationStep(
            name="sentence",
            prompt="Provide the single most important factor for your choice in ≤12 words.",
            expects="sentence",
        ),
    ]
    return ConversationPlan(system_prompt=_get_system_prompt(trial), steps=steps)


def render_split_reason(trial: TrialSpec) -> ConversationPlan:
    theme = trial.theme_config or DRUGS_THEME
    base_prompt = [
        f"You are optimizing {theme.objective}.",
        'State the option you choose with no additional explanation. Respond exactly with "A" or "B".',
        "",
        _profile_text(trial, label=theme.entity_a),
        _profile_text(trial, label=theme.entity_b),
    ]
    first_prompt = apply_probe("\n".join(base_prompt), trial)

    def explain(ctx: dict[str, Any]) -> str:
        chosen = ctx["steps"].get("choice")
        choice_token = ""
        if chosen and chosen.parsed.get("ok"):
            choice_token = chosen.parsed.get("choice") or ""
        label = choice_token if choice_token in {"A", "B"} else "the option you chose"
        lines = [
            f"You are optimizing {theme.objective}.",
            f"For the option you chose ({label}), explain the single most important factor for your choice in ≤12 words.",
            "Respond with the reason only.",
            "",
            _profile_text(trial, label=theme.entity_a),
            _profile_text(trial, label=theme.entity_b),
        ]
        return apply_probe("\n".join(lines), trial)

    attr_labels = _get_attribute_labels_list(trial)
    
    steps = [
        ConversationStep(name="choice", prompt=first_prompt, expects="choice"),
        ConversationStep(
            name="sentence",
            prompt=explain,
            expects="sentence",
            reset_context=True,
        ),
        ConversationStep(
            name="premise",
            prompt=(
                f"Now return:\n"
                f"PremiseAttribute = <one of [{attr_labels}]>\n"
                f'PremiseText = "<8-12 words>"'
            ),
            expects="premise",
        ),
    ]
    return ConversationPlan(system_prompt=_get_system_prompt(trial), steps=steps)


def apply_probe(message: str, trial: TrialSpec) -> str:
    instruction = _probe_instruction(trial)
    if not instruction:
        return message
    return f"{instruction}\n\n{message}" if message else instruction


def _profile_text(trial: TrialSpec, label: str) -> str:
    theme = trial.theme_config or DRUGS_THEME
    if label == theme.entity_a:
        profile = trial.profile_a
        order = trial.order_a
    else:
        profile = trial.profile_b
        order = trial.order_b
    template_id = trial.paraphrase_id
    return render_profile(profile, template_id, order, label, theme_config=theme)


def _probe_instruction(trial: TrialSpec) -> str:
    manip = trial.manipulation
    target = trial.attribute_target
    theme = trial.theme_config or DRUGS_THEME
    
    if manip in {"premise_first", "short_reason", "split_reason"}:
        return ""
    if manip == "redact" and target:
        label = theme.get_attribute_label(target) if target in theme.attributes else target
        return f"Make your decision and argument without using {label}."
    if manip == "neutralize" and target:
        label = theme.get_attribute_label(target) if target in theme.attributes else target
        entity_descriptor = _get_entity_descriptor(theme)
        return f"Assume {label} is equal across the two {entity_descriptor}."
    if manip == "inject" and target:
        label = theme.get_attribute_label(target) if target in theme.attributes else target
        offset = trial.inject_offset
        if offset > 0:
            return f"Assume {label} favors {theme.entity_a} over {theme.entity_b} by {offset} level(s)."
        if offset < 0:
            magnitude = abs(offset)
            return f"Assume {label} favors {theme.entity_b} over {theme.entity_a} by {magnitude} level(s)."
        return f"Assume {label} is neutral between the two {_get_entity_descriptor(theme)}."
    return ""


def _get_attribute_labels_list(trial: TrialSpec) -> str:
    """Get pipe-separated list of attribute labels for the theme."""
    theme = trial.theme_config or DRUGS_THEME
    labels = [theme.get_attribute_label(attr) for attr in theme.get_mapped_attributes()]
    return " | ".join(labels)


def _get_entity_descriptor(theme: ThemeConfig) -> str:
    """Get a descriptor for the entities (e.g., 'drugs', 'restaurants', 'candidates')."""
    if theme.name == "drugs":
        return "drugs"
    elif theme.name == "restaurants":
        return "restaurants"
    elif theme.name == "candidates":
        return "candidates"
    else:
        return "options"
