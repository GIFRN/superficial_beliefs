from __future__ import annotations

from typing import Any, Sequence

from src.data.occlusions import apply_structural_occlusion
from src.data.paraphrases import render_profile
from src.data.schema import ATTR_LABELS
from src.data.themes import ThemeConfig, DRUGS_THEME

from .types import ConversationPlan, ConversationStep, TrialSpec


ATTR_ORDER = tuple(ATTR_LABELS.keys())
PAIRWISE_PAIRS = (
    ("E", "A"),
    ("E", "S"),
    ("E", "D"),
    ("A", "S"),
    ("A", "D"),
    ("S", "D"),
)


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
    if trial.variant.startswith("short_reason__"):
        plan = render_short_reason(trial)
        suffix = trial.variant.split("__", 1)[1]
        plan.steps.extend(_judge_steps(trial, suffix))
        return plan
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


def _judge_steps(trial: TrialSpec, suffix: str) -> list[ConversationStep]:
    if suffix == "judge_scores_joint":
        return [_judge_scores_joint_step(trial)]
    if suffix == "judge_scores_per_feature":
        return _judge_scores_per_feature_steps(trial)
    if suffix == "judge_pairwise_joint":
        return [_judge_pairwise_joint_step(trial)]
    if suffix == "judge_pairwise_stepwise":
        return _judge_pairwise_stepwise_steps(trial)
    return []


def _judge_scores_joint_step(trial: TrialSpec) -> ConversationStep:
    return ConversationStep(
        name="judge_scores_joint",
        prompt=_judge_scores_joint_prompt(trial),
        expects="scores4",
        reset_context=True,
        stop_on_fail=False,
    )


def _judge_scores_per_feature_steps(trial: TrialSpec) -> list[ConversationStep]:
    steps: list[ConversationStep] = []
    for attr in ATTR_ORDER:
        if trial.manipulation == "occlude_drop" and trial.attribute_target == attr:
            continue
        steps.append(
            ConversationStep(
                name=f"judge_score_{attr}",
                prompt=_judge_score_prompt(trial, attr),
                expects="score1",
                reset_context=True,
                stop_on_fail=False,
            )
        )
    return steps


def _judge_pairwise_joint_step(trial: TrialSpec) -> ConversationStep:
    return ConversationStep(
        name="judge_pairwise_joint",
        prompt=_judge_pairwise_joint_prompt(trial),
        expects="pairwise6",
        reset_context=True,
        stop_on_fail=False,
    )


def _judge_pairwise_stepwise_steps(trial: TrialSpec) -> list[ConversationStep]:
    steps: list[ConversationStep] = []
    for attr_a, attr_b in PAIRWISE_PAIRS:
        steps.append(
            ConversationStep(
                name=f"judge_pair_{attr_a}{attr_b}",
                prompt=_judge_pairwise_step_prompt(trial, attr_a, attr_b),
                expects="pairwise1",
                reset_context=True,
                stop_on_fail=False,
            )
        )
    return steps


def _judge_scores_joint_prompt(trial: TrialSpec) -> str:
    theme = trial.theme_config or DRUGS_THEME
    labels = _attribute_code_label_string(trial)
    lines = [
        "You are an evaluator. Do NOT choose A/B.",
        f"Score how decisive each attribute difference is for {theme.objective}.",
        "Use tau in [0,1], where 0 = no effect and 1 = fully decisive.",
        "For each attribute, imagine all other attributes equal; only that attribute differs as shown.",
        f"Attributes: {labels}.",
        "If an attribute is not shown, set its tau to 0.",
        "",
        _profile_text(trial, label=theme.entity_a),
        _profile_text(trial, label=theme.entity_b),
        "",
        "Return 4 lines: E=..., A=..., S=..., D=... (or strict JSON).",
    ]
    return "\n".join(lines)


def _judge_score_prompt(trial: TrialSpec, attr: str) -> str:
    theme = trial.theme_config or DRUGS_THEME
    label = theme.get_attribute_label(attr)
    profile_a, profile_b, _, _ = apply_structural_occlusion(
        trial.profile_a,
        trial.profile_b,
        trial.order_a,
        trial.order_b,
        trial.manipulation,
        trial.attribute_target,
    )
    level_a = profile_a.levels.get(attr, "Unknown")
    level_b = profile_b.levels.get(attr, "Unknown")
    lines = [
        "You are an evaluator. Do NOT choose A/B.",
        f"Score how decisive the {label} difference (code {attr}) is for {theme.objective}.",
        "Assume all other attributes are equal; only this attribute differs as shown.",
        "Use tau in [0,1], where 0 = no effect and 1 = fully decisive.",
        f"{theme.entity_a}: {label} = {level_a}",
        f"{theme.entity_b}: {label} = {level_b}",
        "Return: tau=...",
    ]
    return "\n".join(lines)


def _judge_pairwise_joint_prompt(trial: TrialSpec) -> str:
    theme = trial.theme_config or DRUGS_THEME
    labels = _attribute_code_label_string(trial)
    lines = [
        "You are an evaluator. Do NOT choose A/B.",
        f"For each attribute pair, decide which attribute difference is more decisive for {theme.objective} (in either direction), or tie.",
        f"Attributes: {labels}.",
        "If an attribute is not shown, treat it as neutral and answer tie for pairs involving it.",
        "",
        _profile_text(trial, label=theme.entity_a),
        _profile_text(trial, label=theme.entity_b),
        "",
        "Return 6 lines:",
        "EA=E|A|tie, ES=E|S|tie, ED=E|D|tie, AS=A|S|tie, AD=A|D|tie, SD=S|D|tie",
    ]
    return "\n".join(lines)


def _judge_pairwise_step_prompt(trial: TrialSpec, attr_a: str, attr_b: str) -> str:
    theme = trial.theme_config or DRUGS_THEME
    label_a = theme.get_attribute_label(attr_a)
    label_b = theme.get_attribute_label(attr_b)
    lines = [
        "You are an evaluator. Do NOT choose A/B.",
        f"Between {label_a} (code {attr_a}) and {label_b} (code {attr_b}), which attribute difference is more decisive for {theme.objective} (in either direction), or tie?",
        "If either attribute is not shown, answer tie.",
        "",
        _profile_text(trial, label=theme.entity_a),
        _profile_text(trial, label=theme.entity_b),
        "",
        f"Return: winner={attr_a}|{attr_b}|tie",
    ]
    return "\n".join(lines)


def _attribute_code_label_string(trial: TrialSpec) -> str:
    theme = trial.theme_config or DRUGS_THEME
    attrs = [attr for attr in ATTR_ORDER if attr in theme.get_mapped_attributes()]
    pairs = [f"{attr}={theme.get_attribute_label(attr)}" for attr in attrs]
    return ", ".join(pairs)


def apply_probe(message: str, trial: TrialSpec) -> str:
    instruction = _probe_instruction(trial)
    if not instruction:
        return message
    return f"{instruction}\n\n{message}" if message else instruction


def _profile_text(trial: TrialSpec, label: str) -> str:
    theme = trial.theme_config or DRUGS_THEME
    profile_a, profile_b, order_a, order_b = apply_structural_occlusion(
        trial.profile_a,
        trial.profile_b,
        trial.order_a,
        trial.order_b,
        trial.manipulation,
        trial.attribute_target,
    )
    if label == theme.entity_a:
        profile = profile_a
        order = order_a
    else:
        profile = profile_b
        order = order_b
    template_id = trial.paraphrase_id
    return render_profile(profile, template_id, order, label, theme_config=theme)


def _probe_instruction(trial: TrialSpec) -> str:
    manip = trial.manipulation
    target = trial.attribute_target
    theme = trial.theme_config or DRUGS_THEME

    if manip in {
        "premise_first",
        "short_reason",
        "split_reason",
        "occlude_drop",
        "occlude_equalize",
        "occlude_swap",
    }:
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
    attrs = theme.get_mapped_attributes()
    if trial.manipulation == "occlude_drop" and trial.attribute_target in attrs:
        attrs = [attr for attr in attrs if attr != trial.attribute_target]
    labels = [theme.get_attribute_label(attr) for attr in attrs]
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
