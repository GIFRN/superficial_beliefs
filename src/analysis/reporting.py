from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir


def make_report(run_dir: str | Path, out_dir: str | Path, artifacts: dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    out_dir = ensure_dir(out_dir)
    report_path = out_dir / "report.md"
    stageA = artifacts.get("stageA", {})
    stageB = artifacts.get("stageB", {})
    diagnostics = artifacts.get("diagnostics", {})

    lines = [
        f"# Superficial Beliefs Run Summary", "",
        f"**Run directory:** `{run_dir}`", "",
    ]

    if stageA:
        lines.extend(_stageA_section(stageA))
    if stageB:
        lines.extend(_stageB_section(stageB))
    if diagnostics:
        lines.extend(_diagnostics_section(diagnostics))
    
    # Add B1 validation section if available
    b1_validation = artifacts.get("b1_validation")
    b1_probes = artifacts.get("b1_probes")
    if b1_validation or b1_probes:
        lines.extend(_b1_validation_section(b1_validation, b1_probes))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _stageA_section(stageA: dict[str, Any]) -> list[str]:
    weights = stageA.get("weights", {})
    beta = stageA.get("beta", {})
    lines = ["## Stage A", "", "| Attribute | Weight | Beta |", "| --- | --- | --- |"]
    for attr in ["E", "A", "S", "D"]:
        lines.append(f"| {attr} | {weights.get(attr, 0):.3f} | {beta.get(attr, 0):.3f} |")
    lines.append("")
    if stageA.get("cv"):
        cv = stageA["cv"]
        lines.append("**Cross-validation metrics**")
        lines.append("")
        lines.extend([f"- log_loss: {cv.get('log_loss', float('nan')):.4f}", f"- brier: {cv.get('brier', float('nan')):.4f}", f"- accuracy: {cv.get('accuracy', float('nan')):.4f}"])
        lines.append("")
    return lines


def _stageB_section(stageB: dict[str, Any]) -> list[str]:
    lines = ["## Stage B", ""]
    alignment = stageB.get("alignment", {})
    if alignment:
        lines.extend(
            [
                f"- ECRB_top1_driver: {alignment.get('ECRB_top1_driver', float('nan')):.3f}",
                f"- ECRB_top1_weights: {alignment.get('ECRB_top1_weights', float('nan')):.3f}",
                f"- Rank correlation: {alignment.get('rank_corr', float('nan')):.3f}",
                "",
            ]
        )
    probes = stageB.get("probes", {})
    for manip, stats in probes.items():
        if manip == "baseline":
            continue
        lines.append(f"### Probe: {manip}")
        delta = stats.get("delta_beta", {})
        for attr, value in delta.items():
            lines.append(f"- Δβ_{attr}: {value:.3f}")
        lines.append("")
    return lines


def _diagnostics_section(diagnostics: dict[str, Any]) -> list[str]:
    lines = ["## Diagnostics", ""]
    for key, value in diagnostics.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return lines


def _b1_validation_section(b1_validation: dict[str, Any] | None, b1_probes: dict[str, Any] | None) -> list[str]:
    """Add B1 validation section to report"""
    lines = ["## B1 Rationality Validation", ""]
    
    if b1_validation:
        if b1_validation.get("rationality_check_passed", False):
            lines.append("✅ **B1 Rationality Check PASSED**")
            lines.append(f"- All B1 trials show P(choose A) ≥ 0.95")
            lines.append(f"- Min probability: {b1_validation.get('min_probability', 0):.3f}")
            lines.append(f"- Max probability: {b1_validation.get('max_probability', 0):.3f}")
            lines.append(f"- Total B1 trials: {b1_validation.get('total_b1_trials', 0)}")
        else:
            lines.append("❌ **B1 Rationality Check FAILED**")
            lines.append(f"- Failure rate: {b1_validation.get('failure_rate', 0):.1%}")
            lines.append(f"- Failed trials: {len(b1_validation.get('failed_trials', []))}")
            lines.append(f"- Min probability: {b1_validation.get('min_probability', 0):.3f}")
            lines.append(f"- Max probability: {b1_validation.get('max_probability', 0):.3f}")
        
        if b1_validation.get('message'):
            lines.append(f"- {b1_validation['message']}")
    
    lines.append("")
    
    if b1_probes:
        lines.append("### B1 Probe Validation")
        if b1_probes.get("probe_effectiveness", False):
            lines.append("✅ **B1 Probes EFFECTIVE**")
            lines.append(f"- Baseline probability: {b1_probes.get('baseline_probability', 0):.3f}")
            lines.append(f"- Manipulated probability: {b1_probes.get('manipulated_probability', 0):.3f}")
            lines.append(f"- Probe effect size: {b1_probes.get('probe_effect_size', 0):.3f}")
        else:
            lines.append("❌ **B1 Probes INEFFECTIVE**")
            lines.append(f"- Baseline probability: {b1_probes.get('baseline_probability', 0):.3f}")
            lines.append(f"- Manipulated probability: {b1_probes.get('manipulated_probability', 0):.3f}")
            lines.append(f"- Probe effect size: {b1_probes.get('probe_effect_size', 0):.3f}")
            lines.append("- Probes may not be affecting model behavior as expected")
        
        if b1_probes.get('message'):
            lines.append(f"- {b1_probes['message']}")
    
    lines.append("")
    return lines
