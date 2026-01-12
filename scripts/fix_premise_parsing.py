#!/usr/bin/env python3
"""
Fix premise attribute parsing in reasoning effort comparison results.

The original parsing only recognized drug attribute names (Efficacy, Adherence, etc.),
but themed datasets use different attribute labels (Experience, Culture Fit, etc.).
This script re-parses all premise responses using the correct theme mapping.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.themes import theme_from_dict, DRUGS_THEME


def load_theme_from_dataset(dataset_dir: Path):
    """Load theme configuration from dataset's MANIFEST.json."""
    manifest_path = dataset_dir / "MANIFEST.json"
    if not manifest_path.exists():
        return None
    
    try:
        with manifest_path.open("r") as f:
            manifest = json.load(f)
        
        if "theme" in manifest:
            return theme_from_dict(manifest["theme"])
        return None
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load theme from MANIFEST: {e}")
        return None


def build_theme_attr_map(theme_config) -> dict[str, str]:
    """Build a mapping from theme attribute labels to attribute codes.
    
    Returns a dict like:
    {
        "EXPERIENCE": "E",
        "E": "E",
        "CULTURE FIT": "A",
        "CULTUREFIT": "A",
        "A": "A",
        ...
    }
    """
    attr_map = {}
    
    # Add standard attribute codes
    attr_map.update({
        "E": "E",
        "A": "A",
        "S": "S",
        "D": "D",
    })
    
    # Add theme-specific mappings
    if theme_config:
        for attr_code, mapping in theme_config.attributes.items():
            # Map label (e.g., "Experience")
            label_upper = mapping.label.upper()
            attr_map[label_upper] = attr_code
            
            # Map label with spaces removed (e.g., "CULTUREFIT")
            label_no_space = label_upper.replace(" ", "")
            attr_map[label_no_space] = attr_code
            
            # Map short name if different
            if mapping.name.upper() != attr_code:
                attr_map[mapping.name.upper()] = attr_code
            
            # Map individual words in multi-word labels
            # e.g., "Culture Fit" -> both "CULTURE" and "FIT" map to "A"
            for word in label_upper.split():
                if len(word) > 2:  # Only map substantial words
                    attr_map[word] = attr_code
    
    return attr_map


def parse_premise_with_theme(text: str, attr_map: dict[str, str]) -> dict[str, Any]:
    """Parse premise response using theme-aware attribute mapping."""
    attr_code = "UNK"
    attr_ok = False
    text_ok = False
    body = text.strip()
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    premise_text = ""
    
    for line in lines:
        if line.lower().startswith("premiseattribute"):
            _, _, value = line.partition("=")
            token = value.strip().strip("[]<> ")
            token_upper = token.upper()
            
            # Try exact match
            if token_upper in attr_map:
                attr_code = attr_map[token_upper]
                attr_ok = True
            else:
                # Try cleaning and matching
                cleaned_token = token_upper.replace("]", "").replace("[", "").replace("<", "").replace(">", "")
                # Try matching cleaned token
                if cleaned_token in attr_map:
                    attr_code = attr_map[cleaned_token]
                    attr_ok = True
                else:
                    # Try splitting by common delimiters
                    pieces = [part.strip() for part in re.split(r"[|,/]+", cleaned_token) if part.strip()]
                    matched = {attr_map[part] for part in pieces if part in attr_map}
                    if len(matched) == 1:
                        attr_code = matched.pop()
                        attr_ok = True
                    else:
                        # Try matching first word
                        first_word = cleaned_token.split()[0] if cleaned_token.split() else ""
                        if first_word in attr_map:
                            attr_code = attr_map[first_word]
                            attr_ok = True
        
        if line.lower().startswith("premisetext"):
            _, _, value = line.partition("=")
            value = value.strip()
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                premise_text = value[1:-1].strip()
                text_ok = True
            elif value.startswith("'") and value.endswith("'") and len(value) >= 2:
                premise_text = value[1:-1].strip()
                text_ok = True
            else:
                premise_text = value.strip('"\'')
                text_ok = bool(premise_text)
    
    # If we didn't find attribute in structured format, try to extract from text
    if not attr_ok and body:
        # Try matching first word or phrase
        first_line = lines[0] if lines else ""
        first_line_upper = first_line.upper()
        
        # Try exact match of first word
        first_word = first_line_upper.split()[0] if first_line_upper.split() else ""
        if first_word in attr_map:
            attr_code = attr_map[first_word]
            attr_ok = True
        else:
            # Try matching any word in the first line
            for word in first_line_upper.split():
                if word in attr_map:
                    attr_code = attr_map[word]
                    attr_ok = True
                    break
        
        # If still not found, try matching anywhere in the text
        if not attr_ok:
            body_upper = body.upper()
            for keyword, code in attr_map.items():
                if len(keyword) > 2 and keyword in body_upper:
                    # Prefer longer matches
                    if not attr_ok or len(keyword) > len(attr_map.get(attr_code, "")):
                        attr_code = code
                        attr_ok = True
    
    # Extract premise text if not found in structured format
    if not text_ok and body and attr_ok:
        # Try to extract text after attribute
        first_line = lines[0] if lines else ""
        # If first line is just the attribute, get next line
        if first_line.upper().strip() in attr_map or first_line.upper().split()[0] in attr_map:
            if len(lines) > 1:
                premise_text = lines[1].strip()
                text_ok = bool(premise_text)
            else:
                # Try to extract text after attribute in same line
                parts = first_line.split(".", 1)
                if len(parts) > 1:
                    premise_text = parts[1].strip()
                    text_ok = bool(premise_text)
    
    return {"ok": attr_ok and text_ok, "attr": attr_code if attr_ok else "UNK", "text": premise_text}


def fix_responses_file(jsonl_path: Path, attr_map: dict[str, str], dry_run: bool = False) -> dict[str, int]:
    """Fix premise parsing in a responses JSONL file.
    
    Returns:
        dict with counts of fixed responses
    """
    if not jsonl_path.exists():
        print(f"  ⚠️  File not found: {jsonl_path}")
        return {"total": 0, "fixed": 0, "unchanged": 0}
    
    fixed_responses = []
    stats = {"total": 0, "fixed": 0, "unchanged": 0, "still_unk": 0}
    
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                response = json.loads(line)
                stats["total"] += 1
                modified = False
                
                for run in response.get("responses", []):
                    for step in run.get("steps", []):
                        if step.get("name") == "premise":
                            old_parsed = step.get("parsed", {})
                            old_attr = old_parsed.get("attr", "UNK")
                            old_ok = old_parsed.get("ok", False)
                            
                            # Re-parse with theme-aware mapping
                            content = step.get("content", "")
                            new_parsed = parse_premise_with_theme(content, attr_map)
                            
                            # Update if different
                            if new_parsed != old_parsed:
                                step["parsed"] = new_parsed
                                modified = True
                                
                                if old_attr == "UNK" and new_parsed["attr"] != "UNK":
                                    stats["fixed"] += 1
                                elif new_parsed["attr"] == "UNK":
                                    stats["still_unk"] += 1
                                # Note: if old_ok was True but we're changing it, we don't count as unchanged
                            elif old_ok:
                                # Truly unchanged - was OK and still is
                                stats["unchanged"] += 1
                
                fixed_responses.append(response)
                
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Error parsing line in {jsonl_path}: {e}")
                continue
    
    # Write back if not dry run and we made changes
    if not dry_run and stats["fixed"] > 0:
        with jsonl_path.open("w") as f:
            for resp in fixed_responses:
                f.write(json.dumps(resp) + "\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fix premise attribute parsing in reasoning effort comparison results"
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to results directory (e.g., results/reasoning_effort_comparison/candidates)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory to load theme from (e.g., data/generated/candidates)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes"
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    dataset_dir = Path(args.dataset)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Load theme
    print(f"📦 Loading theme from {dataset_dir}...")
    theme_config = load_theme_from_dataset(dataset_dir)
    if not theme_config:
        print(f"⚠️  No theme found, using default (drugs)")
        theme_config = DRUGS_THEME
    else:
        print(f"✅ Loaded theme: {theme_config.name}")
        for attr, mapping in theme_config.attributes.items():
            print(f"   {attr} -> {mapping.label}")
    
    # Build attribute mapping
    attr_map = build_theme_attr_map(theme_config)
    print(f"\n📋 Attribute mapping:")
    for key, val in sorted(attr_map.items()):
        if key != val:  # Only show non-trivial mappings
            print(f"   {key} -> {val}")
    
    # Find all response files
    response_files = list(results_dir.rglob("responses_*.jsonl"))
    
    if not response_files:
        print(f"\n❌ No response files found in {results_dir}")
        sys.exit(1)
    
    print(f"\n🔧 Processing {len(response_files)} response files...")
    if args.dry_run:
        print("   (DRY RUN - no files will be modified)")
    
    total_stats = {"total": 0, "fixed": 0, "unchanged": 0, "still_unk": 0}
    
    for resp_file in sorted(response_files):
        rel_path = resp_file.relative_to(results_dir)
        print(f"\n  📄 {rel_path}")
        stats = fix_responses_file(resp_file, attr_map, dry_run=args.dry_run)
        
        for key in total_stats:
            total_stats[key] += stats[key]
        
        if stats["fixed"] > 0:
            print(f"     ✅ Fixed {stats['fixed']} premise attributes")
        if stats["still_unk"] > 0:
            print(f"     ⚠️  {stats['still_unk']} still UNK after parsing")
        if stats["unchanged"] > 0:
            print(f"     ℹ️  {stats['unchanged']} already correct")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total responses: {total_stats['total']}")
    print(f"Fixed: {total_stats['fixed']}")
    print(f"Still UNK: {total_stats['still_unk']}")
    print(f"Already correct: {total_stats['unchanged']}")
    
    if args.dry_run:
        print(f"\n⚠️  DRY RUN - no files were modified")
        print(f"   Run without --dry-run to apply fixes")
    else:
        print(f"\n✅ Fixes applied! You may need to re-run analysis scripts.")


if __name__ == "__main__":
    main()

