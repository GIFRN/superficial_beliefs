#!/usr/bin/env python3
"""Validate consistency between reasoning sentences and extracted attributes.

This script checks if the attribute mentioned in the reasoning sentence
consistently matches the attribute extracted in the premise step.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Attribute name mappings
ATTR_TO_FULL_NAME = {
    "E": ["efficacy", "effectiveness", "effective"],
    "A": ["adherence", "compliance", "patient compliance"],
    "S": ["safety", "safe", "tolerability", "tolerable"],
    "D": ["durability", "duration", "lasting", "sustained"]
}

FULL_NAME_TO_ATTR = {}
for attr, names in ATTR_TO_FULL_NAME.items():
    for name in names:
        FULL_NAME_TO_ATTR[name.lower()] = attr


def load_responses(responses_path: Path) -> list[dict]:
    """Load all responses from a JSONL file."""
    responses = []
    with responses_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    responses.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return responses


def extract_attribute_from_text(text: str) -> str | None:
    """Extract which attribute is mentioned in the text based on keywords."""
    text_lower = text.lower()
    
    # Count mentions of each attribute
    attr_scores = Counter()
    
    for full_name, attr in FULL_NAME_TO_ATTR.items():
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(full_name) + r'\b'
        matches = len(re.findall(pattern, text_lower))
        if matches > 0:
            attr_scores[attr] += matches
    
    if not attr_scores:
        return None
    
    # Return the most mentioned attribute
    return attr_scores.most_common(1)[0][0]


def analyze_consistency(responses: list[dict]) -> pd.DataFrame:
    """Analyze consistency between sentence content and extracted attribute."""
    
    rows = []
    
    for response in responses:
        trial_id = response.get("trial_id")
        if not trial_id:
            continue
        
        # Each response has multiple replicates
        for replicate_idx, run in enumerate(response.get("responses", [])):
            steps = run.get("steps", [])
            
            if len(steps) < 3:
                continue
            
            # Extract data from steps
            sentence_step = steps[1] if len(steps) > 1 else None
            premise_step = steps[2] if len(steps) > 2 else None
            
            if not sentence_step or not premise_step:
                continue
            
            # Get sentence text
            sentence_parsed = sentence_step.get("parsed", {})
            if not sentence_parsed.get("ok", False):
                continue
            sentence_text = sentence_parsed.get("text", "")
            
            # Get premise attribute
            premise_parsed = premise_step.get("parsed", {})
            if not premise_parsed.get("ok", False):
                continue
            stated_attr = premise_parsed.get("attr")
            premise_text = premise_parsed.get("text", "")
            
            # Infer attribute from sentence
            inferred_attr_from_sentence = extract_attribute_from_text(sentence_text)
            inferred_attr_from_premise = extract_attribute_from_text(premise_text)
            
            # Check consistency
            consistent = (inferred_attr_from_sentence == stated_attr)
            
            rows.append({
                "trial_id": trial_id,
                "replicate_idx": replicate_idx,
                "sentence_text": sentence_text,
                "premise_text": premise_text,
                "stated_attr": stated_attr,
                "inferred_from_sentence": inferred_attr_from_sentence,
                "inferred_from_premise": inferred_attr_from_premise,
                "consistent": consistent,
                "sentence_mentions_stated": stated_attr in sentence_text.upper() if stated_attr else False
            })
    
    return pd.DataFrame(rows)


def print_summary(consistency_df: pd.DataFrame):
    """Print summary statistics about attribute consistency."""
    
    print("\n" + "="*70)
    print("SENTENCE-ATTRIBUTE CONSISTENCY ANALYSIS")
    print("="*70)
    
    total_samples = len(consistency_df)
    consistent_samples = consistency_df["consistent"].sum()
    inconsistent_samples = total_samples - consistent_samples
    
    print(f"\nTotal samples analyzed: {total_samples}")
    print(f"Consistent (sentence matches stated attribute): {consistent_samples} ({consistent_samples/total_samples*100:.1f}%)")
    print(f"Inconsistent: {inconsistent_samples} ({inconsistent_samples/total_samples*100:.1f}%)")
    
    # Check how often the sentence mentions the stated attribute
    if "sentence_mentions_stated" in consistency_df.columns:
        mentions_stated = consistency_df["sentence_mentions_stated"].sum()
        print(f"\nSentences explicitly mentioning the stated attribute: {mentions_stated} ({mentions_stated/total_samples*100:.1f}%)")
    
    # Attribute distribution
    print("\nStated Attribute Distribution:")
    attr_counts = consistency_df["stated_attr"].value_counts()
    for attr, count in attr_counts.items():
        print(f"  {attr}: {count} ({count/total_samples*100:.1f}%)")
    
    # Consistency by attribute
    print("\nConsistency Rate by Attribute:")
    unique_attrs = [a for a in consistency_df["stated_attr"].unique() if a is not None]
    for attr in sorted(unique_attrs):
        attr_df = consistency_df[consistency_df["stated_attr"] == attr]
        attr_consistent = attr_df["consistent"].sum()
        attr_total = len(attr_df)
        print(f"  {attr}: {attr_consistent}/{attr_total} ({attr_consistent/attr_total*100:.1f}%)")


def print_inconsistencies(consistency_df: pd.DataFrame, n_examples: int = 20):
    """Print examples of inconsistent cases."""
    
    inconsistent = consistency_df[~consistency_df["consistent"]]
    
    if len(inconsistent) == 0:
        print("\n✅ No inconsistencies found! All sentences match their stated attributes.")
        return
    
    print("\n" + "="*70)
    print("EXAMPLES OF INCONSISTENT CASES")
    print("="*70)
    
    for idx, (_, row) in enumerate(inconsistent.head(n_examples).iterrows()):
        print(f"\n[{idx+1}] Trial {row['trial_id']}, Replicate {row['replicate_idx']}:")
        print(f"  Sentence: \"{row['sentence_text']}\"")
        print(f"  Stated attribute: {row['stated_attr']}")
        print(f"  Inferred from sentence: {row['inferred_from_sentence']}")
        print(f"  Premise: \"{row['premise_text']}\"")
        print(f"  Inferred from premise: {row['inferred_from_premise']}")


def print_examples(consistency_df: pd.DataFrame, n_examples: int = 10):
    """Print some example cases showing the pattern."""
    
    print("\n" + "="*70)
    print("EXAMPLE CONSISTENT CASES (showing the pattern)")
    print("="*70)
    
    consistent = consistency_df[consistency_df["consistent"]]
    
    # Sample across different attributes
    unique_attrs = [a for a in consistency_df["stated_attr"].unique() if a is not None]
    for attr in sorted(unique_attrs):
        attr_samples = consistent[consistent["stated_attr"] == attr].head(2)
        
        if len(attr_samples) > 0:
            print(f"\n--- Attribute {attr} ---")
            for _, row in attr_samples.iterrows():
                print(f"  Trial {row['trial_id']}:")
                print(f"    Sentence: \"{row['sentence_text']}\"")
                print(f"    Premise:  \"{row['premise_text']}\"")
                print(f"    Stated: {row['stated_attr']}, Inferred: {row['inferred_from_sentence']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate consistency between reasoning sentences and extracted attributes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze responses from a run
  python scripts/validate_sentence_attribute_consistency.py \\
    --responses results/quick_ecrb_test/responses.jsonl
  
  # Save detailed results to CSV
  python scripts/validate_sentence_attribute_consistency.py \\
    --responses data/runs/v1_short/responses.jsonl \\
    --output sentence_attr_consistency.csv
        """
    )
    parser.add_argument("--responses", required=True, help="Path to responses.jsonl file")
    parser.add_argument("--examples", type=int, default=20, help="Number of inconsistency examples to show")
    parser.add_argument("--output", default=None, help="Path to save detailed CSV (optional)")
    
    args = parser.parse_args()
    
    responses_path = Path(args.responses)
    if not responses_path.exists():
        print(f"Error: File not found: {responses_path}")
        return
    
    print(f"Loading responses from: {responses_path}")
    responses = load_responses(responses_path)
    print(f"Loaded {len(responses)} trial responses")
    
    print("\nAnalyzing sentence-attribute consistency...")
    consistency_df = analyze_consistency(responses)
    
    if len(consistency_df) == 0:
        print("Error: No valid samples found in responses")
        return
    
    # Print summary statistics
    print_summary(consistency_df)
    
    # Print inconsistencies
    print_inconsistencies(consistency_df, n_examples=args.examples)
    
    # Print some example consistent cases
    print_examples(consistency_df, n_examples=10)
    
    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        consistency_df.to_csv(output_path, index=False)
        print(f"\n✅ Detailed results saved to: {output_path}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

