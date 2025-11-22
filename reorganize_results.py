#!/usr/bin/env python3
"""Reorganize reasoning effort comparison results into folders."""

from pathlib import Path
import shutil
import re

def main():
    base_dir = Path("results/reasoning_effort_comparison")
    
    # Create folder structure: model/effort/split
    model = "gpt-5-mini"
    efforts = ["minimal", "low", "medium", "high"]
    splits = ["train", "test"]
    
    print("Creating folder structure...")
    for effort in efforts:
        for split in splits:
            folder = base_dir / model / effort / split
            folder.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created: {model}/[effort]/[train|test]/\n")
    
    # Move files
    files = list(base_dir.glob("*"))
    moved_count = 0
    
    print("Moving files...")
    for file in files:
        if not file.is_file():
            continue
            
        name = file.name
        
        # Skip comparison summary - keep at top level
        if name == "comparison_summary.json":
            continue
        
        # Parse filename: responses_EFFORT_SPLIT.jsonl or results_EFFORT_SPLIT_XXX.json
        match = re.match(r'(responses|results)_(minimal|low|medium|high)_(train|test)', name)
        
        if match:
            file_type, effort, split = match.groups()
            dest_dir = base_dir / model / effort / split
            dest_path = dest_dir / name
            
            shutil.move(str(file), str(dest_path))
            print(f"  ✓ {name} → {model}/{effort}/{split}/")
            moved_count += 1
            
        # Handle iCloud placeholder files
        elif name.startswith('.responses_') and name.endswith('.icloud'):
            match = re.match(r'\.responses_(minimal|low|medium|high)_(train|test)\.jsonl\.icloud', name)
            if match:
                effort, split = match.groups()
                dest_dir = base_dir / model / effort / split
                dest_path = dest_dir / name
                
                shutil.move(str(file), str(dest_path))
                print(f"  ✓ {name} → {model}/{effort}/{split}/")
                moved_count += 1
    
    print(f"\n✅ Reorganized {moved_count} files")
    print(f"📋 Kept comparison_summary.json at top level")

if __name__ == "__main__":
    main()

