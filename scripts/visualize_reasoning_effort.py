#!/usr/bin/env python3
"""
Visualize the differences between minimal and high reasoning effort.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Load Stage A data and summaries for both conditions."""
    # Load design data
    minimal_data = pd.read_parquet(ROOT / 'results/stage_A_openai_gpt5mini/stageA_design.parquet')
    high_data = pd.read_parquet(ROOT / 'results/stage_A_openai_gpt5mini_high/stageA_design.parquet')
    
    # Load contributions
    min_contrib = pd.read_parquet(ROOT / 'results/stage_A_openai_gpt5mini/stageA_contributions.parquet')
    high_contrib = pd.read_parquet(ROOT / 'results/stage_A_openai_gpt5mini_high/stageA_contributions.parquet')
    
    # Load summaries
    with open(ROOT / 'results/stage_A_openai_gpt5mini/stageA_summary.json') as f:
        min_summary = json.load(f)
    
    with open(ROOT / 'results/stage_A_openai_gpt5mini_high/stageA_summary.json') as f:
        high_summary = json.load(f)
    
    return {
        'minimal': {'data': minimal_data, 'contrib': min_contrib, 'summary': min_summary},
        'high': {'data': high_data, 'contrib': high_contrib, 'summary': high_summary},
    }


def plot_weight_comparison(results, save_path=None):
    """Compare attribute weights between conditions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    attributes = ['E', 'A', 'S', 'D']
    attr_labels = ['Efficacy', 'Adherence', 'Safety', 'Durability']
    
    min_weights = [results['minimal']['summary']['weights'][a] for a in attributes]
    high_weights = [results['high']['summary']['weights'][a] for a in attributes]
    
    x = np.arange(len(attributes))
    width = 0.35
    
    # Bar plot
    ax1.bar(x - width/2, min_weights, width, label='Minimal Effort', alpha=0.8)
    ax1.bar(x + width/2, high_weights, width, label='High Effort', alpha=0.8)
    ax1.set_xlabel('Attribute')
    ax1.set_ylabel('Weight')
    ax1.set_title('Attribute Weights Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(attr_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Difference plot
    differences = [h - m for h, m in zip(high_weights, min_weights)]
    colors = ['red' if d < 0 else 'green' for d in differences]
    ax2.bar(x, differences, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Attribute')
    ax2.set_ylabel('Weight Change (High - Minimal)')
    ax2.set_title('Weight Changes with High Reasoning Effort')
    ax2.set_xticks(x)
    ax2.set_xticklabels(attr_labels, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved weights comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ambiguity_distribution(results, save_path=None):
    """Plot distribution of decision ambiguity."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, (condition, data) in enumerate(results.items()):
        # Get contributions
        contrib = data['contrib']
        contrib_cols = ['C_E', 'C_A', 'C_S', 'C_D']
        contribs = contrib[contrib_cols].values
        
        # Calculate ambiguity metrics
        sorted_contribs = np.sort(np.abs(contribs), axis=1)
        max_contrib = sorted_contribs[:, -1]
        second_contrib = sorted_contribs[:, -2]
        ambiguity = second_contrib / (max_contrib + 1e-9)
        
        # Plot max contribution histogram
        ax = axes[0, idx]
        ax.hist(max_contrib, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(max_contrib.mean(), color='red', linestyle='--', 
                   label=f'Mean: {max_contrib.mean():.2f}')
        ax.set_xlabel('Maximum Contribution')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{condition.capitalize()} - Decisiveness')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot ambiguity ratio histogram
        ax = axes[1, idx]
        ax.hist(ambiguity, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(ambiguity.mean(), color='red', linestyle='--',
                   label=f'Mean: {ambiguity.mean():.2f}')
        ax.set_xlabel('Second-best / Best Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{condition.capitalize()} - Ambiguity')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ambiguity distribution to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_driver_distribution(results, save_path=None):
    """Compare driver attribute distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    attributes = ['E', 'A', 'S', 'D']
    attr_labels = ['Efficacy', 'Adherence', 'Safety', 'Durability']
    
    # Get driver counts
    min_drivers = results['minimal']['contrib']['driver'].value_counts()
    high_drivers = results['high']['contrib']['driver'].value_counts()
    
    min_counts = [min_drivers.get(a, 0) for a in attributes]
    high_counts = [high_drivers.get(a, 0) for a in attributes]
    
    x = np.arange(len(attributes))
    width = 0.35
    
    # Bar plot
    ax1.bar(x - width/2, min_counts, width, label='Minimal Effort', alpha=0.8)
    ax1.bar(x + width/2, high_counts, width, label='High Effort', alpha=0.8)
    ax1.set_xlabel('Attribute')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Driver Attribute Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(attr_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Percentage plot
    min_pct = [c / sum(min_counts) * 100 for c in min_counts]
    high_pct = [c / sum(high_counts) * 100 for c in high_counts]
    
    ax2.bar(x - width/2, min_pct, width, label='Minimal Effort', alpha=0.8)
    ax2.bar(x + width/2, high_pct, width, label='High Effort', alpha=0.8)
    ax2.set_xlabel('Attribute')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Driver Attribute Distribution (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(attr_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=25, color='gray', linestyle=':', linewidth=1, label='Uniform (25%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved driver distribution to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary_stats(results):
    """Print summary statistics."""
    print("=" * 80)
    print("REASONING EFFORT COMPARISON SUMMARY")
    print("=" * 80)
    
    for condition, data in results.items():
        print(f"\n{condition.upper()} EFFORT:")
        print("-" * 40)
        
        summary = data['summary']
        contrib = data['contrib']
        
        # Weights
        print("\nWeights:")
        for attr in ['E', 'A', 'S', 'D']:
            print(f"  {attr}: {summary['weights'][attr]:.3f}")
        
        # CV
        weights = np.array([summary['weights'][a] for a in ['E', 'A', 'S', 'D']])
        cv = np.std(weights) / np.mean(weights)
        print(f"  CV: {cv:.3f}")
        
        # Performance
        print("\nPerformance:")
        print(f"  Log Loss: {summary['cv']['log_loss']:.4f}")
        print(f"  Accuracy: {summary['cv']['accuracy']:.4f}")
        
        # Contributions
        contrib_cols = ['C_E', 'C_A', 'C_S', 'C_D']
        contribs = contrib[contrib_cols].values
        max_contrib = np.max(np.abs(contribs), axis=1).mean()
        print(f"  Avg Max Contribution: {max_contrib:.3f}")
        
        # Ambiguity
        sorted_contribs = np.sort(np.abs(contribs), axis=1)
        ambiguity = (sorted_contribs[:, -2] / (sorted_contribs[:, -1] + 1e-9)).mean()
        print(f"  Avg Ambiguity Ratio: {ambiguity:.3f}")


def main():
    """Generate all visualizations."""
    print("Loading data...")
    results = load_data()
    
    print("\nGenerating visualizations...")
    
    output_dir = ROOT / 'results' / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    plot_weight_comparison(results, output_dir / 'reasoning_effort_weights.png')
    plot_ambiguity_distribution(results, output_dir / 'reasoning_effort_ambiguity.png')
    plot_driver_distribution(results, output_dir / 'reasoning_effort_drivers.png')
    
    # Print summary
    print_summary_stats(results)
    
    print(f"\n✅ All visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()

