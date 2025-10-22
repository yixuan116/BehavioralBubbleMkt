#!/usr/bin/env python3
"""
Last Price Comparison Analysis - Step 1
Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics

This script compares LastPrice between Choice A and Choice B
before analyzing bubbles.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style
sns.set_style('whitegrid')
sns.set_palette('husl')

def load_data():
    """Load and prepare data"""
    print("Loading data for Last Price Comparison...")
    
    df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    
    return df_a, df_b

def analyze_last_prices():
    """Analyze LastPrice comparison between Choice A and B"""
    print("\n" + "="*80)
    print("STEP 1: LAST PRICE COMPARISON - Choice A vs Choice B")
    print("="*80)
    
    df_a, df_b = load_data()
    
    # Clean data
    price_a = df_a['LastPrice'].dropna()
    price_b = df_b['LastPrice'].dropna()
    
    print(f"Choice A - Last Price Statistics:")
    print(f"  Sample size: {len(price_a):,}")
    print(f"  Mean price: {price_a.mean():.2f}")
    print(f"  Median price: {price_a.median():.2f}")
    print(f"  Std deviation: {price_a.std():.2f}")
    print(f"  Min: {price_a.min():.2f}, Max: {price_a.max():.2f}")
    
    print(f"\nChoice B - Last Price Statistics:")
    print(f"  Sample size: {len(price_b):,}")
    print(f"  Mean price: {price_b.mean():.2f}")
    print(f"  Median price: {price_b.median():.2f}")
    print(f"  Std deviation: {price_b.std():.2f}")
    print(f"  Min: {price_b.min():.2f}, Max: {price_b.max():.2f}")
    
    # Statistical tests
    print(f"\nStatistical Tests:")
    
    # Two-sample t-test
    t_stat, p_val = stats.ttest_ind(price_a, price_b)
    print(f"\nTwo-sample t-test:")
    print(f"  t = {t_stat:.3f}, p = {p_val:.6f}")
    
    # Mann-Whitney U test
    u_stat, p_val_u = stats.mannwhitneyu(price_a, price_b, alternative='two-sided')
    print(f"\nMann-Whitney U test:")
    print(f"  U = {u_stat:.3f}, p = {p_val_u:.6f}")
    
    # Effect size
    pooled_std = np.sqrt(((len(price_a)-1)*price_a.var() + (len(price_b)-1)*price_b.var()) / 
                        (len(price_a) + len(price_b) - 2))
    cohens_d = (price_a.mean() - price_b.mean()) / pooled_std
    print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if p_val < 0.05:
        print(f"  ✓ Significant difference in prices between Choice A and B (p < 0.05)")
        if price_a.mean() > price_b.mean():
            print(f"  → Choice A has higher average prices")
        else:
            print(f"  → Choice B has higher average prices")
    else:
        print(f"  ✗ No significant difference in prices between Choice A and B (p ≥ 0.05)")
        print(f"  → Both choices have similar price levels")
    
    return price_a, price_b

def create_lastprice_visualizations(price_a, price_b):
    """Create visualizations for LastPrice comparison"""
    print("\nCreating LastPrice comparison visualizations...")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Step 1: Last Price Comparison - Choice A vs Choice B', 
                 fontsize=16, fontweight='bold')
    
    # 1. Histogram comparison
    axes[0,0].hist(price_a, bins=60, alpha=0.7, color='blue', 
                   label=f'Choice A (n={len(price_a):,})', density=True, edgecolor='black')
    axes[0,0].hist(price_b, bins=60, alpha=0.7, color='red', 
                   label=f'Choice B (n={len(price_b):,})', density=True, edgecolor='black')
    axes[0,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[0,0].set_ylabel('Density', fontsize=12)
    axes[0,0].set_title('Last Price Distribution Comparison', fontsize=14, fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    data_to_plot = [price_a, price_b]
    box_plot = axes[0,1].boxplot(data_to_plot, tick_labels=['Choice A', 'Choice B'], 
                                patch_artist=True, showfliers=True)
    box_plot['boxes'][0].set_facecolor('blue')
    box_plot['boxes'][1].set_facecolor('red')
    axes[0,1].set_ylabel('Last Price (Francs)', fontsize=12)
    axes[0,1].set_title('Last Price Box Plot Comparison', fontsize=14, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Cumulative distribution functions
    sorted_a = np.sort(price_a)
    sorted_b = np.sort(price_b)
    y_a = np.arange(1, len(sorted_a) + 1) / len(sorted_a)
    y_b = np.arange(1, len(sorted_b) + 1) / len(sorted_b)
    
    axes[1,0].plot(sorted_a, y_a, color='blue', linewidth=2, label='Choice A')
    axes[1,0].plot(sorted_b, y_b, color='red', linewidth=2, label='Choice B')
    axes[1,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[1,0].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1,0].set_title('Cumulative Distribution Functions', fontsize=14, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Statistical summary
    axes[1,1].axis('off')
    
    # Calculate statistics
    t_stat, p_val = stats.ttest_ind(price_a, price_b)
    u_stat, p_val_u = stats.mannwhitneyu(price_a, price_b, alternative='two-sided')
    pooled_std = np.sqrt(((len(price_a)-1)*price_a.var() + (len(price_b)-1)*price_b.var()) / 
                        (len(price_a) + len(price_b) - 2))
    cohens_d = (price_a.mean() - price_b.mean()) / pooled_std
    
    summary_text = f"""
LAST PRICE COMPARISON SUMMARY

Choice A:
• Sample size: {len(price_a):,}
• Mean price: {price_a.mean():.2f}
• Median price: {price_a.median():.2f}
• Std deviation: {price_a.std():.2f}
• Range: {price_a.min():.2f} - {price_a.max():.2f}

Choice B:
• Sample size: {len(price_b):,}
• Mean price: {price_b.mean():.2f}
• Median price: {price_b.median():.2f}
• Std deviation: {price_b.std():.2f}
• Range: {price_b.min():.2f} - {price_b.max():.2f}

Statistical Tests:
• t-test: t = {t_stat:.3f}, p = {p_val:.6f}
• Mann-Whitney U: U = {u_stat:.3f}, p = {p_val_u:.6f}
• Effect size: d = {cohens_d:.3f}

CONCLUSION:
{'Significant difference' if p_val < 0.05 else 'No significant difference'} 
in prices between Choice A and B
"""
    
    axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('lastprice_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("LastPrice comparison saved as 'lastprice_comparison.png'")

def main():
    """Main function"""
    print("="*80)
    print("LAST PRICE COMPARISON ANALYSIS")
    print("="*80)
    
    # Analyze LastPrice comparison
    price_a, price_b = analyze_last_prices()
    
    # Create visualizations
    create_lastprice_visualizations(price_a, price_b)
    
    print("\n" + "="*80)
    print("LAST PRICE COMPARISON ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
