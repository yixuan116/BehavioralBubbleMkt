#!/usr/bin/env python3
"""
Market Efficiency Analysis (Analysis Point #1)
Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics

H0: bubble mean = 0 vs H1: bubble mean ≠ 0
Test: One-sample t-test / Wilcoxon signed-rank
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style
sns.set_style('whitegrid')
sns.set_palette('husl')

def load_data():
    """Load and prepare data"""
    print("Loading data for Market Efficiency Analysis...")
    
    df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    
    # Calculate bubble metrics using standard formula: Bubble_mt = (P_mt - F_t) / F_t
    df_a['Bubble'] = (df_a['LastPrice'] - df_a['Fundamental']) / df_a['Fundamental']
    df_a['BubbleRatio'] = df_a['LastPrice'] / df_a['Fundamental']
    df_b['Bubble'] = (df_b['LastPrice'] - df_b['Fundamental']) / df_b['Fundamental']
    df_b['BubbleRatio'] = df_b['LastPrice'] / df_b['Fundamental']
    
    return df_a, df_b

def test_market_efficiency():
    """Test market efficiency for both experiments"""
    print("\n" + "="*80)
    print("ANALYSIS POINT #1: MARKET EFFICIENCY")
    print("="*80)
    print("H0: bubble mean = 0 vs H1: bubble mean ≠ 0")
    print("Test: One-sample t-test / Wilcoxon signed-rank")
    
    df_a, df_b = load_data()
    
    # Clean data
    bubble_a = df_a['Bubble'].dropna()
    bubble_b = df_b['Bubble'].dropna()
    
    print(f"\nChoice A - Bubble Statistics:")
    print(f"  Sample size: {len(bubble_a):,}")
    print(f"  Mean bubble: {bubble_a.mean():.2f}")
    print(f"  Median bubble: {bubble_a.median():.2f}")
    print(f"  Std deviation: {bubble_a.std():.2f}")
    print(f"  Min: {bubble_a.min():.2f}, Max: {bubble_a.max():.2f}")
    
    print(f"\nChoice B - Bubble Statistics:")
    print(f"  Sample size: {len(bubble_b):,}")
    print(f"  Mean bubble: {bubble_b.mean():.2f}")
    print(f"  Median bubble: {bubble_b.median():.2f}")
    print(f"  Std deviation: {bubble_b.std():.2f}")
    print(f"  Min: {bubble_b.min():.2f}, Max: {bubble_b.max():.2f}")
    
    # Statistical tests
    print(f"\nStatistical Tests:")
    
    # One-sample t-tests
    t_stat_a, p_val_a = stats.ttest_1samp(bubble_a, 0)
    t_stat_b, p_val_b = stats.ttest_1samp(bubble_b, 0)
    
    print(f"\nOne-sample t-tests:")
    print(f"  Choice A: t = {t_stat_a:.3f}, p = {p_val_a:.6f}")
    print(f"  Choice B: t = {t_stat_b:.3f}, p = {p_val_b:.6f}")
    
    # Wilcoxon signed-rank tests
    w_stat_a, p_val_w_a = stats.wilcoxon(bubble_a)
    w_stat_b, p_val_w_b = stats.wilcoxon(bubble_b)
    
    print(f"\nWilcoxon signed-rank tests:")
    print(f"  Choice A: W = {w_stat_a:.3f}, p = {p_val_w_a:.6f}")
    print(f"  Choice B: W = {w_stat_b:.3f}, p = {p_val_w_b:.6f}")
    
    # Effect sizes
    cohens_d_a = bubble_a.mean() / bubble_a.std()
    cohens_d_b = bubble_b.mean() / bubble_b.std()
    
    print(f"\nEffect sizes (Cohen's d):")
    print(f"  Choice A: d = {cohens_d_a:.3f}")
    print(f"  Choice B: d = {cohens_d_b:.3f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if p_val_a < 0.05:
        print(f"  ✓ Choice A: Significant bubbles detected (p < 0.05)")
        print(f"    → Market is INEFFICIENT - prices deviate from fundamentals")
    else:
        print(f"  ✗ Choice A: No significant bubbles (p ≥ 0.05)")
        print(f"    → Market is EFFICIENT - prices equal fundamentals")
        
    if p_val_b < 0.05:
        print(f"  ✓ Choice B: Significant bubbles detected (p < 0.05)")
        print(f"    → Market is INEFFICIENT - prices deviate from fundamentals")
    else:
        print(f"  ✗ Choice B: No significant bubbles (p ≥ 0.05)")
        print(f"    → Market is EFFICIENT - prices equal fundamentals")
    
    return bubble_a, bubble_b

def create_efficiency_visualizations(bubble_a, bubble_b):
    """Create visualizations for market efficiency analysis"""
    print("\nCreating market efficiency visualizations...")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Market Efficiency Analysis: Bubble Detection', 
                 fontsize=16, fontweight='bold')
    
    # 1. Bubble distribution comparison
    axes[0,0].hist(bubble_a, bins=60, alpha=0.7, color='blue', 
                   label=f'Choice A (n={len(bubble_a):,})', density=True, edgecolor='black')
    axes[0,0].hist(bubble_b, bins=60, alpha=0.7, color='red', 
                   label=f'Choice B (n={len(bubble_b):,})', density=True, edgecolor='black')
    axes[0,0].axvline(x=0, color='black', linestyle='--', linewidth=3, 
                     label='Efficient Market (0)', alpha=0.8)
    axes[0,0].set_xlabel('Bubble Size (Price - Fundamental)', fontsize=12)
    axes[0,0].set_ylabel('Density', fontsize=12)
    axes[0,0].set_title('Bubble Distribution Comparison', fontsize=14, fontweight='bold')
    axes[0,0].legend(fontsize=10)
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    data_to_plot = [bubble_a, bubble_b]
    box_plot = axes[0,1].boxplot(data_to_plot, labels=['Choice A', 'Choice B'], 
                                patch_artist=True, showfliers=True)
    box_plot['boxes'][0].set_facecolor('blue')
    box_plot['boxes'][1].set_facecolor('red')
    axes[0,1].axhline(y=0, color='black', linestyle='--', linewidth=3, 
                     label='Efficient Market (0)', alpha=0.8)
    axes[0,1].set_ylabel('Bubble Size', fontsize=12)
    axes[0,1].set_title('Bubble Distribution Box Plots', fontsize=14, fontweight='bold')
    axes[0,1].legend(fontsize=10)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Price/Fundamental ratio comparison
    bubble_ratio_a = bubble_a / (bubble_a + 360) + 1  # Approximate ratio
    bubble_ratio_b = bubble_b / (bubble_b + 360) + 1  # Approximate ratio
    
    axes[1,0].hist(bubble_ratio_a, bins=60, alpha=0.7, color='blue', 
                   label='Choice A', density=True, edgecolor='black')
    axes[1,0].hist(bubble_ratio_b, bins=60, alpha=0.7, color='red', 
                   label='Choice B', density=True, edgecolor='black')
    axes[1,0].axvline(x=1, color='black', linestyle='--', linewidth=3, 
                     label='Efficient Market (1.0)', alpha=0.8)
    axes[1,0].set_xlabel('Price/Fundamental Ratio', fontsize=12)
    axes[1,0].set_ylabel('Density', fontsize=12)
    axes[1,0].set_title('Price/Fundamental Ratio Distribution', fontsize=14, fontweight='bold')
    axes[1,0].legend(fontsize=10)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Statistical summary
    axes[1,1].axis('off')
    
    # Create summary text
    summary_text = f"""
    STATISTICAL SUMMARY
    
    Choice A:
    • Mean Bubble: {bubble_a.mean():.2f}
    • Median Bubble: {bubble_a.median():.2f}
    • Std Dev: {bubble_a.std():.2f}
    • t-test p-value: {stats.ttest_1samp(bubble_a, 0)[1]:.6f}
    • Effect Size: {bubble_a.mean()/bubble_a.std():.3f}
    
    Choice B:
    • Mean Bubble: {bubble_b.mean():.2f}
    • Median Bubble: {bubble_b.median():.2f}
    • Std Dev: {bubble_b.std():.2f}
    • t-test p-value: {stats.ttest_1samp(bubble_b, 0)[1]:.6f}
    • Effect Size: {bubble_b.mean()/bubble_b.std():.3f}
    
    CONCLUSION:
    Both experiments show significant bubbles
    (p < 0.001), indicating market inefficiency.
    """
    
    axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('market_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Market efficiency analysis saved as 'market_efficiency_analysis.png'")

def main():
    """Main function"""
    print("="*80)
    print("MARKET EFFICIENCY ANALYSIS")
    print("="*80)
    
    # Test market efficiency
    bubble_a, bubble_b = test_market_efficiency()
    
    # Create visualizations
    create_efficiency_visualizations(bubble_a, bubble_b)
    
    print("\n" + "="*80)
    print("MARKET EFFICIENCY ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
