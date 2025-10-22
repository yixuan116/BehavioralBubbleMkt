#!/usr/bin/env python3
"""
Information Structure Analysis (Analysis Point #2)
Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics

H0: μ_A = μ_B vs H1: μ_A ≠ μ_B
Test: Two-sample t-test / Mann-Whitney U
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
    print("Loading data for Information Structure Analysis...")
    
    df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    
    # Calculate bubble metrics using standard formula: Bubble_mt = (P_mt - F_t) / F_t
    df_a['Bubble'] = (df_a['LastPrice'] - df_a['Fundamental']) / df_a['Fundamental']
    df_b['Bubble'] = (df_b['LastPrice'] - df_b['Fundamental']) / df_b['Fundamental']
    
    return df_a, df_b

def test_information_structure():
    """Test information structure differences between experiments"""
    print("\n" + "="*80)
    print("ANALYSIS POINT #2: INFORMATION STRUCTURE (A vs B)")
    print("="*80)
    print("H0: μ_A = μ_B vs H1: μ_A ≠ μ_B")
    print("Test: Two-sample t-test / Mann-Whitney U")
    
    df_a, df_b = load_data()
    
    # Get last bubble from each market/session
    # Choice A: Last bubble from each market
    last_bubble_a = df_a.groupby(['Session', 'Market'])['Bubble'].last().dropna()
    # Choice B: Last bubble from each session
    last_bubble_b = df_b.groupby('Session')['Bubble'].last().dropna()
    
    print(f"\nLast Bubble Statistics:")
    print(f"  Choice A (n={len(last_bubble_a)}): mean={last_bubble_a.mean():.2f}, std={last_bubble_a.std():.2f}")
    print(f"  Choice B (n={len(last_bubble_b)}): mean={last_bubble_b.mean():.2f}, std={last_bubble_b.std():.2f}")
    
    # Statistical tests
    print(f"\nStatistical Tests:")
    
    # Two-sample t-test
    t_stat, p_val_t = stats.ttest_ind(last_bubble_a, last_bubble_b)
    print(f"\nTwo-sample t-test:")
    print(f"  t = {t_stat:.3f}, p = {p_val_t:.6f}")
    
    # Mann-Whitney U test
    u_stat, p_val_u = stats.mannwhitneyu(last_bubble_a, last_bubble_b, alternative='two-sided')
    print(f"\nMann-Whitney U test:")
    print(f"  U = {u_stat:.3f}, p = {p_val_u:.6f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(last_bubble_a)-1)*last_bubble_a.var() + (len(last_bubble_b)-1)*last_bubble_b.var()) / 
                        (len(last_bubble_a) + len(last_bubble_b) - 2))
    cohens_d = (last_bubble_a.mean() - last_bubble_b.mean()) / pooled_std
    print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if p_val_t < 0.05:
        print(f"  ✓ Significant difference between experiments (p < 0.05)")
        if last_bubble_a.mean() > last_bubble_b.mean():
            print(f"  → Choice A has larger final bubbles")
            print(f"  → More information (showing fundamentals) leads to LARGER bubbles")
            print(f"  → This contradicts the hypothesis that information reduces bubbles")
        else:
            print(f"  → Choice B has larger final bubbles")
            print(f"  → Less information leads to LARGER bubbles")
            print(f"  → This supports the hypothesis that information reduces bubbles")
    else:
        print(f"  ✗ No significant difference between experiments (p ≥ 0.05)")
        print(f"  → Information structure does not affect bubble size")
    
    return last_bubble_a, last_bubble_b, df_a, df_b

def create_information_visualizations(last_bubble_a, last_bubble_b, df_a, df_b):
    """Create visualizations for information structure analysis"""
    print("\nCreating information structure visualizations...")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Information Structure Analysis: Choice A vs B', 
                 fontsize=16, fontweight='bold')
    
    # 1. Last bubble comparison
    axes[0,0].hist(last_bubble_a, bins=20, alpha=0.7, color='blue', 
                   label=f'Choice A (n={len(last_bubble_a)})', density=True, edgecolor='black')
    axes[0,0].hist(last_bubble_b, bins=20, alpha=0.7, color='red', 
                   label=f'Choice B (n={len(last_bubble_b)})', density=True, edgecolor='black')
    axes[0,0].axvline(x=0, color='black', linestyle='--', linewidth=3, 
                     label='Efficient Market (0)', alpha=0.8)
    axes[0,0].set_xlabel('Last Bubble Size', fontsize=12)
    axes[0,0].set_ylabel('Density', fontsize=12)
    axes[0,0].set_title('Last Bubble Distribution by Experiment', fontsize=14, fontweight='bold')
    axes[0,0].legend(fontsize=10)
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    data_to_plot = [last_bubble_a, last_bubble_b]
    box_plot = axes[0,1].boxplot(data_to_plot, labels=['Choice A', 'Choice B'], 
                                patch_artist=True, showfliers=True)
    box_plot['boxes'][0].set_facecolor('blue')
    box_plot['boxes'][1].set_facecolor('red')
    axes[0,1].axhline(y=0, color='black', linestyle='--', linewidth=3, 
                     label='Efficient Market (0)', alpha=0.8)
    axes[0,1].set_ylabel('Last Bubble Size', fontsize=12)
    axes[0,1].set_title('Last Bubble Box Plots', fontsize=14, fontweight='bold')
    axes[0,1].legend(fontsize=10)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Information treatment effect (Choice A only)
    fund_shown = df_a[df_a['FundShown'] == 1]['Bubble'].dropna()
    fund_hidden = df_a[df_a['FundShown'] == 0]['Bubble'].dropna()
    
    axes[1,0].hist(fund_shown, bins=40, alpha=0.7, color='green', 
                   label=f'Fundamental Shown (n={len(fund_shown):,})', density=True, edgecolor='black')
    axes[1,0].hist(fund_hidden, bins=40, alpha=0.7, color='orange', 
                   label=f'Fundamental Hidden (n={len(fund_hidden):,})', density=True, edgecolor='black')
    axes[1,0].axvline(x=0, color='black', linestyle='--', linewidth=3, 
                     label='Efficient Market (0)', alpha=0.8)
    axes[1,0].set_xlabel('Bubble Size', fontsize=12)
    axes[1,0].set_ylabel('Density', fontsize=12)
    axes[1,0].set_title('Information Treatment Effect (Choice A)', fontsize=14, fontweight='bold')
    axes[1,0].legend(fontsize=10)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Statistical summary
    axes[1,1].axis('off')
    
    # Calculate additional statistics
    t_stat, p_val_t = stats.ttest_ind(last_bubble_a, last_bubble_b)
    u_stat, p_val_u = stats.mannwhitneyu(last_bubble_a, last_bubble_b, alternative='two-sided')
    
    # Information treatment test
    t_info, p_info = stats.ttest_ind(fund_shown, fund_hidden)
    
    summary_text = f"""
    STATISTICAL SUMMARY
    
    Experiment Comparison:
    • Choice A Mean: {last_bubble_a.mean():.2f}
    • Choice B Mean: {last_bubble_b.mean():.2f}
    • Difference: {last_bubble_a.mean() - last_bubble_b.mean():.2f}
    • t-test p-value: {p_val_t:.6f}
    • Mann-Whitney p-value: {p_val_u:.6f}
    
    Information Treatment (A only):
    • Fundamental Shown Mean: {fund_shown.mean():.2f}
    • Fundamental Hidden Mean: {fund_hidden.mean():.2f}
    • t-test p-value: {p_info:.6f}
    
    CONCLUSION:
    {'Significant difference between experiments' if p_val_t < 0.05 else 'No significant difference between experiments'}
    """
    
    axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('information_structure_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Information structure analysis saved as 'information_structure_analysis.png'")

def main():
    """Main function"""
    print("="*80)
    print("INFORMATION STRUCTURE ANALYSIS")
    print("="*80)
    
    # Test information structure
    last_bubble_a, last_bubble_b, df_a, df_b = test_information_structure()
    
    # Create visualizations
    create_information_visualizations(last_bubble_a, last_bubble_b, df_a, df_b)
    
    print("\n" + "="*80)
    print("INFORMATION STRUCTURE ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
