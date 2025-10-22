#!/usr/bin/env python3
"""
Market Layer Analysis for Bubble Market Experiments
Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics

This script implements the Market layer analysis points:
1. Market Efficiency - Test for bubble existence
2. Information Structure (A vs B) - Compare experiments
3. Learning Across Sessions - Test bubble convergence
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

def load_and_prepare_data():
    """Load and prepare data for market layer analysis"""
    print("Loading and preparing data...")
    
    # Load data
    df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    
    # Calculate bubble metrics
    df_a['Bubble'] = df_a['LastPrice'] - df_a['Fundamental']
    df_a['BubbleRatio'] = df_a['LastPrice'] / df_a['Fundamental']
    df_b['Bubble'] = df_b['LastPrice'] - df_b['Fundamental']
    df_b['BubbleRatio'] = df_b['LastPrice'] / df_b['Fundamental']
    
    # Add experiment labels
    df_a['Experiment'] = 'A'
    df_b['Experiment'] = 'B'
    
    # Combine for comparison
    df_combined = pd.concat([df_a, df_b], ignore_index=True)
    
    print(f"Experiment A: {len(df_a):,} observations")
    print(f"Experiment B: {len(df_b):,} observations")
    print(f"Combined: {len(df_combined):,} observations")
    
    return df_a, df_b, df_combined

def test_market_efficiency(df_a, df_b):
    """
    Analysis Point #1: Market Efficiency
    H0: bubble mean = 0 vs H1: bubble mean ≠ 0
    Test: One-sample t-test / Wilcoxon signed-rank
    """
    print("\n" + "="*80)
    print("ANALYSIS POINT #1: MARKET EFFICIENCY")
    print("="*80)
    print("H0: bubble mean = 0 vs H1: bubble mean ≠ 0")
    print("Test: One-sample t-test / Wilcoxon signed-rank")
    
    # Clean data (remove NaN values)
    bubble_a = df_a['Bubble'].dropna()
    bubble_b = df_b['Bubble'].dropna()
    
    print(f"\nExperiment A - Bubble Statistics:")
    print(f"  Sample size: {len(bubble_a):,}")
    print(f"  Mean bubble: {bubble_a.mean():.2f}")
    print(f"  Median bubble: {bubble_a.median():.2f}")
    print(f"  Std deviation: {bubble_a.std():.2f}")
    print(f"  Min: {bubble_a.min():.2f}, Max: {bubble_a.max():.2f}")
    
    print(f"\nExperiment B - Bubble Statistics:")
    print(f"  Sample size: {len(bubble_b):,}")
    print(f"  Mean bubble: {bubble_b.mean():.2f}")
    print(f"  Median bubble: {bubble_b.median():.2f}")
    print(f"  Std deviation: {bubble_b.std():.2f}")
    print(f"  Min: {bubble_b.min():.2f}, Max: {bubble_b.max():.2f}")
    
    # One-sample t-tests
    print(f"\nOne-sample t-tests:")
    t_stat_a, p_val_a = stats.ttest_1samp(bubble_a, 0)
    t_stat_b, p_val_b = stats.ttest_1samp(bubble_b, 0)
    
    print(f"  Experiment A: t = {t_stat_a:.3f}, p = {p_val_a:.6f}")
    print(f"  Experiment B: t = {t_stat_b:.3f}, p = {p_val_b:.6f}")
    
    # Wilcoxon signed-rank tests
    print(f"\nWilcoxon signed-rank tests:")
    w_stat_a, p_val_w_a = stats.wilcoxon(bubble_a)
    w_stat_b, p_val_w_b = stats.wilcoxon(bubble_b)
    
    print(f"  Experiment A: W = {w_stat_a:.3f}, p = {p_val_w_a:.6f}")
    print(f"  Experiment B: W = {w_stat_b:.3f}, p = {p_val_w_b:.6f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if p_val_a < 0.05:
        print(f"  ✓ Experiment A: Significant bubbles detected (p < 0.05)")
    else:
        print(f"  ✗ Experiment A: No significant bubbles (p ≥ 0.05)")
        
    if p_val_b < 0.05:
        print(f"  ✓ Experiment B: Significant bubbles detected (p < 0.05)")
    else:
        print(f"  ✗ Experiment B: No significant bubbles (p ≥ 0.05)")
    
    return bubble_a, bubble_b

def test_information_structure(df_a, df_b):
    """
    Analysis Point #2: Information Structure (A vs B)
    H0: μ_A = μ_B vs H1: μ_A ≠ μ_B
    Test: Two-sample t-test / Mann-Whitney U
    """
    print("\n" + "="*80)
    print("ANALYSIS POINT #2: INFORMATION STRUCTURE (A vs B)")
    print("="*80)
    print("H0: μ_A = μ_B vs H1: μ_A ≠ μ_B")
    print("Test: Two-sample t-test / Mann-Whitney U")
    
    # Use LastBubble (final bubble in each market)
    # For Experiment A: Get last bubble from each market
    last_bubble_a = df_a.groupby(['Session', 'Market'])['Bubble'].last().dropna()
    # For Experiment B: Get last bubble from each session (no market structure)
    last_bubble_b = df_b.groupby('Session')['Bubble'].last().dropna()
    
    print(f"\nLast Bubble Statistics:")
    print(f"  Experiment A (n={len(last_bubble_a)}): mean={last_bubble_a.mean():.2f}, std={last_bubble_a.std():.2f}")
    print(f"  Experiment B (n={len(last_bubble_b)}): mean={last_bubble_b.mean():.2f}, std={last_bubble_b.std():.2f}")
    
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
            print(f"  → Experiment A has larger bubbles (more inefficiency)")
        else:
            print(f"  → Experiment B has larger bubbles (more inefficiency)")
    else:
        print(f"  ✗ No significant difference between experiments (p ≥ 0.05)")
    
    return last_bubble_a, last_bubble_b

def test_learning_across_sessions(df_a, df_b):
    """
    Analysis Point #3: Learning Across Sessions
    H0: bubble means equal across sessions
    Test: One-way ANOVA / Kruskal-Wallis
    """
    print("\n" + "="*80)
    print("ANALYSIS POINT #3: LEARNING ACROSS SESSIONS")
    print("="*80)
    print("H0: bubble means equal across sessions")
    print("Test: One-way ANOVA / Kruskal-Wallis")
    
    # Calculate session-level bubble means
    session_bubbles_a = df_a.groupby('Session')['Bubble'].mean()
    session_bubbles_b = df_b.groupby('Session')['Bubble'].mean()
    
    print(f"\nSession-level bubble means:")
    print("Experiment A:")
    for session, bubble in session_bubbles_a.items():
        print(f"  Session {session}: {bubble:.2f}")
    
    print("Experiment B:")
    for session, bubble in session_bubbles_b.items():
        print(f"  Session {session}: {bubble:.2f}")
    
    # One-way ANOVA for Experiment A
    session_groups_a = [df_a[df_a['Session'] == s]['Bubble'].dropna() for s in df_a['Session'].unique()]
    f_stat_a, p_val_a = stats.f_oneway(*session_groups_a)
    
    print(f"\nOne-way ANOVA - Experiment A:")
    print(f"  F = {f_stat_a:.3f}, p = {p_val_a:.6f}")
    
    # One-way ANOVA for Experiment B
    session_groups_b = [df_b[df_b['Session'] == s]['Bubble'].dropna() for s in df_b['Session'].unique()]
    f_stat_b, p_val_b = stats.f_oneway(*session_groups_b)
    
    print(f"\nOne-way ANOVA - Experiment B:")
    print(f"  F = {f_stat_b:.3f}, p = {p_val_b:.6f}")
    
    # Kruskal-Wallis tests
    h_stat_a, p_val_h_a = stats.kruskal(*session_groups_a)
    h_stat_b, p_val_h_b = stats.kruskal(*session_groups_b)
    
    print(f"\nKruskal-Wallis tests:")
    print(f"  Experiment A: H = {h_stat_a:.3f}, p = {p_val_h_a:.6f}")
    print(f"  Experiment B: H = {h_stat_b:.3f}, p = {p_val_h_b:.6f}")
    
    # Trend analysis
    print(f"\nTrend Analysis:")
    sessions_a = np.array(session_bubbles_a.index)
    bubbles_a = np.array(session_bubbles_a.values)
    slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(sessions_a, bubbles_a)
    
    sessions_b = np.array(session_bubbles_b.index)
    bubbles_b = np.array(session_bubbles_b.values)
    slope_b, intercept_b, r_value_b, p_value_b, std_err_b = stats.linregress(sessions_b, bubbles_b)
    
    print(f"  Experiment A: slope = {slope_a:.3f}, R² = {r_value_a**2:.3f}, p = {p_value_a:.6f}")
    print(f"  Experiment B: slope = {slope_b:.3f}, R² = {r_value_b**2:.3f}, p = {p_value_b:.6f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if p_val_a < 0.05:
        print(f"  ✓ Experiment A: Significant differences across sessions (p < 0.05)")
    else:
        print(f"  ✗ Experiment A: No significant differences across sessions (p ≥ 0.05)")
        
    if p_val_b < 0.05:
        print(f"  ✓ Experiment B: Significant differences across sessions (p < 0.05)")
    else:
        print(f"  ✗ Experiment B: No significant differences across sessions (p ≥ 0.05)")
    
    if slope_a < 0 and p_value_a < 0.05:
        print(f"  → Experiment A shows declining bubbles (learning effect)")
    if slope_b < 0 and p_value_b < 0.05:
        print(f"  → Experiment B shows declining bubbles (learning effect)")
    
    return session_bubbles_a, session_bubbles_b

def create_market_visualizations(df_a, df_b, bubble_a, bubble_b, last_bubble_a, last_bubble_b, session_bubbles_a, session_bubbles_b):
    """Create visualizations for market layer analysis"""
    print("\nCreating market layer visualizations...")
    
    # Create comprehensive market analysis figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Market Layer Analysis: Efficiency, Information Structure, and Learning', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Market Efficiency
    # 1.1 Bubble distribution comparison
    axes[0,0].hist(bubble_a, bins=50, alpha=0.7, color='blue', label='Experiment A', density=True)
    axes[0,0].hist(bubble_b, bins=50, alpha=0.7, color='red', label='Experiment B', density=True)
    axes[0,0].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Efficient Market (0)')
    axes[0,0].set_xlabel('Bubble Size (Price - Fundamental)')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Bubble Distribution Comparison')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 1.2 Box plot comparison
    data_to_plot = [bubble_a, bubble_b]
    box_plot = axes[0,1].boxplot(data_to_plot, labels=['Experiment A', 'Experiment B'], 
                                 patch_artist=True)
    box_plot['boxes'][0].set_facecolor('blue')
    box_plot['boxes'][1].set_facecolor('red')
    axes[0,1].axhline(y=0, color='black', linestyle='--', linewidth=2)
    axes[0,1].set_ylabel('Bubble Size')
    axes[0,1].set_title('Bubble Distribution Box Plots')
    axes[0,1].grid(True, alpha=0.3)
    
    # 1.3 Bubble ratio comparison
    bubble_ratio_a = df_a['BubbleRatio'].dropna()
    bubble_ratio_b = df_b['BubbleRatio'].dropna()
    axes[0,2].hist(bubble_ratio_a, bins=50, alpha=0.7, color='blue', label='Experiment A', density=True)
    axes[0,2].hist(bubble_ratio_b, bins=50, alpha=0.7, color='red', label='Experiment B', density=True)
    axes[0,2].axvline(x=1, color='black', linestyle='--', linewidth=2, label='Efficient Market (1.0)')
    axes[0,2].set_xlabel('Price/Fundamental Ratio')
    axes[0,2].set_ylabel('Density')
    axes[0,2].set_title('Price/Fundamental Ratio Distribution')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Row 2: Information Structure
    # 2.1 Last bubble comparison
    axes[1,0].hist(last_bubble_a, bins=20, alpha=0.7, color='blue', label='Experiment A', density=True)
    axes[1,0].hist(last_bubble_b, bins=20, alpha=0.7, color='red', label='Experiment B', density=True)
    axes[1,0].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Efficient Market (0)')
    axes[1,0].set_xlabel('Last Bubble Size')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Last Bubble Distribution by Experiment')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 2.2 Information treatment effect (Experiment A only)
    fund_shown = df_a[df_a['FundShown'] == 1]['Bubble'].dropna()
    fund_hidden = df_a[df_a['FundShown'] == 0]['Bubble'].dropna()
    axes[1,1].hist(fund_shown, bins=30, alpha=0.7, color='green', label='Fundamental Shown', density=True)
    axes[1,1].hist(fund_hidden, bins=30, alpha=0.7, color='orange', label='Fundamental Hidden', density=True)
    axes[1,1].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Efficient Market (0)')
    axes[1,1].set_xlabel('Bubble Size')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Information Treatment Effect (Experiment A)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 2.3 Professional share effect (Experiment B only)
    pro_share_effect = df_b.groupby('Session')['ProShare'].first()
    session_bubbles_b_series = session_bubbles_b
    axes[1,2].scatter(pro_share_effect, session_bubbles_b_series, s=100, alpha=0.7, color='red')
    axes[1,2].set_xlabel('Professional Share')
    axes[1,2].set_ylabel('Average Bubble Size')
    axes[1,2].set_title('Professional Share vs Bubble Size (Experiment B)')
    axes[1,2].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(pro_share_effect, session_bubbles_b_series, 1)
    p = np.poly1d(z)
    axes[1,2].plot(pro_share_effect, p(pro_share_effect), "r--", alpha=0.8, linewidth=2)
    
    # Row 3: Learning Across Sessions
    # 3.1 Session trends - Experiment A
    sessions_a = session_bubbles_a.index
    bubbles_a_values = session_bubbles_a.values
    axes[2,0].plot(sessions_a, bubbles_a_values, 'o-', color='blue', linewidth=2, markersize=8)
    axes[2,0].set_xlabel('Session')
    axes[2,0].set_ylabel('Average Bubble Size')
    axes[2,0].set_title('Learning Across Sessions - Experiment A')
    axes[2,0].grid(True, alpha=0.3)
    
    # Add trend line
    z_a = np.polyfit(sessions_a, bubbles_a_values, 1)
    p_a = np.poly1d(z_a)
    axes[2,0].plot(sessions_a, p_a(sessions_a), "b--", alpha=0.8, linewidth=2)
    
    # 3.2 Session trends - Experiment B
    sessions_b = session_bubbles_b.index
    bubbles_b_values = session_bubbles_b.values
    axes[2,1].plot(sessions_b, bubbles_b_values, 'o-', color='red', linewidth=2, markersize=8)
    axes[2,1].set_xlabel('Session')
    axes[2,1].set_ylabel('Average Bubble Size')
    axes[2,1].set_title('Learning Across Sessions - Experiment B')
    axes[2,1].grid(True, alpha=0.3)
    
    # Add trend line
    z_b = np.polyfit(sessions_b, bubbles_b_values, 1)
    p_b = np.poly1d(z_b)
    axes[2,1].plot(sessions_b, p_b(sessions_b), "r--", alpha=0.8, linewidth=2)
    
    # 3.3 Combined session comparison
    axes[2,2].plot(sessions_a, bubbles_a_values, 'o-', color='blue', linewidth=2, markersize=8, label='Experiment A')
    axes[2,2].plot(sessions_b, bubbles_b_values, 'o-', color='red', linewidth=2, markersize=8, label='Experiment B')
    axes[2,2].set_xlabel('Session')
    axes[2,2].set_ylabel('Average Bubble Size')
    axes[2,2].set_title('Learning Comparison: A vs B')
    axes[2,2].legend()
    axes[2,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('market_layer_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Market layer analysis visualization saved as 'market_layer_analysis.png'")

def main():
    """Main function to run market layer analysis"""
    print("="*80)
    print("MARKET LAYER ANALYSIS FOR BUBBLE MARKET EXPERIMENTS")
    print("="*80)
    
    # Load and prepare data
    df_a, df_b, df_combined = load_and_prepare_data()
    
    # Analysis Point #1: Market Efficiency
    bubble_a, bubble_b = test_market_efficiency(df_a, df_b)
    
    # Analysis Point #2: Information Structure
    last_bubble_a, last_bubble_b = test_information_structure(df_a, df_b)
    
    # Analysis Point #3: Learning Across Sessions
    session_bubbles_a, session_bubbles_b = test_learning_across_sessions(df_a, df_b)
    
    # Create visualizations
    create_market_visualizations(df_a, df_b, bubble_a, bubble_b, last_bubble_a, last_bubble_b, 
                                session_bubbles_a, session_bubbles_b)
    
    print("\n" + "="*80)
    print("MARKET LAYER ANALYSIS COMPLETE!")
    print("="*80)
    print("Generated files:")
    print("- market_layer_analysis.png")
    print("="*80)

if __name__ == "__main__":
    main()
