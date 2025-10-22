#!/usr/bin/env python3
"""
Learning Across Sessions Analysis (Analysis Point #3)
Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics

H0: bubble means equal across sessions
Test: One-way ANOVA / Kruskal-Wallis
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
    print("Loading data for Learning Analysis...")
    
    df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    
    # Calculate bubble metrics
    df_a['Bubble'] = df_a['LastPrice'] - df_a['Fundamental']
    df_b['Bubble'] = df_b['LastPrice'] - df_b['Fundamental']
    
    return df_a, df_b

def test_learning_across_sessions():
    """Test learning effects across sessions"""
    print("\n" + "="*80)
    print("ANALYSIS POINT #3: LEARNING ACROSS SESSIONS")
    print("="*80)
    print("H0: bubble means equal across sessions")
    print("Test: One-way ANOVA / Kruskal-Wallis")
    
    df_a, df_b = load_data()
    
    # Calculate session-level bubble means
    session_bubbles_a = df_a.groupby('Session')['Bubble'].mean()
    session_bubbles_b = df_b.groupby('Session')['Bubble'].mean()
    
    print(f"\nSession-level bubble means:")
    print("Choice A:")
    for session, bubble in session_bubbles_a.items():
        print(f"  Session {session}: {bubble:.2f}")
    
    print("Choice B:")
    for session, bubble in session_bubbles_b.items():
        print(f"  Session {session}: {bubble:.2f}")
    
    # Statistical tests
    print(f"\nStatistical Tests:")
    
    # One-way ANOVA for Choice A
    session_groups_a = [df_a[df_a['Session'] == s]['Bubble'].dropna() for s in df_a['Session'].unique()]
    f_stat_a, p_val_a = stats.f_oneway(*session_groups_a)
    
    print(f"\nOne-way ANOVA - Choice A:")
    print(f"  F = {f_stat_a:.3f}, p = {p_val_a:.6f}")
    
    # One-way ANOVA for Choice B
    session_groups_b = [df_b[df_b['Session'] == s]['Bubble'].dropna() for s in df_b['Session'].unique()]
    f_stat_b, p_val_b = stats.f_oneway(*session_groups_b)
    
    print(f"\nOne-way ANOVA - Choice B:")
    print(f"  F = {f_stat_b:.3f}, p = {p_val_b:.6f}")
    
    # Kruskal-Wallis tests
    h_stat_a, p_val_h_a = stats.kruskal(*session_groups_a)
    h_stat_b, p_val_h_b = stats.kruskal(*session_groups_b)
    
    print(f"\nKruskal-Wallis tests:")
    print(f"  Choice A: H = {h_stat_a:.3f}, p = {p_val_h_a:.6f}")
    print(f"  Choice B: H = {h_stat_b:.3f}, p = {p_val_h_b:.6f}")
    
    # Trend analysis
    print(f"\nTrend Analysis:")
    sessions_a = np.array(session_bubbles_a.index)
    bubbles_a = np.array(session_bubbles_a.values)
    slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(sessions_a, bubbles_a)
    
    sessions_b = np.array(session_bubbles_b.index)
    bubbles_b = np.array(session_bubbles_b.values)
    slope_b, intercept_b, r_value_b, p_value_b, std_err_b = stats.linregress(sessions_b, bubbles_b)
    
    print(f"  Choice A: slope = {slope_a:.3f}, R² = {r_value_a**2:.3f}, p = {p_value_a:.6f}")
    print(f"  Choice B: slope = {slope_b:.3f}, R² = {r_value_b**2:.3f}, p = {p_value_b:.6f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if p_val_a < 0.05:
        print(f"  ✓ Choice A: Significant differences across sessions (p < 0.05)")
        print(f"    → Learning effect detected in Choice A")
    else:
        print(f"  ✗ Choice A: No significant differences across sessions (p ≥ 0.05)")
        print(f"    → No learning effect in Choice A")
        
    if p_val_b < 0.05:
        print(f"  ✓ Choice B: Significant differences across sessions (p < 0.05)")
        print(f"    → Learning effect detected in Choice B")
    else:
        print(f"  ✗ Choice B: No significant differences across sessions (p ≥ 0.05)")
        print(f"    → No learning effect in Choice B")
    
    if slope_a < 0 and p_value_a < 0.05:
        print(f"  → Choice A shows declining bubbles (learning effect)")
    elif slope_a > 0 and p_value_a < 0.05:
        print(f"  → Choice A shows increasing bubbles (anti-learning)")
    else:
        print(f"  → Choice A shows no significant trend")
        
    if slope_b < 0 and p_value_b < 0.05:
        print(f"  → Choice B shows declining bubbles (learning effect)")
    elif slope_b > 0 and p_value_b < 0.05:
        print(f"  → Choice B shows increasing bubbles (anti-learning)")
    else:
        print(f"  → Choice B shows no significant trend")
    
    return session_bubbles_a, session_bubbles_b, df_a, df_b

def create_learning_visualizations(session_bubbles_a, session_bubbles_b, df_a, df_b):
    """Create visualizations for learning analysis"""
    print("\nCreating learning analysis visualizations...")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Learning Across Sessions Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Session trends - Choice A
    sessions_a = session_bubbles_a.index
    bubbles_a_values = session_bubbles_a.values
    axes[0,0].plot(sessions_a, bubbles_a_values, 'o-', color='blue', linewidth=3, markersize=10)
    axes[0,0].set_xlabel('Session', fontsize=12)
    axes[0,0].set_ylabel('Average Bubble Size', fontsize=12)
    axes[0,0].set_title('Learning Across Sessions - Choice A', fontsize=14, fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add trend line
    slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(sessions_a, bubbles_a_values)
    x_trend = np.linspace(sessions_a.min(), sessions_a.max(), 100)
    y_trend = slope_a * x_trend + intercept_a
    axes[0,0].plot(x_trend, y_trend, 'b--', alpha=0.8, linewidth=2, 
                  label=f'Trend: slope={slope_a:.2f}, R²={r_value_a**2:.3f}')
    axes[0,0].legend()
    
    # 2. Session trends - Choice B
    sessions_b = session_bubbles_b.index
    bubbles_b_values = session_bubbles_b.values
    axes[0,1].plot(sessions_b, bubbles_b_values, 'o-', color='red', linewidth=3, markersize=10)
    axes[0,1].set_xlabel('Session', fontsize=12)
    axes[0,1].set_ylabel('Average Bubble Size', fontsize=12)
    axes[0,1].set_title('Learning Across Sessions - Choice B', fontsize=14, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add trend line
    slope_b, intercept_b, r_value_b, p_value_b, std_err_b = stats.linregress(sessions_b, bubbles_b_values)
    x_trend = np.linspace(sessions_b.min(), sessions_b.max(), 100)
    y_trend = slope_b * x_trend + intercept_b
    axes[0,1].plot(x_trend, y_trend, 'r--', alpha=0.8, linewidth=2, 
                  label=f'Trend: slope={slope_b:.2f}, R²={r_value_b**2:.3f}')
    axes[0,1].legend()
    
    # 3. Combined comparison
    axes[1,0].plot(sessions_a, bubbles_a_values, 'o-', color='blue', linewidth=3, markersize=10, label='Choice A')
    axes[1,0].plot(sessions_b, bubbles_b_values, 'o-', color='red', linewidth=3, markersize=10, label='Choice B')
    axes[1,0].set_xlabel('Session', fontsize=12)
    axes[1,0].set_ylabel('Average Bubble Size', fontsize=12)
    axes[1,0].set_title('Learning Comparison: A vs B', fontsize=14, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Statistical summary
    axes[1,1].axis('off')
    
    # Calculate additional statistics
    f_stat_a, p_val_a = stats.f_oneway(*[df_a[df_a['Session'] == s]['Bubble'].dropna() for s in df_a['Session'].unique()])
    f_stat_b, p_val_b = stats.f_oneway(*[df_b[df_b['Session'] == s]['Bubble'].dropna() for s in df_b['Session'].unique()])
    
    summary_text = f"""
    STATISTICAL SUMMARY
    
    Choice A:
    • ANOVA F-statistic: {f_stat_a:.3f}
    • ANOVA p-value: {p_val_a:.6f}
    • Trend slope: {slope_a:.3f}
    • Trend R²: {r_value_a**2:.3f}
    • Trend p-value: {p_value_a:.6f}
    
    Choice B:
    • ANOVA F-statistic: {f_stat_b:.3f}
    • ANOVA p-value: {p_val_b:.6f}
    • Trend slope: {slope_b:.3f}
    • Trend R²: {r_value_b**2:.3f}
    • Trend p-value: {p_value_b:.6f}
    
    CONCLUSION:
    {'Significant learning effect' if p_val_b < 0.05 else 'No significant learning effect'}
    {'Declining bubbles' if slope_b < 0 and p_value_b < 0.05 else 'No declining trend'}
    """
    
    axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('learning_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Learning analysis saved as 'learning_analysis.png'")

def main():
    """Main function"""
    print("="*80)
    print("LEARNING ACROSS SESSIONS ANALYSIS")
    print("="*80)
    
    # Test learning effects
    session_bubbles_a, session_bubbles_b, df_a, df_b = test_learning_across_sessions()
    
    # Create visualizations
    create_learning_visualizations(session_bubbles_a, session_bubbles_b, df_a, df_b)
    
    print("\n" + "="*80)
    print("LEARNING ACROSS SESSIONS ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
