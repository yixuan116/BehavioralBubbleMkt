#!/usr/bin/env python3
"""
Simple Bubble Analysis: Choice A vs Choice B
Just LastPrice vs Fundamental Value
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_data():
    """Load Choice A and Choice B data"""
    print("Loading data...")
    df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    
    print(f"✓ Choice A: {len(df_a)} records")
    print(f"✓ Choice B: {len(df_b)} records")
    return df_a, df_b

def calculate_bubble_by_period(df, experiment_name):
    """Calculate bubble % by period"""
    print(f"\nCalculating bubble % for {experiment_name}...")
    
    # Remove NaN values first
    df_clean = df.dropna(subset=['LastPrice', 'Fundamental'])
    print(f"  Removed {len(df) - len(df_clean)} records with NaN values")
    
    if experiment_name == 'Choice A':
        # Group by Session, Market, Period
        period_data = df_clean.groupby(['Session', 'Market', 'Period']).agg({
            'LastPrice': 'mean',
            'Fundamental': 'first'
        }).reset_index()
    else:  # Choice B
        # Group by Session, Period
        period_data = df_clean.groupby(['Session', 'Period']).agg({
            'LastPrice': 'mean',
            'Fundamental': 'first'
        }).reset_index()
    
    # Calculate bubble %
    period_data['BubblePct'] = ((period_data['LastPrice'] - period_data['Fundamental']) / 
                                period_data['Fundamental']) * 100
    
    return period_data

def create_simple_bubble_chart(df_a_periods, df_b_periods):
    """Create simple bubble comparison chart"""
    print("\nCreating simple bubble chart...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Choice A: Period 1-15 average across all markets
    choice_a_periods = df_a_periods.groupby('Period')['BubblePct'].mean()
    
    # Choice B: Period 1-15 average
    choice_b_periods = df_b_periods.groupby('Period')['BubblePct'].mean()
    
    # Plot 1: Period trends
    ax1.plot(choice_a_periods.index, choice_a_periods.values, 'ro-', 
             linewidth=2, markersize=6, label='Choice A')
    ax1.plot(choice_b_periods.index, choice_b_periods.values, 'bs-', 
             linewidth=2, markersize=6, label='Choice B')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Period', fontsize=12)
    ax1.set_ylabel('Bubble %', fontsize=12)
    ax1.set_title('Bubble % Over Periods', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution comparison
    ax2.hist(df_a_periods['BubblePct'], bins=20, alpha=0.7, label='Choice A', 
             color='red', density=True)
    ax2.hist(df_b_periods['BubblePct'], bins=20, alpha=0.7, label='Choice B', 
             color='blue', density=True)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Bubble %', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Bubble % Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_bubble_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Simple bubble chart saved as 'simple_bubble_analysis.png'")

def print_basic_stats(df_a_periods, df_b_periods):
    """Print basic bubble statistics"""
    print("\n" + "="*60)
    print("BASIC BUBBLE STATISTICS")
    print("="*60)
    
    print("\nChoice A:")
    print(f"  Mean Bubble %: {df_a_periods['BubblePct'].mean():.2f}%")
    print(f"  Median Bubble %: {df_a_periods['BubblePct'].median():.2f}%")
    print(f"  Max Bubble %: {df_a_periods['BubblePct'].max():.2f}%")
    print(f"  Min Bubble %: {df_a_periods['BubblePct'].min():.2f}%")
    print(f"  Std Dev: {df_a_periods['BubblePct'].std():.2f}%")
    print(f"  Above Fundamental: {(df_a_periods['BubblePct'] > 0).mean()*100:.1f}%")
    
    print("\nChoice B:")
    print(f"  Mean Bubble %: {df_b_periods['BubblePct'].mean():.2f}%")
    print(f"  Median Bubble %: {df_b_periods['BubblePct'].median():.2f}%")
    print(f"  Max Bubble %: {df_b_periods['BubblePct'].max():.2f}%")
    print(f"  Min Bubble %: {df_b_periods['BubblePct'].min():.2f}%")
    print(f"  Std Dev: {df_b_periods['BubblePct'].std():.2f}%")
    print(f"  Above Fundamental: {(df_b_periods['BubblePct'] > 0).mean()*100:.1f}%")
    
    # Statistical tests
    print("\n" + "="*60)
    print("STATISTICAL TESTS (Choice A vs Choice B)")
    print("="*60)
    
    # T-test
    t_stat, t_p = stats.ttest_ind(df_a_periods['BubblePct'], df_b_periods['BubblePct'])
    print(f"\nT-test:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {t_p:.3f}")
    print(f"  Significant: {'Yes' if t_p < 0.05 else 'No'} (α=0.05)")
    
    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(df_a_periods['BubblePct'], df_b_periods['BubblePct'])
    print(f"\nMann-Whitney U test:")
    print(f"  U-statistic: {u_stat:.0f}")
    print(f"  p-value: {u_p:.3f}")
    print(f"  Significant: {'Yes' if u_p < 0.05 else 'No'} (α=0.05)")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(df_a_periods)-1)*df_a_periods['BubblePct'].std()**2 + 
                         (len(df_b_periods)-1)*df_b_periods['BubblePct'].std()**2) / 
                        (len(df_a_periods) + len(df_b_periods) - 2))
    cohens_d = (df_a_periods['BubblePct'].mean() - df_b_periods['BubblePct'].mean()) / pooled_std
    print(f"\nEffect Size:")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  Interpretation: {'Small' if abs(cohens_d) < 0.2 else 'Medium' if abs(cohens_d) < 0.5 else 'Large'} effect")
    
    # Simple comparison
    print(f"\nDifference (A - B): {df_a_periods['BubblePct'].mean() - df_b_periods['BubblePct'].mean():.2f}%")

def main():
    """Main function"""
    print("SIMPLE BUBBLE ANALYSIS")
    print("="*30)
    
    # Load data
    df_a, df_b = load_data()
    
    # Calculate bubble by period
    df_a_periods = calculate_bubble_by_period(df_a, 'Choice A')
    df_b_periods = calculate_bubble_by_period(df_b, 'Choice B')
    
    # Create chart
    create_simple_bubble_chart(df_a_periods, df_b_periods)
    
    # Print stats
    print_basic_stats(df_a_periods, df_b_periods)
    
    print("\n✓ Simple bubble analysis completed!")
    
    return df_a_periods, df_b_periods

if __name__ == "__main__":
    main()
