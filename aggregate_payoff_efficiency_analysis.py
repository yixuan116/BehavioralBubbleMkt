#!/usr/bin/env python3
"""
Aggregate Payoff Efficiency Analysis (N=12)
Behaviors Layer Analysis: Systematic Inefficiency in Payoffs

Hypothesis: H0: Payoff mean = Expected fundamental value vs H1: Payoff mean ≠ Expected value
Expected: Deviation → Systematic inefficiency
Data: Market-mean Final Payoff
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load both Choice A and B data for aggregate payoff efficiency analysis"""
    try:
        df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
        df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
        
        # Add experiment identifier
        df_a['Experiment'] = 'Choice A'
        df_b['Experiment'] = 'Choice B'
        
        # Combine datasets
        df = pd.concat([df_a, df_b], ignore_index=True)
        print(f"Loaded {len(df)} records total")
        print(f"Choice A: {len(df_a)} records")
        print(f"Choice B: {len(df_b)} records")
        return df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

def calculate_expected_fundamental_values(df):
    """Calculate expected fundamental values for each session"""
    session_fundamentals = []
    
    for session in df['Session'].unique():
        session_data = df[df['Session'] == session]
        
        # Calculate expected fundamental value for the session
        # This is the sum of all expected dividends over all periods
        expected_fundamental = session_data['Fundamental'].mean()
        
        # Calculate total expected dividends
        total_expected_dividends = session_data['Fundamental'].sum()
        
        # Calculate expected final payoff (cash + expected dividends)
        initial_cash = 600  # Starting cash per subject
        expected_final_payoff = initial_cash + total_expected_dividends
        
        session_fundamentals.append({
            'Session': session,
            'Experiment': session_data['Experiment'].iloc[0],
            'ExpectedFundamental': expected_fundamental,
            'TotalExpectedDividends': total_expected_dividends,
            'ExpectedFinalPayoff': expected_final_payoff,
            'NumPeriods': len(session_data['Period'].unique())
        })
    
    return pd.DataFrame(session_fundamentals)

def calculate_actual_payoffs(df):
    """Calculate actual final payoffs for each subject"""
    final_payoffs = []
    
    for session in df['Session'].unique():
        session_data = df[df['Session'] == session]
        for subject in session_data['SubjectID'].unique():
            subject_data = session_data[session_data['SubjectID'] == subject]
            
            # Get final period data
            final_period = subject_data[subject_data['Period'] == subject_data['Period'].max()]
            if len(final_period) > 0:
                final_row = final_period.iloc[0]
                
                # Calculate final payoff
                final_cash = final_row['Cash_t']
                final_shares = final_row['Shares_t']
                final_price = final_row['LastPrice']
                total_dividends = subject_data['DivEarn'].sum()
                
                final_payoff = final_cash + (final_shares * final_price) + total_dividends
                
                final_payoffs.append({
                    'Session': session,
                    'SubjectID': subject,
                    'Experiment': final_row['Experiment'],
                    'FinalPayoff': final_payoff,
                    'TotalDividends': total_dividends
                })
    
    return pd.DataFrame(final_payoffs)

def analyze_payoff_efficiency(actual_payoffs, expected_fundamentals):
    """Analyze aggregate payoff efficiency"""
    print("\n" + "="*60)
    print("AGGREGATE PAYOFF EFFICIENCY ANALYSIS (N=12)")
    print("="*60)
    
    # Calculate session-level mean payoffs
    session_payoffs = actual_payoffs.groupby(['Session', 'Experiment']).agg({
        'FinalPayoff': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    session_payoffs.columns = ['Session', 'Experiment', 'MeanPayoff', 'PayoffStd', 'NumSubjects']
    
    # Merge with expected fundamentals
    efficiency_analysis = pd.merge(session_payoffs, expected_fundamentals, on=['Session', 'Experiment'])
    
    # Calculate efficiency metrics
    efficiency_analysis['PayoffEfficiency'] = efficiency_analysis['MeanPayoff'] / efficiency_analysis['ExpectedFinalPayoff']
    efficiency_analysis['EfficiencyDeviation'] = efficiency_analysis['MeanPayoff'] - efficiency_analysis['ExpectedFinalPayoff']
    efficiency_analysis['EfficiencyRatio'] = efficiency_analysis['MeanPayoff'] / efficiency_analysis['ExpectedFinalPayoff']
    
    print(f"\nSession-level Efficiency Analysis:")
    print(efficiency_analysis[['Session', 'Experiment', 'MeanPayoff', 'ExpectedFinalPayoff', 
                              'EfficiencyRatio', 'EfficiencyDeviation']].to_string(index=False))
    
    # Overall efficiency analysis
    print(f"\nOverall Efficiency Statistics:")
    overall_mean_payoff = efficiency_analysis['MeanPayoff'].mean()
    overall_expected_payoff = efficiency_analysis['ExpectedFinalPayoff'].mean()
    overall_efficiency_ratio = overall_mean_payoff / overall_expected_payoff
    overall_deviation = overall_mean_payoff - overall_expected_payoff
    
    print(f"Overall mean actual payoff: {overall_mean_payoff:.2f}")
    print(f"Overall expected payoff: {overall_expected_payoff:.2f}")
    print(f"Overall efficiency ratio: {overall_efficiency_ratio:.4f}")
    print(f"Overall deviation: {overall_deviation:.2f}")
    
    # Statistical tests
    print(f"\nStatistical Tests:")
    
    # Test 1: One-sample t-test against expected value
    # H0: Mean payoff = Expected payoff vs H1: Mean payoff ≠ Expected payoff
    t_stat, p_value = stats.ttest_1samp(efficiency_analysis['MeanPayoff'], 
                                       efficiency_analysis['ExpectedFinalPayoff'].mean())
    
    print(f"Test 1: One-sample t-test against expected value")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    # Test 2: Efficiency ratio significantly different from 1
    # H0: Efficiency ratio = 1 vs H1: Efficiency ratio ≠ 1
    efficiency_ratios = efficiency_analysis['EfficiencyRatio']
    t_stat_ratio, p_value_ratio = stats.ttest_1samp(efficiency_ratios, 1.0)
    
    print(f"\nTest 2: Efficiency ratio ≠ 1")
    print(f"  Mean efficiency ratio: {efficiency_ratios.mean():.4f}")
    print(f"  t-statistic: {t_stat_ratio:.4f}")
    print(f"  p-value: {p_value_ratio:.4f}")
    
    # Test 3: Comparison between experiments
    choice_a = efficiency_analysis[efficiency_analysis['Experiment'] == 'Choice A']
    choice_b = efficiency_analysis[efficiency_analysis['Experiment'] == 'Choice B']
    
    if len(choice_a) > 1 and len(choice_b) > 1:
        t_stat_exp, p_value_exp = stats.ttest_ind(choice_a['EfficiencyRatio'], choice_b['EfficiencyRatio'])
        
        print(f"\nTest 3: Efficiency difference between experiments")
        print(f"  Choice A efficiency ratio: {choice_a['EfficiencyRatio'].mean():.4f}")
        print(f"  Choice B efficiency ratio: {choice_b['EfficiencyRatio'].mean():.4f}")
        print(f"  t-statistic: {t_stat_exp:.4f}")
        print(f"  p-value: {p_value_exp:.4f}")
    
    # Test 4: Systematic inefficiency
    # H0: No systematic deviation vs H1: Systematic deviation
    deviations = efficiency_analysis['EfficiencyDeviation']
    t_stat_dev, p_value_dev = stats.ttest_1samp(deviations, 0)
    
    print(f"\nTest 4: Systematic inefficiency")
    print(f"  Mean deviation: {deviations.mean():.2f}")
    print(f"  t-statistic: {t_stat_dev:.4f}")
    print(f"  p-value: {p_value_dev:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if p_value < 0.05:
        if overall_mean_payoff > overall_expected_payoff:
            print("✓ Significant over-efficiency detected")
            print("  → Actual payoffs are significantly higher than expected")
        else:
            print("✓ Significant under-efficiency detected")
            print("  → Actual payoffs are significantly lower than expected")
    else:
        print("✗ No significant deviation from expected payoffs")
        print("  → Markets show efficient payoff allocation")
    
    if p_value_ratio < 0.05:
        if efficiency_ratios.mean() > 1:
            print("✓ Efficiency ratio significantly greater than 1 (over-efficiency)")
        else:
            print("✓ Efficiency ratio significantly less than 1 (under-efficiency)")
    else:
        print("✗ Efficiency ratio not significantly different from 1")
    
    if p_value_dev < 0.05:
        print("✓ Systematic inefficiency detected")
        print("  → Consistent deviation from expected payoffs")
    else:
        print("✗ No systematic inefficiency detected")
        print("  → Payoffs are consistent with expectations")
    
    return efficiency_analysis, t_stat, p_value, t_stat_ratio, p_value_ratio

def create_visualizations(efficiency_analysis):
    """Create visualizations for aggregate payoff efficiency analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Aggregate Payoff Efficiency Analysis (N=12): Systematic Inefficiency', 
                 fontsize=16, fontweight='bold')
    
    # 1. Actual vs Expected payoffs
    ax1 = axes[0, 0]
    ax1.scatter(efficiency_analysis['ExpectedFinalPayoff'], efficiency_analysis['MeanPayoff'], 
               alpha=0.7, s=100, color='blue')
    
    # Add 45-degree line (perfect efficiency)
    min_payoff = min(efficiency_analysis['ExpectedFinalPayoff'].min(), efficiency_analysis['MeanPayoff'].min())
    max_payoff = max(efficiency_analysis['ExpectedFinalPayoff'].max(), efficiency_analysis['MeanPayoff'].max())
    ax1.plot([min_payoff, max_payoff], [min_payoff, max_payoff], 'r--', alpha=0.8, label='Perfect Efficiency')
    
    ax1.set_xlabel('Expected Final Payoff')
    ax1.set_ylabel('Actual Mean Payoff')
    ax1.set_title('Actual vs Expected Payoffs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Efficiency ratio by session
    ax2 = axes[0, 1]
    colors = ['red' if row['Experiment'] == 'Choice A' else 'blue' for _, row in efficiency_analysis.iterrows()]
    ax2.bar(range(len(efficiency_analysis)), efficiency_analysis['EfficiencyRatio'], 
           color=colors, alpha=0.7)
    ax2.axhline(1, color='black', linestyle='--', alpha=0.5, label='Perfect Efficiency (1.0)')
    ax2.set_xlabel('Session')
    ax2.set_ylabel('Efficiency Ratio')
    ax2.set_title('Efficiency Ratio by Session')
    ax2.set_xticks(range(len(efficiency_analysis)))
    ax2.set_xticklabels(efficiency_analysis['Session'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Efficiency deviation
    ax3 = axes[1, 0]
    ax3.bar(range(len(efficiency_analysis)), efficiency_analysis['EfficiencyDeviation'], 
           color=colors, alpha=0.7)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5, label='No Deviation')
    ax3.set_xlabel('Session')
    ax3.set_ylabel('Efficiency Deviation')
    ax3.set_title('Efficiency Deviation by Session')
    ax3.set_xticks(range(len(efficiency_analysis)))
    ax3.set_xticklabels(efficiency_analysis['Session'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Efficiency distribution
    ax4 = axes[1, 1]
    choice_a = efficiency_analysis[efficiency_analysis['Experiment'] == 'Choice A']
    choice_b = efficiency_analysis[efficiency_analysis['Experiment'] == 'Choice B']
    
    ax4.hist(choice_a['EfficiencyRatio'], alpha=0.7, label='Choice A', bins=5, color='red')
    ax4.hist(choice_b['EfficiencyRatio'], alpha=0.7, label='Choice B', bins=5, color='blue')
    ax4.axvline(1, color='black', linestyle='--', alpha=0.5, label='Perfect Efficiency')
    ax4.set_xlabel('Efficiency Ratio')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Efficiency Ratio Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('aggregate_payoff_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'aggregate_payoff_efficiency_analysis.png'")

def main():
    """Main analysis function"""
    print("Loading data for aggregate payoff efficiency analysis...")
    df = load_data()
    
    if df is None:
        return
    
    print("Calculating expected fundamental values...")
    expected_fundamentals = calculate_expected_fundamental_values(df)
    
    print("Calculating actual payoffs...")
    actual_payoffs = calculate_actual_payoffs(df)
    
    print(f"Analyzing {len(expected_fundamentals)} sessions")
    
    # Analyze payoff efficiency
    efficiency_analysis, t_stat, p_value, t_stat_ratio, p_value_ratio = analyze_payoff_efficiency(actual_payoffs, expected_fundamentals)
    
    # Create visualizations
    create_visualizations(efficiency_analysis)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Key finding: {'Significant inefficiency' if p_value < 0.05 else 'No significant inefficiency'}")
    print(f"Efficiency type: {'Over-efficiency' if efficiency_analysis['EfficiencyRatio'].mean() > 1 else 'Under-efficiency'}")
    print(f"Systematic deviation: {'Yes' if p_value_ratio < 0.05 else 'No'}")

if __name__ == "__main__":
    main()
