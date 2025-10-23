#!/usr/bin/env python3
"""
Outcome Inequality Analysis (N=8)
Outcomes Layer Analysis: Payoff Variance Comparison Between Choice A and B

Hypothesis: H0: Var_A = Var_B vs H1: Var_A ≠ Var_B
Expected: Var_A > Var_B → Increased inequality under low info
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
    """Load both Choice A and B data for outcome inequality analysis"""
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

def calculate_final_payoffs(df):
    """Calculate final payoffs for each subject"""
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

def analyze_outcome_inequality(payoffs_df):
    """Analyze outcome inequality between Choice A and B"""
    print("\n" + "="*60)
    print("OUTCOME INEQUALITY ANALYSIS (N=8)")
    print("="*60)
    
    # Separate by experiment
    choice_a = payoffs_df[payoffs_df['Experiment'] == 'Choice A']
    choice_b = payoffs_df[payoffs_df['Experiment'] == 'Choice B']
    
    print(f"\nSample sizes:")
    print(f"Choice A: {len(choice_a)} subjects")
    print(f"Choice B: {len(choice_b)} subjects")
    
    # Descriptive statistics
    print(f"\nFinal Payoff Statistics:")
    print(f"Choice A:")
    print(f"  Mean: {choice_a['FinalPayoff'].mean():.2f}")
    print(f"  Median: {choice_a['FinalPayoff'].median():.2f}")
    print(f"  Std: {choice_a['FinalPayoff'].std():.2f}")
    print(f"  Variance: {choice_a['FinalPayoff'].var():.2f}")
    print(f"  Min: {choice_a['FinalPayoff'].min():.2f}")
    print(f"  Max: {choice_a['FinalPayoff'].max():.2f}")
    print(f"  Range: {choice_a['FinalPayoff'].max() - choice_a['FinalPayoff'].min():.2f}")
    
    print(f"\nChoice B:")
    print(f"  Mean: {choice_b['FinalPayoff'].mean():.2f}")
    print(f"  Median: {choice_b['FinalPayoff'].median():.2f}")
    print(f"  Std: {choice_b['FinalPayoff'].std():.2f}")
    print(f"  Variance: {choice_b['FinalPayoff'].var():.2f}")
    print(f"  Min: {choice_b['FinalPayoff'].min():.2f}")
    print(f"  Max: {choice_b['FinalPayoff'].max():.2f}")
    print(f"  Range: {choice_b['FinalPayoff'].max() - choice_b['FinalPayoff'].min():.2f}")
    
    # Inequality measures
    print(f"\nInequality Measures:")
    
    # Coefficient of Variation (CV)
    cv_a = choice_a['FinalPayoff'].std() / choice_a['FinalPayoff'].mean()
    cv_b = choice_b['FinalPayoff'].std() / choice_b['FinalPayoff'].mean()
    
    print(f"Coefficient of Variation:")
    print(f"  Choice A: {cv_a:.4f}")
    print(f"  Choice B: {cv_b:.4f}")
    print(f"  Difference: {cv_a - cv_b:.4f}")
    
    # Gini coefficient (approximation using relative mean difference)
    def gini_coefficient(x):
        x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(x)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    gini_a = gini_coefficient(choice_a['FinalPayoff'])
    gini_b = gini_coefficient(choice_b['FinalPayoff'])
    
    print(f"\nGini Coefficient:")
    print(f"  Choice A: {gini_a:.4f}")
    print(f"  Choice B: {gini_b:.4f}")
    print(f"  Difference: {gini_a - gini_b:.4f}")
    
    # Statistical tests for variance equality
    print(f"\nStatistical Tests:")
    
    # Levene's test for equal variances
    levene_result = stats.levene(choice_a['FinalPayoff'], choice_b['FinalPayoff'])
    print(f"Levene's test for equal variances:")
    print(f"  W-statistic: {levene_result.statistic:.4f}")
    print(f"  p-value: {levene_result.pvalue:.4f}")
    
    # F-test for variance ratio
    var_a = choice_a['FinalPayoff'].var()
    var_b = choice_b['FinalPayoff'].var()
    f_ratio = var_a / var_b if var_b > 0 else np.inf
    
    # F-test (two-tailed)
    f_test = stats.f.cdf(f_ratio, len(choice_a)-1, len(choice_b)-1)
    f_pvalue = 2 * min(f_test, 1 - f_test)
    
    print(f"\nF-test for variance ratio:")
    print(f"  Choice A variance: {var_a:.2f}")
    print(f"  Choice B variance: {var_b:.2f}")
    print(f"  F-ratio (A/B): {f_ratio:.4f}")
    print(f"  p-value: {f_pvalue:.4f}")
    
    # Brown-Forsythe test (alternative to Levene's)
    bf_result = stats.levene(choice_a['FinalPayoff'], choice_b['FinalPayoff'], center='median')
    print(f"\nBrown-Forsythe test (median-centered):")
    print(f"  W-statistic: {bf_result.statistic:.4f}")
    print(f"  p-value: {bf_result.pvalue:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if levene_result.pvalue < 0.05:
        if var_a > var_b:
            print("✓ Significant difference in payoff variance between experiments")
            print("  → Choice A shows higher inequality (larger variance)")
        else:
            print("✓ Significant difference in payoff variance between experiments")
            print("  → Choice B shows higher inequality (larger variance)")
    else:
        print("✗ No significant difference in payoff variance between experiments")
    
    if f_pvalue < 0.05:
        print("✓ F-test confirms significant variance difference")
    else:
        print("✗ F-test shows no significant variance difference")
    
    # Effect size for variance difference
    variance_effect_size = abs(var_a - var_b) / max(var_a, var_b)
    print(f"\nVariance difference effect size: {variance_effect_size:.3f}")
    
    return choice_a, choice_b, levene_result, f_ratio, f_pvalue, cv_a, cv_b, gini_a, gini_b

def create_visualizations(choice_a, choice_b):
    """Create visualizations for outcome inequality analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Outcome Inequality Analysis (N=8): Payoff Distribution Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. Distribution comparison
    ax1 = axes[0, 0]
    ax1.hist(choice_a['FinalPayoff'], alpha=0.7, label='Choice A', bins=20, color='red')
    ax1.hist(choice_b['FinalPayoff'], alpha=0.7, label='Choice B', bins=20, color='blue')
    ax1.set_xlabel('Final Payoff (Francs)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Final Payoff Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = axes[0, 1]
    data_for_box = [choice_a['FinalPayoff'], choice_b['FinalPayoff']]
    labels = ['Choice A', 'Choice B']
    bp = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('red')
    bp['boxes'][1].set_facecolor('blue')
    ax2.set_ylabel('Final Payoff (Francs)')
    ax2.set_title('Final Payoff Box Plot Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution function
    ax3 = axes[1, 0]
    choice_a_sorted = np.sort(choice_a['FinalPayoff'])
    choice_b_sorted = np.sort(choice_b['FinalPayoff'])
    
    ax3.plot(choice_a_sorted, np.linspace(0, 1, len(choice_a_sorted)), 
             label='Choice A', color='red', linewidth=2)
    ax3.plot(choice_b_sorted, np.linspace(0, 1, len(choice_b_sorted)), 
             label='Choice B', color='blue', linewidth=2)
    ax3.set_xlabel('Final Payoff (Francs)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Function')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Inequality measures comparison
    ax4 = axes[1, 1]
    measures = ['CV', 'Gini', 'Std Dev']
    choice_a_values = [choice_a['FinalPayoff'].std() / choice_a['FinalPayoff'].mean(),
                      gini_coefficient(choice_a['FinalPayoff']),
                      choice_a['FinalPayoff'].std()]
    choice_b_values = [choice_b['FinalPayoff'].std() / choice_b['FinalPayoff'].mean(),
                       gini_coefficient(choice_b['FinalPayoff']),
                       choice_b['FinalPayoff'].std()]
    
    x = np.arange(len(measures))
    width = 0.35
    
    ax4.bar(x - width/2, choice_a_values, width, label='Choice A', color='red', alpha=0.7)
    ax4.bar(x + width/2, choice_b_values, width, label='Choice B', color='blue', alpha=0.7)
    
    ax4.set_xlabel('Inequality Measures')
    ax4.set_ylabel('Value')
    ax4.set_title('Inequality Measures Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(measures)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outcome_inequality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'outcome_inequality_analysis.png'")

def gini_coefficient(x):
    """Calculate Gini coefficient"""
    x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def main():
    """Main analysis function"""
    print("Loading data for outcome inequality analysis...")
    df = load_data()
    
    if df is None:
        return
    
    print("Calculating final payoffs...")
    payoffs_df = calculate_final_payoffs(df)
    
    print(f"Calculated final payoffs for {len(payoffs_df)} subjects")
    
    # Analyze outcome inequality
    choice_a, choice_b, levene_result, f_ratio, f_pvalue, cv_a, cv_b, gini_a, gini_b = analyze_outcome_inequality(payoffs_df)
    
    # Create visualizations
    create_visualizations(choice_a, choice_b)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Key finding: {'Significant inequality difference' if levene_result.pvalue < 0.05 else 'No significant inequality difference'} between experiments")
    print(f"Variance ratio (A/B): {f_ratio:.3f}")
    print(f"CV difference (A-B): {cv_a - cv_b:.3f}")

if __name__ == "__main__":
    main()
