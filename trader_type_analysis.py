#!/usr/bin/env python3
"""
Trader Type Effect Analysis (N=4)
Individual Layer Analysis: Professional vs Student Trader Performance

Hypothesis: H0: μ_pro = μ_student vs H1: μ_pro ≠ μ_student
Expected: μ_pro > μ_student → Higher average returns for professionals
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
    """Load Choice B data for trader type analysis"""
    try:
        df = pd.read_csv('Experiment_B_Trading_Data.csv')
        print(f"Loaded {len(df)} records from Choice B")
        return df
    except FileNotFoundError:
        print("Error: Experiment_B_Trading_Data.csv not found")
        return None

def calculate_final_payoffs(df):
    """Calculate final payoffs for each subject"""
    # Group by Session and SubjectID to get final state
    final_payoffs = []
    
    for session in df['Session'].unique():
        session_data = df[df['Session'] == session]
        for subject in session_data['SubjectID'].unique():
            subject_data = session_data[session_data['SubjectID'] == subject]
            
            # Get final period data
            final_period = subject_data[subject_data['Period'] == subject_data['Period'].max()]
            if len(final_period) > 0:
                final_row = final_period.iloc[0]
                
                # Calculate final payoff: Cash + (Shares × Final Price) + Total Dividends
                final_cash = final_row['Cash_t']
                final_shares = final_row['Shares_t']
                final_price = final_row['LastPrice']
                total_dividends = subject_data['DivEarn'].sum()
                
                final_payoff = final_cash + (final_shares * final_price) + total_dividends
                
                final_payoffs.append({
                    'Session': session,
                    'SubjectID': subject,
                    'Role': final_row['Role'],
                    'ProShare': final_row['ProShare'],
                    'FinalPayoff': final_payoff,
                    'FinalCash': final_cash,
                    'FinalShares': final_shares,
                    'TotalDividends': total_dividends
                })
    
    return pd.DataFrame(final_payoffs)

def analyze_trader_performance(payoffs_df):
    """Analyze professional vs student trader performance"""
    print("\n" + "="*60)
    print("TRADER TYPE EFFECT ANALYSIS (N=4)")
    print("="*60)
    
    # Separate professional and student traders
    pro_traders = payoffs_df[payoffs_df['Role'] == 'Pro']
    student_traders = payoffs_df[payoffs_df['Role'] == 'Student']
    
    print(f"\nSample sizes:")
    print(f"Professional traders: {len(pro_traders)}")
    print(f"Student traders: {len(student_traders)}")
    
    # Descriptive statistics
    print(f"\nFinal Payoff Statistics:")
    print(f"Professional traders:")
    print(f"  Mean: {pro_traders['FinalPayoff'].mean():.2f}")
    print(f"  Median: {pro_traders['FinalPayoff'].median():.2f}")
    print(f"  Std: {pro_traders['FinalPayoff'].std():.2f}")
    print(f"  Min: {pro_traders['FinalPayoff'].min():.2f}")
    print(f"  Max: {pro_traders['FinalPayoff'].max():.2f}")
    
    print(f"\nStudent traders:")
    print(f"  Mean: {student_traders['FinalPayoff'].mean():.2f}")
    print(f"  Median: {student_traders['FinalPayoff'].median():.2f}")
    print(f"  Std: {student_traders['FinalPayoff'].std():.2f}")
    print(f"  Min: {student_traders['FinalPayoff'].min():.2f}")
    print(f"  Max: {student_traders['FinalPayoff'].max():.2f}")
    
    # Statistical tests
    print(f"\nStatistical Tests:")
    
    # Normality tests
    pro_shapiro = stats.shapiro(pro_traders['FinalPayoff'])
    student_shapiro = stats.shapiro(student_traders['FinalPayoff'])
    
    print(f"Normality tests (Shapiro-Wilk):")
    print(f"  Professional traders: W = {pro_shapiro.statistic:.4f}, p = {pro_shapiro.pvalue:.4f}")
    print(f"  Student traders: W = {student_shapiro.statistic:.4f}, p = {student_shapiro.pvalue:.4f}")
    
    # Two-sample t-test
    ttest_result = stats.ttest_ind(pro_traders['FinalPayoff'], student_traders['FinalPayoff'])
    print(f"\nTwo-sample t-test:")
    print(f"  t-statistic: {ttest_result.statistic:.4f}")
    print(f"  p-value: {ttest_result.pvalue:.4f}")
    
    # Mann-Whitney U test (non-parametric)
    mw_result = stats.mannwhitneyu(pro_traders['FinalPayoff'], student_traders['FinalPayoff'], 
                                 alternative='two-sided')
    print(f"\nMann-Whitney U test:")
    print(f"  U-statistic: {mw_result.statistic:.4f}")
    print(f"  p-value: {mw_result.pvalue:.4f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(pro_traders)-1)*pro_traders['FinalPayoff'].var() + 
                         (len(student_traders)-1)*student_traders['FinalPayoff'].var()) / 
                        (len(pro_traders) + len(student_traders) - 2))
    cohens_d = (pro_traders['FinalPayoff'].mean() - student_traders['FinalPayoff'].mean()) / pooled_std
    print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if ttest_result.pvalue < 0.05:
        if pro_traders['FinalPayoff'].mean() > student_traders['FinalPayoff'].mean():
            print("✓ Professional traders have significantly higher final payoffs")
        else:
            print("✓ Student traders have significantly higher final payoffs")
    else:
        print("✗ No significant difference in final payoffs between trader types")
    
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    print(f"Effect size: {effect_size} (|d| = {abs(cohens_d):.3f})")
    
    return pro_traders, student_traders, ttest_result, mw_result, cohens_d

def create_visualizations(pro_traders, student_traders, payoffs_df):
    """Create visualizations for trader type analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Trader Type Effect Analysis (N=4): Professional vs Student Performance', 
                 fontsize=16, fontweight='bold')
    
    # 1. Distribution comparison
    ax1 = axes[0, 0]
    ax1.hist(pro_traders['FinalPayoff'], alpha=0.7, label='Professional', bins=20, color='blue')
    ax1.hist(student_traders['FinalPayoff'], alpha=0.7, label='Student', bins=20, color='orange')
    ax1.set_xlabel('Final Payoff (Francs)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Final Payoff Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = axes[0, 1]
    data_for_box = [pro_traders['FinalPayoff'], student_traders['FinalPayoff']]
    labels = ['Professional', 'Student']
    bp = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('orange')
    ax2.set_ylabel('Final Payoff (Francs)')
    ax2.set_title('Final Payoff Box Plot Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Professional share vs performance
    ax3 = axes[1, 0]
    pro_share_performance = payoffs_df.groupby(['Session', 'ProShare'])['FinalPayoff'].agg(['mean', 'std']).reset_index()
    ax3.scatter(pro_share_performance['ProShare'], pro_share_performance['mean'], 
               s=100, alpha=0.7, color='green')
    ax3.errorbar(pro_share_performance['ProShare'], pro_share_performance['mean'], 
                yerr=pro_share_performance['std'], fmt='none', color='green', alpha=0.5)
    ax3.set_xlabel('Professional Share in Session')
    ax3.set_ylabel('Mean Final Payoff (Francs)')
    ax3.set_title('Professional Share vs Session Performance')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance by session
    ax4 = axes[1, 1]
    session_performance = payoffs_df.groupby(['Session', 'Role'])['FinalPayoff'].mean().unstack()
    session_performance.plot(kind='bar', ax=ax4, color=['blue', 'orange'])
    ax4.set_xlabel('Session')
    ax4.set_ylabel('Mean Final Payoff (Francs)')
    ax4.set_title('Performance by Session and Trader Type')
    ax4.legend(['Professional', 'Student'])
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('trader_type_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'trader_type_analysis.png'")

def main():
    """Main analysis function"""
    print("Loading Choice B data for trader type analysis...")
    df = load_data()
    
    if df is None:
        return
    
    print("Calculating final payoffs for each subject...")
    payoffs_df = calculate_final_payoffs(df)
    
    print(f"Calculated final payoffs for {len(payoffs_df)} subjects")
    
    # Analyze trader performance
    pro_traders, student_traders, ttest_result, mw_result, cohens_d = analyze_trader_performance(payoffs_df)
    
    # Create visualizations
    create_visualizations(pro_traders, student_traders, payoffs_df)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Key finding: {'Professional traders outperform' if ttest_result.pvalue < 0.05 and pro_traders['FinalPayoff'].mean() > student_traders['FinalPayoff'].mean() else 'No significant difference'} in final payoffs")
    print(f"Effect size: {abs(cohens_d):.3f} ({'negligible' if abs(cohens_d) < 0.2 else 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'})")

if __name__ == "__main__":
    main()
