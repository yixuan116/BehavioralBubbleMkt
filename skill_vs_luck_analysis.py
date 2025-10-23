#!/usr/bin/env python3
"""
Skill vs Luck Analysis (N=6)
Individual Layer Analysis: Skill Heterogeneity in Trader Performance

Hypothesis: H0: σ²_within = σ²_expected vs H1: σ²_within ≠ σ²_expected
Expected: Significant variance differences → Evidence of skill heterogeneity
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
    """Load Choice B data for skill vs luck analysis"""
    try:
        df = pd.read_csv('Experiment_B_Trading_Data.csv')
        print(f"Loaded {len(df)} records from Choice B")
        return df
    except FileNotFoundError:
        print("Error: Experiment_B_Trading_Data.csv not found")
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
                    'Role': final_row['Role'],
                    'ProShare': final_row['ProShare'],
                    'FinalPayoff': final_payoff,
                    'TotalDividends': total_dividends
                })
    
    return pd.DataFrame(final_payoffs)

def calculate_skill_metrics(payoffs_df):
    """Calculate skill-related metrics for each subject"""
    skill_metrics = []
    
    for session in payoffs_df['Session'].unique():
        session_data = payoffs_df[payoffs_df['Session'] == session]
        
        # Calculate session statistics
        session_mean = session_data['FinalPayoff'].mean()
        session_std = session_data['FinalPayoff'].std()
        
        for _, subject in session_data.iterrows():
            # Calculate relative performance
            relative_performance = (subject['FinalPayoff'] - session_mean) / session_std if session_std > 0 else 0
            
            # Calculate percentile rank within session
            percentile_rank = stats.percentileofscore(session_data['FinalPayoff'], subject['FinalPayoff'])
            
            skill_metrics.append({
                'Session': session,
                'SubjectID': subject['SubjectID'],
                'Role': subject['Role'],
                'ProShare': subject['ProShare'],
                'FinalPayoff': subject['FinalPayoff'],
                'RelativePerformance': relative_performance,
                'PercentileRank': percentile_rank,
                'SessionMean': session_mean,
                'SessionStd': session_std
            })
    
    return pd.DataFrame(skill_metrics)

def analyze_skill_heterogeneity(skill_df):
    """Analyze skill heterogeneity in trader performance"""
    print("\n" + "="*60)
    print("SKILL VS LUCK ANALYSIS (N=6)")
    print("="*60)
    
    # Separate professional and student traders
    pro_traders = skill_df[skill_df['Role'] == 'Pro']
    student_traders = skill_df[skill_df['Role'] == 'Student']
    
    print(f"\nSample sizes:")
    print(f"Professional traders: {len(pro_traders)}")
    print(f"Student traders: {len(student_traders)}")
    
    # Descriptive statistics for relative performance
    print(f"\nRelative Performance Statistics (standardized):")
    print(f"Professional traders:")
    print(f"  Mean: {pro_traders['RelativePerformance'].mean():.4f}")
    print(f"  Std: {pro_traders['RelativePerformance'].std():.4f}")
    print(f"  Min: {pro_traders['RelativePerformance'].min():.4f}")
    print(f"  Max: {pro_traders['RelativePerformance'].max():.4f}")
    
    print(f"\nStudent traders:")
    print(f"  Mean: {student_traders['RelativePerformance'].mean():.4f}")
    print(f"  Std: {student_traders['RelativePerformance'].std():.4f}")
    print(f"  Min: {student_traders['RelativePerformance'].min():.4f}")
    print(f"  Max: {student_traders['RelativePerformance'].max():.4f}")
    
    # Variance tests
    print(f"\nVariance Analysis:")
    
    # Levene's test for equal variances
    levene_result = stats.levene(pro_traders['RelativePerformance'], 
                               student_traders['RelativePerformance'])
    print(f"Levene's test for equal variances:")
    print(f"  W-statistic: {levene_result.statistic:.4f}")
    print(f"  p-value: {levene_result.pvalue:.4f}")
    
    # F-test for variance ratio
    pro_var = pro_traders['RelativePerformance'].var()
    student_var = student_traders['RelativePerformance'].var()
    f_ratio = pro_var / student_var if student_var > 0 else np.inf
    
    # F-test (two-tailed)
    f_test = stats.f.cdf(f_ratio, len(pro_traders)-1, len(student_traders)-1)
    f_pvalue = 2 * min(f_test, 1 - f_test)
    
    print(f"\nF-test for variance ratio:")
    print(f"  Professional variance: {pro_var:.4f}")
    print(f"  Student variance: {student_var:.4f}")
    print(f"  F-ratio: {f_ratio:.4f}")
    print(f"  p-value: {f_pvalue:.4f}")
    
    # Coefficient of variation (CV) comparison
    pro_cv = pro_traders['RelativePerformance'].std() / abs(pro_traders['RelativePerformance'].mean()) if pro_traders['RelativePerformance'].mean() != 0 else np.inf
    student_cv = student_traders['RelativePerformance'].std() / abs(student_traders['RelativePerformance'].mean()) if student_traders['RelativePerformance'].mean() != 0 else np.inf
    
    print(f"\nCoefficient of Variation:")
    print(f"  Professional traders: {pro_cv:.4f}")
    print(f"  Student traders: {student_cv:.4f}")
    
    # Percentile rank analysis
    print(f"\nPercentile Rank Analysis:")
    print(f"Professional traders:")
    print(f"  Mean percentile: {pro_traders['PercentileRank'].mean():.2f}")
    print(f"  Std percentile: {pro_traders['PercentileRank'].std():.2f}")
    
    print(f"\nStudent traders:")
    print(f"  Mean percentile: {student_traders['PercentileRank'].mean():.2f}")
    print(f"  Std percentile: {student_traders['PercentileRank'].std():.2f}")
    
    # Skill consistency analysis (within-subject variance)
    print(f"\nSkill Consistency Analysis:")
    
    # Calculate within-subject variance for each trader type
    pro_consistency = []
    student_consistency = []
    
    for session in skill_df['Session'].unique():
        session_data = skill_df[skill_df['Session'] == session]
        pro_session = session_data[session_data['Role'] == 'Pro']
        student_session = session_data[session_data['Role'] == 'Student']
        
        if len(pro_session) > 1:
            pro_consistency.append(pro_session['RelativePerformance'].var())
        if len(student_session) > 1:
            student_consistency.append(student_session['RelativePerformance'].var())
    
    if pro_consistency and student_consistency:
        print(f"  Professional within-session variance: {np.mean(pro_consistency):.4f}")
        print(f"  Student within-session variance: {np.mean(student_consistency):.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if levene_result.pvalue < 0.05:
        print("✓ Significant difference in performance variance between trader types")
        if pro_var > student_var:
            print("  → Professional traders show higher performance variance (more skill heterogeneity)")
        else:
            print("  → Student traders show higher performance variance (more skill heterogeneity)")
    else:
        print("✗ No significant difference in performance variance between trader types")
    
    if f_pvalue < 0.05:
        print("✓ F-test confirms significant variance difference")
    else:
        print("✗ F-test shows no significant variance difference")
    
    return pro_traders, student_traders, levene_result, f_ratio, f_pvalue

def create_visualizations(pro_traders, student_traders, skill_df):
    """Create visualizations for skill vs luck analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Skill vs Luck Analysis (N=6): Performance Heterogeneity', 
                 fontsize=16, fontweight='bold')
    
    # 1. Relative performance distribution
    ax1 = axes[0, 0]
    ax1.hist(pro_traders['RelativePerformance'], alpha=0.7, label='Professional', bins=15, color='blue')
    ax1.hist(student_traders['RelativePerformance'], alpha=0.7, label='Student', bins=15, color='orange')
    ax1.axvline(0, color='red', linestyle='--', alpha=0.7, label='Session Mean')
    ax1.set_xlabel('Relative Performance (Standardized)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Relative Performance Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot of relative performance
    ax2 = axes[0, 1]
    data_for_box = [pro_traders['RelativePerformance'], student_traders['RelativePerformance']]
    labels = ['Professional', 'Student']
    bp = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('orange')
    ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Relative Performance (Standardized)')
    ax2.set_title('Relative Performance Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Percentile rank distribution
    ax3 = axes[1, 0]
    ax3.hist(pro_traders['PercentileRank'], alpha=0.7, label='Professional', bins=15, color='blue')
    ax3.hist(student_traders['PercentileRank'], alpha=0.7, label='Student', bins=15, color='orange')
    ax3.axvline(50, color='red', linestyle='--', alpha=0.7, label='Median (50th percentile)')
    ax3.set_xlabel('Percentile Rank in Session')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Percentile Rank Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance variance by professional share
    ax4 = axes[1, 1]
    variance_by_proshare = skill_df.groupby('ProShare')['RelativePerformance'].agg(['mean', 'var']).reset_index()
    ax4.scatter(variance_by_proshare['ProShare'], variance_by_proshare['var'], 
               s=100, alpha=0.7, color='green')
    ax4.set_xlabel('Professional Share in Session')
    ax4.set_ylabel('Performance Variance')
    ax4.set_title('Performance Variance vs Professional Share')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('skill_vs_luck_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'skill_vs_luck_analysis.png'")

def main():
    """Main analysis function"""
    print("Loading Choice B data for skill vs luck analysis...")
    df = load_data()
    
    if df is None:
        return
    
    print("Calculating final payoffs for each subject...")
    payoffs_df = calculate_final_payoffs(df)
    
    print("Calculating skill metrics...")
    skill_df = calculate_skill_metrics(payoffs_df)
    
    print(f"Calculated skill metrics for {len(skill_df)} subjects")
    
    # Analyze skill heterogeneity
    pro_traders, student_traders, levene_result, f_ratio, f_pvalue = analyze_skill_heterogeneity(skill_df)
    
    # Create visualizations
    create_visualizations(pro_traders, student_traders, skill_df)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Key finding: {'Significant skill heterogeneity' if levene_result.pvalue < 0.05 else 'No significant skill heterogeneity'} between trader types")
    print(f"Variance ratio (Pro/Student): {f_ratio:.3f}")

if __name__ == "__main__":
    main()
