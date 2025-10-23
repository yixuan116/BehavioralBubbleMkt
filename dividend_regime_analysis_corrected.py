#!/usr/bin/env python3
"""
Dividend Regime Analysis (N=5) - CORRECTED
Individual Layer Analysis: High vs Low Dividend Effects on Payoffs

Hypothesis: H0: μ_highdiv = μ_lowdiv vs H1: μ_highdiv ≠ μ_lowdiv
Expected: μ_highdiv > μ_lowdiv → Structural luck component
Data: Final Payoff by Dividend level (not by session)
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
    """Load both Choice A and B data for dividend regime analysis"""
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

def classify_dividend_levels(df):
    """Classify dividend levels for each period"""
    # Get all unique dividend values
    unique_dividends = sorted(df['Dividend'].unique())
    print(f"Unique dividend values: {unique_dividends}")
    
    # Classify as high, medium, low dividend levels
    # Use tertiles (33rd and 67th percentiles)
    dividend_33rd = np.percentile(unique_dividends, 33)
    dividend_67th = np.percentile(unique_dividends, 67)
    
    print(f"Dividend classification thresholds:")
    print(f"  Low: ≤ {dividend_33rd}")
    print(f"  Medium: {dividend_33rd} < x ≤ {dividend_67th}")
    print(f"  High: > {dividend_67th}")
    
    def classify_dividend(dividend):
        if dividend <= dividend_33rd:
            return 'Low'
        elif dividend <= dividend_67th:
            return 'Medium'
        else:
            return 'High'
    
    df['DividendLevel'] = df['Dividend'].apply(classify_dividend)
    return df

def calculate_period_payoffs_by_dividend(df):
    """Calculate payoffs for each period grouped by dividend level"""
    period_payoffs = []
    
    for session in df['Session'].unique():
        session_data = df[df['Session'] == session]
        for period in session_data['Period'].unique():
            period_data = session_data[session_data['Period'] == period]
            
            # Get dividend level for this period
            dividend_level = period_data['DividendLevel'].iloc[0]
            dividend_value = period_data['Dividend'].iloc[0]
            
            # Calculate period performance for each subject
            for _, subject in period_data.iterrows():
                # Calculate period payoff: trading profit + dividend earned
                trading_profit = subject['TradingProfit']
                dividend_earned = subject['DivEarn']
                period_payoff = trading_profit + dividend_earned
                
                period_payoffs.append({
                    'Session': session,
                    'Period': period,
                    'SubjectID': subject['SubjectID'],
                    'Experiment': subject['Experiment'],
                    'DividendLevel': dividend_level,
                    'DividendValue': dividend_value,
                    'PeriodPayoff': period_payoff,
                    'TradingProfit': trading_profit,
                    'DividendEarned': dividend_earned
                })
    
    return pd.DataFrame(period_payoffs)

def analyze_dividend_regime_effects(period_payoffs_df):
    """Analyze high vs low dividend regime effects on period payoffs"""
    print("\n" + "="*60)
    print("DIVIDEND REGIME ANALYSIS (N=5) - CORRECTED")
    print("="*60)
    
    # Separate by dividend level
    high_dividend = period_payoffs_df[period_payoffs_df['DividendLevel'] == 'High']
    medium_dividend = period_payoffs_df[period_payoffs_df['DividendLevel'] == 'Medium']
    low_dividend = period_payoffs_df[period_payoffs_df['DividendLevel'] == 'Low']
    
    print(f"\nSample sizes by dividend level:")
    print(f"High dividend: {len(high_dividend)}")
    print(f"Medium dividend: {len(medium_dividend)}")
    print(f"Low dividend: {len(low_dividend)}")
    
    # Descriptive statistics
    print(f"\nPeriod Payoff Statistics by Dividend Level:")
    for level, data in [('High', high_dividend), ('Medium', medium_dividend), ('Low', low_dividend)]:
        print(f"{level} dividend:")
        print(f"  Mean: {data['PeriodPayoff'].mean():.2f}")
        print(f"  Median: {data['PeriodPayoff'].median():.2f}")
        print(f"  Std: {data['PeriodPayoff'].std():.2f}")
        print(f"  Min: {data['PeriodPayoff'].min():.2f}")
        print(f"  Max: {data['PeriodPayoff'].max():.2f}")
        print(f"  Mean Dividend Value: {data['DividendValue'].mean():.2f}")
    
    # One-way ANOVA test
    print(f"\nOne-way ANOVA Test:")
    anova_result = stats.f_oneway(high_dividend['PeriodPayoff'], 
                                 medium_dividend['PeriodPayoff'], 
                                 low_dividend['PeriodPayoff'])
    print(f"  F-statistic: {anova_result.statistic:.4f}")
    print(f"  p-value: {anova_result.pvalue:.4f}")
    
    # Post-hoc pairwise comparisons (Tukey's HSD)
    from scipy.stats import tukey_hsd
    tukey_result = tukey_hsd(high_dividend['PeriodPayoff'], 
                            medium_dividend['PeriodPayoff'], 
                            low_dividend['PeriodPayoff'])
    print(f"\nTukey's HSD post-hoc comparisons:")
    print(f"  High vs Medium: p = {tukey_result.pvalue[0,1]:.4f}")
    print(f"  High vs Low: p = {tukey_result.pvalue[0,2]:.4f}")
    print(f"  Medium vs Low: p = {tukey_result.pvalue[1,2]:.4f}")
    
    # Kruskal-Wallis test (non-parametric)
    kw_result = stats.kruskal(high_dividend['PeriodPayoff'], 
                             medium_dividend['PeriodPayoff'], 
                             low_dividend['PeriodPayoff'])
    print(f"\nKruskal-Wallis test (non-parametric):")
    print(f"  H-statistic: {kw_result.statistic:.4f}")
    print(f"  p-value: {kw_result.pvalue:.4f}")
    
    # Effect size (eta-squared)
    ss_between = sum([len(group) * (group['PeriodPayoff'].mean() - period_payoffs_df['PeriodPayoff'].mean())**2 
                     for group in [high_dividend, medium_dividend, low_dividend]])
    ss_total = ((period_payoffs_df['PeriodPayoff'] - period_payoffs_df['PeriodPayoff'].mean())**2).sum()
    eta_squared = ss_between / ss_total
    
    print(f"\nEffect size (eta-squared): {eta_squared:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if anova_result.pvalue < 0.05:
        print("✓ Significant difference in period payoffs between dividend levels")
        if high_dividend['PeriodPayoff'].mean() > low_dividend['PeriodPayoff'].mean():
            print("  → High dividend periods show higher payoffs (structural luck component)")
        else:
            print("  → Low dividend periods show higher payoffs")
    else:
        print("✗ No significant difference in period payoffs between dividend levels")
    
    if eta_squared < 0.01:
        effect_size = "negligible"
    elif eta_squared < 0.06:
        effect_size = "small"
    elif eta_squared < 0.14:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    print(f"Effect size: {effect_size} (η² = {eta_squared:.3f})")
    
    return high_dividend, medium_dividend, low_dividend, anova_result, eta_squared

def create_visualizations(high_dividend, medium_dividend, low_dividend, period_payoffs_df):
    """Create visualizations for dividend regime analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Dividend Regime Analysis (N=5): Period Payoffs by Dividend Level', 
                 fontsize=16, fontweight='bold')
    
    # 1. Distribution comparison
    ax1 = axes[0, 0]
    ax1.hist(high_dividend['PeriodPayoff'], alpha=0.7, label='High Dividend', bins=20, color='green')
    ax1.hist(medium_dividend['PeriodPayoff'], alpha=0.7, label='Medium Dividend', bins=20, color='orange')
    ax1.hist(low_dividend['PeriodPayoff'], alpha=0.7, label='Low Dividend', bins=20, color='red')
    ax1.set_xlabel('Period Payoff (Francs)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Period Payoff Distribution by Dividend Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = axes[0, 1]
    data_for_box = [high_dividend['PeriodPayoff'], medium_dividend['PeriodPayoff'], low_dividend['PeriodPayoff']]
    labels = ['High', 'Medium', 'Low']
    bp = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('orange')
    bp['boxes'][2].set_facecolor('red')
    ax2.set_ylabel('Period Payoff (Francs)')
    ax2.set_title('Period Payoff Box Plot by Dividend Level')
    ax2.grid(True, alpha=0.3)
    
    # 3. Dividend value vs payoff scatter
    ax3 = axes[1, 0]
    ax3.scatter(period_payoffs_df['DividendValue'], period_payoffs_df['PeriodPayoff'], 
               alpha=0.6, s=20, color='purple')
    ax3.set_xlabel('Dividend Value (Francs)')
    ax3.set_ylabel('Period Payoff (Francs)')
    ax3.set_title('Dividend Value vs Period Payoff')
    ax3.grid(True, alpha=0.3)
    
    # 4. Mean payoffs by dividend level
    ax4 = axes[1, 1]
    mean_payoffs = period_payoffs_df.groupby('DividendLevel')['PeriodPayoff'].agg(['mean', 'std']).reset_index()
    ax4.bar(mean_payoffs['DividendLevel'], mean_payoffs['mean'], 
           yerr=mean_payoffs['std'], capsize=5, 
           color=['red', 'orange', 'green'], alpha=0.7)
    ax4.set_xlabel('Dividend Level')
    ax4.set_ylabel('Mean Period Payoff (Francs)')
    ax4.set_title('Mean Period Payoff by Dividend Level')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dividend_regime_analysis_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'dividend_regime_analysis_corrected.png'")

def main():
    """Main analysis function"""
    print("Loading data for dividend regime analysis...")
    df = load_data()
    
    if df is None:
        return
    
    print("Classifying dividend levels...")
    df = classify_dividend_levels(df)
    
    print("Calculating period payoffs by dividend level...")
    period_payoffs_df = calculate_period_payoffs_by_dividend(df)
    
    print(f"Calculated period payoffs for {len(period_payoffs_df)} period-subject observations")
    
    # Analyze dividend regime effects
    high_dividend, medium_dividend, low_dividend, anova_result, eta_squared = analyze_dividend_regime_effects(period_payoffs_df)
    
    # Create visualizations
    create_visualizations(high_dividend, medium_dividend, low_dividend, period_payoffs_df)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Key finding: {'Significant dividend effect' if anova_result.pvalue < 0.05 else 'No significant dividend effect'} on period payoffs")
    print(f"Effect size: {eta_squared:.3f} ({'negligible' if eta_squared < 0.01 else 'small' if eta_squared < 0.06 else 'medium' if eta_squared < 0.14 else 'large'})")

if __name__ == "__main__":
    main()
