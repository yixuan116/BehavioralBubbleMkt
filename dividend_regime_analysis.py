#!/usr/bin/env python3
"""
Dividend Regime Analysis (N=5)
Individual Layer Analysis: High vs Low Dividend Effects on Payoffs

Hypothesis: H0: μ_highdiv = μ_lowdiv vs H1: μ_highdiv ≠ μ_lowdiv
Expected: μ_highdiv > μ_lowdiv → Structural luck component
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

def calculate_final_payoffs_by_regime(df, session_regimes):
    """Calculate final payoffs grouped by dividend regime"""
    final_payoffs = []
    
    for session in df['Session'].unique():
        session_data = df[df['Session'] == session]
        session_regime = session_regimes[session_regimes['Session'] == session]['Regime'].iloc[0]
        
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
                    'Regime': session_regime,
                    'AvgDividend': session_regimes[session_regimes['Session'] == session]['AvgDividend'].iloc[0],
                    'FinalPayoff': final_payoff,
                    'TotalDividends': total_dividends
                })
    
    return pd.DataFrame(final_payoffs)

def analyze_dividend_regime_effects(payoffs_df):
    """Analyze high vs low dividend regime effects"""
    print("\n" + "="*60)
    print("DIVIDEND REGIME ANALYSIS (N=5)")
    print("="*60)
    
    # Separate high and low dividend regimes
    high_regime = payoffs_df[payoffs_df['Regime'] == 'High']
    low_regime = payoffs_df[payoffs_df['Regime'] == 'Low']
    
    print(f"\nSample sizes:")
    print(f"High dividend regime: {len(high_regime)} subjects")
    print(f"Low dividend regime: {len(low_regime)} subjects")
    
    # Descriptive statistics
    print(f"\nFinal Payoff Statistics by Dividend Regime:")
    print(f"High dividend regime:")
    print(f"  Mean: {high_regime['FinalPayoff'].mean():.2f}")
    print(f"  Median: {high_regime['FinalPayoff'].median():.2f}")
    print(f"  Std: {high_regime['FinalPayoff'].std():.2f}")
    print(f"  Min: {high_regime['FinalPayoff'].min():.2f}")
    print(f"  Max: {high_regime['FinalPayoff'].max():.2f}")
    
    print(f"\nLow dividend regime:")
    print(f"  Mean: {low_regime['FinalPayoff'].mean():.2f}")
    print(f"  Median: {low_regime['FinalPayoff'].median():.2f}")
    print(f"  Std: {low_regime['FinalPayoff'].std():.2f}")
    print(f"  Min: {low_regime['FinalPayoff'].min():.2f}")
    print(f"  Max: {low_regime['FinalPayoff'].max():.2f}")
    
    # Statistical tests
    print(f"\nStatistical Tests:")
    
    # Normality tests
    high_shapiro = stats.shapiro(high_regime['FinalPayoff'])
    low_shapiro = stats.shapiro(low_regime['FinalPayoff'])
    
    print(f"Normality tests (Shapiro-Wilk):")
    print(f"  High dividend regime: W = {high_shapiro.statistic:.4f}, p = {high_shapiro.pvalue:.4f}")
    print(f"  Low dividend regime: W = {low_shapiro.statistic:.4f}, p = {low_shapiro.pvalue:.4f}")
    
    # Two-sample t-test
    ttest_result = stats.ttest_ind(high_regime['FinalPayoff'], low_regime['FinalPayoff'])
    print(f"\nTwo-sample t-test:")
    print(f"  t-statistic: {ttest_result.statistic:.4f}")
    print(f"  p-value: {ttest_result.pvalue:.4f}")
    
    # Mann-Whitney U test (non-parametric)
    mw_result = stats.mannwhitneyu(high_regime['FinalPayoff'], low_regime['FinalPayoff'], 
                                 alternative='two-sided')
    print(f"\nMann-Whitney U test:")
    print(f"  U-statistic: {mw_result.statistic:.4f}")
    print(f"  p-value: {mw_result.pvalue:.4f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(high_regime)-1)*high_regime['FinalPayoff'].var() + 
                         (len(low_regime)-1)*low_regime['FinalPayoff'].var()) / 
                        (len(high_regime) + len(low_regime) - 2))
    cohens_d = (high_regime['FinalPayoff'].mean() - low_regime['FinalPayoff'].mean()) / pooled_std
    print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if ttest_result.pvalue < 0.05:
        if high_regime['FinalPayoff'].mean() > low_regime['FinalPayoff'].mean():
            print("✓ High dividend regime sessions show significantly higher final payoffs")
        else:
            print("✓ Low dividend regime sessions show significantly higher final payoffs")
    else:
        print("✗ No significant difference in final payoffs between dividend regimes")
    
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    print(f"Effect size: {effect_size} (|d| = {abs(cohens_d):.3f})")
    
    return high_regime, low_regime, ttest_result, mw_result, cohens_d

def create_visualizations(high_regime, low_regime, payoffs_df):
    """Create visualizations for dividend regime analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Dividend Regime Analysis (N=5): High vs Low Dividend Effects', 
                 fontsize=16, fontweight='bold')
    
    # 1. Distribution comparison
    ax1 = axes[0, 0]
    ax1.hist(high_regime['FinalPayoff'], alpha=0.7, label='High Dividend', bins=20, color='green')
    ax1.hist(low_regime['FinalPayoff'], alpha=0.7, label='Low Dividend', bins=20, color='red')
    ax1.set_xlabel('Final Payoff (Francs)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Final Payoff Distribution by Dividend Regime')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = axes[0, 1]
    data_for_box = [high_regime['FinalPayoff'], low_regime['FinalPayoff']]
    labels = ['High Dividend', 'Low Dividend']
    bp = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    ax2.set_ylabel('Final Payoff (Francs)')
    ax2.set_title('Final Payoff Box Plot by Dividend Regime')
    ax2.grid(True, alpha=0.3)
    
    # 3. Average dividend vs performance by session
    ax3 = axes[1, 0]
    session_performance = payoffs_df.groupby(['Session', 'AvgDividend'])['FinalPayoff'].agg(['mean', 'std']).reset_index()
    ax3.scatter(session_performance['AvgDividend'], session_performance['mean'], 
               s=100, alpha=0.7, color='purple')
    ax3.errorbar(session_performance['AvgDividend'], session_performance['mean'], 
                yerr=session_performance['std'], fmt='none', color='purple', alpha=0.5)
    ax3.set_xlabel('Average Dividend in Session')
    ax3.set_ylabel('Mean Final Payoff (Francs)')
    ax3.set_title('Session Average Dividend vs Performance')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance by experiment and regime
    ax4 = axes[1, 1]
    experiment_regime_performance = payoffs_df.groupby(['Experiment', 'Regime'])['FinalPayoff'].mean().unstack()
    experiment_regime_performance.plot(kind='bar', ax=ax4, color=['red', 'green'])
    ax4.set_xlabel('Experiment')
    ax4.set_ylabel('Mean Final Payoff (Francs)')
    ax4.set_title('Performance by Experiment and Dividend Regime')
    ax4.legend(['Low Dividend', 'High Dividend'])
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('dividend_regime_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'dividend_regime_analysis.png'")

def main():
    """Main analysis function"""
    print("Loading data for dividend regime analysis...")
    df = load_data()
    
    if df is None:
        return
    
    print("Calculating dividend regimes for each session...")
    session_regimes = calculate_session_dividend_regimes(df)
    print(f"Dividend regime classification:")
    print(session_regimes[['Session', 'Experiment', 'AvgDividend', 'Regime']].to_string(index=False))
    
    print("Calculating final payoffs by dividend regime...")
    payoffs_df = calculate_final_payoffs_by_regime(df, session_regimes)
    
    print(f"Calculated final payoffs for {len(payoffs_df)} subjects")
    
    # Analyze dividend regime effects
    high_regime, low_regime, ttest_result, mw_result, cohens_d = analyze_dividend_regime_effects(payoffs_df)
    
    # Create visualizations
    create_visualizations(high_regime, low_regime, payoffs_df)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Key finding: {'High dividend regime outperforms' if ttest_result.pvalue < 0.05 and high_regime['FinalPayoff'].mean() > low_regime['FinalPayoff'].mean() else 'No significant difference'} in final payoffs")
    print(f"Effect size: {abs(cohens_d):.3f} ({'negligible' if abs(cohens_d) < 0.2 else 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'})")

if __name__ == "__main__":
    main()
