#!/usr/bin/env python3
"""
Bubble-Profit Link Analysis (N=7)
Outcomes Layer Analysis: Correlation between Bubble Size and Final Payoffs

Hypothesis: H0: ρ = 0 vs H1: ρ ≠ 0
Expected: ρ < 0 → Negative relationship (bubbles reduce profits)
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
    """Load both Choice A and B data for bubble-profit analysis"""
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

def calculate_bubble_metrics(df):
    """Calculate bubble metrics for each period"""
    bubble_metrics = []
    
    for session in df['Session'].unique():
        session_data = df[df['Session'] == session]
        for period in session_data['Period'].unique():
            period_data = session_data[session_data['Period'] == period]
            
            # Calculate period-level bubble metrics
            avg_price = period_data['LastPrice'].mean()
            fundamental = period_data['Fundamental'].iloc[0]
            
            # Calculate bubble as (P - F) / F
            bubble = (avg_price - fundamental) / fundamental if fundamental > 0 else 0
            
            # Calculate bubble variance (dispersion)
            bubble_variance = period_data['LastPrice'].var()
            
            # Calculate absolute bubble
            abs_bubble = abs(bubble)
            
            bubble_metrics.append({
                'Session': session,
                'Period': period,
                'Experiment': period_data['Experiment'].iloc[0],
                'AvgPrice': avg_price,
                'Fundamental': fundamental,
                'Bubble': bubble,
                'AbsBubble': abs_bubble,
                'BubbleVariance': bubble_variance,
                'PriceStd': period_data['LastPrice'].std()
            })
    
    return pd.DataFrame(bubble_metrics)

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

def calculate_session_bubble_profit_correlation(bubble_df, payoffs_df):
    """Calculate correlation between session-level bubble and average payoffs"""
    # Aggregate to session level
    session_bubbles = bubble_df.groupby(['Session', 'Experiment']).agg({
        'Bubble': 'mean',
        'AbsBubble': 'mean',
        'BubbleVariance': 'mean'
    }).reset_index()
    
    session_payoffs = payoffs_df.groupby(['Session', 'Experiment']).agg({
        'FinalPayoff': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    session_payoffs.columns = ['Session', 'Experiment', 'MeanPayoff', 'PayoffStd']
    
    # Merge datasets
    session_analysis = pd.merge(session_bubbles, session_payoffs, on=['Session', 'Experiment'])
    
    return session_analysis

def analyze_bubble_profit_correlation(session_analysis):
    """Analyze correlation between bubble size and final payoffs"""
    print("\n" + "="*60)
    print("BUBBLE-PROFIT LINK ANALYSIS (N=7)")
    print("="*60)
    
    # Separate by experiment
    choice_a = session_analysis[session_analysis['Experiment'] == 'Choice A']
    choice_b = session_analysis[session_analysis['Experiment'] == 'Choice B']
    
    print(f"\nSample sizes:")
    print(f"Choice A sessions: {len(choice_a)}")
    print(f"Choice B sessions: {len(choice_b)}")
    
    # Descriptive statistics
    print(f"\nSession-level Statistics:")
    print(f"Choice A:")
    print(f"  Mean bubble: {choice_a['Bubble'].mean():.4f}")
    print(f"  Mean absolute bubble: {choice_a['AbsBubble'].mean():.4f}")
    print(f"  Mean payoff: {choice_a['MeanPayoff'].mean():.2f}")
    
    print(f"\nChoice B:")
    print(f"  Mean bubble: {choice_b['Bubble'].mean():.4f}")
    print(f"  Mean absolute bubble: {choice_b['AbsBubble'].mean():.4f}")
    print(f"  Mean payoff: {choice_b['MeanPayoff'].mean():.2f}")
    
    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    
    # Overall correlation
    overall_pearson = stats.pearsonr(session_analysis['Bubble'], session_analysis['MeanPayoff'])
    overall_spearman = stats.spearmanr(session_analysis['Bubble'], session_analysis['MeanPayoff'])
    
    print(f"Overall (all sessions):")
    print(f"  Pearson correlation: r = {overall_pearson[0]:.4f}, p = {overall_pearson[1]:.4f}")
    print(f"  Spearman correlation: ρ = {overall_spearman[0]:.4f}, p = {overall_spearman[1]:.4f}")
    
    # Choice A correlation
    if len(choice_a) > 2:
        choice_a_pearson = stats.pearsonr(choice_a['Bubble'], choice_a['MeanPayoff'])
        choice_a_spearman = stats.spearmanr(choice_a['Bubble'], choice_a['MeanPayoff'])
        
        print(f"\nChoice A:")
        print(f"  Pearson correlation: r = {choice_a_pearson[0]:.4f}, p = {choice_a_pearson[1]:.4f}")
        print(f"  Spearman correlation: ρ = {choice_a_spearman[0]:.4f}, p = {choice_a_spearman[1]:.4f}")
    
    # Choice B correlation
    if len(choice_b) > 2:
        choice_b_pearson = stats.pearsonr(choice_b['Bubble'], choice_b['MeanPayoff'])
        choice_b_spearman = stats.spearmanr(choice_b['Bubble'], choice_b['MeanPayoff'])
        
        print(f"\nChoice B:")
        print(f"  Pearson correlation: r = {choice_b_pearson[0]:.4f}, p = {choice_b_pearson[1]:.4f}")
        print(f"  Spearman correlation: ρ = {choice_b_spearman[0]:.4f}, p = {choice_b_spearman[1]:.4f}")
    
    # Absolute bubble correlation
    abs_pearson = stats.pearsonr(session_analysis['AbsBubble'], session_analysis['MeanPayoff'])
    abs_spearman = stats.spearmanr(session_analysis['AbsBubble'], session_analysis['MeanPayoff'])
    
    print(f"\nAbsolute Bubble vs Payoff:")
    print(f"  Pearson correlation: r = {abs_pearson[0]:.4f}, p = {abs_pearson[1]:.4f}")
    print(f"  Spearman correlation: ρ = {abs_spearman[0]:.4f}, p = {abs_spearman[1]:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if overall_pearson[1] < 0.05:
        if overall_pearson[0] < 0:
            print("✓ Significant negative correlation between bubble size and payoffs")
            print("  → Larger bubbles are associated with lower payoffs")
        else:
            print("✓ Significant positive correlation between bubble size and payoffs")
            print("  → Larger bubbles are associated with higher payoffs")
    else:
        print("✗ No significant correlation between bubble size and payoffs")
    
    if abs_pearson[1] < 0.05:
        if abs_pearson[0] < 0:
            print("✓ Significant negative correlation between absolute bubble size and payoffs")
            print("  → Higher bubble volatility is associated with lower payoffs")
        else:
            print("✓ Significant positive correlation between absolute bubble size and payoffs")
            print("  → Higher bubble volatility is associated with higher payoffs")
    else:
        print("✗ No significant correlation between absolute bubble size and payoffs")
    
    return overall_pearson, overall_spearman, abs_pearson, abs_spearman

def create_visualizations(session_analysis, choice_a, choice_b):
    """Create visualizations for bubble-profit link analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bubble-Profit Link Analysis (N=7): Bubble Size vs Final Payoffs', 
                 fontsize=16, fontweight='bold')
    
    # 1. Bubble vs Payoff scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(session_analysis['Bubble'], session_analysis['MeanPayoff'], 
               alpha=0.7, s=100, color='blue')
    ax1.set_xlabel('Average Bubble (P-F)/F')
    ax1.set_ylabel('Mean Final Payoff (Francs)')
    ax1.set_title('Bubble Size vs Session Mean Payoff')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(session_analysis['Bubble'], session_analysis['MeanPayoff'], 1)
    p = np.poly1d(z)
    ax1.plot(session_analysis['Bubble'], p(session_analysis['Bubble']), "r--", alpha=0.8)
    
    # 2. Absolute bubble vs payoff
    ax2 = axes[0, 1]
    ax2.scatter(session_analysis['AbsBubble'], session_analysis['MeanPayoff'], 
               alpha=0.7, s=100, color='green')
    ax2.set_xlabel('Average Absolute Bubble |(P-F)/F|')
    ax2.set_ylabel('Mean Final Payoff (Francs)')
    ax2.set_title('Absolute Bubble Size vs Session Mean Payoff')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(session_analysis['AbsBubble'], session_analysis['MeanPayoff'], 1)
    p = np.poly1d(z)
    ax2.plot(session_analysis['AbsBubble'], p(session_analysis['AbsBubble']), "r--", alpha=0.8)
    
    # 3. Bubble by experiment
    ax3 = axes[1, 0]
    ax3.scatter(choice_a['Bubble'], choice_a['MeanPayoff'], 
               alpha=0.7, s=100, color='red', label='Choice A')
    ax3.scatter(choice_b['Bubble'], choice_b['MeanPayoff'], 
               alpha=0.7, s=100, color='blue', label='Choice B')
    ax3.set_xlabel('Average Bubble (P-F)/F')
    ax3.set_ylabel('Mean Final Payoff (Francs)')
    ax3.set_title('Bubble vs Payoff by Experiment')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Bubble variance vs payoff
    ax4 = axes[1, 1]
    ax4.scatter(session_analysis['BubbleVariance'], session_analysis['MeanPayoff'], 
               alpha=0.7, s=100, color='purple')
    ax4.set_xlabel('Bubble Variance (Price Dispersion)')
    ax4.set_ylabel('Mean Final Payoff (Francs)')
    ax4.set_title('Bubble Variance vs Session Mean Payoff')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(session_analysis['BubbleVariance'], session_analysis['MeanPayoff'], 1)
    p = np.poly1d(z)
    ax4.plot(session_analysis['BubbleVariance'], p(session_analysis['BubbleVariance']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('bubble_profit_link_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'bubble_profit_link_analysis.png'")

def main():
    """Main analysis function"""
    print("Loading data for bubble-profit link analysis...")
    df = load_data()
    
    if df is None:
        return
    
    print("Calculating bubble metrics...")
    bubble_df = calculate_bubble_metrics(df)
    
    print("Calculating final payoffs...")
    payoffs_df = calculate_final_payoffs(df)
    
    print("Calculating session-level bubble-profit correlations...")
    session_analysis = calculate_session_bubble_profit_correlation(bubble_df, payoffs_df)
    
    print(f"Analyzing {len(session_analysis)} sessions")
    
    # Separate by experiment for analysis
    choice_a = session_analysis[session_analysis['Experiment'] == 'Choice A']
    choice_b = session_analysis[session_analysis['Experiment'] == 'Choice B']
    
    # Analyze bubble-profit correlation
    overall_pearson, overall_spearman, abs_pearson, abs_spearman = analyze_bubble_profit_correlation(session_analysis)
    
    # Create visualizations
    create_visualizations(session_analysis, choice_a, choice_b)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Key finding: {'Significant correlation' if overall_pearson[1] < 0.05 else 'No significant correlation'} between bubble size and payoffs")
    print(f"Correlation strength: {abs(overall_pearson[0]):.3f} ({'weak' if abs(overall_pearson[0]) < 0.3 else 'moderate' if abs(overall_pearson[0]) < 0.7 else 'strong'})")

if __name__ == "__main__":
    main()
