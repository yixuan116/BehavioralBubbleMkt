#!/usr/bin/env python3
"""
Coordination Failure Analysis (N=11)
Behaviors Layer Analysis: Market Coordination Breakdown

Hypothesis: H0: Given fundamentals, bubble variance = 0 vs H1: variance > 0
Expected: Persistent dispersion → Coordination breakdown
Data: Market-level Bubble
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
    """Load both Choice A and B data for coordination failure analysis"""
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

def calculate_market_bubbles(df):
    """Calculate market-level bubble metrics"""
    market_bubbles = []
    
    for session in df['Session'].unique():
        session_data = df[df['Session'] == session]
        for period in session_data['Period'].unique():
            period_data = session_data[session_data['Period'] == period]
            
            # Calculate period-level bubble metrics
            prices = period_data['LastPrice'].values
            fundamental = period_data['Fundamental'].iloc[0]
            
            # Calculate individual bubbles
            individual_bubbles = (prices - fundamental) / fundamental if fundamental > 0 else np.zeros_like(prices)
            
            # Market-level metrics
            mean_bubble = np.mean(individual_bubbles)
            bubble_variance = np.var(individual_bubbles)
            bubble_std = np.std(individual_bubbles)
            
            # Price dispersion metrics
            price_variance = np.var(prices)
            price_std = np.std(prices)
            price_range = np.max(prices) - np.min(prices)
            price_cv = price_std / np.mean(prices) if np.mean(prices) > 0 else 0
            
            # Coordination metrics
            coordination_ratio = bubble_variance / (fundamental**2) if fundamental > 0 else 0
            
            market_bubbles.append({
                'Session': session,
                'Period': period,
                'Experiment': period_data['Experiment'].iloc[0],
                'Fundamental': fundamental,
                'MeanBubble': mean_bubble,
                'BubbleVariance': bubble_variance,
                'BubbleStd': bubble_std,
                'PriceVariance': price_variance,
                'PriceStd': price_std,
                'PriceRange': price_range,
                'PriceCV': price_cv,
                'CoordinationRatio': coordination_ratio,
                'NumTraders': len(prices)
            })
    
    return pd.DataFrame(market_bubbles)

def analyze_coordination_failure(market_df):
    """Analyze coordination failure in market bubbles"""
    print("\n" + "="*60)
    print("COORDINATION FAILURE ANALYSIS (N=11)")
    print("="*60)
    
    # Separate by experiment
    choice_a = market_df[market_df['Experiment'] == 'Choice A']
    choice_b = market_df[market_df['Experiment'] == 'Choice B']
    
    print(f"\nSample sizes:")
    print(f"Choice A markets: {len(choice_a)}")
    print(f"Choice B markets: {len(choice_b)}")
    
    # Descriptive statistics
    print(f"\nMarket-level Bubble Statistics:")
    print(f"Choice A:")
    print(f"  Mean bubble: {choice_a['MeanBubble'].mean():.4f}")
    print(f"  Bubble variance: {choice_a['BubbleVariance'].mean():.4f}")
    print(f"  Bubble std: {choice_a['BubbleStd'].mean():.4f}")
    print(f"  Price CV: {choice_a['PriceCV'].mean():.4f}")
    print(f"  Coordination ratio: {choice_a['CoordinationRatio'].mean():.4f}")
    
    print(f"\nChoice B:")
    print(f"  Mean bubble: {choice_b['MeanBubble'].mean():.4f}")
    print(f"  Bubble variance: {choice_b['BubbleVariance'].mean():.4f}")
    print(f"  Bubble std: {choice_b['BubbleStd'].mean():.4f}")
    print(f"  Price CV: {choice_b['PriceCV'].mean():.4f}")
    print(f"  Coordination ratio: {choice_b['CoordinationRatio'].mean():.4f}")
    
    # Test for coordination failure
    print(f"\nCoordination Failure Tests:")
    
    # Test 1: Is bubble variance significantly greater than 0?
    # H0: bubble variance = 0 vs H1: bubble variance > 0
    overall_bubble_var = market_df['BubbleVariance'].mean()
    bubble_var_std = market_df['BubbleVariance'].std()
    n_markets = len(market_df)
    
    # One-sample t-test
    t_stat_var = (overall_bubble_var - 0) / (bubble_var_std / np.sqrt(n_markets))
    p_value_var = 1 - stats.t.cdf(t_stat_var, n_markets - 1)
    
    print(f"Test 1: Bubble variance > 0")
    print(f"  Mean bubble variance: {overall_bubble_var:.4f}")
    print(f"  t-statistic: {t_stat_var:.4f}")
    print(f"  p-value: {p_value_var:.4f}")
    
    # Test 2: Is price dispersion significantly greater than expected?
    # Expected dispersion under perfect coordination = 0
    overall_price_cv = market_df['PriceCV'].mean()
    price_cv_std = market_df['PriceCV'].std()
    
    t_stat_cv = (overall_price_cv - 0) / (price_cv_std / np.sqrt(n_markets))
    p_value_cv = 1 - stats.t.cdf(t_stat_cv, n_markets - 1)
    
    print(f"\nTest 2: Price dispersion > 0")
    print(f"  Mean price CV: {overall_price_cv:.4f}")
    print(f"  t-statistic: {t_stat_cv:.4f}")
    print(f"  p-value: {p_value_cv:.4f}")
    
    # Test 3: Comparison between experiments
    # H0: Var_A = Var_B vs H1: Var_A ≠ Var_B
    levene_result = stats.levene(choice_a['BubbleVariance'], choice_b['BubbleVariance'])
    
    print(f"\nTest 3: Coordination difference between experiments")
    print(f"  Levene's test:")
    print(f"    W-statistic: {levene_result.statistic:.4f}")
    print(f"    p-value: {levene_result.pvalue:.4f}")
    
    # Test 4: Chi-square test for coordination failure
    # Expected: If perfectly coordinated, all prices should be equal to fundamental
    # Observed: Actual price dispersion
    
    # Calculate expected vs observed coordination
    expected_coordination = 0  # Perfect coordination = 0 variance
    observed_coordination = market_df['BubbleVariance'].mean()
    
    # Chi-square test (simplified)
    chi_square_stat = n_markets * (observed_coordination - expected_coordination)**2 / (bubble_var_std**2)
    p_value_chi = 1 - stats.chi2.cdf(chi_square_stat, n_markets - 1)
    
    print(f"\nTest 4: Chi-square test for coordination failure")
    print(f"  Expected coordination (perfect): {expected_coordination:.4f}")
    print(f"  Observed coordination: {observed_coordination:.4f}")
    print(f"  Chi-square statistic: {chi_square_stat:.4f}")
    print(f"  p-value: {p_value_chi:.4f}")
    
    # Session-level analysis
    print(f"\nSession-level Coordination Analysis:")
    session_coordination = []
    
    for session in market_df['Session'].unique():
        session_data = market_df[market_df['Session'] == session]
        
        mean_bubble_var = session_data['BubbleVariance'].mean()
        mean_price_cv = session_data['PriceCV'].mean()
        mean_coordination = session_data['CoordinationRatio'].mean()
        
        session_coordination.append({
            'Session': session,
            'Experiment': session_data['Experiment'].iloc[0],
            'MeanBubbleVariance': mean_bubble_var,
            'MeanPriceCV': mean_price_cv,
            'MeanCoordination': mean_coordination,
            'NumPeriods': len(session_data)
        })
    
    session_df = pd.DataFrame(session_coordination)
    print(session_df.to_string(index=False))
    
    # Interpretation
    print(f"\nInterpretation:")
    if p_value_var < 0.05:
        print("✓ Significant coordination failure detected")
        print("  → Bubble variance is significantly greater than 0")
    else:
        print("✗ No significant coordination failure detected")
        print("  → Bubble variance is not significantly different from 0")
    
    if p_value_cv < 0.05:
        print("✓ Significant price dispersion detected")
        print("  → Price coefficient of variation is significantly greater than 0")
    else:
        print("✗ No significant price dispersion detected")
        print("  → Prices show good coordination")
    
    if levene_result.pvalue < 0.05:
        print("✓ Significant difference in coordination between experiments")
        if choice_a['BubbleVariance'].mean() > choice_b['BubbleVariance'].mean():
            print("  → Choice A shows higher coordination failure")
        else:
            print("  → Choice B shows higher coordination failure")
    else:
        print("✗ No significant difference in coordination between experiments")
    
    return market_df, session_df, p_value_var, p_value_cv, levene_result

def create_visualizations(market_df, session_df):
    """Create visualizations for coordination failure analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Coordination Failure Analysis (N=11): Market Coordination Breakdown', 
                 fontsize=16, fontweight='bold')
    
    # 1. Bubble variance over time
    ax1 = axes[0, 0]
    choice_a = market_df[market_df['Experiment'] == 'Choice A']
    choice_b = market_df[market_df['Experiment'] == 'Choice B']
    
    ax1.scatter(choice_a['Period'], choice_a['BubbleVariance'], alpha=0.6, s=30, color='red', label='Choice A')
    ax1.scatter(choice_b['Period'], choice_b['BubbleVariance'], alpha=0.6, s=30, color='blue', label='Choice B')
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Bubble Variance')
    ax1.set_title('Bubble Variance Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Price coefficient of variation
    ax2 = axes[0, 1]
    ax2.scatter(choice_a['Period'], choice_a['PriceCV'], alpha=0.6, s=30, color='red', label='Choice A')
    ax2.scatter(choice_b['Period'], choice_b['PriceCV'], alpha=0.6, s=30, color='blue', label='Choice B')
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Price Coefficient of Variation')
    ax2.set_title('Price Dispersion Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Coordination ratio by session
    ax3 = axes[1, 0]
    colors = ['red' if row['Experiment'] == 'Choice A' else 'blue' for _, row in session_df.iterrows()]
    ax3.bar(range(len(session_df)), session_df['MeanCoordination'], color=colors, alpha=0.7)
    ax3.set_xlabel('Session')
    ax3.set_ylabel('Mean Coordination Ratio')
    ax3.set_title('Coordination Ratio by Session')
    ax3.set_xticks(range(len(session_df)))
    ax3.set_xticklabels(session_df['Session'])
    ax3.grid(True, alpha=0.3)
    
    # 4. Bubble variance distribution
    ax4 = axes[1, 1]
    ax4.hist(choice_a['BubbleVariance'], alpha=0.7, label='Choice A', bins=15, color='red')
    ax4.hist(choice_b['BubbleVariance'], alpha=0.7, label='Choice B', bins=15, color='blue')
    ax4.set_xlabel('Bubble Variance')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Bubble Variance Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('coordination_failure_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'coordination_failure_analysis.png'")

def main():
    """Main analysis function"""
    print("Loading data for coordination failure analysis...")
    df = load_data()
    
    if df is None:
        return
    
    print("Calculating market-level bubble metrics...")
    market_df = calculate_market_bubbles(df)
    
    print(f"Calculated metrics for {len(market_df)} market periods")
    
    # Analyze coordination failure
    market_df, session_df, p_value_var, p_value_cv, levene_result = analyze_coordination_failure(market_df)
    
    # Create visualizations
    create_visualizations(market_df, session_df)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Key finding: {'Significant coordination failure' if p_value_var < 0.05 else 'No significant coordination failure'}")
    print(f"Price dispersion: {'Significant' if p_value_cv < 0.05 else 'Not significant'}")
    print(f"Experiment difference: {'Significant' if levene_result.pvalue < 0.05 else 'Not significant'}")

if __name__ == "__main__":
    main()
