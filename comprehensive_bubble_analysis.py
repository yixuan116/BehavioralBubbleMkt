#!/usr/bin/env python3
"""
Comprehensive Bubble Analysis: Choice A vs Choice B
Focus on LastPrice vs Fundamental Value analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load Choice A and Choice B data"""
    print("Loading data...")
    df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    
    # Add experiment labels
    df_a['Experiment'] = 'Choice A'
    df_b['Experiment'] = 'Choice B'
    
    print(f"✓ Choice A: {len(df_a)} records")
    print(f"✓ Choice B: {len(df_b)} records")
    return df_a, df_b

def calculate_bubble_metrics(df, experiment_name):
    """Calculate comprehensive bubble metrics"""
    print(f"\nCalculating bubble metrics for {experiment_name}...")
    
    # Period-level aggregation
    period_metrics = []
    
    if experiment_name == 'Choice A':
        group_cols = ['Session', 'Market', 'Period']
    else:  # Choice B
        group_cols = ['Session', 'Period']
    
    for group, data in df.groupby(group_cols):
        if experiment_name == 'Choice A':
            session, market, period = group
        else:
            session, period = group
            market = 1  # Choice B has only 1 market per session
        
        # Calculate metrics
        avg_price = data['LastPrice'].mean()
        fundamental = data['Fundamental'].iloc[0]
        bubble_pct = ((avg_price - fundamental) / fundamental) * 100 if fundamental > 0 else 0
        
        # Trading activity
        total_units = data['UnitsBuy'].sum() + data['UnitsSell'].sum()
        participants = len(data[data['UnitsBuy'] + data['UnitsSell'] > 0])
        participation_rate = participants / len(data) * 100
        
        period_metrics.append({
            'Session': session,
            'Market': market,
            'Period': period,
            'AvgPrice': avg_price,
            'Fundamental': fundamental,
            'BubblePct': bubble_pct,
            'TotalUnits': total_units,
            'Participants': participants,
            'ParticipationRate': participation_rate
        })
    
    return pd.DataFrame(period_metrics)

def analyze_market_reset_jumps(period_df, experiment_name):
    """Analyze price jumps due to market resets"""
    print(f"\nAnalyzing market reset jumps for {experiment_name}...")
    
    jumps = []
    
    if experiment_name == 'Choice A':
        # For Choice A: analyze jumps between markets within same session
        for session in period_df['Session'].unique():
            session_data = period_df[period_df['Session'] == session].sort_values(['Market', 'Period'])
            
            for market in range(1, 5):  # Markets 1-4 to 2-5
                market_end = session_data[(session_data['Market'] == market) & 
                                        (session_data['Period'] == 15)]
                market_start = session_data[(session_data['Market'] == market + 1) & 
                                          (session_data['Period'] == 1)]
                
                if len(market_end) > 0 and len(market_start) > 0:
                    end_price = market_end['AvgPrice'].iloc[0]
                    start_price = market_start['AvgPrice'].iloc[0]
                    start_fundamental = market_start['Fundamental'].iloc[0]
                    
                    jump_pct = ((start_price - end_price) / start_fundamental) * 100
                    
                    jumps.append({
                        'Session': session,
                        'FromMarket': market,
                        'ToMarket': market + 1,
                        'EndPrice': end_price,
                        'StartPrice': start_price,
                        'JumpPct': jump_pct
                    })
    
    return pd.DataFrame(jumps)

def analyze_learning_effects(period_df, experiment_name):
    """Analyze learning effects within markets and sessions"""
    print(f"\nAnalyzing learning effects for {experiment_name}...")
    
    learning_metrics = []
    
    if experiment_name == 'Choice A':
        # Market-level learning (periods 1-15 within each market)
        for session in period_df['Session'].unique():
            for market in period_df['Market'].unique():
                market_data = period_df[(period_df['Session'] == session) & 
                                      (period_df['Market'] == market)].sort_values('Period')
                
                if len(market_data) >= 10:  # Need enough periods for trend analysis
                    periods = market_data['Period'].values
                    bubble_pcts = market_data['BubblePct'].values
                    
                    # Linear regression for trend
                    slope, intercept, r_value, p_value, std_err = stats.linregress(periods, bubble_pcts)
                    
                    learning_metrics.append({
                        'Session': session,
                        'Market': market,
                        'Type': 'Market_Learning',
                        'Slope': slope,
                        'R_squared': r_value**2,
                        'P_value': p_value,
                        'MeanBubble': bubble_pcts.mean(),
                        'Volatility': np.std(bubble_pcts)
                    })
        
        # Session-level learning (markets 1-5 within each session)
        for session in period_df['Session'].unique():
            session_data = period_df[period_df['Session'] == session]
            market_means = session_data.groupby('Market')['BubblePct'].mean().reset_index()
            
            if len(market_means) >= 3:
                markets = market_means['Market'].values
                mean_bubbles = market_means['BubblePct'].values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(markets, mean_bubbles)
                
                learning_metrics.append({
                    'Session': session,
                    'Market': None,
                    'Type': 'Session_Learning',
                    'Slope': slope,
                    'R_squared': r_value**2,
                    'P_value': p_value,
                    'MeanBubble': mean_bubbles.mean(),
                    'Volatility': np.std(mean_bubbles)
                })
    
    return pd.DataFrame(learning_metrics)

def calculate_overall_efficiency(period_df, experiment_name):
    """Calculate overall efficiency metrics"""
    print(f"\nCalculating overall efficiency for {experiment_name}...")
    
    bubble_pcts = period_df['BubblePct'].values
    prices = period_df['AvgPrice'].values
    fundamentals = period_df['Fundamental'].values
    
    efficiency_metrics = {
        'Experiment': experiment_name,
        'MeanBubblePct': np.mean(bubble_pcts),
        'MedianBubblePct': np.median(bubble_pcts),
        'MeanAbsBubblePct': np.mean(np.abs(bubble_pcts)),
        'MaxBubblePct': np.max(bubble_pcts),
        'MinBubblePct': np.min(bubble_pcts),
        'BubbleVolatility': np.std(bubble_pcts),
        'ShareAboveFundamental': np.mean(bubble_pcts > 0) * 100,
        'RMSE_to_Fundamental': np.sqrt(np.mean((prices - fundamentals)**2)) / np.mean(fundamentals) * 100,
        'TotalPeriods': len(period_df)
    }
    
    return efficiency_metrics

def create_bubble_time_series_charts(df_a_metrics, df_b_metrics):
    """Create comprehensive bubble time series charts"""
    print("\nCreating bubble time series charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Period-level bubble trends (Choice A: Market 1 only for clarity)
    ax1 = axes[0, 0]
    market1_a = df_a_metrics[df_a_metrics['Market'] == 1].groupby('Period')['BubblePct'].mean()
    periods_b = df_b_metrics.groupby('Period')['BubblePct'].mean()
    
    ax1.plot(market1_a.index, market1_a.values, 'ro-', linewidth=2, markersize=6, label='Choice A (Market 1)')
    ax1.plot(periods_b.index, periods_b.values, 'bs-', linewidth=2, markersize=6, label='Choice B')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Period', fontsize=12)
    ax1.set_ylabel('Bubble %', fontsize=12)
    ax1.set_title('Bubble % Over Periods', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Session-level average bubbles
    ax2 = axes[0, 1]
    session_a = df_a_metrics.groupby('Session')['BubblePct'].mean()
    session_b = df_b_metrics.groupby('Session')['BubblePct'].mean()
    
    ax2.bar(session_a.index - 0.2, session_a.values, 0.4, label='Choice A', alpha=0.7, color='red')
    ax2.bar(session_b.index + 0.2, session_b.values, 0.4, label='Choice B', alpha=0.7, color='blue')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Session', fontsize=12)
    ax2.set_ylabel('Average Bubble %', fontsize=12)
    ax2.set_title('Average Bubble % by Session', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Bubble distribution comparison
    ax3 = axes[1, 0]
    ax3.hist(df_a_metrics['BubblePct'], bins=30, alpha=0.7, label='Choice A', color='red', density=True)
    ax3.hist(df_b_metrics['BubblePct'], bins=30, alpha=0.7, label='Choice B', color='blue', density=True)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Bubble %', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Bubble % Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Participation rate vs Bubble %
    ax4 = axes[1, 1]
    ax4.scatter(df_a_metrics['ParticipationRate'], df_a_metrics['BubblePct'], 
               alpha=0.6, color='red', label='Choice A', s=50)
    ax4.scatter(df_b_metrics['ParticipationRate'], df_b_metrics['BubblePct'], 
               alpha=0.6, color='blue', label='Choice B', s=50)
    ax4.set_xlabel('Participation Rate %', fontsize=12)
    ax4.set_ylabel('Bubble %', fontsize=12)
    ax4.set_title('Participation Rate vs Bubble %', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_bubble_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Comprehensive bubble charts saved as 'comprehensive_bubble_analysis.png'")

def create_market_reset_analysis(jumps_df):
    """Create market reset jump analysis"""
    if len(jumps_df) == 0:
        print("No market reset jumps to analyze (Choice B has only 1 market per session)")
        return
    
    print("\nCreating market reset jump analysis...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Jump distribution
    ax1.hist(jumps_df['JumpPct'], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Jump % (Relative to Fundamental)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Market Reset Jump Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Jump by session
    session_jumps = jumps_df.groupby('Session')['JumpPct'].mean()
    ax2.bar(session_jumps.index, session_jumps.values, alpha=0.7, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Session', fontsize=12)
    ax2.set_ylabel('Average Jump %', fontsize=12)
    ax2.set_title('Average Jump % by Session', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('market_reset_jump_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Market reset jump analysis saved as 'market_reset_jump_analysis.png'")

def print_summary_tables(df_a_metrics, df_b_metrics, efficiency_a, efficiency_b, learning_a, learning_b):
    """Print comprehensive summary tables"""
    print("\n" + "="*80)
    print("COMPREHENSIVE BUBBLE ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall Efficiency Comparison
    print("\n1. OVERALL EFFICIENCY COMPARISON")
    print("-" * 50)
    efficiency_df = pd.DataFrame([efficiency_a, efficiency_b])
    print(efficiency_df.to_string(index=False, float_format='%.2f'))
    
    # Learning Effects Summary
    if len(learning_a) > 0:
        print("\n2. LEARNING EFFECTS (Choice A)")
        print("-" * 50)
        market_learning = learning_a[learning_a['Type'] == 'Market_Learning']
        session_learning = learning_a[learning_a['Type'] == 'Session_Learning']
        
        print("Market-level Learning (Periods 1-15 within each market):")
        print(f"  Average Slope: {market_learning['Slope'].mean():.3f}")
        print(f"  Significant Trends: {len(market_learning[market_learning['P_value'] < 0.05])}/{len(market_learning)}")
        
        print("\nSession-level Learning (Markets 1-5 within each session):")
        print(f"  Average Slope: {session_learning['Slope'].mean():.3f}")
        print(f"  Significant Trends: {len(session_learning[session_learning['P_value'] < 0.05])}/{len(session_learning)}")
    
    # Statistical Tests
    print("\n3. STATISTICAL TESTS (Choice A vs Choice B)")
    print("-" * 50)
    
    # T-test for bubble means
    t_stat, t_p = stats.ttest_ind(df_a_metrics['BubblePct'], df_b_metrics['BubblePct'])
    print(f"Bubble % Mean Difference:")
    print(f"  Choice A: {df_a_metrics['BubblePct'].mean():.2f}%")
    print(f"  Choice B: {df_b_metrics['BubblePct'].mean():.2f}%")
    print(f"  T-test: t={t_stat:.3f}, p={t_p:.3f}")
    
    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(df_a_metrics['BubblePct'], df_b_metrics['BubblePct'])
    print(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_p:.3f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(df_a_metrics)-1)*df_a_metrics['BubblePct'].std()**2 + 
                         (len(df_b_metrics)-1)*df_b_metrics['BubblePct'].std()**2) / 
                        (len(df_a_metrics) + len(df_b_metrics) - 2))
    cohens_d = (df_a_metrics['BubblePct'].mean() - df_b_metrics['BubblePct'].mean()) / pooled_std
    print(f"  Effect Size (Cohen's d): {cohens_d:.3f}")

def main():
    """Main analysis function"""
    print("COMPREHENSIVE BUBBLE ANALYSIS")
    print("="*50)
    
    # Load data
    df_a, df_b = load_data()
    
    # Calculate metrics
    df_a_metrics = calculate_bubble_metrics(df_a, 'Choice A')
    df_b_metrics = calculate_bubble_metrics(df_b, 'Choice B')
    
    # Analyze market reset jumps
    jumps_a = analyze_market_reset_jumps(df_a_metrics, 'Choice A')
    jumps_b = analyze_market_reset_jumps(df_b_metrics, 'Choice B')
    
    # Analyze learning effects
    learning_a = analyze_learning_effects(df_a_metrics, 'Choice A')
    learning_b = analyze_learning_effects(df_b_metrics, 'Choice B')
    
    # Calculate overall efficiency
    efficiency_a = calculate_overall_efficiency(df_a_metrics, 'Choice A')
    efficiency_b = calculate_overall_efficiency(df_b_metrics, 'Choice B')
    
    # Create visualizations
    create_bubble_time_series_charts(df_a_metrics, df_b_metrics)
    create_market_reset_analysis(jumps_a)
    
    # Print summary
    print_summary_tables(df_a_metrics, df_b_metrics, efficiency_a, efficiency_b, learning_a, learning_b)
    
    print("\n✓ Comprehensive bubble analysis completed!")
    
    return df_a_metrics, df_b_metrics, efficiency_a, efficiency_b

if __name__ == "__main__":
    main()
