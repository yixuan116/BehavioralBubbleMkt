#!/usr/bin/env python3
"""
Price Anchoring Analysis
Analyzes price anchoring bias and candidate "price setters" for each period

Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load both Choice A and B data"""
    try:
        df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
        df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
        
        # Add experiment identifier
        df_a['Experiment'] = 'Choice A'
        df_b['Experiment'] = 'Choice B'
        
        # Combine datasets
        df = pd.concat([df_a, df_b], ignore_index=True)
        return df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

def calculate_price_anchoring_metrics(df):
    """Calculate price anchoring metrics for each period"""
    print("Calculating Price Anchoring Metrics...")
    
    period_metrics = []
    
    for experiment in ['Choice A', 'Choice B']:
        exp_data = df[df['Experiment'] == experiment]
        
        for session in sorted(exp_data['Session'].unique()):
            session_data = exp_data[exp_data['Session'] == session]
            
            for market in sorted(session_data['Market'].unique()):
                market_data = session_data[session_data['Market'] == market]
                
                for period in sorted(market_data['Period'].unique()):
                    period_data = market_data[market_data['Period'] == period]
                    
                    # Calculate VWAP (Volume Weighted Average Price)
                    total_buy_volume = period_data['UnitsBuy'].sum()
                    total_sell_volume = period_data['UnitsSell'].sum()
                    
                    if total_buy_volume > 0:
                        vwap_buy = (period_data['UnitsBuy'] * period_data['AvgBuyPrice']).sum() / total_buy_volume
                    else:
                        vwap_buy = 0
                    
                    if total_sell_volume > 0:
                        vwap_sell = (period_data['UnitsSell'] * period_data['AvgSellPrice']).sum() / total_sell_volume
                    else:
                        vwap_sell = 0
                    
                    # Overall VWAP (weighted by total volume)
                    total_volume = total_buy_volume + total_sell_volume
                    if total_volume > 0:
                        vwap_overall = ((period_data['UnitsBuy'] * period_data['AvgBuyPrice']).sum() + 
                                      (period_data['UnitsSell'] * period_data['AvgSellPrice']).sum()) / total_volume
                    else:
                        vwap_overall = 0
                    
                    # LastPrice
                    last_price = period_data['LastPrice'].iloc[0]
                    fundamental = period_data['Fundamental'].iloc[0]
                    
                    # Price anchoring metrics
                    close_drift = last_price - vwap_overall if vwap_overall > 0 else 0
                    close_drift_pct = (close_drift / vwap_overall * 100) if vwap_overall > 0 else 0
                    
                    # Find candidate price setters
                    candidates_sell = period_data[(period_data['AvgSellPrice'] == last_price) & 
                                                 (period_data['UnitsSell'] > 0)]
                    candidates_buy = period_data[(period_data['AvgBuyPrice'] == last_price) & 
                                                (period_data['UnitsBuy'] > 0)]
                    
                    period_metrics.append({
                        'Experiment': experiment,
                        'Session': session,
                        'Market': market,
                        'Period': period,
                        'LastPrice': last_price,
                        'Fundamental': fundamental,
                        'VWAP_Buy': vwap_buy,
                        'VWAP_Sell': vwap_sell,
                        'VWAP_Overall': vwap_overall,
                        'CloseDrift': close_drift,
                        'CloseDrift_Pct': close_drift_pct,
                        'TotalVolume': total_volume,
                        'Candidates_Sell': len(candidates_sell),
                        'Candidates_Buy': len(candidates_buy),
                        'Candidate_SubjectIDs_Sell': list(candidates_sell['SubjectID'].values) if len(candidates_sell) > 0 else [],
                        'Candidate_SubjectIDs_Buy': list(candidates_buy['SubjectID'].values) if len(candidates_buy) > 0 else []
                    })
    
    return pd.DataFrame(period_metrics)

def create_price_anchoring_visualizations(metrics_df):
    """Create visualizations for price anchoring analysis"""
    print("Creating Price Anchoring Visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Price Anchoring Analysis: Close Drift and Candidate Price Setters', 
                 fontsize=16, fontweight='bold')
    
    # 1. Close Drift Distribution
    choice_a_drift = metrics_df[metrics_df['Experiment'] == 'Choice A']['CloseDrift_Pct']
    choice_b_drift = metrics_df[metrics_df['Experiment'] == 'Choice B']['CloseDrift_Pct']
    
    axes[0,0].hist(choice_a_drift, bins=30, alpha=0.7, color='red', 
                   label=f'Choice A (n={len(choice_a_drift)})', density=True)
    axes[0,0].hist(choice_b_drift, bins=30, alpha=0.7, color='blue', 
                   label=f'Choice B (n={len(choice_b_drift)})', density=True)
    axes[0,0].set_xlabel('Close Drift %', fontsize=12)
    axes[0,0].set_ylabel('Density', fontsize=12)
    axes[0,0].set_title('Close Drift Distribution', fontsize=14, fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Close Drift Time Series
    periods = sorted(metrics_df['Period'].unique())
    choice_a_avg_drift = []
    choice_b_avg_drift = []
    
    for period in periods:
        period_data = metrics_df[metrics_df['Period'] == period]
        choice_a_period = period_data[period_data['Experiment'] == 'Choice A']
        choice_b_period = period_data[period_data['Experiment'] == 'Choice B']
        
        choice_a_avg_drift.append(choice_a_period['CloseDrift_Pct'].mean() if len(choice_a_period) > 0 else 0)
        choice_b_avg_drift.append(choice_b_period['CloseDrift_Pct'].mean() if len(choice_b_period) > 0 else 0)
    
    axes[0,1].plot(periods, choice_a_avg_drift, color='red', marker='o', linewidth=2, 
                   markersize=6, label='Choice A')
    axes[0,1].plot(periods, choice_b_avg_drift, color='blue', marker='s', linewidth=2, 
                   markersize=6, label='Choice B')
    axes[0,1].set_xlabel('Period', fontsize=12)
    axes[0,1].set_ylabel('Average Close Drift %', fontsize=12)
    axes[0,1].set_title('Close Drift Time Series', fontsize=14, fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Candidate Price Setters Count
    choice_a_candidates = metrics_df[metrics_df['Experiment'] == 'Choice A']
    choice_b_candidates = metrics_df[metrics_df['Experiment'] == 'Choice B']
    
    choice_a_total_candidates = choice_a_candidates['Candidates_Sell'].sum() + choice_a_candidates['Candidates_Buy'].sum()
    choice_b_total_candidates = choice_b_candidates['Candidates_Sell'].sum() + choice_b_candidates['Candidates_Buy'].sum()
    
    experiments = ['Choice A', 'Choice B']
    candidate_counts = [choice_a_total_candidates, choice_b_total_candidates]
    colors = ['red', 'blue']
    
    bars = axes[1,0].bar(experiments, candidate_counts, color=colors, alpha=0.7)
    axes[1,0].set_ylabel('Total Candidate Price Setters', fontsize=12)
    axes[1,0].set_title('Candidate Price Setters Count', fontsize=14, fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, candidate_counts):
        axes[1,0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(candidate_counts)*0.01,
                       f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Close Drift vs Volume Scatter
    axes[1,1].scatter(choice_a_candidates['TotalVolume'], choice_a_candidates['CloseDrift_Pct'], 
                     color='red', alpha=0.6, s=50, label='Choice A')
    axes[1,1].scatter(choice_b_candidates['TotalVolume'], choice_b_candidates['CloseDrift_Pct'], 
                     color='blue', alpha=0.6, s=50, label='Choice B')
    axes[1,1].set_xlabel('Total Volume', fontsize=12)
    axes[1,1].set_ylabel('Close Drift %', fontsize=12)
    axes[1,1].set_title('Close Drift vs Volume', fontsize=14, fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('price_anchoring_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Price anchoring analysis saved as 'price_anchoring_analysis.png'")

def print_detailed_results(metrics_df):
    """Print detailed results and tables"""
    print("\n" + "="*100)
    print("PRICE ANCHORING ANALYSIS RESULTS")
    print("="*100)
    
    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 50)
    
    for experiment in ['Choice A', 'Choice B']:
        exp_data = metrics_df[metrics_df['Experiment'] == experiment]
        print(f"\n{experiment}:")
        print(f"  Total Periods: {len(exp_data)}")
        print(f"  Mean Close Drift %: {exp_data['CloseDrift_Pct'].mean():.2f}%")
        print(f"  Std Close Drift %: {exp_data['CloseDrift_Pct'].std():.2f}%")
        print(f"  Max Close Drift %: {exp_data['CloseDrift_Pct'].max():.2f}%")
        print(f"  Min Close Drift %: {exp_data['CloseDrift_Pct'].min():.2f}%")
        print(f"  Total Candidate Price Setters: {(exp_data['Candidates_Sell'].sum() + exp_data['Candidates_Buy'].sum())}")
        print(f"  Periods with Candidates: {len(exp_data[(exp_data['Candidates_Sell'] > 0) | (exp_data['Candidates_Buy'] > 0)])}")
    
    # Top periods with largest close drift
    print("\nTOP 10 PERIODS WITH LARGEST CLOSE DRIFT:")
    print("-" * 80)
    top_drift = metrics_df.nlargest(10, 'CloseDrift_Pct')[['Experiment', 'Session', 'Market', 'Period', 
                                                           'LastPrice', 'VWAP_Overall', 'CloseDrift_Pct', 
                                                           'Candidates_Sell', 'Candidates_Buy']]
    
    from tabulate import tabulate
    print(tabulate(top_drift, headers='keys', tablefmt='grid', stralign='center', floatfmt='.2f'))
    
    # Periods with candidate price setters
    print("\nPERIODS WITH CANDIDATE PRICE SETTERS:")
    print("-" * 80)
    candidates_data = metrics_df[(metrics_df['Candidates_Sell'] > 0) | (metrics_df['Candidates_Buy'] > 0)]
    
    if len(candidates_data) > 0:
        candidates_summary = candidates_data[['Experiment', 'Session', 'Market', 'Period', 
                                             'LastPrice', 'CloseDrift_Pct', 
                                             'Candidates_Sell', 'Candidates_Buy']].head(20)
        print(tabulate(candidates_summary, headers='keys', tablefmt='grid', stralign='center', floatfmt='.2f'))
    else:
        print("No periods with candidate price setters found.")
    
    # Detailed candidate analysis for first few periods
    print("\nDETAILED CANDIDATE ANALYSIS (First 5 Periods):")
    print("-" * 80)
    
    for experiment in ['Choice A', 'Choice B']:
        exp_data = metrics_df[metrics_df['Experiment'] == experiment]
        for period in sorted(exp_data['Period'].unique())[:5]:
            period_data = exp_data[exp_data['Period'] == period]
            if len(period_data) > 0:
                period_info = period_data.iloc[0]
                print(f"\n{experiment} - Period {period}:")
                print(f"  LastPrice: {period_info['LastPrice']:.2f}")
                print(f"  VWAP: {period_info['VWAP_Overall']:.2f}")
                print(f"  Close Drift: {period_info['CloseDrift_Pct']:.2f}%")
                print(f"  Sell Candidates: {period_info['Candidate_SubjectIDs_Sell']}")
                print(f"  Buy Candidates: {period_info['Candidate_SubjectIDs_Buy']}")

def main():
    """Main function"""
    print("Loading data for Price Anchoring Analysis...")
    df = load_data()
    
    if df is None:
        return
    
    # Calculate metrics
    metrics_df = calculate_price_anchoring_metrics(df)
    
    # Create visualizations
    create_price_anchoring_visualizations(metrics_df)
    
    # Print detailed results
    print_detailed_results(metrics_df)
    
    # Save detailed data
    metrics_df.to_csv('price_anchoring_metrics.csv', index=False)
    print("\n✓ Detailed metrics saved to 'price_anchoring_metrics.csv'")
    
    print(f"\n" + "="*60)
    print("PRICE ANCHORING ANALYSIS COMPLETE")
    print("="*60)
    print("✓ Analyzed close drift (LastPrice - VWAP)")
    print("✓ Identified candidate price setters")
    print("✓ Created visualizations and detailed tables")

if __name__ == "__main__":
    main()
