#!/usr/bin/env python3
"""
Market Efficiency Analysis Data Table Generator
为Market Efficiency Analysis生成数据表格

Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics
"""

import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy import stats
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
            
            bubble_metrics.append({
                'Session': session,
                'Period': period,
                'Experiment': period_data['Experiment'].iloc[0],
                'AvgPrice': avg_price,
                'Fundamental': fundamental,
                'Bubble': bubble,
                'AbsBubble': abs(bubble)
            })
    
    return pd.DataFrame(bubble_metrics)

def generate_market_efficiency_data_table():
    """Generate data table for Market Efficiency Analysis"""
    print("Generating Market Efficiency Analysis Data Table...")
    
    df = load_data()
    if df is None:
        return
    
    bubble_df = calculate_bubble_metrics(df)
    
    # Separate by experiment
    choice_a = bubble_df[bubble_df['Experiment'] == 'Choice A']
    choice_b = bubble_df[bubble_df['Experiment'] == 'Choice B']
    
    # Calculate statistics for each experiment
    stats_data = []
    
    for experiment, data in [('Choice A', choice_a), ('Choice B', choice_b)]:
        bubble_mean = data['Bubble'].mean()
        bubble_std = data['Bubble'].std()
        bubble_min = data['Bubble'].min()
        bubble_max = data['Bubble'].max()
        
        # One-sample t-test against 0
        t_stat, p_val = stats.ttest_1samp(data['Bubble'], 0)
        
        # Effect size (Cohen's d)
        cohens_d = bubble_mean / bubble_std if bubble_std > 0 else 0
        
        stats_data.append([
            experiment,
            len(data),
            f"{bubble_mean:.3f}",
            f"{bubble_std:.3f}",
            f"{bubble_min:.3f}",
            f"{bubble_max:.3f}",
            f"{bubble_mean*100:.1f}%",
            f"t = {t_stat:.3f}",
            f"p = {p_val:.6f}",
            f"d = {cohens_d:.3f}"
        ])
    
    headers = [
        'Experiment', 'Sample Size', 'Mean Bubble', 'Std Dev', 'Min', 'Max', 
        'Overpricing %', 't-statistic', 'p-value', "Effect Size"
    ]
    
    print("\n" + "="*100)
    print("MARKET EFFICIENCY ANALYSIS DATA TABLE")
    print("="*100)
    print(tabulate(stats_data, headers=headers, tablefmt='grid', stralign='center'))
    
    # Bubble size categories
    print("\n" + "="*80)
    print("BUBBLE SIZE CATEGORIES")
    print("="*80)
    
    categories_data = []
    for experiment, data in [('Choice A', choice_a), ('Choice B', choice_b)]:
        small_bubbles = len(data[(data['Bubble'] >= 0) & (data['Bubble'] < 0.1)])
        medium_bubbles = len(data[(data['Bubble'] >= 0.1) & (data['Bubble'] < 0.3)])
        large_bubbles = len(data[data['Bubble'] >= 0.3])
        negative_bubbles = len(data[data['Bubble'] < 0])
        
        categories_data.append([
            experiment,
            small_bubbles,
            medium_bubbles,
            large_bubbles,
            negative_bubbles,
            f"{len(data)}"
        ])
    
    categories_headers = [
        'Experiment', 'Small (0-10%)', 'Medium (10-30%)', 'Large (>30%)', 
        'Negative (<0%)', 'Total'
    ]
    
    print(tabulate(categories_data, headers=categories_headers, tablefmt='grid', stralign='center'))
    
    # Period-by-period bubble analysis
    print("\n" + "="*120)
    print("PERIOD-BY-PERIOD BUBBLE ANALYSIS")
    print("="*120)
    
    period_data = []
    for period in sorted(bubble_df['Period'].unique()):
        period_bubbles = bubble_df[bubble_df['Period'] == period]
        
        choice_a_period = period_bubbles[period_bubbles['Experiment'] == 'Choice A']
        choice_b_period = period_bubbles[period_bubbles['Experiment'] == 'Choice B']
        
        if len(choice_a_period) > 0 and len(choice_b_period) > 0:
            period_data.append([
                period,
                f"{choice_a_period['Bubble'].iloc[0]:.3f}",
                f"{choice_b_period['Bubble'].iloc[0]:.3f}",
                f"{choice_a_period['AvgPrice'].iloc[0]:.1f}",
                f"{choice_b_period['AvgPrice'].iloc[0]:.1f}",
                f"{choice_a_period['Fundamental'].iloc[0]:.1f}"
            ])
    
    period_headers = [
        'Period', 'Choice A Bubble', 'Choice B Bubble', 
        'Choice A Price', 'Choice B Price', 'Fundamental'
    ]
    
    print(tabulate(period_data, headers=period_headers, tablefmt='grid', stralign='center'))
    
    # Save to file
    with open('market_efficiency_data_table.txt', 'w') as f:
        f.write("MARKET EFFICIENCY ANALYSIS DATA TABLE\n")
        f.write("="*100 + "\n")
        f.write(tabulate(stats_data, headers=headers, tablefmt='grid', stralign='center'))
        f.write("\n\nBUBBLE SIZE CATEGORIES\n")
        f.write("="*80 + "\n")
        f.write(tabulate(categories_data, headers=categories_headers, tablefmt='grid', stralign='center'))
        f.write("\n\nPERIOD-BY-PERIOD BUBBLE ANALYSIS\n")
        f.write("="*120 + "\n")
        f.write(tabulate(period_data, headers=period_headers, tablefmt='grid', stralign='center'))
    
    print(f"\n✓ Data table saved to 'market_efficiency_data_table.txt'")

if __name__ == "__main__":
    generate_market_efficiency_data_table()
