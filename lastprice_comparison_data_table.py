#!/usr/bin/env python3
"""
Last Price Comparison Data Table Generator
为Last Price Comparison分析生成数据表格

Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics
"""

import pandas as pd
import numpy as np
from tabulate import tabulate
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
        
        return df_a, df_b
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def generate_lastprice_data_table():
    """Generate data table for Last Price Comparison"""
    print("Generating Last Price Comparison Data Table...")
    
    df_a, df_b = load_data()
    if df_a is None or df_b is None:
        return
    
    # Clean data
    price_a = df_a['LastPrice'].dropna()
    price_b = df_b['LastPrice'].dropna()
    
    # Calculate statistics
    stats_data = [
        ['Choice A', len(price_a), f"{price_a.mean():.2f}", f"{price_a.median():.2f}", 
         f"{price_a.std():.2f}", f"{price_a.min():.2f}", f"{price_a.max():.2f}"],
        ['Choice B', len(price_b), f"{price_b.mean():.2f}", f"{price_b.median():.2f}", 
         f"{price_b.std():.2f}", f"{price_b.min():.2f}", f"{price_b.max():.2f}"]
    ]
    
    headers = ['Experiment', 'Sample Size', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
    
    print("\n" + "="*80)
    print("LAST PRICE COMPARISON DATA TABLE")
    print("="*80)
    print(tabulate(stats_data, headers=headers, tablefmt='grid', stralign='center'))
    
    # Statistical tests
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(price_a, price_b)
    u_stat, p_val_u = stats.mannwhitneyu(price_a, price_b, alternative='two-sided')
    
    pooled_std = np.sqrt(((len(price_a)-1)*price_a.var() + (len(price_b)-1)*price_b.var()) / 
                        (len(price_a) + len(price_b) - 2))
    cohens_d = (price_a.mean() - price_b.mean()) / pooled_std
    
    test_data = [
        ['Two-sample t-test', f"t = {t_stat:.3f}", f"p = {p_val:.6f}"],
        ['Mann-Whitney U test', f"U = {u_stat:.3f}", f"p = {p_val_u:.6f}"],
        ["Effect size (Cohen's d)", f"d = {cohens_d:.3f}", ""]
    ]
    
    test_headers = ['Test', 'Statistic', 'p-value']
    
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    print(tabulate(test_data, headers=test_headers, tablefmt='grid', stralign='center'))
    
    # Price range analysis
    low_prices_a = len(price_a[price_a <= 30])
    low_prices_b = len(price_b[price_b <= 30])
    
    range_data = [
        ['Low prices (≤30 Francs)', low_prices_a, low_prices_b, f"{low_prices_b/low_prices_a:.1f}x more in Choice B"],
        ['High prices (>300 Francs)', len(price_a[price_a > 300]), len(price_b[price_b > 300]), ""]
    ]
    
    range_headers = ['Price Range', 'Choice A', 'Choice B', 'Note']
    
    print("\n" + "="*60)
    print("PRICE RANGE ANALYSIS")
    print("="*60)
    print(tabulate(range_data, headers=range_headers, tablefmt='grid', stralign='center'))
    
    # Save to file
    with open('lastprice_comparison_data_table.txt', 'w') as f:
        f.write("LAST PRICE COMPARISON DATA TABLE\n")
        f.write("="*80 + "\n")
        f.write(tabulate(stats_data, headers=headers, tablefmt='grid', stralign='center'))
        f.write("\n\nSTATISTICAL TESTS\n")
        f.write("="*60 + "\n")
        f.write(tabulate(test_data, headers=test_headers, tablefmt='grid', stralign='center'))
        f.write("\n\nPRICE RANGE ANALYSIS\n")
        f.write("="*60 + "\n")
        f.write(tabulate(range_data, headers=range_headers, tablefmt='grid', stralign='center'))
    
    print(f"\n✓ Data table saved to 'lastprice_comparison_data_table.txt'")

if __name__ == "__main__":
    generate_lastprice_data_table()
