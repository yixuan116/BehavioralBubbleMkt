#!/usr/bin/env python3
"""
Price Distribution Analysis for Bubble Market Experiments
Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    """Load experiment data"""
    df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    
    # Calculate price deviation
    df_a['PriceDeviation'] = df_a['LastPrice'] / df_a['Fundamental']
    df_b['PriceDeviation'] = df_b['LastPrice'] / df_b['Fundamental']
    
    return df_a, df_b

def create_price_distribution_plot(df_a, df_b):
    """Create price distribution analysis plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Experiment A Price Distribution
    axes[0,0].hist(df_a['LastPrice'], bins=60, alpha=0.8, color='blue', 
                   edgecolor='black', linewidth=0.5)
    axes[0,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[0,0].set_ylabel('Frequency', fontsize=12)
    axes[0,0].set_title('Experiment A: Price Distribution\n(Learning Effect)', 
                       fontsize=14, fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Experiment B Price Distribution
    axes[0,1].hist(df_b['LastPrice'], bins=60, alpha=0.8, color='red', 
                   edgecolor='black', linewidth=0.5)
    axes[0,1].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[0,1].set_ylabel('Frequency', fontsize=12)
    axes[0,1].set_title('Experiment B: Price Distribution\n(Professional vs Students)', 
                       fontsize=14, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Experiment A Price Deviation
    axes[1,0].hist(df_a['PriceDeviation'], bins=60, alpha=0.8, color='blue', 
                   edgecolor='black', linewidth=0.5)
    axes[1,0].axvline(x=1, color='red', linestyle='--', linewidth=2, 
                     label='Fundamental Value (1.0)')
    axes[1,0].set_xlabel('Price Deviation (Price/Fundamental)', fontsize=12)
    axes[1,0].set_ylabel('Frequency', fontsize=12)
    axes[1,0].set_title(f'Experiment A: Price Deviation\n(Average: {df_a["PriceDeviation"].mean():.2f}x)', 
                       fontsize=14, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Experiment B Price Deviation
    axes[1,1].hist(df_b['PriceDeviation'], bins=60, alpha=0.8, color='red', 
                   edgecolor='black', linewidth=0.5)
    axes[1,1].axvline(x=1, color='blue', linestyle='--', linewidth=2, 
                     label='Fundamental Value (1.0)')
    axes[1,1].set_xlabel('Price Deviation (Price/Fundamental)', fontsize=12)
    axes[1,1].set_ylabel('Frequency', fontsize=12)
    axes[1,1].set_title(f'Experiment B: Price Deviation\n(Average: {df_b["PriceDeviation"].mean():.2f}x)', 
                       fontsize=14, fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('price_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Price distribution plot saved as 'price_distribution_analysis.png'")

def create_session_trends_plot(df_a, df_b):
    """Create session trends analysis plot"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Experiment A Session Trends
    session_stats_a = df_a.groupby('Session')['LastPrice'].agg(['mean', 'std', 'min', 'max'])
    session_means_a = session_stats_a['mean']
    session_stds_a = session_stats_a['std']
    
    axes[0].errorbar(session_means_a.index, session_means_a.values, 
                    yerr=session_stds_a.values, fmt='o-', color='blue', 
                    linewidth=3, markersize=8, capsize=5, capthick=2)
    axes[0].set_xlabel('Session Number', fontsize=12)
    axes[0].set_ylabel('Average Price (Francs)', fontsize=12)
    axes[0].set_title('Experiment A: Price Trends by Session\n(Learning Effect - 5 Markets per Session)', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(200, 300)
    
    # Experiment B Session Trends
    session_stats_b = df_b.groupby('Session')['LastPrice'].agg(['mean', 'std', 'min', 'max'])
    session_means_b = session_stats_b['mean']
    session_stds_b = session_stats_b['std']
    
    axes[1].errorbar(session_means_b.index, session_means_b.values, 
                    yerr=session_stds_b.values, fmt='o-', color='red', 
                    linewidth=3, markersize=8, capsize=5, capthick=2)
    axes[1].set_xlabel('Session Number', fontsize=12)
    axes[1].set_ylabel('Average Price (Francs)', fontsize=12)
    axes[1].set_title('Experiment B: Price Trends by Session\n(Professional Trader Influence)', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('session_trends_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Session trends plot saved as 'session_trends_analysis.png'")

def print_summary_statistics(df_a, df_b):
    """Print summary statistics"""
    print("="*60)
    print("BUBBLE MARKET EXPERIMENT - PRICE ANALYSIS SUMMARY")
    print("="*60)
    
    print("\nEXPERIMENT A (Learning Effect):")
    print(f"  Data points: {len(df_a):,}")
    print(f"  Price range: {df_a['LastPrice'].min():.2f} - {df_a['LastPrice'].max():.2f}")
    print(f"  Average price: {df_a['LastPrice'].mean():.2f}")
    print(f"  Average deviation: {df_a['PriceDeviation'].mean():.3f}x")
    print(f"  Maximum deviation: {df_a['PriceDeviation'].max():.3f}x")
    
    print("\nEXPERIMENT B (Professional vs Students):")
    print(f"  Data points: {len(df_b):,}")
    print(f"  Price range: {df_b['LastPrice'].min():.2f} - {df_b['LastPrice'].max():.2f}")
    print(f"  Average price: {df_b['LastPrice'].mean():.2f}")
    print(f"  Average deviation: {df_b['PriceDeviation'].mean():.3f}x")
    print(f"  Maximum deviation: {df_b['PriceDeviation'].max():.3f}x")
    
    print("\nBUBBLE EVIDENCE:")
    print(f"  Both experiments show significant overpricing (>1.0x fundamental)")
    print(f"  Experiment A: {df_a['PriceDeviation'].mean():.1%} above fundamental")
    print(f"  Experiment B: {df_b['PriceDeviation'].mean():.1%} above fundamental")

def main():
    """Main analysis function"""
    print("Loading data...")
    df_a, df_b = load_data()
    
    print("Generating summary statistics...")
    print_summary_statistics(df_a, df_b)
    
    print("\nCreating price distribution plot...")
    create_price_distribution_plot(df_a, df_b)
    
    print("\nCreating session trends plot...")
    create_session_trends_plot(df_a, df_b)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
