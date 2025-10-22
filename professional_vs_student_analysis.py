#!/usr/bin/env python3
"""
Professional vs Student Trader Analysis for Experiment B
Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_experiment_b_data():
    """Load and prepare Experiment B data"""
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    
    # Calculate key metrics
    df_b['PriceDeviation'] = df_b['LastPrice'] / df_b['Fundamental']
    df_b['TradingVolume'] = df_b['UnitsBuy'] + df_b['UnitsSell']
    
    return df_b

def analyze_by_role(df_b):
    """Analyze trading behavior by role (Professional vs Student)"""
    print("="*60)
    print("PROFESSIONAL vs STUDENT TRADER ANALYSIS")
    print("="*60)
    
    # Group by role
    role_stats = df_b.groupby('Role').agg({
        'LastPrice': ['mean', 'std', 'min', 'max'],
        'PriceDeviation': ['mean', 'std', 'min', 'max'],
        'TradingVolume': ['mean', 'std'],
        'TradingProfit': ['mean', 'std'],
        'UnitsBuy': 'mean',
        'UnitsSell': 'mean'
    }).round(3)
    
    print("\nTRADING BEHAVIOR BY ROLE:")
    print(role_stats)
    
    # Professional trader share analysis
    pro_share_stats = df_b.groupby('Session').agg({
        'ProShare': 'first',
        'LastPrice': 'mean',
        'PriceDeviation': 'mean'
    }).round(3)
    
    print("\nPROFESSIONAL TRADER SHARE BY SESSION:")
    print(pro_share_stats)
    
    return role_stats, pro_share_stats

def create_role_comparison_plots(df_b):
    """Create comparison plots for Professional vs Student traders"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Price Distribution by Role
    pro_prices = df_b[df_b['Role'] == 'Pro']['LastPrice']
    student_prices = df_b[df_b['Role'] == 'Student']['LastPrice']
    
    axes[0,0].hist(pro_prices, bins=50, alpha=0.7, color='green', 
                   label='Professional Traders', edgecolor='black')
    axes[0,0].hist(student_prices, bins=50, alpha=0.7, color='orange', 
                   label='Student Traders', edgecolor='black')
    axes[0,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[0,0].set_ylabel('Frequency', fontsize=12)
    axes[0,0].set_title('Price Distribution by Trader Type', fontsize=14, fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Price Deviation by Role
    pro_deviation = df_b[df_b['Role'] == 'Pro']['PriceDeviation']
    student_deviation = df_b[df_b['Role'] == 'Student']['PriceDeviation']
    
    axes[0,1].hist(pro_deviation, bins=50, alpha=0.7, color='green', 
                   label='Professional Traders', edgecolor='black')
    axes[0,1].hist(student_deviation, bins=50, alpha=0.7, color='orange', 
                   label='Student Traders', edgecolor='black')
    axes[0,1].axvline(x=1, color='red', linestyle='--', linewidth=2, 
                     label='Fundamental Value (1.0)')
    axes[0,1].set_xlabel('Price Deviation (Price/Fundamental)', fontsize=12)
    axes[0,1].set_ylabel('Frequency', fontsize=12)
    axes[0,1].set_title('Price Deviation by Trader Type', fontsize=14, fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Trading Volume by Role
    pro_volume = df_b[df_b['Role'] == 'Pro']['TradingVolume']
    student_volume = df_b[df_b['Role'] == 'Student']['TradingVolume']
    
    axes[1,0].hist(pro_volume, bins=30, alpha=0.7, color='green', 
                   label='Professional Traders', edgecolor='black')
    axes[1,0].hist(student_volume, bins=30, alpha=0.7, color='orange', 
                   label='Student Traders', edgecolor='black')
    axes[1,0].set_xlabel('Trading Volume (Units)', fontsize=12)
    axes[1,0].set_ylabel('Frequency', fontsize=12)
    axes[1,0].set_title('Trading Volume by Trader Type', fontsize=14, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Professional Share vs Average Price
    session_stats = df_b.groupby('Session').agg({
        'ProShare': 'first',
        'LastPrice': 'mean',
        'PriceDeviation': 'mean'
    })
    
    axes[1,1].scatter(session_stats['ProShare'], session_stats['LastPrice'], 
                     s=100, color='blue', alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Professional Trader Share', fontsize=12)
    axes[1,1].set_ylabel('Average Price (Francs)', fontsize=12)
    axes[1,1].set_title('Professional Share vs Average Price', fontsize=14, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(session_stats['ProShare'], session_stats['LastPrice'], 1)
    p = np.poly1d(z)
    axes[1,1].plot(session_stats['ProShare'], p(session_stats['ProShare']), 
                  "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('professional_vs_student_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Professional vs Student analysis plot saved as 'professional_vs_student_analysis.png'")

def create_professional_share_analysis(df_b):
    """Analyze the effect of professional trader share on market outcomes"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Group by session to get professional share and outcomes
    session_analysis = df_b.groupby('Session').agg({
        'ProShare': 'first',
        'LastPrice': 'mean',
        'PriceDeviation': 'mean',
        'TradingVolume': 'mean'
    }).reset_index()
    
    # Professional Share vs Price
    axes[0].scatter(session_analysis['ProShare'], session_analysis['LastPrice'], 
                   s=100, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Professional Trader Share', fontsize=12)
    axes[0].set_ylabel('Average Price (Francs)', fontsize=12)
    axes[0].set_title('Professional Share vs Average Price by Session', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(session_analysis['ProShare'], session_analysis['LastPrice'], 1)
    p = np.poly1d(z)
    axes[0].plot(session_analysis['ProShare'], p(session_analysis['ProShare']), 
                "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.1f}x+{z[1]:.1f}')
    axes[0].legend()
    
    # Professional Share vs Price Deviation
    axes[1].scatter(session_analysis['ProShare'], session_analysis['PriceDeviation'], 
                   s=100, color='red', alpha=0.7, edgecolor='black')
    axes[1].axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Fundamental Value')
    axes[1].set_xlabel('Professional Trader Share', fontsize=12)
    axes[1].set_ylabel('Average Price Deviation', fontsize=12)
    axes[1].set_title('Professional Share vs Price Deviation by Session', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Add trend line
    z2 = np.polyfit(session_analysis['ProShare'], session_analysis['PriceDeviation'], 1)
    p2 = np.poly1d(z2)
    axes[1].plot(session_analysis['ProShare'], p2(session_analysis['ProShare']), 
                "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z2[0]:.2f}x+{z2[1]:.2f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('professional_share_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Professional share analysis plot saved as 'professional_share_analysis.png'")

def main():
    """Main analysis function"""
    print("Loading Experiment B data...")
    df_b = load_experiment_b_data()
    
    print("Analyzing trading behavior by role...")
    role_stats, pro_share_stats = analyze_by_role(df_b)
    
    print("\nCreating role comparison plots...")
    create_role_comparison_plots(df_b)
    
    print("\nCreating professional share analysis...")
    create_professional_share_analysis(df_b)
    
    print("\nProfessional vs Student analysis complete!")

if __name__ == "__main__":
    main()
