#!/usr/bin/env python3
"""
Bubble Data Table Generator
生成类似图片中的bubble数据表格，包含Choice A和Choice B的详细数据

Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        # Combine datasets
        df = pd.concat([df_a, df_b], ignore_index=True)
        print(f"Loaded {len(df)} records total")
        return df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

def calculate_period_metrics(df):
    """Calculate detailed metrics for each period"""
    period_metrics = []
    
    for experiment in ['Choice A', 'Choice B']:
        exp_data = df[df['Experiment'] == experiment]
        
        for period in sorted(exp_data['Period'].unique()):
            period_data = exp_data[exp_data['Period'] == period]
            
            # Calculate metrics
            avg_price = period_data['LastPrice'].mean()
            fundamental = period_data['Fundamental'].iloc[0]
            bubble_abs = avg_price - fundamental
            bubble_pct = (bubble_abs / fundamental) * 100 if fundamental > 0 else 0
            
            period_metrics.append({
                'Experiment': experiment,
                'Period': period,
                'AvgPrice': avg_price,
                'Fundamental': fundamental,
                'BubbleAbs': bubble_abs,
                'BubblePct': bubble_pct
            })
    
    return pd.DataFrame(period_metrics)

def create_bubble_data_table(metrics_df):
    """Create the bubble data table similar to the image"""
    print("\n" + "="*100)
    print("BUBBLE DATA TABLE - Choice A vs Choice B")
    print("="*100)
    
    # Create table structure
    periods = list(range(1, 16))
    
    # Choice A data
    choice_a = metrics_df[metrics_df['Experiment'] == 'Choice A'].sort_values('Period')
    choice_b = metrics_df[metrics_df['Experiment'] == 'Choice B'].sort_values('Period')
    
    # Create table data
    table_data = []
    
    # Choice A rows
    table_data.append(['Choice A Lastprice-fundamental = Bubble', 'Choice A'] + 
                     [f"{row['BubbleAbs']:.1f}" for _, row in choice_a.iterrows()])
    
    table_data.append(['Choice A Fundamental', 'Choice A'] + 
                     [f"{row['Fundamental']:.1f}" for _, row in choice_a.iterrows()])
    
    table_data.append(['Choice A (Lastprice-fundamental)/fundamental = bubble %', 'Choice A'] + 
                     [f"{row['BubblePct']:.0f}%" for _, row in choice_a.iterrows()])
    
    # Choice B rows
    table_data.append(['Choice B Lastprice-fundamental = Bubble', 'Choice B'] + 
                     [f"{row['BubbleAbs']:.1f}" for _, row in choice_b.iterrows()])
    
    table_data.append(['Choice B Fundamental', 'Choice B'] + 
                     [f"{row['Fundamental']:.1f}" for _, row in choice_b.iterrows()])
    
    table_data.append(['Choice B (Lastprice-fundamental)/fundamental = bubble %', 'Choice B'] + 
                     [f"{row['BubblePct']:.0f}%" for _, row in choice_b.iterrows()])
    
    # Create headers
    headers = ['Title', 'Choice'] + [f'Period {p}' for p in periods]
    
    # Print table
    print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='center'))
    
    return table_data, headers

def create_detailed_summary_table(metrics_df):
    """Create a more detailed summary table"""
    print("\n" + "="*120)
    print("DETAILED BUBBLE SUMMARY TABLE")
    print("="*120)
    
    summary_data = []
    
    for experiment in ['Choice A', 'Choice B']:
        exp_data = metrics_df[metrics_df['Experiment'] == experiment]
        
        # Calculate summary statistics
        avg_bubble_pct = exp_data['BubblePct'].mean()
        max_bubble_pct = exp_data['BubblePct'].max()
        min_bubble_pct = exp_data['BubblePct'].min()
        std_bubble_pct = exp_data['BubblePct'].std()
        
        # Count positive and negative bubbles
        positive_bubbles = len(exp_data[exp_data['BubblePct'] > 0])
        negative_bubbles = len(exp_data[exp_data['BubblePct'] < 0])
        
        summary_data.append([
            experiment,
            f"{avg_bubble_pct:.1f}%",
            f"{max_bubble_pct:.1f}%",
            f"{min_bubble_pct:.1f}%",
            f"{std_bubble_pct:.1f}%",
            positive_bubbles,
            negative_bubbles,
            f"{positive_bubbles/(positive_bubbles+negative_bubbles)*100:.0f}%"
        ])
    
    summary_headers = [
        'Experiment', 'Avg Bubble %', 'Max Bubble %', 'Min Bubble %', 
        'Std Dev %', 'Positive Periods', 'Negative Periods', '% Positive'
    ]
    
    print(tabulate(summary_data, headers=summary_headers, tablefmt='grid', stralign='center'))
    
    return summary_data, summary_headers

def save_data_to_files(metrics_df, table_data, headers):
    """Save data to CSV and Excel files"""
    print("\n" + "="*60)
    print("SAVING DATA TO FILES")
    print("="*60)
    
    # Save detailed metrics to CSV
    metrics_df.to_csv('bubble_metrics_detailed.csv', index=False)
    print("✓ Detailed metrics saved to 'bubble_metrics_detailed.csv'")
    
    # Save table data to CSV
    table_df = pd.DataFrame(table_data, columns=headers)
    table_df.to_csv('bubble_data_table.csv', index=False)
    print("✓ Bubble data table saved to 'bubble_data_table.csv'")
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter('bubble_analysis_data.xlsx', engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='Detailed_Metrics', index=False)
        table_df.to_excel(writer, sheet_name='Bubble_Table', index=False)
        
        # Add summary statistics
        summary_stats = metrics_df.groupby('Experiment').agg({
            'BubblePct': ['mean', 'std', 'min', 'max'],
            'BubbleAbs': ['mean', 'std', 'min', 'max']
        }).round(2)
        summary_stats.to_excel(writer, sheet_name='Summary_Stats')
    
    print("✓ All data saved to 'bubble_analysis_data.xlsx'")

def create_visualization_with_table(metrics_df):
    """Create visualization alongside the data table"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bubble Analysis: Data Tables and Visualizations', fontsize=16, fontweight='bold')
    
    # 1. Bubble percentage over periods
    ax1 = axes[0, 0]
    choice_a = metrics_df[metrics_df['Experiment'] == 'Choice A']
    choice_b = metrics_df[metrics_df['Experiment'] == 'Choice B']
    
    ax1.plot(choice_a['Period'], choice_a['BubblePct'], 'ro-', label='Choice A', linewidth=2, markersize=6)
    ax1.plot(choice_b['Period'], choice_b['BubblePct'], 'bo-', label='Choice B', linewidth=2, markersize=6)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Bubble %')
    ax1.set_title('Bubble Percentage Over Periods')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Absolute bubble over periods
    ax2 = axes[0, 1]
    ax2.plot(choice_a['Period'], choice_a['BubbleAbs'], 'ro-', label='Choice A', linewidth=2, markersize=6)
    ax2.plot(choice_b['Period'], choice_b['BubbleAbs'], 'bo-', label='Choice B', linewidth=2, markersize=6)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Bubble (Francs)')
    ax2.set_title('Absolute Bubble Over Periods')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Price vs Fundamental
    ax3 = axes[1, 0]
    ax3.plot(choice_a['Period'], choice_a['AvgPrice'], 'ro-', label='Choice A Price', linewidth=2, markersize=6)
    ax3.plot(choice_b['Period'], choice_b['AvgPrice'], 'bo-', label='Choice B Price', linewidth=2, markersize=6)
    ax3.plot(choice_a['Period'], choice_a['Fundamental'], 'g--', label='Fundamental', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Period')
    ax3.set_ylabel('Price (Francs)')
    ax3.set_title('Price vs Fundamental Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Bubble distribution
    ax4 = axes[1, 1]
    ax4.hist(choice_a['BubblePct'], bins=10, alpha=0.7, color='red', label='Choice A', density=True)
    ax4.hist(choice_b['BubblePct'], bins=10, alpha=0.7, color='blue', label='Choice B', density=True)
    ax4.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Bubble %')
    ax4.set_ylabel('Density')
    ax4.set_title('Bubble Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bubble_data_table_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualization saved as 'bubble_data_table_analysis.png'")

def main():
    """Main function"""
    print("Loading data for bubble data table generation...")
    df = load_data()
    
    if df is None:
        return
    
    print("Calculating period-level metrics...")
    metrics_df = calculate_period_metrics(df)
    
    print(f"Calculated metrics for {len(metrics_df)} period observations")
    
    # Create bubble data table
    table_data, headers = create_bubble_data_table(metrics_df)
    
    # Create detailed summary table
    summary_data, summary_headers = create_detailed_summary_table(metrics_df)
    
    # Save data to files
    save_data_to_files(metrics_df, table_data, headers)
    
    # Create visualization
    create_visualization_with_table(metrics_df)
    
    print(f"\n" + "="*60)
    print("BUBBLE DATA TABLE GENERATION COMPLETE")
    print("="*60)
    print("✓ Data tables created and displayed")
    print("✓ Files saved: bubble_metrics_detailed.csv, bubble_data_table.csv, bubble_analysis_data.xlsx")
    print("✓ Visualization saved: bubble_data_table_analysis.png")

if __name__ == "__main__":
    main()
