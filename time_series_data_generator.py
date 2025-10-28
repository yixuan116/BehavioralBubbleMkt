#!/usr/bin/env python3
"""
Time Series Data Table Generator
生成真正的time series格式数据表格，按period顺序显示价格变化

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
        
        # Combine datasets
        df = pd.concat([df_a, df_b], ignore_index=True)
        return df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

def generate_time_series_tables(df):
    """Generate time series data tables for all analyses"""
    print("Generating Time Series Data Tables...")
    
    # Calculate period-level metrics
    period_metrics = []
    
    for experiment in ['Choice A', 'Choice B']:
        exp_data = df[df['Experiment'] == experiment]
        
        for period in sorted(exp_data['Period'].unique()):
            period_data = exp_data[exp_data['Period'] == period]
            
            # Calculate metrics
            avg_price = period_data['LastPrice'].mean()
            price_std = period_data['LastPrice'].std()
            fundamental = period_data['Fundamental'].iloc[0]
            bubble_abs = avg_price - fundamental
            bubble_pct = (bubble_abs / fundamental) * 100 if fundamental > 0 else 0
            
            period_metrics.append({
                'Experiment': experiment,
                'Period': period,
                'AvgPrice': avg_price,
                'PriceStd': price_std,
                'Fundamental': fundamental,
                'BubbleAbs': bubble_abs,
                'BubblePct': bubble_pct
            })
    
    metrics_df = pd.DataFrame(period_metrics)
    
    # Generate Last Price Time Series Table
    print("\n" + "="*100)
    print("LAST PRICE TIME SERIES DATA TABLE")
    print("="*100)
    
    # Create time series table
    periods = sorted(metrics_df['Period'].unique())
    choice_a = metrics_df[metrics_df['Experiment'] == 'Choice A'].sort_values('Period')
    choice_b = metrics_df[metrics_df['Experiment'] == 'Choice B'].sort_values('Period')
    
    time_series_data = []
    for i, period in enumerate(periods):
        a_data = choice_a[choice_a['Period'] == period]
        b_data = choice_b[choice_b['Period'] == period]
        
        if len(a_data) > 0 and len(b_data) > 0:
            time_series_data.append([
                period,
                f"{a_data['AvgPrice'].iloc[0]:.1f}",
                f"{b_data['AvgPrice'].iloc[0]:.1f}",
                f"{a_data['Fundamental'].iloc[0]:.1f}",
                f"{a_data['BubbleAbs'].iloc[0]:.1f}",
                f"{b_data['BubbleAbs'].iloc[0]:.1f}",
                f"{a_data['BubblePct'].iloc[0]:.1f}%",
                f"{b_data['BubblePct'].iloc[0]:.1f}%"
            ])
    
    headers = [
        'Period', 'Choice A Price', 'Choice B Price', 'Fundamental', 
        'Choice A Bubble', 'Choice B Bubble', 'Choice A Bubble %', 'Choice B Bubble %'
    ]
    
    print(tabulate(time_series_data, headers=headers, tablefmt='grid', stralign='center'))
    
    # Generate Market Efficiency Time Series Table
    print("\n" + "="*120)
    print("MARKET EFFICIENCY TIME SERIES DATA TABLE")
    print("="*120)
    
    efficiency_data = []
    for i, period in enumerate(periods):
        a_data = choice_a[choice_a['Period'] == period]
        b_data = choice_b[choice_b['Period'] == period]
        
        if len(a_data) > 0 and len(b_data) > 0:
            efficiency_data.append([
                period,
                f"{a_data['AvgPrice'].iloc[0]:.1f}",
                f"{b_data['AvgPrice'].iloc[0]:.1f}",
                f"{a_data['Fundamental'].iloc[0]:.1f}",
                f"{a_data['BubblePct'].iloc[0]:.1f}%",
                f"{b_data['BubblePct'].iloc[0]:.1f}%",
                f"{a_data['PriceStd'].iloc[0]:.1f}",
                f"{b_data['PriceStd'].iloc[0]:.1f}"
            ])
    
    efficiency_headers = [
        'Period', 'Choice A Price', 'Choice B Price', 'Fundamental',
        'Choice A Bubble %', 'Choice B Bubble %', 'Choice A Std Dev', 'Choice B Std Dev'
    ]
    
    print(tabulate(efficiency_data, headers=efficiency_headers, tablefmt='grid', stralign='center'))
    
    # Generate Learning Analysis Time Series Table
    print("\n" + "="*100)
    print("LEARNING ANALYSIS TIME SERIES DATA TABLE")
    print("="*100)
    
    # Calculate session-level bubble means for learning analysis
    session_bubbles = []
    for session in df['Session'].unique():
        session_data = df[df['Session'] == session]
        for period in session_data['Period'].unique():
            period_data = session_data[session_data['Period'] == period]
            
            avg_price = period_data['LastPrice'].mean()
            fundamental = period_data['Fundamental'].iloc[0]
            bubble = (avg_price - fundamental) / fundamental if fundamental > 0 else 0
            
            session_bubbles.append({
                'Session': session,
                'Period': period,
                'Experiment': period_data['Experiment'].iloc[0],
                'Bubble': bubble
            })
    
    session_df = pd.DataFrame(session_bubbles)
    
    # Create learning time series table
    learning_data = []
    for session in sorted(session_df['Session'].unique()):
        session_data = session_df[session_df['Session'] == session]
        experiment = session_data['Experiment'].iloc[0]
        
        # Calculate session-level average bubble
        session_bubble = session_data['Bubble'].mean()
        
        learning_data.append([
            session,
            experiment,
            f"{session_bubble:.3f}",
            f"{session_bubble*100:.1f}%",
            len(session_data)
        ])
    
    learning_headers = [
        'Session', 'Experiment', 'Mean Bubble', 'Bubble %', 'Periods'
    ]
    
    print(tabulate(learning_data, headers=learning_headers, tablefmt='grid', stralign='center'))
    
    # Generate Aggregate Dynamics Time Series Table
    print("\n" + "="*120)
    print("AGGREGATE DYNAMICS TIME SERIES DATA TABLE")
    print("="*120)
    
    # Calculate overall trend
    from sklearn.linear_model import LinearRegression
    
    # Overall trend
    X = metrics_df[['Period']].values
    y = metrics_df['BubblePct'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    
    # Create trend analysis table
    trend_data = []
    for i, period in enumerate(periods):
        a_data = choice_a[choice_a['Period'] == period]
        b_data = choice_b[choice_b['Period'] == period]
        
        if len(a_data) > 0 and len(b_data) > 0:
            # Calculate trend values
            trend_value = slope * period + intercept
            
            trend_data.append([
                period,
                f"{a_data['BubblePct'].iloc[0]:.1f}%",
                f"{b_data['BubblePct'].iloc[0]:.1f}%",
                f"{trend_value:.1f}%",
                f"{a_data['AvgPrice'].iloc[0]:.1f}",
                f"{b_data['AvgPrice'].iloc[0]:.1f}"
            ])
    
    trend_headers = [
        'Period', 'Choice A Bubble %', 'Choice B Bubble %', 'Trend Line %',
        'Choice A Price', 'Choice B Price'
    ]
    
    print(tabulate(trend_data, headers=trend_headers, tablefmt='grid', stralign='center'))
    
    # Print trend statistics
    print(f"\nTrend Statistics:")
    print(f"Slope: {slope:.3f}% per period")
    print(f"Intercept: {intercept:.3f}%")
    print(f"R-squared: {r_squared:.3f}")
    
    # Save all tables to files
    with open('time_series_data_tables.txt', 'w') as f:
        f.write("TIME SERIES DATA TABLES\n")
        f.write("="*100 + "\n\n")
        
        f.write("LAST PRICE TIME SERIES DATA TABLE\n")
        f.write("="*100 + "\n")
        f.write(tabulate(time_series_data, headers=headers, tablefmt='grid', stralign='center'))
        
        f.write("\n\nMARKET EFFICIENCY TIME SERIES DATA TABLE\n")
        f.write("="*120 + "\n")
        f.write(tabulate(efficiency_data, headers=efficiency_headers, tablefmt='grid', stralign='center'))
        
        f.write("\n\nLEARNING ANALYSIS TIME SERIES DATA TABLE\n")
        f.write("="*100 + "\n")
        f.write(tabulate(learning_data, headers=learning_headers, tablefmt='grid', stralign='center'))
        
        f.write("\n\nAGGREGATE DYNAMICS TIME SERIES DATA TABLE\n")
        f.write("="*120 + "\n")
        f.write(tabulate(trend_data, headers=trend_headers, tablefmt='grid', stralign='center'))
        
        f.write(f"\n\nTrend Statistics:\n")
        f.write(f"Slope: {slope:.3f}% per period\n")
        f.write(f"Intercept: {intercept:.3f}%\n")
        f.write(f"R-squared: {r_squared:.3f}\n")
    
    print(f"\n✓ All time series data tables saved to 'time_series_data_tables.txt'")
    
    return metrics_df, time_series_data, efficiency_data, learning_data, trend_data

def main():
    """Main function"""
    print("Loading data for time series data table generation...")
    df = load_data()
    
    if df is None:
        return
    
    # Generate time series tables
    metrics_df, time_series_data, efficiency_data, learning_data, trend_data = generate_time_series_tables(df)
    
    print(f"\n" + "="*60)
    print("TIME SERIES DATA TABLE GENERATION COMPLETE")
    print("="*60)
    print("✓ Generated true time series format data tables")
    print("✓ Tables show period-by-period progression")
    print("✓ All data organized chronologically")
    print("✓ Ready for time series analysis and verification")

if __name__ == "__main__":
    main()
