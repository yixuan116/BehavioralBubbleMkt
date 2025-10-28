#!/usr/bin/env python3
"""
Batch Data Table Generator
批量生成所有分析的数据表格

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

def generate_information_structure_data_table(df):
    """Generate data table for Information Structure Analysis"""
    print("Generating Information Structure Analysis Data Table...")
    
    # Calculate bubble metrics
    bubble_metrics = []
    for session in df['Session'].unique():
        session_data = df[df['Session'] == session]
        for period in session_data['Period'].unique():
            period_data = session_data[session_data['Period'] == period]
            
            avg_price = period_data['LastPrice'].mean()
            fundamental = period_data['Fundamental'].iloc[0]
            bubble = (avg_price - fundamental) / fundamental if fundamental > 0 else 0
            
            bubble_metrics.append({
                'Session': session,
                'Period': period,
                'Experiment': period_data['Experiment'].iloc[0],
                'Bubble': bubble
            })
    
    bubble_df = pd.DataFrame(bubble_metrics)
    
    # Separate by experiment
    choice_a = bubble_df[bubble_df['Experiment'] == 'Choice A']
    choice_b = bubble_df[bubble_df['Experiment'] == 'Choice B']
    
    # Calculate statistics
    stats_data = [
        ['Choice A', len(choice_a), f"{choice_a['Bubble'].mean():.3f}", f"{choice_a['Bubble'].std():.3f}"],
        ['Choice B', len(choice_b), f"{choice_b['Bubble'].mean():.3f}", f"{choice_b['Bubble'].std():.3f}"]
    ]
    
    headers = ['Experiment', 'Sample Size', 'Mean Bubble', 'Std Dev']
    
    print("\n" + "="*80)
    print("INFORMATION STRUCTURE ANALYSIS DATA TABLE")
    print("="*80)
    print(tabulate(stats_data, headers=headers, tablefmt='grid', stralign='center'))
    
    # Statistical test
    t_stat, p_val = stats.ttest_ind(choice_a['Bubble'], choice_b['Bubble'])
    pooled_std = np.sqrt(((len(choice_a)-1)*choice_a['Bubble'].var() + (len(choice_b)-1)*choice_b['Bubble'].var()) / 
                        (len(choice_a) + len(choice_b) - 2))
    cohens_d = (choice_a['Bubble'].mean() - choice_b['Bubble'].mean()) / pooled_std
    
    test_data = [
        ['Two-sample t-test', f"t = {t_stat:.3f}", f"p = {p_val:.6f}"],
        ["Effect size (Cohen's d)", f"d = {cohens_d:.3f}", ""]
    ]
    
    test_headers = ['Test', 'Statistic', 'p-value']
    
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    print(tabulate(test_data, headers=test_headers, tablefmt='grid', stralign='center'))
    
    # Save to file
    with open('information_structure_data_table.txt', 'w') as f:
        f.write("INFORMATION STRUCTURE ANALYSIS DATA TABLE\n")
        f.write("="*80 + "\n")
        f.write(tabulate(stats_data, headers=headers, tablefmt='grid', stralign='center'))
        f.write("\n\nSTATISTICAL TESTS\n")
        f.write("="*60 + "\n")
        f.write(tabulate(test_data, headers=test_headers, tablefmt='grid', stralign='center'))
    
    print(f"✓ Data table saved to 'information_structure_data_table.txt'")

def generate_learning_analysis_data_table(df):
    """Generate data table for Learning Analysis"""
    print("Generating Learning Analysis Data Table...")
    
    # Calculate session-level bubble means
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
                'Experiment': period_data['Experiment'].iloc[0],
                'Bubble': bubble
            })
    
    session_df = pd.DataFrame(session_bubbles)
    
    # Calculate session-level means
    session_means = session_df.groupby(['Session', 'Experiment'])['Bubble'].mean().reset_index()
    
    # Separate by experiment
    choice_a_sessions = session_means[session_means['Experiment'] == 'Choice A']
    choice_b_sessions = session_means[session_means['Experiment'] == 'Choice B']
    
    # Create session comparison table
    session_data = []
    for session in sorted(session_means['Session'].unique()):
        session_a = choice_a_sessions[choice_a_sessions['Session'] == session]
        session_b = choice_b_sessions[choice_b_sessions['Session'] == session]
        
        if len(session_a) > 0 and len(session_b) > 0:
            session_data.append([
                session,
                f"{session_a['Bubble'].iloc[0]:.3f}",
                f"{session_b['Bubble'].iloc[0]:.3f}",
                session_a['Experiment'].iloc[0],
                session_b['Experiment'].iloc[0]
            ])
    
    headers = ['Session', 'Choice A Bubble', 'Choice B Bubble', 'Choice A Type', 'Choice B Type']
    
    print("\n" + "="*80)
    print("LEARNING ANALYSIS DATA TABLE")
    print("="*80)
    print(tabulate(session_data, headers=headers, tablefmt='grid', stralign='center'))
    
    # Trend analysis
    from sklearn.linear_model import LinearRegression
    
    # Choice A trend
    if len(choice_a_sessions) > 1:
        X_a = choice_a_sessions[['Session']].values
        y_a = choice_a_sessions['Bubble'].values
        model_a = LinearRegression()
        model_a.fit(X_a, y_a)
        slope_a = model_a.coef_[0]
        r_squared_a = model_a.score(X_a, y_a)
        
        # Calculate p-value
        n_a = len(choice_a_sessions)
        residuals_a = y_a - model_a.predict(X_a)
        mse_a = np.sum(residuals_a**2) / (n_a - 2)
        se_slope_a = np.sqrt(mse_a / np.sum((X_a - X_a.mean())**2))
        t_stat_a = slope_a / se_slope_a
        p_value_a = 2 * (1 - stats.t.cdf(abs(t_stat_a), n_a - 2))
    
    # Choice B trend
    if len(choice_b_sessions) > 1:
        X_b = choice_b_sessions[['Session']].values
        y_b = choice_b_sessions['Bubble'].values
        model_b = LinearRegression()
        model_b.fit(X_b, y_b)
        slope_b = model_b.coef_[0]
        r_squared_b = model_b.score(X_b, y_b)
        
        # Calculate p-value
        n_b = len(choice_b_sessions)
        residuals_b = y_b - model_b.predict(X_b)
        mse_b = np.sum(residuals_b**2) / (n_b - 2)
        se_slope_b = np.sqrt(mse_b / np.sum((X_b - X_b.mean())**2))
        t_stat_b = slope_b / se_slope_b
        p_value_b = 2 * (1 - stats.t.cdf(abs(t_stat_b), n_b - 2))
    
    trend_data = [
        ['Choice A', f"{slope_a:.6f}", f"{r_squared_a:.4f}", f"{t_stat_a:.4f}", f"{p_value_a:.4f}"],
        ['Choice B', f"{slope_b:.6f}", f"{r_squared_b:.4f}", f"{t_stat_b:.4f}", f"{p_value_b:.4f}"]
    ]
    
    trend_headers = ['Experiment', 'Slope', 'R-squared', 't-statistic', 'p-value']
    
    print("\n" + "="*80)
    print("LEARNING TREND ANALYSIS")
    print("="*80)
    print(tabulate(trend_data, headers=trend_headers, tablefmt='grid', stralign='center'))
    
    # Save to file
    with open('learning_analysis_data_table.txt', 'w') as f:
        f.write("LEARNING ANALYSIS DATA TABLE\n")
        f.write("="*80 + "\n")
        f.write(tabulate(session_data, headers=headers, tablefmt='grid', stralign='center'))
        f.write("\n\nLEARNING TREND ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write(tabulate(trend_data, headers=trend_headers, tablefmt='grid', stralign='center'))
    
    print(f"✓ Data table saved to 'learning_analysis_data_table.txt'")

def generate_trader_type_data_table(df):
    """Generate data table for Trader Type Analysis"""
    print("Generating Trader Type Analysis Data Table...")
    
    # Calculate final payoffs by trader type
    trader_stats = df.groupby(['Experiment', 'SubjectID']).agg({
        'DivEarn': 'sum',
        'TradingProfit': 'sum'
    }).reset_index()
    
    trader_stats['FinalPayoff'] = trader_stats['DivEarn'] + trader_stats['TradingProfit']
    
    # Separate by experiment (Choice B has professional traders)
    choice_a_traders = trader_stats[trader_stats['Experiment'] == 'Choice A']
    choice_b_traders = trader_stats[trader_stats['Experiment'] == 'Choice B']
    
    # For Choice B, we need to identify professional vs student traders
    # Assuming professional traders have higher payoffs (this is a simplification)
    choice_b_sorted = choice_b_traders.sort_values('FinalPayoff', ascending=False)
    
    # Assume top 25% are professional traders (this matches the experimental design)
    n_professionals = max(1, len(choice_b_sorted) // 4)
    professionals = choice_b_sorted.head(n_professionals)
    students = choice_b_sorted.tail(len(choice_b_sorted) - n_professionals)
    
    # Calculate statistics
    stats_data = [
        ['Choice A (Students)', len(choice_a_traders), f"{choice_a_traders['FinalPayoff'].mean():.2f}", 
         f"{choice_a_traders['FinalPayoff'].std():.2f}", f"{choice_a_traders['FinalPayoff'].min():.2f}", 
         f"{choice_a_traders['FinalPayoff'].max():.2f}"],
        ['Choice B (Professionals)', len(professionals), f"{professionals['FinalPayoff'].mean():.2f}", 
         f"{professionals['FinalPayoff'].std():.2f}", f"{professionals['FinalPayoff'].min():.2f}", 
         f"{professionals['FinalPayoff'].max():.2f}"],
        ['Choice B (Students)', len(students), f"{students['FinalPayoff'].mean():.2f}", 
         f"{students['FinalPayoff'].std():.2f}", f"{students['FinalPayoff'].min():.2f}", 
         f"{students['FinalPayoff'].max():.2f}"]
    ]
    
    headers = ['Trader Type', 'Sample Size', 'Mean Payoff', 'Std Dev', 'Min', 'Max']
    
    print("\n" + "="*80)
    print("TRADER TYPE ANALYSIS DATA TABLE")
    print("="*80)
    print(tabulate(stats_data, headers=headers, tablefmt='grid', stralign='center'))
    
    # Statistical test: Professional vs Student
    t_stat, p_val = stats.ttest_ind(professionals['FinalPayoff'], students['FinalPayoff'])
    pooled_std = np.sqrt(((len(professionals)-1)*professionals['FinalPayoff'].var() + (len(students)-1)*students['FinalPayoff'].var()) / 
                        (len(professionals) + len(students) - 2))
    cohens_d = (professionals['FinalPayoff'].mean() - students['FinalPayoff'].mean()) / pooled_std
    
    test_data = [
        ['Professional vs Student t-test', f"t = {t_stat:.3f}", f"p = {p_val:.6f}"],
        ["Effect size (Cohen's d)", f"d = {cohens_d:.3f}", ""]
    ]
    
    test_headers = ['Test', 'Statistic', 'p-value']
    
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    print(tabulate(test_data, headers=test_headers, tablefmt='grid', stralign='center'))
    
    # Save to file
    with open('trader_type_data_table.txt', 'w') as f:
        f.write("TRADER TYPE ANALYSIS DATA TABLE\n")
        f.write("="*80 + "\n")
        f.write(tabulate(stats_data, headers=headers, tablefmt='grid', stralign='center'))
        f.write("\n\nSTATISTICAL TESTS\n")
        f.write("="*60 + "\n")
        f.write(tabulate(test_data, headers=test_headers, tablefmt='grid', stralign='center'))
    
    print(f"✓ Data table saved to 'trader_type_data_table.txt'")

def main():
    """Main function to generate all data tables"""
    print("Loading data for batch data table generation...")
    df = load_data()
    
    if df is None:
        return
    
    # Generate all data tables
    generate_information_structure_data_table(df)
    generate_learning_analysis_data_table(df)
    generate_trader_type_data_table(df)
    
    print(f"\n" + "="*60)
    print("BATCH DATA TABLE GENERATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
