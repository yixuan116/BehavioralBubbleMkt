#!/usr/bin/env python3
"""
Aggregate Dynamics Analysis (N=9)
Behaviors Layer Analysis: Bubble Trends Across Periods

Hypothesis: H0: No trend vs H1: Trend exists
Expected: Negative β → Learning or self-correction
Data: Session-level means
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load both Choice A and B data for aggregate dynamics analysis"""
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
            
            # Calculate bubble variance (dispersion)
            bubble_variance = period_data['LastPrice'].var()
            
            # Calculate absolute bubble
            abs_bubble = abs(bubble)
            
            bubble_metrics.append({
                'Session': session,
                'Period': period,
                'Experiment': period_data['Experiment'].iloc[0],
                'AvgPrice': avg_price,
                'Fundamental': fundamental,
                'Bubble': bubble,
                'AbsBubble': abs_bubble,
                'BubbleVariance': bubble_variance,
                'PriceStd': period_data['LastPrice'].std()
            })
    
    return pd.DataFrame(bubble_metrics)

def analyze_aggregate_dynamics(bubble_df):
    """Analyze aggregate dynamics and trends in bubble formation"""
    print("\n" + "="*60)
    print("AGGREGATE DYNAMICS ANALYSIS (N=9)")
    print("="*60)
    
    # Separate by experiment
    choice_a = bubble_df[bubble_df['Experiment'] == 'Choice A']
    choice_b = bubble_df[bubble_df['Experiment'] == 'Choice B']
    
    print(f"\nSample sizes:")
    print(f"Choice A periods: {len(choice_a)}")
    print(f"Choice B periods: {len(choice_b)}")
    
    # Overall trend analysis
    print(f"\nOverall Trend Analysis:")
    
    # Linear regression: Bubble ~ Period
    X = bubble_df[['Period']].values
    y = bubble_df['Bubble'].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate statistics
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    
    # Calculate p-value for slope
    n = len(bubble_df)
    residuals = y - model.predict(X)
    mse = np.sum(residuals**2) / (n - 2)
    se_slope = np.sqrt(mse / np.sum((X - X.mean())**2))
    t_stat = slope / se_slope
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    
    print(f"Overall trend (all periods):")
    print(f"  Slope: {slope:.6f}")
    print(f"  Intercept: {intercept:.6f}")
    print(f"  R-squared: {r_squared:.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    # Choice A trend analysis
    if len(choice_a) > 1:
        X_a = choice_a[['Period']].values
        y_a = choice_a['Bubble'].values
        
        model_a = LinearRegression()
        model_a.fit(X_a, y_a)
        
        slope_a = model_a.coef_[0]
        r_squared_a = model_a.score(X_a, y_a)
        
        # Calculate p-value for Choice A
        n_a = len(choice_a)
        residuals_a = y_a - model_a.predict(X_a)
        mse_a = np.sum(residuals_a**2) / (n_a - 2)
        se_slope_a = np.sqrt(mse_a / np.sum((X_a - X_a.mean())**2))
        t_stat_a = slope_a / se_slope_a
        p_value_a = 2 * (1 - stats.t.cdf(abs(t_stat_a), n_a - 2))
        
        print(f"\nChoice A trend:")
        print(f"  Slope: {slope_a:.6f}")
        print(f"  R-squared: {r_squared_a:.4f}")
        print(f"  t-statistic: {t_stat_a:.4f}")
        print(f"  p-value: {p_value_a:.4f}")
    
    # Choice B trend analysis
    if len(choice_b) > 1:
        X_b = choice_b[['Period']].values
        y_b = choice_b['Bubble'].values
        
        model_b = LinearRegression()
        model_b.fit(X_b, y_b)
        
        slope_b = model_b.coef_[0]
        r_squared_b = model_b.score(X_b, y_b)
        
        # Calculate p-value for Choice B
        n_b = len(choice_b)
        residuals_b = y_b - model_b.predict(X_b)
        mse_b = np.sum(residuals_b**2) / (n_b - 2)
        se_slope_b = np.sqrt(mse_b / np.sum((X_b - X_b.mean())**2))
        t_stat_b = slope_b / se_slope_b
        p_value_b = 2 * (1 - stats.t.cdf(abs(t_stat_b), n_b - 2))
        
        print(f"\nChoice B trend:")
        print(f"  Slope: {slope_b:.6f}")
        print(f"  R-squared: {r_squared_b:.4f}")
        print(f"  t-statistic: {t_stat_b:.4f}")
        print(f"  p-value: {p_value_b:.4f}")
    
    # Session-level trend analysis
    print(f"\nSession-level Trend Analysis:")
    session_trends = []
    
    for session in bubble_df['Session'].unique():
        session_data = bubble_df[bubble_df['Session'] == session]
        
        if len(session_data) > 1:
            X_session = session_data[['Period']].values
            y_session = session_data['Bubble'].values
            
            model_session = LinearRegression()
            model_session.fit(X_session, y_session)
            
            slope_session = model_session.coef_[0]
            r_squared_session = model_session.score(X_session, y_session)
            
            # Calculate p-value for session
            n_session = len(session_data)
            residuals_session = y_session - model_session.predict(X_session)
            mse_session = np.sum(residuals_session**2) / (n_session - 2)
            se_slope_session = np.sqrt(mse_session / np.sum((X_session - X_session.mean())**2))
            t_stat_session = slope_session / se_slope_session
            p_value_session = 2 * (1 - stats.t.cdf(abs(t_stat_session), n_session - 2))
            
            session_trends.append({
                'Session': session,
                'Experiment': session_data['Experiment'].iloc[0],
                'Slope': slope_session,
                'R_squared': r_squared_session,
                'P_value': p_value_session,
                'Significant': p_value_session < 0.05
            })
    
    session_trends_df = pd.DataFrame(session_trends)
    print(session_trends_df.to_string(index=False))
    
    # Interpretation
    print(f"\nInterpretation:")
    if p_value < 0.05:
        if slope < 0:
            print("✓ Significant negative trend in bubble formation")
            print("  → Bubbles decrease over time (learning/self-correction)")
        else:
            print("✓ Significant positive trend in bubble formation")
            print("  → Bubbles increase over time (bubble growth)")
    else:
        print("✗ No significant trend in bubble formation")
    
    # Count significant sessions
    significant_sessions = session_trends_df[session_trends_df['Significant']]
    print(f"\nSignificant trends by session: {len(significant_sessions)}/{len(session_trends_df)}")
    
    if len(significant_sessions) > 0:
        negative_trends = significant_sessions[significant_sessions['Slope'] < 0]
        positive_trends = significant_sessions[significant_sessions['Slope'] > 0]
        
        print(f"  Negative trends (learning): {len(negative_trends)}")
        print(f"  Positive trends (bubble growth): {len(positive_trends)}")
    
    return bubble_df, session_trends_df, slope, p_value, r_squared

def create_visualizations(bubble_df, session_trends_df):
    """Create visualizations for aggregate dynamics analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Aggregate Dynamics Analysis (N=9): Bubble Trends Across Periods', 
                 fontsize=16, fontweight='bold')
    
    # 1. Overall bubble trend
    ax1 = axes[0, 0]
    ax1.scatter(bubble_df['Period'], bubble_df['Bubble'], alpha=0.6, s=20, color='blue')
    
    # Add trend line
    z = np.polyfit(bubble_df['Period'], bubble_df['Bubble'], 1)
    p = np.poly1d(z)
    ax1.plot(bubble_df['Period'], p(bubble_df['Period']), "r-", linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Bubble (P-F)/F')
    ax1.set_title('Overall Bubble Trend Across Periods')
    ax1.grid(True, alpha=0.3)
    
    # 2. Bubble trends by experiment
    ax2 = axes[0, 1]
    choice_a = bubble_df[bubble_df['Experiment'] == 'Choice A']
    choice_b = bubble_df[bubble_df['Experiment'] == 'Choice B']
    
    ax2.scatter(choice_a['Period'], choice_a['Bubble'], alpha=0.6, s=20, color='red', label='Choice A')
    ax2.scatter(choice_b['Period'], choice_b['Bubble'], alpha=0.6, s=20, color='blue', label='Choice B')
    
    # Add trend lines
    if len(choice_a) > 1:
        z_a = np.polyfit(choice_a['Period'], choice_a['Bubble'], 1)
        p_a = np.poly1d(z_a)
        ax2.plot(choice_a['Period'], p_a(choice_a['Period']), "r-", linewidth=2, alpha=0.8)
    
    if len(choice_b) > 1:
        z_b = np.polyfit(choice_b['Period'], choice_b['Bubble'], 1)
        p_b = np.poly1d(z_b)
        ax2.plot(choice_b['Period'], p_b(choice_b['Period']), "b-", linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Bubble (P-F)/F')
    ax2.set_title('Bubble Trends by Experiment')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Session-level slopes
    ax3 = axes[1, 0]
    colors = ['red' if row['Experiment'] == 'Choice A' else 'blue' for _, row in session_trends_df.iterrows()]
    ax3.bar(range(len(session_trends_df)), session_trends_df['Slope'], color=colors, alpha=0.7)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Session')
    ax3.set_ylabel('Slope (Bubble Trend)')
    ax3.set_title('Session-level Bubble Slopes')
    ax3.set_xticks(range(len(session_trends_df)))
    ax3.set_xticklabels(session_trends_df['Session'])
    ax3.grid(True, alpha=0.3)
    
    # 4. R-squared by session
    ax4 = axes[1, 1]
    ax4.bar(range(len(session_trends_df)), session_trends_df['R_squared'], color=colors, alpha=0.7)
    ax4.set_xlabel('Session')
    ax4.set_ylabel('R-squared (Trend Strength)')
    ax4.set_title('Session-level Trend Strength (R²)')
    ax4.set_xticks(range(len(session_trends_df)))
    ax4.set_xticklabels(session_trends_df['Session'])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('aggregate_dynamics_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'aggregate_dynamics_analysis.png'")

def main():
    """Main analysis function"""
    print("Loading data for aggregate dynamics analysis...")
    df = load_data()
    
    if df is None:
        return
    
    print("Calculating bubble metrics...")
    bubble_df = calculate_bubble_metrics(df)
    
    print(f"Calculated bubble metrics for {len(bubble_df)} period observations")
    
    # Analyze aggregate dynamics
    bubble_df, session_trends_df, slope, p_value, r_squared = analyze_aggregate_dynamics(bubble_df)
    
    # Create visualizations
    create_visualizations(bubble_df, session_trends_df)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Key finding: {'Significant trend' if p_value < 0.05 else 'No significant trend'} in bubble formation")
    print(f"Trend direction: {'Negative (learning)' if slope < 0 else 'Positive (bubble growth)'}")
    print(f"Trend strength: R² = {r_squared:.3f}")

if __name__ == "__main__":
    main()
