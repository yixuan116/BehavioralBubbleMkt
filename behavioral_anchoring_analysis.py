#!/usr/bin/env python3
"""
Behavioral Anchoring Analysis (N=10)
Behaviors Layer Analysis: Anchoring Bias in Price Formation

Hypothesis: H0: β_trade = β_fundamental vs H1: β_trade ≠ β_fundamental
Expected: β_trade > β_fundamental → Anchoring bias
Data: Period-level panel
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
    """Load both Choice A and B data for behavioral anchoring analysis"""
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

def calculate_lagged_variables(df):
    """Calculate lagged prices and fundamentals for anchoring analysis"""
    df_sorted = df.sort_values(['Session', 'Period']).copy()
    
    # Calculate lagged variables
    df_sorted['Price_lag1'] = df_sorted.groupby('Session')['LastPrice'].shift(1)
    df_sorted['Price_lag2'] = df_sorted.groupby('Session')['LastPrice'].shift(2)
    df_sorted['Fundamental_lag1'] = df_sorted.groupby('Session')['Fundamental'].shift(1)
    df_sorted['Fundamental_lag2'] = df_sorted.groupby('Session')['Fundamental'].shift(2)
    
    # Calculate price changes
    df_sorted['Price_change'] = df_sorted['LastPrice'] - df_sorted['Price_lag1']
    df_sorted['Fundamental_change'] = df_sorted['Fundamental'] - df_sorted['Fundamental_lag1']
    
    # Calculate relative changes
    df_sorted['Price_change_rel'] = df_sorted['Price_change'] / df_sorted['Price_lag1']
    df_sorted['Fundamental_change_rel'] = df_sorted['Fundamental_change'] / df_sorted['Fundamental_lag1']
    
    # Remove rows with missing lagged values
    df_clean = df_sorted.dropna(subset=['Price_lag1', 'Fundamental_lag1', 'LastPrice']).copy()
    
    print(f"After calculating lagged variables: {len(df_clean)} records")
    print(f"Removed {len(df_sorted) - len(df_clean)} records with missing lagged values")
    
    return df_clean

def analyze_anchoring_bias(df_clean):
    """Analyze anchoring bias in price formation"""
    print("\n" + "="*60)
    print("BEHAVIORAL ANCHORING ANALYSIS (N=10)")
    print("="*60)
    
    # Separate by experiment
    choice_a = df_clean[df_clean['Experiment'] == 'Choice A']
    choice_b = df_clean[df_clean['Experiment'] == 'Choice B']
    
    print(f"\nSample sizes:")
    print(f"Choice A: {len(choice_a)} periods")
    print(f"Choice B: {len(choice_b)} periods")
    
    # Overall anchoring analysis
    print(f"\nOverall Anchoring Analysis:")
    
    # Model 1: Price_t = α + β_trade * Price_{t-1} + β_fundamental * Fundamental_{t-1} + ε
    X_overall = df_clean[['Price_lag1', 'Fundamental_lag1']].values
    y_overall = df_clean['LastPrice'].values
    
    model_overall = LinearRegression()
    model_overall.fit(X_overall, y_overall)
    
    beta_trade = model_overall.coef_[0]
    beta_fundamental = model_overall.coef_[1]
    alpha = model_overall.intercept_
    r_squared = model_overall.score(X_overall, y_overall)
    
    print(f"Overall model: Price_t = {alpha:.4f} + {beta_trade:.4f}*Price_{{t-1}} + {beta_fundamental:.4f}*Fundamental_{{t-1}}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"β_trade: {beta_trade:.4f}")
    print(f"β_fundamental: {beta_fundamental:.4f}")
    print(f"β_trade / β_fundamental ratio: {beta_trade / beta_fundamental:.4f}")
    
    # Statistical test for anchoring bias
    # H0: β_trade = β_fundamental vs H1: β_trade ≠ β_fundamental
    diff_beta = beta_trade - beta_fundamental
    
    # Calculate standard error for the difference
    n = len(df_clean)
    residuals = y_overall - model_overall.predict(X_overall)
    mse = np.sum(residuals**2) / (n - 3)
    
    # Calculate covariance matrix
    X_with_intercept = np.column_stack([np.ones(n), X_overall])
    cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    
    # Standard error for difference
    se_diff = np.sqrt(cov_matrix[1, 1] + cov_matrix[2, 2] - 2 * cov_matrix[1, 2])
    t_stat = diff_beta / se_diff
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 3))
    
    print(f"\nStatistical test for anchoring bias:")
    print(f"  Difference (β_trade - β_fundamental): {diff_beta:.4f}")
    print(f"  Standard error: {se_diff:.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    # Choice A analysis
    if len(choice_a) > 2:
        X_a = choice_a[['Price_lag1', 'Fundamental_lag1']].values
        y_a = choice_a['LastPrice'].values
        
        model_a = LinearRegression()
        model_a.fit(X_a, y_a)
        
        beta_trade_a = model_a.coef_[0]
        beta_fundamental_a = model_a.coef_[1]
        r_squared_a = model_a.score(X_a, y_a)
        
        print(f"\nChoice A model:")
        print(f"  β_trade: {beta_trade_a:.4f}")
        print(f"  β_fundamental: {beta_fundamental_a:.4f}")
        print(f"  β_trade / β_fundamental ratio: {beta_trade_a / beta_fundamental_a:.4f}")
        print(f"  R-squared: {r_squared_a:.4f}")
    
    # Choice B analysis
    if len(choice_b) > 2:
        X_b = choice_b[['Price_lag1', 'Fundamental_lag1']].values
        y_b = choice_b['LastPrice'].values
        
        model_b = LinearRegression()
        model_b.fit(X_b, y_b)
        
        beta_trade_b = model_b.coef_[0]
        beta_fundamental_b = model_b.coef_[1]
        r_squared_b = model_b.score(X_b, y_b)
        
        print(f"\nChoice B model:")
        print(f"  β_trade: {beta_trade_b:.4f}")
        print(f"  β_fundamental: {beta_fundamental_b:.4f}")
        print(f"  β_trade / β_fundamental ratio: {beta_trade_b / beta_fundamental_b:.4f}")
        print(f"  R-squared: {r_squared_b:.4f}")
    
    # Session-level analysis
    print(f"\nSession-level Anchoring Analysis:")
    session_results = []
    
    for session in df_clean['Session'].unique():
        session_data = df_clean[df_clean['Session'] == session]
        
        if len(session_data) > 2:
            X_session = session_data[['Price_lag1', 'Fundamental_lag1']].values
            y_session = session_data['LastPrice'].values
            
            model_session = LinearRegression()
            model_session.fit(X_session, y_session)
            
            beta_trade_session = model_session.coef_[0]
            beta_fundamental_session = model_session.coef_[1]
            r_squared_session = model_session.score(X_session, y_session)
            
            session_results.append({
                'Session': session,
                'Experiment': session_data['Experiment'].iloc[0],
                'Beta_trade': beta_trade_session,
                'Beta_fundamental': beta_fundamental_session,
                'Ratio': beta_trade_session / beta_fundamental_session,
                'R_squared': r_squared_session
            })
    
    session_df = pd.DataFrame(session_results)
    print(session_df.to_string(index=False))
    
    # Interpretation
    print(f"\nInterpretation:")
    if p_value < 0.05:
        if beta_trade > beta_fundamental:
            print("✓ Significant anchoring bias detected")
            print("  → Prices anchor more to previous prices than fundamentals")
        else:
            print("✓ Significant anti-anchoring bias detected")
            print("  → Prices anchor more to fundamentals than previous prices")
    else:
        print("✗ No significant anchoring bias detected")
        print("  → Prices anchor equally to previous prices and fundamentals")
    
    # Anchoring strength
    if beta_trade > 0 and beta_fundamental > 0:
        anchoring_strength = beta_trade / (beta_trade + beta_fundamental)
        print(f"\nAnchoring strength: {anchoring_strength:.3f}")
        print(f"  → {anchoring_strength*100:.1f}% of price formation comes from previous prices")
        print(f"  → {(1-anchoring_strength)*100:.1f}% of price formation comes from fundamentals")
    
    return df_clean, beta_trade, beta_fundamental, p_value, r_squared, session_df

def create_visualizations(df_clean, session_df):
    """Create visualizations for behavioral anchoring analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Behavioral Anchoring Analysis (N=10): Price Anchoring Bias', 
                 fontsize=16, fontweight='bold')
    
    # 1. Price vs Lagged Price
    ax1 = axes[0, 0]
    ax1.scatter(df_clean['Price_lag1'], df_clean['LastPrice'], alpha=0.6, s=20, color='blue')
    
    # Add 45-degree line (perfect anchoring)
    min_price = min(df_clean['Price_lag1'].min(), df_clean['LastPrice'].min())
    max_price = max(df_clean['Price_lag1'].max(), df_clean['LastPrice'].max())
    ax1.plot([min_price, max_price], [min_price, max_price], 'r--', alpha=0.8, label='Perfect Anchoring')
    
    # Add regression line
    z = np.polyfit(df_clean['Price_lag1'], df_clean['LastPrice'], 1)
    p = np.poly1d(z)
    ax1.plot(df_clean['Price_lag1'], p(df_clean['Price_lag1']), "g-", linewidth=2, alpha=0.8, label='Regression')
    
    ax1.set_xlabel('Lagged Price (P_{t-1})')
    ax1.set_ylabel('Current Price (P_t)')
    ax1.set_title('Price Anchoring: Current vs Lagged Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Price vs Lagged Fundamental
    ax2 = axes[0, 1]
    ax2.scatter(df_clean['Fundamental_lag1'], df_clean['LastPrice'], alpha=0.6, s=20, color='green')
    
    # Add 45-degree line (perfect fundamental anchoring)
    min_fund = min(df_clean['Fundamental_lag1'].min(), df_clean['LastPrice'].min())
    max_fund = max(df_clean['Fundamental_lag1'].max(), df_clean['LastPrice'].max())
    ax2.plot([min_fund, max_fund], [min_fund, max_fund], 'r--', alpha=0.8, label='Perfect Fundamental Anchoring')
    
    # Add regression line
    z = np.polyfit(df_clean['Fundamental_lag1'], df_clean['LastPrice'], 1)
    p = np.poly1d(z)
    ax2.plot(df_clean['Fundamental_lag1'], p(df_clean['Fundamental_lag1']), "b-", linewidth=2, alpha=0.8, label='Regression')
    
    ax2.set_xlabel('Lagged Fundamental (F_{t-1})')
    ax2.set_ylabel('Current Price (P_t)')
    ax2.set_title('Fundamental Anchoring: Current Price vs Lagged Fundamental')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Beta coefficients by session
    ax3 = axes[1, 0]
    x_pos = range(len(session_df))
    width = 0.35
    
    ax3.bar([x - width/2 for x in x_pos], session_df['Beta_trade'], width, 
           label='β_trade', color='blue', alpha=0.7)
    ax3.bar([x + width/2 for x in x_pos], session_df['Beta_fundamental'], width, 
           label='β_fundamental', color='green', alpha=0.7)
    
    ax3.set_xlabel('Session')
    ax3.set_ylabel('Beta Coefficient')
    ax3.set_title('Anchoring Coefficients by Session')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(session_df['Session'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Anchoring ratio by session
    ax4 = axes[1, 1]
    colors = ['red' if row['Experiment'] == 'Choice A' else 'blue' for _, row in session_df.iterrows()]
    ax4.bar(range(len(session_df)), session_df['Ratio'], color=colors, alpha=0.7)
    ax4.axhline(1, color='black', linestyle='--', alpha=0.5, label='Equal Anchoring')
    ax4.set_xlabel('Session')
    ax4.set_ylabel('β_trade / β_fundamental Ratio')
    ax4.set_title('Anchoring Ratio by Session')
    ax4.set_xticks(range(len(session_df)))
    ax4.set_xticklabels(session_df['Session'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('behavioral_anchoring_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'behavioral_anchoring_analysis.png'")

def main():
    """Main analysis function"""
    print("Loading data for behavioral anchoring analysis...")
    df = load_data()
    
    if df is None:
        return
    
    print("Calculating lagged variables...")
    df_clean = calculate_lagged_variables(df)
    
    print(f"Analyzing {len(df_clean)} periods with lagged data")
    
    # Analyze anchoring bias
    df_clean, beta_trade, beta_fundamental, p_value, r_squared, session_df = analyze_anchoring_bias(df_clean)
    
    # Create visualizations
    create_visualizations(df_clean, session_df)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Key finding: {'Significant anchoring bias' if p_value < 0.05 else 'No significant anchoring bias'}")
    print(f"Anchoring direction: {'Price-anchored' if beta_trade > beta_fundamental else 'Fundamental-anchored'}")
    print(f"Model fit: R² = {r_squared:.3f}")

if __name__ == "__main__":
    main()
