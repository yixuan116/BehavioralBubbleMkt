"""
================================================================================
Google Colab - Ready to Run Version
================================================================================
This script automatically:
1. Installs required dependencies
2. Prompts for data file uploads
3. Generates two figures

Generated files:
- figs/mc_null_distribution.png
- choiceB_scatter_absolute.png
================================================================================
"""

# ============================================================================
# Auto-install dependencies (Colab environment)
# ============================================================================
import subprocess
import sys

def install_package(package):
    """Install Python package if not already installed."""
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully")

# Detect if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("Colab environment detected. Installing dependencies...")
    install_package('numpy')
    install_package('pandas')
    install_package('matplotlib')
    print("All dependencies installed!\n")
except ImportError:
    IN_COLAB = False
    print("Local environment detected. Skipping auto-install.\n")

# ============================================================================
# Import libraries
# ============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib

# For Colab, no need to set 'Agg' backend
if not IN_COLAB:
    try:
        matplotlib.use('Agg')  # Non-interactive backend for servers
    except:
        pass

import matplotlib.pyplot as plt

# ============================================================================
# File upload (Colab environment only)
# ============================================================================
if IN_COLAB:
    from google.colab import files
    
    def check_and_upload_file(filename, description):
        """Check if file exists, prompt for upload if not."""
        if os.path.exists(filename):
            print(f"✓ Found file: {filename}")
            return True
        else:
            print(f"\nPlease upload file: {filename}")
            print(f"Description: {description}")
            uploaded = files.upload()
            if filename in uploaded:
                print(f"✓ {filename} uploaded successfully!\n")
                return True
            else:
                print(f"✗ {filename} not found. Please upload again.\n")
                return False
    
    print("=" * 60)
    print("Checking data files...")
    print("=" * 60)
    
    if not check_and_upload_file('partB_sessions.csv', 'Session data file'):
        raise FileNotFoundError("Required file missing: partB_sessions.csv")
    
    if not check_and_upload_file('Experiment_B_Trading_Data.csv', 'Trading data file'):
        raise FileNotFoundError("Required file missing: Experiment_B_Trading_Data.csv")
    
    print("=" * 60)
    print("All files ready. Starting figure generation...")
    print("=" * 60)
    print()

FILE_PATH = 'partB_sessions.csv'
RAW_CHOICE_B_FILE = 'Experiment_B_Trading_Data.csv'
FIG_DIR = 'figs'
SCATTER_OUTPUT = 'choiceB_scatter_absolute.png'
PRO_LOW = {0.25, 0.50}
PRO_HIGH = {0.75, 1.00}


def compute_choiceb_absolute_bubbles():
    """Return session-level absolute mispricing for Choice B."""
    raw = pd.read_csv(RAW_CHOICE_B_FILE)
    period_prices = (
        raw.groupby(['Session', 'Period'], as_index=False)
        .agg({
            'LastPrice': 'last',
            'Fundamental': 'last',
            'ProShare': 'last'
        })
    )
    period_prices['BubbleAbs'] = (period_prices['LastPrice'] - period_prices['Fundamental']).abs()
    session_summary = (
        period_prices.groupby('Session', as_index=False)
        .agg({
            'BubbleAbs': 'mean',
            'ProShare': 'first'
        })
        .rename(columns={'BubbleAbs': 'BubbleAbsMean'})
        .sort_values('Session')
    )
    return session_summary


def plot_absolute_scatter(session_summary):
    """Plot absolute mispricing vs. pro share and return low/high means."""
    low_mask = session_summary['ProShare'].isin(PRO_LOW)
    high_mask = session_summary['ProShare'].isin(PRO_HIGH)
    low_mean = session_summary.loc[low_mask, 'BubbleAbsMean'].mean()
    high_mean = session_summary.loc[high_mask, 'BubbleAbsMean'].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(session_summary['ProShare'], session_summary['BubbleAbsMean'], s=80, color='tab:blue')

    for _, row in session_summary.iterrows():
        y_offset = -4
        x_offset = 0.02
        if row['BubbleAbsMean'] < 30:
            y_offset = -2
        if 30 <= row['BubbleAbsMean'] < 33:
            y_offset = -6
        if row['ProShare'] >= 0.9:
            x_offset = -0.11
            y_offset = 6 if row['BubbleAbsMean'] > 25 else -10
        if row['ProShare'] == 1.0 and np.isclose(row['BubbleAbsMean'], 26.50, atol=0.05):
            x_offset = -0.02
        if row['ProShare'] == 1.0 and np.isclose(row['BubbleAbsMean'], 24.01, atol=0.05):
            y_offset = -4
        ax.text(
            row['ProShare'] + x_offset,
            row['BubbleAbsMean'] + y_offset,
            f"{row['BubbleAbsMean']:.1f}",
            fontsize=9,
            ha='left',
            va='bottom'
        )

    ax.axhline(low_mean, color='tab:blue', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.axhline(high_mean, color='tab:orange', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.text(
        0.35,
        0.92,
        f"LowPro mean = {low_mean:.1f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    ax.text(
        0.55,
        0.18,
        f"HighPro mean = {high_mean:.1f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    ax.set_xlabel('Professional Share')
    ax.set_ylabel('Bubble Size (Absolute Mispricing)')
    ax.set_title('Bubble Size vs Professional Share (Absolute Mispricing)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SCATTER_OUTPUT, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return low_mean, high_mean


def run_low_high_mc_test(df: pd.DataFrame, n_perm: int = 50_000) -> dict:
    """
    Monte Carlo permutation test comparing LOW_PRO (25%+50%) vs
    HIGH_PRO (75%+100%).

    Returns:
        {
            "T_obs": ...,
            "p_value": ...,
            "mean_low": ...,
            "mean_high": ...,
            "n_perm": ...,
            "t_null": ...  # null distribution for plotting
        }
    """
    low = df[df['pro_share'].isin(PRO_LOW)]
    high = df[df['pro_share'].isin(PRO_HIGH)]

    mean_low = low['Bubble'].mean()
    mean_high = high['Bubble'].mean()
    T_obs = mean_high - mean_low

    pro_labels = df['pro_share'].values
    bubble_values = df['Bubble'].values

    t_null = np.zeros(n_perm)
    for m in range(n_perm):
        permuted = np.random.permutation(pro_labels)
        low_mask = np.isin(permuted, list(PRO_LOW))
        high_mask = np.isin(permuted, list(PRO_HIGH))
        t_null[m] = bubble_values[high_mask].mean() - bubble_values[low_mask].mean()

    p_value = np.mean(t_null <= T_obs)

    return {
        "T_obs": T_obs,
        "p_value": p_value,
        "mean_low": mean_low,
        "mean_high": mean_high,
        "n_perm": n_perm,
        "t_null": t_null
    }


def main():
    """Generate two figures:
    1. Monte Carlo Null Distribution of T (LOW_PRO vs HIGH_PRO)
    2. Bubble Size vs Professional Share (scatter plot)
    """
    # Load data
    df = pd.read_csv(FILE_PATH)
    
    # ============================================================================
    # Figure 1: Monte Carlo Null Distribution
    # ============================================================================
    print("Generating Monte Carlo Null Distribution plot...")
    low_high_result = run_low_high_mc_test(df, n_perm=50_000)
    
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(low_high_result['t_null'], bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(low_high_result['T_obs'], color='red', linewidth=2.5, 
               label=f'Observed T = {low_high_result["T_obs"]:.2f}')
    ax.set_xlabel('T under Null')
    ax.set_ylabel('Frequency')
    ax.set_title('Monte Carlo Null Distribution of T')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(
        0.02,
        0.95,
        f"T_obs = {low_high_result['T_obs']:.2f}\np-value = {low_high_result['p_value']:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        va='top',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', linewidth=1)
    )
    plt.tight_layout()
    null_path = os.path.join(FIG_DIR, 'mc_null_distribution.png')
    plt.savefig(null_path, dpi=300)
    
    # Display image in Colab
    if IN_COLAB:
        from IPython.display import Image, display
        display(Image(null_path))
    else:
        plt.close()
    
    print(f"✓ Saved: {null_path}")
    
    # ============================================================================
    # Figure 2: Bubble Size vs Professional Share (Scatter Plot)
    # ============================================================================
    print("\nGenerating Bubble Size vs Professional Share scatter plot...")
    session_summary = compute_choiceb_absolute_bubbles()
    plot_absolute_scatter(session_summary)
    
    # Display image in Colab
    if IN_COLAB:
        from IPython.display import Image, display
        display(Image(SCATTER_OUTPUT))
    else:
        plt.close()
    
    print(f"✓ Saved: {SCATTER_OUTPUT}")
    
    print("\n" + "=" * 60)
    print("Done! Both figures have been generated.")
    print("=" * 60)


if __name__ == '__main__':
    main()

