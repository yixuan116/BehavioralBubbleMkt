import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FILE_PATH = 'partB_sessions.csv'
RAW_CHOICE_B_FILE = 'Experiment_B_Trading_Data.csv'
FIG_DIR = 'figs'
SCATTER_OUTPUT = 'choiceB_scatter_absolute.png'
M = 10_000  # number of Monte Carlo permutations
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


def main():
    df = pd.read_csv(FILE_PATH)

    low = df[df['pro_share'].isin(PRO_LOW)]
    high = df[df['pro_share'].isin(PRO_HIGH)]

    x_low = low['Bubble'].mean()
    x_high = high['Bubble'].mean()
    t_obs = x_high - x_low

    print(f"X_low = {x_low:.4f}")
    print(f"X_high = {x_high:.4f}")
    print(f"Observed T = {t_obs:.4f}")

    pro_labels = df['pro_share'].values
    bubble_values = df['Bubble'].values

    t_null = np.zeros(M)
    for m in range(M):
        permuted = np.random.permutation(pro_labels)
        low_mask = np.isin(permuted, list(PRO_LOW))
        high_mask = np.isin(permuted, list(PRO_HIGH))
        t_null[m] = bubble_values[high_mask].mean() - bubble_values[low_mask].mean()

    p_value = np.mean(t_null <= t_obs)
    print(f"Monte Carlo p-value = {p_value:.4f}")

    session_summary = compute_choiceb_absolute_bubbles()
    low_abs_mean, high_abs_mean = plot_absolute_scatter(session_summary)
    print("\nChoice B absolute mispricing per session:")
    print(session_summary.to_string(index=False, formatters={'BubbleAbsMean': lambda v: f"{v:.2f}"}))
    print(f"\nLowPro mean (absolute) = {low_abs_mean:.2f}")
    print(f"HighPro mean (absolute) = {high_abs_mean:.2f}")

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(t_null, bins=40, alpha=0.7)
    plt.axvline(t_obs, color='red', linewidth=2, label='Observed T')
    plt.xlabel('T under Null')
    plt.ylabel('Frequency')
    plt.title('Monte Carlo Null Distribution of T')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.text(
        0.02,
        0.95,
        f"T_obs = {t_obs:.2f}\np-value = {p_value:.4f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        va='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )
    plt.tight_layout()
    null_path = os.path.join(FIG_DIR, 'mc_null_distribution.png')
    plt.savefig(null_path, dpi=300)
    plt.close()

    print(f'Saved figures to {FIG_DIR}/')


if __name__ == '__main__':
    main()
