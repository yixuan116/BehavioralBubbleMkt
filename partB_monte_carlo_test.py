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


def run_pairwise_composition_tests(
    df: pd.DataFrame,
    pro_levels=(0.25, 0.50, 0.75, 1.00),
    n_perm: int = 50_000
) -> pd.DataFrame:
    """
    Monte Carlo permutation tests for all pairwise comparisons
    between professional-share levels.

    For each ordered pair (p_low, p_high) with p_low < p_high:
        - Extract the 2 bubble_abs values at p_low and the 2 at p_high.
        - Compute the observed statistic:
              T_obs = mean(bubble_abs_high) - mean(bubble_abs_low)
          (We expect T_obs < 0 if higher professional share reduces bubbles.)
        - Under H0: F_{p_low} = F_{p_high} (no composition effect),
          generate a permutation-based null distribution:
              * pool the 4 bubble values
              * for r = 1..n_perm:
                  - randomly assign 2 values to "high" and 2 to "low"
                  - compute T_r = mean(high) - mean(low)
          - Compute the one-sided p-value:
              p_value = (# of T_r <= T_obs) / n_perm

    Return a DataFrame with columns:
        pro_low, pro_high, T_obs, p_value, mean_low, mean_high, n_perm, t_null
    sorted by (pro_low, pro_high).
    """
    results = []

    for i, p_low in enumerate(pro_levels):
        for p_high in pro_levels[i+1:]:
            # Extract values for this pair
            low_data = df[df['pro_share'] == p_low]['Bubble'].values
            high_data = df[df['pro_share'] == p_high]['Bubble'].values

            mean_low = low_data.mean()
            mean_high = high_data.mean()
            T_obs = mean_high - mean_low

            # Pool all 4 values
            pooled = np.concatenate([low_data, high_data])

            # Generate null distribution
            t_null = np.zeros(n_perm)
            for r in range(n_perm):
                # Randomly assign 2 to "high" and 2 to "low"
                permuted = np.random.permutation(pooled)
                t_null[r] = permuted[:2].mean() - permuted[2:].mean()

            p_value = np.mean(t_null <= T_obs)

            results.append({
                'pro_low': p_low,
                'pro_high': p_high,
                'T_obs': T_obs,
                'p_value': p_value,
                'mean_low': mean_low,
                'mean_high': mean_high,
                'n_perm': n_perm,
                't_null': t_null
            })

    return pd.DataFrame(results).sort_values(['pro_low', 'pro_high'])


def plot_pairwise_null_distribution(
    t_null: np.ndarray,
    T_obs: float,
    p_value: float,
    pro_low: float,
    pro_high: float,
    output_path: str
):
    """
    Plot null distribution for a single pairwise comparison.
    Matches the style of the existing 25 vs 75 figure with 6 spikes.
    """
    # Plot - use enough bins to show the 6 spikes clearly
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(t_null, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(T_obs, color='red', linewidth=2.5, label=f'Observed T = {T_obs:.2f}')
    ax.set_xlabel('T = mean(bubble_high) - mean(bubble_low)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Monte Carlo Permutation Null Distribution ({int(pro_low*100)}% vs {int(pro_high*100)}%)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(
        0.02,
        0.95,
        f"T_obs = {T_obs:.2f}\np-value = {p_value:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        va='top',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', linewidth=1)
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_all_pairwise_null_distributions(pairwise_results: pd.DataFrame, output_path: str = 'mc_null_all_pairs.png'):
    """
    Create a combined figure with all 6 pairwise null distributions in subplots.
    """
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Sort to ensure consistent order
    pairwise_results = pairwise_results.sort_values(['pro_low', 'pro_high'])
    
    for idx, (_, row) in enumerate(pairwise_results.iterrows()):
        ax = axes[idx]
        pro_low = row['pro_low']
        pro_high = row['pro_high']
        t_null = row['t_null']
        T_obs = row['T_obs']
        p_value = row['p_value']
        
        # Plot histogram
        ax.hist(t_null, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(T_obs, color='red', linewidth=2.5, label=f'T_obs = {T_obs:.2f}')
        ax.set_xlabel('T = mean(bubble_high) - mean(bubble_low)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{int(pro_low*100)}% vs {int(pro_high*100)}%', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        
        # Add text box with statistics
        ax.text(
            0.02,
            0.95,
            f"T_obs = {T_obs:.2f}\np = {p_value:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            va='top',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', linewidth=1)
        )
    
    # Add overall title
    fig.suptitle('Monte Carlo Permutation Null Distributions: All Pairwise Comparisons', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    df = pd.read_csv(FILE_PATH)

    # ============================================================================
    # (A) Purpose
    # ============================================================================
    print("=" * 80)
    print("Purpose: Test the Professional Composition Effect – whether higher")
    print("professional share reduces session-level bubble magnitude.")
    print("=" * 80)
    print()

    # ============================================================================
    # (B) LOW_PRO vs HIGH_PRO Test
    # ============================================================================
    print("-" * 80)
    print("(B) LOW_PRO vs HIGH_PRO Monte Carlo Permutation Test")
    print("-" * 80)
    print()
    print("Hypotheses:")
    print("  H0: F_LowPro = F_HighPro")
    print("  H1: F_HighPro < F_LowPro")
    print("  Test statistic: T = mean(Bubble_abs_HighPro) - mean(Bubble_abs_LowPro)")
    print()

    low_high_result = run_low_high_mc_test(df, n_perm=50_000)

    print("Results:")
    print(f"  mean_low (LOW_PRO: 25% + 50%) = {low_high_result['mean_low']:.4f}")
    print(f"  mean_high (HIGH_PRO: 75% + 100%) = {low_high_result['mean_high']:.4f}")
    print(f"  T_obs = {low_high_result['T_obs']:.4f}")
    print(f"  p_value = {low_high_result['p_value']:.4f}")
    print(f"  n_perm = {low_high_result['n_perm']:,}")
    print()

    alpha = 0.05
    decision = "Reject H0" if low_high_result['p_value'] < alpha else "Fail to reject H0"
    print(f"Decision at α = {alpha}:")
    print(f"  p = {low_high_result['p_value']:.4f} {'<' if low_high_result['p_value'] < alpha else '>='} {alpha} → {decision} at the 5% level.")
    print()

    # Plot null distribution for LOW_PRO vs HIGH_PRO
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(low_high_result['t_null'], bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(low_high_result['T_obs'], color='red', linewidth=2.5, label=f'Observed T = {low_high_result["T_obs"]:.2f}')
    ax.set_xlabel('T = mean(bubble_high) - mean(bubble_low)')
    ax.set_ylabel('Frequency')
    ax.set_title('Monte Carlo Permutation Null Distribution (LOW_PRO vs HIGH_PRO)')
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
    plt.close()

    # ============================================================================
    # (C) Pairwise Tests
    # ============================================================================
    print("-" * 80)
    print("(C) Pairwise Composition Tests")
    print("-" * 80)
    print()
    print("Hypotheses for each pair (p_low, p_high):")
    print("  H0: F_{p_low} = F_{p_high}")
    print("  H1: F_{p_high} < F_{p_low}")
    print("  T = mean_high - mean_low")
    print()

    pairwise_results = run_pairwise_composition_tests(df, n_perm=50_000)

    # ============================================================================
    # Generate all pairwise null distribution plots
    # ============================================================================
    print("-" * 80)
    print("Generating null distribution plots for all pairwise comparisons")
    print("-" * 80)
    
    for _, row in pairwise_results.iterrows():
        p_low = row['pro_low']
        p_high = row['pro_high']
        p_low_pct = int(p_low * 100)
        p_high_pct = int(p_high * 100)
        
        # Generate plot for this pair
        output_path = f'mc_null_{p_low_pct}vs{p_high_pct}.png'
        plot_pairwise_null_distribution(
            t_null=row['t_null'],
            T_obs=row['T_obs'],
            p_value=row['p_value'],
            pro_low=p_low,
            pro_high=p_high,
            output_path=output_path
        )
        print(f"  Saved: {output_path}")
    
    print()

    # ============================================================================
    # Representative figure: 25% vs 75%
    # ============================================================================
    print("-" * 80)
    print("Generating representative null distribution plot: 25% vs 75%")
    print("-" * 80)
    
    # Find the 25 vs 75 row
    row_25vs75 = pairwise_results[
        (pairwise_results['pro_low'] == 0.25) & 
        (pairwise_results['pro_high'] == 0.75)
    ].iloc[0]
    
    plot_pairwise_null_distribution(
        t_null=row_25vs75['t_null'],
        T_obs=row_25vs75['T_obs'],
        p_value=row_25vs75['p_value'],
        pro_low=0.25,
        pro_high=0.75,
        output_path='mc_null_representative.png'
    )
    print(f"  Saved: mc_null_representative.png")
    print()

    # ============================================================================
    # Combined figure: All pairwise comparisons
    # ============================================================================
    print("-" * 80)
    print("Generating combined figure with all pairwise null distributions")
    print("-" * 80)
    plot_all_pairwise_null_distributions(pairwise_results, output_path='mc_null_all_pairs.png')
    print(f"  Saved: mc_null_all_pairs.png")
    print()

    # ============================================================================
    # Summary Table
    # ============================================================================
    print("-" * 80)
    print("Summary Table: Pairwise Composition Tests")
    print("-" * 80)
    print()
    
    # Create summary table without t_null column
    summary_cols = ['pro_low', 'pro_high', 'mean_low', 'mean_high', 'T_obs', 'p_value']
    summary_df = pairwise_results[summary_cols].copy()
    
    # Format for display
    print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print()
    
    print("Pairwise Test Results (detailed):")
    print()
    for _, row in pairwise_results.iterrows():
        p_low_pct = int(row['pro_low'] * 100)
        p_high_pct = int(row['pro_high'] * 100)
        decision = "Reject H0 at 5% level" if row['p_value'] < 0.05 else "Fail to reject H0"
        print(f"  {p_low_pct}% vs {p_high_pct}%: mean_low = {row['mean_low']:.2f}, "
              f"mean_high = {row['mean_high']:.2f}, "
              f"T_obs = {row['T_obs']:.2f}, "
              f"p = {row['p_value']:.4f} ({decision})")
    print()

    # ============================================================================
    # Original scatter plot (keep existing functionality)
    # ============================================================================
    session_summary = compute_choiceb_absolute_bubbles()
    low_abs_mean, high_abs_mean = plot_absolute_scatter(session_summary)
    print("\nChoice B absolute mispricing per session:")
    print(session_summary.to_string(index=False, formatters={'BubbleAbsMean': lambda v: f"{v:.2f}"}))
    print(f"\nLowPro mean (absolute) = {low_abs_mean:.2f}")
    print(f"HighPro mean (absolute) = {high_abs_mean:.2f}")
    print()

    # ============================================================================
    # (D) Pros and Cons
    # ============================================================================
    print("=" * 80)
    print("Pros and Cons of the Monte Carlo Permutation Test")
    print("=" * 80)
    print()
    print("Pros:")
    print("  - Correct unit of analysis (session-level, n=8).")
    print("  - Distribution-free (no normality or equal-variance assumptions).")
    print("  - Works with very small n (2 sessions per group).")
    print("  - Directly implements the randomization logic of the experiment.")
    print()
    print("Cons:")
    print("  - Limited power with only 2 sessions per cell.")
    print("  - Results are sensitive to outliers at the session level.")
    print("  - Pairwise tests involve multiple comparisons (no adjustment here).")
    print("  - Only tests mean differences; cannot detect changes in higher moments.")
    print()
    print("=" * 80)
    print(f'Saved figures to {FIG_DIR}/')
    print("=" * 80)


if __name__ == '__main__':
    main()
