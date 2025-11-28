import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

FILE_PATH = 'Bubble_Markets_2025.xlsx'
ALPHA = 0.05
C_ALPHA = 1.36  # for alpha = 0.05


def load_bubble_series(sheet: str) -> pd.Series:
    df = pd.read_excel(FILE_PATH, sheet_name=sheet)
    if 'BubblePct' in df.columns:
        bubble = df['BubblePct']
    else:
        bubble = (df['LastPrice'] - df['Fundamental']) / df['Fundamental'] * 100
    bubble = bubble.dropna()
    return bubble


def empirical_cdf(series: pd.Series):
    sorted_vals = np.sort(series.values)
    cdf_vals = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    x = np.insert(sorted_vals, 0, sorted_vals[0])
    y = np.insert(cdf_vals, 0, 0.0)
    return x, y, sorted_vals


def compute_grid_cdfs(sorted_a: np.ndarray, sorted_b: np.ndarray):
    grid = np.sort(np.unique(np.concatenate([sorted_a, sorted_b])))
    fa = np.searchsorted(sorted_a, grid, side='right') / len(sorted_a)
    fb = np.searchsorted(sorted_b, grid, side='right') / len(sorted_b)
    return grid, fa, fb


def main():
    bubble_a = load_bubble_series('Choices_A')
    bubble_b = load_bubble_series('Choices_B')
    n_a, n_b = len(bubble_a), len(bubble_b)

    x_a, y_a, sorted_a = empirical_cdf(bubble_a)
    x_b, y_b, sorted_b = empirical_cdf(bubble_b)
    grid, fa_grid, fb_grid = compute_grid_cdfs(sorted_a, sorted_b)

    ks_result = ks_2samp(bubble_a, bubble_b)
    d_obs = ks_result.statistic
    p_value = ks_result.pvalue

    d_crit = C_ALPHA * np.sqrt((n_a + n_b) / (n_a * n_b))
    decision = 'Reject H₀' if d_obs > d_crit else 'Fail to reject H₀'

    diff = np.abs(fa_grid - fb_grid)
    idx_max = np.argmax(diff)
    x_d = grid[idx_max]
    y_low, y_high = sorted([fa_grid[idx_max], fb_grid[idx_max]])

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 6))

    # Plot ECDFs
    plt.step(x_a, y_a, where='post', label=f'Choice A (n={n_a})', linewidth=2)
    plt.step(x_b, y_b, where='post', label=f'Choice B (n={n_b})', linewidth=2)

    # Shade rejection region where |Fa-Fb| >= D_crit
    plt.fill_between(grid, fa_grid, fb_grid,
                     where=(diff >= d_crit), color='red', alpha=0.15,
                     label='|F_A - F_B| ≥ D_crit')

    # Vertical line for observed D
    plt.vlines(x_d, y_low, y_high, color='black', linestyle='--', linewidth=2,
               label=f'Observed D = {d_obs:.3f}')

    # Annotate observed D
    plt.text(x_d, (y_low + y_high) / 2,
             f"D_obs = {d_obs:.3f}", fontsize=11,
             ha='left', va='bottom', color='black')

    # Annotate critical value band on left margin
    base_y = 0.05
    plt.annotate('', xy=(plt.xlim()[0], base_y + d_crit), xytext=(plt.xlim()[0], base_y),
                 arrowprops=dict(arrowstyle='<->', color='gray', linewidth=1.5))
    plt.text(plt.xlim()[0], base_y + d_crit / 2,
             f"D_crit = {d_crit:.3f}", color='gray', ha='left', va='center', fontsize=11)

    plt.title('Kolmogorov–Smirnov Test: Bubble% CDFs (Choice A vs Choice B)', fontsize=14, fontweight='bold')
    plt.xlabel('Bubble %')
    plt.ylabel('Empirical CDF')
    plt.ylim(-0.02, 1.02)
    plt.legend(loc='lower right')

    decision_text = (
        f"Observed D = {d_obs:.3f}\n"
        f"Critical D_crit = {d_crit:.3f}\n"
        f"p-value = {p_value:.4f}\n"
        f"Decision: {decision}"
    )
    plt.gca().text(0.02, 0.65, decision_text, transform=plt.gca().transAxes,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    output_path = 'bubble_ks_ecdf.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f'Sample sizes: Choice A = {n_a}, Choice B = {n_b}')
    print(f'Observed D = {d_obs:.4f}, p-value = {p_value:.4f}')
    print(f'Critical D_crit = {d_crit:.4f}')
    print(f'Decision: {decision}')
    print(f'Plot saved to {output_path}')


if __name__ == '__main__':
    main()
