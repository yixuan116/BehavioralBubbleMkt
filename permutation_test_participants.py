import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seed for reproducibility
np.random.seed(123)

FILE_PATH = 'Bubble_Markets_2025.xlsx'
N_PERM = 50000  # Number of permutations


def load_data():
    """Load participant-level bubble data from Excel."""
    choices_a = pd.read_excel(FILE_PATH, sheet_name='Choices_A').assign(Choice='A')
    choices_b = pd.read_excel(FILE_PATH, sheet_name='Choices_B').assign(Choice='B')
    
    df = pd.concat([choices_a, choices_b], ignore_index=True, sort=False)
    df['BubblePct'] = ((df['LastPrice'] - df['Fundamental']) / df['Fundamental']) * 100
    df = df.dropna(subset=['BubblePct', 'Choice', 'SubjectID', 'Session']).copy()
    
    df['ParticipantID'] = df.apply(
        lambda r: f"{r['Choice']}_S{int(r['Session'])}_ID{int(r['SubjectID'])}", axis=1
    )
    
    participant_df = (
        df.groupby(['ParticipantID', 'Choice'])['BubblePct']
        .mean()
        .reset_index()
    )
    
    return participant_df


def identify_columns(df):
    """Identify the group and bubble columns."""
    # Group column: "Choice" (values 'A' and 'B')
    group_col = "Choice"
    # Bubble column: "BubblePct" (participant-level average bubble%)
    bubble_col = "BubblePct"
    
    print(f"Using group column: '{group_col}'")
    print(f"Using bubble column: '{bubble_col}'")
    
    return group_col, bubble_col


def clean_and_subset(df, group_col, bubble_col):
    """Clean data and create arrays for A and B."""
    df_clean = df.copy()
    
    # Keep only A or B (case-insensitive)
    df_clean = df_clean[df_clean[group_col].isin(['A', 'B'])]
    
    # Drop missing values in bubble column
    df_clean = df_clean.dropna(subset=[bubble_col])
    
    # Convert to numeric if needed (should already be numeric)
    df_clean[bubble_col] = pd.to_numeric(df_clean[bubble_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[bubble_col])
    
    # Create arrays
    bubble_A = df_clean[df_clean[group_col] == 'A'][bubble_col].values
    bubble_B = df_clean[df_clean[group_col] == 'B'][bubble_col].values
    
    print(f"\nGroup A: n = {len(bubble_A)}")
    print(f"  Mean = {np.mean(bubble_A):.2f}%, Median = {np.median(bubble_A):.2f}%, "
          f"Std = {np.std(bubble_A):.2f}%, Min = {np.min(bubble_A):.2f}%, Max = {np.max(bubble_A):.2f}%")
    
    print(f"\nGroup B: n = {len(bubble_B)}")
    print(f"  Mean = {np.mean(bubble_B):.2f}%, Median = {np.median(bubble_B):.2f}%, "
          f"Std = {np.std(bubble_B):.2f}%, Min = {np.min(bubble_B):.2f}%, Max = {np.max(bubble_B):.2f}%")
    
    return bubble_A, bubble_B


def compute_observed_statistic(bubble_A, bubble_B):
    """Compute observed mean difference T_obs = mean(B) - mean(A)."""
    mean_A = np.mean(bubble_A)
    mean_B = np.mean(bubble_B)
    T_obs = mean_B - mean_A
    
    print(f"\nObserved statistic:")
    print(f"  Mean A = {mean_A:.2f}%")
    print(f"  Mean B = {mean_B:.2f}%")
    print(f"  T_obs = mean(B) - mean(A) = {T_obs:.2f}%")
    
    return T_obs, mean_A, mean_B


def run_permutation_test(bubble_A, bubble_B, n_perm=N_PERM):
    """Run permutation test."""
    # Combine data
    all_values = np.concatenate([bubble_A, bubble_B])
    n_A = len(bubble_A)
    n_B = len(bubble_B)
    
    print(f"\nRunning permutation test with {n_perm} permutations...")
    
    T_perm = np.zeros(n_perm)
    
    for i in range(n_perm):
        # Randomly permute
        permuted = np.random.permutation(all_values)
        perm_A = permuted[:n_A]
        perm_B = permuted[n_A:]
        
        # Compute statistic
        T_perm[i] = np.mean(perm_B) - np.mean(perm_A)
        
        if (i + 1) % 10000 == 0:
            print(f"  Completed {i + 1} permutations...")
    
    print(f"  Completed all {n_perm} permutations.")
    
    return T_perm


def compute_p_value(T_perm, T_obs):
    """Compute two-sided p-value."""
    p_perm = np.mean(np.abs(T_perm) >= np.abs(T_obs))
    return p_perm


def plot_null_distribution(T_perm, T_obs, p_perm, output_path):
    """Visualize permutation null distribution."""
    sns.set_theme(style='whitegrid')
    
    plt.figure(figsize=(10, 6))
    
    # Histogram of null distribution
    plt.hist(T_perm, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    
    # Vertical line for observed statistic
    plt.axvline(T_obs, color='red', linestyle='--', linewidth=2, label=f'Observed: {T_obs:.2f}%')
    plt.axvline(-T_obs, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Observed (symmetric): {-T_obs:.2f}%')
    
    plt.xlabel(r'Mean difference B − A under $H_0$ (permuted)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Permutation Test for A vs B (Participant-Level Bubble%)', fontsize=14, fontweight='bold')
    
    # Annotate
    textstr = f'T_obs = {T_obs:.2f}%\np_perm = {p_perm:.4f}\nN_perm = {len(T_perm):,}'
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    # Create figures directory if needed
    os.makedirs('figures', exist_ok=True)
    full_path = os.path.join('figures', output_path)
    plt.savefig(full_path, dpi=300)
    plt.close()
    
    print(f"\nFigure saved to: {full_path}")


def create_summary_table(bubble_A, bubble_B, T_obs, p_perm):
    """Create summary table."""
    summary_data = {
        'Group': ['A', 'B'],
        'N': [len(bubble_A), len(bubble_B)],
        'Mean': [np.mean(bubble_A), np.mean(bubble_B)],
        'Median': [np.median(bubble_A), np.median(bubble_B)],
        'Std': [np.std(bubble_A), np.std(bubble_B)],
        'Min': [np.min(bubble_A), np.min(bubble_B)],
        'Max': [np.max(bubble_A), np.max(bubble_B)]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add permutation test row
    test_row = pd.DataFrame({
        'Group': ['Permutation Test'],
        'N': [np.nan],
        'Mean': [np.nan],
        'Median': [np.nan],
        'Std': [np.nan],
        'Min': [np.nan],
        'Max': [np.nan]
    })
    
    # Create a separate row for test statistics
    test_stats = pd.DataFrame({
        'Statistic': ['T_obs (B - A)', 'p_perm (two-sided)'],
        'Value': [f"{T_obs:.2f}%", f"{p_perm:.4f}"]
    })
    
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(summary_df.round(2))
    print("\nPermutation Test Results:")
    print(test_stats)
    print("="*60)
    
    return summary_df, test_stats


def print_interpretation(T_obs, p_perm, n_perm):
    """Print interpretation of results."""
    alpha = 0.05
    reject = "reject" if p_perm < alpha else "do not reject"
    
    print("\n" + "="*60)
    print("Interpretation")
    print("="*60)
    print(f"Observed mean difference (B − A) in participant-level average bubble% "
          f"is {T_obs:.2f} percentage points.")
    print(f"Based on {n_perm:,} permutations, the two-sided permutation p-value is "
          f"p_perm = {p_perm:.4f}.")
    print(f"At alpha = {alpha}, we {reject} the null hypothesis of no treatment effect "
          f"on mean participant-level bubble%.")
    print("="*60)


def main():
    # 0. Setup
    print("="*60)
    print("Permutation Test: Participant-Level Bubble% (A vs B)")
    print("="*60)
    
    # 1. Load data
    print("\n1. Loading data...")
    df = load_data()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # 2. Identify columns
    print("\n2. Identifying columns...")
    group_col, bubble_col = identify_columns(df)
    
    # 3. Clean and subset
    print("\n3. Cleaning and subsetting data...")
    bubble_A, bubble_B = clean_and_subset(df, group_col, bubble_col)
    
    # 4. Compute observed statistic
    print("\n4. Computing observed statistic...")
    T_obs, mean_A, mean_B = compute_observed_statistic(bubble_A, bubble_B)
    
    # 5. Permutation test
    print("\n5. Running permutation test...")
    T_perm = run_permutation_test(bubble_A, bubble_B, n_perm=N_PERM)
    p_perm = compute_p_value(T_perm, T_obs)
    
    print(f"\nPermutation test results:")
    print(f"  T_obs = {T_obs:.2f}%")
    print(f"  p_perm (two-sided) = {p_perm:.4f}")
    print(f"  N_perm = {len(T_perm):,}")
    
    # 6. Visualization
    print("\n6. Creating visualization...")
    plot_null_distribution(T_perm, T_obs, p_perm, 
                          "permutation_null_distribution_A_vs_B_participants.png")
    
    # 7. Summary table
    print("\n7. Creating summary table...")
    summary_df, test_stats = create_summary_table(bubble_A, bubble_B, T_obs, p_perm)
    
    # 8. Interpretation
    print_interpretation(T_obs, p_perm, N_PERM)


if __name__ == '__main__':
    main()

