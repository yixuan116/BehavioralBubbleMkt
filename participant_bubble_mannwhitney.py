import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu


def load_participant_level(file_path: str) -> pd.DataFrame:
    """Return participant-level average Bubble% for Choices A and B."""
    choices_a = pd.read_excel(file_path, sheet_name='Choices_A').assign(Choice='A')
    choices_b = pd.read_excel(file_path, sheet_name='Choices_B').assign(Choice='B')

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


def summarize(participant_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        participant_df.groupby('Choice')['BubblePct']
        .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        .rename(columns={
            'count': 'N',
            'mean': 'Mean',
            'median': 'Median',
            'std': 'Std',
            'min': 'Min',
            'max': 'Max',
        })
    )
    return summary


def run_mannwhitney(participant_df: pd.DataFrame):
    bubble_a = participant_df[participant_df['Choice'] == 'A']['BubblePct']
    bubble_b = participant_df[participant_df['Choice'] == 'B']['BubblePct']

    u_two, p_two = mannwhitneyu(bubble_a, bubble_b, alternative='two-sided')
    u_greater, p_greater = mannwhitneyu(bubble_a, bubble_b, alternative='greater')
    return (u_two, p_two, u_greater, p_greater)


def plot_participant_bubbles(participant_df: pd.DataFrame, summary: pd.DataFrame,
                              stats: tuple, combo_output_path: str,
                              violin_output_path: str):
    sns.set_theme(style='whitegrid')
    sns.set_palette(['#4e79a7', '#f28e2b'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sns.violinplot(data=participant_df, x='Choice', y='BubblePct', inner='box',
                   cut=0, ax=ax)
    sns.stripplot(data=participant_df, x='Choice', y='BubblePct',
                  dodge=False, color='black', alpha=0.6, size=4, ax=ax)
    ax.set_title('Participant-Level Average Bubble%', fontsize=14, fontweight='bold')
    ax.set_xlabel('Choice (Market Type)')
    ax.set_ylabel('Average Bubble%')

    # annotate medians near the mid-40% band for readability
    desired_y = 45
    y_min, y_max = ax.get_ylim()
    y_axes = (desired_y - y_min) / (y_max - y_min) if y_max != y_min else 0.9
    y_axes = min(max(y_axes, 0.02), 0.98)
    medians = participant_df.groupby('Choice')['BubblePct'].median()
    for choice, median_value in medians.items():
        text_x = 0.02 if choice == 'A' else 0.72
        ax.text(
            text_x, y_axes,
            f"Median {choice}: {median_value:.1f}%",
            transform=ax.transAxes,
            ha='left', va='bottom', fontsize=11, fontweight='bold',
            color='darkgreen',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2)
        )

    # Table on the right
    ax_table = axes[1]
    ax_table.axis('off')

    table_df = summary.reset_index()
    table_df['N'] = table_df['N'].astype(int).astype(str)
    for col in ['Mean', 'Median', 'Std', 'Min', 'Max']:
        table_df[col] = table_df[col].apply(lambda v: f"{v:.1f}%")

    u_two, p_two, _, p_greater = stats
    table_df.loc[len(table_df)] = [
        'Mann-Whitney',
        f"U={u_two:.1f}",
        f"p(two)={p_two:.4f}",
        f"p(A>B)={p_greater:.4f}",
        '',
        '',
        ''
    ]

    table = ax_table.table(
        cellText=table_df.values,
        colLabels=['Group', 'N / U', 'Mean', 'Median', 'Std', 'Min', 'Max'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax_table.set_title('Summary Statistics & Mann-Whitney Test', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(combo_output_path, dpi=300)
    plt.close(fig)

    # Left-only violin/strip plot for PPT use
    fig_violin, ax_violin = plt.subplots(figsize=(7, 6))
    sns.violinplot(data=participant_df, x='Choice', y='BubblePct', inner='box',
                   cut=0, ax=ax_violin)
    sns.stripplot(data=participant_df, x='Choice', y='BubblePct',
                  dodge=False, color='black', alpha=0.6, size=4, ax=ax_violin)
    ax_violin.set_title('Participant-Level Average Bubble%', fontsize=14, fontweight='bold')
    ax_violin.set_xlabel('Choice (Market Type)')
    ax_violin.set_ylabel('Average Bubble%')

    desired_y = 45
    y_min_v, y_max_v = ax_violin.get_ylim()
    y_axes_v = (desired_y - y_min_v) / (y_max_v - y_min_v) if y_max_v != y_min_v else 0.9
    y_axes_v = min(max(y_axes_v, 0.02), 0.98)
    medians = participant_df.groupby('Choice')['BubblePct'].median()
    for choice, median_value in medians.items():
        text_x = 0.02 if choice == 'A' else 0.72
        ax_violin.text(
            text_x, y_axes_v,
            f"Median {choice}: {median_value:.1f}%",
            transform=ax_violin.transAxes,
            ha='left', va='bottom', fontsize=11, fontweight='bold',
            color='darkgreen',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2)
        )

    plt.tight_layout()
    plt.savefig(violin_output_path, dpi=300)
    plt.close(fig_violin)


def main():
    file_path = 'Bubble_Markets_2025.xlsx'
    participant_df = load_participant_level(file_path)
    summary = summarize(participant_df)
    stats = run_mannwhitney(participant_df)

    combo_output = 'participant_bubble_distribution.png'
    violin_output = 'participant_bubble_distribution_left.png'
    plot_participant_bubbles(participant_df, summary, stats, combo_output, violin_output)

    bubble_a = participant_df[participant_df['Choice'] == 'A']['BubblePct']
    bubble_b = participant_df[participant_df['Choice'] == 'B']['BubblePct']

    print('Subjects in Choice A:', len(bubble_a))
    print('Subjects in Choice B:', len(bubble_b))
    print('\nChoice A: mean = {:.2f}%, median = {:.2f}%'.format(bubble_a.mean(), bubble_a.median()))
    print('Choice B: mean = {:.2f}%, median = {:.2f}%'.format(bubble_b.mean(), bubble_b.median()))

    u_two, p_two, u_greater, p_greater = stats
    print('\nMann-Whitney U (two-sided): U = {:.2f}, p = {:.4f}'.format(u_two, p_two))
    print('Mann-Whitney U (one-sided, H1: A > B): U = {:.2f}, p = {:.4f}'.format(u_greater, p_greater))

    print('\nSummary: Median Bubble% -> A = {:.1f}%, B = {:.1f}%. Mann-Whitney p(two-sided) = {:.4f}, p(one-sided A>B) = {:.4f}'.format(
        bubble_a.median(), bubble_b.median(), p_two, p_greater))
    print(f"Combo chart saved to {combo_output}")
    print(f"Left-only chart saved to {violin_output}")


if __name__ == '__main__':
    main()
