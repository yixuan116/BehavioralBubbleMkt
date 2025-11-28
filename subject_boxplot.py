import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH = 'Bubble_Markets_2025.xlsx'


def load_participant_level():
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


def plot_boxplot(participant_df: pd.DataFrame):
    sns.set_theme(style='whitegrid')
    sns.set_context('talk')

    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(data=participant_df, x='Choice', y='BubblePct',
                     showfliers=False, width=0.5, palette=['#4e79a7', '#f28e2b'])
    sns.stripplot(data=participant_df, x='Choice', y='BubblePct', jitter=True,
                  color='black', alpha=0.6, size=5)

    ax.set_title('Subject-Level Average Bubble%', fontsize=16, fontweight='bold')
    ax.set_xlabel('Choice (Market Type)', fontsize=13)
    ax.set_ylabel('Average Bubble %', fontsize=13)
    ax.set_ylim(10, 45)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    medians = participant_df.groupby('Choice')['BubblePct'].median()
    for choice, median_val in medians.items():
        xpos = 0 if choice == 'A' else 1
        ax.text(xpos, median_val + 0.5,
                f"Median {choice} = {median_val:.1f}%",
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkgreen')

    ax.annotate('Note: extreme values beyond 45% exist',
                xy=(0.5, 44.5), xycoords='data', ha='center', fontsize=11, color='gray')

    plt.tight_layout()
    output_path = 'subject_boxplot_bubble.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'Plot saved to {output_path}')


def main():
    participant_df = load_participant_level()
    plot_boxplot(participant_df)


if __name__ == '__main__':
    main()
