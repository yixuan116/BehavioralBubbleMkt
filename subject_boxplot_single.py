import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH = 'Bubble_Markets_2025.xlsx'
COLOR_MAP = {'A': '#4e79a7', 'B': '#f28e2b'}
LABEL_MAP = {'A': 'Choice A (Students)', 'B': 'Choice B (Student-Pro Mix)'}


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


def plot_single_choice(participant_df: pd.DataFrame, choice: str):
    data = participant_df[participant_df['Choice'] == choice].copy()
    data['Label'] = LABEL_MAP[choice]

    sns.set_theme(style='whitegrid', context='talk')
    plt.figure(figsize=(7, 6))

    ax = sns.boxplot(data=data, x='Label', y='BubblePct', showfliers=False,
                     width=0.4, color=COLOR_MAP[choice])
    sns.stripplot(data=data, x='Label', y='BubblePct', jitter=True,
                  color='black', alpha=0.7, size=5)

    ax.set_title(f'Subject-Level Bubble% – {LABEL_MAP[choice]}', fontsize=16, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Average Bubble %')
    ax.set_ylim(10, 45)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    median_val = data['BubblePct'].median()
    ax.text(0, median_val + 0.5,
            f"Median {choice} = {median_val:.1f}%",
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkgreen')

    ax.annotate('Note: extreme values beyond 45% exist',
                xy=(0, 44.5), xytext=(0, 44.5), textcoords='data',
                ha='center', fontsize=11, color='gray')

    plt.tight_layout()
    output_path = f'subject_boxplot_{choice}.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'Saved {output_path}')


def main():
    participant_df = load_participant_level()
    for choice in ['A', 'B']:
        plot_single_choice(participant_df, choice)


if __name__ == '__main__':
    main()
