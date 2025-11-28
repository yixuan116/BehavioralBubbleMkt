import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

FILE_PATH = 'Bubble_Markets_2025.xlsx'
COLORS = {'A': '#4c72b0', 'B': '#55a868'}
LABELS = {'A': 'Choice A (students)', 'B': 'Choice B (mixed)'}


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
    participant_df['ChoiceLabel'] = participant_df['Choice'].map(LABELS)
    return participant_df


def plot_raincloud(participant_df):
    medians = (
        participant_df.groupby('ChoiceLabel')['BubblePct']
        .median()
        .round(1)
    )

    sns.set_theme(style='whitegrid')
    sns.set_context('talk')
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax = plt.subplots(figsize=(10, 6))

    # Half violin (left/right)
    sns.violinplot(
        data=participant_df,
        x='ChoiceLabel',
        y='BubblePct',
        palette=[COLORS['A'], COLORS['B']],
        cut=0,
        linewidth=0,
        inner=None,
        ax=ax
    )
    for i, artist in enumerate(ax.collections):
        # Set alpha and clip to half
        artist.set_alpha(0.5)
        path = artist.get_paths()[0]
        vertices = path.vertices
        x = vertices[:, 0]
        y = vertices[:, 1]
        center = i // 2  # violin index repeated for mirrored sides
        if i % 2 == 0:
            x = np.clip(x, center, center)
        else:
            x = np.clip(x, center, np.max(x))
        vertices[:, 0] = x

    # Boxplot overlay
    sns.boxplot(
        data=participant_df,
        x='ChoiceLabel',
        y='BubblePct',
        width=0.3,
        boxprops={'facecolor': 'white', 'zorder': 3},
        medianprops={'color': 'black', 'linewidth': 2},
        whiskerprops={'color': 'black'},
        capprops={'color': 'black'},
        showfliers=False,
        ax=ax
    )

    # Jittered points
    sns.stripplot(
        data=participant_df,
        x='ChoiceLabel',
        y='BubblePct',
        hue='ChoiceLabel',
        palette=[COLORS['A'], COLORS['B']],
        dodge=False,
        marker='o',
        alpha=0.7,
        size=5,
        jitter=0.2,
        edgecolor='none',
        ax=ax
    )
    ax.legend([], [], frameon=False)

    ax.set_title('Participant-Level Average Bubble%', fontsize=18, fontweight='bold')
    ax.set_xlabel('Market Type (A = students, B = mixed)', fontsize=13)
    ax.set_ylabel('Average Bubble (%)', fontsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.2)
    ax.set_facecolor('white')

    label_to_x = {
        'Choice A (students)': 0.22,
        'Choice B (mixed)': 0.78
    }
    label_to_short = {'Choice A (students)': 'A', 'Choice B (mixed)': 'B'}
    for label, value in medians.items():
        ax.text(
            label_to_x[label],
            1.02,
            f"Median {label_to_short[label]} = {value:.1f}%",
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            color='darkgreen',
            clip_on=False
        )

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    output_path = 'figures/bubble_raincloud.png'
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main():
    participant_df = load_participant_level()
    output_path = plot_raincloud(participant_df)
    print(f'Saved plot to {output_path}')


if __name__ == '__main__':
    main()
