import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_process_data():
    file_path = 'Bubble_Markets_2025.xlsx'
    
    # Load data
    df_a = pd.read_excel(file_path, sheet_name='Choices_A')
    df_b = pd.read_excel(file_path, sheet_name='Choices_B')
    
    # Calculate Bubble %
    # Filter out rows where Fundamental is 0 to avoid division by zero if necessary, 
    # though usually Fundamental is > 0 until after the experiment or handled.
    # Based on previous scripts, we just calculate it.
    
    df_a = df_a[df_a['Fundamental'] > 0].copy()
    df_b = df_b[df_b['Fundamental'] > 0].copy()
    
    df_a['BubblePct'] = ((df_a['LastPrice'] - df_a['Fundamental']) / df_a['Fundamental']) * 100
    df_b['BubblePct'] = ((df_b['LastPrice'] - df_b['Fundamental']) / df_b['Fundamental']) * 100
    
    # Drop NaNs in BubblePct (e.g. if LastPrice was NaN)
    df_a = df_a.dropna(subset=['BubblePct'])
    df_b = df_b.dropna(subset=['BubblePct'])
    
    return df_a, df_b

def plot_distribution(df_a, df_b):
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Plot A
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_a, x='BubblePct', color='blue', label='Choice A', kde=True, stat="density", alpha=0.4)
    
    mean_a = df_a['BubblePct'].mean()
    median_a = df_a['BubblePct'].median()
    std_a = df_a['BubblePct'].std()
    min_a = df_a['BubblePct'].min()
    max_a = df_a['BubblePct'].max()
    count_a = len(df_a)
    
    # Mean line and text
    plt.axvline(mean_a, color='darkblue', linestyle='--', linewidth=2, alpha=0.8)
    plt.text(mean_a + 2, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_a:.1f}%', color='darkblue', fontweight='bold', rotation=0)
    
    # Median line and text
    plt.axvline(median_a, color='green', linestyle=':', linewidth=2, alpha=0.8)
    plt.text(median_a - 2, plt.gca().get_ylim()[1] * 0.8, f'Median: {median_a:.1f}%', color='green', fontweight='bold', rotation=0, ha='right')
    
    # Min/Max markers on X-axis
    plt.plot(min_a, 0, marker='^', color='red', markersize=10)
    plt.text(min_a, plt.gca().get_ylim()[1] * 0.02, f'Min: {min_a:.1f}%', color='red', ha='center', va='bottom', fontsize=10)
    
    plt.plot(max_a, 0, marker='^', color='red', markersize=10)
    plt.text(max_a, plt.gca().get_ylim()[1] * 0.02, f'Max: {max_a:.1f}%', color='red', ha='center', va='bottom', fontsize=10)
    
    # N and Std in a text block
    info_text = f"N: {count_a}\nStd: {std_a:.1f}%"
    plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes, fontsize=12, fontweight='bold', 
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.title('Distribution of Bubble % - Choice A', fontsize=14, fontweight='bold')
    plt.xlabel('Bubble % (Deviation from Fundamental)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.tight_layout()
    plt.savefig('distribution_A.png', dpi=300)
    print("Chart saved to distribution_A.png")
    plt.close()

    # Plot B
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_b, x='BubblePct', color='orange', label='Choice B', kde=True, stat="density", alpha=0.4)
    
    mean_b = df_b['BubblePct'].mean()
    median_b = df_b['BubblePct'].median()
    std_b = df_b['BubblePct'].std()
    min_b = df_b['BubblePct'].min()
    max_b = df_b['BubblePct'].max()
    count_b = len(df_b)
    
    # Mean line and text
    plt.axvline(mean_b, color='darkorange', linestyle='--', linewidth=2, alpha=0.8)
    plt.text(mean_b + 2, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_b:.1f}%', color='darkorange', fontweight='bold', rotation=0)
    
    # Median line and text
    plt.axvline(median_b, color='green', linestyle=':', linewidth=2, alpha=0.8)
    plt.text(median_b - 2, plt.gca().get_ylim()[1] * 0.8, f'Median: {median_b:.1f}%', color='green', fontweight='bold', rotation=0, ha='right')
    
    # Min/Max markers on X-axis
    plt.plot(min_b, 0, marker='^', color='red', markersize=10)
    plt.text(min_b, plt.gca().get_ylim()[1] * 0.02, f'Min: {min_b:.1f}%', color='red', ha='center', va='bottom', fontsize=10)
    
    plt.plot(max_b, 0, marker='^', color='red', markersize=10)
    plt.text(max_b, plt.gca().get_ylim()[1] * 0.02, f'Max: {max_b:.1f}%', color='red', ha='center', va='bottom', fontsize=10)

    # N and Std in a text block
    info_text = f"N: {count_b}\nStd: {std_b:.1f}%"
    plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes, fontsize=12, fontweight='bold', 
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.title('Distribution of Bubble % - Choice B', fontsize=14, fontweight='bold')
    plt.xlabel('Bubble % (Deviation from Fundamental)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.tight_layout()
    plt.savefig('distribution_B.png', dpi=300)
    print("Chart saved to distribution_B.png")
    plt.close()

if __name__ == "__main__":
    df_a, df_b = load_and_process_data()
    plot_distribution(df_a, df_b)
    
    # Print summary stats
    print("\nSummary Statistics for Choice A:")
    print(df_a['BubblePct'].describe())
    print("\nSummary Statistics for Choice B:")
    print(df_b['BubblePct'].describe())

