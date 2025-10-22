#!/usr/bin/env python3
"""
Individual Price Distribution Analysis for Experiments A and B
Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_experiment_a_distribution():
    """Create price distribution analysis for Experiment A"""
    print("Creating Experiment A price distribution analysis...")
    
    # 读取数据
    df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
    df_a_clean = df_a['LastPrice'].dropna()
    
    # 创建实验A的分布图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment A: Price Distribution Analysis\n(Learning Effect - 5 Markets per Session)', 
                 fontsize=16, fontweight='bold')
    
    # 1. 直方图
    axes[0,0].hist(df_a_clean, bins=80, alpha=0.8, color='blue', edgecolor='black', linewidth=0.5)
    axes[0,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[0,0].set_ylabel('Frequency', fontsize=12)
    axes[0,0].set_title(f'Price Distribution\n(n={len(df_a_clean):,} observations)', fontsize=14)
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 箱线图
    axes[0,1].boxplot(df_a_clean, patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.7))
    axes[0,1].set_ylabel('Last Price (Francs)', fontsize=12)
    axes[0,1].set_title('Box Plot', fontsize=14)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 累积分布函数
    sorted_a = np.sort(df_a_clean)
    y_a = np.arange(1, len(sorted_a) + 1) / len(sorted_a)
    axes[1,0].plot(sorted_a, y_a, color='blue', linewidth=2)
    axes[1,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[1,0].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1,0].set_title('Cumulative Distribution Function', fontsize=14)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 密度图
    axes[1,1].hist(df_a_clean, bins=100, alpha=0.8, color='blue', density=True, 
                   edgecolor='black', linewidth=0.3)
    axes[1,1].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[1,1].set_ylabel('Density', fontsize=12)
    axes[1,1].set_title('Probability Density', fontsize=14)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_a_price_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("实验A价格分布图已保存为: experiment_a_price_distribution.png")
    
    # 打印统计信息
    print(f"\n实验A统计信息:")
    print(f"数据量: {len(df_a_clean):,} 行")
    print(f"价格范围: {df_a_clean.min():.2f} - {df_a_clean.max():.2f}")
    print(f"平均价格: {df_a_clean.mean():.2f}")
    print(f"中位数: {df_a_clean.median():.2f}")
    print(f"标准差: {df_a_clean.std():.2f}")

def create_experiment_b_distribution():
    """Create price distribution analysis for Experiment B"""
    print("\nCreating Experiment B price distribution analysis...")
    
    # 读取数据
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    df_b_clean = df_b['LastPrice'].dropna()
    
    # 创建实验B的分布图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment B: Price Distribution Analysis\n(Professional vs Students)', 
                 fontsize=16, fontweight='bold')
    
    # 1. 直方图
    axes[0,0].hist(df_b_clean, bins=80, alpha=0.8, color='red', edgecolor='black', linewidth=0.5)
    axes[0,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[0,0].set_ylabel('Frequency', fontsize=12)
    axes[0,0].set_title(f'Price Distribution\n(n={len(df_b_clean):,} observations)', fontsize=14)
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 箱线图
    axes[0,1].boxplot(df_b_clean, patch_artist=True, boxprops=dict(facecolor='red', alpha=0.7))
    axes[0,1].set_ylabel('Last Price (Francs)', fontsize=12)
    axes[0,1].set_title('Box Plot', fontsize=14)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 累积分布函数
    sorted_b = np.sort(df_b_clean)
    y_b = np.arange(1, len(sorted_b) + 1) / len(sorted_b)
    axes[1,0].plot(sorted_b, y_b, color='red', linewidth=2)
    axes[1,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[1,0].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1,0].set_title('Cumulative Distribution Function', fontsize=14)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 密度图
    axes[1,1].hist(df_b_clean, bins=100, alpha=0.8, color='red', density=True, 
                   edgecolor='black', linewidth=0.3)
    axes[1,1].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[1,1].set_ylabel('Density', fontsize=12)
    axes[1,1].set_title('Probability Density', fontsize=14)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_b_price_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("实验B价格分布图已保存为: experiment_b_price_distribution.png")
    
    # 打印统计信息
    print(f"\n实验B统计信息:")
    print(f"数据量: {len(df_b_clean):,} 行")
    print(f"价格范围: {df_b_clean.min():.2f} - {df_b_clean.max():.2f}")
    print(f"平均价格: {df_b_clean.mean():.2f}")
    print(f"中位数: {df_b_clean.median():.2f}")
    print(f"标准差: {df_b_clean.std():.2f}")

def create_comparison_summary():
    """Create a summary comparison of both experiments"""
    print("\nCreating comparison summary...")
    
    # 读取数据
    df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    
    df_a_clean = df_a['LastPrice'].dropna()
    df_b_clean = df_b['LastPrice'].dropna()
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Experiment A vs Experiment B: Price Distribution Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. 直方图对比
    axes[0].hist(df_a_clean, bins=60, alpha=0.7, color='blue', 
                 label=f'Experiment A (n={len(df_a_clean):,})', edgecolor='black', linewidth=0.5)
    axes[0].hist(df_b_clean, bins=60, alpha=0.7, color='red', 
                 label=f'Experiment B (n={len(df_b_clean):,})', edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Price Distribution Comparison', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 箱线图对比
    data_to_plot = [df_a_clean, df_b_clean]
    box_plot = axes[1].boxplot(data_to_plot, tick_labels=['Experiment A', 'Experiment B'], 
                               patch_artist=True, showfliers=True)
    box_plot['boxes'][0].set_facecolor('blue')
    box_plot['boxes'][1].set_facecolor('red')
    axes[1].set_ylabel('Last Price (Francs)', fontsize=12)
    axes[1].set_title('Box Plot Comparison', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # 3. 累积分布函数对比
    sorted_a = np.sort(df_a_clean)
    sorted_b = np.sort(df_b_clean)
    y_a = np.arange(1, len(sorted_a) + 1) / len(sorted_a)
    y_b = np.arange(1, len(sorted_b) + 1) / len(sorted_b)
    
    axes[2].plot(sorted_a, y_a, color='blue', linewidth=2, label='Experiment A')
    axes[2].plot(sorted_b, y_b, color='red', linewidth=2, label='Experiment B')
    axes[2].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[2].set_ylabel('Cumulative Probability', fontsize=12)
    axes[2].set_title('Cumulative Distribution Functions', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("实验对比图已保存为: experiments_comparison.png")
    
    # 打印对比统计
    print(f"\n对比统计:")
    print(f"实验A平均价格: {df_a_clean.mean():.2f}")
    print(f"实验B平均价格: {df_b_clean.mean():.2f}")
    print(f"平均价格差异: {df_a_clean.mean() - df_b_clean.mean():.2f}")
    print(f"实验A标准差: {df_a_clean.std():.2f}")
    print(f"实验B标准差: {df_b_clean.std():.2f}")

def main():
    """Main function to run all analyses"""
    print("="*80)
    print("INDIVIDUAL PRICE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # 创建实验A分布图
    create_experiment_a_distribution()
    
    # 创建实验B分布图
    create_experiment_b_distribution()
    
    # 创建对比图
    create_comparison_summary()
    
    print("\n" + "="*80)
    print("所有分析完成！生成的文件:")
    print("- experiment_a_price_distribution.png")
    print("- experiment_b_price_distribution.png") 
    print("- experiments_comparison.png")
    print("="*80)

if __name__ == "__main__":
    main()
