#!/usr/bin/env python3
"""
Seaborn Style Price Distribution Analysis for Experiments A and B
Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置seaborn样式
sns.set_style('whitegrid')
sns.set_palette('husl')

def create_experiment_a_seaborn():
    """Create seaborn style distribution analysis for Experiment A"""
    print("Creating Experiment A seaborn style analysis...")
    
    # 读取数据
    df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
    df_a_clean = df_a['LastPrice'].dropna()
    
    # 创建实验A的seaborn分布图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment A: Price Distribution Analysis (Seaborn Style)\n(Learning Effect - 5 Markets per Session)', 
                 fontsize=16, fontweight='bold')
    
    # 1. 直方图 + KDE
    sns.histplot(data=df_a_clean, kde=True, ax=axes[0,0], color='blue', alpha=0.7)
    axes[0,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[0,0].set_ylabel('Frequency', fontsize=12)
    axes[0,0].set_title(f'Price Distribution with KDE\n(n={len(df_a_clean):,} observations)', fontsize=14)
    
    # 2. 箱线图
    sns.boxplot(y=df_a_clean, ax=axes[0,1], color='blue')
    axes[0,1].set_ylabel('Last Price (Francs)', fontsize=12)
    axes[0,1].set_title('Box Plot', fontsize=14)
    
    # 3. 密度图
    sns.histplot(data=df_a_clean, stat='density', kde=True, ax=axes[1,0], color='blue', alpha=0.7)
    axes[1,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[1,0].set_ylabel('Density', fontsize=12)
    axes[1,0].set_title('Probability Density with KDE', fontsize=14)
    
    # 4. 小提琴图
    sns.violinplot(y=df_a_clean, ax=axes[1,1], color='blue')
    axes[1,1].set_ylabel('Last Price (Francs)', fontsize=12)
    axes[1,1].set_title('Violin Plot', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('experiment_a_seaborn_style.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("实验A seaborn风格图表已保存为: experiment_a_seaborn_style.png")

def create_experiment_b_seaborn():
    """Create seaborn style distribution analysis for Experiment B"""
    print("Creating Experiment B seaborn style analysis...")
    
    # 读取数据
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    df_b_clean = df_b['LastPrice'].dropna()
    
    # 创建实验B的seaborn分布图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment B: Price Distribution Analysis (Seaborn Style)\n(Professional vs Students)', 
                 fontsize=16, fontweight='bold')
    
    # 1. 直方图 + KDE
    sns.histplot(data=df_b_clean, kde=True, ax=axes[0,0], color='red', alpha=0.7)
    axes[0,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[0,0].set_ylabel('Frequency', fontsize=12)
    axes[0,0].set_title(f'Price Distribution with KDE\n(n={len(df_b_clean):,} observations)', fontsize=14)
    
    # 2. 箱线图
    sns.boxplot(y=df_b_clean, ax=axes[0,1], color='red')
    axes[0,1].set_ylabel('Last Price (Francs)', fontsize=12)
    axes[0,1].set_title('Box Plot', fontsize=14)
    
    # 3. 密度图
    sns.histplot(data=df_b_clean, stat='density', kde=True, ax=axes[1,0], color='red', alpha=0.7)
    axes[1,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[1,0].set_ylabel('Density', fontsize=12)
    axes[1,0].set_title('Probability Density with KDE', fontsize=14)
    
    # 4. 小提琴图
    sns.violinplot(y=df_b_clean, ax=axes[1,1], color='red')
    axes[1,1].set_ylabel('Last Price (Francs)', fontsize=12)
    axes[1,1].set_title('Violin Plot', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('experiment_b_seaborn_style.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("实验B seaborn风格图表已保存为: experiment_b_seaborn_style.png")

def create_comparison_seaborn():
    """Create seaborn style comparison plots"""
    print("Creating seaborn style comparison plots...")
    
    # 读取数据
    df_a = pd.read_csv('Experiment_A_Trading_Data.csv')
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    
    # 准备对比数据
    df_a_clean = df_a['LastPrice'].dropna()
    df_b_clean = df_b['LastPrice'].dropna()
    
    # 创建DataFrame用于seaborn
    data_a = pd.DataFrame({'LastPrice': df_a_clean, 'Experiment': 'A'})
    data_b = pd.DataFrame({'LastPrice': df_b_clean, 'Experiment': 'B'})
    combined_data = pd.concat([data_a, data_b], ignore_index=True)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment A vs B: Seaborn Style Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. 直方图对比
    sns.histplot(data=combined_data, x='LastPrice', hue='Experiment', kde=True, 
                 ax=axes[0,0], alpha=0.7)
    axes[0,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[0,0].set_ylabel('Frequency', fontsize=12)
    axes[0,0].set_title('Price Distribution Comparison', fontsize=14)
    
    # 2. 箱线图对比
    sns.boxplot(data=combined_data, x='Experiment', y='LastPrice', ax=axes[0,1])
    axes[0,1].set_xlabel('Experiment', fontsize=12)
    axes[0,1].set_ylabel('Last Price (Francs)', fontsize=12)
    axes[0,1].set_title('Box Plot Comparison', fontsize=14)
    
    # 3. 小提琴图对比
    sns.violinplot(data=combined_data, x='Experiment', y='LastPrice', ax=axes[1,0])
    axes[1,0].set_xlabel('Experiment', fontsize=12)
    axes[1,0].set_ylabel('Last Price (Francs)', fontsize=12)
    axes[1,0].set_title('Violin Plot Comparison', fontsize=14)
    
    # 4. 密度图对比
    sns.histplot(data=combined_data, x='LastPrice', hue='Experiment', stat='density', 
                 kde=True, ax=axes[1,1], alpha=0.7)
    axes[1,1].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[1,1].set_ylabel('Density', fontsize=12)
    axes[1,1].set_title('Density Comparison', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('experiments_seaborn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("实验对比seaborn风格图表已保存为: experiments_seaborn_comparison.png")

def create_professional_vs_student_seaborn():
    """Create seaborn style professional vs student analysis"""
    print("Creating professional vs student seaborn analysis...")
    
    # 读取实验B数据
    df_b = pd.read_csv('Experiment_B_Trading_Data.csv')
    
    # 创建专业vs学生对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Professional vs Student Traders: Seaborn Style Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. 价格分布对比
    sns.histplot(data=df_b, x='LastPrice', hue='Role', kde=True, 
                 ax=axes[0,0], alpha=0.7)
    axes[0,0].set_xlabel('Last Price (Francs)', fontsize=12)
    axes[0,0].set_ylabel('Frequency', fontsize=12)
    axes[0,0].set_title('Price Distribution by Role', fontsize=14)
    
    # 2. 箱线图对比
    sns.boxplot(data=df_b, x='Role', y='LastPrice', ax=axes[0,1])
    axes[0,1].set_xlabel('Role', fontsize=12)
    axes[0,1].set_ylabel('Last Price (Francs)', fontsize=12)
    axes[0,1].set_title('Price by Role', fontsize=14)
    
    # 3. 小提琴图对比
    sns.violinplot(data=df_b, x='Role', y='LastPrice', ax=axes[1,0])
    axes[1,0].set_xlabel('Role', fontsize=12)
    axes[1,0].set_ylabel('Last Price (Francs)', fontsize=12)
    axes[1,0].set_title('Price Distribution by Role (Violin)', fontsize=14)
    
    # 4. 专业交易员比例vs价格散点图
    sns.scatterplot(data=df_b, x='ProShare', y='LastPrice', hue='Role', 
                    ax=axes[1,1], alpha=0.6)
    axes[1,1].set_xlabel('Professional Share', fontsize=12)
    axes[1,1].set_ylabel('Last Price (Francs)', fontsize=12)
    axes[1,1].set_title('Professional Share vs Price', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('professional_vs_student_seaborn.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("专业vs学生seaborn风格图表已保存为: professional_vs_student_seaborn.png")

def main():
    """Main function to run all seaborn analyses"""
    print("="*80)
    print("SEABORN STYLE PRICE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # 创建实验A seaborn风格图表
    create_experiment_a_seaborn()
    
    # 创建实验B seaborn风格图表
    create_experiment_b_seaborn()
    
    # 创建对比图
    create_comparison_seaborn()
    
    # 创建专业vs学生分析
    create_professional_vs_student_seaborn()
    
    print("\n" + "="*80)
    print("所有seaborn风格分析完成！生成的文件:")
    print("- experiment_a_seaborn_style.png")
    print("- experiment_b_seaborn_style.png") 
    print("- experiments_seaborn_comparison.png")
    print("- professional_vs_student_seaborn.png")
    print("="*80)

if __name__ == "__main__":
    main()
