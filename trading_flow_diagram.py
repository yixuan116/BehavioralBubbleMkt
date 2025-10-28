#!/usr/bin/env python3
"""
Trading Flow Diagram Generator
Creates a visual representation of the trading mechanism

Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_trading_flow_diagram():
    """Create a comprehensive trading flow diagram"""
    print("Creating Trading Flow Diagram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Trading Flow Mechanism', fontsize=20, fontweight='bold', ha='center')
    
    # Session Level
    session_box = FancyBboxPatch((0.5, 10.5), 9, 0.8, boxstyle="round,pad=0.1", 
                                facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(session_box)
    ax.text(5, 10.9, 'SESSION (6 sessions total)', fontsize=14, fontweight='bold', ha='center')
    
    # Market Level
    market_box = FancyBboxPatch((0.5, 9.5), 9, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(market_box)
    ax.text(5, 9.9, 'MARKET (5 consecutive markets per session)', fontsize=14, fontweight='bold', ha='center')
    
    # Period Level
    period_box = FancyBboxPatch((0.5, 8.5), 9, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(period_box)
    ax.text(5, 8.9, 'PERIOD (15 periods per market)', fontsize=14, fontweight='bold', ha='center')
    
    # Trading Process
    ax.text(5, 8.2, 'TRADING PROCESS', fontsize=16, fontweight='bold', ha='center')
    
    # Step 1: Initial Setup
    step1_box = FancyBboxPatch((0.5, 7.2), 4, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='lightcoral', edgecolor='black', linewidth=1)
    ax.add_patch(step1_box)
    ax.text(2.5, 7.6, '1. Initial Setup', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.5, 7.3, 'Cash: 600 francs\nShares: 4 units', fontsize=10, ha='center')
    
    # Step 2: Continuous Trading
    step2_box = FancyBboxPatch((5.5, 7.2), 4, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='lightcoral', edgecolor='black', linewidth=1)
    ax.add_patch(step2_box)
    ax.text(7.5, 7.6, '2. Continuous Trading', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.5, 7.3, 'Post bids/asks\nDouble auction', fontsize=10, ha='center')
    
    # Trading Details
    ax.text(5, 6.5, 'TRADING MECHANISM', fontsize=14, fontweight='bold', ha='center')
    
    # Participants
    participants_box = FancyBboxPatch((0.5, 5.5), 4, 1.5, boxstyle="round,pad=0.1", 
                                    facecolor='lightpink', edgecolor='black', linewidth=1)
    ax.add_patch(participants_box)
    ax.text(2.5, 6.8, '12 Participants', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.5, 6.4, '• Post bids (buy orders)\n• Post asks (sell orders)\n• System matches orders\n• Instant execution', fontsize=10, ha='center')
    
    # Price Formation
    price_box = FancyBboxPatch((5.5, 5.5), 4, 1.5, boxstyle="round,pad=0.1", 
                              facecolor='lightcyan', edgecolor='black', linewidth=1)
    ax.add_patch(price_box)
    ax.text(7.5, 6.8, 'Price Formation', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.5, 6.4, '• LastPrice = Last trade\n• Period ends\n• Display to all\n• Next period starts', fontsize=10, ha='center')
    
    # Constraints
    ax.text(5, 4.8, 'CONSTRAINTS', fontsize=14, fontweight='bold', ha='center')
    
    constraints_box = FancyBboxPatch((1, 3.5), 8, 1, boxstyle="round,pad=0.1", 
                                    facecolor='lightgray', edgecolor='black', linewidth=1)
    ax.add_patch(constraints_box)
    ax.text(5, 4.2, '• Cash constraint: Only 600 francs per market', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 3.8, '• No reinvestment: Trading profits cannot be used for more trading', fontsize=12, ha='center')
    ax.text(5, 3.6, '• Final payoff: Cash + Dividends - Purchases', fontsize=12, ha='center')
    
    # Market Reset
    ax.text(5, 3.2, 'MARKET RESET', fontsize=14, fontweight='bold', ha='center')
    
    reset_box = FancyBboxPatch((1, 2.2), 8, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='lightsteelblue', edgecolor='black', linewidth=1)
    ax.add_patch(reset_box)
    ax.text(5, 2.6, 'Each new market: Cash reset to 600, Shares reset to 4', fontsize=12, fontweight='bold', ha='center')
    
    # Data Flow
    ax.text(5, 1.8, 'DATA FLOW', fontsize=14, fontweight='bold', ha='center')
    
    data_box = FancyBboxPatch((1, 0.5), 8, 1, boxstyle="round,pad=0.1", 
                             facecolor='lightgoldenrodyellow', edgecolor='black', linewidth=1)
    ax.add_patch(data_box)
    ax.text(5, 1.2, 'CSV Data: Session, Market, Period, SubjectID, LastPrice, UnitsBuy, UnitsSell, AvgBuyPrice, AvgSellPrice', fontsize=10, ha='center')
    ax.text(5, 0.8, 'LastPrice = Period closing price (source unknown, but constrained by cash limits)', fontsize=10, ha='center')
    
    # Arrows
    # Session to Market
    arrow1 = ConnectionPatch((5, 10.5), (5, 9.5), "data", "data", 
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, fc="black")
    ax.add_patch(arrow1)
    
    # Market to Period
    arrow2 = ConnectionPatch((5, 9.5), (5, 8.5), "data", "data", 
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, fc="black")
    ax.add_patch(arrow2)
    
    # Period to Trading
    arrow3 = ConnectionPatch((5, 8.5), (5, 7.2), "data", "data", 
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, fc="black")
    ax.add_patch(arrow3)
    
    plt.tight_layout()
    plt.savefig('trading_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Trading flow diagram saved as 'trading_flow_diagram.png'")

def create_detailed_trading_sequence():
    """Create a detailed trading sequence diagram"""
    print("\nCreating Detailed Trading Sequence...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Detailed Trading Sequence', fontsize=18, fontweight='bold', ha='center')
    
    # Time sequence
    time_points = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    time_labels = ['Period Start', 'Participant 1\nPosts Bid', 'Participant 2\nPosts Ask', 
                   'System Matches\nBid=Ask', 'Trade Executed', 'More Trades\nContinue', 
                   'Period Nears\nEnd', 'Last Trade\nExecuted', 'Period End\nLastPrice Set']
    
    for i, (time, label) in enumerate(zip(time_points, time_labels)):
        # Time point
        circle = plt.Circle((time, 7), 0.3, color='blue', alpha=0.7)
        ax.add_patch(circle)
        ax.text(time, 7, str(i+1), fontsize=10, fontweight='bold', ha='center', va='center', color='white')
        
        # Label
        ax.text(time, 6, label, fontsize=9, ha='center', va='top')
        
        # Arrow to next
        if i < len(time_points) - 1:
            arrow = ConnectionPatch((time+0.3, 7), (time+1-0.3, 7), "data", "data", 
                                  arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=15, fc="black")
            ax.add_patch(arrow)
    
    # Key insights
    insights_box = FancyBboxPatch((0.5, 3), 9, 2.5, boxstyle="round,pad=0.1", 
                                facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(insights_box)
    
    ax.text(5, 5.2, 'KEY INSIGHTS', fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 4.8, '• LastPrice = Last executed trade price in the period', fontsize=12, ha='center')
    ax.text(5, 4.5, '• Price formation is continuous throughout the period', fontsize=12, ha='center')
    ax.text(5, 4.2, '• Cash constraints prevent extreme price movements', fontsize=12, ha='center')
    ax.text(5, 3.9, '• Each market resets with fresh 600 francs', fontsize=12, ha='center')
    ax.text(5, 3.6, '• Trading profits accumulate but cannot be reinvested', fontsize=12, ha='center')
    
    # Data structure
    data_box = FancyBboxPatch((0.5, 0.5), 9, 2, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(data_box)
    
    ax.text(5, 2.2, 'DATA STRUCTURE', fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 1.8, 'Each row = One participant in one period', fontsize=12, ha='center')
    ax.text(5, 1.5, 'LastPrice = Same for all participants in same period', fontsize=12, ha='center')
    ax.text(5, 1.2, 'AvgBuyPrice/AvgSellPrice = Individual participant prices', fontsize=12, ha='center')
    ax.text(5, 0.9, 'LastPrice source: Unknown calculation method', fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.savefig('trading_sequence_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Trading sequence diagram saved as 'trading_sequence_diagram.png'")

def main():
    """Main function"""
    print("Creating Trading Flow Diagrams...")
    
    # Create main flow diagram
    create_trading_flow_diagram()
    
    # Create detailed sequence diagram
    create_detailed_trading_sequence()
    
    print(f"\n" + "="*60)
    print("TRADING FLOW DIAGRAMS COMPLETE")
    print("="*60)
    print("✓ Created main trading flow diagram")
    print("✓ Created detailed trading sequence diagram")
    print("✓ Shows complete trading mechanism and constraints")

if __name__ == "__main__":
    main()
