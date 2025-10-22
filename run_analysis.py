#!/usr/bin/env python3
"""
Main analysis runner for Bubble Market Experiments
Author: Yixuan
Course: MGSC 406 Advanced Experimental Design & Statistics
"""

import subprocess
import sys

def run_script(script_name):
    """Run a Python script and handle errors"""
    try:
        print(f"\n{'='*60}")
        print(f"RUNNING: {script_name}")
        print('='*60)
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Run all analysis scripts"""
    print("BUBBLE MARKET EXPERIMENT ANALYSIS")
    print("="*60)
    
    scripts = [
        'price_analysis.py',
        'professional_vs_student_analysis.py'
    ]
    
    success_count = 0
    
    for script in scripts:
        if run_script(script):
            success_count += 1
        print(f"\n{script} completed")
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY: {success_count}/{len(scripts)} scripts completed successfully")
    print('='*60)
    
    if success_count == len(scripts):
        print("All analyses completed successfully!")
        print("\nGenerated files:")
        print("- price_distribution_analysis.png")
        print("- session_trends_analysis.png") 
        print("- professional_vs_student_analysis.png")
        print("- professional_share_analysis.png")
    else:
        print("Some analyses failed. Check error messages above.")

if __name__ == "__main__":
    main()
