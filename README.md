# Behavioral Bubble Markets Analysis

Empirical analysis of experimental asset bubbles (Choice A vs Choice B) for MGSC 406 Advanced Experimental Design & Statistics course at Chapman University Fall 2025.

## Project Overview

This project analyzes experimental data from two distinct bubble market experiments designed to test the hypothesis that **bubbles are a result of inexperience**. The research investigates whether experienced traders anchor prices closer to fundamentals, leading to diminished or eliminated bubbles.


## Research Question

**Are bubbles a result of inexperience?** 

If bubbles reflect confusion, misunderstanding of fundamental values, or lack of familiarity with trading in laboratory experiments, then repeated exposure should allow subjects to learn and converge toward fundamental pricing as predicted by theory.

## Experimental Design

### Choice A: Learning Through Repetition
- **Subjects**: 72 undergraduate students (6 sessions of 12 subjects)
- **Structure**: 5 consecutive markets per session
- **Trading**: 15 periods of continuous double-auction trading per market
- **Dividend**: Random dividend each period (0, 8, 28, or 60 francs, equally likely 25%)
- **Initial Endowment**: 600 francs cash + 4 assets per subject
- **Information Treatment**:

| Session | Information Display | Description |
|---------|-------------------|-------------|
| 1-3 | Fundamental values shown | High information transparency |
| 4-6 | Fundamental values hidden | Low information transparency |

### Choice B: Professional vs Student Traders
- **Subjects**: 96 participants (undergraduate students + NYSE professional traders)
- **Structure**: 8 sessions of 12 subjects each
- **Trading**: 15 periods of continuous double-auction trading per market
- **Professional Trader Mix**:

| Session | Professionals | Students | Professional Share | Description |
|---------|---------------|----------|-------------------|-------------|
| 1-2 | 3 | 9 | 25% | Low professional influence |
| 3-4 | 6 | 6 | 50% | Balanced mix |
| 5-6 | 9 | 3 | 75% | High professional influence |
| 7-8 | 12 | 0 | 100% | Professional traders only |

**Note**: Each session has 12 subjects total, with 15 trading periods per session.


## Analysis Framework

Based on behavioral finance literature, this project follows a structured analysis framework across four analytical layers:

| Layer | N | Behavioral Idea | Hypothesis (H0 vs H1) | Test Type | Data Used | Expected Direction/Evidence |
|-------|---|-----------------|----------------------|-----------|-----------|---------------------------|
| **Market** | 0 | **Last Price Comparison** | H0: μ_A = μ_B vs H1: μ_A ≠ μ_B | Two-sample t-test / Mann-Whitney U | LastPrice by experiment | **RESULT: No significant difference (p=0.878)** |
| | 1 | Market Efficiency | H0: bubble mean = 0 vs H1: bubble mean ≠ 0 | One-sample t-test / Wilcoxon signed-rank | All trades (Price - Fundamental) | If p < 0.05 → Evidence of bubbles (inefficiency) |
| | 2 | Information Structure (Choice A vs B) | H0: μ_A = μ_B vs H1: μ_A ≠ μ_B | Two-sample t-test / Mann-Whitney U | LastBubble by experiment | Expected μ_B < μ_A → More info = smaller bubbles |
| | 3 | Learning Across Sessions | H0: bubble means equal across sessions | One-way ANOVA / Kruskal-Wallis | Bubble by Session | Declining trend; later sessions show smaller bubbles |
| **Individual** | 4 | Trader Type Effect | H0: μ_pro = μ_student vs H1: μ_pro ≠ μ_student | Two-sample t-test / Mann-Whitney U | Final Payoff by TraderType | Expected μ_pro > μ_student → Higher average returns |
| | 5 | Dividend Regime | H0: μ_highdiv = μ_lowdiv vs H1: μ_highdiv ≠ μ_lowdiv | One-way ANOVA | Final Payoff by Dividend level | μ_highdiv > μ_lowdiv → Structural luck component |
| | 6 | Skill vs Luck | H0: σ²_within = σ²_expected vs H1: σ²_within ≠ σ²_expected | F-test / Levene's test | Payoff conditional on Dividend | Significant variance differences → Evidence of skill heterogeneity |
| **Outcomes** | 7 | Bubble-Profit Link | H0: ρ = 0 vs H1: ρ ≠ 0 | Pearson / Spearman correlation | Bubble × Final Payoff | Expected ρ < 0 → Negative relationship |
| | 8 | Outcome Inequality | H0: Var_A = Var_B vs H1: Var_A ≠ Var_B | Equal variance F-test | Final Payoff by Experiment | Var_A > Var_B → Increased inequality under low info |
| **Behaviors** | 9 | Aggregate Dynamics | H0: No trend vs H1: Trend exists | Regression of mean bubble on period | Session-level means | Negative β → Learning or self-correction |
| | 10 | Behavioral Anchoring | H0: β_trade = β_fundamental vs H1: β_trade ≠ β_fundamental | Regression on lagged prices and fundamentals | Period-level panel | β_trade > β_fundamental → Anchoring bias |
| | 11 | Coordination Failure | H0: Given fundamentals, bubble variance = 0 vs H1: variance > 0 | Variance decomposition / χ² test | Market-level Bubble | Persistent dispersion → Coordination breakdown |
| | 12 | Aggregate Payoff Efficiency | H0: Payoff mean = Expected fundamental value vs H1: Payoff mean ≠ Expected value | One-sample t-test | Market-mean Final Payoff | Deviation → Systematic inefficiency |

## Analysis Plan

1. **Game Description**: Explain the trading game, standard economic predictions, and early experimental findings
2. **Experimental Comparison**: Compare the strengths and weaknesses of Experiment A vs Experiment B
3. **Statistical Analysis**: Conduct appropriate statistical tests following the 12-point framework
4. **Critical Reflection**: Evaluate experimental design effectiveness
5. **Master's Extension**: Propose new hypothesis, design, and analysis plan

## Key Findings

### **Primary Result: No Significant Price Difference**
- **Choice A (Students only)** vs **Choice B (Professional + Students mix)** show **no significant difference** in LastPrice (p = 0.878)
- **Mean prices**: Choice A = 256.58, Choice B = 257.26
- **Implication**: Professional traders do not significantly affect overall price levels in laboratory markets

### **Research Question Remains Open**
While price levels are similar, the analysis continues to examine:
- **Bubble dynamics** (price deviations from fundamentals)
- **Learning effects** across sessions
- **Professional trader influence** on market behavior
- **Information structure effects** on market efficiency


#Appendix

## Data Files

- `Bubble_Markets_2025.csv` - Main dataset in CSV format
- `Bubble_Markets_2025.xlsx` - Main dataset in Excel format
- `Assignment Bubbles in Financial Markets.txt` - Assignment description


## Deliverables

- 10-12 slide presentation deck (15 minutes)
- 3-5 page memo (design, methods, results, critique)
- Code and data analysis scripts
- Statistical test results and interpretation

## Repository Structure

```
BehavioralBubbleMkt/
├── README.md
├── Assignment Bubbles in Financial Markets.txt
├── Bubble_Markets_2025.xlsx
├── Experiment_A_Trading_Data.csv
├── Experiment_B_Trading_Data.csv
├── Analysis Scripts/
│   ├── individual_price_distributions.py
│   ├── seaborn_price_analysis.py
│   ├── price_analysis.py
│   ├── professional_vs_student_analysis.py
│   └── run_analysis.py
├── Generated Charts/
│   ├── experiment_a_price_distribution.png
│   ├── experiment_b_price_distribution.png
│   ├── experiments_comparison.png
│   ├── experiment_a_seaborn_style.png
│   ├── experiment_b_seaborn_style.png
│   ├── experiments_seaborn_comparison.png
│   └── professional_vs_student_seaborn.png
└── LICENSE
```

## Course Information

- **Course**: MGSC 406 Advanced Experimental Design & Statistics
- **Institution**: Chapman University
- **Term**: Fall 2025
- **Student**: Yixuan

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Getting Started

1. Clone the repository
2. Load the data files (`Experiment_A_Trading_Data.csv` and `Experiment_B_Trading_Data.csv`)
3. Review the assignment description in `Assignment Bubbles in Financial Markets.txt`
4. Run the analysis scripts:
   ```bash
   # Run all analyses
   python3 run_analysis.py
   
   # Or run individual analyses
   python3 individual_price_distributions.py
   python3 seaborn_price_analysis.py
   python3 professional_vs_student_analysis.py
   ```

## Contact

For questions about this analysis, please contact the repository owner.
