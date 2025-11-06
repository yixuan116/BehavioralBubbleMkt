"""
Phase 1 – Step 3: Within-Mix Session Consistency (Mann–Whitney U) and Probability of Bubble > 0 (Chi-square test)
Automated statistical analysis for Experiment B (Bubble_Markets_2025.xlsx)
Calculate Bubble% from raw data and perform pairwise Mann-Whitney U tests
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency

# ------------------------------------------
# 1. Load Excel and calculate Bubble% from raw data
# ------------------------------------------
print("Loading data from Bubble_Markets_2025.xlsx...")
df = pd.read_excel("Bubble_Markets_2025.xlsx", sheet_name="Choices_B")

# Calculate Bubble% for each period: (LastPrice - Fundamental) / Fundamental * 100
df['BubblePct'] = ((df['LastPrice'] - df['Fundamental']) / df['Fundamental']) * 100

# Group by Session and Period to get period-level Bubble%
period_bubble = df.groupby(['Session', 'Period']).agg({
    'BubblePct': 'mean',  # Average bubble% across all subjects in this period
    'ProShare': 'first'   # Professional share percentage (same for all rows in a session)
}).reset_index()

print(f"✅ Calculated Bubble% for {len(period_bubble)} session-period combinations")
print(f"   Sessions: {sorted(period_bubble['Session'].unique())}")
print(f"   Professional share percentages: {sorted(period_bubble['ProShare'].unique())}")

# ------------------------------------------
# 2. Extract Bubble% arrays for each session (15 periods each)
# ------------------------------------------
session_bubbles = {}
for session in sorted(period_bubble['Session'].unique()):
    session_data = period_bubble[period_bubble['Session'] == session]
    bubble_array = session_data.sort_values('Period')['BubblePct'].values
    pro_share = session_data['ProShare'].iloc[0]
    session_bubbles[session] = {
        'bubbles': bubble_array,
        'pro_share': pro_share
    }
    print(f"   Session {session} ({pro_share}%): {len(bubble_array)} periods, mean={np.mean(bubble_array):.2f}%")

# ------------------------------------------
# 3. Q2 – Within-Mix Session Consistency (Mann–Whitney U)
# ------------------------------------------
pairs = [
    (1, 2, "25%"),
    (3, 4, "50%"),
    (5, 6, "75%"),
    (7, 8, "100%")
]

results_q2 = []
print("\n" + "="*60)
print("Q2: Pairwise Mann-Whitney U Tests (Session-Level Consistency)")
print("="*60)

for s1, s2, mix_label in pairs:
    x = session_bubbles[s1]['bubbles']
    y = session_bubbles[s2]['bubbles']
    
    # Perform Mann-Whitney U test
    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    
    results_q2.append({
        "Pro Mix": mix_label,
        "Sessions": f"S{s1}_{mix_label} vs S{s2}_{mix_label}",
        "U": stat,
        "p_value": p,
        "Median1": np.median(x),
        "Median2": np.median(y),
        "Mean1": np.mean(x),
        "Mean2": np.mean(y),
        "N1": len(x),
        "N2": len(y)
    })
    
    print(f"\n{mix_label} Professional Share:")
    print(f"  S{s1} vs S{s2}:")
    print(f"    U-statistic: {stat:.1f}")
    print(f"    p-value: {p:.3f}")
    print(f"    Significant (α=0.05): {'No' if p > 0.05 else 'Yes'}")
    print(f"    S{s1} - Mean: {np.mean(x):.2f}%, Median: {np.median(x):.2f}%")
    print(f"    S{s2} - Mean: {np.mean(y):.2f}%, Median: {np.median(y):.2f}%")

results_q2_df = pd.DataFrame(results_q2)
print("\n" + "="*60)
print("Summary Table:")
print("="*60)
print(results_q2_df[['Pro Mix', 'Sessions', 'U', 'p_value']].round(3))

# ------------------------------------------
# 4. Q3 – Probability of Bubble > 0 (Chi-square test)
# ------------------------------------------
print("\n" + "="*60)
print("Q3: Probability of Bubble > 0 by Professional Share")
print("="*60)

bubble_probs = []
for mix in ["25%", "50%", "75%", "100%"]:
    # Get all bubble values for this mix
    vals = []
    for s1, s2, mix_label in pairs:
        if mix_label == mix:
            vals.extend(session_bubbles[s1]['bubbles'])
            vals.extend(session_bubbles[s2]['bubbles'])
    
    pos = np.sum(np.array(vals) > 0)
    neg = np.sum(np.array(vals) <= 0)
    prob = pos / (pos + neg) if (pos + neg) > 0 else 0
    
    bubble_probs.append({
        "Pro Mix": mix,
        "P(Bubble>0)": prob,
        "N(Bubble>0)": pos,
        "N(Bubble<=0)": neg,
        "Total": pos + neg
    })
    print(f"{mix}: P(Bubble>0) = {prob:.3f} ({pos}/{pos+neg})")

prob_df = pd.DataFrame(bubble_probs)
print("\n" + prob_df.to_string(index=False))

# Chi-square test across mixes
contingency = np.array([
    [prob_df.loc[prob_df['Pro Mix'] == mix, 'N(Bubble>0)'].values[0],
     prob_df.loc[prob_df['Pro Mix'] == mix, 'N(Bubble<=0)'].values[0]]
    for mix in ["25%", "50%", "75%", "100%"]
])

chi2, p_chi, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test across mixes:")
print(f"  χ² = {chi2:.3f}")
print(f"  df = {dof}")
print(f"  p-value = {p_chi:.4f}")
print(f"  Significant (α=0.05): {'Yes' if p_chi < 0.05 else 'No'}")

# ------------------------------------------
# 5. Export all results
# ------------------------------------------
with pd.ExcelWriter("Step3_withinMix_results.xlsx") as writer:
    results_q2_df.to_excel(writer, sheet_name="Q2_MannWhitney", index=False)
    prob_df.to_excel(writer, sheet_name="Q3_BubbleProb", index=False)

print("\n✅ Exported: Step3_withinMix_results.xlsx")

# ------------------------------------------
# 6. Compare with yesterday's results
# ------------------------------------------
print("\n" + "="*60)
print("Comparison with Yesterday's Results (from README.md):")
print("="*60)
print("Yesterday's results:")
print("  S1_25 vs S2_25: U=121.5, p=0.724")
print("  S3_50 vs S4_50: U=127.0, p=0.561")
print("  S5_75 vs S6_75: U=108.0, p=0.868")
print("  S7_100 vs S8_100: U=136.5, p=0.329")
print("\nToday's results:")
for _, row in results_q2_df.iterrows():
    print(f"  {row['Sessions']}: U={row['U']:.1f}, p={row['p_value']:.3f}")

