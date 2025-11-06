#!/usr/bin/env python3
"""
Event-time reaction analysis around Dividend = 0 shocks
- Event-time line chart (A vs B) for tau = 0, +1, +2 with error bars
- A-only: impact by Market (bar), optional underreaction
- B-only: impact vs ProShare% (scatter + trend)
- Hypothesis tests reported in console
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["figure.figsize"] = (16, 8)
plt.rcParams["axes.grid"] = True


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_a = pd.read_csv("Experiment_A_Trading_Data.csv")
    df_b = pd.read_csv("Experiment_B_Trading_Data.csv")
    df_a["Experiment"] = "Choice A"
    df_b["Experiment"] = "Choice B"
    return df_a, df_b


def compute_bubble(df: pd.DataFrame, is_choice_a: bool) -> pd.DataFrame:
    keys = ["Session", "Period"] if not is_choice_a else ["Session", "Market", "Period"]
    grouped = df.groupby(keys).agg({
        "LastPrice": "mean",
        "Fundamental": "first",
        "Dividend": "first",
    }).reset_index()
    grouped["BubblePct"] = (grouped["LastPrice"] - grouped["Fundamental"]) / grouped["Fundamental"] * 100
    return grouped


def build_event_time_table(df: pd.DataFrame, is_choice_a: bool) -> pd.DataFrame:
    # Filter dividend = 0
    shocked = df[df["Dividend"] == 0].copy()
    key_cols = ["Session", "Period"] if not is_choice_a else ["Session", "Market", "Period"]

    # tau=0
    shocked = shocked.rename(columns={"BubblePct": "Shock_t"})

    # tau=+1
    next_keys = ["Session", "Period"] if not is_choice_a else ["Session", "Market", "Period"]
    df_plus1 = df.copy()
    df_plus1["Period"] = df_plus1["Period"] - 1  # so that merge aligns t with t+1
    df_plus1 = df_plus1[key_cols + ["BubblePct"]].rename(columns={"BubblePct": "Shock_t_plus_1"})

    # tau=+2
    df_plus2 = df.copy()
    df_plus2["Period"] = df_plus2["Period"] - 2
    df_plus2 = df_plus2[key_cols + ["BubblePct"]].rename(columns={"BubblePct": "Shock_t_plus_2"})

    out = shocked.merge(df_plus1, on=key_cols, how="left").merge(df_plus2, on=key_cols, how="left")
    return out


def event_time_line_a_vs_b(df_a_evt: pd.DataFrame, df_b_evt: pd.DataFrame) -> None:
    def mean_se(x: pd.Series) -> tuple[float, float]:
        x = x.dropna()
        if len(x) == 0:
            return np.nan, np.nan
        se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
        return x.mean(), se

    stats_a = {
        "tau0": mean_se(df_a_evt["Shock_t"]),
        "tau1": mean_se(df_a_evt["Shock_t_plus_1"]),
        "tau2": mean_se(df_a_evt["Shock_t_plus_2"]),
    }
    stats_b = {
        "tau0": mean_se(df_b_evt["Shock_t"]),
        "tau1": mean_se(df_b_evt["Shock_t_plus_1"]),
        "tau2": mean_se(df_b_evt["Shock_t_plus_2"]),
    }

    x = np.array([0, 1, 2])
    means_a = np.array([stats_a["tau0"][0], stats_a["tau1"][0], stats_a["tau2"][0]])
    ses_a = np.array([stats_a["tau0"][1], stats_a["tau1"][1], stats_a["tau2"][1]])
    means_b = np.array([stats_b["tau0"][0], stats_b["tau1"][0], stats_b["tau2"][0]])
    ses_b = np.array([stats_b["tau0"][1], stats_b["tau1"][1], stats_b["tau2"][1]])

    fig, ax = plt.subplots()
    ax.errorbar(x, means_a, yerr=ses_a, color="red", marker="o", linewidth=2, label="Choice A")
    ax.errorbar(x, means_b, yerr=ses_b, color="blue", marker="s", linewidth=2, label="Choice B")
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["τ = 0", "τ = +1", "τ = +2"]) 
    ax.set_ylabel("Average Bubble %")
    ax.set_title("Event-time Reaction to Dividend = 0 (A vs B)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("event_time_line_A_vs_B.png", dpi=300, bbox_inches="tight")
    plt.close()


def a_only_market_impact(df_a_evt: pd.DataFrame) -> None:
    # Impact metrics per market
    impact = df_a_evt.groupby("Market")["Shock_t"].mean()
    underreaction = (df_a_evt["Shock_t"] - df_a_evt["Shock_t_plus_2"]).groupby(df_a_evt["Market"]).mean()

    fig, ax = plt.subplots()
    ax.bar(impact.index - 0.2, impact.values, width=0.4, label="Shock impact τ=0", color="darkorange", alpha=0.8)
    ax.bar(underreaction.index + 0.2, underreaction.values, width=0.4, label="Underreaction (t − t+2)", color="teal", alpha=0.8)
    ax.set_xlabel("Market")
    ax.set_ylabel("Bubble %")
    ax.set_title("Choice A: Shock Impact by Market")
    ax.legend()
    plt.tight_layout()
    plt.savefig("event_time_A_market_impact.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Slope test for decreasing impact with market index
    m_idx = impact.index.values.astype(float)
    slope, intercept, r, p, se = stats.linregress(m_idx, impact.values)
    print(f"A-only Impact slope: {slope:.3f} per market, R^2={r**2:.2f}, p(two-sided)={p:.4g}")


def b_only_proshare_scatter(df_b_evt: pd.DataFrame, df_b_raw: pd.DataFrame) -> None:
    # ProShare per session
    proshare = df_b_raw.groupby("Session")["ProShare"].first()
    impact_by_session = df_b_evt.groupby("Session")["Shock_t"].mean()
    common = impact_by_session.index.intersection(proshare.index)
    x = proshare.loc[common].values * 100  # percentage
    y = impact_by_session.loc[common].values

    fig, ax = plt.subplots()
    ax.scatter(x, y, color="purple", s=60, alpha=0.8)
    # trend line
    if len(x) >= 2:
        slope, intercept, r, p, se = stats.linregress(x, y)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, intercept + slope * xs, color="black", linestyle="--", label=f"Trend (R^2={r**2:.2f})")
        print(f"B-only Impact vs ProShare: slope={slope:.3f}, R^2={r**2:.2f}, p(two-sided)={p:.4g}")
    ax.set_xlabel("Professional Share %")
    ax.set_ylabel("Shock impact at τ=0 (Bubble %)")
    ax.set_title("Choice B: Shock Impact vs Professional Share")
    ax.legend()
    plt.tight_layout()
    plt.savefig("event_time_B_impact_vs_proshare.png", dpi=300, bbox_inches="tight")
    plt.close()


def hypothesis_tests(df_a_evt: pd.DataFrame, df_b_evt: pd.DataFrame) -> None:
    # H0 (Reaction size): mean Shock_t = 0 (one-sample, two-sided report)
    for name, arr in {
        "Choice A τ=0": df_a_evt["Shock_t"].dropna(),
        "Choice B τ=0": df_b_evt["Shock_t"].dropna(),
        "Choice A τ=+1": df_a_evt["Shock_t_plus_1"].dropna(),
        "Choice B τ=+1": df_b_evt["Shock_t_plus_1"].dropna(),
        "Choice A τ=+2": df_a_evt["Shock_t_plus_2"].dropna(),
        "Choice B τ=+2": df_b_evt["Shock_t_plus_2"].dropna(),
    }.items():
        if len(arr) >= 2:
            t, p = stats.ttest_1samp(arr, 0.0)
            print(f"One-sample t-test [{name}]: mean={arr.mean():.2f}%, t={t:.3f}, p(two-sided)={p:.4g}, N={len(arr)}")

    # H0 (A vs B reaction): equal means at τ=0 (Welch t)
    a0 = df_a_evt["Shock_t"].dropna()
    b0 = df_b_evt["Shock_t"].dropna()
    if len(a0) >= 2 and len(b0) >= 2:
        t_welch, p_welch = stats.ttest_ind(a0, b0, equal_var=False)
        u_stat, p_u = stats.mannwhitneyu(a0, b0, alternative="two-sided")
        print(f"Welch t-test A vs B (τ=0): t={t_welch:.3f}, p={p_welch:.4g}; Mann-Whitney U: U={u_stat:.0f}, p={p_u:.4g}")

    # H0 (Persistence): Shock_t+1 = 0, Shock_t+2 = 0 already covered above
    # Also test mean reversion: Shock_t - Shock_t+2 > 0
    a_rev = (df_a_evt["Shock_t"] - df_a_evt["Shock_t_plus_2"]).dropna()
    b_rev = (df_b_evt["Shock_t"] - df_b_evt["Shock_t_plus_2"]).dropna()
    if len(a_rev) >= 2:
        t, p = stats.ttest_1samp(a_rev, 0.0)
        p_one = p/2 if t > 0 else 1 - p/2
        print(f"Mean reversion test A (Shock_t − Shock_t+2 > 0): t={t:.3f}, p(one-sided)={p_one:.4g}, N={len(a_rev)}")
    if len(b_rev) >= 2:
        t, p = stats.ttest_1samp(b_rev, 0.0)
        p_one = p/2 if t > 0 else 1 - p/2
        print(f"Mean reversion test B (Shock_t − Shock_t+2 > 0): t={t:.3f}, p(one-sided)={p_one:.4g}, N={len(b_rev)}")


def main():
    df_a_raw, df_b_raw = load_data()
    df_a = compute_bubble(df_a_raw, is_choice_a=True)
    df_b = compute_bubble(df_b_raw, is_choice_a=False)

    df_a_evt = build_event_time_table(df_a, is_choice_a=True)
    df_b_evt = build_event_time_table(df_b, is_choice_a=False)

    # Plots
    event_time_line_a_vs_b(df_a_evt, df_b_evt)
    a_only_market_impact(df_a_evt)
    b_only_proshare_scatter(df_b_evt, df_b_raw)

    # Tests
    hypothesis_tests(df_a_evt, df_b_evt)

    print("\n✓ Event-time reaction analysis complete. Saved plots:\n - event_time_line_A_vs_B.png\n - event_time_A_market_impact.png\n - event_time_B_impact_vs_proshare.png")


if __name__ == "__main__":
    main()
