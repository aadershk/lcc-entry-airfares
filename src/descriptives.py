"""Descriptive figures for the data and the raw treatment dynamics.

These plots give a feel for the sample before any modelling: the spread of LCC
entry over time, how treated routes sit in the fare and distance distributions,
the raw fare path around entry, and the propensity-score overlap after matching.

Input:  data/processed/{market_quarter_panel,matched_market_panel,cate_estimates}.parquet
        outputs/tables/propensity_scores.csv
Output: outputs/figures/{data_overview,fare_event_time,propensity_overlap}.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config

TREAT_COLOUR = "#2a3087"
CONTROL_COLOUR = "#af3a2a"


def data_overview() -> None:
    """Four-panel overview: entry timing, fare and distance spreads, fare-distance."""
    panel = pd.read_parquet(config.PROCESSED_DIR / "market_quarter_panel.parquet")
    cate = pd.read_parquet(config.PROCESSED_DIR / "cate_estimates.parquet")
    matched = pd.read_parquet(config.PROCESSED_DIR / "matched_market_panel.parquet")

    baseline = panel[panel["Year"] == config.BASELINE_YEAR]
    full = baseline.groupby(["Origin", "Dest"], as_index=False).agg(
        log_Fare=("Mkt_AvgFare", lambda s: np.log(s).mean()),
        log_Dist=("MktDistance", lambda s: np.log(s).mean()),
    )
    treated = cate[cate["T"] == 1]
    control = cate[cate["T"] == 0]

    entry = matched[matched["Never_Treated"] == 0].groupby(["Origin", "Dest"])["Treatment_Time"].first()
    counts = entry.value_counts().sort_index()
    labels = {row.Time_Idx: f"{int(row.Year)}Q{int(row.Quarter)}"
              for row in matched[["Time_Idx", "Year", "Quarter"]].drop_duplicates().itertuples()}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.bar(range(len(counts)), counts.values, color=TREAT_COLOUR, alpha=0.85)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([labels.get(int(i), str(int(i))) for i in counts.index], rotation=90, fontsize=7)
    ax.set_ylabel("Matched treated routes")
    ax.set_title("(a) LCC entry events by quarter")

    ax = axes[0, 1]
    bins = np.linspace(4.5, 7.0, 40)
    ax.hist(full["log_Fare"], bins=bins, density=True, alpha=0.5, color="grey", label="All markets")
    ax.hist(treated["log_Fare_pre"], bins=bins, density=True, alpha=0.6, color=TREAT_COLOUR, label="Matched treated")
    ax.set_xlabel("Pre-entry log average fare")
    ax.set_ylabel("Density")
    ax.set_title("(b) Fare distribution")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    bins = np.linspace(4.5, 8.5, 40)
    ax.hist(full["log_Dist"], bins=bins, density=True, alpha=0.5, color="grey", label="All markets")
    ax.hist(treated["log_Dist"], bins=bins, density=True, alpha=0.6, color=TREAT_COLOUR, label="Matched treated")
    ax.set_xlabel("Log route distance (miles)")
    ax.set_ylabel("Density")
    ax.set_title("(c) Route-distance distribution")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.scatter(control["log_Dist"], control["log_Fare_pre"], s=8, alpha=0.35,
               color=CONTROL_COLOUR, label="Matched control", edgecolors="none")
    ax.scatter(treated["log_Dist"], treated["log_Fare_pre"], s=8, alpha=0.35,
               color=TREAT_COLOUR, label="Matched treated", edgecolors="none")
    ax.set_xlabel("Log route distance (miles)")
    ax.set_ylabel("Pre-entry log average fare")
    ax.set_title("(d) Fare versus distance")
    ax.legend(fontsize=8, markerscale=2)

    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "data_overview.png", dpi=200)
    plt.close(fig)


def fare_event_time() -> None:
    """Raw mean log fare on treated routes by quarter relative to entry."""
    matched = pd.read_parquet(config.PROCESSED_DIR / "matched_market_panel.parquet")
    matched["log_Fare"] = np.log(matched["Mkt_AvgFare"].clip(lower=0.01))
    matched["EventTime"] = matched["Time_Idx"] - matched["Treatment_Time"]

    treated = matched[matched["Never_Treated"] == 0]
    window = treated[(treated["EventTime"] >= -config.EVENT_WINDOW) & (treated["EventTime"] <= config.EVENT_WINDOW)]
    means = window.groupby("EventTime")["log_Fare"].mean()
    sems = window.groupby("EventTime")["log_Fare"].sem()
    control_mean = matched[matched["Never_Treated"] == 1]["log_Fare"].mean()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(means.index, means.values, marker="o", color=TREAT_COLOUR, linewidth=2, label="Treated routes (raw mean)")
    ax.fill_between(means.index, means - 1.96 * sems, means + 1.96 * sems, color=TREAT_COLOUR, alpha=0.15)
    ax.axhline(control_mean, color=CONTROL_COLOUR, linestyle="--", linewidth=1.6, label="Never-treated mean")
    ax.axvline(0, color="grey", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Quarters relative to LCC entry")
    ax.set_ylabel("Mean log average fare")
    ax.set_title("Raw fare trajectory of treated routes around LCC entry")
    ax.set_xticks(range(-config.EVENT_WINDOW, config.EVENT_WINDOW + 1))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fare_event_time.png", dpi=200)
    plt.close(fig)


def propensity_overlap() -> None:
    """Logit propensity-score distributions for treated and control after matching."""
    scores = pd.read_csv(config.TABLES_DIR / "propensity_scores.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(scores["logit_ps"].min(), scores["logit_ps"].max(), 40)
    ax.hist(scores.loc[scores["T"] == 1, "logit_ps"], bins=bins, alpha=0.6,
            color=TREAT_COLOUR, label="Treated")
    ax.hist(scores.loc[scores["T"] == 0, "logit_ps"], bins=bins, alpha=0.6,
            color=CONTROL_COLOUR, label="Never-treated control")
    ax.set_xlabel("Logit propensity score")
    ax.set_ylabel("Number of routes")
    ax.set_title("Propensity-score overlap after matching")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "propensity_overlap.png", dpi=200)
    plt.close(fig)


def run_descriptives() -> None:
    """Produce the descriptive figures."""
    config.make_output_dirs()
    data_overview()
    fare_event_time()
    propensity_overlap()
    print("Descriptive figures saved to", config.FIGURES_DIR)


if __name__ == "__main__":
    run_descriptives()
