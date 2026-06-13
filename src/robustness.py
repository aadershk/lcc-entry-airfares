"""Robustness checks for the Stage 1 ATT.

Three checks, each stressing a different part of the design:
  1. Placebo: shift every entry date back four quarters and re-estimate on
     pre-entry data only. A valid design should show no comparable effect.
  2. JetBlue exclusion: rebuild treatment over the four remaining carriers,
     re-match, and re-estimate. Tests sensitivity to the hybrid carrier.
  3. Distance subsample: split the matched panel at the median route distance
     and estimate separately, as an independent check on the distance gradient.

Input:  data/processed/matched_market_panel.parquet, data/interim/db1b_collapsed.parquet
Output: outputs/tables/robustness_*.csv, outputs/figures/robustness_*.png
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt

from src import config
from src import matching
from src.event_study import fit_event_study
from src.data.build_panel import _add_treatment_timing


def _twfe(panel: pd.DataFrame, tau_cols: list[str], outcome: str) -> pd.DataFrame:
    """TWFE estimate with clustered errors for a prepared panel and dummy set."""
    indexed = panel.set_index(["Market", "Time_Idx"])
    res = PanelOLS(indexed[outcome], indexed[tau_cols],
                   entity_effects=True, time_effects=True, check_rank=False).fit(
        cov_type="clustered", cluster_entity=True)
    return pd.DataFrame({
        "Coefficient": res.params, "Std_Error": res.std_errors, "P_Value": res.pvalues,
        "CI_Lower": res.params - 1.96 * res.std_errors,
        "CI_Upper": res.params + 1.96 * res.std_errors,
    })


def placebo_test() -> pd.DataFrame:
    """Shift treatment back four quarters and estimate on pre-entry data only."""
    matched = pd.read_parquet(config.PROCESSED_DIR / "matched_market_panel.parquet")
    matched["log_Fare"] = np.log(matched["Mkt_AvgFare"].clip(lower=0.01))
    matched["Market"] = matched["Origin"] + "_" + matched["Dest"]

    treated = matched[(matched["Never_Treated"] == 0) & (matched["Time_Idx"] < matched["Treatment_Time"])].copy()
    treated["Placebo_TT"] = treated["Treatment_Time"] - 4
    keep = treated.groupby(["Origin", "Dest"])["Placebo_TT"].first()
    keep = keep[keep >= 1].index
    treated = treated.set_index(["Origin", "Dest"]).loc[keep].reset_index()

    controls = matched[matched["Never_Treated"] == 1].copy()
    controls["Placebo_TT"] = treated.groupby(["Origin", "Dest"])["Placebo_TT"].first().median()

    placebo = pd.concat([treated, controls], ignore_index=True)
    placebo["Tau"] = (placebo["Time_Idx"] - placebo["Placebo_TT"]).clip(-6, 6)
    tau_cols = []
    for t in [x for x in range(-6, 7) if x != -1]:
        col = f"Tau_{t}"
        placebo[col] = ((placebo["Never_Treated"] == 0) & (placebo["Tau"] == t)).astype(int)
        if placebo[col].sum() > 5:
            tau_cols.append(col)

    results = _twfe(placebo, tau_cols, "log_Fare")
    results.to_csv(config.TABLES_DIR / "robustness_placebo_fare_results.csv")
    _plot_single(results, "Placebo test: false treatment dates (4-quarter backward shift)",
                 "Quarters from placebo entry", "robustness_placebo_event_study.png",
                 xticks=range(-6, 7))
    print(f"Placebo ATT(t=0): {results.loc['Tau_0', 'Coefficient']:.4f}")
    return results


def jetblue_exclusion() -> pd.DataFrame:
    """Rebuild treatment without JetBlue, re-match, and re-estimate the fare ATT."""
    panel = pd.read_parquet(config.PROCESSED_DIR / "market_quarter_panel.parquet")
    db1b = pd.read_parquet(config.INTERIM_DIR / "db1b_collapsed.parquet").rename(columns={"OpCarrier": "Carrier"})

    carriers = [c for c in config.LCC_CARRIERS if c != "B6"]
    db1b["Is_LCC"] = db1b["Carrier"].isin(carriers).astype(int)
    presence = db1b.groupby(["Year", "Quarter", "Origin", "Dest"], as_index=False).agg(LCC_Present=("Is_LCC", "max"))

    panel = panel.drop(columns=["LCC_Present", "Treatment_Time", "Never_Treated", "Post_Treatment"], errors="ignore")
    panel = panel.merge(presence, on=["Year", "Quarter", "Origin", "Dest"], how="left")
    panel["LCC_Present"] = panel["LCC_Present"].fillna(0).astype(int)
    panel = _add_treatment_timing(panel)

    cross = matching.build_baseline_cross_section(panel)
    cross = matching.estimate_propensity(cross)
    matches = matching.match_nearest_neighbour(cross)
    matched_routes = cross.loc[list(matches["treated_idx"]) + list(matches["control_idx"]), ["Origin", "Dest"]]
    matched = panel.merge(matched_routes.drop_duplicates(), on=["Origin", "Dest"], how="inner")
    matched["log_Fare"] = np.log(matched["Mkt_AvgFare"].clip(lower=0.01))
    matched["Market"] = matched["Origin"] + "_" + matched["Dest"]

    results = fit_event_study(matched, "log_Fare")
    results.to_csv(config.TABLES_DIR / "robustness_no_jetblue_fare_results.csv")

    baseline = pd.read_csv(config.TABLES_DIR / "event_study_fare_results.csv", index_col=0)
    _plot_comparison({"Baseline (all LCCs)": baseline, "No JetBlue": results},
                     "Event study: baseline versus JetBlue excluded",
                     "robustness_jetblue_comparison.png")
    print(f"No-JetBlue ATT(t=0): {results.loc['Tau_0', 'Coefficient']:.4f}")
    return results


def distance_subsample() -> dict[str, pd.DataFrame]:
    """Split the matched panel at the median distance and estimate each half."""
    matched = pd.read_parquet(config.PROCESSED_DIR / "matched_market_panel.parquet")
    matched["log_Fare"] = np.log(matched["Mkt_AvgFare"].clip(lower=0.01))
    matched["Market"] = matched["Origin"] + "_" + matched["Dest"]

    route_dist = matched.groupby(["Origin", "Dest"])["MktDistance"].first()
    median_dist = route_dist.median()
    halves = {
        "short_haul": route_dist[route_dist <= median_dist].index,
        "long_haul": route_dist[route_dist > median_dist].index,
    }

    results = {}
    for tag, routes in halves.items():
        keys = pd.MultiIndex.from_tuples(routes, names=["Origin", "Dest"])
        sub = matched.set_index(["Origin", "Dest"]).loc[keys].reset_index()
        sub["Tau"] = (sub["Time_Idx"] - sub["Treatment_Time"]).clip(-config.EVENT_WINDOW, config.EVENT_WINDOW)
        tau_cols = []
        for t in [x for x in range(-config.EVENT_WINDOW, config.EVENT_WINDOW + 1) if x != -1]:
            col = f"Tau_{t}"
            sub[col] = ((sub["Never_Treated"] == 0) & (sub["Tau"] == t)).astype(int)
            tau_cols.append(col)
        res = _twfe(sub, tau_cols, "log_Fare")
        res.to_csv(config.TABLES_DIR / f"robustness_{tag}_fare_results.csv")
        results[tag] = res
        print(f"{tag} ATT(t=0): {res.loc['Tau_0', 'Coefficient']:.4f}")

    _plot_comparison({"Short-haul": results["short_haul"], "Long-haul": results["long_haul"]},
                     "Event study by route distance: short-haul versus long-haul",
                     "robustness_distance_subsample.png")
    return results


def _coef_path(results: pd.DataFrame) -> pd.DataFrame:
    path = results.copy()
    path["Tau"] = [int(i.replace("Tau_", "")) for i in path.index]
    ref = pd.DataFrame({"Coefficient": [0], "CI_Lower": [0], "CI_Upper": [0], "Tau": [-1]})
    return pd.concat([path, ref]).sort_values("Tau")


def _plot_single(results: pd.DataFrame, title: str, xlabel: str, filename: str, xticks) -> None:
    path = _coef_path(results)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(path["Tau"], path["Coefficient"], marker="o", color="b", linewidth=2)
    ax.fill_between(path["Tau"], path["CI_Lower"], path["CI_Upper"], color="b", alpha=0.2)
    ax.axhline(0, color="red", linestyle="--")
    ax.axvline(0, color="gray", linestyle=":")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Estimated effect on log fare")
    ax.set_xticks(list(xticks))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def _plot_comparison(series: dict[str, pd.DataFrame], title: str, filename: str) -> None:
    colours = ["#3498db", "#e74c3c"]
    markers = ["o", "s"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for (label, results), colour, marker in zip(series.items(), colours, markers):
        path = _coef_path(results)
        ax.plot(path["Tau"], path["Coefficient"], marker=marker, color=colour, linewidth=2, label=label)
        ax.fill_between(path["Tau"], path["CI_Lower"], path["CI_Upper"], color=colour, alpha=0.1)
    ax.axhline(0, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Quarters from LCC entry")
    ax.set_ylabel("Estimated ATT (log fare)")
    ax.set_xticks(range(-config.EVENT_WINDOW, config.EVENT_WINDOW + 1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def run_robustness() -> None:
    """Run all three robustness checks."""
    config.make_output_dirs()
    placebo_test()
    jetblue_exclusion()
    distance_subsample()


if __name__ == "__main__":
    run_robustness()
