"""Stage 1a: propensity score matching of treated to never-treated routes.

Treated routes are markets that first see an LCC in 2016 or later, so a full 2015
pre-entry baseline is observable. Each is matched 1:1 to a never-treated control
on the logit propensity score and the six pre-entry covariates (Mahalanobis
distance), subject to a 0.2 standard-deviation caliper on the logit score. The
matched routes define the panel used by every later stage.

Input:  data/processed/market_quarter_panel.parquet
Output: data/processed/matched_market_panel.parquet
        outputs/figures/covariate_balance.png
"""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src import config

COVARIATES = ["log_Pax", "log_Fare", "log_Dist", "HHI", "N_Carriers", "Delta_Fare_Pre"]


def standardised_mean_diff(df: pd.DataFrame, covariates: list[str]) -> dict[str, float]:
    """Absolute standardised mean difference between treated and control groups."""
    treated = df[df["Treated"] == 1]
    control = df[df["Treated"] == 0]
    smd = {}
    for cov in covariates:
        pooled_sd = np.sqrt((treated[cov].var() + control[cov].var()) / 2)
        smd[cov] = np.abs(treated[cov].mean() - control[cov].mean()) / (pooled_sd + 1e-10)
    return smd


def build_baseline_cross_section(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per eligible route with its 2015 pre-entry covariates."""
    treated = panel[(panel["Never_Treated"] == 0) & (panel["Treatment_Time"] >= 4)][["Origin", "Dest"]]
    treated = treated.drop_duplicates()
    treated["Treated"] = 1
    control = panel[panel["Never_Treated"] == 1][["Origin", "Dest"]].drop_duplicates()
    control["Treated"] = 0

    eligible = pd.concat([treated, control])
    df = panel.merge(eligible, on=["Origin", "Dest"], how="inner")
    df = df[df["Year"] == config.BASELINE_YEAR].copy()
    df["log_Pax"] = np.log(df["Mkt_Total_Pax"])
    df["log_Fare"] = np.log(df["Mkt_AvgFare"])
    df["log_Dist"] = np.log(df["MktDistance"])

    cross = df.groupby(["Origin", "Dest"], as_index=False).agg(
        log_Pax=("log_Pax", "mean"),
        log_Fare=("log_Fare", "mean"),
        log_Dist=("log_Dist", "mean"),
        HHI=("HHI", "mean"),
        N_Carriers=("N_Carriers", "mean"),
        Treated=("Treated", "max"),
    )

    # Pre-entry fare trajectory: log fare change across the 2015 baseline year.
    q1 = df[df["Quarter"] == 1].groupby(["Origin", "Dest"], as_index=False).agg(Fare_Q1=("log_Fare", "mean"))
    q4 = df[df["Quarter"] == 4].groupby(["Origin", "Dest"], as_index=False).agg(Fare_Q4=("log_Fare", "mean"))
    trend = q1.merge(q4, on=["Origin", "Dest"])
    trend["Delta_Fare_Pre"] = trend["Fare_Q4"] - trend["Fare_Q1"]

    cross = cross.merge(trend[["Origin", "Dest", "Delta_Fare_Pre"]], on=["Origin", "Dest"])
    return cross.dropna(subset=COVARIATES).reset_index(drop=True)


def estimate_propensity(cross: pd.DataFrame) -> pd.DataFrame:
    """Fit the logistic propensity model and store the logit score."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(cross[COVARIATES])
    model = LogisticRegression(solver="liblinear", max_iter=1000)
    model.fit(x_scaled, cross["Treated"])
    cross["ps"] = model.predict_proba(x_scaled)[:, 1]
    cross["logit_ps"] = logit(np.clip(cross["ps"], 1e-10, 1 - 1e-10))
    return cross


def match_nearest_neighbour(cross: pd.DataFrame) -> pd.DataFrame:
    """1:1 Mahalanobis matching on logit-PS plus covariates, with a PS caliper."""
    caliper = 0.2 * cross["logit_ps"].std()
    treated_idx = cross[cross["Treated"] == 1].index
    control_idx = cross[cross["Treated"] == 0].index

    match_cols = ["logit_ps"] + COVARIATES
    x_scaled = StandardScaler().fit_transform(cross[match_cols])
    cov_inv = np.linalg.inv(np.cov(x_scaled.T))

    nn = NearestNeighbors(n_neighbors=1, metric="mahalanobis", metric_params={"VI": cov_inv})
    nn.fit(x_scaled[control_idx])
    distances, neighbours = nn.kneighbors(x_scaled[treated_idx])

    matches = pd.DataFrame({
        "treated_idx": treated_idx,
        "control_idx": control_idx[neighbours[:, 0]],
        "distance": distances[:, 0],
    })

    # Drop pairs whose propensity scores differ by more than the caliper.
    ps_gap = np.abs(
        cross.loc[matches["treated_idx"], "logit_ps"].values
        - cross.loc[matches["control_idx"], "logit_ps"].values
    )
    matches = matches[ps_gap <= caliper]
    # Match without replacement: keep the closest pair for each control.
    matches = matches.sort_values("distance").drop_duplicates("control_idx")
    return matches


def plot_love(balance: pd.DataFrame) -> None:
    """Love plot of pre- and post-match standardised mean differences."""
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(balance))
    ax.scatter(balance["Unmatched"], y, color="red", marker="x", s=100, label="Unmatched")
    ax.scatter(balance["Matched"], y, color="blue", marker="o", s=100, label="Matched")
    ax.axvline(0.1, color="gray", linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(balance["Covariate"])
    ax.set_xlabel("Absolute standardised mean difference")
    ax.set_title("Covariate balance before and after matching")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "covariate_balance.png", dpi=200)
    plt.close(fig)


def run_matching() -> pd.DataFrame:
    """Run Stage 1a end to end and write the matched panel."""
    config.make_output_dirs()
    panel = pd.read_parquet(config.PROCESSED_DIR / "market_quarter_panel.parquet")

    cross = build_baseline_cross_section(panel)
    print(f"Eligible routes: {len(cross)} ({int(cross['Treated'].sum())} treated)")

    cross = estimate_propensity(cross)
    matches = match_nearest_neighbour(cross)
    matched_idx = list(matches["treated_idx"]) + list(matches["control_idx"])
    matched_cross = cross.loc[matched_idx]
    print(f"Matched pairs: {len(matches)} ({len(matched_cross)} routes)")

    balance = pd.DataFrame({
        "Covariate": COVARIATES,
        "Unmatched": [standardised_mean_diff(cross, COVARIATES)[c] for c in COVARIATES],
        "Matched": [standardised_mean_diff(matched_cross, COVARIATES)[c] for c in COVARIATES],
    })
    print("\nCovariate balance (ASMD):")
    print(balance.to_string(index=False))
    plot_love(balance)

    matched_routes = matched_cross[["Origin", "Dest"]].drop_duplicates()
    matched_panel = panel.merge(matched_routes, on=["Origin", "Dest"], how="inner")
    out_path = config.PROCESSED_DIR / "matched_market_panel.parquet"
    matched_panel.to_parquet(out_path, index=False)
    print(f"\nSaved matched panel: {matched_panel.shape}, {out_path}")
    return matched_panel


if __name__ == "__main__":
    run_matching()
