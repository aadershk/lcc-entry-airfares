"""Stage 2: conditional average treatment effects with a causal forest.

A route-level cross-section is built from the matched panel using pre/post
windows aligned to each route's own entry quarter (never-treated routes use the
median entry quarter as a pseudo-anchor). A CausalForestDML is fitted for the
fare and passenger DiD outcomes, giving a CATE per route. The systematic part of
that heterogeneity is summarised by group ATEs across quartiles, by the forest's
variable importance, and by a best-linear-projection test.

Input:  data/processed/matched_market_panel.parquet
Output: data/processed/cate_estimates.parquet
        outputs/tables/{variable_importance,cate_quartile_summary,blp_results,propensity_scores}.csv
        outputs/figures/{cate_distribution,cate_by_hhi_quartile,cate_by_distance,variable_importance}.png
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

from src import config

MODERATORS = ["HHI_pre", "log_Dist", "log_Pax_pre", "N_Carriers_pre"]
CONTROLS = ["log_Fare_pre"] + MODERATORS
PANEL_END = 19  # last Time_Idx in the 2015 Q1 - 2019 Q4 window

QUARTILE_DIMS = [
    ("HHI_pre", "HHI_quartile", "HHI", ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]),
    ("log_Dist", "Dist_quartile", "Distance", ["Q1 (Short)", "Q2", "Q3", "Q4 (Long)"]),
    ("log_Pax_pre", "Pax_quartile", "Passenger Density", ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]),
]


def _route_record(route: pd.DataFrame, treated: int, pre_start: int, pre_end: int,
                   post_start: int, post_end: int) -> dict | None:
    """Pre/post means for one route, or None if a window has no observations."""
    pre = route[(route["Time_Idx"] >= pre_start) & (route["Time_Idx"] <= pre_end)]
    post = route[(route["Time_Idx"] >= post_start) & (route["Time_Idx"] <= post_end)]
    if pre.empty or post.empty:
        return None
    return {
        "Origin": route["Origin"].iloc[0], "Dest": route["Dest"].iloc[0], "T": treated,
        "log_Fare_pre": pre["log_Fare"].mean(), "log_Fare_post": post["log_Fare"].mean(),
        "log_Pax_pre": pre["log_Pax"].mean(), "log_Pax_post": post["log_Pax"].mean(),
        "log_Dist": pre["log_Dist"].mean(), "HHI_pre": pre["HHI"].mean(),
        "N_Carriers_pre": pre["N_Carriers"].mean(),
    }


def build_cross_section(matched: pd.DataFrame) -> pd.DataFrame:
    """Construct the route-level pre/post DiD cross-section with aligned windows.

    Treated routes are processed before never-treated controls, matching the order
    used elsewhere so the cross-fitting folds are reproducible.
    """
    matched = matched.copy()
    matched["log_Fare"] = np.log(matched["Mkt_AvgFare"].clip(lower=0.01))
    matched["log_Pax"] = np.log(matched["Mkt_Total_Pax"].clip(lower=1))
    matched["log_Dist"] = np.log(matched["MktDistance"].clip(lower=1))

    groups = dict(tuple(matched.groupby(["Origin", "Dest"], sort=False)))
    treated_routes = matched[matched["Never_Treated"] == 0][["Origin", "Dest"]].drop_duplicates()
    control_routes = matched[matched["Never_Treated"] == 1][["Origin", "Dest"]].drop_duplicates()

    pseudo_tt = int(round(
        matched[matched["Never_Treated"] == 0].groupby(["Origin", "Dest"])["Treatment_Time"].first().median()))
    print(f"Pseudo-treatment-time for controls: {pseudo_tt}")

    records = []
    for od in treated_routes.itertuples(index=False):
        route = groups[(od.Origin, od.Dest)]
        tt = int(route["Treatment_Time"].iloc[0])
        rec = _route_record(route, 1, max(0, tt - 4), tt - 1, tt, min(PANEL_END, tt + 4))
        if rec is not None:
            records.append(rec)
    for od in control_routes.itertuples(index=False):
        route = groups[(od.Origin, od.Dest)]
        rec = _route_record(route, 0, max(0, pseudo_tt - 4), pseudo_tt - 1,
                            pseudo_tt, min(PANEL_END, pseudo_tt + 4))
        if rec is not None:
            records.append(rec)

    cross = pd.DataFrame(records)
    cross["Y_fare"] = cross["log_Fare_post"] - cross["log_Fare_pre"]
    cross["Y_pax"] = cross["log_Pax_post"] - cross["log_Pax_pre"]
    cross = cross.dropna().reset_index(drop=True)
    print(f"Cross-section: {len(cross)} routes ({int(cross['T'].sum())} treated, "
          f"{int((cross['T'] == 0).sum())} control)")
    return cross


def _new_forest() -> CausalForestDML:
    """CausalForestDML with the configuration used throughout the analysis."""
    return CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=config.SEED),
        model_t=LogisticRegressionCV(cv=5, max_iter=1000, random_state=config.SEED),
        discrete_treatment=True,
        n_estimators=4000,
        min_samples_leaf=20,
        max_depth=None,
        criterion="het",
        cv=5,
        random_state=config.SEED,
        verbose=0,
    )


def fit_cate(cross: pd.DataFrame, outcome: str) -> tuple[CausalForestDML, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the forest for one outcome and return the model and route-level CATEs."""
    x = cross[MODERATORS].values
    forest = _new_forest()
    forest.fit(cross[outcome].values, cross["T"].values, X=x, W=cross[CONTROLS].values)
    cate = forest.effect(x)
    lower, upper = forest.effect_interval(x, alpha=0.05)
    return forest, cate, lower, upper


def best_linear_projection(cross: pd.DataFrame, outcome: str, tau_hat: np.ndarray, label: str) -> dict:
    """BLP test: regress residual outcome on residual treatment and CATE interaction."""
    w = cross[CONTROLS].values
    t = cross["T"].values
    y = cross[outcome].values

    y_resid = y - cross_val_predict(
        GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=config.SEED), w, y, cv=5)
    t_resid = t - cross_val_predict(
        LogisticRegressionCV(cv=5, max_iter=1000, random_state=config.SEED),
        w, t, cv=5, method="predict_proba")[:, 1]

    design = np.column_stack([t_resid, (tau_hat - tau_hat.mean()) * t_resid])
    fit = sm.OLS(y_resid, design).fit(cov_type="HC3")
    return {
        "outcome": label,
        "theta_0": fit.params[0], "se_0": fit.bse[0], "p_0": fit.pvalues[0],
        "theta_1": fit.params[1], "se_1": fit.bse[1],
        "t_1": fit.tvalues[1], "p_1": fit.pvalues[1], "n": len(y),
    }


def assign_quartiles(cross: pd.DataFrame) -> pd.DataFrame:
    """Tag each route with its quartile on the three moderator dimensions."""
    for col, label_col, _, labels in QUARTILE_DIMS:
        cross[label_col] = pd.qcut(cross[col], 4, labels=labels)
    return cross


def quartile_summary(cross: pd.DataFrame) -> pd.DataFrame:
    """Mean CATE within each moderator quartile (group ATEs)."""
    rows = []
    for _, label_col, dim, _ in QUARTILE_DIMS:
        for quartile, grp in cross.groupby(label_col, observed=True):
            rows.append({
                "Dimension": dim, "Quartile": quartile, "N": len(grp),
                "CATE_fare_mean": grp["CATE_fare"].mean(),
                "CATE_fare_pct": (np.exp(grp["CATE_fare"].mean()) - 1) * 100,
                "CATE_fare_std": grp["CATE_fare"].std(),
                "CATE_pax_mean": grp["CATE_pax"].mean(),
                "CATE_pax_pct": (np.exp(grp["CATE_pax"].mean()) - 1) * 100,
            })
    return pd.DataFrame(rows)


def _plot_cate_distribution(cross: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, colour, title in [
        (axes[0], "CATE_fare", "steelblue", "Distribution of fare CATEs"),
        (axes[1], "CATE_pax", "darkorange", "Distribution of passenger CATEs"),
    ]:
        ax.hist(cross[col], bins=50, color=colour, edgecolor="white", alpha=0.8)
        ax.axvline(cross[col].mean(), color="darkgreen", linewidth=2,
                   label=f"Mean: {cross[col].mean():.3f}")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(f"CATE ({'log fare' if col == 'CATE_fare' else 'log pax'} change)")
        ax.set_ylabel("Number of routes")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "cate_distribution.png", dpi=200)
    plt.close(fig)


def _plot_quartile_bars(summary: pd.DataFrame, dimension: str, filename: str, colours: list[str]) -> None:
    sub = summary[summary["Dimension"] == dimension]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(sub)), sub["CATE_fare_mean"], color=colours, edgecolor="black")
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["Quartile"])
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_ylabel("Mean CATE (log fare)")
    ax.set_title(f"Fare CATE by {dimension} quartile")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def _plot_variable_importance(vi: pd.DataFrame) -> None:
    labels = {"HHI_pre": "HHI", "log_Dist": "Distance",
              "log_Pax_pre": "Pax volume", "N_Carriers_pre": "Carrier count"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title, colour in [
        (axes[0], "Importance_Fare", "Fare model", "steelblue"),
        (axes[1], "Importance_Pax", "Pax model", "darkorange"),
    ]:
        ordered = vi.sort_values(col)
        ax.barh([labels[f] for f in ordered["Feature"]], ordered[col], color=colour, edgecolor="black")
        ax.set_xlabel("Feature importance")
        ax.set_title(f"Variable importance ({title})")
        ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "variable_importance.png", dpi=200)
    plt.close(fig)


def run_causal_forest() -> pd.DataFrame:
    """Run Stage 2 end to end: fit forests, summarise, validate, and save."""
    np.random.seed(config.SEED)
    config.make_output_dirs()
    matched = pd.read_parquet(config.PROCESSED_DIR / "matched_market_panel.parquet")
    cross = build_cross_section(matched)

    forest_fare, cross["CATE_fare"], cross["CATE_fare_lb"], cross["CATE_fare_ub"] = fit_cate(cross, "Y_fare")
    forest_pax, cross["CATE_pax"], cross["CATE_pax_lb"], cross["CATE_pax_ub"] = fit_cate(cross, "Y_pax")
    print(f"Mean fare CATE: {cross['CATE_fare'].mean():.4f}, mean pax CATE: {cross['CATE_pax'].mean():.4f}")

    vi = pd.DataFrame({
        "Feature": MODERATORS,
        "Importance_Fare": forest_fare.feature_importances_,
        "Importance_Pax": forest_pax.feature_importances_,
    }).sort_values("Importance_Fare", ascending=False)
    vi.to_csv(config.TABLES_DIR / "variable_importance.csv", index=False)

    cross = assign_quartiles(cross)
    summary = quartile_summary(cross)
    summary.to_csv(config.TABLES_DIR / "cate_quartile_summary.csv", index=False)

    blp = pd.DataFrame([
        best_linear_projection(cross, "Y_fare", forest_fare.effect(cross[MODERATORS].values), "log(Fare)"),
        best_linear_projection(cross, "Y_pax", forest_pax.effect(cross[MODERATORS].values), "log(Pax)"),
    ])
    blp.to_csv(config.TABLES_DIR / "blp_results.csv", index=False)
    print(blp[["outcome", "theta_0", "theta_1", "t_1", "p_1"]].to_string(index=False))

    _save_propensity_scores(cross)
    _plot_cate_distribution(cross)
    _plot_quartile_bars(summary, "HHI", "cate_by_hhi_quartile.png",
                        ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"])
    _plot_quartile_bars(summary, "Distance", "cate_by_distance.png",
                        ["#3498db", "#2980b9", "#1abc9c", "#16a085"])
    _plot_variable_importance(vi)

    cross.to_parquet(config.PROCESSED_DIR / "cate_estimates.parquet", index=False)
    print(f"Saved CATE estimates: {config.PROCESSED_DIR / 'cate_estimates.parquet'}")
    return cross


def _save_propensity_scores(cross: pd.DataFrame) -> None:
    """Propensity scores on the matched cross-section, for the overlap figure."""
    model = LogisticRegressionCV(cv=5, max_iter=1000, random_state=config.SEED)
    model.fit(cross[CONTROLS].values, cross["T"].values)
    ps = model.predict_proba(cross[CONTROLS].values)[:, 1]
    clipped = np.clip(ps, 1e-6, 1 - 1e-6)
    pd.DataFrame({
        "T": cross["T"].values,
        "propensity_score": ps,
        "logit_ps": np.log(clipped / (1 - clipped)),
    }).to_csv(config.TABLES_DIR / "propensity_scores.csv", index=False)


if __name__ == "__main__":
    run_causal_forest()
