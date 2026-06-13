"""Stage 1b: two-way fixed-effects event study on the matched panel.

For each outcome (log fare, log passengers) we regress on event-time indicators
relative to LCC entry, with route and quarter fixed effects and standard errors
clustered at the route level. Never-treated routes enter with all event-time
indicators set to zero, so no already-treated market serves as a control for a
later entrant. The omitted reference period is the quarter before entry.

Input:  data/processed/matched_market_panel.parquet
Output: outputs/tables/event_study_{fare,pax}_results.csv
        outputs/figures/event_study_{fare,pax}.png
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt

from src import config


def _event_time_dummies(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add event-time indicator columns, omitting tau = -1 as the reference."""
    panel = panel.copy()
    panel["Tau"] = (panel["Time_Idx"] - panel["Treatment_Time"]).clip(-config.EVENT_WINDOW, config.EVENT_WINDOW)

    taus = [t for t in range(-config.EVENT_WINDOW, config.EVENT_WINDOW + 1) if t != -1]
    cols = []
    for t in taus:
        col = f"Tau_{t}"
        panel[col] = ((panel["Never_Treated"] == 0) & (panel["Tau"] == t)).astype(int)
        cols.append(col)
    return panel, cols


def fit_event_study(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Fit the TWFE event study for one outcome and return the coefficient table."""
    panel, tau_cols = _event_time_dummies(panel)
    indexed = panel.set_index(["Market", "Time_Idx"])
    model = PanelOLS(indexed[outcome], indexed[tau_cols],
                     entity_effects=True, time_effects=True, check_rank=False)
    res = model.fit(cov_type="clustered", cluster_entity=True)
    return pd.DataFrame({
        "Coefficient": res.params,
        "Std_Error": res.std_errors,
        "P_Value": res.pvalues,
        "CI_Lower": res.params - 1.96 * res.std_errors,
        "CI_Upper": res.params + 1.96 * res.std_errors,
    })


def plot_event_study(results: pd.DataFrame, title: str, path) -> None:
    """Plot event-study coefficients with their 95% confidence band."""
    plot_df = results.copy()
    plot_df["Tau"] = [int(i.replace("Tau_", "")) for i in plot_df.index]
    plot_df.loc["ref"] = [0, 0, 0, 0, 0, -1]
    plot_df = plot_df.sort_values("Tau")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_df["Tau"], plot_df["Coefficient"], marker="o", color="b", linewidth=2)
    ax.fill_between(plot_df["Tau"], plot_df["CI_Lower"], plot_df["CI_Upper"], color="b", alpha=0.2)
    ax.axhline(0, color="red", linestyle="--")
    ax.axvline(0, color="gray", linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Quarters from LCC entry")
    ax.set_ylabel("Estimated treatment effect (ATT)")
    ax.set_xticks(range(-config.EVENT_WINDOW, config.EVENT_WINDOW + 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_event_study() -> None:
    """Run the fare and passenger event studies and save tables and figures."""
    config.make_output_dirs()
    panel = pd.read_parquet(config.PROCESSED_DIR / "matched_market_panel.parquet")
    panel["log_Fare"] = np.log(panel["Mkt_AvgFare"])
    panel["log_Pax"] = np.log(panel["Mkt_Total_Pax"])
    panel["Market"] = panel["Origin"] + "_" + panel["Dest"]

    specs = [
        ("log_Fare", "fare", "LCC entry event study: effect on log average fare"),
        ("log_Pax", "pax", "LCC entry event study: effect on log market passengers"),
    ]
    for outcome, tag, title in specs:
        results = fit_event_study(panel, outcome)
        results.to_csv(config.TABLES_DIR / f"event_study_{tag}_results.csv")
        plot_event_study(results, title, config.FIGURES_DIR / f"event_study_{tag}.png")
        print(f"{tag}: ATT(t=0) = {results.loc['Tau_0', 'Coefficient']:.4f}")


if __name__ == "__main__":
    run_event_study()
