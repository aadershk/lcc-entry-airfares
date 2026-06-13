"""Model-fit diagnostics for the propensity, event-study and forest models.

The treatment-effect counterfactuals are never observed, but each fitted model is
also a standard predictor whose fit can be measured on the data at hand. This
script reports the propensity model's discrimination, the within-R-squared of the
two-way fixed-effects event studies, and the out-of-fold fit of the forest's
nuisance models.

Output: outputs/tables/model_fit_diagnostics.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, brier_score_loss, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src import config
from src.matching import build_baseline_cross_section, COVARIATES
from src.event_study import _event_time_dummies
from src.causal_forest import build_cross_section, MODERATORS, CONTROLS


def propensity_fit(panel: pd.DataFrame) -> list[dict]:
    """AUC, accuracy, Brier and McFadden pseudo-R-squared for the logit."""
    cross = build_baseline_cross_section(panel)
    x = StandardScaler().fit_transform(cross[COVARIATES])
    y = cross["Treated"].values

    in_sample = LogisticRegression(solver="liblinear", max_iter=1000).fit(x, y).predict_proba(x)[:, 1]
    cv = cross_val_predict(
        LogisticRegression(solver="liblinear", max_iter=1000), x, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED), method="predict_proba")[:, 1]
    mcfadden = sm.Logit(y, sm.add_constant(x)).fit(disp=0).prsquared

    return [
        {"model": "propensity_logit", "metric": "n", "value": len(y)},
        {"model": "propensity_logit", "metric": "n_treated", "value": int(y.sum())},
        {"model": "propensity_logit", "metric": "auc_in_sample", "value": round(roc_auc_score(y, in_sample), 4)},
        {"model": "propensity_logit", "metric": "auc_cv", "value": round(roc_auc_score(y, cv), 4)},
        {"model": "propensity_logit", "metric": "accuracy_cv", "value": round(accuracy_score(y, cv >= 0.5), 4)},
        {"model": "propensity_logit", "metric": "brier_cv", "value": round(brier_score_loss(y, cv), 4)},
        {"model": "propensity_logit", "metric": "mcfadden_r2", "value": round(mcfadden, 4)},
    ]


def event_study_fit(matched: pd.DataFrame) -> list[dict]:
    """Within-R-squared of the two-way fixed-effects event studies."""
    matched = matched.copy()
    matched["log_Fare"] = np.log(matched["Mkt_AvgFare"])
    matched["log_Pax"] = np.log(matched["Mkt_Total_Pax"])
    matched["Market"] = matched["Origin"] + "_" + matched["Dest"]
    panel, tau_cols = _event_time_dummies(matched)
    indexed = panel.set_index(["Market", "Time_Idx"])

    rows = []
    for outcome, tag in [("log_Fare", "twfe_fare"), ("log_Pax", "twfe_pax")]:
        res = PanelOLS(indexed[outcome], indexed[tau_cols],
                       entity_effects=True, time_effects=True, check_rank=False).fit(
            cov_type="clustered", cluster_entity=True)
        rows.append({"model": tag, "metric": "within_r2", "value": round(res.rsquared_within, 4)})
    return rows


def nuisance_fit(matched: pd.DataFrame) -> list[dict]:
    """Out-of-fold fit of the forest's treatment and outcome nuisance models."""
    cross = build_cross_section(matched)
    w = cross[CONTROLS].values
    t = cross["T"].values

    t_pred = cross_val_predict(
        LogisticRegressionCV(cv=5, max_iter=1000, random_state=config.SEED), w, t,
        cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED), method="predict_proba")[:, 1]
    rows = [{"model": "dml_treatment", "metric": "auc_oof", "value": round(roc_auc_score(t, t_pred), 4)}]

    folds = KFold(5, shuffle=True, random_state=config.SEED)
    for outcome, tag in [("Y_fare", "dml_outcome_fare"), ("Y_pax", "dml_outcome_pax")]:
        pred = cross_val_predict(
            GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=config.SEED),
            w, cross[outcome].values, cv=folds)
        rows.append({"model": tag, "metric": "r2_oof", "value": round(r2_score(cross[outcome].values, pred), 4)})
    return rows


def run_diagnostics() -> pd.DataFrame:
    """Compute all fit diagnostics and write the summary table."""
    config.make_output_dirs()
    panel = pd.read_parquet(config.PROCESSED_DIR / "market_quarter_panel.parquet")
    matched = pd.read_parquet(config.PROCESSED_DIR / "matched_market_panel.parquet")

    rows = propensity_fit(panel) + event_study_fit(matched) + nuisance_fit(matched)
    table = pd.DataFrame(rows)
    table.to_csv(config.TABLES_DIR / "model_fit_diagnostics.csv", index=False)
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    run_diagnostics()
