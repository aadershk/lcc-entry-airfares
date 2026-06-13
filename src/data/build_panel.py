"""Build the market-quarter panel from the parsed DB1B and T-100 tables.

For every directed airport pair and quarter the panel records total passengers,
the passenger-weighted average fare, the carrier count, the HHI, the route
distance, and the LCC entry timing used to define treatment.

Input:  data/interim/db1b_collapsed.parquet, data/interim/t100_collapsed.parquet
Output: data/processed/market_quarter_panel.parquet
"""
import numpy as np
import pandas as pd

from src import config

MARKET_KEYS = ["Year", "Quarter", "Origin", "Dest"]


def build_panel() -> pd.DataFrame:
    """Merge the two sources and construct the market-quarter panel."""
    config.make_output_dirs()
    db1b = pd.read_parquet(config.INTERIM_DIR / "db1b_collapsed.parquet")
    t100 = pd.read_parquet(config.INTERIM_DIR / "t100_collapsed.parquet")

    # Match DB1B operating carrier to the T-100 carrier field before merging.
    db1b = db1b.rename(columns={"OpCarrier": "Carrier"})
    merged = pd.merge(
        db1b, t100, on=["Year", "Quarter", "Origin", "Dest", "Carrier"],
        how="left", suffixes=("_mkt", "_cap"),
    )
    merged["Seats"] = merged["Seats"].fillna(0)
    merged["Departures"] = merged["Departures"].fillna(0)

    # Market totals: DB1B passengers and the number of active carriers.
    totals = merged.groupby(MARKET_KEYS, as_index=False).agg(
        Mkt_Total_Pax=("Passengers_mkt", "sum"),
        N_Carriers=("Carrier", "nunique"),
    )

    # Passenger-weighted average fare per market.
    merged["Revenue"] = merged["AvgFare"] * merged["Passengers_mkt"]
    revenue = merged.groupby(MARKET_KEYS, as_index=False).agg(
        Mkt_Total_Rev=("Revenue", "sum"),
    )

    # HHI from squared passenger shares (0-10,000 scale).
    merged = merged.merge(totals[MARKET_KEYS + ["Mkt_Total_Pax"]], on=MARKET_KEYS, how="left")
    merged["Share_Sq"] = (merged["Passengers_mkt"] / merged["Mkt_Total_Pax"] * 100) ** 2
    hhi = merged.groupby(MARKET_KEYS, as_index=False).agg(HHI=("Share_Sq", "sum"))

    distance = merged.groupby(MARKET_KEYS, as_index=False).agg(MktDistance=("MktDistance", "mean"))

    merged["Is_LCC"] = merged["Carrier"].isin(config.LCC_CARRIERS).astype(int)
    lcc_present = merged.groupby(MARKET_KEYS, as_index=False).agg(LCC_Present=("Is_LCC", "max"))

    panel = (
        totals
        .merge(revenue, on=MARKET_KEYS)
        .merge(hhi, on=MARKET_KEYS)
        .merge(distance, on=MARKET_KEYS)
        .merge(lcc_present, on=MARKET_KEYS)
    )
    panel["Mkt_AvgFare"] = panel["Mkt_Total_Rev"] / panel["Mkt_Total_Pax"]
    panel = panel.drop(columns="Mkt_Total_Rev")

    panel = _add_treatment_timing(panel)

    out_path = config.PROCESSED_DIR / "market_quarter_panel.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"Saved panel: {panel.shape[0]:,} market-quarters, {out_path}")
    return panel


def _add_treatment_timing(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the time index and the per-route LCC entry / control flags."""
    # Continuous quarter index: 2015 Q1 = 0, 2015 Q2 = 1, ...
    panel["Time_Idx"] = (panel["Year"] - config.START_YEAR) * 4 + (panel["Quarter"] - 1)

    entry = (
        panel[panel["LCC_Present"] == 1]
        .groupby(["Origin", "Dest"])["Time_Idx"].min()
        .reset_index()
        .rename(columns={"Time_Idx": "Treatment_Time"})
    )
    panel = panel.merge(entry, on=["Origin", "Dest"], how="left")

    panel["Never_Treated"] = panel["Treatment_Time"].isna().astype(int)
    panel["Post_Treatment"] = np.where(
        panel["Treatment_Time"].notna(),
        panel["Time_Idx"] >= panel["Treatment_Time"],
        0,
    ).astype(int)
    return panel


if __name__ == "__main__":
    build_panel()
