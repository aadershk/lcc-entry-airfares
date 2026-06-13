"""Parse the raw DB1B Market zip files into a carrier-level quarterly table.

Each quarterly DB1B Market file is a 10% sample of US itinerary tickets. We keep
the columns needed for the analysis, apply the fare screens, and collapse to one
row per market-quarter-carrier with total passengers and a passenger-weighted
average fare.

Input:  data/raw/db1b/*.zip   (BTS DB1B Market, one file per quarter)
Output: data/interim/db1b_collapsed.parquet
"""
import zipfile
from pathlib import Path

import pandas as pd

from src import config

# BTS column names vary slightly across vintages, so map a lower-cased version
# of whatever is in the file onto the names used downstream.
COLUMN_ALIASES = {
    "opcarrier": "OpCarrier",
    "operating_carrier": "OpCarrier",
    "mktfare": "MktFare",
    "market_fare": "MktFare",
    "mktdistance": "MktDistance",
    "market_distance": "MktDistance",
    "bulkfare": "BulkFare",
    "bulk_fare": "BulkFare",
    "passengers": "Passengers",
    "year": "Year",
    "quarter": "Quarter",
    "origin": "Origin",
    "dest": "Dest",
}


def _read_one_zip(path: Path) -> pd.DataFrame:
    """Read and collapse a single DB1B quarterly zip to market-carrier rows."""
    with zipfile.ZipFile(path) as archive:
        csv_name = next(n for n in archive.namelist() if n.endswith(".csv"))
        with archive.open(csv_name) as handle:
            header = pd.read_csv(handle, nrows=0)
            present = {c.lower(): c for c in header.columns}
            usecols = [present[k] for k in present if k in COLUMN_ALIASES]
            rename = {present[k]: COLUMN_ALIASES[k] for k in present if k in COLUMN_ALIASES}

            handle.seek(0)
            df = pd.read_csv(handle, usecols=usecols).rename(columns=rename)

    if "BulkFare" in df.columns:
        df = df[df["BulkFare"] != 1]
    df = df[(df["MktFare"] >= config.FARE_MIN) & (df["MktFare"] <= config.FARE_MAX)]
    df = df.dropna(subset=["Origin", "Dest", "OpCarrier"])
    df = df[df["Origin"] != df["Dest"]]

    df["TotalRevenue"] = df["MktFare"] * df["Passengers"]
    group_cols = ["Year", "Quarter", "Origin", "Dest", "OpCarrier"]
    if "MktDistance" in df.columns:
        group_cols.append("MktDistance")

    collapsed = df.groupby(group_cols, as_index=False).agg(
        Passengers=("Passengers", "sum"),
        TotalRevenue=("TotalRevenue", "sum"),
    )
    collapsed["AvgFare"] = collapsed["TotalRevenue"] / collapsed["Passengers"]
    return collapsed.drop(columns="TotalRevenue")


def parse_db1b() -> pd.DataFrame:
    """Parse every DB1B zip in the raw directory and write the interim table."""
    config.make_output_dirs()
    files = sorted(config.RAW_DB1B_DIR.glob("*.zip"))
    if not files:
        raise FileNotFoundError(
            f"No DB1B zip files found in {config.RAW_DB1B_DIR}. "
            "See the README for download instructions."
        )

    print(f"Parsing {len(files)} DB1B files...")
    chunks = []
    for path in files:
        collapsed = _read_one_zip(path)
        chunks.append(collapsed)
        print(f"  {path.name}: {len(collapsed):,} market-carrier rows")

    panel = pd.concat(chunks, ignore_index=True)
    out_path = config.INTERIM_DIR / "db1b_collapsed.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"Saved {len(panel):,} rows to {out_path}")
    return panel


if __name__ == "__main__":
    parse_db1b()
