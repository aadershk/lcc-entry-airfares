"""Parse the raw T-100 Domestic Segment zip files into carrier-quarter capacity.

T-100 reports monthly non-stop operating statistics by carrier and segment. We
keep US-to-US scheduled service, aggregate the monthly files to quarters, and
sum passengers, seats and departures per market-quarter-carrier. These counts
feed the market shares and the Herfindahl-Hirschman Index in build_panel.

Input:  data/raw/t100/*.zip   (BTS T-100 Domestic Segment, monthly files)
Output: data/interim/t100_collapsed.parquet
"""
import zipfile
from pathlib import Path

import pandas as pd

from src import config

COLUMN_ALIASES = {
    "year": "Year",
    "month": "Month",
    "origin": "Origin",
    "dest": "Dest",
    "carrier": "Carrier",
    "unique_carrier": "Carrier",
    "passengers": "Passengers",
    "seats": "Seats",
    "departures_performed": "Departures",
    "class": "Class",
    "origin_country": "OriginCountry",
    "dest_country": "DestCountry",
}


def _read_one_zip(path: Path) -> pd.DataFrame:
    """Read and collapse a single T-100 monthly zip to market-carrier rows."""
    with zipfile.ZipFile(path) as archive:
        csv_name = next(n for n in archive.namelist() if n.endswith(".csv"))
        with archive.open(csv_name) as handle:
            header = pd.read_csv(handle, nrows=0)
            present = {c.lower(): c for c in header.columns}

            usecols, rename, carrier_taken = [], {}, False
            for lower, exact in present.items():
                if lower not in COLUMN_ALIASES:
                    continue
                # Some vintages carry several carrier columns; keep the first.
                if COLUMN_ALIASES[lower] == "Carrier":
                    if carrier_taken:
                        continue
                    carrier_taken = True
                usecols.append(exact)
                rename[exact] = COLUMN_ALIASES[lower]

            handle.seek(0)
            df = pd.read_csv(handle, usecols=usecols).rename(columns=rename)

    if "OriginCountry" in df.columns and "DestCountry" in df.columns:
        df = df[(df["OriginCountry"] == "US") & (df["DestCountry"] == "US")]
    df = df.dropna(subset=["Origin", "Dest", "Carrier"])
    df = df[df["Origin"] != df["Dest"]]

    df["Quarter"] = ((df["Month"] - 1) // 3) + 1
    group_cols = ["Year", "Quarter", "Origin", "Dest", "Carrier"]
    agg = {col: "sum" for col in ("Passengers", "Seats", "Departures") if col in df.columns}
    return df.groupby(group_cols, as_index=False).agg(agg)


def parse_t100() -> pd.DataFrame:
    """Parse every T-100 zip in the raw directory and write the interim table."""
    config.make_output_dirs()
    files = sorted(config.RAW_T100_DIR.glob("*.zip"))
    if not files:
        raise FileNotFoundError(
            f"No T-100 zip files found in {config.RAW_T100_DIR}. "
            "See the README for download instructions."
        )

    print(f"Parsing {len(files)} T-100 files...")
    chunks = []
    for path in files:
        collapsed = _read_one_zip(path)
        chunks.append(collapsed)
        print(f"  {path.name}: {len(collapsed):,} market-carrier rows")

    panel = pd.concat(chunks, ignore_index=True)
    # A market-quarter can appear in several monthly files, so sum once more.
    group_cols = ["Year", "Quarter", "Origin", "Dest", "Carrier"]
    panel = panel.groupby(group_cols, as_index=False).sum()

    out_path = config.INTERIM_DIR / "t100_collapsed.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"Saved {len(panel):,} rows to {out_path}")
    return panel


if __name__ == "__main__":
    parse_t100()
