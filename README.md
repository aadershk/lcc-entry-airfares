# Heterogeneous effects of low-cost carrier entry on US airfares

Code for my MSc thesis (Data Science and Business Analytics, University of
Amsterdam). The analysis estimates how much low-cost carrier (LCC) entry lowers
fares on US domestic routes, and how that effect varies across markets.

The design has two stages:

1. **PSM-DiD.** Treated routes (markets that first see an LCC in 2016 or later)
   are matched 1:1 to never-treated controls on six pre-entry covariates, and a
   two-way fixed-effects event study estimates the average treatment effect on
   the treated.
2. **Causal forest.** A `CausalForestDML` is fit on the matched cross-section to
   recover a conditional average treatment effect (CATE) per route, which is then
   summarised by market characteristics (distance, concentration, density).

Treatment group: Southwest (WN), JetBlue (B6), Spirit (NK), Frontier (F9),
Allegiant (G4). Panel: 2015 Q1 to 2019 Q4.

## Repository layout

```
src/
  config.py          paths and fixed parameters
  data/
    parse_db1b.py    DB1B Market zips -> carrier-quarter table
    parse_t100.py    T-100 Segment zips -> carrier-quarter capacity
    build_panel.py   merge into the market-quarter panel
  matching.py        Stage 1a: propensity score matching
  event_study.py     Stage 1b: TWFE event study (fare, passengers)
  causal_forest.py   Stage 2: CATE, group ATEs, variable importance, BLP test
  robustness.py      placebo, JetBlue exclusion, distance subsample
  diagnostics.py     model-fit metrics (AUC, within-R2, nuisance fit)
  descriptives.py    descriptive and raw-trajectory figures
run_pipeline.py      runs every stage in order
```

## Setup

Python 3.11. Create an environment and install the pinned dependencies:

```
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

## Getting the data

Both datasets are free from the Bureau of Transportation Statistics (BTS). They
are not included in this repository. Download them and drop the zip files into
`data/raw/db1b/` and `data/raw/t100/` (the parsers read every `.zip` in those
folders, so the original BTS filenames are fine).

**DB1B Market (20 quarterly files, 2015 Q1 - 2019 Q4).** The prezipped Market
tables can be downloaded directly:

```
https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_<YEAR>_<QUARTER>.zip
```

with `<YEAR>` from 2015 to 2019 and `<QUARTER>` from 1 to 4, for example:

```
# bash
for y in 2015 2016 2017 2018 2019; do for q in 1 2 3 4; do
  curl -o "data/raw/db1b/DB1BMarket_${y}_${q}.zip" \
    "https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_${y}_${q}.zip"
done; done
```

Alternatively, on the TranStats site go to Aviation → *Airline Origin and
Destination Survey (DB1B)* → *DB1BMarket*, and download each quarter.

**T-100 Domestic Segment (US carriers), 2015-2019.** On the TranStats site go to
Aviation → *Air Carrier Statistics (Form 41 Traffic)* → *T-100 Domestic Segment
(U.S. Carriers)*. Select the period (one year at a time returns all twelve
months), keep the geography as all US, and include the fields:

```
YEAR, MONTH, UNIQUE_CARRIER, ORIGIN, DEST, PASSENGERS, SEATS,
DEPARTURES_PERFORMED, CLASS, ORIGIN_COUNTRY, DEST_COUNTRY
```

Download each year 2015-2019 and save the zip files into `data/raw/t100/`.

## Running the analysis

With the raw data in place:

```
python run_pipeline.py
```

This runs the stages in order: parse DB1B and T-100, build the panel, match, run
the event studies, fit the causal forest, run the robustness checks, and produce
the diagnostics and descriptive figures. The causal forest is the slow stage
(roughly 30 minutes on a laptop); everything else takes a few minutes.

To resume from a later stage (for example, after the panel is already built):

```
python run_pipeline.py --from matching
```

Individual stages can also be run on their own, e.g. `python -m src.matching`.

## Outputs

Tables are written to `outputs/tables/` and figures to `outputs/figures/`.
Intermediate and processed data go to `data/interim/` and `data/processed/`.
All stochastic steps use seed 42, so a clean run reproduces the reported numbers.
