# Heterogeneous Treatment Effects of Low-Cost Carrier Entry in US Airline Markets

Code and reproduction steps for my MSc thesis (Data Science and Business
Analytics, University of Amsterdam). The study measures how much low-cost carrier
(LCC) entry lowers fares on US domestic routes, and shows that the effect is far
from uniform. The entry-quarter effect averages a 10 percent fare reduction, but
the route-level estimates range from about 16 percent on the shortest routes to
3 percent on the longest. A single average hides most of what is going on, which
is the point the thesis makes.

## What this is about

LCCs do not enter routes at random. They target dense, high-fare, concentrated
corridors, so a plain before-and-after comparison confounds where carriers choose
to enter with what entry does to price. Entry is also staggered across the panel,
which biases the usual two-way fixed-effects estimator once effects change with
time since entry. The design here corrects for both and then asks where the effect
is largest.

Two datasets from the Bureau of Transportation Statistics cover 2015 Q1 to
2019 Q4: the DB1B Market fare survey (a 10 percent sample of itineraries) and the
T-100 Domestic Segment traffic file. After the fare and scheduled-service screens,
the panel holds 1,327,550 directed market-quarter observations. A market is treated
the first quarter any of five carriers enters a previously LCC-free route:
Southwest (WN), JetBlue (B6), Spirit (NK), Frontier (F9), Allegiant (G4).

## Method

The analysis runs in two stages.

**Stage 1, PSM-DiD.** Each treated route (first entry in 2016 or later, so a full
2015 baseline is observed) is matched 1:1 to a never-treated control on six
pre-entry covariates: log fare, log passengers, log distance, HHI, carrier count,
and the 2015 fare trajectory. Matching is nearest-neighbour on the propensity
score and the covariates jointly, under a 0.2 standard-deviation caliper. A
two-way fixed-effects event study on the matched panel, with never-treated routes
as the only controls, gives the average treatment effect on the treated. Errors
are clustered by route.

**Stage 2, causal forest.** A `CausalForestDML` is fit on the matched
cross-section with event-time-aligned pre/post windows, returning a conditional
average treatment effect (CATE) per route. The systematic part of that variation
is read off through group ATEs across quartiles, the forest's variable importance,
and a best-linear-projection test.

The pipeline follows the same order, stage by stage: parse DB1B and T-100, build
the panel, match, run the event studies, fit the forest, run the robustness
checks, and produce the fit diagnostics and descriptive figures.

## Main findings

**Stage 1.** After matching, all six covariates balance to below 0.02 standardised
mean difference, down from values above 1 before matching, and the propensity
model separates entered from non-entered routes well (AUC 0.87). The entry-quarter
ATT is a 10.0 percent fare reduction (-0.105 log points, clustered SE 0.008),
averaging 8.1 percent across the nine post-entry quarters. Passenger traffic rises
33.7 percent at entry. The pre-entry coefficients are small and point the opposite
way to the effect, so the estimate reads as a conservative bound rather than an
overstatement.

**Stage 2.** The route-level CATE distribution has a mean of -8.0 percent, with
94.9 percent of routes negative, and the best-linear-projection test rejects the
no-heterogeneity null for both outcomes (theta_1 = 1.62, t = 12.3, p < 0.001).
Route distance is the dominant moderator, taking 74 percent of the forest's
importance: mean fare reductions fall from 16.3 percent on the shortest-quartile
routes to 3.1 percent on the longest, a 5.3-times gap. Market concentration adds a
clean monotone gradient, from 3.7 percent in the least concentrated quartile to
12.8 percent in the most. Pre-entry passenger density predicts where LCCs enter
but barely moves the size of the fare effect once distance and concentration are
held fixed.

**Robustness.** A four-quarter placebo shift produces an entry-quarter coefficient
an order of magnitude smaller than the real one. Rebuilding the treatment without
JetBlue leaves the ATT at 9.4 percent. Splitting the matched panel at the median
distance gives short- and long-haul ATTs of 14.7 and 4.8 percent, a 3.2-times gap
that reproduces the forest's distance gradient through a separate estimator.

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

## Reproducing the analysis

**Setup.** Python 3.11. Create an environment and install the pinned versions:

```
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

**Data.** Both sources are free from BTS and are not stored in this repository.
Download them and drop the zip files into `data/raw/db1b/` and `data/raw/t100/`.
The parsers read every `.zip` in those folders, so the original BTS filenames are
fine.

DB1B Market (20 quarterly files, 2015 Q1 - 2019 Q4) can be pulled directly from
the prezipped tables:

```
https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_<YEAR>_<QUARTER>.zip
```

with `<YEAR>` from 2015 to 2019 and `<QUARTER>` from 1 to 4, for example:

```
for y in 2015 2016 2017 2018 2019; do for q in 1 2 3 4; do
  curl -o "data/raw/db1b/DB1BMarket_${y}_${q}.zip" \
    "https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_${y}_${q}.zip"
done; done
```

Alternatively, on the TranStats site go to Aviation -> *Airline Origin and
Destination Survey (DB1B)* -> *DB1BMarket* and download each quarter.

T-100 Domestic Segment (US carriers) comes from the same site, under Aviation ->
*Air Carrier Statistics (Form 41 Traffic)* -> *T-100 Domestic Segment
(U.S. Carriers)*. Select one year at a time (which returns all twelve months),
keep the geography as all US, and include the fields:

```
YEAR, MONTH, UNIQUE_CARRIER, ORIGIN, DEST, PASSENGERS, SEATS,
DEPARTURES_PERFORMED, CLASS, ORIGIN_COUNTRY, DEST_COUNTRY
```

Save each year 2015-2019 into `data/raw/t100/`.

**Run.** With the raw data in place:

```
python run_pipeline.py
```

The stages run in order. The causal forest is the slow one, roughly 30 minutes on
a laptop, and the rest take a few minutes. To resume from a later stage, for
example once the panel is already built:

```
python run_pipeline.py --from matching
```

Individual stages also run on their own, for example `python -m src.matching`.

## Outputs

Coefficient tables and summaries are written to `outputs/tables/` (event-study
estimates, CATE quartile summary, variable importance, BLP test, model-fit
diagnostics, and the robustness results). Figures go to `outputs/figures/`, and
the cleaned panel and route-level CATE estimates to `data/processed/`. Every
stochastic step uses seed 42, so a clean run reproduces the numbers reported above
and in the thesis.
