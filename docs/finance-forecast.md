# The forecast model: expected price Δ minutes ahead

Status: design, 2026-10-24. Agreed in discussion; this is the written form to argue with before code.
Builds on [finance-ml.md](finance-ml.md) (Fit, FitRun, Model, InferRun, Evaluation) and the data in
[finance-app.md](finance-app.md). Supersedes [finance-modelling.md](finance-modelling.md) as the first
thing we fit.

## The problem

**Given a stock now, what is its expected price Δ minutes from now?**

The eventual consumer is a trader that acts once per decision point inside a date range, at a grain
that may be a day, 15 minutes, or anything else. The trader is not designed here; the only thing this
document takes from it is the contract: every feature must be computable at any decision point from
data strictly before it, and the horizon Δ must be a parameter, so the same model definition serves
any grain.

## The Fit

### Row

One per (symbol, decision minute t), for t every `:step_minutes` (default 5) inside the session, over
the Kite universe: `equity_1m`, 463 symbols including the indices, 756 sessions from 2023-10-03 to
2026-10-08 as of writing, 120 million minute bars. At 5-minute spacing that is about 20 million rows
before the liquidity filter.

**Liquidity filter.** A row is kept only if the stock's rupee turnover (volume × price, summed) over the
trailing `:liq_minutes` (default 60) up to t is at least `:min_turnover`. A thin window gives a stale
last price and a bar open nobody could trade at size, so its target is partly fiction, and a model
trained on it learns the stale print catching up rather than anything about the stock. The floor is a
Fit parameter so it is recorded and applied identically at fit and inference; a stock below it at some
minute gets no prediction there, which the consumer reads as "do nothing". The Kite universe already
needs 25 crore a session, about 7 lakh a minute on average, so 20 lakh over 60 minutes is a loose floor
and 1 crore a strict one; the IC screen reports rows and IC at both and the better one is kept. The
minute bars carry volume in shares, not trade counts, so turnover is the liquidity measure available.

### Target

    r = log(close of the bar ending at t + Δ) - log(open of the bar after t)
    y = r / σ_t

The return, not the price: prices differ by orders of magnitude across stocks and drift over years, so
a model on prices learns the level, not the move. The entry is the next bar's open because that is
what a decision at t gets; the minute's last trade can be stale for a thin minute.

Scaled by volatility: `σ_t` is the stock's realised volatility of minute returns over the trailing
`:vol_minutes` (default 60) up to t, scaled to the horizon. A 0.3 percent move in a stock that
normally moves 0.2 percent in 15 minutes is an event; in one that normally moves 0.8 it is noise. A
raw-return model spends its capacity learning which stocks are volatile, which is already known; the
scaled target makes rows comparable across stocks and leaves the model the direction. The expected
price is recovered at read time:

    expected price = price now × exp(prediction × σ_t)

`Δ` is the query parameter `:horizon_minutes`. One Fit per horizon: 15, 30, 60 minutes and one
session. Same features, separate models, so each horizon's errors are read on their own. Pooling
horizons into one model is for later, once the separate ones are understood. `:vol_minutes` is tried
at 60 and at a 20-session daily window scaled to the horizon in the IC screen; the better one is kept.

The consumer also needs σ. First version: `σ_t` itself, free and adequate at 15 minutes. A second Fit
on `|y|` with the same features, predicting whether the next Δ is calmer or wilder than the trailing
window, is a follow-up; it matters around the open and around news, not elsewhere.

### Features

All ratios or returns, never rupee levels. Each is z-scored or ranked within its decision point so
that no stock dominates and the model learns relative moves. Nothing from after t.

| layer | features |
|---|---|
| intraday, bars up to t | returns over the last 1, 5, 15, 30, 60 min and since open; position within today's range; deviation from today's VWAP; realised volatility over the last 30 min and since open; volume in the last 5 and 15 min relative to the same clock time over the past 20 sessions; rupee turnover over the trailing `:liq_minutes`, as a level across stocks (how much can be traded here, beyond passing the floor); NIFTY 50's returns over the same spans; the stock's return minus the index's over 15 and 60 min |
| buying pressure, bars up to t, over 5, 15 and 30 min | signed volume imbalance (volume on up minutes minus volume on down minutes, over total volume; an up minute closes above the previous close); close-location volume (volume × ((close − low) − (high − close)) / (high − low), summed, the accumulation-distribution line); volume-weighted return; share of up minutes; price impact (absolute return over rupee turnover); turnover acceleration (last 5 min over last 30) |
| daily, frozen at yesterday's close | yesterday's return; today's gap at open; 5- and 20-day return; 20-day volatility; distance from the trailing 52-week high and low; turnover relative to its 20-day median; deliverable percentage where NSE reports it; relative strength against NIFTY 50 over 20 days |
| clock | minutes since open; minutes to close; day of week; expiry day; day after a holiday |

About 45 features. The final list is decided by the IC screen below, not by this table.

**Buying pressure is inferred, not observed.** The bars have no trade direction, bid, ask or book, so
"many traders buying" is read from what buying does to a bar: price up on volume. The tick rule and the
close-location line are the standard bar-level proxies and they work tolerably at the minute scale.
The predictive question is not whether buying moves the price, which is simultaneous, but whether
pressure over the last window continues into the next one (a large order being worked) or reverses (a
crowd chasing a print); the proxies together with relative volume and the clock let the model learn
which. Deliverable percentage, at the daily layer, is the one genuine buyer-side number NSE gives.

Kite's live feed carries total buy and sell quantities and the top-five book per tick, which is real
order flow, but only live. A collector running during the session would build that history from now
on; in a year it would be a feature the bars cannot provide. A separate leg, noted here as the next
data source, not part of this Fit.

### Search

`sklearn.linear` as the baseline that must be beaten, then `xgboost` on the GPU over
max_depth {4, 6}, learning_rate {0.05, 0.1}, n_estimators {500, 1500}, min_child_weight {10, 100}:
sixteen models plus the baseline.

### Split

Fit on 2023-10-03 to 2025-12-31. `Infer` on 2026-01-01 to the end of the data with the label
`validation 2026`. One session's gap between fit and validation so no target crosses the cut.

### Evaluation

On the validation table, in this order of importance:

1. **Information coefficient.** Spearman correlation between prediction and realised scaled return within
   each decision point, averaged over decision points, with its standard error. The number that says
   the model orders stocks better than chance. New metric for the registry.
2. **Calibration.** Realised mean return per decile of prediction. Must be monotone and near the
   diagonal. A linear rescale is fitted and recorded if the slope is off; the consumer needs the
   magnitude, not just the order. New metric.
3. **Net return per trade after costs**, in raw return, at predicted-move thresholds of 0.1, 0.2 and
   0.5 percent (prediction × σ_t), with the
   trade count at each, using the cost model from the old repo's `docs/brokers.md` for an intraday
   round trip. Shows where, if anywhere, the forecast clears the cost. Exists; needs the threshold param.
4. All three **per month**, so one good quarter cannot carry the year.

### What we expect

Fifteen-minute returns on liquid stocks are mostly noise. A good model has an IC of 0.02 to 0.05 and
explains under one percent of variance. Intraday round-trip costs on NSE, STT on the sell side,
exchange charges and the spread, are around 0.05 to 0.1 percent, the size of a typical 15-minute move.
So a useful model is one whose top decile clears cost by a margin, not one that is right on average.
If the 15-minute IC is flat we read the 60-minute and one-session Fits before changing anything else;
signal usually grows with horizon.

## Before the Fit: the IC screen

Each feature on its own, against the realised return at each horizon, as the same within-decision-point
Spearman averaged over decision points, per year. One DuckDB query over the feature view. A feature
whose IC is within two standard errors of zero in every year is dropped. A feature whose IC flips sign
between years is flagged. This runs before any GPU time and decides the feature list.

## Changes to what exists

- **A minute feature view** in `ml/data.py` over `equity_1m` with NIFTY 50 joined by minute and the
  daily layer joined from `equity` and the reference tables by (symbol, previous session).
- **`LoadRows` streams.** It materialises the query's frame today. Twenty million rows × 30 features
  fit the GTX 1080's 8 GB as float32 but not as a Python-side copy alongside. The loader writes the
  frame to Parquet in the work dir and the fitter reads it in batches; xgboost takes an iterator.
- **Metrics** `ic` and `calibration`, and a `threshold` param on `net_return_per_trade`.
- **Data currency.** The Kite leg is behind its token; `make kite-login` brings `equity_1m` past
  2026-10-08. The Fit does not wait for it.

## Order of work

1. The feature view and the query with `:horizon_minutes`, `:step_minutes`, `:vol_minutes`,
   `:liq_minutes` and `:min_turnover`, checked for leakage by
   construction: every window ends at or before t, every daily join is on the previous session.
2. The IC screen at 15, 60 and 375 minutes. Its table goes in this document.
3. The streaming loader and the two metrics.
4. The 15-minute Fit registered and run on the GPU; `Infer` on 2026; `Evaluate`. Then 60 minutes and
   one session with the same query.
5. The results, per horizon, appended here: IC with standard error, calibration slope, net return at
   each threshold, and the monthly series. Decision after that: more features, another horizon, or
   on to the consumer.

## Settled and open

Settled: return not price, scaled by trailing volatility; Δ, the volatility window and the liquidity floor as parameters; features computable at any decision point from data
before it; within-decision-point normalisation; the IC screen before fitting; evaluation on 2026 by
IC, calibration and net return after costs, per month; baseline linear, then xgboost.

Open: the liquidity floor's value (decided by the screen); step spacing (5 minutes, or every minute with subsampling); whether to include the indices as
rows or only as features (default: features only, they are not tradable in cash); whether rows in the
first and last 15 minutes of the session are kept (default: kept, with the clock features carrying
the information).

## Later, not here

The consumer: a trader of signals, a decision rule and execution; the hindsight-optimal trades as
the ceiling it is scored against; the product `(collection, principal, start, end) -> expected
return`. Discussed and parked until the forecast has a measured IC.

## Results

### IC screen, 15 minutes (2026-10-24)

Rows: 22,112,730 at 5-minute spacing, 60-minute volatility window, 20 lakh over 60 minutes as the
liquidity floor, 2023-10-03 to 2026-10-08; built in 529 s, screened in 116 s, by
`python -m finance.ml.forecast --horizon 15`. IC is the Spearman correlation between the feature and
the scaled 15-minute return within each decision point, averaged over the points of the year; t is the
mean over its standard error. 2023 is October to December only.

| feature | 2023 | 2024 | 2025 | 2026 | verdict |
|---|---|---|---|---|---|
| r1 | -0.0528 (t=-55.1) | -0.0482 (t=-87.5) | -0.0456 (t=-87.2) | -0.0421 (t=-72.7) | keep |
| r5 | -0.0624 (t=-56.2) | -0.0558 (t=-88.5) | -0.0486 (t=-80.5) | -0.0415 (t=-62.3) | keep |
| r15 | -0.0534 (t=-45.5) | -0.0463 (t=-70.7) | -0.0373 (t=-59.5) | -0.0335 (t=-48.4) | keep |
| r30 | -0.0408 (t=-34.2) | -0.0375 (t=-56.0) | -0.0273 (t=-42.6) | -0.0249 (t=-35.3) | keep |
| r60 | -0.0290 (t=-23.0) | -0.0253 (t=-36.4) | -0.0154 (t=-23.1) | -0.0116 (t=-15.9) | keep |
| r_open | -0.0139 (t=-12.2) | -0.0085 (t=-13.3) | -0.0015 (t=-2.4) | -0.0019 (t=-2.7) | keep |
| range_pos | -0.0175 (t=-14.1) | -0.0134 (t=-20.7) | -0.0054 (t=-8.3) | -0.0058 (t=-7.9) | keep |
| vwap_dev | -0.0154 (t=-12.3) | -0.0130 (t=-19.7) | -0.0036 (t=-5.6) | -0.0013 (t=-1.8) | keep |
| rv30 | -0.0048 (t=-3.2) | +0.0006 (t=0.8) | -0.0049 (t=-5.7) | +0.0031 (t=3.3) | flips |
| rv_open | -0.0133 (t=-8.1) | -0.0055 (t=-6.1) | -0.0098 (t=-10.5) | -0.0036 (t=-3.5) | keep |
| relvol5 | +0.0150 (t=14.8) | +0.0112 (t=21.0) | +0.0083 (t=15.7) | +0.0095 (t=15.9) | keep |
| relvol15 | +0.0149 (t=14.2) | +0.0111 (t=19.9) | +0.0086 (t=15.3) | +0.0102 (t=15.7) | keep |
| turn_liq | +0.0117 (t=8.8) | +0.0047 (t=6.1) | +0.0068 (t=9.7) | +0.0076 (t=9.5) | keep |
| i_r5 |  |  |  |  | context |
| i_r15 |  |  |  |  | context |
| i_r30 |  |  |  |  | context |
| i_r60 |  |  |  |  | context |
| i_r_open |  |  |  |  | context |
| x_r15 | -0.0534 (t=-45.5) | -0.0463 (t=-70.7) | -0.0373 (t=-59.5) | -0.0335 (t=-48.4) | keep |
| x_r60 | -0.0290 (t=-23.0) | -0.0253 (t=-36.4) | -0.0154 (t=-23.1) | -0.0116 (t=-15.9) | keep |
| imb5 | -0.0433 (t=-44.2) | -0.0359 (t=-67.9) | -0.0302 (t=-57.8) | -0.0275 (t=-48.1) | keep |
| imb15 | -0.0306 (t=-31.0) | -0.0238 (t=-45.1) | -0.0176 (t=-33.6) | -0.0174 (t=-30.6) | keep |
| imb30 | -0.0212 (t=-21.1) | -0.0170 (t=-31.8) | -0.0103 (t=-19.6) | -0.0106 (t=-18.2) | keep |
| clv5 | -0.0379 (t=-42.4) | -0.0308 (t=-65.0) | -0.0247 (t=-53.1) | -0.0242 (t=-46.4) | keep |
| clv15 | -0.0250 (t=-27.4) | -0.0183 (t=-39.2) | -0.0123 (t=-26.6) | -0.0152 (t=-29.2) | keep |
| clv30 | -0.0182 (t=-19.9) | -0.0131 (t=-27.9) | -0.0064 (t=-13.7) | -0.0097 (t=-18.7) | keep |
| vwret5 | -0.0491 (t=-48.2) | -0.0424 (t=-76.1) | -0.0352 (t=-65.0) | -0.0304 (t=-51.3) | keep |
| vwret15 | -0.0342 (t=-34.2) | -0.0268 (t=-50.3) | -0.0196 (t=-37.4) | -0.0185 (t=-32.6) | keep |
| vwret30 | -0.0234 (t=-23.7) | -0.0190 (t=-36.6) | -0.0114 (t=-22.2) | -0.0115 (t=-20.5) | keep |
| up5 | -0.0477 (t=-47.1) | -0.0411 (t=-74.1) | -0.0359 (t=-66.4) | -0.0327 (t=-54.6) | keep |
| up15 | -0.0362 (t=-34.9) | -0.0320 (t=-56.6) | -0.0265 (t=-47.4) | -0.0259 (t=-41.4) | keep |
| up30 | -0.0264 (t=-25.5) | -0.0243 (t=-42.3) | -0.0191 (t=-34.0) | -0.0187 (t=-29.6) | keep |
| impact15 | -0.0051 (t=-4.4) | +0.0002 (t=0.2) | -0.0047 (t=-7.0) | -0.0047 (t=-6.3) | keep |
| impact30 | -0.0052 (t=-4.3) | +0.0009 (t=1.3) | -0.0052 (t=-7.5) | -0.0037 (t=-4.8) | keep |
| turn_accel | +0.0062 (t=6.6) | +0.0037 (t=7.4) | +0.0012 (t=2.6) | +0.0017 (t=3.3) | keep |
| ret_1d | -0.0021 (t=-1.8) | +0.0024 (t=3.9) | +0.0045 (t=7.3) | -0.0010 (t=-1.5) | keep |
| gap | -0.0003 (t=-0.4) | +0.0005 (t=1.1) | +0.0003 (t=0.7) | +0.0003 (t=0.5) | drop |
| ret_5d | -0.0050 (t=-4.2) | +0.0035 (t=5.5) | +0.0044 (t=6.6) | +0.0010 (t=1.4) | flips |
| ret_20d | -0.0051 (t=-4.5) | +0.0033 (t=5.4) | +0.0055 (t=7.9) | +0.0010 (t=1.2) | flips |
| vol_20d | -0.0209 (t=-14.0) | -0.0136 (t=-17.2) | -0.0139 (t=-17.2) | -0.0068 (t=-8.6) | keep |
| dist_hi | +0.0069 (t=6.4) | +0.0131 (t=19.6) | +0.0135 (t=17.5) | +0.0097 (t=11.5) | keep |
| dist_lo | -0.0150 (t=-11.4) | -0.0056 (t=-8.0) | -0.0024 (t=-3.4) | +0.0023 (t=2.6) | flips |
| turn_rel | -0.0008 (t=-0.8) | +0.0005 (t=0.9) | +0.0012 (t=2.3) | -0.0004 (t=-0.7) | keep |
| deliv_pct | +0.0184 (t=13.8) | +0.0133 (t=17.1) | +0.0175 (t=22.9) | +0.0134 (t=18.6) | keep |
| rs_20 | -0.0051 (t=-4.5) | +0.0033 (t=5.4) | +0.0055 (t=7.9) | +0.0010 (t=1.2) | flips |
| m |  |  |  |  | context |
| m_left |  |  |  |  | context |
| dow |  |  |  |  | context |
| expiry |  |  |  |  | context |
| after_holiday |  |  |  |  | context |

What it says:

- **Fifteen minutes ahead is reversal, everywhere.** Every recent-return and buying-pressure feature
  is negative, large by the standards of this problem (r5 at −0.06 to −0.04, the 5-minute pressure
  proxies at −0.03 to −0.05) and stable in sign across all four years. A stock that just rose, on
  volume, with its minutes closing near their highs, is the one that gives back over the next quarter
  hour. "Many traders buying" predicts the next 15 minutes *down*: at this horizon the crowd chasing a
  print dominates the large order being worked.
- **Part of that is microstructure, and the model will not get all of it.** A minute close is at the
  bid or the ask; the next one tends to bounce back. The entry at the next bar's open removes some of
  this, the exit at a later close does not. The IC is a true statistic of the data; the fraction that
  survives the spread and costs is what the net-return evaluation is for, and it will be smaller.
- **The edge decays by year**, from −0.062 (2023) to −0.042 (2026) on r5, the same shape on every
  reversal feature. Either the market has become more efficient at this horizon or the three months of
  2023 were unusual. Validation on 2026 is the right test either way.
- **Volume is the one positive.** Relative volume at +0.01 and turnover at +0.005 to +0.01, both
  stable: given the reversal, more participation now means a slightly better next 15 minutes. Price
  impact is small and inconsistent; turnover acceleration small and positive.
- **The daily layer is weak but two features are stable**: deliverable percentage at +0.013 to +0.018
  every year, the one genuine buyer-side number, and distance from the 52-week high at +0.007 to
  +0.014. Twenty-day volatility is stably negative. Daily momentum (5- and 20-day return, relative
  strength) flips sign between 2023 and the rest; the gap is zero. These earn their place at longer
  horizons if anywhere.
- **Dropped for the 15-minute Fit:** gap. **Held out as flips:** rv30, ret_5d, ret_20d, rs_20,
  dist_lo. `x_r15`/`x_r60` equal `r15`/`r60` within a decision point (the index is the same for every
  stock), so their IC is identical; they stay because the model sees them together with the index's
  own return as context, which the cross-sectional IC cannot measure.
- The t-statistics are large because there are about 150,000 decision points; the magnitude is what to
  read, not the t.
