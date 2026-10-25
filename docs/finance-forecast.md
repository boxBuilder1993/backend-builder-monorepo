# The forecast model: expected price Δ minutes ahead

Status: design, 2026-10-24. Agreed in discussion; this is the written form to argue with before code.
Consolidated results of the October 2026 work: [finance-research-2026-10.md](finance-research-2026-10.md).
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

### IC screen, 60 minutes

Rows: 19,282,872, same parameters; built in 489 s, screened in 100 s.

| feature | 2023 | 2024 | 2025 | 2026 | verdict |
|---|---|---|---|---|---|
| r1 | -0.0259 (t=-27.0) | -0.0268 (t=-48.6) | -0.0223 (t=-42.3) | -0.0199 (t=-34.5) | keep |
| r5 | -0.0320 (t=-28.8) | -0.0303 (t=-48.3) | -0.0229 (t=-37.8) | -0.0190 (t=-28.8) | keep |
| r15 | -0.0299 (t=-25.3) | -0.0275 (t=-40.4) | -0.0170 (t=-26.5) | -0.0141 (t=-20.1) | keep |
| r30 | -0.0234 (t=-18.6) | -0.0228 (t=-32.1) | -0.0114 (t=-17.2) | -0.0084 (t=-11.6) | keep |
| r60 | -0.0140 (t=-10.4) | -0.0147 (t=-19.3) | -0.0029 (t=-4.0) | +0.0000 (t=0.1) | keep |
| r_open | -0.0129 (t=-10.1) | -0.0054 (t=-7.5) | +0.0001 (t=0.2) | -0.0038 (t=-5.0) | keep |
| range_pos | -0.0038 (t=-2.9) | -0.0019 (t=-2.6) | +0.0028 (t=3.9) | +0.0008 (t=1.1) | flips |
| vwap_dev | -0.0006 (t=-0.5) | -0.0010 (t=-1.4) | +0.0046 (t=6.5) | +0.0059 (t=8.0) | keep |
| rv30 | -0.0076 (t=-4.5) | +0.0049 (t=5.1) | -0.0035 (t=-3.7) | +0.0060 (t=5.8) | flips |
| rv_open | -0.0213 (t=-11.5) | -0.0056 (t=-5.4) | -0.0138 (t=-13.6) | -0.0063 (t=-5.6) | keep |
| relvol5 | +0.0227 (t=21.1) | +0.0144 (t=25.4) | +0.0102 (t=17.8) | +0.0119 (t=18.9) | keep |
| relvol15 | +0.0238 (t=21.0) | +0.0156 (t=25.7) | +0.0115 (t=18.6) | +0.0136 (t=20.0) | keep |
| turn_liq | +0.0237 (t=17.1) | +0.0058 (t=7.3) | +0.0070 (t=9.5) | +0.0108 (t=13.6) | keep |
| i_r5 |  |  |  |  | context |
| i_r15 |  |  |  |  | context |
| i_r30 |  |  |  |  | context |
| i_r60 |  |  |  |  | context |
| i_r_open |  |  |  |  | context |
| x_r15 | -0.0299 (t=-25.3) | -0.0275 (t=-40.4) | -0.0170 (t=-26.5) | -0.0141 (t=-20.1) | keep |
| x_r60 | -0.0140 (t=-10.4) | -0.0147 (t=-19.3) | -0.0029 (t=-4.0) | +0.0000 (t=0.1) | keep |
| imb5 | -0.0211 (t=-21.2) | -0.0189 (t=-34.8) | -0.0120 (t=-22.4) | -0.0111 (t=-19.0) | keep |
| imb15 | -0.0157 (t=-15.4) | -0.0136 (t=-24.2) | -0.0049 (t=-8.9) | -0.0047 (t=-7.8) | keep |
| imb30 | -0.0109 (t=-10.3) | -0.0096 (t=-16.8) | -0.0012 (t=-2.1) | -0.0006 (t=-1.0) | keep |
| clv5 | -0.0198 (t=-21.0) | -0.0173 (t=-34.9) | -0.0111 (t=-22.9) | -0.0140 (t=-25.7) | keep |
| clv15 | -0.0142 (t=-14.6) | -0.0118 (t=-23.3) | -0.0047 (t=-9.6) | -0.0101 (t=-18.2) | keep |
| clv30 | -0.0111 (t=-11.2) | -0.0093 (t=-18.1) | -0.0020 (t=-4.0) | -0.0076 (t=-13.9) | keep |
| vwret5 | -0.0246 (t=-24.0) | -0.0223 (t=-39.5) | -0.0142 (t=-25.9) | -0.0124 (t=-20.9) | keep |
| vwret15 | -0.0186 (t=-18.1) | -0.0159 (t=-28.3) | -0.0057 (t=-10.7) | -0.0057 (t=-9.7) | keep |
| vwret30 | -0.0128 (t=-12.2) | -0.0118 (t=-21.3) | -0.0019 (t=-3.6) | -0.0021 (t=-3.6) | keep |
| up5 | -0.0244 (t=-23.3) | -0.0229 (t=-40.1) | -0.0177 (t=-31.9) | -0.0159 (t=-25.9) | keep |
| up15 | -0.0194 (t=-18.2) | -0.0193 (t=-32.1) | -0.0141 (t=-24.0) | -0.0124 (t=-19.2) | keep |
| up30 | -0.0142 (t=-13.0) | -0.0151 (t=-24.3) | -0.0115 (t=-19.2) | -0.0091 (t=-13.9) | keep |
| impact15 | -0.0171 (t=-13.2) | -0.0022 (t=-3.0) | -0.0059 (t=-8.4) | -0.0081 (t=-10.7) | keep |
| impact30 | -0.0168 (t=-12.4) | -0.0016 (t=-2.1) | -0.0057 (t=-7.8) | -0.0069 (t=-8.9) | keep |
| turn_accel | +0.0063 (t=6.5) | +0.0024 (t=4.6) | +0.0006 (t=1.2) | +0.0009 (t=1.8) | keep |
| ret_1d | -0.0026 (t=-2.1) | +0.0036 (t=5.1) | +0.0072 (t=10.3) | -0.0015 (t=-2.0) | flips |
| gap | -0.0041 (t=-4.4) | -0.0002 (t=-0.3) | -0.0014 (t=-2.7) | +0.0006 (t=1.0) | keep |
| ret_5d | -0.0057 (t=-4.3) | +0.0077 (t=10.2) | +0.0080 (t=10.3) | +0.0028 (t=3.5) | flips |
| ret_20d | -0.0067 (t=-5.3) | +0.0066 (t=9.2) | +0.0093 (t=11.4) | -0.0000 (t=-0.0) | flips |
| vol_20d | -0.0380 (t=-22.7) | -0.0188 (t=-20.7) | -0.0203 (t=-23.3) | -0.0103 (t=-12.0) | keep |
| dist_hi | +0.0118 (t=9.7) | +0.0197 (t=25.9) | +0.0209 (t=23.5) | +0.0140 (t=14.9) | keep |
| dist_lo | -0.0303 (t=-20.4) | -0.0087 (t=-10.7) | -0.0044 (t=-5.5) | +0.0012 (t=1.2) | keep |
| turn_rel | +0.0000 (t=0.0) | +0.0015 (t=2.6) | +0.0010 (t=1.7) | +0.0000 (t=0.1) | keep |
| deliv_pct | +0.0299 (t=20.5) | +0.0192 (t=23.0) | +0.0277 (t=33.7) | +0.0229 (t=29.9) | keep |
| rs_20 | -0.0067 (t=-5.3) | +0.0066 (t=9.2) | +0.0093 (t=11.4) | -0.0000 (t=-0.0) | flips |
| m |  |  |  |  | context |
| m_left |  |  |  |  | context |
| dow |  |  |  |  | context |
| expiry |  |  |  |  | context |
| after_holiday |  |  |  |  | context |

### IC screen, one session (to the close)

Rows: 23,060,452, same parameters; built in 554 s, screened in 121 s. The horizon of a row is the
minutes to the close, so the scaled return uses that, not 375; within a decision point every stock
has the same horizon, so the ranking and the IC are unaffected either way.

| feature | 2023 | 2024 | 2025 | 2026 | verdict |
|---|---|---|---|---|---|
| r1 | -0.0206 (t=-23.0) | -0.0195 (t=-39.9) | -0.0189 (t=-39.8) | -0.0163 (t=-30.5) | keep |
| r5 | -0.0248 (t=-23.8) | -0.0243 (t=-43.5) | -0.0206 (t=-38.5) | -0.0163 (t=-27.3) | keep |
| r15 | -0.0262 (t=-22.5) | -0.0266 (t=-42.4) | -0.0208 (t=-35.4) | -0.0164 (t=-25.5) | keep |
| r30 | -0.0247 (t=-19.8) | -0.0271 (t=-40.0) | -0.0201 (t=-31.9) | -0.0145 (t=-21.4) | keep |
| r60 | -0.0247 (t=-18.3) | -0.0247 (t=-33.5) | -0.0167 (t=-24.9) | -0.0100 (t=-13.9) | keep |
| r_open | -0.0186 (t=-15.2) | -0.0214 (t=-31.6) | -0.0083 (t=-13.3) | -0.0087 (t=-12.4) | keep |
| range_pos | -0.0097 (t=-7.3) | -0.0162 (t=-22.8) | -0.0069 (t=-10.3) | -0.0038 (t=-5.2) | keep |
| vwap_dev | -0.0049 (t=-3.7) | -0.0056 (t=-7.8) | +0.0025 (t=3.8) | +0.0050 (t=7.3) | flips |
| rv30 | -0.0061 (t=-3.4) | -0.0026 (t=-2.8) | -0.0133 (t=-15.8) | -0.0016 (t=-1.6) | keep |
| rv_open | -0.0222 (t=-11.3) | -0.0171 (t=-17.0) | -0.0287 (t=-32.3) | -0.0200 (t=-18.6) | keep |
| relvol5 | +0.0339 (t=35.4) | +0.0254 (t=50.3) | +0.0182 (t=35.2) | +0.0206 (t=35.6) | keep |
| relvol15 | +0.0376 (t=37.1) | +0.0287 (t=53.3) | +0.0220 (t=39.2) | +0.0232 (t=36.6) | keep |
| turn_liq | +0.0329 (t=24.3) | +0.0030 (t=3.8) | -0.0040 (t=-6.0) | +0.0141 (t=19.0) | flips |
| i_r5 |  |  |  |  | context |
| i_r15 |  |  |  |  | context |
| i_r30 |  |  |  |  | context |
| i_r60 |  |  |  |  | context |
| i_r_open |  |  |  |  | context |
| x_r15 | -0.0262 (t=-22.5) | -0.0266 (t=-42.4) | -0.0208 (t=-35.4) | -0.0164 (t=-25.5) | keep |
| x_r60 | -0.0247 (t=-18.3) | -0.0247 (t=-33.5) | -0.0167 (t=-24.9) | -0.0100 (t=-13.9) | keep |
| imb5 | -0.0139 (t=-14.7) | -0.0153 (t=-30.8) | -0.0105 (t=-21.7) | -0.0090 (t=-16.9) | keep |
| imb15 | -0.0106 (t=-10.6) | -0.0144 (t=-27.4) | -0.0065 (t=-12.8) | -0.0057 (t=-10.3) | keep |
| imb30 | -0.0074 (t=-7.2) | -0.0139 (t=-25.5) | -0.0041 (t=-7.8) | -0.0031 (t=-5.5) | keep |
| clv5 | -0.0178 (t=-20.4) | -0.0164 (t=-36.1) | -0.0116 (t=-26.3) | -0.0122 (t=-24.8) | keep |
| clv15 | -0.0148 (t=-16.3) | -0.0155 (t=-33.3) | -0.0090 (t=-20.0) | -0.0106 (t=-21.3) | keep |
| clv30 | -0.0126 (t=-13.8) | -0.0156 (t=-32.7) | -0.0079 (t=-17.1) | -0.0090 (t=-18.1) | keep |
| vwret5 | -0.0158 (t=-16.2) | -0.0173 (t=-33.7) | -0.0114 (t=-23.3) | -0.0092 (t=-17.1) | keep |
| vwret15 | -0.0119 (t=-11.7) | -0.0155 (t=-29.2) | -0.0064 (t=-12.7) | -0.0055 (t=-10.1) | keep |
| vwret30 | -0.0087 (t=-8.3) | -0.0150 (t=-27.6) | -0.0038 (t=-7.5) | -0.0029 (t=-5.3) | keep |
| up5 | -0.0196 (t=-20.5) | -0.0188 (t=-37.0) | -0.0169 (t=-34.2) | -0.0152 (t=-27.6) | keep |
| up15 | -0.0205 (t=-20.3) | -0.0208 (t=-38.1) | -0.0196 (t=-36.7) | -0.0170 (t=-28.6) | keep |
| up30 | -0.0194 (t=-19.0) | -0.0206 (t=-36.3) | -0.0213 (t=-38.3) | -0.0172 (t=-28.4) | keep |
| impact15 | -0.0228 (t=-16.9) | -0.0021 (t=-2.9) | -0.0018 (t=-2.8) | -0.0120 (t=-17.2) | keep |
| impact30 | -0.0210 (t=-15.0) | -0.0002 (t=-0.2) | -0.0002 (t=-0.3) | -0.0105 (t=-14.5) | keep |
| turn_accel | +0.0063 (t=6.9) | +0.0025 (t=5.4) | -0.0008 (t=-1.9) | +0.0022 (t=4.6) | keep |
| ret_1d | -0.0146 (t=-13.3) | -0.0135 (t=-22.2) | -0.0208 (t=-33.6) | -0.0225 (t=-31.7) | keep |
| gap | +0.0005 (t=0.6) | +0.0027 (t=6.0) | +0.0038 (t=7.9) | +0.0026 (t=4.3) | keep |
| ret_5d | -0.0230 (t=-19.5) | -0.0093 (t=-13.8) | -0.0185 (t=-26.3) | -0.0262 (t=-33.1) | keep |
| ret_20d | -0.0126 (t=-10.6) | -0.0061 (t=-9.0) | -0.0111 (t=-15.1) | -0.0237 (t=-26.9) | keep |
| vol_20d | -0.0448 (t=-25.8) | -0.0332 (t=-39.7) | -0.0396 (t=-53.1) | -0.0144 (t=-18.8) | keep |
| dist_hi | +0.0193 (t=16.5) | +0.0231 (t=32.5) | +0.0121 (t=14.6) | +0.0049 (t=5.6) | keep |
| dist_lo | -0.0310 (t=-20.5) | -0.0093 (t=-12.6) | -0.0226 (t=-32.8) | -0.0050 (t=-5.6) | keep |
| turn_rel | -0.0022 (t=-2.1) | -0.0048 (t=-9.0) | -0.0055 (t=-10.6) | -0.0117 (t=-19.1) | keep |
| deliv_pct | +0.0572 (t=43.9) | +0.0439 (t=61.2) | +0.0555 (t=75.8) | +0.0372 (t=52.8) | keep |
| rs_20 | -0.0126 (t=-10.6) | -0.0061 (t=-9.0) | -0.0111 (t=-15.1) | -0.0237 (t=-26.9) | keep |
| m |  |  |  |  | context |
| m_left |  |  |  |  | context |
| dow |  |  |  |  | context |
| expiry |  |  |  |  | context |
| after_holiday |  |  |  |  | context |

### Reading the three horizons together

- **Reversal weakens with horizon but never turns.** The 5-minute return's IC goes −0.05 (15 min),
  −0.03 (60 min), −0.02 (session). The buying-pressure proxies follow the same curve. Half of the
  15-minute reversal is gone within the hour, which is the microstructure share.
- **Participation grows with horizon.** Relative volume goes +0.01 (15 min), +0.015 (60 min), +0.02 to
  +0.04 (session): a stock trading heavily at this clock time, relative to its own norm, does better
  into the close. The strongest intraday feature at the session horizon, and stable every year.
- **Deliverable percentage is the best daily feature at every horizon** and the single best feature
  at the session horizon: +0.04 to +0.06 every year. Yesterday's delivery-heavy names outperform into
  today's close. Twenty-day volatility is the mirror image, −0.015 to −0.045: calm names drift up,
  volatile names bleed. Distance from the high stays positive; distance from the low negative.
- **Daily momentum resolves at the session horizon into reversal.** Yesterday's, 5-day and 20-day
  returns are stably negative at the session horizon (−0.01 to −0.026), after flipping at the
  shorter ones. The gap turns slightly positive. The daily layer earns its place here, not at 15
  minutes.
- **VWAP deviation and range position change sign between 2024 and 2025** at 60 minutes and the
  session. Held out as flips at those horizons; the model can still read them through the clock.
- **Drop list per horizon.** 15 min: gap; flips held out: rv30, ret_5d, ret_20d, rs_20, dist_lo.
  60 min: flips held out: range_pos, rv30, ret_1d, ret_5d, ret_20d, rs_20. Session: flips held out:
  vwap_dev, turn_liq. Everything else stays. The first Fit is the 15-minute one, as planned, with the
  session Fit the one the daily layer argues for next.

## What was built for this (2026-10-24)

- `finance/ml/forecast.py`: `ROWS_SQL` with the five parameters, `fit_definition(horizon)`, the
  IC screen (`screen`, `screen_table`) and the CLI `python -m finance.ml.forecast --horizon N`.
- Rows no longer pass through a Python frame: `LoadRows` has DuckDB write the Parquet and hash it
  (`data.write_rows`, `data.file_hash`, order-independent, no Python-side copy); the fit child and
  the scorer read with `data.read_rows`, numerics as Float32 while reading. Twenty-two million rows
  at 45 features are about 4 GB as X; the frame beside it is freed before the fit. Not a batch
  iterator into xgboost yet; the box's 31 GB carries this size, and the iterator is the next step if
  a horizon's rows outgrow it.
- The inference table carries the query's extra columns (`r`, `scale`, `entry`, `sig1`, ...), so
  metrics can read the raw return: `prepare` returns them as `extra`, `_score` writes them.
- Metrics that need the table: `ic`, `ic_se`, `ic_points` (within `group`, default `ts`),
  `calibration_slope`, `calibration_monotone`, `net_return_raw`, `trades_raw` (predicted raw move =
  `pred_y × scale` against `threshold`, minus `cost_bps`); and `by: month` on any Evaluation for the
  per-month series. `forecast.evaluation_params(threshold)` builds the params.

## Results: forecast 15m v2 on 2026 (2026-10-24)

Fit `forecast 15m v2` (`apps/finance/fits/forecast_15m.json`): 44 features, fit on 2023-10-03 to
2025-12-31 (16,004,953 rows after the liquidity floor), five models. `Infer` with the label
`validation 2026` on 2026-01-02 to 2026-10-08 (6,069,732 rows), `Evaluate` at three thresholds with
10 bps round-trip cost. Every number below is out of sample. The first attempt (`forecast 15m`) had
6,295 NaN targets from the Muhurat evening session and was superseded.

| model | fit s | IC (se) | IC by month, min to max | r2 | calibration slope | net/trade at 0.1% (trades) | at 0.2% | at 0.5% |
|---|---|---|---|---|---|---|---|---|
| linear (ridge), the baseline | 130 | 0.0384 (0.0007) | 0.033 to 0.043 | +0.0005 | 0.80 | -8.8 bps (11,166) | -5.7 bps (2,941) | +6.3 bps (553) |
| xgboost depth 4, 500 trees | 104 | 0.0500 (0.0007) | 0.044 to 0.060 | -0.0027 | 0.36 | -8.5 bps (56,783) | -5.2 bps (13,269) | +31.2 bps (2,464) |
| xgboost depth 4, 1500 trees | 201 | 0.0481 (0.0007) | 0.042 to 0.058 | -0.0116 | 0.20 | -9.0 bps (105,041) | -9.1 bps (26,753) | +11.7 bps (4,373) |
| xgboost depth 6, 500 trees | 111 | 0.0473 (0.0007) | 0.043 to 0.057 | -0.0088 | 0.23 | -8.7 bps (105,337) | -5.8 bps (24,200) | +21.1 bps (3,025) |
| xgboost depth 6, 1500 trees | 222 | 0.0463 (0.0007) | 0.041 to 0.055 | -0.0237 | 0.14 | -8.7 bps (188,576) | -7.7 bps (47,295) | +8.5 bps (5,385) |
Net per trade is the mean raw 15-minute return over the rows whose predicted raw move (prediction ×
scale) exceeds the threshold, minus 10 bps. Trades are over the ten months.

The best model by IC, xgboost depth 4 with 500 trees, by month at the 0.5 percent threshold:

| month | 2026-01 | 2026-02 | 2026-03 | 2026-04 | 2026-05 | 2026-06 | 2026-07 | 2026-08 | 2026-09 | 2026-10 |
|---|---|---|---|---|---|---|---|---|---|---|
| net/trade at 0.5%, bps | +17 | +43 | +5 | +8 | -15 | +2 | +4 | +92 | -74 | -63 |
| trades | 121 | 1739 | 165 | 73 | 134 | 72 | 70 | 37 | 46 | 7 |
| IC | 0.045 | 0.048 | 0.050 | 0.044 | 0.051 | 0.046 | 0.054 | 0.060 | 0.051 | 0.052 |

### Reading

- **The signal is real and holds every month.** The linear baseline has an IC of 0.038 and every
  xgboost point beats it, the best at 0.050 with a standard error of 0.0007; the monthly IC of the best
  model never drops below 0.044 and rises through the year. This is at the top of the 0.02 to 0.05
  range the design expected, and the within-month stability is the part that was not guaranteed.
- **Explained variance is nil and that is expected**: r² is zero for the baseline and negative for
  the trees. Fifteen-minute returns are noise with a thin ordering signal on top; the ordering is what
  IC measures and what a trader uses.
- **The trees are badly calibrated and more trees make it worse**: slope 0.36 at 500 trees, 0.14 at
  1500 for depth 6. They predict moves three to seven times larger than what happens. The ordering
  survives, the magnitude does not; a consumer must rescale predictions by the recorded slope before
  comparing them with a cost. Depth 4 with 500 trees is the least overfit and the best on every count,
  so the next search goes smaller, not larger: fewer trees, lower learning rate, stronger
  regularisation.
- **After costs, only the top slice pays.** At 0.1 and 0.2 percent predicted moves every model loses
  5 to 9 bps a trade: the typical 15-minute move does not clear the cost, as the design said. At 0.5
  percent the best model makes +31 bps a trade over 2,464 trades, about twelve a day. But 1,739 of those
  trades are in February 2026, a volatile month, and the months since have tens of trades each with
  noisy means, two of them negative. The honest statement is: the forecast selects a small set of
  moments with positive expectancy after costs, concentrated in high-volatility periods, and the
  sample outside those periods is too thin to put a number on yet.
- **The threshold is the wrong knob.** Because the trees are uncalibrated, a fixed predicted-move
  threshold selects different quantiles for different models; February dominating is a symptom. The
  evaluation should also select by prediction quantile within the decision point (the top k names at
  each t), which is how a cross-sectional model is used. `top_k` as a metric param is the next change.

### Next

1. Rescale by the calibration slope, and evaluate by within-point quantile as well as by threshold.
2. A smaller search around depth 3 and 4, 200 to 500 trees, learning rate 0.03 to 0.05, more
   regularisation; `lightgbm` beside xgboost.
3. The 60-minute and one-session Fits from the same query: the IC screen says the daily layer earns
   its place there.
4. The volatility forecast (σ) as its own Fit, since position sizing needs it and the trailing window
   is crude around the open.
5. `make kite-login` so the minute data runs past 2026-10-08 and the validation window grows.

### Review of the best model (xgboost depth 4, 500 trees)

How to review any model: the console (http://127.0.0.1:9400) for the Fit → FitRun → Model → InferRun →
Evaluation chain and failures; the tables above for the comparison; the prediction table
`parquet/inference/<infer run id>.parquet` (key, truth, prediction, raw return, scale) for any
question a registered metric does not answer; the model directory `models/<model id>/` for the fitted
object (`model.ubj`, `schema.json`, `fit.json`).

Share of total gain, from the fitted booster: `i_r_open` 10.8%, `clv15` 10.4%, `clv30` 9.6%,
`turn_rel` 7.1%, `ret_1d` 5.5%, `rv_open` 5.5%, `vwap_dev` 5.4%, `clv5` 4.9%, `m` 4.8%, `i_r15` 3.8%,
`deliv_pct` 3.3%, `dist_hi` 3.2%, `i_r60` 3.0%, `dow` 2.3%, `range_pos` 2.0%; the other 29 share
18.4%. The market's move since the open is the single biggest input; the buying-pressure proxies are
the strongest stock-specific ones; the clock lets the model use the market move differently through
the day.

Realised return by prediction decile within each decision point, 2026:

| decile | predicted, scaled | realised, scaled | realised, bps | share of ups |
|---|---|---|---|---|
| 1 (lowest) | −0.086 | −0.082 | −2.3 | 41.5% |
| 2 | −0.048 | −0.047 | −1.3 | 44.3% |
| 5 | −0.013 | −0.015 | −0.4 | 46.5% |
| 9 | +0.028 | +0.023 | +0.8 | 48.6% |
| 10 (highest) | +0.061 | +0.048 | +1.5 | 49.6% |

- **The ranking is clean, monotone, and the decile means match the predictions.** The poor
  calibration slope is the extreme individual predictions overshooting, not the ranking.
- **Top minus bottom decile is 3.9 bps per 15 minutes before costs**, against a round trip near 10
  bps. A decile long-short loses money; only the far tail (predicted moves above 0.5 percent) clears
  cost, and that tail is thin and volatility-clustered, as the monthly table shows.
- **Every decile has a negative mean and under half up-moves**: the average 15-minute return over all
  6 million rows of 2026 is about −0.5 bps. Either 2026 to date drifted down intraday, or entering at
  the next bar's open and exiting at a later close carries a small systematic bias (an open printing
  on the ask side would do it). Checked the same day: close to next open averages −0.02 bps, so the
  entry is clean; a single minute bar's return averages −0.01 to −0.03 bps in every year 2023 to 2026,
  which over 15 bars is the −0.3 bps seen. Liquid NSE names drift slightly down inside the session,
  equal-weighted, and the positive drift is overnight. A fact of the market, not of the target; a
  long-only intraday consumer fights it, a long-short use of the ranking nets it out.

### In plain words

At a given minute, for each liquid stock, the model guesses whether it will be higher or lower 15
minutes later and by roughly how much, trained on 2023 to 2025 and tested on 2026, which it never saw.
It guesses better than a coin flip, modestly and in every month. The stocks it is most confident about
rise about 1.5 basis points (0.015 percent) over the next 15 minutes and the ones it is most negative
about fall about 2.3; the gap, around 4 basis points, is less than the roughly 10 basis points a
round trip costs on NSE, so acting on a typical call loses money. Only its rare large calls, half a
percent or more, made money after costs in 2026, and most of those were in one volatile month. It
exaggerates the size of moves by three to seven times, which a scaling factor corrects. It relies most
on how the market has moved since the open, on whether recent minutes closed near their highs on
heavy volume, and on how active the stock is versus its norm. Stocks drift slightly down during the
session every year, with the gains overnight, so a day-only buyer fights a small headwind. The signal
is real; at 15 minutes it is too small to trade on its own. Longer horizons and using the ranking to
choose what to hold, rather than to time 15-minute trades, are the two directions worth trying.

## Operational notes (2026-10-24)

- A fit holds about 7 GB (16 million rows × 44 features as float32, plus the frame while it loads).
  The worker runs one deferred run at a time (`BACKEND_BUILDER_OUTBOX_CONCURRENCY=1`); four at once
  reached 30 of 31 GB.
- A rows build is about 10 minutes and one Postgres transaction; `LoadRows` caps DuckDB at 12 GB.
  During one build Postgres lost a backend and restarted, which dropped the run's connection; the
  runtime retried the run and it committed. Postgres now has a memory floor and the lowest OOM
  priority in `compose.yaml`.
- `Fit.name` is unique: a corrected definition is a new name (`forecast 15m v2`).
- Scoring 6 million rows is 9 minutes, almost all of it the rows build; the prediction itself is
  seconds. Evaluations are 4 seconds each.

## Second round, in progress (2026-10-24)

Agreed: all the features discussed (attributes, peers and sectors, breadth and regime, overseas,
participation, opening range, calendar), horizons 15/30/45/60 minutes on minute rows and 1 and 21
sessions on daily rows, two splits (fit through 2025 and test on 2026; fit on 2026 H1 and test on
2026 H2, both judged on July to October), top-k evaluation with the calibration rescale, and the same
daily-grain features for the US market to compare how well it predicts against the NSE.

Built: `ROWS_SQL` with 99 features; `daily_sql(market)` for NSE and US from one template
(`DAILY_MARKETS`), 64 and 53 features; the Kite universe widened to the Nifty 500 with sector labels
from NSE's constituent files and a per-symbol history backfill; the US screener snapshot and
`us_sectors`; the sector indices (Kite) and sector ETFs (Yahoo) as context. The screens and fits
follow once the backfills have landed.

## Second round: the screens (2026-10-24, evening)

Seven screens, each a `Screen` run (the last five as one `ScreenSweep`), tables in
`parquet/screens/`. The minute rows carry 99 features (70 screenable, 29 context), the NSE daily rows
64, the US daily rows 53; categoricals and context features have no cross-sectional IC and are not
in the tables. "Stable" means significant with one sign in every year it is significant; a feature
is listed when its mean IC over the years is at least 0.01.

#### 15 min: 28,180,780 rows, 2023 to 2026

| feature | mean IC | range over years |
|---|---|---|
| r5 | -0.058 | -0.067 to -0.048 |
| r1 | -0.052 | -0.057 to -0.047 |
| r15 | -0.047 | -0.057 to -0.039 |
| x_r15 | -0.047 | -0.057 to -0.039 |
| xp_r15 | -0.046 | -0.054 to -0.039 |
| xs_r15 | -0.046 | -0.055 to -0.039 |
| up5 | -0.043 | -0.051 to -0.037 |
| vwret5 | -0.043 | -0.052 to -0.035 |
| imb5 | -0.037 | -0.046 to -0.032 |
| r30 | -0.036 | -0.044 to -0.030 |
| up15 | -0.033 | -0.039 to -0.029 |
| clv5 | -0.033 | -0.041 to -0.028 |
| vwret15 | -0.027 | -0.035 to -0.021 |
| up30 | -0.025 | -0.028 to -0.021 |
| imb15 | -0.024 | -0.032 to -0.020 |
| r60 | -0.023 | -0.031 to -0.016 |

Flips (sign changes between years): none. Noise: none.

#### 60 min: 24,581,069 rows, 2023 to 2026

| feature | mean IC | range over years |
|---|---|---|
| r5 | -0.029 | -0.034 to -0.023 |
| r1 | -0.026 | -0.029 to -0.023 |
| xp_r15 | -0.025 | -0.030 to -0.020 |
| r15 | -0.025 | -0.031 to -0.019 |
| x_r15 | -0.025 | -0.031 to -0.019 |
| xs_r15 | -0.025 | -0.030 to -0.020 |
| vol_60d | -0.024 | -0.037 to -0.013 |
| deliv_pct | +0.023 | +0.019 to +0.026 |
| up5 | -0.022 | -0.026 to -0.019 |
| vwret5 | -0.020 | -0.026 to -0.016 |
| vol_20d | -0.020 | -0.035 to -0.010 |
| r30 | -0.019 | -0.025 to -0.013 |
| up15 | -0.018 | -0.021 to -0.015 |
| clv5 | -0.018 | -0.022 to -0.013 |
| imb5 | -0.018 | -0.022 to -0.014 |
| in_small250 | -0.016 | -0.029 to -0.009 |

Flips (sign changes between years): none. Noise: none.

#### 375 min (to the close): 29,387,001 rows, 2023 to 2026

| feature | mean IC | range over years |
|---|---|---|
| deliv_pct | +0.044 | +0.035 to +0.048 |
| vol_60d | -0.033 | -0.043 to -0.016 |
| vol_20d | -0.030 | -0.042 to -0.014 |
| relvol15 | +0.027 | +0.020 to +0.040 |
| in_small250 | -0.027 | -0.044 to -0.017 |
| xp_r15 | -0.025 | -0.029 to -0.021 |
| xs_r15 | -0.024 | -0.028 to -0.021 |
| x_r15 | -0.024 | -0.028 to -0.019 |
| r15 | -0.024 | -0.028 to -0.019 |
| relvol5 | +0.024 | +0.016 to +0.036 |
| r5 | -0.023 | -0.026 to -0.019 |
| r30 | -0.023 | -0.028 to -0.018 |
| xp_r60 | -0.022 | -0.028 to -0.016 |
| up15 | -0.020 | -0.022 to -0.018 |
| up30 | -0.020 | -0.023 to -0.018 |
| r1 | -0.020 | -0.021 to -0.017 |

Flips (sign changes between years): turn_liq, impact30. Noise: none.

#### NSE 1 session: 1,668,749 rows, 2021 to 2026

| feature | mean IC | range over years |
|---|---|---|
| vol_20d | +0.052 | +0.038 to +0.075 |
| log_turn20 | +0.050 | +0.043 to +0.055 |
| gap_d | -0.044 | -0.051 to -0.041 |
| in_n500 | +0.043 | +0.033 to +0.048 |
| ret_1d | -0.039 | -0.051 to -0.027 |
| trades_rel | -0.033 | -0.044 to -0.025 |
| log_px | +0.032 | +0.017 to +0.046 |
| avg_trade | +0.031 | +0.027 to +0.036 |
| vol_60d | +0.030 | +0.012 to +0.065 |
| turn_rel | -0.030 | -0.040 to -0.020 |
| log_trades | +0.029 | +0.024 to +0.034 |
| sp_5d | -0.029 | -0.040 to -0.013 |
| ret_5d | -0.026 | -0.040 to -0.011 |
| relvol_day | -0.024 | -0.035 to -0.015 |
| p_ret_1d | +0.024 | +0.016 to +0.029 |
| in_n50 | +0.023 | +0.018 to +0.031 |

Flips (sign changes between years): dist_hi. Noise: xs_ret_20d, sp_20d, rv_day, r_last60, r_first60.

#### NSE 21 sessions: 1,631,829 rows, 2021 to 2026

| feature | mean IC | range over years |
|---|---|---|
| tick_bps | +0.076 | +0.041 to +0.098 |
| p_ret_20d | +0.038 | +0.014 to +0.061 |
| deliv_pct | +0.037 | +0.018 to +0.066 |
| p_ret_5d | +0.036 | +0.007 to +0.048 |
| p_ret_1d | +0.032 | +0.009 to +0.045 |
| s_ret_5d | +0.027 | -0.007 to +0.052 |
| s_ret_1d | +0.026 | +0.005 to +0.041 |
| in_small250 | +0.025 | +0.013 to +0.039 |
| rs_60 | +0.025 | -0.007 to +0.070 |
| ret_60d | +0.025 | -0.007 to +0.070 |
| pw_ret_20d | +0.024 | -0.002 to +0.046 |
| dist_lo | +0.020 | -0.007 to +0.048 |
| sp_60d | +0.018 | -0.012 to +0.056 |
| sp_5d | -0.013 | -0.023 to +0.004 |
| ret_1d | -0.011 | -0.016 to -0.001 |
| gap_d | -0.010 | -0.020 to +0.008 |

Flips (sign changes between years): in_n500, dist_hi, avg_trade, s_ret_20d, log_turn20, ret_20d, rs_20, in_n50, in_mid100, vol_60d, xs_ret_20d, sp_20d, log_trades. Noise: r_first60.

#### US 1 session: 4,872,101 rows, 2015 to 2026

| feature | mean IC | range over years |
|---|---|---|
| vol_20d | -0.018 | -0.038 to -0.002 |
| vol_60d | -0.018 | -0.030 to -0.002 |
| in_n500 | +0.015 | +0.005 to +0.025 |
| dist_hi | +0.014 | +0.004 to +0.024 |
| in_small250 | -0.013 | -0.023 to -0.003 |

Flips (sign changes between years): none. Noise: p_ret_1d, sp_60d, ret_60d, rs_60, sp_20d, xs_ret_20d, ret_20d, rs_20, p_ret_20d.

#### US 21 sessions: 4,822,191 rows, 2015 to 2026

| feature | mean IC | range over years |
|---|---|---|
| in_n500 | +0.064 | +0.028 to +0.115 |
| in_small250 | -0.055 | -0.093 to -0.011 |
| vol_20d | -0.046 | -0.153 to +0.015 |
| vol_60d | -0.043 | -0.160 to +0.020 |
| dist_hi | +0.038 | -0.009 to +0.087 |
| in_n50 | +0.034 | +0.007 to +0.067 |

Flips (sign changes between years): log_turn20, log_px, tick_bps, p_ret_20d, in_mid100, s_ret_20d, ret_20d, rs_20, sp_5d. Noise: s_ret_1d, clv_d, range_pos_d.

### Reading across the seven

- **Minutes: reversal, fading with horizon.** The stock's own recent return, its return against its
  industry peers and against its sector index are the strongest features at 15 minutes (−0.05 to
  −0.06), half that at 60 and a third at the close. Peers and sector add little beyond the stock's
  own move at these horizons: `xp_r15` and `xs_r15` track `r15` almost exactly. Relative volume is
  the one stable positive intraday feature and grows with horizon (+0.01 at 15 minutes, +0.027 at the
  close).
- **The slow layer takes over by the close.** At 375 minutes delivery percentage is the single best
  feature (+0.044, every year +0.035 to +0.048), followed by 20- and 60-day volatility (−0.03, calm
  names drift up), small-cap membership (−0.027) and relative volume (+0.027). This is what the
  first round's screen predicted.
- **NSE one session ahead is the richest horizon.** Fourteen features above +/−0.02 and all stable
  over five years: 20-day volatility (+0.052; here the target is the next session's open-to-close
  move, so volatile names move more intraday relative to their 20-day daily volatility, and the
  sign must not be read as "volatility pays"), 20-session turnover (+0.05), today's gap (−0.044),
  Nifty 500 membership (+0.043), yesterday's return (−0.039), trades against the norm (−0.033),
  price level (+0.032), average trade size (+0.031), peers' and the sector's recent returns (+0.02),
  distance from the low (+0.022). Liquid, large, calm, un-gapped names have the better next session.
- **NSE one month ahead is where flips appear.** Tick size over price (+0.076, cheap stocks win the
  month, a small-cap effect), peers' 20-day return (+0.038) and delivery percentage (+0.037) are
  stable; Nifty 500 membership, distance from the high, average trade size, turnover and 20-day
  momentum all flip sign between years. Five years is three regimes; monthly horizons need the long
  daily history and a model that conditions on regime features rather than fixed signs.
- **The US is weaker and larger-cap-led.** One session ahead, nothing reaches 0.02: 20- and 60-day
  volatility −0.018, large-cap membership +0.015, small-cap −0.013, distance from the high +0.014.
  One month ahead, large-cap membership (+0.064) and small-cap (−0.055) dominate with low
  volatility (−0.046): the 2015 to 2026 large-cap decade, stable each year but a regime fact more
  than a stock-selection signal. Price level, turnover, momentum and sector returns flip.
- **US versus NSE, as asked.** The NSE has more and stronger stock-level signal at every horizon the
  two share: the one-session screen has fourteen stable features above 0.02 against the US's none;
  the monthly screen has stable microstructure and delivery signals the US data cannot express
  (no delivery, no trade counts). The US signal that exists is size and volatility, which is
  mostly what the indices did. The fits will say whether the US combination is still worth a model.

### Caveats

- The US daily rows carry a turnover floor of 10 million dollars a day (median over 20 sessions),
  about 1,700 stocks; the NSE floor is 1 crore, about 1,300 stocks. The floors decide the universe
  the ICs describe.
- The NSE sector index returns at daily grain and the sector ETFs in the US are joined by the
  industry's mapped index; industries without one get the market index and so a zero spread.
- The screens are univariate. Context features (market move, breadth, regime, overseas, clock) and
  interactions are only measured by the fits.

### Next: the fits

Seven Fits from these screens, features kept where stable with mean IC above 0.003 plus every
context and categorical feature, and two runs each: fit through 2025 and fit on 2026 January to
June, both validated on 2026 July to October with IC, calibration, top-k and threshold metrics after
costs. Run as `FitSweep` and `ValidateSweep` reactions.

## The session study: overnight, daytime short, both (2026-10-24, night)

The owner's question: buy about 15 minutes before the close and sell 15 minutes after the open,
every day. Which stocks pay for that, how consistently, what about indices, the US and
commodities, and what if we also short the daytime? Three variants per instrument per year:
**overnight** (long from the close to the next open), **daytime short** (short from the open to
the close) and **both**. The `SessionStudy` reaction computes them from the daily tables for
`nse`, `us` and `mcx`; the minute-bar check below uses the Kite bars at 15:14 and 09:29.

### The official open is not a price you get

The daily tables' open is the pre-open auction print. For ETFs and thin stocks it sits about one
percent above the previous close almost every day and is gone within minutes, so the daily-table
study makes NIFTYBEES look like +70% a year overnight with a 73% hit rate. Measured on the minute
bars at the owner's actual times, NIFTYBEES overnight is +8% in 2024, +9% in 2025 and −8% so far
in 2026: the index's own overnight drift, nothing more. Every NSE number below is from the minute
bars (602 stocks with Kite history since October 2023, liquidity floor ₹5 crore a day). The US has
no minute data here, so its numbers are open-to-close and carry this caveat; for SPY-sized names
the auction print is close to tradeable, for small caps it is not.

### NSE: the whole market, per liquid stock, median across stocks

| year | overnight | hit | daytime short | hit | both | buy and hold |
|---|---|---|---|---|---|---|
| 2024 | +50% | 95% of stocks positive | +26% | 78% | +77% | +21% |
| 2025 | +13% | 66% | +6% | 58% | +19% | +0% |
| 2026 to Oct 22 | +3% | 57% | +2% | 55% | +7% | −2% |

The edge is fading fast: 2.2 bps a night and 1.4 bps a day for the median stock in 2026, against
about 4 bps round trip in a stock future and about 25 bps round trip for delivery in cash (STT
0.1% each side). For the median stock neither variant covers its costs in 2026.

### NSE indices (minute bars, % per year)

| index | 2024 night | 2025 night | 2026 night | 2024 day short | 2025 day short | 2026 day short | 2026 both | 2026 buy and hold |
|---|---|---|---|---|---|---|---|---|
| Nifty 50 | +15 | +8 | −3 | +7 | −2 | +12 | +9 | −15 |
| Nifty Bank | +10 | +9 | −4 | +5 | −7 | +5 | +1 | −10 |
| Nifty Next 50 | +46 | +18 | −2 | +22 | +16 | +2 | 0 | −6 |
| Nifty Midcap 150 | +40 | +11 | −1 | +19 | +6 | +5 | +4 | −8 |
| Nifty IT | +32 | −7 | +5 | +12 | +6 | +20 | +25 | −18 |
| Nifty PSU Bank | +45 | +36 | +14 | +32 | +9 | +18 | +31 | −6 |

Overnight carried nearly all of the index return in 2024 and 2025 and has stopped in 2026. The
daytime short on Nifty 50 is positive in 2026 only because the market fell. Tradeable with
futures only: Nifty 50, Bank, Financial Services, Midcap Select; no futures on Next 50, Midcap
150, PSU Bank or the sector indices, so their rows are what the prices did, not a trade.

### NSE: which stocks, and does picking them persist

Ranked by the worst of the three periods (2026 annualised), among stocks positive in all three.
Returns in % per year before costs; turnover in ₹ crore a day; hit = fraction of sessions positive.

Overnight (buy 15:14, sell 09:29):

| stock | ₹cr/day | 2024 | 2025 | 2026 ytd | hit | buy and hold 2024 / 25 / 26 |
|---|---|---|---|---|---|---|
| NETWEB | 84 | +168 | +86 | +80 | 58% | +89 / +7 / +35 |
| APOLLO | 71 | +134 | +83 | +77 | 60% | +1 / +83 / +31 |
| CUPID | 24 | +114 | +77 | +97 | 57% | +26 / +193 / +107 |
| AXISCADES | 11 | +126 | +74 | +69 | 57% | +1 / +70 / +37 |
| VIYASH | 14 | +83 | +71 | +60 | 56% | +47 / +12 / −2 |
| SYRMA | 60 | +87 | +79 | +54 | 59% | −6 / +14 / +69 |
| ASTRAMICRO | 29 | +105 | +69 | +53 | 58% | +25 / +23 / +42 |
| PARAS | 35 | +96 | +77 | +53 | 58% | +29 / +31 / +54 |
| TDPOWERSYS | 34 | +132 | +101 | +50 | 59% | +51 / +41 / +64 |
| REDINGTON | 52 | +72 | +62 | +50 | 54% | +13 / +32 / +29 |
| BSE | 974 | +93 | +87 | +29 | 56% | +90 / +37 / +33 |
| MCX | 354 | +48 | +53 | +30 | 57% | +69 / +56 / +20 |

Daytime short (sell 09:29, cover 15:14):

| stock | ₹cr/day | 2024 | 2025 | 2026 ytd | hit | buy and hold 2024 / 25 / 26 |
|---|---|---|---|---|---|---|
| INOXWIND | 93 | +102 | +96 | +41 | 60% | +37 / −41 / −45 |
| JBMA | 24 | +83 | +78 | +56 | 64% | −5 / −21 / −25 |
| OLECTRA | 66 | +103 | +69 | +59 | 66% | +8 / −21 / −23 |
| SWSOLAR | 48 | +123 | +108 | +52 | 62% | +6 / −79 / −41 |
| SWANCORP | 67 | +60 | +65 | +56 | 59% | +35 / −44 / −65 |
| JWL | 45 | +78 | +102 | +48 | 62% | +46 / −41 / −66 |
| TEJASNET | 65 | +67 | +108 | +46 | 63% | +30 / −96 / −10 |
| HCC | 57 | +90 | +88 | +47 | 61% | +33 / −54 / −9 |
| RVNL | 223 | +54 | +48 | +38 | 59% | +86 / −17 / −48 |

Both:

| stock | ₹cr/day | 2024 | 2025 | 2026 ytd | hit |
|---|---|---|---|---|---|
| NETWEB | 84 | +251 | +162 | +125 | 61% |
| VIYASH | 14 | +122 | +127 | +123 | 60% |
| OLECTRA | 66 | +210 | +118 | +98 | 63% |
| JBMA | 24 | +164 | +135 | +89 | 62% |
| TIMETECHNO | 42 | +152 | +117 | +88 | 60% |
| POCL | 16 | +284 | +141 | +85 | 59% |
| TEJASNET | 65 | +164 | +119 | +84 | 59% |
| HCC | 57 | +211 | +124 | +83 | 60% |
| ATGL | 62 | +113 | +104 | +81 | 62% |
| HBLENGINE | 80 | +251 | +124 | +76 | 63% |

The two lists are two different kinds of stock. The overnight names are small-cap electronics,
defence and capital-goods stocks that also rose over the period: a stock that is bid up is bid up
at the open. The daytime-short names are stocks that fell for two years (wind, solar, rail, infra):
they open firm and sell off through the day. Both is best where the two coincide (NETWEB, HCC,
OLECTRA).

Selection out of sample, which is the only test that matters:

| picked on | variant | top 20 next period | universe median | top 20 positive |
|---|---|---|---|---|
| 2024 | overnight, 2025 | +45% | +13% | 90% |
| 2024 | daytime short, 2025 | +41% | +6% | 85% |
| 2024 | both, 2025 | +95% | +19% | 100% |
| 2024 + 2025 (worst of two) | overnight, 2026 ytd | +41% | +3% | 95% |
| 2024 + 2025 | daytime short, 2026 ytd | +31% | +2% | 90% |
| 2024 + 2025 | both, 2026 ytd | +50% | +7% | 95% |

The stock ranking persists: the top twenty of one period keep roughly 40% a year in the next while
the market's own edge has gone to zero. That is about 20 bps a night and 15 bps a day gross for
the picked names. Two years of history and one out-of-sample year is thin evidence, and the names
are small: a few crore a day of turnover, so the strategy is capped at a few lakh of principal per
stock before it moves the open.

### What it costs, and the minimum principal (India)

| leg | instrument | round trip | net on a picked stock (gross 20 bps night / 15 bps day) |
|---|---|---|---|
| overnight | cash delivery (CNC) | ~25 bps (STT 0.1% both sides, fees) | **negative**: −5 bps a night |
| overnight | stock future | ~4 bps | +16 bps a night |
| daytime short | cash intraday (MIS) | ~5 bps (STT 0.025% on the sell, fees) | +10 bps a day |
| daytime short | stock future | ~4 bps | +11 bps a day |
| both | stock future | ~8 bps | +27 bps a session |

So: the **overnight hold only works in a stock that has futures**; in cash the STT alone is more
than the edge. Almost none of the stocks in the tables are in the F&O list (BSE, MCX, RVNL and
ATGL are; the rest must be checked against the live NSE list, our Kite snapshot holds only the
cash segment). The **daytime short works in cash** with an intraday order on any of them, and
needs no futures. For the picked daytime-short names in 2026, +31% gross less ~10% of costs over
199 sessions leaves about +20% a year on the capital, which intraday leverage (about 5x on MIS)
multiplies, with the matching risk.

Minimum principal: one cash share, so a few hundred rupees, for the daytime short; one futures
lot, ₹5 to 15 lakh notional and ₹1 to 3 lakh margin, for the overnight leg. For the index legs the
Nifty 50 future is one lot of 75 (about ₹19 lakh notional, ~₹2 lakh margin), and on the 2026
numbers the both variant on Nifty 50 (+9% gross, 199 sessions, ~1.5 bps a session in costs)
nets about +6%: ₹1,000 of notional becomes about ₹1,060, against ₹850 for buy and hold.

### United States (open to close, daily tables)

| year | SPY night | SPY day short | SPY both | SPY buy and hold | QQQ night | IWM night | median stock night | median stock day short |
|---|---|---|---|---|---|---|---|---|
| 2021 | +15 | −9 | +6 | +26 | +15 | +30 | +24 | +8 |
| 2022 | −16 | +6 | −10 | −23 | −24 | −10 | −10 | +7 |
| 2023 | +3 | −18 | −15 | +22 | +8 | +7 | +6 | −6 |
| 2024 | +21 | −1 | +21 | +21 | +29 | +28 | +19 | +9 |
| 2025 | +7 | −8 | −2 | +16 | +15 | +2 | +2 | −3 |
| 2026 ytd | +7 | −3 | +5 | +11 | +8 | +8 | +7 | +5 |

The US overnight carries most of the index return in most years, but it is not a hedge: 2022 was
−16% overnight on SPY, and the daytime short on the US indices is negative in six of nine years.
Nothing to trade at the index level. Per stock, requiring a price of at least $20, $20 million a
day and a positive year every year 2021 to 2026:

| overnight | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 ytd | worst |
|---|---|---|---|---|---|---|---|
| YPF | +66 | +45 | +60 | +41 | +58 | +26 | +33 |
| CCJ | +47 | +26 | +24 | +50 | +56 | +21 | +24 |
| WFRD | +83 | +28 | +28 | +45 | +24 | +34 | +24 |
| TDW | +55 | +22 | +20 | +54 | +28 | +21 | +20 |
| MOD | +23 | +24 | +30 | +39 | +20 | +61 | +20 |
| AVAV | +61 | +19 | +61 | +62 | +45 | +76 | +19 |
| STRL | +75 | +18 | +65 | +58 | +59 | +56 | +18 |

| daytime short | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 ytd | worst |
|---|---|---|---|---|---|---|---|
| MP | +46 | +51 | +41 | +58 | +50 | +61 | +41 |
| FUBO | +134 | +132 | +32 | +143 | +44 | +49 | +32 |
| OUST | +156 | +133 | +89 | +18 | +24 | +24 | +18 |

Same pattern as India: the overnight names are energy, uranium, defence and offshore stocks that
rose for five years (buy and hold +180% to +300%); the daytime-short names fell. The US list is
at the official open, so take 5 to 10 bps a session off for the open-print effect; the overnight
names still clear that. Costs are near zero in the US (no STT; commissions ~0, SEC fee on the
sell). Minimum principal: one share, but a daytime short every day needs a margin account above
the $25,000 pattern-day-trader floor; the overnight hold alone does not.

### MCX commodities

Continuous front-month series, pairs straddling a roll excluded, contracts with at least a
thousand lots a day. No consistent edge:

| | 2022 | 2023 | 2024 | 2025 | 2026 ytd |
|---|---|---|---|---|---|
| GOLD night | +3 | +7 | +7 | +12 | +8 |
| GOLD day short | −7 | −5 | −8 | −35 | +2 |
| SILVER night | −5 | +9 | +11 | +14 | +12 |
| CRUDEOIL night | +4 | −5 | +3 | −1 | +46 |
| NATURALGAS day short | +7 | +84 | +41 | +31 | +15 |

Gold's night is positive five years running but at 1 to 3 bps a night against roughly 2 to 3 bps
round trip, so nothing is left. The daytime short on natural gas looks consistent from 2023 but
is the three-year fall in the price itself (buy and hold −109% in 2023); 2021 and 2022 were the
other sign. The MCX night (23:30 to 09:00) carries the London and New York sessions, and the
study says the Indian close does not systematically misprice them.

### What this says for the model

The overnight gap is the richest single return in the data and it is stock-specific and
persistent at the one-year scale, which is exactly the kind of structure the Fit can learn: the
"close to next open" horizon should become a target for both markets (noted under Next in the
second round), with the stock's own trailing overnight return as a feature. The daily-table open
must not be the fill price in that target for NSE; the minute bars at 09:29 should be.

## The window study: every pair of times, and the weekend question (2026-10-25)

The session study's one window (15:14 to 09:29) generalised to every pair of points on a 15-minute
grid: 09:29, 09:44, … 15:14 and the 15:29 close, 25 points. Same-day windows are the 300 pairs
with entry before exit; cross-session windows are all 625 pairs of an entry today and an exit in the
next session, split into plain **nights** (the next session is the next calendar day) and **breaks**
(a weekend or holiday in between). Per stock per year: sessions, total, mean, hit rate. 576 liquid
NSE stocks (₹5 crore a day), Kite minute bars from October 2023, 3.1 million summary rows. The
`WindowStudy` reaction runs it in ten symbol buckets (the whole universe at once, 420 million
pairs, ran DuckDB out of its 12 GB).

### Where the day's return lives, market-wide

Median across stocks of the yearly total, per window. The three families each have a clear shape:

- **Same day: the morning falls.** Every one of the twelve most consistently negative windows starts
  at 09:29 or soon after and ends between 13:14 and 14:29. Short 09:29 → 13:14 made the median
  stock +6% in 2023, +21% in 2024, +10% in 2025 and +8% so far in 2026 (about 10 bps a day in
  2023 and 2024, 4 bps a day since). Nothing inside the day is consistently positive: the best
  same-day long windows are a few minutes around 14:29 to 15:29 at 0 to 5% a year, which is noise.
- **Into the next session: buy mid-afternoon, not at the close.** The strongest cross-session
  windows all enter between 13:29 and 14:29 and exit between 09:29 and 11:44 the next morning:
  14:14 → 09:29 made +16%, +39%, +19%, +14% over 2023 to 2026 for the median stock (36, 21, 10
  and 10 bps a night). Entering at 15:14 instead of 14:14 gives up the last hour's rise, which is
  why the earlier session study's +50% in 2024 fell to +13% and +3%: the edge moved earlier in the
  afternoon and shrank, but did not vanish.
- **Breaks pay less than nights, and in 2026 they lose.** For the 15:14 → 09:29 window, the median
  stock made 24 bps on a plain night in 2024 against 11 bps across a weekend or holiday; 5.5 against
  2.2 bps in 2025; and in 2026, +7 bps on plain nights against **−20 bps** across breaks, with 45%
  of them positive. Across all 625 cross-session windows in 2026, 82% of break windows were
  negative for the median stock. The indices agree: Nifty 50 across breaks was +13 bps in 2024
  and +7 in 2025 (better than its plain nights then), and −14 bps in 2026. Holding a stock over a
  weekend or holiday is not a better version of the overnight trade; this year it was the worst one.

### Per stock: the best window, picked on 2024 and 2025 and scored on 2026

For each stock, the window with the highest worst-year total over 2024 and 2025, positive in both,
then that same window's 2026 result. The 2026 column is the only one that counts, since the pick
had 1,250 windows to choose from.

| family | stocks | picks in 2026 (mean / median / % positive) | all windows that year | top 20 picks in 2026 |
|---|---|---|---|---|
| same day, long | 515 | +1.4% / +1.1% / 52% | −2.2% median, 42% | +8.6%, 60% |
| same day, short | 517 | +7.7% / +5.2% / 65% | +2.2%, 58% | **+33%, 80%** |
| into next session, long (plain nights) | 501 | +14.7% / +13.1% / 73% | +6.9%, 62% | **+29%, 75%** |
| across breaks, long | 322 | −10.7% / −10.8% / 19% | −14.0%, 18% | −8.6%, 35% |
| across breaks, short | 370 | +16.1% / +15.1% / 82% | +14.0%, 82% | +20%, 75% |

Reading it:

- **The same-day short is the stock-specific edge that persists.** Most picks are 09:29 → 14:29 or
  09:29 → 15:29 on stocks that fell for two years (INOXWIND +50% in 2026, SWSOLAR +35%, ZEEL +47%,
  SAMMAANCAP +44%, HCC +45%, RAILTEL +56%, SWANCORP +69%; TDPOWERSYS and NIACL went the other way).
  Hit rates are 35 to 45% because the stock rises on most days and falls hard on the others. This
  is a cash trade (intraday sell, cover before 15:14, about 5 bps round trip).
- **The afternoon-to-morning long is the other.** Picks cluster at 14:29 → 09:29 and 15:29 → 09:29
  on the same small-cap electronics and defence names as before (CUPID +123%, NETWEB +69%, GVT&D
  +57%, ZENTEC +53%, GRAPHITE +54%, LLOYDSENGG +49%, TDPOWERSYS +42%). Futures only, as before:
  delivery STT alone exceeds the edge.
- **Picking the best of 1,250 windows overfits.** The single-window overnight pick of the session
  study kept +41% out of sample for its top 20; the best-of-all-windows pick keeps +29% for its
  top 20 and +15% across all stocks. More choice bought more noise, not more edge. For a trader,
  the lesson is to fix the window by family (the morning short, the afternoon-to-morning long) and
  pick the stock, not to search the grid per stock.
- **Breaks: avoid longs, the shorts are the market.** Short across breaks looks good in 2026 only
  because every stock fell over weekends this year (the universe is +14% too); the picks add
  nothing over it. There is no stock-specific weekend edge either way.

### What it says for the model

Two features for the Fit, both computable at any decision point: the stock's own trailing return in
the morning-short window and in the afternoon-to-morning window (say over the last 60 sessions),
since both are stock-specific and persist across a year. And one target change: the overnight
horizon should run from 14:14 or 14:29, not the close, with the 09:29 exit; the last hour belongs
to the long. The break flag (is the next session after a weekend or holiday) is a cheap feature and
in 2026 it carried a strongly negative return; the model should see it.

## Second round: the fits, validated on 2026 H2 (2026-10-25)

Seven Fits, each fitted twice (through 2025-12-31; and on 2026-01-01 to 2026-06-30 alone), five
models each (ridge baseline; xgboost depth 3 and 4, 300 and 600 trees, learning rate 0.05, min child
weight 200, subsample 0.8, colsample 0.7, lambda 5). All 70 fitted; the two 20-million-row minute
runs went through the streaming path on the CPU (about 13 minutes each). Every FitRun was scored on
2026-07-01 to 2026-10-23 (label `validation 2026 H2`), thresholds 0.1 / 0.2 / 0.5 %, 10 bps cost,
top-5 per decision point. Best xgboost per Fit and split; IC is the within-decision-point Spearman,
×1000.

| Fit | split | best model | IC ×1000 (se) | IC by month Jul / Aug / Sep / Oct | calibration slope | top-5 spread, bps | top-5 net, bps | ridge IC |
|---|---|---|---|---|---|---|---|---|
| 15m | to 2025 | d4 n600 | **70 (0.9)** | 73 / 77 / 69 / 61 | 0.31 | 8.4 | −5.9 | 53 |
| 15m | 2026 H1 | d3 n600 | 62 (0.8) | 69 / 71 / 60 / 52 | 0.13 | 5.6 | −7.2 | 47 |
| 60m | to 2025 | d4 n600 | 40 (1.1) | 34 / 50 / 42 / 39 | 0.16 | 7.8 | −7.9 | 33 |
| 60m | 2026 H1 | d4 n600 | 23 (1.0) | 27 / 26 / 22 / 21 | −0.03 | 6.0 | −8.0 | 17 |
| 375m (session) | to 2025 | d3 n600 | **76 (0.9)** | 76 / 84 / 75 / 64 | 0.41 | 23.7 | −9.3 | 61 |
| 375m | 2026 H1 | d4 n600 | 51 (1.0) | 63 / 66 / 55 / 18 | 0.10 | 25.2 | −10.3 | 36 |
| 1d NSE | to 2025 | d4 n600 | 49 (7.0) | 29 / 59 / 56 / 41 | 0.28 | 25.3 | +3.4 | 27 |
| 1d NSE | 2026 H1 | d4 n300 | 50 (4.7) | 40 / 53 / 58 / 46 | 1.44 | 4.1 | −10.1 | 14 |
| 1d US | to 2025 | d3 n600 | 35 (12) | 56 / 11 / 44 / 91 | −0.13 | 69 | −6.4 | 30 |
| 1d US | 2026 H1 | linear | 11 (13) | 32 / −26 / 20 / 78 | 0.13 | 17 | −38 | 11 |
| 21d NSE | to 2025 | d4 n300 | 61 (5.3) | 51 / 89 / 50 / – | 0.16 | −226 | −561 | −107 |
| 21d NSE | 2026 H1 | linear | 4 (5.0); trees −34 to −99 | | | | | |
| 21d US | to 2025 | d3 n600 | **100 (8.2)** | 102 / 65 / 142 / – | 1.13 | 221 | +203 | 69 |
| 21d US | 2026 H1 | linear | −12 (15); trees −15 to −42 | | | | | |

### Reading

- **The new features raised the 15-minute IC from 0.050 to 0.070**, with the ridge baseline from
  0.038 to 0.053, and the monthly ICs barely move (0.061 to 0.077). The round-one review's
  importances said time-of-day, the stock's recent returns and delivery would carry it; they did.
  Calibration is still poor (slope 0.31: the trees predict four times too large), which is what
  the rescale knob is for before any threshold means anything.
- **The session horizon is the best signal in the data: IC 0.076.** It is also the only minute
  horizon with a thresholded result above costs: predictions over 0.5 % made +21 bps a trade after
  10 bps cost over 60,863 trades, −17 / +15 / +21 / +51 bps by month. The top-5 long is still
  negative (−9 bps) because the daytime drifts down for everyone (the window study's morning
  fall); the spread between the top and bottom quintile is 24 bps. The signal says which stock
  falls least, and a long-only trade pays for the market's drift. A long-short reading of the
  same model, or the model applied to the daytime short, is the next thing to measure.
- **Fitting through 2025 beats fitting on 2026 H1 alone, every time.** 70 vs 62 at 15 minutes, 40
  vs 23 at 60, 76 vs 51 for the session; for the 21-session horizons the H1-only trees flip sign
  (−0.07 to −0.10 on NSE). Six months is not enough history for 99 features; recency does not
  compensate. The question the owner asked is answered: more data wins, and the 2026-only fit
  should not be used.
- **60 minutes is the weak horizon** (0.040, nothing above costs), as the screens predicted: the
  minute reversals have faded and the daily features have not yet taken over.
- **One session on NSE at the daily grain** holds its 0.05 but on 79 sessions and 128k rows the
  standard error is 0.007 and the top-5 net of +3.4 bps is within noise. Two more quarters before
  it is believed.
- **21 sessions is not a result yet.** NSE: a positive IC in July and August and −100 to −560 bps
  per trade, because 2026 H2 fell 15% and every threshold trade is a 21-day long. US: IC 0.10 and
  +203 bps net per top-5 trade looks like the best line in the table, but it is three overlapping
  months of 21-day windows, roughly three independent observations, and the same Fit trained on
  2026 H1 scores −0.015 to −0.04. The standard errors in the table assume independent points and
  understate this badly. Not actionable until a year of non-overlapping windows exists.
- **US daily is weak and noisy**, as the screens said: IC 0.035 with se 0.012, month to month from
  0.01 to 0.09.

### What comes next

1. Rescale by the calibration slope and re-evaluate the 15m and 375m models (the `rescale`
   knob exists), so the thresholds are in return units that mean something.
2. A long-short reading: `net_return_topk` minus the bottom-k, and the session model applied as a
   daytime short on the bottom quintile.
3. The overnight horizon as a Fit target (14:14 or 14:29 to 09:29, both markets) with the window
   study's two stock-specific features and the break flag.
4. Keep fitting through the latest full year; drop the 2026-only split.
