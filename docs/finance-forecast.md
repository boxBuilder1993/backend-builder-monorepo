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
