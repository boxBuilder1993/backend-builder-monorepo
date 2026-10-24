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
2026-10-08 as of writing, 120 million minute bars. At 5-minute spacing that is about 20 million rows.

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
| intraday, bars up to t | returns over the last 1, 5, 15, 30, 60 min and since open; position within today's range; deviation from today's VWAP; realised volatility over the last 30 min and since open; volume in the last 5 and 15 min relative to the same clock time over the past 20 sessions; NIFTY 50's returns over the same spans; the stock's return minus the index's over 15 and 60 min |
| daily, frozen at yesterday's close | yesterday's return; today's gap at open; 5- and 20-day return; 20-day volatility; distance from the trailing 52-week high and low; turnover relative to its 20-day median; deliverable percentage where NSE reports it; relative strength against NIFTY 50 over 20 days |
| clock | minutes since open; minutes to close; day of week; expiry day; day after a holiday |

About 30 features. The final list is decided by the IC screen below, not by this table.

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

1. The feature view and the query with `:horizon_minutes`, `:step_minutes` and `:vol_minutes`, checked for leakage by
   construction: every window ends at or before t, every daily join is on the previous session.
2. The IC screen at 15, 60 and 375 minutes. Its table goes in this document.
3. The streaming loader and the two metrics.
4. The 15-minute Fit registered and run on the GPU; `Infer` on 2026; `Evaluate`. Then 60 minutes and
   one session with the same query.
5. The results, per horizon, appended here: IC with standard error, calibration slope, net return at
   each threshold, and the monthly series. Decision after that: more features, another horizon, or
   on to the consumer.

## Settled and open

Settled: return not price, scaled by trailing volatility; Δ and the volatility window as parameters; features computable at any decision point from data
before it; within-decision-point normalisation; the IC screen before fitting; evaluation on 2026 by
IC, calibration and net return after costs, per month; baseline linear, then xgboost.

Open: step spacing (5 minutes, or every minute with subsampling); whether to include the indices as
rows or only as features (default: features only, they are not tradable in cash); whether rows in the
first and last 15 minutes of the session are kept (default: kept, with the clock features carrying
the information).

## Later, not here

The consumer: a trader of signals, a decision rule and execution; the hindsight-optimal trades as
the ceiling it is scored against; the product `(collection, principal, start, end) -> expected
return`. Discussed and parked until the forecast has a measured IC.
