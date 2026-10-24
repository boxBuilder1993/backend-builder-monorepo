# The trader: what we are building and in what order

Status: design, 2026-10-24. Agreed in discussion; this is the written form to argue with before code.
Builds on [finance-ml.md](finance-ml.md) (Fit, FitRun, Model, InferRun, Evaluation) and the data in
[finance-app.md](finance-app.md). Supersedes the two problems in [finance-modelling.md](finance-modelling.md)
as the first thing we fit.

## The product

A function with four inputs and one output:

    (stock collection, principal, start date, end date) -> expected return

Behind it is a **trader**: a procedure that, at every decision point inside the date range, looks at
each stock in the collection and decides to buy, sell or hold, once per decision point. The grain of
the decision points is a parameter: one a day to begin with, later one every 15 minutes, in principle
anything from a minute to a quarter.

The same four inputs mean two different things:

- **End date in the past.** The output is a backtest: run the trader over the range on the data we
  hold and report what it made, next to what the hindsight-optimal trades, cash and the index made
  over the same range. Exact, not expected. This is also how the trader is built and judged.
- **End date in the future.** The output is a distribution of the trader's own outcomes, measured by
  running it over every past window of the same length on similar collections. Median, and the bad
  tail. Measured, not predicted; no extra ML.

A past range is only called out-of-sample when it starts after the training cutoff of every model the
trader uses. The product shows the cutoff next to the number.

## What the trader is made of

The trader is **not one ML model**. It is three layers, each with its own scoreboard, so a bad quarter
can be traced to the layer that caused it.

1. **Signals.** A signal is a function from (instrument, decision point) to a number, computable from
   data strictly before that point. Some signals are ML models' predictions; some are formulas such as
   20-day momentum. The layers above cannot tell them apart.
2. **Decision rule.** Named code with parameters. Takes the signals for every instrument in the
   collection plus the current book, returns orders. Hand-written to begin with.
3. **Execution.** Fills orders at the next decision point's price with the cost model, updates the book,
   records trades. The simulator from the old repo, brought into the finance app.

Several traders may exist, differing in grain, signals, model or rule. Each is tested the same way.

## The first ML problem

**Given a stock now, what is its expected price Δ minutes from now?**

Stated as a Fit:

- **Row.** One per (symbol, decision minute t), for decision minutes every 5 minutes inside the
  session, over the Kite universe (`equity_1m`: 463 symbols including the indices, 756 sessions,
  2023-10-03 to 2026-10-08 as of writing, 120 million minute bars).
- **Target.** `log(close at t+Δ) - log(open of the bar after t)`. The return, not the price: prices
  differ by orders of magnitude across stocks and drift over years; the expected price is
  `price now × exp(prediction)`, computed at read time. The entry is the next bar's open because that
  is what a decision at t actually gets; the minute's last trade can be stale.
- **Δ is a query parameter**, not model code. One Fit per horizon: 15, 30, 60 minutes, and one session.
  Same features, separate models, so each horizon's errors can be read on their own.
- **Features**, all ratios or returns, never rupee levels, all z-scored or ranked within the decision
  point so no stock dominates; computable at every decision point of every grain:

  | layer | features |
  |---|---|
  | intraday, bars up to t | returns over the last 1, 5, 15, 30, 60 min and since open; position within today's range; deviation from today's VWAP; realised volatility over the last 30 min and since open; volume in the last 5 and 15 min relative to the same clock time over the past 20 sessions; the index's returns over the same spans; the stock's return minus the index's |
  | daily, frozen at yesterday's close | yesterday's return; today's gap at open; 5- and 20-day return; 20-day volatility; distance from the trailing 52-week high and low; turnover relative to its 20-day median; deliverable percentage where NSE reports it; relative strength against NIFTY 50 over 20 days |
  | clock | minutes since open; minutes to close; day of week; expiry day; day after a holiday |

- **Search.** `sklearn.linear` as the baseline that must be beaten, then `xgboost` on the GPU over
  max_depth {4, 6}, learning_rate {0.05, 0.1}, n_estimators {500, 1500}, min_child_weight {10, 100}.
- **Split.** Fit through 2025-12-31. `Infer` on 2026-01-01 to the end of the data with the label
  `validation 2026`. A gap of one session between fit and validation so no target crosses the cut.
- **Evaluation** on the validation table, in this order of importance:
  1. Information coefficient: rank correlation between prediction and realised return within each
     decision point, averaged over decision points, with its standard error. The number that says the
     model orders stocks better than chance. New metric to add to the registry.
  2. Calibration: realised mean return per decile of prediction. Must be monotone and near the
     diagonal; a linear rescale is applied if the slope is off, and recorded.
  3. Net return per trade after the cost model at prediction thresholds of 0.1, 0.2 and 0.5 percent,
     with the trade count at each. Shows where, if anywhere, the forecast clears the cost.
  4. The same three per month, so one good quarter cannot carry the year.

**What we expect.** Fifteen-minute returns on liquid stocks are mostly noise. A good model has an IC of
0.02 to 0.05 and explains under one percent of variance. Intraday round-trip costs on NSE, STT on the
sell side, exchange charges and the spread, run around 0.05 to 0.1 percent, the size of a typical
15-minute move. So the forecast's job in the trader is to pick the few moments and names where the
predicted move clears cost by a margin, not to call every interval. If the 15-minute IC is flat we read
the 60-minute and one-session Fits before changing anything else; signal usually grows with horizon.

## What the forecast feeds

For each instrument at each decision point the trader needs:

| quantity | source |
|---|---|
| expected return μ | the Fit above |
| spread σ | the model's residual standard deviation on the validation table, per prediction decile; a dedicated Fit on absolute return later if this is too coarse |
| switching cost | the cost model, a function of trade value: brokerage, STT, exchange charges, GST, stamp duty, plus slippage |
| book state | current holding, unrealised return, periods held, cash |

The first decision rule, for one position at a time: pick the instrument with the highest
`μ - σ²/2 - switching cost`, cash included as the candidate with zero. Maximising ending capital is
maximising the sum of log returns, and that expression is the expected log growth; the rule needs
calibrated μ and σ, which is why calibration is the second evaluation. Fractional sizing by `μ/σ²`
scaled by a safety factor comes when the principal allows more than one position.

## The oracle

For any (collection, principal, start, end, grain) the hindsight-optimal trades are computable:
without costs, hold the instrument with the highest return to the next decision point; with switching
costs, a dynamic programme over (decision point, holding); with integer shares at a small principal, a
bounded knapsack per step. It yields the ceiling for the range, the best buy and sell points, and a
value for being in each instrument at each point.

Uses, in order: the ceiling every trader is scored against ("fraction of oracle captured" is the single
number); the per-day regret series that says *where* a trader lost; and, later, advantage labels
(value of a candidate minus value of the best) for fitting a trader that imitates the oracle directly,
with book state among its features. That third use is not in the first version.

## Entities to add

Minimal, in the style of the ML ones. Names provisional.

- `Trader`: grain, the signals (model ids or named formulas), the decision rule's name and params, the
  cost model. Immutable once an episode has used it.
- `Episode`: trader id (the oracle is one), collection, principal, start, end, grain; status for
  deferral; results: ending capital, equity curve, trades, drawdown, Sharpe, and the oracle, cash and
  index figures for the same inputs.
- `Trade`: one fill, keyed to the episode.

One reaction, `RunEpisode`, deferred like `FitModel`: triggers `Infer` for any model signal whose
predictions do not cover the range, then walks the decision points. The product's past-range answer is
one Episode; its future-range answer is a summary over Episodes.

## Changes to what exists

- `LoadRows` materialises the query's frame; at 5-minute spacing the 15-minute Fit is about 20 million
  rows × 30 features, which fits the GTX 1080's 8 GB as float32 but not as a Python-side copy. The
  loader streams to a Parquet frame on disk and xgboost reads it in batches.
- `ic` and `calibration` metrics in the registry.
- The DuckDB views gain a minute-level feature view over `equity_1m` with the index joined by minute.
- The Kite leg is behind its token (data ends 2026-10-08); `make kite-login` brings it current. The
  fit does not wait for it.

## Order of work

1. The feature query with `:horizon_minutes` and `:step_minutes` as parameters, and the IC screen
   over each feature on its own, in DuckDB, before any fit. A feature with no IC is dropped here.
2. The 15-minute Fit, registered, run on the GPU, inferred on 2026, evaluated as above. Then 60
   minutes and one session with the same query.
3. The cost model and the episode runner with the oracle, cash and index baselines, daily grain
   first, momentum as the first signal so the product runs end to end before a model is wired in.
4. Wire the best model in as a signal with the decision rule above; score against the oracle on 2026
   windows; compare with momentum.
5. Extend the grain to 15 minutes in the runner, which is only a parameter if steps 1 to 4 kept the
   row contract.

## What is settled and what is not

Settled: the product's four inputs; the three-layer trader; return not price as the target; Δ as a
parameter; features computable at any decision point; evaluation by IC, calibration and net return
after costs on a later range; the oracle as the ceiling; one position at a time first.

Open: whether a position may be held across sessions at intraday grain (default: no, square off by
close); how collections are specified (list of symbols, or a rule such as a turnover floor); the
principal values to report at (default: 5,000 and 5,00,000, since integer shares dominate the small
one); whether to pool horizons into one model once the separate ones are understood.
