# Holding portfolios: forecasts for weeks to a quarter, and a backtest that trades them

Status: **design for review**, 2026-10-25. Nothing here is built yet.
Builds on [finance-ml.md](finance-ml.md) (Fit, FitRun, Model, InferRun, Evaluation),
[finance-forecast.md](finance-forecast.md) (the daily rows and their features) and the results in
[finance-research-2026-10.md](finance-research-2026-10.md).

## 1. Why

The October work found one real signal, the session model's daily ranking, and showed that Indian
retail costs consume it: an intraday round trip costs about 10.6 bps a stock at Zerodha below
₹66,667 an order, a delivery round trip about 22 bps plus a ₹18 depository charge per sale. Every
strategy that traded every day lost to those costs, and every stock list that looked good had been
chosen with hindsight.

The way out is to trade less often. A stock held for a month pays its delivery cost once against
moves of several percent. This document designs the work to answer one question honestly:

> Does a portfolio of stocks chosen by a forecast of the next week, month or quarter, rebalanced
> at that pace and charged Zerodha's real delivery costs, beat buying the index, out of sample,
> from 2023 to 2026?

## 2. What is decided here

| Decision | Choice | Why |
|---|---|---|
| Instrument | NSE cash equity, delivery (CNC), long only | Delivery has zero brokerage and holds overnight; shorts cannot be held past the day in cash |
| Horizons | 5, 21 and 63 sessions (a week, a month, a quarter) | The owner's range; each pays its cost once per hold |
| Signal | A daily-grain Fit per horizon, predicting the volatility-scaled return to the close of the h-th session | The existing daily rows, extended (§4) |
| Training | Walk-forward: retrain each quarter on all history up to an embargo, predict only the next quarter | One test of 3.75 years instead of one test of 4 months; covers rising and falling markets |
| Portfolio rule | Hold the top N by forecast; keep a holding while it ranks within the top B; sell when it falls below B | Separates entry from exit so costs are paid only when a stock clearly loses its rank |
| Costs | Zerodha's published 2026 schedule, per order, plus slippage | The schedule the owner pasted on 2026-10-25 (§6) |
| Benchmarks | Nifty 50, Nifty 500, equal-weight universe, and two model-free strategies (12-1 momentum, low volatility) | The model has to beat simple ideas, not only the index |
| Selection | Settings chosen on 2023-01 to 2024-12, confirmed untouched on 2025-01 to 2026-10 | 72 configurations would otherwise find a lucky one |
| Execution | Everything as reactions with entities, deferred children, tests | The standing rule; no result comes from a script |

## 3. The pieces

```
RegisterFit (×3: forecast 5d/21d/63d nse r3)
      │
RunWalkForward ──creates──► WalkForward + WalkForwardWindow ×16
      │                         │
      └─defers─► RunWindow ×16  (one per window, one at a time on the worker)
                    ├─ RunFit {inline: true} ─► FitModel (in the same run) ─► UploadModel
                    └─ Infer on the window's test range
                         │
      OnChange(WalkForwardWindow.status) ─► CloseWalkForward
                    └─ when all windows are done: one predictions file, per-window IC, status done

RunBacktest  (input: a WalkForward, or a model-free baseline, and the portfolio rules)
      └─ BacktestRun: daily equity, trades, holdings files; summary in the row
RunBacktestSweep ──defers─► RunBacktest ×N  (a grid; one BacktestSweep row collects them)
      OnChange(BacktestRun.status) ─► CloseBacktestSweep ─► the comparison table, split by period
```

Long orchestrations defer their children and close on a status change, the way `RunFit`,
`FitModel` and `CloseFitRun` already work. No parent holds one transaction for an hour; that is
what lost the first FitSweep on 2026-10-24 (issue #69).

## 4. The forecasts

### Fits

Three new Fits, registered from JSON under `apps/finance/fits/round3/`, on the existing
[`daily_sql("nse")`](../apps/finance/src/finance/ml/forecast.py) with these changes:

| Change | Detail | Why |
|---|---|---|
| Adjusted prices | Returns, features and targets from `equity_adj` (`adj_open`, `adj_close`, …) instead of `equity` | A split or bonus inside a 63-session hold reads as a −50 % or −90 % return on unadjusted prices. Today's daily rows read `equity` |
| Horizons | `horizon_sessions` 5, 21, 63 | One Fit each |
| Slow features | `ret_120d`, `ret_250d`, `mom_12_1` (250-session return less the last 21), `vol_120d`, `max_dd_60d` (worst fall from a 60-session high), `deliv_pct_20d` (20-session mean delivery %), `turn_trend` (20-session turnover over 120-session), `s_ret_60d` (sector index, 60 sessions), `rs_120` (stock minus Nifty 50 over 120 sessions) | What the literature and our screens say matters at weeks to months: momentum, its reversal at a month, volatility, accumulation |
| Warm-up | Rows start 2022-09-01 so the 250-session features are complete; queries read from 2021-09-01 | |
| Liquidity | 20-session median turnover ≥ ₹1 crore (unchanged) | Enough for ₹1 lakh positions without moving the price |
| Universe | `series = 'EQ'` only | BE (trade-for-trade) stocks cannot be bought and sold freely |

Each Fit keeps the screen rule of the October round: the new features go through `Screen` on
2022-09 to 2024-12 first, and the Fit's feature list is what `features_from_screen` keeps.

Search: one spec per Fit, chosen before the walk-forward and fixed for it: xgboost depth 3,
600 trees, learning rate 0.05, min child weight 200, subsample 0.8, colsample 0.7, lambda 5, and
the ridge as the baseline. Two specs × 16 windows × 3 horizons = 96 fits; daily rows are about
1.4 million, so a fit takes under a minute.

### Walk-forward

New entities in `finance/ml/entities.py`:

```python
class WalkForward(Entity):
    """One walk-forward evaluation of a Fit and one model spec."""
    fit_id: str
    algorithm: str
    params: dict[str, Any]          # the model spec
    query_params: dict[str, Any]    # the Fit's query parameters, minus start/end
    train_start: date               # first row date used in training
    first_test: date                # start of the first test window
    last_test_end: date
    step_months: int = 3            # test window length
    embargo_sessions: int           # = horizon_sessions: no training target ends inside a test window
    label: str
    status: str = "running"         # running | done | failed
    windows: int = 0
    predictions_path: str | None = None   # parquet/walkforward/<id>.parquet
    summary: dict[str, Any] | None = None # IC per window, overall IC, rows
    error: str | None = None

class WalkForwardWindow(Entity):
    walk_forward_id: str
    index: int
    train_end: date                 # last row date with a target; = test_start minus embargo sessions
    test_start: date
    test_end: date
    status: str = "queued"          # queued | running | done | failed
    fit_run_id: str | None = None
    model_id: str | None = None
    infer_run_id: str | None = None
    error: str | None = None
```

Windows for each horizon: test quarters 2023-Q1 … 2026-Q4 (the last one ends at the data's end),
16 in all. Training is expanding: always from 2022-09-01 to `train_end`. The embargo makes the last
training row's target end before the first test day: for 63 sessions, a quarter's gap.

Reactions in `finance/ml/reactions.py`:

| Reaction | Input | What it does |
|---|---|---|
| `RunWalkForward` | `{fit_id, algorithm, params, first_test, last_test_end?, step_months?, label}` | Creates the WalkForward and its windows (status `queued`), defers one `RunWindow` per window with a budget of 30 minutes. Returns in seconds |
| `RunWindow` | `{window_id}` | `RunFit` with `{"inline": true, "params": {"end": train_end}, "specs": [the one spec]}`, then `Infer` with `{"start": test_start, "end": test_end}` and label `wf <walk_forward_id> w<index>`; records the ids, status `done` or `failed` with the error |
| `CloseWalkForward` | on `WalkForwardWindow.status` | When no window is `queued` or `running`: concatenates the windows' inference tables (key, `pred_y`, `y`, `r`, `scale`, `entry`, `exit_`, window index) into `parquet/walkforward/<id>.parquet`, publishes it, computes IC per window and overall, sets the WalkForward `done` (or `failed` if any window failed) |

Two small changes to existing reactions:

- `RunFit` gains `inline` (default false): `FitModel` is triggered immediately inside the run
  instead of deferred, so `RunWindow` can score the model it just made. It also gains `specs` to
  fit a subset of the Fit's search.
- The daily rows drop every row whose exit lies beyond the data (`where entry is not null and exit_
  is not null`), so the last h sessions of data get no forecast. A query parameter
  `:need_target` (default true) keeps those rows with a null target when false; windows' `Infer`
  sets it false, so the backtest has a forecast for every day it trades, including today. The
  daily rows already carry `entry`, `exit_`, `r` and `scale`, which the backtest uses to check its
  own fills.

### Evaluate per window

Each window's InferRun is evaluated with the existing metrics (`ic`, `ic_se`, `calibration_slope`,
`spread_topk` with `group = ts`), and `CloseWalkForward` stores the per-window IC table in the
WalkForward's `summary`. This answers "is there a signal at this horizon, and is it stable across 16
quarters" before any money is simulated.

## 5. The backtest

### Entities

```python
class BacktestRun(Entity):
    """One simulated account."""
    source: dict[str, Any]          # {"walk_forward_id": ...} | {"baseline": "momentum_12_1" | "low_vol" | "equal_weight" | "index", "index": "Nifty 50"}
    rules: dict[str, Any]           # see the table below
    costs: str                      # cost schedule name, e.g. "zerodha_2026_delivery"
    start: date
    end: date
    capital: float
    status: str = "running"         # running | done | failed
    equity_path: str | None = None  # parquet/backtests/<id>_equity.parquet: date, value, cash, invested, charges_to_date
    trades_path: str | None = None  # parquet/backtests/<id>_trades.parquet: date, symbol, side, shares, price, value, charges
    summary: dict[str, Any] | None = None
    sweep_id: str | None = None
    error: str | None = None

class BacktestSweep(Entity):
    """A grid of BacktestRuns judged together."""
    label: str
    selection_period: dict[str, str]    # {"start": "2023-01-01", "end": "2024-12-31"}
    confirmation_period: dict[str, str] # {"start": "2025-01-01", "end": "2026-10-23"}
    runs: int = 0
    status: str = "running"
    table: dict[str, Any] | None = None # the comparison, per run and per period
```

### Rules

| Rule | Values | Meaning |
|---|---|---|
| `rebalance` | `weekly` (every 5 sessions), `monthly` (21), `quarterly` (63) | Days on which the portfolio may change |
| `top_n` | 10, 20 | Stocks bought when the portfolio fills |
| `exit` | `band` (default), `fixed`, `score`, `stop` | How a holding is sold, below |
| `band` | B = 2 × top_n by default | `band`: keep while rank ≤ B, sell when rank > B |
| `stop_loss`, `take_profit` | e.g. 0.08, 0.15 | `stop`: band rule plus a sell at the next open after a close beyond either |
| `weighting` | `equal` | Each new position gets cash ÷ open slots; holdings are not trimmed back to equal weight (trimming costs a sale) |
| `slippage_bps` | 0, 5, 10 per trade | Against the trader on every fill |
| `max_position_turnover` | 2 % of the stock's 20-session median turnover | A position larger than this is cut to it; a stock too thin for even one share is skipped |

`fixed` sells every holding at each rebalance and buys the new top N (the honest "re-pick" rule).
`score` sells when the stock's forecast falls below zero after rescaling by the walk-forward's
calibration slope.

### Mechanics

- **Decision and fills.** A rebalance is decided after day t's close from the forecasts made at t,
  which use only data to t's close. Orders fill at day t+1's open: sells first, then buys with the
  cash. Prices are unadjusted (what one actually pays); a split or bonus between the buy and the
  sale multiplies the shares held by the factor from `adj_factors`, so the value is unchanged on the
  ex-date. Dividends are not credited; the doc's results will state this and the benchmark uses the
  price index for the same reason.
- **Whole shares.** Shares = floor(allocation ÷ price × (1 + slippage)). Leftover cash stays as cash
  and earns nothing.
- **Missing prices.** A held stock with no price on a rebalance day is held until it trades; a stock
  delisted for good is sold at its last close with a flag in the trades file.
- **Charges.** Every order is charged by the cost module (§6) at its own value; the depository
  charge is once per stock per sale day.
- **Survivorship.** The universe on each day is the stocks with rows that day; delisted stocks stay
  in the history. No list from today is applied to the past.

### Baselines (no model)

| Baseline | Score used for ranking | Purpose |
|---|---|---|
| `index` | Nifty 50 or Nifty 500 price index, bought once | What doing nothing earns |
| `equal_weight` | Every liquid stock, equal weight, rebalanced at the same pace, charged | The market as the backtest sees it |
| `momentum_12_1` | `mom_12_1` | The best-known simple stock-picking rule |
| `low_vol` | −`vol_120d` | The other well-known one |

Baselines use the same rules and costs as the model portfolios, so the comparison is like for like.

### Reactions

| Reaction | Input | What it does |
|---|---|---|
| `RunBacktest` | `{source, rules, costs, start, end, capital, sweep_id?}` | Loads the scores (the WalkForward's predictions file, or the baseline's SQL over the daily rows) and prices through DuckDB, runs the account in `finance/ml/backtest.py` (a pure function of scores, prices and rules, no runtime), writes the equity and trades files, publishes them, fills `summary` |
| `RunBacktestSweep` | `{label, sources, grid, costs, capital, periods}` | Creates a BacktestSweep and defers one `RunBacktest` per (source × grid point × period) |
| `CloseBacktestSweep` | on `BacktestRun.status` | When the sweep's runs are all done: the comparison table into the sweep row, and as markdown in its vars |

### Summary of a run

`summary` holds, for the run's period and per calendar year: end value, CAGR, annualised volatility,
Sharpe (risk-free rate 6.5 %), max drawdown and its dates, return of the matching benchmark, excess
return and its t-statistic over months, months beating the benchmark, turnover (₹ traded ÷ average
equity, per year), charges paid (₹ and % of average equity per year), average holding period,
number of trades, and the share of positions cut by the turnover limit.

## 6. Costs

A new module `finance/ml/costs.py` with the schedule the owner pasted from Zerodha's charges page
on 2026-10-25, one function per segment returning the ₹ charges for one executed order:

| Item | Delivery (CNC) | Intraday (MIS) | Futures |
|---|---|---|---|
| Brokerage | 0 | min(₹20, 0.03 %) | min(₹20, 0.03 %) |
| STT | 0.1 % buy and sell | 0.025 % sell | 0.05 % sell |
| Exchange (NSE) | 0.00307 % | 0.00307 % | 0.00183 % |
| SEBI | ₹10 / crore | ₹10 / crore | ₹10 / crore |
| Stamp | 0.015 % buy | 0.003 % buy | 0.002 % buy |
| GST | 18 % on brokerage + exchange + SEBI | same | same |
| Depository (DP) | ₹15.34 + GST per stock per sale day | – | – |

The DP charge is not on the pasted page; it is Zerodha's published CDSL charge and is flagged as
an assumption to confirm. Tests check the module against two worked examples from Zerodha's
brokerage calculator, entered by hand into the test.

## 7. The experiment

1. Register the three Fits; run `Screen` for the new features; finalise the feature lists.
2. Six walk-forwards: 3 horizons × {xgboost, ridge}, label `wf r3`.
3. Read the per-window IC tables. A horizon whose IC is not positive in most windows stops here.
4. `RunBacktestSweep` over: sources {the surviving walk-forwards, 4 baselines} × rebalance {weekly,
   monthly, quarterly} × top_n {10, 20} × exit {band, fixed} × slippage {5 bps}, capital ₹10 lakh
   and ₹1 lakh, in the selection period 2023-01-01 to 2024-12-31.
5. Choose the configuration with the best risk-adjusted excess return over the equal-weight
   baseline in the selection period, and write the choice into the results doc before step 6.
6. Run that one configuration, unchanged, on 2025-01-01 to 2026-10-23, with slippage 0, 5 and 10.
7. Write the results: the selection table, the confirmation result, the stop/score exit variants as
   a robustness check, and the per-window ICs.

The pass bar, set now: in the confirmation period, after charges and 5 bps slippage, the chosen
portfolio beats both Nifty 500 and the equal-weight baseline, with a positive excess-return
t-statistic, at ₹1 lakh capital. Anything less is reported as a failure.

## 8. Tests

| Test | Checks |
|---|---|
| `test_walk_forward_windows` | Window boundaries, embargo by sessions not days, no training row's exit date on or after its window's test start |
| `test_walk_forward_end_to_end` | On the synthetic mirror: RunWalkForward → windows done → CloseWalkForward writes one predictions file with every window's rows |
| `test_costs_schedule` | Each segment against hand-entered calculator examples; DP charge once per stock per sale day |
| `test_backtest_known_outcome` | Synthetic prices where the top-ranked stock rises 1 % a day: exact end value after charges |
| `test_backtest_band_rule` | A stock moving from rank 3 to rank 15 with N = 10, B = 20 is held; to rank 25 is sold |
| `test_backtest_split` | A 1:2 split mid-hold doubles shares and leaves value unchanged |
| `test_backtest_whole_shares_and_cash` | Allocation below one share skips the stock; leftover cash carries |
| `test_backtest_fills_next_open` | Decisions at t fill at t+1's open, never at t's close |
| `test_baselines_same_rules` | A baseline and a model source with identical scores give identical runs |
| `test_daily_rows_adjusted` | The daily rows' returns are continuous across a split |

## 9. What changes where

| File | Change |
|---|---|
| `apps/finance/src/finance/ml/forecast.py` | Daily rows on adjusted prices; the slow features; `daily_fit_definition` for 63 sessions |
| `apps/finance/src/finance/ml/entities.py` | WalkForward, WalkForwardWindow, BacktestRun, BacktestSweep |
| `apps/finance/src/finance/ml/reactions.py` | RunWalkForward, RunWindow, CloseWalkForward, RunBacktest, RunBacktestSweep, CloseBacktestSweep; `inline` and `specs` on RunFit |
| `apps/finance/src/finance/ml/backtest.py` | New: the account simulation as a pure function |
| `apps/finance/src/finance/ml/costs.py` | New: the cost schedules |
| `apps/finance/src/finance/triggers.py` | OnChange for WalkForwardWindow.status and BacktestRun.status |
| `apps/finance/migrations/` | The four new tables |
| `apps/finance/fits/round3/` | Three Fit JSONs |
| `apps/finance/tests/` | §8 |
| `docs/finance-ml.md` | The walk-forward and backtest sections |

Order of PRs: costs and backtest with baselines (testable without any model), then the walk-forward,
then the Fits and the experiment, then the results document.

## 10. Open questions for review

1. **Capital levels.** ₹1 lakh and ₹10 lakh proposed. With N = 20, ₹1 lakh is ₹5,000 a stock and
   the DP charge is 0.36 % per sale; is ₹1 lakh the size that matters, or a larger one?
2. **N and the band.** 10 and 20 stocks, band at 2N. Narrower bands trade more; wider ones hold
   losers longer.
3. **Dividends.** Ignored on both sides for now. NSE corporate actions carry dividends; crediting
   them is a follow-up that raises every portfolio's return by about 1 % a year.
4. **Short side.** Out of scope: delivery cannot short overnight, and stock futures cover about 200
   names. A long-only portfolio against the index is the test.
5. **Model specs.** One xgboost spec fixed in advance, plus the ridge. A small search inside each
   window is possible but multiplies fits by four.
6. **The US.** The same machinery runs on US daily rows with a US cost schedule; proposed as a
   second step, not part of this build.
