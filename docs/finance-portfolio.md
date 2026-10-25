# Holding portfolios for a small account: forecasts for days to a quarter, and a backtest that trades them

Status: **reviewed and agreed**, 2026-10-25, revision 3. Owner's decisions: no dividends, no shorting,
search the space as widely as possible, study capital first, target ₹10,000, capitals from ₹10,000
up, about a day of overnight compute is fine, the affordability tilt is accepted, the US later.
Builds on [finance-ml.md](finance-ml.md) (Fit, FitRun, Model, InferRun, Evaluation),
[finance-forecast.md](finance-forecast.md) (the daily rows and their features) and the results in
[finance-research-2026-10.md](finance-research-2026-10.md).

## 1. Why

The October work found one real signal, the session model's daily ranking, and showed that Indian
retail costs consume it: an intraday round trip costs about 10.6 bps a stock at Zerodha below
₹66,667 an order, a delivery round trip about 22 bps plus a depository charge of about ₹18 per stock
sold. Every strategy that traded every day lost to those costs, and every stock list that looked good
had been chosen with hindsight.

The way out is to trade less often: a stock held for a month pays its delivery cost once against moves
of several percent. But the account is small. **The target is at most ₹10,000.** At that size the
fixed depository charge dominates: ₹18 on a ₹2,000 position is 0.9 % per sale, four times the
percentage charges. So the question has two halves, studied in this order:

1. **Capital:** for an account of a given size, how many stocks, how often traded, before charges eat
   the return? (§4)
2. **Signal:** within what that allows, does a forecast of the next days, weeks, months or quarter pick
   a portfolio that beats simply buying the index, out of sample, 2023 to 2026? (§5 to §7)

## 2. What is decided here

| Decision | Choice | Why |
|---|---|---|
| Instrument | NSE cash equity, delivery (CNC), long only. **No shorting, no dividends** | Owner's decision. Delivery has zero brokerage and holds overnight; dividends are ignored on every side, benchmarks included, so comparisons stay like for like |
| Capital | **₹10,000** is the target. The capital study covers ₹10,000 to ₹10 lakh to show where the costs stop mattering | Owner's decision |
| Horizons | 5, 10, 21, 42, 63 and 126 sessions (a week to half a year) | Wide, per the owner; each pays its cost once per hold |
| Signal | Daily-grain Fits, one per horizon and target, on split-adjusted rows | §5 |
| Training | Walk-forward: retrain each quarter on all history up to an embargo, predict only the next quarter | 16 quarters of out-of-sample forecasts, rising and falling markets |
| Model search | As wide as the box allows: 3 algorithms, ~50 settings, 2 feature sets, 2 targets, 6 horizons, in two stages | Owner's decision; nested so the search cannot leak into the test |
| Portfolio search | Every combination of stocks held, rebalance pace, hold band and exit rule that a ₹10,000 account can run | §6 |
| Costs | Zerodha's published 2026 schedule per order, depository charge per stock sold, slippage | §8 |
| Benchmarks | NIFTYBEES bought once with ₹10,000; Nifty 50 and 500; equal-weight portfolios of affordable stocks; 12-1 momentum; low volatility | The model must beat what a ₹10,000 investor could do instead |
| Selection | Everything chosen on 2023-01 to 2024-12; one choice confirmed untouched on 2025-01 to 2026-10 | Thousands of configurations will find lucky ones (§7) |
| Execution | Everything as entities and reactions, long jobs as deferred children closed by OnChange, tests | The standing rule; no result comes from a script |

## 3. The pieces

```
Phase 1  capital study (no model)
  RunBacktestGrid {source: baselines, capitals, grid} ──► BacktestGrid rows ──► cost-drag tables

Phase 2  forecasts
  RegisterFit ×12 (6 horizons × 2 targets)
  RunModelSearch ──creates──► ModelSearch, and one WalkForward per (Fit × spec × feature set)
        └─defers─► RunWindow per window (one at a time on the worker)
                      ├─ RunFit {inline: true, specs: [one]} ─► FitModel ─► UploadModel
                      └─ Infer on the window's test range (rows without a target kept)
        OnChange(WalkForwardWindow.status) ─► CloseWalkForward ─► predictions file, IC per window
        OnChange(WalkForward.status)       ─► CloseModelSearch ─► leaderboard, stage-2 shortlist

Phase 3  portfolios
  RunBacktestGrid {source: each shortlisted WalkForward + baselines, capital ₹10,000, grid}
        └─ one BacktestGrid row per source: every rule combination evaluated, results file
  SelectPortfolio ─► the chosen configuration (selection period only) ─► BacktestRun (confirmation)
```

Long orchestrations defer their children and close on a status change, the way `RunFit`, `FitModel`
and `CloseFitRun` already work. No parent holds one transaction for hours (issue #69).

## 4. Phase 1: the capital study

No model. It answers what a small account can afford before any forecast is involved, using the
model-free baselines of §6 under the full portfolio grid.

**Grid:** capital {₹10,000, ₹25,000, ₹50,000, ₹1 lakh, ₹10 lakh} × stocks held N
{1, 2, 3, 4, 5, 6, 8, 10, 15, 20} × rebalance every {5, 10, 21, 42, 63, 126} sessions × exit {band,
fixed} × sources {equal-weight affordable, 12-1 momentum, low volatility}, 2023-01 to 2026-10.

**Output, per capital (a `BacktestGrid` table and a markdown summary in the run's vars):**

| Measure | Meaning |
|---|---|
| Charges per year, % of capital | What the account pays to run the rule |
| Break-even gross return | The yearly return before charges needed just to cover them |
| Affordable universe | Share of liquid stocks whose single share costs no more than capital ÷ N (at ₹10,000 with N = 5, a stock must cost under ₹2,000) |
| Cash drag | Average share of capital left uninvested because of whole shares |
| Realised net return of each baseline | What the rule actually earned after charges |

This phase fixes, before any model exists, the region of N and rebalance pace a ₹10,000 account can
live in. It is also where the plain answer may already sit: if no baseline at ₹10,000 beats buying
NIFTYBEES once, the model has to carry the whole difference.

## 5. Phase 2: the forecasts

### Fits

Twelve Fits under `apps/finance/fits/round3/`: horizons {5, 10, 21, 42, 63, 126} × targets
{volatility-scaled return, raw return}, on the existing
[`daily_sql("nse")`](../apps/finance/src/finance/ml/forecast.py) with these changes:

| Change | Detail | Why |
|---|---|---|
| Adjusted prices | Returns, features and targets from `equity_adj` (`adj_open`, `adj_close`, …) instead of `equity` | A split or bonus inside a long hold reads as a −50 % or −90 % return on unadjusted prices. Today's daily rows read `equity` |
| Raw-return target | `y = ln(exit / entry)` as a second target | The scaled target ranks well on risk-adjusted moves; a portfolio earns raw returns. Both are searched |
| Slow features | `ret_120d`, `ret_250d`, `mom_12_1` (250-session return less the last 21), `vol_120d`, `max_dd_60d`, `deliv_pct_20d`, `turn_trend` (20-session turnover over 120-session), `s_ret_60d`, `rs_120`, `px_level` (log price, for affordability effects) | What matters at weeks to months: momentum and its one-month reversal, volatility, accumulation |
| Rows without a target | Query parameter `:need_target` (default true). False keeps the last h sessions with a null target, so `Infer` forecasts every day including today | Today's rows drop any row whose exit lies beyond the data |
| Warm-up | Rows from 2022-09-01 so the 250-session features are complete; queries read from 2021-09-01 | |
| Liquidity | 20-session median turnover ≥ ₹1 crore | A ₹10,000 order never moves these |
| Universe | `series = 'EQ'` only | BE (trade-for-trade) stocks cannot be traded freely |

**Feature sets**, both searched: `screened` (what `Screen` and `features_from_screen` keep on
2022-09 to 2024-12) and `all` (every candidate).

### The model search space

| Algorithm | Settings | Count |
|---|---|---|
| Ridge | alpha {0.1, 1, 10, 100} | 4 |
| xgboost | max_depth {2, 3, 4, 6} × n_estimators {300, 800} × learning_rate {0.03, 0.1} × min_child_weight {50, 500}; subsample 0.8, colsample 0.7, lambda 5 | 32 |
| lightgbm | num_leaves {15, 63} × n_estimators {300, 800} × learning_rate {0.03, 0.1} × min_child_samples {100, 1000} | 16 |
| **Total** | | **52 specs** |

52 specs × 2 feature sets × 12 Fits = 1,248 candidate models, each walked forward.

### The search, in two stages, inside the selection period only

Fitting all 1,248 on 16 quarterly windows is about 20,000 fits. Instead:

| Stage | Windows | Models | Fits | Picks by |
|---|---|---|---|---|
| 1. Coarse | 2 yearly windows: test 2023, test 2024 | all 1,248 | 2,496 | Mean IC over the two years, with a minimum in each year |
| 2. Fine | 8 quarterly windows, 2023-Q1 to 2024-Q4 | top 5 per (horizon, target) from stage 1 = 60 | 480 | Mean quarterly IC and its t-statistic |
| 3. Forward | 7 quarterly windows, 2025-Q1 to 2026-Q4 | the stage-2 top 5 per (horizon, target), fixed | 420 | Nothing: these forecasts feed the confirmation only |

Daily rows are about 1.4 million; a fit takes 10 to 60 seconds, so the three stages are roughly a day
of worker time, run overnight, one fit at a time. Models are chosen by **forecast quality (IC)**, not
by backtest profit, so the model choice and the portfolio choice cannot overfit each other.

### Entities

```python
class ModelSearch(Entity):
    """A grid of walk-forwards judged together."""
    label: str
    fit_ids: list[str]
    specs: list[dict[str, Any]]            # algorithm + params
    feature_sets: list[str]                # "screened", "all"
    windows: dict[str, Any]                # {"kind": "yearly"|"quarterly", "first_test", "last_test_end"}
    stage: int
    status: str = "running"
    leaderboard: dict[str, Any] | None = None

class WalkForward(Entity):
    """One Fit, one spec, one feature set, walked over windows."""
    search_id: str | None = None
    fit_id: str
    algorithm: str
    params: dict[str, Any]
    features: list[str]
    query_params: dict[str, Any]
    train_start: date
    embargo_sessions: int                  # = horizon: no training target ends inside a test window
    label: str
    status: str = "running"
    windows: int = 0
    predictions_path: str | None = None    # parquet/walkforward/<id>.parquet
    summary: dict[str, Any] | None = None  # IC per window, overall, rows
    error: str | None = None

class WalkForwardWindow(Entity):
    walk_forward_id: str
    index: int
    train_end: date                        # test_start minus embargo sessions
    test_start: date
    test_end: date
    status: str = "queued"                 # queued | running | done | failed
    fit_run_id: str | None = None
    model_id: str | None = None
    infer_run_id: str | None = None
    error: str | None = None
```

### Reactions

| Reaction | Input | What it does |
|---|---|---|
| `RunModelSearch` | `{label, fit_ids, specs, feature_sets, windows, stage}` | Creates the ModelSearch and one WalkForward per (Fit × spec × feature set) with its windows, defers one `RunWindow` per window. Returns in seconds |
| `RunWindow` | `{window_id}` | `RunFit` with `{"inline": true, "params": {"end": train_end}, "specs": [one], "features": [...]}`, then `Infer` on the test range with `need_target` false; records ids; status `done` or `failed` |
| `CloseWalkForward` | OnChange `WalkForwardWindow.status` | When its windows are all finished: one predictions file (key, `pred_y`, `y`, `r`, `scale`, `entry`, `exit_`, window), IC per window and overall, status |
| `CloseModelSearch` | OnChange `WalkForward.status` | When the search's walk-forwards are all finished: the leaderboard (per horizon and target: IC, its t, worst window) and the shortlist for the next stage |

Changes to existing reactions: `RunFit` gains `inline` (FitModel inside the run, so the window can
score the model it just made), `specs` (fit a subset of the search) and `features` (a subset of the
Fit's features).

## 6. Phase 3: the portfolios

### The account

A pure function in `finance/ml/backtest.py`: scores by date and stock, prices, rules and a cost
schedule in; daily equity, trades and holdings out. No runtime, fully tested.

- **Decision and fills.** A rebalance is decided after day t's close from forecasts made with data
  to t's close; orders fill at t+1's open, sells first, then buys with the cash.
- **Prices.** Unadjusted fills, what one actually pays. A split or bonus during a hold multiplies the
  shares by the factor from `adj_factors`; value is unchanged on the ex-date. No dividends.
- **Whole shares and affordability.** Shares = floor(allocation ÷ (price × (1 + slippage))). On a
  rebalance the ranking is taken over affordable stocks only (one share within the allocation);
  leftover cash stays as cash and earns nothing.
- **Charges.** Every order through the cost module; the depository charge once per stock per sale day.
- **Missing prices and delistings.** Held until it trades; sold at the last close if it never does.
- **Survivorship.** Each day's universe is the stocks with rows that day; nothing from today is
  applied to the past.

### Rules and the grid at ₹10,000

| Rule | Values searched | Meaning |
|---|---|---|
| `top_n` | 1, 2, 3, 4, 5, 6, 8, 10 | Stocks held when full. Phase 1 may narrow this |
| `rebalance` | every 5, 10, 21, 42, 63, 126 sessions | When the portfolio may change |
| `exit` | `band`, `fixed`, `score`, `stop` | `band`: keep while rank ≤ B, sell below. `fixed`: sell all and re-pick each rebalance. `score`: sell when the calibrated forecast turns negative. `stop`: band plus a sell after a close 8 % below cost or 20 % above, checked daily |
| `band` | B = N, 1.5N, 2N, 3N | Only for `band` and `stop` |
| `min_hold` | 0, 1 rebalance | A new holding cannot be sold at the next rebalance |
| `slippage_bps` | 5 (selection); 0, 5, 10 (confirmation) | Against the trader on every fill |

8 × 6 × (1 + 1 + 4 + 4) × 2 = 960 rule combinations per source. Sources: the 60 stage-2 forecasts
(their 2023–2024 predictions) and the 4 baselines, so about 61,000 account simulations. Each is a
pass over about 500 trading days; the grid runs in-process per source, not one run per combination.

### Baselines (no model)

| Baseline | Ranking | Purpose |
|---|---|---|
| `niftybees` | NIFTYBEES bought once with all the capital, held | What a ₹10,000 investor does instead |
| `index` | Nifty 50 and Nifty 500 price indices | The market itself, without costs |
| `equal_weight` | Random draw of N affordable stocks each rebalance, averaged over 50 draws | What picking without skill earns under the same rules and costs |
| `momentum_12_1` | `mom_12_1` | The best-known simple rule |
| `low_vol` | −`vol_120d` | The other one |

### Entities

```python
class BacktestGrid(Entity):
    """Every rule combination for one source, capital and period, evaluated in one run."""
    source: dict[str, Any]       # {"walk_forward_id": ...} | {"baseline": ...}
    capital: float
    costs: str                   # e.g. "zerodha_2026_delivery"
    period: dict[str, str]
    grid: dict[str, Any]         # the values of each rule
    status: str = "running"
    results_path: str | None = None   # parquet/backtests/grid_<id>.parquet: one row per combination
    best: dict[str, Any] | None = None
    error: str | None = None

class BacktestRun(Entity):
    """One account, kept in full: the chosen configurations and their confirmations."""
    source: dict[str, Any]
    rules: dict[str, Any]
    costs: str
    capital: float
    start: date
    end: date
    status: str = "running"
    equity_path: str | None = None    # date, value, cash, invested, charges_to_date
    trades_path: str | None = None    # date, symbol, side, shares, price, value, charges
    summary: dict[str, Any] | None = None
    error: str | None = None
```

### Reactions

| Reaction | Input | What it does |
|---|---|---|
| `RunBacktestGrid` | `{sources, capitals, period, grid, costs}` | Defers one `BacktestGridSource` per (source × capital). Returns in seconds |
| `BacktestGridSource` | `{grid_id}` | Loads scores and prices through DuckDB; runs every combination; writes the results file; publishes it |
| `SelectPortfolio` | `{grid_ids, criterion}` | Over the selection period only: the best configuration by excess return over the matching `equal_weight` baseline, net of everything, with its t-statistic; also the top 20; writes them to its vars |
| `RunBacktest` | `{source, rules, costs, capital, start, end}` | One account in full, for the chosen configurations and the confirmation |

### Summary of an account

End value, CAGR, volatility, Sharpe (risk-free 6.5 %), max drawdown and dates, return of NIFTYBEES
and of the matching equal-weight baseline, excess return and its t-statistic over months, months
beating NIFTYBEES, turnover, charges paid (₹ and % of capital per year), cash drag, average holding
period, number of trades, and per calendar year.

## 7. Selection and the pass bar

With about 1,250 models and 61,000 account configurations, the best one in any period will look
excellent by luck. The protocol:

1. Models are chosen by IC on 2023–2024 (§5). Their 2025–2026 forecasts are made but not looked at.
2. Portfolios are chosen on 2023–2024 only, by `SelectPortfolio`. The choice, and the top 20, are
   committed to the results document **before** any 2025–2026 account is run.
3. The chosen configuration and the top 20 are run unchanged on 2025-01-01 to 2026-10-23.
4. Reported: the chosen one's result; the distribution of the top 20's results (if the best in
   selection is ordinary in confirmation, that is overfitting and is stated as such); the rank
   correlation between selection and confirmation results across all 61,000 configurations.

**The pass bar, set now:** at ₹10,000, in 2025-01 to 2026-10, after charges and 5 bps slippage, the
chosen portfolio ends above NIFTYBEES bought once with ₹10,000, and above its equal-weight baseline,
with a positive t-statistic on monthly excess return. Anything less is reported as a failure.

## 8. Costs

A new module `finance/ml/costs.py` with the schedule the owner pasted from Zerodha's charges page on
2026-10-25, one function per segment returning the ₹ charges for one executed order:

| Item | Delivery (CNC) | Intraday (MIS) | Futures |
|---|---|---|---|
| Brokerage | 0 | min(₹20, 0.03 %) | min(₹20, 0.03 %) |
| STT | 0.1 % buy and sell | 0.025 % sell | 0.05 % sell |
| Exchange (NSE) | 0.00307 % | 0.00307 % | 0.00183 % |
| SEBI | ₹10 / crore | ₹10 / crore | ₹10 / crore |
| Stamp | 0.015 % buy | 0.003 % buy | 0.002 % buy |
| GST | 18 % on brokerage + exchange + SEBI | same | same |
| Depository (DP) | ₹15.34 + GST per stock per sale day | – | – |

The DP charge is not on the pasted page; it is Zerodha's published CDSL charge and is flagged as an
assumption to confirm. Tests check the module against worked examples from Zerodha's brokerage
calculator, entered by hand.

## 9. Tests

| Test | Checks |
|---|---|
| `test_walk_forward_windows` | Window boundaries, embargo in sessions, no training target's exit on or after its window's test start |
| `test_walk_forward_end_to_end` | Synthetic mirror: RunModelSearch → windows → CloseWalkForward → CloseModelSearch leaderboard |
| `test_runfit_inline_specs_features` | RunFit fits only the given spec and features, inside the run |
| `test_costs_schedule` | Each segment against hand-entered calculator examples; DP charge once per stock per sale day |
| `test_backtest_known_outcome` | Synthetic prices where the top-ranked stock rises 1 % a day: exact end value after charges |
| `test_backtest_band_rule` | Rank 3 → 15 with N = 10, B = 20 is held; → 25 is sold |
| `test_backtest_split` | A 1:2 split mid-hold doubles shares, value unchanged |
| `test_backtest_affordability_and_cash` | A stock dearer than the allocation is passed over; leftover cash carries |
| `test_backtest_fills_next_open` | Decisions at t fill at t+1's open, never at t's close |
| `test_backtest_stop_rule` | Stop-loss and take-profit trigger on the close and fill the next open |
| `test_grid_matches_single_runs` | A combination's row in a grid equals the same combination run alone |
| `test_daily_rows_adjusted` | The daily rows' returns are continuous across a split |
| `test_need_target_false` | The last h sessions appear with a null target |

## 10. What changes where

| File | Change |
|---|---|
| `apps/finance/src/finance/ml/forecast.py` | Daily rows on adjusted prices; slow features; raw-return target; `need_target` |
| `apps/finance/src/finance/ml/entities.py` | ModelSearch, WalkForward, WalkForwardWindow, BacktestGrid, BacktestRun |
| `apps/finance/src/finance/ml/reactions.py` | RunModelSearch, RunWindow, CloseWalkForward, CloseModelSearch, RunBacktestGrid, BacktestGridSource, SelectPortfolio, RunBacktest; `inline`, `specs`, `features` on RunFit |
| `apps/finance/src/finance/ml/backtest.py` | New: the account, the grid, the baselines |
| `apps/finance/src/finance/ml/costs.py` | New: the cost schedules |
| `apps/finance/src/finance/ml/estimators/` | lightgbm `fit_batches` streams like xgboost (it concatenates today); fine at 1.4 million rows |
| `apps/finance/src/finance/triggers.py` | OnChange for WalkForwardWindow.status and WalkForward.status |
| `apps/finance/migrations/` | The five new tables |
| `apps/finance/fits/round3/` | Twelve Fit JSONs |
| `apps/finance/tests/` | §9 |
| `docs/finance-ml.md` | The walk-forward, search and backtest sections |

Order of PRs:

1. Costs, the account, baselines and `RunBacktestGrid`; then **Phase 1 is run and its results
   written** (the capital study needs no model).
2. Adjusted daily rows, the new features and targets, the Fits and the screen.
3. Walk-forward and model search; stages 1 to 3 run overnight.
4. Portfolio grid, selection, confirmation; the results document.

## 11. Review decisions (2026-10-25)

1. **Capitals:** ₹10,000 and up; smaller accounts are out of scope.
2. **Compute:** about a day of overnight worker time for the model search is acceptable.
3. **Affordability tilt:** accepted; the baselines carry the same restriction.
4. **The US:** later, as a separate step.

## 12. Phase 1 results: the capital study (2026-10-25)

Label `capital study 2026-10-25 v3`: 20 `BacktestGrid` rows (4 sources × 5 capitals), 120 rule
combinations each (N {1, 2, 3, 4, 5, 6, 8, 10, 15, 20} × rebalance every {5, 10, 21, 42, 63, 126}
sessions × exit {band 2N, fixed}), random picks averaged over 20 draws; 2023-01-02 to 2026-10-23,
944 trading days; Zerodha delivery charges and 5 bps slippage. `SummarizeBacktestGrids` holds the
tables. Two bugs were found and fixed on the way (#82, #83): an account eaten by charges went below
zero, and a ruined account broke the following years' returns.

### What it costs to run a ₹10,000 account

Charges per year as % of capital, random picks, band exit (rows: stocks held; columns: sessions
between rebalances):

| N | 5 (weekly) | 10 | 21 (monthly) | 42 | 63 (quarterly) | 126 (half-yearly) |
|---|---|---|---|---|---|---|
| 1 | 20.6 | 9.4 | 4.4 | 2.0 | 1.4 | 0.7 |
| 2 | 35.0 | 14.0 | 5.9 | 2.6 | 1.8 | 0.8 |
| 3 | 51.9 | 18.6 | 7.5 | 3.5 | 2.4 | 1.1 |
| 5 | 87.7 | 31.8 | 11.1 | 5.0 | 3.3 | 1.5 |
| 10 | 183.8 | 74.4 | 23.3 | 9.4 | 5.8 | 2.5 |
| 20 | 365.7 | 169.4 | 63.2 | 19.6 | 11.3 | 5.0 |

At ₹1 lakh the same grid costs 0.5 to 2.6 % a year for anything rebalanced monthly or slower; the
difference is the fixed ₹18 DP charge per sale. **For ₹10,000: rebalance every two to six months;
up to 3 stocks monthly, 5 every two months, 10 quarterly or half-yearly.** Weekly trading ruins
the account (18.5 % of the random-pick setups at ₹10,000 reached zero).

### What the simple strategies earned at ₹10,000

NIFTYBEES bought once: ₹12,790 (+31.5 % in 2023–24, −2.7 % in 2025–26).

| Source | Median end value over 120 setups | 10th percentile | Best | Setups beating NIFTYBEES |
|---|---|---|---|---|
| Momentum 12-1 | ₹11,594 | ₹2,423 | ₹53,734 (N 10, monthly, band) | 42 % |
| Low volatility | ₹10,528 | ₹9,835 | ₹16,518 | 19 % |
| Random picks | ₹11,063 | ₹0 | ₹20,341 | 42 % |

### Chosen on 2023–24, judged on 2025–26 (₹10,000)

| Source | Best setup on 2023–24 | Its 2023–24 | Its 2025–26 | Top 20 setups of 2023–24, median 2025–26 | Of those, beating NIFTYBEES | Rank correlation 2023–24 vs 2025–26 |
|---|---|---|---|---|---|---|
| Momentum 12-1 | N 1, every 10 sessions, fixed | +269 % | **−88 %** | +9 % | 60 % | +0.45 |
| Low volatility | N 20, half-yearly, band | +54 % | +2 % | +5 % | 70 % | +0.55 |
| Random picks | N 1, every 42 sessions, band | +101 % | −3 % | −5 % | 20 % | +0.84 |

Momentum by number of stocks at ₹10,000 (median over rebalance and exit):

| N | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2023–24 | +38 % | +17 % | +18 % | +44 % | +53 % | +66 % | +72 % | +61 % | +74 % | +83 % |
| 2025–26 | −79 % | −78 % | −42 % | −32 % | −28 % | −15 % | −1 % | +81 % | +58 % | +18 % |
| Max drawdown | −85 % | −83 % | −60 % | −53 % | −45 % | −43 % | −39 % | −38 % | −35 % | −40 % |

### Reading

- **Picking the single best setup from history is a trap.** The best momentum setup of 2023–24 lost
  88 % in 2025–26. The selection protocol of §7 (commit to the top configurations, judge the
  group) is needed, not optional.
- **Concentration is what kills a small account, not charges.** Momentum with 1 to 5 stocks lost
  28 to 79 % in 2025–26; with 10 to 15 it gained 58 to 81 %. At ₹10,000, 10 stocks is affordable
  only quarterly or slower (5.8 % or 2.5 % a year in charges).
- **Low volatility is steady and modest.** Rarely the best, almost never a disaster; its 2023–24
  ranking carries into 2025–26 best of the three.
- **The bar for the model (phase 2):** beat NIFTYBEES and 12-1 momentum with about 10 stocks,
  rebalanced quarterly or half-yearly, at ₹10,000, out of sample.
