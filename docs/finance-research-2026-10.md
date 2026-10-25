# Finance research, October 2026: forecasts, timing studies and what trades

Status: complete as of 2026-10-25. Data through 2026-10-23 (minute bars) and 2026-10-23 (daily).
Repo state: `main` at the merge of this document. Every number in this document is reproducible
from a run, a table or a query linked here.

This is the end-to-end record of the October 2026 work on the finance app: the data, the
forecasting models (what they predict, their features, their parameters, how they were trained and
validated), the three timing studies (overnight and daytime, every pair of times, weekends), the
portfolio-level tests, the costs assumed, and what is and is not established. The design notes and
the day-by-day reasoning live in [finance-forecast.md](finance-forecast.md); this document is the
consolidated result.

## Contents

1. [Findings](#1-findings)
2. [Data](#2-data)
3. [How the work runs](#3-how-the-work-runs)
4. [The forecasting problem](#4-the-forecasting-problem)
5. [Features](#5-features)
6. [Models and training](#6-models-and-training)
7. [Validation design and metrics](#7-validation-design-and-metrics)
8. [Validation results](#8-validation-results)
9. [Analysis A: the session model as a daily ranking](#9-analysis-a-the-session-model-as-a-daily-ranking)
10. [Analysis B: the session study, overnight and daytime](#10-analysis-b-the-session-study-overnight-and-daytime)
11. [Analysis C: the window study](#11-analysis-c-the-window-study)
12. [Analysis D: picked-stock portfolios and weekends, out of sample](#12-analysis-d-picked-stock-portfolios-and-weekends-out-of-sample)
13. [Costs](#13-costs)
14. [Limitations](#14-limitations)
15. [Corrections to earlier statements](#15-corrections-to-earlier-statements)
16. [Next steps](#16-next-steps)
17. [Appendix](#17-appendix)

## 1. Findings

| # | Finding | Evidence | Status |
|---|---|---|---|
| 1 | The rest-of-session model, asked at 11:00, ranks stocks well enough to trade: long its top 10 %, short its bottom 10 %, close both at the close. | 25.1 bps a day gross, t = 7.6, positive on 81 % of 80 test days; +12.9 % net of 10 bps per pair over Jul to Oct 2026, worst drawdown 0.6 %. [§9](#9-analysis-a-the-session-model-as-a-daily-ranking) | Strongest result; paper-trade before money |
| 2 | The same model is not a long-only signal: its top decile barely breaks even because the whole market drifted down during the day. | Top decile +0.7 bps from 11:00 to close; all stocks −10.5 bps. [§9](#9-analysis-a-the-session-model-as-a-daily-ranking) | Established |
| 3 | Overnight holds (15:14 → 09:29) pay for stocks picked on their past record, but only after futures-level costs. | Top 20 of 2024 and 2025 made 21.0 bps a night in 2026 (t = 2.4 gross, 1.9 after 4 bps); ₹1 lakh → ₹1.40 lakh through futures, ₹92,346 through cash delivery. [§12](#12-analysis-d-picked-stock-portfolios-and-weekends-out-of-sample) | Borderline; most picks have no futures |
| 4 | The market-wide overnight edge has nearly disappeared. | Equal-weight all liquid stocks: 33 bps a night (late 2023), 23 (2024), 5 (2025), 1.8 (2026). [§12](#12-analysis-d-picked-stock-portfolios-and-weekends-out-of-sample) | Established |
| 5 | A fixed daytime short (09:29 → 14:29) on picked stocks does not beat shorting everything. | 8.7 bps a day gross, 3.7 after 5 bps, t = 0.5. [§12](#12-analysis-d-picked-stock-portfolios-and-weekends-out-of-sample) | Not established |
| 6 | Holding across a weekend or holiday pays less than an ordinary night, every year. | 15:14 → 09:29, all stocks: 2024 +15 vs +26 bps; 2025 +0.3 vs +6.3; 2026 −18.9 vs +8.5. [§12](#12-analysis-d-picked-stock-portfolios-and-weekends-out-of-sample) | Consistent direction; individual years not significant |
| 7 | New features raised the 15-minute IC from 0.050 to 0.070, but 15-minute edges are smaller than one round-trip cost. | Top-minus-bottom decile 5 bps per 15 minutes vs 10 bps cost. [§8](#8-validation-results), [§9](#9-analysis-a-the-session-model-as-a-daily-ranking) | Established |
| 8 | Training on 2023 to 2025 beats training on the first half of 2026 alone at every horizon. | IC 0.070 vs 0.062 (15 m), 0.040 vs 0.023 (60 m), 0.076 vs 0.051 (session); 21-session H1 fits flip sign. [§8](#8-validation-results) | Established |
| 9 | Daily and 21-session horizons, NSE and US, are not usable yet. | Next-day IC 0.049 (NSE) and 0.035 (US) with large standard errors; 21-session has about three independent observations in the test window. [§8](#8-validation-results) | Open |
| 10 | US indices and MCX commodities show no tradeable overnight or daytime pattern. | [§10](#10-analysis-b-the-session-study-overnight-and-daytime) | Established for the periods studied |

## 2. Data

All data lives in the private Hugging Face dataset `shr22avan/finance-data`, mirrored on the box at
`~/Workspaces/indian-finance-data/data`. The ML code reads it through DuckDB views installed by
[`data.install_views`](../apps/finance/src/finance/ml/data.py).

| View | Source | Range used | Grain | Used for |
|---|---|---|---|---|
| `equity_1m` | Kite Connect historical minute bars | 2023-10-03 → 2026-10-23, 602 symbols (stocks, ETFs and 27 NSE indices) | 1 minute | Minute Fits (15 m, 60 m, session), Analyses A, C, D |
| `equity` | NSE bhavcopy | 2021-09-01 → 2026-10-23, every EQ series stock | 1 day | Daily NSE Fits, session study (NSE) |
| `index_daily` | niftyindices.com | used from 2021-09-01 to 2026-10-23 | 1 day | Index returns, sector indices, session study (NSE indices) |
| `us_equity` | Yahoo Finance | 2015-01-01 → 2026-10-23, listed US stocks plus ETFs and ^VIX (2,627 pass the study's $5 million floor) | 1 day | Daily US Fits, session study (US) |
| `commodity_continuous` | MCX bhavcopy, front-month continuous | 2021-09-01 → 2026-10-23, 32 contracts | 1 day | Overseas/commodity features, session study (MCX) |
| `index_constituents` | NSE index files, snapshots | 2026 snapshots | per snapshot | Industry, sector index and index-membership features |
| `us_sectors` | Nasdaq screener snapshots | 2026 snapshots | per snapshot | US sector and size-bucket features |

**Universe.** Minute bars cover the Kite universe: the Nifty 500 members, every stock with enough
turnover, and stocks that stayed in either set during the last 250 sessions. Each analysis then
applies its own liquidity floor, stated where it is used. The most common floor below is a median
daily rupee turnover of at least ₹5 crore since 2024-01-01, which leaves 576 stocks.

**Prices.** NSE prices are unadjusted (corporate actions do not occur inside a minute or a one-day
hold often enough to matter at these horizons; the daily Fits use log returns over at most 21
sessions). US prices from Yahoo are split-adjusted. The official NSE daily open is the pre-open
auction print, which is not a tradeable price for most stocks; see [§10](#10-analysis-b-the-session-study-overnight-and-daytime).

## 3. How the work runs

Every job is a reaction in the backend-builder runtime (the `finance` app), visible as a run in the
admin console at http://127.0.0.1:9400 with its inputs, outputs (`vars`) and child runs.

```
RegisterFit ─► RunFit ─► LoadRows (DuckDB query → Parquet rows file)
                    └──► FitModel ×5 (deferred, one per model) ─► UploadModel ─► CloseFitRun
Validate ─► LoadRows (validation range) ─► Infer ×5 ─► Evaluate ×3 thresholds
FitSweep / ValidateSweep: one parent run over many FitRuns
SessionStudy, WindowStudy: query → Parquet in parquet/studies/ → PublishArtifact
```

| Reaction | Code | What it does |
|---|---|---|
| `LoadRows` | [reactions.py#L96](../apps/finance/src/finance/ml/reactions.py#L96) | Runs a Fit's query with its parameters through [`data.write_rows`](../apps/finance/src/finance/ml/data.py#L252) (quarterly chunks, 45-day warm-up, DuckDB capped at 12 GB with an 80 GB spill cap); hashes the file with [`data.file_hash`](../apps/finance/src/finance/ml/data.py#L295). |
| `RunFit` | [reactions.py#L127](../apps/finance/src/finance/ml/reactions.py#L127) | Builds the rows once, creates one `Model` per search point, defers one `FitModel` each. |
| `FitModel` | [reactions.py#L174](../apps/finance/src/finance/ml/reactions.py#L174) | Fits in a spawned subprocess ([`fitting.fit_job`](../apps/finance/src/finance/ml/fitting.py#L44)), saves the model directory, uploads it. |
| `RetryModels` | [reactions.py#L245](../apps/finance/src/finance/ml/reactions.py#L245) | Requeues failed models after their cause is fixed. |
| `Infer` | [reactions.py#L333](../apps/finance/src/finance/ml/reactions.py#L333) | Scores a rows file with a model; writes `parquet/inference/<infer_run_id>.parquet` with the key, prediction, truth, raw return `r` and `scale`. |
| `Evaluate` | [reactions.py#L457](../apps/finance/src/finance/ml/reactions.py#L457) | Computes the metrics of [§7](#7-validation-design-and-metrics) on an inference table. |
| `Validate` / `ValidateSweep` | [reactions.py#L547](../apps/finance/src/finance/ml/reactions.py#L547), [#L681](../apps/finance/src/finance/ml/reactions.py#L681) | Infer and evaluate every model of one or many FitRuns on a later range under a label. |
| `SessionStudy` | [reactions.py#L699](../apps/finance/src/finance/ml/reactions.py#L699) | [§10](#10-analysis-b-the-session-study-overnight-and-daytime). |
| `WindowStudy` | [reactions.py#L727](../apps/finance/src/finance/ml/reactions.py#L727) | [§11](#11-analysis-c-the-window-study). |

**Fitting at scale.** A fit never holds the full matrix. The child completes the schema from the rows
file in one aggregate pass ([`data.complete_schema`](../apps/finance/src/finance/ml/data.py#L328)),
then streams float32 chunks of one million rows ([`data.iter_batches`](../apps/finance/src/finance/ml/data.py#L376))
to the estimator's `fit_batches`. Measured on the 20-million-row 15-minute file: xgboost peaks at
14 GB, the ridge at 5 GB (the in-memory path peaked at 27 GB and was killed twice on 2026-10-25).
The worker unit has `MemoryMax=24G` ([systemd/bb@.service](../systemd/bb@.service)). With
`device=auto` the GTX 1080 (8 GB) is tried first; a CUDA out-of-memory falls back to the CPU
([`fitting.cuda_out_of_memory`](../apps/finance/src/finance/ml/fitting.py#L80)). The two
20-million-row runs therefore fitted on the CPU; every smaller run fitted its trees on the GPU.

## 4. The forecasting problem

Each Fit predicts one number per (stock, decision time): the stock's **volatility-scaled log return**
over a fixed horizon, entered at the first price a decision could get.

### Minute Fits (15 m, 60 m, session)

Query: [`ROWS_SQL`](../apps/finance/src/finance/ml/forecast.py#L206), built by
[`fit_definition`](../apps/finance/src/finance/ml/forecast.py#L870).

- **Row:** one per (symbol, decision minute t), with t every `step_minutes` = 5 minutes inside the
  regular session (minute index 0 to 374; the Muhurat evening session is excluded).
- **Liquidity filter:** rupee turnover over the trailing `liq_minutes` = 60 minutes ≥
  `min_turnover` = ₹20 lakh. Rows below it get no prediction.
- **Entry:** the open of the bar after t. **Exit:** the close of the bar `horizon_minutes` later,
  capped at the session's last bar.
- **Target:**

      r     = ln(exit / entry)
      scale = σ₁ · sqrt(min(horizon_minutes, minutes_left − 1))
      y     = r / scale

  where σ₁ is the stock's realised one-minute volatility over the trailing `vol_minutes` = 60
  minutes. A prediction converts back to a price with `price × exp(pred × scale)`.
- **Horizons:** 15 minutes, 60 minutes, and 375 minutes, which always reaches the close (the
  "session" Fit: from any decision point to the end of the day).

### Daily Fits (1 session and 21 sessions, NSE and US)

Query: [`daily_sql(market)`](../apps/finance/src/finance/ml/forecast.py#L833) over
[`_DAILY_TEMPLATE`](../apps/finance/src/finance/ml/forecast.py#L714), built by
[`daily_fit_definition`](../apps/finance/src/finance/ml/forecast.py#L846).

- **Row:** one per (symbol, session t), decided after t's close.
- **Liquidity filter:** 20-session median turnover ≥ `min_turnover_day` = ₹1 crore (NSE) or
  $10 million (US; the same number in the market's currency).
- **Entry:** the next session's open. **Exit:** the close of the `horizon_sessions`-th session
  (1 = the entry session's own close; 21 ≈ one month).
- **Target:** `y = ln(exit / entry) / (σ_d · sqrt(horizon_sessions))`, σ_d the 20-session daily
  volatility.
- **History:** NSE from 2021-09-01, US from 2015-01-01.

### Fit definitions

All seven are registered from JSON files in [apps/finance/fits/round2/](../apps/finance/fits/round2/).

| Fit | Fit id | File | Query parameters | Features |
|---|---|---|---|---|
| `forecast 15m r2` | `38b41524-6c7f-40ed-8282-68920ad5589a` | [forecast_15m_r2.json](../apps/finance/fits/round2/forecast_15m_r2.json) | horizon_minutes 15, step_minutes 5, vol_minutes 60, liq_minutes 60, min_turnover 2,000,000, start 2023-10-03 | 77 |
| `forecast 60m r2` | `c92c5f27-6595-43ea-a1c2-d027370be0fa` | [forecast_60m_r2.json](../apps/finance/fits/round2/forecast_60m_r2.json) | horizon_minutes 60, otherwise as 15m | 73 |
| `forecast 375m r2` | `659b68fb-d2a7-4ad2-939a-006e98df7b7f` | [forecast_375m_r2.json](../apps/finance/fits/round2/forecast_375m_r2.json) | horizon_minutes 375, otherwise as 15m | 80 |
| `forecast 1d nse r2` | `20e339af-065d-435f-b613-a9e67eb76c21` | [forecast_1d_nse_r2.json](../apps/finance/fits/round2/forecast_1d_nse_r2.json) | horizon_sessions 1, vol_sessions 20, min_turnover_day 10,000,000, start 2021-09-01 | 55 |
| `forecast 1d us r2` | `20e0e7df-1d47-40b2-9208-28956d5730d4` | [forecast_1d_us_r2.json](../apps/finance/fits/round2/forecast_1d_us_r2.json) | as 1d NSE, start 2015-01-01 | 33 |
| `forecast 21d nse r2` | `b13efc10-181f-459b-b90f-3318ee7aff4f` | [forecast_21d_nse_r2.json](../apps/finance/fits/round2/forecast_21d_nse_r2.json) | horizon_sessions 21, otherwise as 1d NSE | 43 |
| `forecast 21d us r2` | `4fafe105-7392-4bc7-a43a-30a0db5c3b05` | [forecast_21d_us_r2.json](../apps/finance/fits/round2/forecast_21d_us_r2.json) | horizon_sessions 21, otherwise as 1d US | 33 |

The `end` parameter is set per FitRun ([§6](#6-models-and-training)); the files carry 2030-12-31.

## 5. Features

Every feature is computed from data available at the decision time;
[tests/test_forecast.py](../apps/finance/tests/test_forecast.py) checks it
(`test_nothing_at_t_sees_a_bar_after_t`, `test_daily_layer_is_the_previous_session_and_filters_hold`,
`test_daily_rows_decide_at_the_close_and_enter_at_the_next_open`). Lists and one-line definitions are in
[forecast.py#L45](../apps/finance/src/finance/ml/forecast.py#L45) (minute) and
[forecast.py#L459](../apps/finance/src/finance/ml/forecast.py#L459) (daily).

### Minute rows: 99 candidate features in 11 layers

| Layer | Features | Meaning |
|---|---|---|
| Intraday | `r1 r5 r15 r30 r60 r_open` | Log return over the last 1, 5, 15, 30, 60 minutes and since the open |
| | `range_pos`, `vwap_dev` | Position in today's high–low range; deviation from today's VWAP |
| | `rv30`, `rv_open` | Realised volatility over 30 minutes and since the open |
| | `relvol5`, `relvol15` | Volume in the last 5 / 15 minutes against the same clock time over 20 sessions |
| | `turn_liq` | Rupee turnover over the trailing 60 minutes |
| | `i_r5 … i_r_open`, `x_r15`, `x_r60` | Nifty 50 returns over the same spans; the stock minus the index over 15 and 60 minutes |
| Buying pressure | `imb5/15/30` | Signed volume imbalance: volume on up minutes minus down minutes, over total |
| | `clv5/15/30` | Close-location volume (accumulation–distribution) |
| | `vwret5/15/30`, `up5/15/30` | Volume-weighted return; share of up minutes |
| | `impact15/30`, `turn_accel` | Absolute return per rupee traded; last 5 minutes' turnover over the last 30 |
| Shape | `or_pos`, `or_break`, `clv_prev` | Position in and break of the first 15 minutes' range; yesterday's close location |
| Daily (as of yesterday) | `ret_1d gap ret_5d ret_20d vol_20d dist_hi dist_lo turn_rel deliv_pct rs_20` | Yesterday's return, today's gap, 5/20-day returns, 20-day volatility, distance from 52-week high and low, turnover against its 20-day median, NSE delivery percentage, 20-day strength against Nifty 50 |
| Participation | `log_trades avg_trade trades_rel` | Yesterday's trade count, average trade size, trades against the 20-day median |
| Attributes | `industry sector_idx in_n50 in_n500 in_mid100 in_small250 log_px tick_bps log_turn20 vol_60d` | NSE industry and sector index, index membership, price level, tick size over price, typical turnover and volatility |
| Peers and sector | `p_r5 … p_r_open`, `xp_r15 xp_r60`, `pw_r15`, `sp_5d sp_20d`, `s_r15 s_r60 s_r_open xs_r15` | Same-industry peers' returns, the stock minus peers, turnover-weighted peer return, slow spread against peers, sector index returns and the stock minus its sector |
| Breadth | `breadth_open breadth_15 disp_15` | Share of stocks up since the open and over 15 minutes; cross-sectional spread of 15-minute returns |
| Regime | `mkt_vol20 mkt_vol60 mkt_ret60 vix` | Nifty 50 volatility and 60-session return, India VIX, as of yesterday |
| Overseas | `spy_oc spy_cc qqq_oc qqq_cc spy_5d spy_range gold_cc silver_cc crude_cc` | The previous US session and the previous MCX session |
| Clock | `m m_left dow dom moy expiry dte after_holiday` | Minutes since open and to close, weekday, day of month, month, expiry day, days to expiry, day after a holiday |

### Daily rows: candidate features (the 1d NSE Fit keeps 55, the US Fits 33)

Price (`ret_1d ret_5d ret_20d ret_60d vol_20d vol_60d dist_hi dist_lo turn_rel deliv_pct rs_20
rs_60 gap_d range_pos_d clv_d`), session shape from minute bars where held (`rv_day vwap_dev_close
r_first60 r_last60 relvol_day`), attributes, peers (`p_ret_1d p_ret_5d p_ret_20d pw_ret_20d sp_5d
sp_20d sp_60d`), sector (`s_ret_1d s_ret_5d s_ret_20d xs_ret_20d`), breadth (`breadth_d disp_d`),
regime (`mkt_ret_1d mkt_ret60 mkt_vol20 mkt_vol60 vix`), overseas (NSE rows: the US session and
MCX session dated t; US rows: the Nifty session dated the entry day plus `qqq_cc iwm_cc gld_cc
slv_cc uso_cc`), clock (`dow dom moy dte after_holiday`). The US has no trade counts, delivery or
minute bars, so those columns are absent.

### Selection: the IC screen

Each Fit's feature list was chosen by the `Screen` reaction
([reactions.py#L482](../apps/finance/src/finance/ml/reactions.py#L482),
[`forecast.screen`](../apps/finance/src/finance/ml/forecast.py#L978)): for each candidate, the
within-decision-point Spearman IC with the target, per year, with its t-statistic. A feature is kept
by [`features_from_screen`](../apps/finance/src/finance/ml/forecast.py#L925) when it is significant
(|t| ≥ 2) in at least one year, has the same sign in every year it is significant, and its mean
absolute IC over the years is at least 0.003; categorical and context features
(industry, sector index, month, weekday) are always kept. The full screen tables are in
[finance-forecast.md, second round screens](finance-forecast.md#second-round-the-screens-2026-10-24-evening).
The kept lists are exactly the `features` arrays of the Fit JSON files.

## 6. Models and training

### Search space (identical for all seven Fits)

| Model | Library | Parameters |
|---|---|---|
| Ridge baseline | closed form (equal to scikit-learn `Ridge`) | alpha 1.0; nulls imputed with per-column medians from a 1-million-row reservoir sample; intercept unpenalised |
| xgboost d3 n300 | xgboost 3.2.0 | max_depth 3, n_estimators 300 |
| xgboost d3 n600 | | max_depth 3, n_estimators 600 |
| xgboost d4 n300 | | max_depth 4, n_estimators 300 |
| xgboost d4 n600 | | max_depth 4, n_estimators 600 |

Common xgboost settings: learning_rate 0.05, min_child_weight 200, subsample 0.8,
colsample_bytree 0.7, reg_lambda 5.0, objective `reg:squarederror`, tree_method `hist`, 256 bins
(`QuantileDMatrix`), seed 42. Categorical features enter as integer codes (ordinal). Implementations:
[xgboost_.py](../apps/finance/src/finance/ml/estimators/xgboost_.py#L86),
[linear.py](../apps/finance/src/finance/ml/estimators/linear.py#L107).

### Training runs

Each Fit was trained twice by one `FitSweep` run (`fff1066a-1b93-43ba-8902-7c8535f608f8`,
2026-10-24 22:42 to 23:46 IST, after a first attempt failed on a database serialization conflict,
issue #69): once on all history through 2025-12-31 and once on 2026-01-01 to 2026-06-30 alone.
Nine models that failed (seven on GPU or system memory, two killed by a worker restart) were
requeued with `RetryModels` after the fixes and all 70 models fitted.

| Fit | Fit on | FitRun id | Fit rows | Models |
|---|---|---|---|---|
| forecast 15m r2 | 2023-10-03 → 2025-12-31 | `20c2fb75-9a36-48a7-932a-d8925fa34c08` | 20,122,891 | 5 |
| forecast 15m r2 | 2026-01-01 → 2026-06-30 | `c04e452a-5c55-4f18-a486-e36bcdf23443` | 4,840,707 | 5 |
| forecast 60m r2 | 2023-10-03 → 2025-12-31 | `8e53b42c-53b5-42c5-920e-f788b6ce1266` | 17,552,475 | 5 |
| forecast 60m r2 | 2026-01-01 → 2026-06-30 | `cdcbbd77-4e8a-46b1-b4b1-22b83e004860` | 4,225,189 | 5 |
| forecast 375m r2 | 2023-10-03 → 2025-12-31 | `51121bd8-f9b8-4ced-ba48-ec730d2cfe82` | 20,985,571 | 5 |
| forecast 375m r2 | 2026-01-01 → 2026-06-30 | `b7fb407b-c8ff-411e-b3e9-d74c6c72ee06` | 5,046,127 | 5 |
| forecast 1d nse r2 | 2021-09-01 → 2025-12-31 | `94661d55-f37a-42de-9405-45c4ccbabcd1` | 1,356,849 | 5 |
| forecast 1d nse r2 | 2026-01-01 → 2026-06-30 | `3c9bdbaf-dc4e-49b0-bafe-c08c77b001ad` | 184,057 | 5 |
| forecast 1d us r2 | 2015-01-01 → 2025-12-31 | `21bd0fa2-666a-4a3f-bacf-e34774344baf` | 4,380,630 | 5 |
| forecast 1d us r2 | 2026-01-01 → 2026-06-30 | `6035dca5-9907-4c0a-9655-2829b5a9d3f3` | 293,088 | 5 |
| forecast 21d nse r2 | 2021-09-01 → 2025-12-31 | `6de0cb13-e991-492f-b2e1-e055e0f485ae` | 1,353,690 | 5 |
| forecast 21d nse r2 | 2026-01-01 → 2026-06-30 | `2635cdc9-07e2-4dc6-88fd-2428b201da59` | 183,723 | 5 |
| forecast 21d us r2 | 2015-01-01 → 2025-12-31 | `df43d483-a752-4695-8ce3-27ea09311b7f` | 4,380,630 | 5 |
| forecast 21d us r2 | 2026-01-01 → 2026-06-30 | `41e5a4dc-06e7-43a2-8f00-ab35b26e3c1e` | 293,088 | 5 |

Models are stored at `models/<model_id>` in `shr22avan/finance-data`; each directory holds the model
file, `schema.json` (feature levels) and `fit.json` (algorithm, parameters, device, seed, rows).

## 7. Validation design and metrics

**Range and label.** Every model was scored on **2026-07-01 to 2026-10-23** by one `ValidateSweep`
run (`e7a3937e-eaa4-4570-b505-afa7f0a23d55`, 125 minutes) under the label `validation 2026 H2`.
This range follows both training ranges, so every number is out of sample for both splits.

**Metrics** ([metrics.py](../apps/finance/src/finance/ml/metrics.py)), computed on the inference
table of each model:

| Metric | Definition |
|---|---|
| IC | Mean over decision points of the Spearman correlation between prediction and realised `y` across stocks at that point ([#L92](../apps/finance/src/finance/ml/metrics.py#L92)). Points with fewer than 20 stocks are skipped. |
| IC se | Standard deviation of the per-point ICs over the square root of their number. Assumes independent points; overlapping horizons make it too small (see [§14](#14-limitations)). |
| IC by month | The same, per calendar month. |
| Calibration slope | Slope of realised `y` on predicted `y` ([#L137](../apps/finance/src/finance/ml/metrics.py#L137)). 1 is right-sized; below 1 the model predicts moves too large. |
| Net return at threshold | Mean raw return `r` over rows whose predicted raw move `pred × scale` exceeds the threshold (0.1, 0.2, 0.5 %), minus 10 bps ([#L160](../apps/finance/src/finance/ml/metrics.py#L160)). Long-only. |
| Top-5 net | Mean raw return of the 5 highest predictions at each decision point, minus 10 bps ([#L196](../apps/finance/src/finance/ml/metrics.py#L196)). |
| Top-5 spread | Mean raw return of the top 5 minus the bottom 5 at each decision point. Before costs. |

## 8. Validation results

### Best model per Fit and split

| Horizon | Fit on | Best model | IC | IC by month (Jul / Aug / Sep / Oct) | Ridge IC | Calibration slope |
|---|---|---|---|---|---|---|
| 15 minutes | to 2025 | xgboost d4 n600 | **0.070** (se 0.0009) | 0.073 / 0.077 / 0.069 / 0.061 | 0.053 | 0.31 |
| 15 minutes | 2026 H1 | xgboost d3 n600 | 0.062 | 0.069 / 0.071 / 0.060 / 0.052 | 0.047 | 0.13 |
| 60 minutes | to 2025 | xgboost d4 n600 | 0.040 | 0.034 / 0.050 / 0.042 / 0.039 | 0.033 | 0.16 |
| 60 minutes | 2026 H1 | xgboost d4 n600 | 0.023 | 0.027 / 0.026 / 0.022 / 0.021 | 0.017 | −0.03 |
| Session | to 2025 | xgboost d3 n600 | **0.076** (se 0.0009) | 0.076 / 0.084 / 0.075 / 0.064 | 0.061 | 0.41 |
| Session | 2026 H1 | xgboost d4 n600 | 0.051 | 0.063 / 0.066 / 0.055 / 0.018 | 0.036 | 0.10 |
| 1 day NSE | to 2025 | xgboost d4 n600 | 0.049 (se 0.007) | 0.029 / 0.059 / 0.056 / 0.041 | 0.027 | 0.28 |
| 1 day NSE | 2026 H1 | xgboost d4 n300 | 0.050 (se 0.005) | 0.040 / 0.053 / 0.058 / 0.046 | 0.014 | 1.44 |
| 1 day US | to 2025 | xgboost d3 n600 | 0.035 (se 0.012) | 0.056 / 0.011 / 0.044 / 0.091 | 0.030 | −0.13 |
| 1 day US | 2026 H1 | ridge | 0.011 (se 0.013) | | 0.011 | 0.13 |
| 21 days NSE | to 2025 | xgboost d4 n300 | 0.061 (se 0.005) | 0.051 / 0.089 / 0.050 / – | −0.107 | 0.16 |
| 21 days NSE | 2026 H1 | ridge | 0.004 (trees −0.034 to −0.099) | | 0.004 | 0.00 |
| 21 days US | to 2025 | xgboost d3 n600 | 0.100 (se 0.008) | 0.102 / 0.065 / 0.142 / – | 0.069 | 1.13 |
| 21 days US | 2026 H1 | ridge | −0.012 (trees −0.015 to −0.042) | | −0.012 | 0.53 |

The first round (`forecast 15m v2`, 44 features, same split) reached IC 0.050 with xgboost depth 4,
500 trees; its ridge baseline 0.038.

### Reading

- **Feature additions worked.** The 15-minute IC rose by 40 % and is steady month to month.
- **More history wins.** The 2026-H1-only fits are weaker at every horizon and flip sign at 21
  sessions. Six months is too little for 70 to 80 features.
- **The session horizon carries the most information** (IC 0.076). It is the subject of
  [§9](#9-analysis-a-the-session-model-as-a-daily-ranking).
- **Calibration is poor for trees** (slope 0.1 to 0.5): predicted magnitudes are two to ten times
  too large, so the threshold metrics trade far more often than intended. The `rescale` parameter
  of `Evaluate` corrects this and has not yet been applied.
- **Long-only threshold results are mostly negative.** The best is the session model at the 0.5 %
  threshold: +20.8 bps a trade after 10 bps over 60,863 trades (by month −17, +15, +21, +51 bps).
  [§9](#9-analysis-a-the-session-model-as-a-daily-ranking) shows why long-only undersells it.
- **21 sessions:** with a four-month test window and a one-month horizon there are about three
  independent outcomes per stock. The US line (IC 0.10, top-5 +203 bps net) is not evidence yet.

### All 70 models

IC columns are ×1000. Net return per trade is after 10 bps, with the number of trades in brackets.
Model ids are the first eight characters; the full ids are in the `model` table and in
[§17](#17-appendix).

| Fit | Fit on | Model | Model id | Device | Fit time | Validation rows | IC ×1000 | se ×1000 | Jul | Aug | Sep | Oct | Calib. slope | Top-5 spread bps | Top-5 net bps | Net bps @0.1% (trades) | @0.2% | @0.5% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15m | to 2025-12-31 | ridge (alpha 1.0) | `21e7e71e` | cpu | 100 s | 3,217,182 | 52.6 | 0.8 | 55 | 56 | 51 | 46 | 0.76 | 5.5 | -7.7 | -9.4 (11,302) | -15.0 (2,517) | -15.7 (376) |
| 15m | to 2025-12-31 | xgboost d3 n300 | `cfd73f9a` | cpu | 729 s | 3,217,182 | 68.3 | 0.8 | 72 | 75 | 65 | 59 | 0.60 | 8.2 | -6.1 | -13.4 (26,065) | -13.1 (6,451) | +5.2 (304) |
| 15m | to 2025-12-31 | xgboost d3 n600 | `341d7909` | cpu | 1029 s | 3,217,182 | 69.8 | 0.9 | 72 | 76 | 67 | 61 | 0.50 | 8.7 | -5.7 | -13.6 (40,734) | -16.6 (12,765) | -5.8 (1,262) |
| 15m | to 2025-12-31 | xgboost d4 n300 | `74b85ff0` | cpu | 786 s | 3,217,182 | 69.4 | 0.9 | 72 | 76 | 67 | 60 | 0.46 | 8.5 | -5.9 | -14.6 (36,112) | -16.4 (10,618) | -5.6 (1,026) |
| 15m | to 2025-12-31 | xgboost d4 n600 | `36c058a3` | cpu | 1123 s | 3,217,182 | 70.3 | 0.9 | 73 | 77 | 68 | 61 | 0.31 | 8.4 | -5.9 | -13.5 (59,011) | -18.9 (19,998) | -12.1 (3,101) |
| 15m | 2026 H1 | ridge (alpha 1.0) | `63353122` | cpu | 26 s | 3,217,182 | 47.1 | 0.8 | 51 | 48 | 47 | 41 | 0.50 | 5.1 | -8.0 | -7.2 (6,863) | -4.3 (1,020) | +40.9 (95) |
| 15m | 2026 H1 | xgboost d3 n300 | `59907ce5` | cuda | 62 s | 3,217,182 | 60.3 | 0.8 | 70 | 70 | 45 | 54 | 0.20 | 5.7 | -7.1 | -8.7 (34,989) | -7.0 (5,288) | +16.8 (187) |
| 15m | 2026 H1 | xgboost d3 n600 | `26ae8ec3` | cuda | 72 s | 3,217,182 | 61.8 | 0.8 | 69 | 71 | 51 | 52 | 0.13 | 5.6 | -7.2 | -9.4 (82,168) | -10.9 (15,627) | +7.7 (852) |
| 15m | 2026 H1 | xgboost d4 n300 | `338e59a4` | cuda | 65 s | 3,217,182 | 55.7 | 0.9 | 66 | 68 | 40 | 45 | 0.10 | 5.5 | -7.1 | -9.1 (116,967) | -8.7 (21,555) | -1.8 (1,071) |
| 15m | 2026 H1 | xgboost d4 n600 | `12ab2b6e` | cuda | 76 s | 3,217,182 | 55.9 | 0.9 | 65 | 68 | 41 | 47 | 0.06 | 5.6 | -7.6 | -9.8 (192,607) | -9.7 (41,747) | -8.1 (2,224) |
| 60m | to 2025-12-31 | ridge (alpha 1.0) | `983d4d8f` | cpu | 429 s | 2,803,405 | 33.3 | 1.1 | 28 | 36 | 38 | 31 | 0.38 | 5.5 | -9.2 | -12.1 (69,340) | -12.1 (13,592) | -28.8 (1,118) |
| 60m | to 2025-12-31 | xgboost d3 n300 | `32629683` | cuda | 144 s | 2,803,405 | 34.3 | 1.0 | 28 | 40 | 34 | 35 | 0.32 | 6.7 | -8.6 | -13.6 (75,777) | -18.9 (16,849) | -30.3 (801) |
| 60m | to 2025-12-31 | xgboost d3 n600 | `25597948` | cuda | 187 s | 2,803,405 | 39.2 | 1.1 | 33 | 47 | 39 | 38 | 0.24 | 7.6 | -7.9 | -11.3 (134,272) | -10.8 (53,052) | -15.0 (17,825) |
| 60m | to 2025-12-31 | xgboost d4 n300 | `ddfbfd10` | cuda | 163 s | 2,803,405 | 37.0 | 1.0 | 31 | 48 | 33 | 36 | 0.17 | 6.5 | -8.3 | -10.8 (99,765) | -12.0 (43,220) | -20.0 (12,352) |
| 60m | to 2025-12-31 | xgboost d4 n600 | `eb174216` | cuda | 266 s | 2,803,405 | 40.3 | 1.1 | 34 | 50 | 39 | 39 | 0.16 | 7.8 | -7.9 | -10.6 (158,922) | -9.9 (69,995) | -11.1 (19,638) |
| 60m | 2026 H1 | ridge (alpha 1.0) | `85c2c91a` | cpu | 22 s | 2,803,405 | 17.4 | 0.8 | 22 | 16 | 20 | 8 | -0.00 | 3.1 | -10.3 | -4.5 (103,252) | -0.4 (15,874) | -3.8 (847) |
| 60m | 2026 H1 | xgboost d3 n300 | `d6942869` | cuda | 52 s | 2,803,405 | 7.1 | 0.9 | 19 | 3 | 9 | -7 | -0.00 | 3.6 | -8.1 | -6.4 (247,581) | -6.8 (70,452) | -3.1 (4,203) |
| 60m | 2026 H1 | xgboost d3 n600 | `eb89408b` | cuda | 61 s | 2,803,405 | 21.6 | 1.0 | 27 | 23 | 23 | 10 | -0.01 | 4.5 | -8.1 | -8.9 (339,707) | -5.9 (126,633) | -7.1 (10,606) |
| 60m | 2026 H1 | xgboost d4 n300 | `ec84b138` | cuda | 53 s | 2,803,405 | 16.1 | 1.0 | 20 | 15 | 15 | 14 | -0.02 | 3.2 | -8.5 | -9.6 (340,617) | -4.3 (114,618) | +3.6 (8,178) |
| 60m | 2026 H1 | xgboost d4 n600 | `82cd8019` | cuda | 64 s | 2,803,405 | 23.4 | 1.0 | 27 | 26 | 19 | 21 | -0.03 | 6.0 | -8.0 | -12.0 (443,151) | -8.2 (192,707) | +2.5 (22,804) |
| 375m | to 2025-12-31 | ridge (alpha 1.0) | `f03cfd08` | cpu | 109 s | 3,355,303 | 60.6 | 0.7 | 62 | 61 | 68 | 49 | 0.47 | 25.7 | -4.9 | -14.1 (481,743) | -12.2 (224,120) | -5.6 (48,644) |
| 375m | to 2025-12-31 | xgboost d3 n300 | `458a7372` | cpu | 769 s | 3,355,303 | 70.2 | 0.8 | 67 | 78 | 70 | 63 | 0.50 | 24.5 | -8.9 | +0.9 (357,319) | +6.0 (169,619) | +19.4 (34,254) |
| 375m | to 2025-12-31 | xgboost d3 n600 | `4255253b` | cpu | 1081 s | 3,355,303 | 75.7 | 0.9 | 76 | 84 | 75 | 64 | 0.41 | 23.7 | -9.3 | -0.2 (505,871) | +6.3 (255,513) | +20.8 (60,863) |
| 375m | to 2025-12-31 | xgboost d4 n300 | `756696d4` | cpu | 820 s | 3,355,303 | 68.8 | 0.9 | 66 | 80 | 67 | 61 | 0.60 | 24.3 | -7.9 | +5.4 (423,025) | +13.5 (197,929) | +30.8 (41,191) |
| 375m | to 2025-12-31 | xgboost d4 n600 | `52b8a073` | cpu | 1200 s | 3,355,303 | 71.1 | 0.9 | 70 | 85 | 69 | 55 | 0.46 | 21.4 | -8.4 | +1.5 (551,265) | +7.6 (274,508) | +20.4 (65,483) |
| 375m | 2026 H1 | ridge (alpha 1.0) | `a218dffa` | cpu | 29 s | 3,355,303 | 36.1 | 0.9 | 37 | 43 | 49 | 9 | 0.14 | 17.5 | -8.5 | -14.2 (852,373) | -16.3 (491,526) | -17.9 (159,001) |
| 375m | 2026 H1 | xgboost d3 n300 | `cc9577a6` | cuda | 66 s | 3,355,303 | 36.6 | 1.0 | 48 | 42 | 38 | 9 | 0.11 | 25.1 | -10.9 | -9.0 (920,593) | -9.0 (556,599) | -7.8 (171,059) |
| 375m | 2026 H1 | xgboost d3 n600 | `5d227264` | cuda | 78 s | 3,355,303 | 45.2 | 1.1 | 57 | 56 | 43 | 14 | 0.08 | 23.4 | -8.2 | -9.5 (1,036,884) | -8.1 (635,951) | -5.7 (198,541) |
| 375m | 2026 H1 | xgboost d4 n300 | `a74d3816` | cuda | 40 s | 3,355,303 | 45.2 | 1.1 | 61 | 54 | 43 | 12 | 0.10 | 30.6 | -8.8 | -13.6 (1,094,128) | -13.2 (694,854) | -7.3 (217,265) |
| 375m | 2026 H1 | xgboost d4 n600 | `da0aee06` | cuda | 53 s | 3,355,303 | 50.8 | 1.0 | 63 | 66 | 45 | 18 | 0.10 | 25.2 | -10.3 | -14.2 (1,147,004) | -13.6 (743,760) | -5.7 (229,952) |
| 1d nse | to 2025-12-31 | ridge (alpha 1.0) | `b1027df0` | cpu | 11 s | 127,843 | 26.6 | 5.8 | 22 | 25 | 44 | 11 | -0.01 | -85.2 | -82.4 | -62.8 (30,498) | -63.8 (28,974) | -66.3 (24,906) |
| 1d nse | to 2025-12-31 | xgboost d3 n300 | `dfe1c76b` | cuda | 8 s | 127,843 | 41.0 | 5.2 | 20 | 53 | 49 | 45 | 0.35 | -1.1 | -9.7 | -22.3 (3,844) | -22.6 (3,514) | -20.0 (2,909) |
| 1d nse | to 2025-12-31 | xgboost d3 n600 | `6707db7d` | cuda | 9 s | 127,843 | 29.7 | 5.5 | 10 | 40 | 44 | 25 | 0.30 | 3.3 | -3.5 | -10.6 (3,313) | -13.7 (2,870) | -13.7 (2,568) |
| 1d nse | to 2025-12-31 | xgboost d4 n300 | `cbbaee42` | cuda | 7 s | 127,843 | 30.5 | 5.3 | 17 | 34 | 40 | 32 | 0.49 | 15.8 | -1.9 | -9.1 (4,967) | -9.3 (4,758) | -9.4 (3,475) |
| 1d nse | to 2025-12-31 | xgboost d4 n600 | `d2a7deff` | cuda | 10 s | 127,843 | 49.2 | 7.0 | 29 | 59 | 67 | 41 | 0.28 | 25.3 | +3.4 | -18.8 (6,598) | -18.2 (5,749) | -21.6 (4,331) |
| 1d nse | 2026 H1 | ridge (alpha 1.0) | `da4ef7bf` | cpu | 2 s | 127,843 | 14.0 | 5.0 | 7 | 18 | 13 | 21 | 2.09 | 23.1 | -11.2 | -32.6 (42,190) | -30.5 (37,105) | -26.0 (24,392) |
| 1d nse | 2026 H1 | xgboost d3 n300 | `afbaeaf4` | cuda | 2 s | 127,843 | 38.7 | 4.6 | 24 | 42 | 59 | 26 | 1.60 | 8.1 | -8.8 | -39.0 (11,385) | -35.3 (5,861) | -39.5 (2,936) |
| 1d nse | 2026 H1 | xgboost d3 n600 | `bac2bac9` | cuda | 2 s | 127,843 | 37.4 | 4.1 | 36 | 35 | 43 | 34 | 1.33 | 8.2 | -12.0 | -34.4 (15,571) | -37.0 (9,260) | -41.8 (4,364) |
| 1d nse | 2026 H1 | xgboost d4 n300 | `f526cac5` | cuda | 2 s | 127,843 | 49.5 | 4.7 | 40 | 53 | 59 | 46 | 1.44 | 4.1 | -10.1 | -38.4 (17,601) | -47.5 (10,499) | -41.4 (3,470) |
| 1d nse | 2026 H1 | xgboost d4 n600 | `a736348c` | cuda | 3 s | 127,843 | 45.9 | 4.7 | 42 | 46 | 50 | 46 | 1.28 | 3.1 | -12.9 | -36.9 (22,354) | -42.9 (14,796) | -42.9 (5,526) |
| 1d us | to 2025-12-31 | ridge (alpha 1.0) | `66430298` | cpu | 22 s | 198,383 | 29.7 | 12.5 | 53 | 15 | -18 | 79 | 0.12 | 72.2 | -59.6 | -39.1 (4,882) | -55.5 (1,123) | -168.8 (115) |
| 1d us | to 2025-12-31 | xgboost d3 n300 | `ebb48a45` | cuda | 17 s | 198,383 | 17.2 | 9.3 | 6 | 8 | 7 | 57 | -0.25 | 26.8 | -27.2 | -42.9 (22,327) | -57.3 (10,238) | -142.1 (1,425) |
| 1d us | to 2025-12-31 | xgboost d3 n600 | `7bf08dcb` | cuda | 23 s | 198,383 | 35.2 | 12.0 | 56 | 11 | -5 | 91 | -0.13 | 69.4 | -6.4 | -27.9 (32,099) | -25.6 (16,793) | +3.4 (3,209) |
| 1d us | to 2025-12-31 | xgboost d4 n300 | `e515c69c` | cuda | 18 s | 198,383 | 17.7 | 12.2 | 12 | -13 | 16 | 67 | 0.29 | 52.6 | -17.1 | -17.7 (23,350) | -7.7 (7,074) | +182.7 (902) |
| 1d us | to 2025-12-31 | xgboost d4 n600 | `d2d1986f` | cuda | 24 s | 198,383 | 15.2 | 14.2 | 7 | -3 | -2 | 74 | 0.00 | 52.8 | -10.1 | -10.0 (41,209) | -6.9 (21,904) | -30.2 (2,593) |
| 1d us | 2026 H1 | ridge (alpha 1.0) | `484173d5` | cpu | 2 s | 198,383 | 10.8 | 13.0 | 32 | -26 | -25 | 78 | 0.13 | 16.6 | -37.9 | -20.1 (21,385) | -11.7 (10,688) | -8.5 (1,404) |
| 1d us | 2026 H1 | xgboost d3 n300 | `983d86d0` | cuda | 2 s | 198,383 | 4.2 | 14.7 | 35 | 16 | -53 | 20 | -0.09 | 8.1 | -13.1 | -25.9 (93,282) | -25.8 (75,856) | -26.0 (37,870) |
| 1d us | 2026 H1 | xgboost d3 n600 | `35cdaca5` | cuda | 3 s | 198,383 | 1.3 | 15.4 | 3 | 9 | -20 | 16 | -0.06 | 22.6 | -42.0 | -25.1 (90,086) | -25.3 (74,551) | -25.7 (39,577) |
| 1d us | 2026 H1 | xgboost d4 n300 | `5971b92a` | cuda | 2 s | 198,383 | -6.6 | 15.4 | -13 | -0 | -36 | 32 | -0.08 | -9.1 | -26.5 | -25.6 (91,473) | -26.8 (74,973) | -27.2 (38,720) |
| 1d us | 2026 H1 | xgboost d4 n600 | `2378d326` | cuda | 3 s | 198,383 | -8.0 | 15.4 | -26 | -9 | -14 | 26 | -0.07 | 3.4 | -34.4 | -26.6 (86,003) | -27.8 (71,188) | -29.0 (39,146) |
| 21d nse | to 2025-12-31 | ridge (alpha 1.0) | `7997866e` | cpu | 8 s | 94,416 | -107.1 | 10.1 | -18 | -163 | -165 | – | 2.34 | -745.0 | -270.1 | -288.0 (59,181) | -289.9 (58,558) | -294.8 (56,638) |
| 21d nse | to 2025-12-31 | xgboost d3 n300 | `3bdd52fa` | cuda | 6 s | 94,416 | 57.7 | 4.3 | 41 | 86 | 43 | – | 0.28 | -2.4 | -240.6 | -59.6 (33,494) | -88.9 (17,517) | -318.0 (6,504) |
| 21d nse | to 2025-12-31 | xgboost d3 n600 | `7e35ab6a` | cuda | 8 s | 94,416 | 16.4 | 3.0 | 17 | 25 | 3 | – | 0.21 | -115.1 | -454.0 | -156.9 (68,302) | -144.3 (64,400) | -133.4 (50,554) |
| 21d nse | to 2025-12-31 | xgboost d4 n300 | `c255b81f` | cuda | 7 s | 94,416 | 61.3 | 5.3 | 51 | 89 | 39 | – | 0.16 | -226.3 | -561.3 | -151.7 (77,236) | -139.4 (74,968) | -104.5 (63,830) |
| 21d nse | to 2025-12-31 | xgboost d4 n600 | `aaaacdfe` | cuda | 9 s | 94,416 | 27.5 | 6.0 | 42 | 41 | -14 | – | 0.06 | -159.4 | -449.4 | -146.5 (72,671) | -142.9 (70,949) | -125.2 (63,624) |
| 21d nse | 2026 H1 | ridge (alpha 1.0) | `af7c24ac` | cpu | 2 s | 94,416 | 3.7 | 5.0 | 31 | 3 | -36 | – | 0.00 | -308.7 | -657.3 | -189.7 (40,846) | -189.7 (40,846) | -189.7 (40,846) |
| 21d nse | 2026 H1 | xgboost d3 n300 | `6d2c7184` | cuda | 2 s | 94,416 | -34.0 | 6.4 | 14 | -48 | -88 | – | -0.00 | 12.4 | -239.4 | -293.9 (759) | -293.9 (759) | -293.9 (759) |
| 21d nse | 2026 H1 | xgboost d3 n600 | `993e2334` | cuda | 2 s | 94,416 | -70.4 | 6.5 | -22 | -84 | -125 | – | -0.00 | 44.7 | -345.0 | -200.6 (88,508) | -200.6 (88,508) | -200.6 (88,508) |
| 21d nse | 2026 H1 | xgboost d4 n300 | `2f549654` | cuda | 3 s | 94,416 | -73.4 | 7.4 | -18 | -88 | -137 | – | -0.00 | -0.3 | -331.7 | -197.0 (82,639) | -197.0 (82,639) | -197.0 (82,639) |
| 21d nse | 2026 H1 | xgboost d4 n600 | `3f55bbd5` | cuda | 2 s | 94,416 | -98.9 | 7.4 | -43 | -114 | -163 | – | -0.00 | -61.8 | -311.3 | -208.8 (85,167) | -208.8 (85,167) | -208.8 (85,167) |
| 21d us | to 2025-12-31 | ridge (alpha 1.0) | `749c2263` | cpu | 23 s | 148,473 | 68.5 | 12.7 | 44 | 20 | 159 | – | 0.54 | 774.2 | +18.8 | +56.2 (9,448) | +59.6 (7,473) | +105.5 (3,339) |
| 21d us | to 2025-12-31 | xgboost d3 n300 | `daae022f` | cuda | 17 s | 148,473 | 86.7 | 9.9 | 85 | 38 | 149 | – | 1.36 | 213.1 | +95.6 | -56.8 (71,225) | -35.2 (65,199) | +33.8 (50,638) |
| 21d us | to 2025-12-31 | xgboost d3 n600 | `624630a5` | cuda | 23 s | 148,473 | 100.4 | 8.2 | 102 | 65 | 142 | – | 1.13 | 221.1 | +202.8 | -57.5 (87,251) | -46.6 (83,414) | -6.2 (71,457) |
| 21d us | to 2025-12-31 | xgboost d4 n300 | `1b7db6fa` | cuda | 18 s | 148,473 | 91.9 | 10.6 | 116 | 32 | 134 | – | 1.07 | 427.6 | +345.3 | -45.2 (80,894) | -35.5 (77,132) | +20.0 (64,531) |
| 21d us | to 2025-12-31 | xgboost d4 n600 | `111ff633` | cuda | 24 s | 148,473 | 97.5 | 9.8 | 130 | 57 | 106 | – | 0.84 | 301.3 | +269.2 | -73.2 (90,697) | -60.9 (87,882) | -26.7 (79,512) |
| 21d us | 2026 H1 | ridge (alpha 1.0) | `814373d4` | cpu | 2 s | 148,473 | -12.0 | 15.1 | -46 | -67 | 100 | – | 0.53 | 1742.5 | -237.7 | -180.5 (126,009) | -178.0 (124,663) | -169.3 (119,581) |
| 21d us | 2026 H1 | xgboost d3 n300 | `8feade47` | cuda | 2 s | 148,473 | -41.5 | 18.8 | 2 | -121 | -0 | – | -0.27 | 321.2 | -169.0 | -189.0 (91,682) | -189.4 (88,941) | -192.1 (80,720) |
| 21d us | 2026 H1 | xgboost d3 n600 | `80e66055` | cuda | 3 s | 148,473 | -35.0 | 18.5 | 26 | -118 | -12 | – | -0.23 | 249.6 | -101.6 | -189.7 (90,906) | -189.8 (88,556) | -192.6 (80,899) |
| 21d us | 2026 H1 | xgboost d4 n300 | `897d964d` | cuda | 2 s | 148,473 | -25.0 | 17.7 | 32 | -100 | -5 | – | -0.16 | 304.0 | -159.0 | -165.4 (87,030) | -165.5 (84,818) | -164.5 (77,830) |
| 21d us | 2026 H1 | xgboost d4 n600 | `43e6c4bd` | cuda | 3 s | 148,473 | -15.2 | 17.9 | 58 | -97 | -10 | – | -0.14 | 172.3 | -140.7 | -174.1 (85,351) | -174.6 (83,116) | -177.5 (75,908) |

## 9. Analysis A: the session model as a daily ranking

**Question.** The threshold and top-5 metrics are long-only. Does the session model's ranking pay
when used as a cross-section, once a day?

**Model.** `forecast 375m r2`, fit through 2025-12-31, xgboost d3 n600, model
`4255253b-dbbc-4ba5-bf91-8721ffca573c` (FitRun `51121bd8-f9b8-4ced-ba48-ec730d2cfe82`, fitted on the
CPU in 1,081 s). Inference table:
`parquet/inference/8402fb7e-1b51-46e9-ad16-1eb2563e3278.parquet` (3,355,303 rows, 2026-07-01 to
2026-10-23).

**Method.** At one decision time per day (09:30, 11:00 or 13:00), sort that minute's stocks
(about 450) into ten equal groups by prediction. For each day, record the mean realised return `r`
(next minute's open to the close) of each group. The long-short day return is the top group minus
the bottom group. Statistics are over the 80 test days, one observation per day, so there is no
overlap. Query: [Appendix A3](#a3-analysis-a-deciles-and-long-short).

**Return from 11:00 to the close by prediction group, mean over 80 days:**

| Group (1 = lowest prediction) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean return, bps | −24.4 | −19.0 | −14.6 | −7.9 | −12.9 | −8.5 | −8.3 | −5.5 | −4.5 | +0.7 |
| Stocks that rose, % | 37.9 | 40.0 | 41.4 | 43.1 | 40.9 | 43.1 | 44.0 | 44.2 | 45.2 | 46.4 |

The average stock lost 10.5 bps from 11:00 to the close over this period.

**Trades built from it:**

| Decision time | Trade | Gross per day | t-stat | Days positive | Net per day | Net over 80 days | Max drawdown | Sharpe (annualised) |
|---|---|---|---|---|---|---|---|---|
| 09:30 | Long top 10 %, short bottom 10 % | 22.5 bps | 4.8 | 68 % | 12.5 bps | +10.5 % | −1.2 % | 4.7 |
| 09:30 | Short bottom 10 % | 24.4 bps | 3.1 | 60 % | 19.4 bps | +16.8 % | −2.6 % | 4.3 |
| **11:00** | **Long top 10 %, short bottom 10 %** | **25.1 bps** | **7.6** | **81 %** | **15.1 bps** | **+12.9 %** | **−0.6 %** | **8.0** |
| 11:00 | Short bottom 10 % | 24.4 bps | 4.3 | 68 % | 19.4 bps | +16.8 % | −1.6 % | 6.0 |
| 11:00 | Top 10 % minus all stocks | 11.2 bps | 5.1 | – | – | – | – | – |
| 13:00 | Long top 10 %, short bottom 10 % | 15.5 bps | 5.7 | 76 % | 5.5 bps | +4.5 % | −1.4 % | 3.5 |
| 13:00 | Short bottom 10 % | 14.2 bps | 2.9 | 70 % | 9.2 bps | +7.6 % | −2.8 % | 3.3 |

Costs: 5 bps per leg (10 bps per long-short pair). Break-even cost at 11:00: 25 bps per pair.

**The 15-minute model, same method** (model `36c058a3-c202-4c95-9aa5-180e6123d49c`, inference
`2677462a-9922-4164-b52b-379cade99750`, all 5,680 decision points): top-minus-bottom decile
5.0 bps per 15 minutes (t = 47, positive at 77 % of points), deciles rising monotonically from
−2.9 to +1.9 bps. The ranking is real and precise, but 5 bps is half of one round trip, so it does
not trade on its own.

**Reading.** The session model is good at saying which stocks will do worse than others for the
rest of the day, and weak at saying which will rise. Over this period the whole market drifted down
intraday, so the short side carried the profit and the long side only beat the average. The
long-short pair removes the market's direction and is the cleanest form. Its edge is two and a half
times the assumed cost.

**What this does not yet show.** 80 days is one regime. The test assumes fills at the next minute's
open with no slippage. Some bottom-group stocks may be barred from intraday shorting on a given day
(surveillance lists). Paper trading with real fills is the next step ([§16](#16-next-steps)).

## 10. Analysis B: the session study, overnight and daytime

**Question.** If you buy near the close and sell after the next open, which stocks, indices and
commodities paid, how consistently, and what about shorting the daytime instead, or both?

**Reaction and query.** `SessionStudy` ([reactions.py#L699](../apps/finance/src/finance/ml/reactions.py#L699)),
query [`study.STUDY_SQL`](../apps/finance/src/finance/ml/study.py#L54) with market definitions in
[`study.MARKETS`](../apps/finance/src/finance/ml/study.py#L31) and defaults in
[`study.DEFAULTS`](../apps/finance/src/finance/ml/study.py#L96). Per instrument and year:

    overnight  = ln(next session's open / close)
    short_day  = −ln(close / open)
    combined   = overnight + short_day
    buy_hold   = ln(next close / close)

Sessions more than 5 days apart are dropped; MCX pairs that straddle a contract roll are dropped.

| Run | Market | Liquidity floor | History | Rows | Output |
|---|---|---|---|---|---|
| `19cf4e17-d92b-41fb-8c36-d0940afa58d0` | NSE stocks and 8 indices | ₹5 crore/day median since 2024 | 2021-09 → | 5,339 | `parquet/studies/session_nse_9977454ed15c44b49f1e49d611fc0f23.parquet` |
| `2772b4ac-60a6-4867-ba91-e8ee0210f794` | US stocks, SPY/QQQ/IWM | $5 million/day | 2015-01 → | 26,440 | `parquet/studies/session_us_dc83dc4c437e4c449802c8f97917c8be.parquet` |
| `02d05e30-959a-4d56-9435-a5ce8077abf3` | MCX front-month futures | 1,000 lots/day | 2021-09 → | 93 | `parquet/studies/session_mcx_86b2f02c792c47e8abaa1bda7649266a.parquet` |

**The NSE daily open is not a fill.** The official open is the pre-open auction price. For ETFs and
thin stocks it sits about 1 % above the previous close almost daily and is gone within minutes. On
daily data NIFTYBEES shows +70 % a year overnight; on minute bars at 15:14 and 09:29 it shows +8 %
(2024), +9 % (2025), −8 % (2026). Every NSE conclusion in this document therefore uses minute bars
(Analyses A, C and D). The daily-table NSE file is kept for reference only. US opens on large
names are close to tradeable; for small US names subtract 5 to 10 bps a session.

**NSE on minute bars, median liquid stock, yearly total of daily returns:**

| Year | Overnight (15:14 → 09:29) | Stocks positive | Daytime short (09:29 → 15:14) | Stocks positive | Both | Buy and hold |
|---|---|---|---|---|---|---|
| 2024 | +50 % | 95 % | +26 % | 78 % | +77 % | +21 % |
| 2025 | +13 % | 66 % | +6 % | 58 % | +19 % | 0 % |
| 2026 to 22 Oct | +3 % | 57 % | +2 % | 55 % | +7 % | −2 % |

**NSE indices on minute bars, % per year:**

| Index | 2024 night | 2025 night | 2026 night | 2024 day short | 2025 day short | 2026 day short | 2026 buy and hold |
|---|---|---|---|---|---|---|---|
| Nifty 50 | +15 | +8 | −3 | +7 | −2 | +12 | −15 |
| Nifty Bank | +10 | +9 | −4 | +5 | −7 | +5 | −10 |
| Nifty Next 50 | +46 | +18 | −2 | +22 | +16 | +2 | −6 |
| Nifty Midcap 150 | +40 | +11 | −1 | +19 | +6 | +5 | −8 |
| Nifty IT | +32 | −7 | +5 | +12 | +6 | +20 | −18 |
| Nifty PSU Bank | +45 | +36 | +14 | +32 | +9 | +18 | −6 |

**US, official open and close, % per year:**

| Year | SPY night | SPY day short | SPY both | SPY buy and hold | QQQ night | IWM night | Median stock night | Median stock day short |
|---|---|---|---|---|---|---|---|---|
| 2021 | +15 | −9 | +6 | +26 | +15 | +30 | +24 | +8 |
| 2022 | −16 | +6 | −10 | −23 | −24 | −10 | −10 | +7 |
| 2023 | +3 | −18 | −15 | +22 | +8 | +7 | +6 | −6 |
| 2024 | +21 | −1 | +21 | +21 | +29 | +28 | +19 | +9 |
| 2025 | +7 | −8 | −2 | +16 | +15 | +2 | +2 | −3 |
| 2026 | +7 | −3 | +5 | +11 | +8 | +8 | +7 | +5 |

US stocks positive overnight every year 2021 to 2026 (price ≥ $20, ≥ $20 million a day): YPF, CCJ,
WFRD, TDW, MOD, AVAV, STRL (worst year +18 to +33 %). Daytime short: MP, FUBO, OUST.

**MCX, % per year:**

| Contract | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| GOLD night | +3 | +7 | +7 | +12 | +8 |
| GOLD day short | −7 | −5 | −8 | −35 | +2 |
| SILVER night | −5 | +9 | +11 | +14 | +12 |
| CRUDEOIL night | +4 | −5 | +3 | −1 | +46 |
| NATURALGAS day short | +7 | +84 | +41 | +31 | +15 |

Gold's night is 1 to 3 bps a session against 2 to 3 bps of cost. Natural gas's daytime short tracks
its three-year price fall. Nothing here is a timing edge.

## 11. Analysis C: the window study

**Question.** The session study tests one pair of times. Which pair of times, within one session or
into the next, paid for each stock, and are weekend and holiday gaps different from ordinary nights?

**Reaction and query.** `WindowStudy` run `6a50eebf-09bc-43e9-b402-2f2feb6c604e` (2026-10-25 01:21
IST, 3.6 minutes), query [`windows.STUDY_SQL`](../apps/finance/src/finance/ml/windows.py#L31) run
in 10 symbol buckets ([`BUCKETS`](../apps/finance/src/finance/ml/windows.py#L89)). Output:
`parquet/studies/windows_nse_7dc320f68e1d4342a1a79c8578f824db.parquet`, 3,122,266 rows.

- **Grid:** the close of the 09:29 bar, then every 15 minutes to 15:14, plus 15:29; 25 points.
  The 09:15 auction print is excluded.
- **Windows:** 300 same-day pairs (entry before exit) and 625 entry-today, exit-next-session pairs.
  The next-session pairs are labelled `night` (next calendar day) or `break` (a weekend or holiday
  between).
- **Universe:** 576 stocks at ₹5 crore a day, plus 27 NSE indices.
- **Per stock, window and year:** sessions, total, mean, hit rate, standard deviation; a
  window-year needs 20 sessions.

**Market-wide, median stock, yearly total:**

| Window | 2023 (Oct–Dec) | 2024 | 2025 | 2026 | Reading |
|---|---|---|---|---|---|
| Short 09:29 → 13:14 | +6 % | +21 % | +10 % | +8 % | Mornings fall; about 10 bps a day in 2023–24, 4 since |
| Long 14:14 → 09:29 next session | +16 % | +39 % | +19 % | +14 % | Strongest cross-session window for the median stock |
| Best same-day long windows | 0 to 5 % | 0 to 5 % | 0 to 5 % | 0 to 5 % | Nothing inside the day is reliably positive |

**Per stock, best window chosen on 2024 and 2025, scored on 2026:**

| Family | Stocks | 2026 mean / median / % positive | All windows that year (median, % positive) | Top 20 picks in 2026 |
|---|---|---|---|---|
| Same day, long | 515 | +1.4 % / +1.1 % / 52 % | −2.2 %, 42 % | +8.6 %, 60 % |
| Same day, short | 517 | +7.7 % / +5.2 % / 65 % | +2.2 %, 58 % | +33 %, 80 % |
| Next session, ordinary nights, long | 501 | +14.7 % / +13.1 % / 73 % | +6.9 %, 62 % | +29 %, 75 % |
| Across breaks, long | 322 | −10.7 % / −10.8 % / 19 % | −14.0 %, 18 % | −8.6 %, 35 % |
| Across breaks, short | 370 | +16.1 % / +15.1 % / 82 % | +14.0 %, 82 % | +20 %, 75 % |

These are sums of daily log returns per stock, not portfolio returns, and each stock had 1,250
windows to choose from, so they overstate what a trader would get. [§12](#12-analysis-d-picked-stock-portfolios-and-weekends-out-of-sample)
measures fixed windows as real portfolios.

## 12. Analysis D: picked-stock portfolios and weekends, out of sample

**Question.** Measured as an equal-weight portfolio traded every day, with costs, do the stocks
picked on 2024 and 2025 earn in 2026, and do weekends differ from ordinary nights?

**Data.** A per-stock, per-day table of closes at 09:29, 14:14, 14:29, 15:14 and 15:29 and the next
session's 09:29, built from `equity_1m` (418,156 stock-days, 602 symbols, 2023-10-03 to
2026-10-22). Query: [Appendix A1](#a1-analysis-d-the-per-day-window-table). Statistics:
[Appendix A2](#a2-analysis-d-portfolio-statistics).

**Method.** For each window, rank stocks with at least 120 sessions in both 2024 and 2025 by the
lower of their two yearly totals, keep those positive in both, take the top 20. Each day of 2026,
hold all 20 in equal amounts. Daily return = mean across the 20; costs subtracted per day.
t-statistics are over days. The same method picked on 2024 alone is tested on 2025 as a second
out-of-sample year.

**Results, 2026-01-01 to 2026-10-22:**

| Trade | Days | Gross per day | Cost | Net per day | t (net) | Win days | ₹1 lakh became | Max drawdown | Sharpe | Up months |
|---|---|---|---|---|---|---|---|---|---|---|
| Overnight 15:14 → 09:29, picked 20, futures | 199 | 21.0 bps | 4 | 17.0 bps | 1.94 | 56.8 % | ₹1,40,252 | −9.1 % | 2.16 | 8 of 10 |
| Same, cash delivery | 199 | 21.0 bps | 25 | −4.0 bps | −0.46 | 50.3 % | ₹92,346 | −18.9 % | −0.51 | 5 of 10 |
| Same, all stocks, no cost | 199 | 1.8 bps | 0 | 1.8 bps | 0.28 | 54.3 % | ₹1,03,596 | −10.8 % | 0.31 | 5 of 10 |
| Overnight 14:14 → 09:29, picked 20, futures | 199 | 11.3 bps | 4 | 7.3 bps | 0.82 | 53.8 % | ₹1,15,624 | −8.3 % | 0.91 | 6 of 10 |
| Daytime short 09:29 → 14:29, picked 20, intraday cash | 199 | 8.7 bps | 5 | 3.7 bps | 0.52 | 49.7 % | ₹1,07,603 | −15.0 % | 0.58 | 6 of 10 |
| Same, all stocks, no cost | 199 | 3.7 bps | 0 | 3.7 bps | 0.82 | 49.7 % | ₹1,07,653 | −7.4 % | 0.91 | 7 of 10 |

**Second test year (picked on 2024, traded in 2025, gross):**

| Trade | Picked 20 | All stocks |
|---|---|---|
| Overnight 15:14 → 09:29 | 17.7 bps a night, t = 2.2, 63 % win nights, +55 % for the year | 4.9 bps, t = 1.1, +13 % |
| Overnight 14:14 → 09:29 | 16.1 bps, t = 2.0 | 9.3 bps |
| Daytime short 09:29 → 14:29 | 12.8 bps, t = 1.9, +37 % | 8.4 bps, t = 1.9, +23 % |

**Monthly net return, overnight picked 20 through futures, 2026:** Jan +0.4 %, Feb +4.9 %,
Mar +6.1 %, Apr +10.7 %, May +8.5 %, Jun +8.2 %, Jul +1.4 %, Aug +2.0 %, Sep −0.5 %, Oct −6.2 %.

**The 20 overnight picks:**

| Stock | ₹ crore/day | 2024 | 2025 | 2026 | bps/night 2026 | Win nights 2026 |
|---|---|---|---|---|---|---|
| CUPID | 24 | +111 % | +78 % | +96.9 % | 48.7 | 63.3 % |
| NETWEB | 84 | +167 % | +87 % | +80.3 % | 40.4 | 56.3 % |
| APOLLO | 71 | +138 % | +81 % | +77.4 % | 38.9 | 56.3 % |
| AVALON | 19 | +68 % | +67 % | +65.7 % | 33.0 | 51.3 % |
| VIYASH | 14 | +81 % | +70 % | +60.3 % | 30.3 | 56.3 % |
| SYRMA | 60 | +86 % | +79 % | +53.6 % | 26.9 | 57.8 % |
| PARAS | 35 | +92 % | +77 % | +53.5 % | 26.9 | 57.3 % |
| ASTRAMICRO | 29 | +104 % | +70 % | +53.4 % | 26.8 | 55.3 % |
| TDPOWERSYS | 34 | +127 % | +99 % | +50.4 % | 25.3 | 55.8 % |
| LLOYDSENGG | 30 | +133 % | +79 % | +49.5 % | 24.9 | 53.8 % |
| GRSE | 177 | +147 % | +67 % | +32.1 % | 16.1 | 51.3 % |
| HBLENGINE | 80 | +146 % | +81 % | +29.4 % | 14.8 | 56.3 % |
| BSE | 974 | +95 % | +87 % | +25.7 % | 12.9 | 51.8 % |
| POCL | 16 | +186 % | +98 % | +25.6 % | 12.9 | 51.3 % |
| FORCEMOT | 88 | +102 % | +86 % | +24.0 % | 12.1 | 51.8 % |
| SAMMAANCAP | 112 | +97 % | +67 % | +15.5 % | 7.8 | 49.7 % |
| POWERINDIA | 154 | +83 % | +80 % | +14.7 % | 7.4 | 50.3 % |
| ASHAPURMIN | 17 | +80 % | +69 % | +12.7 % | 6.4 | 55.3 % |
| SMLMAH | 12 | +91 % | +97 % | +6.9 % | 3.5 | 54.4 % |
| PFOCUS | 6 | +84 % | +121 % | +5.6 % | 2.9 | 48.0 % |

All 20 were positive in 2026. Yearly figures are sums of nightly log returns, gross.

**The 20 daytime-short picks** (09:29 → 14:29), 2026 total, gross: RAILTEL +53.6 %, OLECTRA
+51.1 %, INOXWIND +46.4 %, JWL +38.2 %, ZEEL +37.2 %, HCC +34.9 %, SWSOLAR +32.8 %, SJVN +31.9 %,
IREDA +30.8 %, NLCINDIA +30.5 %, PFC +22.3 %, SAMMAANCAP +19.5 %, KAYNES +10.6 %, ANGELONE +4.3 %,
KALYANKJIL +2.8 %, JSWENERGY −1.2 %, PAISALO −5.4 %, NIACL −9.3 %, TDPOWERSYS −29.4 %, JINDALSAW
−55.9 %. Fifteen of twenty positive, but two large losers carry the portfolio to the market's
result.

**Weekends and holidays.** All liquid stocks, equal weight, one observation per session:

| Window | Year | Ordinary nights: sessions, mean, t, % up | Breaks: sessions, mean, t, % up |
|---|---|---|---|
| 15:14 → 09:29 | 2024 | 188, +25.7 bps, 5.66, 80 % | 61, +15.3 bps, 1.42, 67 % |
| 15:14 → 09:29 | 2025 | 188, +6.3 bps, 1.41, 61 % | 59, +0.3 bps, 0.02, 56 % |
| 15:14 → 09:29 | 2026 | 150, +8.5 bps, 1.37, 59 % | 49, −18.9 bps, −1.12, 39 % |
| 14:14 → 09:29 | 2024 | 188, +22.8 bps, 4.26, 73 % | 61, +8.9 bps, 0.76, 64 % |
| 14:14 → 09:29 | 2025 | 188, +11.8 bps, 2.34, 64 % | 59, +1.3 bps, 0.10, 54 % |
| 14:14 → 09:29 | 2026 | 150, +10.9 bps, 1.54, 56 % | 49, −21.8 bps, −1.20, 45 % |

Breaks were worse than ordinary nights in every year and both windows, by 10 to 33 bps. No single
year is significant on its own (about 50 breaks a year), but the direction has held six times out
of six.

## 13. Costs

| Trade type | Round trip used | Components |
|---|---|---|
| Intraday cash, long or short (MIS) | 5 bps per leg | STT 0.025 % on the sell; exchange, SEBI and stamp fees; brokerage; small slippage allowance |
| Overnight cash delivery (CNC) | 25 bps | STT 0.1 % on both the buy and the sell; exchange and stamp fees; depository charge |
| Stock or index futures | 4 bps | STT 0.02 % on the sell; exchange fees. One lot is ₹5 to 15 lakh notional, about ₹1 to 3 lakh margin |
| Model metrics (`Evaluate`) | 10 bps | One round trip, any instrument |
| US equities | about 0 to 2 bps | No STT; SEC fee on sales. Daytime shorts every day need a margin account above the $25,000 pattern-day-trader floor |

Slippage beyond these allowances is not modelled anywhere in this document.

## 14. Limitations

- **Short test windows.** The models are tested on 80 trading days; the picked-stock portfolios on
  199. Regimes change: the intraday market drift was negative throughout the model test window.
- **Fills.** Every result assumes trading at the next bar's open (models) or at a bar's close
  (studies), with no market impact. Several picked stocks trade under ₹20 crore a day.
- **Shortability.** Intraday shorts need the stock on the broker's MIS list that day. Stocks in
  NSE's surveillance frameworks are often excluded, and small caps that ran up are frequent
  members. Not modelled.
- **Futures availability.** Most overnight picks have no stock futures, which is the only route at
  which the overnight trade clears costs.
- **Overlapping targets.** For the 15-minute, 60-minute and session models, decision points every 5
  minutes overlap in their targets, so the reported IC standard errors are too small. Analysis A
  avoids this by using one decision per day.
- **Selection.** The window study chose the best of 1,250 windows per stock; its per-stock results
  overstate. Analysis D fixes the window in advance.
- **Unadjusted NSE prices.** A split or bonus inside a hold would appear as a large return. Holds
  here are at most one night; affected stock-days are rare but not removed.

## 15. Corrections to earlier statements

| Earlier statement | Corrected by | Correct statement |
|---|---|---|
| The overnight trade made +50 % to +70 % a year on ETFs (daily-table study) | Minute bars, [§10](#10-analysis-b-the-session-study-overnight-and-daytime) | The daily open is the auction print; on minute bars ETFs earn the index's own overnight drift |
| The daytime short on picked stocks works in cash at about +20 % a year | Portfolio test, [§12](#12-analysis-d-picked-stock-portfolios-and-weekends-out-of-sample) | +7.6 % net in 2026, t = 0.5, no better than shorting every stock |
| Buy at 14:14, not at the close | Portfolio test, [§12](#12-analysis-d-picked-stock-portfolios-and-weekends-out-of-sample) | True for the median stock; for picked stocks 15:14 earned more (21.0 vs 11.3 bps) |
| The session model's trades lose money | Analysis A, [§9](#9-analysis-a-the-session-model-as-a-daily-ranking) | Long-only loses; the long-short ranking at one decision a day is the strongest result |
| Top 20 picks keep about +40 % a year out of sample | Portfolio test, [§12](#12-analysis-d-picked-stock-portfolios-and-weekends-out-of-sample) | Overnight picks: +52 % gross, +40 % after futures costs in 2026; the earlier figure averaged per-stock totals |

## 16. Next steps

1. **Paper-trade Analysis A.** Each day at 11:00, record the model's top and bottom 10 %, whether
   each bottom stock is allowed for intraday shorting, and the actual fill prices; close at 15:20.
   Compare with the backtest after 20 days.
2. **Stress Analysis A.** Re-run with 5 to 10 bps slippage per leg, with only shortable stocks, and
   on 2026-01 to 2026-06 using the model fit through 2025 (a second test period).
3. **Rescale the trees** by their calibration slope and re-evaluate the thresholds.
4. **An overnight Fit.** Target 15:14 → 09:29 for both markets, with the stock's trailing overnight
   return, trailing morning-short return and the break flag as features.
5. **Make Analyses A and D reactions.** They were run as one-off queries ([Appendix](#17-appendix));
   they should become a `PortfolioStudy` reaction so they rerun on new data.
6. **Drop the 2026-only training split.** Keep training through the latest full year.

## 17. Appendix

### Runs

| Run id | Reaction | Started (UTC) | Duration | Input | Output |
|---|---|---|---|---|---|
| `fff1066a-1b93-43ba-8902-7c8535f608f8` | FitSweep | 2026-10-24 17:12 | 63 min | 14 runs: 7 Fits × {end 2025-12-31; start 2026-01-01, end 2026-06-30} | 14 FitRuns, 70 Models |
| `17618ca3-0030-49ab-bab2-965b49261611` | RetryModels | 2026-10-24 20:11 | – | FitRun `20c2fb75…` | 5 models requeued |
| `d5abb61f-5956-4869-9a4d-b936e85fca45` | RetryModels | 2026-10-24 20:11 | – | FitRun `51121bd8…` | 2 models requeued |
| `89368513-07e4-4ec9-be4a-d28c3a77ee6b` | RetryModels | 2026-10-24 20:11 | – | FitRun `c04e452a…` | 1 model requeued |
| `bbd65783-fcd8-4672-8ea8-f8ccf38c1a8d` | RetryModels | 2026-10-24 20:12 | – | model `cc9577a6…` | 1 model requeued |
| `e7a3937e-eaa4-4570-b505-afa7f0a23d55` | ValidateSweep | 2026-10-24 22:43 | 125 min | label `validation 2026 H2`, 2026-07-01 → 2026-10-23, thresholds 0.001/0.002/0.005, cost 10, top_k 5 | 70 InferRuns, 210 Evaluations |
| `19cf4e17-d92b-41fb-8c36-d0940afa58d0` | SessionStudy | 2026-10-24 16:51 | 1 min | `{"market": "nse"}` | session_nse_9977…parquet |
| `2772b4ac-60a6-4867-ba91-e8ee0210f794` | SessionStudy | 2026-10-24 16:52 | 9 min | `{"market": "us"}` | session_us_dc83…parquet |
| `02d05e30-959a-4d56-9435-a5ce8077abf3` | SessionStudy | 2026-10-24 17:06 | 6 min | `{"market": "mcx"}` | session_mcx_86b2…parquet |
| `6a50eebf-09bc-43e9-b402-2f2feb6c604e` | WindowStudy | 2026-10-24 19:51 | 4 min | defaults | windows_nse_7dc3…parquet |

Find any of them in the admin console, or with:

```sql
select reaction, outcome, input, vars from _run where id = '<run id>';
```

### Model, inference and evaluation ids

```sql
select f.name, fr.params->>'end' as fit_end, m.id as model_id, m.algorithm, m.params,
       ir.id as infer_run_id, ir.path, e.id as evaluation_id, e.params->>'threshold' as threshold
from model m
join fit_run fr on fr.id::text = m.fit_run_id::text
join fit f on f.id::text = fr.fit_id::text
join infer_run ir on ir.model_id::text = m.id::text and ir.label = 'validation 2026 H2'
join evaluation e on e.infer_run_id::text = ir.id::text
where f.name like '%r2'
order by f.name, fit_end, m.algorithm;
```

### Reproduce

```bash
cd ~/Workspaces/backend-builder-monorepo/apps/finance
set -a; . ~/.config/backend-builder/finance.env; set +a
uv run backend-builder trigger finance finance.ml.reactions.SessionStudy --input '{"market": "nse"}'
uv run backend-builder trigger finance finance.ml.reactions.WindowStudy --input '{}'
uv run backend-builder trigger finance finance.ml.reactions.ValidateSweep --input '{"fit_run_ids": [...], "label": "...", "params": {"start": "2026-07-01", "end": "2026-10-23"}, "thresholds": [0.001, 0.002, 0.005], "cost_bps": 10, "top_k": 5}'
```

The three analyses below were run as one-off queries against the mirror and the inference tables,
not as reactions. They are reproduced here as run (A3 with its formatting tidied) so they can be
rerun or turned into a reaction.

### A1. Analysis D: the per-day window table

Runs on a DuckDB connection from `finance.ml.data.connect(<mirror>)`.

```sql
create table g as
with b as (
  select symbol, ts::date d, strftime(ts,'%H:%M') t, close from equity_1m
  where strftime(ts,'%H:%M') in ('09:29','14:14','14:29','15:14','15:29')
  qualify row_number() over (partition by symbol, ts order by close) = 1
),
day as (select symbol, d,
  max(case when t='09:29' then close end) c0929, max(case when t='14:14' then close end) c1414,
  max(case when t='14:29' then close end) c1429, max(case when t='15:14' then close end) c1514,
  max(case when t='15:29' then close end) c1529 from b group by 1,2),
liq as (select symbol, median(tv) mt from (select symbol, ts::date d, sum(close*volume) tv
        from equity_1m where ts >= '2024-01-01' group by 1,2) group by 1)
select day.*, liq.mt, lead(d) over w nd, lead(c0929) over w n0929
from day join liq using(symbol)
window w as (partition by symbol order by d);

create table m as
select symbol, d, year(d) y, month(d) mo, mt, symbol like 'NIFTY %' is_index,
  case when nd - d = 1 then 'night' else 'break' end across,
  ln(n0929/c1514) on1514, ln(n0929/c1414) on1414,
  -ln(c1429/c0929) sd0929_1429, -ln(c1514/c0929) sd0929_1514
from g
where nd is not null and nd - d <= 5
  and c0929>0 and c1414>0 and c1429>0 and c1514>0 and n0929>0;
```

### A2. Analysis D: portfolio statistics

Python (polars, numpy) over the table `m` from A1 written to Parquet. Stocks: not an index and
`mt >= 5e7`.

```python
import json, math, sys
import numpy as np, polars as pl
S = sys.argv[1]
m = pl.read_parquet(f"{S}/grid_daily.parquet")
stocks = m.filter(~pl.col("is_index") & (pl.col("mt") >= 5e7))
out = {}

def series_stats(daily: pl.DataFrame, col: str, cost_bps: float, periods=248):
    r = daily[col].to_numpy()
    n = len(r); mu = r.mean(); sd = r.std(ddof=1)
    net = r - cost_bps / 1e4
    eq = np.cumprod(1 + (np.exp(net) - 1)); peak = np.maximum.accumulate(eq); dd = (eq / peak - 1).min()
    months = daily.with_columns(pl.Series("net", net)).group_by("mo").agg(((pl.col("net").exp() - 1 + 1).log().sum().exp() - 1).alias("ret"), pl.len().alias("days")).sort("mo")
    return {
        "days": n, "gross_bps": round(mu * 1e4, 2), "cost_bps": cost_bps, "net_bps": round(mu * 1e4 - cost_bps, 2),
        "sd_bps": round(sd * 1e4, 1), "t_gross": round(mu / (sd / math.sqrt(n)), 2), "t_net": round((mu - cost_bps / 1e4) / (sd / math.sqrt(n)), 2),
        "win_days_pct": round(float((net > 0).mean()) * 100, 1), "sharpe_net": round((mu - cost_bps / 1e4) / sd * math.sqrt(periods), 2),
        "total_net_pct": round((eq[-1] - 1) * 100, 1), "max_dd_pct": round(dd * 100, 1),
        "worst_day_pct": round(float(net.min()) * 100, 2), "best_day_pct": round(float(net.max()) * 100, 2),
        "lakh_to": int(round(1e5 * eq[-1])), "months": [(int(a), round(b * 100, 1)) for a, b in months.select("mo", "ret").rows()],
        "months_positive": int((months["ret"] > 0).sum()), "months_total": months.height,
    }

def pick(col, sign, k=20):
    yr = stocks.group_by("symbol", "y").agg((pl.col(col).sum() * sign).alias("t"), pl.len().alias("n"))
    p = (yr.filter(pl.col("y").is_in([2024, 2025]) & (pl.col("n") >= 120))
         .group_by("symbol").agg(pl.col("t").min().alias("worst"), pl.len().alias("k"))
         .filter((pl.col("k") == 2) & (pl.col("worst") > 0)).sort("worst", descending=True).head(k))
    return p

def portfolio(col, sign, symbols=None, year=2026):
    df = stocks.filter(pl.col("y") == year)
    if symbols is not None:
        df = df.filter(pl.col("symbol").is_in(symbols))
    return df.group_by("d", "mo").agg((pl.col(col) * sign).mean().alias("r"), pl.len().alias("names")).sort("d")

strategies = [
    ("overnight_1514", "on1514", 1, "Buy 15:14, sell 09:29 next session", 25, 4),
    ("overnight_1414", "on1414", 1, "Buy 14:14, sell 09:29 next session", 25, 4),
    ("dayshort_1429", "sd0929_1429", -1, "Short 09:29, cover 14:29", 5, 4),
]
for key, col, sign, label, cash_cost, fut_cost in strategies:
    sgn = 1 if sign == 1 else 1  # columns already hold the short's return for sd*
    p = pick(col, 1)
    picks = p["symbol"].to_list()
    top = portfolio(col, 1, picks); allv = portfolio(col, 1)
    # per-stock 2026 results for the picks
    per = (stocks.filter((pl.col("y") == 2026) & pl.col("symbol").is_in(picks)).group_by("symbol")
           .agg((pl.col(col).mean() * 1e4).round(1).alias("bps"), (pl.col(col) > 0).mean().round(3).alias("hit"), pl.len().alias("n"), (pl.col(col).sum() * 100).round(1).alias("total_pct"), pl.col("mt").first().alias("mt"))
           .join(p, on="symbol").sort("worst", descending=True))
    hist = (stocks.filter(pl.col("symbol").is_in(picks)).group_by("symbol", "y").agg((pl.col(col).sum() * 100).round(0).alias("t")).pivot(on="y", index="symbol", values="t"))
    per = per.join(hist, on="symbol")
    out[key] = {"label": label, "picks": [
        {"symbol": r["symbol"], "turnover_cr": round(r["mt"] / 1e7), "y2024": r.get("2024"), "y2025": r.get("2025"), "y2026": r["total_pct"], "bps_2026": r["bps"], "hit_2026": round(r["hit"] * 100, 1), "sessions": r["n"]}
        for r in per.iter_rows(named=True)],
        "top20_cash": series_stats(top, "r", cash_cost), "top20_futures": series_stats(top, "r", fut_cost), "top20_gross": series_stats(top, "r", 0),
        "universe_gross": series_stats(allv, "r", 0), "universe_names": int(allv["names"].median()),
        "breakeven_bps": round(float(top["r"].mean()) * 1e4, 1)}
    # 2025 out-of-sample from 2024-only pick
    yr = stocks.group_by("symbol", "y").agg(pl.col(col).sum().alias("t"), pl.len().alias("n"))
    p24 = yr.filter((pl.col("y") == 2024) & (pl.col("n") >= 120)).sort("t", descending=True).head(20)["symbol"].to_list()
    out[key]["oos_2025_from_2024"] = series_stats(portfolio(col, 1, p24, 2025), "r", 0)
    out[key]["universe_2025"] = series_stats(portfolio(col, 1, None, 2025), "r", 0)
    out[key]["universe_by_year_bps"] = [(int(y), round(v * 1e4, 1)) for y, v in stocks.group_by("y").agg(pl.col(col).mean()).sort("y").rows()]

# weekend vs plain night, equal-weight portfolio per session, 15:14 -> 09:29 and 14:14 -> 09:29
wk = []
for col in ("on1514", "on1414"):
    s = stocks.group_by("d", "y", "across").agg(pl.col(col).mean().alias("r"))
    for (y, a), g in s.group_by(["y", "across"]):
        r = g["r"].to_numpy()
        wk.append({"window": col, "year": int(y), "across": a, "sessions": len(r), "mean_bps": round(r.mean() * 1e4, 1), "t": round(r.mean() / (r.std(ddof=1) / math.sqrt(len(r))), 2) if len(r) > 2 else None, "pos_pct": round(float((r > 0).mean()) * 100)})
out["weekend"] = sorted(wk, key=lambda x: (x["window"], x["year"], x["across"]))
# indices
idx = m.filter(pl.col("is_index") & pl.col("symbol").is_in(["NIFTY 50", "NIFTY BANK", "NIFTY NEXT 50", "NIFTY MIDCAP 150"]))
out["indices"] = [{"symbol": s, "year": int(y), "on1414_pct": round(a * 100, 1), "on1514_pct": round(b * 100, 1), "dayshort_pct": round(c * 100, 1), "n": int(n)}
                  for s, y, a, b, c, n in idx.group_by("symbol", "y").agg(pl.col("on1414").sum(), pl.col("on1514").sum(), pl.col("sd0929_1429").sum(), pl.len()).sort("symbol", "y").rows()]
json.dump(out, open(f"{S}/stats_strategies.json", "w"), indent=1, default=str)
for k in ("overnight_1514", "overnight_1414", "dayshort_1429"):
    o = out[k]; print(k, "breakeven", o["breakeven_bps"], "| cash", {x: o["top20_cash"][x] for x in ("net_bps","t_net","win_days_pct","sharpe_net","total_net_pct","max_dd_pct","lakh_to","months_positive")}, "| fut", {x: o["top20_futures"][x] for x in ("net_bps","t_net","sharpe_net","total_net_pct","max_dd_pct")}, "| universe", o["universe_gross"]["gross_bps"], o["universe_gross"]["t_gross"], "| oos25", o["oos_2025_from_2024"]["gross_bps"], o["oos_2025_from_2024"]["t_gross"], "univ25", o["universe_2025"]["gross_bps"])
    print("   univ by year", o["universe_by_year_bps"])
for w in out["weekend"]: print(w)
```

### A3. Analysis A: deciles and long-short

```python
import json, math
import numpy as np, polars as pl

D = "/home/shravan/Workspaces/indian-finance-data/data/"
out = {}
for key, p, times in [
    ("session", "parquet/inference/8402fb7e-1b51-46e9-ad16-1eb2563e3278.parquet", ["09:30", "11:00", "13:00"]),
    ("m15", "parquet/inference/2677462a-9922-4164-b52b-379cade99750.parquet", None),
]:
    df = (pl.scan_parquet(D + p).select("symbol", "ts", "day", "r", "pred_y", "y")
          .filter(pl.col("r").is_not_null() & pl.col("pred_y").is_not_null())
          .with_columns(pl.col("ts").dt.strftime("%H:%M").alias("t")))
    if times:
        df = df.filter(pl.col("t").is_in(times))
    df = df.collect()
    # decile of the prediction within each decision point (1 = lowest)
    df = df.with_columns((pl.col("pred_y").rank("ordinal").over("ts") / pl.len().over("ts") * 10)
                         .ceil().clip(1, 10).cast(pl.Int32).alias("dec"))
    res = {}
    for t in (times or ["all"]):
        d = df if t == "all" else df.filter(pl.col("t") == t)
        dec = (d.group_by("dec").agg((pl.col("r").mean() * 1e4).round(2).alias("bps"),
                                     ((pl.col("r") > 0).mean() * 100).round(1).alias("up_pct"), pl.len().alias("n")).sort("dec"))
        per = (d.group_by("ts").agg(pl.col("r").filter(pl.col("dec") == 10).mean().alias("top"),
                                    pl.col("r").filter(pl.col("dec") == 1).mean().alias("bot"),
                                    pl.col("r").mean().alias("all")).drop_nulls())
        ls = (per["top"] - per["bot"]).to_numpy(); top = per["top"].to_numpy()
        bot = per["bot"].to_numpy(); allr = per["all"].to_numpy()
        def tt(x): return round(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))), 2)
        res[t] = {"deciles": dec.rows(named=True), "points": len(per),
                  "long_short_bps": round(ls.mean() * 1e4, 2), "long_short_t": tt(ls),
                  "long_short_pos_pct": round(float((ls > 0).mean()) * 100, 1),
                  "top_bps": round(top.mean() * 1e4, 2), "top_minus_all_bps": round((top - allr).mean() * 1e4, 2),
                  "top_minus_all_t": tt(top - allr), "bottom_short_bps": round(-bot.mean() * 1e4, 2),
                  "bottom_short_t": tt(-bot), "bottom_short_pos_pct": round(float((-bot > 0).mean()) * 100, 1),
                  "all_bps": round(allr.mean() * 1e4, 2)}
        if t != "all":
            daily = (-bot) - 5 / 1e4                     # short the bottom decile, 5 bps a day
            eq = np.cumprod(np.exp(daily))
            res[t]["bottom_short_net5_total_pct"] = round((eq[-1] - 1) * 100, 1)
            res[t]["bottom_short_net5_sharpe"] = round(daily.mean() / daily.std(ddof=1) * math.sqrt(248), 2)
            res[t]["bottom_short_max_dd_pct"] = round(float((eq / np.maximum.accumulate(eq) - 1).min()) * 100, 1)
            lsd = ls - 10 / 1e4                          # long-short pair, 10 bps a day
            eq2 = np.cumprod(np.exp(lsd))
            res[t]["long_short_net10_total_pct"] = round((eq2[-1] - 1) * 100, 1)
            res[t]["long_short_net10_sharpe"] = round(lsd.mean() / lsd.std(ddof=1) * math.sqrt(248), 2)
            res[t]["long_short_max_dd_pct"] = round(float((eq2 / np.maximum.accumulate(eq2) - 1).min()) * 100, 1)
    out[key] = res
json.dump(out, open("stats_models.json", "w"), indent=1, default=str)
```
