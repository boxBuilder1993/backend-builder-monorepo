# What we fit first

Status: proposal, 2026-10-22. The two problems from the dip research, stated as Fits for
[finance-ml.md](finance-ml.md). Short on purpose; this is the part to argue with.

## Ground rules, carried over from the research

- **Costs in the target, not in the metric afterwards.** A return is net of brokerage, STT, exchange
  charges and the flat DP fee from the old repo's `docs/brokers.md`, at the position size being
  modelled. A model that is good before costs is not interesting.
- **Two periods, every start date.** Nothing is believed from one window. Validation is a later date
  range than the fit, and a result is reported on at least two disjoint validation ranges before it is
  called anything.
- **No hindsight universe.** Liquidity and price filters use trailing data up to the row's date.
- **Unadjusted prices, corporate actions as unit scaling.** As in the simulator rules.
- **The metric that matters is net return per trade, then its consistency.** Accuracy and AUC are
  reported, never optimised.

## The row

One (market, symbol, date) from `equity_adj` for NSE and `us_equity` for the US, with features computed
in the query as window functions over the trailing data: returns over 1, 5, 20 and 60 sessions, the
day's move and gap, position in the trailing range, volume relative to its trailing median, delivery
percentage where NSE reports it, distance from trailing high and low, trailing volatility, the market
and the sector as categoricals, the day of week. Nothing from after the row's date.

## Problem 1: exit, "I bought on day d; hold or sell now?"

A row per (symbol, buy date d, current date t) for t within a maximum hold. Features: the row's
features at t, plus the position's own state: sessions held, return since entry, drawdown since entry,
distance from the target. Target, a vector of two: the net return of selling now versus holding to the
rule's horizon, and whether the next five sessions' best exit beats selling now. Task: regression on
the first for the decision, with the second kept as a binary diagnostic in a separate Fit.

Why first: it is the better-posed problem, the hold decision has a bounded horizon and a clear
counterfactual, and the dip research already showed the entry edge exists while the exit rule was
fixed by hand.

## Problem 2: entry, "given a principal, buy this today, and for how long?"

A row per (symbol, date). Target: the net return of the best exit within 1, 5, 10 and 20 sessions, as a
four-column vector, so one model gives the expected net return at each horizon and the horizon with the
highest expectation is the recommended hold. A second, binary Fit predicts whether the 20-session
exit beats the baseline of holding the index, which is the honest comparison the research used.

Position size enters through the costs in the target: the fee floor at Rs 5,000 differs from the one at
Rs 1 lakh, so the Fit's query takes `:principal` as a parameter and the same definition is run at
several sizes.

## Splits

Fit on 2021-09 to 2024-12, validate on 2025-01 to 2025-12 and again on 2026-01 to today, as two Fits
differing only in the validation query, or one Fit run twice with `:fit_end` and `:val_start`
parameters. The same for the US data, separately; the markets are not pooled until a pooled model beats
both single-market ones on both validations.

## Search

Each Fit: `sklearn.linear` as the baseline, `xgboost` on the GPU over max_depth {3, 5, 7}, learning rate
{0.03, 0.1}, n_estimators {300, 1000}, min_child_weight {1, 10}: 24 models plus the baseline, about an
hour on the GTX 1080 for the NSE universe. A model that does not beat the baseline on both validations
is noise, whatever its training metrics say.

## What we expect to learn

Whether the dip edge is predictable per stock-day rather than as a flat rule; whether anything beats
"sell at the target or after 20 sessions" as an exit; and at which position size either survives costs.
If the answer is no on both validations, that is the result, and it goes in the findings next to the
twenty-one already there.
