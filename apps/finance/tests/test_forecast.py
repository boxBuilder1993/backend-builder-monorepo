"""The forecast rows on synthetic bars: the target is aligned to the next bar's open and the close Δ
later, nothing at t sees a bar after t, the daily layer is the previous session, the filters hold.
No network, no Postgres."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from finance.ml import data, forecast

DAYS = [date(2025, 6, 2), date(2025, 6, 3)]
SYMBOLS = ["AAA", "BBB", "CCC"]


def _bars(day: date, symbol: str, seed: int, thin: bool = False) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = forecast.SESSION_MINUTES
    ts = [
        datetime.combine(day, datetime.min.time()) + timedelta(hours=9, minutes=15 + i)
        for i in range(n)
    ]
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.0002, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.0003, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.0003, n)))
    volume = rng.integers(10, 100, n) if thin else rng.integers(10_000, 50_000, n)
    return pl.DataFrame(
        {
            "ts": ts,
            "symbol": [symbol] * n,
            "token": [seed] * n,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    ).with_columns(pl.col("ts").cast(pl.Datetime("us")), pl.col("volume").cast(pl.Int64))


def synthetic_mirror(
    root: Path, *, perturb_after: datetime | None = None, muhurat: bool = False
) -> None:
    for day in DAYS:
        frames = [_bars(day, s, i + 1, thin=(s == "CCC")) for i, s in enumerate(SYMBOLS)]
        frames.append(
            _bars(day, forecast.INDEX, 99).with_columns(pl.lit(0).cast(pl.Int64).alias("volume"))
        )
        for k, name in enumerate(("NIFTY FIN SERVICE", "NIFTY IT", "NIFTY 500")):
            frames.append(
                _bars(day, name, 90 + k).with_columns(pl.lit(0).cast(pl.Int64).alias("volume"))
            )
        if muhurat and day == DAYS[0]:  # an evening session, 18:00 to 19:15, like Diwali's
            frames = [
                f.filter(pl.col("ts").dt.hour() < 10).with_columns(
                    (pl.col("ts") + timedelta(hours=8, minutes=45)).alias("ts")
                )
                for f in frames
            ]
        df = pl.concat(frames)
        df = pl.concat(
            [df, df.filter(pl.col("symbol") == "AAA")]
        )  # a duplicated symbol, as the ETFs are
        if perturb_after is not None:
            df = df.with_columns(
                pl.when(pl.col("ts") > perturb_after)
                .then(pl.col("close") * 1.5)
                .otherwise(pl.col("close"))
                .alias("close")
            )
        out = root / f"parquet/equity_1m/year={day.year}"
        out.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out / f"equity_1m_{day:%Y%m%d}.parquet")
    daily_days = [DAYS[0] - timedelta(days=d) for d in range(40, 0, -1)] + DAYS
    rows: list[dict[str, object]] = []
    for i, d in enumerate(daily_days):
        for s in SYMBOLS:
            rows.append(
                {
                    "date": d,
                    "isin": f"IN{s}",
                    "symbol": s,
                    "series": "EQ",
                    "instrument": None,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0 + i,
                    "last": None,
                    "prev_close": None,
                    "volume": 1000,
                    "turnover": 1e8,
                    "trades": None,
                    "avg_price": None,
                    "deliv_qty": None,
                    "deliv_pct": 40.0 + i,
                    "fmt": None,
                }
            )
    eq = pl.DataFrame(rows)
    for d in daily_days:
        out = root / f"parquet/equity/year={d.year}"
        out.mkdir(parents=True, exist_ok=True)
        eq.filter(pl.col("date") == d).write_parquet(out / f"equity_{d:%Y%m%d}.parquet")
    idx = pl.concat(
        [
            pl.DataFrame(
                {
                    "date": daily_days,
                    "index_name": [forecast.INDEX_DAILY] * len(daily_days),
                    "close": [20000.0 + i for i in range(len(daily_days))],
                }
            ),
            pl.DataFrame(
                {
                    "date": daily_days,
                    "index_name": ["India VIX"] * len(daily_days),
                    "close": [12.0 + i / 10 for i in range(len(daily_days))],
                }
            ),
        ]
    )
    ref = root / "parquet/reference"
    ref.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "index": ["NIFTY 500", "NIFTY 500", "NIFTY BANK", "NIFTY 50"],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "isin": ["INAAA", "INBBB", "INAAA", "INBBB"],
            "company": ["a", "b", "a", "b"],
            "industry": [
                "Financial Services",
                "Information Technology",
                "Financial Services",
                "Information Technology",
            ],
            "series": ["EQ"] * 4,
            "as_of": [date(2026, 10, 24)]
            * 4,  # a snapshot after the rows' days: the first one applies
        }
    ).write_parquet(ref / "index_constituents.parquet")
    pl.DataFrame(
        {
            "instrument_token": [1, 2, 3],
            "exchange_token": [1, 2, 3],
            "tradingsymbol": SYMBOLS,
            "name": SYMBOLS,
            "segment": ["NSE"] * 3,
            "instrument_type": ["EQ"] * 3,
            "exchange": ["NSE"] * 3,
            "tick_size": [0.05, 0.05, 0.1],
            "lot_size": [1, 1, 1],
        }
    ).write_parquet(ref / "kite_instruments.parquet")
    us_rows = [
        {
            "date": d,
            "symbol": sym,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0 + i * 0.1,
            "adj_close": 100.0,
            "volume": 1,
            "dividend": 0.0,
            "split": 1.0,
        }
        for i, d in enumerate(daily_days)
        for sym in ("SPY", "QQQ", "IWM", "GLD", "SLV", "USO", "XLK", "XLF", "^VIX")
    ] + [
        {
            "date": d,
            "symbol": sym,
            "open": 50.0,
            "high": 51.0,
            "low": 49.0,
            "close": 50.0 + i + k,
            "adj_close": 50.0,
            "volume": 1000,
            "dividend": 0.0,
            "split": 1.0,
        }
        for i, d in enumerate(daily_days)
        for k, sym in enumerate(("UAA", "UBB"))
    ]
    pl.DataFrame(
        {
            "symbol": ["UAA", "UBB"],
            "name": ["ua", "ub"],
            "sector": ["Technology", "Finance"],
            "industry": ["Software", "Banks"],
            "country": ["United States"] * 2,
            "ipo_year": [1999, 2005],
            "cap_bucket": ["mega", "small"],
            "as_of": [date(2026, 10, 24)] * 2,
        }
    ).with_columns(pl.col("ipo_year").cast(pl.Int32)).write_parquet(ref / "us_sectors.parquet")
    for d in daily_days:
        out = root / f"parquet/us_equity/year={d.year}"
        out.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([r for r in us_rows if r["date"] == d]).write_parquet(
            out / f"us_equity_{d:%Y%m%d}.parquet"
        )
    for d in daily_days:
        out = root / f"parquet/commodity/year={d.year}"
        out.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "date": [d] * 3,
                "symbol": ["GOLD", "SILVER", "CRUDEOIL"],
                "instrument": ["FUTCOM"] * 3,
                "expiry": [d + timedelta(days=40)] * 3,
                "open": [1.0] * 3,
                "high": [1.0] * 3,
                "low": [1.0] * 3,
                "close": [100.0 + daily_days.index(d)] * 3,
                "volume": [1] * 3,
                "open_interest": [1] * 3,
            }
        ).write_parquet(out / f"commodity_{d:%Y%m%d}.parquet")
    for d in daily_days:
        out = root / f"parquet/index/year={d.year}"
        out.mkdir(parents=True, exist_ok=True)
        idx.filter(pl.col("date") == d).write_parquet(out / f"index_{d:%Y%m%d}.parquet")


def _rows(root: Path, **over: object) -> pl.DataFrame:
    con = data.connect(root)
    params = {
        **forecast.PARAMS,
        "start": DAYS[0].isoformat(),
        "end": DAYS[-1].isoformat(),
        "min_turnover": 1.0,
        **over,
    }
    return data.run_query(con, forecast.ROWS_SQL, params)


def test_target_is_next_open_to_close_h_later_scaled_by_trailing_vol(tmp_path: Path) -> None:
    synthetic_mirror(tmp_path)
    rows = _rows(tmp_path, horizon_minutes=15, step_minutes=1, vol_minutes=60)
    bars = pl.read_parquet(
        tmp_path / "parquet/equity_1m/year=2025/equity_1m_20250602.parquet"
    ).unique(["symbol", "ts"])
    a = bars.filter(pl.col("symbol") == "AAA").sort("ts")
    t = a["ts"][100]
    row = rows.filter((pl.col("symbol") == "AAA") & (pl.col("ts") == t)).row(0, named=True)
    assert row["entry"] == a["open"][101] and row["exit_"] == a["close"][115]
    r = math.log(a["close"][115] / a["open"][101])
    ret1 = np.diff(np.log(a["close"].to_numpy()))  # ret1[i] is the return of bar i+1
    sig = np.std(
        ret1[100 - 60 : 100], ddof=1
    )  # the 60 bar returns ending at bar 100, t's own included
    assert row["r"] == pytest.approx(r) and row["y"] == pytest.approx(
        r / (sig * math.sqrt(15)), rel=1e-6
    )
    assert row["r1"] == pytest.approx(math.log(a["close"][100] / a["close"][99]))
    assert row["r5"] == pytest.approx(math.log(a["close"][100] / a["close"][95]))
    assert row["m"] == 100 and row["m_left"] == forecast.SESSION_MINUTES - 100
    # the last Δ minutes of the day have no exit and are not rows; the duplicate symbol is one row
    last = rows.filter(pl.col("symbol") == "AAA").group_by("day").agg(pl.col("m").max())
    assert set(last["m"].to_list()) == {forecast.SESSION_MINUTES - 1 - 15}
    assert rows.filter(pl.col("symbol") == "AAA").select(["symbol", "ts"]).is_unique().all()
    assert set(rows["symbol"].to_list()) == set(SYMBOLS)  # the index is not a row
    assert set(rows.columns) >= {*forecast.FEATURES, *forecast.KEY, "y", "r"}


def test_nothing_at_t_sees_a_bar_after_t(tmp_path: Path) -> None:
    synthetic_mirror(tmp_path)
    cut = datetime(2025, 6, 2, 12, 0)
    before = _rows(tmp_path, step_minutes=1).filter(pl.col("ts") <= cut).sort(["symbol", "ts"])
    synthetic_mirror(tmp_path, perturb_after=cut)
    after = _rows(tmp_path, step_minutes=1).filter(pl.col("ts") <= cut).sort(["symbol", "ts"])
    slow = (
        set(forecast.DAILY)
        | set(forecast.PARTICIPATION)
        | set(forecast.ATTRIBUTES)
        | {"sp_5d", "sp_20d", "clv_prev"}
    )
    cross = [
        f
        for f in forecast.FEATURES
        if f not in forecast.CONTEXT and f not in slow and f not in forecast.CATEGORICAL
    ]
    for f in [*cross, *forecast.CLOCK, "i_r5", "i_r15", "i_r30", "i_r60", "i_r_open"]:
        a, b = before[f].to_numpy().astype(float), after[f].to_numpy().astype(float)
        assert np.array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9)), f
    # while the target, which is the future, did move for the last rows before the cut
    assert not np.allclose(before["y"].to_numpy(), after["y"].to_numpy(), equal_nan=True)


def test_daily_layer_is_the_previous_session_and_filters_hold(tmp_path: Path) -> None:
    synthetic_mirror(tmp_path)
    rows = _rows(tmp_path, step_minutes=5)
    day2 = rows.filter((pl.col("symbol") == "AAA") & (pl.col("day") == DAYS[1])).row(0, named=True)
    # the daily rows close at 100 + i; the previous session of DAYS[1] is DAYS[0], i = 40
    assert day2["ret_1d"] == pytest.approx(math.log(140 / 139)) and day2["deliv_pct"] == 80.0
    bars = pl.read_parquet(tmp_path / "parquet/equity_1m/year=2025/equity_1m_20250603.parquet")
    day_open = bars.filter(pl.col("symbol") == "AAA").sort("ts")["open"][0]
    assert day2["gap"] == pytest.approx(day_open / 140 - 1)
    assert day2["rs_20"] == pytest.approx(math.log(140 / 120) - math.log(20040 / 20020))
    assert day2["dow"] == 2 and day2["after_holiday"] == 0 and day2["expiry"] == 0
    assert (rows["m"] % 5 == 0).all()
    # the liquidity floor removes the thin symbol
    strict = _rows(tmp_path, step_minutes=5, min_turnover=1e6)
    assert "CCC" in rows["symbol"].to_list() and "CCC" not in strict["symbol"].to_list()
    assert (
        strict.filter(pl.col("symbol") == "AAA").height
        == rows.filter(pl.col("symbol") == "AAA").height
    )


def test_fit_definition_and_screen_shapes(tmp_path: Path) -> None:
    body = forecast.fit_definition(60, min_turnover=5e6)
    assert body["params"]["horizon_minutes"] == 60 and body["params"]["min_turnover"] == 5e6
    assert set(data.param_names(body["query"])) == set(body["params"])
    assert {f["name"] for f in body["features"]} == set(forecast.FEATURES)
    assert next(f for f in body["features"] if f["name"] == "dow")["kind"] == "categorical"
    synthetic_mirror(tmp_path)
    con = data.connect(tmp_path)
    out = tmp_path / "rows.parquet"
    n = forecast.materialise(
        con,
        {
            **forecast.PARAMS,
            "start": DAYS[0].isoformat(),
            "end": DAYS[-1].isoformat(),
            "min_turnover": 1.0,
        },
        out,
    )
    assert n > 0
    ics = forecast.screen(con, out, ["r5", "dow"])
    assert ics == []  # three symbols: below MIN_CROSS_SECTION, no decision point counts
    table = forecast.screen_table(ics)
    assert "| dow |" in table and "context" in table and "| r5 |" in table


def test_rows_file_hash_read_rows_and_extras(tmp_path: Path) -> None:
    synthetic_mirror(tmp_path)
    con = data.connect(tmp_path)
    params = {
        **forecast.PARAMS,
        "start": DAYS[0].isoformat(),
        "end": DAYS[-1].isoformat(),
        "min_turnover": 1.0,
    }
    out = tmp_path / "rows.parquet"
    n = data.write_rows(con, forecast.ROWS_SQL, params, out)
    assert n == pl.read_parquet(out).height > 0
    h1 = data.file_hash(out)
    assert h1 == data.file_hash(out) and len(h1) == 64
    pl.read_parquet(out).reverse().write_parquet(tmp_path / "rev.parquet")
    assert data.file_hash(tmp_path / "rev.parquet") == h1  # order-independent
    pl.read_parquet(out).head(n - 1).write_parquet(tmp_path / "less.parquet")
    assert data.file_hash(tmp_path / "less.parquet") != h1
    from finance.ml import schema as S

    body = forecast.fit_definition(15)
    declared = S.declare(body["key"], body["features"], body["target"])
    frame = data.read_rows(out, declared, fitting=True)
    assert frame["r5"].dtype == pl.Float32 and frame["y"].dtype == pl.Float32
    assert frame["dow"].dtype != pl.Float32 and frame["ts"].dtype == pl.Datetime("us")
    prepared = data.prepare(frame, declared, fitting=True)
    assert prepared.X.dtype == "float32" and prepared.X.shape == (n, len(forecast.FEATURES))
    assert prepared.extra is not None
    assert {"r", "scale", "entry", "exit_", "sig1", "day", "close"} <= set(prepared.extra.columns)
    assert not ({*forecast.FEATURES, "y", "symbol", "ts"} & set(prepared.extra.columns))


def test_table_metrics_ic_calibration_and_raw_net_return() -> None:
    from datetime import datetime

    from finance.ml import metrics

    rng = np.random.default_rng(0)
    points = [datetime(2026, 1, 5, 9, 30) + timedelta(days=d) for d in range(40)]
    rows: list[dict[str, object]] = []
    for ts in points:
        y = rng.normal(size=30)
        for i in range(30):
            rows.append(
                {
                    "ts": ts,
                    "symbol": f"S{i}",
                    "y": y[i],
                    "pred_y": y[i] + rng.normal(scale=2.0),
                    "r": y[i] * 0.002,
                    "scale": 0.002,
                }
            )
    table = pl.DataFrame(rows)
    got = metrics.evaluate(
        table,
        ["y"],
        [
            "ic",
            "ic_se",
            "ic_points",
            "calibration_slope",
            "calibration_monotone",
            "net_return_raw",
            "trades_raw",
        ],
        forecast.evaluation_params(threshold=0.002, cost_bps=5.0, by="month"),
    )
    assert 0.2 < got["ic.y"] < 0.7 and got["ic_se.y"] > 0 and got["ic_points.y"] == 40
    assert (
        0.1 < got["calibration_slope.y"] < 0.4
    )  # noisy predictions: truth moves less than they say
    assert got["calibration_monotone.y"] >= 0.7
    picked = table.filter(pl.col("pred_y") * pl.col("scale") > 0.002)
    assert got["trades_raw.y"] == picked.height
    assert got["net_return_raw.y"] == pytest.approx(float(picked["r"].to_numpy().mean()) - 5e-4)
    assert "ic.y@2026-01" in got and "ic.y@2026-02" in got
    perfect = table.with_columns(pl.col("y").alias("pred_y"))
    p = metrics.evaluate(
        perfect, ["y"], ["ic", "calibration_slope", "calibration_monotone"], {"group": "ts"}
    )
    assert p["ic.y"] == pytest.approx(1.0) and p["calibration_slope.y"] == pytest.approx(1.0)
    assert p["calibration_monotone.y"] == 1.0
    with pytest.raises(ValueError, match="needs the column"):
        metrics.evaluate(table.drop("ts"), ["y"], ["ic"], {})
    with pytest.raises(ValueError, match="raw metrics need"):
        metrics.evaluate(table.drop("scale"), ["y"], ["net_return_raw"], {})
    with pytest.raises(ValueError, match="not a column"):
        metrics.evaluate(table, ["y"], ["ic"], {"by": "sector"})


def test_an_evening_session_yields_no_rows_and_no_nan_targets(tmp_path: Path) -> None:
    synthetic_mirror(tmp_path, muhurat=True)
    rows = _rows(tmp_path, step_minutes=1)
    assert set(rows["day"].to_list()) == {DAYS[1]}  # the evening session is not a decision point
    assert rows["y"].is_finite().all()
    assert float(rows["m"].to_numpy().min()) >= 0 and float(rows["m_left"].to_numpy().min()) >= 1
    assert float(rows["scale"].to_numpy().min()) > 0


def test_new_layers_resolve_from_the_reference_and_context_tables(tmp_path: Path) -> None:
    synthetic_mirror(tmp_path)
    rows = _rows(tmp_path, step_minutes=5)
    a = rows.filter(
        (pl.col("symbol") == "AAA") & (pl.col("day") == DAYS[1]) & (pl.col("m") == 100)
    ).row(0, named=True)
    b = rows.filter(
        (pl.col("symbol") == "BBB") & (pl.col("day") == DAYS[1]) & (pl.col("m") == 100)
    ).row(0, named=True)
    c = rows.filter(
        (pl.col("symbol") == "CCC") & (pl.col("day") == DAYS[1]) & (pl.col("m") == 100)
    ).row(0, named=True)
    # attributes from the first snapshot (dated after the rows); the fallback for the unlabelled
    assert a["industry"] == "Financial Services" and a["sector_idx"] == "NIFTY FIN SERVICE"
    assert (
        b["industry"] == "Information Technology"
        and b["sector_idx"] == "NIFTY IT"
        and b["in_n50"] == 1
    )
    assert (
        c["industry"] is None and c["sector_idx"] == forecast.SECTOR_FALLBACK and c["in_n500"] == 0
    )
    assert a["in_n500"] == 1 and a["in_n50"] == 0
    assert a["tick_bps"] == pytest.approx(0.05 / 140 * 1e4) and a["log_px"] == pytest.approx(
        math.log(140)
    )
    # peers: AAA is alone in its industry, so its peer return is its own and the spread is zero
    assert (
        a["p_r15"] == pytest.approx(a["r15"])
        and a["xp_r15"] == pytest.approx(0.0)
        and a["sp_5d"] == pytest.approx(0.0)
    )
    # the sector index joined by name: one 15-minute return for every stock of the sector
    assert a["s_r15"] is not None and a["xs_r15"] == pytest.approx(a["r15"] - a["s_r15"])
    # breadth is the same for all three stocks at the minute and lies in [0, 1]
    assert (
        a["breadth_open"] == b["breadth_open"] == c["breadth_open"] and 0 <= a["breadth_open"] <= 1
    )
    # overseas and regime: the US row before DAYS[1] is the 41st (i = 40)
    assert a["spy_cc"] is not None and a["spy_cc"] == pytest.approx(
        math.log((100 + 40 * 0.1) / (100 + 39 * 0.1))
    )
    assert a["gold_cc"] == pytest.approx(math.log(140 / 139)) and a["vix"] == pytest.approx(
        12.0 + 40 / 10
    )
    assert (
        a["mkt_ret60"] is None or a["mkt_ret60"] == a["mkt_ret60"]
    )  # 42 sessions of index: 60-day return is null
    assert a["log_trades"] is None  # the fixture's daily rows carry no trades
    # calendar
    assert a["dom"] == 3 and a["moy"] == 6 and a["dte"] >= 0
    # opening range: defined from minute 15 on, position within it, break flag in {-1, 0, 1}
    early = rows.filter((pl.col("symbol") == "AAA") & (pl.col("m") == 10)).row(0, named=True)
    assert early["or_pos"] is None and early["or_break"] is None
    assert a["or_break"] in (-1, 0, 1) and a["or_pos"] is not None
    assert set(rows.columns) >= set(forecast.FEATURES)


def test_daily_rows_decide_at_the_close_and_enter_at_the_next_open(tmp_path: Path) -> None:
    synthetic_mirror(tmp_path)
    con = data.connect(tmp_path)
    params = {
        **forecast.DAILY_PARAMS,
        "start": "2025-04-01",
        "end": DAYS[-1].isoformat(),
        "min_turnover_day": 1.0,
    }
    rows = data.run_query(con, forecast.DAILY_SQL, {**params, "horizon_sessions": 1})
    assert set(rows.columns) >= {
        *forecast.DAILY_FEATURES,
        "symbol",
        "ts",
        "y",
        "r",
        "scale",
        "entry",
        "exit_",
    }
    a = rows.filter((pl.col("symbol") == "AAA") & (pl.col("day") == DAYS[0])).row(0, named=True)
    # the fixture's daily rows: open 100, close 100 + i; DAYS[0] is i = 40, DAYS[1] is i = 41
    assert a["entry"] == 100.0 and a["exit_"] == 141.0 and a["entry_date"] == DAYS[1]
    assert a["r"] == pytest.approx(math.log(141 / 100)) and a["y"] == pytest.approx(
        a["r"] / a["scale"]
    )
    assert a["ts"] == datetime.combine(DAYS[0], datetime.min.time()).replace(hour=15, minute=30)
    assert a["ret_1d"] == pytest.approx(math.log(140 / 139)) and a["deliv_pct"] == 80.0
    # the US session dated t is known before the entry at t+1's open: it is the row used
    assert a["spy_cc"] == pytest.approx(math.log((100 + 40 * 0.1) / (100 + 39 * 0.1)))
    assert a["industry"] == "Financial Services" and a["sector_idx"] == "Nifty Financial Services"
    assert a["s_ret_1d"] is None  # the fixture's index table has no sector index
    # the session's shape comes from the minute bars held for DAYS[0]
    assert (
        a["rv_day"] is not None and a["vwap_dev_close"] is not None and a["r_first60"] is not None
    )
    assert a["dow"] == 2 and a["after_holiday"] == 0  # the entry day is DAYS[1], a Tuesday
    # the last session has no next open and is not a row; the horizon moves the exit
    assert rows.filter(pl.col("day") == DAYS[1]).height == 0
    h3 = data.run_query(con, forecast.DAILY_SQL, {**params, "horizon_sessions": 3})
    last_h3 = h3.filter(pl.col("symbol") == "AAA")["day"].max()
    assert isinstance(last_h3, date) and last_h3 < DAYS[0]
    body = forecast.daily_fit_definition(21)
    assert body["params"]["horizon_sessions"] == 21
    assert set(data.param_names(body["query"])) == set(body["params"])


def test_us_daily_rows_use_the_screener_sectors_etfs_and_the_nifty_of_the_entry_day(
    tmp_path: Path,
) -> None:
    synthetic_mirror(tmp_path)
    con = data.connect(tmp_path)
    params = {
        **forecast.DAILY_PARAMS,
        "horizon_sessions": 1,
        "start": "2025-04-01",
        "end": DAYS[-1].isoformat(),
        "min_turnover_day": 1.0,
    }
    rows = data.run_query(con, forecast.US_DAILY_SQL, params)
    assert set(rows.columns) >= {*forecast.daily_features("us"), "symbol", "ts", "y", "r", "scale"}
    assert set(rows["symbol"].to_list()) == {
        "UAA",
        "UBB",
    }  # the ETFs and the VIX are context, not rows
    a = rows.filter((pl.col("symbol") == "UAA") & (pl.col("day") == DAYS[0])).row(0, named=True)
    assert a["entry"] == 50.0 and a["exit_"] == 50.0 + 41 and a["entry_date"] == DAYS[1]
    assert a["industry"] == "Technology" and a["sector_idx"] == "XLK"
    assert a["in_n50"] == 1 and a["in_small250"] == 0
    assert a["tick_bps"] == pytest.approx(0.01 / 90 * 1e4)  # a cent on a 90-dollar close
    assert a["s_ret_1d"] == pytest.approx(
        math.log((100 + 40 * 0.1) / (100 + 39 * 0.1))
    )  # XLK's day
    assert a["vix"] == pytest.approx(100 + 40 * 0.1)  # the fixture's ^VIX series
    # the Nifty session of the entry day (closing 06:00 ET) is the one used; the ETFs are day t's

    assert a["nifty_cc"] == pytest.approx(math.log(20041 / 20040))
    assert a["gld_cc"] == pytest.approx(math.log(104.0 / 103.9))
    assert a["dte"] >= 0 and a["dow"] == 2
    b = rows.filter((pl.col("symbol") == "UBB") & (pl.col("day") == DAYS[0])).row(0, named=True)
    assert b["sector_idx"] == "XLF" and b["in_small250"] == 1 and b["in_n500"] == 0
    us_features = forecast.daily_features("us")
    assert "deliv_pct" not in us_features and "log_trades" not in us_features
    body = forecast.daily_fit_definition(21, "us")
    assert body["name"] == "forecast 21d us" and body["params"]["start"] == "2015-01-01"
    assert set(data.param_names(body["query"])) == set(body["params"])


def test_topk_and_rescale_metrics() -> None:
    from datetime import datetime

    from finance.ml import metrics

    rows: list[dict[str, object]] = []
    for d in range(30):
        ts = datetime(2026, 1, 5, 9, 30) + timedelta(days=d)
        for i in range(25):
            pred = (
                i - 12
            ) / 10  # ordered predictions; realised return follows them with noise-free order
            rows.append(
                {
                    "ts": ts,
                    "symbol": f"S{i}",
                    "y": pred,
                    "pred_y": pred * 3,
                    "r": pred * 0.001,
                    "scale": 0.001,
                }
            )
    table = pl.DataFrame(rows)
    got = metrics.evaluate(
        table,
        ["y"],
        ["net_return_topk", "trades_topk", "spread_topk", "net_return_raw", "trades_raw"],
        forecast.evaluation_params(threshold=0.0005, cost_bps=0.0, by=None, top_k=3, rescale=1 / 3),
    )
    # top 3 per point are i = 24, 23, 22: mean pred (1.2 + 1.1 + 1.0) / 3 * 0.001
    assert got["net_return_topk.y"] == pytest.approx(1.1 * 0.001) and got["trades_topk.y"] == 90
    # bottom 3 are i = 0, 1, 2 with mean -1.1 * 0.001: the spread is 2.2 bps... in return units
    assert got["spread_topk.y"] == pytest.approx(2.2 * 0.001)
    # rescaled, the predicted raw move is pred_y / 3 * scale = pred * 0.001: 0.0005 keeps i >= 18
    assert got["trades_raw.y"] == 30 * 7
    unscaled = metrics.evaluate(
        table, ["y"], ["trades_raw"], forecast.evaluation_params(0.0005, by=None)
    )
    assert (
        unscaled["trades_raw.y"] > got["trades_raw.y"]
    )  # the exaggerated magnitudes clear the threshold more often
    with pytest.raises(ValueError, match="top-k metrics need"):
        metrics.evaluate(table.drop("r"), ["y"], ["net_return_topk"], {})


def test_write_rows_in_chunks_matches_one_shot_and_drops_the_warm_up(tmp_path: Path) -> None:
    import duckdb

    con = duckdb.connect()
    con.execute(
        "create table t as select make_timestamp(2025, 1, 1, 10, 0, 0) + interval (i) day as ts, "
        "i as x from range(300) r(i)"
    )
    sql = (
        "select ts, x, sum(x) over (order by ts rows between 2 preceding and current row) as s3 "
        "from t where ts::date between :start and :end"
    )
    params = {"start": "2025-02-01", "end": "2025-09-30"}
    one = tmp_path / "one.parquet"
    chunked = tmp_path / "chunked.parquet"
    n1 = data.write_rows(con, sql, params, one, chunk_months=0)
    n2 = data.write_rows(con, sql, params, chunked, chunk_months=3, warmup_days=10)
    a = pl.read_parquet(one).sort("ts")
    b = pl.read_parquet(chunked).sort("ts")
    assert n1 == n2 == a.height == b.height == 242  # Feb 1 to Sep 30
    assert a["ts"].to_list() == b["ts"].to_list() and a["x"].to_list() == b["x"].to_list()
    # the window needs two prior rows: the one-shot build lacks them at its start, the chunked one
    # has them from the warm-up at every chunk boundary, the first included
    assert a["s3"][0] == a["x"][0] and b["s3"][0] == b["x"][0] + b["x"][0] - 1 + b["x"][0] - 2
    i = b["ts"].to_list().index(datetime(2025, 5, 1, 10, 0))
    assert b["s3"][i] == b["x"][i] + b["x"][i - 1] + b["x"][i - 2]
    assert not (tmp_path / "chunked.parts").exists()
    assert data.chunks(date(2025, 11, 15), date(2026, 2, 10), 3) == [
        (date(2025, 11, 15), date(2026, 1, 31)),
        (date(2026, 2, 1), date(2026, 2, 10)),
    ]


def test_features_from_screen_keeps_stable_signal_and_every_context_feature(tmp_path: Path) -> None:
    years = [2023, 2024, 2025, 2026]
    rows: list[dict[str, Any]] = []
    for y in years:
        rows += [
            {"feature": "r5", "year": y, "ic": -0.05, "se": 0.001, "points": 100},  # stable, strong
            {"feature": "gap", "year": y, "ic": 0.0005, "se": 0.001, "points": 100},  # noise
            {
                "feature": "ret_5d",
                "year": y,
                "ic": 0.004 if y % 2 else -0.004,
                "se": 0.001,
                "points": 100,
            },  # flips
            {
                "feature": "deliv_pct",
                "year": y,
                "ic": 0.004,
                "se": 0.001,
                "points": 100,
            },  # stable, just above the floor
        ]
    pl.DataFrame(rows).write_parquet(tmp_path / "screen.parquet")
    kept = forecast.features_from_screen(
        tmp_path / "screen.parquet", forecast.FEATURES, forecast.CONTEXT
    )
    assert "r5" in kept and "deliv_pct" in kept
    assert "gap" not in kept and "ret_5d" not in kept
    assert set(forecast.CONTEXT) <= set(kept) and "industry" in kept and "dow" in kept
    assert kept == [f for f in forecast.FEATURES if f in set(kept)]  # order of the feature list
    assert "r15" not in kept  # not in the screen table at all: not kept
