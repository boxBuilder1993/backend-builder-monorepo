"""The forecast rows on synthetic bars: the target is aligned to the next bar's open and the close Δ
later, nothing at t sees a bar after t, the daily layer is the previous session, the filters hold.
No network, no Postgres."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path

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


def _mirror(root: Path, *, perturb_after: datetime | None = None, muhurat: bool = False) -> None:
    for day in DAYS:
        frames = [_bars(day, s, i + 1, thin=(s == "CCC")) for i, s in enumerate(SYMBOLS)]
        frames.append(
            _bars(day, forecast.INDEX, 99).with_columns(pl.lit(0).cast(pl.Int64).alias("volume"))
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
    idx = pl.DataFrame(
        {
            "date": daily_days,
            "index_name": [forecast.INDEX_DAILY] * len(daily_days),
            "close": [20000.0 + i for i in range(len(daily_days))],
        }
    )
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
    _mirror(tmp_path)
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
    _mirror(tmp_path)
    cut = datetime(2025, 6, 2, 12, 0)
    before = _rows(tmp_path, step_minutes=1).filter(pl.col("ts") <= cut).sort(["symbol", "ts"])
    _mirror(tmp_path, perturb_after=cut)
    after = _rows(tmp_path, step_minutes=1).filter(pl.col("ts") <= cut).sort(["symbol", "ts"])
    cross = [f for f in forecast.FEATURES if f not in forecast.CONTEXT and f not in forecast.DAILY]
    for f in [*cross, *forecast.CLOCK, "i_r5", "i_r15", "i_r30", "i_r60", "i_r_open"]:
        a, b = before[f].to_numpy().astype(float), after[f].to_numpy().astype(float)
        assert np.array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9)), f
    # while the target, which is the future, did move for the last rows before the cut
    assert not np.allclose(before["y"].to_numpy(), after["y"].to_numpy(), equal_nan=True)


def test_daily_layer_is_the_previous_session_and_filters_hold(tmp_path: Path) -> None:
    _mirror(tmp_path)
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
    _mirror(tmp_path)
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
    _mirror(tmp_path)
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
    _mirror(tmp_path, muhurat=True)
    rows = _rows(tmp_path, step_minutes=1)
    assert set(rows["day"].to_list()) == {DAYS[1]}  # the evening session is not a decision point
    assert rows["y"].is_finite().all() and rows["m"].min() >= 0
    assert rows["m_left"].min() >= 1 and rows["scale"].min() > 0
