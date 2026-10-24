"""The reference, us and kite legs on synthetic files and fakes: parsers, gap logic, outcomes."""

# ruff: noqa: E501
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from finance.build import kite as BK
from finance.build import reference as R
from finance.build import us as BU
from finance.files import marker_rel, raw_rel
from finance.settings import KITE_EXTRA
from finance.sources import Ledger
from finance.sources import kite as K
from finance.sources import us as US
from finance.sources.us_worker import frame_from_yahoo


@pytest.mark.parametrize(
    "subject, kind, factor",
    [
        ("Bonus 1:1", "bonus", 0.5),
        ("Bonus 2:1", "bonus", 1 / 3),
        ("Bonus- 1:2", "bonus", 2 / 3),
        (
            "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share",
            "split",
            0.2,
        ),
        (
            "Face Value Split (Sub-Division) - From Rs10/- Per Share To Re 1/- Per Share",
            "split",
            0.1,
        ),
        (
            "Consolidation Of Shares - From Re 1/- Per Share To Rs 10/- Per Share",
            "consolidation",
            10.0,
        ),
        (
            "Bonus 1:1 and Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 5/- Per Share",
            "bonus+split",
            0.25,
        ),
        ("Interim Dividend - Rs 2 Per Share", "dividend", None),
        ("Rights 1:5 @ Premium Rs 100", "rights", None),
        ("Buy Back Of Shares", "buyback", None),
        ("Scheme Of Arrangement - Bonus Ncrps 3:1", "other", None),
    ],
)
def test_classify_action(subject: str, kind: str, factor: float | None) -> None:
    k, f = R.classify_action(subject)
    assert k == kind
    assert f is None if factor is None else f == pytest.approx(factor)


def test_reference_parsers(tmp_path: Path) -> None:
    corp = tmp_path / "corp_actions_2026.json.gz"
    corp.write_bytes(
        gzip.compress(
            json.dumps(
                [
                    {
                        "exDate": "10-Jul-2026",
                        "recDate": "10-Jul-2026",
                        "isin": "INE1",
                        "symbol": "AAA",
                        "series": "EQ",
                        "comp": "Aaa Ltd",
                        "faceVal": "10",
                        "subject": "Bonus 1:1",
                    },
                    {
                        "exDate": "10-Jul-2026",
                        "recDate": "10-Jul-2026",
                        "isin": "INE1",
                        "symbol": "AAA",
                        "series": "EQ",
                        "comp": "Aaa Ltd",
                        "faceVal": "10",
                        "subject": "Bonus 1:1",
                    },  # duplicate row from NSE
                    {
                        "exDate": "01-Jul-2026",
                        "recDate": None,
                        "isin": "INE2",
                        "symbol": "BBB",
                        "series": "EQ",
                        "comp": "Bbb",
                        "faceVal": "2",
                        "subject": "Dividend - Rs 3 Per Share",
                    },
                ]
            ).encode()
        )
    )
    df = R.parse_corp_actions([corp])
    assert df.height == 2 and df["symbol"].to_list() == ["BBB", "AAA"]  # sorted by ex_date, deduped
    assert df.row(1, named=True)["factor"] == 0.5 and df.row(0, named=True)["kind"] == "dividend"
    hol = tmp_path / "holidays_2026.json.gz"
    hol.write_bytes(
        gzip.compress(
            json.dumps(
                {"CM": [{"tradingDate": "26-Jan-2026", "description": "Republic Day"}], "FO": None}
            ).encode()
        )
    )
    h = R.parse_holidays([hol])
    assert h.row(0, named=True) == {
        "date": date(2026, 1, 26),
        "segment": "CM",
        "description": "Republic Day",
    }
    sc = tmp_path / "symbolchange.csv.gz"
    sc.write_bytes(gzip.compress(b"Some Company, Ltd,OLD,NEW,05-Jan-2026\nbad line\n"))
    s = R.parse_symbol_changes(sc)
    assert s.row(0, named=True) == {
        "company": "Some Company, Ltd",
        "old_symbol": "OLD",
        "new_symbol": "NEW",
        "change_date": date(2026, 1, 5),
    }
    eq = tmp_path / "equity_l.csv.gz"
    eq.write_bytes(
        gzip.compress(
            b"SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING, PAID UP VALUE, MARKET LOT, ISIN NUMBER, FACE VALUE\nAAA,Aaa Ltd,EQ,12-Mar-2010,10,1,INE1,10\n"
        )
    )
    m = R.parse_equity_master(eq)
    assert m.row(0, named=True)["listing_date"] == date(2010, 3, 12) and m["market_lot"][0] == 1


def test_securities_link_isin_chains(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for i, d in enumerate([date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]):
        isin = "INE_OLD" if i < 2 else "INE_NEW"  # face-value split: new ISIN on the same symbol
        rows.append({"date": d, "symbol": "AAA", "series": "EQ", "isin": isin, "close": 1.0})
        rows.append({"date": d, "symbol": "BBB", "series": "EQ", "isin": "INE_B", "close": 2.0})
    for d, part in pl.DataFrame(rows).partition_by("date", as_dict=True).items():
        p = tmp_path / f"parquet/equity/year=2026/equity_{d[0]:%Y%m%d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        part.write_parquet(p)
    sec = R.build_securities(tmp_path)
    assert sec is not None
    by = {r["isin"]: r["sid"] for r in sec.iter_rows(named=True)}
    assert by["INE_OLD"] == by["INE_NEW"] == "INE_OLD" and by["INE_B"] == "INE_B"
    assert R.build_securities(tmp_path / "empty") is None


NASDAQ = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\nAAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\nQQQ|Invesco QQQ Trust|G|N|N|100|Y|N\nZTST|Test Co - Common Stock|Q|Y|N|100|N|N\nFile Creation Time: 1022202623:00|||||||\n"
OTHER = "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\nBRK B|Berkshire Hathaway Inc. Class B|N|BRK B|N|100|N|BRK.B\nABC$A|Abc Preferred Series A|N|ABCpA|N|100|N|ABC$A\nSPY|SPDR S&P 500|P|SPY|Y|100|N|SPY\nFile Creation Time: 1022202623:00|||||||\n"


def test_listings_parse_and_universe(tmp_path: Path) -> None:
    d = date(2026, 10, 22)
    for f, text in (("nasdaqlisted.txt", NASDAQ), ("otherlisted.txt", OTHER)):
        p = tmp_path / BU.listing_rel(d, f)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(text.encode(), mtime=0))
    df = BU.parse_listings(tmp_path, d)
    rows = {r["symbol"]: r for r in df.iter_rows(named=True)}
    assert (
        rows["BRK.B"]["yahoo"] == "BRK-B"
        and rows["BRK.B"]["exchange"] == "NYSE"
        and rows["BRK.B"]["kind"] == "stock"
    )
    assert rows["ABC$A"]["kind"] == "preferred" and rows["ABC$A"]["yahoo"] == "ABC-PA"
    assert rows["QQQ"]["kind"] == "etf" and rows["ZTST"]["test"] is True
    uni = BU.price_universe(tmp_path, d)["symbol"].to_list()
    assert uni == [
        "AAPL",
        "BRK.B",
        "QQQ",
        "SPY",
    ]  # stocks plus benchmarks; no test issue, no preferred
    assert BU.listing_days(tmp_path) == [d]


def test_yahoo_wide_frame_to_rows() -> None:
    import pandas as pd

    idx = pd.to_datetime(["2026-10-20", "2026-10-21"])
    cols = pd.MultiIndex.from_product(
        [
            ["AAPL", "GONE"],
            ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"],
        ]
    )
    wide = pd.DataFrame(
        [
            [1, 2, 0.5, 1.5, 1.5, 100, 0, 0] + [None] * 8,
            [1, 2, 0.5, 1.6, 1.6, 200, 0.1, 0] + [None] * 8,
        ],
        index=idx,
        columns=cols,
    )
    df = frame_from_yahoo(wide, ["AAPL", "GONE", "ABSENT"])
    assert df["symbol"].to_list() == ["AAPL", "AAPL"] and df["volume"].dtype == pl.Int64
    assert df.row(1, named=True)["dividend"] == 0.1 and df.row(0, named=True)["date"] == date(
        2026, 10, 20
    )
    assert frame_from_yahoo(None, ["X"]).is_empty()


def test_us_gaps_and_dormant() -> None:
    uni = pl.DataFrame({"yahoo": ["NEW", "BEHIND", "CURRENT", "DEAD"]})
    last = {"BEHIND": date(2026, 10, 10), "CURRENT": date(2026, 10, 21)}
    dormant = {"DEAD": {"empty": 3, "since": "2026-09-01"}}
    full, tail, latest = US.gaps(uni, last, dormant, date(2026, 10, 22))
    assert (full, tail, latest) == (["NEW"], ["BEHIND"], date(2026, 10, 21))


def test_fetch_prices_writes_batches_and_parks_empty_symbols(tmp_path: Path) -> None:
    uni = pl.DataFrame({"yahoo": ["AAA", "BBB", "CCC"]})
    calls: list[tuple[list[str], date | None]] = []

    def batch(symbols: list[str], start: date | None) -> pl.DataFrame:
        calls.append((symbols, start))
        return pl.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "date": [date(2026, 10, 21)] * 2,
                "open": [1.0, 2.0],
                "high": [1.0, 2.0],
                "low": [1.0, 2.0],
                "close": [1.0, 2.0],
                "adj_close": [1.0, 2.0],
                "volume": [1, 2],
                "dividend": [0.0, 0.0],
                "split": [0.0, 0.0],
            },
            schema=BU.RAW_SCHEMA,
        )

    written, counts = US.fetch_prices(
        tmp_path, date(2026, 10, 22), uni, batch=batch, sleep=lambda s: None
    )
    assert calls == [(["AAA", "BBB", "CCC"], None)]
    assert [(w.source, w.kind) for w in written] == [
        ("yahoo", "raw"),
        ("yahoo", "raw"),
    ]  # the batch and the dormant file
    assert counts["fetched_symbols"] == 2 and counts["dormant"] == 0
    assert US.load_dormant(tmp_path) == {"CCC": {"empty": 1, "since": "2026-10-22"}}
    sessions = BU.build_us_equity(tmp_path)
    assert sessions == [date(2026, 10, 21)]
    assert pl.read_parquet(tmp_path / BU.us_equity_rel(date(2026, 10, 21)))["symbol"].to_list() == [
        "AAA",
        "BBB",
    ]
    assert BU.build_us_equity(tmp_path) == []  # manifest: nothing new to fold


def test_kite_parse_ranges_and_missing_days(tmp_path: Path) -> None:
    line = {
        "symbol": "RELIANCE",
        "token": 1,
        "candles": [["2026-10-21T09:15:00+05:30", 1.0, 2.0, 0.5, 1.5, 10]],
    }
    raw = gzip.compress((json.dumps(line) + "\n").encode())
    df = BK.parse(raw, date(2026, 10, 21))
    assert (
        df.row(0, named=True)["ts"] == datetime(2026, 10, 21, 9, 15)
        and df["volume"].dtype == pl.Int64
    )
    with pytest.raises(ValueError, match="candle"):
        BK.parse(raw, date(2026, 10, 22))
    days = [date(2024, 1, d) for d in range(1, 31)] + [date(2024, 6, 1)]
    assert K.ranges(days) == [
        (date(2024, 1, 1), date(2024, 1, 30)),
        (date(2024, 6, 1), date(2024, 6, 1)),
    ]
    (tmp_path / raw_rel(K.NAME, date(2023, 10, 3), K.EXT)).parent.mkdir(parents=True)
    (tmp_path / raw_rel(K.NAME, date(2023, 10, 3), K.EXT)).write_bytes(b"x")
    (tmp_path / marker_rel(K.NAME, date(2023, 10, 4))).write_text("no candles")
    missing = K.missing_days(
        tmp_path,
        date(2023, 10, 6),
        date(2023, 10, 6),
        Ledger(no_data=frozenset({(K.NAME, date(2023, 10, 5))})),
    )
    assert missing == [date(2023, 10, 1), date(2023, 10, 2), date(2023, 10, 6)]


def test_kite_fetch_writes_days_and_reports_empty_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(K, "KITE_START", date(2026, 10, 20))
    insts = [K.Instrument("RELIANCE", 1), K.Instrument("TCS", 2)]

    def fetcher(kite: Any, inst: K.Instrument, a: date, b: date) -> list[dict[str, Any]]:
        return [
            {
                "date": datetime(2026, 10, 21, 9, 15),
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 10,
            }
        ]

    monkeypatch.setattr(K, "datetime", _Clock)  # after 16:00, so today counts
    written, no_data, counts = K.fetch(
        tmp_path, date(2026, 10, 22), Ledger(), object(), insts, fetcher=fetcher
    )
    assert [(w.source, w.day) for w in written] == [("kite_1m", date(2026, 10, 21))]
    assert sorted(n.day for n in no_data) == [date(2026, 10, 20), date(2026, 10, 22)]
    assert counts == {"kite_requests": 2, "kite_instruments": 2}
    built = K.build(tmp_path)
    assert [(w.source, w.day) for w in built] == [("equity_1m", date(2026, 10, 21))]
    assert pl.read_parquet(tmp_path / built[0].path)["symbol"].to_list() == ["RELIANCE", "TCS"]


class _Clock(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> _Clock:  # type: ignore[override]
        return cls(2026, 10, 22, 19, 0)


def test_kite_login_needed_is_an_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(K, "KITE_CONFIG", tmp_path / "none.toml")
    r = K.leg(tmp_path, date(2026, 10, 22), Ledger())
    assert r.implemented and r.written == [] and "login needed" in r.notes["kite"]


CONSTITUENTS = (
    b"Company Name,Industry,Symbol,Series,ISIN Code\r\n"
    b"HDFC Bank Ltd.,Financial Services,HDFCBANK,EQ,INE040A01034\r\n"
    b"Karur Vysya Bank Ltd.,Financial Services,KARURVYSYA,EQ,INE036D01028\r\n"
)


class _Archives:
    """NseArchives fake: the same bytes for every list, unless told otherwise per file name."""

    def __init__(self, by_name: dict[str, bytes] | None = None, default: bytes = CONSTITUENTS):
        self.by_name, self.default, self.calls = by_name or {}, default, 0

    def get(self, url: str, **kw: Any) -> Any:
        self.calls += 1
        name = url.rsplit("/", 1)[1].removesuffix(".csv")
        content = self.by_name.get(name, self.default)

        class R:
            status_code = 200 if content else 404

        r = R()
        r.content = content  # type: ignore[attr-defined]
        return r


def test_index_snapshots_are_written_on_change_only_and_build_one_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from finance.sources import reference as REF

    monkeypatch.setattr(
        REF, "INDEX_FILES", {"NIFTY 500": "ind_nifty500list", "NIFTY BANK": "ind_niftybanklist"}
    )

    def _no_retry(fn: Callable[[], bytes], what: str) -> bytes:
        return fn()

    monkeypatch.setattr(REF, "retry", _no_retry)
    day1, day2, day3 = date(2026, 10, 24), date(2026, 10, 25), date(2026, 10, 26)
    w1 = REF.fetch_indices(tmp_path, day1, _Archives())  # type: ignore[arg-type]
    assert sorted(w.path for w in w1) == [
        "raw/nse_indices/2026/nse_indices_20261024_ind_nifty500list.csv.gz",
        "raw/nse_indices/2026/nse_indices_20261024_ind_niftybanklist.csv.gz",
    ]
    assert all(w.source == "nse_indices" and w.day == day1 for w in w1)
    w2 = REF.fetch_indices(tmp_path, day2, _Archives())  # type: ignore[arg-type]
    assert [w.path for w in w2] == [w.path for w in w1]  # unchanged bytes: no new snapshot
    changed = CONSTITUENTS + b"ACC Ltd.,Construction Materials,ACC,EQ,INE012A01025\r\n"
    w3 = REF.fetch_indices(tmp_path, day3, _Archives({"ind_nifty500list": changed}))  # type: ignore[arg-type]
    assert len(w3) == 3 and any(w.day == day3 and "nifty500" in w.path for w in w3)
    built = REF.build_reference(tmp_path)
    table = pl.read_parquet(
        tmp_path / [w for w in built if w.source == "index_constituents"][0].path
    )
    assert table.columns == ["index", "symbol", "isin", "company", "industry", "series", "as_of"]
    assert table.filter(pl.col("index") == "NIFTY 500").group_by("as_of").len().sort("as_of")[
        "len"
    ].to_list() == [2, 3]
    assert (
        table.filter(pl.col("symbol") == "ACC").row(0, named=True)["industry"]
        == "Construction Materials"
    )
    # a list the archive lacks is skipped, not the leg's failure
    w4 = REF.fetch_indices(tmp_path, day3, _Archives({"ind_niftybanklist": b""}))  # type: ignore[arg-type]
    assert len(w4) == 3


def test_kite_universe_is_liquid_or_index_or_recent(tmp_path: Path) -> None:
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(130)]
    rows = []
    for d in days:
        rows.append(
            {"date": d, "isin": "INLIQ", "symbol": "LIQUID", "series": "EQ", "turnover": 30e7}
        )
        rows.append(
            {"date": d, "isin": "INTHIN", "symbol": "THIN", "series": "EQ", "turnover": 1e7}
        )
        rows.append(
            {"date": d, "isin": "INMEM", "symbol": "MEMBER", "series": "EQ", "turnover": 2e7}
        )
    eq = pl.DataFrame(rows)
    out = tmp_path / "parquet/equity/year=2026"
    out.mkdir(parents=True)
    eq.write_parquet(out / "equity_all.parquet")
    ref = tmp_path / "parquet/reference"
    ref.mkdir(parents=True)
    pl.DataFrame(
        {
            "index": ["NIFTY 500", "NIFTY 500", "NIFTY BANK"],
            "symbol": ["MEMBER", "OLDMEMBER", "THIN"],
            "isin": ["INMEM", "INOLD", "INTHIN"],
            "company": ["m", "o", "t"],
            "industry": ["x", "x", "y"],
            "series": ["EQ"] * 3,
            "as_of": [date(2026, 6, 1), date(2026, 1, 1), date(2026, 6, 1)],
        }
    ).write_parquet(ref / "index_constituents.parquet")
    pl.DataFrame(
        {
            "instrument_token": [1, 2, 3, 4, 5],
            "exchange_token": [1, 2, 3, 4, 5],
            "tradingsymbol": ["LIQUID", "THIN", "MEMBER", "FADED", "NIFTY 50"],
            "name": [""] * 5,
            "segment": ["NSE", "NSE", "NSE", "NSE", "INDICES"],
            "instrument_type": ["EQ"] * 5,
            "exchange": ["NSE"] * 5,
            "tick_size": [0.05] * 5,
            "lot_size": [1] * 5,
        }
    ).write_parquet(ref / "kite_instruments.parquet")
    m = tmp_path / "parquet/equity_1m/year=2026"
    m.mkdir(parents=True)
    pl.DataFrame(
        {
            "ts": [datetime(2026, 5, 4, 9, 15)],
            "symbol": ["FADED"],
            "token": [4],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1],
        }
    ).write_parquet(m / "equity_1m_20260504.parquet")
    insts, missing, counts = K.universe(tmp_path)
    got = {i.symbol for i in insts}
    assert {
        "LIQUID",
        "MEMBER",
        "FADED",
        "NIFTY 50",
    } <= got  # turnover rule, index member, stayed in
    assert (
        "THIN" not in got and "OLDMEMBER" not in got
    )  # thin and not a member; member of an old snapshot only
    assert counts == {"liquid": 1, "index": 1, "recent": 1, "universe": 3 + len(KITE_EXTRA)}
    assert set(missing) >= set(KITE_EXTRA)  # the ETFs have no token in this fixture


def test_kite_fetch_stops_cleanly_at_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(K, "KITE_START", date(2026, 8, 1))
    monkeypatch.setattr(K, "datetime", _Clock)
    insts = [K.Instrument("RELIANCE", 1)]
    calls = 0

    def fetcher(kite: Any, inst: K.Instrument, a: date, b: date) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return []

    import time as _t

    written, no_data, counts = K.fetch(
        tmp_path,
        date(2026, 10, 22),
        Ledger(),
        object(),
        insts,
        fetcher=fetcher,
        deadline=_t.monotonic(),
    )
    assert calls == 0 and written == [] and no_data == []
    assert counts["kite_ranges_left"] == len(
        K.ranges(K.missing_days(tmp_path, date(2026, 10, 22), date(2026, 10, 22), Ledger()))
    )
    written, no_data, counts = K.fetch(
        tmp_path,
        date(2026, 10, 22),
        Ledger(),
        object(),
        insts,
        fetcher=fetcher,
        deadline=_t.monotonic() + 10_000,
    )
    assert calls > 0 and "kite_ranges_left" not in counts
