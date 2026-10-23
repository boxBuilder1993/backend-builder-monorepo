"""The reference, us and kite legs on synthetic files and fakes: parsers, gap logic, outcomes."""

# ruff: noqa: E501
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

import gzip
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from finance.build import kite as BK
from finance.build import reference as R
from finance.build import us as BU
from finance.files import marker_rel, raw_rel
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
