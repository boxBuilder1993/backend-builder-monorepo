"""The nse leg on synthetic files: every quirk the old poller found, as a test, with no network."""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from datetime import date
from pathlib import Path
from typing import cast

import polars as pl
import pytest
from finance.build import mcx as M
from finance.build import nse as N
from finance.build.xlsx import is_xlsx, xlsx_to_csv
from finance.files import parquet_rel, raw_rel
from finance.http import Mcx, Transient
from finance.sources import Ledger, LegResult
from finance.sources import nse as S
from finance.sources.daily import Got, NoFile
from finance.sources.mcx import McxBhav

D = date(2024, 7, 8)

OLD_CM = (
    "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,\n"
    "RELIANCE,EQ,2900,2950.5,2890,2940,2941,2895,1000,2930000.5,08-JUL-2024,120,INE002A01018,\n"
    "ZZZ,BE,10,11,9,10.5,10.5,10,5,52.5,08-JUL-2024,2,INE000Z00000,\n"
)
UDIFF_CM = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,"
    "OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,"
    "ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n"
    "2024-07-08,2024-07-08,CM,NSE,STK,2885,INE002A01018,RELIANCE,EQ,,,,,Reliance,2900,2950.5,2890,2940,2941,"
    "2895,,,,,1000,2930000.5,120,F1,1,,,,,\n"
    "2024-07-08,2024-07-08,CM,NSE,STK,1,INE000Z00000,ZZZ,BE,,,,,Zzz,10,11,9,10.5,10.5,10,,,,,5,52.5,2,F1,1,,,,,\n"
)
FULL = (
    "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE,"
    " AVG_PRICE,"
    " TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
    "RELIANCE, EQ, 08-Jul-2024, 2895, 2900, 2950.5, 2890, 2941, 2940, 2930.0, 1000, 29.3, 120, 600,"
    " 60.00\n"
    "ZZZ, BE, 08-Jul-2024, 10, 10, 11, 9, 10.5, 10.5, 10.5, 5, 0.0005, 2, -, -\n"
)
INDEX = (
    "Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,"
    "Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield\n"
    "Nifty 50,08-07-2024,24300,24400,24250,24320.55,20.5,0.08,250000000,25000.5,22.5,4.1,1.2\n"
    "Nifty Bank,08-07-2024,52000,52500,51900,52400,100,0.19,100000000,12000,15,2.5,1.0\n"
)


def zipped(csv: str, name: str = "x.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, csv)
    return buf.getvalue()


def test_cm_both_layouts_parse_to_one_schema() -> None:
    old, new = N.parse_cm(zipped(OLD_CM)), N.parse_cm(zipped(UDIFF_CM))
    assert old.columns == new.columns == N.EQUITY_COLS[:14] + ["fmt"]
    assert old["fmt"][0] == "old" and new["fmt"][0] == "udiff"
    assert old.drop("fmt", "instrument").equals(new.drop("fmt", "instrument"))
    assert N.cm_date(zipped(OLD_CM)) == N.cm_date(zipped(UDIFF_CM)) == D
    assert old["turnover"][0] == 2930000.5 and old["volume"].dtype == pl.Int64


def test_cm_urls_follow_the_layout_cutovers() -> None:
    cm = S.NseCm(client=None)  # type: ignore[arg-type]
    assert [u[:45] for u in cm.urls(date(2023, 12, 29))] == [
        "https://nsearchives.nseindia.com/content/hist"
    ]
    assert len(cm.urls(date(2024, 3, 1))) == 2  # both layouts available: UDiFF first
    assert cm.urls(date(2024, 3, 1))[0].endswith("BhavCopy_NSE_CM_0_0_0_20240301_F_0000.csv.zip")
    assert [u[:45] for u in cm.urls(date(2024, 7, 8))] == [
        "https://nsearchives.nseindia.com/content/cm/B"
    ]
    assert cm.urls(date(2024, 7, 5))[1].endswith("/2024/JUL/cm05JUL2024bhav.csv.zip")


class _Resp:
    def __init__(self, status: int, content: bytes) -> None:
        self.status_code, self.content = status, content

    @property
    def text(self) -> str:
        return self.content.decode()


class _Client:
    def __init__(self, answers: dict[str, _Resp]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    def get(self, url: str, **kw: object) -> _Resp:
        self.calls.append(url)
        for key, resp in self.answers.items():
            if key in url:
                return resp
        return _Resp(404, b"")


def test_full_rejects_previous_sessions_file() -> None:
    stale = FULL.replace("08-Jul-2024", "05-Jul-2024")
    full = S.NseFull(_Client({"sec_bhavdata_full_08072024": _Resp(200, stale.encode())}))  # type: ignore[arg-type]
    assert full.fetch(D) == NoFile("stale:2024-07-05")
    ok = S.NseFull(_Client({"sec_bhavdata_full_08072024": _Resp(200, FULL.encode())}))  # type: ignore[arg-type]
    assert isinstance(ok.fetch(D), Got)
    html = S.NseFull(_Client({"sec_bhavdata": _Resp(200, b"<!DOCTYPE html><html>blocked</html>")}))  # type: ignore[arg-type]
    with pytest.raises(Transient):
        html.fetch(D)


def test_index_html_page_is_no_data_and_cm_falls_back_to_old_layout() -> None:
    idx = S.NseIndex(_Client({"ind_close_all": _Resp(200, b"<html><body>No file</body></html>")}))  # type: ignore[arg-type]
    assert idx.fetch(D) == NoFile("html")
    march = zipped(OLD_CM.replace("08-JUL-2024", "01-MAR-2024"))
    client = _Client({"/historical/": _Resp(200, march)})
    cm = S.NseCm(client)  # type: ignore[arg-type]
    assert cm.fetch(date(2024, 3, 1)) == Got(march)  # UDiFF 404 -> old layout hit
    assert len(client.calls) == 2 and "/historical/" in client.calls[1]
    assert cm.fetch(date(2024, 8, 1)) == NoFile("404")  # only UDiFF is tried after the cutover


def test_full_delivery_dash_is_null_and_equity_join() -> None:
    df = N.equity_day(N.parse_cm(zipped(UDIFF_CM)), N.parse_full(FULL.encode()))
    assert df.columns == N.EQUITY_COLS
    rel, zzz = df.row(0, named=True), df.row(1, named=True)
    assert (rel["deliv_qty"], rel["deliv_pct"], rel["avg_price"]) == (600, 60.0, 2930.0)
    assert zzz["deliv_qty"] is None and zzz["deliv_pct"] is None and zzz["avg_price"] == 10.5
    without = N.equity_day(N.parse_cm(zipped(UDIFF_CM)), None)
    assert without["deliv_qty"].null_count() == 2 and without.columns == N.EQUITY_COLS


def test_excel_at_csv_url_is_converted_and_kept_as_served(tmp_path: Path) -> None:
    rows = FULL.splitlines()
    shared = sorted(
        {c.strip() for r in rows for c in r.split(",") if not c.strip().replace(".", "").isdigit()}
    )
    si = "".join(f"<si><t>{s}</t></si>" for s in shared)
    sheet_rows: list[str] = []
    for ri, r in enumerate(rows, 1):
        cells: list[str] = []
        for ci, c in enumerate(r.split(",")):
            c = c.strip()
            col = chr(65 + ci) if ci < 26 else "A" + chr(65 + ci - 26)
            if c.replace(".", "").isdigit():
                cells.append(f'<c r="{col}{ri}"><v>{c}</v></c>')
            else:
                cells.append(f'<c r="{col}{ri}" t="s"><v>{shared.index(c)}</v></c>')
        sheet_rows.append(f"<row>{''.join(cells)}</row>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("xl/sharedStrings.xml", f"<sst>{si}</sst>")
        z.writestr(
            "xl/worksheets/sheet1.xml",
            f"<worksheet><sheetData>{''.join(sheet_rows)}</sheetData></worksheet>",
        )
    blob = buf.getvalue()
    assert is_xlsx(blob)
    csv = xlsx_to_csv(blob)
    assert N.full_date(blob) == D
    assert N.parse_full(csv).equals(N.parse_full(FULL.encode()))


def test_mcx_no_success_is_no_data() -> None:
    class C:
        def __init__(self, j: dict[str, object]) -> None:
            self.j = j

        def get_json(self, params: dict[str, str]) -> dict[str, object]:
            return self.j

    assert McxBhav(cast("Mcx", C({"IsSuccess": False, "Message": "No Record Found"}))).fetch(
        D
    ) == NoFile("No Record Found")
    row: dict[str, object] = {
        "Date": "07/05/2024",
        "Symbol": "GOLD",
        "InstrumentName": "FUTCOM",
        "ExpiryDate": "05AUG2024",
        "Open": 1.0,
        "High": 2.0,
        "Low": 0.5,
        "Close": 1.5,
        "PreviousClose": 1.4,
        "Volume": 10,
        "Value": 1.1,
        "OpenInterest": 100,
        "VolumeInThousands": "10.000 KGS",
    }
    assert McxBhav(cast("Mcx", C({"IsSuccess": True, "Data": [row]}))).fetch(D) == NoFile(
        "stale:2024-07-05"
    )
    row["Date"] = "07/08/2024"
    got = McxBhav(cast("Mcx", C({"IsSuccess": True, "Data": [row]}))).fetch(D)
    assert isinstance(got, Got)
    df = M.parse_mcx(got.content)
    assert df.row(0, named=True)["expiry"] == date(2024, 8, 5) and df["volume"].dtype == pl.Int64


def test_check_day_catches_a_wrong_date() -> None:
    df = N.parse_index(INDEX.encode())
    N.check_day(df, D, "index")
    with pytest.raises(ValueError):
        N.check_day(df, date(2024, 7, 9), "index")


# ---- the leg on fakes -------------------------------------------------------------------------


class _Fake:
    """A fetcher answering from a dict; anything else is a 404."""

    def __init__(self, name: str, ext: str, files: dict[date, bytes]) -> None:
        self.name, self.ext, self.host, self.files = name, ext, "nsearchives", files
        self.asked: list[date] = []

    def fetch(self, day: date) -> Got | NoFile:
        self.asked.append(day)
        return Got(self.files[day]) if day in self.files else NoFile("404")


def test_fetch_sources_registers_skips_and_fetches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(S, "BACKFILL_START", date(2024, 7, 5))
    today = date(2024, 7, 9)
    existing = tmp_path / raw_rel("nse_cm", date(2024, 7, 5), "zip")
    existing.parent.mkdir(parents=True)
    existing.write_bytes(zipped(OLD_CM.replace("08-JUL-2024", "05-JUL-2024")))
    cm = _Fake("nse_cm", "zip", {D: zipped(UDIFF_CM)})
    ledger = Ledger(no_data=frozenset({("nse_cm", date(2024, 7, 6))}))
    pauses: list[float] = []
    r = S.fetch_sources(tmp_path, today, ledger, [cm], sleep=pauses.append)

    assert cm.asked == [date(2024, 7, 7), D, date(2024, 7, 9)]  # 5th on disk, 6th trusted no_data
    assert [(w.source, w.day, w.kind) for w in r.written] == [
        ("nse_cm", date(2024, 7, 5), "raw"),
        ("nse_cm", D, "raw"),
    ]
    assert [(n.source, n.day) for n in r.no_data] == [
        ("nse_cm", date(2024, 7, 7)),
        ("nse_cm", date(2024, 7, 9)),
    ]
    assert (tmp_path / raw_rel("nse_cm", D, "zip")).read_bytes() == zipped(UDIFF_CM)
    assert r.notes["requests"] == 3 and pauses == [0.25] * 3
    assert all(w.sha256 and w.bytes > 0 for w in r.written)


def test_build_tables_equity_waits_for_delivery(tmp_path: Path) -> None:
    for rel, blob in [
        (raw_rel("nse_cm", D, "zip"), zipped(UDIFF_CM)),
        (
            raw_rel("nse_cm", date(2024, 7, 9), "zip"),
            zipped(UDIFF_CM.replace("2024-07-08", "2024-07-09")),
        ),
        (raw_rel("nse_full", D, "csv.gz"), gzip.compress(FULL.encode(), mtime=0)),
        (raw_rel("nse_index", D, "csv.gz"), gzip.compress(INDEX.encode(), mtime=0)),
    ]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
    result = LegResult()
    for src, day, ext in [
        ("nse_cm", D, "zip"),
        ("nse_cm", date(2024, 7, 9), "zip"),
        ("nse_full", D, "csv.gz"),
        ("nse_index", D, "csv.gz"),
    ]:
        result.written.append(S.written_file(tmp_path, raw_rel(src, day, ext), "raw", src, day))
    S.build_tables(tmp_path, result, Ledger())

    parquet = [(w.source, w.day) for w in result.written if w.kind == "parquet"]
    assert parquet == [
        ("equity", D),
        ("index", D),
    ]  # the 9th waits: no delivery file, not known empty
    assert result.notes["built"] == 2 and result.notes["equity_waiting"] == ["2024-07-09"]
    eq = pl.read_parquet(tmp_path / parquet_rel("equity", D))
    assert eq.columns == N.EQUITY_COLS and eq["deliv_qty"].to_list() == [600, None]

    # known empty delivery file: equity is built without delivery fields; nothing is rebuilt
    result2 = LegResult(written=[w for w in result.written if w.kind == "raw"])
    S.build_tables(tmp_path, result2, Ledger(no_data=frozenset({("nse_full", date(2024, 7, 9))})))
    assert [(w.source, w.day) for w in result2.written if w.kind == "parquet"] == [
        ("equity", D),
        ("equity", date(2024, 7, 9)),
        ("index", D),
    ]
    assert result2.notes["built"] == 1


def test_leg_end_to_end_on_fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(S, "BACKFILL_START", D)
    fakes = [
        _Fake("nse_cm", "zip", {D: zipped(UDIFF_CM)}),
        _Fake("nse_full", "csv.gz", {D: FULL.encode()}),
        _Fake("nse_index", "csv.gz", {D: INDEX.encode()}),
        _Fake("mcx", "json.gz", {}),
    ]
    monkeypatch.setattr(S, "fetchers", lambda: fakes)

    def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("time.sleep", no_sleep)
    r = S.leg(tmp_path, D, Ledger())
    assert sorted((w.kind, w.source) for w in r.written) == [
        ("parquet", "equity"),
        ("parquet", "index"),
        ("raw", "nse_cm"),
        ("raw", "nse_full"),
        ("raw", "nse_index"),
    ]
    assert [(n.source, n.day) for n in r.no_data] == [("mcx", D)]
    assert json.dumps(r.notes)  # notes are JSON for the run record
