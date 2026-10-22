"""Minimal XLSX reader for the one day NSE served an Excel workbook at a .csv URL (2022-08-08).

The workbook is plain: shared strings and numbers, no styles, no date serials, no formulas.
Converting it to CSV keeps the parser unchanged, and the raw file stays exactly what was served.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from xml.sax.saxutils import unescape

_SI = re.compile(r"<si>(.*?)</si>", re.S)
_T = re.compile(r"<t[^>]*>(.*?)</t>", re.S)
_ROW = re.compile(r"<row[^>]*>(.*?)</row>", re.S)
_CELL = re.compile(r'<c\s+r="([A-Z]+)\d+"([^>]*)>(.*?)</c>|<c\s+r="([A-Z]+)\d+"([^>]*)/>', re.S)
_V = re.compile(r"<v>(.*?)</v>", re.S)
_IS = re.compile(r"<is>(.*?)</is>", re.S)


def is_xlsx(blob: bytes) -> bool:
    return blob[:2] == b"PK" and b"[Content_Types].xml" in blob[:2000]


def _col_index(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def xlsx_to_csv(blob: bytes) -> bytes:
    """First worksheet to CSV bytes. ValueError if the workbook is not the simple shape above."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    names = z.namelist()
    sheets = sorted(n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
    if not sheets:
        raise ValueError("no worksheet in workbook")
    shared: list[str] = []
    if "xl/sharedStrings.xml" in names:
        ss = z.read("xl/sharedStrings.xml").decode("utf-8", "replace")
        shared = ["".join(_T.findall(si)) for si in _SI.findall(ss)]
    sheet = z.read(sheets[0]).decode("utf-8", "replace")

    out = io.StringIO()
    w = csv.writer(out, lineterminator="\r\n")
    for row_xml in _ROW.findall(sheet):
        cells: dict[int, str] = {}
        for m in _CELL.finditer(row_xml):
            if m.group(1):
                ref, attrs, body = m.group(1), m.group(2), m.group(3)
            else:
                ref, attrs, body = m.group(4), m.group(5), ""
            kind = re.search(r'\bt="([^"]+)"', attrs or "")
            ctype = kind.group(1) if kind else "n"
            if ctype == "s":
                v = _V.search(body)
                text = shared[int(v.group(1))] if v and int(v.group(1)) < len(shared) else ""
            elif ctype == "inlineStr":
                inline = _IS.search(body)
                text = "".join(_T.findall(inline.group(1))) if inline else ""
            else:
                v = _V.search(body)
                text = v.group(1) if v else ""
            cells[_col_index(ref)] = unescape(text, {"&quot;": '"', "&apos;": "'"})
        if not cells:
            continue
        w.writerow([cells.get(i, "") for i in range(max(cells) + 1)])
    return out.getvalue().encode()


def to_csv_if_xlsx(blob: bytes) -> bytes:
    return xlsx_to_csv(blob) if is_xlsx(blob) else blob
