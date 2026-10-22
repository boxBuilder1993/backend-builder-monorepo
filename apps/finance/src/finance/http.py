"""HTTP sessions for the exchanges, and the retry used around every request.

nsearchives.nseindia.com and niftyindices.com answer a browser user agent with no cookies.
www.nseindia.com/api needs cookies from the landing page first. mcxindia.com sits behind Akamai:
plain requests get 403, Chrome TLS impersonation through curl_cffi passes.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# requests and curl_cffi type their JSON loosely; callers validate the payloads they use.

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import requests
from curl_cffi import requests as cr

log = logging.getLogger("finance.http")

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)
HEADERS = {"user-agent": UA, "accept": "*/*", "accept-language": "en-US,en;q=0.9"}
TIMEOUT = 30


class Transient(Exception):
    """May succeed on retry: network error, 5xx, 429, block page, non-JSON body."""


def retry[T](fn: Callable[[], T], what: str, attempts: int = 4, base_delay: float = 2.0) -> T:
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i == attempts - 1:
                raise
            delay = base_delay * 2**i
            log.warning("%s: %s (retry %d/%d in %.0fs)", what, exc, i + 1, attempts - 1, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def looks_html(blob: bytes) -> bool:
    head = blob[:1000].lower()
    return b"<html" in head or b"<!doctype" in head


class NseArchives:
    """nsearchives.nseindia.com and niftyindices.com."""

    def __init__(self) -> None:
        self.s = requests.Session()
        self.s.headers.update(HEADERS)

    def get(self, url: str, **kw: Any) -> requests.Response:
        r = self.s.get(url, timeout=TIMEOUT, **kw)
        if r.status_code >= 500 or r.status_code == 429:
            raise Transient(f"HTTP {r.status_code}")
        return r


class NseApi:
    """www.nseindia.com/api/*: a cookie from the landing page first, renewed on 401/403."""

    def __init__(self) -> None:
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self.warm = False

    def get_json(self, url: str) -> Any:
        if not self.warm:
            self.s.get("https://www.nseindia.com", timeout=TIMEOUT)
            time.sleep(1)
            self.warm = True
        r = self.s.get(
            url,
            headers={"referer": "https://www.nseindia.com/", "accept": "application/json"},
            timeout=TIMEOUT,
        )
        if r.status_code in (401, 403) or r.status_code >= 500:
            self.warm = False
            raise Transient(f"HTTP {r.status_code}")
        try:
            data: Any = r.json()
        except ValueError:
            self.warm = False
            raise Transient("non-JSON response") from None
        return data


class Mcx:
    """mcxindia.com behind Akamai. A warm-up GET of the bhavcopy page, then the JSON endpoint."""

    PAGE = "https://www.mcxindia.com/market-data/bhavcopy"
    URL = PAGE + "/GetDateWiseBhavCopy"
    HDR = {
        "x-requested-with": "XMLHttpRequest",
        "referer": PAGE,
        "accept": "application/json, text/javascript, */*; q=0.01",
    }

    def __init__(self) -> None:
        self.s: cr.Session | None = None

    def _warm(self) -> cr.Session:
        s = cr.Session(impersonate="chrome")
        r = s.get(self.PAGE, timeout=60)
        if r.status_code != 200:
            raise Transient(f"warm-up HTTP {r.status_code}")
        self.s = s
        return s

    def get_json(self, params: dict[str, str]) -> Any:
        s = self.s or self._warm()
        r = s.get(self.URL, params=params, headers=self.HDR, timeout=60)
        if r.status_code != 200:
            self.s = None
            raise Transient(f"HTTP {r.status_code}")
        try:
            data: Any = r.json()
        except ValueError:
            self.s = None
            raise Transient("non-JSON (blocked or 404 page)") from None
        return data
