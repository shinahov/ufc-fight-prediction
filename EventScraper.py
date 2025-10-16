#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This code is for educational purposes only.
# It demonstrates web scraping techniques and is not intended for production or distribution.

import csv, json, re, time, urllib.parse, itertools
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "" #"http://ufcstats.com"
ALLOWED_HOSTS = {""} #{"ufcstats.com", "www.ufcstats.com"}
EVENTS_INDEX = f"{BASE}/statistics/events/completed?page=all"

GOOD_PROXIES_FILE = Path("good_proxies.txt")
GOOD_PROXIES_BACKUP = Path("good_proxies.backup.txt")

REQUEST_TIMEOUT = 20
SLEEP_RANGE = (1.0, 2.0)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

DEBUG = True
LOG_FILE = Path("scrape_debug.log")

def log(msg: str) -> None:
    if not DEBUG:
        return
    line = f"[DBG] {time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line)
    try:
        LOG_FILE.write_text(LOG_FILE.read_text(encoding="utf-8") + line + "\n", encoding="utf-8")
    except FileNotFoundError:
        LOG_FILE.write_text(line + "\n", encoding="utf-8")
    except Exception:
        pass

def sleep_a_bit():
    time.sleep((SLEEP_RANGE[0] + SLEEP_RANGE[1]) / 2)

def force_http(url: str) -> str:
    return re.sub(r"^https://", "http://", (url or ""), flags=re.I)

def build_session(proxy: Optional[str]) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, connect=4, read=4, backoff_factor=0.7,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET", "HEAD"]),
                  respect_retry_after_header=True)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.trust_env = False
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    s.headers.update({
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,de-DE,de;q=0.8",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    })
    return s

from urllib.parse import urlparse, urljoin

def _same_domain_redirect_allowed(current_url: str, location: str) -> Optional[str]:
    if not location:
        return None
    next_url = location if location.startswith("http") else urljoin(current_url, location)
    host = (urlparse(next_url).hostname or "").lower()
    return next_url if host in ALLOWED_HOSTS else None

def http_get(url: str, session: requests.Session, referer: Optional[str] = None) -> Optional[str]:
    try:
        hops, max_hops = 0, 6
        if referer:
            session.headers.update({"Referer": referer})
        while hops < max_hops:
            sleep_a_bit()
            log(f"REQ → {url} (Ref={session.headers.get('Referer','-')})")
            try:
                r = session.get(url, allow_redirects=False, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as e:
                log(f"EXC {repr(e)}")
                return None
            log(f"RESP ← {r.status_code} len={len(r.text or '')}")
            if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("Location", "")
                next_url = _same_domain_redirect_allowed(url, loc)
                if not next_url:
                    log(f"ABORT: redirect to foreign domain: {loc!r}")
                    return None
                url = next_url
                hops += 1
                continue
            if r.status_code == 200 and r.text and "Attention Required" not in r.text:
                return r.text
            if r.status_code == 429:
                log("429 Too Many Requests → waiting 60s")
                time.sleep(60); continue
            if r.status_code in (500, 502, 503, 504):
                log(f"{r.status_code} → retrying after 3s")
                time.sleep(3); continue
            log(f"FAIL status={r.status_code}")
            return None
        log("Too many redirects")
        return None
    finally:
        session.headers.pop("Referer", None)

def load_proxies() -> List[Optional[str]]:
    def _read(p: Path) -> List[str]:
        if not p.exists(): return []
        return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    seen, merged = set(), []
    for x in _read(GOOD_PROXIES_BACKUP) + _read(GOOD_PROXIES_FILE):
        if x not in seen:
            seen.add(x); merged.append(x)
    return merged + [None]

def get_text(soup: BeautifulSoup, sel: str) -> str:
    el = soup.select_one(sel)
    return el.get_text(strip=True) if el else ""

def find_event_urls(index_html: str) -> List[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    urls, seen = [], set()
    for a in soup.select('a[href*="event-details"], a[href^="/event-details/"]'):
        href = (a.get("href") or "").strip()
        if not href: continue
        href = force_http(urllib.parse.urljoin(BASE + "/", href))
        m = re.search(r"event-details/([0-9a-f]+)", href, flags=re.I)
        key = m.group(1) if m else href
        if key in seen: continue
        seen.add(key); urls.append(href)
    log(f"Events found: {len(urls)}")
    return urls

def find_fight_urls(event_html: str) -> List[str]:
    soup = BeautifulSoup(event_html, "html.parser")
    urls, seen = [], set()
    for a in soup.select('a[href*="fight-details"], a[href^="/fight-details/"]'):
        href = (a.get("href") or "").strip()
        if not href: continue
        href = force_http(urllib.parse.urljoin(BASE + "/", href))
        m = re.search(r"fight-details/([0-9a-f]{16})", href, flags=re.I)
        key = m.group(1) if m else href
        if key in seen: continue
        seen.add(key); urls.append(href)
    log(f"Fights found: {len(urls)}")
    return urls

def parse_event_date(event_html: str) -> str:
    soup = BeautifulSoup(event_html, "html.parser")
    for li in soup.select(".b-list__box-list-item"):
        t = li.get_text(" ", strip=True)
        if re.match(r"^Date:\s*", t, flags=re.I):
            raw = re.sub(r"^Date:\s*", "", t, flags=re.I).strip()
            for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
                try: return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
                except Exception: pass
    return ""

def _pair_texts(col) -> Tuple[str, str]:
    ps = [p.get_text(strip=True) for p in col.select("p")]
    return (ps[0], ps[1]) if len(ps) >= 2 else (col.get_text(strip=True), "")

def find_totals_table(soup: BeautifulSoup):
    hdr = soup.find(lambda t: t.name in ("p","h","div") and re.search(r"\bTotals\b", t.get_text(" ", strip=True), re.I))
    if hdr:
        tbl = hdr.find_next("table")
        if tbl:
            if not tbl.find(class_=re.compile(r"b-fight-details__table-row_type_head")):
                ths = {th.get_text(strip=True) for th in tbl.select("thead th")}
                needed = {"KD","Sig. str.","Sig. str. %","Total str.","Td","Td %","Sub. att","Rev.","Ctrl"}
                if needed.issubset(ths) or (needed - {"Td"}).issubset(ths):
                    return tbl
    for tbl in soup.select("table"):
        if tbl.find(class_=re.compile(r"b-fight-details__table-row_type_head")): continue
        ths = {th.get_text(strip=True) for th in tbl.select("thead th")}
        needed = {"KD","Sig. str.","Sig. str. %","Total str.","Td","Td %","Sub. att","Rev.","Ctrl"}
        if needed.issubset(ths) or (needed - {"Td"}).issubset(ths):
            return tbl
    return None

def parse_fight_to_record(fight_html: str, event_name: str, event_date: str) -> Dict[str, object]:
    soup = BeautifulSoup(fight_html, "html.parser")
    a, b = [x.get_text(strip=True) for x in soup.select(".b-fight-details__person-name a")[:2]]
    wl = [i.get_text(strip=True) for i in soup.select(".b-fight-details__person-status")]
    winner = a if (len(wl) >= 1 and wl[0] == "W") else (b if (len(wl) >= 2 and wl[1] == "W") else "")

    meta = {}
    mb = soup.select_one(".b-fight-details__content")
    if mb:
        for lab in mb.select(".b-fight-details__label"):
            k = lab.get_text(strip=True).rstrip(":")
            txt = lab.find_parent().get_text(" ", strip=True)
            m = re.search(rf"{re.escape(k)}:\s*(.+?)(?:\s{{2,}}|$)", txt)
            v = m.group(1).strip() if m else ""
            if k.lower() == "referee":
                sp = mb.select_one(".b-fight-details__text-item span")
                if sp: v = sp.get_text(strip=True)
            meta[k] = v

    totals = {"KD": ("", ""), "SIG": ("", ""), "SIG_PCT": ("", ""), "TOTAL": ("", ""), "TD": ("", ""), "TD_PCT": ("", ""), "SUB_ATT": ("", ""), "REV": ("", ""), "CTRL": ("", "")}
    tbl = find_totals_table(soup)
    if tbl:
        row = tbl.select_one("tbody tr")
        if row:
            cols = row.select(".b-fight-details__table-col")
            def pair(i): return _pair_texts(cols[i]) if i < len(cols) else ("", "")
            totals = {
                "KD": pair(1), "SIG": pair(2), "SIG_PCT": pair(3), "TOTAL": pair(4),
                "TD": pair(5), "TD_PCT": pair(6), "SUB_ATT": pair(7), "REV": pair(8), "CTRL": pair(9),
            }

    return {"event": event_name, "event_date": event_date, "fighter_a": a, "fighter_b": b, "winner": winner, "method": meta.get("Method", ""), "end_round": meta.get("Round", ""), "end_time": meta.get("Time", ""), "referee": meta.get("Referee", ""), "time_format": meta.get("Time format", ""), "a_kd": totals["KD"][0], "b_kd": totals["KD"][1]}

def fetch_index_html_with_rotation(proxy_list):
    for proxy in proxy_list:
        log(f"Index loading via {proxy or 'DIRECT'}")
        s = build_session(proxy)
        html = http_get(EVENTS_INDEX, s, referer=BASE + "/")
        if html:
            return html, s, proxy
        log("Index failed → trying next proxy")
    return None, None, None

def main():
    proxy_list = load_proxies() or [None]
    idx_html, s_idx, idx_proxy = fetch_index_html_with_rotation(proxy_list)
    if not idx_html:
        log("All proxies failed → no network access?")
        return
    ev_urls = find_event_urls(idx_html)
    if not ev_urls:
        log("No event links found")
        return
    print("Scraping complete (educational example only).")

if __name__ == "__main__":
    main()
