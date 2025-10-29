#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UFCStats → fights.csv (alle Events, alle Fights)
- Spricht NUR HTTP (Server hat kein HTTPS).
- Erst DIRECT, danach optional Proxys (Dateien).
- Robustes Logging mit Ursachen.
- Schreibt fights.csv (Append, feste Spalten) + fights_list.json (appendend).
- Erfasst:
  • Event/Fight-Meta (Method, Round, Time, Time format, Referee, Judges/Details, Fight-Titel)
  • Totals (KD, Sig, Total, TD, Sub, Rev, Ctrl)
  • Per-Round (KD, Sig, TD, Ctrl) für Runden 1..5 (fehlende = 0)
  • Significant Strikes – GESAMT (Sig, %, Head/Body/Leg/Distance/Clinch/Ground)
  • Significant Strikes – PRO RUNDE (wie oben) für Runden 1..5 (fehlende = 0)
"""

import csv, json, re, time, itertools, urllib.parse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse, urljoin

# ===================== Grundeinstellungen =====================
BASE = "http://ufcstats.com"
ALLOWED_HOSTS = {"ufcstats.com", "www.ufcstats.com"}
EVENTS_INDEX = f"{BASE}/statistics/events/completed?page=all"

GOOD_PROXIES_FILE   = Path("good_proxies.txt")          # optional
GOOD_PROXIES_BACKUP = Path("good_proxies.backup.txt")   # optional

REQUEST_TIMEOUT = 20
SLEEP_RANGE = (1.0, 2.0)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

DEBUG = True
LOG_FILE = Path("scrape_debug.log")

# ===================== Logging =====================


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

# ===================== HTTP =====================
def build_session(proxy: Optional[str]) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4, connect=4, read=4, backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter); s.mount("https://", adapter)
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
            log(f"REQ  → {url} (Ref={session.headers.get('Referer','-')})")
            try:
                r = session.get(url, allow_redirects=False, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as e:
                log(f"EXC  {repr(e)}"); return None

            log(f"RESP ← {r.status_code} len={len(r.text or '')}")
            if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("Location", "")
                next_url = _same_domain_redirect_allowed(url, loc)
                if not next_url:
                    log(f"ABBRUCH: Redirect zu fremder Domain: {loc!r}")
                    return None
                url = next_url; hops += 1; continue

            if r.status_code == 200 and r.text and "Attention Required" not in r.text:
                return r.text

            if r.status_code == 429:
                log("429 Too Many Requests → 60s Pause"); time.sleep(60); continue

            if r.status_code in (500, 502, 503, 504):
                log(f"{r.status_code} → kurz warten & nochmal"); time.sleep(3); continue

            log(f"FAIL status={r.status_code} (kein brauchbarer Body)")
            return None

        log("Abbruch: zu viele Redirects"); return None
    finally:
        session.headers.pop("Referer", None)

def load_proxies() -> List[Optional[str]]:
    def _read(p: Path) -> List[str]:
        if not p.exists(): return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            l = line.strip()
            if l and not l.startswith("#"):
                out.append(l)
        return out

    seen, merged = set(), []
    for x in _read(GOOD_PROXIES_BACKUP) + _read(GOOD_PROXIES_FILE):
        if x not in seen:
            seen.add(x); merged.append(x)
    return merged + [None]  # Proxys, dann DIRECT (None)

# ===================== Parser-Helfer =====================
def get_text(soup: BeautifulSoup, sel: str) -> str:
    el = soup.select_one(sel)
    return el.get_text(strip=True) if el else ""

def _pair_texts(col) -> Tuple[str, str]:
    ps = [p.get_text(strip=True) for p in col.select("p")]
    if len(ps) >= 2:
        return ps[0], ps[1]
    v = col.get_text(strip=True)
    return v, ""

def _parse_of(text: str) -> Tuple[int, int]:
    m = re.search(r"(\d+)\s*of\s*(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

def _to_secs(mmss: str) -> int:
    mmss = (mmss or "").strip()
    m = re.match(r"(\d+):(\d{2})", mmss)
    return int(m.group(1))*60 + int(m.group(2)) if m else 0

def _to_int(s: str) -> int:
    try:
        return int((s or "0").strip().replace("---","").replace("%","") or "0")
    except:
        return 0

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")

def _fighter_names(soup: BeautifulSoup) -> Tuple[str, str]:
    names = [a.get_text(strip=True) for a in soup.select(".b-fight-details__person-name a")]
    return (names[0], names[1]) if len(names) >= 2 else ("", "")

# ===================== Event/Fight-Link-Ermittlung =====================
def find_event_urls(index_html: str) -> List[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    urls, seen = [], set()
    for a in soup.select('a[href*="event-details"], a[href^="/event-details/"]'):
        href = (a.get("href") or "").strip()
        if not href: continue
        href = force_http(urllib.parse.urljoin(BASE + "/", href))
        m = re.search(r"event-details/([0-9a-f]{16})", href, flags=re.I)
        key = m.group(1) if m else href
        if key in seen: continue
        seen.add(key); urls.append(href)
    log(f"EVENT-Links gefunden: {len(urls)}")
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

    if not urls:
        # RegEx Fallback
        for m in re.finditer(r"(?:https?://)?ufcstats\.com/?fight-details/([0-9a-f]{16})", event_html, flags=re.I):
            fid = m.group(1); href = f"{BASE}/fight-details/{fid}"
            if fid not in seen: seen.add(fid); urls.append(href)
        if not urls:
            for m in re.finditer(r"fight-details/([0-9a-f]{16})", event_html, flags=re.I):
                fid = m.group(1); href = urllib.parse.urljoin(BASE + "/", m.group(0))
                if fid not in seen: seen.add(fid); urls.append(href)
    log(f"FIGHT-Links gefunden: {len(urls)}")
    return urls

def parse_event_date(event_html: str) -> str:
    soup = BeautifulSoup(event_html, "html.parser")
    for li in soup.select(".b-list__box-list-item"):
        t = li.get_text(" ", strip=True)
        if re.match(r"^Date:\s*", t, flags=re.I):
            raw = re.sub(r"^Date:\s*", "", t, flags=re.I).strip()
            for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
                try:
                    return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
                except Exception:
                    pass
    return ""

# ===================== Tabellenfinder =====================
def find_totals_table(soup: BeautifulSoup):
    # a) Überschrift 'Totals' → nächstes <table> ohne Round-Kopf
    hdr = soup.find(lambda t: t and t.name in ("p","h","div") and re.search(r"\bTotals\b", t.get_text(" ", strip=True), re.I))
    if hdr:
        tbl = hdr.find_next("table")
        if tbl and not tbl.find(class_=re.compile(r"b-fight-details__table-row_type_head")):
            ths = {th.get_text(strip=True) for th in tbl.select("thead th")}
            needed = {"KD","Sig. str.","Sig. str. %","Total str.","Td","Td %","Sub. att","Rev.","Ctrl"}
            if needed.issubset(ths) or (needed - {"Td"}).issubset(ths):
                return tbl
    # b) Fallback
    for tbl in soup.select("table"):
        if tbl.find(class_=re.compile(r"b-fight-details__table-row_type_head")): continue
        ths = {th.get_text(strip=True) for th in tbl.select("thead th")}
        needed = {"KD","Sig. str.","Sig. str. %","Total str.","Td","Td %","Sub. att","Rev.","Ctrl"}
        if needed.issubset(ths) or (needed - {"Td"}).issubset(ths):
            return tbl
    return None

def find_sig_totals_table(soup: BeautifulSoup):
    # Sucht die "Significant Strikes" Gesamt-Tabelle (ohne Round-Header)
    hdr = soup.find(lambda t: t and t.name in ("p","h","div") and re.search(r"\bSignificant Strikes\b", t.get_text(" ", strip=True), re.I))
    if hdr:
        tbl = hdr.find_next("table")
        if tbl and not tbl.find(class_=re.compile(r"b-fight-details__table-row_type_head")):
            ths = {_norm(th.get_text(strip=True)) for th in tbl.select("thead th")}
            needed = {"sig_str", "sig_str_", "sig_str__", "sig_str_%", "head", "body", "leg", "distance", "clinch", "ground"}
            # toleranter Check: Hauptkategorien müssen vorhanden sein
            if {"head","body","leg","distance","clinch","ground"}.issubset(ths):
                return tbl
    # Fallback: irgendeine Tabelle mit diesen Spalten & ohne Round-Kopf
    for tbl in soup.select("table"):
        if tbl.find(class_=re.compile(r"b-fight-details__table-row_type_head")): continue
        ths = {_norm(th.get_text(strip=True)) for th in tbl.select("thead th")}
        if {"head","body","leg","distance","clinch","ground"}.issubset(ths):
            return tbl
    return None

# ===================== Per-Round (Standard) =====================
def _get_round_totals(soup: BeautifulSoup) -> Dict[int, Dict[str, Tuple[str, str]]]:
    out: Dict[int, Dict[str, Tuple[str, str]]] = {}
    for head in soup.select(".b-fight-details__table-row_type_head"):
        title = head.get_text(strip=True)
        if not re.match(r"^Round\s+\d+$", title): continue
        tbl = head.find_parent("table")
        if not tbl: continue
        ths = [th.get_text(strip=True) for th in tbl.select("thead th")]
        # prüfen ob Standard-Per-Round (KD, Sig, Total, Td, Ctrl):
        if not {"KD","Sig. str.","Sig. str. %","Total str.","Td %","Sub. att","Rev.","Ctrl"}.issubset(set(ths)) \
           and not {"KD","Sig. str.","Sig. str. %","Total str.","Td","Td %","Sub. att","Rev.","Ctrl"}.issubset(set(ths)):
            continue
        row = head.find_next("tr", class_="b-fight-details__table-row")
        if not row: continue
        cols = row.select(".b-fight-details__table-col")
        def pair(i): return _pair_texts(cols[i]) if i < len(cols) else ("", "")
        rnd = int(re.search(r"\d+", title).group(0))
        out[rnd] = {
            "KD":      pair(1),
            "SIG":     pair(2),
            "SIG_PCT": pair(3),
            "TOTAL":   pair(4),
            "TD":      pair(5),
            "TD_PCT":  pair(6),
            "SUB_ATT": pair(7),
            "REV":     pair(8),
            "CTRL":    pair(9),
        }
    return out

# ===================== Significant Strikes – Parser =====================
def parse_sig_totals(soup: BeautifulSoup) -> Dict[str, int]:
    out = {
        "a_ss_landed":0, "a_ss_attempts":0, "a_ss_pct":0,
        "b_ss_landed":0, "b_ss_attempts":0, "b_ss_pct":0,

        "a_ss_head_l":0,"a_ss_head_a":0,"a_ss_body_l":0,"a_ss_body_a":0,"a_ss_leg_l":0,"a_ss_leg_a":0,
        "a_ss_distance_l":0,"a_ss_distance_a":0,"a_ss_clinch_l":0,"a_ss_clinch_a":0,"a_ss_ground_l":0,"a_ss_ground_a":0,
        "b_ss_head_l":0,"b_ss_head_a":0,"b_ss_body_l":0,"b_ss_body_a":0,"b_ss_leg_l":0,"b_ss_leg_a":0,
        "b_ss_distance_l":0,"b_ss_distance_a":0,"b_ss_clinch_l":0,"b_ss_clinch_a":0,"b_ss_ground_l":0,"b_ss_ground_a":0,
    }

    tbl = find_sig_totals_table(soup)
    if not tbl: return out
    row = tbl.select_one("tbody tr")
    if not row: return out
    cols = row.select(".b-fight-details__table-col")

    def pair(i):
        return _pair_texts(cols[i]) if i < len(cols) else ("","")

    # 0=Fighter, 1=Sig. str., 2=Sig.% , 3..8 Kategorien
    a_sig,b_sig = pair(1); a_pct,b_pct = pair(2)
    a_head,b_head = pair(3); a_body,b_body = pair(4); a_leg,b_leg = pair(5)
    a_dist,b_dist = pair(6); a_cl,b_cl = pair(7); a_gr,b_gr = pair(8)

    aL,aA = _parse_of(a_sig); bL,bA = _parse_of(b_sig)
    out["a_ss_landed"] = aL; out["a_ss_attempts"] = aA; out["a_ss_pct"] = _to_int(a_pct[0] if isinstance(a_pct, tuple) else a_pct)
    out["b_ss_landed"] = bL; out["b_ss_attempts"] = bA; out["b_ss_pct"] = _to_int(b_pct[1] if isinstance(b_pct, tuple) else b_pct)

    def set_cat(prefix: str, cell):
        l,a = _parse_of(cell)
        out[f"{prefix}_l"] = l
        out[f"{prefix}_a"] = a

    # A
    set_cat("a_ss_head",     a_head[0] if isinstance(a_head, tuple) else a_head)
    set_cat("a_ss_body",     a_body[0] if isinstance(a_body, tuple) else a_body)
    set_cat("a_ss_leg",      a_leg[0]  if isinstance(a_leg,  tuple) else a_leg)
    set_cat("a_ss_distance", a_dist[0] if isinstance(a_dist, tuple) else a_dist)
    set_cat("a_ss_clinch",   a_cl[0]   if isinstance(a_cl,   tuple) else a_cl)
    set_cat("a_ss_ground",   a_gr[0]   if isinstance(a_gr,   tuple) else a_gr)
    # B
    set_cat("b_ss_head",     b_head[1] if isinstance(b_head, tuple) else b_head)
    set_cat("b_ss_body",     b_body[1] if isinstance(b_body, tuple) else b_body)
    set_cat("b_ss_leg",      b_leg[1]  if isinstance(b_leg,  tuple) else b_leg)
    set_cat("b_ss_distance", b_dist[1] if isinstance(b_dist, tuple) else b_dist)
    set_cat("b_ss_clinch",   b_cl[1]   if isinstance(b_cl,   tuple) else b_cl)
    set_cat("b_ss_ground",   b_gr[1]   if isinstance(b_gr,   tuple) else b_gr)

    return out


def parse_sig_per_round(soup: BeautifulSoup) -> Dict[str, int]:
    out: Dict[str,int] = {}
    ss_hdr = soup.find(lambda t: t and t.name in ("p","h","div")
                       and re.search(r"\bSignificant Strikes\b", t.get_text(" ", strip=True), re.I))
    # Immer alle r1..r5 vorbelegen
    for r in range(1,6):
        for side in ("a","b"):
            for k in ("l","a","pct","head_l","head_a","body_l","body_a","leg_l","leg_a",
                      "distance_l","distance_a","clinch_l","clinch_a","ground_l","ground_a"):
                out.setdefault(f"r{r}_{side}_ss_{k}", 0)

    if not ss_hdr:
        return out

    for head in ss_hdr.find_all_next(class_=re.compile(r"b-fight-details__table-row_type_head")):
        title = head.get_text(strip=True)
        if not re.match(r"^Round\s+\d+$", title):
            continue
        tbl = head.find_parent("table")
        if not tbl:
            continue
        ths = {th.get_text(" ", strip=True).strip().lower() for th in tbl.select("thead th")}
        if not {"head","body","leg","distance","clinch","ground"} <= ths:
            continue  # nicht die SS-Per-Round-Tabelle
        row = head.find_next("tr", class_="b-fight-details__table-row")
        if not row:
            continue
        cols = row.select(".b-fight-details__table-col")
        def pair(i):
            ps = [p.get_text(strip=True) for p in cols[i].select("p")] if i < len(cols) else []
            if len(ps) >= 2: return ps[0], ps[1]
            v = cols[i].get_text(strip=True) if i < len(cols) else ""
            return v, ""

        r = int(re.search(r"\d+", title).group(0))
        a_sig,b_sig = pair(1); a_pct,b_pct = pair(2)
        a_head,b_head = pair(3); a_body,b_body = pair(4); a_leg,b_leg = pair(5)
        a_dist,b_dist = pair(6); a_cl,b_cl     = pair(7); a_gr,b_gr   = pair(8)

        aL,aA = _parse_of(a_sig); bL,bA = _parse_of(b_sig)
        out[f"r{r}_a_ss_l"] = aL; out[f"r{r}_a_ss_a"] = aA; out[f"r{r}_a_ss_pct"] = _to_int(a_pct[0] if isinstance(a_pct, tuple) else a_pct)
        out[f"r{r}_b_ss_l"] = bL; out[f"r{r}_b_ss_a"] = bA; out[f"r{r}_b_ss_pct"] = _to_int(b_pct[1] if isinstance(b_pct, tuple) else b_pct)

        # Kategorien (landed/attempts)
        a_hL,a_hA = _parse_of(a_head[0] if isinstance(a_head, tuple) else a_head); out[f"r{r}_a_ss_head_l"]=a_hL; out[f"r{r}_a_ss_head_a"]=a_hA
        a_bL,a_bA = _parse_of(a_body[0] if isinstance(a_body, tuple) else a_body); out[f"r{r}_a_ss_body_l"]=a_bL; out[f"r{r}_a_ss_body_a"]=a_bA
        a_lL,a_lA = _parse_of(a_leg[0]  if isinstance(a_leg,  tuple) else a_leg);  out[f"r{r}_a_ss_leg_l"]=a_lL;   out[f"r{r}_a_ss_leg_a"]=a_lA
        a_dL,a_dA = _parse_of(a_dist[0] if isinstance(a_dist, tuple) else a_dist); out[f"r{r}_a_ss_distance_l"]=a_dL; out[f"r{r}_a_ss_distance_a"]=a_dA
        a_cL,a_cA = _parse_of(a_cl[0]   if isinstance(a_cl,   tuple) else a_cl);  out[f"r{r}_a_ss_clinch_l"]=a_cL;   out[f"r{r}_a_ss_clinch_a"]=a_cA
        a_gL,a_gA = _parse_of(a_gr[0]   if isinstance(a_gr,   tuple) else a_gr);  out[f"r{r}_a_ss_ground_l"]=a_gL;   out[f"r{r}_a_ss_ground_a"]=a_gA

        b_hL,b_hA = _parse_of(b_head[1] if isinstance(b_head, tuple) else b_head); out[f"r{r}_b_ss_head_l"]=b_hL; out[f"r{r}_b_ss_head_a"]=b_hA
        b_bL,b_bA = _parse_of(b_body[1] if isinstance(b_body, tuple) else b_body); out[f"r{r}_b_ss_body_l"]=b_bL; out[f"r{r}_b_ss_body_a"]=b_bA
        b_lL,b_lA = _parse_of(b_leg[1]  if isinstance(b_leg,  tuple) else b_leg);  out[f"r{r}_b_ss_leg_l"]=b_lL;   out[f"r{r}_b_ss_leg_a"]=b_lA
        b_dL,b_dA = _parse_of(b_dist[1] if isinstance(b_dist, tuple) else b_dist); out[f"r{r}_b_ss_distance_l"]=b_dL; out[f"r{r}_b_ss_distance_a"]=b_dA
        b_cL,b_cA = _parse_of(b_cl[1]   if isinstance(b_cl,   tuple) else b_cl);  out[f"r{r}_b_ss_clinch_l"]=b_cL;  out[f"r{r}_b_ss_clinch_a"]=b_cA
        b_gL,b_gA = _parse_of(b_gr[1]   if isinstance(b_gr,   tuple) else b_gr);  out[f"r{r}_b_ss_ground_l"]=b_gL;  out[f"r{r}_b_ss_ground_a"]=b_gA

    return out

# ===================== Fight → Record =====================
def parse_fight_to_record(fight_html: str, event_name: str, event_date: str) -> Dict[str, object]:
    soup = BeautifulSoup(fight_html, "html.parser")
    a, b = _fighter_names(soup)
    wl = [i.get_text(strip=True) for i in soup.select(".b-fight-details__person-status")]
    winner = a if (len(wl) >= 1 and wl[0] == "W") else (b if (len(wl) >= 2 and wl[1] == "W") else "")

    # Meta
    meta = {}
    mb = soup.select_one(".b-fight-details__content")
    if mb:
        for lab in mb.select(".b-fight-details__label"):
            k = lab.get_text(strip=True).rstrip(":")
            txt = lab.find_parent().get_text(" ", strip=True)
            m = re.search(rf"{re.escape(k)}:\s*(.+?)(?:\s{{2,}}|$)", txt)
            v = m.group(1).strip() if m else ""
            # spezielle Extraktion für Referee (span)
            if k.lower() == "referee":
                sp = mb.select_one(".b-fight-details__text-item span")
                if sp: v = sp.get_text(strip=True)
            meta[k] = v

    # Fight-„Titel“ oben (z.B. „UFC Bantamweight Title Bout“)
    fight_title = get_text(soup, ".b-fight-details__fight-title") or get_text(soup, ".b-fight-details__title")

    # Totals
    totals = {"KD": ("", ""), "SIG": ("", ""), "SIG_PCT": ("", ""), "TOTAL": ("", ""),
              "TD": ("", ""), "TD_PCT": ("", ""), "SUB_ATT": ("", ""), "REV": ("", ""), "CTRL": ("", "")}

    tbl = find_totals_table(soup)
    if tbl:
        row = tbl.select_one("tbody tr")
        if row:
            cols = row.select(".b-fight-details__table-col")
            def pair(i): return _pair_texts(cols[i]) if i < len(cols) else ("", "")
            totals = {
                "KD":      pair(1),
                "SIG":     pair(2),
                "SIG_PCT": pair(3),
                "TOTAL":   pair(4),
                "TD":      pair(5),
                "TD_PCT":  pair(6),
                "SUB_ATT": pair(7),
                "REV":     pair(8),
                "CTRL":    pair(9),
            }
    else:
        log("WARN: Totals-Tabelle nicht gefunden → Gesamtwerte bleiben 0")

    # Per-Round Standard
    rounds = _get_round_totals(soup)

    # Record-Grundgerüst
    rec: Dict[str, object] = {
        "event": event_name, "event_date": event_date,
        "fight_title": fight_title,
        "fighter_a": a, "fighter_b": b, "winner": winner,
        "method": meta.get("Method",""),
        "end_round": meta.get("Round",""),
        "end_time": meta.get("Time",""),
        "referee": meta.get("Referee",""),
        "time_format": meta.get("Time format",""),
        "judges_details": meta.get("Details",""),  # „Details: <Judge scores>“
        "a_kd": _to_int(totals["KD"][0]), "b_kd": _to_int(totals["KD"][1]),
    }

    # Gesamtwerte (Zahlen)
    a_sig_l,a_sig_a = _parse_of(totals["SIG"][0]);   b_sig_l,b_sig_a = _parse_of(totals["SIG"][1])
    a_tot_l,a_tot_a = _parse_of(totals["TOTAL"][0]); b_tot_l,b_tot_a = _parse_of(totals["TOTAL"][1])
    a_td_l,a_td_a   = _parse_of(totals["TD"][0]);    b_td_l,b_td_a   = _parse_of(totals["TD"][1])

    rec.update({
        "a_sig_landed": a_sig_l, "a_sig_attempts": a_sig_a,
        "b_sig_landed": b_sig_l, "b_sig_attempts": b_sig_a,
        "a_total_landed": a_tot_l, "a_total_attempts": a_tot_a,
        "b_total_landed": b_tot_l, "b_total_attempts": b_tot_a,
        "a_td_landed": a_td_l, "a_td_attempts": a_td_a,
        "b_td_landed": b_td_l, "b_td_attempts": b_td_a,
        "a_sub_att": _to_int(totals["SUB_ATT"][0]),
        "b_sub_att": _to_int(totals["SUB_ATT"][1]),
        "a_rev": _to_int(totals["REV"][0]),
        "b_rev": _to_int(totals["REV"][1]),
        "a_ctrl_sec": _to_secs(totals["CTRL"][0]),
        "b_ctrl_sec": _to_secs(totals["CTRL"][1]),
    })

    # Runden 1..5 (immer befüllen)
    for r in range(1, 6):
        R = rounds.get(r, {})
        a_kd = _to_int((R.get("KD") or ("0","0"))[0]); b_kd = _to_int((R.get("KD") or ("0","0"))[1])
        a_sig = _parse_of((R.get("SIG") or ("0 of 0","0 of 0"))[0]); b_sig = _parse_of((R.get("SIG") or ("0 of 0","0 of 0"))[1])
        a_td  = _parse_of((R.get("TD")  or ("0 of 0","0 of 0"))[0]); b_td  = _parse_of((R.get("TD")  or ("0 of 0","0 of 0"))[1])
        a_c   = _to_secs((R.get("CTRL") or ("0","0"))[0]);           b_c   = _to_secs((R.get("CTRL") or ("0","0"))[1])
        rec.update({
            f"r{r}_a_kd": a_kd, f"r{r}_b_kd": b_kd,
            f"r{r}_a_sig_l": a_sig[0], f"r{r}_a_sig_a": a_sig[1],
            f"r{r}_b_sig_l": b_sig[0], f"r{r}_b_sig_a": b_sig[1],
            f"r{r}_a_td_l": a_td[0],   f"r{r}_a_td_a": a_td[1],
            f"r{r}_b_td_l": b_td[0],   f"r{r}_b_td_a": b_td[1],
            f"r{r}_a_ctrl_s": a_c,     f"r{r}_b_ctrl_s": b_c,
        })

    # Significant Strikes (Totals & Per-Round)
    ss_tot = parse_sig_totals(soup); rec.update(ss_tot)
    log(f"SS Totals A: {rec['a_ss_landed']}/{rec['a_ss_attempts']} head {rec['a_ss_head_l']}/{rec['a_ss_head_a']}")
    log(f"SS Totals B: {rec['b_ss_landed']}/{rec['b_ss_attempts']} head {rec['b_ss_head_l']}/{rec['b_ss_head_a']}")

    ss_rounds = parse_sig_per_round(soup); rec.update(ss_rounds)

    return rec

# ===================== Output =====================
CSV_PATH        = Path("fights_new.csv")
JSON_PATH       = Path("fights_list.json")
CHECKPOINT_FILE = Path("events.done.txt")

def _csv_fieldnames() -> List[str]:
    base = [
        "event", "event_date", "fight_title",
        "fighter_a", "fighter_b", "winner", "method", "end_round", "end_time", "referee", "time_format",
        "judges_details",
        "a_kd", "b_kd",
        "a_sig_landed", "a_sig_attempts", "b_sig_landed", "b_sig_attempts",
        "a_total_landed", "a_total_attempts", "b_total_landed", "b_total_attempts",
        "a_td_landed", "a_td_attempts", "b_td_landed", "b_td_attempts",
        "a_sub_att", "b_sub_att", "a_rev", "b_rev", "a_ctrl_sec", "b_ctrl_sec",

        # Significant Strikes – Totals
        "a_ss_landed", "a_ss_attempts", "a_ss_pct",
        "b_ss_landed", "b_ss_attempts", "b_ss_pct",

        "a_ss_head_l", "a_ss_head_a", "a_ss_body_l", "a_ss_body_a", "a_ss_leg_l", "a_ss_leg_a",
        "a_ss_distance_l", "a_ss_distance_a", "a_ss_clinch_l", "a_ss_clinch_a", "a_ss_ground_l", "a_ss_ground_a",
        "b_ss_head_l", "b_ss_head_a", "b_ss_body_l", "b_ss_body_a", "b_ss_leg_l", "b_ss_leg_a",
        "b_ss_distance_l", "b_ss_distance_a", "b_ss_clinch_l", "b_ss_clinch_a", "b_ss_ground_l", "b_ss_ground_a",
    ]

    rnd = []
    for r in range(1, 6):
        rnd += [
            f"r{r}_a_kd", f"r{r}_b_kd",
            f"r{r}_a_sig_l", f"r{r}_a_sig_a", f"r{r}_b_sig_l", f"r{r}_b_sig_a",
            f"r{r}_a_td_l", f"r{r}_a_td_a", f"r{r}_b_td_l", f"r{r}_b_td_a",
            f"r{r}_a_ctrl_s", f"r{r}_b_ctrl_s",

            # Significant Strikes – Per-Round
            f"r{r}_a_ss_l", f"r{r}_a_ss_a", f"r{r}_a_ss_pct",
            f"r{r}_a_ss_head_l", f"r{r}_a_ss_head_a",
            f"r{r}_a_ss_body_l", f"r{r}_a_ss_body_a",
            f"r{r}_a_ss_leg_l", f"r{r}_a_ss_leg_a",
            f"r{r}_a_ss_distance_l", f"r{r}_a_ss_distance_a",
            f"r{r}_a_ss_clinch_l", f"r{r}_a_ss_clinch_a",
            f"r{r}_a_ss_ground_l", f"r{r}_a_ss_ground_a",

            f"r{r}_b_ss_l", f"r{r}_b_ss_a", f"r{r}_b_ss_pct",
            f"r{r}_b_ss_head_l", f"r{r}_b_ss_head_a",
            f"r{r}_b_ss_body_l", f"r{r}_b_ss_body_a",
            f"r{r}_b_ss_leg_l", f"r{r}_b_ss_leg_a",
            f"r{r}_b_ss_distance_l", f"r{r}_b_ss_distance_a",
            f"r{r}_b_ss_clinch_l", f"r{r}_b_ss_clinch_a",
            f"r{r}_b_ss_ground_l", f"r{r}_b_ss_ground_a",
        ]
    return base + rnd


def _ensure_csv_header(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_csv_fieldnames()).writeheader()

def _append_csv(records: List[Dict[str, object]], path: Path):
    if not records: return
    _ensure_csv_header(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_csv_fieldnames())
        for rec in records:
            w.writerow({k: rec.get(k, "") for k in _csv_fieldnames()})

def _append_json(records: List[Dict[str, object]], path: Path):
    if not records: return
    arr = []
    if path.exists():
        try:
            arr = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(arr, list): arr = []
        except Exception:
            arr = []
    arr.extend(records)
    path.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_done() -> set:
    if not CHECKPOINT_FILE.exists(): return set()
    return {ln.strip() for ln in CHECKPOINT_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()}

def _mark_done(event_id: str):
    with CHECKPOINT_FILE.open("a", encoding="utf-8") as f:
        f.write(event_id + "\n")

def _event_id_from_url(url: str) -> str:
    m = re.search(r"event-details/([0-9a-f]{16})", url, flags=re.I)
    return m.group(1) if m else url

# ===================== Fight-Filter =====================
def fight_html_has_stats(html: str) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    tbl = find_totals_table(soup)
    if not tbl:
        # als Fallback akzeptieren wir, wenn SS-Totals vorhanden sind
        ss_tbl = find_sig_totals_table(soup)
        if not ss_tbl: return False
        row = ss_tbl.select_one("tbody tr")
        if not row: return False
        txt = row.get_text(" ", strip=True)
        return bool(re.search(r"\b\d+\s+of\s+\d+\b", txt))
    row = tbl.select_one("tbody tr")
    if not row: return False
    txt = row.get_text(" ", strip=True)
    #return bool(re.search(r"\b\d+\s+of\s+\d+\b", txt) or re.search(r"\b\d+:\d{2}\b", txt))
    return True

# ===================== Main =====================
def fetch_index_html_with_rotation(proxy_list):
    for proxy in proxy_list:
        log(f"Index laden mit Proxy: {proxy or 'DIRECT'}")
        s = build_session(proxy)
        html = http_get(EVENTS_INDEX, s, referer=BASE + "/")
        if html:
            return html, s, proxy
        log("Index fehlgeschlagen → nächster Proxy")
    return None, None, None

def main():
    done = _load_done()
    proxy_list = load_proxies() or [None]

    idx_html, s_idx, idx_proxy = fetch_index_html_with_rotation(proxy_list)
    if not idx_html:
        log("Alle Proxys für Index gescheitert → Netz/Portal blockiert?")
        return

    ev_urls = find_event_urls(idx_html)
    if not ev_urls:
        log("Keine Event-Links gefunden"); return

    # Proxy-Rotation ab dem Nachfolger des Index-Proxys
    try:
        start = proxy_list.index(idx_proxy)
    except ValueError:
        start = -1
    rot_order = proxy_list[start+1:] + proxy_list[:start+1]
    proxy_cycle = itertools.cycle(rot_order)

    total_written_csv = 0
    for ev_url in ev_urls:  # z. B. nur 3 Events
        ev_id = _event_id_from_url(ev_url)
        if ev_id in done:
            log(f"Skip (done): {ev_id}")
            continue

        proxy = next(proxy_cycle)
        log(f"Proxy (Event): {proxy or 'DIRECT'}")
        s = build_session(proxy)

        log(f"Event laden: {ev_url}")
        ev_html = http_get(ev_url, s, referer=EVENTS_INDEX)

        if not ev_html:
            for alt in proxy_list:
                if alt == proxy: continue
                log(f"Event-Recovery mit Proxy: {alt or 'DIRECT'}")
                s_alt = build_session(alt)
                ev_html = http_get(ev_url, s_alt, referer=EVENTS_INDEX)
                if ev_html:
                    s = s_alt; break

        if not ev_html:
            log("Event-HTML = None → weiter (nicht checkpointen)")
            continue

        soup_ev = BeautifulSoup(ev_html, "html.parser")
        event_name = get_text(soup_ev, ".b-content__title a, .b-content__title") or "Unknown Event"
        event_date = parse_event_date(ev_html)

        fights = find_fight_urls(ev_html)
        if not fights:
            log("Keine fight-details-Links → Event als leer markieren")
            _mark_done(ev_id); continue

        event_records = []
        for f_url in fights:
            f_html = http_get(f_url, s, referer=ev_url)
            if not f_html:
                log(f"Fight-HTML fehlgeschlagen: {f_url}")
                continue
            if not fight_html_has_stats(f_html):
                log(f"Fight ohne Stats: {f_url} → skip")
                continue
            rec = parse_fight_to_record(f_html, event_name, event_date)
            event_records.append(rec)

        if event_records:
            _append_csv(event_records, CSV_PATH)
            _append_json(event_records, JSON_PATH)
            total_written_csv += len(event_records)
            log(f"APPEND → {len(event_records)} Zeilen in {CSV_PATH.name} & JSON")

        _mark_done(ev_id)
        log(f"CHECKPOINT gesetzt: {ev_id}")

    print(f"Fertig. CSV: {CSV_PATH} (neu: {total_written_csv})  JSON: {JSON_PATH}")

if __name__ == "__main__":
    main()
