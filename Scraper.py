#!/usr/bin/env python3
# scraper.py
"""
DISCLAIMER:
This script is provided purely as a code sample for educational/testing purposes.
It is not intended for real-world use. Do not use it to scrape any website without
explicit permission and compliance with that website's Terms of Service and the law.
"""

import requests
import random
import time
import csv
import re
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from bs4 import BeautifulSoup

# ---------------- Config ----------------
BASE = "<REDACTED_BASE_URL>"  # intentionally redacted
LIST_URL_TEMPLATE = BASE + "/athletes/all?gender=All&search=&page={}"  # path example kept for structure
TARGET_TOTAL_FIGHTERS = 1000
MAX_CONSECUTIVE_FAILS = 10
PROXY_ROTATE_AFTER = 10
REQUEST_TIMEOUT = 20
MIN_DELAY = 0.8
MAX_DELAY = 2.0
CHECK_PROXY_URL = BASE + "/"  # intentionally generic
OUTPUT_CSV = Path("fighters_collected.csv")
GOOD_PROXIES_FILE = Path("good_proxies.txt")
REMOVED_PROXIES_FILE = Path("removed_proxies.txt")
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
]

# Desired CSV column order first, then any extras encountered
PREFERRED_FIELD_ORDER = [
    "Name","Nickname","Weight Class","Status","Record",
    "Last Fight Result","Last Fight Event","Last Fight Date",
    "Knockouts","Wins by Submission","First Round Finishes",
    "Striking accuracy","Takedown Accuracy",
    "Sig. Str. Landed Per Min","Sig. Str. Absorbed Per Min",
    "Takedown avg Per 15 Min","Submission avg Per 15 Min",
    "Sig. Str. Defense","Takedown Defense","Knockdown Avg","Average fight time",
    "Standing (Sig Str)","Clinch (Sig Str)","Ground (Sig Str)",
    "Head (Sig Str)","Body (Sig Str)","Leg (Sig Str)",
    "Sig Str Landed Total","Sig Str Attempted Total",
    "Takedowns Landed Total","Takedowns Attempted Total",
    "Win by KO/TKO","Win by DEC","Win by SUB",
    "ProfileURL",
    # "Last Fight Block",  # keep excluded unless you want raw block text
]


# ---------------- CSV I/O ----------------
def save_checkpoint(rows: List[Dict], path: Path = OUTPUT_CSV) -> None:
    if not rows:
        return
    all_keys = list({k for r in rows for k in r.keys()})
    ordered = [k for k in PREFERRED_FIELD_ORDER if k in all_keys]
    remaining = [k for k in all_keys if k not in ordered]
    fieldnames = ordered + remaining
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------- File helpers ----------------
def load_proxies_from_file(path: Path) -> List[str]:
    if not path.exists():
        print(f"Missing: {path}")
        return []
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l and not l.startswith("#")]

def save_removed_proxies(removed: List[Tuple[str, str]], path: Path = REMOVED_PROXIES_FILE) -> None:
    if not removed:
        return
    with path.open("w", encoding="utf-8") as f:
        for p, reason in removed:
            f.write(f"{p}\t{reason}\n")


# ---------------- Proxy helpers ----------------
def is_valid_proxy_format(p: str) -> bool:
    try:
        parsed = urlparse(p)
        return parsed.scheme != "" and parsed.hostname is not None and parsed.port is not None
    except Exception:
        return False

def build_proxies_dict(proxy_url: str) -> Dict[str, str]:
    return {"http": proxy_url, "https": proxy_url}

def check_proxy(proxy_url: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """Basic health check: can we reach CHECK_PROXY_URL through this proxy?"""
    if not is_valid_proxy_format(proxy_url):
        return False, "invalid_format"
    proxies = build_proxies_dict(proxy_url)
    headers = {"User-Agent": random.choice(USER_AGENTS), "Accept-Language": "en-US,en;q=0.9"}
    try:
        with requests.Session() as s:
            s.proxies.update(proxies)
            s.headers.update(headers)
            r = s.get(CHECK_PROXY_URL, timeout=timeout)
        if r.status_code == 200:
            return True, None
        return False, f"http_{r.status_code}"
    except requests.exceptions.ProxyError as e:
        return False, f"ProxyError:{e}"
    except requests.exceptions.ConnectTimeout:
        return False, "ConnectTimeout"
    except requests.exceptions.ReadTimeout:
        return False, "ReadTimeout"
    except requests.exceptions.SSLError as e:
        return False, f"SSLError:{e}"
    except Exception as e:
        return False, f"Other:{type(e).__name__}:{e}"

def get_session_with_proxy(proxy_url: Optional[str]) -> requests.Session:
    s = requests.Session()
    if proxy_url:
        s.proxies.update(build_proxies_dict(proxy_url))
    s.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    return s


# ---------------- Fetch helpers ----------------
def get_soup(url: str, session: requests.Session, timeout: int = REQUEST_TIMEOUT) -> BeautifulSoup:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def extract_fighter_links_from_list_soup(soup: BeautifulSoup) -> List[str]:
    # Note: selector path is from the original site structure; adjust to your target site.
    links = set()
    for a in soup.select('a[href^="/athlete/"]'):
        href = a.get("href")
        if href:
            links.add(BASE + href.split("?")[0])
    return sorted(links)


# ---------------- Parse helpers ----------------
def _txt(el):
    return el.get_text(strip=True) if el else ""

def _sel_txt(soup: BeautifulSoup, css: str) -> str:
    el = soup.select_one(css)
    return _txt(el) if el else ""

def _svg_txt(soup: BeautifulSoup, svg_id: str) -> str:
    el = soup.find("text", id=svg_id)
    return _txt(el) if el else ""


# ---------------- Profile parser ----------------
def extract_full_fighter_info(soup: BeautifulSoup) -> Dict:
    data = {}

    # Basic
    data["Name"] = _sel_txt(soup, "h1.hero-profile__name")
    data["Nickname"] = _sel_txt(soup, "p.hero-profile__nickname")
    data["Weight Class"] = _sel_txt(soup, "p.hero-profile__division-title")
    division_block = _sel_txt(soup, "div.hero-profile__info") or ""
    if division_block and data["Weight Class"] and division_block != data["Weight Class"]:
        data["Status"] = division_block.replace(data["Weight Class"], "").strip()
    else:
        data["Status"] = ""
    data["Record"] = _sel_txt(soup, "p.hero-profile__division-body")

    # Last fight (heuristic)
    last_fight_wrap = soup.find(lambda tag: tag.name in ("section", "div") and "last fight" in tag.get_text(strip=True).lower())
    if last_fight_wrap:
        text = last_fight_wrap.get_text(" ", strip=True)
        data["Last Fight Block"] = text
        for res in ("Win", "Loss", "Draw", "No Contest"):
            if f" {res} " in f" {text} ":
                data["Last Fight Result"] = res
                break
        m = re.search(r"([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+)?)\s+vs\s+([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+)?)", text)
        if m:
            data["Last Fight Event"] = f"{m.group(1)} vs {m.group(2)}"
        m = re.search(r"(Jan\.|Feb\.|Mar\.|Apr\.|May|Jun\.|Jul\.|Aug\.|Sep\.|Sept\.|Oct\.|Nov\.|Dec\.)\s+\d{1,2},\s+\d{4}", text)
        if m:
            data["Last Fight Date"] = m.group(0)

    # Stat tiles
    tiles = [t.get_text(strip=True).lower() for t in soup.select("p.athlete-stats__text.athlete-stats__stat-text")]
    nums = [n.get_text(strip=True) for n in soup.select("p.athlete-stats__text.athlete-stats__stat-numb")]
    for lab, val in zip(tiles, nums):
        if "knock" in lab:
            data["Knockouts"] = val
        elif "subm" in lab:
            data["Wins by Submission"] = val
        elif "first round" in lab or "finish" in lab:
            data["First Round Finishes"] = val

    # Overlap totals
    labs = [x.get_text(strip=True).lower() for x in soup.select("dt.c-overlap__stats-text")]
    vals = [x.get_text(strip=True) for x in soup.select("dd.c-overlap__stats-value")]
    for lab, val in zip(labs, vals):
        if "strikes landed" in lab:
            data["Sig Str Landed Total"] = val
        elif "strikes attempted" in lab:
            data["Sig Str Attempted Total"] = val
        elif "takedowns landed" in lab:
            data["Takedowns Landed Total"] = val
        elif "takedowns attempted" in lab:
            data["Takedowns Attempted Total"] = val

    # Accuracy
    acc_labels = [x.get_text(strip=True).lower() for x in soup.select("h2.e-t3")]
    acc_vals = [x.get_text(strip=True) for x in soup.select("text.e-chart-circle__percent")]
    for lab, val in zip(acc_labels, acc_vals):
        if "strik" in lab:
            data["Striking accuracy"] = val
        elif "taked" in lab:
            data["Takedown Accuracy"] = val

    # Compare groups
    def parse_compare(group_cls, mapping):
        for grp in soup.select(f"div.c-stat-compare__group.{group_cls}"):
            label = _sel_txt(grp, ".c-stat-compare__label").lower()
            num = _sel_txt(grp, ".c-stat-compare__number") or ""
            for key, needle in mapping.items():
                if needle in label:
                    data[key] = num

    parse_compare("c-stat-compare__group-1", {
        "Sig. Str. Landed Per Min": "str. landed",
        "Takedown avg Per 15 Min": "taked",
        "Sig. Str. Defense": "str. def",
        "Knockdown Avg": "knock",
    })
    parse_compare("c-stat-compare__group-2", {
        "Sig. Str. Absorbed Per Min": "str. abs",
        "Submission avg Per 15 Min": "subm",
        "Takedown Defense": "takedown def",
        "Average fight time": "fight time",
    })

    # 3-bar / Win by
    for grp in soup.select("div.c-stat-3bar__group"):
        label = _sel_txt(grp, ".c-stat-3bar__label").lower()
        val = _sel_txt(grp, ".c-stat-3bar__value").replace(" ", "")
        if "standing" in label:
            data["Standing (Sig Str)"] = val
        elif "clinch" in label:
            data["Clinch (Sig Str)"] = val
        elif "ground" in label:
            data["Ground (Sig Str)"] = val
        elif "tko" in label:
            data["Win by KO/TKO"] = val or "0(0%)"
        elif "dec" in label:
            data["Win by DEC"] = val or "0(0%)"
        elif "sub" in label:
            data["Win by SUB"] = val or "0(0%)"

    # Target breakdown (SVG)
    head_total = _svg_txt(soup, "e-stat-body_x5F__x5F_head_value")
    head_pct = _svg_txt(soup, "e-stat-body_x5F__x5F_head_percent")
    body_total = _svg_txt(soup, "e-stat-body_x5F__x5F_body_value")
    body_pct = _svg_txt(soup, "e-stat-body_x5F__x5F_body_percent")
    leg_total = _svg_txt(soup, "e-stat-body_x5F__x5F_leg_value")
    leg_pct = _svg_txt(soup, "e-stat-body_x5F__x5F_leg_percent")
    if head_total or head_pct:
        data["Head (Sig Str)"] = f"{head_total}({head_pct})".replace("  ", "")
    if body_total or body_pct:
        data["Body (Sig Str)"] = f"{body_total}({body_pct})".replace("  ", "")
    if leg_total or leg_pct:
        data["Leg (Sig Str)"] = f"{leg_total}({leg_pct})".replace("  ", "")

    # Stabilize keys for CSV
    expected = [
        "Name","Nickname","Weight Class","Status","Record",
        "Knockouts","Wins by Submission","First Round Finishes",
        "Striking accuracy","Takedown Accuracy",
        "Sig Str Landed Total","Sig Str Attempted Total",
        "Takedowns Landed Total","Takedowns Attempted Total",
        "Sig. Str. Landed Per Min","Sig. Str. Absorbed Per Min",
        "Takedown avg Per 15 Min","Submission avg Per 15 Min",
        "Sig. Str. Defense","Takedown Defense","Knockdown Avg","Average fight time",
        "Standing (Sig Str)","Clinch (Sig Str)","Ground (Sig Str)",
        "Head (Sig Str)","Body (Sig Str)","Leg (Sig Str)",
        "Win by KO/TKO","Win by DEC","Win by SUB",
        "Last Fight Result","Last Fight Event","Last Fight Date",
    ]
    for k in expected:
        data.setdefault(k, "")

    return data


# ---------------- Scrape flow ----------------
def scrape_with_proxies():
    proxies = load_proxies_from_file(GOOD_PROXIES_FILE)
    proxy_state: Dict[str, int] = {p: 0 for p in proxies if is_valid_proxy_format(p)}
    if not proxy_state:
        print("No valid proxies in good_proxies.txt")
        return

    removed_proxies: List[Tuple[str, str]] = []
    collected: List[Dict] = []
    seen_profiles = set()
    page = 0
    profile_requests_with_current_proxy = 0

    def pick_proxy(exclude=None) -> Optional[str]:
        ex = set(exclude or [])
        if proxy_state:
            m = min(proxy_state.values())
            candidates = [p for p, c in proxy_state.items() if c == m and p not in ex]
            if candidates:
                return random.choice(candidates)
        return None

    # find initial healthy proxy
    current_proxy = None
    while proxy_state and current_proxy is None:
        cand = pick_proxy()
        ok, reason = check_proxy(cand)
        if ok:
            current_proxy = cand
            proxy_state[cand] = 0
            print(f"Initial proxy OK: {cand}")
        else:
            proxy_state[cand] += 1
            print(f"Initial proxy bad: {cand} ({reason}), count={proxy_state[cand]}")
            if proxy_state[cand] >= 10:
                removed_proxies.append((cand, reason))
                print(f"Proxy removed: {cand}")
                del proxy_state[cand]

    if current_proxy is None:
        print("No working proxy available.")
        save_removed_proxies(removed_proxies)
        return

    try:
        while True:
            list_url = LIST_URL_TEMPLATE.format(page)
            print(f"\nFetching list page {page} via {current_proxy}")
            try:
                sess = get_session_with_proxy(current_proxy)
                list_soup = get_soup(list_url, sess)
                proxy_state[current_proxy] = 0
            except Exception as e:
                print(f"List page error with {current_proxy}: {e}")
                proxy_state[current_proxy] += 1
                if proxy_state[current_proxy] >= 10:
                    removed_proxies.append((current_proxy, "too_many_consecutive_fails"))
                    print(f"Proxy removed: {current_proxy}")
                    del proxy_state[current_proxy]
                current_proxy = pick_proxy()
                if current_proxy is None:
                    print("Out of proxies.")
                    break
                profile_requests_with_current_proxy = 0
                continue

            links = extract_fighter_links_from_list_soup(list_soup)
            if not links:
                print("No fighter links found. Done.")
                break

            new_links = [u for u in links if u not in seen_profiles]
            if not new_links:
                print("No new profiles on this page. Done.")
                break

            for profile_url in new_links:
                if profile_requests_with_current_proxy >= PROXY_ROTATE_AFTER:
                    profile_requests_with_current_proxy = 0
                    np = pick_proxy(exclude=[current_proxy])
                    if np:
                        print(f"Rotating proxy: {current_proxy} -> {np}")
                        current_proxy = np

                try:
                    print(f"Fetching: {profile_url} via {current_proxy}")
                    sess = get_session_with_proxy(current_proxy)
                    prof_soup = get_soup(profile_url, sess)
                    proxy_state[current_proxy] = 0

                    info = extract_full_fighter_info(prof_soup)
                    info["ProfileURL"] = profile_url
                    collected.append(info)
                    seen_profiles.add(profile_url)
                    profile_requests_with_current_proxy += 1

                    if len(collected) % 10 == 0:
                        save_checkpoint(collected)
                        print(f"Checkpoint saved: {len(collected)} fighters")
                except Exception as e:
                    print(f"Profile error {profile_url} via {current_proxy}: {e}")
                    if current_proxy in proxy_state:
                        proxy_state[current_proxy] += 1
                        if proxy_state[current_proxy] >= 10:
                            removed_proxies.append((current_proxy, "too_many_consecutive_fails"))
                            print(f"Proxy removed: {current_proxy}")
                            del proxy_state[current_proxy]
                    np = pick_proxy()
                    if not np:
                        print("Out of proxies, stopping.")
                        page = 10**9
                        break
                    current_proxy = np
                    profile_requests_with_current_proxy = 0

                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

            page += 1
            if page > 2000:
                print("Page limit reached.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    if collected:
        save_checkpoint(collected)
        print(f"Collected: {len(collected)} fighters -> {OUTPUT_CSV}")
    if removed_proxies:
        save_removed_proxies(removed_proxies)
        print(f"{len(removed_proxies)} proxies removed -> {REMOVED_PROXIES_FILE}")
    print("Done.")


# ---------------- Retry failed profiles ----------------
def _pick_proxy_from_state(state: Dict[str,int], exclude: Optional[List[str]] = None) -> Optional[str]:
    ex = set(exclude or [])
    if state:
        m = min(state.values())
        ties = [p for p,c in state.items() if c == m and p not in ex]
        if ties:
            return random.choice(ties)
    return None

def _load_existing_collected(path: Path) -> (List[Dict], set):
    if not path.exists():
        return [], set()
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
            if "ProfileURL" in r and r["ProfileURL"]:
                seen.add(r["ProfileURL"])
    return rows, seen

def retry_failed_profiles(failed_file: Path = Path("failed_urls.txt"),
                          out_after_fail: Path = Path("failed_urls_after_retry.txt"),
                          max_retries: int = 3,
                          backoff_factor: float = 1.8):
    """
    Re-try failed profile URLs listed in failed_urls.txt.
    Use if some profiles failed during the main scrape.
    """
    if not failed_file.exists():
        print(f"{failed_file} not found.")
        return

    urls = [ln.strip() for ln in failed_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not urls:
        print("No URLs in failed file.")
        return

    base = load_proxies_from_file(GOOD_PROXIES_FILE)
    proxy_state: Dict[str,int] = {p:0 for p in base if is_valid_proxy_format(p)}

    removed_proxies: List[Tuple[str,str]] = []
    collected_rows, seen_profiles = _load_existing_collected(OUTPUT_CSV)

    still_failed = []
    for url in urls:
        if url in seen_profiles:
            print(f"Already present, skip: {url}")
            continue

        attempt = 0
        ok = False
        wait = 1.0
        while attempt < max_retries and not ok:
            attempt += 1
            proxy = _pick_proxy_from_state(proxy_state)
            label = proxy or "DIRECT"
            print(f"Try {attempt}/{max_retries} for {url} via {label}")
            try:
                sess = get_session_with_proxy(proxy)
                prof_soup = get_soup(url, sess)
                info = extract_full_fighter_info(prof_soup)
                info["ProfileURL"] = url
                collected_rows.append(info)
                seen_profiles.add(url)
                if proxy and proxy in proxy_state:
                    proxy_state[proxy] = 0
                ok = True
                print(f"SUCCESS: {url}")
            except Exception as e:
                print(f"ERROR {url} via {label}: {e}")
                if proxy:
                    proxy_state[proxy] = proxy_state.get(proxy, 0) + 1
                    if proxy_state[proxy] >= MAX_CONSECUTIVE_FAILS:
                        print(f"Removing proxy {proxy} (>= {MAX_CONSECUTIVE_FAILS} fails)")
                        removed_proxies.append((proxy, "too_many_consecutive_fails"))
                        del proxy_state[proxy]
                time.sleep(wait)
                wait *= backoff_factor

        if not ok:
            still_failed.append(url)

        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    if collected_rows:
        save_checkpoint(collected_rows)
        print(f"Checkpoint: now {len(collected_rows)} total fighters saved to {OUTPUT_CSV}")

    if removed_proxies:
        with REMOVED_PROXIES_FILE.open("a", encoding="utf-8") as f:
            for p, r in removed_proxies:
                f.write(f"{p}\t{r}\n")
        print(f"{len(removed_proxies)} proxies recorded to {REMOVED_PROXIES_FILE}")

    out_text = "\n".join(still_failed)
    out_after_fail.write_text(out_text, encoding="utf-8")
    print(f"{len(still_failed)} URLs still failed -> {out_after_fail}")

    return collected_rows, still_failed, removed_proxies


# ---------------- Entry point ----------------
if __name__ == "__main__":
    # Main run: scrape with proxy rotation and periodic CSV checkpoints.
    scrape_with_proxies()

    # If some profiles failed during the main run, you can later un-comment this
    # to re-try the URLs listed in failed_urls.txt:
    # retry_failed_profiles(max_retries=5)
