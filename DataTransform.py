import re
from pathlib import Path
import numpy as np
import pandas as pd

IN_FIGHTERS_RAW = Path("fighters_collected - Kopie.csv")
IN_FIGHTERS_OLD = Path("formated_fighters.csv")
IN_FIGHTS = Path("fights.csv")

OUT_FIGHTERS_FULL = Path("formated_fighters_full.csv")
OUT_FIGHTERS_FULL_FINAL = Path("formated_fighters_full_final.csv")
OUT_FIGHTS_FULL_FINAL = Path("fights_full_final.csv")
OUT_FINAL_MERGED = Path("final_df.csv")

def safe_read_csv(path): return pd.read_csv(path)

def mmss_to_sec(v):
    try:
        m, s = str(v).split(":")
        return int(m) * 60 + int(s)
    except: return np.nan

def parse_first_int(x):
    m = re.search(r"\d+", str(x))
    return int(m.group(0)) if m else None

def try_parse_b_mon_dot_date(s):
    if not isinstance(s, str): return pd.NaT
    for f in ("%b. %d, %Y", "%b %d, %Y"):
        try: return pd.to_datetime(s, format=f)
        except: pass
    return pd.NaT

def extract_last_fight_date_text(t):
    if not isinstance(t, str): return None
    m = re.search(r"([A-Z][a-z]{2}\.\s\d{1,2},\s\d{4})", t)
    if m: return m.group(1)
    m2 = re.search(r"([A-Z][a-z]{2}\s\d{1,2},\s\d{4})", t)
    return m2.group(1) if m2 else None

def extract_status_and_rank(t):
    is_active, rank = np.nan, np.nan
    if isinstance(t, str):
        if re.search(r"active", t, re.I):
            is_active = 1
            m = re.search(r"#\s*(\d+)", t)
            if m: rank = int(m.group(1))
        elif re.search(r"not\s*fighting", t, re.I): is_active = 0
    return pd.Series({"is_active": is_active, "rank": rank})

def parse_count_and_percent(v):
    if pd.isna(v): return pd.Series({"count": np.nan, "percent": np.nan})
    s = str(v).strip()
    m = re.match(r"^(\d+)\((\d+)%\)$", s)
    if m: return pd.Series({"count": int(m[1]), "percent": int(m[2]) / 100})
    m = re.match(r"^(\d+)%$", s)
    if m: return pd.Series({"count": np.nan, "percent": int(m[1]) / 100})
    m = re.match(r"^(\d+)$", s)
    if m: return pd.Series({"count": int(m[1]), "percent": np.nan})
    return pd.Series({"count": np.nan, "percent": np.nan})

WEIGHT_KG = {
    "strawweight": 52.2, "flyweight": 56.7, "bantamweight": 61.2,
    "featherweight": 65.8, "lightweight": 70.3, "welterweight": 77.1,
    "middleweight": 83.9, "light heavyweight": 93.0, "heavyweight": 120.2,
}

def parse_weight_and_gender(t):
    if not isinstance(t, str): return pd.Series({"weight_kg": np.nan, "gender": np.nan})
    g = "F" if "women" in t.lower() else "M"
    x = re.sub(r"\b(women's|women|men's|men|division)\b", "", t.lower())
    for k in sorted(WEIGHT_KG, key=len, reverse=True):
        if k in x: return pd.Series({"weight_kg": WEIGHT_KG[k], "gender": g})
    return pd.Series({"weight_kg": np.nan, "gender": g})

def step_fighters_full():
    df = safe_read_csv(IN_FIGHTERS_RAW)
    df["last_fight_info"] = df["Last Fight Block"].apply(extract_last_fight_date_text).apply(try_parse_b_mon_dot_date)
    df[["is_active", "rank"]] = df["Status"].apply(extract_status_and_rank)
    df["active"] = df["Status"].str.contains("Active", case=False, na=False).astype(int)
    df["rank"] = df["rank"].fillna(-1).astype(int)
    df = df.dropna(subset=["Weight Class", "Last Fight Block"])
    for c in ["Takedowns Landed Total","Takedowns Attempted Total","Takedown Defense",
              "Knockouts","Wins by Submission","First Round Finishes","Takedown Accuracy"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df.dropna(subset=["Striking accuracy","Sig. Str. Landed Per Min","Sig. Str. Defense"])
    df["Striking accuracy"] = df["Striking accuracy"].astype(str).str.replace("%","").replace("",np.nan).astype(float)/100
    if IN_FIGHTERS_OLD.exists(): df = pd.concat([safe_read_csv(IN_FIGHTERS_OLD), df], ignore_index=True)
    df = df.drop(columns=[c for c in ["Unnamed: 0","ProfileURL","Last Fight Block"] if c in df])
    df.to_csv(OUT_FIGHTERS_FULL, index=False)
    return df

def step_fighters_full_final():
    df = safe_read_csv(OUT_FIGHTERS_FULL)
    if "Weight Class" in df: df[["weight_kg","gender"]] = df["Weight Class"].apply(parse_weight_and_gender); df = df.drop(columns=["Weight Class"])
    if "Status" in df: df = df.drop(columns=["Status"])
    if "Record" in df:
        df["Record"] = df["Record"].astype(str).str.replace(r"\s*\(.*\)","",regex=True)
        wld = df["Record"].str.split("-",expand=True)
        if wld.shape[1]>=3: df[["W","L","D"]] = wld.iloc[:,:3].astype(int); df["total_fights"]=df["W"]+df["L"]+df["D"]
        df = df.drop(columns=["Record"])
    for c in ["Last Fight Date","last_fight_info"]:
        if c in df: df = df.drop(columns=[c])
    df["Striking accuracy"] = df["Striking accuracy"].astype(str).str.replace("%","").replace("",np.nan).astype(float)/100
    cols = ["Takedown Accuracy","Sig. Str. Defense","Takedown Defense","Standing (Sig Str)",
            "Clinch (Sig Str)","Ground (Sig Str)","Head (Sig Str)","Body (Sig Str)",
            "Leg (Sig Str)","Win by KO/TKO","Win by DEC","Win by SUB"]
    for c in cols:
        if c in df:
            tmp=df[c].apply(parse_count_and_percent)
            df[f"{c}_count"]=tmp["count"]
            df[f"{c}_percent"]=tmp["percent"]
    df = df.drop(columns=[c for c in ["Takedown Defense_count","Sig. Str. Defense_count","Takedown Accuracy_count"] if c in df])
    if "Average fight time" in df: df["Average fight time"]=df["Average fight time"].apply(mmss_to_sec)
    df = df.drop(columns=[c for c in cols if c in df])
    df["gender"]=df["gender"].map({"F":0,"M":1}).astype("Int64")
    df.to_csv(OUT_FIGHTERS_FULL_FINAL,index=False)
    return df

def step_fights_and_merge():
    df = safe_read_csv(IN_FIGHTS)
    dt = pd.to_datetime(df["event_date"], errors="coerce")
    df["date"] = (dt - dt.min()).dt.days + 1
    df = df.dropna(subset=["winner"])
    df["end_time"] = df["end_time"].apply(mmss_to_sec)
    mapping={"KO/TKO":6,"Submission":5,"TKO - Doctor's Stoppage":4,"Decision - Unanimous":3,"Decision - Majority":2,"Decision - Split":1,"DQ":0}
    df["result_score"]=df["method"].map(mapping)
    df["time_format"]=df["time_format"].apply(parse_first_int)
    df=df.dropna(subset=["time_format"])
    df=df[df["time_format"].isin([3,5])]
    df["Winner_bin"]=df.apply(lambda r:0 if r["winner"]==r["fighter_a"] else (1 if r["winner"]==r["fighter_b"] else np.nan),axis=1)
    keep=df.drop(columns=[c for c in ["event_date","event","referee"] if c in df])
    keep.to_csv(OUT_FIGHTS_FULL_FINAL,index=False)
    fighters=safe_read_csv(OUT_FIGHTERS_FULL_FINAL)
    dfA=keep.merge(fighters.add_prefix("A_"),left_on="fighter_a",right_on="A_Name",how="inner")
    dfAB=dfA.merge(fighters.add_prefix("B_"),left_on="fighter_b",right_on="B_Name",how="inner")
    dfAB=dfAB.drop(columns=[c for c in ["fighter_a","fighter_b","winner","method"] if c in dfAB])
    essential=[c for c in ["Winner_bin","end_time","result_score","time_format","A_weight_kg","B_weight_kg","A_Striking accuracy","B_Striking accuracy"] if c in dfAB]
    if essential: dfAB=dfAB.dropna(subset=essential)
    dfAB.to_csv(OUT_FINAL_MERGED,index=False)
    return dfAB

def main():
    step_fighters_full()
    step_fighters_full_final()
    step_fights_and_merge()

if __name__ == "__main__":
    main()
