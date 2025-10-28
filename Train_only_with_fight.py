import numpy as np
import pandas as pd
from collections import defaultdict

df = pd.read_csv("fights.csv")

print(df.head())
print(df.columns)
print(len(df))

df["event_date"] = pd.to_datetime(df["event_date"])
df = df.sort_values("event_date", ascending=True).reset_index(drop=True)


def zero():
    return {
        "fights": 0, "wins": 0,
        "sig_l": 0, "sig_a": 0,
        "tot_l": 0, "tot_a": 0,
        "td_l": 0, "td_a": 0,
        "sub": 0, "rev": 0,
        "ctrl": 0, "kd": 0
    }


cum = defaultdict(zero)

fighters = []

for _, z in df.iterrows():
    if z["fighter_a"] not in fighters:
        fighters.append(z["fighter_a"])
    if z["fighter_b"] not in fighters:
        fighters.append(z["fighter_b"])

# print(len(fighters))
fighter_Stats = []

# for _, z in df.iterrows():
for i, z in df.head(100).iterrows():
    #  Fighter A
    fighterA, fighterB = z["fighter_a"], z["fighter_b"]
    date = z["event_date"]

    A = cum[fighterA]
    B = cum[fighterB]

    fighter_Stats.append({
        "event_date": date, "fight_order": i,
        "fighter": fighterA,
        "fights_prior": A["fights"], "wins_prior": A["wins"],
        "sig_l_prior": A["sig_l"], "sig_a_prior": A["sig_a"],
        "tot_l_prior": A["tot_l"], "tot_a_prior": A["tot_a"],
        "td_l_prior": A["td_l"], "td_a_prior": A["td_a"],
        "sub_prior": A["sub"], "rev_prior": A["rev"],
        "ctrl_prior": A["ctrl"], "kd_prior": A["kd"],
        "side": "A", "opponent": fighterB, "winner": z["winner"]
    })
    fighter_Stats.append({
        "event_date": date, "fight_order": i,
        "fighter": fighterB,
        "fights_prior": B["fights"], "wins_prior": B["wins"],
        "sig_l_prior": B["sig_l"], "sig_a_prior": B["sig_a"],
        "tot_l_prior": B["tot_l"], "tot_a_prior": B["tot_a"],
        "td_l_prior": B["td_l"], "td_a_prior": B["td_a"],
        "sub_prior": B["sub"], "rev_prior": B["rev"],
        "ctrl_prior": B["ctrl"], "kd_prior": B["kd"],
        "side": "B", "opponent": fighterA, "winner": z["winner"]
    })

    cum[fighterA]["fights"] += 1
    cum[fighterA]["wins"] += 1 if z["winner"] == fighterA else 0
    cum[fighterA]["sig_l"] += z["a_sig_landed"]
    cum[fighterA]["sig_a"] += z["a_sig_attempts"]
    cum[fighterA]["tot_l"] += z["a_total_landed"]
    cum[fighterA]["tot_a"] += z["a_total_attempts"]
    cum[fighterA]["td_l"] += z["a_td_landed"]
    cum[fighterA]["td_a"] += z["a_td_attempts"]
    cum[fighterA]["sub"] += z["a_sub_att"]
    cum[fighterA]["rev"] += z["a_rev"]
    cum[fighterA]["ctrl"] += z["a_ctrl_sec"]
    cum[fighterA]["kd"] += z["a_kd"]

    cum[fighterB]["fights"] += 1
    cum[fighterB]["wins"] += 1 if z["winner"] == fighterB else 0
    cum[fighterB]["sig_l"] += z["a_sig_landed"]
    cum[fighterB]["sig_a"] += z["a_sig_attempts"]
    cum[fighterB]["tot_l"] += z["a_total_landed"]
    cum[fighterB]["tot_a"] += z["a_total_attempts"]
    cum[fighterB]["td_l"] += z["a_td_landed"]
    cum[fighterB]["td_a"] += z["a_td_attempts"]
    cum[fighterB]["sub"] += z["a_sub_att"]
    cum[fighterB]["rev"] += z["a_rev"]
    cum[fighterB]["ctrl"] += z["a_ctrl_sec"]
    cum[fighterB]["kd"] += z["a_kd"]


snap = pd.DataFrame(fighter_Stats)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
print(snap)
print(cum)
#print(fighter_Stats)
