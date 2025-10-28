import numpy as np
import pandas as pd
from collections import defaultdict

df = pd.read_csv("fights.csv")

print(df.head())
print(df.columns)
print(len(df))

df["event_date"] = pd.to_datetime(df["event_date"])
df = df.sort_values("event_date", ascending=True).reset_index(drop=True)


fighters = []

for _, z in df.iterrows():
    if z["fighter_a"] not in fighters:
        fighters.append(z["fighter_a"])
    if z["fighter_b"] not in fighters:
        fighters.append(z["fighter_b"])

#print(len(fighters))
fighter_Stats = {}

#for _, z in df.iterrows():
for _, z in df.head(100).iterrows():
    #  Fighter A
    key_a = (z["fighter_a"], z["event_date"])
    if key_a in fighter_Stats:
        old = fighter_Stats[key_a]
        fighter_Stats[key_a] = [
            old[0] + z["a_sig_landed"],      # sig_landed
            old[1] + z["a_sig_attempts"],    # sig_attempts
            old[2] + z["a_total_landed"],    # total_landed
            old[3] + z["a_total_attempts"],  # total_attempts
            old[4] + z["a_td_landed"],       # td_landed
            old[5] + z["a_td_attempts"],     # td_attempts
            old[6] + z["a_sub_att"],         # sub_att
            old[7] + z["a_rev"],             # reversals
            old[8] + z["a_ctrl_sec"],        # control time
            old[9] + z["a_kd"],              # knockdowns
            old[10] + (1 if z["fighter_a"] == z["winner"] else 0)  # wins
        ]
    else:
        fighter_Stats[key_a] = [
            z["a_sig_landed"],
            z["a_sig_attempts"],
            z["a_total_landed"],
            z["a_total_attempts"],
            z["a_td_landed"],
            z["a_td_attempts"],
            z["a_sub_att"],
            z["a_rev"],
            z["a_ctrl_sec"],
            z["a_kd"],
            1 if z["fighter_a"] == z["winner"] else 0
        ]

    # Fighter B
    key_b = (z["fighter_b"], z["event_date"])
    if key_b in fighter_Stats:
        old = fighter_Stats[key_b]
        fighter_Stats[key_b] = [
            old[0] + z["b_sig_landed"],
            old[1] + z["b_sig_attempts"],
            old[2] + z["b_total_landed"],
            old[3] + z["b_total_attempts"],
            old[4] + z["b_td_landed"],
            old[5] + z["b_td_attempts"],
            old[6] + z["b_sub_att"],
            old[7] + z["b_rev"],
            old[8] + z["b_ctrl_sec"],
            old[9] + z["b_kd"],
            old[10] + (1 if z["fighter_b"] == z["winner"] else 0)
        ]
    else:
        fighter_Stats[key_b] = [
            z["b_sig_landed"],
            z["b_sig_attempts"],
            z["b_total_landed"],
            z["b_total_attempts"],
            z["b_td_landed"],
            z["b_td_attempts"],
            z["b_sub_att"],
            z["b_rev"],
            z["b_ctrl_sec"],
            z["b_kd"],
            1 if z["fighter_b"] == z["winner"] else 0
        ]





print(fighter_Stats)