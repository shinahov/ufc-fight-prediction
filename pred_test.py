import re
import pandas as pd
import joblib
import numpy as np


import pandas as pd


fights0 = [
    ("Charles Oliveira", "Mateusz Gamrot", 5),
    ("Deiveson Figueiredo", "Montel Jackson", 3),
    ("Joel Álvarez", "Vicente Luque", 3),
    ("Mario Pinto", "Jhonata Diniz", 3),
    ("Kaan Ofli", "Ricardo Ramos", 3),
    ("Michael Aswell", "Lucas Almeida", 3),
    ("Jafel Filho", "Clayton Carpenter", 3),
    ("Vitor Petrino", "Thomas Petersen", 3),
    ("Beatriz Mesquita", "Irina Alekseeva", 3),
    ("Lucas Rocha", "Stewart Nicoll", 3),
    ("Julia Polastri", "Karolina Kowalkiewicz", 3),
    ("Luan Lacerda", "Saimon Oliveira", 3),
]

fights1 = [
    ("Brendan Allen", "Reinier de Ridder", 5),
    ("Mike Malott", "Kevin Holland", 3),
    ("Aiemann Zahabi", "Marlon Vera", 3),
    ("Manon Fiorot", "Jasmine Jasudavicius", 3),
    ("Charles Jourdain", "Davey Grant", 3),
    ("Kyle Nelson", "Matt Frevola", 3),
    ("Drew Dober", "Kyle Prepolec", 3),
    ("Aori Qileng", "Cody Gibson", 3),
    ("Bruno Gustavo da Silva", "Park Hyun-sung", 3),
    ("Djorden Ribeiro dos Santos", "Danny Barlow", 3),
    ("Stephanie Luciano", "Ravena Oliveira", 3),
    ("Yousri Belgaroui", "Azamat Bekoev", 3),
    ("Melissa Croden", "Tainara Lisboa", 3),
]

fights2 = [
    ("Alex Pereira", "Magomed Ankalaev", 5),         # Main Event Light Heavyweight
    ("Merab Dvalishvili", "Cory Sandhagen", 5),      # Title fight, 5 rounds
    ("Jiří Procházka", "Khalil Rountree Jr.", 3),
    ("Youssef Zalal", "Josh Emmett", 3),
    ("Joe Pyfer", "Abusupiyan Magomedov", 3),
    ("Ateba Abega Gautier", "Tre'ston Vines", 3),
    ("Daniel Santos", "Yoo Joo-sang", 3),
    ("Jakub Wikłacz", "Patchy Mix", 3),
    ("Edmen Shahbazyan", "André Muniz", 3),
    ("Punahele Soriano", "Nikolay Veretennikov", 3),
    ("Yana Santos", "Macy Chiasson", 3),
    ("Farid Basharat", "Chris Gutiérrez", 3),
    ("Ramiz Brahimaj", "Austin Vanderford", 3),
    ("Veronica Hardy", "Brogan Walker", 3),
]

fights = [
    ("Carlos Ulberg", "Dominick Reyes", 5),
    ("Jimmy Crute", "Ivan Erslan", 3),
    ("Jack Jenkins", "Ramon Taveras", 3),
    ("Neil Magny", "Jake Matthews", 3),
    ("Tom Nolan", "Charlie Campbell", 3),
    ("Navajo Stirling", "Rodolfo Bellato", 3),
    ("Cameron Rowston", "Andre Petroski", 3),
    ("Jamie Mullarkey", "Rolando Bedoya", 3),
    ("Colby Thicknesse", "Josias Musasa", 3),
    ("Michelle Montague", "Luana Carolina", 3),
    ("Brando Peričić", "Elisha Ellison", 3),
    ("Alexia Thainara", "Loma Lookboonmee", 3),
]

fights = fights0 + fights1 + fights2 + fights




df_fights = pd.DataFrame(fights, columns=["A_Name", "B_Name", "time_format"])








m = joblib.load("fight_predict_pipeline.pkl")
sc_diff = m["sc_diff"]
z_reg   = m["z_reg"]
clf     = m["clf"]

df = pd.read_csv("final_df.csv")


def build_diff(nameA, nameB, df, sc_diff):

    if nameA in df["A_Name"].values:
        rowA = df[df["A_Name"] == nameA].iloc[0]
    elif nameA in df["B_Name"].values:
        rowA = df[df["B_Name"] == nameA].iloc[0]
    else:
        raise ValueError(f"Kämpfer nicht gefunden: {nameA}")


    if nameB in df["B_Name"].values:
        rowB = df[df["B_Name"] == nameB].iloc[0]
    elif nameB in df["A_Name"].values:
        rowB = df[df["A_Name"] == nameB].iloc[0]
    else:
        raise ValueError(f"Kämpfer nicht gefunden: {nameB}")


    diff = {}
    for col in df.columns:
        if col.startswith("A_") and col != "A_Name":
            suff = col[2:]
            b = "B_" + suff
            if b in df.columns and b != "B_Name":
                diff["diff_" + suff] = float(rowA[col]) - float(rowB[b])


    diff["time_format"] = int("5" in str(rowA.get("time_format", "3")))


    X_diff = pd.DataFrame([diff])
    cols_fit = getattr(sc_diff, "feature_names_in_", None)
    if cols_fit is not None:
        X_diff = X_diff.reindex(columns=cols_fit, fill_value=0.0)
    X_diff_s = sc_diff.transform(X_diff)
    return X_diff_s

def predict_fight(nameA, nameB, df, model):
    sc_diff = model["sc_diff"]
    z_reg   = model["z_reg"]
    clf     = model["clf"]
    X_diff_new_s = build_diff(nameA, nameB, df, sc_diff)
    Z_pred = z_reg.predict(X_diff_new_s)
    p_win = clf.predict_proba(Z_pred)[:, 1][0]
    return p_win

def symmetric_prediction(nameA, nameB, df, model):

    all_names = set(df["A_Name"]).union(set(df["B_Name"]))
    if nameA not in all_names:
        print(f"Skippe {nameA} vs {nameB}: {nameA} ")
        return None
    if nameB not in all_names:
        print(f"Skippe {nameA} vs {nameB}: {nameB} ")
        return None


    p_AB = predict_fight(nameA, nameB, df, model)
    p_BA = predict_fight(nameB, nameA, df, model)
    return (p_AB + (1 - p_BA)) / 2.0


errors = []

for _, row in df_fights.iterrows():
    nameA = row["A_Name"]
    nameB = row["B_Name"]
    p = symmetric_prediction(nameA, nameB, df, m)
    if p is None:

        continue
    print(f"{nameA} vs {nameB}:  P(A gewinnt, sym) = {p:.3f}")
    error = abs(1 - p)
    errors.append(error)

if errors:
    error_avg = sum(errors) / len(errors)
    print(f"\nFehler: {error_avg:.3f}")
else:
    print("\nKeine gültigen Kämpfe gefunden.")




