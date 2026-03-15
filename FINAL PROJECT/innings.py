# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:02:12 2026

@author: Amit2
"""

import pandas as pd

file_path = r"C:\Users\Amit2\Desktop\ML Project Cric\t20_full_ball_by_ball.csv"

df = pd.read_csv(file_path, low_memory=False)

# keep men's matches
df = df[df["gender"] == "male"]

# remove wides because they are not faced balls
df["legal_ball"] = df["wides"].fillna(0) == 0
df = df[df["legal_ball"]]

# aggregate innings
innings = df.groupby(["match_id","batter"]).agg(
    runs=("runs_batter","sum"),
    balls=("runs_batter","count"),
    venue=("venue","first"),
    batting_team=("batting_team","first"),
    opponent=("bowling_team","first"),
    season=("season","first")
).reset_index()
# count innings per batter
batter_counts = innings.groupby("batter").size().reset_index(name="innings_played")

# merge counts
innings = innings.merge(batter_counts, on="batter")

# keep batters with enough innings
innings = innings[innings["innings_played"] >= 20]
# strike rate
innings["strike_rate"] = 100 * innings["runs"] / innings["balls"]

# create run buckets (target variable)
def bucket_runs(r):
    if r < 10:
        return 0
    elif r < 25:
        return 1
    elif r < 40:
        return 2
    elif r < 60:
        return 3
    else:
        return 4

innings["run_bucket"] = innings["runs"].apply(bucket_runs)

output_file = r"C:\Users\Amit2\Desktop\ML Project Cric\batter_innings.csv"

innings.to_csv(output_file, index=False)

print("Batter innings dataset created")
print("Total innings:", len(innings))
print(innings.head(20))