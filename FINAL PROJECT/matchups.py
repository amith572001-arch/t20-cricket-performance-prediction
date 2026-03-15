# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 21:58:28 2026

@author: Amit2
"""

import pandas as pd
import numpy as np

file_path = r"C:\Users\Amit2\Desktop\ML Project Cric\t20_full_ball_by_ball.csv"

df = pd.read_csv(file_path, low_memory=False)

# keep only male matches (same as before)
df = df[df["gender"] == "male"]

# runs scored by batter
df["runs"] = df["runs_batter"]

# balls faced (ignore wides because they are not legal deliveries)
df["legal_ball"] = df["wides"].fillna(0) == 0

# boundary flag
df["boundary"] = df["runs"].isin([4,6]).astype(int)

# dot ball
df["dot"] = (df["runs"] == 0).astype(int)

# dismissal credited to bowler
df["dismissal"] = df["wicket_flag"].fillna(0)

# keep only legal deliveries
df = df[df["legal_ball"]]

# aggregate batter vs bowler stats
matchups = df.groupby(["batter","bowler"]).agg(
    runs_scored=("runs","sum"),
    balls_faced=("runs","count"),
    dismissals=("dismissal","sum"),
    boundaries=("boundary","sum"),
    dot_balls=("dot","sum")
).reset_index()

# derived metrics
matchups["strike_rate"] = 100 * matchups["runs_scored"] / matchups["balls_faced"]

matchups["average"] = np.where(
    matchups["dismissals"] > 0,
    matchups["runs_scored"] / matchups["dismissals"],
    np.nan
)

matchups["boundary_rate"] = matchups["boundaries"] / matchups["balls_faced"]

matchups["dot_rate"] = matchups["dot_balls"] / matchups["balls_faced"]

# reliability (sample size)
matchups["reliability"] = matchups["balls_faced"] / (matchups["balls_faced"] + 20)

# save dataset
output_file = r"C:\Users\Amit2\Desktop\ML Project Cric\batter_bowler_matchups.csv"

matchups.to_csv(output_file, index=False)

print("Matchup dataset created")
print("Total matchups:", len(matchups))
print(matchups.head(20))
