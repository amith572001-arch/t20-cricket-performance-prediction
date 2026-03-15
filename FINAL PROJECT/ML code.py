# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:13:02 2026

@author: Amit2
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# load dataset
df = pd.read_csv(r"C:\Users\Amit2\Desktop\ML Project Cric\training_dataset.csv")

# features for prediction
features = [
    "strike_rate",
    "avg_matchup_sr",
    "avg_dismissal_rate",
    "avg_bowler_rating"
]

X = df[features]

y = df["run_bucket"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))