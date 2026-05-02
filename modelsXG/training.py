import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# ── Load data ─────────────────────────────────────────────
df = pd.read_csv("training_data.csv")

TARGET = "qty_total"

# ── Feature selection ─────────────────────────────────────
cat_cols = [
    "fabric",
    "length",
    "sleeves",
    "neck",
    "work",
    "pattern",
    "occasion",
    # "price_tier",
    # "work_pattern"       # ONLY interaction to keep
]

bool_cols = [
    "dupatta",
    "pants_included"
]

num_cols = [
    "rate_avg",          # absolute price
    # "price_relative"     # relative price (IMPORTANT)
]

# ── Remove unwanted columns ───────────────────────────────
drop_cols = [
    "product_code", "product_name", "design_no",
    "original_filename", "new_filename", "status", "extra", "duatta"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ── Fix missing numeric features ──────────────────────────
if "sales_velocity" not in df.columns:
    df["sales_velocity"] = df["qty_total"] / (df["days_on_market"] + 1)

if "price_tier" not in df.columns:
    df["price_tier"] = pd.qcut(df["rate_avg"], 3, labels=["low", "mid", "high"])

# ── Clean data ────────────────────────────────────────────
def simplify_color(c):
    c = str(c).lower()
    if "red" in c: return "red"
    if "blue" in c: return "blue"
    if "green" in c: return "green"
    if "yellow" in c: return "yellow"
    if "pink" in c: return "pink"
    if "black" in c: return "black"
    if "white" in c: return "white"
    return "other"

df["primary_color"] = df["primary_color"].apply(simplify_color)
df["work_pattern"] = df["work"] + "_" + df["pattern"]
df["price_relative"] = df["rate_avg"] / df["rate_avg"].mean()
df["price_tier"] = pd.qcut(
    df["rate_avg"],
    4,
    labels=["low", "mid", "high", "premium"]
)
df[cat_cols] = df[cat_cols].fillna("unknown")
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for col in bool_cols:
    df[col] = df[col].fillna(False).astype(bool)

# ── Train-test split ──────────────────────────────────────
X = df[cat_cols + bool_cols + num_cols]
y = np.log1p(df[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── OHE ───────────────────────────────────────────────────
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_train_cat = ohe.fit_transform(X_train[cat_cols])
X_test_cat  = ohe.transform(X_test[cat_cols])

# ── Combine all features ──────────────────────────────────
X_train_final = np.hstack([
    X_train_cat,
    X_train[bool_cols].astype(int).values,
    X_train[num_cols].values
])

X_test_final = np.hstack([
    X_test_cat,
    X_test[bool_cols].astype(int).values,
    X_test[num_cols].values
])

# ── Model ─────────────────────────────────────────────────
# model = XGBRegressor(
#     n_estimators=400,
#     learning_rate=0.05,
#     max_depth=4,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_final, y_train)

# ── Evaluation ────────────────────────────────────────────
y_pred = model.predict(X_test_final)

mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
r2  = r2_score(np.expm1(y_test), np.expm1(y_pred))

print(f"MAE: {mae:.2f}")
print(f"R2 : {r2:.3f}")

# ── Save artifacts ────────────────────────────────────────
os.makedirs("models_clean", exist_ok=True)

joblib.dump(model, "models_clean/model.pkl")
joblib.dump(ohe, "models_clean/ohe.pkl")

config = {
    "cat_cols": cat_cols,
    "bool_cols": bool_cols,
    "num_cols": num_cols,
    "target": TARGET
}

with open("models_clean/config.json", "w") as f:
    json.dump(config, f)

print("✅ Model saved successfully")