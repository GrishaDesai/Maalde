import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ── LOAD FILES ─────────────────────────────
df = pd.read_csv("final_dataset_training.csv")

# ── TARGET ────────────────────────────────
TARGET = "qty_total"

# ── FEATURES ──────────────────────────────
cat_cols = [
    "fabric", "length", "sleeves", "neck",
    "work", "pattern", "primary_color",
    "occasion", "price_tier"
]

bool_cols = ["dupatta", "pants_included"]

num_cols = [
    "rate_avg",
    "days_on_market",

    # visual stats (USE _y)
    "colorfulness_y",
    "brightness_mean_y",
    "contrast_y",
    "edge_density_y",
    "texture_complexity_y",
    "color_variety_y"

]

resnet_cols = [col for col in df.columns if col.startswith("resnet_")]

# ── CLEAN DATA ────────────────────────────
df[cat_cols] = df[cat_cols].fillna("unknown")
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for col in bool_cols:
    df[col] = df[col].fillna(False).astype(bool)

# ─────────────────────────────────────────
# 🔥 ENCODING
# ─────────────────────────────────────────
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
scaler = StandardScaler()

X_cat  = ohe.fit_transform(df[cat_cols])
X_bool = df[bool_cols].astype(int).values
X_num  = scaler.fit_transform(df[num_cols])
X_resnet = df[resnet_cols].fillna(0).values

joblib.dump(X_resnet, "models_clean/resnet_features.pkl")
joblib.dump(df[TARGET].values, "models_clean/targets.pkl")

# ─────────────────────────────────────────
# 🔥 FINAL FEATURES
# ─────────────────────────────────────────
X = np.hstack([
    X_cat,
    X_bool,
    X_num,
    X_resnet 
])

y = np.log1p(df[TARGET])

# ─────────────────────────────────────────
# TRAIN TEST
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.025,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.5,
    random_state=42
)

model.fit(X_train, y_train)

# ─────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────
y_pred = model.predict(X_test)

mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
r2  = r2_score(np.expm1(y_test), np.expm1(y_pred))

print("\n📊 MODEL PERFORMANCE")
print(f"MAE: {mae:.2f}")
print(f"R2 : {r2:.3f}")

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────
os.makedirs("models_clean", exist_ok=True)

joblib.dump(model, "models_clean/model.pkl")
joblib.dump(ohe, "models_clean/ohe.pkl")
joblib.dump(scaler, "models_clean/scaler.pkl")

config = {
    "cat_cols": cat_cols,
    "bool_cols": bool_cols,
    "num_cols": num_cols
}

with open("models_clean/config.json", "w") as f:
    json.dump(config, f)

print("\n✅ Model saved successfully")