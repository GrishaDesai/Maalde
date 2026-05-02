# test_vector_hybrid.py

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import json

# load
df = pd.read_csv("training_data.csv")

model = joblib.load("models_clean/model.pkl")
ohe = joblib.load("models_clean/ohe.pkl")
scaler = joblib.load("models_clean/scaler.pkl")
embeddings = joblib.load("models_clean/embeddings.pkl")

with open("models_clean/config.json") as f:
    config = json.load(f)

cat_cols = config["cat_cols"]
bool_cols = config["bool_cols"]
num_cols = config["num_cols"]

# helpers
def build_query_embedding(df_input):
    X_cat = ohe.transform(df_input[cat_cols])
    X_bool = df_input[bool_cols].astype(int).values
    X_num = scaler.transform(df_input[num_cols])

    vec = np.hstack([X_cat, X_bool, X_num])
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


def similarity_pred(row, df_train, emb_train):
    query = build_query_embedding(row)

    sims = cosine_similarity(query, emb_train)[0]

    temp = df_train.copy()
    temp["sim"] = sims

    top = temp.nlargest(5, "sim")

    if len(top) == 0:
        return df_train["qty_total"].median()

    return np.average(top["qty_total"], weights=top["sim"])


def model_pred(row):
    X_cat = ohe.transform(row[cat_cols])
    X_bool = row[bool_cols].astype(int).values
    X_num = scaler.transform(row[num_cols])

    X = np.hstack([X_cat, X_bool, X_num])

    return np.expm1(model.predict(X)[0])


def final_pred(row, df_train, emb_train):
    sp = similarity_pred(row, df_train, emb_train)
    mp = model_pred(row)

    diff = abs(sp - mp)

    if diff < 10:
        return (sp + mp) / 2
    elif diff < 25:
        return 0.7 * sp + 0.3 * mp
    else:
        return sp


# leave-one-out test
y_true, y_pred = [], []

for i in range(len(df)):
    test = df.iloc[[i]]
    train = df.drop(i)
    emb_train = np.delete(embeddings, i, axis=0)

    pred = final_pred(test, train, emb_train)

    y_true.append(test["qty_total"].values[0])
    y_pred.append(pred)

print("\nFINAL SYSTEM")
print("MAE:", mean_absolute_error(y_true, y_pred))
print("R2 :", r2_score(y_true, y_pred))