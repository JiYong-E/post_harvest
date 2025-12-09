# train_abs_sweet_model_min.py
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR


SEED = 42


def load_and_filter(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)

    df = df[df["source_dir"].isin(["2022_AI_분리", "2022_분리"])].copy()

    pat = re.compile(r"^0.*_(2022_AI_분리|2022_분리)_.+\.csv$", re.IGNORECASE)
    df = df[df["source_file"].astype(str).apply(lambda s: bool(pat.match(s)))].copy()

    needed = [
        "sweet_mean", "storage_day", "hwabang",
        "L_mean", "A_mean", "B_mean",
        "source_file"
    ]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    for c in ["sweet_mean", "storage_day", "hwabang", "L_mean", "A_mean", "B_mean"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["source_file"] = df["source_file"].astype(str)

    df = df.dropna(subset=["sweet_mean", "storage_day", "hwabang"]).copy()
    df = df[df["storage_day"] >= 0].copy()
    return df.reset_index(drop=True)


def make_preprocess():
    # sweet_day0_mean 없이 절대 당도 회귀
    num = ["storage_day", "L_mean", "A_mean", "B_mean"]
    cat = ["hwabang"]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),   # 비트리(선형/SVR) 필수급
    ])

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num),
            ("cat", cat_pipe, cat),
        ],
        remainder="drop",
    )
    return pre


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return r2, rmse, mae


def group_split(X, y, groups, test_size=0.2, random_state=SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    return tr_idx, te_idx


def get_models_min():
    return {
        "RidgeCV": RidgeCV(alphas=np.logspace(-4, 4, 41)),
        "SVR_RBF": SVR(C=30.0, gamma="scale", epsilon=0.1),
    }


def main():
    data_path = "참외_통합_표준컬럼_FROM_SPLIT_DIRS.csv"
    df = load_and_filter(data_path)

    features = ["storage_day", "hwabang", "L_mean", "A_mean", "B_mean"]
    X = df[features].copy()
    y = df["sweet_mean"].astype(float).values
    groups = df["source_file"].astype(str).values

    tr_idx, te_idx = group_split(X, y, groups, test_size=0.2, random_state=SEED)
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    pre = make_preprocess()
    models = get_models_min()

    rows = []
    best = None  # (rmse, -r2, name, pipe)

    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("m", model)])
        pipe.fit(X_train, y_train)

        pred_tr = pipe.predict(X_train)
        pred_te = pipe.predict(X_test)

        r2_tr, rmse_tr, mae_tr = metrics(y_train, pred_tr)
        r2_te, rmse_te, mae_te = metrics(y_test, pred_te)

        rows.append({
            "model": name,
            "train_R2": r2_tr, "train_RMSE": rmse_tr, "train_MAE": mae_tr,
            "test_R2": r2_te, "test_RMSE": rmse_te, "test_MAE": mae_te,
        })

        joblib.dump(pipe, f"abs_sweet_model__{name}.joblib")

        out = X_test.copy()
        out["sweet_true"] = y_test
        out["sweet_pred"] = pred_te
        out.to_csv(f"abs_sweet_pred_test__{name}.csv", index=False, encoding="utf-8-sig")

        key = (rmse_te, -r2_te)
        if best is None or key < best[0]:
            best = (key, name, pipe)

    report = pd.DataFrame(rows).sort_values(["test_RMSE", "test_R2"], ascending=[True, False])
    report.to_csv("abs_sweet_metrics_min.csv", index=False, encoding="utf-8-sig")

    best_name = best[1]
    joblib.dump(best[2], "best_abs_sweet_model_min.joblib")

    top = report.iloc[0]
    print("\n=== BEST ABS SWEET MODEL (no sweet_day0_mean) ===")
    print(top[["model", "test_R2", "test_RMSE", "test_MAE"]].to_string(index=False))
    print("\nSaved: abs_sweet_metrics_min.csv")
    print("Saved: best_abs_sweet_model_min.joblib")


if __name__ == "__main__":
    main()
