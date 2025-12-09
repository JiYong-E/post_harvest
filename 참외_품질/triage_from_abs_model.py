# triage_from_abs_model.py
import argparse
import joblib
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="triage_out.csv")
    ap.add_argument("--pass_t", type=float, default=12.85)
    ap.add_argument("--fail_t", type=float, default=10.8)
    ap.add_argument("--encoding", default="utf-8-sig")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding=args.encoding)
    df.columns = [str(c).strip() for c in df.columns]

    need = ["storage_day", "storage_type", "hwabang", "L_mean", "A_mean", "B_mean"]
    X = df[need].copy()

    pipe = joblib.load(args.model)
    df["sweet_pred"] = pipe.predict(X).astype(float)

    def triage(v):
        if v >= args.pass_t: return "PASS"  # proceed to shape model
        if v < args.fail_t:  return "FAIL"  # reject
        return "GRAY"                        # re-check

    df["triage"] = df["sweet_pred"].apply(triage)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.out}")
    print(df["triage"].value_counts())

if __name__ == "__main__":
    main()


"""
python .\eval_binary_from_abs_model.py `
  --csv .\test.csv `
  --model .\best_abs_sweet_model_min.joblib `
  --target 12.0 `
  --threshold 12.0
"""