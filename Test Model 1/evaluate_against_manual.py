import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


LABEL_ORDER = [-1, 0, 1]
LABEL_NAME = {-1: "negative", 0: "neutral", 1: "positive"}


def read_csv_robust(path: Path) -> pd.DataFrame:
    attempts = [
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "latin1"},
    ]
    last_error = None
    for attempt in attempts:
        try:
            df = pd.read_csv(
                path,
                sep=attempt["sep"],
                encoding=attempt["encoding"],
                engine="python",
                on_bad_lines="skip",
            )
            if len(df) > 0 and df.shape[1] <= 1:
                continue
            return df
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read {path}: {last_error}")


def normalize_label(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map(
        {
            "-1": -1,
            "0": 0,
            "1": 1,
            "negative": -1,
            "neutral": 0,
            "positive": 1,
        }
    )
    return mapped


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.lower().startswith("unnamed")]]

    alias = {
        "topic": "TOPIK",
        "username_posting": "Username Posting Owner",
        "link_post": "Link post IG",
        "username_komentar": "Username komentar",
        "isi_komentar": "Isi komentar",
        "comment": "Isi komentar",
        "comments": "Isi komentar",
        "text": "Isi komentar",
        "username": "Username komentar",
        "user": "Username komentar",
        "owner": "Username Posting Owner",
        "post_owner": "Username Posting Owner",
        "url": "Link post IG",
        "link": "Link post IG",
    }
    for old_name, new_name in alias.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    return df


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    per_class = {}
    for i, label in enumerate(LABEL_ORDER):
        per_class[LABEL_NAME[label]] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    return {
        "n_compared": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_avg": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "per_class": per_class,
        "confusion_matrix_order": LABEL_ORDER,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare predicted sentiment file against manual labels and compute metrics."
    )
    parser.add_argument("--manual", default="data/combined_data.csv", help="Path to manual-labeled CSV")
    parser.add_argument(
        "--pred",
        default="output/hasil_sentimen_combined_preview.csv",
        help="Path to prediction CSV containing Sentimen column",
    )
    parser.add_argument(
        "--out-json",
        default="output/manual_vs_pred_metrics.json",
        help="Output JSON metrics path",
    )
    parser.add_argument(
        "--out-detail",
        default="output/manual_vs_pred_detail.csv",
        help="Output detail CSV path",
    )
    args = parser.parse_args()

    manual_path = Path(args.manual)
    pred_path = Path(args.pred)

    if not manual_path.exists():
        raise FileNotFoundError(f"Manual label file not found: {manual_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    manual_df = standardize_columns(read_csv_robust(manual_path))
    pred_df = standardize_columns(read_csv_robust(pred_path))

    needed_manual = ["TOPIK", "Username Posting Owner", "Link post IG", "Username komentar", "Isi komentar", "Sentimen"]
    needed_pred = ["TOPIK", "Username Posting Owner", "Link post IG", "Username komentar", "Isi komentar", "Sentimen"]

    miss_manual = [c for c in needed_manual if c not in manual_df.columns]
    miss_pred = [c for c in needed_pred if c not in pred_df.columns]
    if miss_manual:
        raise ValueError(f"Missing manual columns: {miss_manual}")
    if miss_pred:
        raise ValueError(f"Missing prediction columns: {miss_pred}")

    key_cols = ["TOPIK", "Username Posting Owner", "Link post IG", "Username komentar", "Isi komentar"]

    # Handle duplicated keys safely by matching occurrences in order.
    manual_df["__dup_idx"] = manual_df.groupby(key_cols).cumcount()
    pred_df["__dup_idx"] = pred_df.groupby(key_cols).cumcount()

    merged = manual_df.merge(
        pred_df,
        on=key_cols + ["__dup_idx"],
        how="inner",
        suffixes=("_manual", "_pred"),
    )

    if merged.empty:
        raise ValueError("No matched rows between manual and prediction files. Check schema/content.")

    merged["Sentimen_manual"] = normalize_label(merged["Sentimen_manual"])
    merged["Sentimen_pred"] = normalize_label(merged["Sentimen_pred"])
    merged = merged.dropna(subset=["Sentimen_manual", "Sentimen_pred"]).copy()

    merged["Sentimen_manual"] = merged["Sentimen_manual"].astype(int)
    merged["Sentimen_pred"] = merged["Sentimen_pred"].astype(int)
    merged["correct"] = merged["Sentimen_manual"] == merged["Sentimen_pred"]
    merged["error_type"] = merged.apply(
        lambda r: "" if r["correct"] else f"{r['Sentimen_manual']}->{r['Sentimen_pred']}", axis=1
    )

    metrics = compute_metrics(merged["Sentimen_manual"], merged["Sentimen_pred"])
    metrics["manual_file"] = str(manual_path)
    metrics["prediction_file"] = str(pred_path)
    metrics["n_manual"] = int(len(manual_df))
    metrics["n_prediction"] = int(len(pred_df))
    metrics["n_matched"] = int(len(merged))

    out_json = Path(args.out_json)
    out_detail = Path(args.out_detail)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_detail.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    detail_cols = key_cols + ["Sentimen_manual", "Sentimen_pred", "correct", "error_type"]
    merged[detail_cols].to_csv(out_detail, index=False, encoding="utf-8-sig")

    print("Comparison completed.")
    print(f"Manual file    : {manual_path}")
    print(f"Prediction file: {pred_path}")
    print(f"Matched rows   : {len(merged)}")
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"Macro F1       : {metrics['macro_avg']['f1']:.4f}")
    print(f"Saved metrics  : {out_json}")
    print(f"Saved details  : {out_detail}")


if __name__ == "__main__":
    main()
