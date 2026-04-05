import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from preprocessing import clean_text


CLASSIC_MODEL_PATH = Path("sentiment_pipeline.pkl")
INDOBERT_DIR = Path("output/indobert_model")
OUTPUT_DIR = Path("output")


def has_indobert_artifact() -> bool:
    required_files = [
        INDOBERT_DIR / "config.json",
        INDOBERT_DIR / "tokenizer_config.json",
    ]
    return all(path.exists() for path in required_files)


def load_raw_json_to_dataframe(json_path: Path, default_topic: str) -> pd.DataFrame:
    """Convert scraped nested JSON into a flat table ready for prediction."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw_posts = json.load(f)

    rows = []
    for post in raw_posts:
        owner = post.get("username", "")
        link = post.get("url", "")
        comments = post.get("comments_details", []) or []

        for comment in comments:
            rows.append(
                {
                    "TOPIK": default_topic,
                    "Username Posting Owner": owner,
                    "Link post IG": link,
                    "Username komentar": comment.get("username", ""),
                    "Isi komentar": comment.get("text", ""),
                }
            )

    return pd.DataFrame(rows)


def load_flat_csv_to_dataframe(csv_path: Path, default_topic: str) -> pd.DataFrame:
    """Read flat CSV and map common column names to the expected output schema."""
    read_attempts = [
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "latin1"},
        {"sep": ",", "encoding": "latin1"},
    ]

    df = None
    last_error = None
    for attempt in read_attempts:
        try:
            candidate = pd.read_csv(
                csv_path,
                sep=attempt["sep"],
                encoding=attempt["encoding"],
                engine="python",
                on_bad_lines="skip",
            )

            # Guard against obviously wrong parsing (e.g., whole line as one column).
            if candidate.shape[1] <= 1 and len(candidate) > 0:
                continue

            df = candidate
            break
        except Exception as exc:  # pragma: no cover
            last_error = exc

    if df is None:
        raise RuntimeError(f"Unable to read input CSV {csv_path}: {last_error}")

    df.columns = [str(c).strip() for c in df.columns]

    # Accept either Indonesian or English-like source headers.
    column_alias = {
        "comment": "Isi komentar",
        "comments": "Isi komentar",
        "text": "Isi komentar",
        "isi_komentar": "Isi komentar",
        "username": "Username komentar",
        "user": "Username komentar",
        "username_komentar": "Username komentar",
        "owner": "Username Posting Owner",
        "post_owner": "Username Posting Owner",
        "username_posting": "Username Posting Owner",
        "url": "Link post IG",
        "link": "Link post IG",
        "link_post": "Link post IG",
        "topic": "TOPIK",
    }
    for old_name, new_name in column_alias.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    required = [
        "TOPIK",
        "Username Posting Owner",
        "Link post IG",
        "Username komentar",
        "Isi komentar",
    ]
    for col in required:
        if col not in df.columns:
            if col == "TOPIK":
                df[col] = default_topic
            else:
                df[col] = ""

    return df[required].copy()


def load_input_data(input_path: Path, default_topic: str) -> pd.DataFrame:
    if input_path.suffix.lower() == ".json":
        return load_raw_json_to_dataframe(input_path, default_topic)
    if input_path.suffix.lower() == ".csv":
        return load_flat_csv_to_dataframe(input_path, default_topic)
    raise ValueError("Input file must be .json or .csv")


def resolve_input_path(raw_path: str) -> Path:
    input_path = Path(raw_path)
    if input_path.exists():
        return input_path

    fallback_map = {
        Path("data/raw_comments_regine.json"): Path("data/data_1/raw_comments_regine.json"),
    }
    fallback = fallback_map.get(input_path)
    if fallback and fallback.exists():
        return fallback

    return input_path


def predict_classic(texts: pd.Series) -> list[int]:
    if not CLASSIC_MODEL_PATH.exists():
        raise FileNotFoundError("Classic model not found. Run train_model.py --model-type classic first.")
    model = joblib.load(CLASSIC_MODEL_PATH)
    return model.predict(texts).tolist()


def predict_indobert(texts: pd.Series, max_length: int, batch_size: int) -> list[int]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not has_indobert_artifact():
        raise FileNotFoundError("IndoBERT model folder not found. Run train_model.py first.")

    tokenizer = AutoTokenizer.from_pretrained(INDOBERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(INDOBERT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id_to_sent = {0: -1, 1: 0, 2: 1}
    predictions = []

    all_texts = texts.tolist()
    with torch.no_grad():
        for start in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            pred_ids = torch.argmax(logits, dim=1).cpu().tolist()
            predictions.extend([id_to_sent[int(i)] for i in pred_ids])

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Indonesian sentiment from comments.")
    parser.add_argument(
        "--input",
        default="data/data_1/raw_comments_regine.json",
        help="Path to input comments (.json nested scrape or .csv flat table)",
    )
    parser.add_argument(
        "--topic",
        default="Topik Umum",
        help="Default topic value if not present in input",
    )
    parser.add_argument(
        "--out-prefix",
        default="hasil_sentimen",
        help="Output filename prefix under output/",
    )
    parser.add_argument(
        "--model-type",
        choices=["auto", "indobert", "classic"],
        default="auto",
        help="Inference model type. 'auto' prefers indobert when available.",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = load_input_data(input_path, args.topic)
    if df.empty:
        raise ValueError("No comments found in input data.")

    df["Isi komentar"] = df["Isi komentar"].fillna("").astype(str)
    df["clean_comment"] = df["Isi komentar"].apply(clean_text)

    selected_model_type = args.model_type
    if selected_model_type == "auto":
        selected_model_type = "indobert" if has_indobert_artifact() else "classic"

    if selected_model_type == "indobert":
        pred_numeric = predict_indobert(df["clean_comment"], max_length=args.max_length, batch_size=args.batch_size)
    else:
        pred_numeric = predict_classic(df["clean_comment"])

    numeric_to_label = {-1: "negative", 0: "neutral", 1: "positive"}

    df["Sentimen"] = pred_numeric
    df["Sentimen_Label"] = pd.Series(pred_numeric).map(numeric_to_label).values

    final_cols = [
        "TOPIK",
        "Username Posting Owner",
        "Link post IG",
        "Username komentar",
        "Isi komentar",
        "Sentimen",
        "Sentimen_Label",
    ]
    result = df[final_cols].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{args.out_prefix}.csv"
    xlsx_path = OUTPUT_DIR / f"{args.out_prefix}.xlsx"
    json_path = OUTPUT_DIR / f"{args.out_prefix}.json"

    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    result.to_excel(xlsx_path, index=False)
    result.to_json(json_path, orient="records", force_ascii=False, indent=2)

    print("Prediction completed.")
    print(f"Saved CSV  : {csv_path}")
    print(f"Saved Excel: {xlsx_path}")
    print(f"Saved JSON : {json_path}")
    print("\nSentiment distribution (%):")
    print(result["Sentimen_Label"].value_counts(normalize=True).mul(100).round(2))


if __name__ == "__main__":
    main()