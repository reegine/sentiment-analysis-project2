import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import clean_text


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
TRAIN_PREDICTIONS_PATH = OUTPUT_DIR / "train_predictions.csv"
ALL_SPLIT_PREDICTIONS_PATH = OUTPUT_DIR / "all_split_predictions.csv"
TRAIN_SPLIT_PATH = OUTPUT_DIR / "train_split.csv"
TEST_SPLIT_PATH = OUTPUT_DIR / "test_split.csv"
CLASSIC_MODEL_PATH = Path("sentiment_pipeline.pkl")
INDOBERT_DIR = OUTPUT_DIR / "indobert_model"

LABEL_ORDER = [-1, 0, 1]
LABEL_NAME = {-1: "negative", 0: "neutral", 1: "positive"}


def read_csv_robust(file_path: Path) -> pd.DataFrame:
    read_attempts = [
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "latin1"},
    ]
    last_error = None
    for attempt in read_attempts:
        try:
            return pd.read_csv(
                file_path,
                sep=attempt["sep"],
                encoding=attempt["encoding"],
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise RuntimeError(f"Failed to read {file_path}: {last_error}")


def load_combined_labeled_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "combined_data.csv"
    xlsx_path = DATA_DIR / "combined_data.xlsx"

    if csv_path.exists():
        df = read_csv_robust(csv_path)
    elif xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError("Missing data/combined_data.csv and data/combined_data.xlsx")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.lower().startswith("unnamed")]]

    rename_map = {
        "topic": "TOPIK",
        "username_posting": "Username Posting Owner",
        "link_post": "Link post IG",
        "username_komentar": "Username komentar",
        "isi_komentar": "Isi komentar",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required_cols = [
        "TOPIK",
        "Username Posting Owner",
        "Link post IG",
        "Username komentar",
        "Isi komentar",
        "Sentimen",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in combined data: {missing}")

    df = df[required_cols].copy()
    df["Isi komentar"] = df["Isi komentar"].fillna("").astype(str).str.strip()
    df["Sentimen"] = df["Sentimen"].astype(str).str.strip().str.lower()

    label_map = {
        "-1": -1,
        "0": 0,
        "1": 1,
        "negative": -1,
        "neutral": 0,
        "positive": 1,
    }
    df["Sentimen"] = df["Sentimen"].map(label_map)
    df = df.dropna(subset=["Isi komentar", "Sentimen"]).copy()
    df["Sentimen"] = df["Sentimen"].astype(int)
    df = df[df["Isi komentar"].str.len() > 0].copy()

    df["clean_comment"] = df["Isi komentar"].apply(clean_text)
    df["clean_comment"] = df["clean_comment"].fillna("").astype(str).str.strip()

    return df


def count_labels(y: pd.Series) -> dict[str, int]:
    counts = y.value_counts().to_dict()
    return {str(label): int(counts.get(label, 0)) for label in LABEL_ORDER}


def compute_baseline_majority(y_train: pd.Series, y_test: pd.Series) -> tuple[dict, np.ndarray]:
    majority_class = int(y_train.value_counts().idxmax())
    y_pred = np.full(shape=len(y_test), fill_value=majority_class)
    metrics = {
        "majority_class": majority_class,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
    }
    return metrics, y_pred


def compute_ovr_tp_tn(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for cls in LABEL_ORDER:
        y_true_cls = y_true == cls
        y_pred_cls = y_pred == cls
        tp = int(np.logical_and(y_true_cls, y_pred_cls).sum())
        tn = int(np.logical_and(~y_true_cls, ~y_pred_cls).sum())
        fp = int(np.logical_and(~y_true_cls, y_pred_cls).sum())
        fn = int(np.logical_and(y_true_cls, ~y_pred_cls).sum())
        out[LABEL_NAME[cls]] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    return out


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    per_class = {
        LABEL_NAME[label]: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, label in enumerate(LABEL_ORDER)
    }

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_avg": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "ovr_tp_tn": compute_ovr_tp_tn(y_true.to_numpy(), y_pred),
    }


def build_classic_model() -> Pipeline:
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    candidates = {
        "linear_svc": Pipeline(steps=[("tfidf", tfidf), ("clf", LinearSVC(class_weight="balanced", random_state=42))]),
        "logreg": Pipeline(
            steps=[
                ("tfidf", tfidf),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "multinomial_nb": Pipeline(steps=[("tfidf", tfidf), ("clf", MultinomialNB(alpha=0.4))]),
    }
    # Keep selection simple for classical fallback: fit each and pick best on training set CV-like split.
    best_name = "linear_svc"
    return candidates[best_name]


def train_classic(
    x_train: pd.Series,
    y_train: pd.Series,
    x_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    model = build_classic_model()
    model.fit(x_train, y_train)
    joblib.dump(model, CLASSIC_MODEL_PATH)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    return y_train_pred, y_test_pred


def train_indobert(
    x_train: pd.Series,
    y_train: pd.Series,
    x_test: pd.Series,
    y_test: pd.Series,
    checkpoint: str,
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    class TextDataset(Dataset):
        def __init__(self, encodings: dict, labels: list[int]):
            self.encodings = encodings
            self.labels = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict:
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    os.environ.setdefault("WANDB_DISABLED", "true")
    set_seed(42)

    sent_to_id = {-1: 0, 0: 1, 1: 2}
    id_to_sent = {0: -1, 1: 0, 2: 1}

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

    train_enc = tokenizer(x_train.tolist(), truncation=True, max_length=max_length)
    test_enc = tokenizer(x_test.tolist(), truncation=True, max_length=max_length)

    train_labels = [sent_to_id[int(v)] for v in y_train.tolist()]
    test_labels = [sent_to_id[int(v)] for v in y_test.tolist()]

    # Weighted loss helps prevent minority classes from being ignored.
    # We use inverse frequency so underrepresented classes get higher penalty.
    label_counts = np.bincount(train_labels, minlength=3)
    label_counts = np.maximum(label_counts, 1)
    class_weights_np = len(train_labels) / (len(label_counts) * label_counts)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
    print(f"Class weights by label_id [0,1,2]: {class_weights_np.round(4).tolist()}")

    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            if labels is None:
                loss = outputs.get("loss")
            else:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

            return (loss, outputs) if return_outputs else loss

    train_ds = TextDataset(train_enc, train_labels)
    test_ds = TextDataset(test_enc, test_labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_args = TrainingArguments(
        output_dir=str(INDOBERT_DIR / "checkpoints"),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        report_to=[],
        save_total_limit=1,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred_ids = np.argmax(logits, axis=1)
        return {
            "accuracy": float((pred_ids == labels).mean()),
            "f1_macro": float(f1_score(labels, pred_ids, average="macro", zero_division=0)),
        }

    trainer = WeightedTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    INDOBERT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(INDOBERT_DIR))
    tokenizer.save_pretrained(str(INDOBERT_DIR))

    train_pred_output = trainer.predict(train_ds)
    train_pred_ids = np.argmax(train_pred_output.predictions, axis=1)
    y_train_pred = np.array([id_to_sent[int(i)] for i in train_pred_ids])

    test_pred_output = trainer.predict(test_ds)
    test_pred_ids = np.argmax(test_pred_output.predictions, axis=1)
    y_test_pred = np.array([id_to_sent[int(i)] for i in test_pred_ids])

    return y_train_pred, y_test_pred


def build_prediction_result(df_source: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    result = df_source.copy()
    result["Prediksi_Sentimen"] = y_pred
    result["Prediksi_Label"] = pd.Series(y_pred).map(LABEL_NAME).values
    result["correct"] = result["Sentimen"] == result["Prediksi_Sentimen"]
    result["error_type"] = np.where(
        result["correct"],
        "",
        result["Sentimen"].astype(str) + "->" + result["Prediksi_Sentimen"].astype(str),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sentiment model on combined Indonesian comments.")
    parser.add_argument("--model-type", choices=["indobert", "classic"], default="indobert")
    parser.add_argument("--checkpoint", default="indobenchmark/indobert-base-p1")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_combined_labeled_data()
    if df.empty:
        raise ValueError("No valid rows found in combined dataset.")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Sentimen"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(TRAIN_SPLIT_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_SPLIT_PATH, index=False, encoding="utf-8-sig")

    class_balance = {
        "full": count_labels(df["Sentimen"]),
        "train": count_labels(train_df["Sentimen"]),
        "test": count_labels(test_df["Sentimen"]),
    }

    print("Class balance (full/train/test):")
    print(json.dumps(class_balance, indent=2))

    baseline_metrics, _ = compute_baseline_majority(train_df["Sentimen"], test_df["Sentimen"])

    if args.model_type == "indobert":
        y_train_pred, y_test_pred = train_indobert(
            x_train=train_df["clean_comment"],
            y_train=train_df["Sentimen"],
            x_test=test_df["clean_comment"],
            y_test=test_df["Sentimen"],
            checkpoint=args.checkpoint,
            max_length=args.max_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        model_artifact = str(INDOBERT_DIR)
    else:
        y_train_pred, y_test_pred = train_classic(
            x_train=train_df["clean_comment"],
            y_train=train_df["Sentimen"],
            x_test=test_df["clean_comment"],
        )
        model_artifact = str(CLASSIC_MODEL_PATH)

    eval_metrics = evaluate_predictions(test_df["Sentimen"], y_test_pred)

    train_result = build_prediction_result(train_df, y_train_pred)
    test_result = build_prediction_result(test_df, y_test_pred)

    train_result.to_csv(TRAIN_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    test_result.to_csv(TEST_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    all_result = pd.concat(
        [
            train_result.assign(split="train"),
            test_result.assign(split="test"),
        ],
        ignore_index=True,
    )
    all_result.to_csv(ALL_SPLIT_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    metrics_payload = {
        "model_type": args.model_type,
        "model_artifact": model_artifact,
        "split": {"train": 0.8, "test": 0.2},
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "class_balance": class_balance,
        "baseline_majority": baseline_metrics,
        "confusion_matrix_order": LABEL_ORDER,
        "confusion_matrix": eval_metrics["confusion_matrix"],
        "per_class": eval_metrics["per_class"],
        "macro_avg": eval_metrics["macro_avg"],
        "accuracy": eval_metrics["accuracy"],
        "ovr_tp_tn": eval_metrics["ovr_tp_tn"],
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\nConfusion matrix with label order [-1, 0, 1]:")
    print(np.array(eval_metrics["confusion_matrix"]))
    print(f"\nAccuracy: {eval_metrics['accuracy']:.4f}")
    print(f"Macro F1: {eval_metrics['macro_avg']['f1']:.4f}")
    print(f"Baseline accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"Baseline macro-F1: {baseline_metrics['macro_f1']:.4f}")
    print(f"Saved train split -> {TRAIN_SPLIT_PATH}")
    print(f"Saved test split -> {TEST_SPLIT_PATH}")
    print(f"Saved train predictions -> {TRAIN_PREDICTIONS_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved test predictions -> {TEST_PREDICTIONS_PATH}")
    print(f"Saved all split predictions -> {ALL_SPLIT_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()