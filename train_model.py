#!/usr/bin/env python3
"""Train a small local Hacker News score predictor.

The model learns token embeddings from the collected HN data itself. It uses
only title and URL-derived text as inputs and predicts log1p(points) with a
linear output and MSE loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DEFAULT_DB = os.path.join(os.path.dirname(__file__), "data", "hn_submissions.sqlite3")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "model")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def url_tokens(url: str | None) -> str:
    if not url:
        return "no_url"
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    host = re.sub(r"[^a-z0-9]+", " ", host)
    path = re.sub(r"[^a-z0-9]+", " ", parsed.path.lower())
    return f"domain {host} path {path}".strip()


def make_text(title: str, url: str | None) -> str:
    return f"title {title or ''} {url_tokens(url)}"


def load_rows(args: argparse.Namespace) -> list[sqlite3.Row]:
    if not os.path.isfile(args.db):
        raise SystemExit(f"Database not found: {args.db}")

    where = ["points is not null", "title is not null", "title != ''"]
    params: list[object] = []
    if args.min_age_days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.min_age_days)
        where.append("created_at_i <= ?")
        params.append(int(cutoff.timestamp()))

    sql = f"""
        select id, title, url, points, created_at, created_at_i
        from submissions
        where {' and '.join(where)}
        order by created_at_i
    """
    if args.limit:
        sql += " limit ?"
        params.append(args.limit)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    if len(rows) < 20:
        raise SystemExit(f"Need at least 20 rows to train; found {len(rows)}")
    return rows


def split_rows(rows: list[sqlite3.Row], validation_fraction: float) -> tuple[list, list]:
    split_at = int(len(rows) * (1 - validation_fraction))
    split_at = min(max(split_at, 1), len(rows) - 1)
    return rows[:split_at], rows[split_at:]


def row_arrays(rows: list[sqlite3.Row]) -> tuple[np.ndarray, np.ndarray]:
    texts = np.array([make_text(row["title"], row["url"]) for row in rows], dtype=object)
    targets = np.array([math.log1p(max(row["points"], 0)) for row in rows], dtype="float32")
    return texts, targets


def build_model(args: argparse.Namespace, train_texts: np.ndarray) -> keras.Model:
    vectorizer = layers.TextVectorization(
        max_tokens=args.vocab_size,
        output_mode="int",
        output_sequence_length=args.sequence_length,
        name="text_vectorizer",
    )
    vectorizer.adapt(train_texts)

    text = keras.Input(shape=(), dtype=tf.string, name="submission_text")
    tokens = vectorizer(text)
    x = layers.Embedding(
        input_dim=args.vocab_size,
        output_dim=args.embedding_dim,
        mask_zero=True,
        name="token_embedding",
    )(tokens)
    x = layers.GlobalAveragePooling1D(name="mean_embedding")(x)
    x = layers.Dense(args.hidden_dim, activation="relu", name="dense_1")(x)
    x = layers.Dropout(args.dropout)(x)
    x = layers.Dense(args.hidden_dim // 2, activation="relu", name="dense_2")(x)
    output = layers.Dense(1, name="log1p_points")(x)
    model = keras.Model(text, output, name="hn_score_predictor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="log_mae")],
    )
    return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = y_pred.reshape(-1)
    log_errors = y_pred - y_true
    pred_points = np.maximum(np.expm1(y_pred), 0)
    true_points = np.expm1(y_true)
    abs_factor = np.exp(np.abs(log_errors))
    return {
        "log_mse": float(np.mean(np.square(log_errors))),
        "log_rmse": float(np.sqrt(np.mean(np.square(log_errors)))),
        "log_mae": float(np.mean(np.abs(log_errors))),
        "median_factor_error": float(np.median(abs_factor)),
        "mean_absolute_points_error": float(np.mean(np.abs(pred_points - true_points))),
    }


def baseline_metrics(train_y: np.ndarray, val_y: np.ndarray) -> dict[str, float]:
    pred = np.full_like(val_y, np.mean(train_y))
    return evaluate_predictions(val_y, pred)


def write_validation_predictions(path: str, rows: list[sqlite3.Row], predictions: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "created_at", "points", "predicted_points", "title", "url"])
        for row, pred in zip(rows, predictions.reshape(-1)):
            writer.writerow(
                [
                    row["id"],
                    row["created_at"],
                    row["points"],
                    round(max(math.expm1(float(pred)), 0), 2),
                    row["title"],
                    row["url"] or "",
                ]
            )


def train(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    rows = load_rows(args)
    train_rows, val_rows = split_rows(rows, args.validation_fraction)
    train_x, train_y = row_arrays(train_rows)
    val_x, val_y = row_arrays(val_rows)

    print(
        f"Training on {len(train_rows)} stories; validating on {len(val_rows)} newer stories.",
        flush=True,
    )
    print(
        f"Target is log1p(points); train range={train_rows[0]['created_at']}..{train_rows[-1]['created_at']}; "
        f"val range={val_rows[0]['created_at']}..{val_rows[-1]['created_at']}",
        flush=True,
    )

    model = build_model(args, train_x)
    os.makedirs(args.model_dir, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        keras.callbacks.CSVLogger(os.path.join(args.model_dir, "training_log.csv")),
    ]
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    val_pred = model.predict(val_x, batch_size=args.batch_size, verbose=0).reshape(-1)
    metrics = evaluate_predictions(val_y, val_pred)
    baseline = baseline_metrics(train_y, val_y)

    model_path = os.path.join(args.model_dir, "model.keras")
    model.save(model_path)
    predictions_path = os.path.join(args.model_dir, "validation_predictions.csv")
    write_validation_predictions(predictions_path, val_rows, val_pred)

    metadata = {
        "trained_at": utc_now(),
        "db": os.path.abspath(args.db),
        "model_path": os.path.abspath(model_path),
        "target": "log1p(points)",
        "loss": "mse",
        "input": "title plus normalized URL/domain/path tokens",
        "rows": len(rows),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_date_range": [train_rows[0]["created_at"], train_rows[-1]["created_at"]],
        "validation_date_range": [val_rows[0]["created_at"], val_rows[-1]["created_at"]],
        "args": vars(args),
        "metrics": metrics,
        "baseline_metrics": baseline,
        "epochs_ran": len(history.history.get("loss", [])),
    }
    with open(os.path.join(args.model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(json.dumps({"metrics": metrics, "baseline_metrics": baseline}, indent=2))
    print(f"Saved model to {model_path}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small local HN score model.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests")
    parser.add_argument("--min-age-days", type=int, default=0, help="Ignore very recent stories")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if not 0 < args.validation_fraction < 0.9:
        parser.error("--validation-fraction must be between 0 and 0.9")
    if args.hidden_dim < 2:
        parser.error("--hidden-dim must be at least 2")
    return args


if __name__ == "__main__":
    raise SystemExit(train(parse_args(sys.argv[1:])))
