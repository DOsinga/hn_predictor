#!/usr/bin/env python3
"""Train a binary classifier for whether an HN story reaches the daily front page."""

from __future__ import annotations

import argparse
import csv
import json
import os
import json as jsonlib
import sqlite3
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from train_pretrained_embedder import (
    DEFAULT_DB,
    DEFAULT_EMBEDDER,
    embed_rows,
    load_rows,
    split_rows,
    utc_now,
)

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "frontpage_classifier")


def labels_for_rows(db_path: str, rows: list[sqlite3.Row]) -> np.ndarray:
    ids = [row["id"] for row in rows]
    if not ids:
        return np.array([], dtype="int64")
    conn = sqlite3.connect(db_path)
    try:
        placeholders = ",".join("?" for _ in ids)
        front_ids = {
            row[0]
            for row in conn.execute(
                f"""
                select distinct item_id
                from frontpage_items
                where item_id in ({placeholders})
                """,
                ids,
            )
        }
    finally:
        conn.close()
    return np.array([1 if row["id"] in front_ids else 0 for row in rows], dtype="int64")


def annotated_days(db_path: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[0] for row in conn.execute("select day from frontpage_days")}
    finally:
        conn.close()


def keep_annotated_days(db_path: str, rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    days = annotated_days(db_path)
    return [row for row in rows if row["created_at"][:10] in days]


def row_category(row: sqlite3.Row) -> str:
    tags = set(jsonlib.loads(row["tags_json"] or "[]"))
    if "ask_hn" in tags:
        return "ask_hn"
    if "show_hn" in tags:
        return "show_hn"
    if "launch_hn" in tags:
        return "launch_hn"
    if not row["url"]:
        return "no_url_other"
    return "url_story"


def keep_categories(rows: list[sqlite3.Row], categories: set[str] | None) -> list[sqlite3.Row]:
    if not categories:
        return rows
    return [row for row in rows if row_category(row) in categories]


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    order = np.argsort(scores)[::-1][: min(k, len(scores))]
    return float(np.mean(y_true[order])) if len(order) else 0.0


def threshold_at_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    candidates = np.where(recall[:-1] >= target_recall)[0]
    if len(candidates) == 0:
        return {"threshold": None, "precision": None, "recall": None}
    best = candidates[np.argmax(precision[candidates])]
    return {
        "threshold": float(thresholds[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def write_predictions(path: str, rows: list[sqlite3.Row], labels: np.ndarray, scores: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "created_at", "frontpage", "probability", "points", "title", "url"])
        for row, label, score in zip(rows, labels, scores):
            writer.writerow(
                [
                    row["id"],
                    row["created_at"],
                    int(label),
                    round(float(score), 5),
                    row["points"],
                    row["title"],
                    row["url"] or "",
                ]
            )


def train(args: argparse.Namespace) -> int:
    rows = load_rows(args)
    categories = set(args.category) if args.category else None
    if categories:
        before = len(rows)
        rows = keep_categories(rows, categories)
        print(f"Kept {len(rows)} of {before} stories in categories: {', '.join(sorted(categories))}.")
    if args.annotated_days_only:
        before = len(rows)
        rows = keep_annotated_days(args.db, rows)
        print(f"Kept {len(rows)} of {before} stories from days with front-page labels.")
    if len(rows) < 20:
        raise SystemExit(f"Need at least 20 annotated rows to train; found {len(rows)}")
    train_rows, val_rows = split_rows(rows, args.validation_fraction)
    train_y = labels_for_rows(args.db, train_rows)
    val_y = labels_for_rows(args.db, val_rows)
    if train_y.sum() == 0 or val_y.sum() == 0:
        raise SystemExit(
            f"Need positive front-page labels in train and validation; got train={train_y.sum()} val={val_y.sum()}"
        )

    print(
        f"Training front-page classifier on {len(train_rows)} stories "
        f"({int(train_y.sum())} positives); validating on {len(val_rows)} "
        f"({int(val_y.sum())} positives).",
        flush=True,
    )

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(args.embedder, device=args.device)
    train_x = embed_rows(args, train_rows, split_name="frontpage-train", embedder=embedder)
    val_x = embed_rows(args, val_rows, split_name="frontpage-val", embedder=embedder)

    if args.classifier == "logistic":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                C=args.c,
                max_iter=args.max_iter,
                random_state=args.seed,
                solver="lbfgs",
            ),
        )
    elif args.classifier == "forest":
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2_regularization,
            class_weight="balanced",
            random_state=args.seed,
        )
    clf.fit(train_x, train_y)
    val_scores = clf.predict_proba(val_x)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(val_y, val_scores)),
        "average_precision": float(average_precision_score(val_y, val_scores)),
        "base_rate": float(np.mean(val_y)),
        "precision_at_50": precision_at_k(val_y, val_scores, 50),
        "precision_at_100": precision_at_k(val_y, val_scores, 100),
        "precision_at_250": precision_at_k(val_y, val_scores, 250),
        "threshold_recall_50": threshold_at_recall(val_y, val_scores, 0.5),
        "threshold_recall_80": threshold_at_recall(val_y, val_scores, 0.8),
    }

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "classifier.joblib")
    joblib.dump(clf, model_path)
    write_predictions(
        os.path.join(args.model_dir, "validation_predictions.csv"), val_rows, val_y, val_scores
    )
    metadata = {
        "trained_at": utc_now(),
        "kind": "frontpage_classifier",
        "classifier": args.classifier,
        "embedder": args.embedder,
        "embedding_dimension": int(train_x.shape[1]),
        "db": os.path.abspath(args.db),
        "model_path": os.path.abspath(model_path),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_positives": int(train_y.sum()),
        "validation_positives": int(val_y.sum()),
        "input": "title plus normalized URL/domain/path text",
        "input_mode": args.input_mode,
        "categories": sorted(categories) if categories else None,
        "target": "appeared in HN daily /front archive",
        "args": vars(args),
        "metrics": metrics,
    }
    with open(os.path.join(args.model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(json.dumps(metrics, indent=2))
    print(f"Saved classifier to {model_path}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a daily-front-page classifier.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "embeddings"),
    )
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--input-mode", choices=("title_url", "title"), default="title_url")
    parser.add_argument(
        "--category",
        action="append",
        choices=("url_story", "show_hn", "ask_hn", "launch_hn", "no_url_other"),
        help="Restrict to one category; repeat for multiple categories.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-age-days", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--classifier", choices=("histgb", "forest", "logistic"), default="histgb")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--annotated-days-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 0 < args.validation_fraction < 0.9:
        parser.error("--validation-fraction must be between 0 and 0.9")
    return args


if __name__ == "__main__":
    raise SystemExit(train(parse_args(sys.argv[1:])))
