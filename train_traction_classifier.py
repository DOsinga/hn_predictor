#!/usr/bin/env python3
"""Train a binary classifier for whether an HN story gets meaningful traction.

This uses a margin target: low-score stories are negative, high-score stories
are positive, and the uncertain middle band is excluded from training/eval.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_frontpage_classifier import (
    annotated_days,
    keep_categories,
    precision_at_k,
    row_category,
    threshold_at_recall,
)
from train_pretrained_embedder import (
    DEFAULT_DB,
    DEFAULT_EMBEDDER,
    embed_rows,
    load_rows,
    split_rows,
    utc_now,
)

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "traction_classifier")


def keep_annotated_days(db_path: str, rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    days = annotated_days(db_path)
    return [row for row in rows if row["created_at"][:10] in days]


def keep_margin_rows(
    rows: list[sqlite3.Row], *, negative_max: int, positive_min: int
) -> list[sqlite3.Row]:
    return [
        row
        for row in rows
        if row["points"] <= negative_max or row["points"] >= positive_min
    ]


def labels_for_rows(
    rows: list[sqlite3.Row], *, negative_max: int, positive_min: int
) -> np.ndarray:
    labels = []
    for row in rows:
        if row["points"] >= positive_min:
            labels.append(1)
        elif row["points"] <= negative_max:
            labels.append(0)
        else:
            raise ValueError(f"row {row['id']} is in the ignored score band")
    return np.array(labels, dtype="int64")


def write_predictions(
    path: str, rows: list[sqlite3.Row], labels: np.ndarray, scores: np.ndarray
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "created_at", "traction", "probability", "points", "title", "url"])
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
    if args.positive_min <= args.negative_max:
        raise SystemExit("--positive-min must be greater than --negative-max")

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

    before_margin = len(rows)
    rows = keep_margin_rows(
        rows, negative_max=args.negative_max, positive_min=args.positive_min
    )
    ignored = before_margin - len(rows)
    print(
        f"Kept {len(rows)} of {before_margin} stories after excluding "
        f"{args.negative_max + 1}..{args.positive_min - 1} point stories "
        f"({ignored} ignored)."
    )
    if len(rows) < 20:
        raise SystemExit(f"Need at least 20 rows to train; found {len(rows)}")

    train_rows, val_rows = split_rows(rows, args.validation_fraction)
    train_y = labels_for_rows(
        train_rows, negative_max=args.negative_max, positive_min=args.positive_min
    )
    val_y = labels_for_rows(
        val_rows, negative_max=args.negative_max, positive_min=args.positive_min
    )
    if train_y.sum() == 0 or val_y.sum() == 0:
        raise SystemExit(
            f"Need positive labels in train and validation; got train={train_y.sum()} val={val_y.sum()}"
        )

    print(
        f"Training traction classifier on {len(train_rows)} stories "
        f"({int(train_y.sum())} positives); validating on {len(val_rows)} "
        f"({int(val_y.sum())} positives).",
        flush=True,
    )

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(args.embedder, device=args.device)
    train_x = embed_rows(args, train_rows, split_name="traction-train", embedder=embedder)
    val_x = embed_rows(args, val_rows, split_name="traction-val", embedder=embedder)

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
        "precision_at_500": precision_at_k(val_y, val_scores, 500),
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
        "kind": "traction_classifier",
        "classifier": args.classifier,
        "embedder": args.embedder,
        "embedding_dimension": int(train_x.shape[1]),
        "db": os.path.abspath(args.db),
        "model_path": os.path.abspath(model_path),
        "rows": len(rows),
        "ignored_middle_rows": ignored,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_positives": int(train_y.sum()),
        "validation_positives": int(val_y.sum()),
        "negative_max_points": args.negative_max,
        "positive_min_points": args.positive_min,
        "input": "title plus normalized URL/domain/path text",
        "input_mode": args.input_mode,
        "categories": sorted(categories) if categories else None,
        "target": f"points <= {args.negative_max} vs points >= {args.positive_min}",
        "args": vars(args),
        "metrics": metrics,
    }
    with open(os.path.join(args.model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(json.dumps(metrics, indent=2))
    print(f"Saved classifier to {model_path}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a score-margin traction classifier.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "embeddings"),
    )
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--input-mode", choices=("title_url", "title"), default="title_url")
    parser.add_argument("--negative-max", type=int, default=15)
    parser.add_argument("--positive-min", type=int, default=30)
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
