#!/usr/bin/env python3
"""Train HN score prediction on a pretrained local text embedder.

This freezes a Hugging Face / SentenceTransformers embedding model, embeds the
title plus URL text, and trains a small sklearn MLP on log1p(points).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

DEFAULT_DB = os.path.join(os.path.dirname(__file__), "data", "hn_submissions.sqlite3")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "pretrained_embedder")
DEFAULT_EMBEDDER = "BAAI/bge-base-en-v1.5"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name).strip("_")


def url_tokens(url: str | None) -> str:
    if not url:
        return "No URL."
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"[-_/+.]+", " ", parsed.path.lower()).strip()
    if path:
        return f"Domain: {host}. Path words: {path}."
    return f"Domain: {host}."


def make_text(title: str, url: str | None, input_mode: str = "title_url") -> str:
    if input_mode == "title":
        return f"Title: {title or ''}"
    return f"Title: {title or ''}\n{url_tokens(url)}"


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
        select id, title, url, points, created_at, created_at_i, tags_json
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


def targets(rows: list[sqlite3.Row]) -> np.ndarray:
    return np.array([math.log1p(max(row["points"], 0)) for row in rows], dtype="float32")


def cache_key(args: argparse.Namespace, rows: list[sqlite3.Row]) -> str:
    first_id = rows[0]["id"]
    last_id = rows[-1]["id"]
    return (
        f"{args.input_mode}-{slug_model_name(args.embedder)}-"
        f"{len(rows)}-{first_id}-{last_id}-"
        f"norm{int(args.normalize_embeddings)}"
    )


def embed_rows(
    args: argparse.Namespace,
    rows: list[sqlite3.Row],
    *,
    split_name: str,
    embedder: SentenceTransformer,
) -> np.ndarray:
    os.makedirs(args.cache_dir, exist_ok=True)
    path = os.path.join(args.cache_dir, f"{split_name}-{cache_key(args, rows)}.npy")
    if os.path.isfile(path) and not args.rebuild_embeddings:
        print(f"Loading cached {split_name} embeddings: {path}", flush=True)
        return np.load(path)

    texts = [make_text(row["title"], row["url"], args.input_mode) for row in rows]
    embeddings = embedder.encode(
        texts,
        batch_size=args.embed_batch_size,
        show_progress_bar=True,
        normalize_embeddings=args.normalize_embeddings,
        convert_to_numpy=True,
    )
    np.save(path, embeddings)
    return embeddings


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


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_layers:
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def standardize(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (
        ((train_x - mean) / std).astype("float32"),
        ((val_x - mean) / std).astype("float32"),
        {"mean": mean.astype("float32"), "std": std.astype("float32")},
    )


def train_head(
    args: argparse.Namespace,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> tuple[RegressionHead, dict[str, float], list[dict[str, float]]]:
    device = torch.device(args.head_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    model = RegressionHead(train_x.shape[1], args.hidden_layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype("float32")))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_tensor_x = torch.from_numpy(val_x).to(device)
    val_tensor_y = torch.from_numpy(val_y.astype("float32")).to(device)

    best_state = None
    best_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_tensor_x)
            val_loss = float(loss_fn(val_pred, val_tensor_y).detach().cpu())
            val_mae = float(torch.mean(torch.abs(val_pred - val_tensor_y)).detach().cpu())
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_log_mae": val_mae})
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_log_mae={val_mae:.4f}",
            flush=True,
        )

        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), {"best_val_loss": best_loss}, history


def predict_head(model: RegressionHead, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size)
    with torch.no_grad():
        for batch_x in loader:
            preds.append(model(batch_x).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


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
    rows = load_rows(args)
    train_rows, val_rows = split_rows(rows, args.validation_fraction)
    train_y = targets(train_rows)
    val_y = targets(val_rows)

    print(
        f"Embedding with {args.embedder}; training on {len(train_rows)} stories; "
        f"validating on {len(val_rows)} newer stories.",
        flush=True,
    )
    print(
        f"Target is log1p(points); train range={train_rows[0]['created_at']}..{train_rows[-1]['created_at']}; "
        f"val range={val_rows[0]['created_at']}..{val_rows[-1]['created_at']}",
        flush=True,
    )

    embedder = SentenceTransformer(args.embedder, device=args.device)
    train_x = embed_rows(args, train_rows, split_name="train", embedder=embedder)
    val_x = embed_rows(args, val_rows, split_name="val", embedder=embedder)

    baseline_pred = np.full_like(val_y, np.mean(train_y))
    train_x, val_x, scaler = standardize(train_x, val_x)
    head, head_info, history = train_head(args, train_x, train_y, val_x, val_y)
    val_pred = predict_head(head, val_x, args.batch_size)

    metrics = evaluate_predictions(val_y, val_pred)
    baseline_metrics = evaluate_predictions(val_y, baseline_pred)

    os.makedirs(args.model_dir, exist_ok=True)
    head_path = os.path.join(args.model_dir, "head.pt")
    torch.save(
        {
            "state_dict": head.state_dict(),
            "input_dim": int(train_x.shape[1]),
            "hidden_layers": args.hidden_layers,
            "dropout": args.dropout,
            "scaler": scaler,
        },
        head_path,
    )
    joblib.dump(scaler, os.path.join(args.model_dir, "scaler.joblib"))
    with open(os.path.join(args.model_dir, "training_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_log_mae"])
        writer.writeheader()
        writer.writerows(history)
    predictions_path = os.path.join(args.model_dir, "validation_predictions.csv")
    write_validation_predictions(predictions_path, val_rows, val_pred)

    metadata = {
        "trained_at": utc_now(),
        "kind": "pretrained_text_embedder_plus_mlp",
        "embedder": args.embedder,
        "embedding_dimension": int(train_x.shape[1]),
        "head_path": os.path.abspath(head_path),
        "db": os.path.abspath(args.db),
        "target": "log1p(points)",
        "loss": "squared_error",
        "input": "title plus normalized URL/domain/path text",
        "rows": len(rows),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_date_range": [train_rows[0]["created_at"], train_rows[-1]["created_at"]],
        "validation_date_range": [val_rows[0]["created_at"], val_rows[-1]["created_at"]],
        "args": vars(args),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "head_info": head_info,
        "epochs_ran": len(history),
    }
    with open(os.path.join(args.model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(json.dumps({"metrics": metrics, "baseline_metrics": baseline_metrics}, indent=2))
    print(f"Saved regression head to {head_path}")
    return 0


def parse_hidden_layers(value: str) -> list[int]:
    layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not layers or any(width < 1 for width in layers):
        raise argparse.ArgumentTypeError("Use comma-separated positive integers")
    return layers


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HN score prediction from a pretrained text embedder."
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "embeddings"),
    )
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--input-mode", choices=("title_url", "title"), default="title_url")
    parser.add_argument("--device", default=None, help="Example: cpu, mps, cuda")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests")
    parser.add_argument("--min-age-days", type=int, default=0, help="Ignore very recent stories")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument("--hidden-layers", type=parse_hidden_layers, default=[128, 64])
    parser.add_argument("--head-device", default=None, help="Example: cpu, mps, cuda")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if not 0 < args.validation_fraction < 0.9:
        parser.error("--validation-fraction must be between 0 and 0.9")
    return args


if __name__ == "__main__":
    raise SystemExit(train(parse_args(sys.argv[1:])))
