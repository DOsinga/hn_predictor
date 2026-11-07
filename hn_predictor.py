import os
import json
import sqlite3
from datetime import datetime, timezone

from projects.common import JsonResponse, Project

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "hn_submissions.sqlite3")
TOKEN_MODEL_METADATA_PATH = os.path.join(os.path.dirname(__file__), "data", "model", "metadata.json")
PRETRAINED_MODEL_METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "pretrained_embedder", "metadata.json"
)
FRONTPAGE_MODEL_METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "frontpage_classifier", "metadata.json"
)


def _fmt_day(ts):
    if not ts:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).date().isoformat()


def _stats():
    if not os.path.isfile(DB_PATH):
        return {"exists": False}
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            select
                count(*) as stories,
                min(created_at_i) as oldest,
                max(created_at_i) as newest,
                avg(points) as avg_points,
                max(points) as max_points
            from submissions
            """
        ).fetchone()
        run = conn.execute(
            """
            select started_at, finished_at, rows_seen, rows_written, status
            from collection_runs
            order by id desc
            limit 1
            """
        ).fetchone()
    finally:
        conn.close()
    return {
        "exists": True,
        "stories": row["stories"],
        "oldest": _fmt_day(row["oldest"]),
        "newest": _fmt_day(row["newest"]),
        "avg_points": round(row["avg_points"] or 0, 1),
        "max_points": row["max_points"] or 0,
        "last_run": dict(run) if run else None,
    }


def _model_metadata(path, label):
    if not os.path.isfile(path):
        return {"exists": False}
    with open(path) as f:
        metadata = json.load(f)
    metrics = metadata.get("metrics") or {}
    baseline = metadata.get("baseline_metrics") or {}
    return {
        "exists": True,
        "label": label,
        "kind": metadata.get("kind", label),
        "embedder": metadata.get("embedder"),
        "embedding_dimension": metadata.get("embedding_dimension"),
        "trained_at": metadata.get("trained_at"),
        "rows": metadata.get("rows"),
        "train_rows": metadata.get("train_rows"),
        "validation_rows": metadata.get("validation_rows"),
        "epochs_ran": metadata.get("epochs_ran"),
        "log_rmse": round(metrics.get("log_rmse", 0), 3),
        "log_mae": round(metrics.get("log_mae", 0), 3),
        "median_factor_error": round(metrics.get("median_factor_error", 0), 2),
        "baseline_log_rmse": round(baseline.get("log_rmse", 0), 3),
        "baseline_log_mae": round(baseline.get("log_mae", 0), 3),
    }


def _frontpage_model_metadata():
    if not os.path.isfile(FRONTPAGE_MODEL_METADATA_PATH):
        return {"exists": False}
    with open(FRONTPAGE_MODEL_METADATA_PATH) as f:
        metadata = json.load(f)
    metrics = metadata.get("metrics") or {}
    return {
        "exists": True,
        "classifier": metadata.get("classifier"),
        "embedder": metadata.get("embedder"),
        "rows": metadata.get("rows"),
        "train_positives": metadata.get("train_positives"),
        "validation_positives": metadata.get("validation_positives"),
        "base_rate": round(metrics.get("base_rate", 0), 3),
        "roc_auc": round(metrics.get("roc_auc", 0), 3),
        "average_precision": round(metrics.get("average_precision", 0), 3),
        "precision_at_50": round(metrics.get("precision_at_50", 0), 3),
        "precision_at_100": round(metrics.get("precision_at_100", 0), 3),
        "precision_at_250": round(metrics.get("precision_at_250", 0), 3),
    }


class HnPredictor(Project):
    def fill_dict(self, request, d):
        d["stats"] = _stats()
        d["token_model"] = _model_metadata(TOKEN_MODEL_METADATA_PATH, "learned token model")
        d["pretrained_model"] = _model_metadata(
            PRETRAINED_MODEL_METADATA_PATH, "pretrained text embedder"
        )
        d["model"] = d["pretrained_model"] if d["pretrained_model"]["exists"] else d["token_model"]
        d["frontpage_model"] = _frontpage_model_metadata()

    def handle_request(self, handler, request):
        if handler == "stats":
            return JsonResponse(_stats())
        if handler == "model":
            return JsonResponse(
                {
                    "token_model": _model_metadata(TOKEN_MODEL_METADATA_PATH, "learned token model"),
                    "pretrained_model": _model_metadata(
                        PRETRAINED_MODEL_METADATA_PATH, "pretrained text embedder"
                    ),
                    "frontpage_model": _frontpage_model_metadata(),
                }
            )
