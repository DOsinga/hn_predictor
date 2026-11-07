#!/usr/bin/env python3
"""Collect Hacker News story submissions for score prediction.

The prediction problem should only use fields available to the submitter at
submission time: title and URL. This collector still stores target, timing, and
raw payload fields so future training code can filter and audit the data.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any

import requests

API_URL = "https://hn.algolia.com/api/v1/search_by_date"
DEFAULT_DB = os.path.join(os.path.dirname(__file__), "data", "hn_submissions.sqlite3")
DEFAULT_HITS_PER_PAGE = 1000


def parse_date(value: str | None, *, end: bool = False) -> int | None:
    if not value:
        return None
    if len(value) == 10:
        value = value + ("T23:59:59+00:00" if end else "T00:00:00+00:00")
    elif value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("pragma journal_mode = wal")
    conn.execute("pragma synchronous = normal")
    conn.execute(
        """
        create table if not exists submissions (
            id integer primary key,
            title text not null,
            url text,
            author text,
            points integer,
            num_comments integer,
            created_at text,
            created_at_i integer not null,
            story_text text,
            tags_json text,
            raw_json text not null,
            fetched_at text not null
        )
        """
    )
    conn.execute(
        "create index if not exists submissions_created_idx on submissions(created_at_i)"
    )
    conn.execute("create index if not exists submissions_points_idx on submissions(points)")
    conn.execute(
        """
        create table if not exists collection_runs (
            id integer primary key autoincrement,
            started_at text not null,
            finished_at text,
            start_cursor integer,
            end_cursor integer,
            min_created_at_i integer,
            max_created_at_i integer,
            rows_seen integer not null default 0,
            rows_written integer not null default 0,
            status text not null
        )
        """
    )
    conn.commit()
    return conn


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def item_id(hit: dict[str, Any]) -> int | None:
    raw = hit.get("objectID") or hit.get("story_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def normalize_hit(hit: dict[str, Any], fetched_at: str) -> tuple[Any, ...] | None:
    hn_id = item_id(hit)
    title = hit.get("title") or hit.get("story_title")
    created_at_i = hit.get("created_at_i")
    if hn_id is None or not title or created_at_i is None:
        return None
    return (
        hn_id,
        title,
        hit.get("url") or hit.get("story_url"),
        hit.get("author"),
        hit.get("points"),
        hit.get("num_comments"),
        hit.get("created_at"),
        int(created_at_i),
        hit.get("story_text"),
        json.dumps(hit.get("_tags") or [], ensure_ascii=False),
        json.dumps(hit, ensure_ascii=False, sort_keys=True),
        fetched_at,
    )


def upsert_hits(conn: sqlite3.Connection, hits: list[dict[str, Any]]) -> tuple[int, int]:
    fetched_at = utc_now()
    rows = [row for hit in hits if (row := normalize_hit(hit, fetched_at))]
    if not rows:
        return 0, 0
    before = conn.total_changes
    conn.executemany(
        """
        insert into submissions (
            id, title, url, author, points, num_comments, created_at,
            created_at_i, story_text, tags_json, raw_json, fetched_at
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(id) do update set
            title = excluded.title,
            url = excluded.url,
            author = excluded.author,
            points = excluded.points,
            num_comments = excluded.num_comments,
            created_at = excluded.created_at,
            created_at_i = excluded.created_at_i,
            story_text = excluded.story_text,
            tags_json = excluded.tags_json,
            raw_json = excluded.raw_json,
            fetched_at = excluded.fetched_at
        """,
        rows,
    )
    conn.commit()
    return len(rows), conn.total_changes - before


def build_numeric_filters(cursor: int | None, start_ts: int | None) -> str | None:
    filters = []
    if start_ts is not None:
        filters.append(f"created_at_i>={start_ts}")
    if cursor is not None:
        filters.append(f"created_at_i<{cursor}")
    return ",".join(filters) if filters else None


def fetch_page(
    session: requests.Session,
    *,
    cursor: int | None,
    start_ts: int | None,
    hits_per_page: int,
    timeout: int,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "tags": "story",
        "hitsPerPage": hits_per_page,
    }
    numeric_filters = build_numeric_filters(cursor, start_ts)
    if numeric_filters:
        params["numericFilters"] = numeric_filters
    response = session.get(API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json().get("hits") or []


def collect(args: argparse.Namespace) -> int:
    start_ts = parse_date(args.start_date)
    end_ts = parse_date(args.end_date, end=True)
    cursor = end_ts
    total_seen = 0
    total_written = 0
    min_seen = None
    max_seen = None

    conn = connect(args.db)
    started_at = utc_now()
    run_id = conn.execute(
        """
        insert into collection_runs (
            started_at, start_cursor, end_cursor, status
        ) values (?, ?, ?, ?)
        """,
        (started_at, cursor, start_ts, "running"),
    ).lastrowid
    conn.commit()

    session = requests.Session()
    session.headers.update({"User-Agent": "douwe.com hn_predictor data collector"})

    try:
        while True:
            hits = fetch_page(
                session,
                cursor=cursor,
                start_ts=start_ts,
                hits_per_page=args.hits_per_page,
                timeout=args.timeout,
            )
            if not hits:
                break

            seen, written = upsert_hits(conn, hits)
            total_seen += seen
            total_written += written

            timestamps = [int(h["created_at_i"]) for h in hits if h.get("created_at_i")]
            if not timestamps:
                break
            page_min = min(timestamps)
            page_max = max(timestamps)
            min_seen = page_min if min_seen is None else min(min_seen, page_min)
            max_seen = page_max if max_seen is None else max(max_seen, page_max)
            cursor = page_min

            print(
                f"fetched {seen:4d} rows; total={total_seen:7d}; "
                f"oldest={datetime.fromtimestamp(cursor, timezone.utc).date()}",
                flush=True,
            )

            conn.execute(
                """
                update collection_runs
                set rows_seen = ?, rows_written = ?, min_created_at_i = ?,
                    max_created_at_i = ?, end_cursor = ?
                where id = ?
                """,
                (total_seen, total_written, min_seen, max_seen, cursor, run_id),
            )
            conn.commit()

            if args.max_items and total_seen >= args.max_items:
                break
            if start_ts is not None and cursor <= start_ts:
                break
            if args.sleep:
                time.sleep(args.sleep)

        status = "complete"
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        print("Interrupted; partial data is committed.", file=sys.stderr)
        return 130
    except Exception as exc:
        status = "failed"
        print(f"Collection failed: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.execute(
            """
            update collection_runs
            set finished_at = ?, rows_seen = ?, rows_written = ?,
                min_created_at_i = ?, max_created_at_i = ?, end_cursor = ?,
                status = ?
            where id = ?
            """,
            (utc_now(), total_seen, total_written, min_seen, max_seen, cursor, status, run_id),
        )
        conn.commit()
        conn.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect historical Hacker News story submissions into SQLite."
    )
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite DB path ({DEFAULT_DB})")
    parser.add_argument("--start-date", help="Oldest creation date/time to include")
    parser.add_argument("--end-date", help="Newest creation date/time to include")
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Stop after roughly this many fetched rows; 0 means no limit",
    )
    parser.add_argument(
        "--hits-per-page",
        type=int,
        default=DEFAULT_HITS_PER_PAGE,
        help="Algolia hits per request; max is 1000",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds between requests")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    args = parser.parse_args(argv)
    if args.hits_per_page < 1 or args.hits_per_page > 1000:
        parser.error("--hits-per-page must be between 1 and 1000")
    return args


if __name__ == "__main__":
    raise SystemExit(collect(parse_args(sys.argv[1:])))
