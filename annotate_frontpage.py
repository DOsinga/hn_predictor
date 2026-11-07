#!/usr/bin/env python3
"""Backfill whether collected HN stories appeared on HN's daily front pages."""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

DEFAULT_DB = os.path.join(os.path.dirname(__file__), "data", "hn_submissions.sqlite3")
FRONT_URL = "https://news.ycombinator.com/front"


class RateLimited(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_day(value: str) -> date:
    return date.fromisoformat(value)


def date_range(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def db_date_range(conn: sqlite3.Connection) -> tuple[date, date]:
    row = conn.execute(
        "select min(date(created_at)), max(date(created_at)) from submissions"
    ).fetchone()
    if not row or not row[0] or not row[1]:
        raise SystemExit("No submissions found")
    return parse_day(row[0]), parse_day(row[1])


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists frontpage_days (
            day text primary key,
            fetched_at text not null,
            source_url text not null,
            item_count integer not null
        )
        """
    )
    conn.execute(
        """
        create table if not exists frontpage_items (
            day text not null,
            item_id integer not null,
            rank integer,
            title text,
            url text,
            site text,
            points integer,
            comments integer,
            author text,
            age_text text,
            primary key(day, item_id)
        )
        """
    )
    conn.execute("create index if not exists frontpage_items_item_idx on frontpage_items(item_id)")
    conn.commit()


def parse_int(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"\d+", text.replace(",", ""))
    return int(match.group()) if match else None


def parse_front_page(html: str, day: date) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for row in soup.select("tr.athing"):
        item_id = parse_int(row.get("id"))
        if item_id is None:
            continue
        rank = parse_int(row.select_one(".rank").get_text(" ", strip=True) if row.select_one(".rank") else None)
        title_link = row.select_one(".titleline > a")
        title = title_link.get_text(" ", strip=True) if title_link else None
        url = urljoin("https://news.ycombinator.com/", title_link.get("href")) if title_link else None
        site = row.select_one(".sitestr")
        subtext = row.find_next_sibling("tr")
        score = None
        comments = None
        author = None
        age_text = None
        if subtext:
            score_node = subtext.select_one(".score")
            score = parse_int(score_node.get_text(" ", strip=True) if score_node else None)
            user_node = subtext.select_one(".hnuser")
            author = user_node.get_text(" ", strip=True) if user_node else None
            age_node = subtext.select_one(".age")
            age_text = age_node.get_text(" ", strip=True) if age_node else None
            links = subtext.select("a")
            for link in links:
                text = link.get_text(" ", strip=True).lower()
                if "comment" in text or text == "discuss":
                    comments = 0 if text == "discuss" else parse_int(text)
        out.append(
            {
                "day": day.isoformat(),
                "item_id": item_id,
                "rank": rank,
                "title": title,
                "url": url,
                "site": site.get_text(" ", strip=True) if site else None,
                "points": score,
                "comments": comments,
                "author": author,
                "age_text": age_text,
            }
        )
    return out


def fetch_day(session: requests.Session, day: date, timeout: int, retries: int) -> tuple[str, str]:
    for attempt in range(retries + 1):
        response = session.get(FRONT_URL, params={"day": day.isoformat()}, timeout=timeout)
        if response.status_code != 429:
            response.raise_for_status()
            return response.url, response.text
        if attempt == retries:
            raise RateLimited(f"{day}: HTTP 429 Too Many Requests")
        wait = 2 ** attempt
        print(f"{day}: rate limited; waiting {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError("unreachable")


def store_day(conn: sqlite3.Connection, day: date, source_url: str, items: list[dict]) -> None:
    conn.executemany(
        """
        insert into frontpage_items (
            day, item_id, rank, title, url, site, points, comments, author, age_text
        ) values (
            :day, :item_id, :rank, :title, :url, :site, :points, :comments, :author, :age_text
        )
        on conflict(day, item_id) do update set
            rank = excluded.rank,
            title = excluded.title,
            url = excluded.url,
            site = excluded.site,
            points = excluded.points,
            comments = excluded.comments,
            author = excluded.author,
            age_text = excluded.age_text
        """,
        items,
    )
    conn.execute(
        """
        insert into frontpage_days (day, fetched_at, source_url, item_count)
        values (?, ?, ?, ?)
        on conflict(day) do update set
            fetched_at = excluded.fetched_at,
            source_url = excluded.source_url,
            item_count = excluded.item_count
        """,
        (day.isoformat(), utc_now(), source_url, len(items)),
    )
    conn.commit()


def annotate(args: argparse.Namespace) -> int:
    conn = sqlite3.connect(args.db)
    ensure_schema(conn)
    db_start, db_end = db_date_range(conn)
    start = parse_day(args.start_date) if args.start_date else db_start
    end = parse_day(args.end_date) if args.end_date else db_end

    session = requests.Session()
    session.headers.update({"User-Agent": "douwe.com hn_predictor frontpage backfill"})
    total = 0
    fetched_missing_days = 0
    for day in date_range(start, end):
        if not args.refresh and conn.execute(
            "select 1 from frontpage_days where day = ?", (day.isoformat(),)
        ).fetchone():
            count = conn.execute(
                "select item_count from frontpage_days where day = ?", (day.isoformat(),)
            ).fetchone()[0]
            print(f"{day}: already have {count} items")
            total += count
            continue
        if args.max_missing_days and fetched_missing_days >= args.max_missing_days:
            print(f"Stopping after {fetched_missing_days} missing day(s).")
            break
        try:
            source_url, html = fetch_day(session, day, args.timeout, args.retries)
        except RateLimited as e:
            print(e)
            if args.stop_on_rate_limit:
                print("Stopping on rate limit; committed progress is preserved.")
                break
            raise
        items = parse_front_page(html, day)
        store_day(conn, day, source_url, items)
        fetched_missing_days += 1
        total += len(items)
        print(f"{day}: stored {len(items)} front-page items")
        if args.sleep:
            time.sleep(args.sleep)

    labeled = conn.execute(
        """
        select count(distinct s.id)
        from submissions s
        join frontpage_items f on f.item_id = s.id
        """
    ).fetchone()[0]
    stories = conn.execute("select count(*) from submissions").fetchone()[0]
    print(f"Matched {labeled} of {stories} collected stories to daily front pages.")
    conn.close()
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate collected HN stories with front-page archive data.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--stop-on-rate-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-missing-days",
        type=int,
        default=0,
        help="Stop after fetching this many missing days; 0 means no limit.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(annotate(parse_args(sys.argv[1:])))
