#!/bin/zsh
set -u

ROOT="/Users/douwe/Dropbox/Douwe/Proj/douweosinga/djangosite"
PROJECT="$ROOT/projects/hn_predictor"
LOG_DIR="$PROJECT/data/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT" || exit 1
source "$ROOT/venv3/bin/activate"

{
  echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') scheduled HN crawl ==="
  python "$PROJECT/collect_hn.py" --max-items 5000 --sleep 0.1
  python "$PROJECT/annotate_frontpage.py" --sleep 10 --retries 1 --max-missing-days 1
  sqlite3 "$PROJECT/data/hn_submissions.sqlite3" \
    "select 'submissions', count(*), min(date(created_at)), max(date(created_at)) from submissions;
     select 'frontpage_days', count(*), min(day), max(day), sum(item_count) from frontpage_days;
     select 'frontpage_matches', count(distinct s.id) from submissions s join frontpage_items f on f.item_id = s.id;"
} >> "$LOG_DIR/scheduled_crawl.log" 2>&1
