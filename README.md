# Hacker News Predictor

Predict Hacker News story scores from the fields submitters control: title and
URL.

Run the web project locally with:

```bash
uvx douwe hn_predictor
```

The first step is collecting historical submissions. `collect_hn.py` reads the
public Algolia Hacker News Search API and writes normalized story rows into a
local SQLite database.

## Collect Data

Fetch the newest 5,000 stories:

```bash
python collect_hn.py --max-items 5000
```

Fetch a bounded date range:

```bash
python collect_hn.py \
  --start-date 2024-01-01 \
  --end-date 2025-01-01 \
  --db data/hn_submissions.sqlite3
```

Continue an older crawl by using the same database path. Rows are upserted by
HN item id, so reruns refresh scores and comment counts without duplicating
submissions.

## Train Model

Train a small local neural model:

```bash
python train_model.py
```

For a quick smoke test:

```bash
python train_model.py --limit 1000 --epochs 1
```

Train the pretrained text-embedder version:

```bash
python train_pretrained_embedder.py
```

This defaults to `BAAI/bge-base-en-v1.5`, an English pretrained transformer
embedder. It freezes the embedder, caches dense vectors under
`data/embeddings/`, and trains a small neural regression head on
`log1p(points)`.

## Front-Page Labels

Backfill daily front-page archive labels:

```bash
python annotate_frontpage.py
```

Train a binary classifier for whether a story appeared in HN's daily `/front`
archive:

```bash
python train_frontpage_classifier.py --device mps
```

The model uses:

- input: title plus normalized URL/domain/path tokens
- text layer: Keras `TextVectorization`
- embedding: learned token embedding, default 64 dimensions
- head: small dense network
- target: `log1p(points)`
- loss: mean squared error on that log target

Artifacts are written to `data/model/`:

- `model.keras` — saved model, including the vectorizer
- `metadata.json` — metrics and training settings
- `training_log.csv` — per-epoch losses
- `validation_predictions.csv` — newest validation rows with predicted scores

## Stored Fields

The model should train only on the fields available at submission time:

- `title`
- `url`

The collector also stores target and audit fields:

- `points`
- `num_comments`
- `author`
- `created_at`
- `created_at_i`
- `story_text`
- `tags_json`
- `raw_json`

## Notes

Algolia limits each query window to 1,000 hits. The collector paginates by
moving a `created_at_i < cursor` boundary backward in time, which avoids that
window limit for long crawls.
