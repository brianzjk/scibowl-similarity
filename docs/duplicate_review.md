# Duplicate Review

This project includes a local duplicate-mining and duplicate-review workflow for Science Bowl questions.

## What It Does

The duplicate pipeline has three stages:

1. Build semantic candidate pairs from a normalized question corpus with a local embedding model.
2. Export those pairs to CSV if you want a spreadsheet review path.
3. Review them in a lightweight local website and save decisions back to JSONL.

The current local embedding model is `mixedbread-ai/mxbai-embed-large-v1`.

## Candidate Labels

The duplicate review website uses three decisions:

- `duplicate_question`
- `similar_material`
- `false_positive`

These are stored directly in the reviewed duplicate JSONL output.

## How Candidate Mining Works

`build-duplicate-candidates`:

- reads a normalized question JSONL
- builds one embedding per question using category, subcategory, question text, and optionally the answer
- blocks comparisons by category
- computes top semantic neighbors within each category
- writes candidate pairs above the similarity threshold

The default mining threshold is intentionally lower now (`0.5`) so the review website can later narrow the active queue in the browser without requiring a separate re-export for every threshold experiment.

Each pair includes:

- embedding similarity
- lexical overlap
- question ids and source ids
- question types and answer modes
- both question texts and answer texts

## Build Candidate Pairs

Example:

```powershell
.venv\Scripts\python.exe -m scibowl.cli.main build-duplicate-candidates `
  --questions ../data/interim/question_sets/style_corpus_all.jsonl `
  --output-path ../data/interim/duplicates/style_corpus_all_candidates.jsonl `
  --summary-path ../data/interim/duplicates/style_corpus_all_candidates_summary.json `
  --device cuda `
  --top-k 10 `
  --threshold 0.5
```

## Export Reviewer CSV

Example:

```powershell
.venv\Scripts\python.exe -m scibowl.cli.main export-duplicate-candidates-csv `
  ../data/interim/duplicates/style_corpus_all_candidates.jsonl `
  ../data/interim/duplicates/style_corpus_all_candidates_review.csv
```

This writes a flat CSV with blank `review_decision` and `review_notes` columns for spreadsheet review.

## Run The Local Review Website

Example:

```powershell
.venv\Scripts\python.exe -m scibowl.cli.main review-duplicates `
  ../data/interim/duplicates/style_corpus_all_candidates.jsonl `
  --questions ../data/interim/question_sets/style_corpus_all.jsonl `
  --output-path ../data/interim/duplicates/style_corpus_all_reviewed.jsonl `
  --host 127.0.0.1 `
  --port 8765 `
  --title "Science Bowl Duplicate Review"
```

Then open:

```text
http://127.0.0.1:8765
```

## What The Website Shows

For each pair, the website shows:

- question ids
- source ids
- tournament and original round for each question
- question type and answer mode
- question text
- multiple-choice options, when present
- answer line
- embedding similarity and lexical overlap

The website lets you filter the candidate queue by a minimum embedding similarity value in the browser, and that filtering is applied server-side when loading the review session. That means you can keep one broad candidate file and avoid loading the entire low-threshold set into the page at once. The page starts with a higher default minimum similarity for responsiveness, and you can lower it when you want broader review.

## How Reviews Are Saved

The review website writes a reviewed JSONL file at the `--output-path` you provide.

It preserves the original candidate order and updates:

- `label`
- `review_status`
- `notes`

The review output only stores reviewed rows, not a full copy of the candidate set. That means you can stop and restart the review session without losing progress, even when the candidate file is large.

## Keyboard Shortcuts

- `1` mark as `duplicate_question`
- `2` mark as `similar_material`
- `3` mark as `false_positive`
- `j` next pair
- `k` previous pair

## Recommended Workflow

1. Build candidate pairs on the full style corpus.
2. Review the highest-similarity pairs first.
3. Save reviewed decisions into JSONL.
4. Use those reviewed pairs later for:
   - duplicate filtering before training
   - split cleanup to avoid leakage
   - possible embedding-model fine-tuning


## Data Location

After the repo split, the recommended convention is to keep question corpora and duplicate artifacts under `../data`.
