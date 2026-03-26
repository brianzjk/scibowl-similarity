# scibowl-similarity

Standalone Science Bowl duplicate mining and review tool.

This repo owns:

- embedding-based duplicate candidate generation
- duplicate review CSV export
- the local duplicate review website
- reviewed-label JSONL artifacts

It expects normalized question JSONL as input and writes candidate/review artifacts to explicit paths.

## Embedding Model

Duplicate mining defaults to [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1), an Apache-2.0 `sentence-transformers` embedding model from Mixedbread. The CLI default is `--model-name mixedbread-ai/mxbai-embed-large-v1`, and you can override it if you want to compare a different local Hugging Face model.

This repo uses that model for symmetric duplicate detection rather than query-vs-document retrieval:

- each question is embedded from category, subcategory, question text, and, by default, a normalized answer line
- embeddings are generated with `normalize_embeddings=True`, so the stored similarity score is cosine similarity via dot product
- comparisons are blocked by Science Bowl category before nearest-neighbor search
- the model card's retrieval prompt (`Represent this sentence for searching relevant passages:`) is not used here, because both sides of each comparison are question records
- this pipeline uses the model's default output size and does not currently enable truncation, Matryoshka shortening, or quantization

Install the optional embedding dependency with:

```powershell
python -m pip install -e ".[embeddings]"
```

## MIT CSV Import

The repo can normalize a 2025 MIT-style question sheet into `NormalizedQuestion` JSONL with:

```powershell
python -m scibowl.cli.main parse-mit-questions-csv INPUT.csv OUTPUT.jsonl `
  --source-id mit_2025_questions `
  --tournament "2025 MIT Science Bowl" `
  --year 2025
```

The sheet must include these columns:

- `Type`
- `Category`
- `Format`
- `Question`
- `W`
- `X`
- `Y`
- `Z`
- `Answer`
- `Accept`
- `Do Not Accept`

All other columns are optional. When present, `Subcategory`, `Writer`, `Source`, `Date`, `Division (Approx)`, and `Round` are copied into the normalized metadata. `Visual Bonus` rows are normalized as `bonus` questions and tagged with `visual_bonus`.

## Corpus Embedding Store

To persist embeddings for the full scraped corpus, build an on-disk store from the normalized question set:

```powershell
python -m scibowl.cli.main build-embedding-store `
  --questions ../data/interim/question_sets/style_corpus_all.jsonl `
  --output-dir ../data/processed/embeddings/style_corpus_all_mxbai_embed_large_v1 `
  --model-name mixedbread-ai/mxbai-embed-large-v1
```

That directory becomes the durable corpus artifact for the website. It contains:

- `manifest.json`
- `questions.jsonl`
- `embeddings.npy`
- `category_indices.json`

For this repo, `../data/processed/embeddings/` is the right place to keep the source-of-truth embeddings outside git. For a published site, you can either mount that directory on the app server or load the same vectors into a vector database later.

## Upload Matching

To parse a MIT-format upload and search it both against the persistent corpus and within the uploaded file itself:

```powershell
python -m scibowl.cli.main match-mit-csv INPUT.csv `
  --embedding-store ../data/processed/embeddings/style_corpus_all_mxbai_embed_large_v1 `
  --output-dir ../data/interim/uploads/my_upload
```

This writes:

- `uploaded_questions.jsonl`
- `matches_against_corpus.jsonl`
- `matches_within_upload.jsonl`
- `corpus_match_questions.jsonl`
- `summary.json`

`corpus_match_questions.jsonl` contains the uploaded questions plus only the matched corpus questions, so you can point the existing review app at `matches_against_corpus.jsonl` without copying the entire corpus question file.

## Static Browser Site

If you want to publish this as a free or low-cost static site, convert the persistent corpus store into browser bundle shards under `docs/corpus`:

```powershell
python -m scibowl.cli.main build-browser-corpus-bundle `
  --embedding-store ../data/processed/embeddings/style_corpus_all_mxbai_embed_large_v1 `
  --exclude-source-id mit_2025 `
  --output-dir docs/corpus
```

That writes:

- `docs/corpus/manifest.json`
- `docs/corpus/<category>.questions.json`
- `docs/corpus/<category>.embeddings.f32`

The current full corpus is sharded per category, so each embedding file stays small enough for common static hosts. The website in `docs/index.html` loads `./corpus/manifest.json` by default and runs similarity search entirely in the browser.

If you need to exclude a source from the published corpus, `build-embedding-store` and `build-browser-corpus-bundle` both accept repeatable `--exclude-source-id` and `--exclude-tournament` flags.

This deployment model has two important properties:

- the published site is static HTML, CSS, JS, and public corpus bundle files, so it can be hosted on GitHub Pages, Vercel, or Cloudflare Pages
- the corpus bundle is public-downloadable, but user upload bundles stay local to the browser unless the user chooses to share them elsewhere

## Local Upload Bundle Generation

Users should generate upload embeddings on their own machine, then load the resulting JSON bundle into the static website:

```powershell
python -m pip install -e ".[embeddings]"
python -m scibowl.cli.main build-browser-upload-bundle `
  "YOUR_QUESTIONS.csv" `
  "your_upload_bundle.json" `
  --source-id your_upload `
  --tournament "Your Tournament" `
  --year 2026
```

The input CSV must use the MIT columns documented above. The output `your_upload_bundle.json` contains the normalized questions plus their embeddings and is the file users select in the browser app.

Run the CLI with:

```powershell
python -m scibowl.cli.main --help
```

Use `../data` for normalized question corpora and duplicate artifacts if you are keeping this repo next to `scibowl-gpt`.
