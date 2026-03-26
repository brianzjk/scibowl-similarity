# scibowl-similarity

Standalone Science Bowl duplicate mining and review tool.

This repo owns:

- embedding-based duplicate candidate generation
- duplicate review CSV export
- the local duplicate review website
- reviewed-label JSONL artifacts

It expects normalized question JSONL as input and writes candidate/review artifacts to explicit paths.

Run the CLI with:

```powershell
python -m scibowl.cli.main --help
```

Use `../data` for normalized question corpora and duplicate artifacts if you are keeping this repo next to `scibowl-gpt`.
