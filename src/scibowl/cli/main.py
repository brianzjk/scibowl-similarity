from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scibowl.dedupe.browser_bundle import build_browser_corpus_bundle, build_browser_upload_bundle_from_mit_csv
from scibowl.dedupe.candidates import mine_duplicate_candidates
from scibowl.dedupe.embedding_store import build_embedding_store, default_embedding_store_dir, load_embedding_store
from scibowl.dedupe.export import export_duplicate_candidates_csv
from scibowl.dedupe.review_server import run_duplicate_review_server
from scibowl.dedupe.upload_matches import default_upload_match_dir, match_uploaded_questions, write_upload_match_artifacts
from scibowl.ingest import MitCsvSchemaError, MitCsvValidationError, parse_mit_questions_csv
from scibowl.schema.question import NormalizedQuestion
from scibowl.utils.io import read_jsonl, write_json, write_jsonl


def cmd_build_duplicate_candidates(args: argparse.Namespace) -> None:
    questions = read_jsonl(Path(args.questions), NormalizedQuestion)
    if args.max_questions:
        questions = questions[: args.max_questions]

    candidates, summary = mine_duplicate_candidates(
        questions,
        model_name=args.model_name,
        threshold=args.threshold,
        top_k=args.top_k,
        include_answer=args.include_answer,
        batch_size=args.batch_size,
        device=args.device,
        cache_folder=args.cache_folder,
    )
    write_jsonl(Path(args.output_path), candidates)
    if args.summary_path:
        write_json(Path(args.summary_path), summary.model_dump())
    print(f"Wrote {len(candidates)} duplicate candidates to {args.output_path}")


def cmd_export_duplicate_candidates_csv(args: argparse.Namespace) -> None:
    count = export_duplicate_candidates_csv(Path(args.input_path), Path(args.output_path))
    print(f"Wrote {count} duplicate review rows to {args.output_path}")


def cmd_parse_mit_questions_csv(args: argparse.Namespace) -> None:
    questions = parse_mit_questions_csv(
        Path(args.input_path),
        source_id=args.source_id,
        tournament=args.tournament,
        year=args.year,
        default_difficulty=args.default_difficulty,
    )
    write_jsonl(Path(args.output_path), questions)
    print(f"Wrote {len(questions)} normalized questions to {args.output_path}")


def cmd_build_embedding_store(args: argparse.Namespace) -> None:
    questions_path = Path(args.questions)
    questions = read_jsonl(questions_path, NormalizedQuestion)
    output_dir = Path(args.output_dir) if args.output_dir else default_embedding_store_dir(
        questions_path,
        model_name=args.model_name,
    )
    manifest = build_embedding_store(
        questions,
        output_dir=output_dir,
        source_questions_path=questions_path,
        model_name=args.model_name,
        include_answer=args.include_answer,
        batch_size=args.batch_size,
        device=args.device,
        cache_folder=args.cache_folder,
    )
    print(
        "Wrote "
        f"{manifest.question_count} question embeddings "
        f"({manifest.embedding_dimension} dims) "
        f"to {output_dir}"
    )


def cmd_build_browser_corpus_bundle(args: argparse.Namespace) -> None:
    corpus_store = load_embedding_store(Path(args.embedding_store))
    manifest = build_browser_corpus_bundle(
        corpus_store,
        output_dir=Path(args.output_dir),
    )
    print(
        "Wrote browser corpus bundle to "
        f"{args.output_dir} "
        f"({manifest.question_count} questions across {len(manifest.categories)} categories)"
    )


def cmd_build_browser_upload_bundle(args: argparse.Namespace) -> None:
    payload = build_browser_upload_bundle_from_mit_csv(
        Path(args.input_path),
        output_path=Path(args.output_path),
        source_id=args.source_id,
        tournament=args.tournament,
        year=args.year,
        default_difficulty=args.default_difficulty,
        model_name=args.model_name,
        include_answer=args.include_answer,
        batch_size=args.batch_size,
        device=args.device,
        cache_folder=args.cache_folder,
    )
    print(
        "Wrote browser upload bundle to "
        f"{args.output_path} "
        f"({payload['question_count']} questions, {payload['embedding_dimension']} dims)"
    )


def cmd_match_mit_csv(args: argparse.Namespace) -> None:
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else default_upload_match_dir(input_path)
    upload_questions = parse_mit_questions_csv(
        input_path,
        source_id=args.source_id,
        tournament=args.tournament,
        year=args.year,
        default_difficulty=args.default_difficulty,
    )
    corpus_store = load_embedding_store(Path(args.embedding_store))
    _, within_upload_matches, corpus_matches = match_uploaded_questions(
        upload_questions,
        corpus_store=corpus_store,
        threshold=args.threshold,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=args.device,
        cache_folder=args.cache_folder,
    )
    summary = write_upload_match_artifacts(
        output_dir=output_dir,
        upload_questions=upload_questions,
        within_upload_matches=within_upload_matches,
        corpus_matches=corpus_matches,
        corpus_store=corpus_store,
        input_csv_path=input_path,
        threshold=args.threshold,
        top_k=args.top_k,
    )
    print(
        "Wrote upload match artifacts to "
        f"{output_dir} "
        f"({summary['uploaded_question_count']} uploaded questions, "
        f"{summary['corpus_match_count']} corpus matches, "
        f"{summary['within_upload_match_count']} within-upload matches)"
    )


def cmd_review_duplicates(args: argparse.Namespace) -> None:
    output_path = Path(args.output_path) if args.output_path else Path(args.input_path).with_name(
        Path(args.input_path).stem + "_reviewed.jsonl"
    )
    run_duplicate_review_server(
        candidates_path=Path(args.input_path),
        questions_path=Path(args.questions),
        output_path=output_path,
        host=args.host,
        port=args.port,
        title=args.title,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scibowl-similarity")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_mit_questions_csv_cmd = subparsers.add_parser("parse-mit-questions-csv")
    parse_mit_questions_csv_cmd.add_argument("input_path")
    parse_mit_questions_csv_cmd.add_argument("output_path")
    parse_mit_questions_csv_cmd.add_argument("--source-id")
    parse_mit_questions_csv_cmd.add_argument("--tournament")
    parse_mit_questions_csv_cmd.add_argument("--year", type=int)
    parse_mit_questions_csv_cmd.add_argument("--default-difficulty", type=int, default=4)
    parse_mit_questions_csv_cmd.set_defaults(func=cmd_parse_mit_questions_csv)

    build_embedding_store_cmd = subparsers.add_parser("build-embedding-store")
    build_embedding_store_cmd.add_argument("--questions", required=True)
    build_embedding_store_cmd.add_argument("--output-dir")
    build_embedding_store_cmd.add_argument("--model-name", default="mixedbread-ai/mxbai-embed-large-v1")
    build_embedding_store_cmd.add_argument("--batch-size", type=int, default=32)
    build_embedding_store_cmd.add_argument("--device")
    build_embedding_store_cmd.add_argument("--cache-folder")
    build_embedding_store_cmd.add_argument(
        "--include-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    build_embedding_store_cmd.set_defaults(func=cmd_build_embedding_store)

    build_browser_corpus_bundle_cmd = subparsers.add_parser("build-browser-corpus-bundle")
    build_browser_corpus_bundle_cmd.add_argument("--embedding-store", required=True)
    build_browser_corpus_bundle_cmd.add_argument("--output-dir", required=True)
    build_browser_corpus_bundle_cmd.set_defaults(func=cmd_build_browser_corpus_bundle)

    build_browser_upload_bundle_cmd = subparsers.add_parser("build-browser-upload-bundle")
    build_browser_upload_bundle_cmd.add_argument("input_path")
    build_browser_upload_bundle_cmd.add_argument("output_path")
    build_browser_upload_bundle_cmd.add_argument("--source-id")
    build_browser_upload_bundle_cmd.add_argument("--tournament")
    build_browser_upload_bundle_cmd.add_argument("--year", type=int)
    build_browser_upload_bundle_cmd.add_argument("--default-difficulty", type=int, default=4)
    build_browser_upload_bundle_cmd.add_argument("--model-name", default="mixedbread-ai/mxbai-embed-large-v1")
    build_browser_upload_bundle_cmd.add_argument("--batch-size", type=int, default=32)
    build_browser_upload_bundle_cmd.add_argument("--device")
    build_browser_upload_bundle_cmd.add_argument("--cache-folder")
    build_browser_upload_bundle_cmd.add_argument(
        "--include-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    build_browser_upload_bundle_cmd.set_defaults(func=cmd_build_browser_upload_bundle)

    match_mit_csv_cmd = subparsers.add_parser("match-mit-csv")
    match_mit_csv_cmd.add_argument("input_path")
    match_mit_csv_cmd.add_argument("--embedding-store", required=True)
    match_mit_csv_cmd.add_argument("--output-dir")
    match_mit_csv_cmd.add_argument("--source-id")
    match_mit_csv_cmd.add_argument("--tournament")
    match_mit_csv_cmd.add_argument("--year", type=int)
    match_mit_csv_cmd.add_argument("--default-difficulty", type=int, default=4)
    match_mit_csv_cmd.add_argument("--threshold", type=float, default=0.5)
    match_mit_csv_cmd.add_argument("--top-k", type=int, default=10)
    match_mit_csv_cmd.add_argument("--batch-size", type=int, default=32)
    match_mit_csv_cmd.add_argument("--device")
    match_mit_csv_cmd.add_argument("--cache-folder")
    match_mit_csv_cmd.set_defaults(func=cmd_match_mit_csv)

    build_duplicate_candidates_cmd = subparsers.add_parser("build-duplicate-candidates")
    build_duplicate_candidates_cmd.add_argument("--questions", required=True)
    build_duplicate_candidates_cmd.add_argument("--output-path", required=True)
    build_duplicate_candidates_cmd.add_argument("--summary-path")
    build_duplicate_candidates_cmd.add_argument("--model-name", default="mixedbread-ai/mxbai-embed-large-v1")
    build_duplicate_candidates_cmd.add_argument("--threshold", type=float, default=0.5)
    build_duplicate_candidates_cmd.add_argument("--top-k", type=int, default=10)
    build_duplicate_candidates_cmd.add_argument("--batch-size", type=int, default=32)
    build_duplicate_candidates_cmd.add_argument("--device")
    build_duplicate_candidates_cmd.add_argument("--cache-folder")
    build_duplicate_candidates_cmd.add_argument("--max-questions", type=int)
    build_duplicate_candidates_cmd.add_argument(
        "--include-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    build_duplicate_candidates_cmd.set_defaults(func=cmd_build_duplicate_candidates)

    export_duplicate_candidates_cmd = subparsers.add_parser("export-duplicate-candidates-csv")
    export_duplicate_candidates_cmd.add_argument("input_path")
    export_duplicate_candidates_cmd.add_argument("output_path")
    export_duplicate_candidates_cmd.set_defaults(func=cmd_export_duplicate_candidates_csv)

    review_duplicates_cmd = subparsers.add_parser("review-duplicates")
    review_duplicates_cmd.add_argument("input_path")
    review_duplicates_cmd.add_argument("--questions", required=True)
    review_duplicates_cmd.add_argument("--output-path")
    review_duplicates_cmd.add_argument("--host", default="127.0.0.1")
    review_duplicates_cmd.add_argument("--port", type=int, default=8765)
    review_duplicates_cmd.add_argument("--title", default="Duplicate Review")
    review_duplicates_cmd.set_defaults(func=cmd_review_duplicates)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (MitCsvSchemaError, MitCsvValidationError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
