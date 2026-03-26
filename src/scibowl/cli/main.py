from __future__ import annotations

import argparse
from pathlib import Path

from scibowl.dedupe.candidates import mine_duplicate_candidates
from scibowl.dedupe.export import export_duplicate_candidates_csv
from scibowl.dedupe.review_server import run_duplicate_review_server
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
    args.func(args)


if __name__ == "__main__":
    main()
