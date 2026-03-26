from __future__ import annotations

import csv
from pathlib import Path

from scibowl.schema.duplicate import DuplicateCandidate
from scibowl.utils.io import ensure_parent, read_jsonl


REVIEW_COLUMNS = [
    "review_decision",
    "review_notes",
    "pair_id",
    "embedding_similarity",
    "lexical_overlap",
    "same_subcategory",
    "same_question_type",
    "same_answer_mode",
    "same_normalized_answer",
    "category",
    "subcategory_a",
    "subcategory_b",
    "question_type_a",
    "question_type_b",
    "answer_mode_a",
    "answer_mode_b",
    "source_id_a",
    "source_id_b",
    "question_id_a",
    "question_id_b",
    "question_text_a",
    "question_text_b",
    "answer_text_a",
    "answer_text_b",
    "review_status",
    "label",
    "notes",
]


def export_duplicate_candidates_csv(input_path: Path, output_path: Path) -> int:
    candidates = read_jsonl(input_path, DuplicateCandidate)
    ensure_parent(output_path)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(_candidate_row(candidate))

    return len(candidates)


def _candidate_row(candidate: DuplicateCandidate) -> dict[str, object]:
    row = candidate.model_dump(mode="json")
    row["review_decision"] = ""
    row["review_notes"] = ""
    if row["label"] is None:
        row["label"] = ""
    return {column: row.get(column, "") for column in REVIEW_COLUMNS}
