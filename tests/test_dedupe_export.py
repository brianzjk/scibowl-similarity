import csv
import shutil
from pathlib import Path

from scibowl.dedupe.export import export_duplicate_candidates_csv
from scibowl.schema.duplicate import DuplicateCandidate
from scibowl.schema.duplicate import DuplicateReviewStatus
from scibowl.utils.ids import make_id


def _make_temp_dir() -> Path:
    path = Path("tests_runtime") / make_id("case")
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_export_duplicate_candidates_csv_writes_reviewer_columns() -> None:
    tmp_path = _make_temp_dir()
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "candidates.csv"
    candidate = DuplicateCandidate(
        pair_id="dup__a__b",
        question_id_a="a",
        question_id_b="b",
        source_id_a="set1",
        source_id_b="set2",
        category="biology",
        subcategory_a="cells",
        subcategory_b="cells",
        question_type_a="tossup",
        question_type_b="tossup",
        answer_mode_a="short_answer",
        answer_mode_b="short_answer",
        question_text_a="What organelle performs photosynthesis?",
        question_text_b="Which organelle is responsible for photosynthesis?",
        answer_text_a="ANSWER: chloroplast",
        answer_text_b="ANSWER: chloroplast",
        embedding_similarity=0.95,
        lexical_overlap=0.4,
        same_subcategory=True,
        same_question_type=True,
        same_answer_mode=True,
        same_normalized_answer=True,
        review_status=DuplicateReviewStatus.CANDIDATE,
        label=None,
        notes=None,
    )
    input_path.write_text(candidate.model_dump_json() + "\n", encoding="utf-8")

    count = export_duplicate_candidates_csv(input_path, output_path)

    assert count == 1
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["review_decision"] == ""
    assert rows[0]["review_notes"] == ""
    assert rows[0]["pair_id"] == "dup__a__b"
    assert rows[0]["embedding_similarity"] == "0.95"
    assert rows[0]["question_text_a"] == "What organelle performs photosynthesis?"
    assert rows[0]["question_text_b"] == "Which organelle is responsible for photosynthesis?"
    shutil.rmtree(tmp_path)
