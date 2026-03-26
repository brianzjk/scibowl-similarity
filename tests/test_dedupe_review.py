import shutil
from pathlib import Path

from scibowl.dedupe.review import DuplicateReviewStore, build_round_label
from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.duplicate import DuplicateCandidate
from scibowl.schema.question import Choice, NormalizedQuestion, Provenance, SourceMetadata
from scibowl.utils.ids import make_id


def _make_temp_dir() -> Path:
    path = Path("tests_runtime") / make_id("case")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _question(
    question_id: str,
    *,
    round_value: int | None = None,
    raw_file: str | None = None,
    answer_mode: AnswerMode = AnswerMode.SHORT_ANSWER,
    choices: list[Choice] | None = None,
) -> NormalizedQuestion:
    return NormalizedQuestion(
        question_id=question_id,
        source_type=SourceType.PACKET,
        source_id="packet_set",
        category=Category.BIOLOGY,
        subcategory="cells",
        question_type=QuestionType.TOSSUP,
        answer_mode=answer_mode,
        difficulty=3,
        question_text=f"Question {question_id}",
        answer_text="ANSWER: chloroplast",
        choices=choices or [],
        source_metadata=SourceMetadata(tournament="Test Tournament", round=round_value),
        provenance=Provenance(raw_file=raw_file),
    )


def test_build_round_label_prefers_structured_round() -> None:
    question = _question("q1", round_value=4)
    assert build_round_label(question) == "Round 4"


def test_build_round_label_falls_back_to_raw_file_stem() -> None:
    question = _question("q1", raw_file="data/raw/packets/stanford_2025/stanford_2025__de_01.pdf")
    assert build_round_label(question) == "DE 01"


def test_duplicate_review_store_saves_review_and_exposes_rounds() -> None:
    tmp_path = _make_temp_dir()
    questions_path = tmp_path / "questions.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "reviewed.jsonl"

    questions = [
        _question(
            "q1",
            round_value=2,
            answer_mode=AnswerMode.MULTIPLE_CHOICE,
            choices=[
                Choice(label="W", text="Golgi apparatus"),
                Choice(label="X", text="Chloroplast"),
                Choice(label="Y", text="Lysosome"),
                Choice(label="Z", text="Ribosome"),
            ],
        ),
        _question("q2", raw_file="data/raw/packets/nsb_set_15/nsb_set_15__round_07.pdf"),
    ]
    questions_path.write_text("\n".join(question.model_dump_json() for question in questions) + "\n", encoding="utf-8")

    candidate = DuplicateCandidate(
        pair_id="dup__q1__q2",
        question_id_a="q1",
        question_id_b="q2",
        source_id_a="set_a",
        source_id_b="set_b",
        category="biology",
        subcategory_a="cells",
        subcategory_b="cells",
        question_type_a="tossup",
        question_type_b="tossup",
        answer_mode_a="short_answer",
        answer_mode_b="short_answer",
        question_text_a="Question q1",
        question_text_b="Question q2",
        answer_text_a="ANSWER: chloroplast",
        answer_text_b="ANSWER: chloroplast",
        embedding_similarity=0.94,
        lexical_overlap=0.5,
        same_subcategory=True,
        same_question_type=True,
        same_answer_mode=True,
        same_normalized_answer=True,
    )
    candidates_path.write_text(candidate.model_dump_json() + "\n", encoding="utf-8")

    store = DuplicateReviewStore(candidates_path, questions_path, output_path)
    payload = store.candidate_payload("dup__q1__q2")

    assert payload["origin_a"]["round"] == "Round 2"
    assert payload["origin_b"]["round"] == "ROUND 07"
    assert payload["choices_a"] == [
        {"label": "W", "text": "Golgi apparatus"},
        {"label": "X", "text": "Chloroplast"},
        {"label": "Y", "text": "Lysosome"},
        {"label": "Z", "text": "Ribosome"},
    ]
    assert payload["choices_b"] == []

    updated = store.save_review(pair_id="dup__q1__q2", label="similar_material", notes="same concept family")

    assert updated["label"] == "similar_material"
    assert output_path.exists()
    assert "same concept family" in output_path.read_text(encoding="utf-8")
    shutil.rmtree(tmp_path)


def test_duplicate_review_store_filters_by_min_similarity_and_writes_sparse_overlay() -> None:
    tmp_path = _make_temp_dir()
    questions_path = tmp_path / "questions.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "reviewed.jsonl"

    questions = [_question("q1"), _question("q2"), _question("q3")]
    questions_path.write_text("\n".join(question.model_dump_json() for question in questions) + "\n", encoding="utf-8")

    candidates = [
        DuplicateCandidate(
            pair_id="dup__q1__q2",
            question_id_a="q1",
            question_id_b="q2",
            source_id_a="set_a",
            source_id_b="set_b",
            category="biology",
            subcategory_a="cells",
            subcategory_b="cells",
            question_type_a="tossup",
            question_type_b="tossup",
            answer_mode_a="short_answer",
            answer_mode_b="short_answer",
            question_text_a="Question q1",
            question_text_b="Question q2",
            answer_text_a="ANSWER: chloroplast",
            answer_text_b="ANSWER: chloroplast",
            embedding_similarity=0.96,
            lexical_overlap=0.5,
            same_subcategory=True,
            same_question_type=True,
            same_answer_mode=True,
            same_normalized_answer=True,
        ),
        DuplicateCandidate(
            pair_id="dup__q1__q3",
            question_id_a="q1",
            question_id_b="q3",
            source_id_a="set_a",
            source_id_b="set_b",
            category="biology",
            subcategory_a="cells",
            subcategory_b="cells",
            question_type_a="tossup",
            question_type_b="tossup",
            answer_mode_a="short_answer",
            answer_mode_b="short_answer",
            question_text_a="Question q1",
            question_text_b="Question q3",
            answer_text_a="ANSWER: chloroplast",
            answer_text_b="ANSWER: chloroplast",
            embedding_similarity=0.62,
            lexical_overlap=0.4,
            same_subcategory=True,
            same_question_type=True,
            same_answer_mode=True,
            same_normalized_answer=True,
        ),
    ]
    candidates_path.write_text("\n".join(candidate.model_dump_json() for candidate in candidates) + "\n", encoding="utf-8")

    store = DuplicateReviewStore(candidates_path, questions_path, output_path)
    filtered = store.session_items(filter_name="all", min_similarity=0.9)

    assert [item["pair_id"] for item in filtered] == ["dup__q1__q2"]

    store.save_review(pair_id="dup__q1__q2", label="duplicate_question", notes=None)
    written_lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(written_lines) == 1
    assert "dup__q1__q2" in written_lines[0]
    assert "dup__q1__q3" not in written_lines[0]
    shutil.rmtree(tmp_path)
