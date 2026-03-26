import shutil
from pathlib import Path

import pytest

from scibowl.ingest import MitCsvRowError, MitCsvSchemaError, parse_mit_questions_csv
from scibowl.schema.common import AnswerMode, Category, QuestionType
from scibowl.utils.ids import make_id


def _make_temp_dir() -> Path:
    path = Path("tests_runtime") / make_id("mit_csv")
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_parse_mit_questions_csv_requires_named_columns() -> None:
    tmp_path = _make_temp_dir()
    csv_path = tmp_path / "missing_required.csv"
    csv_path.write_text(
        "Type,Category,Format,Question,W,X,Y,Z,Accept,Do NOt Accept\n"
        "Toss-up,Biology,Short Answer,What is DNA?,,,,,,\n",
        encoding="utf-8",
    )

    with pytest.raises(MitCsvSchemaError, match="Answer"):
        parse_mit_questions_csv(csv_path)

    shutil.rmtree(tmp_path)


def test_parse_mit_questions_csv_maps_optional_metadata_and_case_insensitive_headers() -> None:
    tmp_path = _make_temp_dir()
    csv_path = tmp_path / "2025-full-questions.csv"
    csv_path.write_text(
        "Type,Category,Format,Question,W,X,Y,Z,Answer,Accept,Do NOt Accept,Subcategory,Writer,Source,Date,Division (Approx),Round,,\n"
        "Toss-up,Biology,Short Answer,What organelle carries out photosynthesis?,,,,,Chloroplast,Plastid,Cell wall,Cell Biology,Sean,Campbell,5/10/2025,RR,1,,\n",
        encoding="utf-8",
    )

    questions = parse_mit_questions_csv(csv_path, source_id="mit_2025", tournament="MIT 2025")

    assert len(questions) == 1
    question = questions[0]
    assert question.question_id == "mit_2025_0001"
    assert question.category == Category.BIOLOGY
    assert question.question_type == QuestionType.TOSSUP
    assert question.answer_mode == AnswerMode.SHORT_ANSWER
    assert question.question_text == "What organelle carries out photosynthesis?"
    assert question.answer_text == "ANSWER: Chloroplast"
    assert question.answer_guidance.accept == ["Plastid"]
    assert question.answer_guidance.do_not_accept == ["Cell wall"]
    assert question.source_metadata.tournament == "MIT 2025"
    assert question.source_metadata.year == 2025
    assert question.source_metadata.round == 1
    assert question.source_metadata.writer == "Sean"
    assert question.source_metadata.source == "Campbell"
    assert question.source_metadata.date == "5/10/2025"
    assert question.source_metadata.division == "RR"
    assert question.source_metadata.source_row == 2
    assert question.subcategory == "Cell Biology"
    assert question.choices == []
    assert question.style_tags == []

    shutil.rmtree(tmp_path)


def test_parse_mit_questions_csv_expands_multiple_choice_answers_and_marks_visual_bonus() -> None:
    tmp_path = _make_temp_dir()
    csv_path = tmp_path / "mit_upload.csv"
    csv_path.write_text(
        "Type,Category,Format,Question,W,X,Y,Z,Answer,Accept,Do Not Accept,Subcategory\n"
        "Visual Bonus,Earth and Space,Multiple Choice,Which planet is known as the Red Planet?,Mercury,Venus,Earth,Mars,Z,,,Planetary Science\n",
        encoding="utf-8",
    )

    questions = parse_mit_questions_csv(csv_path, source_id="upload_batch")

    assert len(questions) == 1
    question = questions[0]
    assert question.question_type == QuestionType.BONUS
    assert question.answer_mode == AnswerMode.MULTIPLE_CHOICE
    assert question.category == Category.EARTH_SPACE
    assert question.answer_text == "ANSWER: Z) Mars"
    assert [(choice.label, choice.text) for choice in question.choices] == [
        ("W", "Mercury"),
        ("X", "Venus"),
        ("Y", "Earth"),
        ("Z", "Mars"),
    ]
    assert question.style_tags == ["visual_bonus"]

    shutil.rmtree(tmp_path)


def test_parse_mit_questions_csv_rejects_bad_multiple_choice_answer_labels() -> None:
    tmp_path = _make_temp_dir()
    csv_path = tmp_path / "bad_mc.csv"
    csv_path.write_text(
        "Type,Category,Format,Question,W,X,Y,Z,Answer,Accept,Do Not Accept\n"
        "Bonus,Chemistry,Multiple Choice,Pick one,One,Two,Three,Four,Q,,\n",
        encoding="utf-8",
    )

    with pytest.raises(MitCsvRowError, match="W, X, Y, or Z"):
        parse_mit_questions_csv(csv_path)

    shutil.rmtree(tmp_path)
