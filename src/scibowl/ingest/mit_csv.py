from __future__ import annotations

import csv
import re
from pathlib import Path

from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.question import AnswerGuidance, Choice, NormalizedQuestion, Provenance, SourceMetadata
from scibowl.utils.ids import slugify


MIT_CSV_PARSER_VERSION = "mit_csv_v1"
CHOICE_LABELS = ("W", "X", "Y", "Z")

REQUIRED_COLUMNS = {
    "type": "Type",
    "category": "Category",
    "format": "Format",
    "question": "Question",
    "w": "W",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "answer": "Answer",
    "accept": "Accept",
    "do not accept": "Do Not Accept",
}

OPTIONAL_COLUMNS = {
    "subcategory": "Subcategory",
    "writer": "Writer",
    "source": "Source",
    "date": "Date",
    "division (approx)": "Division (Approx)",
    "round": "Round",
}

CATEGORY_MAP = {
    "biology": Category.BIOLOGY,
    "chemistry": Category.CHEMISTRY,
    "earth and space": Category.EARTH_SPACE,
    "earth & space": Category.EARTH_SPACE,
    "earth/space": Category.EARTH_SPACE,
    "energy": Category.ENERGY,
    "math": Category.MATH,
    "physics": Category.PHYSICS,
}

QUESTION_TYPE_MAP = {
    "toss-up": QuestionType.TOSSUP,
    "tossup": QuestionType.TOSSUP,
    "bonus": QuestionType.BONUS,
    "visual bonus": QuestionType.BONUS,
}

ANSWER_MODE_MAP = {
    "short answer": AnswerMode.SHORT_ANSWER,
    "multiple choice": AnswerMode.MULTIPLE_CHOICE,
}

CHOICE_ANSWER_PATTERN = re.compile(r"^\s*([WXYZ])(?:\s*[\)\.\:\-]\s*.*)?$", flags=re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


class MitCsvSchemaError(ValueError):
    pass


class MitCsvRowError(ValueError):
    pass


def parse_mit_questions_csv(
    path: Path,
    *,
    source_id: str | None = None,
    tournament: str | None = None,
    year: int | None = None,
    default_difficulty: int = 4,
    parser_version: str = MIT_CSV_PARSER_VERSION,
) -> list[NormalizedQuestion]:
    if not 1 <= default_difficulty <= 7:
        raise ValueError("default_difficulty must be between 1 and 7")

    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header_map = _build_header_map(reader.fieldnames or [])

        resolved_source_id = source_id or slugify(resolved_path.stem)
        resolved_tournament = tournament or resolved_path.stem
        resolved_year = year if year is not None else _infer_year(resolved_tournament) or _infer_year(resolved_source_id)

        questions: list[NormalizedQuestion] = []
        for row_index, row in enumerate(reader, start=1):
            if _row_is_blank(row, header_map):
                continue
            questions.append(
                _parse_question_row(
                    row,
                    row_index=row_index,
                    path=resolved_path,
                    source_id=resolved_source_id,
                    tournament=resolved_tournament,
                    year=resolved_year,
                    default_difficulty=default_difficulty,
                    parser_version=parser_version,
                    header_map=header_map,
                )
            )
    return questions


def _build_header_map(fieldnames: list[str]) -> dict[str, str]:
    aliases = {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}
    header_map: dict[str, str] = {}

    for header in fieldnames:
        clean_header = _clean_text(header)
        if not clean_header:
            continue
        normalized = _normalize_header(clean_header)
        canonical = aliases.get(normalized, clean_header)
        if canonical in header_map:
            raise MitCsvSchemaError(f"Duplicate column after normalization: {clean_header}")
        header_map[canonical] = clean_header

    missing = [canonical for canonical in REQUIRED_COLUMNS.values() if canonical not in header_map]
    if missing:
        raise MitCsvSchemaError("Missing required MIT CSV columns: " + ", ".join(missing))
    return header_map


def _parse_question_row(
    row: dict[str, str],
    *,
    row_index: int,
    path: Path,
    source_id: str,
    tournament: str,
    year: int | None,
    default_difficulty: int,
    parser_version: str,
    header_map: dict[str, str],
) -> NormalizedQuestion:
    line_number = row_index + 1
    raw_type = _get_column(row, header_map, "Type")
    raw_category = _get_column(row, header_map, "Category")
    raw_format = _get_column(row, header_map, "Format")
    question_text = _require_value(_get_column(row, header_map, "Question"), line_number, "Question")
    raw_answer = _require_value(_get_column(row, header_map, "Answer"), line_number, "Answer")
    accept = _get_column(row, header_map, "Accept")
    do_not_accept = _get_column(row, header_map, "Do Not Accept")

    question_type = _parse_question_type(raw_type, line_number)
    category = _parse_category(raw_category, line_number)
    answer_mode = _parse_answer_mode(raw_format, line_number)
    subcategory = _get_column(row, header_map, "Subcategory")
    style_tags = ["visual_bonus"] if _normalize_header(raw_type) == "visual bonus" else []

    choices: list[Choice] = []
    if answer_mode == AnswerMode.MULTIPLE_CHOICE:
        choices = _parse_choices(row, header_map, line_number)

    answer_text = _build_answer_text(raw_answer, answer_mode=answer_mode, choices=choices, line_number=line_number)

    source_metadata = SourceMetadata(
        source_row=line_number,
        round=_parse_optional_int(_get_column(row, header_map, "Round"), line_number, "Round"),
        tournament=tournament,
        year=year,
        writer=_get_column(row, header_map, "Writer") or None,
        source=_get_column(row, header_map, "Source") or None,
        date=_get_column(row, header_map, "Date") or None,
        division=_get_column(row, header_map, "Division (Approx)") or None,
    )

    answer_guidance = AnswerGuidance(
        accept=[accept] if accept else [],
        do_not_accept=[do_not_accept] if do_not_accept else [],
    )

    return NormalizedQuestion(
        question_id=f"{slugify(source_id)}_{row_index:04d}",
        source_type=SourceType.DATASET,
        source_id=source_id,
        category=category,
        subcategory=subcategory,
        question_type=question_type,
        answer_mode=answer_mode,
        difficulty=default_difficulty,
        question_text=question_text,
        answer_text=answer_text,
        choices=choices,
        answer_guidance=answer_guidance,
        style_tags=style_tags,
        source_metadata=source_metadata,
        provenance=Provenance(raw_file=str(path), parser_version=parser_version),
    )


def _parse_choices(row: dict[str, str], header_map: dict[str, str], line_number: int) -> list[Choice]:
    choices: list[Choice] = []
    for label in CHOICE_LABELS:
        text = _get_column(row, header_map, label)
        if not text:
            raise MitCsvRowError(f"Row {line_number}: multiple-choice row is missing option {label}")
        choices.append(Choice(label=label, text=text))
    return choices


def _build_answer_text(
    raw_answer: str,
    *,
    answer_mode: AnswerMode,
    choices: list[Choice],
    line_number: int,
) -> str:
    if answer_mode != AnswerMode.MULTIPLE_CHOICE:
        return f"ANSWER: {raw_answer}"

    match = CHOICE_ANSWER_PATTERN.match(raw_answer)
    if not match:
        raise MitCsvRowError(
            f"Row {line_number}: multiple-choice answers must reference one of W, X, Y, or Z; got {raw_answer!r}"
        )

    label = match.group(1).upper()
    choice_lookup = {choice.label: choice.text for choice in choices}
    return f"ANSWER: {label}) {choice_lookup[label]}"


def _parse_category(value: str, line_number: int) -> Category:
    normalized = _normalize_header(value)
    try:
        return CATEGORY_MAP[normalized]
    except KeyError as exc:
        raise MitCsvRowError(f"Row {line_number}: unsupported Category value {value!r}") from exc


def _parse_question_type(value: str, line_number: int) -> QuestionType:
    normalized = _normalize_header(value)
    try:
        return QUESTION_TYPE_MAP[normalized]
    except KeyError as exc:
        raise MitCsvRowError(f"Row {line_number}: unsupported Type value {value!r}") from exc


def _parse_answer_mode(value: str, line_number: int) -> AnswerMode:
    normalized = _normalize_header(value)
    try:
        return ANSWER_MODE_MAP[normalized]
    except KeyError as exc:
        raise MitCsvRowError(f"Row {line_number}: unsupported Format value {value!r}") from exc


def _parse_optional_int(value: str, line_number: int, column_name: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise MitCsvRowError(f"Row {line_number}: {column_name} must be an integer if present; got {value!r}") from exc


def _get_column(row: dict[str, str], header_map: dict[str, str], canonical_name: str) -> str:
    header_name = header_map.get(canonical_name)
    if header_name is None:
        return ""
    return _clean_text(row.get(header_name))


def _row_is_blank(row: dict[str, str], header_map: dict[str, str]) -> bool:
    for header_name in header_map.values():
        if _clean_text(row.get(header_name)):
            return False
    return True


def _require_value(value: str, line_number: int, column_name: str) -> str:
    if value:
        return value
    raise MitCsvRowError(f"Row {line_number}: {column_name} is required")


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _infer_year(value: str) -> int | None:
    match = YEAR_PATTERN.search(value)
    if not match:
        return None
    return int(match.group(1))
