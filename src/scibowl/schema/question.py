from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .common import AnswerMode, Category, QuestionType, SourceType


class Choice(BaseModel):
    label: str
    text: str


class Provenance(BaseModel):
    raw_file: str | None = None
    parser_version: str | None = None
    review_status: str | None = None


class SourceMetadata(BaseModel):
    source_row: int | None = None
    round: int | None = None
    question_number: int | None = None
    tournament: str | None = None
    year: int | None = None
    original_difficulty: float | None = None
    original_quality: float | None = None
    writer: str | None = None
    source: str | None = None
    date: str | None = None
    division: str | None = None


class AnswerGuidance(BaseModel):
    accept: list[str] = Field(default_factory=list)
    do_not_accept: list[str] = Field(default_factory=list)


class NormalizedQuestion(BaseModel):
    question_id: str
    source_type: SourceType
    source_id: str
    category: Category
    subcategory: str
    question_type: QuestionType
    answer_mode: AnswerMode
    difficulty: int
    question_text: str
    answer_text: str
    choices: list[Choice] = Field(default_factory=list)
    answer_guidance: AnswerGuidance = Field(default_factory=AnswerGuidance)
    style_tags: list[str] = Field(default_factory=list)
    content_tags: list[str] = Field(default_factory=list)
    source_metadata: SourceMetadata = Field(default_factory=SourceMetadata)
    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, value: int) -> int:
        if not 1 <= value <= 7:
            raise ValueError("difficulty must be between 1 and 7")
        return value


class AcceptedQuestion(BaseModel):
    question_id: str
    source_type: SourceType = SourceType.GENERATED
    origin_spec_id: str
    accepted_draft_id: str
    category: Category
    subcategory: str
    question_type: QuestionType
    answer_mode: AnswerMode
    difficulty: int
    question_text: str
    answer_text: str
    choices: list[Choice] = Field(default_factory=list)
    citations: list[dict[str, str | None]] = Field(default_factory=list)
    style_reference_ids: list[str] = Field(default_factory=list)
    verification: dict[str, object] = Field(default_factory=dict)
    provenance: dict[str, object] = Field(default_factory=dict)
