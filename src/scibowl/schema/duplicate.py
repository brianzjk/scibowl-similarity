from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DuplicateLabel(str, Enum):
    DUPLICATE_QUESTION = "duplicate_question"
    SIMILAR_MATERIAL = "similar_material"
    FALSE_POSITIVE = "false_positive"
    EXACT_DUPLICATE = "duplicate_question"
    NEAR_DUPLICATE_SURFACE = "duplicate_question"
    SAME_UNDERLYING_IDEA = "similar_material"
    DISTINCT = "false_positive"


class DuplicateReviewStatus(str, Enum):
    CANDIDATE = "candidate"
    REVIEWED = "reviewed"


class DuplicateCandidate(BaseModel):
    pair_id: str
    question_id_a: str
    question_id_b: str
    source_id_a: str
    source_id_b: str
    category: str
    subcategory_a: str
    subcategory_b: str
    question_type_a: str
    question_type_b: str
    answer_mode_a: str
    answer_mode_b: str
    question_text_a: str
    question_text_b: str
    answer_text_a: str
    answer_text_b: str
    embedding_similarity: float
    lexical_overlap: float
    same_subcategory: bool
    same_question_type: bool
    same_answer_mode: bool
    same_normalized_answer: bool
    review_status: DuplicateReviewStatus = DuplicateReviewStatus.CANDIDATE
    label: DuplicateLabel | None = None
    notes: str | None = None


class DuplicateMiningSummary(BaseModel):
    run_id: str
    model_name: str
    question_count: int
    candidate_count: int
    threshold: float
    top_k: int
    include_answer: bool
    blocked_by_category: bool = True
    counts_by_category: dict[str, int] = Field(default_factory=dict)
