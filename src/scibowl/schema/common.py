from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    PHYSICS = "physics"
    EARTH_SPACE = "earth_space"
    MATH = "math"
    ENERGY = "energy"


class QuestionType(str, Enum):
    TOSSUP = "tossup"
    BONUS = "bonus"


class AnswerMode(str, Enum):
    SHORT_ANSWER = "short_answer"
    MULTIPLE_CHOICE = "multiple_choice"


class SourceType(str, Enum):
    PACKET = "packet"
    DATASET = "dataset"
    TEXTBOOK = "textbook"
    GENERATED = "generated"


class Verdict(str, Enum):
    PASS = "pass"
    REVISE = "revise"
    FAIL = "fail"


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    locator: str | None = None
    text: str | None = None


class ModelInfo(BaseModel):
    provider: str
    model_name: str
    prompt_version: str


class TimestampedModel(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
