import numpy as np

from scibowl.dedupe.candidates import build_duplicate_candidates, lexical_overlap_score, normalize_answer_text
from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.question import NormalizedQuestion


def _question(question_id: str, text: str, answer: str) -> NormalizedQuestion:
    return NormalizedQuestion(
        question_id=question_id,
        source_type=SourceType.DATASET,
        source_id="set1",
        category=Category.CHEMISTRY,
        subcategory="acids",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=3,
        question_text=text,
        answer_text=answer,
    )


def test_normalize_answer_text_strips_prefix_and_choice_label() -> None:
    assert normalize_answer_text("ANSWER: X) Argon") == "argon"


def test_lexical_overlap_score_is_jaccard() -> None:
    score = lexical_overlap_score("alpha beta gamma", "beta gamma delta")
    assert score == 0.5


def test_build_duplicate_candidates_returns_high_similarity_pair() -> None:
    questions = [
        _question("q1", "What acid is also called muriatic acid?", "ANSWER: hydrochloric acid"),
        _question("q2", "Which acid is commonly known as muriatic acid?", "ANSWER: hydrochloric acid"),
        _question("q3", "What gas makes up most of Earth's atmosphere?", "ANSWER: nitrogen"),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    candidates = build_duplicate_candidates(questions, embeddings, threshold=0.9, top_k=2)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.question_id_a == "q1"
    assert candidate.question_id_b == "q2"
    assert candidate.same_normalized_answer is True
