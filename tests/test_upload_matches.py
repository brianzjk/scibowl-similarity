import shutil
from pathlib import Path

import numpy as np

from scibowl.dedupe.candidates import build_candidate
from scibowl.dedupe.embedding_store import build_embedding_store, load_embedding_store
from scibowl.dedupe.upload_matches import build_corpus_match_candidates, write_upload_match_artifacts
from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.question import NormalizedQuestion
from scibowl.utils.ids import make_id


def _make_temp_dir() -> Path:
    path = Path("tests_runtime") / make_id("upload_matches")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _question(question_id: str, *, source_id: str, category: Category, text: str, answer: str) -> NormalizedQuestion:
    return NormalizedQuestion(
        question_id=question_id,
        source_type=SourceType.DATASET,
        source_id=source_id,
        category=category,
        subcategory="core",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        question_text=text,
        answer_text=f"ANSWER: {answer}",
    )


class FakeEmbedder:
    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = embeddings

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._embeddings


def test_build_corpus_match_candidates_blocks_by_category_and_preserves_upload_side() -> None:
    tmp_path = _make_temp_dir()
    store_dir = tmp_path / "store"
    corpus_questions = [
        _question("corpus_bio_1", source_id="corpus", category=Category.BIOLOGY, text="Bio 1", answer="A"),
        _question("corpus_bio_2", source_id="corpus", category=Category.BIOLOGY, text="Bio 2", answer="B"),
        _question("corpus_phys_1", source_id="corpus", category=Category.PHYSICS, text="Phys 1", answer="C"),
    ]
    corpus_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    build_embedding_store(
        corpus_questions,
        output_dir=store_dir,
        model_name="fake-model",
        embedder=FakeEmbedder(corpus_embeddings),
    )
    corpus_store = load_embedding_store(store_dir, mmap_mode=None)

    upload_questions = [
        _question("upload_bio_1", source_id="upload", category=Category.BIOLOGY, text="Upload bio", answer="A"),
        _question("upload_phys_1", source_id="upload", category=Category.PHYSICS, text="Upload phys", answer="C"),
    ]
    upload_embeddings = np.asarray(
        [
            [0.99, 0.01],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    candidates = build_corpus_match_candidates(
        upload_questions,
        upload_embeddings,
        corpus_store,
        threshold=0.5,
        top_k=2,
    )

    assert [(candidate.question_id_a, candidate.question_id_b) for candidate in candidates] == [
        ("upload_phys_1", "corpus_phys_1"),
        ("upload_bio_1", "corpus_bio_1"),
        ("upload_bio_1", "corpus_bio_2"),
    ]
    assert all(candidate.source_id_a == "upload" for candidate in candidates)
    assert all(candidate.source_id_b == "corpus" for candidate in candidates)

    shutil.rmtree(tmp_path)


def test_write_upload_match_artifacts_writes_reviewable_subset() -> None:
    tmp_path = _make_temp_dir()
    store_dir = tmp_path / "store"
    output_dir = tmp_path / "outputs"
    upload_question = _question("upload_bio_1", source_id="upload", category=Category.BIOLOGY, text="Upload bio", answer="A")
    corpus_question = _question("corpus_bio_1", source_id="corpus", category=Category.BIOLOGY, text="Corpus bio", answer="A")

    build_embedding_store(
        [corpus_question],
        output_dir=store_dir,
        model_name="fake-model",
        embedder=FakeEmbedder(np.asarray([[1.0, 0.0]], dtype=np.float32)),
    )
    corpus_store = load_embedding_store(store_dir, mmap_mode=None)

    corpus_matches = [build_candidate(upload_question, corpus_question, 0.95, preserve_order=True)]
    summary = write_upload_match_artifacts(
        output_dir=output_dir,
        upload_questions=[upload_question],
        within_upload_matches=[],
        corpus_matches=corpus_matches,
        corpus_store=corpus_store,
        input_csv_path=Path("upload.csv"),
        threshold=0.5,
        top_k=10,
    )

    review_questions_path = output_dir / "corpus_match_questions.jsonl"
    review_questions = review_questions_path.read_text(encoding="utf-8")

    assert summary["uploaded_question_count"] == 1
    assert summary["corpus_match_count"] == 1
    assert "upload_bio_1" in review_questions
    assert "corpus_bio_1" in review_questions

    shutil.rmtree(tmp_path)
