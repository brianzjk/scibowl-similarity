import shutil
from pathlib import Path

import numpy as np

from scibowl.dedupe.embedding_store import build_embedding_store, load_embedding_store
from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.question import NormalizedQuestion
from scibowl.utils.ids import make_id


def _make_temp_dir() -> Path:
    path = Path("tests_runtime") / make_id("embedding_store")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _question(question_id: str, *, category: Category) -> NormalizedQuestion:
    return NormalizedQuestion(
        question_id=question_id,
        source_type=SourceType.DATASET,
        source_id="style_corpus",
        category=category,
        subcategory="core",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        question_text=f"Question {question_id}",
        answer_text=f"ANSWER: Answer {question_id}",
    )


class FakeEmbedder:
    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = embeddings
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return self._embeddings


def test_build_embedding_store_writes_self_contained_artifacts() -> None:
    tmp_path = _make_temp_dir()
    output_dir = tmp_path / "store"
    questions = [
        _question("q1", category=Category.BIOLOGY),
        _question("q2", category=Category.CHEMISTRY),
        _question("q3", category=Category.BIOLOGY),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    embedder = FakeEmbedder(embeddings)

    manifest = build_embedding_store(
        questions,
        output_dir=output_dir,
        source_questions_path=Path("corpus.jsonl"),
        model_name="fake-model",
        include_answer=False,
        embedder=embedder,
    )

    assert manifest.model_name == "fake-model"
    assert manifest.question_count == 3
    assert manifest.embedding_dimension == 3
    assert manifest.include_answer is False
    assert manifest.counts_by_category == {"biology": 2, "chemistry": 1}
    assert embedder.calls

    store = load_embedding_store(output_dir, mmap_mode=None)

    assert store.manifest.model_name == "fake-model"
    assert [question.question_id for question in store.questions] == ["q1", "q2", "q3"]
    np.testing.assert_allclose(store.embeddings, embeddings)
    assert {key: value.tolist() for key, value in store.category_indices.items()} == {
        "biology": [0, 2],
        "chemistry": [1],
    }
    assert store.question_index == {"q1": 0, "q2": 1, "q3": 2}

    shutil.rmtree(tmp_path)
