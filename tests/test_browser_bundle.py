import json
import shutil
from pathlib import Path

import numpy as np

from scibowl.dedupe.browser_bundle import (
    build_browser_corpus_bundle,
    build_browser_upload_bundle,
    build_browser_upload_bundle_from_mit_csv,
)
from scibowl.dedupe.embedding_store import build_embedding_store, load_embedding_store
from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.question import NormalizedQuestion
from scibowl.utils.ids import make_id


def _make_temp_dir() -> Path:
    path = Path("tests_runtime") / make_id("browser_bundle")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _question(question_id: str, *, category: Category, answer_mode: AnswerMode = AnswerMode.SHORT_ANSWER) -> NormalizedQuestion:
    return NormalizedQuestion(
        question_id=question_id,
        source_type=SourceType.DATASET,
        source_id="corpus",
        category=category,
        subcategory="core",
        question_type=QuestionType.TOSSUP,
        answer_mode=answer_mode,
        difficulty=4,
        question_text=f"Question {question_id}",
        answer_text=f"ANSWER: Answer {question_id}",
    )


class FakeEmbedder:
    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = embeddings

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._embeddings


def test_build_browser_corpus_bundle_writes_category_shards() -> None:
    tmp_path = _make_temp_dir()
    store_dir = tmp_path / "store"
    bundle_dir = tmp_path / "bundle"
    questions = [
        _question("q1", category=Category.BIOLOGY),
        _question("q2", category=Category.CHEMISTRY),
        _question("q3", category=Category.BIOLOGY),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    build_embedding_store(
        questions,
        output_dir=store_dir,
        model_name="fake-model",
        embedder=FakeEmbedder(embeddings),
    )
    corpus_store = load_embedding_store(store_dir, mmap_mode=None)

    manifest = build_browser_corpus_bundle(corpus_store, output_dir=bundle_dir)

    assert manifest.model_name == "fake-model"
    assert [entry.category for entry in manifest.categories] == ["biology", "chemistry"]
    biology_questions = json.loads((bundle_dir / "biology.questions.json").read_text(encoding="utf-8"))
    biology_embeddings = np.fromfile(bundle_dir / "biology.embeddings.f32", dtype=np.float32).reshape(2, 2)
    assert [question["question_id"] for question in biology_questions] == ["q1", "q3"]
    np.testing.assert_allclose(biology_embeddings, np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32))

    shutil.rmtree(tmp_path)


def test_build_browser_upload_bundle_writes_single_json_payload() -> None:
    tmp_path = _make_temp_dir()
    output_path = tmp_path / "upload_bundle.json"
    questions = [
        _question("q1", category=Category.BIOLOGY),
        _question("q2", category=Category.CHEMISTRY),
    ]
    payload = build_browser_upload_bundle(
        questions,
        output_path=output_path,
        model_name="fake-model",
        include_answer=False,
        embedder=FakeEmbedder(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)),
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "fake-model"
    assert payload["include_answer"] is False
    assert written["question_count"] == 2
    assert written["embedding_dimension"] == 2
    assert written["questions"][0]["question_id"] == "q1"
    assert written["embeddings"][1] == [0.0, 1.0]

    shutil.rmtree(tmp_path)


def test_build_browser_upload_bundle_from_mit_csv_uses_parser() -> None:
    tmp_path = _make_temp_dir()
    input_path = tmp_path / "upload.csv"
    output_path = tmp_path / "upload_bundle.json"
    input_path.write_text(
        "Type,Category,Format,Question,W,X,Y,Z,Answer,Accept,Do Not Accept,Subcategory\n"
        "Toss-up,Biology,Short Answer,What organelle carries out photosynthesis?,,,,,Chloroplast,,,Cell Biology\n",
        encoding="utf-8",
    )

    payload = build_browser_upload_bundle_from_mit_csv(
        input_path,
        output_path=output_path,
        source_id="upload_batch",
        tournament="Upload Batch",
        year=2025,
        embedder=FakeEmbedder(np.asarray([[1.0, 0.0]], dtype=np.float32)),
    )

    assert payload["question_count"] == 1
    assert payload["questions"][0]["source_id"] == "upload_batch"
    assert payload["questions"][0]["source_metadata"]["tournament"] == "Upload Batch"

    shutil.rmtree(tmp_path)
