from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from scibowl.dedupe.candidates import EmbeddingConfig, SentenceTransformerEmbedder, build_embedding_text
from scibowl.dedupe.embedding_store import LoadedEmbeddingStore
from scibowl.ingest import parse_mit_questions_csv
from scibowl.schema.question import NormalizedQuestion
from scibowl.utils.io import ensure_parent


BROWSER_CORPUS_BUNDLE_VERSION = "browser_corpus_bundle_v1"
BROWSER_UPLOAD_BUNDLE_VERSION = "browser_upload_bundle_v1"


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class BrowserCorpusCategoryManifest(BaseModel):
    category: str
    count: int
    questions_path: str
    embeddings_path: str


class BrowserCorpusBundleManifest(BaseModel):
    version: str = BROWSER_CORPUS_BUNDLE_VERSION
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_name: str
    include_answer: bool
    embedding_dimension: int
    question_count: int
    categories: list[BrowserCorpusCategoryManifest] = Field(default_factory=list)


def build_browser_corpus_bundle(
    corpus_store: LoadedEmbeddingStore,
    *,
    output_dir: Path,
) -> BrowserCorpusBundleManifest:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    manifest = BrowserCorpusBundleManifest(
        model_name=corpus_store.manifest.model_name,
        include_answer=corpus_store.manifest.include_answer,
        embedding_dimension=corpus_store.manifest.embedding_dimension,
        question_count=corpus_store.manifest.question_count,
    )

    for category, indices in sorted(corpus_store.category_indices.items()):
        questions_filename = f"{category}.questions.json"
        embeddings_filename = f"{category}.embeddings.f32"

        category_questions = [corpus_store.questions[int(index)].model_dump(mode="json") for index in indices]
        with (resolved_output_dir / questions_filename).open("w", encoding="utf-8") as handle:
            json.dump(category_questions, handle, ensure_ascii=False, separators=(",", ":"))

        category_embeddings = np.asarray(corpus_store.embeddings[indices], dtype=np.float32)
        with (resolved_output_dir / embeddings_filename).open("wb") as handle:
            handle.write(category_embeddings.tobytes(order="C"))

        manifest.categories.append(
            BrowserCorpusCategoryManifest(
                category=category,
                count=len(indices),
                questions_path=questions_filename,
                embeddings_path=embeddings_filename,
            )
        )

    with (resolved_output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
    return manifest


def build_browser_upload_bundle(
    questions: list[NormalizedQuestion],
    *,
    output_path: Path,
    model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
    include_answer: bool = True,
    batch_size: int = 32,
    device: str | None = None,
    cache_folder: str | None = None,
    embedder: Embedder | None = None,
    source_input_path: Path | None = None,
) -> dict[str, object]:
    texts = [build_embedding_text(question, include_answer=include_answer) for question in questions]
    bundle_embedder = embedder or SentenceTransformerEmbedder(
        EmbeddingConfig(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            cache_folder=cache_folder,
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        )
    )
    embeddings = np.asarray(bundle_embedder.encode(texts), dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if len(questions) != len(embeddings):
        raise ValueError("questions and embeddings must have the same length")

    payload = {
        "version": BROWSER_UPLOAD_BUNDLE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "include_answer": include_answer,
        "embedding_dimension": int(embeddings.shape[1]) if embeddings.size else 0,
        "question_count": len(questions),
        "source_input_path": str(Path(source_input_path).resolve()) if source_input_path else None,
        "questions": [question.model_dump(mode="json") for question in questions],
        "embeddings": embeddings.tolist(),
    }

    ensure_parent(Path(output_path))
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    return payload


def build_browser_upload_bundle_from_mit_csv(
    input_path: Path,
    *,
    output_path: Path,
    source_id: str | None = None,
    tournament: str | None = None,
    year: int | None = None,
    default_difficulty: int = 4,
    model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
    include_answer: bool = True,
    batch_size: int = 32,
    device: str | None = None,
    cache_folder: str | None = None,
    embedder: Embedder | None = None,
) -> dict[str, object]:
    questions = parse_mit_questions_csv(
        Path(input_path),
        source_id=source_id,
        tournament=tournament,
        year=year,
        default_difficulty=default_difficulty,
    )
    return build_browser_upload_bundle(
        questions,
        output_path=output_path,
        model_name=model_name,
        include_answer=include_answer,
        batch_size=batch_size,
        device=device,
        cache_folder=cache_folder,
        embedder=embedder,
        source_input_path=Path(input_path),
    )
