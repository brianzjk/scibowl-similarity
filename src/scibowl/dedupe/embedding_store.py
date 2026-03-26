from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from scibowl.dedupe.candidates import EmbeddingConfig, SentenceTransformerEmbedder, build_embedding_text
from scibowl.schema.question import NormalizedQuestion
from scibowl.utils.ids import slugify
from scibowl.utils.io import read_json, read_jsonl, write_json, write_jsonl


MANIFEST_FILENAME = "manifest.json"
QUESTIONS_FILENAME = "questions.jsonl"
EMBEDDINGS_FILENAME = "embeddings.npy"
CATEGORY_INDICES_FILENAME = "category_indices.json"


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class EmbeddingStoreManifest(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str
    model_name: str
    question_count: int
    embedding_dimension: int
    embedding_dtype: str = "float32"
    include_answer: bool
    source_questions_path: str | None = None
    questions_filename: str = QUESTIONS_FILENAME
    embeddings_filename: str = EMBEDDINGS_FILENAME
    category_indices_filename: str = CATEGORY_INDICES_FILENAME
    counts_by_category: dict[str, int] = Field(default_factory=dict)


@dataclass
class LoadedEmbeddingStore:
    root_dir: Path
    manifest: EmbeddingStoreManifest
    questions: list[NormalizedQuestion]
    embeddings: np.ndarray
    category_indices: dict[str, np.ndarray]
    question_index: dict[str, int]


def filter_questions(
    questions: list[NormalizedQuestion],
    *,
    exclude_source_ids: list[str] | None = None,
    exclude_tournaments: list[str] | None = None,
) -> list[NormalizedQuestion]:
    normalized_source_ids = _normalize_exclusions(exclude_source_ids)
    normalized_tournaments = _normalize_exclusions(exclude_tournaments)
    if not normalized_source_ids and not normalized_tournaments:
        return list(questions)
    return [
        question
        for question in questions
        if not _question_is_excluded(
            question,
            normalized_source_ids=normalized_source_ids,
            normalized_tournaments=normalized_tournaments,
        )
    ]


def build_embedding_store(
    questions: list[NormalizedQuestion],
    *,
    output_dir: Path,
    source_questions_path: Path | None = None,
    model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
    include_answer: bool = True,
    batch_size: int = 32,
    device: str | None = None,
    cache_folder: str | None = None,
    token: str | None = None,
    embedder: Embedder | None = None,
) -> EmbeddingStoreManifest:
    texts = [build_embedding_text(question, include_answer=include_answer) for question in questions]
    store_embedder = embedder or SentenceTransformerEmbedder(
        EmbeddingConfig(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            cache_folder=cache_folder,
            token=token,
        )
    )
    embeddings = np.asarray(store_embedder.encode(texts), dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if len(questions) != len(embeddings):
        raise ValueError("questions and embeddings must have the same length")

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(resolved_output_dir / QUESTIONS_FILENAME, questions)
    np.save(resolved_output_dir / EMBEDDINGS_FILENAME, embeddings, allow_pickle=False)

    category_indices = _build_category_indices(questions)
    write_json(
        resolved_output_dir / CATEGORY_INDICES_FILENAME,
        {category: indices for category, indices in sorted(category_indices.items())},
    )

    manifest = EmbeddingStoreManifest(
        run_id=f"embeddings_{slugify(model_name)}_{len(questions)}",
        model_name=model_name,
        question_count=len(questions),
        embedding_dimension=int(embeddings.shape[1]) if embeddings.size else 0,
        include_answer=include_answer,
        source_questions_path=str(Path(source_questions_path).resolve()) if source_questions_path else None,
        counts_by_category={category: len(indices) for category, indices in sorted(category_indices.items())},
    )
    write_json(resolved_output_dir / MANIFEST_FILENAME, manifest.model_dump(mode="json"))
    return manifest


def load_embedding_store(root_dir: Path, *, mmap_mode: str | None = "r") -> LoadedEmbeddingStore:
    resolved_root = Path(root_dir)
    manifest = EmbeddingStoreManifest.model_validate(read_json(resolved_root / MANIFEST_FILENAME))
    questions = read_jsonl(resolved_root / manifest.questions_filename, NormalizedQuestion)
    embeddings = np.load(resolved_root / manifest.embeddings_filename, mmap_mode=mmap_mode, allow_pickle=False)
    category_indices_payload = read_json(resolved_root / manifest.category_indices_filename)
    category_indices = {
        category: np.asarray(indices, dtype=np.int32)
        for category, indices in category_indices_payload.items()
    }

    if len(questions) != manifest.question_count:
        raise ValueError("question count does not match manifest")
    if embeddings.shape[0] != manifest.question_count:
        raise ValueError("embedding count does not match manifest")
    if embeddings.ndim != 2:
        raise ValueError("stored embeddings must be a 2D array")

    question_index = {question.question_id: index for index, question in enumerate(questions)}
    return LoadedEmbeddingStore(
        root_dir=resolved_root,
        manifest=manifest,
        questions=questions,
        embeddings=embeddings,
        category_indices=category_indices,
        question_index=question_index,
    )


def filter_loaded_embedding_store(
    store: LoadedEmbeddingStore,
    *,
    exclude_source_ids: list[str] | None = None,
    exclude_tournaments: list[str] | None = None,
) -> LoadedEmbeddingStore:
    normalized_source_ids = _normalize_exclusions(exclude_source_ids)
    normalized_tournaments = _normalize_exclusions(exclude_tournaments)
    if not normalized_source_ids and not normalized_tournaments:
        return store

    kept_indices = np.asarray(
        [
            index
            for index, question in enumerate(store.questions)
            if not _question_is_excluded(
                question,
                normalized_source_ids=normalized_source_ids,
                normalized_tournaments=normalized_tournaments,
            )
        ],
        dtype=np.int32,
    )
    filtered_questions = [store.questions[int(index)] for index in kept_indices]
    filtered_embeddings = np.asarray(store.embeddings[kept_indices], dtype=np.float32)
    filtered_category_indices = {
        category: np.asarray(indices, dtype=np.int32)
        for category, indices in _build_category_indices(filtered_questions).items()
    }
    filtered_question_index = {question.question_id: index for index, question in enumerate(filtered_questions)}
    filtered_manifest = store.manifest.model_copy(
        update={
            "question_count": len(filtered_questions),
            "counts_by_category": {category: len(indices) for category, indices in sorted(filtered_category_indices.items())},
        }
    )
    return LoadedEmbeddingStore(
        root_dir=store.root_dir,
        manifest=filtered_manifest,
        questions=filtered_questions,
        embeddings=filtered_embeddings,
        category_indices=filtered_category_indices,
        question_index=filtered_question_index,
    )


def default_embedding_store_dir(questions_path: Path, *, model_name: str) -> Path:
    slug = slugify(model_name)
    base_name = Path(questions_path).stem
    return Path("..") / "data" / "processed" / "embeddings" / f"{base_name}_{slug}"


def _build_category_indices(questions: list[NormalizedQuestion]) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = {}
    for index, question in enumerate(questions):
        indices.setdefault(question.category.value, []).append(index)
    return indices


def _normalize_exclusions(values: list[str] | None) -> set[str]:
    return {_normalize_text(value) for value in values or [] if _normalize_text(value)}


def _question_is_excluded(
    question: NormalizedQuestion,
    *,
    normalized_source_ids: set[str],
    normalized_tournaments: set[str],
) -> bool:
    if normalized_source_ids and _normalize_text(question.source_id) in normalized_source_ids:
        return True
    tournament = question.source_metadata.tournament if question.source_metadata else None
    if normalized_tournaments and _normalize_text(tournament) in normalized_tournaments:
        return True
    return False


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().casefold()
