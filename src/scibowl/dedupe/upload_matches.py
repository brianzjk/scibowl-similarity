from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scibowl.dedupe.candidates import (
    EmbeddingConfig,
    SentenceTransformerEmbedder,
    build_candidate,
    build_duplicate_candidates,
    build_embedding_text,
)
from scibowl.dedupe.embedding_store import LoadedEmbeddingStore
from scibowl.schema.duplicate import DuplicateCandidate
from scibowl.schema.question import NormalizedQuestion
from scibowl.utils.ids import slugify
from scibowl.utils.io import write_json, write_jsonl


def build_corpus_match_candidates(
    upload_questions: list[NormalizedQuestion],
    upload_embeddings: np.ndarray,
    corpus_store: LoadedEmbeddingStore,
    *,
    threshold: float = 0.5,
    top_k: int = 10,
) -> list[DuplicateCandidate]:
    if len(upload_questions) != len(upload_embeddings):
        raise ValueError("upload_questions and upload_embeddings must have the same length")
    if upload_embeddings.ndim != 2:
        raise ValueError("upload_embeddings must be a 2D array")
    if corpus_store.embeddings.ndim != 2:
        raise ValueError("corpus embeddings must be a 2D array")
    if upload_embeddings.shape[1] != corpus_store.embeddings.shape[1]:
        raise ValueError("upload and corpus embeddings must have the same dimension")

    candidates: list[DuplicateCandidate] = []
    for upload_index, question in enumerate(upload_questions):
        corpus_indices = corpus_store.category_indices.get(question.category.value)
        if corpus_indices is None or len(corpus_indices) == 0:
            continue

        block_embeddings = corpus_store.embeddings[corpus_indices]
        scores = np.asarray(block_embeddings @ upload_embeddings[upload_index], dtype=np.float32)
        if scores.ndim != 1:
            raise ValueError("corpus similarity scores must be a 1D array")

        top_indices = _top_indices(scores, top_k=top_k)
        for local_index in top_indices:
            score = float(scores[local_index])
            if score < threshold:
                continue

            corpus_question = corpus_store.questions[int(corpus_indices[local_index])]
            if question.question_id == corpus_question.question_id and question.source_id == corpus_question.source_id:
                continue
            candidates.append(build_candidate(question, corpus_question, score, preserve_order=True))

    return sorted(candidates, key=lambda candidate: candidate.embedding_similarity, reverse=True)


def match_uploaded_questions(
    upload_questions: list[NormalizedQuestion],
    *,
    corpus_store: LoadedEmbeddingStore,
    threshold: float = 0.5,
    top_k: int = 10,
    batch_size: int = 32,
    device: str | None = None,
    cache_folder: str | None = None,
) -> tuple[np.ndarray, list[DuplicateCandidate], list[DuplicateCandidate]]:
    embedder = SentenceTransformerEmbedder(
        EmbeddingConfig(
            model_name=corpus_store.manifest.model_name,
            batch_size=batch_size,
            device=device,
            cache_folder=cache_folder,
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        )
    )
    texts = [
        build_embedding_text(question, include_answer=corpus_store.manifest.include_answer)
        for question in upload_questions
    ]
    upload_embeddings = np.asarray(embedder.encode(texts), dtype=np.float32)
    within_upload = build_duplicate_candidates(
        upload_questions,
        upload_embeddings,
        threshold=threshold,
        top_k=top_k,
    )
    against_corpus = build_corpus_match_candidates(
        upload_questions,
        upload_embeddings,
        corpus_store,
        threshold=threshold,
        top_k=top_k,
    )
    return upload_embeddings, within_upload, against_corpus


def write_upload_match_artifacts(
    *,
    output_dir: Path,
    upload_questions: list[NormalizedQuestion],
    within_upload_matches: list[DuplicateCandidate],
    corpus_matches: list[DuplicateCandidate],
    corpus_store: LoadedEmbeddingStore,
    input_csv_path: Path,
    threshold: float,
    top_k: int,
) -> dict[str, str | int | float | bool | dict[str, int]]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    uploaded_questions_path = resolved_output_dir / "uploaded_questions.jsonl"
    within_upload_path = resolved_output_dir / "matches_within_upload.jsonl"
    corpus_matches_path = resolved_output_dir / "matches_against_corpus.jsonl"
    corpus_review_questions_path = resolved_output_dir / "corpus_match_questions.jsonl"
    summary_path = resolved_output_dir / "summary.json"

    write_jsonl(uploaded_questions_path, upload_questions)
    write_jsonl(within_upload_path, within_upload_matches)
    write_jsonl(corpus_matches_path, corpus_matches)
    write_jsonl(corpus_review_questions_path, _build_corpus_review_questions(upload_questions, corpus_matches, corpus_store))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_csv_path": str(Path(input_csv_path).resolve()),
        "embedding_store_dir": str(corpus_store.root_dir.resolve()),
        "model_name": corpus_store.manifest.model_name,
        "include_answer": corpus_store.manifest.include_answer,
        "uploaded_question_count": len(upload_questions),
        "within_upload_match_count": len(within_upload_matches),
        "corpus_match_count": len(corpus_matches),
        "threshold": threshold,
        "top_k": top_k,
        "counts_by_uploaded_category": _count_by_category(upload_questions),
        "counts_by_corpus_match_category": _count_by_candidate_category(corpus_matches),
        "uploaded_questions_path": str(uploaded_questions_path.resolve()),
        "within_upload_matches_path": str(within_upload_path.resolve()),
        "corpus_matches_path": str(corpus_matches_path.resolve()),
        "corpus_match_questions_path": str(corpus_review_questions_path.resolve()),
    }
    write_json(summary_path, summary)
    return summary


def default_upload_match_dir(input_csv_path: Path) -> Path:
    stem = slugify(Path(input_csv_path).stem)
    return Path("..") / "data" / "interim" / "uploads" / stem


def _build_corpus_review_questions(
    upload_questions: list[NormalizedQuestion],
    corpus_matches: list[DuplicateCandidate],
    corpus_store: LoadedEmbeddingStore,
) -> list[NormalizedQuestion]:
    questions_by_id = {question.question_id: question for question in upload_questions}
    for candidate in corpus_matches:
        corpus_question = corpus_store.questions[corpus_store.question_index[candidate.question_id_b]]
        questions_by_id.setdefault(corpus_question.question_id, corpus_question)
    return list(questions_by_id.values())


def _count_by_category(questions: list[NormalizedQuestion]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in questions:
        counts[question.category.value] = counts.get(question.category.value, 0) + 1
    return dict(sorted(counts.items()))


def _count_by_candidate_category(candidates: list[DuplicateCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.category] = counts.get(candidate.category, 0) + 1
    return dict(sorted(counts.items()))


def _top_indices(scores: np.ndarray, *, top_k: int) -> np.ndarray:
    if top_k <= 0:
        return np.asarray([], dtype=np.int32)
    if top_k >= len(scores):
        return np.argsort(scores)[::-1]
    candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
    return candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
