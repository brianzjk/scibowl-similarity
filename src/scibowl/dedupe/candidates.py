from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from scibowl.schema.duplicate import DuplicateCandidate, DuplicateMiningSummary
from scibowl.schema.question import NormalizedQuestion
from scibowl.utils.ids import slugify


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class EmbeddingConfig:
    model_name: str = "mixedbread-ai/mxbai-embed-large-v1"
    batch_size: int = 32
    device: str | None = None
    cache_folder: str | None = None
    token: str | None = None


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            self._model = self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for duplicate mining. "
                "Install the optional embedding dependencies first."
            ) from exc

        return SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
            cache_folder=self.config.cache_folder,
            token=self.config.token,
            trust_remote_code=False,
        )


def mine_duplicate_candidates(
    questions: list[NormalizedQuestion],
    *,
    model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
    threshold: float = 0.5,
    top_k: int = 10,
    include_answer: bool = True,
    batch_size: int = 32,
    device: str | None = None,
    cache_folder: str | None = None,
    token: str | None = None,
) -> tuple[list[DuplicateCandidate], DuplicateMiningSummary]:
    texts = [build_embedding_text(question, include_answer=include_answer) for question in questions]
    embedder = SentenceTransformerEmbedder(
        EmbeddingConfig(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            cache_folder=cache_folder,
            token=token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        )
    )
    embeddings = embedder.encode(texts)
    candidates = build_duplicate_candidates(
        questions,
        embeddings,
        threshold=threshold,
        top_k=top_k,
    )
    counts_by_category: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        counts_by_category[candidate.category] += 1
    summary = DuplicateMiningSummary(
        run_id=f"dupes_{slugify(model_name)}_{len(questions)}",
        model_name=model_name,
        question_count=len(questions),
        candidate_count=len(candidates),
        threshold=threshold,
        top_k=top_k,
        include_answer=include_answer,
        counts_by_category=dict(sorted(counts_by_category.items())),
    )
    return candidates, summary


def build_duplicate_candidates(
    questions: list[NormalizedQuestion],
    embeddings: np.ndarray,
    *,
    threshold: float = 0.5,
    top_k: int = 10,
) -> list[DuplicateCandidate]:
    if len(questions) != len(embeddings):
        raise ValueError("questions and embeddings must have the same length")

    pair_candidates: dict[tuple[str, str], DuplicateCandidate] = {}
    block_map: dict[str, list[int]] = defaultdict(list)
    for index, question in enumerate(questions):
        block_map[question.category.value].append(index)

    for block_indices in block_map.values():
        if len(block_indices) < 2:
            continue

        block_embeddings = embeddings[block_indices]
        similarity_matrix = np.asarray(block_embeddings @ block_embeddings.T, dtype=np.float32)
        np.fill_diagonal(similarity_matrix, -np.inf)

        for local_index, question_index in enumerate(block_indices):
            row = similarity_matrix[local_index]
            if top_k >= len(row):
                top_indices = np.argsort(row)[::-1]
            else:
                candidate_indices = np.argpartition(row, -top_k)[-top_k:]
                top_indices = candidate_indices[np.argsort(row[candidate_indices])[::-1]]

            for other_local_index in top_indices:
                score = float(row[other_local_index])
                if score < threshold:
                    continue

                other_question_index = block_indices[other_local_index]
                question_a = questions[question_index]
                question_b = questions[other_question_index]
                pair_key = tuple(sorted((question_a.question_id, question_b.question_id)))
                if question_a.question_id == question_b.question_id or pair_key in pair_candidates:
                    continue

                pair_candidates[pair_key] = build_candidate(question_a, question_b, score)

    return sorted(pair_candidates.values(), key=lambda candidate: candidate.embedding_similarity, reverse=True)


def build_embedding_text(question: NormalizedQuestion, *, include_answer: bool) -> str:
    segments = [question.category.value, question.subcategory, question.question_text]
    if include_answer:
        segments.append(f"answer {normalize_answer_text(question.answer_text)}")
    return "\n".join(segment for segment in segments if segment)


def normalize_answer_text(answer_text: str) -> str:
    normalized = answer_text.removeprefix("ANSWER:").strip()
    normalized = re.sub(r"^[WXYZ]\s*[\)\.]\s*", "", normalized, flags=re.IGNORECASE)
    return normalized.casefold()


def lexical_overlap_score(text_a: str, text_b: str) -> float:
    tokens_a = set(TOKEN_PATTERN.findall(text_a.casefold()))
    tokens_b = set(TOKEN_PATTERN.findall(text_b.casefold()))
    if not tokens_a or not tokens_b:
        return 0.0
    return round(len(tokens_a & tokens_b) / len(tokens_a | tokens_b), 3)


def build_candidate(
    question_a: NormalizedQuestion,
    question_b: NormalizedQuestion,
    score: float,
    *,
    preserve_order: bool = False,
) -> DuplicateCandidate:
    if preserve_order:
        first, second = question_a, question_b
    else:
        first, second = sorted((question_a, question_b), key=lambda question: question.question_id)
    return DuplicateCandidate(
        pair_id=f"dup__{first.question_id}__{second.question_id}",
        question_id_a=first.question_id,
        question_id_b=second.question_id,
        source_id_a=first.source_id,
        source_id_b=second.source_id,
        category=first.category.value,
        subcategory_a=first.subcategory,
        subcategory_b=second.subcategory,
        question_type_a=first.question_type.value,
        question_type_b=second.question_type.value,
        answer_mode_a=first.answer_mode.value,
        answer_mode_b=second.answer_mode.value,
        question_text_a=first.question_text,
        question_text_b=second.question_text,
        answer_text_a=first.answer_text,
        answer_text_b=second.answer_text,
        embedding_similarity=round(score, 4),
        lexical_overlap=lexical_overlap_score(first.question_text, second.question_text),
        same_subcategory=first.subcategory.casefold() == second.subcategory.casefold(),
        same_question_type=first.question_type == second.question_type,
        same_answer_mode=first.answer_mode == second.answer_mode,
        same_normalized_answer=normalize_answer_text(first.answer_text) == normalize_answer_text(second.answer_text),
    )
