from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from scibowl.schema.duplicate import DuplicateCandidate, DuplicateLabel, DuplicateReviewStatus
from scibowl.schema.question import NormalizedQuestion
from scibowl.utils.io import ensure_parent, read_jsonl


def load_question_lookup(path: Path) -> dict[str, NormalizedQuestion]:
    return {question.question_id: question for question in read_jsonl(path, NormalizedQuestion)}


class DuplicateReviewStore:
    def __init__(self, candidates_path: Path, questions_path: Path, output_path: Path) -> None:
        self.candidates_path = candidates_path
        self.questions_path = questions_path
        self.output_path = output_path
        self._lock = Lock()

        self.questions = load_question_lookup(questions_path)
        loaded_candidates = read_jsonl(candidates_path, DuplicateCandidate)
        self._pair_order = [candidate.pair_id for candidate in loaded_candidates]
        self._candidates = {candidate.pair_id: candidate for candidate in loaded_candidates}

        if output_path.exists():
            for reviewed in read_jsonl(output_path, DuplicateCandidate):
                if reviewed.pair_id in self._candidates:
                    self._candidates[reviewed.pair_id] = reviewed

    def summary(self) -> dict[str, object]:
        candidates = [self._candidates[pair_id] for pair_id in self._pair_order]
        reviewed = [candidate for candidate in candidates if candidate.review_status == DuplicateReviewStatus.REVIEWED]
        counts_by_label: dict[str, int] = {}
        for candidate in reviewed:
            if candidate.label is None:
                continue
            counts_by_label[candidate.label.value] = counts_by_label.get(candidate.label.value, 0) + 1
        return {
            "total": len(candidates),
            "reviewed": len(reviewed),
            "remaining": len(candidates) - len(reviewed),
            "counts_by_label": counts_by_label,
        }

    def session_items(self, *, filter_name: str = "unreviewed", min_similarity: float = 0.0) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for pair_id in self._pair_order:
            candidate = self._candidates[pair_id]
            if filter_name == "unreviewed" and candidate.review_status == DuplicateReviewStatus.REVIEWED:
                continue
            if filter_name == "reviewed" and candidate.review_status != DuplicateReviewStatus.REVIEWED:
                continue
            if candidate.embedding_similarity < min_similarity:
                continue
            items.append(
                {
                    "pair_id": candidate.pair_id,
                    "category": candidate.category,
                    "embedding_similarity": candidate.embedding_similarity,
                    "label": candidate.label.value if candidate.label else None,
                    "review_status": candidate.review_status.value,
                    "round_a": build_round_label(self.questions.get(candidate.question_id_a)),
                    "round_b": build_round_label(self.questions.get(candidate.question_id_b)),
                }
            )
        return items

    def candidate_payload(self, pair_id: str) -> dict[str, object]:
        candidate = self._candidates[pair_id]
        question_a = self.questions.get(candidate.question_id_a)
        question_b = self.questions.get(candidate.question_id_b)
        payload = candidate.model_dump(mode="json")
        payload["origin_a"] = build_question_origin(question_a)
        payload["origin_b"] = build_question_origin(question_b)
        payload["choices_a"] = [choice.model_dump(mode="json") for choice in (question_a.choices if question_a else [])]
        payload["choices_b"] = [choice.model_dump(mode="json") for choice in (question_b.choices if question_b else [])]
        return payload

    def save_review(self, *, pair_id: str, label: str | None, notes: str | None) -> dict[str, object]:
        with self._lock:
            candidate = self._candidates[pair_id]
            clean_notes = (notes or "").strip() or None
            if label:
                candidate.label = DuplicateLabel(label)
                candidate.review_status = DuplicateReviewStatus.REVIEWED
            else:
                candidate.label = None
                candidate.review_status = DuplicateReviewStatus.CANDIDATE
            candidate.notes = clean_notes
            self._write_output()
            return self.candidate_payload(pair_id)

    def _write_output(self) -> None:
        ensure_parent(self.output_path)
        tmp_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for pair_id in self._pair_order:
                candidate = self._candidates[pair_id]
                if candidate.review_status != DuplicateReviewStatus.REVIEWED:
                    continue
                handle.write(candidate.model_dump_json() + "\n")
        tmp_path.replace(self.output_path)


def build_question_origin(question: NormalizedQuestion | None) -> dict[str, str | None]:
    if question is None:
        return {"tournament": None, "round": None, "question_id": None}
    return {
        "tournament": question.source_metadata.tournament or question.source_id,
        "round": build_round_label(question),
        "question_id": question.question_id,
    }


def build_round_label(question: NormalizedQuestion | None) -> str | None:
    if question is None:
        return None
    if question.source_metadata.round is not None:
        return f"Round {question.source_metadata.round}"
    raw_file = question.provenance.raw_file
    if raw_file:
        stem = Path(raw_file).stem
        if "__" in stem:
            stem = "__".join(stem.split("__")[1:])
        return stem.replace("_", " ").upper()
    return None


def dump_json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")
