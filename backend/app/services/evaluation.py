"""Repeatable, honest evaluation of retrieval and grounded answer behavior."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from app.models.citation import SourceCitation
from app.models.evaluation import (
    AnswerJudgment,
    EvaluationCase,
    GenerationCaseResult,
    RetrievalMethodResult,
    RetrievalMetrics,
    serializable,
)


class RetrievedCandidate(Protocol):
    chunk: object


class AnswerJudge(Protocol):
    """Optional judge abstraction; callers decide whether to use an LLM judge."""

    def judge(self, case: EvaluationCase, answer: str, sources: Sequence[SourceCitation]) -> AnswerJudgment:
        """Score relevance and groundedness on a documented scale."""


def load_dataset(path: Path) -> list[EvaluationCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvaluationCase(
            case_id=item["case_id"],
            question=item["question"],
            expected_file_paths=item["expected_file_paths"],
            expected_symbols=item.get("expected_symbols", []),
        )
        for item in raw_cases
    ]


def evaluate_retrieval(
    cases: Sequence[EvaluationCase],
    retrieve: Callable[[EvaluationCase], Sequence[RetrievedCandidate]],
    method: str,
) -> RetrievalMethodResult:
    """Evaluate a retrieval callable against curated file/symbol expectations."""
    case_results: list[dict[str, object]] = []
    ranks: list[int | None] = []
    precisions: dict[int, list[float]] = {1: [], 3: [], 5: [], 10: []}
    for case in cases:
        candidates = list(retrieve(case))
        relevant = [_is_relevant(candidate.chunk, case) for candidate in candidates]
        first_rank = next((index for index, matched in enumerate(relevant, start=1) if matched), None)
        ranks.append(first_rank)
        for k in precisions:
            inspected = relevant[:k]
            precisions[k].append(sum(inspected) / k)
        case_results.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "first_relevant_rank": first_rank,
                "returned_chunk_ids": [candidate.chunk.chunk_id for candidate in candidates],
                "returned_file_paths": [candidate.chunk.file_path for candidate in candidates],
            }
        )
    count = len(cases)
    metrics = RetrievalMetrics(
        case_count=count,
        recall_at_1=_recall_at(ranks, 1),
        recall_at_3=_recall_at(ranks, 3),
        recall_at_5=_recall_at(ranks, 5),
        recall_at_10=_recall_at(ranks, 10),
        mrr=sum(1 / rank for rank in ranks if rank is not None) / count if count else 0.0,
        precision_at_1=_mean(precisions[1]),
        precision_at_3=_mean(precisions[3]),
        precision_at_5=_mean(precisions[5]),
        precision_at_10=_mean(precisions[10]),
    )
    return RetrievalMethodResult(method=method, metrics=metrics, case_results=case_results)


def evaluate_generation_case(
    case: EvaluationCase,
    answer: str,
    sources: Sequence[SourceCitation],
    judge: AnswerJudge | None = None,
) -> GenerationCaseResult:
    """Measure citation correctness deterministically; optionally add judged quality scores."""
    source_matches = [_source_matches(source, case) for source in sources]
    citation_precision = sum(source_matches) / len(sources) if sources else None
    expected_total = len(case.expected_file_paths)
    matched_paths = {source.file_path for source, matched in zip(sources, source_matches, strict=True) if matched}
    citation_recall = len(matched_paths) / expected_total if expected_total else None
    judgment = judge.judge(case, answer, sources) if judge else AnswerJudgment(None, None)
    return GenerationCaseResult(
        case_id=case.case_id,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        answer_relevance=judgment.relevance,
        groundedness=judgment.groundedness,
    )


def write_retrieval_report(results: Sequence[RetrievalMethodResult], output_directory: Path) -> tuple[Path, Path]:
    """Write executed metrics as JSON and Markdown; never create synthetic scores."""
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "retrieval-results.json"
    markdown_path = output_directory / "retrieval-report.md"
    json_path.write_text(json.dumps(serializable(list(results)), indent=2), encoding="utf-8")
    rows = ["| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Precision@5 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        metric = result.metrics
        rows.append(
            f"| {result.method} | {metric.recall_at_1:.3f} | {metric.recall_at_3:.3f} | "
            f"{metric.recall_at_5:.3f} | {metric.recall_at_10:.3f} | {metric.mrr:.3f} | {metric.precision_at_5:.3f} |"
        )
    markdown_path.write_text("# Retrieval evaluation report\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return json_path, markdown_path


def write_generation_report(results: Sequence[GenerationCaseResult], output_directory: Path) -> tuple[Path, Path]:
    """Persist executed answer/citation evaluation separately from retrieval metrics."""
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "generation-results.json"
    markdown_path = output_directory / "generation-report.md"
    json_path.write_text(json.dumps(serializable(list(results)), indent=2), encoding="utf-8")
    rows = ["| Case | Citation precision | Citation recall | Answer relevance | Groundedness |", "|---|---:|---:|---:|---:|"]
    for result in results:
        rows.append(
            f"| {result.case_id} | {_format_optional(result.citation_precision)} | "
            f"{_format_optional(result.citation_recall)} | {_format_optional(result.answer_relevance)} | "
            f"{_format_optional(result.groundedness)} |"
        )
    markdown_path.write_text("# Generation evaluation report\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return json_path, markdown_path


def _is_relevant(chunk: object, case: EvaluationCase) -> bool:
    path_matches = chunk.file_path in case.expected_file_paths
    symbol_matches = not case.expected_symbols or chunk.symbol_name in case.expected_symbols
    return path_matches and symbol_matches


def _source_matches(source: SourceCitation, case: EvaluationCase) -> bool:
    return source.file_path in case.expected_file_paths and (
        not case.expected_symbols or source.symbol_name in case.expected_symbols
    )


def _recall_at(ranks: Sequence[int | None], k: int) -> float:
    return sum(rank is not None and rank <= k for rank in ranks) / len(ranks) if ranks else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _format_optional(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A"
