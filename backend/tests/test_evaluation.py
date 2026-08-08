from dataclasses import dataclass
from pathlib import Path

from app.models.citation import SourceCitation
from app.models.chunk import CodeChunk
from app.models.evaluation import AnswerJudgment, EvaluationCase
from app.services.evaluation import evaluate_generation_case, evaluate_retrieval, write_retrieval_report


@dataclass
class Candidate:
    chunk: CodeChunk


def candidate(path: str, symbol: str | None = None) -> Candidate:
    return Candidate(CodeChunk("id-" + path, "repo", path, "Python", "pass", 1, 1, symbol, "fallback"))


def test_retrieval_metrics_match_known_ranks() -> None:
    cases = [
        EvaluationCase("one", "first", ["src/a.py"]),
        EvaluationCase("two", "second", ["src/b.py"]),
    ]

    result = evaluate_retrieval(
        cases,
        lambda case: [candidate("src/a.py")] if case.case_id == "one" else [candidate("src/x.py"), candidate("src/b.py")],
        "test",
    )

    assert result.metrics.recall_at_1 == 0.5
    assert result.metrics.recall_at_3 == 1.0
    assert result.metrics.recall_at_5 == 1.0
    assert result.metrics.mrr == 0.75
    assert result.metrics.precision_at_1 == 0.5


def test_generation_evaluation_scores_citation_correctness_without_llm_judge() -> None:
    case = EvaluationCase("one", "auth", ["src/auth.py"], ["authenticate"])
    source = SourceCitation("1", "src/auth.py", 10, 20, "authenticate", "chunk-1")

    result = evaluate_generation_case(case, "Answer [1]", [source])

    assert result.citation_precision == 1.0
    assert result.citation_recall == 1.0
    assert result.answer_relevance is None
    assert result.groundedness is None


def test_generation_evaluation_uses_optional_judge() -> None:
    class FakeJudge:
        def judge(self, case, answer, sources):
            return AnswerJudgment(relevance=4.0, groundedness=5.0, rationale="grounded")

    result = evaluate_generation_case(EvaluationCase("one", "auth", ["src/auth.py"]), "Answer", [], FakeJudge())

    assert result.answer_relevance == 4.0
    assert result.groundedness == 5.0


def test_retrieval_report_writes_only_calculated_metrics(tmp_path: Path) -> None:
    result = evaluate_retrieval(
        [EvaluationCase("one", "question", ["src/a.py"])],
        lambda _: [candidate("src/a.py")],
        "Vector",
    )

    json_path, markdown_path = write_retrieval_report([result], tmp_path)

    assert json_path.exists()
    assert "| Vector | 1.000" in markdown_path.read_text(encoding="utf-8")
