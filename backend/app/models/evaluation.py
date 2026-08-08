from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_file_paths: list[str]
    expected_symbols: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    case_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    precision_at_10: float


@dataclass(frozen=True, slots=True)
class RetrievalMethodResult:
    method: str
    metrics: RetrievalMetrics
    case_results: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class AnswerJudgment:
    relevance: float | None
    groundedness: float | None
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class GenerationCaseResult:
    case_id: str
    citation_precision: float | None
    citation_recall: float | None
    answer_relevance: float | None
    groundedness: float | None


def serializable(value: object) -> object:
    """Turn the evaluation dataclasses into JSON-ready nested structures."""
    if hasattr(value, "__dataclass_fields__"):
        return {key: serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    return value
