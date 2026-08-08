"""CLI runner for an already-indexed evaluation repository."""

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.services.evaluation import evaluate_retrieval, load_dataset, write_retrieval_report
from app.services.hybrid_retrieval import create_hybrid_retrieval_service
from app.services.reranked_retrieval import create_reranked_retrieval_service
from app.services.retrieval import create_retrieval_service
from app.services.repository_ingestion import iter_repository_files
from app.services.chunking import CodeChunkingService
from app.services.embedding import create_embedding_service
from app.services.indexing import RepositoryIndexingService
from app.services.qdrant_store import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vector, hybrid, and reranked retrieval")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--repository-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation-results"))
    args = parser.parse_args()

    settings = get_settings()
    cases = load_dataset(args.dataset)
    files = list(iter_repository_files(args.repository_path, settings.max_file_size_bytes))
    RepositoryIndexingService(
        CodeChunkingService(),
        create_embedding_service(settings),
        QdrantStore.from_settings(
            settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection_name,
            settings.qdrant_max_retries, settings.qdrant_initial_backoff_seconds,
        ),
    ).index_repository(args.repository_id, files)
    vector = create_retrieval_service(settings)
    hybrid = create_hybrid_retrieval_service(settings)
    reranked = create_reranked_retrieval_service(settings)
    results = [
        evaluate_retrieval(cases, lambda case: vector.retrieve(args.repository_id, case.question, top_k=10), "Vector"),
        evaluate_retrieval(cases, lambda case: hybrid.retrieve(args.repository_id, case.question, top_k=10), "Hybrid"),
        evaluate_retrieval(cases, lambda case: reranked.retrieve(args.repository_id, case.question, top_k=10), "Hybrid + reranker"),
    ]
    json_path, markdown_path = write_retrieval_report(results, args.output)
    print(f"Wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
