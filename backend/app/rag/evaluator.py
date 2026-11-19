"""Retrieval Evaluation Benchmark Runner.

Evaluates HybridRetriever (and individual Semantic & Keyword engines)
against the curated benchmark dataset measuring Recall@5, Top-1 accuracy,
source discrimination, latency, and out-of-domain handling.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from backend.app.rag.retriever import HybridRetriever
from backend.app.api.retrieve import get_retriever
from backend.app.core.logging import logger


class RetrievalEvaluator:
    """Benchmark runner for real estate hybrid retrieval."""

    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.retriever = retriever or get_retriever()

    def evaluate(
        self,
        dataset_path: Path = Path("backend/tests/fixtures/retrieval_eval_dataset.json"),
        top_k: int = 5,
        method: str = "hybrid",
        out_of_domain_threshold: float = 0.45,
    ) -> Dict[str, Any]:
        """Run benchmark evaluation over test cases."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases = data.get("test_cases", [])
        total_test_cases = len(test_cases)
        logger.info(f"Running retrieval evaluation on {total_test_cases} test queries (method={method})...")

        answerable_count = 0
        recall_at_k_hits = 0
        top1_hits = 0
        source_matches = 0
        unanswerable_count = 0
        unanswerable_flagged = 0

        latencies_ms: List[float] = []
        detailed_results: List[Dict[str, Any]] = []

        for case in test_cases:
            qid = case["id"]
            query = case["query"]
            expected_ids = set(case.get("expected_entity_ids", []))
            expected_source = case.get("expected_source")
            is_answerable = case.get("is_answerable", True)

            t0 = time.perf_counter()
            retrieved = self.retriever.retrieve(query=query, top_k=top_k, method=method)
            latency = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency)

            retrieved_entity_ids = [item.entity_id for item in retrieved]
            top_entity_id = retrieved_entity_ids[0] if retrieved_entity_ids else None
            top_source = retrieved[0].source if retrieved else None
            top_similarity = retrieved[0].semantic_similarity if retrieved and retrieved[0].semantic_similarity is not None else 0.0

            result_entry = {
                "id": qid,
                "query": query,
                "is_answerable": is_answerable,
                "expected_entities": list(expected_ids),
                "retrieved_top_entities": retrieved_entity_ids,
                "top_score": retrieved[0].rrf_score if retrieved else 0.0,
                "top_similarity": top_similarity,
                "latency_ms": round(latency, 2),
            }

            if is_answerable:
                answerable_count += 1
                # Check Recall@K
                hit = any(eid in expected_ids for eid in retrieved_entity_ids)
                if hit:
                    recall_at_k_hits += 1

                # Check Top-1
                top1_hit = top_entity_id in expected_ids
                if top1_hit:
                    top1_hits += 1

                # Check Source Accuracy
                if expected_source and expected_source != "cross_source":
                    if top_source == expected_source:
                        source_matches += 1

                result_entry["recall_hit"] = hit
                result_entry["top1_hit"] = top1_hit
            else:
                unanswerable_count += 1
                # Out-of-domain discrimination check:
                # Is top similarity below threshold or does the score indicate weak match?
                is_flagged_unanswerable = top_similarity < out_of_domain_threshold
                if is_flagged_unanswerable:
                    unanswerable_flagged += 1
                result_entry["flagged_unanswerable"] = is_flagged_unanswerable

            detailed_results.append(result_entry)

        # Aggregate Metrics
        recall_at_k = round(recall_at_k_hits / answerable_count, 4) if answerable_count > 0 else 0.0
        top1_accuracy = round(top1_hits / answerable_count, 4) if answerable_count > 0 else 0.0
        source_accuracy = (
            round(source_matches / (answerable_count - 1), 4)  # exclude cross_source
            if answerable_count > 1
            else 0.0
        )
        unanswerable_rejection_rate = (
            round(unanswerable_flagged / unanswerable_count, 4) if unanswerable_count > 0 else 0.0
        )
        avg_latency = round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0.0
        max_latency = round(max(latencies_ms), 2) if latencies_ms else 0.0

        summary = {
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method": method,
            "top_k": top_k,
            "total_queries": total_test_cases,
            "answerable_queries": answerable_count,
            "recall_at_k": recall_at_k,
            "top1_accuracy": top1_accuracy,
            "source_accuracy": source_accuracy,
            "unanswerable_queries": unanswerable_count,
            "unanswerable_rejection_rate": unanswerable_rejection_rate,
            "avg_latency_ms": avg_latency,
            "max_latency_ms": max_latency,
            "detailed_results": detailed_results,
        }

        # Save Report
        report_path = Path("data/processed/retrieval_evaluation_report.json")
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Retrieval evaluation report written to '{report_path}'.")
        except Exception as e:
            logger.warning(f"Could not save evaluation report: {e}")

        return summary


def main():
    """Run retrieval evaluation CLI."""
    evaluator = RetrievalEvaluator()
    summary = evaluator.evaluate()
    print("\n================ RETRIEVAL BENCHMARK EVALUATION ================")
    print(f"Method:                     {summary['method'].upper()}")
    print(f"Total Test Queries:         {summary['total_queries']}")
    print(f"Recall@{summary['top_k']}:                   {summary['recall_at_k'] * 100:.1f}%")
    print(f"Top-1 Precision:            {summary['top1_accuracy'] * 100:.1f}%")
    print(f"Source Match Accuracy:      {summary['source_accuracy'] * 100:.1f}%")
    print(f"Out-of-Domain Rejection:    {summary['unanswerable_rejection_rate'] * 100:.1f}%")
    print(f"Average Latency:            {summary['avg_latency_ms']} ms")
    print(f"Max Latency:                {summary['max_latency_ms']} ms")
    print("================================================================\n")


if __name__ == "__main__":
    main()
