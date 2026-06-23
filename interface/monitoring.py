import os
from dataclasses import dataclass
from functools import lru_cache

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server


MODEL_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)
DOCUMENT_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)
DOCUMENT_CHUNK_BUCKETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


@dataclass(frozen=True)
class AppMetrics:
    model_requests_total: Counter
    model_request_latency_seconds: Histogram
    model_errors_total: Counter
    document_index_latency_seconds: Histogram
    document_index_errors_total: Counter
    document_chunks: Histogram
    model_prompt_tokens_total: Counter
    model_generated_tokens_total: Counter
    model_total_tokens_total: Counter
    model_prompt_tokens_per_second: Gauge
    model_generated_tokens_per_second: Gauge


@lru_cache(maxsize=1)
def get_metrics() -> AppMetrics:
    registry = CollectorRegistry()

    metrics = AppMetrics(
        model_requests_total=Counter(
            "document_sharelock_model_requests_total",
            "Total model completion requests made by the Streamlit app.",
            ("purpose", "answer_mode", "status"),
            registry=registry,
        ),
        model_request_latency_seconds=Histogram(
            "document_sharelock_model_request_latency_seconds",
            "Latency of model completion requests made by the Streamlit app.",
            ("purpose", "answer_mode"),
            buckets=MODEL_LATENCY_BUCKETS,
            registry=registry,
        ),
        model_errors_total=Counter(
            "document_sharelock_model_errors_total",
            "Total model completion errors seen by the Streamlit app.",
            ("purpose", "answer_mode", "error_type"),
            registry=registry,
        ),
        document_index_latency_seconds=Histogram(
            "document_sharelock_document_index_latency_seconds",
            "Latency of document extraction, chunking, embedding, and summary preparation.",
            ("status",),
            buckets=DOCUMENT_LATENCY_BUCKETS,
            registry=registry,
        ),
        document_index_errors_total=Counter(
            "document_sharelock_document_index_errors_total",
            "Total document indexing failures.",
            ("error_type",),
            registry=registry,
        ),
        document_chunks=Histogram(
            "document_sharelock_document_chunks",
            "Number of chunks produced for successfully indexed uploaded documents.",
            buckets=DOCUMENT_CHUNK_BUCKETS,
            registry=registry,
        ),
        model_prompt_tokens_total=Counter(
            "document_sharelock_model_prompt_tokens_total",
            "Prompt tokens reported by llama.cpp responses.",
            ("purpose", "answer_mode"),
            registry=registry,
        ),
        model_generated_tokens_total=Counter(
            "document_sharelock_model_generated_tokens_total",
            "Generated tokens reported by llama.cpp responses.",
            ("purpose", "answer_mode"),
            registry=registry,
        ),
        model_total_tokens_total=Counter(
            "document_sharelock_model_total_tokens_total",
            "Total tokens reported by llama.cpp responses.",
            ("purpose", "answer_mode"),
            registry=registry,
        ),
        model_prompt_tokens_per_second=Gauge(
            "document_sharelock_model_prompt_tokens_per_second",
            "Prompt evaluation tokens per second from the latest llama.cpp response.",
            ("purpose", "answer_mode"),
            registry=registry,
        ),
        model_generated_tokens_per_second=Gauge(
            "document_sharelock_model_generated_tokens_per_second",
            "Generation tokens per second from the latest llama.cpp response.",
            ("purpose", "answer_mode"),
            registry=registry,
        ),
    )

    metrics_port = int(os.getenv("PROMETHEUS_METRICS_PORT", "9100"))
    try:
        start_http_server(metrics_port, addr="0.0.0.0", registry=registry)
    except OSError:
        pass

    return metrics
