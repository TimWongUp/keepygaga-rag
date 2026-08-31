from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from keepygaga_rag.config import EmbeddingProfileConfig, RerankProfileConfig


class ProviderActionRequiredError(RuntimeError):
    """Raised when a provider rejects requests until the user intervenes."""


def _raise_for_provider_status(
    response: httpx.Response,
    *,
    provider_kind: str,
) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_code in {401, 402, 403}:
            raise ProviderActionRequiredError(
                f"{provider_kind} provider request failed with HTTP {status_code}; "
                "check credentials, account balance, quota, or model access"
            ) from exc
        raise


class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class RerankProvider(Protocol):
    @property
    def identity(self) -> str: ...

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[tuple[int, float]]: ...


@dataclass
class OpenAICompatibleEmbeddingProvider:
    profile: EmbeddingProfileConfig
    timeout_seconds: float = 60.0
    batch_size: int = 32

    @property
    def identity(self) -> str:
        return self.profile.identity

    @property
    def dimensions(self) -> int:
        return self.profile.dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        key = os.environ.get(self.profile.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"embedding API key is not set: {self.profile.api_key_env}"
            )
        vectors: list[list[float]] = []
        inputs = [
            f"{self.profile.instruction}\n{text}"
            if self.profile.instruction
            else text
            for text in texts
        ]
        with httpx.Client(
            base_url=self.profile.base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=self.timeout_seconds,
            trust_env=False,
        ) as client:
            for offset in range(0, len(inputs), self.batch_size):
                batch = inputs[offset : offset + self.batch_size]
                response = client.post(
                    "/embeddings",
                    json={
                        "model": self.profile.model,
                        "input": batch,
                        "dimensions": self.profile.dimensions,
                    },
                )
                _raise_for_provider_status(
                    response,
                    provider_kind="embedding",
                )
                payload = response.json()
                rows = payload.get("data")
                if not isinstance(rows, list):
                    raise RuntimeError("embedding response data is not a list")
                indexed_rows: list[tuple[int, dict[str, object]]] = []
                for item in rows:
                    if not isinstance(item, dict):
                        raise RuntimeError(
                            "embedding response contains an invalid index"
                        )
                    index = item.get("index")
                    if type(index) is not int:
                        raise RuntimeError(
                            "embedding response contains an invalid index"
                        )
                    indexed_rows.append((index, item))
                if len(indexed_rows) != len(rows) or sorted(
                    index for index, _item in indexed_rows
                ) != list(range(len(batch))):
                    raise RuntimeError(
                        "embedding response indices do not match the request"
                    )
                ordered = [item for _index, item in sorted(indexed_rows)]
                batch_vectors = [item.get("embedding") for item in ordered]
                if len(batch_vectors) != len(batch):
                    raise RuntimeError(
                        "embedding response count does not match the request"
                    )
                for vector in batch_vectors:
                    if not isinstance(vector, list):
                        raise RuntimeError(
                            "embedding response count does not match the request"
                        )
                    values = [float(value) for value in vector]
                    if len(values) != self.profile.dimensions:
                        raise RuntimeError(
                            "embedding response dimensions do not match the profile"
                        )
                    if not all(math.isfinite(value) for value in values):
                        raise RuntimeError(
                            "embedding response contains a non-finite value"
                        )
                    if self.profile.normalization == "l2":
                        norm = math.sqrt(sum(value * value for value in values))
                        if norm == 0 or not math.isfinite(norm):
                            raise RuntimeError(
                                "embedding response contains a zero vector"
                            )
                        values = [value / norm for value in values]
                    vectors.append(values)
        return vectors


@dataclass
class CohereCompatibleRerankProvider:
    profile: RerankProfileConfig

    @property
    def identity(self) -> str:
        return self.profile.identity

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[tuple[int, float]]:
        key = os.environ.get(self.profile.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"rerank API key is not set: {self.profile.api_key_env}"
            )
        with httpx.Client(
            base_url=self.profile.base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=self.profile.timeout_seconds,
            trust_env=False,
        ) as client:
            response = client.post(
                "/rerank",
                json={
                    "model": self.profile.model,
                    "query": query,
                    "documents": list(documents),
                    "top_n": min(top_n, len(documents)),
                    "return_documents": False,
                },
            )
            _raise_for_provider_status(
                response,
                provider_kind="rerank",
            )
            payload = response.json()
        rows = payload.get("results")
        if not isinstance(rows, list):
            raise RuntimeError("rerank response results is not a list")
        result: list[tuple[int, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            result.append((int(row["index"]), float(row["relevance_score"])))
        return result
