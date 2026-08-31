from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol, cast

import lancedb
import pyarrow as pa


class VectorStore(Protocol):
    def ensure_table(self) -> None: ...

    def add(self, rows: Sequence[dict[str, object]]) -> None: ...

    def delete(self, chunk_ids: Sequence[str]) -> None: ...

    def delete_across_history(
        self,
        chunk_ids: Sequence[str],
        *,
        table_names: Sequence[str],
    ) -> None: ...

    def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]: ...

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        source_ids: Sequence[str] = (),
    ) -> list[dict[str, object]]: ...


class VectorSchemaMismatch(RuntimeError):
    """Raised when a Lance table does not match the configured vector schema."""


class UnavailableVectorStore:
    """Read-only placeholder that lets lexical search continue."""

    def __init__(self, error: Exception):
        self.error = error
        self.table_name = ""

    def _raise(self) -> None:
        raise RuntimeError(
            f"vector store unavailable: {type(self.error).__name__}"
        ) from self.error

    def ensure_table(self) -> None:
        self._raise()

    def reopen(self) -> None:
        self._raise()

    def add(self, rows: Sequence[dict[str, object]]) -> None:
        self._raise()

    def delete(self, chunk_ids: Sequence[str]) -> None:
        self._raise()

    def delete_across_history(
        self,
        chunk_ids: Sequence[str],
        *,
        table_names: Sequence[str],
    ) -> None:
        self._raise()

    def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        self._raise()
        return {}

    def get_vector_metadata(
        self, chunk_ids: Sequence[str]
    ) -> dict[str, dict[str, str]]:
        self._raise()
        return {}

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        source_ids: Sequence[str] = (),
    ) -> list[dict[str, object]]:
        self._raise()
        return []


def _sql_strings(values: Sequence[str]) -> str:
    return ", ".join(f"'{value.replace(chr(39), chr(39) * 2)}'" for value in values)


class LanceVectorStore:
    def __init__(
        self,
        root: Path,
        *,
        table_name: str,
        table_prefix: str | None = None,
        dimensions: int,
        embedding_space_id: str,
        distance_metric: str = "cosine",
        readonly: bool = False,
    ):
        self.root = root.expanduser().resolve()
        self.readonly = readonly
        if readonly:
            if not self.root.is_dir():
                raise FileNotFoundError(
                    f"LanceDB store does not exist: {self.root}"
                )
        else:
            self.root.mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self.dimensions = dimensions
        self.embedding_space_id = embedding_space_id
        self.distance_metric = distance_metric
        self.database = lancedb.connect(
            str(self.root),
            read_consistency_interval=timedelta(seconds=0),
        )

    def reopen(self) -> None:
        """Replace the database handle after a cross-process or stale-handle error."""
        self.database = lancedb.connect(
            str(self.root),
            read_consistency_interval=timedelta(seconds=0),
        )

    def drop(self) -> None:
        if self.readonly:
            raise RuntimeError("vector store is read-only")
        self.database.drop_table(self.table_name)

    def _expected_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("chunk_id", pa.string(), nullable=False),
                pa.field("source_id", pa.string(), nullable=False),
                pa.field("source_file_id", pa.int64(), nullable=False),
                pa.field("embedding_input_hash", pa.string(), nullable=False),
                pa.field("embedding_space_id", pa.string(), nullable=False),
                pa.field("generation", pa.int64(), nullable=False),
                pa.field(
                    "vector",
                    pa.list_(pa.float32(), self.dimensions),
                    nullable=False,
                ),
            ]
        )

    def _validate_schema(self, table: Any) -> None:
        actual = table.schema
        expected = self._expected_schema()
        if actual.names != expected.names:
            raise VectorSchemaMismatch(
                f"table {self.table_name} fields do not match configured schema"
            )
        for expected_field in expected:
            actual_field = actual.field(expected_field.name)
            if (
                actual_field.type != expected_field.type
                or actual_field.nullable != expected_field.nullable
            ):
                raise VectorSchemaMismatch(
                    f"table {self.table_name} field {expected_field.name} "
                    f"does not match configured schema"
                )

    def _open_table(self) -> Any:
        table = self.database.open_table(self.table_name)
        self._validate_schema(table)
        return table

    def list_tables(self) -> list[str]:
        return [str(name) for name in self.database.list_tables(limit=None).tables]

    def ensure_table(self) -> None:
        if self.readonly:
            self._open_table()
            return
        self.database.create_table(
            self.table_name,
            schema=self._expected_schema(),
            exist_ok=True,
        )
        self._open_table()

    def add(self, rows: Sequence[dict[str, object]]) -> None:
        if not rows:
            return
        if self.readonly:
            raise RuntimeError("vector store is read-only")
        self.ensure_table()
        self.delete([str(row["chunk_id"]) for row in rows])
        self._open_table().add(list(rows))

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        if self.readonly:
            raise RuntimeError("vector store is read-only")
        self.ensure_table()
        self._open_table().delete(
            f"chunk_id IN ({_sql_strings(chunk_ids)})"
        )

    def delete_across_history(
        self,
        chunk_ids: Sequence[str],
        *,
        table_names: Sequence[str],
    ) -> None:
        if not chunk_ids:
            return
        if self.readonly:
            raise RuntimeError("vector store is read-only")
        existing = set(self.database.list_tables(limit=None).tables)
        for table_name in dict.fromkeys(table_names):
            if table_name not in existing:
                continue
            self.database.open_table(table_name).delete(
                f"chunk_id IN ({_sql_strings(chunk_ids)})"
            )

    def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        if not chunk_ids:
            return {}
        self.ensure_table()
        arrow = (
            self._open_table()
            .search()
            .where(f"chunk_id IN ({_sql_strings(chunk_ids)})")
            .select(["chunk_id", "vector"])
            .limit(len(chunk_ids))
            .to_arrow()
        )
        result: dict[str, list[float]] = {}
        for row in arrow.to_pylist():
            result[str(row["chunk_id"])] = [
                float(value) for value in row["vector"]
            ]
        return result

    def get_vector_metadata(
        self, chunk_ids: Sequence[str]
    ) -> dict[str, dict[str, str]]:
        """Return vector identity fields without materializing vector values."""
        if not chunk_ids:
            return {}
        self.ensure_table()
        arrow = (
            self._open_table()
            .search()
            .where(f"chunk_id IN ({_sql_strings(chunk_ids)})")
            .select(["chunk_id", "embedding_space_id", "embedding_input_hash"])
            .limit(len(chunk_ids))
            .to_arrow()
        )
        return {
            str(row["chunk_id"]): {
                "embedding_space_id": str(row["embedding_space_id"]),
                "embedding_input_hash": str(row["embedding_input_hash"]),
            }
            for row in arrow.to_pylist()
        }

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        source_ids: Sequence[str] = (),
    ) -> list[dict[str, object]]:
        self.ensure_table()
        query = (
            cast(
                Any,
                self._open_table().search(list(vector)),
            )
            .distance_type(self.distance_metric)
            .select(["chunk_id", "_distance"])
            .limit(limit)
        )
        filters = [f"embedding_space_id = '{self.embedding_space_id}'"]
        if source_ids:
            filters.append(f"source_id IN ({_sql_strings(source_ids)})")
        arrow = query.where(" AND ".join(filters), prefilter=True).to_arrow()
        return [
            {
                "chunk_id": str(row["chunk_id"]),
                "distance": float(row["_distance"]),
            }
            for row in arrow.to_pylist()
        ]
