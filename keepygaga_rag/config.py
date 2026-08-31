from __future__ import annotations

import os
import re
import shlex
import stat
import tempfile
import tomllib
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "keepygaga-rag.toml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
KNOWLEDGE_HARD_EXCLUDED_DIRS = frozenset({"agents-memory", "_context-backups"})
KNOWLEDGE_SCAN_INTERVALS = {300, 600, 1800}
_TABLE_HEADER = re.compile(
    r"(?m)^[ \t]*\[([^\]\r\n]+)\][ \t]*(?:#[^\r\n]*)?(?:\r?\n|$)"
)


@dataclass(frozen=True)
class EmbeddingProfileConfig:
    provider: str
    base_url: str
    api_key_env: str
    model: str
    dimensions: int
    model_revision: str = ""
    instruction: str = ""
    normalization: str = "l2"
    distance_metric: str = "cosine"
    preprocessor_version: str = "text-v1"
    input_cost_per_million_tokens: float | None = None
    price_as_of: str = ""

    @property
    def identity(self) -> str:
        values = (
            self.provider,
            self.base_url,
            self.model,
            self.model_revision,
            str(self.dimensions),
            self.instruction,
            self.normalization,
            self.distance_metric,
            self.preprocessor_version,
        )
        return "|".join(values)


@dataclass(frozen=True)
class RerankProfileConfig:
    provider: str
    protocol: str
    base_url: str
    api_key_env: str
    model: str
    candidate_limit: int = 20
    timeout_seconds: float = 60.0

    @property
    def identity(self) -> str:
        return "|".join((self.provider, self.protocol, self.base_url, self.model))


@dataclass
class KnowledgeConfig:
    enabled: bool = False
    store: str = ".keepygaga/knowledge-v1"
    table_id: str = "text_chunks_v1"
    embedding_profile: str = "qwen3_4b_2560"
    rerank_profile: str = "qwen3_reranker_4b"
    scan_interval_seconds: int = 300
    stability_seconds: int = 30
    missing_confirmations: int = 2
    chunk_target_chars: int = 2400
    chunk_max_chars: int = 3200
    chunk_mode: str = "structure"
    chunk_overlap_chars: int = 0
    vector_recall_limit: int = 40
    keyword_recall_limit: int = 20
    max_chunks_per_source_file: int = 2
    include_patterns: list[str] = field(default_factory=lambda: ["**/*.md", "**/*.txt"])
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "**/.*/**",
            "**/.obsidian/**",
            "**/.git/**",
            "**/.keepygaga/**",
            "**/.venv/**",
            "**/node_modules/**",
        ]
    )


@dataclass
class KnowledgeAppConfig:
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    embedding_profiles: dict[str, EmbeddingProfileConfig] = field(default_factory=dict)
    rerank_profiles: dict[str, RerankProfileConfig] = field(default_factory=dict)


def _validate_knowledge_chunk_settings(
    target_chars: int,
    max_chars: int,
    chunk_mode: str = "structure",
    overlap_chars: int = 0,
) -> None:
    for key, value in (
        ("chunk_target_chars", target_chars),
        ("chunk_max_chars", max_chars),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"knowledge.{key} must be a positive integer")
    if target_chars > max_chars:
        raise ValueError("knowledge.chunk_target_chars must not exceed chunk_max_chars")
    if not isinstance(chunk_mode, str) or chunk_mode not in {
        "structure",
        "length",
    }:
        raise ValueError("knowledge.chunk_mode must be structure or length")
    if (
        not isinstance(overlap_chars, int)
        or isinstance(overlap_chars, bool)
        or overlap_chars < 0
    ):
        raise ValueError(
            "knowledge.chunk_overlap_chars must be a non-negative integer"
        )
    if overlap_chars >= max_chars:
        raise ValueError(
            "knowledge.chunk_overlap_chars must be less than chunk_max_chars"
        )


def _set_integer_in_section(
    text: str,
    section_name: str,
    key: str,
    value: int,
) -> str:
    newline = "\r\n" if "\r\n" in text else "\n"
    headers = list(_TABLE_HEADER.finditer(text))
    section_index = next(
        (
            index
            for index, header in enumerate(headers)
            if header.group(1).strip() == section_name
        ),
        None,
    )
    if section_index is None:
        prefix = text
        if prefix and not prefix.endswith(("\n", "\r")):
            prefix += newline
        if prefix and not prefix.endswith(newline * 2):
            prefix += newline
        return f"{prefix}[{section_name}]{newline}{key} = {value}{newline}"

    header = headers[section_index]
    section_start = header.end()
    section_end = (
        headers[section_index + 1].start()
        if section_index + 1 < len(headers)
        else len(text)
    )
    section = text[section_start:section_end]
    assignment = re.compile(
        rf"(?m)^(?P<indent>[ \t]*){re.escape(key)}"
        r"(?P<separator>[ \t]*=[ \t]*)"
        r"(?P<value>[+-]?\d+)"
        r"(?P<suffix>[ \t]*(?:#[^\r\n]*)?)$"
    )
    match = assignment.search(section)
    if match is not None:
        replacement = (
            f"{match.group('indent')}{key}{match.group('separator')}"
            f"{value}{match.group('suffix')}"
        )
        updated_section = (
            section[: match.start()] + replacement + section[match.end() :]
        )
        return text[:section_start] + updated_section + text[section_end:]

    content = section.rstrip("\r\n")
    trailing = section[len(content) :]
    if content and not content.endswith(("\n", "\r")):
        content += newline
    content += f"{key} = {value}{newline}"
    if section_end < len(text) and not trailing:
        trailing = newline
    return text[:section_start] + content + trailing + text[section_end:]


def _set_knowledge_integer(text: str, key: str, value: int) -> str:
    return _set_integer_in_section(text, "knowledge", key, value)


def _set_knowledge_string(text: str, key: str, value: str) -> str:
    newline = "\r\n" if "\r\n" in text else "\n"
    headers = list(_TABLE_HEADER.finditer(text))
    knowledge_index = next(
        (
            index
            for index, header in enumerate(headers)
            if header.group(1).strip() == "knowledge"
        ),
        None,
    )
    if knowledge_index is None:
        prefix = text
        if prefix and not prefix.endswith(("\n", "\r")):
            prefix += newline
        if prefix and not prefix.endswith(newline * 2):
            prefix += newline
        return f'{prefix}[knowledge]{newline}{key} = "{value}"{newline}'

    header = headers[knowledge_index]
    section_start = header.end()
    section_end = (
        headers[knowledge_index + 1].start()
        if knowledge_index + 1 < len(headers)
        else len(text)
    )
    section = text[section_start:section_end]
    assignment = re.compile(
        rf"(?m)^(?P<indent>[ \t]*){re.escape(key)}"
        r"(?P<separator>[ \t]*=[ \t]*)"
        r"(?P<quote>[\"'])(?P<value>[^\"']*)(?P=quote)"
        r"(?P<suffix>[ \t]*(?:#[^\r\n]*)?)$"
    )
    match = assignment.search(section)
    if match is not None:
        replacement = (
            f"{match.group('indent')}{key}{match.group('separator')}"
            f"{match.group('quote')}{value}{match.group('quote')}"
            f"{match.group('suffix')}"
        )
        updated_section = (
            section[: match.start()] + replacement + section[match.end() :]
        )
        return text[:section_start] + updated_section + text[section_end:]

    content = section.rstrip("\r\n")
    trailing = section[len(content) :]
    if content and not content.endswith(("\n", "\r")):
        content += newline
    content += f'{key} = "{value}"{newline}'
    if section_end < len(text) and not trailing:
        trailing = newline
    return text[:section_start] + content + trailing + text[section_end:]


def _atomic_write_config(path: Path, text: str, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def update_knowledge_chunk_settings(
    path: Path | str,
    *,
    target_chars: int,
    max_chars: int,
    chunk_mode: str = "structure",
    overlap_chars: int = 0,
) -> KnowledgeAppConfig:
    _validate_knowledge_chunk_settings(
        target_chars,
        max_chars,
        chunk_mode,
        overlap_chars,
    )
    config_path = Path(path).expanduser().resolve()
    load_config(config_path)
    if config_path.exists():
        file_stat = config_path.lstat()
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("keepygaga-rag.toml must be a regular file")
        original = config_path.read_bytes().decode("utf-8")
        mode = stat.S_IMODE(file_stat.st_mode)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        original = ""
        mode = 0o600
    updated = _set_knowledge_string(
        _set_knowledge_integer(
            _set_knowledge_integer(original, "chunk_target_chars", target_chars),
            "chunk_max_chars",
            max_chars,
        ),
        "chunk_mode",
        chunk_mode,
    )
    updated = _set_knowledge_integer(
        updated,
        "chunk_overlap_chars",
        overlap_chars,
    )
    parsed = tomllib.loads(updated)
    knowledge = parsed.get("knowledge")
    if not isinstance(knowledge, dict):  # pragma: no cover
        raise RuntimeError("knowledge settings did not persist as a TOML table")
    if (
        knowledge.get("chunk_target_chars") != target_chars
        or knowledge.get("chunk_max_chars") != max_chars
        or knowledge.get("chunk_mode") != chunk_mode
        or knowledge.get("chunk_overlap_chars") != overlap_chars
    ):  # pragma: no cover
        raise RuntimeError("knowledge chunk settings did not persist")
    if updated != original:
        _atomic_write_config(config_path, updated, mode)
    return load_config(config_path)


def update_knowledge_retrieval_settings(
    path: Path | str,
    *,
    vector_recall_limit: int,
    keyword_recall_limit: int,
    rerank_candidate_limit: int,
    max_chunks_per_source_file: int,
    rerank_profile: str,
) -> KnowledgeAppConfig:
    values = {
        "vector_recall_limit": vector_recall_limit,
        "keyword_recall_limit": keyword_recall_limit,
        "rerank_candidate_limit": rerank_candidate_limit,
        "max_chunks_per_source_file": max_chunks_per_source_file,
    }
    for key, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"knowledge.{key} must be a positive integer")

    config_path = Path(path).expanduser().resolve()
    current = load_config(config_path)
    if rerank_profile not in current.rerank_profiles:
        raise ValueError(
            "knowledge.rerank_profile does not name a configured profile"
        )
    if config_path.exists():
        file_stat = config_path.lstat()
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("keepygaga-rag.toml must be a regular file")
        original = config_path.read_bytes().decode("utf-8")
        mode = stat.S_IMODE(file_stat.st_mode)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        original = ""
        mode = 0o600

    updated = original
    updated = _set_knowledge_integer(
        updated, "vector_recall_limit", vector_recall_limit
    )
    updated = _set_knowledge_integer(
        updated, "keyword_recall_limit", keyword_recall_limit
    )
    updated = _set_knowledge_integer(
        updated,
        "max_chunks_per_source_file",
        max_chunks_per_source_file,
    )
    headers = list(_TABLE_HEADER.finditer(updated))
    profile_section = f"rerank.profiles.{rerank_profile}"
    if not any(
        header.group(1).strip() == profile_section for header in headers
    ):
        profile_section = "rerank"
    updated = _set_integer_in_section(
        updated,
        profile_section,
        "candidate_limit",
        rerank_candidate_limit,
    )
    parsed = tomllib.loads(updated)
    knowledge = parsed.get("knowledge")
    if not isinstance(knowledge, dict):  # pragma: no cover
        raise RuntimeError("knowledge settings did not persist as a TOML table")
    if any(
        knowledge.get(key) != value
        for key, value in (
            ("vector_recall_limit", vector_recall_limit),
            ("keyword_recall_limit", keyword_recall_limit),
            ("max_chunks_per_source_file", max_chunks_per_source_file),
        )
    ):  # pragma: no cover
        raise RuntimeError("knowledge retrieval settings did not persist")
    parsed_rerank = parsed.get("rerank")
    persisted_candidate = None
    if isinstance(parsed_rerank, dict):
        profiles = parsed_rerank.get("profiles")
        if isinstance(profiles, dict) and isinstance(
            profiles.get(rerank_profile), dict
        ):
            persisted_candidate = profiles[rerank_profile].get("candidate_limit")
        elif profile_section == "rerank":
            persisted_candidate = parsed_rerank.get("candidate_limit")
    if persisted_candidate != rerank_candidate_limit:  # pragma: no cover
        raise RuntimeError("rerank candidate setting did not persist")
    if updated != original:
        _atomic_write_config(config_path, updated, mode)
    return load_config(config_path)


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_double:
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return value[:index].rstrip()
    return value.strip()


def parse_env_lines(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("\ufeff").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, raw_value = line.partition("=")
        if not separator or not key.strip():
            continue
        cleaned = _strip_inline_comment(raw_value.strip())
        try:
            parts = shlex.split(cleaned, posix=True)
        except ValueError:
            parts = [cleaned.strip("\"'")]
        values[key.strip()] = parts[0] if len(parts) == 1 else cleaned.strip("\"'")
    return values


def load_env_file(path: Path = DEFAULT_ENV_PATH) -> None:
    if not path.is_file():
        return
    for key, value in parse_env_lines(path.read_text(encoding="utf-8")).items():
        os.environ.setdefault(key, value)


def _bool(table: dict[str, Any], key: str, default: bool, scope: str) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{scope}.{key} must be a boolean")
    return value


def _string_list(value: Any, scope: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{scope} must be a string list")
    return list(value)


def _positive_int(
    table: dict[str, Any], key: str, default: int, scope: str
) -> int:
    value = table.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{scope}.{key} must be a positive integer")
    return value


def _positive_float(
    table: dict[str, Any], key: str, default: float, scope: str
) -> float:
    value = table.get(key, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) <= 0
    ):
        raise ValueError(f"{scope}.{key} must be a positive number")
    return float(value)


def _optional_nonnegative_float(
    table: dict[str, Any], key: str, scope: str
) -> float | None:
    value = table.get(key)
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) < 0
    ):
        raise ValueError(f"{scope}.{key} must be a non-negative number")
    return float(value)


def _required_string(table: dict[str, Any], key: str, scope: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{scope}.{key} must be a non-empty string")
    return value.strip()


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> KnowledgeAppConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    load_env_file(config_path.parent / ".env")

    config = KnowledgeAppConfig()
    data: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)

    knowledge = data.get("knowledge", {})
    if not isinstance(knowledge, dict):
        raise ValueError("knowledge must be a table")
    config.knowledge.enabled = _bool(
        knowledge, "enabled", config.knowledge.enabled, "knowledge"
    )
    config.knowledge.store = str(
        knowledge.get("store", config.knowledge.store)
    ).strip()
    config.knowledge.table_id = str(
        knowledge.get("table_id", config.knowledge.table_id)
    ).strip()
    config.knowledge.embedding_profile = str(
        knowledge.get(
            "embedding_profile", config.knowledge.embedding_profile
        )
    ).strip()
    config.knowledge.rerank_profile = str(
        knowledge.get("rerank_profile", config.knowledge.rerank_profile)
    ).strip()
    config.knowledge.scan_interval_seconds = _positive_int(
        knowledge,
        "scan_interval_seconds",
        config.knowledge.scan_interval_seconds,
        "knowledge",
    )
    if config.knowledge.scan_interval_seconds not in KNOWLEDGE_SCAN_INTERVALS:
        raise ValueError(
            "knowledge.scan_interval_seconds must be one of: 300, 600, 1800"
        )
    config.knowledge.stability_seconds = _positive_int(
        knowledge,
        "stability_seconds",
        config.knowledge.stability_seconds,
        "knowledge",
    )
    config.knowledge.missing_confirmations = _positive_int(
        knowledge,
        "missing_confirmations",
        config.knowledge.missing_confirmations,
        "knowledge",
    )
    config.knowledge.chunk_target_chars = _positive_int(
        knowledge,
        "chunk_target_chars",
        config.knowledge.chunk_target_chars,
        "knowledge",
    )
    config.knowledge.chunk_max_chars = _positive_int(
        knowledge,
        "chunk_max_chars",
        config.knowledge.chunk_max_chars,
        "knowledge",
    )
    raw_chunk_mode = knowledge.get("chunk_mode", config.knowledge.chunk_mode)
    if not isinstance(raw_chunk_mode, str) or raw_chunk_mode not in {
        "structure",
        "length",
    }:
        raise ValueError("knowledge.chunk_mode must be structure or length")
    config.knowledge.chunk_mode = raw_chunk_mode
    raw_overlap = knowledge.get(
        "chunk_overlap_chars",
        config.knowledge.chunk_overlap_chars,
    )
    if (
        not isinstance(raw_overlap, int)
        or isinstance(raw_overlap, bool)
        or raw_overlap < 0
    ):
        raise ValueError(
            "knowledge.chunk_overlap_chars must be a non-negative integer"
        )
    config.knowledge.chunk_overlap_chars = raw_overlap
    if config.knowledge.chunk_target_chars > config.knowledge.chunk_max_chars:
        raise ValueError(
            "knowledge.chunk_target_chars must not exceed chunk_max_chars"
        )
    if config.knowledge.chunk_overlap_chars >= config.knowledge.chunk_max_chars:
        raise ValueError(
            "knowledge.chunk_overlap_chars must be less than chunk_max_chars"
        )
    config.knowledge.vector_recall_limit = _positive_int(
        knowledge,
        "vector_recall_limit",
        config.knowledge.vector_recall_limit,
        "knowledge",
    )
    config.knowledge.keyword_recall_limit = _positive_int(
        knowledge,
        "keyword_recall_limit",
        config.knowledge.keyword_recall_limit,
        "knowledge",
    )
    config.knowledge.max_chunks_per_source_file = _positive_int(
        knowledge,
        "max_chunks_per_source_file",
        config.knowledge.max_chunks_per_source_file,
        "knowledge",
    )
    if "include_patterns" in knowledge:
        config.knowledge.include_patterns = _string_list(
            knowledge["include_patterns"], "knowledge.include_patterns"
        )
    if "exclude_patterns" in knowledge:
        config.knowledge.exclude_patterns = _string_list(
            knowledge["exclude_patterns"], "knowledge.exclude_patterns"
        )
    if not config.knowledge.store or not config.knowledge.table_id:
        raise ValueError("knowledge.store and knowledge.table_id must not be empty")

    embedding = data.get("embedding", {})
    if not isinstance(embedding, dict):
        raise ValueError("embedding must be a table")
    embedding_profiles = embedding.get("profiles", {})
    if not isinstance(embedding_profiles, dict):
        raise ValueError("embedding.profiles must be a table")
    for name, raw_profile in embedding_profiles.items():
        scope = f"embedding.profiles.{name}"
        if not isinstance(raw_profile, dict):
            raise ValueError(f"{scope} must be a table")
        config.embedding_profiles[str(name)] = EmbeddingProfileConfig(
            provider=_required_string(raw_profile, "provider", scope),
            base_url=_required_string(raw_profile, "base_url", scope).rstrip("/"),
            api_key_env=_required_string(raw_profile, "api_key_env", scope),
            model=_required_string(raw_profile, "model", scope),
            dimensions=_positive_int(raw_profile, "dimensions", 0, scope),
            model_revision=str(raw_profile.get("revision", "")).strip(),
            instruction=str(raw_profile.get("instruction", "")).strip(),
            normalization=str(
                raw_profile.get("normalization", "l2")
            ).strip(),
            distance_metric=str(
                raw_profile.get("distance_metric", "cosine")
            ).strip(),
            preprocessor_version=str(
                raw_profile.get("preprocessor_version", "text-v1")
            ).strip(),
            input_cost_per_million_tokens=_optional_nonnegative_float(
                raw_profile, "input_cost_per_million_tokens", scope
            ),
            price_as_of=str(raw_profile.get("price_as_of", "")).strip(),
        )
        profile = config.embedding_profiles[str(name)]
        if profile.normalization not in {"l2", "none"}:
            raise ValueError(f"{scope}.normalization must be l2 or none")
        if profile.distance_metric not in {"cosine", "l2", "dot"}:
            raise ValueError(
                f"{scope}.distance_metric must be cosine, l2, or dot"
            )

    rerank = data.get("rerank", {})
    if not isinstance(rerank, dict):
        raise ValueError("rerank must be a table")
    rerank_profiles = rerank.get("profiles")
    if rerank_profiles is None and rerank.get("provider"):
        rerank_profiles = {config.knowledge.rerank_profile: rerank}
    if rerank_profiles is None:
        rerank_profiles = {}
    if not isinstance(rerank_profiles, dict):
        raise ValueError("rerank.profiles must be a table")
    for name, raw_profile in rerank_profiles.items():
        scope = f"rerank.profiles.{name}"
        if not isinstance(raw_profile, dict):
            raise ValueError(f"{scope} must be a table")
        config.rerank_profiles[str(name)] = RerankProfileConfig(
            provider=_required_string(raw_profile, "provider", scope),
            protocol=_required_string(raw_profile, "protocol", scope),
            base_url=_required_string(raw_profile, "base_url", scope).rstrip("/"),
            api_key_env=_required_string(raw_profile, "api_key_env", scope),
            model=_required_string(raw_profile, "model", scope),
            candidate_limit=_positive_int(
                raw_profile, "candidate_limit", 20, scope
            ),
            timeout_seconds=_positive_float(
                raw_profile, "timeout_seconds", 60.0, scope
            ),
        )

    if config.knowledge.enabled:
        if config.knowledge.embedding_profile not in config.embedding_profiles:
            raise ValueError(
                "knowledge.embedding_profile does not name a configured profile"
            )
        if config.knowledge.rerank_profile not in config.rerank_profiles:
            raise ValueError(
                "knowledge.rerank_profile does not name a configured profile"
            )
    return config
