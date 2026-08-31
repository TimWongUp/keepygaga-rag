from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import jieba

jieba.setLogLevel(30)

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN = re.compile(r"[\w.+#/-]+", re.UNICODE)
SENTENCE_BOUNDARY = frozenset("。！？.!?；;，,：:")
FENCE_MARKER = re.compile(r"^[ \t]*(`{3,}|~{3,})")
URL = re.compile(
    r"(?:https?://|ftp://|www\.)"
    r"[^\s<>()\u3400-\u9fff\u3000-\u303f\uff00-\uff65]+",
    re.IGNORECASE,
)
INLINE_CODE = re.compile(r"`+[^`\n]+`+")
ASCII_SENTENCE_END = frozenset(".!?")
CLAUSE_PUNCTUATION = frozenset("；;，,：:")
EMBEDDING_INPUT_VERSION = "text-title-heading_path-filename-v1"
TOKENIZER_VERSION = "jieba-v1"
CHUNKER_VERSION = "markdown-text-v1"
FTS_BM25_WEIGHTS = (0.0, 1.0, 6.0, 4.0, 2.0)


@dataclass(frozen=True)
class TextChunk:
    ordinal: int
    text: str
    heading_path: str
    title: str
    filename: str
    content_hash: str
    search_text: str
    fts_fields: tuple[str, str, str, str]
    embedding_text: str
    embedding_input_hash: str


@dataclass(frozen=True)
class _ChunkUnit:
    heading_path: str
    text: str
    paragraph_id: int
    separator_before: str
    force_split: bool = False


@dataclass(frozen=True)
class _ChunkDraft:
    heading_path: str
    text: str
    first_paragraph_id: int
    last_paragraph_id: int


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def filename_for_path(value: str | Path) -> str:
    return Path(value).name


def normalize_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = "\n".join(line.rstrip() for line in value.splitlines())
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def strip_frontmatter(value: str) -> str:
    normalized = value.removeprefix("\ufeff").replace("\r\n", "\n").replace(
        "\r", "\n"
    )
    if not normalized.startswith("---\n"):
        return normalized
    closing = normalized.find("\n---\n", 4)
    if closing == -1:
        return normalized
    return normalized[closing + 5 :]


def lexical_text(value: str) -> str:
    tokens: list[str] = []
    seen: set[str] = set()
    for piece in jieba.cut(value, cut_all=False):
        for token in TOKEN.findall(piece.casefold()):
            token = token.strip("_-/")
            if token and token not in seen:
                seen.add(token)
                tokens.append(token)
    return " ".join(tokens)


def lexical_fields(
    *,
    text: str,
    title: str,
    heading_path: str,
    filename: str,
) -> dict[str, str]:
    return {
        "text": lexical_text(text),
        "title": lexical_text(title),
        "heading_path": lexical_text(heading_path),
        "filename": lexical_text(filename),
    }


def lexical_field_values(
    *,
    text: str,
    title: str,
    heading_path: str,
    filename: str,
) -> tuple[str, str, str, str]:
    fields = lexical_fields(
        text=text,
        title=title,
        heading_path=heading_path,
        filename=filename,
    )
    return (
        fields["text"],
        fields["title"],
        fields["heading_path"],
        fields["filename"],
    )


def build_retrieval_text(
    *,
    text: str,
    title: str,
    heading_path: str,
    filename: str,
) -> str:
    return "\n".join(
        part
        for part in (
            f"Title: {title}" if title else "",
            f"Heading path: {heading_path}" if heading_path else "",
            f"Filename: {filename}" if filename else "",
            f"Text:\n{text}" if text else "",
        )
        if part
    )


def build_embedding_input_hash(
    *,
    retrieval_text: str,
    embedding_identity: str,
) -> str:
    return sha256_text(
        f"{EMBEDDING_INPUT_VERSION}\0{embedding_identity}\0{retrieval_text}"
    )


def fts_query(value: str) -> str:
    tokens = lexical_text(value).split()
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _protected_ranges(value: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for match in URL.finditer(value):
        start, end = match.span()
        if end > start and value[end - 1] == ".":
            end -= 1
        ranges.append((start, end))
    ranges.extend(match.span() for match in INLINE_CODE.finditer(value))
    fence_start: int | None = None
    fence_char = ""
    offset = 0
    for line in value.splitlines(keepends=True):
        marker = FENCE_MARKER.match(line)
        if marker is not None:
            if fence_start is None:
                fence_start = offset
                fence_char = marker.group(1)[0]
            elif marker.group(1)[0] == fence_char:
                ranges.append((fence_start, offset + len(line)))
                fence_start = None
                fence_char = ""
        offset += len(line)
    if fence_start is not None:
        ranges.append((fence_start, len(value)))
    return sorted(ranges)


def _in_ranges(index: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in ranges)


def _looks_like_abbreviation(value: str, end: int) -> bool:
    prefix = value[:end].casefold()
    return re.search(
        r"(?:^|[\s(])(?:mr|mrs|ms|dr|prof|e\.g|i\.e|etc|vs)\.$",
        prefix,
    ) is not None


def _is_clear_boundary(value: str, index: int, ranges: list[tuple[int, int]]) -> int | None:
    if _in_ranges(index, ranges):
        return None
    character = value[index]
    if character not in SENTENCE_BOUNDARY:
        return None
    end = index + 1
    while end < len(value) and value[end] in SENTENCE_BOUNDARY:
        if _in_ranges(end, ranges):
            break
        end += 1
    if character == ".":
        previous = value[index - 1] if index else ""
        following = value[end] if end < len(value) else ""
        if previous.isdigit() and following.isdigit():
            return None
        if _looks_like_abbreviation(value, end):
            return None
        if following and not following.isspace() and not (
            "\u3400" <= following <= "\u9fff"
        ):
            return None
    elif character in ASCII_SENTENCE_END:
        following = value[end] if end < len(value) else ""
        if following and not following.isspace() and not (
            "\u3400" <= following <= "\u9fff"
        ):
            return None
    elif character in CLAUSE_PUNCTUATION:
        previous = value[index - 1] if index else ""
        following = value[end] if end < len(value) else ""
        if previous.isdigit() and following.isdigit():
            return None
        if (
            character in ",;:"
            and following
            and not following.isspace()
            and not ("\u3400" <= following <= "\u9fff")
        ):
            return None
    return end


def _semantic_units(value: str) -> list[str]:
    protected = _protected_ranges(value)
    units: list[str] = []
    start = 0
    index = 0
    while index < len(value):
        end = _is_clear_boundary(value, index, protected)
        if end is None:
            index += 1
            continue
        piece = value[start:end]
        if piece.strip():
            units.append(piece)
        start = end
        index = end
    tail = value[start:]
    if tail.strip():
        units.append(tail)
    return units


def _split_oversized(block: str, max_chars: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]
    parts: list[str] = []
    current = ""
    for sentence in _semantic_units(block):
        if len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.extend(
                sentence[index : index + max_chars]
                for index in range(0, len(sentence), max_chars)
            )
            continue
        candidate = f"{current}{sentence}"
        if current and len(candidate) > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _blocks(value: str) -> list[tuple[str, str]]:
    headings: list[str] = []
    blocks: list[tuple[str, str]] = []
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        if current:
            text = "\n".join(current).strip()
            if text:
                blocks.append((" > ".join(headings), text))
            current.clear()

    for line in value.splitlines():
        heading = HEADING.match(line) if not in_fence else None
        if heading is not None:
            flush()
            level = len(heading.group(1))
            headings[:] = headings[: level - 1]
            headings.append(heading.group(2).strip())
            continue
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            flush()
        else:
            current.append(line)
    flush()
    return blocks


def _units_for_block(
    heading_path: str,
    block: str,
    *,
    paragraph_id: int,
    separator_before: str,
    chunk_mode: str,
    max_chars: int,
    fallback_without_heading: bool,
) -> list[_ChunkUnit]:
    if chunk_mode == "structure" and len(block) <= max_chars:
        pieces = (
            _semantic_units(block)
            if fallback_without_heading
            else [block]
        )
    else:
        pieces = _semantic_units(block)

    units: list[_ChunkUnit] = []
    for index, piece in enumerate(pieces):
        before = separator_before if index == 0 else ""
        if len(piece) > max_chars:
            units.append(
                _ChunkUnit(
                    heading_path=heading_path,
                    text=piece,
                    paragraph_id=paragraph_id,
                    separator_before=before,
                    force_split=True,
                )
            )
        else:
            units.append(
                _ChunkUnit(
                    heading_path=heading_path,
                    text=piece,
                    paragraph_id=paragraph_id,
                    separator_before=before,
                )
            )
    return units


def _chunk_drafts(
    units: list[_ChunkUnit],
    *,
    target_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> list[_ChunkDraft]:
    drafts: list[_ChunkDraft] = []
    current_text = ""
    current_heading = ""
    current_first_paragraph: int | None = None
    current_last_paragraph: int | None = None

    def flush() -> None:
        nonlocal current_text
        nonlocal current_heading
        nonlocal current_first_paragraph
        nonlocal current_last_paragraph
        if current_text:
            assert current_first_paragraph is not None
            assert current_last_paragraph is not None
            drafts.append(
                _ChunkDraft(
                    heading_path=current_heading,
                    text=current_text,
                    first_paragraph_id=current_first_paragraph,
                    last_paragraph_id=current_last_paragraph,
                )
            )
        current_text = ""
        current_heading = ""
        current_first_paragraph = None
        current_last_paragraph = None

    def previous_context(
        heading_path: str,
        paragraph_id: int,
    ) -> tuple[str, bool]:
        if not drafts:
            return "", False
        previous = drafts[-1]
        if (
            previous.heading_path != heading_path
            or previous.last_paragraph_id != paragraph_id
        ):
            return "", False
        if not overlap_chars:
            return "", True
        return previous.text[-overlap_chars:], True

    def start_unit(unit: _ChunkUnit) -> None:
        nonlocal current_text
        nonlocal current_heading
        nonlocal current_first_paragraph
        nonlocal current_last_paragraph
        prefix, same_paragraph = previous_context(
            unit.heading_path,
            unit.paragraph_id,
        )
        separator = unit.separator_before if same_paragraph and prefix else ""
        if len(prefix) + len(separator) + len(unit.text) > max_chars:
            separator = ""
            available_prefix = max_chars - len(unit.text)
            prefix = prefix[-max(0, available_prefix) :]
            if not prefix:
                separator = ""
        if len(prefix) + len(separator) + len(unit.text) > max_chars:
            prefix = ""
            separator = ""
        current_text = f"{prefix}{separator}{unit.text}"
        current_heading = unit.heading_path
        current_first_paragraph = unit.paragraph_id
        current_last_paragraph = unit.paragraph_id

    def split_forced_unit(unit: _ChunkUnit) -> None:
        nonlocal current_text
        if current_text:
            flush()
        position = 0
        first_piece = True
        while position < len(unit.text):
            prefix, same_paragraph = previous_context(
                unit.heading_path,
                unit.paragraph_id,
            )
            separator = (
                unit.separator_before
                if first_piece and same_paragraph and prefix
                else ""
            )
            max_prefix = max_chars - len(separator) - 1
            prefix = prefix[-max(0, max_prefix) :]
            if not prefix:
                separator = ""
            available = max_chars - len(prefix) - len(separator)
            piece = unit.text[position : position + available]
            if not piece:  # pragma: no cover - guarded by max_chars validation
                separator = ""
                prefix = ""
                available = max_chars
                piece = unit.text[position : position + available]
            drafts.append(
                _ChunkDraft(
                    heading_path=unit.heading_path,
                    text=f"{prefix}{separator}{piece}",
                    first_paragraph_id=unit.paragraph_id,
                    last_paragraph_id=unit.paragraph_id,
                )
            )
            position += len(piece)
            first_piece = False

    for unit in units:
        if unit.force_split:
            split_forced_unit(unit)
            continue
        if current_text and current_heading != unit.heading_path:
            flush()
        if not current_text:
            start_unit(unit)
        else:
            candidate = (
                f"{current_text}{unit.separator_before}{unit.text}"
            )
            if len(candidate) > max_chars:
                flush()
                start_unit(unit)
            else:
                current_text = candidate
                current_last_paragraph = unit.paragraph_id
        if len(current_text) >= target_chars:
            flush()
    flush()
    return drafts


def chunk_text(
    value: str,
    *,
    source_path: Path,
    target_chars: int,
    max_chars: int,
    embedding_identity: str,
    chunk_mode: str = "structure",
    overlap_chars: int = 0,
) -> list[TextChunk]:
    if (
        not isinstance(target_chars, int)
        or isinstance(target_chars, bool)
        or target_chars <= 0
    ):
        raise ValueError("target_chars must be a positive integer")
    if (
        not isinstance(max_chars, int)
        or isinstance(max_chars, bool)
        or max_chars <= 0
    ):
        raise ValueError("max_chars must be a positive integer")
    if target_chars > max_chars:
        raise ValueError("target_chars must not exceed max_chars")
    if not isinstance(chunk_mode, str) or chunk_mode not in {
        "structure",
        "length",
    }:
        raise ValueError("chunk_mode must be structure or length")
    if (
        not isinstance(overlap_chars, int)
        or isinstance(overlap_chars, bool)
        or overlap_chars < 0
    ):
        raise ValueError("overlap_chars must be a non-negative integer")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be less than max_chars")
    normalized = normalize_text(strip_frontmatter(value))
    if not normalized:
        return []
    raw_blocks = _blocks(normalized)
    title = source_path.stem
    for heading_path, _ in raw_blocks:
        if heading_path:
            title = heading_path.split(" > ", 1)[0]
            break

    fallback_without_heading = not any(
        heading_path for heading_path, _ in raw_blocks
    ) and len(raw_blocks) == 1
    units: list[_ChunkUnit] = []
    for paragraph_id, (heading_path, raw_block) in enumerate(raw_blocks):
        units.extend(
            _units_for_block(
                heading_path,
                raw_block,
                paragraph_id=paragraph_id,
                separator_before="\n\n" if paragraph_id else "",
                chunk_mode=chunk_mode,
                max_chars=max_chars,
                fallback_without_heading=fallback_without_heading,
            )
        )
    grouped = _chunk_drafts(
        units,
        target_chars=target_chars,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    chunks: list[TextChunk] = []
    filename = filename_for_path(source_path)
    for ordinal, draft in enumerate(grouped):
        heading_path = draft.heading_path
        text = draft.text
        embedding_text = build_retrieval_text(
            text=text,
            title=title,
            heading_path=heading_path,
            filename=filename,
        )
        content_hash = sha256_text(text)
        fts_fields = lexical_field_values(
            text=text,
            title=title,
            heading_path=heading_path,
            filename=filename,
        )
        input_hash = build_embedding_input_hash(
            retrieval_text=embedding_text,
            embedding_identity=embedding_identity,
        )
        chunks.append(
            TextChunk(
                ordinal=ordinal,
                text=text,
                heading_path=heading_path,
                title=title,
                filename=filename,
                content_hash=content_hash,
                search_text=fts_fields[0],
                fts_fields=fts_fields,
                embedding_text=embedding_text,
                embedding_input_hash=input_hash,
            )
        )
    return chunks
