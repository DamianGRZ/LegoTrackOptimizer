"""ruamel.yaml + Pydantic v2 catalog loader with file+line error UX."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from .specs import TrackCatalogSpec, check_schema_version

log = logging.getLogger(__name__)


class CatalogLoadError(ValueError):
    """Raised when a YAML catalog fails schema validation after the loader
    has attached file+line context to each error."""


def load_catalog_spec(path: str | Path) -> TrackCatalogSpec:
    """Load a V2 catalog from YAML and return a validated TrackCatalogSpec.

    Uses ruamel.yaml round-trip mode so CommentedMap/CommentedSeq preserve
    .lc (line/column) on each node. Line numbers are re-attached to any
    Pydantic ValidationError as a wrapped CatalogLoadError.
    """
    path = Path(path)
    yaml = YAML()                       # default typ='rt' — preserves .lc
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.load(fh)

    if not isinstance(raw, CommentedMap):
        raise CatalogLoadError(f"{path}: root must be a mapping, got {type(raw).__name__}")

    # Enforce schema version before Pydantic sees the data
    meta = raw.get("meta") or {}
    version_str = meta.get("schema_version") if isinstance(meta, CommentedMap) else None
    if not version_str:
        raise CatalogLoadError(f"{path}: missing meta.schema_version")
    check_schema_version(str(version_str), str(path))

    # Map piece list index → 1-based line number for error reporting
    piece_lines: dict[int, int] = {}
    pieces = raw.get("pieces") or []
    if isinstance(pieces, CommentedSeq):
        for i, node in enumerate(pieces):
            if isinstance(node, CommentedMap) and node.lc is not None:
                piece_lines[i] = node.lc.line + 1   # 0-based → 1-based

    try:
        return TrackCatalogSpec.model_validate(_strip_comments(raw))
    except ValidationError as exc:
        _raise_with_location(exc, path, piece_lines)


def _strip_comments(obj):
    """Recursively convert CommentedMap/CommentedSeq to plain dict/list."""
    if isinstance(obj, CommentedMap):
        return {k: _strip_comments(v) for k, v in obj.items()}
    if isinstance(obj, CommentedSeq):
        return [_strip_comments(v) for v in obj]
    return obj


def _raise_with_location(exc: ValidationError, path: Path,
                         piece_lines: dict[int, int]) -> None:
    messages = []
    for err in exc.errors():
        loc = err["loc"]
        msg = err["msg"]
        typ = err["type"]

        line_hint = ""
        if loc and loc[0] == "pieces" and len(loc) > 1 and isinstance(loc[1], int):
            line_no = piece_lines.get(loc[1], "?")
            line_hint = f"{path.name}:{line_no} "

        field_path = ".".join(str(s) for s in loc)
        messages.append(f"{line_hint}in {field_path}: {msg} [type={typ}]")

    raise CatalogLoadError("\n".join(messages)) from exc
