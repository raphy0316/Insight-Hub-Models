from __future__ import annotations

from typing import Iterable, List, Dict, Any
import re


def _normalize_token(token: str, lowercase: bool, min_len: int) -> str | None:
    t = token.strip()
    if lowercase:
        t = t.lower()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" \t\r\n")
    if len(t) < min_len:
        return None
    return t


def normalize_keywords(
    keywords: Iterable[str],
    lowercase: bool = True,
    deduplicate: bool = True,
    min_token_length: int = 2,
) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for kw in keywords:
        norm = _normalize_token(kw, lowercase=lowercase, min_len=min_token_length)
        if not norm:
            continue
        if deduplicate:
            if norm in seen:
                continue
            seen.add(norm)
        normalized.append(norm)
    return normalized


def build_music_text(
    item: Dict[str, Any],
    fields_order: List[str],
    lowercase: bool = True,
    deduplicate_tokens: bool = True,
    joiner: str = " ",
    min_token_length: int = 2,
) -> str:
    segments: List[str] = []
    for field in fields_order:
        if field not in item or item[field] is None:
            continue
        value = item[field]
        if isinstance(value, (list, tuple)):
            tokens = [str(v) for v in value if v is not None]
        else:
            tokens = [str(value)]
        tokens = normalize_keywords(
            tokens,
            lowercase=lowercase,
            deduplicate=deduplicate_tokens,
            min_token_length=min_token_length,
        )
        if tokens:
            segments.append(joiner.join(tokens))
    text = joiner.join([s for s in segments if s])
    return text


def build_image_query_text(
    image_keywords: Iterable[str],
    lowercase: bool = True,
    deduplicate: bool = True,
    joiner: str = " ",
    min_token_length: int = 2,
) -> str:
    tokens = normalize_keywords(
        image_keywords,
        lowercase=lowercase,
        deduplicate=deduplicate,
        min_token_length=min_token_length,
    )
    return joiner.join(tokens)


