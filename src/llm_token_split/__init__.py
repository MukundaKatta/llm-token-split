"""llm-token-split: split long text into overlapping chunks for LLM context windows."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Chunk:
    """A single piece of a split document with position metadata."""

    text: str
    start_char: int  # inclusive char offset in original text
    end_char: int    # exclusive char offset in original text
    index: int       # 0-based chunk index
    total: int       # total number of chunks


class TokenSplitter:
    """Split long documents into overlapping chunks that each fit within an LLM context window.

    Uses a caller-supplied tokenizer (or a chars/4 heuristic) to measure token budgets.
    Overlap lets consecutive chunks share context so retrieval can stitch results cleanly.

    Args:
        chunk_size: Maximum tokens per chunk.
        overlap: Tokens shared between consecutive chunks. Must be < chunk_size.
        tokenizer: Callable that takes a string and returns an int token count.
                   Defaults to ``len(text) // 4`` (chars-per-4 heuristic).

    Raises:
        ValueError: If ``overlap >= chunk_size``.
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        overlap: int = 200,
        tokenizer: Callable[[str], int] | None = None,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._tokenizer: Callable[[str], int] = tokenizer if tokenizer is not None else (
            lambda text: len(text) // 4
        )

    @property
    def chunk_size(self) -> int:
        """Maximum tokens per chunk."""
        return self._chunk_size

    @property
    def overlap(self) -> int:
        """Tokens of overlap shared between consecutive chunks."""
        return self._overlap

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _token_count(self, text: str) -> int:
        return self._tokenizer(text)

    def _find_chunk_end(self, text: str, start: int, budget: int) -> int:
        """Return the exclusive end char index for a chunk starting at ``start``
        that fits within ``budget`` tokens.

        Binary search over character lengths so it works with any tokenizer.
        Returns len(text) if the full remainder fits.
        """
        remaining = text[start:]
        if self._token_count(remaining) <= budget:
            return len(text)

        # Binary search: find largest end such that token_count(text[start:end]) <= budget
        lo, hi = start, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._token_count(text[start:mid]) <= budget:
                lo = mid
            else:
                hi = mid - 1

        # If even one char exceeds budget, advance by 1 to avoid an infinite loop
        return max(lo, start + 1)

    def _find_overlap_start(self, text: str, end: int, overlap_tokens: int) -> int:
        """Return the start char index for the next chunk so that
        text[new_start:end] is approximately ``overlap_tokens`` tokens of overlap.

        Finds the leftmost position such that slicing back from ``end`` gives
        at most ``overlap_tokens`` tokens. This is the start of the next chunk.
        """
        if overlap_tokens == 0:
            return end

        # Binary search: find the smallest start such that token_count(text[start:end]) <= overlap
        # That means: go as far left as possible while keeping the overlap slice within budget.
        lo, hi = 0, end
        best = end  # fallback: no overlap (start at end)
        while lo < hi:
            mid = (lo + hi) // 2
            count = self._token_count(text[mid:end])
            if count <= overlap_tokens:
                # mid is a valid start (overlap fits); try going further left
                best = mid
                hi = mid
            else:
                # too many overlap tokens from mid; move right
                lo = mid + 1
        return best

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fits(self, text: str) -> bool:
        """Return True if the text fits within a single chunk (token count <= chunk_size)."""
        return self._token_count(text) <= self._chunk_size

    def split(self, text: str) -> list[str]:
        """Split text into a list of overlapping chunk strings.

        - Empty string returns ``[""]``.
        - If text fits in one chunk, returns ``[text]``.
        - All chunks have token count <= chunk_size.
        - Consecutive chunks share approximately ``overlap`` tokens of content.
        """
        if not text:
            return [""]
        if self.fits(text):
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = self._find_chunk_end(text, start, self._chunk_size)
            chunks.append(text[start:end])

            if end == len(text):
                break

            # Find where the next chunk should start to preserve overlap
            next_start = self._find_overlap_start(text, end, self._overlap)

            # Guard: always make forward progress
            if next_start >= end:
                next_start = end
            if next_start <= start:
                next_start = start + 1

            # If remaining text from next_start fits, take it and stop
            if self.fits(text[next_start:]):
                chunks.append(text[next_start:])
                break

            start = next_start

        return chunks

    def split_with_meta(self, text: str) -> list[Chunk]:
        """Split text and return Chunk objects with position metadata.

        Each Chunk carries:
        - ``text``: the chunk string
        - ``start_char``: inclusive char offset in the original text
        - ``end_char``: exclusive char offset in the original text
        - ``index``: 0-based chunk index
        - ``total``: total number of chunks
        """
        raw_chunks = self.split(text)
        total = len(raw_chunks)
        result: list[Chunk] = []

        search_from = 0
        for idx, chunk_text in enumerate(raw_chunks):
            # Find the chunk text in the original string starting from search_from
            start_char = text.find(chunk_text, search_from)
            if start_char == -1:
                # Fallback in case of edge cases
                start_char = text.find(chunk_text)
            end_char = start_char + len(chunk_text)
            result.append(
                Chunk(
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    index=idx,
                    total=total,
                )
            )
            # Next search must start after the current chunk's start (chunks can overlap)
            search_from = start_char + 1

        return result

    def split_messages(self, messages: list[dict]) -> list[list[dict]]:
        """Split the content of the last user message into chunks.

        Returns a list of message-list variants, one per chunk. Each variant is
        a deep copy of the original messages with only the last user message's
        ``content`` replaced by the chunk text.

        - If there are no user messages, or the last user message fits, returns ``[messages]``.
        - Never mutates the input.
        """
        # Find the last user message
        last_user_idx: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return [messages]

        content = messages[last_user_idx].get("content", "")
        if not isinstance(content, str):
            # Non-string content (multipart list etc.) — skip splitting
            return [messages]

        if self.fits(content):
            return [messages]

        chunks = self.split(content)
        result: list[list[dict]] = []
        for chunk_text in chunks:
            msgs_copy = copy.deepcopy(messages)
            msgs_copy[last_user_idx]["content"] = chunk_text
            result.append(msgs_copy)

        return result
