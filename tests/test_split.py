"""Tests for llm_token_split.TokenSplitter."""

import copy

import pytest

from llm_token_split import Chunk, TokenSplitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def word_tokenizer(text: str) -> int:
    """Toy tokenizer: one token per whitespace-separated word (or 1 if empty)."""
    return len(text.split()) if text.strip() else 0


def char_tokenizer(text: str) -> int:
    """Exact char-count tokenizer for precise size assertions."""
    return len(text)


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_overlap_equal_chunk_size_raises():
    with pytest.raises(ValueError, match="overlap"):
        TokenSplitter(chunk_size=100, overlap=100)


def test_overlap_greater_than_chunk_size_raises():
    with pytest.raises(ValueError, match="overlap"):
        TokenSplitter(chunk_size=100, overlap=150)


def test_properties_returned_correctly():
    s = TokenSplitter(chunk_size=500, overlap=50)
    assert s.chunk_size == 500
    assert s.overlap == 50


def test_default_chunk_size_and_overlap():
    s = TokenSplitter()
    assert s.chunk_size == 2000
    assert s.overlap == 200


# ---------------------------------------------------------------------------
# fits()
# ---------------------------------------------------------------------------


def test_fits_true_when_under_budget():
    s = TokenSplitter(chunk_size=10, overlap=1, tokenizer=char_tokenizer)
    assert s.fits("hello") is True


def test_fits_true_exactly_at_budget():
    s = TokenSplitter(chunk_size=5, overlap=1, tokenizer=char_tokenizer)
    assert s.fits("hello") is True


def test_fits_false_when_over_budget():
    s = TokenSplitter(chunk_size=4, overlap=1, tokenizer=char_tokenizer)
    assert s.fits("hello") is False


# ---------------------------------------------------------------------------
# split() — basic behaviour
# ---------------------------------------------------------------------------


def test_empty_string_returns_list_with_empty_string():
    s = TokenSplitter()
    assert s.split("") == [""]


def test_short_text_returns_single_chunk():
    s = TokenSplitter(chunk_size=1000, tokenizer=char_tokenizer)
    text = "This is a short document."
    assert s.split(text) == [text]


def test_long_text_produces_multiple_chunks():
    s = TokenSplitter(chunk_size=10, overlap=2, tokenizer=char_tokenizer)
    text = "a" * 50
    chunks = s.split(text)
    assert len(chunks) > 1


def test_all_chunks_fit_within_chunk_size():
    s = TokenSplitter(chunk_size=20, overlap=5, tokenizer=char_tokenizer)
    text = "x" * 200
    for chunk in s.split(text):
        assert char_tokenizer(chunk) <= 20


def test_chunks_cover_full_text_content():
    """Removing the overlap between consecutive chunks should recover all unique content."""
    s = TokenSplitter(chunk_size=10, overlap=3, tokenizer=char_tokenizer)
    # Use a text with distinct characters to verify coverage without ambiguity
    text = "abcdefghijklmnopqrstuvwxyz" * 3  # 78 chars
    chunks = s.split(text)
    # The first chunk starts at 0 and the last chunk must end at len(text)
    assert text.startswith(chunks[0])
    assert text.endswith(chunks[-1])


def test_consecutive_chunks_share_overlap_content():
    """Each pair of consecutive chunks should share some text (overlap > 0)."""
    s = TokenSplitter(chunk_size=15, overlap=5, tokenizer=char_tokenizer)
    text = "a" * 100
    chunks = s.split(text)
    assert len(chunks) >= 2
    for i in range(len(chunks) - 1):
        # The tail of chunk[i] should appear at the head of chunk[i+1]
        shared = chunks[i][-5:]  # last `overlap` chars of previous chunk
        assert chunks[i + 1].startswith(shared)


def test_single_chunk_when_exactly_fits():
    s = TokenSplitter(chunk_size=5, overlap=1, tokenizer=char_tokenizer)
    text = "hello"
    assert s.split(text) == ["hello"]


def test_custom_tokenizer_is_used():
    """A tokenizer that always returns 0 makes everything fit → single chunk."""
    always_zero = lambda _: 0  # noqa: E731
    s = TokenSplitter(chunk_size=1, overlap=0, tokenizer=always_zero)
    long_text = "word " * 500
    chunks = s.split(long_text)
    assert chunks == [long_text]


def test_word_tokenizer_chunks():
    """With a word tokenizer, chunks should contain at most chunk_size words."""
    s = TokenSplitter(chunk_size=5, overlap=1, tokenizer=word_tokenizer)
    text = " ".join(f"word{i}" for i in range(30))
    chunks = s.split(text)
    for chunk in chunks:
        assert word_tokenizer(chunk) <= 5


# ---------------------------------------------------------------------------
# split_with_meta()
# ---------------------------------------------------------------------------


def test_split_with_meta_index_and_total():
    s = TokenSplitter(chunk_size=10, overlap=2, tokenizer=char_tokenizer)
    text = "x" * 60
    chunks = s.split_with_meta(text)
    assert all(c.total == len(chunks) for c in chunks)
    for i, c in enumerate(chunks):
        assert c.index == i


def test_split_with_meta_returns_chunk_objects():
    s = TokenSplitter(chunk_size=10, overlap=2, tokenizer=char_tokenizer)
    text = "abcdefghij" * 5
    chunks = s.split_with_meta(text)
    assert all(isinstance(c, Chunk) for c in chunks)


def test_split_with_meta_start_end_bracket_text():
    s = TokenSplitter(chunk_size=15, overlap=3, tokenizer=char_tokenizer)
    text = "Hello, World! This is a longer piece of text for testing."
    chunks = s.split_with_meta(text)
    for c in chunks:
        assert text[c.start_char : c.end_char] == c.text


def test_split_with_meta_single_chunk():
    s = TokenSplitter(chunk_size=1000, tokenizer=char_tokenizer)
    text = "short"
    chunks = s.split_with_meta(text)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].total == 1
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(text)


# ---------------------------------------------------------------------------
# split_messages()
# ---------------------------------------------------------------------------


def test_split_messages_fits_returns_original():
    s = TokenSplitter(chunk_size=1000, tokenizer=char_tokenizer)
    messages = [{"role": "user", "content": "hi"}]
    result = s.split_messages(messages)
    assert result == [messages]


def test_split_messages_no_user_message_returns_original():
    s = TokenSplitter(chunk_size=10, overlap=2, tokenizer=char_tokenizer)
    messages = [{"role": "system", "content": "You are helpful."}]
    result = s.split_messages(messages)
    assert result == [messages]


def test_split_messages_long_content_returns_multiple_variants():
    s = TokenSplitter(chunk_size=10, overlap=2, tokenizer=char_tokenizer)
    long_content = "x" * 80
    messages = [{"role": "user", "content": long_content}]
    result = s.split_messages(messages)
    assert len(result) > 1


def test_split_messages_does_not_mutate_input():
    s = TokenSplitter(chunk_size=10, overlap=2, tokenizer=char_tokenizer)
    long_content = "x" * 80
    messages = [{"role": "user", "content": long_content}]
    original = copy.deepcopy(messages)
    s.split_messages(messages)
    assert messages == original


def test_split_messages_non_user_messages_preserved():
    s = TokenSplitter(chunk_size=10, overlap=2, tokenizer=char_tokenizer)
    long_content = "y" * 80
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "assistant", "content": "Previous reply"},
        {"role": "user", "content": long_content},
    ]
    result = s.split_messages(messages)
    assert len(result) > 1
    for variant in result:
        assert variant[0] == {"role": "system", "content": "System prompt"}
        assert variant[1] == {"role": "assistant", "content": "Previous reply"}
        assert variant[2]["role"] == "user"
        assert variant[2]["content"] != long_content  # replaced by chunk


def test_split_messages_last_user_message_is_split():
    """Only the last user message should be split; earlier ones stay unchanged."""
    s = TokenSplitter(chunk_size=10, overlap=2, tokenizer=char_tokenizer)
    long_content = "z" * 80
    messages = [
        {"role": "user", "content": "first user turn"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": long_content},
    ]
    result = s.split_messages(messages)
    for variant in result:
        # First user message untouched
        assert variant[0]["content"] == "first user turn"
        # Last user message is a chunk
        assert variant[2]["content"] != long_content


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


def test_all_chunk_token_counts_respect_budget_with_custom_tokenizer():
    """Every chunk produced must have token count <= chunk_size under custom tokenizer."""
    s = TokenSplitter(chunk_size=5, overlap=1, tokenizer=word_tokenizer)
    text = " ".join(f"word{i}" for i in range(40))
    chunks = s.split(text)
    for chunk in chunks:
        assert word_tokenizer(chunk) <= 5


def test_split_with_meta_total_matches_split_length():
    """split_with_meta().total must equal len(split())."""
    s = TokenSplitter(chunk_size=12, overlap=3, tokenizer=char_tokenizer)
    text = "abcdefghij" * 6
    raw = s.split(text)
    meta = s.split_with_meta(text)
    assert all(c.total == len(raw) for c in meta)
    assert len(meta) == len(raw)


def test_split_with_meta_offsets_correct_with_repeated_content():
    """Offsets must map to the true chunk positions even when content repeats.

    Regression: with identical characters the old implementation re-located
    chunks via ``str.find`` advancing one char at a time, which collapsed the
    reported offsets to 0,1,2,... instead of the real chunk boundaries.
    """
    s = TokenSplitter(chunk_size=10, overlap=4, tokenizer=char_tokenizer)
    text = "a" * 40
    meta = s.split_with_meta(text)
    raw = s.split(text)
    assert len(meta) == len(raw)
    for c, chunk_text in zip(meta, raw, strict=True):
        # Offsets bracket the original text exactly...
        assert text[c.start_char : c.end_char] == c.text
        # ...and agree with what split() returned for that index.
        assert c.text == chunk_text
    # Consecutive chunks must advance (no two chunks share a start offset).
    starts = [c.start_char for c in meta]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)
    # First chunk anchored at 0, last chunk reaches the end of the text.
    assert meta[0].start_char == 0
    assert meta[-1].end_char == len(text)
