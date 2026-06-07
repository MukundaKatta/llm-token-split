# llm-token-split

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Split long documents into overlapping chunks that each fit within an LLM context window.**

For RAG pipelines, summarization, and agent workflows that need to feed a long
document to a model with a bounded context. Chunks overlap so consecutive pieces
share context, which helps retrieval stitch results back together cleanly.

Zero runtime dependencies.

## Install

```bash
pip install llm-token-split
```

## Use

```python
from llm_token_split import TokenSplitter

splitter = TokenSplitter(chunk_size=2000, overlap=200)

chunks = splitter.split(long_document)
for chunk in chunks:
    summary = call_llm(chunk)  # each chunk fits the context window
```

By default token counts use a `len(text) // 4` chars-per-token heuristic. Pass
your own tokenizer (anything that takes a string and returns an `int` count) for
exact budgeting:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
splitter = TokenSplitter(
    chunk_size=8000,
    overlap=400,
    tokenizer=lambda text: len(enc.encode(text)),
)
```

`overlap` must be less than `chunk_size`; otherwise the constructor raises
`ValueError`.

### Position metadata

`split_with_meta` returns `Chunk` objects that carry each chunk's offsets in the
original text, its index, and the total chunk count. The offsets bracket the
source text exactly (`text[chunk.start_char:chunk.end_char] == chunk.text`),
even when the document contains repeated content.

```python
for chunk in splitter.split_with_meta(long_document):
    print(chunk.index, chunk.total, chunk.start_char, chunk.end_char)
```

### Check before splitting

```python
if splitter.fits(text):
    result = call_llm(text)        # no need to split
else:
    results = [call_llm(c) for c in splitter.split(text)]
```

### Chat messages

`split_messages` splits the content of the *last* user message into chunks and
returns one message-list variant per chunk. The input is never mutated; system
and assistant messages (and earlier user turns) are preserved in every variant.

```python
messages = [
    {"role": "system", "content": "You are a careful summarizer."},
    {"role": "user", "content": very_long_text},
]

for variant in splitter.split_messages(messages):
    response = call_llm(variant)
```

If there is no user message, or the last user message already fits, the original
list is returned unchanged as the single variant.

## How it works

- Boundaries are found by binary search over character offsets, so any tokenizer
  works (no assumption that tokens map 1:1 to characters or words).
- Every returned chunk has a token count `<= chunk_size`.
- Consecutive chunks share approximately `overlap` tokens of content.
- An empty string returns `[""]`; text that already fits returns `[text]`.

## License

MIT
