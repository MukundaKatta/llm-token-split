# token-budget-py

[![PyPI](https://img.shields.io/pypi/v/token-budget-py.svg)](https://pypi.org/project/token-budget-py/)
[![Python](https://img.shields.io/pypi/pyversions/token-budget-py.svg)](https://pypi.org/project/token-budget-py/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Thread-safe shared token + USD budget for concurrent LLM tasks.**

Fan-out workloads — agents, parallel summarizers, batch evals — race many
tasks to consume from one shared budget. This library is a small,
zero-dependency counter with two axes (tokens, USD) that returns
`BudgetExceeded` when a record would push past a configured cap.

Sibling to the Rust crate
[`token-budget-pool`](https://crates.io/crates/token-budget-pool).

## Install

```bash
pip install token-budget-py
```

## Use

```python
from token_budget import BudgetPool, BudgetExceeded

pool = BudgetPool(token_cap=1_000_000, usd_cap=10.0)

try:
    pool.record(tokens=1200, usd=0.0036)
except BudgetExceeded as e:
    # tell this worker to skip
    print(f"out of budget: {e}")
```

Two-phase commit (reserve before the LLM call, commit the actual usage):

```python
with pool.reserve(tokens=2000, usd=0.012) as r:
    result = call_llm(prompt)
    r.commit(tokens=result.usage.total_tokens, usd=result.cost_usd)
```

If the `with` block exits without `r.commit()` (e.g. the LLM call raised),
the reservation is auto-released — no orphaned slots.

Either axis is optional:

```python
only_tokens = BudgetPool(token_cap=500_000)        # USD unbounded
only_usd    = BudgetPool(usd_cap=5.0)              # tokens unbounded
unbounded   = BudgetPool()                         # both unbounded (counter only)
```

Atomic read of current state:

```python
snap = pool.snapshot()
snap.tokens_used         # 1200
snap.usd_remaining       # 9.9964
snap.tokens_remaining    # 998800 (cap - used - reserved)
```

## What it does NOT do

- No async runtime lock-in. Works under `asyncio`, `trio`, threads, sync.
  The internal lock is a plain `threading.Lock` (held only for the
  microseconds of a counter update).
- No HTTP. Doesn't talk to any LLM provider.
- No cost calculation. Wrap a cost calculator that returns USD per call
  and feed the result into `record`. (See `claude-cost`, `openai-cost`,
  `gemini-cost`, `bedrock-cost` on crates.io for Rust cost calculators
  with the same authorship.)
- No persistence. Counts live in process. For multi-process / multi-host
  budgets, wrap a Redis or DB increment instead.
- No automatic rollover. Call `pool.reset()` from your own cron / time
  loop if you want a periodic window.

## License

MIT
