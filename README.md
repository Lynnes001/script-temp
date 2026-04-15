# script-temp

Temporary scripts for testing and benchmarking. Managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter the repo
git clone https://github.com/Lynnes001/script-temp.git
cd script-temp

# Install dependencies (auto-creates .venv)
uv sync
```

## Scripts

### benchmark_llm.py

Benchmarks OpenRouter LLM models for response quality selection. Measures TTFT (time to first token), throughput (tokens/sec), total latency, and success rate across multiple models.

**Requirements:** Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY=your-key
```

**Run with defaults** (tests 5 candidate models, 5 rounds each):

```bash
uv run benchmark_llm.py
```

**Common options:**

```bash
# Test specific models only
uv run benchmark_llm.py --models google/gemini-2.5-flash-lite qwen/qwen3-235b-a22b-2507 anthropic/claude-sonnet-4

# More rounds for higher confidence
uv run benchmark_llm.py --rounds 10

# Concurrent requests (stress test)
uv run benchmark_llm.py --rounds 10 --concurrency 2

# Save results to JSON
uv run benchmark_llm.py --rounds 10 --output results.json
```

**Example output:**

```
BENCHMARK SUMMARY
================================================================================
Model                                                    OK       Rate    TTFT p50    TTFT p95   Total p50     TPS p50
---------------------------------------------------------------------------------------------------------------
google/gemini-2.5-flash-lite                            5/5    100.0%         312         489        1823        78.3
qwen/qwen3-235b-a22b-2507                               5/5    100.0%         521         703        2104        61.2
anthropic/claude-sonnet-4                               5/5    100.0%        1243        1891        4521        42.1
```

All times in milliseconds. TPS = output tokens/sec (higher = faster).
