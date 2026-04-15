#!/usr/bin/env python3
"""
LLM Response Model Benchmark
Tests TTFT, throughput, total latency, and success rate across OpenRouter models.

Usage:
    export OPENROUTER_API_KEY=your-key
    python benchmark_llm.py
    python benchmark_llm.py --models gemini-2.5-flash-lite gemini-2.0-flash-001
    python benchmark_llm.py --rounds 10 --concurrency 2
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime

import httpx

# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------
DEFAULT_MODELS = [
    # --- Baseline ---
    "anthropic/claude-sonnet-4",        # current prod, $15.00/M out

    # --- Same price as qwen3-235b (~$0.10/M out) ---
    "qwen/qwen3-235b-a22b-2507",        # $0.10/M, MoE 235B, strong Chinese
    "z-ai/glm-4-32b",                   # $0.10/M, GLM-4 32B, strong Chinese

    # --- Slightly more expensive but competitive ($0.20-0.40/M out) ---
    "meta-llama/llama-4-scout",         # $0.30/M, MoE, 327k ctx, fast
    "meta-llama/llama-3.3-70b-instruct", # $0.32/M, well-validated chat model
    "bytedance-seed/seed-1.6-flash",    # $0.30/M, ByteDance, 262k ctx
    "google/gemini-2.0-flash-lite-001", # $0.30/M, 1048k ctx, low latency
    "deepseek/deepseek-v3.2",           # $0.38/M, top Chinese model
]

# ---------------------------------------------------------------------------
# Test prompt (mirrors cocomates chat response style)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are Coco, a friendly digital pet companion.
Always respond in the same language as the user's last message.
Keep responses warm and natural, like chatting with a close friend.
Aim for 3-5 sentences per reply — not too short, not too long.
"""

# Prompts designed to elicit ~100-200 token responses for stable TPS measurement
USER_MESSAGES = [
    "今天好累，工作压力好大，感觉什么都不想做，你能陪我聊聊吗？",
    "帮我想一个这个周末放松的计划吧，我想出去走走但又不知道去哪",
    "I've been feeling a bit anxious and overwhelmed lately. Any tips on how to calm down and feel better?",
    "明天要做一个重要的演讲，我特别紧张，怕自己表现不好，你觉得我该怎么准备？",
    "最近睡眠很差，总是睡不着或者半夜醒来，你有什么好的睡前建议吗？",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RoundResult:
    success: bool
    ttft_ms: float = 0.0          # time to first token
    total_ms: float = 0.0         # total response time
    output_tokens: int = 0
    tokens_per_sec: float = 0.0
    error: str = ""


@dataclass
class ModelResult:
    model: str
    rounds: list[RoundResult] = field(default_factory=list)

    @property
    def successful(self) -> list[RoundResult]:
        return [r for r in self.rounds if r.success]

    def summary(self) -> dict:
        ok = self.successful
        total = len(self.rounds)
        if not ok:
            return {
                "model": self.model,
                "success_rate": "0.0%",
                "rounds": f"0/{total}",
                "ttft_p50_ms": "N/A",
                "ttft_p95_ms": "N/A",
                "total_p50_ms": "N/A",
                "total_p95_ms": "N/A",
                "tokens_per_sec": "N/A",
            }
        ttfts = [r.ttft_ms for r in ok]
        totals = [r.total_ms for r in ok]
        tps = [r.tokens_per_sec for r in ok if r.tokens_per_sec > 0]
        return {
            "model": self.model,
            "success_rate": f"{len(ok)/total*100:.1f}%",
            "rounds": f"{len(ok)}/{total}",
            "ttft_p50_ms": f"{_p50(ttfts):.0f}",
            "ttft_p95_ms": f"{_p95(ttfts):.0f}",
            "total_p50_ms": f"{_p50(totals):.0f}",
            "total_p95_ms": f"{_p95(totals):.0f}",
            "tokens_per_sec": f"{_p50(tps):.1f}" if tps else "N/A",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _p50(data: list[float]) -> float:
    return statistics.median(data) if data else 0.0


def _p95(data: list[float]) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = max(0, int(len(s) * 0.95) - 1)
    return s[idx]


# ---------------------------------------------------------------------------
# Core benchmark function (streaming)
# ---------------------------------------------------------------------------
async def benchmark_round(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    api_key: str,
    timeout: float = 30.0,
) -> RoundResult:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://cocomates.app",
        "X-Title": "cocomates-llm-benchmark",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_tokens": 300,
        "temperature": 0.7,
    }

    start = time.perf_counter()
    ttft_ms = 0.0
    output_tokens = 0
    first_token = False

    try:
        async with client.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return RoundResult(success=False, error=f"HTTP {resp.status_code}: {body[:200]}")

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content and not first_token:
                    ttft_ms = (time.perf_counter() - start) * 1000
                    first_token = True
                if content:
                    # rough token count: ~4 chars per token
                    output_tokens += max(1, len(content) // 4)

                # also read usage if provided
                usage = chunk.get("usage", {})
                if usage.get("completion_tokens"):
                    output_tokens = usage["completion_tokens"]

        total_ms = (time.perf_counter() - start) * 1000
        tokens_per_sec = output_tokens / (total_ms / 1000) if total_ms > 0 else 0
        return RoundResult(
            success=True,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            output_tokens=output_tokens,
            tokens_per_sec=tokens_per_sec,
        )

    except httpx.TimeoutException:
        return RoundResult(success=False, error=f"Timeout after {timeout}s")
    except Exception as e:
        return RoundResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# Run all rounds for one model
# ---------------------------------------------------------------------------
async def benchmark_model(
    model: str,
    api_key: str,
    rounds: int,
    concurrency: int,
) -> ModelResult:
    result = ModelResult(model=model)
    prompts = [USER_MESSAGES[i % len(USER_MESSAGES)] for i in range(rounds)]
    sem = asyncio.Semaphore(concurrency)

    async def run_one(prompt: str, idx: int) -> RoundResult:
        async with sem:
            async with httpx.AsyncClient() as client:
                r = await benchmark_round(client, model, prompt, api_key)
            status = "✓" if r.success else f"✗ {r.error[:60]}"
            ttft_str = f"TTFT={r.ttft_ms:.0f}ms" if r.success else ""
            tps_str = f"TPS={r.tokens_per_sec:.1f}" if r.success else ""
            print(f"  [{idx+1:02d}/{rounds}] {status}  {ttft_str}  {tps_str}")
            return r

    short_model = model.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    tasks = [run_one(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    result.rounds = list(results)
    return result


# ---------------------------------------------------------------------------
# Pretty print summary table
# ---------------------------------------------------------------------------
def print_summary(model_results: list[ModelResult]) -> None:
    print(f"\n\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    summaries = [r.summary() for r in model_results]

    cols = [
        ("Model", "model", 40),
        ("OK", "rounds", 7),
        ("Rate", "success_rate", 7),
        ("TTFT p50", "ttft_p50_ms", 10),
        ("TTFT p95", "ttft_p95_ms", 10),
        ("Total p50", "total_p50_ms", 10),
        ("TPS p50", "tokens_per_sec", 9),
    ]

    header = "  ".join(f"{label:<{w}}" for label, _, w in cols)
    print(header)
    print("-" * len(header))
    for s in summaries:
        row = "  ".join(f"{s[key]:<{w}}" for _, key, w in cols)
        print(row)

    print(f"\nAll times in milliseconds. TPS = output tokens/sec (higher = faster).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OpenRouter LLM models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model IDs to benchmark (e.g. google/gemini-2.5-flash-lite)",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Requests per model (default: 5)")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests per model (default: 1)")
    parser.add_argument("--output", type=str, help="Save JSON results to file")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_RESPONSE_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        raise SystemExit(1)

    print(f"Benchmark started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {len(args.models)}  |  Rounds: {args.rounds}  |  Concurrency: {args.concurrency}")

    all_results: list[ModelResult] = []
    for model in args.models:
        result = await benchmark_model(model, api_key, args.rounds, args.concurrency)
        all_results.append(result)

    print_summary(all_results)

    if args.output:
        data = {
            "timestamp": datetime.now().isoformat(),
            "rounds": args.rounds,
            "concurrency": args.concurrency,
            "results": [
                {
                    "model": r.model,
                    "summary": r.summary(),
                    "rounds": [
                        {
                            "success": rr.success,
                            "ttft_ms": rr.ttft_ms,
                            "total_ms": rr.total_ms,
                            "output_tokens": rr.output_tokens,
                            "tokens_per_sec": rr.tokens_per_sec,
                            "error": rr.error,
                        }
                        for rr in r.rounds
                    ],
                }
                for r in all_results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
