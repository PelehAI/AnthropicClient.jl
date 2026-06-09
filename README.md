# AnthropicClient.jl

[![CI](https://github.com/PelehAI/AnthropicClient.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/PelehAI/AnthropicClient.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Julia client for [Anthropic's Messages API](https://docs.anthropic.com/en/api/messages).
Anthropic-only, by design. Built for long-running batch and pipeline
workloads where prompt caching, rate limiting, and cost accounting are
the things that matter.

## Features

- `chat` / `chat_async` against `/v1/messages` with HTTP keep-alive
  pooling.
- Prompt caching via Anthropic's `cache_control` markers — large
  surcharge on first write, ~10% of input price on subsequent reads.
  The big perf lever for system-prompt-heavy workloads.
- Per-client sliding-window RPM semaphore shared across concurrent calls.
- Per-reply token + USD cost accounting (uncached input, cache reads,
  cache writes, output) against a bundled per-model price table.
- `Budget` wrapper that throws `BudgetExceeded` on cap.
- `retry-after`-aware 429 handling; bounded exponential backoff on 5xx.
- Stub-friendly: body-building and reply-parsing are pure functions, so
  tests run with no network and no API key.
- `Base.show` never prints the API key.

## Install

While pre-1.0, use as a git dependency:

```julia
using Pkg
Pkg.add(url="https://github.com/PelehAI/AnthropicClient.jl")
```

Or pin in your `Project.toml`:

```toml
[deps]
AnthropicClient = "e82deec3-d3b5-4b85-aba1-b8e1f470db1f"
```

Set your API key in the environment:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick start

```julia
using AnthropicClient

c = Client(
    api_key       = ENV["ANTHROPIC_API_KEY"],
    model_default = "claude-haiku-4-5",
    rpm           = 5,            # Anthropic Tier 1
)

reply = chat(c;
    system     = "You are a helpful assistant.",
    messages   = [(:user, "Say hi.")],
    max_tokens = 64,
)
@show reply.text reply.cost_usd reply.input_tokens reply.output_tokens
```

`messages` accepts `Msg`, `(:user, "...")` tuples, or `:user => "..."`
pairs. `system` accepts `String`, `SystemPrompt(text; cache=true)`, or
`(text="...", cache=true)`.

## Prompt caching

Mark the end of any block (system or message) as a cache breakpoint.
Everything up to and including that block is cached on the next call:

```julia
sys = SystemPrompt("""
    <huge schema, instructions, few-shot examples...>
""", cache=true)                  # cache the whole system block

reply = chat(c;
    system     = sys,
    messages   = [(:user, "actual question")],
    max_tokens = 256,
)
@show reply.cached_read_tokens    # > 0 on subsequent calls
@show reply.cached_write_tokens   # > 0 on the first call (the write)
```

Mid-conversation caching with message-level markers:

```julia
chat(c;
    messages = [
        Msg(:user, large_context_blob; cache=true),
        Msg(:user, real_question),
    ],
    max_tokens = 256,
)
```

## Concurrency + RPM throttling

`chat_async` returns a `Task` that runs on a background thread. Many
concurrent tasks share one rate budget — the per-client sliding-window
semaphore will block tasks that would exceed `rpm` requests in the
trailing 60s.

```julia
tasks   = [chat_async(c; messages=[(:user, "Q$i")], max_tokens=32) for i in 1:20]
replies = fetch.(tasks)
```

On Tier 1 (`rpm=5`) this serialises 20 calls into 4 minutes. Tier 2
(`rpm=50`) finishes in seconds.

## Cost accounting + budgets

Each `Reply` carries token counts and a USD cost computed against the
bundled price table. Use `known_models()` to list what's billable
without hitting the warn-fallback path; update `src/pricing.jl` when
new models ship.

```julia
budget = Budget(c; max_usd = 0.10)

for prompt in prompts
    reply = chat(budget; messages=[(:user, prompt)], max_tokens=128)
    # raises BudgetExceeded once spent_usd(budget) crosses max_usd
end

@show spent_usd(budget)
```

`Budget` enforces the cap *post-hoc* on the call that crosses it: that
call's reply is recorded, and the *next* call refuses. Concurrent calls
that all cross simultaneously may each get one reply through — protect
against that with a tighter cap or external coordination if it matters.

## Stub mode (no API key)

```julia
c = Client(api_key="", rpm=5)
has_key(c)                        # false
```

Library code can degrade to identity passes / placeholders without a
key. Calling `chat` on a keyless client throws — guard with `has_key`.

## Health & speed probes

`has_key` only tells you a key string is *set*, not that it works. Two live
probes go further — both make minimal real calls (a few output tokens) and
never throw:

```julia
hc = healthcheck(c)              # one minimal call, classified
hc.ok, hc.status                 # e.g. (true, :ok) or (false, :billing)

sp = speedtest(c; n = 5)         # n concurrent calls under the rpm cap
sp.throughput_rps, sp.latency_median_ms
```

`healthcheck` returns a `HealthStatus` whose `status` is one of `:ok`,
`:no_key`, `:auth`, `:quota`, `:billing`, `:bad_request`, `:server`,
`:network`, `:error` — enough for a dashboard to show green/red and say *why*.
`speedtest` returns a `SpeedResult` (ok / rate-limited / failed counts, achieved
`throughput_rps`, and min/median/max latency). Both short-circuit on a keyless
client.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

All tests are pure-function / wiring-only — no live API calls.

## Roadmap

Tracked as GitHub issues — contributions welcome:

- Streaming (SSE) responses
- Tool use / function calling
- Vision inputs (image content blocks)
- Document inputs (PDF)
- Batches API (24h SLA, 50% off list price)
- `count_tokens` endpoint
- Extended thinking blocks
- AWS Bedrock + Vertex AI transports

## Used by

- [peleh.ai](https://peleh.ai) — academic paper to slide deck.

## License

MIT. See `LICENSE`.
