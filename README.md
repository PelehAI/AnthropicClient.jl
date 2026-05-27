# LLMClient.jl

Minimal, fast Julia client for the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages).
One concrete `AnthropicClient`, no provider abstraction. Designed for
long-running batch and pipeline workloads where prompt caching, rate
limiting, and cost accounting matter.

## Features

- `chat` / `chat_async` against `/v1/messages` with HTTP keep-alive pooling.
- Prompt caching via Anthropic's `cache_control` markers — large surcharge
  on first write, ~10% of input price on subsequent reads. The big perf
  lever for system-prompt-heavy workloads.
- Per-client sliding-window RPM semaphore shared across concurrent calls.
- Per-reply token + USD cost accounting (uncached input, cache reads, cache
  writes, output).
- `Budget` wrapper that throws `BudgetExceeded` on cap.
- `retry-after`-aware 429 handling; exponential backoff on 5xx.
- Stub-friendly: body-building and reply-parsing are pure functions, so
  tests run with no network and no API key.

## Status

Early. Used in production for one workload (see "Used by" below) but the
API surface may still shift. Not yet registered in Julia's General
registry — use it as a path or git dependency.

## Install

Git dependency (recommended while pre-1.0):

```julia
using Pkg
Pkg.add(url="https://github.com/PelehAI/LLMClient.jl")
```

Or as a path dependency in your `Project.toml`:

```toml
[deps]
LLMClient = "e82deec3-d3b5-4b85-aba1-b8e1f470db1f"

[sources]
LLMClient = {path = "../LLMClient.jl"}
```

Set your API key in the environment:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick start

```julia
using LLMClient

client = AnthropicClient(
    api_key       = ENV["ANTHROPIC_API_KEY"],
    model_default = "claude-haiku-4-5",
    rpm           = 5,            # Anthropic Tier 1
)

reply = chat(client;
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

Mark the end of any block (system, user message) with a cache breakpoint.
Everything up to and including that block is cached:

```julia
sys = SystemPrompt("""
    <huge schema, instructions, few-shot examples...>
""", cache=true)                  # cache the whole system block

reply = chat(client;
    system     = sys,
    messages   = [(:user, "actual question")],
    max_tokens = 256,
)
@show reply.cached_read_tokens    # >0 on subsequent calls
@show reply.cached_write_tokens   # >0 on the first call (the write)
```

You can also cache mid-conversation with message-level markers:

```julia
chat(client;
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
tasks   = [chat_async(client; messages=[(:user, "Q$i")], max_tokens=32) for i in 1:20]
replies = fetch.(tasks)
```

On Tier 1 (`rpm=5`) this serialises 20 calls into 4 minutes. Tier 2
(`rpm=50`) lets them complete in seconds.

## Cost accounting + budgets

Each `Reply` carries token counts and USD cost computed against a
per-model price table (see `src/pricing.jl`; update as new models ship).

```julia
budget = Budget(client; max_usd = 0.10)

for prompt in prompts
    reply = chat(budget; messages=[(:user, prompt)], max_tokens=128)
    # raises BudgetExceeded once spent_usd(budget) crosses max_usd
end

@show spent_usd(budget)
```

`Budget` enforces the cap *post-hoc* on the call that crosses it: that
call's reply is recorded, and the *next* call refuses. Concurrent calls
that all cross simultaneously may each get one reply through; protect
against that with a tighter cap or external coordination if it matters.

## Stub mode (no API key)

```julia
client = AnthropicClient(api_key="", rpm=5)
has_key(client)                   # false
```

Library code can degrade to identity passes / placeholders without a key.
Calling `chat` on a keyless client throws — guard with `has_key`.

## Non-goals

- **Multi-provider abstraction.** If OpenAI / Gemini are wanted, extract
  the right abstraction *then*, not now.
- **Streaming.** Request/response only for now.
- **Tool use / function calling.** Add when needed.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

All tests are pure-function / wiring-only — no live API calls.

## Used by

- [peleh.ai](https://peleh.ai) — academic paper to slide deck.

## License

MIT. See `LICENSE`.
