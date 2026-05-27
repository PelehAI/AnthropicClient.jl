# LLMClient.jl

Minimal, fast Julia client for the Anthropic API. Built for Peleh's semantic
pipeline; designed so it can be spun out as a standalone OSS package later.

## Goals

- One concrete `AnthropicClient` type, no provider abstraction yet.
- `chat` / `chat_async` against Anthropic's `/v1/messages`.
- HTTP keep-alive connection pool (via HTTP.jl).
- Prompt caching (Anthropic's `cache_control` markers) — the big perf lever
  for system-prompt-heavy workloads.
- RPM-aware shared semaphore for parallel calls.
- Cost ledger per reply (`input_tokens`, `cached_read_tokens`,
  `cached_write_tokens`, `output_tokens`, `cost_usd`).
- `Budget` wrapper that throws on cap exceeded.
- Batch API (24h SLA, 50% off) — for non-realtime workloads.
- Stub-friendly testing: no live calls in CI, body-building and
  reply-parsing are pure functions.

## Non-goals (deliberately)

- No multi-provider abstraction. If we add OpenAI / Gemini later we'll
  extract one then.
- No streaming in v0.1 (request/response is enough for the use case).
- No JuliaHub registration yet; path dep from peleh-app/engine/.

## Usage

```julia
using LLMClient

client = AnthropicClient(
    api_key       = ENV["ANTHROPIC_API_KEY"],
    model_default = "claude-haiku-4-5",
    rpm           = 5,
)

reply = chat(client;
    system     = "You are a helpful assistant.",
    messages   = [(:user, "Say hi.")],
    max_tokens = 64,
)
@show reply.text reply.cost_usd

# Async — many tasks share the per-client RPM semaphore
tasks   = [chat_async(client; messages=[(:user, "Q$i")], max_tokens=32) for i in 1:5]
replies = fetch.(tasks)

# Budget cap
budget = Budget(client; max_usd = 0.05)
reply  = chat(budget; messages=[(:user, "Hi")], max_tokens=32)  # throws BudgetExceeded if over
spent_usd(budget)
```

## License

MIT. See `LICENSE`.
