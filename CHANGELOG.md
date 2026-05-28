# Changelog

All notable changes to this project will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `AnthropicAPIError <: Exception` — typed exception thrown by `chat`
  on non-recoverable HTTP errors or after retry exhaustion. Carries
  `status`, `message`, and (truncated) `response_body`.
- Module-level docstring for `?AnthropicClient` at the REPL.

### Changed

- Invalid-argument errors now throw `ArgumentError` instead of plain
  `error()`. Affected sites: `chat` (missing `messages`, keyless
  client), `Msg` (unknown role — validated at construction).
- Transport / HTTP error sites now throw `AnthropicAPIError` with the
  status code preserved instead of an opaque `ErrorException`.
- Tests expanded to 192 (was 179), covering the new exception types
  and the early-validation path on `Msg`.

### Removed

- Dead `_HTTP_KEEPALIVE` `Ref`; HTTP.jl already pools connections
  per-host without any extra wiring.

## [0.1.0] - 2026-05-28

Initial public release.

### Added

- `Client` type with `chat` / `chat_async` against Anthropic's
  `/v1/messages`.
- Prompt caching via `cache_control` markers on system and message
  blocks (`SystemPrompt(...; cache=true)`, `Msg(...; cache=true)`).
- Per-client sliding-window RPM semaphore (`rpm` kwarg) shared across
  concurrent calls.
- Per-reply token + USD cost accounting (uncached input / cache read /
  cache write / output) against a bundled per-model price table
  covering current Opus, Sonnet, and Haiku families.
- `Budget(client; max_usd=...)` wrapper that throws `BudgetExceeded`
  on cap.
- `retry-after`-aware 429 handling; bounded exponential backoff on 5xx
  (capped at 30s).
- `known_models()` for discovering the bundled pricing-table entries.
- Custom `Base.show` for `Client` (api_key masked) and `Reply`
  (compact one-liner with cost).
- 179 pure-function unit tests; no live API calls in CI.
