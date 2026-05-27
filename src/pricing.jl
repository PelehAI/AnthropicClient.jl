# Anthropic per-model pricing (USD per million tokens).
# Update as new models ship. Numbers from anthropic.com/pricing as of
# 2026-05.

const _PRICES = Dict{String, NamedTuple{(:input, :cache_read, :cache_write, :output), NTuple{4, Float64}}}(
    # Opus 4 family
    "claude-opus-4-5"            => (input=15.00, cache_read=1.50, cache_write=18.75, output=75.00),
    "claude-opus-4-5-20251101"   => (input=15.00, cache_read=1.50, cache_write=18.75, output=75.00),
    "claude-opus-4-7"            => (input=15.00, cache_read=1.50, cache_write=18.75, output=75.00),

    # Sonnet 4 family
    "claude-sonnet-4-5"          => (input=3.00,  cache_read=0.30, cache_write=3.75,  output=15.00),
    "claude-sonnet-4-5-20250929" => (input=3.00,  cache_read=0.30, cache_write=3.75,  output=15.00),
    "claude-sonnet-4-6"          => (input=3.00,  cache_read=0.30, cache_write=3.75,  output=15.00),

    # Haiku 4 family
    "claude-haiku-4-5"            => (input=1.00, cache_read=0.10, cache_write=1.25, output=5.00),
    "claude-haiku-4-5-20251001"   => (input=1.00, cache_read=0.10, cache_write=1.25, output=5.00),
)

"""
    price_for(model) -> NamedTuple

Returns the per-million-token rates for `model`. Falls back to a conservative
default (Sonnet pricing) if the exact model isn't in the table — the user can
update `_PRICES` when a new model ships.
"""
function price_for(model::AbstractString)
    if haskey(_PRICES, model)
        return _PRICES[model]
    end
    # Conservative fallback: Sonnet-class. Logged once-ish via @warn.
    @warn "AnthropicClient: unknown model in pricing table; using Sonnet rates as fallback" model
    return _PRICES["claude-sonnet-4-5"]
end

"""
    calc_cost(model, input_tokens, cached_read, cached_write, output_tokens) -> Float64

USD cost for a single Anthropic call. All token counts are integers from the
API response's `usage` block.
"""
function calc_cost(model::AbstractString, input::Integer, cached_read::Integer,
                   cached_write::Integer, output::Integer)
    p = price_for(model)
    # input_tokens in the API response excludes cache_read + cache_write
    # (Anthropic reports them separately). So uncached_input = input.
    return (input        * p.input        +
            cached_read  * p.cache_read   +
            cached_write * p.cache_write  +
            output       * p.output) / 1_000_000
end

"""
    known_models() -> Vector{String}

Sorted list of every model identifier in the bundled pricing table. Use
this to discover what `model_default` values are billable without falling
back to the Sonnet default.
"""
known_models() = sort!(collect(keys(_PRICES)))
