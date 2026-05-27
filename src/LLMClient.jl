module LLMClient

# LLMClient.jl — minimal, fast Julia client for Anthropic's Messages API.
# See README.md for the design rationale.

using HTTP
using JSON3

include("pricing.jl")
include("types.jl")
include("concurrency.jl")
include("anthropic.jl")

# ------------------- Public API ---------------------------------------------

export AnthropicClient, has_key,
       Msg, SystemPrompt,
       Reply,
       Budget, BudgetExceeded, spent_usd,
       chat, chat_async,
       calc_cost, price_for, known_models

"""
    chat(client; system=nothing, messages, max_tokens, model=nothing,
                 temperature=nothing, max_retries=3) -> Reply

Make one Anthropic Messages API call. Returns a `Reply` with text, token
counts, and computed `cost_usd`.

# Arguments
- `system` — `nothing`, a `String`, a `SystemPrompt(text; cache=true)`, or a
  `NamedTuple` `(text="...", cache=true)` for inline construction.
- `messages` — vector of `Msg`, or `(:user, "...")` tuples, or
  `:user => "..."` pairs. Roles must be `:user` or `:assistant`.
- `max_tokens` — required.
- `model` — defaults to `client.model_default`.
- `temperature` — optional Float; Anthropic default is 1.0.
- `max_retries` — for 429 / 5xx retries; default 3.
"""
function chat(client::AnthropicClient;
    system   = nothing,
    messages = nothing,
    max_tokens::Integer,
    model::Union{Nothing, AbstractString} = nothing,
    temperature::Union{Nothing, Real}     = nothing,
    max_retries::Integer = 3,
)
    messages === nothing && error("LLMClient.chat: `messages` is required")
    msgs_normalized = Msg[to_msg(m) for m in messages]
    sys_normalized  = to_system(system)
    body = build_body(client;
        system = sys_normalized,
        messages = msgs_normalized,
        max_tokens = max_tokens,
        model = model,
        temperature = temperature,
    )
    json = post_messages(client, body; max_retries=max_retries)
    return parse_reply(json, model === nothing ? client.model_default : String(model))
end

"""
    chat_async(client; kwargs...) -> Task{Reply}

Returns a Task running `chat` on a background thread. The task respects the
client's RPM cap through the shared semaphore — many concurrent tasks share
one rate budget.
"""
function chat_async(client::AnthropicClient; kwargs...)
    Threads.@spawn chat(client; kwargs...)
end

# Budget wrapper methods are defined in budget.jl, which is included AFTER
# `chat(client; ...)` is in scope above.
include("budget.jl")

end # module
