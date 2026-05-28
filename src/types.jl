# Core types: Client, Msg, SystemPrompt, Reply, Budget, BudgetExceeded.

"""
    Client(; api_key, model_default, rpm, base_url, timeout)

A reusable client for Anthropic's Messages API. Maintains an HTTP keep-alive
connection pool and a rate-limit semaphore shared across all calls.

Fields:
- `api_key::String`           — from `ANTHROPIC_API_KEY` env by default.
- `model_default::String`     — used when a call doesn't override.
- `rpm::Int`                  — requests-per-minute cap. Anthropic Tier 1 is 5;
                                Tier 2 is 50.
- `base_url::String`          — typically `https://api.anthropic.com`.
- `timeout::Int`              — seconds, per-call HTTP timeout.
- internal: rate-limit window state + lock.
"""
struct Client
    api_key::String
    model_default::String
    rpm::Int
    base_url::String
    timeout::Int
    # RPM sliding window: timestamps of recent calls. Vector is mutable
    # through its reference, so no Ref wrapper needed.
    rpm_window::Vector{Float64}
    rpm_lock::ReentrantLock
end

function Client(;
    api_key::AbstractString       = get(ENV, "ANTHROPIC_API_KEY", ""),
    model_default::AbstractString = "claude-haiku-4-5",
    rpm::Integer                  = 5,
    base_url::AbstractString      = "https://api.anthropic.com",
    timeout::Integer              = 120,
)
    return Client(
        String(api_key),
        String(model_default),
        Int(rpm),
        String(base_url),
        Int(timeout),
        Float64[],
        ReentrantLock(),
    )
end

"Has-key sanity check. Stub mode is when no api_key is set."
has_key(c::Client) = !isempty(c.api_key)

# Custom show — never leak the api_key in a repr or error message.
function Base.show(io::IO, c::Client)
    masked = isempty(c.api_key) ? "<unset>" :
             length(c.api_key) <= 8 ? "***" :
             string(first(c.api_key, 4), "…", last(c.api_key, 4))
    print(io, "AnthropicClient.Client(api_key=", masked,
              ", model_default=", repr(c.model_default),
              ", rpm=", c.rpm, ")")
end

# A message is one (role, content) pair. content can carry an optional cache
# breakpoint (Anthropic's prompt-cache control).
"""
    Msg(role, content; cache=false)

One message in a conversation. `role` is `:user` or `:assistant`. `content` is
a String. `cache=true` marks this message's end as a cache breakpoint
(everything up to and including this message gets cached).
"""
struct Msg
    role::Symbol     # :user or :assistant
    content::String
    cache::Bool
end
Msg(role::Symbol, content::AbstractString; cache::Bool=false) = Msg(role, String(content), cache)

# Sugar so callers can use tuples or Pair where clarity wins.
to_msg(m::Msg) = m
to_msg(p::Tuple{Symbol,<:AbstractString}) = Msg(p[1], p[2])
to_msg(p::Pair{Symbol,<:AbstractString})  = Msg(p[1], p[2])

# System prompt can be plain String, or NamedTuple (text=..., cache=true).
"""
    SystemPrompt(text; cache=false)

System prompt block. `cache=true` adds a cache breakpoint at the end of the
system block, which is the typical place to cache (long reusable instructions
+ schema).
"""
struct SystemPrompt
    text::String
    cache::Bool
end
SystemPrompt(text::AbstractString; cache::Bool=false) = SystemPrompt(String(text), cache)

to_system(::Nothing) = nothing
to_system(sp::SystemPrompt) = sp
to_system(s::AbstractString) = SystemPrompt(String(s), false)
to_system(nt::NamedTuple) = SystemPrompt(String(nt.text), get(nt, :cache, false))

"""
    Reply

Result of one chat call. Numbers come from Anthropic's `usage` field;
`cost_usd` is computed via `pricing.calc_cost`.
"""
struct Reply
    text::String
    model::String
    stop_reason::Symbol             # :end_turn | :max_tokens | :stop_sequence | :tool_use | :other
    input_tokens::Int               # uncached input
    cached_read_tokens::Int         # cache-hit input (cheap)
    cached_write_tokens::Int        # cache-write input (slight surcharge)
    output_tokens::Int
    cost_usd::Float64
    raw::Any                        # full JSON response (JSON3.Object), kept for debugging
end

function Base.show(io::IO, r::Reply)
    snippet = length(r.text) <= 40 ? r.text : string(first(r.text, 37), "...")
    print(io, "Reply(", repr(snippet),
              ", model=", repr(r.model),
              ", in=", r.input_tokens,
              ", out=", r.output_tokens,
              ", \$", round(r.cost_usd; digits=6), ")")
end

"""
    Budget(client; max_usd)

Wrap a client with a per-session spend cap. Calls via `chat(budget; ...)`
deduct from the budget; over-cap throws `BudgetExceeded`.
"""
mutable struct Budget
    client::Client
    max_usd::Float64
    used_usd::Float64
    lock::ReentrantLock
end
Budget(client::Client; max_usd::Real = 1.0) =
    Budget(client, Float64(max_usd), 0.0, ReentrantLock())

spent_usd(b::Budget) = b.used_usd

struct BudgetExceeded <: Exception
    used_usd::Float64
    max_usd::Float64
    attempt_cost_usd::Float64
end
Base.showerror(io::IO, e::BudgetExceeded) = print(io,
    "BudgetExceeded: already spent \$", round(e.used_usd; digits=4),
    ", cap is \$", round(e.max_usd; digits=4),
    ", this call would add \$", round(e.attempt_cost_usd; digits=4))
