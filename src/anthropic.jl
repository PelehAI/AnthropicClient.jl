# HTTP wiring for Anthropic's /v1/messages endpoint.
#
# All API access from this package goes through `_post` which:
# - waits for an RPM slot
# - issues an HTTP POST with the standard Anthropic headers
# - honours `retry-after` on 429
# - retries up to `max_retries` on 429/5xx

const _API_VERSION = "2023-06-01"

# Pure function: build the request body. Tested as a unit.
"""
    build_body(client, system, messages, max_tokens, model, temperature)

Build the JSON body for `/v1/messages`. Returns a `Dict{String,Any}` ready to
hand to JSON3.write. Pure function; no I/O.
"""
function build_body(client::Client;
    system::Union{Nothing, SystemPrompt},
    messages::AbstractVector{Msg},
    max_tokens::Integer,
    model::Union{Nothing, AbstractString},
    temperature::Union{Nothing, Real},
)
    body = Dict{String,Any}(
        "model"      => model === nothing ? client.model_default : String(model),
        "max_tokens" => Int(max_tokens),
        "messages"   => _serialize_messages(messages),
    )
    if system !== nothing
        body["system"] = _serialize_system(system)
    end
    if temperature !== nothing
        body["temperature"] = Float64(temperature)
    end
    return body
end

function _serialize_system(sp::SystemPrompt)
    # Anthropic accepts both a String (no cache) or an array of blocks (which
    # supports cache_control). We always emit the array form so cache markers
    # are uniform.
    block = Dict{String,Any}("type" => "text", "text" => sp.text)
    if sp.cache
        block["cache_control"] = Dict{String,Any}("type" => "ephemeral")
    end
    return [block]
end

function _serialize_messages(messages::AbstractVector{Msg})
    out = Vector{Dict{String,Any}}(undef, length(messages))
    for (i, m) in pairs(messages)
        # Role is validated at Msg construction time, so this is total.
        role = m.role === :user ? "user" : "assistant"
        block = Dict{String,Any}("type" => "text", "text" => m.content)
        if m.cache
            block["cache_control"] = Dict{String,Any}("type" => "ephemeral")
        end
        out[i] = Dict{String,Any}("role" => role, "content" => [block])
    end
    return out
end

# Pure function: turn an Anthropic /v1/messages response (already parsed
# JSON3.Object) into a Reply. Tested as a unit.
"""
    parse_reply(json, model_requested)

Parse Anthropic's response object into a `Reply`. `model_requested` is used
for the cost calculation if the response doesn't echo the model name.
"""
function parse_reply(json, model_requested::AbstractString)
    # Anthropic's response: {id, type, role, model, content: [{type:"text", text:...}, ...],
    #                        stop_reason, usage: {input_tokens, cache_creation_input_tokens,
    #                                              cache_read_input_tokens, output_tokens}}
    model_in_resp = haskey(json, "model") ? String(json["model"]) : String(model_requested)
    content_blocks = get(json, "content", ())
    io = IOBuffer()
    for blk in content_blocks
        if haskey(blk, "type") && blk["type"] == "text"
            write(io, blk["text"])
        end
    end
    text = String(take!(io))
    sr_raw = get(json, "stop_reason", "other")
    stop_reason = sr_raw == "end_turn"      ? :end_turn :
                  sr_raw == "max_tokens"    ? :max_tokens :
                  sr_raw == "stop_sequence" ? :stop_sequence :
                  sr_raw == "tool_use"      ? :tool_use :
                                              :other

    usage = get(json, "usage", Dict{String,Any}())
    input_tokens        = Int(get(usage, "input_tokens", 0))
    cached_read_tokens  = Int(get(usage, "cache_read_input_tokens", 0))
    cached_write_tokens = Int(get(usage, "cache_creation_input_tokens", 0))
    output_tokens       = Int(get(usage, "output_tokens", 0))

    cost_usd = calc_cost(model_in_resp, input_tokens, cached_read_tokens,
                         cached_write_tokens, output_tokens)

    return Reply(text, model_in_resp, stop_reason,
                 input_tokens, cached_read_tokens, cached_write_tokens,
                 output_tokens, cost_usd, json)
end

# ---- HTTP transport --------------------------------------------------------

# HTTP.jl maintains a per-(scheme, host, port) keep-alive pool within a
# process, so consecutive calls to /v1/messages share connections without
# any extra wiring on our side.

"""
    post_messages(client, body; max_retries=3)

Issue one POST to `/v1/messages` with the standard headers. Handles
`retry-after` on 429, retries on 5xx. Returns parsed JSON3.Object.

`body` must already be serialisable JSON (typically the return of
`build_body`).
"""
function post_messages(client::Client, body::AbstractDict; max_retries::Integer=3)
    has_key(client) || throw(ArgumentError("ANTHROPIC_API_KEY not set on client"))
    url = string(client.base_url, "/v1/messages")
    headers = [
        "x-api-key"         => client.api_key,
        "anthropic-version" => _API_VERSION,
        "content-type"      => "application/json",
    ]
    body_str = JSON3.write(body)

    attempt = 0
    while true
        attempt += 1
        await_slot!(client)
        local resp
        try
            resp = HTTP.post(url, headers, body_str;
                             readtimeout=client.timeout, retry=false, status_exception=false)
        catch e
            if attempt > max_retries
                rethrow(e)
            end
            # Network-level errors → backoff + retry.
            sleep(min(2.0 ^ attempt, 30.0))
            continue
        end
        if resp.status == 200
            return JSON3.read(resp.body)
        elseif resp.status == 429
            # Rate limit. Honour retry-after if present, else exponential backoff.
            retry_after = _retry_after_seconds(resp)
            if attempt > max_retries
                throw(AnthropicAPIError(429,
                    "rate limit exceeded after $max_retries retries",
                    String(resp.body)))
            end
            sleep(retry_after)
            continue
        elseif 500 <= resp.status < 600
            if attempt > max_retries
                throw(AnthropicAPIError(resp.status,
                    "server error after $max_retries retries",
                    String(resp.body)))
            end
            sleep(min(2.0 ^ attempt, 30.0))
            continue
        else
            # 4xx that isn't 429 — surface the body, don't retry.
            throw(AnthropicAPIError(resp.status,
                "non-recoverable HTTP error from /v1/messages",
                String(resp.body)))
        end
    end
end

function _retry_after_seconds(resp::HTTP.Response)
    for (k, v) in resp.headers
        lowercase(k) == "retry-after" || continue
        secs = tryparse(Float64, v)
        secs !== nothing && secs > 0 && return secs
    end
    return 5.0  # conservative default
end
