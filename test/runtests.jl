using Test
using AnthropicClient
using AnthropicClient: build_body, parse_reply, calc_cost, _serialize_messages,
                 _serialize_system, Msg, SystemPrompt, to_msg, to_system,
                 await_slot!, _retry_after_seconds
using HTTP
using JSON3

# All tests in this file are pure functions / wiring-only. None hit the
# real Anthropic API. Live tests live in test/live/ (not included in CI).

@testset "AnthropicClient pure-function unit tests" begin

    @testset "Msg normalization" begin
        m1 = to_msg(Msg(:user, "hi"))
        m2 = to_msg((:user, "hi"))
        m3 = to_msg(:user => "hi")
        @test m1.role == :user
        @test m1.content == "hi"
        @test m2.role == :user && m2.content == "hi"
        @test m3.role == :user && m3.content == "hi"
        @test m1.cache == false
    end

    @testset "SystemPrompt normalization" begin
        @test to_system(nothing) === nothing
        sp = to_system("hello")
        @test sp.text == "hello" && sp.cache == false
        sp2 = to_system((text="hi", cache=true))
        @test sp2.text == "hi" && sp2.cache == true
        sp3 = to_system(SystemPrompt("X"; cache=true))
        @test sp3.cache == true
    end

    @testset "Pricing table + calc_cost" begin
        p = AnthropicClient.price_for("claude-haiku-4-5")
        @test p.input == 1.00
        @test p.output == 5.00
        # 1M input + 1M output @ Haiku = $1 + $5 = $6
        c = calc_cost("claude-haiku-4-5", 1_000_000, 0, 0, 1_000_000)
        @test c ≈ 6.0 atol=1e-9
        # 500 input + 200 output @ Haiku = (500*1 + 200*5)/1M = $0.0000015
        c2 = calc_cost("claude-haiku-4-5", 500, 0, 0, 200)
        @test c2 ≈ (500*1.0 + 200*5.0)/1_000_000 atol=1e-12
        # Unknown model falls back to Sonnet — should warn but not error
        c3 = (@test_logs (:warn, r"unknown model") calc_cost("claude-future-2030", 1000, 0, 0, 1000))
        @test c3 ≈ (1000*3.0 + 1000*15.0)/1_000_000 atol=1e-12
    end

    @testset "build_body — simple call" begin
        client = Client(api_key="dummy", model_default="claude-haiku-4-5", rpm=5)
        body = build_body(client;
            system = nothing,
            messages = [Msg(:user, "hi")],
            max_tokens = 64,
            model = nothing,
            temperature = nothing,
        )
        @test body["model"] == "claude-haiku-4-5"
        @test body["max_tokens"] == 64
        @test length(body["messages"]) == 1
        @test body["messages"][1]["role"] == "user"
        @test !haskey(body, "system")
        @test !haskey(body, "temperature")
    end

    @testset "build_body — with system + temperature + model override" begin
        client = Client(api_key="dummy", model_default="claude-haiku-4-5", rpm=5)
        body = build_body(client;
            system = SystemPrompt("You are X."),
            messages = [Msg(:user, "hi"), Msg(:assistant, "hello"), Msg(:user, "again")],
            max_tokens = 100,
            model = "claude-sonnet-4-5",
            temperature = 0.3,
        )
        @test body["model"] == "claude-sonnet-4-5"
        @test body["temperature"] == 0.3
        @test length(body["messages"]) == 3
        @test body["messages"][2]["role"] == "assistant"
        @test haskey(body, "system")
        @test length(body["system"]) == 1
        @test body["system"][1]["type"] == "text"
        @test body["system"][1]["text"] == "You are X."
        @test !haskey(body["system"][1], "cache_control")
    end

    @testset "build_body — cache markers" begin
        client = Client(api_key="dummy")
        body = build_body(client;
            system = SystemPrompt("Huge schema..."; cache=true),
            messages = [Msg(:user, "short")],
            max_tokens = 32,
            model = nothing,
            temperature = nothing,
        )
        @test haskey(body["system"][1], "cache_control")
        @test body["system"][1]["cache_control"]["type"] == "ephemeral"
    end

    @testset "build_body — message-level cache marker" begin
        client = Client(api_key="dummy")
        body = build_body(client;
            system = nothing,
            messages = [Msg(:user, "long-context"; cache=true), Msg(:user, "real-question")],
            max_tokens = 32,
            model = nothing,
            temperature = nothing,
        )
        @test haskey(body["messages"][1]["content"][1], "cache_control")
        @test !haskey(body["messages"][2]["content"][1], "cache_control")
    end

    @testset "build_body — unknown role errors" begin
        client = Client(api_key="dummy")
        @test_throws ErrorException build_body(client;
            system = nothing,
            messages = [Msg(:tool, "x")],
            max_tokens = 32,
            model = nothing,
            temperature = nothing,
        )
    end

    @testset "parse_reply — canonical response shape" begin
        # Simulates an Anthropic /v1/messages response.
        json = JSON3.read("""
        {
          "id": "msg_xxx",
          "type": "message",
          "role": "assistant",
          "model": "claude-haiku-4-5-20251001",
          "content": [{"type":"text","text":"Hello there."}],
          "stop_reason": "end_turn",
          "usage": {
            "input_tokens": 100,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 20
          }
        }
        """)
        rep = parse_reply(json, "claude-haiku-4-5")
        @test rep.text == "Hello there."
        @test rep.stop_reason == :end_turn
        @test rep.input_tokens == 100
        @test rep.output_tokens == 20
        @test rep.cached_read_tokens == 0
        @test rep.cached_write_tokens == 0
        @test rep.model == "claude-haiku-4-5-20251001"
        # cost = (100*1 + 20*5)/1M = 0.0002
        @test rep.cost_usd ≈ (100*1.0 + 20*5.0) / 1_000_000 atol=1e-12
    end

    @testset "parse_reply — cache hits" begin
        json = JSON3.read("""
        {
          "id": "msg_yyy",
          "type": "message",
          "role": "assistant",
          "model": "claude-haiku-4-5",
          "content": [{"type":"text","text":"OK"}],
          "stop_reason": "max_tokens",
          "usage": {
            "input_tokens": 50,
            "cache_creation_input_tokens": 1000,
            "cache_read_input_tokens": 4000,
            "output_tokens": 10
          }
        }
        """)
        rep = parse_reply(json, "claude-haiku-4-5")
        @test rep.cached_read_tokens == 4000
        @test rep.cached_write_tokens == 1000
        @test rep.stop_reason == :max_tokens
        # cost = (50*1 + 4000*0.1 + 1000*1.25 + 10*5) / 1M
        @test rep.cost_usd ≈ (50*1.0 + 4000*0.1 + 1000*1.25 + 10*5.0)/1_000_000 atol=1e-12
    end

    @testset "parse_reply — multi-block content concatenates" begin
        json = JSON3.read("""
        {
          "id": "msg_zzz",
          "type": "message",
          "role": "assistant",
          "model": "claude-haiku-4-5",
          "content": [
            {"type":"text","text":"part one. "},
            {"type":"text","text":"part two."}
          ],
          "stop_reason": "end_turn",
          "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        """)
        rep = parse_reply(json, "claude-haiku-4-5")
        @test rep.text == "part one. part two."
    end

    @testset "RPM semaphore — under cap is instant" begin
        client = Client(api_key="dummy", rpm=50)
        t0 = time()
        for _ in 1:5
            await_slot!(client)
        end
        @test time() - t0 < 0.5
    end

    @testset "RPM semaphore — over cap blocks" begin
        # Tight cap, then issue 1 more than cap — last should block.
        client = Client(api_key="dummy", rpm=2)
        # Fill the window with two "old" timestamps then check that the
        # next slot waits. We can't easily fast-forward time, so just check
        # that the window state evolves correctly under instant calls:
        await_slot!(client)
        await_slot!(client)
        @test length(client.rpm_window[]) == 2
        # A 3rd call would block ~60s. Don't actually wait; test the state
        # logic by manually evicting one entry and checking it proceeds.
        client.rpm_window[][1] = time() - 100  # mark old
        await_slot!(client)                    # should evict + proceed
        @test length(client.rpm_window[]) == 2
    end

    @testset "Budget — under cap accepts" begin
        client = Client(api_key="dummy")
        b = Budget(client; max_usd=1.0)
        @test spent_usd(b) == 0.0
        @test b.max_usd == 1.0
    end

    @testset "Budget — pre-call check throws when already at cap" begin
        client = Client(api_key="dummy")
        b = Budget(client; max_usd=0.0)
        @test_throws BudgetExceeded chat(b; messages=[(:user, "x")], max_tokens=32)
    end

    @testset "has_key — keyless client returns false" begin
        @test !has_key(Client(api_key=""))
        @test  has_key(Client(api_key="sk-ant-anything"))
    end

    @testset "chat — keyless client errors" begin
        client = Client(api_key="")
        @test_throws ErrorException chat(client; messages=[(:user, "x")], max_tokens=8)
    end

    @testset "chat — missing messages errors" begin
        client = Client(api_key="dummy")
        @test_throws ErrorException chat(client; max_tokens=8)
    end

    @testset "parse_reply — stop_reason variants" begin
        for (raw, sym) in [("end_turn",      :end_turn),
                           ("max_tokens",    :max_tokens),
                           ("stop_sequence", :stop_sequence),
                           ("tool_use",      :tool_use),
                           ("refusal",       :other),       # unknown → :other
                           ("anything-else", :other)]
            json = JSON3.read("""
            {"model":"claude-haiku-4-5","content":[{"type":"text","text":""}],
             "stop_reason":"$raw","usage":{"input_tokens":1,"output_tokens":1}}
            """)
            @test parse_reply(json, "claude-haiku-4-5").stop_reason === sym
        end
    end

    @testset "parse_reply — missing model field falls back to requested" begin
        json = JSON3.read("""
        {"content":[{"type":"text","text":"x"}],"stop_reason":"end_turn",
         "usage":{"input_tokens":1,"output_tokens":1}}
        """)
        rep = parse_reply(json, "claude-haiku-4-5")
        @test rep.model == "claude-haiku-4-5"
    end

    @testset "_retry_after_seconds — parses header" begin
        # Synthesise a Response with retry-after header.
        resp = HTTP.Response(429, ["retry-after" => "12"]; body=UInt8[])
        @test _retry_after_seconds(resp) ≈ 12.0

        # Case-insensitive lookup.
        resp2 = HTTP.Response(429, ["Retry-After" => "3.5"]; body=UInt8[])
        @test _retry_after_seconds(resp2) ≈ 3.5

        # Missing header → conservative default 5.0.
        resp3 = HTTP.Response(429, Pair{String,String}[]; body=UInt8[])
        @test _retry_after_seconds(resp3) == 5.0

        # Non-numeric / zero → fall back to default.
        resp4 = HTTP.Response(429, ["retry-after" => "garbage"]; body=UInt8[])
        @test _retry_after_seconds(resp4) == 5.0
        resp5 = HTTP.Response(429, ["retry-after" => "0"]; body=UInt8[])
        @test _retry_after_seconds(resp5) == 5.0
    end

    @testset "BudgetExceeded — showerror renders dollars" begin
        e = BudgetExceeded(0.123, 0.10, 0.045)
        io = IOBuffer()
        showerror(io, e)
        msg = String(take!(io))
        @test occursin("0.123", msg)
        @test occursin("0.1", msg)
        @test occursin("0.045", msg)
        @test occursin("BudgetExceeded", msg)
    end

    @testset "Client — show never leaks api_key" begin
        c1 = Client(api_key="sk-ant-supersecret-DEADBEEF", rpm=5)
        io = IOBuffer(); show(io, c1)
        s1 = String(take!(io))
        @test !occursin("supersecret", s1)
        @test !occursin("DEADBEEF", s1)
        @test occursin("sk-a", s1)             # first 4
        @test occursin("BEEF", s1)             # last 4
        @test occursin("rpm=5", s1)

        # Short key still masked.
        c2 = Client(api_key="abc", rpm=5)
        io = IOBuffer(); show(io, c2)
        s2 = String(take!(io))
        @test !occursin("abc", s2)
        @test occursin("***", s2)

        # Empty key shows <unset>.
        c3 = Client(api_key="", rpm=5)
        io = IOBuffer(); show(io, c3)
        s3 = String(take!(io))
        @test occursin("<unset>", s3)
    end

    @testset "Reply — show is compact and includes cost" begin
        rep = Reply("hello", "claude-haiku-4-5", :end_turn, 100, 0, 0, 20, 0.0002, nothing)
        io = IOBuffer(); show(io, rep)
        s = String(take!(io))
        @test occursin("hello", s)
        @test occursin("claude-haiku-4-5", s)
        @test occursin("in=100", s)
        @test occursin("out=20", s)
        @test occursin("0.0002", s)

        # Long text gets truncated.
        long = "x" ^ 200
        rep2 = Reply(long, "m", :end_turn, 1, 0, 0, 1, 0.0, nothing)
        io = IOBuffer(); show(io, rep2)
        s2 = String(take!(io))
        @test length(s2) < 200
        @test occursin("...", s2)
    end

    @testset "parse_reply — missing usage block defaults to zero tokens" begin
        json = JSON3.read("""
        {"model":"claude-haiku-4-5","content":[{"type":"text","text":"x"}],
         "stop_reason":"end_turn"}
        """)
        rep = parse_reply(json, "claude-haiku-4-5")
        @test rep.input_tokens == 0
        @test rep.output_tokens == 0
        @test rep.cached_read_tokens == 0
        @test rep.cached_write_tokens == 0
        @test rep.cost_usd == 0.0
    end

    @testset "parse_reply — non-text content blocks are skipped" begin
        json = JSON3.read("""
        {"model":"claude-haiku-4-5",
         "content":[
            {"type":"text","text":"answer is "},
            {"type":"tool_use","id":"t1","name":"calc","input":{}},
            {"type":"text","text":"42"}
         ],
         "stop_reason":"tool_use",
         "usage":{"input_tokens":10,"output_tokens":5}}
        """)
        rep = parse_reply(json, "claude-haiku-4-5")
        @test rep.text == "answer is 42"
        @test rep.stop_reason == :tool_use
    end

    @testset "known_models — sorted, non-empty, includes our defaults" begin
        ms = known_models()
        @test ms isa Vector{String}
        @test issorted(ms)
        @test "claude-haiku-4-5" in ms
        @test "claude-sonnet-4-5" in ms
        @test "claude-opus-4-7" in ms
        # Every entry must have a price entry (round-trip).
        for m in ms
            p = price_for(m)
            @test p.input > 0
            @test p.output > 0
        end
    end

    @testset "chat_async — returns a Task without making a call" begin
        # Keyless client → chat throws. chat_async still returns a Task; the
        # error surfaces when you fetch.
        client = Client(api_key="")
        t = chat_async(client; messages=[(:user, "x")], max_tokens=8)
        @test t isa Task
        @test_throws TaskFailedException fetch(t)
    end

    @testset "Two clients have independent RPM windows" begin
        c1 = Client(api_key="dummy", rpm=2)
        c2 = Client(api_key="dummy", rpm=2)
        await_slot!(c1)
        await_slot!(c1)
        @test length(c1.rpm_window[]) == 2
        @test length(c2.rpm_window[]) == 0
        await_slot!(c2)
        @test length(c1.rpm_window[]) == 2
        @test length(c2.rpm_window[]) == 1
    end

end
