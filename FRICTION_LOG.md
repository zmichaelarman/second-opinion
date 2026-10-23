# Friction log

Friction hit with Nebius Token Factory's API or docs while building Second Opinion. It feeds `FEEDBACK_DRAFT.md` and the Devpost feedback.

| Time (ET) | Step | Expected | What happened | Minutes lost | Severity |
|---|---|---|---|---|---|
| 2026-10-23 11:13 | Finding how to switch Nemotron reasoning off | A documented request field for the chat-template switch (`chat_template_kwargs` / `enable_thinking`), as in NVIDIA's and vLLM's docs | `openapi.json` (version `20261022-1d60f32ca`) has no `chat_template_kwargs`. Instead it lists `extra_body` as a body field, though in the OpenAI SDK that's a client-side option, plus a `reasoning_effort` enum. It doesn't say which one controls Nemotron thinking. | 5 | Medium |
| 2026-10-23 11:13 | Checking allowed `tool_choice` values | The prose docs and the OpenAPI spec agree | `function-calling.md` lists `"auto"` and a named function. `openapi.json` also accepts `"none"`, `"required"` and null. | 2 | Low |
| 2026-10-23 11:20 | Looking for docs on Nemotron reasoning control | A docs page on reasoning models: how to turn thinking on/off and where reasoning text is returned | `llms.txt` has no page mentioning reasoning or thinking. Reasoning control and the reasoning response fields are undocumented outside the raw OpenAPI schema. | 3 | Medium |
