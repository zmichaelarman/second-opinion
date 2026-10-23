# Friction Log

Timestamped record of developer experience friction encountered while building and running Second Opinion against Nebius Token Factory and NVIDIA Nemotron endpoints.

| Time (ET) | Step | Expected | What happened | Minutes lost | Severity |
|---|---|---|---|---|---|
| 2026-10-23 11:20 | Docs inspection (`ai-models-inference/json.md`) | Copy-pasteable Python & cURL structured output examples | Python example is missing commas in/after `response_format` (`SyntaxError` at line 31); cURL example uses `"response format"` (with a space) and misses a comma (`CURL_JSON_INVALID`); bare vs wrapped `json_schema` shapes are mixed across examples | 15 | S2 |
| 2026-10-23 12:02 | Docs static check (`api-reference/examples/text-generation.md`) | Valid cURL JSON body and JS snippet | cURL snippet has invalid JSON (`Expecting property name enclosed in double quotes` at line 31); JS snippet contains literal `{"properties": {...},}` ellipsis syntax causing `node --check` `SyntaxError: Unexpected token '}'` | 10 | S4 |
| 2026-10-23 12:02 | Docs static check (`api-reference/examples/vision-capabilities.md`) | Valid JS snippet | JS snippet contains literal `{"properties": {...},}` ellipsis syntax causing `node --check` `SyntaxError: Unexpected token '}'` | 5 | S4 |
| 2026-10-23 12:02 | Docs static check (`api-reference/examples/text-generation-for-code-autocomplete.md`) | Valid cURL JSON body | cURL snippet is missing a comma after line 2 in the JSON payload (`Expecting ',' delimiter: line 3 column 5`) | 5 | S4 |
