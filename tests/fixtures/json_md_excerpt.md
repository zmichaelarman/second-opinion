## JSON Schema following

JSON that follows a schema provided in your request. Make a request to the Nebius Token Factory API with `{"type": "json_schema"}` in `response_format `and provide desirable schema in a [JSON Schema Specification](https://json-schema.org/specification) compliant format.

> [!TIP]
>
> Many practical tests show that it's always better to provide the schema both in text prompt for the model and`"json_schema"`**parameter**

```python Python expandable theme={null}
import os
import json
from openai import OpenAI
from typing import List, Literal
from pydantic import BaseModel

# 1. Define a schema using Pydantic's `BaseModel`. You can also define 
# a JSON Schema directly; see details after this code example

class Film(BaseModel):
  title: str
  year: int
  director: str
  cast: List[str]
  genre: Literal[
      "drama", "thriller", "sci-fi",
      "comedy", "horror", "fantasy"
  ]

# 2. Add the schema to your request

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

completion = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B",
	response_format={
        	"type": "json_schema"
			"json_schema": Film.model_json_schema()
    	}
    messages=[
        {
            "role": "system",
            "content": (
                "I will give you an actor or actress, and you will respond "
                "with details of a real film they have starred in, according "
                "to the provided structure."
            )
        },
        {
            "role": "user",
            "content": "Jack Nicholson"
        }
    ],

)

# 3. Work with the JSON output or a refusal

output = completion.choices[0].message
if output.refusal:
    # Handle refusal
    print(output.refusal)
elif output.content:
    try:
        output_json = json.loads(output.content)
        print(output_json)
        print("Film: {} ({})".format(output_json['title'], 
                                     output_json['year']))
        # etc.
    except Exception as e:
        # Handle possible exceptions, e.g. invalid JSON
        print(e)
        pass
```

```bash cURL expandable theme={null}
# 1. Define a JSON Schema

export SCHEMA='
{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "year": {"type": "integer"},
    "director": {"type": "string"},
    "cast": {
      "type": "array",
      "items": {"type": "string"}
    },
    "genre": {
      "enum": [
        "drama", "thriller", "sci-fi",
        "comedy", "horror", "fantasy"
      ]
    }
  },
  "required": ["title", "year", "director", "cast", "genre"]
}
'

# 2. Add the schema to your request

export OUTPUT=$(curl 'https://api.tokenfactory.nebius.com/v1/chat/completions' \
  -X 'POST' \
  -H 'Content-Type: application/json' \
  -H 'Accept: */*' \
  -H "Authorization: Bearer $NEBIUS_API_KEY" \
  --data-binary '
    {
      "model": "mistralai/Mistral-Nemo-Instruct-2407",
      "messages": [
        {
          "role": "system",
          "content": "I will give you an actor or actress, and you will respond with details of a real film they have starred in, according to the provided structure."

        },
        {
          "role": "user",
          "content": "Jack Nicholson"
        }
      ],
      "response format": {
        	"type": "json_schema"
			"json_schema": '"$SCHEMA"'
    	}  
    }
  '
)

# 3. Work with the JSON output or a refusal

echo $OUTPUT | jq '
  .choices[0].message.refusal
  // (.choices[0].message.content | fromjson | .title)
'
```
