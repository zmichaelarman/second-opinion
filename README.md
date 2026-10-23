# Second Opinion

A CLI that checks whether Nebius Token Factory keeps its promises to developers building agents on NVIDIA Nemotron models: that JSON, tool calls, reasoning and streaming come back the way the docs say.

> Work in progress for the Nebius x NVIDIA Global AI Hackathon. The demo link, findings and full write-up land here as milestones complete.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # then set NEBIUS_API_KEY (and NVIDIA_API_KEY for the reference replay)
pytest                 # offline tests only; tests marked `live` are skipped by default
```

## License

Apache-2.0. See [LICENSE](LICENSE).
