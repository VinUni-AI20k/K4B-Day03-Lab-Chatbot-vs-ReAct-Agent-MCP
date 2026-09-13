# MoodMix – Music Discovery and Playlist Export Agent

- **Full name:** Bui Duc Thanh
- **Student ID:** 2A202602364
- **Selected topic:** MoodMix – Music Discovery and Playlist Export Agent

## Agentic Fit scoring

| Criterion | Score | Explanation |
| --- | ---: | --- |
| Multi-step Reasoning | 4/5 | Export requests require a search result before selecting track IDs and exporting. |
| Tool Interaction | 5/5 | The agent uses iTunes search and a CSV export tool through the MCP-style dispatcher. |
| Dynamic Decision | 4/5 | The export decision uses IDs returned in the latest search Observation. |
| Long Horizon Goal | 3/5 | The workflow is short, but maintains the request across search, export, and final response. |
| **Total** | **16/20** | MoodMix is a good fit for a small ReAct workflow. |

## Trace note

The offline mock verifies the expected TC04 sequence: `search_tracks → export_playlist → final answer`.
No live Gemini or OpenAI trace is claimed here. Before final submission, the student must configure a real API key in `.env` and run `python src/app.py --all` to capture a live-provider trace.
