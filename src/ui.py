"""MoodMix web server: a small FastAPI shell around the existing ReAct tools."""

import asyncio
import json
import re
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.append(str(Path(__file__).parent))
from mcp_server import MCPAcademicServer
from prompts import MAX_ITERATIONS, REACT_AGENT_SYSTEM_PROMPT
from providers import get_llm_provider

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
EXPORT_PATH = ROOT_DIR / "docs" / "playlist_export.csv"
SESSIONS = {}

app = FastAPI(title="MoodMix")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def event(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def summary(result):
    if result.get("status") != "SUCCESS":
        return result.get("message", "The request could not be completed.")
    if "playlist_name" in result:
        return f"Exported {result.get('track_count', 0)} tracks to a CSV file."
    tracks = result.get("tracks", [])
    names = ", ".join(track.get("title", "") for track in tracks[:3])
    return f"Found {len(tracks)} tracks" + (f": {names}." if names else ".")


def merge_tracks(existing, incoming):
    tracks = {track["track_id"]: track for track in existing if track.get("track_id")}
    tracks.update({track["track_id"]: track for track in incoming if track.get("track_id")})
    return list(tracks.values())


def visible_text(text):
    return re.sub(r"itunes|apple music", "music catalog", text, flags=re.IGNORECASE)


async def stream_chat(message, session):
    provider, server = get_llm_provider(), MCPAcademicServer()
    context = "\n\n".join(session["history"][-6:] + [f"User: {message}"])
    final_context = f"User: {message}"
    yield event({"type": "status", "content": "Understanding your request"})
    yield event({"type": "trace", "stage": "thought", "content": "Reviewing the music request"})
    final_text = ""

    try:
        for _ in range(MAX_ITERATIONS):
            decision = provider.generate_with_tools(context, server.list_tools(), REACT_AGENT_SYSTEM_PROMPT)
            if decision.get("type") != "tool_call":
                final_prompt = final_context + "\nProvide the final helpful answer now. Do not call a tool."
                yield event({"type": "trace", "stage": "result", "content": "Preparing the response"})
                for token in provider.generate_stream(final_prompt, REACT_AGENT_SYSTEM_PROMPT):
                    token = visible_text(token)
                    final_text += token
                    yield event({"type": "token", "content": token})
                    await asyncio.sleep(0)
                break

            tool_name = decision.get("tool_name", "")
            arguments = decision.get("arguments", {})
            action = "Searching the music catalog" if tool_name == "search_tracks" else "Preparing the playlist export"
            yield event({"type": "trace", "stage": "action", "tool": tool_name, "arguments": arguments, "content": action})
            result = server.call_tool(tool_name, arguments).get("result", {})
            yield event({"type": "trace", "stage": "observation", "content": summary(result)})

            if tool_name == "search_tracks":
                tracks = result.get("tracks", []) if result.get("status") == "SUCCESS" else []
                session["tracks"] = merge_tracks(session["tracks"], tracks)
                yield event({"type": "tracks", "tracks": session["tracks"]})
            elif tool_name == "export_playlist" and result.get("status") == "SUCCESS":
                yield event({"type": "export", "playlist_name": result.get("playlist_name", "Playlist"), "download_url": "/api/download/playlist_export.csv"})

            final_context += f"\n{summary(result)}"
            context += f"\n\nObservation from {tool_name}: {json.dumps(result, ensure_ascii=False)}"
            if tool_name == "export_playlist" and result.get("status") == "SUCCESS":
                break
        else:
            final_text = "I reached the request limit before I could finish. Please try a simpler request."
            yield event({"type": "token", "content": final_text})

        if not final_text:
            yield event({"type": "trace", "stage": "result", "content": "Preparing the response"})
            final_prompt = final_context + "\nProvide the final helpful answer now. Do not call a tool."
            for token in provider.generate_stream(final_prompt, REACT_AGENT_SYSTEM_PROMPT):
                token = visible_text(token)
                final_text += token
                yield event({"type": "token", "content": token})
                await asyncio.sleep(0)
        if not final_text:
            final_text = "Your request is complete."
        session["history"].append(f"User: {message}\nAssistant: {final_text}\n{context}")
        session["history"] = session["history"][-6:]
        yield event({"type": "done"})
    except (json.JSONDecodeError, KeyError, TypeError):
        yield event({"type": "error", "content": "The agent returned an invalid response. Please try again."})
    except Exception:
        yield event({"type": "error", "content": "Something went wrong while processing your request. Please try again."})


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat/stream")
async def chat(request: Request):
    try:
        body = await request.json()
        message = str(body.get("message", "")).strip()
    except (json.JSONDecodeError, TypeError):
        return JSONResponse({"error": "Invalid request."}, status_code=400)
    if not message:
        return JSONResponse({"error": "A message is required."}, status_code=400)
    session_id = str(body.get("session_id") or uuid.uuid4())
    session = SESSIONS.setdefault(session_id, {"history": [], "tracks": []})
    return StreamingResponse(stream_chat(message, session), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.delete("/api/session/{session_id}")
def clear_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"status": "cleared"}


@app.get("/api/download/playlist_export.csv")
def download_csv():
    if not EXPORT_PATH.exists():
        return JSONResponse({"error": "No playlist export is available yet."}, status_code=404)
    return FileResponse(EXPORT_PATH, media_type="text/csv", filename="playlist_export.csv")


if __name__ == "__main__":
    import uvicorn
    print("MoodMix is running at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
