"""Music tool schemas and small execution helpers for the MoodMix agent."""

import csv
import json
from pathlib import Path
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

TOOLS_SCHEMA = [
    {
        "name": "search_tracks",
        "description": "Search real songs in the iTunes catalog.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Song, artist, genre, or mood to search."},
                "limit": {"type": "integer", "description": "Number of tracks, from 1 to 10.", "default": 5, "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "export_playlist",
        "description": "Export tracks from the latest search as a Soundiiz-importable CSV.",
        "parameters": {
            "type": "object",
            "properties": {
                "playlist_name": {"type": "string", "description": "Name for the exported playlist."},
                "track_ids": {"type": "array", "items": {"type": "string"}, "description": "IDs returned by the latest search."},
            },
            "required": ["playlist_name", "track_ids"],
        },
    },
]

LATEST_TRACKS: Dict[str, Dict[str, Any]] = {}
EXPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "playlist_export.csv"


def _track(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "track_id": str(item.get("trackId", "")), "title": item.get("trackName", ""),
        "artist": item.get("artistName", ""), "album": item.get("collectionName", ""),
        "artwork_url": item.get("artworkUrl100", ""), "preview_url": item.get("previewUrl", ""),
        "track_url": item.get("trackViewUrl", ""), "duration_seconds": round(item.get("trackTimeMillis", 0) / 1000),
    }


def execute_search_tracks(query: str, limit: int = 5) -> str:
    """Search iTunes and keep only the newest normalized results in memory."""
    if not isinstance(query, str) or not query.strip():
        return json.dumps({"status": "ERROR", "message": "A non-empty search query is required."})
    try:
        limit = max(1, min(10, int(limit)))
    except (TypeError, ValueError):
        limit = 5
    params = urlencode({"term": query.strip(), "media": "music", "entity": "song", "country": "VN", "limit": limit})
    try:
        with urlopen(f"https://itunes.apple.com/search?{params}", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        return json.dumps({"status": "NETWORK_ERROR", "message": f"Could not search iTunes: {error}"})
    tracks = [track for item in payload.get("results", []) if (track := _track(item))["track_id"]]
    LATEST_TRACKS.clear()
    LATEST_TRACKS.update({track["track_id"]: track for track in tracks})
    return json.dumps({"status": "SUCCESS", "query": query.strip(), "track_count": len(tracks), "tracks": tracks}, ensure_ascii=False)


def execute_export_playlist(playlist_name: str, track_ids: list[str]) -> str:
    """Export selected latest-search tracks to a fixed CSV path."""
    if not isinstance(playlist_name, str) or not playlist_name.strip():
        return json.dumps({"status": "ERROR", "message": "A playlist name is required."})
    selected = [LATEST_TRACKS[track_id] for track_id in track_ids if isinstance(track_id, str) and track_id in LATEST_TRACKS]
    if not selected:
        return json.dumps({"status": "ERROR", "message": "No valid track IDs were provided from the latest search."})
    EXPORT_PATH.parent.mkdir(exist_ok=True)
    with EXPORT_PATH.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=["title", "artist", "album", "url"])
        writer.writeheader()
        writer.writerows({"title": t["title"], "artist": t["artist"], "album": t["album"], "url": t["track_url"]} for t in selected)
    return json.dumps({"status": "SUCCESS", "playlist_name": playlist_name.strip(), "file_path": "docs/playlist_export.csv", "track_count": len(selected), "tracks": selected}, ensure_ascii=False)


TOOL_ROUTER = {"search_tracks": execute_search_tracks, "export_playlist": execute_export_playlist}


def dispatch_tool_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Route a tool call and always return a JSON string."""
    if tool_name not in TOOL_ROUTER:
        return json.dumps({"status": "UNKNOWN_TOOL", "error": f"Tool '{tool_name}' does not exist."})
    try:
        return TOOL_ROUTER[tool_name](**arguments)
    except Exception as error:
        return json.dumps({"status": "EXECUTION_ERROR", "error": str(error)})
