"""Small in-process MCP-style server for MoodMix tools."""

import json
from typing import Any, Dict, List
from tools import TOOLS_SCHEMA, dispatch_tool_call


class MCPAcademicServer:
    """Keeps the starter server shape while exposing MoodMix tools."""

    def __init__(self, server_name: str = "moodmix-mcp-server"):
        self.server_name = server_name
        self.version = "2026.1.0"

    def list_tools(self) -> List[Dict[str, Any]]:
        return TOOLS_SCHEMA

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = json.loads(dispatch_tool_call(tool_name, arguments))
        return {"jsonrpc": "2.0", "server": self.server_name, "tool": tool_name, "result": result}


if __name__ == "__main__":
    server = MCPAcademicServer()
    print(f"MCP server: {server.server_name} ({server.version})")
    print(f"Tools: {', '.join(tool['name'] for tool in server.list_tools())}")
    print(json.dumps(server.call_tool("export_playlist", {"playlist_name": "Check", "track_ids": []})))
