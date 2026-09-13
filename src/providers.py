"""LLM providers for MoodMix, including native tool calling and final-response streaming."""

import json
import os
import re
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()


class BaseLLMProvider:
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        raise NotImplementedError

    def generate_with_tools(self, prompt: str, tools_schema: List[Dict[str, Any]], system_prompt: str = "") -> Dict[str, Any]:
        raise NotImplementedError

    def generate_stream(self, prompt: str, system_prompt: str = ""):
        yield self.generate(prompt, system_prompt)


class MockOfflineProvider(BaseLLMProvider):
    """Deterministic offline behavior for local tests without API credentials."""

    def __init__(self):
        self.model_name = "Offline-Mock-Model-2026"

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        return "MoodMix can search the music catalog and export selected results as a CSV for manual Soundiiz import."

    def generate_with_tools(self, prompt: str, tools_schema: List[Dict[str, Any]], system_prompt: str = "") -> Dict[str, Any]:
        lower = prompt.lower()
        has_search = "observation from search_tracks:" in lower
        wants_export = any(word in lower for word in ("export", "playlist", "xuất"))
        if "observation from export_playlist:" in lower:
            return {"type": "text", "content": "Your playlist CSV is ready for manual import through Soundiiz.", "thought": "Export complete."}
        if has_search and wants_export:
            ids = re.findall(r'"track_id"\s*:\s*"([^"\\]+)"', prompt)
            if not ids:
                return {"type": "text", "content": "No matching tracks were returned, so there is nothing to export.", "thought": "The search was empty."}
            return {"type": "tool_call", "tool_name": "export_playlist", "arguments": {"playlist_name": "Night Edit", "track_ids": ids[:5]}, "thought": "Preparing the playlist export."}
        if has_search:
            if '"track_count": 0' in prompt:
                return {"type": "text", "content": "No matching tracks were found in the music catalog.", "thought": "The search was empty."}
            return {"type": "text", "content": "Here are the tracks from the latest music search.", "thought": "Results are ready."}
        if any(word in lower for word in ("find", "song", "songs", "music", "track", "playlist", "tìm", "nhạc")):
            query = "electronic" if "electronic" in lower else prompt.split("\n", 1)[0]
            return {"type": "tool_call", "tool_name": "search_tracks", "arguments": {"query": query, "limit": 5}, "thought": "Music data is required."}
        return {"type": "text", "content": self.generate(prompt, system_prompt), "thought": "A direct answer is sufficient."}


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model or os.getenv("LLM_MODEL") or "gemini-2.5-flash"

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            return MockOfflineProvider().generate(prompt, system_prompt)
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(model=self.model_name, contents=f"{system_prompt}\n\n{prompt}")
            return response.text or ""
        except Exception as error:
            return f"[Gemini Exception]: {error}"

    def generate_with_tools(self, prompt: str, tools_schema: List[Dict[str, Any]], system_prompt: str = "") -> Dict[str, Any]:
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            return MockOfflineProvider().generate_with_tools(prompt, tools_schema, system_prompt)
        try:
            from google import genai
            from google.genai import types
            declarations = [{"name": tool["name"], "description": tool.get("description", ""), "parameters": tool["parameters"]} for tool in tools_schema if tool.get("name") and tool.get("parameters")]
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(model=self.model_name, contents=prompt, config=types.GenerateContentConfig(system_instruction=system_prompt, tools=[{"function_declarations": declarations}], temperature=0.2))
            if response.function_calls:
                call = response.function_calls[0]
                return {"type": "tool_call", "tool_name": call.name, "arguments": dict(call.args or {}), "thought": "LLM selected a tool."}
            return {"type": "text", "content": response.text or "", "thought": "LLM responded directly."}
        except Exception:
            return MockOfflineProvider().generate_with_tools(prompt, tools_schema, system_prompt)

    def generate_stream(self, prompt: str, system_prompt: str = ""):
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            yield from MockOfflineProvider().generate_stream(prompt, system_prompt)
            return
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            for chunk in client.models.generate_content_stream(model=self.model_name, contents=f"{system_prompt}\n\n{prompt}"):
                if chunk.text:
                    yield chunk.text
        except Exception:
            yield "I could not complete the response. Please try again."


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible provider; DeepSeek subclasses it with another base URL."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        self.base_url = None

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, **({"base_url": self.base_url} if self.base_url else {}))

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            return MockOfflineProvider().generate(prompt, system_prompt)
        try:
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}]
            return self._client().chat.completions.create(model=self.model_name, messages=messages).choices[0].message.content or ""
        except Exception as error:
            return f"[LLM Exception]: {error}"

    def generate_with_tools(self, prompt: str, tools_schema: List[Dict[str, Any]], system_prompt: str = "") -> Dict[str, Any]:
        if not self.api_key or self.api_key in {"your_openai_api_key_here", "your_deepseek_api_key_here"}:
            return MockOfflineProvider().generate_with_tools(prompt, tools_schema, system_prompt)
        try:
            tools = [{"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""), "parameters": tool.get("parameters", {})}} for tool in tools_schema if tool.get("name")]
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}]
            message = self._client().chat.completions.create(model=self.model_name, messages=messages, tools=tools or None, tool_choice="auto" if tools else None).choices[0].message
            if message.tool_calls:
                call = message.tool_calls[0]
                return {"type": "tool_call", "tool_name": call.function.name, "arguments": json.loads(call.function.arguments or "{}"), "thought": "LLM selected a tool."}
            return {"type": "text", "content": message.content or "", "thought": "LLM responded directly."}
        except Exception:
            return MockOfflineProvider().generate_with_tools(prompt, tools_schema, system_prompt)

    def generate_stream(self, prompt: str, system_prompt: str = ""):
        if not self.api_key or self.api_key in {"your_openai_api_key_here", "your_deepseek_api_key_here"}:
            yield from MockOfflineProvider().generate_stream(prompt, system_prompt)
            return
        try:
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}]
            for chunk in self._client().chat.completions.create(model=self.model_name, messages=messages, stream=True):
                content = chunk.choices[0].delta.content if chunk.choices else None
                if content:
                    yield content
        except Exception:
            yield "I could not complete the response. Please try again."


class DeepSeekProvider(OpenAIProvider):
    def __init__(self, api_key: str = None, model: str = None):
        super().__init__(api_key=api_key or os.getenv("DEEPSEEK_API_KEY"), model=model or os.getenv("LLM_MODEL") or "deepseek-flash")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def get_llm_provider() -> BaseLLMProvider:
    provider_type = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider_type == "gemini" and os.getenv("GEMINI_API_KEY") not in {None, "", "your_gemini_api_key_here"}:
        return GeminiProvider()
    if provider_type == "openai" and os.getenv("OPENAI_API_KEY") not in {None, "", "your_openai_api_key_here"}:
        return OpenAIProvider()
    if provider_type == "deepseek" and os.getenv("DEEPSEEK_API_KEY") not in {None, "", "your_deepseek_api_key_here"}:
        return DeepSeekProvider()
    return MockOfflineProvider()
