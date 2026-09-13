"""MoodMix CLI application with a compact ReAct loop."""

import json
import os
import sys
import time
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mcp_server import MCPAcademicServer
from prompts import CHATBOT_BASELINE_PROMPT, MAX_ITERATIONS, REACT_AGENT_SYSTEM_PROMPT
from providers import get_llm_provider

load_dotenv()
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_test_cases():
    with open(os.path.join(ROOT_DIR, "config", "test_cases.json"), encoding="utf-8") as file:
        return json.load(file)


def save_waterfall_trace(trace_data: list):
    path = os.path.join(ROOT_DIR, "docs", "trace_waterfall.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(trace_data, file, ensure_ascii=False, indent=2)
    print(f"Saved {len(trace_data)} trace events to {path}")


def run_baseline_chatbot(user_query: str, provider):
    response = provider.generate(user_query, system_prompt=CHATBOT_BASELINE_PROMPT)
    print(f"MoodMix: {response}")


def format_observation(observation: dict) -> str:
    """Provide a short fallback answer if an agent reaches its iteration limit."""
    if observation.get("status") != "SUCCESS":
        return observation.get("message", "The tool could not complete the request.")
    if "playlist_name" in observation:
        return f"Exported {observation['track_count']} tracks to {observation['file_path']}."
    return f"Found {observation.get('track_count', 0)} matching tracks."


def run_react_agent(user_query: str, provider, mcp_server: MCPAcademicServer) -> list:
    """Run Thought → Action → Observation until text is returned or the limit is reached."""
    print(f"\nMoodMix request: {user_query}")
    trace_logs, working_context, latest_observation = [], user_query, {}
    tools_list = mcp_server.list_tools()

    for step in range(1, MAX_ITERATIONS + 1):
        started = time.time()
        llm_response = provider.generate_with_tools(working_context, tools_list, system_prompt=REACT_AGENT_SYSTEM_PROMPT)
        latency_ms = round((time.time() - started) * 1000, 2)
        thought = llm_response.get("thought", "Choosing the next step.")
        print(f"Thought: {thought}")

        if llm_response.get("type") == "text":
            content = llm_response.get("content", "")
            print(f"Final answer: {content}")
            trace_logs.append({"step": step, "query": user_query, "action_type": "FINAL_ANSWER", "thought": thought, "output": content, "latency_ms": latency_ms})
            return trace_logs

        if llm_response.get("type") != "tool_call":
            break
        tool_name, arguments = llm_response.get("tool_name"), llm_response.get("arguments", {})
        result = mcp_server.call_tool(tool_name, arguments)
        latest_observation = result.get("result", {})
        print(f"Action: {tool_name}({arguments})")
        print(f"Observation: {json.dumps(latest_observation, ensure_ascii=False)}")
        trace_logs.append({"step": step, "query": user_query, "action_type": "TOOL_EXECUTION", "tool_name": tool_name, "arguments": arguments, "observation": latest_observation, "latency_ms": latency_ms})
        working_context += f"\n\nObservation from {tool_name}: {json.dumps(latest_observation, ensure_ascii=False)}"

    content = format_observation(latest_observation)
    print(f"Final answer: {content}")
    trace_logs.append({"step": len(trace_logs) + 1, "query": user_query, "action_type": "FINAL_ANSWER", "thought": "Reached the ReAct iteration limit.", "output": content, "latency_ms": 0.0})
    return trace_logs


if __name__ == "__main__":
    provider, server = get_llm_provider(), MCPAcademicServer()
    if "--interactive" in sys.argv:
        print("MoodMix interactive mode. Type exit to finish.")
        while True:
            try:
                question = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not question or question.lower() in {"exit", "quit"}:
                break
            save_waterfall_trace(run_react_agent(question, provider, server))
    elif "--all" in sys.argv:
        all_traces = []
        for test in load_test_cases():
            print(f"\n[{test['id']}] {test['question']}")
            all_traces.extend(run_react_agent(test["question"], provider, server))
        save_waterfall_trace(all_traces)
    else:
        print("Use: python src/app.py --all | --interactive")
