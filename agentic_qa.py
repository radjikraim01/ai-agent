#!/usr/bin/env python3
import json
import time
import sys
import requests
import re
from urllib.parse import quote_plus

# ============================
# Retry helper
# ============================

def retry_sync(fn, attempts=3, backoff_s=0.5):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i + 1 == attempts:
                raise
            time.sleep(backoff_s * (i + 1))


# ============================
# Tool: DuckDuckGo HTML Search (WORKS)
# ============================

def duckduckgo_search_tool(query: str) -> dict:
    """
    Scrape DuckDuckGo search results because JSON API is unreliable.
    """
    if not query:
        return {"query": query, "results": [], "error": "empty query"}

    encoded = quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={encoded}"

    def _call():
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        r.raise_for_status()
        return r.text

    html = retry_sync(_call, attempts=3, backoff_s=0.5)

    # Extract results
    pattern = r'<a rel="nofollow" class="result__a" href="(.*?)">(.*?)</a>'
    matches = re.findall(pattern, html)

    results = []
    for link, title in matches[:5]:
        clean_title = re.sub("<.*?>", "", title)
        results.append({
            "text": clean_title,
            "url": link
        })

    return {
        "query": query,
        "results": results,
        "raw": matches
    }


# ============================
# Tool: Dummy Weather
# ============================

def weather_tool(location: str) -> dict:
    dummy_db = {
        "doha": {"temperature_c": 32, "condition": "Sunny"},
        "algiers": {"temperature_c": 18, "condition": "Cloudy"},
        "london": {"temperature_c": 8, "condition": "Rain"},
    }
    loc = location.lower().strip()
    return dummy_db.get(loc, {"temperature_c": 25, "condition": "Clear"})


# ============================
# Tool Registry
# ============================

TOOLS = {
    "duckduckgo_search": duckduckgo_search_tool,
    "weather": weather_tool
}


# ============================
# Planner (simple rule-based)
# ============================

def planner(query: str) -> list:
    q = query.lower()

    if "weather" in q:
        # extract location
        if "in" in q:
            location = q.split("in", 1)[1].strip()
        else:
            location = q
        return [{"name": "weather", "args": {"location": location}}]

    return [{"name": "duckduckgo_search", "args": {"query": query}}]


# ============================
# Agent Execution
# ============================

def run_agent(query: str) -> dict:
    t0 = time.time()
    by_step = {}
    sources = []
    tool_calls = []
    final_answer = None
    reasoning_steps = []

    steps = planner(query)

    for step in steps:
        name = step["name"]
        fn = TOOLS[name]
        args = step["args"]

        t1 = time.time()
        out = fn(**args)
        latency = int((time.time() - t1) * 1000)
        by_step[name] = latency

        tool_calls.append({
            "name": name,
            "output": out,
            "latency_ms": latency
        })

        # Interpretation
        if name == "weather":
            final_answer = (
                f"The weather in {args['location']} is "
                f"{out['condition']} at {out['temperature_c']}°C."
            )
            sources.append({"name": "DummyWeather", "url": "local-dummy"})
            reasoning_steps.append("Used dummy weather tool.")

        elif name == "duckduckgo_search":
            reasoning_steps.append("Used web search (DuckDuckGo).")
            if out["results"]:
                first = out["results"][0]
                final_answer = first["text"]
                sources.append({
                    "name": "DuckDuckGo",
                    "url": first["url"]
                })
            else:
                final_answer = "I could not find an answer from the tools."
            reasoning_steps.append("DuckDuckGo returned results." if out["results"]
                                   else "DuckDuckGo returned no results.")

    return {
        "answer": final_answer,
        "sources": sources,
        "latency_ms": {
            "total": int((time.time() - t0) * 1000),
            "by_step": by_step
        },
        "tokens": {"prompt": 0, "completion": 0},
        "tool_calls": tool_calls,
        "reasoning": "\n".join(reasoning_steps)
    }


# ============================
# CLI Entry
# ============================

if __name__ == "__main__":
    # If arguments exist → one-shot mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(json.dumps(run_agent(query), indent=2))
        sys.exit()

    # Otherwise REPL mode
    print("Agentic QA REPL. Type a question, or 'quit' to exit.")
    while True:
        q = input(">> ").strip()
        if q.lower() in ("quit", "exit"):
            break
        result = run_agent(q)
        print(json.dumps(result, indent=2))
