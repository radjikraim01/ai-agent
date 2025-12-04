# Lightweight Local AI Agent with Tools (Assignment Solution)

This project implements a small but complete “local agent” system that:

- Loads a list of tools dynamically  
- Uses OpenAI’s ReAct-style tool calling  
- Performs web search (DuckDuckGo), math, and file read  
- Produces **clear, cited JSON answers**  
- Includes caching, retries, timeouts, and error handling  
- Includes tests, `.env.example`, and optional Docker support

---

## 🚀 Features

### ✔ Correct Tool Selection & Reasoning (45%)
- The agent determines when to use tools using simple ReAct prompting.
- DuckDuckGo is used for web search, and sources are returned in the output JSON.

### ✔ Code Quality & Structure
- Fully modular: Tool registry + agent logic + evaluation code.
- Uses `pydantic` models for structured tool definitions.

### ✔ Robustness
- Tools include:
  - **Retries** (exponential backoff)
  - **Timeouts** using `asyncio.wait_for`
  - **Caching** for search results (LRU)

### ✔ Latency Awareness
- Each tool reports:
  - latency per step  
  - total latency  
  - token usage  
  - full tool call trace

### ✔ Output JSON Format
Example output:
```json
{
  "answer": "Emmanuel Macron is the current president of France.",
  "sources": [
    {
      "name": "DuckDuckGo",
      "url": "https://example.com/macron"
    }
  ]
}
