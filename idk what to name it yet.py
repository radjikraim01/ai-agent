#!/usr/bin/env python3
"""
Agentic QA System with Local LLM Support
=========================================

Enhanced agent with Ollama integration for intelligent answer generation.
Falls back to basic formatting if Ollama is not available.

Requirements:
- requests (already installed)
- Ollama (optional, for LLM enhancement)

Install Ollama: https://ollama.ai
Then run: ollama pull llama3.2
"""

import json
import time
import sys
import re
import os
import html
import logging
import hashlib
import argparse
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import (
    Dict, List, Any, Optional, Callable, 
    Tuple, Union, TypeVar, Generic
)
from urllib.parse import quote_plus, unquote
from enum import Enum, auto
from functools import wraps
from datetime import datetime, timedelta, timezone

# Third-party imports with graceful fallback
try:
    import requests
    from requests.exceptions import RequestException, Timeout, HTTPError
except ImportError:
    print("Error: 'requests' library is required. Install with: pip install requests")
    sys.exit(1)

try:
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SearchConfig:
    """Configuration for search-related settings."""
    timeout: int = 10
    max_results: int = 5
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    base_url: str = "https://duckduckgo.com/html/"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_backoff_seconds: float = 0.5
    use_exponential_backoff: bool = True


@dataclass
class CacheConfig:
    """Configuration for caching behavior."""
    enabled: bool = True
    ttl_seconds: int = 300  # 5 minutes
    max_size: int = 100


@dataclass
class LLMConfig:
    """Configuration for LLM settings."""
    enabled: bool = True
    provider: str = "ollama"  # "ollama" or "none"
    model: str = "llama3.2:latest"
    endpoint: str = "http://localhost:11434"
    timeout: int = 30
    max_tokens: int = 200


@dataclass
class AgentConfig:
    """Centralized configuration for the entire agent system."""
    search: SearchConfig = field(default_factory=SearchConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    max_query_length: int = 500
    log_level: str = "INFO"
    
    @classmethod
    def from_environment(cls) -> 'AgentConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Search config
        config.search.timeout = int(
            os.getenv("AGENT_SEARCH_TIMEOUT", config.search.timeout)
        )
        config.search.max_results = int(
            os.getenv("AGENT_MAX_RESULTS", config.search.max_results)
        )
        
        # Retry config
        config.retry.max_attempts = int(
            os.getenv("AGENT_RETRY_ATTEMPTS", config.retry.max_attempts)
        )
        
        # Cache config
        config.cache.enabled = os.getenv(
            "AGENT_CACHE_ENABLED", "true"
        ).lower() == "true"
        config.cache.ttl_seconds = int(
            os.getenv("AGENT_CACHE_TTL", config.cache.ttl_seconds)
        )
        
        # LLM config
        config.llm.enabled = os.getenv(
            "AGENT_LLM_ENABLED", "true"
        ).lower() == "true"
        config.llm.provider = os.getenv("AGENT_LLM_PROVIDER", config.llm.provider)
        config.llm.model = os.getenv("AGENT_LLM_MODEL", config.llm.model)
        config.llm.endpoint = os.getenv("AGENT_LLM_ENDPOINT", config.llm.endpoint)
        
        config.log_level = os.getenv("AGENT_LOG_LEVEL", config.log_level)
        
        return config


# Global configuration instance
CONFIG = AgentConfig.from_environment()


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return a logger instance."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    return logging.getLogger("AgenticQA")


logger = setup_logging(CONFIG.log_level)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class AgentError(Exception):
    """Base exception for all agent-related errors."""
    pass


class ToolExecutionError(AgentError):
    """Raised when a tool fails to execute."""
    def __init__(self, tool_name: str, message: str, original_error: Exception = None):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ValidationError(AgentError):
    """Raised when input validation fails."""
    pass


class ConfigurationError(AgentError):
    """Raised when configuration is invalid."""
    pass


class CacheError(AgentError):
    """Raised when cache operations fail."""
    pass


# ============================================================================
# DATA MODELS
# ============================================================================

class Intent(Enum):
    """Enumeration of supported user intents."""
    WEATHER = auto()
    SEARCH = auto()
    HELP = auto()
    UNKNOWN = auto()


@dataclass
class ToolResult:
    """Standardized result from tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    latency_ms: int = 0
    cached: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "cached": self.cached
        }


@dataclass
class SearchResult:
    """Individual search result."""
    title: str
    url: str
    snippet: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary representation."""
        result = {"title": self.title, "url": self.url}
        if self.snippet:
            result["snippet"] = self.snippet
        return result


@dataclass
class WeatherData:
    """Weather information for a location."""
    location: str
    temperature_celsius: float
    condition: str
    humidity: Optional[int] = None
    wind_speed_kmh: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "location": self.location,
            "temperature_c": self.temperature_celsius,
            "condition": self.condition,
            "humidity": self.humidity,
            "wind_speed_kmh": self.wind_speed_kmh
        }


@dataclass
class ToolCall:
    """Record of a tool invocation."""
    name: str
    args: Dict[str, Any]
    result: ToolResult
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "args": self.args,
            "output": self.result.to_dict(),
            "timestamp": self.timestamp
        }


@dataclass
class AgentResponse:
    """Complete response from the agent."""
    answer: str
    sources: List[Dict[str, str]]
    latency_ms: Dict[str, Any]
    tokens: Dict[str, int]
    tool_calls: List[ToolCall]
    reasoning: str
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "reasoning": self.reasoning,
            "success": self.success,
            "error": self.error
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sanitize_query(query: str, max_length: int = None) -> str:
    """Sanitize and validate user input."""
    if max_length is None:
        max_length = CONFIG.max_query_length
    
    if query is None:
        raise ValidationError("Query cannot be None")
    
    if not isinstance(query, str):
        raise ValidationError(f"Query must be a string, got {type(query).__name__}")
    
    sanitized = query.strip()
    
    if not sanitized:
        raise ValidationError("Query cannot be empty")
    
    if len(sanitized) > max_length:
        logger.warning(
            f"Query truncated from {len(sanitized)} to {max_length} characters"
        )
        sanitized = sanitized[:max_length]
    
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
    
    return sanitized


def clean_html(text: str) -> str:
    """Remove HTML tags and unescape HTML entities."""
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_cache_key(*args, **kwargs) -> str:
    """Generate a unique cache key from arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def retry_with_backoff(
    func: Callable,
    max_attempts: int = None,
    base_backoff: float = None,
    exponential: bool = None,
    exceptions: Tuple[type, ...] = (Exception,)
) -> Any:
    """Execute a function with retry logic and backoff."""
    if max_attempts is None:
        max_attempts = CONFIG.retry.max_attempts
    if base_backoff is None:
        base_backoff = CONFIG.retry.base_backoff_seconds
    if exponential is None:
        exponential = CONFIG.retry.use_exponential_backoff
    
    last_exception = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            
            if attempt < max_attempts:
                if exponential:
                    wait_time = base_backoff * (2 ** (attempt - 1))
                else:
                    wait_time = base_backoff * attempt
                
                logger.warning(
                    f"Attempt {attempt}/{max_attempts} failed: {e}. "
                    f"Retrying in {wait_time:.2f}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_attempts} attempts failed. Last error: {e}")
    
    raise last_exception


# ============================================================================
# CACHING SYSTEM
# ============================================================================

class Cache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(
        self,
        ttl_seconds: int = None,
        max_size: int = None,
        enabled: bool = None
    ):
        """Initialize the cache."""
        self.ttl_seconds = ttl_seconds or CONFIG.cache.ttl_seconds
        self.max_size = max_size or CONFIG.cache.max_size
        self.enabled = enabled if enabled is not None else CONFIG.cache.enabled
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._access_order: List[str] = []
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from cache."""
        if not self.enabled:
            return None
        
        if key not in self._cache:
            return None
        
        timestamp, value = self._cache[key]
        
        if time.time() - timestamp > self.ttl_seconds:
            self._remove(key)
            logger.debug(f"Cache entry expired: {key[:16]}...")
            return None
        
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        logger.debug(f"Cache hit: {key[:16]}...")
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Store a value in cache."""
        if not self.enabled:
            return
        
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._remove(oldest_key)
            logger.debug(f"Cache evicted: {oldest_key[:16]}...")
        
        self._cache[key] = (time.time(), value)
        self._access_order.append(key)
        logger.debug(f"Cache set: {key[:16]}...")
    
    def _remove(self, key: str) -> None:
        """Remove an entry from cache."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self.enabled,
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


# Global cache instance
cache = Cache()


# ============================================================================
# ABSTRACT BASE TOOL
# ============================================================================

class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, use_cache: bool = True):
        """Initialize the tool."""
        self.use_cache = use_cache
        self._execution_count = 0
        self._total_latency_ms = 0
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the tool."""
        pass
    
    @abstractmethod
    def _execute(self, **kwargs) -> Any:
        """Internal execution logic. Override in subclasses."""
        pass
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with caching and error handling."""
        start_time = time.perf_counter()
        
        cache_key = generate_cache_key(self.name, **kwargs)
        
        if self.use_cache:
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                return ToolResult(
                    success=True,
                    data=cached_result,
                    latency_ms=latency_ms,
                    cached=True
                )
        
        try:
            result_data = self._execute(**kwargs)
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            
            self._execution_count += 1
            self._total_latency_ms += latency_ms
            
            if self.use_cache:
                cache.set(cache_key, result_data)
            
            logger.info(
                f"Tool '{self.name}' executed successfully in {latency_ms}ms"
            )
            
            return ToolResult(
                success=True,
                data=result_data,
                latency_ms=latency_ms,
                cached=False
            )
            
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            
            logger.error(f"Tool '{self.name}' failed: {e}")
            
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                latency_ms=latency_ms,
                cached=False
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics for this tool."""
        avg_latency = (
            self._total_latency_ms / self._execution_count
            if self._execution_count > 0 else 0
        )
        return {
            "name": self.name,
            "execution_count": self._execution_count,
            "total_latency_ms": self._total_latency_ms,
            "average_latency_ms": round(avg_latency, 2)
        }


# ============================================================================
# LOCAL LLM TOOL (OLLAMA)
# ============================================================================

class LocalLLMTool(BaseTool):
    """Local LLM tool using Ollama."""
    
    name = "local_llm"
    description = "Generate intelligent responses using local LLM"
    
    def __init__(
        self,
        model: str = None,
        endpoint: str = None,
        timeout: int = None,
        **kwargs
    ):
        """Initialize the LLM tool."""
        super().__init__(**kwargs)
        self.model = model or CONFIG.llm.model
        self.endpoint = endpoint or CONFIG.llm.endpoint
        self.timeout = timeout or CONFIG.llm.timeout
        self.available = False
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test if Ollama is available."""
        try:
            response = requests.get(
                f"{self.endpoint}/api/tags",
                timeout=2
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                logger.info(f"✓ Ollama connected. Available models: {model_names}")
                
                # Check if our model is available
                if not any(self.model in name for name in model_names):
                    logger.warning(
                        f"Model '{self.model}' not found. "
                        f"Run: ollama pull {self.model}"
                    )
                else:
                    self.available = True
            else:
                logger.warning("Ollama responded but returned non-200 status")
        except Exception as e:
            logger.warning(
                f"Ollama not available: {e}. "
                f"Install from https://ollama.ai and run: ollama pull {self.model}"
            )
    
    def _execute(self, prompt: str, max_tokens: int = None) -> str:
        """Execute LLM generation."""
        if not self.available:
            raise ToolExecutionError(
                self.name,
                "Ollama not available. Install from https://ollama.ai"
            )
        
        if max_tokens is None:
            max_tokens = CONFIG.llm.max_tokens
        
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": max_tokens,
                        "stop": ["\n\n", "Question:", "User:"]
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()["response"].strip()
                
                # Clean up common artifacts
                result = re.sub(r'^(Answer:|Response:)\s*', '', result, flags=re.IGNORECASE)
                result = re.sub(r'Based on (the )?search results?,?\s*', '', result, flags=re.IGNORECASE)
                
                return result
            else:
                raise ToolExecutionError(
                    self.name,
                    f"Ollama returned status {response.status_code}: {response.text}"
                )
                
        except requests.exceptions.ConnectionError:
            raise ToolExecutionError(
                self.name,
                "Could not connect to Ollama. Is it running? Try: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise ToolExecutionError(
                self.name,
                f"Request timed out after {self.timeout}s"
            )
        except Exception as e:
            raise ToolExecutionError(self.name, str(e), e)


# ============================================================================
# DUCKDUCKGO SEARCH TOOL
# ============================================================================

class DuckDuckGoSearchTool(BaseTool):
    """Web search tool using DuckDuckGo."""
    
    name = "duckduckgo_search"
    description = "Search the web using DuckDuckGo"
    
    RESULT_PATTERN = re.compile(
        r'<a rel="nofollow" class="result__a" href="(.*?)">(.*?)</a>',
        re.DOTALL
    )
    
    SNIPPET_PATTERN = re.compile(
        r'<a class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL
    )
    
    def __init__(
        self,
        timeout: int = None,
        max_results: int = None,
        user_agent: str = None,
        **kwargs
    ):
        """Initialize the search tool."""
        super().__init__(**kwargs)
        self.timeout = timeout or CONFIG.search.timeout
        self.max_results = max_results or CONFIG.search.max_results
        self.user_agent = user_agent or CONFIG.search.user_agent
        self.base_url = CONFIG.search.base_url
    
    def _execute(self, query: str) -> Dict[str, Any]:
        """Execute a search query."""
        query = sanitize_query(query)
        
        encoded_query = quote_plus(query)
        url = f"{self.base_url}?q={encoded_query}"
        
        def make_request():
            response = requests.get(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.text
        
        try:
            html_content = retry_with_backoff(
                make_request,
                exceptions=(RequestException, Timeout, HTTPError)
            )
        except Exception as e:
            raise ToolExecutionError(self.name, f"Request failed: {e}", e)
        
        results = self._parse_results(html_content)
        
        return {
            "query": query,
            "results": [r.to_dict() for r in results],
            "result_count": len(results)
        }
    
    def _parse_results(self, html_content: str) -> List[SearchResult]:
        """Parse search results from HTML content."""
        results = []
        
        matches = self.RESULT_PATTERN.findall(html_content)
        snippets = self.SNIPPET_PATTERN.findall(html_content)
        
        for i, (link, title) in enumerate(matches[:self.max_results]):
            clean_title = clean_html(title)
            clean_url = unquote(link)
            
            # Remove DuckDuckGo redirect wrapper
            if 'uddg=' in clean_url:
                clean_url = clean_url.split('uddg=')[1].split('&')[0]
                clean_url = unquote(clean_url)
            
            snippet = None
            if i < len(snippets):
                snippet = clean_html(snippets[i])
            
            if clean_title and clean_url:
                results.append(SearchResult(
                    title=clean_title,
                    url=clean_url,
                    snippet=snippet
                ))
        
        logger.debug(f"Parsed {len(results)} search results")
        return results


# ============================================================================
# WEATHER TOOL
# ============================================================================

class WeatherTool(BaseTool):
    """Weather information tool."""
    
    name = "weather"
    description = "Get weather information for a location"
    
    WEATHER_DATA = {
        "doha": WeatherData(
            location="Doha, Qatar",
            temperature_celsius=32,
            condition="Sunny",
            humidity=45,
            wind_speed_kmh=15
        ),
        "algiers": WeatherData(
            location="Algiers, Algeria",
            temperature_celsius=18,
            condition="Cloudy",
            humidity=65,
            wind_speed_kmh=20
        ),
        "london": WeatherData(
            location="London, UK",
            temperature_celsius=8,
            condition="Rainy",
            humidity=85,
            wind_speed_kmh=25
        ),
        "new york": WeatherData(
            location="New York, USA",
            temperature_celsius=12,
            condition="Partly Cloudy",
            humidity=55,
            wind_speed_kmh=18
        ),
        "tokyo": WeatherData(
            location="Tokyo, Japan",
            temperature_celsius=15,
            condition="Clear",
            humidity=50,
            wind_speed_kmh=10
        ),
        "sydney": WeatherData(
            location="Sydney, Australia",
            temperature_celsius=25,
            condition="Sunny",
            humidity=60,
            wind_speed_kmh=12
        ),
    }
    
    DEFAULT_WEATHER = WeatherData(
        location="Unknown",
        temperature_celsius=20,
        condition="Clear",
        humidity=50,
        wind_speed_kmh=10
    )
    
    def __init__(self, **kwargs):
        """Initialize the weather tool."""
        super().__init__(**kwargs)
    
    def _execute(self, location: str) -> Dict[str, Any]:
        """Get weather for a location."""
        if not location or not isinstance(location, str):
            raise ValidationError("Location must be a non-empty string")
        
        normalized = location.lower().strip()
        
        for suffix in [" weather", " forecast", "?"]:
            normalized = normalized.rstrip(suffix).strip()
        
        weather = self.WEATHER_DATA.get(normalized)
        
        if weather is None:
            weather = WeatherData(
                location=location.title(),
                temperature_celsius=self.DEFAULT_WEATHER.temperature_celsius,
                condition=self.DEFAULT_WEATHER.condition,
                humidity=self.DEFAULT_WEATHER.humidity,
                wind_speed_kmh=self.DEFAULT_WEATHER.wind_speed_kmh
            )
            logger.info(f"No data for '{location}', using defaults")
        
        return weather.to_dict()


# ============================================================================
# TOOL REGISTRY
# ============================================================================

class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        """Initialize the registry."""
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")
        
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List all registered tools."""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all tools."""
        return [tool.get_stats() for tool in self._tools.values()]


# Global tool registry
tool_registry = ToolRegistry()
tool_registry.register(DuckDuckGoSearchTool())
tool_registry.register(WeatherTool())

# ============================================================================
# INTENT DETECTION & PLANNER
# ============================================================================

class IntentDetector:
    """Detects user intent from natural language queries."""
    
    WEATHER_PATTERNS = [
        re.compile(r'\bweather\s+(?:in|for|at)\s+(.+)', re.IGNORECASE),
        re.compile(r"(?:what'?s|what\s+is|how'?s|how\s+is)\s+(?:the\s+)?weather\s+(?:in|at|for)\s+(.+)", re.IGNORECASE),
        re.compile(r'\btemperature\s+(?:in|at|for)\s+(.+)', re.IGNORECASE),
        re.compile(r'\bforecast\s+(?:in|for|at)\s+(.+)', re.IGNORECASE),
        re.compile(r'(?:is\s+it|will\s+it)\s+(?:rain|snow|sunny|cold|hot)\s+(?:in|at)\s+(.+)', re.IGNORECASE),
    ]
    
    HELP_PATTERNS = [
        re.compile(r'^(?:help|commands|usage|\?)$', re.IGNORECASE),
        re.compile(r'\bwhat\s+can\s+you\s+do\b', re.IGNORECASE),
        re.compile(r'\bshow\s+(?:me\s+)?help\b', re.IGNORECASE),
    ]
    
    @classmethod
    def detect(cls, query: str) -> Tuple[Intent, Dict[str, Any]]:
        """Detect intent and extract entities from query."""
        query = query.strip()
        
        for pattern in cls.HELP_PATTERNS:
            if pattern.search(query):
                return Intent.HELP, {}
        
        for pattern in cls.WEATHER_PATTERNS:
            match = pattern.search(query)
            if match:
                location = match.group(1).strip().rstrip("?.!")
                return Intent.WEATHER, {"location": location}
        
        if "weather" in query.lower():
            words = query.lower().split()
            if "weather" in words:
                idx = words.index("weather")
                if idx + 1 < len(words):
                    location = " ".join(words[idx + 1:]).strip("?.!")
                    if location and location not in ["in", "at", "for"]:
                        return Intent.WEATHER, {"location": location}
        
        return Intent.SEARCH, {"query": query}


class Planner:
    """Plans tool execution based on detected intent."""
    
    @staticmethod
    def plan(query: str) -> List[Dict[str, Any]]:
        """Create an execution plan for the query."""
        intent, entities = IntentDetector.detect(query)
        
        logger.debug(f"Detected intent: {intent.name}, entities: {entities}")
        
        if intent == Intent.WEATHER:
            return [{
                "name": "weather",
                "args": {"location": entities["location"]}
            }]
        
        elif intent == Intent.HELP:
            return [{
                "name": "_help",
                "args": {}
            }]
        
        else:  # SEARCH or UNKNOWN
            return [{
                "name": "duckduckgo_search",
                "args": {"query": entities.get("query", query)}
            }]


# ============================================================================
# RESPONSE FORMATTER WITH LLM SUPPORT
# ============================================================================

class EnhancedResponseFormatter:
    """Response formatter with optional LLM enhancement."""
    
    def __init__(self):
        """Initialize the formatter."""
        self.llm = None
        
        # Try to initialize LLM if enabled
        if CONFIG.llm.enabled and CONFIG.llm.provider == "ollama":
            try:
                self.llm = LocalLLMTool()
                if self.llm.available:
                    logger.info("✓ LLM enhancement enabled")
                else:
                    logger.info("○ LLM unavailable - using basic formatting")
                    self.llm = None
            except Exception as e:
                logger.warning(f"Could not initialize LLM: {e}")
                self.llm = None
        else:
            logger.info("○ LLM disabled in config - using basic formatting")
    
    def format_weather(self, weather_data: Dict[str, Any], location: str) -> str:
        """Format weather data into a readable response."""
        temp = weather_data.get("temperature_c", "N/A")
        condition = weather_data.get("condition", "Unknown")
        humidity = weather_data.get("humidity")
        wind = weather_data.get("wind_speed_kmh")
        
        response = (
            f"The weather in {weather_data.get('location', location)} is "
            f"{condition} with a temperature of {temp}°C."
        )
        
        if humidity is not None:
            response += f" Humidity is {humidity}%."
        
        if wind is not None:
            response += f" Wind speed is {wind} km/h."
        
        return response
    
    def format_search(self, search_data: Dict[str, Any], original_query: str = "") -> str:
        """Format search results with optional LLM enhancement."""
        results = search_data.get("results", [])
        
        if not results:
            return "I couldn't find any relevant results for your query."
        
        # Try LLM enhancement if available
        if self.llm and self.llm.available:
            try:
                return self._format_with_llm(results, original_query)
            except Exception as e:
                logger.warning(f"LLM formatting failed: {e}, using fallback")
        
        # Fallback to basic formatting
        return self._format_basic(results)
    
    def _format_with_llm(self, results: List[Dict], query: str) -> str:
        """Generate natural answer using LLM."""
        
        # Build context from top 3 results
        context_parts = []
        for i, result in enumerate(results[:3], 1):
            title = result.get('title', 'Unknown')
            snippet = result.get('snippet', '')
            if snippet:
                context_parts.append(f"[Source {i}] {title}\n{snippet}")
            else:
                context_parts.append(f"[Source {i}] {title}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Answer this question directly and concisely using the search results below. Give a 1-2 sentence answer with the key information.

Question: {query}

Search Results:
{context}

Direct Answer:"""
        
        # Get LLM response
        result = self.llm.execute(prompt=prompt)
        
        if result.success and result.data:
            answer = result.data
            
            # Clean up
            answer = answer.strip()
            if len(answer) > 500:
                answer = answer[:500] + "..."
            
            return answer
        else:
            raise Exception("LLM returned no data")
    
    def _format_basic(self, results: List[Dict]) -> str:
        """Basic formatting without LLM."""
        first = results[0]
        response = first.get("title", "No title available")
        
        snippet = first.get("snippet")
        if snippet:
            response += f"\n\n{snippet}"
        
        if len(results) > 1:
            response += f"\n\n(Found {len(results)} results)"
        
        return response
    
    def format_help(self) -> str:
        """Format help information."""
        tools = tool_registry.list_tools()
        
        help_text = "**Available Commands:**\n\n"
        help_text += "• Ask any question to search the web\n"
        help_text += "• Ask about weather in any city (e.g., 'weather in London')\n"
        help_text += "• Type 'quit' or 'exit' to leave\n\n"
        help_text += "**Available Tools:**\n"
        
        for tool in tools:
            help_text += f"• **{tool['name']}**: {tool['description']}\n"
        
        if self.llm and self.llm.available:
            help_text += f"\n✓ LLM Enhancement: Enabled ({CONFIG.llm.model})\n"
        else:
            help_text += "\n○ LLM Enhancement: Disabled (install Ollama for smart answers)\n"
        
        return help_text
    
    def format_error(self, error: str) -> str:
        """Format an error message."""
        return f"I encountered an error: {error}. Please try again."


# ============================================================================
# MAIN AGENT
# ============================================================================

class Agent:
    """Main agent orchestrator."""
    
    def __init__(self):
        """Initialize the agent."""
        self.planner = Planner()
        self.formatter = EnhancedResponseFormatter()
    
    def run(self, query: str) -> AgentResponse:
        """Process a user query and return a response."""
        start_time = time.perf_counter()
        
        tool_calls: List[ToolCall] = []
        sources: List[Dict[str, str]] = []
        reasoning_steps: List[str] = []
        by_step_latency: Dict[str, int] = {}
        
        try:
            original_query = query
            query = sanitize_query(query)
            reasoning_steps.append(
                f"Received query: '{query[:50]}...' " if len(query) > 50 
                else f"Received query: '{query}'"
            )
            
            steps = self.planner.plan(query)
            reasoning_steps.append(f"Planned {len(steps)} tool call(s)")
            
            final_answer = None
            
            for step in steps:
                tool_name = step["name"]
                tool_args = step["args"]
                
                if tool_name == "_help":
                    final_answer = self.formatter.format_help()
                    reasoning_steps.append("Displayed help information")
                    continue
                
                tool = tool_registry.get(tool_name)
                if tool is None:
                    raise ToolExecutionError(tool_name, "Tool not found")
                
                reasoning_steps.append(f"Executing tool: {tool_name}")
                result = tool.execute(**tool_args)
                
                tool_call = ToolCall(
                    name=tool_name,
                    args=tool_args,
                    result=result
                )
                tool_calls.append(tool_call)
                by_step_latency[tool_name] = result.latency_ms
                
                if result.success:
                    if result.cached:
                        reasoning_steps.append(f"Retrieved cached result for {tool_name}")
                    else:
                        reasoning_steps.append(f"Tool {tool_name} completed in {result.latency_ms}ms")
                    
                    if tool_name == "weather":
                        final_answer = self.formatter.format_weather(
                            result.data, 
                            tool_args.get("location", "")
                        )
                        sources.append({
                            "name": "Weather Service",
                            "url": "internal://weather-service"
                        })
                    
                    elif tool_name == "duckduckgo_search":
                        final_answer = self.formatter.format_search(
                            result.data, 
                            original_query
                        )
                        results = result.data.get("results", [])
                        if results:
                            sources.append({
                                "name": "DuckDuckGo",
                                "url": results[0].get("url", "https://duckduckgo.com")
                            })
                            for r in results[1:3]:
                                sources.append({
                                    "name": r.get("title", "Web Result")[:50],
                                    "url": r.get("url", "")
                                })
                else:
                    reasoning_steps.append(f"Tool {tool_name} failed: {result.error}")
                    final_answer = self.formatter.format_error(result.error)
            
            if final_answer is None:
                final_answer = "I couldn't process your request. Please try rephrasing."
            
            total_latency = int((time.perf_counter() - start_time) * 1000)
            
            return AgentResponse(
                answer=final_answer,
                sources=sources,
                latency_ms={
                    "total": total_latency,
                    "by_step": by_step_latency
                },
                tokens={"prompt": 0, "completion": 0},
                tool_calls=tool_calls,
                reasoning="\n".join(reasoning_steps),
                success=True
            )
            
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            total_latency = int((time.perf_counter() - start_time) * 1000)
            
            return AgentResponse(
                answer=f"Invalid input: {e}",
                sources=[],
                latency_ms={"total": total_latency, "by_step": {}},
                tokens={"prompt": 0, "completion": 0},
                tool_calls=[],
                reasoning="Query validation failed",
                success=False,
                error=str(e)
            )
            
        except Exception as e:
            logger.exception(f"Agent error: {e}")
            total_latency = int((time.perf_counter() - start_time) * 1000)
            
            return AgentResponse(
                answer=f"An unexpected error occurred: {e}",
                sources=[],
                latency_ms={"total": total_latency, "by_step": {}},
                tokens={"prompt": 0, "completion": 0},
                tool_calls=tool_calls,
                reasoning="\n".join(reasoning_steps + [f"Error: {e}"]),
                success=False,
                error=str(e)
            )


# Global agent instance
agent = Agent()


def run_agent(query: str) -> dict:
    """Convenience function to run the agent."""
    response = agent.run(query)
    return response.to_dict()


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="agentic_qa",
        description="Agentic Question-Answering System with Optional LLM Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What is Python?"
  %(prog)s "weather in London"
  %(prog)s --interactive
  %(prog)s --no-llm "who is the president of France"
        """
    )
    
    parser.add_argument(
        "query",
        nargs="*",
        help="Query to process (omit for interactive mode)"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive REPL mode"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug output"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable result caching"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM enhancement (use basic formatting)"
    )
    
    parser.add_argument(
        "--llm-model",
        type=str,
        default=CONFIG.llm.model,
        help=f"LLM model to use (default: {CONFIG.llm.model})"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show tool statistics and exit"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 2.1.0 (with LLM support)"
    )
    
    return parser


def print_banner():
    """Print the application banner."""
    llm_status = "✓ LLM Enabled" if CONFIG.llm.enabled else "○ LLM Disabled"
    
    banner = f"""
╔═══════════════════════════════════════════════════════════╗
║           Agentic QA System v2.1.0                        ║
║           {llm_status:45} ║
║           Type 'help' for commands, 'quit' to exit        ║
╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_interactive_mode():
    """Run the interactive REPL mode."""
    print_banner()
    
    while True:
        try:
            query = input("\n>> ").strip()
            
            if query.lower() in ("quit", "exit", "q", ":q"):
                print("\nGoodbye! 👋")
                break
            
            if not query:
                continue
            
            if query.lower() == "clear":
                cache.clear()
                print("Cache cleared.")
                continue
            
            if query.lower() == "stats":
                stats = tool_registry.get_all_stats()
                print(json.dumps(stats, indent=2))
                continue
            
            result = run_agent(query)
            
            print(f"\n📝 Answer: {result['answer']}")
            
            if result['sources']:
                print(f"\n📚 Sources:")
                for source in result['sources']:
                    print(f"   • {source['name']}: {source['url']}")
            
            print(f"\n⏱️  Latency: {result['latency_ms']['total']}ms")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
            continue
            
        except EOFError:
            print("\n\nGoodbye! 👋")
            break


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    if args.no_cache:
        cache.enabled = False
        logger.info("Caching disabled")
    
    if args.no_llm:
        CONFIG.llm.enabled = False
        logger.info("LLM disabled")
    
    if args.llm_model != CONFIG.llm.model:
        CONFIG.llm.model = args.llm_model
        logger.info(f"LLM model set to {args.llm_model}")
    
    if args.stats:
        stats = {
            "tools": tool_registry.get_all_stats(),
            "cache": cache.stats(),
            "config": {
                "llm_enabled": CONFIG.llm.enabled,
                "llm_model": CONFIG.llm.model,
                "search_timeout": CONFIG.search.timeout,
                "max_results": CONFIG.search.max_results,
                "cache_enabled": CONFIG.cache.enabled
            }
        }
        print(json.dumps(stats, indent=2))
        return
    
    if args.query and not args.interactive:
        query = " ".join(args.query)
        result = run_agent(query)
        print(json.dumps(result, indent=2))
    else:
        run_interactive_mode()


if __name__ == "__main__":
    main()