"""Python client for LLM Gateway V3. Adds auto_route kwarg on top of V2."""

import os
import json
import httpx
import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any, Optional

DEFAULT_URL = os.getenv("LLM_GATEWAY_V3_URL", "http://localhost:8101")
_LOCAL_MAIN = None


def _is_local_gateway_url(base_url: str) -> bool:
    local_prefixes = (
        "http://localhost",
        "http://127.0.0.1",
        "http://0.0.0.0",
    )
    return base_url.startswith(local_prefixes)


def _should_fallback_local_status_error(
    base_url: str, exc: httpx.HTTPStatusError
) -> bool:
    if not _is_local_gateway_url(base_url):
        return False
    status = exc.response.status_code if exc.response is not None else None
    return status in {502, 503, 504}


def _gateway_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_local_main():
    global _LOCAL_MAIN
    if _LOCAL_MAIN is not None:
        return _LOCAL_MAIN

    gateway_dir = str(_gateway_dir())
    if gateway_dir not in sys.path:
        sys.path.insert(0, gateway_dir)
    _LOCAL_MAIN = importlib.import_module("main")
    return _LOCAL_MAIN


def _ensure_local_gateway_state(module) -> None:
    app_state = module.app.state
    if getattr(app_state, "router", None) is not None:
        return
    module.db.init()
    app_state.cache = module.GeminiCache(ttl_seconds=300)
    app_state.providers = module.P.build_providers(app_state.cache)
    app_state.router = module.Router(app_state.providers, module.ORDER)
    app_state.router_providers = module.P.build_router_providers()
    app_state.router_pool = module.RouterPool(
        app_state.router_providers, module.ROUTER_ORDER
    )


def _local_chat(body: dict[str, Any]) -> dict[str, Any]:
    module = _load_local_main()
    _ensure_local_gateway_state(module)
    req = module.ChatRequest(**body)
    return asyncio.run(module.chat(req))


def _local_capabilities() -> dict[str, Any]:
    module = _load_local_main()
    _ensure_local_gateway_state(module)
    return asyncio.run(module.capabilities())


class LLM:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        prompt: str = None,
        *,
        messages: Optional[list] = None,
        system: Any = None,
        provider: str = None,
        model: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: Optional[list] = None,
        tool_choice: Any = None,
        cache_system: Optional[bool] = None,
        reasoning: Optional[str] = None,
        response_format: Any = None,
        auto_route: Optional[str] = None,
    ) -> dict:
        body = {
            "prompt": prompt,
            "messages": messages,
            "system": system,
            "provider": provider,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "tools": tools,
            "tool_choice": tool_choice,
            "cache_system": cache_system,
            "reasoning": reasoning,
            "response_format": response_format,
            "auto_route": auto_route,
        }
        body = {k: v for k, v in body.items() if v is not None}
        try:
            r = httpx.post(f"{self.base_url}/v1/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except httpx.ConnectError:
            if not _is_local_gateway_url(self.base_url):
                raise
            return _local_chat(body)
        except httpx.HTTPStatusError as exc:
            if not _should_fallback_local_status_error(self.base_url, exc):
                raise
            return _local_chat(body)

    def stream(
        self,
        prompt: str = None,
        *,
        messages=None,
        system=None,
        provider: str = None,
        model: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools=None,
        tool_choice=None,
        cache_system=None,
        reasoning=None,
        response_format=None,
    ):
        body = {
            "prompt": prompt,
            "messages": messages,
            "system": system,
            "provider": provider,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "tools": tools,
            "tool_choice": tool_choice,
            "cache_system": cache_system,
            "reasoning": reasoning,
            "response_format": response_format,
        }
        body = {k: v for k, v in body.items() if v is not None}
        with httpx.stream(
            "POST", f"{self.base_url}/v1/chat", json=body, timeout=self.timeout
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                d = json.loads(line[6:])
                if "delta" in d:
                    yield d["delta"]
                if d.get("done") or d.get("error"):
                    return

    def capabilities(self):
        try:
            r = httpx.get(f"{self.base_url}/v1/capabilities", timeout=30)
            r.raise_for_status()
            return r.json()
        except httpx.ConnectError:
            if not _is_local_gateway_url(self.base_url):
                raise
            return _local_capabilities()
        except httpx.HTTPStatusError as exc:
            if not _should_fallback_local_status_error(self.base_url, exc):
                raise
            return _local_capabilities()


def ask(prompt: str, provider: str = None, **kw) -> str:
    return LLM().chat(prompt, provider=provider, **kw)["text"]


if __name__ == "__main__":
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else None
    print(ask("Say hello in one short line.", provider=p))
