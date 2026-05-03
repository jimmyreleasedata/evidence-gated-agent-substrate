"""Common OpenAI-compatible model backend client."""

from __future__ import annotations

import json
from typing import Any, Callable
import urllib.request


OpenerFn = Callable[[Any, float], Any]


class OpenAICompatibleClient:
    def __init__(self, *, api_base_url: str, opener: OpenerFn | None = None) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self._opener = opener or self._default_open

    def _api_root(self) -> str:
        if self.api_base_url.endswith("/v1"):
            return self.api_base_url
        return f"{self.api_base_url}/v1"

    @staticmethod
    def _default_open(request_or_url: Any, timeout: float = 60.0):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request_or_url, timeout=timeout)

    def served_model_name(self) -> str:
        with self._opener(f"{self._api_root()}/models", 30.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["data"][0]["id"])

    def generate(self, *, messages: list[dict[str, Any]], temperature: float = 0.0, max_tokens: int = 128) -> dict[str, Any]:
        served_model = self.served_model_name()
        request = urllib.request.Request(
            f"{self._api_root()}/chat/completions",
            data=json.dumps(
                {
                    "model": served_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener(request, 60.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        usage = payload.get("usage") or {}
        return {
            "id": payload.get("id"),
            "model": served_model,
            "text": payload.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "raw_payload": payload,
        }
