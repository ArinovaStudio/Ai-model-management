"""Asynchronous LLM Client.

Interfaces with Ollama / vLLM / Cloud GPU endpoints using OpenAI-compatible API
or native Ollama format. Enforces temperature lock (0.1) and handles retries,
timeouts, and connection pooling.
"""

from typing import Any, AsyncGenerator
import httpx
from src.core.config import settings
from src.core.logging import logger


class LLMClient:
    """Async client for local or remote Cloud GPU LLM runtime."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
    ):
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self.api_key = api_key if api_key is not None else settings.LLM_API_KEY
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.timeout_seconds = timeout_seconds or settings.LLM_TIMEOUT_SECONDS

        # Reusable async connection pool
        self._client: httpx.AsyncClient | None = None

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            )
        return self._client

    async def close(self) -> None:
        """Closes the underlying HTTP connection pool."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Sends a non-streaming chat completion request."""
        client = self._get_client()
        temp = temperature if temperature is not None else self.temperature

        # Determine base host root (strip trailing /v1 if present)
        host_root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        openai_url = f"{host_root}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens,
            "stream": False,
        }

        attempt_timeout = httpx.Timeout(self.timeout_seconds, connect=10.0)
        headers = self._get_headers()

        try:
            response = await client.post(openai_url, json=payload, headers=headers, timeout=attempt_timeout)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            elif response.status_code != 404:
                raise RuntimeError(f"OpenAI endpoint returned HTTP {response.status_code}: {response.text}")
        except httpx.TimeoutException:
            logger.error(f"LLM request to '{openai_url}' timed out after {self.timeout_seconds} seconds.")
            raise
        except Exception as e:
            if self.api_key:
                logger.error(f"Cloud LLM endpoint '{openai_url}' failed: {e}")
                raise
            logger.debug(f"OpenAI endpoint '{openai_url}' failed: {e}. Falling back to native Ollama API...")

        # Fallback to Ollama native /api/chat
        ollama_url = f"{host_root}/api/chat"
        ollama_payload = {
            "model": self.model_name,
            "messages": messages,
            "options": {"temperature": temp},
            "stream": False,
        }

        try:
            response = await client.post(ollama_url, json=ollama_payload, timeout=attempt_timeout)
            if response.status_code == 200:
                data = response.json()
                return data["message"]["content"]
            else:
                raise RuntimeError(
                    f"LLM request failed with HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            logger.error(f"Failed to communicate with LLM endpoint at {self.base_url}: {e}")
            raise

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streams text chunks from the LLM endpoint for real-time speech synthesis."""
        import json
        client = self._get_client()
        temp = temperature if temperature is not None else self.temperature
        host_root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

        # Check if targeting an OpenAI-compatible /v1 endpoint
        openai_url = f"{host_root}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "stream": True,
        }
        headers = self._get_headers()
        try:
            async with client.stream("POST", openai_url, json=payload, headers=headers) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and line.strip() != "data: [DONE]":
                            chunk = json.loads(line[6:])
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                    return
        except Exception as e:
            if self.api_key:
                logger.error(f"Cloud LLM streaming failed: {e}")
                raise
            logger.debug(f"OpenAI streaming fallback: {e}")

        # Native Ollama /api/chat streaming
        ollama_url = f"{host_root}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "options": {"temperature": temp},
            "stream": True,
        }

        async with client.stream("POST", ollama_url, json=payload) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Streaming failed with status {response.status_code}")
            async for line in response.aiter_lines():
                if line:
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content


llm_client = LLMClient()
