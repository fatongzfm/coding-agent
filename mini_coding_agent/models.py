import json
from typing import Iterator

import httpx


class FakeModelClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def complete(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)

    def stream_complete(self, prompt, max_new_tokens) -> Iterator[str]:
        full = self.complete(prompt, max_new_tokens)
        for ch in full:
            yield ch


class OllamaModelClient:
    def __init__(self, model, host, temperature, top_p, timeout):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    def complete(self, prompt, max_new_tokens):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.host + "/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama request failed with HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ) from exc

        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")

    def stream_complete(self, prompt, max_new_tokens) -> Iterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    self.host + "/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if data.get("done"):
                            break
                        token = data.get("response", "")
                        if token:
                            yield token
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama request failed with HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ) from exc


class OpenAiCompatibleClient:
    def __init__(self, model, base_url, api_key, temperature, top_p, timeout):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    def complete(self, prompt, max_new_tokens):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        endpoint = "/chat/completions" if self.base_url.endswith("/v1") else "/v1/chat/completions"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.base_url + endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"API request failed with HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                "Could not reach API endpoint.\n"
                f"Base URL: {self.base_url}\n"
                f"Model: {self.model}"
            ) from exc

        if data.get("error"):
            raise RuntimeError(f"API error: {data['error']}")

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected API response format: {data}") from exc

    def stream_complete(self, prompt, max_new_tokens) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True,
        }
        endpoint = "/chat/completions" if self.base_url.endswith("/v1") else "/v1/chat/completions"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    self.base_url + endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line or line == "data: [DONE]":
                            continue
                        if not line.startswith("data: "):
                            continue
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"API request failed with HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                "Could not reach API endpoint.\n"
                f"Base URL: {self.base_url}\n"
                f"Model: {self.model}"
            ) from exc
