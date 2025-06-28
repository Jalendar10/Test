"""
Strands‑compatible provider: **custom_model_FI**
================================================
Drop‑in replacement for `strands.models.OpenAIModel` that sends requests to an
internal HMAC‑secured endpoint.  The implementation reproduces **exactly** the
signing logic from your `base_code.py`, so the server’s `HmacVerification`
policy is satisfied.

How it signs:
-------------
```
source = API_KEY + request_id + timestamp + json.dumps(body)  # default dumps
signature = base64(hmac.new(API_SECRET, source, SHA‑256).digest())
```
The JSON body layout and key order are byte‑for‑byte identical to
`create_request_body()` in *base_code.py*.

Environment / .env variables required:
```
API_KEY    = ...
API_SECRET = ...
BASE_URL   = https://<company‑llm>/chat/completions
```
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from typing import Any, Generator, Iterable, Type, TypeVar, Union, cast, TypedDict

import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from typing_extensions import override

from strands.types.content import Messages
from strands.types.models import OpenAIModel as SAOpenAIModel

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class CustomModelFI(SAOpenAIModel):
    """HMAC‑authenticated provider for Strands agents."""

    # --------------------------------------------------------------------- constants
    REQUIRED_ENV = ("API_KEY", "API_SECRET", "BASE_URL")

    class CustomConfig(TypedDict, total=False):
        model_id: str
        params: dict[str, Any] | None

    # --------------------------------------------------------------------- init / env
    def __init__(self, client_args: dict[str, Any] | None = None, **model_config: "CustomModelFI.CustomConfig"):
        self.config: CustomModelFI.CustomConfig = dict(model_config)
        self.api_key, self.api_secret, self.base_url = self._load_secrets()
        self.client = None  # placeholder for reflection compatibility
        logger.debug("custom_model_FI initialised – endpoint=%s", self.base_url)

    @staticmethod
    def _load_secrets() -> tuple[str, str, str]:
        if not getattr(CustomModelFI, "_dotenv_loaded", False):
            load_dotenv()
            CustomModelFI._dotenv_loaded = True  # type: ignore[attr-defined]
        missing = [v for v in CustomModelFI.REQUIRED_ENV if v not in os.environ]
        if missing:
            raise EnvironmentError("Missing env vars: " + ", ".join(missing))
        return os.environ["API_KEY"], os.environ["API_SECRET"], os.environ["BASE_URL"]

    # --------------------------------------------------------------------- config API
    @override
    def update_config(self, **model_config: "CustomModelFI.CustomConfig") -> None:  # type: ignore[override]
        self.config.update(model_config)

    @override
    def get_config(self) -> "CustomModelFI.CustomConfig":  # type: ignore[override]
        return cast(CustomModelFI.CustomConfig, self.config)

    # --------------------------------------------------------------------- helpers
    def _create_body(self, request: dict[str, Any]) -> dict[str, Any]:
        """Construct JSON body **exactly** like base_code.create_request_body."""
        p = self.config.get("params") or {}
        body = {
            # order matters for HMAC
            "model": self.config.get("model_id", "azure-openai-4o-mini-east"),
            "messages": request["messages"],
            "frequency_penalty": p.get("frequency_penalty", 0),
            "max_tokens": p.get("max_tokens", 1000),
            "n": p.get("n", 1),
            "presence_penalty": p.get("presence_penalty", 0),
            "response_format": p.get("response_format", {"type": "text"}),
            "stream": True,
            "temperature": p.get("temperature", 1),
            "top_p": p.get("top_p", 1),
        }
        return body

    def _headers(self, body: dict[str, Any]) -> dict[str, str]:
        ts = int(time.time() * 1000)
        req_id = uuid.uuid4()
        body_json = json.dumps(body)  # default separators – includes spaces
        source = f"{self.api_key}{req_id}{ts}{body_json}"
        signature = base64.b64encode(
            hmac.new(self.api_secret.encode(), source.encode(), hashlib.sha256).digest()
        ).decode()
        return {
            "api-key": self.api_key,
            "Client-Request-Id": str(req_id),
            "Timestamp": str(ts),
            "Authorization": signature,
            "Accept": "application/json",
        }

    # --------------------------------------------------------------------- streaming
    @override
    def stream(self, request: dict[str, Any]) -> Iterable[dict[str, Any]]:
        body = self._create_body(request)
        headers = self._headers(body)

        with requests.post(self.base_url, json=body, headers=headers, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"custom_model_FI HTTP {resp.status_code}: {resp.text}")

            yield {"chunk_type": "message_start"}
            yield {"chunk_type": "content_start", "data_type": "text"}

            tool_calls: dict[int, list[Any]] = {}
            finish_reason: str | None = None

            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                payload = json.loads(data)
                choice = payload.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                if (txt := delta.get("content")):
                    yield {"chunk_type": "content_delta", "data_type": "text", "data": txt}
                if (rc := delta.get("reasoning_content")):
                    yield {"chunk_type": "content_delta", "data_type": "reasoning_content", "data": rc}
                for tc in delta.get("tool_calls", []):
                    tool_calls.setdefault(tc["index"], []).append(tc)
                finish_reason = choice.get("finish_reason") or finish_reason

            yield {"chunk_type": "content_stop", "data_type": "text"}
            for deltas in tool_calls.values():
                yield {"chunk_type": "content_start", "data_type": "tool", "data": deltas[0]}
                for d in deltas:
                    yield {"chunk_type": "content_delta", "data_type": "tool", "data": d}
                yield {"chunk_type": "content_stop", "data_type": "tool"}
            yield {"chunk_type": "message_stop", "data": finish_reason or "stop"}
            yield {"chunk_type": "metadata", "data": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}}

    # --------------------------------------------------------------------- sync helper
    def complete(self, messages: Messages) -> str:
        req = {"messages": messages}
        return "".join(ev["data"] for ev in self.stream(req) if ev["chunk_type"] == "content_delta")

    # --------------------------------------------------------------------- structured output
    @override
    def structured_output(self, output_model: Type[T], prompt: Messages) -> Generator[dict[str, Union[T, Any]], None, None]:
        raw = self.complete(prompt)
        yield {"output": output_model.model_validate_json(raw)}
