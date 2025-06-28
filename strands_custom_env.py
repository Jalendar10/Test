"""
Strands-compatible provider: **custom_model_FI**
-------------------------------------------------
Same interface as Strands’ built‑in ``OpenAIModel`` but proxies requests to a
company‑internal LLM endpoint that is secured with an HMAC signature.

🔑 **Secrets** are pulled from a local ``.env`` file (via *python‑dotenv*) so you
can run the agent without exporting environment variables every time.

Key features
~~~~~~~~~~~~
* Reads ``API_KEY``, ``API_SECRET``, ``BASE_URL`` from **.env** automatically.
* Streaming Server‑Sent‑Events → Strands chunk sequence (`message_start`, …).
* Supports `tool_calls`, `reasoning_content`, and propagates `finish_reason`.
* Provides ``complete()`` (blocking) and ``structured_output()`` helpers.
* Drop‑in provider name: **custom_model_FI** (declared in entry‑points).
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
    """Strands provider that mirrors the OpenAI API while using a custom backend."""

    # ------------------------------------------------------------------ static
    REQUIRED_ENV = ("API_KEY", "API_SECRET", "BASE_URL")

    class CustomConfig(TypedDict, total=False):
        """Model configuration accepted by CustomModelFI.

        Mirrors the keys used by the OpenAI provider so existing YAML
        or Python configs continue to work.
        """

        model_id: str
        params: dict[str, Any] | None
        """No extra keys; inherits ``model_id`` & ``params`` semantics."""

    # ------------------------------------------------------------------ init
    def __init__(self, client_args: dict[str, Any] | None = None, **model_config: CustomConfig):
        self.config: CustomModelFI.CustomConfig = dict(model_config)
        self.api_key, self.api_secret, self.base_url = self._load_secrets()
        self.client = None  # placeholder for reflection compatibility
        logger.debug("custom_model_FI initialised – endpoint=%s", self.base_url)

    # ------------------------------------------------------------------ env helper
    @staticmethod
    def _load_secrets() -> tuple[str, str, str]:
        """Load ``API_KEY``, ``API_SECRET`` and ``BASE_URL`` from .env or os.environ."""
        # Ensure .env is parsed only once even if multiple providers are instantiated
        if not getattr(CustomModelFI, "_dotenv_loaded", False):
            load_dotenv()  # looks for .env in CWD or parents
            CustomModelFI._dotenv_loaded = True  # type: ignore[attr-defined]

        missing = [v for v in CustomModelFI.REQUIRED_ENV if v not in os.environ]
        if missing:
            raise EnvironmentError(
                "Missing required variables in .env or environment: " + ", ".join(missing)
            )
        return os.environ["API_KEY"], os.environ["API_SECRET"], os.environ["BASE_URL"]

    # ------------------------------------------------ config passthrough (OpenAIModel API)
    @override
    def update_config(self, **model_config: CustomConfig) -> None:  # type: ignore[override]
        self.config.update(model_config)

    @override
    def get_config(self) -> CustomConfig:  # type: ignore[override]
        return cast(CustomModelFI.CustomConfig, self.config)

    # ------------------------------------------------------------------ request helpers
    def _create_body(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert Strands request → backend JSON body."""
        return {
            "model": self.config.get("model_id", "azure-openai-4o-mini-east"),
            "stream": True,
            **(self.config.get("params") or {}),
            "messages": request["messages"],
        }

    def _headers(self, body: dict[str, Any]) -> dict[str, str]:
        ts = int(time.time() * 1000)
        req_id = uuid.uuid4()
        raw = self.api_key + str(req_id) + str(ts) + json.dumps(body, separators=(",", ":"))
        sig = base64.b64encode(hmac.new(self.api_secret.encode(), raw.encode(), hashlib.sha256).digest()).decode()
        return {
            "api-key": self.api_key,
            "Client-Request-Id": str(req_id),
            "Timestamp": str(ts),
            "Authorization": sig,
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------ streaming interface
    @override
    def stream(self, request: dict[str, Any]) -> Iterable[dict[str, Any]]:
        body = self._create_body(request)
        headers = self._headers(body)

        with requests.post(self.base_url, json=body, headers=headers, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"custom_model_FI HTTP {resp.status_code}: {resp.text}")

            # ─── start events ────────────────────────────────────────────────
            yield {"chunk_type": "message_start"}
            yield {"chunk_type": "content_start", "data_type": "text"}

            tool_calls: dict[int, list[Any]] = {}
            finish_reason: str | None = None

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                payload = json.loads(data)
                choice = payload.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                # content
                if (text := delta.get("content")):
                    yield {"chunk_type": "content_delta", "data_type": "text", "data": text}

                # reasoning_content (optional)
                if (rc := delta.get("reasoning_content")):
                    yield {"chunk_type": "content_delta", "data_type": "reasoning_content", "data": rc}

                # tool calls
                for tc in delta.get("tool_calls", []):
                    tool_calls.setdefault(tc["index"], []).append(tc)

                finish_reason = choice.get("finish_reason") or finish_reason

            # ─── end content ────────────────────────────────────────────────
            yield {"chunk_type": "content_stop", "data_type": "text"}

            # flush tool deltas
            for deltas in tool_calls.values():
                yield {"chunk_type": "content_start", "data_type": "tool", "data": deltas[0]}
                for d in deltas:
                    yield {"chunk_type": "content_delta", "data_type": "tool", "data": d}
                yield {"chunk_type": "content_stop", "data_type": "tool"}

            yield {"chunk_type": "message_stop", "data": finish_reason or "stop"}
            yield {"chunk_type": "metadata", "data": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}}

    # ------------------------------------------------------------------ sync helper
    def complete(self, messages: Messages) -> str:
        req = {"messages": messages}
        return "".join(e["data"] for e in self.stream(req) if e["chunk_type"] == "content_delta")

    # ------------------------------------------------------------------ structured output
    @override
    def structured_output(
        self, output_model: Type[T], prompt: Messages
    ) -> Generator[dict[str, Union[T, Any]], None, None]:
        data = self.complete(prompt)
        yield {"output": output_model.model_validate_json(data)}
