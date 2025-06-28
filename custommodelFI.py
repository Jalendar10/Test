"""
Strands-compatible provider for custom_model_FI
Implements an OpenAI-style model wrapper that connects to a custom
LLM endpoint secured by HMAC.
"""

import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from typing import Any, Generator, Iterable, Type, TypeVar, Union, cast

import requests
from pydantic import BaseModel
from typing_extensions import override

from strands.types.content import Messages
from strands.types.models import OpenAIModel as SAOpenAIModel

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class CustomModelFI(SAOpenAIModel):
    """
    Custom Strands provider for the custom_model_FI backend LLM
    which uses an OpenAI-style interface and HMAC-secured endpoint.
    """

    REQUIRED_ENV = ("API_KEY", "API_SECRET", "BASE_URL")

    class CustomConfig(SAOpenAIModel.OpenAIConfig, total=False):
        pass

    def __init__(self, client_args: dict[str, Any] | None = None, **model_config: CustomModelFI.CustomConfig):
        self.config: CustomModelFI.CustomConfig = dict(model_config)
        self.api_key, self.api_secret, self.base_url = self._load_env()
        self.client = None  # placeholder for compatibility
        logger.debug("custom_model_FI initialized: base_url=%s", self.base_url)

    @staticmethod
    def _load_env() -> tuple[str, str, str]:
        import os
        missing = [var for var in CustomModelFI.REQUIRED_ENV if var not in os.environ]
        if missing:
            raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")
        return os.environ["API_KEY"], os.environ["API_SECRET"], os.environ["BASE_URL"]

    def _create_request_body(self, messages: Messages) -> dict[str, Any]:
        return {
            "model": self.config.get("model_id", "azure-openai-4o-mini-east"),
            "messages": [
                {"role": m["role"], "content": m["content"]} for m in messages  # type: ignore
            ],
            **(self.config.get("params") or {}),
            "stream": True
        }

    @override
    def update_config(self, **model_config: CustomConfig) -> None:  # type: ignore[override]
        self.config.update(model_config)

    @override
    def get_config(self) -> CustomConfig:  # type: ignore[override]
        return cast(CustomModelFI.CustomConfig, self.config)

    @override
    def stream(self, request: dict[str, Any]) -> Iterable[dict[str, Any]]:
        messages: Messages = request["messages"]
        body = self._create_request_body(messages)

        ts = int(time.time() * 1000)
        req_id = uuid.uuid4()

        signature_raw = self.api_key + str(req_id) + str(ts) + json.dumps(body, separators=(",", ":"))
        signature = base64.b64encode(
            hmac.new(self.api_secret.encode(), signature_raw.encode(), hashlib.sha256).digest()
        ).decode()

        headers = {
            "api-key": self.api_key,
            "Client-Request-Id": str(req_id),
            "Timestamp": str(ts),
            "Authorization": signature,
            "Accept": "application/json",
        }

        with requests.post(self.base_url, json=body, headers=headers, stream=True, timeout=90) as r:
            if r.status_code != 200:
                raise RuntimeError(f"custom_model_FI error {r.status_code}: {r.text}")

            yield {"chunk_type": "message_start"}
            yield {"chunk_type": "content_start", "data_type": "text"}

            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode()
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                if payload == "[DONE]":
                    break
                delta = payload.get("choices", [{}])[0].get("delta", {})
                if (content := delta.get("content")):
                    yield {"chunk_type": "content_delta", "data_type": "text", "data": content}

            yield {"chunk_type": "content_stop", "data_type": "text"}
            yield {"chunk_type": "message_stop", "data": "stop"}
            yield {"chunk_type": "metadata", "data": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}}

    def complete(self, messages: Messages) -> str:
        req = {"messages": messages}
        return "".join(ev["data"] for ev in self.stream(req) if ev["chunk_type"] == "content_delta")

    @override
    def structured_output(self, output_model: Type[T], prompt: Messages) -> Generator[dict[str, Union[T, Any]], None, None]:
        raw = self.complete(prompt)
        obj = output_model.model_validate_json(raw)
        yield {"output": obj}
