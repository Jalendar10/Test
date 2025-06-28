"""
Strands-compatible provider: **custom_model_FI**
-------------------------------------------------
Provides a drop‑in replacement for Strands’ ``OpenAIModel`` but signs every
request with an **HMAC** scheme that must *exactly* match the reference
implementation found in *base_code.py* (no custom JSON separators, identical
body structure).

Key adjustments (2025‑06‑28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **HMAC source string** now uses *default* ``json.dumps`` (includes spaces) —
  this fixes ``steps.hmac.HmacVerificationFailed``.
* Request body layout is **byte‑for‑byte** identical to *base_code.py*:
  - ``frequency_penalty``
  - ``presence_penalty``
  - ``n``
  - ``response_format``
  - ``temperature`` / ``top_p``
  - plus any overrides in ``params``.
* The body is NOT re‑ordered or minified; insertion order is preserved.

With these changes the backend should compute the same HMAC and return 200.
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
    """Strands provider mirroring OpenAI but using company‑internal HMAC auth."""

    REQUIRED_ENV = ("API_KEY", "API_SECRET", "BASE_URL")

    class CustomConfig(TypedDict, total=False):
        model_id: str
        params: dict[str, Any] | None

    # --------------------------------------------------------------------- init
    def __init__(self, client_args: dict[str, Any] | None = None, **model_config: "CustomModelFI.CustomConfig"):
        self.config: CustomModelFI.CustomConfig = dict(model_config)
        self.api_key, self.api_secret, self.base_url = self._load_secrets()
        self.client = None
        logger.debug("custom_model_FI initialised – endpoint=%s", self.base_url)

    # ------------------------------------------------------------------ env helper
    @staticmethod
    def _load_secrets() -> tuple[str, str, str]:
        if not getattr(CustomModelFI, "_dotenv_loaded", False):
            load_dotenv()
            CustomModelFI._dotenv_loaded = True  # type: ignore[attr-defined]
        missing = [v for v in CustomModelFI.REQUIRED_ENV if v not in os.environ]
        if missing:
            raise EnvironmentError("Missing variables: " + ", ".join(missing))
        return os.environ["API_KEY"], os.environ["API_SECRET"], os.environ["BASE_URL"]

    # ------------------------------------------------------------------- config I/F
    @override
    def update_config(self, **model_config: "CustomModelFI.CustomConfig") -> None:  # type: ignore[override]
        self.config.update(model_config)

    @override
    def get_config(self) -> "CustomModelFI.CustomConfig":  # type: ignore[override]
        return cast(CustomModelFI.CustomConfig, self.config)

    # ------------------------------------------------------------------- helpers
    def _create_body(self, request: dict[str, Any]) -> dict[str, Any]:
        """Replicates *base_code.create_request_body* exactly."""
        p = self.config.get("params") or {}
        return {
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

    def _headers(self, body: dict[str, Any]) -> dict[str, str]:
        ts = int(time.time() * 1000)
        req_id = uuid.uuid4()
        # ✨ IMPORTANT: NO separators arg → default JSON with spaces
        raw_body_json = json.dumps(body)
        source = f"{self.api_key}{req_id}{ts}{raw_body_json}"
        sig = base64.b64encode(hmac.new(self.api_secret.encode(), source.encode(), hashlib.sha256).digest()).decode()
        return {
            "api-key": self.api_key,
            "Client-Request-Id": str(req_id),
            "Timestamp": str(ts),
            "Authorization": sig,
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------- stream
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

    # ------------------------------------------------------------------- sync
    def complete(self, messages: Messages) -> str:
        return "".join(c["data"] for c in self.stream({"messages": messages}) if c["chunk_type"] == "content_delta")

    # ------------------------------------------------------------------- structured output
    @override
    def structured_output(self, output_model: Type[T], prompt: Messages) -> Generator[dict[str, Union[T, Any]], None, None]:
        raw = self.complete(prompt)
        yield {"output": output_model.model_validate_json(raw)}


calling
from strands import Agent
from strands.models.custom_model_FI import CustomModelFI
from strands_tools import calculator

model = CustomModelFI(
   
    model_id="azure-openai-4o-mini-east",       # ✅ matches your backend model ID
        params={
            "temperature": 0.7,
            "max_tokens": 500
        }
    )
    
agent = Agent(
    model=model,tools=[calculator])

response= agent("what is 2+2")
print(response)        
error
PS C:\Users\F41n1so\Downloads\MCP_Test> & c:/Users/F41n1so/Downloads/MCP_Test/venv/Scripts/python.exe c:/Users/F41n1so/Downloads/MCP_Test/weather/custom_model_FI_TEST.py
Traceback (most recent call last):
  File "c:\Users\F41n1so\Downloads\MCP_Test\weather\custom_model_FI_TEST.py", line 17, in <module>
    response= agent("what is 2+2")
              ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\F41n1so\Downloads\MCP_Test\venv\Lib\site-packages\strands\agent\agent.py", line 358, in __call__
    result = self._run_loop(prompt, kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\F41n1so\Downloads\MCP_Test\venv\Lib\site-packages\strands\agent\agent.py", line 462, in _run_loop
    return self._execute_event_loop_cycle(invocation_callback_handler, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\F41n1so\Downloads\MCP_Test\venv\Lib\site-packages\strands\agent\agent.py", line 490, in _execute_event_loop_cycle
    stop_reason, message, metrics, state = event_loop_cycle(
                                           ^^^^^^^^^^^^^^^^^
  File "C:\Users\F41n1so\Downloads\MCP_Test\venv\Lib\site-packages\strands\event_loop\event_loop.py", line 190, in event_loop_cycle
    raise e
  File "C:\Users\F41n1so\Downloads\MCP_Test\venv\Lib\site-packages\strands\event_loop\event_loop.py", line 148, in event_loop_cycle
    stop_reason, message, usage, metrics, kwargs["request_state"] = stream_messages(
                                                                    ^^^^^^^^^^^^^^^^
  File "C:\Users\F41n1so\Downloads\MCP_Test\venv\Lib\site-packages\strands\event_loop\streaming.py", line 340, in stream_messages
    return process_stream(chunks, callback_handler, messages, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\F41n1so\Downloads\MCP_Test\venv\Lib\site-packages\strands\event_loop\streaming.py", line 290, in process_stream
    for chunk in chunks:
                 ^^^^^^
  File "C:\Users\F41n1so\Downloads\MCP_Test\venv\Lib\site-packages\strands\types\models\model.py", line 115, in converse
    for event in response:
                 ^^^^^^^^
  File "C:\Users\F41n1so\Downloads\MCP_Test\venv\Lib\site-packages\strands\models\custom_model_FI.py", line 125, in stream
    raise RuntimeError(f"custom_model_FI HTTP {resp.status_code}: {resp.text}")
RuntimeError: custom_model_FI HTTP 404: {"fault":{"faultstring":"Requested resource cannot be found","detail":{"errorcode":"steps.apim.custom.ResourceNotFound"}}}
PS C:\Users\F41n1so\Downloads\MCP_Test> 
