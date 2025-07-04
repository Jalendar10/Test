Folder layout
text
Copy
Edit
your_project/
├─ base_code.py
├─ run_agent.py
└─ strands/
    └─ models/
        ├─ __init__.py          # add one import line
        └─ custom_agent.py      # <— new provider
Tip
If you installed the SDK with pip install strands-agents, you only need the strands/models sub-package from below; the rest is already inside your site-packages directory. Built-in providers such as openai.py and bedrock.py live there too 
raw.githubusercontent.com
raw.githubusercontent.com
.

1️⃣ base_code.py (unchanged)
python
Copy
Edit
import os, uuid, json, base64, hashlib, hmac, requests, time
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

def load_environment() -> tuple[str, str, str]:
    load_dotenv()
    api_key   = os.getenv("API_KEY")
    api_sec   = os.getenv("API_SECRET")
    base_url  = os.getenv("BASE_URL")
    if not all([api_key, api_sec, base_url]):
        raise ValueError("Missing API_KEY, API_SECRET or BASE_URL")
    return api_key, api_sec, base_url

def create_request_body(conv_msgs, agent_behavior) -> dict:
    msgs = [{"role": "system", "content": agent_behavior}] + [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant",
         "content": m.content}
        for m in conv_msgs
    ]
    return {
        "model": "azure-openai-4o-mini-east",
        "messages": msgs,
        "max_tokens": 1000,
        "stream": True,
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "text"},
        "n": 1,
    }

def create_hmac_signature(body, api_key, api_sec, ts, req_id):
    src = api_key + str(req_id) + str(ts) + json.dumps(body)
    return base64.b64encode(hmac.new(api_sec.encode(), src.encode(),
                                     hashlib.sha256).digest()).decode()

def send_request(body, sig, url, api_key, ts, req_id):
    headers = {
        "api-key": api_key,
        "Client-Request-Id": str(req_id),
        "Timestamp": str(ts),
        "Authorization": sig,
        "Accept": "application/json",
    }
    resp = requests.post(url, headers=headers, json=body, stream=True)
    if resp.status_code != 200:
        raise ValueError(f"{resp.status_code}: {resp.content}")
    full = ""
    for line in resp.iter_lines():
        if line:
            chunk = line.decode()
            if chunk.startswith("data: "):
                data = chunk[6:]
                if data != "[DONE]":
                    choice = json.loads(data)["choices"][0]
                    full  += choice.get("delta", {}).get("content", "")
    return full

def process_request(messages, agent_behavior) -> str:
    api_key, api_sec, url = load_environment()
    body = create_request_body(messages, agent_behavior)
    ts   = int(time.time() * 1000)
    rid  = uuid.uuid4()
    sig  = create_hmac_signature(body, api_key, api_sec, ts, rid)
    return send_request(body, sig, url, api_key, ts, rid)
2️⃣ strands/models/custom_agent.py
python
Copy
Edit
from __future__ import annotations
import os, logging
from typing import Any, AsyncGenerator, Optional, TypedDict
from typing_extensions import Unpack, override

from strands.types.models import Model
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec
from strands.types.exceptions import ContextWindowOverflowException

import base_code                     # ← your helper

log = logging.getLogger(__name__)

class CustomAgentModel(Model):
    """Provider that forwards Strands messages to `base_code.process_request`."""

    class Config(TypedDict, total=False):
        model_name: str
        max_tokens: int
        temperature: float

    # ---------- ctor --------------------------------------------------
    def __init__(self, *, api_key: str, api_secret: str, base_url: str,
                 **cfg: Unpack[Config]) -> None:
        os.environ.setdefault("API_KEY", api_key)
        os.environ.setdefault("API_SECRET", api_secret)
        os.environ.setdefault("BASE_URL", base_url)
        self.config: CustomAgentModel.Config = {}
        self.update_config(**cfg)
        log.debug("CustomAgentModel cfg=%s", self.config)

    # ---------- dynamic config hooks ---------------------------------
    @override
    def update_config(self, **cfg: Unpack[Config]) -> None:  # type: ignore[override]
        self.config.update(cfg)

    @override
    def get_config(self) -> Config:  # type: ignore[override]
        return self.config

    # ---------- Strands → backend request ----------------------------
    @override
    def format_request(             # type: ignore[override]
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        return {"messages": messages,
                "agent_behavior": system_prompt or ""}

    # ---------- backend → StreamEvent --------------------------------
    @override
    def format_chunk(self, ev: dict[str, Any]) -> StreamEvent:  # type: ignore[override]
        return ev                           # we emit spec-compliant events below

    # ---------- main async stream ------------------------------------
    @override
    async def stream(   # type: ignore[override]
        self, request: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        try:
            text = base_code.process_request(
                request["messages"], request["agent_behavior"]
            )
        except Exception as exc:
            raise ContextWindowOverflowException() from exc

        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {}}
        yield {"contentBlockDelta": {"delta": {"text": text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
3️⃣ strands/models/__init__.py (add)
python
Copy
Edit
# keep existing imports …
from .custom_agent import CustomAgentModel   # NEW
__all__ += ["CustomAgentModel"]
Built-in providers—OpenAIModel, BedrockModel, etc.—stay untouched 
raw.githubusercontent.com
raw.githubusercontent.com
.

4️⃣ run_agent.py (smoke test)
python
Copy
Edit
from strands import Agent
from strands.models import CustomAgentModel

model = CustomAgentModel(
    api_key    = "YOUR_KEY",
    api_secret = "YOUR_SECRET",
    base_url   = "https://your-endpoint.example.com/v1/chat/completions",
    model_name = "azure-openai-4o-mini-east",
)

agent = Agent(model=model)
print(agent("Hello, Custom Provider!"))
Run:

bash
Copy
Edit
python -u run_agent.py
You should see the response streamed through Strands’ conversation manager, exactly like the built-in providers.

Why this works (and what you can extend later)
Provider interface – mirrors the async API the SDK introduced in its latest commit 
raw.githubusercontent.com
.

Minimal five-event stream – messageStart
