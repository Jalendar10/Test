# tools.py
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

# Register tools by name
tool_registry = {t.name: t for t in [add, multiply]}


# base_code.py
import os
import uuid
import json
import base64
import hashlib
import hmac
import requests
import time
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Optional

class CustomModel(BaseChatModel):
    def __init__(self, agent_behavior: str):
        super().__init__()
        self.agent_behavior = agent_behavior

    def _call(self, messages: List, stop: Optional[List[str]] = None) -> str:
        key, secret, url = load_environment()
        body = create_request_body(messages, self.agent_behavior)
        ts = int(time.time() * 1000)
        req_id = uuid.uuid4()
        signature = create_hmac_signature(body, key, secret, ts, req_id)
        result = send_request(body, signature, url, key, ts, req_id)
        return result["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "custom-model"

def load_environment():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    base_url = os.getenv("BASE_URL")
    if not api_key or not api_secret or not base_url:
        raise ValueError("Missing environment variables.")
    return api_key, api_secret, base_url

def create_request_body(conversation_messages, agent_behavior):
    serialized_messages = [
        {"role": "system", "content": agent_behavior}
    ] + [
        {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
        for msg in conversation_messages
    ]
    return {
        "model": "azure-openai-4o-mini-east",
        "messages": serialized_messages,
        "frequency_penalty": 0,
        "max_tokens": 1000,
        "n": 1,
        "presence_penalty": 0,
        "response_format": {"type": "text"},
        "stream": True,
        "temperature": 1,
        "top_p": 1
    }

def create_hmac_signature(request_body, api_key, api_secret, timestamp, request_id):
    source = api_key + str(request_id) + str(timestamp) + json.dumps(request_body)
    return base64.b64encode(hmac.new(api_secret.encode(), source.encode(), hashlib.sha256).digest()).decode()

def send_request(body, signature, base_url, api_key, timestamp, request_id):
    headers = {
        "api-key": api_key,
        "Client-Request-Id": str(request_id),
        "Timestamp": str(timestamp),
        "Authorization": signature,
        "Accept": "application/json",
    }
    response = requests.post(base_url, headers=headers, json=body, stream=True)
    if response.status_code != 200:
        raise ValueError(f"HTTP {response.status_code}: {response.content}")

    content = ""
    for line in response.iter_lines():
        if line and line.startswith(b"data: "):
            json_data = json.loads(line[6:].decode())
            content += json_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
    return {"choices": [{"message": {"content": content}}]}

def parse_tool_calls(response_text):
    try:
        obj = json.loads(response_text)
        return obj.get("tool_calls", []), obj.get("content", "")
    except json.JSONDecodeError:
        return [], response_text


# invoke_custom_model.py
from langchain_core.messages import HumanMessage
from base_code import CustomModel, parse_tool_calls
from tools import tool_registry
from langchain_core.messages import ToolMessage, AIMessage

# Step 1: Init model and bind tools
agent_behavior = "You are a helpful assistant. Use tools for math operations."
llm = CustomModel(agent_behavior)
llm_with_tools = llm.bind_tools(list(tool_registry.values()))

# Step 2: Start conversation
query = "What is 3 * 12? Also, what is 11 + 49?"
messages = [HumanMessage(content=query)]

# Step 3: Get tool calls
ai_msg = llm_with_tools.invoke(messages)
print("Tool Calls:", ai_msg.tool_calls)
messages.append(ai_msg)

# Step 4: Run tools
for tool_call in ai_msg.tool_calls:
    tool_func = tool_registry[tool_call["name"]]
    tool_msg = tool_func.invoke(tool_call)
    messages.append(tool_msg)

# Step 5: Final answer with tool results
final_response = llm_with_tools.invoke(messages)
print("Final Answer:", final_response.content)
