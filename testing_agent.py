import os
import uuid
import json
import base64
import hashlib
import hmac
import requests
import time
import warnings

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Ignore warnings
warnings.filterwarnings("ignore")

# -------------------- Agent Code --------------------
class AgentSystem:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.agent_behavior = ""
        self.restricted_content = ""

    def set_behavior(self, behavior_data: str):
        self.agent_behavior = behavior_data
        self.restricted_content = (
            "You are strictly limited to the provided content. "
            "Do not answer questions outside that context. "
            "Do not expand abbreviations unless specified. "
            "If information is missing, say: 'I cannot provide that information.'"
        )

    def _combined_behavior(self) -> str:
        return f"{self.agent_behavior}\n\n{self.restricted_content}"

    def process_user_input(self, user_input: str) -> str:
        self.memory.chat_memory.add_user_message(user_input)
        try:
            resp = process_request(
                self.memory.chat_memory.messages,
                self._combined_behavior()
            )
        except ValueError as e:
            resp = f"Error: {e}"
        self.memory.chat_memory.add_ai_message(resp)
        return resp

# -------------------- API Request Logic --------------------
def load_environment():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    base_url = os.getenv("BASE_URL")
    if not all([api_key, api_secret, base_url]):
        raise ValueError("Missing environment variables: API_KEY, API_SECRET, BASE_URL")
    return api_key, api_secret, base_url

def create_request_body(messages, agent_behavior):
    serialized = [{"role": "system", "content": agent_behavior}] + [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in messages
    ]
    return {
        "model": "azure-openai-4o-mini-east",
        "messages": serialized,
        "frequency_penalty": 0,
        "max_tokens": 1000,
        "n": 1,
        "presence_penalty": 0,
        "response_format": {"type": "text"},
        "stream": True,
        "temperature": 1,
        "top_p": 1
    }

def create_hmac_signature(body, api_key, api_secret, timestamp, request_id):
    source = api_key + str(request_id) + str(timestamp) + json.dumps(body)
    digest = hmac.new(api_secret.encode(), source.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

def send_request(body, signature, url, api_key, timestamp, request_id):
    headers = {
        "api-key": api_key,
        "Client-Request-Id": str(request_id),
        "Timestamp": str(timestamp),
        "Authorization": signature,
        "Accept": "application/json"
    }
    resp = requests.post(url, headers=headers, json=body, stream=True)
    if resp.status_code != 200:
        raise ValueError(f"HTTP {resp.status_code}: {resp.content}")
    output = ""
    for line in resp.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith("data: "):
                data = decoded[len("data: "):]
                if data != "[DONE]":
                    chunk = json.loads(data)
                    output += chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
    return {"choices": [{"message": {"content": output}}]}

def process_request(messages, agent_behavior):
    api_key, api_secret, url = load_environment()
    body = create_request_body(messages, agent_behavior)
    timestamp = int(time.time() * 1000)
    request_id = uuid.uuid4()
    signature = create_hmac_signature(body, api_key, api_secret, timestamp, request_id)
    result = send_request(body, signature, url, api_key, timestamp, request_id)
    return result["choices"][0]["message"]["content"]

# -------------------- Interactive CLI --------------------
def main():
    print("🧠 Welcome to the Agent System (Restricted Mode)")
    agent = AgentSystem()
    agent.set_behavior(input("📝 Specify agent behavior: "))

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        reply = agent.process_user_input(user_input)
        print("🤖 Agent:", reply)

# -------------------- Run & Test --------------------
if __name__ == "__main__":
    # To run tests, uncomment the test block below
    main()

    # # -------------------- Inline Test --------------------
    # print("\n--- Running test ---")
    # def fake_send_request(body, sig, url, key, ts, req_id):
    #     user_msg = next((m["content"] for m in body["messages"] if m["role"] == "user"), "")
    #     return {"choices": [{"message": {"content": "Echo: " + user_msg}}]}
    #
    # send_request_backup = send_request
    # send_request = fake_send_request  # mock
    #
    # test_agent = AgentSystem()
    # test_agent.set_behavior("Only answer about fruits.")
    # assert "I cannot provide" in test_agent.process_user_input("What is an airplane?")
    # print("✅ Test passed")
    #
    # send_request = send_request_backup  # restore
