

#basecode

import os
import uuid
import json
import base64
import hashlib
import hmac
import requests
import time
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

def load_environment():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    base_url = os.getenv("BASE_URL")

    if not api_key or not api_secret or not base_url:
        raise ValueError("Missing one or more environment variables: API_KEY, API_SECRET, BASE_URL")

    return api_key, api_secret, base_url

def create_request_body(conversation_messages, agent_behavior):
    # Include the agent behavior as a system message
    serialized_messages = [
        {
            "role": "system",
            "content": agent_behavior
        }
    ] + [
        {
            "role": "user" if isinstance(message, HumanMessage) else "assistant",
            "content": message.content
        }
        for message in conversation_messages
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
    hmac_source_data = api_key + str(request_id) + str(timestamp) + json.dumps(request_body)
    computed_hash = hmac.new(api_secret.encode(), hmac_source_data.encode(), hashlib.sha256)
    return base64.b64encode(computed_hash.digest()).decode()

def send_request(request_body, hmac_signature, base_url, api_key, timestamp, request_id):
    headers = {
        "api-key": api_key,
        "Client-Request-Id": str(request_id),
        "Timestamp": str(timestamp),
        "Authorization": hmac_signature,
        "Accept": "application/json",
    }
    response = requests.post(base_url, headers=headers, json=request_body, stream=True)

    if response.status_code != 200:
        raise ValueError(f"Error: Received status code {response.status_code}\nResponse content: {response.content}")

    full_response = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                data = decoded_line[len("data: "):]
                if data != "[DONE]":
                    json_data = json.loads(data)
                    choices = json_data.get("choices", [])
                    if choices:
                        full_response += choices[0].get("delta", {}).get("content", "")

    return {"choices": [{"message": {"content": full_response}}]}

def process_request(messages, agent_behavior):
    api_key, api_secret, base_url = load_environment()
    request_body = create_request_body(messages, agent_behavior)
    timestamp = int(time.time() * 1000)
    request_id = uuid.uuid4()
    hmac_signature = create_hmac_signature(request_body, api_key, api_secret, timestamp, request_id)
    response = send_request(request_body, hmac_signature, base_url, api_key, timestamp, request_id)
    return response['choices'][0]['message']['content']




from langchain.memory import ConversationBufferMemory
import base_code

class AgentSystem:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.agent_behavior = ""
        self.restricted_content = ""  # To store the restricted content

    def set_behavior(self, behavior_data):
        """
        Process the collected data to set the agent behavior.
        Add a restriction to limit the agent's responses to the given content.
        """
        self.agent_behavior = behavior_data
        self.restricted_content = (
            "You are strictly limited to the provided content. "
            "Do not answer any questions or provide information outside of this context. "
            "Do not assume or expand abbreviations unless explicitly provided in the given data. "
            "you are not allowed to provide any information that is not present in the provided content which includes the abbreviations or full forms. "
            "If the information is not present, respond with 'I cannot provide that information.'"
            "you have right to take decisions based on the provided content. "
        )

    def process_user_input(self, user_input):
        """
        Process user input and generate a response based on memory and agent behavior.
        Ensure the response is restricted to the given content.
        """
        # Add user message to memory
        self.memory.chat_memory.add_user_message(user_input)

        # Generate a response based on memory and agent behavior
        response = ""
        try:
            # Combine the restricted content with the agent behavior
            combined_behavior = f"{self.agent_behavior}"
            base_response = base_code.process_request(self.memory.chat_memory.messages, combined_behavior)
            response += base_response
        except ValueError as e:
            response += f"Error processing request: {e}"

        # Add the response to memory
        self.memory.chat_memory.add_ai_message(response)

        return response
#agent_code.py
from langchain.memory import ConversationBufferMemory
import base_code
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
class AgentSystem:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.agent_behavior = ""

    def set_behavior(self, behavior_data):
        # Process the collected data to set the agent behavior
        self.agent_behavior = behavior_data
        print(f"Agent behavior set to: {self.agent_behavior}")

    def process_user_input(self, user_input):
        # Add user message to memory
        self.memory.chat_memory.add_user_message(user_input)

        # Generate a response based on memory and agent behavior
        response = ""
        try:
            base_response = base_code.process_request(self.memory.chat_memory.messages, self.agent_behavior)
            response += base_response
        except ValueError as e:
            response += f"Error processing request: {e}"

        # Add the response to memory
        self.memory.chat_memory.add_ai_message(response)

        return response

def main():
    print("Welcome to the Agent System!")

    # Initialize memory for the agent
    memory = ConversationBufferMemory(return_messages=True)

    # Ask the user how the agent should act
    agent_behavior = input("Please specify how the agent should act: ")
    print(f"Agent will act as: {agent_behavior}")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the agent system. Goodbye!")
            break

        # Add user message to memory
        memory.chat_memory.add_user_message(user_input)

        # Generate a response based on memory and agent behavior
        response = ""

        try:
            base_response = base_code.process_request(memory.chat_memory.messages, agent_behavior)
            response += base_response
        except ValueError as e:
            response += f"Error processing request: {e}"

        print(f"Agent Response: {response}")

        # Add the response to memory
        memory.chat_memory.add_ai_message(response)

if __name__ == "__main__":
    main()
