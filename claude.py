"""Custom model provider for Strands framework.

Based on your existing base_code.py implementation.
"""

import logging
import os
import uuid
import json
import base64
import hashlib
import hmac
import requests
import time
from typing import Any, Generator, Iterable, Optional, Protocol, Type, TypedDict, TypeVar, Union, cast
from dotenv import load_dotenv
from pydantic import BaseModel
from typing_extensions import Unpack, override

# Assuming these imports exist in your strands framework
from ..types.content import Messages
from ..types.models import BaseModel as SABaseModel  # Adjust import as needed

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Client(Protocol):
    """Protocol defining the custom API interface."""

    def chat_completions_create(self, **kwargs: Any) -> Any:
        """Chat completions interface."""
        ...


class CustomClient:
    """Custom client implementation for your API."""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
    
    def _create_request_body(self, messages: list, **params: Any) -> dict:
        """Create request body for the API."""
        return {
            "model": params.get("model", "azure-openai-4o-mini-east"),
            "messages": messages,
            "frequency_penalty": params.get("frequency_penalty", 0),
            "max_tokens": params.get("max_tokens", 1000),
            "n": params.get("n", 1),
            "presence_penalty": params.get("presence_penalty", 0),
            "response_format": params.get("response_format", {"type": "text"}),
            "stream": params.get("stream", True),
            "temperature": params.get("temperature", 1),
            "top_p": params.get("top_p", 1)
        }
    
    def _create_hmac_signature(self, request_body: dict, timestamp: int, request_id: str) -> str:
        """Create HMAC signature for authentication."""
        hmac_source_data = self.api_key + str(request_id) + str(timestamp) + json.dumps(request_body)
        computed_hash = hmac.new(self.api_secret.encode(), hmac_source_data.encode(), hashlib.sha256)
        return base64.b64encode(computed_hash.digest()).decode()
    
    def chat_completions_create(self, messages: list, **params: Any) -> dict:
        """Send request to the custom API."""
        request_body = self._create_request_body(messages, **params)
        timestamp = int(time.time() * 1000)
        request_id = str(uuid.uuid4())
        hmac_signature = self._create_hmac_signature(request_body, timestamp, request_id)
        
        headers = {
            "api-key": self.api_key,
            "Client-Request-Id": request_id,
            "Timestamp": str(timestamp),
            "Authorization": hmac_signature,
            "Accept": "application/json",
        }
        
        response = requests.post(self.base_url, headers=headers, json=request_body, stream=True)
        
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
        
        return {"choices": [{"message": {"content": full_response}, "finish_reason": "stop"}]}


class CustomModel(SABaseModel):
    """Custom model provider implementation."""

    client: CustomClient

    class CustomConfig(TypedDict, total=False):
        """Configuration options for Custom models.

        Attributes:
            model_id: Model ID (e.g., "azure-openai-4o-mini-east").
            params: Model parameters (e.g., max_tokens).
            api_key: API key for authentication.
            api_secret: API secret for authentication.
            base_url: Base URL for the API.
        """

        model_id: str
        params: Optional[dict[str, Any]]
        api_key: Optional[str]
        api_secret: Optional[str]
        base_url: Optional[str]

    def __init__(self, client_args: Optional[dict[str, Any]] = None, **model_config: Unpack[CustomConfig]) -> None:
        """Initialize provider instance.

        Args:
            client_args: Arguments for the custom client.
            **model_config: Configuration options for the custom model.
        """
        self.config = dict(model_config)

        logger.debug("config=<%s> | initializing", self.config)

        # Load environment variables if not provided in config
        load_dotenv()
        api_key = self.config.get("api_key") or os.getenv("API_KEY")
        api_secret = self.config.get("api_secret") or os.getenv("API_SECRET")
        base_url = self.config.get("base_url") or os.getenv("BASE_URL")

        if not api_key or not api_secret or not base_url:
            raise ValueError("Missing one or more required parameters: api_key, api_secret, base_url")

        client_args = client_args or {}
        self.client = CustomClient(api_key=api_key, api_secret=api_secret, base_url=base_url)

    @override
    def update_config(self, **model_config: Unpack[CustomConfig]) -> None:  # type: ignore[override]
        """Update the custom model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> CustomConfig:
        """Get the custom model configuration.

        Returns:
            The custom model configuration.
        """
        return cast(CustomModel.CustomConfig, self.config)

    @override
    def stream(self, request: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Send the request to the custom model and get the streaming response.

        Args:
            request: The formatted request to send to the custom model.

        Returns:
            An iterable of response events from the custom model.
        """
        # Extract parameters from request
        messages = request.get("messages", [])
        params = self.config.get("params", {})
        
        # Merge request parameters with config parameters
        merged_params = {**params, **{k: v for k, v in request.items() if k != "messages"}}
        
        response = self.client.chat_completions_create(messages=messages, **merged_params)

        yield {"chunk_type": "message_start"}
        yield {"chunk_type": "content_start", "data_type": "text"}

        # Since your API doesn't support true streaming in the same way as OpenAI,
        # we'll simulate it by yielding the complete response
        if response.get("choices") and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            yield {"chunk_type": "content_delta", "data_type": "text", "data": content}
            finish_reason = response["choices"][0].get("finish_reason", "stop")
        else:
            finish_reason = "error"

        yield {"chunk_type": "content_stop", "data_type": "text"}
        yield {"chunk_type": "message_stop", "data": finish_reason}

        # Add metadata if available
        if "usage" in response:
            yield {"chunk_type": "metadata", "data": response["usage"]}

    @override
    def structured_output(
        self, output_model: Type[T], prompt: Messages
    ) -> Generator[dict[str, Union[T, Any]], None, None]:
        """Get structured output from the model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.

        Yields:
            Model events with the last being the structured output.
        """
        # Format the request
        formatted_request = super().format_request(prompt)
        messages = formatted_request["messages"]
        
        # Add instruction for structured output
        if messages:
            system_message = {
                "role": "system",
                "content": f"Please respond with valid JSON that matches this schema: {output_model.model_json_schema()}"
            }
            messages.insert(0, system_message)
        
        params = self.config.get("params", {})
        response = self.client.chat_completions_create(messages=messages, **params)

        if response.get("choices") and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            try:
                # Try to parse the response as JSON and validate against the model
                json_data = json.loads(content)
                parsed = output_model.model_validate(json_data)
                yield {"output": parsed}
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Failed to parse structured output: {e}")
        else:
            raise ValueError("No valid response found in the custom API response.")


# Usage example (similar to your existing pattern)
"""
from strands import Agent
from your_custom_module import CustomModel
from strands_tools import calculator

model = CustomModel(
    model_id="azure-openai-4o-mini-east",
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    },
    # These can also be loaded from environment variables
    api_key="your_api_key",
    api_secret="your_api_secret", 
    base_url="your_base_url"
)

agent = Agent(model=model, tools=[calculator])
response = agent("What is 2+2")
print(response)
"""
