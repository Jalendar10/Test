from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def main():
    # Configure client
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
                "timeout": 30.0
            }
        }
    )

    # Get and verify tools
    tools = await client.get_tools()
    print("Available tools:", [tool.name for tool in tools])

    # Create tool mapping for direct access
    tool_map = {tool.name: tool for tool in tools}

    # Test cases with expected arguments
    test_cases = [
        {
            "tool_name": "multiple",
            "args": {"a": 8, "b": 12},
            "expected": 96
        },
        {
            "tool_name": "add",
            "args": {"a": 15, "b": 20},
            "expected": 35
        },
        {
            "tool_name": "get_weather",
            "args": {"location": "California"},
            "expected": "It's always raining in California"
        }
    ]

    for case in test_cases:
        tool_name = case["tool_name"]
        if tool_name not in tool_map:
            print(f"\nError: Tool '{tool_name}' not found")
            continue

        try:
            print(f"\nCalling {tool_name} with args: {case['args']}")
            result = await tool_map[tool_name].ainvoke(case["args"])
            print(f"Result: {result}")
            
            # Verify the result matches expected
            if result == case["expected"]:
                print("✓ Test passed")
            else:
                print(f"✗ Test failed. Expected {case['expected']}, got {result}")
                
        except Exception as e:
            print(f"Error calling {tool_name}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
