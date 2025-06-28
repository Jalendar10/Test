# basecode_smoketest.py
import json
import pprint
import time
import uuid
from basecode import (
    load_environment,
    create_request_body,
    create_hmac_signature,
    send_request,
)

from langchain.schema import HumanMessage

def main() -> None:
    # 1️⃣  Load creds + URL ---------------------------------------------------
    api_key, api_secret, base_url = load_environment()
    print("✅ Env loaded:")
    print("   BASE_URL =", base_url)

    # 2️⃣  Build a minimal conversation --------------------------------------
    messages = [HumanMessage(content="Hello, what is 2 + 2?")]
    agent_behavior = "You are a helpful assistant."

    request_body = create_request_body(messages, agent_behavior)
    print("\n📝 Request body:")
    print(json.dumps(request_body, indent=2))

    # 3️⃣  Sign exactly like basecode.send_request ----------------------------
    timestamp = int(time.time() * 1000)
    request_id = uuid.uuid4()
    signature = create_hmac_signature(
        request_body, api_key, api_secret, timestamp, request_id
    )
    print("\n🔑 HMAC signature =", signature[:12], "...")

    # 4️⃣  Fire the request & stream the result ------------------------------
    print("\n🚀 Sending request ...\n")
    response = send_request(
        request_body, signature, base_url, api_key, timestamp, request_id
    )

    print("📨 Full response:")
    pprint.pprint(response, width=100)

if __name__ == "__main__":
    main()
