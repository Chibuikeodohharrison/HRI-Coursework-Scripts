
import os
from openai import OpenAI

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Set it in the VS Code terminal."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
    timeout=20.0,
)

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Marty, a friendly robot in a maths "
                    "classroom. Speak naturally and briefly."
                ),
            },
            {
                "role": "user",
                "content": "Hello Marty. What can you help me with?",
            },
        ],
        max_tokens=100,
    )

    print("\nMarty:", response.choices[0].message.content)

except Exception as error:
    print("Groq request failed:", error)
