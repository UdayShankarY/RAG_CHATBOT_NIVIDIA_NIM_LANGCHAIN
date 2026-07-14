from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

query = input("Ask: ").strip()

if not query:
    print("Please enter a question.")
    exit()

completion = client.chat.completions.create(
    model="meta/llama-3.1-8b-instruct",
    messages=[
    {
        "role": "system",
        "content": (
            "You are an expert AI assistant. "
            "Answer accurately and concisely. "
            "If the question is about AI, machine learning, LangChain, or RAG, "
            "use the standard industry meaning."
        )
    },
    {
        "role": "user",
        "content": query
    }
],
    temperature=0.5,
    top_p=1,
    max_tokens=1024,
    stream=True
)

print("\nAssistant:\n")

for chunk in completion:
    if chunk.choices:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)

print()