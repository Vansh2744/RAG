from langchain_litellm import ChatLiteLLM
from dotenv import load_dotenv
import os

load_dotenv()

primary_llm = ChatLiteLLM(model="groq/groq-x")

fallback_llm1 = ChatLiteLLM(model=f"groq/{os.getenv('GROQ_MODEL')}")
fallback_llm2 = ChatLiteLLM(model=f"gemini/{os.getenv('GOOGLE_MODEL')}")

robust_llm = primary_llm.with_fallbacks([fallback_llm1, fallback_llm2])

res = robust_llm.invoke('hi')

print(res.content)