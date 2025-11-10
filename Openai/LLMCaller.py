import sys
from openai import OpenAI
from google import genai

LLM_MAP = {
    "gpt3.5": "gpt-3.5-turbo-0125",
    "gpt4.1pro": "gpt-4.1-2025-04-14"
}

class LLMCaller:
    """
    Wrapper for OpenAI responses API (v1.x SDK).
    """
    def __init__(self, llm: str = "gpt3.5"):
        if llm not in LLM_MAP:
            raise ValueError(f"Unsupported model name: {llm}")
        api_key = "YOUR API KEY"
        self.model_name = LLM_MAP[llm]
        self.client = OpenAI(api_key=api_key)

    def query(self, instruction: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=instruction
        )
        return response.output_text