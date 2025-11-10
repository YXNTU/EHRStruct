import sys

from google import genai

LLM_MAP = {
    "gemini2.5": "gemini-2.5-flash-preview-04-17",
    "gemini2.0": "gemini-2.0-flash",
    "gemini1.5": "gemini-1.5-flash",
}

class LLMCaller:
    """
    Wrapper for Google's Gemini API using genai.Client interface.

    Args:
        llm (str): Logical model key (e.g., 'gemini2.0').
        api_key (str): Your API key from Google AI Studio.
    """
    def __init__(self, llm: str = "gemini2.0", api_key: str = None):
        if llm not in LLM_MAP:
            raise ValueError(f"Unsupported model name: {llm}")
        self.model_name = LLM_MAP[llm]
        self.api_key = api_key or "YOUR API KEY"
        self.client = genai.Client(api_key=self.api_key)

    def query(self, instruction: str) -> str:
        """
        Sends a prompt and returns the model's response text.

        Args:
            instruction (str): Input prompt to the model.

        Returns:
            str: Output text from the model.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=instruction
            )
            return response.text or "[Empty response]"
        except Exception as e:
            return f"[Error] {str(e)}"
