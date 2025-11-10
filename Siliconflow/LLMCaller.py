# Siliconflow/LLMCaller.py

import requests
import json

LLM_MAP = {
    "Qwen7B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen14B": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen32B": "Qwen/Qwen2.5-32B-Instruct",
    "Qwen72B": "Qwen/Qwen2.5-72B-Instruct",
    "deepseekV2.5": "deepseek-ai/DeepSeek-V2.5",
    "deepseekV3": "deepseek-ai/DeepSeek-V3",
}

class LLMCaller:
    def __init__(self, llm: str, temperature: float = 0.7, max_tokens: int = 50, key: int = 1):
        if llm not in LLM_MAP:
            raise ValueError(f"Unsupported LLM alias: {llm}")

        self.model_id = LLM_MAP[llm]
        if key ==1:
            self.api_key = "YOUR API KEY"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def build_prompt(self, user_prompt: str) -> list:
        return [{"role": "user", "content": user_prompt}]

    def query(self, prompt: str) -> str:
        payload = {
            "model": self.model_id,
            "messages": self.build_prompt(prompt),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"LLM call failed: {response.status_code} - {response.text}")

        response_json = response.json()
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
