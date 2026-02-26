from openai import OpenAI
from tools import RECOMMEND_TOOL

class DeepSeekClient:
    def __init__(self):
        self.client = OpenAI(
            api_key="sk-e71eacef903346a89e9b029c4bb08551",
            base_url="https://api.deepseek.com"
        )

    def chat(self, messages):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=[RECOMMEND_TOOL],
            tool_choice="auto",
            stream=False
        )
        return response.choices[0].message
