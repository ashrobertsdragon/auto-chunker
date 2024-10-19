from decouple import config
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion


class OpenAIAPI:
    def __init__(self):
        api_key = config("OPENAI_API_KEY")
        self.client = OpenAI(api_key)
        self.model = config("OPENAI_MODEL")
        self.max_tokens = config("MAX_TOKENS", cast=int, default=4096)
        self.temperature = config("TEMPERATURE", cast=float, default=0.7)

    def call_api(self, messages: list[dict]) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
