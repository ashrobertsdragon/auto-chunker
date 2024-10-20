from decouple import config
from openai import OpenAI
from openai._exceptions import OpenAIError
from openai.types.chat.chat_completion import ChatCompletion


class OpenAIAPI:
    def __init__(self):
        api_key: str = config("OPENAI_API_KEY", default=None)
        organization: str = config("OPENAI_ORG_ID", default=None)
        project: str = config("OPENAI_PROJECT_ID", default=None)
        self.model: str = config("OPENAI_MODEL", default="gpt-4o-mini")
        self.max_tokens: int = config("MAX_TOKENS", cast=int, default=4096)
        self.temperature: float = config(
            "TEMPERATURE", cast=float, default=0.7
        )

        if not api_key:
            raise OpenAIError("OPENAI_API_KEY not set")
        if not organization:
            raise OpenAIError("OPENAI_ORG_ID not set")
        if not project:
            raise OpenAIError("OPENAI_PROJECT_ID not set")
        self.client = OpenAI(
            api_key=api_key, organization=organization, project=project
        )

    def call_api(self, messages: list[dict]) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
