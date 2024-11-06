import pytest
from auto_chunker.errors._exceptions import AuthenticationError
from auto_chunker.outgoing.openai_client import OpenAIAPI


@pytest.fixture
def mock_config(mocker):
    return mocker.patch(
        "auto_chunker.outgoing.openai_client.config",
        side_effect=lambda key, **kwargs: {
            "OPENAI_API_KEY": "test_key",
            "OPENAI_ORG_ID": "test_org",
            "OPENAI_PROJECT_ID": "test_project",
            "OPENAI_MODEL": "test_model",
            "MAX_TOKENS": 2048,
            "TEMPERATURE": 0.5,
        }.get(key, kwargs.get("default")),
    )


@pytest.fixture
def openai_api(mocker, mock_config):
    class FakeCompletions:
        def create(self, model, messages, max_tokens, temperature):
            return {"choices": [{"message": {"content": "response"}}]}

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key, organization, project):
            self.api_key = api_key
            self.organization = organization
            self.project = project
            self.chat = FakeChat()

    mocker.patch("auto_chunker.outgoing.openai_client.OpenAI", FakeOpenAI)
    return OpenAIAPI()


class TestOpenAIAPI:
    def test_initialization_with_valid_config(self, openai_api):
        assert openai_api.client.api_key == "test_key"
        assert openai_api.model == "test_model"
        assert openai_api.max_tokens == 2048
        assert openai_api.temperature == 0.5

    @pytest.mark.asyncio
    async def test_call_api_with_valid_messages(self, openai_api):
        response = await openai_api.call_api([
            {"role": "user", "content": "Hello"}
        ])

        assert response["choices"][0]["message"]["content"] == "response"

    def test_missing_or_invalid_api_key(self, mock_config):
        mock_config.side_effect = lambda key, **kwargs: {
            "OPENAI_API_KEY": None,
            "OPENAI_PROJECT_ID": "test_project",
            "OPENAI_MODEL": "test_model",
            "MAX_TOKENS": 2048,
            "TEMPERATURE": 0.5,
        }.get(key, kwargs.get("default"))
        print(f"Imported AuthenticationError: {AuthenticationError}")
        with pytest.raises(AuthenticationError) as exc_info:
            OpenAIAPI()
        print(f"Exception raised: {exc_info.type}")
        assert "OPENAI_API_KEY not set" in str(exc_info.value)

    def test_missing_or_invalid_project_id(self, mock_config):
        mock_config.side_effect = lambda key, **kwargs: {
            "OPENAI_API_KEY": "test_key",
            "OPENAI_ORG_ID": "test_org",
            "OPENAI_PROJECT_ID": None,
            "OPENAI_MODEL": "test_model",
            "MAX_TOKENS": 2048,
            "TEMPERATURE": 0.5,
        }.get(key, kwargs.get("default"))

        with pytest.raises(AuthenticationError) as exc_info:
            OpenAIAPI()

        assert "OPENAI_PROJECT_ID not set" in str(exc_info.value)

    def test_missing_or_invalid_org_id(self, mock_config):
        mock_config.side_effect = lambda key, **kwargs: {
            "OPENAI_API_KEY": "test_key",
            "OPENAI_ORG_ID": None,
            "OPENAI_PROJECT_ID": "test_project",
            "OPENAI_MODEL": "test_model",
            "MAX_TOKENS": 2048,
            "TEMPERATURE": 0.5,
        }.get(key, kwargs.get("default"))

        with pytest.raises(AuthenticationError) as exc_info:
            OpenAIAPI()

        assert "OPENAI_ORG_ID not set" in str(exc_info.value)

    def test_default_values_for_max_tokens_and_temperature(self, mock_config):
        mock_config.side_effect = lambda key, **kwargs: {
            "OPENAI_API_KEY": "test_key",
            "OPENAI_ORG_ID": "test_org",
            "OPENAI_PROJECT_ID": "test_project",
            "OPENAI_MODEL": "test_model",
        }.get(key, kwargs.get("default"))

        api = OpenAIAPI()
        assert api.max_tokens == 4096
        assert api.temperature == 0.7
