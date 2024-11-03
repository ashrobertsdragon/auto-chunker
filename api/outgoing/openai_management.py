from loguru import logger

from outgoing.openai_client import OpenAIAPI
from errors.error_handling import error_handle
from errors._exceptions import unresolvable_errors, NoMessageError

CLIENT = OpenAIAPI()


def call_gpt_api(
    chapter_prompt: str,
    client=CLIENT,
    retry_count: int = 0,
) -> str:
    """
    Makes API calls to the OpenAI ChatCompletions engine.

    Args:.
        prompt (str): The user's prompt.
        retry_count (int): The number of retry attempts. Defaults to 0.

    Returns:
        str: The generated content from the OpenAI GPT-4o Mini model
    """

    role_script = "You are an expert writer who specializes in "
    "writing scene beats that are clear and concise."
    prompt = (
        "For the following chapter, please reverse engineer the scene beats for the author. Provide only the beats. DO NOT PROVIDE ANY COMMENTARY OR COMMENT ON THE STORY. Return only a list of scene beats."
        + chapter_prompt
    )
    messages = [
        {"role": "system", "content": role_script},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.call_api(messages)
        if response:
            logger.info(response._request_id)
        if response.choices and response.choices[0].message.content:
            answer = response.choices[0].message.content.strip()
        else:
            raise NoMessageError("No message content found")

    except tuple([NoMessageError] + unresolvable_errors()) as e:
        retry_count = error_handle(e=e, retry_count=retry_count)
        answer = call_gpt_api(prompt, client, retry_count)
    return answer
