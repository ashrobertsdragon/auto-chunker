import openai


class APIError(Exception):
    pass


class AuthenticationError(APIError):
    pass


class BadRequestError(APIError):
    def __init__(self, status_code: int):
        self.status_code = status_code


class NoMessageError(Exception):
    pass


def unresolvable_errors() -> list:
    return [
        openai.BadRequestError,
        openai.AuthenticationError,
        openai.NotFoundError,
        openai.PermissionDeniedError,
        openai.UnprocessableEntityError,
        openai.OpenAIError,
    ]
