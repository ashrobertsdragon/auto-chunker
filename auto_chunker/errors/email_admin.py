import resend
from decouple import config
from loguru import logger


class EmailAdminHandler:
    def __init__(self, e: Exception) -> None:
        self.error = e
        self.error_message = str(e)
        self.status_code = getattr(e, "status_code", None)

    def send_email(self) -> None:
        raise NotImplementedError(
            "Subclasses must implement send_email method"
        )

    def _log_send_email(self) -> None:
        logger.info("Email sent to admin")


class ResendSDK(EmailAdminHandler):
    def __init__(self, e: Exception) -> None:
        super().__init__(e)
        resend.api_key = config("RESEND_API_KEY")
        admin_email = config("ADMIN_EMAIL")
        self.params = {
            "from": admin_email,
            "to": admin_email,
            "subject": "Error in Auto Chunker",
            "html": f"<p>{self.error_message}</p>",
        }

    def send_email(self) -> None:
        resend.Emails.send(self.params)
        self._log_send_email()
