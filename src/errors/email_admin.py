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


class EmailAdmin(EmailAdminHandler):
    def send_email(self) -> None:
        self._log_send_email()
