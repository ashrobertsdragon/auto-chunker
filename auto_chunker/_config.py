import sys
from loguru import logger


def create_log_file():
    from pathlib import Path

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir / "server.log"


def serverless_logger():
    logger.remove()
    logger.add(sink=sys.stdout, serialize=True, level="INFO", enqueue=False)


def server_logger():
    log_file = create_log_file()
    logger.remove()
    logger.add(log_file, rotation="5MB", level="INFO")


server_envs = ["ec2", "development", "vps"]
serverless_envs = [
    "lambda",
    "fargate",
    "cloud run",
    "cloud function",
    "app engine",
    "development",
]
