from loguru import logger


def create_log_file():
    from pathlib import Path

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir / "server.log"


def serverless_logger():
    logger.remove()
    logger.add(serialize=True, level="INFO", enqueue=False)
    return logger


def server_logger():
    log_file = create_log_file()
    logger.remove()
    logger.add(log_file, rotation="5MB", level="INFO")
    return logger


server_envs = ["ec2", "development", "vps"]
serverless_envs = [
    "lambda",
    "fargate",
    "cloud run",
    "cloud function",
    "app engine",
    "development",
]
