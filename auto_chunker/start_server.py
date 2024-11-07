from multiprocessing import Process, freeze_support, set_start_method

import uvicorn
from decouple import config

from auto_chunker import _config


def start_server():
    if config("ENV", default="development") in _config.server_envs:
        _config.server_logger()
        uvicorn.run("auto_chunker.incoming.api:app", reload=True)


if __name__ == "__main__":
    freeze_support()
    set_start_method("spawn")
    Process(target=start_server).start()
