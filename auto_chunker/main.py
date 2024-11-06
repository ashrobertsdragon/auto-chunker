from decouple import config

from auto_chunker import _config
from auto_chunker.incoming import api

if config("ENV", default="development") in _config.server_envs:
    import uvicorn

    _config.server_logger()
    uvicorn.run("auto_chunker.incoming.api:app", reload=True)
elif config("ENV", default="development") in _config.serverless_envs:
    from mangum import Mangum

    _config.serverless_logger()
    handler = Mangum(api.app)
