from mangum import Mangum

from auto_chunker import _config
from auto_chunker.incoming import api


_config.serverless_logger()
handler = Mangum(api.app)
