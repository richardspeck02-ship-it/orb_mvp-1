import asyncio
import json
import valkey.asyncio as redis

try:
    from utils import config as conf, log
except ImportError:
    from ..utils import config as conf, log


class RedisOperations:
    def __init__(self):
        self.conf = conf
        self.redis_client = redis.Valkey(host=self.conf['redis']['host'],
                                         port=self.conf['redis']['port'],
                                         db=self.conf['redis'].get('db', 0),
                                         decode_responses=True)
        self.prefix = self.conf.get('envPrefix', 'local') + ':orb-generation'

    async def ping(self):
        try:
            await self.redis_client.ping()
        except Exception:
            error_string = f'Redis not available! Redis config: {self.conf["redis"]}'
            log.error(error_string)
            raise RuntimeError(error_string)

    async def clear_db(self, keys=None):
        if not keys:
            keys = await self.redis_client.keys(f'{self.prefix}:*')
        else:
            keys = [f'{self.prefix}:{key}' for key in keys]
        if keys:
            tasks = [asyncio.create_task(self.redis_client.delete(key))
                     for key in keys]
            await asyncio.gather(*tasks)

    async def insert_or_update_data(self, key, value):
        """
        Insert or update data in Redis.

        :param key: The Redis key.
        :param value: The value to store (dictionary).
        """
        key = f'{self.prefix}:{key}'

        # Convert unsupported types (e.g., bool) to supported types
        converted_value = {k: self._convert_to_redis_supported_type(v) for k, v in value.items()}

        await self.redis_client.hset(key, mapping=converted_value)

    def _convert_to_redis_supported_type(self, value):
        """
        Convert unsupported types (e.g., bool) to Redis-supported types.

        :param value: The value to convert.
        :return: The converted value.
        """
        if isinstance(value, bool):
            return int(value)  # Convert bool to int (True -> 1, False -> 0)
        elif isinstance(value, dict) or isinstance(value, list):
            return json.dumps(value)  # Convert dict or list to JSON string
        return value  # Return other types as-is

    async def get_data(self, key):
        """
        Retrieve data from Redis.

        :param key: The Redis key.
        :return: The retrieved data (dictionary).
        """
        key = f'{self.prefix}:{key}'
        data = await self.redis_client.hgetall(key)

        # Convert data back to original types
        return {k: self._convert_from_redis_supported_type(v) for k, v in data.items()}

    async def add_to_stream(self, stream_name, payload):
        try:
            message_id = await self.redis_client.xadd(
                stream_name,
                fields={'payload': json.dumps(payload)},
                maxlen=10000,
                approximate=True
            )
            log.info(f"Message added: {message_id}")
        except Exception as e:
            log.error(f"Error adding message to Redis stream: {e}")

    def _convert_from_redis_supported_type(self, value):
        """
        Convert Redis-supported types back to original types.

        :param value: The value to convert.
        :return: The converted value.
        """
        try:
            # Attempt to parse JSON strings back to dict or list
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass

        # Convert integers 0 and 1 back to bool
        if value == "0":
            return False
        elif value == "1":
            return True

        return value

    async def get_redis_client(self):
        return self.redis_client
