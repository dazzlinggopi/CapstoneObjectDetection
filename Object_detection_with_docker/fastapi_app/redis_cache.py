import redis

# Connect to Redis
redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)

def set_cache(key, value, ttl=3600):
    redis_client.set(key, value, ex=ttl)

def get_cache(key):
    return redis_client.get(key)

def delete_cache(key):
    redis_client.delete(key)

def purge_cache():
    redis_client.flushdb()
