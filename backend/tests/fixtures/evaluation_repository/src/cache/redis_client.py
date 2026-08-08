def get_cache_client(redis_url: str) -> dict[str, str]:
    return {"url": redis_url}


def cache_response(key: str, value: str) -> tuple[str, str]:
    return key, value
