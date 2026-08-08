JWT_SECRET = "test-only"


def validate_token(token: str) -> bool:
    return bool(token and JWT_SECRET)


def create_access_token(user_id: str) -> str:
    return f"token:{user_id}"
