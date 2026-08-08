def get_user_by_id(user_id: str) -> dict[str, str]:
    return {"id": user_id}


def authenticate_user(username: str, password: str) -> bool:
    return bool(username and password)
