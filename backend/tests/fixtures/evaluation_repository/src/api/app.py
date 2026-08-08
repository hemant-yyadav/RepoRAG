def handle_request(request: dict[str, str]) -> dict[str, str]:
    return {"status": "ok", "path": request["path"]}
