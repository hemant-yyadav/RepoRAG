import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.health import router as health_router
from app.api.routes.answers import router as answers_router
from app.api.routes.chat import router as chat_router
from app.api.routes.code_intelligence import router as code_intelligence_router
from app.api.routes.repositories import router as repositories_router
from app.api.routes.repositories_api import router as repositories_api_router
from app.api.routes.retrieval import router as retrieval_router
from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("request validation failed: %s %s", request.method, request.url.path)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled server error: %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(health_router)
app.include_router(answers_router)
app.include_router(chat_router)
app.include_router(code_intelligence_router)
app.include_router(repositories_router)
app.include_router(repositories_api_router)
app.include_router(retrieval_router)
