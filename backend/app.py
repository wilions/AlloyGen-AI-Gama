"""FastAPI application factory for AlloyGen 2.0."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.auth.router import router as auth_router
from backend.database.engine import init_db
from backend.routes.upload import router as upload_router
from backend.routes.predict import router as predict_router
from backend.routes.models import router as models_router
from backend.routes.sessions import router as sessions_router
from backend.routes.websocket import router as websocket_router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    await init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="AlloyGen 2.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5178",
            "http://127.0.0.1:5178",
            "http://localhost:5176",
            "http://127.0.0.1:5176",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router)
    app.include_router(upload_router)
    app.include_router(predict_router)
    app.include_router(models_router)
    app.include_router(sessions_router)
    app.include_router(websocket_router)

    return app


app = create_app()
