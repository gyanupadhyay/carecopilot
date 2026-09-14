"""API routers, mounted under ``/api`` by :func:`app.main.create_app`."""

from fastapi import APIRouter

from app.api.routes import actions, analytics, auth, chat, graph, health, records

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(records.router)
api_router.include_router(graph.router)
api_router.include_router(analytics.router)
api_router.include_router(chat.router)
api_router.include_router(actions.router)

__all__ = ["api_router"]
