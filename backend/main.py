"""
Main Application Entrypoint for ATLAS Backend.
Configures FastAPI, CORS middleware, API route registration, and health monitoring.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import datetime

from core.settings import settings
from memory.vector_store import global_vector_store
from memory.memory_models import MemoryStatus
from api.chat import router as chat_router
from api.memories import router as memories_router
from api.atlas import router as atlas_router
from api.attacks import router as attacks_router
from api.evaluation import router as evaluation_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure baseline vectors are indexed and model is ready
    print(f"🚀 ATLAS Backend starting up... [Project: {settings.PROJECT_NAME}]")
    print(f"🛡️  ATLAS Trust Threshold: {settings.ATLAS_TRUST_THRESHOLD}")
    print(f"📦 Initial Memory Store Count: {global_vector_store.count()} items")
    yield
    print("🛑 ATLAS Backend shutting down.")


app = FastAPI(
    title="ATLAS: Adaptive Trust Management API",
    description="Backend API for Defending Persistent LLM Agents Against Conversational Memory Poisoning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
origins = [
    settings.FRONTEND_ORIGIN,
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(chat_router, prefix=settings.API_PREFIX)
app.include_router(memories_router, prefix=settings.API_PREFIX)
app.include_router(atlas_router, prefix=settings.API_PREFIX)
app.include_router(attacks_router, prefix=settings.API_PREFIX)
app.include_router(evaluation_router, prefix=settings.API_PREFIX)


@app.get("/api/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint returning system status and memory store telemetry.
    """
    return {
        "status": "healthy",
        "system": "ATLAS: Adaptive Trust Management for Long-term Agent Storage",
        "version": "1.0.0",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gemini_api_configured": bool(settings.GEMINI_API_KEY),
        "gemini_model": settings.GEMINI_MODEL,
        "trust_threshold": settings.ATLAS_TRUST_THRESHOLD,
        "total_memories": global_vector_store.count(),
        "accepted_memories": global_vector_store.count(MemoryStatus.ACCEPTED),
        "quarantined_memories": global_vector_store.count(MemoryStatus.QUARANTINED)
    }


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to ATLAS API. Navigate to /docs for interactive OpenAPI specification.",
        "health_check": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
