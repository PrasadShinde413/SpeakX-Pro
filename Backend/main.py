import sys
import os
# Add the project root to python path so we can import ml_pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import auth, sessions, chat

app = FastAPI(
    title="SpeakX-Pro Enterprise API",
    description="Backend API for Public Speaking Analyzer",
    version="2.0.0"
)

# CORS Configuration for the React Frontend
origins = [
    "http://localhost:5173",     # Vite Default
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["Sessions"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "message": "API is running."}
