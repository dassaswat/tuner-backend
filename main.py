"""Application entry point."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.routers import playlist_tuner, user, playlist

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, tags=["Base"])
def index():
    """Return the homepage."""
    with open("templates/index.html", "r", encoding="utf-8") as file:
        markup = file.read()
    return markup


@app.get("/health-check", tags=["Base"])
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


app.include_router(user.router)
app.include_router(playlist.router)
app.include_router(playlist_tuner.router)
