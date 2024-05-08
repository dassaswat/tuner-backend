"""Flask app for the homepage."""

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from src.config import get_origins
from src.fix_playlist import analyse_and_fix_playlist
from src.database import SessionLocal, engine
from src import models, schemas, crud

load_dotenv()
models.Base.metadata.create_all(bind=engine)
app = FastAPI()


def get_db():
    """Get the database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index():
    """Return the homepage."""
    with open("templates/index.html", "r", encoding="utf-8") as file:
        markup = file.read()
    return markup


@app.post("/user", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Create a new user."""
    db_user = crud.get_user_by_spotify_id(db, user.spotify_id)
    if db_user:
        return db_user
    return crud.create_user(db, user)


@app.get("/user/{spotify_id}", response_model=schemas.User)
def get_user(spotify_id: str, db: Session = Depends(get_db)):
    """Get a user by Spotify ID."""
    db_user = crud.get_user_by_spotify_id(db, spotify_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@app.post("/users/{user_id}/playlists/", response_model=schemas.Playlist)
def create_playlist(
    user_id: int, playlist: schemas.PlaylistCreate, db: Session = Depends(get_db)
):
    """Create a new playlist."""
    return crud.create_user_playlist(db, playlist, user_id)


@app.get("/playlists/{playlist_spotify_id}", response_model=schemas.Playlist)
def get_playlist_by_id(playlist_spotify_id: str, db: Session = Depends(get_db)):
    """Get a playlist by ID."""

    playlist = crud.get_playlist_by_spotify_id(db, playlist_spotify_id)
    if playlist is None:
        raise HTTPException(status_code=404, detail="Playlist not found")
    return playlist


@app.delete("/playlists/{playlist_id}")
def delete_playlist(playlist_id: int, db: Session = Depends(get_db)):
    """Delete a playlist by ID."""
    return crud.delete_playlist(db, playlist_id)


@app.post("/analyse", response_model=schemas.SuggestedPlaylistResponse)
def analyse(features: list[schemas.SongFeatures | None]):
    """Analyse the song features."""
    try:
        song_features = [
            feature.model_dump() for feature in features if feature is not None
        ]

        track_uris = analyse_and_fix_playlist(song_features)
        total_duration = sum(track["duration_ms"] for track in song_features)
        return schemas.SuggestedPlaylistResponse(
            uris=track_uris, length=len(track_uris), duration_ms=total_duration
        )
    except ValueError as error:
        raise HTTPException(
            status_code=500, detail=f"Error analysing playlist {error}"
        ) from error
