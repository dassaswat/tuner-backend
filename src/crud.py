"""CRUD operations."""

from sqlalchemy.orm import Session
from . import models
from . import schemas


def get_user_by_spotify_id(db: Session, spotify_id: str):
    """Get user by Spotify ID."""
    return db.query(models.User).filter(models.User.spotify_id == spotify_id).first()


def create_user(db: Session, user: schemas.UserCreate):
    """Create a new user."""
    db_user = models.User(spotify_id=user.spotify_id)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def create_user_playlist(db: Session, playlist: schemas.PlaylistCreate, user_id: int):
    """Create a new playlist."""
    db_playlist = models.Playlist(**playlist.dict(), user_id=user_id)
    db.add(db_playlist)
    db.commit()
    db.refresh(db_playlist)
    return db_playlist


def get_playlist_by_spotify_id(db: Session, playlist_spotify_id: int):
    """Get playlist by its spotify ID."""
    return (
        db.query(models.Playlist)
        .filter(models.Playlist.spotify_playlist_id == playlist_spotify_id)
        .first()
    )


def delete_playlist(db: Session, playlist_id: int):
    """Delete playlist by ID."""
    db.query(models.Playlist).filter(models.Playlist.id == playlist_id).delete()
    db.commit()
    return True
