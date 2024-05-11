"""User routes."""

from sqlalchemy.orm import Session
from fastapi import APIRouter, status, Depends, HTTPException

from .. import schemas, crud
from ..database import get_db

router = APIRouter(prefix="/api/v1/users", tags=["Users"])


@router.get("/{spotify_user_id}", response_model=schemas.User)
def get_user(spotify_user_id: str, db: Session = Depends(get_db)):
    """Get a user by Spotify ID."""
    db_user = crud.get_user_by_spotify_id(db, spotify_user_id)
    if db_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with Spotify ID {spotify_user_id} not found.",
        )
    return db_user


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=schemas.User,
)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Create a new user."""
    db_user = crud.get_user_by_spotify_id(db, user.spotify_id)
    if db_user:
        return db_user
    return crud.create_user(db, user)


@router.post(
    "/{user_id}/playlists/",
    status_code=status.HTTP_201_CREATED,
    response_model=schemas.Playlist,
)
def create_playlist(
    user_id: int, playlist: schemas.PlaylistCreate, db: Session = Depends(get_db)
):
    """Create a new playlist."""
    return crud.create_user_playlist(db, playlist, user_id)
