"""Playlist Routes."""

from typing import Optional

from sqlalchemy.orm import Session
from fastapi import APIRouter, status, Depends, HTTPException, Response

from .. import schemas, crud
from ..database import get_db

router = APIRouter(prefix="/api/v1/playlists", tags=["Playlists"])


@router.get("/", response_model=list[Optional[schemas.Playlist]])
def get_playlists_by_spotify_ids(ids: str = "", db: Session = Depends(get_db)):
    """Get playlists by their Spotify IDs."""
    spotify_playlist_ids = ids.split(",")
    if spotify_playlist_ids[0] == "":
        return []

    playlists = crud.get_playlists_by_spotify_ids(db, spotify_playlist_ids)
    return playlists


@router.delete("/{playlist_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_playlist(playlist_id: int, db: Session = Depends(get_db)):
    """Delete a playlist by ID."""
    playlist = crud.get_playlist_by_id(db, playlist_id)
    if playlist is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Playlist with ID {playlist_id} not found.",
        )

    crud.delete_playlist(db, playlist_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
