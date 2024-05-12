"""CRUD operations."""

from sqlalchemy.orm import Session
from . import models
from . import schemas


# Spotify Users
def create_user(db: Session, user: schemas.UserCreate):
    """Create a new user."""
    db_user = models.User(spotify_id=user.spotify_id)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user_by_spotify_id(db: Session, spotify_id: str):
    """Get user by Spotify ID."""
    return db.query(models.User).filter(models.User.spotify_id == spotify_id).first()


def create_user_playlist(db: Session, playlist: schemas.PlaylistCreate, user_id: int):
    """Create a new playlist."""
    db_playlist = models.Playlist(**playlist.model_dump(), user_id=user_id)
    db.add(db_playlist)
    db.commit()
    db.refresh(db_playlist)
    return db_playlist


# User Playlists
def get_playlists_by_spotify_ids(db: Session, playlist_spotify_ids: list[str]):
    """Get playlists by their spotify IDs."""
    return db.query(models.Playlist).filter(
        models.Playlist.spotify_playlist_id.in_(playlist_spotify_ids)
    )


def get_playlist_by_id(db: Session, playlist_id: int):
    """Get playlist by ID."""
    return db.query(models.Playlist).filter(models.Playlist.id == playlist_id).first()


def delete_playlist(db: Session, playlist_id: int):
    """Delete playlist by ID."""
    db.query(models.Playlist).filter(models.Playlist.id == playlist_id).delete()
    db.commit()


# Playlist Features - Model Training Data
def create_playlist_feature_datapoint(
    db: Session, datapoint: schemas.PlaylistFeatureCreate
):
    """Add playlist feature datapoint to the database."""
    feature_datapoint = models.PlaylistFeature(**datapoint.model_dump())
    db.add(feature_datapoint)
    db.commit()
    db.refresh(feature_datapoint)
    return feature_datapoint


def get_all_playlist_features(db: Session):
    """Get all playlist features."""
    return db.query(models.PlaylistFeature).all()


def create_model_weights(db: Session, weights: schemas.ModelWeightsCreate):
    """Create model weights."""
    db_weights = models.ModelWeights(**weights.model_dump())
    db.add(db_weights)
    db.commit()
    db.refresh(db_weights)
    return db_weights


def get_latest_model_weights(db: Session):
    """Get the latest model weights."""
    return db.query(models.ModelWeights).order_by(models.ModelWeights.id.desc()).first()
