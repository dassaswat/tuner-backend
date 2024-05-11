"""Pydantic schemas."""

from typing import Optional
from pydantic import BaseModel, ConfigDict


# Playlist Schemas
class PlaylistBase(BaseModel):
    """Playlist base model."""

    spotify_playlist_id: str


class PlaylistCreate(PlaylistBase):
    """Playlist create model."""


class Playlist(PlaylistBase):
    """Playlist model."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int


# User schemas
class UserBase(BaseModel):
    """User base model."""

    spotify_id: str


class UserCreate(UserBase):
    """User create model."""


class User(UserBase):
    """User model."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    tplaylists: list[Playlist] = []


# Playlist Features Schemas
class PlaylistFeatureBase(BaseModel):
    """Playlist dataset base model."""

    spotify_playlist_id: str
    energy: float
    liveness: float
    tempo: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    danceability: float
    loudness: float
    valence: float


class PlaylistFeatureCreate(PlaylistFeatureBase):
    """Playlist dataset create model."""


class PlaylistFeature(PlaylistFeatureBase):
    """Playlist dataset model."""

    model_config = ConfigDict(from_attributes=True)
    id: int


# Model Weights Schemas
class ModelWeightsBase(BaseModel):
    """Model weights base model."""

    energy: float
    liveness: float
    tempo: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    danceability: float
    loudness: float
    evaluation_score: float


class ModelWeightsCreate(ModelWeightsBase):
    """Model weights create model."""


class ModelWeights(ModelWeightsBase):
    """Model weights model."""

    model_config = ConfigDict(from_attributes=True)
    id: int


# Spotify API Schemas
class TrackFeatures(BaseModel):
    """Song features."""

    acousticness: float
    analysis_url: str
    danceability: float
    duration_ms: int
    energy: float
    id: str
    instrumentalness: float
    key: int
    liveness: float
    loudness: float
    mode: int
    speechiness: float
    tempo: float
    time_signature: int
    track_href: str
    type: str
    uri: str
    valence: float


# Playlist Tuner Schemas
class TunePlaylist(BaseModel):
    """Playlist tracks features."""

    spotify_playlist_id: str
    must_train: Optional[bool] = True
    seed_tracks: Optional[list[int]] = None
    tracks_features: list[Optional[TrackFeatures]]


class TunedPlaylist(BaseModel):
    """Suggested playlist response."""

    uris: list[str]
    length: int
    duration_ms: int
