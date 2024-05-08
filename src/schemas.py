"""Pydantic schemas."""

from pydantic import BaseModel


class PlaylistBase(BaseModel):
    """Playlist base model."""

    spotify_playlist_id: str


class PlaylistCreate(PlaylistBase):
    """Playlist create model."""


class Playlist(PlaylistBase):
    """Playlist model."""

    id: int
    user_id: int

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class UserBase(BaseModel):
    """User base model."""

    spotify_id: str


class UserCreate(UserBase):
    """User create model."""


class User(UserBase):
    """User model."""

    id: int
    tplaylists: list[Playlist] = []

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class SongFeatures(BaseModel):
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


class SuggestedPlaylistResponse(BaseModel):
    """Suggested playlist response."""

    uris: list[str]
    length: int
    duration_ms: int
