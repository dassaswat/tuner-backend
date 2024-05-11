"""Database models."""

from sqlalchemy import Column, ForeignKey, Integer, String, TIMESTAMP, text, Numeric
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    """User model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, nullable=False)
    spotify_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(
        TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")
    )

    tplaylists = relationship("Playlist", back_populates="owner")


class Playlist(Base):
    """Playlist model."""

    __tablename__ = "tplaylists"

    id = Column(
        Integer,
        primary_key=True,
        nullable=False,
        index=True,
    )
    spotify_playlist_id = Column(String, unique=True, nullable=False, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(
        TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")
    )

    owner = relationship("User", back_populates="tplaylists")


class PlaylistFeature(Base):
    """Playlist dataset model."""

    __tablename__ = "playlists_features"

    id = Column(Integer, primary_key=True, index=True, nullable=False)
    spotify_playlist_id = Column(String, unique=True, nullable=False, index=True)
    energy = Column(Numeric, nullable=False)
    liveness = Column(Numeric, nullable=False)
    tempo = Column(Numeric, nullable=False)
    speechiness = Column(Numeric, nullable=False)
    acousticness = Column(Numeric, nullable=False)
    instrumentalness = Column(Numeric, nullable=False)
    danceability = Column(Numeric, nullable=False)
    loudness = Column(Numeric, nullable=False)
    valence = Column(Numeric, nullable=False)


class ModelWeights(Base):
    """Model weights model."""

    __tablename__ = "model_weights"

    id = Column(Integer, primary_key=True, index=True, nullable=False)
    energy = Column(Numeric, nullable=False)
    liveness = Column(Numeric, nullable=False)
    tempo = Column(Numeric, nullable=False)
    speechiness = Column(Numeric, nullable=False)
    acousticness = Column(Numeric, nullable=False)
    instrumentalness = Column(Numeric, nullable=False)
    danceability = Column(Numeric, nullable=False)
    loudness = Column(Numeric, nullable=False)
    evaluation_score = Column(Numeric, nullable=False)
    created_at = Column(
        TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")
    )
