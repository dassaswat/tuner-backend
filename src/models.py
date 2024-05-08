"""Database models."""

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from src.database import Base


class User(Base):
    """User model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    spotify_id = Column(String, unique=True, index=True)

    tplaylists = relationship("Playlist", back_populates="owner")


class Playlist(Base):
    """Playlist model."""

    __tablename__ = "tplaylists"

    id = Column(Integer, primary_key=True, index=True)
    spotify_playlist_id = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="tplaylists")
