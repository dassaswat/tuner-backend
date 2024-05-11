"""This module contains basic configurations."""

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Settings class"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    allow_origins: str
    database_user: str
    database_password: str
    database_name: str
    database_host: str
    database_port: str


settings = Settings()
