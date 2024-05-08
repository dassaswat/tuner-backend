"""This module contains basic configurations."""

import os
from dotenv import load_dotenv

load_dotenv()


def get_db_url():
    """Get the database URL."""
    return os.environ.get("DATABASE_URL")


def get_origins():
    """Get the origins."""
    return os.environ.get("ORIGIN_URLS", "*").split(",")
