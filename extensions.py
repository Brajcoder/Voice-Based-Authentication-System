"""
Shared application extensions for Voice-Based Authentication System.
Initializes Flask-Limiter using the client's IP address.
"""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    storage_uri="memory://",
)

