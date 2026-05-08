"""
WSGI entry point for Azure App Service / Gunicorn.
This file lives at the project root so gunicorn can find it.

Usage (Azure startup command):
    gunicorn --bind=0.0.0.0:8000 --timeout 600 wsgi:app
"""

from backend.app import create_app

app = create_app()
