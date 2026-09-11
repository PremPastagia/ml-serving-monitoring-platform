"""ASGI entrypoint.

    uvicorn mlserve.serving.main:app --host 127.0.0.1 --port 8077

Kept separate from ``app.py`` so that importing the application *factory* never
constructs a service, opens a SQLite file or touches the registry. Tests import
``create_app`` and build isolated instances; only this module builds the shared one.
"""

from __future__ import annotations

from mlserve.config import load_config
from mlserve.serving.app import create_app

app = create_app(load_config())
