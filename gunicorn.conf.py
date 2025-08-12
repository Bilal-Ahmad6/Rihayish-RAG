"""Gunicorn configuration for Render deployment.

Notes:
- Single worker by default (embedding model is memory heavy). Scale via WEB_CONCURRENCY.
- Threads provide light concurrency for I/O (retrieval, API calls).
- Extended timeout handles initial model load on cold start.
"""
import os

workers = int(os.getenv("WEB_CONCURRENCY", "1"))
threads = int(os.getenv("GUNICORN_THREADS", "4"))
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = 5
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")
preload_app = False  # Avoid duplicate model load memory

def when_ready(server):  # pragma: no cover
    server.log.info("Gunicorn ready - service started.")
