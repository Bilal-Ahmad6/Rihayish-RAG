import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from requests import Response

from config import settings


def ensure_dirs() -> None:
    """Create required directories if they don't already exist."""
    for p in [settings.data_dir, settings.raw_dir, settings.processed_dir, settings.chroma_persist_dir]:
        Path(p).mkdir(parents=True, exist_ok=True)


def http_get(url: str, *, timeout: Optional[int] = None, sleep_between: float = 0.8) -> Response:
    """HTTP GET with a basic UA and optional polite delay."""
    headers = {"User-Agent": settings.user_agent}
    response = requests.get(url, headers=headers, timeout=timeout or settings.requests_timeout)
    time.sleep(sleep_between)
    response.raise_for_status()
    return response


def save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

