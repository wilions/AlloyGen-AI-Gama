"""Shared helpers for route modules. Manages the in-memory session store."""
from __future__ import annotations

import logging
import os
import re
import time

from backend.config import SESSION_TTL_SECONDS
from backend.pipeline import Session

logger = logging.getLogger(__name__)

# In-memory session store (cache layer; DB becomes source of truth in Phase 1.2)
sessions: dict[str, Session] = {}

GREETING = (
    "Hello! I am your Metallurgic & Mechanical AI expert. "
    "To get started, please upload your dataset and I'll help you explore it!"
)

_INVERSE_KEYWORDS = [
    "design", "find a composition", "find an alloy", "find a material",
    "suggest a composition", "suggest an alloy", "what composition",
    "what alloy", "optimize", "inverse", "achieve", "reach", "get a",
    "need an alloy", "need a material", "want an alloy", "want a material",
    "target of", "target value", "desired", "i want", "i need",
    "how to get", "how to achieve", "how can i get",
]


def sanitize_id(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", raw)[:64]


def sanitize_filename(raw: str) -> str:
    name = os.path.basename(raw)
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)[:128]


def purge_expired_sessions() -> None:
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s.last_active > SESSION_TTL_SECONDS]
    for sid in expired:
        session = sessions.pop(sid, None)
        if session:
            for path in (session.data_path, session.clean_data_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            logger.info("Purged expired session %s", sid)


def touch(session: Session) -> None:
    session.last_active = time.time()


def get_or_create_session(session_id: str) -> Session:
    if session_id not in sessions:
        sessions[session_id] = Session()
    touch(sessions[session_id])
    return sessions[session_id]


def is_inverse_design_request(msg: str) -> bool:
    lower = msg.lower()
    return any(kw in lower for kw in _INVERSE_KEYWORDS)
