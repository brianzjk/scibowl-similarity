from __future__ import annotations

import re
from uuid import uuid4


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"
