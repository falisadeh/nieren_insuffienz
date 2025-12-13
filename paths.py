"""Shared path helpers for accessing the cs-transfer dataset."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_CS_TRANSFER_DIR = Path(
    "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
)
CS_TRANSFER_DIR = Path(os.environ.get("CS_TRANSFER_DIR", DEFAULT_CS_TRANSFER_DIR))
ORIGINAL_DATA_SUBDIR = "Original Daten"
ORIGINAL_DATA_DIR = CS_TRANSFER_DIR / ORIGINAL_DATA_SUBDIR


def cs_transfer_path(*parts: Any) -> Path:
    """Join the configured cs-transfer directory with the provided sub-path parts."""
    if not parts:
        return CS_TRANSFER_DIR
    return CS_TRANSFER_DIR.joinpath(*parts)


__all__ = ["CS_TRANSFER_DIR", "cs_transfer_path", "ORIGINAL_DATA_DIR", "ORIGINAL_DATA_SUBDIR"]
