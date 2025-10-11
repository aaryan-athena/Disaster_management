from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def current_ist_timestamp() -> str:
    try:
        ist = ZoneInfo("Asia/Kolkata")
    except ZoneInfoNotFoundError:
        ist = ZoneInfo("UTC")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S %Z")


if __name__ == "__main__":
    print(f"Current IST timestamp: {current_ist_timestamp()}")
