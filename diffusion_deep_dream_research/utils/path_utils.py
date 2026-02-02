from typing import Optional


def extract_id(filename: str) -> Optional[int]:
    try:
        return int(filename.split("_")[-1].split(".")[0])
    except (ValueError, IndexError):
        return None
