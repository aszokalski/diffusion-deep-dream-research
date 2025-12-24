"""
This script SHOULD NOT BE MOVED to any other directory!!!
It's a workaround to get the root dir of the project
"""
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent
