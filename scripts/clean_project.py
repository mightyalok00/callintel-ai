"""
Utility script to clean temporary Python cache directories, pytest caches, and logs.
Does NOT delete trained models (models/) or datasets (data/).
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".coverage",
    "htmlcov",
]


def clean_project():
    removed_count = 0
    print(f"[*] Scanning {PROJECT_ROOT} for temporary cache artifacts...")

    for path in PROJECT_ROOT.rglob("*"):
        # Never delete inside .venv
        if ".venv" in path.parts:
            continue

        if path.name in CACHE_PATTERNS or path.suffix in [".pyc", ".pyo"]:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"[-] Removed directory: {path.relative_to(PROJECT_ROOT)}")
                elif path.is_file():
                    path.unlink()
                    print(f"[-] Removed file: {path.relative_to(PROJECT_ROOT)}")
                removed_count += 1
            except Exception as e:
                print(f"[!] Warning: Could not delete {path}: {e}")

    print(f"\n[OK] Cleanup complete. Removed {removed_count} temporary items.")


if __name__ == "__main__":
    clean_project()
