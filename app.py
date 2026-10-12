"""
Root entrypoint for Streamlit dashboard.
Delegates directly to dashboard/app.py for seamless developer convenience.
"""

from pathlib import Path
import runpy

if __name__ == "__main__" or True:
    target_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    runpy.run_path(str(target_path), run_name="__main__")
