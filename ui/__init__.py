# ui/__init__.py
# Make UI optional: project can run headless even if viewer.py is missing.

try:
    from .viewer import Viewer  # type: ignore
except Exception:
    Viewer = None  # Viewer unavailable (missing file or missing deps)

__all__ = ["Viewer"]
