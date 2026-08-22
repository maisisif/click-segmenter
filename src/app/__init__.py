"""The user-facing web interface, kept separate from the entry points that launch it."""

from src.app.ui import build_ui

__all__ = ["build_ui"]
