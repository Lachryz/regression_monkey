"""Public compatibility aliases for the refactored package layout."""

from . import common
from .engine import py, r, stata
from .plot import html, png

__all__ = ["common", "html", "png", "py", "r", "stata"]
