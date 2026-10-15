"""Automatinis įrankių suradimas: app/tools/*/__init__.py su kintamuoju TOOL."""
import importlib
import logging
import pkgutil

from .tool import Tool

log = logging.getLogger("toolbox")


def discover() -> dict[str, Tool]:
    from .. import tools as pkg
    found: dict[str, Tool] = {}
    for m in pkgutil.iter_modules(pkg.__path__):
        if not m.ispkg:
            continue
        try:
            mod = importlib.import_module(f"{pkg.__name__}.{m.name}")
        except Exception:                      # vienas sugedęs įrankis neturi nulaužti visko
            log.exception("Nepavyko įkelti įrankio %s", m.name)
            continue
        tool = getattr(mod, "TOOL", None)
        if isinstance(tool, Tool):
            found[tool.id] = tool
    return found
