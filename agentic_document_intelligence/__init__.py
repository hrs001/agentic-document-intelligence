from __future__ import annotations

from pathlib import Path


# Allow imports like `agentic_document_intelligence.backend` and
# `agentic_document_intelligence.scripts` to resolve to the source-tree
# directories that live at the repository root.
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

__path__ = [str(_PACKAGE_DIR), str(_PROJECT_ROOT)]
