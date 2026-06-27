"""Make the repository root importable.

`streamlit run app/main.py` puts the entry script's directory (``app/``) on
``sys.path`` — not the repo root — so ``import app...`` and ``import models...``
fail with ModuleNotFoundError. Importing this module (which Streamlit can find
because ``app/`` is on the path) inserts the repo root. It is idempotent and a
no-op when the root is already present (e.g. under pytest).
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent  # app/ -> repo root

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
