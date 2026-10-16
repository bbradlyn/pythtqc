"""Make the in-repo package importable for no-install example runs.

``import _path`` (before the library imports) is all an example script needs:
if ``pythtqc`` is pip-installed this is a no-op; otherwise it adds this repo's
``src/`` to ``sys.path`` and retries, so a genuinely broken environment still
fails loudly here.
"""

import os
import sys

try:
    import pythtqc  # noqa: F401  (pip-installed)
except ImportError:  # in-repo dev run: add src/
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(_root, "src"))
    import pythtqc  # noqa: F401
