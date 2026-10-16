"""Build hooks for the documentation site; ``mkdocs.yml`` names this file.

The pages are written to be read on GitHub as well, where a link may point at
any file of the repository, and the site carries only the pages under
``docs/``.  So:

* the root ``README.md`` is the site's home page, ``index.md``, its links into
  ``docs/`` read from there;
* the map, ``docs/README.md``, which would otherwise take the home page, is
  served as ``map.md``;
* a link to a file the site does not carry, an example script or the silicon
  data, points at that file in the repository;
* a ``python template`` fence, which tells ``scripts/check_page.py`` not to run a
  block, is a ``python`` fence, the form the site's Markdown reads;
* a pipe escaped in a table's code span, ``{R\\|t}``, which GitHub needs to keep
  the cell whole, is a plain pipe: the site's tables already read a code
  span's pipes as text, and would print the backslash.
"""

import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File

#: an inline Markdown link's target, ``](target)``
_LINK = re.compile(r"(\]\()([^)\s]+)(\))")

#: a page the site serves under another name, by its path under docs/
_MOVED = {"README.md": "map.md"}
_SOURCE = {served: source for source, served in _MOVED.items()}


def _carried(files, site_path):
    """Whether the site builds this file of docs/."""
    found = files.get_file_from_path(site_path)
    return found is not None and not found.inclusion.is_excluded()


def _retarget(markdown, config, files, here, site_dir):
    """Every relative link of a page, pointed at the page the site serves or at the repository.

    ``here`` is the page's source in the repository, and ``site_dir`` the
    folder the site serves it from.
    """
    repository = config["repo_url"].rstrip("/")

    def one(match):
        target = match[2]
        path, hash_, anchor = target.partition("#")
        if not path or re.match(r"^[a-z][a-z0-9+.-]*:", path) or path.startswith("/"):
            return match[0]
        source = posixpath.normpath(posixpath.join(posixpath.dirname(here), path))
        if source.startswith("docs/"):
            site_path = source[len("docs/"):]
            site_path = _MOVED.get(site_path, site_path)
            if _carried(files, site_path):
                new = posixpath.relpath(site_path, site_dir or ".") + hash_ + anchor
                return match[1] + new + match[3]
        return match[1] + f"{repository}/blob/main/{source}" + hash_ + anchor + match[3]

    return _LINK.sub(one, markdown)


def on_files(files, *, config):
    """The map under its own name, and the README as the home page."""
    docs = Path(config["docs_dir"])
    files.append(File.generated(config, "map.md",
                                content=(docs / "README.md").read_text(encoding="utf-8")))
    readme = (Path(config["config_file_path"]).parent / "README.md").read_text(encoding="utf-8")
    files.append(File.generated(config, "index.md",
                                content=_retarget(readme, config, files, "README.md", "")))
    return files


#: a ``python template`` fence's opening line
_TEMPLATE_FENCE = re.compile(r"^(\s*)```python template\s*$", re.M)
#: a table row, and a code span in it
_TABLE_ROW = re.compile(r"^\s*\|.*$", re.M)
_CODE_SPAN = re.compile(r"`[^`\n]*`")


def _for_the_site(markdown):
    """The two spellings GitHub needs and the site's Markdown reads otherwise."""
    markdown = _TEMPLATE_FENCE.sub(r"\1```python", markdown)
    return _TABLE_ROW.sub(lambda row: _CODE_SPAN.sub(
        lambda span: span[0].replace("\\|", "|"), row[0]), markdown)


def on_page_markdown(markdown, *, page, config, files):
    """A page's links to what the site does not carry pointed at the repository, and its spellings read."""
    served = page.file.src_uri
    if served != "index.md":
        source = "docs/" + _SOURCE.get(served, served)
        markdown = _retarget(markdown, config, files, source, posixpath.dirname(served))
    return _for_the_site(markdown)
