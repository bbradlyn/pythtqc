"""Figures for the examples, saved so that a rerun writes the same PDF.

* :func:`save_figure`: a figure saved without the time stamp matplotlib
  writes into a PDF or an SVG.
"""

import os

__all__ = ["save_figure"]


def save_figure(fig, path, **savefig_kwargs):
    """A figure saved without the time stamp matplotlib writes into a PDF or an SVG.

    The figure is written with ``fig.savefig(path)``.  A file whose name ends
    in ``.pdf`` or ``.svg`` is written without the time stamp, and a file of
    any other kind as matplotlib writes it.  A PDF written without the time
    stamp has the same bytes on a rerun.  An SVG has the same bytes on a rerun
    only where ``matplotlib.rcParams["svg.hashsalt"]`` is set, since
    matplotlib otherwise gives its elements random ids.  With ``metadata=``
    given, it is passed unchanged, and the time stamp is written unless it
    sets ``"CreationDate"`` (PDF) or ``"Date"`` (SVG) to ``None``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path : str or os.PathLike
        The file to write.
    **savefig_kwargs
        Keyword arguments passed on to ``Figure.savefig``.
    """
    if "metadata" not in savefig_kwargs:
        suffix = os.path.splitext(os.fspath(path))[1].lower()
        if suffix == ".pdf":
            savefig_kwargs["metadata"] = {"CreationDate": None}
        elif suffix == ".svg":
            savefig_kwargs["metadata"] = {"Date": None}
    fig.savefig(path, **savefig_kwargs)
