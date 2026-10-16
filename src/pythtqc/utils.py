"""Figures for the examples, written so that a rerun writes the same file: :func:`save_figure`."""

import os

__all__ = ["save_figure"]


def save_figure(fig, path, **savefig_kwargs):
    """``fig.savefig(path)``, without the time stamp matplotlib writes into a PDF or an SVG.

    A rerun then writes the same bytes; the figure itself is unchanged.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure.
    path : str or os.PathLike
        Where to write it.  A ``.pdf`` or ``.svg`` is written without its stamp,
        any other as matplotlib writes it.
    **savefig_kwargs
        Passed to ``Figure.savefig``; a ``metadata=`` of its own keeps its stamp.
    """
    if "metadata" not in savefig_kwargs:
        suffix = os.path.splitext(os.fspath(path))[1].lower()
        if suffix == ".pdf":
            savefig_kwargs["metadata"] = {"CreationDate": None}
        elif suffix == ".svg":
            savefig_kwargs["metadata"] = {"Date": None}
    fig.savefig(path, **savefig_kwargs)
