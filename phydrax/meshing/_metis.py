#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""METIS multilevel k-way partitioning as an explicit external comparison route.

``MeshPartitionKind.METIS`` selects this route by name; the canonical GRAPH
route is the native partitioner of :mod:`phydrax.graph` and never falls back to
METIS. METIS is reached through its plain C ABI with ctypes. The library is
resolved from ``PHYDRAX_METIS_LIBRARY`` and then the platform loader search
path; absence is reported with :class:`MetisUnavailableError`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import os

import numpy as np


_ENVIRONMENT = "PHYDRAX_METIS_LIBRARY"
_OPTION_COUNT = 40
_OPTION_SEED = 8
_OPTION_UFACTOR = 16
_OPTION_NUMBERING = 17
_STATUS_OK = 1
_STATUS_MESSAGES = {
    -2: "METIS rejected the graph or options as invalid input.",
    -3: "METIS ran out of memory.",
    -4: "METIS failed with an unspecified error.",
}
# Weights are rescaled so the total stays below this bound for either index width.
_WEIGHT_BUDGET = float(1 << 30)


class MetisUnavailableError(ImportError):
    """No loadable METIS shared library was found for the METIS comparison route."""


class MetisPartitionError(RuntimeError):
    """METIS was loaded but did not return a partition."""


def _library_path() -> str:
    configured = os.environ.get(_ENVIRONMENT, "").strip()
    if configured:
        if not os.path.isfile(configured):
            raise MetisUnavailableError(
                f"{_ENVIRONMENT} names a missing METIS library: {configured!r}."
            )
        return configured
    located = ctypes.util.find_library("metis")
    if located is None:
        raise MetisUnavailableError(
            "The METIS partition route requires METIS; set "
            f"{_ENVIRONMENT} to the libmetis shared library path."
        )
    return located


@functools.cache
def _load(path: str, /) -> tuple[ctypes.CDLL, type[np.signedinteger]]:
    # ctypes reports every loader failure as OSError; it is the only signal.
    try:
        library = ctypes.CDLL(path)
    except OSError as error:
        raise MetisUnavailableError(
            f"METIS library {path!r} cannot be loaded."
        ) from error
    library.METIS_SetDefaultOptions.restype = ctypes.c_int
    library.METIS_SetDefaultOptions.argtypes = (ctypes.c_void_p,)
    library.METIS_PartGraphKway.restype = ctypes.c_int
    library.METIS_PartGraphKway.argtypes = (ctypes.c_void_p,) * 13
    # METIS does not export its idx_t width; SetDefaultOptions writes exactly
    # METIS_NOPTIONS entries of -1, which fills 20 or 40 int64 words.
    sentinel = np.int64(0x5A5A5A5A5A5A5A5A)
    probe = np.full((_OPTION_COUNT,), sentinel, dtype=np.int64)
    if library.METIS_SetDefaultOptions(probe.ctypes.data) != _STATUS_OK:
        raise MetisUnavailableError(f"METIS library {path!r} rejected option setup.")
    half = _OPTION_COUNT // 2
    if np.all(probe == -1):
        return library, np.int64
    if np.all(probe[:half] == -1) and np.all(probe[half:] == sentinel):
        return library, np.int32
    raise MetisUnavailableError(f"METIS library {path!r} has an unrecognized ABI.")


def metis_identity() -> str:
    """Resolved library path and index width; raises when METIS is unavailable."""
    path = _library_path()
    _, index = _load(path)
    return f"metis:{os.path.realpath(path)}:idx{np.dtype(index).itemsize * 8}"


def metis_partition(
    offsets: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    part_count: int,
    /,
    *,
    imbalance: float,
    seed: int,
) -> np.ndarray:
    """Partition a symmetric CSR graph into ``part_count`` weighted parts."""
    library, index = _load(_library_path())
    total = float(np.sum(weights))
    scaled = np.maximum(1, np.rint(weights * (_WEIGHT_BUDGET / total))).astype(index)
    xadj = np.ascontiguousarray(offsets, dtype=index)
    adjncy = np.ascontiguousarray(neighbors, dtype=index)
    vertices = np.asarray((offsets.size - 1,), dtype=index)
    constraints = np.asarray((1,), dtype=index)
    parts = np.asarray((part_count,), dtype=index)
    options = np.empty((_OPTION_COUNT,), dtype=index)
    library.METIS_SetDefaultOptions(options.ctypes.data)
    options[_OPTION_SEED] = seed
    options[_OPTION_NUMBERING] = 0
    options[_OPTION_UFACTOR] = max(1, round((imbalance - 1.0) * 1000.0))
    edge_cut = np.zeros((1,), dtype=index)
    owners = np.zeros((offsets.size - 1,), dtype=index)
    status = library.METIS_PartGraphKway(
        vertices.ctypes.data,
        constraints.ctypes.data,
        xadj.ctypes.data,
        adjncy.ctypes.data,
        scaled.ctypes.data,
        None,
        None,
        parts.ctypes.data,
        None,
        None,
        options.ctypes.data,
        edge_cut.ctypes.data,
        owners.ctypes.data,
    )
    if status != _STATUS_OK:
        raise MetisPartitionError(
            _STATUS_MESSAGES.get(status, f"METIS returned status {status}.")
        )
    return owners.astype(np.int32)


__all__ = [
    "MetisPartitionError",
    "MetisUnavailableError",
    "metis_identity",
    "metis_partition",
]
