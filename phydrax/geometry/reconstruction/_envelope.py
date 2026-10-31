#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reproduction proofs for native surface envelopes of reconstructed triangle sources."""

from __future__ import annotations

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import surface_envelope


# Envelope reproductions proved in this process, addressed by content. The native
# kernel (extraction and band remeshing) is a deterministic function of its input
# complex and controls, so an (input, controls, output) content already produced
# or reproduced here reproduces again without rerunning the kernel.
_REPRODUCED: set[str] = set()


def _reproduction_key(
    vertices: np.ndarray,
    triangles: np.ndarray,
    controls: tuple[float, float, float],
    bounds: np.ndarray,
    surface_vertices: np.ndarray,
    surface_triangles: np.ndarray,
    /,
) -> str:
    return canonical_fingerprint(
        (
            "surface-envelope-reproduction",
            np.asarray(vertices, dtype=np.float64),
            np.asarray(triangles, dtype=np.int64),
            tuple(float(value) for value in controls),
            np.asarray(bounds, dtype=np.float64),
            np.asarray(surface_vertices, dtype=np.float64),
            np.asarray(surface_triangles, dtype=np.int64),
        )
    )


def record_envelope_reproduction(
    vertices: np.ndarray,
    triangles: np.ndarray,
    controls: tuple[float, float, float],
    bounds: np.ndarray,
    surface_vertices: np.ndarray,
    surface_triangles: np.ndarray,
    /,
) -> None:
    """Record a remeshed envelope this process just produced from the given input."""
    _REPRODUCED.add(
        _reproduction_key(
            vertices, triangles, controls, bounds, surface_vertices, surface_triangles
        )
    )


def envelope_reproduces(
    vertices: np.ndarray,
    triangles: np.ndarray,
    controls: tuple[float, float, float],
    bounds: np.ndarray,
    surface_vertices: np.ndarray,
    surface_triangles: np.ndarray,
    /,
) -> bool:
    """Whether the remeshed native envelope of the input reproduces the retained output.

    Raises the kernel's ``MeshcoreError`` when the input no longer admits an envelope.
    """
    key = _reproduction_key(
        vertices, triangles, controls, bounds, surface_vertices, surface_triangles
    )
    if key in _REPRODUCED:
        return True
    fresh = surface_envelope(vertices, triangles, *controls, remesh=True)
    if (
        np.array_equal(fresh.bounds, bounds)
        and np.array_equal(fresh.surface_vertices, surface_vertices)
        and np.array_equal(fresh.surface_triangles, surface_triangles)
    ):
        _REPRODUCED.add(key)
        return True
    return False
