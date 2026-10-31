#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reproduction proofs for native surface envelopes of reconstructed triangle sources."""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import surface_envelope, SurfaceEnvelopeResult
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _input_id(
    vertices: np.ndarray, triangles: np.ndarray, controls: tuple[float, float, float], /
) -> str:
    return canonical_fingerprint(
        (
            "surface-envelope-input",
            np.asarray(vertices, dtype=np.float64),
            np.asarray(triangles, dtype=np.int64),
            tuple(float(value) for value in controls),
        )
    )


def _output_id(
    bounds: np.ndarray, surface_vertices: np.ndarray, surface_triangles: np.ndarray, /
) -> str:
    return canonical_fingerprint(
        (
            "surface-envelope-output",
            np.asarray(bounds, dtype=np.float64),
            np.asarray(surface_vertices, dtype=np.float64),
            np.asarray(surface_triangles, dtype=np.int64),
        )
    )


@final
class SurfaceEnvelopeReproduction(StrictModule, NonTrainableState):
    """Record that the native kernel produced one remeshed envelope from one input.

    The holder that ran the kernel retains it with the output. It proves a
    reproduction only for the exact input complex, controls, and output whose
    content digests it binds; the kernel (extraction and band remeshing) is a
    deterministic function of its input and controls.
    """

    input_id: str = eqx.field(static=True)
    output_id: str = eqx.field(static=True)

    def binds(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        controls: tuple[float, float, float],
        bounds: np.ndarray,
        surface_vertices: np.ndarray,
        surface_triangles: np.ndarray,
        /,
    ) -> bool:
        """Whether this record is for exactly these inputs and this output."""
        return self.input_id == _input_id(
            vertices, triangles, controls
        ) and self.output_id == _output_id(bounds, surface_vertices, surface_triangles)


def produce_envelope(
    vertices: np.ndarray, triangles: np.ndarray, controls: tuple[float, float, float], /
) -> tuple[SurfaceEnvelopeResult, SurfaceEnvelopeReproduction]:
    """Run the remeshed native envelope kernel; return its result and reproduction record.

    Raises the kernel's ``MeshcoreError`` when the input admits no envelope.
    """
    result = surface_envelope(vertices, triangles, *controls, remesh=True)
    record = SurfaceEnvelopeReproduction(
        input_id=_input_id(vertices, triangles, controls),
        output_id=_output_id(
            result.bounds, result.surface_vertices, result.surface_triangles
        ),
    )
    return result, record


def envelope_reproduces(
    vertices: np.ndarray,
    triangles: np.ndarray,
    controls: tuple[float, float, float],
    bounds: np.ndarray,
    surface_vertices: np.ndarray,
    surface_triangles: np.ndarray,
    record: SurfaceEnvelopeReproduction | None,
    /,
) -> bool:
    """Whether the remeshed native envelope of the input reproduces the retained output.

    A retained record that binds exactly this input and output proves it; any
    other record (or none) is not trusted and the kernel reruns. Raises the
    kernel's ``MeshcoreError`` when the input no longer admits an envelope.
    """
    if record is not None and record.binds(
        vertices, triangles, controls, bounds, surface_vertices, surface_triangles
    ):
        return True
    fresh = surface_envelope(vertices, triangles, *controls, remesh=True)
    return (
        np.array_equal(fresh.bounds, bounds)
        and np.array_equal(fresh.surface_vertices, surface_vertices)
        and np.array_equal(fresh.surface_triangles, surface_triangles)
    )
