#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reproduction proofs for native surface envelopes of reconstructed triangle sources."""

from __future__ import annotations

from typing import final, Literal, TypeAlias

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import surface_envelope, SurfaceEnvelopeResult
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import parse


SurfaceEnvelopeMode: TypeAlias = Literal["remeshed", "kernel"]
"""Which native envelope construction a holder retains.

``"remeshed"``: the extracted surface is remeshed inside the certified band
and the marching-tetrahedra carrier is dropped; a volume is filled afresh
inside the remeshed surface. ``"kernel"``: the kernel's marching-tetrahedra
extraction is kept unchanged, its closed skin as the envelope surface and its
positively oriented tetrahedral carrier as the envelope volume.
"""


def parse_envelope_mode(value: object, /) -> SurfaceEnvelopeMode:
    """Parse one closed envelope-mode selector."""
    return parse(value, SurfaceEnvelopeMode, "envelope_mode")


def _input_id(
    vertices: np.ndarray,
    triangles: np.ndarray,
    controls: tuple[float, float, float],
    mode: SurfaceEnvelopeMode,
    /,
) -> str:
    return canonical_fingerprint(
        (
            "surface-envelope-input",
            np.asarray(vertices, dtype=np.float64),
            np.asarray(triangles, dtype=np.int64),
            tuple(float(value) for value in controls),
            mode,
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


def _carrier_id(volume_vertices: np.ndarray, tetrahedra: np.ndarray, /) -> str:
    return canonical_fingerprint(
        (
            "surface-envelope-carrier",
            np.asarray(volume_vertices, dtype=np.float64),
            np.asarray(tetrahedra, dtype=np.int64),
        )
    )


def _run(
    vertices: np.ndarray,
    triangles: np.ndarray,
    controls: tuple[float, float, float],
    mode: SurfaceEnvelopeMode,
    /,
) -> SurfaceEnvelopeResult:
    match mode:
        case "remeshed":
            return surface_envelope(vertices, triangles, *controls, remesh=True)
        case "kernel":
            return surface_envelope(vertices, triangles, *controls, remesh=False)


@final
class SurfaceEnvelopeReproduction(StrictModule, NonTrainableState):
    """Record that the native kernel produced one envelope from one input.

    The holder that ran the kernel retains it with the output. It proves a
    reproduction only for the exact input complex, controls, mode, and output
    whose content digests it binds; the kernel (extraction and, in
    ``"remeshed"`` mode, band remeshing) is a deterministic function of its
    input, controls and mode. ``carrier_id`` binds the retained tetrahedral
    carrier (the surface vertices and no cells in ``"remeshed"`` mode).
    """

    input_id: str = eqx.field(static=True)
    output_id: str = eqx.field(static=True)
    carrier_id: str = eqx.field(static=True)

    def binds(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        controls: tuple[float, float, float],
        mode: SurfaceEnvelopeMode,
        bounds: np.ndarray,
        surface_vertices: np.ndarray,
        surface_triangles: np.ndarray,
        /,
    ) -> bool:
        """Whether this record is for exactly these inputs and this surface output."""
        return self.input_id == _input_id(
            vertices, triangles, controls, mode
        ) and self.output_id == _output_id(bounds, surface_vertices, surface_triangles)


def produce_envelope(
    vertices: np.ndarray,
    triangles: np.ndarray,
    controls: tuple[float, float, float],
    mode: SurfaceEnvelopeMode,
    /,
) -> tuple[SurfaceEnvelopeResult, SurfaceEnvelopeReproduction]:
    """Run the native envelope kernel in ``mode``; return its result and record.

    Raises the kernel's ``MeshcoreError`` when the input admits no envelope.
    """
    checked_mode = parse_envelope_mode(mode)
    result = _run(vertices, triangles, controls, checked_mode)
    record = SurfaceEnvelopeReproduction(
        input_id=_input_id(vertices, triangles, controls, checked_mode),
        output_id=_output_id(
            result.bounds, result.surface_vertices, result.surface_triangles
        ),
        carrier_id=_carrier_id(result.volume_vertices, result.tetrahedra),
    )
    return result, record


def envelope_reproduces(
    vertices: np.ndarray,
    triangles: np.ndarray,
    controls: tuple[float, float, float],
    mode: SurfaceEnvelopeMode,
    bounds: np.ndarray,
    surface_vertices: np.ndarray,
    surface_triangles: np.ndarray,
    record: SurfaceEnvelopeReproduction | None,
    /,
    *,
    carrier: tuple[np.ndarray, np.ndarray] | None = None,
) -> bool:
    """Whether the ``mode`` native envelope of the input reproduces the retained output.

    ``carrier`` (volume vertices and tetrahedra), when given, must reproduce
    too. A retained record that binds exactly this input and output proves
    it; any other record (or none) is not trusted and the kernel reruns.
    Raises the kernel's ``MeshcoreError`` when the input no longer admits an
    envelope.
    """
    checked_mode = parse_envelope_mode(mode)
    if record is not None and record.binds(
        vertices,
        triangles,
        controls,
        checked_mode,
        bounds,
        surface_vertices,
        surface_triangles,
    ):
        return carrier is None or record.carrier_id == _carrier_id(*carrier)
    fresh = _run(vertices, triangles, controls, checked_mode)
    return (
        np.array_equal(fresh.bounds, bounds)
        and np.array_equal(fresh.surface_vertices, surface_vertices)
        and np.array_equal(fresh.surface_triangles, surface_triangles)
        and (
            carrier is None
            or (
                np.array_equal(fresh.volume_vertices, carrier[0])
                and np.array_equal(fresh.tetrahedra, carrier[1])
            )
        )
    )
