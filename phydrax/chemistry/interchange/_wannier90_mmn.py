#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strict clean-room Wannier90 ``*.mmn`` raw overlap adapter."""

from __future__ import annotations

import re

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._reciprocal import PreparedReciprocalConnectivity
from ...operators.periodic._family import PeriodicResourceError
from ..periodic._source import PeriodicSourceContext
from ..periodic._topology import PeriodicBandManifold, PeriodicOverlapBundle


_INTEGER = re.compile(r"[+-]?\d+")
_FLOAT = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[EeDd][+-]?\d+)?")


def _integer(token: str, noun: str) -> int:
    if _INTEGER.fullmatch(token) is None:
        raise ValueError(f"Wannier90 MMN {noun} is not an integer.")
    return int(token)


def _float(token: str, noun: str) -> float:
    if _FLOAT.fullmatch(token) is None:
        raise ValueError(f"Wannier90 MMN {noun} is not a finite decimal.")
    value = float(token.replace("D", "E").replace("d", "e"))
    if not np.isfinite(value):
        raise ValueError(f"Wannier90 MMN {noun} must be finite.")
    return value


class Wannier90MMNImport(StrictModule, NonTrainableState):
    """Raw link matrices; no polar projection, manifold, or topology claim."""

    context: PeriodicSourceContext
    connectivity: PreparedReciprocalConnectivity
    raw_overlaps: Array
    singular_values: Array
    import_id: str = eqx.field(static=True)

    def __init__(
        self,
        context: PeriodicSourceContext,
        connectivity: PreparedReciprocalConnectivity,
        raw_overlaps,
        /,
    ):
        if not isinstance(context, PeriodicSourceContext) or not isinstance(
            connectivity, PreparedReciprocalConnectivity
        ):
            raise TypeError(
                "MMN import requires source context and prepared connectivity."
            )
        overlaps = np.asarray(raw_overlaps)
        if (
            overlaps.ndim != 3
            or overlaps.shape[0] != connectivity.edge_count
            or overlaps.shape[1] != overlaps.shape[2]
            or np.any(~np.isfinite(overlaps))
        ):
            raise ValueError(
                "MMN raw overlaps must be finite with shape (edge, band, band)."
            )
        singular = np.linalg.svd(overlaps, compute_uv=False)
        self.context = context
        self.connectivity = connectivity
        self.raw_overlaps = jnp.asarray(overlaps)
        self.singular_values = jnp.asarray(singular)
        self.import_id = canonical_fingerprint(
            {
                "kind": "wannier90-mmn-import",
                "context": context.context_id,
                "connectivity": connectivity.prepared_id,
                "arrays": array_tree_fingerprint(
                    {"raw_overlaps": overlaps, "singular_values": singular}
                ),
            }
        )


def read_wannier90_mmn(
    payload: bytes,
    context: PeriodicSourceContext,
    connectivity: PreparedReciprocalConnectivity,
    /,
    *,
    maximum_bytes: int = 256 * 1024 * 1024,
    maximum_bands: int = 4096,
    maximum_records: int = 16_000_000,
) -> Wannier90MMNImport:
    """Parse exact MMN records against caller-prepared k/link connectivity."""

    if not isinstance(context, PeriodicSourceContext):
        raise TypeError("Wannier90 MMN parsing requires PeriodicSourceContext.")
    if not isinstance(connectivity, PreparedReciprocalConnectivity):
        raise TypeError("Wannier90 MMN parsing requires PreparedReciprocalConnectivity.")
    context.admit_bytes(payload)
    if connectivity.plan.cell_id != context.basis.cell_id:
        raise ValueError(
            "MMN connectivity and source context use different periodic cells."
        )
    if len(payload) > int(maximum_bytes):
        raise PeriodicResourceError("Wannier90 MMN payload exceeds maximum_bytes.")
    lines = payload.decode("utf-8", errors="strict").splitlines()
    if len(lines) < 3 or not lines[0].strip():
        raise ValueError("Wannier90 MMN requires comment, dimensions, and link records.")
    dimensions = lines[1].split()
    if len(dimensions) != 3:
        raise ValueError("Wannier90 MMN dimension line requires three integers.")
    band_count, point_count, neighbors_per_point = (
        _integer(token, "dimension") for token in dimensions
    )
    if band_count != context.basis.orbital_count:
        raise ValueError("Wannier90 MMN band count does not match source basis order.")
    if point_count != int(connectivity.plan.mesh.fractional_points.shape[0]):
        raise ValueError("Wannier90 MMN k-point count does not match connectivity.")
    if band_count <= 0 or band_count > int(maximum_bands) or neighbors_per_point <= 0:
        raise PeriodicResourceError(
            "Wannier90 MMN band or neighbor count exceeds policy."
        )
    edge_count = point_count * neighbors_per_point
    record_count = edge_count * band_count * band_count
    if edge_count != connectivity.edge_count:
        raise ValueError(
            "Wannier90 MMN neighbor count does not cover prepared connectivity."
        )
    if record_count > int(maximum_records):
        raise PeriodicResourceError("Wannier90 MMN overlap record count exceeds policy.")
    expected_lines = 2 + edge_count * (1 + band_count * band_count)
    if len(lines) != expected_lines:
        raise ValueError("Wannier90 MMN payload is truncated or contains extra records.")
    plan = connectivity.plan
    edge_lookup = {
        (
            int(np.asarray(plan.source_indices)[edge]),
            int(np.asarray(plan.target_indices)[edge]),
            tuple(int(value) for value in np.asarray(plan.reciprocal_shifts)[edge]),
        ): edge
        for edge in range(connectivity.edge_count)
    }
    overlaps = np.empty((edge_count, band_count, band_count), dtype=np.complex128)
    covered = np.zeros((edge_count,), dtype=bool)
    cursor = 2
    for _ in range(edge_count):
        header = lines[cursor].split()
        cursor += 1
        if len(header) != 5:
            raise ValueError("Wannier90 MMN neighbor header requires five integers.")
        source = _integer(header[0], "source k index") - 1
        target = _integer(header[1], "target k index") - 1
        shift_three = tuple(_integer(token, "reciprocal shift") for token in header[2:])
        if context.basis.cell.rank < 3 and any(
            shift_three[axis] != 0 for axis in range(context.basis.cell.rank, 3)
        ):
            raise ValueError(
                "Wannier90 MMN uses reciprocal shifts outside source cell rank."
            )
        key = (source, target, shift_three[: context.basis.cell.rank])
        if key not in edge_lookup:
            raise ValueError(
                "Wannier90 MMN neighbor is absent from prepared connectivity."
            )
        edge = edge_lookup[key]
        if covered[edge]:
            raise ValueError("Wannier90 MMN duplicates a prepared connectivity edge.")
        covered[edge] = True
        for row in range(band_count):
            for column in range(band_count):
                tokens = lines[cursor].split()
                cursor += 1
                if len(tokens) != 2:
                    raise ValueError(
                        "Wannier90 MMN overlap records require real and imaginary fields."
                    )
                overlaps[edge, row, column] = complex(
                    _float(tokens[0], "real part"),
                    _float(tokens[1], "imaginary part"),
                )
    if not np.all(covered):
        raise ValueError("Wannier90 MMN does not completely cover prepared connectivity.")
    reverse = np.asarray(plan.reverse_indices)
    reverse_defect = np.max(
        np.abs(overlaps[reverse] - np.conj(np.swapaxes(overlaps, -1, -2))), initial=0.0
    )
    scale = max(float(np.max(np.abs(overlaps), initial=0.0)), 1.0)
    if reverse_defect > 1.0e-10 * scale:
        raise ValueError("Wannier90 MMN reverse link overlaps are not adjoints.")
    return Wannier90MMNImport(context, connectivity, overlaps)


def lower_wannier90_mmn(
    imported: Wannier90MMNImport,
    manifold: PeriodicBandManifold,
    /,
) -> PeriodicOverlapBundle:
    """Bind raw MMN links to one caller-selected complete band manifold."""

    if not isinstance(imported, Wannier90MMNImport):
        raise TypeError("imported must be Wannier90MMNImport.")
    if not isinstance(manifold, PeriodicBandManifold):
        raise TypeError("manifold must be PeriodicBandManifold.")
    if imported.raw_overlaps.shape[1] != manifold.dimension:
        raise ValueError(
            "MMN lowering requires a manifold covering the complete imported band window."
        )
    if imported.context.basis.basis_id != manifold.spectrum.basis_id:
        raise ValueError("MMN source basis does not match the selected manifold basis.")
    return PeriodicOverlapBundle(
        manifold,
        imported.connectivity,
        raw_overlaps=imported.raw_overlaps,
        source_id=f"wannier90-mmn:{imported.import_id}",
    )


def write_wannier90_mmn(imported: Wannier90MMNImport, /) -> bytes:
    """Emit canonical MMN bytes in source-k then prepared-edge order."""

    if not isinstance(imported, Wannier90MMNImport):
        raise TypeError("imported must be Wannier90MMNImport.")
    plan = imported.connectivity.plan
    source = np.asarray(plan.source_indices)
    target = np.asarray(plan.target_indices)
    shifts = np.asarray(plan.reciprocal_shifts)
    overlap = np.asarray(imported.raw_overlaps)
    point_count = int(plan.mesh.fractional_points.shape[0])
    counts = np.bincount(source, minlength=point_count)
    if np.any(counts != counts[0]):
        raise ValueError(
            "MMN writing requires uniform outgoing neighbor count per k point."
        )
    lines = [
        f"Phydrax canonical MMN export from {imported.context.provenance.source_id}",
        f"{overlap.shape[1]} {point_count} {int(counts[0])}",
    ]
    for point in range(point_count):
        for edge in np.flatnonzero(source == point):
            shift = np.pad(shifts[edge], (0, 3 - shifts.shape[1]))
            lines.append(
                f"{point + 1:5d} {int(target[edge]) + 1:5d} "
                f"{int(shift[0]):5d} {int(shift[1]):5d} {int(shift[2]):5d}"
            )
            for row in range(overlap.shape[1]):
                for column in range(overlap.shape[2]):
                    value = overlap[edge, row, column]
                    lines.append(f"{value.real: .17e} {value.imag: .17e}")
    return ("\n".join(lines) + "\n").encode("utf-8")


__all__ = [
    "Wannier90MMNImport",
    "lower_wannier90_mmn",
    "read_wannier90_mmn",
    "write_wannier90_mmn",
]
