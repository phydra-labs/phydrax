#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded import of G-EQDSK equilibria with explicit magnetic conventions."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ...._fingerprint import canonical_fingerprint
from ....interchange import (
    account_bounded_resource,
    AdapterReport,
    AdapterStatus,
    BoundedResource,
)
from ....qualification import ReferenceArtifactManifest
from .._conventions import (
    AxisymmetricMachineFrame,
    TokamakConventionTransform,
    TokamakMagneticConvention,
)
from .._equilibrium import AxisymmetricEquilibrium


_FLOAT = re.compile(r"[+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[EeDd][+-]?\d+)?")
_INTEGER = re.compile(r"[+-]?\d+")


@dataclass(frozen=True, slots=True)
class EqdskImportResult:
    equilibrium: AxisymmetricEquilibrium
    resource: BoundedResource
    reference: ReferenceArtifactManifest
    transform: TokamakConventionTransform
    descriptor: str
    report: AdapterReport


def _positive_count(value: str, name: str, /) -> int:
    parsed = int(value)
    if parsed < 2:
        raise ValueError(f"{name} must be at least two.")
    return parsed


def _integer_count(value: float, name: str, /) -> int:
    if not np.isfinite(value) or value < 0.0 or value != np.floor(value):
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def import_eqdsk(
    resource: BoundedResource,
    reference: ReferenceArtifactManifest,
    machine_frame: AxisymmetricMachineFrame,
    source_convention: TokamakMagneticConvention,
    /,
) -> EqdskImportResult:
    """Import one exact supported G-EQDSK resource into canonical Phydrax state."""

    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be BoundedResource.")
    if not isinstance(reference, ReferenceArtifactManifest):
        raise TypeError("reference must be ReferenceArtifactManifest.")
    if not isinstance(machine_frame, AxisymmetricMachineFrame):
        raise TypeError("machine_frame must be AxisymmetricMachineFrame.")
    if not isinstance(source_convention, TokamakMagneticConvention):
        raise TypeError("source_convention must be TokamakMagneticConvention.")
    reference.verify_bytes(resource.data)
    reference.require_rights()
    text = resource.data.decode("ascii", errors="strict")
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("G-EQDSK resource must contain a header and numeric body.")
    header_integers = _INTEGER.findall(lines[0][48:] if len(lines[0]) >= 48 else lines[0])
    if len(header_integers) < 2:
        raise ValueError("G-EQDSK header must terminate with grid dimensions.")
    nw = _positive_count(header_integers[-2], "nw")
    nh = _positive_count(header_integers[-1], "nh")
    if nw * nh > resource.manifest.limits.max_nodes:
        raise ValueError("G-EQDSK grid exceeds the bounded node limit.")
    descriptor = lines[0][: -len(header_integers[-1])].strip()
    if not descriptor:
        descriptor = "unnamed-geqdsk"
    tokens = _FLOAT.findall("\n".join(lines[1:]))
    values = np.asarray(
        [float(token.replace("D", "E").replace("d", "e")) for token in tokens],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(values)):
        raise ValueError("G-EQDSK numeric payload must be finite.")
    cursor = 0

    def take(count: int, name: str) -> np.ndarray:
        nonlocal cursor
        stop = cursor + count
        if stop > values.size:
            raise ValueError(f"G-EQDSK payload is truncated while reading {name}.")
        result = values[cursor:stop]
        cursor = stop
        return result

    scalars = take(20, "header scalars")
    rdim, zdim, rcentr, rleft, zmid = scalars[:5]
    rmaxis, zmaxis, simag, sibry, bcentr = scalars[5:10]
    current = scalars[10]
    if rdim <= 0.0 or zdim <= 0.0 or rleft <= 0.0:
        raise ValueError("G-EQDSK R-Z domain dimensions must be positive.")
    fpol = take(nw, "fpol")
    pressure = take(nw, "pressure")
    ffprime = take(nw, "ffprime")
    pprime = take(nw, "pprime")
    psi_rz = take(nw * nh, "poloidal flux").reshape((nh, nw))
    safety_factor = take(nw, "safety factor")
    counts = take(2, "boundary counts")
    boundary_count = _integer_count(counts[0], "boundary point count")
    limiter_count = _integer_count(counts[1], "limiter point count")
    if boundary_count + limiter_count > resource.manifest.limits.max_nodes:
        raise ValueError("G-EQDSK contour points exceed the bounded node limit.")
    boundary = take(2 * boundary_count, "boundary points").reshape((boundary_count, 2))
    limiter = take(2 * limiter_count, "limiter points").reshape((limiter_count, 2))
    if cursor != values.size:
        raise ValueError("G-EQDSK payload contains unsupported trailing numeric fields.")

    canonical = TokamakMagneticConvention.canonical()
    transform = source_convention.transform_to(canonical)
    psi_factor = transform.poloidal_flux_factor
    field_factor = transform.toroidal_field_factor
    source_id = canonical_fingerprint(
        {
            "kind": "eqdsk-import-source",
            "resource": resource.manifest.manifest_id,
            "reference": reference.manifest_id,
            "transform": transform.transform_id,
        }
    )
    equilibrium = AxisymmetricEquilibrium(
        machine_frame,
        canonical,
        np.linspace(rleft, rleft + rdim, nw),
        np.linspace(zmid - 0.5 * zdim, zmid + 0.5 * zdim, nh),
        psi_factor * psi_rz,
        field_factor * fpol,
        pressure,
        field_factor**2 / psi_factor * ffprime,
        pprime / psi_factor,
        transform.safety_factor_factor * safety_factor,
        boundary,
        limiter,
        np.asarray([rmaxis, zmaxis]),
        psi_factor * simag,
        psi_factor * sibry,
        rcentr,
        field_factor * bcentr,
        transform.plasma_current_factor * current,
        source_id,
    )
    accounted = account_bounded_resource(
        resource,
        depth=1,
        nodes=nw * nh + boundary_count + limiter_count,
        attributes=20 + 5 * nw,
        losses=0,
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "G-EQDSK",
        "AxisymmetricEquilibrium",
        source_id=accounted.manifest.manifest_id,
        target_id=equilibrium.equilibrium_id,
        coordinate_mapping=(
            "source R-Z coordinates preserved in metres",
            f"poloidal flux multiplied by {psi_factor:.17g}",
            f"toroidal field and F multiplied by {field_factor:.17g}",
            f"plasma current multiplied by {transform.plasma_current_factor:.17g}",
            f"safety factor multiplied by {transform.safety_factor_factor:.17g}",
        ),
        preserved_fields=(
            "poloidal_flux",
            "fpol",
            "pressure",
            "ffprime",
            "pprime",
            "safety_factor",
            "plasma_boundary",
            "limiter",
            "magnetic_axis",
            "plasma_current",
        ),
        assumptions=(
            "caller supplied the complete source magnetic convention",
            "G-EQDSK coordinates are metres and pressure is pascals",
            "no trailing producer-specific numeric extension is present",
        ),
    )
    return EqdskImportResult(
        equilibrium, accounted, reference, transform, descriptor, report
    )


__all__ = ["EqdskImportResult", "import_eqdsk"]
