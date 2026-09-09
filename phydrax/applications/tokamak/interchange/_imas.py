#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact mapping boundary for one IMAS equilibrium time slice."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....interchange import AdapterReport, AdapterStatus
from ....qualification import ReferenceArtifactManifest
from .._conventions import AxisymmetricMachineFrame, TokamakMagneticConvention
from .._equilibrium import AxisymmetricEquilibrium


_FIELDS = {
    "ids_name",
    "ids_release",
    "occurrence",
    "time_s",
    "convention_id",
    "r_m",
    "z_m",
    "poloidal_flux",
    "f_rb_t_m",
    "pressure_pa",
    "ffprime",
    "pprime",
    "safety_factor",
    "boundary_rz_m",
    "limiter_rz_m",
    "magnetic_axis_rz_m",
    "magnetic_axis_flux",
    "boundary_flux",
    "reference_major_radius_m",
    "reference_toroidal_field_t",
    "plasma_current_a",
    "source_id",
}


@dataclass(frozen=True, slots=True)
class ImasEquilibriumImportResult:
    equilibrium: AxisymmetricEquilibrium
    reference: ReferenceArtifactManifest
    report: AdapterReport
    ids_release: str
    occurrence: int
    time_s: float


@dataclass(frozen=True, slots=True)
class ImasEquilibriumExportResult:
    record: Mapping[str, object]
    report: AdapterReport


def _text(value, name: str):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be non-empty canonical text.")
    return normalized


def import_imas_equilibrium_slice(
    record: Mapping[str, object],
    reference: ReferenceArtifactManifest,
    machine_frame: AxisymmetricMachineFrame,
    source_convention: TokamakMagneticConvention,
    /,
) -> ImasEquilibriumImportResult:
    """Import an explicitly extracted IMAS equilibrium IDS time slice."""

    if not isinstance(record, Mapping) or set(record) != _FIELDS:
        raise ValueError("IMAS equilibrium record must use exactly the supported fields.")
    if record["ids_name"] != "equilibrium":
        raise ValueError("Supported IMAS record must identify the equilibrium IDS.")
    if not isinstance(reference, ReferenceArtifactManifest):
        raise TypeError("reference must be ReferenceArtifactManifest.")
    if not isinstance(machine_frame, AxisymmetricMachineFrame):
        raise TypeError("machine_frame must be AxisymmetricMachineFrame.")
    if not isinstance(source_convention, TokamakMagneticConvention):
        raise TypeError("source_convention must be TokamakMagneticConvention.")
    if record["convention_id"] != source_convention.convention_id:
        raise ValueError(
            "IMAS record convention identity does not match its supplied convention."
        )
    ids_release = _text(record["ids_release"], "ids_release")
    occurrence = int(record["occurrence"])
    time_s = float(record["time_s"])
    if occurrence < 0 or not math.isfinite(time_s):
        raise ValueError("IMAS occurrence and time are invalid.")
    canonical = TokamakMagneticConvention.canonical()
    transform = source_convention.transform_to(canonical)
    flux = transform.poloidal_flux_factor
    field = transform.toroidal_field_factor
    source_id = canonical_fingerprint(
        {
            "kind": "imas-equilibrium-import",
            "record_source": _text(record["source_id"], "source_id"),
            "reference": reference.manifest_id,
            "ids_release": ids_release,
            "occurrence": occurrence,
            "time_s": time_s,
            "transform": transform.transform_id,
        }
    )
    equilibrium = AxisymmetricEquilibrium(
        machine_frame,
        canonical,
        np.asarray(record["r_m"]),
        np.asarray(record["z_m"]),
        flux * np.asarray(record["poloidal_flux"]),
        field * np.asarray(record["f_rb_t_m"]),
        np.asarray(record["pressure_pa"]),
        field**2 / flux * np.asarray(record["ffprime"]),
        np.asarray(record["pprime"]) / flux,
        transform.safety_factor_factor * np.asarray(record["safety_factor"]),
        np.asarray(record["boundary_rz_m"]),
        np.asarray(record["limiter_rz_m"]),
        np.asarray(record["magnetic_axis_rz_m"]),
        flux * float(record["magnetic_axis_flux"]),
        flux * float(record["boundary_flux"]),
        float(record["reference_major_radius_m"]),
        field * float(record["reference_toroidal_field_t"]),
        transform.plasma_current_factor * float(record["plasma_current_a"]),
        source_id,
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "IMAS-equilibrium-slice",
        "AxisymmetricEquilibrium",
        source_id=source_id,
        target_id=equilibrium.equilibrium_id,
        coordinate_mapping=(
            f"poloidal flux multiplied by {flux:.17g}",
            f"toroidal field multiplied by {field:.17g}",
            f"plasma current multiplied by {transform.plasma_current_factor:.17g}",
            f"safety factor multiplied by {transform.safety_factor_factor:.17g}",
        ),
        preserved_fields=tuple(sorted(_FIELDS - {"ids_name"})),
        assumptions=(
            "caller extracted exactly one equilibrium IDS time slice",
            "record arrays use SI units named by their fields",
            "source magnetic convention is supplied independently and exactly",
        ),
    )
    return ImasEquilibriumImportResult(
        equilibrium, reference, report, ids_release, occurrence, time_s
    )


def export_imas_equilibrium_slice(
    equilibrium: AxisymmetricEquilibrium,
    target_convention: TokamakMagneticConvention,
    /,
    *,
    ids_release: str,
    occurrence: int,
    time_s: float,
) -> ImasEquilibriumExportResult:
    """Export one native equilibrium to the supported IMAS slice mapping."""

    if not isinstance(equilibrium, AxisymmetricEquilibrium):
        raise TypeError("equilibrium must be AxisymmetricEquilibrium.")
    if not isinstance(target_convention, TokamakMagneticConvention):
        raise TypeError("target_convention must be TokamakMagneticConvention.")
    release = _text(ids_release, "ids_release")
    occurrence_ = int(occurrence)
    time = float(time_s)
    if occurrence_ < 0 or not math.isfinite(time):
        raise ValueError("IMAS occurrence and time are invalid.")
    transform = equilibrium.convention.transform_to(target_convention)
    flux = transform.poloidal_flux_factor
    field = transform.toroidal_field_factor
    record = {
        "ids_name": "equilibrium",
        "ids_release": release,
        "occurrence": occurrence_,
        "time_s": time,
        "convention_id": target_convention.convention_id,
        "r_m": np.array(equilibrium.r_m, copy=True),
        "z_m": np.array(equilibrium.z_m, copy=True),
        "poloidal_flux": flux * np.asarray(equilibrium.poloidal_flux_wb_per_rad),
        "f_rb_t_m": field * np.asarray(equilibrium.f_rb_t_m),
        "pressure_pa": np.array(equilibrium.pressure_pa, copy=True),
        "ffprime": field**2 / flux * np.asarray(equilibrium.ffprime_t2_m2_per_wb_rad),
        "pprime": np.asarray(equilibrium.pprime_pa_per_wb_rad) / flux,
        "safety_factor": transform.safety_factor_factor
        * np.asarray(equilibrium.safety_factor),
        "boundary_rz_m": np.array(equilibrium.boundary_rz_m, copy=True),
        "limiter_rz_m": np.array(equilibrium.limiter_rz_m, copy=True),
        "magnetic_axis_rz_m": np.array(equilibrium.magnetic_axis_rz_m, copy=True),
        "magnetic_axis_flux": flux * equilibrium.magnetic_axis_flux_wb_per_rad,
        "boundary_flux": flux * equilibrium.boundary_flux_wb_per_rad,
        "reference_major_radius_m": equilibrium.reference_major_radius_m,
        "reference_toroidal_field_t": field * equilibrium.reference_toroidal_field_t,
        "plasma_current_a": transform.plasma_current_factor
        * equilibrium.plasma_current_a,
        "source_id": equilibrium.equilibrium_id,
    }
    target_id = canonical_fingerprint(
        {
            "kind": "imas-equilibrium-export",
            "source": equilibrium.equilibrium_id,
            "ids_release": release,
            "occurrence": occurrence_,
            "time_s": time,
            "transform": transform.transform_id,
            "arrays": array_tree_fingerprint(
                tuple(
                    record[key]
                    for key in sorted(_FIELDS)
                    if isinstance(record[key], np.ndarray)
                )
            ),
        }
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "AxisymmetricEquilibrium",
        "IMAS-equilibrium-slice",
        source_id=equilibrium.equilibrium_id,
        target_id=target_id,
        coordinate_mapping=(
            f"poloidal flux multiplied by {flux:.17g}",
            f"toroidal field multiplied by {field:.17g}",
            f"plasma current multiplied by {transform.plasma_current_factor:.17g}",
            f"safety factor multiplied by {transform.safety_factor_factor:.17g}",
        ),
        preserved_fields=tuple(sorted(_FIELDS - {"ids_name"})),
        assumptions=(
            "consumer maps this exact record into its installed IMAS release",
            "no live IMAS database or actor is invoked by this pure adapter",
        ),
    )
    return ImasEquilibriumExportResult(record, report)


__all__ = [
    "ImasEquilibriumExportResult",
    "ImasEquilibriumImportResult",
    "export_imas_equilibrium_slice",
    "import_imas_equilibrium_slice",
]
