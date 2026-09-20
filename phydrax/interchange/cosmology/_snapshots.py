#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rights-checked admission of documented cosmological snapshot products."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Literal

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from ..._external_resource import BoundedResource
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...applications.cosmology._force_scalability import CosmologySnapshotProduct
from ...artifacts import ScientificArtifactEnvelope
from ...qualification import read_reference_artifact, ReferenceArtifactManifest
from .._report import AdapterLoss, AdapterReport, AdapterStatus


class ConceptSnapshotImport(StrictModule, NonTrainableState):
    """Native particle snapshot together with its exact admitted source."""

    snapshot: CosmologySnapshotProduct
    source: ReferenceArtifactManifest
    report: AdapterReport
    component_name: str = eqx.field(static=True)
    coordinate_frame: str = eqx.field(static=True)
    position_unit: str = eqx.field(static=True)
    momentum_unit: str = eqx.field(static=True)
    mass_unit: str = eqx.field(static=True)
    source_kinematic_kind: str = eqx.field(static=True)
    source_time_unit: str = eqx.field(static=True)
    source_length_unit: str = eqx.field(static=True)
    source_mass_unit: str = eqx.field(static=True)

    def __init__(
        self,
        snapshot: CosmologySnapshotProduct,
        source: ReferenceArtifactManifest,
        report: AdapterReport,
        /,
        *,
        component_name: str,
        coordinate_frame: str,
        position_unit: str,
        momentum_unit: str,
        mass_unit: str,
        source_kinematic_kind: str,
        source_time_unit: str,
        source_length_unit: str,
        source_mass_unit: str,
    ):
        if not isinstance(snapshot, CosmologySnapshotProduct):
            raise TypeError("snapshot must be CosmologySnapshotProduct.")
        if not isinstance(source, ReferenceArtifactManifest):
            raise TypeError("source must be ReferenceArtifactManifest.")
        if not isinstance(report, AdapterReport):
            raise TypeError("report must be AdapterReport.")
        labels = tuple(
            str(value).strip()
            for value in (
                component_name,
                coordinate_frame,
                position_unit,
                momentum_unit,
                mass_unit,
                source_kinematic_kind,
                source_time_unit,
                source_length_unit,
                source_mass_unit,
            )
        )
        if any(not value for value in labels):
            raise ValueError("CONCEPT snapshot semantic labels must be non-empty.")
        self.snapshot = snapshot
        self.source = source
        self.report = report
        (
            self.component_name,
            self.coordinate_frame,
            self.position_unit,
            self.momentum_unit,
            self.mass_unit,
            self.source_kinematic_kind,
            self.source_time_unit,
            self.source_length_unit,
            self.source_mass_unit,
        ) = labels


def _admit_path(
    path: str | Path,
    source: ReferenceArtifactManifest,
    /,
    *,
    maximum_source_bytes: int,
    commercial_use: bool,
    training_use: bool,
    redistribution: bool,
    export: bool,
) -> BoundedResource:
    if not isinstance(source, ReferenceArtifactManifest):
        raise TypeError("source must be ReferenceArtifactManifest.")
    if isinstance(maximum_source_bytes, bool) or int(maximum_source_bytes) <= 0:
        raise ValueError("maximum_source_bytes must be a positive integer.")
    source.require_rights(
        commercial_use=commercial_use,
        training_use=training_use,
        redistribution=redistribution,
        export=export,
    )
    if source.size_bytes > int(maximum_source_bytes):
        raise MemoryError("Cosmology source exceeds maximum_source_bytes.")
    return read_reference_artifact(path, source)


def _text(value: object, name: str, /) -> str:
    if isinstance(value, bytes):
        result = value.decode("utf-8")
    elif isinstance(value, str):
        result = value
    else:
        raise TypeError(f"{name} must be a UTF-8 string attribute.")
    result = result.strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _positive_scale(value: float, name: str, /) -> float:
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return scale


def _artifact(
    source: ReferenceArtifactManifest,
    /,
    *,
    kind: str,
    producer: str,
    producer_version: str,
) -> ScientificArtifactEnvelope:
    return ScientificArtifactEnvelope(
        artifact_kind=kind,
        content_digest=f"{source.checksum_algorithm}:{source.checksum}",
        producer=producer,
        producer_version=producer_version,
        build_id=source.manifest_id,
        license_id=source.license_id,
        resource_id=source.manifest_id,
        status="complete",
    )


def read_concept_snapshot(
    path: str | Path,
    source: ReferenceArtifactManifest,
    /,
    *,
    component: str,
    coordinate_frame: Literal["comoving"] = "comoving",
    kinematic_kind: Literal["canonical_momentum", "peculiar_velocity"] | None = None,
    position_scale: float = 1.0,
    mass_scale: float = 1.0,
    time_scale: float = 1.0,
    target_length_unit: str = "source-length-unit",
    target_mass_unit: str = "source-mass-unit",
    maximum_particles: int = 10_000_000,
    maximum_source_bytes: int = 2_000_000_000,
    commercial_use: bool = False,
    training_use: bool = False,
    redistribution: bool = False,
    export: bool = False,
) -> ConceptSnapshotImport:
    """Read one documented CONCEPT particle component without guessing semantics.

    Native ``mom`` is canonical momentum. A non-native ``vel`` fixture is admitted
    only when explicitly identified as peculiar velocity and is converted as
    ``p = a m v_pec``. Position, mass and time scales map producer base units into
    the target units named on the returned immutable import.
    """

    resource = _admit_path(
        path,
        source,
        maximum_source_bytes=maximum_source_bytes,
        commercial_use=commercial_use,
        training_use=training_use,
        redistribution=redistribution,
        export=export,
    )
    component_name = str(component).strip()
    if not component_name:
        raise ValueError("component must be non-empty.")
    if coordinate_frame != "comoving":
        raise ValueError("CONCEPT particle positions must be declared comoving.")
    if isinstance(maximum_particles, bool) or int(maximum_particles) <= 0:
        raise ValueError("maximum_particles must be a positive integer.")
    length_factor = _positive_scale(position_scale, "position_scale")
    mass_factor = _positive_scale(mass_scale, "mass_scale")
    time_factor = _positive_scale(time_scale, "time_scale")
    length_unit = str(target_length_unit).strip()
    mass_unit = str(target_mass_unit).strip()
    if not length_unit or not mass_unit:
        raise ValueError("Target length and mass units must be explicit.")
    losses: list[AdapterLoss] = []
    with h5py.File(BytesIO(resource.data), "r") as handle:
        required_attributes = ("a", "boxsize", "unit time", "unit length", "unit mass")
        missing_attributes = tuple(
            name for name in required_attributes if name not in handle.attrs
        )
        if missing_attributes:
            raise ValueError(
                f"CONCEPT snapshot has ambiguous units or scale metadata: {missing_attributes}."
            )
        source_time_unit = _text(handle.attrs["unit time"], "unit time")
        source_length_unit = _text(handle.attrs["unit length"], "unit length")
        source_mass_unit = _text(handle.attrs["unit mass"], "unit mass")
        scale_factor = float(handle.attrs["a"])
        box_size = float(handle.attrs["boxsize"]) * length_factor
        if (
            not np.isfinite(scale_factor)
            or scale_factor <= 0.0
            or not np.isfinite(box_size)
            or box_size <= 0.0
        ):
            raise ValueError(
                "CONCEPT scale factor and box size must be finite and positive."
            )
        group_path = f"components/{component_name}"
        if group_path not in handle:
            raise ValueError(f"CONCEPT particle component {component_name!r} is absent.")
        group = handle[group_path]
        if "pos" not in group:
            raise ValueError(
                "CONCEPT particle component omits required comoving positions."
            )
        available = tuple(name for name in ("mom", "vel") if name in group)
        if kinematic_kind is None:
            if available == ("mom",):
                selected_kind = "canonical_momentum"
            elif available == ("vel",):
                raise ValueError(
                    "A velocity dataset requires kinematic_kind='peculiar_velocity'; "
                    "velocity and canonical momentum are not inferred as aliases."
                )
            else:
                raise ValueError(
                    "CONCEPT kinematics are absent or ambiguous; select exactly one semantic."
                )
        else:
            selected_kind = kinematic_kind
        dataset_name = "mom" if selected_kind == "canonical_momentum" else "vel"
        if dataset_name not in group:
            raise ValueError(f"Declared CONCEPT {selected_kind} dataset is absent.")
        if len(available) > 1 and kinematic_kind is None:
            raise ValueError("CONCEPT velocity versus momentum semantics are ambiguous.")
        positions = np.asarray(group["pos"], dtype=np.float64)
        kinematics = np.asarray(group[dataset_name], dtype=np.float64)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or kinematics.shape != positions.shape
        ):
            raise ValueError(
                "CONCEPT particle pos/kinematic datasets must have shape (N, 3)."
            )
        count = positions.shape[0]
        if count < 1 or count > int(maximum_particles):
            raise MemoryError("CONCEPT particle count violates the configured bound.")
        if "N" in group.attrs and int(group.attrs["N"]) != count:
            raise ValueError("CONCEPT component N attribute disagrees with its datasets.")
        if "mass" in group:
            masses = np.asarray(group["mass"], dtype=np.float64)
        elif "mass" in group.attrs:
            masses = np.full(count, float(group.attrs["mass"]), dtype=np.float64)
        else:
            raise ValueError(
                "CONCEPT particle masses are required for canonical momentum."
            )
        if masses.shape == ():
            masses = np.full(count, float(masses), dtype=np.float64)
        if (
            masses.shape != (count,)
            or np.any(~np.isfinite(masses))
            or np.any(masses <= 0.0)
        ):
            raise ValueError(
                "CONCEPT particle masses must be finite, positive and row-aligned."
            )
        if "ids" in group:
            particle_ids = np.asarray(group["ids"], dtype=np.int64)
        else:
            particle_ids = np.arange(count, dtype=np.int64)
            losses.append(
                AdapterLoss(
                    f"{group_path}/ids",
                    "import",
                    "synthesized",
                    "The component omitted persistent particle IDs; bounded row IDs were synthesized.",
                    changes_interpretation=False,
                )
            )
        if particle_ids.shape != (count,) or np.any(particle_ids < 0):
            raise ValueError("CONCEPT particle IDs must be non-negative and row-aligned.")
        if len(set(particle_ids.tolist())) != count:
            raise ValueError("CONCEPT particle IDs must be unique.")
    positions = positions * length_factor
    masses = masses * mass_factor
    velocity_factor = length_factor / time_factor
    if selected_kind == "canonical_momentum":
        canonical_momenta = kinematics * mass_factor * velocity_factor
    else:
        canonical_momenta = masses[:, None] * scale_factor * kinematics * velocity_factor
        losses.append(
            AdapterLoss(
                f"components/{component_name}/vel",
                "import",
                "transformed",
                "Peculiar velocity was exactly canonicalized as p = a m v_pec.",
                changes_interpretation=False,
            )
        )
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(canonical_momenta)):
        raise ValueError("CONCEPT active particle phase-space values must be finite.")
    artifact = _artifact(
        source,
        kind="cosmology-snapshot",
        producer="CONCEPT",
        producer_version="documented-native-hdf5",
    )
    snapshot = CosmologySnapshotProduct(
        jax.lax.stop_gradient(jnp.asarray(particle_ids)),
        jax.lax.stop_gradient(jnp.asarray(positions)),
        jax.lax.stop_gradient(jnp.asarray(canonical_momenta)),
        jax.lax.stop_gradient(jnp.asarray(masses)),
        scale_factor,
        (box_size, box_size, box_size),
        artifact,
    )
    status = AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS
    report = AdapterReport(
        status,
        "CONCEPT native HDF5 particle snapshot",
        "CosmologySnapshotProduct",
        source_id=source.manifest_id,
        target_id=snapshot.snapshot_id,
        coordinate_mapping=(
            f"components/{component_name}/pos -> comoving Cartesian positions [{length_unit}]",
            f"components/{component_name}/{dataset_name} -> canonical momentum [{mass_unit}*{length_unit}/time]",
        ),
        preserved_fields=(
            "particle_ids",
            "positions",
            "masses",
            "scale_factor",
            "box_size",
            "source_unit_time",
            "source_unit_length",
            "source_unit_mass",
            "source_rights",
        ),
        assumptions=(
            "The selected CONCEPT component uses the documented particle representation.",
            "The caller-provided positive scales map declared producer base units to target units.",
        ),
        losses=tuple(losses),
    )
    return ConceptSnapshotImport(
        snapshot,
        source,
        report,
        component_name=component_name,
        coordinate_frame=coordinate_frame,
        position_unit=length_unit,
        momentum_unit=f"{mass_unit}*{length_unit}/time",
        mass_unit=mass_unit,
        source_kinematic_kind=selected_kind,
        source_time_unit=source_time_unit,
        source_length_unit=source_length_unit,
        source_mass_unit=source_mass_unit,
    )


__all__ = ["ConceptSnapshotImport", "read_concept_snapshot"]
