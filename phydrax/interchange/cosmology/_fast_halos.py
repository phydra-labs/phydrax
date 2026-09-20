#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit ASCII admission for documented PINOCCHIO halo products."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...applications.cosmology._halo_models import HaloCatalog
from ...applications.cosmology._halos import SphericalOverdensityMassDefinition
from ...qualification import ReferenceArtifactManifest
from .._report import AdapterLoss, AdapterReport, AdapterStatus
from ._snapshots import _admit_path, _artifact, _positive_scale


_PINOCCHIO_APPROXIMATION = "PINOCCHIO-LPT-fragmentation-approximation"


class PinocchioCatalogSidecar(StrictModule, NonTrainableState):
    initial_positions: Array
    particle_counts: Array
    particle_count_known: Array
    active_mask: Array
    approximation_tag: str = eqx.field(static=True)
    sidecar_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_positions: Array,
        particle_counts: Array,
        particle_count_known: Array,
        active_mask: Array,
        /,
    ):
        initial = jax.lax.stop_gradient(jnp.asarray(initial_positions))
        counts = jax.lax.stop_gradient(jnp.asarray(particle_counts, dtype=jnp.int64))
        known = jax.lax.stop_gradient(jnp.asarray(particle_count_known, dtype=jnp.bool_))
        active = jax.lax.stop_gradient(jnp.asarray(active_mask, dtype=jnp.bool_))
        if (
            initial.ndim != 2
            or initial.shape[1] != 3
            or counts.shape != initial.shape[:1]
            or known.shape != counts.shape
            or active.shape != counts.shape
        ):
            raise ValueError("PINOCCHIO catalog sidecar capacities are inconsistent.")
        self.initial_positions = initial
        self.particle_counts = counts
        self.particle_count_known = known
        self.active_mask = active
        self.approximation_tag = _PINOCCHIO_APPROXIMATION
        self.sidecar_id = canonical_fingerprint(
            {
                "kind": "pinocchio-catalog-sidecar",
                "approximation": self.approximation_tag,
                "arrays": array_tree_fingerprint((initial, counts, known, active)),
            }
        )


class PinocchioLightConeProduct(StrictModule, NonTrainableState):
    halo_ids: Array
    true_redshifts: Array
    comoving_positions: Array
    velocities: Array
    masses: Array
    theta_degrees: Array
    phi_degrees: Array
    radial_velocities: Array
    observed_redshifts: Array
    phase_space_known: Array
    active_mask: Array
    approximation_tag: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        halo_ids: Array,
        true_redshifts: Array,
        comoving_positions: Array,
        velocities: Array,
        masses: Array,
        theta_degrees: Array,
        phi_degrees: Array,
        radial_velocities: Array,
        observed_redshifts: Array,
        phase_space_known: Array,
        active_mask: Array,
        /,
    ):
        ids = jax.lax.stop_gradient(jnp.asarray(halo_ids, dtype=jnp.int64))
        redshift = jax.lax.stop_gradient(jnp.asarray(true_redshifts))
        positions = jax.lax.stop_gradient(jnp.asarray(comoving_positions))
        velocity = jax.lax.stop_gradient(jnp.asarray(velocities))
        mass = jax.lax.stop_gradient(jnp.asarray(masses))
        theta = jax.lax.stop_gradient(jnp.asarray(theta_degrees))
        phi = jax.lax.stop_gradient(jnp.asarray(phi_degrees))
        radial = jax.lax.stop_gradient(jnp.asarray(radial_velocities))
        observed = jax.lax.stop_gradient(jnp.asarray(observed_redshifts))
        known = jax.lax.stop_gradient(jnp.asarray(phase_space_known, dtype=jnp.bool_))
        active = jax.lax.stop_gradient(jnp.asarray(active_mask, dtype=jnp.bool_))
        if (
            ids.ndim != 1
            or redshift.shape != ids.shape
            or positions.shape != (ids.size, 3)
            or velocity.shape != positions.shape
            or mass.shape != ids.shape
            or theta.shape != ids.shape
            or phi.shape != ids.shape
            or radial.shape != ids.shape
            or observed.shape != ids.shape
            or known.shape != ids.shape
            or active.shape != ids.shape
        ):
            raise ValueError("PINOCCHIO light-cone arrays have inconsistent capacities.")
        self.halo_ids = ids
        self.true_redshifts = redshift
        self.comoving_positions = positions
        self.velocities = velocity
        self.masses = mass
        self.theta_degrees = theta
        self.phi_degrees = phi
        self.radial_velocities = radial
        self.observed_redshifts = observed
        self.phase_space_known = known
        self.active_mask = active
        self.approximation_tag = _PINOCCHIO_APPROXIMATION
        self.product_id = canonical_fingerprint(
            {
                "kind": "pinocchio-light-cone",
                "approximation": self.approximation_tag,
                "arrays": array_tree_fingerprint(
                    (
                        ids,
                        redshift,
                        positions,
                        velocity,
                        mass,
                        theta,
                        phi,
                        radial,
                        observed,
                        known,
                        active,
                    )
                ),
            }
        )


class PinocchioCatalogImport(StrictModule, NonTrainableState):
    catalog: HaloCatalog | None
    light_cone: PinocchioLightConeProduct | None
    sidecar: PinocchioCatalogSidecar | None
    source: ReferenceArtifactManifest
    report: AdapterReport
    product_kind: str = eqx.field(static=True)
    coordinate_frame: str = eqx.field(static=True)
    source_position_unit: str = eqx.field(static=True)
    source_mass_unit: str = eqx.field(static=True)
    source_velocity_unit: str = eqx.field(static=True)

    def __init__(
        self,
        catalog: HaloCatalog | None,
        light_cone: PinocchioLightConeProduct | None,
        sidecar: PinocchioCatalogSidecar | None,
        source: ReferenceArtifactManifest,
        report: AdapterReport,
        /,
        *,
        product_kind: str,
        coordinate_frame: str,
        source_position_unit: str,
        source_mass_unit: str,
        source_velocity_unit: str,
    ):
        if (catalog is None) == (light_cone is None):
            raise ValueError(
                "Exactly one PINOCCHIO native catalog projection is required."
            )
        if product_kind == "catalog" and (catalog is None or sidecar is None):
            raise ValueError("PINOCCHIO catalogs require HaloCatalog and its sidecar.")
        if product_kind == "light_cone" and light_cone is None:
            raise ValueError(
                "PINOCCHIO light-cone import requires its native projection."
            )
        self.catalog = catalog
        self.light_cone = light_cone
        self.sidecar = sidecar
        self.source = source
        self.report = report
        self.product_kind = product_kind
        labels = tuple(
            str(value).strip()
            for value in (
                coordinate_frame,
                source_position_unit,
                source_mass_unit,
                source_velocity_unit,
            )
        )
        if any(not value for value in labels):
            raise ValueError(
                "PINOCCHIO coordinate frame and source units must be explicit."
            )
        (
            self.coordinate_frame,
            self.source_position_unit,
            self.source_mass_unit,
            self.source_velocity_unit,
        ) = labels


class PinocchioMergerHistory(StrictModule, NonTrainableState):
    group_ids: Array
    tree_numbers: Array
    indices_within_tree: Array
    linking_indices: Array
    merged_with_indices: Array
    sink_group_ids: Array
    descendant_group_ids: Array
    halo_masses_at_merger_particles: Array
    main_masses_at_merger_particles: Array
    merger_redshifts: Array
    peak_collapse_redshifts: Array
    minimum_mass_redshifts: Array
    active_mask: Array
    approximation_tag: str = eqx.field(static=True)
    history_id: str = eqx.field(static=True)

    def __init__(
        self,
        group_ids: Array,
        tree_numbers: Array,
        indices_within_tree: Array,
        linking_indices: Array,
        merged_with_indices: Array,
        sink_group_ids: Array,
        descendant_group_ids: Array,
        halo_masses_at_merger_particles: Array,
        main_masses_at_merger_particles: Array,
        merger_redshifts: Array,
        peak_collapse_redshifts: Array,
        minimum_mass_redshifts: Array,
        active_mask: Array,
        /,
    ):
        arrays = tuple(
            jax.lax.stop_gradient(jnp.asarray(value))
            for value in (
                group_ids,
                tree_numbers,
                indices_within_tree,
                linking_indices,
                merged_with_indices,
                sink_group_ids,
                descendant_group_ids,
                halo_masses_at_merger_particles,
                main_masses_at_merger_particles,
                merger_redshifts,
                peak_collapse_redshifts,
                minimum_mass_redshifts,
                active_mask,
            )
        )
        if arrays[0].ndim != 1 or any(
            value.shape != arrays[0].shape for value in arrays[1:]
        ):
            raise ValueError("PINOCCHIO history columns must share one fixed capacity.")
        (
            self.group_ids,
            self.tree_numbers,
            self.indices_within_tree,
            self.linking_indices,
            self.merged_with_indices,
            self.sink_group_ids,
            self.descendant_group_ids,
            self.halo_masses_at_merger_particles,
            self.main_masses_at_merger_particles,
            self.merger_redshifts,
            self.peak_collapse_redshifts,
            self.minimum_mass_redshifts,
            self.active_mask,
        ) = arrays
        self.approximation_tag = _PINOCCHIO_APPROXIMATION
        self.history_id = canonical_fingerprint(
            {
                "kind": "pinocchio-merger-history",
                "approximation": self.approximation_tag,
                "arrays": array_tree_fingerprint(arrays),
            }
        )


class PinocchioLineageImport(StrictModule, NonTrainableState):
    history: PinocchioMergerHistory
    source: ReferenceArtifactManifest
    report: AdapterReport
    declared_tree_count: int = eqx.field(static=True)
    declared_branch_count: int = eqx.field(static=True)

    def __init__(
        self,
        history: PinocchioMergerHistory,
        source: ReferenceArtifactManifest,
        report: AdapterReport,
        declared_tree_count: int,
        declared_branch_count: int,
        /,
    ):
        self.history = history
        self.source = source
        self.report = report
        self.declared_tree_count = int(declared_tree_count)
        self.declared_branch_count = int(declared_branch_count)


def _approximation_loss(path: str, rationale: str, /) -> AdapterLoss:
    return AdapterLoss(
        path,
        "import",
        "transformed",
        rationale,
        changes_interpretation=True,
    )


def _load_ascii(data: bytes, /) -> np.ndarray:
    try:
        values = np.loadtxt(io.BytesIO(data), comments="#", ndmin=2)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("PINOCCHIO ASCII product is malformed.") from error
    if values.ndim != 2 or values.shape[0] < 1 or np.any(~np.isfinite(values)):
        raise ValueError("PINOCCHIO ASCII product must contain finite numeric rows.")
    return values


def read_pinocchio_catalog(
    path: str | Path,
    source: ReferenceArtifactManifest,
    /,
    *,
    product_kind: Literal["catalog", "light_cone"] = "catalog",
    maximum_halos: int,
    coordinate_frame: Literal["periodic_comoving_box", "observer_relative_comoving"],
    source_position_unit: str,
    source_mass_unit: str,
    source_velocity_unit: str,
    mass_definition: SphericalOverdensityMassDefinition | None = None,
    scale_factor: float | None = None,
    box_size: tuple[float, float, float] | None = None,
    position_scale: float = 1.0,
    mass_scale: float = 1.0,
    velocity_scale: float = 1.0,
    maximum_source_bytes: int = 2_000_000_000,
    commercial_use: bool = False,
    training_use: bool = False,
    redistribution: bool = False,
    export: bool = False,
) -> PinocchioCatalogImport:
    """Read documented PINOCCHIO ASCII catalog or past-light-cone rows."""

    if isinstance(maximum_halos, bool) or int(maximum_halos) <= 0:
        raise ValueError("maximum_halos must be a positive integer.")
    capacity = int(maximum_halos)
    resource = _admit_path(
        path,
        source,
        maximum_source_bytes=maximum_source_bytes,
        commercial_use=commercial_use,
        training_use=training_use,
        redistribution=redistribution,
        export=export,
    )
    data = _load_ascii(resource.data)
    count = data.shape[0]
    if count > capacity:
        raise MemoryError("PINOCCHIO halo count exceeds maximum_halos.")
    ids = data[:, 0].astype(np.int64)
    if np.any(ids < 0) or np.any(data[:, 0] != ids) or len(set(ids.tolist())) != count:
        raise ValueError("PINOCCHIO group IDs must be unique non-negative integers.")
    position_factor = _positive_scale(position_scale, "position_scale")
    mass_factor = _positive_scale(mass_scale, "mass_scale")
    velocity_factor = _positive_scale(velocity_scale, "velocity_scale")
    labels = tuple(
        str(value).strip()
        for value in (
            source_position_unit,
            source_mass_unit,
            source_velocity_unit,
        )
    )
    if any(not value for value in labels):
        raise ValueError("PINOCCHIO source units must be explicit.")
    required_frame = (
        "periodic_comoving_box"
        if product_kind == "catalog"
        else "observer_relative_comoving"
    )
    if coordinate_frame != required_frame:
        raise ValueError(
            f"PINOCCHIO {product_kind} requires coordinate_frame={required_frame!r}."
        )
    losses = [
        _approximation_loss(
            "producer-model",
            "PINOCCHIO groups are an LPT/fragmentation approximation, not resolved N-body haloes.",
        )
    ]
    active = np.zeros(capacity, dtype=np.bool_)
    active[:count] = True
    if product_kind == "catalog":
        if data.shape[1] not in (11, 12):
            raise ValueError(
                "PINOCCHIO catalog ASCII rows require 11 or 12 documented columns."
            )
        if mass_definition is None or scale_factor is None or box_size is None:
            raise ValueError(
                "PINOCCHIO HaloCatalog projection requires explicit mass definition, scale factor and box size."
            )
        if not isinstance(mass_definition, SphericalOverdensityMassDefinition):
            raise TypeError("mass_definition must be SphericalOverdensityMassDefinition.")
        scale = float(scale_factor)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale_factor must be finite and positive.")
        box = tuple(float(value) * position_factor for value in box_size)
        if len(box) != 3 or any(not np.isfinite(value) or value <= 0.0 for value in box):
            raise ValueError("box_size must contain three finite positive lengths.")
        padded_ids = np.full(capacity, -1, dtype=np.int64)
        masses = np.zeros(capacity, dtype=np.float64)
        initial = np.zeros((capacity, 3), dtype=np.float64)
        positions = np.zeros((capacity, 3), dtype=np.float64)
        velocities = np.zeros((capacity, 3), dtype=np.float64)
        padded_ids[:count] = ids
        masses[:count] = data[:, 1] * mass_factor
        initial[:count] = data[:, 2:5] * position_factor
        positions[:count] = data[:, 5:8] * position_factor
        velocities[:count] = data[:, 8:11] * velocity_factor
        particle_counts = np.zeros(capacity, dtype=np.int64)
        particle_known = np.zeros(capacity, dtype=np.bool_)
        if data.shape[1] == 12:
            counts = data[:, 11].astype(np.int64)
            if np.any(counts < 0) or np.any(counts != data[:, 11]):
                raise ValueError(
                    "PINOCCHIO particle counts must be non-negative integers."
                )
            particle_counts[:count] = counts
            particle_known[:count] = True
        else:
            losses.append(
                AdapterLoss(
                    "catalog.number_of_particles",
                    "import",
                    "dropped",
                    "LIGHT_OUTPUT omitted halo particle counts.",
                    changes_interpretation=False,
                )
            )
        artifact = _artifact(
            source,
            kind="approximate-halo-catalog",
            producer="PINOCCHIO",
            producer_version="5.x-documented-ascii",
        )
        catalog = HaloCatalog(
            padded_ids,
            positions,
            velocities,
            masses,
            active,
            mass_definition,
            scale,
            box,
            artifact,
        )
        sidecar = PinocchioCatalogSidecar(
            initial, particle_counts, particle_known, active
        )
        target_id = catalog.catalog_id
        light_cone = None
        preserved = (
            "group_id",
            "group_mass",
            "initial_position",
            "final_position",
            "velocity",
            "number_of_particles",
            "source_rights",
            "approximation_tag",
        )
        coordinate_mapping = (
            f"columns 3-5 -> initial periodic comoving position [{labels[0]}]",
            f"columns 6-8 -> final periodic comoving position [{labels[0]}]",
            f"columns 9-11 -> peculiar velocity [{labels[2]}]",
            f"column 2 -> halo mass [{labels[1]}]",
        )
    else:
        if data.shape[1] not in (6, 13):
            raise ValueError(
                "PINOCCHIO light-cone ASCII rows require 6 or 13 documented columns."
            )
        halo_ids = np.full(capacity, -1, dtype=np.int64)
        true_redshift = np.zeros(capacity)
        positions = np.zeros((capacity, 3))
        velocities = np.zeros((capacity, 3))
        masses = np.zeros(capacity)
        theta = np.zeros(capacity)
        phi = np.zeros(capacity)
        radial = np.zeros(capacity)
        observed = np.zeros(capacity)
        phase_space_known = np.zeros(capacity, dtype=np.bool_)
        halo_ids[:count] = ids
        true_redshift[:count] = data[:, 1]
        if data.shape[1] == 13:
            positions[:count] = data[:, 2:5] * position_factor
            velocities[:count] = data[:, 5:8] * velocity_factor
            masses[:count] = data[:, 8] * mass_factor
            theta[:count] = data[:, 9]
            phi[:count] = data[:, 10]
            radial[:count] = data[:, 11] * velocity_factor
            observed[:count] = data[:, 12]
            phase_space_known[:count] = True
        else:
            masses[:count] = data[:, 2] * mass_factor
            theta[:count] = data[:, 3]
            phi[:count] = data[:, 4]
            observed[:count] = data[:, 5]
            losses.append(
                AdapterLoss(
                    "light_cone.phase_space",
                    "import",
                    "dropped",
                    "LIGHT_OUTPUT omitted three-dimensional positions, velocities and line-of-sight velocity.",
                    changes_interpretation=False,
                )
            )
        light_cone = PinocchioLightConeProduct(
            halo_ids,
            true_redshift,
            positions,
            velocities,
            masses,
            theta,
            phi,
            radial,
            observed,
            phase_space_known,
            active,
        )
        catalog = None
        sidecar = None
        target_id = light_cone.product_id
        preserved = (
            "group_id",
            "true_redshift",
            "comoving_position",
            "velocity",
            "group_mass",
            "theta_degrees",
            "phi_degrees",
            "radial_velocity",
            "observed_redshift",
            "source_rights",
            "approximation_tag",
        )
        coordinate_mapping = (
            f"columns 3-5 -> observer-relative comoving position [{labels[0]}]",
            "columns 10-11 -> box-axis spherical angles [degree]",
        )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        f"PINOCCHIO documented ASCII {product_kind}",
        "HaloCatalog" if product_kind == "catalog" else "PinocchioLightConeProduct",
        source_id=source.manifest_id,
        target_id=target_id,
        coordinate_mapping=coordinate_mapping,
        preserved_fields=preserved,
        assumptions=(
            "Input units are the explicit producer units supplied by the caller; "
            "positive scales map them to the native projection.",
            "The approximation tag is retained and is not promoted to resolved N-body evidence.",
        ),
        losses=tuple(losses),
    )
    return PinocchioCatalogImport(
        catalog,
        light_cone,
        sidecar,
        source,
        report,
        coordinate_frame=coordinate_frame,
        source_position_unit=labels[0],
        source_mass_unit=labels[1],
        source_velocity_unit=labels[2],
        product_kind=product_kind,
    )


def _history_rows(data: bytes, /) -> tuple[int, int, np.ndarray]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("PINOCCHIO history must be valid UTF-8 text.") from error
    lines = tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(lines) < 2:
        raise ValueError("PINOCCHIO history must declare counts and at least one branch.")
    counts = lines[0].split()
    if len(counts) != 2:
        raise ValueError(
            "PINOCCHIO history count row must contain tree and branch counts."
        )
    declared_trees, declared_branches = (int(value) for value in counts)
    rows = np.asarray(
        [[float(value) for value in line.split()] for line in lines[1:]], dtype=np.float64
    )
    if rows.ndim != 2 or rows.shape[1] != 9 or np.any(~np.isfinite(rows)):
        raise ValueError(
            "PINOCCHIO history branches require nine finite documented columns."
        )
    if rows.shape[0] != declared_branches or declared_trees <= 0:
        raise ValueError("PINOCCHIO declared history counts disagree with branch rows.")
    return declared_trees, declared_branches, rows


def read_pinocchio_lineage(
    path: str | Path,
    source: ReferenceArtifactManifest,
    /,
    *,
    maximum_branches: int,
    maximum_source_bytes: int = 2_000_000_000,
    commercial_use: bool = False,
    training_use: bool = False,
    redistribution: bool = False,
    export: bool = False,
) -> PinocchioLineageImport:
    """Read documented PINOCCHIO merger histories without inventing descendants."""

    if isinstance(maximum_branches, bool) or int(maximum_branches) <= 0:
        raise ValueError("maximum_branches must be a positive integer.")
    capacity = int(maximum_branches)
    resource = _admit_path(
        path,
        source,
        maximum_source_bytes=maximum_source_bytes,
        commercial_use=commercial_use,
        training_use=training_use,
        redistribution=redistribution,
        export=export,
    )
    declared_trees, declared_branches, rows = _history_rows(resource.data)
    if declared_branches > capacity:
        raise MemoryError("PINOCCHIO history exceeds maximum_branches.")
    ids = rows[:, 0].astype(np.int64)
    integer_columns = rows[:, :4].astype(np.int64)
    if (
        np.any(rows[:, :4] != integer_columns)
        or np.any(ids < 0)
        or len(set(ids.tolist())) != ids.size
    ):
        raise ValueError(
            "PINOCCHIO history identifiers/indices must be integral with unique group IDs."
        )
    indices = integer_columns[:, 1]
    linking = integer_columns[:, 2]
    merged = integer_columns[:, 3]
    tree_numbers = np.zeros(declared_branches, dtype=np.int32)
    tree = -1
    tree_maps: list[dict[int, int]] = []
    for row, index in enumerate(indices):
        if row == 0 or index == 1:
            tree += 1
            tree_maps.append({})
        if int(index) in tree_maps[tree]:
            raise ValueError("PINOCCHIO indices within each tree must be unique.")
        tree_maps[tree][int(index)] = int(ids[row])
        tree_numbers[row] = tree
    if tree + 1 != declared_trees:
        raise ValueError("PINOCCHIO branch ordering disagrees with declared tree count.")
    sink_ids = np.full(declared_branches, -1, dtype=np.int64)
    for row, target_index in enumerate(merged):
        if target_index >= 0:
            mapping = tree_maps[int(tree_numbers[row])]
            if int(target_index) not in mapping:
                raise ValueError("PINOCCHIO merged-with index is absent from its tree.")
            sink_ids[row] = mapping[int(target_index)]
    active = np.zeros(capacity, dtype=np.bool_)
    active[:declared_branches] = True

    def pad(values: np.ndarray, fill: float | int = 0) -> np.ndarray:
        result = np.full(capacity, fill, dtype=values.dtype)
        result[:declared_branches] = values
        return result

    group_ids = pad(ids, -1)
    history = PinocchioMergerHistory(
        group_ids,
        pad(tree_numbers, -1),
        pad(indices, -1),
        pad(linking, -1),
        pad(merged, -1),
        pad(sink_ids, -1),
        np.full(capacity, -1, dtype=np.int64),
        pad(rows[:, 4]),
        pad(rows[:, 5]),
        pad(rows[:, 6]),
        pad(rows[:, 7]),
        pad(rows[:, 8]),
        active,
    )
    losses = (
        _approximation_loss(
            "producer-model",
            "PINOCCHIO merger histories arise from approximate fragmentation rather than resolved particle evolution.",
        ),
        AdapterLoss(
            "descendant_group_ids",
            "import",
            "unsupported",
            "Merged-with is a physical merger sink index; no particle-descendant relation is provided or synthesized.",
            changes_interpretation=False,
        ),
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "PINOCCHIO documented ASCII merger histories",
        "PinocchioMergerHistory",
        source_id=source.manifest_id,
        target_id=history.history_id,
        coordinate_mapping=("tree-local merged-with index -> stable sink group ID",),
        preserved_fields=(
            "group_id",
            "index_within_tree",
            "linking_list",
            "merged_with",
            "mass_at_merger_particles",
            "main_mass_at_merger_particles",
            "merger_redshift",
            "peak_collapse_redshift",
            "minimum_mass_redshift",
            "source_rights",
            "approximation_tag",
        ),
        assumptions=(
            "Trees are consecutive and each tree-local branch index is unique as documented.",
            "Sink and descendant semantics remain separate; missing descendants stay missing.",
        ),
        losses=losses,
    )
    return PinocchioLineageImport(
        history,
        source,
        report,
        declared_trees,
        declared_branches,
    )


__all__ = [
    "PinocchioCatalogImport",
    "PinocchioCatalogSidecar",
    "PinocchioLightConeProduct",
    "PinocchioLineageImport",
    "PinocchioMergerHistory",
    "read_pinocchio_catalog",
    "read_pinocchio_lineage",
]
