#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._boundary import LatticeBoltzmannGeometrySnapshot
from ._discretization import LatticeBoltzmannDiscretization


class LatticeBoltzmannGeometryKind(StrEnum):
    """Execution taxonomy for Cartesian, blockwise, and mapped LBM geometry."""

    NATIVE = "native"
    BLOCKWISE = "blockwise"
    MAPPED = "mapped"


def _shifted_target_mask(
    fluid_mask: np.ndarray,
    velocity: tuple[int, ...],
    periodic: tuple[bool, ...],
    /,
) -> np.ndarray:
    axes = tuple(range(fluid_mask.ndim))
    shifted = np.roll(fluid_mask, velocity, axis=axes)
    for axis, (step, wraps) in enumerate(zip(velocity, periodic, strict=True)):
        if wraps or step == 0:
            continue
        shifted_axis = np.moveaxis(shifted, axis, 0)
        shifted_axis[0 if step > 0 else -1] = False
    return shifted


def _link_routes(
    discretization: LatticeBoltzmannDiscretization,
    fluid_mask: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.stack(
        tuple(
            _shifted_target_mask(fluid_mask, velocity, discretization.periodic)
            for velocity in discretization.velocity_set.velocity_tuples
        ),
        axis=-1,
    )
    source = fluid_mask[..., None]
    streaming = source & targets
    boundary = source & ~targets
    return streaming, boundary


def _default_boundary_normals(
    boundary_links: np.ndarray,
    velocities: np.ndarray,
    /,
) -> np.ndarray:
    lengths = np.sqrt(np.sum(velocities.astype(np.float64) ** 2, axis=-1))
    safe_lengths = np.where(lengths > 0.0, lengths, 1.0)
    directions = velocities.astype(np.float64) / safe_lengths[:, None]
    return np.where(boundary_links[..., None], directions, 0.0)


class LatticeBoltzmannLinkEpoch(StrictModule, NonTrainableState):
    """Fixed-shape streaming and wall-link data for one topology/numeric epoch."""

    streaming_links: Array
    boundary_links: Array
    boundary_fraction: Array
    boundary_normals: Array
    discretization_id: str = eqx.field(static=True)
    lattice_id: str = eqx.field(static=True)
    topology_epoch: int = eqx.field(static=True)
    numeric_epoch: int = eqx.field(static=True)
    link_shape: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    numeric_id: str = eqx.field(static=True)
    link_epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        geometry: LatticeBoltzmannGeometrySnapshot,
        /,
        *,
        topology_epoch: int = 0,
        numeric_epoch: int = 0,
        boundary_fraction: ArrayLike | None = None,
        boundary_normals: ArrayLike | None = None,
    ):
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("Link epochs require an LBM discretization.")
        if not isinstance(geometry, LatticeBoltzmannGeometrySnapshot):
            raise TypeError("Link epochs require an LBM geometry snapshot.")
        if geometry.discretization_id != discretization.prepared_id:
            raise ValueError("Link geometry belongs to a different discretization.")
        topology = int(topology_epoch)
        numeric = int(numeric_epoch)
        if topology < 0 or numeric < 0:
            raise ValueError("Topology and numeric epochs must be nonnegative.")

        fluid = np.asarray(geometry.fluid_mask, dtype=bool)
        streaming, boundary = _link_routes(discretization, fluid)
        link_shape = discretization.population_shape
        if streaming.shape != link_shape or boundary.shape != link_shape:
            raise RuntimeError("LBM link construction violated the population shape.")

        fraction = (
            np.where(boundary, 0.5, 0.0)
            if boundary_fraction is None
            else np.asarray(boundary_fraction, dtype=np.float64)
        )
        dimension = discretization.velocity_set.dimension
        normals = (
            _default_boundary_normals(
                boundary,
                np.asarray(discretization.velocity_set.velocities),
            )
            if boundary_normals is None
            else np.asarray(boundary_normals, dtype=np.float64)
        )
        if fraction.shape != link_shape:
            raise ValueError("boundary_fraction must have the trailing-Q link shape.")
        if normals.shape != (*link_shape, dimension):
            raise ValueError(
                "boundary_normals must have the trailing-Q-and-dimension link shape."
            )
        boundary_norm = np.sqrt(np.sum(normals**2, axis=-1))
        if (
            np.any(~np.isfinite(fraction))
            or np.any(~np.isfinite(normals))
            or np.any(boundary & ((fraction <= 0.0) | (fraction > 1.0)))
            or np.any(~boundary & (fraction != 0.0))
            or np.any(~boundary[..., None] & (normals != 0.0))
            or np.any(boundary & ~np.isclose(boundary_norm, 1.0, rtol=1e-10, atol=1e-12))
        ):
            raise ValueError(
                "Boundary-link fractions must lie in (0, 1], boundary normals must "
                "be unit length, and inactive numeric link data must be exactly zero."
            )

        topology_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-link-topology",
                "discretization": discretization.prepared_id,
                "topology_epoch": topology,
                "fluid_mask": array_tree_fingerprint(fluid),
                "streaming_links": array_tree_fingerprint(streaming),
                "boundary_links": array_tree_fingerprint(boundary),
            }
        )
        numeric_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-link-numerics",
                "topology": topology_id,
                "numeric_epoch": numeric,
                "boundary_fraction": array_tree_fingerprint(fraction),
                "boundary_normals": array_tree_fingerprint(normals),
            }
        )
        self.streaming_links = jnp.asarray(streaming, dtype=bool)
        self.boundary_links = jnp.asarray(boundary, dtype=bool)
        self.boundary_fraction = jnp.asarray(fraction)
        self.boundary_normals = jnp.asarray(normals)
        self.discretization_id = discretization.prepared_id
        self.lattice_id = discretization.velocity_set.lattice_id
        self.topology_epoch = topology
        self.numeric_epoch = numeric
        self.link_shape = link_shape
        self.topology_id = topology_id
        self.numeric_id = numeric_id
        self.link_epoch_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-link-epoch",
                "topology": topology_id,
                "numeric": numeric_id,
            }
        )


class LatticeBoltzmannGeometryEpoch(StrictModule, NonTrainableState):
    """Immutable cell classification and link realization for one LBM epoch."""

    discretization: LatticeBoltzmannDiscretization
    snapshot: LatticeBoltzmannGeometrySnapshot
    links: LatticeBoltzmannLinkEpoch
    geometry_kind: LatticeBoltzmannGeometryKind = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        snapshot: LatticeBoltzmannGeometrySnapshot,
        links: LatticeBoltzmannLinkEpoch,
        /,
        *,
        geometry_kind: LatticeBoltzmannGeometryKind = LatticeBoltzmannGeometryKind.NATIVE,
    ):
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("Geometry epochs require an LBM discretization.")
        if not isinstance(snapshot, LatticeBoltzmannGeometrySnapshot) or not isinstance(
            links, LatticeBoltzmannLinkEpoch
        ):
            raise TypeError("Geometry epochs require one snapshot and one link epoch.")
        if not isinstance(geometry_kind, LatticeBoltzmannGeometryKind):
            raise TypeError("geometry_kind must be a LatticeBoltzmannGeometryKind.")
        if (
            snapshot.discretization_id != discretization.prepared_id
            or links.discretization_id != discretization.prepared_id
        ):
            raise ValueError("Geometry epoch constituents use different discretizations.")
        expected_streaming, expected_boundary = _link_routes(
            discretization,
            np.asarray(snapshot.fluid_mask, dtype=bool),
        )
        if not np.array_equal(
            expected_streaming, np.asarray(links.streaming_links)
        ) or not np.array_equal(expected_boundary, np.asarray(links.boundary_links)):
            raise ValueError("Geometry snapshot and fixed link topology disagree.")
        self.discretization = discretization
        self.snapshot = snapshot
        self.links = links
        self.geometry_kind = geometry_kind
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-geometry-epoch",
                "geometry_kind": geometry_kind.value,
                "snapshot": snapshot.snapshot_id,
                "links": links.link_epoch_id,
            }
        )

    @classmethod
    def from_mask(
        cls,
        discretization: LatticeBoltzmannDiscretization,
        fluid_mask: ArrayLike,
        /,
        *,
        source_id: str | None = None,
        topology_epoch: int = 0,
        numeric_epoch: int = 0,
        boundary_fraction: ArrayLike | None = None,
        boundary_normals: ArrayLike | None = None,
        geometry_kind: LatticeBoltzmannGeometryKind = LatticeBoltzmannGeometryKind.NATIVE,
    ) -> LatticeBoltzmannGeometryEpoch:
        snapshot = LatticeBoltzmannGeometrySnapshot(
            discretization,
            fluid_mask,
            source_id=source_id,
        )
        links = LatticeBoltzmannLinkEpoch(
            discretization,
            snapshot,
            topology_epoch=topology_epoch,
            numeric_epoch=numeric_epoch,
            boundary_fraction=boundary_fraction,
            boundary_normals=boundary_normals,
        )
        return cls(discretization, snapshot, links, geometry_kind=geometry_kind)

    @property
    def fluid_mask(self) -> Array:
        return self.snapshot.fluid_mask

    @property
    def streaming_mask(self) -> Array:
        return self.links.streaming_links

    @property
    def boundary_mask(self) -> Array:
        return self.links.boundary_links

    @property
    def boundary_fraction(self) -> Array:
        return self.links.boundary_fraction

    @property
    def boundary_normals(self) -> Array:
        return self.links.boundary_normals

    @property
    def topology_epoch(self) -> int:
        return self.links.topology_epoch

    @property
    def numeric_epoch(self) -> int:
        return self.links.numeric_epoch

    def refresh_numeric(
        self,
        /,
        *,
        boundary_fraction: ArrayLike,
        boundary_normals: ArrayLike,
        numeric_epoch: int | None = None,
    ) -> LatticeBoltzmannGeometryRefresh:
        next_numeric = (
            self.numeric_epoch + 1 if numeric_epoch is None else int(numeric_epoch)
        )
        if next_numeric <= self.numeric_epoch:
            raise ValueError("A numeric refresh must strictly advance the numeric epoch.")
        refreshed_links = LatticeBoltzmannLinkEpoch(
            self.discretization,
            self.snapshot,
            topology_epoch=self.topology_epoch,
            numeric_epoch=next_numeric,
            boundary_fraction=boundary_fraction,
            boundary_normals=boundary_normals,
        )
        refreshed = LatticeBoltzmannGeometryEpoch(
            self.discretization,
            self.snapshot,
            refreshed_links,
            geometry_kind=self.geometry_kind,
        )
        fraction_change = jnp.max(
            jnp.abs(refreshed.boundary_fraction - self.boundary_fraction)
        )
        normal_change = jnp.max(
            jnp.sqrt(
                jnp.sum(
                    (refreshed.boundary_normals - self.boundary_normals) ** 2,
                    axis=-1,
                )
            )
        )
        changed = (refreshed.boundary_fraction != self.boundary_fraction) | jnp.any(
            refreshed.boundary_normals != self.boundary_normals,
            axis=-1,
        )
        evidence = LatticeBoltzmannGeometryRefreshEvidence(
            maximum_boundary_fraction_change=fraction_change,
            maximum_boundary_normal_change=normal_change,
            refreshed_link_count=jnp.sum(changed.astype(jnp.int32)),
            topology_unchanged=jnp.asarray(
                refreshed.links.topology_id == self.links.topology_id
            ),
            finite=jnp.isfinite(fraction_change) & jnp.isfinite(normal_change),
            accepted=jnp.asarray(True),
            source_epoch_id=self.epoch_id,
            target_epoch_id=refreshed.epoch_id,
        )
        return LatticeBoltzmannGeometryRefresh(refreshed, evidence)


class LatticeBoltzmannGeometryRefreshEvidence(StrictModule, NonTrainableState):
    """Evidence that a link refresh changed numerics without changing topology."""

    maximum_boundary_fraction_change: Array
    maximum_boundary_normal_change: Array
    refreshed_link_count: Array
    topology_unchanged: Array
    finite: Array
    accepted: Array
    source_epoch_id: str = eqx.field(static=True)
    target_epoch_id: str = eqx.field(static=True)


class LatticeBoltzmannGeometryRefresh(StrictModule, NonTrainableState):
    epoch: LatticeBoltzmannGeometryEpoch
    evidence: LatticeBoltzmannGeometryRefreshEvidence


class LatticeBoltzmannTopologyEventRequest(StrictModule, NonTrainableState):
    """A topology candidate that may only be committed after its named accepted step."""

    candidate_fluid_mask: Array
    requested_after_step: int = eqx.field(static=True)
    source_epoch_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)
    accepted_step_only: bool = eqx.field(static=True)

    def __init__(
        self,
        source: LatticeBoltzmannGeometryEpoch,
        candidate_fluid_mask: ArrayLike,
        requested_after_step: int,
        /,
        *,
        source_id: str,
    ):
        if not isinstance(source, LatticeBoltzmannGeometryEpoch):
            raise TypeError("Topology requests require a source geometry epoch.")
        mask = np.asarray(candidate_fluid_mask, dtype=bool)
        step = int(requested_after_step)
        identifier = str(source_id)
        if mask.shape != source.discretization.grid.shape:
            raise ValueError(
                "Candidate topology mask must match the fixed LBM grid shape."
            )
        if not np.any(mask):
            raise ValueError("A candidate LBM topology must contain a fluid cell.")
        if np.array_equal(mask, np.asarray(source.fluid_mask)):
            raise ValueError("Topology event requests must change cell classification.")
        if step < 0 or not identifier:
            raise ValueError(
                "Accepted-step index must be nonnegative and source_id non-empty."
            )
        self.candidate_fluid_mask = jnp.asarray(mask, dtype=bool)
        self.requested_after_step = step
        self.source_epoch_id = source.epoch_id
        self.source_id = identifier
        self.request_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-topology-event-request",
                "source_epoch": source.epoch_id,
                "requested_after_step": step,
                "source_id": identifier,
                "candidate_fluid_mask": array_tree_fingerprint(mask),
                "acceptance_policy": "accepted-step-only",
            }
        )
        self.accepted_step_only = True


class LatticeBoltzmannPopulationTransferEvidence(StrictModule, NonTrainableState):
    """Positivity and conserved raw-moment evidence for a geometry transfer."""

    source_mass: Array
    target_mass: Array
    mass_residual: Array
    source_momentum: Array
    target_momentum: Array
    momentum_residual: Array
    minimum_population: Array
    positivity_margin: Array
    positive: Array
    finite: Array
    passed: Array
    covered_count: int = eqx.field(static=True)
    uncovered_count: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)


class LatticeBoltzmannPopulationTransferResult(StrictModule):
    populations: Array
    evidence: LatticeBoltzmannPopulationTransferEvidence


class LatticeBoltzmannPopulationTransferPlan(StrictModule, NonTrainableState):
    """Moment-constrained covered/uncovered transfer between two fixed-shape epochs."""

    source: LatticeBoltzmannGeometryEpoch
    target: LatticeBoltzmannGeometryEpoch
    covered_mask: Array
    uncovered_mask: Array
    persistent_mask: Array
    covered_count: int = eqx.field(static=True)
    uncovered_count: int = eqx.field(static=True)
    positivity_floor: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: LatticeBoltzmannGeometryEpoch,
        target: LatticeBoltzmannGeometryEpoch,
        /,
        *,
        positivity_floor: float = 0.0,
        absolute_tolerance: float = 1.0e-12,
        relative_tolerance: float = 1.0e-12,
    ):
        if not isinstance(source, LatticeBoltzmannGeometryEpoch) or not isinstance(
            target, LatticeBoltzmannGeometryEpoch
        ):
            raise TypeError("Population transfer requires two geometry epochs.")
        if source.discretization.prepared_id != target.discretization.prepared_id:
            raise ValueError("Population transfer epochs must share one discretization.")
        if target.topology_epoch != source.topology_epoch + 1:
            raise ValueError("Topology transfer must advance exactly one topology epoch.")
        floor = float(positivity_floor)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if (
            not np.isfinite(floor)
            or floor < 0.0
            or not np.isfinite(absolute)
            or absolute <= 0.0
            or not np.isfinite(relative)
            or relative < 0.0
        ):
            raise ValueError("Population-transfer floor/tolerances are invalid.")
        source_mask = np.asarray(source.fluid_mask, dtype=bool)
        target_mask = np.asarray(target.fluid_mask, dtype=bool)
        covered = source_mask & ~target_mask
        uncovered = ~source_mask & target_mask
        persistent = source_mask & target_mask
        if not np.any(covered | uncovered):
            raise ValueError(
                "Population transfer requires a topology classification change."
            )
        self.source = source
        self.target = target
        self.covered_mask = jnp.asarray(covered)
        self.uncovered_mask = jnp.asarray(uncovered)
        self.persistent_mask = jnp.asarray(persistent)
        self.covered_count = int(np.count_nonzero(covered))
        self.uncovered_count = int(np.count_nonzero(uncovered))
        self.positivity_floor = floor
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-population-transfer",
                "source": source.epoch_id,
                "target": target.epoch_id,
                "covered": array_tree_fingerprint(covered),
                "uncovered": array_tree_fingerprint(uncovered),
                "positivity_floor": floor,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "moment_constraints": ["mass", "momentum"],
            }
        )

    def transfer(
        self,
        populations: ArrayLike,
        /,
        *,
        uncovered_populations: ArrayLike | None = None,
    ) -> LatticeBoltzmannPopulationTransferResult:
        source_populations = self.source.discretization.validate_populations(populations)
        dtype = source_populations.dtype
        velocities = jnp.asarray(
            self.source.discretization.velocity_set.velocities,
            dtype=dtype,
        )
        weights = jnp.asarray(
            self.source.discretization.velocity_set.weights,
            dtype=dtype,
        )
        sound_speed_squared = jnp.asarray(
            self.source.discretization.velocity_set.sound_speed_squared,
            dtype=dtype,
        )
        source_values = jnp.where(
            self.source.fluid_mask[..., None],
            source_populations,
            jnp.zeros((), dtype=dtype),
        )
        source_mass = jnp.sum(source_values)
        source_momentum = oe.contract("...q,qd->d", source_values, velocities)
        target_cell_count = self.target.snapshot.fluid_count
        mass_per_cell = source_mass / target_cell_count
        momentum_per_cell = source_momentum / target_cell_count
        default_uncovered = weights * (
            mass_per_cell
            + oe.contract("d,qd->q", momentum_per_cell, velocities) / sound_speed_squared
        )
        default_uncovered = jnp.broadcast_to(
            default_uncovered,
            self.source.discretization.population_shape,
        )
        if uncovered_populations is None:
            uncovered_values = default_uncovered
        else:
            uncovered_values = self.source.discretization.validate_populations(
                uncovered_populations
            ).astype(dtype)
        solid_values = jnp.broadcast_to(
            weights,
            self.source.discretization.population_shape,
        )
        base = jnp.where(
            self.persistent_mask[..., None],
            source_populations,
            jnp.where(
                self.uncovered_mask[..., None],
                uncovered_values,
                solid_values,
            ),
        )
        target_values = jnp.where(
            self.target.fluid_mask[..., None],
            base,
            jnp.zeros((), dtype=dtype),
        )
        base_mass = jnp.sum(target_values)
        base_momentum = oe.contract("...q,qd->d", target_values, velocities)
        mass_correction = (source_mass - base_mass) / target_cell_count
        momentum_correction = (source_momentum - base_momentum) / target_cell_count
        population_correction = weights * (
            mass_correction
            + oe.contract("d,qd->q", momentum_correction, velocities)
            / sound_speed_squared
        )
        corrected = jnp.where(
            self.target.fluid_mask[..., None],
            base + population_correction,
            solid_values,
        )
        active_corrected = jnp.where(
            self.target.fluid_mask[..., None],
            corrected,
            jnp.zeros((), dtype=dtype),
        )
        target_mass = jnp.sum(active_corrected)
        target_momentum = oe.contract("...q,qd->d", active_corrected, velocities)
        mass_residual = jnp.abs(target_mass - source_mass)
        momentum_residual = jnp.max(jnp.abs(target_momentum - source_momentum))
        minimum = jnp.min(
            jnp.where(
                self.target.fluid_mask[..., None],
                corrected,
                jnp.asarray(jnp.inf, dtype=dtype),
            )
        )
        floor = jnp.asarray(self.positivity_floor, dtype=dtype)
        positivity_margin = minimum - floor
        finite = (
            jnp.all(jnp.isfinite(corrected))
            & jnp.isfinite(source_mass)
            & jnp.all(jnp.isfinite(source_momentum))
        )
        scale = jnp.maximum(
            jnp.asarray(1.0, dtype=dtype),
            jnp.maximum(jnp.abs(source_mass), jnp.max(jnp.abs(source_momentum))),
        )
        tolerance = (
            jnp.asarray(self.absolute_tolerance, dtype=dtype)
            + jnp.asarray(self.relative_tolerance, dtype=dtype) * scale
        )
        positive = minimum >= floor
        passed = (
            finite
            & positive
            & (mass_residual <= tolerance)
            & (momentum_residual <= tolerance)
        )
        corrected = eqx.error_if(
            corrected,
            ~passed,
            "LBM topology transfer failed positivity or conserved moment constraints.",
        )
        evidence = LatticeBoltzmannPopulationTransferEvidence(
            source_mass=source_mass,
            target_mass=target_mass,
            mass_residual=mass_residual,
            source_momentum=source_momentum,
            target_momentum=target_momentum,
            momentum_residual=momentum_residual,
            minimum_population=minimum,
            positivity_margin=positivity_margin,
            positive=positive,
            finite=finite,
            passed=passed,
            covered_count=self.covered_count,
            uncovered_count=self.uncovered_count,
            tolerance=self.absolute_tolerance,
            transfer_id=self.transfer_id,
        )
        return LatticeBoltzmannPopulationTransferResult(corrected, evidence)


class LatticeBoltzmannGeometryTransitionResult(StrictModule):
    geometry: LatticeBoltzmannGeometryEpoch
    populations: Array
    transfer_evidence: LatticeBoltzmannPopulationTransferEvidence | None
    committed: Array


class LatticeBoltzmannGeometryTransaction(StrictModule, NonTrainableState):
    """Prepared candidate and transfer whose accepted source remains rollback-safe."""

    accepted: LatticeBoltzmannGeometryEpoch
    candidate: LatticeBoltzmannGeometryEpoch
    request: LatticeBoltzmannTopologyEventRequest
    transfer: LatticeBoltzmannPopulationTransferPlan
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        accepted: LatticeBoltzmannGeometryEpoch,
        candidate: LatticeBoltzmannGeometryEpoch,
        request: LatticeBoltzmannTopologyEventRequest,
        transfer: LatticeBoltzmannPopulationTransferPlan,
        /,
    ):
        if (
            not isinstance(accepted, LatticeBoltzmannGeometryEpoch)
            or not isinstance(candidate, LatticeBoltzmannGeometryEpoch)
            or not isinstance(request, LatticeBoltzmannTopologyEventRequest)
            or not isinstance(transfer, LatticeBoltzmannPopulationTransferPlan)
        ):
            raise TypeError(
                "Geometry transactions require epochs, request, and transfer."
            )
        if (
            request.source_epoch_id != accepted.epoch_id
            or transfer.source.epoch_id != accepted.epoch_id
            or transfer.target.epoch_id != candidate.epoch_id
        ):
            raise ValueError("Geometry transaction identities disagree.")
        self.accepted = accepted
        self.candidate = candidate
        self.request = request
        self.transfer = transfer
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-geometry-transaction",
                "accepted": accepted.epoch_id,
                "candidate": candidate.epoch_id,
                "request": request.request_id,
                "transfer": transfer.transfer_id,
            }
        )

    def commit(
        self,
        populations: ArrayLike,
        accepted_step: int,
        /,
        *,
        uncovered_populations: ArrayLike | None = None,
    ) -> LatticeBoltzmannGeometryTransitionResult:
        if int(accepted_step) != self.request.requested_after_step:
            raise ValueError(
                "Topology candidate may only commit after its requested accepted step."
            )
        result = self.transfer.transfer(
            populations,
            uncovered_populations=uncovered_populations,
        )
        return LatticeBoltzmannGeometryTransitionResult(
            self.candidate,
            result.populations,
            result.evidence,
            jnp.asarray(True),
        )

    def rollback(
        self,
        populations: ArrayLike,
        /,
    ) -> LatticeBoltzmannGeometryTransitionResult:
        values = self.accepted.discretization.validate_populations(populations)
        return LatticeBoltzmannGeometryTransitionResult(
            self.accepted,
            values,
            None,
            jnp.asarray(False),
        )


def prepare_lattice_boltzmann_topology_event(
    accepted: LatticeBoltzmannGeometryEpoch,
    request: LatticeBoltzmannTopologyEventRequest,
    /,
    *,
    boundary_fraction: ArrayLike | None = None,
    boundary_normals: ArrayLike | None = None,
    positivity_floor: float = 0.0,
    absolute_tolerance: float = 1.0e-12,
    relative_tolerance: float = 1.0e-12,
) -> LatticeBoltzmannGeometryTransaction:
    """Prepare, but do not commit, an accepted-step topology transaction."""

    if not isinstance(accepted, LatticeBoltzmannGeometryEpoch) or not isinstance(
        request, LatticeBoltzmannTopologyEventRequest
    ):
        raise TypeError("Topology preparation requires an accepted epoch and request.")
    if request.source_epoch_id != accepted.epoch_id:
        raise ValueError("Topology request was created from a different accepted epoch.")
    candidate = LatticeBoltzmannGeometryEpoch.from_mask(
        accepted.discretization,
        request.candidate_fluid_mask,
        source_id=request.source_id,
        topology_epoch=accepted.topology_epoch + 1,
        numeric_epoch=0,
        boundary_fraction=boundary_fraction,
        boundary_normals=boundary_normals,
        geometry_kind=accepted.geometry_kind,
    )
    transfer = LatticeBoltzmannPopulationTransferPlan(
        accepted,
        candidate,
        positivity_floor=positivity_floor,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    return LatticeBoltzmannGeometryTransaction(accepted, candidate, request, transfer)


__all__ = [
    "LatticeBoltzmannGeometryEpoch",
    "LatticeBoltzmannGeometryKind",
    "LatticeBoltzmannGeometryRefresh",
    "LatticeBoltzmannGeometryRefreshEvidence",
    "LatticeBoltzmannGeometryTransaction",
    "LatticeBoltzmannGeometryTransitionResult",
    "LatticeBoltzmannLinkEpoch",
    "LatticeBoltzmannPopulationTransferEvidence",
    "LatticeBoltzmannPopulationTransferPlan",
    "LatticeBoltzmannPopulationTransferResult",
    "LatticeBoltzmannTopologyEventRequest",
    "prepare_lattice_boltzmann_topology_event",
]
