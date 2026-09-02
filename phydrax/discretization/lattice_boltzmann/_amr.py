#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._collision import macroscopic_raw_moments, quadratic_equilibrium
from ._lattice import LatticeBoltzmannVelocitySet
from ._precision import LatticeBoltzmannPrecisionPolicy
from ._scaling import LatticeBoltzmannScaling


class LatticeBoltzmannAMRTransferEvidence(StrictModule):
    mass_defect: Array
    momentum_defect: Array
    minimum_population: Array
    successful: Array


class LatticeBoltzmannAMRTransferPlan(StrictModule, NonTrainableState):
    """Integer-ratio moment-conservative transfer for blockwise populations."""

    velocity_set: LatticeBoltzmannVelocitySet
    refinement_ratio: int = eqx.field(static=True)
    nonequilibrium_scale: float = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        velocity_set: LatticeBoltzmannVelocitySet,
        /,
        *,
        refinement_ratio: int = 2,
        nonequilibrium_scale: float = 1.0,
    ):
        if not isinstance(velocity_set, LatticeBoltzmannVelocitySet):
            raise TypeError("velocity_set must be LatticeBoltzmannVelocitySet.")
        ratio = int(refinement_ratio)
        scale = float(nonequilibrium_scale)
        if ratio < 2:
            raise ValueError("LBM AMR refinement_ratio must be at least two.")
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("nonequilibrium_scale must be finite and nonnegative.")
        self.velocity_set = velocity_set
        self.refinement_ratio = ratio
        self.nonequilibrium_scale = scale
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-amr-transfer",
                "lattice": velocity_set.lattice_id,
                "ratio": ratio,
                "nonequilibrium_scale": scale,
            }
        )

    def restrict(
        self, fine_populations: Array, /
    ) -> tuple[Array, LatticeBoltzmannAMRTransferEvidence]:
        fine = jnp.asarray(fine_populations)
        dimension = self.velocity_set.dimension
        if (
            fine.ndim != dimension + 1
            or fine.shape[-1] != self.velocity_set.population_count
        ):
            raise ValueError("Fine populations have incompatible LBM shape.")
        if any(size % self.refinement_ratio for size in fine.shape[:-1]):
            raise ValueError("Every fine spatial extent must be divisible by the ratio.")
        coarse_shape = tuple(size // self.refinement_ratio for size in fine.shape[:-1])
        reshape = []
        for size in coarse_shape:
            reshape.extend((size, self.refinement_ratio))
        reshape.append(fine.shape[-1])
        blocked = fine.reshape(tuple(reshape))
        reduction_axes = tuple(range(1, 2 * dimension, 2))
        coarse = jnp.mean(blocked, axis=reduction_axes)
        evidence = self._evidence(fine, coarse, fine_to_coarse=True)
        return coarse, evidence

    def prolong(
        self, coarse_populations: Array, /
    ) -> tuple[Array, LatticeBoltzmannAMRTransferEvidence]:
        coarse = jnp.asarray(coarse_populations)
        dimension = self.velocity_set.dimension
        if (
            coarse.ndim != dimension + 1
            or coarse.shape[-1] != self.velocity_set.population_count
        ):
            raise ValueError("Coarse populations have incompatible LBM shape.")
        fine = coarse
        for axis in range(dimension):
            fine = jnp.repeat(fine, self.refinement_ratio, axis=axis)
        evidence = self._evidence(fine, coarse, fine_to_coarse=True)
        return fine, evidence

    def _evidence(
        self, fine: Array, coarse: Array, /, *, fine_to_coarse: bool
    ) -> LatticeBoltzmannAMRTransferEvidence:
        ratio_volume = float(self.refinement_ratio**self.velocity_set.dimension)
        fine_mass = jnp.sum(fine) / ratio_volume
        coarse_mass = jnp.sum(coarse)
        velocities = jnp.asarray(self.velocity_set.velocities, dtype=fine.dtype)
        fine_momentum = ein.contract("...q,qd->d", fine, velocities) / ratio_volume
        coarse_momentum = ein.contract("...q,qd->d", coarse, velocities)
        mass_defect = jnp.abs(fine_mass - coarse_mass)
        momentum_defect = jnp.sqrt(jnp.sum((fine_momentum - coarse_momentum) ** 2))
        minimum = jnp.minimum(jnp.min(fine), jnp.min(coarse))
        successful = (
            jnp.all(jnp.isfinite(fine))
            & jnp.all(jnp.isfinite(coarse))
            & (minimum >= 0.0)
            & (
                mass_defect
                <= 128.0
                * jnp.finfo(fine.dtype).eps
                * jnp.maximum(jnp.abs(coarse_mass), 1.0)
            )
            & (
                momentum_defect
                <= 128.0
                * jnp.finfo(fine.dtype).eps
                * jnp.maximum(jnp.sqrt(jnp.sum(coarse_momentum**2)), 1.0)
            )
        )
        return LatticeBoltzmannAMRTransferEvidence(
            mass_defect, momentum_defect, minimum, successful
        )


class LatticeBoltzmannAMRInterfaceEvidence(StrictModule):
    mass_defect: Array
    momentum_defect: Array
    nonequilibrium_mass_defect: Array
    nonequilibrium_momentum_defect: Array
    minimum_population: Array
    nonequilibrium_scale: Array
    temporal_interpolation_fraction: Array
    successful: Array


class PreparedLatticeBoltzmannAMRTransfer(StrictModule, NonTrainableState):
    """Collision-aware equilibrium/nonequilibrium integer-ratio transfer."""

    transfer: LatticeBoltzmannAMRTransferPlan
    precision: LatticeBoltzmannPrecisionPolicy
    coarse_scaling: LatticeBoltzmannScaling
    fine_scaling: LatticeBoltzmannScaling
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: LatticeBoltzmannAMRTransferPlan,
        precision: LatticeBoltzmannPrecisionPolicy,
        coarse_scaling: LatticeBoltzmannScaling,
        fine_scaling: LatticeBoltzmannScaling,
        /,
    ):
        if not isinstance(transfer, LatticeBoltzmannAMRTransferPlan):
            raise TypeError("transfer must be LatticeBoltzmannAMRTransferPlan.")
        if transfer.nonequilibrium_scale <= 0.0:
            raise ValueError(
                "Collision-aware AMR requires positive nonequilibrium scaling."
            )
        if not isinstance(precision, LatticeBoltzmannPrecisionPolicy):
            raise TypeError("precision must be LatticeBoltzmannPrecisionPolicy.")
        if not isinstance(coarse_scaling, LatticeBoltzmannScaling) or not isinstance(
            fine_scaling, LatticeBoltzmannScaling
        ):
            raise TypeError(
                "coarse_scaling and fine_scaling must be LatticeBoltzmannScaling."
            )
        ratio = transfer.refinement_ratio
        if not np.isclose(
            float(fine_scaling.cell_size) * ratio,
            float(coarse_scaling.cell_size),
        ):
            raise ValueError(
                "LBM AMR spatial scalings do not match the refinement ratio."
            )
        time_ratio = float(coarse_scaling.time_step) / float(fine_scaling.time_step)
        if not np.isfinite(time_ratio) or time_ratio < ratio:
            raise ValueError(
                "LBM AMR fine scaling must take at least one acoustic substep per child."
            )
        reference_density = float(coarse_scaling.reference_density)
        sound_speed_squared = float(transfer.velocity_set.sound_speed_squared)
        if not np.isclose(float(fine_scaling.reference_density), reference_density):
            raise ValueError("LBM AMR levels must share one reference density.")
        if not np.isclose(
            float(coarse_scaling.sound_speed_squared), sound_speed_squared
        ) or not np.isclose(float(fine_scaling.sound_speed_squared), sound_speed_squared):
            raise ValueError("LBM AMR scaling sound speeds must match the velocity set.")
        self.transfer = transfer
        self.precision = precision
        self.coarse_scaling = coarse_scaling
        self.fine_scaling = fine_scaling
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-collision-aware-lattice-boltzmann-amr-transfer",
                "transfer": transfer.transfer_id,
                "precision": precision.policy_id,
                "coarse_scaling": coarse_scaling.scaling_id,
                "fine_scaling": fine_scaling.scaling_id,
            }
        )

    def _scale(
        self,
        coarse_relaxation_rate: Array,
        fine_relaxation_rate: Array,
        /,
    ) -> tuple[Array, Array]:
        coarse = jnp.asarray(coarse_relaxation_rate)
        fine = jnp.asarray(fine_relaxation_rate, dtype=coarse.dtype)
        if coarse.shape != () or fine.shape != ():
            raise ValueError("AMR relaxation rates must be scalar.")
        rate_valid = (
            jnp.isfinite(coarse)
            & (coarse > 0.0)
            & (coarse < 2.0)
            & jnp.isfinite(fine)
            & (fine > 0.0)
            & (fine < 2.0)
        )
        safe_coarse = jnp.where(rate_valid, coarse, jnp.asarray(1.0, coarse.dtype))
        safe_fine = jnp.where(rate_valid, fine, jnp.asarray(1.0, fine.dtype))
        coarse_tau_offset = 1.0 / safe_coarse - 0.5
        fine_tau_offset = 1.0 / safe_fine - 0.5
        scale = self.transfer.nonequilibrium_scale * fine_tau_offset / coarse_tau_offset
        scale_valid = rate_valid & jnp.isfinite(scale) & (scale > 0.0)
        return jnp.where(scale_valid, scale, jnp.asarray(1.0, scale.dtype)), scale_valid

    def _repeat(self, value: Array, /) -> Array:
        result = value
        for axis in range(self.transfer.velocity_set.dimension):
            result = jnp.repeat(result, self.transfer.refinement_ratio, axis=axis)
        return result

    def _block_average(self, value: Array, /) -> Array:
        dimension = self.transfer.velocity_set.dimension
        coarse_shape = tuple(
            size // self.transfer.refinement_ratio for size in value.shape[:-1]
        )
        reshape = []
        for size in coarse_shape:
            reshape.extend((size, self.transfer.refinement_ratio))
        reshape.append(value.shape[-1])
        reduction_axes = tuple(range(1, 2 * dimension, 2))
        return jnp.mean(value.reshape(tuple(reshape)), axis=reduction_axes)

    def _nonequilibrium_defects(self, nonequilibrium: Array, /) -> tuple[Array, Array]:
        velocities = jnp.asarray(
            self.transfer.velocity_set.velocities,
            dtype=nonequilibrium.dtype,
        )
        return (
            jnp.max(jnp.abs(jnp.sum(nonequilibrium, axis=-1))),
            jnp.max(jnp.abs(ein.contract("...q,qd->...d", nonequilibrium, velocities))),
        )

    def prolong(
        self,
        coarse_populations: Array,
        coarse_relaxation_rate: Array,
        fine_relaxation_rate: Array,
        /,
    ) -> tuple[Array, LatticeBoltzmannAMRInterfaceEvidence]:
        coarse = jnp.asarray(coarse_populations)
        density, momentum = macroscopic_raw_moments(
            coarse,
            self.transfer.velocity_set,
            self.precision,
        )
        velocity = (
            momentum / jnp.maximum(density, jnp.finfo(coarse.dtype).tiny)[..., None]
        )
        equilibrium = quadratic_equilibrium(
            density,
            velocity,
            self.transfer.velocity_set,
            self.precision,
        )
        nonequilibrium = coarse - equilibrium
        scale, scale_valid = self._scale(coarse_relaxation_rate, fine_relaxation_rate)
        fine_equilibrium = self._repeat(equilibrium)
        fine_nonequilibrium = scale * self._repeat(nonequilibrium)
        fine = self.precision.population(fine_equilibrium + fine_nonequilibrium)
        restricted, transfer_evidence = self.transfer.restrict(fine)
        restricted_density, restricted_momentum = macroscopic_raw_moments(
            restricted,
            self.transfer.velocity_set,
            self.precision,
        )
        mass_defect = jnp.max(jnp.abs(restricted_density - density))
        momentum_defect = jnp.max(jnp.abs(restricted_momentum - momentum))
        neq_mass, neq_momentum = self._nonequilibrium_defects(nonequilibrium)
        tolerance = (
            256.0
            * jnp.finfo(fine.dtype).eps
            * jnp.maximum(
                jnp.maximum(jnp.max(jnp.abs(density)), jnp.max(jnp.abs(momentum))),
                1.0,
            )
        )
        successful = (
            transfer_evidence.successful
            & scale_valid
            & jnp.all(jnp.isfinite(fine))
            & (jnp.min(fine) >= 0.0)
            & (mass_defect <= tolerance)
            & (momentum_defect <= tolerance)
            & (neq_mass <= tolerance)
            & (neq_momentum <= tolerance)
        )
        evidence = LatticeBoltzmannAMRInterfaceEvidence(
            mass_defect,
            momentum_defect,
            neq_mass,
            neq_momentum,
            jnp.min(fine),
            scale,
            jnp.asarray(0.0, dtype=fine.dtype),
            successful,
        )
        return fine, evidence

    def restrict(
        self,
        fine_populations: Array,
        coarse_relaxation_rate: Array,
        fine_relaxation_rate: Array,
        /,
    ) -> tuple[Array, LatticeBoltzmannAMRInterfaceEvidence]:
        fine = jnp.asarray(fine_populations)
        density, momentum = macroscopic_raw_moments(
            fine,
            self.transfer.velocity_set,
            self.precision,
        )
        velocity = momentum / jnp.maximum(density, jnp.finfo(fine.dtype).tiny)[..., None]
        equilibrium = quadratic_equilibrium(
            density,
            velocity,
            self.transfer.velocity_set,
            self.precision,
        )
        nonequilibrium = fine - equilibrium
        scale, scale_valid = self._scale(coarse_relaxation_rate, fine_relaxation_rate)
        coarse_density = self._block_average(density[..., None])[..., 0]
        coarse_momentum = self._block_average(momentum)
        coarse_velocity = (
            coarse_momentum
            / jnp.maximum(coarse_density, jnp.finfo(fine.dtype).tiny)[..., None]
        )
        coarse_equilibrium = quadratic_equilibrium(
            coarse_density,
            coarse_velocity,
            self.transfer.velocity_set,
            self.precision,
        )
        coarse_nonequilibrium = self._block_average(nonequilibrium) / scale
        coarse = self.precision.population(coarse_equilibrium + coarse_nonequilibrium)
        recovered_density, recovered_momentum = macroscopic_raw_moments(
            coarse,
            self.transfer.velocity_set,
            self.precision,
        )
        mass_defect = jnp.max(jnp.abs(recovered_density - coarse_density))
        momentum_defect = jnp.max(jnp.abs(recovered_momentum - coarse_momentum))
        neq_mass, neq_momentum = self._nonequilibrium_defects(coarse_nonequilibrium)
        tolerance = (
            256.0
            * jnp.finfo(coarse.dtype).eps
            * jnp.maximum(
                jnp.maximum(
                    jnp.max(jnp.abs(coarse_density)),
                    jnp.max(jnp.abs(coarse_momentum)),
                ),
                1.0,
            )
        )
        successful = (
            jnp.all(jnp.isfinite(coarse))
            & scale_valid
            & (jnp.min(coarse) >= 0.0)
            & (mass_defect <= tolerance)
            & (momentum_defect <= tolerance)
            & (neq_mass <= tolerance)
            & (neq_momentum <= tolerance)
        )
        evidence = LatticeBoltzmannAMRInterfaceEvidence(
            mass_defect,
            momentum_defect,
            neq_mass,
            neq_momentum,
            jnp.min(coarse),
            scale,
            jnp.asarray(1.0, dtype=coarse.dtype),
            successful,
        )
        return coarse, evidence


LatticeBoltzmannAMRScalingKind = Literal["acoustic", "diffusive", "declared"]


class LatticeBoltzmannAMRScalingPolicy(StrictModule, NonTrainableState):
    """Static temporal scaling for each integer-ratio AMR interface."""

    kind: LatticeBoltzmannAMRScalingKind = eqx.field(static=True)
    declared_substeps: tuple[int, ...] = eqx.field(static=True)
    viscosity_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: LatticeBoltzmannAMRScalingKind = "acoustic",
        /,
        *,
        declared_substeps: Sequence[int] = (),
        viscosity_tolerance: float = 1.0e-10,
    ):
        if kind not in ("acoustic", "diffusive", "declared"):
            raise ValueError("Unknown LBM AMR scaling kind.")
        steps = tuple(int(value) for value in declared_substeps)
        if any(value < 1 for value in steps):
            raise ValueError("Declared AMR substep counts must be positive.")
        if kind != "declared" and steps:
            raise ValueError("Only declared scaling accepts declared_substeps.")
        tolerance = float(viscosity_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("viscosity_tolerance must be finite and nonnegative.")
        self.kind = kind
        self.declared_substeps = steps
        self.viscosity_tolerance = tolerance
        self.policy_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-amr-scaling",
                "scaling": kind,
                "declared_substeps": steps,
                "viscosity_tolerance": tolerance,
            }
        )

    def substeps(self, ratio: int, interface_index: int, /) -> int:
        if self.kind == "acoustic":
            return int(ratio)
        if self.kind == "diffusive":
            return int(ratio) ** 2
        if interface_index >= len(self.declared_substeps):
            raise ValueError("Declared AMR scaling lacks an interface substep count.")
        return self.declared_substeps[interface_index]


class LatticeBoltzmannAMRTemporalTracePlan(StrictModule, NonTrainableState):
    """Declared polynomial temporal trace evaluated at every fine fraction."""

    nodes: Array
    coefficients: Array
    exactness_degree: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        nodes: Sequence[float] = (0.0, 1.0),
        coefficients: Sequence[Sequence[float]] = ((1.0, -1.0), (0.0, 1.0)),
        /,
        *,
        exactness_degree: int = 1,
    ):
        nodes_ = np.asarray(nodes, dtype=float)
        coefficients_ = np.asarray(coefficients, dtype=float)
        degree = int(exactness_degree)
        if (
            nodes_.ndim != 1
            or nodes_.size < 1
            or coefficients_.ndim != 2
            or coefficients_.shape[0] != nodes_.size
            or coefficients_.shape[1] != degree + 1
        ):
            raise ValueError(
                "AMR temporal coefficients must have shape (node_count, degree + 1)."
            )
        if (
            degree < 0
            or np.any(~np.isfinite(nodes_))
            or np.any(~np.isfinite(coefficients_))
            or np.any(np.diff(nodes_) <= 0.0)
            or nodes_[0] < 0.0
            or nodes_[-1] > 1.0
        ):
            raise ValueError("AMR temporal trace nodes/coefficients are invalid.")
        powers = np.arange(degree + 1)
        interpolation = coefficients_ @ (nodes_[:, None] ** powers[None, :]).T
        if not np.allclose(interpolation, np.eye(nodes_.size), rtol=1e-10, atol=1e-12):
            raise ValueError("AMR temporal coefficients must interpolate declared nodes.")
        self.nodes = jnp.asarray(nodes_)
        self.coefficients = jnp.asarray(coefficients_)
        self.exactness_degree = degree
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-amr-temporal-trace",
                "nodes": nodes_.tolist(),
                "coefficients": coefficients_.tolist(),
                "exactness_degree": degree,
            }
        )

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    def evaluate(self, values: Array, fraction: Array, /) -> Array:
        values_ = jnp.asarray(values)
        if values_.shape[0] != self.node_count:
            raise ValueError("AMR trace values must match the declared temporal nodes.")
        fraction_ = jnp.asarray(fraction, dtype=values_.real.dtype).reshape(())
        powers = fraction_ ** jnp.arange(self.exactness_degree + 1, dtype=fraction_.dtype)
        weights = self.coefficients.astype(fraction_.dtype) @ powers
        return ein.contract("n,n...->...", weights.astype(values_.dtype), values_)


class LatticeBoltzmannAMRState(StrictModule):
    """Fixed-capacity populations and activity for every prepared AMR level."""

    level_populations: tuple[Array, ...]
    active_masks: tuple[Array, ...]
    subcycle_phases: Array

    def __init__(
        self,
        level_populations: Sequence[Array],
        active_masks: Sequence[Array],
        subcycle_phases: Array | None = None,
    ):
        populations = tuple(jnp.asarray(value) for value in level_populations)
        masks = tuple(jnp.asarray(value, dtype=bool) for value in active_masks)
        if not populations or len(populations) != len(masks):
            raise ValueError(
                "AMR state requires matching nonempty level populations and masks."
            )
        for values, mask in zip(populations, masks, strict=True):
            if mask.shape != values.shape[:-1]:
                raise ValueError("AMR active masks must match level spatial shapes.")
        phases = jnp.asarray(
            (
                jnp.zeros((max(len(populations) - 1, 0),), dtype=jnp.int32)
                if subcycle_phases is None
                else subcycle_phases
            ),
            dtype=jnp.int32,
        )
        if phases.shape != (max(len(populations) - 1, 0),):
            raise ValueError("subcycle_phases must contain one entry per interface.")
        self.level_populations = populations
        self.active_masks = masks
        self.subcycle_phases = phases


class LatticeBoltzmannAMRDiagnostics(StrictModule):
    mass_defects: Array
    momentum_defects: Array
    nonequilibrium_scales: Array
    viscosity_defects: Array
    minimum_populations: Array
    temporal_fractions: Array
    interface_successful: Array
    finite: Array
    positive: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class LatticeBoltzmannAMRAdvanceResult(StrictModule):
    candidate_state: LatticeBoltzmannAMRState
    accepted_state: LatticeBoltzmannAMRState
    diagnostics: LatticeBoltzmannAMRDiagnostics
    successful: Array


class PreparedLatticeBoltzmannAMR(StrictModule, NonTrainableState):
    """Pure collision-aware fixed-hierarchy recursive LBM AMR execution."""

    transfers: tuple[PreparedLatticeBoltzmannAMRTransfer, ...]
    scaling: LatticeBoltzmannAMRScalingPolicy
    temporal_trace: LatticeBoltzmannAMRTemporalTracePlan
    substeps: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def _coverage(self, parent: Array, child: Array, ratio: int, /) -> Array:
        dimension = self.transfers[0].transfer.velocity_set.dimension
        expected = tuple(ratio * size for size in parent.shape)
        if child.shape != expected:
            raise ValueError("Adjacent AMR level shapes do not match refinement ratio.")
        reshape: list[int] = []
        for size in parent.shape:
            reshape.extend((size, ratio))
        blocked = child.reshape(tuple(reshape))
        axes = tuple(range(1, 2 * dimension, 2))
        any_child = jnp.any(blocked, axis=axes)
        all_child = jnp.all(blocked, axis=axes)
        child = eqx.error_if(
            child,
            jnp.any(any_child != all_child),
            "AMR activity must cover complete integer-ratio child blocks.",
        )
        parent = eqx.error_if(
            parent,
            jnp.any(all_child & ~parent),
            "Every refined parent cell must be active.",
        )
        return all_child

    def _viscosity_defect(
        self,
        interface_index: int,
        relaxation_rates: tuple[Array, ...],
        /,
    ) -> tuple[Array, Array]:
        transfer = self.transfers[interface_index]
        coarse_rate = relaxation_rates[interface_index]
        fine_rate = relaxation_rates[interface_index + 1]
        rate_valid = (
            jnp.isfinite(coarse_rate)
            & (coarse_rate > 0.0)
            & (coarse_rate < 2.0)
            & jnp.isfinite(fine_rate)
            & (fine_rate > 0.0)
            & (fine_rate < 2.0)
        )
        safe_coarse = jnp.where(
            rate_valid, coarse_rate, jnp.asarray(1.0, coarse_rate.dtype)
        )
        safe_fine = jnp.where(rate_valid, fine_rate, jnp.asarray(1.0, fine_rate.dtype))
        coarse_scaling = transfer.coarse_scaling
        fine_scaling = transfer.fine_scaling
        coarse_lattice_viscosity = coarse_scaling.sound_speed_squared.astype(
            coarse_rate.dtype
        ) * (1.0 / safe_coarse - 0.5)
        fine_lattice_viscosity = fine_scaling.sound_speed_squared.astype(
            fine_rate.dtype
        ) * (1.0 / safe_fine - 0.5)
        coarse_viscosity = coarse_scaling.physical_viscosity(coarse_lattice_viscosity)
        fine_viscosity = fine_scaling.physical_viscosity(fine_lattice_viscosity)
        defect = jnp.abs(fine_viscosity - coarse_viscosity)
        scale = jnp.maximum(jnp.abs(coarse_viscosity), jnp.abs(fine_viscosity))
        tolerance = (
            self.scaling.viscosity_tolerance + 256.0 * jnp.finfo(coarse_rate.dtype).eps
        ) * jnp.maximum(scale, jnp.finfo(coarse_rate.dtype).tiny)
        successful = (
            rate_valid
            & jnp.isfinite(coarse_viscosity)
            & jnp.isfinite(fine_viscosity)
            & (coarse_viscosity > 0.0)
            & (fine_viscosity > 0.0)
            & jnp.isfinite(defect)
            & (defect <= tolerance)
        )
        return defect, successful

    def advance(
        self,
        state: LatticeBoltzmannAMRState,
        level_steps: Sequence[Callable[[Array, Array | None, Any], Array]],
        relaxation_rates: Sequence[Array],
        /,
        *,
        args: Any = None,
        temporal_traces: Sequence[Array] | None = None,
    ) -> LatticeBoltzmannAMRAdvanceResult:
        if not isinstance(state, LatticeBoltzmannAMRState):
            raise TypeError("state must be LatticeBoltzmannAMRState.")
        steps = tuple(level_steps)
        level_count = len(state.level_populations)
        if len(steps) != level_count or len(self.transfers) != level_count - 1:
            raise ValueError(
                "Prepared AMR levels, transfers, and step callbacks mismatch."
            )
        rate_dtype = state.level_populations[0].real.dtype
        rates = tuple(jnp.asarray(value, dtype=rate_dtype) for value in relaxation_rates)
        if len(rates) != level_count or any(value.shape != () for value in rates):
            raise ValueError("AMR requires one scalar relaxation rate per level.")
        if temporal_traces is None:
            if self.temporal_trace.node_count != 2:
                raise ValueError(
                    "Higher-order temporal traces require explicit per-interface values."
                )
            traces: tuple[Array, ...] | None = None
        else:
            traces = tuple(jnp.asarray(value) for value in temporal_traces)
            if len(traces) != level_count - 1:
                raise ValueError("One temporal trace is required per AMR interface.")
        populations = list(state.level_populations)
        evidence: list[LatticeBoltzmannAMRInterfaceEvidence] = [
            LatticeBoltzmannAMRInterfaceEvidence(
                jnp.asarray(0.0, dtype=rate_dtype),
                jnp.asarray(0.0, dtype=rate_dtype),
                jnp.asarray(0.0, dtype=rate_dtype),
                jnp.asarray(0.0, dtype=rate_dtype),
                jnp.asarray(jnp.inf, dtype=rate_dtype),
                jnp.asarray(1.0, dtype=rate_dtype),
                jnp.asarray(0.0, dtype=rate_dtype),
                jnp.asarray(True),
            )
            for _ in self.transfers
        ]
        viscosity = [
            self._viscosity_defect(index, rates) for index in range(len(self.transfers))
        ]
        fractions = jnp.zeros((len(self.transfers),), dtype=rate_dtype)

        def advance_level(level: int, boundary: Array | None) -> Array:
            nonlocal fractions
            old = populations[level]
            active = state.active_masks[level]
            candidate = steps[level](old, boundary, args)
            if candidate.shape != old.shape:
                raise ValueError("LBM AMR level step changed a prepared array shape.")
            candidate = jnp.where(active[..., None], candidate, old)
            if level == level_count - 1:
                populations[level] = candidate
                return candidate
            transfer = self.transfers[level]
            coarse_rate = rates[level]
            fine_rate = rates[level + 1]
            ratio = transfer.transfer.refinement_ratio
            child_active = state.active_masks[level + 1]
            covered = self._coverage(active, child_active, ratio)
            child_old = populations[level + 1]
            prolonged_old, prolong_old_evidence = transfer.prolong(
                old, coarse_rate, fine_rate
            )
            populations[level + 1] = jnp.where(
                child_active[..., None], child_old, prolonged_old
            )
            trace_values = (
                jnp.stack((old, candidate))
                if traces is None
                else traces[level].astype(candidate.dtype)
            )
            if trace_values.shape[1:] != old.shape:
                raise ValueError(
                    "AMR temporal trace spatial shape does not match parent."
                )
            for fine_index in range(self.substeps[level]):
                fraction = jnp.asarray(
                    (fine_index + 1) / self.substeps[level],
                    dtype=fractions.dtype,
                )
                trace = self.temporal_trace.evaluate(trace_values, fraction)
                advance_level(level + 1, trace)
                fractions = fractions.at[level].set(fraction)
            child_candidate = populations[level + 1]
            prolonged_candidate, prolong_candidate_evidence = transfer.prolong(
                candidate, coarse_rate, fine_rate
            )
            child_for_restriction = jnp.where(
                child_active[..., None], child_candidate, prolonged_candidate
            )
            restricted, restriction_evidence = transfer.restrict(
                child_for_restriction, coarse_rate, fine_rate
            )
            populations[level] = jnp.where(covered[..., None], restricted, candidate)
            previous = evidence[level]
            viscosity_successful = viscosity[level][1]
            evidence[level] = LatticeBoltzmannAMRInterfaceEvidence(
                jnp.maximum(
                    previous.mass_defect,
                    jnp.maximum(
                        prolong_old_evidence.mass_defect,
                        jnp.maximum(
                            prolong_candidate_evidence.mass_defect,
                            restriction_evidence.mass_defect,
                        ),
                    ),
                ),
                jnp.maximum(
                    previous.momentum_defect,
                    jnp.maximum(
                        prolong_old_evidence.momentum_defect,
                        jnp.maximum(
                            prolong_candidate_evidence.momentum_defect,
                            restriction_evidence.momentum_defect,
                        ),
                    ),
                ),
                jnp.maximum(
                    previous.nonequilibrium_mass_defect,
                    jnp.maximum(
                        prolong_old_evidence.nonequilibrium_mass_defect,
                        jnp.maximum(
                            prolong_candidate_evidence.nonequilibrium_mass_defect,
                            restriction_evidence.nonequilibrium_mass_defect,
                        ),
                    ),
                ),
                jnp.maximum(
                    previous.nonequilibrium_momentum_defect,
                    jnp.maximum(
                        prolong_old_evidence.nonequilibrium_momentum_defect,
                        jnp.maximum(
                            prolong_candidate_evidence.nonequilibrium_momentum_defect,
                            restriction_evidence.nonequilibrium_momentum_defect,
                        ),
                    ),
                ),
                jnp.minimum(
                    previous.minimum_population,
                    jnp.minimum(
                        prolong_old_evidence.minimum_population,
                        jnp.minimum(
                            prolong_candidate_evidence.minimum_population,
                            restriction_evidence.minimum_population,
                        ),
                    ),
                ),
                restriction_evidence.nonequilibrium_scale,
                jnp.maximum(
                    previous.temporal_interpolation_fraction,
                    restriction_evidence.temporal_interpolation_fraction,
                ),
                previous.successful
                & prolong_old_evidence.successful
                & prolong_candidate_evidence.successful
                & restriction_evidence.successful
                & viscosity_successful,
            )
            return populations[level]

        advance_level(0, None)
        committed_populations = tuple(
            jnp.where(active[..., None], candidate, original)
            for candidate, original, active in zip(
                populations,
                state.level_populations,
                state.active_masks,
                strict=True,
            )
        )
        candidate_state = LatticeBoltzmannAMRState(
            committed_populations,
            state.active_masks,
            jnp.zeros_like(state.subcycle_phases),
        )
        mass_defects = jnp.stack(tuple(value.mass_defect for value in evidence))
        momentum_defects = jnp.stack(tuple(value.momentum_defect for value in evidence))
        nonequilibrium_scales = jnp.stack(
            tuple(value.nonequilibrium_scale for value in evidence)
        )
        viscosity_defects = jnp.stack(tuple(value[0] for value in viscosity))
        minimum_populations = jnp.stack(
            tuple(value.minimum_population for value in evidence)
        )
        interface_successful = jnp.stack(tuple(value.successful for value in evidence))
        finite = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value)) for value in committed_populations)
            )
        )
        positive = jnp.all(
            jnp.stack(tuple(jnp.all(value >= 0.0) for value in committed_populations))
        )
        successful = jnp.all(interface_successful) & finite & positive
        accepted = LatticeBoltzmannAMRState(
            tuple(
                jnp.where(successful, candidate, old)
                for candidate, old in zip(
                    candidate_state.level_populations,
                    state.level_populations,
                    strict=True,
                )
            ),
            state.active_masks,
            jnp.where(successful, candidate_state.subcycle_phases, state.subcycle_phases),
        )
        diagnostics = LatticeBoltzmannAMRDiagnostics(
            mass_defects,
            momentum_defects,
            nonequilibrium_scales,
            viscosity_defects,
            minimum_populations,
            fractions,
            interface_successful,
            finite,
            positive,
            successful,
            self.prepared_id,
        )
        return LatticeBoltzmannAMRAdvanceResult(
            candidate_state, accepted, diagnostics, successful
        )


class LatticeBoltzmannAMRPlan(StrictModule, NonTrainableState):
    """Finite integer-ratio hierarchy and temporal scaling contract."""

    transfers: tuple[LatticeBoltzmannAMRTransferPlan, ...]
    scaling: LatticeBoltzmannAMRScalingPolicy
    temporal_trace: LatticeBoltzmannAMRTemporalTracePlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfers: LatticeBoltzmannAMRTransferPlan
        | Sequence[LatticeBoltzmannAMRTransferPlan],
        /,
        *,
        scaling: LatticeBoltzmannAMRScalingPolicy | None = None,
        temporal_trace: LatticeBoltzmannAMRTemporalTracePlan | None = None,
    ):
        transfer_tuple = (
            (transfers,)
            if isinstance(transfers, LatticeBoltzmannAMRTransferPlan)
            else tuple(transfers)
        )
        if not transfer_tuple or any(
            not isinstance(value, LatticeBoltzmannAMRTransferPlan)
            for value in transfer_tuple
        ):
            raise TypeError("LBM AMR requires one transfer plan per interface.")
        lattice_id = transfer_tuple[0].velocity_set.lattice_id
        if any(value.velocity_set.lattice_id != lattice_id for value in transfer_tuple):
            raise ValueError("Every LBM AMR interface must use the same velocity set.")
        scaling_ = LatticeBoltzmannAMRScalingPolicy() if scaling is None else scaling
        trace_ = (
            LatticeBoltzmannAMRTemporalTracePlan()
            if temporal_trace is None
            else temporal_trace
        )
        if not isinstance(scaling_, LatticeBoltzmannAMRScalingPolicy):
            raise TypeError("scaling must be LatticeBoltzmannAMRScalingPolicy.")
        if not isinstance(trace_, LatticeBoltzmannAMRTemporalTracePlan):
            raise TypeError(
                "temporal_trace must be LatticeBoltzmannAMRTemporalTracePlan."
            )
        if scaling_.kind == "declared" and len(scaling_.declared_substeps) != len(
            transfer_tuple
        ):
            raise ValueError("Declared scaling requires one substep count per interface.")
        self.transfers = transfer_tuple
        self.scaling = scaling_
        self.temporal_trace = trace_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-amr-plan",
                "transfers": tuple(value.transfer_id for value in transfer_tuple),
                "scaling": scaling_.policy_id,
                "temporal_trace": trace_.plan_id,
            }
        )

    def prepare(
        self,
        precision: LatticeBoltzmannPrecisionPolicy,
        level_scalings: Sequence[LatticeBoltzmannScaling],
        /,
    ) -> PreparedLatticeBoltzmannAMR:
        if not isinstance(precision, LatticeBoltzmannPrecisionPolicy):
            raise TypeError("precision must be LatticeBoltzmannPrecisionPolicy.")
        scalings = tuple(level_scalings)
        if len(scalings) != len(self.transfers) + 1 or any(
            not isinstance(value, LatticeBoltzmannScaling) for value in scalings
        ):
            raise TypeError("LBM AMR requires one physical scaling per level.")
        substeps = tuple(
            self.scaling.substeps(transfer.refinement_ratio, index)
            for index, transfer in enumerate(self.transfers)
        )
        prepared_transfers = []
        for index, (transfer, substep_count) in enumerate(
            zip(self.transfers, substeps, strict=True)
        ):
            coarse_scaling = scalings[index]
            fine_scaling = scalings[index + 1]
            time_ratio = float(coarse_scaling.time_step) / float(fine_scaling.time_step)
            if not np.isclose(time_ratio, float(substep_count)):
                raise ValueError(
                    "LBM AMR physical time-step ratios must match the scaling policy."
                )
            prepared_transfers.append(
                PreparedLatticeBoltzmannAMRTransfer(
                    transfer,
                    precision,
                    coarse_scaling,
                    fine_scaling,
                )
            )
        prepared_transfer_tuple = tuple(prepared_transfers)
        return PreparedLatticeBoltzmannAMR(
            prepared_transfer_tuple,
            self.scaling,
            self.temporal_trace,
            substeps,
            prepared_id=canonical_fingerprint(
                {
                    "kind": "prepared-lattice-boltzmann-amr",
                    "plan": self.plan_id,
                    "transfers": tuple(
                        value.prepared_id for value in prepared_transfer_tuple
                    ),
                    "substeps": substeps,
                }
            ),
        )


__all__ = [
    "LatticeBoltzmannAMRAdvanceResult",
    "LatticeBoltzmannAMRDiagnostics",
    "LatticeBoltzmannAMRInterfaceEvidence",
    "LatticeBoltzmannAMRPlan",
    "LatticeBoltzmannAMRScalingKind",
    "LatticeBoltzmannAMRScalingPolicy",
    "LatticeBoltzmannAMRState",
    "LatticeBoltzmannAMRTemporalTracePlan",
    "LatticeBoltzmannAMRTransferEvidence",
    "LatticeBoltzmannAMRTransferPlan",
    "PreparedLatticeBoltzmannAMR",
    "PreparedLatticeBoltzmannAMRTransfer",
]
