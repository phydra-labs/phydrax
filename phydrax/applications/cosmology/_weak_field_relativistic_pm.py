#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Poisson-gauge weak-field relativistic particle-mesh dynamics."""

from __future__ import annotations

from math import isfinite, pi, prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from ...discretization.particle._relativistic_stress_transfer import (
    RelativisticParticleState,
    RelativisticStressDepositPlan,
    RelativisticStressDepositResult,
    RelativisticStressSourceIntegrals,
)
from ...discretization.spectral._incompressible import PeriodicLerayProjector
from ...discretization.spectral._space import TensorSpectralDiscretization
from ...linalg import HermitianSpectrum
from ...metrix import StressEnergyProjection


WeakFieldDifferentiation: TypeAlias = Literal["piecewise-fixed-route", "none"]


def _norm(value: Array, weights: Array, spatial_rank: int, /) -> Array:
    squared = jnp.abs(value) ** 2
    if spatial_rank:
        squared = jnp.sum(
            squared,
            axis=tuple(range(squared.ndim - spatial_rank, squared.ndim)),
        )
    return jnp.sqrt(jnp.maximum(ein.contract("...,...->", squared, weights), 0.0))


def _select(accepted: Array, candidate: Array, original: Array, /) -> Array:
    condition = accepted
    while condition.ndim < candidate.ndim:
        condition = condition[..., None]
    return jnp.where(condition, candidate, original)


class WeakFieldRelativisticPMPolicy(StrictModule, NonTrainableState):
    """Explicit validity, accuracy, step, and resource gates."""

    maximum_scalar_metric_fraction: float = eqx.field(static=True)
    maximum_vector_metric_fraction: float = eqx.field(static=True)
    maximum_tensor_metric_norm: float = eqx.field(static=True)
    maximum_cell_crossing: float = eqx.field(static=True)
    constraint_relative_tolerance: float = eqx.field(static=True)
    gauge_absolute_tolerance: float = eqx.field(static=True)
    force_relative_tolerance: float = eqx.field(static=True)
    conservation_relative_tolerance: float = eqx.field(static=True)
    omitted_channel_tolerance: float = eqx.field(static=True)
    maximum_grid_points: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    differentiation: WeakFieldDifferentiation = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_scalar_metric_fraction: float = 1.0e-2,
        maximum_vector_metric_fraction: float = 1.0e-2,
        maximum_tensor_metric_norm: float = 1.0e-2,
        maximum_cell_crossing: float = 0.5,
        constraint_relative_tolerance: float = 1.0e-8,
        gauge_absolute_tolerance: float = 1.0e-8,
        force_relative_tolerance: float = 1.0e-7,
        conservation_relative_tolerance: float = 1.0e-7,
        omitted_channel_tolerance: float = 0.0,
        maximum_grid_points: int = 16_777_216,
        maximum_workspace_bytes: int = 8 * 1024**3,
        differentiation: WeakFieldDifferentiation = "piecewise-fixed-route",
    ):
        values = (
            float(maximum_scalar_metric_fraction),
            float(maximum_vector_metric_fraction),
            float(maximum_tensor_metric_norm),
            float(maximum_cell_crossing),
            float(constraint_relative_tolerance),
            float(gauge_absolute_tolerance),
            float(force_relative_tolerance),
            float(conservation_relative_tolerance),
            float(omitted_channel_tolerance),
        )
        if (
            any(not isfinite(value) or value < 0.0 for value in values)
            or values[0] <= 0.0
            or values[1] <= 0.0
            or values[2] <= 0.0
            or values[3] <= 0.0
        ):
            raise ValueError(
                "Weak-field policy tolerances must be finite and admissible."
            )
        points = int(maximum_grid_points)
        workspace = int(maximum_workspace_bytes)
        if points <= 0 or workspace <= 0:
            raise ValueError("Weak-field resource limits must be positive integers.")
        if differentiation not in ("piecewise-fixed-route", "none"):
            raise ValueError("Unknown weak-field differentiation contract.")
        (
            self.maximum_scalar_metric_fraction,
            self.maximum_vector_metric_fraction,
            self.maximum_tensor_metric_norm,
            self.maximum_cell_crossing,
            self.constraint_relative_tolerance,
            self.gauge_absolute_tolerance,
            self.force_relative_tolerance,
            self.conservation_relative_tolerance,
            self.omitted_channel_tolerance,
        ) = values
        self.maximum_grid_points = points
        self.maximum_workspace_bytes = workspace
        self.differentiation = differentiation
        self.policy_id = canonical_fingerprint(
            {
                "kind": "weak-field-relativistic-pm-policy",
                "values": values,
                "maximum_grid_points": points,
                "maximum_workspace_bytes": workspace,
                "differentiation": differentiation,
            }
        )


class PoissonGaugeMetricState(StrictModule):
    """Scalar, transverse-vector, and TT metric sectors on one source stage.

    ``phi`` and ``psi`` use the declared potential unit, ``shift_vector`` uses
    velocity units, and ``tensor_metric`` is dimensionless. The line element is
    represented to first weak-field order by ``alpha=1+phi/c^2``,
    ``beta^i=shift_vector^i/c``, and
    ``gamma_ij=a^2[(1-2 psi/c^2) delta_ij+h_ij]``.
    """

    phi: Array
    psi: Array
    shift_vector: Array
    tensor_metric: Array
    lapse: Array
    beta_contravariant: Array
    spatial_metric: Array
    scale_factor: Array
    time: Array
    snapshot_token: Array
    minimum_spatial_eigenvalue: Array
    scalar_weak: Array
    vector_weak: Array
    tensor_weak: Array
    metric_positive: Array
    finite: Array
    scalar_potential_unit_id: str = eqx.field(static=True)
    vector_potential_unit_id: str = eqx.field(static=True)
    tensor_metric_unit_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)


class WeakFieldStressResult(StrictModule):
    """Stress source, all Poisson-gauge sectors, forces, and solve evidence."""

    stress: StressEnergyProjection
    stress_evidence: RelativisticStressDepositResult
    metric: PoissonGaugeMetricState
    acceleration: Array
    particle_force: Array
    coordinate_velocity: Array
    source_integrals: RelativisticStressSourceIntegrals
    scalar_residual: Array
    vector_residual: Array
    tensor_residual: Array
    gauge_defect: Array
    constraint_defect: Array
    omitted_channel_bound: Array
    spectral_support_defect: Array
    background_frame_defect: Array
    net_force: Array
    force_relative_defect: Array
    support_complete: Array
    finite: Array
    successful: Array
    source_id: str = eqx.field(static=True)


class WeakFieldRelativisticPMStepDiagnostics(StrictModule):
    attempted: Array
    accepted: Array
    rolled_back: Array
    time_step: Array
    maximum_cell_crossing: Array
    source_integral_start: Array
    source_integral_end: Array
    source_relative_change: Array
    momentum_conservation_defect: Array
    energy_work_defect: Array
    force_relative_defect: Array
    scalar_residual: Array
    vector_residual: Array
    tensor_residual: Array
    gauge_defect: Array
    omitted_channel_bound: Array
    weak_field_valid: Array
    metric_positive: Array
    source_conserved: Array
    finite: Array
    status: Array
    status_meanings: tuple[str, ...] = eqx.field(
        static=True,
        default=(
            "accepted",
            "invalid-time-or-frame",
            "source-failure",
            "constraint-or-gauge-failure",
            "weak-field-or-metric-failure",
            "time-step-failure",
            "force-or-conservation-failure",
            "non-finite",
        ),
    )


class WeakFieldRelativisticPMStepResult(StrictModule):
    state: RelativisticParticleState
    metric: PoissonGaugeMetricState
    initial: WeakFieldStressResult
    endpoint: WeakFieldStressResult
    diagnostics: WeakFieldRelativisticPMStepDiagnostics
    successful: Array
    plan_id: str = eqx.field(static=True)


class WeakFieldRelativisticPMPlan(StrictModule, NonTrainableState):
    """Fixed-grid periodic weak-field relativistic PM transaction.

    The scalar sector solves both Bardeen potentials and their anisotropic-stress
    slip. Momentum is projected with the existing periodic Leray owner and the
    tensor source with the corresponding mode-wise TT projector. A scalar-only
    profile still computes the omitted vector/tensor candidate and is admitted
    only when its reported bound satisfies the policy.
    """

    stress_transfer: RelativisticStressDepositPlan
    spectral: TensorSpectralDiscretization
    projector: PeriodicLerayProjector
    units: RelativisticUnitContract
    policy: WeakFieldRelativisticPMPolicy
    laplacian_eigenvalues: Array
    inverse_laplacian: Array
    nonzero_modes: Array
    modal_projector: Array
    modal_scalar_projector: Array
    minimum_cell_width: float = eqx.field(static=True)
    characteristic_length: float = eqx.field(static=True)
    gravitational_constant: float = eqx.field(static=True)
    scalar_only: bool = eqx.field(static=True)
    grid_points: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stress_transfer: RelativisticStressDepositPlan,
        spectral: TensorSpectralDiscretization,
        units: RelativisticUnitContract,
        /,
        *,
        gravitational_constant: float | None = None,
        scalar_only: bool = False,
        policy: WeakFieldRelativisticPMPolicy | None = None,
    ):
        if not isinstance(stress_transfer, RelativisticStressDepositPlan):
            raise TypeError("stress_transfer must be RelativisticStressDepositPlan.")
        if not isinstance(spectral, TensorSpectralDiscretization):
            raise TypeError("spectral must be a TensorSpectralDiscretization.")
        if not isinstance(units, RelativisticUnitContract):
            raise TypeError("units must be a RelativisticUnitContract.")
        if stress_transfer.units.contract_id != units.contract_id:
            raise ValueError("Stress transfer and weak-field PM units differ.")
        if len(spectral.axes) != 3 or any(
            axis.family != "fourier" for axis in spectral.axes
        ):
            raise ValueError("Weak-field relativistic PM requires three Fourier axes.")
        if spectral.physical_shape != stress_transfer.transfer.target_shape:
            raise ValueError("Stress transfer and spectral physical shapes differ.")
        if spectral.grid.topology.topology_id != stress_transfer.transfer_topology_id:
            raise ValueError("Stress transfer and spectral transfer topologies differ.")
        if any(not axis.periodic for axis in spectral.axes):
            raise ValueError("Weak-field relativistic PM requires periodic axes.")
        selected_policy = WeakFieldRelativisticPMPolicy() if policy is None else policy
        if not isinstance(selected_policy, WeakFieldRelativisticPMPolicy):
            raise TypeError("policy must be WeakFieldRelativisticPMPolicy or None.")
        coupling = (
            float(units.scale.gravitational_constant)
            if gravitational_constant is None
            else float(gravitational_constant)
        )
        if not isfinite(coupling) or coupling <= 0.0:
            raise ValueError("gravitational_constant must be finite and positive.")
        if not isinstance(scalar_only, bool):
            raise TypeError("scalar_only must be a bool.")
        projector = PeriodicLerayProjector(spectral)
        eigenvalues = spectral.laplacian_eigenvalues()
        threshold = (
            64.0
            * jnp.finfo(eigenvalues.dtype).eps
            * jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
        )
        nonzero = jnp.abs(eigenvalues) > threshold
        safe = jnp.where(nonzero, eigenvalues, 1.0)
        inverse = jnp.where(nonzero, -1.0 / safe, 0.0)
        dimension = 3
        identity = jnp.eye(dimension, dtype=eigenvalues.dtype)
        wave_outer = ein.contract(
            "...i,...j->...ij",
            jnp.stack(projector.wavenumbers, axis=-1),
            jnp.stack(projector.wavenumbers, axis=-1),
            backend="jax",
        )
        modal_projector = (
            identity - wave_outer * projector.inverse_wavenumber_squared[..., None, None]
        )
        tensor_modes_admissible = nonzero & projector.admissibility_mask
        modal_projector = jnp.where(
            tensor_modes_admissible[..., None, None], modal_projector, 0.0
        )
        longitudinal_projector = (
            wave_outer * projector.inverse_wavenumber_squared[..., None, None]
        )
        modal_scalar_projector = jnp.where(
            tensor_modes_admissible[..., None, None],
            longitudinal_projector - identity / 3.0,
            0.0,
        )
        grid_points = int(prod(spectral.physical_shape))
        # Retained: k^2, inverse, mask, P_ij, Q_ij. Peak scalar/vector/tensor
        # source, solution, residual, gradient, and omission workspaces: 96 reals.
        real_itemsize = np.dtype(spectral.plan.precision.physical_dtype).itemsize
        retained_reals = 3 + 18
        workspace_reals = 96
        workspace_bytes = grid_points * real_itemsize * (retained_reals + workspace_reals)
        if grid_points > selected_policy.maximum_grid_points:
            raise ValueError("Weak-field grid exceeds maximum_grid_points.")
        if workspace_bytes > selected_policy.maximum_workspace_bytes:
            raise ValueError("Weak-field workspace exceeds maximum_workspace_bytes.")
        cell_widths = []
        domain_lengths = []
        for axis in spectral.axes:
            nodes = np.asarray(axis.nodes, dtype=float)
            if nodes.size < 2:
                raise ValueError("Each weak-field Fourier axis needs at least two nodes.")
            spacing = np.diff(np.sort(nodes))
            cell_widths.append(float(np.min(spacing)))
            domain_lengths.append(float(axis.length))
        minimum_cell_width = min(cell_widths)
        characteristic_length = max(domain_lengths)
        if (
            not isfinite(minimum_cell_width)
            or minimum_cell_width <= 0.0
            or not isfinite(characteristic_length)
            or characteristic_length <= 0.0
        ):
            raise ValueError("Weak-field Fourier geometry must be finite and positive.")
        self.stress_transfer = stress_transfer
        self.spectral = spectral
        self.projector = projector
        self.units = units
        self.policy = selected_policy
        self.laplacian_eigenvalues = eigenvalues
        self.inverse_laplacian = inverse
        self.nonzero_modes = nonzero
        self.modal_projector = modal_projector
        self.modal_scalar_projector = modal_scalar_projector
        self.minimum_cell_width = minimum_cell_width
        self.characteristic_length = characteristic_length
        self.gravitational_constant = coupling
        self.scalar_only = scalar_only
        self.grid_points = grid_points
        self.workspace_bytes = workspace_bytes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "weak-field-relativistic-particle-mesh",
                "stress_transfer": stress_transfer.plan_id,
                "spectral": spectral.prepared_id,
                "projector": projector.projector_id,
                "units": units.contract_id,
                "gravitational_constant": coupling,
                "scalar_only": scalar_only,
                "policy": selected_policy.policy_id,
                "grid_points": grid_points,
                "workspace_bytes": workspace_bytes,
                "sectors": ["phi", "psi", "transverse-B", "TT-h"],
            }
        )

    def _solve_modal(self, source_modes: Array, /) -> Array:
        payload_rank = source_modes.ndim - len(self.spectral.modal_shape)
        inverse = self.inverse_laplacian.reshape(
            self.inverse_laplacian.shape + (1,) * payload_rank
        )
        mask = self.nonzero_modes.reshape(self.nonzero_modes.shape + (1,) * payload_rank)
        return jnp.where(mask, source_modes * inverse, 0.0)

    def _residual(
        self, solution_modes: Array, source_modes: Array, payload_rank: int, /
    ) -> Array:
        eigenvalues = self.laplacian_eigenvalues.reshape(
            self.laplacian_eigenvalues.shape + (1,) * payload_rank
        )
        mask = self.nonzero_modes.reshape(self.nonzero_modes.shape + (1,) * payload_rank)
        residual = self.spectral.reconstruct(
            -eigenvalues * solution_modes - jnp.where(mask, source_modes, 0.0),
            real_output=True,
        )
        source = self.spectral.reconstruct(
            jnp.where(mask, source_modes, 0.0), real_output=True
        )
        weights = self.spectral.quadrature_weights.astype(residual.dtype)
        residual_norm = _norm(residual, weights, payload_rank)
        source_norm = _norm(source, weights, payload_rank)
        return jnp.where(source_norm > 0.0, residual_norm / source_norm, residual_norm)

    def _metric_from_stress(
        self,
        state: RelativisticParticleState,
        stress: RelativisticStressDepositResult,
        frame: LocalRelativisticFramePlan,
        /,
    ) -> tuple[PoissonGaugeMetricState, Array, Array, Array, Array, Array, Array]:
        projection = stress.projection
        dtype = projection.energy_density.dtype
        c = jnp.asarray(float(self.units.speed_of_light), dtype=dtype)
        gravity = jnp.asarray(self.gravitational_constant, dtype=dtype)
        scale_factor = state.scale_factor.astype(dtype)
        factor = 4.0 * jnp.asarray(pi, dtype=dtype) * gravity * scale_factor**2
        weights = self.spectral.quadrature_weights.astype(dtype)
        volume = jnp.sum(weights)
        mean_energy = (
            ein.contract("...,...->", projection.energy_density, weights, backend="jax")
            / volume
        )
        energy_source = factor * (projection.energy_density - mean_energy) / c**2
        energy_modes = self.spectral.project(energy_source)
        psi_modes = self._solve_modal(energy_modes)
        psi = self.spectral.reconstruct(psi_modes, real_output=True)

        spatial_basis = frame.tetrad.spatial_vectors[..., 1:]
        local_stress = ein.contract(
            "...ai,...ij,...bj->...ab",
            spatial_basis,
            projection.stress_covariant,
            spatial_basis,
            backend="jax",
        )
        stress_modes = self.spectral.project(local_stress)
        identity = jnp.eye(3, dtype=dtype)
        scalar_projector = self.modal_scalar_projector.astype(dtype)
        scalar_anisotropic_modes = ein.contract(
            "...ij,...ij->...", scalar_projector, stress_modes, backend="jax"
        )
        slip_source_modes = 3.0 * factor * scalar_anisotropic_modes / c**2
        slip_modes = self._solve_modal(slip_source_modes)
        phi_modes = psi_modes + slip_modes
        phi = self.spectral.reconstruct(phi_modes, real_output=True)

        local_momentum = ein.contract(
            "...ai,...i->...a",
            spatial_basis,
            projection.momentum_covector,
            backend="jax",
        )
        momentum_modes = self.spectral.project(local_momentum)
        transverse_momentum = self.projector.project(momentum_modes)
        forbidden = ~self.projector.admissibility_mask
        forbidden_momentum = jnp.max(
            jnp.abs(jnp.where(forbidden[..., None], momentum_modes, 0.0)),
            initial=0.0,
        )
        forbidden_stress = jnp.max(
            jnp.abs(jnp.where(forbidden[..., None, None], stress_modes, 0.0)),
            initial=0.0,
        )
        modal_source_scale = jnp.maximum(
            jnp.max(jnp.abs(momentum_modes), initial=0.0),
            jnp.max(jnp.abs(stress_modes), initial=0.0),
        )
        spectral_support_defect = jnp.where(
            modal_source_scale > 0.0,
            jnp.maximum(forbidden_momentum, forbidden_stress)
            / jnp.where(modal_source_scale > 0.0, modal_source_scale, 1.0),
            0.0,
        )
        vector_source_modes = -4.0 * factor * transverse_momentum / c**3
        candidate_shift_modes = self._solve_modal(vector_source_modes)
        candidate_shift = self.spectral.reconstruct(
            candidate_shift_modes, real_output=True
        )

        projected_stress = ein.contract(
            "...ik,...kl,...jl->...ij",
            self.modal_projector,
            stress_modes,
            self.modal_projector,
            backend="jax",
        )
        projected_trace = ein.contract(
            "...ij,...ij->...",
            self.modal_projector,
            stress_modes,
            backend="jax",
        )
        tt_stress = (
            projected_stress
            - 0.5 * self.modal_projector * projected_trace[..., None, None]
        )
        tt_stress = 0.5 * (tt_stress + jnp.swapaxes(tt_stress, -1, -2))
        tensor_source_modes = -4.0 * factor * tt_stress / c**4
        candidate_tensor_modes = self._solve_modal(tensor_source_modes)
        candidate_tensor = self.spectral.reconstruct(
            candidate_tensor_modes, real_output=True
        )
        vector_fraction = jnp.max(jnp.abs(candidate_shift), initial=0.0) / c
        tensor_fraction = jnp.max(jnp.abs(candidate_tensor), initial=0.0)
        omitted = jnp.maximum(vector_fraction, tensor_fraction)
        shift_modes = jnp.where(self.scalar_only, 0.0, candidate_shift_modes)
        tensor_modes = jnp.where(self.scalar_only, 0.0, candidate_tensor_modes)
        shift = jnp.where(self.scalar_only, 0.0, candidate_shift)
        tensor = jnp.where(self.scalar_only, 0.0, candidate_tensor)

        lapse = 1.0 + phi / c**2
        beta = shift / c
        spatial_metric = scale_factor**2 * (
            (1.0 - 2.0 * psi / c**2)[..., None, None] * identity + tensor
        )
        symmetric_metric = 0.5 * (spatial_metric + jnp.swapaxes(spatial_metric, -1, -2))
        metric_spectrum = HermitianSpectrum(
            symmetric_metric,
            tolerance=128.0 * float(jnp.finfo(dtype).eps),
        )
        minimum_eigenvalue = metric_spectrum.minimum_eigenvalue
        scalar_fraction = (
            jnp.maximum(
                jnp.max(jnp.abs(phi), initial=0.0),
                jnp.max(jnp.abs(psi), initial=0.0),
            )
            / c**2
        )
        scalar_weak = scalar_fraction <= self.policy.maximum_scalar_metric_fraction
        vector_weak = vector_fraction <= self.policy.maximum_vector_metric_fraction
        tensor_weak = tensor_fraction <= self.policy.maximum_tensor_metric_norm
        metric_positive = jnp.all(lapse > 0.0) & jnp.all(minimum_eigenvalue > 0.0)
        finite = (
            jnp.all(jnp.isfinite(phi))
            & jnp.all(jnp.isfinite(psi))
            & jnp.all(jnp.isfinite(shift))
            & jnp.all(jnp.isfinite(tensor))
            & jnp.all(jnp.isfinite(spatial_metric))
            & jnp.all(metric_spectrum.valid)
        )
        metric = PoissonGaugeMetricState(
            phi,
            psi,
            shift,
            tensor,
            lapse,
            beta,
            spatial_metric,
            state.scale_factor,
            state.time,
            projection.snapshot_token,
            minimum_eigenvalue,
            scalar_weak,
            vector_weak,
            tensor_weak,
            metric_positive,
            finite,
            self.units.scale.specific_energy_unit.unit_id,
            self.units.scale.dimensional_scale.velocity_unit.unit_id,
            "dimensionless",
            self.plan_id,
        )
        scalar_residual = jnp.maximum(
            self._residual(psi_modes, energy_modes, 0),
            self._residual(slip_modes, slip_source_modes, 0),
        )
        vector_residual = self._residual(candidate_shift_modes, vector_source_modes, 1)
        tensor_residual = self._residual(candidate_tensor_modes, tensor_source_modes, 2)
        return (
            metric,
            scalar_residual,
            vector_residual,
            tensor_residual,
            omitted,
            spectral_support_defect,
            mean_energy,
        )

    def _particle_force(
        self,
        state: RelativisticParticleState,
        stress: RelativisticStressDepositResult,
        metric: PoissonGaugeMetricState,
        /,
    ) -> tuple[Array, Array, Array, Array, Array]:
        c = jnp.asarray(float(self.units.speed_of_light), dtype=state.positions.dtype)
        grad_phi = self.spectral.gradient(metric.phi)
        grad_psi = self.spectral.gradient(metric.psi)
        grad_shift = self.spectral.gradient(metric.shift_vector)
        grad_tensor = self.spectral.gradient(metric.tensor_metric)
        combined = jnp.concatenate(
            (
                grad_phi[..., None, :],
                grad_psi[..., None, :],
                grad_shift.reshape(self.spectral.physical_shape + (3, 3)),
                grad_tensor.reshape(self.spectral.physical_shape + (9, 3)),
                metric.shift_vector[..., None, :],
            ),
            axis=-2,
        )
        gathered = self.stress_transfer.transfer.gather(stress.routes, combined)
        values = gathered.values
        particle_grad_phi = values[:, 0]
        particle_grad_psi = values[:, 1]
        particle_grad_shift = values[:, 2:5]
        particle_grad_tensor = values[:, 5:14].reshape((-1, 3, 3, 3))
        particle_shift = values[:, 14]
        momentum = state.local_momenta
        energy = stress.particle_energy
        safe_energy = jnp.where(state.active_mask & (energy > 0.0), energy, 1.0)
        momentum_squared = ein.contract("pi,pi->p", momentum, momentum, backend="jax")
        scalar_force = (
            -energy[:, None] * particle_grad_phi / c**2
            - momentum_squared[:, None] * particle_grad_psi / safe_energy[:, None]
        )
        vector_force = -ein.contract(
            "pji,pj->pi", particle_grad_shift, momentum, backend="jax"
        )
        tensor_force = (
            0.5
            * c**2
            * ein.contract(
                "pj,pk,pjki->pi",
                momentum,
                momentum,
                particle_grad_tensor,
                backend="jax",
            )
            / safe_energy[:, None]
        )
        force = scalar_force + vector_force + tensor_force
        force = jnp.where(state.active_mask[:, None], force, 0.0)

        inertial_mass = safe_energy / c**2
        acceleration = jnp.where(
            state.active_mask[:, None], force / inertial_mass[:, None], 0.0
        )
        velocity = (stress.local_velocity + particle_shift) / state.scale_factor
        velocity = jnp.where(state.active_mask[:, None], velocity, 0.0)
        net_force = jnp.sum(state.weights[:, None] * force, axis=0)
        force_scale = jnp.sum(state.weights * jnp.sqrt(jnp.sum(force**2, axis=-1)))
        relative = jnp.sqrt(jnp.sum(net_force**2)) / jnp.maximum(force_scale, 1.0)
        support = jnp.all(gathered.support | ~state.active_mask)
        relative = jnp.where(support, relative, jnp.inf)
        return acceleration, force, velocity, net_force, relative

    def _background_frame_defect(
        self,
        state: RelativisticParticleState,
        frame: LocalRelativisticFramePlan,
        /,
    ) -> Array:
        dtype = state.positions.dtype
        scale_factor = state.scale_factor.astype(dtype)
        identity = jnp.eye(3, dtype=dtype)
        spatial_target = scale_factor**2 * identity
        inverse_target = identity / scale_factor**2
        determinant_target = scale_factor**3
        spatial_scale = jnp.maximum(jnp.abs(spatial_target), 1.0)
        inverse_scale = jnp.maximum(jnp.abs(inverse_target), 1.0)
        determinant_scale = jnp.maximum(jnp.abs(determinant_target), 1.0)
        active = frame.geometry.active
        spatial_defect = jnp.max(
            jnp.where(
                active[..., None, None],
                jnp.abs(frame.geometry.spatial_metric - spatial_target) / spatial_scale,
                0.0,
            ),
            initial=0.0,
        )
        inverse_defect = jnp.max(
            jnp.where(
                active[..., None, None],
                jnp.abs(frame.geometry.inverse_spatial_metric - inverse_target)
                / inverse_scale,
                0.0,
            ),
            initial=0.0,
        )
        determinant_defect = jnp.max(
            jnp.where(
                active,
                jnp.abs(frame.geometry.sqrt_det_spatial_metric - determinant_target)
                / determinant_scale,
                0.0,
            ),
            initial=0.0,
        )
        lapse_defect = jnp.max(
            jnp.where(active, jnp.abs(frame.geometry.alpha - 1.0), 0.0),
            initial=0.0,
        )
        shift_defect = jnp.max(
            jnp.where(
                active[..., None],
                jnp.abs(frame.geometry.beta_contravariant),
                0.0,
            ),
            initial=0.0,
        )
        return jnp.maximum(
            jnp.maximum(spatial_defect, inverse_defect),
            jnp.maximum(determinant_defect, jnp.maximum(lapse_defect, shift_defect)),
        )

    def solve_stress(
        self,
        state: RelativisticParticleState,
        frame: LocalRelativisticFramePlan,
        /,
    ) -> WeakFieldStressResult:
        stress = self.stress_transfer.deposit(state, frame)
        (
            metric,
            scalar_residual,
            vector_residual,
            tensor_residual,
            omitted,
            spectral_support_defect,
            _,
        ) = self._metric_from_stress(state, stress, frame)
        acceleration, force, velocity, net_force, force_relative = self._particle_force(
            state, stress, metric
        )
        divergence_shift = self.spectral.divergence(metric.shift_vector)
        trace_tensor = jnp.trace(metric.tensor_metric, axis1=-2, axis2=-1)
        background_frame_defect = self._background_frame_defect(state, frame)
        divergence_tensor = jnp.stack(
            tuple(
                self.spectral.divergence(metric.tensor_metric[..., :, component])
                for component in range(3)
            ),
            axis=-1,
        )
        weights = self.spectral.quadrature_weights.astype(metric.phi.dtype)
        volume = jnp.sum(weights)
        c = jnp.asarray(float(self.units.speed_of_light), dtype=metric.phi.dtype)
        length = jnp.asarray(self.characteristic_length, dtype=metric.phi.dtype)
        scalar_gauge = (
            jnp.maximum(
                jnp.abs(ein.contract("...,...->", metric.phi, weights) / volume),
                jnp.abs(ein.contract("...,...->", metric.psi, weights) / volume),
            )
            / c**2
        )
        vector_gauge = length * jnp.max(jnp.abs(divergence_shift), initial=0.0) / c
        tensor_gauge = jnp.maximum(
            jnp.max(jnp.abs(trace_tensor), initial=0.0),
            length * jnp.max(jnp.abs(divergence_tensor), initial=0.0),
        )
        gauge_defect = jnp.maximum(scalar_gauge, jnp.maximum(vector_gauge, tensor_gauge))
        constraint = jnp.maximum(
            jnp.maximum(scalar_residual, vector_residual),
            jnp.maximum(
                tensor_residual,
                jnp.maximum(spectral_support_defect, background_frame_defect),
            ),
        )
        omitted_valid = jnp.asarray(not self.scalar_only) | (
            omitted <= self.policy.omitted_channel_tolerance
        )
        constraints_valid = (constraint <= self.policy.constraint_relative_tolerance) & (
            gauge_defect <= self.policy.gauge_absolute_tolerance
        )
        force_valid = force_relative <= self.policy.force_relative_tolerance
        weak_valid = metric.scalar_weak & metric.vector_weak & metric.tensor_weak
        finite = (
            stress.finite
            & metric.finite
            & jnp.all(jnp.isfinite(acceleration))
            & jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(velocity))
            & jnp.isfinite(constraint)
            & jnp.isfinite(gauge_defect)
            & jnp.isfinite(omitted)
            & jnp.isfinite(spectral_support_defect)
            & jnp.isfinite(background_frame_defect)
            & jnp.isfinite(force_relative)
        )
        successful = (
            stress.successful
            & constraints_valid
            & weak_valid
            & metric.metric_positive
            & omitted_valid
            & force_valid
            & finite
        )
        return WeakFieldStressResult(
            stress.projection,
            stress,
            metric,
            acceleration,
            force,
            velocity,
            stress.source_integrals,
            scalar_residual,
            vector_residual,
            tensor_residual,
            gauge_defect,
            constraint,
            omitted,
            spectral_support_defect,
            background_frame_defect,
            net_force,
            force_relative,
            stress.support_complete,
            finite,
            successful,
            self.plan_id,
        )

    def _wrap_positions(self, position: Array, /) -> Array:
        wrapped = position
        for axis, (lower, upper) in enumerate(self.stress_transfer.transfer.axis_bounds):
            length = jnp.asarray(upper - lower, dtype=position.dtype)
            wrapped = wrapped.at[:, axis].set(
                jnp.mod(wrapped[:, axis] - lower, length) + lower
            )
        return wrapped

    def _select_metric(
        self,
        accepted: Array,
        candidate: PoissonGaugeMetricState,
        original: PoissonGaugeMetricState,
        /,
    ) -> PoissonGaugeMetricState:
        return PoissonGaugeMetricState(
            _select(accepted, candidate.phi, original.phi),
            _select(accepted, candidate.psi, original.psi),
            _select(accepted, candidate.shift_vector, original.shift_vector),
            _select(accepted, candidate.tensor_metric, original.tensor_metric),
            _select(accepted, candidate.lapse, original.lapse),
            _select(accepted, candidate.beta_contravariant, original.beta_contravariant),
            _select(accepted, candidate.spatial_metric, original.spatial_metric),
            _select(accepted, candidate.scale_factor, original.scale_factor),
            _select(accepted, candidate.time, original.time),
            _select(accepted, candidate.snapshot_token, original.snapshot_token),
            _select(
                accepted,
                candidate.minimum_spatial_eigenvalue,
                original.minimum_spatial_eigenvalue,
            ),
            _select(accepted, candidate.scalar_weak, original.scalar_weak),
            _select(accepted, candidate.vector_weak, original.vector_weak),
            _select(accepted, candidate.tensor_weak, original.tensor_weak),
            _select(accepted, candidate.metric_positive, original.metric_positive),
            _select(accepted, candidate.finite, original.finite),
            original.scalar_potential_unit_id,
            original.vector_potential_unit_id,
            original.tensor_metric_unit_id,
            original.source_id,
        )

    def step(
        self,
        state: RelativisticParticleState,
        start_frame: LocalRelativisticFramePlan,
        endpoint_frame: LocalRelativisticFramePlan,
        /,
    ) -> WeakFieldRelativisticPMStepResult:
        """Execute source/metric/geodesic/endpoint-source as one rollback transaction."""
        if not isinstance(state, RelativisticParticleState):
            raise TypeError("state must be RelativisticParticleState.")
        self.stress_transfer._require_frame(start_frame)
        self.stress_transfer._require_frame(endpoint_frame)
        if (
            start_frame.geometry.geometry_lineage_id
            != endpoint_frame.geometry.geometry_lineage_id
            or start_frame.geometry.topology_id != endpoint_frame.geometry.topology_id
        ):
            raise ValueError("Weak-field PM stage frames must share geometry lineage.")
        initial = self.solve_stress(state, start_frame)
        start_time, start_scale, start_uniform = self.stress_transfer._frame_time(
            start_frame
        )
        end_time, end_scale, end_uniform = self.stress_transfer._frame_time(
            endpoint_frame
        )
        dt = end_time - start_time
        temporal_valid = (
            start_uniform
            & end_uniform
            & (state.time == start_time)
            & (state.scale_factor == start_scale)
            & jnp.isfinite(dt)
            & (dt > 0.0)
            & (end_scale >= start_scale)
        )
        half_covariant = state.covariant_momenta + 0.5 * dt * initial.particle_force
        half_local_start, half_start_valid = self.stress_transfer.local_from_covariant(
            state,
            start_frame,
            half_covariant,
        )
        rest_mass, _ = self.stress_transfer._species_rest_mass(state)
        c = jnp.asarray(float(self.units.speed_of_light), dtype=state.positions.dtype)
        rest_energy = self.units.mass_to_rest_energy(rest_mass)
        half_energy = jnp.sqrt(
            rest_energy**2
            + c**2
            * ein.contract("pi,pi->p", half_local_start, half_local_start, backend="jax")
        )
        half_velocity = (
            c**2
            * half_local_start
            / jnp.where(state.active_mask & (half_energy > 0.0), half_energy, 1.0)[
                :, None
            ]
        )
        start_shift = self.stress_transfer.transfer.gather(
            initial.stress_evidence.routes, initial.metric.shift_vector
        )
        coordinate_velocity = (half_velocity + start_shift.values) / jnp.maximum(
            start_scale, jnp.finfo(half_velocity.dtype).tiny
        )
        coordinate_velocity = jnp.where(
            state.active_mask[:, None], coordinate_velocity, 0.0
        )
        proposed_positions = self._wrap_positions(
            state.positions + dt * coordinate_velocity
        )
        half_local_end, half_end_valid = self.stress_transfer.local_from_covariant(
            state,
            endpoint_frame,
            half_covariant,
            positions=proposed_positions,
        )
        proposal = state.replace_dynamics(
            proposed_positions,
            half_local_end,
            half_covariant,
            end_time,
            end_scale,
            self.stress_transfer.frame_token(endpoint_frame),
        )
        endpoint_half = self.solve_stress(proposal, endpoint_frame)
        final_covariant = half_covariant + 0.5 * dt * endpoint_half.particle_force
        final_state = self.stress_transfer.replace_covariant_dynamics(
            state,
            endpoint_frame,
            proposed_positions,
            final_covariant,
            end_time,
            end_scale,
        )
        endpoint = self.solve_stress(final_state, endpoint_frame)

        displacement = jnp.abs(dt * coordinate_velocity) / self.minimum_cell_width
        maximum_crossing = jnp.max(
            jnp.where(state.active_mask[:, None], displacement, 0.0), initial=0.0
        )
        time_step_valid = maximum_crossing <= self.policy.maximum_cell_crossing
        weighted_initial_momentum = jnp.sum(
            state.weights[:, None] * state.covariant_momenta, axis=0
        )
        weighted_final_momentum = jnp.sum(
            state.weights[:, None] * final_covariant, axis=0
        )
        expected_impulse = (
            0.5
            * dt
            * jnp.sum(
                state.weights[:, None]
                * (initial.particle_force + endpoint_half.particle_force),
                axis=0,
            )
        )
        momentum_defect_vector = (
            weighted_final_momentum - weighted_initial_momentum - expected_impulse
        )
        momentum_scale = jnp.maximum(
            jnp.sqrt(jnp.sum(weighted_initial_momentum**2))
            + jnp.sqrt(jnp.sum(expected_impulse**2)),
            1.0,
        )
        momentum_defect = jnp.sqrt(jnp.sum(momentum_defect_vector**2)) / momentum_scale
        energy_start = initial.source_integrals.energy
        energy_end = endpoint.source_integrals.energy
        power_start = jnp.sum(
            state.weights
            * ein.contract(
                "pi,pi->p",
                initial.particle_force,
                initial.coordinate_velocity,
                backend="jax",
            )
        )
        power_end = jnp.sum(
            state.weights
            * ein.contract(
                "pi,pi->p",
                endpoint.particle_force,
                endpoint.coordinate_velocity,
                backend="jax",
            )
        )
        energy_work = energy_end - energy_start - 0.5 * dt * (power_start + power_end)
        energy_scale = jnp.maximum(
            jnp.maximum(jnp.abs(energy_start), jnp.abs(energy_end)), 1.0
        )
        energy_work_defect = jnp.abs(energy_work) / energy_scale
        source_relative_change = jnp.abs(energy_end - energy_start) / energy_scale
        source_conserved = (
            momentum_defect <= self.policy.conservation_relative_tolerance
        ) & (energy_work_defect <= self.policy.conservation_relative_tolerance)
        force_defect = jnp.maximum(
            initial.force_relative_defect, endpoint.force_relative_defect
        )
        force_valid = force_defect <= self.policy.force_relative_tolerance
        residual = jnp.maximum(
            jnp.maximum(initial.constraint_defect, endpoint.constraint_defect),
            endpoint_half.constraint_defect,
        )
        gauge = jnp.maximum(
            jnp.maximum(initial.gauge_defect, endpoint.gauge_defect),
            endpoint_half.gauge_defect,
        )
        constraints_valid = (residual <= self.policy.constraint_relative_tolerance) & (
            gauge <= self.policy.gauge_absolute_tolerance
        )
        omitted = jnp.maximum(
            jnp.maximum(initial.omitted_channel_bound, endpoint.omitted_channel_bound),
            endpoint_half.omitted_channel_bound,
        )
        weak_valid = (
            initial.metric.scalar_weak
            & initial.metric.vector_weak
            & initial.metric.tensor_weak
            & endpoint.metric.scalar_weak
            & endpoint.metric.vector_weak
            & endpoint.metric.tensor_weak
            & (
                jnp.asarray(not self.scalar_only)
                | (omitted <= self.policy.omitted_channel_tolerance)
            )
        )
        metric_positive = (
            initial.metric.metric_positive
            & endpoint_half.metric.metric_positive
            & endpoint.metric.metric_positive
        )
        source_valid = initial.successful & endpoint_half.successful & endpoint.successful
        finite = (
            initial.finite
            & endpoint_half.finite
            & endpoint.finite
            & jnp.all(jnp.isfinite(proposed_positions))
            & jnp.all(jnp.isfinite(final_covariant))
            & jnp.isfinite(maximum_crossing)
            & jnp.isfinite(momentum_defect)
            & jnp.isfinite(energy_work_defect)
        )
        accepted = (
            temporal_valid
            & half_start_valid
            & half_end_valid
            & source_valid
            & constraints_valid
            & weak_valid
            & metric_positive
            & time_step_valid
            & force_valid
            & source_conserved
            & finite
        )
        accepted_state = state.replace_dynamics(
            _select(accepted, final_state.positions, state.positions),
            _select(accepted, final_state.local_momenta, state.local_momenta),
            _select(accepted, final_state.covariant_momenta, state.covariant_momenta),
            _select(accepted, final_state.time, state.time),
            _select(accepted, final_state.scale_factor, state.scale_factor),
            _select(accepted, final_state.frame_token, state.frame_token),
        )
        accepted_metric = self._select_metric(accepted, endpoint.metric, initial.metric)
        status = jnp.where(
            accepted,
            0,
            jnp.where(
                ~temporal_valid,
                1,
                jnp.where(
                    ~source_valid,
                    2,
                    jnp.where(
                        ~constraints_valid,
                        3,
                        jnp.where(
                            ~(weak_valid & metric_positive),
                            4,
                            jnp.where(
                                ~time_step_valid,
                                5,
                                jnp.where(~(force_valid & source_conserved), 6, 7),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        diagnostics = WeakFieldRelativisticPMStepDiagnostics(
            jnp.asarray(True),
            accepted,
            ~accepted,
            dt,
            maximum_crossing,
            energy_start,
            energy_end,
            source_relative_change,
            momentum_defect,
            energy_work_defect,
            force_defect,
            jnp.maximum(initial.scalar_residual, endpoint.scalar_residual),
            jnp.maximum(initial.vector_residual, endpoint.vector_residual),
            jnp.maximum(initial.tensor_residual, endpoint.tensor_residual),
            gauge,
            omitted,
            weak_valid,
            metric_positive,
            source_conserved,
            finite,
            status,
        )
        return WeakFieldRelativisticPMStepResult(
            accepted_state,
            accepted_metric,
            initial,
            endpoint,
            diagnostics,
            accepted,
            self.plan_id,
        )


__all__ = [
    "PoissonGaugeMetricState",
    "WeakFieldDifferentiation",
    "WeakFieldRelativisticPMPlan",
    "WeakFieldRelativisticPMPolicy",
    "WeakFieldRelativisticPMStepDiagnostics",
    "WeakFieldRelativisticPMStepResult",
    "WeakFieldStressResult",
]
