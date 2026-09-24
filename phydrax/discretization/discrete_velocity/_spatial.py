#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite, prod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import fixed_field, NonTrainableState
from ..lattice_boltzmann._program import (
    KineticProgramManifest,
    smooth_compressible_spatial_dvm_manifest,
)
from ._energy_equilibrium import EnergyEquilibriumEvidence, PositiveEnergyEquilibriumPlan
from ._quadrature import CertifiedDiscreteVelocityQuadrature
from ._smooth_compressible import (
    SmoothCompressibleCollisionEvidence,
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
    SmoothCompressibleLearnedEquilibriumEvidence,
    SmoothCompressibleRealizabilityEvidence,
)
from ._spatial_boundary import (
    AbstractSmoothCompressibleD2VBoundaryPlan,
    SmoothCompressibleD2VBoundaryResult,
)
from ._spatial_forcing import (
    SmoothCompressibleD2VBodyForcingPlan,
    SmoothCompressibleD2VForcingResult,
    ZeroSmoothCompressibleD2VForcingPlan,
)


class SmoothCompressibleD2VStepStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT_STATE = 1
    ENERGY_EQUILIBRIUM_FAILED = 2
    COLLISION_FAILED = 3
    TRANSPORT_STATE_INVALID = 4
    CONSERVATION_FAILED = 5
    FORCING_FAILED = 6
    BOUNDARY_FAILED = 7
    SUPPORT_FAILED = 8


class D2V17PeriodicTransportPlan(StrictModule, NonTrainableState):
    """Exact fixed-step periodic pull transport for one D2V17 population field."""

    quadrature: CertifiedDiscreteVelocityQuadrature
    spatial_shape: tuple[int, int] = eqx.field(static=True)
    cell_spacing: tuple[float, float] = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    pull_offsets: tuple[tuple[int, int], ...] = eqx.field(static=True)
    maximum_reach: tuple[int, int] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        spatial_shape: tuple[int, int],
        cell_spacing: tuple[float, float],
        time_step: float,
        /,
    ):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        if (
            quadrature.name != "D2V17"
            or quadrature.dimension != 2
            or quadrature.population_count != 17
            or quadrature.transport_kind != "integer_lattice"
        ):
            raise ValueError("Exact periodic pull transport requires certified D2V17.")
        shape = tuple(spatial_shape)
        spacing = tuple(float(value) for value in cell_spacing)
        step = float(time_step)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("spatial_shape must contain two positive extents.")
        if len(spacing) != 2 or any(
            not isfinite(value) or value <= 0.0 for value in spacing
        ):
            raise ValueError("cell_spacing must contain two finite positive values.")
        if not isfinite(step) or step <= 0.0:
            raise ValueError("time_step must be finite and positive.")
        velocities = np.asarray(quadrature.velocities, dtype=np.float64)
        scaled = velocities * step / np.asarray(spacing)[None, :]
        rounded = np.rint(scaled)
        tolerance = (
            64.0 * np.finfo(scaled.dtype).eps * max(float(np.max(np.abs(scaled))), 1.0)
        )
        if float(np.max(np.abs(scaled - rounded))) > tolerance:
            raise ValueError(
                "D2V17 pull transport requires velocity*time_step/cell_spacing to be integer."
            )
        offsets = tuple(tuple(row) for row in rounded)
        reach = tuple(max(abs(row[axis]) for row in offsets) for axis in range(2))
        if any(shape[axis] <= 2 * reach[axis] for axis in range(2)):
            raise ValueError(
                "Periodic D2V17 domains must exceed twice the maximum pull reach on each axis."
            )
        self.quadrature = quadrature
        self.spatial_shape = shape
        self.cell_spacing = spacing
        self.time_step = step
        self.pull_offsets = offsets
        self.maximum_reach = reach
        self.plan_id = canonical_fingerprint(
            {
                "kind": "d2v17-periodic-pull-transport",
                "quadrature": quadrature.quadrature_id,
                "spatial_shape": list(shape),
                "cell_spacing": list(spacing),
                "time_step": step,
                "pull_offsets": [list(row) for row in offsets],
            }
        )

    @property
    def cell_volume(self) -> float:
        return prod(self.cell_spacing)

    def validate_populations(self, populations: ArrayLike, /) -> Array:
        values = self.quadrature.validate_populations(populations)
        if values.ndim != 3 or values.shape[:2] != self.spatial_shape:
            raise ValueError(
                "Periodic D2V17 populations must have shape spatial_shape + (17,)."
            )
        return values

    def transport(self, populations: ArrayLike, /) -> Array:
        values = self.validate_populations(populations)
        routed = tuple(
            jnp.roll(values[..., index], shift=offset, axis=(0, 1))
            for index, offset in enumerate(self.pull_offsets)
        )
        return jnp.stack(routed, axis=-1)


class SmoothCompressibleD2VConservationEvidence(StrictModule):
    pre_step_content: Array
    post_collision_content: Array
    post_forcing_content: Array
    post_transport_content: Array
    expected_source_change: Array
    boundary_exchange: Array
    collision_residual: Array
    source_residual: Array
    transport_residual: Array
    total_residual: Array
    balance_residual: Array
    maximum_absolute_residual: Array


class SmoothCompressibleD2VStepEvidence(StrictModule):
    pre_step_realizability: SmoothCompressibleRealizabilityEvidence
    equilibrium: SmoothCompressibleLearnedEquilibriumEvidence
    collision: SmoothCompressibleCollisionEvidence
    forcing: SmoothCompressibleD2VForcingResult | None
    boundary: SmoothCompressibleD2VBoundaryResult | None
    post_transport_realizability: SmoothCompressibleRealizabilityEvidence
    conservation: SmoothCompressibleD2VConservationEvidence
    status: Array
    successful: Array


class SmoothCompressibleD2VStepResult(StrictModule):
    candidate_state: SmoothCompressibleKineticState
    accepted_state: SmoothCompressibleKineticState
    evidence: SmoothCompressibleD2VStepEvidence
    successful: Array
    rollback_applied: Array
    residual: Array
    work: Array


class SmoothCompressibleD2V17SpatialPlan(StrictModule):
    """Construction plan for fixed-grid periodic D2V17 dynamics."""

    method: SmoothCompressibleD2VKineticMethod
    energy_plan: PositiveEnergyEquilibriumPlan
    transport: D2V17PeriodicTransportPlan
    boundary: AbstractSmoothCompressibleD2VBoundaryPlan | None
    forcing: (
        ZeroSmoothCompressibleD2VForcingPlan | SmoothCompressibleD2VBodyForcingPlan | None
    )
    population_floor: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: SmoothCompressibleD2VKineticMethod,
        energy_plan: PositiveEnergyEquilibriumPlan,
        spatial_shape: tuple[int, int],
        cell_spacing: tuple[float, float],
        time_step: float,
        /,
        *,
        population_floor: float = 0.0,
        conservation_tolerance: float = 1.0e-11,
        boundary: AbstractSmoothCompressibleD2VBoundaryPlan | None = None,
        forcing: (
            ZeroSmoothCompressibleD2VForcingPlan
            | SmoothCompressibleD2VBodyForcingPlan
            | None
        ) = None,
    ):
        if not isinstance(method, SmoothCompressibleD2VKineticMethod):
            raise TypeError("method must be a SmoothCompressibleD2VKineticMethod.")
        if not isinstance(energy_plan, PositiveEnergyEquilibriumPlan):
            raise TypeError("energy_plan must be a PositiveEnergyEquilibriumPlan.")
        transport = D2V17PeriodicTransportPlan(
            method.quadrature, spatial_shape, cell_spacing, time_step
        )
        if energy_plan.quadrature.quadrature_id != method.quadrature.quadrature_id:
            raise ValueError("Energy plan and kinetic method quadratures must match.")
        floor = float(population_floor)
        tolerance = float(conservation_tolerance)
        if not isfinite(floor) or floor < 0.0:
            raise ValueError("population_floor must be finite and non-negative.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("conservation_tolerance must be finite and positive.")
        if boundary is not None:
            if not isinstance(boundary, AbstractSmoothCompressibleD2VBoundaryPlan):
                raise TypeError(
                    "boundary must implement AbstractSmoothCompressibleD2VBoundaryPlan."
                )
            topology = boundary.topology
            if (
                topology.quadrature.quadrature_id != method.quadrature.quadrature_id
                or topology.spatial_shape != transport.spatial_shape
                or topology.cell_spacing != transport.cell_spacing
                or topology.time_step != transport.time_step
            ):
                raise ValueError(
                    "Boundary topology must match the prepared D2V17 transport."
                )
        if forcing is not None:
            if not isinstance(
                forcing,
                (
                    ZeroSmoothCompressibleD2VForcingPlan,
                    SmoothCompressibleD2VBodyForcingPlan,
                ),
            ):
                raise TypeError("forcing must be a smooth-compressible D2V forcing plan.")
            if forcing.method.method_id != method.method_id:
                raise ValueError("Forcing plan and kinetic method identities must match.")
        self.method = method
        self.energy_plan = energy_plan
        self.transport = transport
        self.boundary = boundary
        self.forcing = forcing
        self.population_floor = floor
        self.conservation_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "smooth-compressible-d2v17-spatial-plan",
                "method": method.method_id,
                "energy_plan": energy_plan.plan_id,
                "transport": transport.plan_id,
                "boundary": None if boundary is None else boundary.plan_id,
                "forcing": None if forcing is None else forcing.plan_id,
                "population_floor": floor,
                "conservation_tolerance": tolerance,
            }
        )

    def prepare(self, /) -> "PreparedSmoothCompressibleD2V17SpatialDynamics":
        return PreparedSmoothCompressibleD2V17SpatialDynamics(
            self.method,
            self.energy_plan,
            self.transport,
            boundary=self.boundary,
            forcing=self.forcing,
            population_floor=self.population_floor,
            conservation_tolerance=self.conservation_tolerance,
        )


class PreparedSmoothCompressibleD2V17SpatialDynamics(StrictModule):
    """One atomic local-collision and exact-periodic-transport D2V17 owner."""

    method: SmoothCompressibleD2VKineticMethod
    energy_plan: PositiveEnergyEquilibriumPlan
    transport: D2V17PeriodicTransportPlan
    boundary: AbstractSmoothCompressibleD2VBoundaryPlan | None
    forcing: (
        ZeroSmoothCompressibleD2VForcingPlan | SmoothCompressibleD2VBodyForcingPlan | None
    )
    program_manifest: KineticProgramManifest
    safe_state: SmoothCompressibleKineticState = fixed_field()
    population_floor: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: SmoothCompressibleD2VKineticMethod,
        energy_plan: PositiveEnergyEquilibriumPlan,
        transport: D2V17PeriodicTransportPlan,
        /,
        *,
        boundary: AbstractSmoothCompressibleD2VBoundaryPlan | None = None,
        forcing: (
            ZeroSmoothCompressibleD2VForcingPlan
            | SmoothCompressibleD2VBodyForcingPlan
            | None
        ) = None,
        population_floor: float = 0.0,
        conservation_tolerance: float = 1.0e-11,
    ):
        if not isinstance(method, SmoothCompressibleD2VKineticMethod):
            raise TypeError("method must be a SmoothCompressibleD2VKineticMethod.")
        if not isinstance(energy_plan, PositiveEnergyEquilibriumPlan):
            raise TypeError("energy_plan must be a PositiveEnergyEquilibriumPlan.")
        if not isinstance(transport, D2V17PeriodicTransportPlan):
            raise TypeError("transport must be a D2V17PeriodicTransportPlan.")
        if (
            method.quadrature.quadrature_id != energy_plan.quadrature.quadrature_id
            or method.quadrature.quadrature_id != transport.quadrature.quadrature_id
        ):
            raise ValueError("Method, energy plan, and transport quadratures must match.")
        floor = float(population_floor)
        tolerance = float(conservation_tolerance)
        if not isfinite(floor) or floor < 0.0:
            raise ValueError("population_floor must be finite and non-negative.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("conservation_tolerance must be finite and positive.")
        if boundary is not None:
            if not isinstance(boundary, AbstractSmoothCompressibleD2VBoundaryPlan):
                raise TypeError(
                    "boundary must implement AbstractSmoothCompressibleD2VBoundaryPlan."
                )
            topology = boundary.topology
            if (
                topology.quadrature.quadrature_id != method.quadrature.quadrature_id
                or topology.spatial_shape != transport.spatial_shape
                or topology.cell_spacing != transport.cell_spacing
                or topology.time_step != transport.time_step
            ):
                raise ValueError(
                    "Boundary topology must match the prepared D2V17 transport."
                )
        if forcing is not None:
            if not isinstance(
                forcing,
                (
                    ZeroSmoothCompressibleD2VForcingPlan,
                    SmoothCompressibleD2VBodyForcingPlan,
                ),
            ):
                raise TypeError("forcing must be a smooth-compressible D2V forcing plan.")
            if forcing.method.method_id != method.method_id:
                raise ValueError("Forcing plan and kinetic method identities must match.")
        dtype = method.quadrature.velocities.dtype
        reference_temperature = method.quadrature.reference_temperature
        reference_energy = (
            method.material.gas_constant
            * reference_temperature
            / (method.material.gamma - 1.0)
        )
        safe_state = method.equilibrium(
            jnp.asarray((1.0, 0.0, 0.0, reference_energy), dtype=dtype)
        )
        safe_realizability = method.realizability(safe_state, population_floor=floor)
        if not bool(safe_realizability.realizable):
            raise ValueError("Prepared D2V17 runtime reference state is not realizable.")
        has_boundary = boundary is not None
        has_source = forcing is not None
        program_manifest = smooth_compressible_spatial_dvm_manifest(
            method.quadrature.quadrature_id,
            f"dtype:{method.quadrature.velocities.dtype}",
            method.quadrature.population_count,
            method.quadrature.dimension,
            max(transport.maximum_reach),
            has_boundary,
            has_source,
            has_boundary and boundary.retain_history,
        )
        self.method = method
        self.energy_plan = energy_plan
        self.transport = transport
        self.boundary = boundary
        self.forcing = forcing
        self.program_manifest = program_manifest
        self.safe_state = safe_state
        self.population_floor = floor
        self.conservation_tolerance = tolerance
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-smooth-compressible-d2v17-spatial-dynamics",
                "method": method.method_id,
                "energy_plan": energy_plan.plan_id,
                "transport": transport.plan_id,
                "boundary": None if boundary is None else boundary.plan_id,
                "forcing": None if forcing is None else forcing.plan_id,
                "program_manifest": program_manifest.manifest_id,
                "population_floor": floor,
                "conservation_tolerance": tolerance,
                "step_order": [
                    "collision",
                    "source" if forcing is not None else "no_source",
                    "boundary_route"
                    if boundary is not None
                    else "periodic_pull_transport",
                    "atomic_audit",
                ],
            }
        )

    @property
    def required_step_size(self) -> float:
        return self.transport.time_step

    @property
    def allows_step_reduction(self) -> bool:
        return False

    def validate_state(self, state: SmoothCompressibleKineticState, /) -> None:
        self.method.validate_state(state)
        self.transport.validate_populations(state.particle_populations)
        self.transport.validate_populations(state.total_energy_populations)

    def _domain_content(self, state: SmoothCompressibleKineticState, /) -> Array:
        volume = jnp.asarray(
            self.transport.cell_volume, dtype=state.particle_populations.dtype
        )
        mass = jnp.sum(state.particle_populations) * volume
        momentum = (
            jnp.sum(
                ein.contract(
                    "xyq,qd->xyd",
                    state.particle_populations,
                    self.method.quadrature.velocities,
                ),
                axis=(0, 1),
            )
            * volume
        )
        energy = jnp.sum(state.total_energy_populations) * volume
        return jnp.concatenate((mass[None], momentum, energy[None]), axis=0)

    def _broadcast_safe_state(
        self, state: SmoothCompressibleKineticState, /
    ) -> SmoothCompressibleKineticState:
        shape = state.particle_populations.shape
        return SmoothCompressibleKineticState(
            jnp.broadcast_to(self.safe_state.particle_populations, shape),
            jnp.broadcast_to(self.safe_state.total_energy_populations, shape),
        )

    @staticmethod
    def _select_state(
        predicate: Array,
        accepted: SmoothCompressibleKineticState,
        rejected: SmoothCompressibleKineticState,
        /,
    ) -> SmoothCompressibleKineticState:
        return SmoothCompressibleKineticState(
            jnp.where(
                predicate,
                accepted.particle_populations,
                rejected.particle_populations,
            ),
            jnp.where(
                predicate,
                accepted.total_energy_populations,
                rejected.total_energy_populations,
            ),
        )

    def step_with_energy_dual(
        self,
        state: SmoothCompressibleKineticState,
        time_step: ArrayLike,
        dual: ArrayLike,
        transport_args: Any = None,
        /,
        *,
        boundary_parameters: Any = None,
    ) -> SmoothCompressibleD2VStepResult:
        self.validate_state(state)
        step = jnp.asarray(time_step, dtype=state.particle_populations.dtype)
        step_matches = jnp.isfinite(step) & (
            jnp.abs(step - self.transport.time_step)
            <= 8.0
            * jnp.finfo(step.dtype).eps
            * jnp.maximum(jnp.abs(step), self.transport.time_step)
        )
        pre_realizability = self.method.realizability(
            state, population_floor=self.population_floor
        )
        usable = pre_realizability.realizable & step_matches
        safe_state = self._broadcast_safe_state(state)
        working_state = self._select_state(usable, state, safe_state)
        collision = self.method.collide_with_energy_dual_with_evidence(
            working_state,
            jnp.asarray(self.transport.time_step, dtype=step.dtype),
            dual,
            self.energy_plan,
            transport_args,
        )
        forcing_result = (
            None
            if self.forcing is None
            else self.forcing.apply(
                collision.candidate_state,
                jnp.asarray(self.transport.time_step, dtype=step.dtype),
            )
        )
        post_forcing = (
            collision.candidate_state
            if forcing_result is None
            else forcing_result.accepted_state
        )
        forcing_success = (
            jnp.asarray(True) if forcing_result is None else forcing_result.successful
        )
        boundary_result = (
            None
            if self.boundary is None
            else self.boundary.route(post_forcing, boundary_parameters)
        )
        if boundary_result is None:
            candidate = SmoothCompressibleKineticState(
                self.transport.transport(post_forcing.particle_populations),
                self.transport.transport(post_forcing.total_energy_populations),
            )
            boundary_success = jnp.asarray(True)
            boundary_exchange = jnp.zeros((4,), dtype=step.dtype)
        else:
            candidate = boundary_result.candidate_state
            boundary_success = boundary_result.successful
            boundary_exchange = boundary_result.mass_momentum_energy_exchange

        post_realizability = self.method.realizability(
            candidate, population_floor=self.population_floor
        )
        pre_content = self._domain_content(working_state)
        post_collision_content = self._domain_content(collision.candidate_state)
        post_forcing_content = self._domain_content(post_forcing)
        post_transport_content = self._domain_content(candidate)
        volume = jnp.asarray(self.transport.cell_volume, dtype=step.dtype)
        expected_source_change = (
            jnp.zeros((4,), dtype=step.dtype)
            if forcing_result is None
            else jnp.sum(forcing_result.mass_momentum_energy_increment, axis=(0, 1))
            * volume
        )
        collision_residual = post_collision_content - pre_content
        source_residual = (
            post_forcing_content - post_collision_content - expected_source_change
        )
        transport_residual = (
            post_transport_content - post_forcing_content - boundary_exchange
        )
        balance_residual = (
            post_transport_content
            - pre_content
            - expected_source_change
            - boundary_exchange
        )
        maximum_residual = jnp.max(
            jnp.abs(
                jnp.concatenate((collision_residual, source_residual, transport_residual))
            )
        )
        scale = jnp.maximum(jnp.max(jnp.abs(pre_content)), 1.0)
        tolerance = (
            jnp.asarray(self.conservation_tolerance, dtype=step.dtype)
            + 512.0 * jnp.finfo(step.dtype).eps * scale
        )
        conservation_ok = maximum_residual <= tolerance
        successful = (
            usable
            & collision.successful
            & forcing_success
            & boundary_success
            & post_realizability.realizable
            & conservation_ok
        )
        status = jnp.asarray(
            int(SmoothCompressibleD2VStepStatus.SUCCESS), dtype=jnp.int32
        )
        status = jnp.where(
            ~usable,
            int(SmoothCompressibleD2VStepStatus.INVALID_INPUT_STATE),
            status,
        )
        status = jnp.where(
            usable & ~collision.successful,
            int(SmoothCompressibleD2VStepStatus.COLLISION_FAILED),
            status,
        )
        status = jnp.where(
            usable & collision.successful & ~forcing_success,
            int(SmoothCompressibleD2VStepStatus.FORCING_FAILED),
            status,
        )
        status = jnp.where(
            usable & collision.successful & forcing_success & ~boundary_success,
            int(SmoothCompressibleD2VStepStatus.BOUNDARY_FAILED),
            status,
        )
        prior_stages_ok = (
            usable & collision.successful & forcing_success & boundary_success
        )
        status = jnp.where(
            prior_stages_ok & ~post_realizability.realizable,
            int(SmoothCompressibleD2VStepStatus.TRANSPORT_STATE_INVALID),
            status,
        )
        status = jnp.where(
            prior_stages_ok & post_realizability.realizable & ~conservation_ok,
            int(SmoothCompressibleD2VStepStatus.CONSERVATION_FAILED),
            status,
        ).astype(jnp.int32)
        accepted = self._select_state(successful, candidate, state)
        conservation = SmoothCompressibleD2VConservationEvidence(
            pre_step_content=pre_content,
            post_collision_content=post_collision_content,
            post_forcing_content=post_forcing_content,
            post_transport_content=post_transport_content,
            expected_source_change=expected_source_change,
            boundary_exchange=boundary_exchange,
            collision_residual=collision_residual,
            source_residual=source_residual,
            transport_residual=transport_residual,
            balance_residual=balance_residual,
            total_residual=balance_residual,
            maximum_absolute_residual=maximum_residual,
        )
        evidence = SmoothCompressibleD2VStepEvidence(
            pre_step_realizability=pre_realizability,
            equilibrium=collision.equilibrium_evidence,
            collision=collision.collision_evidence,
            forcing=forcing_result,
            boundary=boundary_result,
            post_transport_realizability=post_realizability,
            conservation=conservation,
            status=status,
            successful=successful,
        )
        extra_work = int(self.forcing is not None) + int(self.boundary is not None)
        return SmoothCompressibleD2VStepResult(
            candidate_state=candidate,
            accepted_state=accepted,
            evidence=evidence,
            successful=successful,
            rollback_applied=~successful,
            residual=maximum_residual,
            work=jnp.asarray(
                (2 + extra_work) * self.method.quadrature.population_count,
                dtype=jnp.int32,
            ),
        )

    def step_with_model(
        self,
        state: SmoothCompressibleKineticState,
        time_step: ArrayLike,
        model: Any,
        binding_plan: Any,
        transport_args: Any = None,
        /,
        *,
        boundary_parameters: Any = None,
    ) -> tuple[SmoothCompressibleD2VStepResult, Any]:
        """Advance with explicit trainable model arrays and support evidence."""

        from ...closure_data._kinetic_equilibrium import (
            LearnedEnergyEquilibriumBindingPlan,
        )

        if not isinstance(binding_plan, LearnedEnergyEquilibriumBindingPlan):
            raise TypeError("binding_plan must be a LearnedEnergyEquilibriumBindingPlan.")
        if (
            binding_plan.equilibrium_plan.plan_id != self.energy_plan.plan_id
            or binding_plan.material.material_id != self.method.material.material_id
        ):
            raise ValueError(
                "Learned binding plan does not match the spatial kinetic method."
            )
        moments = self.method.moments(state)
        dual, support = binding_plan.predict_dual_with_evidence(model, moments.conserved)
        result = self.step_with_energy_dual(
            state,
            time_step,
            dual,
            transport_args,
            boundary_parameters=boundary_parameters,
        )
        support_success = jnp.all(support.successful)
        successful = result.successful & support_success
        accepted = self._select_state(successful, result.candidate_state, state)
        status = jnp.where(
            support_success,
            result.evidence.status,
            int(SmoothCompressibleD2VStepStatus.SUPPORT_FAILED),
        ).astype(jnp.int32)
        evidence = eqx.tree_at(
            lambda value: (value.status, value.successful),
            result.evidence,
            (status, successful),
        )
        return (
            eqx.tree_at(
                lambda value: (
                    value.accepted_state,
                    value.evidence,
                    value.successful,
                    value.rollback_applied,
                ),
                result,
                (accepted, evidence, successful, ~successful),
            ),
            support,
        )

    def step_oracle(
        self,
        state: SmoothCompressibleKineticState,
        time_step: ArrayLike,
        transport_args: Any = None,
        /,
        *,
        boundary_parameters: Any = None,
    ) -> tuple[SmoothCompressibleD2VStepResult, EnergyEquilibriumEvidence]:
        self.validate_state(state)
        pre_realizability = self.method.realizability(
            state, population_floor=self.population_floor
        )
        safe_state = self._broadcast_safe_state(state)
        working_state = self._select_state(
            pre_realizability.realizable, state, safe_state
        )
        moments = self.method.moments(working_state)
        target_flux = (moments.total_energy + moments.pressure)[
            ..., None
        ] * moments.velocity
        oracle = self.energy_plan.solve(moments.total_energy, target_flux)
        result = self.step_with_energy_dual(
            state,
            time_step,
            oracle.dual,
            transport_args,
            boundary_parameters=boundary_parameters,
        )
        oracle_success = jnp.all(oracle.evidence.successful & oracle.evidence.converged)
        successful = result.successful & oracle_success
        accepted = self._select_state(successful, result.candidate_state, state)
        status = jnp.where(
            oracle_success,
            result.evidence.status,
            int(SmoothCompressibleD2VStepStatus.ENERGY_EQUILIBRIUM_FAILED),
        ).astype(jnp.int32)
        evidence = eqx.tree_at(
            lambda value: (value.status, value.successful),
            result.evidence,
            (status, successful),
        )
        return (
            eqx.tree_at(
                lambda value: (
                    value.accepted_state,
                    value.evidence,
                    value.successful,
                    value.rollback_applied,
                ),
                result,
                (accepted, evidence, successful, ~successful),
            ),
            oracle.evidence,
        )


__all__ = [
    "D2V17PeriodicTransportPlan",
    "PreparedSmoothCompressibleD2V17SpatialDynamics",
    "SmoothCompressibleD2V17SpatialPlan",
    "SmoothCompressibleD2VConservationEvidence",
    "SmoothCompressibleD2VStepEvidence",
    "SmoothCompressibleD2VStepResult",
    "SmoothCompressibleD2VStepStatus",
]
