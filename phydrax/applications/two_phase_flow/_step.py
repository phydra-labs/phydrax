#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative structured two-phase VOF step.

Per step (temporal split, first order):

1. Geometric phase transport.  Directional sweeps with exact PLIC swept
   volumes and the Weymouth & Yue (2010, J. Comput. Phys. 229:2853) dilation
   term ``c (div u)`` with the step-constant indicator ``c = [alpha^n > 1/2]``;
   the sweep order is the cyclic rotation ``(k, k + 1, ...)`` with
   ``k = step_index mod D``.  Liquid volume stays within ``[0, 1]`` without
   clipping for directional Courant numbers at most one half and is conserved
   to the projection's divergence residual.
2. Momentum transport with the same face liquid/gas mass fluxes, plus the
   penalty moving-body force, then impermeable-wall enforcement.
3. Implicit variable viscosity ``(rho_f + dt A_mu) u = rho_f u*``.
4. Balanced interfacial force ``phi_f (G alpha)_f`` at ``alpha^{n+1}`` with
   ``phi = sigma kappa - (rho_l - rho_g) g . (x_I - Z)`` (height-function
   curvature, reduced gravity at the interface position ``x_I``), added after
   the viscous stage so that its large gradient part reaches the projection
   unaltered: diffusing it first would turn part of it into a non-gradient
   velocity that the projection cannot remove (the same force/viscosity order
   as Popinet's centered solver).
5. Variable-density projection with ``rho(alpha^{n+1})``; the stored pressure
   is the projection pressure, i.e. the dynamic pressure of the reduced-gravity
   formulation (absolute pressure: ``PreparedIncompressibleTwoPhaseVOF.
   absolute_pressure``).
"""

from __future__ import annotations

from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import (
    MACBoundaryStageData,
    MACCapillaryForceResult,
    StructuredPLICReconstruction,
    VariableSurfaceTensionPolicy,
)
from ...solver import (
    AbstractFixedStepMethod,
    FixedStepResult,
    MACCompartmentProjectionResult,
    MACVariationalViscosityResult,
)
from ...typing import checked
from ._bubbly_flow import (
    BubblyFlowEvidence,
    BubblyFlowPlan,
    BubblyFlowState,
    BubblyStepContext,
)
from ._flux_bundle import TwoPhaseFluxBundle
from ._vof import (
    alpha_bound_tolerance,
    FaceTuple,
    PreparedIncompressibleTwoPhaseVOF,
    TwoPhaseInterfaceGeometry,
    TwoPhaseTopologyEvidence,
    TwoPhaseVOFState,
)


def _cell_net_flux(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    difference = (
        jnp.roll(moved, -1, axis=0) - moved if periodic else moved[1:] - moved[:-1]
    )
    return jnp.moveaxis(difference, 0, axis)


def _face_upwind(value: Array, flux: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    moved_flux = jnp.moveaxis(flux, axis, 0)
    if periodic:
        lower = jnp.roll(moved, 1, axis=0)
        upper = moved
    else:
        lower = jnp.concatenate((moved[:1], moved), axis=0)
        upper = jnp.concatenate((moved, moved[-1:]), axis=0)
    return jnp.moveaxis(jnp.where(moved_flux >= 0.0, lower, upper), 0, axis)


def _cell_from_faces(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    centered = (
        0.5 * (moved + jnp.roll(moved, -1, axis=0))
        if periodic
        else 0.5 * (moved[:-1] + moved[1:])
    )
    return jnp.moveaxis(centered, 0, axis)


def _faces_from_cell(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        faces = 0.5 * (jnp.roll(moved, 1, axis=0) + moved)
    else:
        interior = 0.5 * (moved[:-1] + moved[1:])
        faces = jnp.concatenate((moved[:1], interior, moved[-1:]), axis=0)
    return jnp.moveaxis(faces, 0, axis)


def _weighted_norm(values: FaceTuple, weights: FaceTuple, /) -> Array:
    return jnp.sqrt(
        sum(
            (
                jnp.sum(weight * value**2)
                for value, weight in zip(values, weights, strict=True)
            ),
            start=jnp.zeros((), dtype=values[0].dtype),
        )
    )


def _face_inner(left: FaceTuple, right: FaceTuple, weights: FaceTuple, /) -> Array:
    return sum(
        (
            jnp.sum(weight * a * b)
            for a, b, weight in zip(left, right, weights, strict=True)
        ),
        start=jnp.zeros((), dtype=left[0].dtype),
    )


class TwoPhaseMovingBodyPlan(StrictModule, NonTrainableState):
    center: Array
    radius: float = eqx.field(static=True)
    velocity: Array
    penalty: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: ArrayLike,
        radius: float,
        /,
        *,
        velocity: ArrayLike | tuple[float, float, float] = (0.0, 0.0, 0.0),
        penalty: float = 1.0,
    ) -> None:
        center_ = jnp.asarray(center)
        velocity_ = jnp.asarray(velocity, dtype=center_.dtype)
        radius_ = float(radius)
        penalty_ = float(penalty)
        if center_.shape != velocity_.shape or center_.ndim != 1:
            raise ValueError("Two-phase body center/velocity shapes are invalid.")
        if radius_ <= 0.0 or penalty_ < 0.0:
            raise ValueError("Two-phase body radius/penalty are invalid.")
        self.center = center_
        self.radius = radius_
        self.velocity = velocity_
        self.penalty = penalty_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-phase-moving-body-plan",
                "radius": radius_,
                "penalty": penalty_,
            }
        )


class TwoPhaseVOFLedger(StrictModule):
    """Cumulative conservation, work and energy-estimate ledger of accepted steps.

    No single total-energy balance is claimed. ``work_energy_residual`` is the
    discrete kinetic-energy theorem defect
    ``dKE - (W_capillary + W_gravity + W_wall + W_body + W_contact +
    W_pressure) + D_viscous`` with force work paired to the midpoint velocity.
    ``pressure_work`` is nonzero only for the mixed compartment projection;
    ``contact_work`` is the near-contact action and is never folded into the
    capillary term. ``gravitational_energy_residual`` is
    ``dE_gravity + W_gravity``. For the constant-tension route,
    ``surface_energy_change`` estimates ``sigma dA`` from PLIC facet measure;
    the material-scalar route makes no closed surface-free-energy claim.
    """

    liquid_volume_change: Array
    gas_volume_change: Array
    momentum_change: Array
    kinetic_energy_change: Array
    gravitational_energy_change: Array
    surface_energy_change: Array
    viscous_dissipation: Array
    wall_work: Array
    capillary_work: Array
    gravity_work: Array
    body_work: Array
    contact_work: Array
    pressure_work: Array
    reinitialization_dissipation: Array
    pressure_residual: Array
    divergence_residual: Array
    topology_event_count: Array
    work_energy_residual: Array
    gravitational_energy_residual: Array
    surface_energy_residual: Array

    @classmethod
    def zeros(cls, dtype: DTypeLike, /) -> Self:
        zero = jnp.zeros((), dtype=dtype)
        return cls(*((zero,) * 20))


class TwoPhaseStepEvidence(StrictModule):
    """Per-step status evidence.

    ``capillary_balance_residual`` is the dual-measure-weighted relative norm
    of the interfacial force not balanced by the new pressure gradient (zero at
    a discrete static equilibrium); ``parasitic_velocity`` the maximum face
    speed after projection; ``sweep_offset`` the first sweep axis.
    """

    alpha_minimum: Array
    alpha_maximum: Array
    liquid_volume_residual: Array
    mass_flux_residual: Array
    material_scalar_residual: Array
    pressure_residual: Array
    divergence_residual: Array
    capillary_balance_residual: Array
    capillary_pressure_jump: Array
    surface_tension_minimum: Array
    surface_tension_maximum: Array
    marangoni_force_norm: Array
    parasitic_velocity: Array
    advective_courant: Array
    capillary_step_limit: Array
    curvature_valid_count: Array
    curvature_fallback_count: Array
    curvature_underresolved_count: Array
    unsupported_face_count: Array
    plic_residual: Array
    viscous_residual: Array
    viscous_converged: Array
    sweep_offset: Array
    clsvof_correction: Array
    topology_event_count: Array
    finite: Array
    geometry_accepted: Array
    solid_interface_conflict_count: Array
    successful: Array


class TwoPhaseContinuationState(StrictModule):
    state: TwoPhaseVOFState
    pressure: Array
    ledger: TwoPhaseVOFLedger
    topology: TwoPhaseTopologyEvidence
    evidence: TwoPhaseStepEvidence | None
    fluxes: TwoPhaseFluxBundle
    bubbles: BubblyFlowState | None
    bubble_evidence: BubblyFlowEvidence | None


class _PhaseTransport(StrictModule):
    liquid_content: Array
    liquid_rates: FaceTuple
    total_rates: FaceTuple
    dilation_rates: tuple[Array, ...]
    dilation_rate: Array
    initial_reconstruction: StructuredPLICReconstruction
    courant: Array


class _ProjectionOutcome(StrictModule):
    momentum: FaceTuple
    velocity: FaceTuple
    pressure: Array
    divergence: Array
    pressure_residual: Array
    iterations: Array
    finite: Array
    converged: Array
    compartment: MACCompartmentProjectionResult | None


class _StepStages(StrictModule):
    """Computed stages of one step before evidence and ledger construction."""

    step_size: Array
    previous_alpha: Array
    velocity: FaceTuple
    transport: _PhaseTransport
    fluxes: TwoPhaseFluxBundle
    material_scalar_content: dict[str, Array]
    alpha: Array
    face_density: FaceTuple
    geometry: TwoPhaseInterfaceGeometry
    force: MACCapillaryForceResult | None
    contact_force: FaceTuple
    viscous: MACVariationalViscosityResult | None
    projection: _ProjectionOutcome
    bubbles: BubblyFlowState | None
    bubble_evidence: BubblyFlowEvidence | None
    body_work: Array


class _ForceSplit(StrictModule):
    """Capillary and reduced-gravity parts of the balanced interfacial force."""

    capillary: FaceTuple
    gravity: FaceTuple
    balance: Array
    pressure_jump: Array
    valid: Array
    unsupported: Array


class IncompressibleTwoPhaseVOFMethod(AbstractFixedStepMethod):
    """Conservative geometric VOF/mass/momentum step with balanced forces."""

    two_phase: PreparedIncompressibleTwoPhaseVOF
    body: TwoPhaseMovingBodyPlan | None
    bubbles: BubblyFlowPlan | None
    method_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        two_phase: PreparedIncompressibleTwoPhaseVOF,
        /,
        *,
        body: TwoPhaseMovingBodyPlan | None = None,
        bubbles: BubblyFlowPlan | None = None,
    ) -> None:
        if body is not None and not isinstance(body, TwoPhaseMovingBodyPlan):
            raise TypeError("body must be TwoPhaseMovingBodyPlan or None.")
        if bubbles is not None and not isinstance(bubbles, BubblyFlowPlan):
            raise TypeError("bubbles must be BubblyFlowPlan or None.")
        if bubbles is not None and bubbles.two_phase.prepared_id != two_phase.prepared_id:
            raise ValueError("bubbles must bind the same prepared two-phase plan.")
        if body is not None and two_phase.geometry is not None:
            raise ValueError(
                "VOF cannot combine qualified cut measures with the independent penalty moving-body path."
            )
        self.two_phase = two_phase
        self.body = body
        self.bubbles = bubbles
        self.method_id = canonical_fingerprint(
            {
                "kind": "incompressible-two-phase-vof-method",
                "two_phase": two_phase.prepared_id,
                "body": "none" if body is None else body.plan_id,
                "bubbles": "none" if bubbles is None else bubbles.plan_id,
                "interface_authority": "alpha-volume",
                "phase_transport": "weymouth-yue-geometric-plic",
                "splitting": "cyclic-directional-lie",
                "interfacial_force": "balanced-potential",
                "momentum_transport": "consistent-phase-mass-flux",
            }
        )

    def initial_continuation(
        self,
        state: TwoPhaseVOFState,
        /,
        *,
        bubbles: BubblyFlowState | None = None,
    ) -> TwoPhaseContinuationState:
        if self.bubbles is None and bubbles is not None:
            raise ValueError("bubbles requires a bubbly-flow method.")
        if self.bubbles is not None and bubbles is None:
            raise ValueError("A bubbly-flow method requires an initial bubble state.")
        alpha = self.two_phase.alpha(state)
        zero = jnp.asarray(0.0, dtype=alpha.dtype)
        count = jnp.asarray(0, dtype=jnp.int32)
        initial_evidence = TwoPhaseStepEvidence(
            alpha_minimum=jnp.min(alpha),
            alpha_maximum=jnp.max(alpha),
            liquid_volume_residual=zero,
            mass_flux_residual=zero,
            material_scalar_residual=zero,
            pressure_residual=zero,
            divergence_residual=zero,
            capillary_balance_residual=zero,
            capillary_pressure_jump=zero,
            surface_tension_minimum=zero,
            surface_tension_maximum=zero,
            marangoni_force_norm=zero,
            parasitic_velocity=zero,
            advective_courant=zero,
            capillary_step_limit=jnp.asarray(jnp.inf, dtype=alpha.dtype),
            curvature_valid_count=count,
            curvature_fallback_count=count,
            curvature_underresolved_count=count,
            unsupported_face_count=count,
            plic_residual=zero,
            viscous_residual=zero,
            viscous_converged=jnp.asarray(True),
            sweep_offset=count,
            clsvof_correction=zero,
            topology_event_count=count,
            finite=jnp.asarray(True),
            successful=jnp.asarray(True),
            geometry_accepted=(
                jnp.asarray(True)
                if self.two_phase.geometry is None
                else self.two_phase.geometry.accepted
            ),
            solid_interface_conflict_count=count,
        )
        zero_faces = tuple(jnp.zeros_like(value) for value in state.momentum)
        fluxes = TwoPhaseFluxBundle(
            previous_liquid_content=state.liquid_content,
            liquid_rates=zero_faces,
            total_rates=zero_faces,
            dilation_rates=tuple(
                jnp.zeros_like(state.liquid_content) for _ in state.momentum
            ),
            sweep_offset=count,
            step_size=zero,
        )
        return TwoPhaseContinuationState(
            state=state,
            pressure=jnp.zeros_like(alpha),
            ledger=TwoPhaseVOFLedger.zeros(alpha.dtype),
            topology=self.two_phase.topology_evidence(state),
            evidence=initial_evidence,
            fluxes=fluxes,
            bubbles=bubbles,
            bubble_evidence=(
                None
                if self.bubbles is None or bubbles is None
                else self.bubbles.initial_evidence(bubbles, alpha)
            ),
        )

    def _advect(
        self,
        state: TwoPhaseVOFState,
        alpha: Array,
        velocity: FaceTuple,
        dt: Array,
        step_index: Array,
        gas_dilatation: Array,
        /,
    ) -> _PhaseTransport:
        """Directionally split geometric PLIC transport of liquid content."""

        two_phase = self.two_phase
        discretization = two_phase.plan.discretization
        periodic = tuple(axis.periodic for axis in discretization.grid.structured_axes)
        dimension = len(discretization.cell_shape)
        total = tuple(
            component * measure
            for component, measure in zip(
                velocity, two_phase.face_open_measure, strict=True
            )
        )
        indicator = (alpha > 0.5).astype(alpha.dtype)
        initial = two_phase.reconstruct(alpha)
        zeros = tuple(jnp.zeros_like(value) for value in total)
        zero_cells = tuple(jnp.zeros_like(alpha) for _ in range(dimension))
        compartment_rate = gas_dilatation * two_phase.cell_fluid_measure / dimension

        def ordered(offset: int) -> Any:
            def run(content: Array) -> tuple[Array, FaceTuple, tuple[Array, ...]]:
                rates = zeros
                dilation_rates = zero_cells
                reconstruction = initial
                for position in range(dimension):
                    axis = (offset + position) % dimension
                    current = two_phase.alpha_from_content(content)
                    if position > 0:
                        reconstruction = two_phase.reconstruct(current)
                    fraction = two_phase.plic_plan.swept_fraction(
                        current, reconstruction, axis, velocity[axis] * dt
                    )
                    liquid = total[axis] * fraction
                    dilation = indicator * (
                        _cell_net_flux(total[axis], axis, periodic[axis])
                        - compartment_rate
                    )
                    net = _cell_net_flux(liquid, axis, periodic[axis]) - dilation
                    content = content - dt * net
                    rates = rates[:axis] + (liquid,) + rates[axis + 1 :]
                    dilation_rates = (
                        dilation_rates[:axis]
                        + (dilation,)
                        + dilation_rates[axis + 1 :]
                    )
                return content, rates, dilation_rates

            return run

        content, liquid_rates, dilation_rates = jax.lax.switch(
            jnp.asarray(step_index, dtype=jnp.int32) % dimension,
            [ordered(offset) for offset in range(dimension)],
            state.liquid_content,
        )
        dilation = sum(dilation_rates, start=jnp.zeros_like(alpha))
        courant = jnp.max(
            jnp.stack(
                tuple(
                    jnp.max(jnp.abs(component))
                    * dt
                    / jnp.min(jnp.asarray(grid_axis.interval_widths, dtype=dt.dtype))
                    for component, grid_axis in zip(
                        velocity, discretization.grid.structured_axes, strict=True
                    )
                )
            )
        )
        return _PhaseTransport(
            liquid_content=content,
            liquid_rates=liquid_rates,
            total_rates=total,
            dilation_rates=dilation_rates,
            dilation_rate=dilation,
            initial_reconstruction=initial,
            courant=courant,
        )

    def _consistent_momentum_rate(
        self,
        velocity: FaceTuple,
        total_fluxes: FaceTuple,
        liquid_fluxes: FaceTuple,
        /,
    ) -> FaceTuple:
        """Face momentum-density rates from the transported phase mass fluxes.

        The cell momentum-flux divergence uses the same liquid/gas mass fluxes
        as the geometric phase transport and is returned per unit volume
        (``-div(m u)``), not divided by a density: the face momentum then
        changes by the momentum actually carried, and the velocity follows
        from the new face mass.  Dividing by the old cell density would
        amplify liquid momentum entering light gas cells by the density ratio.
        """

        material = self.two_phase.plan.material
        periodic = tuple(
            axis.periodic
            for axis in self.two_phase.plan.discretization.grid.structured_axes
        )
        mass_fluxes = tuple(
            material.liquid_density * liquid + material.gas_density * (total - liquid)
            for total, liquid in zip(total_fluxes, liquid_fluxes, strict=True)
        )
        cell_velocity = jnp.stack(
            tuple(
                _cell_from_faces(component, axis, periodic[axis])
                for axis, component in enumerate(velocity)
            ),
            axis=-1,
        )
        fluid_volume = self.two_phase.cell_fluid_measure
        active = fluid_volume > 0.0
        rates = []
        for component_axis in range(len(velocity)):
            net = sum(
                _cell_net_flux(
                    flux
                    * _face_upwind(
                        cell_velocity[..., component_axis],
                        flux,
                        axis,
                        periodic[axis],
                    ),
                    axis,
                    periodic[axis],
                )
                for axis, flux in enumerate(mass_fluxes)
            )
            rate = jnp.where(active, -net / jnp.where(active, fluid_volume, 1.0), 0.0)
            rates.append(_faces_from_cell(rate, component_axis, periodic[component_axis]))
        return tuple(rates)

    def _interfacial_force(
        self,
        alpha: Array,
        geometry: TwoPhaseInterfaceGeometry,
        material_scalar_content: dict[str, Array],
        /,
    ) -> MACCapillaryForceResult | None:
        """Balanced capillary, Marangoni and reduced-gravity force at ``alpha``."""

        two_phase = self.two_phase
        curvature = geometry.curvature
        if curvature is None:
            return None
        surface = (
            None
            if (
                two_phase.plan.material.surface_tension == 0.0
                and two_phase.plan.surface_tension_law is None
            )
            else curvature.evidence
        )
        variable_evaluation = None
        scalar_name = two_phase.plan.surface_tension_scalar
        if scalar_name is not None:
            content = material_scalar_content[scalar_name]
            volume = two_phase.cell_fluid_measure
            active = volume > 0.0
            scalar = jnp.where(
                active, content / jnp.where(active, volume, 1.0), 0.0
            )
            periodic = tuple(
                axis.periodic
                for axis in two_phase.plan.discretization.grid.structured_axes
            )
            face_gradient = two_phase.operators.gradient(scalar)
            cell_gradient = jnp.stack(
                tuple(
                    _cell_from_faces(value, axis, periodic[axis])
                    for axis, value in enumerate(face_gradient)
                ),
                axis=-1,
            )
            policy = two_phase.capillarity.policy
            if not isinstance(policy, VariableSurfaceTensionPolicy):
                raise ValueError("Variable surface-tension policy is unavailable.")
            variable_evaluation = policy.evaluate(
                two_phase.plan.discretization.cell_centers,
                scalar[..., None],
                geometry.plic.normal,
                cell_gradient[..., None, :],
            )
        if two_phase.plan.gravity_enabled:
            potential, support = two_phase.gravity_potential(curvature)
            return two_phase.capillarity.evaluate(
                alpha,
                surface,
                variable_surface_tension=variable_evaluation,
                body_potential=potential,
                body_support=support,
            )
        return two_phase.capillarity.evaluate(
            alpha, surface, variable_surface_tension=variable_evaluation
        )

    def _body_force(self, velocity: FaceTuple, dt: Array, /) -> tuple[FaceTuple, Array]:
        if self.body is None:
            return tuple(jnp.zeros_like(v) for v in velocity), jnp.asarray(
                0.0, dtype=dt.dtype
            )
        discretization = self.two_phase.plan.discretization
        force = []
        work = jnp.asarray(0.0, dtype=dt.dtype)
        for axis, component in enumerate(velocity):
            coordinates = discretization.face_centers[axis]
            distance = jnp.sqrt(jnp.sum((coordinates - self.body.center) ** 2, axis=-1))
            mask = distance <= self.body.radius
            target = self.body.velocity[axis]
            acceleration = (
                self.body.penalty * jnp.where(mask, target - component, 0.0) / dt
            )
            force.append(acceleration)
            work = work + jnp.sum(
                self.two_phase.face_open_measure[axis] * component * acceleration
            )
        return tuple(force), work

    def _project(
        self,
        velocity: FaceTuple,
        face_density: FaceTuple,
        dt: Array,
        pressure: Array,
        stage: MACBoundaryStageData,
        /,
    ) -> _ProjectionOutcome:
        """Project the stage velocity; the stored momentum is ``rho_f V_f u``.

        The variable-density projection acts on the momentum density
        ``rho_f u`` so that its pressure is the physical (dynamic) pressure.
        """

        two_phase = self.two_phase
        dual = two_phase.face_open_dual_measure
        inverse_density = tuple(1.0 / rho for rho in face_density)
        if two_phase.sharp_projection is None:
            projection = two_phase.projection.project(
                tuple(
                    rho * value for rho, value in zip(face_density, velocity, strict=True)
                ),
                inverse_density,
                dt,
                pressure=pressure,
            )
            return _ProjectionOutcome(
                momentum=tuple(
                    measure * value
                    for measure, value in zip(dual, projection.momentum, strict=True)
                ),
                velocity=projection.velocity,
                pressure=projection.pressure_increment,
                divergence=projection.divergence_after,
                pressure_residual=jnp.sqrt(
                    jnp.sum(
                        two_phase.cell_fluid_measure * projection.pressure_residual**2
                    )
                ),
                iterations=jnp.asarray(
                    projection.linear.diagnostics.iterations, dtype=jnp.int32
                ).reshape(()),
                finite=projection.finite,
                converged=projection.converged,
                compartment=None,
            )
        sharp = two_phase.sharp_projection.project(
            velocity,
            tuple(dt * value for value in inverse_density),
            stage,
            pressure=pressure,
        )
        return _ProjectionOutcome(
            momentum=tuple(
                rho * measure * value
                for rho, measure, value in zip(
                    face_density, dual, sharp.velocity, strict=True
                )
            ),
            velocity=sharp.velocity,
            pressure=sharp.pressure,
            divergence=sharp.divergence_after,
            pressure_residual=sharp.linear.diagnostics.residual_norm,
            iterations=jnp.asarray(
                sharp.linear.diagnostics.iterations, dtype=jnp.int32
            ).reshape(()),
            finite=sharp.force.finite & jnp.isfinite(sharp.divergence_norm),
            converged=sharp.accepted,
            compartment=None,
        )

    def _project_for_bubbles(
        self,
        context: BubblyStepContext | None,
        velocity: FaceTuple,
        face_density: FaceTuple,
        dt: Array,
        pressure: Array,
        stage: MACBoundaryStageData,
        /,
    ) -> _ProjectionOutcome:
        """Use the mixed projection exactly when a compartment constraint exists."""

        if (
            self.bubbles is None
            or context is None
            or context.constraint is None
        ):
            return self._project(velocity, face_density, dt, pressure, stage)
        inverse_density = tuple(1.0 / rho for rho in face_density)
        result = self.bubbles.project(
            context,
            tuple(
                rho * value
                for rho, value in zip(face_density, velocity, strict=True)
            ),
            inverse_density,
            dt,
            pressure,
        )
        dual = self.two_phase.face_open_dual_measure
        residual = jnp.max(
            jnp.stack(
                (
                    jnp.max(jnp.abs(result.continuity_residual)),
                    jnp.max(jnp.abs(result.compartment_pressure_residual)),
                    jnp.abs(result.atmosphere_pressure_residual),
                    jnp.abs(result.schur_residual),
                )
            )
        )
        iterations = jnp.maximum(
            jnp.max(
                jnp.asarray(
                    result.pressure_solves.diagnostics.iterations,
                    dtype=jnp.int32,
                )
            ),
            jnp.asarray(result.schur.diagnostics.iterations, dtype=jnp.int32).reshape(
                ()
            ),
        )
        return _ProjectionOutcome(
            momentum=tuple(
                measure * value
                for measure, value in zip(dual, result.momentum, strict=True)
            ),
            velocity=result.velocity,
            pressure=result.pressure,
            divergence=result.divergence_after - result.divergence_target,
            pressure_residual=residual,
            iterations=iterations,
            finite=result.finite,
            converged=result.successful,
            compartment=result,
        )

    def _transport_scalars(
        self,
        state: TwoPhaseVOFState,
        transport: _PhaseTransport,
        dt: Array,
        /,
    ) -> dict[str, Array]:
        periodic = tuple(
            axis.periodic
            for axis in self.two_phase.plan.discretization.grid.structured_axes
        )
        scalars = {}
        for name, content in state.phase_scalar_content.items():
            concentration = jnp.where(
                state.liquid_content > 0.0,
                content
                / jnp.where(state.liquid_content > 0.0, state.liquid_content, 1.0),
                0.0,
            )
            net = sum(
                _cell_net_flux(
                    flux * _face_upwind(concentration, flux, axis, periodic[axis]),
                    axis,
                    periodic[axis],
                )
                for axis, flux in enumerate(transport.liquid_rates)
            )
            # The Weymouth-Yue dilation term carries liquid with its scalar.
            scalars[name] = (
                content - dt * net + dt * concentration * transport.dilation_rate
            )
        return scalars

    def _transport_material_scalars(
        self,
        state: TwoPhaseVOFState,
        fluxes: TwoPhaseFluxBundle,
        /,
    ) -> dict[str, Array]:
        """Conservatively replay total face fluxes for mixture material scalars."""

        periodic = tuple(
            axis.periodic
            for axis in self.two_phase.plan.discretization.grid.structured_axes
        )
        dimension = len(periodic)
        volume = self.two_phase.cell_fluid_measure
        active = volume > 0.0

        def transport_one(initial: Array) -> Array:
            def ordered(offset: int) -> Any:
                def run(content: Array) -> Array:
                    for position in range(dimension):
                        axis = (offset + position) % dimension
                        concentration = jnp.where(
                            active,
                            content / jnp.where(active, volume, 1.0),
                            0.0,
                        )
                        flux = fluxes.total_rates[axis] * _face_upwind(
                            concentration,
                            fluxes.total_rates[axis],
                            axis,
                            periodic[axis],
                        )
                        content = content - fluxes.step_size * _cell_net_flux(
                            flux, axis, periodic[axis]
                        )
                    return content

                return run

            return jax.lax.switch(
                fluxes.sweep_offset % dimension,
                [ordered(offset) for offset in range(dimension)],
                initial,
            )

        return {
            name: transport_one(content)
            for name, content in state.material_scalar_content.items()
        }

    def step(
        self,
        step_index: Array,
        time: Array,
        state: TwoPhaseContinuationState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        stages = self._stages(step_index, time, state, step_size, args)
        dt = stages.step_size
        candidate_state, clsvof = self._candidate_state(state.state, stages)
        topology = self.two_phase.topology_evidence(
            candidate_state, stages.previous_alpha, stages.geometry.plic
        )
        split = self._force_split(stages)
        evidence = self._evidence(step_index, state, stages, split, topology, clsvof)
        ledger = self._ledger_increment(state, stages, split, topology, clsvof, evidence)
        candidate = TwoPhaseContinuationState(
            state=candidate_state,
            pressure=stages.projection.pressure,
            ledger=jax.tree.map(
                lambda total, increment: total + increment, state.ledger, ledger
            ),
            topology=topology,
            evidence=evidence,
            fluxes=stages.fluxes,
            bubbles=stages.bubbles,
            bubble_evidence=stages.bubble_evidence,
        )
        successful = evidence.successful
        accepted = jax.tree.map(
            lambda proposed, current: jnp.where(successful, proposed, current),
            candidate,
            state,
        )
        work = stages.projection.iterations + (
            jnp.asarray(0, dtype=jnp.int32)
            if stages.viscous is None
            else stages.viscous.iterations
        )
        return FixedStepResult(
            candidate_state=candidate,
            accepted_state=accepted,
            successful=successful,
            residual=jnp.max(
                jnp.stack(
                    (
                        evidence.liquid_volume_residual,
                        evidence.divergence_residual,
                        evidence.pressure_residual,
                    )
                )
            ),
            iterations=work,
            work=work,
            transform_applied=jnp.asarray(False),
            transform_correction_norm=jnp.asarray(0.0, dtype=dt.dtype),
        )

    def _stages(
        self,
        step_index: Array,
        time: Array,
        state: TwoPhaseContinuationState,
        step_size: Array,
        args: Any,
        /,
    ) -> _StepStages:
        """Transport, interfacial force, viscosity and projection of one step."""

        two_phase = self.two_phase
        dt = jnp.asarray(step_size, dtype=state.state.liquid_content.dtype)
        previous_alpha = two_phase.alpha(state.state)
        velocity = two_phase.velocity(state.state)
        stage = two_phase.boundaries.evaluate(time, args)
        gas_dilatation = (
            jnp.zeros_like(previous_alpha)
            if state.bubbles is None
            else state.bubbles.dilatation
        )
        transport = self._advect(
            state.state, previous_alpha, velocity, dt, step_index, gas_dilatation
        )
        fluxes = TwoPhaseFluxBundle(
            previous_liquid_content=state.state.liquid_content,
            liquid_rates=transport.liquid_rates,
            total_rates=transport.total_rates,
            dilation_rates=transport.dilation_rates,
            sweep_offset=jnp.asarray(step_index, dtype=jnp.int32)
            % len(two_phase.plan.discretization.cell_shape),
            step_size=dt,
        )
        alpha = two_phase.alpha_from_content(transport.liquid_content)
        face_density = two_phase.face_density(two_phase.mixture_density(alpha))
        dual = two_phase.face_open_dual_measure
        momentum_rate = self._consistent_momentum_rate(
            velocity, transport.total_rates, transport.liquid_rates
        )
        material_scalar_content = self._transport_material_scalars(state.state, fluxes)
        geometry = two_phase.interface_geometry(alpha)
        force = self._interfacial_force(
            alpha, geometry, material_scalar_content
        )
        face_force = (
            tuple(jnp.zeros_like(value) for value in velocity)
            if force is None
            else force.face_force
        )
        bubble_context: BubblyStepContext | None = None
        if self.bubbles is not None:
            if state.bubbles is None:
                raise ValueError("The continuation carries no bubble state.")
            bubble_context = self.bubbles.prepare(
                state.bubbles,
                fluxes,
                alpha,
                state.pressure,
                geometry.plic.facet_measure,
            )
        contact_force = (
            tuple(jnp.zeros_like(value) for value in velocity)
            if bubble_context is None or bubble_context.near_contact is None
            else bubble_context.near_contact.face_force
        )
        body_force, body_work = self._body_force(velocity, dt)
        predictor = tuple(
            old + dt * measure * (advection + rho * body)
            for old, rho, measure, advection, body in zip(
                state.state.momentum,
                face_density,
                dual,
                momentum_rate,
                body_force,
                strict=True,
            )
        )
        predictor_velocity = two_phase.boundaries.enforce(
            tuple(
                jnp.where(
                    rho * measure > 0.0,
                    value / jnp.where(rho * measure > 0.0, rho * measure, 1.0),
                    0.0,
                )
                for value, rho, measure in zip(predictor, face_density, dual, strict=True)
            ),
            stage,
        )
        viscous = (
            None
            if two_phase.viscosity is None
            else two_phase.viscosity.solve(
                predictor_velocity,
                face_density,
                two_phase.mixture_viscosity(alpha),
                dt,
                stage,
            )
        )
        viscous_velocity = predictor_velocity if viscous is None else viscous.velocity
        stage_velocity = two_phase.boundaries.enforce(
            tuple(
                value + dt * (interfacial + contact) / rho
                for value, interfacial, contact, rho in zip(
                    viscous_velocity,
                    face_force,
                    contact_force,
                    face_density,
                    strict=True,
                )
            ),
            stage,
        )
        projection = self._project_for_bubbles(
            bubble_context, stage_velocity, face_density, dt, state.pressure, stage
        )
        bubble_state = None
        bubble_evidence = None
        if (
            self.bubbles is not None
            and bubble_context is not None
            and state.bubbles is not None
        ):
            midpoint = tuple(
                0.5 * (old + new)
                for old, new in zip(velocity, projection.velocity, strict=True)
            )
            bubble_state, bubble_evidence = self.bubbles.finish(
                state.bubbles,
                bubble_context,
                projection.compartment,
                midpoint,
                dt,
            )
        return _StepStages(
            step_size=dt,
            previous_alpha=previous_alpha,
            velocity=velocity,
            transport=transport,
            fluxes=fluxes,
            material_scalar_content=material_scalar_content,
            alpha=alpha,
            face_density=face_density,
            geometry=geometry,
            force=force,
            contact_force=contact_force,
            viscous=viscous,
            projection=projection,
            bubbles=bubble_state,
            bubble_evidence=bubble_evidence,
            body_work=body_work,
        )

    def _candidate_state(
        self, state: TwoPhaseVOFState, stages: _StepStages, /
    ) -> tuple[TwoPhaseVOFState, Array]:
        """Candidate extensive state and its CLSVOF reconciliation defect."""

        two_phase = self.two_phase
        alpha_level_set = two_phase.level_set_from_alpha(stages.alpha)
        level_set = 0.5 * state.level_set + 0.5 * alpha_level_set
        candidate = TwoPhaseVOFState(
            liquid_content=stages.transport.liquid_content,
            momentum=stages.projection.momentum,
            phase_scalar_content=self._transport_scalars(
                state, stages.transport, stages.step_size
            ),
            material_scalar_content=stages.material_scalar_content,
            level_set=level_set,
            geometry_epoch=(
                jnp.asarray(-1, dtype=jnp.int32)
                if two_phase.geometry is None
                else two_phase.geometry.epoch
            ),
            geometry_id=(
                "" if two_phase.geometry is None else two_phase.geometry.realization_id
            ),
        )
        return candidate, jnp.sqrt(jnp.mean((level_set - alpha_level_set) ** 2))

    def _force_split(self, stages: _StepStages, /) -> _ForceSplit:
        """Capillary/gravity parts of the balanced force and its projection balance."""

        two_phase = self.two_phase
        dtype = stages.step_size.dtype
        alpha_gradient = two_phase.operators.gradient(stages.alpha)
        force = stages.force
        if force is None:
            zero = tuple(jnp.zeros_like(value) for value in alpha_gradient)
            return _ForceSplit(
                capillary=zero,
                gravity=zero,
                balance=jnp.zeros((), dtype=dtype),
                pressure_jump=jnp.zeros((), dtype=dtype),
                valid=jnp.asarray(True),
                unsupported=jnp.asarray(0, dtype=jnp.int32),
            )
        capillary = tuple(
            potential * gradient + marangoni
            for potential, gradient, marangoni in zip(
                force.capillary_face_potential,
                alpha_gradient,
                force.marangoni_face_force,
                strict=True,
            )
        )
        gravity = tuple(
            total - part for total, part in zip(force.face_force, capillary, strict=True)
        )
        dual = two_phase.face_open_dual_measure
        mismatch = tuple(
            value - gradient
            for value, gradient in zip(
                force.face_force,
                two_phase.operators.gradient(stages.projection.pressure),
                strict=True,
            )
        )
        scale = _weighted_norm(force.face_force, dual)
        return _ForceSplit(
            capillary=capillary,
            gravity=gravity,
            balance=jnp.where(
                scale > 0.0,
                _weighted_norm(mismatch, dual) / jnp.where(scale > 0.0, scale, 1.0),
                0.0,
            ),
            pressure_jump=force.pressure_jump,
            valid=force.valid,
            unsupported=force.unsupported_face_count,
        )

    def _capillary_step_limit(self, stages: _StepStages, /) -> Array:
        two_phase = self.two_phase
        curvature = stages.geometry.curvature
        if curvature is None or stages.force is None:
            return jnp.asarray(jnp.inf, dtype=stages.step_size.dtype)
        mean_density = 0.5 * (
            two_phase.plan.material.liquid_density
            + two_phase.plan.material.gas_density
        )
        return two_phase.capillarity.capillary_step(
            jnp.full(stages.alpha.shape, mean_density, dtype=stages.step_size.dtype),
            interface_active=curvature.evidence.interface_active,
            surface_tension=(
                None
                if two_phase.plan.surface_tension_law is None
                else jnp.full_like(
                    stages.alpha, stages.force.surface_tension_maximum
                )
            ),
        )

    def _evidence(
        self,
        step_index: Array,
        state: TwoPhaseContinuationState,
        stages: _StepStages,
        split: _ForceSplit,
        topology: TwoPhaseTopologyEvidence,
        clsvof: Array,
        /,
    ) -> TwoPhaseStepEvidence:
        """Status evidence and the fail-closed acceptance decision."""

        two_phase = self.two_phase
        dt = stages.step_size
        dtype = dt.dtype
        alpha = stages.alpha
        transport = stages.transport
        projection = stages.projection
        viscous = stages.viscous
        count = jnp.asarray(0, dtype=jnp.int32)
        volume_scale = jnp.maximum(jnp.sum(state.state.liquid_content), 1.0)
        liquid_residual = (
            jnp.abs(jnp.sum(transport.liquid_content - state.state.liquid_content))
            / volume_scale
        )
        material_residual = (
            jnp.zeros((), dtype=dtype)
            if not state.state.material_scalar_content
            else jnp.max(
                jnp.stack(
                    tuple(
                        jnp.abs(
                            jnp.sum(stages.material_scalar_content[name])
                            - jnp.sum(content)
                        )
                        / jnp.maximum(jnp.abs(jnp.sum(content)), 1.0)
                        for name, content in state.state.material_scalar_content.items()
                    )
                )
            )
        )
        material_finite = (
            jnp.asarray(True)
            if not stages.material_scalar_content
            else jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(content))
                        for content in stages.material_scalar_content.values()
                    )
                )
            )
        )
        divergence = jnp.sqrt(
            jnp.sum(two_phase.cell_fluid_measure * projection.divergence**2)
        )
        geometry_accepted = (
            jnp.asarray(True)
            if two_phase.geometry is None
            else two_phase.geometry.accepted
            & (state.state.geometry_epoch == two_phase.geometry.epoch)
        )
        conflict_count = jnp.sum(
            stages.geometry.plic.solid_interface_conflict, dtype=jnp.int32
        )
        curvature = stages.geometry.curvature
        capillary_limit = self._capillary_step_limit(stages)
        viscous_converged = jnp.asarray(True) if viscous is None else viscous.successful
        finite = (
            topology.finite
            & projection.finite
            & jnp.all(jnp.isfinite(alpha))
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(v)) for v in projection.momentum))
            )
            & (jnp.asarray(True) if viscous is None else viscous.finite)
            & material_finite
        )
        bubble_ok = (
            jnp.asarray(True)
            if stages.bubble_evidence is None
            else stages.bubble_evidence.successful
        )
        tolerance = alpha_bound_tolerance(alpha.dtype)
        alpha_minimum = jnp.min(alpha)
        alpha_maximum = jnp.max(alpha)
        successful = (
            finite
            & projection.converged
            & geometry_accepted
            & topology.valid
            & transport.initial_reconstruction.valid
            & (conflict_count == 0)
            & (alpha_minimum >= -tolerance)
            & (alpha_maximum <= 1.0 + tolerance)
            & (transport.courant <= 0.5)
            & split.valid
            & (dt <= capillary_limit)
            & viscous_converged
            & bubble_ok
            & (
                material_residual
                <= 4096.0 * jnp.finfo(alpha.dtype).eps
            )
        )
        return TwoPhaseStepEvidence(
            alpha_minimum=alpha_minimum,
            alpha_maximum=alpha_maximum,
            liquid_volume_residual=liquid_residual,
            mass_flux_residual=jnp.abs(dt * jnp.sum(transport.dilation_rate))
            / volume_scale,
            material_scalar_residual=material_residual,
            pressure_residual=projection.pressure_residual,
            divergence_residual=divergence,
            capillary_balance_residual=split.balance,
            capillary_pressure_jump=split.pressure_jump,
            surface_tension_minimum=(
                jnp.zeros((), dtype=dtype)
                if stages.force is None
                else stages.force.surface_tension_minimum
            ),
            surface_tension_maximum=(
                jnp.zeros((), dtype=dtype)
                if stages.force is None
                else stages.force.surface_tension_maximum
            ),
            marangoni_force_norm=(
                jnp.zeros((), dtype=dtype)
                if stages.force is None
                else _weighted_norm(
                    stages.force.marangoni_face_force,
                    two_phase.face_open_dual_measure,
                )
            ),
            parasitic_velocity=jnp.max(
                jnp.stack(tuple(jnp.max(jnp.abs(value)) for value in projection.velocity))
            ),
            advective_courant=transport.courant,
            capillary_step_limit=capillary_limit,
            curvature_valid_count=count if curvature is None else curvature.valid_count,
            curvature_fallback_count=(
                count if curvature is None else curvature.fallback_count
            ),
            curvature_underresolved_count=(
                count if curvature is None else curvature.underresolved_count
            ),
            unsupported_face_count=split.unsupported,
            plic_residual=jnp.maximum(
                jnp.max(jnp.abs(transport.initial_reconstruction.residual)),
                jnp.max(jnp.abs(stages.geometry.plic.reconstruction_residual)),
            ),
            viscous_residual=(
                jnp.zeros((), dtype=dtype)
                if viscous is None
                else jnp.asarray(viscous.residual_norm, dtype=dtype).reshape(())
            ),
            viscous_converged=viscous_converged,
            sweep_offset=jnp.asarray(step_index, dtype=jnp.int32)
            % len(two_phase.plan.discretization.cell_shape),
            clsvof_correction=clsvof,
            topology_event_count=topology.component_proxy,
            finite=finite,
            successful=successful,
            geometry_accepted=geometry_accepted,
            solid_interface_conflict_count=conflict_count,
        )

    def _ledger_increment(
        self,
        state: TwoPhaseContinuationState,
        stages: _StepStages,
        split: _ForceSplit,
        topology: TwoPhaseTopologyEvidence,
        clsvof: Array,
        evidence: TwoPhaseStepEvidence,
        /,
    ) -> TwoPhaseVOFLedger:
        """Conservation, work and energy-estimate increments of one step."""

        two_phase = self.two_phase
        dual = two_phase.face_open_dual_measure
        dt = stages.step_size
        dtype = dt.dtype
        projection = stages.projection
        transport = stages.transport
        viscous = stages.viscous
        # Force power pairs with the midpoint velocity, the discrete form of
        # d(rho u^2 / 2) = rho (u^n + u^{n+1}) / 2 . du.
        midpoint = tuple(
            0.5 * (old + new)
            for old, new in zip(stages.velocity, projection.velocity, strict=True)
        )
        previous_face_density = two_phase.face_density(
            two_phase.mixture_density(stages.previous_alpha)
        )
        kinetic_before = 0.5 * _face_inner(
            stages.velocity,
            stages.velocity,
            tuple(
                rho * measure
                for rho, measure in zip(previous_face_density, dual, strict=True)
            ),
        )
        kinetic_after = 0.5 * _face_inner(
            projection.velocity,
            projection.velocity,
            tuple(
                rho * measure
                for rho, measure in zip(stages.face_density, dual, strict=True)
            ),
        )
        surface_tension = two_phase.plan.material.surface_tension
        surface_change = surface_tension * (
            jnp.sum(stages.geometry.plic.facet_measure)
            - jnp.sum(transport.initial_reconstruction.facet_measure)
        )
        gravity_change = two_phase.gravitational_energy(
            transport.liquid_content
        ) - two_phase.gravitational_energy(state.state.liquid_content)
        zero = jnp.zeros((), dtype=dtype)
        dissipation = zero if viscous is None else dt * viscous.dissipation
        wall_work = zero if viscous is None else dt * viscous.wall_power
        capillary_work = dt * _face_inner(midpoint, split.capillary, dual)
        gravity_work = dt * _face_inner(midpoint, split.gravity, dual)
        body_work = dt * stages.body_work
        contact_work = (
            zero
            if stages.bubble_evidence is None
            else stages.bubble_evidence.near_contact_work
        )
        pressure_work = (
            zero
            if stages.bubble_evidence is None
            else stages.bubble_evidence.pressure_work
        )
        kinetic_change = kinetic_after - kinetic_before
        liquid_change = jnp.sum(transport.liquid_content - state.state.liquid_content)
        return TwoPhaseVOFLedger(
            liquid_volume_change=liquid_change,
            gas_volume_change=-liquid_change,
            momentum_change=sum(
                (
                    jnp.sum(new - old)
                    for new, old in zip(
                        projection.momentum, state.state.momentum, strict=True
                    )
                ),
                start=zero,
            ),
            kinetic_energy_change=kinetic_change,
            gravitational_energy_change=gravity_change,
            surface_energy_change=surface_change,
            viscous_dissipation=dissipation,
            wall_work=wall_work,
            capillary_work=capillary_work,
            gravity_work=gravity_work,
            body_work=body_work,
            contact_work=contact_work,
            pressure_work=pressure_work,
            reinitialization_dissipation=clsvof,
            pressure_residual=evidence.pressure_residual,
            divergence_residual=evidence.divergence_residual,
            topology_event_count=topology.component_proxy,
            work_energy_residual=kinetic_change
            - (
                capillary_work
                + gravity_work
                + wall_work
                + body_work
                + contact_work
                + pressure_work
            )
            + dissipation,
            gravitational_energy_residual=gravity_change + gravity_work,
            surface_energy_residual=surface_change + capillary_work,
        )


__all__ = [
    "IncompressibleTwoPhaseVOFMethod",
    "TwoPhaseContinuationState",
    "TwoPhaseMovingBodyPlan",
    "TwoPhaseStepEvidence",
    "TwoPhaseVOFLedger",
]
