#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Quasi-2D gravity-driven soap-film tunnel on the symmetric plug-flow route.

A vertical soap-film tunnel (Kellay, Wu & Goldburg, Phys. Rev. Lett. 74,
3975, 1995; Rutgers, Wu & Daniel, Rev. Sci. Instrum. 72, 3025, 2001) is a
film that falls between two wires; gravity accelerates it until air friction
balances its weight. The film obeys the symmetric plug-flow equations of
``phydrax.interfacial_transport.SurfacePlugFlowPlan`` (Chomaz, J. Fluid Mech.
442, 2001): thickness and surfactant variations make the film compressible
through the Marangoni elasticity, with wave speed ``c_M = sqrt(2 E / (rho h))``.
The tunnel composes that owner with a ``PlugFlowBoundary``:

- inlet ``x = 0``: prescribed velocity, thickness, surfactant (and dissolved)
  inflow;
- outlet ``x = L``: donor-cell outflow with a traction-free film;
- wires ``y = 0, W``: no-slip (or free-slip) and closed to transport;
- obstacle rim: no-slip and closed; the constraint force on the rim is the
  force the film exerts on the cylinder, including the rim line tension.

With linear air drag ``C (u - u_air)`` per film area and gravity ``g`` along
``+x`` a uniform film has the terminal velocity ``u_T = rho h g / C``.
Every step reports inlet/outlet fluxes, conservation residuals, the momentum
ledger, obstacle and wire forces and the film Mach number ``max |u| / c_M``;
``strouhal`` estimates the shedding frequency from the recorded lift.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import fixed_field, parameter_field, ParameterOwner
from ..._validation import positive_finite_float, positive_integer
from ...interfacial_transport import (
    AdsorptionKinetics,
    FilmStepStatus,
    FilmTransportScheme,
    LangmuirSurfactantLaw,
    PlugFlowBoundary,
    PlugFlowBoundaryKind,
    prepare_film_surface,
    PreparedSurfacePlugFlow,
    SurfacePlugFlowPlan,
    SurfacePlugFlowState,
    SurfacePlugFlowStepResult,
)
from ...linalg import PreparedSparseFactorization
from ...typing import checked, parse
from ._geometry import SoapFilmTunnelGeometry, SoapFilmTunnelMesh


SoapFilmWireKind: TypeAlias = Literal["no-slip", "free-slip"]
"""Wire boundary: the film sticks to the wires (``no-slip``, a real tunnel)
or slides along them (``free-slip``, an idealized frictionless channel)."""


class SoapFilmInflow(StrictModule, ParameterOwner):
    """Film state entering at the inlet: speed along ``+x``, thickness, surfactant.

    ``dissolved_concentration_mol_m3`` is required exactly for soluble films.
    """

    velocity_m_s: Array = parameter_field()
    thickness_m: Array = parameter_field()
    surface_concentration_mol_m2: Array = parameter_field()
    dissolved_concentration_mol_m3: Array | None = parameter_field()

    def __init__(
        self,
        *,
        velocity_m_s: ArrayLike,
        thickness_m: ArrayLike,
        surface_concentration_mol_m2: ArrayLike,
        dissolved_concentration_mol_m3: ArrayLike | None = None,
    ) -> None:
        velocity = _positive(velocity_m_s, "velocity_m_s")
        thickness = _positive(thickness_m, "thickness_m")
        concentration = _nonnegative(
            surface_concentration_mol_m2, "surface_concentration_mol_m2"
        )
        dissolved = (
            None
            if dissolved_concentration_mol_m3 is None
            else _nonnegative(
                dissolved_concentration_mol_m3, "dissolved_concentration_mol_m3"
            )
        )
        self.velocity_m_s = velocity
        self.thickness_m = thickness
        self.surface_concentration_mol_m2 = concentration
        self.dissolved_concentration_mol_m3 = dissolved


class SoapFilmTunnelPlan(StrictModule, ParameterOwner):
    """Tunnel geometry, film material, inflow and gravity/air-drag drive.

    ``gravity_m_s2`` is the gravity component along the tunnel axis ``+x``
    and ``air_drag_coefficient_kg_m2_s`` the positive linear air drag per film
    area (both faces) toward still air. The remaining material parameters
    follow ``SurfacePlugFlowPlan``.
    """

    geometry: SoapFilmTunnelGeometry
    law: LangmuirSurfactantLaw
    inflow: SoapFilmInflow
    kinetics: AdsorptionKinetics | None
    density_kg_m3: Array = parameter_field()
    viscosity_pa_s: Array = parameter_field()
    surface_shear_viscosity_n_s_m: Array = parameter_field()
    surface_dilatational_viscosity_n_s_m: Array = parameter_field()
    surface_diffusivity_m2_s: Array = parameter_field()
    air_drag_coefficient_kg_m2_s: Array = parameter_field()
    gravity_m_s2: Array = parameter_field()
    wire_kind: SoapFilmWireKind = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    transport_scheme: FilmTransportScheme = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        geometry: SoapFilmTunnelGeometry,
        law: LangmuirSurfactantLaw,
        inflow: SoapFilmInflow,
        /,
        *,
        density_kg_m3: ArrayLike,
        air_drag_coefficient_kg_m2_s: ArrayLike,
        gravity_m_s2: ArrayLike = 9.80665,
        viscosity_pa_s: ArrayLike = 1.0e-3,
        surface_shear_viscosity_n_s_m: ArrayLike = 0.0,
        surface_dilatational_viscosity_n_s_m: ArrayLike = 0.0,
        surface_diffusivity_m2_s: ArrayLike = 0.0,
        kinetics: AdsorptionKinetics | None = None,
        wire_kind: SoapFilmWireKind = "no-slip",
        tolerance: float = 1e-10,
        maximum_iterations: int = 30,
        transport_scheme: FilmTransportScheme = "limited-muscl",
    ) -> None:
        if kinetics is not None and not isinstance(kinetics, AdsorptionKinetics):
            raise TypeError("kinetics must be AdsorptionKinetics or None.")
        if (kinetics is None) != (inflow.dissolved_concentration_mol_m3 is None):
            raise ValueError(
                "The inflow dissolved concentration is required exactly for soluble films."
            )
        inflow_evaluation = law.evaluate(inflow.surface_concentration_mol_m2)
        if not np.all(np.asarray(inflow_evaluation.admissible)):
            raise ValueError(
                "The inflow surface concentration must be finite, nonnegative, below "
                "the Langmuir capacity, and give positive surface tension."
            )
        wires = parse(wire_kind, SoapFilmWireKind, "wire_kind")
        scheme = parse(transport_scheme, FilmTransportScheme, "transport_scheme")
        density = _positive(density_kg_m3, "density_kg_m3")
        drag = _positive(air_drag_coefficient_kg_m2_s, "air_drag_coefficient_kg_m2_s")
        gravity = jnp.asarray(_finite(gravity_m_s2, "gravity_m_s2"))
        coefficients = {
            name: _nonnegative(value, name)
            for name, value in (
                ("viscosity_pa_s", viscosity_pa_s),
                ("surface_shear_viscosity_n_s_m", surface_shear_viscosity_n_s_m),
                (
                    "surface_dilatational_viscosity_n_s_m",
                    surface_dilatational_viscosity_n_s_m,
                ),
                ("surface_diffusivity_m2_s", surface_diffusivity_m2_s),
            )
        }
        tolerance_ = positive_finite_float(tolerance, "tolerance")
        iterations = positive_integer(maximum_iterations, "maximum_iterations")
        self.geometry = geometry
        self.law = law
        self.inflow = inflow
        self.kinetics = kinetics
        self.density_kg_m3 = density
        self.viscosity_pa_s = coefficients["viscosity_pa_s"]
        self.surface_shear_viscosity_n_s_m = coefficients["surface_shear_viscosity_n_s_m"]
        self.surface_dilatational_viscosity_n_s_m = coefficients[
            "surface_dilatational_viscosity_n_s_m"
        ]
        self.surface_diffusivity_m2_s = coefficients["surface_diffusivity_m2_s"]
        self.air_drag_coefficient_kg_m2_s = drag
        self.gravity_m_s2 = gravity
        self.wire_kind = wires
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.transport_scheme = scheme
        self.plan_id = canonical_fingerprint(
            {
                "kind": "soap-film-tunnel",
                "geometry_id": geometry.geometry_id,
                "soluble": kinetics is not None,
                "wire_kind": wires,
                "drag": "linear-air-drag",
                "tolerance": tolerance_,
                "maximum_iterations": iterations,
                "transport_scheme": self.transport_scheme,
            }
        )

    @property
    def terminal_velocity_m_s(self) -> Array:
        """Speed ``rho h g / C`` at which a uniform inflow film's weight balances drag."""
        return (
            self.density_kg_m3
            * self.inflow.thickness_m
            * self.gravity_m_s2
            / self.air_drag_coefficient_kg_m2_s
        )

    def prepare(self) -> PreparedSoapFilmTunnel:
        return PreparedSoapFilmTunnel(self)


class SoapFilmTunnelScales(StrictModule):
    """Inflow scales and dimensionless groups of one prepared tunnel.

    ``kinematic_viscosity_m2_s`` is ``(mu h + 2 mu_s) / (rho h)`` of the
    Trouton and doubled Boussinesq--Scriven shear stresses; the Reynolds
    number uses the reference length (obstacle diameter, else channel width)
    and the inflow speed. ``cell_reynolds_number`` is the inflow speed times
    the rim (else boundary) spacing over the viscosity: the first-order
    donor-cell transport adds a numerical viscosity of order
    ``u dx / 2``, so values well above two mean the resolved wake is more
    viscous than the declared film.
    """

    terminal_velocity_m_s: Array
    drag_relaxation_time_s: Array
    marangoni_speed_m_s: Array
    film_mach_number: Array
    kinematic_viscosity_m2_s: Array
    reynolds_number: Array
    cell_reynolds_number: Array
    reference_length_m: float = eqx.field(static=True)
    blockage_ratio: float = eqx.field(static=True)

    def __init__(self, plan: SoapFilmTunnelPlan, /) -> None:
        inflow = plan.inflow
        geometry = plan.geometry
        mass = plan.density_kg_m3 * inflow.thickness_m
        elasticity = plan.law.evaluate(
            inflow.surface_concentration_mol_m2
        ).gibbs_elasticity_n_m
        viscosity = (
            plan.viscosity_pa_s * inflow.thickness_m
            + 2.0 * plan.surface_shear_viscosity_n_s_m
        ) / mass
        spacing = (
            geometry.mesh_size_m
            if geometry.obstacle_diameter_m is None
            else np.pi * geometry.obstacle_diameter_m / geometry.rim_segments
        )
        self.terminal_velocity_m_s = plan.terminal_velocity_m_s
        self.drag_relaxation_time_s = mass / plan.air_drag_coefficient_kg_m2_s
        self.marangoni_speed_m_s = jnp.sqrt(2.0 * elasticity / mass)
        self.film_mach_number = inflow.velocity_m_s / self.marangoni_speed_m_s
        self.kinematic_viscosity_m2_s = viscosity
        self.reynolds_number = (
            inflow.velocity_m_s * geometry.reference_length_m / viscosity
        )
        self.cell_reynolds_number = inflow.velocity_m_s * spacing / viscosity
        self.reference_length_m = geometry.reference_length_m
        self.blockage_ratio = (
            0.0
            if geometry.obstacle_diameter_m is None
            else geometry.obstacle_diameter_m / geometry.width_m
        )


class SoapFilmTunnelState(StrictModule):
    """Plug-flow film state and elapsed accepted time."""

    film: SurfacePlugFlowState
    time_s: Array

    @checked
    def __init__(self, film: SurfacePlugFlowState, time_s: ArrayLike = 0.0, /) -> None:
        self.film = film
        self.time_s = jnp.asarray(time_s, dtype=jnp.float64)


class SoapFilmTunnelEvidence(StrictModule):
    """Fluxes, ledgers, forces and film Mach number of one tunnel step.

    Rates are step averages: inlet inflow and outlet outflow of liquid volume
    (m^3/s) and of surfactant on both interfaces plus dissolved (mol/s).
    Residuals are the film owner's conservation residuals after boundary
    exchange; ``momentum_residual_n_s`` is the momentum change minus the
    external (gravity, drag) and boundary impulses. ``obstacle_force_n`` and
    ``wire_force_n`` are step-averaged forces the film exerts on the rim and
    on the wires. ``film_mach_number`` is ``max |u| / c_M`` over vertices with
    the local ``c_M = sqrt(2 E(Gamma) / (rho h))`` of the attempted candidate.
    Rejected steps keep the previous state and time; their diagnostics,
    including nonfinite values, describe the rejected candidate.
    """

    time_s: Array
    status: Array
    inflow_volume_rate_m3_s: Array
    outflow_volume_rate_m3_s: Array
    inflow_surfactant_rate_mol_s: Array
    outflow_surfactant_rate_mol_s: Array
    volume_residual_m3: Array
    surfactant_residual_mol: Array
    momentum_residual_n_s: Array
    obstacle_force_n: Array
    wire_force_n: Array
    film_mach_number: Array
    maximum_speed_m_s: Array
    minimum_thickness_m: Array
    courant_number: Array
    marangoni_courant_number: Array
    terminal_nonlinear_stage: Array
    nonlinear_status: Array
    nonlinear_iterations: Array
    nonlinear_residual_norm: Array
    converged: Array

    def __init__(
        self,
        *,
        time_s: Array,
        status: Array,
        inflow_volume_rate_m3_s: Array,
        outflow_volume_rate_m3_s: Array,
        inflow_surfactant_rate_mol_s: Array,
        outflow_surfactant_rate_mol_s: Array,
        volume_residual_m3: Array,
        surfactant_residual_mol: Array,
        momentum_residual_n_s: Array,
        obstacle_force_n: Array,
        wire_force_n: Array,
        film_mach_number: Array,
        maximum_speed_m_s: Array,
        minimum_thickness_m: Array,
        courant_number: Array,
        marangoni_courant_number: Array,
        terminal_nonlinear_stage: Array,
        nonlinear_status: Array,
        nonlinear_iterations: Array,
        nonlinear_residual_norm: Array,
        converged: Array,
    ) -> None:
        self.time_s = jnp.asarray(time_s)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.inflow_volume_rate_m3_s = jnp.asarray(inflow_volume_rate_m3_s)
        self.outflow_volume_rate_m3_s = jnp.asarray(outflow_volume_rate_m3_s)
        self.inflow_surfactant_rate_mol_s = jnp.asarray(inflow_surfactant_rate_mol_s)
        self.outflow_surfactant_rate_mol_s = jnp.asarray(outflow_surfactant_rate_mol_s)
        self.volume_residual_m3 = jnp.asarray(volume_residual_m3)
        self.surfactant_residual_mol = jnp.asarray(surfactant_residual_mol)
        self.momentum_residual_n_s = jnp.asarray(momentum_residual_n_s)
        self.obstacle_force_n = jnp.asarray(obstacle_force_n)
        self.wire_force_n = jnp.asarray(wire_force_n)
        self.film_mach_number = jnp.asarray(film_mach_number)
        self.maximum_speed_m_s = jnp.asarray(maximum_speed_m_s)
        self.minimum_thickness_m = jnp.asarray(minimum_thickness_m)
        self.courant_number = jnp.asarray(courant_number)
        self.marangoni_courant_number = jnp.asarray(marangoni_courant_number)
        self.terminal_nonlinear_stage = jnp.asarray(
            terminal_nonlinear_stage, dtype=jnp.int32
        )
        self.nonlinear_status = jnp.asarray(nonlinear_status, dtype=jnp.int32)
        self.nonlinear_iterations = jnp.asarray(nonlinear_iterations, dtype=jnp.int32)
        self.nonlinear_residual_norm = jnp.asarray(nonlinear_residual_norm)
        self.converged = jnp.asarray(converged, dtype=jnp.bool_)

    @property
    def accepted(self) -> Array:
        return self.status == FilmStepStatus.ACCEPTED


class SoapFilmTunnelStepResult(StrictModule):
    state: SoapFilmTunnelState
    film: SurfacePlugFlowStepResult
    evidence: SoapFilmTunnelEvidence

    def __init__(
        self,
        state: SoapFilmTunnelState,
        film: SurfacePlugFlowStepResult,
        evidence: SoapFilmTunnelEvidence,
        /,
    ) -> None:
        self.state = state
        self.film = film
        self.evidence = evidence

    @property
    def accepted(self) -> Array:
        return self.evidence.accepted


class SoapFilmTunnelResult(StrictModule):
    """Final state and per-step evidence (leading step axis) of a tunnel run."""

    state: SoapFilmTunnelState
    evidence: SoapFilmTunnelEvidence

    def __init__(
        self, state: SoapFilmTunnelState, evidence: SoapFilmTunnelEvidence, /
    ) -> None:
        self.state = state
        self.evidence = evidence

    @property
    def accepted_steps(self) -> Array:
        return jnp.sum(self.evidence.accepted, dtype=jnp.int32)

    @property
    def all_accepted(self) -> Array:
        return jnp.all(self.evidence.accepted)


class CylinderWakeReference(StrictModule):
    """Williamson wake reference with an explicit Allen--Vincenti wall correction.

    ``unconfined_strouhal_number`` is the parallel-shedding fit at the declared
    Reynolds number. ``blockage_corrected_strouhal_number`` maps the same fit
    back to the tunnel inflow-speed convention after correcting the effective
    speed and Reynolds number. The correction is a comparison convention, not
    a calibration of the film model.
    """

    unconfined_strouhal_number: Array
    blockage_corrected_strouhal_number: Array
    velocity_correction: Array
    corrected_reynolds_number: Array
    blockage_ratio: Array

    def __init__(
        self,
        *,
        unconfined_strouhal_number: ArrayLike,
        blockage_corrected_strouhal_number: ArrayLike,
        velocity_correction: ArrayLike,
        corrected_reynolds_number: ArrayLike,
        blockage_ratio: ArrayLike,
    ) -> None:
        self.unconfined_strouhal_number = _positive(
            unconfined_strouhal_number, "unconfined_strouhal_number"
        )
        self.blockage_corrected_strouhal_number = _positive(
            blockage_corrected_strouhal_number,
            "blockage_corrected_strouhal_number",
        )
        self.velocity_correction = _positive(velocity_correction, "velocity_correction")
        self.corrected_reynolds_number = _positive(
            corrected_reynolds_number, "corrected_reynolds_number"
        )
        self.blockage_ratio = _nonnegative(blockage_ratio, "blockage_ratio")


class SheddingStatus(IntEnum):
    """Outcome of a Strouhal estimate; only ``SHEDDING`` carries a frequency."""

    SHEDDING = 0
    NO_SHEDDING = 1
    INSUFFICIENT_RECORD = 2
    REJECTED_STEPS = 3


class StrouhalEstimate(StrictModule):
    """Host estimate of periodic vortex shedding from the obstacle lift.

    The lift ``F_y`` after ``transient_s`` is centered; periods are counted
    between upward zero crossings that follow a descent below minus half the
    amplitude (hysteresis against noise). ``strouhal_number = f d / U`` uses
    the obstacle diameter and inflow speed, and force coefficients use
    ``rho h U^2 d / 2``. The reported standard errors propagate the sample
    standard error of the observed cycle periods; they do not include spatial
    or time-discretization uncertainty. ``NO_SHEDDING`` means the lift
    coefficient amplitude stays below ``minimum_lift_coefficient``.
    """

    status: SheddingStatus = eqx.field(static=True)
    frequency_hz: Array
    strouhal_number: Array
    frequency_standard_error_hz: Array
    strouhal_standard_error: Array
    periods: Array
    lift_coefficient_amplitude: Array
    mean_drag_coefficient: Array
    reynolds_number: Array
    window_s: Array

    def __init__(
        self,
        *,
        status: SheddingStatus,
        frequency_hz: ArrayLike,
        strouhal_number: ArrayLike,
        frequency_standard_error_hz: ArrayLike,
        strouhal_standard_error: ArrayLike,
        periods: ArrayLike,
        lift_coefficient_amplitude: ArrayLike,
        mean_drag_coefficient: ArrayLike,
        reynolds_number: ArrayLike,
        window_s: ArrayLike,
    ) -> None:
        self.status = SheddingStatus(status)
        self.frequency_hz = jnp.asarray(frequency_hz, dtype=jnp.float64)
        self.strouhal_number = jnp.asarray(strouhal_number, dtype=jnp.float64)
        self.frequency_standard_error_hz = jnp.asarray(
            frequency_standard_error_hz, dtype=jnp.float64
        )
        self.strouhal_standard_error = jnp.asarray(
            strouhal_standard_error, dtype=jnp.float64
        )
        self.periods = jnp.asarray(periods, dtype=jnp.int32)
        self.lift_coefficient_amplitude = jnp.asarray(
            lift_coefficient_amplitude, dtype=jnp.float64
        )
        self.mean_drag_coefficient = jnp.asarray(mean_drag_coefficient, dtype=jnp.float64)
        self.reynolds_number = jnp.asarray(reynolds_number, dtype=jnp.float64)
        self.window_s = jnp.asarray(window_s, dtype=jnp.float64)


class PreparedSoapFilmTunnel(StrictModule):
    """Triangulated tunnel with its prepared bordered plug-flow film."""

    plan: SoapFilmTunnelPlan
    channel: SoapFilmTunnelMesh = fixed_field()
    flow: PreparedSurfacePlugFlow
    inlet: Array = fixed_field()
    outlet: Array = fixed_field()
    wires: Array = fixed_field()
    rim: Array = fixed_field()
    scales: SoapFilmTunnelScales
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(self, plan: SoapFilmTunnelPlan, /) -> None:
        channel = plan.geometry.triangulate()
        surface = prepare_film_surface(channel.mesh)
        if not bool(surface.evidence.admissible):
            raise ValueError(
                "The channel triangulation is not conductance admissible "
                f"(minimum cotangent conductance "
                f"{float(surface.evidence.minimum_edge_conductance):.3e}); "
                "refine the rim or lower mesh_size_m."
            )
        inflow = plan.inflow
        self.plan = plan
        self.channel = channel
        self.flow = SurfacePlugFlowPlan(
            surface,
            plan.law,
            density_kg_m3=plan.density_kg_m3,
            viscosity_pa_s=plan.viscosity_pa_s,
            surface_shear_viscosity_n_s_m=plan.surface_shear_viscosity_n_s_m,
            surface_dilatational_viscosity_n_s_m=plan.surface_dilatational_viscosity_n_s_m,
            surface_diffusivity_m2_s=plan.surface_diffusivity_m2_s,
            air_drag_coefficient_kg_m2_s=plan.air_drag_coefficient_kg_m2_s,
            gravity_m_s2=jnp.stack((plan.gravity_m_s2, jnp.zeros(()), jnp.zeros(()))),
            kinetics=plan.kinetics,
            boundary=PlugFlowBoundary(
                surface,
                _boundary_kinds(channel, plan.wire_kind),
                inflow_velocity_m_s=jnp.stack(
                    (inflow.velocity_m_s, jnp.zeros(()), jnp.zeros(()))
                ),
                inflow_thickness_m=inflow.thickness_m,
                inflow_surface_concentration_mol_m2=inflow.surface_concentration_mol_m2,
                inflow_dissolved_concentration_mol_m3=inflow.dissolved_concentration_mol_m3,
            ),
            tolerance=plan.tolerance,
            maximum_iterations=plan.maximum_iterations,
            transport_scheme=plan.transport_scheme,
        ).prepare()
        count = surface.topology.num_vertices
        self.inlet = _mask(channel.inlet_vertices, count)
        self.outlet = _mask(channel.outlet_vertices, count)
        self.wires = _mask(channel.wire_vertices, count)
        self.rim = _mask(channel.rim_vertices, count)
        self.scales = SoapFilmTunnelScales(plan)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-soap-film-tunnel",
                "plan_id": plan.plan_id,
                "mesh_id": channel.mesh_id,
                "flow_id": self.flow.prepared_id,
            }
        )

    def initial_state(
        self, velocity_m_s: ArrayLike | None = None, /
    ) -> SoapFilmTunnelState:
        """Return a uniform inflow film; ``velocity_m_s`` defaults to the inflow velocity.

        A ``(3,)`` or ``(num_vertices, 3)`` velocity may seed a perturbation;
        wire, rim and inlet constraints are applied to it.
        """
        inflow = self.plan.inflow
        velocity = (
            jnp.stack((inflow.velocity_m_s, jnp.zeros(()), jnp.zeros(())))
            if velocity_m_s is None
            else jnp.asarray(velocity_m_s, dtype=jnp.float64)
        )
        film = self.flow.initial_state(
            inflow.thickness_m,
            inflow.surface_concentration_mol_m2,
            velocity,
            inflow.dissolved_concentration_mol_m3,
        )
        return SoapFilmTunnelState(film, 0.0)

    def velocity(self, state: SoapFilmTunnelState, /) -> Array:
        return self.flow.velocity(state.film)

    def thickness(self, state: SoapFilmTunnelState, /) -> Array:
        return state.film.liquid_volume_m3 / self.flow.plan.surface.vertex_area

    @checked
    def step(
        self, state: SoapFilmTunnelState, step_size_s: ArrayLike, /
    ) -> SoapFilmTunnelStepResult:
        """Advance one plug-flow step and report tunnel evidence."""
        step_size = jnp.asarray(step_size_s, dtype=jnp.float64)
        return self._step(state, step_size, None, None)

    def _step(
        self,
        state: SoapFilmTunnelState,
        step_size: Array,
        elastic_factorization: PreparedSparseFactorization | None,
        surfactant_factorization: PreparedSparseFactorization | None,
        /,
    ) -> SoapFilmTunnelStepResult:
        film = self.flow._step(
            state.film,
            step_size,
            elastic_factorization,
            surfactant_factorization,
        )
        time = state.time_s + jnp.where(film.accepted, step_size, 0.0)
        next_state = SoapFilmTunnelState(film.state, time)
        return SoapFilmTunnelStepResult(
            next_state, film, self._evidence(film, time, step_size)
        )

    @checked
    def run(
        self, state: SoapFilmTunnelState, step_size_s: ArrayLike, steps: int, /
    ) -> SoapFilmTunnelResult:
        """Advance ``steps`` in one stable compiled scan and stack their evidence."""
        count = positive_integer(steps, "steps")
        step_size = jnp.asarray(step_size_s, dtype=jnp.float64)
        return _run_soap_film_tunnel(self, state, step_size, count)

    @checked
    def strouhal(
        self,
        result: SoapFilmTunnelResult,
        /,
        *,
        transient_s: float,
        minimum_lift_coefficient: float = 1e-3,
    ) -> StrouhalEstimate:
        """Estimate the shedding Strouhal number from the recorded obstacle lift."""
        geometry = self.plan.geometry
        if geometry.obstacle_diameter_m is None:
            raise ValueError("An empty channel has no obstacle lift.")
        transient = float(transient_s)
        if not np.isfinite(transient) or transient < 0.0:
            raise ValueError("transient_s must be finite and nonnegative.")
        threshold = positive_finite_float(
            minimum_lift_coefficient, "minimum_lift_coefficient"
        )
        # Host boundary: the recorded series is post-processed once after the run.
        time = np.asarray(result.evidence.time_s)
        accepted = np.asarray(result.evidence.accepted)
        force = np.asarray(result.evidence.obstacle_force_n)
        inflow = self.plan.inflow
        speed = float(inflow.velocity_m_s)
        diameter = geometry.obstacle_diameter_m
        dynamic = 0.5 * float(self.plan.density_kg_m3 * inflow.thickness_m) * speed**2
        return _strouhal(
            time,
            accepted,
            force[:, 0] / (dynamic * diameter),
            force[:, 1] / (dynamic * diameter),
            transient=transient,
            threshold=threshold,
            diameter=diameter,
            speed=speed,
            reynolds=float(self.scales.reynolds_number),
        )

    def cylinder_wake_reference(
        self, mean_drag_coefficient: ArrayLike, /
    ) -> CylinderWakeReference:
        """Return the declared unconfined and blockage-corrected wake references.

        The wall correction is ``U'/U = 1 + C_D B/4 + 0.82 B^2`` with blockage
        ``B = d/W`` (Allen & Vincenti, NACA TR 782, 1944). The Williamson
        parallel-shedding fit is evaluated at ``Re U'/U`` and converted back to
        the inflow-speed Strouhal convention. Both Reynolds numbers must lie in
        the fit's declared interval ``47 < Re < 180``.
        """
        drag_host = _finite(mean_drag_coefficient, "mean_drag_coefficient")
        if drag_host <= 0.0:
            raise ValueError("mean_drag_coefficient must be positive.")
        drag = float(drag_host)
        reynolds = float(self.scales.reynolds_number)
        blockage = self.scales.blockage_ratio
        velocity_correction = 1.0 + 0.25 * drag * blockage + 0.82 * blockage**2
        corrected_reynolds = reynolds * velocity_correction
        if not 47.0 < reynolds < 180.0 or not 47.0 < corrected_reynolds < 180.0:
            raise ValueError(
                "The Williamson parallel-shedding fit requires both the declared "
                "and blockage-corrected Reynolds numbers to lie between 47 and 180."
            )

        def williamson(value: float, /) -> float:
            return -3.3265 / value + 0.1816 + 1.6e-4 * value

        return CylinderWakeReference(
            unconfined_strouhal_number=williamson(reynolds),
            blockage_corrected_strouhal_number=velocity_correction
            * williamson(corrected_reynolds),
            velocity_correction=velocity_correction,
            corrected_reynolds_number=corrected_reynolds,
            blockage_ratio=blockage,
        )

    def _evidence(
        self, film: SurfacePlugFlowStepResult, time: Array, step_size: Array, /
    ) -> SoapFilmTunnelEvidence:
        plan = self.flow.plan
        exchange = film.plug_flow.boundary
        state = film.candidate_state
        area = plan.surface.vertex_area

        def part_sum(mask: Array, values: Array, /) -> Array:
            shaped = mask.reshape(mask.shape + (1,) * (values.ndim - 1))
            return jnp.sum(jnp.where(shaped, values, 0.0), axis=0)

        thickness = state.liquid_volume_m3 / area
        concentration = state.surfactant_amount_mol / area
        elasticity = plan.law.evaluate(concentration).gibbs_elasticity_n_m
        wave_speed = jnp.sqrt(
            jnp.maximum(2.0 * elasticity / (plan.density_kg_m3 * thickness), 0.0)
        )
        speed = jnp.linalg.norm(self.flow.velocity(state), axis=1)
        regular_mach = jnp.where(
            wave_speed > 0.0,
            speed / jnp.where(wave_speed > 0.0, wave_speed, 1.0),
            jnp.where(speed > 0.0, jnp.inf, 0.0),
        )
        mach = jnp.where(
            jnp.isfinite(speed) & jnp.isfinite(wave_speed),
            regular_mach,
            speed + wave_speed,
        )
        evidence = film.plug_flow
        return SoapFilmTunnelEvidence(
            time_s=time,
            status=film.status,
            inflow_volume_rate_m3_s=-part_sum(self.inlet, exchange.volume_outflow_m3)
            / step_size,
            outflow_volume_rate_m3_s=part_sum(self.outlet, exchange.volume_outflow_m3)
            / step_size,
            inflow_surfactant_rate_mol_s=-part_sum(
                self.inlet, exchange.surfactant_outflow_mol
            )
            / step_size,
            outflow_surfactant_rate_mol_s=part_sum(
                self.outlet, exchange.surfactant_outflow_mol
            )
            / step_size,
            volume_residual_m3=film.evidence.liquid_volume_residual_m3,
            surfactant_residual_mol=evidence.surfactant_residual_mol,
            momentum_residual_n_s=evidence.momentum_change_n_s
            - evidence.external_impulse_n_s
            - evidence.boundary_impulse_n_s,
            obstacle_force_n=-part_sum(self.rim, exchange.constraint_force_n),
            wire_force_n=-part_sum(self.wires, exchange.constraint_force_n),
            film_mach_number=jnp.max(mach),
            maximum_speed_m_s=jnp.max(speed),
            minimum_thickness_m=jnp.min(thickness),
            courant_number=evidence.courant_number,
            marangoni_courant_number=evidence.marangoni_courant_number,
            terminal_nonlinear_stage=evidence.terminal_nonlinear_stage,
            nonlinear_status=film.evidence.nonlinear_status,
            nonlinear_iterations=film.evidence.nonlinear_iterations,
            nonlinear_residual_norm=film.evidence.nonlinear_residual_norm,
            converged=film.evidence.converged,
        )


@eqx.filter_jit
def _run_soap_film_tunnel(
    prepared: PreparedSoapFilmTunnel,
    state: SoapFilmTunnelState,
    step_size: Array,
    steps: int,
    /,
) -> SoapFilmTunnelResult:
    """Execute a fixed recurrence through one stable compiled scan.

    One numeric sparse-LU factor is refreshed at the initial state and reused
    only as the elastic block's frozen right preconditioner. Newton's Jacobian
    actions remain exact at every iterate. Insoluble surface diffusion is a
    fixed linear backward-Euler operator, so its exact numeric factor is also
    prepared once. Every bounded solve reports its status. Every evidence sample
    is public output, so checkpoint rematerialization would retain no less public
    data and is not this owner's execution contract.
    """
    elastic_factorization = prepared.flow._elastic_factorization(state.film, step_size)
    surfactant_factorization = (
        None
        if prepared.plan.kinetics is not None
        else prepared.flow._surfactant_factorization(state.film, step_size)
    )

    def advance(
        current: SoapFilmTunnelState, _: None
    ) -> tuple[SoapFilmTunnelState, SoapFilmTunnelEvidence]:
        result = prepared._step(
            current,
            step_size,
            elastic_factorization,
            surfactant_factorization,
        )
        return result.state, result.evidence

    final, evidence = jax.lax.scan(advance, state, None, length=steps, unroll=1)
    return SoapFilmTunnelResult(final, evidence)


def _boundary_kinds(
    channel: SoapFilmTunnelMesh, wire_kind: SoapFilmWireKind, /
) -> dict[PlugFlowBoundaryKind, np.ndarray]:
    wires = np.asarray(channel.wire_vertices)
    rim = np.asarray(channel.rim_vertices)
    kinds: dict[PlugFlowBoundaryKind, np.ndarray] = {
        "inflow": np.asarray(channel.inlet_vertices),
        "outflow": np.asarray(channel.outlet_vertices),
    }
    match wire_kind:
        case "no-slip":
            kinds["no-slip"] = np.concatenate((wires, rim))
        case "free-slip":
            kinds["free-slip"] = wires
            kinds["no-slip"] = rim
        case _:
            raise ValueError(f"Unknown wire kind {wire_kind!r}.")
    return kinds


def _mask(vertices: Array, count: int, /) -> Array:
    return jnp.zeros((count,), dtype=jnp.bool_).at[vertices].set(True)


def _strouhal(
    time: np.ndarray,
    accepted: np.ndarray,
    drag: np.ndarray,
    lift: np.ndarray,
    /,
    *,
    transient: float,
    threshold: float,
    diameter: float,
    speed: float,
    reynolds: float,
) -> StrouhalEstimate:
    """Count hysteretic upward zero crossings of the centered lift coefficient."""
    window = time >= transient
    span = float(np.ptp(time[window])) if np.count_nonzero(window) > 1 else 0.0
    mean_drag = float(np.mean(drag[window])) if span > 0.0 else np.nan

    def estimate(
        status: SheddingStatus,
        frequency: float,
        frequency_standard_error: float,
        periods: int,
        amplitude: float,
    ) -> StrouhalEstimate:
        return StrouhalEstimate(
            status=status,
            frequency_hz=frequency,
            strouhal_number=frequency * diameter / speed,
            frequency_standard_error_hz=frequency_standard_error,
            strouhal_standard_error=frequency_standard_error * diameter / speed,
            periods=periods,
            lift_coefficient_amplitude=amplitude,
            mean_drag_coefficient=mean_drag,
            reynolds_number=reynolds,
            window_s=span,
        )

    if not np.all(accepted):
        return estimate(SheddingStatus.REJECTED_STEPS, np.nan, np.nan, 0, np.nan)
    if np.count_nonzero(window) < 3:
        return estimate(SheddingStatus.INSUFFICIENT_RECORD, np.nan, np.nan, 0, np.nan)
    signal = lift[window] - np.mean(lift[window])
    times = time[window]
    amplitude = 0.5 * float(np.ptp(signal))
    if amplitude < threshold:
        return estimate(SheddingStatus.NO_SHEDDING, np.nan, np.nan, 0, amplitude)
    # Sample ``k`` completes an upward crossing between ``k - 1`` and ``k``; it
    # counts only if the lift fell below minus half the amplitude since the
    # previous upward crossing, so noise near zero adds no periods.
    upward = np.flatnonzero((signal[:-1] < 0.0) & (signal[1:] >= 0.0)) + 1
    positions = np.arange(signal.size)
    last_low = np.maximum.accumulate(np.where(signal < -0.5 * amplitude, positions, -1))
    previous = np.concatenate((np.zeros((1,), dtype=upward.dtype), upward[:-1]))
    upward = upward[last_low[upward - 1] >= previous]
    before, after = signal[upward - 1], signal[upward]
    crossings = times[upward - 1] + before / (before - after) * (
        times[upward] - times[upward - 1]
    )
    cycle_periods = np.diff(crossings)
    periods = cycle_periods.size
    if periods < 2:
        return estimate(
            SheddingStatus.INSUFFICIENT_RECORD,
            np.nan,
            np.nan,
            periods,
            amplitude,
        )
    mean_period = float(np.mean(cycle_periods))
    period_standard_error = float(np.std(cycle_periods, ddof=1)) / np.sqrt(periods)
    frequency = 1.0 / mean_period
    frequency_standard_error = period_standard_error / mean_period**2
    return estimate(
        SheddingStatus.SHEDDING,
        frequency,
        frequency_standard_error,
        periods,
        amplitude,
    )


def _finite(value: ArrayLike, name: str, /) -> np.ndarray:
    host = np.asarray(value, dtype=np.float64)
    if host.shape != () or not np.isfinite(host):
        raise ValueError(f"{name} must be a finite scalar.")
    return host


def _positive(value: ArrayLike, name: str, /) -> Array:
    host = _finite(value, name)
    if host <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return jnp.asarray(host)


def _nonnegative(value: ArrayLike, name: str, /) -> Array:
    host = _finite(value, name)
    if host < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return jnp.asarray(host)


__all__ = [
    "CylinderWakeReference",
    "PreparedSoapFilmTunnel",
    "SheddingStatus",
    "SoapFilmInflow",
    "SoapFilmTunnelEvidence",
    "SoapFilmTunnelPlan",
    "SoapFilmTunnelResult",
    "SoapFilmTunnelScales",
    "SoapFilmTunnelState",
    "SoapFilmTunnelStepResult",
    "SoapFilmWireKind",
    "StrouhalEstimate",
]
