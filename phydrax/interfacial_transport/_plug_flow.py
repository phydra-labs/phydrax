#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Plug-flow (extensional) dynamics of symmetric soap films with Marangoni forcing.

A symmetric film of thickness ``h`` moves with one tangential velocity ``u``
across its thickness. With shared surface concentration ``Gamma`` on both
interfaces the film obeys (Chomaz, J. Fluid Mech. 442, 2001; Howell, Eur. J.
Appl. Math. 7, 1996)

``rho h Du/Dt = 2 grad_s sigma(Gamma) + div_s(N_T + 2 tau_BS) + rho h P g
               - C_air (u - u_air)``,

``Dh/Dt = -h div_s u`` and ``DGamma/Dt = -Gamma div_s u + D_s lap_s Gamma + j``,

with the Trouton sheet stress ``N_T = 2 mu h (D_s + (div_s u) P)`` and the
Boussinesq--Scriven interfacial stress ``tau_BS`` from
``phydrax.rheology.boussinesq_scriven_stress`` doubled for two interfaces.
The linear Marangoni wave speed is ``c_M^2 = 2 E_s / (rho h)`` with the
single-interface Gibbs elasticity ``E_s``.

One first-order step composes ``phydrax.solver.ConservationIMEXMethod`` on the
extensive state ``(V, N, dissolved, P)`` with a forward--backward Euler tableau
whose implicit part is split block-sequentially:

1. explicit stage: conservative donor-cell transport of liquid volume,
   interfacial and dissolved amount and momentum with ``u^n`` (Courant number
   evidence);
2. implicit elastic/viscous part in ``(u', Gamma')``: the surfactant
   compression is corrected in flux form with ``Gamma_bar (phi(u') - phi(u^n))``,
   so the Marangoni wave carries no explicit stability limit; momentum is
   committed in force form ``P* + dt F(u', Gamma')``;
3. implicit surface diffusion/adsorption part through the symmetric-film
   surfactant route, started from the stage-2 state.

Each implicit stage commits a conservative update built from fluxes at its
nonlinear iterate, so liquid volume and surfactant are conserved to roundoff,
and momentum on a planar film changes only by external forces and boundary
exchange. The tableau is stiffly accurate and stage 3 starts from the stage-2
value exactly, so constrained components are committed without re-summing
stage rates. An optional ``PlugFlowBoundary`` opens inflow/outflow edges
(exact P1 half-edge area fluxes with donor-cell exterior or interior states in
the explicit transport and the implicit compression correction) and holds wall
and inflow velocities through projector rows of the implicit block; boundary
outflow, constraint forces and the boundary impulse reach the evidence.
"""

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field, parameter_field, ParameterOwner
from .._validation import positive_integer
from ..ein import contract
from ..linalg import PreparedSparseFactorization
from ..nonlinear import NonlinearStatus
from ..rheology import boussinesq_scriven_stress
from ..solver import ConservationIMEXMethod, ImplicitConservationStageResult
from ..solver.advanced import AdditiveIMEXTableau
from ..typing import ConvertibleToArray, parse
from ._core import AdsorptionKinetics, LangmuirSurfactantLaw
from ._film_contracts import PreparedFilmSurface
from ._film_evidence import FilmStepStatus, resolve_film_status, SurfaceFilmEvidence
from ._film_solve import film_termination, PreparedFilmNewton, vertex_block_pattern
from ._film_transport import (
    courant_limit,
    FilmTransportScheme,
    outflow_courant,
    transported_edge_flux,
)
from ._plug_flow_boundary import PlugFlowBoundary, PlugFlowBoundaryEvidence
from ._surfactant import (
    PreparedSymmetricFilmSurfactant,
    SurfactantTransportEvidence,
    SymmetricFilmSurfactantPlan,
    SymmetricFilmSurfactantState,
    vertex_gradient_force,
)


class PlugFlowNonlinearStage(IntEnum):
    """Implicit plug-flow stage owning the terminal nonlinear diagnostics."""

    ELASTIC = 0
    SURFACTANT = 1


class SurfacePlugFlowState(StrictModule):
    """Extensive plug-flow film state: volume, amounts and tangential momentum.

    ``surfactant_amount_mol`` is per interface; ``momentum_kg_m_s`` is the
    vertex-cell momentum ``rho V u`` tangent to the vertex normal.
    """

    liquid_volume_m3: Array
    surfactant_amount_mol: Array
    dissolved_amount_mol: Array | None
    momentum_kg_m_s: Array
    geometry_revision: Array
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        liquid_volume_m3: ArrayLike,
        surfactant_amount_mol: ArrayLike,
        dissolved_amount_mol: ArrayLike | None,
        momentum_kg_m_s: ArrayLike,
        /,
        *,
        topology_id: str,
        geometry_revision: ArrayLike = 0,
    ) -> None:
        volume = jnp.asarray(liquid_volume_m3, dtype=jnp.float64)
        amount = jnp.asarray(surfactant_amount_mol, dtype=jnp.float64)
        momentum = jnp.asarray(momentum_kg_m_s, dtype=jnp.float64)
        if volume.ndim != 1 or amount.shape != volume.shape:
            raise ValueError("Volume and surfactant amount must be vertex vectors.")
        if momentum.shape != (volume.shape[0], 3):
            raise ValueError("momentum_kg_m_s must have shape (num_vertices, 3).")
        dissolved = (
            None
            if dissolved_amount_mol is None
            else jnp.asarray(dissolved_amount_mol, dtype=jnp.float64)
        )
        if dissolved is not None and dissolved.shape != volume.shape:
            raise ValueError("dissolved_amount_mol must match the vertex vector.")
        if not isinstance(topology_id, str) or not topology_id:
            raise ValueError("topology_id must be a non-empty string.")
        self.liquid_volume_m3 = volume
        self.surfactant_amount_mol = amount
        self.dissolved_amount_mol = dissolved
        self.momentum_kg_m_s = momentum
        self.geometry_revision = jnp.asarray(geometry_revision, dtype=jnp.int32)
        self.topology_id = topology_id

    def total_surfactant_mol(self) -> Array:
        total = 2.0 * jnp.sum(self.surfactant_amount_mol)
        if self.dissolved_amount_mol is not None:
            total = total + jnp.sum(self.dissolved_amount_mol)
        return total


class PlugFlowEvidence(StrictModule):
    """Momentum, surfactant, dissipation and stability evidence of one step.

    ``momentum_change_n_s`` is the total momentum change,
    ``external_impulse_n_s`` the impulse of gravity and air drag and
    ``boundary_impulse_n_s`` the boundary exchange: momentum transported
    through open edges, the impulse of velocity constraints and the net
    boundary line tension (the net strong-form Marangoni force). On a planar
    film the change equals the sum of both impulses to roundoff because
    transport, viscous and interior elastic exchanges are internal.
    ``surfactant_residual_mol`` is the surfactant change plus the boundary
    outflow. ``viscous_dissipation_w`` and ``drag_dissipation_w`` are
    nonnegative by construction for nonnegative viscosities and drag.
    ``marangoni_courant_number`` is ``c_M dt / l_min``; the implicit elastic
    block does not require it to be small. ``terminal_nonlinear_stage`` names
    the stage whose status, iterations, residual, and convergence are reported
    by the enclosing :class:`SurfaceFilmEvidence`: the elastic stage when it
    fails, otherwise the terminal surfactant stage. Tangential-momentum
    residual and tolerance report admission of the public input state. The
    concentration, coverage, tension and support mask describe the surfactant
    candidate evaluated by the final implicit stage before the atomic commit.
    ``boundary`` holds the per-vertex boundary exchange (zero without a
    declared boundary).
    """

    momentum_change_n_s: Array
    external_impulse_n_s: Array
    boundary_impulse_n_s: Array
    surfactant_residual_mol: Array
    viscous_dissipation_w: Array
    drag_dissipation_w: Array
    kinetic_energy_change_j: Array
    surfactant_free_energy_change_j: Array
    courant_number: Array
    marangoni_courant_number: Array
    minimum_surface_concentration_mol_m2: Array
    maximum_coverage: Array
    minimum_surface_tension_n_m: Array
    surface_state_admissible: Array
    terminal_nonlinear_stage: Array
    tangential_momentum_residual_kg_m_s: Array
    tangential_momentum_tolerance_kg_m_s: Array
    tangential_momentum_admissible: Array
    boundary: PlugFlowBoundaryEvidence

    def __init__(
        self,
        *,
        momentum_change_n_s: Array,
        external_impulse_n_s: Array,
        boundary_impulse_n_s: Array,
        surfactant_residual_mol: Array,
        viscous_dissipation_w: Array,
        drag_dissipation_w: Array,
        kinetic_energy_change_j: Array,
        surfactant_free_energy_change_j: Array,
        courant_number: Array,
        marangoni_courant_number: Array,
        minimum_surface_concentration_mol_m2: Array,
        maximum_coverage: Array,
        minimum_surface_tension_n_m: Array,
        surface_state_admissible: Array,
        terminal_nonlinear_stage: Array,
        tangential_momentum_residual_kg_m_s: Array,
        tangential_momentum_tolerance_kg_m_s: Array,
        tangential_momentum_admissible: Array,
        boundary: PlugFlowBoundaryEvidence,
    ) -> None:
        if not isinstance(boundary, PlugFlowBoundaryEvidence):
            raise TypeError("boundary must be PlugFlowBoundaryEvidence.")
        self.momentum_change_n_s = jnp.asarray(momentum_change_n_s)
        self.external_impulse_n_s = jnp.asarray(external_impulse_n_s)
        self.boundary_impulse_n_s = jnp.asarray(boundary_impulse_n_s)
        self.surfactant_residual_mol = jnp.asarray(surfactant_residual_mol)
        self.viscous_dissipation_w = jnp.asarray(viscous_dissipation_w)
        self.drag_dissipation_w = jnp.asarray(drag_dissipation_w)
        self.kinetic_energy_change_j = jnp.asarray(kinetic_energy_change_j)
        self.surfactant_free_energy_change_j = jnp.asarray(
            surfactant_free_energy_change_j
        )
        self.courant_number = jnp.asarray(courant_number)
        self.marangoni_courant_number = jnp.asarray(marangoni_courant_number)
        self.minimum_surface_concentration_mol_m2 = jnp.asarray(
            minimum_surface_concentration_mol_m2
        )
        self.maximum_coverage = jnp.asarray(maximum_coverage)
        self.minimum_surface_tension_n_m = jnp.asarray(minimum_surface_tension_n_m)
        self.surface_state_admissible = jnp.asarray(
            surface_state_admissible, dtype=jnp.bool_
        )
        self.terminal_nonlinear_stage = jnp.asarray(
            terminal_nonlinear_stage, dtype=jnp.int32
        )
        self.tangential_momentum_residual_kg_m_s = jnp.asarray(
            tangential_momentum_residual_kg_m_s
        )
        self.tangential_momentum_tolerance_kg_m_s = jnp.asarray(
            tangential_momentum_tolerance_kg_m_s
        )
        self.tangential_momentum_admissible = jnp.asarray(
            tangential_momentum_admissible, dtype=jnp.bool_
        )
        self.boundary = boundary


class SurfacePlugFlowStepResult(StrictModule):
    state: SurfacePlugFlowState
    candidate_state: SurfacePlugFlowState
    status: Array
    evidence: SurfaceFilmEvidence
    plug_flow: PlugFlowEvidence

    def __init__(
        self,
        state: SurfacePlugFlowState,
        candidate_state: SurfacePlugFlowState,
        status: Array,
        evidence: SurfaceFilmEvidence,
        plug_flow: PlugFlowEvidence,
        /,
    ) -> None:
        self.state = state
        self.candidate_state = candidate_state
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.evidence = evidence
        self.plug_flow = plug_flow

    @property
    def accepted(self) -> Array:
        return self.status == FilmStepStatus.ACCEPTED


class SurfacePlugFlowPlan(StrictModule, ParameterOwner):
    """Symmetric soap-film plug-flow model on one prepared film surface.

    ``viscosity_pa_s`` is the bulk liquid viscosity of the Trouton sheet
    stress; ``surface_shear_viscosity_n_s_m`` and
    ``surface_dilatational_viscosity_n_s_m`` are single-interface
    Boussinesq--Scriven viscosities; ``air_drag_coefficient_kg_m2_s`` is the
    total linear drag per film area toward ``air_velocity_m_s``. ``boundary``
    declares open and wall edges of a bordered film; without it every
    boundary edge is ``no-flux`` with traction-free velocity. A soluble film
    requires the exterior dissolved inflow concentration on its boundary.
    ``transport_scheme`` selects the interior edge reconstruction of the
    explicit transport (``FilmTransportScheme``); ``"limited-muscl"`` removes
    the donor-cell numerical viscosity ``|u| dx / 2`` that damps resolved
    vortical flow, admits outflow Courant numbers up to one half and does not
    guarantee positivity.
    """

    surface: PreparedFilmSurface = fixed_field()
    law: LangmuirSurfactantLaw
    kinetics: AdsorptionKinetics | None
    density_kg_m3: Array = parameter_field()
    viscosity_pa_s: Array = parameter_field()
    surface_shear_viscosity_n_s_m: Array = parameter_field()
    surface_dilatational_viscosity_n_s_m: Array = parameter_field()
    surface_diffusivity_m2_s: Array = parameter_field()
    air_drag_coefficient_kg_m2_s: Array = parameter_field()
    air_velocity_m_s: Array = parameter_field()
    gravity_m_s2: Array = parameter_field()
    boundary: PlugFlowBoundary | None
    rupture_thickness_m: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    transport_scheme: FilmTransportScheme = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: PreparedFilmSurface,
        law: LangmuirSurfactantLaw,
        /,
        *,
        density_kg_m3: ArrayLike,
        viscosity_pa_s: ArrayLike = 0.0,
        surface_shear_viscosity_n_s_m: ArrayLike = 0.0,
        surface_dilatational_viscosity_n_s_m: ArrayLike = 0.0,
        surface_diffusivity_m2_s: ArrayLike = 0.0,
        air_drag_coefficient_kg_m2_s: ArrayLike = 0.0,
        air_velocity_m_s: ConvertibleToArray = (0.0, 0.0, 0.0),
        gravity_m_s2: ConvertibleToArray = (0.0, 0.0, 0.0),
        kinetics: AdsorptionKinetics | None = None,
        boundary: PlugFlowBoundary | None = None,
        rupture_thickness_m: float = 0.0,
        tolerance: float = 1e-10,
        maximum_iterations: int = 30,
        transport_scheme: FilmTransportScheme = "donor-cell",
    ) -> None:
        if not isinstance(surface, PreparedFilmSurface):
            raise TypeError("surface must be a PreparedFilmSurface.")
        if not isinstance(law, LangmuirSurfactantLaw):
            raise TypeError("law must be a LangmuirSurfactantLaw.")
        if kinetics is not None and not isinstance(kinetics, AdsorptionKinetics):
            raise TypeError("kinetics must be AdsorptionKinetics or None.")
        density = _scalar(density_kg_m3, "density_kg_m3", positive=True)
        scalars = {
            name: _scalar(value, name, positive=False)
            for name, value in (
                ("viscosity_pa_s", viscosity_pa_s),
                ("surface_shear_viscosity_n_s_m", surface_shear_viscosity_n_s_m),
                (
                    "surface_dilatational_viscosity_n_s_m",
                    surface_dilatational_viscosity_n_s_m,
                ),
                ("surface_diffusivity_m2_s", surface_diffusivity_m2_s),
                ("air_drag_coefficient_kg_m2_s", air_drag_coefficient_kg_m2_s),
            )
        }
        vectors = {
            name: _vector(value, name)
            for name, value in (
                ("air_velocity_m_s", air_velocity_m_s),
                ("gravity_m_s2", gravity_m_s2),
            )
        }
        rupture = float(rupture_thickness_m)
        if not np.isfinite(rupture) or rupture < 0.0:
            raise ValueError("rupture_thickness_m must be finite and nonnegative.")
        scheme = parse(transport_scheme, FilmTransportScheme, "transport_scheme")
        if boundary is not None:
            _check_boundary(surface, law, kinetics, boundary)
        self.surface = surface
        self.law = law
        self.kinetics = kinetics
        self.density_kg_m3 = density
        self.viscosity_pa_s = scalars["viscosity_pa_s"]
        self.surface_shear_viscosity_n_s_m = scalars["surface_shear_viscosity_n_s_m"]
        self.surface_dilatational_viscosity_n_s_m = scalars[
            "surface_dilatational_viscosity_n_s_m"
        ]
        self.surface_diffusivity_m2_s = scalars["surface_diffusivity_m2_s"]
        self.air_drag_coefficient_kg_m2_s = scalars["air_drag_coefficient_kg_m2_s"]
        self.air_velocity_m_s = vectors["air_velocity_m_s"]
        self.gravity_m_s2 = vectors["gravity_m_s2"]
        self.boundary = boundary
        self.rupture_thickness_m = rupture
        self.tolerance = float(tolerance)
        self.maximum_iterations = positive_integer(
            maximum_iterations, "maximum_iterations"
        )
        self.transport_scheme = scheme
        self.plan_id = canonical_fingerprint(
            {
                "kind": "surface-plug-flow",
                "operator_id": surface.topology.operator_id,
                "soluble": kinetics is not None,
                "boundary": "no-flux" if boundary is None else boundary.boundary_id,
                "time_integration": "conservation-imex/transport-elastic-exchange",
                "transport_scheme": scheme,
                "stress": "trouton-sheet-plus-two-boussinesq-scriven",
                "tolerance": self.tolerance,
                "maximum_iterations": self.maximum_iterations,
            }
        )

    def prepare(self) -> PreparedSurfacePlugFlow:
        return PreparedSurfacePlugFlow(self)

    def tangent_frame(self, surface: PreparedFilmSurface, /) -> tuple[Array, Array]:
        """Return a deterministic orthonormal tangent frame at every vertex."""
        normal = surface.vertex_normal
        axis = jnp.eye(3, dtype=normal.dtype)[jnp.argmin(jnp.abs(normal), axis=1)]
        first = jnp.cross(normal, axis)
        first = first / jnp.linalg.norm(first, axis=1, keepdims=True)
        return first, jnp.cross(normal, first)

    def membrane_stress(
        self, surface: PreparedFilmSurface, thickness: Array, velocity: Array, /
    ) -> tuple[Array, Array]:
        """Return face stress resultants ``N_f`` (N/m) and rates ``D_f`` (1/s)."""
        operators = surface.operators
        normal = operators.face_normal
        projector = (
            jnp.eye(3, dtype=normal.dtype) - normal[:, :, None] * normal[:, None, :]
        )
        gradient = jnp.swapaxes(operators.gradient(velocity), 1, 2)
        tangential = projector @ gradient @ projector
        rate = 0.5 * (tangential + jnp.swapaxes(tangential, 1, 2))
        divergence = jnp.trace(rate, axis1=1, axis2=2)
        face_thickness = jnp.mean(thickness[operators.faces], axis=1)
        trouton = (
            2.0
            * self.viscosity_pa_s
            * face_thickness[:, None, None]
            * (rate + divergence[:, None, None] * projector)
        )
        interfacial = 2.0 * boussinesq_scriven_stress(
            rate,
            divergence,
            self.surface_shear_viscosity_n_s_m,
            self.surface_dilatational_viscosity_n_s_m,
            projector,
        )
        return trouton + interfacial, rate

    def forces(
        self,
        surface: PreparedFilmSurface,
        volume: Array,
        concentration: Array,
        velocity: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        """Return total vertex force (N), viscous and drag dissipation (W)."""
        operators = surface.operators
        tension = self.law.evaluate(concentration).surface_tension_n_m
        marangoni = 2.0 * vertex_gradient_force(surface, tension)
        stress, rate = self.membrane_stress(
            surface, volume / surface.vertex_area, velocity
        )
        corner = -operators.face_area[:, None, None] * contract(
            "fab,fkb->fka", stress, operators.basis_gradients
        )
        viscous = (
            jnp.zeros_like(velocity)
            .at[operators.faces.reshape((-1,))]
            .add(corner.reshape((-1, 3)))
        )
        relative = velocity - self.air_velocity_m_s
        dissipation = jnp.sum(operators.face_area * jnp.sum(stress * rate, axis=(1, 2)))
        drag_dissipation = self.air_drag_coefficient_kg_m2_s * jnp.sum(
            surface.vertex_area * jnp.sum(relative**2, axis=1)
        )
        external = self.external_force(surface, volume, velocity)
        return marangoni + viscous + external, dissipation, drag_dissipation

    def external_force(
        self, surface: PreparedFilmSurface, volume: Array, velocity: Array, /
    ) -> Array:
        """Return gravity plus linear air drag per vertex cell (N)."""
        relative = velocity - self.air_velocity_m_s
        drag = (
            -self.air_drag_coefficient_kg_m2_s * surface.vertex_area[:, None] * relative
        )
        return drag + self.density_kg_m3 * volume[:, None] * self.gravity_m_s2


class PreparedSurfacePlugFlow(StrictModule):
    """Plan with its IMEX method and prepared elastic/viscous and surfactant solvers."""

    plan: SurfacePlugFlowPlan
    solver: PreparedFilmNewton = fixed_field()
    surfactant: PreparedSymmetricFilmSurfactant = fixed_field()
    method: ConservationIMEXMethod = fixed_field()
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: SurfacePlugFlowPlan, /) -> None:
        if not isinstance(plan, SurfacePlugFlowPlan):
            raise TypeError("plan must be a SurfacePlugFlowPlan.")
        surface = plan.surface
        count = surface.topology.num_vertices
        capacity = plan.law.maximum_surface_concentration_mol_m2
        projector, prescribed = _constraint(plan)
        boundary_sample = (
            None
            if plan.boundary is None
            else jnp.full(plan.boundary.edges.shape, 0.1 * capacity)
        )
        sample = _ElasticArguments(
            plan,
            surface.vertex_area,
            0.1 * capacity * surface.vertex_area,
            jnp.zeros((count, 3)),
            jnp.zeros((count, 3)),
            jnp.full((surface.topology.num_edges,), 0.1 * capacity),
            boundary_sample,
            None if boundary_sample is None else jnp.zeros_like(boundary_sample),
            projector,
            prescribed,
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            jnp.asarray(1.0),
        )
        self.plan = plan
        self.solver = PreparedFilmNewton(
            _elastic_residual,
            vertex_block_pattern(np.asarray(surface.topology.edges), count, 3),
            jnp.concatenate(
                (jnp.zeros((2 * count,)), jnp.full((count,), 0.1 * capacity))
            ),
            sample,
            termination=film_termination(
                tolerance=plan.tolerance, maximum_iterations=plan.maximum_iterations
            ),
            solver_id=f"surface-plug-flow/{plan.plan_id}",
        )
        self.surfactant = SymmetricFilmSurfactantPlan(
            surface,
            plan.law,
            surface_diffusivity_m2_s=plan.surface_diffusivity_m2_s,
            kinetics=plan.kinetics,
            maximum_iterations=plan.maximum_iterations,
        ).prepare()
        self.method = _imex_method()
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-surface-plug-flow",
                "plan_id": plan.plan_id,
                "method_id": self.method.method_id,
            }
        )

    def initial_state(
        self,
        thickness_m: ArrayLike,
        surface_concentration_mol_m2: ArrayLike,
        velocity_m_s: ConvertibleToArray = (0.0, 0.0, 0.0),
        dissolved_concentration_mol_m3: ArrayLike | None = None,
        /,
    ) -> SurfacePlugFlowState:
        """Return extensive state with tangential velocity meeting the boundary constraints."""
        surfactant = self.surfactant.initial_state(
            thickness_m, surface_concentration_mol_m2, dissolved_concentration_mol_m3
        )
        plan = self.plan
        surface = plan.surface
        velocity = jnp.broadcast_to(
            jnp.asarray(velocity_m_s, dtype=jnp.float64),
            (surface.topology.num_vertices, 3),
        )
        mass = plan.density_kg_m3 * surfactant.liquid_volume_m3[:, None]
        projector, prescribed = _constraint(plan)
        momentum = _hold(
            plan, projector, mass * _tangential(surface, velocity), mass * prescribed
        )
        return SurfacePlugFlowState(
            surfactant.liquid_volume_m3,
            surfactant.surfactant_amount_mol,
            surfactant.dissolved_amount_mol,
            momentum,
            topology_id=surfactant.topology_id,
            geometry_revision=surface.geometry_revision,
        )

    def velocity(self, state: SurfacePlugFlowState, /) -> Array:
        return state.momentum_kg_m_s / (
            self.plan.density_kg_m3 * state.liquid_volume_m3[:, None]
        )

    def _elastic_factorization(
        self, state: SurfacePlugFlowState, step_size_s: ArrayLike, /
    ) -> PreparedSparseFactorization:
        """Prepare one elastic sparse-LU factor for a fixed-state recurrence."""
        plan = self.plan
        surface = plan.surface
        _check_state(surface, state, soluble=plan.kinetics is not None)
        step_size = jnp.asarray(step_size_s, dtype=jnp.float64)
        velocity = self.velocity(state)
        moved = _transport(
            plan,
            state.liquid_volume_m3,
            state.surfactant_amount_mol,
            state.dissolved_amount_mol,
            state.momentum_kg_m_s,
            velocity,
        )
        arguments = _StepArguments(self, velocity, moved.boundary_flux, None, None)
        packed = _pack(
            state.liquid_volume_m3,
            state.surfactant_amount_mol,
            state.dissolved_amount_mol,
            state.momentum_kg_m_s,
        )
        provisional = packed + step_size * moved.rate
        elastic, _ = _elastic_setup(arguments, provisional, step_size)
        return self.solver.factorize(
            _elastic_initial(plan, provisional, velocity), elastic
        )

    def _surfactant_factorization(
        self, state: SurfacePlugFlowState, step_size_s: ArrayLike, /
    ) -> PreparedSparseFactorization:
        """Prepare the fixed insoluble surface-diffusion factor."""
        plan = self.plan
        _check_state(plan.surface, state, soluble=plan.kinetics is not None)
        return self.surfactant._factorization(
            _surfactant_state(
                plan,
                state.liquid_volume_m3,
                state.surfactant_amount_mol,
                state.dissolved_amount_mol,
            ),
            step_size_s,
        )

    def step(
        self, state: SurfacePlugFlowState, step_size_s: ArrayLike, /
    ) -> SurfacePlugFlowStepResult:
        """Advance one step of the IMEX method; any failed stage rejects the candidate."""
        return self._step(state, step_size_s, None, None)

    def _step(
        self,
        state: SurfacePlugFlowState,
        step_size_s: ArrayLike,
        elastic_factorization: PreparedSparseFactorization | None,
        surfactant_factorization: PreparedSparseFactorization | None,
        /,
    ) -> SurfacePlugFlowStepResult:
        """Advance one step of the IMEX method; any failed stage rejects the candidate."""
        plan = self.plan
        surface = plan.surface
        _check_state(surface, state, soluble=plan.kinetics is not None)
        step_size = jnp.asarray(step_size_s, dtype=jnp.float64)
        area = surface.vertex_area
        velocity = self.velocity(state)
        normal_momentum = jnp.sum(
            state.momentum_kg_m_s * surface.vertex_normal, axis=1
        )
        tangential_momentum_residual = jnp.max(jnp.abs(normal_momentum))
        momentum_scale = jnp.max(jnp.linalg.norm(state.momentum_kg_m_s, axis=1))
        tangential_momentum_tolerance = (
            64.0 * jnp.finfo(state.momentum_kg_m_s.dtype).eps * momentum_scale
        )
        tangential_momentum_admissible = (
            jnp.isfinite(tangential_momentum_residual)
            & (tangential_momentum_residual <= tangential_momentum_tolerance)
        )
        valid_input = (
            jnp.all(jnp.isfinite(state.liquid_volume_m3) & (state.liquid_volume_m3 > 0.0))
            & jnp.all(jnp.isfinite(state.momentum_kg_m_s))
            & tangential_momentum_admissible
            & jnp.isfinite(step_size)
            & (step_size > 0.0)
        )
        # Transport rates at y^n: the explicit stage rate and the outflow ledger.
        moved = _transport(
            plan,
            state.liquid_volume_m3,
            state.surfactant_amount_mol,
            state.dissolved_amount_mol,
            state.momentum_kg_m_s,
            velocity,
        )
        courant = outflow_courant(
            surface,
            moved.area_flux,
            area,
            step_size,
            boundary_outflow=None
            if plan.boundary is None or moved.boundary_flux is None
            else plan.boundary.outflow_rate(surface, moved.boundary_flux),
        )
        imex = self.method.step(
            jnp.zeros((), dtype=jnp.float64),
            _pack(
                state.liquid_volume_m3,
                state.surfactant_amount_mol,
                state.dissolved_amount_mol,
                state.momentum_kg_m_s,
            ),
            step_size,
            _StepArguments(
                self,
                velocity,
                moved.boundary_flux,
                elastic_factorization,
                surfactant_factorization,
            ),
        )
        volume, candidate_amount, candidate_dissolved, new_momentum = _unpack(
            plan, imex.candidate_state
        )
        elastic = imex.stage_evidence[_ELASTIC_STAGE]
        elastic_status = imex.stage_status[_ELASTIC_STAGE]
        exchange_evidence = imex.stage_evidence[_EXCHANGE_STAGE]
        if not isinstance(exchange_evidence, SurfactantTransportEvidence):
            raise TypeError("The surfactant stage returned invalid evidence.")
        candidate = SurfacePlugFlowState(
            volume,
            candidate_amount,
            candidate_dissolved,
            new_momentum,
            topology_id=state.topology_id,
            geometry_revision=state.geometry_revision,
        )
        safe_volume = jnp.where(valid_input & (volume > 0.0), volume, area)
        elastic_converged = elastic_status == NonlinearStatus.SUCCESS
        elastic_successful = imex.stage_successful[_ELASTIC_STAGE]
        terminal_exchange = elastic_successful
        terminal_nonlinear_stage = jnp.where(
            terminal_exchange,
            int(PlugFlowNonlinearStage.SURFACTANT),
            int(PlugFlowNonlinearStage.ELASTIC),
        )
        terminal_nonlinear_status = jnp.where(
            terminal_exchange, exchange_evidence.nonlinear_status, elastic_status
        )
        terminal_nonlinear_iterations = jnp.where(
            terminal_exchange,
            exchange_evidence.nonlinear_iterations,
            imex.stage_iterations[_ELASTIC_STAGE],
        )
        terminal_nonlinear_residual = jnp.where(
            terminal_exchange,
            exchange_evidence.nonlinear_residual_norm,
            imex.stage_residual_norms[_ELASTIC_STAGE],
        )
        terminal_converged = terminal_nonlinear_status == NonlinearStatus.SUCCESS
        finite = (
            jnp.all(jnp.isfinite(volume))
            & jnp.all(jnp.isfinite(new_momentum))
            & jnp.all(jnp.isfinite(candidate_amount))
            & (
                jnp.asarray(True)
                if candidate_dissolved is None
                else jnp.all(jnp.isfinite(candidate_dissolved))
            )
            & exchange_evidence.finite
        )
        positive = jnp.all(volume > 0.0) & (elastic.minimum_compressed_amount_mol >= 0.0)
        conductance = surface.evidence.admissible
        status = resolve_film_status(
            (FilmStepStatus.INADMISSIBLE_INPUT, ~valid_input),
            (FilmStepStatus.INADMISSIBLE_CONDUCTANCE, ~conductance),
            (
                FilmStepStatus.COURANT_LIMIT,
                courant > courant_limit(plan.transport_scheme),
            ),
            (FilmStepStatus.SOLVE_FAILED, ~elastic_successful),
            (FilmStepStatus.NONFINITE, ~finite),
            (FilmStepStatus.POSITIVITY_VIOLATED, ~positive),
        )
        status = jnp.where(
            status == FilmStepStatus.ACCEPTED,
            imex.stage_status[_EXCHANGE_STAGE],
            status,
        )
        accepted = status == FilmStepStatus.ACCEPTED
        kinetic = 0.5 * jnp.sum(jnp.sum(state.momentum_kg_m_s * velocity, axis=1))
        new_kinetic = 0.5 * jnp.sum(
            jnp.sum(new_momentum**2, axis=1) / (plan.density_kg_m3 * safe_volume)
        )
        new_free_energy = 2.0 * jnp.sum(
            area * plan.law.free_energy_density(candidate_amount / area)
        )
        old_free_energy = 2.0 * jnp.sum(
            area * plan.law.free_energy_density(state.surfactant_amount_mol / area)
        )
        new_thickness = volume / area
        tension = 2.0 * plan.law.evaluate(elastic.concentration).surface_tension_n_m
        boundary = PlugFlowBoundaryEvidence(
            volume_outflow_m3=step_size * moved.volume_outflow,
            surfactant_outflow_mol=2.0
            * step_size
            * (moved.amount_outflow + elastic.boundary_compression)
            + step_size * moved.dissolved_outflow,
            momentum_outflow_n_s=step_size * moved.momentum_outflow,
            constraint_force_n=(new_momentum - elastic.free_momentum) / step_size
            + _constrained_line_tension(plan, tension),
        )
        exchange = -jnp.sum(boundary.volume_outflow_m3)
        evidence = SurfaceFilmEvidence(
            liquid_volume_residual_m3=jnp.sum(volume)
            - jnp.sum(state.liquid_volume_m3)
            - exchange,
            boundary_exchange_m3=exchange,
            minimum_thickness_m=jnp.min(new_thickness),
            rupture_mask=new_thickness < plan.rupture_thickness_m,
            energy_change_j=new_kinetic + new_free_energy - kinetic - old_free_energy,
            dissipation_guaranteed=jnp.asarray(False),
            positivity_guaranteed=conductance
            & (plan.transport_scheme == "donor-cell")
            & (courant <= 1.0)
            & elastic_converged
            & positive
            & finite,
            conductance_admissible=conductance,
            nonlinear_status=terminal_nonlinear_status,
            nonlinear_iterations=terminal_nonlinear_iterations,
            nonlinear_residual_norm=terminal_nonlinear_residual,
            converged=terminal_converged,
            finite=finite,
            geometry_revision=surface.geometry_revision,
        )
        external = plan.external_force(surface, safe_volume, elastic.velocity)
        net_tension = jnp.sum(
            _tangential(surface, vertex_gradient_force(surface, tension)), axis=0
        )
        edges = surface.topology.edges
        plug_flow = PlugFlowEvidence(
            momentum_change_n_s=jnp.sum(new_momentum, axis=0)
            - jnp.sum(state.momentum_kg_m_s, axis=0),
            external_impulse_n_s=step_size
            * jnp.sum(_tangential(surface, external), axis=0),
            boundary_impulse_n_s=jnp.sum(new_momentum - elastic.free_momentum, axis=0)
            + step_size * net_tension
            - jnp.sum(boundary.momentum_outflow_n_s, axis=0),
            surfactant_residual_mol=candidate.total_surfactant_mol()
            - state.total_surfactant_mol()
            + jnp.sum(boundary.surfactant_outflow_mol),
            viscous_dissipation_w=elastic.viscous_dissipation_w,
            drag_dissipation_w=elastic.drag_dissipation_w,
            kinetic_energy_change_j=new_kinetic - kinetic,
            surfactant_free_energy_change_j=new_free_energy - old_free_energy,
            courant_number=courant,
            marangoni_courant_number=elastic.marangoni_speed_m_s
            * step_size
            / jnp.min(
                jnp.linalg.norm(
                    surface.coordinates[edges[:, 1]] - surface.coordinates[edges[:, 0]],
                    axis=1,
                )
            ),
            minimum_surface_concentration_mol_m2=(
                exchange_evidence.minimum_surface_concentration_mol_m2
            ),
            maximum_coverage=exchange_evidence.maximum_coverage,
            minimum_surface_tension_n_m=exchange_evidence.minimum_surface_tension_n_m,
            surface_state_admissible=exchange_evidence.surface_state_admissible,
            terminal_nonlinear_stage=terminal_nonlinear_stage,
            tangential_momentum_residual_kg_m_s=tangential_momentum_residual,
            tangential_momentum_tolerance_kg_m_s=tangential_momentum_tolerance,
            tangential_momentum_admissible=tangential_momentum_admissible,
            boundary=boundary,
        )
        accepted_state = SurfacePlugFlowState(
            jnp.where(accepted, candidate.liquid_volume_m3, state.liquid_volume_m3),
            jnp.where(
                accepted, candidate.surfactant_amount_mol, state.surfactant_amount_mol
            ),
            None
            if state.dissolved_amount_mol is None
            or candidate.dissolved_amount_mol is None
            else jnp.where(
                accepted, candidate.dissolved_amount_mol, state.dissolved_amount_mol
            ),
            jnp.where(accepted, candidate.momentum_kg_m_s, state.momentum_kg_m_s),
            topology_id=state.topology_id,
            geometry_revision=state.geometry_revision,
        )
        return SurfacePlugFlowStepResult(
            accepted_state, candidate, status, evidence, plug_flow
        )


# Stage indices of the plug-flow IMEX tableau.
_ELASTIC_STAGE = 1
_EXCHANGE_STAGE = 2


def _imex_method() -> ConservationIMEXMethod:
    """Return forward--backward Euler with block-sequential implicit parts.

    Stage 1 is explicit at ``y^n``; stage 2 solves the elastic/viscous part
    from the transported state; stage 3 solves diffusion/exchange from the
    stage-2 value. The tableau is stiffly accurate, so the step result is the
    stage-3 value.
    """
    tableau = AdditiveIMEXTableau(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0))),
        np.asarray((0.0, 1.0, 1.0)),
        np.asarray((0.0, 1.0, 1.0)),
        explicit_weights=np.asarray((1.0, 0.0, 0.0)),
        implicit_parts=(None, 0, 1),
    )
    return ConservationIMEXMethod(
        tableau,
        _explicit_rate,
        (_elastic_rate, _exchange_rate),
        (_elastic_solve, _exchange_solve),
        method_id="surface-plug-flow/transport-elastic-exchange",
    )


def _pack(
    volume: Array, amount: Array, dissolved: Array | None, momentum: Array, /
) -> Array:
    """Return the IMEX state ``(V, N, dissolved, P)`` as one flat vector."""
    blocks = (volume, amount) + (() if dissolved is None else (dissolved,))
    return jnp.concatenate(blocks + (momentum.reshape((-1,)),))


def _unpack(
    plan: SurfacePlugFlowPlan, values: Array, /
) -> tuple[Array, Array, Array | None, Array]:
    count = plan.surface.topology.num_vertices
    soluble = plan.kinetics is not None
    offset = (3 if soluble else 2) * count
    return (
        values[:count],
        values[count : 2 * count],
        values[2 * count : offset] if soluble else None,
        values[offset:].reshape((count, 3)),
    )


class _StepArguments(StrictModule):
    """Per-step IMEX arguments, including optional reusable implicit factors."""

    prepared: PreparedSurfacePlugFlow
    velocity: Array
    boundary_flux: Array | None
    elastic_factorization: PreparedSparseFactorization | None
    surfactant_factorization: PreparedSparseFactorization | None

    def __init__(
        self,
        prepared: PreparedSurfacePlugFlow,
        velocity: Array,
        boundary_flux: Array | None,
        elastic_factorization: PreparedSparseFactorization | None,
        surfactant_factorization: PreparedSparseFactorization | None,
        /,
    ) -> None:
        self.prepared = prepared
        self.velocity = velocity
        self.boundary_flux = boundary_flux
        self.elastic_factorization = elastic_factorization
        self.surfactant_factorization = surfactant_factorization


class _ElasticEvidence(StrictModule):
    """Elastic/viscous stage iterate, forces and dissipation for the step evidence."""

    velocity: Array
    concentration: Array
    free_momentum: Array
    boundary_compression: Array
    minimum_compressed_amount_mol: Array
    viscous_dissipation_w: Array
    drag_dissipation_w: Array
    marangoni_speed_m_s: Array

    def __init__(
        self,
        *,
        velocity: Array,
        concentration: Array,
        free_momentum: Array,
        boundary_compression: Array,
        minimum_compressed_amount_mol: Array,
        viscous_dissipation_w: Array,
        drag_dissipation_w: Array,
        marangoni_speed_m_s: Array,
    ) -> None:
        self.velocity = velocity
        self.concentration = concentration
        self.free_momentum = free_momentum
        self.boundary_compression = boundary_compression
        self.minimum_compressed_amount_mol = minimum_compressed_amount_mol
        self.viscous_dissipation_w = viscous_dissipation_w
        self.drag_dissipation_w = drag_dissipation_w
        self.marangoni_speed_m_s = marangoni_speed_m_s


def _explicit_rate(time: Array, state: Array, args: _StepArguments) -> Array:
    """Return the donor-cell transport rate of ``state`` with its own velocity."""
    del time
    plan = args.prepared.plan
    volume, amount, dissolved, momentum = _unpack(plan, state)
    velocity = momentum / (plan.density_kg_m3 * volume[:, None])
    return _transport(plan, volume, amount, dissolved, momentum, velocity).rate


def _elastic_setup(
    args: _StepArguments, state: Array, step_size: Array, /
) -> tuple[_ElasticArguments, Array]:
    """Return the elastic block arguments at ``state`` and the Marangoni speed.

    Edge and open-boundary concentrations of the compression correction are
    frozen at ``state``; the reference velocity is ``u^n``.
    """
    plan = args.prepared.plan
    surface = plan.surface
    area = surface.vertex_area
    volume, amount, _, momentum = _unpack(plan, state)
    safe_volume = jnp.where(jnp.isfinite(volume) & (volume > 0.0), volume, area)
    concentration = amount / area
    edges = surface.topology.edges
    edge_concentration = 0.5 * (concentration[edges[:, 0]] + concentration[edges[:, 1]])
    boundary_concentration = (
        None
        if plan.boundary is None or args.boundary_flux is None
        else plan.boundary.donor_density(
            surface,
            amount,
            plan.boundary.inflow_surface_concentration_mol_m2,
            args.boundary_flux,
        )
    )
    projector, prescribed = _constraint(plan)
    elasticity = plan.law.evaluate(concentration).gibbs_elasticity_n_m
    marangoni_speed = jnp.sqrt(
        jnp.maximum(
            2.0
            * jnp.max(elasticity)
            / (plan.density_kg_m3 * jnp.min(safe_volume / area)),
            0.0,
        )
    )
    speed_scale = marangoni_speed + jnp.max(jnp.linalg.norm(args.velocity, axis=1))
    speed_scale = speed_scale + jnp.linalg.norm(plan.gravity_m_s2) * step_size
    speed_scale = jnp.where(speed_scale > 0.0, speed_scale, 1.0)
    elastic = _ElasticArguments(
        plan,
        safe_volume,
        amount,
        momentum,
        args.velocity,
        edge_concentration,
        boundary_concentration,
        args.boundary_flux,
        projector,
        prescribed,
        step_size,
        plan.density_kg_m3 * jnp.mean(safe_volume) * speed_scale,
        plan.law.maximum_surface_concentration_mol_m2 * jnp.mean(area),
    )
    return elastic, marangoni_speed


def _elastic_initial(
    plan: SurfacePlugFlowPlan, state: Array, reference_velocity: Array, /
) -> Array:
    """Return the elastic unknown at the transported provisional state."""
    _, amount, _, _ = _unpack(plan, state)
    first, second = plan.tangent_frame(plan.surface)
    return jnp.concatenate(
        (
            jnp.sum(reference_velocity * first, axis=1),
            jnp.sum(reference_velocity * second, axis=1),
            amount / plan.surface.vertex_area,
        )
    )


def _elastic_solve(
    provisional: Array, time: Array, coefficient: Array, args: _StepArguments
) -> ImplicitConservationStageResult:
    """Solve the elastic/viscous part and commit its flux- and force-form update."""
    del time
    prepared = args.prepared
    plan = prepared.plan
    surface = plan.surface
    volume, amount, dissolved, momentum = _unpack(plan, provisional)
    elastic, marangoni_speed = _elastic_setup(args, provisional, coefficient)
    first, second = plan.tangent_frame(surface)
    initial = _elastic_initial(plan, provisional, args.velocity)
    solution = prepared.solver.solve(
        initial, elastic, factorization=args.elastic_factorization
    )
    count = surface.topology.num_vertices
    velocity = (
        solution.state[:count, None] * first
        + solution.state[count : 2 * count, None] * second
    )
    concentration = solution.state[2 * count :]
    boundary_compression = _boundary_compression(elastic, velocity)
    compressed = amount - coefficient * (
        _interior_compression(elastic, velocity) + boundary_compression
    )
    force, dissipation, drag_dissipation = plan.forces(
        surface, elastic.volume, concentration, velocity
    )
    free_momentum = _tangential(surface, momentum + coefficient * force)
    new_momentum = _hold(
        plan,
        elastic.projector,
        free_momentum,
        plan.density_kg_m3 * elastic.volume[:, None] * elastic.prescribed,
    )
    return ImplicitConservationStageResult(
        _pack(volume, compressed, dissolved, new_momentum),
        solution.status == NonlinearStatus.SUCCESS,
        solution.diagnostics.iterations,
        solution.diagnostics.final_residual_norm,
        solution.status,
        _ElasticEvidence(
            velocity=velocity,
            concentration=concentration,
            free_momentum=free_momentum,
            boundary_compression=boundary_compression,
            minimum_compressed_amount_mol=jnp.min(compressed),
            viscous_dissipation_w=dissipation,
            drag_dissipation_w=drag_dissipation,
            marangoni_speed_m_s=marangoni_speed,
        ),
    )


def _elastic_rate(time: Array, state: Array, args: _StepArguments) -> Array:
    """Return the elastic/viscous rate: compression correction and held forces."""
    del time
    plan = args.prepared.plan
    surface = plan.surface
    volume, amount, dissolved, momentum = _unpack(plan, state)
    elastic, _ = _elastic_setup(args, state, jnp.zeros((), dtype=state.dtype))
    velocity = momentum / (plan.density_kg_m3 * elastic.volume[:, None])
    compression = _interior_compression(elastic, velocity) + _boundary_compression(
        elastic, velocity
    )
    force, _, _ = plan.forces(
        surface, elastic.volume, amount / surface.vertex_area, velocity
    )
    return _pack(
        jnp.zeros_like(volume),
        -compression,
        None if dissolved is None else jnp.zeros_like(dissolved),
        _hold(
            plan, elastic.projector, _tangential(surface, force), jnp.zeros_like(force)
        ),
    )


def _surfactant_state(
    plan: SurfacePlugFlowPlan,
    volume: Array,
    amount: Array,
    dissolved: Array | None,
    /,
) -> SymmetricFilmSurfactantState:
    surface = plan.surface
    return SymmetricFilmSurfactantState(
        jnp.where(jnp.isfinite(volume) & (volume > 0.0), volume, surface.vertex_area),
        amount,
        dissolved,
        topology_id=surface.topology.topology_id,
        geometry_revision=surface.geometry_revision,
    )


def _exchange_solve(
    provisional: Array, time: Array, coefficient: Array, args: _StepArguments
) -> ImplicitConservationStageResult:
    """Solve surface diffusion and adsorption through the symmetric-film route."""
    del time
    plan = args.prepared.plan
    volume, amount, dissolved, momentum = _unpack(plan, provisional)
    result = args.prepared.surfactant._step(
        _surfactant_state(plan, volume, amount, dissolved),
        coefficient,
        args.surfactant_factorization,
    )
    candidate = result.candidate_state
    return ImplicitConservationStageResult(
        _pack(
            volume,
            candidate.surfactant_amount_mol,
            candidate.dissolved_amount_mol,
            momentum,
        ),
        result.status == FilmStepStatus.ACCEPTED,
        result.evidence.nonlinear_iterations,
        result.evidence.nonlinear_residual_norm,
        result.status,
        result.evidence,
    )


def _exchange_rate(time: Array, state: Array, args: _StepArguments) -> Array:
    """Return the surface diffusion and adsorption rates of ``state``."""
    del time
    plan = args.prepared.plan
    volume, amount, dissolved, momentum = _unpack(plan, state)
    amount_rate, dissolved_rate = args.prepared.surfactant.rates(
        _surfactant_state(plan, volume, amount, dissolved)
    )
    return _pack(
        jnp.zeros_like(volume), amount_rate, dissolved_rate, jnp.zeros_like(momentum)
    )


class _Transport(StrictModule):
    """Explicit donor-cell rates and open-boundary outflow rates per vertex."""

    rate: Array
    area_flux: Array
    boundary_flux: Array | None
    volume_outflow: Array
    amount_outflow: Array
    dissolved_outflow: Array
    momentum_outflow: Array

    def __init__(
        self,
        *,
        rate: Array,
        area_flux: Array,
        boundary_flux: Array | None,
        volume_outflow: Array,
        amount_outflow: Array,
        dissolved_outflow: Array,
        momentum_outflow: Array,
    ) -> None:
        self.rate = rate
        self.area_flux = area_flux
        self.boundary_flux = boundary_flux
        self.volume_outflow = volume_outflow
        self.amount_outflow = amount_outflow
        self.dissolved_outflow = dissolved_outflow
        self.momentum_outflow = momentum_outflow


def _transport(
    plan: SurfacePlugFlowPlan,
    volume: Array,
    amount: Array,
    dissolved: Array | None,
    momentum: Array,
    velocity: Array,
    /,
) -> _Transport:
    """Return transport rates of extensive content through interior and open edges.

    ``rate`` is the packed IMEX rate; outflows are the rates leaving each cell
    through open boundary edges and vanish without a declared boundary.
    """
    surface = plan.surface
    area = surface.vertex_area
    boundary = plan.boundary
    area_flux = surface.edge_area_flux(velocity)
    boundary_flux = None if boundary is None else boundary.area_flux(surface, velocity)

    def rate(content: Array, exterior: Array | None, /) -> tuple[Array, Array]:
        interior = -surface.edge_divergence(
            transported_edge_flux(
                plan.transport_scheme, surface, content, area_flux, area
            )
        )
        if boundary is None or boundary_flux is None or exterior is None:
            return interior, jnp.zeros_like(content)
        leaving = boundary.outflow(surface, content, exterior, boundary_flux)
        return interior - leaving, leaving

    exterior = _exterior_densities(plan)
    volume_rate, volume_outflow = rate(volume, exterior[0])
    amount_rate, amount_outflow = rate(amount, exterior[1])
    dissolved_rate, dissolved_outflow = (
        (None, jnp.zeros_like(volume))
        if dissolved is None
        else rate(dissolved, exterior[2])
    )
    momentum_rate, momentum_outflow = rate(momentum, exterior[3])
    return _Transport(
        rate=_pack(
            volume_rate,
            amount_rate,
            dissolved_rate,
            _tangential(surface, momentum_rate),
        ),
        area_flux=area_flux,
        boundary_flux=boundary_flux,
        volume_outflow=volume_outflow,
        amount_outflow=amount_outflow,
        dissolved_outflow=dissolved_outflow,
        momentum_outflow=momentum_outflow,
    )


def _exterior_densities(
    plan: SurfacePlugFlowPlan, /
) -> tuple[Array | None, Array | None, Array | None, Array | None]:
    """Return inflow densities of volume, amount, dissolved amount and momentum."""
    boundary = plan.boundary
    if boundary is None:
        return None, None, None, None
    thickness = boundary.inflow_thickness_m
    dissolved = boundary.inflow_dissolved_concentration_mol_m3
    return (
        thickness,
        boundary.inflow_surface_concentration_mol_m2,
        None if dissolved is None else dissolved * thickness,
        plan.density_kg_m3
        * thickness[:, None]
        * _tangential(plan.surface, boundary.inflow_velocity_m_s),
    )


def _constraint(plan: SurfacePlugFlowPlan, /) -> tuple[Array, Array]:
    count = plan.surface.topology.num_vertices
    if plan.boundary is None:
        return jnp.zeros((count, 3, 3)), jnp.zeros((count, 3))
    return plan.boundary.velocity_constraint(plan.surface)


def _hold(
    plan: SurfacePlugFlowPlan, projector: Array, free: Array, held: Array, /
) -> Array:
    """Replace the constrained components ``Q free`` by ``Q held``."""
    if plan.boundary is None:
        return free
    return free + contract("vab,vb->va", projector, held - free)


def _constrained_line_tension(plan: SurfacePlugFlowPlan, tension: Array, /) -> Array:
    """Return the boundary line tension carried by velocity constraints (N)."""
    surface = plan.surface
    boundary = plan.boundary
    if boundary is None:
        return jnp.zeros((surface.topology.num_vertices, 3))
    line = _tangential(surface, boundary.line_tension_force(surface, tension))
    return jnp.where(boundary.constrained_vertices[:, None], line, 0.0)


class _ElasticArguments(StrictModule):
    plan: SurfacePlugFlowPlan
    volume: Array
    amount: Array
    momentum: Array
    velocity: Array
    edge_concentration: Array
    boundary_concentration: Array | None
    boundary_flux: Array | None
    projector: Array
    prescribed: Array
    step_size: Array
    momentum_scale: Array
    amount_scale: Array

    def __init__(
        self,
        plan: SurfacePlugFlowPlan,
        volume: Array,
        amount: Array,
        momentum: Array,
        velocity: Array,
        edge_concentration: Array,
        boundary_concentration: Array | None,
        boundary_flux: Array | None,
        projector: Array,
        prescribed: Array,
        step_size: Array,
        momentum_scale: Array,
        amount_scale: Array,
        /,
    ) -> None:
        self.plan = plan
        self.volume = volume
        self.amount = amount
        self.momentum = momentum
        self.velocity = velocity
        self.edge_concentration = edge_concentration
        self.boundary_concentration = boundary_concentration
        self.boundary_flux = boundary_flux
        self.projector = projector
        self.prescribed = prescribed
        self.step_size = step_size
        self.momentum_scale = momentum_scale
        self.amount_scale = amount_scale


def _interior_compression(args: _ElasticArguments, velocity: Array, /) -> Array:
    """Return the implicit interior compression outflow per vertex (mol/s)."""
    surface = args.plan.surface
    return surface.edge_divergence(
        args.edge_concentration
        * (surface.edge_area_flux(velocity) - surface.edge_area_flux(args.velocity))
    )


def _boundary_compression(args: _ElasticArguments, velocity: Array, /) -> Array:
    """Return the implicit compression outflow through open edges (mol/s)."""
    surface = args.plan.surface
    boundary = args.plan.boundary
    if (
        boundary is None
        or args.boundary_flux is None
        or args.boundary_concentration is None
    ):
        return jnp.zeros((surface.topology.num_vertices,))
    change = boundary.area_flux(surface, velocity) - args.boundary_flux
    return boundary.vertex_sum(surface, args.boundary_concentration * change)


def _elastic_residual(unknowns: Array, args: _ElasticArguments) -> Array:
    plan = args.plan
    surface = plan.surface
    count = surface.topology.num_vertices
    first, second = plan.tangent_frame(surface)
    velocity = unknowns[:count, None] * first + unknowns[count : 2 * count, None] * second
    concentration = unknowns[2 * count :]
    force, _, _ = plan.forces(surface, args.volume, concentration, velocity)
    mass = plan.density_kg_m3 * args.volume[:, None]
    balance = _hold(
        plan,
        args.projector,
        (mass * velocity - args.momentum - args.step_size * force) / args.momentum_scale,
        mass * (velocity - args.prescribed) / args.momentum_scale,
    )
    amount_residual = (
        surface.vertex_area * concentration
        - args.amount
        + args.step_size
        * (_interior_compression(args, velocity) + _boundary_compression(args, velocity))
    ) / args.amount_scale
    return jnp.concatenate(
        (
            jnp.sum(balance * first, axis=1),
            jnp.sum(balance * second, axis=1),
            amount_residual,
        )
    )


def _tangential(surface: PreparedFilmSurface, vectors: Array, /) -> Array:
    normal = surface.vertex_normal
    return vectors - jnp.sum(vectors * normal, axis=1, keepdims=True) * normal


def _check_boundary(
    surface: PreparedFilmSurface,
    law: LangmuirSurfactantLaw,
    kinetics: AdsorptionKinetics | None,
    boundary: PlugFlowBoundary,
    /,
) -> None:
    if not isinstance(boundary, PlugFlowBoundary):
        raise TypeError("boundary must be a PlugFlowBoundary or None.")
    if boundary.topology_id != surface.topology.topology_id:
        raise ValueError("Boundary topology does not match the prepared surface.")
    if (boundary.inflow_dissolved_concentration_mol_m3 is None) != (kinetics is None):
        raise ValueError(
            "Exterior dissolved concentration is required exactly for soluble films."
        )
    inflow_evaluation = law.evaluate(
        boundary.inflow_surface_concentration_mol_m2
    )
    if not np.all(np.asarray(inflow_evaluation.admissible)):
        raise ValueError(
            "Inflow surface concentration must be finite, nonnegative, below "
            "the Langmuir capacity, and give positive surface tension."
        )


def _check_state(
    surface: PreparedFilmSurface, state: SurfacePlugFlowState, /, *, soluble: bool
) -> None:
    if not isinstance(state, SurfacePlugFlowState):
        raise TypeError("state must be a SurfacePlugFlowState.")
    if state.topology_id != surface.topology.topology_id:
        raise ValueError("State topology does not match the prepared surface.")
    if state.liquid_volume_m3.shape != (surface.topology.num_vertices,):
        raise ValueError("State does not match the surface vertex count.")
    if (state.dissolved_amount_mol is not None) != soluble:
        raise ValueError("State dissolved amount does not match the declared solubility.")


def _scalar(value: ArrayLike, name: str, /, *, positive: bool) -> Array:
    host = np.asarray(value, dtype=np.float64)
    if (
        host.shape != ()
        or not np.isfinite(host)
        or (host <= 0.0 if positive else host < 0.0)
    ):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {bound} scalar.")
    return jnp.asarray(host)


def _vector(value: ConvertibleToArray, name: str, /) -> Array:
    host = np.asarray(value, dtype=np.float64)
    if host.shape != (3,) or not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must be a finite 3-vector.")
    return jnp.asarray(host)


__all__ = [
    "PlugFlowEvidence",
    "PlugFlowNonlinearStage",
    "PreparedSurfacePlugFlow",
    "SurfacePlugFlowPlan",
    "SurfacePlugFlowState",
    "SurfacePlugFlowStepResult",
]
