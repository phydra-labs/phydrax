#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Incompressible meshfree flow on an admitted positive exterior one-complex.

The physical state is a nodal velocity (and density) on the exterior's compact
nodes. One step is a native additive IMEX step of the projected
(Navier–)Stokes system: conservative momentum/density transport by
``ConservativeTransport`` is explicit, implicit viscous diffusion uses the
exterior's conservative graph Laplacian (momentum-conserving, dissipative),
and every implicit stage ends with a pressure projection by
``CompatibleIncompressibleProjection`` (or the variable-density owner) of the
velocity's de Rham edge cochain ``u_e = (u_i + u_j)/2 . (x_j - x_i)``. The
projected edge volume flux is the algebraic stage component that advects the
next explicit stage, so stage rates are evaluated at projected stage states
with their own advecting flux. The explicit rate also carries the state's
pressure gradient (incremental pressure correction), so the approximate nodal
projection removes only the pressure increment of the step. On a periodic box
the projection commutes with the viscous operator, so this is the IMEX method
applied to the projected ODE ``u' = P(N(u) + nu L u)`` and keeps the
tableau's velocity order; a projection after the whole step, a flux lagged
from the previous step, or a projection of the whole pressure gradient each
step is first order in time.

A graph-solenoidal edge field is not a physical velocity: a radius graph has
``E - V + C`` independent cycles while the domain has ``b_1`` physical
harmonic fields. Nodal velocity is recovered by moment-consistent weighted
least squares ``v_i = M_i^{-1} sum_j w_ij u_ij (x_j - x_i)`` with
``M_i = sum_j w_ij (x_j - x_i)(x_j - x_i)^T`` (``2 V_i I`` for an accepted
degree-two metric), the Hodge adjoint of the de Rham map. Edge content that
this reconstruction cannot represent is an artificial cycle mode; its share of
the edge energy is published and a step whose projected field exceeds the
declared tolerance is refused, never accepted as physical flow.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Any, assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import PreparedPointCloudDiscretization
from ..discretization.meshfree import (
    ConservativeTransport,
    MeshfreeDiffusionOperator,
    PreparedMeshfreeExteriorCalculus,
    TransportScheme,
    TransportVolumeFlux,
)
from ..linalg import (
    ArraySpace,
    bind_numeric,
    FailurePolicy,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveTemplate,
    LinearSystem,
    prepare_template,
    SmallLinearSolvePlan,
    solve,
    solve_small_linear,
    TolerancePolicy,
)
from ..sparse import gather_routes, linear_transpose_apply
from ..typing import checked, parse
from ._balance_law_composition import (
    additive_imex_tableau,
    AdditiveIMEXScheme,
    AdditiveIMEXTableau,
)
from ._compatible_systems import (
    CompatibleIncompressibleProjection,
    CompatibleVariableDensityProjection,
    IncompressibleProjectionResult,
)
from ._conservation_temporal import (
    ConservationIMEXMethod,
    ConservationIMEXResult,
    ImplicitConservationStageResult,
)
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult


IncompressibleDensityModel: TypeAlias = Literal["constant", "variable"]


class MeshfreeFlowStatus(IntEnum):
    """Fail-closed incompressible step acceptance in reporting precedence."""

    ACCEPTED = 0
    INVALID_STEP = 1
    NONFINITE = 2
    CFL_REFUSED = 3
    TRANSPORT_REFUSED = 4
    IMPLICIT_REFUSED = 5
    PROJECTION_REFUSED = 6
    RECONSTRUCTION_REFUSED = 7
    SPURIOUS_CYCLES = 8
    NONPOSITIVE_DENSITY = 9


@final
class MeshfreeCycleEvidence(StrictModule):
    """Graph cycle space versus declared domain topology for one edge field.

    ``cycle_space_dimension = E - V + C`` counts independent graph cycles;
    ``artificial_cycle_dimension`` subtracts the declared first Betti number
    of the physical domain. ``nonphysical_fraction`` is the Hodge-norm share
    ``||u - I R u|| / ||u||`` of edge content the nodal reconstruction ``R``
    cannot represent (``I`` the de Rham map); ``physical`` compares it with
    the declared tolerance.
    """

    nonphysical_fraction: Array
    physical: Array
    cycle_space_dimension: int = eqx.field(static=True)
    domain_betti_number: int = eqx.field(static=True)
    artificial_cycle_dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)


@final
class MeshfreeIncompressibleState(StrictModule):
    """Nodal velocity/density, kinematic pressure, and advecting edge flux.

    ``volume_flux`` is the graph-solenoidal edge volume flux ``w_e u_e`` of the
    state's projection; it advects the first explicit stage of the next step.
    ``pressure`` is the pressure (density-scaled for a constant-density model)
    whose nodal gradient balances the momentum update of the step that
    produced it. The next step's explicit rate carries its gradient, so each
    projection removes only the pressure increment; ``initialize`` starts it
    at zero.
    """

    velocity: Array
    density: Array
    pressure: Array
    volume_flux: Array


@final
class MeshfreeFlowEvidence(StrictModule):
    """Outcome-independent evidence of one attempted incompressible step.

    Projection evidence aggregates every pressure projection of the step:
    ``projection_status`` is the first refused projection's status (the last
    one's when all succeed), ``pressure_iterations`` their sum and
    ``pressure_residual_norm`` their maximum. Graph divergence and cycle
    content describe the committed edge flux.
    """

    status: Array
    successful: Array
    cfl: Array
    transport_admitted: Array
    implicit_successful: Array
    implicit_iterations: Array
    projection_status: Array
    pressure_iterations: Array
    pressure_residual_norm: Array
    graph_divergence_norm: Array
    nodal_divergence_before: Array
    nodal_divergence_after: Array
    reconstruction_condition: Array
    cycles: MeshfreeCycleEvidence
    kinetic_energy_before: Array
    kinetic_energy_after: Array
    mass_before: Array
    mass_after: Array
    momentum_before: Array
    momentum_after: Array


@final
class MeshfreeIncompressibleStepResult(StrictModule):
    state: MeshfreeIncompressibleState
    candidate: MeshfreeIncompressibleState
    evidence: MeshfreeFlowEvidence

    @property
    def successful(self) -> Array:
        return self.evidence.successful


@final
class MeshfreeVelocityReconstruction(StrictModule, NonTrainableState):
    """Moment-consistent weighted least squares between nodes and edge cochains.

    ``interpolate`` is the de Rham map ``u_e = (u_i + u_j)/2 . e_ij``;
    ``reconstruct`` solves ``M_i v_i = sum_j w_ij u_ij e_ij`` per node with the
    native small linear owner. Its normal matrix is the metric's second moment,
    so constant fields are reproduced exactly and affine fields to the third
    edge moment; ``condition`` reports each node's small-solve conditioning.
    """

    exterior: PreparedMeshfreeExteriorCalculus
    moments: Array
    condition: Array
    admitted: Array
    dimension: int = eqx.field(static=True)

    def __init__(self, exterior: PreparedMeshfreeExteriorCalculus, /) -> None:
        if not isinstance(exterior, PreparedMeshfreeExteriorCalculus):
            raise TypeError("exterior must be a PreparedMeshfreeExteriorCalculus.")
        dimension = exterior.points.shape[1]
        edges = exterior.endpoint_displacements[:, 0, :]
        if edges.shape[1] != dimension:
            raise ValueError("Velocity reconstruction needs ambient endpoint charts.")
        weights = exterior.metric_result.weights
        outer = weights[:, None, None] * edges[:, :, None] * edges[:, None, :]
        moments = _node_sum(exterior, outer)
        check = solve_small_linear(
            SmallLinearSolvePlan(dimension),
            moments,
            jnp.broadcast_to(jnp.eye(dimension, dtype=jnp.float64), moments.shape),
        )
        self.exterior = exterior
        self.moments = moments
        self.condition = check.condition_estimate
        self.admitted = jnp.all(check.successful) & jnp.all(weights > 0.0)
        self.dimension = dimension

    @property
    def edge_vectors(self) -> Array:
        return self.exterior.endpoint_displacements[:, 0, :]

    @property
    def edge_weights(self) -> Array:
        return self.exterior.metric_result.weights

    def interpolate(self, velocity: ArrayLike, /) -> Array:
        value = jnp.asarray(velocity, dtype=jnp.float64)
        endpoints = gather_routes(self.exterior.incidence.relation, value).reshape(
            (self.edge_weights.shape[0], 2, self.dimension)
        )
        mean = 0.5 * (endpoints[:, 0] + endpoints[:, 1])
        return jnp.sum(mean * self.edge_vectors, axis=1)

    def reconstruct(self, edge_values: ArrayLike, /) -> Array:
        value = jnp.asarray(edge_values, dtype=jnp.float64)
        moment = _node_sum(
            self.exterior, (self.edge_weights * value)[:, None] * self.edge_vectors
        )
        return solve_small_linear(
            SmallLinearSolvePlan(self.dimension), self.moments, moment
        ).value

    def edge_norm(self, edge_values: Array, /) -> Array:
        return jnp.sqrt(jnp.sum(self.edge_weights * edge_values**2))


def _node_sum(exterior: PreparedMeshfreeExteriorCalculus, values: Array, /) -> Array:
    """Unsigned accumulation of edge payloads onto both endpoint nodes."""
    incidence = exterior.incidence
    return linear_transpose_apply(
        incidence.relation, jnp.abs(incidence.coefficients), values
    )


@final
class MeshfreeIncompressibleFlowPlan(StrictModule):
    """Transient incompressible (Navier–)Stokes flow on a closed exterior graph.

    ``exterior`` must carry an admitted positive metric (its native cochain is
    the pressure complex) and no boundary rows: transient flow is admitted on
    closed graphs (periodic boxes, closed manifolds). Bounded steady flow is
    the ``MeshfreeGeneralizedStokesPlan`` owner. ``reconstruction`` is a
    prepared cloud on exactly the exterior's compact nodes; it provides the
    transport reconstruction and nodal divergence diagnostics. The viscous
    term is the exterior's conservative graph Laplacian. ``viscosity`` is
    dynamic; ``density_model`` selects constant (``reference_density``) or
    transported variable density with the variable-density projection.
    ``body_force(time, points)`` is an acceleration ``(N, d)``.
    ``domain_betti_number`` declares the physical first Betti number used to
    classify cycle content (2 for a planar periodic box). ``method`` is a
    one-part IMEX tableau whose stages after the first, and whose last stage,
    are implicit with a nonzero diagonal (the ARS schemes, forward–backward
    Euler and SSP2-222); each such stage is projected. A tableau that is not
    stiffly accurate projects its combined step result once more.
    """

    exterior: PreparedMeshfreeExteriorCalculus
    reconstruction: PreparedPointCloudDiscretization
    transport: ConservativeTransport
    diffusion: MeshfreeDiffusionOperator
    velocity_reconstruction: MeshfreeVelocityReconstruction
    projection: CompatibleIncompressibleProjection | CompatibleVariableDensityProjection
    tableau: AdditiveIMEXTableau
    linear_template: LinearSolveTemplate
    body_force: Callable[[Array, Array], Array] | None = eqx.field(static=True)
    viscosity: float = eqx.field(static=True)
    reference_density: float = eqx.field(static=True)
    density_model: IncompressibleDensityModel = eqx.field(static=True)
    domain_betti_number: int = eqx.field(static=True)
    cycle_tolerance: float = eqx.field(static=True)
    cfl_limit: float = eqx.field(static=True)
    implicit_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        exterior: PreparedMeshfreeExteriorCalculus,
        reconstruction: PreparedPointCloudDiscretization,
        /,
        *,
        viscosity: float,
        domain_betti_number: int,
        density_model: IncompressibleDensityModel = "constant",
        reference_density: float = 1.0,
        body_force: Callable[[Array, Array], Array] | None = None,
        method: AdditiveIMEXScheme | AdditiveIMEXTableau = "ars-222",
        transport_scheme: TransportScheme = "limited",
        cycle_tolerance: float = 0.05,
        cfl_limit: float = 0.5,
        pressure_policy: LinearSolvePolicy | None = None,
        viscous_policy: LinearSolvePolicy | None = None,
    ) -> None:
        if not bool(np.all(np.asarray(exterior.equation_mask))):
            raise ValueError(
                "Transient meshfree projection admits closed graphs only; bounded "
                "steady flow is MeshfreeGeneralizedStokesPlan."
            )
        model = parse(density_model, IncompressibleDensityModel, "density_model")
        mu, rho = float(viscosity), float(reference_density)
        if not np.isfinite(mu) or mu < 0.0:
            raise ValueError("viscosity must be finite and nonnegative.")
        if not np.isfinite(rho) or rho <= 0.0:
            raise ValueError("reference_density must be finite and positive.")
        betti = int(domain_betti_number)
        if betti < 0:
            raise ValueError("domain_betti_number must be nonnegative.")
        tolerance, limit = float(cycle_tolerance), float(cfl_limit)
        if not 0.0 < tolerance < 1.0:
            raise ValueError("cycle_tolerance must lie in (0, 1).")
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("cfl_limit must be finite and positive.")
        tableau = (
            additive_imex_tableau(parse(method, AdditiveIMEXScheme, "method"))
            if isinstance(method, str)
            else method
        )
        if not isinstance(tableau, AdditiveIMEXTableau) or tableau.part_count != 1:
            raise ValueError(
                "Incompressible flow needs a one-part additive IMEX tableau."
            )
        diagonal = np.diag(np.asarray(tableau.implicit_matrix))
        projected = [
            part is not None and diagonal[stage] != 0.0
            for stage, part in enumerate(tableau.implicit_parts)
        ]
        # Stage values are projected by their implicit solve. Only a first
        # explicit stage may skip it: it is the already projected step state.
        if not all(projected[1:]) or not projected[-1]:
            raise ValueError(
                "Stage-projected incompressible flow needs an implicit stage with a "
                "nonzero diagonal at every stage after the first and at the last."
            )
        if pressure_policy is not None and (
            not isinstance(pressure_policy, LinearSolvePolicy)
            or pressure_policy.failure.mode != "status"
        ):
            raise ValueError(
                "pressure_policy must be a status-returning LinearSolvePolicy."
            )
        cochain = exterior.to_cochain()
        match model:
            case "constant":
                projection: (
                    CompatibleIncompressibleProjection
                    | CompatibleVariableDensityProjection
                ) = CompatibleIncompressibleProjection(
                    cochain, solve_policy=pressure_policy
                )
            case "variable":
                projection = CompatibleVariableDensityProjection(
                    cochain, solve_policy=pressure_policy
                )
            case unknown:
                assert_never(unknown)
        nodes = exterior.points.shape[0]
        vertices = projection.pressure_system.vertex_indices.shape[0]
        edges = projection.pressure_system.edge_indices.shape[0]
        cycles = edges - vertices + projection.pressure_system.nullity
        if betti > cycles:
            raise ValueError(
                "Declared domain Betti number exceeds the graph cycle space."
            )
        if vertices != nodes or edges != exterior.pairs.shape[0]:
            raise ValueError("Projection complex must activate every exterior cell.")
        transport = ConservativeTransport(
            exterior,
            scheme=transport_scheme,
            reconstruction=None if transport_scheme == "upwind" else reconstruction,
        )
        if transport_scheme == "upwind" and not np.allclose(
            np.asarray(reconstruction.points), np.asarray(exterior.points)
        ):
            raise ValueError("reconstruction must lie on the exterior's compact nodes.")
        dimension = exterior.points.shape[1]
        identifier = canonical_fingerprint(
            {
                "kind": "meshfree-incompressible-flow",
                "projection": projection.projection_id,
                "transport": transport.transport_id,
                "reconstruction": reconstruction.prepared_id,
                "tableau": tableau.tableau_id,
                "viscosity": mu,
                "density_model": model,
                "reference_density": rho,
                "betti": betti,
                "cycle_tolerance": tolerance,
                "cfl_limit": limit,
            }
        )
        implicit_id = f"{identifier}:viscous"
        space = ArraySpace((nodes, dimension), dtype=jnp.float64)
        policy = (
            LinearSolvePolicy(
                GMRES(restart=min(60, nodes * dimension)),
                tolerance=TolerancePolicy(relative=1e-11, absolute=1e-13, max_steps=600),
                failure=FailurePolicy("status"),
            )
            if viscous_policy is None
            else viscous_policy
        )
        if not isinstance(policy, LinearSolvePolicy) or policy.failure.mode != "status":
            raise ValueError(
                "viscous_policy must be a status-returning LinearSolvePolicy."
            )
        structure = FunctionLinearOperator(
            lambda value: value, source=space, target=space, operator_id=implicit_id
        )
        self.exterior = exterior
        self.reconstruction = reconstruction
        self.transport = transport
        self.diffusion = exterior.diffusion(1.0)
        self.velocity_reconstruction = MeshfreeVelocityReconstruction(exterior)
        self.projection = projection
        self.tableau = tableau
        self.linear_template = prepare_template(
            LinearSystem(structure, problem_id=implicit_id), policy
        )
        self.body_force = body_force
        self.viscosity = mu
        self.reference_density = rho
        self.density_model = model
        self.domain_betti_number = betti
        self.cycle_tolerance = tolerance
        self.cfl_limit = limit
        self.implicit_id = implicit_id
        self.plan_id = identifier

    @property
    def cycle_space_dimension(self) -> int:
        system = self.projection.pressure_system
        return (
            system.edge_indices.shape[0] - system.vertex_indices.shape[0] + system.nullity
        )

    @property
    def projection_count(self) -> int:
        """Pressure projections per step: projected stages plus a final one."""
        stages = sum(part is not None for part in self.tableau.implicit_parts)
        return stages + (0 if self.tableau.stiffly_accurate else 1)

    def viscous_rate(self, velocity: Array, /) -> Array:
        """Componentwise conservative graph Laplacian of a nodal ``(N, d)`` field."""
        return jax.vmap(self.diffusion.mv, in_axes=1, out_axes=1)(velocity)

    def prepare(self, /) -> PreparedMeshfreeIncompressibleFlow:
        return PreparedMeshfreeIncompressibleFlow(self)


@final
class PreparedMeshfreeIncompressibleFlow(AbstractFixedStepMethod, NonTrainableState):
    """Native fixed-step owner of one incompressible meshfree flow plan.

    ``step`` publishes the IMEX/transport/projection candidate and commits it
    only when every owner accepts it and the projected field is physical; a
    refusal returns the unchanged state (``FixedStepResult`` semantics), so
    native rollouts hold the state after the first refusal.
    """

    plan: MeshfreeIncompressibleFlowPlan
    method_id: str = eqx.field(static=True)

    def __init__(self, plan: MeshfreeIncompressibleFlowPlan, /) -> None:
        if not isinstance(plan, MeshfreeIncompressibleFlowPlan):
            raise TypeError("plan must be a MeshfreeIncompressibleFlowPlan.")
        self.plan = plan
        self.method_id = plan.plan_id

    @property
    def points(self) -> Array:
        return self.plan.exterior.points

    # -- diagnostics -----------------------------------------------------

    def classify_edge_velocity(self, edge_values: ArrayLike, /) -> MeshfreeCycleEvidence:
        """Cycle-content evidence of a degree-one edge velocity cochain."""
        plan = self.plan
        value = jnp.asarray(edge_values, dtype=jnp.float64)
        if value.shape != plan.exterior.lengths.shape:
            raise ValueError("Edge velocity must be one value per exterior edge.")
        rebuild = plan.velocity_reconstruction
        representable = rebuild.interpolate(rebuild.reconstruct(value))
        fraction = rebuild.edge_norm(value - representable) / jnp.maximum(
            rebuild.edge_norm(value), jnp.finfo(jnp.float64).tiny
        )
        return MeshfreeCycleEvidence(
            nonphysical_fraction=fraction,
            physical=fraction <= plan.cycle_tolerance,
            cycle_space_dimension=plan.cycle_space_dimension,
            domain_betti_number=plan.domain_betti_number,
            artificial_cycle_dimension=plan.cycle_space_dimension
            - plan.domain_betti_number,
            tolerance=plan.cycle_tolerance,
        )

    def kinetic_energy(self, state: MeshfreeIncompressibleState, /) -> Array:
        volumes = self.plan.exterior.node_volumes
        return 0.5 * jnp.sum(volumes * state.density * jnp.sum(state.velocity**2, axis=1))

    def nodal_divergence(self, velocity: ArrayLike, /) -> Array:
        """GMLS divergence of a nodal velocity on the reconstruction cloud."""
        return self.plan.reconstruction.divergence(
            jnp.asarray(velocity, dtype=jnp.float64)
        )

    # -- projection --------------------------------------------------------

    def _project(
        self, velocity: Array, density: Array, /
    ) -> tuple[IncompressibleProjectionResult, Array]:
        plan = self.plan
        edges = plan.velocity_reconstruction.interpolate(velocity)
        match plan.projection:
            case CompatibleIncompressibleProjection():
                result = plan.projection.project(edges)
            case CompatibleVariableDensityProjection():
                result = plan.projection.project(edges, density)
            case unknown:
                assert_never(unknown)
        correction = plan.velocity_reconstruction.reconstruct(
            edges - result.candidate_velocity
        )
        return result, velocity - correction

    @property
    def _pressure_scale(self) -> float:
        """Physical pressure per kinematic projection potential rate."""
        plan = self.plan
        match plan.density_model:
            case "constant":
                return plan.reference_density
            case "variable":
                return 1.0
            case unknown:
                assert_never(unknown)

    def _pressure_acceleration(self, pressure: Array, density: Array, /) -> Array:
        """Nodal ``R(c_e d p)``: the projection's correction law for ``p``.

        ``c_e`` is one or the variable-density owner's ``rho_e^{-1}``, so a
        known kinematic pressure enters the momentum rate exactly as the
        projection would remove it.
        """
        plan = self.plan
        gradient = plan.exterior.gradient(pressure)
        match plan.projection:
            case CompatibleIncompressibleProjection():
                edge = gradient
            case CompatibleVariableDensityProjection():
                edge = plan.projection.edge_inverse_density(density) * gradient
            case unknown:
                assert_never(unknown)
        return plan.velocity_reconstruction.reconstruct(edge)

    def initialize(
        self, velocity: ArrayLike, /, *, density: ArrayLike | None = None
    ) -> MeshfreeIncompressibleStepResult:
        """Project an initial velocity; refusal publishes the unprojected input."""
        plan = self.plan
        count, dimension = plan.exterior.points.shape
        value = jnp.asarray(velocity, dtype=jnp.float64)
        if value.shape != (count, dimension):
            raise ValueError("velocity must have shape (compact nodes, dimension).")
        rho = self._density(density, count)
        result, projected = self._project(value, rho)
        zero = jnp.zeros((count,), dtype=jnp.float64)
        flux = plan.velocity_reconstruction.edge_weights * result.candidate_velocity
        candidate = MeshfreeIncompressibleState(projected, rho, zero, flux)
        unprojected = MeshfreeIncompressibleState(
            value,
            rho,
            zero,
            plan.velocity_reconstruction.edge_weights
            * plan.velocity_reconstruction.interpolate(value),
        )
        evidence = self._evidence(
            unprojected,
            candidate,
            (result,),
            jnp.asarray(0.0),
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(True),
        )
        state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(evidence.successful, new, old),
            candidate,
            unprojected,
        )
        return MeshfreeIncompressibleStepResult(state, candidate, evidence)

    def _density(self, density: ArrayLike | None, count: int, /) -> Array:
        plan = self.plan
        match plan.density_model:
            case "constant":
                if density is not None:
                    raise ValueError("Constant-density flow takes no density field.")
                return jnp.full((count,), plan.reference_density, dtype=jnp.float64)
            case "variable":
                if density is None:
                    raise ValueError("Variable-density flow needs a density field.")
                value = jnp.asarray(density, dtype=jnp.float64)
                if value.shape != (count,):
                    raise ValueError("density must have one value per compact node.")
                return value
            case unknown:
                assert_never(unknown)

    # -- IMEX --------------------------------------------------------------

    def _imex(
        self, state: MeshfreeIncompressibleState, time: Array, step: Array, args: Any, /
    ) -> ConservationIMEXResult:
        """One IMEX step whose implicit stages end with a pressure projection.

        The packed stage state is ``(momentum, density?, volume_flux)``. The
        flux is the algebraic stage component: it has zero rates, each
        implicit stage sets it to that stage's projected edge flux, and the
        explicit transport rate of a stage is advected by it. Each implicit
        stage's projection result is its stage evidence.

        The explicit rate carries the state's pressure gradient (incremental
        pressure correction), so each projection removes only the pressure
        increment. The nodal projection ``v - R(I - P) I v`` is approximate:
        it leaves an ``O(h^2)`` share of the removed gradient in the nodal
        velocity, which is ``O(h^2 dt)`` per step when the whole pressure is
        removed but ``O(h^2 dt^2)`` for the increment.
        """
        plan = self.plan
        pressure = state.pressure / self._pressure_scale
        count, dimension = plan.exterior.points.shape
        points = plan.exterior.points
        boundary = jnp.zeros((count,), dtype=jnp.float64)
        variable = plan.density_model == "variable"
        nodal = count * dimension
        flux_start = nodal + count if variable else nodal
        space = ArraySpace((count, dimension), dtype=jnp.float64)

        def unpack(packed: Array, /) -> tuple[Array, Array, Array]:
            momentum = packed[:nodal].reshape((count, dimension))
            density = packed[nodal:flux_start] if variable else state.density
            return momentum, density, packed[flux_start:]

        def pack(momentum: Array, density: Array, flux: Array, /) -> Array:
            parts = [momentum.reshape((-1,))]
            if variable:
                parts.append(density)
            parts.append(flux)
            return jnp.concatenate(parts)

        def explicit_rhs(t: Array, packed: Array, context: Any, /) -> Array:
            del context
            momentum, density, flux = unpack(packed)
            positive = jnp.all(density > 0.0)
            safe_density = jnp.where(positive, density, 1.0)
            advection = TransportVolumeFlux(edge=flux, boundary=boundary)
            rates = jnp.stack(
                [
                    plan.transport.rate(momentum[:, a], advection).value_rate
                    for a in range(dimension)
                ],
                axis=1,
            )
            if plan.body_force is not None:
                rates = rates + density[:, None] * jnp.asarray(
                    plan.body_force(t, points), dtype=jnp.float64
                )
            rates = rates - safe_density[:, None] * self._pressure_acceleration(
                pressure, safe_density
            )
            density_rate = (
                plan.transport.rate(density, advection).value_rate
                if variable
                else jnp.zeros_like(density)
            )
            return pack(rates, density_rate, jnp.zeros_like(flux))

        def implicit_rhs(t: Array, packed: Array, context: Any, /) -> Array:
            del t, context
            momentum, density, flux = unpack(packed)
            viscous = plan.viscosity * plan.viscous_rate(momentum / density[:, None])
            return pack(viscous, jnp.zeros_like(density), jnp.zeros_like(flux))

        def implicit_solve(
            provisional: Array, t: Array, coefficient: Array, context: Any, /
        ) -> ImplicitConservationStageResult:
            del t, context
            momentum, density, _ = unpack(provisional)
            positive = jnp.all(density > 0.0)
            safe_density = jnp.where(positive, density, 1.0)
            weight = safe_density[:, None]
            operator = FunctionLinearOperator(
                lambda value: (
                    weight * value
                    - coefficient * plan.viscosity * plan.viscous_rate(value)
                ),
                source=space,
                target=space,
                operator_id=plan.implicit_id,
            )
            prepared = bind_numeric(
                plan.linear_template,
                LinearSystem(operator, problem_id=plan.implicit_id),
            )
            result = solve(prepared, momentum)
            projection, projected = self._project(result.value, safe_density)
            return ImplicitConservationStageResult(
                pack(
                    weight * projected,
                    density,
                    plan.velocity_reconstruction.edge_weights
                    * projection.candidate_velocity,
                ),
                result.successful & positive,
                result.diagnostics.iterations.astype(jnp.int32),
                result.diagnostics.residual_norm,
                result.status.astype(jnp.int32),
                projection,
            )

        method = ConservationIMEXMethod(
            plan.tableau,
            explicit_rhs,
            implicit_rhs,
            implicit_solve,
            method_id=plan.plan_id,
        )
        initial = pack(
            state.density[:, None] * state.velocity, state.density, state.volume_flux
        )
        return method.step(time, initial, step, args)

    def advance(
        self,
        state: MeshfreeIncompressibleState,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        args: Any = None,
    ) -> MeshfreeIncompressibleStepResult:
        plan = self.plan
        tableau = plan.tableau
        count, dimension = plan.exterior.points.shape
        nodal = count * dimension
        variable = plan.density_model == "variable"
        dt = jnp.asarray(step_size, dtype=jnp.float64)
        t = jnp.asarray(time, dtype=jnp.float64)
        valid_step = jnp.isfinite(dt) & (dt > 0.0) & jnp.isfinite(t)
        safe_dt = jnp.where(valid_step, dt, 1.0)
        cfl = plan.transport.cfl(
            TransportVolumeFlux(
                edge=state.volume_flux, boundary=jnp.zeros((count,), dtype=jnp.float64)
            ),
            safe_dt,
        )
        imex = self._imex(state, t, safe_dt, args)
        packed = imex.candidate_state
        momentum = packed[:nodal].reshape((count, dimension))
        density = packed[nodal : nodal + count] if variable else state.density
        flux = packed[nodal + count :] if variable else packed[nodal:]
        positive = jnp.all(density > 0.0)
        safe_density = jnp.where(positive, density, 1.0)
        velocity = momentum / safe_density[:, None]
        stages = [
            (stage, evidence)
            for stage, evidence in enumerate(imex.stage_evidence)
            if evidence is not None
        ]
        projections: list[IncompressibleProjectionResult] = [
            evidence for _, evidence in stages
        ]
        # Stage j removes dt a_jj grad(p_j - p_n); the weights combine the
        # stage increments into the pressure of the step's momentum update.
        increment = sum(
            (
                tableau.weights[stage]
                / tableau.implicit_matrix[stage, stage]
                * evidence.candidate_pressure
                for stage, evidence in stages
            ),
            start=jnp.zeros((count,), dtype=jnp.float64),
        )
        if not tableau.stiffly_accurate:
            final, velocity = self._project(velocity, safe_density)
            flux = plan.velocity_reconstruction.edge_weights * final.candidate_velocity
            increment = increment + final.candidate_pressure
            projections.append(final)
        candidate = MeshfreeIncompressibleState(
            velocity,
            density,
            state.pressure + self._pressure_scale * increment / safe_dt,
            flux,
        )
        evidence = self._evidence(
            state,
            candidate,
            tuple(projections),
            cfl.cfl,
            jnp.asarray(imex.successful),
            jnp.asarray(imex.implicit_iterations, dtype=jnp.int32),
            valid_step & positive & (cfl.cfl <= plan.cfl_limit),
            transport_admitted=jnp.asarray(
                plan.transport.exterior.metric_result.hilbert_admitted
            ),
            valid_step=valid_step,
            positive=positive,
        )
        committed = jax.tree_util.tree_map(
            lambda new, old: jnp.where(evidence.successful, new, old), candidate, state
        )
        return MeshfreeIncompressibleStepResult(committed, candidate, evidence)

    def _evidence(
        self,
        before: MeshfreeIncompressibleState,
        candidate: MeshfreeIncompressibleState,
        projections: tuple[IncompressibleProjectionResult, ...],
        cfl: Array,
        implicit_successful: Array,
        implicit_iterations: Array,
        cfl_admitted: Array,
        /,
        *,
        transport_admitted: Array | None = None,
        valid_step: Array | None = None,
        positive: Array | None = None,
    ) -> MeshfreeFlowEvidence:
        plan = self.plan
        volumes = plan.exterior.node_volumes
        final = projections[-1]
        refused = jnp.stack([~projection.successful for projection in projections])
        projection_status = jnp.where(
            jnp.any(refused),
            jnp.stack([projection.status for projection in projections])[
                jnp.argmax(refused)
            ],
            final.status,
        )
        cycles = self.classify_edge_velocity(final.candidate_velocity)
        reconstruction_ok = plan.velocity_reconstruction.admitted
        transport_ok = (
            jnp.asarray(True) if transport_admitted is None else transport_admitted
        )
        valid = jnp.asarray(True) if valid_step is None else valid_step
        positive_ = jnp.asarray(True) if positive is None else positive
        finite = (
            jnp.all(jnp.isfinite(candidate.velocity))
            & jnp.all(jnp.isfinite(candidate.density))
            & jnp.all(jnp.isfinite(candidate.volume_flux))
            & jnp.all(jnp.isfinite(candidate.pressure))
        )
        status = jnp.where(
            ~valid,
            int(MeshfreeFlowStatus.INVALID_STEP),
            jnp.where(
                ~finite,
                int(MeshfreeFlowStatus.NONFINITE),
                jnp.where(
                    ~positive_,
                    int(MeshfreeFlowStatus.NONPOSITIVE_DENSITY),
                    jnp.where(
                        ~cfl_admitted,
                        int(MeshfreeFlowStatus.CFL_REFUSED),
                        jnp.where(
                            ~transport_ok,
                            int(MeshfreeFlowStatus.TRANSPORT_REFUSED),
                            jnp.where(
                                ~implicit_successful,
                                int(MeshfreeFlowStatus.IMPLICIT_REFUSED),
                                jnp.where(
                                    jnp.any(refused),
                                    int(MeshfreeFlowStatus.PROJECTION_REFUSED),
                                    jnp.where(
                                        ~reconstruction_ok,
                                        int(MeshfreeFlowStatus.RECONSTRUCTION_REFUSED),
                                        jnp.where(
                                            ~cycles.physical,
                                            int(MeshfreeFlowStatus.SPURIOUS_CYCLES),
                                            int(MeshfreeFlowStatus.ACCEPTED),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        weights = volumes[:, None]
        return MeshfreeFlowEvidence(
            status=status,
            successful=status == int(MeshfreeFlowStatus.ACCEPTED),
            cfl=jnp.asarray(cfl, dtype=jnp.float64),
            transport_admitted=transport_ok,
            implicit_successful=implicit_successful,
            implicit_iterations=implicit_iterations,
            projection_status=projection_status,
            pressure_iterations=sum(
                (projection.linear.diagnostics.iterations for projection in projections),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            pressure_residual_norm=jnp.max(
                jnp.stack(
                    [projection.pressure_residual_norm for projection in projections]
                )
            ),
            graph_divergence_norm=final.divergence_defect_norm,
            nodal_divergence_before=_weighted_norm(
                volumes, self.nodal_divergence(before.velocity)
            ),
            nodal_divergence_after=_weighted_norm(
                volumes, self.nodal_divergence(candidate.velocity)
            ),
            reconstruction_condition=jnp.max(plan.velocity_reconstruction.condition),
            cycles=cycles,
            kinetic_energy_before=self.kinetic_energy(before),
            kinetic_energy_after=self.kinetic_energy(candidate),
            mass_before=jnp.sum(volumes * before.density),
            mass_after=jnp.sum(volumes * candidate.density),
            momentum_before=jnp.sum(
                weights * before.density[:, None] * before.velocity, axis=0
            ),
            momentum_after=jnp.sum(
                weights * candidate.density[:, None] * candidate.velocity, axis=0
            ),
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: MeshfreeIncompressibleState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index
        result = self.advance(state, time, step_size, args=args)
        evidence = result.evidence
        return FixedStepResult(
            result.candidate,
            result.state,
            evidence.successful,
            evidence.pressure_residual_norm,
            evidence.implicit_iterations + evidence.pressure_iterations.astype(jnp.int32),
            jnp.asarray(
                self.plan.tableau.stage_count + self.plan.projection_count,
                dtype=jnp.int32,
            ),
            jnp.asarray(False),
            jnp.zeros((), dtype=jnp.float64),
            evidence=evidence,
        )


def _weighted_norm(weights: Array, values: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(weights * values**2) / jnp.sum(weights))


__all__ = [
    "IncompressibleDensityModel",
    "MeshfreeCycleEvidence",
    "MeshfreeFlowEvidence",
    "MeshfreeFlowStatus",
    "MeshfreeIncompressibleFlowPlan",
    "MeshfreeIncompressibleState",
    "MeshfreeIncompressibleStepResult",
    "MeshfreeVelocityReconstruction",
    "PreparedMeshfreeIncompressibleFlow",
]
