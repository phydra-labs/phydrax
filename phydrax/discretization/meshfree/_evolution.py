#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Semidiscrete meshfree advection-diffusion-reaction problems.

``MeshfreeEvolutionPlan`` assembles the spatial residual of

    d(mu c)/dt = -div(c (u - w)) mu/V + div(K(c) grad c) + mu r(t, x, c)

on a prepared GMLS point cloud (collocation route) or on the conservative
radius graph (``ConservativeTransport``, content route) and publishes native
temporal contracts: the full, explicit and implicit drifts, differential,
split and DAE problems, an SSP fixed-step method publishing the admission
evidence of every attempt and an additive IMEX fixed-step method with native
implicit stage solves. It owns no time loop.

Measures are named, never relabeled. ``"quadrature-volume"`` is the Eulerian/ALE
measure ``V`` evolving with ``dV/dt = V div_h w``; ``"material-mass"`` is the
persistent particle mass of ``particle`` owners, constant under material
motion, while the ALE volume still evolves as the density carrier
``rho = m / V``. Fixed-support motion is stage-correct: every rate evaluation
refreshes the GMLS support at the stage coordinates, and a refused refresh
makes the rate nonfinite (fail closed) and the step admission refuse it. A
support change is an epoch boundary (``rebase`` re-anchors identical nodes;
population changes belong to the meshfree epoch/transfer owners).
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

from ..._fingerprint import canonical_fingerprint
from ..._numerics._ssp_runge_kutta import (
    ssprk33_step_with_evidence,
    ssprk54_step_with_evidence,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import DAEStructure, DifferentialAlgebraicSystem
from ...linalg import (
    ArraySpace,
    bind_numeric,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveTemplate,
    LinearSystem,
    prepare_linearization,
    prepare_template,
    solve,
    TolerancePolicy,
)
from ...nonlinear import (
    NewtonKrylov,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
    root,
)
from ...solver._balance_law_composition import (
    additive_imex_tableau,
    AdditiveIMEXScheme,
    AdditiveIMEXTableau,
)
from ...solver._conservation_temporal import (
    ConservationIMEXFixedStepMethod,
    ConservationIMEXMethod,
    ImplicitConservationStageResult,
)
from ...solver._differential import DifferentialProblem
from ...solver._differential_algebraic import DifferentialAlgebraicProblem
from ...solver._fixed_step import AbstractFixedStepMethod, FixedStepResult
from ...solver._split_differential import SplitDifferentialProblem
from ...sparse import gather_routes
from ...typing import (
    Bool,
    checked,
    Dim,
    Float64,
    Int32,
    Int64,
    parse,
    Scalar,
)
from .._point_cloud import PointCloudPlan, PreparedPointCloudDiscretization
from .._point_cloud_pde import (
    PointDiffusionForm,
    PointDiffusionOperator,
    PointDiffusivityKind,
)
from ..particle._core import ParticleDiscretization
from ..particle._population import ParticlePopulationState
from ..particle._verlet import ParticleVerletState, PreparedVerletParticleNeighborhood
from ..spatial import MortonAddressPlan
from ._stabilization import HyperviscosityPlan, PreparedHyperviscosity
from ._transport import ConservativeTransport, TransportCFL


class EvolutionPointDim(Dim):
    """Nodes of one semidiscrete meshfree evolution."""


class EvolutionRowDim(Dim):
    """Prescribed collocation rows of one evolution."""


EvolutionLayout: TypeAlias = Literal["concentration", "content", "moving-content"]
EvolutionMeasure: TypeAlias = Literal["quadrature-volume", "material-mass"]
MotionKind: TypeAlias = Literal["ale", "material"]
ReactionTreatment: TypeAlias = Literal["explicit", "implicit"]
InflowTreatment: TypeAlias = Literal["weak", "strong"]
EvolutionSSPMethod: TypeAlias = Literal["ssprk33", "ssprk54"]


class MeshfreeEvolutionStatus(IntEnum):
    ACCEPTED = 0
    NONFINITE = 1
    NEGATIVE_STATE = 2
    SUPPORT_EXCEEDED = 3
    NONPOSITIVE_MEASURE = 4
    NEIGHBOR_REBUILD_REQUIRED = 5
    METRIC_REFUSED = 6


@final
class MeshfreeDiffusionLaw(StrictModule):
    """Declared diffusivity ``K(c) = g(c) K0``; ``g`` is an optional nonlinear factor.

    ``K0`` is a scalar, nodal ``(points,)`` scalar field, or symmetric positive
    definite ``(d, d)``/``(points, d, d)`` tensor (collocation route only). A
    constitutive factor that is not finite and positive at a state makes the
    rate nonfinite; it is never clipped.
    """

    diffusivity: Array
    constitutive: Callable[[Array], ArrayLike] | None = eqx.field(static=True)
    kind: PointDiffusivityKind = eqx.field(static=True)
    form: PointDiffusionForm = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        diffusivity: ArrayLike,
        /,
        *,
        kind: PointDiffusivityKind = "scalar",
        constitutive: Callable[[Array], ArrayLike] | None = None,
        form: PointDiffusionForm = "collocated",
        law_id: str,
    ) -> None:
        if constitutive is not None and not callable(constitutive):
            raise TypeError("constitutive must be callable or None.")
        if not str(law_id):
            raise ValueError("law_id must be non-empty.")
        self.diffusivity = jnp.asarray(diffusivity, dtype=jnp.float64)
        self.constitutive = constitutive
        self.kind = parse(kind, PointDiffusivityKind, "kind")
        self.form = parse(form, PointDiffusionForm, "form")
        self.law_id = str(law_id)


@final
class MeshfreeReactionLaw(StrictModule):
    """Pointwise reaction ``r(t, x, c, args)`` per unit measure."""

    rate: Callable[[Array, Array, Array, Any], ArrayLike] = eqx.field(static=True)
    treatment: ReactionTreatment = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        rate: Callable[[Array, Array, Array, Any], ArrayLike],
        /,
        *,
        treatment: ReactionTreatment = "explicit",
        law_id: str,
    ) -> None:
        if not callable(rate):
            raise TypeError("Reaction rate must be callable.")
        if not str(law_id):
            raise ValueError("law_id must be non-empty.")
        self.rate = rate
        self.treatment = parse(treatment, ReactionTreatment, "treatment")
        self.law_id = str(law_id)


@final
class MeshfreeDirichletRows(StrictModule):
    """Collocation rows carrying prescribed values through their exact rate.

    The rate at ``rows`` is ``value_rate(t, x_rows, args) = dg/dt``; bulk
    operators do not act there. Integrating it reproduces the prescribed
    values to the temporal method's accuracy.
    """

    __strict_contract__ = True
    rows: Int32[EvolutionRowDim]
    value_rate: Callable[[Array, Array, Any], ArrayLike] = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        rows: ArrayLike,
        value_rate: Callable[[Array, Array, Any], ArrayLike],
        /,
        *,
        boundary_id: str,
    ) -> None:
        indices = np.asarray(rows)
        if (
            indices.ndim != 1
            or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError("Dirichlet rows must be unique integer row indices.")
        if not callable(value_rate):
            raise TypeError("value_rate must be callable.")
        if not str(boundary_id):
            raise ValueError("boundary_id must be non-empty.")
        self.rows = jnp.asarray(indices, dtype=jnp.int32)
        self.value_rate = value_rate
        self.boundary_id = str(boundary_id)


@final
class MaterialParticleMeasure(StrictModule):
    """Persistent particle mass and identity as the measure of a material cloud.

    Masses and identities come from the runtime ``ParticlePopulationState``
    (every slot active; identities are the 64-bit ``(id_hi, id_lo)`` pairs) and
    must equal the point cloud's stable ids: nodes are never relabeled.
    ``density`` maps mass to the initial ALE volume ``m / rho``. A Verlet
    neighbor cache, when supplied, gates fixed-support motion: a candidate
    that would rebuild the cached relation is an epoch boundary.
    """

    __strict_contract__ = True
    masses: Float64[EvolutionPointDim]
    density: Float64[EvolutionPointDim]
    identities: Int64[EvolutionPointDim]
    neighbors: PreparedVerletParticleNeighborhood | None
    reference: ParticleVerletState | None
    ambient_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        population: ParticlePopulationState,
        density: ArrayLike,
        /,
        *,
        neighbors: PreparedVerletParticleNeighborhood | None = None,
        reference: ParticleVerletState | None = None,
    ) -> None:
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if not isinstance(population, ParticlePopulationState):
            raise TypeError("population must be a ParticlePopulationState.")
        active = np.asarray(jax.device_get(population.active))
        masses = np.asarray(jax.device_get(population.mass), dtype=np.float64)
        if active.shape != (particles.capacity,) or not active.all():
            raise ValueError(
                "A material evolution measure needs every particle slot active."
            )
        if not np.all(np.isfinite(masses) & (masses > 0)):
            raise ValueError("Material particle masses must be finite and positive.")
        rho = np.broadcast_to(np.asarray(density, dtype=np.float64), masses.shape)
        if not np.all(np.isfinite(rho) & (rho > 0)):
            raise ValueError("Material density must be finite and positive.")
        if (neighbors is None) != (reference is None):
            raise ValueError("A Verlet cache needs both its plan and reference state.")
        if neighbors is not None and reference is not None:
            if not isinstance(neighbors, PreparedVerletParticleNeighborhood):
                raise TypeError("neighbors must be a PreparedVerletParticleNeighborhood.")
            if not isinstance(reference, ParticleVerletState):
                raise TypeError("reference must be a ParticleVerletState.")
            if reference.prepared_verlet_id != neighbors.prepared_id:
                raise ValueError("Verlet reference belongs to another neighbor plan.")
        high = np.asarray(jax.device_get(population.id_hi), dtype=np.uint64)
        low = np.asarray(jax.device_get(population.id_lo), dtype=np.uint64)
        self.masses = jnp.asarray(masses)
        self.density = jnp.asarray(rho)
        self.identities = jnp.asarray(((high << np.uint64(32)) | low).astype(np.int64))
        self.neighbors = neighbors
        self.reference = reference
        self.ambient_dimension = particles.ambient_dimension

    @property
    def reference_volumes(self) -> Array:
        return self.masses / self.density


@final
class MeshfreeMotion(StrictModule):
    """Declared node motion: ALE mesh velocity or material (Lagrangian) motion."""

    kind: MotionKind = eqx.field(static=True)
    mesh_velocity: Callable[[Array, Array, Any], ArrayLike] | None = eqx.field(
        static=True
    )
    material: MaterialParticleMeasure | None
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MotionKind,
        /,
        *,
        mesh_velocity: Callable[[Array, Array, Any], ArrayLike] | None = None,
        material: MaterialParticleMeasure | None = None,
        law_id: str,
    ) -> None:
        kind_ = parse(kind, MotionKind, "kind")
        match kind_:
            case "ale":
                if not callable(mesh_velocity) or material is not None:
                    raise ValueError(
                        "ALE motion takes one mesh velocity and no material."
                    )
            case "material":
                if mesh_velocity is not None or not isinstance(
                    material, MaterialParticleMeasure
                ):
                    raise ValueError(
                        "Material motion moves with the material velocity of a particle measure."
                    )
            case unknown:
                assert_never(unknown)
        if not str(law_id):
            raise ValueError("law_id must be non-empty.")
        self.kind = kind_
        self.mesh_velocity = mesh_velocity
        self.material = material
        self.law_id = str(law_id)


@final
class MeshfreeEvolutionFields(StrictModule):
    """Named fields of one packed evolution state."""

    concentration: Array
    content: Array
    measure: Array
    volumes: Array
    points: Array


@final
class MeshfreeEvolutionAdmission(StrictModule):
    """Accepted-step evidence of one candidate state; nothing is repaired."""

    __strict_contract__ = True
    status: Int32[Scalar]
    accepted: Bool[Scalar]
    finite: Bool[Scalar]
    nonnegative: Bool[Scalar]
    support_accepted: Bool[Scalar]
    measure_positive: Bool[Scalar]
    neighbors_reused: Bool[Scalar]
    minimum_concentration: Float64[Scalar]
    support_margin: Float64[Scalar]


@final
class MeshfreeEvolutionStepEvidence(StrictModule):
    """Native evidence of one SSP evolution step attempt, accepted or refused.

    ``admission`` is the candidate's step admission. ``transport_cfl`` is the
    forward-Euler certificate of the conservative graph route at the attempt's
    start time and step size; the collocation route has none (``None``). The
    structure depends only on the route, never on the outcome; a fail-closed
    candidate keeps its nonfinite extremes, nothing is sanitized.

    Over a coupling window ``MeshfreeEvolutionSSPMethod.reduce_evidence``
    reports the admission status of the refusing substep (else the accepted
    status), the conjunction of every admission and certificate flag, the
    minimum concentration and support margin, and the maximum CFL (minimum step
    bound) over executed substeps.
    """

    admission: MeshfreeEvolutionAdmission
    transport_cfl: TransportCFL | None


@final
class MeshfreeSpectralEstimate(StrictModule):
    """Power-iteration estimate of the implicit-part spectral radius.

    An estimate, not a bound: ``explicit_step(interval)`` divides a declared
    real stability interval by it and is no rigorous CFL certificate.
    """

    __strict_contract__ = True
    radius: Float64[Scalar]
    iterations: int = eqx.field(static=True)
    scope: Literal["power-iteration-estimate"] = eqx.field(static=True)

    def explicit_step(self, stability_interval: float, /) -> Array:
        interval = float(stability_interval)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("stability_interval must be finite and positive.")
        return interval / self.radius


@final
class MeshfreeEvolutionCapacity(StrictModule):
    """Static state layout and named measure of one prepared evolution."""

    __strict_contract__ = True
    stable_ids: Int64[EvolutionPointDim]
    support_trust: Float64[Scalar]
    state_size: int = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    layout: EvolutionLayout = eqx.field(static=True)
    measure: EvolutionMeasure = eqx.field(static=True)


@final
class MeshfreeEvolutionPlan(StrictModule):
    """Validated declaration of one semidiscrete meshfree ADR problem.

    ``spatial`` selects the route: a ``PreparedPointCloudDiscretization``
    (collocation; state is the concentration, or ``[content, volumes, points]``
    under motion) or a ``ConservativeTransport`` (graph; state is the nodal
    content ``V c``). ``velocity(t, x, args)`` is the material velocity;
    ``inflow(t, x, args)`` the boundary inflow state of a graph with boundary
    nodes. ``inflow_treatment="weak"`` relaxes inflow nodes toward it (upwind);
    ``"strong"`` holds genuine inflow nodes as prescribed rows following the
    inflow's exact time derivative, which keeps the interior reconstruction
    order (weak inflow lags by ``O(h)`` because boundary nodes carry no
    tangential edges). Declare ``"strong"`` only where the boundary is a true
    inflow: on a characteristic (tangential) boundary roundoff-signed flux
    would prescribe rows. Hyperviscosity is applied only when declared, never
    after a failure. ``positivity`` adds nonnegativity to step admission.
    """

    spatial: PreparedPointCloudDiscretization | ConservativeTransport
    diffusion: MeshfreeDiffusionLaw | None
    reaction: MeshfreeReactionLaw | None
    hyperviscosity: HyperviscosityPlan | None
    boundary: MeshfreeDirichletRows | None
    motion: MeshfreeMotion | None
    linear_policy: LinearSolvePolicy
    nonlinear_termination: NonlinearTermination
    velocity: Callable[[Array, Array, Any], ArrayLike] | None = eqx.field(static=True)
    inflow: Callable[[Array, Array, Any], ArrayLike] | None = eqx.field(static=True)
    inflow_treatment: InflowTreatment = eqx.field(static=True)
    positivity: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial: PreparedPointCloudDiscretization | ConservativeTransport,
        /,
        *,
        velocity: Callable[[Array, Array, Any], ArrayLike] | None = None,
        diffusion: MeshfreeDiffusionLaw | None = None,
        reaction: MeshfreeReactionLaw | None = None,
        hyperviscosity: HyperviscosityPlan | None = None,
        inflow: Callable[[Array, Array, Any], ArrayLike] | None = None,
        inflow_treatment: InflowTreatment = "weak",
        boundary: MeshfreeDirichletRows | None = None,
        motion: MeshfreeMotion | None = None,
        positivity: bool = False,
        linear_policy: LinearSolvePolicy | None = None,
        nonlinear_termination: NonlinearTermination | None = None,
        plan_id: str,
    ) -> None:
        if velocity is not None and not callable(velocity):
            raise TypeError("velocity must be callable or None.")
        if inflow is not None and not callable(inflow):
            raise TypeError("inflow must be callable or None.")
        for value, kind, name in (
            (diffusion, MeshfreeDiffusionLaw, "diffusion"),
            (reaction, MeshfreeReactionLaw, "reaction"),
            (hyperviscosity, HyperviscosityPlan, "hyperviscosity"),
            (boundary, MeshfreeDirichletRows, "boundary"),
            (motion, MeshfreeMotion, "motion"),
        ):
            if value is not None and not isinstance(value, kind):
                raise TypeError(f"{name} must be a {kind.__name__} or None.")
        match spatial:
            case PreparedPointCloudDiscretization():
                _validate_collocation(
                    spatial, diffusion, hyperviscosity, inflow, boundary, motion
                )
            case ConservativeTransport():
                _validate_graph(
                    spatial, velocity, diffusion, hyperviscosity, inflow, boundary, motion
                )
            case _:
                raise TypeError(
                    "spatial must be a PreparedPointCloudDiscretization or ConservativeTransport."
                )
        policy = (
            LinearSolvePolicy(
                GMRES(),
                tolerance=TolerancePolicy(relative=1e-12, absolute=1e-14, max_steps=400),
            )
            if linear_policy is None
            else linear_policy
        )
        if not isinstance(policy, LinearSolvePolicy) or policy.failure.mode != "status":
            raise ValueError(
                "Implicit stages need a status-returning native LinearSolvePolicy."
            )
        termination = (
            NonlinearTermination(
                absolute_residual=1e-11,
                relative_residual=0.0,
                absolute_step=0.0,
                relative_step=0.0,
                maximum_steps=30,
            )
            if nonlinear_termination is None
            else nonlinear_termination
        )
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("nonlinear_termination must be a NonlinearTermination.")
        if not str(plan_id):
            raise ValueError("plan_id must be non-empty.")
        treatment = parse(inflow_treatment, InflowTreatment, "inflow_treatment")
        if treatment == "strong" and inflow is None:
            raise ValueError("Strong inflow rows need a declared inflow state.")
        self.spatial = spatial
        self.diffusion = diffusion
        self.reaction = reaction
        self.hyperviscosity = hyperviscosity
        self.boundary = boundary
        self.motion = motion
        self.linear_policy = policy
        self.nonlinear_termination = termination
        self.velocity = velocity
        self.inflow = inflow
        self.inflow_treatment = treatment
        self.positivity = bool(positivity)
        self.plan_id = str(plan_id)

    def prepare(self) -> PreparedMeshfreeEvolution:
        return PreparedMeshfreeEvolution(self)


def _validate_collocation(
    cloud: PreparedPointCloudDiscretization,
    diffusion: MeshfreeDiffusionLaw | None,
    hyperviscosity: HyperviscosityPlan | None,
    inflow: Callable[[Array, Array, Any], ArrayLike] | None,
    boundary: MeshfreeDirichletRows | None,
    motion: MeshfreeMotion | None,
) -> None:
    count = cloud.state_shape[0]
    if inflow is not None:
        raise ValueError(
            "Collocation boundaries are prescribed rows; graph inflow is not declared here."
        )
    if boundary is not None and int(jnp.max(boundary.rows)) >= count:
        raise ValueError("Dirichlet rows must index the point cloud.")
    if motion is None:
        return
    if boundary is not None or hyperviscosity is not None:
        raise ValueError(
            "Moving clouds take neither prescribed rows nor anchored hyperviscosity."
        )
    if diffusion is not None and diffusion.form != "collocated":
        raise ValueError(
            "The dissipative pairing uses anchored quadrature; moving clouds use collocated diffusion."
        )
    material = motion.material
    if material is not None:
        if material.masses.shape != (count,) or (
            material.ambient_dimension != cloud.spatial_dimension
        ):
            raise ValueError("Material particles must be the cloud's nodes.")
        if not np.array_equal(
            np.asarray(jax.device_get(material.identities)),
            np.asarray(jax.device_get(cloud.stable_ids), dtype=np.int64),
        ):
            raise ValueError(
                "Particle identities must equal the cloud's stable ids; nodes are never relabeled."
            )


def _validate_graph(
    transport: ConservativeTransport,
    velocity: Callable[[Array, Array, Any], ArrayLike] | None,
    diffusion: MeshfreeDiffusionLaw | None,
    hyperviscosity: HyperviscosityPlan | None,
    inflow: Callable[[Array, Array, Any], ArrayLike] | None,
    boundary: MeshfreeDirichletRows | None,
    motion: MeshfreeMotion | None,
) -> None:
    if hyperviscosity is not None or boundary is not None or motion is not None:
        raise ValueError(
            "The conservative graph route takes no point hyperviscosity, rows or in-stage motion; "
            "refresh its transport between steps."
        )
    if diffusion is not None and (
        diffusion.kind != "scalar" or diffusion.diffusivity.ndim > 1
    ):
        raise ValueError("Graph diffusion takes a scalar or nodal scalar diffusivity.")
    needs_inflow = velocity is not None and transport.has_boundary
    if needs_inflow != (inflow is not None):
        raise ValueError(
            "A transported graph with boundary nodes requires exactly one declared inflow state."
        )


@final
class PreparedMeshfreeEvolution(StrictModule, NonTrainableState):
    """Prepared semidiscrete problem publishing native temporal contracts."""

    plan: MeshfreeEvolutionPlan
    diffusion_operator: PointDiffusionOperator | None
    hyperviscosity: PreparedHyperviscosity | None
    linear_template: LinearSolveTemplate
    route: Literal["collocation", "graph"] = eqx.field(static=True)
    layout: EvolutionLayout = eqx.field(static=True)
    measure_kind: EvolutionMeasure = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    linear_implicit: bool = eqx.field(static=True)
    evolution_id: str = eqx.field(static=True)

    def __init__(self, plan: MeshfreeEvolutionPlan, /) -> None:
        if not isinstance(plan, MeshfreeEvolutionPlan):
            raise TypeError("plan must be a MeshfreeEvolutionPlan.")
        spatial = plan.spatial
        law = plan.diffusion
        operator: PointDiffusionOperator | None = None
        hyper: PreparedHyperviscosity | None = None
        if isinstance(spatial, PreparedPointCloudDiscretization):
            route: Literal["collocation", "graph"] = "collocation"
            count = spatial.state_shape[0]
            dimension = spatial.spatial_dimension
            if law is not None:
                operator = PointDiffusionOperator(
                    spatial, law.diffusivity, form=law.form, kind=law.kind
                )
            if plan.hyperviscosity is not None:
                hyper = plan.hyperviscosity.prepare(spatial)
        else:
            route = "graph"
            count = spatial.node_volumes.shape[0]
            dimension = spatial.exterior.points.shape[1]
        motion = plan.motion
        layout: EvolutionLayout = (
            "moving-content"
            if motion is not None
            else "content"
            if route == "graph"
            else "concentration"
        )
        measure: EvolutionMeasure = (
            "material-mass"
            if motion is not None and motion.kind == "material"
            else "quadrature-volume"
        )
        size = count * (2 + dimension) if motion is not None else count
        identifier = canonical_fingerprint(
            {
                "kind": "meshfree-evolution",
                "plan": plan.plan_id,
                "route": route,
                "layout": layout,
                "measure": measure,
                "diffusion": None if law is None else law.law_id,
                "reaction": None
                if plan.reaction is None
                else [plan.reaction.law_id, plan.reaction.treatment],
                "motion": None if motion is None else [motion.kind, motion.law_id],
                "boundary": None if plan.boundary is None else plan.boundary.boundary_id,
                "inflow": plan.inflow_treatment if plan.inflow is not None else None,
            }
        )
        space = ArraySpace((count,), dtype=np.float64)
        structure = FunctionLinearOperator(
            lambda value: value, source=space, target=space, operator_id=identifier
        )
        self.plan = plan
        self.diffusion_operator = operator
        self.hyperviscosity = hyper
        self.linear_template = prepare_template(
            LinearSystem(structure, problem_id=identifier), plan.linear_policy
        )
        self.route = route
        self.layout = layout
        self.measure_kind = measure
        self.state_size = size
        self.point_count = count
        self.linear_implicit = (law is None or law.constitutive is None) and (
            plan.reaction is None or plan.reaction.treatment == "explicit"
        )
        self.evolution_id = identifier

    @property
    def dimension(self) -> int:
        spatial = self.plan.spatial
        if isinstance(spatial, PreparedPointCloudDiscretization):
            return spatial.spatial_dimension
        return spatial.exterior.points.shape[1]

    @property
    def capacity(self) -> MeshfreeEvolutionCapacity:
        spatial = self.plan.spatial
        if isinstance(spatial, PreparedPointCloudDiscretization):
            ids = spatial.stable_ids
            trust = jnp.min(spatial.trust_radius)
            edges = 0
        else:
            exterior = spatial.exterior
            ids = exterior.edge_relation.point_ids[exterior.capacity_indices]
            trust = jnp.asarray(exterior.topology_trust_margin, dtype=jnp.float64)
            edges = exterior.lengths.shape[0]
        return MeshfreeEvolutionCapacity(
            stable_ids=jnp.asarray(ids, dtype=jnp.int64),
            support_trust=jnp.asarray(trust, dtype=jnp.float64),
            state_size=self.state_size,
            point_count=self.point_count,
            edge_count=edges,
            dimension=self.dimension,
            layout=self.layout,
            measure=self.measure_kind,
        )

    def _anchored_points(self) -> Array:
        spatial = self.plan.spatial
        if isinstance(spatial, PreparedPointCloudDiscretization):
            return spatial.points
        return spatial.exterior.points

    def _anchored_volumes(self) -> Array:
        spatial = self.plan.spatial
        if isinstance(spatial, PreparedPointCloudDiscretization):
            return spatial.quadrature_weights
        return spatial.node_volumes

    def _material_masses(self) -> Array | None:
        motion = self.plan.motion
        if motion is None or motion.material is None:
            return None
        return motion.material.masses

    def initial_state(self, concentration: ArrayLike, /) -> Array:
        """Pack nodal concentrations into this evolution's state layout."""
        value = jnp.asarray(concentration, dtype=jnp.float64)
        if value.shape != (self.point_count,):
            raise ValueError("Initial concentration must be one nodal vector.")
        match self.layout:
            case "concentration":
                return value
            case "content":
                return self._anchored_volumes() * value
            case "moving-content":
                motion = self.plan.motion
                material = None if motion is None else motion.material
                if material is None:
                    volumes = self._anchored_volumes()
                    measure = volumes
                else:
                    volumes = material.reference_volumes
                    measure = material.masses
                return jnp.concatenate(
                    (measure * value, volumes, self._anchored_points().reshape(-1))
                )
            case unknown:
                assert_never(unknown)

    def fields(self, state: ArrayLike, /) -> MeshfreeEvolutionFields:
        """Unpack concentration, content, conserved measure, volumes and points."""
        value = self._state(state)
        count = self.point_count
        match self.layout:
            case "concentration":
                volumes = self._anchored_volumes()
                return MeshfreeEvolutionFields(
                    value, volumes * value, volumes, volumes, self._anchored_points()
                )
            case "content":
                volumes = self._anchored_volumes()
                return MeshfreeEvolutionFields(
                    value / volumes, value, volumes, volumes, self._anchored_points()
                )
            case "moving-content":
                content = value[:count]
                volumes = value[count : 2 * count]
                points = value[2 * count :].reshape((count, self.dimension))
                masses = self._material_masses()
                measure = volumes if masses is None else masses
                return MeshfreeEvolutionFields(
                    content / measure, content, measure, volumes, points
                )
            case unknown:
                assert_never(unknown)

    def total_content(self, state: ArrayLike, /) -> Array:
        return jnp.sum(self.fields(state).content)

    def explicit_rate(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> Array:
        """Advection, explicit reaction, prescribed-row and motion rates."""
        return self._rates(jnp.asarray(time), self._state(state), args)[0]

    def implicit_rate(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> Array:
        """Diffusion, hyperviscosity and implicit reaction rates (zero on rows)."""
        return self._rates(jnp.asarray(time), self._state(state), args)[1]

    def rate(self, time: ArrayLike, state: ArrayLike, args: Any = None, /) -> Array:
        """Full semidiscrete drift ``f(t, y, args)``."""
        explicit, implicit = self._rates(jnp.asarray(time), self._state(state), args)
        return explicit + implicit

    def _state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state, dtype=jnp.float64)
        if value.shape != (self.state_size,):
            raise ValueError("Evolution state does not match the prepared layout.")
        return value

    def _rates(self, time: Array, state: Array, args: Any, /) -> tuple[Array, Array]:
        match self.layout:
            case "concentration":
                return self._collocation_rates(time, state, args)
            case "content":
                return self._graph_rates(time, state, args)
            case "moving-content":
                return self._moving_rates(time, state, args)
            case unknown:
                assert_never(unknown)

    def _velocity(self, time: Array, points: Array, args: Any, /) -> Array:
        velocity = self.plan.velocity
        if velocity is None:
            return jnp.zeros_like(points)
        value = jnp.asarray(velocity(time, points, args), dtype=jnp.float64)
        if value.shape != points.shape:
            raise ValueError("Velocity must be nodal with shape (points, dimension).")
        return value

    def _reaction(
        self, time: Array, points: Array, concentration: Array, args: Any, /
    ) -> tuple[Array, Array]:
        """Explicit and implicit reaction rates per unit measure."""
        law = self.plan.reaction
        zero = jnp.zeros_like(concentration)
        if law is None:
            return zero, zero
        value = jnp.asarray(
            law.rate(time, points, concentration, args), dtype=jnp.float64
        )
        if value.shape != concentration.shape:
            raise ValueError("Reaction rate must be one nodal vector.")
        match law.treatment:
            case "explicit":
                return value, zero
            case "implicit":
                return zero, value
            case unknown:
                assert_never(unknown)

    def _coefficient(self, concentration: Array, /) -> Array:
        """Nodal ``K(c) = g(c) K0`` broadcast to the declared diffusivity kind."""
        law = self.plan.diffusion
        if law is None:
            raise ValueError("No diffusion law is declared.")
        base = law.diffusivity
        count = concentration.shape[0]
        match law.kind:
            case "scalar":
                field = jnp.broadcast_to(base, (count,))
            case "tensor":
                field = jnp.broadcast_to(base, (count,) + base.shape[-2:])
            case unknown:
                assert_never(unknown)
        if law.constitutive is None:
            return field
        factor = jnp.asarray(law.constitutive(concentration), dtype=jnp.float64)
        if factor.shape != concentration.shape:
            raise ValueError("Constitutive diffusivity factor must be one nodal vector.")
        return factor.reshape((count,) + (1,) * (field.ndim - 1)) * field

    def _point_diffusion(
        self,
        discretization: PreparedPointCloudDiscretization,
        concentration: Array,
        scale: Array | None,
        /,
    ) -> Array:
        """Point diffusion bound to stage geometry/coefficients; NaN if refused."""
        operator = self.diffusion_operator
        if operator is None:
            return jnp.zeros_like(concentration)
        field = self._coefficient(concentration)
        if scale is not None:
            field = scale.reshape((scale.shape[0],) + (1,) * (field.ndim - 1)) * field
        bound = operator.rebind(discretization, field)
        return jnp.where(bound.evidence.successful, bound.mv(concentration), jnp.nan)

    def _collocation_rates(
        self, time: Array, concentration: Array, args: Any, /
    ) -> tuple[Array, Array]:
        spatial = self.plan.spatial
        if not isinstance(spatial, PreparedPointCloudDiscretization):
            raise ValueError("Collocation rates need a point-cloud spatial owner.")
        points = spatial.points
        velocity = self._velocity(time, points, args)
        explicit = -spatial.divergence(concentration[:, None] * velocity)
        react_explicit, react_implicit = self._reaction(time, points, concentration, args)
        explicit = explicit + react_explicit
        implicit = self._point_diffusion(spatial, concentration, None) + react_implicit
        if self.hyperviscosity is not None:
            implicit = implicit + self.hyperviscosity.mv(concentration)
        boundary = self.plan.boundary
        if boundary is not None:
            rows = boundary.rows
            value = jnp.asarray(
                boundary.value_rate(time, points[rows], args), dtype=jnp.float64
            )
            if value.shape != rows.shape:
                raise ValueError("Prescribed row rates must match the declared rows.")
            explicit = explicit.at[rows].set(value)
            implicit = implicit.at[rows].set(0.0)
        return explicit, implicit

    def _graph_rates(
        self, time: Array, content: Array, args: Any, /
    ) -> tuple[Array, Array]:
        transport = self.plan.spatial
        if not isinstance(transport, ConservativeTransport):
            raise ValueError("Graph rates need a conservative transport owner.")
        exterior = transport.exterior
        volumes = transport.node_volumes
        points = exterior.points
        concentration = content / volumes
        explicit = jnp.zeros_like(content)
        if self.plan.velocity is not None:
            inflow = self.plan.inflow
            incoming: Array | None = None
            incoming_rate: Array | None = None
            if inflow is not None:
                moment = jnp.asarray(time, dtype=jnp.float64)

                def state(t: Array, /) -> Array:
                    return jnp.asarray(inflow(t, points, args), dtype=jnp.float64)

                match self.plan.inflow_treatment:
                    case "weak":
                        incoming = state(moment)
                    case "strong":
                        # Prescribed rows follow the declared law's exact time
                        # derivative: one JVP of the inflow state.
                        incoming, incoming_rate = jax.jvp(
                            state, (moment,), (jnp.ones_like(moment),)
                        )
                    case unknown:
                        assert_never(unknown)
            explicit = transport.rate(
                concentration,
                self._velocity(time, points, args),
                inflow=incoming,
                inflow_rate=incoming_rate,
            ).content_rate
        react_explicit, react_implicit = self._reaction(time, points, concentration, args)
        explicit = explicit + volumes * react_explicit
        implicit = volumes * react_implicit
        if self.plan.diffusion is not None:
            nodal = self._coefficient(concentration)
            edges = exterior.lengths.shape[0]
            endpoints = gather_routes(exterior.incidence.relation, nodal).reshape(
                (edges, 2)
            )
            total = endpoints[:, 0] + endpoints[:, 1]
            # Harmonic edge average, the exterior diffusion law's default.
            edge = (
                2.0 * endpoints[:, 0] * endpoints[:, 1] / jnp.where(total > 0, total, 1.0)
            )
            conductance = exterior.metric_result.weights * edge
            rate = -exterior.stiffness(conductance).mv(concentration)
            implicit = implicit + jnp.where(jnp.all(nodal > 0), rate, jnp.nan)
        return explicit, implicit

    def _moving_rates(
        self, time: Array, state: Array, args: Any, /
    ) -> tuple[Array, Array]:
        spatial = self.plan.spatial
        motion = self.plan.motion
        if not isinstance(spatial, PreparedPointCloudDiscretization) or motion is None:
            raise ValueError("Moving rates need a point cloud and declared motion.")
        fields = self.fields(state)
        refresh = spatial.refresh(fields.points)
        stage = refresh.discretization
        points = fields.points
        volumes = fields.volumes
        concentration = fields.concentration
        velocity = self._velocity(time, points, args)
        react_explicit, react_implicit = self._reaction(time, points, concentration, args)
        match motion.kind:
            case "ale":
                if motion.mesh_velocity is None:
                    raise ValueError("ALE motion lacks its mesh velocity.")
                mesh = jnp.asarray(
                    motion.mesh_velocity(time, points, args), dtype=jnp.float64
                )
                if mesh.shape != points.shape:
                    raise ValueError("Mesh velocity must be nodal (points, dimension).")
                transported = -volumes * stage.divergence(
                    concentration[:, None] * (velocity - mesh)
                )
                scale = None
            case "material":
                mesh = velocity
                transported = jnp.zeros_like(concentration)
                scale = fields.measure / volumes
            case unknown:
                assert_never(unknown)
        measure = fields.measure
        explicit = jnp.concatenate(
            (
                transported + measure * react_explicit,
                volumes * stage.divergence(mesh),
                mesh.reshape(-1),
            )
        )
        diffusion = volumes * self._point_diffusion(stage, concentration, scale)
        implicit = jnp.concatenate(
            (
                diffusion + measure * react_implicit,
                jnp.zeros_like(volumes),
                jnp.zeros_like(mesh.reshape(-1)),
            )
        )
        # A refused fixed-support refresh is fail-closed: no content or volume
        # rate exists there. The declared node motion stays defined, so the
        # candidate coordinates witness where the support trust was exhausted.
        block = 2 * self.point_count
        refused = jnp.where(refresh.accepted, 1.0, jnp.nan)
        return (
            explicit.at[:block].multiply(refused),
            implicit.at[:block].multiply(refused),
        )

    def admission(self, state: ArrayLike, /) -> MeshfreeEvolutionAdmission:
        """Step-admission evidence of one candidate state (nothing is repaired)."""
        value = self._state(state)
        fields = self.fields(value)
        finite = jnp.all(jnp.isfinite(value))
        nonnegative = jnp.all(fields.concentration >= 0)
        measure_positive = jnp.all(fields.volumes > 0) & jnp.all(fields.measure > 0)
        support = jnp.asarray(True)
        margin = jnp.asarray(jnp.inf, dtype=jnp.float64)
        reused = jnp.asarray(True)
        metric = jnp.asarray(True)
        spatial = self.plan.spatial
        if isinstance(spatial, ConservativeTransport):
            metric = spatial.exterior.metric_result.accepted
        if self.layout == "moving-content" and isinstance(
            spatial, PreparedPointCloudDiscretization
        ):
            located = jnp.all(jnp.isfinite(fields.points))
            refresh = spatial.refresh(jnp.where(located, fields.points, spatial.points))
            support = refresh.accepted | ~located
            margin = jnp.min(refresh.support_margin)
            motion = self.plan.motion
            material = None if motion is None else motion.material
            if (
                material is not None
                and material.neighbors is not None
                and material.reference is not None
            ):
                cache = material.neighbors.update(fields.points, material.reference)
                reused = cache.successful & ~cache.rebuilt
        positive = nonnegative | (not self.plan.positivity)
        # The support/neighbor refusal is the root cause of the nonfinite
        # content a fail-closed stage produces, so it is reported first.
        status = jnp.select(
            (
                ~support,
                ~reused,
                ~finite,
                ~metric,
                ~measure_positive,
                ~positive,
            ),
            (
                int(MeshfreeEvolutionStatus.SUPPORT_EXCEEDED),
                int(MeshfreeEvolutionStatus.NEIGHBOR_REBUILD_REQUIRED),
                int(MeshfreeEvolutionStatus.NONFINITE),
                int(MeshfreeEvolutionStatus.METRIC_REFUSED),
                int(MeshfreeEvolutionStatus.NONPOSITIVE_MEASURE),
                int(MeshfreeEvolutionStatus.NEGATIVE_STATE),
            ),
            int(MeshfreeEvolutionStatus.ACCEPTED),
        ).astype(jnp.int32)
        return MeshfreeEvolutionAdmission(
            status=status,
            accepted=status == int(MeshfreeEvolutionStatus.ACCEPTED),
            finite=finite,
            nonnegative=nonnegative,
            support_accepted=support,
            measure_positive=measure_positive,
            neighbors_reused=reused,
            minimum_concentration=jnp.min(fields.concentration),
            support_margin=margin,
        )

    def transport_cfl(
        self, time: ArrayLike, state: ArrayLike, step_size: ArrayLike, args: Any = None, /
    ) -> TransportCFL:
        """Forward-Euler transport certificate at one state (graph route only)."""
        transport = self.plan.spatial
        if not isinstance(transport, ConservativeTransport):
            raise ValueError(
                "Collocation advection has no CFL certificate; use the graph route."
            )
        self._state(state)
        points = transport.exterior.points
        return transport.cfl(self._velocity(jnp.asarray(time), points, args), step_size)

    def spectral_estimate(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        iterations: int = 24,
    ) -> MeshfreeSpectralEstimate:
        """Power-iteration radius of the implicit part's Jacobian at one state."""
        count = int(iterations)
        if count != iterations or count < 1:
            raise ValueError("iterations must be a positive integer.")
        value = self._state(state)
        moment = jnp.asarray(time)
        block = value[: self.point_count]
        linear = prepare_linearization(
            lambda part: self.implicit_rate(
                moment, value.at[: self.point_count].set(part), args
            )[: self.point_count],
            block,
        )
        tiny = jnp.finfo(block.dtype).tiny
        start = jnp.sin(jnp.arange(block.size, dtype=block.dtype) + 0.5)

        def normalized(vector: Array) -> Array:
            return vector / jnp.maximum(jnp.linalg.norm(vector), tiny)

        def advance(_: int, vector: Array) -> Array:
            return normalized(linear.pushforward(vector))

        vector = jax.lax.fori_loop(0, count, advance, normalized(start))
        return MeshfreeSpectralEstimate(
            radius=jnp.linalg.norm(linear.pushforward(vector)),
            iterations=count,
            scope="power-iteration-estimate",
        )

    def differential_problem(
        self,
        initial_state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        args: Any = None,
    ) -> DifferentialProblem:
        """Explicit ODE contract (SSP-RK, Rosenbrock, adaptive Diffrax)."""
        return DifferentialProblem(
            self._drift(),
            self._state(initial_state),
            t0=t0,
            t1=t1,
            args=args,
            problem_id=self.evolution_id,
        )

    def split_problem(
        self,
        initial_state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        args: Any = None,
    ) -> SplitDifferentialProblem:
        """Additive explicit/implicit contract for split solvers."""
        explicit, implicit = self._split_drifts()
        return SplitDifferentialProblem(
            explicit,
            implicit,
            self._state(initial_state),
            t0=t0,
            t1=t1,
            args=args,
            problem_id=self.evolution_id,
        )

    def dae_problem(
        self, initial_state: ArrayLike, /, *, args: Any = None
    ) -> DifferentialAlgebraicProblem:
        """Implicit residual ``y' - f(t, y)`` for native BDF solves."""
        drift = self._drift()
        system = DifferentialAlgebraicSystem(
            lambda time, state, state_rate, context: (
                state_rate - drift(time, state, context)
            ),
            state_shape=(self.state_size,),
            structure=DAEStructure(("differential",), component_axis=None),
            system_id=self.evolution_id,
        )
        return DifferentialAlgebraicProblem(
            system,
            self._state(initial_state),
            args=args,
            problem_id=self.evolution_id,
        )

    def ssprk_method(
        self, method: EvolutionSSPMethod = "ssprk33", /
    ) -> MeshfreeEvolutionSSPMethod:
        """Fixed-step SSP method gated by and publishing this evolution's admission."""
        return MeshfreeEvolutionSSPMethod(self, method)

    def imex_method(
        self, method: AdditiveIMEXScheme | AdditiveIMEXTableau, /
    ) -> ConservationIMEXFixedStepMethod:
        """Additive IMEX fixed-step method with native implicit stage solves.

        Linear implicit parts use the prepared native linear template; a
        constitutive diffusivity or implicit reaction uses native Newton-Krylov.
        The method's validator is this evolution's admission.
        """
        tableau = (
            method
            if isinstance(method, AdditiveIMEXTableau)
            else additive_imex_tableau(parse(method, AdditiveIMEXScheme, "method"))
        )
        if tableau.part_count != 1:
            raise ValueError("Meshfree evolutions publish exactly one implicit part.")
        explicit, implicit = self._split_drifts()
        return ConservationIMEXFixedStepMethod(
            ConservationIMEXMethod(
                tableau,
                explicit,
                implicit,
                self._implicit_solver(),
                validator=lambda candidate: self.admission(candidate).accepted,
                method_id=self.evolution_id,
            )
        )

    def rebase(self, state: ArrayLike, /) -> PreparedMeshfreeEvolution:
        """Re-anchor identical nodes at the state's coordinates (new support epoch).

        The same stable ids, order, stencil policy, boundary data and address are
        prepared at the current coordinates with the current ALE volumes as
        quadrature; periodic coordinates are wrapped into the address cell. The
        packed state is unchanged apart from that wrap. Population changes are
        not rebases: they need a meshfree epoch transfer.
        """
        spatial = self.plan.spatial
        if self.layout != "moving-content" or not isinstance(
            spatial, PreparedPointCloudDiscretization
        ):
            raise ValueError(
                "Only moving collocation evolutions re-anchor their support."
            )
        fields = self.fields(state)
        if not bool(self.admission(state).accepted):
            raise ValueError("Only an admitted state can anchor a new support epoch.")
        cloud = spatial.plan
        address = cloud.address
        periodic = any(address.periodic_axes)
        points = _wrapped(fields.points, address) if periodic else fields.points
        prepared = PointCloudPlan(
            np.asarray(jax.device_get(points)),
            np.asarray(jax.device_get(fields.volumes)),
            boundary_mask=cloud.boundary_mask,
            boundary_normals=cloud.boundary_normals,
            boundary_quadrature_weights=cloud.boundary_quadrature_weights,
            stencil=cloud.stencil,
            neighbors=cloud.neighbors,
            point_ids=cloud.point_ids,
            address=address if periodic else None,
            maximum_candidates=cloud.maximum_candidates,
            target_chunk_size=cloud.target_chunk_size,
        ).prepare()
        motion = self.plan.motion
        material = None if motion is None else motion.material
        if motion is not None and material is not None and material.neighbors is not None:
            reference = material.neighbors.initialize(points)
            motion = eqx.tree_at(lambda item: item.material.reference, motion, reference)
        plan = eqx.tree_at(
            lambda item: (item.spatial, item.motion),
            self.plan,
            (prepared, motion),
            is_leaf=lambda leaf: leaf is None,
        )
        return PreparedMeshfreeEvolution(plan)

    def repacked(self, state: ArrayLike, /) -> Array:
        """The state of a ``rebase`` successor (periodic coordinates wrapped)."""
        value = self._state(state)
        spatial = self.plan.spatial
        if self.layout != "moving-content" or not isinstance(
            spatial, PreparedPointCloudDiscretization
        ):
            raise ValueError(
                "Only moving collocation evolutions re-anchor their support."
            )
        address = spatial.plan.address
        if not any(address.periodic_axes):
            return value
        count = self.point_count
        points = _wrapped(self.fields(value).points, address)
        return value.at[2 * count :].set(points.reshape(-1))

    def _drift(self) -> Callable[[Array, Array, Any], Array]:
        def drift(time: Array, state: Array, args: Any, /) -> Array:
            return self.rate(time, state, args)

        return drift

    def _split_drifts(
        self,
    ) -> tuple[
        Callable[[Array, Array, Any], Array], Callable[[Array, Array, Any], Array]
    ]:
        def explicit(time: Array, state: Array, args: Any, /) -> Array:
            return self.explicit_rate(time, state, args)

        def implicit(time: Array, state: Array, args: Any, /) -> Array:
            return self.implicit_rate(time, state, args)

        return explicit, implicit

    def _implicit_solver(
        self,
    ) -> Callable[[Array, Array, Array, Any], ImplicitConservationStageResult]:
        count = self.point_count
        space = ArraySpace((count,), dtype=np.float64)
        identifier = self.evolution_id

        def block_rate(time: Array, state: Array, block: Array, args: Any, /) -> Array:
            return self.implicit_rate(time, state.at[:count].set(block), args)[:count]

        def linear(
            provisional: Array, time: Array, coefficient: Array, args: Any, /
        ) -> ImplicitConservationStageResult:
            operator = FunctionLinearOperator(
                lambda block: (
                    block - coefficient * block_rate(time, provisional, block, args)
                ),
                source=space,
                target=space,
                operator_id=identifier,
            )
            prepared = bind_numeric(
                self.linear_template, LinearSystem(operator, problem_id=identifier)
            )
            result = solve(prepared, provisional[:count])
            return ImplicitConservationStageResult(
                provisional.at[:count].set(result.value),
                jnp.all(result.successful),
                jnp.max(result.diagnostics.iterations).astype(jnp.int32),
                jnp.max(result.diagnostics.residual_norm),
                jnp.max(result.status).astype(jnp.int32),
            )

        problem = NonlinearSystemProblem(
            lambda block, context: (
                block
                - context[2] * block_rate(context[0], context[1], block, context[3])
                - context[1][:count]
            ),
            state_space=space,
            residual_space=space,
            problem_id=identifier,
        )
        method = NewtonKrylov(linear_policy=self.plan.linear_policy)
        termination = self.plan.nonlinear_termination

        def nonlinear(
            provisional: Array, time: Array, coefficient: Array, args: Any, /
        ) -> ImplicitConservationStageResult:
            result = root(
                problem,
                provisional[:count],
                method=method,
                termination=termination,
                args=(time, provisional, coefficient, args),
            )
            return ImplicitConservationStageResult(
                provisional.at[:count].set(result.state),
                result.status == int(NonlinearStatus.SUCCESS),
                result.diagnostics.iterations.astype(jnp.int32),
                result.diagnostics.final_residual_norm,
                result.status.astype(jnp.int32),
            )

        return linear if self.linear_implicit else nonlinear


@final
class MeshfreeEvolutionSSPMethod(AbstractFixedStepMethod, NonTrainableState):
    """SSP fixed-step method whose accepted step is the evolution's admission.

    The candidate is the raw SSP step of the evolution's drift and is accepted
    exactly when ``PreparedMeshfreeEvolution.admission`` accepts it; a refused
    candidate keeps the previous state. No clipping, limiting or stabilization
    is applied after the fact. Every attempt publishes
    ``MeshfreeEvolutionStepEvidence`` with an outcome-independent structure.
    """

    evolution: PreparedMeshfreeEvolution
    method: EvolutionSSPMethod = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        evolution: PreparedMeshfreeEvolution,
        method: EvolutionSSPMethod = "ssprk33",
        /,
    ) -> None:
        selected = parse(method, EvolutionSSPMethod, "method")
        self.evolution = evolution
        self.method = selected
        self.method_id = canonical_fingerprint(
            {
                "kind": "meshfree-evolution-ssprk",
                "method": selected,
                "evolution": evolution.evolution_id,
            }
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index
        evolution = self.evolution
        match self.method:
            case "ssprk33":
                advanced = ssprk33_step_with_evidence(
                    evolution._drift(), time, state, step_size, args
                )
                order = 3
            case "ssprk54":
                advanced = ssprk54_step_with_evidence(
                    evolution._drift(), time, state, step_size, args
                )
                order = 4
            case unknown:
                assert_never(unknown)
        candidate = advanced.state
        admission = evolution.admission(candidate)
        successful = advanced.successful & admission.accepted
        return FixedStepResult(
            candidate,
            jnp.where(successful, candidate, state),
            successful,
            jnp.zeros((), dtype=state.dtype),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(order, dtype=jnp.int32),
            advanced.applied,
            advanced.correction_norm,
            evidence=MeshfreeEvolutionStepEvidence(
                admission=admission,
                transport_cfl=(
                    evolution.transport_cfl(time, state, step_size, args)
                    if isinstance(evolution.plan.spatial, ConservativeTransport)
                    else None
                ),
            ),
        )

    def reduce_evidence(
        self,
        evidence: object,
        executed: Array,
        successful: Array,
        /,
    ) -> MeshfreeEvolutionStepEvidence:
        """Window evidence of consecutive substeps with admission semantics.

        The status is that of the refusing substep, or the accepted status when
        every executed substep was accepted; flags are conjunctions and
        concentration, support margin and step bound are minima, CFL numbers
        maxima, over executed substeps.
        """
        if not isinstance(evidence, MeshfreeEvolutionStepEvidence):
            raise TypeError("Evolution evidence must be MeshfreeEvolutionStepEvidence.")
        refused = executed & ~successful
        # The first substep always runs, so index 0 is the accepted status when
        # no executed substep was refused.
        terminal = jnp.where(jnp.any(refused), jnp.argmax(refused), 0)

        def every(flags: Array) -> Array:
            return jnp.all(jnp.where(executed, flags, True))

        def least(values: Array) -> Array:
            return jnp.min(jnp.where(executed, values, jnp.inf))

        admission = evidence.admission
        cfl = evidence.transport_cfl
        return MeshfreeEvolutionStepEvidence(
            admission=MeshfreeEvolutionAdmission(
                status=admission.status[terminal],
                accepted=every(admission.accepted),
                finite=every(admission.finite),
                nonnegative=every(admission.nonnegative),
                support_accepted=every(admission.support_accepted),
                measure_positive=every(admission.measure_positive),
                neighbors_reused=every(admission.neighbors_reused),
                minimum_concentration=least(admission.minimum_concentration),
                support_margin=least(admission.support_margin),
            ),
            transport_cfl=(
                None
                if cfl is None
                else TransportCFL(
                    node_cfl=jnp.max(
                        jnp.where(executed[:, None], cfl.node_cfl, 0.0), axis=0
                    ),
                    cfl=jnp.max(jnp.where(executed, cfl.cfl, 0.0)),
                    step_bound=least(cfl.step_bound),
                    certified=every(cfl.certified),
                    admitted=every(cfl.admitted),
                )
            ),
        )


def _wrapped(points: Array, address: MortonAddressPlan, /) -> Array:
    """Periodic coordinates mapped into the address cell; other axes unchanged."""
    lower = jnp.asarray(address.lower, dtype=points.dtype)
    upper = jnp.asarray(address.upper, dtype=points.dtype)
    periodic = jnp.asarray(address.periodic_axes)
    wrapped = lower + jnp.mod(points - lower, upper - lower)
    return jnp.where(periodic, wrapped, points)


__all__ = [
    "EvolutionLayout",
    "EvolutionMeasure",
    "EvolutionSSPMethod",
    "MaterialParticleMeasure",
    "MeshfreeDiffusionLaw",
    "MeshfreeDirichletRows",
    "MeshfreeEvolutionAdmission",
    "MeshfreeEvolutionCapacity",
    "MeshfreeEvolutionFields",
    "MeshfreeEvolutionPlan",
    "MeshfreeEvolutionSSPMethod",
    "MeshfreeEvolutionStatus",
    "MeshfreeEvolutionStepEvidence",
    "MeshfreeMotion",
    "MeshfreeReactionLaw",
    "MeshfreeSpectralEstimate",
    "MotionKind",
    "PreparedMeshfreeEvolution",
    "InflowTreatment",
    "ReactionTreatment",
]
