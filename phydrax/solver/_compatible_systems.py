#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    CochainDiscretization,
    DiagonalHodge,
    StructuredCochainBridge,
)
from ..linalg import (
    AbstractLinearOperator,
    adjoint,
    assemble_sparse,
    ComposedLinearOperator,
    FunctionLinearOperator,
    GaussSeidelPreconditionerBuilder,
    hodge_laplacian,
    JacobiPreconditionerBuilder,
    KernelCertificate,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSubspace,
    LinearSystem,
    MultigridPreconditioner,
    MultigridSetupDiagnostics,
    NullspacePolicy,
    OperatorProperties,
    PreconditionerProperties,
    PreconditioningPolicy,
    prepare,
    PreparedLinearSolve,
    ProjectedPCG,
    refresh,
    SmoothedAggregationHierarchyBuilder,
    SmoothedAggregationPolicy,
    solve,
    TolerancePolicy,
)
from ..sparse import EdgeRelation, SparseCoordinateOperator
from ..typing import parse


CompatiblePressurePreconditioner: TypeAlias = Literal["none", "smoothed-aggregation"]


class CompatibleElasticityState(StrictModule):
    displacement: Array
    velocity: Array


class CompatibleElasticityDynamics(StrictModule, NonTrainableState):
    """Compatible scalar/vector elastic-wave reference on degree-zero cochains."""

    bridge: StructuredCochainBridge
    wave_speed: float = eqx.field(static=True)
    components: int = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        wave_speed: float = 1.0,
        components: int = 1,
    ) -> None:
        speed = float(wave_speed)
        components_ = int(components)
        if (
            not isinstance(bridge, StructuredCochainBridge)
            or speed <= 0.0
            or components_ <= 0
        ):
            raise ValueError("Elasticity bridge, wave speed, and components are invalid.")
        self.bridge = bridge
        self.wave_speed = speed
        self.components = components_
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "compatible-elasticity-dynamics",
                "bridge": bridge.bridge_id,
                "wave_speed": speed,
                "components": components_,
            }
        )

    def _apply_components(
        self, function: Callable[[Array], Array], values: Array, /
    ) -> Array:
        return (
            function(values)
            if self.components == 1
            else jax.vmap(function, in_axes=1, out_axes=1)(values)
        )

    def pack(
        self, displacement: ArrayLike, velocity: ArrayLike, /
    ) -> CompatibleElasticityState:
        displacement_ = jnp.asarray(displacement)
        velocity_ = jnp.asarray(velocity)
        shape = (self.bridge.cochain.cell_counts[0],) + (
            () if self.components == 1 else (self.components,)
        )
        if displacement_.shape != shape or velocity_.shape != shape:
            raise ValueError(
                "Elasticity state must match degree-zero cochain/components."
            )
        return CompatibleElasticityState(displacement=displacement_, velocity=velocity_)

    def stiffness(self, displacement: ArrayLike, /) -> Array:
        value = jnp.asarray(displacement)
        return self._apply_components(
            lambda component: self.bridge.codifferential(
                1,
                self.bridge.exterior_derivative(0, component),
            ),
            value,
        )

    def drift(self, state: CompatibleElasticityState, /) -> CompatibleElasticityState:
        return self.pack(
            state.velocity,
            -(self.wave_speed**2) * self.stiffness(state.displacement),
        )

    def energy(self, state: CompatibleElasticityState, /) -> Array:
        h0 = self.bridge.cochain.hodge_diagonal(0)
        kinetic = jnp.sum(
            h0.reshape((-1,) + (1,) * (state.velocity.ndim - 1)) * state.velocity**2
        )
        gradient = self._apply_components(
            lambda component: self.bridge.exterior_derivative(0, component),
            state.displacement,
        )
        h1 = self.bridge.cochain.hodge_diagonal(1)
        potential = self.wave_speed**2 * jnp.sum(
            h1.reshape((-1,) + (1,) * (gradient.ndim - 1)) * gradient**2
        )
        return 0.5 * (kinetic + potential)


class CompatibleProjectionStatus(IntEnum):
    """Fail-closed projection acceptance, ordered by reporting precedence."""

    SUCCESS = 0
    NONFINITE = 1
    INCOMPATIBLE_SOURCE = 2
    SOLVE_FAILED = 3
    RESIDUAL_TOO_LARGE = 4


@final
class IncompressibleProjectionResult(StrictModule):
    """Committed projection, the retained candidate, and native pressure evidence.

    ``velocity`` and ``divergence_after`` hold the candidate only when ``status``
    is SUCCESS and the unchanged input otherwise; a rejected ``pressure`` is NaN,
    never a plausible zero gauge. Residual, divergence, compatibility, and gauge
    norms use the degree-zero Hodge pairing on the active absolute complex.
    ``nullity`` is the exact number of connected components of its one-skeleton.
    ``acceptance_threshold`` bounds those norms: the solve tolerance on the load
    plus the floating-point floor ``10 eps ||M0⁻¹ |d|ᵀ |M1 u|||`` of evaluating
    the input's divergence, which a small load on a large solenoidal velocity
    (an incremental pressure correction) cannot go below.
    ``preconditioner`` names the pressure preconditioner; ``multigrid`` is the
    native hierarchy setup evidence (level dimensions, grid and operator
    complexity) for ``"smoothed-aggregation"`` and ``None`` otherwise.
    """

    velocity: Array
    pressure: Array
    candidate_velocity: Array
    candidate_pressure: Array
    divergence_before: Array
    divergence_after: Array
    target_divergence: Array
    edge_coefficients: Array
    pressure_residual: Array
    pressure_residual_norm: Array
    divergence_defect_norm: Array
    rhs_norm: Array
    acceptance_threshold: Array
    compatibility_residual: Array
    gauge_residual: Array
    kernel_valid: Array
    linear: LinearSolveResult
    status: Array
    multigrid: MultigridSetupDiagnostics | None
    nullity: int = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    preconditioner: CompatiblePressurePreconditioner = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(CompatibleProjectionStatus.SUCCESS)


def _resolve_cochain(
    complex: CochainDiscretization | StructuredCochainBridge, /
) -> CochainDiscretization:
    match complex:
        case CochainDiscretization():
            cochain = complex
        case StructuredCochainBridge():
            cochain = complex.cochain
        case _:
            raise TypeError(
                "Compatible projection requires a CochainDiscretization or "
                "StructuredCochainBridge."
            )
    if cochain.dimension < 1:
        raise ValueError(
            "Compatible projection requires degree-zero and degree-one cochains."
        )
    return cochain


def _active_one_skeleton(
    cochain: CochainDiscretization, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Host topology preparation: active edge endpoints, orientation, and d0 components.

    Returns realization vertex/edge indices, endpoints in realization and compact
    vertex coordinates, the matching incidence signs, component labels, and count.
    """
    vertices = np.asarray(cochain.active_indices(0, boundary="absolute"))
    edges = np.asarray(cochain.active_indices(1, boundary="absolute"))
    vertex_position = np.full((cochain.cell_counts[0],), -1, dtype=np.int64)
    vertex_position[vertices] = np.arange(vertices.size, dtype=np.int64)
    edge_position = np.full((cochain.cell_counts[1],), -1, dtype=np.int64)
    edge_position[edges] = np.arange(edges.size, dtype=np.int64)
    incidence = cochain.topology.incidences[0]
    relation = incidence.relation
    valid = np.asarray(relation.valid)
    lower = np.asarray(relation.source_indices)[valid]
    upper = edge_position[np.asarray(relation.target_indices)[valid]]
    signs = np.asarray(incidence.signs, dtype=np.float64)[valid]
    kept = upper >= 0
    lower, upper, signs = lower[kept], upper[kept], signs[kept]
    if np.any(np.bincount(upper, minlength=edges.size) != 2):
        raise ValueError("Every active edge must bound exactly two vertex incidences.")
    order = np.argsort(upper, kind="stable")
    endpoints = lower[order].reshape(edges.size, 2)
    compact = vertex_position[endpoints]
    if np.any(compact < 0):
        raise ValueError("Active edges must connect active vertices.")
    graph = coo_matrix(
        (np.ones((edges.size,), dtype=np.int8), (compact[:, 0], compact[:, 1])),
        shape=(vertices.size, vertices.size),
    )
    count, labels = connected_components(graph, directed=False)
    return (
        vertices,
        edges,
        endpoints,
        compact,
        signs[order].reshape(edges.size, 2),
        labels,
        count,
    )


@final
class _WeightedPressureAction(StrictModule):
    """δ(w d p) in compact coordinates; self-adjoint for a diagonal degree-one Hodge."""

    differential: AbstractLinearOperator
    coefficients: Array

    def __call__(self, pressure: Array, /) -> Array:
        target = self.differential.target
        gradient = target.flatten(self.differential.mv(pressure))
        return self.differential.adjoint_mv(
            target.unflatten(self.coefficients * gradient)
        )


def _weighted_pressure_operator(
    cochain: CochainDiscretization, coefficients: Array, operator_id: str, /
) -> AbstractLinearOperator:
    hilbert = cochain.hilbert_complex(boundary="absolute")
    space = hilbert.space(0)
    return FunctionLinearOperator(
        _WeightedPressureAction(hilbert.differential(0), coefficients),
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id=operator_id,
    )


def _pressure_policy(
    policy: LinearSolvePolicy | None, cochain: CochainDiscretization, /
) -> LinearSolvePolicy:
    if policy is None:
        # CG attains roughly eps*cond in floating point; the step budget leaves
        # headroom beyond the exact-arithmetic bound of one step per unknown.
        unknowns = cochain.active_indices(0, boundary="absolute").size
        return LinearSolvePolicy(
            ProjectedPCG(),
            tolerance=TolerancePolicy(
                relative=1.0e-12, absolute=1.0e-14, max_steps=2 * unknowns + 10
            ),
            require_device_binding=True,
        )
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("solve_policy must be a LinearSolvePolicy or None.")
    if not isinstance(policy.method, ProjectedPCG):
        raise ValueError(
            "Compatible pressure solves use ProjectedPCG on the certified kernel complement."
        )
    return policy


def _smoothed_aggregation_preconditioner(
    cochain: CochainDiscretization,
    compact_endpoints: np.ndarray,
    signs: np.ndarray,
    kernel: LinearSubspace,
    policy: LinearSolvePolicy,
    problem_id: str,
    /,
) -> MultigridPreconditioner:
    """Prepare the native smoothed-aggregation V-cycle for δd at unit edge weights.

    The explicit operator is assembled from the oriented active incidence by native
    sparse assembly; the adjoint carries the diagonal degree-one and degree-zero
    Hodges, so the hierarchy acts on the solve's own Hodge-paired space and its
    pairing-aware transfers keep the cycle self-adjoint there. The per-component
    constants are the near-nullspace candidates and are reproduced exactly by every
    smoothed prolongator.

    Every coupling of a graph Laplacian is strong (threshold zero): with the
    default 0.25 threshold the smoothed coarse stencils lose strong neighbors and
    coarsening stalls. Level smoothing is damped Jacobi with ω = 1/2, a vectorized
    action; on the fine graph Laplacian λmax(D⁻¹A) ≤ 2, so it contracts. Coarse
    Galerkin levels carry no such bound, so positive definiteness of the cycle is
    asserted, and the outer solve's independent residual acceptance remains the
    check. The coarsest level uses a symmetric Gauss–Seidel sweep, which is
    positive definite for every positive-semidefinite operator with a positive
    diagonal.
    """
    if not (
        isinstance(cochain.hodges[0], DiagonalHodge)
        and isinstance(cochain.hodges[1], DiagonalHodge)
    ):
        raise ValueError(
            "Smoothed-aggregation pressure preconditioning requires diagonal "
            "degree-zero and degree-one Hodges."
        )
    hilbert = cochain.hilbert_complex(boundary="absolute")
    space, edge_space = hilbert.space(0), hilbert.space(1)
    if np.any(np.bincount(compact_endpoints.reshape(-1), minlength=space.size) == 0):
        raise ValueError(
            "Smoothed-aggregation pressure preconditioning requires every active "
            "vertex to bound an active edge."
        )
    edges = compact_endpoints.shape[0]
    differential = SparseCoordinateOperator(
        EdgeRelation(
            compact_endpoints.reshape(-1).astype(np.int32),
            np.repeat(np.arange(edges, dtype=np.int32), 2),
            source_size=space.size,
            target_size=edges,
        ),
        jnp.asarray(signs.reshape(-1), dtype=jnp.float64),
        source=space,
        target=edge_space,
        operator_id=f"{problem_id}:incidence",
    )
    storage = assemble_sparse(
        ComposedLinearOperator(adjoint(differential), differential)
    ).sparse_storage()
    reference = SparseCoordinateOperator(
        EdgeRelation(
            np.asarray(storage.indices, dtype=np.int32),
            np.repeat(
                np.arange(space.size, dtype=np.int32),
                np.diff(np.asarray(storage.indptr, dtype=np.int64)),
            ),
            source_size=space.size,
            target_size=space.size,
        ),
        storage.values,
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id=f"{problem_id}:reference-laplacian",
    )
    builder = SmoothedAggregationHierarchyBuilder(
        SmoothedAggregationPolicy(strength_threshold=0.0),
        JacobiPreconditionerBuilder(relaxation=0.5),
        GaussSeidelPreconditionerBuilder(direction="symmetric"),
        near_nullspaces=(kernel,),
        properties=PreconditionerProperties(
            linear=True,
            stationary=True,
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "linear": "construction",
                "stationary": "construction",
                "self_adjoint": "construction",
                "positive_definite": "asserted",
            },
        ),
    )
    return MultigridPreconditioner(
        builder.prepare_hierarchy(reference, materialization=policy.materialization)
    )


@final
class _CompatiblePressureSystem(StrictModule, NonTrainableState):
    """Prepared degree-zero pressure Poisson solve on one active absolute complex.

    The right and left kernel is the exact per-component constant subspace of d0,
    taken from the connected components of the active one-skeleton. Every positive
    edge weighting shares it, so the certificate is structural and survives
    numeric refreshes. Solves run native ProjectedPCG in the degree-zero Hodge
    pairing with the minimum-norm gauge. Incompatible loads are projected by the
    native policy and reported; the consumer refuses them.

    A smoothed-aggregation preconditioner is prepared once, at unit edge weights,
    as a frozen left action in the same Hodge-paired space; ``multigrid`` keeps
    its native setup evidence.
    """

    cochain: CochainDiscretization
    vertex_indices: Array
    edge_indices: Array
    edge_endpoints: Array
    compact_endpoints: Array
    certificate: KernelCertificate
    prepared: PreparedLinearSolve
    multigrid: MultigridSetupDiagnostics | None
    nullity: int = eqx.field(static=True)
    preconditioner: CompatiblePressurePreconditioner = eqx.field(static=True)

    def __init__(
        self,
        cochain: CochainDiscretization,
        operator: AbstractLinearOperator,
        policy: LinearSolvePolicy,
        preconditioner: CompatiblePressurePreconditioner,
        problem_id: str,
        /,
    ) -> None:
        vertices, edges, endpoints, compact, signs, labels, count = _active_one_skeleton(
            cochain
        )
        space = cochain.hilbert_complex(boundary="absolute").space(0)
        basis = (labels[:, None] == np.arange(count)[None, :]).astype(np.float64)
        kernel = LinearSubspace(space, basis, subspace_id=f"{problem_id}:kernel")
        certificate = KernelCertificate(
            operator,
            kernel,
            evidence="construction",
            scope="structural",
            complete=True,
            tolerance=1.0e-10,
        )
        match preconditioner:
            case "none":
                multigrid = None
            case "smoothed-aggregation":
                if policy.preconditioning is not None:
                    raise ValueError(
                        "preconditioner='smoothed-aggregation' owns the pressure "
                        "preconditioning; solve_policy must not declare one."
                    )
                action = _smoothed_aggregation_preconditioner(
                    cochain, compact, signs, kernel, policy, problem_id
                )
                policy = eqx.tree_at(
                    lambda value: value.preconditioning,
                    policy,
                    PreconditioningPolicy(action),
                    is_leaf=lambda value: value is None,
                )
                multigrid = action.hierarchy.diagnostics
            case _:
                assert_never(preconditioner)
        problem = LinearSystem(
            operator,
            nullspace_policy=NullspacePolicy(
                certificate=certificate, compatibility="project", gauge="minimum-norm"
            ),
            problem_id=problem_id,
        )
        self.cochain = cochain
        self.vertex_indices = jnp.asarray(vertices, dtype=jnp.int32)
        self.edge_indices = jnp.asarray(edges, dtype=jnp.int32)
        self.edge_endpoints = jnp.asarray(endpoints, dtype=jnp.int32)
        self.compact_endpoints = jnp.asarray(compact, dtype=jnp.int32)
        self.certificate = certificate
        self.prepared = prepare(problem, policy)
        self.multigrid = multigrid
        self.nullity = count
        self.preconditioner = preconditioner

    def _norm(self, values: Array, /) -> Array:
        space = self.cochain.hilbert_complex(boundary="absolute").space(0)
        vector = space.unflatten(values)
        return jnp.sqrt(jnp.maximum(jnp.real(space.inner(vector, vector)), 0.0))

    def _divergence_scale(self, velocity: Array, /) -> Array:
        """Hodge norm of the unsigned divergence ``M0⁻¹ |d|ᵀ |M1 u|``.

        It is the magnitude summed when ``δu`` is evaluated, so ``eps`` times
        it is the floating-point floor of an independently evaluated
        divergence, however small ``δu`` itself is.
        """
        hilbert = self.cochain.hilbert_complex(boundary="absolute")
        space, edge_space = hilbert.space(0), hilbert.space(1)
        flux = jnp.abs(edge_space.riesz(velocity[self.edge_indices]))
        magnitude = (
            jnp.zeros((space.size,), dtype=flux.dtype)
            .at[self.compact_endpoints[:, 0]]
            .add(flux)
            .at[self.compact_endpoints[:, 1]]
            .add(flux)
        )
        return self._norm(jnp.abs(space.inverse_riesz(magnitude)))

    def project(
        self,
        velocity: Array,
        target_divergence: Array | None,
        coefficients: Array,
        refreshed_operator: AbstractLinearOperator | None,
        projection_id: str,
        /,
    ) -> IncompressibleProjectionResult:
        """Solve once; ``refreshed_operator`` rebinds coefficients on the same template."""
        cochain = self.cochain
        hilbert = cochain.hilbert_complex(boundary="absolute")
        space, edge_space = hilbert.space(0), hilbert.space(1)
        divergence_before = cochain.codifferential(1, velocity)
        target = (
            jnp.zeros_like(divergence_before)
            if target_divergence is None
            else target_divergence
        )
        load = (divergence_before - target)[self.vertex_indices]
        if refreshed_operator is None:
            prepared = self.prepared
        else:
            prepared = refresh(
                self.prepared,
                LinearSystem(
                    refreshed_operator,
                    nullspace_policy=self.prepared.problem.nullspace_policy,
                    problem_id=self.prepared.problem.problem_id,
                ),
            )
        operator = prepared.problem.operator
        linear = solve(prepared, space.unflatten(load))
        pressure = space.flatten(linear.value)
        residual = space.flatten(operator.mv(space.unflatten(pressure))) - load
        gradient = edge_space.flatten(
            hilbert.differential(0).mv(space.unflatten(pressure))
        )
        candidate_velocity = velocity - jnp.zeros_like(velocity).at[
            self.edge_indices
        ].set(coefficients * gradient)
        candidate_pressure = (
            jnp.zeros_like(divergence_before).at[self.vertex_indices].set(pressure)
        )
        candidate_divergence = cochain.codifferential(1, candidate_velocity)
        residual_norm = self._norm(residual)
        defect_norm = self._norm((candidate_divergence - target)[self.vertex_indices])
        rhs_norm = self._norm(load)
        tolerance = self.prepared.plan.policy.tolerance
        eps = float(jnp.finfo(load.dtype).eps)
        roundoff = 10.0 * eps * load.size
        # The checks evaluate δu-sized sums independently of the solve, so
        # besides the solve tolerance on the load they carry the evaluation
        # floor of δu: a solenoidal addition to the velocity leaves the load
        # unchanged but raises that floor.
        threshold = (
            tolerance.absolute
            + max(tolerance.relative, roundoff) * rhs_norm
            + 10.0 * eps * self._divergence_scale(velocity)
        )
        compatibility = linear.diagnostics.compatibility_residual
        finite = (
            jnp.all(jnp.isfinite(candidate_velocity))
            & jnp.all(jnp.isfinite(pressure))
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(defect_norm)
        )
        status = jnp.where(
            ~finite,
            int(CompatibleProjectionStatus.NONFINITE),
            jnp.where(
                compatibility > threshold,
                int(CompatibleProjectionStatus.INCOMPATIBLE_SOURCE),
                jnp.where(
                    ~linear.successful | ~self.certificate.valid,
                    int(CompatibleProjectionStatus.SOLVE_FAILED),
                    jnp.where(
                        (residual_norm > threshold) | (defect_norm > threshold),
                        int(CompatibleProjectionStatus.RESIDUAL_TOO_LARGE),
                        int(CompatibleProjectionStatus.SUCCESS),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        accepted = status == int(CompatibleProjectionStatus.SUCCESS)
        return IncompressibleProjectionResult(
            velocity=jnp.where(accepted, candidate_velocity, velocity),
            pressure=jnp.where(accepted, candidate_pressure, jnp.nan),
            candidate_velocity=candidate_velocity,
            candidate_pressure=candidate_pressure,
            divergence_before=divergence_before,
            divergence_after=jnp.where(accepted, candidate_divergence, divergence_before),
            target_divergence=target,
            edge_coefficients=jnp.zeros_like(velocity)
            .at[self.edge_indices]
            .set(coefficients),
            pressure_residual=jnp.zeros_like(divergence_before)
            .at[self.vertex_indices]
            .set(residual),
            pressure_residual_norm=residual_norm,
            divergence_defect_norm=defect_norm,
            rhs_norm=rhs_norm,
            acceptance_threshold=threshold,
            compatibility_residual=compatibility,
            gauge_residual=linear.diagnostics.gauge_residual,
            kernel_valid=self.certificate.valid,
            linear=linear,
            status=status,
            multigrid=self.multigrid,
            nullity=self.nullity,
            precision=jnp.dtype(load.dtype).name,
            projection_id=projection_id,
            preconditioner=self.preconditioner,
        )


def _cochain_input(
    cochain: CochainDiscretization, degree: int, values: ArrayLike, name: str, /
) -> Array:
    value = jnp.asarray(values, dtype=jnp.float64)
    if value.shape != (cochain.cell_counts[degree],):
        raise ValueError(f"{name} must be a degree-{degree} cochain.")
    return value


def _target_input(
    cochain: CochainDiscretization, values: ArrayLike | None, /
) -> Array | None:
    return (
        None
        if values is None
        else _cochain_input(cochain, 0, values, "target_divergence")
    )


@final
class CompatibleIncompressibleProjection(StrictModule, NonTrainableState):
    """Degree-one projection u - d p with δ d p = δ u - s on a cell cochain complex.

    Accepts the canonical ``CochainDiscretization`` (for example a meshfree positive
    one-complex) or a ``StructuredCochainBridge`` through its owned cochain. The
    pressure Poisson problem uses the absolute (natural) boundary; its exact
    per-component constant kernel fixes the minimum-norm gauge. A graph solenoidal
    projection is a discrete constraint, not a qualified Navier-Stokes velocity.

    ``preconditioner="smoothed-aggregation"`` prepares the native
    smoothed-aggregation V-cycle once at construction (host setup); it requires
    diagonal degree-zero and degree-one Hodges and no isolated active vertex, and
    excludes a ``solve_policy`` preconditioner. ``"none"`` (the default) solves
    unpreconditioned.
    """

    pressure_system: _CompatiblePressureSystem
    projection_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex: CochainDiscretization | StructuredCochainBridge,
        /,
        *,
        solve_policy: LinearSolvePolicy | None = None,
        preconditioner: CompatiblePressurePreconditioner = "none",
    ) -> None:
        selected = parse(
            preconditioner, CompatiblePressurePreconditioner, "preconditioner"
        )
        cochain = _resolve_cochain(complex)
        policy = _pressure_policy(solve_policy, cochain)
        identifier = canonical_fingerprint(
            {
                "kind": "compatible-incompressible-projection",
                "realization": cochain.realization_id,
                "numeric_revision": cochain.numeric_revision,
            }
        )
        operator = hodge_laplacian(cochain.hilbert_complex(boundary="absolute"), 0)
        self.pressure_system = _CompatiblePressureSystem(
            cochain, operator, policy, selected, f"{identifier}:pressure"
        )
        self.projection_id = identifier

    def project(
        self,
        velocity: ArrayLike,
        /,
        *,
        target_divergence: ArrayLike | None = None,
    ) -> IncompressibleProjectionResult:
        """Project onto δu = target_divergence (zero when omitted).

        On every closed component the target must integrate to zero against the
        degree-zero Hodge measure; otherwise the result is INCOMPATIBLE_SOURCE.
        """
        system = self.pressure_system
        value = _cochain_input(system.cochain, 1, velocity, "velocity")
        return system.project(
            value,
            _target_input(system.cochain, target_divergence),
            jnp.ones(system.edge_indices.shape, dtype=value.dtype),
            None,
            self.projection_id,
        )


class CompatibleIdealMHDState(StrictModule):
    magnetic: Array


class CompatibleIdealMHDInductionDynamics(StrictModule):
    """Constrained magnetic induction B'=-dE with caller-supplied ideal Ohm field."""

    bridge: StructuredCochainBridge
    electromotive_circulation: Callable[[Array, Array, Any], ArrayLike]
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        electromotive_circulation: Callable[[Array, Array, Any], ArrayLike],
        /,
    ) -> None:
        if (
            not isinstance(bridge, StructuredCochainBridge)
            or bridge.dimension != 3
            or not callable(electromotive_circulation)
        ):
            raise ValueError(
                "Compatible ideal-MHD induction requires a 3D bridge and electric field."
            )
        self.bridge = bridge
        self.electromotive_circulation = electromotive_circulation
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "compatible-ideal-mhd-induction",
                "bridge": bridge.bridge_id,
                "electric_field": repr(electromotive_circulation),
            }
        )

    def pack(self, magnetic: ArrayLike, /) -> CompatibleIdealMHDState:
        value = jnp.asarray(magnetic)
        if value.shape != (self.bridge.cochain.cell_counts[2],):
            raise ValueError("MHD magnetic field must be a degree-two cochain.")
        return CompatibleIdealMHDState(magnetic=value)

    def drift(
        self,
        time: Array,
        state: CompatibleIdealMHDState,
        args: Any = None,
    ) -> CompatibleIdealMHDState:
        electric = jnp.asarray(self.electromotive_circulation(time, state.magnetic, args))
        if electric.shape != (self.bridge.cochain.cell_counts[1],):
            raise ValueError("Ideal Ohm electric field must be a degree-one cochain.")
        return self.pack(-self.bridge.exterior_derivative(1, electric))

    def step(
        self,
        time: Array,
        state: CompatibleIdealMHDState,
        step_size: ArrayLike,
        args: Any = None,
    ) -> CompatibleIdealMHDState:
        return self.pack(
            state.magnetic
            + jnp.asarray(step_size) * self.drift(time, state, args).magnetic
        )

    def magnetic_constraint(self, state: CompatibleIdealMHDState, /) -> Array:
        return self.bridge.exterior_derivative(2, state.magnetic)


@final
class CompatibleVariableDensityProjection(StrictModule, NonTrainableState):
    """Variable-density projection u - ρₑ⁻¹ d p with δ(ρₑ⁻¹ d p) = δ u - s.

    The owner interpolates edge density as the arithmetic mean of its two active
    endpoint densities and reports ρₑ⁻¹ as ``edge_coefficients``. Densities must be
    finite and positive on active vertices. The weighted operator is self-adjoint
    in the degree-zero Hodge pairing only for a diagonal degree-one Hodge, which is
    therefore required. Each call rebinds the prepared native solve; the
    structural constant-kernel certificate is shared by every positive density.

    ``preconditioner="smoothed-aggregation"`` prepares the native
    smoothed-aggregation V-cycle once for unit edge weights and keeps it frozen
    across densities: the solve stays traceable, and the action remains a
    symmetric positive-definite preconditioner for every positive density, but its
    spectral equivalence degrades with the edge-coefficient contrast
    max(ρₑ⁻¹)/min(ρₑ⁻¹), so iteration counts grow with density contrast.
    """

    pressure_system: _CompatiblePressureSystem
    projection_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex: CochainDiscretization | StructuredCochainBridge,
        /,
        *,
        solve_policy: LinearSolvePolicy | None = None,
        preconditioner: CompatiblePressurePreconditioner = "none",
    ) -> None:
        selected = parse(
            preconditioner, CompatiblePressurePreconditioner, "preconditioner"
        )
        cochain = _resolve_cochain(complex)
        if not isinstance(cochain.hodges[1], DiagonalHodge):
            raise ValueError(
                "Variable-density projection requires a diagonal degree-one Hodge."
            )
        policy = _pressure_policy(solve_policy, cochain)
        identifier = canonical_fingerprint(
            {
                "kind": "compatible-variable-density-projection",
                "realization": cochain.realization_id,
                "numeric_revision": cochain.numeric_revision,
            }
        )
        edges = cochain.active_indices(1, boundary="absolute")
        reference = _weighted_pressure_operator(
            cochain,
            jnp.ones(edges.shape, dtype=jnp.float64),
            f"{identifier}:weighted-pressure",
        )
        self.pressure_system = _CompatiblePressureSystem(
            cochain, reference, policy, selected, f"{identifier}:pressure"
        )
        self.projection_id = identifier

    def edge_inverse_density(self, density: ArrayLike, /) -> Array:
        """Owned edge coefficient ``ρₑ⁻¹`` on active edges for a vertex density.

        It is the coefficient of the projection's pressure gradient
        ``ρₑ⁻¹ d p``, so consumers that apply a known pressure gradient use the
        same interpolation as the projection.
        """
        system = self.pressure_system
        value = _cochain_input(system.cochain, 0, density, "density")
        active = value[system.vertex_indices]
        value = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(active)) | jnp.any(active <= 0.0),
            "Projection density must be finite and positive on active vertices.",
        )
        endpoints = system.edge_endpoints
        return 2.0 / (value[endpoints[:, 0]] + value[endpoints[:, 1]])

    def project(
        self,
        velocity: ArrayLike,
        density: ArrayLike,
        /,
        *,
        target_divergence: ArrayLike | None = None,
    ) -> IncompressibleProjectionResult:
        system = self.pressure_system
        value = _cochain_input(system.cochain, 1, velocity, "velocity")
        coefficients = self.edge_inverse_density(density)
        operator = _weighted_pressure_operator(
            system.cochain,
            coefficients,
            system.prepared.problem.operator.operator_id,
        )
        return system.project(
            value,
            _target_input(system.cochain, target_divergence),
            coefficients,
            operator,
            self.projection_id,
        )


class CompatiblePoroelasticState(StrictModule):
    displacement: Array
    velocity: Array
    pressure: Array


class CompatiblePoroelasticDynamics(StrictModule, NonTrainableState):
    """Compatible scalar Biot-like elastic/pressure coupling on degree-zero forms."""

    bridge: StructuredCochainBridge
    wave_speed: float = eqx.field(static=True)
    hydraulic_diffusivity: float = eqx.field(static=True)
    coupling: float = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        wave_speed: float = 1.0,
        hydraulic_diffusivity: float = 0.1,
        coupling: float = 0.2,
    ) -> None:
        if (
            not isinstance(bridge, StructuredCochainBridge)
            or not np.isfinite(wave_speed)
            or not np.isfinite(hydraulic_diffusivity)
            or not np.isfinite(coupling)
            or wave_speed <= 0.0
            or hydraulic_diffusivity < 0.0
        ):
            raise ValueError("Poroelastic coefficients must be finite and physical.")
        self.bridge = bridge
        self.wave_speed = float(wave_speed)
        self.hydraulic_diffusivity = float(hydraulic_diffusivity)
        self.coupling = float(coupling)

    def drift(self, state: CompatiblePoroelasticState, /) -> CompatiblePoroelasticState:
        laplace_u = self.bridge.hodge_laplacian(0, state.displacement)
        laplace_p = self.bridge.hodge_laplacian(0, state.pressure)
        laplace_v = self.bridge.hodge_laplacian(0, state.velocity)
        return CompatiblePoroelasticState(
            displacement=state.velocity,
            velocity=-(self.wave_speed**2) * laplace_u + self.coupling * laplace_p,
            pressure=-self.hydraulic_diffusivity * laplace_p - self.coupling * laplace_v,
        )


class CompatibleThermoelasticState(StrictModule):
    displacement: Array
    velocity: Array
    temperature: Array


class CompatibleThermoelasticDynamics(StrictModule, NonTrainableState):
    """Compatible scalar thermoelastic wave/heat reference."""

    bridge: StructuredCochainBridge
    wave_speed: float = eqx.field(static=True)
    thermal_diffusivity: float = eqx.field(static=True)
    expansion: float = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        wave_speed: float = 1.0,
        thermal_diffusivity: float = 0.1,
        expansion: float = 0.2,
    ) -> None:
        if (
            not isinstance(bridge, StructuredCochainBridge)
            or not np.isfinite(wave_speed)
            or not np.isfinite(thermal_diffusivity)
            or not np.isfinite(expansion)
            or wave_speed <= 0.0
            or thermal_diffusivity < 0.0
        ):
            raise ValueError("Thermoelastic coefficients must be finite and physical.")
        self.bridge = bridge
        self.wave_speed = float(wave_speed)
        self.thermal_diffusivity = float(thermal_diffusivity)
        self.expansion = float(expansion)

    def drift(
        self,
        state: CompatibleThermoelasticState,
        /,
    ) -> CompatibleThermoelasticState:
        laplace_u = self.bridge.hodge_laplacian(0, state.displacement)
        laplace_t = self.bridge.hodge_laplacian(0, state.temperature)
        laplace_v = self.bridge.hodge_laplacian(0, state.velocity)
        return CompatibleThermoelasticState(
            displacement=state.velocity,
            velocity=-(self.wave_speed**2) * laplace_u + self.expansion * laplace_t,
            temperature=-self.thermal_diffusivity * laplace_t
            - self.expansion * laplace_v,
        )


__all__ = [
    "CompatibleElasticityDynamics",
    "CompatibleElasticityState",
    "CompatibleIdealMHDInductionDynamics",
    "CompatibleIdealMHDState",
    "CompatibleIncompressibleProjection",
    "CompatiblePoroelasticDynamics",
    "CompatiblePoroelasticState",
    "CompatiblePressurePreconditioner",
    "CompatibleProjectionStatus",
    "CompatibleThermoelasticDynamics",
    "CompatibleThermoelasticState",
    "CompatibleVariableDensityProjection",
    "IncompressibleProjectionResult",
]
