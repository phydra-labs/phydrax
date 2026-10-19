"""Log-domain randomized-shortest-path free energies and expected costs.

For an absorbing target ``t`` with static reach set ``R(t)`` (nodes with a
directed route path to ``t``), the free energy ``phi`` of the Boltzmann bag of
hitting paths solves the soft Bellman equation

    phi_i = -(1/theta) log sum_{i->j, j in R(t)} p_ij exp(-theta (c_ij + phi_j)),

with ``phi_t = 0``. Its Jacobian ``I - Pi`` is the transient generator of the
optimal (Doob-transformed) random walk. The log-domain formulation avoids
partition-function dynamic range, but near-closed walks can still be poorly
conditioned. The expected cost solves ``(I - Pi) cbar = Pi c``.
"""

from enum import IntEnum
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from .._admissibility import first_order_only
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..combinatorial import ShortestPathFieldMethod, ShortestPathFieldPlan
from ..linalg import (
    ArraySpace,
    GMRES,
    LinearSolvePolicy,
    LinearSystem,
    PreconditioningPolicy,
    prepare,
    PreparedLinearSolve,
    refresh,
    solve,
    SparseFactorizationPolicy,
    SparseFactorizationPreconditionerBuilder,
    TolerancePolicy,
)
from ..nonlinear import (
    JacobianPolicy,
    NewtonKrylov,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
    solve_prepared_nonlinear,
)
from ..sparse import (
    EdgeRelation,
    RelationExecutionPlan,
    RelationExecutionState,
    SparseCoordinateOperator,
)
from ..sparse._linear import SparseStoragePlan
from ..typing import Bool, checked, Dim, Float, Int32, parse, Scalar, Size


class _NodeDim(Dim):
    """Graph nodes."""


class _RouteDim(Dim):
    """Directed transition routes."""


class _QueryDim(Dim):
    """Static query nodes, used as both path sources and absorbing targets."""


class RandomizedShortestPathStatus(IntEnum):
    """Portable outcome of randomized-shortest-path execution."""

    SUCCESS = 0
    INVALID_INPUT = 1
    NONLINEAR_FAILED = 2
    LINEAR_SOLVE_FAILED = 3


@final
class RandomizedShortestPathResult(StrictModule):
    """Directed free energies and expected costs with per-target evidence.

    ``free_energy[a, b]`` and ``expected_cost[a, b]`` describe hitting paths
    from ``nodes[a]`` to the absorbing target ``nodes[b]``; they are ``+inf``
    when ``nodes[a]`` cannot reach ``nodes[b]`` and zero on the diagonal. A
    failed target's column is NaN.
    """

    __strict_contract__ = True

    free_energy: Float[_QueryDim, _QueryDim]
    expected_cost: Float[_QueryDim, _QueryDim]
    reachable: Bool[_QueryDim, _QueryDim]
    status: Int32[Scalar]
    nonlinear_status: Int32[_QueryDim]
    newton_iterations: Int32[_QueryDim]
    residual_norm: Float[_QueryDim]
    linear_status: Int32[_QueryDim]

    @property
    def maximum_derivative_order(self) -> int:
        """Immutable first-order-only AD contract; nested transforms are refused."""
        return 1

    @property
    def free_energy_distance(self) -> Array:
        """Symmetrized free-energy distance ``(phi_ab + phi_ba) / 2``."""
        return 0.5 * (self.free_energy + self.free_energy.T)

    @property
    def expected_cost_dissimilarity(self) -> Array:
        """Symmetrized randomized-shortest-path dissimilarity."""
        return 0.5 * (self.expected_cost + self.expected_cost.T)


@final
class _TargetArguments(StrictModule):
    """Per-target dynamic residual data; Boolean masks carry no tangent."""

    log_reference: Array
    cost: Array
    inverse_temperature: Array
    scale: Array
    active: Array
    kept: Array
    leak: Array


@final
class _FreeEnergySystem(StrictModule):
    """Fixed-topology soft-Bellman residual, transition weights, and generator.

    Prepared solve templates retain this substrate's bound methods, so it is
    owned separately from the plan to keep the plan PyTree acyclic.
    """

    __strict_contract__ = True

    by_source: RelationExecutionState
    jacobian_relation: EdgeRelation
    storage: SparseStoragePlan
    space: ArraySpace
    route_source: Int32[_RouteDim]
    route_target: Int32[_RouteDim]
    node_count: Size[_NodeDim] = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def source_sum(self, values: Array, /) -> Array:
        """Deterministic per-source-node sum of route values."""
        out, _ = self.by_source.reduce(
            values, reduction="sum", accumulation="deterministic"
        )
        if not isinstance(out, Array):
            raise TypeError("Source reductions must return arrays.")
        return out

    def source_max(self, values: Array, /) -> Array:
        out, _ = self.by_source.reduce(values, reduction="max")
        if not isinstance(out, Array):
            raise TypeError("Source reductions must return arrays.")
        return out

    def log_partition(self, phi: Array, args: _TargetArguments, /) -> tuple[Array, Array]:
        """Row log-partition ``T`` and per-route log weights.

        Rows whose kept mass stays near one use ``log1p(sum p expm1(y) - leak)``,
        exact for small ``theta`` where ``T`` is tiny; rows dominated by path
        cost use a max-shifted log-sum-exp. Both equal ``log sum p e^y`` for the
        row-stochastic reference walk.
        """
        kept = args.kept
        exponent = jnp.where(
            kept, -args.inverse_temperature * (args.cost + phi[self.route_target]), 0.0
        )
        probability = jnp.exp(args.log_reference)
        deficit = self.source_sum(
            jnp.where(kept, probability * jnp.expm1(exponent), 0.0)
        ) - self.source_sum(jnp.where(args.leak, probability, 0.0))
        logits = jnp.where(kept, args.log_reference + exponent, -jnp.inf)
        shift = jax.lax.stop_gradient(
            jnp.where(args.active, self.source_max(logits), 0.0)
        )
        mass = self.source_sum(
            jnp.where(kept, jnp.exp(logits - shift[self.route_source]), 0.0)
        )
        log_sum = shift + jnp.log(jnp.where(args.active, mass, 1.0))
        near_one = deficit > -0.5
        total = jnp.where(near_one, jnp.log1p(jnp.where(near_one, deficit, 0.0)), log_sum)
        return jnp.where(args.active, total, 0.0), logits

    def residual(self, phi: Array, args: _TargetArguments, /) -> Array:
        total, _ = self.log_partition(phi, args)
        value = jnp.where(args.active, phi + total / args.inverse_temperature, phi)
        return value / args.scale

    def weights(self, phi: Array, args: _TargetArguments, /) -> Array:
        total, logits = self.log_partition(phi, args)
        return jnp.where(args.kept, jnp.exp(logits - total[self.route_source]), 0.0)

    def generator(self, weights: Array, scale: Array, /) -> SparseCoordinateOperator:
        """``(I - W) / scale`` with ``W`` the masked per-route transition weights."""
        coefficients = (
            jnp.concatenate((-weights, jnp.ones(self.node_count, dtype=jnp.float64)))
            / scale
        )
        return SparseCoordinateOperator(
            self.jacobian_relation,
            coefficients,
            source=self.space,
            target=self.space,
            operator_id=f"{self.system_id}:generator",
            storage_plan=self.storage,
        )

    def linear_system(self, weights: Array, scale: Array, /) -> LinearSystem:
        return LinearSystem(
            self.generator(weights, scale), problem_id=f"{self.system_id}:walk"
        )

    def jacobian(self, phi: Array, args: _TargetArguments, /) -> SparseCoordinateOperator:
        return self.generator(self.weights(phi, args), args.scale)

    def problem(self) -> NonlinearSystemProblem:
        return NonlinearSystemProblem(
            self.residual,
            state_space=self.space,
            residual_space=self.space,
            problem_id=f"{self.system_id}:free-energy",
        )


def _default_linear_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        GMRES(restart=4),
        tolerance=TolerancePolicy(relative=1e-11, absolute=0.0, max_steps=8),
        preconditioning=PreconditioningPolicy(
            SparseFactorizationPreconditionerBuilder(
                SparseFactorizationPolicy("lu", ordering="approximate-minimum-degree")
            ),
            side="right",
        ),
    )


def _reach_masks(
    source: np.ndarray, target: np.ndarray, node_count: int, nodes: np.ndarray, /
) -> np.ndarray:
    """Host reverse breadth-first reachability to each query target."""
    reverse = csr_matrix(
        (np.ones(source.size, dtype=np.int8), (target, source)),
        shape=(node_count, node_count),
    )
    masks = np.zeros((nodes.size, node_count), dtype=np.bool_)
    for row, node in enumerate(nodes):
        order = breadth_first_order(
            reverse, int(node), directed=True, return_predecessors=False
        )
        masks[row, order] = True
    return masks


def _target_masks(
    reach: Array, target: Array, valid: Array, source: Array, destination: Array, /
) -> tuple[Array, Array, Array]:
    """Active nodes, routes inside the reach set, and routes leaking out of it."""
    active = reach & (jnp.arange(reach.shape[0]) != target)
    kept = valid & active[source] & reach[destination]
    leak = valid & active[source] & ~reach[destination]
    return active, kept, leak


@final
class RandomizedShortestPathPlan(NonTrainableState, StrictModule):
    """Prepare reachability and solve structure for fixed query nodes.

    ``relation`` routes are directed transitions ``source -> target`` on one
    route axis; include both orientations for undirected graphs. Positive
    per-route ``affinities`` define the reference walk ``P_ref`` by row
    normalization; impassable transitions are expressed through relation
    validity, never zero affinity. Costs are nonnegative per-route path costs.
    Execution is float64.

    Construction prepares the Newton and walk-generator solve templates once on
    the host (symbolic factorization included); every ``solve`` only refreshes
    their numerics, so the plan may be a traced argument of compiled code. Each
    target runs one Newton solve seeded by the pointwise minimum of two
    certified supersolutions (the single best augmented path, exact as
    ``theta -> inf``, and the Jensen bound ``m - log(h) / theta`` from the
    hitting probability and conditional hitting cost, exact to first order as
    ``theta -> 0``). The free-energy derivative is carried by one exact Newton
    correction ``phi - J^{-1} G(phi)`` at the converged root, whose first
    derivative is the implicit-function derivative ``-J^{-1} dG``.
    Only ``linear_policy.differentiation.mode == "mathematical"`` is admitted.
    First-order JVP, VJP, and gradients are supported; nested derivative
    transforms raise ``DERIVATIVE_UNSUPPORTED`` rather than carrying an
    incomplete higher derivative. ``workspace_bytes`` budgets retained outputs,
    masks and logical per-target derivative arrays; native solver internals are
    budgeted by ``linear_policy.resources``.
    """

    __strict_contract__ = True

    relation: EdgeRelation
    system: _FreeEnergySystem
    seed_plan: ShortestPathFieldPlan
    walk: PreparedLinearSolve
    newton: PreparedNonlinearSolve
    nodes: Int32[_QueryDim]
    reach: Bool[_QueryDim, _NodeDim]
    node_count: Size[_NodeDim] = eqx.field(static=True)
    query_count: Size[_QueryDim] = eqx.field(static=True)
    resource_items: tuple[tuple[str, int], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        relation: EdgeRelation,
        nodes: ArrayLike,
        /,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        termination: NonlinearTermination | None = None,
        seed_method: ShortestPathFieldMethod = "host-dijkstra",
        workspace_bytes: int = 512 * 1024 * 1024,
        seed_maximum_work: int = 268_435_456,
        seed_maximum_host_bytes: int = 268_435_456,
    ) -> None:
        method = parse(seed_method, ShortestPathFieldMethod, "seed_method")
        policy = _default_linear_policy() if linear_policy is None else linear_policy
        if policy.differentiation.mode != "mathematical":
            raise ValueError(
                "Randomized shortest paths require linear_policy differentiation "
                "mode 'mathematical' for operator and right-hand-side derivatives."
            )
        n = relation.source_size
        if n < 1 or n != relation.target_size:
            raise ValueError(
                "Randomized shortest paths require a square relation with at least one node."
            )
        if len(relation.route_shape) != 1:
            raise ValueError("Randomized-shortest-path relations need one route axis.")
        source = np.asarray(relation.source_indices, dtype=np.int64)
        target = np.asarray(relation.target_indices, dtype=np.int64)
        valid = np.asarray(relation.valid, dtype=np.bool_)
        if np.any(valid & (source == target)):
            raise ValueError(
                "Randomized-shortest-path transitions must not be self-loops."
            )
        host = np.asarray(nodes)
        if host.ndim != 1 or host.size < 1 or not np.issubdtype(host.dtype, np.integer):
            raise ValueError(
                "nodes must be a nonempty one-dimensional host integer array."
            )
        if np.any((host < 0) | (host >= n)):
            raise ValueError("Query nodes are outside the relation node space.")
        if workspace_bytes < 0:
            raise ValueError("workspace_bytes must be nonnegative.")
        q = host.size
        routes = relation.capacity
        itemsize = np.dtype(np.float64).itemsize
        # Logical outer arrays: two q-by-q outputs, reach masks, and per target
        # the retained root, correction, weights, Jacobian values and seeds
        # that reverse-mode differentiation through the target map keeps.
        per_target = (8 * n + 4 * routes) * itemsize
        estimate = {
            "output_bytes": 2 * q * q * itemsize,
            "reach_mask_bytes": q * n,
            "primal_workspace_bytes": 2 * q * q * itemsize + q * n + per_target,
            "derivative_workspace_bytes": 2 * q * q * itemsize + q * n + q * per_target,
        }
        estimate["budgeted_workspace_bytes"] = max(
            estimate["primal_workspace_bytes"], estimate["derivative_workspace_bytes"]
        )
        if estimate["budgeted_workspace_bytes"] > workspace_bytes:
            raise ValueError(
                "Randomized-shortest-path workspace estimate "
                f"{estimate['budgeted_workspace_bytes']} exceeds the byte budget "
                f"{workspace_bytes}."
            )
        safe_source = np.where(valid, source, 0)
        safe_target = np.where(valid, target, 0)
        reach = _reach_masks(source[valid], target[valid], n, host.astype(np.int64))
        diagonal = np.arange(n, dtype=np.int32)
        # Jacobian rows are transition sources, columns transition targets.
        jacobian_relation = EdgeRelation(
            np.concatenate((safe_target.astype(np.int32), diagonal)),
            np.concatenate((safe_source.astype(np.int32), diagonal)),
            valid=np.concatenate((valid, np.ones(n, dtype=np.bool_))),
            source_size=n,
            target_size=n,
        )
        stop = (
            NonlinearTermination(
                absolute_residual=1e-14 * float(np.sqrt(n)),
                relative_residual=0.0,
                maximum_steps=100,
            )
            if termination is None
            else termination
        )
        seed_plan = ShortestPathFieldPlan(
            relation.transpose(),
            method=method,
            maximum_work=seed_maximum_work,
            maximum_host_bytes=seed_maximum_host_bytes,
        )
        estimate["seed_work"] = q * seed_plan.projected_work
        estimate["seed_host_peak_bytes"] = (
            seed_plan.projected_host_bytes if method == "host-dijkstra" else 0
        )
        estimate["seed_host_aggregate_bytes"] = q * estimate["seed_host_peak_bytes"]
        if estimate["seed_work"] > seed_maximum_work:
            raise ValueError(
                f"Total randomized-shortest-path seed work {estimate['seed_work']} "
                f"exceeds the work budget {seed_maximum_work}."
            )
        if estimate["seed_host_aggregate_bytes"] > seed_maximum_host_bytes:
            raise ValueError(
                "Aggregate randomized-shortest-path seed host bytes "
                f"{estimate['seed_host_aggregate_bytes']} exceeds the host byte budget "
                f"{seed_maximum_host_bytes}."
            )
        plan_id = canonical_fingerprint(
            {
                "kind": "randomized-shortest-path-plan",
                "relation": array_tree_fingerprint((source, target, valid)),
                "nodes": host,
                "seed": seed_plan.plan_id,
                "maximum_derivative_order": 1,
                "linear_policy": {
                    "structure": str(jax.tree_util.tree_structure(policy)),
                    "leaves": jax.tree_util.tree_leaves(policy),
                },
                "termination": {
                    "structure": str(jax.tree_util.tree_structure(stop)),
                    "leaves": jax.tree_util.tree_leaves(stop),
                },
            }
        )
        route_source = jnp.asarray(safe_source, dtype=jnp.int32)
        route_target = jnp.asarray(safe_target, dtype=jnp.int32)
        system = _FreeEnergySystem(
            RelationExecutionPlan().prepare(relation.transpose()),
            jacobian_relation,
            SparseStoragePlan(jacobian_relation),
            ArraySpace((n,), dtype=jnp.float64),
            route_source,
            route_target,
            n,
            plan_id,
        )
        # Template numerics: the uniform reference walk at unit cost and
        # temperature. Only structure is retained; every solve refreshes values.
        out_degree = np.maximum(np.bincount(safe_source[valid], minlength=n), 1)
        uniform = np.where(valid, 1.0 / out_degree[safe_source], 0.0)
        walk = prepare(
            system.linear_system(jnp.asarray(0.5 * uniform), jnp.asarray(1.0)), policy
        )
        active, kept, leak = _target_masks(
            jnp.asarray(reach[0]),
            jnp.asarray(host[0]),
            jnp.asarray(valid),
            route_source,
            route_target,
        )
        template_args = _TargetArguments(
            jnp.asarray(
                np.where(valid, np.log(uniform, where=valid, out=np.zeros(routes)), 0.0)
            ),
            jnp.ones(routes, dtype=jnp.float64),
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            active,
            kept,
            leak,
        )
        newton = prepare_nonlinear(
            system.problem(),
            jnp.zeros(n, dtype=jnp.float64),
            method=NewtonKrylov(
                jacobian_policy=JacobianPolicy("explicit", operator=system.jacobian),
                linear_policy=policy,
            ),
            termination=stop,
            args=template_args,
        )
        self.relation = relation
        self.system = system
        self.seed_plan = seed_plan
        self.walk = walk
        self.newton = newton
        self.nodes = jnp.asarray(host, dtype=jnp.int32)
        self.reach = jnp.asarray(reach)
        self.node_count = n
        self.query_count = q
        self.resource_items = tuple(sorted(estimate.items()))
        self.plan_id = plan_id

    @property
    def resource_estimate(self) -> dict[str, int]:
        """Logical outer arrays, total seed work, and peak/aggregate callback bytes."""
        return dict(self.resource_items)

    @property
    def maximum_derivative_order(self) -> int:
        """Immutable first-order-only AD contract, also retained by results."""
        return 1

    def _jensen_seed(
        self,
        probability: Array,
        cost: Array,
        inverse_temperature: Array,
        target: Array,
        active: Array,
        kept: Array,
        /,
    ) -> Array:
        """``m - log(h)/theta`` from hitting probability ``h`` and conditional cost ``m``.

        Both come from the absorbing reference walk ``I - P``: with
        ``H = diag(h)`` the Doob walk satisfies ``I - P_h = H^{-1} (I - P) H``,
        so one refreshed factorization serves both solves. Failed or
        nonpositive values defer to the augmented-path seed.
        """
        system = self.system
        walk = jnp.where(kept, probability, 0.0)
        prepared = refresh(self.walk, system.linear_system(walk, jnp.asarray(1.0)))
        into_target = system.source_sum(
            jnp.where(kept & (system.route_target == target), probability, 0.0)
        )
        hit = solve(prepared, into_target)
        is_target = jnp.arange(self.node_count) == target
        hitting = jnp.where(active, hit.value, jnp.where(is_target, 1.0, 0.0))
        positive = active & (hitting > 0)
        safe = jnp.where(positive, hitting, 1.0)
        conditional = solve(
            prepared, system.source_sum(walk * hitting[system.route_target] * cost)
        )
        moment = conditional.value / safe
        usable = positive & (hit.status == 0) & (conditional.status == 0)
        return jnp.where(usable, moment - jnp.log(safe) / inverse_temperature, jnp.inf)

    def _solve_target(
        self,
        index: Array,
        log_reference: Array,
        cost: Array,
        inverse_temperature: Array,
        /,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        system = self.system
        target = self.nodes[index]
        reach = self.reach[index]
        valid = self.relation.valid
        active, kept, leak = _target_masks(
            reach, target, valid, system.route_source, system.route_target
        )
        # Seeds are numerical starting points only. Stopping their inputs keeps
        # every derivative transform out of the custom-VJP path field.
        seed_reference, seed_cost, seed_temperature = jax.lax.stop_gradient(
            (log_reference, cost, inverse_temperature)
        )
        field = self.seed_plan.solve(
            jnp.where(valid, seed_temperature * seed_cost - seed_reference, 0.0),
            target[None],
        )
        augmented = jnp.where(reach, field.distance, 0.0) / seed_temperature
        jensen = self._jensen_seed(
            jnp.where(valid, jnp.exp(seed_reference), 0.0),
            seed_cost,
            seed_temperature,
            target,
            active,
            kept,
        )
        seed = jnp.where(active, jnp.minimum(augmented, jensen), 0.0)
        largest = jnp.max(seed)
        scale = jnp.where(largest > 0, largest, 1.0)
        args = _TargetArguments(
            log_reference, cost, inverse_temperature, scale, active, kept, leak
        )
        result = solve_prepared_nonlinear(
            refresh_nonlinear(
                self.newton, system.problem(), seed, args=jax.lax.stop_gradient(args)
            )
        )
        converged = jax.lax.stop_gradient(result.state)
        correction = solve(
            refresh(
                self.walk,
                system.linear_system(system.weights(converged, args), args.scale),
            ),
            system.residual(converged, args),
        )
        phi = converged - correction.value
        weights = system.weights(phi, args)
        expected = solve(
            refresh(self.walk, system.linear_system(weights, jnp.asarray(1.0))),
            system.source_sum(weights * cost),
        )
        linear_status = jnp.where(
            correction.status != 0, correction.status, expected.status
        ).astype(jnp.int32)
        return (
            phi[self.nodes],
            expected.value[self.nodes],
            jnp.asarray(result.status, dtype=jnp.int32),
            jnp.asarray(result.diagnostics.iterations, dtype=jnp.int32),
            jnp.asarray(result.diagnostics.final_residual_norm, dtype=jnp.float64),
            linear_status,
        )

    def solve(
        self, affinities: ArrayLike, costs: ArrayLike, inverse_temperature: ArrayLike, /
    ) -> RandomizedShortestPathResult:
        """Solve every query target; invalid inputs never enter a solve."""
        route_shape = self.relation.route_shape
        weights = jnp.asarray(affinities)
        cost = jnp.asarray(costs)
        temperature = jnp.asarray(inverse_temperature)
        for name, value, shape in (
            ("affinities", weights, route_shape),
            ("costs", cost, route_shape),
            ("inverse_temperature", temperature, ()),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")
            if value.dtype != jnp.float64:
                raise TypeError(f"{name} must have dtype float64.")
        weights, cost, temperature = first_order_only(
            (weights, cost, temperature),
            message="Randomized shortest paths support maximum_derivative_order=1.",
        )
        q = self.query_count
        valid = self.relation.valid
        reachable = self.reach[:, self.nodes].T
        admissible = (
            jnp.all(~valid | (jnp.isfinite(weights) & (weights > 0)))
            & jnp.all(~valid | (jnp.isfinite(cost) & (cost >= 0)))
            & jnp.isfinite(temperature)
            & (temperature > 0)
        )

        def invalid(_: tuple[Array, Array, Array]) -> RandomizedShortestPathResult:
            missing = jnp.full((q, q), jnp.nan, dtype=jnp.float64)
            return RandomizedShortestPathResult(
                missing,
                missing,
                reachable,
                jnp.asarray(RandomizedShortestPathStatus.INVALID_INPUT, dtype=jnp.int32),
                jnp.full(q, -1, dtype=jnp.int32),
                jnp.zeros(q, dtype=jnp.int32),
                jnp.full(q, jnp.nan, dtype=jnp.float64),
                jnp.full(q, -1, dtype=jnp.int32),
            )

        def execute(
            operands: tuple[Array, Array, Array],
        ) -> RandomizedShortestPathResult:
            affinity, path_cost, theta = operands
            route_source = self.system.route_source
            log_affinity = jnp.log(jnp.where(valid, affinity, 1.0))
            row_max = jax.lax.stop_gradient(
                self.system.source_max(jnp.where(valid, log_affinity, -jnp.inf))
            )
            shifted = jnp.where(valid, log_affinity - row_max[route_source], 0.0)
            row_mass = self.system.source_sum(jnp.where(valid, jnp.exp(shifted), 0.0))
            log_reference = jnp.where(
                valid,
                shifted - jnp.log(jnp.where(row_mass > 0, row_mass, 1.0))[route_source],
                0.0,
            )
            clean_cost = jnp.where(valid, path_cost, 0.0)
            free, expected, nonlinear, iterations, residual, linear = jax.lax.map(
                lambda index: self._solve_target(index, log_reference, clean_cost, theta),
                jnp.arange(q, dtype=jnp.int32),
            )
            ok = (nonlinear == 0) & (linear == 0)
            diagonal = jnp.eye(q, dtype=jnp.bool_)

            def assemble(columns: Array) -> Array:
                matrix = jnp.where(reachable, columns.T, jnp.inf)
                matrix = jnp.where(diagonal, 0.0, matrix)
                return jnp.where(ok[None, :], matrix, jnp.nan)

            status = jnp.where(
                jnp.all(ok),
                RandomizedShortestPathStatus.SUCCESS,
                jnp.where(
                    jnp.any(nonlinear != 0),
                    RandomizedShortestPathStatus.NONLINEAR_FAILED,
                    RandomizedShortestPathStatus.LINEAR_SOLVE_FAILED,
                ),
            ).astype(jnp.int32)
            return RandomizedShortestPathResult(
                assemble(free),
                assemble(expected),
                reachable,
                status,
                nonlinear,
                iterations,
                residual,
                linear,
            )

        return jax.lax.cond(admissible, execute, invalid, (weights, cost, temperature))
