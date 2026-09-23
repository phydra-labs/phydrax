#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._sampling._addressing import derive_key, SampleAddress
from .._strict import StrictModule
from ..metrix import (
    AbstractStateGeometry,
    EuclideanStateGeometry,
    FlatTorusStateGeometry,
    LieAlgebraCoordinateMetric,
    LieGroupStateGeometry,
    PointwiseStateGeometry,
)
from ..operators.path_integral._pseudofermion import (
    evaluate_pseudofermion_action,
    FractionalPowerPseudofermionTerm,
    HasenbuschRatioPseudofermionTerm,
    pseudofermion_force,
    PseudofermionTerm,
    refresh_pseudofermion,
    TwoFlavorPseudofermionTerm,
)


_MOMENTUM_ADDRESS = SampleAddress(
    "markov",
    "rational-hamiltonian",
    target="momentum",
    role="transition",
)
_PSEUDOFERMION_ADDRESS = SampleAddress(
    "markov",
    "rational-hamiltonian",
    target="pseudofermion",
    role="refresh",
)
_ACCEPT_ADDRESS = SampleAddress(
    "markov",
    "rational-hamiltonian",
    target="acceptance",
    role="transition",
)


class SeparableActionTerm(StrictModule):
    """One bosonic action term independently assignable to a force scale."""

    evaluate: Callable[[Array], Array]
    term_id: str = eqx.field(static=True)

    def __init__(self, evaluate: Callable[[Array], Array], /, *, term_id: str):
        if not callable(evaluate):
            raise TypeError("evaluate must be callable.")
        identifier = str(term_id)
        if not identifier:
            raise ValueError("term_id must be nonempty.")
        self.evaluate = evaluate
        self.term_id = identifier

    def __call__(self, configuration: ArrayLike, /) -> Array:
        value = jnp.asarray(self.evaluate(jnp.asarray(configuration)))
        if value.shape != () or jnp.iscomplexobj(value):
            raise ValueError("A separable action term must return one real scalar.")
        return value


class SeparableActionRegistry(StrictModule):
    """Ordered bosonic and pseudofermion action terms with canonical identity."""

    action_terms: tuple[SeparableActionTerm, ...]
    pseudofermion_terms: tuple[PseudofermionTerm, ...]
    term_ids: tuple[str, ...] = eqx.field(static=True)
    registry_id: str = eqx.field(static=True)

    def __init__(
        self,
        action_terms: Sequence[SeparableActionTerm],
        pseudofermion_terms: Sequence[PseudofermionTerm] = (),
        /,
    ):
        actions = tuple(action_terms)
        pseudofermions = tuple(pseudofermion_terms)
        if not actions and not pseudofermions:
            raise ValueError(
                "A separable action registry must contain at least one term."
            )
        if any(not isinstance(term, SeparableActionTerm) for term in actions):
            raise TypeError("action_terms must contain SeparableActionTerm values.")
        pseudofermion_types = (
            TwoFlavorPseudofermionTerm,
            HasenbuschRatioPseudofermionTerm,
            FractionalPowerPseudofermionTerm,
        )
        if any(not isinstance(term, pseudofermion_types) for term in pseudofermions):
            raise TypeError("pseudofermion_terms contain an unknown term type.")
        identifiers = tuple(term.term_id for term in actions) + tuple(
            term.term_id for term in pseudofermions
        )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Separable action term IDs must be unique.")
        self.action_terms = actions
        self.pseudofermion_terms = pseudofermions
        self.term_ids = identifiers
        self.registry_id = canonical_fingerprint(
            {
                "kind": "separable-lattice-action-registry",
                "action_terms": [term.term_id for term in actions],
                "pseudofermion_terms": [term.term_id for term in pseudofermions],
            }
        )

    @property
    def num_terms(self) -> int:
        return len(self.term_ids)


class NestedForcePartition(StrictModule):
    """Term indices advanced at one reversible nested time scale."""

    term_indices: tuple[int, ...] = eqx.field(static=True)
    substeps: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        term_indices: Sequence[int],
        /,
        *,
        substeps: int,
    ):
        indices = tuple(term_indices)
        count = int(substeps)
        if not indices or any(index < 0 for index in indices):
            raise ValueError("A force partition requires non-negative term indices.")
        if len(set(indices)) != len(indices):
            raise ValueError("A force partition cannot repeat a term index.")
        if count < 1:
            raise ValueError("Force-partition substeps must be positive.")
        self.term_indices = indices
        self.substeps = count
        self.partition_id = canonical_fingerprint(
            {
                "kind": "nested-force-partition",
                "term_indices": list(indices),
                "substeps": count,
            }
        )


class NestedForcePlan(StrictModule):
    """Outer-to-inner force partitions for a symmetric nested integrator."""

    partitions: tuple[NestedForcePartition, ...] = eqx.field(static=True)
    force_evaluations_per_step: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, partitions: Sequence[NestedForcePartition], /):
        values = tuple(partitions)
        if not values or any(
            not isinstance(value, NestedForcePartition) for value in values
        ):
            raise ValueError("NestedForcePlan requires at least one force partition.")
        multiplicity = 1
        evaluations = 0
        for partition in values:
            multiplicity *= partition.substeps
            evaluations += 2 * multiplicity * len(partition.term_indices)
        self.partitions = values
        self.force_evaluations_per_step = evaluations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "symmetric-nested-force-plan",
                "partitions": [value.partition_id for value in values],
                "force_evaluations_per_step": evaluations,
            }
        )


class RHMCResourcePolicy(StrictModule):
    """Hard fixed-shape term, force-work, retained-state, and output budgets."""

    maximum_terms: int = eqx.field(static=True)
    maximum_force_evaluations: int = eqx.field(static=True)
    maximum_retained_bytes: int = eqx.field(static=True)
    maximum_output_bytes: int = eqx.field(static=True)
    maximum_draws: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_terms: int = 64,
        maximum_force_evaluations: int = 1_000_000,
        maximum_retained_bytes: int = 2 * 1024 * 1024 * 1024,
        maximum_output_bytes: int = 8 * 1024 * 1024 * 1024,
        maximum_draws: int = 1_000_000,
    ):
        terms = int(maximum_terms)
        evaluations = int(maximum_force_evaluations)
        retained = int(maximum_retained_bytes)
        output = int(maximum_output_bytes)
        draws = int(maximum_draws)
        if terms < 1 or evaluations < 1 or retained < 1 or output < 1 or draws < 1:
            raise ValueError("RHMC resource limits must be positive.")
        self.maximum_terms = terms
        self.maximum_force_evaluations = evaluations
        self.maximum_retained_bytes = retained
        self.maximum_output_bytes = output
        self.maximum_draws = draws
        self.policy_id = canonical_fingerprint(
            {
                "kind": "rhmc-resource-policy",
                "maximum_terms": terms,
                "maximum_force_evaluations": evaluations,
                "maximum_retained_bytes": retained,
                "maximum_output_bytes": output,
                "maximum_draws": draws,
            }
        )


class RHMCPlan(StrictModule):
    """Immutable production trajectory and nested-work declaration."""

    force_plan: NestedForcePlan = eqx.field(static=True)
    resources: RHMCResourcePolicy = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    trajectory_steps: int = eqx.field(static=True)
    divergence_threshold: float = eqx.field(static=True)
    force_evaluations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        step_size: float,
        trajectory_steps: int,
        force_plan: NestedForcePlan,
        divergence_threshold: float = 1000.0,
        resources: RHMCResourcePolicy | None = None,
    ):
        size = float(step_size)
        steps = int(trajectory_steps)
        threshold = float(divergence_threshold)
        policy = RHMCResourcePolicy() if resources is None else resources
        if not isinstance(force_plan, NestedForcePlan):
            raise TypeError("force_plan must be a NestedForcePlan.")
        if not isinstance(policy, RHMCResourcePolicy):
            raise TypeError("resources must be RHMCResourcePolicy or None.")
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if steps < 1:
            raise ValueError("trajectory_steps must be positive.")
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("divergence_threshold must be finite and positive.")
        evaluations = steps * force_plan.force_evaluations_per_step
        if evaluations > policy.maximum_force_evaluations:
            raise MemoryError("RHMC force work exceeds its fixed resource policy.")
        self.force_plan = force_plan
        self.resources = policy
        self.step_size = size
        self.trajectory_steps = steps
        self.divergence_threshold = threshold
        self.force_evaluations = evaluations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "production-rhmc-plan",
                "force_plan": force_plan.plan_id,
                "step_size": size,
                "trajectory_steps": steps,
                "divergence_threshold": threshold,
                "force_evaluations": evaluations,
                "resources": policy.policy_id,
            }
        )


class PreparedRHMCKernel(StrictModule):
    """Registry and geometry bound to one frozen production plan."""

    registry: SeparableActionRegistry
    geometry: AbstractStateGeometry
    coordinate_metric: LieAlgebraCoordinateMetric | None
    configuration_template: Array
    valid: Array
    plan: RHMCPlan = eqx.field(static=True)
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    local_coordinate_shape: tuple[int, ...] = eqx.field(static=True)
    retained_bytes: int = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)


class RegistryActionEvaluation(StrictModule):
    """Full potential and fixed-length per-term solve/finite evidence."""

    value: Array
    term_status: Array
    term_residual_indicator: Array
    term_solve_error_upper_bound: Array
    term_solve_error_bound_available: Array
    term_solve_error_bound_certified: Array
    term_successful: Array
    finite: Array


class RHMCChainState(StrictModule):
    """Checkpoint-complete accepted state and semantic random-stream position."""

    configuration: Array
    bosonic_action: Array
    step_index: Array
    accepted_count: Array
    root_key: Array
    valid: Array
    kernel_id: str = eqx.field(static=True)


class RHMCTrajectoryResult(StrictModule):
    """Unaccepted reversible proposal and force-path evidence."""

    configuration: Array
    momentum: Array
    force_successful: Array
    finite: Array
    membership_failure: Array
    force_evaluations: Array
    maximum_shifted_solution_error_upper_bound: Array
    solve_error_bound_available: Array
    solve_error_bound_certified: Array


class RHMCTransitionEvidence(StrictModule):
    """Exact endpoint Hamiltonians and accepted-transition decision evidence."""

    accepted: Array
    acceptance_probability: Array
    initial_action: Array
    proposed_action: Array
    initial_kinetic: Array
    proposed_kinetic: Array
    initial_energy: Array
    proposed_energy: Array
    energy_error: Array
    divergent: Array
    nonfinite: Array
    membership_failure: Array
    refresh_status: Array
    refresh_gaussian_action: Array
    refresh_solve_error_upper_bound: Array
    refresh_solve_error_bound_available: Array
    refresh_solve_error_bound_certified: Array
    initial_term_status: Array
    initial_term_residual_indicator: Array
    initial_term_solve_error_upper_bound: Array
    initial_term_solve_error_bound_available: Array
    initial_term_solve_error_bound_certified: Array
    proposed_term_status: Array
    proposed_term_residual_indicator: Array
    proposed_term_solve_error_upper_bound: Array
    proposed_term_solve_error_bound_available: Array
    proposed_term_solve_error_bound_certified: Array
    force_successful: Array
    force_evaluations: Array
    maximum_force_shifted_solution_error_upper_bound: Array
    force_solve_error_bound_available: Array
    force_solve_error_bound_certified: Array


class RHMCTransitionResult(StrictModule):
    state: RHMCChainState
    evidence: RHMCTransitionEvidence


class RHMCSampleResult(StrictModule):
    """Production draws plus a restart-equivalent final chain checkpoint."""

    configurations: Array
    bosonic_action: Array
    accepted: Array
    acceptance_probability: Array
    energy_error: Array
    divergent: Array
    nonfinite: Array
    membership_failure: Array
    force_successful: Array
    force_evaluations: Array
    refresh_solve_error_upper_bound: Array
    refresh_solve_error_bound_available: Array
    refresh_solve_error_bound_certified: Array
    initial_term_solve_error_upper_bound: Array
    initial_term_solve_error_bound_available: Array
    initial_term_solve_error_bound_certified: Array
    proposed_term_solve_error_upper_bound: Array
    proposed_term_solve_error_bound_available: Array
    proposed_term_solve_error_bound_certified: Array
    maximum_force_shifted_solution_error_upper_bound: Array
    force_solve_error_bound_available: Array
    force_solve_error_bound_certified: Array
    final_state: RHMCChainState
    kernel_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    @property
    def num_draws(self) -> int:
        return self.accepted.shape[0]

    @property
    def acceptance_rate(self) -> Array:
        return jnp.mean(self.accepted.astype("float64"))


def plan_rhmc(
    *,
    step_size: float,
    trajectory_steps: int,
    force_plan: NestedForcePlan,
    divergence_threshold: float = 1000.0,
    resources: RHMCResourcePolicy | None = None,
) -> RHMCPlan:
    """Plan a frozen production RHMC trajectory before binding numeric state."""
    return RHMCPlan(
        step_size=step_size,
        trajectory_steps=trajectory_steps,
        force_plan=force_plan,
        divergence_threshold=divergence_threshold,
        resources=resources,
    )


def prepare_rhmc(
    registry: SeparableActionRegistry,
    plan: RHMCPlan,
    configuration_template: ArrayLike,
    /,
    *,
    geometry: AbstractStateGeometry,
    local_coordinate_shape: Sequence[int],
) -> PreparedRHMCKernel:
    """Bind a registry and configuration geometry without creating chain state."""
    if not isinstance(registry, SeparableActionRegistry):
        raise TypeError("registry must be a SeparableActionRegistry.")
    if not isinstance(plan, RHMCPlan):
        raise TypeError("plan must be an RHMCPlan.")
    if not isinstance(geometry, AbstractStateGeometry):
        raise TypeError("geometry must implement AbstractStateGeometry.")
    configuration = jnp.asarray(configuration_template)
    if configuration.size < 1 or not jnp.issubdtype(configuration.dtype, jnp.inexact):
        raise ValueError("configuration_template must be a nonempty inexact array.")
    local_shape = tuple(local_coordinate_shape)
    if not local_shape or any(size <= 0 for size in local_shape):
        raise ValueError("local_coordinate_shape must contain positive dimensions.")
    all_indices = tuple(
        index
        for partition in plan.force_plan.partitions
        for index in partition.term_indices
    )
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Nested force partitions cannot assign a term more than once.")
    if set(all_indices) != set(range(registry.num_terms)):
        raise ValueError(
            "Nested force partitions must assign every registry term exactly once."
        )
    if registry.num_terms > plan.resources.maximum_terms:
        raise MemoryError("RHMC registry exceeds its fixed term budget.")
    coordinate_metric = _coordinate_metric(geometry, configuration.shape, local_shape)
    local_itemsize = jnp.empty((), dtype=jnp.real(configuration).dtype).dtype.itemsize
    configuration_bytes = configuration.size * configuration.dtype.itemsize
    local_bytes = int(np.prod(local_shape) * local_itemsize)
    pseudofermion_bytes = sum(
        _space_storage_bytes(term.dirac.source) for term in registry.pseudofermion_terms
    )
    retained = 3 * configuration_bytes + 2 * local_bytes + pseudofermion_bytes
    if retained > plan.resources.maximum_retained_bytes:
        raise MemoryError("Prepared RHMC retained state exceeds its resource policy.")
    member = jnp.asarray(geometry.contains(configuration), dtype=jnp.bool_)
    valid = member & jnp.all(jnp.isfinite(configuration))
    kernel_id = canonical_fingerprint(
        {
            "kind": "prepared-production-rhmc-kernel",
            "registry": registry.registry_id,
            "plan": plan.plan_id,
            "geometry": geometry.geometry_id,
            "configuration_shape": list(configuration.shape),
            "local_coordinate_shape": list(local_shape),
            "configuration_dtype": np.dtype(configuration.dtype).str,
            "retained_bytes": retained,
        }
    )
    return PreparedRHMCKernel(
        registry=registry,
        geometry=geometry,
        coordinate_metric=coordinate_metric,
        configuration_template=configuration,
        valid=valid,
        plan=plan,
        configuration_shape=configuration.shape,
        local_coordinate_shape=local_shape,
        retained_bytes=retained,
        kernel_id=kernel_id,
    )


def initialize_rhmc_state(
    kernel: PreparedRHMCKernel,
    initial_configuration: ArrayLike,
    /,
    *,
    key: Key[Array, ""],
) -> RHMCChainState:
    """Initialize a checkpoint-safe accepted state at semantic step zero."""
    _validate_kernel(kernel)
    configuration = jnp.asarray(
        initial_configuration, dtype=kernel.configuration_template.dtype
    )
    if configuration.shape != kernel.configuration_shape:
        raise ValueError(
            f"initial_configuration must have shape {kernel.configuration_shape}."
        )
    bosonic = _bosonic_action(kernel.registry, configuration)
    member = kernel.geometry.contains(configuration)
    valid = (
        kernel.valid
        & member
        & jnp.isfinite(bosonic)
        & jnp.all(jnp.isfinite(configuration))
    )
    bosonic = eqx.error_if(
        bosonic,
        ~valid,
        "Initial RHMC configuration must be a finite geometry member with finite action.",
    )
    return RHMCChainState(
        configuration=configuration,
        bosonic_action=bosonic,
        step_index=jnp.asarray(0, dtype=jnp.uint32),
        accepted_count=jnp.asarray(0, dtype=jnp.uint32),
        root_key=jnp.asarray(key),
        valid=valid,
        kernel_id=kernel.kernel_id,
    )


def evaluate_registry_action(
    registry: SeparableActionRegistry,
    pseudofermion_fields: tuple[PyTree[Array], ...],
    configuration: ArrayLike,
    /,
    *,
    role: str = "action",
) -> RegistryActionEvaluation:
    """Evaluate every registered term with solve-error evidence."""
    if role not in ("action", "force", "acceptance"):
        raise ValueError("Unknown registry action role.")
    if len(pseudofermion_fields) != len(registry.pseudofermion_terms):
        raise ValueError("Pseudofermion fields must match the registry.")
    configuration_ = jnp.asarray(configuration)
    values: list[Array] = []
    statuses: list[Array] = []
    successful: list[Array] = []
    residuals: list[Array] = []
    solve_bounds: list[Array] = []
    bound_available: list[Array] = []
    bound_certified: list[Array] = []
    for term in registry.action_terms:
        value = term(configuration_)
        finite = jnp.isfinite(value)
        values.append(value)
        statuses.append(jnp.where(finite, 0, 2).astype(jnp.int32))
        successful.append(finite)
        residuals.append(jnp.asarray(0.0, dtype=value.dtype))
        solve_bounds.append(jnp.asarray(0.0, dtype=jnp.real(value).dtype))
        bound_available.append(jnp.asarray(True))
        bound_certified.append(jnp.asarray(True))
    for term, field in zip(
        registry.pseudofermion_terms,
        pseudofermion_fields,
        strict=True,
    ):
        result = evaluate_pseudofermion_action(
            term,
            field,
            role=role,
            links=configuration_,
        )
        values.append(result.value)
        statuses.append(result.status)
        successful.append(result.successful)
        residuals.append(result.residual_indicator)
        solve_bounds.append(result.solve_error_upper_bound)
        bound_available.append(result.solve_error_bound_available)
        bound_certified.append(result.solve_error_bound_certified)
    value = sum(values, jnp.asarray(0.0, dtype=jnp.real(configuration_).dtype))
    status_array = jnp.stack(tuple(statuses))
    residual_array = jnp.stack(tuple(residuals))
    successful_array = jnp.stack(tuple(successful))
    solve_bound_array = jnp.stack(tuple(solve_bounds))
    available_array = jnp.stack(tuple(bound_available))
    certified_array = jnp.stack(tuple(bound_certified))
    finite = (
        jnp.isfinite(value)
        & jnp.all(jnp.isfinite(status_array))
        & jnp.all(jnp.isfinite(residual_array))
    )
    return RegistryActionEvaluation(
        value=value,
        term_status=status_array,
        term_successful=successful_array,
        term_residual_indicator=residual_array,
        term_solve_error_upper_bound=solve_bound_array,
        term_solve_error_bound_available=available_array,
        term_solve_error_bound_certified=certified_array,
        finite=finite,
    )


def integrate_rhmc_trajectory(
    kernel: PreparedRHMCKernel,
    configuration: ArrayLike,
    momentum: ArrayLike,
    pseudofermion_fields: tuple[PyTree[Array], ...],
    /,
    *,
    step_size: float | ArrayLike | None = None,
) -> RHMCTrajectoryResult:
    """Apply the symmetric nested map with force-solve evidence."""
    _validate_kernel(kernel)
    q = jnp.asarray(configuration, dtype=kernel.configuration_template.dtype)
    p = jnp.asarray(momentum, dtype=jnp.real(kernel.configuration_template).dtype)
    if q.shape != kernel.configuration_shape or p.shape != kernel.local_coordinate_shape:
        raise ValueError("Trajectory configuration or momentum shape is incompatible.")
    size = kernel.plan.step_size if step_size is None else step_size
    dt = jnp.asarray(size, dtype=p.dtype)
    if dt.shape != ():
        raise ValueError("Trajectory step_size must be scalar.")
    dt = eqx.error_if(
        dt, ~jnp.isfinite(dt) | (dt == 0.0), "step_size must be finite and nonzero."
    )
    active = jnp.asarray(True)
    finite = jnp.asarray(True)
    membership_failure = jnp.asarray(False)
    evaluations = jnp.asarray(0, dtype=jnp.int32)
    maximum_bound = jnp.asarray(0.0, dtype=p.dtype)
    bound_available = jnp.asarray(True)
    bound_certified = jnp.asarray(True)
    for _ in range(kernel.plan.trajectory_steps):
        (
            q,
            p,
            active,
            step_finite,
            step_membership,
            step_evaluations,
            step_bound,
            step_available,
            step_certified,
        ) = _nested_level(
            kernel,
            pseudofermion_fields,
            0,
            q,
            p,
            dt,
            active,
        )
        finite = finite & step_finite
        membership_failure = membership_failure | step_membership
        evaluations = evaluations + step_evaluations
        maximum_bound = jnp.maximum(maximum_bound, step_bound)
        bound_available = bound_available & step_available
        bound_certified = bound_certified & step_certified
    maximum_bound = jnp.where(bound_available, maximum_bound, jnp.inf)
    return RHMCTrajectoryResult(
        configuration=q,
        momentum=p,
        force_successful=active,
        finite=finite,
        membership_failure=membership_failure,
        force_evaluations=evaluations,
        maximum_shifted_solution_error_upper_bound=maximum_bound,
        solve_error_bound_available=bound_available,
        solve_error_bound_certified=bound_certified,
    )


def rhmc_transition(
    kernel: PreparedRHMCKernel,
    state: RHMCChainState,
    /,
) -> RHMCTransitionResult:
    """Refresh auxiliaries, propose reversibly, and commit only an accepted state."""
    _validate_state(kernel, state)
    configuration = state.configuration
    fields: list[PyTree[Array]] = []
    refresh_statuses: list[Array] = []
    refresh_gaussian_actions: list[Array] = []
    refresh_solve_bounds: list[Array] = []
    refresh_bound_available: list[Array] = []
    refresh_bound_certified: list[Array] = []
    refresh_successful = jnp.asarray(True)
    for term_index, term in enumerate(kernel.registry.pseudofermion_terms):
        refresh_key = derive_key(
            state.root_key,
            _PSEUDOFERMION_ADDRESS,
            state.step_index,
            term_index,
        )
        refresh = refresh_pseudofermion(term, refresh_key, links=configuration)
        fields.append(refresh.field)
        refresh_statuses.append(refresh.status)
        refresh_gaussian_actions.append(refresh.gaussian_action)
        refresh_solve_bounds.append(refresh.solve_error_upper_bound)
        refresh_bound_available.append(refresh.solve_error_bound_available)
        refresh_bound_certified.append(refresh.solve_error_bound_certified)
        refresh_successful = refresh_successful & refresh.successful
    pseudofermion_fields = tuple(fields)
    refresh_status = (
        jnp.stack(tuple(refresh_statuses))
        if refresh_statuses
        else jnp.zeros((0,), dtype=jnp.int32)
    )
    refresh_gaussian_action = (
        jnp.stack(tuple(refresh_gaussian_actions))
        if refresh_gaussian_actions
        else jnp.zeros((0,), dtype=jnp.real(configuration).dtype)
    )
    refresh_solve_error_upper_bound = (
        jnp.stack(tuple(refresh_solve_bounds))
        if refresh_solve_bounds
        else jnp.zeros((0,), dtype=jnp.real(configuration).dtype)
    )
    refresh_solve_error_bound_available = (
        jnp.stack(tuple(refresh_bound_available))
        if refresh_bound_available
        else jnp.ones((0,), dtype=jnp.bool_)
    )
    refresh_solve_error_bound_certified = (
        jnp.stack(tuple(refresh_bound_certified))
        if refresh_bound_certified
        else jnp.ones((0,), dtype=jnp.bool_)
    )
    momentum_key = derive_key(
        state.root_key,
        _MOMENTUM_ADDRESS,
        state.step_index,
    )
    accept_key = derive_key(
        state.root_key,
        _ACCEPT_ADDRESS,
        state.step_index,
    )
    momentum = _sample_momentum(kernel, momentum_key)
    initial_action = evaluate_registry_action(
        kernel.registry,
        pseudofermion_fields,
        configuration,
        role="acceptance",
    )
    proposal = integrate_rhmc_trajectory(
        kernel,
        configuration,
        momentum,
        pseudofermion_fields,
    )
    proposed_momentum = -proposal.momentum
    proposed_action = evaluate_registry_action(
        kernel.registry,
        pseudofermion_fields,
        proposal.configuration,
        role="acceptance",
    )
    initial_kinetic = _kinetic(kernel, momentum)
    proposed_kinetic = _kinetic(kernel, proposed_momentum)
    initial_energy = initial_action.value + initial_kinetic
    proposed_energy = proposed_action.value + proposed_kinetic
    energy_error = proposed_energy - initial_energy
    finite = (
        initial_action.finite
        & proposed_action.finite
        & jnp.isfinite(initial_kinetic)
        & jnp.isfinite(proposed_kinetic)
        & jnp.isfinite(energy_error)
        & proposal.finite
    )
    endpoint_successful = jnp.all(initial_action.term_successful) & jnp.all(
        proposed_action.term_successful
    )
    divergent = (
        ~state.valid
        | ~refresh_successful
        | ~proposal.force_successful
        | ~endpoint_successful
        | ~finite
        | proposal.membership_failure
        | (jnp.abs(energy_error) > kernel.plan.divergence_threshold)
    )
    log_acceptance = jnp.minimum(-energy_error, 0.0)
    accepted = ~divergent & (jnp.log(jr.uniform(accept_key)) < log_acceptance)
    next_configuration = jnp.where(
        accepted,
        proposal.configuration,
        configuration,
    )
    proposed_bosonic = _bosonic_action(kernel.registry, proposal.configuration)
    next_bosonic = jnp.where(accepted, proposed_bosonic, state.bosonic_action)
    next_state = RHMCChainState(
        configuration=next_configuration,
        bosonic_action=next_bosonic,
        step_index=state.step_index + jnp.asarray(1, dtype=jnp.uint32),
        accepted_count=state.accepted_count + accepted.astype(jnp.uint32),
        root_key=state.root_key,
        valid=state.valid,
        kernel_id=state.kernel_id,
    )
    probability = jnp.where(
        ~divergent & jnp.isfinite(log_acceptance),
        jnp.exp(log_acceptance),
        0.0,
    )
    evidence = RHMCTransitionEvidence(
        accepted=accepted,
        acceptance_probability=probability,
        initial_action=initial_action.value,
        proposed_action=proposed_action.value,
        initial_kinetic=initial_kinetic,
        proposed_kinetic=proposed_kinetic,
        initial_energy=initial_energy,
        proposed_energy=proposed_energy,
        energy_error=energy_error,
        divergent=divergent,
        nonfinite=~finite,
        membership_failure=proposal.membership_failure,
        refresh_status=refresh_status,
        refresh_gaussian_action=refresh_gaussian_action,
        refresh_solve_error_upper_bound=refresh_solve_error_upper_bound,
        refresh_solve_error_bound_available=refresh_solve_error_bound_available,
        refresh_solve_error_bound_certified=refresh_solve_error_bound_certified,
        initial_term_status=initial_action.term_status,
        initial_term_residual_indicator=initial_action.term_residual_indicator,
        initial_term_solve_error_upper_bound=(
            initial_action.term_solve_error_upper_bound
        ),
        initial_term_solve_error_bound_available=(
            initial_action.term_solve_error_bound_available
        ),
        initial_term_solve_error_bound_certified=(
            initial_action.term_solve_error_bound_certified
        ),
        proposed_term_status=proposed_action.term_status,
        proposed_term_residual_indicator=proposed_action.term_residual_indicator,
        proposed_term_solve_error_upper_bound=(
            proposed_action.term_solve_error_upper_bound
        ),
        proposed_term_solve_error_bound_available=(
            proposed_action.term_solve_error_bound_available
        ),
        proposed_term_solve_error_bound_certified=(
            proposed_action.term_solve_error_bound_certified
        ),
        force_successful=proposal.force_successful,
        force_evaluations=proposal.force_evaluations,
        maximum_force_shifted_solution_error_upper_bound=(
            proposal.maximum_shifted_solution_error_upper_bound
        ),
        force_solve_error_bound_available=proposal.solve_error_bound_available,
        force_solve_error_bound_certified=proposal.solve_error_bound_certified,
    )
    return RHMCTransitionResult(next_state, evidence)


def sample_rhmc(
    kernel: PreparedRHMCKernel,
    state: RHMCChainState,
    /,
    *,
    num_draws: int,
) -> RHMCSampleResult:
    """Advance production RHMC while preserving deterministic restart addressing."""
    _validate_state(kernel, state)
    draws = int(num_draws)
    if draws < 1:
        raise ValueError("num_draws must be positive.")
    if draws > kernel.plan.resources.maximum_draws:
        raise MemoryError("RHMC draw count exceeds its fixed resource policy.")
    real_itemsize = jnp.real(kernel.configuration_template).dtype.itemsize
    pseudofermion_count = len(kernel.registry.pseudofermion_terms)
    term_count = len(kernel.registry.term_ids)
    solve_evidence_count = pseudofermion_count + 2 * term_count + 1
    per_draw_bytes = (
        kernel.configuration_template.size * kernel.configuration_template.dtype.itemsize
        + (3 + solve_evidence_count) * real_itemsize
        + (5 + 2 * solve_evidence_count) * jnp.dtype(jnp.bool_).itemsize
        + jnp.dtype(jnp.int32).itemsize
    )
    if draws * per_draw_bytes > kernel.plan.resources.maximum_output_bytes:
        raise MemoryError("RHMC sample output exceeds its fixed resource policy.")
    configurations: list[Array] = []
    bosonic_actions: list[Array] = []
    accepted: list[Array] = []
    probabilities: list[Array] = []
    energy_errors: list[Array] = []
    divergent: list[Array] = []
    nonfinite: list[Array] = []
    membership_failures: list[Array] = []
    force_successful: list[Array] = []
    force_evaluations: list[Array] = []
    refresh_solve_bounds: list[Array] = []
    refresh_bound_available: list[Array] = []
    refresh_bound_certified: list[Array] = []
    initial_solve_bounds: list[Array] = []
    initial_bound_available: list[Array] = []
    initial_bound_certified: list[Array] = []
    proposed_solve_bounds: list[Array] = []
    proposed_bound_available: list[Array] = []
    proposed_bound_certified: list[Array] = []
    force_solve_bounds: list[Array] = []
    force_bound_available: list[Array] = []
    force_bound_certified: list[Array] = []
    current = state
    for _ in range(draws):
        transition = rhmc_transition(kernel, current)
        current = transition.state
        evidence = transition.evidence
        configurations.append(current.configuration)
        bosonic_actions.append(current.bosonic_action)
        accepted.append(evidence.accepted)
        probabilities.append(evidence.acceptance_probability)
        energy_errors.append(evidence.energy_error)
        divergent.append(evidence.divergent)
        nonfinite.append(evidence.nonfinite)
        membership_failures.append(evidence.membership_failure)
        force_successful.append(evidence.force_successful)
        force_evaluations.append(evidence.force_evaluations)
        refresh_solve_bounds.append(evidence.refresh_solve_error_upper_bound)
        refresh_bound_available.append(evidence.refresh_solve_error_bound_available)
        refresh_bound_certified.append(evidence.refresh_solve_error_bound_certified)
        initial_solve_bounds.append(evidence.initial_term_solve_error_upper_bound)
        initial_bound_available.append(evidence.initial_term_solve_error_bound_available)
        initial_bound_certified.append(evidence.initial_term_solve_error_bound_certified)
        proposed_solve_bounds.append(evidence.proposed_term_solve_error_upper_bound)
        proposed_bound_available.append(
            evidence.proposed_term_solve_error_bound_available
        )
        proposed_bound_certified.append(
            evidence.proposed_term_solve_error_bound_certified
        )
        force_solve_bounds.append(
            evidence.maximum_force_shifted_solution_error_upper_bound
        )
        force_bound_available.append(evidence.force_solve_error_bound_available)
        force_bound_certified.append(evidence.force_solve_error_bound_certified)
    return RHMCSampleResult(
        configurations=jnp.stack(tuple(configurations)),
        bosonic_action=jnp.stack(tuple(bosonic_actions)),
        accepted=jnp.stack(tuple(accepted)),
        acceptance_probability=jnp.stack(tuple(probabilities)),
        energy_error=jnp.stack(tuple(energy_errors)),
        divergent=jnp.stack(tuple(divergent)),
        nonfinite=jnp.stack(tuple(nonfinite)),
        membership_failure=jnp.stack(tuple(membership_failures)),
        force_successful=jnp.stack(tuple(force_successful)),
        force_evaluations=jnp.stack(tuple(force_evaluations)),
        refresh_solve_error_upper_bound=jnp.stack(tuple(refresh_solve_bounds)),
        refresh_solve_error_bound_available=jnp.stack(tuple(refresh_bound_available)),
        refresh_solve_error_bound_certified=jnp.stack(tuple(refresh_bound_certified)),
        initial_term_solve_error_upper_bound=jnp.stack(tuple(initial_solve_bounds)),
        initial_term_solve_error_bound_available=jnp.stack(
            tuple(initial_bound_available)
        ),
        initial_term_solve_error_bound_certified=jnp.stack(
            tuple(initial_bound_certified)
        ),
        proposed_term_solve_error_upper_bound=jnp.stack(tuple(proposed_solve_bounds)),
        proposed_term_solve_error_bound_available=jnp.stack(
            tuple(proposed_bound_available)
        ),
        proposed_term_solve_error_bound_certified=jnp.stack(
            tuple(proposed_bound_certified)
        ),
        maximum_force_shifted_solution_error_upper_bound=jnp.stack(
            tuple(force_solve_bounds)
        ),
        force_solve_error_bound_available=jnp.stack(tuple(force_bound_available)),
        force_solve_error_bound_certified=jnp.stack(tuple(force_bound_certified)),
        final_state=current,
        kernel_id=kernel.kernel_id,
        claim=(
            "Frozen production RHMC with semantic randomness, exact endpoint Hamiltonians, and rejection rollback"
        ),
    )


def rhmc_checkpoint_fingerprint(state: RHMCChainState, /) -> str:
    """Content-address one complete restart state without hidden RNG state."""
    if not isinstance(state, RHMCChainState):
        raise TypeError("state must be an RHMCChainState.")
    return canonical_fingerprint(
        {
            "kind": "rhmc-chain-checkpoint",
            "kernel": state.kernel_id,
            "arrays": array_tree_fingerprint(
                (
                    state.configuration,
                    state.bosonic_action,
                    state.step_index,
                    state.accepted_count,
                    jr.key_data(state.root_key),
                    state.valid,
                )
            ),
        }
    )


def _nested_level(
    kernel: PreparedRHMCKernel,
    fields: tuple[PyTree[Array], ...],
    level: int,
    q: Array,
    p: Array,
    dt: Array,
    active: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    partition = kernel.plan.force_plan.partitions[level]
    substep = dt / partition.substeps
    finite = jnp.asarray(True)
    membership_failure = jnp.asarray(False)
    evaluations = jnp.asarray(0, dtype=jnp.int32)
    maximum_bound = jnp.asarray(0.0, dtype=jnp.real(q).dtype)
    bound_available = jnp.asarray(True)
    bound_certified = jnp.asarray(True)
    for _ in range(partition.substeps):
        (
            p,
            active,
            force_finite,
            force_bound,
            force_available,
            force_certified,
        ) = _kick(
            kernel,
            fields,
            partition,
            q,
            p,
            0.5 * substep,
            active,
        )
        finite = finite & force_finite
        evaluations = evaluations + jnp.asarray(1, dtype=jnp.int32)
        maximum_bound = jnp.maximum(maximum_bound, force_bound)
        bound_available = bound_available & force_available
        bound_certified = bound_certified & force_certified
        if level + 1 == len(kernel.plan.force_plan.partitions):
            q, active, drift_finite, drift_membership = _drift(
                kernel,
                q,
                p,
                substep,
                active,
            )
            finite = finite & drift_finite
            membership_failure = membership_failure | drift_membership
        else:
            (
                q,
                p,
                active,
                inner_finite,
                inner_membership,
                inner_evaluations,
                inner_bound,
                inner_available,
                inner_certified,
            ) = _nested_level(
                kernel,
                fields,
                level + 1,
                q,
                p,
                substep,
                active,
            )
            finite = finite & inner_finite
            membership_failure = membership_failure | inner_membership
            evaluations = evaluations + inner_evaluations
            maximum_bound = jnp.maximum(maximum_bound, inner_bound)
            bound_available = bound_available & inner_available
            bound_certified = bound_certified & inner_certified
        (
            p,
            active,
            force_finite,
            force_bound,
            force_available,
            force_certified,
        ) = _kick(
            kernel,
            fields,
            partition,
            q,
            p,
            0.5 * substep,
            active,
        )
        finite = finite & force_finite
        evaluations = evaluations + jnp.asarray(1, dtype=jnp.int32)
        maximum_bound = jnp.maximum(maximum_bound, force_bound)
        bound_available = bound_available & force_available
        bound_certified = bound_certified & force_certified
    maximum_bound = jnp.where(bound_available, maximum_bound, jnp.inf)
    return (
        q,
        p,
        active,
        finite,
        membership_failure,
        evaluations,
        maximum_bound,
        bound_available,
        bound_certified,
    )


def _kick(
    kernel: PreparedRHMCKernel,
    fields: tuple[PyTree[Array], ...],
    partition: NestedForcePartition,
    q: Array,
    p: Array,
    amount: Array,
    active: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    (
        force,
        force_successful,
        force_finite,
        solve_bound,
        bound_available,
        bound_certified,
    ) = _partition_force(
        kernel,
        fields,
        partition,
        q,
    )
    candidate = p + amount * force
    finite = force_finite & jnp.all(jnp.isfinite(candidate))
    commit = active & force_successful & finite
    return (
        jnp.where(commit, candidate, p),
        commit,
        finite,
        solve_bound,
        bound_available,
        bound_certified,
    )


def _drift(
    kernel: PreparedRHMCKernel,
    q: Array,
    p: Array,
    amount: Array,
    active: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    candidate = kernel.geometry.retract(q, amount * _velocity(kernel, p))
    member = kernel.geometry.contains(candidate)
    finite = jnp.all(jnp.isfinite(candidate))
    commit = active & member & finite
    return (
        jnp.where(commit, candidate, q),
        commit,
        finite,
        active & ~member,
    )


def _partition_force(
    kernel: PreparedRHMCKernel,
    fields: tuple[PyTree[Array], ...],
    partition: NestedForcePartition,
    configuration: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    zero = jnp.zeros(
        kernel.local_coordinate_shape,
        dtype=jnp.real(configuration).dtype,
    )
    total = jnp.zeros_like(zero)
    successful = jnp.asarray(True)
    finite = jnp.asarray(True)
    maximum_bound = jnp.asarray(0.0, dtype=zero.dtype)
    bound_available = jnp.asarray(True)
    bound_certified = jnp.asarray(True)
    action_count = len(kernel.registry.action_terms)
    selected = set(partition.term_indices)
    for index, term in enumerate(kernel.registry.action_terms):
        if index in selected:
            gradient = jax.grad(
                lambda local, action_term=term: action_term(
                    kernel.geometry.retract(configuration, local)
                )
            )(zero)
            term_finite = jnp.all(jnp.isfinite(gradient))
            total = total - gradient
            successful = successful & term_finite
            finite = finite & term_finite
    for field_index, (term, field) in enumerate(
        zip(kernel.registry.pseudofermion_terms, fields, strict=True)
    ):
        registry_index = action_count + field_index
        if registry_index in selected:
            result = pseudofermion_force(term, field, configuration)
            local_force = kernel.geometry.retraction_vjp(
                configuration,
                zero,
                result.force,
            )
            term_finite = result.finite & jnp.all(jnp.isfinite(local_force))
            total = total + local_force
            successful = successful & result.successful & term_finite
            finite = finite & term_finite
            maximum_bound = jnp.maximum(
                maximum_bound,
                result.maximum_shifted_solution_error_upper_bound,
            )
            bound_available = bound_available & result.solve_error_bound_available
            bound_certified = bound_certified & result.solve_error_bound_certified
    maximum_bound = jnp.where(bound_available, maximum_bound, jnp.inf)
    return (
        total,
        successful,
        finite & jnp.all(jnp.isfinite(total)),
        maximum_bound,
        bound_available,
        bound_certified,
    )


def _coordinate_metric(
    geometry: AbstractStateGeometry,
    configuration_shape: tuple[int, ...],
    local_shape: tuple[int, ...],
    /,
) -> LieAlgebraCoordinateMetric | None:
    if isinstance(geometry, (EuclideanStateGeometry, FlatTorusStateGeometry)) or (
        geometry.trivial and geometry.retraction_method == "addition"
    ):
        if configuration_shape != local_shape:
            raise ValueError(
                "Euclidean/flat-torus RHMC requires equal point and local shapes."
            )
        return None
    if isinstance(geometry, PointwiseStateGeometry) and isinstance(
        geometry.geometry,
        LieGroupStateGeometry,
    ):
        metric = LieAlgebraCoordinateMetric(geometry.geometry.group)
        if local_shape[-1] != metric.dimension:
            raise ValueError("RHMC local coordinates do not match the Lie algebra.")
        if configuration_shape[:-2] != local_shape[:-1]:
            raise ValueError(
                "Pointwise gauge-link and local leading shapes do not match."
            )
        return metric
    raise TypeError(
        "RHMC supports EuclideanStateGeometry, FlatTorusStateGeometry, or pointwise LieGroupStateGeometry."
    )


def _sample_momentum(
    kernel: PreparedRHMCKernel,
    key: Key[Array, ""],
    /,
) -> Array:
    dtype = jnp.real(kernel.configuration_template).dtype
    if kernel.coordinate_metric is None:
        return jr.normal(key, kernel.local_coordinate_shape, dtype=dtype)
    leading_shape = kernel.local_coordinate_shape[:-1]
    return kernel.coordinate_metric.sample_momentum(key, leading_shape, dtype=dtype)


def _velocity(kernel: PreparedRHMCKernel, momentum: Array, /) -> Array:
    if kernel.coordinate_metric is None:
        return momentum
    return kernel.coordinate_metric.solve(momentum)


def _kinetic(kernel: PreparedRHMCKernel, momentum: Array, /) -> Array:
    if kernel.coordinate_metric is None:
        return 0.5 * jnp.sum(momentum**2)
    return kernel.coordinate_metric.kinetic_energy(momentum)


def _bosonic_action(registry: SeparableActionRegistry, configuration: Array, /) -> Array:
    return sum(
        (term(configuration) for term in registry.action_terms),
        jnp.asarray(0.0, dtype=jnp.real(configuration).dtype),
    )


def _space_storage_bytes(space: Any, /) -> int:
    return sum(
        int(np.prod(spec.shape) * np.dtype(spec.dtype).itemsize)
        for spec in jax.tree.leaves(space.structure())
    )


def _validate_kernel(kernel: PreparedRHMCKernel, /) -> None:
    if not isinstance(kernel, PreparedRHMCKernel):
        raise TypeError("kernel must be a PreparedRHMCKernel.")


def _validate_state(kernel: PreparedRHMCKernel, state: RHMCChainState, /) -> None:
    _validate_kernel(kernel)
    if not isinstance(state, RHMCChainState):
        raise TypeError("state must be an RHMCChainState.")
    if state.kernel_id != kernel.kernel_id:
        raise ValueError("RHMC state belongs to a different prepared kernel.")
    if state.configuration.shape != kernel.configuration_shape:
        raise ValueError("RHMC state configuration shape is incompatible.")


__all__ = [
    "NestedForcePartition",
    "NestedForcePlan",
    "PreparedRHMCKernel",
    "RegistryActionEvaluation",
    "RHMCChainState",
    "RHMCPlan",
    "RHMCResourcePolicy",
    "RHMCSampleResult",
    "RHMCTransitionEvidence",
    "RHMCTransitionResult",
    "RHMCTrajectoryResult",
    "SeparableActionRegistry",
    "SeparableActionTerm",
    "evaluate_registry_action",
    "initialize_rhmc_state",
    "integrate_rhmc_trajectory",
    "plan_rhmc",
    "prepare_rhmc",
    "rhmc_checkpoint_fingerprint",
    "rhmc_transition",
    "sample_rhmc",
]
