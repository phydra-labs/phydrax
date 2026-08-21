#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import cast, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.linalg as spla
import scipy.sparse as sp
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._assembly import (
    plan_sparse_assembly,
    prepare_sparse_assembly,
    PreparedSparseAssembly,
    refresh_sparse_assembly,
    SparseAssemblyPolicy,
)
from ._costs import _array_tree_storage_bytes, PreconditionerCostEstimate
from ._materialization import MaterializationPolicy, materialize
from ._multigrid import (
    _composition_properties,
    _source_properties,
    MultigridCyclePolicy,
    MultigridHierarchy,
    MultigridLevel,
    MultigridPreconditioner,
    MultigridSetupDiagnostics,
)
from ._operators import (
    AbstractLinearOperator,
    AdjointLinearOperator,
    DenseLinearOperator,
)
from ._pairings import DiagonalPairing, EuclideanPairing
from ._preconditioner_properties import (
    _preconditioner_properties_payload,
    PreconditionerProperties,
)
from ._preconditioners import AbstractPreconditioner
from ._preconditioning import (
    _source_cost,
    AbstractPreconditionerBuilder,
    PreconditionerSource,
)
from ._properties import LinearCapabilityError, OperatorCapabilities, OperatorProperties
from ._spaces import _coordinate_dtype, AbstractVectorSpace, ArraySpace
from ._sparse_contract import AbstractSparseLinearOperator
from ._subspaces import LinearSubspace


MultigridRefreshMode: TypeAlias = Literal[
    "rebuild-all",
    "reuse-aggregates",
    "reuse-transfers",
    "reuse-symbolic-sparse-products",
]


class GalerkinHierarchyBuilder(AbstractPreconditionerBuilder):
    """Generate every coarse operator from fixed restriction/prolongation pairs."""

    transfers: tuple[tuple[AbstractLinearOperator, AbstractLinearOperator], ...]
    smoothers: tuple[PreconditionerSource, ...]
    coarse_solver: PreconditionerSource
    properties: PreconditionerProperties
    cycle_policy: MultigridCyclePolicy
    refresh_mode: MultigridRefreshMode = eqx.field(static=True)
    pre_smoothing: int = eqx.field(static=True)
    post_smoothing: int = eqx.field(static=True)
    _properties_supplied: bool = eqx.field(static=True)
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfers: tuple[tuple[AbstractLinearOperator, AbstractLinearOperator], ...],
        smoothers: tuple[PreconditionerSource, ...],
        coarse_solver: PreconditionerSource,
        /,
        *,
        properties: PreconditionerProperties | None = None,
        cycle_policy: MultigridCyclePolicy | None = None,
        refresh_mode: MultigridRefreshMode = "rebuild-all",
        pre_smoothing: int = 1,
        post_smoothing: int = 1,
    ):
        transfers_ = tuple(tuple(pair) for pair in transfers)
        smoothers_ = tuple(smoothers)
        if not transfers_:
            raise ValueError("A Galerkin hierarchy requires at least one transfer pair.")
        if len(smoothers_) != len(transfers_):
            raise ValueError("smoothers must contain one source per non-coarse level.")
        for pair in transfers_:
            if len(pair) != 2 or not all(
                isinstance(value, AbstractLinearOperator) for value in pair
            ):
                raise TypeError(
                    "Each transfer must be a (restriction, prolongation) operator pair."
                )
            if pair[0].batch_shape or pair[1].batch_shape:
                raise ValueError("Galerkin transfer operators must be unbatched.")
        for source in (*smoothers_, coarse_solver):
            _validate_preconditioner_source(source)
        mode = _refresh_mode(refresh_mode)
        if mode == "reuse-aggregates":
            raise ValueError(
                "GalerkinHierarchyBuilder has no aggregate dependency to reuse."
            )
        pre, post = int(pre_smoothing), int(post_smoothing)
        if pre < 0 or post < 0:
            raise ValueError("Smoothing counts must be nonnegative.")
        properties_ = PreconditionerProperties() if properties is None else properties
        if not isinstance(properties_, PreconditionerProperties):
            raise TypeError("properties must be PreconditionerProperties.")
        cycle_policy_ = MultigridCyclePolicy() if cycle_policy is None else cycle_policy
        if not isinstance(cycle_policy_, MultigridCyclePolicy):
            raise TypeError("cycle_policy must be a MultigridCyclePolicy.")
        self.transfers = transfers_
        self.smoothers = smoothers_
        self.coarse_solver = coarse_solver
        self.properties = properties_
        self.cycle_policy = cycle_policy_
        self.refresh_mode = mode
        self.pre_smoothing = pre
        self.post_smoothing = post
        self._properties_supplied = properties is not None
        self._builder_id = canonical_fingerprint(
            {
                "kind": "galerkin-hierarchy-builder",
                "transfers": [
                    [restriction.operator_id, prolongation.operator_id]
                    for restriction, prolongation in transfers_
                ],
                "smoothers": [_source_identifier(value) for value in smoothers_],
                "coarse_solver": _source_identifier(coarse_solver),
                "properties": _preconditioner_properties_payload(properties_),
                "properties_supplied": properties is not None,
                "refresh_mode": mode,
                "pre_smoothing": pre,
                "post_smoothing": post,
                "cycle": cycle_policy_.cycle_id,
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> str:
        return "numeric"

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        operators = _symbolic_galerkin_operators(setup_operator, self.transfers)
        components = tuple(
            _source_properties(source, operator)
            for source, operator in zip(
                (*self.smoothers, self.coarse_solver), operators, strict=True
            )
        )
        return _composition_properties(
            components,
            self.properties if self._properties_supplied else None,
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        sources = (*self.smoothers, self.coarse_solver)
        plan = _plan_galerkin_hierarchy(
            setup_operator,
            self.transfers,
            sources,
            materialization,
            role_prefix="galerkin-coarse",
        )
        itemsize = _coordinate_dtype(setup_operator.source).itemsize
        dimensions = (
            setup_operator.source.size,
            *(prolongation.source.size for _, prolongation in self.transfers),
        )
        preparation_workspace = max(
            (
                plan.construction_workspace_bytes,
                *(estimate.preparation_workspace_bytes for estimate in plan.estimates),
            )
        )
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=(
                _galerkin_stored_state_bytes(
                    self.transfers,
                    plan.operators[1:],
                    plan.sparse_assemblies,
                )
                + sum(estimate.storage_bytes for estimate in plan.estimates)
            ),
            preparation_workspace_bytes=preparation_workspace,
            apply_workspace_bytes_per_rhs=(
                sum(4 * size * itemsize for size in dimensions[:-1])
                + sum(
                    estimate.apply_workspace_bytes_per_rhs for estimate in plan.estimates
                )
            ),
            setup_matvec_count=sum(
                estimate.setup_matvec_count for estimate in plan.estimates
            ),
            accepted=plan.accepted,
            reason=plan.reason,
        )

    def prepare_hierarchy(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> MultigridHierarchy:
        self.properties_for(setup_operator)
        decisions = tuple(
            f"level-{index}:builder-transfers-used;coarse-values-recomputed"
            for index in range(len(self.transfers))
        )
        return _prepare_galerkin_hierarchy(
            setup_operator,
            self.transfers,
            self.smoothers,
            self.coarse_solver,
            materialization=materialization,
            supplied_properties=(self.properties if self._properties_supplied else None),
            pre_smoothing=self.pre_smoothing,
            post_smoothing=self.post_smoothing,
            builder_id=self.builder_id,
            reuse_decisions=decisions,
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        return MultigridPreconditioner(
            self.prepare_hierarchy(setup_operator, materialization=materialization),
            cycle_policy=self.cycle_policy,
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(preconditioner, MultigridPreconditioner):
            raise TypeError("Galerkin refresh requires a MultigridPreconditioner.")
        self.properties_for(setup_operator)
        old = preconditioner.hierarchy
        fine_pattern = _operator_pattern_fingerprint(setup_operator)
        old_patterns = old.diagnostics.operator_pattern_fingerprints
        pattern_matches = bool(old_patterns) and old_patterns[0] == fine_pattern
        expected_transfer_ids = tuple(
            (restriction.operator_id, prolongation.operator_id)
            for restriction, prolongation in self.transfers
        )
        dependencies_match = (
            old.diagnostics.transfer_ids == expected_transfer_ids
            and len(old.levels) == len(self.transfers) + 1
        )
        reuse_requested = self.refresh_mode in (
            "reuse-transfers",
            "reuse-symbolic-sparse-products",
        )
        reuse = reuse_requested and pattern_matches and dependencies_match
        if reuse:
            old_transfers = tuple(
                (level.restriction, level.prolongation) for level in old.levels[:-1]
            )
            reuse = all(
                restriction is not None and prolongation is not None
                for restriction, prolongation in old_transfers
            )
        if reuse:
            concrete_transfers = tuple(
                (restriction, prolongation)
                for restriction, prolongation in old_transfers
                if restriction is not None and prolongation is not None
            )
            previous_sparse_assemblies = (
                old.sparse_assemblies
                if self.refresh_mode == "reuse-symbolic-sparse-products"
                else ()
            )
            decisions = tuple(
                (
                    f"level-{index}:transfers-reused;"
                    "sparse-route-reused;coarse-values-refreshed"
                    if (
                        previous_sparse_assemblies
                        and previous_sparse_assemblies[index] is not None
                    )
                    else f"level-{index}:transfers-reused;coarse-values-recomputed"
                )
                for index in range(len(concrete_transfers))
            )
        else:
            concrete_transfers = self.transfers
            previous_sparse_assemblies = ()
            if reuse_requested and not pattern_matches:
                invalidation = "reuse-invalidated-pattern-change;"
            elif reuse_requested and not dependencies_match:
                invalidation = "reuse-invalidated-transfer-dependency-change;"
            else:
                invalidation = ""
            decisions = tuple(
                f"level-{index}:{invalidation}builder-transfers-used;"
                "coarse-values-recomputed"
                for index in range(len(concrete_transfers))
            )
        hierarchy = _prepare_galerkin_hierarchy(
            setup_operator,
            concrete_transfers,
            self.smoothers,
            self.coarse_solver,
            materialization=materialization,
            supplied_properties=(self.properties if self._properties_supplied else None),
            pre_smoothing=self.pre_smoothing,
            post_smoothing=self.post_smoothing,
            builder_id=self.builder_id,
            reuse_decisions=decisions,
            previous_sparse_assemblies=previous_sparse_assemblies,
            previous_levels=(
                old.levels
                if (
                    reuse
                    and old.diagnostics.reuse_dependency_fingerprint == self.builder_id
                )
                else ()
            ),
        )
        return MultigridPreconditioner(hierarchy, cycle_policy=self.cycle_policy)


class SmoothedAggregationPolicy(StrictModule):
    """Deterministic coarsening, tentative-basis, and stopping controls."""

    strength_threshold: float = eqx.field(static=True)
    max_levels: int = eqx.field(static=True)
    minimum_coarse_size: int = eqx.field(static=True)
    prolongation_smoothing_steps: int = eqx.field(static=True)
    prolongation_damping: float = eqx.field(static=True)
    candidate_rank_tolerance: float | None = eqx.field(static=True)
    pre_smoothing: int = eqx.field(static=True)
    post_smoothing: int = eqx.field(static=True)
    maximum_grid_complexity: float | None = eqx.field(static=True)
    maximum_operator_complexity: float | None = eqx.field(static=True)
    maximum_level_storage_bytes: int | None = eqx.field(static=True)
    maximum_compatible_relaxation_factor: float | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        strength_threshold: float = 0.25,
        max_levels: int = 10,
        minimum_coarse_size: int = 8,
        prolongation_smoothing_steps: int = 1,
        prolongation_damping: float = 2.0 / 3.0,
        candidate_rank_tolerance: float | None = None,
        pre_smoothing: int = 1,
        post_smoothing: int = 1,
        maximum_grid_complexity: float | None = None,
        maximum_operator_complexity: float | None = None,
        maximum_level_storage_bytes: int | None = None,
        maximum_compatible_relaxation_factor: float | None = None,
    ):
        threshold = float(strength_threshold)
        levels = int(max_levels)
        coarse_size = int(minimum_coarse_size)
        steps = int(prolongation_smoothing_steps)
        damping = float(prolongation_damping)
        rank_tolerance = (
            None if candidate_rank_tolerance is None else float(candidate_rank_tolerance)
        )
        pre, post = int(pre_smoothing), int(post_smoothing)
        grid_limit = (
            None if maximum_grid_complexity is None else float(maximum_grid_complexity)
        )
        operator_limit = (
            None
            if maximum_operator_complexity is None
            else float(maximum_operator_complexity)
        )
        level_byte_limit = (
            None
            if maximum_level_storage_bytes is None
            else int(maximum_level_storage_bytes)
        )
        relaxation_limit = (
            None
            if maximum_compatible_relaxation_factor is None
            else float(maximum_compatible_relaxation_factor)
        )
        if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError("strength_threshold must lie in [0, 1].")
        if levels < 2:
            raise ValueError("max_levels must be at least two.")
        if coarse_size < 1:
            raise ValueError("minimum_coarse_size must be positive.")
        if steps < 0:
            raise ValueError("prolongation_smoothing_steps must be nonnegative.")
        if not np.isfinite(damping) or damping <= 0.0:
            raise ValueError("prolongation_damping must be finite and positive.")
        if rank_tolerance is not None and (
            not np.isfinite(rank_tolerance) or rank_tolerance <= 0.0
        ):
            raise ValueError(
                "candidate_rank_tolerance must be finite and positive when supplied."
            )
        if pre < 0 or post < 0:
            raise ValueError("Smoothing counts must be nonnegative.")
        if grid_limit is not None and (not np.isfinite(grid_limit) or grid_limit < 1.0):
            raise ValueError("maximum_grid_complexity must be finite and at least one.")
        if operator_limit is not None and (
            not np.isfinite(operator_limit) or operator_limit < 1.0
        ):
            raise ValueError(
                "maximum_operator_complexity must be finite and at least one."
            )
        if level_byte_limit is not None and level_byte_limit < 1:
            raise ValueError("maximum_level_storage_bytes must be positive.")
        if relaxation_limit is not None and (
            not np.isfinite(relaxation_limit) or relaxation_limit < 0.0
        ):
            raise ValueError(
                "maximum_compatible_relaxation_factor must be finite and nonnegative."
            )
        self.strength_threshold = threshold
        self.max_levels = levels
        self.minimum_coarse_size = coarse_size
        self.prolongation_smoothing_steps = steps
        self.prolongation_damping = damping
        self.candidate_rank_tolerance = rank_tolerance
        self.pre_smoothing = pre
        self.post_smoothing = post
        self.maximum_grid_complexity = grid_limit
        self.maximum_operator_complexity = operator_limit
        self.maximum_level_storage_bytes = level_byte_limit
        self.maximum_compatible_relaxation_factor = relaxation_limit


class SmoothedAggregationHierarchyBuilder(AbstractPreconditionerBuilder):
    """Host-setup deterministic smoothed aggregation for explicit operators."""

    policy: SmoothedAggregationPolicy
    smoother: PreconditionerSource
    coarse_solver: PreconditionerSource
    near_nullspaces: tuple[LinearSubspace, ...]
    properties: PreconditionerProperties
    cycle_policy: MultigridCyclePolicy
    refresh_mode: MultigridRefreshMode = eqx.field(static=True)
    _properties_supplied: bool = eqx.field(static=True)
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        policy: SmoothedAggregationPolicy,
        smoother: PreconditionerSource,
        coarse_solver: PreconditionerSource,
        /,
        *,
        near_nullspaces: tuple[LinearSubspace, ...] = (),
        properties: PreconditionerProperties | None = None,
        cycle_policy: MultigridCyclePolicy | None = None,
        refresh_mode: MultigridRefreshMode = "rebuild-all",
    ):
        if not isinstance(policy, SmoothedAggregationPolicy):
            raise TypeError("policy must be SmoothedAggregationPolicy.")
        _validate_preconditioner_source(smoother)
        _validate_preconditioner_source(coarse_solver)
        near_nullspaces_ = tuple(near_nullspaces)
        if not all(
            isinstance(candidate, LinearSubspace) for candidate in near_nullspaces_
        ):
            raise TypeError("near_nullspaces must contain LinearSubspace values.")
        properties_ = PreconditionerProperties() if properties is None else properties
        if not isinstance(properties_, PreconditionerProperties):
            raise TypeError("properties must be PreconditionerProperties.")
        cycle_policy_ = MultigridCyclePolicy() if cycle_policy is None else cycle_policy
        if not isinstance(cycle_policy_, MultigridCyclePolicy):
            raise TypeError("cycle_policy must be a MultigridCyclePolicy.")
        mode = _refresh_mode(refresh_mode)
        self.policy = policy
        self.smoother = smoother
        self.coarse_solver = coarse_solver
        self.near_nullspaces = near_nullspaces_
        self.properties = properties_
        self.cycle_policy = cycle_policy_
        self.refresh_mode = mode
        self._properties_supplied = properties is not None
        self._builder_id = canonical_fingerprint(
            {
                "kind": "smoothed-aggregation-hierarchy-builder",
                "policy": {
                    "strength_threshold": policy.strength_threshold,
                    "max_levels": policy.max_levels,
                    "minimum_coarse_size": policy.minimum_coarse_size,
                    "prolongation_smoothing_steps": (policy.prolongation_smoothing_steps),
                    "prolongation_damping": policy.prolongation_damping,
                    "candidate_rank_tolerance": policy.candidate_rank_tolerance,
                    "pre_smoothing": policy.pre_smoothing,
                    "post_smoothing": policy.post_smoothing,
                    "maximum_grid_complexity": policy.maximum_grid_complexity,
                    "maximum_operator_complexity": policy.maximum_operator_complexity,
                    "maximum_level_storage_bytes": (policy.maximum_level_storage_bytes),
                    "maximum_compatible_relaxation_factor": (
                        policy.maximum_compatible_relaxation_factor
                    ),
                },
                "smoother": _source_identifier(smoother),
                "coarse_solver": _source_identifier(coarse_solver),
                "near_nullspaces": [
                    {
                        "id": candidate.subspace_id,
                        "basis": array_tree_fingerprint(candidate.basis),
                        "dimension": int(np.asarray(candidate.dimension)),
                    }
                    for candidate in near_nullspaces_
                ],
                "properties": _preconditioner_properties_payload(properties_),
                "properties_supplied": properties is not None,
                "refresh_mode": mode,
                "cycle": cycle_policy_.cycle_id,
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> str:
        return "numeric"

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        _validate_sa_operator(setup_operator, self.near_nullspaces)
        components = (
            _source_properties(self.smoother, setup_operator),
            _source_properties(self.coarse_solver, setup_operator),
        )
        return _composition_properties(
            components,
            self.properties if self._properties_supplied else None,
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        _validate_endomorphism(setup_operator)
        if materialization is not None and not isinstance(
            materialization, MaterializationPolicy
        ):
            raise TypeError("materialization must be a MaterializationPolicy or None.")
        explicit = isinstance(
            setup_operator, (DenseLinearOperator, AbstractSparseLinearOperator)
        )
        semantic_candidates = isinstance(setup_operator.source, ArraySpace) or bool(
            self.near_nullspaces
        )
        candidate_space_matches = all(
            candidate.space.compatible(setup_operator.source)
            for candidate in self.near_nullspaces
        )
        if isinstance(setup_operator, AbstractSparseLinearOperator):
            storage = setup_operator.sparse_storage()
            canonical_sparse = storage.canonical and storage.sorted_indices
        else:
            canonical_sparse = True
        coarsening_allowed = setup_operator.source.size > self.policy.minimum_coarse_size
        if (
            not explicit
            or not semantic_candidates
            or not candidate_space_matches
            or not canonical_sparse
            or not coarsening_allowed
        ):
            return PreconditionerCostEstimate(
                component=self.builder_id,
                accepted=False,
                reason=(
                    "smoothed aggregation requires an explicit dense/canonical-CSR "
                    "operator, matching explicit candidates for semantic spaces, "
                    "and a fine dimension above minimum_coarse_size"
                ),
            )
        (
            operators,
            hierarchy_storage,
            setup_workspace,
            construction_modes,
            planning_reason,
        ) = _plan_smoothed_aggregation_levels(
            self,
            setup_operator,
            materialization=materialization,
        )
        if not operators:
            return PreconditionerCostEstimate(
                component=self.builder_id,
                accepted=False,
                reason=planning_reason,
            )
        estimates = tuple(
            _source_cost(self.smoother, operator, materialization=materialization)
            for operator in operators[:-1]
        ) + (
            _source_cost(
                self.coarse_solver,
                operators[-1],
                materialization=materialization,
            ),
        )
        rejected = tuple(
            (index, estimate)
            for index, estimate in enumerate(estimates)
            if not estimate.accepted
        )
        itemsize = _coordinate_dtype(setup_operator.source).itemsize
        accepted = not rejected
        if accepted:
            downstream_reason = ";".join(
                f"level-{index}:{estimate.reason}"
                for index, estimate in enumerate(estimates)
            )
            reason = (
                "deterministic smoothed-aggregation plan "
                f"({','.join(construction_modes)});{planning_reason};"
                f"{downstream_reason}"
            )
        else:
            index, estimate = rejected[0]
            role = "coarse solver" if index == len(estimates) - 1 else "smoother"
            route = "input-operator" if index == 0 else construction_modes[index - 1]
            reason = (
                f"level-{index} {role} on {route} rejected setup: "
                f"{estimate.reason}; {planning_reason}"
            )
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=hierarchy_storage
            + sum(estimate.storage_bytes for estimate in estimates),
            preparation_workspace_bytes=max(
                setup_workspace,
                *(estimate.preparation_workspace_bytes for estimate in estimates),
            ),
            apply_workspace_bytes_per_rhs=(
                sum(4 * operator.source.size * itemsize for operator in operators[:-1])
                + sum(estimate.apply_workspace_bytes_per_rhs for estimate in estimates)
            ),
            setup_matvec_count=sum(estimate.setup_matvec_count for estimate in estimates),
            accepted=accepted,
            reason=reason,
        )

    def prepare_hierarchy(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> MultigridHierarchy:
        self.properties_for(setup_operator)
        return self._prepare_hierarchy(
            setup_operator,
            materialization=materialization,
            old_hierarchy=None,
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        return MultigridPreconditioner(
            self.prepare_hierarchy(setup_operator, materialization=materialization),
            cycle_policy=self.cycle_policy,
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(preconditioner, MultigridPreconditioner):
            raise TypeError(
                "Smoothed-aggregation refresh requires a MultigridPreconditioner."
            )
        self.properties_for(setup_operator)
        hierarchy = self._prepare_hierarchy(
            setup_operator,
            materialization=materialization,
            old_hierarchy=preconditioner.hierarchy,
        )
        return MultigridPreconditioner(hierarchy, cycle_policy=self.cycle_policy)

    def _prepare_hierarchy(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
        old_hierarchy: MultigridHierarchy | None,
    ) -> MultigridHierarchy:
        initial_rejection = _hierarchy_limit_rejection(
            self.policy,
            [setup_operator],
        )
        if initial_rejection is not None:
            raise LinearCapabilityError(initial_rejection)
        matrix = _explicit_host_matrix(setup_operator)
        candidates = _initial_candidates(setup_operator.source, self.near_nullspaces)
        requested_mode = self.refresh_mode
        effective_mode: MultigridRefreshMode = requested_mode
        invalidation = ""
        old_patterns: tuple[str, ...] = ()
        old_assignments: tuple[tuple[int, ...], ...] = ()
        old_transfers: tuple[
            tuple[AbstractLinearOperator, AbstractLinearOperator], ...
        ] = ()
        old_sparse_assemblies: tuple[PreparedSparseAssembly | None, ...] = ()
        old_relaxation_factors: tuple[float, ...] = ()
        old_candidate_ranks: tuple[tuple[int, ...], ...] = ()
        if old_hierarchy is None and requested_mode != "rebuild-all":
            effective_mode = "rebuild-all"
            invalidation = "initial-prepare-no-reusable-state;"
        if old_hierarchy is not None and requested_mode != "rebuild-all":
            diagnostics = old_hierarchy.diagnostics
            old_patterns = diagnostics.operator_pattern_fingerprints
            old_relaxation_factors = diagnostics.compatible_relaxation_factors
            old_candidate_ranks = diagnostics.aggregate_candidate_ranks
            pattern_matches = bool(old_patterns) and old_patterns[
                0
            ] == _operator_pattern_fingerprint(setup_operator)
            dependency_matches = (
                diagnostics.reuse_dependency_fingerprint == self.builder_id
            )
            if not dependency_matches:
                effective_mode = "rebuild-all"
                invalidation = "reuse-invalidated-builder-dependency-change;"
            elif not pattern_matches:
                effective_mode = "rebuild-all"
                invalidation = "reuse-invalidated-pattern-change;"
            elif requested_mode == "reuse-aggregates":
                old_assignments = diagnostics.aggregate_assignments
                if not old_assignments:
                    effective_mode = "rebuild-all"
                    invalidation = "reuse-invalidated-missing-aggregate-state;"
            elif requested_mode in (
                "reuse-transfers",
                "reuse-symbolic-sparse-products",
            ):
                transfer_values = tuple(
                    (level.restriction, level.prolongation)
                    for level in old_hierarchy.levels[:-1]
                )
                aggregate_state = diagnostics.aggregate_assignments
                if (
                    any(
                        restriction is None or prolongation is None
                        for restriction, prolongation in transfer_values
                    )
                    or len(aggregate_state) != len(transfer_values)
                    or len(old_patterns) != len(transfer_values) + 1
                ):
                    effective_mode = "rebuild-all"
                    invalidation = "reuse-invalidated-missing-transfer-dependencies;"
                else:
                    old_transfers = tuple(
                        (restriction, prolongation)
                        for restriction, prolongation in transfer_values
                        if restriction is not None and prolongation is not None
                    )
                    old_assignments = aggregate_state
                    old_sparse_assemblies = old_hierarchy.sparse_assemblies
        operators: list[AbstractLinearOperator] = [setup_operator]
        transfers: list[tuple[AbstractLinearOperator, AbstractLinearOperator]] = []
        assignments: list[tuple[int, ...]] = []
        construction_modes: list[str] = []
        sparse_assemblies: list[PreparedSparseAssembly | None] = []
        decisions: list[str] = []
        relaxation_factors: list[float] = []
        candidate_ranks: list[tuple[int, ...]] = []
        workspace_bytes = int(
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        )
        stop_reason = "maximum-levels"
        if effective_mode in (
            "reuse-transfers",
            "reuse-symbolic-sparse-products",
        ):
            current_operator = setup_operator
            for index, pair in enumerate(old_transfers):
                restriction, prolongation = pair
                _validate_transfer_pair(
                    current_operator, restriction, prolongation, index
                )
                coarse, construction_mode, workspace, sparse_assembly = _galerkin_step(
                    current_operator,
                    restriction,
                    prolongation,
                    self.coarse_solver
                    if index == len(old_transfers) - 1
                    else self.smoother,
                    materialization,
                    role=f"sa-coarse-{index}",
                    allow_matrix_free=index == len(old_transfers) - 1,
                    resident_operators=(
                        *operators,
                        *(value for used_pair in transfers for value in used_pair),
                        restriction,
                        prolongation,
                    ),
                    previous_sparse_assembly=(
                        old_sparse_assemblies[index]
                        if effective_mode == "reuse-symbolic-sparse-products"
                        else None
                    ),
                )
                operators.append(coarse)
                limit_rejection = _hierarchy_limit_rejection(self.policy, operators)
                if limit_rejection is not None:
                    raise LinearCapabilityError(limit_rejection)
                transfers.append(pair)
                construction_modes.append(construction_mode)
                sparse_assemblies.append(sparse_assembly)
                workspace_bytes = max(workspace_bytes, workspace)
                product_decision = (
                    "sparse-route-reused;coarse-values-refreshed"
                    if (
                        effective_mode == "reuse-symbolic-sparse-products"
                        and old_sparse_assemblies[index] is not None
                    )
                    else "coarse-values-recomputed"
                )
                decisions.append(
                    f"level-{index}:{invalidation}aggregates-reused;"
                    f"transfers-reused;{product_decision}"
                )
                current_operator = coarse
            refreshed_patterns = tuple(
                _operator_pattern_fingerprint(operator) for operator in operators
            )
            if refreshed_patterns == old_patterns:
                assignments.extend(old_assignments)
                relaxation_factors.extend(old_relaxation_factors)
                candidate_ranks.extend(old_candidate_ranks)
                stop_reason = "reused-transfer-depth"
            else:
                effective_mode = "rebuild-all"
                invalidation = "reuse-invalidated-dependent-pattern-change;"
                operators = [setup_operator]
                transfers = []
                assignments = []
                construction_modes = []
                sparse_assemblies = []
                relaxation_factors = []
                candidate_ranks = []
                decisions.append(
                    "reuse-transfers:discarded-after-dependent-pattern-change"
                )
        if effective_mode not in (
            "reuse-transfers",
            "reuse-symbolic-sparse-products",
        ):
            current_matrix = matrix
            current_operator = setup_operator
            current_candidates = candidates
            transition_limit = self.policy.max_levels - 1
            for index in range(transition_limit):
                dimension = current_operator.source.size
                if dimension <= self.policy.minimum_coarse_size:
                    stop_reason = "minimum-coarse-size"
                    break
                if effective_mode == "reuse-aggregates":
                    if index >= len(old_assignments):
                        stop_reason = "reused-aggregate-depth"
                        break
                    if (
                        index >= len(old_patterns)
                        or _operator_pattern_fingerprint(current_operator)
                        != old_patterns[index]
                    ):
                        effective_mode = "rebuild-all"
                        invalidation = f"reuse-invalidated-level-{index}-pattern-change;"
                if effective_mode == "reuse-aggregates":
                    aggregate = np.asarray(old_assignments[index], dtype=np.int64)
                    _validate_aggregate_assignment(aggregate, dimension)
                    strength_workspace = int(aggregate.nbytes)
                    aggregate_decision = "aggregates-reused"
                else:
                    strength = _strength_graph(
                        current_matrix, self.policy.strength_threshold
                    )
                    aggregate = _deterministic_aggregates(strength)
                    strength_workspace = _csr_bytes(strength) + int(aggregate.nbytes)
                    aggregate_decision = "aggregates-rebuilt"
                aggregate_count = int(aggregate.max()) + 1
                if aggregate_count >= dimension:
                    stop_reason = "no-coarsening"
                    break
                tentative, coarse_candidates, ranks = _tentative_prolongator(
                    aggregate,
                    current_candidates,
                    self.policy.candidate_rank_tolerance,
                )
                if tentative.shape[1] <= 0 or tentative.shape[1] >= dimension:
                    stop_reason = "rank-revealing-no-coarsening"
                    break
                relaxation_factor = _compatible_relaxation_factor(
                    current_matrix,
                    tentative,
                    damping=self.policy.prolongation_damping,
                )
                relaxation_limit = self.policy.maximum_compatible_relaxation_factor
                if relaxation_limit is not None and relaxation_factor > relaxation_limit:
                    raise LinearCapabilityError(
                        f"Level {index} compatible-relaxation factor "
                        f"{relaxation_factor:.6g} exceeds limit "
                        f"{relaxation_limit:.6g}."
                    )
                prolongator_matrix = _smooth_prolongator(
                    current_matrix,
                    tentative,
                    steps=self.policy.prolongation_smoothing_steps,
                    damping=self.policy.prolongation_damping,
                )
                coarse_space = ArraySpace(
                    (int(prolongator_matrix.shape[1]),),
                    dtype=_coordinate_dtype(current_operator.source),
                    space_id=canonical_fingerprint(
                        {
                            "kind": "sa-coarse-space",
                            "builder": self.builder_id,
                            "fine_space": current_operator.source.space_id,
                            "level": index + 1,
                            "aggregate": aggregate.tolist(),
                            "ranks": list(ranks),
                        }
                    ),
                )
                restriction, prolongation, transfer_workspace = _pairing_aware_transfers(
                    prolongator_matrix,
                    coarse_space,
                    current_operator.source,
                    materialization,
                    transfer_id_payload={
                        "builder": self.builder_id,
                        "fine_space": current_operator.source.space_id,
                        "level": index,
                        "aggregate": aggregate.tolist(),
                        "ranks": list(ranks),
                    },
                )
                terminal = (
                    index == transition_limit - 1
                    or prolongation.source.size <= self.policy.minimum_coarse_size
                )
                next_source = self.coarse_solver if terminal else self.smoother
                (
                    coarse,
                    construction_mode,
                    galerkin_workspace,
                    sparse_assembly,
                ) = _galerkin_step(
                    current_operator,
                    restriction,
                    prolongation,
                    next_source,
                    materialization,
                    role=f"sa-coarse-{index}",
                    allow_matrix_free=terminal,
                    resident_operators=(
                        *operators,
                        *(value for used_pair in transfers for value in used_pair),
                        restriction,
                        prolongation,
                    ),
                )
                operators.append(coarse)
                limit_rejection = _hierarchy_limit_rejection(self.policy, operators)
                if limit_rejection is not None:
                    raise LinearCapabilityError(limit_rejection)
                transfers.append((restriction, prolongation))
                assignments.append(tuple(int(value) for value in aggregate))
                relaxation_factors.append(relaxation_factor)
                candidate_ranks.append(tuple(int(rank) for rank in ranks))
                construction_modes.append(construction_mode)
                sparse_assemblies.append(sparse_assembly)
                decisions.append(
                    f"level-{index}:{invalidation}{aggregate_decision};"
                    "transfers-rebuilt;coarse-values-recomputed"
                )
                workspace_bytes = max(
                    workspace_bytes,
                    transfer_workspace,
                    galerkin_workspace,
                    strength_workspace,
                    _csr_bytes(tentative),
                    _csr_bytes(prolongator_matrix),
                    int(current_candidates.nbytes + coarse_candidates.nbytes),
                )
                current_operator = coarse
                if terminal:
                    stop_reason = (
                        "minimum-coarse-size"
                        if current_operator.source.size <= self.policy.minimum_coarse_size
                        else "maximum-levels"
                    )
                    break
                current_matrix = _explicit_host_matrix(coarse)
                current_candidates = coarse_candidates
        if not transfers:
            raise LinearCapabilityError(
                "Smoothed aggregation could not produce a smaller coarse level under "
                "the configured stopping policy."
            )
        decisions.append(f"stop:{stop_reason}")
        return _prepare_levels_from_operators(
            tuple(operators),
            tuple(transfers),
            (self.smoother,) * (len(operators) - 1),
            self.coarse_solver,
            materialization=materialization,
            supplied_properties=(self.properties if self._properties_supplied else None),
            pre_smoothing=self.policy.pre_smoothing,
            post_smoothing=self.policy.post_smoothing,
            builder_id=self.builder_id,
            construction_modes=tuple(construction_modes),
            reuse_decisions=tuple(decisions),
            setup_workspace_bytes=workspace_bytes,
            aggregate_assignments=tuple(assignments),
            sparse_assemblies=tuple(sparse_assemblies),
            compatible_relaxation_factors=tuple(relaxation_factors),
            aggregate_candidate_ranks=tuple(candidate_ranks),
            previous_levels=(
                old_hierarchy.levels
                if (
                    old_hierarchy is not None
                    and requested_mode != "rebuild-all"
                    and tuple(
                        _operator_pattern_fingerprint(operator) for operator in operators
                    )
                    == old_hierarchy.diagnostics.operator_pattern_fingerprints
                )
                else ()
            ),
        )


def _plan_smoothed_aggregation_levels(
    builder: SmoothedAggregationHierarchyBuilder,
    setup_operator: AbstractLinearOperator,
    /,
    *,
    materialization: MaterializationPolicy | None,
) -> tuple[
    tuple[AbstractLinearOperator, ...],
    int,
    int,
    tuple[str, ...],
    str,
]:
    """Construct deterministic typed level operators without preparing actions."""
    initial_rejection = _hierarchy_limit_rejection(
        builder.policy,
        [setup_operator],
    )
    if initial_rejection is not None:
        return (), 0, 0, (), initial_rejection
    current_matrix = _explicit_host_matrix(setup_operator)
    current_operator = setup_operator
    current_candidates = _initial_candidates(
        setup_operator.source, builder.near_nullspaces
    )
    operators: list[AbstractLinearOperator] = [setup_operator]
    construction_modes: list[str] = []
    transfers: list[tuple[AbstractLinearOperator, AbstractLinearOperator]] = []
    sparse_assemblies: list[PreparedSparseAssembly | None] = []
    fallback_reasons: list[str] = []
    workspace_bytes = max(_csr_bytes(current_matrix), int(current_candidates.nbytes))
    aggregate_entries = 0
    stop_reason = "maximum-levels"
    for index in range(builder.policy.max_levels - 1):
        dimension = current_operator.source.size
        if dimension <= builder.policy.minimum_coarse_size:
            stop_reason = "minimum-coarse-size"
            break
        strength = _strength_graph(current_matrix, builder.policy.strength_threshold)
        aggregate = _deterministic_aggregates(strength)
        strength_workspace = _csr_bytes(strength) + int(aggregate.nbytes)
        aggregate_count = int(aggregate.max()) + 1
        if aggregate_count >= dimension:
            stop_reason = "no-coarsening"
            break
        tentative, coarse_candidates, ranks = _tentative_prolongator(
            aggregate,
            current_candidates,
            builder.policy.candidate_rank_tolerance,
        )
        if tentative.shape[1] <= 0 or tentative.shape[1] >= dimension:
            stop_reason = "rank-revealing-no-coarsening"
            break
        relaxation_factor = _compatible_relaxation_factor(
            current_matrix,
            tentative,
            damping=builder.policy.prolongation_damping,
        )
        relaxation_limit = builder.policy.maximum_compatible_relaxation_factor
        if relaxation_limit is not None and relaxation_factor > relaxation_limit:
            return (
                (),
                0,
                workspace_bytes,
                (),
                f"level-{index} compatible-relaxation factor "
                f"{relaxation_factor:.6g} exceeds limit {relaxation_limit:.6g}",
            )
        diagonal = current_matrix.diagonal()
        if builder.policy.prolongation_smoothing_steps and (
            np.any(~np.isfinite(diagonal)) or np.any(np.abs(diagonal) == 0.0)
        ):
            return (
                (),
                0,
                workspace_bytes,
                (),
                f"level-{index} smoothed prolongation requires finite nonzero "
                "diagonal entries",
            )
        prolongator = _smooth_prolongator(
            current_matrix,
            tentative,
            steps=builder.policy.prolongation_smoothing_steps,
            damping=builder.policy.prolongation_damping,
        )
        coarse_dimension = int(prolongator.shape[1])
        coarse_space = ArraySpace(
            (coarse_dimension,),
            dtype=_coordinate_dtype(current_operator.source),
            space_id=canonical_fingerprint(
                {
                    "kind": "sa-coarse-space",
                    "builder": builder.builder_id,
                    "fine_space": current_operator.source.space_id,
                    "level": index + 1,
                    "aggregate": aggregate.tolist(),
                    "ranks": list(ranks),
                }
            ),
        )
        sparse_pairing_route = isinstance(
            current_operator.source, ArraySpace
        ) and isinstance(
            current_operator.source.pairing,
            (EuclideanPairing, DiagonalPairing),
        )
        if not sparse_pairing_route and materialization is None:
            return (
                (),
                0,
                workspace_bytes,
                (),
                f"level-{index} pairing-aware dense transfers require an "
                "active materialization policy",
            )
        prolongator_matrix = prolongator.tocsr(copy=True)
        prolongator_matrix.sum_duplicates()
        prolongator_matrix.sort_indices()
        restriction, prolongation, transfer_workspace = _pairing_aware_transfers(
            prolongator_matrix,
            coarse_space,
            current_operator.source,
            cast(MaterializationPolicy, materialization),
            transfer_id_payload={
                "builder": builder.builder_id,
                "fine_space": current_operator.source.space_id,
                "level": index,
                "aggregate": aggregate.tolist(),
                "ranks": list(ranks),
            },
        )
        transfers.append((restriction, prolongation))
        used_transfers = tuple(value for pair in transfers for value in pair)
        terminal = (
            coarse_dimension <= builder.policy.minimum_coarse_size
            or index == builder.policy.max_levels - 2
        )
        construction = _construct_galerkin_operator(
            current_operator,
            restriction,
            prolongation,
            materialization,
            role=f"sa-coarse-{index}",
            resident_operators=(*operators, *used_transfers),
            allow_matrix_free=terminal,
        )
        coarse_operator = construction.operator
        if construction.mode == "matrix-free-composition" and not terminal:
            fallback = (
                "matrix-free fallback is not allowed before a nonterminal "
                "smoothed-aggregation level"
                if construction.fallback_reason is None
                else construction.fallback_reason
            )
            return (
                (),
                0,
                workspace_bytes,
                (),
                f"level-{index} requires an explicit coarse operator for the "
                f"next strength/aggregate setup: {fallback}",
            )
        operators.append(coarse_operator)
        limit_rejection = _hierarchy_limit_rejection(builder.policy, operators)
        if limit_rejection is not None:
            return (), 0, workspace_bytes, (), limit_rejection
        construction_modes.append(construction.mode)
        sparse_assemblies.append(construction.sparse_assembly)
        aggregate_entries += int(aggregate.size)
        if construction.fallback_reason is not None:
            fallback_reasons.append(f"level-{index}:{construction.fallback_reason}")
        workspace_bytes = max(
            workspace_bytes,
            strength_workspace,
            _csr_bytes(tentative),
            _csr_bytes(prolongator_matrix),
            int(current_candidates.nbytes + coarse_candidates.nbytes),
            transfer_workspace,
            construction.workspace_bytes,
        )
        current_operator = coarse_operator
        current_candidates = coarse_candidates
        if terminal:
            stop_reason = (
                "minimum-coarse-size"
                if coarse_dimension <= builder.policy.minimum_coarse_size
                else "maximum-levels"
            )
            break
        current_matrix = _explicit_host_matrix(coarse_operator)
    if len(operators) == 1:
        return (
            (),
            0,
            workspace_bytes,
            (),
            "smoothed aggregation could not produce a smaller coarse level "
            f"(stop:{stop_reason})",
        )
    planning_reasons = [f"stop:{stop_reason}", *fallback_reasons]
    hierarchy_storage = (
        _galerkin_stored_state_bytes(
            tuple(transfers),
            tuple(operators[1:]),
            tuple(sparse_assemblies),
        )
        + 8 * aggregate_entries
    )
    return (
        tuple(operators),
        hierarchy_storage,
        workspace_bytes,
        tuple(construction_modes),
        ";".join(planning_reasons),
    )


def _validate_preconditioner_source(source: PreconditionerSource, /) -> None:
    if not isinstance(source, (AbstractPreconditioner, AbstractPreconditionerBuilder)):
        raise TypeError("Preconditioner sources must be prepared actions or builders.")


def _source_identifier(source: PreconditionerSource, /) -> str:
    return (
        source.preconditioner_id
        if isinstance(source, AbstractPreconditioner)
        else source.builder_id
    )


def _refresh_mode(value: str, /) -> MultigridRefreshMode:
    mode = str(value)
    if mode not in (
        "rebuild-all",
        "reuse-aggregates",
        "reuse-transfers",
        "reuse-symbolic-sparse-products",
    ):
        raise ValueError(f"Unsupported multigrid refresh mode {mode!r}.")
    return mode


def _validate_endomorphism(operator: AbstractLinearOperator, /) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("setup_operator must be an AbstractLinearOperator.")
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("Hierarchy setup requires an unbatched endomorphism.")


def _validate_sa_operator(
    operator: AbstractLinearOperator,
    near_nullspaces: tuple[LinearSubspace, ...],
    /,
) -> None:
    _validate_endomorphism(operator)
    if not isinstance(operator, (DenseLinearOperator, AbstractSparseLinearOperator)):
        raise LinearCapabilityError(
            "Smoothed aggregation requires an explicit dense or canonical sparse "
            "operator; matrix-free setup is not supported."
        )
    if not near_nullspaces and not isinstance(operator.source, ArraySpace):
        raise LinearCapabilityError(
            "Semantic vector spaces require explicit near-nullspace candidates."
        )
    if any(
        not candidate.space.compatible(operator.source) for candidate in near_nullspaces
    ):
        raise ValueError("Every near-nullspace block must belong to the fine space.")
    if isinstance(operator, AbstractSparseLinearOperator):
        storage = operator.sparse_storage()
        if not storage.canonical or not storage.sorted_indices:
            raise LinearCapabilityError(
                "Smoothed aggregation requires canonical sorted CSR storage."
            )


def _validate_transfer_pair(
    operator: AbstractLinearOperator,
    restriction: AbstractLinearOperator,
    prolongation: AbstractLinearOperator,
    level: int,
    /,
) -> None:
    _validate_endomorphism(operator)
    if restriction.batch_shape or prolongation.batch_shape:
        raise ValueError("Galerkin transfers must be unbatched.")
    if not restriction.source.compatible(operator.source):
        raise ValueError(f"Restriction source mismatch at transition {level}.")
    if not prolongation.target.compatible(operator.target):
        raise ValueError(f"Prolongation target mismatch at transition {level}.")
    if not restriction.target.compatible(prolongation.source):
        raise ValueError(f"Coarse transfer-space mismatch at transition {level}.")


def _symbolic_galerkin_operators(
    setup_operator: AbstractLinearOperator,
    transfers: tuple[tuple[AbstractLinearOperator, AbstractLinearOperator], ...],
    /,
) -> tuple[AbstractLinearOperator, ...]:
    _validate_endomorphism(setup_operator)
    operators: list[AbstractLinearOperator] = [setup_operator]
    current = setup_operator
    for index, (restriction, prolongation) in enumerate(transfers):
        _validate_transfer_pair(current, restriction, prolongation, index)
        current = restriction @ current @ prolongation
        operators.append(current)
    return tuple(operators)


class _MatrixFreeGalerkinOperator(AbstractLinearOperator):
    """Composition envelope that permanently forbids downstream materialization."""

    operator: AbstractLinearOperator

    def __init__(self, operator: AbstractLinearOperator, /, *, role: str):
        self.source = operator.source
        self.target = operator.target
        self.operator = operator
        self.properties = operator.properties
        self.capabilities = OperatorCapabilities(
            transpose=operator.capabilities.transpose,
            adjoint=operator.capabilities.adjoint,
            materialize=False,
        )
        self.batch_shape = operator.batch_shape
        self.operator_id = canonical_fingerprint(
            {
                "kind": "matrix-free-galerkin",
                "role": role,
                "composition": operator.operator_id,
            }
        )

    def mv(self, vector, /):
        return self.operator.mv(vector)

    def transpose_mv(self, vector, /):
        return self.operator.transpose_mv(vector)

    def adjoint_mv(self, vector, /):
        return self.operator.adjoint_mv(vector)

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "Matrix-free Galerkin operators cannot be materialized."
        )


class _GalerkinConstruction(NamedTuple):
    operator: AbstractLinearOperator
    mode: str
    workspace_bytes: int
    fallback_reason: str | None
    sparse_assembly: PreparedSparseAssembly | None


class _GalerkinPlan(NamedTuple):
    operators: tuple[AbstractLinearOperator, ...]
    modes: tuple[str, ...]
    estimates: tuple[PreconditionerCostEstimate, ...]
    construction_workspace_bytes: int
    sparse_assemblies: tuple[PreparedSparseAssembly | None, ...]
    accepted: bool
    reason: str


def _plan_galerkin_hierarchy(
    setup_operator: AbstractLinearOperator,
    transfers: tuple[tuple[AbstractLinearOperator, AbstractLinearOperator], ...],
    sources: tuple[PreconditionerSource, ...],
    materialization: MaterializationPolicy | None,
    /,
    *,
    role_prefix: str,
    previous_sparse_assemblies: tuple[PreparedSparseAssembly | None, ...] = (),
) -> _GalerkinPlan:
    if len(sources) != len(transfers) + 1:
        raise ValueError("Generated hierarchy source counts are inconsistent.")
    if materialization is not None and not isinstance(
        materialization, MaterializationPolicy
    ):
        raise TypeError("materialization must be a MaterializationPolicy or None.")
    if previous_sparse_assemblies and (len(previous_sparse_assemblies) != len(transfers)):
        raise ValueError(
            "Previous sparse assemblies must align with Galerkin transitions."
        )
    _validate_endomorphism(setup_operator)
    operators: list[AbstractLinearOperator] = [setup_operator]
    modes: list[str] = []
    estimates: list[PreconditionerCostEstimate] = []
    sparse_assemblies: list[PreparedSparseAssembly | None] = []
    route_descriptions: list[str] = []
    workspace = 0
    for source_index, source in enumerate(sources):
        operator = operators[-1]
        if isinstance(source, AbstractPreconditioner) and not (
            source.space.compatible(operator.source)
        ):
            return _GalerkinPlan(
                tuple(operators),
                tuple(modes),
                tuple(estimates),
                workspace,
                tuple(sparse_assemblies),
                False,
                f"level-{source_index} {modes[-1] if modes else 'input-operator'} "
                "requires a supplied action on the generated level space",
            )
        estimate = _source_cost(
            source,
            operator,
            materialization=materialization,
        )
        estimates.append(estimate)
        if not estimate.accepted:
            route = modes[-1] if modes else "input-operator"
            fallback = (
                f" ({route_descriptions[-1]})"
                if modes and route_descriptions[-1] != route
                else ""
            )
            return _GalerkinPlan(
                tuple(operators),
                tuple(modes),
                tuple(estimates),
                workspace,
                tuple(sparse_assemblies),
                False,
                f"level-{source_index} {route}{fallback} rejected setup: "
                f"{estimate.reason}",
            )
        if source_index == len(transfers):
            break
        restriction, prolongation = transfers[source_index]
        _validate_transfer_pair(
            operator,
            restriction,
            prolongation,
            source_index,
        )
        used_transfers = tuple(
            value for pair in transfers[: source_index + 1] for value in pair
        )
        previous = (
            None
            if not previous_sparse_assemblies
            else previous_sparse_assemblies[source_index]
        )
        construction = (
            _construct_galerkin_operator(
                operator,
                restriction,
                prolongation,
                materialization,
                role=f"{role_prefix}-{source_index}",
                resident_operators=(*operators, *used_transfers),
            )
            if previous is None
            else _refresh_galerkin_operator(
                operator,
                restriction,
                prolongation,
                previous,
            )
        )
        operators.append(construction.operator)
        modes.append(construction.mode)
        sparse_assemblies.append(construction.sparse_assembly)
        route_descriptions.append(
            construction.mode
            if construction.fallback_reason is None
            else f"{construction.mode}; {construction.fallback_reason}"
        )
        workspace = max(workspace, construction.workspace_bytes)
    routes = ", ".join(route_descriptions)
    return _GalerkinPlan(
        tuple(operators),
        tuple(modes),
        tuple(estimates),
        workspace,
        tuple(sparse_assemblies),
        True,
        f"generated Galerkin hierarchy via {routes}",
    )


def _prepare_galerkin_hierarchy(
    setup_operator: AbstractLinearOperator,
    transfers: tuple[tuple[AbstractLinearOperator, AbstractLinearOperator], ...],
    smoothers: tuple[PreconditionerSource, ...],
    coarse_solver: PreconditionerSource,
    /,
    *,
    materialization: MaterializationPolicy,
    supplied_properties: PreconditionerProperties | None,
    pre_smoothing: int,
    post_smoothing: int,
    builder_id: str,
    reuse_decisions: tuple[str, ...],
    previous_sparse_assemblies: tuple[PreparedSparseAssembly | None, ...] = (),
    previous_levels: tuple[MultigridLevel, ...] = (),
) -> MultigridHierarchy:
    plan = _plan_galerkin_hierarchy(
        setup_operator,
        transfers,
        (*smoothers, coarse_solver),
        materialization,
        role_prefix="galerkin-coarse",
        previous_sparse_assemblies=previous_sparse_assemblies,
    )
    if not plan.accepted:
        raise LinearCapabilityError(plan.reason)
    return _prepare_levels_from_operators(
        plan.operators,
        transfers,
        smoothers,
        coarse_solver,
        materialization=materialization,
        supplied_properties=supplied_properties,
        pre_smoothing=pre_smoothing,
        post_smoothing=post_smoothing,
        builder_id=builder_id,
        construction_modes=plan.modes,
        reuse_decisions=reuse_decisions,
        setup_workspace_bytes=plan.construction_workspace_bytes,
        aggregate_assignments=(),
        sparse_assemblies=plan.sparse_assemblies,
        previous_levels=previous_levels,
    )


def _prepare_levels_from_operators(
    operators: tuple[AbstractLinearOperator, ...],
    transfers: tuple[tuple[AbstractLinearOperator, AbstractLinearOperator], ...],
    smoothers: tuple[PreconditionerSource, ...],
    coarse_solver: PreconditionerSource,
    /,
    *,
    materialization: MaterializationPolicy,
    supplied_properties: PreconditionerProperties | None,
    pre_smoothing: int,
    post_smoothing: int,
    builder_id: str,
    construction_modes: tuple[str, ...],
    reuse_decisions: tuple[str, ...],
    setup_workspace_bytes: int,
    aggregate_assignments: tuple[tuple[int, ...], ...],
    sparse_assemblies: tuple[PreparedSparseAssembly | None, ...],
    compatible_relaxation_factors: tuple[float, ...] = (),
    aggregate_candidate_ranks: tuple[tuple[int, ...], ...] = (),
    previous_levels: tuple[MultigridLevel, ...] = (),
) -> MultigridHierarchy:
    if len(operators) != len(transfers) + 1 or len(smoothers) != len(transfers):
        raise ValueError("Generated hierarchy source counts are inconsistent.")
    if len(sparse_assemblies) != len(transfers):
        raise ValueError("Sparse assemblies must align with Galerkin transitions.")
    if previous_levels and len(previous_levels) != len(operators):
        raise ValueError("Previous multigrid levels must align with refreshed operators.")
    prepared_levels: list[MultigridLevel] = []
    component_properties: list[PreconditionerProperties] = []
    action_decisions: list[str] = []
    preparation_workspace = int(setup_workspace_bytes)
    for index, source in enumerate((*smoothers, coarse_solver)):
        operator = operators[index]
        estimate = _source_cost(source, operator, materialization=materialization)
        if not estimate.accepted:
            raise LinearCapabilityError(
                f"Level {index} preconditioner rejected setup: {estimate.reason}"
            )
        preparation_workspace = max(
            preparation_workspace, estimate.preparation_workspace_bytes
        )
        if isinstance(source, AbstractPreconditioner):
            if not source.space.compatible(operator.source):
                raise ValueError(
                    f"Prepared level preconditioner space mismatch at level {index}."
                )
            prepared = source
            action_decisions.append(f"level-{index}:supplied-action-reused")
        else:
            if previous_levels and source.default_refresh == "numeric":
                prepared = source.refresh(
                    previous_levels[index].smoother,
                    operator,
                    materialization=materialization,
                )
                action_decisions.append(f"level-{index}:builder-action-refreshed")
            else:
                prepared = source.prepare(operator, materialization=materialization)
                action_decisions.append(f"level-{index}:builder-action-prepared")
        component_properties.append(prepared.properties)
        if index < len(transfers):
            restriction, prolongation = transfers[index]
            prepared_levels.append(
                MultigridLevel(
                    operator,
                    prepared,
                    restriction=restriction,
                    prolongation=prolongation,
                    pre_smoothing=pre_smoothing,
                    post_smoothing=post_smoothing,
                )
            )
        else:
            prepared_levels.append(MultigridLevel(operator, prepared))
    properties = _composition_properties(tuple(component_properties), supplied_properties)
    levels = tuple(prepared_levels)
    diagnostics = _setup_diagnostics(
        levels,
        construction_modes=construction_modes,
        reuse_decisions=(*reuse_decisions, *action_decisions),
        setup_workspace_bytes=preparation_workspace,
        aggregate_assignments=aggregate_assignments,
        reuse_dependency_fingerprint=builder_id,
        sparse_assemblies=sparse_assemblies,
        compatible_relaxation_factors=compatible_relaxation_factors,
        aggregate_candidate_ranks=aggregate_candidate_ranks,
    )
    return MultigridHierarchy(
        levels,
        properties=properties,
        hierarchy_id=canonical_fingerprint(
            {
                "kind": "generated-multigrid-hierarchy",
                "builder": builder_id,
                "fine_pattern": diagnostics.operator_pattern_fingerprints[0],
                "transfers": diagnostics.transfer_ids,
            }
        ),
        diagnostics=diagnostics,
        sparse_assemblies=sparse_assemblies,
    )


def _galerkin_step(
    operator: AbstractLinearOperator,
    restriction: AbstractLinearOperator,
    prolongation: AbstractLinearOperator,
    downstream: PreconditionerSource,
    policy: MaterializationPolicy,
    /,
    *,
    role: str,
    allow_matrix_free: bool = True,
    resident_operators: tuple[AbstractLinearOperator, ...] | None = None,
    previous_sparse_assembly: PreparedSparseAssembly | None = None,
) -> tuple[
    AbstractLinearOperator,
    str,
    int,
    PreparedSparseAssembly | None,
]:
    construction = (
        _construct_galerkin_operator(
            operator,
            restriction,
            prolongation,
            policy,
            role=role,
            resident_operators=(
                (operator, restriction, prolongation)
                if resident_operators is None
                else resident_operators
            ),
            allow_matrix_free=allow_matrix_free,
        )
        if previous_sparse_assembly is None
        else _refresh_galerkin_operator(
            operator,
            restriction,
            prolongation,
            previous_sparse_assembly,
        )
    )
    if construction.mode == "matrix-free-composition" and not allow_matrix_free:
        raise LinearCapabilityError(
            "Generated Galerkin setup requires an explicit coarse operator before "
            f"the next setup level: {construction.fallback_reason}"
        )
    coarse = construction.operator
    if isinstance(downstream, AbstractPreconditioner) and not (
        downstream.space.compatible(coarse.source)
    ):
        raise LinearCapabilityError(
            f"Generated Galerkin {construction.mode} route requires a downstream "
            "prepared action on the generated coarse space."
        )
    estimate = _source_cost(downstream, coarse, materialization=policy)
    if not estimate.accepted:
        fallback = (
            f" ({construction.fallback_reason})"
            if construction.fallback_reason is not None
            else ""
        )
        raise LinearCapabilityError(
            f"Generated Galerkin {construction.mode}{fallback} route was rejected "
            f"by the downstream preconditioner before level action preparation: "
            f"{estimate.reason}"
        )
    return (
        coarse,
        construction.mode,
        construction.workspace_bytes,
        construction.sparse_assembly,
    )


def _construct_galerkin_operator(
    operator: AbstractLinearOperator,
    restriction: AbstractLinearOperator,
    prolongation: AbstractLinearOperator,
    policy: MaterializationPolicy | None,
    /,
    *,
    role: str,
    resident_operators: tuple[AbstractLinearOperator, ...] = (),
    allow_matrix_free: bool = True,
) -> _GalerkinConstruction:
    composed = restriction @ operator @ prolongation
    operands = (operator, restriction, prolongation)
    sparse_rejection: str | None = None
    if _all_canonical_sparse(operands):
        try:
            sparse_plan = plan_sparse_assembly(
                composed,
                SparseAssemblyPolicy(materialization=policy),
            )
            sparse_assembly = prepare_sparse_assembly(sparse_plan, composed)
        except LinearCapabilityError as error:
            sparse_rejection = str(error)
        else:
            sparse_cost = sparse_plan.cost
            workspace = max(
                sparse_cost.output_bytes,
                sparse_cost.recipe_bytes,
                sparse_cost.symbolic_workspace_bytes,
                sparse_cost.numeric_workspace_bytes,
            )
            return _GalerkinConstruction(
                sparse_assembly.operator,
                "planned-sparse-assembly",
                workspace,
                None,
                sparse_assembly,
            )
    explicit = all(
        _is_explicit_operator(value)
        and (
            not isinstance(value, AbstractSparseLinearOperator)
            or _is_canonical_sparse(value)
        )
        for value in operands
    )
    if explicit:
        required_entries, required_bytes, workspace = _dense_galerkin_resources(
            operands,
            resident_operators,
        )
        dense_rejection = _dense_budget_rejection(
            required_entries,
            required_bytes,
            policy,
        )
        if dense_rejection is None:
            dense_matrices: dict[int, np.ndarray] = {}
            for value in operands:
                if id(value) not in dense_matrices:
                    dense_matrices[id(value)] = _explicit_dense_matrix(value, policy)
            dense_middle = dense_matrices[id(operator)]
            dense_left = dense_matrices[id(restriction)]
            dense_right = dense_matrices[id(prolongation)]
            intermediate = dense_middle @ dense_right
            coarse_matrix = dense_left @ intermediate
            return _GalerkinConstruction(
                DenseLinearOperator(
                    jnp.asarray(
                        coarse_matrix,
                        dtype=_coordinate_dtype(restriction.target),
                    ),
                    source=prolongation.source,
                    target=restriction.target,
                    operator_id=canonical_fingerprint(
                        {
                            "kind": role,
                            "mode": "bounded-dense-product",
                            "left": restriction.operator_id,
                            "middle_pattern": _operator_pattern_fingerprint(operator),
                            "right": prolongation.operator_id,
                        }
                    ),
                ),
                "bounded-dense-product",
                workspace,
                None,
                None,
            )
        fallback_reason = f"bounded-dense-product unavailable: {dense_rejection}"
        if sparse_rejection is not None:
            fallback_reason = (
                f"planned-sparse-assembly unavailable: {sparse_rejection}; "
                f"{fallback_reason}"
            )
        if not allow_matrix_free:
            fallback_reason += (
                "; matrix-free-composition unavailable because the next setup "
                "level requires an explicit coarse operator"
            )
        return _GalerkinConstruction(
            _MatrixFreeGalerkinOperator(composed, role=role),
            "matrix-free-composition",
            0,
            fallback_reason,
            None,
        )
    fallback_reason = None
    if not allow_matrix_free:
        fallback_reason = (
            "matrix-free-composition unavailable because the next setup level "
            "requires an explicit coarse operator"
        )
    return _GalerkinConstruction(
        _MatrixFreeGalerkinOperator(composed, role=role),
        "matrix-free-composition",
        0,
        fallback_reason,
        None,
    )


def _refresh_galerkin_operator(
    operator: AbstractLinearOperator,
    restriction: AbstractLinearOperator,
    prolongation: AbstractLinearOperator,
    previous: PreparedSparseAssembly,
    /,
) -> _GalerkinConstruction:
    composed = restriction @ operator @ prolongation
    refreshed = refresh_sparse_assembly(previous, composed)
    cost = refreshed.plan.cost
    return _GalerkinConstruction(
        refreshed.operator,
        "planned-sparse-assembly",
        max(
            cost.output_bytes,
            cost.recipe_bytes,
            cost.numeric_workspace_bytes,
        ),
        None,
        refreshed,
    )


def _explicit_host_matrix(operator: AbstractLinearOperator, /) -> sp.csr_matrix:
    if isinstance(operator, DenseLinearOperator):
        return sp.csr_matrix(np.asarray(operator.matrix))
    if isinstance(operator, AbstractSparseLinearOperator):
        return _canonical_csr(operator)
    raise LinearCapabilityError(
        "Host setup requires an explicit dense or canonical sparse operator."
    )


def _is_canonical_sparse(operator: AbstractSparseLinearOperator, /) -> bool:
    storage = operator.sparse_storage()
    return bool(storage.canonical and storage.sorted_indices)


def _all_canonical_sparse(
    operators: tuple[AbstractLinearOperator, ...],
    /,
) -> bool:
    return all(
        isinstance(operator, AbstractSparseLinearOperator)
        and _is_canonical_sparse(operator)
        for operator in operators
    )


def _resident_dense_arrays(
    operator: AbstractLinearOperator,
    /,
) -> tuple[Array, ...]:
    if isinstance(operator, DenseLinearOperator):
        return (operator.matrix,)
    if isinstance(operator, AdjointLinearOperator):
        return _resident_dense_arrays(operator.operator)
    return ()


def _dense_galerkin_resources(
    operands: tuple[
        AbstractLinearOperator,
        AbstractLinearOperator,
        AbstractLinearOperator,
    ],
    resident_operators: tuple[AbstractLinearOperator, ...],
    /,
) -> tuple[int, int, int]:
    operator, restriction, prolongation = operands
    resident_arrays = {
        id(array): array
        for resident in resident_operators
        for array in _resident_dense_arrays(resident)
    }
    entries = sum(int(array.size) for array in resident_arrays.values())
    required_bytes = sum(
        int(array.size * array.dtype.itemsize) for array in resident_arrays.values()
    )
    temporary_operands: dict[int, AbstractLinearOperator] = {}
    for operand in operands:
        if not isinstance(operand, DenseLinearOperator):
            temporary_operands.setdefault(id(operand), operand)
    for operand in temporary_operands.values():
        operand_entries = operand.target.size * operand.source.size
        entries += operand_entries
        required_bytes += operand_entries * _explicit_operator_dtype(operand).itemsize
    result_itemsize = np.result_type(
        *(_explicit_operator_dtype(value) for value in operands)
    ).itemsize
    intermediate_entries = operator.source.size * prolongation.source.size
    coarse_entries = restriction.target.size * prolongation.source.size
    entries += intermediate_entries + coarse_entries
    required_bytes += (intermediate_entries + coarse_entries) * result_itemsize
    output_bytes = coarse_entries * _coordinate_dtype(restriction.target).itemsize
    return int(entries), int(required_bytes), int(required_bytes + output_bytes)


def _dense_budget_rejection(
    entries: int,
    required_bytes: int,
    policy: MaterializationPolicy | None,
    /,
) -> str | None:
    if policy is None:
        return "cumulative dense setup requires an active materialization policy"
    if entries > policy.max_entries:
        return (
            f"cumulative dense setup requires {entries} entries, exceeding the "
            f"materialization limit {policy.max_entries}"
        )
    if required_bytes > policy.max_bytes:
        return (
            f"cumulative dense setup requires {required_bytes} bytes, exceeding "
            f"the materialization limit {policy.max_bytes}"
        )
    return None


def _nested_sparse_operators(
    operator: AbstractLinearOperator,
    /,
) -> tuple[AbstractSparseLinearOperator, ...]:
    if isinstance(operator, AbstractSparseLinearOperator):
        return (operator,)
    if isinstance(operator, AdjointLinearOperator):
        return _nested_sparse_operators(operator.operator)
    return ()


def _galerkin_stored_state_bytes(
    transfers: tuple[tuple[AbstractLinearOperator, AbstractLinearOperator], ...],
    coarse_operators: tuple[AbstractLinearOperator, ...],
    sparse_assemblies: tuple[PreparedSparseAssembly | None, ...] = (),
    /,
) -> int:
    explicit_coarse = tuple(
        operator
        for operator in coarse_operators
        if isinstance(
            operator,
            (DenseLinearOperator, AbstractSparseLinearOperator),
        )
    )
    artifacts = (transfers, explicit_coarse, sparse_assemblies)
    stored_bytes = _array_tree_storage_bytes(artifacts)
    operators = (
        *(value for pair in transfers for value in pair),
        *explicit_coarse,
    )
    sparse_operators = {
        id(sparse): sparse
        for operator in operators
        for sparse in _nested_sparse_operators(operator)
    }
    for sparse in sparse_operators.values():
        storage = sparse.sparse_storage()
        csr_bytes = sum(
            int(array.size * array.dtype.itemsize)
            for array in (storage.values, storage.indices, storage.indptr)
        )
        stored_bytes += max(
            0,
            csr_bytes - _array_tree_storage_bytes(sparse),
        )
    return int(stored_bytes)


def _is_explicit_operator(operator: AbstractLinearOperator, /) -> bool:
    if isinstance(operator, (DenseLinearOperator, AbstractSparseLinearOperator)):
        return True
    if isinstance(operator, AdjointLinearOperator):
        return _is_explicit_operator(operator.operator)
    return False


def _explicit_operator_dtype(operator: AbstractLinearOperator, /) -> np.dtype:
    if isinstance(operator, DenseLinearOperator):
        return np.dtype(operator.matrix.dtype)
    if isinstance(operator, AbstractSparseLinearOperator):
        return np.dtype(operator.sparse_storage().values.dtype)
    if isinstance(operator, AdjointLinearOperator):
        return _explicit_operator_dtype(operator.operator)
    raise LinearCapabilityError("An explicit operator dtype was required.")


def _explicit_dense_matrix(
    operator: AbstractLinearOperator,
    policy: MaterializationPolicy | None,
    /,
) -> np.ndarray:
    if policy is None:
        raise LinearCapabilityError(
            "Dense Galerkin construction requires an active materialization policy."
        )
    if isinstance(operator, DenseLinearOperator):
        return np.asarray(operator.matrix)
    if isinstance(operator, AbstractSparseLinearOperator):
        return _canonical_csr(operator).toarray()
    if isinstance(operator, AdjointLinearOperator) and _is_explicit_operator(
        operator.operator
    ):
        return np.asarray(materialize(operator, policy))
    raise LinearCapabilityError("An explicit operator matrix was required.")


def _canonical_csr(operator: AbstractSparseLinearOperator, /) -> sp.csr_matrix:
    storage = operator.sparse_storage()
    if not storage.canonical or not storage.sorted_indices:
        raise LinearCapabilityError("Sparse Galerkin products require canonical CSR.")
    return sp.csr_matrix(
        (
            np.asarray(storage.values),
            np.asarray(storage.indices),
            np.asarray(storage.indptr),
        ),
        shape=storage.shape,
    )


def _sparse_operator_from_csr(
    matrix: sp.csr_matrix,
    /,
    *,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    operator_id: str,
) -> AbstractSparseLinearOperator:
    from ..sparse import EdgeRelation, SparseCoordinateOperator

    canonical = matrix.tocsr(copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    canonical.data = canonical.data.astype(_coordinate_dtype(target), copy=False)
    identifier = canonical_fingerprint(
        {
            "declared": str(operator_id),
            "shape": list(canonical.shape),
            "indices": array_tree_fingerprint(canonical.indices),
            "indptr": array_tree_fingerprint(canonical.indptr),
        }
    )
    rows = np.repeat(
        np.arange(canonical.shape[0], dtype=np.int64), np.diff(canonical.indptr)
    )
    relation = EdgeRelation(
        canonical.indices,
        rows,
        source_size=canonical.shape[1],
        target_size=canonical.shape[0],
    )
    return SparseCoordinateOperator(
        relation,
        canonical.data,
        source=source,
        target=target,
        properties=OperatorProperties(),
        operator_id=identifier,
    )


def _initial_candidates(
    space: AbstractVectorSpace,
    near_nullspaces: tuple[LinearSubspace, ...],
    /,
) -> np.ndarray:
    if near_nullspaces:
        blocks: list[np.ndarray] = []
        for candidate in near_nullspaces:
            dimension = int(np.asarray(candidate.dimension))
            if dimension <= 0:
                raise ValueError(
                    "Every near-nullspace block must contain an active candidate."
                )
            blocks.append(np.asarray(candidate.basis[:, :dimension]))
        return np.concatenate(blocks, axis=1)
    if not isinstance(space, ArraySpace):
        raise LinearCapabilityError(
            "A semantic vector space requires explicit near-nullspace candidates."
        )
    candidate = jnp.ones((space.size,), dtype=_coordinate_dtype(space))
    vector = space.unflatten(candidate)
    norm = float(np.asarray(jnp.sqrt(jnp.real(space.inner(vector, vector)))))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("The default constant candidate has invalid pairing norm.")
    return np.asarray(candidate / norm)[:, None]


def _strength_graph(matrix: sp.csr_matrix, threshold: float, /) -> sp.csr_matrix:
    canonical = matrix.tocsr(copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    diagonal = np.abs(canonical.diagonal())
    coo = canonical.tocoo()
    off_diagonal = coo.row != coo.col
    rows = coo.row[off_diagonal]
    columns = coo.col[off_diagonal]
    magnitudes = np.abs(coo.data[off_diagonal])
    scales = np.sqrt(diagonal[rows] * diagonal[columns])
    strong = magnitudes >= threshold * scales
    strong &= magnitudes > 0.0
    adjacency = sp.coo_matrix(
        (
            np.ones(int(np.count_nonzero(strong)), dtype=np.int8),
            (rows[strong], columns[strong]),
        ),
        shape=canonical.shape,
    ).tocsr()
    adjacency = ((adjacency + adjacency.transpose()) != 0).astype(np.int8).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def _deterministic_aggregates(strength: sp.csr_matrix, /) -> np.ndarray:
    dimension = strength.shape[0]
    aggregate = np.full((dimension,), -1, dtype=np.int64)
    aggregate_index = 0
    for seed in range(dimension):
        if aggregate[seed] >= 0:
            continue
        aggregate[seed] = aggregate_index
        neighbors = strength.indices[strength.indptr[seed] : strength.indptr[seed + 1]]
        for neighbor in neighbors:
            if aggregate[neighbor] < 0:
                aggregate[neighbor] = aggregate_index
        aggregate_index += 1
    return aggregate


def _validate_aggregate_assignment(
    aggregate: np.ndarray,
    dimension: int,
    /,
) -> None:
    if aggregate.shape != (dimension,) or np.any(aggregate < 0):
        raise ValueError("Reused aggregate assignment does not match the level space.")
    labels = np.unique(aggregate)
    if not np.array_equal(labels, np.arange(labels.size)):
        raise ValueError("Reused aggregate labels must be contiguous from zero.")


def _tentative_prolongator(
    aggregate: np.ndarray,
    candidates: np.ndarray,
    rank_tolerance: float | None,
    /,
) -> tuple[sp.csr_matrix, np.ndarray, tuple[int, ...]]:
    if candidates.ndim != 2 or candidates.shape[0] != aggregate.size:
        raise ValueError("Candidate coordinates must align with aggregate nodes.")
    number_aggregates = int(aggregate.max()) + 1
    local_data: list[tuple[np.ndarray, np.ndarray, int]] = []
    ranks: list[int] = []
    coarse_size = 0
    for aggregate_index in range(number_aggregates):
        nodes = np.flatnonzero(aggregate == aggregate_index)
        local = candidates[nodes, :]
        q, r, _ = spla.qr(local, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(r))
        if diagonal.size:
            relative = (
                np.finfo(local.real.dtype).eps * max(local.shape)
                if rank_tolerance is None
                else rank_tolerance
            )
            cutoff = relative * diagonal[0]
            rank = int(np.count_nonzero(diagonal > cutoff))
        else:
            rank = 0
        q_active = q[:, :rank]
        local_data.append((nodes, q_active, coarse_size))
        ranks.append(rank)
        coarse_size += rank
    if coarse_size <= 0:
        raise LinearCapabilityError(
            "Rank-revealing aggregation eliminated every candidate."
        )
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    coarse_candidates = np.zeros(
        (coarse_size, candidates.shape[1]), dtype=candidates.dtype
    )
    for nodes, q_active, offset in local_data:
        rank = q_active.shape[1]
        if rank == 0:
            continue
        row_parts.append(np.repeat(nodes, rank))
        column_parts.append(
            np.tile(np.arange(offset, offset + rank, dtype=np.int64), nodes.size)
        )
        value_parts.append(q_active.reshape((-1,)))
        coarse_candidates[offset : offset + rank, :] = (
            q_active.conj().T @ candidates[nodes, :]
        )
    tentative = sp.coo_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(aggregate.size, coarse_size),
    ).tocsr()
    tentative.sum_duplicates()
    tentative.sort_indices()
    return tentative, coarse_candidates, tuple(ranks)


def _smooth_prolongator(
    matrix: sp.csr_matrix,
    tentative: sp.csr_matrix,
    /,
    *,
    steps: int,
    damping: float,
) -> sp.csr_matrix:
    prolongator = tentative.tocsr(copy=True)
    if steps == 0:
        return prolongator
    diagonal = matrix.diagonal()
    if np.any(~np.isfinite(diagonal)) or np.any(np.abs(diagonal) == 0.0):
        raise LinearCapabilityError(
            "Smoothed aggregation requires finite nonzero diagonal entries."
        )
    inverse_diagonal = sp.diags(1.0 / diagonal, format="csr")
    for _ in range(steps):
        prolongator = (
            prolongator - damping * (inverse_diagonal @ (matrix @ prolongator))
        ).tocsr()
        prolongator.sum_duplicates()
        prolongator.eliminate_zeros()
        prolongator.sort_indices()
    return prolongator


def _pairing_aware_transfers(
    prolongator: sp.csr_matrix,
    coarse_space: ArraySpace,
    fine_space: AbstractVectorSpace,
    policy: MaterializationPolicy,
    /,
    *,
    transfer_id_payload: dict[str, object],
) -> tuple[AbstractLinearOperator, AbstractLinearOperator, int]:
    prolongator_id = canonical_fingerprint(
        {"kind": "sa-prolongation", **transfer_id_payload}
    )
    restriction_id = canonical_fingerprint(
        {"kind": "sa-restriction-adjoint", **transfer_id_payload}
    )
    if isinstance(fine_space, ArraySpace) and isinstance(
        fine_space.pairing, (EuclideanPairing, DiagonalPairing)
    ):
        restriction_matrix = prolongator.conjugate().transpose().tocsr()
        if isinstance(fine_space.pairing, DiagonalPairing):
            weights = np.asarray(fine_space.pairing.weights).reshape((-1,))
            restriction_matrix = (
                restriction_matrix @ sp.diags(weights, format="csr")
            ).tocsr()
        prolongation = _sparse_operator_from_csr(
            prolongator,
            source=coarse_space,
            target=fine_space,
            operator_id=prolongator_id,
        )
        restriction = _sparse_operator_from_csr(
            restriction_matrix,
            source=fine_space,
            target=coarse_space,
            operator_id=restriction_id,
        )
        return (
            restriction,
            prolongation,
            _csr_bytes(prolongator) + _csr_bytes(restriction_matrix),
        )
    entries = 2 * prolongator.shape[0] * prolongator.shape[1]
    _check_dense_budget(entries, prolongator.dtype.itemsize, policy)
    dense = prolongator.toarray()
    prolongation = DenseLinearOperator(
        jnp.asarray(dense),
        source=coarse_space,
        target=fine_space,
        operator_id=prolongator_id,
    )
    adjoint = AdjointLinearOperator(prolongation)
    restriction_matrix = materialize(adjoint, policy)
    restriction = DenseLinearOperator(
        restriction_matrix,
        source=fine_space,
        target=coarse_space,
        operator_id=restriction_id,
    )
    return restriction, prolongation, int(entries * dense.dtype.itemsize)


def _setup_diagnostics(
    levels: tuple[MultigridLevel, ...],
    /,
    *,
    construction_modes: tuple[str, ...],
    reuse_decisions: tuple[str, ...],
    setup_workspace_bytes: int,
    aggregate_assignments: tuple[tuple[int, ...], ...],
    reuse_dependency_fingerprint: str,
    sparse_assemblies: tuple[PreparedSparseAssembly | None, ...],
    compatible_relaxation_factors: tuple[float, ...] = (),
    aggregate_candidate_ranks: tuple[tuple[int, ...], ...] = (),
) -> MultigridSetupDiagnostics:
    dimensions = tuple(level.operator.source.size for level in levels)
    nonzeros = tuple(_operator_nnz(level.operator) for level in levels)
    operator_complexity = (
        sum(value for value in nonzeros if value is not None) / nonzeros[0]
        if all(value is not None for value in nonzeros)
        and nonzeros[0] is not None
        and nonzeros[0] > 0
        else None
    )
    transfer_ids = tuple(
        (level.restriction.operator_id, level.prolongation.operator_id)
        for level in levels[:-1]
        if level.restriction is not None and level.prolongation is not None
    )
    prepared_bytes = _array_tree_storage_bytes((levels, sparse_assemblies)) + 8 * sum(
        len(assignment) for assignment in aggregate_assignments
    )
    return MultigridSetupDiagnostics(
        level_dimensions=dimensions,
        level_nnz=nonzeros,
        grid_complexity=(sum(dimensions) / dimensions[0] if dimensions[0] else 0.0),
        operator_complexity=operator_complexity,
        prepared_state_bytes=prepared_bytes,
        setup_workspace_bytes=setup_workspace_bytes,
        transfer_ids=transfer_ids,
        coarse_construction_modes=construction_modes,
        reuse_decisions=reuse_decisions,
        operator_pattern_fingerprints=tuple(
            _operator_pattern_fingerprint(level.operator) for level in levels
        ),
        aggregate_assignments=aggregate_assignments,
        level_storage_bytes=tuple(
            _array_tree_storage_bytes(level.operator) for level in levels
        ),
        compatible_relaxation_factors=compatible_relaxation_factors,
        aggregate_candidate_ranks=aggregate_candidate_ranks,
        reuse_dependency_fingerprint=reuse_dependency_fingerprint,
    )


def _operator_nnz(operator: AbstractLinearOperator, /) -> int | None:
    if isinstance(operator, AbstractSparseLinearOperator):
        return int(operator.sparse_storage().values.size)
    if isinstance(operator, DenseLinearOperator):
        return int(np.count_nonzero(np.asarray(operator.matrix)))
    return None


def _operator_storage_bytes(operator: AbstractLinearOperator, /) -> int:
    if isinstance(operator, AbstractSparseLinearOperator):
        storage = operator.sparse_storage()
        return int(storage.values.nbytes + storage.indices.nbytes + storage.indptr.nbytes)
    if isinstance(operator, DenseLinearOperator):
        return int(operator.matrix.nbytes)
    return _array_tree_storage_bytes(operator)


def _hierarchy_limit_rejection(
    policy: SmoothedAggregationPolicy,
    operators: list[AbstractLinearOperator],
    /,
) -> str | None:
    dimensions = tuple(operator.source.size for operator in operators)
    grid_complexity = sum(dimensions) / dimensions[0]
    if (
        policy.maximum_grid_complexity is not None
        and grid_complexity > policy.maximum_grid_complexity
    ):
        return (
            f"Grid complexity {grid_complexity:.6g} exceeds limit "
            f"{policy.maximum_grid_complexity:.6g}."
        )
    nonzeros = tuple(_operator_nnz(operator) for operator in operators)
    if all(value is not None for value in nonzeros) and nonzeros[0]:
        operator_complexity = sum(int(value) for value in nonzeros) / int(nonzeros[0])
        if (
            policy.maximum_operator_complexity is not None
            and operator_complexity > policy.maximum_operator_complexity
        ):
            return (
                f"Operator complexity {operator_complexity:.6g} exceeds limit "
                f"{policy.maximum_operator_complexity:.6g}."
            )
    if policy.maximum_level_storage_bytes is not None:
        for index, operator in enumerate(operators):
            storage = _operator_storage_bytes(operator)
            if storage > policy.maximum_level_storage_bytes:
                return (
                    f"Level {index} storage {storage} bytes exceeds limit "
                    f"{policy.maximum_level_storage_bytes}."
                )
    return None


def _compatible_relaxation_factor(
    matrix: sp.csr_matrix,
    tentative: sp.csr_matrix,
    /,
    *,
    damping: float,
) -> float:
    dimension = matrix.shape[0]
    indices = np.arange(dimension, dtype=float)
    probe = np.sin((indices + 1.0) * np.sqrt(2.0)) + np.cos(
        (indices + 1.0) * np.sqrt(3.0)
    )
    coefficients = tentative.conjugate().transpose() @ probe
    error = probe - tentative @ coefficients
    norm = float(np.linalg.norm(error))
    if norm == 0.0:
        return 0.0
    diagonal = matrix.diagonal()
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal == 0):
        return float("inf")
    relaxed = error - damping * (matrix @ error) / diagonal
    return float(np.linalg.norm(relaxed) / norm)


def _operator_pattern_fingerprint(
    operator: AbstractLinearOperator,
    /,
) -> str:
    if isinstance(operator, AbstractSparseLinearOperator):
        storage = operator.sparse_storage()
        return canonical_fingerprint(
            {
                "kind": "canonical-csr-pattern",
                "shape": list(storage.shape),
                "indices": array_tree_fingerprint(storage.indices),
                "indptr": array_tree_fingerprint(storage.indptr),
                "source": operator.source.space_id,
                "target": operator.target.space_id,
            }
        )
    if isinstance(operator, DenseLinearOperator):
        return canonical_fingerprint(
            {
                "kind": "dense-storage-pattern",
                "shape": list(operator.matrix.shape),
                "dtype": np.dtype(operator.matrix.dtype).str,
                "source": operator.source.space_id,
                "target": operator.target.space_id,
            }
        )
    return canonical_fingerprint(
        {
            "kind": "matrix-free-composition-pattern",
            "operator": operator.operator_id,
            "source": operator.source.space_id,
            "target": operator.target.space_id,
        }
    )


def _check_dense_budget(
    entries: int,
    itemsize: int,
    policy: MaterializationPolicy,
    /,
) -> None:
    if entries > policy.max_entries:
        raise LinearCapabilityError(
            f"Dense Galerkin setup requires {entries} entries, exceeding the "
            f"materialization limit {policy.max_entries}."
        )
    required_bytes = int(entries) * int(itemsize)
    if required_bytes > policy.max_bytes:
        raise LinearCapabilityError(
            f"Dense Galerkin setup requires {required_bytes} bytes, exceeding the "
            f"materialization limit {policy.max_bytes}."
        )


def _csr_bytes(matrix: sp.csr_matrix, /) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


__all__ = [
    "GalerkinHierarchyBuilder",
    "MultigridRefreshMode",
    "SmoothedAggregationHierarchyBuilder",
    "SmoothedAggregationPolicy",
]
