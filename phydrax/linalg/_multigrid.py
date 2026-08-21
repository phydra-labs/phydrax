#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._assembly import PreparedSparseAssembly
from ._costs import _array_tree_storage_bytes, PreconditionerCostEstimate
from ._materialization import MaterializationPolicy
from ._operators import AbstractLinearOperator, DenseLinearOperator
from ._preconditioner_properties import (
    _preconditioner_properties_payload,
    PreconditionerProperties,
)
from ._preconditioners import AbstractPreconditioner
from ._preconditioning import (
    _source_cost,
    AbstractPreconditionerBuilder,
    DenseInversePreconditionerBuilder,
    JacobiPreconditionerBuilder,
    PreconditionerSource,
)
from ._properties import LinearCapabilityError, OperatorProperties
from ._spaces import _coordinate_dtype
from ._sparse_contract import AbstractSparseLinearOperator


def _composition_properties(
    components: tuple[PreconditionerProperties, ...],
    supplied: PreconditionerProperties | None,
    /,
) -> PreconditionerProperties:
    if not all(isinstance(value, PreconditionerProperties) for value in components):
        raise TypeError("Every level source must expose PreconditionerProperties.")
    linear = all(value.certifies("linear") for value in components)
    stationary = all(value.certifies("stationary") for value in components)
    if supplied is None:
        claims = {"linear": linear, "stationary": stationary}
        return PreconditionerProperties(
            **claims,
            evidence={name: "transformed" for name, claimed in claims.items() if claimed},
        )
    if not isinstance(supplied, PreconditionerProperties):
        raise TypeError("properties must be PreconditionerProperties.")
    requires_linear = any(
        supplied.certifies(name)
        for name in ("linear", "self_adjoint", "positive_definite")
    )
    requires_stationary = any(
        supplied.certifies(name)
        for name in ("stationary", "self_adjoint", "positive_definite")
    )
    if (requires_linear and not linear) or (requires_stationary and not stationary):
        raise ValueError(
            "Certified hierarchy properties require every level source to certify "
            "the corresponding linear and stationary contracts."
        )
    return supplied


def _source_properties(
    source: PreconditionerSource,
    operator: AbstractLinearOperator,
    /,
) -> PreconditionerProperties:
    if isinstance(source, AbstractPreconditioner):
        return source.properties
    return source.properties_for(operator)


MultigridCycleKind: TypeAlias = Literal["v", "w", "f", "full"]


class MultigridCyclePolicy(StrictModule):
    """Static recursive-cycle semantics for one prepared hierarchy action."""

    kind: MultigridCycleKind = eqx.field(static=True)
    cycle_id: str = eqx.field(static=True)

    def __init__(self, kind: MultigridCycleKind = "v", /):
        if kind not in ("v", "w", "f", "full"):
            raise ValueError(f"Unknown multigrid cycle kind {kind!r}.")
        self.kind = kind
        self.cycle_id = canonical_fingerprint(
            {"kind": "multigrid-cycle-policy", "cycle": kind}
        )


class MultigridLevel(StrictModule):
    """Prepared operator, transfer pair, smoother, and schedule for one level."""

    operator: AbstractLinearOperator
    smoother: AbstractPreconditioner
    restriction: AbstractLinearOperator | None
    prolongation: AbstractLinearOperator | None
    pre_smoothing: int = eqx.field(static=True)
    post_smoothing: int = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        smoother: AbstractPreconditioner,
        /,
        *,
        restriction: AbstractLinearOperator | None = None,
        prolongation: AbstractLinearOperator | None = None,
        pre_smoothing: int = 1,
        post_smoothing: int = 1,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape or not operator.source.compatible(operator.target):
            raise ValueError(
                "A multigrid level operator must be an unbatched endomorphism."
            )
        if not isinstance(smoother, AbstractPreconditioner):
            raise TypeError("smoother must be an AbstractPreconditioner.")
        if not smoother.space.compatible(operator.source):
            raise ValueError("The smoother must act on the level space.")
        if (restriction is None) != (prolongation is None):
            raise ValueError("restriction and prolongation must be supplied together.")
        if restriction is not None:
            if not isinstance(restriction, AbstractLinearOperator) or not isinstance(
                prolongation, AbstractLinearOperator
            ):
                raise TypeError("restriction and prolongation must be linear operators.")
            if restriction.batch_shape or prolongation.batch_shape:
                raise ValueError("Multigrid transfer operators must be unbatched.")
        pre = int(pre_smoothing)
        post = int(post_smoothing)
        if pre < 0 or post < 0:
            raise ValueError("Smoothing counts must be non-negative.")
        self.operator = operator
        self.smoother = smoother
        self.restriction = restriction
        self.prolongation = prolongation
        self.pre_smoothing = pre
        self.post_smoothing = post


class MultigridSetupDiagnostics(StrictModule):
    """Static setup accounting and dependency decisions for one hierarchy."""

    level_dimensions: tuple[int, ...] = eqx.field(static=True)
    level_nnz: tuple[int | None, ...] = eqx.field(static=True)
    grid_complexity: float = eqx.field(static=True)
    operator_complexity: float | None = eqx.field(static=True)
    prepared_state_bytes: int = eqx.field(static=True)
    setup_workspace_bytes: int = eqx.field(static=True)
    transfer_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    coarse_construction_modes: tuple[str, ...] = eqx.field(static=True)
    reuse_decisions: tuple[str, ...] = eqx.field(static=True)
    operator_pattern_fingerprints: tuple[str, ...] = eqx.field(static=True)
    aggregate_assignments: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    level_storage_bytes: tuple[int, ...] = eqx.field(static=True)
    compatible_relaxation_factors: tuple[float, ...] = eqx.field(static=True)
    aggregate_candidate_ranks: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    reuse_dependency_fingerprint: str | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        level_dimensions: tuple[int, ...],
        level_nnz: tuple[int | None, ...],
        grid_complexity: float,
        operator_complexity: float | None,
        prepared_state_bytes: int,
        setup_workspace_bytes: int,
        transfer_ids: tuple[tuple[str, str], ...],
        coarse_construction_modes: tuple[str, ...],
        reuse_decisions: tuple[str, ...],
        operator_pattern_fingerprints: tuple[str, ...] = (),
        aggregate_assignments: tuple[tuple[int, ...], ...] = (),
        reuse_dependency_fingerprint: str | None = None,
        level_storage_bytes: tuple[int, ...] = (),
        compatible_relaxation_factors: tuple[float, ...] = (),
        aggregate_candidate_ranks: tuple[tuple[int, ...], ...] = (),
    ):
        dimensions = tuple(int(value) for value in level_dimensions)
        nonzeros = tuple(None if value is None else int(value) for value in level_nnz)
        if not dimensions or any(value < 0 for value in dimensions):
            raise ValueError("Multigrid level dimensions must be nonnegative.")
        if len(nonzeros) != len(dimensions) or any(
            value is not None and value < 0 for value in nonzeros
        ):
            raise ValueError("level_nnz must align with the nonnegative level counts.")
        grid = float(grid_complexity)
        operator = None if operator_complexity is None else float(operator_complexity)
        if not isfinite(grid) or grid < 0.0:
            raise ValueError("grid_complexity must be finite and nonnegative.")
        if operator is not None and (not isfinite(operator) or operator < 0.0):
            raise ValueError(
                "operator_complexity must be finite and nonnegative when known."
            )
        state_bytes = int(prepared_state_bytes)
        workspace_bytes = int(setup_workspace_bytes)
        if state_bytes < 0 or workspace_bytes < 0:
            raise ValueError("Multigrid setup byte counts must be nonnegative.")
        transfers = tuple(
            (str(restriction), str(prolongation))
            for restriction, prolongation in transfer_ids
        )
        if len(transfers) != len(dimensions) - 1 or any(
            not restriction or not prolongation for restriction, prolongation in transfers
        ):
            raise ValueError("transfer_ids must identify every hierarchy transition.")
        modes = tuple(str(value) for value in coarse_construction_modes)
        decisions = tuple(str(value) for value in reuse_decisions)
        if len(modes) != len(transfers) or any(not value for value in modes):
            raise ValueError(
                "coarse_construction_modes must identify every hierarchy transition."
            )
        if any(not value for value in decisions):
            raise ValueError("reuse_decisions entries must be non-empty.")
        fingerprints = tuple(str(value) for value in operator_pattern_fingerprints)
        if fingerprints and (
            len(fingerprints) != len(dimensions)
            or any(not value for value in fingerprints)
        ):
            raise ValueError(
                "operator_pattern_fingerprints must identify every hierarchy level."
            )
        assignments = tuple(
            tuple(int(index) for index in level) for level in aggregate_assignments
        )
        if assignments and len(assignments) != len(transfers):
            raise ValueError(
                "aggregate_assignments must identify every aggregated transition."
            )
        dependency_fingerprint = (
            None
            if reuse_dependency_fingerprint is None
            else str(reuse_dependency_fingerprint)
        )
        if dependency_fingerprint == "":
            raise ValueError("reuse_dependency_fingerprint must be non-empty.")
        self.level_dimensions = dimensions
        level_bytes = tuple(int(value) for value in level_storage_bytes)
        if level_bytes and (
            len(level_bytes) != len(dimensions) or any(value < 0 for value in level_bytes)
        ):
            raise ValueError(
                "level_storage_bytes must align with nonnegative hierarchy levels."
            )
        relaxation_factors = tuple(
            float(value) for value in compatible_relaxation_factors
        )
        if relaxation_factors and (
            len(relaxation_factors) != len(transfers)
            or any(not isfinite(value) or value < 0.0 for value in relaxation_factors)
        ):
            raise ValueError(
                "compatible_relaxation_factors must align with hierarchy transitions."
            )
        candidate_ranks = tuple(
            tuple(int(rank) for rank in ranks) for ranks in aggregate_candidate_ranks
        )
        if candidate_ranks and (
            len(candidate_ranks) != len(transfers)
            or any(rank <= 0 for ranks in candidate_ranks for rank in ranks)
        ):
            raise ValueError(
                "aggregate_candidate_ranks must contain positive ranks per transition."
            )
        self.level_nnz = nonzeros
        self.grid_complexity = grid
        self.operator_complexity = operator
        self.prepared_state_bytes = state_bytes
        self.setup_workspace_bytes = workspace_bytes
        self.transfer_ids = transfers
        self.coarse_construction_modes = modes
        self.reuse_decisions = decisions
        self.operator_pattern_fingerprints = fingerprints
        self.aggregate_assignments = assignments
        self.reuse_dependency_fingerprint = dependency_fingerprint
        self.level_storage_bytes = level_bytes
        self.compatible_relaxation_factors = relaxation_factors
        self.aggregate_candidate_ranks = candidate_ranks


def _default_setup_diagnostics(
    levels: tuple[MultigridLevel, ...],
    sparse_assemblies: tuple[PreparedSparseAssembly | None, ...] = (),
    /,
    *,
    setup_workspace_bytes: int = 0,
    reuse_dependency_fingerprint: str | None = None,
) -> MultigridSetupDiagnostics:
    dimensions = tuple(level.operator.source.size for level in levels)
    nonzeros = tuple(
        (
            int(level.operator.sparse_storage().values.size)
            if isinstance(level.operator, AbstractSparseLinearOperator)
            else (
                int(jnp.count_nonzero(level.operator.matrix))
                if isinstance(level.operator, DenseLinearOperator)
                else None
            )
        )
        for level in levels
    )
    known_complexity = (
        all(value is not None for value in nonzeros)
        and nonzeros[0] is not None
        and nonzeros[0] > 0
    )
    transfer_ids = tuple(
        (level.restriction.operator_id, level.prolongation.operator_id)
        for level in levels[:-1]
        if level.restriction is not None and level.prolongation is not None
    )
    prepared_bytes = _array_tree_storage_bytes((levels, sparse_assemblies))
    return MultigridSetupDiagnostics(
        level_dimensions=dimensions,
        level_nnz=nonzeros,
        grid_complexity=(sum(dimensions) / dimensions[0] if dimensions[0] else 0.0),
        operator_complexity=(
            sum(value for value in nonzeros if value is not None) / nonzeros[0]
            if known_complexity
            else None
        ),
        prepared_state_bytes=prepared_bytes,
        setup_workspace_bytes=setup_workspace_bytes,
        transfer_ids=transfer_ids,
        coarse_construction_modes=("supplied",) * (len(levels) - 1),
        reuse_decisions=("prepared-explicit",),
        reuse_dependency_fingerprint=reuse_dependency_fingerprint,
        level_storage_bytes=tuple(
            _array_tree_storage_bytes(level.operator) for level in levels
        ),
    )


class MultigridHierarchy(StrictModule):
    """Immutable prepared hierarchy whose transfers carry explicit level spaces."""

    levels: tuple[MultigridLevel, ...]
    properties: PreconditionerProperties
    diagnostics: MultigridSetupDiagnostics
    sparse_assemblies: tuple[PreparedSparseAssembly | None, ...]
    hierarchy_id: str = eqx.field(static=True)

    def __init__(
        self,
        levels: tuple[MultigridLevel, ...],
        /,
        *,
        properties: PreconditionerProperties | None = None,
        hierarchy_id: str | None = None,
        diagnostics: MultigridSetupDiagnostics | None = None,
        sparse_assemblies: tuple[PreparedSparseAssembly | None, ...] | None = None,
    ):
        levels_ = tuple(levels)
        if len(levels_) < 2:
            raise ValueError("A multigrid hierarchy requires at least two levels.")
        if not all(isinstance(level, MultigridLevel) for level in levels_):
            raise TypeError("levels must contain MultigridLevel values.")
        for index, level in enumerate(levels_[:-1]):
            if level.restriction is None or level.prolongation is None:
                raise ValueError(
                    "Every non-coarse level requires both transfer operators."
                )
            coarse_space = levels_[index + 1].operator.source
            if not level.restriction.source.compatible(
                level.operator.source
            ) or not level.restriction.target.compatible(coarse_space):
                raise ValueError(f"Restriction space mismatch at transition {index}.")
            if not level.prolongation.source.compatible(
                coarse_space
            ) or not level.prolongation.target.compatible(level.operator.source):
                raise ValueError(f"Prolongation space mismatch at transition {index}.")
        coarse = levels_[-1]
        if coarse.restriction is not None or coarse.prolongation is not None:
            raise ValueError("The coarsest level cannot carry transfer operators.")
        properties_ = _composition_properties(
            tuple(level.smoother.properties for level in levels_),
            properties,
        )
        assemblies_ = (
            (None,) * (len(levels_) - 1)
            if sparse_assemblies is None
            else tuple(sparse_assemblies)
        )
        if len(assemblies_) != len(levels_) - 1 or any(
            value is not None and not isinstance(value, PreparedSparseAssembly)
            for value in assemblies_
        ):
            raise TypeError(
                "sparse_assemblies must align with the hierarchy transitions."
            )
        for index, assembly in enumerate(assemblies_):
            if assembly is None:
                continue
            coarse = levels_[index + 1].operator
            if (
                not assembly.operator.source.compatible(coarse.source)
                or not assembly.operator.target.compatible(coarse.target)
                or assembly.operator.operator_id != coarse.operator_id
            ):
                raise ValueError(
                    f"Sparse assembly mismatch at hierarchy transition {index}."
                )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "multigrid-hierarchy",
                    "levels": [
                        {
                            "operator": level.operator.operator_id,
                            "smoother": level.smoother.preconditioner_id,
                            "restriction": (
                                None
                                if level.restriction is None
                                else level.restriction.operator_id
                            ),
                            "prolongation": (
                                None
                                if level.prolongation is None
                                else level.prolongation.operator_id
                            ),
                            "pre": level.pre_smoothing,
                            "post": level.post_smoothing,
                        }
                        for level in levels_
                    ],
                    "properties": _preconditioner_properties_payload(properties_),
                    "sparse_assembly_plans": [
                        None if value is None else value.plan.plan_id
                        for value in assemblies_
                    ],
                }
            )
            if hierarchy_id is None
            else str(hierarchy_id)
        )
        if not identifier:
            raise ValueError("hierarchy_id must be non-empty.")
        self.levels = levels_
        self.properties = properties_
        self.sparse_assemblies = assemblies_
        diagnostics_ = (
            _default_setup_diagnostics(levels_, assemblies_)
            if diagnostics is None
            else diagnostics
        )
        if not isinstance(diagnostics_, MultigridSetupDiagnostics):
            raise TypeError("diagnostics must be MultigridSetupDiagnostics.")
        if diagnostics_.level_dimensions != tuple(
            level.operator.source.size for level in levels_
        ):
            raise ValueError("Diagnostics level dimensions must match the hierarchy.")
        expected_transfer_ids = tuple(
            (level.restriction.operator_id, level.prolongation.operator_id)
            for level in levels_[:-1]
            if level.restriction is not None and level.prolongation is not None
        )
        if diagnostics_.transfer_ids != expected_transfer_ids:
            raise ValueError("Diagnostics transfer IDs must match the hierarchy.")
        self.diagnostics = diagnostics_
        self.hierarchy_id = identifier


class MultigridPreconditioner(AbstractPreconditioner):
    """One policy-selected cycle over an immutable prepared hierarchy."""

    hierarchy: MultigridHierarchy
    cycle_policy: MultigridCyclePolicy

    def __init__(
        self,
        hierarchy: MultigridHierarchy,
        /,
        *,
        cycle_policy: MultigridCyclePolicy | None = None,
    ):
        if not isinstance(hierarchy, MultigridHierarchy):
            raise TypeError("hierarchy must be a MultigridHierarchy.")
        policy = MultigridCyclePolicy() if cycle_policy is None else cycle_policy
        if not isinstance(policy, MultigridCyclePolicy):
            raise TypeError("cycle_policy must be a MultigridCyclePolicy.")
        self.hierarchy = hierarchy
        self.cycle_policy = policy
        self.space = hierarchy.levels[0].operator.source
        self.properties = hierarchy.properties
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "multigrid-preconditioner",
                "hierarchy": hierarchy.hierarchy_id,
                "cycle": policy.cycle_id,
            }
        )

    def apply(
        self,
        residual: PyTree[Any],
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[Array]:
        residual_ = self.space.validate(residual)
        if self.cycle_policy.kind == "full":
            return self._full_cycle(0, residual_, iteration=iteration)
        return self._cycle(
            0,
            residual_,
            iteration=iteration,
            cycle_kind=self.cycle_policy.kind,
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        if not self.space.compatible(setup_operator.source):
            raise ValueError("Multigrid action and setup operator spaces must match.")
        apply_workspace = 0
        for level in self.hierarchy.levels:
            estimate = _source_cost(
                level.smoother,
                level.operator,
                materialization=materialization,
            )
            apply_workspace += estimate.apply_workspace_bytes_per_rhs
            apply_workspace += (
                4
                * level.operator.source.size
                * _coordinate_dtype(level.operator.source).itemsize
            )
        return PreconditionerCostEstimate(
            component=self.preconditioner_id,
            storage_bytes=self.hierarchy.diagnostics.prepared_state_bytes,
            apply_workspace_bytes_per_rhs=apply_workspace,
            reason="supplied prepared multigrid hierarchy state",
        )

    def _cycle(
        self,
        level_index: int,
        residual: PyTree[Array],
        /,
        *,
        iteration: ArrayLike | None,
        cycle_kind: Literal["v", "w", "f"],
    ) -> PyTree[Array]:
        level = self.hierarchy.levels[level_index]
        if level_index == len(self.hierarchy.levels) - 1:
            return level.smoother.apply(residual, iteration=iteration)
        estimate = jax.tree.map(jnp.zeros_like, residual)
        for _ in range(level.pre_smoothing):
            defect = _subtract(residual, level.operator.mv(estimate))
            estimate = _add(
                estimate,
                level.smoother.apply(defect, iteration=iteration),
            )
        defect = _subtract(residual, level.operator.mv(estimate))
        if level.restriction is None or level.prolongation is None:
            raise RuntimeError("Prepared non-coarse level is missing transfers.")
        coarse_residual = level.restriction.mv(defect)
        coarse_level = self.hierarchy.levels[level_index + 1]
        coarse_correction = jax.tree.map(jnp.zeros_like, coarse_residual)
        visits = 1 if cycle_kind == "v" else 2
        for visit in range(visits):
            coarse_defect = _subtract(
                coarse_residual,
                coarse_level.operator.mv(coarse_correction),
            )
            nested_kind = "v" if cycle_kind == "f" and visit == 1 else cycle_kind
            nested = self._cycle(
                level_index + 1,
                coarse_defect,
                iteration=iteration,
                cycle_kind=nested_kind,
            )
            coarse_correction = _add(coarse_correction, nested)
        estimate = _add(estimate, level.prolongation.mv(coarse_correction))
        for _ in range(level.post_smoothing):
            defect = _subtract(residual, level.operator.mv(estimate))
            estimate = _add(
                estimate,
                level.smoother.apply(defect, iteration=iteration),
            )
        return estimate

    def _full_cycle(
        self,
        level_index: int,
        residual: PyTree[Array],
        /,
        *,
        iteration: ArrayLike | None,
    ) -> PyTree[Array]:
        level = self.hierarchy.levels[level_index]
        if level_index == len(self.hierarchy.levels) - 1:
            return level.smoother.apply(residual, iteration=iteration)
        if level.restriction is None or level.prolongation is None:
            raise RuntimeError("Prepared non-coarse level is missing transfers.")
        coarse_residual = level.restriction.mv(residual)
        coarse_estimate = self._full_cycle(
            level_index + 1,
            coarse_residual,
            iteration=iteration,
        )
        estimate = level.prolongation.mv(coarse_estimate)
        defect = _subtract(residual, level.operator.mv(estimate))
        correction = self._cycle(
            level_index,
            defect,
            iteration=iteration,
            cycle_kind="v",
        )
        return _add(estimate, correction)


class MultigridLevelBuilder(StrictModule):
    """Symbolic level operator, transfers, and smoother/coarse-solve recipe."""

    operator: AbstractLinearOperator
    smoother: PreconditionerSource
    restriction: AbstractLinearOperator | None
    prolongation: AbstractLinearOperator | None
    pre_smoothing: int = eqx.field(static=True)
    post_smoothing: int = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        smoother: PreconditionerSource,
        /,
        *,
        restriction: AbstractLinearOperator | None = None,
        prolongation: AbstractLinearOperator | None = None,
        pre_smoothing: int = 1,
        post_smoothing: int = 1,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape or not operator.source.compatible(operator.target):
            raise ValueError(
                "A multigrid level operator must be an unbatched endomorphism."
            )
        if not isinstance(
            smoother, (AbstractPreconditioner, AbstractPreconditionerBuilder)
        ):
            raise TypeError("smoother must be a preconditioner or builder.")
        if isinstance(smoother, AbstractPreconditioner) and not (
            smoother.space.compatible(operator.source)
        ):
            raise ValueError("The smoother must act on the level space.")
        if (restriction is None) != (prolongation is None):
            raise ValueError("restriction and prolongation must be supplied together.")
        if restriction is not None:
            if not isinstance(restriction, AbstractLinearOperator) or not isinstance(
                prolongation, AbstractLinearOperator
            ):
                raise TypeError("restriction and prolongation must be linear operators.")
            if restriction.batch_shape or prolongation.batch_shape:
                raise ValueError("Multigrid transfer operators must be unbatched.")
        pre = int(pre_smoothing)
        post = int(post_smoothing)
        if pre < 0 or post < 0:
            raise ValueError("Smoothing counts must be non-negative.")
        self.operator = operator
        self.smoother = smoother
        self.restriction = restriction
        self.prolongation = prolongation
        self.pre_smoothing = pre
        self.post_smoothing = post


class MultigridHierarchyBuilder(AbstractPreconditionerBuilder):
    """Prepare a policy-selected cycle from explicit immutable levels."""

    levels: tuple[MultigridLevelBuilder, ...]
    properties: PreconditionerProperties
    cycle_policy: MultigridCyclePolicy
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        levels: tuple[MultigridLevelBuilder, ...],
        /,
        *,
        properties: PreconditionerProperties | None = None,
        cycle_policy: MultigridCyclePolicy | None = None,
    ):
        levels_ = tuple(levels)
        if len(levels_) < 2:
            raise ValueError("A multigrid builder requires at least two levels.")
        if not all(isinstance(level, MultigridLevelBuilder) for level in levels_):
            raise TypeError("levels must contain MultigridLevelBuilder values.")
        properties_ = _composition_properties(
            tuple(
                _source_properties(level.smoother, level.operator) for level in levels_
            ),
            properties,
        )
        cycle_policy_ = MultigridCyclePolicy() if cycle_policy is None else cycle_policy
        if not isinstance(cycle_policy_, MultigridCyclePolicy):
            raise TypeError("cycle_policy must be a MultigridCyclePolicy.")
        for index, level in enumerate(levels_[:-1]):
            if level.restriction is None or level.prolongation is None:
                raise ValueError("Every non-coarse builder level requires transfers.")
            coarse_space = levels_[index + 1].operator.source
            if not level.restriction.source.compatible(
                level.operator.source
            ) or not level.restriction.target.compatible(coarse_space):
                raise ValueError(f"Restriction space mismatch at transition {index}.")
            if not level.prolongation.source.compatible(
                coarse_space
            ) or not level.prolongation.target.compatible(level.operator.source):
                raise ValueError(f"Prolongation space mismatch at transition {index}.")
        if levels_[-1].restriction is not None or levels_[-1].prolongation is not None:
            raise ValueError("The coarsest builder level cannot carry transfers.")
        self.levels = levels_
        self.properties = properties_
        self.cycle_policy = cycle_policy_
        self._builder_id = canonical_fingerprint(
            {
                "kind": "multigrid-hierarchy-builder",
                "levels": [
                    {
                        "operator": level.operator.operator_id,
                        "smoother": (
                            level.smoother.preconditioner_id
                            if isinstance(level.smoother, AbstractPreconditioner)
                            else level.smoother.builder_id
                        ),
                        "restriction": (
                            None
                            if level.restriction is None
                            else level.restriction.operator_id
                        ),
                        "prolongation": (
                            None
                            if level.prolongation is None
                            else level.prolongation.operator_id
                        ),
                        "pre": level.pre_smoothing,
                        "post": level.post_smoothing,
                    }
                    for level in levels_
                ],
                "properties": _preconditioner_properties_payload(properties_),
                "cycle": cycle_policy_.cycle_id,
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> str:
        return "frozen"

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        if setup_operator.operator_id != self.levels[0].operator.operator_id:
            raise ValueError(
                "The setup operator must preserve the finest hierarchy operator ID."
            )
        if not setup_operator.source.compatible(self.levels[0].operator.source):
            raise ValueError("The setup operator must preserve the finest level space.")
        return self.properties

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        self.properties_for(setup_operator)
        storage = 0
        preparation_workspace = 0
        apply_workspace = 0
        setup_matvecs = 0
        accepted = True
        rejected_reasons: list[str] = []
        for index, level in enumerate(self.levels):
            operator = setup_operator if index == 0 else level.operator
            estimate = _source_cost(
                level.smoother, operator, materialization=materialization
            )
            storage += estimate.storage_bytes
            preparation_workspace = max(
                preparation_workspace,
                estimate.preparation_workspace_bytes,
            )
            apply_workspace += estimate.apply_workspace_bytes_per_rhs
            setup_matvecs += estimate.setup_matvec_count
            if not estimate.accepted:
                accepted = False
                rejected_reasons.append(
                    f"level {index} smoother {estimate.component}: {estimate.reason}"
                )
            apply_workspace += (
                4 * operator.source.size * _coordinate_dtype(operator.source).itemsize
            )
        hierarchy_arrays = tuple(
            (
                None if index == 0 else level.operator,
                level.restriction,
                level.prolongation,
            )
            for index, level in enumerate(self.levels)
        )
        storage += _array_tree_storage_bytes(hierarchy_arrays)
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=storage,
            preparation_workspace_bytes=preparation_workspace,
            apply_workspace_bytes_per_rhs=apply_workspace,
            setup_matvec_count=setup_matvecs,
            accepted=accepted,
            reason=(
                "explicit multigrid hierarchy state"
                if accepted
                else "; ".join(rejected_reasons)
            ),
        )

    def prepare_hierarchy(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> MultigridHierarchy:
        self.properties_for(setup_operator)
        prepared_levels: list[MultigridLevel] = []
        preparation_workspace = 0
        for index, specification in enumerate(self.levels):
            operator = setup_operator if index == 0 else specification.operator
            source = specification.smoother
            estimate = _source_cost(
                source,
                operator,
                materialization=materialization,
            )
            if not estimate.accepted:
                raise LinearCapabilityError(
                    f"Level {index} preconditioner rejected setup: {estimate.reason}"
                )
            preparation_workspace = max(
                preparation_workspace,
                estimate.preparation_workspace_bytes,
            )
            if isinstance(source, AbstractPreconditioner):
                smoother = source
            else:
                smoother = source.prepare(operator, materialization=materialization)
            prepared_levels.append(
                MultigridLevel(
                    operator,
                    smoother,
                    restriction=specification.restriction,
                    prolongation=specification.prolongation,
                    pre_smoothing=specification.pre_smoothing,
                    post_smoothing=specification.post_smoothing,
                )
            )
        levels = tuple(prepared_levels)
        diagnostics = _default_setup_diagnostics(
            levels,
            setup_workspace_bytes=preparation_workspace,
            reuse_dependency_fingerprint=self.builder_id,
        )
        return MultigridHierarchy(
            levels,
            properties=self.properties,
            diagnostics=diagnostics,
            hierarchy_id=canonical_fingerprint(
                {
                    "kind": "prepared-multigrid-hierarchy",
                    "builder": self.builder_id,
                    "setup_operator": setup_operator.operator_id,
                }
            ),
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
            raise TypeError("Multigrid refresh requires a MultigridPreconditioner.")
        return self.prepare(setup_operator, materialization=materialization)


def multigrid_hierarchy_from_pyamg(
    solver: Any,
    /,
    *,
    properties: PreconditionerProperties | None = None,
    relaxation: float = 2.0 / 3.0,
    pre_smoothing: int = 1,
    post_smoothing: int = 1,
    materialization: MaterializationPolicy | None = None,
) -> MultigridHierarchy:
    """Convert a host-built PyAMG hierarchy into JAX-resident Phydrax values."""
    pyamg_levels = tuple(solver.levels)
    if len(pyamg_levels) < 2:
        raise ValueError("The PyAMG solver must contain at least two levels.")
    operator_properties = OperatorProperties()
    specifications: list[MultigridLevelBuilder] = []
    for index, pyamg_level in enumerate(pyamg_levels):
        operator = _operator_from_scipy(
            pyamg_level.A,
            properties=operator_properties,
        )
        if index == len(pyamg_levels) - 1:
            specifications.append(
                MultigridLevelBuilder(
                    operator,
                    DenseInversePreconditionerBuilder(),
                    pre_smoothing=0,
                    post_smoothing=0,
                )
            )
            continue
        restriction = _operator_from_scipy(pyamg_level.R)
        prolongation = _operator_from_scipy(pyamg_level.P)
        specifications.append(
            MultigridLevelBuilder(
                operator,
                JacobiPreconditionerBuilder(relaxation=relaxation),
                restriction=restriction,
                prolongation=prolongation,
                pre_smoothing=pre_smoothing,
                post_smoothing=post_smoothing,
            )
        )
    builder = MultigridHierarchyBuilder(
        tuple(specifications),
        properties=properties,
    )
    return builder.prepare_hierarchy(
        specifications[0].operator,
        materialization=(
            MaterializationPolicy() if materialization is None else materialization
        ),
    )


def _operator_from_scipy(
    matrix: Any,
    /,
    *,
    properties: OperatorProperties | None = None,
) -> AbstractLinearOperator:
    from ..sparse import EdgeRelation, SparseLinearMap

    coo = matrix.tocoo()
    relation = EdgeRelation(
        coo.col,
        coo.row,
        source_size=int(coo.shape[1]),
        target_size=int(coo.shape[0]),
    )
    return SparseLinearMap(
        relation,
        coo.data,
        properties=properties,
    )


def _add(left: PyTree[Array], right: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x + y, left, right)


def _subtract(left: PyTree[Array], right: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x - y, left, right)


__all__ = [
    "MultigridCycleKind",
    "MultigridCyclePolicy",
    "MultigridHierarchy",
    "MultigridSetupDiagnostics",
    "MultigridHierarchyBuilder",
    "MultigridPreconditioner",
    "MultigridLevel",
    "MultigridLevelBuilder",
    "multigrid_hierarchy_from_pyamg",
]
