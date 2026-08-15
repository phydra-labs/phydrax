#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from .._materialization import MaterializationPolicy
from .._preconditioning import PreconditionerPlan
from .._spaces import _coordinate_dtype
from ._policies import (
    AutoEigenMethod,
    DenseEigh,
    EigenMethod,
    EigenSolvePolicy,
    LOBPCG,
    RestartedLanczos,
)
from ._problems import Eigenproblem, EigenproblemLike, GeneralizedEigenproblem


class EigenCostEstimate(StrictModule):
    """Fixed-capacity storage, workspace, and action-count estimate."""

    component: str = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    apply_workspace_bytes: int = eqx.field(static=True)
    operator_matvec_count: int = eqx.field(static=True)
    metric_matvec_count: int = eqx.field(static=True)
    preconditioner_apply_count: int = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    def __init__(
        self,
        component: str,
        storage_bytes: int,
        preparation_workspace_bytes: int,
        apply_workspace_bytes: int,
        operator_matvec_count: int,
        metric_matvec_count: int,
        preconditioner_apply_count: int,
        accepted: bool,
        reason: str,
        /,
    ):
        component_, reason_ = str(component), str(reason)
        if not component_ or not reason_:
            raise ValueError("Eigen cost component and reason must be non-empty.")
        integers = tuple(
            int(value)
            for value in (
                storage_bytes,
                preparation_workspace_bytes,
                apply_workspace_bytes,
                operator_matvec_count,
                metric_matvec_count,
                preconditioner_apply_count,
            )
        )
        if any(value < 0 for value in integers):
            raise ValueError("Eigen cost estimates must be non-negative.")
        self.component = component_
        (
            self.storage_bytes,
            self.preparation_workspace_bytes,
            self.apply_workspace_bytes,
            self.operator_matvec_count,
            self.metric_matvec_count,
            self.preconditioner_apply_count,
        ) = integers
        self.accepted = bool(accepted)
        self.reason = reason_


class EigenSolvePlan(StrictModule):
    """Immutable symbolic selection with all candidate estimates retained."""

    problem_id: str = eqx.field(static=True)
    policy: EigenSolvePolicy
    selected_method: EigenMethod
    available_dimension: int = eqx.field(static=True)
    block_dimension: int = eqx.field(static=True)
    subspace_dimension: int = eqx.field(static=True)
    restart_dimension: int = eqx.field(static=True)
    preconditioner_plan: PreconditionerPlan | None
    candidates: tuple[EigenCostEstimate, ...]
    rejections: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: EigenproblemLike,
        policy: EigenSolvePolicy,
        selected_method: EigenMethod,
        available_dimension: int,
        block_dimension: int,
        subspace_dimension: int,
        restart_dimension: int,
        preconditioner_plan: PreconditionerPlan | None,
        candidates: tuple[EigenCostEstimate, ...],
        rejections: tuple[str, ...],
        /,
    ):
        if not isinstance(problem, (Eigenproblem, GeneralizedEigenproblem)):
            raise TypeError("problem must be an Eigenproblem or GeneralizedEigenproblem.")
        if not isinstance(policy, EigenSolvePolicy):
            raise TypeError("policy must be an EigenSolvePolicy.")
        if not isinstance(selected_method, (DenseEigh, LOBPCG, RestartedLanczos)):
            raise TypeError(
                "selected_method must be DenseEigh, LOBPCG, or RestartedLanczos."
            )
        available = int(available_dimension)
        block, subspace, restart = (
            int(block_dimension),
            int(subspace_dimension),
            int(restart_dimension),
        )
        constraint_capacity = (
            0 if problem.constraints is None else problem.constraints.capacity
        )
        expected_available = problem.dimension - constraint_capacity
        if available != expected_available or available < policy.count:
            raise ValueError("available_dimension is incompatible with the eigenproblem.")
        if (
            policy.differentiation == "eigenvalues"
            and policy.count >= available
            and not isinstance(selected_method, DenseEigh)
        ):
            raise ValueError(
                "Iterative eigenvalue differentiation requires an additional "
                "isolation mode."
            )
        if (
            policy.differentiation == "eigenvalues"
            and problem.constraints is not None
            and problem.constraints.capacity > 0
        ):
            raise ValueError(
                "Eigenvalue differentiation does not support parameter-dependent "
                "excluded-subspace stationarity."
            )
        if isinstance(selected_method, DenseEigh):
            if block != policy.count or subspace != 0 or restart != 0:
                raise ValueError("Invalid resolved DenseEigh dimensions.")
            if problem.constraints is not None and problem.constraints.capacity > 0:
                raise ValueError("DenseEigh does not support excluded constraints.")
            if policy.preconditioning is not None or preconditioner_plan is not None:
                raise ValueError("DenseEigh does not accept preconditioning.")
        elif isinstance(selected_method, LOBPCG):
            if not policy.count <= block <= available or subspace != 0 or restart != 0:
                raise ValueError("Invalid resolved LOBPCG dimensions.")
            if selected_method.block_dimension != block:
                raise ValueError("Selected LOBPCG configuration must match the plan.")
            if (policy.preconditioning is None) != (preconditioner_plan is None):
                raise ValueError(
                    "LOBPCG policy and plan preconditioning state must agree."
                )
        else:
            if not (
                1 <= block <= restart and policy.count <= restart < subspace <= available
            ):
                raise ValueError("Invalid resolved restarted-Lanczos dimensions.")
            if (
                selected_method.subspace_dimension != subspace
                or selected_method.restart_dimension != restart
            ):
                raise ValueError(
                    "Selected restarted-Lanczos configuration must match the plan."
                )
            if policy.preconditioning is not None or preconditioner_plan is not None:
                raise ValueError("Restarted Lanczos cannot own preconditioning.")
        if preconditioner_plan is not None:
            if not isinstance(preconditioner_plan, PreconditionerPlan):
                raise TypeError(
                    "preconditioner_plan must be a PreconditionerPlan or None."
                )
            if preconditioner_plan.space_id != problem.operator.source.space_id:
                raise ValueError(
                    "Preconditioner plan space must match the eigenproblem space."
                )
            required = ("linear", "stationary", "self_adjoint", "positive_definite")
            if any(
                not preconditioner_plan.properties.certifies(name) for name in required
            ):
                raise ValueError(
                    "Eigen preconditioner plan lacks required certified properties."
                )
        estimates = tuple(candidates)
        if not estimates or any(
            not isinstance(estimate, EigenCostEstimate) for estimate in estimates
        ):
            raise TypeError("candidates must contain EigenCostEstimate values.")
        selected_name = selected_method.name
        selected_estimate = next(
            (estimate for estimate in estimates if estimate.accepted),
            None,
        )
        if selected_estimate is None or selected_estimate.component != selected_name:
            raise ValueError(
                "The selected method must be the first accepted candidate estimate."
            )
        rejections_ = tuple(str(value) for value in rejections)
        if any(not value for value in rejections_):
            raise ValueError("Plan rejection reasons must be non-empty.")
        if len(rejections_) != sum(not estimate.accepted for estimate in estimates):
            raise ValueError("Plan rejections must cover every rejected candidate.")
        payload = {
            "kind": "eigen-solve-plan",
            "problem": problem.problem_id,
            "operator": problem.operator.operator_id,
            "metric_operator": (
                problem.metric_operator.operator_id
                if isinstance(problem, GeneralizedEigenproblem)
                else None
            ),
            "constraints": (
                None
                if problem.constraints is None
                else {
                    "subspace": problem.constraints.subspace_id,
                    "capacity": problem.constraints.capacity,
                }
            ),
            "count": policy.count,
            "which": policy.which,
            "max_steps": policy.max_steps,
            "differentiation": policy.differentiation,
            "failure": policy.failure.mode,
            "tolerance": {
                "relative": policy.tolerance.relative,
                "absolute": policy.tolerance.absolute,
                "orthogonality": policy.tolerance.orthogonality,
            },
            "materialization": {
                "max_entries": policy.materialization.max_entries,
                "max_bytes": policy.materialization.max_bytes,
            },
            "resources": {
                "preparation_bytes": policy.resources.preparation_bytes,
                "workspace_bytes": policy.resources.workspace_bytes,
                "krylov_basis_bytes": policy.resources.krylov_basis_bytes,
                "preconditioner_bytes": policy.resources.preconditioner_bytes,
                "operator_matvecs": policy.resources.operator_matvecs,
                "metric_matvecs": policy.resources.metric_matvecs,
                "preconditioner_applies": policy.resources.preconditioner_applies,
            },
            "initial_basis": _array_structure(policy.initial_basis),
            "key": _array_structure(policy.key),
            "selected_method": selected_name,
            "available_dimension": available,
            "block_dimension": block,
            "subspace_dimension": subspace,
            "restart_dimension": restart,
            "preconditioner_plan": (
                None if preconditioner_plan is None else preconditioner_plan.plan_id
            ),
            "candidates": [_estimate_payload(estimate) for estimate in estimates],
            "rejections": list(rejections_),
        }
        self.problem_id = problem.problem_id
        self.policy = policy
        self.selected_method = selected_method
        self.available_dimension = available
        self.block_dimension = block
        self.subspace_dimension = subspace
        self.restart_dimension = restart
        self.preconditioner_plan = preconditioner_plan
        self.candidates = estimates
        self.rejections = rejections_
        self.plan_id = canonical_fingerprint(payload)


def make_eigensolve_plan(
    problem: EigenproblemLike,
    policy: EigenSolvePolicy,
    /,
) -> EigenSolvePlan:
    """Select feasible dense, LOBPCG, and restarted-Lanczos candidates."""
    if not isinstance(problem, (Eigenproblem, GeneralizedEigenproblem)):
        raise TypeError("problem must be an Eigenproblem or GeneralizedEigenproblem.")
    if not isinstance(policy, EigenSolvePolicy):
        raise TypeError("policy must be an EigenSolvePolicy.")
    _validate_problem_certificates(problem)
    constraint_capacity = (
        0 if problem.constraints is None else problem.constraints.capacity
    )
    available = problem.dimension - constraint_capacity
    if available < 1:
        raise ValueError("Constraints leave no available eigenproblem dimension.")
    if policy.count > available:
        raise ValueError(
            "Requested eigenvalue count exceeds the conservatively available dimension."
        )
    if (
        policy.differentiation == "eigenvalues"
        and problem.constraints is not None
        and problem.constraints.capacity > 0
    ):
        raise ValueError(
            "Eigenvalue differentiation does not support parameter-dependent "
            "excluded-subspace stationarity."
        )
    _validate_initial_basis(problem, policy)
    preconditioner_plan = _make_preconditioner_plan(problem, policy)
    requested = policy.method
    methods: tuple[EigenMethod, ...]
    if isinstance(requested, AutoEigenMethod):
        dense_full_spectrum = (
            policy.count == available
            and constraint_capacity == 0
            and policy.preconditioning is None
        )
        methods = (
            (DenseEigh(), LOBPCG(), RestartedLanczos())
            if dense_full_spectrum
            else (LOBPCG(), RestartedLanczos())
        )
    else:
        methods = (requested,)
    evaluated = tuple(
        _evaluate_candidate(
            problem,
            policy,
            method,
            available,
            preconditioner_plan,
        )
        for method in methods
    )
    selected_index = next(
        (index for index, entry in enumerate(evaluated) if entry[0].accepted),
        None,
    )
    estimates = tuple(entry[0] for entry in evaluated)
    rejections = tuple(
        f"{estimate.component}: {estimate.reason}"
        for estimate in estimates
        if not estimate.accepted
    )
    if selected_index is None:
        raise ValueError(f"No feasible eigen method: {'; '.join(rejections)}.")
    _, block, subspace, restart = evaluated[selected_index]
    selected = methods[selected_index]
    selected_method: EigenMethod
    selected_preconditioner: PreconditionerPlan | None
    if isinstance(selected, DenseEigh):
        selected_method = selected
        selected_preconditioner = None
    elif isinstance(selected, LOBPCG):
        selected_method = LOBPCG(block_dimension=block)
        selected_preconditioner = preconditioner_plan
    else:
        selected_method = RestartedLanczos(
            subspace_dimension=subspace,
            restart_dimension=restart,
        )
        selected_preconditioner = None
    return EigenSolvePlan(
        problem,
        policy,
        selected_method,
        available,
        block,
        subspace,
        restart,
        selected_preconditioner,
        estimates,
        rejections,
    )


def _validate_problem_certificates(problem: EigenproblemLike, /) -> None:
    operator = problem.operator
    if not operator.source.compatible(operator.target):
        raise ValueError("Eigen plans require an operator endomorphism.")
    if not operator.properties.certifies("self_adjoint"):
        raise ValueError("Eigen plans require certified operator self-adjointness.")
    coordinate_dtype = _coordinate_dtype(operator.source)
    if not np.issubdtype(coordinate_dtype, np.inexact) or coordinate_dtype.itemsize < 4:
        raise TypeError(
            "Native eigen solves require float32/complex64 or wider coordinates; "
            "lower-precision projected eigendecompositions are unsupported."
        )
    if isinstance(problem, GeneralizedEigenproblem):
        metric = problem.metric_operator
        if not metric.source.compatible(operator.source):
            raise ValueError("The metric_operator must be an endomorphism.")
        if not metric.properties.certifies(
            "self_adjoint"
        ) or not metric.properties.certifies("positive_definite"):
            raise ValueError(
                "metric_operator must be certified self-adjoint and positive-definite."
            )


def _validate_initial_basis(
    problem: EigenproblemLike,
    policy: EigenSolvePolicy,
    /,
) -> None:
    basis = policy.initial_basis
    if basis is None:
        return
    if basis.shape[0] != problem.dimension:
        raise ValueError("initial_basis row count must equal the problem dimension.")
    if np.dtype(basis.dtype) != _coordinate_dtype(problem.operator.source):
        raise TypeError("initial_basis dtype must match the problem coordinate dtype.")


def _make_preconditioner_plan(
    problem: EigenproblemLike,
    policy: EigenSolvePolicy,
    /,
) -> PreconditionerPlan | None:
    preconditioning = policy.preconditioning
    if preconditioning is None:
        return None
    if preconditioning.side not in ("auto", "left"):
        raise ValueError("Eigen solves only support left/automatic preconditioning.")
    byte_budget = policy.resources.preconditioner_bytes
    itemsize = np.dtype(_coordinate_dtype(problem.operator.source)).itemsize
    materialization = MaterializationPolicy(
        max_entries=max(1, byte_budget // itemsize),
        max_bytes=max(1, byte_budget),
    )
    plan = PreconditionerPlan(
        preconditioning,
        problem.operator,
        side="left",
        materialization=materialization,
    )
    properties = plan.properties
    required = ("linear", "stationary", "self_adjoint", "positive_definite")
    if any(not properties.certifies(name) for name in required):
        raise ValueError(
            "Eigen preconditioners must be certified linear, stationary, "
            "self-adjoint, and positive-definite."
        )
    cost = plan.cost
    if cost.storage_bytes > policy.resources.preconditioner_bytes:
        raise ValueError("Preconditioner storage exceeds the eigen resource budget.")
    if (
        cost.storage_bytes + cost.preparation_workspace_bytes
        > policy.resources.preparation_bytes
    ):
        raise ValueError("Preconditioner preparation exceeds the eigen resource budget.")
    return plan


def _evaluate_candidate(
    problem: EigenproblemLike,
    policy: EigenSolvePolicy,
    method: EigenMethod,
    available: int,
    preconditioner_plan: PreconditionerPlan | None,
    /,
) -> tuple[EigenCostEstimate, int, int, int]:
    extra = int(policy.differentiation == "eigenvalues")
    supplied_columns = (
        0 if policy.initial_basis is None else int(policy.initial_basis.shape[1])
    )
    if isinstance(method, DenseEigh):
        structural_reason = None
        if problem.batch_shape and policy.differentiation != "none":
            structural_reason = (
                "batched dense eigendecomposition does not expose raw eigenvalue derivatives"
            )
        elif problem.constraints is not None and problem.constraints.capacity > 0:
            structural_reason = "dense full-spectrum solves do not support constraints"
        elif preconditioner_plan is not None:
            structural_reason = "dense eigendecomposition does not accept preconditioning"
        estimate = _dense_cost_estimate(
            problem,
            policy,
            method,
            structural_reason=structural_reason,
        )
        return estimate, policy.count, 0, 0
    if isinstance(method, LOBPCG):
        minimum_block = policy.count + extra
        block = (
            method.block_dimension
            if method.block_dimension is not None
            else min(
                available,
                max(minimum_block, 2 * policy.count, supplied_columns + 1),
            )
        )
        structural_reason = None
        if problem.batch_shape:
            structural_reason = "LOBPCG does not support operator batch axes"
        elif block < minimum_block:
            structural_reason = f"block_dimension must be at least {minimum_block}"
        elif block > available:
            structural_reason = "block_dimension exceeds the available dimension"
        elif supplied_columns > block:
            structural_reason = "initial_basis columns exceed block_dimension"
        estimate = _cost_estimate(
            problem,
            policy,
            component="lobpcg",
            block=max(block, 0),
            subspace=0,
            restart=0,
            preconditioner_plan=preconditioner_plan,
            structural_reason=structural_reason,
        )
        return estimate, max(block, 1), 0, 0
    if isinstance(method, RestartedLanczos):
        restart = (
            policy.count + extra
            if method.restart_dimension is None
            else method.restart_dimension
        )
        subspace = (
            min(available, max(restart + 1, 2 * restart + 8))
            if method.subspace_dimension is None
            else method.subspace_dimension
        )
        block = min(
            restart,
            max(policy.count + extra, supplied_columns + 1),
        )
        structural_reason = None
        if problem.batch_shape:
            structural_reason = "restarted Lanczos does not support operator batch axes"
        elif preconditioner_plan is not None:
            structural_reason = "restarted Lanczos does not support preconditioning"
        elif supplied_columns > block:
            structural_reason = (
                "initial_basis columns exceed the restarted-Lanczos seed dimension"
            )
        elif restart < policy.count + extra:
            structural_reason = (
                f"restart_dimension must be at least {policy.count + extra}"
            )
        elif subspace <= restart:
            structural_reason = "subspace_dimension must exceed restart_dimension"
        elif subspace > available:
            structural_reason = "subspace_dimension exceeds the available dimension"
        estimate = _cost_estimate(
            problem,
            policy,
            component="restarted-lanczos",
            block=block,
            subspace=max(subspace, 0),
            restart=max(restart, 0),
            preconditioner_plan=None,
            structural_reason=structural_reason,
        )
        return estimate, block, max(subspace, 2), max(restart, 1)
    raise TypeError("Unsupported eigen method.")


def _dense_cost_estimate(
    problem: EigenproblemLike,
    policy: EigenSolvePolicy,
    method: DenseEigh,
    /,
    *,
    structural_reason: str | None,
) -> EigenCostEstimate:
    n = problem.dimension
    itemsize = _coordinate_dtype(problem.operator.source).itemsize
    batch_count = int(np.prod(problem.batch_shape)) if problem.batch_shape else 1
    matrix_entries = batch_count * n * n
    matrix_bytes = matrix_entries * itemsize
    generalized = isinstance(problem, GeneralizedEigenproblem)
    materialized_count = 2 if generalized else 1
    storage = 2 * matrix_bytes
    preparation_workspace = (5 + materialized_count) * matrix_bytes
    apply_workspace = 4 * matrix_bytes
    operator_matvecs = batch_count * n
    metric_matvecs = batch_count * n if generalized else 0
    failures: list[str] = []
    if structural_reason is not None:
        failures.append(structural_reason)
    materialization = policy.materialization
    if matrix_entries > materialization.max_entries:
        failures.append(
            f"dense entries {matrix_entries} exceed materialization limit "
            f"{materialization.max_entries}"
        )
    if matrix_bytes > materialization.max_bytes:
        failures.append(
            f"dense bytes {matrix_bytes} exceed materialization limit "
            f"{materialization.max_bytes}"
        )
    resources = policy.resources
    checks = (
        (preparation_workspace, resources.preparation_bytes, "preparation bytes"),
        (apply_workspace, resources.workspace_bytes, "workspace bytes"),
        (operator_matvecs, resources.operator_matvecs, "operator matvecs"),
        (metric_matvecs, resources.metric_matvecs, "metric matvecs"),
    )
    failures.extend(
        f"{name} estimate {required} exceeds budget {budget}"
        for required, budget, name in checks
        if required > budget
    )
    accepted = not failures
    return EigenCostEstimate(
        method.name,
        storage,
        preparation_workspace,
        apply_workspace,
        operator_matvecs,
        metric_matvecs,
        0,
        accepted,
        (
            "dense full-spectrum estimate fits declared budgets"
            if accepted
            else "; ".join(failures)
        ),
    )


def _cost_estimate(
    problem: EigenproblemLike,
    policy: EigenSolvePolicy,
    /,
    *,
    component: str,
    block: int,
    subspace: int,
    restart: int,
    preconditioner_plan: PreconditionerPlan | None,
    structural_reason: str | None,
) -> EigenCostEstimate:
    n = problem.dimension
    itemsize = _coordinate_dtype(problem.operator.source).itemsize
    constraint_capacity = (
        0 if problem.constraints is None else problem.constraints.capacity
    )
    preconditioner_cost = (
        None if preconditioner_plan is None else preconditioner_plan.cost
    )
    preconditioner_storage = (
        0 if preconditioner_cost is None else preconditioner_cost.storage_bytes
    )
    preconditioner_preparation = (
        0
        if preconditioner_cost is None
        else preconditioner_cost.preparation_workspace_bytes
    )
    preconditioner_apply_workspace = (
        0
        if preconditioner_cost is None
        else preconditioner_cost.apply_workspace_bytes_per_rhs * block
    )
    preconditioner_setup_matvecs = (
        0 if preconditioner_cost is None else preconditioner_cost.setup_matvec_count
    )
    generalized = isinstance(problem, GeneralizedEigenproblem)
    derivative_metric_matvecs = (
        policy.count if generalized and policy.differentiation == "eigenvalues" else 0
    )
    supplied_columns = (
        0 if policy.initial_basis is None else int(policy.initial_basis.shape[1])
    )
    supplied_storage = n * supplied_columns * itemsize
    prepared_storage = n * (block + 2 * constraint_capacity) * itemsize
    preparation_scratch = 2 * n * (block + supplied_columns) * itemsize
    constraint_workspace = (
        n * constraint_capacity
        + 4 * constraint_capacity * constraint_capacity
        + 2 * constraint_capacity * max(block, subspace)
    ) * itemsize
    constraint_preparation = (
        n * constraint_capacity
        + 6 * constraint_capacity * constraint_capacity
        + constraint_capacity * (block + supplied_columns)
    ) * itemsize
    if component == "lobpcg":
        algorithm_storage = (7 * n * block + 12 * block * block) * itemsize
        apply_workspace = (
            (12 * n * block + 20 * block * block) * itemsize
            + constraint_workspace
            + preconditioner_apply_workspace
        )
        operator_matvecs = (
            3 * block * (policy.max_steps + 1) + preconditioner_setup_matvecs
        )
        metric_matvecs = (
            (
                3 * block * (policy.max_steps + 1)
                + constraint_capacity
                + block
                + supplied_columns
            )
            if generalized
            else 0
        ) + derivative_metric_matvecs
        preconditioner_applies = (
            block * policy.max_steps if preconditioner_plan is not None else 0
        )
        krylov_storage = algorithm_storage
    else:
        algorithm_storage = (
            3 * n * subspace + 4 * n * restart + 6 * subspace * subspace
        ) * itemsize
        apply_workspace = (
            4 * n * subspace + 10 * subspace * subspace
        ) * itemsize + constraint_workspace
        operator_matvecs = (
            subspace * (policy.max_steps + 1) + preconditioner_setup_matvecs
        )
        metric_matvecs = (
            (
                subspace * (policy.max_steps + 1)
                + constraint_capacity
                + block
                + supplied_columns
            )
            if generalized
            else 0
        ) + derivative_metric_matvecs
        preconditioner_applies = 0
        krylov_storage = algorithm_storage
    storage = (
        supplied_storage + prepared_storage + algorithm_storage + preconditioner_storage
    )
    preparation_workspace = (
        prepared_storage
        + preparation_scratch
        + constraint_preparation
        + preconditioner_preparation
    )
    failures: list[str] = []
    if structural_reason is not None:
        failures.append(structural_reason)
    resources = policy.resources
    checks = (
        (
            preparation_workspace + preconditioner_storage,
            resources.preparation_bytes,
            "preparation bytes",
        ),
        (apply_workspace, resources.workspace_bytes, "workspace bytes"),
        (krylov_storage, resources.krylov_basis_bytes, "Krylov basis bytes"),
        (preconditioner_storage, resources.preconditioner_bytes, "preconditioner bytes"),
        (operator_matvecs, resources.operator_matvecs, "operator matvecs"),
        (metric_matvecs, resources.metric_matvecs, "metric matvecs"),
        (
            preconditioner_applies,
            resources.preconditioner_applies,
            "preconditioner applies",
        ),
    )
    failures.extend(
        f"{name} estimate {required} exceeds budget {budget}"
        for required, budget, name in checks
        if required > budget
    )
    accepted = not failures
    reason = (
        "fixed-capacity estimate fits declared budgets"
        if accepted
        else "; ".join(failures)
    )
    return EigenCostEstimate(
        component,
        storage,
        preparation_workspace,
        apply_workspace,
        operator_matvecs,
        metric_matvecs,
        preconditioner_applies,
        accepted,
        reason,
    )


def _array_structure(value, /) -> dict[str, object] | None:
    if value is None:
        return None
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


def _estimate_payload(estimate: EigenCostEstimate, /) -> dict[str, object]:
    return {
        "component": estimate.component,
        "storage_bytes": estimate.storage_bytes,
        "preparation_workspace_bytes": estimate.preparation_workspace_bytes,
        "apply_workspace_bytes": estimate.apply_workspace_bytes,
        "operator_matvec_count": estimate.operator_matvec_count,
        "metric_matvec_count": estimate.metric_matvec_count,
        "preconditioner_apply_count": estimate.preconditioner_apply_count,
        "accepted": estimate.accepted,
        "reason": estimate.reason,
    }


__all__ = ["EigenCostEstimate", "EigenSolvePlan"]
