#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._materialization import materialize
from .._operators import AbstractLinearOperator, DenseLinearOperator
from .._pairings import DiagonalPairing, EuclideanPairing
from .._spaces import ArraySpace
from ._general import _dense_generalized_schur, GeneralEigenproblem
from ._schur import (
    prepare_schur_eigensolve,
    PreparedSchurSolve,
    schur_eigensolve,
    SchurEigenproblem,
    SchurSolvePolicy,
)


class ResolventScanStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_INPUT = 1
    SCHUR_FAILURE = 2
    NONFINITE_OUTPUT = 3


class ResolventScanProblem(StrictModule):
    """One standard endomorphism and a fixed vector of complex shifts."""

    operator: AbstractLinearOperator
    shifts: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        shifts: ArrayLike,
        /,
        *,
        problem_id: str | None = None,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape or not operator.source.compatible(operator.target):
            raise ValueError("Resolvent scans require an unbatched endomorphism.")
        if not isinstance(operator.source, ArraySpace):
            raise TypeError("Dense resolvent scans initially require ArraySpace values.")
        shifts_ = jnp.asarray(shifts)
        if shifts_.ndim != 1 or shifts_.size == 0:
            raise ValueError("shifts must be one nonempty rank-one array.")
        if not jnp.issubdtype(shifts_.dtype, jnp.inexact):
            shifts_ = shifts_.astype(float)
        shifts_ = shifts_.astype(jnp.result_type(shifts_.dtype, 1j))
        shifts_ = eqx.error_if(
            shifts_,
            jnp.any(~jnp.isfinite(shifts_)),
            "Resolvent shifts must be finite.",
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "resolvent-scan-problem",
                    "operator": operator.operator_id,
                    "source": operator.source.space_id,
                    "shifts": array_tree_fingerprint(shifts_),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.operator = operator
        self.shifts = shifts_
        self.problem_id = identifier


class ResolventScanPolicy(StrictModule, NonTrainableState):
    schur: SchurSolvePolicy
    relative_singularity_tolerance: float = eqx.field(static=True)
    absolute_singularity_tolerance: float = eqx.field(static=True)
    maximum_shifts: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        schur: SchurSolvePolicy | None = None,
        relative_singularity_tolerance: float = 1e-12,
        absolute_singularity_tolerance: float = 0.0,
        maximum_shifts: int = 65_536,
        maximum_workspace_bytes: int = 1024 * 1024**2,
    ):
        schur_ = SchurSolvePolicy() if schur is None else schur
        relative = float(relative_singularity_tolerance)
        absolute = float(absolute_singularity_tolerance)
        maximum = int(maximum_shifts)
        workspace = int(maximum_workspace_bytes)
        if not isinstance(schur_, SchurSolvePolicy):
            raise TypeError("schur must be a SchurSolvePolicy or None.")
        if (
            not math.isfinite(relative)
            or not math.isfinite(absolute)
            or relative < 0.0
            or absolute < 0.0
            or maximum <= 0
            or workspace <= 0
        ):
            raise ValueError("Resolvent scan policy values are invalid.")
        self.schur = schur_
        self.relative_singularity_tolerance = relative
        self.absolute_singularity_tolerance = absolute
        self.maximum_shifts = maximum
        self.maximum_workspace_bytes = workspace


class ResolventScanPlan(StrictModule, NonTrainableState):
    policy: ResolventScanPolicy
    dimension: int = eqx.field(static=True)
    shift_count: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedResolventScan(StrictModule):
    problem: ResolventScanProblem
    schur: PreparedSchurSolve
    plan: ResolventScanPlan = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: Array


class ResolventScanDiagnostics(StrictModule):
    minimum_singular_values: Array
    singular_mask: Array
    finite: Array
    schur_status: Array
    decomposition_count: Array
    workspace_bytes: int = eqx.field(static=True)


class ResolventScanProvenance(StrictModule):
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    pairing_id: str = eqx.field(static=True)
    norm_definition: str = eqx.field(static=True)
    numeric_version: Array


class ResolventScanResult(StrictModule):
    shifts: Array
    minimum_singular_values: Array
    resolvent_norms: Array
    status: Array
    diagnostics: ResolventScanDiagnostics
    provenance: ResolventScanProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(ResolventScanStatus.SUCCESS)


def plan_resolvent_scan(
    problem: ResolventScanProblem,
    policy: ResolventScanPolicy | None = None,
    /,
) -> ResolventScanPlan:
    if not isinstance(problem, ResolventScanProblem):
        raise TypeError("problem must be a ResolventScanProblem.")
    policy_ = ResolventScanPolicy() if policy is None else policy
    if not isinstance(policy_, ResolventScanPolicy):
        raise TypeError("policy must be a ResolventScanPolicy or None.")
    dimension = problem.operator.source.size
    shift_count = int(problem.shifts.size)
    if shift_count > policy_.maximum_shifts:
        raise ValueError("Resolvent shift count exceeds maximum_shifts.")
    itemsize = np.dtype(
        jnp.result_type(problem.shifts.dtype, problem.operator.source.dtype)
    ).itemsize
    workspace = shift_count * dimension * dimension * itemsize
    if workspace > policy_.maximum_workspace_bytes:
        raise ValueError("Resolvent scan exceeds maximum_workspace_bytes.")
    return ResolventScanPlan(
        policy=policy_,
        dimension=dimension,
        shift_count=shift_count,
        workspace_bytes=workspace,
        problem_id=problem.problem_id,
        source_space_id=problem.operator.source.space_id,
        operator_id=problem.operator.operator_id,
        plan_id=canonical_fingerprint(
            {
                "kind": "resolvent-scan-plan",
                "problem": problem.problem_id,
                "operator": problem.operator.operator_id,
                "dimension": dimension,
                "shift_count": shift_count,
                "relative_tolerance": policy_.relative_singularity_tolerance,
                "absolute_tolerance": policy_.absolute_singularity_tolerance,
                "workspace_bytes": workspace,
            }
        ),
    )


def prepare_resolvent_scan(
    problem: ResolventScanProblem,
    policy: ResolventScanPolicy | ResolventScanPlan | None = None,
    /,
) -> PreparedResolventScan:
    plan = (
        policy
        if isinstance(policy, ResolventScanPlan)
        else plan_resolvent_scan(problem, policy)
    )
    return _prepare_resolvent(problem, plan, numeric_version=0)


def refresh_resolvent_scan(
    prepared: PreparedResolventScan,
    problem: ResolventScanProblem,
    /,
) -> PreparedResolventScan:
    if not isinstance(prepared, PreparedResolventScan):
        raise TypeError("prepared must be a PreparedResolventScan.")
    return _prepare_resolvent(
        problem,
        prepared.plan,
        numeric_version=int(np.asarray(prepared.numeric_version)) + 1,
        prepared_id=prepared.prepared_id,
    )


def resolvent_scan(
    problem_or_prepared: ResolventScanProblem | PreparedResolventScan,
    /,
    *,
    policy: ResolventScanPolicy | ResolventScanPlan | None = None,
) -> ResolventScanResult:
    if isinstance(problem_or_prepared, PreparedResolventScan):
        if policy is not None:
            raise ValueError("policy must be omitted for a prepared resolvent scan.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, ResolventScanProblem):
        prepared = prepare_resolvent_scan(problem_or_prepared, policy)
    else:
        raise TypeError("Expected a ResolventScanProblem or PreparedResolventScan.")
    schur = schur_eigensolve(prepared.schur)
    form = schur.schur_form
    identity = jnp.eye(prepared.plan.dimension, dtype=form.dtype)

    def minimum_singular_value(shift):
        singular_values = jnp.linalg.svd(
            form - shift.astype(form.dtype) * identity,
            full_matrices=False,
            compute_uv=False,
        )
        return singular_values[-1]

    minimum = jax.vmap(minimum_singular_value)(prepared.problem.shifts)
    scale = jnp.maximum(jnp.linalg.norm(form), jnp.asarray(1.0, dtype=form.real.dtype))
    tolerance = (
        prepared.plan.policy.absolute_singularity_tolerance
        + prepared.plan.policy.relative_singularity_tolerance * scale
    )
    singular = minimum <= tolerance
    norms = jnp.where(singular, jnp.inf, jnp.reciprocal(minimum))
    input_finite = jnp.all(jnp.isfinite(prepared.problem.shifts)) & jnp.all(
        jnp.isfinite(prepared.schur.matrix)
    )
    output_finite = jnp.all(jnp.isfinite(minimum))
    status = jnp.where(
        ~input_finite,
        int(ResolventScanStatus.NONFINITE_INPUT),
        jnp.where(
            ~schur.successful,
            int(ResolventScanStatus.SCHUR_FAILURE),
            jnp.where(
                ~output_finite,
                int(ResolventScanStatus.NONFINITE_OUTPUT),
                int(ResolventScanStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    pairing = prepared.problem.operator.source.pairing
    return ResolventScanResult(
        shifts=prepared.problem.shifts,
        minimum_singular_values=minimum,
        resolvent_norms=norms,
        status=status,
        diagnostics=ResolventScanDiagnostics(
            minimum_singular_values=minimum,
            singular_mask=singular,
            finite=input_finite & output_finite,
            schur_status=schur.status,
            decomposition_count=schur.diagnostics.decomposition_count,
            workspace_bytes=prepared.plan.workspace_bytes,
        ),
        provenance=ResolventScanProvenance(
            problem_id=prepared.problem.problem_id,
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=prepared.problem.operator.operator_id,
            source_space_id=prepared.problem.operator.source.space_id,
            pairing_id=pairing.pairing_id,
            norm_definition="induced Hilbert norm after pairing square-root coordinates",
            numeric_version=prepared.numeric_version,
        ),
    )


def _prepare_resolvent(
    problem: ResolventScanProblem,
    plan: ResolventScanPlan,
    /,
    *,
    numeric_version: int,
    prepared_id: str | None = None,
) -> PreparedResolventScan:
    if not isinstance(problem, ResolventScanProblem) or not isinstance(
        plan, ResolventScanPlan
    ):
        raise TypeError("Resolvent preparation requires a problem and plan.")
    if (
        problem.problem_id != plan.problem_id
        or problem.operator.operator_id != plan.operator_id
        or problem.operator.source.size != plan.dimension
        or int(problem.shifts.size) != plan.shift_count
        or problem.operator.source.space_id != plan.source_space_id
    ):
        raise ValueError("Resolvent problem is incompatible with the symbolic plan.")
    matrix = materialize(
        problem.operator,
        plan.policy.schur.materialization,
    )
    canonical = _canonical_pairing_matrix(problem.operator.source, matrix)
    space = ArraySpace((plan.dimension,), dtype=canonical.dtype)
    canonical_operator = DenseLinearOperator(
        canonical,
        source=space,
        target=space,
        operator_id=canonical_fingerprint(
            {
                "kind": "pairing-canonical-resolvent-operator",
                "operator": problem.operator.operator_id,
                "pairing": problem.operator.source.pairing.pairing_id,
            }
        ),
    )
    schur = prepare_schur_eigensolve(
        SchurEigenproblem(canonical_operator),
        plan.policy.schur,
    )
    version = jnp.asarray(numeric_version, dtype=jnp.int32)
    return PreparedResolventScan(
        problem=problem,
        schur=schur,
        plan=plan,
        prepared_id=(
            canonical_fingerprint(
                {
                    "kind": "prepared-resolvent-scan",
                    "plan": plan.plan_id,
                    "operator": problem.operator.operator_id,
                    "shifts": array_tree_fingerprint(problem.shifts),
                    "numeric_version": numeric_version,
                }
            )
            if prepared_id is None
            else str(prepared_id)
        ),
        numeric_version=version,
    )


def _canonical_pairing_matrix(space: ArraySpace, matrix: Array, /) -> Array:
    pairing = space.pairing
    if isinstance(pairing, EuclideanPairing):
        return jnp.asarray(matrix)
    if isinstance(pairing, DiagonalPairing):
        weights = jnp.asarray(pairing.weights).reshape((-1,))
        square_root = jnp.sqrt(weights)
        return square_root[:, None] * jnp.asarray(matrix) / square_root[None, :]
    raise TypeError("Resolvent scans require Euclidean or positive diagonal pairings.")


class PencilPseudospectrumStatus(IntEnum):
    """Portable execution status for an explicit-norm pencil scan."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    QZ_FAILURE = 2
    NONFINITE_OUTPUT = 3


class PencilPerturbationNorm(StrictModule, NonTrainableState):
    """Weighted joint complex-Frobenius perturbations of ``(A, B)``.

    A zero scale freezes that pencil member.  At least one member must remain
    perturbable.
    """

    operator_scale: float = eqx.field(static=True)
    mass_scale: float = eqx.field(static=True)
    norm_id: str = eqx.field(static=True)

    def __init__(self, operator_scale: float = 1.0, mass_scale: float = 0.0, /):
        operator = float(operator_scale)
        mass = float(mass_scale)
        if (
            not math.isfinite(operator)
            or not math.isfinite(mass)
            or operator < 0.0
            or mass < 0.0
            or (operator == 0.0 and mass == 0.0)
        ):
            raise ValueError(
                "Pencil perturbation scales must be finite and nonnegative, "
                "and cannot both be zero."
            )
        self.operator_scale = operator
        self.mass_scale = mass
        self.norm_id = canonical_fingerprint(
            {
                "kind": "joint-complex-frobenius-pencil-norm",
                "operator_scale": operator,
                "mass_scale": mass,
            }
        )


class PencilPseudospectrumProblem(StrictModule):
    """One dense finite-dimensional pencil and fixed projective shifts."""

    eigenproblem: GeneralEigenproblem
    homogeneous_shifts: Array
    perturbation_norm: PencilPerturbationNorm
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        eigenproblem: GeneralEigenproblem,
        homogeneous_shifts: ArrayLike,
        perturbation_norm: PencilPerturbationNorm,
        /,
        *,
        problem_id: str | None = None,
    ):
        if not isinstance(eigenproblem, GeneralEigenproblem):
            raise TypeError("eigenproblem must be a GeneralEigenproblem.")
        if not isinstance(eigenproblem.operator, DenseLinearOperator) or (
            eigenproblem.mass_operator is not None
            and not isinstance(eigenproblem.mass_operator, DenseLinearOperator)
        ):
            raise TypeError(
                "Pencil pseudospectra require explicitly dense pencil members."
            )
        if not isinstance(eigenproblem.operator.source, ArraySpace):
            raise TypeError("Pencil pseudospectra require an ArraySpace.")
        if not isinstance(perturbation_norm, PencilPerturbationNorm):
            raise TypeError("perturbation_norm must be a PencilPerturbationNorm.")
        shifts = jnp.asarray(homogeneous_shifts)
        if shifts.ndim != 2 or shifts.shape[1] != 2 or shifts.shape[0] == 0:
            raise ValueError(
                "homogeneous_shifts must have nonempty shape (shift_count, 2)."
            )
        if not jnp.issubdtype(shifts.dtype, jnp.inexact):
            shifts = shifts.astype(float)
        shifts = shifts.astype(jnp.result_type(shifts.dtype, 1j))
        if not bool(jnp.all(jnp.isfinite(shifts))):
            raise ValueError("Homogeneous pencil shifts must be finite.")
        magnitudes = jnp.sqrt(jnp.abs(shifts[:, 0]) ** 2 + jnp.abs(shifts[:, 1]) ** 2)
        if bool(jnp.any(magnitudes == 0.0)):
            raise ValueError("Each homogeneous shift must be nonzero.")
        normalized = shifts / magnitudes[:, None]
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "pencil-pseudospectrum-problem",
                    "eigenproblem": eigenproblem.problem_id,
                    "shifts": array_tree_fingerprint(normalized),
                    "norm": perturbation_norm.norm_id,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.eigenproblem = eigenproblem
        self.homogeneous_shifts = normalized
        self.perturbation_norm = perturbation_norm
        self.problem_id = identifier

    @classmethod
    def from_complex_shifts(
        cls,
        eigenproblem: GeneralEigenproblem,
        shifts: ArrayLike,
        perturbation_norm: PencilPerturbationNorm,
        /,
        *,
        problem_id: str | None = None,
    ) -> "PencilPseudospectrumProblem":
        values = jnp.asarray(shifts)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Complex shifts must be one nonempty rank-one array.")
        homogeneous = jnp.stack((values, jnp.ones_like(values)), axis=-1)
        return cls(
            eigenproblem,
            homogeneous,
            perturbation_norm,
            problem_id=problem_id,
        )

    @property
    def alpha(self) -> Array:
        return self.homogeneous_shifts[:, 0]

    @property
    def beta(self) -> Array:
        return self.homogeneous_shifts[:, 1]


class PencilPseudospectrumPolicy(StrictModule, NonTrainableState):
    reconstruction_tolerance: float = eqx.field(static=True)
    singularity_tolerance: float = eqx.field(static=True)
    maximum_dimension: int = eqx.field(static=True)
    maximum_shifts: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        reconstruction_tolerance: float = 5e-5,
        singularity_tolerance: float = 1e-12,
        maximum_dimension: int = 4096,
        maximum_shifts: int = 65_536,
        maximum_workspace_bytes: int = 1024 * 1024**2,
    ):
        reconstruction = float(reconstruction_tolerance)
        singularity = float(singularity_tolerance)
        dimension = int(maximum_dimension)
        shifts = int(maximum_shifts)
        workspace = int(maximum_workspace_bytes)
        if (
            not math.isfinite(reconstruction)
            or not math.isfinite(singularity)
            or reconstruction <= 0.0
            or singularity < 0.0
            or dimension <= 0
            or shifts <= 0
            or workspace <= 0
        ):
            raise ValueError("Pencil pseudospectrum policy values are invalid.")
        self.reconstruction_tolerance = reconstruction
        self.singularity_tolerance = singularity
        self.maximum_dimension = dimension
        self.maximum_shifts = shifts
        self.maximum_workspace_bytes = workspace
        self.policy_id = canonical_fingerprint(
            {
                "kind": "pencil-pseudospectrum-policy",
                "reconstruction_tolerance": reconstruction,
                "singularity_tolerance": singularity,
                "maximum_dimension": dimension,
                "maximum_shifts": shifts,
                "maximum_workspace_bytes": workspace,
            }
        )


class PencilPseudospectrumPlan(StrictModule, NonTrainableState):
    policy: PencilPseudospectrumPolicy
    perturbation_norm: PencilPerturbationNorm
    dimension: int = eqx.field(static=True)
    shift_count: int = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    pairing_id: str = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedPencilPseudospectrum(StrictModule, NonTrainableState):
    problem: PencilPseudospectrumProblem
    schur_operator: Array
    schur_mass: Array
    left_schur_vectors: Array
    right_schur_vectors: Array
    plan: PencilPseudospectrumPlan = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: Array
    operator_reconstruction_residual: Array
    mass_reconstruction_residual: Array
    left_unitarity_residual: Array
    right_unitarity_residual: Array


class PencilPseudospectrumDiagnostics(StrictModule):
    minimum_singular_values: Array
    backward_errors: Array
    singular_mask: Array
    frozen_direction_mask: Array
    finite: Array
    decomposition_count: Array
    operator_reconstruction_residual: Array
    mass_reconstruction_residual: Array
    left_unitarity_residual: Array
    right_unitarity_residual: Array
    workspace_bytes: int = eqx.field(static=True)


class PencilPseudospectrumProvenance(StrictModule):
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    mass_operator_id: str | None = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    pairing_id: str = eqx.field(static=True)
    perturbation_norm_id: str = eqx.field(static=True)
    norm_definition: str = eqx.field(static=True)
    operator_scale: float = eqx.field(static=True)
    mass_scale: float = eqx.field(static=True)
    numeric_version: Array


class PencilPseudospectrumResult(StrictModule):
    homogeneous_shifts: Array
    display_values: Array
    minimum_singular_values: Array
    backward_errors: Array
    status: Array
    diagnostics: PencilPseudospectrumDiagnostics
    provenance: PencilPseudospectrumProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(PencilPseudospectrumStatus.SUCCESS)

    def membership(self, epsilon: ArrayLike, /) -> Array:
        threshold = jnp.asarray(epsilon, dtype=self.backward_errors.dtype)
        threshold = eqx.error_if(
            threshold,
            jnp.any(~jnp.isfinite(threshold)) | jnp.any(threshold < 0.0),
            "Pseudospectrum epsilon must be finite and nonnegative.",
        )
        return self.backward_errors <= threshold


def plan_pencil_pseudospectrum(
    problem: PencilPseudospectrumProblem,
    policy: PencilPseudospectrumPolicy | None = None,
    /,
) -> PencilPseudospectrumPlan:
    if not isinstance(problem, PencilPseudospectrumProblem):
        raise TypeError("problem must be a PencilPseudospectrumProblem.")
    policy_ = PencilPseudospectrumPolicy() if policy is None else policy
    if not isinstance(policy_, PencilPseudospectrumPolicy):
        raise TypeError("policy must be a PencilPseudospectrumPolicy or None.")
    dimension = problem.eigenproblem.dimension
    shift_count = int(problem.homogeneous_shifts.shape[0])
    if dimension > policy_.maximum_dimension:
        raise ValueError("Pencil dimension exceeds maximum_dimension.")
    if shift_count > policy_.maximum_shifts:
        raise ValueError("Pencil shift count exceeds maximum_shifts.")
    dtype = jnp.result_type(
        problem.eigenproblem.operator.matrix.dtype,
        (
            problem.eigenproblem.mass_operator.matrix.dtype
            if problem.eigenproblem.mass_operator is not None
            else problem.eigenproblem.operator.matrix.dtype
        ),
        problem.homogeneous_shifts.dtype,
        1j,
    )
    itemsize = np.dtype(dtype).itemsize
    workspace = (shift_count + 6) * dimension * dimension * itemsize
    if workspace > policy_.maximum_workspace_bytes:
        raise ValueError("Pencil scan exceeds maximum_workspace_bytes.")
    space = problem.eigenproblem.operator.source
    _canonical_pairing_matrix(space, problem.eigenproblem.operator.matrix)
    return PencilPseudospectrumPlan(
        policy=policy_,
        perturbation_norm=problem.perturbation_norm,
        dimension=dimension,
        shift_count=shift_count,
        source_space_id=space.space_id,
        pairing_id=space.pairing.pairing_id,
        workspace_bytes=workspace,
        plan_id=canonical_fingerprint(
            {
                "kind": "pencil-pseudospectrum-plan",
                "policy": policy_.policy_id,
                "dimension": dimension,
                "shift_count": shift_count,
                "source": space.space_id,
                "pairing": space.pairing.pairing_id,
                "norm": problem.perturbation_norm.norm_id,
                "workspace_bytes": workspace,
            }
        ),
    )


def prepare_pencil_pseudospectrum(
    problem: PencilPseudospectrumProblem,
    policy: PencilPseudospectrumPolicy | PencilPseudospectrumPlan | None = None,
    /,
) -> PreparedPencilPseudospectrum:
    plan = (
        policy
        if isinstance(policy, PencilPseudospectrumPlan)
        else plan_pencil_pseudospectrum(problem, policy)
    )
    return _prepare_pencil_pseudospectrum(problem, plan, numeric_version=0)


def refresh_pencil_pseudospectrum(
    prepared: PreparedPencilPseudospectrum,
    problem: PencilPseudospectrumProblem,
    /,
) -> PreparedPencilPseudospectrum:
    if not isinstance(prepared, PreparedPencilPseudospectrum):
        raise TypeError("prepared must be a PreparedPencilPseudospectrum.")
    version = int(np.asarray(prepared.numeric_version)) + 1
    return _prepare_pencil_pseudospectrum(
        problem,
        prepared.plan,
        numeric_version=version,
        prepared_id=prepared.prepared_id,
    )


def pencil_pseudospectrum(
    problem_or_prepared: PencilPseudospectrumProblem | PreparedPencilPseudospectrum,
    /,
    *,
    policy: PencilPseudospectrumPolicy | PencilPseudospectrumPlan | None = None,
) -> PencilPseudospectrumResult:
    if isinstance(problem_or_prepared, PreparedPencilPseudospectrum):
        if policy is not None:
            raise ValueError("policy must be omitted for a prepared pencil scan.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, PencilPseudospectrumProblem):
        prepared = prepare_pencil_pseudospectrum(problem_or_prepared, policy)
    else:
        raise TypeError("Expected a PencilPseudospectrumProblem or prepared pencil scan.")
    shifts = jax.lax.stop_gradient(prepared.problem.homogeneous_shifts)
    alpha = shifts[:, 0]
    beta = shifts[:, 1]
    schur_a = jax.lax.stop_gradient(prepared.schur_operator)
    schur_b = jax.lax.stop_gradient(prepared.schur_mass)

    def minimum_singular_value(alpha_beta):
        alpha_, beta_ = alpha_beta
        singular_values = jnp.linalg.svd(
            beta_.astype(schur_a.dtype) * schur_a
            - alpha_.astype(schur_b.dtype) * schur_b,
            full_matrices=False,
            compute_uv=False,
        )
        return singular_values[-1]

    minimum = jax.vmap(minimum_singular_value)(shifts)
    norm = prepared.plan.perturbation_norm
    denominator = jnp.sqrt(
        jnp.abs(beta) ** 2 * norm.operator_scale**2
        + jnp.abs(alpha) ** 2 * norm.mass_scale**2
    ).astype(minimum.dtype)
    frozen = denominator == 0.0
    backward = jnp.where(
        frozen,
        jnp.where(minimum == 0.0, 0.0, jnp.inf),
        minimum / jnp.where(frozen, 1.0, denominator),
    )
    scale = jnp.maximum(
        jnp.maximum(jnp.linalg.norm(schur_a), jnp.linalg.norm(schur_b)),
        jnp.asarray(1.0, dtype=minimum.dtype),
    )
    singular = minimum <= prepared.plan.policy.singularity_tolerance * scale
    beta_nonzero = beta != 0.0
    display = jnp.where(
        beta_nonzero,
        alpha / jnp.where(beta_nonzero, beta, jnp.ones_like(beta)),
        jnp.asarray(jnp.inf + 0j, dtype=shifts.dtype),
    )
    input_finite = jnp.all(jnp.isfinite(shifts))
    output_valid = jnp.all(jnp.isfinite(minimum)) & jnp.all(
        jnp.isfinite(backward) | (frozen & jnp.isinf(backward))
    )
    qz_valid = (
        prepared.operator_reconstruction_residual
        <= prepared.plan.policy.reconstruction_tolerance
    ) & (
        prepared.mass_reconstruction_residual
        <= prepared.plan.policy.reconstruction_tolerance
    )
    status = jnp.where(
        ~input_finite,
        int(PencilPseudospectrumStatus.NONFINITE_INPUT),
        jnp.where(
            ~qz_valid,
            int(PencilPseudospectrumStatus.QZ_FAILURE),
            jnp.where(
                ~output_valid,
                int(PencilPseudospectrumStatus.NONFINITE_OUTPUT),
                int(PencilPseudospectrumStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    problem = prepared.problem.eigenproblem
    mass_id = None if problem.mass_operator is None else problem.mass_operator.operator_id
    return PencilPseudospectrumResult(
        homogeneous_shifts=shifts,
        display_values=display,
        minimum_singular_values=minimum,
        backward_errors=backward,
        status=status,
        diagnostics=PencilPseudospectrumDiagnostics(
            minimum_singular_values=minimum,
            backward_errors=backward,
            singular_mask=singular,
            frozen_direction_mask=frozen,
            finite=input_finite & output_valid,
            decomposition_count=jnp.asarray(1, dtype=jnp.int32),
            operator_reconstruction_residual=(prepared.operator_reconstruction_residual),
            mass_reconstruction_residual=prepared.mass_reconstruction_residual,
            left_unitarity_residual=prepared.left_unitarity_residual,
            right_unitarity_residual=prepared.right_unitarity_residual,
            workspace_bytes=prepared.plan.workspace_bytes,
        ),
        provenance=PencilPseudospectrumProvenance(
            problem_id=prepared.problem.problem_id,
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=problem.operator.operator_id,
            mass_operator_id=mass_id,
            source_space_id=problem.operator.source.space_id,
            pairing_id=problem.operator.source.pairing.pairing_id,
            perturbation_norm_id=norm.norm_id,
            norm_definition=(
                "joint weighted unstructured complex Frobenius norm in "
                "pairing-square-root coordinates"
            ),
            operator_scale=norm.operator_scale,
            mass_scale=norm.mass_scale,
            numeric_version=prepared.numeric_version,
        ),
    )


def _prepare_pencil_pseudospectrum(
    problem: PencilPseudospectrumProblem,
    plan: PencilPseudospectrumPlan,
    /,
    *,
    numeric_version: int,
    prepared_id: str | None = None,
) -> PreparedPencilPseudospectrum:
    if not isinstance(problem, PencilPseudospectrumProblem) or not isinstance(
        plan, PencilPseudospectrumPlan
    ):
        raise TypeError("Pencil preparation requires a problem and plan.")
    eigenproblem = problem.eigenproblem
    space = eigenproblem.operator.source
    if (
        eigenproblem.dimension != plan.dimension
        or int(problem.homogeneous_shifts.shape[0]) != plan.shift_count
        or space.space_id != plan.source_space_id
        or space.pairing.pairing_id != plan.pairing_id
        or problem.perturbation_norm.norm_id != plan.perturbation_norm.norm_id
    ):
        raise ValueError("Pencil problem is incompatible with the symbolic plan.")
    matrix = _canonical_pairing_matrix(space, eigenproblem.operator.matrix)
    mass = (
        jnp.eye(plan.dimension, dtype=matrix.dtype)
        if eigenproblem.mass_operator is None
        else _canonical_pairing_matrix(space, eigenproblem.mass_operator.matrix)
    )
    dtype = jnp.result_type(matrix.dtype, mass.dtype, 1j)
    matrix = matrix.astype(dtype)
    mass = mass.astype(dtype)
    schur_a_np, schur_b_np, left_np, right_np = _dense_generalized_schur(
        np.asarray(matrix),
        np.asarray(mass),
    )
    schur_a = jax.lax.stop_gradient(jnp.asarray(schur_a_np, dtype=dtype))
    schur_b = jax.lax.stop_gradient(jnp.asarray(schur_b_np, dtype=dtype))
    left = jax.lax.stop_gradient(jnp.asarray(left_np, dtype=dtype))
    right = jax.lax.stop_gradient(jnp.asarray(right_np, dtype=dtype))
    identity = jnp.eye(plan.dimension, dtype=dtype)
    matrix_scale = jnp.maximum(jnp.linalg.norm(matrix), 1.0)
    mass_scale = jnp.maximum(jnp.linalg.norm(mass), 1.0)
    operator_residual = (
        jnp.linalg.norm(jnp.conj(left.T) @ matrix @ right - schur_a) / matrix_scale
    )
    mass_residual = (
        jnp.linalg.norm(jnp.conj(left.T) @ mass @ right - schur_b) / mass_scale
    )
    left_unitarity = jnp.linalg.norm(jnp.conj(left.T) @ left - identity)
    right_unitarity = jnp.linalg.norm(jnp.conj(right.T) @ right - identity)
    maximum_residual = jnp.max(
        jnp.stack(
            (
                operator_residual,
                mass_residual,
                left_unitarity,
                right_unitarity,
            )
        )
    )
    if not bool(jnp.isfinite(maximum_residual)) or float(maximum_residual) > (
        plan.policy.reconstruction_tolerance
    ):
        raise ValueError(
            "Generalized Schur reconstruction or unitarity certification failed."
        )
    version = jnp.asarray(numeric_version, dtype=jnp.int32)
    return PreparedPencilPseudospectrum(
        problem=problem,
        schur_operator=schur_a,
        schur_mass=schur_b,
        left_schur_vectors=left,
        right_schur_vectors=right,
        plan=plan,
        prepared_id=(
            canonical_fingerprint(
                {
                    "kind": "prepared-pencil-pseudospectrum",
                    "plan": plan.plan_id,
                    "operator": eigenproblem.operator.operator_id,
                    "mass": (
                        None
                        if eigenproblem.mass_operator is None
                        else eigenproblem.mass_operator.operator_id
                    ),
                    "shifts": array_tree_fingerprint(problem.homogeneous_shifts),
                    "numeric_version": numeric_version,
                }
            )
            if prepared_id is None
            else str(prepared_id)
        ),
        numeric_version=version,
        operator_reconstruction_residual=operator_residual,
        mass_reconstruction_residual=mass_residual,
        left_unitarity_residual=left_unitarity,
        right_unitarity_residual=right_unitarity,
    )


__all__ = [
    "PencilPerturbationNorm",
    "PencilPseudospectrumDiagnostics",
    "PencilPseudospectrumPlan",
    "PencilPseudospectrumPolicy",
    "PencilPseudospectrumProblem",
    "PencilPseudospectrumProvenance",
    "PencilPseudospectrumResult",
    "PencilPseudospectrumStatus",
    "PreparedPencilPseudospectrum",
    "PreparedResolventScan",
    "ResolventScanDiagnostics",
    "ResolventScanPlan",
    "ResolventScanPolicy",
    "ResolventScanProblem",
    "ResolventScanProvenance",
    "ResolventScanResult",
    "ResolventScanStatus",
    "plan_resolvent_scan",
    "prepare_resolvent_scan",
    "refresh_resolvent_scan",
    "resolvent_scan",
    "pencil_pseudospectrum",
    "plan_pencil_pseudospectrum",
    "prepare_pencil_pseudospectrum",
    "refresh_pencil_pseudospectrum",
]
