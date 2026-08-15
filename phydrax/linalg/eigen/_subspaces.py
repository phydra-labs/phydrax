#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy.linalg as scipy_linalg
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from .._materialization import MaterializationPolicy, materialize
from .._operators import AbstractLinearOperator
from .._policies import FailurePolicy
from .._spaces import _coordinate_dtype
from ._schur import SchurEigenproblem


SpectralSelectionKind: TypeAlias = Literal[
    "real-below",
    "real-above",
    "disk",
    "exterior-disk",
]


class SpectralSubspaceStatus(IntEnum):
    """Portable status for an isolated Riesz spectral subspace."""

    SUCCESS = 0
    NONFINITE = 1
    PROJECTOR_RESIDUAL_TOO_LARGE = 2
    ILL_CONDITIONED = 3


class SpectralProjectorDerivativeStatus(IntEnum):
    """Portable status for one local spectral-projector Fréchet derivative."""

    SUCCESS = 0
    SOURCE_FAILURE = 1
    NONFINITE = 2
    RESIDUAL_TOO_LARGE = 3


class SpectralSelection(StrictModule):
    """Serializable half-plane or disk selector with a protected boundary."""

    kind: SpectralSelectionKind = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    center: complex = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    inclusive: bool = eqx.field(static=True)
    boundary_tolerance: float = eqx.field(static=True)
    expected_dimension: int | None = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: SpectralSelectionKind,
        /,
        *,
        threshold: float = 0.0,
        center: complex = 0.0,
        radius: float = 1.0,
        inclusive: bool = False,
        boundary_tolerance: float = 1e-8,
        expected_dimension: int | None = None,
        selection_id: str | None = None,
    ):
        if kind not in ("real-below", "real-above", "disk", "exterior-disk"):
            raise ValueError("Unknown spectral selection kind.")
        threshold_ = float(threshold)
        center_ = complex(center)
        radius_ = float(radius)
        boundary = float(boundary_tolerance)
        expected = None if expected_dimension is None else int(expected_dimension)
        if not math.isfinite(threshold_):
            raise ValueError("threshold must be finite.")
        if not math.isfinite(center_.real) or not math.isfinite(center_.imag):
            raise ValueError("center must be finite.")
        if not math.isfinite(radius_) or radius_ <= 0.0:
            raise ValueError("radius must be finite and positive.")
        if not math.isfinite(boundary) or boundary < 0.0:
            raise ValueError("boundary_tolerance must be finite and non-negative.")
        if expected is not None and expected < 1:
            raise ValueError("expected_dimension must be positive or None.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "spectral-selection",
                    "selection_kind": kind,
                    "threshold": threshold_,
                    "center": [center_.real, center_.imag],
                    "radius": radius_,
                    "inclusive": bool(inclusive),
                    "boundary_tolerance": boundary,
                    "expected_dimension": expected,
                }
            )
            if selection_id is None
            else str(selection_id)
        )
        if not identifier:
            raise ValueError("selection_id must be non-empty.")
        self.kind = kind
        self.threshold = threshold_
        self.center = center_
        self.radius = radius_
        self.inclusive = bool(inclusive)
        self.boundary_tolerance = boundary
        self.expected_dimension = expected
        self.selection_id = identifier

    @classmethod
    def real_below(cls, threshold: float = 0.0, /, **kwargs) -> "SpectralSelection":
        return cls("real-below", threshold=threshold, **kwargs)

    @classmethod
    def real_above(cls, threshold: float = 0.0, /, **kwargs) -> "SpectralSelection":
        return cls("real-above", threshold=threshold, **kwargs)

    @classmethod
    def disk(
        cls,
        center: complex,
        radius: float,
        /,
        **kwargs,
    ) -> "SpectralSelection":
        return cls("disk", center=center, radius=radius, **kwargs)

    @classmethod
    def exterior_disk(
        cls,
        center: complex,
        radius: float,
        /,
        **kwargs,
    ) -> "SpectralSelection":
        return cls("exterior-disk", center=center, radius=radius, **kwargs)

    def mask(self, eigenvalues: ArrayLike, /) -> Array:
        values = jnp.asarray(eigenvalues)
        signed = self._signed_distance(values)
        return signed >= 0 if self.inclusive else signed > 0

    def boundary_distance(self, eigenvalues: ArrayLike, /) -> Array:
        return jnp.abs(self._signed_distance(jnp.asarray(eigenvalues)))

    def _signed_distance(self, values: Array, /) -> Array:
        if self.kind == "real-below":
            return self.threshold - jnp.real(values)
        if self.kind == "real-above":
            return jnp.real(values) - self.threshold
        radial = jnp.abs(values - self.center)
        if self.kind == "disk":
            return self.radius - radial
        return radial - self.radius

    def _matches_scalar(self, value: complex, /) -> bool:
        if self.kind == "real-below":
            signed = self.threshold - value.real
        elif self.kind == "real-above":
            signed = value.real - self.threshold
        elif self.kind == "disk":
            signed = self.radius - abs(value - self.center)
        else:
            signed = abs(value - self.center) - self.radius
        return signed >= 0 if self.inclusive else signed > 0


class SpectralSubspaceResourcePolicy(StrictModule):
    """Hard dimension, retained-state, workspace, and exact-separation limits."""

    max_dimension: int = eqx.field(static=True)
    max_retained_bytes: int = eqx.field(static=True)
    max_workspace_bytes: int = eqx.field(static=True)
    max_separation_entries: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_dimension: int = 2048,
        max_retained_bytes: int = 512 * 1024 * 1024,
        max_workspace_bytes: int = 1024 * 1024 * 1024,
        max_separation_entries: int = 1_000_000,
    ):
        values = tuple(
            int(value)
            for value in (
                max_dimension,
                max_retained_bytes,
                max_workspace_bytes,
                max_separation_entries,
            )
        )
        if values[0] < 1 or any(value < 0 for value in values[1:]):
            raise ValueError(
                "max_dimension must be positive and byte/entry limits non-negative."
            )
        (
            self.max_dimension,
            self.max_retained_bytes,
            self.max_workspace_bytes,
            self.max_separation_entries,
        ) = values


class SpectralSubspacePolicy(StrictModule):
    """Ordered-Schur materialization, verification, conditioning, and failure policy."""

    materialization: MaterializationPolicy = eqx.field(static=True)
    resources: SpectralSubspaceResourcePolicy = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    max_projector_norm: float | None = eqx.field(static=True)
    minimum_eigenvalue_gap: float = eqx.field(static=True)
    require_exact_separation: bool = eqx.field(static=True)
    failure: FailurePolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        materialization: MaterializationPolicy | None = None,
        resources: SpectralSubspaceResourcePolicy | None = None,
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-10,
        max_projector_norm: float | None = None,
        minimum_eigenvalue_gap: float = 0.0,
        require_exact_separation: bool = False,
        failure: FailurePolicy | None = None,
    ):
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        resources_ = SpectralSubspaceResourcePolicy() if resources is None else resources
        failure_ = FailurePolicy() if failure is None else failure
        if not isinstance(materialization_, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy or None.")
        if not isinstance(resources_, SpectralSubspaceResourcePolicy):
            raise TypeError("resources must be a SpectralSubspaceResourcePolicy or None.")
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy or None.")
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        gap = float(minimum_eigenvalue_gap)
        limit = None if max_projector_norm is None else float(max_projector_norm)
        if any(
            not math.isfinite(value) or value < 0.0 for value in (relative, absolute, gap)
        ):
            raise ValueError(
                "Spectral-subspace tolerances must be finite and non-negative."
            )
        if limit is not None and (not math.isfinite(limit) or limit < 1.0):
            raise ValueError(
                "max_projector_norm must be finite and at least one or None."
            )
        self.materialization = materialization_
        self.resources = resources_
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.max_projector_norm = limit
        self.minimum_eigenvalue_gap = gap
        self.require_exact_separation = bool(require_exact_separation)
        self.failure = failure_


class SpectralSubspaceCostEstimate(StrictModule):
    """Dense ordered-Schur retained state and conservative workspace estimate."""

    dimension: int = eqx.field(static=True)
    input_matrix_bytes: int = eqx.field(static=True)
    retained_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    selected_dimension_known: bool = eqx.field(static=True)
    exact: bool = eqx.field(static=True)


class SpectralSubspacePlan(StrictModule):
    """Immutable symbolic ordered-Schur subspace plan."""

    selection: SpectralSelection = eqx.field(static=True)
    policy: SpectralSubspacePolicy = eqx.field(static=True)
    cost: SpectralSubspaceCostEstimate = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class SpectralSubspaceDiagnostics(StrictModule):
    """Isolation, invariance, projector, and biorthogonality evidence."""

    invariance_residual_norm: Array
    commutator_residual_norm: Array
    idempotence_error: Array
    biorthogonality_error: Array
    orthonormal_basis_error: Array
    projector_norm: Array
    eigenvalue_gap: Array
    sylvester_separation: Array
    boundary_distance: Array
    finite: Array
    converged: Array
    separation_exact: bool = eqx.field(static=True)
    retained_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)


class PreparedSpectralSubspace(StrictModule):
    """Ordered Schur factors and block diagonalization for one isolated cluster."""

    problem: SchurEigenproblem
    matrix: Array
    schur_form: Array
    schur_vectors: Array
    right_transform: Array
    left_transform: Array
    projector: Array
    orthogonal_projector: Array
    selected_eigenvalues: Array
    complement_eigenvalues: Array
    selected_schur_form: Array
    complement_schur_form: Array
    coupling_solution: Array
    status: Array
    diagnostics: SpectralSubspaceDiagnostics
    numeric_version: Array
    refresh_count: Array
    plan: SpectralSubspacePlan = eqx.field(static=True)
    selected_dimension: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_fingerprint: str = eqx.field(static=True)


class SpectralSubspaceProvenance(StrictModule):
    """Ordered-Schur identities, convention, and numerical version."""

    backend: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)
    projector_kind: str = eqx.field(static=True)
    numeric_version: Array


class SpectralSubspace(StrictModule):
    """Selected invariant bases and orthogonal and Riesz projectors."""

    selected_eigenvalues: Array
    complement_eigenvalues: Array
    basis: Array
    left_dual_basis: Array
    invariant_complement_basis: Array
    selected_schur_form: Array
    complement_schur_form: Array
    projector: Array
    orthogonal_projector: Array
    status: Array
    diagnostics: SpectralSubspaceDiagnostics
    provenance: SpectralSubspaceProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(SpectralSubspaceStatus.SUCCESS)

    def project_coordinates(self, vector: ArrayLike, /) -> Array:
        value = jnp.asarray(vector)
        if value.shape != (self.projector.shape[1],):
            raise ValueError("vector must match the projector coordinate dimension.")
        return self.projector @ value


class SpectralProjectorDerivativeDiagnostics(StrictModule):
    """Block Sylvester and differentiated projector-identity residuals."""

    upper_sylvester_residual_norm: Array
    lower_sylvester_residual_norm: Array
    commutator_residual_norm: Array
    tangent_residual_norm: Array
    relative_residual: Array
    perturbation_norm: Array
    derivative_norm: Array
    finite: Array
    converged: Array


class SpectralProjectorDerivativeProvenance(StrictModule):
    """Source subspace identities and derivative method."""

    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    numeric_version: Array


class SpectralProjectorDerivativeResult(StrictModule):
    """Fréchet derivative ``D P(A)[E]`` of an isolated Riesz projector."""

    value: Array
    status: Array
    diagnostics: SpectralProjectorDerivativeDiagnostics
    provenance: SpectralProjectorDerivativeProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(SpectralProjectorDerivativeStatus.SUCCESS)


def plan_spectral_subspace(
    problem: SchurEigenproblem,
    selection: SpectralSelection,
    policy: SpectralSubspacePolicy | None = None,
    /,
) -> SpectralSubspacePlan:
    """Plan a host ordered-Schur decomposition for an isolated spectral cluster."""
    if not isinstance(problem, SchurEigenproblem):
        raise TypeError("problem must be a SchurEigenproblem.")
    if not isinstance(selection, SpectralSelection):
        raise TypeError("selection must be a SpectralSelection.")
    selected = SpectralSubspacePolicy() if policy is None else policy
    if not isinstance(selected, SpectralSubspacePolicy):
        raise TypeError("policy must be a SpectralSubspacePolicy or None.")
    cost = _subspace_cost(problem)
    resources = selected.resources
    if cost.dimension > resources.max_dimension:
        raise ValueError(
            f"Spectral subspace dimension {cost.dimension} exceeds limit "
            f"{resources.max_dimension}."
        )
    if cost.retained_bytes > resources.max_retained_bytes:
        raise ValueError(
            f"Spectral subspace retained estimate {cost.retained_bytes} exceeds "
            f"limit {resources.max_retained_bytes}."
        )
    if cost.workspace_bytes > resources.max_workspace_bytes:
        raise ValueError(
            f"Spectral subspace workspace estimate {cost.workspace_bytes} exceeds "
            f"limit {resources.max_workspace_bytes}."
        )
    return SpectralSubspacePlan(
        selection=selection,
        policy=selected,
        cost=cost,
        problem_id=problem.problem_id,
        operator_id=problem.operator.operator_id,
        plan_id=canonical_fingerprint(
            {
                "kind": "spectral-subspace-plan",
                "problem": problem.problem_id,
                "operator": problem.operator.operator_id,
                "selection": selection.selection_id,
                "relative_tolerance": selected.relative_tolerance,
                "absolute_tolerance": selected.absolute_tolerance,
            }
        ),
    )


def prepare_spectral_subspace(
    problem: SchurEigenproblem,
    selection_or_plan: SpectralSelection | SpectralSubspacePlan,
    policy: SpectralSubspacePolicy | None = None,
    /,
) -> PreparedSpectralSubspace:
    """Materialize, order the Schur form, and construct the Riesz projector."""
    if isinstance(selection_or_plan, SpectralSubspacePlan):
        if policy is not None:
            raise ValueError("policy must be omitted when a subspace plan is supplied.")
        plan = selection_or_plan
    else:
        plan = plan_spectral_subspace(problem, selection_or_plan, policy)
    _validate_plan(problem, plan)
    return _prepare_numeric(problem, plan, numeric_version=0, refresh_count=0)


def refresh_spectral_subspace(
    prepared: PreparedSpectralSubspace,
    problem: SchurEigenproblem,
    /,
) -> PreparedSpectralSubspace:
    """Refresh an isolated cluster while forbidding dimension-changing crossings."""
    if not isinstance(prepared, PreparedSpectralSubspace):
        raise TypeError("prepared must be a PreparedSpectralSubspace.")
    _validate_plan(problem, prepared.plan)
    refreshed = _prepare_numeric(
        problem,
        prepared.plan,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        refresh_count=prepared.refresh_count + jnp.asarray(1, dtype=jnp.int32),
        prepared_id=prepared.prepared_id,
    )
    if refreshed.selected_dimension != prepared.selected_dimension:
        raise ValueError(
            "Spectral refresh changed the selected dimension; a new plan is required."
        )
    return refreshed


def spectral_subspace(
    problem_or_prepared: SchurEigenproblem | PreparedSpectralSubspace,
    selection: SpectralSelection | None = None,
    /,
    *,
    policy: SpectralSubspacePolicy | None = None,
) -> SpectralSubspace:
    """Return selected invariant bases and the Riesz and orthogonal projectors."""
    if isinstance(problem_or_prepared, PreparedSpectralSubspace):
        if selection is not None or policy is not None:
            raise ValueError("selection and policy must be omitted for prepared state.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, SchurEigenproblem):
        if selection is None:
            raise ValueError("selection is required for an unprepared problem.")
        prepared = prepare_spectral_subspace(problem_or_prepared, selection, policy)
    else:
        raise TypeError("Expected a SchurEigenproblem or PreparedSpectralSubspace.")
    count = prepared.selected_dimension
    basis = prepared.schur_vectors[:, :count]
    left_dual_basis = jnp.conj(prepared.left_transform[:count, :].T)
    invariant_complement = prepared.right_transform[:, count:]
    return SpectralSubspace(
        selected_eigenvalues=prepared.selected_eigenvalues,
        complement_eigenvalues=prepared.complement_eigenvalues,
        basis=basis,
        left_dual_basis=left_dual_basis,
        invariant_complement_basis=invariant_complement,
        selected_schur_form=prepared.selected_schur_form,
        complement_schur_form=prepared.complement_schur_form,
        projector=prepared.projector,
        orthogonal_projector=prepared.orthogonal_projector,
        status=prepared.status,
        diagnostics=prepared.diagnostics,
        provenance=SpectralSubspaceProvenance(
            backend="scipy-ordered-complex-schur",
            problem_id=prepared.problem.problem_id,
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=prepared.problem.operator.operator_id,
            selection_id=prepared.plan.selection.selection_id,
            projector_kind="Riesz oblique spectral projector",
            numeric_version=prepared.numeric_version,
        ),
    )


def spectral_projector_derivative(
    prepared: PreparedSpectralSubspace,
    perturbation: AbstractLinearOperator | ArrayLike,
    /,
) -> SpectralProjectorDerivativeResult:
    """Evaluate the local Fréchet derivative of the isolated Riesz projector."""
    if not isinstance(prepared, PreparedSpectralSubspace):
        raise TypeError("prepared must be a PreparedSpectralSubspace.")
    perturbation_matrix = _perturbation_matrix(prepared, perturbation)
    derivative, upper_residual, lower_residual = _projector_derivative_value(
        prepared,
        perturbation_matrix,
    )
    projector = prepared.projector
    matrix = prepared.matrix
    commutator_target = projector @ perturbation_matrix - perturbation_matrix @ projector
    commutator_residual_matrix = (
        matrix @ derivative - derivative @ matrix - commutator_target
    )
    tangent_residual_matrix = projector @ derivative + derivative @ projector - derivative
    commutator_residual = jnp.linalg.norm(commutator_residual_matrix)
    tangent_residual = jnp.linalg.norm(tangent_residual_matrix)
    perturbation_norm = jnp.linalg.norm(perturbation_matrix)
    derivative_norm = jnp.linalg.norm(derivative)
    scale = (
        jnp.linalg.norm(matrix) * derivative_norm
        + jnp.linalg.norm(projector) * perturbation_norm
    )
    tiny = jnp.asarray(jnp.finfo(derivative.real.dtype).tiny)
    relative = (commutator_residual + tangent_residual) / jnp.maximum(scale, tiny)
    tolerance = (
        prepared.plan.policy.absolute_tolerance
        + prepared.plan.policy.relative_tolerance * scale
    )
    finite = (
        jnp.all(jnp.isfinite(derivative))
        & jnp.isfinite(commutator_residual)
        & jnp.isfinite(tangent_residual)
    )
    source_success = prepared.status == int(SpectralSubspaceStatus.SUCCESS)
    residual_ok = (commutator_residual + tangent_residual) <= tolerance
    converged = finite & source_success & residual_ok
    status = jnp.where(
        ~finite,
        int(SpectralProjectorDerivativeStatus.NONFINITE),
        jnp.where(
            ~source_success,
            int(SpectralProjectorDerivativeStatus.SOURCE_FAILURE),
            jnp.where(
                ~residual_ok,
                int(SpectralProjectorDerivativeStatus.RESIDUAL_TOO_LARGE),
                int(SpectralProjectorDerivativeStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    if prepared.plan.policy.failure.mode == "error":
        derivative = eqx.error_if(
            derivative,
            status != int(SpectralProjectorDerivativeStatus.SUCCESS),
            "Spectral projector derivative did not satisfy its numerical contract.",
        )
    return SpectralProjectorDerivativeResult(
        value=derivative,
        status=status,
        diagnostics=SpectralProjectorDerivativeDiagnostics(
            upper_sylvester_residual_norm=upper_residual,
            lower_sylvester_residual_norm=lower_residual,
            commutator_residual_norm=commutator_residual,
            tangent_residual_norm=tangent_residual,
            relative_residual=relative,
            perturbation_norm=perturbation_norm,
            derivative_norm=derivative_norm,
            finite=finite,
            converged=converged,
        ),
        provenance=SpectralProjectorDerivativeProvenance(
            problem_id=prepared.problem.problem_id,
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=prepared.problem.operator.operator_id,
            method="ordered-Schur block triangular Sylvester derivative",
            numeric_version=prepared.numeric_version,
        ),
    )


def _prepare_numeric(
    problem: SchurEigenproblem,
    plan: SpectralSubspacePlan,
    *,
    numeric_version: Any,
    refresh_count: Any,
    prepared_id: str | None = None,
) -> PreparedSpectralSubspace:
    matrix = jnp.asarray(materialize(problem.operator, plan.policy.materialization))
    matrix_numpy = np.asarray(matrix)
    schur_form_numpy, schur_vectors_numpy, selected_count = scipy_linalg.schur(
        matrix_numpy,
        output="complex",
        sort=plan.selection._matches_scalar,
    )
    dimension = problem.operator.source.size
    selected_count = int(selected_count)
    if selected_count < 1 or selected_count >= dimension:
        raise ValueError("Spectral selection must define a nonempty proper subspace.")
    if (
        plan.selection.expected_dimension is not None
        and selected_count != plan.selection.expected_dimension
    ):
        raise ValueError(
            f"Selected dimension {selected_count} does not match expected dimension "
            f"{plan.selection.expected_dimension}."
        )
    eigenvalues_numpy = np.diag(schur_form_numpy)
    boundary_distance = float(
        np.min(np.asarray(plan.selection.boundary_distance(eigenvalues_numpy)))
    )
    if boundary_distance <= plan.selection.boundary_tolerance:
        raise ValueError(
            "An eigenvalue lies within the protected spectral-selection boundary."
        )
    selected_form_numpy = schur_form_numpy[:selected_count, :selected_count]
    complement_form_numpy = schur_form_numpy[selected_count:, selected_count:]
    coupling_numpy = schur_form_numpy[:selected_count, selected_count:]
    coupling_solution_numpy = scipy_linalg.solve_sylvester(
        selected_form_numpy,
        -complement_form_numpy,
        coupling_numpy,
    )
    identity = np.eye(dimension, dtype=schur_form_numpy.dtype)
    transform = identity.copy()
    transform[:selected_count, selected_count:] = -coupling_solution_numpy
    inverse_transform = identity.copy()
    inverse_transform[:selected_count, selected_count:] = coupling_solution_numpy
    right_transform_numpy = schur_vectors_numpy @ transform
    left_transform_numpy = inverse_transform @ np.conj(schur_vectors_numpy.T)
    projector_block = np.zeros_like(schur_form_numpy)
    projector_block[:selected_count, :selected_count] = np.eye(
        selected_count,
        dtype=schur_form_numpy.dtype,
    )
    projector_block[:selected_count, selected_count:] = coupling_solution_numpy
    projector_numpy = (
        schur_vectors_numpy @ projector_block @ np.conj(schur_vectors_numpy.T)
    )
    basis_numpy = schur_vectors_numpy[:, :selected_count]
    orthogonal_projector_numpy = basis_numpy @ np.conj(basis_numpy.T)
    exact_separation, separation_exact = _sylvester_separation(
        selected_form_numpy,
        complement_form_numpy,
        plan.policy.resources.max_separation_entries,
    )
    selected_eigenvalues_numpy = eigenvalues_numpy[:selected_count]
    complement_eigenvalues_numpy = eigenvalues_numpy[selected_count:]
    eigenvalue_gap = float(
        np.min(
            np.abs(
                selected_eigenvalues_numpy[:, None]
                - complement_eigenvalues_numpy[None, :]
            )
        )
    )
    if plan.policy.require_exact_separation and not separation_exact:
        raise ValueError("Exact Sylvester separation exceeds max_separation_entries.")
    matrix_complex = jnp.asarray(matrix_numpy, dtype=schur_form_numpy.dtype)
    schur_form = jnp.asarray(schur_form_numpy)
    schur_vectors = jnp.asarray(schur_vectors_numpy)
    right_transform = jnp.asarray(right_transform_numpy)
    left_transform = jnp.asarray(left_transform_numpy)
    projector = jnp.asarray(projector_numpy)
    orthogonal_projector = jnp.asarray(orthogonal_projector_numpy)
    selected_form = jnp.asarray(selected_form_numpy)
    complement_form = jnp.asarray(complement_form_numpy)
    coupling_solution = jnp.asarray(coupling_solution_numpy)
    selected_eigenvalues = jnp.asarray(selected_eigenvalues_numpy)
    complement_eigenvalues = jnp.asarray(complement_eigenvalues_numpy)
    invariance = jnp.linalg.norm(
        matrix_complex @ basis_numpy - basis_numpy @ selected_form_numpy
    )
    commutator = jnp.linalg.norm(matrix_complex @ projector - projector @ matrix_complex)
    idempotence = jnp.linalg.norm(projector @ projector - projector)
    biorthogonality = jnp.linalg.norm(
        left_transform @ right_transform - jnp.eye(dimension, dtype=projector.dtype)
    )
    orthonormality = jnp.linalg.norm(
        jnp.conj(basis_numpy.T) @ basis_numpy
        - jnp.eye(selected_count, dtype=projector.dtype)
    )
    projector_norm = jnp.linalg.norm(projector, ord=2)
    finite = all(
        bool(np.all(np.isfinite(value)))
        for value in (
            schur_form_numpy,
            schur_vectors_numpy,
            right_transform_numpy,
            left_transform_numpy,
            projector_numpy,
        )
    )
    scale = jnp.linalg.norm(matrix_complex) * jnp.maximum(projector_norm, 1)
    residual_tolerance = (
        plan.policy.absolute_tolerance + plan.policy.relative_tolerance * scale
    )
    residual_ok = (
        invariance + commutator + idempotence + biorthogonality + orthonormality
        <= residual_tolerance
    )
    gap_ok = eigenvalue_gap >= plan.policy.minimum_eigenvalue_gap
    norm_ok = (
        True
        if plan.policy.max_projector_norm is None
        else projector_norm <= plan.policy.max_projector_norm
    )
    condition_ok = gap_ok & norm_ok
    status = jnp.asarray(
        int(SpectralSubspaceStatus.SUCCESS)
        if finite and bool(residual_ok) and bool(condition_ok)
        else (
            int(SpectralSubspaceStatus.NONFINITE)
            if not finite
            else (
                int(SpectralSubspaceStatus.PROJECTOR_RESIDUAL_TOO_LARGE)
                if not bool(residual_ok)
                else int(SpectralSubspaceStatus.ILL_CONDITIONED)
            )
        ),
        dtype=jnp.int32,
    )
    if plan.policy.failure.mode == "error" and int(status) != int(
        SpectralSubspaceStatus.SUCCESS
    ):
        raise ValueError("Spectral subspace did not satisfy its numerical contract.")
    diagnostics = SpectralSubspaceDiagnostics(
        invariance_residual_norm=invariance,
        commutator_residual_norm=commutator,
        idempotence_error=idempotence,
        biorthogonality_error=biorthogonality,
        orthonormal_basis_error=orthonormality,
        projector_norm=projector_norm,
        eigenvalue_gap=jnp.asarray(eigenvalue_gap),
        sylvester_separation=jnp.asarray(exact_separation),
        boundary_distance=jnp.asarray(boundary_distance),
        finite=jnp.asarray(finite),
        converged=status == int(SpectralSubspaceStatus.SUCCESS),
        separation_exact=separation_exact,
        retained_bytes=plan.cost.retained_bytes,
        workspace_bytes=plan.cost.workspace_bytes,
    )
    operator_fingerprint = canonical_fingerprint(array_tree_fingerprint(problem.operator))
    identifier = (
        canonical_fingerprint(
            {
                "kind": "prepared-spectral-subspace",
                "plan": plan.plan_id,
                "operator": operator_fingerprint,
                "selected_dimension": selected_count,
            }
        )
        if prepared_id is None
        else prepared_id
    )
    return PreparedSpectralSubspace(
        problem=problem,
        matrix=matrix_complex,
        schur_form=schur_form,
        schur_vectors=schur_vectors,
        right_transform=right_transform,
        left_transform=left_transform,
        projector=projector,
        orthogonal_projector=orthogonal_projector,
        selected_eigenvalues=selected_eigenvalues,
        complement_eigenvalues=complement_eigenvalues,
        selected_schur_form=selected_form,
        complement_schur_form=complement_form,
        coupling_solution=coupling_solution,
        status=status,
        diagnostics=diagnostics,
        numeric_version=jnp.asarray(numeric_version, dtype=jnp.int32),
        refresh_count=jnp.asarray(refresh_count, dtype=jnp.int32),
        plan=plan,
        selected_dimension=selected_count,
        prepared_id=identifier,
        operator_fingerprint=operator_fingerprint,
    )


def _projector_derivative_value(
    prepared: PreparedSpectralSubspace,
    perturbation: Array,
    /,
) -> tuple[Array, Array, Array]:
    count = prepared.selected_dimension
    selected = prepared.selected_schur_form
    complement = prepared.complement_schur_form
    transformed = prepared.left_transform @ perturbation @ prepared.right_transform
    upper_forcing = transformed[:count, count:]
    lower_forcing = transformed[count:, :count]
    upper = _solve_triangular_sylvester(selected, complement, upper_forcing)
    lower = _solve_triangular_sylvester(complement, selected, -lower_forcing)
    block_derivative = jnp.zeros_like(prepared.schur_form)
    block_derivative = block_derivative.at[:count, count:].set(upper)
    block_derivative = block_derivative.at[count:, :count].set(lower)
    derivative = prepared.right_transform @ block_derivative @ prepared.left_transform
    upper_residual = jnp.linalg.norm(
        selected @ upper - upper @ complement - upper_forcing
    )
    lower_residual = jnp.linalg.norm(
        complement @ lower - lower @ selected + lower_forcing
    )
    return derivative, upper_residual, lower_residual


def _solve_triangular_sylvester(
    left: Array,
    right: Array,
    forcing: Array,
    /,
) -> Array:
    columns = right.shape[0]
    identity = jnp.eye(left.shape[0], dtype=left.dtype)
    indices = jnp.arange(columns)

    def body(index, solution):
        previous = jnp.where(indices < index, right[:, index], 0)
        right_hand_side = forcing[:, index] + solution @ previous
        column = jsp.linalg.solve_triangular(
            left - right[index, index] * identity,
            right_hand_side,
            lower=False,
        )
        return solution.at[:, index].set(column)

    initial = jnp.zeros_like(forcing)
    return jax.lax.fori_loop(0, columns, body, initial)


def _perturbation_matrix(
    prepared: PreparedSpectralSubspace,
    perturbation: AbstractLinearOperator | ArrayLike,
    /,
) -> Array:
    if isinstance(perturbation, AbstractLinearOperator):
        if (
            perturbation.batch_shape
            or not perturbation.source.compatible(prepared.problem.operator.source)
            or not perturbation.target.compatible(prepared.problem.operator.target)
        ):
            raise ValueError("Perturbation operator must match the spectral problem.")
        value = materialize(perturbation, prepared.plan.policy.materialization)
    else:
        value = jnp.asarray(perturbation)
    expected = prepared.matrix.shape
    if value.shape != expected:
        raise ValueError(f"perturbation must have shape {expected}; got {value.shape}.")
    return jnp.asarray(value, dtype=prepared.matrix.dtype)


def _sylvester_separation(
    selected: np.ndarray,
    complement: np.ndarray,
    max_entries: int,
    /,
) -> tuple[float, bool]:
    rows = selected.shape[0] * complement.shape[0]
    entries = rows * rows
    if entries > max_entries:
        return math.nan, False
    operator = np.kron(selected, np.eye(complement.shape[0])) - np.kron(
        np.eye(selected.shape[0]),
        complement.T,
    )
    singular_values = np.linalg.svd(operator, compute_uv=False)
    return float(singular_values[-1]), True


def _subspace_cost(problem: SchurEigenproblem, /) -> SpectralSubspaceCostEstimate:
    dimension = problem.operator.source.size
    input_itemsize = _coordinate_dtype(problem.operator.source).itemsize
    output_itemsize = jnp.result_type(
        _coordinate_dtype(problem.operator.source), jnp.complex64
    ).itemsize
    entries = dimension * dimension
    input_bytes = entries * input_itemsize
    retained = input_bytes + 7 * entries * output_itemsize
    workspace = 10 * entries * output_itemsize
    return SpectralSubspaceCostEstimate(
        dimension=dimension,
        input_matrix_bytes=input_bytes,
        retained_bytes=retained,
        workspace_bytes=workspace,
        selected_dimension_known=False,
        exact=False,
    )


def _validate_plan(problem: SchurEigenproblem, plan: SpectralSubspacePlan, /) -> None:
    if not isinstance(problem, SchurEigenproblem):
        raise TypeError("problem must be a SchurEigenproblem.")
    if not isinstance(plan, SpectralSubspacePlan):
        raise TypeError("plan must be a SpectralSubspacePlan.")
    if (
        problem.problem_id != plan.problem_id
        or problem.operator.operator_id != plan.operator_id
    ):
        raise ValueError(
            "Spectral-subspace plan belongs to a different symbolic problem."
        )


__all__ = [
    "PreparedSpectralSubspace",
    "SpectralProjectorDerivativeDiagnostics",
    "SpectralProjectorDerivativeProvenance",
    "SpectralProjectorDerivativeResult",
    "SpectralProjectorDerivativeStatus",
    "SpectralSelection",
    "SpectralSelectionKind",
    "SpectralSubspace",
    "SpectralSubspaceCostEstimate",
    "SpectralSubspaceDiagnostics",
    "SpectralSubspacePlan",
    "SpectralSubspacePolicy",
    "SpectralSubspaceProvenance",
    "SpectralSubspaceResourcePolicy",
    "SpectralSubspaceStatus",
    "plan_spectral_subspace",
    "prepare_spectral_subspace",
    "refresh_spectral_subspace",
    "spectral_projector_derivative",
    "spectral_subspace",
]
