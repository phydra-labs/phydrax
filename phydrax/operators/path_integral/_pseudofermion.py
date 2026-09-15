#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg._certificates import SpectralInterval
from ...linalg._operators import _materialize_by_basis, AbstractLinearOperator
from ...linalg._policies import FailurePolicy
from ...linalg._properties import OperatorCapabilities, OperatorProperties
from ...linalg._rational_functions import (
    PartialFractionRationalFunction,
    rational_function_action,
    RationalFunctionPolicy,
)
from ...linalg._shifted import (
    ShiftedLinearSystemFamily,
    ShiftedSolvePolicy,
    ShiftedSolveStatus,
    solve_shifted,
)
from ._lattice_fermion import AbstractLatticeDiracOperator
from ._rational_approximation import (
    CertifiedRationalApproximation,
    RationalApproximationTarget,
)


PseudofermionSolveRole: TypeAlias = Literal["action", "force", "acceptance"]


class _DiracNormalOperator(AbstractLinearOperator):
    """Matrix-free ``D†D`` retaining the differentiable lattice Dirac value."""

    dirac: AbstractLatticeDiracOperator

    def __init__(self, dirac: AbstractLatticeDiracOperator, /):
        if not isinstance(dirac, AbstractLatticeDiracOperator):
            raise TypeError("dirac must implement AbstractLatticeDiracOperator.")
        self.dirac = dirac
        self.source = dirac.source
        self.target = dirac.source
        self.properties = OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        )
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "lattice-dirac-normal-operator",
                "dirac": dirac.operator_id,
                "source": dirac.source.space_id,
                "target": dirac.target.space_id,
            }
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.dirac.adjoint_mv(self.dirac.mv(vector))

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        conjugated = jax.tree.map(jnp.conj, vector)
        return jax.tree.map(jnp.conj, self.mv(conjugated))

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.mv(vector)

    def _materialize(self, /) -> Array:
        return _materialize_by_basis(self)


def dirac_normal_operator(
    dirac: AbstractLatticeDiracOperator,
    /,
) -> AbstractLinearOperator:
    """Construct the canonical matrix-free normal operator for one Dirac value."""
    return _DiracNormalOperator(dirac)


class PseudofermionSolveRoles(StrictModule):
    """Distinct molecular-dynamics and exact-acceptance solve policies."""

    action: RationalFunctionPolicy = eqx.field(static=True)
    force: RationalFunctionPolicy = eqx.field(static=True)
    acceptance: RationalFunctionPolicy = eqx.field(static=True)
    roles_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        action: RationalFunctionPolicy | None = None,
        force: RationalFunctionPolicy | None = None,
        acceptance: RationalFunctionPolicy | None = None,
    ):
        action_ = _default_policy(1.0e-5, 1.0e-7) if action is None else action
        force_ = _default_policy(1.0e-4, 1.0e-6) if force is None else force
        acceptance_ = (
            _default_policy(5.0e-6, 1.0e-7) if acceptance is None else acceptance
        )
        if any(
            not isinstance(value, RationalFunctionPolicy)
            for value in (action_, force_, acceptance_)
        ):
            raise TypeError(
                "Pseudofermion solve roles require RationalFunctionPolicy values."
            )
        self.action = action_
        self.force = force_
        self.acceptance = acceptance_
        self.roles_id = canonical_fingerprint(
            {
                "kind": "pseudofermion-solve-roles",
                "action": _policy_payload(action_),
                "force": _policy_payload(force_),
                "acceptance": _policy_payload(acceptance_),
            }
        )

    def select(self, role: PseudofermionSolveRole, /) -> RationalFunctionPolicy:
        if role == "action":
            return self.action
        if role == "force":
            return self.force
        if role == "acceptance":
            return self.acceptance
        raise ValueError("Unknown pseudofermion solve role.")


class PseudofermionRefreshResult(StrictModule):
    """Refreshed auxiliary field and the Gaussian identity it must preserve."""

    field: PyTree[Array]
    gaussian: PyTree[Array]
    gaussian_action: Array
    status: Array
    successful: Array
    term_id: str = eqx.field(static=True)


class PseudofermionActionResult(StrictModule):
    """One pseudofermion action with rational-solve evidence."""

    value: Array
    status: Array
    residual_indicator: Array
    successful: Array
    finite: Array
    role: PseudofermionSolveRole = eqx.field(static=True)
    term_id: str = eqx.field(static=True)


class PseudofermionForceResult(StrictModule):
    """Negative link gradient and the shifted-solve evidence used to form it."""

    force: Array
    shifted_status: Array
    residual_norm: Array
    successful: Array
    finite: Array
    term_id: str = eqx.field(static=True)


class TwoFlavorPseudofermionTerm(StrictModule):
    """Exact two-degenerate-flavor determinant represented by ``phi†M^-1phi``."""

    dirac: AbstractLatticeDiracOperator
    spectral_interval: SpectralInterval
    action_approximation: CertifiedRationalApproximation
    solves: PseudofermionSolveRoles = eqx.field(static=True)
    determinant_power: float = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        dirac: AbstractLatticeDiracOperator,
        spectral_interval: SpectralInterval,
        /,
        *,
        solves: PseudofermionSolveRoles | None = None,
    ):
        normal = _validate_dirac_interval(dirac, spectral_interval)
        solves_ = PseudofermionSolveRoles() if solves is None else solves
        if not isinstance(solves_, PseudofermionSolveRoles):
            raise TypeError("solves must be PseudofermionSolveRoles or None.")
        target = RationalApproximationTarget(((0.0, -1.0),))
        action = _exact_rational_certificate(
            target,
            spectral_interval,
            PartialFractionRationalFunction(
                jnp.asarray((0.0,), dtype=spectral_interval.lower.dtype),
                jnp.asarray((-1.0,), dtype=spectral_interval.lower.dtype),
                polynomial_coefficients=jnp.asarray(
                    (0.0,), dtype=spectral_interval.lower.dtype
                ),
            ),
            "two-flavor-inverse",
        )
        self.dirac = dirac
        self.spectral_interval = spectral_interval
        self.action_approximation = action
        self.solves = solves_
        self.determinant_power = 1.0
        self.term_id = canonical_fingerprint(
            {
                "kind": "two-flavor-pseudofermion-term",
                "normal_operator": normal.operator_id,
                "interval": spectral_interval.certificate_id,
                "solves": solves_.roles_id,
            }
        )


class HasenbuschRatioPseudofermionTerm(StrictModule):
    """Two-flavor determinant ratio ``det(M/(M+mu^2))``."""

    dirac: AbstractLatticeDiracOperator
    spectral_interval: SpectralInterval
    action_approximation: CertifiedRationalApproximation
    refresh_approximation: CertifiedRationalApproximation
    solves: PseudofermionSolveRoles = eqx.field(static=True)
    mass_shift: float = eqx.field(static=True)
    determinant_power: float = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        dirac: AbstractLatticeDiracOperator,
        spectral_interval: SpectralInterval,
        refresh_approximation: CertifiedRationalApproximation,
        /,
        *,
        mass_shift: float,
        solves: PseudofermionSolveRoles | None = None,
    ):
        normal = _validate_dirac_interval(dirac, spectral_interval)
        shift = float(mass_shift)
        if not math.isfinite(shift) or shift <= 0.0:
            raise ValueError("mass_shift must be finite and positive.")
        solves_ = PseudofermionSolveRoles() if solves is None else solves
        if not isinstance(solves_, PseudofermionSolveRoles):
            raise TypeError("solves must be PseudofermionSolveRoles or None.")
        refresh_target = RationalApproximationTarget(((0.0, 0.5), (shift, -0.5)))
        _validate_approximation(
            refresh_approximation,
            spectral_interval,
            refresh_target,
            "Hasenbusch refresh",
        )
        action_target = RationalApproximationTarget(((0.0, -1.0), (shift, 1.0)))
        action = _exact_rational_certificate(
            action_target,
            spectral_interval,
            PartialFractionRationalFunction(
                jnp.asarray((0.0,), dtype=spectral_interval.lower.dtype),
                jnp.asarray((-shift,), dtype=spectral_interval.lower.dtype),
                polynomial_coefficients=jnp.asarray(
                    (1.0,), dtype=spectral_interval.lower.dtype
                ),
            ),
            "hasenbusch-ratio-inverse",
        )
        self.dirac = dirac
        self.spectral_interval = spectral_interval
        self.action_approximation = action
        self.refresh_approximation = refresh_approximation
        self.solves = solves_
        self.mass_shift = shift
        self.determinant_power = 1.0
        self.term_id = canonical_fingerprint(
            {
                "kind": "hasenbusch-ratio-pseudofermion-term",
                "normal_operator": normal.operator_id,
                "interval": spectral_interval.certificate_id,
                "refresh": refresh_approximation.certificate_id,
                "mass_shift": shift,
                "solves": solves_.roles_id,
            }
        )


class FractionalPowerPseudofermionTerm(StrictModule):
    """Positive fractional determinant ``det(M)**power`` for RHMC."""

    dirac: AbstractLatticeDiracOperator
    spectral_interval: SpectralInterval
    action_approximation: CertifiedRationalApproximation
    refresh_approximation: CertifiedRationalApproximation
    solves: PseudofermionSolveRoles = eqx.field(static=True)
    determinant_power: float = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        dirac: AbstractLatticeDiracOperator,
        spectral_interval: SpectralInterval,
        action_approximation: CertifiedRationalApproximation,
        refresh_approximation: CertifiedRationalApproximation,
        /,
        *,
        determinant_power: float,
        solves: PseudofermionSolveRoles | None = None,
    ):
        normal = _validate_dirac_interval(dirac, spectral_interval)
        power = float(determinant_power)
        if not math.isfinite(power) or power <= 0.0:
            raise ValueError("determinant_power must be finite and positive.")
        solves_ = PseudofermionSolveRoles() if solves is None else solves
        if not isinstance(solves_, PseudofermionSolveRoles):
            raise TypeError("solves must be PseudofermionSolveRoles or None.")
        action_target = RationalApproximationTarget(((0.0, -power),))
        refresh_target = RationalApproximationTarget(((0.0, 0.5 * power),))
        _validate_approximation(
            action_approximation,
            spectral_interval,
            action_target,
            "fractional action",
        )
        _validate_approximation(
            refresh_approximation,
            spectral_interval,
            refresh_target,
            "fractional refresh",
        )
        self.dirac = dirac
        self.spectral_interval = spectral_interval
        self.action_approximation = action_approximation
        self.refresh_approximation = refresh_approximation
        self.solves = solves_
        self.determinant_power = power
        self.term_id = canonical_fingerprint(
            {
                "kind": "fractional-power-pseudofermion-term",
                "normal_operator": normal.operator_id,
                "interval": spectral_interval.certificate_id,
                "action": action_approximation.certificate_id,
                "refresh": refresh_approximation.certificate_id,
                "determinant_power": power,
                "solves": solves_.roles_id,
            }
        )


PseudofermionTerm: TypeAlias = (
    TwoFlavorPseudofermionTerm
    | HasenbuschRatioPseudofermionTerm
    | FractionalPowerPseudofermionTerm
)


def refresh_pseudofermion(
    term: PseudofermionTerm,
    key: Key[Array, ""],
    /,
    *,
    links: ArrayLike | None = None,
) -> PseudofermionRefreshResult:
    """Draw one Gaussian and transform it by the term's covariance root."""
    if not isinstance(
        term,
        (
            TwoFlavorPseudofermionTerm,
            HasenbuschRatioPseudofermionTerm,
            FractionalPowerPseudofermionTerm,
        ),
    ):
        raise TypeError("Unknown pseudofermion term.")
    dirac = term.dirac if links is None else term.dirac.with_links(jnp.asarray(links))
    if isinstance(term, TwoFlavorPseudofermionTerm):
        gaussian = _standard_action_gaussian(dirac.target, key)
        field = dirac.adjoint_mv(gaussian)
        gaussian_action = jnp.real(dirac.target.inner(gaussian, gaussian))
        finite = _tree_all_finite(field) & jnp.isfinite(gaussian_action)
        status = jnp.where(finite, 0, 2).astype(jnp.int32)
        return PseudofermionRefreshResult(
            field,
            gaussian,
            gaussian_action,
            status,
            finite,
            term.term_id,
        )
    approximation = term.refresh_approximation
    normal = _DiracNormalOperator(dirac)
    gaussian = _standard_action_gaussian(normal.source, key)
    result = rational_function_action(
        normal,
        gaussian,
        approximation.function,
        policy=term.solves.action,
    )
    gaussian_action = jnp.real(normal.source.inner(gaussian, gaussian))
    finite = _tree_all_finite(result.value) & jnp.isfinite(gaussian_action)
    successful = result.successful & finite
    return PseudofermionRefreshResult(
        result.value,
        gaussian,
        gaussian_action,
        result.status,
        successful,
        term.term_id,
    )


def evaluate_pseudofermion_action(
    term: PseudofermionTerm,
    field: PyTree[Any],
    /,
    *,
    role: PseudofermionSolveRole = "action",
    links: ArrayLike | None = None,
) -> PseudofermionActionResult:
    """Evaluate a frozen pseudofermion using the policy assigned to ``role``."""
    if not isinstance(
        term,
        (
            TwoFlavorPseudofermionTerm,
            HasenbuschRatioPseudofermionTerm,
            FractionalPowerPseudofermionTerm,
        ),
    ):
        raise TypeError("Unknown pseudofermion term.")
    dirac = term.dirac if links is None else term.dirac.with_links(jnp.asarray(links))
    normal = _DiracNormalOperator(dirac)
    vector = normal.source.validate(field)
    result = rational_function_action(
        normal,
        vector,
        term.action_approximation.function,
        policy=term.solves.select(role),
    )
    value = jnp.real(normal.source.inner(vector, result.value))
    finite = jnp.isfinite(value) & _tree_all_finite(result.value)
    return PseudofermionActionResult(
        value=value,
        status=result.status,
        residual_indicator=result.diagnostics.residual_indicator,
        successful=result.successful & finite,
        finite=finite,
        role=role,
        term_id=term.term_id,
    )


def pseudofermion_force(
    term: PseudofermionTerm,
    field: PyTree[Any],
    links: ArrayLike,
    /,
) -> PseudofermionForceResult:
    """Return the exact partial-fraction link force with stopped solve vectors.

    For ``y_j = (p_j I - M)^-1 phi``, differentiation of the inverse gives
    ``d(phi† r(M) phi) = sum_j residue_j y_j† (dM) y_j``. The Krylov
    algorithm is therefore outside autodiff while the physical ``D†D`` action
    remains differentiated exactly.
    """
    if not isinstance(
        term,
        (
            TwoFlavorPseudofermionTerm,
            HasenbuschRatioPseudofermionTerm,
            FractionalPowerPseudofermionTerm,
        ),
    ):
        raise TypeError("Unknown pseudofermion term.")
    links_ = jnp.asarray(links)
    dirac = term.dirac.with_links(links_)
    normal = _DiracNormalOperator(dirac)
    vector = normal.source.validate(field)
    function = term.action_approximation.function
    if function.polynomial_degree != 0:
        raise ValueError("Pseudofermion force requires a constant polynomial part.")
    family = ShiftedLinearSystemFamily(normal, function.poles)
    shifted = solve_shifted(
        family,
        vector,
        policy=term.solves.force.shifted,
    )
    stopped = jax.tree.map(jax.lax.stop_gradient, shifted.value)
    active = jnp.abs(function.residues) > 0

    def differentiated_action(candidate):
        candidate_normal = _DiracNormalOperator(term.dirac.with_links(candidate))
        total = jnp.asarray(0.0, dtype=jnp.real(links_).dtype)
        for index in range(function.num_poles):
            solution = jax.tree.map(lambda leaf: leaf[index], stopped)
            normal_solution = candidate_normal.mv(solution)
            contribution = jnp.real(
                candidate_normal.source.inner(solution, normal_solution)
            )
            total = total + function.residues[index] * contribution
        return jnp.real(total)

    _, pullback = jax.vjp(differentiated_action, links_)
    force = -pullback(jnp.asarray(1.0, dtype=jnp.real(links_).dtype))[0]
    shifted_success = jnp.where(
        active,
        shifted.status == int(ShiftedSolveStatus.SUCCESS),
        True,
    )
    finite = _tree_all_finite(force) & jnp.all(
        jnp.where(active, jnp.isfinite(shifted.diagnostics.residual_norm), True)
    )
    return PseudofermionForceResult(
        force=force,
        shifted_status=shifted.status,
        residual_norm=shifted.diagnostics.residual_norm,
        successful=jnp.all(shifted_success) & finite,
        finite=finite,
        term_id=term.term_id,
    )


def _default_policy(relative: float, absolute: float, /) -> RationalFunctionPolicy:
    return RationalFunctionPolicy(
        shifted=ShiftedSolvePolicy(
            "lanczos",
            max_dimension=64,
            orthogonalization="double",
            relative_tolerance=relative,
            absolute_tolerance=absolute,
        ),
        failure=FailurePolicy("status"),
    )


def _policy_payload(policy: RationalFunctionPolicy, /) -> dict[str, Any]:
    shifted = policy.shifted
    return {
        "method": shifted.method,
        "max_dimension": shifted.max_dimension,
        "orthogonalization": shifted.orthogonalization,
        "breakdown_tolerance": shifted.breakdown_tolerance,
        "relative_tolerance": shifted.relative_tolerance,
        "absolute_tolerance": shifted.absolute_tolerance,
        "shifted_max_matvec_count": shifted.resources.max_matvec_count,
        "shifted_max_storage_bytes": shifted.resources.max_storage_bytes,
        "shifted_max_workspace_bytes": shifted.resources.max_workspace_bytes,
        "failure": policy.failure.mode,
        "max_matvec_count": policy.resources.max_matvec_count,
        "max_workspace_bytes": policy.resources.max_workspace_bytes,
    }


def _validate_dirac_interval(
    dirac: AbstractLatticeDiracOperator,
    interval: SpectralInterval,
    /,
) -> _DiracNormalOperator:
    if not isinstance(dirac, AbstractLatticeDiracOperator):
        raise TypeError("dirac must implement AbstractLatticeDiracOperator.")
    if not isinstance(interval, SpectralInterval):
        raise TypeError("spectral_interval must be a SpectralInterval.")
    if interval.scope != "structural":
        raise ValueError(
            "Production pseudofermions require a structural spectral interval "
            "that remains valid as gauge links evolve."
        )
    normal = _DiracNormalOperator(dirac)
    if not interval.matches(normal):
        raise ValueError(
            "The spectral interval does not certify this Dirac normal operator."
        )
    lower = jnp.asarray(interval.lower)
    if not isinstance(lower, jax.core.Tracer) and float(lower) <= 0.0:
        raise ValueError(
            "Pseudofermions require a strictly positive spectral lower bound."
        )
    return normal


def _validate_approximation(
    approximation: CertifiedRationalApproximation,
    interval: SpectralInterval,
    target: RationalApproximationTarget,
    name: str,
    /,
) -> None:
    if not isinstance(approximation, CertifiedRationalApproximation):
        raise TypeError(f"{name} must be a CertifiedRationalApproximation.")
    if approximation.spectral_interval.certificate_id != interval.certificate_id:
        raise ValueError(f"{name} uses a different spectral interval.")
    if approximation.target.target_id != target.target_id:
        raise ValueError(f"{name} approximates the wrong scalar function.")
    if not isinstance(approximation.successful, jax.core.Tracer) and not bool(
        approximation.successful
    ):
        raise ValueError(f"{name} did not satisfy its requested approximation contract.")


def _exact_rational_certificate(
    target: RationalApproximationTarget,
    interval: SpectralInterval,
    function: PartialFractionRationalFunction,
    construction: str,
    /,
) -> CertifiedRationalApproximation:
    endpoints = jnp.stack((interval.lower, interval.upper))
    zeros = jnp.zeros_like(endpoints)
    plan_id = canonical_fingerprint(
        {
            "kind": "algebraic-rational-construction",
            "construction": construction,
            "target": target.target_id,
            "interval": interval.certificate_id,
            "function": function.function_id,
        }
    )
    return CertifiedRationalApproximation(
        target,
        interval,
        function,
        endpoints,
        zeros,
        jnp.asarray(0.0, dtype=endpoints.dtype),
        jnp.asarray(0.0, dtype=endpoints.dtype),
        interval.lower,
        jnp.asarray(True),
        plan_id=plan_id,
        metric="relative",
        verification_points=2,
        evidence="algebraic identity on the positive spectral interval",
    )


def _standard_action_gaussian(space: Any, key: Key[Array, ""], /) -> PyTree[Array]:
    structure = space.structure()
    leaves, treedef = jax.tree.flatten(structure)
    keys = jr.split(key, len(leaves))
    samples = []
    for spec, leaf_key in zip(leaves, keys, strict=True):
        if jnp.issubdtype(spec.dtype, jnp.complexfloating):
            real_dtype = jnp.empty((), dtype=spec.dtype).real.dtype
            real_key, imaginary_key = jr.split(leaf_key)
            sample = (
                jr.normal(real_key, spec.shape, dtype=real_dtype)
                + 1j * jr.normal(imaginary_key, spec.shape, dtype=real_dtype)
            ) / jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype))
        elif jnp.issubdtype(spec.dtype, jnp.floating):
            sample = jr.normal(leaf_key, spec.shape, dtype=spec.dtype) / jnp.sqrt(
                jnp.asarray(2.0, dtype=spec.dtype)
            )
        else:
            raise TypeError(
                "Pseudofermion vector spaces require real or complex coordinates."
            )
        samples.append(sample)
    return jax.tree.unflatten(treedef, samples)


def _tree_all_finite(value: PyTree[Any], /) -> Array:
    leaves = jax.tree.leaves(value)
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)))


__all__ = [
    "FractionalPowerPseudofermionTerm",
    "HasenbuschRatioPseudofermionTerm",
    "PseudofermionActionResult",
    "PseudofermionRefreshResult",
    "PseudofermionForceResult",
    "PseudofermionSolveRole",
    "PseudofermionSolveRoles",
    "PseudofermionTerm",
    "TwoFlavorPseudofermionTerm",
    "dirac_normal_operator",
    "evaluate_pseudofermion_action",
    "pseudofermion_force",
    "refresh_pseudofermion",
]
