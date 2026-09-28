#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Implicit coupled radial accelerations of a bubble cloud.

Each bubble's composed radial equation is written in the per-unit-density form
`a_i R̈_i = f_i`, where `a_i = φ_i R_i` carries every hidden `R̈` coefficient of
the equation (`φ_i` is `RadialBubbleRates.inertia_fraction`). The incompressible
near field of the neighbour monopoles adds (Mettin et al. 1997, generalized to
every radial equation)

    a_i R̈_i + Σ_{j≠i} (R_j² R̈_j + 2 R_j Ṙ_j²)/d_ij = f_i.

With `y = R² R̈`, `e = 2 R Ṙ²`, `D = diag(a/R²)` and the Laplace pair kernel
`G_ij = 1/d_ij` (zero diagonal) the system is `(D + G) y = f − G e`. It is solved
in the symmetrically scaled form `(I + W) z = D^{-1/2}(f − G e)` with
`W = D^{-1/2} G D^{-1/2}` and `y = D^{-1/2} z`. For non-overlapping spheres and
`a = R` (Rayleigh–Plesset) `D + G` is the Coulomb energy matrix of uniformly
charged spherical shells and is positive definite; compressible inertia
fractions can lose definiteness, which is reported, never repaired.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    ArraySpace,
    ConjugateGradient,
    DenseCholesky,
    DenseLinearOperator,
    DensePropertyVerificationPolicy,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    refresh,
    solve,
    TolerancePolicy,
    verify_dense_properties,
)
from ..solver import (
    CartesianFMMResourceEvidence,
    PreparedUniformFMMStructure,
    UniformFMMPlan,
)


_SPD = OperatorProperties(
    self_adjoint=True,
    positive_definite=True,
    evidence={"self_adjoint": "construction", "positive_definite": "construction"},
)


class CloudCouplingResult(StrictModule):
    """Coupled accelerations and the neighbour monopole field at every bubble.

    `potential[i] = Σ_{j≠i} Q̇_j/d_ij` with `Q̇ = R²R̈ + 2RṘ²` (m² s⁻²); the
    neighbour pressure is `ρ·potential` and its gradient `ρ·gradient`.
    `relative_residual` is the native solve's relative residual of the scaled
    system and `iterations` its iteration count (zero for the dense route).
    """

    acceleration: Array
    potential: Array
    gradient: Array
    successful: Array
    iterations: Array
    relative_residual: Array


class CloudField(StrictModule):
    """Neighbour monopole potential and gradient of one strength vector."""

    potential: Array
    gradient: Array
    successful: Array


def _scaled_rhs(
    inertia: Array, forcing: Array, radius: Array, neighbor_explicit: Array, /
) -> tuple[Array, Array]:
    scale = radius / jnp.sqrt(inertia)
    return scale, scale * (forcing - neighbor_explicit)


class AbstractCloudCoupling(StrictModule):
    """Route owning the pair kernel `G` of one prepared cloud."""

    @abc.abstractmethod
    def field(self, position: Array, strength: Array, /) -> CloudField:
        """`Σ_{j≠i} q_j/d_ij` and its gradient at every bubble."""
        raise NotImplementedError

    @abc.abstractmethod
    def solve(
        self,
        position: Array,
        inertia: Array,
        forcing: Array,
        radius: Array,
        wall_velocity: Array,
        /,
    ) -> CloudCouplingResult:
        """Implicit coupled accelerations `R̈` of every bubble."""
        raise NotImplementedError


def _dense_field(
    inverse_distance: Array, position: Array, strength: Array, /
) -> CloudField:
    potential = inverse_distance @ strength
    displacement = position[:, None, :] - position[None, :, :]
    weight = strength[None, :] * inverse_distance**3
    gradient = -jnp.sum(weight[..., None] * displacement, axis=1)
    successful = jnp.all(jnp.isfinite(potential)) & jnp.all(jnp.isfinite(gradient))
    return CloudField(potential, gradient, successful)


def dense_inverse_distance(position: Array, /) -> Array:
    """Laplace pair kernel `1/d_ij` with a zero diagonal (bounded dense route)."""
    count = position.shape[0]
    displacement = position[:, None, :] - position[None, :, :]
    distance = jnp.sqrt(jnp.sum(displacement**2, axis=-1))
    off_diagonal = ~jnp.eye(count, dtype=jnp.bool_)
    return jnp.where(off_diagonal, 1.0 / jnp.where(off_diagonal, distance, 1.0), 0.0)


def scaled_coupling_matrix(
    inverse_distance: Array, inertia: Array, radius: Array, /
) -> Array:
    """`W = D^{-1/2} G D^{-1/2}`: the dimensionless neighbour coupling."""
    scale = radius / jnp.sqrt(inertia)
    return scale[:, None] * inverse_distance * scale[None, :]


def coupling_spectrum(
    inverse_distance: Array, inertia: Array, radius: Array, /
) -> tuple[Array, Array, Array]:
    """Condition number of `I + W`, its definiteness and the spectral radius of `W`.

    One native eigendecomposition of the symmetric `W` gives all three: `I + W`
    is positive definite iff `λ_min(W) > −1`, its 2-norm condition number is
    `(1 + λ_max)/(1 + λ_min)`, and the retarded (neutral) coupling is
    delay-independently stable iff `ρ(W) < 1`.
    """
    matrix = scaled_coupling_matrix(inverse_distance, inertia, radius)
    evidence = verify_dense_properties(
        matrix, policy=DensePropertyVerificationPolicy(require_hermitian=True)
    )
    lowest = jnp.min(evidence.eigenvalues)
    highest = jnp.max(evidence.eigenvalues)
    definite = evidence.finite & evidence.hermitian & (lowest > -1.0)
    condition = jnp.where(definite, (1.0 + highest) / (1.0 + lowest), jnp.inf)
    radius_ = jnp.max(jnp.abs(evidence.eigenvalues))
    return condition, definite, radius_


class DenseCloudCoupling(AbstractCloudCoupling):
    """Bounded dense pair geometry with a native dense Cholesky solve.

    With fixed bubble positions the pair kernel is prepared once; translating
    bubbles recompute it from the current positions.
    """

    fixed_inverse_distance: Array | None

    def __init__(self, fixed_inverse_distance: Array | None, /) -> None:
        self.fixed_inverse_distance = fixed_inverse_distance

    def inverse_distance(self, position: Array, /) -> Array:
        """Pair kernel at `position` (the prepared one for fixed positions)."""
        if self.fixed_inverse_distance is None:
            return dense_inverse_distance(position)
        return self.fixed_inverse_distance

    def field(self, position: Array, strength: Array, /) -> CloudField:
        return _dense_field(self.inverse_distance(position), position, strength)

    def solve(
        self,
        position: Array,
        inertia: Array,
        forcing: Array,
        radius: Array,
        wall_velocity: Array,
        /,
    ) -> CloudCouplingResult:
        kernel = self.inverse_distance(position)
        explicit = 2.0 * radius * wall_velocity**2
        scale, rhs = _scaled_rhs(inertia, forcing, radius, kernel @ explicit)
        count = radius.shape[0]
        matrix = (
            jnp.eye(count, dtype=rhs.dtype) + scale[:, None] * kernel * scale[None, :]
        )
        linear = solve(
            LinearSystem(DenseLinearOperator(matrix, properties=_SPD)),
            rhs,
            policy=LinearSolvePolicy(DenseCholesky()),
        )
        weighted = scale * linear.value
        field = _dense_field(kernel, position, weighted + explicit)
        return CloudCouplingResult(
            weighted / radius**2,
            field.potential,
            field.gradient,
            linear.successful & field.successful,
            jnp.zeros((), dtype=jnp.int32),
            jnp.asarray(linear.diagnostics.relative_residual).reshape(()),
        )


class _ScaledFMMCouplingAction(StrictModule, NonTrainableState):
    fmm: UniformFMMPlan
    structure: PreparedUniformFMMStructure
    scale: Array

    def __init__(
        self, fmm: UniformFMMPlan, structure: PreparedUniformFMMStructure, scale: Array, /
    ) -> None:
        self.fmm = fmm
        self.structure = structure
        self.scale = scale

    def __call__(self, value: Array, /) -> Array:
        potential = self.fmm.evaluate_monopole(
            self.structure, self.scale * value
        ).potential
        return value + self.scale * potential


class FMMCloudCoupling(AbstractCloudCoupling):
    """Matrix-free conjugate gradients whose pair action is the Laplace FMM.

    The FMM structure depends only on the (fixed) positions and is prepared
    once; the scaled operator `I + W` is refreshed with the current radii and
    inertia coefficients at every evaluation. `resource` is the FMM capacity
    evidence of the prepared structure.
    """

    fmm: UniformFMMPlan
    structure: PreparedUniformFMMStructure
    linear: PreparedLinearSolve
    resource: CartesianFMMResourceEvidence
    structure_successful: Array
    operator_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        fmm: UniformFMMPlan,
        structure: PreparedUniformFMMStructure,
        /,
        *,
        tolerance: float,
        maximum_iterations: int,
    ) -> None:
        count = structure.positions.shape[0]
        dtype = structure.positions.dtype
        operator_id = canonical_fingerprint(
            {
                "kind": "bubble-cloud-fmm-coupling-operator",
                "fmm": fmm.plan_id,
                "structure": structure.structure_id,
                "form": "jacobi-scaled",
            }
        )
        problem_id = canonical_fingerprint(
            {"kind": "bubble-cloud-fmm-coupling-system", "operator": operator_id}
        )
        template = self._operator(
            fmm, structure, jnp.ones((count,), dtype=dtype), operator_id
        )
        policy = LinearSolvePolicy(
            ConjugateGradient(),
            tolerance=TolerancePolicy(
                relative=float(tolerance),
                absolute=float(tolerance),
                max_steps=int(maximum_iterations),
            ),
        )
        probe = fmm.evaluate_monopole(structure, jnp.zeros((count,), dtype=dtype))
        self.fmm = fmm
        self.structure = structure
        self.linear = prepare(LinearSystem(template, problem_id=problem_id), policy)
        self.resource = probe.evidence
        self.structure_successful = probe.successful
        self.operator_id = operator_id
        self.problem_id = problem_id

    @staticmethod
    def _operator(
        fmm: UniformFMMPlan,
        structure: PreparedUniformFMMStructure,
        scale: Array,
        operator_id: str,
        /,
    ) -> FunctionLinearOperator:
        space = ArraySpace(scale.shape, dtype=scale.dtype)
        return FunctionLinearOperator(
            _ScaledFMMCouplingAction(fmm, structure, scale),
            source=space,
            target=space,
            properties=_SPD,
            operator_id=operator_id,
        )

    def field(self, position: Array, strength: Array, /) -> CloudField:
        del position
        result = self.fmm.evaluate_monopole(self.structure, strength)
        return CloudField(result.potential, result.gradient, result.successful)

    def solve(
        self,
        position: Array,
        inertia: Array,
        forcing: Array,
        radius: Array,
        wall_velocity: Array,
        /,
    ) -> CloudCouplingResult:
        explicit = 2.0 * radius * wall_velocity**2
        neighbor = self.field(position, explicit)
        scale, rhs = _scaled_rhs(inertia, forcing, radius, neighbor.potential)
        system = LinearSystem(
            self._operator(self.fmm, self.structure, scale, self.operator_id),
            problem_id=self.problem_id,
        )
        linear = solve(refresh(self.linear, system), rhs, initial_guess=rhs)
        weighted = scale * linear.value
        field = self.field(position, weighted + explicit)
        return CloudCouplingResult(
            weighted / radius**2,
            field.potential,
            field.gradient,
            linear.successful & neighbor.successful & field.successful,
            jnp.asarray(linear.diagnostics.iterations, dtype=jnp.int32).reshape(()),
            jnp.asarray(linear.diagnostics.relative_residual).reshape(()),
        )


__all__ = [
    "AbstractCloudCoupling",
    "CloudCouplingResult",
    "CloudField",
    "DenseCloudCoupling",
    "FMMCloudCoupling",
    "coupling_spectrum",
    "dense_inverse_distance",
    "scaled_coupling_matrix",
]
