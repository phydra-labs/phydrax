#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Setting-invariant cell metrics, Cartesian realization and bounded lattice reduction.

Cells are row lattices ``H`` (rows are the lattice vectors) as in
`phydrax.discretization.PeriodicCell`; fractional column coordinates ``s`` map
to Cartesian ``r = H^T s`` and the metric is ``G = H H^T``.  A space-group
operation ``s -> R s + t`` is a Euclidean isometry exactly when
``R^T G R = G``.

`SettingMetricChart` derives the invariant metric subspace of one exact
setting from its operations through the native exact Smith decomposition, so
monoclinic unique axes, rhombohedral axes and other nonstandard settings get
their own constraints instead of a crystal-system template.  Cells are
recovered with the native dense Cholesky factorization and carry definiteness,
factorization and per-operation realization evidence.

`niggli_reduce` implements the Krivy-Gruber reduction (I. Krivy and B. Gruber,
Acta Cryst. A32 (1976) 297-298) with the epsilon comparisons of R. W.
Grosse-Kunstleve, N. K. Sauter and P. D. Adams, Acta Cryst. A60 (2004) 1-6,
as a bounded host preparation that returns its integer basis change and
termination evidence.  `enumerate_sublattice_transforms` lists every
sublattice of a given index once through Hermite normal forms.
"""

from __future__ import annotations

from fractions import Fraction
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...algebraic._exact_integer import (
    exact_integer_determinant,
    smith_normal_decomposition,
    smith_rank,
)
from ...discretization import PeriodicCell
from ...ein import contract
from ...linalg import (
    DenseCholesky,
    DenseLinearOperator,
    DensePropertyEvidence,
    DensePropertyVerificationPolicy,
    determinant_small_linear,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    SmallLinearSolvePlan,
    solve_small_linear,
    verify_dense_properties,
)
from ...typing import Dim, HostFloat64, HostInt64, parse
from ._hall import IntegerMatrix
from ._setting import SpaceGroupSetting


NiggliStatus: TypeAlias = Literal["reduced", "iteration-limit"]

# Symmetric-metric coordinates g = (G00, G11, G22, G12, G02, G01).
_METRIC_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_METRIC_INDEX = {
    **{pair: index for index, pair in enumerate(_METRIC_PAIRS)},
    **{(j, i): index for index, (i, j) in enumerate(_METRIC_PAIRS)},
}
_SMALL3 = SmallLinearSolvePlan(3)


class SublatticeDim(Dim, minimum=1):
    """Enumerated sublattices of one index."""


def _invariance_constraints(rotations: tuple[IntegerMatrix, ...], /) -> list[list[int]]:
    """Return integer rows of ``(R^T G R - G)_{ij} = 0`` over metric coordinates."""
    rows: list[list[int]] = []
    for rotation in rotations:
        for i, j in _METRIC_PAIRS:
            row = [0] * 6
            for k in range(3):
                for m in range(3):
                    row[_METRIC_INDEX[(k, m)]] += (
                        rotation[3 * k + i] * rotation[3 * m + j]
                    )
            row[_METRIC_INDEX[(i, j)]] -= 1
            rows.append(row)
    return rows or [[0] * 6]


def _canonical_sign(column: list[int], /) -> tuple[int, ...]:
    leading = next(value for value in column if value != 0)
    return tuple(value if leading > 0 else -value for value in column)


@final
class SettingMetricChart(StrictModule, NonTrainableState):
    """Invariant metric subspace of one exact setting and its cell chart.

    ``basis[k]`` lists integer metric coordinates ``(G00, G11, G22, G12, G02,
    G01)`` spanning ``{G : R^T G R = G for every operation}``; ``dimension`` is
    its exact dimension (6 triclinic, 4 monoclinic, 3 orthorhombic, 2
    tetragonal/trigonal/hexagonal, 1 cubic).  ``left_inverse`` is the exact
    rational left inverse of the basis, so coordinates of an invariant metric
    are recovered without a runtime solve.
    """

    setting_id: str = eqx.field(static=True)
    rotations: tuple[IntegerMatrix, ...] = eqx.field(static=True)
    point_rotations: tuple[IntegerMatrix, ...] = eqx.field(static=True)
    translation_numerators: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    translation_denominator: int = eqx.field(static=True)
    basis: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    left_inverse: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    constraint_rank: int = eqx.field(static=True)

    def __init__(self, setting: SpaceGroupSetting, /) -> None:
        if not isinstance(setting, SpaceGroupSetting):
            raise TypeError("setting must be a SpaceGroupSetting.")
        point_rotations = setting.rotations[: setting.point_group_order]
        constraints = _invariance_constraints(point_rotations[1:])
        diagonal, _, right = smith_normal_decomposition(constraints)
        rank = smith_rank(diagonal)
        basis = tuple(
            _canonical_sign([int(right[row, column]) for row in range(6)])
            for column in range(rank, 6)
        )
        if not basis:
            raise RuntimeError("A space-group setting must admit an invariant metric.")
        # basis^T is 6 x K with full column rank: U B V = [diag(d); 0] gives the
        # exact left inverse V diag(1 / d) U[:K].
        columns = [[basis[k][row] for k in range(len(basis))] for row in range(6)]
        pivots, left, right_basis = smith_normal_decomposition(columns)
        size = len(basis)
        scales = [int(pivots[k, k]) for k in range(size)]
        if any(scale == 0 for scale in scales):
            raise RuntimeError("Invariant metric basis lost full column rank.")
        inverse = tuple(
            tuple(
                sum(
                    (
                        Fraction(
                            int(right_basis[k, m]) * int(left[m, coordinate]), scales[m]
                        )
                        for m in range(size)
                    ),
                    Fraction(0),
                )
                for coordinate in range(6)
            )
            for k in range(size)
        )
        self.setting_id = setting.setting_id
        self.rotations = setting.rotations
        self.point_rotations = point_rotations
        self.translation_numerators = setting.translation_numerators
        self.translation_denominator = setting.translation_denominator
        self.basis = basis
        self.left_inverse = inverse
        self.constraint_rank = rank

    @property
    def dimension(self) -> int:
        return len(self.basis)

    def basis_metrics(self, dtype: DTypeLike, /) -> Array:
        """Return the basis as symmetric metric matrices ``(dimension, 3, 3)``."""
        matrices = np.zeros((self.dimension, 3, 3), dtype=np.float64)
        for k, coordinates in enumerate(self.basis):
            for index, (i, j) in enumerate(_METRIC_PAIRS):
                matrices[k, i, j] = coordinates[index]
                matrices[k, j, i] = coordinates[index]
        return jnp.asarray(matrices, dtype=dtype)

    def metric(self, coefficients: ArrayLike, /) -> Array:
        """Return ``G = sum_k c_k B_k`` for chart coordinates of shape ``(dimension,)``."""
        values = jnp.asarray(coefficients)
        if values.shape != (self.dimension,):
            raise ValueError(f"coefficients must have shape ({self.dimension},).")
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("coefficients must be real floating values.")
        return contract(
            "k,kij->ij", values, self.basis_metrics(values.dtype), backend="jax"
        )

    def symmetrize(self, metric: ArrayLike, /) -> Array:
        """Return the group average ``mean_R R^T G R``, a projection onto invariants."""
        value = jnp.asarray(metric)
        if value.shape != (3, 3) or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError("metric must be a real floating 3x3 matrix.")
        rotations = jnp.asarray(
            np.asarray(self.point_rotations, dtype=np.float64).reshape((-1, 3, 3)),
            dtype=value.dtype,
        )
        images = contract("rki,kl,rlj->rij", rotations, value, rotations, backend="jax")
        return jnp.mean(images, axis=0)

    def coefficients(self, metric: ArrayLike, /) -> Array:
        """Return chart coordinates of the group-averaged ``metric``."""
        averaged = self.symmetrize(metric)
        vector = jnp.stack([averaged[i, j] for i, j in _METRIC_PAIRS])
        inverse = jnp.asarray(
            np.asarray(
                [[float(value) for value in row] for row in self.left_inverse],
                dtype=np.float64,
            ),
            dtype=averaged.dtype,
        )
        return inverse @ vector

    def cell(
        self,
        coefficients: ArrayLike,
        /,
        *,
        realization_tolerance: float | None = None,
    ) -> SettingCellResult:
        """Recover a positive row cell from chart coordinates with full evidence.

        The metric is certified positive definite by native dense property
        verification, factored by the native dense Cholesky owner into the
        lower-triangular row cell ``H`` (``H H^T = G``), and every operation is
        realized as a Cartesian map ``r -> C r + H^T t`` with
        ``C = H^T R H^{-T}`` obtained from a native small solve.  Failure is
        reported through ``successful``; no metric is repaired.
        """
        metric = self.metric(coefficients)
        dtype = metric.dtype
        tolerance = (
            1024.0 * float(jnp.finfo(dtype).eps)
            if realization_tolerance is None
            else float(realization_tolerance)
        )
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("realization_tolerance must be finite and positive.")
        evidence = verify_dense_properties(
            metric,
            policy=DensePropertyVerificationPolicy(require_positive_definite=True),
        )
        operator = DenseLinearOperator(
            metric,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "verified",
                    "positive_semidefinite": "verified",
                    "positive_definite": "verified",
                },
            ),
        )
        state = prepare(LinearSystem(operator), LinearSolvePolicy(DenseCholesky())).state
        vectors = state.factor
        factorization_valid = ~state.invalid
        rotations = jnp.asarray(
            np.asarray(self.rotations, dtype=np.float64).reshape((-1, 3, 3)), dtype=dtype
        )
        translations = jnp.asarray(
            np.asarray(self.translation_numerators, dtype=np.float64)
            / self.translation_denominator,
            dtype=dtype,
        )
        count = rotations.shape[0]
        # H C^T = R^T H for every operation.
        realization = solve_small_linear(
            _SMALL3,
            jnp.broadcast_to(vectors, (count, 3, 3)),
            contract("oki,kj->oij", rotations, vectors, backend="jax"),
        )
        cartesian = jnp.swapaxes(realization.value, -1, -2)
        scale = jnp.max(jnp.abs(metric))
        invariance = contract(
            "oki,kl,olj->oij", rotations, metric, rotations, backend="jax"
        )
        metric_residual = jnp.max(jnp.abs(invariance - metric[None])) / scale
        orthogonality = contract("oik,ojk->oij", cartesian, cartesian, backend="jax")
        orthogonality_residual = jnp.max(jnp.abs(orthogonality - jnp.eye(3, dtype=dtype)))
        solves = jnp.all(realization.successful)
        successful = (
            evidence.successful
            & factorization_valid
            & solves
            & (metric_residual <= tolerance)
            & (orthogonality_residual <= tolerance)
        )
        return SettingCellResult(
            vectors,
            metric,
            cartesian,
            translations @ vectors,
            metric_residual,
            orthogonality_residual,
            evidence,
            factorization_valid,
            solves,
            successful,
            self.setting_id,
            tolerance,
        )

    def periodic_cell(
        self,
        coefficients: ArrayLike,
        /,
        *,
        realization_tolerance: float | None = None,
        maximum_condition_number: float | None = None,
        maximum_image_count: int = 4096,
    ) -> PeriodicCell:
        """Return the certified `PeriodicCell` of chart coordinates (host route).

        Raises `ValueError` with the failed evidence when the cell is not a
        positive, symmetry-realizing lattice.
        """
        result = self.cell(coefficients, realization_tolerance=realization_tolerance)
        if not bool(jax.device_get(result.successful)):
            raise ValueError(
                "Chart coordinates do not give a positive symmetry-realizing cell: "
                f"positive_definite={bool(result.property_evidence.positive_definite)}, "
                f"factorization_valid={bool(result.factorization_valid)}, "
                f"metric_residual={float(result.metric_residual)}, "
                f"orthogonality_residual={float(result.orthogonality_residual)}."
            )
        return PeriodicCell(
            np.asarray(jax.device_get(result.vectors)),
            maximum_condition_number=maximum_condition_number,
            maximum_image_count=maximum_image_count,
        )


@final
class SettingCellResult(StrictModule, NonTrainableState):
    """Cell recovered from an invariant metric with realization evidence.

    ``vectors`` is the lower-triangular row cell, ``cartesian_rotations[o]``
    and ``cartesian_translations[o]`` realize operation ``o`` as
    ``r -> C r + d`` on Cartesian column vectors.  ``metric_residual`` is the
    relative ``max |R^T G R - G|`` and ``orthogonality_residual`` the largest
    ``|C C^T - I|`` entry.
    """

    vectors: Array
    metric: Array
    cartesian_rotations: Array
    cartesian_translations: Array
    metric_residual: Array
    orthogonality_residual: Array
    property_evidence: DensePropertyEvidence
    factorization_valid: Array
    realization_solves_successful: Array
    successful: Array
    setting_id: str = eqx.field(static=True)
    realization_tolerance: float = eqx.field(static=True)


def metric_from_cell_parameters(
    lengths: ArrayLike, angles_degrees: ArrayLike, /
) -> Array:
    """Return ``G_ij = a_i a_j cos(theta_ij)`` from ``(a, b, c)`` and ``(alpha, beta, gamma)``.

    ``alpha`` is the angle between ``b`` and ``c``, ``beta`` between ``a`` and
    ``c`` and ``gamma`` between ``a`` and ``b``.  Positivity is not implied;
    certify it through `SettingMetricChart.cell`.
    """
    a = jnp.asarray(lengths)
    angles = jnp.asarray(angles_degrees)
    if a.shape != (3,) or angles.shape != (3,):
        raise ValueError("lengths and angles_degrees must each have shape (3,).")
    if not jnp.issubdtype(a.dtype, jnp.floating) or a.dtype != angles.dtype:
        raise TypeError("lengths and angles must share one real floating dtype.")
    cosines = jnp.cos(jnp.deg2rad(angles))
    one = jnp.ones((), dtype=a.dtype)
    correlation = jnp.stack(
        (
            jnp.stack((one, cosines[2], cosines[1])),
            jnp.stack((cosines[2], one, cosines[0])),
            jnp.stack((cosines[1], cosines[0], one)),
        )
    )
    return a[:, None] * correlation * a[None, :]


@final
class NiggliReduction(StrictModule, NonTrainableState):
    """Bounded Krivy-Gruber reduction with its exact integer basis change.

    ``transformation`` is the unimodular integer ``P`` whose columns are the
    reduced basis vectors in input fractional coordinates, so the reduced row
    cell is ``P^T H`` and the reduced metric ``P^T G P`` (recomputed from the
    input at every step, never accumulated).  ``conditions_satisfied`` is an
    independent check of every Niggli condition at ``tolerance`` on the final
    metric; ``status`` is ``"iteration-limit"`` when the bound ended the loop.
    """

    __strict_contract__ = True

    transformation: HostInt64[Literal[3], Literal[3]]
    vectors: HostFloat64[Literal[3], Literal[3]]
    metric: HostFloat64[Literal[3], Literal[3]]
    iterations: int = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    determinant: int = eqx.field(static=True)
    conditions_satisfied: bool = eqx.field(static=True)
    status: NiggliStatus = eqx.field(static=True)

    @property
    def successful(self) -> bool:
        return (
            self.status == "reduced"
            and self.conditions_satisfied
            and self.determinant == 1
        )


def _parameters(metric: np.ndarray, /) -> tuple[float, float, float, float, float, float]:
    return (
        float(metric[0, 0]),
        float(metric[1, 1]),
        float(metric[2, 2]),
        float(2.0 * metric[1, 2]),
        float(2.0 * metric[0, 2]),
        float(2.0 * metric[0, 1]),
    )


def _sign_flip(
    xi: float, eta: float, zeta: float, eps: float, positive: bool, /
) -> tuple[int, int, int]:
    """Return the diagonal sign change of Krivy-Gruber steps 3 and 4."""
    if positive:
        return (
            -1 if xi < -eps else 1,
            -1 if eta < -eps else 1,
            -1 if zeta < -eps else 1,
        )
    flips = [1, 1, 1]
    zero_index = None
    for index, value in enumerate((xi, eta, zeta)):
        if value > eps:
            flips[index] = -1
        elif zero_index is None and value >= -eps:
            zero_index = index
    if flips[0] * flips[1] * flips[2] < 0:
        if zero_index is None:
            raise RuntimeError("Niggli step 4 found no zero component to fix the sign.")
        flips[zero_index] = -1
    return flips[0], flips[1], flips[2]


type _Columns = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]


def _niggli_step(metric: np.ndarray, eps: float, /) -> _Columns | None:
    """Return the next Krivy-Gruber basis change, or ``None`` when reduced.

    The result lists the new basis vectors (matrix columns) in the current
    basis; each step's metric update is the one tabulated by Krivy and Gruber.
    """
    a, b, c, xi, eta, zeta = _parameters(metric)
    # Step 1: swap a and b (c -> -c keeps the orientation).
    if a > b + eps or (abs(a - b) <= eps and abs(xi) > abs(eta) + eps):
        return ((0, 1, 0), (1, 0, 0), (0, 0, -1))
    # Step 2: swap b and c (a -> -a keeps the orientation).
    if b > c + eps or (abs(b - c) <= eps and abs(eta) > abs(zeta) + eps):
        return ((-1, 0, 0), (0, 0, 1), (0, 1, 0))
    # Steps 3 and 4: make the off-diagonal signs all positive or all nonpositive.
    positive = xi > eps and eta > eps and zeta > eps
    nonpositive = xi <= eps and eta <= eps and zeta <= eps
    if not positive and not nonpositive:
        all_positive = sum(value < -eps for value in (xi, eta, zeta)) % 2 == 0 and all(
            abs(value) > eps for value in (xi, eta, zeta)
        )
        i, j, k = _sign_flip(xi, eta, zeta, eps, all_positive)
        return ((i, 0, 0), (0, j, 0), (0, 0, k))
    # Step 5: c -> c - sign(xi) b.
    s = 1 if xi > 0 else -1
    if (
        abs(xi) > b + eps
        or (abs(xi - b) <= eps and 2 * eta < zeta - eps)
        or (abs(xi + b) <= eps and zeta < -eps)
    ):
        return ((1, 0, 0), (0, 1, 0), (0, -s, 1))
    # Step 6: c -> c - sign(eta) a.
    s = 1 if eta > 0 else -1
    if (
        abs(eta) > a + eps
        or (abs(eta - a) <= eps and 2 * xi < zeta - eps)
        or (abs(eta + a) <= eps and zeta < -eps)
    ):
        return ((1, 0, 0), (0, 1, 0), (-s, 0, 1))
    # Step 7: b -> b - sign(zeta) a.
    s = 1 if zeta > 0 else -1
    if (
        abs(zeta) > a + eps
        or (abs(zeta - a) <= eps and 2 * xi < eta - eps)
        or (abs(zeta + a) <= eps and eta < -eps)
    ):
        return ((1, 0, 0), (-s, 1, 0), (0, 0, 1))
    # Step 8: c -> a + b + c.
    total = xi + eta + zeta + a + b
    if total < -eps or (abs(total) <= eps and 2 * (a + eta) + zeta > eps):
        return ((1, 0, 0), (0, 1, 0), (1, 1, 1))
    return None


def _niggli_conditions(metric: np.ndarray, eps: float, /) -> bool:
    """Check the Niggli conditions independently of the step selection."""
    a, b, c, xi, eta, zeta = _parameters(metric)
    ordered = a <= b + eps and b <= c + eps
    bounded = abs(xi) <= b + eps and abs(eta) <= a + eps and abs(zeta) <= a + eps
    positive = xi > eps and eta > eps and zeta > eps
    nonpositive = xi <= eps and eta <= eps and zeta <= eps
    special = all(
        (
            not abs(a - b) <= eps or abs(xi) <= abs(eta) + eps,
            not abs(b - c) <= eps or abs(eta) <= abs(zeta) + eps,
            not abs(xi - b) <= eps or zeta <= 2 * eta + eps,
            not abs(eta - a) <= eps or zeta <= 2 * xi + eps,
            not abs(zeta - a) <= eps or eta <= 2 * xi + eps,
            not abs(xi + b) <= eps or abs(zeta) <= eps,
            not abs(eta + a) <= eps or abs(zeta) <= eps,
            not abs(zeta + a) <= eps or abs(eta) <= eps,
            not abs(xi + eta + zeta + a + b) <= eps or 2 * (a + eta) + zeta <= eps,
        )
    )
    return ordered and bounded and (positive or nonpositive) and special


def niggli_reduce(
    vectors: ArrayLike,
    /,
    *,
    relative_tolerance: float = 1.0e-5,
    maximum_iterations: int = 1000,
) -> NiggliReduction:
    """Reduce a 3D row cell to its Niggli cell with explicit bounds and evidence.

    ``relative_tolerance`` scales the comparison epsilon as
    ``relative_tolerance * V^(2/3)`` in metric units, ``V`` the cell volume
    from the native small determinant.  The input metric must be certified
    positive definite by native dense property verification.  Equivalent
    boundary cases are resolved by the Krivy-Gruber step order; consumers
    comparing cells compare reduced metrics, not basis spellings.
    """
    cell = np.asarray(jax.device_get(jnp.asarray(vectors)), dtype=np.float64)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise ValueError("vectors must be a finite 3x3 row cell.")
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive.")
    if isinstance(maximum_iterations, bool) or not isinstance(maximum_iterations, int):
        raise TypeError("maximum_iterations must be an integer.")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive.")
    metric = cell @ cell.T
    evidence = verify_dense_properties(
        jnp.asarray(metric),
        policy=DensePropertyVerificationPolicy(require_positive_definite=True),
    )
    if not bool(jax.device_get(evidence.successful)):
        raise ValueError("Niggli reduction requires a positive-definite cell metric.")
    volume_squared = float(jax.device_get(determinant_small_linear(_SMALL3, metric)))
    eps = relative_tolerance * volume_squared ** (1.0 / 3.0)
    transformation = np.eye(3, dtype=np.int64)
    iterations = 0
    status: NiggliStatus = "iteration-limit"
    while iterations < maximum_iterations:
        current = transformation.T @ metric @ transformation
        step = _niggli_step(current, eps)
        if step is None:
            status = "reduced"
            break
        transformation = transformation @ np.asarray(step, dtype=np.int64).T
        iterations += 1
    reduced_metric = transformation.T @ metric @ transformation
    return NiggliReduction(
        parse(transformation, HostInt64[Literal[3], Literal[3]], "transformation"),
        transformation.T @ cell,
        reduced_metric,
        iterations,
        maximum_iterations,
        eps,
        exact_integer_determinant(transformation),
        _niggli_conditions(reduced_metric, eps),
        status,
    )


def sublattice_count(index: int, /) -> int:
    """Return the number of index-``index`` sublattices of ``Z^3``: ``sum c f^2``."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer.")
    if index < 1:
        raise ValueError("index must be positive.")
    return sum(
        c * (index // (a * c)) ** 2
        for a in range(1, index + 1)
        if index % a == 0
        for c in range(1, index // a + 1)
        if (index // a) % c == 0
    )


def enumerate_sublattice_transforms(
    index: int, /, *, maximum_count: int
) -> HostInt64[SublatticeDim, Literal[3], Literal[3]]:
    """Return every index-``index`` sublattice basis change exactly once.

    Each matrix ``P`` has columns giving the sublattice basis in the parent
    fractional basis (row cell ``P^T H``). It is the unique lower triangular
    column Hermite normal form ``[[a, 0, 0], [b, c, 0], [d, e, f]]`` with
    ``a c f = index``, ``0 <= b < c`` and ``0 <= d, e < f``. The exact count is
    admitted against ``maximum_count`` before anything is enumerated.
    """
    count = sublattice_count(index)
    if isinstance(maximum_count, bool) or not isinstance(maximum_count, int):
        raise TypeError("maximum_count must be an integer.")
    if count > maximum_count:
        raise ValueError(
            f"Index {index} has {count} sublattices, exceeding maximum_count={maximum_count}."
        )
    transforms = np.empty((count, 3, 3), dtype=np.int64)
    position = 0
    for a in range(1, index + 1):
        if index % a:
            continue
        for c in range(1, index // a + 1):
            if (index // a) % c:
                continue
            f = index // (a * c)
            for b in range(c):
                for d in range(f):
                    for e in range(f):
                        transforms[position] = ((a, 0, 0), (b, c, 0), (d, e, f))
                        position += 1
    if position != count:
        raise RuntimeError("Sublattice enumeration disagrees with its exact count.")
    return parse(
        transforms,
        HostInt64[SublatticeDim, Literal[3], Literal[3]],
        "sublattice transforms",
    )


__all__ = [
    "enumerate_sublattice_transforms",
    "metric_from_cell_parameters",
    "niggli_reduce",
    "NiggliReduction",
    "NiggliStatus",
    "SettingCellResult",
    "SettingMetricChart",
    "sublattice_count",
    "SublatticeDim",
]
