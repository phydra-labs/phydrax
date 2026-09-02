#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._strict import StrictModule
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    matrix_exponential_action,
    solve,
)
from ._cayley_dickson import OctonionAlgebraSpec
from ._product import AlgebraProductPlan


_LU_POLICY = LinearSolvePolicy(DenseLU(), failure=FailurePolicy("status"))


@cache
def _canonical_octonion_algebra_id() -> str:
    return OctonionAlgebraSpec().algebra_id


class UnitOctonionStateGeometry(StrictModule):
    """S7 state geometry, deliberately separate from octonion multiplication."""

    product: AlgebraProductPlan
    tolerance: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(self, product: AlgebraProductPlan, /, *, tolerance: float = 1e-8):
        if not isinstance(product, AlgebraProductPlan):
            raise TypeError("Unit octonion geometry requires a prepared algebra product.")
        canonical_octonion = (
            product.algebra.algebra_id == _canonical_octonion_algebra_id()
        )
        if not canonical_octonion:
            raise TypeError(
                "Unit octonion geometry requires the canonical audited octonion product."
            )
        if float(tolerance) <= 0.0:
            raise ValueError("tolerance must be positive.")
        self.product = product
        self.tolerance = float(tolerance)
        self.geometry_id = f"unit-octonion:{product.plan_id}"

    @staticmethod
    def _real_coordinates(value: ArrayLike, owner: str, /) -> Array:
        coordinates = jnp.asarray(value)
        if coordinates.shape[-1:] != (8,):
            raise ValueError(f"{owner} must have trailing shape (8,).")
        if jnp.iscomplexobj(coordinates):
            raise TypeError(f"{owner} must use real octonion coordinates.")
        return coordinates

    def contains(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape[-1:] != (8,) or jnp.iscomplexobj(value):
            return jnp.asarray(False)
        return jnp.all(jnp.isfinite(value)) & jnp.all(
            jnp.abs(jnp.linalg.norm(value, axis=-1) - 1.0) <= self.tolerance
        )

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        point = self._real_coordinates(state, "Unit-octonion state")
        tangent = self._real_coordinates(vector, "Unit-octonion tangent")
        if point.shape != tangent.shape:
            raise ValueError("Unit-octonion tangent must match the state shape.")
        return tangent - point * jnp.sum(point * tangent, axis=-1, keepdims=True)

    def inner(self, state: ArrayLike, left: ArrayLike, right: ArrayLike, /) -> Array:
        return jnp.sum(
            self.project_tangent(state, left) * self.project_tangent(state, right)
        )

    def retract(self, state: ArrayLike, tangent: ArrayLike, /) -> Array:
        candidate = self._real_coordinates(
            state, "Unit-octonion state"
        ) + self.project_tangent(state, tangent)
        return candidate / jnp.linalg.norm(candidate, axis=-1, keepdims=True)

    def exp(self, state: ArrayLike, tangent: ArrayLike, /) -> Array:
        point = self._real_coordinates(state, "Unit-octonion state")
        direction = self.project_tangent(point, tangent)
        norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        safe = jnp.where(norm > 0.0, norm, 1.0)
        return jnp.cos(norm) * point + jnp.sin(norm) * direction / safe


class MoufangLoopOperations(StrictModule):
    geometry: UnitOctonionStateGeometry

    def __init__(self, geometry: UnitOctonionStateGeometry, /):
        self.geometry = geometry

    def _require_unit(self, *values: ArrayLike) -> None:
        if not all(bool(self.geometry.contains(value)) for value in values):
            raise ValueError("Moufang operations require real unit octonions.")

    def multiply(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        self._require_unit(left, right)
        return self.geometry.product(left, right)

    def inverse(self, value: ArrayLike, /) -> Array:
        self._require_unit(value)
        point = jnp.asarray(value)
        return point.at[..., 1:].multiply(-1)

    def associator(
        self, left: ArrayLike, middle: ArrayLike, right: ArrayLike, /
    ) -> Array:
        self._require_unit(left, middle, right)
        return self.geometry.product(
            self.geometry.product(left, middle), right
        ) - self.geometry.product(left, self.geometry.product(middle, right))

    def moufang_residual(
        self, left: ArrayLike, middle: ArrayLike, right: ArrayLike, /
    ) -> Array:
        self._require_unit(left, middle, right)
        product = self.geometry.product
        lhs = product(product(left, middle), product(right, left))
        rhs = product(left, product(product(middle, right), left))
        return jnp.max(jnp.abs(lhs - rhs))


class BracketingPlan(StrictModule):
    """One immutable binary bracket tree over operand indices."""

    tree: tuple = eqx.field(static=True)
    operand_count: int = eqx.field(static=True)

    def __init__(self, tree: tuple, /, *, operand_count: int):
        count = int(operand_count)
        seen = []

        def visit(node):
            if isinstance(node, int):
                if not 0 <= node < count:
                    raise ValueError("Bracketing operand index is out of range.")
                seen.append(node)
                return
            if not isinstance(node, tuple) or len(node) != 2:
                raise ValueError(
                    "BracketingPlan nodes must be operand indices or binary tuples."
                )
            visit(node[0])
            visit(node[1])

        visit(tree)
        if sorted(seen) != list(range(count)):
            raise ValueError("BracketingPlan must use every operand exactly once.")
        self.tree = tree
        self.operand_count = count

    def evaluate(
        self, product: AlgebraProductPlan, operands: Sequence[ArrayLike], /
    ) -> Array:
        values = tuple(jnp.asarray(value) for value in operands)
        if len(values) != self.operand_count:
            raise ValueError("Bracketing operands do not match operand_count.")

        def evaluate_node(node):
            if isinstance(node, int):
                return values[node]
            return product(evaluate_node(node[0]), evaluate_node(node[1]))

        return evaluate_node(self.tree)


class PreparedUnitOctonionEvolution(StrictModule):
    geometry: UnitOctonionStateGeometry
    brackets: BracketingPlan
    vector_field: Callable
    steps: int = eqx.field(static=True)

    def __init__(
        self,
        geometry: UnitOctonionStateGeometry,
        vector_field,
        brackets: BracketingPlan,
        /,
        *,
        steps: int,
    ):
        if not callable(vector_field) or int(steps) < 1:
            raise ValueError(
                "Evolution vector_field must be callable and steps positive."
            )
        self.geometry = geometry
        self.vector_field = vector_field
        self.brackets = brackets
        self.steps = int(steps)

    def evolve(self, state: ArrayLike, time: ArrayLike, step_size: ArrayLike, /) -> Array:
        current = jnp.asarray(state)
        time_ = jnp.asarray(time)
        dt = jnp.asarray(step_size)
        for _ in range(self.steps):
            velocity = self.vector_field(time_, current, self.brackets)
            current = self.geometry.retract(
                current, dt * self.geometry.project_tangent(current, velocity)
            )
            time_ = time_ + dt
        return current


class G2MatrixElement(StrictModule):
    matrix: Array
    orthogonality_residual: Array
    determinant_residual: Array
    phi_residual: Array
    valid: Array

    def __init__(self, matrix: ArrayLike, phi: ArrayLike, /, *, tolerance: float = 1e-8):
        value = jnp.asarray(matrix)
        phi_ = jnp.asarray(phi, dtype=value.dtype)
        if value.shape != (7, 7) or phi_.shape != (7, 7, 7):
            raise ValueError("G2 element requires a 7x7 matrix and 7x7x7 three-form.")
        orthogonality = jnp.max(jnp.abs(value.T @ value - jnp.eye(7, dtype=value.dtype)))
        determinant = jnp.abs(jnp.linalg.det(value) - 1.0)
        transformed = contract("ia,jb,kc,abc->ijk", value, value, value, phi_)
        phi_residual = jnp.max(jnp.abs(transformed - phi_))
        valid = (
            jnp.all(jnp.isfinite(value))
            & (orthogonality <= tolerance)
            & (determinant <= tolerance)
            & (phi_residual <= tolerance)
        )
        self.matrix = value
        self.orthogonality_residual = orthogonality
        self.determinant_residual = determinant
        self.phi_residual = phi_residual
        self.valid = valid


class G2LocalLogResult(StrictModule):
    coordinates: Array
    residual: Array
    valid: Array

    def __init__(
        self,
        coordinates: ArrayLike,
        residual: ArrayLike,
        valid: ArrayLike,
        /,
    ):
        self.coordinates = jnp.asarray(coordinates)
        self.residual = jnp.asarray(residual)
        self.valid = jnp.asarray(valid, dtype=bool)


class G2GroupOperations(StrictModule):
    phi: Array
    derivation_basis: Array
    tolerance: float = eqx.field(static=True)

    def __init__(
        self, phi: ArrayLike, derivation_basis: ArrayLike, /, *, tolerance: float = 1e-8
    ):
        phi_ = jnp.asarray(phi)
        basis = jnp.asarray(derivation_basis, dtype=phi_.dtype)
        if phi_.shape != (7, 7, 7) or basis.shape != (14, 7, 7):
            raise ValueError(
                "G2 operations require the invariant three-form and a 14D derivation basis."
            )
        skew_residual = jnp.max(jnp.abs(basis + jnp.swapaxes(basis, -1, -2)))
        infinitesimal = (
            contract("aip,pjk->aijk", basis, phi_)
            + contract("ajp,ipk->aijk", basis, phi_)
            + contract("akp,ijp->aijk", basis, phi_)
        )
        derivation_residual = jnp.max(jnp.abs(infinitesimal))
        basis_rank = jnp.linalg.matrix_rank(basis.reshape((14, 49)))
        if bool(
            (skew_residual > tolerance)
            | (derivation_residual > tolerance)
            | (basis_rank != 14)
        ):
            raise ValueError(
                "G2 derivation basis must be independent, skew, and preserve phi."
            )
        self.phi = phi_
        self.derivation_basis = basis
        self.tolerance = float(tolerance)

    def element(self, matrix: ArrayLike, /) -> G2MatrixElement:
        return G2MatrixElement(matrix, self.phi, tolerance=self.tolerance)

    def compose(
        self, left: G2MatrixElement, right: G2MatrixElement, /
    ) -> G2MatrixElement:
        return self.element(left.matrix @ right.matrix)

    def inverse(self, element: G2MatrixElement, /) -> G2MatrixElement:
        return self.element(element.matrix.T)

    def exp(self, coordinates: ArrayLike, /) -> G2MatrixElement:
        coefficients = jnp.asarray(coordinates)
        if coefficients.shape != (14,):
            raise ValueError("G2 exponential coordinates must have shape (14,).")
        generator = contract("a,aij->ij", coefficients, self.derivation_basis)
        operator = DenseLinearOperator(generator, operator_id="g2:generator")
        exponential = matrix_exponential_action(
            operator, jnp.eye(7, dtype=generator.dtype)
        ).value
        return self.element(exponential)

    def local_log(
        self,
        element: G2MatrixElement,
        /,
    ) -> G2LocalLogResult:
        if not isinstance(element, G2MatrixElement) or not bool(element.valid):
            raise ValueError("G2 local logarithm requires a certified element.")
        difference = element.matrix - jnp.eye(7, dtype=element.matrix.dtype)
        logarithm = jnp.zeros_like(difference)
        power = jnp.eye(7, dtype=element.matrix.dtype)
        for order in range(1, 25):
            power = power @ difference
            logarithm = logarithm + ((1.0 if order % 2 else -1.0) * power / order)
        design = self.derivation_basis.reshape((14, 49)).T
        coordinates = solve(
            LinearSystem(
                DenseLinearOperator(
                    design.T @ design,
                    operator_id="g2:log-normal-operator",
                ),
                problem_id="g2:local-log",
            ),
            design.T @ logarithm.reshape((49,)),
            policy=_LU_POLICY,
        ).value
        reconstructed = self.exp(coordinates)
        residual = jnp.max(jnp.abs(reconstructed.matrix - element.matrix))
        valid = (
            reconstructed.valid & jnp.isfinite(residual) & (residual <= self.tolerance)
        )
        return G2LocalLogResult(
            jnp.where(valid, coordinates, jnp.full_like(coordinates, jnp.nan)),
            residual,
            valid,
        )

    def adjoint(self, element: G2MatrixElement, generator: ArrayLike, /) -> Array:
        value = jnp.asarray(generator)
        return element.matrix @ value @ element.matrix.T


class AlgebraMatrixLayout(StrictModule):
    product: AlgebraProductPlan
    rows: int = eqx.field(static=True)
    columns: int = eqx.field(static=True)

    def __init__(self, product: AlgebraProductPlan, rows: int, columns: int, /):
        if int(rows) < 1 or int(columns) < 1:
            raise ValueError("Algebra matrix dimensions must be positive.")
        self.product = product
        self.rows = int(rows)
        self.columns = int(columns)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.rows, self.columns, self.product.algebra.coordinate_dimension

    def validate(self, value: ArrayLike, /) -> Array:
        matrix = jnp.asarray(value)
        if matrix.shape != self.shape:
            raise ValueError(f"Algebra matrix must have shape {self.shape}.")
        return matrix


class AlgebraMatrixProductPlan(StrictModule):
    left: AlgebraMatrixLayout
    right: AlgebraMatrixLayout

    def __init__(self, left: AlgebraMatrixLayout, right: AlgebraMatrixLayout, /):
        if left.columns != right.rows or left.product.plan_id != right.product.plan_id:
            raise ValueError(
                "Algebra matrix layouts have incompatible dimensions or products."
            )
        self.left = left
        self.right = right

    def __call__(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        a = self.left.validate(left)
        b = self.right.validate(right)
        output = jnp.zeros(
            (
                self.left.rows,
                self.right.columns,
                self.left.product.algebra.coordinate_dimension,
            ),
            dtype=jnp.result_type(a, b),
        )
        for row in range(self.left.rows):
            for column in range(self.right.columns):
                value = jnp.zeros(
                    (self.left.product.algebra.coordinate_dimension,), dtype=output.dtype
                )
                for inner in range(self.left.columns):
                    value = value + self.left.product(a[row, inner], b[inner, column])
                output = output.at[row, column].set(value)
        return output


class AlgebraOperatorInverse(StrictModule):
    value: Array
    residual: Array
    side: Literal["left", "right"] = eqx.field(static=True)
    valid: Array

    def __init__(
        self,
        value: ArrayLike,
        residual: ArrayLike,
        side: Literal["left", "right"],
        valid: ArrayLike,
        /,
    ):
        self.value = jnp.asarray(value)
        self.residual = jnp.asarray(residual)
        self.side = side
        self.valid = jnp.asarray(valid, dtype=bool)


def _regular_matrix(
    product: AlgebraProductPlan, value: Array, side: Literal["left", "right"], /
) -> Array:
    basis = jnp.eye(product.algebra.coordinate_dimension, dtype=value.dtype)
    columns = jax.vmap(
        lambda vector: (
            product(value, vector) if side == "left" else product(vector, value)
        )
    )(basis)
    return columns.T


def _algebra_solve(
    product: AlgebraProductPlan,
    value: ArrayLike,
    rhs: ArrayLike,
    side: Literal["left", "right"],
    /,
    *,
    tolerance: float = 1e-8,
) -> AlgebraOperatorInverse:
    coefficient = jnp.asarray(value)
    rhs_ = jnp.asarray(rhs, dtype=coefficient.dtype)
    matrix = _regular_matrix(product, coefficient, side)
    result = solve(
        LinearSystem(
            DenseLinearOperator(matrix, operator_id=f"{product.plan_id}:{side}-regular"),
            problem_id=f"{product.plan_id}:{side}-solve",
        ),
        rhs_,
        policy=_LU_POLICY,
    )
    residual = jnp.max(jnp.abs(matrix @ result.value - rhs_))
    valid = result.diagnostics.converged & (residual <= tolerance)
    return AlgebraOperatorInverse(
        jnp.where(valid, result.value, jnp.full_like(result.value, jnp.nan)),
        residual,
        side,
        valid,
    )


def algebra_left_solve(
    product: AlgebraProductPlan,
    value: ArrayLike,
    rhs: ArrayLike,
    /,
    *,
    tolerance: float = 1e-8,
) -> AlgebraOperatorInverse:
    return _algebra_solve(product, value, rhs, "left", tolerance=tolerance)


def algebra_right_solve(
    product: AlgebraProductPlan,
    value: ArrayLike,
    rhs: ArrayLike,
    /,
    *,
    tolerance: float = 1e-8,
) -> AlgebraOperatorInverse:
    return _algebra_solve(product, value, rhs, "right", tolerance=tolerance)


class AlgebraRegularSpectrum(StrictModule):
    eigenvalues: Array
    operator: Array
    side: Literal["left", "right"] = eqx.field(static=True)
    valid: Array

    def __init__(
        self,
        product: AlgebraProductPlan,
        value: ArrayLike,
        /,
        *,
        side: Literal["left", "right"] = "left",
    ):
        if side not in ("left", "right"):
            raise ValueError("Regular spectrum side must be left or right.")
        operator = _regular_matrix(product, jnp.asarray(value), side)
        eigenvalues = jnp.linalg.eigvals(operator)
        self.eigenvalues = eigenvalues
        self.operator = operator
        self.side = side
        self.valid = jnp.all(jnp.isfinite(eigenvalues))


__all__ = [
    "AlgebraMatrixLayout",
    "AlgebraMatrixProductPlan",
    "AlgebraOperatorInverse",
    "AlgebraRegularSpectrum",
    "BracketingPlan",
    "G2GroupOperations",
    "G2LocalLogResult",
    "G2MatrixElement",
    "MoufangLoopOperations",
    "PreparedUnitOctonionEvolution",
    "UnitOctonionStateGeometry",
    "algebra_left_solve",
    "algebra_right_solve",
]
