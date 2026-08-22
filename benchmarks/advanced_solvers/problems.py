#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .schema import stable_fingerprint


Capability = Literal[
    "linear.scalar",
    "linear.block",
    "nonlinear.root",
    "nonlinear.vi",
    "eigen.general",
    "continuation.fold",
    "optimization.unconstrained",
    "optimization.constrained",
    "optimization.proximal",
    "optimization.linear-program",
    "optimization.quadratic-program",
    "optimization.conic-program",
]


@dataclass(frozen=True)
class SparseLinearProblem:
    name: str
    variant: str
    seed: int
    rows: np.ndarray
    columns: np.ndarray
    coefficients: np.ndarray
    rhs: np.ndarray
    dimension: int
    block_size: int

    @property
    def capability(self) -> Capability:
        return "linear.scalar" if self.block_size == 1 else "linear.block"

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.zeros((self.dimension, self.dimension), dtype=self.coefficients.dtype)
        np.add.at(matrix, (self.rows, self.columns), self.coefficients)
        return matrix

    def identity(self) -> dict[str, Any]:
        parameters = {
            "construction": "one-dimensional block Poisson",
            "symmetric": True,
            "positive_definite": True,
        }
        return _identity(
            family="linear",
            name=self.name,
            variant=self.variant,
            seed=self.seed,
            dtype=self.coefficients.dtype,
            parameters=parameters,
            arrays=(self.rows, self.columns, self.coefficients, self.rhs),
        )

    def sizes(self) -> dict[str, int]:
        rhs_count = 1 if self.rhs.ndim == 1 else self.rhs.shape[1]
        return {
            "dimension": self.dimension,
            "rows": self.dimension,
            "columns": self.dimension,
            "nnz": int(self.coefficients.size),
            "block_size": self.block_size,
            "right_hand_sides": int(rhs_count),
        }


@dataclass(frozen=True)
class NonlinearProblem:
    name: str
    variant: str
    seed: int
    initial: np.ndarray
    target: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    diagonal: np.ndarray | None = None
    root_kind: Literal["separable", "semilinear-poisson-1d"] = "separable"
    nonlinearity: float = 1.0
    grid_spacing: float | None = None

    @property
    def capability(self) -> Capability:
        return "nonlinear.root" if self.variant == "root" else "nonlinear.vi"

    def residual(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self.variant == "root":
            if self.root_kind == "separable":
                return value * value - self.target
            if self.grid_spacing is None:
                raise ValueError("semilinear Poisson problem is missing grid spacing")
            extended = np.pad(value, (1, 1))
            laplacian = (
                2.0 * value - extended[:-2] - extended[2:]
            ) / self.grid_spacing**2
            return laplacian + self.nonlinearity * value**3 - self.target
        if self.diagonal is None:
            raise ValueError("VI problem is missing its diagonal operator")
        return self.diagonal * value - self.target

    def natural_map(self, value: np.ndarray) -> np.ndarray:
        if self.variant != "vi" or self.lower is None or self.upper is None:
            raise ValueError("natural_map is defined only for the VI problem")
        residual = self.residual(value)
        projected = np.clip(np.asarray(value) - residual, self.lower, self.upper)
        return np.asarray(value) - projected

    def jacobian(self, value: np.ndarray) -> np.ndarray:
        if self.variant == "root":
            value = np.asarray(value, dtype=np.float64)
            if self.root_kind == "separable":
                return np.diag(2.0 * value)
            if self.grid_spacing is None:
                raise ValueError("semilinear Poisson problem is missing grid spacing")
            dimension = value.size
            inverse_spacing_squared = 1.0 / self.grid_spacing**2
            jacobian = np.diag(
                2.0 * inverse_spacing_squared + 3.0 * self.nonlinearity * value * value
            )
            off_diagonal = np.full(dimension - 1, -inverse_spacing_squared)
            jacobian += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
            return jacobian
        if self.diagonal is None:
            raise ValueError("VI problem is missing its diagonal operator")
        return np.diag(self.diagonal)

    def identity(self) -> dict[str, Any]:
        if self.variant == "root":
            construction = (
                "separable positive root"
                if self.root_kind == "separable"
                else "finite-difference semilinear Poisson root"
            )
        else:
            construction = "box-constrained diagonal VI"
        parameters: dict[str, Any] = {"construction": construction}
        if self.root_kind == "semilinear-poisson-1d":
            parameters.update(
                nonlinearity=self.nonlinearity,
                grid_spacing=self.grid_spacing,
            )
        arrays: list[np.ndarray] = [self.initial, self.target]
        if self.lower is not None:
            arrays.append(self.lower)
        if self.upper is not None:
            arrays.append(self.upper)
        if self.diagonal is not None:
            arrays.append(self.diagonal)
        return _identity(
            family="nonlinear",
            name=self.name,
            variant=self.variant,
            seed=self.seed,
            dtype=self.initial.dtype,
            parameters=parameters,
            arrays=tuple(arrays),
        )

    def sizes(self) -> dict[str, int]:
        dimension = int(self.initial.size)
        nnz = (
            3 * dimension - 2
            if self.variant == "root" and self.root_kind == "semilinear-poisson-1d"
            else dimension
        )
        return {
            "dimension": dimension,
            "rows": dimension,
            "columns": dimension,
            "nnz": nnz,
            "block_size": 1,
            "right_hand_sides": 1,
        }


@dataclass(frozen=True)
class GeneralEigenProblem:
    name: str
    variant: str
    seed: int
    matrix: np.ndarray
    eigenpairs: int

    @property
    def capability(self) -> Capability:
        return "eigen.general"

    def identity(self) -> dict[str, Any]:
        return _identity(
            family="eigen",
            name=self.name,
            variant=self.variant,
            seed=self.seed,
            dtype=self.matrix.dtype,
            parameters={
                "construction": "real nonsymmetric tridiagonal",
                "requested_eigenpairs": self.eigenpairs,
                "target": "largest-magnitude",
                "normal": False,
            },
            arrays=(self.matrix,),
        )

    def sizes(self) -> dict[str, int]:
        dimension = self.matrix.shape[0]
        return {
            "dimension": int(dimension),
            "rows": int(dimension),
            "columns": int(dimension),
            "nnz": int(np.count_nonzero(self.matrix)),
            "block_size": 1,
            "right_hand_sides": self.eigenpairs,
        }


@dataclass(frozen=True)
class ContinuationProblem:
    name: str
    variant: str
    seed: int
    initial_state: np.ndarray
    initial_coordinate: float
    direction: int
    initial_step: float
    min_step: float
    max_step: float
    max_points: int

    @property
    def capability(self) -> Capability:
        return "continuation.fold"

    @staticmethod
    def residual(state: np.ndarray, coordinate: float) -> np.ndarray:
        return np.asarray(state, dtype=np.float64) ** 2 - float(coordinate)

    def identity(self) -> dict[str, Any]:
        return _identity(
            family="continuation",
            name=self.name,
            variant=self.variant,
            seed=self.seed,
            dtype=self.initial_state.dtype,
            parameters={
                "construction": "quadratic fold x^2 - mu = 0",
                "initial_coordinate": self.initial_coordinate,
                "direction": self.direction,
                "initial_step": self.initial_step,
                "min_step": self.min_step,
                "max_step": self.max_step,
                "max_points": self.max_points,
            },
            arrays=(self.initial_state,),
        )

    def sizes(self) -> dict[str, int]:
        dimension = int(self.initial_state.size)
        return {
            "dimension": dimension,
            "rows": dimension,
            "columns": dimension,
            "nnz": dimension,
            "block_size": 1,
            "right_hand_sides": 1,
        }


@dataclass(frozen=True)
class OptimizationProblem:
    name: str
    variant: str
    seed: int
    initial: np.ndarray
    optimum: np.ndarray
    target: np.ndarray | None = None
    l1_weight: float = 0.0

    @property
    def capability(self) -> Capability:
        if self.variant == "unconstrained":
            return "optimization.unconstrained"
        if self.variant == "constrained":
            return "optimization.constrained"
        if self.variant == "proximal":
            return "optimization.proximal"
        raise ValueError(f"Unsupported optimization variant {self.variant!r}")

    def objective(self, value: np.ndarray) -> float:
        point = np.asarray(value, dtype=np.float64)
        if self.variant == "unconstrained":
            left = point[:-1]
            right = point[1:]
            return float(np.sum(100.0 * (right - left * left) ** 2 + (1.0 - left) ** 2))
        if self.variant == "constrained":
            return float(2.0 * (point @ point - 1.0) - point[0])
        if self.target is None:
            raise ValueError("proximal problem is missing its target")
        return float(
            0.5 * np.sum((point - self.target) ** 2)
            + self.l1_weight * np.sum(np.abs(point))
        )

    def gradient(self, value: np.ndarray) -> np.ndarray:
        point = np.asarray(value, dtype=np.float64)
        if self.variant == "unconstrained":
            gradient = np.zeros_like(point)
            gradient[:-1] += -400.0 * point[:-1] * (point[1:] - point[:-1] ** 2) - 2.0 * (
                1.0 - point[:-1]
            )
            gradient[1:] += 200.0 * (point[1:] - point[:-1] ** 2)
            return gradient
        if self.variant == "constrained":
            gradient = 4.0 * point
            gradient[0] -= 1.0
            return gradient
        if self.target is None:
            raise ValueError("proximal problem is missing its target")
        return point - self.target

    def equality(self, value: np.ndarray) -> float:
        if self.variant != "constrained":
            raise ValueError("equality is defined only for constrained optimization")
        point = np.asarray(value, dtype=np.float64)
        return float(point @ point - 1.0)

    def inequality(self, value: np.ndarray) -> float:
        if self.variant != "constrained":
            raise ValueError("inequality is defined only for constrained optimization")
        point = np.asarray(value, dtype=np.float64)
        return float(point[0] + point[1] - 2.0)

    def proximal_stationarity(self, value: np.ndarray) -> np.ndarray:
        if self.variant != "proximal" or self.target is None:
            raise ValueError(
                "proximal_stationarity is defined only for proximal optimization"
            )
        point = np.asarray(value, dtype=np.float64)
        trial = point - self.gradient(point)
        prox = np.sign(trial) * np.maximum(
            np.abs(trial) - self.l1_weight,
            0.0,
        )
        return point - prox

    def identity(self) -> dict[str, Any]:
        arrays = [self.initial, self.optimum]
        if self.target is not None:
            arrays.append(self.target)
        return _identity(
            family="optimization",
            name=self.name,
            variant=self.variant,
            seed=self.seed,
            dtype=self.initial.dtype,
            parameters={
                "construction": {
                    "unconstrained": "nonquadratic Rosenbrock chain",
                    "constrained": "Maratos-style circle equality with affine inequality",
                    "proximal": "separable quadratic plus L1",
                }[self.variant],
                "l1_weight": self.l1_weight,
            },
            arrays=tuple(arrays),
        )

    def sizes(self) -> dict[str, int]:
        dimension = int(self.initial.size)
        return {
            "dimension": dimension,
            "rows": dimension,
            "columns": dimension,
            "nnz": dimension,
            "block_size": 1,
            "right_hand_sides": 1,
        }


@dataclass(frozen=True)
class MathematicalProgramProblem:
    """Deterministic LP, QP, or SOCP benchmark with an independent reference."""

    name: str
    variant: Literal["lp", "qp", "socp"]
    seed: int
    quadratic: np.ndarray | None
    linear: np.ndarray
    equality_matrix: np.ndarray
    equality_rhs: np.ndarray
    inequality_matrix: np.ndarray
    inequality_rhs: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    optimum: np.ndarray
    conic_matrix: np.ndarray | None = None
    conic_rhs: np.ndarray | None = None
    cone_blocks: tuple[tuple[str, int], ...] = ()

    @property
    def capability(self) -> Capability:
        if self.variant == "lp":
            return "optimization.linear-program"
        if self.variant == "qp":
            return "optimization.quadratic-program"
        return "optimization.conic-program"

    def objective(self, value: np.ndarray) -> float:
        point = np.asarray(value, dtype=np.float64)
        quadratic = (
            0.0 if self.quadratic is None else 0.5 * float(point @ self.quadratic @ point)
        )
        return quadratic + float(self.linear @ point)

    def gradient(self, value: np.ndarray) -> np.ndarray:
        point = np.asarray(value, dtype=np.float64)
        return (
            self.linear
            if self.quadratic is None
            else self.quadratic @ point + self.linear
        )

    def identity(self) -> dict[str, Any]:
        arrays = [
            self.linear,
            self.equality_matrix,
            self.equality_rhs,
            self.inequality_matrix,
            self.inequality_rhs,
            self.lower,
            self.upper,
            self.optimum,
        ]
        if self.quadratic is not None:
            arrays.append(self.quadratic)
        if self.conic_matrix is not None and self.conic_rhs is not None:
            arrays.extend((self.conic_matrix, self.conic_rhs))
        return _identity(
            family="optimization-program",
            name=self.name,
            variant=self.variant,
            seed=self.seed,
            dtype=self.linear.dtype,
            parameters={
                "construction": {
                    "lp": "separable bounded linear program",
                    "qp": "diagonal positive-definite bounded quadratic program",
                    "socp": "active Lorentz-cone quadratic program",
                }[self.variant],
                "cone_blocks": list(self.cone_blocks),
            },
            arrays=tuple(arrays),
        )

    def sizes(self) -> dict[str, int]:
        dimension = int(self.linear.size)
        rows = (
            int(self.conic_matrix.shape[0])
            if self.conic_matrix is not None
            else int(self.equality_matrix.shape[0] + self.inequality_matrix.shape[0])
        )
        matrices = [self.equality_matrix, self.inequality_matrix]
        if self.quadratic is not None:
            matrices.append(self.quadratic)
        if self.conic_matrix is not None:
            matrices.append(self.conic_matrix)
        return {
            "dimension": dimension,
            "rows": max(1, rows),
            "columns": dimension,
            "nnz": int(
                np.count_nonzero(self.linear)
                + sum(np.count_nonzero(value) for value in matrices)
            ),
            "block_size": 1,
            "right_hand_sides": 1,
        }


BenchmarkProblem = (
    SparseLinearProblem
    | NonlinearProblem
    | GeneralEigenProblem
    | ContinuationProblem
    | OptimizationProblem
    | MathematicalProgramProblem
)


def sparse_scalar_linear(
    *, size: int, right_hand_sides: int, seed: int
) -> SparseLinearProblem:
    """Create a deterministic scalar 1-D Poisson system in coordinate form."""
    _validate_sizes(size, right_hand_sides)
    rows, columns, coefficients = _block_poisson_coordinates(size, block_size=1)
    rhs = _random_rhs(size, right_hand_sides, seed)
    return SparseLinearProblem(
        name="poisson-1d",
        variant="scalar-sparse-spd",
        seed=seed,
        rows=rows,
        columns=columns,
        coefficients=coefficients,
        rhs=rhs,
        dimension=size,
        block_size=1,
    )


def sparse_block_linear(
    *, block_count: int, block_size: int, right_hand_sides: int, seed: int
) -> SparseLinearProblem:
    """Create a deterministic block-tridiagonal SPD Poisson system."""
    _validate_sizes(block_count, right_hand_sides)
    if block_size < 2:
        raise ValueError("block_size must be at least two for a block problem")
    rows, columns, coefficients = _block_poisson_coordinates(
        block_count,
        block_size=block_size,
    )
    dimension = block_count * block_size
    rhs = _random_rhs(dimension, right_hand_sides, seed)
    return SparseLinearProblem(
        name="block-poisson-1d",
        variant="block-sparse-spd",
        seed=seed,
        rows=rows,
        columns=columns,
        coefficients=coefficients,
        rhs=rhs,
        dimension=dimension,
        block_size=block_size,
    )


def nonlinear_root(*, size: int, seed: int) -> NonlinearProblem:
    """Create a deterministic positive separable nonlinear root problem."""
    if size < 1:
        raise ValueError("size must be positive")
    generator = np.random.Generator(np.random.PCG64(seed))
    target = np.linspace(0.5, 1.5, size, dtype=np.float64)
    target += 0.05 * generator.uniform(size=size)
    initial = 0.75 * np.sqrt(target)
    return NonlinearProblem(
        name="positive-square-root",
        variant="root",
        seed=seed,
        initial=initial,
        target=target,
    )


def semilinear_poisson_root(*, size: int, seed: int) -> NonlinearProblem:
    """Create a sparse monotone semilinear 1-D Poisson root problem."""
    if size < 2:
        raise ValueError("semilinear Poisson size must be at least two")
    generator = np.random.Generator(np.random.PCG64(seed))
    spacing = 1.0 / (size + 1)
    coordinates = np.arange(1, size + 1, dtype=np.float64) * spacing
    exact = np.sin(np.pi * coordinates)
    nonlinearity = float(0.75 + 0.5 * generator.uniform())
    extended = np.pad(exact, (1, 1))
    target = (
        2.0 * exact - extended[:-2] - extended[2:]
    ) / spacing**2 + nonlinearity * exact**3
    return NonlinearProblem(
        name="semilinear-poisson-1d",
        variant="root",
        seed=seed,
        initial=np.zeros_like(exact),
        target=target,
        root_kind="semilinear-poisson-1d",
        nonlinearity=nonlinearity,
        grid_spacing=spacing,
    )


def variational_inequality(*, size: int, seed: int) -> NonlinearProblem:
    """Create a deterministic box VI with both active and inactive variables."""
    if size < 2:
        raise ValueError("VI size must be at least two")
    generator = np.random.Generator(np.random.PCG64(seed))
    diagonal = np.linspace(1.0, 2.0, size, dtype=np.float64)
    magnitude = 0.5 + generator.uniform(size=size)
    signs = np.where(np.arange(size) % 2 == 0, 1.0, -1.0)
    target = signs * magnitude
    lower = np.zeros(size, dtype=np.float64)
    upper = np.full(size, np.inf, dtype=np.float64)
    initial = np.maximum(target / diagonal, 0.1)
    return NonlinearProblem(
        name="diagonal-complementarity",
        variant="vi",
        seed=seed,
        initial=initial,
        target=target,
        lower=lower,
        upper=upper,
        diagonal=diagonal,
    )


def general_eigenproblem(*, size: int, eigenpairs: int, seed: int) -> GeneralEigenProblem:
    """Create a deterministic real nonsymmetric tridiagonal eigenproblem."""
    if size < 4:
        raise ValueError("general eigenproblem size must be at least four")
    if eigenpairs < 1 or eigenpairs >= size - 1:
        raise ValueError("eigenpairs must lie in [1, size - 2] for ARPACK comparability")
    generator = np.random.Generator(np.random.PCG64(seed))
    diagonal = np.linspace(-1.5, 2.0, size, dtype=np.float64)
    diagonal += 0.01 * generator.standard_normal(size)
    matrix = np.diag(diagonal)
    matrix += np.diag(np.full(size - 1, 0.3, dtype=np.float64), 1)
    matrix += np.diag(np.full(size - 1, -0.07, dtype=np.float64), -1)
    return GeneralEigenProblem(
        name="nonsymmetric-tridiagonal",
        variant="standard-largest-magnitude",
        seed=seed,
        matrix=matrix,
        eigenpairs=eigenpairs,
    )


def quadratic_fold(*, seed: int, max_points: int = 36) -> ContinuationProblem:
    """Create the canonical deterministic quadratic-fold continuation problem."""
    if max_points < 4:
        raise ValueError("max_points must be at least four")
    return ContinuationProblem(
        name="quadratic-fold",
        variant="pseudo-arclength",
        seed=seed,
        initial_state=np.asarray(1.0, dtype=np.float64),
        initial_coordinate=1.0,
        direction=-1,
        initial_step=0.08,
        min_step=0.01,
        max_step=0.08,
        max_points=max_points,
    )


def rosenbrock_optimization(*, size: int, seed: int) -> OptimizationProblem:
    if size < 2:
        raise ValueError("Rosenbrock dimension must be at least two")
    initial = np.ones(size, dtype=np.float64)
    initial[::2] = -1.2
    return OptimizationProblem(
        name="rosenbrock-chain",
        variant="unconstrained",
        seed=seed,
        initial=initial,
        optimum=np.ones(size, dtype=np.float64),
    )


def maratos_constrained_optimization(*, seed: int) -> OptimizationProblem:
    return OptimizationProblem(
        name="maratos-circle",
        variant="constrained",
        seed=seed,
        initial=np.asarray([0.7, 0.7], dtype=np.float64),
        optimum=np.asarray([1.0, 0.0], dtype=np.float64),
    )


def l1_composite_optimization(*, size: int, seed: int) -> OptimizationProblem:
    if size < 1:
        raise ValueError("proximal dimension must be positive")
    generator = np.random.Generator(np.random.PCG64(seed))
    target = generator.standard_normal(size)
    weight = 0.35
    optimum = np.sign(target) * np.maximum(np.abs(target) - weight, 0.0)
    return OptimizationProblem(
        name="quadratic-l1",
        variant="proximal",
        seed=seed,
        initial=np.zeros(size, dtype=np.float64),
        optimum=optimum,
        target=target,
        l1_weight=weight,
    )


def bounded_linear_program(*, size: int, seed: int) -> MathematicalProgramProblem:
    generator = np.random.Generator(np.random.PCG64(seed))
    linear = generator.standard_normal(size)
    lower = np.zeros(size)
    upper = np.ones(size)
    optimum = np.where(linear < 0.0, upper, lower)
    empty_matrix = np.empty((0, size))
    empty_rhs = np.empty((0,))
    return MathematicalProgramProblem(
        name="bounded-separable-lp",
        variant="lp",
        seed=seed,
        quadratic=None,
        linear=linear,
        equality_matrix=empty_matrix,
        equality_rhs=empty_rhs,
        inequality_matrix=empty_matrix,
        inequality_rhs=empty_rhs,
        lower=lower,
        upper=upper,
        optimum=optimum,
    )


def bounded_quadratic_program(*, size: int, seed: int) -> MathematicalProgramProblem:
    generator = np.random.Generator(np.random.PCG64(seed))
    diagonal = 0.5 + generator.random(size)
    linear = generator.standard_normal(size)
    lower = np.zeros(size)
    upper = np.ones(size)
    optimum = np.clip(-linear / diagonal, lower, upper)
    empty_matrix = np.empty((0, size))
    empty_rhs = np.empty((0,))
    return MathematicalProgramProblem(
        name="bounded-diagonal-qp",
        variant="qp",
        seed=seed,
        quadratic=np.diag(diagonal),
        linear=linear,
        equality_matrix=empty_matrix,
        equality_rhs=empty_rhs,
        inequality_matrix=empty_matrix,
        inequality_rhs=empty_rhs,
        lower=lower,
        upper=upper,
        optimum=optimum,
    )


def active_second_order_cone_program(*, seed: int) -> MathematicalProgramProblem:
    empty_matrix = np.empty((0, 2))
    empty_rhs = np.empty((0,))
    return MathematicalProgramProblem(
        name="active-socp",
        variant="socp",
        seed=seed,
        quadratic=np.eye(2),
        linear=np.asarray([-2.0, 0.0]),
        equality_matrix=empty_matrix,
        equality_rhs=empty_rhs,
        inequality_matrix=empty_matrix,
        inequality_rhs=empty_rhs,
        lower=np.full(2, -np.inf),
        upper=np.full(2, np.inf),
        optimum=np.asarray([1.0, 0.0]),
        conic_matrix=np.asarray([[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
        conic_rhs=np.asarray([1.0, 0.0, 0.0]),
        cone_blocks=(("soc", 3),),
    )


def default_problems(*, size: int, seed: int) -> dict[str, BenchmarkProblem]:
    """Return the common deterministic cross-adapter problem campaign."""
    if size < 8:
        raise ValueError("default campaign size must be at least eight")
    eigenpairs = min(4, size - 2)
    return {
        "linear-scalar": sparse_scalar_linear(
            size=size,
            right_hand_sides=1,
            seed=seed,
        ),
        "linear-block": sparse_block_linear(
            block_count=max(4, size // 2),
            block_size=2,
            right_hand_sides=2,
            seed=seed + 1,
        ),
        "nonlinear-root": nonlinear_root(size=size, seed=seed + 2),
        "nonlinear-root-dense": nonlinear_root(size=size, seed=seed + 2),
        "nonlinear-root-matrix-free": nonlinear_root(size=size, seed=seed + 2),
        "nonlinear-root-sparse-pde": semilinear_poisson_root(
            size=size,
            seed=seed + 9,
        ),
        "nonlinear-vi": variational_inequality(size=size, seed=seed + 3),
        "general-eigen": general_eigenproblem(
            size=size,
            eigenpairs=eigenpairs,
            seed=seed + 4,
        ),
        "continuation-fold": quadratic_fold(seed=seed + 5),
        "optimization-unconstrained": rosenbrock_optimization(
            size=size,
            seed=seed + 6,
        ),
        "optimization-constrained": maratos_constrained_optimization(
            seed=seed + 7,
        ),
        "optimization-proximal": l1_composite_optimization(
            size=size,
            seed=seed + 8,
        ),
        "optimization-linear-program": bounded_linear_program(
            size=size,
            seed=seed + 10,
        ),
        "optimization-quadratic-program": bounded_quadratic_program(
            size=size,
            seed=seed + 11,
        ),
        "optimization-conic-program": active_second_order_cone_program(
            seed=seed + 12,
        ),
    }


def _block_poisson_coordinates(
    block_count: int,
    *,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = block_count * block_size
    diagonal_block = 4.0 * np.eye(block_size, dtype=np.float64)
    if block_size > 1:
        coupling = -0.25 * (
            np.eye(block_size, k=1, dtype=np.float64)
            + np.eye(block_size, k=-1, dtype=np.float64)
        )
        diagonal_block += coupling
    off_block = -np.eye(block_size, dtype=np.float64)
    dense = np.zeros((dimension, dimension), dtype=np.float64)
    for block in range(block_count):
        start = block * block_size
        stop = start + block_size
        dense[start:stop, start:stop] = diagonal_block
        if block + 1 < block_count:
            next_start = stop
            next_stop = next_start + block_size
            dense[start:stop, next_start:next_stop] = off_block
            dense[next_start:next_stop, start:stop] = off_block
    rows, columns = np.nonzero(dense)
    return (
        rows.astype(np.int64),
        columns.astype(np.int64),
        dense[rows, columns],
    )


def _random_rhs(dimension: int, count: int, seed: int) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(seed))
    rhs = generator.standard_normal((dimension, count), dtype=np.float64)
    return rhs[:, 0] if count == 1 else rhs


def _identity(
    *,
    family: str,
    name: str,
    variant: str,
    seed: int,
    dtype: np.dtype[Any],
    parameters: dict[str, Any],
    arrays: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    array_hash = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        array_hash.update(str(contiguous.dtype).encode("ascii"))
        array_hash.update(str(contiguous.shape).encode("ascii"))
        array_hash.update(contiguous.tobytes(order="C"))
    evidence = {
        "family": family,
        "name": name,
        "variant": variant,
        "seed": seed,
        "dtype": str(dtype),
        "parameters": parameters,
        "array_sha256": array_hash.hexdigest(),
    }
    return {
        "family": family,
        "name": name,
        "variant": variant,
        "seed": seed,
        "dtype": str(dtype),
        "fingerprint": stable_fingerprint(evidence),
        "parameters": parameters,
    }


def _validate_sizes(size: int, right_hand_sides: int) -> None:
    if size < 1 or right_hand_sides < 1:
        raise ValueError("size and right_hand_sides must be positive")


__all__ = [
    "BenchmarkProblem",
    "Capability",
    "ContinuationProblem",
    "GeneralEigenProblem",
    "NonlinearProblem",
    "OptimizationProblem",
    "SparseLinearProblem",
    "default_problems",
    "general_eigenproblem",
    "l1_composite_optimization",
    "maratos_constrained_optimization",
    "nonlinear_root",
    "semilinear_poisson_root",
    "quadratic_fold",
    "rosenbrock_optimization",
    "sparse_block_linear",
    "sparse_scalar_linear",
    "variational_inequality",
]
