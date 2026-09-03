#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg._operators import DenseLinearOperator
from ..linalg.eigen import (
    DenseSchurQZ,
    general_eigensolve,
    GeneralEigenproblem,
    GeneralEigenResourcePolicy,
    GeneralEigenSelection,
    GeneralEigenSolvePolicy,
)
from ._precision import TensorNetworkPrecisionPolicy


class UniformMatrixProductState(StrictModule):
    """Immutable finite-unit-cell uniform MPS with periodic virtual bonds."""

    tensors: tuple[Array, ...]
    precision: TensorNetworkPrecisionPolicy
    unit_cell_size: int = eqx.field(static=True)
    physical_dimensions: tuple[int, ...] = eqx.field(static=True)
    bond_dimension: int = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        tensors: Sequence[ArrayLike],
        /,
        *,
        precision: TensorNetworkPrecisionPolicy | None = None,
    ):
        precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, TensorNetworkPrecisionPolicy):
            raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
        values = tuple(precision_.storage(jnp.asarray(value)) for value in tensors)
        if not values or any(value.ndim != 3 for value in values):
            raise ValueError("Uniform MPS tensors require (left, physical, right).")
        bond = int(values[0].shape[0])
        if bond < 1 or any(
            value.shape[0] != bond or value.shape[-1] != bond for value in values
        ):
            raise ValueError(
                "Uniform MPS unit-cell virtual bonds must share one dimension."
            )
        precision_.validate_storage(values)
        self.tensors = values
        self.precision = precision_
        self.unit_cell_size = len(values)
        self.physical_dimensions = tuple(int(value.shape[1]) for value in values)
        self.bond_dimension = bond
        self.structure_id = canonical_fingerprint(
            {
                "kind": "uniform-matrix-product-state",
                "shapes": tuple(
                    tuple(int(size) for size in value.shape) for value in values
                ),
                "dtype": str(values[0].dtype),
                "precision": precision_.policy_id,
            }
        )


class UniformMatrixProductOperator(StrictModule):
    """Immutable finite-unit-cell uniform MPO with periodic virtual bonds."""

    tensors: tuple[Array, ...]
    precision: TensorNetworkPrecisionPolicy
    unit_cell_size: int = eqx.field(static=True)
    output_dimensions: tuple[int, ...] = eqx.field(static=True)
    input_dimensions: tuple[int, ...] = eqx.field(static=True)
    bond_dimension: int = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        tensors: Sequence[ArrayLike],
        /,
        *,
        precision: TensorNetworkPrecisionPolicy | None = None,
    ):
        precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, TensorNetworkPrecisionPolicy):
            raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
        values = tuple(precision_.storage(jnp.asarray(value)) for value in tensors)
        if not values or any(value.ndim != 4 for value in values):
            raise ValueError("Uniform MPO tensors require (left, output, input, right).")
        bond = int(values[0].shape[0])
        if bond < 1 or any(
            value.shape[0] != bond or value.shape[-1] != bond for value in values
        ):
            raise ValueError(
                "Uniform MPO unit-cell virtual bonds must share one dimension."
            )
        precision_.validate_storage(values)
        self.tensors = values
        self.precision = precision_
        self.unit_cell_size = len(values)
        self.output_dimensions = tuple(int(value.shape[1]) for value in values)
        self.input_dimensions = tuple(int(value.shape[2]) for value in values)
        self.bond_dimension = bond
        self.structure_id = canonical_fingerprint(
            {
                "kind": "uniform-matrix-product-operator",
                "shapes": tuple(
                    tuple(int(size) for size in value.shape) for value in values
                ),
                "dtype": str(values[0].dtype),
                "precision": precision_.policy_id,
            }
        )


class UniformTransferStatus(IntEnum):
    SUCCESS = 0
    NONINJECTIVE = 1
    NONFINITE = 2
    EIGENSOLVE_FAILED = 3


class UniformTransferPolicy(StrictModule):
    maximum_modes: int = eqx.field(static=True)
    injectivity_tolerance: float = eqx.field(static=True)
    positivity_tolerance: float = eqx.field(static=True)
    maximum_transfer_elements: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_modes: int = 4,
        injectivity_tolerance: float = 1e-8,
        positivity_tolerance: float = 1e-8,
        maximum_transfer_elements: int = 10_000_000,
    ):
        modes = int(maximum_modes)
        elements = int(maximum_transfer_elements)
        tolerances = (float(injectivity_tolerance), float(positivity_tolerance))
        if modes < 1 or elements < 1:
            raise ValueError("Uniform transfer capacities must be positive.")
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError(
                "Uniform transfer tolerances must be finite and nonnegative."
            )
        self.maximum_modes = modes
        self.injectivity_tolerance = tolerances[0]
        self.positivity_tolerance = tolerances[1]
        self.maximum_transfer_elements = elements
        self.policy_id = canonical_fingerprint(
            {
                "kind": "uniform-transfer-policy",
                "modes": modes,
                "injectivity_tolerance": tolerances[0],
                "positivity_tolerance": tolerances[1],
                "maximum_transfer_elements": elements,
            }
        )


class UniformTransferFixedPoints(StrictModule):
    left: Array
    right: Array
    eigenvalues: Array
    active_modes: Array
    dominant_residual: Array
    injectivity_gap: Array
    left_positivity_floor: Array
    right_positivity_floor: Array
    status: Array
    state_structure_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(UniformTransferStatus.SUCCESS)

    @property
    def injective(self) -> Array:
        return self.successful


def uniform_cell_transfer_matrix(state: UniformMatrixProductState, /) -> Array:
    if not isinstance(state, UniformMatrixProductState):
        raise TypeError("state must be UniformMatrixProductState.")
    bond = state.bond_dimension
    transfer = jnp.eye(bond * bond, dtype=state.tensors[0].dtype)
    for tensor in state.precision.accumulation(state.tensors):
        local = ein.contract("apr,bps->abrs", jnp.conj(tensor), tensor).reshape(
            (bond * bond, bond * bond)
        )
        transfer = transfer @ local
    return state.precision.accumulation(transfer)


def uniform_transfer_fixed_points(
    state: UniformMatrixProductState,
    policy: UniformTransferPolicy | None = None,
    /,
) -> UniformTransferFixedPoints:
    selected = UniformTransferPolicy() if policy is None else policy
    if not isinstance(selected, UniformTransferPolicy):
        raise TypeError("policy must be UniformTransferPolicy or None.")
    transfer = uniform_cell_transfer_matrix(state)
    if int(transfer.size) > selected.maximum_transfer_elements:
        raise MemoryError("Uniform transfer matrix exceeds maximum_transfer_elements.")
    dimension = int(transfer.shape[0])
    count = min(selected.maximum_modes, dimension)
    eigen_policy = GeneralEigenSolvePolicy(
        DenseSchurQZ(),
        selection=GeneralEigenSelection("largest-magnitude", count=count),
        resources=GeneralEigenResourcePolicy(max_dimension=dimension),
    )
    solve = general_eigensolve(
        GeneralEigenproblem(
            DenseLinearOperator(transfer), problem_id="uniform-cell-transfer"
        ),
        policy=eigen_policy,
    )
    complex_dtype = jnp.result_type(transfer, jnp.complex64)
    eigenvalues = jnp.full((selected.maximum_modes,), jnp.nan + 0j, dtype=complex_dtype)
    active = jnp.arange(selected.maximum_modes) < count
    eigenvalues = eigenvalues.at[:count].set(solve.eigenvalues[:count])
    right_vector = solve.right_eigenvector_coordinates[:, 0]
    left_vector = solve.left_eigenvector_coordinates[:, 0]
    bond = state.bond_dimension
    right = right_vector.reshape((bond, bond))
    left = left_vector.reshape((bond, bond))
    right = 0.5 * (right + jnp.conj(right.T))
    left = 0.5 * (left + jnp.conj(left.T))
    right_phase = jnp.where(jnp.real(jnp.trace(right)) < 0.0, -1.0, 1.0)
    left_phase = jnp.where(jnp.real(jnp.trace(left)) < 0.0, -1.0, 1.0)
    right = right_phase * right
    left = left_phase * left
    pairing = jnp.real(jnp.trace(left @ right))
    scale = jnp.sqrt(jnp.maximum(jnp.abs(pairing), jnp.finfo(right.real.dtype).tiny))
    left = left / scale
    right = right / scale
    dominant = solve.eigenvalues[0]
    residual = jnp.linalg.norm(
        transfer @ right_vector - dominant * right_vector
    ) / jnp.maximum(jnp.linalg.norm(right_vector), 1.0)
    if count > 1:
        gap = (jnp.abs(dominant) - jnp.abs(solve.eigenvalues[1])) / jnp.maximum(
            jnp.abs(dominant), jnp.finfo(right.real.dtype).tiny
        )
    else:
        gap = jnp.asarray(1.0, dtype=right.real.dtype)
    left_floor = jnp.min(jnp.linalg.eigvalsh(left))
    right_floor = jnp.min(jnp.linalg.eigvalsh(right))
    finite = (
        jnp.all(jnp.isfinite(eigenvalues[:count]))
        & jnp.all(jnp.isfinite(left))
        & jnp.all(jnp.isfinite(right))
        & jnp.isfinite(residual)
    )
    noninjective = (
        (gap <= selected.injectivity_tolerance)
        | (left_floor < -selected.positivity_tolerance)
        | (right_floor < -selected.positivity_tolerance)
    )
    status = jnp.where(
        ~solve.successful,
        int(UniformTransferStatus.EIGENSOLVE_FAILED),
        jnp.where(
            ~finite,
            int(UniformTransferStatus.NONFINITE),
            jnp.where(
                noninjective,
                int(UniformTransferStatus.NONINJECTIVE),
                int(UniformTransferStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    return UniformTransferFixedPoints(
        state.precision.output(left),
        state.precision.output(right),
        state.precision.output(eigenvalues),
        active,
        residual,
        gap,
        left_floor,
        right_floor,
        status,
        state.structure_id,
        selected.policy_id,
    )


def uniform_correlation_length(
    fixed_points: UniformTransferFixedPoints, unit_cell_size: int, /
) -> Array:
    if not isinstance(fixed_points, UniformTransferFixedPoints):
        raise TypeError("fixed_points must be UniformTransferFixedPoints.")
    cell = int(unit_cell_size)
    if cell < 1:
        raise ValueError("unit_cell_size must be positive.")
    second_active = fixed_points.active_modes.shape[0] > 1 and bool(
        fixed_points.active_modes[1]
    )
    if not second_active:
        return jnp.asarray(0.0, dtype=fixed_points.dominant_residual.dtype)
    ratio = jnp.abs(fixed_points.eigenvalues[1] / fixed_points.eigenvalues[0])
    return jnp.where(
        ratio >= 1.0,
        jnp.asarray(jnp.inf, dtype=ratio.dtype),
        -float(cell) / jnp.log(jnp.maximum(ratio, jnp.finfo(ratio.dtype).tiny)),
    )


__all__ = [
    "UniformMatrixProductOperator",
    "UniformMatrixProductState",
    "UniformTransferFixedPoints",
    "UniformTransferPolicy",
    "UniformTransferStatus",
    "uniform_cell_transfer_matrix",
    "uniform_correlation_length",
    "uniform_transfer_fixed_points",
]
