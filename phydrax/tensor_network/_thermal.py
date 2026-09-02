#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._core import LocallyPurifiedDensity, MatrixProductOperator, MatrixProductState
from ._environments import mpo_hermiticity_residual, mps_mpo_expectation
from ._mpo import apply_mpo, compress_mps


class FiniteThermalStatus(IntEnum):
    SUCCESS = 0
    INVALID_HAMILTONIAN = 1
    NONFINITE = 2
    MAXIMUM_ORDER_REACHED = 3


class FiniteThermalPolicy(StrictModule):
    maximum_bond_dimension: int = eqx.field(static=True)
    maximum_order: int = eqx.field(static=True)
    term_tolerance: float = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    maximum_history_elements: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_bond_dimension: int,
        maximum_order: int = 16,
        term_tolerance: float = 1e-9,
        hermiticity_tolerance: float = 1e-9,
        maximum_history_elements: int = 1_000_000,
    ):
        bond = int(maximum_bond_dimension)
        order = int(maximum_order)
        history = int(maximum_history_elements)
        tolerances = (float(term_tolerance), float(hermiticity_tolerance))
        if bond < 1 or order < 1 or history < 3 * order:
            raise ValueError("Thermal bond/order/history capacities are insufficient.")
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Thermal tolerances must be finite and nonnegative.")
        self.maximum_bond_dimension = bond
        self.maximum_order = order
        self.term_tolerance = tolerances[0]
        self.hermiticity_tolerance = tolerances[1]
        self.maximum_history_elements = history
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-thermal-policy",
                "bond": bond,
                "order": order,
                "term_tolerance": tolerances[0],
                "hermiticity_tolerance": tolerances[1],
            }
        )


class FiniteThermalEvidence(StrictModule):
    term_norm_history: Array
    discarded_weight_history: Array
    active_orders: Array
    beta: Array
    raw_trace: Array
    normalized_trace: Array
    hermiticity_residual: Array
    status: Array

    @property
    def successful(self) -> Array:
        return self.status == int(FiniteThermalStatus.SUCCESS)


class FiniteThermalResult(StrictModule):
    purification: LocallyPurifiedDensity
    evidence: FiniteThermalEvidence
    policy_id: str = eqx.field(static=True)


def infinite_temperature_purification(
    physical_dimensions: tuple[int, ...],
    /,
    *,
    precision=None,
) -> LocallyPurifiedDensity:
    """Return the normalized beta=0 product purification of the identity."""
    dimensions = tuple(int(value) for value in physical_dimensions)
    if not dimensions or any(value < 1 for value in dimensions):
        raise ValueError("physical_dimensions must be nonempty and positive.")
    tensors = []
    for dimension in dimensions:
        local = jnp.eye(dimension, dtype=jnp.complex128) / jnp.sqrt(float(dimension))
        tensors.append(local[None, :, :, None])
    return LocallyPurifiedDensity(tuple(tensors), precision=precision)


def _purification_mps(state: LocallyPurifiedDensity, /) -> MatrixProductState:
    tensors = tuple(
        tensor.reshape(
            (tensor.shape[0], tensor.shape[1] * tensor.shape[2], tensor.shape[3])
        )
        for tensor in state.tensors
    )
    return MatrixProductState(tensors, precision=state.precision)


def _mps_purification(
    state: MatrixProductState,
    physical_dimensions: tuple[int, ...],
    /,
) -> LocallyPurifiedDensity:
    tensors = []
    for tensor, dimension in zip(state.tensors, physical_dimensions, strict=True):
        if tensor.shape[1] != dimension * dimension:
            raise ValueError("Purification MPS physical dimension is inconsistent.")
        tensors.append(
            tensor.reshape((tensor.shape[0], dimension, dimension, tensor.shape[-1]))
        )
    return LocallyPurifiedDensity(tuple(tensors), precision=state.precision)


def _lift_physical_mpo(operator: MatrixProductOperator, /) -> MatrixProductOperator:
    tensors = []
    for tensor, dimension in zip(
        operator.tensors, operator.input_dimensions, strict=True
    ):
        identity = jnp.eye(dimension, dtype=tensor.dtype)
        lifted = oe.contract("apqb,kl->apkqlb", tensor, identity)
        tensors.append(
            lifted.reshape(
                (
                    tensor.shape[0],
                    tensor.shape[1] * dimension,
                    tensor.shape[2] * dimension,
                    tensor.shape[-1],
                )
            )
        )
    return MatrixProductOperator(tuple(tensors), precision=operator.precision)


def _scale_mps(state: MatrixProductState, coefficient: Array, /) -> MatrixProductState:
    return MatrixProductState(
        (state.tensors[0] * coefficient,) + state.tensors[1:], precision=state.precision
    )


def _add_mps(
    left: MatrixProductState, right: MatrixProductState, /
) -> MatrixProductState:
    if left.physical_dimensions != right.physical_dimensions:
        raise ValueError("MPS physical dimensions must match for addition.")
    if left.precision.policy_id != right.precision.policy_id:
        raise ValueError("MPS precision policies must match for addition.")
    if left.site_count == 1:
        return MatrixProductState(
            (left.tensors[0] + right.tensors[0],), precision=left.precision
        )
    tensors = [jnp.concatenate((left.tensors[0], right.tensors[0]), axis=-1)]
    for first, second in zip(left.tensors[1:-1], right.tensors[1:-1], strict=True):
        shape = (
            first.shape[0] + second.shape[0],
            first.shape[1],
            first.shape[-1] + second.shape[-1],
        )
        block = jnp.zeros(shape, dtype=jnp.result_type(first, second))
        block = block.at[: first.shape[0], :, : first.shape[-1]].set(first)
        block = block.at[first.shape[0] :, :, first.shape[-1] :].set(second)
        tensors.append(block)
    tensors.append(jnp.concatenate((left.tensors[-1], right.tensors[-1]), axis=0))
    return MatrixProductState(tuple(tensors), precision=left.precision)


def finite_temperature_purification(
    hamiltonian: MatrixProductOperator,
    beta: float,
    policy: FiniteThermalPolicy,
    /,
) -> FiniteThermalResult:
    """Apply a bounded Taylor imaginary-time filter to the beta=0 purification."""
    if not isinstance(hamiltonian, MatrixProductOperator):
        raise TypeError("hamiltonian must be MatrixProductOperator.")
    if not isinstance(policy, FiniteThermalPolicy):
        raise TypeError("policy must be FiniteThermalPolicy.")
    beta_ = float(beta)
    if not isfinite(beta_) or beta_ < 0.0:
        raise ValueError("beta must be finite and nonnegative.")
    if hamiltonian.output_dimensions != hamiltonian.input_dimensions:
        raise ValueError("Thermal purification requires a square Hamiltonian MPO.")
    hermiticity = mpo_hermiticity_residual(hamiltonian)
    initial = infinite_temperature_purification(
        hamiltonian.input_dimensions, precision=hamiltonian.precision
    )
    state = _purification_mps(initial)
    term = state
    lifted = _lift_physical_mpo(hamiltonian)
    real_dtype = state.tensors[0].real.dtype
    term_norms = jnp.full((policy.maximum_order,), jnp.nan, dtype=real_dtype)
    discarded = jnp.full((policy.maximum_order,), jnp.nan, dtype=real_dtype)
    active = jnp.zeros((policy.maximum_order,), dtype=bool)
    status = (
        FiniteThermalStatus.SUCCESS
        if beta_ == 0.0
        else FiniteThermalStatus.MAXIMUM_ORDER_REACHED
    )
    if float(hermiticity) > policy.hermiticity_tolerance or not bool(
        jnp.isfinite(hermiticity)
    ):
        status = FiniteThermalStatus.INVALID_HAMILTONIAN
    elif beta_ > 0.0:
        for order in range(1, policy.maximum_order + 1):
            applied, apply_evidence = apply_mpo(
                lifted,
                term,
                maximum_bond_dimension=policy.maximum_bond_dimension,
                normalize=False,
            )
            term = _scale_mps(
                applied, jnp.asarray(-0.5 * beta_ / order, dtype=real_dtype)
            )
            combined = _add_mps(state, term)
            state, sum_evidence = compress_mps(
                combined,
                maximum_bond_dimension=policy.maximum_bond_dimension,
                normalize=False,
            )
            term_norm = term.norm()
            loss = (
                apply_evidence.accumulated_discarded_weight
                + sum_evidence.accumulated_discarded_weight
            )
            term_norms = term_norms.at[order - 1].set(term_norm)
            discarded = discarded.at[order - 1].set(loss)
            active = active.at[order - 1].set(True)
            if not bool(jnp.isfinite(term_norm) & jnp.isfinite(loss)):
                status = FiniteThermalStatus.NONFINITE
                break
            if float(term_norm) <= policy.term_tolerance:
                status = FiniteThermalStatus.SUCCESS
                break
    purification = _mps_purification(state, hamiltonian.input_dimensions)
    raw_trace = purification.raw_trace()
    purification = purification.normalized()
    normalized_trace = purification.raw_trace()
    evidence = FiniteThermalEvidence(
        term_norms,
        discarded,
        active,
        jnp.asarray(beta_, dtype=real_dtype),
        raw_trace,
        normalized_trace,
        hermiticity,
        jnp.asarray(int(status), dtype=jnp.int32),
    )
    return FiniteThermalResult(purification, evidence, policy.policy_id)


def thermal_mpo_expectation(
    purification: LocallyPurifiedDensity,
    operator: MatrixProductOperator,
    /,
) -> Array:
    """Compute Tr(rho O)/Tr(rho) directly in the purification representation."""
    if purification.physical_dimensions != operator.input_dimensions:
        raise ValueError("Purification and observable dimensions must match.")
    amplitude = _purification_mps(purification)
    lifted = _lift_physical_mpo(operator)
    numerator = mps_mpo_expectation(amplitude, lifted)
    denominator = purification.raw_trace()
    denominator = eqx.error_if(
        denominator,
        ~jnp.isfinite(denominator) | (denominator <= 0.0),
        "Purification trace must be finite and positive.",
    )
    return purification.precision.output(numerator / denominator)


__all__ = [
    "FiniteThermalEvidence",
    "FiniteThermalPolicy",
    "FiniteThermalResult",
    "FiniteThermalStatus",
    "finite_temperature_purification",
    "infinite_temperature_purification",
    "thermal_mpo_expectation",
]
