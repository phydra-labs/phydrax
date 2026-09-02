#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._core import LocallyPurifiedDensity, MatrixProductOperator, MatrixProductState


def _mps_inner_accumulation(
    left: MatrixProductState,
    right: MatrixProductState,
    /,
) -> Array:
    left_tensors = tuple(left.precision.accumulation(value) for value in left.tensors)
    right_tensors = tuple(right.precision.accumulation(value) for value in right.tensors)
    environment = jnp.ones(
        (1, 1), dtype=jnp.result_type(left_tensors[0], right_tensors[0])
    )
    for left_tensor, right_tensor in zip(left_tensors, right_tensors, strict=True):
        environment = oe.contract(
            "ab,api,bpj->ij",
            environment,
            jnp.conj(left_tensor),
            right_tensor,
        )
    return environment.reshape(())


def mps_inner(left: MatrixProductState, right: MatrixProductState, /) -> Array:
    if left.physical_dimensions != right.physical_dimensions:
        raise ValueError("MPS physical dimensions must match.")
    if left.precision.policy_id != right.precision.policy_id:
        raise ValueError("MPS precision policies must match.")
    return left.precision.output(_mps_inner_accumulation(left, right))


def mps_norm_squared(state: MatrixProductState, /) -> Array:
    return state.precision.decision(jnp.real(_mps_inner_accumulation(state, state)))


def mps_one_site_expectation(
    state: MatrixProductState,
    site: int,
    operator: ArrayLike,
    /,
) -> Array:
    site_ = int(site)
    if not 0 <= site_ < state.site_count:
        raise ValueError("One-site expectation site is outside the MPS.")
    value = state.precision.accumulation(jnp.asarray(operator))
    if value.shape != (
        state.physical_dimensions[site_],
        state.physical_dimensions[site_],
    ):
        raise ValueError("One-site operator shape does not match the MPS.")
    tensors = tuple(state.precision.accumulation(value) for value in state.tensors)
    environment = jnp.ones((1, 1), dtype=tensors[0].dtype)
    for index, tensor in enumerate(tensors):
        if index == site_:
            environment = oe.contract(
                "ab,api,pq,bqj->ij",
                environment,
                jnp.conj(tensor),
                value,
                tensor,
            )
        else:
            environment = oe.contract(
                "ab,api,bpj->ij", environment, jnp.conj(tensor), tensor
            )
    return state.precision.output(environment.reshape(()))


def lpdo_raw_trace(state: LocallyPurifiedDensity, /) -> Array:
    tensors = tuple(state.precision.accumulation(value) for value in state.tensors)
    environment = jnp.ones((1, 1), dtype=tensors[0].dtype)
    for tensor in tensors:
        environment = oe.contract(
            "ab,apkr,bpks->rs",
            environment,
            jnp.conj(tensor),
            tensor,
        )
    return state.precision.decision(jnp.real(environment.reshape(())))


def lpdo_one_site_reduced(state: LocallyPurifiedDensity, site: int, /) -> Array:
    site_ = int(site)
    if not 0 <= site_ < state.site_count:
        raise ValueError("One-site reduced-density site is outside the LPDO.")
    dimension = state.physical_dimensions[site_]
    output_exemplar = state.precision.output(
        jnp.zeros((dimension, dimension), dtype=state.tensors[0].dtype)
    )
    result = jnp.zeros((dimension, dimension), dtype=output_exemplar.dtype)
    for row in range(dimension):
        for column in range(dimension):
            operator = (
                jnp.zeros((dimension, dimension), dtype=result.dtype)
                .at[column, row]
                .set(1.0)
            )
            value = _lpdo_one_site_expectation(state, site_, operator)
            result = result.at[row, column].set(value)
    return result


def _lpdo_one_site_expectation(
    state: LocallyPurifiedDensity, site: int, operator: Array, /
) -> Array:
    tensors = tuple(state.precision.accumulation(value) for value in state.tensors)
    operator_ = state.precision.accumulation(operator)
    environment = jnp.ones((1, 1), dtype=tensors[0].dtype)
    for index, tensor in enumerate(tensors):
        if index == site:
            environment = oe.contract(
                "ab,apkr,pq,bqks->rs",
                environment,
                jnp.conj(tensor),
                operator_,
                tensor,
            )
        else:
            environment = oe.contract(
                "ab,apkr,bpks->rs",
                environment,
                jnp.conj(tensor),
                tensor,
            )
    return state.precision.output(environment.reshape(()))


def _validate_mps_mpo(
    bra: MatrixProductState,
    operator: MatrixProductOperator,
    ket: MatrixProductState,
    /,
) -> None:
    if not isinstance(bra, MatrixProductState) or not isinstance(ket, MatrixProductState):
        raise TypeError("bra and ket must be MatrixProductState values.")
    if not isinstance(operator, MatrixProductOperator):
        raise TypeError("operator must be a MatrixProductOperator.")
    if bra.site_count != operator.site_count or ket.site_count != operator.site_count:
        raise ValueError("MPS and MPO site counts must match.")
    if bra.physical_dimensions != operator.output_dimensions:
        raise ValueError("Bra dimensions must match MPO output dimensions.")
    if ket.physical_dimensions != operator.input_dimensions:
        raise ValueError("Ket dimensions must match MPO input dimensions.")
    policy_ids = {
        bra.precision.policy_id,
        operator.precision.policy_id,
        ket.precision.policy_id,
    }
    if len(policy_ids) != 1:
        raise ValueError("MPS and MPO precision policies must match.")


def _left_mps_mpo_step(
    environment: Array,
    bra_tensor: Array,
    operator_tensor: Array,
    ket_tensor: Array,
    /,
) -> Array:
    return oe.contract(
        "abc,apd,bpqe,cqf->def",
        environment,
        jnp.conj(bra_tensor),
        operator_tensor,
        ket_tensor,
    )


def _right_mps_mpo_step(
    environment: Array,
    bra_tensor: Array,
    operator_tensor: Array,
    ket_tensor: Array,
    /,
) -> Array:
    return oe.contract(
        "apd,bpqe,cqf,def->abc",
        jnp.conj(bra_tensor),
        operator_tensor,
        ket_tensor,
        environment,
    )


def build_mps_mpo_environments(
    bra: MatrixProductState,
    operator: MatrixProductOperator,
    ket: MatrixProductState,
    /,
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    """Build all open-boundary left and right bra--MPO--ket environments."""
    _validate_mps_mpo(bra, operator, ket)
    precision = bra.precision
    bra_tensors = precision.accumulation(bra.tensors)
    operator_tensors = precision.accumulation(operator.tensors)
    ket_tensors = precision.accumulation(ket.tensors)
    dtype = jnp.result_type(bra_tensors[0], operator_tensors[0], ket_tensors[0])
    left = [jnp.ones((1, 1, 1), dtype=dtype)]
    for bra_tensor, operator_tensor, ket_tensor in zip(
        bra_tensors, operator_tensors, ket_tensors, strict=True
    ):
        left.append(_left_mps_mpo_step(left[-1], bra_tensor, operator_tensor, ket_tensor))
    right: list[Array] = [
        jnp.zeros((0, 0, 0), dtype=dtype) for _ in range(operator.site_count)
    ] + [jnp.ones((1, 1, 1), dtype=dtype)]
    for index in range(operator.site_count - 1, -1, -1):
        right[index] = _right_mps_mpo_step(
            right[index + 1],
            bra_tensors[index],
            operator_tensors[index],
            ket_tensors[index],
        )
    return tuple(left), tuple(right)


def mps_mpo_inner(
    bra: MatrixProductState,
    operator: MatrixProductOperator,
    ket: MatrixProductState,
    /,
) -> Array:
    left, _ = build_mps_mpo_environments(bra, operator, ket)
    return bra.precision.output(left[-1].reshape(()))


def mps_mpo_expectation(
    state: MatrixProductState,
    operator: MatrixProductOperator,
    /,
) -> Array:
    return mps_mpo_inner(state, operator, state)


def _mpo_inner_accumulation(
    left: MatrixProductOperator,
    right: MatrixProductOperator,
    /,
) -> Array:
    precision = left.precision
    left_tensors = precision.accumulation(left.tensors)
    right_tensors = precision.accumulation(right.tensors)
    environment = jnp.ones(
        (1, 1), dtype=jnp.result_type(left_tensors[0], right_tensors[0])
    )
    for first, second in zip(left_tensors, right_tensors, strict=True):
        environment = oe.contract(
            "ab,aoic,boid->cd",
            environment,
            jnp.conj(first),
            second,
        )
    return environment.reshape(())


def mpo_inner(
    left: MatrixProductOperator,
    right: MatrixProductOperator,
    /,
) -> Array:
    if not isinstance(left, MatrixProductOperator) or not isinstance(
        right, MatrixProductOperator
    ):
        raise TypeError("left and right must be MatrixProductOperator values.")
    if (
        left.output_dimensions != right.output_dimensions
        or left.input_dimensions != right.input_dimensions
    ):
        raise ValueError("MPO dimensions must match.")
    if left.precision.policy_id != right.precision.policy_id:
        raise ValueError("MPO precision policies must match.")
    return left.precision.output(_mpo_inner_accumulation(left, right))


def mpo_norm(operator: MatrixProductOperator, /) -> Array:
    if not isinstance(operator, MatrixProductOperator):
        raise TypeError("operator must be a MatrixProductOperator.")
    value = operator.precision.decision(
        jnp.real(_mpo_inner_accumulation(operator, operator))
    )
    return operator.precision.decision(jnp.sqrt(jnp.maximum(value, 0.0)))


def mpo_hermiticity_residual(operator: MatrixProductOperator, /) -> Array:
    if not isinstance(operator, MatrixProductOperator):
        raise TypeError("operator must be a MatrixProductOperator.")
    if operator.output_dimensions != operator.input_dimensions:
        raise ValueError("MPO Hermiticity requires a square operator.")
    from ._mpo import adjoint_mpo

    adjoint = adjoint_mpo(operator)
    own = jnp.real(_mpo_inner_accumulation(operator, operator))
    adjoint_own = jnp.real(_mpo_inner_accumulation(adjoint, adjoint))
    cross = jnp.real(_mpo_inner_accumulation(operator, adjoint))
    difference_squared = jnp.maximum(own + adjoint_own - 2.0 * cross, 0.0)
    scale = jnp.maximum(jnp.sqrt(jnp.maximum(own, 0.0)), 1.0)
    return operator.precision.decision(jnp.sqrt(difference_squared) / scale)


class PreparedChainEnvironments(StrictModule):
    """Reusable finite-chain contraction environments with refresh identity."""

    bra: MatrixProductState
    operator: MatrixProductOperator
    ket: MatrixProductState
    left: tuple[Array, ...]
    right: tuple[Array, ...]
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class OneSiteMPOEffectiveAction(StrictModule):
    left_environment: Array
    operator_tensor: Array
    right_environment: Array

    def __call__(self, vector: Array, /) -> Array:
        return oe.contract(
            "abc,bpqe,def,cqf->apd",
            self.left_environment,
            self.operator_tensor,
            self.right_environment,
            vector,
        )


class TwoSiteMPOEffectiveAction(StrictModule):
    left_environment: Array
    left_operator: Array
    right_operator: Array
    right_environment: Array

    def __call__(self, vector: Array, /) -> Array:
        return oe.contract(
            "abc,bpim,mqjn,dne,cije->apqd",
            self.left_environment,
            self.left_operator,
            self.right_operator,
            self.right_environment,
            vector,
        )


class BondOverlapEffectiveAction(StrictModule):
    left_environment: Array
    right_environment: Array

    def __call__(self, vector: Array, /) -> Array:
        return oe.contract(
            "abc,dbf,cf->ad",
            self.left_environment,
            self.right_environment,
            vector,
        )


def prepare_chain_environments(
    bra: MatrixProductState,
    operator: MatrixProductOperator,
    ket: MatrixProductState,
    /,
) -> PreparedChainEnvironments:
    left, right = build_mps_mpo_environments(bra, operator, ket)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-chain-environments",
            "bra": bra.structure_id,
            "operator": operator.structure_id,
            "ket": ket.structure_id,
        }
    )
    return PreparedChainEnvironments(
        bra,
        operator,
        ket,
        left,
        right,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
    )


def refresh_chain_environments(
    prepared: PreparedChainEnvironments,
    bra: MatrixProductState,
    operator: MatrixProductOperator,
    ket: MatrixProductState,
    /,
) -> PreparedChainEnvironments:
    if not isinstance(prepared, PreparedChainEnvironments):
        raise TypeError("prepared must be PreparedChainEnvironments.")
    expected = canonical_fingerprint(
        {
            "kind": "prepared-chain-environments",
            "bra": bra.structure_id,
            "operator": operator.structure_id,
            "ket": ket.structure_id,
        }
    )
    if expected != prepared.prepared_id:
        raise ValueError("Chain environment structure changed; prepare again.")
    left, right = build_mps_mpo_environments(bra, operator, ket)
    return PreparedChainEnvironments(
        bra,
        operator,
        ket,
        left,
        right,
        prepared.numeric_version + 1,
        prepared.prepared_id,
    )


def one_site_effective_action(
    prepared: PreparedChainEnvironments, site: int, /
) -> OneSiteMPOEffectiveAction:
    site_ = int(site)
    if not 0 <= site_ < prepared.operator.site_count:
        raise ValueError("Effective-action site is outside the chain.")
    precision = prepared.bra.precision
    return OneSiteMPOEffectiveAction(
        precision.accumulation(prepared.left[site_]),
        precision.accumulation(prepared.operator.tensors[site_]),
        precision.accumulation(prepared.right[site_ + 1]),
    )


def two_site_effective_action(
    prepared: PreparedChainEnvironments, bond: int, /
) -> TwoSiteMPOEffectiveAction:
    bond_ = int(bond)
    if not 0 <= bond_ < prepared.operator.site_count - 1:
        raise ValueError("Effective-action bond is outside the chain.")
    precision = prepared.bra.precision
    return TwoSiteMPOEffectiveAction(
        precision.accumulation(prepared.left[bond_]),
        precision.accumulation(prepared.operator.tensors[bond_]),
        precision.accumulation(prepared.operator.tensors[bond_ + 1]),
        precision.accumulation(prepared.right[bond_ + 2]),
    )


__all__ = [
    "BondOverlapEffectiveAction",
    "OneSiteMPOEffectiveAction",
    "PreparedChainEnvironments",
    "TwoSiteMPOEffectiveAction",
    "build_mps_mpo_environments",
    "lpdo_one_site_reduced",
    "lpdo_raw_trace",
    "mpo_hermiticity_residual",
    "mpo_inner",
    "mpo_norm",
    "mps_inner",
    "mps_mpo_expectation",
    "mps_mpo_inner",
    "mps_norm_squared",
    "mps_one_site_expectation",
    "one_site_effective_action",
    "prepare_chain_environments",
    "refresh_chain_environments",
    "two_site_effective_action",
]
