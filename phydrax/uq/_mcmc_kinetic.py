#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, PyTree

from .._strict import StrictModule
from ..nn.parameters import ParameterSubspace


MCMCKineticKind = Literal["diagonal", "blocks", "diagonal_low_rank"]


class MCMCMassAdaptationPlan(StrictModule):
    """Finite-resource structured inverse-mass adaptation declaration."""

    kind: MCMCKineticKind = eqx.field(static=True)
    parameter_blocks: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    max_block_size: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    memory_cap_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        kind: MCMCKineticKind,
        /,
        *,
        parameter_blocks: Sequence[Sequence[str]] = (),
        max_block_size: int = 0,
        rank: int = 0,
        memory_cap_bytes: int = 2**30,
    ):
        if kind not in ("diagonal", "blocks", "diagonal_low_rank"):
            raise ValueError("Unknown MCMC kinetic kind.")
        blocks = tuple(tuple(str(path) for path in block) for block in parameter_blocks)
        cap = int(memory_cap_bytes)
        if cap <= 0:
            raise ValueError("memory_cap_bytes must be positive.")
        if kind == "blocks":
            if not blocks or any(not block for block in blocks):
                raise ValueError("Block metrics require nonempty parameter path blocks.")
            flattened = tuple(path for block in blocks for path in block)
            if len(flattened) != len(set(flattened)):
                raise ValueError("MCMC metric parameter blocks must be disjoint.")
            if int(max_block_size) <= 0:
                raise ValueError("Block metrics require max_block_size > 0.")
        elif blocks:
            raise ValueError("parameter_blocks are valid only for block metrics.")
        if kind == "diagonal_low_rank" and int(rank) <= 0:
            raise ValueError("Diagonal-low-rank metrics require rank > 0.")
        if kind != "diagonal_low_rank" and int(rank) != 0:
            raise ValueError("rank is valid only for diagonal-low-rank metrics.")
        self.kind = kind
        self.parameter_blocks = blocks
        self.max_block_size = int(max_block_size)
        self.rank = int(rank)
        self.memory_cap_bytes = cap

    @classmethod
    def diagonal(cls, *, memory_cap_bytes: int = 2**30) -> MCMCMassAdaptationPlan:
        return cls("diagonal", memory_cap_bytes=memory_cap_bytes)

    @classmethod
    def blocks(
        cls,
        parameter_paths: Sequence[Sequence[str]],
        /,
        *,
        max_block_size: int,
        memory_cap_bytes: int = 2**30,
    ) -> MCMCMassAdaptationPlan:
        return cls(
            "blocks",
            parameter_blocks=parameter_paths,
            max_block_size=max_block_size,
            memory_cap_bytes=memory_cap_bytes,
        )

    @classmethod
    def diagonal_low_rank(
        cls,
        rank: int,
        /,
        *,
        memory_cap_bytes: int = 2**30,
    ) -> MCMCMassAdaptationPlan:
        return cls(
            "diagonal_low_rank",
            rank=rank,
            memory_cap_bytes=memory_cap_bytes,
        )


class PreparedMCMCKinetic(StrictModule):
    """Fixed-layout kinetic actions without dense full-network storage."""

    diagonal: Array
    low_rank_factor: Array
    block_factors: tuple[Array, ...]
    block_indices: tuple[Array, ...]
    subspace: ParameterSubspace
    kind: MCMCKineticKind = eqx.field(static=True)
    parameter_count: int = eqx.field(static=True)
    block_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    memory_bytes: int = eqx.field(static=True)
    condition_estimate: float = eqx.field(static=True)

    def pack(self, value: PyTree[Array], /) -> Array:
        selected, _ = eqx.partition(value, eqx.is_inexact_array)
        return self.subspace.pack(selected)

    def unpack(self, value: ArrayLike, /) -> PyTree[Array]:
        return self.subspace.unpack(jnp.asarray(value))

    def inverse_mass_action_vector(self, value: ArrayLike, /) -> Array:
        vector = jnp.asarray(value)
        if vector.shape != (self.parameter_count,):
            raise ValueError("Kinetic vector has an incompatible parameter layout.")
        if self.kind == "diagonal":
            return self.diagonal * vector
        if self.kind == "diagonal_low_rank":
            return self.diagonal * vector + oe.contract(
                "pr,qr,q->p", self.low_rank_factor, self.low_rank_factor, vector
            )
        output = jnp.zeros_like(vector)
        for indices, factor in zip(self.block_indices, self.block_factors, strict=True):
            block = vector[indices]
            output = output.at[indices].set(
                oe.contract("ik,jk,j->i", factor, factor, block)
            )
        return output

    def inverse_mass_action(self, value: PyTree[Array], /) -> PyTree[Array]:
        return self.unpack(self.inverse_mass_action_vector(self.pack(value)))

    def kinetic_energy_vector(self, momentum: ArrayLike, /) -> Array:
        value = jnp.asarray(momentum)
        return 0.5 * oe.contract("p,p->", value, self.inverse_mass_action_vector(value))

    def sample_momentum_vector(self, key: Array, /) -> Array:
        dtype = self.diagonal.dtype
        if self.kind == "diagonal":
            return jr.normal(key, (self.parameter_count,), dtype=dtype) / jnp.sqrt(
                self.diagonal
            )
        if self.kind == "blocks":
            keys = jr.split(key, len(self.block_factors))
            output = jnp.zeros((self.parameter_count,), dtype=dtype)
            for indices, factor, block_key in zip(
                self.block_indices, self.block_factors, keys, strict=True
            ):
                standard = jr.normal(block_key, (indices.size,), dtype=dtype)
                momentum = jsp.linalg.solve_triangular(
                    jnp.conj(factor.T), standard, lower=False
                )
                output = output.at[indices].set(momentum)
            return output
        diagonal_inverse = 1.0 / self.diagonal
        position_key, auxiliary_key = jr.split(key)
        position = jnp.sqrt(diagonal_inverse) * jr.normal(
            position_key, (self.parameter_count,), dtype=dtype
        )
        auxiliary = jr.normal(auxiliary_key, (self.rank,), dtype=dtype)
        scaled_factor = diagonal_inverse[:, None] * self.low_rank_factor
        correction = jnp.eye(self.rank, dtype=dtype) + oe.contract(
            "pr,ps->rs", self.low_rank_factor, scaled_factor
        )
        right = oe.contract("pr,p->r", self.low_rank_factor, position) + auxiliary
        solved = jnp.linalg.solve(correction, right)
        return position - oe.contract("pr,r->p", scaled_factor, solved)

    def sample_momentum(self, key: Array, /) -> PyTree[Array]:
        return self.unpack(self.sample_momentum_vector(key))

    def generalized_uturn(
        self,
        left_momentum: ArrayLike,
        right_momentum: ArrayLike,
        displacement: ArrayLike,
        /,
    ) -> Array:
        delta = jnp.asarray(displacement)
        left_velocity = self.inverse_mass_action_vector(left_momentum)
        right_velocity = self.inverse_mass_action_vector(right_momentum)
        return (oe.contract("p,p->", delta, left_velocity) < 0.0) | (
            oe.contract("p,p->", delta, right_velocity) < 0.0
        )


def prepare_mcmc_kinetic(
    reference_position: PyTree[Array],
    plan: MCMCMassAdaptationPlan,
    /,
    *,
    diagonal: ArrayLike | None = None,
    block_inverse_masses: Sequence[ArrayLike] = (),
    low_rank_factor: ArrayLike | None = None,
) -> PreparedMCMCKinetic:
    """Prepare and validate one complete real-PyTree structured metric."""
    if not isinstance(plan, MCMCMassAdaptationPlan):
        raise TypeError("plan must be MCMCMassAdaptationPlan.")
    subspace = ParameterSubspace(reference_position, eqx.is_inexact_array)
    if any(
        not jnp.issubdtype(jnp.dtype(dtype), jnp.floating)
        for dtype in subspace.leaf_dtypes
    ):
        raise TypeError("MCMC kinetic layouts require real floating parameter leaves.")
    count = subspace.total_dimension
    dtype = jnp.result_type(*[jnp.dtype(value) for value in subspace.leaf_dtypes])
    diagonal_array = (
        jnp.ones((count,), dtype=dtype)
        if diagonal is None
        else jnp.asarray(diagonal, dtype=dtype)
    )
    if (
        diagonal_array.shape != (count,)
        or bool(jnp.any(~jnp.isfinite(diagonal_array)))
        or bool(jnp.any(diagonal_array <= 0.0))
    ):
        raise ValueError("Inverse-mass diagonal must be finite and strictly positive.")
    blocks: tuple[Array, ...] = ()
    indices: tuple[Array, ...] = ()
    low_rank = jnp.zeros((count, 0), dtype=dtype)
    block_paths: tuple[tuple[str, ...], ...] = ()
    if plan.kind == "blocks":
        available = subspace.leaf_paths
        supplied = tuple(path for block in plan.parameter_blocks for path in block)
        if set(supplied) != set(available) or len(supplied) != len(available):
            raise ValueError(
                "Block metric paths must cover every parameter leaf exactly once."
            )
        if len(block_inverse_masses) != len(plan.parameter_blocks):
            raise ValueError("One inverse-mass matrix is required per parameter block.")
        offsets: dict[str, Array] = {}
        start = 0
        for path, shape in zip(subspace.leaf_paths, subspace.leaf_shapes, strict=True):
            size = math_prod(shape)
            offsets[path] = jnp.arange(start, start + size, dtype=jnp.int32)
            start += size
        factors = []
        block_indices = []
        for paths, matrix in zip(
            plan.parameter_blocks, block_inverse_masses, strict=True
        ):
            current_indices = jnp.concatenate(tuple(offsets[path] for path in paths))
            size = int(current_indices.size)
            if size > plan.max_block_size:
                raise ValueError("A prepared MCMC block exceeds max_block_size.")
            value = jnp.asarray(matrix, dtype=dtype)
            if value.shape != (size, size) or bool(jnp.any(~jnp.isfinite(value))):
                raise ValueError("Block inverse mass has invalid shape or values.")
            factor = jnp.linalg.cholesky(value)
            if bool(jnp.any(~jnp.isfinite(factor))) or bool(
                jnp.any(jnp.diag(factor) <= 0.0)
            ):
                raise ValueError("Block inverse mass must be positive definite.")
            factors.append(factor)
            block_indices.append(current_indices)
        blocks = tuple(factors)
        indices = tuple(block_indices)
        block_paths = plan.parameter_blocks
    elif plan.kind == "diagonal_low_rank":
        if low_rank_factor is None:
            raise ValueError("Diagonal-low-rank metrics require low_rank_factor.")
        low_rank = jnp.asarray(low_rank_factor, dtype=dtype)
        if low_rank.shape != (count, plan.rank) or bool(jnp.any(~jnp.isfinite(low_rank))):
            raise ValueError("low_rank_factor has invalid shape or nonfinite values.")
    memory = int(
        diagonal_array.nbytes + low_rank.nbytes + sum(value.nbytes for value in blocks)
    )
    if memory > plan.memory_cap_bytes:
        raise MemoryError("Prepared MCMC kinetic exceeds memory_cap_bytes.")
    maximum = float(jnp.max(diagonal_array))
    minimum = float(jnp.min(diagonal_array))
    return PreparedMCMCKinetic(
        diagonal=diagonal_array,
        low_rank_factor=low_rank,
        block_factors=blocks,
        block_indices=indices,
        subspace=subspace,
        kind=plan.kind,
        parameter_count=count,
        block_paths=block_paths,
        rank=plan.rank,
        memory_bytes=memory,
        condition_estimate=maximum / minimum,
    )


def math_prod(shape: tuple[int, ...], /) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


__all__ = [
    "MCMCKineticKind",
    "MCMCMassAdaptationPlan",
    "PreparedMCMCKinetic",
    "prepare_mcmc_kinetic",
]
