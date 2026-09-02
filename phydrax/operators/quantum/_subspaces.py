#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import TypeAlias

import equinox as eqx
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._register import HilbertRegisterLayout


class QuantumSubspaceEvidence(StrictModule):
    """Finite isometry evidence for one logical-to-physical embedding."""

    isometry_residual: Array
    finite: Array
    valid: Array


class BasisStateSubspace(StrictModule):
    """Ordered logical subspace represented by selected physical basis indices."""

    basis_indices: Array
    evidence: QuantumSubspaceEvidence
    physical_dimension: int = eqx.field(static=True)
    logical_dimension: int = eqx.field(static=True)
    subspace_id: str = eqx.field(static=True)

    def __init__(
        self,
        physical_dimension: int,
        basis_indices: Sequence[int] | ArrayLike,
        /,
        *,
        subspace_id: str | None = None,
    ):
        if isinstance(physical_dimension, bool) or not isinstance(
            physical_dimension, Integral
        ):
            raise TypeError("physical_dimension must be a positive integer.")
        physical = int(physical_dimension)
        if physical <= 0:
            raise ValueError("physical_dimension must be positive.")
        indices = jnp.asarray(basis_indices)
        if (
            indices.ndim != 1
            or indices.size == 0
            or not jnp.issubdtype(indices.dtype, jnp.integer)
        ):
            raise TypeError("basis_indices must be one nonempty integer vector.")
        if isinstance(indices, jax_core.Tracer):
            raise TypeError("basis_indices are structural and must be concrete.")
        host = np.asarray(indices, dtype=np.int64)
        if np.any(host < 0) or np.any(host >= physical):
            raise ValueError("basis_indices must lie within the physical dimension.")
        if np.unique(host).size != host.size:
            raise ValueError("basis_indices must be unique.")
        indices = jnp.asarray(host, dtype=jnp.int32)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "basis-state-subspace",
                    "physical_dimension": physical,
                    "basis_indices": host.tolist(),
                }
            )
            if subspace_id is None
            else str(subspace_id)
        )
        if not identifier:
            raise ValueError("subspace_id must be nonempty.")
        self.basis_indices = indices
        self.evidence = QuantumSubspaceEvidence(
            jnp.asarray(0.0),
            jnp.asarray(True),
            jnp.asarray(True),
        )
        self.physical_dimension = physical
        self.logical_dimension = int(indices.size)
        self.subspace_id = identifier

    def embed(self, logical_state: ArrayLike, /) -> Array:
        state = jnp.asarray(logical_state)
        if state.shape[-1:] != (self.logical_dimension,):
            raise ValueError("logical_state has the wrong trailing dimension.")
        output = jnp.zeros(
            state.shape[:-1] + (self.physical_dimension,),
            dtype=state.dtype,
        )
        return output.at[..., self.basis_indices].set(state)

    def restrict(self, physical_state: ArrayLike, /) -> Array:
        state = jnp.asarray(physical_state)
        if state.shape[-1:] != (self.physical_dimension,):
            raise ValueError("physical_state has the wrong trailing dimension.")
        return state[..., self.basis_indices]


class DenseQuantumSubspace(StrictModule):
    """Ordered logical subspace represented by a dense physical isometry."""

    isometry: Array
    evidence: QuantumSubspaceEvidence
    physical_dimension: int = eqx.field(static=True)
    logical_dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    subspace_id: str = eqx.field(static=True)

    def __init__(
        self,
        isometry: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
        maximum_entries: int = 1 << 26,
        subspace_id: str | None = None,
    ):
        value = jnp.asarray(isometry)
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError("isometry must be one nonempty matrix.")
        if value.shape[1] > value.shape[0]:
            raise ValueError("logical dimension must not exceed physical dimension.")
        if value.size > int(maximum_entries):
            raise ValueError("Dense subspace exceeds maximum_entries.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        gram = oe.contract("ai,aj->ij", jnp.conj(value), value)
        residual = jnp.max(jnp.abs(gram - jnp.eye(value.shape[1], dtype=gram.dtype)))
        finite = jnp.all(jnp.isfinite(value))
        valid = finite & (residual <= tolerance_)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "dense-quantum-subspace",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )
            if subspace_id is None
            else str(subspace_id)
        )
        if not identifier:
            raise ValueError("subspace_id must be nonempty.")
        self.isometry = value
        self.evidence = QuantumSubspaceEvidence(residual, finite, valid)
        self.physical_dimension = int(value.shape[0])
        self.logical_dimension = int(value.shape[1])
        self.tolerance = tolerance_
        self.subspace_id = identifier

    def embed(self, logical_state: ArrayLike, /) -> Array:
        state = jnp.asarray(logical_state)
        if state.shape[-1:] != (self.logical_dimension,):
            raise ValueError("logical_state has the wrong trailing dimension.")
        return oe.contract("...i,ai->...a", state, self.isometry)

    def restrict(self, physical_state: ArrayLike, /) -> Array:
        state = jnp.asarray(physical_state)
        if state.shape[-1:] != (self.physical_dimension,):
            raise ValueError("physical_state has the wrong trailing dimension.")
        return oe.contract("ai,...a->...i", jnp.conj(self.isometry), state)


QuantumSubspace: TypeAlias = BasisStateSubspace | DenseQuantumSubspace


def basis_state_subspace(
    layout: HilbertRegisterLayout,
    level_tuples: Sequence[Sequence[int]],
    /,
    *,
    subspace_id: str | None = None,
) -> BasisStateSubspace:
    """Construct ordered flat basis indices from local-level tuples."""

    if not isinstance(layout, HilbertRegisterLayout):
        raise TypeError("layout must be a HilbertRegisterLayout.")
    labels = tuple(tuple(int(level) for level in levels) for levels in level_tuples)
    if not labels:
        raise ValueError("level_tuples must be nonempty.")
    flat: list[int] = []
    for levels in labels:
        if len(levels) != layout.wire_count:
            raise ValueError("Every level tuple must cover every register wire.")
        if any(
            level < 0 or level >= dimension
            for level, dimension in zip(levels, layout.local_dimensions, strict=True)
        ):
            raise ValueError("A level tuple contains an out-of-range local level.")
        index = 0
        for level, dimension in zip(levels, layout.local_dimensions, strict=True):
            index = index * dimension + level
        flat.append(index)
    identifier = (
        canonical_fingerprint(
            {
                "kind": "register-basis-state-subspace",
                "layout": layout.layout_id,
                "levels": [list(levels) for levels in labels],
            }
        )
        if subspace_id is None
        else str(subspace_id)
    )
    return BasisStateSubspace(
        layout.dimension,
        flat,
        subspace_id=identifier,
    )


def embed_quantum_subspace(
    subspace: QuantumSubspace,
    logical_state: ArrayLike,
    /,
) -> Array:
    """Embed one logical state using a declared subspace representation."""

    if isinstance(subspace, BasisStateSubspace):
        return subspace.embed(logical_state)
    if isinstance(subspace, DenseQuantumSubspace):
        return subspace.embed(logical_state)
    raise TypeError("subspace must be a BasisStateSubspace or DenseQuantumSubspace.")


def restrict_quantum_subspace(
    subspace: QuantumSubspace,
    physical_state: ArrayLike,
    /,
) -> Array:
    """Restrict one physical state using a declared subspace representation."""

    if isinstance(subspace, BasisStateSubspace):
        return subspace.restrict(physical_state)
    if isinstance(subspace, DenseQuantumSubspace):
        return subspace.restrict(physical_state)
    raise TypeError("subspace must be a BasisStateSubspace or DenseQuantumSubspace.")


def project_quantum_operator(
    operator: ArrayLike,
    input_subspace: QuantumSubspace,
    output_subspace: QuantumSubspace | None = None,
    /,
) -> Array:
    """Project a physical operator between explicit input and output subspaces."""

    output = input_subspace if output_subspace is None else output_subspace
    if not isinstance(input_subspace, (BasisStateSubspace, DenseQuantumSubspace)):
        raise TypeError("input_subspace must be a quantum subspace.")
    if not isinstance(output, (BasisStateSubspace, DenseQuantumSubspace)):
        raise TypeError("output_subspace must be a quantum subspace.")
    value = jnp.asarray(operator)
    expected = (output.physical_dimension, input_subspace.physical_dimension)
    if value.shape[-2:] != expected:
        raise ValueError(f"operator trailing shape must be {expected}.")

    if isinstance(input_subspace, BasisStateSubspace):
        restricted_input = jnp.take(value, input_subspace.basis_indices, axis=-1)
    else:
        restricted_input = oe.contract(
            "...ab,bi->...ai",
            value,
            input_subspace.isometry,
        )
    if isinstance(output, BasisStateSubspace):
        return jnp.take(restricted_input, output.basis_indices, axis=-2)
    return oe.contract(
        "ao,...ai->...oi",
        jnp.conj(output.isometry),
        restricted_input,
    )


__all__ = [
    "BasisStateSubspace",
    "DenseQuantumSubspace",
    "QuantumSubspace",
    "QuantumSubspaceEvidence",
    "basis_state_subspace",
    "embed_quantum_subspace",
    "project_quantum_operator",
    "restrict_quantum_subspace",
]
