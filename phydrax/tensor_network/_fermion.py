#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._core import MatrixProductOperator, MatrixProductState
from ._mpo import add_mpo, apply_mpo, product_mpo


class FermionModeOrder(StrictModule):
    """Explicit global order used for every fermionic sign decision."""

    labels: tuple[str, ...] = eqx.field(static=True)
    order_id: str = eqx.field(static=True)

    def __init__(self, labels: Sequence[str], /):
        values = tuple(str(label) for label in labels)
        if (
            not values
            or any(not label for label in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("Fermion mode labels must be nonempty and unique.")
        self.labels = values
        self.order_id = canonical_fingerprint(
            {"kind": "fermion-mode-order", "labels": values}
        )

    def ordinal(self, label: str, /) -> int:
        value = str(label)
        if value not in self.labels:
            raise ValueError("Fermion mode is absent from the explicit order.")
        return self.labels.index(value)


class FermionTopologySignPlan(StrictModule):
    """Static Jordan-Wigner parity routes for edges of an arbitrary topology."""

    mode_order: FermionModeOrder = eqx.field(static=True)
    edges: tuple[tuple[str, str], ...] = eqx.field(static=True)
    parity_masks: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, mode_order: FermionModeOrder, edges: Sequence[tuple[str, str]], /):
        if not isinstance(mode_order, FermionModeOrder):
            raise TypeError("mode_order must be FermionModeOrder.")
        values = tuple((str(left), str(right)) for left, right in edges)
        if any(left == right for left, right in values) or len(set(values)) != len(
            values
        ):
            raise ValueError("Fermion topology edges must be unique nonloops.")
        masks = []
        for left, right in values:
            first = mode_order.ordinal(left)
            second = mode_order.ordinal(right)
            low, high = sorted((first, second))
            masks.append(
                tuple(
                    1 if low < index < high else 0
                    for index in range(len(mode_order.labels))
                )
            )
        self.mode_order = mode_order
        self.edges = values
        self.parity_masks = jnp.asarray(masks, dtype=jnp.int32).reshape(
            (len(values), len(mode_order.labels))
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fermion-topology-sign-plan",
                "mode_order": mode_order.order_id,
                "edges": values,
            }
        )

    def signs(self, occupations: ArrayLike, /) -> Array:
        values = jnp.asarray(occupations, dtype=jnp.int32)
        if values.shape != (len(self.mode_order.labels),):
            raise ValueError("Occupation vector does not match the fermion mode order.")
        values = eqx.error_if(
            values,
            jnp.any((values != 0) & (values != 1)),
            "Fermion occupations must be binary.",
        )
        exponents = self.parity_masks @ values
        return jnp.where(exponents % 2 == 0, 1, -1)


def fermionic_swap_gate(parities: Sequence[int] = (0, 1), /) -> Array:
    """Return FSWAP in row-output/column-input tensor convention."""

    values = tuple(int(value) for value in parities)
    if not values or any(value not in (0, 1) for value in values):
        raise ValueError("FSWAP basis parities must be a nonempty binary sequence.")
    dimension = len(values)
    gate = jnp.zeros((dimension, dimension, dimension, dimension), dtype=jnp.float64)
    for left in range(dimension):
        for right in range(dimension):
            sign = -1.0 if values[left] * values[right] else 1.0
            gate = gate.at[right, left, left, right].set(sign)
    return gate


def jordan_wigner_monomial_mpo(
    mode_order: FermionModeOrder,
    operations: Sequence[tuple[str, str]],
    /,
) -> MatrixProductOperator:
    """Compile an ordered creation/annihilation monomial to a bond-one MPO."""

    if not isinstance(mode_order, FermionModeOrder):
        raise TypeError("mode_order must be FermionModeOrder.")
    values = tuple((str(label), str(kind)) for label, kind in operations)
    if not values:
        raise ValueError("A Jordan-Wigner monomial requires at least one operation.")
    identity = jnp.eye(2, dtype=jnp.complex128)
    parity = jnp.diag(jnp.asarray((1.0, -1.0), dtype=jnp.complex128))
    creation = jnp.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=jnp.complex128)
    annihilation = jnp.asarray(((0.0, 1.0), (0.0, 0.0)), dtype=jnp.complex128)
    local = [identity for _ in mode_order.labels]
    for label, kind in values:
        site = mode_order.ordinal(label)
        if kind not in ("create", "annihilate"):
            raise ValueError("Fermion operation kind must be create or annihilate.")
        for index in range(site):
            local[index] = local[index] @ parity
        local[site] = local[site] @ (creation if kind == "create" else annihilation)
    return product_mpo(jnp.stack(local))


def jordan_wigner_hopping_mpo(
    mode_order: FermionModeOrder,
    left_mode: str,
    right_mode: str,
    /,
    *,
    amplitude: ArrayLike = 1.0,
) -> MatrixProductOperator:
    """Compile t c†_i c_j + conjugate(t) c†_j c_i with exact parity strings."""

    value = jnp.asarray(amplitude)
    if value.shape != ():
        raise ValueError("Fermion hopping amplitude must be scalar.")
    forward = jordan_wigner_monomial_mpo(
        mode_order, ((left_mode, "create"), (right_mode, "annihilate"))
    )
    backward = jordan_wigner_monomial_mpo(
        mode_order, ((right_mode, "create"), (left_mode, "annihilate"))
    )
    scaled_forward = MatrixProductOperator(
        (forward.tensors[0] * value,) + forward.tensors[1:],
        precision=forward.precision,
    )
    scaled_backward = MatrixProductOperator(
        (backward.tensors[0] * jnp.conj(value),) + backward.tensors[1:],
        precision=backward.precision,
    )
    return add_mpo(scaled_forward, scaled_backward)


class FermionChainState(StrictModule):
    """One-dimensional fermion state with an inseparable explicit mode order."""

    state: MatrixProductState
    mode_order: FermionModeOrder = eqx.field(static=True)
    chain_id: str = eqx.field(static=True)

    def __init__(self, state: MatrixProductState, mode_order: FermionModeOrder, /):
        if not isinstance(state, MatrixProductState) or not isinstance(
            mode_order, FermionModeOrder
        ):
            raise TypeError("state and mode_order have invalid types.")
        if state.site_count != len(mode_order.labels) or any(
            dimension != 2 for dimension in state.physical_dimensions
        ):
            raise ValueError("Fermion chains require one two-level site per mode.")
        self.state = state
        self.mode_order = mode_order
        self.chain_id = canonical_fingerprint(
            {
                "kind": "fermion-chain-state",
                "mode_order": mode_order.order_id,
                "structure": state.structure_id,
            }
        )

    @classmethod
    def occupation_basis(
        cls, mode_order: FermionModeOrder, occupations: Sequence[int], /
    ) -> FermionChainState:
        values = tuple(int(value) for value in occupations)
        if len(values) != len(mode_order.labels) or any(
            value not in (0, 1) for value in values
        ):
            raise ValueError("One binary occupation is required per fermion mode.")
        local = tuple(
            jnp.asarray((1.0, 0.0), dtype=jnp.complex128)
            if value == 0
            else jnp.asarray((0.0, 1.0), dtype=jnp.complex128)
            for value in values
        )
        return cls(
            MatrixProductState(tuple(value[None, :, None] for value in local)),
            mode_order,
        )


class FermionOperationEvidence(StrictModule):
    input_norm: Array
    output_norm: Array
    parity_changed: Array
    discarded_weight: Array
    valid: Array
    operation_id: str = eqx.field(static=True)


def apply_fermion_chain_operation(
    chain: FermionChainState,
    operations: Sequence[tuple[str, str]],
    /,
    *,
    maximum_bond_dimension: int,
) -> tuple[FermionChainState, FermionOperationEvidence]:
    """Apply a finite Jordan-Wigner monomial and report the actual output norm."""

    if not isinstance(chain, FermionChainState):
        raise TypeError("chain must be FermionChainState.")
    values = tuple((str(label), str(kind)) for label, kind in operations)
    operator = jordan_wigner_monomial_mpo(chain.mode_order, values)
    result, compression = apply_mpo(
        operator,
        chain.state,
        maximum_bond_dimension=int(maximum_bond_dimension),
        normalize=False,
    )
    output = FermionChainState(result, chain.mode_order)
    input_norm = chain.state.norm()
    output_norm = result.norm()
    parity_changed = jnp.asarray(len(values) % 2 == 1)
    operation_id = canonical_fingerprint(
        {
            "kind": "fermion-chain-operation",
            "mode_order": chain.mode_order.order_id,
            "operations": values,
            "maximum_bond_dimension": int(maximum_bond_dimension),
        }
    )
    valid = (
        jnp.isfinite(input_norm)
        & jnp.isfinite(output_norm)
        & compression.valid
        & (output_norm >= 0)
    )
    return output, FermionOperationEvidence(
        input_norm,
        output_norm,
        parity_changed,
        compression.accumulated_discarded_weight,
        valid,
        operation_id,
    )


__all__ = [
    "FermionChainState",
    "FermionModeOrder",
    "FermionOperationEvidence",
    "FermionTopologySignPlan",
    "apply_fermion_chain_operation",
    "fermionic_swap_gate",
    "jordan_wigner_hopping_mpo",
    "jordan_wigner_monomial_mpo",
]
