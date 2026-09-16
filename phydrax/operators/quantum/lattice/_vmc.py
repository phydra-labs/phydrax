#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""VMC connected-configuration lowering for compiled quantum lattices."""

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._discrete import AbstractDiscreteQuantumOperator, ConnectedConfigurations
from ._operator import apply_compiled_to_coordinate, QuantumSectorOperator


class QuantumLatticeVMCOperator(AbstractDiscreteQuantumOperator):
    """Fixed-sector VMC view with H[current, connected] matrix elements."""

    sector_operator: QuantumSectorOperator
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    max_connections: int = eqx.field(static=True)

    def __init__(self, sector_operator: QuantumSectorOperator, /):
        if not isinstance(sector_operator, QuantumSectorOperator):
            raise TypeError("sector_operator must be QuantumSectorOperator.")
        if not sector_operator.properties.certifies("self_adjoint"):
            raise ValueError(
                "The VMC lowerer requires a self-adjoint fixed-sector operator."
            )
        self.sector_operator = sector_operator
        self.configuration_shape = (len(sector_operator.charge_map.source.site_ids),)
        self.operator_id = f"{sector_operator.operator_id}:vmc-connected"
        self.max_connections = sector_operator.prepared.plan.total_branches_per_input

    def _flat(self, configurations: Array, /) -> tuple[Array, tuple[int, ...]]:
        raw = jnp.asarray(configurations)
        if not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError("VMC configurations must use an integer dtype.")
        values = raw.astype(jnp.int32)
        if values.ndim < 1 or values.shape[-1:] != self.configuration_shape:
            raise ValueError(
                f"configurations must end in shape {self.configuration_shape}."
            )
        batch_shape = tuple(int(value) for value in values.shape[:-1])
        count = prod(batch_shape) if batch_shape else 1
        return values.reshape((count,) + self.configuration_shape), batch_shape

    def _one(self, configuration: Array, /) -> tuple[Array, Array, Array, Array]:
        basis = self.sector_operator.charge_map.source
        configuration = eqx.error_if(
            configuration,
            ~basis.contains(configuration),
            "VMC configuration is outside the fixed sector.",
        )
        connected, outgoing = apply_compiled_to_coordinate(
            self.sector_operator.prepared, configuration
        )
        same = jnp.all(connected == configuration[None, :], axis=-1)
        in_sector = jax.vmap(basis.contains)(connected)
        active = in_sector & ~same & (jnp.abs(outgoing) > 0)
        diagonal = jnp.sum(jnp.where(in_sector & same, outgoing, 0.0j))
        # The compiler emits H[out, current]. Local estimators consume
        # H[current, connected], hence the conjugate for a certified Hermitian H.
        incoming = jnp.conj(outgoing)
        return diagonal, connected, incoming, active

    def diagonal(self, configurations: Array, /) -> Array:
        flat, batch_shape = self._flat(configurations)
        values = jax.vmap(lambda value: self._one(value)[0])(flat)
        return values.reshape(batch_shape)

    def connections(self, configurations: Array, /) -> ConnectedConfigurations:
        flat, batch_shape = self._flat(configurations)
        connected, elements, active = jax.vmap(lambda value: self._one(value)[1:])(flat)
        return ConnectedConfigurations(
            connected.reshape(
                batch_shape + (self.max_connections,) + self.configuration_shape
            ),
            elements.reshape(batch_shape + (self.max_connections,)),
            active.reshape(batch_shape + (self.max_connections,)),
            configuration_shape=self.configuration_shape,
        )


def lower_quantum_lattice_to_vmc(
    sector_operator: QuantumSectorOperator, /
) -> QuantumLatticeVMCOperator:
    """Construct the VMC target explicitly; no backend dispatch is performed."""
    return QuantumLatticeVMCOperator(sector_operator)


__all__ = ["QuantumLatticeVMCOperator", "lower_quantum_lattice_to_vmc"]
