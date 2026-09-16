#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Matrix-free quantum-lattice action between direct conserved sectors."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ....linalg import (
    AbstractLinearOperator,
    ArraySpace,
    LinearCapabilityError,
    OperatorCapabilities,
    OperatorProperties,
)
from ._compile import (
    certify_charge_map,
    ChargeMapCertification,
    CompiledMonomial,
    PreparedQuantumLattice,
)
from ._sector import SectorChargeMap


def _fermion_predecessor_sites(
    prepared: PreparedQuantumLattice, mode_label: str, /
) -> tuple[int, ...]:
    order = prepared.specification.fermion_mode_order
    if order is None:
        raise ValueError("A fermion factor is missing its FermionModeOrder.")
    ordinal = order.ordinal(mode_label)
    predecessors = set(order.labels[:ordinal])
    return tuple(
        index
        for index, space in enumerate(prepared.specification.spaces)
        if space.fermion_mode_label in predecessors
    )


def apply_monomial_to_coordinate(
    prepared: PreparedQuantumLattice,
    monomial: CompiledMonomial,
    coordinate: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Return fixed-capacity output coordinates and H[out, in] amplitudes."""
    if not isinstance(prepared, PreparedQuantumLattice) or not isinstance(
        monomial, CompiledMonomial
    ):
        raise TypeError("Expected prepared lattice and compiled monomial.")
    raw = jnp.asarray(coordinate)
    if not jnp.issubdtype(raw.dtype, jnp.integer):
        raise TypeError("Quantum-lattice coordinates must use an integer dtype.")
    value = raw.astype(jnp.int32)
    site_count = len(prepared.specification.spaces)
    if value.shape != (site_count,):
        raise ValueError("coordinate must provide one local state per lattice site.")
    dimensions = jnp.asarray(prepared.specification.local_dimensions, dtype=jnp.int32)
    value = eqx.error_if(
        value,
        jnp.any((value < 0) | (value >= dimensions)),
        "Quantum-lattice coordinate is outside a local space.",
    )
    return _apply_monomial_unchecked(prepared, monomial, value)


def _apply_monomial_unchecked(
    prepared: PreparedQuantumLattice,
    monomial: CompiledMonomial,
    coordinate: Array,
    /,
) -> tuple[Array, Array]:
    configurations = coordinate[None, :]
    amplitudes = jnp.asarray((monomial.coefficient,), dtype=jnp.complex128)
    for factor in reversed(monomial.factors):
        site = prepared.specification.site_ids.index(factor.space.site_id)
        dimension = factor.space.dimension
        incoming = configurations[:, site]
        local = jnp.swapaxes(factor.matrix[:, incoming], 0, 1)
        if factor.fermion_parity:
            mode = factor.space.fermion_mode_label
            if mode is None:
                raise ValueError("Odd fermion operators require a mode label.")
            predecessors = _fermion_predecessor_sites(prepared, mode)
            parity = (
                jnp.sum(configurations[:, jnp.asarray(predecessors)], axis=1)
                if predecessors
                else jnp.zeros((configurations.shape[0],), dtype=jnp.int32)
            )
            sign = jnp.where(parity % 2 == 0, 1.0, -1.0)
        else:
            sign = jnp.ones((configurations.shape[0],), dtype=jnp.float64)
        configurations = jnp.repeat(configurations, dimension, axis=0)
        configurations = configurations.at[:, site].set(
            jnp.tile(jnp.arange(dimension, dtype=jnp.int32), amplitudes.shape[0])
        )
        amplitudes = jnp.repeat(amplitudes * sign, dimension) * local.reshape((-1,))
    return configurations, amplitudes


def apply_compiled_to_coordinate(
    prepared: PreparedQuantumLattice, coordinate: ArrayLike, /
) -> tuple[Array, Array]:
    """Apply every compiled monomial without coalescing or dense materialization."""
    if not isinstance(prepared, PreparedQuantumLattice):
        raise TypeError("prepared must be PreparedQuantumLattice.")
    outputs = []
    amplitudes = []
    for monomial in prepared.monomials:
        coordinates, values = apply_monomial_to_coordinate(prepared, monomial, coordinate)
        outputs.append(coordinates)
        amplitudes.append(values)
    return jnp.concatenate(outputs, axis=0), jnp.concatenate(amplitudes, axis=0)


def _apply_compiled_unchecked(
    prepared: PreparedQuantumLattice, coordinate: Array, /
) -> tuple[Array, Array]:
    outputs = []
    amplitudes = []
    for monomial in prepared.monomials:
        coordinates, values = _apply_monomial_unchecked(prepared, monomial, coordinate)
        outputs.append(coordinates)
        amplitudes.append(values)
    return jnp.concatenate(outputs, axis=0), jnp.concatenate(amplitudes, axis=0)


class QuantumSectorOperator(AbstractLinearOperator):
    """Matrix-free compiled action with direct source/target rank and unrank.

    The action iterates only source-sector coordinates. It never enumerates an
    ambient tensor-product basis and intentionally has no dense fallback.
    """

    prepared: PreparedQuantumLattice
    charge_map: SectorChargeMap
    certification: ChargeMapCertification
    action_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedQuantumLattice,
        charge_map: SectorChargeMap,
        /,
    ):
        if not isinstance(prepared, PreparedQuantumLattice):
            raise TypeError("prepared must be PreparedQuantumLattice.")
        if not isinstance(charge_map, SectorChargeMap):
            raise TypeError("charge_map must be SectorChargeMap.")
        certification = certify_charge_map(prepared, charge_map)
        if not bool(certification.accepted):
            raise ValueError("Compiled terms do not have the declared unique charge map.")
        policy = prepared.plan.resources
        if (
            charge_map.source.dimension > policy.maximum_sector_dimension
            or charge_map.target.dimension > policy.maximum_sector_dimension
        ):
            raise ValueError(
                "A direct sector dimension exceeds compiler resource admission."
            )
        workspace = (
            prepared.plan.action_workspace_bytes
            + charge_map.target.dimension * np.dtype(np.complex128).itemsize
        )
        if workspace > policy.maximum_workspace_bytes:
            raise ValueError("Matrix-free sector action exceeds maximum_workspace_bytes.")
        source_space = ArraySpace(
            (charge_map.source.dimension,),
            dtype=np.complex128,
            space_id=f"quantum-sector:{charge_map.source.basis_id}",
        )
        target_space = ArraySpace(
            (charge_map.target.dimension,),
            dtype=np.complex128,
            space_id=f"quantum-sector:{charge_map.target.basis_id}",
        )
        self_adjoint = (
            prepared.specification.self_adjoint
            and charge_map.source.basis_id == charge_map.target.basis_id
            and charge_map.charge_delta == 0
        )
        self.prepared = prepared
        self.charge_map = charge_map
        self.certification = certification
        self.action_workspace_bytes = workspace
        self.source = source_space
        self.target = target_space
        self.properties = OperatorProperties(
            self_adjoint=self_adjoint,
            evidence={"self_adjoint": "construction"} if self_adjoint else None,
        )
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "matrix-free-quantum-sector-operator",
                "prepared": prepared.prepared_id,
                "charge_map": charge_map.map_id,
                "certification": certification.certification_id,
            }
        )

    def _mv_coordinates(self, vector: Array, /) -> Array:
        source = self.charge_map.source
        target = self.charge_map.target
        result = jnp.zeros((target.dimension,), dtype=jnp.complex128)
        fallback = target._coordinate_unchecked(jnp.asarray(0, dtype=jnp.int64))

        def add_column(index, accumulated):
            coordinate = source._coordinate_unchecked(jnp.asarray(index, dtype=jnp.int64))
            outputs, amplitudes = _apply_compiled_unchecked(self.prepared, coordinate)
            valid = jax.vmap(target._contains_unchecked)(outputs) & (
                jnp.abs(amplitudes) > 0
            )
            safe_outputs = jnp.where(valid[:, None], outputs, fallback[None, :])
            rows = jax.vmap(target._rank_unchecked)(safe_outputs)
            contributions = jnp.where(valid, amplitudes * vector[index], 0.0j)
            return accumulated.at[rows].add(contributions)

        return jax.lax.fori_loop(0, source.dimension, add_column, result)

    def _transpose_coordinates(self, vector: Array, /, *, conjugate: bool) -> Array:
        source = self.charge_map.source
        target = self.charge_map.target
        fallback = target._coordinate_unchecked(jnp.asarray(0, dtype=jnp.int64))
        result = jnp.zeros((source.dimension,), dtype=jnp.complex128)

        def set_column(index, accumulated):
            coordinate = source._coordinate_unchecked(jnp.asarray(index, dtype=jnp.int64))
            outputs, amplitudes = _apply_compiled_unchecked(self.prepared, coordinate)
            valid = jax.vmap(target._contains_unchecked)(outputs) & (
                jnp.abs(amplitudes) > 0
            )
            safe_outputs = jnp.where(valid[:, None], outputs, fallback[None, :])
            rows = jax.vmap(target._rank_unchecked)(safe_outputs)
            coefficients = jnp.conj(amplitudes) if conjugate else amplitudes
            value = jnp.sum(jnp.where(valid, coefficients * vector[rows], 0.0j))
            return accumulated.at[index].set(value)

        return jax.lax.fori_loop(0, source.dimension, set_column, result)

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(vector)
        return self.target.validate(self._mv_coordinates(value))

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        if self.properties.certifies("self_adjoint"):
            return self.source.validate(jnp.conj(self._mv_coordinates(jnp.conj(value))))
        return self.source.validate(self._transpose_coordinates(value, conjugate=False))

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        if self.properties.certifies("self_adjoint"):
            return self.source.validate(self._mv_coordinates(value))
        return self.source.validate(self._transpose_coordinates(value, conjugate=True))

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "QuantumSectorOperator intentionally forbids dense materialization."
        )


__all__ = [
    "QuantumSectorOperator",
    "apply_compiled_to_coordinate",
    "apply_monomial_to_coordinate",
]
