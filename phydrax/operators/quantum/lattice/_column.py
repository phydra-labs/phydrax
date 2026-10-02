#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""True outgoing sparse columns and one-path raw excitation proposals."""

from __future__ import annotations

from collections.abc import Callable
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._sampling import derive_key, SampleAddress
from ...._strict import StrictModule
from ....sparse import KeyGroupPlan, reduce_key_groups
from ....typing import (
    Bool,
    Complex128,
    Dim,
    Float64,
    Int32,
    parse,
    PRNGKey,
    Scalar,
    UInt32,
)
from ._address import (
    AddressWordDim,
    ConfigurationSiteDim,
    QuantumAddress,
    QuantumConfigurationDomain,
)
from ._column_compile import PreparedQuantumLatticeColumns, SparseQuantumMonomial
from ._operator import _ordered_factor_sign


class ColumnTargetDim(Dim):
    """Compact exact column target slots."""


@final
class QuantumColumnSource(StrictModule):
    __strict_contract__ = True

    address: QuantumAddress
    coordinate: Int32[ConfigurationSiteDim]
    valid: Bool[Scalar]
    guide_log: Float64[Scalar] | None = None
    guide_id: str | None = eqx.field(static=True, default=None)


@final
class RawQuantumExcitation(StrictModule):
    __strict_contract__ = True

    target_key: UInt32[AddressWordDim]
    matrix_element: Complex128[Scalar]
    p_raw: Float64[Scalar]
    route_index: Int32[Scalar]
    valid: Bool[Scalar]
    off_diagonal: Bool[Scalar]
    successful: Bool[Scalar]


@final
class QuantumColumn(StrictModule):
    __strict_contract__ = True

    target_keys: UInt32[ColumnTargetDim, AddressWordDim]
    matrix_elements: Complex128[ColumnTargetDim]
    valid: Bool[ColumnTargetDim]
    diagonal: Complex128[Scalar]
    raw_route_count: Int32[Scalar]
    unique_target_count: Int32[Scalar]
    cancellation_count: Int32[Scalar]
    successful: Bool[Scalar]


@final
class QuantumLatticeColumnOperator(StrictModule):
    """Configuration action, deliberately not an ArraySpace linear operator."""

    prepared: PreparedQuantumLatticeColumns
    domain: QuantumConfigurationDomain
    groups: KeyGroupPlan = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    self_adjoint: bool = eqx.field(static=True)
    raw_route_bound: int = eqx.field(static=True)
    route_offsets: tuple[int, ...] = eqx.field(static=True)
    sampling_address: SampleAddress = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedQuantumLatticeColumns,
        domain: QuantumConfigurationDomain,
        /,
    ) -> None:
        if not isinstance(prepared, PreparedQuantumLatticeColumns) or not isinstance(
            domain, QuantumConfigurationDomain
        ):
            raise TypeError(
                "Column operator requires prepared sparse columns and a configuration domain."
            )
        if prepared.physics_id != domain.physics_id:
            raise ValueError(
                "Operator and domain have different physical local spaces or mode order."
            )
        if domain.charge_group is not None:
            for monomial in prepared.monomials:
                changes = dict(monomial.charge_delta)
                delta = tuple(changes.get(label, 0) for label in domain.charge_labels)
                if domain.charge_group.normalize(delta) != domain.charge_group.zero:
                    raise ValueError(
                        "Every monomial must preserve the admitted exact charge domain."
                    )
        offsets = []
        offset = 0
        for monomial in prepared.monomials:
            offsets.append(offset)
            offset += monomial.branch_capacity
        self.prepared = prepared
        self.domain = domain
        self.groups = KeyGroupPlan(
            prepared.raw_route_bound,
            min(prepared.raw_route_bound, prepared.resources.maximum_column_targets),
            (2**32 - 1,) * domain.codec.word_count,
        )
        self.operator_id = canonical_fingerprint(
            {
                "kind": "quantum-domain-column-operator",
                "operator": prepared.operator_id,
                "domain": domain.domain_id,
            }
        )
        self.self_adjoint = prepared.self_adjoint
        self.raw_route_bound = prepared.raw_route_bound
        self.route_offsets = tuple(offsets)
        self.sampling_address = SampleAddress(
            "quantum-column", "raw-excitation", target=self.operator_id
        )

    def prepare_source(self, address: QuantumAddress, /) -> QuantumColumnSource:
        self.domain.validate_address(address)
        coordinate = self.domain.codec.decode(address.key_words)
        return QuantumColumnSource(
            address=address,
            coordinate=coordinate,
            valid=self.domain._contains_coordinates(address.key_words, coordinate),
        )

    def _source(
        self, source: QuantumAddress | QuantumColumnSource, /
    ) -> QuantumColumnSource:
        if isinstance(source, QuantumColumnSource):
            self.domain.validate_address(source.address)
            if source.coordinate.shape != (len(self.domain.site_ids),):
                raise ValueError(
                    "Source cache coordinate shape does not match its domain."
                )
            return source
        return self.prepare_source(source)

    def _walk(
        self,
        source: QuantumColumnSource,
        monomial: SparseQuantumMonomial,
        local_route: Array,
        offset: int,
        key: PRNGKey | None,
        /,
    ) -> RawQuantumExcitation:
        coordinate = source.coordinate
        amplitude = monomial.coefficient
        probability = jnp.asarray(1.0 / len(self.prepared.monomials), dtype=jnp.float64)
        valid = source.valid
        route_index = jnp.asarray(offset, dtype=jnp.int32)
        stride = 1
        remaining = local_route
        for role, (binding_index, width) in enumerate(
            zip(
                reversed(monomial.factor_bindings), reversed(monomial.widths), strict=True
            )
        ):
            binding = self.prepared.bindings[binding_index]
            topology = self.prepared.topologies[binding.topology_index]
            incoming = coordinate[binding.site]
            count = topology.counts[incoming]
            if key is None:
                choice = remaining % width
                remaining = remaining // width
            else:
                factor_key = derive_key(key, self.sampling_address, 1, role)
                choice = jax.random.randint(
                    factor_key, (), 0, jnp.maximum(count, 1), dtype=jnp.int32
                )
            permitted = topology.valid[incoming, choice]
            sign = _ordered_factor_sign(coordinate, binding.predecessors)
            amplitude = (amplitude * sign.astype(jnp.complex128)) * binding.values[
                incoming, choice
            ]
            valid = valid & permitted
            probability = probability / jnp.maximum(count, 1).astype(jnp.float64)
            coordinate = coordinate.at[binding.site].set(
                topology.outputs[incoming, choice]
            )
            route_index = route_index + choice * stride
            stride *= width
        target = self.domain.codec._encode(coordinate)
        finite = jnp.isfinite(amplitude) & jnp.isfinite(probability) & (probability > 0)
        domain_valid = self.domain._contains_coordinates(target, coordinate)
        successful = source.valid & (~valid | (finite & domain_valid))
        return RawQuantumExcitation(
            target_key=target,
            matrix_element=jnp.where(valid, amplitude, jnp.complex128(0)),
            p_raw=probability,
            route_index=route_index,
            valid=valid,
            off_diagonal=valid & jnp.any(target != source.address.key_words),
            successful=successful,
        )

    def raw_route(
        self, address: QuantumAddress | QuantumColumnSource, index: ArrayLike, /
    ) -> RawQuantumExcitation:
        source = self._source(address)
        raw = jnp.asarray(index)
        if raw.shape != () or not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError("Raw route index must be an integer scalar.")
        admitted = (raw >= 0) & (raw < self.raw_route_bound)
        safe = jnp.clip(raw, 0, self.raw_route_bound - 1).astype(jnp.int32)
        offsets = jnp.asarray(self.route_offsets, dtype=jnp.int32)
        component = jnp.sum(safe >= offsets, dtype=jnp.int32) - 1
        branches: list[Callable[[Array], RawQuantumExcitation]] = []
        for monomial, offset in zip(
            self.prepared.monomials, self.route_offsets, strict=True
        ):

            def branch(
                route: Array,
                recipe: SparseQuantumMonomial = monomial,
                start: int = offset,
            ) -> RawQuantumExcitation:
                return self._walk(source, recipe, route - start, start, None)

            branches.append(branch)
        result = jax.lax.switch(component, branches, safe)
        return eqx.tree_at(
            lambda value: (value.valid, value.off_diagonal, value.successful),
            result,
            (
                result.valid & admitted,
                result.off_diagonal & admitted,
                result.successful & admitted,
            ),
        )

    def sample_raw_excitation(
        self, key: PRNGKey, address: QuantumAddress | QuantumColumnSource, /
    ) -> RawQuantumExcitation:
        root = parse(key, PRNGKey, "key")
        source = self._source(address)
        component = jax.random.randint(
            derive_key(root, self.sampling_address, 0),
            (),
            0,
            len(self.prepared.monomials),
            dtype=jnp.int32,
        )
        branches: list[Callable[[Array], RawQuantumExcitation]] = []
        for monomial, offset in zip(
            self.prepared.monomials, self.route_offsets, strict=True
        ):

            def branch(
                unused: Array,
                recipe: SparseQuantumMonomial = monomial,
                start: int = offset,
            ) -> RawQuantumExcitation:
                return self._walk(source, recipe, unused, start, root)

            branches.append(branch)
        return jax.lax.switch(component, branches, jnp.asarray(0, dtype=jnp.int32))

    def outgoing_column(
        self, address: QuantumAddress | QuantumColumnSource, /
    ) -> QuantumColumn:
        source = self._source(address)

        def route_scan(carry: Array, index: Array) -> tuple[Array, RawQuantumExcitation]:
            route = self.raw_route(source, index)
            return carry & route.successful, route

        successful, routes = jax.lax.scan(
            route_scan, source.valid, jnp.arange(self.raw_route_bound, dtype=jnp.int32)
        )
        grouped = self.groups.build(routes.target_key, routes.valid)
        total, evidence = reduce_key_groups(
            grouped,
            routes.matrix_element,
            accumulation="compensated",
            value_valid=routes.valid,
        )
        values = total.value
        diagonal_mask = grouped.group_active & jnp.all(
            grouped.group_keys
            == jnp.broadcast_to(source.address.key_words, grouped.group_keys.shape),
            axis=-1,
        )
        valid = grouped.group_active & (values != 0)
        return QuantumColumn(
            target_keys=grouped.group_keys,
            matrix_elements=values,
            valid=valid,
            diagonal=jnp.sum(
                jnp.where(diagonal_mask, values, jnp.complex128(0)), dtype=jnp.complex128
            ),
            raw_route_count=jnp.sum(routes.valid, dtype=jnp.int32),
            unique_target_count=jnp.sum(grouped.group_active, dtype=jnp.int32),
            cancellation_count=jnp.sum(grouped.group_active & ~valid, dtype=jnp.int32),
            successful=successful & grouped.evidence.successful & evidence.successful,
        )

    def diagonal(self, address: QuantumAddress | QuantumColumnSource, /) -> Array:
        source = self._source(address)

        def route_scan(carry: Array, index: Array) -> tuple[Array, tuple[Array, Array]]:
            route = self.raw_route(source, index)
            diagonal_route = route.valid & ~route.off_diagonal
            return carry & route.successful, (route.matrix_element, diagonal_route)

        successful, (values, valid) = jax.lax.scan(
            route_scan, source.valid, jnp.arange(self.raw_route_bound, dtype=jnp.int32)
        )
        plan = KeyGroupPlan(self.raw_route_bound, 1, 0)
        grouped = plan.build(jnp.zeros((self.raw_route_bound,), dtype=jnp.int32), valid)
        total, evidence = reduce_key_groups(
            grouped, values, accumulation="compensated", value_valid=valid
        )
        return eqx.error_if(
            total.value[0],
            ~(successful & evidence.successful & grouped.evidence.successful),
            "Exact quantum diagonal traversal failed.",
        )


__all__ = [
    "QuantumColumn",
    "QuantumColumnSource",
    "QuantumLatticeColumnOperator",
    "RawQuantumExcitation",
]
