#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sparse transition topology and occurrence-owned numerical bindings."""

from __future__ import annotations

from math import prod
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ....typing import Bool, Complex128, Dim, Int32, Scalar
from ._compile import _expanded_monomials, CompiledMonomial
from ._model import LocalOperatorPlan, QuantumLatticeSpecification
from ._operator import _fermion_predecessor_sites


class LocalInputDim(Dim):
    """Local incoming-state axis."""


class LocalTransitionDim(Dim):
    """Padded sparse outgoing local transitions."""


@final
class QuantumColumnResourcePolicy(StrictModule):
    maximum_monomials: int = eqx.field(static=True)
    maximum_factors_per_monomial: int = eqx.field(static=True)
    maximum_transition_table_bytes: int = eqx.field(static=True)
    maximum_raw_routes: int = eqx.field(static=True)
    maximum_column_targets: int = eqx.field(static=True)
    maximum_decoded_coordinate_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_monomials: int = 4096,
        maximum_factors_per_monomial: int = 64,
        maximum_transition_table_bytes: int = 67108864,
        maximum_raw_routes: int = 65536,
        maximum_column_targets: int = 65536,
        maximum_decoded_coordinate_bytes: int = 67108864,
        maximum_workspace_bytes: int = 134217728,
    ) -> None:
        limits = (
            maximum_monomials,
            maximum_factors_per_monomial,
            maximum_transition_table_bytes,
            maximum_raw_routes,
            maximum_column_targets,
            maximum_decoded_coordinate_bytes,
            maximum_workspace_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in limits):
            raise TypeError("Column resource limits must be integers.")
        if any(value < 1 for value in limits):
            raise ValueError("Column resource limits must be positive.")
        self.maximum_monomials = maximum_monomials
        self.maximum_factors_per_monomial = maximum_factors_per_monomial
        self.maximum_transition_table_bytes = maximum_transition_table_bytes
        self.maximum_raw_routes = maximum_raw_routes
        self.maximum_column_targets = maximum_column_targets
        self.maximum_decoded_coordinate_bytes = maximum_decoded_coordinate_bytes
        self.maximum_workspace_bytes = maximum_workspace_bytes
        self.policy_id = canonical_fingerprint(
            {"kind": "quantum-column-resources", "limits": limits}
        )


@final
class SparseLocalTransitionTopology(StrictModule):
    __strict_contract__ = True

    outputs: Int32[LocalInputDim, LocalTransitionDim]
    valid: Bool[LocalInputDim, LocalTransitionDim]
    counts: Int32[LocalInputDim]
    operator_id: str = eqx.field(static=True)
    width: int = eqx.field(static=True)


@final
class SparseLocalTransitionBinding(StrictModule):
    __strict_contract__ = True

    values: Complex128[LocalInputDim, LocalTransitionDim]
    topology_index: int = eqx.field(static=True)
    site: int = eqx.field(static=True)
    predecessors: tuple[int, ...] = eqx.field(static=True)
    logical_slot: tuple[int, bool, int] = eqx.field(static=True)


@final
class SparseQuantumMonomial(StrictModule):
    __strict_contract__ = True

    coefficient: Complex128[Scalar]
    factor_bindings: tuple[int, ...] = eqx.field(static=True)
    widths: tuple[int, ...] = eqx.field(static=True)
    branch_capacity: int = eqx.field(static=True)
    charge_delta: tuple[tuple[str, int], ...] = eqx.field(static=True)
    source_term_id: str = eqx.field(static=True)
    source_term_ordinal: int = eqx.field(static=True)
    adjoint_component: bool = eqx.field(static=True)


@final
class PreparedQuantumLatticeColumns(StrictModule):
    """No specification or dense local matrix is retained in this runtime tree."""

    topologies: tuple[SparseLocalTransitionTopology, ...]
    bindings: tuple[SparseLocalTransitionBinding, ...]
    monomials: tuple[SparseQuantumMonomial, ...]
    resources: QuantumColumnResourcePolicy = eqx.field(static=True)
    site_ids: tuple[str, ...] = eqx.field(static=True)
    space_ids: tuple[str, ...] = eqx.field(static=True)
    local_dimensions: tuple[int, ...] = eqx.field(static=True)
    physics_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    structural_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    self_adjoint: bool = eqx.field(static=True)
    raw_route_bound: int = eqx.field(static=True)
    transition_table_bytes: int = eqx.field(static=True)
    decoded_coordinate_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)


def _bind_local_transition_occurrence(
    specification: QuantumLatticeSpecification,
    factor: LocalOperatorPlan,
    topology_index: int,
    outputs: npt.NDArray[np.int32],
    valid: npt.NDArray[np.bool_],
    logical_slot: tuple[int, bool, int],
    /,
) -> tuple[npt.NDArray[np.complex128], int, int, tuple[int, ...], tuple[int, bool, int]]:
    """Gather one occurrence's numeric values and fermionic binding metadata."""
    matrix = np.asarray(factor.matrix, dtype=np.complex128)
    values = np.where(
        valid,
        matrix[outputs, np.arange(factor.space.dimension, dtype=np.int32)[:, None]],
        np.complex128(0),
    )
    predecessors: tuple[int, ...] = ()
    if factor.fermion_parity:
        mode = factor.space.fermion_mode_label
        if mode is None:
            raise ValueError("Odd fermion factor requires mode identity.")
        predecessors = _fermion_predecessor_sites(specification, mode)
    return (
        values,
        topology_index,
        specification.site_ids.index(factor.space.site_id),
        predecessors,
        logical_slot,
    )


def _prepare_column_hosts(
    specification: QuantumLatticeSpecification,
    resources: QuantumColumnResourcePolicy,
    expanded: tuple[tuple[int, CompiledMonomial], ...],
    /,
) -> tuple[
    list[
        tuple[
            npt.NDArray[np.int32], npt.NDArray[np.bool_], npt.NDArray[np.int32], str, int
        ]
    ],
    list[
        tuple[
            npt.NDArray[np.complex128], int, int, tuple[int, ...], tuple[int, bool, int]
        ]
    ],
    list[SparseQuantumMonomial],
    int,
    int,
]:
    """Admit sparse tables, deduplicate topology, and bind ordered occurrences."""
    topology_hosts: list[
        tuple[
            npt.NDArray[np.int32], npt.NDArray[np.bool_], npt.NDArray[np.int32], str, int
        ]
    ] = []
    topology_slots: dict[str, int] = {}
    binding_hosts: list[
        tuple[
            npt.NDArray[np.complex128], int, int, tuple[int, ...], tuple[int, bool, int]
        ]
    ] = []
    recipes: list[SparseQuantumMonomial] = []
    table_bytes = 0
    route_bound = 0
    for ordinal, monomial in expanded:
        slots = []
        widths = []
        for position, factor in enumerate(monomial.factors):
            support = np.asarray(factor.support_mask)
            counts = np.count_nonzero(support, axis=0).astype(np.int32)
            width = int(counts.max())
            table_bytes += factor.space.dimension * width * 16
            if factor.operator_id not in topology_slots:
                table_bytes += factor.space.dimension * (width * 5 + 4)
            if table_bytes > resources.maximum_transition_table_bytes:
                raise ValueError("Sparse transition-table byte budget exceeded.")
            if factor.operator_id not in topology_slots:
                outputs = np.zeros((factor.space.dimension, width), dtype=np.int32)
                valid = np.zeros_like(outputs, dtype=np.bool_)
                for incoming in range(factor.space.dimension):
                    rows = np.flatnonzero(support[:, incoming])
                    outputs[incoming, : rows.size] = rows
                    valid[incoming, : rows.size] = True
                topology_slots[factor.operator_id] = len(topology_hosts)
                topology_hosts.append((outputs, valid, counts, factor.operator_id, width))
            topology_index = topology_slots[factor.operator_id]
            outputs, valid, _, _, _ = topology_hosts[topology_index]
            binding = _bind_local_transition_occurrence(
                specification,
                factor,
                topology_index,
                outputs,
                valid,
                (ordinal, monomial.adjoint_component, position),
            )
            slots.append(len(binding_hosts))
            widths.append(width)
            binding_hosts.append(binding)
        capacity = prod(widths)
        route_bound += capacity
        if (
            route_bound > resources.maximum_raw_routes
            or route_bound > np.iinfo(np.int32).max
        ):
            raise ValueError("Physical sparse raw-route budget exceeded.")
        recipes.append(
            SparseQuantumMonomial(
                coefficient=jnp.asarray(monomial.coefficient, dtype=jnp.complex128),
                factor_bindings=tuple(slots),
                widths=tuple(widths),
                branch_capacity=capacity,
                charge_delta=monomial.charge_delta,
                source_term_id=monomial.source_term_id,
                source_term_ordinal=ordinal,
                adjoint_component=monomial.adjoint_component,
            )
        )
    return topology_hosts, binding_hosts, recipes, route_bound, table_bytes


def _finalize_quantum_lattice_columns(
    specification: QuantumLatticeSpecification,
    resources: QuantumColumnResourcePolicy,
    hosts: tuple[
        list[
            tuple[
                npt.NDArray[np.int32],
                npt.NDArray[np.bool_],
                npt.NDArray[np.int32],
                str,
                int,
            ]
        ],
        list[
            tuple[
                npt.NDArray[np.complex128],
                int,
                int,
                tuple[int, ...],
                tuple[int, bool, int],
            ]
        ],
        list[SparseQuantumMonomial],
        int,
        int,
    ],
    decoded_bytes: int,
    /,
) -> PreparedQuantumLatticeColumns:
    """Admit runtime scratch before materializing array leaves and identities."""
    topology_hosts, binding_hosts, recipes, route_bound, table_bytes = hosts
    word_count = max(
        1, (sum((d - 1).bit_length() for d in specification.local_dimensions) + 31) // 32
    )
    targets = min(route_bound, resources.maximum_column_targets)
    workspace = (
        decoded_bytes
        + route_bound * (word_count * 4 + 64)
        + targets * (word_count * 4 + 48)
    )
    if workspace > resources.maximum_workspace_bytes:
        raise ValueError("Sparse column scratch budget exceeded.")
    topologies = tuple(
        SparseLocalTransitionTopology(
            outputs=jnp.asarray(out, dtype=jnp.int32),
            valid=jnp.asarray(valid, dtype=jnp.bool_),
            counts=jnp.asarray(counts, dtype=jnp.int32),
            operator_id=identity,
            width=width,
        )
        for out, valid, counts, identity, width in topology_hosts
    )
    bindings = tuple(
        SparseLocalTransitionBinding(
            values=jnp.asarray(values, dtype=jnp.complex128),
            topology_index=topology,
            site=site,
            predecessors=predecessors,
            logical_slot=slot,
        )
        for values, topology, site, predecessors, slot in binding_hosts
    )
    physics = canonical_fingerprint(
        {
            "kind": "quantum-configuration-physics",
            "spaces": tuple(s.space_id for s in specification.spaces),
            "fermion_order": None
            if specification.fermion_mode_order is None
            else specification.fermion_mode_order.order_id,
        }
    )
    structural = canonical_fingerprint(
        {
            "kind": "quantum-column-structure",
            "specification": specification.specification_id,
            "slots": tuple(
                (binding.logical_slot, topologies[binding.topology_index].operator_id)
                for binding in bindings
            ),
        }
    )
    operator = canonical_fingerprint(
        {
            "kind": "quantum-column-operator",
            "structure": structural,
            "numeric": array_tree_fingerprint(
                tuple(m.coefficient for m in recipes) + tuple(b.values for b in bindings)
            ),
        }
    )
    return PreparedQuantumLatticeColumns(
        topologies=topologies,
        bindings=bindings,
        monomials=tuple(recipes),
        resources=resources,
        site_ids=specification.site_ids,
        space_ids=tuple(s.space_id for s in specification.spaces),
        local_dimensions=specification.local_dimensions,
        physics_id=physics,
        specification_id=specification.specification_id,
        structural_id=structural,
        operator_id=operator,
        prepared_id=canonical_fingerprint(
            {"operator": operator, "resources": resources.policy_id}
        ),
        self_adjoint=specification.self_adjoint,
        raw_route_bound=route_bound,
        transition_table_bytes=table_bytes,
        decoded_coordinate_bytes=decoded_bytes,
        workspace_bytes=workspace,
    )


def prepare_quantum_lattice_columns(
    specification: QuantumLatticeSpecification,
    resources: QuantumColumnResourcePolicy,
    /,
) -> PreparedQuantumLatticeColumns:
    if not isinstance(specification, QuantumLatticeSpecification) or not isinstance(
        resources, QuantumColumnResourcePolicy
    ):
        raise TypeError(
            "Sparse columns require a specification and column resource policy."
        )
    if any(d > np.iinfo(np.int32).max for d in specification.local_dimensions):
        raise ValueError("Sparse local-state indices must fit int32.")
    decoded_bytes = len(specification.spaces) * 4
    if decoded_bytes > resources.maximum_decoded_coordinate_bytes:
        raise ValueError("Decoded coordinate workset budget exceeded.")
    if (
        sum(2 if term.add_adjoint else 1 for term in specification.terms)
        > resources.maximum_monomials
    ):
        raise ValueError("Sparse monomial budget exceeded.")
    if any(
        len(term.factors) > resources.maximum_factors_per_monomial
        for term in specification.terms
    ):
        raise ValueError("Sparse factor budget exceeded.")
    expanded = tuple(
        (ordinal, monomial)
        for ordinal, term in enumerate(specification.terms)
        for monomial in _expanded_monomials(term)
    )
    hosts = _prepare_column_hosts(specification, resources, expanded)
    return _finalize_quantum_lattice_columns(
        specification, resources, hosts, decoded_bytes
    )


def refresh_quantum_lattice_columns(
    prepared: PreparedQuantumLatticeColumns,
    specification: QuantumLatticeSpecification,
    /,
) -> PreparedQuantumLatticeColumns:
    """Rebind every logical occurrence; adjoints are derived from fresh originals."""
    if not isinstance(prepared, PreparedQuantumLatticeColumns):
        raise TypeError("Expected prepared sparse quantum columns.")
    replacement = prepare_quantum_lattice_columns(specification, prepared.resources)
    if (
        replacement.structural_id != prepared.structural_id
        or replacement.physics_id != prepared.physics_id
    ):
        raise ValueError(
            "Numeric refresh requires identical ordered terms and sparse support."
        )
    return eqx.tree_at(lambda value: value.topologies, replacement, prepared.topologies)


__all__ = [
    "PreparedQuantumLatticeColumns",
    "QuantumColumnResourcePolicy",
    "prepare_quantum_lattice_columns",
    "refresh_quantum_lattice_columns",
]
