#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-graph gauge-invariant SU(2) spin-network basis and area evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import pi, prod, sqrt

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...operators.quantum.lattice import (
    prepare_su2_sector_basis,
    PreparedSU2SectorBasis,
    SU2CouplingTreePlan,
    SU2SectorResourcePolicy,
)


class SpinNetworkEdge(StrictModule):
    label: str = eqx.field(static=True)
    source: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    twice_spin: int = eqx.field(static=True)
    edge_id: str = eqx.field(static=True)

    def __init__(self, label: str, source: str, target: str, twice_spin: int, /):
        values = tuple(str(value) for value in (label, source, target))
        spin = int(twice_spin)
        if any(not value for value in values) or values[1] == values[2] or spin < 0:
            raise ValueError("Spin-network edge identity/endpoints/spin are invalid.")
        self.label, self.source, self.target = values
        self.twice_spin = spin
        self.edge_id = canonical_fingerprint(
            {
                "kind": "oriented-spin-network-edge",
                "label": values[0],
                "source": values[1],
                "target": values[2],
                "twice_spin": spin,
            }
        )


class SpinNetworkGraphPlan(StrictModule):
    vertices: tuple[str, ...] = eqx.field(static=True)
    edges: tuple[SpinNetworkEdge, ...]
    vertex_edge_order: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True)
    resources: SU2SectorResourcePolicy = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: Sequence[str],
        edges: Sequence[SpinNetworkEdge],
        vertex_edge_order: Mapping[str, Sequence[str]],
        resources: SU2SectorResourcePolicy,
        /,
    ):
        vertex_values = tuple(str(value) for value in vertices)
        edge_values = tuple(edges)
        if (
            not vertex_values
            or any(not value for value in vertex_values)
            or len(set(vertex_values)) != len(vertex_values)
        ):
            raise ValueError("Spin-network vertices must be unique and non-empty.")
        if not edge_values or any(
            not isinstance(value, SpinNetworkEdge) for value in edge_values
        ):
            raise TypeError("At least one SpinNetworkEdge is required.")
        if len({value.label for value in edge_values}) != len(edge_values):
            raise ValueError("Spin-network edge labels must be unique.")
        if any(
            value.source not in vertex_values or value.target not in vertex_values
            for value in edge_values
        ):
            raise ValueError("A spin-network edge endpoint is outside the graph.")
        if not isinstance(resources, SU2SectorResourcePolicy):
            raise TypeError("resources must be SU2SectorResourcePolicy.")
        orders = tuple(
            (str(vertex), tuple(str(edge) for edge in order))
            for vertex, order in sorted(vertex_edge_order.items())
        )
        if tuple(vertex for vertex, _ in orders) != tuple(sorted(vertex_values)):
            raise ValueError("vertex_edge_order must specify every vertex exactly once.")
        edge_lookup = {value.label: value for value in edge_values}
        for vertex, order in orders:
            incident = {
                value.label
                for value in edge_values
                if value.source == vertex or value.target == vertex
            }
            if set(order) != incident or len(order) != len(incident) or len(order) < 2:
                raise ValueError(
                    "Every vertex edge order must list its incident edges once."
                )
            if any(label not in edge_lookup for label in order):
                raise ValueError("A vertex edge order references an unknown edge.")
        self.vertices = vertex_values
        self.edges = edge_values
        self.vertex_edge_order = orders
        self.resources = resources
        self.graph_id = canonical_fingerprint(
            {
                "kind": "fixed-su2-spin-network-graph-plan",
                "vertices": vertex_values,
                "edges": tuple(value.edge_id for value in edge_values),
                "vertex_edge_order": orders,
                "resources": resources.policy_id,
                "convention": "oriented-edges-su2-self-dual-left-associated-intertwiners",
            }
        )


class SpinNetworkEvidence(StrictModule):
    vertex_orthonormality_residuals: Array
    vertex_projector_residuals: Array
    gauge_invariant: Array
    finite: Array
    accepted: Array
    graph_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class PreparedSpinNetwork(StrictModule):
    plan: SpinNetworkGraphPlan = eqx.field(static=True)
    vertex_bases: tuple[PreparedSU2SectorBasis, ...]
    vertex_multiplicities: Array
    edge_area_eigenvalues: Array
    basis_dimension: int = eqx.field(static=True)
    evidence: SpinNetworkEvidence
    prepared_id: str = eqx.field(static=True)


class SpinNetworkState(StrictModule):
    amplitudes: Array
    norm: Array
    normalized: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def prepare_spin_network(
    plan: SpinNetworkGraphPlan,
    /,
    *,
    immirzi_parameter: float,
    planck_length_squared: float,
) -> PreparedSpinNetwork:
    if not isinstance(plan, SpinNetworkGraphPlan):
        raise TypeError("plan must be SpinNetworkGraphPlan.")
    gamma = float(immirzi_parameter)
    planck = float(planck_length_squared)
    if not np.isfinite(gamma) or gamma <= 0.0 or not np.isfinite(planck) or planck <= 0.0:
        raise ValueError(
            "Immirzi parameter and Planck length squared must be positive and finite."
        )
    edge_lookup = {value.label: value for value in plan.edges}
    vertex_bases = []
    for vertex, order in plan.vertex_edge_order:
        spins = tuple(edge_lookup[label].twice_spin for label in order)
        vertex_bases.append(
            prepare_su2_sector_basis(
                SU2CouplingTreePlan(
                    tuple(f"{vertex}:{label}" for label in order),
                    spins,
                    0,
                    plan.resources,
                )
            )
        )
    multiplicities = np.asarray(
        [value.plan.multiplicity_dimension for value in vertex_bases], dtype=np.int32
    )
    dimension = prod(int(value) for value in multiplicities)
    if dimension > plan.resources.maximum_sector_dimension:
        raise ValueError("Global intertwiner basis exceeds sector resource admission.")
    areas = np.asarray(
        [
            8.0
            * pi
            * gamma
            * planck
            * 0.5
            * sqrt(edge.twice_spin * (edge.twice_spin + 2))
            for edge in plan.edges
        ]
    )
    orthonormality = jnp.stack(
        tuple(value.evidence.orthonormality_residual for value in vertex_bases)
    )
    projector = jnp.stack(
        tuple(value.evidence.projector_idempotence_residual for value in vertex_bases)
    )
    finite = jnp.all(jnp.isfinite(orthonormality)) & jnp.all(jnp.isfinite(areas))
    accepted = finite & jnp.all(
        jnp.stack(tuple(value.evidence.accepted for value in vertex_bases))
    )
    evidence = SpinNetworkEvidence(
        vertex_orthonormality_residuals=orthonormality,
        vertex_projector_residuals=projector,
        gauge_invariant=accepted,
        finite=finite,
        accepted=accepted,
        graph_id=plan.graph_id,
        claim="finite-fixed-graph-su2-gauss-invariant-spin-network-basis",
    )
    bases = tuple(vertex_bases)
    return PreparedSpinNetwork(
        plan=plan,
        vertex_bases=bases,
        vertex_multiplicities=jnp.asarray(multiplicities),
        edge_area_eigenvalues=jnp.asarray(areas),
        basis_dimension=dimension,
        evidence=evidence,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-fixed-su2-spin-network",
                "graph": plan.graph_id,
                "vertex_bases": tuple(value.prepared_id for value in bases),
                "areas": array_tree_fingerprint(areas),
                "immirzi_parameter": gamma,
                "planck_length_squared": planck,
            }
        ),
    )


def spin_network_state(
    prepared: PreparedSpinNetwork,
    amplitudes: ArrayLike,
    /,
    *,
    normalization_tolerance: float = 1e-10,
) -> SpinNetworkState:
    if not isinstance(prepared, PreparedSpinNetwork):
        raise TypeError("prepared must be PreparedSpinNetwork.")
    values = jnp.asarray(amplitudes)
    if values.shape != (prepared.basis_dimension,):
        raise ValueError("Spin-network amplitudes have the wrong basis dimension.")
    norm = jnp.real(jnp.vdot(values, values))
    finite = jnp.all(jnp.isfinite(values)) & jnp.isfinite(norm)
    normalized = finite & (jnp.abs(norm - 1.0) <= float(normalization_tolerance))
    return SpinNetworkState(
        amplitudes=values,
        norm=norm,
        normalized=normalized,
        prepared_id=prepared.prepared_id,
        claim="finite-fixed-graph-spin-network-state-no-graph-changing-or-continuum-claim",
    )


__all__ = [
    "PreparedSpinNetwork",
    "SpinNetworkEdge",
    "SpinNetworkEvidence",
    "SpinNetworkGraphPlan",
    "SpinNetworkState",
    "prepare_spin_network",
    "spin_network_state",
]
