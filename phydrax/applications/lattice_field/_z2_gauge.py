#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellComplexTopology, OrientedEdgePathPlan
from ...operators.quantum import BasisStateSubspace, HilbertRegisterLayout
from ...solver import LocalHamiltonian, LocalHamiltonianTerm
from ...topology import CellSubcomplex, compute_homology, PrimeField


_PAULI_X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
_PAULI_Z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)


def _binary_support(topology: CellComplexTopology, degree: int, /) -> np.ndarray:
    incidence = topology.incidences[degree]
    relation = incidence.relation
    valid = np.asarray(relation.valid, dtype=bool)
    lower = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    upper = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    support = np.zeros(
        (topology.entities(degree + 1).count, topology.entities(degree).count),
        dtype=bool,
    )
    support[upper, lower] = True
    return support


def _gf2_rank(matrix: np.ndarray, /) -> int:
    value = np.asarray(matrix, dtype=np.uint8).copy()
    rows, columns = value.shape
    rank = 0
    for column in range(columns):
        pivots = np.flatnonzero(value[rank:, column])
        if pivots.size == 0:
            continue
        pivot = rank + int(pivots[0])
        value[[rank, pivot]] = value[[pivot, rank]]
        for row in range(rows):
            if row != rank and value[row, column]:
                value[row] ^= value[rank]
        rank += 1
        if rank == rows:
            break
    return rank


class Z2GaugeModel(StrictModule, NonTrainableState):
    """Finite edge-qubit Z2 gauge Hamiltonian in the electric basis."""

    topology: CellComplexTopology
    layout: HilbertRegisterLayout
    vertex_edge_support: Array
    face_edge_support: Array
    external_charges: Array
    electric_coupling: Array
    magnetic_coupling: Array
    chain_residual: Array
    valid: Array
    num_vertices: int = eqx.field(static=True)
    num_edges: int = eqx.field(static=True)
    num_faces: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)

    def __init__(
        self,
        topology: CellComplexTopology,
        /,
        *,
        electric_coupling: float,
        magnetic_coupling: float,
        external_charges: ArrayLike | None = None,
    ):
        if not isinstance(topology, CellComplexTopology):
            raise TypeError("topology must be CellComplexTopology.")
        if topology.dimension < 1:
            raise ValueError("Z2 gauge models require a topology with edges.")
        electric, magnetic = float(electric_coupling), float(magnetic_coupling)
        if not np.isfinite(electric) or not np.isfinite(magnetic):
            raise ValueError("Z2 gauge couplings must be finite.")
        num_vertices = topology.entities(0).count
        num_edges = topology.entities(1).count
        num_faces = topology.entities(2).count if topology.dimension >= 2 else 0
        charges = (
            np.zeros((num_vertices,), dtype=np.uint8)
            if external_charges is None
            else np.asarray(external_charges)
        )
        if charges.shape != (num_vertices,):
            raise ValueError("external_charges must provide one binary value per vertex.")
        if not np.all((charges == 0) | (charges == 1)):
            raise ValueError("external_charges must contain only zero or one.")
        vertex_edge = _binary_support(topology, 0).T
        face_edge = (
            _binary_support(topology, 1)
            if num_faces
            else np.zeros((0, num_edges), dtype=bool)
        )
        chain = (
            (vertex_edge.astype(np.uint8) @ face_edge.astype(np.uint8).T) % 2
            if num_faces
            else np.zeros((num_vertices, 0), dtype=np.uint8)
        )
        chain_residual = jnp.asarray(int(np.count_nonzero(chain)), dtype=jnp.int32)
        layout = HilbertRegisterLayout(
            tuple(f"edge:{int(value)}" for value in topology.entities(1).entity_ids),
            (2,) * num_edges,
        )
        model_id = canonical_fingerprint(
            {
                "kind": "z2-gauge-model",
                "topology": topology.topology_id,
                "electric_coupling": electric,
                "magnetic_coupling": magnetic,
                "external_charges": array_tree_fingerprint(charges),
                "convention": "electric-z-gauss-z-magnetic-x",
            }
        )
        self.topology = topology
        self.layout = layout
        self.vertex_edge_support = jnp.asarray(vertex_edge)
        self.face_edge_support = jnp.asarray(face_edge)
        self.external_charges = jnp.asarray(charges, dtype=jnp.int32)
        self.electric_coupling = jnp.asarray(electric)
        self.magnetic_coupling = jnp.asarray(magnetic)
        self.chain_residual = chain_residual
        self.valid = chain_residual == 0
        self.num_vertices = num_vertices
        self.num_edges = num_edges
        self.num_faces = num_faces
        self.model_id = model_id
        self.convention = "electric-z-gauss-z-magnetic-x"


class Z2GaussSector(StrictModule, NonTrainableState):
    """Resource-bounded computational-basis realization of one Gauss sector."""

    subspace: BasisStateSubspace
    constraint_rank: int = eqx.field(static=True)
    expected_dimension: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    sector_id: str = eqx.field(static=True)


def z2_gauss_eigenvalues(
    model: Z2GaugeModel,
    occupations: ArrayLike,
    /,
) -> Array:
    """Evaluate all vertex Gauss generators on binary edge occupations."""
    if not isinstance(model, Z2GaugeModel):
        raise TypeError("model must be Z2GaugeModel.")
    values = jnp.asarray(occupations, dtype=jnp.int32)
    if values.shape[-1:] != (model.num_edges,):
        raise ValueError("occupations must have trailing edge axis.")
    parity = (values @ model.vertex_edge_support.T.astype(jnp.int32)) % 2
    return 1 - 2 * parity


def prepare_z2_gauss_sector(
    model: Z2GaugeModel,
    /,
    *,
    maximum_basis_states: int = 1 << 20,
) -> Z2GaussSector:
    """Enumerate a small exact Gauss sector after a fail-before-allocation guard."""
    if not isinstance(model, Z2GaugeModel):
        raise TypeError("model must be Z2GaugeModel.")
    maximum = int(maximum_basis_states)
    if maximum <= 0:
        raise ValueError("maximum_basis_states must be positive.")
    physical_dimension = 1 << model.num_edges
    if physical_dimension > maximum:
        raise ValueError("Z2 physical basis exceeds maximum_basis_states.")
    indices = np.arange(physical_dimension, dtype=np.int64)
    shifts = np.arange(model.num_edges - 1, -1, -1, dtype=np.int64)
    occupations = ((indices[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    constraints = np.asarray(model.vertex_edge_support, dtype=np.uint8)
    charges = np.asarray(model.external_charges, dtype=np.uint8)
    matching = np.all((occupations @ constraints.T) % 2 == charges[None, :], axis=1)
    selected = indices[matching]
    if selected.size == 0:
        raise ValueError("External charges define an inconsistent Z2 Gauss sector.")
    rank = _gf2_rank(constraints)
    expected = 1 << (model.num_edges - rank)
    if selected.size != expected:
        raise ValueError("Enumerated Z2 sector dimension disagrees with GF(2) rank.")
    sector_id = canonical_fingerprint(
        {
            "kind": "z2-gauss-sector",
            "model": model.model_id,
            "basis_indices": array_tree_fingerprint(selected),
            "constraint_rank": rank,
        }
    )
    return Z2GaussSector(
        subspace=BasisStateSubspace(
            physical_dimension,
            selected,
            subspace_id=sector_id,
        ),
        constraint_rank=rank,
        expected_dimension=expected,
        model_id=model.model_id,
        sector_id=sector_id,
    )


def z2_gauss_terms(model: Z2GaugeModel, /) -> tuple[LocalHamiltonianTerm, ...]:
    """Return ordered Hermitian vertex Gauss generators."""
    if not isinstance(model, Z2GaugeModel):
        raise TypeError("model must be Z2GaugeModel.")
    terms = []
    support = np.asarray(model.vertex_edge_support)
    for vertex in range(model.num_vertices):
        edges = np.flatnonzero(support[vertex])
        if edges.size == 0:
            continue
        terms.append(
            LocalHamiltonianTerm.from_product(
                (_PAULI_Z,) * edges.size,
                tuple(model.layout.wire_ids[int(edge)] for edge in edges),
                term_id=f"{model.model_id}:gauss:{vertex}",
            )
        )
    return tuple(terms)


def z2_gauge_hamiltonian(model: Z2GaugeModel, /) -> LocalHamiltonian:
    """Lower electric links and magnetic plaquettes to one local Hamiltonian."""
    if not isinstance(model, Z2GaugeModel):
        raise TypeError("model must be Z2GaugeModel.")
    terms = [
        LocalHamiltonianTerm.from_product(
            (-model.electric_coupling * _PAULI_Z,),
            (wire_id,),
            term_id=f"{model.model_id}:electric:{edge}",
        )
        for edge, wire_id in enumerate(model.layout.wire_ids)
    ]
    face_support = np.asarray(model.face_edge_support)
    for face in range(model.num_faces):
        edges = np.flatnonzero(face_support[face])
        factors = [_PAULI_X] * edges.size
        factors[0] = -model.magnetic_coupling * factors[0]
        terms.append(
            LocalHamiltonianTerm.from_product(
                factors,
                tuple(model.layout.wire_ids[int(edge)] for edge in edges),
                term_id=f"{model.model_id}:magnetic:{face}",
            )
        )
    return LocalHamiltonian(
        model.layout,
        terms,
        hamiltonian_id=f"{model.model_id}:hamiltonian",
    )


def z2_loop_operator(
    model: Z2GaugeModel,
    paths: OrientedEdgePathPlan,
    path_index: int,
    /,
) -> LocalHamiltonianTerm:
    """Return the electric-basis X loop for one declared closed path."""
    if not isinstance(model, Z2GaugeModel):
        raise TypeError("model must be Z2GaugeModel.")
    if not isinstance(paths, OrientedEdgePathPlan) or not paths.require_closed:
        raise TypeError("paths must be a closed OrientedEdgePathPlan.")
    if paths.topology_id != model.topology.topology_id:
        raise ValueError("Z2 model and path topology identities must agree.")
    index = int(path_index)
    if index < 0 or index >= paths.num_paths:
        raise ValueError("path_index lies outside the path plan.")
    active = np.asarray(paths.valid[index])
    edges = np.asarray(paths.edge_indices[index])[active]
    counts = np.bincount(edges, minlength=model.num_edges) % 2
    support = np.flatnonzero(counts)
    if support.size == 0:
        raise ValueError("Z2 loop reduces to the identity and has no active support.")
    return LocalHamiltonianTerm.from_product(
        (_PAULI_X,) * support.size,
        tuple(model.layout.wire_ids[int(edge)] for edge in support),
        term_id=f"{model.model_id}:loop:{paths.path_names[index]}",
    )


def z2_homology(model: Z2GaugeModel, /):
    """Compute exact GF(2) homology for topological-sector labeling."""
    if not isinstance(model, Z2GaugeModel):
        raise TypeError("model must be Z2GaugeModel.")
    return compute_homology(
        CellSubcomplex.full(model.topology),
        coefficients=PrimeField(2),
        representatives="both",
    )


__all__ = [
    "Z2GaussSector",
    "Z2GaugeModel",
    "prepare_z2_gauss_sector",
    "z2_gauss_eigenvalues",
    "z2_gauss_terms",
    "z2_gauge_hamiltonian",
    "z2_homology",
    "z2_loop_operator",
]
