#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._polynomial._orthogonal import legendre_rule_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_mesh import CellBlock, CellMesh
from ._generic import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    IntegrationDomain,
)
from ._high_order import ReferenceNodalFamily
from ._hp import (
    finite_element_hp_workset_plan,
    FiniteElementHPLineage,
    FiniteElementHPTopology,
    FiniteElementHPTransferPlan,
    FiniteElementHPWorksetPlan,
)
from ._reference import FiniteElementSpec


_HP_RELATIONS = {"conforming": 0, "mortar": 1, "exterior": 2, "periodic": 3}
_HP_LINEAGE_RELATIONS = {"unchanged": 0, "refinement": 1, "coarsening": 2}
_QUAD_FACETS = ((0, 1), (1, 2), (2, 3), (3, 0))
_HEX_FACETS = (
    (0, 3, 2, 1),
    (1, 2, 6, 5),
    (4, 5, 6, 7),
    (0, 4, 7, 3),
    (0, 1, 5, 4),
    (3, 7, 6, 2),
)


def _cell_facets(cell_kind: str, /) -> tuple[tuple[int, ...], ...]:
    if cell_kind == "quadrilateral":
        return _QUAD_FACETS
    if cell_kind == "hexahedron":
        return _HEX_FACETS
    raise ValueError("Adaptive hp facets require quadrilateral or hexahedron cells.")


def _corner_points(dimension: int, /) -> np.ndarray:
    if dimension == 2:
        return np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    if dimension == 3:
        return np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
                (0.0, 1.0, 1.0),
            )
        )
    raise ValueError("Tensor hp geometry requires dimension two or three.")


def _multilinear_map(vertices: np.ndarray, points: np.ndarray, /) -> np.ndarray:
    dimension = points.shape[1]
    corners = _corner_points(dimension)
    factors = np.where(
        corners[None, :, :] == 0.0,
        1.0 - points[:, None, :],
        points[:, None, :],
    )
    basis = np.prod(factors, axis=-1)
    return basis @ vertices


def _child_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    child_ordinal: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    midpoint = 0.5 * (lower + upper)
    bits = np.asarray(
        tuple((child_ordinal >> axis) & 1 for axis in range(lower.size)),
        dtype=bool,
    )
    return np.where(bits, midpoint, lower), np.where(bits, upper, midpoint)


class FiniteElementHPGeometry(StrictModule, NonTrainableState):
    """Fixed-capacity cell geometry and root-reference boxes for one hp forest."""

    cell_vertices: Array
    reference_lower: Array
    reference_upper: Array
    allocated: Array
    topology_id: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: FiniteElementHPTopology,
        cell_vertices: ArrayLike,
        reference_lower: ArrayLike,
        reference_upper: ArrayLike,
        /,
    ):
        if not isinstance(topology, FiniteElementHPTopology):
            raise TypeError("topology must be FiniteElementHPTopology.")
        vertices = np.asarray(cell_vertices)
        lower = np.asarray(reference_lower)
        upper = np.asarray(reference_upper)
        allocated = np.asarray(topology.allocated)
        vertex_count = 2**topology.dimension
        if topology.dimension == 2:
            vertex_count = 4
        if (
            vertices.shape[:2] != (topology.capacity, vertex_count)
            or vertices.ndim != 3
            or vertices.shape[2] < topology.dimension
            or lower.shape != (topology.capacity, topology.dimension)
            or upper.shape != lower.shape
            or not np.issubdtype(vertices.dtype, np.inexact)
            or np.any(~np.isfinite(vertices[allocated]))
            or np.any(lower[allocated] < 0.0)
            or np.any(upper[allocated] > 1.0)
            or np.any(lower[allocated] >= upper[allocated])
            or np.any(vertices[~allocated] != 0.0)
            or np.any(lower[~allocated] != 0.0)
            or np.any(upper[~allocated] != 0.0)
        ):
            raise ValueError("hp geometry arrays or inactive padding are invalid.")
        self.cell_vertices = jnp.asarray(vertices)
        self.reference_lower = jnp.asarray(lower)
        self.reference_upper = jnp.asarray(upper)
        self.allocated = topology.allocated
        self.topology_id = topology.topology_id
        self.cell_kind = topology.cell_kind
        self.capacity = topology.capacity
        self.dimension = topology.dimension
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-geometry",
                "topology": topology.plan_id,
                "vertices": array_tree_fingerprint(vertices),
                "lower": array_tree_fingerprint(lower),
                "upper": array_tree_fingerprint(upper),
            }
        )


class FiniteElementHPGeometryEvidence(StrictModule, NonTrainableState):
    child_coverage_error: Array
    interface_coordinate_error: Array
    minimum_measure: Array
    tolerance: float = eqx.field(static=True)
    child_coverage_passed: bool = eqx.field(static=True)
    interfaces_watertight: bool = eqx.field(static=True)
    positive_measures: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        child_coverage_error: ArrayLike,
        interface_coordinate_error: ArrayLike,
        minimum_measure: ArrayLike,
        tolerance: float,
        /,
    ):
        coverage = np.asarray(child_coverage_error)
        interface = np.asarray(interface_coordinate_error)
        measure = np.asarray(minimum_measure)
        tolerance_ = float(tolerance)
        if (
            coverage.shape != ()
            or interface.shape != ()
            or measure.shape != ()
            or tolerance_ <= 0.0
            or not np.all(np.isfinite((coverage, interface, measure)))
        ):
            raise ValueError("hp geometry evidence values are invalid.")
        self.child_coverage_error = jnp.asarray(coverage)
        self.interface_coordinate_error = jnp.asarray(interface)
        self.minimum_measure = jnp.asarray(measure)
        self.tolerance = tolerance_
        self.child_coverage_passed = bool(coverage <= tolerance_)
        self.interfaces_watertight = bool(interface <= tolerance_)
        self.positive_measures = bool(measure > 0.0)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-geometry-evidence",
                "coverage": float(coverage),
                "interface": float(interface),
                "measure": float(measure),
                "tolerance": tolerance_,
            }
        )

    @property
    def passed(self) -> bool:
        return (
            self.child_coverage_passed
            and self.interfaces_watertight
            and self.positive_measures
        )


class FiniteElementHPInterfacePlan(StrictModule, NonTrainableState):
    """Canonical conforming, mortar, exterior, and periodic leaf-facet overlay."""

    owner_slots: Array
    neighbour_slots: Array
    owner_local_facets: Array
    neighbour_local_facets: Array
    relation_codes: Array
    child_indices: Array
    child_counts: Array
    owner_orientations: Array
    neighbour_orientations: Array
    valid: Array
    interface_ids: tuple[str, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: FiniteElementHPTopology,
        owner_slots: ArrayLike,
        neighbour_slots: ArrayLike,
        owner_local_facets: ArrayLike,
        neighbour_local_facets: ArrayLike,
        relations: Sequence[Literal["conforming", "mortar", "exterior", "periodic"]],
        /,
        *,
        child_indices: ArrayLike | None = None,
        child_counts: ArrayLike | None = None,
        owner_orientations: ArrayLike | None = None,
        neighbour_orientations: ArrayLike | None = None,
        valid: ArrayLike | None = None,
    ):
        owners = np.asarray(owner_slots, dtype=np.int32)
        neighbours = np.asarray(neighbour_slots, dtype=np.int32)
        owner_facets = np.asarray(owner_local_facets, dtype=np.int32)
        neighbour_facets = np.asarray(neighbour_local_facets, dtype=np.int32)
        relation_names = tuple(str(value) for value in relations)
        valid_ = (
            np.ones(owners.shape, dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        count = owners.size
        children = (
            np.zeros((count,), dtype=np.int32)
            if child_indices is None
            else np.asarray(child_indices, dtype=np.int32)
        )
        child_count = (
            np.ones((count,), dtype=np.int32)
            if child_counts is None
            else np.asarray(child_counts, dtype=np.int32)
        )
        owner_orientation = (
            np.zeros((count,), dtype=np.int8)
            if owner_orientations is None
            else np.asarray(owner_orientations, dtype=np.int8)
        )
        neighbour_orientation = (
            np.zeros((count,), dtype=np.int8)
            if neighbour_orientations is None
            else np.asarray(neighbour_orientations, dtype=np.int8)
        )
        arrays = (
            neighbours,
            owner_facets,
            neighbour_facets,
            children,
            child_count,
            owner_orientation,
            neighbour_orientation,
            valid_,
        )
        if (
            owners.ndim != 1
            or any(value.shape != owners.shape for value in arrays)
            or len(relation_names) != count
            or any(value not in _HP_RELATIONS for value in relation_names)
        ):
            raise ValueError("hp interface routes have incompatible shapes or relations.")
        codes = np.asarray(
            [_HP_RELATIONS[value] for value in relation_names], dtype=np.int8
        )
        active = np.asarray(topology.active)
        exterior = codes == _HP_RELATIONS["exterior"]
        if (
            np.any(owners[valid_] < 0)
            or np.any(owners[valid_] >= topology.capacity)
            or np.any(~active[owners[valid_]])
            or np.any(neighbours[valid_ & ~exterior] < 0)
            or np.any(neighbours[valid_ & ~exterior] >= topology.capacity)
            or np.any(neighbours[valid_ & exterior] != -1)
            or np.any(owner_facets[valid_] < 0)
            or np.any(neighbour_facets[valid_ & ~exterior] < 0)
            or np.any(child_count[valid_] < 1)
            or np.any(children[valid_] < 0)
            or np.any(children[valid_] >= child_count[valid_])
        ):
            raise ValueError(
                "hp interface ownership, facets, or child routes are invalid."
            )
        identifiers = tuple(
            canonical_fingerprint(
                {
                    "kind": "finite-element-hp-interface",
                    "topology": topology.topology_id,
                    "owner_tree": [
                        int(np.asarray(topology.root_cell_ids)[owner]),
                        int(np.asarray(topology.path_codes)[owner]),
                    ],
                    "neighbour_tree": None
                    if neighbour < 0
                    else [
                        int(np.asarray(topology.root_cell_ids)[neighbour]),
                        int(np.asarray(topology.path_codes)[neighbour]),
                    ],
                    "owner_facet": int(owner_facet),
                    "neighbour_facet": int(neighbour_facet),
                    "relation": relation,
                    "child": int(child),
                    "child_count": int(children_),
                }
            )
            if active_
            else ""
            for owner, neighbour, owner_facet, neighbour_facet, relation, child, children_, active_ in zip(
                owners,
                neighbours,
                owner_facets,
                neighbour_facets,
                relation_names,
                children,
                child_count,
                valid_,
                strict=True,
            )
        )
        active_ids = tuple(value for value in identifiers if value)
        if len(set(active_ids)) != len(active_ids):
            raise ValueError("hp interfaces require unique stable identities.")
        self.owner_slots = jnp.asarray(np.where(valid_, owners, -1))
        self.neighbour_slots = jnp.asarray(np.where(valid_, neighbours, -1))
        self.owner_local_facets = jnp.asarray(np.where(valid_, owner_facets, -1))
        self.neighbour_local_facets = jnp.asarray(np.where(valid_, neighbour_facets, -1))
        self.relation_codes = jnp.asarray(codes)
        self.child_indices = jnp.asarray(np.where(valid_, children, 0))
        self.child_counts = jnp.asarray(np.where(valid_, child_count, 1))
        self.owner_orientations = jnp.asarray(owner_orientation)
        self.neighbour_orientations = jnp.asarray(neighbour_orientation)
        self.valid = jnp.asarray(valid_)
        self.interface_ids = identifiers
        self.topology_id = topology.topology_id
        self.capacity = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-interfaces",
                "topology": topology.plan_id,
                "owners": array_tree_fingerprint(owners),
                "neighbours": array_tree_fingerprint(neighbours),
                "owner_facets": array_tree_fingerprint(owner_facets),
                "neighbour_facets": array_tree_fingerprint(neighbour_facets),
                "relations": list(relation_names),
                "children": array_tree_fingerprint(children),
                "child_count": array_tree_fingerprint(child_count),
                "valid": array_tree_fingerprint(valid_),
            }
        )

    def relation_mask(self, relation: str, /) -> Array:
        if relation not in _HP_RELATIONS:
            raise ValueError("Unknown hp interface relation.")
        return self.valid & (self.relation_codes == _HP_RELATIONS[relation])


class FiniteElementHPEpoch(StrictModule, NonTrainableState):
    """Immutable prepared mesh/discretization/interface snapshot for one hp epoch."""

    mesh: CellMesh
    topology: FiniteElementHPTopology
    geometry: FiniteElementHPGeometry
    worksets: FiniteElementHPWorksetPlan
    interfaces: FiniteElementHPInterfacePlan
    active_cell_slots: Array
    discretization: FiniteElementDiscretization | None
    constraints: tuple[tuple[str, object], ...]
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        topology: FiniteElementHPTopology,
        geometry: FiniteElementHPGeometry,
        interfaces: FiniteElementHPInterfacePlan,
        /,
        *,
        worksets: FiniteElementHPWorksetPlan | None = None,
        discretization: FiniteElementDiscretization | None = None,
        constraints: Sequence[tuple[str, object]] = (),
    ):
        if (
            not isinstance(mesh, CellMesh)
            or not isinstance(topology, FiniteElementHPTopology)
            or not isinstance(geometry, FiniteElementHPGeometry)
            or not isinstance(interfaces, FiniteElementHPInterfacePlan)
            or geometry.topology_id != topology.topology_id
            or interfaces.topology_id != topology.topology_id
        ):
            raise ValueError("hp epoch mesh, topology, geometry, or interfaces disagree.")
        selected = (
            finite_element_hp_workset_plan(topology) if worksets is None else worksets
        )
        if selected.topology_id != topology.topology_id:
            raise ValueError("hp epoch worksets belong to a different topology.")
        if discretization is not None and not isinstance(
            discretization, FiniteElementDiscretization
        ):
            raise TypeError("discretization must be FiniteElementDiscretization or None.")
        constraints_ = tuple(sorted((str(name), value) for name, value in constraints))
        if any(not name for name, _ in constraints_):
            raise ValueError("hp epoch constraint names must be non-empty.")
        slot_by_global_id = {
            int(value): slot
            for slot, value in enumerate(np.asarray(topology.cell_global_ids))
            if bool(np.asarray(topology.active)[slot])
        }
        mesh_global_ids = np.concatenate(
            tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
        )
        if set(mesh_global_ids.tolist()) != set(slot_by_global_id):
            raise ValueError("hp epoch active mesh and topology cell IDs disagree.")
        active_slots = np.asarray(
            [slot_by_global_id[int(value)] for value in mesh_global_ids],
            dtype=np.int32,
        )
        self.mesh = mesh
        self.topology = topology
        self.geometry = geometry
        self.worksets = selected
        self.interfaces = interfaces
        self.active_cell_slots = jnp.asarray(active_slots)
        self.discretization = discretization
        self.constraints = constraints_
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-epoch",
                "mesh": mesh.mesh_id,
                "topology": topology.plan_id,
                "geometry": geometry.geometry_id,
                "worksets": selected.plan_id,
                "interfaces": interfaces.plan_id,
                "active_cell_slots": array_tree_fingerprint(active_slots),
                "discretization": None
                if discretization is None
                else discretization.prepared_id,
                "constraints": [name for name, _ in constraints_],
            }
        )


class FiniteElementHPTransaction(StrictModule, NonTrainableState):
    """Rollback-safe promotion between fully prepared hp epochs."""

    accepted: FiniteElementHPEpoch
    candidate: FiniteElementHPEpoch
    lineage: FiniteElementHPLineage
    p_transfers: tuple[FiniteElementHPTransferPlan, ...]
    h_transfers: tuple[FiniteElementHPTransferPlan, ...]
    state_payload: object
    temporal_payload: object
    robustness_payload: object
    observer_payload: object
    conservation_error: Array
    admissible: Array
    geometry_valid: Array
    conservation_tolerance: float = eqx.field(static=True)
    diagnostics: tuple[str, ...] = eqx.field(static=True)
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        accepted: FiniteElementHPEpoch,
        candidate: FiniteElementHPEpoch,
        lineage: FiniteElementHPLineage,
        /,
        *,
        p_transfers: Sequence[FiniteElementHPTransferPlan] = (),
        h_transfers: Sequence[FiniteElementHPTransferPlan] = (),
        diagnostics: Sequence[str] = (),
        state_payload: object = (),
        temporal_payload: object = (),
        robustness_payload: object = (),
        observer_payload: object = (),
        conservation_error: ArrayLike = 0.0,
        admissible: ArrayLike = True,
        geometry_valid: ArrayLike = True,
        conservation_tolerance: float = 1.0e-10,
    ):
        if (
            not isinstance(accepted, FiniteElementHPEpoch)
            or not isinstance(candidate, FiniteElementHPEpoch)
            or not isinstance(lineage, FiniteElementHPLineage)
        ):
            raise TypeError("hp transaction requires prepared epochs and one lineage.")
        source = accepted.topology
        target = candidate.topology
        if (
            source.capacity != target.capacity
            or lineage.source_topology_id != source.topology_id
            or lineage.target_topology_id != target.topology_id
        ):
            raise ValueError("hp transaction epoch and lineage identities disagree.")
        p_transfers_ = tuple(p_transfers)
        h_transfers_ = tuple(h_transfers)
        if any(
            not isinstance(value, FiniteElementHPTransferPlan)
            for value in p_transfers_ + h_transfers_
        ):
            raise TypeError("hp transaction transfers have invalid types.")
        if any(value.transfer_kind != "p" for value in p_transfers_) or any(
            value.transfer_kind not in ("h-refinement", "h-coarsening")
            for value in h_transfers_
        ):
            raise ValueError("hp transaction transfer roles are inconsistent.")
        for transfer in p_transfers_ + h_transfers_:
            if (
                transfer.source_topology_id != source.topology_id
                or transfer.target_topology_id != target.topology_id
                or transfer.source_plan_id != source.plan_id
                or transfer.target_plan_id != target.plan_id
            ):
                raise ValueError("hp transaction transfer identities disagree.")
        diagnostics_ = tuple(str(value) for value in diagnostics)
        conservation = jnp.asarray(conservation_error)
        admissible_ = jnp.asarray(admissible, dtype=bool)
        geometry_valid_ = jnp.asarray(geometry_valid, dtype=bool)
        tolerance = float(conservation_tolerance)
        if (
            conservation.shape != ()
            or admissible_.shape != ()
            or geometry_valid_.shape != ()
            or not np.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("hp transaction acceptance evidence is invalid.")
        self.accepted = accepted
        self.candidate = candidate
        self.lineage = lineage
        self.p_transfers = p_transfers_
        self.h_transfers = h_transfers_
        self.diagnostics = diagnostics_
        self.state_payload = state_payload
        self.temporal_payload = temporal_payload
        self.robustness_payload = robustness_payload
        self.observer_payload = observer_payload
        self.conservation_error = conservation
        self.admissible = admissible_
        self.geometry_valid = geometry_valid_
        self.conservation_tolerance = tolerance
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-transaction",
                "accepted": accepted.epoch_id,
                "candidate": candidate.epoch_id,
                "lineage": lineage.lineage_id,
                "p_transfers": [value.transfer_id for value in p_transfers_],
                "h_transfers": [value.transfer_id for value in h_transfers_],
                "diagnostics": list(diagnostics_),
                "state_payload": array_tree_fingerprint(state_payload),
                "temporal_payload": array_tree_fingerprint(temporal_payload),
                "robustness_payload": array_tree_fingerprint(robustness_payload),
                "observer_payload": array_tree_fingerprint(observer_payload),
                "conservation_tolerance": tolerance,
            }
        )

    def rollback(self, /) -> FiniteElementHPEpoch:
        return self.accepted

    def promote(self, candidate_accepted: bool, /) -> FiniteElementHPEpoch:
        if not isinstance(candidate_accepted, (bool, np.bool_)):
            raise TypeError("hp candidate promotion is one explicit host decision.")
        evidence_accepted = bool(
            np.asarray(self.admissible)
            & np.asarray(self.geometry_valid)
            & (
                abs(float(np.asarray(self.conservation_error)))
                <= self.conservation_tolerance
            )
        )
        return (
            self.candidate
            if bool(candidate_accepted) and evidence_accepted
            else self.accepted
        )


class FiniteElementHPRefinementResult(StrictModule, NonTrainableState):
    topology: FiniteElementHPTopology
    geometry: FiniteElementHPGeometry
    lineage: FiniteElementHPLineage
    requested_slots: Array
    closure_slots: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: FiniteElementHPTopology,
        geometry: FiniteElementHPGeometry,
        lineage: FiniteElementHPLineage,
        requested_slots: ArrayLike,
        closure_slots: ArrayLike,
        /,
    ):
        requested = jnp.asarray(requested_slots, dtype=jnp.int32)
        closure = jnp.asarray(closure_slots, dtype=jnp.int32)
        self.topology = topology
        self.geometry = geometry
        self.lineage = lineage
        self.requested_slots = requested
        self.closure_slots = closure
        self.result_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-refinement-result",
                "topology": topology.plan_id,
                "geometry": geometry.geometry_id,
                "lineage": lineage.lineage_id,
                "requested": array_tree_fingerprint(np.asarray(requested)),
                "closure": array_tree_fingerprint(np.asarray(closure)),
            }
        )


def initial_finite_element_hp_topology(
    mesh: CellMesh,
    degree: int | tuple[int, ...],
    capacity: int,
    /,
) -> tuple[FiniteElementHPTopology, FiniteElementHPGeometry]:
    if not isinstance(mesh, CellMesh) or len(mesh.blocks) == 0:
        raise TypeError("Initial hp topology requires a non-empty CellMesh.")
    kinds = {block.cell_kind for block in mesh.blocks}
    if len(kinds) != 1 or next(iter(kinds)) not in ("quadrilateral", "hexahedron"):
        raise ValueError(
            "Initial hp topology requires only quadrilateral or hexahedron blocks."
        )
    kind = next(iter(kinds))
    dimension = 2 if kind == "quadrilateral" else 3
    degrees = (
        (int(degree),) * dimension
        if isinstance(degree, (int, np.integer))
        else tuple(int(value) for value in degree)
    )
    cell_vertices = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks), axis=0
    )
    global_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks),
        axis=0,
    )
    count = global_ids.size
    capacity_ = int(capacity)
    if len(degrees) != dimension or min(degrees) < 1 or capacity_ < count:
        raise ValueError("Initial hp degree or capacity is invalid.")
    identifiers = np.full((capacity_,), -1, dtype=np.int64)
    identifiers[:count] = global_ids
    allocated = np.zeros((capacity_,), dtype=bool)
    allocated[:count] = True
    active = allocated.copy()
    cell_degrees = np.zeros((capacity_, dimension), dtype=np.int32)
    cell_degrees[:count] = degrees
    topology = FiniteElementHPTopology(
        kind,
        mesh.topology_id,
        identifiers,
        allocated,
        active,
        cell_degrees,
    )
    vertex_count = 4 if dimension == 2 else 8
    geometry_vertices = np.zeros(
        (capacity_, vertex_count, mesh.coordinates.shape[1]), dtype=mesh.coordinates.dtype
    )
    geometry_vertices[:count] = np.asarray(mesh.coordinates)[cell_vertices]
    lower = np.zeros((capacity_, dimension), dtype=float)
    upper = np.zeros((capacity_, dimension), dtype=float)
    upper[:count] = 1.0
    return topology, FiniteElementHPGeometry(topology, geometry_vertices, lower, upper)


def refine_tensor_hp_cells(
    topology: FiniteElementHPTopology,
    geometry: FiniteElementHPGeometry,
    marked_cell_ids: ArrayLike,
    /,
    *,
    coordinate_evaluator: Callable[[int, np.ndarray], np.ndarray] | None = None,
    target_degrees: ArrayLike | None = None,
) -> FiniteElementHPRefinementResult:
    if geometry.topology_id != topology.topology_id:
        raise ValueError("hp refinement topology and geometry identities disagree.")
    marked_ids = np.asarray(marked_cell_ids, dtype=np.int64)
    identifiers = np.asarray(topology.cell_global_ids).copy()
    allocated = np.asarray(topology.allocated).copy()
    active = np.asarray(topology.active).copy()
    degrees = np.asarray(topology.cell_degrees).copy()
    roots = np.asarray(topology.root_cell_ids).copy()
    paths = np.asarray(topology.path_codes).copy()
    levels = np.asarray(topology.levels).copy()
    parents = np.asarray(topology.parent_slots).copy()
    children = np.asarray(topology.child_slots).copy()
    child_valid = np.asarray(topology.child_valid).copy()
    vertices = np.asarray(geometry.cell_vertices).copy()
    lower = np.asarray(geometry.reference_lower).copy()
    upper = np.asarray(geometry.reference_upper).copy()
    active_slots = np.flatnonzero(active)
    slot_by_id = {int(identifiers[slot]): int(slot) for slot in active_slots}
    unknown = set(marked_ids.tolist()) - set(slot_by_id)
    if marked_ids.ndim != 1 or np.unique(marked_ids).size != marked_ids.size or unknown:
        raise ValueError(f"Marked hp cell IDs are invalid: {sorted(unknown)!r}.")
    requested = np.asarray(
        sorted(
            (slot_by_id[int(value)] for value in marked_ids),
            key=lambda slot: (roots[slot], paths[slot]),
        ),
        dtype=np.int32,
    )
    required = requested.size * topology.child_capacity
    free = np.flatnonzero(~allocated)
    if free.size < required:
        raise ValueError("hp refinement exceeds the fixed cell capacity.")
    requested_degrees = (
        None if target_degrees is None else np.asarray(target_degrees, dtype=np.int32)
    )
    if requested_degrees is not None and requested_degrees.shape != (
        requested.size,
        topology.dimension,
    ):
        raise ValueError("target_degrees must contain one tuple per marked cell.")
    next_global = int(np.max(identifiers[allocated], initial=-1)) + 1
    relation_source = []
    relation_target = []
    relation_names = []
    unchanged = [slot for slot in active_slots if slot not in set(requested.tolist())]
    for slot in unchanged:
        relation_source.append(int(slot))
        relation_target.append(int(slot))
        relation_names.append("unchanged")
    free_cursor = 0
    corners = _corner_points(topology.dimension)
    for marked_index, parent in enumerate(requested):
        parent = int(parent)
        active[parent] = False
        parent_degree = degrees[parent].copy()
        degrees[parent] = 0
        for ordinal in range(topology.child_capacity):
            child = int(free[free_cursor])
            free_cursor += 1
            child_lower, child_upper = _child_bounds(
                lower[parent], upper[parent], ordinal
            )
            local_points = child_lower + corners * (child_upper - child_lower)
            if coordinate_evaluator is None:
                parent_points = (local_points - lower[parent]) / (
                    upper[parent] - lower[parent]
                )
                child_vertices = _multilinear_map(vertices[parent], parent_points)
            else:
                child_vertices = np.asarray(coordinate_evaluator(parent, local_points))
            if child_vertices.shape != vertices[child].shape or np.any(
                ~np.isfinite(child_vertices)
            ):
                raise ValueError("coordinate_evaluator returned invalid child vertices.")
            identifiers[child] = next_global
            next_global += 1
            allocated[child] = True
            active[child] = True
            degrees[child] = (
                parent_degree
                if requested_degrees is None
                else requested_degrees[marked_index]
            )
            roots[child] = roots[parent]
            paths[child] = paths[parent] * topology.child_capacity + ordinal + 1
            levels[child] = levels[parent] + 1
            parents[child] = parent
            children[parent, ordinal] = child
            child_valid[parent, ordinal] = True
            vertices[child] = child_vertices
            lower[child] = child_lower
            upper[child] = child_upper
            relation_source.append(parent)
            relation_target.append(child)
            relation_names.append("refinement")
    new_topology = FiniteElementHPTopology(
        topology.cell_kind,
        canonical_fingerprint(
            {
                "kind": "finite-element-hp-refined-topology",
                "source": topology.plan_id,
                "marked": marked_ids.tolist(),
            }
        ),
        identifiers,
        allocated,
        active,
        degrees,
        root_cell_ids=roots,
        path_codes=paths,
        levels=levels,
        parent_slots=parents,
        child_slots=children,
        child_valid=child_valid,
    )
    new_geometry = FiniteElementHPGeometry(new_topology, vertices, lower, upper)
    if finite_element_hp_balance_error(new_topology, new_geometry) > 1:
        raise ValueError(
            "hp refinement violates 2:1 balance; close marked IDs before refinement."
        )
    lineage = FiniteElementHPLineage(
        topology.topology_id,
        new_topology.topology_id,
        topology.capacity,
        new_topology.capacity,
        np.asarray(relation_source, dtype=np.int32),
        np.asarray(relation_target, dtype=np.int32),
        tuple(relation_names),
    )
    return FiniteElementHPRefinementResult(
        new_topology,
        new_geometry,
        lineage,
        requested,
        np.empty((0,), dtype=np.int32),
    )


def coarsen_tensor_hp_cells(
    topology: FiniteElementHPTopology,
    geometry: FiniteElementHPGeometry,
    parent_cell_ids: ArrayLike,
    /,
) -> FiniteElementHPRefinementResult:
    parent_ids = np.asarray(parent_cell_ids, dtype=np.int64)
    identifiers = np.asarray(topology.cell_global_ids).copy()
    allocated = np.asarray(topology.allocated).copy()
    active = np.asarray(topology.active).copy()
    degrees = np.asarray(topology.cell_degrees).copy()
    roots = np.asarray(topology.root_cell_ids).copy()
    paths = np.asarray(topology.path_codes).copy()
    levels = np.asarray(topology.levels).copy()
    parents = np.asarray(topology.parent_slots).copy()
    children = np.asarray(topology.child_slots).copy()
    child_valid = np.asarray(topology.child_valid).copy()
    slot_by_id = {int(identifiers[slot]): int(slot) for slot in np.flatnonzero(allocated)}
    unknown = set(parent_ids.tolist()) - set(slot_by_id)
    if parent_ids.ndim != 1 or np.unique(parent_ids).size != parent_ids.size or unknown:
        raise ValueError(f"Coarsening parent IDs are invalid: {sorted(unknown)!r}.")
    selected = np.asarray(
        sorted(
            (slot_by_id[int(value)] for value in parent_ids),
            key=lambda slot: (roots[slot], paths[slot]),
        ),
        dtype=np.int32,
    )
    relation_source = []
    relation_target = []
    relation_names = []
    selected_children: set[int] = set()
    for parent in selected:
        parent = int(parent)
        local_children = children[parent, child_valid[parent]]
        if (
            local_children.size != topology.child_capacity
            or active[parent]
            or np.any(~active[local_children])
            or np.any(np.any(child_valid[local_children], axis=1))
        ):
            raise ValueError(
                "Coarsening requires one complete active leaf sibling family."
            )
        parent_degree = np.min(degrees[local_children], axis=0)
        active[parent] = True
        degrees[parent] = parent_degree
        for child in local_children:
            child = int(child)
            selected_children.add(child)
            active[child] = False
            degrees[child] = 0
            relation_source.append(child)
            relation_target.append(parent)
            relation_names.append("coarsening")
    for slot in np.flatnonzero(np.asarray(topology.active)):
        if int(slot) not in selected_children:
            relation_source.append(int(slot))
            relation_target.append(int(slot))
            relation_names.append("unchanged")
    new_topology = FiniteElementHPTopology(
        topology.cell_kind,
        canonical_fingerprint(
            {
                "kind": "finite-element-hp-coarsened-topology",
                "source": topology.plan_id,
                "parents": parent_ids.tolist(),
            }
        ),
        identifiers,
        allocated,
        active,
        degrees,
        root_cell_ids=roots,
        path_codes=paths,
        levels=levels,
        parent_slots=parents,
        child_slots=children,
        child_valid=child_valid,
    )
    new_geometry = FiniteElementHPGeometry(
        new_topology,
        geometry.cell_vertices,
        geometry.reference_lower,
        geometry.reference_upper,
    )
    if finite_element_hp_balance_error(new_topology, new_geometry) > 1:
        raise ValueError("hp coarsening would violate the 2:1 face-balance rule.")
    lineage = FiniteElementHPLineage(
        topology.topology_id,
        new_topology.topology_id,
        topology.capacity,
        new_topology.capacity,
        np.asarray(relation_source, dtype=np.int32),
        np.asarray(relation_target, dtype=np.int32),
        tuple(relation_names),
    )
    return FiniteElementHPRefinementResult(
        new_topology,
        new_geometry,
        lineage,
        selected,
        np.empty((0,), dtype=np.int32),
    )


def hp_active_cell_mesh(
    topology: FiniteElementHPTopology,
    geometry: FiniteElementHPGeometry,
    /,
    *,
    numeric_version: str = "hp-active",
) -> tuple[CellMesh, tuple[tuple[int, ...], ...], Array]:
    if geometry.topology_id != topology.topology_id:
        raise ValueError("hp active mesh topology and geometry disagree.")
    active = np.flatnonzero(np.asarray(topology.active))
    degrees = np.asarray(topology.cell_degrees)
    identifiers = np.asarray(topology.cell_global_ids)
    vertices = np.asarray(geometry.cell_vertices)
    point_map: dict[tuple[float, ...], int] = {}
    points: list[np.ndarray] = []
    local_cells: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    local_ids: dict[tuple[int, ...], list[int]] = {}
    local_slots: dict[tuple[int, ...], list[int]] = {}
    ordered_slots = sorted(
        active.tolist(),
        key=lambda value: (
            int(np.asarray(topology.root_cell_ids)[value]),
            int(np.asarray(topology.path_codes)[value]),
        ),
    )
    for slot in ordered_slots:
        cell = []
        for point in vertices[slot]:
            key = tuple(np.round(point, decimals=14).tolist())
            if key not in point_map:
                point_map[key] = len(points)
                points.append(point.copy())
            cell.append(point_map[key])
        degree = tuple(int(value) for value in degrees[slot])
        local_cells.setdefault(degree, []).append(tuple(cell))
        local_ids.setdefault(degree, []).append(int(identifiers[slot]))
        local_slots.setdefault(degree, []).append(slot)
    blocks = tuple(
        CellBlock(
            f"hp-{topology.cell_kind}-{'x'.join(str(value) for value in degree)}",
            topology.cell_kind,
            np.asarray(local_cells[degree], dtype=np.int32),
            global_ids=np.asarray(local_ids[degree], dtype=np.int64),
        )
        for degree in sorted(local_cells)
    )
    mesh = CellMesh(np.asarray(points), blocks, numeric_version=numeric_version)
    slot_routes = np.asarray(
        [slot for degree in sorted(local_slots) for slot in local_slots[degree]],
        dtype=np.int32,
    )
    return mesh, tuple(sorted(local_cells)), jnp.asarray(slot_routes)


def _facet_vertices(
    topology: FiniteElementHPTopology,
    geometry: FiniteElementHPGeometry,
    slot: int,
    local_facet: int,
    /,
) -> np.ndarray:
    indices = _cell_facets(topology.cell_kind)[local_facet]
    return np.asarray(geometry.cell_vertices)[slot, np.asarray(indices)]


def _facet_contains(coarse: np.ndarray, fine: np.ndarray, tolerance: float, /) -> bool:
    if coarse.shape[1] == 2:
        start = coarse[0]
        tangent = coarse[-1] - start
        denominator = float(np.dot(tangent, tangent))
        if denominator <= tolerance**2:
            return False
        parameters = (fine - start) @ tangent / denominator
        residual = fine - (start + parameters[:, None] * tangent)
        return bool(
            np.max(np.linalg.norm(residual, axis=1), initial=0.0) <= tolerance
            and np.min(parameters, initial=0.0) >= -tolerance
            and np.max(parameters, initial=1.0) <= 1.0 + tolerance
        )
    origin = coarse[0]
    first = coarse[1] - origin
    second = coarse[-1] - origin
    basis = np.stack((first, second), axis=1)
    if np.linalg.matrix_rank(basis, tol=tolerance) < 2:
        return False
    parameters, _, _, _ = np.linalg.lstsq(basis, (fine - origin).T, rcond=None)
    reconstructed = origin + (basis @ parameters).T
    return bool(
        np.max(np.linalg.norm(reconstructed - fine, axis=1), initial=0.0) <= tolerance
        and np.min(parameters, initial=0.0) >= -tolerance
        and np.max(parameters, initial=1.0) <= 1.0 + tolerance
    )


def _facet_measure(points: np.ndarray, /) -> float:
    if points.shape[1] == 2:
        return float(np.linalg.norm(points[-1] - points[0]))
    first = points[1] - points[0]
    second = points[-1] - points[0]
    return float(np.linalg.norm(np.cross(first, second)))


def finite_element_hp_balance_error(
    topology: FiniteElementHPTopology,
    geometry: FiniteElementHPGeometry,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> int:
    """Return the largest refinement-level jump across geometrically adjacent faces."""

    active = np.flatnonzero(np.asarray(topology.active))
    levels = np.asarray(topology.levels)
    facets = [
        (
            int(slot),
            _facet_vertices(topology, geometry, int(slot), local_facet),
        )
        for slot in active
        for local_facet in range(len(_cell_facets(topology.cell_kind)))
    ]
    maximum = 0
    for left in range(len(facets)):
        left_slot, left_points = facets[left]
        left_measure = _facet_measure(left_points)
        for right in range(left + 1, len(facets)):
            right_slot, right_points = facets[right]
            if left_slot == right_slot:
                continue
            right_measure = _facet_measure(right_points)
            adjacent = False
            if abs(left_measure - right_measure) <= tolerance * max(
                left_measure, right_measure, 1.0
            ):
                left_key = tuple(
                    sorted(tuple(value) for value in np.round(left_points, decimals=13))
                )
                right_key = tuple(
                    sorted(tuple(value) for value in np.round(right_points, decimals=13))
                )
                adjacent = left_key == right_key
            elif left_measure > right_measure:
                adjacent = _facet_contains(left_points, right_points, tolerance)
            else:
                adjacent = _facet_contains(right_points, left_points, tolerance)
            if adjacent:
                maximum = max(
                    maximum,
                    abs(int(levels[left_slot]) - int(levels[right_slot])),
                )
    return maximum


def finite_element_hp_interface_plan(
    topology: FiniteElementHPTopology,
    geometry: FiniteElementHPGeometry,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> FiniteElementHPInterfacePlan:
    """Build a canonical leaf-facet overlay, including one-to-many mortar patches."""

    if geometry.topology_id != topology.topology_id:
        raise ValueError("hp interface topology and geometry disagree.")
    tolerance_ = float(tolerance)
    active = np.flatnonzero(np.asarray(topology.active))
    roots = np.asarray(topology.root_cell_ids)
    paths = np.asarray(topology.path_codes)
    facets: list[tuple[int, int, np.ndarray, float]] = []
    for slot in active:
        for local_facet in range(len(_cell_facets(topology.cell_kind))):
            points = _facet_vertices(topology, geometry, int(slot), local_facet)
            facets.append((int(slot), local_facet, points, _facet_measure(points)))
    used: set[int] = set()
    rows: list[tuple[int, int, int, int, str, int, int]] = []
    rounded_keys: dict[tuple[tuple[float, ...], ...], list[int]] = {}
    for index, (_, _, points, _) in enumerate(facets):
        key = tuple(sorted(tuple(value) for value in np.round(points, decimals=13)))
        rounded_keys.setdefault(key, []).append(index)
    for indices in rounded_keys.values():
        if len(indices) == 2:
            left, right = indices
            left_slot, left_facet, _, _ = facets[left]
            right_slot, right_facet, _, _ = facets[right]
            left_key = (roots[left_slot], paths[left_slot])
            right_key = (roots[right_slot], paths[right_slot])
            if right_key < left_key:
                left, right = right, left
                left_slot, left_facet, _, _ = facets[left]
                right_slot, right_facet, _, _ = facets[right]
            rows.append(
                (
                    left_slot,
                    right_slot,
                    left_facet,
                    right_facet,
                    "conforming",
                    0,
                    1,
                )
            )
            used.update((left, right))
    for coarse_index, (
        coarse_slot,
        coarse_facet,
        coarse_points,
        coarse_measure,
    ) in sorted(enumerate(facets), key=lambda item: (-item[1][3], item[0])):
        if coarse_index in used:
            continue
        children = []
        for fine_index, (fine_slot, fine_facet, fine_points, fine_measure) in enumerate(
            facets
        ):
            if (
                fine_index == coarse_index
                or fine_index in used
                or coarse_slot == fine_slot
                or fine_measure >= coarse_measure * (1.0 - tolerance_)
            ):
                continue
            if _facet_contains(coarse_points, fine_points, tolerance_):
                children.append((fine_index, fine_slot, fine_facet))
        expected = 2 if topology.dimension == 2 else 4
        if len(children) == expected:
            children.sort(key=lambda item: (roots[item[1]], paths[item[1]], item[2]))
            for child, (fine_index, fine_slot, fine_facet) in enumerate(children):
                rows.append(
                    (
                        coarse_slot,
                        fine_slot,
                        coarse_facet,
                        fine_facet,
                        "mortar",
                        child,
                        expected,
                    )
                )
                used.add(fine_index)
            used.add(coarse_index)
    for index, (slot, local_facet, _, _) in enumerate(facets):
        if index not in used:
            rows.append((slot, -1, local_facet, -1, "exterior", 0, 1))
    rows.sort(
        key=lambda row: (
            int(roots[row[0]]),
            int(paths[row[0]]),
            row[2],
            row[4],
            row[5],
        )
    )
    return FiniteElementHPInterfacePlan(
        topology,
        np.asarray([row[0] for row in rows], dtype=np.int32),
        np.asarray([row[1] for row in rows], dtype=np.int32),
        np.asarray([row[2] for row in rows], dtype=np.int32),
        np.asarray([row[3] for row in rows], dtype=np.int32),
        tuple(row[4] for row in rows),
        child_indices=np.asarray([row[5] for row in rows], dtype=np.int32),
        child_counts=np.asarray([row[6] for row in rows], dtype=np.int32),
    )


def balanced_hp_refinement_ids(
    topology: FiniteElementHPTopology,
    interfaces: FiniteElementHPInterfacePlan,
    marked_cell_ids: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Close a requested refinement set under the deterministic 2:1 face rule."""

    identifiers = np.asarray(topology.cell_global_ids)
    levels = np.asarray(topology.levels)
    active = np.asarray(topology.active)
    slot_by_id = {int(identifiers[slot]): int(slot) for slot in np.flatnonzero(active)}
    marked = np.asarray(marked_cell_ids, dtype=np.int64)
    unknown = set(marked.tolist()) - set(slot_by_id)
    if marked.ndim != 1 or np.unique(marked).size != marked.size or unknown:
        raise ValueError(f"Marked refinement IDs are invalid: {sorted(unknown)!r}.")
    requested = {slot_by_id[int(value)] for value in marked}
    closure = set(requested)
    changed = True
    owners = np.asarray(interfaces.owner_slots)
    neighbours = np.asarray(interfaces.neighbour_slots)
    valid = np.asarray(interfaces.valid)
    while changed:
        changed = False
        for owner, neighbour in zip(owners[valid], neighbours[valid], strict=True):
            if neighbour < 0:
                continue
            owner_target = levels[owner] + (1 if int(owner) in closure else 0)
            neighbour_target = levels[neighbour] + (1 if int(neighbour) in closure else 0)
            if owner_target > neighbour_target + 1 and int(neighbour) not in closure:
                closure.add(int(neighbour))
                changed = True
            if neighbour_target > owner_target + 1 and int(owner) not in closure:
                closure.add(int(owner))
                changed = True
    requested_slots = np.asarray(
        sorted(
            requested,
            key=lambda slot: (
                int(np.asarray(topology.root_cell_ids)[slot]),
                int(np.asarray(topology.path_codes)[slot]),
            ),
        ),
        dtype=np.int32,
    )
    added_slots = np.asarray(
        sorted(
            closure - requested,
            key=lambda slot: (
                int(np.asarray(topology.root_cell_ids)[slot]),
                int(np.asarray(topology.path_codes)[slot]),
            ),
        ),
        dtype=np.int32,
    )
    all_ids = identifiers[
        np.asarray(
            sorted(
                closure,
                key=lambda slot: (
                    int(np.asarray(topology.root_cell_ids)[slot]),
                    int(np.asarray(topology.path_codes)[slot]),
                ),
            ),
            dtype=np.int32,
        )
    ]
    return jnp.asarray(all_ids), jnp.asarray(added_slots)


def certify_finite_element_hp_geometry(
    topology: FiniteElementHPTopology,
    geometry: FiniteElementHPGeometry,
    interfaces: FiniteElementHPInterfacePlan,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> FiniteElementHPGeometryEvidence:
    vertices = np.asarray(geometry.cell_vertices)
    active = np.asarray(topology.active)
    child_valid = np.asarray(topology.child_valid)
    children = np.asarray(topology.child_slots)
    coverage_error = 0.0
    for parent in np.flatnonzero(np.any(child_valid, axis=1)):
        local_children = children[parent, child_valid[parent]]
        if local_children.size != topology.child_capacity:
            continue
        parent_lower = np.min(vertices[parent], axis=0)
        parent_upper = np.max(vertices[parent], axis=0)
        child_lower = np.min(vertices[local_children], axis=(0, 1))
        child_upper = np.max(vertices[local_children], axis=(0, 1))
        coverage_error = max(
            coverage_error,
            float(np.max(np.abs(parent_lower - child_lower), initial=0.0)),
            float(np.max(np.abs(parent_upper - child_upper), initial=0.0)),
        )
    interface_error = 0.0
    for row in np.flatnonzero(np.asarray(interfaces.valid)):
        neighbour = int(np.asarray(interfaces.neighbour_slots)[row])
        if neighbour < 0:
            continue
        owner = int(np.asarray(interfaces.owner_slots)[row])
        owner_points = _facet_vertices(
            topology,
            geometry,
            owner,
            int(np.asarray(interfaces.owner_local_facets)[row]),
        )
        neighbour_points = _facet_vertices(
            topology,
            geometry,
            neighbour,
            int(np.asarray(interfaces.neighbour_local_facets)[row]),
        )
        relation = int(np.asarray(interfaces.relation_codes)[row])
        if relation == _HP_RELATIONS["conforming"]:
            first = np.asarray(sorted(tuple(point) for point in owner_points))
            second = np.asarray(sorted(tuple(point) for point in neighbour_points))
            interface_error = max(
                interface_error,
                float(np.max(np.abs(first - second), initial=0.0)),
            )
        elif relation == _HP_RELATIONS["mortar"] and not _facet_contains(
            owner_points, neighbour_points, tolerance
        ):
            interface_error = np.inf
    measures = []
    for slot in np.flatnonzero(active):
        cell = vertices[slot]
        if topology.dimension == 2:
            x = cell[:, 0]
            y = cell[:, 1]
            measures.append(
                0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            )
        else:
            first = cell[1] - cell[0]
            second = cell[3] - cell[0]
            third = cell[4] - cell[0]
            measures.append(abs(float(np.linalg.det(np.stack((first, second, third))))))
    return FiniteElementHPGeometryEvidence(
        coverage_error,
        interface_error,
        min(measures, default=0.0),
        tolerance,
    )


class FiniteElementHPTraceConstraintPlan(StrictModule, NonTrainableState):
    """One linear master-trace parameterization of broken cell coordinates."""

    prolongation: Array
    row_columns: Array
    row_weights: Array
    row_valid: Array
    full_dof_count: int = eqx.field(static=True)
    reduced_dof_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, prolongation: ArrayLike, /):
        matrix = np.asarray(prolongation)
        if (
            matrix.ndim != 2
            or matrix.shape[0] < matrix.shape[1]
            or not np.issubdtype(matrix.dtype, np.inexact)
            or np.any(~np.isfinite(matrix))
            or np.any(np.count_nonzero(matrix, axis=1) == 0)
        ):
            raise ValueError("hp trace prolongation matrix is invalid.")
        width = int(np.max(np.count_nonzero(matrix, axis=1)))
        columns = np.zeros((matrix.shape[0], width), dtype=np.int32)
        weights = np.zeros((matrix.shape[0], width), dtype=matrix.dtype)
        valid = np.zeros((matrix.shape[0], width), dtype=bool)
        for row in range(matrix.shape[0]):
            local = np.flatnonzero(matrix[row])
            columns[row, : local.size] = local
            weights[row, : local.size] = matrix[row, local]
            valid[row, : local.size] = True
        self.prolongation = jnp.asarray(matrix)
        self.row_columns = jnp.asarray(columns)
        self.row_weights = jnp.asarray(weights)
        self.row_valid = jnp.asarray(valid)
        self.full_dof_count, self.reduced_dof_count = matrix.shape
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-trace-constraint",
                "prolongation": array_tree_fingerprint(matrix),
                "row_columns": array_tree_fingerprint(columns),
                "row_weights": array_tree_fingerprint(weights),
                "row_valid": array_tree_fingerprint(valid),
            }
        )

    def expand(self, reduced: ArrayLike, /) -> Array:
        value = jnp.asarray(reduced)
        if value.shape[0] != self.reduced_dof_count:
            raise ValueError("Reduced trace values have incompatible shape.")
        gathered = value[self.row_columns]
        weights = self.row_weights.reshape(
            self.row_weights.shape + (1,) * (gathered.ndim - 2)
        )
        valid = self.row_valid.reshape(self.row_valid.shape + (1,) * (gathered.ndim - 2))
        return jnp.sum(jnp.where(valid, weights * gathered, 0.0), axis=1)

    def affine_lift(self, reduced_lift: ArrayLike, /) -> Array:
        """Expand nonzero master/Dirichlet data through every hanging trace route."""

        return self.expand(reduced_lift)

    def pullback_raw(self, full_dual: ArrayLike, /) -> Array:
        value = jnp.asarray(full_dual)
        if value.shape[0] != self.full_dof_count:
            raise ValueError("Full trace dual has incompatible shape.")
        result = jnp.zeros(
            (self.reduced_dof_count,) + value.shape[1:],
            dtype=value.dtype,
        )
        for column in range(self.row_columns.shape[1]):
            routes = self.row_columns[:, column]
            weights = self.row_weights[:, column].reshape(
                self.row_weights[:, column].shape + (1,) * (value.ndim - 1)
            )
            contribution = jnp.where(
                self.row_valid[:, column].reshape(
                    self.row_valid[:, column].shape + (1,) * (value.ndim - 1)
                ),
                weights * value,
                0.0,
            )
            result = result.at[routes].add(contribution)
        return result


def finite_element_hp_trace_constraint_plan(
    full_dof_count: int,
    slave_dofs: ArrayLike,
    master_dofs: ArrayLike,
    interpolation: ArrayLike,
    /,
) -> FiniteElementHPTraceConstraintPlan:
    """Build one master-trace prolongation from flattened slave interpolation rows."""

    full_count = int(full_dof_count)
    slaves = np.asarray(slave_dofs, dtype=np.int32)
    masters = np.asarray(master_dofs, dtype=np.int32)
    weights = np.asarray(interpolation)
    if (
        full_count <= 0
        or slaves.ndim != 1
        or masters.ndim != 2
        or weights.shape != masters.shape
        or masters.shape[0] != slaves.size
        or np.unique(slaves).size != slaves.size
        or np.any(slaves < 0)
        or np.any(slaves >= full_count)
        or np.any(masters < 0)
        or np.any(masters >= full_count)
        or np.intersect1d(slaves, masters).size
        or np.any(~np.isfinite(weights))
    ):
        raise ValueError("hp trace slave, master, or interpolation routes are invalid.")
    independent = np.setdiff1d(np.arange(full_count, dtype=np.int32), slaves)
    column_by_dof = np.full((full_count,), -1, dtype=np.int32)
    column_by_dof[independent] = np.arange(independent.size, dtype=np.int32)
    if np.any(column_by_dof[masters] < 0):
        raise ValueError("Every hp trace master must be an independent DOF.")
    matrix = np.zeros((full_count, independent.size), dtype=weights.dtype)
    matrix[independent, np.arange(independent.size)] = 1.0
    for row, slave in enumerate(slaves):
        matrix[slave, column_by_dof[masters[row]]] = weights[row]
    return FiniteElementHPTraceConstraintPlan(matrix)


def tensor_trace_interpolation(
    master_nodes: ArrayLike,
    evaluation_points: ArrayLike,
    /,
) -> Array:
    """Tabulate a tensor Lagrange master trace at arbitrary child/side points."""

    nodes = np.asarray(master_nodes)
    points = np.asarray(evaluation_points)
    if (
        nodes.ndim != 2
        or points.ndim != 2
        or nodes.shape[1] != points.shape[1]
        or nodes.shape[0] == 0
    ):
        raise ValueError("Tensor trace nodes and evaluation points are incompatible.")
    axes = tuple(np.unique(nodes[:, axis]) for axis in range(nodes.shape[1]))
    shape = tuple(values.size for values in axes)
    if int(np.prod(shape, dtype=int)) != nodes.shape[0]:
        raise ValueError("Master trace nodes must form one complete tensor grid.")
    indices = np.stack(
        tuple(
            np.searchsorted(axis_values, nodes[:, axis]).astype(np.int32)
            for axis, axis_values in enumerate(axes)
        ),
        axis=1,
    )
    lexicographic = np.ravel_multi_index(indices.T, shape)
    if np.unique(lexicographic).size != nodes.shape[0]:
        raise ValueError("Master trace tensor nodes are duplicated.")
    values_by_axis = []
    for axis_values, coordinates in zip(axes, points.T, strict=True):
        differences = axis_values[:, None] - axis_values[None, :]
        np.fill_diagonal(differences, 1.0)
        barycentric = 1.0 / np.prod(differences, axis=1)
        delta = coordinates[:, None] - axis_values[None, :]
        exact = np.isclose(delta, 0.0, rtol=0.0, atol=32.0 * np.finfo(float).eps)
        safe = np.where(exact, 1.0, delta)
        raw = barycentric[None, :] / safe
        denominator = np.sum(raw, axis=1, keepdims=True)
        values = raw / np.where(
            np.abs(denominator) > np.finfo(float).tiny, denominator, 1.0
        )
        for row in np.flatnonzero(np.any(exact, axis=1)):
            values[row] = 0.0
            values[row, int(np.argmax(exact[row]))] = 1.0
        values_by_axis.append(values)
    tensor_values = np.ones((points.shape[0],) + shape, dtype=float)
    for axis, values in enumerate(values_by_axis):
        reshape = (
            (points.shape[0],)
            + (1,) * axis
            + (shape[axis],)
            + (1,) * (len(shape) - axis - 1)
        )
        tensor_values *= values.reshape(reshape)
    canonical = tensor_values.reshape((points.shape[0], -1))
    inverse = np.argsort(lexicographic)
    return jnp.asarray(canonical[:, inverse])


def _epoch_slot_elements(
    epoch: FiniteElementHPEpoch,
    field_index: int,
    /,
) -> dict[int, tuple[FiniteElementSpec, int]]:
    if epoch.discretization is None:
        raise ValueError("hp transfer construction requires prepared epochs.")
    result = {}
    offset = 0
    for block, element in zip(
        epoch.discretization.mesh.blocks,
        epoch.discretization.elements[field_index],
        strict=True,
    ):
        for local_cell in range(block.cell_count):
            slot = int(np.asarray(epoch.active_cell_slots)[offset + local_cell])
            result[slot] = (element, element.local_dof_count)
        offset += block.cell_count
    return result


def finite_element_hp_transfer_plan(
    source: FiniteElementHPEpoch,
    target: FiniteElementHPEpoch,
    lineage: FiniteElementHPLineage,
    field_name: str,
    transfer_kind: Literal["p", "h-refinement", "h-coarsening"],
    /,
) -> FiniteElementHPTransferPlan:
    """Build padded tensor interpolation/projection routes for one hp field."""

    if source.discretization is None or target.discretization is None:
        raise ValueError("hp transfer construction requires prepared discretizations.")
    source_index = source.discretization._field_index(field_name)
    target_index = target.discretization._field_index(field_name)
    source_elements = _epoch_slot_elements(source, source_index)
    target_elements = _epoch_slot_elements(target, target_index)
    source_slots_all = np.asarray(lineage.source_slots)
    target_slots_all = np.asarray(lineage.target_slots)
    valid_all = np.asarray(lineage.valid)
    relation_codes = np.asarray(lineage.relation_codes)
    relation_code = {
        "p": _HP_LINEAGE_RELATIONS["unchanged"],
        "h-refinement": _HP_LINEAGE_RELATIONS["refinement"],
        "h-coarsening": _HP_LINEAGE_RELATIONS["coarsening"],
    }[transfer_kind]
    selected = valid_all & (relation_codes == relation_code)
    if not np.any(selected):
        raise ValueError(f"No lineage routes support transfer kind {transfer_kind!r}.")
    source_slots = source_slots_all[selected]
    target_slots = target_slots_all[selected]
    source_count = np.asarray(
        [source_elements[int(slot)][1] for slot in source_slots], dtype=np.int32
    )
    target_count = np.asarray(
        [target_elements[int(slot)][1] for slot in target_slots], dtype=np.int32
    )
    source_width = int(np.max(source_count))
    target_width = int(np.max(target_count))
    matrices = np.zeros((source_slots.size, target_width, source_width), dtype=float)
    projection = np.zeros_like(matrices)
    if transfer_kind != "h-coarsening":
        for route, (source_slot, target_slot) in enumerate(
            zip(source_slots, target_slots, strict=True)
        ):
            source_element = source_elements[int(source_slot)][0]
            target_element = target_elements[int(target_slot)][0]
            target_nodes = np.asarray(target_element.reference_nodes)
            if transfer_kind == "h-refinement":
                source_lower = np.asarray(source.geometry.reference_lower)[source_slot]
                source_upper = np.asarray(source.geometry.reference_upper)[source_slot]
                target_lower = np.asarray(target.geometry.reference_lower)[target_slot]
                target_upper = np.asarray(target.geometry.reference_upper)[target_slot]
                global_points = target_lower + target_nodes * (
                    target_upper - target_lower
                )
                target_nodes = (global_points - source_lower) / (
                    source_upper - source_lower
                )
            local = np.asarray(
                tensor_trace_interpolation(
                    source_element.reference_nodes,
                    target_nodes,
                )
            )
            matrices[route, : target_count[route], : source_count[route]] = local
            projection[route, : target_count[route], : source_count[route]] = local
    else:
        grouped: dict[int, list[int]] = {}
        for route, target_slot in enumerate(target_slots):
            grouped.setdefault(int(target_slot), []).append(route)
        for target_slot, routes in grouped.items():
            parent_element = target_elements[target_slot][0]
            refinement_matrices = []
            for route in routes:
                child_slot = int(source_slots[route])
                child_element = source_elements[child_slot][0]
                child_nodes = np.asarray(child_element.reference_nodes)
                parent_lower = np.asarray(target.geometry.reference_lower)[target_slot]
                parent_upper = np.asarray(target.geometry.reference_upper)[target_slot]
                child_lower = np.asarray(source.geometry.reference_lower)[child_slot]
                child_upper = np.asarray(source.geometry.reference_upper)[child_slot]
                global_points = child_lower + child_nodes * (child_upper - child_lower)
                parent_points = (global_points - parent_lower) / (
                    parent_upper - parent_lower
                )
                refinement_matrices.append(
                    np.asarray(
                        tensor_trace_interpolation(
                            parent_element.reference_nodes,
                            parent_points,
                        )
                    )
                )
            normal = sum(matrix.T @ matrix for matrix in refinement_matrices)
            for route, refinement in zip(routes, refinement_matrices, strict=True):
                local = np.linalg.solve(normal, refinement.T)
                matrices[route, : target_count[route], : source_count[route]] = local
                projection[route, : target_count[route], : source_count[route]] = local
    pairing_adjoint = np.swapaxes(matrices, 1, 2)
    return FiniteElementHPTransferPlan(
        source.topology.topology_id,
        target.topology.topology_id,
        transfer_kind,
        source.topology.capacity,
        target.topology.capacity,
        source_slots,
        target_slots,
        source_count,
        target_count,
        matrices,
        source_plan_id=source.topology.plan_id,
        target_plan_id=target.topology.plan_id,
        pairing_adjoint=pairing_adjoint,
        mass_projection=projection,
    )


class FiniteElementHPStateTransferPolicy(StrictModule, NonTrainableState):
    """Declared transfer/recompute/invalidate policy for one accepted-state entry."""

    name: str = eqx.field(static=True)
    role: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, name: str, role: str, /):
        name_ = str(name)
        role_ = str(role)
        if not name_ or role_ not in (
            "primal",
            "mass-projection",
            "raw-dual",
            "pairing-adjoint",
            "recompute",
            "invalidate",
            "discard",
        ):
            raise ValueError("Unknown hp state-transfer policy.")
        self.name = name_
        self.role = role_
        self.policy_id = canonical_fingerprint(
            {"kind": "finite-element-hp-state-transfer", "name": name_, "role": role_}
        )

    def apply(
        self,
        transfer: FiniteElementHPTransferPlan,
        values: ArrayLike,
        /,
    ) -> Array | None:
        if self.role == "primal":
            return transfer.apply_primal(values)
        if self.role == "mass-projection":
            return transfer.apply_mass_projection(values)
        if self.role == "raw-dual":
            return transfer.pullback_raw(values)
        if self.role == "pairing-adjoint":
            return transfer.apply_pairing_adjoint(values)
        if self.role in ("invalidate", "discard"):
            return None
        raise ValueError("Recompute policies require an explicit recomputation callback.")


class FiniteElementHPResidualJumpLedger(StrictModule):
    """Exactly-once cell residual and interface jump evidence."""

    cell_residual: Array
    cell_measure: Array
    facet_jump: Array
    facet_measure: Array
    estimate: FiniteElementHPErrorEstimate | None
    topology_id: str = eqx.field(static=True)
    interface_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: FiniteElementHPTopology,
        interfaces: FiniteElementHPInterfacePlan,
        cell_residual: ArrayLike,
        cell_measure: ArrayLike,
        facet_jump: ArrayLike,
        facet_measure: ArrayLike,
        /,
    ):
        residual = jnp.asarray(cell_residual)
        cells = jnp.asarray(cell_measure)
        jumps = jnp.asarray(facet_jump)
        facets = jnp.asarray(facet_measure)
        if (
            residual.shape != (topology.capacity,)
            or cells.shape != residual.shape
            or jumps.shape != (interfaces.capacity,)
            or facets.shape != jumps.shape
        ):
            raise ValueError("hp residual/jump ledger arrays have incompatible shapes.")
        valid_cells = topology.active
        valid_facets = interfaces.valid
        contributions = jnp.where(valid_cells, cells * residual**2, 0.0)
        facet_contributions = jnp.where(valid_facets, facets * jumps**2, 0.0)
        owner = jnp.maximum(interfaces.owner_slots, 0)
        neighbour = jnp.maximum(interfaces.neighbour_slots, 0)
        contributions = contributions.at[owner].add(0.5 * facet_contributions)
        has_neighbour = valid_facets & (interfaces.neighbour_slots >= 0)
        contributions = contributions.at[neighbour].add(
            jnp.where(has_neighbour, 0.5 * facet_contributions, 0.0)
        )
        self.cell_residual = residual
        self.cell_measure = cells
        self.facet_jump = jumps
        self.facet_measure = facets
        self.estimate = FiniteElementHPErrorEstimate(
            topology,
            jnp.sqrt(jnp.maximum(contributions, 0.0)),
            estimator_id="hp-residual-jump",
        )
        self.topology_id = topology.topology_id
        self.interface_plan_id = interfaces.plan_id


class FiniteElementHPErrorEstimate(StrictModule):
    cell_tree_ids: Array
    cell_indicators: Array
    smoothness: Array
    valid: Array
    global_estimate: Array
    topology_id: str = eqx.field(static=True)
    estimator_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: FiniteElementHPTopology,
        cell_indicators: ArrayLike,
        /,
        *,
        smoothness: ArrayLike | None = None,
        estimator_id: str = "hp-error-estimate",
    ):
        indicators = jnp.asarray(cell_indicators)
        smooth = (
            jnp.zeros((topology.capacity, topology.dimension), dtype=indicators.dtype)
            if smoothness is None
            else jnp.asarray(smoothness)
        )
        if indicators.shape != (topology.capacity,) or smooth.shape != (
            topology.capacity,
            topology.dimension,
        ):
            raise ValueError("hp error indicators or smoothness have invalid shapes.")
        identifier = str(estimator_id)
        if not identifier:
            raise ValueError("estimator_id must be non-empty.")
        valid = topology.active
        self.cell_tree_ids = topology.stable_tree_ids()
        self.cell_indicators = jnp.where(valid, indicators, 0.0)
        self.smoothness = jnp.where(valid[:, None], smooth, 0.0)
        self.valid = valid
        self.global_estimate = jnp.linalg.norm(self.cell_indicators)
        self.topology_id = topology.topology_id
        self.estimator_id = identifier


def tensor_modal_decay_estimate(
    values: ArrayLike,
    orders: tuple[int, ...],
    /,
    *,
    nodes_by_axis: Sequence[ArrayLike] | None = None,
) -> Array:
    """Return one modal tail-energy ratio per tensor axis."""

    data = np.asarray(values)
    shape = tuple(int(value) + 1 for value in orders)
    if data.shape[: len(shape)] != shape or min(orders) < 1:
        raise ValueError("Tensor modal values and orders have incompatible shapes.")
    axes = (
        tuple(
            np.asarray(legendre_rule_data(order + 1, "lobatto").nodes) for order in orders
        )
        if nodes_by_axis is None
        else tuple(np.asarray(value) for value in nodes_by_axis)
    )
    if len(axes) != len(orders) or any(
        nodes.shape != (order + 1,) for nodes, order in zip(axes, orders, strict=True)
    ):
        raise ValueError("nodes_by_axis must match the tensor orders.")
    coefficients = data
    for axis, order in enumerate(orders):
        nodes = axes[axis]
        vandermonde = np.polynomial.legendre.legvander(nodes, order)
        coefficients = np.moveaxis(coefficients, axis, 0)
        flat = coefficients.reshape((order + 1, -1))
        flat = np.linalg.solve(vandermonde, flat)
        coefficients = np.moveaxis(flat.reshape(coefficients.shape), 0, axis)
    total = np.sum(np.abs(coefficients) ** 2)
    ratios = []
    for axis, order in enumerate(orders):
        tail = np.take(coefficients, indices=order, axis=axis)
        ratios.append(float(np.sum(np.abs(tail) ** 2) / max(total, np.finfo(float).tiny)))
    return jnp.asarray(ratios)


class FiniteElementHPDecision(StrictModule, NonTrainableState):
    target_degrees: Array
    refine: Array
    coarsen: Array
    requested_refine: Array
    balance_added: Array
    coarsen_history: Array
    topology_id: str = eqx.field(static=True)
    decision_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: FiniteElementHPTopology,
        target_degrees: ArrayLike,
        refine: ArrayLike,
        coarsen: ArrayLike,
        /,
        *,
        requested_refine: ArrayLike | None = None,
        balance_added: ArrayLike | None = None,
        coarsen_history: ArrayLike | None = None,
    ):
        degrees = np.asarray(target_degrees, dtype=np.int32)
        refine_ = np.asarray(refine, dtype=bool)
        coarsen_ = np.asarray(coarsen, dtype=bool)
        requested = (
            refine_.copy()
            if requested_refine is None
            else np.asarray(requested_refine, dtype=bool)
        )
        added = (
            refine_ & ~requested
            if balance_added is None
            else np.asarray(balance_added, dtype=bool)
        )
        shape = (topology.capacity,)
        history = (
            np.zeros(shape, dtype=np.int32)
            if coarsen_history is None
            else np.asarray(coarsen_history, dtype=np.int32)
        )
        if (
            degrees.shape != (topology.capacity, topology.dimension)
            or any(
                value.shape != shape
                for value in (refine_, coarsen_, requested, added, history)
            )
            or np.any(history < 0)
            or np.any(refine_ & coarsen_)
            or np.any((refine_ | coarsen_) & ~np.asarray(topology.active))
            or np.any(degrees[np.asarray(topology.active)] < 1)
            or np.any(degrees[~np.asarray(topology.active)] != 0)
        ):
            raise ValueError("hp decision degrees, masks, or active cells are invalid.")
        self.target_degrees = jnp.asarray(degrees)
        self.refine = jnp.asarray(refine_)
        self.coarsen = jnp.asarray(coarsen_)
        self.requested_refine = jnp.asarray(requested)
        self.balance_added = jnp.asarray(added)
        self.coarsen_history = jnp.asarray(history)
        self.topology_id = topology.topology_id
        self.decision_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-decision",
                "topology": topology.plan_id,
                "degrees": array_tree_fingerprint(degrees),
                "refine": array_tree_fingerprint(refine_),
                "coarsen": array_tree_fingerprint(coarsen_),
                "requested": array_tree_fingerprint(requested),
                "balance_added": array_tree_fingerprint(added),
                "coarsen_history": array_tree_fingerprint(history),
            }
        )


def finite_element_hp_decision(
    topology: FiniteElementHPTopology,
    estimate: FiniteElementHPErrorEstimate,
    /,
    *,
    refine_fraction: float = 0.3,
    coarsen_fraction: float = 0.05,
    smoothness_threshold: float = 1.0e-4,
    minimum_degree: int = 1,
    maximum_degree: int = 12,
    maximum_level: int = 20,
    maximum_active_cells: int | None = None,
    maximum_estimated_dofs: int | None = None,
    coarsen_history: ArrayLike | None = None,
    coarsen_epochs: int = 2,
) -> FiniteElementHPDecision:
    if estimate.topology_id != topology.topology_id:
        raise ValueError("hp estimate belongs to a different topology.")
    active = np.asarray(topology.active)
    indicators = np.asarray(estimate.cell_indicators)
    smoothness = np.asarray(estimate.smoothness)
    degrees = np.asarray(topology.cell_degrees).copy()
    refine = np.zeros((topology.capacity,), dtype=bool)
    coarsen = np.zeros_like(refine)
    history = (
        np.zeros((topology.capacity,), dtype=np.int32)
        if coarsen_history is None
        else np.asarray(coarsen_history, dtype=np.int32).copy()
    )
    if history.shape != (topology.capacity,) or int(coarsen_epochs) < 1:
        raise ValueError("hp coarsening history or epoch threshold is invalid.")
    if np.any(active):
        maximum = float(np.max(indicators[active], initial=0.0))
        if maximum > 0.0:
            high = indicators >= float(refine_fraction) * maximum
            low = indicators <= float(coarsen_fraction) * maximum
            history = np.where(active & low, history + 1, 0).astype(np.int32)
            for slot in np.flatnonzero(active & high):
                axis = int(np.argmin(smoothness[slot]))
                if (
                    smoothness[slot, axis] <= smoothness_threshold
                    and degrees[slot, axis] < maximum_degree
                ):
                    degrees[slot, axis] += 1
                elif int(np.asarray(topology.levels)[slot]) < maximum_level:
                    refine[slot] = True
            for slot in np.flatnonzero(active & low & (history >= coarsen_epochs)):
                if np.any(degrees[slot] > minimum_degree):
                    axis = int(np.argmax(degrees[slot]))
                    degrees[slot, axis] -= 1
                elif int(np.asarray(topology.parent_slots)[slot]) >= 0:
                    coarsen[slot] = True
        else:
            history = np.where(active, history, 0).astype(np.int32)
    stable_ids = np.asarray(topology.stable_tree_ids())
    refinement_order = sorted(
        np.flatnonzero(refine).tolist(),
        key=lambda slot: (
            -float(indicators[slot]),
            int(stable_ids[slot, 0]),
            int(stable_ids[slot, 1]),
        ),
    )
    if maximum_active_cells is not None:
        budget = int(maximum_active_cells)
        if budget < np.count_nonzero(active):
            raise ValueError("maximum_active_cells is below the accepted active count.")
        allowed = (budget - np.count_nonzero(active)) // (topology.child_capacity - 1)
        for slot in refinement_order[allowed:]:
            refine[slot] = False
    if maximum_estimated_dofs is not None:
        budget = int(maximum_estimated_dofs)

        def estimated_dofs() -> int:
            local = np.prod(degrees[active] + 1, axis=1)
            extra = sum(
                (topology.child_capacity - 1) * int(np.prod(degrees[slot] + 1, dtype=int))
                for slot in np.flatnonzero(refine)
            )
            return int(np.sum(local)) + extra

        for slot in reversed(refinement_order):
            if estimated_dofs() <= budget:
                break
            refine[slot] = False
        p_changed = [
            slot
            for slot in np.flatnonzero(active)
            if np.any(degrees[slot] > np.asarray(topology.cell_degrees)[slot])
        ]
        for slot in sorted(
            p_changed,
            key=lambda value: (
                float(indicators[value]),
                int(stable_ids[value, 0]),
                int(stable_ids[value, 1]),
            ),
        ):
            if estimated_dofs() <= budget:
                break
            degrees[slot] = np.asarray(topology.cell_degrees)[slot]
        if estimated_dofs() > budget:
            raise ValueError("maximum_estimated_dofs cannot fit the accepted topology.")
    return FiniteElementHPDecision(
        topology,
        degrees,
        refine,
        coarsen,
        coarsen_history=history,
    )


def close_finite_element_hp_decision(
    topology: FiniteElementHPTopology,
    interfaces: FiniteElementHPInterfacePlan,
    decision: FiniteElementHPDecision,
    /,
) -> FiniteElementHPDecision:
    """Add deterministic 2:1 closure cells to one requested hp decision."""

    if decision.topology_id != topology.topology_id:
        raise ValueError("hp decision belongs to a different topology.")
    identifiers = np.asarray(topology.cell_global_ids)
    requested_slots = np.flatnonzero(np.asarray(decision.refine))
    requested_ids = identifiers[requested_slots]
    closed_ids, added_slots = balanced_hp_refinement_ids(
        topology,
        interfaces,
        requested_ids,
    )
    slot_by_id = {
        int(identifiers[slot]): int(slot)
        for slot in np.flatnonzero(np.asarray(topology.active))
    }
    refine = np.zeros((topology.capacity,), dtype=bool)
    refine[[slot_by_id[int(value)] for value in np.asarray(closed_ids)]] = True
    added = np.zeros_like(refine)
    added[np.asarray(added_slots, dtype=np.int32)] = True
    requested = np.zeros_like(refine)
    requested[requested_slots] = True
    return FiniteElementHPDecision(
        topology,
        decision.target_degrees,
        refine,
        decision.coarsen,
        requested_refine=requested,
        balance_added=added,
        coarsen_history=decision.coarsen_history,
    )


def _tensor_facet_axis_side(
    cell_kind: str,
    local_facet: int,
    /,
) -> tuple[int, int, tuple[int, ...]]:
    if cell_kind == "quadrilateral":
        values = (
            (1, 0, (0,)),
            (0, 1, (1,)),
            (1, 1, (0,)),
            (0, 0, (1,)),
        )
    elif cell_kind == "hexahedron":
        values = (
            (2, 0, (0, 1)),
            (0, 1, (1, 2)),
            (2, 1, (0, 1)),
            (0, 0, (1, 2)),
            (1, 0, (0, 2)),
            (1, 1, (0, 2)),
        )
    else:
        raise ValueError("Tensor trace constraints require quad/hex cells.")
    index = int(local_facet)
    if index < 0 or index >= len(values):
        raise ValueError("local_facet is out of range.")
    return values[index]


def _hp_trace_constraint_for_field(
    topology: FiniteElementHPTopology,
    interfaces: FiniteElementHPInterfacePlan,
    discretization: FiniteElementDiscretization,
    active_cell_slots: np.ndarray,
    field_index: int,
    /,
) -> FiniteElementHPTraceConstraintPlan:
    slot_data = {}
    offset = 0
    for block_index, (block, cell_dofs, element) in enumerate(
        zip(
            discretization.mesh.blocks,
            discretization.dof_maps[field_index].cell_dofs,
            discretization.elements[field_index],
            strict=True,
        )
    ):
        for local_cell in range(block.cell_count):
            slot = int(active_cell_slots[offset + local_cell])
            slot_data[slot] = (
                block_index,
                np.asarray(cell_dofs[local_cell], dtype=np.int32),
                element,
            )
        offset += block.cell_count
    slave_rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    valid = np.asarray(interfaces.valid)
    relation_codes = np.asarray(interfaces.relation_codes)
    for row in np.flatnonzero(valid):
        neighbour_slot = int(np.asarray(interfaces.neighbour_slots)[row])
        if neighbour_slot < 0:
            continue
        owner_slot = int(np.asarray(interfaces.owner_slots)[row])
        owner_facet = int(np.asarray(interfaces.owner_local_facets)[row])
        neighbour_facet = int(np.asarray(interfaces.neighbour_local_facets)[row])
        _, owner_dofs, owner_element = slot_data[owner_slot]
        _, neighbour_dofs, neighbour_element = slot_data[neighbour_slot]
        owner_axis, owner_side, owner_tangent = _tensor_facet_axis_side(
            topology.cell_kind, owner_facet
        )
        neighbour_axis, neighbour_side, neighbour_tangent = _tensor_facet_axis_side(
            topology.cell_kind, neighbour_facet
        )
        owner_nodes = np.asarray(owner_element.reference_nodes)
        neighbour_nodes = np.asarray(neighbour_element.reference_nodes)
        owner_trace = np.flatnonzero(
            np.isclose(owner_nodes[:, owner_axis], float(owner_side))
        )
        neighbour_trace = np.flatnonzero(
            np.isclose(neighbour_nodes[:, neighbour_axis], float(neighbour_side))
        )
        if (
            owner_trace.size > neighbour_trace.size
            and relation_codes[row] == _HP_RELATIONS["conforming"]
        ):
            (
                owner_slot,
                neighbour_slot,
                owner_dofs,
                neighbour_dofs,
                owner_element,
                neighbour_element,
                owner_nodes,
                neighbour_nodes,
                owner_trace,
                neighbour_trace,
                owner_tangent,
                neighbour_tangent,
            ) = (
                neighbour_slot,
                owner_slot,
                neighbour_dofs,
                owner_dofs,
                neighbour_element,
                owner_element,
                neighbour_nodes,
                owner_nodes,
                neighbour_trace,
                owner_trace,
                neighbour_tangent,
                owner_tangent,
            )
        master_global = owner_dofs[owner_trace]
        slave_global = neighbour_dofs[neighbour_trace]
        master_nodes = owner_nodes[owner_trace][:, owner_tangent]
        evaluation = neighbour_nodes[neighbour_trace][:, neighbour_tangent]
        if relation_codes[row] == _HP_RELATIONS["mortar"]:
            child = int(np.asarray(interfaces.child_indices)[row])
            child_count = int(np.asarray(interfaces.child_counts)[row])
            if topology.dimension == 2:
                evaluation = (evaluation + child) / child_count
            else:
                width = int(round(child_count**0.5))
                child_coordinates = np.asarray(
                    (child % width, child // width), dtype=float
                )
                evaluation = (evaluation + child_coordinates) / width
        interpolation = np.asarray(tensor_trace_interpolation(master_nodes, evaluation))
        for local_row, slave in enumerate(slave_global):
            slave_ = int(slave)
            if slave_ in set(master_global.tolist()):
                continue
            values = interpolation[local_row]
            if slave_ in slave_rows:
                continue
            slave_rows[slave_] = (master_global.copy(), values.copy())
    full_count = discretization.dof_maps[field_index].global_dof_count
    if not slave_rows:
        return FiniteElementHPTraceConstraintPlan(np.eye(full_count))
    max_width = max(masters.size for masters, _ in slave_rows.values())
    slaves = np.asarray(sorted(slave_rows), dtype=np.int32)
    masters = np.empty((slaves.size, max_width), dtype=np.int32)
    weights = np.zeros((slaves.size, max_width), dtype=float)
    for row, slave in enumerate(slaves):
        local_masters, local_weights = slave_rows[int(slave)]
        masters[row] = local_masters[0]
        masters[row, : local_masters.size] = local_masters
        weights[row, : local_weights.size] = local_weights
    return finite_element_hp_trace_constraint_plan(
        full_count,
        slaves,
        masters,
        weights,
    )


def prepare_finite_element_hp_epoch(
    topology: FiniteElementHPTopology,
    geometry: FiniteElementHPGeometry,
    field_name: str,
    /,
    *,
    conformity: Literal["H1", "L2"] = "H1",
    component_shape: Sequence[int] = (),
) -> FiniteElementHPEpoch:
    """Prepare degree-bucket elements and one compiler-ready hp epoch."""

    if conformity not in ("H1", "L2"):
        raise ValueError("Adaptive tensor hp fields currently require H1 or L2.")
    mesh, degree_tuples, _ = hp_active_cell_mesh(topology, geometry)
    elements = {}
    for block, degree in zip(mesh.blocks, degree_tuples, strict=True):
        element = ReferenceNodalFamily(topology.cell_kind, degree).finite_element()
        if conformity == "L2":
            entities: list[tuple[tuple[int, ...], ...]] = [
                tuple(() for _ in dimension_entities)
                for dimension_entities in element.entity_dofs
            ]
            entities[-1] = (tuple(range(element.local_dof_count)),)
            element = FiniteElementSpec(
                "DiscontinuousTensorProductLagrange",
                element.cell_kind,
                element.degree,
                element.reference_nodes,
                tuple(entities),
                conformity="L2",
                representation=element.representation,
                tabulator=element.tabulate,
                tabulator_id=f"discontinuous:{element.element_id}",
            )
        elements[block.name] = element
    field = FiniteElementFieldSpec(
        field_name,
        elements,
        component_shape=component_shape,
    )
    discretization = FiniteElementPlan(mesh, field).prepare()
    interfaces = finite_element_hp_interface_plan(topology, geometry)
    constraints = ()
    if conformity == "H1":
        slot_by_global_id = {
            int(value): slot
            for slot, value in enumerate(np.asarray(topology.cell_global_ids))
            if bool(np.asarray(topology.active)[slot])
        }
        mesh_global_ids = np.concatenate(
            tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
        )
        active_slots = np.asarray(
            [slot_by_global_id[int(value)] for value in mesh_global_ids],
            dtype=np.int32,
        )
        trace_plan = _hp_trace_constraint_for_field(
            topology,
            interfaces,
            discretization,
            active_slots,
            0,
        )
        constraints = ((field_name, trace_plan),)
    return FiniteElementHPEpoch(
        mesh,
        topology,
        geometry,
        interfaces,
        discretization=discretization,
        constraints=constraints,
    )


def prepare_multi_field_finite_element_hp_epoch(
    topology: FiniteElementHPTopology,
    geometry: FiniteElementHPGeometry,
    fields: Mapping[
        str,
        tuple[Literal["H1", "L2"], Sequence[int], Sequence[int]],
    ],
    /,
) -> FiniteElementHPEpoch:
    """Prepare one hp epoch with independent fieldwise p offsets and components."""

    if not fields:
        raise ValueError("Multi-field hp epochs require at least one field.")
    mesh, degree_tuples, _ = hp_active_cell_mesh(topology, geometry)
    field_specs = []
    for field_name, (conformity, component_shape, degree_offset) in fields.items():
        offsets = tuple(int(value) for value in degree_offset)
        if conformity not in ("H1", "L2") or len(offsets) != topology.dimension:
            raise ValueError("Multi-field hp conformity or degree offsets are invalid.")
        elements = {}
        for block, degree in zip(mesh.blocks, degree_tuples, strict=True):
            field_degree = tuple(
                max(1, value + offset)
                for value, offset in zip(degree, offsets, strict=True)
            )
            element = ReferenceNodalFamily(
                topology.cell_kind, field_degree
            ).finite_element()
            if conformity == "L2":
                entities: list[tuple[tuple[int, ...], ...]] = [
                    tuple(() for _ in dimension_entities)
                    for dimension_entities in element.entity_dofs
                ]
                entities[-1] = (tuple(range(element.local_dof_count)),)
                element = FiniteElementSpec(
                    "DiscontinuousTensorProductLagrange",
                    element.cell_kind,
                    element.degree,
                    element.reference_nodes,
                    tuple(entities),
                    conformity="L2",
                    representation=element.representation,
                    tabulator=element.tabulate,
                    tabulator_id=f"discontinuous:{element.element_id}",
                )
            elements[block.name] = element
        field_specs.append(
            FiniteElementFieldSpec(
                field_name,
                elements,
                component_shape=component_shape,
            )
        )
    discretization = FiniteElementPlan(mesh, tuple(field_specs)).prepare()
    interfaces = finite_element_hp_interface_plan(topology, geometry)
    slot_by_global_id = {
        int(value): slot
        for slot, value in enumerate(np.asarray(topology.cell_global_ids))
        if bool(np.asarray(topology.active)[slot])
    }
    mesh_global_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    active_slots = np.asarray(
        [slot_by_global_id[int(value)] for value in mesh_global_ids],
        dtype=np.int32,
    )
    constraints = []
    for field_index, (field_name, (conformity, _, _)) in enumerate(fields.items()):
        if conformity == "H1":
            constraints.append(
                (
                    field_name,
                    _hp_trace_constraint_for_field(
                        topology,
                        interfaces,
                        discretization,
                        active_slots,
                        field_index,
                    ),
                )
            )
    return FiniteElementHPEpoch(
        mesh,
        topology,
        geometry,
        interfaces,
        discretization=discretization,
        constraints=constraints,
    )


def finite_element_hp_domains(
    epoch: FiniteElementHPEpoch,
    /,
) -> tuple[IntegrationDomain, IntegrationDomain]:
    """Build compiler-facing interior and exterior domains from the hp overlay."""

    if epoch.discretization is None:
        raise ValueError("hp domains require one prepared finite-element discretization.")
    slot_to_cell = np.full((epoch.topology.capacity,), -1, dtype=np.int32)
    active_slots = np.asarray(epoch.active_cell_slots, dtype=np.int32)
    slot_to_cell[active_slots] = np.arange(active_slots.size, dtype=np.int32)
    valid = np.asarray(epoch.interfaces.valid)
    relations = np.asarray(epoch.interfaces.relation_codes)
    owners = np.asarray(epoch.interfaces.owner_slots)
    neighbours = np.asarray(epoch.interfaces.neighbour_slots)
    owner_facets = np.asarray(epoch.interfaces.owner_local_facets)
    neighbour_facets = np.asarray(epoch.interfaces.neighbour_local_facets)
    exterior_mask = valid & (relations == _HP_RELATIONS["exterior"])
    interior_mask = valid & ~exterior_mask
    interior_rows = np.flatnonzero(interior_mask)
    exterior_rows = np.flatnonzero(exterior_mask)
    support_id = epoch.discretization.support.support_id
    interior_entity_set = canonical_fingerprint(
        {
            "kind": "finite-element-hp-interior-entities",
            "epoch": epoch.epoch_id,
            "rows": interior_rows.tolist(),
        }
    )
    exterior_entity_set = canonical_fingerprint(
        {
            "kind": "finite-element-hp-exterior-entities",
            "epoch": epoch.epoch_id,
            "rows": exterior_rows.tolist(),
        }
    )
    interior = IntegrationDomain(
        "interior_facet",
        interior_rows,
        support_id,
        interior_entity_set,
        owner_cells=slot_to_cell[owners[interior_rows]],
        neighbour_cells=slot_to_cell[neighbours[interior_rows]],
        owner_local_entities=owner_facets[interior_rows],
        neighbour_local_entities=neighbour_facets[interior_rows],
        periodic_face_mask=relations[interior_rows] == _HP_RELATIONS["periodic"],
    )
    exterior = IntegrationDomain(
        "exterior_facet",
        exterior_rows,
        support_id,
        exterior_entity_set,
        owner_cells=slot_to_cell[owners[exterior_rows]],
        neighbour_cells=np.full((exterior_rows.size,), -1, dtype=np.int32),
        owner_local_entities=owner_facets[exterior_rows],
        neighbour_local_entities=np.full((exterior_rows.size,), -1, dtype=np.int32),
    )
    return interior, exterior


__all__ = [
    "FiniteElementHPEpoch",
    "FiniteElementHPDecision",
    "FiniteElementHPErrorEstimate",
    "FiniteElementHPResidualJumpLedger",
    "FiniteElementHPGeometry",
    "FiniteElementHPGeometryEvidence",
    "FiniteElementHPInterfacePlan",
    "FiniteElementHPTraceConstraintPlan",
    "FiniteElementHPRefinementResult",
    "FiniteElementHPStateTransferPolicy",
    "FiniteElementHPTransaction",
    "balanced_hp_refinement_ids",
    "certify_finite_element_hp_geometry",
    "coarsen_tensor_hp_cells",
    "close_finite_element_hp_decision",
    "finite_element_hp_decision",
    "finite_element_hp_domains",
    "finite_element_hp_trace_constraint_plan",
    "finite_element_hp_balance_error",
    "finite_element_hp_transfer_plan",
    "finite_element_hp_interface_plan",
    "hp_active_cell_mesh",
    "initial_finite_element_hp_topology",
    "prepare_finite_element_hp_epoch",
    "prepare_multi_field_finite_element_hp_epoch",
    "refine_tensor_hp_cells",
    "tensor_modal_decay_estimate",
    "tensor_trace_interpolation",
]
