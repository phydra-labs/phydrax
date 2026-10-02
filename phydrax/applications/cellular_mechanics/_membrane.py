#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared, differentiable mechanics for closed triangulated biomembranes.

Static-topology energy, force, transport, and thermal paths are fixed-shape JAX
programs.  Topology changes are deliberately host-side proposal/evaluation/commit
transactions; a committed transaction always produces a newly prepared identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import ArrayLike as NumPyArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import unique_identifiers
from ...discretization._topology_epoch import TopologyEpochTransition
from ...discretization.lattice_boltzmann import (
    ImmersedBoundaryForcingPlan,
    ImmersedBoundaryForcingResult,
)
from ...geometry.multiregion_surface import (
    apply_surface_events,
    EdgeCollapseProposal,
    EdgeFlipProposal,
    EdgeSplitProposal,
    ExtensiveTransferEvidence,
    MultiRegionSurfaceCapacityPlan,
    MultiRegionSurfaceLineage,
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    MultiRegionSurfaceValidationPolicy,
    PreparedMultiRegionSurface,
    SurfaceEventKind,
    SurfaceEventPassEvidence,
    SurfaceEventPassResult,
    SurfaceEventPolicy,
)
from ...sparse import EdgeRelation, route_reduce
from ...typing import checked


class BiomembraneState(StrictModule):
    """Fixed-topology coordinates and species amounts bound to one preparation."""

    positions: Array
    species_mass: Array
    prepared_id: str = eqx.field(static=True)


class BiomembraneEnergy(StrictModule):
    """Scalar decomposition whose sum is the conservative potential."""

    helfrich: Array
    gaussian: Array
    local_area: Array
    global_area: Array
    volume_constraint: Array
    tension: Array
    pressure: Array
    adhesion: Array
    total: Array


class BiomembraneGeometryEvidence(StrictModule):
    """Compiled geometry, constraint, and invariance evidence."""

    face_area: Array
    local_area_residual: Array
    vertex_area: Array
    total_area: Array
    enclosed_volume: Array
    area_residual: Array
    volume_residual: Array
    minimum_face_area: Array
    minimum_vertex_normal: Array
    conservative_force_residual: Array
    conservative_torque_residual: Array
    finite: Array
    nondegenerate: Array
    normal_defined: Array
    positively_oriented: Array
    valid: Array


class BiomembraneEvaluation(StrictModule):
    """Energy-derived conservative force plus explicit active traction."""

    energy: BiomembraneEnergy
    conservative_force: Array
    active_force: Array
    force: Array
    mean_curvature: Array
    gaussian_curvature: Array
    vertex_normal: Array
    species_concentration: Array
    geometry: BiomembraneGeometryEvidence
    finite: Array
    valid: Array
    prepared_id: str = eqx.field(static=True)


class BiomembraneTransportEvidence(StrictModule):
    """Surface finite-volume conservation and positivity evidence."""

    species_mass_before: Array
    species_mass_after: Array
    species_mass_residual: Array
    total_mass_residual: Array
    minimum_species_mass: Array
    finite: Array
    nonnegative: Array
    conservative: Array
    successful: Array


class BiomembraneTransportResult(StrictModule):
    candidate_state: BiomembraneState
    accepted_state: BiomembraneState
    mass_rate: Array
    edge_flux: Array
    reaction_rate: Array
    evidence: BiomembraneTransportEvidence
    prepared_id: str = eqx.field(static=True)


class BiomembraneThermalEvidence(StrictModule):
    """Fluctuation-dissipation and fail-closed step evidence."""

    deterministic_displacement: Array
    stochastic_displacement: Array
    expected_coordinate_variance: Array
    observed_whitened_norm: Array
    finite: Array
    geometry_valid: Array
    successful: Array
    rng_identity: str = eqx.field(static=True)


class BiomembraneThermalStepResult(StrictModule):
    candidate_state: BiomembraneState
    accepted_state: BiomembraneState
    initial_evaluation: BiomembraneEvaluation
    evaluation: BiomembraneEvaluation
    evidence: BiomembraneThermalEvidence
    prepared_id: str = eqx.field(static=True)


class BiomembraneFluidCouplingResult(StrictModule):
    """Explicit membrane/immersed-fluid action-reaction composition."""

    membrane_force: Array
    mechanical_force: Array
    total_force: Array
    forcing: ImmersedBoundaryForcingResult
    force_balance_residual: Array
    work: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class BiomembraneRemeshProposal(StrictModule, NonTrainableState):
    """Biomembrane candidate produced by one shared surface-event transaction."""

    source: PreparedBiomembrane
    source_state: BiomembraneState
    candidate: PreparedBiomembrane
    candidate_state: BiomembraneState
    surface_result: SurfaceEventPassResult
    vertex_transfer_evidence: ExtensiveTransferEvidence | None
    face_transfer_evidence: ExtensiveTransferEvidence | None
    stencil_valid: bool = eqx.field(static=True)
    event_kind: SurfaceEventKind = eqx.field(static=True)
    edge_vertex_ids: tuple[int, int] = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    @property
    def lineage(self) -> MultiRegionSurfaceLineage | None:
        return self.surface_result.evidence.lineage


class BiomembraneRemeshEvidence(StrictModule):
    """Mechanics jumps and shared transaction evidence for a remesh candidate."""

    area_jump: Array
    relative_area_jump: Array
    volume_jump: Array
    relative_volume_jump: Array
    energy_jump: Array
    relative_energy_jump: Array
    species_mass_jump: Array
    material_integral_jump: Array
    finite: Array
    stencil_valid: Array
    conservative_transfer: Array
    within_jump_limits: Array
    accepted: Array
    surface: SurfaceEventPassEvidence
    derivative_available: bool = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)


class BiomembraneRemeshResult(StrictModule, NonTrainableState):
    """Committed membrane epoch, or the bitwise-identical source objects."""

    prepared: PreparedBiomembrane
    state: BiomembraneState
    proposal: BiomembraneRemeshProposal
    evidence: BiomembraneRemeshEvidence
    lineage: MultiRegionSurfaceLineage | None
    vertex_transition: TopologyEpochTransition | None
    face_transition: TopologyEpochTransition | None
    committed: bool = eqx.field(static=True)


def _real_array(value: ArrayLike, shape: tuple[int, ...], name: str, /) -> np.ndarray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must be real numerical data.")
    values = np.asarray(raw, dtype=np.float64)
    if values.shape == ():
        values = np.full(shape, float(values), dtype=np.float64)
    if values.shape != shape:
        raise ValueError(f"{name} must be scalar or have shape {shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")
    return values


def _nonnegative_array(
    value: ArrayLike, shape: tuple[int, ...], name: str, /
) -> np.ndarray:
    values = _real_array(value, shape, name)
    if np.any(values < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return values


def _stable_ids(value: ArrayLike | None, count: int, name: str, /) -> np.ndarray:
    if value is None:
        identifiers = np.arange(count, dtype=np.int64)
    else:
        raw = np.asarray(value)
        if raw.shape != (count,) or not np.issubdtype(raw.dtype, np.integer):
            raise TypeError(f"{name} must contain one integer identifier per entity.")
        identifiers = np.asarray(raw, dtype=np.int64)
    if np.any(identifiers < 0):
        raise ValueError(f"{name} must be nonnegative.")
    if np.unique(identifiers).shape[0] != count:
        raise ValueError(f"{name} must be unique.")
    return identifiers


def _vertex_links_valid(faces: np.ndarray, vertex_count: int, /) -> bool:
    for vertex in range(vertex_count):
        incident = faces[np.any(faces == vertex, axis=1)]
        link_edges: list[tuple[int, int]] = []
        for face in incident.tolist():
            opposite = [int(item) for item in face if int(item) != vertex]
            if len(opposite) != 2:
                return False
            link_edges.append((opposite[0], opposite[1]))
        link_vertices = {item for edge in link_edges for item in edge}
        adjacency = {item: [] for item in link_vertices}
        for first, second in link_edges:
            adjacency[first].append(second)
            adjacency[second].append(first)
        if not adjacency or any(len(items) != 2 for items in adjacency.values()):
            return False
        reached = {next(iter(adjacency))}
        frontier = list(reached)
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        if reached != link_vertices:
            return False
    return True


def _closed_topology(
    faces: np.ndarray, vertex_count: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(np.min(faces)) < 0 or int(np.max(faces)) >= vertex_count:
        raise ValueError("face vertex indices are out of range.")
    if np.any(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    ):
        raise ValueError("A membrane face cannot repeat a vertex.")
    canonical_faces = np.sort(faces, axis=1)
    if np.unique(canonical_faces, axis=0).shape[0] != faces.shape[0]:
        raise ValueError("Membrane faces must be unique regardless of orientation.")
    used = np.unique(faces.reshape((-1,)))
    if used.shape[0] != vertex_count or not np.array_equal(used, np.arange(vertex_count)):
        raise ValueError(
            "Every membrane vertex must be used and indices must be contiguous."
        )
    if not _vertex_links_valid(faces, vertex_count):
        raise ValueError("Every membrane vertex link must be one closed cycle.")

    uses: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for face_index, face in enumerate(faces.tolist()):
        a, b, c = (int(item) for item in face)
        for start, end, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            key = (min(start, end), max(start, end))
            uses.setdefault(key, []).append((face_index, start, end, opposite))
    if any(len(items) != 2 for items in uses.values()):
        raise ValueError("A biomembrane must be a closed two-manifold at every edge.")

    edges: list[tuple[int, int]] = []
    opposites: list[tuple[int, int]] = []
    adjacent: list[tuple[int, int]] = []
    for key in sorted(uses):
        left, right = uses[key]
        if left[1] != right[2] or left[2] != right[1]:
            raise ValueError("Adjacent faces must use opposite edge orientations.")
        edges.append((left[1], left[2]))
        opposites.append((left[3], right[3]))
        adjacent.append((left[0], right[0]))

    neighbors: list[list[int]] = [[] for _ in range(faces.shape[0])]
    for left, right in adjacent:
        neighbors[left].append(right)
        neighbors[right].append(left)
    reached = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in neighbors[current]:
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    if len(reached) != faces.shape[0]:
        raise ValueError("A biomembrane must have one connected closed component.")
    return (
        np.asarray(edges, dtype=np.int32),
        np.asarray(opposites, dtype=np.int32),
        np.asarray(adjacent, dtype=np.int32),
    )


def _host_geometry(
    positions: np.ndarray, faces: np.ndarray, tolerance: float, /
) -> tuple[np.ndarray, np.ndarray, float, float]:
    points = positions[faces]
    area_vectors = 0.5 * np.cross(
        points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    )
    areas = np.linalg.norm(area_vectors, axis=1)
    if np.any(~np.isfinite(areas)) or np.any(areas <= tolerance):
        raise ValueError("Membrane geometry contains a degenerate face.")
    vertex_area = np.zeros((positions.shape[0],), dtype=np.float64)
    np.add.at(vertex_area, faces.reshape((-1,)), np.repeat(areas / 3.0, 3))
    normal_sum = np.zeros_like(positions)
    np.add.at(
        normal_sum,
        faces.reshape((-1,)),
        np.repeat(area_vectors, 3, axis=0),
    )
    normal_magnitude = np.linalg.norm(normal_sum, axis=1)
    if np.any(~np.isfinite(normal_magnitude)) or np.any(normal_magnitude <= tolerance):
        raise ValueError("Every membrane vertex must have a defined oriented normal.")
    center = np.mean(positions, axis=0)
    relative = points - center
    volume = float(
        np.sum(
            np.sum(
                relative[:, 0] * np.cross(relative[:, 1], relative[:, 2]),
                axis=1,
            )
        )
        / 6.0
    )
    total_area = float(np.sum(areas))
    volume_tolerance = tolerance * max(total_area, 1.0)
    if not isfinite(volume) or volume <= volume_tolerance:
        raise ValueError("Membrane faces must enclose positive outward-oriented volume.")
    return areas, vertex_area, total_area, volume


def _host_cotangent_sums(
    positions: np.ndarray,
    edges: np.ndarray,
    opposites: np.ndarray,
    tolerance: float,
    /,
) -> np.ndarray:
    first = positions[edges[:, 0]]
    second = positions[edges[:, 1]]

    def cotangent(opposite: np.ndarray) -> np.ndarray:
        left = first - positions[opposite]
        right = second - positions[opposite]
        cross_length = np.linalg.norm(np.cross(left, right), axis=1)
        if np.any(cross_length <= tolerance):
            raise ValueError("Membrane cotangent stencil is singular.")
        return np.sum(left * right, axis=1) / cross_length

    weights = cotangent(opposites[:, 0]) + cotangent(opposites[:, 1])
    if np.any(~np.isfinite(weights)):
        raise ValueError("Membrane cotangent stencil must be finite.")
    return weights


def _default_remesh_capacity(
    vertex_count: int, edge_count: int, face_count: int, /
) -> MultiRegionSurfaceCapacityPlan:
    vertex_capacity = max(vertex_count + 1, 2 * vertex_count)
    growth = vertex_capacity - vertex_count
    return MultiRegionSurfaceCapacityPlan(
        vertex_capacity=vertex_capacity,
        edge_capacity=edge_count + 3 * growth,
        face_capacity=face_count + 2 * growth,
        region_capacity=2,
        region_pair_capacity=1,
        maximum_edge_valence=2,
        maximum_vertex_region_pairs=1,
        event_capacity=1,
        resource_id="biomembrane-remesh",
    )


class BiomembranePlan(StrictModule, NonTrainableState):
    """Closed oriented topology and constitutive data for a biomembrane.

    Constitutive arrays follow the active mechanics topology exactly; the
    shared remesh owner separately keeps fixed padded transaction capacities.
    Species are nodal amounts with explicit IDs, and concentration uses the
    current barycentric dual area.
    """

    faces: Array
    vertex_ids: Array
    face_ids: Array
    edge_vertices: Array
    edge_opposites: Array
    edge_faces: Array
    bending_rigidity: Array
    gaussian_rigidity: Array
    spontaneous_curvature: Array
    curvature_coupling: Array
    local_area_modulus: Array
    active_traction: Array
    mobility: Array
    species_diffusivity: Array
    reaction_matrix: Array
    remesh_capacity: MultiRegionSurfaceCapacityPlan
    remesh_policy: SurfaceEventPolicy
    adhesion_normal: Array
    global_area_modulus: Array
    volume_modulus: Array
    tension: Array
    pressure: Array
    adhesion_strength: Array
    adhesion_offset: Array
    adhesion_length: Array
    vertex_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    species_count: int = eqx.field(static=True)
    species_ids: tuple[str, ...] = eqx.field(static=True)
    remesh_region_ids: tuple[str, str] = eqx.field(static=True)
    target_area: float | None = eqx.field(static=True)
    target_volume: float | None = eqx.field(static=True)
    geometry_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        faces: ArrayLike,
        /,
        *,
        vertex_ids: ArrayLike | None = None,
        face_ids: ArrayLike | None = None,
        bending_rigidity: ArrayLike = 1.0,
        gaussian_rigidity: ArrayLike = 0.0,
        spontaneous_curvature: ArrayLike = 0.0,
        curvature_coupling: ArrayLike | None = None,
        local_area_modulus: ArrayLike = 0.0,
        global_area_modulus: float = 0.0,
        volume_modulus: float = 0.0,
        target_area: float | None = None,
        target_volume: float | None = None,
        tension: float = 0.0,
        pressure: float = 0.0,
        adhesion_strength: float = 0.0,
        adhesion_normal: NumPyArrayLike = (0.0, 0.0, 1.0),
        adhesion_offset: float = 0.0,
        adhesion_length: float = 1.0,
        active_traction: ArrayLike = 0.0,
        mobility: ArrayLike = 1.0,
        species_diffusivity: NumPyArrayLike = (),
        reaction_matrix: ArrayLike | None = None,
        species_ids: Sequence[str] | None = None,
        remesh_capacity: MultiRegionSurfaceCapacityPlan | None = None,
        remesh_policy: SurfaceEventPolicy | None = None,
        remesh_region_ids: Sequence[str] = (
            "biomembrane-interior",
            "biomembrane-ambient",
        ),
        geometry_tolerance: float = 1.0e-12,
        plan_id: str | None = None,
    ) -> None:
        raw_faces = np.asarray(faces)
        if raw_faces.ndim != 2 or raw_faces.shape[1:] != (3,) or raw_faces.shape[0] < 4:
            raise ValueError(
                "faces must have shape (face_count, 3) with at least four faces."
            )
        if not np.issubdtype(raw_faces.dtype, np.integer):
            raise TypeError("faces must contain integer vertex indices.")
        topology = np.asarray(raw_faces, dtype=np.int32)
        if np.any(topology < 0):
            raise ValueError("face vertex indices must be nonnegative.")
        vertex_count = int(np.max(topology)) + 1
        face_count = topology.shape[0]
        edges, opposites, adjacent = _closed_topology(topology, vertex_count)
        identifiers = _stable_ids(vertex_ids, vertex_count, "vertex_ids")
        face_identifiers = _stable_ids(face_ids, face_count, "face_ids")

        kappa = _nonnegative_array(bending_rigidity, (vertex_count,), "bending_rigidity")
        gaussian = _real_array(gaussian_rigidity, (vertex_count,), "gaussian_rigidity")
        spontaneous = _real_array(
            spontaneous_curvature, (vertex_count,), "spontaneous_curvature"
        )
        local_modulus = _nonnegative_array(
            local_area_modulus, (face_count,), "local_area_modulus"
        )
        traction = _real_array(active_traction, (vertex_count,), "active_traction")
        mobility_values = _nonnegative_array(mobility, (vertex_count,), "mobility")

        raw_diffusivity = np.asarray(species_diffusivity)
        if not np.issubdtype(raw_diffusivity.dtype, np.number) or np.issubdtype(
            raw_diffusivity.dtype, np.complexfloating
        ):
            raise TypeError("species_diffusivity must be real numerical data.")
        diffusivity = np.asarray(raw_diffusivity, dtype=np.float64)
        if diffusivity.ndim != 1:
            raise ValueError("species_diffusivity must be one-dimensional.")
        if np.any(~np.isfinite(diffusivity)) or np.any(diffusivity < 0.0):
            raise ValueError("species_diffusivity must be finite and nonnegative.")
        species_count = diffusivity.shape[0]
        species_names = (
            tuple(f"species-{index}" for index in range(species_count))
            if species_ids is None
            else unique_identifiers(species_ids, "species_ids", allow_empty=True)
        )
        if len(species_names) != species_count:
            raise ValueError("species_ids must identify every surface species.")
        if reaction_matrix is None:
            reaction = np.zeros((species_count, species_count), dtype=np.float64)
        else:
            raw_reaction = np.asarray(reaction_matrix)
            if (
                raw_reaction.shape != (species_count, species_count)
                or not np.issubdtype(raw_reaction.dtype, np.number)
                or np.issubdtype(raw_reaction.dtype, np.complexfloating)
            ):
                raise TypeError("reaction_matrix must be a real square species matrix.")
            reaction = np.asarray(raw_reaction, dtype=np.float64)
            if np.any(~np.isfinite(reaction)):
                raise ValueError("reaction_matrix must be finite.")
            if reaction.size:
                scale = max(float(np.max(np.abs(reaction))), 1.0)
                if np.max(np.abs(np.sum(reaction, axis=0))) > 1.0e-12 * scale:
                    raise ValueError(
                        "reaction_matrix columns must sum to zero for mass conservation."
                    )
        if curvature_coupling is None:
            coupling = np.zeros((species_count,), dtype=np.float64)
        else:
            coupling = _real_array(
                curvature_coupling, (species_count,), "curvature_coupling"
            )
        capacity = (
            _default_remesh_capacity(vertex_count, edges.shape[0], face_count)
            if remesh_capacity is None
            else remesh_capacity
        )
        if not isinstance(capacity, MultiRegionSurfaceCapacityPlan):
            raise TypeError("remesh_capacity must be MultiRegionSurfaceCapacityPlan.")
        if (
            capacity.vertex_capacity < vertex_count
            or capacity.edge_capacity < edges.shape[0]
            or capacity.face_capacity < face_count
            or capacity.region_capacity != 2
            or capacity.region_pair_capacity != 1
            or capacity.maximum_edge_valence != 2
            or capacity.maximum_vertex_region_pairs != 1
            or capacity.event_capacity < 1
            or capacity.coordinate_dtype != "float64"
        ):
            raise ValueError(
                "remesh_capacity must fit the membrane and declare exactly two "
                "regions, one pair/slot, edge valence two, float64 coordinates, "
                "and at least one event."
            )
        region_names = unique_identifiers(
            remesh_region_ids, "remesh_region_ids", allow_empty=False
        )
        if len(region_names) != 2:
            raise ValueError(
                "remesh_region_ids must identify the finite interior and ambient."
            )
        policy = (
            SurfaceEventPolicy(
                validation=MultiRegionSurfaceValidationPolicy(
                    profile="manifold_two_region"
                )
            )
            if remesh_policy is None
            else remesh_policy
        )
        if not isinstance(policy, SurfaceEventPolicy):
            raise TypeError("remesh_policy must be SurfaceEventPolicy.")
        if policy.validation.profile != "manifold_two_region":
            raise ValueError(
                "Biomembrane remeshing requires the manifold_two_region profile."
            )

        scalars = {
            "global_area_modulus": float(global_area_modulus),
            "volume_modulus": float(volume_modulus),
            "tension": float(tension),
            "pressure": float(pressure),
            "adhesion_strength": float(adhesion_strength),
            "adhesion_offset": float(adhesion_offset),
            "adhesion_length": float(adhesion_length),
            "geometry_tolerance": float(geometry_tolerance),
        }
        if any(not isfinite(value) for value in scalars.values()):
            raise ValueError("Biomembrane scalar parameters must be finite.")
        if scalars["global_area_modulus"] < 0.0 or scalars["volume_modulus"] < 0.0:
            raise ValueError("Global area and volume moduli must be nonnegative.")
        if scalars["adhesion_strength"] < 0.0 or scalars["adhesion_length"] <= 0.0:
            raise ValueError("Adhesion strength must be nonnegative and length positive.")
        if scalars["geometry_tolerance"] <= 0.0:
            raise ValueError("geometry_tolerance must be positive.")
        targets = (
            None if target_area is None else float(target_area),
            None if target_volume is None else float(target_volume),
        )
        if any(
            value is not None and (not isfinite(value) or value <= 0.0)
            for value in targets
        ):
            raise ValueError(
                "Explicit area and volume targets must be finite and positive."
            )
        normal_raw = np.asarray(adhesion_normal)
        if (
            normal_raw.shape != (3,)
            or not np.issubdtype(normal_raw.dtype, np.number)
            or np.issubdtype(normal_raw.dtype, np.complexfloating)
        ):
            raise TypeError("adhesion_normal must be a real three-vector.")
        normal = np.array(normal_raw, dtype=np.float64, copy=True)
        normal_length = float(np.linalg.norm(normal))
        if not np.all(np.isfinite(normal)) or normal_length <= 0.0:
            raise ValueError("adhesion_normal must be finite and nonzero.")
        normal /= normal_length

        payload = {
            "kind": "biomembrane-plan",
            "faces": array_tree_fingerprint(topology),
            "vertex_ids": array_tree_fingerprint(identifiers),
            "face_ids": array_tree_fingerprint(face_identifiers),
            "bending": array_tree_fingerprint(kappa),
            "gaussian": array_tree_fingerprint(gaussian),
            "spontaneous": array_tree_fingerprint(spontaneous),
            "coupling": array_tree_fingerprint(coupling),
            "local_area": array_tree_fingerprint(local_modulus),
            "scalars": scalars,
            "targets": targets,
            "adhesion_normal": normal.tolist(),
            "active_traction": array_tree_fingerprint(traction),
            "mobility": array_tree_fingerprint(mobility_values),
            "diffusivity": array_tree_fingerprint(diffusivity),
            "reaction": array_tree_fingerprint(reaction),
            "species_ids": list(species_names),
            "remesh_capacity": capacity.plan_id,
            "remesh_policy": policy.policy_id,
            "remesh_region_ids": list(region_names),
        }
        generated = canonical_fingerprint(payload)
        resolved = generated if plan_id is None else str(plan_id)
        if not resolved:
            raise ValueError("plan_id must be nonempty.")

        dtype = jnp.asarray(kappa).dtype
        self.faces = jnp.asarray(topology)
        self.vertex_ids = jnp.asarray(identifiers)
        self.face_ids = jnp.asarray(face_identifiers)
        self.edge_vertices = jnp.asarray(edges)
        self.edge_opposites = jnp.asarray(opposites)
        self.edge_faces = jnp.asarray(adjacent)
        self.bending_rigidity = jnp.asarray(kappa, dtype=dtype)
        self.gaussian_rigidity = jnp.asarray(gaussian, dtype=dtype)
        self.spontaneous_curvature = jnp.asarray(spontaneous, dtype=dtype)
        self.curvature_coupling = jnp.asarray(coupling, dtype=dtype)
        self.local_area_modulus = jnp.asarray(local_modulus, dtype=dtype)
        self.active_traction = jnp.asarray(traction, dtype=dtype)
        self.mobility = jnp.asarray(mobility_values, dtype=dtype)
        self.species_diffusivity = jnp.asarray(diffusivity, dtype=dtype)
        self.remesh_capacity = capacity
        self.remesh_policy = policy
        reaction_array = jnp.asarray(reaction, dtype=dtype)
        if species_count:
            reaction_array = reaction_array.at[-1, :].add(
                -jnp.sum(reaction_array, axis=0)
            )
        self.reaction_matrix = reaction_array
        self.adhesion_normal = jnp.asarray(normal, dtype=dtype)
        self.global_area_modulus = jnp.asarray(
            scalars["global_area_modulus"], dtype=dtype
        )
        self.volume_modulus = jnp.asarray(scalars["volume_modulus"], dtype=dtype)
        self.tension = jnp.asarray(scalars["tension"], dtype=dtype)
        self.pressure = jnp.asarray(scalars["pressure"], dtype=dtype)
        self.adhesion_strength = jnp.asarray(scalars["adhesion_strength"], dtype=dtype)
        self.adhesion_offset = jnp.asarray(scalars["adhesion_offset"], dtype=dtype)
        self.adhesion_length = jnp.asarray(scalars["adhesion_length"], dtype=dtype)
        self.vertex_count = vertex_count
        self.face_count = face_count
        self.edge_count = edges.shape[0]
        self.species_count = species_count
        self.species_ids = species_names
        self.remesh_region_ids = (region_names[0], region_names[1])
        self.target_area, self.target_volume = targets
        self.geometry_tolerance = scalars["geometry_tolerance"]
        self.plan_id = resolved

    def prepare(self, reference_positions: ArrayLike, /) -> PreparedBiomembrane:
        return PreparedBiomembrane(self, reference_positions)


class PreparedBiomembrane(StrictModule, NonTrainableState):
    """Prepared static-topology biomembrane runtime."""

    plan: BiomembranePlan
    remesh_topology: MultiRegionSurfaceTopology
    prepared_remesh: PreparedMultiRegionSurface
    reference_positions: Array
    reference_face_area: Array
    reference_vertex_area: Array
    reference_area: Array
    reference_volume: Array
    target_area: Array
    target_volume: Array
    rng_tag: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: BiomembranePlan,
        reference_positions: ArrayLike,
        /,
        *,
        reference_face_area: ArrayLike | None = None,
        remesh_topology: MultiRegionSurfaceTopology | None = None,
    ) -> None:
        raw = np.asarray(reference_positions)
        if raw.shape != (plan.vertex_count, 3):
            raise ValueError("reference_positions must have shape (vertex_count, 3).")
        if not np.issubdtype(raw.dtype, np.floating):
            raise TypeError(
                "reference_positions must be real floating-point coordinates."
            )
        reference = np.asarray(raw, dtype=np.float64)
        if np.any(~np.isfinite(reference)):
            raise ValueError("reference_positions must be finite.")
        faces = np.asarray(plan.faces)
        if remesh_topology is None:
            labels = np.tile(np.asarray((0, 1), dtype=np.int32), (plan.face_count, 1))
            topology = MultiRegionSurfaceTopology(
                plan.remesh_capacity,
                faces,
                labels,
                plan.remesh_region_ids,
                ("finite", "boundary"),
                vertex_count=plan.vertex_count,
                vertex_global_ids=np.asarray(plan.vertex_ids),
                face_global_ids=np.asarray(plan.face_ids),
            )
        else:
            if not isinstance(remesh_topology, MultiRegionSurfaceTopology):
                raise TypeError("remesh_topology must be MultiRegionSurfaceTopology.")
            topology = remesh_topology
            compatible = (
                topology.plan.plan_id == plan.remesh_capacity.plan_id
                and topology.vertex_count == plan.vertex_count
                and topology.face_count == plan.face_count
                and topology.region_ids == plan.remesh_region_ids
                and topology.region_kinds == ("finite", "boundary")
                and np.array_equal(topology.host_faces(), faces)
                and np.array_equal(
                    np.asarray(topology.vertex_global_ids[: topology.vertex_count]),
                    np.asarray(plan.vertex_ids),
                )
                and np.array_equal(
                    np.asarray(topology.face_global_ids[: topology.face_count]),
                    np.asarray(plan.face_ids),
                )
            )
            if not compatible:
                raise ValueError("remesh_topology does not match the biomembrane plan.")
        padded_positions = np.zeros((topology.vertex_capacity, 3), dtype=np.float64)
        padded_positions[: plan.vertex_count] = reference
        remesh_state = MultiRegionSurfaceState(topology, padded_positions)
        prepared_remesh = PreparedMultiRegionSurface(
            topology, remesh_state, policy=plan.remesh_policy.validation
        )
        areas, vertex_area, total_area, volume = _host_geometry(
            reference, faces, plan.geometry_tolerance
        )
        weights = _host_cotangent_sums(
            reference,
            np.asarray(plan.edge_vertices),
            np.asarray(plan.edge_opposites),
            plan.geometry_tolerance,
        )
        if np.any(weights < -np.sqrt(plan.geometry_tolerance)):
            raise ValueError(
                "Prepared membrane transport requires intrinsic Delaunay edges."
            )
        if reference_face_area is None:
            rest_areas = areas
        else:
            raw_rest = np.asarray(reference_face_area)
            if raw_rest.shape != (plan.face_count,) or not np.issubdtype(
                raw_rest.dtype, np.number
            ):
                raise TypeError("reference_face_area must be one real value per face.")
            rest_areas = np.asarray(raw_rest, dtype=np.float64)
            if np.any(~np.isfinite(rest_areas)) or np.any(rest_areas <= 0.0):
                raise ValueError("reference_face_area must be finite and positive.")
        rest_vertex_area = np.zeros((plan.vertex_count,), dtype=np.float64)
        np.add.at(
            rest_vertex_area,
            faces.reshape((-1,)),
            np.repeat(rest_areas / 3.0, 3),
        )
        target_area = total_area if plan.target_area is None else plan.target_area
        target_volume = volume if plan.target_volume is None else plan.target_volume
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-biomembrane",
                "plan": plan.plan_id,
                "reference_positions": array_tree_fingerprint(reference),
                "reference_face_area": array_tree_fingerprint(rest_areas),
                "reference_area": total_area,
                "reference_volume": volume,
                "target_area": target_area,
                "target_volume": target_volume,
                "remesh_lineage": topology.lineage_id,
            }
        )
        dtype = plan.bending_rigidity.dtype
        self.plan = plan
        self.remesh_topology = topology
        self.prepared_remesh = prepared_remesh
        self.reference_positions = jnp.asarray(reference, dtype=dtype)
        self.reference_face_area = jnp.asarray(rest_areas, dtype=dtype)
        self.reference_vertex_area = jnp.asarray(rest_vertex_area, dtype=dtype)
        self.reference_area = jnp.asarray(total_area, dtype=dtype)
        self.reference_volume = jnp.asarray(volume, dtype=dtype)
        self.target_area = jnp.asarray(target_area, dtype=dtype)
        self.target_volume = jnp.asarray(target_volume, dtype=dtype)
        self.rng_tag = int(prepared_id[:8], 16) & 0xFFFFFFFF
        self.prepared_id = prepared_id

    def state(
        self, positions: ArrayLike | None = None, /, species_mass: ArrayLike | None = None
    ) -> BiomembraneState:
        coordinates = (
            self.reference_positions
            if positions is None
            else jnp.asarray(positions, dtype=self.reference_positions.dtype)
        )
        if coordinates.shape != (self.plan.vertex_count, 3):
            raise ValueError("positions must match the prepared vertex shape.")
        if species_mass is None:
            mass = jnp.zeros(
                (self.plan.vertex_count, self.plan.species_count),
                dtype=coordinates.dtype,
            )
        else:
            mass = jnp.asarray(species_mass, dtype=coordinates.dtype)
            if mass.shape != (
                self.plan.vertex_count,
                self.plan.species_count,
            ):
                raise ValueError(
                    "species_mass must have shape (vertex_count, species_count)."
                )
        return BiomembraneState(coordinates, mass, self.prepared_id)

    @checked
    def _validate_state(self, state: BiomembraneState, /) -> BiomembraneState:
        if state.prepared_id != self.prepared_id:
            raise ValueError("state belongs to a different membrane preparation.")
        if state.positions.shape != (self.plan.vertex_count, 3):
            raise ValueError("state positions do not match the prepared topology.")
        if state.species_mass.shape != (
            self.plan.vertex_count,
            self.plan.species_count,
        ):
            raise ValueError("state species mass does not match the prepared topology.")
        return state

    def _surface_geometry(
        self, positions: Array, /
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        points = positions[self.plan.faces]
        area_vector = 0.5 * jnp.cross(
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
        )
        tolerance = jnp.asarray(self.plan.geometry_tolerance, dtype=positions.dtype)
        face_area = jnp.sqrt(
            jnp.maximum(
                jnp.sum(area_vector * area_vector, axis=1),
                tolerance**2,
            )
        )
        face_count = self.plan.faces.shape[0]
        face_relation = EdgeRelation(
            jnp.repeat(jnp.arange(face_count, dtype=jnp.int32), 3),
            self.plan.faces.reshape((-1,)),
            source_size=face_count,
            target_size=self.plan.vertex_count,
        )
        vertex_area = route_reduce(
            face_relation,
            jnp.repeat(face_area / 3.0, 3),
        )
        normal_sum = route_reduce(
            face_relation,
            jnp.repeat(area_vector, 3, axis=0),
        )
        normal_magnitude = jnp.sqrt(jnp.sum(normal_sum * normal_sum, axis=1))
        safe_normal_magnitude = jnp.sqrt(jnp.maximum(normal_magnitude**2, tolerance**2))
        vertex_normal = normal_sum / safe_normal_magnitude[:, None]
        center = jnp.mean(positions, axis=0)
        relative = points - center
        volume = (
            jnp.sum(
                contract(
                    "fi,fi->f",
                    relative[:, 0],
                    jnp.cross(relative[:, 1], relative[:, 2]),
                )
            )
            / 6.0
        )
        return (
            points,
            face_area,
            vertex_area,
            vertex_normal,
            normal_magnitude,
            volume,
        )

    def _curvature(
        self,
        positions: Array,
        vertex_area: Array,
        vertex_normal: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        edge = self.plan.edge_vertices
        opposite = self.plan.edge_opposites
        first = positions[edge[:, 0]]
        second = positions[edge[:, 1]]
        first_opposite = positions[opposite[:, 0]]
        second_opposite = positions[opposite[:, 1]]
        tolerance = jnp.asarray(self.plan.geometry_tolerance, dtype=positions.dtype)

        def cotangent(opposite_point: Array) -> Array:
            left = first - opposite_point
            right = second - opposite_point
            denominator = jnp.sqrt(
                jnp.maximum(jnp.sum(jnp.cross(left, right) ** 2, axis=1), tolerance**2)
            )
            return contract("ei,ei->e", left, right) / denominator

        cotangent_sum = cotangent(first_opposite) + cotangent(second_opposite)
        weighted = cotangent_sum[:, None] * (first - second)
        edge_count = edge.shape[0]
        edge_relation = EdgeRelation(
            jnp.repeat(jnp.arange(edge_count, dtype=jnp.int32), 2),
            edge.reshape((-1,)),
            source_size=edge_count,
            target_size=self.plan.vertex_count,
        )
        laplace = route_reduce(
            edge_relation,
            jnp.stack((weighted, -weighted), axis=1).reshape((-1, 3)),
        )
        two_mean_curvature = contract("vi,vi->v", laplace, vertex_normal) / (
            2.0 * vertex_area
        )

        points = positions[self.plan.faces]
        edge_ab = points[:, 1] - points[:, 0]
        edge_ac = points[:, 2] - points[:, 0]
        edge_ba = -edge_ab
        edge_bc = points[:, 2] - points[:, 1]
        edge_ca = -edge_ac
        edge_cb = -edge_bc

        def angle(left: Array, right: Array) -> Array:
            return jnp.arctan2(
                jnp.sqrt(jnp.maximum(jnp.sum(jnp.cross(left, right) ** 2, axis=1), 0.0)),
                contract("fi,fi->f", left, right),
            )

        face_angles = jnp.stack(
            (angle(edge_ab, edge_ac), angle(edge_ba, edge_bc), angle(edge_ca, edge_cb)),
            axis=1,
        )
        face_count = self.plan.faces.shape[0]
        face_relation = EdgeRelation(
            jnp.repeat(jnp.arange(face_count, dtype=jnp.int32), 3),
            self.plan.faces.reshape((-1,)),
            source_size=face_count,
            target_size=self.plan.vertex_count,
        )
        angle_sum = route_reduce(
            face_relation,
            face_angles.reshape((-1,)),
        )
        angle_defect = 2.0 * jnp.pi - angle_sum
        gaussian_curvature = angle_defect / vertex_area
        return two_mean_curvature, gaussian_curvature, cotangent_sum

    def _energy_components(
        self, positions: Array, species_mass: Array, /
    ) -> tuple[Array, BiomembraneEnergy]:
        (
            _,
            face_area,
            vertex_area,
            vertex_normal,
            _,
            volume,
        ) = self._surface_geometry(positions)
        mean_curvature, gaussian_curvature, _ = self._curvature(
            positions, vertex_area, vertex_normal
        )
        concentration = species_mass / vertex_area[:, None]
        spontaneous = self.plan.spontaneous_curvature
        if self.plan.species_count:
            spontaneous = spontaneous + contract(
                "vs,s->v", concentration, self.plan.curvature_coupling
            )
        helfrich = 0.5 * jnp.sum(
            vertex_area * self.plan.bending_rigidity * (mean_curvature - spontaneous) ** 2
        )
        gaussian = jnp.sum(vertex_area * self.plan.gaussian_rigidity * gaussian_curvature)
        local_area = 0.5 * jnp.sum(
            self.plan.local_area_modulus
            * (face_area - self.reference_face_area) ** 2
            / self.reference_face_area
        )
        total_area = jnp.sum(face_area)
        global_area = (
            0.5
            * self.plan.global_area_modulus
            * (total_area - self.target_area) ** 2
            / self.target_area
        )
        volume_constraint = (
            0.5
            * self.plan.volume_modulus
            * (volume - self.target_volume) ** 2
            / self.target_volume
        )
        tension = self.plan.tension * total_area
        pressure = -self.plan.pressure * volume
        distance = (
            contract("vi,i->v", positions, self.plan.adhesion_normal)
            - self.plan.adhesion_offset
        )
        adhesion = -self.plan.adhesion_strength * jnp.sum(
            vertex_area * jnp.exp(-0.5 * (distance / self.plan.adhesion_length) ** 2)
        )
        total = (
            helfrich
            + gaussian
            + local_area
            + global_area
            + volume_constraint
            + tension
            + pressure
            + adhesion
        )
        return total, BiomembraneEnergy(
            helfrich,
            gaussian,
            local_area,
            global_area,
            volume_constraint,
            tension,
            pressure,
            adhesion,
            total,
        )

    def energy(self, state: BiomembraneState, /) -> BiomembraneEnergy:
        values = self._validate_state(state)
        return self._energy_components(values.positions, values.species_mass)[1]

    def evaluate(self, state: BiomembraneState, /) -> BiomembraneEvaluation:
        values = self._validate_state(state)

        def potential(position: Array) -> tuple[Array, BiomembraneEnergy]:
            return self._energy_components(position, values.species_mass)

        (_, energy), gradient = jax.value_and_grad(potential, has_aux=True)(
            values.positions
        )
        conservative_force = -gradient
        (
            points,
            face_area,
            vertex_area,
            vertex_normal,
            normal_magnitude,
            volume,
        ) = self._surface_geometry(values.positions)
        mean_curvature, gaussian_curvature, _ = self._curvature(
            values.positions, vertex_area, vertex_normal
        )
        concentration = values.species_mass / vertex_area[:, None]
        area_vector = 0.5 * jnp.cross(
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
        )
        face_traction = jnp.mean(self.plan.active_traction[self.plan.faces], axis=1)
        face_active_load = face_traction[:, None] * area_vector / 3.0
        active_force = jnp.zeros_like(values.positions)
        active_force = active_force.at[self.plan.faces.reshape((-1,))].add(
            jnp.repeat(face_active_load, 3, axis=0)
        )
        force = conservative_force + active_force
        total_area = jnp.sum(face_area)
        local_area_residual = (
            face_area - self.reference_face_area
        ) / self.reference_face_area
        area_residual = (total_area - self.target_area) / self.target_area
        volume_residual = (volume - self.target_volume) / self.target_volume
        net_force = jnp.sum(conservative_force, axis=0)
        center = jnp.mean(values.positions, axis=0)
        net_torque = jnp.sum(
            jnp.cross(values.positions - center, conservative_force), axis=0
        )
        force_scale = jnp.maximum(jnp.sum(jnp.abs(conservative_force)), 1.0)
        length_scale = jnp.maximum(
            jnp.max(jnp.sqrt(jnp.sum((values.positions - center) ** 2, axis=1))),
            1.0,
        )
        force_residual = jnp.sqrt(jnp.sum(net_force**2)) / force_scale
        torque_residual = jnp.sqrt(jnp.sum(net_torque**2)) / (force_scale * length_scale)
        minimum_area = jnp.min(face_area)
        minimum_normal = jnp.min(normal_magnitude)
        finite = (
            jnp.all(jnp.isfinite(values.positions))
            & jnp.all(jnp.isfinite(values.species_mass))
            & jnp.all(jnp.isfinite(face_area))
            & jnp.all(jnp.isfinite(vertex_area))
            & jnp.all(jnp.isfinite(normal_magnitude))
            & jnp.isfinite(volume)
            & jnp.isfinite(energy.total)
            & jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(mean_curvature))
            & jnp.all(jnp.isfinite(gaussian_curvature))
        )
        nondegenerate = minimum_area > self.plan.geometry_tolerance
        normal_defined = minimum_normal > self.plan.geometry_tolerance
        volume_tolerance = self.plan.geometry_tolerance * jnp.maximum(total_area, 1.0)
        positively_oriented = volume > volume_tolerance
        geometry_valid = finite & nondegenerate & normal_defined & positively_oriented
        species_nonnegative = jnp.all(
            values.species_mass >= -64.0 * jnp.finfo(values.positions.dtype).eps
        )
        valid = geometry_valid & species_nonnegative
        geometry = BiomembraneGeometryEvidence(
            face_area,
            local_area_residual,
            vertex_area,
            total_area,
            volume,
            area_residual,
            volume_residual,
            minimum_area,
            minimum_normal,
            force_residual,
            torque_residual,
            finite,
            nondegenerate,
            normal_defined,
            positively_oriented,
            geometry_valid,
        )
        return BiomembraneEvaluation(
            energy,
            conservative_force,
            active_force,
            force,
            mean_curvature,
            gaussian_curvature,
            vertex_normal,
            concentration,
            geometry,
            finite,
            valid,
            self.prepared_id,
        )

    def diffuse_react(
        self, state: BiomembraneState, step_size: ArrayLike, /
    ) -> BiomembraneTransportResult:
        values = self._validate_state(state)
        step = jnp.asarray(step_size, dtype=values.positions.dtype)
        if step.shape != ():
            raise ValueError("step_size must be scalar.")
        (
            _,
            face_area,
            vertex_area,
            vertex_normal,
            normal_magnitude,
            volume,
        ) = self._surface_geometry(values.positions)
        _, _, cotangent_sum = self._curvature(
            values.positions,
            vertex_area,
            vertex_normal,
        )
        edge = self.plan.edge_vertices
        concentration = values.species_mass / vertex_area[:, None]
        conductance = 0.5 * cotangent_sum
        edge_flux = (
            conductance[:, None]
            * self.plan.species_diffusivity[None, :]
            * (concentration[edge[:, 1]] - concentration[edge[:, 0]])
        )
        diffusion_rate = jnp.zeros_like(values.species_mass)
        diffusion_rate = diffusion_rate.at[edge[:, 0]].add(edge_flux)
        diffusion_rate = diffusion_rate.at[edge[:, 1]].add(-edge_flux)
        reaction_rate = vertex_area[:, None] * contract(
            "st,vt->vs", self.plan.reaction_matrix, concentration
        )
        mass_rate = diffusion_rate + reaction_rate
        candidate_mass = values.species_mass + step * mass_rate
        candidate = BiomembraneState(values.positions, candidate_mass, self.prepared_id)
        before = jnp.sum(values.species_mass, axis=0)
        after = jnp.sum(candidate_mass, axis=0)
        residual = after - before
        total_residual = jnp.sum(after) - jnp.sum(before)
        minimum = (
            jnp.min(candidate_mass)
            if self.plan.species_count
            else jnp.asarray(0.0, dtype=step.dtype)
        )
        finite = (
            jnp.isfinite(step)
            & (step >= 0.0)
            & jnp.all(jnp.isfinite(face_area))
            & jnp.all(jnp.isfinite(vertex_area))
            & jnp.all(jnp.isfinite(normal_magnitude))
            & jnp.isfinite(volume)
            & jnp.all(jnp.isfinite(concentration))
            & jnp.all(jnp.isfinite(cotangent_sum))
            & jnp.all(jnp.isfinite(candidate_mass))
            & jnp.all(jnp.isfinite(edge_flux))
            & jnp.all(jnp.isfinite(reaction_rate))
            & jnp.isfinite(total_residual)
        )
        nonnegative = minimum >= -64.0 * jnp.finfo(step.dtype).eps
        total_area = jnp.sum(face_area)
        volume_tolerance = self.plan.geometry_tolerance * jnp.maximum(total_area, 1.0)
        source_valid = (
            jnp.all(jnp.isfinite(values.positions))
            & jnp.all(jnp.isfinite(values.species_mass))
            & (jnp.min(face_area) > self.plan.geometry_tolerance)
            & (jnp.min(normal_magnitude) > self.plan.geometry_tolerance)
            & (volume > volume_tolerance)
            & jnp.all(
                values.species_mass >= -64.0 * jnp.finfo(values.positions.dtype).eps
            )
        )
        mass_scale = jnp.maximum(
            jnp.maximum(
                jnp.sum(jnp.abs(before)),
                jnp.sum(jnp.abs(after)),
            ),
            1.0,
        )
        mass_tolerance = 128.0 * jnp.finfo(step.dtype).eps * mass_scale
        conservative = jnp.abs(total_residual) <= mass_tolerance
        successful = finite & nonnegative & source_valid & conservative
        accepted = BiomembraneState(
            values.positions,
            jnp.where(successful, candidate_mass, values.species_mass),
            self.prepared_id,
        )
        evidence = BiomembraneTransportEvidence(
            before,
            after,
            residual,
            total_residual,
            minimum,
            finite,
            nonnegative,
            conservative,
            successful,
        )
        return BiomembraneTransportResult(
            candidate,
            accepted,
            mass_rate,
            edge_flux,
            reaction_rate,
            evidence,
            self.prepared_id,
        )

    def thermal_step(
        self,
        state: BiomembraneState,
        key: Array,
        step_size: ArrayLike,
        temperature: ArrayLike,
        /,
        *,
        boltzmann_constant: float = 1.0,
        step_index: ArrayLike = 0,
    ) -> BiomembraneThermalStepResult:
        values = self._validate_state(state)
        random_key = jnp.asarray(key)
        if random_key.shape != () and random_key.shape != (2,):
            raise ValueError("key must be a JAX scalar or legacy two-word PRNG key.")
        step = jnp.asarray(step_size, dtype=values.positions.dtype)
        thermal = jnp.asarray(temperature, dtype=values.positions.dtype)
        index = jnp.asarray(step_index, dtype=jnp.uint32)
        if step.shape != () or thermal.shape != () or index.shape != ():
            raise ValueError("step_size, temperature, and step_index must be scalars.")
        boltzmann = float(boltzmann_constant)
        if not isfinite(boltzmann) or boltzmann <= 0.0:
            raise ValueError("boltzmann_constant must be finite and positive.")
        initial = self.evaluate(values)
        mobility = self.plan.mobility[:, None]
        deterministic = step * mobility * initial.force
        folded = jax.random.fold_in(jax.random.fold_in(random_key, self.rng_tag), index)
        normal = jax.random.normal(
            folded, values.positions.shape, dtype=values.positions.dtype
        )
        variance = 2.0 * boltzmann * thermal * step * mobility
        stochastic = jnp.sqrt(jnp.maximum(variance, 0.0)) * normal
        candidate = BiomembraneState(
            values.positions + deterministic + stochastic,
            values.species_mass,
            self.prepared_id,
        )
        candidate_evaluation = self.evaluate(candidate)
        finite = (
            jnp.isfinite(step)
            & (step >= 0.0)
            & jnp.isfinite(thermal)
            & (thermal >= 0.0)
            & jnp.all(jnp.isfinite(stochastic))
            & candidate_evaluation.finite
        )
        successful = finite & initial.valid & candidate_evaluation.valid
        accepted = BiomembraneState(
            jnp.where(successful, candidate.positions, values.positions),
            values.species_mass,
            self.prepared_id,
        )
        positive_variance = variance > 0.0
        whitened = jnp.where(
            positive_variance,
            stochastic**2 / jnp.where(positive_variance, variance, 1.0),
            0.0,
        )
        active_coordinates = jnp.maximum(
            values.positions.shape[1] * jnp.sum(positive_variance), 1
        )
        observed = jnp.sum(whitened) / active_coordinates
        rng_identity = canonical_fingerprint(
            {"kind": "biomembrane-thermal-stream", "prepared": self.prepared_id}
        )
        evidence = BiomembraneThermalEvidence(
            deterministic,
            stochastic,
            jnp.broadcast_to(variance, values.positions.shape),
            observed,
            finite,
            candidate_evaluation.geometry.valid,
            successful,
            rng_identity,
        )
        return BiomembraneThermalStepResult(
            candidate,
            accepted,
            initial,
            candidate_evaluation,
            evidence,
            self.prepared_id,
        )

    @checked
    def couple_immersed_boundary(
        self,
        state: BiomembraneState,
        marker_velocity: ArrayLike,
        forcing: ImmersedBoundaryForcingPlan,
        fluid_velocity: ArrayLike,
        fluid_density: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        fluid_mask: ArrayLike | None = None,
    ) -> BiomembraneFluidCouplingResult:
        values = self._validate_state(state)
        velocity = jnp.asarray(marker_velocity, dtype=values.positions.dtype)
        if velocity.shape != values.positions.shape:
            raise ValueError("marker_velocity must match membrane positions.")
        evaluation = self.evaluate(values)
        result = forcing.apply(
            fluid_velocity,
            fluid_density,
            values.positions,
            velocity,
            evaluation.geometry.vertex_area,
            step_size,
            fluid_mask=fluid_mask,
            body_indices=jnp.zeros((self.plan.vertex_count,), dtype=jnp.int32),
            body_centers=jnp.mean(values.positions, axis=0, keepdims=True),
        )
        membrane_force = -result.marker_force
        work = jnp.sum(result.ledger.body_work)
        total_force = evaluation.force + membrane_force
        finite = (
            jnp.all(jnp.isfinite(membrane_force))
            & jnp.all(jnp.isfinite(total_force))
            & jnp.isfinite(result.ledger.force_balance_residual)
            & jnp.isfinite(work)
        )
        successful = finite & result.evidence.successful & evaluation.valid
        return BiomembraneFluidCouplingResult(
            membrane_force,
            evaluation.force,
            total_force,
            result,
            result.ledger.force_balance_residual,
            work,
            finite,
            successful,
            self.prepared_id,
        )

    def _remesh_state(self, state: BiomembraneState, /) -> MultiRegionSurfaceState:
        """Capacity-shaped E3 state with every transported field made extensive."""
        values = self._validate_state(state)
        topology = self.remesh_topology
        positions = np.zeros((topology.vertex_capacity, 3), dtype=np.float64)
        positions[: self.plan.vertex_count] = np.asarray(values.positions)
        _, vertex_area, _, _ = _host_geometry(
            np.asarray(values.positions),
            np.asarray(self.plan.faces),
            self.plan.geometry_tolerance,
        )
        material = np.stack(
            (
                np.asarray(self.plan.bending_rigidity),
                np.asarray(self.plan.gaussian_rigidity),
                np.asarray(self.plan.spontaneous_curvature),
                np.asarray(self.plan.active_traction),
                np.asarray(self.plan.mobility),
            ),
            axis=1,
        )
        active_fields = np.concatenate(
            (np.asarray(values.species_mass), vertex_area[:, None] * material), axis=1
        )
        fields = np.zeros(
            (
                topology.vertex_capacity,
                topology.slot_width,
                active_fields.shape[1],
            ),
            dtype=np.float64,
        )
        fields[: self.plan.vertex_count, 0] = active_fields
        names = tuple(
            f"species-content-{index}-{identifier}"
            for index, identifier in enumerate(self.plan.species_ids)
        ) + (
            "bending-rigidity-area-content",
            "gaussian-rigidity-area-content",
            "spontaneous-curvature-area-content",
            "active-traction-area-content",
            "mobility-area-content",
        )
        return MultiRegionSurfaceState(
            topology,
            positions,
            sheet_fields=fields,
            sheet_field_names=names,
        )

    def _face_contents(self) -> np.ndarray:
        topology = self.remesh_topology
        values = np.zeros((topology.face_capacity, 2), dtype=np.float64)
        rest_area = np.asarray(self.reference_face_area)
        values[: self.plan.face_count, 0] = rest_area
        values[: self.plan.face_count, 1] = rest_area * np.asarray(
            self.plan.local_area_modulus
        )
        return values

    def _candidate_geometry_admitted(
        self, positions: np.ndarray, faces: np.ndarray, /
    ) -> bool:
        """Membrane-only oriented-normal and transport-stencil admission."""
        edges, opposites, _ = _closed_topology(faces, positions.shape[0])
        points = positions[faces]
        area_vectors = 0.5 * np.cross(
            points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
        )
        normal_sum = np.zeros_like(positions)
        np.add.at(
            normal_sum,
            faces.reshape((-1,)),
            np.repeat(area_vectors, 3, axis=0),
        )
        normal_magnitude = np.linalg.norm(normal_sum, axis=1)
        normals_valid = bool(
            np.all(np.isfinite(normal_magnitude))
            and np.all(normal_magnitude > self.plan.geometry_tolerance)
        )
        weights = _host_cotangent_sums(
            positions, edges, opposites, self.plan.geometry_tolerance
        )
        stencil_valid = bool(np.all(weights >= -np.sqrt(self.plan.geometry_tolerance)))
        return normals_valid and stencil_valid

    def _candidate_plan(
        self,
        surface: SurfaceEventPassResult,
        vertex_fields: np.ndarray,
        face_fields: np.ndarray,
        /,
    ) -> BiomembranePlan:
        topology = surface.topology
        positions = np.asarray(surface.state.positions[: topology.vertex_count])
        faces = topology.host_faces().astype(np.int32)
        _, vertex_area, _, _ = _host_geometry(
            positions, faces, self.plan.geometry_tolerance
        )
        species = self.plan.species_count
        material = vertex_fields[:, species:] / vertex_area[:, None]
        rest_area = face_fields[:, 0]
        local_modulus = face_fields[:, 1] / rest_area
        plan = self.plan
        return BiomembranePlan(
            faces,
            vertex_ids=np.asarray(topology.vertex_global_ids[: topology.vertex_count]),
            face_ids=np.asarray(topology.face_global_ids[: topology.face_count]),
            bending_rigidity=material[:, 0],
            gaussian_rigidity=material[:, 1],
            spontaneous_curvature=material[:, 2],
            curvature_coupling=np.asarray(plan.curvature_coupling),
            local_area_modulus=local_modulus,
            global_area_modulus=float(plan.global_area_modulus),
            volume_modulus=float(plan.volume_modulus),
            target_area=float(self.target_area),
            target_volume=float(self.target_volume),
            tension=float(plan.tension),
            pressure=float(plan.pressure),
            adhesion_strength=float(plan.adhesion_strength),
            adhesion_normal=np.asarray(plan.adhesion_normal),
            adhesion_offset=float(plan.adhesion_offset),
            adhesion_length=float(plan.adhesion_length),
            active_traction=material[:, 3],
            mobility=material[:, 4],
            species_diffusivity=np.asarray(plan.species_diffusivity),
            reaction_matrix=np.asarray(plan.reaction_matrix),
            species_ids=plan.species_ids,
            remesh_capacity=plan.remesh_capacity,
            remesh_policy=plan.remesh_policy,
            remesh_region_ids=plan.remesh_region_ids,
            geometry_tolerance=plan.geometry_tolerance,
        )

    def propose_remesh(
        self,
        state: BiomembraneState,
        event: EdgeSplitProposal | EdgeCollapseProposal | EdgeFlipProposal,
        /,
    ) -> BiomembraneRemeshProposal:
        values = self._validate_state(state)
        if not isinstance(
            event, (EdgeSplitProposal, EdgeCollapseProposal, EdgeFlipProposal)
        ):
            raise TypeError(
                "event must be EdgeSplitProposal, EdgeCollapseProposal, "
                "or EdgeFlipProposal."
            )
        event_state = self._remesh_state(values)
        surface = apply_surface_events(
            self.remesh_topology,
            event_state,
            (event,),
            policy=self.plan.remesh_policy,
        )
        candidate = self
        candidate_state = values
        stencil_valid = False
        face_evidence = None
        if surface.committed:
            target = surface.topology
            positions = np.asarray(surface.state.positions[: target.vertex_count])
            faces = target.host_faces().astype(np.int32)
            stencil_valid = self._candidate_geometry_admitted(positions, faces)
            if stencil_valid:
                vertex_fields = np.asarray(
                    surface.state.sheet_fields[: target.vertex_count, 0]
                )
                face_transfer = surface.face_transfer
                if face_transfer is None:
                    raise RuntimeError(
                        "A committed surface event must expose its face transfer."
                    )
                source_face_fields = self._face_contents()
                transferred_faces = face_transfer.apply(source_face_fields)
                face_evidence = face_transfer.evidence(
                    source_face_fields, transferred_faces
                )
                target_face_fields = np.asarray(transferred_faces[: target.face_count])
                candidate_plan = self._candidate_plan(
                    surface, vertex_fields, target_face_fields
                )
                candidate = PreparedBiomembrane(
                    candidate_plan,
                    positions,
                    reference_face_area=target_face_fields[:, 0],
                    remesh_topology=target,
                )
                candidate_state = candidate.state(
                    positions, vertex_fields[:, : self.plan.species_count]
                )
        proposal_id = canonical_fingerprint(
            {
                "kind": "biomembrane-remesh-proposal",
                "source": self.prepared_id,
                "surface_evidence": surface.evidence.evidence_id,
                "event": event.kind.name,
                "edge_vertex_ids": list(event.edge_vertex_ids),
                "candidate": candidate.prepared_id,
                "stencil_valid": stencil_valid,
                "source_state": array_tree_fingerprint(
                    (
                        np.asarray(values.positions),
                        np.asarray(values.species_mass),
                    )
                ),
                "candidate_state": array_tree_fingerprint(
                    (
                        np.asarray(candidate_state.positions),
                        np.asarray(candidate_state.species_mass),
                    )
                ),
                "sheet_transfer": (
                    None
                    if surface.sheet_transfer is None
                    else surface.sheet_transfer.transfer_id
                ),
                "face_transfer": (
                    None
                    if surface.face_transfer is None
                    else surface.face_transfer.transfer_id
                ),
            }
        )
        return BiomembraneRemeshProposal(
            self,
            values,
            candidate,
            candidate_state,
            surface,
            surface.evidence.sheet_transfer,
            face_evidence,
            stencil_valid,
            event.kind,
            event.edge_vertex_ids,
            proposal_id,
        )

    @checked
    def evaluate_remesh(
        self,
        proposal: BiomembraneRemeshProposal,
        /,
        *,
        maximum_relative_area_jump: float = 0.25,
        maximum_relative_volume_jump: float = 0.25,
        maximum_relative_energy_jump: float = 0.25,
        conservation_tolerance: float = 1.0e-10,
    ) -> BiomembraneRemeshEvidence:
        if proposal.source.prepared_id != self.prepared_id:
            raise ValueError("proposal source does not match this preparation.")
        limits = tuple(
            float(value)
            for value in (
                maximum_relative_area_jump,
                maximum_relative_volume_jump,
                maximum_relative_energy_jump,
                conservation_tolerance,
            )
        )
        if any(not isfinite(value) or value < 0.0 for value in limits):
            raise ValueError("Remesh jump limits must be finite and nonnegative.")
        source_evaluation = self.evaluate(proposal.source_state)
        candidate_evaluation = proposal.candidate.evaluate(proposal.candidate_state)
        source_area = source_evaluation.geometry.total_area
        candidate_area = candidate_evaluation.geometry.total_area
        source_volume = source_evaluation.geometry.enclosed_volume
        candidate_volume = candidate_evaluation.geometry.enclosed_volume
        source_energy = source_evaluation.energy.total
        candidate_energy = candidate_evaluation.energy.total
        area_jump = candidate_area - source_area
        volume_jump = candidate_volume - source_volume
        energy_jump = candidate_energy - source_energy
        tiny = jnp.finfo(source_area.dtype).tiny
        relative_area = jnp.abs(area_jump) / jnp.maximum(jnp.abs(source_area), tiny)
        relative_volume = jnp.abs(volume_jump) / jnp.maximum(jnp.abs(source_volume), tiny)
        energy_scale = sum(
            jnp.abs(component)
            for component in (
                source_evaluation.energy.helfrich,
                source_evaluation.energy.gaussian,
                source_evaluation.energy.local_area,
                source_evaluation.energy.global_area,
                source_evaluation.energy.volume_constraint,
                source_evaluation.energy.tension,
                source_evaluation.energy.pressure,
                source_evaluation.energy.adhesion,
            )
        )
        energy_floor = jnp.sqrt(jnp.finfo(source_area.dtype).eps) * jnp.maximum(
            energy_scale, jnp.asarray(1.0, dtype=source_area.dtype)
        )
        relative_energy = jnp.abs(energy_jump) / jnp.maximum(
            jnp.abs(source_energy), energy_floor
        )
        source_mass = jnp.sum(proposal.source_state.species_mass, axis=0)
        candidate_mass = jnp.sum(proposal.candidate_state.species_mass, axis=0)
        species_jump = candidate_mass - source_mass
        source_material = jnp.sum(
            source_evaluation.geometry.vertex_area * self.plan.bending_rigidity
        )
        candidate_material = jnp.sum(
            candidate_evaluation.geometry.vertex_area
            * proposal.candidate.plan.bending_rigidity
        )
        material_jump = candidate_material - source_material
        finite = (
            source_evaluation.valid
            & candidate_evaluation.valid
            & jnp.isfinite(area_jump)
            & jnp.isfinite(volume_jump)
            & jnp.isfinite(energy_jump)
            & jnp.isfinite(relative_area)
            & jnp.isfinite(relative_volume)
            & jnp.isfinite(relative_energy)
            & jnp.all(jnp.isfinite(species_jump))
            & jnp.isfinite(material_jump)
        )
        mass_scale = jnp.maximum(jnp.sum(jnp.abs(source_mass)), 1.0)
        species_error = (
            jnp.max(jnp.abs(species_jump))
            if self.plan.species_count
            else jnp.asarray(0.0, dtype=source_area.dtype)
        )
        material_scale = jnp.maximum(jnp.abs(source_material), 1.0)
        jumps_conservative = (species_error <= limits[3] * mass_scale) & (
            jnp.abs(material_jump) <= limits[3] * material_scale
        )
        vertex_transfer = (
            jnp.asarray(False)
            if proposal.vertex_transfer_evidence is None
            else proposal.vertex_transfer_evidence.successful
        )
        face_transfer = (
            jnp.asarray(False)
            if proposal.face_transfer_evidence is None
            else proposal.face_transfer_evidence.successful
        )
        conservative_transfer = jumps_conservative & vertex_transfer & face_transfer
        within_limits = (
            (relative_area <= limits[0])
            & (relative_volume <= limits[1])
            & (relative_energy <= limits[2])
        )
        guard = jnp.asarray(
            proposal.surface_result.committed
            and proposal.stencil_valid
            and proposal.candidate.prepared_id != self.prepared_id
            and not proposal.surface_result.evidence.derivative_available
        )
        accepted = finite & conservative_transfer & within_limits & guard
        return BiomembraneRemeshEvidence(
            area_jump,
            relative_area,
            volume_jump,
            relative_volume,
            energy_jump,
            relative_energy,
            species_jump,
            material_jump,
            finite,
            jnp.asarray(proposal.stencil_valid),
            conservative_transfer,
            within_limits,
            accepted,
            proposal.surface_result.evidence,
            False,
            proposal.proposal_id,
        )

    @checked
    def commit_remesh(
        self,
        proposal: BiomembraneRemeshProposal,
        evidence: BiomembraneRemeshEvidence,
        /,
    ) -> BiomembraneRemeshResult:
        if (
            proposal.source.prepared_id != self.prepared_id
            or evidence.proposal_id != proposal.proposal_id
        ):
            raise ValueError("Remesh proposal/evidence identity mismatch.")
        committed = bool(np.asarray(evidence.accepted))
        prepared = proposal.candidate if committed else self
        state = proposal.candidate_state if committed else proposal.source_state
        surface = proposal.surface_result
        return BiomembraneRemeshResult(
            prepared,
            state,
            proposal,
            evidence,
            surface.evidence.lineage if committed else None,
            surface.transition if committed else None,
            surface.face_transition if committed else None,
            committed,
        )


__all__ = [
    "BiomembraneEnergy",
    "BiomembraneEvaluation",
    "BiomembraneFluidCouplingResult",
    "BiomembraneGeometryEvidence",
    "BiomembranePlan",
    "BiomembraneRemeshEvidence",
    "BiomembraneRemeshProposal",
    "BiomembraneRemeshResult",
    "BiomembraneState",
    "BiomembraneThermalEvidence",
    "BiomembraneThermalStepResult",
    "BiomembraneTransportEvidence",
    "BiomembraneTransportResult",
    "PreparedBiomembrane",
]
