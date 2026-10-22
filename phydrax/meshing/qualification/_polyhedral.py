# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from time import perf_counter
from typing import cast, final, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ... import discretization as D, equations as E, linalg as L
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._frozendict import frozendict
from ..._model._structure import model_array_bytes
from ..._strict import StrictModule
from ...discretization import PolyhedralConnectivity
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization.fem._cell_map import PreparedFiniteElementCellMap
from ...discretization.fem._point_interpolation import (
    prepare_finite_element_point_interpolation,
)
from ...discretization.finite_volume import (
    HybridMimeticDiffusion,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from ...discretization.vem import (
    prepare_polyhedral_h1_virtual_element_3d,
    PreparedPolyhedralH1VirtualElement3D,
)
from ...ein import contract
from ...geometry._embedding_state import (
    coordinate_certificate_request,
    MeshCertificateLimits,
)
from ...geometry._exact_polyhedral_geometry import (
    determinant3,
    star_tetrahedra,
    triangulate_loop,
)
from ...linalg._small_batched import prepare_exact_small_linear_actions
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ...typing import checked
from .._result import CellMeshingResult
from ._native_lifecycle import QualificationFailureFactory


class _CompiledSourceStarProgram(Protocol):
    compiled: jax.stages.Compiled

    def __call__(
        self,
        problem: E.CompiledFiniteElementProblem,
        seed: Array,
        policy: L.LinearSolvePolicy | None,
    ) -> tuple[Array, Array, Array]: ...


class _LoweredSourceStarProgram(Protocol):
    def compile(self) -> _CompiledSourceStarProgram: ...


class _SourceStarJitEntry(Protocol):
    def lower(
        self,
        problem: E.CompiledFiniteElementProblem,
        seed: Array,
        policy: L.LinearSolvePolicy | None,
    ) -> _LoweredSourceStarProgram: ...


@final
class PeriodicSourceQuadrature(StrictModule):
    points: np.ndarray
    cell_vertices: tuple[np.ndarray, ...]
    cell_points: tuple[np.ndarray, ...]
    cell_weights: tuple[np.ndarray, ...]


@checked
def prepare_periodic_source_quadrature(
    result: CellMeshingResult,
    /,
    *,
    failure_factory: QualificationFailureFactory,
    budget: CoordinateEnclosureBudget | None = None,
) -> PeriodicSourceQuadrature:
    """Integrate degree-four source polynomials with oriented star tetrahedra."""
    mesh, geometry = (result.mesh, result.geometry)
    if geometry is None or mesh.periodic_topology is None:
        raise ValueError(
            "qualification.polyhedral.result must provide geometry and periodic topology; "
            f"got geometry={type(geometry).__name__}, "
            f"periodic_topology={type(mesh.periodic_topology).__name__}"
        )
    certification = result.certification
    if certification is None:
        raise ValueError(
            "qualification.polyhedral_certificate must retain request limits; got None"
        )
    with coordinate_certificate_request(certification.request.limits, budget=budget) as budget:
        exact = geometry.source_coordinates(budget=budget)
        points = np.asarray(exact, dtype=np.float64)
        connectivity = mesh.connectivity
        match connectivity:
            case PolyhedralConnectivity():
                pass
            case _:
                raise TypeError(
                    "qualification.polyhedral.connectivity must retain polyhedral cells; "
                    f"got {type(connectivity).__name__}"
                )
        exact_points = np.asarray(exact, dtype=object)
        tetrahedra = star_tetrahedra(
            mesh, exact_points, require_positive=False, budget=budget
        )
        vertex_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)
        vertices = np.asarray(connectivity.cell_vertex_values, dtype=np.int64)
        nodes, weights = np.polynomial.legendre.leggauss(4)
        nodes, weights = ((nodes + 1.0) / 2.0, weights / 2.0)
        r, s, t = np.meshgrid(nodes, nodes, nodes, indexing="ij")
        wr, ws, wt = np.meshgrid(weights, weights, weights, indexing="ij")
        r, s, t = (r.ravel(), s.ravel(), t.ravel())
        reference_weights = (wr * ws * wt).ravel() * (1 - r) ** 2 * (1 - s)
        barycentric = np.stack(
            (
                1 - r - (1 - r) * s - (1 - r) * (1 - s) * t,
                r,
                (1 - r) * s,
                (1 - r) * (1 - s) * t,
            ),
            axis=1,
        )
        cell_points, cell_weights, cell_vertices = ([], [], [])
        for cell in range(connectivity.cell_count):
            ids = vertices[vertex_offsets[cell] : vertex_offsets[cell + 1]]
            local_points, local_weights = ([], [])
            for tetrahedron in tetrahedra[cell]:
                center, *corners_exact = tetrahedron
                columns = tuple(
                    (
                        tuple(
                            (
                                value - base
                                for value, base in zip(corner, center, strict=True)
                            )
                        )
                        for corner in corners_exact
                    )
                )
                determinant = float(determinant3(*columns, budget=budget))
                if not np.isfinite(determinant) or determinant <= 0:
                    raise failure_factory(
                        "_polyhedral_qualification_source_quadrature",
                        "Source cell is not positively star-visible under source-face orientation.",
                        {"determinant": determinant},
                    )
                corners = np.asarray(tetrahedron, dtype=np.float64)
                local_points.append(barycentric @ corners)
                local_weights.append(reference_weights * determinant)
            cell_points.append(np.concatenate(local_points))
            cell_weights.append(np.concatenate(local_weights))
            cell_vertices.append(ids)
        return PeriodicSourceQuadrature(
            points=points,
            cell_vertices=tuple(cell_vertices),
            cell_points=tuple(cell_points),
            cell_weights=tuple(cell_weights),
        )


@final
class PolyhedralSourceStarFEMPlan(StrictModule):
    """Prepared conforming P1 support, bound to exact source tetrahedron parents.

    Physical coordinates are the binary64 evaluation of the certified exact
    source, never the incidental published polyhedral vertex presentation.
    Periodic seams use the same translation-invariant polygon triangulation.
    The lifted tetrahedral mesh deliberately has no periodic topology: its
    scalar quotient is the explicit sparse constraint retained here.
    """

    space: D.FiniteElementDiscretization
    limits: MeshCertificateLimits
    constraint: L.ConstraintMap | D.FiniteElementDirichletConstraint
    tetrahedra: Array
    tetrahedron_parents: Array
    tetrahedron_volumes: Array
    basis_gradients: Array
    cell_volumes: Array
    reduced_dofs: Array
    quotient_dofs: Array
    coefficient_source_points: Array
    source_preparation_work_units: int = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)


@final
class PolyhedralSourceStarFEMState(StrictModule):
    """Accepted coefficients and independently evaluated physical diagnostics.

    ``history`` is transported accepted history, not analytic initialization.
    All numerical state is plain arrays, so callers can archive it in their
    existing field bank and independently reprepare the plan after cold restore.
    ``boundary_reactions`` are weak FE reactions, not a claimed cellwise
    conservative reconstructed H(div) flux.
    Compiler evidence is ephemeral numerical metadata, never an executable or
    prepared-operator archive.
    """

    coefficients: Array
    seed: Array
    history: tuple[Array, ...]
    gradients: Array
    vector_flux: Array
    cell_values: Array
    cell_gradients: Array
    cell_vector_flux: Array
    forcing: Array
    material: Array
    nodal_load: Array
    physical_residual: Array
    boundary_reactions: Array
    solver_status: Array
    l2_error: Array
    maximum_free_residual: Array
    conservation_defect: Array
    initial_residual: Array
    removed_density_mean: Array
    load_compatibility_residual: Array
    load_compatibility_bound: Array
    compiler: frozendict[str, int | float | str | tuple[float, ...] | None]
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)


def _canonical_source_face(
    row: tuple[int, ...], points: np.ndarray
) -> tuple[tuple[int, ...], int]:
    """Choose a translation-invariant cyclic direction, retaining orientation."""
    forward = min(row, key=lambda index: tuple(points[index]))
    position = row.index(forward)
    rotated = row[position:] + row[:position]
    reverse = (rotated[0], *reversed(rotated[1:]))
    origin = points[rotated[0]]

    def key(loop: tuple[int, ...]) -> tuple:
        return tuple(tuple(points[index] - origin) for index in loop)

    return (rotated, 1) if key(rotated) <= key(reverse) else (reverse, -1)


@checked
def prepare_polyhedral_source_star_fem(
    result: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> PolyhedralSourceStarFEMPlan:
    """Prepare the exact-source star P1 consumer under the carrier's limits.

    Shared faces are triangulated once. Periodic translated face copies choose
    the same exact coordinate-relative loop, including its direction, before
    ear clipping. This avoids opposite diagonals on reversed seam polygons.
    Nonpositive stars refuse rather than silently taking absolute volumes.
    """
    geometry, certification = result.geometry, result.certification
    connectivity = result.mesh.connectivity
    if geometry is None or certification is None:
        raise ValueError("Source-star FEM requires certified source geometry.")
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Source-star FEM requires polyhedral connectivity.")
    with coordinate_certificate_request(certification.request.limits, budget=budget) as budget:
        preparation_began = budget.work_units
        exact = np.asarray(geometry.source_coordinates(budget=budget), dtype=object)
        # Reuse the exact owner for its cell-star centers and source admission.
        stars = star_tetrahedra(result.mesh, exact, budget=budget)
        offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        triangles = []
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
            budget.reserve(12 * int(stop-start))
            row, orientation = _canonical_source_face(
                tuple(int(index) for index in vertices[start:stop]), exact
            )
            face = triangulate_loop(exact, row, budget=budget)
            triangles.append(
                face if orientation > 0 else tuple((a, c, b) for a, b, c in face)
            )
        centers = tuple(star[0][0] for star in stars)
        physical = np.asarray((*exact, *centers), dtype=np.float64)
        cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
        faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
        signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int64)
        tetrahedra, parents, volumes = [], [], []
        for cell, (start, stop) in enumerate(
            zip(cell_offsets[:-1], cell_offsets[1:], strict=True)
        ):
            for occurrence in range(start, stop):
                for a, b, c in triangles[faces[occurrence]]:
                    if signs[occurrence] < 0:
                        b, c = c, b
                    center = centers[cell]
                    determinant = determinant3(
                        *(tuple(exact[index] - center) for index in (a, b, c)),
                        budget=budget,
                    )
                    if determinant <= 0:
                        raise ValueError("Canonical source-star FEM tetrahedron is not positive.")
                    tetrahedra.append((len(exact) + cell, a, b, c))
                    parents.append(cell)
                    volumes.append(float(determinant / 6))
        cells = np.asarray(tetrahedra, dtype=np.int32)
        parent_rows = np.asarray(parents, dtype=np.int32)
        measures = np.asarray(volumes, dtype=np.float64)
        mesh = D.CellMesh.from_tetrahedra(physical, cells, block_name="source-stars")
        # Affine P1 evaluation is explicitly the rounded exact-source image;
        # its physical/source/parent binding is included in plan_id below.
        space = D.FiniteElementPlan(
            mesh, D.FiniteElementFieldSpec("u", D.lagrange_element("tetrahedron", 1)),
            budget=budget,
        ).prepare(budget=budget)
        preparation_work = budget.work_units - preparation_began
    dof_map = space.dof_maps[0]
    point_dofs = np.full(dof_map.global_dof_count, -1, dtype=np.int32)
    vertex_dofs = space.elements[0][0].entity_dofs[0]
    if len(vertex_dofs) != 4 or any(len(row) != 1 for row in vertex_dofs):
        raise ValueError("Source-star P1 coefficients require one actual functional on each tetrahedral vertex.")
    local_vertices = np.asarray([row[0] for row in vertex_dofs], dtype=np.int32)
    for route, cell in zip(np.asarray(dof_map.cell_dofs[0]), cells, strict=True):
        route = route[local_vertices]
        previous = point_dofs[route]
        if np.any((previous >= 0) & (previous != cell)):
            raise ValueError("Source-star P1 coefficient incidence must bind one actual source point.")
        point_dofs[route] = cell
    if not np.array_equal(np.sort(point_dofs), np.arange(len(physical), dtype=np.int32)):
        raise ValueError("Source-star P1 coefficients must bijectively cover the actual constructed points.")
    periodic = result.mesh.periodic_topology is not None
    if periodic:
        topology = result.mesh.periodic_topology
        if topology is None:
            raise ValueError("Periodic source-star FEM requires quotient topology.")
        vertex_orbits = np.asarray(topology.orbits(0)[0], dtype=np.int32)
        vertex_labels = np.concatenate(
            (vertex_orbits, np.arange(len(centers), dtype=np.int32) + int(vertex_orbits.max()) + 1)
        )
        labels = vertex_labels[point_dofs]
        _, quotient = np.unique(labels, return_inverse=True)
        # Scalar P1 diffusion has only componentwise constants in its kernel.
        # A single quotient gauge is coercive only on one connected component.
        adjacency: dict[int, set[int]] = {index: set() for index in range(int(quotient.max()) + 1)}
        for row in np.asarray(dof_map.cell_dofs[0], dtype=np.int32):
            labels = quotient[row]
            anchor = int(labels[0])
            for index in range(1, labels.size):
                other = int(labels[index])
                adjacency[anchor].add(other)
                adjacency[other].add(anchor)
        visited: set[int] = set()
        pending = [0]
        while pending:
            current = pending.pop()
            if current not in visited:
                visited.add(current)
                pending.extend(adjacency[current] - visited)
        if len(visited) != len(adjacency):
            raise ValueError("Periodic source-star FEM requires one connected quotient component for its one gauge.")
        # Pin one quotient vertex; the published solution is shifted to its
        # volume-weighted zero gauge after solving. All other seam orbits share
        # one sparse master and no dense incidence matrix is materialized.
        quotient_count = int(quotient.max()) + 1
        selected = quotient != quotient[0]
        masters = np.delete(np.arange(quotient_count, dtype=np.int32), quotient[0])
        columns = np.full(quotient_count, -1, dtype=np.int32)
        columns[masters] = np.arange(len(masters), dtype=np.int32)
        reduced = np.asarray([np.flatnonzero(quotient == master)[0] for master in masters], dtype=np.int32)
        relation = EdgeRelation(
            columns[quotient[selected]], np.flatnonzero(selected).astype(np.int32),
            source_size=len(masters), target_size=dof_map.global_dof_count,
        )
        prolongation = SparseCoordinateOperator(
            relation, jnp.ones(int(selected.sum()), dtype=jnp.float64),
            source=L.ArraySpace((len(masters),), dtype=jnp.float64),
            target=space.field_spaces[0].vector_space,
        )
        constraint = D.affine_dof_constraint(space, "u", prolongation)
    else:
        constraint = D.dirichlet_constraint(space, "u")
        reduced = np.asarray(constraint.free_dofs, dtype=np.int32)
        quotient = np.arange(dof_map.global_dof_count, dtype=np.int32)
    cell_map = PreparedFiniteElementCellMap(space, 0)
    reference = jnp.full((len(cells), 3), 0.25, dtype=jnp.float64)
    mapped = cell_map.evaluate(
        space.default_runtime.coordinates, jnp.arange(len(cells), dtype=jnp.int32), reference
    )
    if not np.all(np.asarray(mapped.valid)):
        raise ValueError("Source-star FEM physical coordinate maps are invalid.")
    _, reference_gradients = space.elements[0][0].tabulate(reference[:1])
    gradients = contract("ir,crd->cid", reference_gradients[0], mapped.inverse_jacobian)
    cell_volumes = np.bincount(parent_rows, weights=measures, minlength=len(centers))
    source_geometry = cell_geometry_id(geometry)
    identifier = canonical_fingerprint({
        "kind": "canonical-exact-source-star-p1",
        "source_mesh": result.mesh.mesh_id, "source_geometry": source_geometry,
        "physical_coordinates": array_tree_fingerprint(physical),
        "tetrahedra": array_tree_fingerprint(cells),
        "tetrahedron_parents": array_tree_fingerprint(parent_rows),
        "prepared": space.prepared_id, "constraint": constraint.constraint_id,
        "coefficient_source_points": array_tree_fingerprint(point_dofs),
    })
    return PolyhedralSourceStarFEMPlan(
        space=space, limits=certification.request.limits,
        constraint=constraint, tetrahedra=jnp.asarray(cells),
        tetrahedron_parents=jnp.asarray(parent_rows), tetrahedron_volumes=jnp.asarray(measures),
        basis_gradients=gradients, cell_volumes=jnp.asarray(cell_volumes),
        reduced_dofs=jnp.asarray(reduced), quotient_dofs=jnp.asarray(quotient, dtype=jnp.int32),
        coefficient_source_points=jnp.asarray(point_dofs),
        source_mesh_id=result.mesh.mesh_id, source_geometry_id=source_geometry,
        plan_id=identifier, periodic=periodic,
        source_preparation_work_units=preparation_work,
    )


@final
class PolyhedralArchiveOwners(StrictModule):
    """Actual ephemeral physical owners of one accepted polyhedral carrier."""

    vem: PreparedPolyhedralH1VirtualElement3D
    finite_volume: UnstructuredFiniteVolumeDiscretization
    hybrid: HybridMimeticDiffusion
    source_star: PolyhedralSourceStarFEMPlan


@checked
def prepare_polyhedral_archive_owners(
    carrier: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> PolyhedralArchiveOwners:
    certification = carrier.certification
    if certification is None or carrier.geometry is None:
        raise ValueError("Polyhedral archive owners require certified source geometry.")
    periodic = carrier.mesh.periodic_topology is not None
    finite_volume = UnstructuredFiniteVolumePlan.from_cell_mesh(
        carrier.mesh,
        periodic_geometry=carrier.geometry if periodic else None,
        periodic_embedding=certification.embedding if periodic else None,
        budget=budget,
    ).prepare(
        cell_geometry=carrier.geometry,
        periodic_embedding=certification.embedding if periodic else None,
        budget=budget,
    )
    return PolyhedralArchiveOwners(
        vem=prepare_polyhedral_h1_virtual_element_3d(
            carrier.mesh, cell_geometry=carrier.geometry, budget=budget
        ),
        finite_volume=finite_volume,
        hybrid=HybridMimeticDiffusion(finite_volume),
        source_star=prepare_polyhedral_source_star_fem(carrier, budget=budget),
    )


@checked
def polyhedral_archive_index_banks(
    owners: PolyhedralArchiveOwners, /
) -> frozendict[str, Array | np.ndarray]:
    fv, hybrid, star = owners.finite_volume, owners.hybrid, owners.source_star
    relation = hybrid.facet_accumulation.relation
    if not isinstance(relation, EdgeRelation):
        raise TypeError("Hybrid facet accumulation must retain its actual edge relation.")
    topology = (
        owners.vem.mesh.topology if owners.vem.mesh.periodic_topology is None
        else owners.vem.mesh.periodic_topology.quotient
    )
    return frozendict({
        "cell_ids": fv.cell_global_ids,
        "active_face_ids": fv.face_global_ids[hybrid.source_face_indices],
        "source_face_indices": hybrid.source_face_indices,
        "fv_source_face_indices": fv.source_face_indices,
        "cell_faces": hybrid.cell_faces,
        "valid": hybrid.valid,
        "owner_cells": hybrid.owner_cells,
        "neighbor_cells": hybrid.neighbor_cells,
        "owner_slots": hybrid.owner_slots,
        "facet_accumulation_source_indices": relation.source_indices,
        "facet_accumulation_target_indices": relation.target_indices,
        "facet_accumulation_valid": relation.valid,
        "vem_vertex_ids": topology.entities(0).entity_ids,
        "source_star_tetrahedron_ids": star.space.mesh.topology.entities(3).entity_ids,
        "source_star_tetrahedra": star.tetrahedra,
        "source_star_parent_cells": star.tetrahedron_parents,
        "source_star_reduced_dofs": star.reduced_dofs,
        "source_star_quotient_dofs": star.quotient_dofs,
        "source_star_coefficient_source_points": star.coefficient_source_points,
    })


def _source_star_load_bound(
    plan: PolyhedralSourceStarFEMPlan,
    density: Array,
    mean_removed: bool,
) -> float:
    """A-priori bound of the independent P1 load's global compatibility.

    Rational arithmetic covers arbitrary binary64 reduction association and
    FTZ/DAZ, not merely relative gamma bounds. The reference mean uses the
    retained parent volumes; the exact mismatch to summed tetrahedron weights
    is included explicitly. This bounds the independently assembled physical
    load, not an undocumented quadrature ordering inside the FE compiler.
    """
    u, tiny = Fraction(1, 2**53), Fraction(1, 2**1022)
    rho = tuple(Fraction(float(value)) for value in np.asarray(density))
    weights = tuple(Fraction(float(value)) for value in np.asarray(plan.cell_volumes))
    tet_weights = tuple(
        Fraction(float(value)) for value in np.asarray(plan.tetrahedron_volumes)
    )
    parents = np.asarray(plan.tetrahedron_parents)

    def multiplication_error(a: Fraction, b: Fraction) -> Fraction:
        return u * abs(a*b) + (1+u)*(tiny*(abs(a)+abs(b))+tiny*tiny) + tiny

    def reduction_error(magnitude: Fraction, count: int) -> Fraction:
        if count*u >= 1:
            raise ValueError("Source-star load reduction exceeds binary64 error admission.")
        return (count*u*magnitude + 3*count*tiny) / (1-count*u)

    denominator = sum(weights, Fraction(0))
    numerator = sum((a*b for a, b in zip(rho, weights, strict=True)), Fraction(0))
    exact_mean = numerator / denominator if mean_removed else Fraction(0)
    if not mean_removed and numerator != 0:
        raise ValueError("Periodic source-star forcing has a physically incompatible mean.")
    mean_error = Fraction(0)
    if mean_removed:
        product_error = sum(
            (multiplication_error(a, b) for a, b in zip(rho, weights, strict=True)),
            Fraction(0),
        )
        magnitude = sum((abs(a*b) for a, b in zip(rho, weights, strict=True)), Fraction(0))
        numerator_error = product_error + reduction_error(magnitude + product_error, len(rho))
        denominator_error = reduction_error(denominator, len(weights))
        # Input flush at division is covered by widening each operand by tiny.
        lower_denominator = denominator - denominator_error - tiny
        if lower_denominator <= 0:
            raise ValueError("Source-star mean denominator has no positive rounding enclosure.")
        mean_error = (
            numerator_error + tiny + abs(exact_mean)*(denominator_error+tiny)
        ) / lower_denominator
        mean_error += u*(abs(exact_mean)+mean_error) + tiny
    cell_tet_weights = [Fraction(0)] * len(weights)
    assembled_magnitude, assembly_error = Fraction(0), Fraction(0)
    for parent, volume in zip(parents, tet_weights, strict=True):
        index = int(parent)
        cell_tet_weights[index] += volume
        ideal_force = rho[index] - exact_mean
        subtraction_error = (
            mean_error + u*(abs(rho[index])+abs(exact_mean)+mean_error) + 3*tiny
            if mean_removed else Fraction(0)
        )
        force_magnitude = abs(ideal_force) + subtraction_error
        assembled_magnitude += force_magnitude * volume
        # Four nodal shares; cover multiply, identity multiplication, division
        # by four, and flushed intermediate operands conservatively.
        share_error = multiplication_error(force_magnitude, volume)
        share_error += multiplication_error(force_magnitude*volume+share_error, Fraction(1))
        share_error += u*(force_magnitude*volume+share_error) + 8*tiny
        assembly_error += subtraction_error*volume + 4*share_error
    mismatch = abs(sum(
        ((value-weight)*(value_rho-exact_mean) for value, weight, value_rho in
         zip(cell_tet_weights, weights, rho, strict=True)), Fraction(0)
    ))
    # Both scatter assembly and final reduction have at most this many adds.
    additions = 8*len(tet_weights) + len(plan.space.dof_maps[0].dof_coordinates)
    bound = mismatch + assembly_error + reduction_error(
        assembled_magnitude+assembly_error, additions
    )
    rounded = float(bound)
    return float(np.nextafter(rounded, np.inf))


@checked
def prepare_polyhedral_source_star_field_transfer(
    source: PolyhedralSourceStarFEMPlan,
    target: PolyhedralSourceStarFEMPlan,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> D.FieldTransfer:
    """Prepare one exact-located P1 interpolation and its true transpose.

    Scalar H1 values agree on every containing tetrahedron, so the first exact
    containing cell is a complete value functional even at high-valence seams.
    Exact native small actions resolve boundary points without Newton iteration
    or a candidate quota. Reference tabulation belongs to the FE owner.
    """
    coordinates = np.asarray(source.space.mesh.coordinates, dtype=np.float64)
    queries = np.asarray(target.space.dof_maps[0].dof_coordinates, dtype=np.float64)
    cells = np.asarray(source.tetrahedra, dtype=np.int32)
    owners = np.full(queries.shape[0], -1, dtype=np.int32)
    reference = np.zeros((queries.shape[0], 3), dtype=np.float64)
    with coordinate_certificate_request(source.limits, budget=budget) as budget:
        for cell, indices in enumerate(cells):
            corners = coordinates[indices]
            candidates = np.flatnonzero(
                (owners < 0)
                & np.all(queries >= np.min(corners, axis=0), axis=1)
                & np.all(queries <= np.max(corners, axis=0), axis=1)
            )
            budget.reserve(7 * len(queries), 128 + 16 * len(candidates))
            if candidates.size == 0:
                continue
            exact = tuple(
                tuple(Fraction(float(corners[row, axis])) for axis in range(3))
                for row in range(4)
            )
            matrix = tuple(
                tuple(exact[column + 1][axis] - exact[0][axis] for column in range(3))
                for axis in range(3)
            )
            right = tuple(
                tuple(Fraction(float(queries[row, axis])) - exact[0][axis] for row in candidates)
                for axis in range(3)
            )
            actions = prepare_exact_small_linear_actions(matrix, right, budget=budget)
            if actions.actions is None:
                raise ValueError("Source-star transfer requires nonsingular physical tetrahedra.")
            for column, row in enumerate(candidates):
                point = tuple(actions.actions[axis][column] for axis in range(3))
                if min(point) >= 0 and sum(point, Fraction(0)) <= 1:
                    owners[row] = cell
                    reference[row] = tuple(float(value) for value in point)
            if np.all(owners >= 0):
                break
    if np.any(owners < 0):
        raise ValueError("Source-star FEM transfer queried outside the accepted source support.")
    interpolation = prepare_finite_element_point_interpolation(
        source.space, "u", source.space.mesh.blocks[0].name, owners, reference
    )
    operator = L.FunctionLinearOperator(
        interpolation.interpolate,
        source=source.space.field_spaces[0].vector_space,
        target=target.space.field_spaces[0].vector_space,
        transpose_action=interpolation.transpose_scatter,
        operator_id=interpolation.prepared_id,
    )
    binding = D.TransferGeometryBinding(
        source.source_geometry_id, target.source_geometry_id, "topology-correspondence",
        source_topology_id=source.space.mesh.topology_id,
        target_topology_id=target.space.mesh.topology_id, coverage_defect=None,
    )
    return D.FieldTransfer(
        source.space.field_spaces[0], target.space.field_spaces[0], operator,
        dual_pullback_operator=L.transpose(operator),
        hilbert_adjoint_operator=L.adjoint(operator),
        properties=D.TransferProperties(
            constant_preserving=True, adjoint_paired=True,
            exact_on=("constants", "affine"), semantics="interpolation",
        ),
        geometry=binding,
    )


@checked
def prepare_polyhedral_vem_reconstruction_transfer(
    source: PreparedPolyhedralH1VirtualElement3D,
    target: PreparedPolyhedralH1VirtualElement3D,
    source_star: PolyhedralSourceStarFEMPlan,
    target_star: PolyhedralSourceStarFEMPlan,
    star_transfer: D.FieldTransfer,
    /,
) -> D.FieldTransfer:
    """Transfer through the actual continuous source-star H1 reconstruction.

    This is not evaluation of an unknown virtual interior. Source vertices keep
    their VEM values; each exact arithmetic-mean star center gets that same
    arithmetic-mean coefficient functional. Ordinary P1 FE interpolation then
    evaluates target vertices, with its actual reverse sparse action.
    """
    for vem, star in ((source, source_star), (target, target_star)):
        if vem.mesh.periodic_topology is not None:
            raise ValueError("Regenerated VEM reconstruction currently requires nonperiodic carriers.")
        if vem.mesh.mesh_id != star.source_mesh_id or vem.cell_geometry is None or cell_geometry_id(vem.cell_geometry) != star.source_geometry_id:
            raise ValueError("VEM reconstruction must bind its actual canonical source-star plan.")
    if star_transfer.source.field_space_id != source_star.space.field_spaces[0].field_space_id or star_transfer.target.field_space_id != target_star.space.field_spaces[0].field_space_id:
        raise ValueError("VEM reconstruction must use the actual source/target FE functional map.")
    source_space, target_space = source.vertex_field_space("vem"), target.vertex_field_space("vem")
    rows, columns, weights = [], [], []
    connectivity = source.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("VEM source-star reconstruction requires actual polyhedral vertex incidence.")
    vertex_count = connectivity.vertex_count
    vertex_offsets = np.asarray(connectivity.cell_vertex_offsets)
    vertex_ids = np.asarray(connectivity.cell_vertex_values)
    for coefficient, point in enumerate(np.asarray(source_star.coefficient_source_points)):
        if point < vertex_count:
            indices = (int(point),)
        else:
            cell = int(point) - vertex_count
            indices = tuple(map(int, vertex_ids[vertex_offsets[cell]:vertex_offsets[cell + 1]]))
        for index in indices:
            rows.append(coefficient)
            columns.append(index)
            weights.append(1.0 / len(indices))
    lift_id = canonical_fingerprint({
        "kind": "vem-to-canonical-source-star-h1",
        "vem": source.prepared_id, "star": source_star.plan_id,
        "center_functional": "arithmetic-mean-of-actual-cell-vertices",
    })
    lift = SparseCoordinateOperator(
        EdgeRelation(
            np.asarray(columns, dtype=np.int32), np.asarray(rows, dtype=np.int32),
            source_size=source.dof_count,
            target_size=source_star.space.dof_maps[0].global_dof_count,
        ),
        jnp.asarray(weights, dtype=jnp.float64),
        source=source_space.vector_space,
        target=source_star.space.field_spaces[0].vector_space,
        operator_id=lift_id,
    )
    points = np.asarray(target_star.coefficient_source_points)
    inverse = np.empty(points.size, dtype=np.int32)
    inverse[points] = np.arange(points.size, dtype=np.int32)
    restriction = SparseCoordinateOperator(
        EdgeRelation(
            inverse[:target.dof_count], np.arange(target.dof_count, dtype=np.int32),
            source_size=target_star.space.dof_maps[0].global_dof_count,
            target_size=target.dof_count,
        ),
        jnp.ones(target.dof_count, dtype=jnp.float64),
        source=target_star.space.field_spaces[0].vector_space,
        target=target_space.vector_space,
        operator_id=canonical_fingerprint({
            "kind": "canonical-source-star-to-vem-vertices",
            "star": target_star.plan_id, "vem": target.prepared_id,
        }),
    )
    primal = L.ComposedLinearOperator(
        restriction, L.ComposedLinearOperator(star_transfer.primal_operator, lift)
    )
    return D.FieldTransfer(
        source_space, target_space, primal,
        dual_pullback_operator=L.transpose(primal),
        hilbert_adjoint_operator=L.adjoint(primal),
        properties=D.TransferProperties(
            constant_preserving=True, adjoint_paired=True,
            exact_on=("constants", "affine"), semantics="reconstruction",
        ),
        geometry=D.TransferGeometryBinding(
            source_star.source_geometry_id, target_star.source_geometry_id,
            "topology-correspondence",
            source_topology_id=source.mesh.topology_id,
            target_topology_id=target.mesh.topology_id,
            coverage_defect=None,
        ),
    )


@eqx.filter_jit
def _solve_source_star_problem(
    problem: E.CompiledFiniteElementProblem,
    seed: Array,
    policy: L.LinearSolvePolicy | None,
) -> tuple[Array, Array, Array]:
    system, rhs = problem.linear_system()
    # Positive material, valid P1 tetrahedra and the admitted boundary/gauge
    # constraints make the scalar energy coercive. The periodic plan explicitly
    # proves quotient connectedness; its sparse prolongation is injective and
    # removes the sole constant. This is not an empirical spectrum assertion.
    base = system.operator
    coercive = L.FunctionLinearOperator(
        base.mv, source=base.source, target=base.target,
        transpose_action=base.transpose_mv,
        properties=L.OperatorProperties(
            self_adjoint=True, positive_definite=True,
            evidence={"self_adjoint": "construction", "positive_definite": "construction"},
        ),
    )
    system = L.LinearSystem(coercive)
    residual = rhs - system.operator.mv(seed)
    solved = L.solve(
        system, rhs, initial_guess=seed,
        policy=L.LinearSolvePolicy(L.ConjugateGradient()) if policy is None else policy,
    )
    expanded = problem.expand(solved.value)
    if not isinstance(expanded, Array):
        raise TypeError("Source-star scalar FE solve must return an array.")
    return expanded, solved.status, residual


def _star_l2_error(
    plan: PolyhedralSourceStarFEMPlan, coefficients: Array,
    exact_solution: Callable[[Array], ArrayLike],
) -> Array:
    # Independent high-order physical quadrature; no solver residual or nodal
    # interpolation-error surrogate is used as the physical error oracle.
    nodes, weights = np.polynomial.legendre.leggauss(4)
    nodes, weights = (nodes + 1) / 2, weights / 2
    r, s, t = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    wr, ws, wt = np.meshgrid(weights, weights, weights, indexing="ij")
    r, s, t = r.ravel(), s.ravel(), t.ravel()
    barycentric = jnp.asarray(np.stack((
        (1-r)*(1-s)*(1-t), r, (1-r)*s, (1-r)*(1-s)*t,
    ), axis=1))
    quadrature_weights = jnp.asarray((wr*ws*wt).ravel() * (1-r)**2 * (1-s) * 6)
    coordinates = plan.space.mesh.coordinates[plan.tetrahedra]
    sites = contract("qi,cid->cqd", barycentric, coordinates)
    local = coefficients[plan.space.dof_maps[0].cell_dofs[0]]
    basis, _ = plan.space.elements[0][0].tabulate(barycentric[:, 1:])
    numerical = contract("qi,ci->cq", basis, local)
    analytic = jnp.asarray(exact_solution(sites.reshape((-1, 3)))).reshape(numerical.shape)
    if plan.periodic:
        analytic = analytic - jnp.sum(
            analytic * quadrature_weights[None] * plan.tetrahedron_volumes[:, None]
        ) / jnp.sum(plan.tetrahedron_volumes)
    return jnp.sqrt(jnp.sum(
        (numerical-analytic)**2 * quadrature_weights[None] * plan.tetrahedron_volumes[:, None]
    ) / jnp.sum(plan.tetrahedron_volumes))


@checked
def solve_polyhedral_source_star_diffusion(
    plan: PolyhedralSourceStarFEMPlan,
    cell_forcing: ArrayLike,
    /,
    *,
    cell_material: ArrayLike,
    exact_solution: Callable[[Array], ArrayLike],
    seed: ArrayLike | None = None,
    history: Sequence[ArrayLike] = (),
    periodic_density: bool = False,
    policy: L.LinearSolvePolicy | None = None,
    budget: CoordinateEnclosureBudget | None = None,
    repeats: int = 0,
) -> PolyhedralSourceStarFEMState:
    """Solve ``-div(k grad(u))=f`` with actual transported cell coefficients.

    A seed and every historical row must already be in this prepared FE space;
    use ``prepare_polyhedral_source_star_field_transfer`` once for all old rows.
    Periodic density is converted to a compatible physical forcing by its
    volume-weighted mean. Nonperiodic forcing is not changed. Exterior data
    are supplied by the caller's analytic callable, independently of the seed.
    ``repeats`` measures warm calls on the unchanged original arguments; the
    first solve remains the scientific result.
    """
    if repeats < 0:
        raise ValueError("Source-star warm measurement repeats must be nonnegative.")
    force = jnp.asarray(cell_forcing, dtype=jnp.float64)
    material = jnp.asarray(cell_material, dtype=jnp.float64)
    expected = plan.cell_volumes.shape
    if force.shape != expected or material.shape != expected:
        raise ValueError("Source-star forcing and material must have one value per physical cell.")
    if not np.all(np.isfinite(np.asarray(force))) or not np.all(
        np.isfinite(np.asarray(material)) & (np.asarray(material) > 0)
    ):
        raise ValueError("Source-star forcing must be finite and material strictly positive.")
    if periodic_density and not plan.periodic:
        raise ValueError("Density mean removal requires a periodic quotient.")
    mean = jnp.sum(force * plan.cell_volumes) / jnp.sum(plan.cell_volumes) if periodic_density else jnp.asarray(0., dtype=jnp.float64)
    compatibility_bound = (
        _source_star_load_bound(plan, force, periodic_density) if plan.periodic else 0.0
    )
    force = force - mean
    if not np.all(np.isfinite(np.asarray(force))) or not np.isfinite(compatibility_bound):
        raise ValueError("Source-star mean/load arithmetic must remain finite.")
    space = plan.space

    def coefficient(values: Array) -> E.VariationalCoefficient:
        return E.coefficient(
            values[plan.tetrahedron_parents], location="cell",
            support_id=space.support.support_id, entity_set_id=space.cell_domain.entity_set_id,
        )

    form = E.FiniteElementForm("polyhedral-source-star-physical-diffusion", "u", (
        E.DiffusionAction("u", coefficient(material)),
        E.SourceAction("u", coefficient(force)),
    ))
    problem = E.compile_finite_element_problem(
        form, space, constraint=plan.constraint,
        dirichlet_values=None if plan.periodic else exact_solution,
        budget=budget,
    )
    count = space.dof_maps[0].global_dof_count
    initial = jnp.zeros(count, dtype=jnp.float64) if seed is None else jnp.asarray(seed, dtype=jnp.float64)
    retained_history = tuple(jnp.asarray(row, dtype=jnp.float64) for row in history)
    if any(row.shape != (count,) for row in (initial, *retained_history)):
        raise ValueError("Source-star seed/history must match the prepared coefficient space.")
    if any(not np.all(np.isfinite(np.asarray(row))) for row in (initial, *retained_history)):
        raise ValueError("Source-star seed/history must be finite accepted coefficients.")
    reduced_seed = initial[plan.reduced_dofs]
    if plan.periodic:
        reduced_seed = reduced_seed - initial[0]
    lowering_started = perf_counter()
    lowered = cast(_SourceStarJitEntry, _solve_source_star_problem).lower(
        problem, reduced_seed, policy
    )
    lowering_seconds = perf_counter() - lowering_started
    compilation_started = perf_counter()
    compiled = lowered.compile()
    compile_seconds = perf_counter() - compilation_started
    memory = compiled.compiled.memory_analysis()
    execution_started = perf_counter()
    coefficients, status, initial_residual = jax.block_until_ready(
        compiled(problem, reduced_seed, policy)
    )
    execution_seconds = perf_counter() - execution_started
    warm_samples: list[float] = []
    for _ in range(repeats):
        warm_started = perf_counter()
        jax.block_until_ready(compiled(problem, reduced_seed, policy))
        warm_samples.append(perf_counter() - warm_started)
    compiler = frozendict({
        "kernel": "_solve_source_star_problem",
        "temporary_bytes": None if memory is None else memory.temp_size_in_bytes,
        "output_bytes": None if memory is None else memory.output_size_in_bytes,
        "argument_bytes": None if memory is None else memory.argument_size_in_bytes,
        "generated_code_bytes": None if memory is None else memory.generated_code_size_in_bytes,
        "memory_status": "unavailable" if memory is None else "measured",
        "memory_scope": "compiled-executable-static-memory-requirements; outside native admission; not compiler-process peak",
        "lowering_seconds": lowering_seconds,
        "compile_seconds": compile_seconds,
        "first_execution_seconds": execution_seconds,
        "compilation_count": 1,
        "compilation_count_semantics": "explicit compile calls; not backend cache misses",
        "warm_samples_seconds": tuple(warm_samples),
        "logical_retained_bytes": model_array_bytes((
            plan, problem, initial, retained_history, reduced_seed,
            (coefficients, status, initial_residual),
        )),
    })
    if not np.all(np.asarray(status) == int(L.LinearSolveStatus.SUCCESS)):
        raise RuntimeError("Native source-star diffusion solve did not converge.")
    if plan.periodic:
        local = coefficients[space.dof_maps[0].cell_dofs[0]]
        gauge = jnp.sum(jnp.mean(local, axis=1) * plan.tetrahedron_volumes) / jnp.sum(plan.tetrahedron_volumes)
        coefficients = coefficients - gauge
    physical = evaluate_polyhedral_source_star_fields(
        plan, coefficients, force, material, compatibility_bound=compatibility_bound
    )
    return PolyhedralSourceStarFEMState(
        coefficients=coefficients, seed=initial, history=retained_history,
        gradients=physical.gradients, vector_flux=physical.vector_flux,
        cell_values=physical.cell_values, cell_gradients=physical.cell_gradients,
        cell_vector_flux=physical.cell_vector_flux, forcing=force, material=material,
        nodal_load=physical.nodal_load, physical_residual=physical.physical_residual,
        boundary_reactions=physical.boundary_reactions,
        solver_status=status, l2_error=_star_l2_error(plan, coefficients, exact_solution),
        maximum_free_residual=jnp.max(jnp.abs(physical.free_residual), initial=0),
        conservation_defect=jnp.abs(jnp.sum(physical.boundary_reactions) + jnp.sum(force * plan.cell_volumes)),
        initial_residual=initial_residual, removed_density_mean=mean,
        load_compatibility_residual=jnp.asarray(abs(float(np.sum(np.asarray(physical.nodal_load))))),
        load_compatibility_bound=jnp.asarray(compatibility_bound, dtype=jnp.float64),
        compiler=compiler,
        plan_id=plan.plan_id, prepared_id=space.prepared_id,
        runtime_id=space.default_runtime.runtime_id,
        source_geometry_id=plan.source_geometry_id,
    )


@final
class PolyhedralSourceStarPhysicalFields(StrictModule):
    gradients: Array
    vector_flux: Array
    cell_values: Array
    cell_gradients: Array
    cell_vector_flux: Array
    nodal_load: Array
    physical_residual: Array
    boundary_reactions: Array
    free_residual: Array


@checked
def evaluate_polyhedral_source_star_fields(
    plan: PolyhedralSourceStarFEMPlan,
    coefficients: Array,
    force: Array,
    material: Array,
    /,
    *,
    compatibility_bound: float,
) -> PolyhedralSourceStarPhysicalFields:
    """Evaluate physical fields of real accepted coefficients without resolving."""
    count = plan.space.dof_maps[0].global_dof_count
    if coefficients.shape != (count,) or force.shape != plan.cell_volumes.shape or material.shape != force.shape:
        raise ValueError("Source-star physical evaluation must match actual coefficient and cell supports.")
    if not np.isfinite(compatibility_bound) or compatibility_bound < 0:
        raise ValueError("Source-star physical evaluation requires a finite nonnegative compatibility bound.")
    if any(not np.all(np.isfinite(np.asarray(value))) for value in (coefficients, force, material)) or np.any(np.asarray(material) <= 0):
        raise ValueError("Source-star accepted physical fields must be finite with positive material.")
    routes = plan.space.dof_maps[0].cell_dofs[0]
    local = coefficients[routes]
    gradients = contract("ci,cid->cd", local, plan.basis_gradients)
    conductivity = material[plan.tetrahedron_parents]
    flux = -conductivity[:, None] * gradients
    loads = force[plan.tetrahedron_parents, None] * plan.tetrahedron_volumes[:, None] * jnp.ones((1, 4), dtype=jnp.float64) / 4
    local_residual = plan.tetrahedron_volumes[:, None] * conductivity[:, None] * contract(
        "cid,cd->ci", plan.basis_gradients, gradients
    ) - loads
    # An independent host assembly of physical weak balance, separate from the
    # equations compiler and its reduced algebraic residual.
    residual = np.zeros(count, dtype=np.float64)
    nodal_load = np.zeros(count, dtype=np.float64)
    np.add.at(residual, np.asarray(routes).ravel(), np.asarray(local_residual).ravel())
    np.add.at(nodal_load, np.asarray(routes).ravel(), np.asarray(loads).ravel())
    if plan.periodic and abs(float(np.sum(nodal_load))) > compatibility_bound:
        raise ValueError("Independent source-star physical load exceeds its rounding bound.")
    if plan.periodic:
        quotient_residual = np.bincount(np.asarray(plan.quotient_dofs), weights=residual)
        free_residual = jnp.asarray(quotient_residual)
        reactions = jnp.zeros(count, dtype=jnp.float64)
    else:
        free_residual = jnp.asarray(residual)[plan.reduced_dofs]
        reactions = jnp.asarray(residual).at[plan.reduced_dofs].set(0)
    weighted = plan.tetrahedron_volumes / plan.cell_volumes[plan.tetrahedron_parents]
    cell_values = jnp.zeros_like(plan.cell_volumes).at[plan.tetrahedron_parents].add(weighted * jnp.mean(local, axis=1))
    cell_gradients = jnp.zeros((len(plan.cell_volumes), 3), dtype=jnp.float64).at[plan.tetrahedron_parents].add(weighted[:, None] * gradients)
    cell_flux = jnp.zeros_like(cell_gradients).at[plan.tetrahedron_parents].add(weighted[:, None] * flux)
    return PolyhedralSourceStarPhysicalFields(
        gradients=gradients, vector_flux=flux, cell_values=cell_values,
        cell_gradients=cell_gradients, cell_vector_flux=cell_flux,
        nodal_load=jnp.asarray(nodal_load), physical_residual=jnp.asarray(residual),
        boundary_reactions=reactions, free_residual=free_residual,
    )
