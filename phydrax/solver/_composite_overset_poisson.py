#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Serial P1 Poisson on an exact, priority-owned overset partition.

The symmetric Nitsche terms use a cut-dependent, explicitly certified penalty.
For each original cell K, let A_K be its owned area and S_K the sum of the
lengths of its physical and inter-part atomic edges divided by A_K.  On an
edge e set gamma_e = 4 sum(S_K), over its incident cells.  Since gradients of
P1 functions are constant on K, Cauchy--Schwarz gives

    sum_e ||{d_n v}||²_e / gamma_e <= (1/4) sum_K A_K |grad v_K|².

The average is one-sided on a physical boundary. Consequently Young's
inequality bounds the two consistency terms by one half of the volume energy
plus one half of the penalty energy. Thus the bilinear form is coercive in
half the sum of those energies, independently of cut-cell size. No ghost
penalty, overlap volume, or artificial boundary data is introduced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from ..discretization.fem import FiniteElementDiscretization
from ..geometry._overset_interfaces import AtomicEdge, partition_overset_interfaces
from ..geometry._overset_partition import OversetPartition
from ..linalg import (
    ArraySpace,
    DifferentiationPolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve,
    SparseLU,
    TolerancePolicy,
)
from ..linalg._results import LinearSolveResult
from ..sparse import EdgeRelation, SparseCoordinateOperator


@dataclass(frozen=True)
class CompositeOversetPoissonResult:
    partition: OversetPartition
    interfaces: tuple[AtomicEdge, ...]
    discretizations: Mapping[str, FiniteElementDiscretization]
    coefficients_by_name: Mapping[str, Array]
    source_area_by_name: Mapping[str, float]
    physical_flux_by_name: Mapping[str, float]
    interface_flux_by_name: Mapping[str, float]
    raw_gradient_flux_by_name: Mapping[str, float]
    physical_boundary_fluxes: tuple[tuple[AtomicEdge, float], ...]
    source_content: float
    boundary_flux: float
    linear_residual: float
    normalized_residual: float
    flux_defect: float
    solve_result: LinearSolveResult


def _values(
    function: Callable[[Array], Array], points: NDArray[np.float64]
) -> NDArray[np.float64]:
    values = np.asarray(function(jnp.asarray(points)), dtype=np.float64)
    if values.shape == ():
        values = np.full(len(points), values)
    if values.shape != (len(points),) or not np.all(np.isfinite(values)):
        raise ValueError("Poisson data must return finite scalar values at each point.")
    return values


def solve_composite_poisson(
    partition: OversetPartition,
    discretizations: Mapping[str, FiniteElementDiscretization],
    field_name: str,
    boundary_values: Callable[[Array], Array],
    /,
    *,
    solve_tolerance: float,
    source: Callable[[Array], Array] | None = None,
    interfaces: tuple[AtomicEdge, ...] | None = None,
) -> CompositeOversetPoissonResult:
    """Solve one coupled system; ``source=None`` denotes a unit source.

    Data callbacks receive an (n, 2) array of physical coordinates. Three-point
    triangle and two-point edge Gaussian quadrature are used for data; geometry,
    P1 stiffness, and polynomial interface products are integrated exactly.
    Zero-support DOFs are eliminated and returned as zero, not regularized.
    The equation is -Δu=f. Boundary and interface fluxes are outward
    Nitsche numerical fluxes, ∂ₙu−gamma(u−g) on the physical boundary;
    hence source content plus physical flux vanishes in the discrete balance.
    ``raw_gradient_flux_by_name`` separately reports the uncorrected ∂ₙu.
    """
    tolerance = float(solve_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("solve_tolerance must be positive and finite.")
    if not callable(boundary_values) or (source is not None and not callable(source)):
        raise TypeError("Poisson boundary and source data must be callable.")
    if not discretizations or not partition.pieces:
        raise ValueError("A nonempty partition and discretizations are required.")
    names = tuple(discretizations)
    cells: dict[
        tuple[str, int],
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            tuple[int, int, int],
        ],
    ] = {}
    offsets = {}
    size = 0
    for name, disc in discretizations.items():
        if len(disc.dof_maps) != 1:
            raise ValueError("Composite Poisson requires one scalar P1 field per part.")
        index = disc._field_index(field_name)
        dm = disc.dof_maps[index]
        if dm.association != "vertex" or dm.component_shape:
            raise ValueError("Composite Poisson requires scalar vertex P1 DOFs.")
        offsets[name] = size
        size += dm.global_dof_count
        coordinates = np.asarray(disc.default_runtime.coordinates, dtype=np.float64)
        cell_index = 0
        geometries = disc.evaluate_geometry(field_name, disc.default_runtime.coordinates)
        for block, dofs, coordinate_dofs, geometry in zip(
            disc.mesh.blocks, dm.cell_dofs, disc.coordinate_dofs, geometries, strict=True
        ):
            vertices = np.asarray(block.vertices, dtype=np.int64)
            dofs = np.asarray(dofs, dtype=np.int64)
            coordinate_dofs = np.asarray(coordinate_dofs, dtype=np.int64)
            physical_gradients = np.asarray(geometry.physical_gradients, dtype=np.float64)
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or dofs.shape != vertices.shape
                or coordinate_dofs.shape != vertices.shape
            ):
                raise ValueError(
                    "Composite Poisson requires affine triangular P1 cells and coordinates."
                )
            for local_index, coordinate_rows in enumerate(coordinate_dofs):
                points = coordinates[coordinate_rows]
                jac = np.column_stack((points[1] - points[0], points[2] - points[0]))
                det = np.linalg.det(jac)
                if not np.isfinite(det) or det == 0:
                    raise ValueError("Degenerate triangular cell.")
                inverse = np.linalg.inv(jac)
                gradients = physical_gradients[local_index, 0]
                cell_dofs = tuple(
                    offsets[name] + int(dofs[local_index, i]) for i in range(3)
                )
                cells[name, cell_index] = (
                    points[0],
                    inverse,
                    gradients,
                    (cell_dofs[0], cell_dofs[1], cell_dofs[2]),
                )
                cell_index += 1
    areas = {}
    active = set()
    for piece in partition.pieces:
        key = (piece.part_name, piece.cell_index)
        if key not in cells:
            raise ValueError("Partition cell does not match the supplied FE spaces.")
        areas[key] = areas.get(key, 0.0) + float(piece.area)
        active.update(int(x) for x in cells[key][3])
    if any(piece.part_name not in names for piece in partition.pieces):
        raise ValueError("Partition contains an unknown part.")
    active = np.asarray(sorted(active), dtype=np.int64)
    route = {int(dof): i for i, dof in enumerate(active)}
    if interfaces is None:
        interfaces = partition_overset_interfaces(partition)
    perimeter = {key: 0.0 for key in areas}
    for edge in interfaces:
        if edge.kind in ("physical_boundary", "inter_part"):
            length = np.sqrt(float(edge.length_squared))
            for owner in edge.owners:
                perimeter[owner.part_name, owner.cell_index] += length
    ratios = {key: perimeter[key] / area for key, area in areas.items()}
    entries = {}
    rhs = np.zeros(len(active))
    source_by_name = {name: 0.0 for name in names}

    def add_matrix(indices: list[int], matrix: NDArray[np.float64]) -> None:
        for i, row in enumerate(indices):
            for j, column in enumerate(indices):
                key = (route[int(row)], route[int(column)])
                entries[key] = entries.get(key, 0.0) + matrix[i, j]

    # The exact owned area scales each original P1 cell tensor.  Route those
    # tensors through the prepared FE dof transformations and sparse assembly.
    for name, disc in discretizations.items():
        tensors = []
        cell_index = 0
        for block in disc.mesh.blocks:
            block_tensors = np.zeros((block.cell_count, 3, 3))
            for local_index in range(block.cell_count):
                key = (name, cell_index)
                if key in areas:
                    gradients = cells[key][2]
                    block_tensors[local_index] = areas[key] * (gradients @ gradients.T)
                cell_index += 1
            tensors.append(block_tensors)
        volume = disc.assemble_cell_operator(
            field_name, tensors, operator_id=f"composite-poisson-volume-{name}"
        )
        if not isinstance(volume.relation, EdgeRelation):
            raise RuntimeError(
                "Composite Poisson volume assembly requires an edge relation."
            )
        sources = np.asarray(volume.relation.source_indices, dtype=np.int64)
        targets = np.asarray(volume.relation.target_indices, dtype=np.int64)
        coefficients = np.asarray(volume.coefficients, dtype=np.float64)
        for target, source_dof, value in zip(targets, sources, coefficients, strict=True):
            if value != 0.0:
                key = (
                    route[offsets[name] + int(target)],
                    route[offsets[name] + int(source_dof)],
                )
                entries[key] = entries.get(key, 0.0) + float(value)

    def basis(key: tuple[str, int], points: NDArray[np.float64]) -> NDArray[np.float64]:
        origin, inverse, _, _ = cells[key]
        uv = (points - origin) @ inverse.T
        return np.column_stack((1 - uv.sum(axis=1), uv))

    for piece in partition.pieces:
        key = (piece.part_name, piece.cell_index)
        _, _, _, dofs = cells[key]
        vertices = np.asarray(piece.vertices, dtype=np.float64)
        for i in range(1, len(vertices) - 1):
            triangle = vertices[[0, i, i + 1]]
            a, b = triangle[1] - triangle[0], triangle[2] - triangle[0]
            area = abs(a[0] * b[1] - a[1] * b[0]) / 2
            points = (triangle.sum(axis=0) + 3 * triangle) / 6
            values = np.ones(3) if source is None else _values(source, points)
            weights = area / 3 * values
            rhs[[route[int(d)] for d in dofs]] += weights @ basis(key, points)
            source_by_name[piece.part_name] += float(weights.sum())
    edge_data = []
    t = np.asarray((0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)))
    for edge in interfaces:
        if edge.kind not in ("physical_boundary", "inter_part"):
            continue
        length = np.sqrt(float(edge.length_squared))
        start = np.asarray(edge.start, dtype=np.float64)
        end = np.asarray(edge.end, dtype=np.float64)
        points = start[None, :] * (1 - t[:, None]) + end[None, :] * t[:, None]
        normal = np.asarray(edge.normal, dtype=np.float64) / length
        gamma = 4 * sum(ratios[o.part_name, o.cell_index] for o in edge.owners)
        first = edge.owners[0]
        normal *= first.orientation
        jumps, derivatives, indices = [], [], []
        for k, owner in enumerate(edge.owners):
            key = (owner.part_name, owner.cell_index)
            _, _, gradients, dofs = cells[key]
            jumps.append((1 if k == 0 else -1) * basis(key, points))
            derivatives.append(
                (gradients @ normal) * (1 if len(edge.owners) == 1 else 0.5)
            )
            indices.extend(dofs)
        jump = np.concatenate(jumps, axis=1)
        derivative = np.concatenate(derivatives)
        weight = length / 2
        matrix = np.zeros((len(indices), len(indices)))
        for row in jump:
            matrix += weight * (
                -np.outer(row, derivative)
                - np.outer(derivative, row)
                + gamma * np.outer(row, row)
            )
        add_matrix(indices, matrix)
        if edge.kind == "physical_boundary":
            data = _values(boundary_values, points)
            rhs[[route[int(d)] for d in indices]] += weight * (
                (gamma * jump - derivative).T @ data
            )
            edge_data.append(
                (
                    first.part_name,
                    np.asarray(indices),
                    jump,
                    derivative,
                    gamma,
                    data,
                    weight,
                    edge,
                )
            )
        else:
            edge_data.append(
                (
                    None,
                    np.asarray(indices),
                    jump,
                    derivative,
                    gamma,
                    (first.part_name, edge.owners[1].part_name),
                    weight,
                    edge,
                )
            )
    if not any(edge[0] is not None for edge in edge_data):
        raise ValueError("Composite Poisson requires a physical Dirichlet boundary.")
    ordered = sorted(entries)
    rows = np.asarray([p[0] for p in ordered], dtype=np.int32)
    columns = np.asarray([p[1] for p in ordered], dtype=np.int32)
    values = jnp.asarray([entries[p] for p in ordered])
    space = ArraySpace((len(active),), dtype=values.dtype)
    operator = SparseCoordinateOperator(
        EdgeRelation(columns, rows, source_size=len(active), target_size=len(active)),
        values,
        source=space,
        target=space,
    )
    result = solve(
        LinearSystem(operator),
        jnp.asarray(rhs, dtype=values.dtype),
        policy=LinearSolvePolicy(
            SparseLU(),
            tolerance=TolerancePolicy(relative=0, absolute=tolerance),
            differentiation=DifferentiationPolicy("none"),
        ),
    )
    if not bool(jnp.all(result.successful)):
        raise RuntimeError("Composite Poisson sparse solve failed.")
    solution = np.asarray(result.value, dtype=np.float64)
    residual = float(
        np.max(np.abs(np.asarray(operator.mv(result.value)) - rhs), initial=0.0)
    )
    if (
        not np.all(np.isfinite(solution))
        or not np.isfinite(residual)
        or residual > tolerance
    ):
        raise RuntimeError("Composite Poisson sparse residual exceeds tolerance.")
    full = np.zeros(size)
    full[active] = solution
    coefficients = {
        name: jax.device_put(
            jnp.asarray(
                full[
                    offsets[name] : offsets[name]
                    + disc.dof_maps[disc._field_index(field_name)].global_dof_count
                ],
                dtype=disc.coefficient_dtype,
            ),
            disc.default_runtime.coordinates.device,
        )
        for name, disc in discretizations.items()
    }
    fluxes = {name: 0.0 for name in names}
    interface_fluxes = {name: 0.0 for name in names}
    raw_fluxes = {name: 0.0 for name in names}
    boundary_fluxes: list[tuple[AtomicEdge, float]] = []
    for name, indices, jump, derivative, gamma, data, weight, edge in edge_data:
        u = full[indices]
        if name is None:
            value = float(weight * np.sum(derivative @ u - gamma * (jump @ u)))
            interface_fluxes[data[0]] += value
            interface_fluxes[data[1]] -= value
        else:
            raw_fluxes[name] += float(2 * weight * (derivative @ u))
            value = float(weight * np.sum(derivative @ u - gamma * (jump @ u - data)))
            fluxes[name] += value
            boundary_fluxes.append((edge, value))
    total_source = sum(source_by_name.values())
    total_flux = sum(fluxes.values())
    return CompositeOversetPoissonResult(
        partition,
        interfaces,
        discretizations,
        coefficients,
        source_by_name,
        fluxes,
        interface_fluxes,
        raw_fluxes,
        tuple(boundary_fluxes),
        total_source,
        total_flux,
        residual,
        residual / max(1.0, float(np.max(np.abs(rhs), initial=0.0))),
        abs(total_source + total_flux),
        result,
    )
