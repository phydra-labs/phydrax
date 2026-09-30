# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...ein import contract
from ...exterior._algebra import map_reference_values, pullback, vector_to_form
from ...exterior._complex import ComplexBoundary, DiscreteForm
from ...exterior._form_type import FormProxy, FormTwist
from ...linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    ComplexMap,
    HilbertComplex,
    LinearSolvePolicy,
    OperatorProperties,
)
from ...linalg._algebra_operators import apply_real_map_componentwise
from ...linalg._complexes import (
    coordinate_space,
    hodge_laplacian,
    HodgeLaplacianPart,
)
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ...typing import parse
from .._cell_de_rham import AbstractCellDeRhamComplex
from .._cell_mesh import CellMesh
from .._cochain_hodge import SparseHodge
from .._side_actions import SideTraceProvider
from .._topology import CellComplexTopology
from .._views import PreparedFieldReconstruction
from ._cell_map import PreparedFiniteElementCellMap
from ._form_elements import form_element, FormBasis, FormElementFamily
from ._generic import (
    _degree_aware_reference_rule,
    _linear_reference_element,
    _local_mass_tensor,
    FiniteElementBlockGeometry,
    FiniteElementDiscretization,
    FiniteElementDofMap,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from ._point_interpolation import prepare_finite_element_field_reconstruction
from ._reference import FiniteElementSpec


def _moment_rule(
    element: FiniteElementSpec, polynomial_degree: int, /
) -> tuple[Array, Array]:
    basis = element.form_basis
    if basis is None:
        raise ValueError("Moment quadrature requires canonical form-basis metadata.")
    points, weights = [], []
    for dimension, faces in enumerate(basis.entity_vertices):
        for face, dofs in zip(faces, element.entity_dofs[dimension], strict=True):
            if not dofs:
                continue
            if dimension == 0:
                sites, cubature = (
                    jnp.zeros((1, 0), dtype=jnp.float64),
                    jnp.ones((1,), dtype=jnp.float64),
                )
            else:
                kind = (
                    f"tensor:{dimension}"
                    if basis.family == "tensor-trimmed"
                    else f"simplex:{dimension}"
                )
                sites, cubature = _degree_aware_reference_rule(kind, polynomial_degree)
            reference, functional = basis.functional_weights_at(face, sites, cubature)
            points.append(reference)
            weights.append(functional)
    return jnp.concatenate(points, axis=0), jnp.concatenate(weights, axis=1)


def _invert_known_cell(
    cell_map: PreparedFiniteElementCellMap,
    coordinates: Array,
    cell: int,
    physical_points: Array,
    /,
) -> Array:
    """Invert a designated parent without a nearest-cell or boundary-side guess."""
    rows = jnp.full((physical_points.shape[0],), cell, dtype=jnp.int32)
    tensor = cell_map.coordinate_element.cell_kind in (
        "quadrilateral",
        "hexahedron",
    ) or cell_map.coordinate_element.cell_kind.startswith("tensor:")
    initial = jnp.full(
        (physical_points.shape[0], cell_map.reference_dimension),
        0.5 if tensor else 1 / (cell_map.reference_dimension + 1),
        dtype=coordinates.dtype,
    )

    def step(index: int, reference: Array, /) -> Array:
        del index
        mapped = cell_map.evaluate(coordinates, rows, reference)
        return reference - contract(
            "prd,pd->pr",
            mapped.inverse_jacobian,
            mapped.physical_points - physical_points,
        )

    reference = jax.lax.fori_loop(0, 24, step, initial)
    mapped = cell_map.evaluate(coordinates, rows, reference)
    if (
        np.any(~np.asarray(mapped.valid))
        or np.max(
            np.asarray(jnp.linalg.norm(mapped.physical_points - physical_points, axis=-1))
        )
        > 1e-10
    ):
        raise ValueError(
            "The declared refinement parent does not invert the child coordinates."
        )
    return reference


def _require_array_space(space: AbstractVectorSpace, /) -> ArraySpace:
    if not isinstance(space, ArraySpace):
        raise TypeError("Finite-element moments require an array coefficient space.")
    return space


def _sparse_operator(
    rows: np.ndarray,
    columns: np.ndarray,
    values: ArrayLike,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    identifier: str,
    /,
    *,
    properties: OperatorProperties | None = None,
) -> SparseCoordinateOperator:
    relation = EdgeRelation(
        columns, rows, source_size=source.size, target_size=target.size
    )
    return SparseCoordinateOperator(
        relation,
        jnp.asarray(values, dtype=_require_array_space(target).dtype),
        source=source,
        target=target,
        operator_id=identifier,
        properties=properties,
    )


def _coalesce_entries(
    rows: np.ndarray, columns: np.ndarray, values: Array, size: int, /
) -> tuple[np.ndarray, np.ndarray, Array]:
    keys = rows.astype(np.int64) * size + columns
    unique, inverse = np.unique(keys, return_inverse=True)
    summed = (
        jnp.zeros((unique.size,), dtype=values.dtype).at[jnp.asarray(inverse)].add(values)
    )
    return (unique // size).astype(np.int32), (unique % size).astype(np.int32), summed


def _metric_hodge(
    discretization: FiniteElementDiscretization,
    degree: int,
    policy: LinearSolvePolicy | None,
    /,
) -> SparseHodge:
    dofs = discretization.dof_maps[degree]
    rows, columns, values = [], [], []
    for routes, transform, geometry in zip(
        dofs.cell_dofs,
        dofs.cell_transforms,
        discretization.block_geometries[degree],
        strict=True,
    ):
        local = _local_mass_tensor(geometry)
        local = contract("cai,cab,cbj->cij", transform, local, transform)
        indices = np.asarray(routes)
        first = np.broadcast_to(indices[:, :, None], local.shape)
        second = np.broadcast_to(indices[:, None, :], local.shape)
        keep = first <= second
        rows.append(first[keep])
        columns.append(second[keep])
        values.append(local[jnp.asarray(keep)])
    row, column, data = _coalesce_entries(
        np.concatenate(rows),
        np.concatenate(columns),
        jnp.concatenate(values),
        dofs.global_dof_count,
    )
    return SparseHodge(row, column, data, dofs.global_dof_count, policy=policy)


def _local_derivatives(
    discretization: FiniteElementDiscretization, degree: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, target = discretization.dof_maps[degree : degree + 2]
    rows, columns, values = [], [], []
    owners: set[int] = set()
    for (
        block,
        first,
        second,
        source_routes,
        target_routes,
        first_transform,
        second_transform,
    ) in zip(
        discretization.mesh.blocks,
        discretization.elements[degree],
        discretization.elements[degree + 1],
        source.cell_dofs,
        target.cell_dofs,
        source.cell_transforms,
        target.cell_transforms,
        strict=True,
    ):
        first_basis, second_basis = first.form_basis, second.form_basis
        if first_basis is None or second_basis is None:
            raise ValueError(
                "An FE de Rham differential requires canonical polynomial forms."
            )
        reference = np.asarray(first_basis.exterior_derivative_matrix(second_basis))
        for cell in range(block.cell_count):
            local = np.linalg.solve(
                np.asarray(second_transform[cell]),
                reference @ np.asarray(first_transform[cell]),
            )
            for target_local, global_row in enumerate(np.asarray(target_routes[cell])):
                if int(global_row) in owners:
                    continue
                owners.add(int(global_row))
                nonzero = np.flatnonzero(np.abs(local[target_local]) > 1e-12)
                rows.extend([int(global_row)] * nonzero.size)
                columns.extend(np.asarray(source_routes[cell])[nonzero].tolist())
                values.extend(local[target_local, nonzero].tolist())
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        np.asarray(values, dtype=np.float64),
    )


def _complex_proxy(dimension: int, degree: int, /) -> FormProxy:
    """Choose the realization's physical proxy, including ambiguous endpoint degrees."""
    if degree == 0:
        return "scalar"
    if degree == dimension:
        return "density"
    if degree == 1:
        return "circulation"
    if degree == dimension - 1:
        return "flux"
    return "components"


def _complex_fields(
    mesh: CellMesh, family: FormElementFamily, order: int, twist: FormTwist, /
) -> tuple[FiniteElementFieldSpec, ...]:
    fields = []
    for degree in range(mesh.topological_dimension + 1):
        polynomial_order = (
            order + mesh.topological_dimension - degree if family == "full" else order
        )
        proxy = _complex_proxy(mesh.topological_dimension, degree)
        elements = {
            block.name: form_element(
                block.cell_kind,
                degree,
                polynomial_order,
                family=family,
                twist=twist,
                proxy=proxy,
            )
            for block in mesh.blocks
        }
        fields.append(FiniteElementFieldSpec(f"form_{degree}", elements))
    return tuple(fields)


def _absolute_fe_complex(
    discretization: FiniteElementDiscretization,
    hodges: tuple[SparseHodge, ...],
    coefficient_dtype: DTypeLike,
    family: FormElementFamily,
    order: int,
    identifier: str,
    /,
) -> HilbertComplex:
    spaces = tuple(
        hodge.make_space(
            space_id=f"{identifier}:absolute:{degree}", dtype=coefficient_dtype
        )[0]
        for degree, hodge in enumerate(hodges)
    )
    differentials = []
    for degree in range(discretization.mesh.topological_dimension):
        if family in ("trimmed", "tensor-trimmed") and order == 1:
            incidence = discretization.mesh.topology.incidences[degree]
            relation = incidence.relation
            valid = np.asarray(relation.valid, dtype=np.bool_)
            rows = np.asarray(relation.target_indices)[valid]
            columns = np.asarray(relation.source_indices)[valid]
            values = incidence.signs[jnp.asarray(valid)]
        else:
            rows, columns, values = _local_derivatives(discretization, degree)
        differentials.append(
            _sparse_operator(
                rows,
                columns,
                values,
                spaces[degree],
                spaces[degree + 1],
                f"{identifier}:d:{degree}",
            )
        )
    return HilbertComplex(
        spaces, tuple(differentials), complex_id=f"{identifier}:absolute"
    )


def _boundary_entity_closure(
    topology: CellComplexTopology, boundary_mask: ArrayLike | None, /
) -> list[np.ndarray]:
    n = topology.dimension
    facet_boundary = np.asarray(topology.entity_sets[n - 1].subset("boundary").mask)
    selected = facet_boundary if boundary_mask is None else np.asarray(boundary_mask)
    if (
        selected.dtype != np.bool_
        or selected.shape != facet_boundary.shape
        or np.any(selected & ~facet_boundary)
    ):
        raise ValueError("boundary_mask must select topological boundary facets.")
    closure = [
        np.zeros((entities.count,), dtype=np.bool_) for entities in topology.entity_sets
    ]
    closure[n - 1] = selected
    for degree in range(n - 1, 0, -1):
        relation = topology.incidences[degree - 1].relation
        keep = (
            np.asarray(relation.valid)
            & closure[degree][np.asarray(relation.target_indices)]
        )
        closure[degree - 1][np.asarray(relation.source_indices)[keep]] = True
    return closure


def _record_boundary_cell(
    basis: FormBasis,
    entity_dofs: tuple[tuple[tuple[int, ...], ...], ...],
    cell_vertices: np.ndarray,
    cell_dofs: Array,
    cell: int,
    lookups: tuple[dict[tuple[int, ...], int], ...],
    closure: list[np.ndarray],
    mask: np.ndarray,
    orientation: np.ndarray,
    facet_signs: np.ndarray,
    degree: int,
    dimension: int,
    /,
) -> None:
    for entity_dimension, faces in enumerate(basis.entity_vertices):
        for face, local_dofs in zip(faces, entity_dofs[entity_dimension], strict=True):
            entity = lookups[entity_dimension][tuple(sorted(cell_vertices[list(face)]))]
            if closure[entity_dimension][entity]:
                mask[np.asarray(cell_dofs[cell])[list(local_dofs)]] = True
                if 0 < degree == dimension - 1 and entity_dimension == dimension - 1:
                    orientation[np.asarray(cell_dofs[cell])[list(local_dofs)]] = (
                        facet_signs[entity]
                    )


def _boundary_moment_selection(
    discretization: FiniteElementDiscretization,
    closure: list[np.ndarray],
    counts: tuple[int, ...],
    /,
) -> tuple[tuple[tuple[int, ...], ...], list[np.ndarray]]:
    from ._generic import _topology_vertex_sets

    mesh = discretization.mesh
    vertices = _topology_vertex_sets(mesh)
    lookups = tuple({face: i for i, face in enumerate(level)} for level in vertices)
    masks = [np.zeros((count,), dtype=np.bool_) for count in counts]
    orientation = [np.ones((count,), dtype=np.float64) for count in counts]
    incidence = mesh.topology.incidences[-1]
    facet_signs = np.zeros_like(closure[mesh.topological_dimension - 1], dtype=np.float64)
    valid = np.asarray(incidence.relation.valid)
    np.add.at(
        facet_signs,
        np.asarray(incidence.relation.source_indices)[valid],
        np.asarray(incidence.signs)[valid],
    )
    for degree, dofs in enumerate(discretization.dof_maps):
        for block_index, (block, element) in enumerate(
            zip(mesh.blocks, discretization.elements[degree], strict=True)
        ):
            basis = element.form_basis
            if basis is None:
                raise ValueError("FE trace requires canonical entity functionals.")
            for cell, cell_vertices in enumerate(np.asarray(block.vertices)):
                _record_boundary_cell(
                    basis,
                    element.entity_dofs,
                    cell_vertices,
                    dofs.cell_dofs[block_index],
                    cell,
                    lookups,
                    closure,
                    masks[degree],
                    orientation[degree],
                    facet_signs,
                    degree,
                    mesh.topological_dimension,
                )
    active = tuple(tuple(np.flatnonzero(mask).tolist()) for mask in masks)
    return active, orientation


def _oriented_boundary_complex(
    source: HilbertComplex,
    restricted: HilbertComplex,
    hodges: tuple[SparseHodge, ...],
    active: tuple[tuple[int, ...], ...],
    signs: tuple[Array, ...],
    identifier: str,
    /,
) -> HilbertComplex:
    spaces = []
    for degree, indices in enumerate(active[:-1]):
        hodge = hodges[degree].restrict(np.asarray(indices, dtype=np.int32))
        data = (
            hodge.upper_values
            * signs[degree][jnp.asarray(hodge.rows, dtype=jnp.int32)]
            * signs[degree][jnp.asarray(hodge.columns, dtype=jnp.int32)]
        )
        spaces.append(
            hodge.refresh(data).make_space(
                space_id=f"{identifier}:boundary:{degree}",
                dtype=_require_array_space(source.space(degree)).dtype,
            )[0]
        )
    derivatives = []
    for degree, derivative in enumerate(restricted.differentials[:-1]):
        if not isinstance(derivative, SparseCoordinateOperator):
            raise TypeError("FE boundary derivatives must be sparse coordinate maps.")
        relation = derivative.relation
        if not isinstance(relation, EdgeRelation):
            raise TypeError("FE boundary differentials require explicit edge relations.")
        data = (
            derivative.coefficients
            * signs[degree + 1][relation.target_indices]
            * signs[degree][relation.source_indices]
        )
        derivatives.append(
            _sparse_operator(
                np.asarray(relation.target_indices),
                np.asarray(relation.source_indices),
                data,
                spaces[degree],
                spaces[degree + 1],
                f"{identifier}:boundary:d:{degree}",
            )
        )
    return HilbertComplex(
        tuple(spaces), tuple(derivatives), complex_id=f"{identifier}:boundary"
    )


def _constitutive_admission(
    data: Array, cells: int, components: int, /
) -> tuple[Array, OperatorProperties]:
    allowed = ((), (cells,), (components, components), (cells, components, components))
    if data.shape not in allowed:
        raise ValueError(
            "Constitutive coefficient must be scalar, cell scalar, tensor, or cell tensor."
        )
    properties = OperatorProperties()
    if not jnp.issubdtype(data.dtype, jnp.complexfloating):
        if data.ndim <= 1:
            invalid = jnp.any(~jnp.isfinite(data) | (data <= 0))
        else:
            invalid = jnp.any(~jnp.isfinite(data)) | jnp.any(
                jnp.abs(data - jnp.swapaxes(data, -1, -2)) > 1e-12
            )
            invalid |= jnp.any(jnp.linalg.eigvalsh(data) <= 0)
        data = eqx.error_if(
            data,
            invalid,
            "Real constitutive coefficients must be symmetric positive definite.",
        )
        properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        )
    return data, properties


def _weighted_local_mass(
    geometry: FiniteElementBlockGeometry,
    data: Array,
    cells: int,
    components: int,
    offset: int,
    /,
) -> Array:
    basis = geometry.basis_values
    if basis.ndim == 2:
        basis = jnp.broadcast_to(basis[None, :, :, None], (cells, *basis.shape, 1))
    elif basis.ndim == 3:
        basis = basis[..., None]
    if data.ndim in (0, 1):
        weight = data if data.ndim == 0 else data[offset : offset + cells, None]
        return contract(
            "cq,cqiv,cqjv->cij", geometry.physical_weights * weight, basis, basis
        )
    tensor = (
        jnp.broadcast_to(data, (cells, components, components))
        if data.ndim == 2
        else data[offset : offset + cells]
    )
    return contract(
        "cq,cqiv,cvw,cqjw->cij", geometry.physical_weights, basis, tensor, basis
    )


def _constitutive_entries(
    dofs: FiniteElementDofMap,
    geometries: tuple[FiniteElementBlockGeometry, ...],
    data: Array,
    components: int,
    /,
) -> tuple[np.ndarray, np.ndarray, Array]:
    rows, columns, values = [], [], []
    offset = 0
    for routes, transform, geometry in zip(
        dofs.cell_dofs, dofs.cell_transforms, geometries, strict=True
    ):
        count = routes.shape[0]
        local = _weighted_local_mass(geometry, data, count, components, offset)
        local = contract("cai,cab,cbj->cij", transform, local, transform)
        indices = np.asarray(routes)
        rows.append(np.broadcast_to(indices[:, :, None], local.shape).reshape((-1,)))
        columns.append(np.broadcast_to(indices[:, None, :], local.shape).reshape((-1,)))
        values.append(local.reshape((-1,)))
        offset += count
    return _coalesce_entries(
        np.concatenate(rows),
        np.concatenate(columns),
        jnp.concatenate(values),
        dofs.global_dof_count,
    )


@final
class FiniteElementDeRhamComplex(AbstractCellDeRhamComplex):
    """Exact conforming polynomial forms with metric-only quadrature Riesz maps.

    Trimmed and tensor-trimmed sequences use ``order`` in every degree.
    Full sequences use polynomial order ``order + dimension - degree``:
    ``order`` specifies the top-form polynomial order.
    """

    mesh: CellMesh
    discretization: FiniteElementDiscretization
    topology: CellComplexTopology
    boundary_masks: tuple[Array, ...]
    hodges: tuple[SparseHodge, ...]
    _absolute: HilbertComplex
    _relative: HilbertComplex
    _active_absolute: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    _active_relative: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    family: FormElementFamily = eqx.field(static=True)
    order: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    primal_twist: FormTwist = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        /,
        *,
        family: FormElementFamily,
        order: int,
        twist: FormTwist = "untwisted",
        hodge_solve: LinearSolvePolicy | None = None,
        coefficient_dtype: DTypeLike = jnp.float64,
    ) -> None:
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be a CellMesh.")
        selected = parse(family, FormElementFamily, "family")
        twist_ = parse(twist, FormTwist, "twist")
        minimum_order = 0 if selected == "full" else 1
        if isinstance(order, bool) or not isinstance(order, int) or order < minimum_order:
            raise ValueError(
                f"FE complex order must be an integer at least {minimum_order} for {selected!r}."
            )
        if hodge_solve is not None and not isinstance(hodge_solve, LinearSolvePolicy):
            raise TypeError("hodge_solve must be an owning LinearSolvePolicy or None.")
        dimension = mesh.topological_dimension
        fields = _complex_fields(mesh, selected, order, twist_)
        discretization = FiniteElementDiscretization(
            FiniteElementPlan(mesh, fields, coefficient_dtype=coefficient_dtype)
        )
        hodges = tuple(
            _metric_hodge(discretization, degree, hodge_solve)
            for degree in range(dimension + 1)
        )
        identifier = canonical_fingerprint(
            {
                "kind": "finite-element-de-rham",
                "mesh": mesh.mesh_id,
                "family": selected,
                "order": order,
                "twist": twist_,
                "dtype": np.dtype(coefficient_dtype).str,
            }
        )
        absolute = _absolute_fe_complex(
            discretization, hodges, coefficient_dtype, selected, order, identifier
        )
        masks = tuple(dofs.boundary_dof_mask for dofs in discretization.dof_maps)
        active = tuple(
            tuple(np.flatnonzero(~np.asarray(mask)).tolist()) for mask in masks
        )
        relative = self._restricted_complex(
            absolute, hodges, active, f"{identifier}:relative"
        )
        self.mesh, self.discretization, self.topology = (
            mesh,
            discretization,
            mesh.topology,
        )
        self.hodges, self.boundary_masks = hodges, masks
        self._absolute, self._relative = absolute, relative
        self._active_absolute = tuple(
            tuple(range(space.size)) for space in absolute.spaces
        )
        self._active_relative = active
        self.family, self.order, self.dimension = selected, order, dimension
        self.primal_twist, self.realization_id = twist_, identifier

    @staticmethod
    def _restricted_complex(
        complex: HilbertComplex,
        hodges: tuple[SparseHodge, ...],
        active: tuple[tuple[int, ...], ...],
        identifier: str,
        /,
    ) -> HilbertComplex:
        identifier = canonical_fingerprint(
            {
                "kind": "finite-element-restriction",
                "owner": identifier,
                "source": complex.complex_id,
                "active_indices": active,
            }
        )
        spaces = tuple(
            hodge.restrict(np.asarray(indices, dtype=np.int32)).make_space(
                space_id=f"{identifier}:{degree}",
                dtype=_require_array_space(complex.space(degree)).dtype,
            )[0]
            for degree, (hodge, indices) in enumerate(zip(hodges, active, strict=True))
        )
        maps = []
        for degree, differential in enumerate(complex.differentials):
            if not isinstance(differential, SparseCoordinateOperator):
                raise TypeError(
                    "FE restriction requires sparse coordinate differentials."
                )
            relation = differential.relation
            if not isinstance(relation, EdgeRelation):
                raise TypeError(
                    "FE polynomial differentials require explicit edge relations."
                )
            row_lookup = np.full((complex.space(degree + 1).size,), -1, dtype=np.int32)
            col_lookup = np.full((complex.space(degree).size,), -1, dtype=np.int32)
            row_lookup[list(active[degree + 1])] = np.arange(
                len(active[degree + 1]), dtype=np.int32
            )
            col_lookup[list(active[degree])] = np.arange(
                len(active[degree]), dtype=np.int32
            )
            rows = row_lookup[np.asarray(relation.target_indices)]
            columns = col_lookup[np.asarray(relation.source_indices)]
            keep = np.asarray(relation.valid) & (rows >= 0) & (columns >= 0)
            maps.append(
                _sparse_operator(
                    rows[keep],
                    columns[keep],
                    differential.coefficients[jnp.asarray(keep)],
                    spaces[degree],
                    spaces[degree + 1],
                    f"{identifier}:d:{degree}",
                )
            )
        return HilbertComplex(spaces, tuple(maps), complex_id=identifier)

    def hilbert_complex(
        self, /, *, boundary: ComplexBoundary = "absolute"
    ) -> HilbertComplex:
        boundary_ = parse(boundary, ComplexBoundary, "boundary")
        return self._absolute if boundary_ == "absolute" else self._relative

    def active_indices(
        self, degree: int, /, *, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        self.form_type(degree)
        boundary_ = parse(boundary, ComplexBoundary, "boundary")
        indices = (
            self._active_absolute if boundary_ == "absolute" else self._active_relative
        )
        return jnp.asarray(indices[degree], dtype=jnp.int32)

    def _values(self, degree: int, values: ArrayLike, /) -> Array:
        self.form_type(degree)
        value = jnp.asarray(values)
        count = self._absolute.space(degree).size
        if value.shape != (count,):
            raise ValueError(
                f"Form coefficients require shape {(count,)}; got {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(jnp.float64)
        return value

    def _compact(self, degree: int, values: Array, boundary: ComplexBoundary, /) -> Array:
        if self.hilbert_complex(boundary=boundary).space(degree).size == values.size:
            return values
        return values[self.active_indices(degree, boundary=boundary)]

    def _extend(self, degree: int, values: Array, boundary: ComplexBoundary, /) -> Array:
        if values.size == self._absolute.space(degree).size:
            return values
        return (
            jnp.zeros((self._absolute.space(degree).size,), dtype=values.dtype)
            .at[self.active_indices(degree, boundary=boundary)]
            .set(values)
        )

    def exterior_derivative(
        self, degree: int, values: ArrayLike, /, *, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        """Differentiate full moment coordinates, zero extending boundary constraints."""
        value = self._values(degree, values)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        complex_ = self.hilbert_complex(boundary=boundary)
        if degree >= self.dimension:
            raise ValueError("Exterior derivative degree must be below dimension.")
        active = self._compact(degree, value, boundary)
        output = apply_real_map_componentwise(complex_.differential(degree).mv, active)
        return self._extend(degree + 1, output, boundary)

    def codifferential(
        self, degree: int, values: ArrayLike, /, *, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        """Apply the metric adjoint in the declared full moment coordinates."""
        value = self._values(degree, values)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        complex_ = self.hilbert_complex(boundary=boundary)
        if degree <= 0:
            raise ValueError("Codifferential degree must be positive.")
        active = self._compact(degree, value, boundary)
        weighted = apply_real_map_componentwise(complex_.space(degree).riesz, active)
        transposed = apply_real_map_componentwise(
            complex_.differential(degree - 1).transpose_mv, weighted
        )
        output = apply_real_map_componentwise(
            complex_.space(degree - 1).inverse_riesz, transposed
        )
        return self._extend(degree - 1, output, boundary)

    def hodge_laplacian(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
        part: HodgeLaplacianPart = "complete",
    ) -> Array:
        """Apply the requested metric Laplacian in full moment coordinates."""
        value = self._values(degree, values)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        operator = hodge_laplacian(
            self.hilbert_complex(boundary=boundary), degree, part=part
        )
        active = self._compact(degree, value, boundary)
        output = apply_real_map_componentwise(operator.mv, active)
        return self._extend(degree, output, boundary)

    def hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        space = _require_array_space(self._absolute.space(degree))
        if jnp.issubdtype(space.dtype, jnp.complexfloating):
            return space.riesz(values)
        return apply_real_map_componentwise(space.riesz, values)

    def inverse_hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        space = _require_array_space(self._absolute.space(degree))
        if jnp.issubdtype(space.dtype, jnp.complexfloating):
            return space.inverse_riesz(values)
        return apply_real_map_componentwise(space.inverse_riesz, values)

    def interpolant(
        self, degree: int, form: Callable[[Array], ArrayLike], /
    ) -> DiscreteForm:
        form_type = self.form_type(degree)
        if not callable(form):
            raise TypeError("form must evaluate canonical physical form components.")
        dofs = self.discretization.dof_maps[degree]
        result = jnp.zeros(
            (dofs.global_dof_count,),
            dtype=_require_array_space(self._absolute.space(degree)).dtype,
        )
        counts = jnp.zeros((dofs.global_dof_count,), dtype=jnp.int32)
        for block_index, element in enumerate(self.discretization.elements[degree]):
            basis = element.form_basis
            if basis is None:
                raise ValueError("Interpolation requires canonical form functionals.")
            cell_map = PreparedFiniteElementCellMap(self.discretization, block_index)
            for cell in range(cell_map.cell_count):
                points = basis.functional_points
                cells = jnp.full((points.shape[0],), cell, dtype=jnp.int32)
                geometry = cell_map.evaluate(self.mesh.coordinates, cells, points)
                values = jnp.asarray(form(geometry.physical_points))
                expected = (points.shape[0], form_type.component_count)
                if values.shape != expected:
                    raise ValueError(
                        f"Form interpolation requires canonical component shape {expected}."
                    )
                reference = pullback(values, form_type, geometry.jacobian)
                local = basis.interpolate(reference)
                canonical = jnp.linalg.solve(
                    dofs.cell_transforms[block_index][cell], local
                )
                routes = dofs.cell_dofs[block_index][cell]
                result = result.at[routes].add(canonical)
                counts = counts.at[routes].add(1)
        return DiscreteForm(self.realization_id, form_type, result / counts)

    def reconstruction(self, degree: int, /) -> PreparedFieldReconstruction:
        self.form_type(degree)
        return prepare_finite_element_field_reconstruction(
            self.discretization, f"form_{degree}"
        )

    def side_traces(self, degree: int, /) -> SideTraceProvider:
        self.form_type(degree)
        return self.discretization

    def constitutive_operator(
        self,
        degree: int,
        coefficient: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
    ) -> AbstractLinearOperator:
        """Weighted coordinate Gram, separate from the metric Hilbert pairing."""
        self.form_type(degree)
        data = jnp.asarray(coefficient)
        cells = sum(block.cell_count for block in self.mesh.blocks)
        component_count = prod(self.discretization.elements[degree][0].value_shape) or 1
        data, properties = _constitutive_admission(data, cells, component_count)
        dofs = self.discretization.dof_maps[degree]
        row, column, payload = _constitutive_entries(
            dofs, self.discretization.block_geometries[degree], data, component_count
        )
        active = np.asarray(self.active_indices(degree, boundary=boundary))
        lookup = np.full((dofs.global_dof_count,), -1, dtype=np.int32)
        lookup[active] = np.arange(active.size, dtype=np.int32)
        keep = (lookup[row] >= 0) & (lookup[column] >= 0)
        space = coordinate_space(self.hilbert_complex(boundary=boundary).space(degree))
        if not isinstance(space, ArraySpace):
            raise TypeError("FE coordinate spaces must be array spaces.")
        promoted = jnp.result_type(space.dtype, data.dtype)
        if promoted != space.dtype:
            space = ArraySpace(
                space.shape,
                dtype=promoted,
                space_id=f"{space.space_id}:complexification:{np.dtype(promoted).str}",
            )
        coefficient_id = array_tree_fingerprint(coefficient)
        return _sparse_operator(
            lookup[row[keep]],
            lookup[column[keep]],
            payload[jnp.asarray(keep)],
            space,
            space,
            f"{self.realization_id}:constitutive:{degree}:{boundary}:{coefficient_id}",
            properties=properties,
        )

    def trace_complex_map(
        self, /, *, boundary_mask: ArrayLike | None = None
    ) -> ComplexMap:
        """Outward-oriented restriction to the selected boundary facet closure."""
        closure = _boundary_entity_closure(self.topology, boundary_mask)
        active, orientation = _boundary_moment_selection(
            self.discretization, closure, self.cell_counts
        )
        selected_signs = tuple(
            tuple(orientation[degree][list(indices)].tolist())
            for degree, indices in enumerate(active[:-1])
        )
        identifier = canonical_fingerprint(
            {
                "kind": "finite-element-boundary-trace",
                "realization": self.realization_id,
                "entity_closure": tuple(
                    tuple(np.flatnonzero(mask).tolist()) for mask in closure
                ),
                "active_indices": active,
                "orientation": selected_signs,
            }
        )
        restricted = self._restricted_complex(
            self._absolute, self.hodges, active, f"{identifier}:restriction"
        )
        signs = tuple(jnp.asarray(values) for values in selected_signs)
        target = _oriented_boundary_complex(
            self._absolute, restricted, self.hodges, active, signs, identifier
        )
        maps = tuple(
            _sparse_operator(
                np.arange(len(indices), dtype=np.int32),
                np.asarray(indices, dtype=np.int32),
                signs[degree],
                self._absolute.space(degree),
                target.space(degree),
                f"{identifier}:trace:{degree}",
            )
            for degree, indices in enumerate(active[:-1])
        )
        return ComplexMap(self._absolute, target, maps, map_id=f"{identifier}:trace")

    def vector_interpolation(
        self, degree: int, /, *, boundary: ComplexBoundary = "absolute"
    ) -> AbstractLinearOperator:
        """Canonical moments of a physical piecewise linear nodal field."""
        self.form_type(degree)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        components = prod(self.discretization.elements[degree][0].value_shape) or 1
        vertices = self.mesh.coordinates.shape[0]
        nodal_mask = np.asarray(self.mesh.topology.entity_sets[0].subset("boundary").mask)
        selected = (
            np.arange(vertices, dtype=np.int32)
            if boundary == "absolute"
            else np.flatnonzero(~nodal_mask).astype(np.int32)
        )
        source = ArraySpace(
            (selected.size, components),
            dtype=_require_array_space(self._absolute.space(degree)).dtype,
            space_id=f"{self.realization_id}:nodal-vector:{components}:{boundary}",
        )
        target = self.hilbert_complex(boundary=boundary).space(degree)
        if not isinstance(target, ArraySpace):
            raise TypeError("FE vector interpolation requires array coefficient spaces.")
        vertex_lookup = np.full((vertices,), -1, dtype=np.int32)
        vertex_lookup[selected] = np.arange(selected.size, dtype=np.int32)
        active = np.asarray(self.active_indices(degree, boundary=boundary))
        target_lookup = np.full((self.cell_counts[degree],), -1, dtype=np.int32)
        target_lookup[active] = np.arange(active.size, dtype=np.int32)
        rows, columns, payload = [], [], []
        owners: set[int] = set()
        dofs = self.discretization.dof_maps[degree]
        for block_index, (block, element) in enumerate(
            zip(self.mesh.blocks, self.discretization.elements[degree], strict=True)
        ):
            basis = element.form_basis
            if basis is None:
                raise ValueError("Vector interpolation requires form functionals.")
            points = basis.functional_points
            nodal = _linear_reference_element(block.cell_kind).tabulate(points)[0]
            cell_map = PreparedFiniteElementCellMap(self.discretization, block_index)
            for cell, cell_vertices in enumerate(np.asarray(block.vertices)):
                geometry = cell_map.evaluate(
                    self.mesh.coordinates,
                    jnp.full((points.shape[0],), cell, dtype=jnp.int32),
                    points,
                )
                physical = (
                    nodal[:, :, None, None]
                    * jnp.eye(components, dtype=nodal.dtype)[None, None]
                )
                physical = physical.reshape((points.shape[0], -1, *element.value_shape))
                reference = pullback(
                    vector_to_form(physical, element.value_spec),
                    self.form_type(degree),
                    geometry.jacobian[:, None],
                )
                moments = contract("dqc,qbc->db", basis.functional_weights, reference)
                local = jnp.linalg.solve(dofs.cell_transforms[block_index][cell], moments)
                source_columns = (
                    vertex_lookup[cell_vertices, None] * components
                    + np.arange(components)[None]
                ).reshape((-1,))
                for dof, global_row in enumerate(
                    np.asarray(dofs.cell_dofs[block_index][cell])
                ):
                    row = target_lookup[global_row]
                    if row < 0 or int(global_row) in owners:
                        continue
                    owners.add(int(global_row))
                    keep = np.repeat(vertex_lookup[cell_vertices] >= 0, components)
                    rows.extend([int(row)] * int(np.sum(keep)))
                    columns.extend(source_columns[keep].tolist())
                    payload.append(local[dof, jnp.asarray(keep)])
        return _sparse_operator(
            np.asarray(rows, dtype=np.int32),
            np.asarray(columns, dtype=np.int32),
            jnp.concatenate(payload) if payload else jnp.zeros((0,), dtype=source.dtype),
            source,
            target,
            f"{self.realization_id}:vector-interpolation:{degree}:{boundary}",
        )

    def transfer(
        self,
        target: FiniteElementDeRhamComplex,
        /,
        *,
        parent_cells: ArrayLike | None = None,
    ) -> ComplexMap:
        """Same-mesh p transfer, or nested h transfer through explicit parents.

        ``parent_cells`` indexes source cells in mesh block-concatenation order.
        Child containment and the commuting interpolation are admitted here;
        this map does not claim a nonoverlap or whole-domain coverage certificate.
        """
        if not isinstance(target, FiniteElementDeRhamComplex):
            raise TypeError("target must be a FiniteElementDeRhamComplex.")
        if target.dimension != self.dimension or target.primal_twist != self.primal_twist:
            raise ValueError("FE transfer requires the same dimension and twist.")
        parents = None if parent_cells is None else np.asarray(parent_cells)
        if parents is None and target.mesh.mesh_id != self.mesh.mesh_id:
            raise ValueError(
                "Distinct meshes require an explicit nested parent_cells relation."
            )
        if parents is not None:
            shape = (sum(block.cell_count for block in target.mesh.blocks),)
            count = sum(block.cell_count for block in self.mesh.blocks)
            if (
                parents.shape != shape
                or not np.issubdtype(parents.dtype, np.integer)
                or np.any((parents < 0) | (parents >= count))
            ):
                raise ValueError(
                    "parent_cells must assign each target child one valid source cell."
                )
            self._admit_nested_parents(target, parents)
        maps = tuple(
            self._transfer_degree(target, degree, parents)
            for degree in range(self.dimension + 1)
        )
        lineage = "p" if parents is None else array_tree_fingerprint(parents)
        return ComplexMap(
            self._absolute,
            target._absolute,
            maps,
            map_id=f"{self.realization_id}:transfer:{target.realization_id}:{lineage}",
        )

    def _admit_nested_parents(
        self, target: FiniteElementDeRhamComplex, parents: np.ndarray, /
    ) -> None:
        offsets = np.cumsum((0, *(block.cell_count for block in self.mesh.blocks)))
        maps = tuple(
            PreparedFiniteElementCellMap(self.discretization, block)
            for block in range(len(self.mesh.blocks))
        )
        target_offset = 0
        for block_index, block in enumerate(target.mesh.blocks):
            target_map = PreparedFiniteElementCellMap(target.discretization, block_index)
            probe, _ = _degree_aware_reference_rule(block.cell_kind, self.dimension + 1)
            reference = jnp.concatenate(
                (target_map.coordinate_element.reference_nodes, probe), axis=0
            )
            for cell in range(block.cell_count):
                parent = int(parents[target_offset + cell])
                source_block = int(np.searchsorted(offsets[1:], parent, side="right"))
                source_cell = parent - int(offsets[source_block])
                geometry = target_map.evaluate(
                    target.mesh.coordinates,
                    jnp.full((reference.shape[0],), cell, dtype=jnp.int32),
                    reference,
                )
                source_reference = _invert_known_cell(
                    maps[source_block],
                    self.mesh.coordinates,
                    source_cell,
                    geometry.physical_points,
                )
                host = np.asarray(source_reference)
                tensor = self.family == "tensor-trimmed"
                inside = (
                    np.all((host >= -1e-10) & (host <= 1 + 1e-10))
                    if tensor
                    else np.all(host >= -1e-10)
                    and np.all(np.sum(host, axis=-1) <= 1 + 1e-10)
                )
                if not inside:
                    raise ValueError(
                        "A target child is not contained in its declared source parent."
                    )
                corners = source_reference[
                    : target_map.coordinate_element.local_dof_count
                ]
                represented = (
                    target_map.coordinate_element.tabulate(reference)[0] @ corners
                )
                if np.max(np.abs(np.asarray(represented - source_reference))) > 1e-10:
                    raise ValueError(
                        "The parent relation is not a native reference-cell refinement."
                    )
            target_offset += block.cell_count

    def _transfer_degree(
        self,
        target: FiniteElementDeRhamComplex,
        degree: int,
        parents: np.ndarray | None,
        /,
    ) -> SparseCoordinateOperator:
        source_dofs, target_dofs = (
            self.discretization.dof_maps[degree],
            target.discretization.dof_maps[degree],
        )
        rows, columns, payload = [], [], []
        owners: set[int] = set()
        offsets = np.cumsum((0, *(block.cell_count for block in self.mesh.blocks)))
        source_maps = tuple(
            PreparedFiniteElementCellMap(self.discretization, block)
            for block in range(len(self.mesh.blocks))
        )
        target_offset = 0
        for block_index, (block, element) in enumerate(
            zip(target.mesh.blocks, target.discretization.elements[degree], strict=True)
        ):
            points, functional = _moment_rule(
                element, element.degree + self.discretization.elements[degree][0].degree
            )
            target_map = PreparedFiniteElementCellMap(target.discretization, block_index)
            for cell in range(block.cell_count):
                if parents is None:
                    source_block = self.discretization.dof_maps[degree].block_names.index(
                        block.name
                    )
                    source_cell = cell
                    source_basis = self.discretization.elements[degree][
                        source_block
                    ].form_basis
                    if source_basis is None:
                        raise ValueError(
                            "FE transfer requires canonical source basis metadata."
                        )
                    reference = source_basis.tabulate_components(points)[0]
                else:
                    parent = int(parents[target_offset + cell])
                    source_block = int(np.searchsorted(offsets[1:], parent, side="right"))
                    source_cell = parent - int(offsets[source_block])
                    geometry = target_map.evaluate(
                        target.mesh.coordinates,
                        jnp.full((points.shape[0],), cell, dtype=jnp.int32),
                        points,
                    )
                    source_points = _invert_known_cell(
                        source_maps[source_block],
                        self.mesh.coordinates,
                        source_cell,
                        geometry.physical_points,
                    )
                    source_geometry = source_maps[source_block].evaluate(
                        self.mesh.coordinates,
                        jnp.full((points.shape[0],), source_cell, dtype=jnp.int32),
                        source_points,
                    )
                    source_element = self.discretization.elements[degree][source_block]
                    physical = map_reference_values(
                        source_element.tabulate(source_points)[0],
                        source_element.value_spec,
                        source_geometry.jacobian[:, None],
                    )
                    reference = pullback(
                        vector_to_form(physical, source_element.value_spec),
                        self.form_type(degree),
                        geometry.jacobian[:, None],
                    )
                local = contract("dqc,qbc->db", functional, reference)
                local = jnp.linalg.solve(
                    target_dofs.cell_transforms[block_index][cell],
                    local @ source_dofs.cell_transforms[source_block][source_cell],
                )
                source_columns = np.asarray(
                    source_dofs.cell_dofs[source_block][source_cell]
                )
                for dof, global_row in enumerate(
                    np.asarray(target_dofs.cell_dofs[block_index][cell])
                ):
                    if int(global_row) in owners:
                        continue
                    owners.add(int(global_row))
                    keep = np.abs(np.asarray(local[dof])) > 1e-12
                    rows.extend([int(global_row)] * int(np.sum(keep)))
                    columns.extend(source_columns[keep].tolist())
                    payload.append(local[dof, jnp.asarray(keep)])
            target_offset += block.cell_count
        row, column, values = _coalesce_entries(
            np.asarray(rows, dtype=np.int32),
            np.asarray(columns, dtype=np.int32),
            jnp.concatenate(payload)
            if payload
            else jnp.zeros(
                (0,), dtype=_require_array_space(self._absolute.space(degree)).dtype
            ),
            source_dofs.global_dof_count,
        )
        return _sparse_operator(
            row,
            column,
            values,
            self._absolute.space(degree),
            target._absolute.space(degree),
            f"{self.realization_id}:transfer:{target.realization_id}:{degree}",
        )


__all__ = ["FiniteElementDeRhamComplex"]
