#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-volume cell reconstructions as evidenced coordinate field views.

Structured grids locate points by certified index arithmetic on the prepared axis
edges; unstructured meshes locate points with an explicit inverse cell map
(`AbstractCellLocator`). Every finite-volume reconstruction is discontinuous
across faces, so the trace policy is cell-sided: a point on a face lies in
several cells and needs a bound trace side. No kernel searches for a nearest
cell; points outside every cell are `OUTSIDE_SUPPORT`.
"""

from __future__ import annotations

from math import isfinite, prod
from typing import assert_never, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.discretization.fem._reference_types as reference_types_owner
from phydrax.ein import contract

from ... import geometry as imported_geometry
from ..._differentiation import (
    branch_policy_contract,
    BranchDifferentiationPolicy,
    DerivativeRegularity,
    DerivativeSurface,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import GatherStencil
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import linear_apply, linear_transpose_apply
from ...typing import checked, parse
from .. import (
    _cell_geometry as imported_cell_geometry,
    _cell_geometry_validity as imported_cell_geometry_validity,
)
from .._cell_geometry import CellGeometrySpec
from .._field_query import PreparedFieldQuery
from .._reference_cell_types import CellGeometryElement
from .._simplicial_locator import (
    AbstractCellLocator,
    CellLocationStatus,
    PreparedSimplicialCellLocator,
    SimplicialLocationPolicy,
)
from .._view_support import (
    cell_location_status,
    simplicial_mesh_support_geometry,
    tensor_box_support_geometry,
    verify_mesh_support_geometry,
)
from .._views import (
    AbstractFieldReconstructionKernel,
    FieldQueryEvidence,
    FieldQueryStatus,
    FieldSideBinding,
    FieldTracePolicy,
    FieldTraceSide,
    PreparedFieldReconstruction,
)
from ..fem import (
    _cell_map as imported_fem__cell_map,
    _generic as imported_fem__generic,
    _reference as imported_fem__reference,
)
from ._cell_polynomial import PreparedCellPolynomialReconstruction
from ._reconstruction import (
    AbstractFaceReconstructionPlan,
    PiecewiseConstantReconstruction,
)
from ._side_trace import (
    finite_volume_field_space_id,
    finite_volume_scalar_space,
    FiniteVolumeCoefficientLayout,
    mesh_cell_dimension,
    structured_axis_edges,
)
from ._structured import FiniteVolumeDiscretization
from ._triangle_fv import TriangleFiniteVolumeDiscretization
from ._triangle_polynomial import TriangleKExactReconstructionPlan
from ._triangle_reconstruction import TriangleMUSCLReconstructionPlan
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_types import AbstractUnstructuredFiniteVolumeDiscretization
from ._unstructured_weno import PreparedUnstructuredWENOZReconstruction


_SIMPLICES = ("triangle", "tetrahedron")
# WENO-Z weights and MUSCL limiters branch on max/min indicators; derivatives
# with respect to the cell averages are those of the executed branch.
_WENO_BRANCH_POLICY = BranchDifferentiationPolicy.BRANCHWISE

UnstructuredCellReconstruction = (
    PiecewiseConstantReconstruction
    | PreparedCellPolynomialReconstruction
    | PreparedUnstructuredWENOZReconstruction
    | TriangleKExactReconstructionPlan
    | TriangleMUSCLReconstructionPlan
)
_MeshDiscretization = (
    UnstructuredFiniteVolumeDiscretization | TriangleFiniteVolumeDiscretization
)

type CellReconstruction = UnstructuredCellReconstruction | AbstractFaceReconstructionPlan


@final
class _CellReconstructionPolicy(StrictModule):
    reconstruction: UnstructuredCellReconstruction
    regularity: DerivativeRegularity
    maximum_order: int = eqx.field(static=True)
    coefficient_linear: bool = eqx.field(static=True)


def _mesh_coordinate_geometry(
    discretization: _MeshDiscretization, /
) -> CellGeometrySpec | None:
    """Select the coordinate source supported by the finite-volume owner kind."""
    match discretization:
        case UnstructuredFiniteVolumeDiscretization():
            return discretization.cell_geometry
        case TriangleFiniteVolumeDiscretization():
            return None
        case _:
            assert_never(discretization)


def _corner_coordinate_element_matches(
    element: CellGeometryElement, cell_kind: str, local_arity: int, /
) -> bool:
    """Admit vertex descriptors or degree-one coordinate interpolation."""
    match element:
        case imported_cell_geometry.CellVertexGeometryElement():
            return (
                element.cell_kind == cell_kind and element.local_dof_count == local_arity
            )
        case reference_types_owner.FiniteElementSpec():
            return element.degree == 1
        case _:
            return False


def _selection(accepted: Array, side: FieldSideBinding | None, /) -> tuple[Array, Array]:
    """Per-candidate shares: every containing cell for averages, else the first."""
    count = jnp.sum(accepted, axis=1, dtype=jnp.int32)
    if side is not None and side.side == "average":
        share = accepted / jnp.maximum(count, 1)[:, None]
    else:
        first = jnp.argmax(accepted, axis=1)
        share = (jnp.arange(accepted.shape[1])[None, :] == first[:, None]) & accepted
    return share, count


def _sided_status(status: Array, count: Array, side: FieldSideBinding | None, /) -> Array:
    """Discontinuous cell reconstructions need one side where cells meet."""
    if side is not None and side.side == "average":
        return status
    unresolved = (
        FieldQueryStatus.SIDE_REQUIRED
        if side is None
        else FieldQueryStatus.SIDE_UNRESOLVED
    )
    return jnp.where(
        (status == int(FieldQueryStatus.VALID)) & (count > 1), int(unresolved), status
    ).astype(jnp.int32)


def _masked_share(share: Array, status: Array, dtype: jnp.dtype, /) -> Array:
    valid = status == int(FieldQueryStatus.VALID)
    return jnp.where(valid[:, None], share, 0).astype(dtype)


def _require_value_order(derivative: tuple[int, ...], /) -> None:
    if any(derivative):
        raise ValueError(
            "finite_volume_field.derivative must have order zero for piecewise-constant averages; got a coordinate derivative."
        )


@final
class StructuredFiniteVolumeFieldReconstructionKernel(
    AbstractFieldReconstructionKernel, NonTrainableState
):
    """Piecewise-constant cell averages on a structured tensor grid.

    A point is located by index arithmetic on the prepared axis edges: along
    each axis it lies in the cells whose closed intervals contain it (within an
    absolute `tolerance` per axis), so interior faces belong to both adjacent
    cells. Face orientation is the prepared positive-axis one: the owner of an
    interior face is the lower cell along its normal axis.
    """

    edges: tuple[Array, ...]
    tolerances: Array
    cell_shape: tuple[int, ...] = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        edges: tuple[np.ndarray, ...],
        /,
        *,
        location_tolerance: float,
        field_space_id: str,
    ) -> None:
        axes = tuple(np.asarray(values, dtype=np.float64) for values in edges)
        if not axes or any(
            values.ndim != 1
            or values.size < 2
            or not np.all(np.isfinite(values))
            or np.any(np.diff(values) <= 0.0)
            for values in axes
        ):
            raise ValueError(
                "StructuredFiniteVolumeFieldReconstructionKernel.axis_edges must be finite, increasing, and span a cell; got invalid edges."
            )
        tolerance = float(location_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "prepare_finite_volume_field_reconstruction.location_tolerance must be finite and non-negative; got an invalid tolerance."
            )
        absolute = np.asarray(
            [tolerance * (values[-1] - values[0]) for values in axes], dtype=np.float64
        )
        if any(
            2.0 * width >= np.min(np.diff(values))
            for width, values in zip(absolute, axes, strict=True)
        ):
            raise ValueError(
                "prepare_finite_volume_field_reconstruction.location_tolerance must be below half the smallest cell width; got an overlapping tolerance."
            )
        self.edges = tuple(jnp.asarray(values) for values in axes)
        self.tolerances = jnp.asarray(absolute)
        self.cell_shape = tuple(values.size - 1 for values in axes)
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "structured-finite-volume-cell-average-kernel",
                "edges": [array_tree_fingerprint(values) for values in axes],
                "location_tolerance": tolerance,
                "field_space": field_space_id,
            }
        )

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    @property
    def cell_count(self) -> int:
        return prod(self.cell_shape)

    @property
    def support_coverage(self) -> str:
        return "complete"

    def _candidates(self, points: Array, /) -> tuple[Array, Array, Array]:
        """Row-major ids of every containing cell, their validity, and inside flags."""
        finite = jnp.all(jnp.isfinite(points), axis=1)
        inside = finite
        lower_cells, upper_cells = [], []
        for axis, edges in enumerate(self.edges):
            coordinate = points[:, axis]
            tolerance = self.tolerances[axis]
            last = self.cell_shape[axis] - 1
            lower = jnp.searchsorted(edges, coordinate - tolerance, side="left") - 1
            upper = jnp.searchsorted(edges, coordinate + tolerance, side="right") - 1
            inside = (
                inside
                & (coordinate >= edges[0] - tolerance)
                & (coordinate <= edges[-1] + tolerance)
            )
            lower_cells.append(jnp.clip(lower, 0, last).astype(jnp.int32))
            upper_cells.append(jnp.clip(upper, 0, last).astype(jnp.int32))
        cells, valid = [], []
        for corner in range(1 << len(self.edges)):
            index = jnp.zeros(points.shape[:1], dtype=jnp.int32)
            corner_valid = inside
            for axis, size in enumerate(self.cell_shape):
                upper_corner = bool(corner & (1 << axis))
                if upper_corner:
                    corner_valid = corner_valid & (upper_cells[axis] > lower_cells[axis])
                axis_cell = upper_cells[axis] if upper_corner else lower_cells[axis]
                index = index * size + axis_cell
            cells.append(index)
            valid.append(corner_valid)
        return jnp.stack(cells, axis=1), jnp.stack(valid, axis=1), inside

    def locate(
        self,
        points: Array,
        derivative: tuple[int, ...],
        side: FieldSideBinding | None,
        /,
    ) -> tuple[GatherStencil, FieldQueryEvidence]:
        _require_value_order(derivative)
        cells, accepted, inside = self._candidates(points)
        if side is not None and side.cell_mask is not None:
            accepted = accepted & side.cell_mask[cells]
        share, count = _selection(accepted, side)
        finite = jnp.all(jnp.isfinite(points), axis=1)
        status = jnp.where(
            ~finite,
            int(FieldQueryStatus.NONFINITE),
            jnp.where(
                ~inside,
                int(FieldQueryStatus.OUTSIDE_SUPPORT),
                jnp.where(
                    count > 0,
                    int(FieldQueryStatus.VALID),
                    int(FieldQueryStatus.SIDE_UNRESOLVED),
                ),
            ),
        ).astype(jnp.int32)
        status = _sided_status(status, count, side)
        route = GatherStencil(
            indices=cells,
            weights=_masked_share(share, status, points.dtype),
            source_size=self.cell_count,
            valid=accepted,
        )
        conditioning = jnp.ones(points.shape[:1], dtype=points.dtype)
        return route, FieldQueryEvidence(
            status, conditioning, count, kernel_id=self.kernel_id
        )

    def apply(self, route: GatherStencil, coefficients: Array, /) -> Array:
        cells = coefficients.reshape(
            (self.cell_count, *coefficients.shape[len(self.cell_shape) :])
        )
        return linear_apply(route.relation, route.weights, cells)

    def transpose(self, route: GatherStencil, cotangent: Array, /) -> Array:
        scattered = linear_transpose_apply(route.relation, route.weights, cotangent)
        return scattered.reshape((*self.cell_shape, *cotangent.shape[1:]))

    def bind_side(
        self,
        sites: np.ndarray,
        side: FieldTraceSide,
        cell_ids: np.ndarray | None,
        /,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        cells, valid, inside = (
            np.asarray(value) for value in self._candidates(jnp.asarray(sites))
        )
        if not np.all(inside):
            raise ValueError(
                "finite_volume_field.trace_sites must lie in the grid; got sites outside support."
            )
        containing = np.where(valid, cells, -1)
        if side == "average":
            if cell_ids is not None:
                raise ValueError(
                    "finite_volume_field.cell_ids must be absent for average traces; got an explicit cell selection."
                )
            return None, np.full(sites.shape[0], -1, dtype=np.int32)
        if cell_ids is None:
            site_cells = _oriented_side_cells(containing, side)
        else:
            if np.any(np.all(containing != cell_ids[:, None], axis=1)):
                raise ValueError(
                    "finite_volume_field.cell_ids must name a containing cell at each trace site; got an invalid cell selection."
                )
            site_cells = cell_ids.astype(np.int32)
        mask = np.zeros((self.cell_count,), dtype=np.bool_)
        mask[site_cells] = True
        if np.any(np.sum(valid & mask[cells], axis=1) != 1):
            raise ValueError(
                "finite_volume_field.cell_ids must resolve each trace site to its declared cell; got an ambiguous side region."
            )
        return mask, site_cells


def _oriented_side_cells(containing: np.ndarray, side: FieldTraceSide, /) -> np.ndarray:
    """Owner/neighbor cells of face sites under positive-axis face orientation.

    A site in one cell is a boundary face site: its owner is that cell and it
    has no neighbor. A site in two cells lies on one interior face whose owner
    is the lower (smaller row-major id) cell.
    """
    present = containing >= 0
    count = np.sum(present, axis=1)
    lowest = np.min(np.where(present, containing, np.iinfo(np.int32).max), axis=1)
    highest = np.max(containing, axis=1)
    match side:
        case "owner":
            resolved = (count == 1) | (count == 2)
            cells = lowest
        case "neighbor":
            resolved = count == 2
            cells = highest
        case _:
            raise ValueError(
                f"finite_volume_field.trace_side must be owner or neighbor; got {side!r}."
            )
    if not np.all(resolved):
        raise ValueError(
            f"finite_volume_field.cell_ids must explicitly resolve {side!r} traces on ambiguous sites; got multiple faces or a missing neighbor."
        )
    return cells.astype(np.int32)


@final
class _WENORoute(StrictModule):
    cells: Array
    basis: Array
    share: Array
    constant: bool = eqx.field(static=True)


@final
class _TriangleMUSCLRoute(StrictModule):
    cells: Array
    directions: Array
    share: Array
    constant: bool = eqx.field(static=True)


@final
class UnstructuredFiniteVolumeFieldReconstructionKernel(
    AbstractFieldReconstructionKernel, NonTrainableState
):
    """Located cell reconstruction on an unstructured or triangular FV mesh.

    Points are located by an explicit `AbstractCellLocator` inverse cell map.
    Inside a cell the value is the prepared reconstruction: the cell average
    (`PiecewiseConstantReconstruction`), the conservative k-certified polynomial
    (`PreparedCellPolynomialReconstruction` or `TriangleKExactReconstructionPlan`,
    linear in the averages), the WLSQ MUSCL plane
    (`TriangleMUSCLReconstructionPlan`; linear when unlimited, otherwise scaled
    by the cell's limiter factor), or the WENO-Z blended polynomial
    (`PreparedUnstructuredWENOZReconstruction`, nonlinear in the averages).
    Coordinate derivatives differentiate the cell polynomial exactly.
    """

    locator: AbstractCellLocator
    reconstruction: UnstructuredCellReconstruction
    discretization: _MeshDiscretization
    owner_cells: Array
    neighbor_cells: Array
    _cell_count: int = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        locator: AbstractCellLocator,
        reconstruction: UnstructuredCellReconstruction,
        discretization: _MeshDiscretization,
        /,
        *,
        field_space_id: str,
    ) -> None:
        match reconstruction:
            case (
                PiecewiseConstantReconstruction()
                | TriangleKExactReconstructionPlan()
                | TriangleMUSCLReconstructionPlan()
            ):
                reconstruction_id = reconstruction.plan_id
            case (
                PreparedCellPolynomialReconstruction()
                | PreparedUnstructuredWENOZReconstruction()
            ):
                reconstruction_id = reconstruction.prepared_id
            case _:
                raise TypeError(
                    "finite_volume_field.reconstruction must define a supported cell reconstruction; got an unsupported policy."
                )
        self.locator = locator
        self.reconstruction = reconstruction
        self.discretization = discretization
        self.owner_cells = discretization.owner_cells
        self.neighbor_cells = discretization.neighbor_cells
        self._cell_count = discretization.cell_count
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "unstructured-finite-volume-field-reconstruction-kernel",
                "locator": locator.locator_id,
                "reconstruction": reconstruction_id,
                "geometry": discretization.prepared_id,
                "field_space": field_space_id,
            }
        )

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    @property
    def cell_count(self) -> int:
        return self._cell_count

    @property
    def support_coverage(self) -> str:
        return "complete"

    @checked
    def prepare_packet_query(
        self,
        owner: PreparedFieldReconstruction,
        locator: AbstractCellLocator,
        points: Array,
        admissible: Array,
        /,
    ) -> PreparedFieldQuery:
        """Rebind the chart while retaining this owner's reconstruction policy.

        A donor's full coefficient-read stencil must be active. Filtering the
        containing cell alone cannot authorize a degree-k neighbor stencil
        to read a hole or fringe coefficient.
        """
        if owner.kernel is not self:
            raise ValueError(
                "finite_volume_field.packet_owner must bind this reconstruction kernel; got another kernel."
            )
        located = _checked_locator(self.discretization, locator)
        kernel = UnstructuredFiniteVolumeFieldReconstructionKernel(
            located,
            self.reconstruction,
            self.discretization,
            field_space_id=owner.field_space_id,
        )
        rebound = PreparedFieldReconstruction(
            kernel,
            support_geometry=owner.support_geometry,
            value_port=owner.value_port,
            regularity=owner.regularity,
            trace_policy=owner.trace_policy,
            coefficient_shape=owner.coefficient_shape,
            physical_dimension=owner.physical_dimension,
            maximum_derivative_order=owner.maximum_derivative_order,
            field_space_id=owner.field_space_id,
            support_id=owner.support_id,
            coefficient_linear=owner.coefficient_linear,
            approximation=owner.approximation,
            coefficient_dtype=owner.coefficient_dtype,
        )
        query = PreparedFieldQuery(rebound, points)
        support = self.query_support_rows(query)
        if not np.all(np.asarray(admissible, dtype=np.bool_)[support]):
            raise ValueError(
                "finite_volume_field.donor_stencil must contain only eligible source cells; got an inactive hole/fringe cell."
            )
        return query

    @checked
    def query_support_rows(self, query: PreparedFieldQuery, /) -> np.ndarray:
        """Return coefficient-read cells, including nonlinear global reads."""
        match query.reconstruction.kernel:
            case UnstructuredFiniteVolumeFieldReconstructionKernel() as kernel if (
                kernel.reconstruction is self.reconstruction
                and kernel.discretization is self.discretization
            ):
                pass
            case _:
                raise ValueError(
                    "finite_volume_field.query_support must bind this reconstruction and discretization; got another query owner."
                )
        match query.route:
            case GatherStencil() as route:
                return np.unique(np.asarray(route.indices)[np.asarray(route.valid)])
            case _WENORoute() | _TriangleMUSCLRoute():
                # Both nonlinear policies compute coefficients from the global operator.
                return np.arange(self.cell_count, dtype=np.int32)
            case _:
                raise ValueError(
                    "finite_volume_field.query_support must contain a supported FV route; got another route kind."
                )

    @checked
    def require_source_geometry(self, geometry: CellGeometrySpec | None, /) -> None:
        """Bind vertex geometry or the coordinate map used to prepare this field.

        Vertex descriptors must match the mesh kind, arity, coordinates, and
        routes. Mapped sources must retain the prepared coordinate geometry.
        """
        owning = _mesh_coordinate_geometry(self.discretization)
        if owning is not None:
            if geometry is None or imported_cell_geometry_validity.cell_geometry_id(
                owning
            ) != imported_cell_geometry_validity.cell_geometry_id(geometry):
                raise ValueError(
                    "finite_volume_field.cell_geometry must match the prepared source map; got a missing or different geometry ID."
                )
            return
        if geometry is None:
            return
        if (
            geometry.exact_source is not None
            or any(
                not _corner_coordinate_element_matches(
                    element, block.cell_kind, block.vertices.shape[1]
                )
                for element, block in zip(
                    geometry.elements, self.discretization.mesh.blocks, strict=True
                )
            )
            or not np.array_equal(
                np.asarray(geometry.coordinates), np.asarray(self.discretization.vertices)
            )
            or any(
                not np.array_equal(np.asarray(route), np.asarray(block.vertices))
                for route, block in zip(
                    geometry.geometry_dofs, self.discretization.mesh.blocks, strict=True
                )
            )
        ):
            raise ValueError(
                "finite_volume_field.cell_geometry must bind the mapped or restricted donor coordinate map; got a corner-only or different geometry."
            )

    def locate(
        self,
        points: Array,
        derivative: tuple[int, ...],
        side: FieldSideBinding | None,
        /,
    ) -> tuple[GatherStencil | _WENORoute | _TriangleMUSCLRoute, FieldQueryEvidence]:
        mask = None if side is None else side.cell_mask
        location = self.locator.locate(points, cell_mask=mask)
        accepted = location.candidate_cells >= 0
        cells = jnp.where(accepted, location.candidate_cells, 0)
        share, count = _selection(accepted, side)
        status = _sided_status(cell_location_status(location.status), count, side)
        share = _masked_share(share, status, points.dtype)
        evidence = FieldQueryEvidence(
            status, location.jacobian_condition, count, kernel_id=self.kernel_id
        )
        constant = float(not any(derivative))
        route: GatherStencil | _WENORoute | _TriangleMUSCLRoute
        match self.reconstruction:
            case PiecewiseConstantReconstruction():
                _require_value_order(derivative)
                route = GatherStencil(
                    indices=cells,
                    weights=share,
                    source_size=self._cell_count,
                    valid=accepted,
                )
            case PreparedCellPolynomialReconstruction() as polynomial:
                route = _stencil_route(
                    _candidate_basis(polynomial, points, derivative, cells),
                    polynomial.stencil_cells,
                    polynomial.stencil_valid,
                    polynomial.factors,
                    cells,
                    accepted,
                    share,
                    constant,
                    self._cell_count,
                )
            case PreparedUnstructuredWENOZReconstruction():
                route = _WENORoute(
                    cells,
                    _candidate_basis(
                        self.reconstruction.optimal, points, derivative, cells
                    ),
                    share,
                    not any(derivative),
                )
            case TriangleKExactReconstructionPlan() as kexact:
                prepared = kexact.prepared
                count_, width = cells.shape
                repeated = jnp.broadcast_to(points[:, None, :], (count_, width, 2))
                basis = prepared.basis_derivative(
                    cells.reshape((-1,)),
                    repeated.reshape((-1, 1, 2)),
                    (int(derivative[0]), int(derivative[1])),
                )
                route = _stencil_route(
                    basis.reshape((count_, width, basis.shape[-1])),
                    prepared.neighbor_cells,
                    prepared.valid,
                    prepared.factors,
                    cells,
                    accepted,
                    share,
                    constant,
                    self._cell_count,
                )
            case TriangleMUSCLReconstructionPlan() as muscl:
                directions = _plane_directions(
                    muscl.gradient.discretization.cell_centers.astype(points.dtype),
                    points,
                    derivative,
                    cells,
                )
                if muscl.limiter == "unlimited":
                    route = _stencil_route(
                        directions,
                        muscl.gradient.neighbor_cells,
                        muscl.gradient.valid,
                        muscl.gradient.factors,
                        cells,
                        accepted,
                        share,
                        constant,
                        self._cell_count,
                    )
                else:
                    route = _TriangleMUSCLRoute(
                        cells, directions, share, not any(derivative)
                    )
            case _:
                raise TypeError(
                    "finite_volume_field.reconstruction must define a supported cell reconstruction; got an unsupported policy."
                )
        return route, evidence

    def apply(
        self,
        route: GatherStencil | _WENORoute | _TriangleMUSCLRoute,
        coefficients: Array,
        /,
    ) -> Array:
        match route, self.reconstruction:
            case GatherStencil(), _:
                # locate() binds gather stencils for linear reconstructions.
                return linear_apply(route.relation, route.weights, coefficients)
            case (
                _WENORoute(),
                PreparedUnstructuredWENOZReconstruction() as reconstruction,
            ):
                modal = reconstruction.coefficients(coefficients)
                values = contract("pkf,pk...f->pk...", route.basis, modal[route.cells])
                if route.constant:
                    values = values + coefficients[route.cells]
                return contract("pk,pk...->p...", route.share, values)
            case _WENORoute(), _:
                raise RuntimeError(
                    "finite_volume_field.route must pair a WENO route with WENO-Z reconstruction; got a different reconstruction."
                )
            case (
                _TriangleMUSCLRoute(),
                TriangleMUSCLReconstructionPlan() as reconstruction,
            ):
                flat = route.cells.reshape((-1,))
                gradient = reconstruction.gradient.cell_gradients(coefficients, flat)
                factor = reconstruction.cell_limiter_factors(coefficients, flat)
                values = factor * contract(
                    "r...i,ri->r...",
                    gradient,
                    route.directions.reshape((-1, route.directions.shape[-1])),
                )
                if route.constant:
                    values = values + coefficients[flat]
                return contract(
                    "pk,pk...->p...",
                    route.share,
                    values.reshape(route.cells.shape + values.shape[1:]),
                )
            case _TriangleMUSCLRoute(), _:
                raise RuntimeError(
                    "finite_volume_field.route must pair a limited route with triangle MUSCL; got a different reconstruction."
                )
            case _:
                raise TypeError(
                    "finite_volume_field.route must be a supported gather or nonlinear reconstruction route; got an unsupported route."
                )

    def transpose(
        self,
        route: GatherStencil | _WENORoute | _TriangleMUSCLRoute,
        cotangent: Array,
        /,
    ) -> Array:
        match route:
            case GatherStencil():
                return linear_transpose_apply(route.relation, route.weights, cotangent)
            case _WENORoute() | _TriangleMUSCLRoute():
                raise ValueError(
                    "finite_volume_field.reconstruction must be coefficient-linear for an algebraic transpose; got WENO-Z or limited MUSCL."
                )
            case _:
                raise TypeError(
                    "finite_volume_field.route must be a supported gather or nonlinear reconstruction route; got an unsupported route."
                )

    def bind_side(
        self,
        sites: np.ndarray,
        side: FieldTraceSide,
        cell_ids: np.ndarray | None,
        /,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        location = self.locator.locate(jnp.asarray(sites))
        if np.any(np.asarray(location.status) != int(CellLocationStatus.LOCATED)):
            raise ValueError(
                "finite_volume_field.trace_sites must lie in the mesh; got sites outside support."
            )
        containing = np.asarray(location.candidate_cells)
        if side == "average":
            if cell_ids is not None:
                raise ValueError(
                    "finite_volume_field.cell_ids must be absent for average traces; got an explicit cell selection."
                )
            return None, np.full(sites.shape[0], -1, dtype=np.int32)
        if cell_ids is None:
            site_cells = _face_side_cells(
                containing,
                side,
                np.asarray(self.owner_cells),
                np.asarray(self.neighbor_cells),
            )
        else:
            if np.any(np.all(containing != cell_ids[:, None], axis=1)):
                raise ValueError(
                    "finite_volume_field.cell_ids must name a containing cell at each trace site; got an invalid cell selection."
                )
            site_cells = cell_ids.astype(np.int32)
        mask = np.zeros((self._cell_count,), dtype=np.bool_)
        mask[site_cells] = True
        masked = self.locator.locate(jnp.asarray(sites), cell_mask=jnp.asarray(mask))
        if np.any(np.asarray(masked.candidate_count) != 1) or np.any(
            np.asarray(masked.cell_ids) != site_cells
        ):
            raise ValueError(
                "finite_volume_field.cell_ids must resolve each trace site to its declared cell; got an ambiguous side region."
            )
        return mask, site_cells


def _candidate_basis(
    polynomial: PreparedCellPolynomialReconstruction,
    points: Array,
    derivative: tuple[int, ...],
    cells: Array,
    /,
) -> Array:
    """certified basis derivatives of every candidate cell at its query point."""
    count, width = cells.shape
    repeated = jnp.broadcast_to(points[:, None, :], (count, width, points.shape[1]))
    basis = polynomial.basis_derivative(
        cells.reshape((-1,)), repeated.reshape((-1, 1, points.shape[1])), derivative
    )
    return basis.reshape((count, width, basis.shape[-1]))


def _stencil_route(
    basis: Array,
    stencil_cells: Array,
    stencil_valid: Array,
    factors: Array,
    cells: Array,
    accepted: Array,
    share: Array,
    constant: float,
    cell_count: int,
    /,
) -> GatherStencil:
    """Fixed linear route `u(x) = c u_c + sum_f B_f(x) sum_s F[c, f, s] (u_s - u_c)`."""
    valid_stencil = stencil_valid[cells]
    neighbor_weights = jnp.where(
        valid_stencil,
        contract("pkf,pkfs->pks", basis, factors[cells].astype(basis.dtype)),
        0.0,
    )
    cell_weight = constant - jnp.sum(neighbor_weights, axis=-1)
    weights = (
        jnp.concatenate((cell_weight[..., None], neighbor_weights), axis=-1)
        * share[..., None]
    )
    indices = jnp.concatenate((cells[..., None], stencil_cells[cells]), axis=-1)
    valid = jnp.concatenate(
        (accepted[..., None], valid_stencil & accepted[..., None]), axis=-1
    )
    width = indices.shape[1] * indices.shape[2]
    return GatherStencil(
        indices=indices.reshape((-1, width)),
        weights=weights.reshape((-1, width)),
        source_size=cell_count,
        valid=valid.reshape((-1, width)),
    )


def _plane_directions(
    centers: Array, points: Array, derivative: tuple[int, ...], cells: Array, /
) -> Array:
    """Directions `d` with `D^a u(x) = [a = 0] u_c + phi_c g_c . d` on a plane."""
    order = sum(derivative)
    shape = cells.shape + (points.shape[1],)
    if order == 0:
        return points[:, None, :] - centers[cells]
    if order == 1:
        axis = derivative.index(1)
        unit = jnp.zeros((points.shape[1],), dtype=points.dtype).at[axis].set(1.0)
        return jnp.broadcast_to(unit, shape)
    return jnp.zeros(shape, dtype=points.dtype)


def _face_side_cells(
    containing: np.ndarray,
    side: FieldTraceSide,
    owner_cells: np.ndarray,
    neighbor_cells: np.ndarray,
    /,
) -> np.ndarray:
    """Owner/neighbor cells of face sites from the mesh's oriented face table."""
    faces = {
        (min(int(left), int(right)), max(int(left), int(right))): (int(left), int(right))
        for left, right in zip(owner_cells, neighbor_cells, strict=True)
        if right >= 0
    }
    resolved = []
    for row in containing:
        present = sorted(int(cell) for cell in row if cell >= 0)
        if len(present) == 1 and side == "owner":
            resolved.append(present[0])
            continue
        oriented = faces.get(tuple(present)) if len(present) == 2 else None
        if oriented is None:
            raise ValueError(
                f"finite_volume_field.cell_ids must explicitly resolve {side!r} traces on ambiguous sites; got multiple faces or a missing neighbor."
            )
        match side:
            case "owner":
                resolved.append(oriented[0])
            case "neighbor":
                resolved.append(oriented[1])
            case _:
                raise ValueError(
                    f"finite_volume_field.trace_side must be owner or neighbor; got {side!r}."
                )
    return np.asarray(resolved, dtype=np.int32)


def _default_locator(
    discretization: _MeshDiscretization,
    policy: SimplicialLocationPolicy | None,
    /,
) -> PreparedSimplicialCellLocator:

    blocks = discretization.mesh.blocks
    if len(blocks) != 1 or blocks[0].cell_kind not in _SIMPLICES:
        raise ValueError(
            "prepare_finite_volume_field_reconstruction.locator must be explicit for nonsimplicial or mixed meshes; got None."
        )
    geometry = imported_fem__generic.FiniteElementPlan(
        discretization.mesh,
        imported_fem__generic.FiniteElementFieldSpec(
            "cell_average",
            imported_fem__reference.discontinuous_element(blocks[0].cell_kind, 0),
        ),
        coordinate_spec=_mesh_coordinate_geometry(discretization),
    ).prepare()
    cell_map = imported_fem__cell_map.PreparedFiniteElementCellMap(geometry, 0)
    # Candidate capacity must cover every BVH leaf box around a query point.
    policy_ = (
        SimplicialLocationPolicy(min(cell_map.cell_count, 64), 16, 1)
        if policy is None
        else policy
    )
    return PreparedSimplicialCellLocator(
        cell_map, geometry.default_runtime.coordinates, policy_
    )


def _checked_locator(
    discretization: _MeshDiscretization,
    locator: AbstractCellLocator,
    /,
) -> AbstractCellLocator:
    geometry = _mesh_coordinate_geometry(discretization)
    coordinates = discretization.vertices if geometry is None else geometry.coordinates
    if (
        locator.cell_map.topology_id != discretization.mesh.topology_id
        or locator.cell_map.cell_count != discretization.cell_count
        or (
            geometry is not None
            and locator.cell_map.geometry_layout_id != geometry.geometry_layout_id
        )
        or not np.array_equal(np.asarray(locator.coordinates), np.asarray(coordinates))
    ):
        raise ValueError(
            "prepare_finite_volume_field_reconstruction.locator must invert the prepared coordinate map; got different topology, layout, or coordinates."
        )
    return locator


def _require_mesh(
    prepared_on: FiniteVolumeDiscretization
    | AbstractUnstructuredFiniteVolumeDiscretization
    | TriangleFiniteVolumeDiscretization,
    discretization: FiniteVolumeDiscretization | _MeshDiscretization,
    /,
) -> None:
    match discretization:
        case (
            UnstructuredFiniteVolumeDiscretization()
            | TriangleFiniteVolumeDiscretization()
        ) if prepared_on.prepared_id == discretization.prepared_id:
            return
        case _:
            raise ValueError(
                "prepare_finite_volume_field_reconstruction.reconstruction must bind this FV discretization; got a structured endpoint or a different prepared ID."
            )


def _cell_reconstruction(
    reconstruction: CellReconstruction,
    discretization: FiniteVolumeDiscretization | _MeshDiscretization,
    /,
) -> _CellReconstructionPolicy:
    """Regularity, maximum derivative order, and coefficient linearity."""
    branchwise = branch_policy_contract(
        _WENO_BRANCH_POLICY, surfaces=(DerivativeSurface.PRIMAL_STATE,)
    ).conditions
    match reconstruction:
        case PiecewiseConstantReconstruction():
            return _CellReconstructionPolicy(
                reconstruction,
                DerivativeRegularity.piecewise_polynomial(continuity=-1, degree_bound=0),
                0,
                True,
            )
        case AbstractFaceReconstructionPlan():
            raise ValueError(
                "prepare_finite_volume_field_reconstruction.reconstruction must define values inside cells; got a directional face-trace plan."
            )
        case PreparedCellPolynomialReconstruction():
            _require_mesh(reconstruction.discretization, discretization)
            degree = reconstruction.basis.degree
            return _CellReconstructionPolicy(
                reconstruction,
                DerivativeRegularity.piecewise_polynomial(
                    continuity=-1, degree_bound=degree
                ),
                degree,
                True,
            )
        case PreparedUnstructuredWENOZReconstruction():
            _require_mesh(reconstruction.discretization, discretization)
            if reconstruction.limiter != "none":
                raise ValueError(
                    "prepare_finite_volume_field_reconstruction.reconstruction.limiter must be 'none' for a WENO-Z point view; got a trace limiter."
                )
            return _CellReconstructionPolicy(
                reconstruction,
                DerivativeRegularity.piecewise_smooth(
                    continuity=-1, conditions=branchwise
                ),
                reconstruction.optimal.basis.degree,
                False,
            )
        case TriangleKExactReconstructionPlan():
            _require_mesh(reconstruction.prepared.discretization, discretization)
            return _CellReconstructionPolicy(
                reconstruction,
                DerivativeRegularity.piecewise_polynomial(continuity=-1, degree_bound=2),
                2,
                True,
            )
        case TriangleMUSCLReconstructionPlan():
            _require_mesh(reconstruction.gradient.discretization, discretization)
            if reconstruction.limiter == "unlimited":
                return _CellReconstructionPolicy(
                    reconstruction,
                    DerivativeRegularity.piecewise_polynomial(
                        continuity=-1, degree_bound=1
                    ),
                    1,
                    True,
                )
            return _CellReconstructionPolicy(
                reconstruction,
                DerivativeRegularity.piecewise_smooth(
                    continuity=-1, conditions=branchwise
                ),
                1,
                False,
            )
        case _:
            assert_never(reconstruction)


@checked
def prepare_finite_volume_field_reconstruction(
    discretization: FiniteVolumeDiscretization | _MeshDiscretization,
    reconstruction: CellReconstruction,
    /,
    *,
    locator: AbstractCellLocator | None = None,
    location_policy: SimplicialLocationPolicy | None = None,
    location_tolerance: float = 1.0e-12,
    support_geometry: imported_geometry.CompiledGeometry | None = None,
    value_port: ValuePort | None = None,
    support_tolerance: float = 1.0e-9,
    layout: FiniteVolumeCoefficientLayout = "state",
) -> PreparedFieldReconstruction:
    """Prepare an evidenced coordinate reconstruction of one finite-volume field.

    `reconstruction` is the explicit cell reconstruction policy:
    `PiecewiseConstantReconstruction()` (cell averages; `C^-1` degree zero, no
    coordinate derivatives), a `PreparedCellPolynomialReconstruction` or
    `TriangleKExactReconstructionPlan` (k-certified, `C^-1` piecewise degree `k`,
    certified derivatives through order `k`), a `TriangleMUSCLReconstructionPlan`
    (WLSQ plane; linear when unlimited, otherwise scaled by the cell's limiter
    factor and nonlinear), or a `PreparedUnstructuredWENOZReconstruction` with
    `limiter="none"` (nonlinear in the averages, branchwise-differentiable,
    certified coordinate derivatives through the candidate degree). Polynomial and
    WENO-Z reconstructions exist on unstructured meshes and triangle
    reconstructions on triangular meshes; directional face-trace plans refuse.

    Structured grids locate points by index arithmetic on the axis edges
    (closed intervals within `location_tolerance` of each axis extent) and
    cover the grid box. Unstructured and triangular meshes need an inverse cell
    map: single-block triangle/tetrahedron meshes build a
    `PreparedSimplicialCellLocator` from `location_policy`, retaining the bound
    `cell_geometry` when present. Other meshes need an explicit `locator` over
    the same topology and represented coordinate source, not an affine corner
    replacement. Polyhedral Cartesian locators retain their full PLC
    support. An explicit support geometry must be covered by the mesh.
    Coefficients are cell averages in the discretization's `state_shape`;
    packet rebinding preserves that reconstruction/operator and full layout.
    `layout="scalar"` reconstructs a one-component structured field from its
    cell coordinates (`finite_volume_scalar_space`) as a scalar-valued field.
    """

    layout_ = parse(layout, FiniteVolumeCoefficientLayout, "layout")
    policy = _cell_reconstruction(reconstruction, discretization)
    reconstruction = policy.reconstruction
    regularity = policy.regularity
    maximum_order = policy.maximum_order
    linear = policy.coefficient_linear
    field_space_id = finite_volume_field_space_id(discretization)
    match discretization:
        case FiniteVolumeDiscretization():
            if locator is not None or location_policy is not None:
                raise ValueError(
                    "prepare_finite_volume_field_reconstruction.locator must be absent for structured grids; got a locator or location policy."
                )
            edges = structured_axis_edges(discretization)
            kernel = StructuredFiniteVolumeFieldReconstructionKernel(
                edges,
                location_tolerance=location_tolerance,
                field_space_id=field_space_id,
            )
            lower = np.asarray([values[0] for values in edges], dtype=np.float64)
            upper = np.asarray([values[-1] for values in edges], dtype=np.float64)
            support_id = canonical_fingerprint(
                {
                    "kind": "structured-finite-volume-support",
                    "support": discretization.support.support_id,
                    "box": array_tree_fingerprint(np.stack((lower, upper))),
                }
            )
            geometry = tensor_box_support_geometry(
                lower,
                upper,
                support_geometry,
                feature_id=f"structured-finite-volume-support:{support_id}",
                tolerance=support_tolerance,
            )
            dimension = lower.size
        case (
            UnstructuredFiniteVolumeDiscretization()
            | TriangleFiniteVolumeDiscretization()
        ):
            located = (
                _default_locator(discretization, location_policy)
                if locator is None
                else _checked_locator(discretization, locator)
            )
            kernel = UnstructuredFiniteVolumeFieldReconstructionKernel(
                located, reconstruction, discretization, field_space_id=field_space_id
            )
            match discretization:
                case UnstructuredFiniteVolumeDiscretization():
                    topology_id = discretization.topology_id
                    geometry_id = discretization.geometry_id
                case TriangleFiniteVolumeDiscretization():
                    topology_id = discretization.mesh.topology_id
                    geometry_id = discretization.mesh.geometry_id
                case _:
                    assert_never(discretization)
            support_id = canonical_fingerprint(
                {
                    "kind": "unstructured-finite-volume-support",
                    "topology": topology_id,
                    "geometry": geometry_id,
                }
            )
            tolerance = float(support_tolerance)
            if not isfinite(tolerance) or tolerance < 0.0:
                raise ValueError(
                    "prepare_finite_volume_field_reconstruction.support_tolerance must be finite and non-negative; got an invalid tolerance."
                )
            if support_geometry is None:
                geometry = simplicial_mesh_support_geometry(located, support_id)
            else:
                verify_mesh_support_geometry(
                    support_geometry, located, tolerance=tolerance
                )
                geometry = support_geometry
            dimension = mesh_cell_dimension(discretization)
        case _:
            assert_never(discretization)
    match layout_:
        case "state":
            components: tuple[int, ...] = (discretization.component_count,)
            coefficient_shape = discretization.state_shape
        case "scalar":
            match discretization:
                case FiniteVolumeDiscretization():
                    components = ()
                    coefficient_shape = finite_volume_scalar_space(discretization).shape
                case (
                    UnstructuredFiniteVolumeDiscretization()
                    | TriangleFiniteVolumeDiscretization()
                ):
                    raise ValueError(
                        "prepare_finite_volume_field_reconstruction.layout must use state for mesh discretizations; got scalar layout on an unstructured mesh."
                    )
                case _:
                    assert_never(discretization)
        case _:
            assert_never(layout_)
    port = (
        ValuePort(
            discretization.cell_space.name,
            event_shape=components,
            component_ids=discretization.component_names,
            representation="finite-volume-field",
            space_id=field_space_id,
        )
        if value_port is None
        else value_port
    )
    if port.event_shape != components:
        raise ValueError(
            "prepare_finite_volume_field_reconstruction.value_port must have the FV component event_shape; got a different shape."
        )
    return PreparedFieldReconstruction(
        kernel,
        support_geometry=geometry,
        value_port=port,
        regularity=regularity,
        trace_policy=FieldTracePolicy("cell-sided"),
        coefficient_shape=coefficient_shape,
        physical_dimension=dimension,
        maximum_derivative_order=maximum_order,
        field_space_id=field_space_id,
        support_id=support_id,
        coefficient_linear=linear,
    )


__all__ = [
    "StructuredFiniteVolumeFieldReconstructionKernel",
    "UnstructuredFiniteVolumeFieldReconstructionKernel",
    "prepare_finite_volume_field_reconstruction",
]
