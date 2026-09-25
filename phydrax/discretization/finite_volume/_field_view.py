#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-volume cell reconstructions as evidenced coordinate field views.

Structured grids locate points by exact index arithmetic on the prepared axis
edges; unstructured meshes locate points with an explicit inverse cell map
(`AbstractCellLocator`). Every finite-volume reconstruction is discontinuous
across faces, so the trace policy is cell-sided: a point on a face lies in
several cells and needs a bound trace side. No kernel searches for a nearest
cell; points outside every cell are `OUTSIDE_SUPPORT`.
"""

from __future__ import annotations

from math import isfinite, prod
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax.ein import contract

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
from ._cell_polynomial import PreparedCellPolynomialReconstruction
from ._reconstruction import (
    AbstractFaceReconstructionPlan,
    PiecewiseConstantReconstruction,
)
from ._structured import FiniteVolumeDiscretization
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_weno import PreparedUnstructuredWENOZReconstruction


_SIMPLICES = ("triangle", "tetrahedron")
# WENO-Z weights branch on max/min smoothness indicators; derivatives with
# respect to the cell averages are those of the executed branch.
_WENO_BRANCH_POLICY = BranchDifferentiationPolicy.BRANCHWISE

UnstructuredCellReconstruction = (
    PiecewiseConstantReconstruction
    | PreparedCellPolynomialReconstruction
    | PreparedUnstructuredWENOZReconstruction
)


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
            "Piecewise-constant cell averages have no coordinate derivatives."
        )


@final
class StructuredFiniteVolumeFieldReconstructionKernel(
    AbstractFieldReconstructionKernel, NonTrainableState
):
    """Piecewise-constant cell averages on a structured tensor grid.

    A point is located by index arithmetic on the prepared axis edges: along
    each axis it lies in the cells whose closed intervals contain it (within an
    absolute `tolerance` per axis), so interior faces belong to both adjacent
    cells. Face orientation is the canonical positive-axis one: the owner of an
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
    ):
        axes = tuple(np.asarray(values, dtype=np.float64) for values in edges)
        if not axes or any(
            values.ndim != 1
            or values.size < 2
            or not np.all(np.isfinite(values))
            or np.any(np.diff(values) <= 0.0)
            for values in axes
        ):
            raise ValueError("Axis edges must be finite, increasing, and span a cell.")
        tolerance = float(location_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("location_tolerance must be finite and non-negative.")
        absolute = np.asarray(
            [tolerance * (values[-1] - values[0]) for values in axes], dtype=np.float64
        )
        if any(
            2.0 * width >= np.min(np.diff(values))
            for width, values in zip(absolute, axes, strict=True)
        ):
            raise ValueError(
                "location_tolerance must be below half the smallest cell width."
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
            raise ValueError("Every trace site must lie in the finite-volume grid.")
        containing = np.where(valid, cells, -1)
        if side == "average":
            if cell_ids is not None:
                raise ValueError("Average traces combine every containing cell.")
            return None, np.full(sites.shape[0], -1, dtype=np.int32)
        if cell_ids is None:
            site_cells = _oriented_side_cells(containing, side)
        else:
            if np.any(np.all(containing != cell_ids[:, None], axis=1)):
                raise ValueError("cell_ids must name a cell containing each trace site.")
            site_cells = cell_ids.astype(np.int32)
        mask = np.zeros((self.cell_count,), dtype=np.bool_)
        mask[site_cells] = True
        if np.any(np.sum(valid & mask[cells], axis=1) != 1):
            raise ValueError(
                "The side cells do not resolve every trace site to exactly its "
                "declared cell; the side region is ambiguous."
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
            raise ValueError(f"Unknown trace side {side!r}.")
    if not np.all(resolved):
        raise ValueError(
            f"{side!r} traces at sites on several faces (or without a neighbor "
            "cell) require explicit cell_ids."
        )
    return cells.astype(np.int32)


@final
class _WENORoute(StrictModule):
    cells: Array
    basis: Array
    share: Array
    constant: bool = eqx.field(static=True)


@final
class UnstructuredFiniteVolumeFieldReconstructionKernel(
    AbstractFieldReconstructionKernel, NonTrainableState
):
    """Located cell reconstruction on an unstructured finite-volume mesh.

    Points are located by an explicit `AbstractCellLocator` inverse cell map.
    Inside a cell the value is the prepared reconstruction: the cell average
    (`PiecewiseConstantReconstruction`), the conservative k-exact polynomial
    (`PreparedCellPolynomialReconstruction`, linear in the averages), or the
    WENO-Z blended polynomial (`PreparedUnstructuredWENOZReconstruction`,
    nonlinear in the averages). Coordinate derivatives differentiate the cell
    polynomial exactly.
    """

    locator: AbstractCellLocator
    reconstruction: UnstructuredCellReconstruction
    owner_cells: Array
    neighbor_cells: Array
    _cell_count: int = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        locator: AbstractCellLocator,
        reconstruction: UnstructuredCellReconstruction,
        discretization: UnstructuredFiniteVolumeDiscretization,
        /,
        *,
        field_space_id: str,
    ):
        if not isinstance(locator, AbstractCellLocator):
            raise TypeError("locator must be an AbstractCellLocator.")
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError(
                "discretization must be UnstructuredFiniteVolumeDiscretization."
            )
        match reconstruction:
            case PiecewiseConstantReconstruction():
                reconstruction_id = reconstruction.plan_id
            case (
                PreparedCellPolynomialReconstruction()
                | PreparedUnstructuredWENOZReconstruction()
            ):
                reconstruction_id = reconstruction.prepared_id
            case _:
                raise TypeError(
                    "reconstruction must be PiecewiseConstantReconstruction, "
                    "PreparedCellPolynomialReconstruction, or "
                    "PreparedUnstructuredWENOZReconstruction."
                )
        self.locator = locator
        self.reconstruction = reconstruction
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

    def locate(
        self,
        points: Array,
        derivative: tuple[int, ...],
        side: FieldSideBinding | None,
        /,
    ) -> tuple[GatherStencil | _WENORoute, FieldQueryEvidence]:
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
        match self.reconstruction:
            case PiecewiseConstantReconstruction():
                _require_value_order(derivative)
                route = GatherStencil(
                    indices=cells,
                    weights=share,
                    source_size=self._cell_count,
                    valid=accepted,
                )
            case PreparedCellPolynomialReconstruction():
                route = self._polynomial_route(points, derivative, cells, accepted, share)
            case PreparedUnstructuredWENOZReconstruction():
                route = _WENORoute(
                    cells,
                    _candidate_basis(
                        self.reconstruction.optimal, points, derivative, cells
                    ),
                    share,
                    not any(derivative),
                )
            case _:
                raise TypeError("Unknown unstructured finite-volume reconstruction.")
        return route, evidence

    def _polynomial_route(
        self,
        points: Array,
        derivative: tuple[int, ...],
        cells: Array,
        accepted: Array,
        share: Array,
        /,
    ) -> GatherStencil:
        # u(x) = u_c + sum_f B_f(x) sum_s F[c, f, s] (u_s - u_c): a fixed linear
        # route over the cell and its prepared stencil.
        polynomial = self.reconstruction
        basis = _candidate_basis(polynomial, points, derivative, cells)
        stencil_valid = polynomial.stencil_valid[cells]
        neighbor_weights = jnp.where(
            stencil_valid,
            contract(
                "pkf,pkfs->pks", basis, polynomial.factors[cells].astype(basis.dtype)
            ),
            0.0,
        )
        cell_weight = float(not any(derivative)) - jnp.sum(neighbor_weights, axis=-1)
        weights = (
            jnp.concatenate((cell_weight[..., None], neighbor_weights), axis=-1)
            * share[..., None]
        )
        indices = jnp.concatenate(
            (cells[..., None], polynomial.stencil_cells[cells]), axis=-1
        )
        valid = jnp.concatenate(
            (accepted[..., None], stencil_valid & accepted[..., None]), axis=-1
        )
        width = indices.shape[1] * indices.shape[2]
        return GatherStencil(
            indices=indices.reshape((-1, width)),
            weights=weights.reshape((-1, width)),
            source_size=self._cell_count,
            valid=valid.reshape((-1, width)),
        )

    def apply(self, route: GatherStencil | _WENORoute, coefficients: Array, /) -> Array:
        match self.reconstruction:
            case (
                PiecewiseConstantReconstruction() | PreparedCellPolynomialReconstruction()
            ):
                return linear_apply(route.relation, route.weights, coefficients)
            case PreparedUnstructuredWENOZReconstruction():
                modal = self.reconstruction.coefficients(coefficients)
                values = contract("pkf,pk...f->pk...", route.basis, modal[route.cells])
                if route.constant:
                    values = values + coefficients[route.cells]
                return contract("pk,pk...->p...", route.share, values)
            case _:
                raise TypeError("Unknown unstructured finite-volume reconstruction.")

    def transpose(self, route: GatherStencil | _WENORoute, cotangent: Array, /) -> Array:
        match self.reconstruction:
            case (
                PiecewiseConstantReconstruction() | PreparedCellPolynomialReconstruction()
            ):
                return linear_transpose_apply(route.relation, route.weights, cotangent)
            case PreparedUnstructuredWENOZReconstruction():
                raise ValueError(
                    "WENO-Z reconstructions are nonlinear in the cell averages and "
                    "have no algebraic transpose."
                )
            case _:
                raise TypeError("Unknown unstructured finite-volume reconstruction.")

    def bind_side(
        self,
        sites: np.ndarray,
        side: FieldTraceSide,
        cell_ids: np.ndarray | None,
        /,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        location = self.locator.locate(jnp.asarray(sites))
        if np.any(np.asarray(location.status) != int(CellLocationStatus.LOCATED)):
            raise ValueError("Every trace site must lie in the finite-volume mesh.")
        containing = np.asarray(location.candidate_cells)
        if side == "average":
            if cell_ids is not None:
                raise ValueError("Average traces combine every containing cell.")
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
                raise ValueError("cell_ids must name a cell containing each trace site.")
            site_cells = cell_ids.astype(np.int32)
        mask = np.zeros((self._cell_count,), dtype=np.bool_)
        mask[site_cells] = True
        masked = self.locator.locate(jnp.asarray(sites), cell_mask=jnp.asarray(mask))
        if np.any(np.asarray(masked.candidate_count) != 1) or np.any(
            np.asarray(masked.cell_ids) != site_cells
        ):
            raise ValueError(
                "The side cells do not resolve every trace site to exactly its "
                "declared cell; the side region is ambiguous."
            )
        return mask, site_cells


def _candidate_basis(
    polynomial: PreparedCellPolynomialReconstruction,
    points: Array,
    derivative: tuple[int, ...],
    cells: Array,
    /,
) -> Array:
    """Exact basis derivatives of every candidate cell at its query point."""
    count, width = cells.shape
    repeated = jnp.broadcast_to(points[:, None, :], (count, width, points.shape[1]))
    basis = polynomial.basis_derivative(
        cells.reshape((-1,)), repeated.reshape((-1, 1, points.shape[1])), derivative
    )
    return basis.reshape((count, width, basis.shape[-1]))


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
                f"{side!r} traces at sites that are not on exactly one face (or "
                "without a neighbor cell) require explicit cell_ids."
            )
        match side:
            case "owner":
                resolved.append(oriented[0])
            case "neighbor":
                resolved.append(oriented[1])
            case _:
                raise ValueError(f"Unknown trace side {side!r}.")
    return np.asarray(resolved, dtype=np.int32)


def _structured_edges(
    discretization: FiniteVolumeDiscretization, /
) -> tuple[np.ndarray, ...]:
    edges = []
    for axis in discretization.grid.structured_axes:
        widths = np.asarray(axis.interval_widths, dtype=np.float64)
        lower = float(np.asarray(axis.bounds)[0])
        edges.append(lower + np.concatenate(([0.0], np.cumsum(widths))))
    return tuple(edges)


def _default_locator(
    discretization: UnstructuredFiniteVolumeDiscretization,
    policy: SimplicialLocationPolicy | None,
    /,
) -> PreparedSimplicialCellLocator:
    from ..fem._cell_map import PreparedFiniteElementCellMap
    from ..fem._generic import FiniteElementFieldSpec, FiniteElementPlan
    from ..fem._reference import discontinuous_element

    blocks = discretization.mesh.blocks
    if len(blocks) != 1 or blocks[0].cell_kind not in _SIMPLICES:
        raise ValueError(
            "Arbitrary-point location on non-simplicial or mixed finite-volume meshes "
            "requires an explicit AbstractCellLocator inverse provider."
        )
    geometry = FiniteElementPlan(
        discretization.mesh,
        FiniteElementFieldSpec(
            "cell_average", discontinuous_element(blocks[0].cell_kind, 0)
        ),
    ).prepare()
    cell_map = PreparedFiniteElementCellMap(geometry, 0)
    # Candidate capacity must cover every BVH leaf box around a query point.
    policy_ = (
        SimplicialLocationPolicy(min(cell_map.cell_count, 64), 16, 1)
        if policy is None
        else policy
    )
    if not isinstance(policy_, SimplicialLocationPolicy):
        raise TypeError("location_policy must be a SimplicialLocationPolicy or None.")
    return PreparedSimplicialCellLocator(
        cell_map, geometry.default_runtime.coordinates, policy_
    )


def _checked_locator(
    discretization: UnstructuredFiniteVolumeDiscretization,
    locator: AbstractCellLocator,
    /,
) -> AbstractCellLocator:
    if not isinstance(locator, AbstractCellLocator):
        raise TypeError("locator must be an AbstractCellLocator or None.")
    if (
        len(discretization.mesh.blocks) != 1
        or locator.cell_map.topology_id != discretization.mesh.topology_id
        or locator.cell_map.cell_count != discretization.cell_count
        or not np.array_equal(
            np.asarray(locator.coordinates), np.asarray(discretization.vertices)
        )
    ):
        raise ValueError("The locator does not invert this finite-volume mesh.")
    return locator


def _require_mesh(
    reconstruction: PreparedCellPolynomialReconstruction
    | PreparedUnstructuredWENOZReconstruction,
    discretization: FiniteVolumeDiscretization | UnstructuredFiniteVolumeDiscretization,
    /,
) -> None:
    if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization) or (
        reconstruction.discretization.prepared_id != discretization.prepared_id
    ):
        raise ValueError(
            "The cell reconstruction was prepared on a different finite-volume "
            "discretization."
        )


def _cell_reconstruction(
    reconstruction: Any,
    discretization: FiniteVolumeDiscretization | UnstructuredFiniteVolumeDiscretization,
    /,
) -> tuple[DerivativeRegularity, int, bool]:
    """Regularity, maximum derivative order, and coefficient linearity."""
    match reconstruction:
        case PiecewiseConstantReconstruction():
            return (
                DerivativeRegularity.piecewise_polynomial(continuity=-1, degree_bound=0),
                0,
                True,
            )
        case AbstractFaceReconstructionPlan():
            raise ValueError(
                "Directional face-trace plans (MUSCL, WENO, ...) reconstruct traces at "
                "faces axis by axis and define no point reconstruction inside cells; "
                "use PiecewiseConstantReconstruction, or a cell polynomial or WENO-Z "
                "reconstruction on an unstructured mesh."
            )
        case PreparedCellPolynomialReconstruction():
            _require_mesh(reconstruction, discretization)
            degree = reconstruction.basis.degree
            return (
                DerivativeRegularity.piecewise_polynomial(
                    continuity=-1, degree_bound=degree
                ),
                degree,
                True,
            )
        case PreparedUnstructuredWENOZReconstruction():
            _require_mesh(reconstruction, discretization)
            if reconstruction.limiter != "none":
                raise ValueError(
                    "A WENO-Z trace limiter scales each trace by its own evaluation "
                    "point set and defines no point reconstruction; prepare the view "
                    "with limiter='none'."
                )
            conditions = branch_policy_contract(
                _WENO_BRANCH_POLICY, surfaces=(DerivativeSurface.PRIMAL_STATE,)
            ).conditions
            return (
                DerivativeRegularity.piecewise_smooth(
                    continuity=-1, conditions=conditions
                ),
                reconstruction.optimal.basis.degree,
                False,
            )
        case _:
            raise TypeError(
                "reconstruction must be PiecewiseConstantReconstruction, "
                "PreparedCellPolynomialReconstruction, or "
                "PreparedUnstructuredWENOZReconstruction."
            )


def prepare_finite_volume_field_reconstruction(
    discretization: FiniteVolumeDiscretization | UnstructuredFiniteVolumeDiscretization,
    reconstruction: Any,
    /,
    *,
    locator: AbstractCellLocator | None = None,
    location_policy: SimplicialLocationPolicy | None = None,
    location_tolerance: float = 1.0e-12,
    support_geometry: Any = None,
    value_port: ValuePort | None = None,
    support_tolerance: float = 1.0e-9,
) -> PreparedFieldReconstruction:
    """Prepare an evidenced coordinate reconstruction of one finite-volume field.

    `reconstruction` is the explicit cell reconstruction policy:
    `PiecewiseConstantReconstruction()` (cell averages; `C^-1` degree zero, no
    coordinate derivatives), a `PreparedCellPolynomialReconstruction` (k-exact,
    `C^-1` piecewise degree `k`, exact derivatives through order `k`), or a
    `PreparedUnstructuredWENOZReconstruction` with `limiter="none"` (nonlinear
    in the averages, branchwise-differentiable, exact coordinate derivatives
    through the candidate degree). Polynomial and WENO-Z reconstructions exist
    on unstructured meshes; directional face-trace plans refuse.

    Structured grids locate points by index arithmetic on the axis edges
    (closed intervals within `location_tolerance` of each axis extent) and
    cover the grid box. Unstructured meshes need an inverse cell map:
    single-block triangle/tetrahedron meshes build a
    `PreparedSimplicialCellLocator` from `location_policy`; other meshes need an
    explicit `locator` over the same topology and vertices. The support is the
    mesh region (an explicit geometry must be covered by the mesh). Coefficients
    are cell averages with the discretization's `state_shape`.
    """
    from ...geometry import CompiledGeometry

    if not isinstance(
        discretization,
        (FiniteVolumeDiscretization, UnstructuredFiniteVolumeDiscretization),
    ):
        raise TypeError(
            "discretization must be a FiniteVolumeDiscretization or an "
            "UnstructuredFiniteVolumeDiscretization."
        )
    regularity, maximum_order, linear = _cell_reconstruction(
        reconstruction, discretization
    )
    field_space_id = canonical_fingerprint(
        {
            "kind": "finite-volume-field-space",
            "discretization": discretization.prepared_id,
            "field": discretization.cell_space.name,
            "components": list(discretization.component_names),
        }
    )
    match discretization:
        case FiniteVolumeDiscretization():
            if locator is not None or location_policy is not None:
                raise ValueError(
                    "Structured grids locate points by index arithmetic; locators "
                    "apply to unstructured meshes."
                )
            edges = _structured_edges(discretization)
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
        case UnstructuredFiniteVolumeDiscretization():
            located = (
                _default_locator(discretization, location_policy)
                if locator is None
                else _checked_locator(discretization, locator)
            )
            kernel = UnstructuredFiniteVolumeFieldReconstructionKernel(
                located, reconstruction, discretization, field_space_id=field_space_id
            )
            support_id = canonical_fingerprint(
                {
                    "kind": "unstructured-finite-volume-support",
                    "topology": discretization.topology_id,
                    "geometry": discretization.geometry_id,
                }
            )
            tolerance = float(support_tolerance)
            if not isfinite(tolerance) or tolerance < 0.0:
                raise ValueError("support_tolerance must be finite and non-negative.")
            if support_geometry is None:
                geometry = simplicial_mesh_support_geometry(located, support_id)
            elif isinstance(support_geometry, CompiledGeometry):
                verify_mesh_support_geometry(
                    support_geometry, located, tolerance=tolerance
                )
                geometry = support_geometry
            else:
                raise TypeError("support_geometry must be a CompiledGeometry or None.")
            dimension = discretization.cell_dimension
        case _:
            raise TypeError(
                "discretization must be a FiniteVolumeDiscretization or an "
                "UnstructuredFiniteVolumeDiscretization."
            )
    components = (discretization.component_count,)
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
    if not isinstance(port, ValuePort):
        raise TypeError("value_port must be a ValuePort or None.")
    if port.event_shape != components:
        raise ValueError(
            "value_port event_shape must be the finite-volume component shape."
        )
    return PreparedFieldReconstruction(
        kernel,
        support_geometry=geometry,
        value_port=port,
        regularity=regularity,
        trace_policy=FieldTracePolicy("cell-sided"),
        coefficient_shape=discretization.state_shape,
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
