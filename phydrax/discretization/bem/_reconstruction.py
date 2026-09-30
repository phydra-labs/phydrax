#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, FunctionLinearOperator
from ...typing import parse
from .._views import (
    _checked_queries,
    AbstractFieldReconstructionKernel,
    FieldQueryEvidence,
    FieldQueryStatus,
    FieldSideBinding,
    FieldSupportCoverage,
    FieldTraceSide,
)
from ._bc_dual import BuffaChristiansenDualSpace3D
from ._rwg import _rwg_tabulate, RWGSurfaceCurrentSpace3D


@final
class _SurfaceRoute(StrictModule):
    weights: Array
    edges: Array
    transform: Array | None
    coefficient_count: int = eqx.field(static=True)

    def apply(self, coefficients: Array, /) -> Array:
        refined = (
            coefficients if self.transform is None else self.transform @ coefficients
        )
        return contract("pfec,fe->pc", self.weights, refined[self.edges])

    def transpose(self, cotangent: Array, /) -> Array:
        local = contract("pfec,pc->fe", self.weights, cotangent)
        size = (
            self.coefficient_count if self.transform is None else self.transform.shape[0]
        )
        refined = jnp.zeros((size,), dtype=cotangent.dtype).at[self.edges].add(local)
        return refined if self.transform is None else self.transform.T @ refined


@final
class SurfaceFormReconstructionKernel(
    AbstractFieldReconstructionKernel, NonTrainableState
):
    """Exact RWG/BC reconstruction on its surface, not in the enclosed volume.

    Location tests the triangle plane and barycentric inequalities. A shared-edge
    Cartesian value requires an explicit side (or average), since flux conformity
    alone does not make that vector single-valued. Coordinate derivatives are
    tangential ambient directional derivatives of the affine triangle field.
    """

    space: RWGSurfaceCurrentSpace3D | BuffaChristiansenDualSpace3D
    rwg: RWGSurfaceCurrentSpace3D
    tolerance: float = eqx.field(static=True)
    maximum_location_pairs: int = eqx.field(static=True)
    identity: str = eqx.field(static=True)

    def __init__(
        self,
        space: RWGSurfaceCurrentSpace3D | BuffaChristiansenDualSpace3D,
        /,
        *,
        tolerance: float = 1.0e-10,
        maximum_location_pairs: int = 2_000_000,
    ) -> None:
        if not isinstance(
            space, (RWGSurfaceCurrentSpace3D, BuffaChristiansenDualSpace3D)
        ):
            raise TypeError("space must be an RWG or Buffa-Christiansen surface space.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if maximum_location_pairs <= 0:
            raise ValueError("maximum_location_pairs must be positive.")
        self.space = space
        self.rwg = (
            space
            if isinstance(space, RWGSurfaceCurrentSpace3D)
            else space.barycentric_rwg
        )
        self.tolerance = tolerance
        self.maximum_location_pairs = maximum_location_pairs
        self.identity = canonical_fingerprint(
            {
                "kind": "surface-form-reconstruction",
                "space": space.space_id,
                "tolerance": tolerance,
                "capacity": maximum_location_pairs,
            }
        )

    @property
    def kernel_id(self) -> str:
        return self.identity

    @property
    def cell_count(self) -> int:
        return self.rwg.surface.face_count

    @property
    def support_coverage(self) -> FieldSupportCoverage:
        return "complete"

    @property
    def value_port(self) -> ValuePort:
        return ValuePort(
            "surface-current"
            if isinstance(self.space, RWGSurfaceCurrentSpace3D)
            else "surface-current-dual",
            event_shape=(3,),
            component_ids=("x", "y", "z"),
            representation="embedded-surface-flux",
            space_id=self.space.space_id,
            form=self.space.value_spec,
        )

    def _containing(self, points: Array, /) -> tuple[Array, Array]:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Surface query points must have shape (points, 3).")
        if points.shape[0] * self.cell_count > self.maximum_location_pairs:
            raise ValueError("Surface query exceeds maximum_location_pairs.")
        vertices = self.rwg.surface.vertices[self.rwg.surface.triangles]
        jacobian = jnp.swapaxes(vertices[:, 1:] - vertices[:, :1], -1, -2)
        gram = jnp.swapaxes(jacobian, -1, -2) @ jacobian
        inverse_map = jnp.linalg.solve(gram, jnp.swapaxes(jacobian, -1, -2))
        offsets = points[:, None] - vertices[None, :, 0]
        reference = contract("fij,pfj->pfi", inverse_map, offsets)
        projected = contract("fij,pfj->pfi", jacobian, reference)
        scale = jnp.maximum(1.0, jnp.max(self.rwg.surface.edge_lengths))
        inside_plane = (
            jnp.linalg.norm(offsets - projected, axis=-1) <= self.tolerance * scale
        )
        barycentric = jnp.concatenate(
            (1.0 - jnp.sum(reference, axis=-1, keepdims=True), reference), axis=-1
        )
        contained = inside_plane & jnp.all(barycentric >= -self.tolerance, axis=-1)
        return contained, jnp.linalg.cond(gram)

    def locate(
        self, points: Array, derivative: tuple[int, ...], side: FieldSideBinding | None, /
    ) -> tuple[_SurfaceRoute, FieldQueryEvidence]:
        if (
            len(derivative) != 3
            or any(index < 0 for index in derivative)
            or sum(derivative) > 1
        ):
            raise ValueError(
                "Surface reconstruction admits value and first tangential derivatives."
            )
        if side is not None and side.reconstruction_id != self.kernel_id:
            raise ValueError("The surface side belongs to another reconstruction.")
        contained, conditioning = self._containing(points)
        counts = jnp.sum(contained, axis=1, dtype=jnp.int32)
        selected = contained
        average = side is not None and side.side == "average"
        if side is not None and not average:
            if side.cell_mask is None or side.cell_mask.shape != (self.cell_count,):
                raise ValueError("A surface side must bind the exact surface cell mask.")
            selected = selected & side.cell_mask[None]
            if side.site_cells.shape == points.shape[:1]:
                explicit = side.site_cells[:, None]
                selected = selected & (
                    (explicit < 0) | (jnp.arange(self.cell_count)[None] == explicit)
                )
        selected_count = jnp.sum(selected, axis=1, dtype=jnp.int32)
        status = jnp.where(
            counts == 0,
            int(FieldQueryStatus.OUTSIDE_SUPPORT),
            int(FieldQueryStatus.VALID),
        )
        if side is None:
            status = jnp.where(counts > 1, int(FieldQueryStatus.SIDE_REQUIRED), status)
        elif not average:
            status = jnp.where(
                (counts > 0) & (selected_count != 1),
                int(FieldQueryStatus.SIDE_UNRESOLVED),
                status,
            )
        status = jnp.where(
            jnp.all(jnp.isfinite(points), axis=1), status, int(FieldQueryStatus.NONFINITE)
        )
        face_points = jnp.broadcast_to(points[None], (self.cell_count, *points.shape))

        def basis_at(sites: Array) -> Array:
            return _rwg_tabulate(self.rwg.surface, sites, self.rwg.element)[0]

        if sum(derivative):
            direction = jnp.asarray(derivative, dtype=points.dtype)
            _, basis = jax.jvp(
                basis_at,
                (face_points,),
                (jnp.broadcast_to(direction, face_points.shape),),
            )
        else:
            basis = basis_at(face_points)
        weights = jnp.swapaxes(basis, 0, 1)
        factors = selected.astype(weights.dtype) / jnp.maximum(selected_count[:, None], 1)
        weights = weights * factors[:, :, None, None]
        weights = jnp.where(
            (status == int(FieldQueryStatus.VALID))[:, None, None, None], weights, 0.0
        )
        weights = weights.astype(self.space.vector_space.dtype)
        transform = (
            None
            if isinstance(self.space, RWGSurfaceCurrentSpace3D)
            else self.space.barycentric_transform.astype(self.space.vector_space.dtype)
        )
        route = _SurfaceRoute(
            weights, self.rwg.surface.face_edges, transform, self.space.size
        )
        worst_condition = jnp.max(jnp.where(contained, conditioning[None], 0.0), axis=1)
        return route, FieldQueryEvidence(
            status, worst_condition, counts, kernel_id=self.kernel_id
        )

    def apply(self, route: _SurfaceRoute, coefficients: Array, /) -> Array:
        return route.apply(self.space.validate(coefficients))

    def transpose(self, route: _SurfaceRoute, cotangent: Array, /) -> Array:
        return route.transpose(cotangent)

    def bind_side(
        self, sites: np.ndarray, side: FieldTraceSide, cell_ids: np.ndarray | None, /
    ) -> tuple[np.ndarray | None, np.ndarray]:
        side = parse(side, FieldTraceSide, "side")
        contained, _ = self._containing(jnp.asarray(sites))
        host = np.asarray(contained)
        if np.any(~np.any(host, axis=1)):
            raise ValueError(
                "Surface side sites must lie in the actual triangle support."
            )
        if side == "average":
            return None, np.full((sites.shape[0],), -1, dtype=np.int32)
        if cell_ids is None:
            candidates = [np.flatnonzero(row) for row in host]
            offset = 0 if side == "owner" else 1
            if any(indices.size <= offset for indices in candidates):
                raise ValueError(
                    "Neighbor trace requires a second containing surface triangle."
                )
            cells = np.array([indices[offset] for indices in candidates], dtype=np.int32)
        else:
            cells = np.asarray(cell_ids, dtype=np.int32)
            if (
                cells.shape != (sites.shape[0],)
                or np.any(cells < 0)
                or np.any(cells >= self.cell_count)
            ):
                raise ValueError(
                    "Surface cell_ids must select one in-range triangle per site."
                )
            if np.any(~host[np.arange(sites.shape[0]), cells]):
                raise ValueError(
                    "Selected triangles do not contain their surface trace sites."
                )
        mask = np.zeros((self.cell_count,), dtype=np.bool_)
        mask[cells] = True
        return mask, cells


@final
class PreparedBoundaryFormQuery(StrictModule, NonTrainableState):
    """Reusable exact boundary reconstruction and its native algebraic transpose."""

    kernel: SurfaceFormReconstructionKernel
    route: _SurfaceRoute
    evidence: FieldQueryEvidence
    operator: FunctionLinearOperator
    value_port: ValuePort
    query_id: str = eqx.field(static=True)

    def apply(self, coefficients: ArrayLike, /) -> Array:
        return self.route.apply(self.kernel.space.validate(coefficients))

    def transpose(self, covectors: ArrayLike, /) -> Array:
        return self.kernel.space.vector_space.validate(
            self.operator.transpose_mv(covectors)
        )


def prepare_boundary_form_query(
    space: RWGSurfaceCurrentSpace3D | BuffaChristiansenDualSpace3D,
    points: ArrayLike,
    /,
    *,
    derivative: tuple[int, ...] = (0, 0, 0),
    side: FieldTraceSide | None = None,
    cell_ids: ArrayLike | None = None,
    tolerance: float = 1.0e-10,
    maximum_location_pairs: int = 2_000_000,
) -> PreparedBoundaryFormQuery:
    """Locate once on actual boundary triangles, refusing invalid/off-surface sites."""
    kernel = SurfaceFormReconstructionKernel(
        space, tolerance=tolerance, maximum_location_pairs=maximum_location_pairs
    )
    sites = np.asarray(points, dtype=np.float64)
    binding = None
    if side is not None:
        side = parse(side, FieldTraceSide, "side")
        mask, cells = kernel.bind_side(
            sites,
            side,
            None if cell_ids is None else np.asarray(cell_ids, dtype=np.int32),
        )
        binding = FieldSideBinding(
            side, mask, sites, cells, reconstruction_id=kernel.kernel_id
        )
    elif cell_ids is not None:
        raise ValueError("cell_ids requires an explicit trace side.")
    route, evidence = kernel.locate(jnp.asarray(sites), derivative, binding)
    _checked_queries(jnp.asarray(sites), evidence)
    identity = canonical_fingerprint(
        {
            "kind": "prepared-boundary-form-query",
            "kernel": kernel.kernel_id,
            "points": array_tree_fingerprint(sites),
            "derivative": derivative,
            "side": None if binding is None else binding.binding_id,
        }
    )
    operator = FunctionLinearOperator(
        route.apply,
        source=space.vector_space,
        target=ArraySpace(
            (sites.shape[0], 3),
            dtype=space.vector_space.dtype,
            space_id=f"{identity}:values",
        ),
        transpose_action=route.transpose,
        operator_id=identity,
    )
    return PreparedBoundaryFormQuery(
        kernel, route, evidence, operator, kernel.value_port, identity
    )


__all__ = [
    "PreparedBoundaryFormQuery",
    "SurfaceFormReconstructionKernel",
    "prepare_boundary_form_query",
]
