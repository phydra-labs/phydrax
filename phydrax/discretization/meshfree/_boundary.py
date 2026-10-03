#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Side-labeled point support, ghost layers, physical boundary samples, and point SBP forms.

Side identity is always declared: a point belongs to the material sides listed
in an explicit membership table, and a one-sided stencil uses only actual points
of its side. Physical boundary samples come from oriented geometry charts; raw
point coordinates never fabricate facets, normals, measures, or side identity.
A ghost layer is an explicitly selected PDE+BC boundary route with its own
declared extension law and evidence; it is never inserted implicitly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import assert_never, final, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...geometry._atlas import BoundaryAtlas
from ...geometry._cubature import CubatureAtlas
from ...geometry.surface._contracts import SurfaceInterface
from ...linalg import (
    ArraySpace,
    assemble_sparse,
    FailurePolicy,
    IdentityLinearOperator,
    KroneckerLinearOperator,
    OperatorProperties,
)
from ...sparse import EdgeRelation, RowRelation, SparseCoordinateOperator
from ...typing import checked, ConvertibleToArray
from .._integration_domain import IntegrationDomain
from .._point_cloud import PreparedPointCloudDiscretization
from .._reference_cell import FacetShape
from .._side_actions import (
    FacetTraceRule,
    PreparedTraceAction,
    SideActionDescriptor,
    SideGatherRoute,
)
from ..finite_difference._sbp import PreparedSBPOperator, SBPFamily, SBPGridNorm
from ..spatial import MortonAddressPlan
from ._neighbors import MeshfreeNeighborhoodPlan
from ._stencils import (
    LocalStencilPolicy,
    LocalStencilReport,
    MeshfreeFunctional,
    prepare_local_stencils,
)


if TYPE_CHECKING:
    from ...optim import ConvexProgramResult, ConvexSolvePolicy
    from .._point_cloud_pde import PointBoundaryPlan


type MultiIndex = tuple[int, ...]


def derivative_multi_indices(dimension: int, /, *, value: bool) -> tuple[MultiIndex, ...]:
    """Value (optional), first, pure second, then mixed second multi-indices."""
    indices: list[MultiIndex] = [(0,) * dimension] if value else []
    for order in (1, 2):
        indices.extend(
            tuple(order if d == axis else 0 for d in range(dimension))
            for axis in range(dimension)
        )
    indices.extend(
        tuple(int(d in (first, second)) for d in range(dimension))
        for first in range(dimension)
        for second in range(first + 1, dimension)
    )
    return tuple(indices)


@final
class PointDerivativeFamily(StrictModule, NonTrainableState):
    """Fixed-width derivative rows on an equation row set.

    Rows without an admitted stencil carry only invalid routes; applying the
    family there yields zero rather than a fabricated derivative.
    """

    relation: RowRelation
    weights: tuple[Array, ...]
    row_active: Array
    multi_indices: tuple[MultiIndex, ...] = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation: RowRelation,
        weights: Sequence[ArrayLike],
        row_active: ArrayLike,
        multi_indices: Sequence[MultiIndex],
        /,
        *,
        family_id: str,
    ) -> None:
        if not isinstance(relation, RowRelation) or relation.case_shape:
            raise TypeError("relation must be an uncased RowRelation.")
        indices = tuple(tuple(int(v) for v in index) for index in multi_indices)
        values = tuple(jnp.asarray(w, dtype=jnp.float64) for w in weights)
        active = jnp.asarray(row_active, dtype=jnp.bool_)
        if len(values) != len(indices) or len(set(indices)) != len(indices):
            raise ValueError("Derivative weights need unique multi-indices.")
        if any(w.shape != relation.route_shape for w in values):
            raise ValueError("Derivative weights must match the relation routes.")
        if active.shape != relation.target_shape:
            raise ValueError("row_active must have one entry per relation row.")
        self.relation = relation
        self.weights = values
        self.row_active = active
        self.multi_indices = indices
        self.family_id = canonical_identifier(family_id, "family_id")

    @property
    def row_count(self) -> int:
        return self.relation.target_shape[0]

    @property
    def source_count(self) -> int:
        return self.relation.source_size

    def weights_for(self, multi_index: MultiIndex, /) -> Array:
        for index, weights in zip(self.multi_indices, self.weights, strict=True):
            if index == tuple(multi_index):
                return weights
        raise ValueError(f"Derivative {multi_index} is not prepared in this family.")

    def apply(self, values: ArrayLike, multi_index: MultiIndex, /) -> Array:
        value = jnp.asarray(values)
        if value.shape[:1] != (self.source_count,):
            raise ValueError("Family values must begin with the source count.")
        weights = self.weights_for(multi_index)
        payload = value.shape[1:]
        routes = (self.relation.valid.shape) + (1,) * len(payload)
        patches = jnp.where(
            self.relation.valid.reshape(routes), value[self.relation.source_indices], 0
        )
        return jnp.sum(weights.reshape(routes) * patches, axis=1)

    def transpose_apply(self, cotangent: ArrayLike, multi_index: MultiIndex, /) -> Array:
        value = jnp.asarray(cotangent)
        if value.shape[:1] != (self.row_count,):
            raise ValueError("Family cotangents must begin with the row count.")
        weights = self.weights_for(multi_index)
        payload = value.shape[1:]
        routes = (self.relation.valid.shape) + (1,) * len(payload)
        messages = jnp.where(
            self.relation.valid.reshape(routes),
            weights.reshape(routes) * value[:, None],
            0,
        )
        output = jnp.zeros((self.source_count,) + payload, dtype=value.dtype)
        return output.at[self.relation.source_indices].add(messages)

    def operator(
        self,
        multi_index: MultiIndex,
        /,
        *,
        source: ArraySpace,
        target: ArraySpace,
        coefficients: ArrayLike | None = None,
    ) -> SparseCoordinateOperator:
        """Sparse ``diag(coefficients) D_alpha`` without forming a product."""
        weights = self.weights_for(multi_index)
        if coefficients is not None:
            scale = jnp.asarray(coefficients, dtype=weights.dtype)
            if scale.shape != (self.row_count,):
                raise ValueError("Row coefficients must have one value per row.")
            weights = scale[:, None] * weights
        return SparseCoordinateOperator(
            self.relation,
            weights,
            source=source,
            target=target,
            operator_id=f"{self.family_id}:{'-'.join(map(str, multi_index))}",
        )


def discretization_family(
    discretization: PreparedPointCloudDiscretization, /
) -> PointDerivativeFamily:
    """The cloud's own admitted stencils as a full-row derivative family."""
    if not isinstance(discretization, PreparedPointCloudDiscretization):
        raise TypeError("discretization must be PreparedPointCloudDiscretization.")
    indices = tuple(index for index, _ in discretization.mixed_weights)
    weights = tuple(weights for _, weights in discretization.mixed_weights)
    return PointDerivativeFamily(
        discretization.relation,
        weights,
        jnp.ones(discretization.state_shape, dtype=jnp.bool_),
        indices,
        family_id=canonical_fingerprint(
            {"kind": "point-derivative-family", "cloud": discretization.prepared_id}
        ),
    )


def prepare_point_family(
    cloud: PreparedPointCloudDiscretization,
    targets: np.ndarray,
    rows: np.ndarray,
    /,
    *,
    source_active: np.ndarray,
    neighbors: int,
    policy: LocalStencilPolicy,
    value: bool,
    maximum_candidates: int | None,
    target_chunk_size: int | None,
    label: str,
) -> tuple[PointDerivativeFamily, LocalStencilReport, np.ndarray]:
    """Host preparation of admitted stencils for ``targets[rows]`` from active cloud points.

    Sources keep the cloud's stable identities and canonical address, so
    periodic clouds use minimum-image offsets. Returns the full-row family, its
    stencil report, and per-row status/condition/amplification evidence of
    shape ``(3, rows.size)``.
    """
    return _prepare_family(
        np.asarray(cloud.points),
        np.asarray(cloud.plan.point_ids),
        cloud.plan.address,
        targets,
        rows,
        source_active=source_active,
        neighbors=neighbors,
        policy=policy,
        value=value,
        maximum_candidates=maximum_candidates,
        target_chunk_size=target_chunk_size,
        label=label,
    )


def _prepare_family(
    sources: np.ndarray,
    source_ids: np.ndarray,
    address: MortonAddressPlan | None,
    targets: np.ndarray,
    rows: np.ndarray,
    /,
    *,
    source_active: np.ndarray,
    neighbors: int,
    policy: LocalStencilPolicy,
    value: bool,
    maximum_candidates: int | None,
    target_chunk_size: int | None,
    label: str,
    support: np.ndarray | None = None,
) -> tuple[PointDerivativeFamily, LocalStencilReport, np.ndarray]:
    """Admitted stencils for ``targets[rows]`` from explicit identified sources.

    ``support`` (default ``targets``) supplies the points whose nearest sources
    form each row's stencil; the stencil is still fitted at ``targets``.
    """
    dimension = sources.shape[1]
    indices = derivative_multi_indices(dimension, value=value)
    functionals = tuple(
        MeshfreeFunctional((index,), (1.0,), name=f"{label}:{'-'.join(map(str, index))}")
        for index in indices
    )
    neighborhood = MeshfreeNeighborhoodPlan(
        sources,
        neighbors,
        targets=(targets if support is None else support)[rows],
        source_active=source_active,
        source_ids=source_ids,
        address=address,
        maximum_candidates=maximum_candidates,
        target_chunk_size=target_chunk_size,
    ).prepare()
    stencils = prepare_local_stencils(
        neighborhood, sources, targets[rows], functionals, policy
    )
    count, width = targets.shape[0], neighborhood.relation.width
    full_indices = np.zeros((count, width), dtype=np.int32)
    full_valid = np.zeros((count, width), dtype=np.bool_)
    full_indices[rows] = np.asarray(neighborhood.relation.source_indices)
    full_valid[rows] = np.asarray(neighborhood.relation.valid)
    full_weights: list[np.ndarray] = []
    for weights in stencils.weights:
        full = np.zeros((count, width), dtype=np.float64)
        full[rows] = np.asarray(weights)
        full_weights.append(full)
    active = np.zeros(count, dtype=np.bool_)
    active[rows] = True
    family = PointDerivativeFamily(
        RowRelation(full_indices, source_size=sources.shape[0], valid=full_valid),
        full_weights,
        active,
        indices,
        family_id=canonical_fingerprint(
            {
                "kind": "point-derivative-family",
                "label": label,
                "stencils": stencils.prepared_id,
            }
        ),
    )
    evidence = np.stack(
        (
            np.asarray(stencils.evidence.status, dtype=np.float64),
            np.asarray(stencils.evidence.condition, dtype=np.float64),
            np.asarray(stencils.evidence.amplification, dtype=np.float64),
        )
    )
    return family, stencils.report, evidence


@final
class PointGhostLayerEvidence(StrictModule, NonTrainableState):
    """Placement and admission evidence of a prepared ghost layer.

    ``offsets`` are the normal distances of the ghosts from their boundary
    rows. ``separation`` is a lower bound on each ghost's distance to its
    nearest other point (real or ghost), divided by its offset. ``ghost_weight`` is the weight of a
    ghost in its own boundary row's geometric normal-derivative stencil, times
    its offset (dimensionless, positive for an outward ghost), and
    ``ghost_dominance`` that weight over the summed magnitude of the row's
    other ghost weights. Dominance above one at every row makes the ghost block
    of the normal-derivative equations strictly diagonally dominant, hence
    nonsingular (Levy–Desplanques): the ghost values are then uniquely
    determined by the cloud values. It is evidence, not an admission gate; the
    consuming plan's spectral assessment owns admission of the full system.
    ``direction`` is each ghost's placement direction and
    ``component_normals`` the per-component declared flux normals of its row
    (the plan's fields).
    """

    offsets: Array
    separation: Array
    ghost_weight: Array
    ghost_dominance: Array
    direction: Array
    component_normals: Array
    extended_stencil_report: LocalStencilReport
    minimum_separation: float = eqx.field(static=True)
    minimum_ghost_dominance: float = eqx.field(static=True)

    @property
    def ghost_block_dominant(self) -> bool:
        return self.minimum_ghost_dominance > 1.0


@final
class PointGhostLayerPlan(StrictModule):
    """One ghost unknown per Neumann or Robin boundary row (the PDE+BC route).

    Square collocation that replaces the PDE at a boundary point by its flux
    condition leaves one-sided stencils whose difference modes can be spurious
    eigenvalues with nonpositive real part. This route keeps two equations per
    flux boundary point ``x_b``: the PDE and the boundary condition. Each such
    row receives a ghost point ``x_g = x_b + δ_b n_b`` outside the domain, with
    ``δ_b = offset · h_b`` and ``h_b`` the distance from ``x_b`` to its nearest
    cloud point, and one ghost unknown ``u_g``; ghost-extended stencils
    collocate the PDE at every non-Dirichlet cloud point, and the boundary
    condition at ``x_b`` occupies the ghost's equation row, so the system stays
    square.

    Extension law: ``u_g`` is the value at ``x_g`` of the smooth continuation
    of ``u`` that satisfies both the PDE and the boundary condition at ``x_b``.
    It is determined by the coupled equations, not prescribed by
    extrapolation. Its evidence is the extension defect
    ``max_g |u_g - (E u)(x_g)|`` reported with every solve, where ``E`` is the
    one-sided reconstruction at ``x_g`` from real cloud points only; for smooth
    solutions it decreases at the reconstruction order. Ghosts closer to
    another point than ``minimum_separation · δ_b`` are refused (they would
    duplicate a real sample, typically at a reentrant corner). Ghost
    layers serve single-sided bounded clouds; side-labeled rows and periodic
    images are refused. For a block plan (``components > 1``) every row that
    any component owns with a Neumann or Robin condition receives one ghost
    point (sorted by row), shared by every component: each component then has
    one ghost unknown there and two equations at ``x_b``, its PDE and its own
    condition (Dirichlet, traction/Neumann, or Robin). ``normals`` holds the
    ghost direction and ``component_normals[g, a]`` component ``a``'s declared
    flux normal at that row (zero where ``a`` is not flux-owned). Where the
    flux-owning components declare different normals (a corner between two
    traction faces) the ghost lies along their normalized sum; normals that
    cancel are refused.
    """

    rows: Array
    normals: Array
    component_normals: Array
    labels: tuple[str, ...] = eqx.field(static=True)
    offset: float = eqx.field(static=True)
    minimum_separation: float = eqx.field(static=True)
    maximum_candidates: int | None = eqx.field(static=True)
    target_chunk_size: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        boundary: PointBoundaryPlan,
        /,
        *,
        offset: float = 1.0,
        minimum_separation: float = 0.5,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
    ) -> None:
        # Runtime import: the boundary plan's owner imports this module, so a
        # module-level import (and hence a `checked` annotation) would cycle.
        from .._point_cloud_pde import PointBoundaryPlan

        if not isinstance(boundary, PointBoundaryPlan):
            raise TypeError("boundary must be a PointBoundaryPlan.")
        offset_, separation = float(offset), float(minimum_separation)
        if not np.isfinite(offset_) or offset_ <= 0.0:
            raise ValueError("offset must be finite and positive.")
        if not np.isfinite(separation) or separation <= 0.0:
            raise ValueError("minimum_separation must be finite and positive.")
        rows: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        owners: list[int] = []
        labels: list[str] = []
        for condition in boundary.conditions:
            match condition.kind:
                case "dirichlet":
                    continue
                case "neumann" | "robin":
                    if condition.side is not None:
                        raise ValueError(
                            f"Condition {condition.label!r} names a side; ghost layers are single-sided."
                        )
                    if condition.normals is None:
                        raise RuntimeError("Flux conditions always carry normals.")
                    rows.append(np.asarray(condition.rows))
                    normals.append(np.asarray(condition.normals))
                    owners.append(condition.component)
                    labels.append(condition.label)
                case "periodic":
                    raise ValueError(
                        "Periodic rows identify images; ghost layers serve Neumann and Robin rows only."
                    )
                case _:
                    assert_never(condition.kind)
        if not rows:
            raise ValueError("The boundary plan has no Neumann or Robin rows to extend.")
        stacked = np.concatenate(rows).astype(np.int32)
        stacked_normals = np.concatenate(normals)
        if boundary.components == 1:
            # Scalar plans keep the declaration order of their conditions.
            index, direction = stacked, stacked_normals
            component_normals = stacked_normals[:, None, :]
        else:
            # One ghost point per flux row, shared by the components flux-owned
            # there. Each component's condition keeps its own declared normal;
            # where they differ (a corner between two traction faces) the ghost
            # lies along their normalized sum, the outward bisector.
            index = np.unique(stacked)
            position = np.searchsorted(index, stacked)
            component = np.concatenate(
                [
                    np.full(r.size, owner, dtype=np.int32)
                    for r, owner in zip(rows, owners, strict=True)
                ]
            )
            component_normals = np.zeros(
                (index.size, boundary.components, stacked_normals.shape[1]),
                dtype=np.float64,
            )
            component_normals[position, component] = stacked_normals
            total = np.zeros((index.size, stacked_normals.shape[1]), dtype=np.float64)
            for normal in np.unique(stacked_normals, axis=0):
                declared = np.any(np.all(component_normals == normal, axis=2), axis=1)
                total[declared] += normal
            length = np.linalg.norm(total, axis=1)
            if np.any(length <= 1e-12):
                worst = int(index[np.argmin(length)])
                raise ValueError(
                    f"Flux normals declared at row {worst} cancel; no outward ghost "
                    "direction exists there."
                )
            direction = total / length[:, None]
        self.rows = jnp.asarray(index)
        self.normals = jnp.asarray(direction)
        self.component_normals = jnp.asarray(component_normals)
        self.labels = tuple(labels)
        self.offset = offset_
        self.minimum_separation = separation
        self.maximum_candidates = maximum_candidates
        self.target_chunk_size = target_chunk_size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "point-ghost-layer-plan",
                "rows": array_tree_fingerprint(index),
                "normals": array_tree_fingerprint(direction),
                "component_normals": array_tree_fingerprint(component_normals),
                "labels": tuple(labels),
                "offset": offset_,
                "minimum_separation": separation,
                "candidates": maximum_candidates,
                "chunk": target_chunk_size,
            }
        )

    def prepare(
        self, discretization: PreparedPointCloudDiscretization, /
    ) -> PreparedPointGhostLayer:
        return PreparedPointGhostLayer(self, discretization)


@final
class PreparedPointGhostLayer(StrictModule, NonTrainableState):
    """Ghost points and ghost-extended stencils over one prepared cloud.

    Extended coordinates list the cloud points first, then one ghost per
    ``plan.rows`` entry in that order. ``family`` holds derivative stencils at
    every cloud point from all extended points (ghost rows carry none);
    ``extension`` holds value stencils at the ghosts from cloud points only.
    """

    plan: PointGhostLayerPlan
    points: Array
    point_ids: Array
    offsets: Array
    family: PointDerivativeFamily
    extension: PointDerivativeFamily
    evidence: PointGhostLayerEvidence
    cloud_count: int = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: PointGhostLayerPlan,
        discretization: PreparedPointCloudDiscretization,
        /,
    ) -> None:
        if any(discretization.plan.address.periodic_axes):
            raise ValueError(
                "Ghost layers extend bounded clouds; periodic axes have no exterior."
            )
        points = np.asarray(discretization.points)
        count, dimension = points.shape
        rows = np.asarray(plan.rows)
        normals = np.asarray(plan.normals)
        if normals.shape[1] != dimension or np.any(rows >= count):
            raise ValueError("Ghost rows and normals must belong to this cloud.")
        identifiers = np.asarray(discretization.plan.point_ids)
        # Undeclared query capacities inherit the cloud's own declaration.
        candidates = (
            discretization.plan.maximum_candidates
            if plan.maximum_candidates is None
            else plan.maximum_candidates
        )
        chunk = (
            discretization.plan.target_chunk_size
            if plan.target_chunk_size is None
            else plan.target_chunk_size
        )
        nearest = MeshfreeNeighborhoodPlan(
            points,
            2,
            targets=points[rows],
            source_ids=identifiers,
            address=discretization.plan.address,
            maximum_candidates=candidates,
            target_chunk_size=chunk,
        ).prepare()
        offsets = plan.offset * _nearest_other(nearest.distances, nearest.relation.valid)
        ghosts = points[rows] + offsets[:, None] * normals
        extended = np.concatenate((points, ghosts))
        if np.unique(extended, axis=0).shape[0] != extended.shape[0]:
            raise ValueError(
                "A ghost coincides with another point (separation 0 < minimum_separation "
                f"{plan.minimum_separation:g}); the boundary geometry, normals, or offset "
                "do not admit a ghost layer there."
            )
        extended_ids = np.concatenate(
            (
                identifiers,
                identifiers.max() + 1 + np.arange(rows.size, dtype=identifiers.dtype),
            )
        )
        family, report, _ = _prepare_family(
            extended,
            extended_ids,
            None,
            extended,
            np.arange(count),
            source_active=np.ones(extended.shape[0], dtype=np.bool_),
            neighbors=discretization.plan.neighbors,
            policy=discretization.plan.stencil,
            value=False,
            maximum_candidates=candidates,
            target_chunk_size=chunk,
            label=f"{plan.plan_id}:ghost-extended",
        )
        if report.refused_rows:
            raise ValueError(
                f"Ghost-extended stencils refused {report.refused_rows} cloud rows (worst row {report.worst_row})."
            )
        # The extension at x_g uses the cloud support of its boundary row x_b:
        # the nearest real samples, without an exterior candidate search.
        extension, extension_report, _ = _prepare_family(
            points,
            identifiers,
            discretization.plan.address,
            ghosts,
            np.arange(rows.size),
            source_active=np.ones(count, dtype=np.bool_),
            neighbors=discretization.plan.neighbors,
            policy=discretization.plan.stencil,
            value=True,
            maximum_candidates=candidates,
            target_chunk_size=chunk,
            label=f"{plan.plan_id}:ghost-extension",
            support=points[rows],
        )
        if extension_report.refused_rows:
            raise ValueError(
                f"One-sided extension stencils refused {extension_report.refused_rows} ghosts."
            )
        separation = _ghost_separation(extended, family, rows, offsets, count)
        if np.min(separation) < plan.minimum_separation:
            worst = int(np.argmin(separation))
            raise ValueError(
                f"Ghost of row {int(rows[worst])} lies within {float(separation[worst]):.3g} "
                f"of its offset from another point (minimum_separation "
                f"{plan.minimum_separation:g}); the boundary geometry or offset "
                "does not admit a ghost layer there."
            )
        weight, dominance = _ghost_block(family, rows, normals, offsets, count)
        if np.any(weight <= 0.0):
            raise ValueError(
                f"Ghost of row {int(rows[np.argmin(weight)])} enters its outward "
                "normal derivative with nonpositive weight; normals must point "
                "out of the domain."
            )
        self.plan = plan
        self.points = jnp.asarray(extended)
        self.point_ids = jnp.asarray(extended_ids)
        self.offsets = jnp.asarray(offsets)
        self.family = family
        self.extension = extension
        self.evidence = PointGhostLayerEvidence(
            offsets=jnp.asarray(offsets),
            separation=jnp.asarray(separation),
            ghost_weight=jnp.asarray(weight),
            ghost_dominance=jnp.asarray(dominance),
            direction=plan.normals,
            component_normals=plan.component_normals,
            minimum_separation=float(np.min(separation)),
            minimum_ghost_dominance=float(np.min(dominance)),
            extended_stencil_report=report,
        )
        self.cloud_count = count
        self.discretization_id = discretization.prepared_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-point-ghost-layer",
                "plan": plan.plan_id,
                "cloud": discretization.prepared_id,
                "family": family.family_id,
                "extension": extension.family_id,
            }
        )

    @property
    def ghost_count(self) -> int:
        return self.plan.rows.shape[0]

    @property
    def row_count(self) -> int:
        """Extended unknowns: cloud points then ghosts."""
        return self.points.shape[0]

    def extension_defect(self, values: ArrayLike, /) -> Array:
        """``max_g |u_g - (E u)(x_g)|`` of extended values (cloud, then ghosts)."""
        value = jnp.asarray(values)
        if value.shape != (self.row_count,):
            raise ValueError("Extended values must list cloud points then ghosts.")
        dimension = self.points.shape[1]
        reconstructed = self.extension.apply(value[: self.cloud_count], (0,) * dimension)
        return jnp.max(jnp.abs(value[self.cloud_count :] - reconstructed))


def _nearest_other(distances: ArrayLike, valid: ArrayLike, /) -> np.ndarray:
    """Per-row distance to the nearest source other than a coincident self."""
    distance = np.asarray(distances)
    mask = np.asarray(valid) & (distance > 0.0)
    return np.min(np.where(mask, distance, np.inf), axis=1)


def _ghost_separation(
    extended: np.ndarray,
    family: PointDerivativeFamily,
    rows: np.ndarray,
    offsets: np.ndarray,
    count: int,
    /,
) -> np.ndarray:
    """Lower bound on each ghost's distance to any other point, over its offset.

    The ghost-extended support of boundary row ``x_b`` holds its nearest
    extended points within radius ``r``. A point outside it is at least
    ``r - δ`` from ``x_g = x_b + δ n``, so ``min(found, r - δ)`` bounds the
    nearest-other distance from below without an exterior search.
    """
    indices = np.asarray(family.relation.source_indices)[rows]
    valid = np.asarray(family.relation.valid)[rows]
    ghosts = extended[count:]
    own = indices == (count + np.arange(rows.size))[:, None]
    to_ghost = np.linalg.norm(extended[indices] - ghosts[:, None, :], axis=-1)
    found = np.min(np.where(valid & ~own, to_ghost, np.inf), axis=1)
    radius = np.max(
        np.where(
            valid,
            np.linalg.norm(extended[indices] - extended[rows][:, None, :], axis=-1),
            0.0,
        ),
        axis=1,
    )
    return np.minimum(found, radius - offsets) / offsets


def _ghost_block(
    family: PointDerivativeFamily,
    rows: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    count: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Own-ghost weight (times offset) and ghost-block row dominance of ``n · grad``."""
    dimension = normals.shape[1]
    indices = np.asarray(family.relation.source_indices)[rows]
    valid = np.asarray(family.relation.valid)[rows]
    weights = sum(
        normals[:, axis, None]
        * np.asarray(family.weights_for(tuple(int(d == axis) for d in range(dimension))))[
            rows
        ]
        for axis in range(dimension)
    )
    own = valid & (indices == (count + np.arange(rows.size))[:, None])
    other = valid & (indices >= count) & ~own
    weight = offsets * np.sum(np.where(own, weights, 0.0), axis=1)
    others = offsets * np.sum(np.where(other, np.abs(weights), 0.0), axis=1)
    dominance = weight / np.where(others > 0.0, others, np.inf)
    dominance = np.where(others > 0.0, dominance, np.inf)
    return weight, dominance


@final
class PointSideAdmissionEvidence(StrictModule, NonTrainableState):
    """Per-side stencil admission, with one-sided rows identified explicitly.

    A one-sided row belongs to several sides; each side's stencil there uses
    only that side's actual points. Arrays use shape ``(sides, points)`` and
    are zero outside a side's membership.
    """

    membership: Array
    one_sided: Array
    status: Array
    condition: Array
    amplification: Array
    maximum_one_sided_condition: float = eqx.field(static=True)
    maximum_one_sided_amplification: float = eqx.field(static=True)
    refused_rows: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @property
    def admitted(self) -> bool:
        return self.refused_rows == 0


@final
class PointSideSupportPlan(StrictModule, NonTrainableState):
    """Declared side membership over one prepared point cloud.

    ``membership[i, s]`` states that point ``i`` belongs to side ``sides[s]``.
    Points on several sides (material interfaces) receive one one-sided stencil
    per side. Membership is scientific input: it is never inferred from
    coordinates or coefficient values.
    """

    discretization: PreparedPointCloudDiscretization
    membership: Array
    stencil: LocalStencilPolicy
    sides: tuple[str, ...] = eqx.field(static=True)
    neighbors: int = eqx.field(static=True)
    maximum_candidates: int | None = eqx.field(static=True)
    target_chunk_size: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: PreparedPointCloudDiscretization,
        membership: ArrayLike,
        /,
        *,
        sides: Sequence[str],
        stencil: LocalStencilPolicy | None = None,
        neighbors: int | None = None,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
    ) -> None:
        if not isinstance(discretization, PreparedPointCloudDiscretization):
            raise TypeError("discretization must be PreparedPointCloudDiscretization.")
        names = tuple(canonical_identifier(name, "side") for name in sides)
        if not names or len(set(names)) != len(names):
            raise ValueError("sides must be nonempty unique identifiers.")
        table = np.asarray(membership)
        count = discretization.state_shape[0]
        if table.dtype != np.bool_ or table.shape != (count, len(names)):
            raise ValueError("membership must be Boolean with shape (points, sides).")
        if not np.all(np.any(table, axis=1)):
            raise ValueError("Every point must belong to at least one declared side.")
        policy = discretization.plan.stencil if stencil is None else stencil
        if not isinstance(policy, LocalStencilPolicy):
            raise TypeError("stencil must be a LocalStencilPolicy.")
        width = discretization.plan.neighbors if neighbors is None else int(neighbors)
        features = math.comb(
            discretization.spatial_dimension + policy.polynomial_degree,
            policy.polynomial_degree,
        )
        for name, column in zip(names, table.T, strict=True):
            if np.count_nonzero(column) < features:
                raise ValueError(
                    f"Side {name!r} has fewer points than the polynomial basis."
                )
        if width < features:
            raise ValueError("neighbors must cover the polynomial basis.")
        self.discretization = discretization
        self.membership = jnp.asarray(table)
        self.stencil = policy
        self.sides = names
        self.neighbors = width
        self.maximum_candidates = maximum_candidates
        self.target_chunk_size = target_chunk_size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "point-side-support-plan",
                "cloud": discretization.prepared_id,
                "sides": names,
                "membership": array_tree_fingerprint(table),
                "approximation": policy.approximation,
                "degree": policy.polynomial_degree,
                "phs_power": policy.phs_power,
                "weight_kernel": policy.weight_kernel,
                "condition_limit": policy.condition_limit,
                "amplification_limit": policy.amplification_limit,
                "acceptance": policy.acceptance,
                "neighbors": width,
                "candidates": maximum_candidates,
                "chunk": target_chunk_size,
            }
        )

    def prepare(self, /) -> PreparedPointSideSupport:
        return PreparedPointSideSupport(self)


@final
class PreparedPointSideSupport(StrictModule, NonTrainableState):
    """One admitted derivative family per side, restricted to that side's points."""

    plan: PointSideSupportPlan
    families: tuple[PointDerivativeFamily, ...]
    reports: tuple[LocalStencilReport, ...]
    evidence: PointSideAdmissionEvidence
    support_id: str = eqx.field(static=True)

    def __init__(self, plan: PointSideSupportPlan, /) -> None:
        if not isinstance(plan, PointSideSupportPlan):
            raise TypeError("plan must be PointSideSupportPlan.")
        points = np.asarray(plan.discretization.points)
        table = np.asarray(plan.membership)
        families: list[PointDerivativeFamily] = []
        reports: list[LocalStencilReport] = []
        status = np.zeros(table.T.shape, dtype=np.int32)
        condition = np.zeros(table.T.shape, dtype=np.float64)
        amplification = np.zeros(table.T.shape, dtype=np.float64)
        for side, (name, column) in enumerate(zip(plan.sides, table.T, strict=True)):
            rows = np.flatnonzero(column)
            family, report, row_evidence = prepare_point_family(
                plan.discretization,
                points,
                rows,
                source_active=column,
                neighbors=min(plan.neighbors, rows.size),
                policy=plan.stencil,
                value=False,
                maximum_candidates=plan.maximum_candidates,
                target_chunk_size=plan.target_chunk_size,
                label=f"{plan.plan_id}:{name}",
            )
            families.append(family)
            reports.append(report)
            status[side, rows] = row_evidence[0].astype(np.int32)
            condition[side, rows] = row_evidence[1]
            amplification[side, rows] = row_evidence[2]
        one_sided = np.count_nonzero(table, axis=1) > 1
        shared = table.T & one_sided[None, :]
        evidence = PointSideAdmissionEvidence(
            membership=jnp.asarray(table),
            one_sided=jnp.asarray(one_sided),
            status=jnp.asarray(status),
            condition=jnp.asarray(condition),
            amplification=jnp.asarray(amplification),
            maximum_one_sided_condition=float(np.max(condition[shared], initial=0.0)),
            maximum_one_sided_amplification=float(
                np.max(amplification[shared], initial=0.0)
            ),
            refused_rows=int(np.count_nonzero(status[table.T])),
            evidence_id=canonical_fingerprint(
                {
                    "kind": "point-side-admission",
                    "reports": tuple(report.report_id for report in reports),
                }
            ),
        )
        self.plan = plan
        self.families = tuple(families)
        self.reports = tuple(reports)
        self.evidence = evidence
        self.support_id = canonical_fingerprint(
            {
                "kind": "prepared-point-side-support",
                "plan": plan.plan_id,
                "families": tuple(family.family_id for family in families),
            }
        )

    @property
    def sides(self) -> tuple[str, ...]:
        return self.plan.sides

    def side_index(self, side: str, /) -> int:
        if side not in self.plan.sides:
            raise ValueError(
                f"Unknown side {side!r}; declared sides are {self.plan.sides}."
            )
        return self.plan.sides.index(side)

    def family(self, side: str, /) -> PointDerivativeFamily:
        return self.families[self.side_index(side)]


def _unit_normals(normals: np.ndarray, name: str, /) -> np.ndarray:
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(~np.isfinite(normals)) or np.any(lengths <= 0.0):
        raise ValueError(f"{name} must be finite and nonzero.")
    return normals / lengths[:, None]


def _rows(rows: ArrayLike, name: str, /) -> np.ndarray:
    value = np.asarray(rows)
    if value.ndim != 1 or value.size == 0 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must be a nonempty integer vector.")
    if np.any(value < 0) or np.unique(value).size != value.size:
        raise ValueError(f"{name} must be unique nonnegative row indices.")
    return value.astype(np.int32)


def _row_values(values: ArrayLike, count: int, name: str, /) -> np.ndarray:
    value = np.broadcast_to(np.asarray(values, dtype=np.float64), (count,)).copy()
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite.")
    return value


@final
class PointInterfaceCondition(StrictModule):
    """Flux transmission across a declared material interface.

    The unknown is single-valued at each interface row (value continuity), and
    the row equation is ``k_minus grad_minus(u).n - k_plus grad_plus(u).n =
    flux_jump`` with ``n`` the unit normal oriented from ``minus`` into
    ``plus`` and each gradient evaluated by its side's one-sided stencil.
    """

    rows: Array
    normals: Array
    flux_jump: Array
    measure: Array | None
    label: str = eqx.field(static=True)
    minus: str = eqx.field(static=True)
    plus: str = eqx.field(static=True)
    interface_id: str | None = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)

    def __init__(
        self,
        rows: ArrayLike,
        normals: ArrayLike,
        flux_jump: ArrayLike = 0.0,
        /,
        *,
        label: str,
        minus: str | None = None,
        plus: str | None = None,
        interface: SurfaceInterface | None = None,
        measure: ArrayLike | None = None,
    ) -> None:
        name = canonical_identifier(label, "label")
        if interface is not None:
            if not isinstance(interface, SurfaceInterface):
                raise TypeError("interface must be a SurfaceInterface.")
            if (minus is not None and minus != interface.minus_region) or (
                plus is not None and plus != interface.plus_region
            ):
                raise ValueError(
                    "Interface sides must match the SurfaceInterface regions."
                )
            minus, plus = interface.minus_region, interface.plus_region
        if minus is None or plus is None:
            raise ValueError("Interface conditions require minus and plus sides.")
        minus_ = canonical_identifier(minus, "minus")
        plus_ = canonical_identifier(plus, "plus")
        if minus_ == plus_:
            raise ValueError("Interface sides must differ.")
        index = _rows(rows, "rows")
        direction = np.asarray(normals, dtype=np.float64)
        if direction.ndim != 2 or direction.shape[0] != index.size:
            raise ValueError("normals must have shape (rows, dimension).")
        direction = _unit_normals(direction, "Interface normals")
        jump = _row_values(flux_jump, index.size, "flux_jump")
        weights = None if measure is None else _row_values(measure, index.size, "measure")
        if weights is not None and np.any(weights <= 0.0):
            raise ValueError("Interface measure must be positive.")
        self.rows = jnp.asarray(index)
        self.normals = jnp.asarray(direction)
        self.flux_jump = jnp.asarray(jump)
        self.measure = None if weights is None else jnp.asarray(weights)
        self.label = name
        self.minus = minus_
        self.plus = plus_
        self.interface_id = None if interface is None else interface.interface_id
        self.condition_id = canonical_fingerprint(
            {
                "kind": "point-interface-condition",
                "label": name,
                "minus": minus_,
                "plus": plus_,
                "interface": self.interface_id,
                "rows": array_tree_fingerprint(index),
                "normals": array_tree_fingerprint(direction),
                "flux_jump": array_tree_fingerprint(jump),
                "measure": None if weights is None else array_tree_fingerprint(weights),
            }
        )


@final
class PointBoundarySamples(StrictModule, NonTrainableState):
    """Physical boundary samples from an oriented geometry chart authority.

    ``measure`` is the physical surface measure (reference weight times the
    chart Jacobian), ``normals`` are the chart's oriented unit normals.
    """

    points: Array
    normals: Array
    measure: Array
    chart_indices: Array
    entity_ids: Array
    chart_tags: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    samples_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        normals: ArrayLike,
        measure: ArrayLike,
        chart_indices: ArrayLike,
        entity_ids: ArrayLike,
        /,
        *,
        chart_tags: Sequence[str],
        source_id: str,
    ) -> None:
        points_ = np.asarray(points, dtype=np.float64)
        normals_ = np.asarray(normals, dtype=np.float64)
        weights = np.asarray(measure, dtype=np.float64)
        charts = np.asarray(chart_indices, dtype=np.int32)
        entities = np.asarray(entity_ids, dtype=np.int32)
        if (
            points_.ndim != 2
            or normals_.shape != points_.shape
            or weights.shape != points_.shape[:1]
            or charts.shape != weights.shape
            or entities.shape != weights.shape
        ):
            raise ValueError("Boundary samples need matching (samples, dimension) data.")
        if not np.all(np.isfinite(points_)) or np.any(~np.isfinite(weights)):
            raise ValueError("Boundary sample points and measure must be finite.")
        if np.any(weights <= 0.0):
            raise ValueError("Boundary sample measure must be positive.")
        normals_ = _unit_normals(normals_, "Boundary sample normals")
        self.points = jnp.asarray(points_)
        self.normals = jnp.asarray(normals_)
        self.measure = jnp.asarray(weights)
        self.chart_indices = jnp.asarray(charts)
        self.entity_ids = jnp.asarray(entities)
        self.chart_tags = tuple(chart_tags)
        self.source_id = canonical_identifier(source_id, "source_id")
        self.samples_id = canonical_fingerprint(
            {
                "kind": "point-boundary-samples",
                "source": self.source_id,
                "points": array_tree_fingerprint(points_),
                "normals": array_tree_fingerprint(normals_),
                "measure": array_tree_fingerprint(weights),
                "charts": array_tree_fingerprint(charts),
            }
        )

    def select(self, tags: Sequence[str], /) -> np.ndarray:
        """Host sample indices whose chart carries one of ``tags``."""
        wanted = np.asarray([tag in set(tags) for tag in self.chart_tags], dtype=np.bool_)
        if not np.any(wanted):
            raise ValueError(f"No boundary chart carries tags {tuple(tags)}.")
        return np.flatnonzero(wanted[np.asarray(self.chart_indices)])


def sample_boundary_atlas(
    atlas: BoundaryAtlas,
    rule: FacetTraceRule,
    shape: FacetShape,
    /,
    *,
    charts: ConvertibleToArray | None = None,
) -> PointBoundarySamples:
    """Tensor a facet reference rule over oriented boundary charts.

    Every sample must lie in its chart's trim domain with a regular, finite,
    Jacobian-consistent frame; otherwise preparation refuses.
    """
    if not isinstance(atlas, BoundaryAtlas) or not isinstance(rule, FacetTraceRule):
        raise TypeError("A BoundaryAtlas and FacetTraceRule are required.")
    reference, weights = rule.reference(shape)
    if reference.shape[1] != atlas.reference_dimension:
        raise ValueError("Facet rule dimension must match the atlas reference charts.")
    selected = np.arange(atlas.num_charts) if charts is None else np.asarray(charts)
    if (
        selected.ndim != 1
        or not np.issubdtype(selected.dtype, np.integer)
        or np.any((selected < 0) | (selected >= atlas.num_charts))
    ):
        raise ValueError("charts must be valid boundary chart indices.")
    chart_indices = np.repeat(selected, reference.shape[0]).astype(np.int32)
    coordinates = np.tile(reference, (selected.size, 1))
    frame = atlas.frame(jnp.asarray(chart_indices), jnp.asarray(coordinates))
    inside = atlas.reference_mask(jnp.asarray(chart_indices), jnp.asarray(coordinates))
    admissible = np.asarray(
        frame.regular & frame.finite & frame.jacobian_consistent & inside
    )
    if not np.all(admissible):
        raise ValueError(
            "Boundary chart samples are outside the trim domain or have irregular frames."
        )
    measure = np.tile(weights, selected.size) * np.abs(np.asarray(frame.jacobian))
    return PointBoundarySamples(
        frame.origin,
        frame.normal,
        measure,
        chart_indices,
        np.asarray(atlas.source_entity_ids)[chart_indices],
        chart_tags=atlas.physical_tags,
        source_id=atlas.source_id,
    )


def sample_cubature_atlas(
    atlas: CubatureAtlas,
    chart_indices: ArrayLike,
    reference: ArrayLike,
    reference_weights: ArrayLike,
    /,
) -> PointBoundarySamples:
    """Physical boundary samples from a cubature map that supplies normals."""
    if not isinstance(atlas, CubatureAtlas):
        raise TypeError("atlas must be a CubatureAtlas.")
    charts = jnp.asarray(chart_indices, dtype=jnp.int32)
    evaluation = atlas.evaluate(charts, jnp.asarray(reference, dtype=jnp.float64))
    if evaluation.normal is None:
        raise ValueError(
            "The cubature map supplies no boundary normal; refusing to fabricate one."
        )
    if not bool(np.all(np.asarray(evaluation.admissible))):
        raise ValueError("Cubature samples are not admissible on their charts.")
    weights = np.asarray(reference_weights, dtype=np.float64)
    if weights.shape != tuple(charts.shape):
        raise ValueError("reference_weights must match chart_indices.")
    return PointBoundarySamples(
        evaluation.points,
        evaluation.normal,
        weights * np.asarray(evaluation.measure_scale),
        charts,
        np.asarray(atlas.source_entity_ids)[np.asarray(charts)],
        chart_tags=atlas.physical_tags,
        source_id=atlas.source_id,
    )


def sbp_identity_residuals(
    source_indices: np.ndarray,
    valid: np.ndarray,
    weights: Sequence[np.ndarray],
    mass: np.ndarray,
    boundary: np.ndarray,
    normals: np.ndarray,
    /,
) -> tuple[float, float]:
    """Maximum ``|M D + Dᵀ M - B n|`` coefficient and ``|1ᵀ M D - 1ᵀ B n|``.

    A complete host audit of every sparse coefficient for every axis.
    """
    count, width = source_indices.shape
    rows = np.repeat(np.arange(count), width)[valid.reshape(-1)]
    columns = source_indices.reshape(-1)[valid.reshape(-1)]
    maximum_green = maximum_conservation = 0.0
    for axis, axis_weights in enumerate(weights):
        values = (mass[:, None] * axis_weights).reshape(-1)[valid.reshape(-1)]
        keys = np.concatenate((rows * count + columns, columns * count + rows))
        contributions = np.concatenate((values, values))
        diagonal = np.arange(count)
        keys = np.concatenate((keys, diagonal * count + diagonal))
        contributions = np.concatenate((contributions, -boundary * normals[:, axis]))
        unique, inverse = np.unique(keys, return_inverse=True)
        green = np.zeros(unique.size, dtype=np.float64)
        np.add.at(green, inverse, contributions)
        conservation = -boundary * normals[:, axis]
        np.add.at(conservation, columns, values)
        maximum_green = max(maximum_green, float(np.max(np.abs(green), initial=0.0)))
        maximum_conservation = max(
            maximum_conservation, float(np.max(np.abs(conservation)))
        )
    return maximum_green, maximum_conservation


@final
class PointSBPDerivatives(StrictModule, NonTrainableState):
    """First derivatives with separate algebraic and native realization evidence.

    Each axis solves ``min 0.5 ||D - D_0||²`` subject to polynomial
    reproduction through ``reproduction_degree`` and every coefficient of
    ``M D + Dᵀ M = B n``, with ``D_0`` the cloud's admitted derivative. The
    native conic owner reports optimality or a validated infeasibility ray;
    an infeasible or unconverged axis is never relabeled as SBP.

    Constrained identities do not certify continuum stability. Only
    ``prepare_tensor_point_sbp`` binds the actual native tensor SBP owners,
    physical measures, and declared point-row permutation.
    """

    relation: RowRelation
    weights: tuple[Array, ...]
    status: Array
    certificate_valid: Array
    reproduction_residual: Array
    green_residual: Array
    conservation_residual: Array
    deviation: Array
    program_results: tuple[ConvexProgramResult, ...]
    tolerance: float = eqx.field(static=True)
    reproduction_degree: int = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    cloud_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    native_derivatives: tuple[PreparedSBPOperator, ...] = ()
    native_binding_id: str | None = eqx.field(static=True, default=None)

    @property
    def stable_realization(self) -> bool:
        """Whether intact native tensor SBP owners authorize these derivative rows."""
        return bool(self.native_derivatives) and self.native_binding_id == (
            _native_sbp_binding_id(
                self.native_derivatives, self.binding_id, self.relation, self.weights
            )
        )

    @checked
    def bind(
        self, discretization: PreparedPointCloudDiscretization, /
    ) -> PointDerivativeFamily:
        """Admit the original cloud, quadrature, axes, and sparse derivative identity."""
        if (
            self.cloud_id != discretization.prepared_id
            or self.binding_id
            != _sbp_binding_id(discretization, self.relation, self.weights)
        ):
            raise ValueError(
                "SBP derivatives belong to a different cloud or numerical revision."
            )
        if self.native_derivatives and not self.stable_realization:
            raise ValueError(
                "Native SBP realization belongs to a different numerical revision."
            )
        dimension = discretization.spatial_dimension
        count = discretization.state_shape[0]
        if (
            len(self.weights) != dimension
            or self.relation.source_size != count
            or self.relation.output_shape != (count,)
            or not bool(np.asarray(self.successful))
        ):
            raise ValueError(
                "SBP derivatives require admitted evidence for every cloud axis."
            )
        boundary = discretization.plan.boundary_quadrature_weights
        if boundary is None:
            raise ValueError("SBP binding requires the prepared boundary quadrature.")
        green, conservation = sbp_identity_residuals(
            np.asarray(self.relation.source_indices),
            np.asarray(self.relation.valid),
            tuple(np.asarray(weight) for weight in self.weights),
            np.asarray(discretization.quadrature_weights),
            np.asarray(boundary),
            np.asarray(discretization.plan.boundary_normals),
        )
        constant = max(
            float(
                np.max(
                    np.abs(
                        np.sum(
                            np.where(
                                np.asarray(self.relation.valid), np.asarray(weight), 0.0
                            ),
                            axis=1,
                        )
                    )
                )
            )
            for weight in self.weights
        )
        if max(green, conservation, constant) > self.tolerance:
            raise ValueError(
                "SBP binding failed its full Green or constant-derivative audit."
            )
        return PointDerivativeFamily(
            self.relation,
            self.weights,
            jnp.ones((count,), dtype=jnp.bool_),
            tuple(
                tuple(int(a == axis) for a in range(dimension))
                for axis in range(dimension)
            ),
            family_id=self.result_id,
        )

    @property
    def feasible(self) -> Array:
        return self.status == 0

    @property
    def successful(self) -> Array:
        return jnp.all(
            self.feasible
            & (self.reproduction_residual <= self.tolerance)
            & (self.green_residual <= self.tolerance)
            & (self.conservation_residual <= self.tolerance)
        )

    def operator(self, axis: int, /, *, space: ArraySpace) -> SparseCoordinateOperator:
        if not 0 <= axis < len(self.weights):
            raise ValueError("axis must select a prepared SBP derivative.")
        return SparseCoordinateOperator(
            self.relation,
            self.weights[axis],
            source=space,
            target=space,
            operator_id=f"{self.result_id}:sbp-derivative:{axis}",
        )


def _sbp_binding_id(
    discretization: PreparedPointCloudDiscretization,
    relation: RowRelation,
    weights: tuple[Array, ...],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "cloud": discretization.prepared_id,
            "dimension": discretization.spatial_dimension,
            "numerical": array_tree_fingerprint(
                (
                    discretization.points,
                    discretization.quadrature_weights,
                    discretization.plan.boundary_quadrature_weights,
                    discretization.plan.boundary_normals,
                    discretization.plan.point_ids,
                    relation.source_indices,
                    relation.valid,
                    weights,
                )
            ),
        }
    )


def _native_sbp_binding_id(
    derivatives: tuple[PreparedSBPOperator, ...],
    binding_id: str,
    relation: RowRelation,
    weights: tuple[Array, ...],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "native-tensor-point-sbp-binding",
            "cloud-binding": binding_id,
            "owners": tuple(value.prepared_id for value in derivatives),
            "evidence": tuple(
                {
                    "grid": value.grid.prepared_id,
                    "axes": value.grid.axis_names,
                    "axis": value.axis,
                    "axis_index": value.axis_index,
                    "family": value.family.family_id,
                    "order": value.family.interior_order,
                    "closure": value.family.closure_order,
                    "boundary_width": value.family.boundary_width,
                    "norm_coefficients": value.family.norm_boundary_weights,
                    "plan": value.plan.plan_id,
                    "operator": value.operator.operator_id,
                    "stencil": value.operator.stencil_set.stencil.stencil_id,
                    "report": value.stability_report.report_id,
                    "report_residual": value.stability_report.residual,
                    "report_tolerance": value.stability_report.tolerance,
                    "report_passed": value.stability_report.passed,
                }
                for value in derivatives
            ),
            "numerical": array_tree_fingerprint(
                (derivatives, relation.source_indices, relation.valid, weights)
            ),
        }
    )


def _tensor_sbp_geometry(
    cloud: PreparedPointCloudDiscretization,
    derivatives: tuple[PreparedSBPOperator, ...],
    /,
) -> tuple[SBPGridNorm, np.ndarray]:
    if not derivatives or any(
        type(value) is not PreparedSBPOperator for value in derivatives
    ):
        raise TypeError("derivatives must contain native PreparedSBPOperator owners.")
    grid = derivatives[0].grid
    if len(derivatives) != len(grid.shape):
        raise ValueError("Tensor point SBP requires one derivative per grid axis.")
    if any(
        value.grid is not grid
        or value.plan.grid is not grid
        or value.operator.stencil_set.stencil.source_location
        is not grid.centered_location
        or value.operator.stencil_set.stencil.target_location
        is not grid.centered_location
        or value.family is not value.plan.family
        or value.axis != grid.axis_names[axis]
        or value.plan.axis != value.axis
        or value.operator.stencil_set.stencil.request.axis != value.axis
        or value.operator.stencil_set.stencil.request.derivative_order != 1
        or not eqx.tree_equal(value.family, SBPFamily(value.family.interior_order))
        or value.axis_index != axis
        or value.operator.axis != axis
        or value.operator.source.shape != grid.shape
        or value.operator.target.shape != grid.shape
        or value.stability_report.passed is not True
        for axis, value in enumerate(derivatives)
    ):
        raise ValueError(
            "Native SBP owners must share their actual grid and grid-axis order."
        )
    for value in derivatives:
        stencil = value.operator.stencil_set.stencil
        if (
            not np.array_equal(
                np.asarray(value.operator.indices), np.asarray(stencil.indices)
            )
            or not np.array_equal(
                np.asarray(value.operator.valid), np.asarray(stencil.valid)
            )
            or not np.array_equal(
                np.asarray(value.operator.weights),
                np.asarray(stencil.weights).astype(value.operator.weights.dtype),
                equal_nan=True,
            )
        ):
            raise ValueError(
                "Native SBP stencil and applied coefficients have stale source identity."
            )
    if any(
        axis.primary_entity != "point" or axis.periodic for axis in grid.structured_axes
    ):
        raise ValueError("Tensor point SBP requires bounded point-primary axes.")
    norm = SBPGridNorm(derivatives)
    rows = np.asarray(cloud.plan.point_ids)
    if (
        cloud.spatial_dimension != len(grid.shape)
        or rows.shape != (grid.size,)
        or not np.array_equal(np.sort(rows), np.arange(grid.size))
    ):
        raise ValueError(
            "point_ids must declare a permutation of C-order tensor grid rows."
        )
    if not np.array_equal(np.asarray(cloud.points), np.asarray(grid.points)[rows]):
        raise ValueError(
            "Cloud geometry must exactly match the declared native grid row map."
        )
    if not np.array_equal(
        np.asarray(cloud.quadrature_weights), np.asarray(norm.weights).reshape(-1)[rows]
    ):
        raise ValueError(
            "Cloud volume cubature must equal the physical native tensor SBP norm."
        )
    signed = np.zeros(grid.shape + (len(grid.shape),), dtype=np.float64)
    for axis, derivative in enumerate(derivatives):
        factors = [
            np.asarray(derivative.boundary_diagonal)
            if other == axis
            else np.asarray(norm.axis_weights[other])
            for other in range(len(grid.shape))
        ]
        face = factors[0]
        for factor in factors[1:]:
            face = np.multiply.outer(face, factor)
        signed[..., axis] = face
    signed = signed.reshape((-1, len(grid.shape)))[rows]
    measure = np.linalg.norm(signed, axis=1)
    declared = cloud.plan.boundary_quadrature_weights
    if declared is None:
        raise ValueError(
            "Cloud boundary cubature must declare the physical tensor SBP face measure."
        )
    boundary = np.asarray(declared)
    if not np.allclose(boundary, measure, rtol=0.0, atol=1e-14) or not np.array_equal(
        np.asarray(cloud.plan.boundary_mask), measure > 0.0
    ):
        raise ValueError(
            "Cloud boundary cubature must equal the physical tensor SBP face measure."
        )
    if not np.allclose(
        boundary[:, None] * np.asarray(cloud.plan.boundary_normals),
        signed,
        rtol=0.0,
        atol=1e-14,
    ):
        raise ValueError(
            "Cloud boundary normals must orient the physical tensor SBP faces."
        )
    return norm, rows


def _tensor_sbp_sparse_rows(
    derivatives: tuple[PreparedSBPOperator, ...], rows: np.ndarray, /
) -> tuple[RowRelation, tuple[Array, ...]]:
    """Use native sparse Kronecker assembly and the declared C-order row map."""
    grid = derivatives[0].grid
    inverse = np.empty(grid.size, dtype=np.int32)
    inverse[rows] = np.arange(grid.size, dtype=np.int32)
    coordinates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for axis, derivative in enumerate(derivatives):
        native = derivative.operator
        space = ArraySpace((grid.shape[axis],), dtype=native.source.dtype)
        factor = SparseCoordinateOperator(
            RowRelation(native.indices, source_size=grid.shape[axis], valid=native.valid),
            jnp.where(native.valid, native.weights, 0.0),
            source=space,
            target=space,
            operator_id=f"{native.operator_id}:axis-factor",
        )
        product = KroneckerLinearOperator(
            tuple(
                factor
                if other == axis
                else IdentityLinearOperator(
                    ArraySpace((size,), dtype=native.source.dtype)
                )
                for other, size in enumerate(grid.shape)
            )
        )
        storage = assemble_sparse(product).sparse_storage()
        native_rows = np.repeat(np.arange(grid.size), np.diff(np.asarray(storage.indptr)))
        coordinates.append(
            (
                inverse[native_rows],
                inverse[np.asarray(storage.indices)],
                np.asarray(storage.values),
            )
        )
    pairs = np.unique(
        np.concatenate(
            [np.stack((output, source), axis=1) for output, source, _ in coordinates]
        ),
        axis=0,
    )
    width = int(np.max(np.bincount(pairs[:, 0], minlength=grid.size)))
    indices = np.zeros((grid.size, width), dtype=np.int32)
    valid = np.zeros_like(indices, dtype=np.bool_)
    slots: dict[tuple[int, int], int] = {}
    offsets = np.zeros(grid.size, dtype=np.int32)
    for output, source in pairs:
        slot = offsets[output]
        indices[output, slot] = source
        valid[output, slot] = True
        slots[(int(output), int(source))] = int(slot)
        offsets[output] += 1
    weights = []
    for outputs, sources, values in coordinates:
        axis_weights = np.zeros_like(indices, dtype=np.float64)
        for output, source, value in zip(outputs, sources, values, strict=True):
            axis_weights[output, slots[(int(output), int(source))]] = value
        weights.append(jnp.asarray(axis_weights))
    return RowRelation(indices, source_size=grid.size, valid=valid), tuple(weights)


@checked
def prepare_tensor_point_sbp(
    cloud: PreparedPointCloudDiscretization,
    derivatives: Sequence[PreparedSBPOperator],
    /,
) -> PointSBPDerivatives:
    """Bridge native bounded tensor SBP families to their exact identified cloud.

    ``cloud.plan.point_ids`` declares the permutation from cloud rows to the
    native grid's C-order flattened rows. Coordinate columns follow grid-axis
    order. Positive volume norms and oriented physical face measures must match;
    no coefficient recipe is inferred from an SBP identity.
    """
    owners = tuple(derivatives)
    norm, rows = _tensor_sbp_geometry(cloud, owners)
    relation, weights = _tensor_sbp_sparse_rows(owners, rows)
    green, conservation = sbp_identity_residuals(
        np.asarray(relation.source_indices),
        np.asarray(relation.valid),
        tuple(np.asarray(value) for value in weights),
        np.asarray(cloud.quadrature_weights),
        np.asarray(cloud.plan.boundary_quadrature_weights),
        np.asarray(cloud.plan.boundary_normals),
    )
    binding = _sbp_binding_id(cloud, relation, weights)
    result = PointSBPDerivatives(
        relation=relation,
        weights=weights,
        status=jnp.zeros((len(owners),), dtype=jnp.int32),
        certificate_valid=jnp.ones((len(owners),), dtype=jnp.bool_),
        reproduction_residual=jnp.asarray(
            [
                value.operator.consistency_report.maximum_moment_residual
                for value in owners
            ]
        ),
        green_residual=jnp.full((len(owners),), green),
        conservation_residual=jnp.full((len(owners),), conservation),
        deviation=jnp.zeros((len(owners),), dtype=jnp.float64),
        program_results=(),
        tolerance=1e-7,
        reproduction_degree=min(value.closure_order for value in norm.evidence),
        result_id=canonical_fingerprint(
            {
                "kind": "tensor-point-sbp",
                "binding": binding,
                "norm": norm.norm_id,
            }
        ),
        cloud_id=cloud.prepared_id,
        binding_id=binding,
        native_derivatives=owners,
        native_binding_id=_native_sbp_binding_id(owners, binding, relation, weights),
    )
    result.bind(cloud)
    return result


def _monomial_exponents(dimension: int, degree: int, /) -> np.ndarray:
    exponents = [
        exponent
        for exponent in np.ndindex(*((degree + 1,) * dimension))
        if sum(exponent) <= degree
    ]
    return np.asarray(sorted(exponents, key=lambda e: (sum(e), tuple(-v for v in e))))


def _symmetric_pattern(
    indices: np.ndarray, valid: np.ndarray, count: int, /
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.repeat(np.arange(count), indices.shape[1])[valid.reshape(-1)]
    columns = indices.reshape(-1)[valid.reshape(-1)]
    pairs = np.concatenate(
        (
            np.stack((rows, columns), axis=1),
            np.stack((columns, rows), axis=1),
            np.stack((np.arange(count), np.arange(count)), axis=1),
        )
    )
    pairs = np.unique(pairs, axis=0)
    return pairs[:, 0], pairs[:, 1]


def prepare_point_sbp_derivatives(
    discretization: PreparedPointCloudDiscretization,
    /,
    *,
    reproduction_degree: int,
    policy: ConvexSolvePolicy | None = None,
    tolerance: float = 1e-7,
    maximum_variables: int = 200_000,
) -> PointSBPDerivatives:
    """Explicitly selected constrained point-SBP derivative preparation.

    Requires the cloud's positive volume cubature and boundary measure/normals.
    The diagonal-norm identity forces the volume cubature to integrate degree
    ``2 * reproduction_degree - 1`` exactly; clouds whose cubature cannot do so
    are reported infeasible by the conic owner, not repaired.
    """
    from ...optim import (
        ConicProgram,
        ConvexSolvePolicy,
        ConvexTermination,
        NativeHomogeneousConic,
        solve_conic_program,
        ZeroCone,
    )

    if not isinstance(discretization, PreparedPointCloudDiscretization):
        raise TypeError("discretization must be PreparedPointCloudDiscretization.")
    degree = int(reproduction_degree)
    if degree < 1 or degree > discretization.plan.stencil.polynomial_degree:
        raise ValueError(
            "reproduction_degree must lie between one and the stencil degree."
        )
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("SBP tolerance must be finite and positive.")
    boundary_weights = discretization.plan.boundary_quadrature_weights
    if boundary_weights is None:
        raise ValueError("Point SBP preparation requires boundary_quadrature_weights.")
    if any(discretization.plan.address.periodic_axes):
        raise ValueError(
            "Point SBP preparation uses Euclidean moment offsets; periodic clouds are unsupported."
        )
    solve_policy = (
        ConvexSolvePolicy(
            NativeHomogeneousConic(),
            termination=ConvexTermination(absolute=threshold, maximum_steps=200),
            failure=FailurePolicy("status"),
        )
        if policy is None
        else policy
    )
    if not isinstance(solve_policy, ConvexSolvePolicy):
        raise TypeError("policy must be a ConvexSolvePolicy.")
    points = np.asarray(discretization.points)
    count, dimension = points.shape
    mass = np.asarray(discretization.quadrature_weights)
    boundary = np.asarray(boundary_weights)
    normals = np.asarray(discretization.plan.boundary_normals)
    relation = discretization.relation
    rows, columns = _symmetric_pattern(
        np.asarray(relation.source_indices), np.asarray(relation.valid), count
    )
    variables = rows.size
    if variables > maximum_variables:
        raise ValueError("Point SBP preparation exceeds maximum_variables.")
    exponents = _monomial_exponents(dimension, degree)
    features = exponents.shape[0]
    scale = np.asarray(discretization.stencils.neighborhood.row_scale)
    offsets = (points[columns] - points[rows]) / scale[rows][:, None]
    moments = np.prod(offsets[:, None, :] ** exponents[None, :, :], axis=-1)
    reproduction_rows = rows[:, None] * features + np.arange(features)[None, :]
    pair_keys = np.minimum(rows, columns) * count + np.maximum(rows, columns)
    unique_pairs, pair_index = np.unique(pair_keys, return_inverse=True)
    pair_ids = count * features + pair_index
    # Pair rows are M_i D_ij + M_j D_ji = (B n)_ij, scaled by the mean pair mass;
    # a diagonal pair has one variable carrying 2 M_i D_ii.
    pair_scale = 0.5 * (mass[rows] + mass[columns])
    pair_coefficient = np.where(rows == columns, 2.0, 1.0) * mass[rows] / pair_scale
    constraint_rows = np.concatenate((reproduction_rows.reshape(-1), pair_ids))
    constraint_columns = np.concatenate(
        (np.repeat(np.arange(variables), features), np.arange(variables))
    )
    constraint_values = np.concatenate((moments.reshape(-1), pair_coefficient))
    constraints = count * features + unique_pairs.size
    variable_space = ArraySpace(
        (variables,),
        dtype=np.float64,
        space_id=f"{discretization.prepared_id}:sbp-entries",
    )
    constraint_space = ArraySpace(
        (constraints,),
        dtype=np.float64,
        space_id=f"{discretization.prepared_id}:sbp-equations",
    )
    matrix = SparseCoordinateOperator(
        EdgeRelation(
            constraint_columns.astype(np.int32),
            constraint_rows.astype(np.int32),
            source_size=variables,
            target_size=constraints,
        ),
        constraint_values,
        source=variable_space,
        target=constraint_space,
    )
    diagonal = np.arange(variables, dtype=np.int32)
    quadratic = SparseCoordinateOperator(
        EdgeRelation(diagonal, diagonal, source_size=variables, target_size=variables),
        np.ones(variables, dtype=np.float64),
        source=variable_space,
        target=variable_space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    route_rows = np.repeat(np.arange(count), relation.width)
    valid_routes = np.asarray(relation.valid).reshape(-1)
    route_keys = (route_rows * count + np.asarray(relation.source_indices).reshape(-1))[
        valid_routes
    ]
    variable_keys = rows * count + columns
    location = np.searchsorted(variable_keys, route_keys)
    if not np.array_equal(variable_keys[location], route_keys):
        raise RuntimeError("Symmetric SBP pattern lost an admitted stencil route.")
    width = int(np.max(np.bincount(rows, minlength=count)))
    slot = np.arange(variables) - np.searchsorted(rows, rows)
    pattern_indices = np.zeros((count, width), dtype=np.int32)
    pattern_valid = np.zeros((count, width), dtype=np.bool_)
    pattern_indices[rows, slot] = columns
    pattern_valid[rows, slot] = True
    axis_weights: list[Array] = []
    statuses: list[int] = []
    certificates: list[bool] = []
    reproduction_residuals: list[float] = []
    deviations: list[float] = []
    results: list[ConvexProgramResult] = []
    for axis, (first, _) in enumerate(discretization.derivative_weights):
        initial = np.zeros(variables, dtype=np.float64)
        initial[location] = np.asarray(first).reshape(-1)[valid_routes]
        target = np.zeros((count, features), dtype=np.float64)
        linear = np.all(exponents == np.eye(dimension, dtype=np.int64)[axis], axis=1)
        target[:, linear] = 1.0 / scale[:, None]
        diagonal_rhs = np.zeros(unique_pairs.size, dtype=np.float64)
        on_diagonal = rows == columns
        diagonal_rhs[pair_index[on_diagonal]] = (
            boundary[rows[on_diagonal]]
            * normals[rows[on_diagonal], axis]
            / pair_scale[on_diagonal]
        )
        rhs = np.concatenate((target.reshape(-1), diagonal_rhs))
        program = ConicProgram(
            quadratic,
            -jnp.asarray(initial),
            matrix,
            jnp.asarray(rhs),
            ZeroCone(constraints),
            problem_id=f"{discretization.prepared_id}:point-sbp:{axis}",
            convexity_evidence="construction",
        )
        result = solve_conic_program(program, policy=solve_policy)
        entries = np.asarray(result.primal)
        reproduction = np.zeros((count, features), dtype=np.float64)
        np.add.at(reproduction, rows, entries[:, None] * moments)
        reproduction_residuals.append(
            float(np.max(np.abs(reproduction - target) * scale[:, None]))
        )
        dense = np.zeros((count, width), dtype=np.float64)
        dense[rows, slot] = entries
        axis_weights.append(jnp.asarray(dense))
        statuses.append(int(np.asarray(result.status)))
        certificates.append(bool(np.asarray(result.certificate.dual_ray_valid)))
        deviations.append(float(np.linalg.norm(entries - initial)))
        results.append(result)
    green, conservation = zip(
        *(
            sbp_identity_residuals(
                pattern_indices,
                pattern_valid,
                (np.asarray(weights),),
                mass,
                boundary,
                normals[:, axis : axis + 1],
            )
            for axis, weights in enumerate(axis_weights)
        ),
        strict=True,
    )
    return PointSBPDerivatives(
        relation=RowRelation(pattern_indices, source_size=count, valid=pattern_valid),
        weights=tuple(axis_weights),
        status=jnp.asarray(statuses, dtype=jnp.int32),
        certificate_valid=jnp.asarray(certificates),
        reproduction_residual=jnp.asarray(reproduction_residuals),
        green_residual=jnp.asarray(green),
        conservation_residual=jnp.asarray(conservation),
        deviation=jnp.asarray(deviations),
        program_results=tuple(results),
        tolerance=threshold,
        reproduction_degree=degree,
        cloud_id=discretization.prepared_id,
        binding_id=_sbp_binding_id(
            discretization,
            RowRelation(pattern_indices, source_size=count, valid=pattern_valid),
            tuple(axis_weights),
        ),
        result_id=canonical_fingerprint(
            {
                "kind": "point-sbp-derivatives",
                "cloud": discretization.prepared_id,
                "degree": degree,
                "policy": solve_policy.policy_id,
                "tolerance": threshold,
                "weights": array_tree_fingerprint(
                    tuple(np.asarray(w) for w in axis_weights)
                ),
            }
        ),
    )


def _lagrange_weights(nodes: np.ndarray, parameters: np.ndarray, /) -> np.ndarray:
    """Values ``(P, Q)`` of the Lagrange basis on ``nodes`` at ``parameters``."""
    weights = np.ones((parameters.shape[0], nodes.shape[0]), dtype=np.float64)
    for node in range(nodes.shape[0]):
        for other in range(nodes.shape[0]):
            if other != node:
                weights[:, node] *= (parameters - nodes[other]) / (
                    nodes[node] - nodes[other]
                )
    return weights


@final
class PointBoundaryCharts(StrictModule, NonTrainableState):
    """Oriented straight boundary charts whose authoritative sites are cloud points.

    The facet partition, outward normals, and physical measures come from a
    geometry ``BoundaryAtlas`` sampled with a Gauss--Lobatto--Legendre ``rule``;
    every sample must coincide with one point of the prepared cloud, so no facet
    is ever inferred from point coordinates. The published facet trace of a nodal
    field is the chart-local Lagrange interpolant of degree ``rule.points - 1``
    through those points, and ``measure`` lumps the authoritative sample measures
    onto their points (the nodal boundary quadrature of this trace space). Only
    affinely parameterized straight charts of a 2-D boundary are admitted: the
    trace is then a polynomial of the declared degree in arc length.
    """

    atlas: BoundaryAtlas
    rule: FacetTraceRule
    charts: Array
    nodes: Array
    sites: Array
    normals: Array
    weights: Array
    measure: Array
    support_id: str = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    revision_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: PreparedPointCloudDiscretization,
        atlas: BoundaryAtlas,
        rule: FacetTraceRule,
        /,
        *,
        charts: Sequence[int] | None = None,
        tolerance: float = 1e-10,
    ) -> None:
        from scipy.spatial import cKDTree

        if rule.family != "gauss-lobatto-legendre":
            raise ValueError(
                "Boundary chart traces are anchored at Gauss-Lobatto-Legendre sites, "
                "which include the chart end points."
            )
        if atlas.reference_dimension != 1 or atlas.ambient_dimension != 2:
            raise ValueError(
                "Point boundary charts are straight edges of a 2-D boundary atlas."
            )
        if discretization.spatial_dimension != 2:
            raise ValueError("Point boundary charts require a 2-D point cloud.")
        threshold = float(tolerance)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        samples = sample_boundary_atlas(atlas, rule, "edge", charts=charts)
        selected = (
            np.arange(atlas.num_charts, dtype=np.int32)
            if charts is None
            else np.asarray(charts, dtype=np.int32)
        )
        if np.unique(selected).size != selected.size:
            raise ValueError("charts must be distinct boundary chart indices.")
        count, points = selected.size, rule.points
        sites = np.asarray(samples.points).reshape((count, points, 2))
        normals = np.asarray(samples.normals).reshape((count, points, 2))
        weights = np.asarray(samples.measure).reshape((count, points))
        reference, reference_weights = rule.reference("edge")
        parameters = np.asarray(reference)[:, 0]
        cloud = np.asarray(discretization.points)
        scale = float(np.max(np.ptp(cloud, axis=0)))
        start, stop = sites[:, :1, :], sites[:, -1:, :]
        lengths = np.linalg.norm(stop[:, 0] - start[:, 0], axis=1)
        affine = start + parameters[None, :, None] * (stop - start)
        if (
            np.any(lengths <= threshold * scale)
            or np.max(np.abs(sites - affine)) > threshold * scale
        ):
            raise ValueError(
                "Boundary charts must be nondegenerate straight affinely parameterized edges."
            )
        jacobian = weights / np.asarray(reference_weights)[None, :]
        if np.max(np.abs(jacobian - lengths[:, None])) > threshold * max(scale, 1.0):
            raise ValueError(
                "Boundary chart measures must be affine arc-length measures."
            )
        if np.max(np.abs(normals - normals[:, :1, :])) > threshold:
            raise ValueError("A straight boundary chart carries one outward normal.")
        distance, index = cKDTree(cloud).query(sites.reshape((-1, 2)))
        if np.max(distance) > threshold * scale:
            raise ValueError(
                "Every authoritative boundary site must coincide with a cloud point; "
                "the trace never interpolates between unrelated points."
            )
        nodes = np.asarray(index, dtype=np.int32).reshape((count, points))
        if any(np.unique(row).size != points for row in nodes):
            raise ValueError("The sites of one chart must be distinct cloud points.")
        measure = np.bincount(
            nodes.reshape(-1), weights=weights.reshape(-1), minlength=cloud.shape[0]
        )
        self.atlas = atlas
        self.rule = rule
        self.charts = jnp.asarray(selected)
        self.nodes = jnp.asarray(nodes)
        self.sites = jnp.asarray(sites)
        self.normals = jnp.asarray(normals)
        self.weights = jnp.asarray(weights)
        self.measure = jnp.asarray(measure)
        self.support_id = discretization.prepared_id
        self.entity_set_id = canonical_fingerprint(
            {
                "kind": "point-boundary-charts",
                "atlas": atlas.source_id,
                "cloud": discretization.prepared_id,
                "charts": array_tree_fingerprint(selected),
            }
        )
        self.revision_id = canonical_fingerprint(
            {
                "kind": "point-boundary-chart-revision",
                "charts": self.entity_set_id,
                "samples": samples.samples_id,
                "nodes": array_tree_fingerprint(nodes),
            }
        )

    @property
    def trace_degree(self) -> int:
        return self.rule.points - 1

    @property
    def support_rows(self) -> np.ndarray:
        """Host cloud rows on which the chart traces depend."""
        return np.unique(np.asarray(self.nodes))

    def domain(self, charts: ConvertibleToArray | None = None, /) -> IntegrationDomain:
        """Exterior-facet domain of the selected charts (all by default)."""
        selected = np.asarray(self.charts) if charts is None else np.asarray(charts)
        if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError("charts must be a rank-1 array of integer chart indices.")
        selected = selected.astype(np.int32, copy=False)
        positions = self._positions(selected)
        return IntegrationDomain(
            "exterior_facet",
            selected,
            self.support_id,
            self.entity_set_id,
            owner_cells=np.asarray(self.nodes)[positions, 0],
        )

    def _positions(self, charts: np.ndarray, /) -> np.ndarray:
        known = np.asarray(self.charts)
        order = np.argsort(known)
        location = np.searchsorted(known[order], charts)
        location = np.minimum(location, known.size - 1)
        if charts.size == 0 or not np.array_equal(known[order][location], charts):
            raise ValueError("The domain selects charts this boundary does not own.")
        return order[location]

    @checked
    def prepare_trace(
        self,
        space: ArraySpace,
        domain: IntegrationDomain,
        rule: FacetTraceRule,
        /,
        *,
        owner_id: str,
        field_space_id: str,
    ) -> PreparedTraceAction:
        """Value trace of nodal coefficients at the GLL sites of ``rule``.

        The route evaluates the chart interpolant through the authoritative
        sites; its exact coordinate transpose is the scatter of that gather.
        """
        if space.shape != self.measure.shape:
            raise ValueError("Chart traces act on nodal coefficients of this cloud.")
        if (
            domain.kind != "exterior_facet"
            or domain.support_id != self.support_id
            or domain.entity_set_id != self.entity_set_id
        ):
            raise ValueError("The domain is not an exterior domain of these charts.")
        if rule.family != "gauss-lobatto-legendre":
            raise ValueError("Chart traces are prepared on Gauss-Lobatto-Legendre sites.")
        selected = np.asarray(domain.entity_indices)
        positions = self._positions(selected)
        nodes = np.asarray(self.nodes)[positions]
        if not np.array_equal(np.asarray(domain.owner_cells), nodes[:, 0]):
            raise ValueError("The domain's owner routes are not these charts' points.")
        samples = sample_boundary_atlas(self.atlas, rule, "edge", charts=selected)
        count, points = selected.size, rule.points
        authoritative = np.asarray(self.rule.reference("edge")[0])[:, 0]
        requested = np.asarray(rule.reference("edge")[0])[:, 0]
        basis = _lagrange_weights(authoritative, requested)
        route = SideGatherRoute(
            nodes,
            np.broadcast_to(basis, (count, points, basis.shape[1])).copy(),
            coefficient_shape=space.shape,
            mode="componentwise",
            value_shape=(),
        )
        descriptor = SideActionDescriptor(
            owner_id=owner_id,
            field_space_id=field_space_id,
            quantity="value",
            representation="quadrature-values",
            orientation="unoriented",
            approximation="exact",
            side="owner",
            domain=domain,
            revision_id=self.revision_id,
            rule=rule,
            trace_degree=self.trace_degree,
            quadrature_exact_degree=rule.exact_degree("edge"),
        )
        return PreparedTraceAction(
            descriptor,
            route,
            space,
            sites=np.asarray(samples.points).reshape((count, points, 2)),
            weights=np.asarray(samples.measure).reshape((count, points)),
            normals=np.asarray(samples.normals).reshape((count, points, 2)),
            support_rows=np.unique(nodes),
        )


__all__ = [
    "PointBoundaryCharts",
    "PointBoundarySamples",
    "PointDerivativeFamily",
    "PointGhostLayerEvidence",
    "PointGhostLayerPlan",
    "PointInterfaceCondition",
    "PointSBPDerivatives",
    "PointSideAdmissionEvidence",
    "PointSideSupportPlan",
    "PreparedPointGhostLayer",
    "PreparedPointSideSupport",
    "prepare_point_sbp_derivatives",
    "prepare_tensor_point_sbp",
    "sample_boundary_atlas",
    "sample_cubature_atlas",
]
