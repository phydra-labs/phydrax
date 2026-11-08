#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-topology design realization of a periodic trigonometric lattice volume mesh.

A certified periodic tetrahedral mesh of a `TrigonometricLevelSet` region (or a
two-phase mesh whose region blocks are named) is frozen as a topology. Its
lattice parameters become `DesignState` parameters of one feature:

* ``level`` — the level ``c``;
* ``half_width`` — the sheet half-width ``tau`` (sheet regions only);
* ``cell_scale`` — positive per-axis factors ``s`` of the lattice vectors,
  ``vectors(s) = diag(s) @ vectors``.

`PeriodicImplicitRealizationPlan.realize` moves the interface vertex orbits
onto the design's boundary levels with the canonical fixed-anchor normal-gauge
projection (`ImplicitPointProjectionPlan`) in reduced lattice coordinates, where
the field ``F(u)`` does not depend on the cell scale. Every other orbit keeps its
reduced coordinates; Cartesian coordinates are ``origin + u @ vectors(s)``. The
result is differentiable in every parameter: projected points carry the
implicit-function tangent ``-dF/dp grad F / |grad F|^2`` and the cell scale is a
linear map.

Sensitivities hold only while the realized mesh keeps the frozen topology. Every
violation is an event that sets ``refresh_required`` (and poisons derivatives):

* ``CRITICAL_GAP_EXIT`` — a boundary level leaves the open interval between the
  source's certified critical-value enclosures that contains its base value.
  Inside one such interval the level sets are diffeomorphic (Morse theory on the
  three-torus), so the quotient topology is unchanged; outside it is not proved.
* ``SIGN_PATTERN_CHANGED`` — a non-interface orbit changed side (the surface
  swept through a mesh vertex).
* ``INVERTED_CELL`` — a realized tetrahedron is not positively oriented.
* ``PROJECTION_FAILED`` — the projection plan's own status is not success.

`PeriodicImplicitRealizationPlan.assess` certifies an accepted realization on
the host with `MeshMotionMonitor` (Bernstein validity, quality, displacement and
the projection root residual as boundary residual). The assessment is made at
the base lattice; the scaled mesh is its image under ``diag(s)`` with ``s > 0``,
an orientation-preserving linear map that preserves P1 validity.
"""

from __future__ import annotations

import dataclasses
from enum import IntFlag
from typing import final, NoReturn

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._admissibility import guard_derivative_validity
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..discretization._periodic_exact import apply_periodic
from ..discretization.fem._periodic_cell import PeriodicSimplexSpace, simplex_gradients
from ..geometry._capabilities import GeometryCapability
from ..geometry._certificate import (
    DistanceSemantics,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from ..geometry._contracts import CompiledGeometry, GeometryKernel, GeometryKind
from ..geometry._validity import GeometryValidityEvidence
from ..geometry.analytic._trigonometric_level_set import (
    TrigonometricLevelSet,
    TrigonometricRegion,
    TrigonometricTable,
)
from ..geometry.design._schema import (
    DesignState,
    ParameterId,
    ParameterSchema,
    ParameterSpec,
)
from ..geometry.implicit._policy import ImplicitProjectionPolicy
from ..geometry.implicit._projection import (
    ImplicitPointProjectionEvidence,
    ImplicitPointProjectionPlan,
)
from ..typing import PRNGKey
from ._motion import MeshMotionAssessment, MeshMotionMonitor, MeshMotionMonitorPolicy
from ._periodic_implicit_volume import PeriodicImplicitVolume


class PeriodicImplicitRealizationStatus(IntFlag):
    """Topology events of a fixed-topology periodic implicit realization."""

    SUCCESS = 0
    CRITICAL_GAP_EXIT = 1
    SIGN_PATTERN_CHANGED = 2
    INVERTED_CELL = 4
    PROJECTION_FAILED = 8


_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _boundary_levels(
    region: TrigonometricRegion, level: Array, half_width: Array | None, /
) -> Array:
    if region == "sheet":
        assert half_width is not None
        return jnp.stack((level - half_width, level + half_width))
    return level[None]


class _DesignLevelKernel(GeometryKernel):
    """Reduced-coordinate level field whose level and half-width are design state."""

    table: TrigonometricTable
    gaps: Array
    region: TrigonometricRegion = eqx.field(static=True)
    level_index: int = eqx.field(static=True)
    half_width_index: int | None = eqx.field(static=True)
    lipschitz: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        table: TrigonometricTable,
        gaps: np.ndarray,
        /,
        *,
        region: TrigonometricRegion,
        level_index: int,
        half_width_index: int | None,
        lipschitz: float,
        source_id: str,
    ) -> None:
        self.table = table
        self.gaps = jnp.asarray(gaps, dtype=jnp.float64)
        self.region = region
        self.level_index = level_index
        self.half_width_index = half_width_index
        self.lipschitz = lipschitz
        self.source_id = source_id

    def _parameters(self, state: DesignState, /) -> tuple[Array, Array | None]:
        level = state.values[self.level_index]
        width = (
            None if self.half_width_index is None else state.values[self.half_width_index]
        )
        return level, width

    def boundary_levels(self, state: DesignState, /) -> Array:
        level, width = self._parameters(state)
        return _boundary_levels(self.region, level, width)

    def gap_margins(self, state: DesignState, /) -> Array:
        levels = self.boundary_levels(state)
        return jnp.minimum(levels - self.gaps[:, 0], self.gaps[:, 1] - levels)

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 3

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return frozenset({GeometryCapability.REGION_QUERY})

    @property
    def field_certificate(self) -> FieldCertificate:
        return FieldCertificate(
            zero_set_accuracy=ZeroSetAccuracy.EXACT,
            sign_reliability=SignReliability.RELIABLE,
            distance_semantics=DistanceSemantics.LEVEL_SET,
            regularity=FieldRegularity.SMOOTH,
            safe_step_factor=None,
            validity_region=(
                "reduced lattice coordinates of a trigonometric nodal level set with "
                "design level and half-width inside their certified critical gaps"
            ),
            parameter_differentiable=True,
            provenance=("trigonometric_design_level", self.table.table_id),
            lipschitz_upper_bound=self.lipschitz,
        )

    def geometry_validity(self, state: DesignState, /) -> GeometryValidityEvidence:
        margins = self.gap_margins(state)
        return GeometryValidityEvidence(
            finite=jnp.all(jnp.isfinite(margins)),
            conditions_satisfied=jnp.all(margins > 0.0),
            resolved=True,
            margins=margins,
            margin_names=tuple(
                f"critical_gap_{index}" for index in range(margins.shape[0])
            ),
            contract_id="trigonometric_design_level",
        )

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        value = self.table.evaluate_reduced(points)
        level, width = self._parameters(state)
        match self.region:
            case "network":
                return value - level
            case "complement":
                return level - value
            case "sheet":
                assert width is not None
                offset = value - level
                return (jnp.square(offset) - jnp.square(width)) / (2.0 * width)

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state: DesignState, points: Array, /) -> NoReturn:
        del state, points
        raise NotImplementedError("Design level fields expose only the boundary field.")

    def bounds(self, state: DesignState, /) -> Array:
        del state
        return jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=jnp.float64)

    def measure(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("Measure the realized mesh, not the design field.")

    def boundary_measure(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("Measure the realized mesh, not the design field.")

    def sample_interior(
        self, state: DesignState, num_points: int, /, *, key: PRNGKey, plan: object = None
    ) -> NoReturn:
        del state, num_points, key, plan
        raise NotImplementedError("Design level fields have no sampler.")

    def sample_boundary(
        self, state: DesignState, num_points: int, /, *, key: PRNGKey
    ) -> NoReturn:
        del state, num_points, key
        raise NotImplementedError("Design level fields have no sampler.")

    def boundary_atlas(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("Design level fields have no boundary atlas.")


class PeriodicImplicitRealizationEventError(ValueError):
    """Sensitivities were requested from a realization with a topology event."""

    def __init__(self, status: PeriodicImplicitRealizationStatus, /) -> None:
        names = [flag.name for flag in PeriodicImplicitRealizationStatus if flag & status]
        super().__init__(
            "Periodic implicit realization left its fixed-topology window "
            f"({', '.join(str(name) for name in names)}); its sensitivities are invalid "
            "and the mesh must be refreshed."
        )
        self.status = status


@final
class PeriodicImplicitRealization(StrictModule):
    """One realized design of a fixed-topology periodic lattice mesh.

    ``representative_coordinates`` ``(dofs, 3)`` and ``lattice_vectors``
    ``(3, 3)`` feed `PeriodicSimplexSpace` consumers directly;
    ``reduced_coordinates`` are the orbit representatives in lattice units.
    ``volume_fraction`` is the realized region measure over the cell measure.
    ``gap_margins`` are the distances of the boundary levels to the ends of their
    certified critical gaps.

    Contract: ``status`` is always returned with the realization, and consumers
    must check it. A non-success status is an event: ``refresh_required`` is
    set and every derivative of the realized arrays is NaN, inside or outside
    ``jit``. On the host, `require_sensitivities` raises
    `PeriodicImplicitRealizationEventError` for such a realization.
    """

    representative_coordinates: Array
    reduced_coordinates: Array
    lattice_vectors: Array
    volume_fraction: Array
    gap_margins: Array
    minimum_cell_volume: Array
    projection: ImplicitPointProjectionEvidence
    status: Array
    plan_id: str = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    @property
    def accepted(self) -> Array:
        return self.status == int(PeriodicImplicitRealizationStatus.SUCCESS)

    @property
    def refresh_required(self) -> Array:
        return ~self.accepted

    def require_sensitivities(self, /) -> PeriodicImplicitRealization:
        """Host check before consuming sensitivities: raise on any event bit."""
        status = PeriodicImplicitRealizationStatus(int(np.asarray(self.status)))
        if status != PeriodicImplicitRealizationStatus.SUCCESS:
            raise PeriodicImplicitRealizationEventError(status)
        return self


def _gap(lower: np.ndarray, upper: np.ndarray, level: float, /) -> tuple[float, float]:
    below = upper[upper < level]
    above = lower[lower > level]
    if not below.size or not above.size or np.any((lower <= level) & (level <= upper)):
        raise ValueError(
            f"Boundary level {level!r} is not inside a certified critical gap."
        )
    return float(np.max(below)), float(np.min(above))


@final
class PeriodicImplicitRealizationPlan(StrictModule, NonTrainableState):
    """Frozen periodic lattice mesh topology with a differentiable design realization.

    Build with `prepare_periodic_implicit_realization`. ``schema`` and
    ``base_state`` hold the ``level``, optional ``half_width`` and
    ``cell_scale`` parameters of ``feature_id``.
    """

    space: PeriodicSimplexSpace
    projection: ImplicitPointProjectionPlan
    monitor: MeshMotionMonitor
    base_state: DesignState
    base_reduced: Array
    base_vectors: Array
    origin: Array
    interface_dofs: Array
    interior_dofs: Array
    interior_inside: Array
    region_cells: Array
    schema: ParameterSchema = eqx.field(static=True)
    scale_index: int = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        space: PeriodicSimplexSpace,
        projection: ImplicitPointProjectionPlan,
        monitor: MeshMotionMonitor,
        base_state: DesignState,
        base_reduced: np.ndarray,
        interface_dofs: np.ndarray,
        interior_inside: np.ndarray,
        region_cells: np.ndarray,
        scale_index: int,
        feature_id: str,
    ) -> None:
        interface = np.asarray(interface_dofs, dtype=np.int64)
        interior = np.setdiff1d(np.arange(space.dof_count), interface)
        self.space = space
        self.projection = projection
        self.monitor = monitor
        self.base_state = base_state
        self.base_reduced = jnp.asarray(base_reduced, dtype=jnp.float64)
        self.base_vectors = space.lattice_vectors
        self.origin = jnp.asarray(space.cell.origin, dtype=jnp.float64)
        self.interface_dofs = jnp.asarray(interface, dtype=jnp.int32)
        self.interior_dofs = jnp.asarray(interior, dtype=jnp.int32)
        self.interior_inside = jnp.asarray(interior_inside, dtype=jnp.bool_)
        self.region_cells = jnp.asarray(region_cells, dtype=jnp.bool_)
        self.schema = base_state.schema
        self.scale_index = scale_index
        self.feature_id = feature_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-implicit-realization",
                "space": space.space_id,
                "projection": projection.plan_id,
                "schema": base_state.schema.schema_id,
                "region_cells": np.asarray(region_cells, dtype=np.bool_),
            }
        )

    def parameter_id(self, name: str, /) -> ParameterId:
        """Stable identity of ``level``, ``half_width`` or ``cell_scale``."""
        return ParameterId(self.feature_id, name)

    def realize(self, state: DesignState, /) -> PeriodicImplicitRealization:
        """Differentiable realization of ``state`` with its topology events."""
        if not isinstance(state, DesignState) or state.schema != self.schema:
            raise ValueError("Realization state must use the plan's parameter schema.")
        kernel = self.projection.kernel
        assert isinstance(kernel, _DesignLevelKernel)
        projected = self.projection.realize(state)
        reduced = self.base_reduced.at[self.interface_dofs].set(projected.points)
        scale = state.values[self.scale_index]
        vectors = scale[:, None] * self.base_vectors
        cartesian = self.origin + reduced @ vectors
        interior_field = kernel.boundary_field(state, reduced[self.interior_dofs])
        sign_kept = jnp.all((interior_field < 0.0) == self.interior_inside) & jnp.all(
            interior_field != 0.0
        )
        volumes, _ = simplex_gradients(self.space.cell_coordinates(cartesian, vectors))
        margins = kernel.gap_margins(state)
        measure = jnp.abs(jnp.linalg.det(vectors))
        fraction = jnp.sum(jnp.where(self.region_cells, volumes, 0.0)) / measure
        flags = (
            (PeriodicImplicitRealizationStatus.CRITICAL_GAP_EXIT, jnp.all(margins > 0.0)),
            (PeriodicImplicitRealizationStatus.SIGN_PATTERN_CHANGED, sign_kept),
            (PeriodicImplicitRealizationStatus.INVERTED_CELL, jnp.all(volumes > 0.0)),
            (PeriodicImplicitRealizationStatus.PROJECTION_FAILED, projected.accepted),
        )
        status = jnp.asarray(0, dtype=jnp.int32)
        for flag, kept in flags:
            status = status | jnp.where(kept, 0, int(flag)).astype(jnp.int32)
        accepted = status == 0
        guarded = guard_derivative_validity(
            (cartesian, reduced, vectors, fraction),
            accepted,
            dependencies=state,
            failure="status",
            message="Periodic implicit realization left its fixed-topology window.",
        )
        return PeriodicImplicitRealization(
            representative_coordinates=guarded[0],
            reduced_coordinates=guarded[1],
            lattice_vectors=guarded[2],
            volume_fraction=guarded[3],
            gap_margins=margins,
            minimum_cell_volume=jnp.min(volumes),
            projection=projected.evidence,
            status=status,
            plan_id=self.plan_id,
            space_id=self.space.space_id,
        )

    def assess(self, realization: PeriodicImplicitRealization, /) -> MeshMotionAssessment:
        """Host certification of a realization's mesh at the base lattice."""
        if realization.plan_id != self.plan_id:
            raise ValueError("The realization belongs to a different plan.")
        mesh = self.space.mesh
        topology = mesh.periodic_topology
        assert topology is not None
        reduced = np.asarray(realization.reduced_coordinates, dtype=np.float64)
        representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
        rows = np.asarray(self.space.representative_rows, dtype=np.int64)
        coordinates = np.array(mesh.coordinates, dtype=np.float64)
        moved = np.asarray(self.origin) + reduced @ np.asarray(self.base_vectors)
        coordinates[rows] = moved
        referenced = np.isin(representatives, rows)
        coordinates[referenced] = apply_periodic(
            topology.cell,
            coordinates[representatives[referenced]],
            np.asarray(topology.vertex_shifts)[referenced],
        )
        return self.monitor.assess(
            coordinates, boundary_residual=float(realization.projection.root_residual)
        )


def _interface_faces(
    space: PeriodicSimplexSpace, region_cells: np.ndarray, /
) -> np.ndarray:
    """Orbits of quotient faces bounded by exactly one region cell."""
    dofs = np.asarray(space.cell_dofs, dtype=np.int64)
    shifts = np.asarray(space.cell_shifts, dtype=np.int64)
    if np.any(np.sort(dofs, axis=1)[:, 1:] == np.sort(dofs, axis=1)[:, :-1]):
        raise ValueError("A realization cell may not repeat one vertex orbit.")
    keys = []
    owners = []
    for face in _FACES:
        corner_dofs = dofs[:, face]
        order = np.argsort(corner_dofs, axis=1)
        sorted_dofs = np.take_along_axis(corner_dofs, order, axis=1)
        sorted_shifts = np.take_along_axis(shifts[:, face], order[:, :, None], axis=1)
        relative = sorted_shifts[:, 1:] - sorted_shifts[:, :1]
        keys.append(np.concatenate((sorted_dofs, relative.reshape((-1, 6))), axis=1))
        owners.append(sorted_dofs)
    key = np.concatenate(keys)
    face_dofs = np.concatenate(owners)
    in_region = np.tile(region_cells, len(_FACES))
    unique, inverse = np.unique(key, axis=0, return_inverse=True)
    counts = np.bincount(
        inverse.reshape(-1), weights=in_region, minlength=unique.shape[0]
    )
    interface = counts[inverse.reshape(-1)] == 1
    return face_dofs[interface & in_region]


def _same_source(
    recorded: TrigonometricLevelSet, source: TrigonometricLevelSet, /
) -> bool:
    """Exact identity of the region a fill was discovered from."""
    return (
        recorded.table.table_id == source.table.table_id
        and recorded.region == source.region
        and recorded.level == source.level
        and recorded.half_width == source.half_width
        and recorded.feature_id == source.feature_id
        and recorded.cell.cell_id == source.cell.cell_id
    )


def _recorded_projection_policy(
    volume: PeriodicImplicitVolume, source: TrigonometricLevelSet, /
) -> ImplicitProjectionPolicy:
    """Field root tolerance proved by the fill's recorded boundary certificate.

    Every contour vertex lies within reduced distance ``root_error`` of a root,
    and the reduced-coordinate gradient is bounded by ``table.gradient_bound``.
    A level field therefore satisfies ``|F - l| <= L e``. The sheet field
    ``((F - c)^2 - tau^2) / (2 tau)`` satisfies ``|g| <= L e (1 + L e / (2 tau))``.
    The bound only replaces the default tolerance when it is larger.
    """
    default = ImplicitProjectionPolicy()
    bound = float(np.nextafter(source.table.gradient_bound * volume.root_error, np.inf))
    if source.region == "sheet":
        assert source.half_width is not None
        bound = float(
            np.nextafter(bound * (1.0 + bound / (2.0 * source.half_width)), np.inf)
        )
    return dataclasses.replace(default, root_tolerance=max(default.root_tolerance, bound))


def prepare_periodic_implicit_realization(
    mesh: CellMesh | PeriodicImplicitVolume,
    source: TrigonometricLevelSet,
    /,
    *,
    region_blocks: tuple[str, ...] | None = None,
    projection_policy: ImplicitProjectionPolicy | None = None,
    monitor_policy: MeshMotionMonitorPolicy | None = None,
) -> PeriodicImplicitRealizationPlan:
    """Freeze a periodic lattice mesh and bind its lattice parameters as design state.

    ``mesh`` is a periodic tetrahedral mesh of ``source`` on the same lattice,
    or the `PeriodicImplicitVolume` fill that produced it. A fill must record a
    discovery of exactly ``source`` (table, region, level, half-width, feature
    and cell), or it is refused. Without an explicit ``projection_policy``, a
    fill's recorded contour root-error bound sets the projection root
    tolerance (see `_recorded_projection_policy`); a bare mesh uses the default
    policy. ``region_blocks`` names the blocks of the source region (default:
    every block, a region-only mesh). Interface orbits are those of quotient
    faces bounded by exactly one region cell; they must lie on the source's
    boundary levels within the projection root tolerance. Every other orbit
    must lie strictly off the boundary. Each boundary level must sit inside a
    certified critical gap of ``source.critical_levels``.
    """
    if not isinstance(source, TrigonometricLevelSet):
        raise TypeError("source must be a TrigonometricLevelSet.")
    if isinstance(mesh, PeriodicImplicitVolume):
        volume: PeriodicImplicitVolume | None = mesh
        if not _same_source(mesh.surface.source, source):
            raise ValueError(
                "The periodic fill was discovered from a different trigonometric source."
            )
        cells = mesh.mesh
    else:
        volume, cells = None, mesh
    space = PeriodicSimplexSpace(cells)
    if not (
        np.array_equal(np.asarray(space.cell.vectors), np.asarray(source.cell.vectors))
        and np.array_equal(np.asarray(space.cell.origin), np.asarray(source.cell.origin))
    ):
        raise ValueError("The mesh and the source must share one periodic lattice.")
    names = space.block_names if region_blocks is None else tuple(region_blocks)
    region_cells = np.asarray(space.block_mask(names))
    feature = source.feature_id
    critical = source.critical_levels
    lower = np.asarray(critical.value_lower, dtype=np.float64)
    upper = np.asarray(critical.value_upper, dtype=np.float64)
    width = source.half_width if source.region == "sheet" else None
    levels = np.asarray(
        _boundary_levels(
            source.region,
            jnp.asarray(source.level, dtype=jnp.float64),
            None if width is None else jnp.asarray(width, dtype=jnp.float64),
        )
    )
    gaps = np.asarray([_gap(lower, upper, float(level)) for level in levels])
    specs = [
        ParameterSpec(ParameterId(feature, "level"), (), "float64", "trigonometric-level")
    ]
    values: list[Array] = [jnp.asarray(source.level, dtype=jnp.float64)]
    width_index = None
    if width is not None:
        width_index = len(specs)
        specs.append(
            ParameterSpec(
                ParameterId(feature, "half_width"),
                (),
                "float64",
                "sheet-half-width",
                bounds=(0.0, None),
            )
        )
        values.append(jnp.asarray(width, dtype=jnp.float64))
    scale_index = len(specs)
    specs.append(
        ParameterSpec(
            ParameterId(feature, "cell_scale"),
            (3,),
            "float64",
            "lattice-scale",
            bounds=(0.0, None),
        )
    )
    values.append(jnp.ones((3,), dtype=jnp.float64))
    schema = ParameterSchema(tuple(specs))
    state = DesignState(schema, values)
    kernel = _DesignLevelKernel(
        source.table,
        gaps,
        region=source.region,
        level_index=0,
        half_width_index=width_index,
        lipschitz=float(source.table.gradient_bound),
        source_id=feature,
    )
    reduced = np.asarray(
        space.cell.fractional(space.representative_coordinates), dtype=np.float64
    )
    face_dofs = _interface_faces(space, region_cells)
    interface = np.unique(face_dofs)
    if not interface.size:
        raise ValueError("The region mesh has no interface faces to realize.")
    policy = (
        _recorded_projection_policy(volume, source)
        if projection_policy is None and volume is not None
        else ImplicitProjectionPolicy()
        if projection_policy is None
        else projection_policy
    )
    dofs = np.asarray(space.cell_dofs, dtype=np.int64)
    shifts = np.asarray(space.cell_shifts, dtype=np.float64)
    corners = reduced[dofs] + shifts
    lengths = np.concatenate(
        [np.linalg.norm(corners[:, b] - corners[:, a], axis=1) for a, b in _EDGES]
    )
    ends = np.concatenate(
        [
            np.concatenate([dofs[:, a] for a, _ in _EDGES]),
            np.concatenate([dofs[:, b] for _, b in _EDGES]),
        ]
    )
    shortest = np.full((space.dof_count,), np.inf)
    np.minimum.at(shortest, ends, np.concatenate((lengths, lengths)))
    projection = ImplicitPointProjectionPlan(
        CompiledGeometry(kernel, state),
        reduced[interface],
        policy.trust_fraction * shortest[interface],
        policy=policy,
        source_id=f"{feature}:periodic-interface",
    )
    interior = np.setdiff1d(np.arange(space.dof_count), interface)
    field = np.asarray(kernel.boundary_field(state, jnp.asarray(reduced[interior])))
    if np.any(np.abs(field) <= policy.root_tolerance):
        raise ValueError(
            "A non-interface vertex orbit lies on a boundary level; the frozen sign "
            "pattern would be ambiguous."
        )
    monitor = MeshMotionMonitor(space.mesh, policy=monitor_policy)
    return PeriodicImplicitRealizationPlan(
        space=space,
        projection=projection,
        monitor=monitor,
        base_state=state,
        base_reduced=reduced,
        interface_dofs=interface,
        interior_inside=field < 0.0,
        region_cells=region_cells,
        scale_index=scale_index,
        feature_id=feature,
    )


__all__ = [
    "PeriodicImplicitRealization",
    "PeriodicImplicitRealizationEventError",
    "PeriodicImplicitRealizationPlan",
    "PeriodicImplicitRealizationStatus",
    "prepare_periodic_implicit_realization",
]
