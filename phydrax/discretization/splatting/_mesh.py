#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._interpolation import apply_gather_stencil, GatherStencil
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_mesh import CellMesh
from .._measure import DiscreteMeasure
from ..particle._population import ParticlePopulationState
from ._reduction import deposit_routes


MeshPartitionPolicy: TypeAlias = Literal["normalize", "raw"]
MeshSplatBoundaryPolicy: TypeAlias = Literal["reject", "drop"]
MeshSplatGeometryAD: TypeAlias = Literal["piecewise", "frozen"]


class MeshSplatTarget(StrictModule, NonTrainableState):
    """A finite vertex- or cell-valued splat target on a canonical ``CellMesh``."""

    mesh: CellMesh
    measure: DiscreteMeasure
    entity_dimension: int = eqx.field(static=True)
    entity_count: int = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        *,
        entity_dimension: int,
        measure: DiscreteMeasure,
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be a CellMesh.")
        if not isinstance(measure, DiscreteMeasure):
            raise TypeError("measure must be a DiscreteMeasure.")
        dimension = int(entity_dimension)
        if dimension not in (0, mesh.topological_dimension):
            raise ValueError("Mesh splats support vertices or top-dimensional cells.")
        entities = mesh.topology.entities(dimension)
        if measure.support_id != mesh.support.support_id:
            raise ValueError("Mesh splat measure belongs to a different support.")
        if measure.entity_set_id != entities.entity_set_id:
            raise ValueError("Mesh splat measure belongs to a different entity set.")
        weights = np.asarray(measure.weights)
        active = np.asarray(measure.active_mask)
        if weights.shape != (entities.count,) or active.shape != weights.shape:
            raise ValueError("Mesh splat measure must have one value per target entity.")
        if np.any(~np.isfinite(weights[active])) or np.any(weights[active] <= 0.0):
            raise ValueError("Active mesh splat measures must be finite and positive.")
        self.mesh = mesh
        self.measure = measure
        self.entity_dimension = dimension
        self.entity_count = entities.count
        self.target_id = canonical_fingerprint(
            {
                "kind": "mesh-splat-target",
                "mesh": mesh.mesh_id,
                "entity_dimension": dimension,
                "measure": measure.measure_id,
            }
        )

    @property
    def stable_entity_ids(self) -> Array:
        return self.mesh.topology.entities(self.entity_dimension).entity_ids


class MeshSplatRouteEvidence(StrictModule):
    raw_partition_sum: Array
    normalization_factor: Array
    tie_margin: Array
    query_count: Array
    route_overflow: Array
    supported: Array
    derivative_valid: Array
    finite: Array
    complete: Array


class MeshSplatRoutes(StrictModule):
    indices: Array
    weights: Array
    gradients: Array
    valid: Array
    evidence: MeshSplatRouteEvidence


class MeshSplatDepositResult(StrictModule):
    content: Array
    represented_content: Array
    omitted_content: Array
    balance_residual: Array
    evidence: MeshSplatRouteEvidence
    successful: Array


class MeshSplatGatherResult(StrictModule):
    values: Array
    support: Array
    evidence: MeshSplatRouteEvidence
    successful: Array


class SimplicialBarycentricSplatAssignment(StrictModule, NonTrainableState):
    """Frozen-route affine barycentric assignment for triangle/tetrahedron meshes."""

    boundary: MeshSplatBoundaryPolicy = eqx.field(static=True)
    geometry_ad: MeshSplatGeometryAD = eqx.field(static=True)
    tie_tolerance: float = eqx.field(static=True)
    assignment_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        boundary: MeshSplatBoundaryPolicy = "reject",
        geometry_ad: MeshSplatGeometryAD = "piecewise",
        tie_tolerance: float = 1.0e-10,
    ):
        if boundary not in ("reject", "drop"):
            raise ValueError("boundary must be 'reject' or 'drop'.")
        if geometry_ad not in ("piecewise", "frozen"):
            raise ValueError("geometry_ad must be 'piecewise' or 'frozen'.")
        tolerance = float(tie_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tie_tolerance must be finite and nonnegative.")
        self.boundary = boundary
        self.geometry_ad = geometry_ad
        self.tie_tolerance = tolerance
        self.assignment_id = canonical_fingerprint(
            {
                "kind": "simplicial-barycentric-splat",
                "boundary": boundary,
                "geometry_ad": geometry_ad,
                "tie_tolerance": tolerance,
            }
        )

    def prepare(
        self,
        target: MeshSplatTarget,
        particle_positions: ArrayLike,
        active: ArrayLike,
        stable_source_ids: ArrayLike,
        /,
    ) -> PreparedMeshParticleGridSplat:
        if target.entity_dimension != 0:
            raise ValueError("Barycentric splatting requires a vertex target.")
        if len(target.mesh.blocks) != 1:
            raise ValueError("Barycentric splatting requires one simplex block.")
        block = target.mesh.blocks[0]
        expected_kind = (
            "triangle" if target.mesh.topological_dimension == 2 else "tetrahedron"
        )
        if block.cell_kind != expected_kind:
            raise ValueError("Barycentric splatting requires affine simplices.")
        positions = np.asarray(particle_positions)
        if not np.issubdtype(positions.dtype, np.inexact):
            positions = positions.astype(float)
        active_host = np.asarray(active, dtype=bool)
        source_ids = np.asarray(stable_source_ids, dtype=np.int64)
        expected = (positions.shape[0], target.mesh.ambient_dimension)
        if positions.ndim != 2 or positions.shape != expected:
            raise ValueError(
                "particle_positions must have shape (capacity, ambient_dimension)."
            )
        if (
            active_host.shape != (positions.shape[0],)
            or source_ids.shape != active_host.shape
        ):
            raise ValueError(
                "active and stable_source_ids must have particle-capacity shape."
            )
        if np.unique(source_ids).size != source_ids.size:
            raise ValueError("stable_source_ids must be unique.")
        if np.any(~np.isfinite(positions[active_host])):
            raise ValueError("Active particle positions must be finite.")

        coordinates = np.asarray(target.mesh.coordinates)
        cells = np.asarray(block.vertices, dtype=np.int32)
        cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        order = np.argsort(cell_ids, kind="stable")
        cells = cells[order]
        cell_ids = cell_ids[order]
        width = target.mesh.topological_dimension + 1
        particle_count = positions.shape[0]
        route_indices = np.zeros((particle_count, width), dtype=np.int32)
        selected_origins = np.zeros((particle_count, target.mesh.ambient_dimension))
        selected_inverse = np.zeros(
            (
                particle_count,
                target.mesh.topological_dimension,
                target.mesh.ambient_dimension,
            )
        )
        supported = np.zeros((particle_count,), dtype=bool)
        tie_margin = np.full((particle_count,), np.inf)
        query_count = np.zeros((particle_count,), dtype=np.int32)
        tolerance = self.tie_tolerance
        for particle in range(particle_count):
            if not active_host[particle]:
                continue
            candidates: list[tuple[int, float, np.ndarray, np.ndarray]] = []
            for cell_index, vertices in enumerate(cells):
                simplex = coordinates[vertices]
                jacobian = (simplex[1:] - simplex[0]).T
                gram = jacobian.T @ jacobian
                if not np.all(np.isfinite(gram)) or np.linalg.det(gram) <= 0.0:
                    continue
                inverse = np.linalg.solve(gram, jacobian.T)
                reduced = inverse @ (positions[particle] - simplex[0])
                barycentric = np.concatenate(([1.0 - np.sum(reduced)], reduced))
                margin = float(np.min(barycentric))
                if margin >= -tolerance:
                    candidates.append((cell_index, margin, barycentric, inverse))
            query_count[particle] = len(candidates)
            if not candidates:
                continue
            candidates.sort(key=lambda item: int(cell_ids[item[0]]))
            chosen = candidates[0]
            route_indices[particle] = cells[chosen[0]]
            selected_origins[particle] = coordinates[cells[chosen[0], 0]]
            selected_inverse[particle] = chosen[3]
            supported[particle] = True
            if len(candidates) > 1:
                tie_margin[particle] = min(candidate[1] for candidate in candidates)
            else:
                tie_margin[particle] = chosen[1]

        return PreparedMeshParticleGridSplat(
            target=target,
            stable_source_ids=jnp.asarray(source_ids),
            prepared_active=jnp.asarray(active_host),
            route_indices=jnp.asarray(route_indices),
            route_origins=jnp.asarray(selected_origins, dtype=positions.dtype),
            route_inverse=jnp.asarray(selected_inverse, dtype=positions.dtype),
            cell_centers=jnp.empty(
                (0, target.mesh.ambient_dimension), dtype=positions.dtype
            ),
            prepared_supported=jnp.asarray(supported),
            prepared_query_count=jnp.asarray(query_count),
            prepared_tie_margin=jnp.asarray(tie_margin, dtype=positions.dtype),
            support_radius=0.0,
            partition_policy="raw",
            boundary=self.boundary,
            geometry_ad=self.geometry_ad,
            assignment_kind="barycentric",
            assignment_id=self.assignment_id,
        )


class MeshCompactKernelSplatAssignment(StrictModule, NonTrainableState):
    """Fixed-capacity compact Wendland assignment to cell entities."""

    support_radius: float = eqx.field(static=True)
    maximum_entities_per_particle: int = eqx.field(static=True)
    partition_policy: MeshPartitionPolicy = eqx.field(static=True)
    boundary: MeshSplatBoundaryPolicy = eqx.field(static=True)
    geometry_ad: MeshSplatGeometryAD = eqx.field(static=True)
    assignment_id: str = eqx.field(static=True)

    def __init__(
        self,
        support_radius: float,
        maximum_entities_per_particle: int,
        *,
        partition_policy: MeshPartitionPolicy = "normalize",
        boundary: MeshSplatBoundaryPolicy = "reject",
        geometry_ad: MeshSplatGeometryAD = "piecewise",
    ):
        radius = float(support_radius)
        width = int(maximum_entities_per_particle)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("support_radius must be finite and positive.")
        if width <= 0:
            raise ValueError("maximum_entities_per_particle must be positive.")
        if partition_policy not in ("normalize", "raw"):
            raise ValueError("partition_policy must be 'normalize' or 'raw'.")
        if boundary not in ("reject", "drop"):
            raise ValueError("boundary must be 'reject' or 'drop'.")
        if geometry_ad not in ("piecewise", "frozen"):
            raise ValueError("geometry_ad must be 'piecewise' or 'frozen'.")
        self.support_radius = radius
        self.maximum_entities_per_particle = width
        self.partition_policy = partition_policy
        self.boundary = boundary
        self.geometry_ad = geometry_ad
        self.assignment_id = canonical_fingerprint(
            {
                "kind": "mesh-compact-wendland-splat",
                "support_radius": radius,
                "route_width": width,
                "partition_policy": partition_policy,
                "boundary": boundary,
                "geometry_ad": geometry_ad,
            }
        )

    def prepare(
        self,
        target: MeshSplatTarget,
        particle_positions: ArrayLike,
        active: ArrayLike,
        stable_source_ids: ArrayLike,
        /,
    ) -> PreparedMeshParticleGridSplat:
        if target.entity_dimension != target.mesh.topological_dimension:
            raise ValueError("Compact-kernel splatting requires a cell target.")
        positions = np.asarray(particle_positions)
        if not np.issubdtype(positions.dtype, np.inexact):
            positions = positions.astype(float)
        active_host = np.asarray(active, dtype=bool)
        source_ids = np.asarray(stable_source_ids, dtype=np.int64)
        if positions.ndim != 2 or positions.shape[1] != target.mesh.ambient_dimension:
            raise ValueError("particle_positions have the wrong ambient dimension.")
        if (
            active_host.shape != positions.shape[:1]
            or source_ids.shape != active_host.shape
        ):
            raise ValueError(
                "active and stable_source_ids must have particle-capacity shape."
            )
        if np.unique(source_ids).size != source_ids.size:
            raise ValueError("stable_source_ids must be unique.")
        if np.any(~np.isfinite(positions[active_host])):
            raise ValueError("Active particle positions must be finite.")
        centers = []
        cell_ids = []
        for block in target.mesh.blocks:
            local = np.asarray(target.mesh.coordinates)[np.asarray(block.vertices)]
            centers.append(np.mean(local, axis=1))
            cell_ids.append(np.asarray(block.global_ids))
        centers_host = np.concatenate(centers, axis=0)
        cell_ids_host = np.concatenate(cell_ids, axis=0)
        width = self.maximum_entities_per_particle
        indices = np.zeros((positions.shape[0], width), dtype=np.int32)
        supported = np.zeros((positions.shape[0],), dtype=bool)
        query_count = np.zeros((positions.shape[0],), dtype=np.int32)
        for particle in range(positions.shape[0]):
            if not active_host[particle]:
                continue
            distance = np.linalg.norm(centers_host - positions[particle], axis=-1)
            candidates = np.nonzero(distance < self.support_radius)[0]
            candidates = candidates[np.argsort(cell_ids_host[candidates], kind="stable")]
            query_count[particle] = candidates.size
            take = candidates[:width]
            indices[particle, : take.size] = take
            supported[particle] = take.size > 0
        return PreparedMeshParticleGridSplat(
            target=target,
            stable_source_ids=jnp.asarray(source_ids),
            prepared_active=jnp.asarray(active_host),
            route_indices=jnp.asarray(indices),
            route_origins=jnp.empty(
                (0, target.mesh.ambient_dimension), dtype=positions.dtype
            ),
            route_inverse=jnp.empty(
                (0, target.mesh.topological_dimension, target.mesh.ambient_dimension),
                dtype=positions.dtype,
            ),
            cell_centers=jnp.asarray(centers_host, dtype=positions.dtype),
            prepared_supported=jnp.asarray(supported),
            prepared_query_count=jnp.asarray(query_count),
            prepared_tie_margin=jnp.full(
                (positions.shape[0],), jnp.inf, dtype=positions.dtype
            ),
            support_radius=self.support_radius,
            partition_policy=self.partition_policy,
            boundary=self.boundary,
            geometry_ad=self.geometry_ad,
            assignment_kind="compact_wendland",
            assignment_id=self.assignment_id,
        )


class PreparedMeshParticleGridSplat(StrictModule, NonTrainableState):
    """One immutable mesh route epoch with pure fixed-shape deposit/gather."""

    target: MeshSplatTarget
    stable_source_ids: Array
    prepared_active: Array
    route_indices: Array
    route_origins: Array
    route_inverse: Array
    cell_centers: Array
    prepared_supported: Array
    prepared_query_count: Array
    prepared_tie_margin: Array
    support_radius: float = eqx.field(static=True)
    partition_policy: MeshPartitionPolicy = eqx.field(static=True)
    boundary: MeshSplatBoundaryPolicy = eqx.field(static=True)
    geometry_ad: MeshSplatGeometryAD = eqx.field(static=True)
    assignment_kind: str = eqx.field(static=True)
    assignment_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        target: MeshSplatTarget,
        stable_source_ids: Array,
        prepared_active: Array,
        route_indices: Array,
        route_origins: Array,
        route_inverse: Array,
        cell_centers: Array,
        prepared_supported: Array,
        prepared_query_count: Array,
        prepared_tie_margin: Array,
        support_radius: float,
        partition_policy: MeshPartitionPolicy,
        boundary: MeshSplatBoundaryPolicy,
        geometry_ad: MeshSplatGeometryAD,
        assignment_kind: str,
        assignment_id: str,
    ):
        self.target = target
        self.stable_source_ids = stable_source_ids
        self.prepared_active = prepared_active
        self.route_indices = route_indices.astype(jnp.int32)
        self.route_origins = route_origins
        self.route_inverse = route_inverse
        self.cell_centers = cell_centers
        self.prepared_supported = prepared_supported
        self.prepared_query_count = prepared_query_count.astype(jnp.int32)
        self.prepared_tie_margin = prepared_tie_margin
        self.support_radius = float(support_radius)
        self.partition_policy = partition_policy
        self.boundary = boundary
        self.geometry_ad = geometry_ad
        self.assignment_kind = assignment_kind
        self.assignment_id = assignment_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mesh-particle-grid-splat",
                "target": target.target_id,
                "assignment": assignment_id,
                "source_capacity": int(stable_source_ids.size),
                "route_width": int(route_indices.shape[1]),
            }
        )

    @property
    def particle_capacity(self) -> int:
        return int(self.stable_source_ids.size)

    @property
    def route_width(self) -> int:
        return int(self.route_indices.shape[1])

    def routes(self, positions: ArrayLike, active: ArrayLike, /) -> MeshSplatRoutes:
        value = jnp.asarray(positions)
        active_ = jnp.asarray(active, dtype=bool)
        expected = (self.particle_capacity, self.target.mesh.ambient_dimension)
        if value.shape != expected or active_.shape != (self.particle_capacity,):
            raise ValueError("Runtime mesh splat state has incompatible fixed shapes.")
        geometry = (
            value if self.geometry_ad == "piecewise" else jax.lax.stop_gradient(value)
        )
        finite = jnp.all(jnp.where(active_[:, None], jnp.isfinite(value), True), axis=-1)
        if self.assignment_kind == "barycentric":
            relative = geometry - self.route_origins.astype(geometry.dtype)
            reduced = oe.contract(
                "pij,pj->pi",
                self.route_inverse.astype(geometry.dtype),
                relative,
                backend="jax",
            )
            weights = jnp.concatenate(
                (1.0 - jnp.sum(reduced, axis=-1, keepdims=True), reduced), axis=-1
            )
            first_gradient = -jnp.sum(self.route_inverse, axis=1)
            gradients = jnp.concatenate(
                (first_gradient[:, None, :], self.route_inverse), axis=1
            ).astype(geometry.dtype)
            tolerance = jnp.asarray(self.prepared_tie_margin, dtype=geometry.dtype)
            route_tolerance = 64.0 * jnp.finfo(geometry.dtype).eps
            supported = self.prepared_supported & jnp.all(
                weights >= -route_tolerance, axis=-1
            )
            valid = active_[:, None] & supported[:, None]
            raw_sum = jnp.sum(jnp.where(valid, weights, 0.0), axis=-1)
            normalization = jnp.ones_like(raw_sum)
            overflow = jnp.zeros_like(active_)
            derivative_valid = (~active_) | (
                supported & (tolerance > self.prepared_tie_margin.dtype.type(0))
            )
        else:
            centers = self.cell_centers[self.route_indices]
            delta = geometry[:, None, :] - centers.astype(geometry.dtype)
            distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
            q = distance / jnp.asarray(self.support_radius, dtype=geometry.dtype)
            one_minus = jnp.maximum(1.0 - q, 0.0)
            raw = one_minus**4 * (1.0 + 4.0 * q)
            in_support = q < 1.0
            overflow = self.prepared_query_count > self.route_width
            valid = active_[:, None] & in_support & ~overflow[:, None]
            raw_sum = jnp.sum(jnp.where(valid, raw, 0.0), axis=-1)
            supported = raw_sum > 0.0
            normalization = jnp.where(supported, 1.0 / raw_sum, 0.0)
            weights = raw * (
                normalization[:, None]
                if self.partition_policy == "normalize"
                else jnp.ones_like(raw)
            )
            safe_distance = jnp.where(distance > 0.0, distance, 1.0)
            derivative_q = -20.0 * q * one_minus**3
            raw_gradients = (
                derivative_q[..., None]
                * delta
                / (
                    jnp.asarray(self.support_radius, dtype=geometry.dtype) * safe_distance
                )[..., None]
            )
            if self.partition_policy == "normalize":
                gradient_sum = jnp.sum(
                    jnp.where(valid[..., None], raw_gradients, 0.0), axis=1
                )
                gradients = normalization[:, None, None] * (
                    raw_gradients - weights[..., None] * gradient_sum[:, None, :]
                )
            else:
                gradients = raw_gradients
            derivative_valid = (~active_) | (
                supported
                & ~overflow
                & jnp.all(jnp.abs(q - 1.0) > 32.0 * jnp.finfo(q.dtype).eps, axis=-1)
            )
        valid = jnp.broadcast_to(valid, weights.shape) & finite[:, None]
        complete = finite & (~active_ | (supported & ~overflow))
        evidence = MeshSplatRouteEvidence(
            raw_sum,
            normalization,
            self.prepared_tie_margin,
            self.prepared_query_count,
            overflow,
            supported,
            derivative_valid,
            finite,
            complete,
        )
        return MeshSplatRoutes(
            self.route_indices,
            jnp.where(valid, weights, 0.0),
            jnp.where(valid[..., None], gradients, 0.0),
            valid,
            evidence,
        )

    def deposit(
        self,
        positions: ArrayLike,
        active: ArrayLike,
        source_content: ArrayLike,
        /,
        *,
        accumulation: Literal["fast", "deterministic", "compensated"] = "deterministic",
    ) -> MeshSplatDepositResult:
        content = jnp.asarray(source_content)
        if content.ndim < 1 or content.shape[0] != self.particle_capacity:
            raise ValueError("source_content must begin with particle capacity.")
        active_ = jnp.asarray(active, dtype=bool)
        routes = self.routes(positions, active_)
        stencil = GatherStencil(
            indices=routes.indices,
            weights=routes.weights,
            source_size=self.target.entity_count,
            valid=routes.valid,
            support=routes.evidence.supported,
        )
        order = jnp.argsort(self.stable_source_ids).astype(jnp.int32)
        target_content = deposit_routes(
            stencil, content, order, self.target.entity_count, accumulation
        )
        payload_mask = active_.reshape(
            (self.particle_capacity,) + (1,) * (content.ndim - 1)
        )
        represented_fraction = jnp.sum(routes.weights, axis=-1)
        fraction_shape = represented_fraction.shape + (1,) * (content.ndim - 1)
        represented = jnp.sum(
            jnp.where(payload_mask, content, 0.0)
            * represented_fraction.reshape(fraction_shape),
            axis=0,
        )
        expected = jnp.sum(jnp.where(payload_mask, content, 0.0), axis=0)
        omitted = expected - represented
        residual = jnp.sum(target_content, axis=0) - represented
        scale = jnp.maximum(jnp.max(jnp.abs(expected), initial=0.0), 1.0)
        tolerance = 256.0 * jnp.finfo(target_content.real.dtype).eps * scale
        route_success = jnp.all(routes.evidence.complete)
        if self.boundary == "drop":
            route_success = jnp.all(
                routes.evidence.finite & ~routes.evidence.route_overflow
            )
        successful = (
            route_success
            & jnp.all(jnp.isfinite(target_content))
            & jnp.all(jnp.abs(residual) <= tolerance)
        )
        return MeshSplatDepositResult(
            target_content, represented, omitted, residual, routes.evidence, successful
        )

    def gather(
        self,
        positions: ArrayLike,
        active: ArrayLike,
        target_values: ArrayLike,
        /,
    ) -> MeshSplatGatherResult:
        values = jnp.asarray(target_values)
        if values.ndim < 1 or values.shape[0] != self.target.entity_count:
            raise ValueError("target_values must begin with target entity count.")
        active_ = jnp.asarray(active, dtype=bool)
        routes = self.routes(positions, active_)
        stencil = GatherStencil(
            indices=routes.indices,
            weights=routes.weights,
            source_size=self.target.entity_count,
            valid=routes.valid,
            support=routes.evidence.supported,
        )
        interpolation = apply_gather_stencil(values, stencil)
        successful = jnp.all(routes.evidence.finite & ~routes.evidence.route_overflow)
        if self.boundary == "reject":
            successful = successful & jnp.all(routes.evidence.complete)
        return MeshSplatGatherResult(
            interpolation.values, interpolation.support, routes.evidence, successful
        )


class ParticleGridSplatEpoch(StrictModule):
    """Accepted fixed-capacity particle/mesh transfer epoch."""

    prepared: PreparedMeshParticleGridSplat
    population: ParticlePopulationState
    positions: Array
    target_content: Array
    epoch_number: Array
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedMeshParticleGridSplat,
        population: ParticlePopulationState,
        positions: ArrayLike,
        target_content: ArrayLike,
        /,
        *,
        epoch_number: ArrayLike = 0,
    ):
        if not isinstance(prepared, PreparedMeshParticleGridSplat):
            raise TypeError("prepared must be PreparedMeshParticleGridSplat.")
        if not isinstance(population, ParticlePopulationState):
            raise TypeError("population must be ParticlePopulationState.")
        positions_ = jnp.asarray(positions)
        content = jnp.asarray(target_content)
        if positions_.shape != (
            prepared.particle_capacity,
            prepared.target.mesh.ambient_dimension,
        ):
            raise ValueError("Epoch positions have incompatible capacity/dimension.")
        if population.active.shape != (prepared.particle_capacity,):
            raise ValueError("Epoch population has incompatible capacity.")
        if content.ndim < 1 or content.shape[0] != prepared.target.entity_count:
            raise ValueError("Epoch target content has incompatible entity capacity.")
        number = jnp.asarray(epoch_number, dtype=jnp.int32)
        if number.shape != ():
            raise ValueError("epoch_number must be scalar.")
        self.prepared = prepared
        self.population = population
        self.positions = positions_
        self.target_content = content
        self.epoch_number = number
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "particle-grid-splat-epoch",
                "prepared": prepared.prepared_id,
                "position_shape": positions_.shape,
                "content_shape": content.shape,
            }
        )


class ParticleGridSplatEpochTransition(StrictModule):
    source_epoch: ParticleGridSplatEpoch
    candidate_epoch: ParticleGridSplatEpoch
    accepted_epoch: ParticleGridSplatEpoch
    particle_coverage: Array
    target_coverage: Array
    particle_mass_residual: Array
    target_content_residual: Array
    successful: Array


def prepare_particle_grid_splat_transition(
    source: ParticleGridSplatEpoch,
    target_prepared: PreparedMeshParticleGridSplat,
    target_population: ParticlePopulationState,
    target_positions: ArrayLike,
    /,
    *,
    target_transfer: ArrayLike | None = None,
) -> ParticleGridSplatEpochTransition:
    """Prepare and atomically accept one stable-ID particle/mesh epoch transition.

    ``target_transfer`` is a caller-prepared conservative target-by-source map.  A
    topology change without this mathematical transfer fails closed.
    """

    if not isinstance(source, ParticleGridSplatEpoch):
        raise TypeError("source must be ParticleGridSplatEpoch.")
    if not isinstance(target_prepared, PreparedMeshParticleGridSplat):
        raise TypeError("target_prepared must be PreparedMeshParticleGridSplat.")
    if not isinstance(target_population, ParticlePopulationState):
        raise TypeError("target_population must be ParticlePopulationState.")
    positions = jnp.asarray(target_positions)
    source_keys = np.stack(
        (
            np.asarray(source.prepared.stable_source_ids),
            np.asarray(source.population.incarnation),
        ),
        axis=-1,
    )
    target_keys = np.stack(
        (
            np.asarray(target_prepared.stable_source_ids),
            np.asarray(target_population.incarnation),
        ),
        axis=-1,
    )
    source_lookup = {tuple(key): index for index, key in enumerate(source_keys)}
    matched = np.asarray(
        [source_lookup.get(tuple(key), -1) for key in target_keys], dtype=np.int32
    )
    active_target = np.asarray(target_population.active)
    particle_coverage_host = np.all((matched >= 0) | ~active_target)
    old_active_keys = {
        tuple(key)
        for key, active in zip(source_keys, np.asarray(source.population.active))
        if active
    }
    new_active_keys = {
        tuple(key) for key, active in zip(target_keys, active_target) if active
    }
    particle_coverage_host = particle_coverage_host and old_active_keys.issubset(
        new_active_keys
    )
    source_mass = jnp.sum(
        jnp.where(source.population.active, source.population.mass, 0.0)
    )
    target_mass = jnp.sum(
        jnp.where(target_population.active, target_population.mass, 0.0)
    )
    particle_residual = target_mass - source_mass

    same_topology = source.prepared.target.target_id == target_prepared.target.target_id
    if same_topology:
        migrated_content = source.target_content
        target_coverage = jnp.asarray(True)
    elif target_transfer is None:
        migrated_content = jnp.zeros(
            (target_prepared.target.entity_count,) + source.target_content.shape[1:],
            dtype=source.target_content.dtype,
        )
        target_coverage = jnp.asarray(False)
    else:
        transfer = jnp.asarray(target_transfer)
        expected = (
            target_prepared.target.entity_count,
            source.prepared.target.entity_count,
        )
        if transfer.shape != expected:
            raise ValueError(f"target_transfer must have shape {expected}.")
        migrated_content = oe.contract(
            "ts,s...->t...",
            transfer.astype(source.target_content.dtype),
            source.target_content,
        )
        column_sum = jnp.sum(transfer, axis=0)
        target_coverage = jnp.all(jnp.isfinite(transfer)) & jnp.all(
            jnp.abs(column_sum - 1.0) <= 256.0 * jnp.finfo(transfer.dtype).eps
        )
    source_total = jnp.sum(source.target_content, axis=0)
    target_total = jnp.sum(migrated_content, axis=0)
    content_residual = target_total - source_total
    scale = jnp.maximum(jnp.max(jnp.abs(source_total), initial=0.0), 1.0)
    tolerance = 256.0 * jnp.finfo(migrated_content.real.dtype).eps * scale
    successful = (
        jnp.asarray(particle_coverage_host)
        & target_coverage
        & jnp.isfinite(particle_residual)
        & (jnp.abs(particle_residual) <= tolerance)
        & jnp.all(jnp.isfinite(migrated_content))
        & jnp.all(jnp.abs(content_residual) <= tolerance)
    )
    candidate = ParticleGridSplatEpoch(
        target_prepared,
        target_population,
        positions,
        migrated_content,
        epoch_number=source.epoch_number + 1,
    )
    accepted = candidate if bool(np.asarray(successful)) else source
    return ParticleGridSplatEpochTransition(
        source,
        candidate,
        accepted,
        jnp.asarray(particle_coverage_host),
        target_coverage,
        particle_residual,
        content_residual,
        successful,
    )


__all__ = [
    "MeshCompactKernelSplatAssignment",
    "MeshPartitionPolicy",
    "MeshSplatBoundaryPolicy",
    "MeshSplatDepositResult",
    "MeshSplatGatherResult",
    "MeshSplatGeometryAD",
    "MeshSplatRouteEvidence",
    "MeshSplatRoutes",
    "MeshSplatTarget",
    "ParticleGridSplatEpoch",
    "ParticleGridSplatEpochTransition",
    "PreparedMeshParticleGridSplat",
    "SimplicialBarycentricSplatAssignment",
    "prepare_particle_grid_splat_transition",
]
