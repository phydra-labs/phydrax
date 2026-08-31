#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._boundary_open import (
    apply_open_boundaries,
    LatticeBoltzmannBoundaryState,
    OpenNormal,
    VelocityOpenNormal,
)
from ._boundary_wall import apply_wall_boundaries, LatticeBoltzmannWallLedger
from ._discretization import LatticeBoltzmannDiscretization
from ._link_geometry import FixedSDFLinkGeometry
from ._link_topology import (
    CompiledLatticeBoltzmannLinkTopology,
    LatticeBoltzmannBodyBoundary,
    LatticeBoltzmannBoundaryStage,
    LatticeBoltzmannCornerRule,
    LatticeBoltzmannFaceBoundary,
    LatticeBoltzmannLinkOwner,
)


WallSide: TypeAlias = Literal["lower", "upper"]
WallFace: TypeAlias = tuple[str, WallSide]


def _face_slice(dimension: int, axis: int, side: WallSide, /) -> tuple[object, ...]:
    values: list[object] = [slice(None)] * dimension
    values[axis] = 0 if side == "lower" else -1
    return tuple(values)


class LatticeBoltzmannGeometrySnapshot(StrictModule, NonTrainableState):
    """Frozen fluid-cell classification detached from live geometry state."""

    fluid_mask: Array
    discretization_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    fluid_count: int = eqx.field(static=True)
    solid_count: int = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        fluid_mask: ArrayLike,
        /,
        *,
        source_id: str | None = None,
    ):
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("Geometry snapshot requires an LBM discretization.")
        mask = np.asarray(fluid_mask, dtype=bool)
        if mask.shape != discretization.grid.shape:
            raise ValueError(
                f"fluid_mask must have shape {discretization.grid.shape}; got {mask.shape}."
            )
        fluid_count = int(np.sum(mask))
        if fluid_count == 0:
            raise ValueError("LBM geometry must contain at least one fluid cell.")
        source = (
            canonical_fingerprint(
                {
                    "kind": "lattice-boltzmann-mask-source",
                    "fluid_mask": array_tree_fingerprint(mask),
                }
            )
            if source_id is None
            else str(source_id)
        )
        if not source:
            raise ValueError("source_id must be non-empty.")
        self.fluid_mask = jnp.asarray(mask, dtype=bool)
        self.discretization_id = discretization.prepared_id
        self.source_id = source
        self.fluid_count = fluid_count
        self.solid_count = int(mask.size - fluid_count)
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-geometry-snapshot",
                "discretization": discretization.prepared_id,
                "source": source,
                "fluid_mask": array_tree_fingerprint(mask),
            }
        )

    @classmethod
    def all_fluid(
        cls, discretization: LatticeBoltzmannDiscretization, /
    ) -> "LatticeBoltzmannGeometrySnapshot":
        return cls(
            discretization,
            np.ones(discretization.grid.shape, dtype=bool),
            source_id="all-fluid",
        )


class LatticeBoltzmannBoundaryPlan(StrictModule, NonTrainableState):
    """Periodic or halfway-wall ownership for every nearest-neighbour link."""

    geometry: LatticeBoltzmannGeometrySnapshot | None
    moving_faces: tuple[WallFace, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        geometry: LatticeBoltzmannGeometrySnapshot | None = None,
        moving_faces: Sequence[WallFace] = (),
    ):
        if geometry is not None and not isinstance(
            geometry, LatticeBoltzmannGeometrySnapshot
        ):
            raise TypeError("geometry must be an LBM geometry snapshot or None.")
        faces_list: list[WallFace] = []
        for axis, side in moving_faces:
            axis_ = str(axis)
            if not axis_ or side not in ("lower", "upper"):
                raise ValueError(
                    "Moving wall faces require an axis and lower/upper side."
                )
            faces_list.append((axis_, side))
        faces = tuple(faces_list)
        if len(set(faces)) != len(faces):
            raise ValueError("Moving wall faces must be unique.")
        for index, (axis, _) in enumerate(faces):
            if any(other_axis != axis for other_axis, _ in faces[index + 1 :]):
                raise ValueError(
                    "Intersecting moving wall faces are not supported initially."
                )
        self.geometry = geometry
        self.moving_faces = faces
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-boundary-plan",
                "geometry": None if geometry is None else geometry.snapshot_id,
                "moving_faces": [list(face) for face in faces],
            }
        )

    def prepare(
        self, discretization: LatticeBoltzmannDiscretization, /
    ) -> "PreparedLatticeBoltzmannBoundary":
        return PreparedLatticeBoltzmannBoundary(discretization, self)


class PreparedLatticeBoltzmannBoundary(StrictModule, NonTrainableState):
    discretization: LatticeBoltzmannDiscretization
    geometry: LatticeBoltzmannGeometrySnapshot
    moving_faces: tuple[tuple[int, WallSide], ...] = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        plan: LatticeBoltzmannBoundaryPlan,
        /,
    ):
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("Boundary preparation requires an LBM discretization.")
        if not isinstance(plan, LatticeBoltzmannBoundaryPlan):
            raise TypeError("plan must be a LatticeBoltzmannBoundaryPlan.")
        geometry = (
            LatticeBoltzmannGeometrySnapshot.all_fluid(discretization)
            if plan.geometry is None
            else plan.geometry
        )
        if geometry.discretization_id != discretization.prepared_id:
            raise ValueError("Geometry snapshot belongs to a different discretization.")
        faces: list[tuple[int, WallSide]] = []
        for axis, side in plan.moving_faces:
            if axis not in discretization.grid.axis_names:
                raise ValueError(f"Unknown moving-wall axis {axis!r}.")
            axis_index = discretization.grid.axis_names.index(axis)
            if discretization.periodic[axis_index]:
                raise ValueError("A periodic face cannot also be a moving wall.")
            faces.append((axis_index, side))
        self.discretization = discretization
        self.geometry = geometry
        self.moving_faces = tuple(faces)
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "prepared-lattice-boltzmann-boundary",
                "discretization": discretization.prepared_id,
                "plan": plan.plan_id,
                "geometry": geometry.snapshot_id,
                "moving_faces": [list(face) for face in faces],
            }
        )

    @property
    def moving_face_count(self) -> int:
        return len(self.moving_faces)

    def route(
        self,
        post_collision: Array,
        density: Array,
        moving_wall_velocities: Array,
        /,
    ) -> Array:
        populations = self.discretization.validate_populations(post_collision)
        rho = jnp.asarray(density, dtype=populations.dtype)
        if rho.shape != self.discretization.grid.shape:
            raise ValueError("Boundary density must match the LBM grid shape.")
        wall_velocity = jnp.asarray(moving_wall_velocities, dtype=populations.dtype)
        if wall_velocity.size == 0 and self.moving_face_count == 0:
            wall_velocity = wall_velocity.reshape(
                (0, self.discretization.velocity_set.dimension)
            )
        expected_wall_shape = (
            self.moving_face_count,
            self.discretization.velocity_set.dimension,
        )
        if wall_velocity.shape != expected_wall_shape:
            raise ValueError(
                f"moving_wall_velocities must have shape {expected_wall_shape}."
            )
        for wall_index, (axis, _) in enumerate(self.moving_faces):
            wall_velocity = eqx.error_if(
                wall_velocity,
                ~jnp.isfinite(wall_velocity[wall_index, axis])
                | (wall_velocity[wall_index, axis] != 0.0),
                "Moving halfway walls require finite tangential velocity.",
            )
        wall_velocity = eqx.error_if(
            wall_velocity,
            jnp.any(~jnp.isfinite(wall_velocity)),
            "Moving wall velocities must be finite.",
        )

        dimension = self.discretization.velocity_set.dimension
        axes = tuple(range(dimension))
        fluid = self.geometry.fluid_mask
        velocities = self.discretization.velocity_set.velocity_tuples
        opposite = self.discretization.velocity_set.opposite_indices
        weights = jnp.asarray(
            self.discretization.velocity_set.weights, dtype=populations.dtype
        )
        cs2 = jnp.asarray(
            self.discretization.velocity_set.sound_speed_squared,
            dtype=populations.dtype,
        )
        routed = []
        for direction, lattice_velocity in enumerate(velocities):
            shift = lattice_velocity
            pulled = jnp.roll(populations[..., direction], shift=shift, axis=axes)
            source_fluid = jnp.roll(fluid, shift=shift, axis=axes)
            correction = jnp.zeros_like(rho)
            for axis, component in enumerate(lattice_velocity):
                if component == 0 or self.discretization.periodic[axis]:
                    continue
                side: WallSide = "lower" if component > 0 else "upper"
                face = _face_slice(dimension, axis, side)
                source_fluid = source_fluid.at[face].set(False)
                for wall_index, moving_face in enumerate(self.moving_faces):
                    if moving_face != (axis, side):
                        continue
                    projection = jnp.sum(
                        jnp.asarray(lattice_velocity, dtype=populations.dtype)
                        * wall_velocity[wall_index]
                    )
                    wall_correction = (
                        2.0 * weights[direction] * rho[face] * projection / cs2
                    )
                    correction = correction.at[face].set(wall_correction)
            reflected = populations[..., opposite[direction]] + correction
            selected = jnp.where(source_fluid, pulled, reflected)
            routed.append(jnp.where(fluid, selected, populations[..., direction]))
        return jnp.stack(routed, axis=-1)


class LatticeBoltzmannBoundaryParameters(StrictModule):
    """Dynamic data consumed by one compiled staged boundary program."""

    halo_populations: Array | None
    velocity_targets: Array
    pressure_densities: Array
    pressure_tangential_velocities: Array
    convective_speeds: Array
    half_force_density: Array
    body_centers: Array
    body_linear_velocities: Array
    body_angular_velocities: Array
    time_step: Array

    def __init__(
        self,
        /,
        *,
        halo_populations: ArrayLike | None = None,
        velocity_targets: ArrayLike | None = None,
        pressure_densities: ArrayLike | None = None,
        pressure_tangential_velocities: ArrayLike | None = None,
        convective_speeds: ArrayLike | None = None,
        half_force_density: ArrayLike = 0.0,
        body_centers: ArrayLike | None = None,
        body_linear_velocities: ArrayLike | None = None,
        body_angular_velocities: ArrayLike | None = None,
        time_step: ArrayLike = 1.0,
    ):
        self.halo_populations = (
            None if halo_populations is None else jnp.asarray(halo_populations)
        )
        self.velocity_targets = jnp.asarray(
            () if velocity_targets is None else velocity_targets
        )
        self.pressure_densities = jnp.asarray(
            () if pressure_densities is None else pressure_densities
        )
        self.pressure_tangential_velocities = jnp.asarray(
            ()
            if pressure_tangential_velocities is None
            else pressure_tangential_velocities
        )
        self.convective_speeds = jnp.asarray(
            () if convective_speeds is None else convective_speeds
        )
        self.half_force_density = jnp.asarray(half_force_density)
        self.body_centers = jnp.asarray(() if body_centers is None else body_centers)
        self.body_linear_velocities = jnp.asarray(
            () if body_linear_velocities is None else body_linear_velocities
        )
        self.body_angular_velocities = jnp.asarray(
            () if body_angular_velocities is None else body_angular_velocities
        )
        self.time_step = jnp.asarray(time_step)


class StagedLatticeBoltzmannBoundaryPlan(StrictModule, NonTrainableState):
    """Plan for compiled stream/wall/open ownership and runtime parameters."""

    topology: CompiledLatticeBoltzmannLinkTopology
    body_ids: tuple[str, ...] = eqx.field(static=True)
    velocity_normals: tuple[VelocityOpenNormal, ...] = eqx.field(static=True)
    pressure_normals: tuple[OpenNormal, ...] = eqx.field(static=True)
    convective_normals: tuple[OpenNormal, ...] = eqx.field(static=True)
    halo_parameter_ids: tuple[str, ...] = eqx.field(static=True)
    velocity_parameter_ids: tuple[str, ...] = eqx.field(static=True)
    pressure_parameter_ids: tuple[str, ...] = eqx.field(static=True)
    convective_parameter_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: CompiledLatticeBoltzmannLinkTopology,
        /,
        *,
        body_ids: Sequence[str] = (),
        velocity_normals: Sequence[VelocityOpenNormal] = (),
        pressure_normals: Sequence[OpenNormal] = (),
        convective_normals: Sequence[OpenNormal] = (),
        halo_parameter_ids: Sequence[str] = (),
        velocity_parameter_ids: Sequence[str] = (),
        pressure_parameter_ids: Sequence[str] = (),
        convective_parameter_ids: Sequence[str] = (),
    ):
        if not isinstance(topology, CompiledLatticeBoltzmannLinkTopology):
            raise TypeError("topology must be CompiledLatticeBoltzmannLinkTopology.")
        self.topology = topology
        self.body_ids = tuple(str(value) for value in body_ids)
        self.velocity_normals = tuple(velocity_normals)
        self.pressure_normals = tuple(pressure_normals)
        self.convective_normals = tuple(convective_normals)
        self.halo_parameter_ids = tuple(str(value) for value in halo_parameter_ids)
        self.velocity_parameter_ids = tuple(
            str(value) for value in velocity_parameter_ids
        )
        self.pressure_parameter_ids = tuple(
            str(value) for value in pressure_parameter_ids
        )
        self.convective_parameter_ids = tuple(
            str(value) for value in convective_parameter_ids
        )
        if len(self.velocity_parameter_ids) != len(self.velocity_normals):
            raise ValueError("velocity_parameter_ids must match velocity_normals.")
        if len(self.pressure_parameter_ids) != len(self.pressure_normals):
            raise ValueError("pressure_parameter_ids must match pressure_normals.")
        if len(self.convective_parameter_ids) != len(self.convective_normals):
            raise ValueError("convective_parameter_ids must match convective_normals.")
        self.plan_id = canonical_fingerprint(
            {
                "kind": "staged-lattice-boltzmann-boundary-plan",
                "topology": topology.topology_id,
                "body_ids": self.body_ids,
                "velocity_normals": self.velocity_normals,
                "pressure_normals": self.pressure_normals,
                "convective_normals": self.convective_normals,
                "halo_parameter_ids": self.halo_parameter_ids,
                "velocity_parameter_ids": self.velocity_parameter_ids,
                "pressure_parameter_ids": self.pressure_parameter_ids,
                "convective_parameter_ids": self.convective_parameter_ids,
            }
        )

    def prepare(
        self, discretization: LatticeBoltzmannDiscretization, /
    ) -> "PreparedStagedLatticeBoltzmannBoundary":
        return PreparedStagedLatticeBoltzmannBoundary(
            discretization,
            self.topology,
            body_ids=self.body_ids,
            velocity_normals=self.velocity_normals,
            pressure_normals=self.pressure_normals,
            convective_normals=self.convective_normals,
            halo_parameter_ids=self.halo_parameter_ids,
            velocity_parameter_ids=self.velocity_parameter_ids,
            pressure_parameter_ids=self.pressure_parameter_ids,
            convective_parameter_ids=self.convective_parameter_ids,
        )


def compile_staged_lattice_boltzmann_boundary(
    discretization: LatticeBoltzmannDiscretization,
    /,
    *,
    faces: Sequence[LatticeBoltzmannFaceBoundary] = (),
    geometry: FixedSDFLinkGeometry | None = None,
    body_boundaries: Sequence[LatticeBoltzmannBodyBoundary] = (),
    corner_rules: Sequence[LatticeBoltzmannCornerRule] = (),
    default_nonperiodic_wall: bool = True,
) -> StagedLatticeBoltzmannBoundaryPlan:
    """Compile exclusive population ownership from typed face and SDF declarations."""

    if not isinstance(discretization, LatticeBoltzmannDiscretization):
        raise TypeError("discretization must be LatticeBoltzmannDiscretization.")
    if geometry is not None and (
        not isinstance(geometry, FixedSDFLinkGeometry)
        or geometry.discretization_id != discretization.prepared_id
    ):
        raise ValueError("geometry must be FixedSDFLinkGeometry for this discretization.")
    declarations = tuple(faces)
    if any(not isinstance(face, LatticeBoltzmannFaceBoundary) for face in declarations):
        raise TypeError("faces must contain LatticeBoltzmannFaceBoundary declarations.")
    face_map = {face.face: face for face in declarations}
    if len(face_map) != len(declarations):
        raise ValueError("Each exterior face may be declared only once.")
    axis_lookup = {name: axis for axis, name in enumerate(discretization.grid.axis_names)}
    if any(face.axis not in axis_lookup for face in declarations):
        raise ValueError("A face declaration names an axis outside the lattice grid.")
    if default_nonperiodic_wall:
        for axis, name in enumerate(discretization.grid.axis_names):
            if discretization.periodic[axis]:
                continue
            for side in ("lower", "upper"):
                key = (name, side)
                if key not in face_map:
                    face_map[key] = LatticeBoltzmannFaceBoundary(
                        name,
                        side,
                        LatticeBoltzmannLinkOwner.HALFWAY,
                        body_id="__exterior_wall__",
                    )
    else:
        missing = tuple(
            (name, side)
            for axis, name in enumerate(discretization.grid.axis_names)
            if not discretization.periodic[axis]
            for side in ("lower", "upper")
            if (name, side) not in face_map
        )
        if missing:
            raise ValueError(f"Nonperiodic faces require declarations: {missing}.")

    bodies = tuple(body_boundaries)
    if any(not isinstance(body, LatticeBoltzmannBodyBoundary) for body in bodies):
        raise TypeError("body_boundaries must contain LatticeBoltzmannBodyBoundary.")
    body_boundary_map = {body.body_id: body for body in bodies}
    if len(body_boundary_map) != len(bodies):
        raise ValueError("Each immersed body may be declared only once.")
    geometry_body_ids = () if geometry is None else geometry.body_names
    if any(body.body_id not in geometry_body_ids for body in bodies):
        raise ValueError("A body boundary names a body absent from the SDF geometry.")

    rules = tuple(corner_rules)
    if any(not isinstance(rule, LatticeBoltzmannCornerRule) for rule in rules):
        raise TypeError("corner_rules must contain LatticeBoltzmannCornerRule.")
    corner_map = {frozenset(rule.faces): rule.source_face for rule in rules}
    if len(corner_map) != len(rules):
        raise ValueError("Each intersecting-face set may have only one corner rule.")
    if any(
        face[0] not in axis_lookup
        or (face not in face_map and not discretization.periodic[axis_lookup[face[0]]])
        for rule in rules
        for face in rule.faces
    ):
        raise ValueError("A corner rule references an undeclared exterior face.")

    ordered_faces = tuple(
        face_map[key]
        for key in sorted(
            face_map,
            key=lambda face: (axis_lookup[face[0]], 0 if face[1] == "lower" else 1),
        )
    )

    def parameter_table(owner: LatticeBoltzmannLinkOwner) -> tuple[str, ...]:
        identifiers = tuple(
            face.parameter_id
            for face in ordered_faces
            if face.owner is owner and face.parameter_id is not None
        )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(
                f"{owner.name} parameter identifiers must be unique per exterior face."
            )
        return identifiers

    halo_ids = parameter_table(LatticeBoltzmannLinkOwner.HALO)
    velocity_ids = parameter_table(LatticeBoltzmannLinkOwner.VELOCITY)
    pressure_ids = parameter_table(LatticeBoltzmannLinkOwner.PRESSURE)
    convective_ids = parameter_table(LatticeBoltzmannLinkOwner.CONVECTIVE)
    parameter_tables = {
        LatticeBoltzmannLinkOwner.HALO: {
            value: index for index, value in enumerate(halo_ids)
        },
        LatticeBoltzmannLinkOwner.VELOCITY: {
            value: index for index, value in enumerate(velocity_ids)
        },
        LatticeBoltzmannLinkOwner.PRESSURE: {
            value: index for index, value in enumerate(pressure_ids)
        },
        LatticeBoltzmannLinkOwner.CONVECTIVE: {
            value: index for index, value in enumerate(convective_ids)
        },
    }

    wall_body_ids = tuple(
        face.body_id
        for face in ordered_faces
        if face.body_id is not None and face.body_id not in geometry_body_ids
    )
    body_ids = geometry_body_ids + tuple(dict.fromkeys(wall_body_ids))
    body_indices = {value: index for index, value in enumerate(body_ids)}
    shape = discretization.population_shape
    owner = np.full(shape, int(LatticeBoltzmannLinkOwner.LOCAL), dtype=np.int8)
    parameter_index = np.full(shape, -1, dtype=np.int32)
    normal_axis = np.full(shape, -1, dtype=np.int8)
    normal_sign = np.zeros(shape, dtype=np.int8)
    body_index = np.full(shape, -1, dtype=np.int32)
    link_fraction = np.zeros(shape, dtype=np.float64)
    fluid = (
        np.ones(discretization.grid.shape, dtype=bool)
        if geometry is None
        else np.asarray(geometry.fluid_mask, dtype=bool)
    )
    velocities = discretization.velocity_set.velocity_tuples

    def assign_face(
        index: tuple[int, ...],
        face: LatticeBoltzmannFaceBoundary,
    ) -> None:
        axis = axis_lookup[face.axis]
        sign = -1 if face.side == "lower" else 1
        owner[index] = int(face.owner)
        normal_axis[index] = axis
        normal_sign[index] = sign
        if face.parameter_id is not None:
            parameter_index[index] = parameter_tables[face.owner][face.parameter_id]
        if face.body_id is not None:
            body_index[index] = body_indices[face.body_id]
            link_fraction[index] = (
                face.link_fraction
                if face.owner is LatticeBoltzmannLinkOwner.BOUZIDI
                else 0.5
            )

    def assign_geometry(index: tuple[int, ...], source: tuple[int, ...]) -> None:
        if geometry is None:
            raise RuntimeError("Geometry assignment requires FixedSDFLinkGeometry.")
        label = int(np.asarray(geometry.body_labels)[source])
        body_id = geometry.body_names[label]
        declaration = body_boundary_map.get(body_id)
        wall_owner = (
            LatticeBoltzmannLinkOwner.BOUZIDI
            if declaration is None
            else declaration.owner
        )
        owner[index] = int(wall_owner)
        body_index[index] = body_indices[body_id]
        link_fraction[index] = (
            float(np.asarray(geometry.link_fraction)[index])
            if wall_owner is LatticeBoltzmannLinkOwner.BOUZIDI
            else 0.5
        )

    for cell in np.ndindex(discretization.grid.shape):
        if not fluid[cell]:
            continue
        for direction, velocity in enumerate(velocities):
            if not any(velocity):
                continue
            destination = cell + (direction,)
            source_values = []
            crossed = []
            for axis, component in enumerate(velocity):
                source = cell[axis] - component
                if source < 0:
                    crossed.append((discretization.grid.axis_names[axis], "lower"))
                    source %= discretization.grid.shape[axis]
                elif source >= discretization.grid.shape[axis]:
                    crossed.append((discretization.grid.axis_names[axis], "upper"))
                    source %= discretization.grid.shape[axis]
                source_values.append(source)
            source_cell = tuple(source_values)
            if not crossed:
                if geometry is not None and not fluid[source_cell]:
                    assign_geometry(destination, source_cell)
                continue
            selected_face = None
            if len(crossed) == 1:
                selected_face = crossed[0]
            else:
                key = frozenset(crossed)
                selected_face = corner_map.get(key)
                candidates = tuple(face_map.get(face) for face in crossed)
                ownership = tuple(
                    None
                    if candidate is None
                    else (
                        candidate.owner,
                        candidate.parameter_id,
                        candidate.body_id,
                        candidate.link_fraction,
                        candidate.flow_direction,
                    )
                    for candidate in candidates
                )
                equivalent = len(set(ownership)) == 1
                if selected_face is None and not equivalent:
                    raise ValueError(
                        f"Conflicting corner ownership requires an explicit rule: {crossed}."
                    )
                if selected_face is None:
                    selected_face = crossed[0]
            declaration = face_map.get(selected_face)
            if declaration is None:
                selected_axis = axis_lookup[selected_face[0]]
                if not discretization.periodic[selected_axis]:
                    raise ValueError(
                        f"Exterior face {selected_face} has no boundary declaration."
                    )
                if geometry is not None and not fluid[source_cell]:
                    assign_geometry(destination, source_cell)
                else:
                    owner[destination] = int(LatticeBoltzmannLinkOwner.PERIODIC)
                continue
            assign_face(destination, declaration)

    topology_id = canonical_fingerprint(
        {
            "kind": "compiled-staged-lattice-boltzmann-boundary",
            "discretization": discretization.prepared_id,
            "geometry": None if geometry is None else geometry.geometry_id,
            "faces": [face.declaration_id for face in ordered_faces],
            "bodies": [body.declaration_id for body in bodies],
            "corners": [rule.rule_id for rule in rules],
            "owner": array_tree_fingerprint(owner),
            "parameter_index": array_tree_fingerprint(parameter_index),
            "body_index": array_tree_fingerprint(body_index),
            "link_fraction": array_tree_fingerprint(link_fraction),
        }
    )
    topology = CompiledLatticeBoltzmannLinkTopology(
        owner,
        parameter_index,
        normal_axis,
        normal_sign,
        body_index,
        link_fraction,
        fluid,
        topology_id=topology_id,
    )

    def normals(
        owner_kind: LatticeBoltzmannLinkOwner,
    ) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                axis_lookup[face.axis],
                -1 if face.side == "lower" else 1,
            )
            for face in ordered_faces
            if face.owner is owner_kind
        )

    velocity_normals = tuple(
        (
            axis_lookup[face.axis],
            -1 if face.side == "lower" else 1,
            face.flow_direction,
        )
        for face in ordered_faces
        if face.owner is LatticeBoltzmannLinkOwner.VELOCITY
    )
    return StagedLatticeBoltzmannBoundaryPlan(
        topology,
        body_ids=body_ids,
        velocity_normals=velocity_normals,
        pressure_normals=normals(LatticeBoltzmannLinkOwner.PRESSURE),
        convective_normals=normals(LatticeBoltzmannLinkOwner.CONVECTIVE),
        halo_parameter_ids=halo_ids,
        velocity_parameter_ids=velocity_ids,
        pressure_parameter_ids=pressure_ids,
        convective_parameter_ids=convective_ids,
    )


class LatticeBoltzmannBoundaryResult(StrictModule):
    populations: Array
    state: LatticeBoltzmannBoundaryState
    ledger: LatticeBoltzmannWallLedger


class PreparedStagedLatticeBoltzmannBoundary(StrictModule, NonTrainableState):
    """One write-once stream/wall/open program over compiled link ownership."""

    geometry: LatticeBoltzmannGeometrySnapshot
    discretization: LatticeBoltzmannDiscretization
    topology: CompiledLatticeBoltzmannLinkTopology
    body_ids: tuple[str, ...] = eqx.field(static=True)
    velocity_normals: tuple[VelocityOpenNormal, ...] = eqx.field(static=True)
    pressure_normals: tuple[OpenNormal, ...] = eqx.field(static=True)
    convective_normals: tuple[OpenNormal, ...] = eqx.field(static=True)
    halo_parameter_ids: tuple[str, ...] = eqx.field(static=True)
    velocity_parameter_ids: tuple[str, ...] = eqx.field(static=True)
    pressure_parameter_ids: tuple[str, ...] = eqx.field(static=True)
    convective_parameter_ids: tuple[str, ...] = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        topology: CompiledLatticeBoltzmannLinkTopology,
        /,
        *,
        body_ids: Sequence[str] = (),
        velocity_normals: Sequence[VelocityOpenNormal] = (),
        pressure_normals: Sequence[OpenNormal] = (),
        convective_normals: Sequence[OpenNormal] = (),
        halo_parameter_ids: Sequence[str] = (),
        velocity_parameter_ids: Sequence[str] = (),
        pressure_parameter_ids: Sequence[str] = (),
        convective_parameter_ids: Sequence[str] = (),
    ):
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("discretization must be LatticeBoltzmannDiscretization.")
        if not isinstance(topology, CompiledLatticeBoltzmannLinkTopology):
            raise TypeError("topology must be CompiledLatticeBoltzmannLinkTopology.")
        if topology.population_shape != discretization.population_shape:
            raise ValueError("Boundary topology and population shapes do not match.")
        bodies = tuple(str(value) for value in body_ids)
        if any(not value for value in bodies) or len(set(bodies)) != len(bodies):
            raise ValueError("body_ids must be unique nonempty values.")
        self.discretization = discretization
        self.topology = topology
        self.geometry = LatticeBoltzmannGeometrySnapshot(
            discretization,
            topology.fluid_mask,
            source_id=topology.topology_id,
        )
        self.body_ids = bodies
        self.velocity_normals = tuple(velocity_normals)
        self.pressure_normals = tuple(pressure_normals)
        self.convective_normals = tuple(convective_normals)
        self.halo_parameter_ids = tuple(str(value) for value in halo_parameter_ids)
        self.velocity_parameter_ids = tuple(
            str(value) for value in velocity_parameter_ids
        )
        self.pressure_parameter_ids = tuple(
            str(value) for value in pressure_parameter_ids
        )
        self.convective_parameter_ids = tuple(
            str(value) for value in convective_parameter_ids
        )
        if len(self.velocity_parameter_ids) != len(self.velocity_normals):
            raise ValueError("velocity_parameter_ids must match velocity_normals.")
        if len(self.pressure_parameter_ids) != len(self.pressure_normals):
            raise ValueError("pressure_parameter_ids must match pressure_normals.")
        if len(self.convective_parameter_ids) != len(self.convective_normals):
            raise ValueError("convective_parameter_ids must match convective_normals.")
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "prepared-staged-lattice-boltzmann-boundary",
                "discretization": discretization.prepared_id,
                "topology": topology.topology_id,
                "body_ids": bodies,
                "velocity_normals": self.velocity_normals,
                "pressure_normals": self.pressure_normals,
                "convective_normals": self.convective_normals,
                "halo_parameter_ids": self.halo_parameter_ids,
                "velocity_parameter_ids": self.velocity_parameter_ids,
                "pressure_parameter_ids": self.pressure_parameter_ids,
                "convective_parameter_ids": self.convective_parameter_ids,
            }
        )

    def initial_state(
        self, populations: ArrayLike | None = None, /
    ) -> LatticeBoltzmannBoundaryState:
        values = (
            jnp.zeros(self.discretization.population_shape)
            if populations is None
            else self.discretization.validate_populations(populations)
        )
        convective = self.topology.owner == int(LatticeBoltzmannLinkOwner.CONVECTIVE)
        initialized = (
            convective if populations is not None else jnp.zeros(values.shape, dtype=bool)
        )
        return LatticeBoltzmannBoundaryState(
            jnp.where(convective, values, 0.0),
            initialized,
        )

    def apply(
        self,
        post_collision: ArrayLike,
        density: ArrayLike,
        state: LatticeBoltzmannBoundaryState,
        parameters: LatticeBoltzmannBoundaryParameters,
        /,
    ) -> LatticeBoltzmannBoundaryResult:
        values = self.discretization.validate_populations(post_collision)
        rho = jnp.asarray(density, dtype=values.dtype)
        if rho.shape != self.discretization.grid.shape:
            raise ValueError("Boundary density must match grid shape.")
        if not isinstance(state, LatticeBoltzmannBoundaryState):
            raise TypeError("state must be LatticeBoltzmannBoundaryState.")
        if not isinstance(parameters, LatticeBoltzmannBoundaryParameters):
            raise TypeError("parameters must be LatticeBoltzmannBoundaryParameters.")
        axes = tuple(range(self.discretization.velocity_set.dimension))
        pulled = jnp.stack(
            tuple(
                jnp.roll(values[..., direction], shift=velocity, axis=axes)
                for direction, velocity in enumerate(
                    self.discretization.velocity_set.velocity_tuples
                )
            ),
            axis=-1,
        )
        halo_owned = self.topology.owner == int(LatticeBoltzmannLinkOwner.HALO)
        if parameters.halo_populations is None:
            pulled = eqx.error_if(
                pulled,
                jnp.any(halo_owned),
                "Halo-owned populations require halo_populations.",
            )
        else:
            halo = jnp.asarray(parameters.halo_populations, dtype=values.dtype)
            if halo.shape != values.shape:
                raise ValueError("halo_populations must match population shape.")
            pulled = jnp.where(halo_owned, halo, pulled)
        stage = self.topology.begin(jnp.zeros_like(values))
        stage = self.topology.commit(
            stage,
            pulled,
            LatticeBoltzmannBoundaryStage.STREAM,
            (
                LatticeBoltzmannLinkOwner.LOCAL,
                LatticeBoltzmannLinkOwner.PERIODIC,
                LatticeBoltzmannLinkOwner.HALO,
            ),
        )
        wall_candidate, ledger = apply_wall_boundaries(
            stage.populations,
            values,
            rho,
            self.topology,
            self.discretization.velocity_set.velocity_tuples,
            self.discretization.velocity_set.velocities,
            self.discretization.velocity_set.opposite,
            self.discretization.velocity_set.weights,
            self.discretization.velocity_set.sound_speed_squared,
            self.discretization.grid.points.reshape(
                self.discretization.grid.shape
                + (self.discretization.velocity_set.dimension,)
            ),
            self.discretization.cell_size,
            parameters.body_centers,
            parameters.body_linear_velocities,
            parameters.body_angular_velocities,
            parameters.time_step,
        )
        stage = self.topology.commit(
            stage,
            wall_candidate,
            LatticeBoltzmannBoundaryStage.WALL,
            (
                LatticeBoltzmannLinkOwner.HALFWAY,
                LatticeBoltzmannLinkOwner.BOUZIDI,
            ),
        )
        open_candidate, next_state = apply_open_boundaries(
            stage.populations,
            values,
            self.topology,
            state,
            self.discretization.velocity_set.velocities,
            self.discretization.velocity_set.velocity_tuples,
            self.discretization.velocity_set.opposite,
            self.discretization.velocity_set.weights,
            parameters.velocity_targets,
            parameters.pressure_densities,
            parameters.pressure_tangential_velocities,
            parameters.convective_speeds,
            parameters.half_force_density,
            self.velocity_normals,
            self.pressure_normals,
            self.convective_normals,
        )
        stage = self.topology.commit(
            stage,
            open_candidate,
            LatticeBoltzmannBoundaryStage.OPEN,
            (
                LatticeBoltzmannLinkOwner.VELOCITY,
                LatticeBoltzmannLinkOwner.PRESSURE,
                LatticeBoltzmannLinkOwner.CONVECTIVE,
            ),
        )
        return LatticeBoltzmannBoundaryResult(
            self.topology.finish(stage), next_state, ledger
        )


__all__ = [
    "compile_staged_lattice_boltzmann_boundary",
    "LatticeBoltzmannBoundaryParameters",
    "LatticeBoltzmannBoundaryPlan",
    "LatticeBoltzmannBoundaryResult",
    "LatticeBoltzmannGeometrySnapshot",
    "PreparedLatticeBoltzmannBoundary",
    "PreparedStagedLatticeBoltzmannBoundary",
    "StagedLatticeBoltzmannBoundaryPlan",
]
