#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-neutral executable native robot model.

`NativeRobotModel` is the single model authority consumed by native robot
execution, whether it was constructed directly or produced by a source adapter.
It composes the existing native owners rather than redefining their physics:
particle masses, COM-centered rigid-body inertias, the rigid-joint graph, the
fixed-base reduced articulation and its zero-configuration reference, plus the
stateless joint motor, viscous damping, unilateral stop, and collision
declarations owned by their substrates.

Body, joint, frame, actuator, and collision-primitive identities are explicit
sorted-name maps; numerical arrays are ordered by those identities, never by
source position. A direct model carries no adapter report. A source adapter
returns `RobotAdaptation`, which pairs the same model type with its genuine
`AdapterReport` and source evidence.

The articulation root must be an active fixed body with its own declared
positive mass properties. A massless world anchor is refused rather than given
an invented mass: the reduced owners represent the root as a physical body.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import assert_never, Generic, Literal, TypeVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import DimensionalScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._articulated_geometry import (
    ArticulatedCollisionGeometry,
    ArticulatedPrimitiveKind,
)
from ...discretization.particle._core import ParticleDiscretization, ParticleSetPlan
from ...discretization.particle._reduced_articulation import (
    PreparedReducedArticulation,
    ReducedArticulationPlan,
)
from ...discretization.particle._rigid_body import (
    PreparedRigidBodySet,
    quaternion_rotation_matrix,
    RigidBodyKinematics,
    RigidBodySetPlan,
)
from ...discretization.particle._rigid_joint_coordinates import (
    PreparedLiftedJointCoordinates,
)
from ...discretization.particle._rigid_joint_laws import (
    PreparedScalarJointDamping,
    PreparedScalarJointMotors,
    ScalarJointDampingPlan,
    ScalarJointMotorPlan,
)
from ...discretization.particle._rigid_joints import (
    FixedJointSetPlan,
    HingeJointSetPlan,
    PreparedRigidJointGraph,
    PrismaticJointSetPlan,
    RigidJointGraphPlan,
)
from ...discretization.particle._rigid_unilateral import (
    PreparedReducedJointStops,
    ReducedJointStopPlan,
)
from ...interchange._report import AdapterReport
from ...typing import checked, parse


type RobotContactLaw = Literal["inelastic-coulomb-hard-contact"]

_UNIT_TOLERANCE = 1.0e-12


class RobotNameIDMap(StrictModule, NonTrainableState):
    """Deterministic bijection between sorted names and signed 64-bit IDs.

    The ID of a name is its rank in lexicographic order, so permuting source
    declarations never changes an identity.
    """

    names: tuple[str, ...] = eqx.field(static=True)
    ids: tuple[int, ...] = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)

    def __init__(self, names: Sequence[str], /) -> None:
        names_ = tuple(str(name) for name in names)
        if (
            any(not name for name in names_)
            or tuple(sorted(names_)) != names_
            or len(set(names_)) != len(names_)
        ):
            raise ValueError("Robot name maps require unique, non-empty, sorted names.")
        ids_host = np.arange(len(names_), dtype=np.int64)
        self.names = names_
        self.ids = tuple(int(identifier) for identifier in ids_host)
        self.mapping_id = canonical_fingerprint(
            {
                "kind": "robot-name-id-map",
                "entries": [
                    [name, int(identifier)] for name, identifier in zip(names_, ids_host)
                ],
                "integer_dtype": "int64",
            }
        )

    @property
    def count(self) -> int:
        return len(self.names)

    @property
    def name_to_id(self) -> tuple[tuple[str, np.int64], ...]:
        return tuple(
            (name, np.int64(self.ids[index])) for index, name in enumerate(self.names)
        )

    @property
    def id_to_name(self) -> tuple[tuple[np.int64, str], ...]:
        return tuple((identifier, name) for name, identifier in self.name_to_id)

    def id_for_name(self, name: str, /) -> np.int64:
        name_ = str(name)
        if name_ not in self.names:
            raise KeyError(name_)
        return np.int64(np.asarray(self.ids, dtype=np.int64)[self.names.index(name_)])

    def name_for_id(self, identifier: int, /) -> str:
        identifier_ = int(identifier)
        if identifier_ < 0 or identifier_ >= len(self.names):
            raise KeyError(identifier_)
        return self.names[identifier_]


class AbstractRobotSourceEvidence(StrictModule, NonTrainableState):
    """Format-specific evidence retained beside an adapted native model."""

    source_id: eqx.AbstractVar[str]
    target_id: eqx.AbstractVar[str]
    evidence_id: eqx.AbstractVar[str]
    loss_paths: eqx.AbstractVar[tuple[str, ...]]


def _host_vector(
    value: ArrayLike, count: int, name: str, /, *, integer: bool = False
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},).")
    if integer:
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"{name} must contain integers.")
        return array.astype(np.int64, copy=False)
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _host_points(value: ArrayLike, count: int, width: int, name: str, /) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (count, width):
        raise ValueError(f"{name} must have shape ({count}, {width}).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _body_pairs(value: ArrayLike, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.issubdtype(
        array.dtype, np.integer
    ):
        raise TypeError(f"{name} must be an integer array of shape (E, 2).")
    canonical = np.sort(array.astype(np.int64), axis=1)
    if np.any(canonical[:, 0] == canonical[:, 1]):
        raise ValueError(f"{name} cannot pair a body with itself.")
    unique = np.unique(canonical, axis=0)
    if unique.shape[0] != canonical.shape[0]:
        raise ValueError(f"{name} must not repeat a pair.")
    return unique


class RobotFramePlan(StrictModule, NonTrainableState):
    """Named source frames rigidly attached to native COM-centered bodies.

    ``translation`` is the frame origin and ``orientation`` the scalar-first
    unit quaternion of the frame axes, both expressed in the body's COM frame,
    so a source link frame offset from its center of mass is retained exactly.
    """

    names: RobotNameIDMap
    body_ids: Array
    translation: Array
    orientation: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        names: RobotNameIDMap,
        body_ids: ArrayLike,
        translation: ArrayLike,
        orientation: ArrayLike,
        /,
    ) -> None:
        if not isinstance(names, RobotNameIDMap):
            raise TypeError("names must be a RobotNameIDMap.")
        count = names.count
        bodies = _host_vector(body_ids, count, "body_ids", integer=True)
        offset = _host_points(translation, count, 3, "translation")
        rotation = _host_points(orientation, count, 4, "orientation")
        if count and np.any(
            np.abs(np.linalg.norm(rotation, axis=-1) - 1.0) > _UNIT_TOLERANCE
        ):
            raise ValueError("Frame orientations must be unit quaternions.")
        self.names = names
        self.body_ids = jnp.asarray(bodies)
        self.translation = jnp.asarray(offset)
        self.orientation = jnp.asarray(rotation)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "robot-frame-plan",
                "names": names.mapping_id,
                "values": array_tree_fingerprint(
                    {
                        "body_ids": bodies,
                        "translation": offset,
                        "orientation": rotation,
                    }
                ),
            }
        )

    @property
    def count(self) -> int:
        return self.names.count

    def local_transforms(self, /) -> Array:
        """Body-to-frame homogeneous transforms in name-ID order."""

        rotation = quaternion_rotation_matrix(self.orientation)
        upper = jnp.concatenate((rotation, self.translation[..., None]), axis=-1)
        bottom = jnp.broadcast_to(
            jnp.asarray([0.0, 0.0, 0.0, 1.0], dtype=upper.dtype),
            (self.count, 1, 4),
        )
        return jnp.concatenate((upper, bottom), axis=-2)


class RobotContactMaterialTable(StrictModule, NonTrainableState):
    """Explicit isotropic Coulomb coefficients for unordered material pairs.

    No mixing rule is implied: every material pair that can meet must have its
    own entry with ``0 <= dynamic_friction <= static_friction``.
    """

    material_pairs: Array
    static_friction: Array
    dynamic_friction: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        material_pairs: ArrayLike,
        static_friction: ArrayLike,
        dynamic_friction: ArrayLike,
        /,
    ) -> None:
        pairs = np.asarray(material_pairs)
        if (
            pairs.ndim != 2
            or pairs.shape[1] != 2
            or not np.issubdtype(pairs.dtype, np.integer)
        ):
            raise TypeError("material_pairs must be an integer array of shape (M, 2).")
        canonical = np.sort(pairs.astype(np.int64), axis=1)
        count = canonical.shape[0]
        static = _host_vector(static_friction, count, "static_friction")
        dynamic = _host_vector(dynamic_friction, count, "dynamic_friction")
        if np.any(dynamic < 0.0) or np.any(dynamic > static):
            raise ValueError(
                "Friction coefficients require 0 <= dynamic_friction <= static_friction."
            )
        order = np.lexsort((canonical[:, 1], canonical[:, 0]))
        canonical = canonical[order]
        if count > 1 and np.any(np.all(canonical[1:] == canonical[:-1], axis=1)):
            raise ValueError("material_pairs must not repeat an unordered pair.")
        static = static[order]
        dynamic = dynamic[order]
        self.material_pairs = jnp.asarray(canonical)
        self.static_friction = jnp.asarray(static)
        self.dynamic_friction = jnp.asarray(dynamic)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "robot-contact-material-table",
                "values": array_tree_fingerprint(
                    {
                        "material_pairs": canonical,
                        "static_friction": static,
                        "dynamic_friction": dynamic,
                    }
                ),
            }
        )

    def entry(self, first: int, second: int, /) -> int:
        """Return the row of an unordered material pair or raise `KeyError`."""

        key = tuple(sorted((int(first), int(second))))
        pairs = np.asarray(self.material_pairs)
        matches = np.flatnonzero((pairs[:, 0] == key[0]) & (pairs[:, 1] == key[1]))
        if matches.size != 1:
            raise KeyError(key)
        return int(matches[0])


class RobotCollisionGeometryPlan(StrictModule, NonTrainableState):
    """Body-attached spheres/capsules and static planes with explicit materials.

    Primitive points are in the owning body's COM frame. A sphere has
    ``local_first == local_second``; a capsule is the swept sphere of the
    segment ``local_first -> local_second``, whose body-attached endpoint order
    defines its axial orientation. Planes are ``{x : n . x = offset}`` in world
    coordinates with free space on ``n . x > offset``. Radii are positive
    physical radii. Same-body pairs, pairs between geometry attached to the
    fixed root, and listed ``excluded_body_pairs`` never form candidates. The
    contact law is the native position-level inelastic hard-contact law with
    isotropic static/dynamic Coulomb friction from ``materials``; it is a native
    declaration, not an equivalent of any provider's regularized contact.
    """

    names: RobotNameIDMap
    body_ids: Array
    local_first: Array
    local_second: Array
    radii: Array
    material_ids: Array
    plane_names: RobotNameIDMap
    plane_normals: Array
    plane_offsets: Array
    plane_material_ids: Array
    excluded_body_pairs: Array
    materials: RobotContactMaterialTable
    kinds: tuple[ArticulatedPrimitiveKind, ...] = eqx.field(static=True)
    contact_law: RobotContactLaw = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        names: RobotNameIDMap,
        kinds: Sequence[ArticulatedPrimitiveKind],
        body_ids: ArrayLike,
        local_first: ArrayLike,
        local_second: ArrayLike,
        radii: ArrayLike,
        material_ids: ArrayLike,
        /,
        *,
        materials: RobotContactMaterialTable,
        plane_names: RobotNameIDMap | None = None,
        plane_normals: ArrayLike | None = None,
        plane_offsets: ArrayLike | None = None,
        plane_material_ids: ArrayLike | None = None,
        excluded_body_pairs: ArrayLike | None = None,
        contact_law: RobotContactLaw = "inelastic-coulomb-hard-contact",
    ) -> None:
        planes = RobotNameIDMap(()) if plane_names is None else plane_names
        if not isinstance(names, RobotNameIDMap) or not isinstance(
            planes, RobotNameIDMap
        ):
            raise TypeError("names and plane_names must be RobotNameIDMap values.")
        if not isinstance(materials, RobotContactMaterialTable):
            raise TypeError("materials must be a RobotContactMaterialTable.")
        law = parse(contact_law, RobotContactLaw, "contact_law")
        count = names.count
        if count == 0:
            raise ValueError(
                "Collision geometry requires at least one body-attached primitive; "
                "static planes alone have no admissible contact pair."
            )
        kinds_ = tuple(ArticulatedPrimitiveKind(kind) for kind in kinds)
        if len(kinds_) != count:
            raise ValueError("kinds must contain one primitive kind per name.")
        bodies = _host_vector(body_ids, count, "body_ids", integer=True)
        first = _host_points(local_first, count, 3, "local_first")
        second = _host_points(local_second, count, 3, "local_second")
        radius = _host_vector(radii, count, "radii")
        materials_ = _host_vector(material_ids, count, "material_ids", integer=True)
        if np.any(radius <= 0.0):
            raise ValueError("Collision radii must be positive physical radii.")
        length = np.linalg.norm(second - first, axis=-1)
        for index, kind in enumerate(kinds_):
            match kind:
                case ArticulatedPrimitiveKind.SPHERE:
                    if np.any(first[index] != second[index]):
                        raise ValueError(
                            f"Sphere {names.names[index]!r} must have equal endpoints."
                        )
                case ArticulatedPrimitiveKind.CAPSULE:
                    if not length[index] > 0.0:
                        raise ValueError(
                            f"Capsule {names.names[index]!r} must have distinct "
                            "endpoints."
                        )
                case unsupported:
                    assert_never(unsupported)
        plane_count = planes.count
        normals = _host_points(
            np.zeros((0, 3)) if plane_normals is None else plane_normals,
            plane_count,
            3,
            "plane_normals",
        )
        offsets = _host_vector(
            np.zeros((0,)) if plane_offsets is None else plane_offsets,
            plane_count,
            "plane_offsets",
        )
        plane_materials = _host_vector(
            np.zeros((0,), dtype=np.int64)
            if plane_material_ids is None
            else plane_material_ids,
            plane_count,
            "plane_material_ids",
            integer=True,
        )
        if plane_count and np.any(
            np.abs(np.linalg.norm(normals, axis=-1) - 1.0) > _UNIT_TOLERANCE
        ):
            raise ValueError("Plane normals must be unit vectors.")
        excluded = _body_pairs(
            np.zeros((0, 2), dtype=np.int64)
            if excluded_body_pairs is None
            else excluded_body_pairs,
            "excluded_body_pairs",
        )
        # Coverage is checked over distinct material IDs, never geometry pairs:
        # host work is O(distinct materials^2) and nothing pairwise is retained.
        declared = {
            (int(first), int(second))
            for first, second in np.asarray(materials.material_pairs).tolist()
        }
        primitive_materials = np.unique(materials_).tolist()
        partners = sorted(set(primitive_materials) | set(np.unique(plane_materials).tolist()))
        for left in primitive_materials:
            for right in partners:
                if (min(left, right), max(left, right)) not in declared:
                    raise ValueError(
                        "The contact material table has no entry for materials "
                        f"{left} and {right}."
                    )
        self.names = names
        self.body_ids = jnp.asarray(bodies)
        self.local_first = jnp.asarray(first)
        self.local_second = jnp.asarray(second)
        self.radii = jnp.asarray(radius)
        self.material_ids = jnp.asarray(materials_)
        self.plane_names = planes
        self.plane_normals = jnp.asarray(normals)
        self.plane_offsets = jnp.asarray(offsets)
        self.plane_material_ids = jnp.asarray(plane_materials)
        self.excluded_body_pairs = jnp.asarray(excluded)
        self.materials = materials
        self.kinds = kinds_
        self.contact_law = law
        self.plan_id = canonical_fingerprint(
            {
                "kind": "robot-collision-geometry-plan",
                "names": names.mapping_id,
                "plane_names": planes.mapping_id,
                "kinds": [int(kind) for kind in kinds_],
                "contact_law": law,
                "materials": materials.plan_id,
                "values": array_tree_fingerprint(
                    {
                        "body_ids": bodies,
                        "local_first": first,
                        "local_second": second,
                        "radii": radius,
                        "material_ids": materials_,
                        "plane_normals": normals,
                        "plane_offsets": offsets,
                        "plane_material_ids": plane_materials,
                        "excluded_body_pairs": excluded,
                    }
                ),
            }
        )

    @property
    def count(self) -> int:
        return self.names.count

    @property
    def plane_count(self) -> int:
        return self.plane_names.count

    def prepare(
        self, articulation: PreparedReducedArticulation, /
    ) -> ArticulatedCollisionGeometry:
        """Bind the declaration to articulated forward kinematics."""

        return ArticulatedCollisionGeometry(
            articulation,
            primitive_ids=np.asarray(self.names.ids, dtype=np.int64),
            primitive_body_ids=np.asarray(self.body_ids, dtype=np.int64),
            primitive_kinds=self.kinds,
            local_first=self.local_first,
            local_second=self.local_second,
            radii=self.radii,
            plane_ids=np.asarray(self.plane_names.ids, dtype=np.int64),
            plane_normals=self.plane_normals,
            plane_offsets=self.plane_offsets,
            excluded_body_pairs=self.excluded_body_pairs,
            primitive_material_ids=self.material_ids,
            plane_material_ids=self.plane_material_ids,
        )


class NativeRobotModelManifest(StrictModule, NonTrainableState):
    """Counts, executed capabilities, and component identities of one model."""

    body_count: int = eqx.field(static=True)
    joint_count: int = eqx.field(static=True)
    hinge_count: int = eqx.field(static=True)
    prismatic_count: int = eqx.field(static=True)
    fixed_count: int = eqx.field(static=True)
    frame_count: int = eqx.field(static=True)
    actuator_count: int = eqx.field(static=True)
    damped_joint_count: int = eqx.field(static=True)
    stopped_joint_count: int = eqx.field(static=True)
    primitive_count: int = eqx.field(static=True)
    plane_count: int = eqx.field(static=True)
    capabilities: tuple[str, ...] = eqx.field(static=True)
    components: tuple[tuple[str, str], ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        counts: Sequence[int],
        capabilities: Sequence[str],
        components: Sequence[tuple[str, str]],
        /,
    ) -> None:
        values = tuple(int(value) for value in counts)
        if len(values) != 11 or any(value < 0 for value in values):
            raise ValueError("Native robot manifests require eleven nonnegative counts.")
        capabilities_ = tuple(sorted(str(item) for item in capabilities))
        components_ = tuple(
            sorted((str(name), str(identity)) for name, identity in components)
        )
        (
            self.body_count,
            self.joint_count,
            self.hinge_count,
            self.prismatic_count,
            self.fixed_count,
            self.frame_count,
            self.actuator_count,
            self.damped_joint_count,
            self.stopped_joint_count,
            self.primitive_count,
            self.plane_count,
        ) = values
        self.capabilities = capabilities_
        self.components = components_
        self.model_id = canonical_fingerprint(
            {
                "kind": "native-robot-model",
                "counts": list(values),
                "capabilities": list(capabilities_),
                "components": [list(item) for item in components_],
            }
        )


def _graph_joint_ids(joints: RigidJointGraphPlan, /) -> tuple[set[int], set[int], set[int]]:
    if joints.ball is not None or joints.distance is not None:
        raise ValueError(
            "The native robot profile admits only fixed, hinge, and prismatic joints."
        )
    groups = (joints.fixed, joints.hinge, joints.prismatic)
    fixed, hinge, prismatic = (
        set() if group is None else {int(item) for item in np.asarray(group.joint_ids)}
        for group in groups
    )
    return fixed, hinge, prismatic


def _require_scalar_joints(
    name: str, joint_ids: Array, scalar: set[int], /
) -> None:
    requested = {int(item) for item in np.asarray(joint_ids)}
    if not requested <= scalar:
        raise ValueError(
            f"{name} must reference hinge or prismatic joints: "
            f"{sorted(requested - scalar)!r}."
        )


def _require_bodies(name: str, body_ids: Array, bodies: set[int], /) -> None:
    requested = {int(item) for item in np.asarray(body_ids).reshape((-1,))}
    if not requested <= bodies:
        raise ValueError(f"{name} reference unknown body IDs {sorted(requested - bodies)!r}.")


class NativeRobotModel(StrictModule, NonTrainableState):
    """The executable native fixed-base robot model.

    ``reference`` is the zero-configuration COM-centered pose used to prepare
    both the joint graph and the articulation. Body IDs equal the IDs of
    ``body_names`` and joint IDs those of ``joint_names``. ``motors`` rows are
    in ``actuator_names`` ID order. Optional components are ``None`` when the
    model does not execute them; nothing is inferred from source limits.
    """

    particles: ParticleSetPlan
    bodies: RigidBodySetPlan
    joints: RigidJointGraphPlan
    articulation: ReducedArticulationPlan
    reference: RigidBodyKinematics
    dimensions: DimensionalScaleContract
    body_names: RobotNameIDMap
    joint_names: RobotNameIDMap
    frames: RobotFramePlan
    actuator_names: RobotNameIDMap
    motors: ScalarJointMotorPlan | None
    damping: ScalarJointDampingPlan | None
    stops: ReducedJointStopPlan | None
    collision: RobotCollisionGeometryPlan | None
    manifest: NativeRobotModelManifest

    @checked
    def __init__(
        self,
        particles: ParticleSetPlan,
        bodies: RigidBodySetPlan,
        joints: RigidJointGraphPlan,
        articulation: ReducedArticulationPlan,
        reference: RigidBodyKinematics,
        dimensions: DimensionalScaleContract,
        body_names: RobotNameIDMap,
        joint_names: RobotNameIDMap,
        frames: RobotFramePlan,
        /,
        *,
        actuator_names: RobotNameIDMap | None = None,
        motors: ScalarJointMotorPlan | None = None,
        damping: ScalarJointDampingPlan | None = None,
        stops: ReducedJointStopPlan | None = None,
        collision: RobotCollisionGeometryPlan | None = None,
    ) -> None:
        actuators = RobotNameIDMap(()) if actuator_names is None else actuator_names
        body_ids = np.asarray(particles.particle_ids, dtype=np.int64)
        if body_ids.tolist() != list(body_names.ids):
            raise ValueError("Particle IDs must equal the body-name IDs in order.")
        body_set = set(body_names.ids)
        count = body_names.count
        if (
            np.asarray(bodies.material_ids).shape != (count,)
            or reference.position.shape != (count, 3)
            or reference.velocity.shape != (count, 3)
            or reference.orientation.shape != (count, 4)
            or reference.angular_velocity.shape != (count, 3)
        ):
            raise ValueError("Body plans and reference kinematics must share capacity.")
        if particles.ambient_dimension != 3:
            raise ValueError("Native robot models are three-dimensional.")
        root = articulation.root_body_id
        if root not in body_set:
            raise ValueError("The articulation root is not a declared body.")
        root_index = body_names.ids.index(root)
        if not bool(np.asarray(bodies.fixed_mask)[root_index]):
            raise ValueError("The articulation root must be a fixed body.")
        if not float(np.asarray(particles.masses)[root_index]) > 0.0:
            raise ValueError(
                "A massless world anchor is not a supported native root; declare "
                "a physical fixed base link with its own mass properties."
            )
        fixed, hinge, prismatic = _graph_joint_ids(joints)
        graph_joints = fixed | hinge | prismatic
        if sorted(graph_joints) != list(joint_names.ids):
            raise ValueError("Joint-graph IDs must equal the joint-name IDs.")
        if {int(item) for item in np.asarray(articulation.joint_ids)} != graph_joints:
            raise ValueError("The articulation must select every joint-graph edge.")
        scalar = hinge | prismatic
        if (motors is None) != (actuators.count == 0):
            raise ValueError("motors and non-empty actuator_names must be supplied together.")
        if motors is not None:
            if motors.count != actuators.count:
                raise ValueError("motors must have one row per actuator name.")
            _require_scalar_joints("motors", motors.joint_ids, scalar)
        if damping is not None:
            _require_scalar_joints("damping", damping.joint_ids, scalar)
        if stops is not None:
            _require_scalar_joints("stops", stops.joint_ids, scalar)
        _require_bodies("frames", frames.body_ids, body_set)
        if collision is not None:
            _require_bodies("collision primitives", collision.body_ids, body_set)
            _require_bodies(
                "excluded collision body pairs", collision.excluded_body_pairs, body_set
            )
        # One string identity covering the array signature and content digest.
        reference_id = canonical_fingerprint(
            {
                "kind": "robot-reference-kinematics",
                "values": array_tree_fingerprint(
                    {
                        "position": np.asarray(reference.position),
                        "velocity": np.asarray(reference.velocity),
                        "orientation": np.asarray(reference.orientation),
                        "angular_velocity": np.asarray(reference.angular_velocity),
                    }
                ),
            }
        )
        components = [
            ("particles", particles.plan_id),
            ("bodies", bodies.plan_id),
            ("joints", joints.plan_id),
            ("articulation", articulation.plan_id),
            ("reference", reference_id),
            ("dimensions", dimensions.scale_id),
            ("body_names", body_names.mapping_id),
            ("joint_names", joint_names.mapping_id),
            ("frames", frames.plan_id),
            ("actuator_names", actuators.mapping_id),
        ]
        capabilities = ["fixed-base-reduced-tree"]
        if motors is not None:
            components.append(("motors", motors.plan_id))
            capabilities.append("stateless-scalar-joint-motors")
        if damping is not None:
            components.append(("damping", damping.plan_id))
            capabilities.append("executed-viscous-joint-damping")
        if stops is not None:
            components.append(("stops", stops.plan_id))
            capabilities.append("unilateral-joint-stops")
        if collision is not None:
            components.append(("collision", collision.plan_id))
            capabilities.append(collision.contact_law)
            if collision.count:
                capabilities.append("body-attached-sphere-capsule-collision")
            if collision.plane_count:
                capabilities.append("static-plane-collision")
        manifest = NativeRobotModelManifest(
            (
                count,
                joint_names.count,
                len(hinge),
                len(prismatic),
                len(fixed),
                frames.count,
                actuators.count,
                0 if damping is None else damping.count,
                0 if stops is None else stops.count,
                0 if collision is None else collision.count,
                0 if collision is None else collision.plane_count,
            ),
            capabilities,
            components,
        )
        self.particles = particles
        self.bodies = bodies
        self.joints = joints
        self.articulation = articulation
        self.reference = reference
        self.dimensions = dimensions
        self.body_names = body_names
        self.joint_names = joint_names
        self.frames = frames
        self.actuator_names = actuators
        self.motors = motors
        self.damping = damping
        self.stops = stops
        self.collision = collision
        self.manifest = manifest

    @property
    def model_id(self) -> str:
        return self.manifest.model_id

    def prepare(self, /, *, maximum_turns: int = 1024) -> PreparedNativeRobotModel:
        return PreparedNativeRobotModel(self, maximum_turns=maximum_turns)


class PreparedNativeRobotModel(StrictModule, NonTrainableState):
    """All native owners of one model prepared against its exact reference."""

    model: NativeRobotModel
    particles: ParticleDiscretization
    bodies: PreparedRigidBodySet
    reference: RigidBodyKinematics
    graph: PreparedRigidJointGraph
    articulation: PreparedReducedArticulation
    coordinates: PreparedLiftedJointCoordinates
    motors: PreparedScalarJointMotors | None
    damping: PreparedScalarJointDamping | None
    stops: PreparedReducedJointStops | None
    collision: ArticulatedCollisionGeometry | None
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, model: NativeRobotModel, /, *, maximum_turns: int = 1024
    ) -> None:
        particles = model.particles.prepare()
        bodies = model.bodies.prepare(particles)
        reference = bodies.kinematics(
            model.reference.position,
            model.reference.velocity,
            model.reference.orientation,
            model.reference.angular_velocity,
        )
        graph = model.joints.prepare(bodies, reference)
        articulation = model.articulation.prepare(graph, reference)
        coordinates = PreparedLiftedJointCoordinates(
            articulation, maximum_turns=maximum_turns
        )
        motors = None if model.motors is None else model.motors.prepare(articulation)
        damping = None if model.damping is None else model.damping.prepare(articulation)
        stops = None if model.stops is None else model.stops.prepare(articulation)
        collision = (
            None if model.collision is None else model.collision.prepare(articulation)
        )
        self.model = model
        self.particles = particles
        self.bodies = bodies
        self.reference = reference
        self.graph = graph
        self.articulation = articulation
        self.coordinates = coordinates
        self.motors = motors
        self.damping = damping
        self.stops = stops
        self.collision = collision
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-robot-model",
                "model": model.model_id,
                "articulation": articulation.prepared_id,
                "coordinates": coordinates.prepared_id,
            }
        )


type NativeTreeJointKind = Literal["fixed", "hinge", "prismatic"]


class NativeTreeJoint(StrictModule, NonTrainableState):
    """One oriented tree edge at the zero configuration, in world coordinates.

    ``anchor`` is the joint point and ``axis`` the unit hinge/slide axis of
    the zero configuration; both are ignored for fixed joints, whose child is
    welded at its reference pose.
    """

    name: str = eqx.field(static=True)
    kind: NativeTreeJointKind = eqx.field(static=True)
    parent: str = eqx.field(static=True)
    child: str = eqx.field(static=True)
    anchor: tuple[float, float, float] = eqx.field(static=True)
    axis: tuple[float, float, float] = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        kind: NativeTreeJointKind,
        parent: str,
        child: str,
        /,
        *,
        anchor: Sequence[float] = (0.0, 0.0, 0.0),
        axis: Sequence[float] = (1.0, 0.0, 0.0),
    ) -> None:
        kind_ = parse(kind, NativeTreeJointKind, "kind")
        anchor_ = _host_points(np.asarray(anchor)[None, :], 1, 3, "anchor")[0]
        axis_ = _host_points(np.asarray(axis)[None, :], 1, 3, "axis")[0]
        if kind_ != "fixed" and abs(float(np.linalg.norm(axis_)) - 1.0) > _UNIT_TOLERANCE:
            raise ValueError("Hinge and prismatic joint axes must be unit vectors.")
        if not str(name) or not str(parent) or not str(child) or parent == child:
            raise ValueError("Tree joints need a name and two distinct body names.")
        self.name = str(name)
        self.kind = kind_
        self.parent = str(parent)
        self.child = str(child)
        self.anchor = (float(anchor_[0]), float(anchor_[1]), float(anchor_[2]))
        self.axis = (float(axis_[0]), float(axis_[1]), float(axis_[2]))


def _joint_set_arrays(
    joints: Sequence[NativeTreeJoint],
    joint_names: RobotNameIDMap,
    body_names: RobotNameIDMap,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(joints, key=lambda joint: joint.name)
    return (
        np.asarray([joint_names.id_for_name(joint.name) for joint in ordered], np.int64),
        np.asarray([body_names.id_for_name(joint.parent) for joint in ordered], np.int64),
        np.asarray([body_names.id_for_name(joint.child) for joint in ordered], np.int64),
        np.asarray([joint.anchor for joint in ordered], np.float64).reshape((-1, 3)),
        np.asarray([joint.axis for joint in ordered], np.float64).reshape((-1, 3)),
    )


def lower_native_robot_tree(
    name: str,
    body_names: RobotNameIDMap,
    masses: ArrayLike,
    inertia_body: ArrayLike,
    reference_position: ArrayLike,
    reference_orientation: ArrayLike,
    root_body: str,
    joints: Sequence[NativeTreeJoint],
    dimensions: DimensionalScaleContract,
    frames: RobotFramePlan,
    /,
    *,
    actuator_names: RobotNameIDMap | None = None,
    motors: ScalarJointMotorPlan | None = None,
    damping: ScalarJointDampingPlan | None = None,
    stops: ReducedJointStopPlan | None = None,
    collision: RobotCollisionGeometryPlan | None = None,
) -> NativeRobotModel:
    """Lower a fixed-base tree of COM-centered bodies into a native model.

    This is the one shared host lowering for source adapters. Arrays are in
    ``body_names`` ID order: masses, COM inertia tensors in body axes, and the
    zero-configuration COM positions and scalar-first body orientations.
    ``joints`` are in the source traversal order used for the articulation;
    joint IDs are the ranks of the sorted joint names. Only ``root_body`` is
    fixed. Body and joint IDs used by the optional components must be the
    ``body_names``/sorted-joint-name IDs.
    """

    if not isinstance(body_names, RobotNameIDMap):
        raise TypeError("body_names must be a RobotNameIDMap.")
    joints_ = tuple(joints)
    if not all(isinstance(joint, NativeTreeJoint) for joint in joints_):
        raise TypeError("joints must contain NativeTreeJoint values.")
    count = body_names.count
    mass = _host_vector(masses, count, "masses")
    inertia = np.asarray(inertia_body, dtype=np.float64)
    if inertia.shape != (count, 3, 3) or not np.all(np.isfinite(inertia)):
        raise ValueError(f"inertia_body must be finite with shape ({count}, 3, 3).")
    position = _host_points(reference_position, count, 3, "reference_position")
    orientation = _host_points(reference_orientation, count, 4, "reference_orientation")
    if np.any(np.abs(np.linalg.norm(orientation, axis=-1) - 1.0) > _UNIT_TOLERANCE):
        raise ValueError("Reference orientations must be unit quaternions.")
    joint_names = RobotNameIDMap(tuple(sorted(joint.name for joint in joints_)))
    known = frozenset(body_names.names)
    if root_body not in known or any(
        joint.parent not in known or joint.child not in known for joint in joints_
    ):
        raise ValueError("Tree joints and the root must reference declared bodies.")
    body_ids = np.asarray(body_names.ids, dtype=np.int64)
    particles = ParticleSetPlan(
        body_ids,
        mass,
        ambient_dimension=3,
        name=f"{name}-links",
        domain_labels=("rigid_body", "robot_link"),
        coordinate_dtype="float64",
    )
    bodies = RigidBodySetPlan(
        np.zeros((count,), dtype=np.int64),
        inertia,
        fixed_mask=body_ids == int(body_names.id_for_name(root_body)),
        name=f"{name}-rigid-bodies",
    )
    reference = RigidBodyKinematics(
        jnp.asarray(position),
        jnp.zeros((count, 3), dtype=jnp.asarray(position).dtype),
        jnp.asarray(orientation),
        jnp.zeros((count, 3), dtype=jnp.asarray(position).dtype),
    )
    groups: dict[NativeTreeJointKind, list[NativeTreeJoint]] = {
        "fixed": [],
        "hinge": [],
        "prismatic": [],
    }
    for joint in joints_:
        groups[joint.kind].append(joint)
    fixed = groups["fixed"]
    hinge = groups["hinge"]
    prismatic = groups["prismatic"]
    fixed_arrays = _joint_set_arrays(fixed, joint_names, body_names)
    hinge_arrays = _joint_set_arrays(hinge, joint_names, body_names)
    prismatic_arrays = _joint_set_arrays(prismatic, joint_names, body_names)
    graph = RigidJointGraphPlan(
        fixed=None if not fixed else FixedJointSetPlan(*fixed_arrays[:3]),
        hinge=None if not hinge else HingeJointSetPlan(*hinge_arrays),
        prismatic=None if not prismatic else PrismaticJointSetPlan(*prismatic_arrays),
    )
    articulation = ReducedArticulationPlan(
        int(body_names.id_for_name(root_body)),
        np.asarray([joint_names.id_for_name(joint.name) for joint in joints_], np.int64),
        np.asarray([body_names.id_for_name(joint.parent) for joint in joints_], np.int64),
        np.asarray([body_names.id_for_name(joint.child) for joint in joints_], np.int64),
    )
    return NativeRobotModel(
        particles,
        bodies,
        graph,
        articulation,
        reference,
        dimensions,
        body_names,
        joint_names,
        frames,
        actuator_names=actuator_names,
        motors=motors,
        damping=damping,
        stops=stops,
        collision=collision,
    )


SourceEvidence = TypeVar("SourceEvidence", bound=AbstractRobotSourceEvidence)


class RobotAdaptation(StrictModule, NonTrainableState, Generic[SourceEvidence]):
    """One source adaptation: the native model, its report, and source evidence.

    The evidence type is the adapter's own format evidence, so consumers keep
    its exact source fields while the model stays source-neutral.
    """

    model: NativeRobotModel
    report: AdapterReport
    evidence: SourceEvidence

    def __init__(
        self,
        model: NativeRobotModel,
        report: AdapterReport,
        evidence: SourceEvidence,
        /,
    ) -> None:
        if not isinstance(model, NativeRobotModel):
            raise TypeError("model must be a NativeRobotModel.")
        if not isinstance(report, AdapterReport):
            raise TypeError("report must be an AdapterReport.")
        if not isinstance(evidence, AbstractRobotSourceEvidence):
            raise TypeError("evidence must be robot source evidence.")
        if not report.valid:
            raise ValueError("A robot adaptation requires a valid adapter report.")
        if report.target_id != model.model_id or evidence.target_id != model.model_id:
            raise ValueError("Adapter report and evidence must target the native model.")
        if report.source_id != evidence.source_id:
            raise ValueError("Adapter report and evidence must share one source.")
        self.model = model
        self.report = report
        self.evidence = evidence


__all__ = [
    "AbstractRobotSourceEvidence",
    "NativeRobotModel",
    "NativeRobotModelManifest",
    "NativeTreeJoint",
    "NativeTreeJointKind",
    "PreparedNativeRobotModel",
    "RobotAdaptation",
    "RobotCollisionGeometryPlan",
    "RobotContactLaw",
    "RobotContactMaterialTable",
    "RobotFramePlan",
    "RobotNameIDMap",
    "lower_native_robot_tree",
]
