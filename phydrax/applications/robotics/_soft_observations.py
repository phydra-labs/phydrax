#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import PlantRuntimeState, PlantStepResult
from ..solid_mechanics._rod_plant import (
    PreparedReducedRodPlant,
    ReducedRodPlantEvidence,
    ReducedRodPlantState,
)
from ..solid_mechanics._rod_reconstruction import (
    prepare_rod_reconstruction,
    PreparedRodReconstruction,
    RodReconstructionEvaluation,
    RodReconstructionPlan,
)
from ..solid_mechanics._rod_reduced_integrators import ReducedRodStepResult
from ..solid_mechanics._rod_reduction import (
    ReducedRodEvaluation,
    ReducedRodState,
)
from ..solid_mechanics._rod_tendon import (
    PreparedFrictionlessElasticTendon,
    TendonActuatorState,
)


SoftTwistKind: TypeAlias = Literal["body", "world_origin", "frame_world"]

_FRAME_CONVENTION = "material-body/world-origin/world-frame"
_DIRECT_SENSOR_ID = "mechanics-direct"
_DIRECT_SENSOR_PLAN_ID = canonical_fingerprint(
    {"kind": "direct-noiseless-soft-robot-observation"}
)


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be a non-empty string.")
    return value.strip()


def _string_tuple(values: tuple[str, ...], owner: str, /) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{owner} must be a tuple of strings.")
    result = tuple(_identifier(value, owner) for value in values)
    return result


def _real_scalar(value: float, owner: str, /, *, nonnegative: bool = False) -> float:
    result = float(value)
    if not isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{owner} must be {qualifier}.")
    return result


def _component_vector(
    value: ArrayLike,
    size: int,
    dtype: np.dtype,
    owner: str,
    /,
    *,
    nonnegative: bool = False,
) -> Array:
    host = np.asarray(value)
    if host.ndim == 0:
        host = np.full((size,), host, dtype=dtype)
    elif host.shape == (size,):
        host = host.astype(dtype, copy=False)
    else:
        raise ValueError(f"{owner} must be scalar or have shape ({size},).")
    if not np.issubdtype(host.dtype, np.floating) or not np.all(np.isfinite(host)):
        raise TypeError(f"{owner} must be finite and real floating-point data.")
    if nonnegative and np.any(host < 0.0):
        raise ValueError(f"{owner} must be nonnegative componentwise.")
    return jnp.asarray(host)


def _runtime_key_data(key: ArrayLike, /) -> Array:
    value = jnp.asarray(key)
    if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        if value.shape != ():
            raise ValueError(
                "A scalar soft-robot observation requires one typed PRNG key."
            )
    else:
        if value.dtype != jnp.uint32:
            raise TypeError("A legacy plant PRNG key must have uint32 dtype.")
        if value.shape != (2,):
            raise ValueError(
                "A scalar soft-robot observation requires one size-two PRNG key."
            )
    return jax.random.key_data(value)


def _tree_exact_equal(left, right, /) -> Array:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree or len(left_leaves) != len(right_leaves):
        return jnp.asarray(False)
    equal = jnp.asarray(True)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False)
        equal = equal & jnp.all(left_array == right_array)
    return equal


def _runtime_ids(state: PlantRuntimeState, /) -> tuple[str, str, str, str]:
    return (
        state.semantic_provenance_id,
        state.numeric_revision_id,
        state.state_schema_id,
        state.execution_signature_id,
    )


class SoftObservationLayout(StrictModule, NonTrainableState):
    """Exact flat observation ABI with per-component unit, frame, and query origin."""

    component_names: tuple[str, ...] = eqx.field(static=True)
    component_units: tuple[str, ...] = eqx.field(static=True)
    component_frames: tuple[str, ...] = eqx.field(static=True)
    component_query_ids: tuple[str, ...] = eqx.field(static=True)
    groups: tuple[tuple[str, int, int], ...] = eqx.field(static=True)
    size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_names: tuple[str, ...],
        component_units: tuple[str, ...],
        component_frames: tuple[str, ...],
        component_query_ids: tuple[str, ...],
        groups: tuple[tuple[str, int, int], ...],
        /,
    ):
        names = _string_tuple(component_names, "component_names")
        units = _string_tuple(component_units, "component_units")
        frames = _string_tuple(component_frames, "component_frames")
        queries = _string_tuple(component_query_ids, "component_query_ids")
        size = len(names)
        if size < 1:
            raise ValueError(
                "A soft observation layout must contain at least one component."
            )
        if len(set(names)) != size:
            raise ValueError("Soft observation component names must be unique.")
        if len(units) != size or len(frames) != size or len(queries) != size:
            raise ValueError(
                "Names, units, frames, and query IDs must have identical component counts."
            )
        if not isinstance(groups, tuple) or not groups:
            raise ValueError("groups must be a nonempty tuple covering the layout.")
        checked_groups: list[tuple[str, int, int]] = []
        expected_start = 0
        for item in groups:
            if not isinstance(item, tuple) or len(item) != 3:
                raise TypeError("Every layout group must be (name, start, stop).")
            name = _identifier(item[0], "group name")
            start, stop = int(item[1]), int(item[2])
            if start != item[1] or stop != item[2]:
                raise TypeError("Layout group bounds must be integers.")
            if start != expected_start or stop <= start or stop > size:
                raise ValueError("Layout groups must form one ordered exact partition.")
            checked_groups.append((name, start, stop))
            expected_start = stop
        if expected_start != size:
            raise ValueError("Layout groups must cover every observation component.")
        if len({name for name, _, _ in checked_groups}) != len(checked_groups):
            raise ValueError("Soft observation group names must be unique.")
        groups_ = tuple(checked_groups)
        self.component_names = names
        self.component_units = units
        self.component_frames = frames
        self.component_query_ids = queries
        self.groups = groups_
        self.size = size
        self.layout_id = canonical_fingerprint(
            {
                "kind": "soft-robot-observation-layout",
                "names": names,
                "units": units,
                "frames": frames,
                "query_ids": queries,
                "groups": groups_,
            }
        )

    def slice_for(self, group: str, /) -> slice:
        name = _identifier(group, "group")
        for candidate, start, stop in self.groups:
            if candidate == name:
                return slice(start, stop)
        raise KeyError(f"Unknown soft observation group {name!r}.")

    def index_for(self, component: str, /) -> int:
        name = _identifier(component, "component")
        try:
            return self.component_names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown soft observation component {name!r}.") from error


class SoftRobotObservation(StrictModule):
    """One identity-bound observation with explicit ideal, bias, noise, and age."""

    values: Array
    ideal_values: Array
    bias: Array
    noise: Array
    source_key: Array
    noise_key: Array
    timestamp: Array
    epoch: Array
    sample_timestamp: Array
    sample_epoch: Array
    age: Array
    fresh: Array
    sample_held: Array
    finite: Array
    mechanics_valid: Array
    valid: Array
    layout: SoftObservationLayout
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)
    query_plan_id: str = eqx.field(static=True)
    query_ids: tuple[str, ...] = eqx.field(static=True)
    sensor_id: str = eqx.field(static=True)
    sensor_plan_id: str = eqx.field(static=True)
    observation_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        ideal_values: ArrayLike,
        bias: ArrayLike,
        noise: ArrayLike,
        source_key: ArrayLike,
        noise_key: ArrayLike,
        timestamp: ArrayLike,
        epoch: ArrayLike,
        sample_timestamp: ArrayLike,
        sample_epoch: ArrayLike,
        age: ArrayLike,
        fresh: ArrayLike,
        sample_held: ArrayLike,
        finite: ArrayLike,
        mechanics_valid: ArrayLike,
        valid: ArrayLike,
        layout: SoftObservationLayout,
        semantic_provenance_id: str,
        numeric_revision_id: str,
        state_schema_id: str,
        execution_signature_id: str,
        query_plan_id: str,
        query_ids: tuple[str, ...],
        sensor_id: str,
        sensor_plan_id: str,
        observation_plan_id: str,
        /,
    ):
        if not isinstance(layout, SoftObservationLayout):
            raise TypeError("layout must be SoftObservationLayout.")
        arrays = tuple(
            jnp.asarray(value) for value in (values, ideal_values, bias, noise)
        )
        if any(value.shape != (layout.size,) for value in arrays):
            raise ValueError(
                "Observation value, ideal, bias, and noise arrays must match layout."
            )
        if any(value.dtype != arrays[0].dtype for value in arrays[1:]):
            raise TypeError(
                "Observation value, ideal, bias, and noise arrays must share a dtype."
            )
        source_key_ = jnp.asarray(source_key, dtype=jnp.uint32)
        noise_key_ = jnp.asarray(noise_key, dtype=jnp.uint32)
        if source_key_.shape != (2,) or noise_key_.shape != (2,):
            raise ValueError("Observation source and noise keys must be uint32 pairs.")
        scalars = tuple(
            jnp.asarray(value)
            for value in (
                timestamp,
                epoch,
                sample_timestamp,
                sample_epoch,
                age,
                fresh,
                sample_held,
                finite,
                mechanics_valid,
                valid,
            )
        )
        if any(value.shape != () for value in scalars):
            raise ValueError(
                "Observation time, epoch, age, and evidence must be scalars."
            )
        query_ids_ = _string_tuple(query_ids, "query_ids")
        if not query_ids_:
            raise ValueError("query_ids must identify at least one prepared query.")
        self.values, self.ideal_values, self.bias, self.noise = arrays
        self.source_key = source_key_
        self.noise_key = noise_key_
        (
            self.timestamp,
            self.epoch,
            self.sample_timestamp,
            self.sample_epoch,
            self.age,
            self.fresh,
            self.sample_held,
            self.finite,
            self.mechanics_valid,
            self.valid,
        ) = scalars
        self.layout = layout
        self.semantic_provenance_id = _identifier(
            semantic_provenance_id, "semantic_provenance_id"
        )
        self.numeric_revision_id = _identifier(numeric_revision_id, "numeric_revision_id")
        self.state_schema_id = _identifier(state_schema_id, "state_schema_id")
        self.execution_signature_id = _identifier(
            execution_signature_id, "execution_signature_id"
        )
        self.query_plan_id = _identifier(query_plan_id, "query_plan_id")
        self.query_ids = query_ids_
        self.sensor_id = _identifier(sensor_id, "sensor_id")
        self.sensor_plan_id = _identifier(sensor_plan_id, "sensor_plan_id")
        self.observation_plan_id = _identifier(observation_plan_id, "observation_plan_id")

    @property
    def step_index(self) -> Array:
        return self.epoch

    @property
    def sample_step_index(self) -> Array:
        return self.sample_epoch


class SoftReducedStateQueryPlan(StrictModule, NonTrainableState):
    """Select dimensionless reduced coordinates and/or their physical-time rates."""

    include_configuration: bool = eqx.field(static=True)
    include_velocity: bool = eqx.field(static=True)
    coordinate_unit: str = eqx.field(static=True)
    velocity_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        include_configuration: bool = True,
        include_velocity: bool = True,
        coordinate_unit: str = "1",
        velocity_unit: str = "s^-1",
    ):
        if not isinstance(include_configuration, bool) or not isinstance(
            include_velocity, bool
        ):
            raise TypeError("Reduced-state selection flags must be bool values.")
        if not include_configuration and not include_velocity:
            raise ValueError(
                "A reduced-state query must select configuration or velocity."
            )
        coordinate = _identifier(coordinate_unit, "coordinate_unit")
        velocity = _identifier(velocity_unit, "velocity_unit")
        self.include_configuration = include_configuration
        self.include_velocity = include_velocity
        self.coordinate_unit = coordinate
        self.velocity_unit = velocity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "soft-reduced-state-query-plan",
                "configuration": include_configuration,
                "velocity": include_velocity,
                "coordinate_unit": coordinate,
                "velocity_unit": velocity,
            }
        )


class SoftFrameQueryPlan(StrictModule, NonTrainableState):
    """Pose and declared twist views at arbitrary physical arc-length frames."""

    reconstruction: RodReconstructionPlan
    twists: tuple[SoftTwistKind, ...] = eqx.field(static=True)
    include_pose: bool = eqx.field(static=True)
    position_unit: str = eqx.field(static=True)
    linear_velocity_unit: str = eqx.field(static=True)
    angular_velocity_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: RodReconstructionPlan,
        /,
        *,
        include_pose: bool = True,
        twists: tuple[SoftTwistKind, ...] = ("frame_world",),
        position_unit: str = "m",
        linear_velocity_unit: str = "m/s",
        angular_velocity_unit: str = "rad/s",
    ):
        if not isinstance(reconstruction, RodReconstructionPlan):
            raise TypeError("reconstruction must be RodReconstructionPlan.")
        if not isinstance(include_pose, bool):
            raise TypeError("include_pose must be bool.")
        if not isinstance(twists, tuple) or any(
            value not in ("body", "world_origin", "frame_world") for value in twists
        ):
            raise ValueError(
                "twists must contain only 'body', 'world_origin', or 'frame_world'."
            )
        if len(set(twists)) != len(twists):
            raise ValueError("Frame twist kinds must be unique.")
        if not include_pose and not twists:
            raise ValueError("A frame query must select pose or at least one twist.")
        position = _identifier(position_unit, "position_unit")
        linear = _identifier(linear_velocity_unit, "linear_velocity_unit")
        angular = _identifier(angular_velocity_unit, "angular_velocity_unit")
        self.reconstruction = reconstruction
        self.twists = twists
        self.include_pose = include_pose
        self.position_unit = position
        self.linear_velocity_unit = linear
        self.angular_velocity_unit = angular
        self.plan_id = canonical_fingerprint(
            {
                "kind": "soft-frame-pose-twist-query-plan",
                "reconstruction": reconstruction.plan_id,
                "pose": include_pose,
                "twists": twists,
                "position_unit": position,
                "linear_velocity_unit": linear,
                "angular_velocity_unit": angular,
            }
        )


class SoftStrainQueryPlan(StrictModule, NonTrainableState):
    """Total material strain and optional reduced increment at arbitrary arc lengths."""

    reconstruction: RodReconstructionPlan
    include_total: bool = eqx.field(static=True)
    include_reduced: bool = eqx.field(static=True)
    stretch_shear_unit: str = eqx.field(static=True)
    bend_twist_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: RodReconstructionPlan,
        /,
        *,
        include_total: bool = True,
        include_reduced: bool = False,
        stretch_shear_unit: str = "1",
        bend_twist_unit: str = "rad/m",
    ):
        if not isinstance(reconstruction, RodReconstructionPlan):
            raise TypeError("reconstruction must be RodReconstructionPlan.")
        if not isinstance(include_total, bool) or not isinstance(include_reduced, bool):
            raise TypeError("Strain selection flags must be bool values.")
        if not include_total and not include_reduced:
            raise ValueError("A strain query must select total or reduced strain.")
        stretch = _identifier(stretch_shear_unit, "stretch_shear_unit")
        bend = _identifier(bend_twist_unit, "bend_twist_unit")
        self.reconstruction = reconstruction
        self.include_total = include_total
        self.include_reduced = include_reduced
        self.stretch_shear_unit = stretch
        self.bend_twist_unit = bend
        self.plan_id = canonical_fingerprint(
            {
                "kind": "soft-material-strain-query-plan",
                "reconstruction": reconstruction.plan_id,
                "total": include_total,
                "reduced": include_reduced,
                "stretch_shear_unit": stretch,
                "bend_twist_unit": bend,
            }
        )


class SoftTendonQueryPlan(StrictModule, NonTrainableState):
    """Fixed tendon route-length, rate, unilateral tension, and stored-energy query."""

    tendons: tuple[PreparedFrictionlessElasticTendon, ...]
    tendon_names: tuple[str, ...] = eqx.field(static=True)
    include_length: bool = eqx.field(static=True)
    include_length_rate: bool = eqx.field(static=True)
    include_tension: bool = eqx.field(static=True)
    include_stored_energy: bool = eqx.field(static=True)
    length_unit: str = eqx.field(static=True)
    length_rate_unit: str = eqx.field(static=True)
    tension_unit: str = eqx.field(static=True)
    energy_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        tendons: tuple[PreparedFrictionlessElasticTendon, ...],
        /,
        *,
        tendon_names: tuple[str, ...] | None = None,
        include_length: bool = True,
        include_length_rate: bool = False,
        include_tension: bool = True,
        include_stored_energy: bool = False,
        length_unit: str = "m",
        length_rate_unit: str = "m/s",
        tension_unit: str = "N",
        energy_unit: str = "J",
    ):
        if not isinstance(tendons, tuple) or not tendons:
            raise ValueError("tendons must be a nonempty tuple of prepared tendons.")
        if not all(
            isinstance(value, PreparedFrictionlessElasticTendon) for value in tendons
        ):
            raise TypeError("Every tendon query entry must be prepared.")
        flags = (
            include_length,
            include_length_rate,
            include_tension,
            include_stored_energy,
        )
        if any(not isinstance(value, bool) for value in flags):
            raise TypeError("Tendon selection flags must be bool values.")
        if not any(flags):
            raise ValueError("A tendon query must select at least one observable.")
        names = (
            tuple(f"tendon_{index}" for index in range(len(tendons)))
            if tendon_names is None
            else _string_tuple(tendon_names, "tendon_names")
        )
        if len(names) != len(tendons) or len(set(names)) != len(names):
            raise ValueError("tendon_names must uniquely name every prepared tendon.")
        length = _identifier(length_unit, "length_unit")
        length_rate = _identifier(length_rate_unit, "length_rate_unit")
        tension = _identifier(tension_unit, "tension_unit")
        energy = _identifier(energy_unit, "energy_unit")
        self.tendons = tendons
        self.tendon_names = names
        self.include_length = include_length
        self.include_length_rate = include_length_rate
        self.include_tension = include_tension
        self.include_stored_energy = include_stored_energy
        self.length_unit = length
        self.length_rate_unit = length_rate
        self.tension_unit = tension
        self.energy_unit = energy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "soft-frictionless-tendon-observation-query-plan",
                "tendons": tuple(value.tendon_id for value in tendons),
                "names": names,
                "length": include_length,
                "length_rate": include_length_rate,
                "tension": include_tension,
                "stored_energy": include_stored_energy,
                "units": (length, length_rate, tension, energy),
            }
        )

    @property
    def requires_actuator_state(self) -> bool:
        return self.include_tension or self.include_stored_energy


class SoftEnergyLoadQueryPlan(StrictModule, NonTrainableState):
    """Native-authority energy/load view with an optional complete accepted-step ledger."""

    include_mechanics: bool = eqx.field(static=True)
    include_step_ledger: bool = eqx.field(static=True)
    energy_unit: str = eqx.field(static=True)
    power_unit: str = eqx.field(static=True)
    reduced_effort_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        include_mechanics: bool = True,
        include_step_ledger: bool = False,
        energy_unit: str = "J",
        power_unit: str = "W",
        reduced_effort_unit: str = "J",
    ):
        if not isinstance(include_mechanics, bool) or not isinstance(
            include_step_ledger, bool
        ):
            raise TypeError("Energy/load selection flags must be bool values.")
        if not include_mechanics and not include_step_ledger:
            raise ValueError(
                "An energy/load query must select mechanics or a step ledger."
            )
        energy = _identifier(energy_unit, "energy_unit")
        power = _identifier(power_unit, "power_unit")
        effort = _identifier(reduced_effort_unit, "reduced_effort_unit")
        self.include_mechanics = include_mechanics
        self.include_step_ledger = include_step_ledger
        self.energy_unit = energy
        self.power_unit = power
        self.reduced_effort_unit = effort
        self.plan_id = canonical_fingerprint(
            {
                "kind": "soft-native-energy-load-ledger-query-plan",
                "mechanics": include_mechanics,
                "step_ledger": include_step_ledger,
                "energy_unit": energy,
                "power_unit": power,
                "reduced_effort_unit": effort,
            }
        )


class SoftSensorPlan(StrictModule, NonTrainableState):
    """Explicit additive sensor law; randomness is derived only from the plant key."""

    noise_standard_deviation: Array
    sample_period: float = eqx.field(static=True)
    sensor_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sensor_id: str,
        /,
        *,
        noise_standard_deviation: ArrayLike = 0.0,
        sample_period: float = 0.0,
    ):
        identifier = _identifier(sensor_id, "sensor_id")
        noise = np.asarray(noise_standard_deviation)
        if noise.ndim > 1 or not np.issubdtype(noise.dtype, np.floating):
            raise TypeError("noise_standard_deviation must be a real scalar or vector.")
        if not np.all(np.isfinite(noise)) or np.any(noise < 0.0):
            raise ValueError("noise_standard_deviation must be finite and nonnegative.")
        period = _real_scalar(sample_period, "sample_period", nonnegative=True)
        self.noise_standard_deviation = jnp.asarray(noise)
        self.sample_period = period
        self.sensor_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "explicit-soft-robot-additive-sensor-plan",
                "sensor_id": identifier,
                "noise": array_tree_fingerprint(noise),
                "sample_period": period,
            }
        )


class SoftTendonObservationState(StrictModule):
    """Explicit tendon actuator snapshot bound to one plant state and tendon query."""

    actuator_states: tuple[TendonActuatorState, ...]
    source_key: Array
    timestamp: Array
    epoch: Array
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)
    tendon_query_id: str = eqx.field(static=True)

    def __init__(
        self,
        actuator_states: tuple[TendonActuatorState, ...],
        source_key: ArrayLike,
        timestamp: ArrayLike,
        epoch: ArrayLike,
        semantic_provenance_id: str,
        numeric_revision_id: str,
        state_schema_id: str,
        execution_signature_id: str,
        tendon_query_id: str,
        /,
    ):
        if not isinstance(actuator_states, tuple) or not all(
            isinstance(value, TendonActuatorState) for value in actuator_states
        ):
            raise TypeError(
                "actuator_states must be a tuple of TendonActuatorState values."
            )
        key = jnp.asarray(source_key, dtype=jnp.uint32)
        time = jnp.asarray(timestamp)
        step = jnp.asarray(epoch, dtype=jnp.int32)
        if key.shape != (2,) or time.shape != () or step.shape != ():
            raise ValueError(
                "A tendon observation state requires scalar time/epoch and one key."
            )
        self.actuator_states = actuator_states
        self.source_key = key
        self.timestamp = time
        self.epoch = step
        self.semantic_provenance_id = _identifier(
            semantic_provenance_id, "semantic_provenance_id"
        )
        self.numeric_revision_id = _identifier(numeric_revision_id, "numeric_revision_id")
        self.state_schema_id = _identifier(state_schema_id, "state_schema_id")
        self.execution_signature_id = _identifier(
            execution_signature_id, "execution_signature_id"
        )
        self.tendon_query_id = _identifier(tendon_query_id, "tendon_query_id")


class SoftSensorState(StrictModule):
    """Caller-owned bias and sample-hold payload; observation never mutates it."""

    bias: Array
    held_values: Array
    held_ideal_values: Array
    held_bias: Array
    held_noise: Array
    held_noise_key: Array
    sample_timestamp: Array
    sample_epoch: Array
    initialized: Array
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)
    query_plan_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    sensor_id: str = eqx.field(static=True)
    sensor_plan_id: str = eqx.field(static=True)


class SoftObservationPlan(StrictModule, NonTrainableState):
    """Unprepared composition of independent fixed-shape soft-robot queries."""

    reduced_state: SoftReducedStateQueryPlan | None
    frame: SoftFrameQueryPlan | None
    strain: SoftStrainQueryPlan | None
    tendon: SoftTendonQueryPlan | None
    energy_load: SoftEnergyLoadQueryPlan | None
    sensor: SoftSensorPlan | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        reduced_state: SoftReducedStateQueryPlan | None = None,
        frame: SoftFrameQueryPlan | None = None,
        strain: SoftStrainQueryPlan | None = None,
        tendon: SoftTendonQueryPlan | None = None,
        energy_load: SoftEnergyLoadQueryPlan | None = None,
        sensor: SoftSensorPlan | None = None,
    ):
        values = (
            (reduced_state, SoftReducedStateQueryPlan, "reduced_state"),
            (frame, SoftFrameQueryPlan, "frame"),
            (strain, SoftStrainQueryPlan, "strain"),
            (tendon, SoftTendonQueryPlan, "tendon"),
            (energy_load, SoftEnergyLoadQueryPlan, "energy_load"),
            (sensor, SoftSensorPlan, "sensor"),
        )
        for value, expected, name in values:
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"{name} has the wrong query-plan type.")
        if all(
            value is None for value in (reduced_state, frame, strain, tendon, energy_load)
        ):
            raise ValueError("A soft observation plan must contain at least one query.")
        self.reduced_state = reduced_state
        self.frame = frame
        self.strain = strain
        self.tendon = tendon
        self.energy_load = energy_load
        self.sensor = sensor
        self.plan_id = canonical_fingerprint(
            {
                "kind": "soft-robot-observation-plan",
                "reduced_state": None if reduced_state is None else reduced_state.plan_id,
                "frame": None if frame is None else frame.plan_id,
                "strain": None if strain is None else strain.plan_id,
                "tendon": None if tendon is None else tendon.plan_id,
                "energy_load": None if energy_load is None else energy_load.plan_id,
                "sensor": None if sensor is None else sensor.plan_id,
            }
        )

    def prepare(self, plant: PreparedReducedRodPlant, /) -> "PreparedSoftObservationPlan":
        return PreparedSoftObservationPlan(plant, self)


class _PreparedReducedStateQuery(StrictModule, NonTrainableState):
    plan: SoftReducedStateQueryPlan
    coordinate_count: int = eqx.field(static=True)
    query_id: str = eqx.field(static=True)

    def values(self, state: ReducedRodState, /) -> Array:
        blocks = []
        if self.plan.include_configuration:
            blocks.append(state.coefficients)
        if self.plan.include_velocity:
            blocks.append(state.coefficient_velocities)
        return jnp.concatenate(blocks)

    def specification(
        self, /
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        names: list[str] = []
        units: list[str] = []
        frames: list[str] = []
        if self.plan.include_configuration:
            names.extend(f"reduced.q[{index}]" for index in range(self.coordinate_count))
            units.extend((self.plan.coordinate_unit,) * self.coordinate_count)
            frames.extend(("reduced-strain-coordinate",) * self.coordinate_count)
        if self.plan.include_velocity:
            names.extend(
                f"reduced.qdot[{index}]" for index in range(self.coordinate_count)
            )
            units.extend((self.plan.velocity_unit,) * self.coordinate_count)
            frames.extend(("reduced-strain-coordinate-rate",) * self.coordinate_count)
        return tuple(names), tuple(units), tuple(frames)


class _PreparedFrameQuery(StrictModule, NonTrainableState):
    plan: SoftFrameQueryPlan
    reconstruction: PreparedRodReconstruction
    query_id: str = eqx.field(static=True)

    def _validate(self, evaluation: RodReconstructionEvaluation, /) -> None:
        if not isinstance(evaluation, RodReconstructionEvaluation):
            raise TypeError("frame_evaluation must be RodReconstructionEvaluation.")
        expected = (
            self.reconstruction.plan.queries.plan_id,
            self.reconstruction.route_id,
            self.reconstruction.reduced.prepared_id,
            self.reconstruction.reconstruction_id,
        )
        observed = (
            evaluation.query_plan_id,
            evaluation.route_id,
            evaluation.reduction_id,
            evaluation.reconstruction_id,
        )
        if observed != expected or evaluation.frame_convention != _FRAME_CONVENTION:
            raise ValueError(
                "Frame reconstruction evaluation belongs to a different query."
            )

    def evaluate(
        self,
        state: ReducedRodState,
        supplied: RodReconstructionEvaluation | None,
        /,
    ) -> tuple[Array, Array]:
        evaluation = self.reconstruction.evaluate(state) if supplied is None else supplied
        self._validate(evaluation)
        blocks = []
        if self.plan.include_pose:
            blocks.append(evaluation.poses.reshape((-1,)))
        for kind in self.plan.twists:
            if kind == "body":
                blocks.append(evaluation.body_twists.reshape((-1,)))
            elif kind == "world_origin":
                blocks.append(evaluation.world_origin_velocities.reshape((-1,)))
            else:
                blocks.append(evaluation.frame_velocities.reshape((-1,)))
        values = jnp.concatenate(blocks)
        values = eqx.error_if(
            values,
            ~evaluation.valid,
            "Soft frame observation rejected invalid reconstruction evidence.",
        )
        return values, evaluation.valid

    def specification(
        self, /
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        names: list[str] = []
        units: list[str] = []
        frames: list[str] = []
        count = self.reconstruction.plan.queries.query_count
        pose_components = ("qw", "qx", "qy", "qz", "x", "y", "z")
        if self.plan.include_pose:
            for index in range(count):
                names.extend(f"frame[{index}].pose.{name}" for name in pose_components)
                units.extend(("1",) * 4 + (self.plan.position_unit,) * 3)
                frames.extend(("world<-material",) * 4 + ("world",) * 3)
        twist_frames = {
            "body": "material-body@frame-origin",
            "world_origin": "world@world-origin",
            "frame_world": "world@frame-origin",
        }
        for kind in self.plan.twists:
            for index in range(count):
                names.extend(
                    f"frame[{index}].twist.{kind}.{name}"
                    for name in ("vx", "vy", "vz", "wx", "wy", "wz")
                )
                units.extend(
                    (self.plan.linear_velocity_unit,) * 3
                    + (self.plan.angular_velocity_unit,) * 3
                )
                frames.extend((twist_frames[kind],) * 6)
        return tuple(names), tuple(units), tuple(frames)


class _PreparedStrainQuery(StrictModule, NonTrainableState):
    plan: SoftStrainQueryPlan
    reconstruction: PreparedRodReconstruction
    query_id: str = eqx.field(static=True)

    def _validate(self, evaluation: RodReconstructionEvaluation, /) -> None:
        if not isinstance(evaluation, RodReconstructionEvaluation):
            raise TypeError("strain_evaluation must be RodReconstructionEvaluation.")
        expected = (
            self.reconstruction.plan.queries.plan_id,
            self.reconstruction.route_id,
            self.reconstruction.reduced.prepared_id,
            self.reconstruction.reconstruction_id,
        )
        observed = (
            evaluation.query_plan_id,
            evaluation.route_id,
            evaluation.reduction_id,
            evaluation.reconstruction_id,
        )
        if observed != expected or evaluation.frame_convention != _FRAME_CONVENTION:
            raise ValueError(
                "Strain reconstruction evaluation belongs to a different query."
            )

    def evaluate(
        self,
        state: ReducedRodState,
        supplied: RodReconstructionEvaluation | None,
        /,
    ) -> tuple[Array, Array]:
        evaluation = self.reconstruction.evaluate(state) if supplied is None else supplied
        self._validate(evaluation)
        blocks = []
        if self.plan.include_total:
            blocks.append(evaluation.strains.reshape((-1,)))
        if self.plan.include_reduced:
            blocks.append(evaluation.reduced_strains.reshape((-1,)))
        values = jnp.concatenate(blocks)
        values = eqx.error_if(
            values,
            ~evaluation.valid,
            "Soft strain observation rejected invalid reconstruction evidence.",
        )
        return values, evaluation.valid

    def specification(
        self, /
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        names: list[str] = []
        units: list[str] = []
        frames: list[str] = []
        count = self.reconstruction.plan.queries.query_count
        components = ("nu_x", "nu_y", "nu_z", "kappa_x", "kappa_y", "kappa_z")
        for label, selected in (
            ("total", self.plan.include_total),
            ("reduced", self.plan.include_reduced),
        ):
            if not selected:
                continue
            for index in range(count):
                names.extend(f"strain[{index}].{label}.{name}" for name in components)
                units.extend(
                    (self.plan.stretch_shear_unit,) * 3 + (self.plan.bend_twist_unit,) * 3
                )
                frames.extend(("material",) * 6)
        return tuple(names), tuple(units), tuple(frames)


class _PreparedTendonQuery(StrictModule, NonTrainableState):
    plan: SoftTendonQueryPlan
    reduction_id: str = eqx.field(static=True)
    query_id: str = eqx.field(static=True)

    def bind_state(
        self,
        runtime: PlantRuntimeState,
        actuator_states: tuple[TendonActuatorState, ...],
        /,
    ) -> SoftTendonObservationState:
        if not isinstance(runtime, PlantRuntimeState):
            raise TypeError("runtime must be PlantRuntimeState.")
        if not isinstance(actuator_states, tuple) or len(actuator_states) != len(
            self.plan.tendons
        ):
            raise ValueError("actuator_states must provide one state per tendon.")
        return SoftTendonObservationState(
            actuator_states,
            _runtime_key_data(runtime.key),
            runtime.time,
            runtime.step_index,
            *_runtime_ids(runtime),
            self.query_id,
        )

    def _states(
        self,
        runtime: PlantRuntimeState,
        bound: SoftTendonObservationState | None,
        values: Array,
        /,
    ) -> tuple[tuple[TendonActuatorState, ...] | None, Array]:
        if not self.plan.requires_actuator_state:
            if bound is not None:
                raise ValueError("This geometric tendon query accepts no actuator state.")
            return None, values
        if not isinstance(bound, SoftTendonObservationState):
            raise TypeError(
                "This tendon query requires a bound SoftTendonObservationState."
            )
        if bound.tendon_query_id != self.query_id:
            raise ValueError("Tendon actuator state belongs to a different query plan.")
        if (
            bound.semantic_provenance_id,
            bound.numeric_revision_id,
            bound.state_schema_id,
            bound.execution_signature_id,
        ) != _runtime_ids(runtime):
            raise ValueError(
                "Tendon actuator state belongs to a different plant identity."
            )
        if len(bound.actuator_states) != len(self.plan.tendons):
            raise ValueError("Tendon actuator-state count changed after preparation.")
        freshness = (
            (bound.timestamp == runtime.time)
            & (bound.epoch == runtime.step_index)
            & jnp.all(bound.source_key == _runtime_key_data(runtime.key))
        )
        checked = eqx.error_if(
            values,
            ~freshness,
            "Tendon actuator state is stale relative to the observed plant state.",
        )
        return bound.actuator_states, checked

    def evaluate(
        self,
        runtime: PlantRuntimeState,
        state: ReducedRodState,
        bound: SoftTendonObservationState | None,
        /,
    ) -> tuple[Array, Array]:
        states, checked = self._states(runtime, bound, state.values)
        state = ReducedRodState(
            checked[: state.coordinate_count], checked[state.coordinate_count :]
        )
        values: list[Array] = []
        finite = jnp.asarray(True)
        for index, tendon in enumerate(self.plan.tendons):
            length = tendon.route.length(state)
            if self.plan.include_length:
                values.append(length.reshape((1,)))
            if self.plan.include_length_rate:
                rate = tendon.route.length_rate(state)
                values.append(rate.reshape((1,)))
            if states is not None:
                actuator = states[index]
                if actuator.free_length.dtype != state.values.dtype:
                    raise TypeError(
                        "Tendon actuator state dtype must match the reduced rod."
                    )
                extension = jnp.maximum(length - actuator.free_length, 0.0)
                tension = (
                    jnp.asarray(tendon.plan.stiffness, dtype=length.dtype) * extension
                )
                if self.plan.include_tension:
                    values.append(tension.reshape((1,)))
                if self.plan.include_stored_energy:
                    values.append((0.5 * tension * extension).reshape((1,)))
            finite = finite & jnp.all(jnp.isfinite(values[-1]))
        output = jnp.concatenate(values)
        output = eqx.error_if(
            output, ~finite, "Soft tendon observation produced nonfinite mechanics."
        )
        return output, finite

    def specification(
        self, /
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        names: list[str] = []
        units: list[str] = []
        frames: list[str] = []
        for name in self.plan.tendon_names:
            if self.plan.include_length:
                names.append(f"tendon[{name}].length")
                units.append(self.plan.length_unit)
                frames.append("world-route")
            if self.plan.include_length_rate:
                names.append(f"tendon[{name}].length_rate")
                units.append(self.plan.length_rate_unit)
                frames.append("world-route")
            if self.plan.include_tension:
                names.append(f"tendon[{name}].tension")
                units.append(self.plan.tension_unit)
                frames.append("tendon-route")
            if self.plan.include_stored_energy:
                names.append(f"tendon[{name}].stored_energy")
                units.append(self.plan.energy_unit)
                frames.append("tendon-route")
        return tuple(names), tuple(units), tuple(frames)


class _PreparedEnergyLoadQuery(StrictModule, NonTrainableState):
    plan: SoftEnergyLoadQueryPlan
    plant_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    coordinate_count: int = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    channel_names: tuple[str, ...] = eqx.field(static=True)
    query_id: str = eqx.field(static=True)

    def _integration_result(
        self,
        runtime: PlantRuntimeState,
        step: PlantStepResult | None,
        values: Array,
        /,
    ) -> ReducedRodStepResult:
        if not self.plan.include_step_ledger:
            if step is not None:
                raise ValueError("This energy/load query does not consume step evidence.")
            raise RuntimeError("Step evidence requested for a mechanics-only query.")
        if not isinstance(step, PlantStepResult):
            raise TypeError("A step-ledger query requires PlantStepResult.")
        if not isinstance(step.evidence, ReducedRodPlantEvidence):
            raise TypeError("PlantStepResult does not contain reduced-rod step evidence.")
        if step.evidence.plant_id != self.plant_id:
            raise ValueError("Plant step evidence belongs to a different prepared plant.")
        integration = step.evidence.integration_result
        if not isinstance(integration, ReducedRodStepResult):
            raise TypeError("Reduced-rod plant evidence lost its integration result.")
        if integration.policy_id != self.policy_id:
            raise ValueError("Plant step evidence uses a different integration policy.")
        if (
            integration.evidence.candidate_evaluation.dynamics_id != self.dynamics_id
            or integration.evidence.ledger.source_ids != self.source_ids
            or integration.evidence.ledger.channel_names != self.channel_names
        ):
            raise ValueError("Energy/load ledger identities changed after preparation.")
        accepted = step.accepted_state
        if not isinstance(accepted, PlantRuntimeState) or _runtime_ids(
            accepted
        ) != _runtime_ids(runtime):
            raise ValueError("Step accepted state belongs to a different plant identity.")
        freshness = (
            step.successful
            & integration.successful
            & (accepted.time == runtime.time)
            & (accepted.step_index == runtime.step_index)
            & jnp.all(_runtime_key_data(accepted.key) == _runtime_key_data(runtime.key))
            & _tree_exact_equal(accepted.payload, runtime.payload)
            & _tree_exact_equal(
                integration.accepted_state.reduced_state,
                runtime.payload.reduced_state,
            )
        )
        checked_scale = eqx.error_if(
            integration.evidence.ledger.balance_scale,
            ~freshness,
            "Energy/load ledger is stale or was not accepted for this plant state.",
        )
        return eqx.tree_at(
            lambda result: result.evidence.ledger.balance_scale,
            integration,
            checked_scale,
        )

    def evaluate(
        self,
        runtime: PlantRuntimeState,
        reduced: ReducedRodEvaluation,
        step: PlantStepResult | None,
        /,
    ) -> tuple[Array, Array]:
        values: list[Array] = []
        valid = reduced.valid
        if self.plan.include_mechanics:
            values.extend(
                value.reshape((1,))
                for value in (
                    reduced.potential_energy,
                    reduced.kinetic_energy,
                    reduced.total_energy,
                )
            )
            values.append(reduced.generalized_internal_load.reshape((-1,)))
        if self.plan.include_step_ledger:
            probe = reduced.generalized_internal_load
            integration = self._integration_result(runtime, step, probe)
            ledger = integration.evidence.ledger
            values.extend(
                (
                    ledger.source_power_before.reshape((-1,)),
                    ledger.source_power_after.reshape((-1,)),
                    ledger.source_work.reshape((-1,)),
                    ledger.channel_power_before.reshape((-1,)),
                    ledger.channel_power_after.reshape((-1,)),
                    ledger.channel_work.reshape((-1,)),
                )
            )
            values.extend(
                value.reshape((1,))
                for value in (
                    ledger.total_power_before,
                    ledger.total_power_after,
                    ledger.external_work,
                    ledger.kinetic_energy_before,
                    ledger.kinetic_energy_after,
                    ledger.stored_energy_before,
                    ledger.stored_energy_after,
                    ledger.mechanical_energy_before,
                    ledger.mechanical_energy_after,
                    ledger.viscous_dissipation,
                    ledger.balance_residual,
                    ledger.balance_scale,
                )
            )
            valid = valid & integration.successful & ledger.valid
        output = jnp.concatenate(values)
        finite = jnp.all(jnp.isfinite(output))
        valid = valid & finite
        output = eqx.error_if(
            output,
            ~valid,
            "Soft energy/load observation rejected invalid mechanics evidence.",
        )
        return output, valid

    def specification(
        self, /
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        names: list[str] = []
        units: list[str] = []
        frames: list[str] = []
        if self.plan.include_mechanics:
            names.extend(("energy.potential", "energy.kinetic", "energy.total"))
            units.extend((self.plan.energy_unit,) * 3)
            frames.extend(("world",) * 3)
            names.extend(
                f"load.generalized_internal[{index}]"
                for index in range(self.coordinate_count)
            )
            units.extend((self.plan.reduced_effort_unit,) * self.coordinate_count)
            frames.extend(("reduced-cotangent",) * self.coordinate_count)
        if self.plan.include_step_ledger:
            for field in ("power_before", "power_after", "work"):
                for source in self.source_ids:
                    names.append(f"ledger.source[{source}].{field}")
                    units.append(
                        self.plan.energy_unit if field == "work" else self.plan.power_unit
                    )
                    frames.append("power-ledger")
            for field in ("power_before", "power_after", "work"):
                for channel in self.channel_names:
                    names.append(f"ledger.channel[{channel}].{field}")
                    units.append(
                        self.plan.energy_unit if field == "work" else self.plan.power_unit
                    )
                    frames.append("power-ledger")
            names.extend(
                (
                    "ledger.total_power_before",
                    "ledger.total_power_after",
                    "ledger.external_work",
                    "ledger.kinetic_energy_before",
                    "ledger.kinetic_energy_after",
                    "ledger.stored_energy_before",
                    "ledger.stored_energy_after",
                    "ledger.mechanical_energy_before",
                    "ledger.mechanical_energy_after",
                    "ledger.viscous_dissipation",
                    "ledger.balance_residual",
                    "ledger.balance_scale",
                )
            )
            units.extend((self.plan.power_unit,) * 2 + (self.plan.energy_unit,) * 10)
            frames.extend(("power-ledger",) * 2 + ("energy-ledger",) * 10)
        return tuple(names), tuple(units), tuple(frames)


class _PreparedSensorPlan(StrictModule, NonTrainableState):
    plan: SoftSensorPlan
    noise_standard_deviation: Array
    layout: SoftObservationLayout
    key_tag: int = eqx.field(static=True)
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)
    query_plan_id: str = eqx.field(static=True)
    sensor_plan_id: str = eqx.field(static=True)

    def initialize_state(self, bias: ArrayLike = 0.0, /) -> SoftSensorState:
        bias_ = _component_vector(
            bias,
            self.layout.size,
            np.dtype(self.noise_standard_deviation.dtype),
            "bias",
        )
        zeros = jnp.zeros_like(bias_)
        return SoftSensorState(
            bias_,
            zeros,
            zeros,
            zeros,
            zeros,
            jnp.zeros((2,), dtype=jnp.uint32),
            jnp.asarray(0.0, dtype=bias_.dtype),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(False),
            self.semantic_provenance_id,
            self.numeric_revision_id,
            self.state_schema_id,
            self.execution_signature_id,
            self.query_plan_id,
            self.layout.layout_id,
            self.plan.sensor_id,
            self.sensor_plan_id,
        )

    def _validate_state(self, state: SoftSensorState, /) -> None:
        if not isinstance(state, SoftSensorState):
            raise TypeError("sensor_state must be SoftSensorState.")
        static_observed = (
            state.semantic_provenance_id,
            state.numeric_revision_id,
            state.state_schema_id,
            state.execution_signature_id,
            state.query_plan_id,
            state.layout_id,
            state.sensor_id,
            state.sensor_plan_id,
        )
        static_expected = (
            self.semantic_provenance_id,
            self.numeric_revision_id,
            self.state_schema_id,
            self.execution_signature_id,
            self.query_plan_id,
            self.layout.layout_id,
            self.plan.sensor_id,
            self.sensor_plan_id,
        )
        if static_observed != static_expected:
            raise ValueError(
                "Soft sensor state belongs to another plant/query/sensor plan."
            )
        arrays = (
            state.bias,
            state.held_values,
            state.held_ideal_values,
            state.held_bias,
            state.held_noise,
        )
        if any(
            value.shape != (self.layout.size,)
            or value.dtype != self.noise_standard_deviation.dtype
            for value in arrays
        ):
            raise ValueError("Soft sensor state arrays do not match the prepared layout.")
        if state.held_noise_key.shape != (2,) or state.held_noise_key.dtype != jnp.uint32:
            raise ValueError("Soft sensor held_noise_key must be one uint32 key pair.")
        if (
            state.sample_timestamp.shape != ()
            or state.sample_epoch.shape != ()
            or state.initialized.shape != ()
        ):
            raise ValueError("Soft sensor sample metadata must be scalar.")

    def sample(
        self,
        mechanics: Array,
        runtime: PlantRuntimeState,
        source: SoftSensorState,
        /,
    ) -> tuple[
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        SoftSensorState,
    ]:
        self._validate_state(source)
        time = jnp.asarray(runtime.time, dtype=mechanics.dtype)
        epoch = jnp.asarray(runtime.step_index, dtype=jnp.int32)
        monotone = (~source.initialized) | (
            (time >= source.sample_timestamp) & (epoch >= source.sample_epoch)
        )
        mechanics = eqx.error_if(
            mechanics,
            ~monotone,
            "Soft sensor state was sampled from a future plant state.",
        )
        elapsed = time - source.sample_timestamp
        due = (~source.initialized) | (elapsed >= self.plan.sample_period)
        plant_key = jnp.asarray(runtime.key)
        sample_key = jax.random.fold_in(plant_key, self.key_tag)
        sample_key = jax.random.fold_in(sample_key, epoch.astype(jnp.uint32))
        proposed_noise = self.noise_standard_deviation * jax.random.normal(
            sample_key, mechanics.shape, dtype=mechanics.dtype
        )
        proposed_values = mechanics + source.bias + proposed_noise
        key_data = jax.random.key_data(sample_key)
        held_values = jnp.where(due, proposed_values, source.held_values)
        held_ideal = jnp.where(due, mechanics, source.held_ideal_values)
        held_bias = jnp.where(due, source.bias, source.held_bias)
        held_noise = jnp.where(due, proposed_noise, source.held_noise)
        held_key = jnp.where(due, key_data, source.held_noise_key)
        sample_time = jnp.where(due, time, source.sample_timestamp)
        sample_epoch = jnp.where(due, epoch, source.sample_epoch)
        initialized = source.initialized | due
        candidate = SoftSensorState(
            source.bias,
            held_values,
            held_ideal,
            held_bias,
            held_noise,
            held_key,
            sample_time,
            sample_epoch,
            initialized,
            self.semantic_provenance_id,
            self.numeric_revision_id,
            self.state_schema_id,
            self.execution_signature_id,
            self.query_plan_id,
            self.layout.layout_id,
            self.plan.sensor_id,
            self.sensor_plan_id,
        )
        fresh = (sample_time == time) & (sample_epoch == epoch)
        age = time - sample_time
        finite = (
            jnp.all(jnp.isfinite(held_values))
            & jnp.all(jnp.isfinite(held_ideal))
            & jnp.all(jnp.isfinite(held_bias))
            & jnp.all(jnp.isfinite(held_noise))
            & jnp.isfinite(sample_time)
            & jnp.isfinite(age)
        )
        return (
            held_values,
            held_ideal,
            held_bias,
            held_noise,
            held_key,
            sample_time,
            sample_epoch,
            age,
            fresh,
            candidate,
        )


class PreparedSoftObservationPlan(StrictModule, NonTrainableState):
    """Prepared pure observation composition for one exact fixed-base reduced plant."""

    plant: PreparedReducedRodPlant
    plan: SoftObservationPlan
    reduced_state: _PreparedReducedStateQuery | None
    frame: _PreparedFrameQuery | None
    strain: _PreparedStrainQuery | None
    tendon: _PreparedTendonQuery | None
    energy_load: _PreparedEnergyLoadQuery | None
    sensor: _PreparedSensorPlan | None
    layout: SoftObservationLayout
    query_ids: tuple[str, ...] = eqx.field(static=True)
    query_plan_id: str = eqx.field(static=True)
    observation_plan_id: str = eqx.field(static=True)

    def __init__(self, plant: PreparedReducedRodPlant, plan: SoftObservationPlan, /):
        if not isinstance(plant, PreparedReducedRodPlant):
            raise TypeError("plant must be PreparedReducedRodPlant.")
        if not isinstance(plan, SoftObservationPlan):
            raise TypeError("plan must be SoftObservationPlan.")
        reduction = plant.dynamics.reduction
        prepared_reduced = None
        prepared_frame = None
        prepared_strain = None
        prepared_tendon = None
        prepared_energy = None
        entries: list[
            tuple[
                str,
                str,
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
            ]
        ] = []
        query_ids: list[str] = []
        if plan.reduced_state is not None:
            query_id = canonical_fingerprint(
                {
                    "kind": "prepared-soft-reduced-state-query",
                    "plan": plan.reduced_state.plan_id,
                    "reduction": reduction.prepared_id,
                }
            )
            prepared_reduced = _PreparedReducedStateQuery(
                plan.reduced_state, reduction.plan.coordinate_count, query_id
            )
            names, units, frames = prepared_reduced.specification()
            entries.append(("reduced_state", query_id, names, units, frames))
            query_ids.append(query_id)
        if plan.frame is not None:
            reconstruction = prepare_rod_reconstruction(
                reduction, plan.frame.reconstruction
            )
            query_id = canonical_fingerprint(
                {
                    "kind": "prepared-soft-frame-query",
                    "plan": plan.frame.plan_id,
                    "reconstruction": reconstruction.reconstruction_id,
                }
            )
            prepared_frame = _PreparedFrameQuery(plan.frame, reconstruction, query_id)
            names, units, frames = prepared_frame.specification()
            entries.append(("frame", query_id, names, units, frames))
            query_ids.append(query_id)
        if plan.strain is not None:
            reconstruction = prepare_rod_reconstruction(
                reduction, plan.strain.reconstruction
            )
            query_id = canonical_fingerprint(
                {
                    "kind": "prepared-soft-strain-query",
                    "plan": plan.strain.plan_id,
                    "reconstruction": reconstruction.reconstruction_id,
                }
            )
            prepared_strain = _PreparedStrainQuery(plan.strain, reconstruction, query_id)
            names, units, frames = prepared_strain.specification()
            entries.append(("strain", query_id, names, units, frames))
            query_ids.append(query_id)
        if plan.tendon is not None:
            for tendon in plan.tendon.tendons:
                if (
                    tendon.route.reduction is None
                    or tendon.route.reduction.prepared_id != reduction.prepared_id
                ):
                    raise ValueError(
                        "Every tendon observation route must use this exact reduced rod."
                    )
            query_id = canonical_fingerprint(
                {
                    "kind": "prepared-soft-tendon-query",
                    "plan": plan.tendon.plan_id,
                    "reduction": reduction.prepared_id,
                }
            )
            prepared_tendon = _PreparedTendonQuery(
                plan.tendon, reduction.prepared_id, query_id
            )
            names, units, frames = prepared_tendon.specification()
            entries.append(("tendon", query_id, names, units, frames))
            query_ids.append(query_id)
        if plan.energy_load is not None:
            sources = ["elastic", "kelvin_voigt"]
            channels = ["elastic", "kelvin_voigt"]
            native = (
                ()
                if plant.dynamics.gravity_load is None
                else (plant.dynamics.gravity_load,)
            ) + (() if plant.native_loads is None else plant.native_loads.loads)
            for load in native:
                sources.append(load.source_id)
                channels.append(load.power_channel)
            source_ids = tuple(sources)
            channel_names = tuple(dict.fromkeys(channels))
            query_id = canonical_fingerprint(
                {
                    "kind": "prepared-soft-energy-load-query",
                    "plan": plan.energy_load.plan_id,
                    "plant": plant.plant_id,
                    "dynamics": plant.dynamics.dynamics_id,
                    "policy": plant.policy.policy_id,
                    "sources": source_ids,
                    "channels": channel_names,
                }
            )
            prepared_energy = _PreparedEnergyLoadQuery(
                plan.energy_load,
                plant.plant_id,
                plant.dynamics.dynamics_id,
                plant.policy.policy_id,
                reduction.plan.coordinate_count,
                source_ids,
                channel_names,
                query_id,
            )
            names, units, frames = prepared_energy.specification()
            entries.append(("energy_load", query_id, names, units, frames))
            query_ids.append(query_id)
        names: list[str] = []
        units: list[str] = []
        frames: list[str] = []
        component_queries: list[str] = []
        groups: list[tuple[str, int, int]] = []
        offset = 0
        for group, query_id, entry_names, entry_units, entry_frames in entries:
            stop = offset + len(entry_names)
            groups.append((group, offset, stop))
            names.extend(entry_names)
            units.extend(entry_units)
            frames.extend(entry_frames)
            component_queries.extend((query_id,) * len(entry_names))
            offset = stop
        layout = SoftObservationLayout(
            tuple(names),
            tuple(units),
            tuple(frames),
            tuple(component_queries),
            tuple(groups),
        )
        query_plan_id = canonical_fingerprint(
            {
                "kind": "prepared-soft-observation-query-composition",
                "queries": tuple(query_ids),
                "layout": layout.layout_id,
                "reduction": reduction.prepared_id,
            }
        )
        sensor = None
        if plan.sensor is not None:
            dtype = np.dtype(reduction.coefficient_space.dtype)
            noise = _component_vector(
                plan.sensor.noise_standard_deviation,
                layout.size,
                dtype,
                "noise_standard_deviation",
                nonnegative=True,
            )
            sensor_plan_id = canonical_fingerprint(
                {
                    "kind": "prepared-explicit-soft-robot-sensor-plan",
                    "plan": plan.sensor.plan_id,
                    "layout": layout.layout_id,
                    "query_plan": query_plan_id,
                    "plant": plant.plant_id,
                    "noise": array_tree_fingerprint(np.asarray(noise)),
                }
            )
            sensor = _PreparedSensorPlan(
                plan.sensor,
                noise,
                layout,
                int(sensor_plan_id[:8], 16),
                plant.semantic_provenance.semantic_id,
                plant.numeric_revision.revision_id,
                plant.state_schema.schema_id,
                plant.execution_signature.signature_id,
                query_plan_id,
                sensor_plan_id,
            )
        observation_plan_id = canonical_fingerprint(
            {
                "kind": "prepared-soft-robot-observation-plan",
                "plan": plan.plan_id,
                "plant": plant.plant_id,
                "query_plan": query_plan_id,
                "layout": layout.layout_id,
                "sensor": None if sensor is None else sensor.sensor_plan_id,
            }
        )
        self.plant = plant
        self.plan = plan
        self.reduced_state = prepared_reduced
        self.frame = prepared_frame
        self.strain = prepared_strain
        self.tendon = prepared_tendon
        self.energy_load = prepared_energy
        self.sensor = sensor
        self.layout = layout
        self.query_ids = tuple(query_ids)
        self.query_plan_id = query_plan_id
        self.observation_plan_id = observation_plan_id

    def initialize_sensor_state(self, bias: ArrayLike = 0.0, /) -> SoftSensorState:
        if self.sensor is None:
            raise TypeError("This observation plan has no stateful/noisy sensor plan.")
        return self.sensor.initialize_state(bias)

    def bind_tendon_state(
        self,
        runtime: PlantRuntimeState,
        actuator_states: tuple[TendonActuatorState, ...],
        /,
    ) -> SoftTendonObservationState:
        self._payload(runtime)
        if self.tendon is None:
            raise TypeError("This observation plan has no tendon query.")
        return self.tendon.bind_state(runtime, actuator_states)

    def _payload(self, state: PlantRuntimeState, /) -> ReducedRodPlantState:
        if not isinstance(state, PlantRuntimeState):
            raise TypeError("state must be PlantRuntimeState.")
        expected = (
            self.plant.semantic_provenance.semantic_id,
            self.plant.numeric_revision.revision_id,
            self.plant.state_schema.schema_id,
            self.plant.execution_signature.signature_id,
        )
        if _runtime_ids(state) != expected:
            raise ValueError("Plant runtime state belongs to a different prepared plant.")
        if self.plant.state_schema.validate(state.payload) != ():
            raise ValueError("Soft observations currently require one scalar plant case.")
        if not isinstance(state.payload, ReducedRodPlantState):
            raise TypeError("Plant payload must be ReducedRodPlantState.")
        if state.time.shape != () or state.step_index.shape != ():
            raise ValueError("Soft observation time and epoch must be scalar.")
        if np.dtype(state.time.dtype).kind not in "biufc":
            raise TypeError("Soft observation timestamp must have numeric dtype.")
        _runtime_key_data(state.key)
        checked = eqx.error_if(
            state.payload.reduced_state.values,
            (~jnp.isfinite(state.time)) | (state.step_index < 0),
            "Soft observation plant timestamp/epoch is invalid.",
        )
        return eqx.tree_at(
            lambda value: value.reduced_state.values,
            state.payload,
            checked,
        )

    def observe(
        self,
        state: PlantRuntimeState,
        /,
        *,
        tendon_state: SoftTendonObservationState | None = None,
        plant_step: PlantStepResult | None = None,
        frame_evaluation: RodReconstructionEvaluation | None = None,
        strain_evaluation: RodReconstructionEvaluation | None = None,
        sensor_state: SoftSensorState | None = None,
    ) -> tuple[SoftRobotObservation, SoftSensorState | None]:
        """Evaluate current mechanics and propose, but never commit, sensor state."""
        payload = self._payload(state)
        reduced_state = payload.reduced_state
        blocks: list[Array] = []
        mechanics_valid = jnp.asarray(True)
        if self.reduced_state is not None:
            blocks.append(self.reduced_state.values(reduced_state))
        if self.frame is not None:
            values, valid = self.frame.evaluate(reduced_state, frame_evaluation)
            blocks.append(values)
            mechanics_valid = mechanics_valid & valid
        elif frame_evaluation is not None:
            raise ValueError("frame_evaluation was supplied without a frame query.")
        if self.strain is not None:
            values, valid = self.strain.evaluate(reduced_state, strain_evaluation)
            blocks.append(values)
            mechanics_valid = mechanics_valid & valid
        elif strain_evaluation is not None:
            raise ValueError("strain_evaluation was supplied without a strain query.")
        if self.tendon is not None:
            values, valid = self.tendon.evaluate(state, reduced_state, tendon_state)
            blocks.append(values)
            mechanics_valid = mechanics_valid & valid
        elif tendon_state is not None:
            raise ValueError("tendon_state was supplied without a tendon query.")
        if self.energy_load is not None:
            reduced_evaluation = self.plant.dynamics.reduction.evaluate(reduced_state)
            values, valid = self.energy_load.evaluate(
                state, reduced_evaluation, plant_step
            )
            blocks.append(values)
            mechanics_valid = mechanics_valid & valid
        elif plant_step is not None:
            raise ValueError("plant_step was supplied without an energy/load query.")
        mechanics = jnp.concatenate(blocks)
        if mechanics.shape != (self.layout.size,):
            raise ValueError("Prepared query output no longer matches its fixed layout.")
        mechanics_finite = jnp.all(jnp.isfinite(mechanics))
        mechanics_valid = mechanics_valid & mechanics_finite
        mechanics = eqx.error_if(
            mechanics,
            ~mechanics_valid,
            "Soft robot observation rejected invalid mechanics evidence.",
        )
        source_key = _runtime_key_data(state.key)
        if self.sensor is None:
            if sensor_state is not None:
                raise ValueError("sensor_state was supplied to a direct mechanics query.")
            values = mechanics
            ideal = mechanics
            bias = jnp.zeros_like(mechanics)
            noise = jnp.zeros_like(mechanics)
            noise_key = source_key
            sample_time = state.time
            sample_epoch = state.step_index
            age = jnp.zeros_like(state.time)
            fresh = jnp.asarray(True)
            held = jnp.asarray(False)
            sensor_id = _DIRECT_SENSOR_ID
            sensor_plan_id = _DIRECT_SENSOR_PLAN_ID
            candidate_sensor_state = None
        else:
            if not isinstance(sensor_state, SoftSensorState):
                raise TypeError("This prepared sensor requires explicit SoftSensorState.")
            (
                values,
                ideal,
                bias,
                noise,
                noise_key,
                sample_time,
                sample_epoch,
                age,
                fresh,
                candidate_sensor_state,
            ) = self.sensor.sample(mechanics, state, sensor_state)
            held = ~fresh
            sensor_id = self.sensor.plan.sensor_id
            sensor_plan_id = self.sensor.sensor_plan_id
        finite = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(ideal))
            & jnp.all(jnp.isfinite(bias))
            & jnp.all(jnp.isfinite(noise))
            & jnp.isfinite(age)
        )
        valid = finite & mechanics_valid
        observation = SoftRobotObservation(
            values,
            ideal,
            bias,
            noise,
            source_key,
            noise_key,
            state.time,
            state.step_index,
            sample_time,
            sample_epoch,
            age,
            fresh,
            held,
            finite,
            mechanics_valid,
            valid,
            self.layout,
            state.semantic_provenance_id,
            state.numeric_revision_id,
            state.state_schema_id,
            state.execution_signature_id,
            self.query_plan_id,
            self.query_ids,
            sensor_id,
            sensor_plan_id,
            self.observation_plan_id,
        )
        return observation, candidate_sensor_state


def prepare_soft_observation_plan(
    plant: PreparedReducedRodPlant,
    plan: SoftObservationPlan,
    /,
) -> PreparedSoftObservationPlan:
    """Bind one exact observation ABI to one exact prepared reduced-rod plant."""
    return PreparedSoftObservationPlan(plant, plan)


__all__ = [
    "prepare_soft_observation_plan",
    "PreparedSoftObservationPlan",
    "SoftEnergyLoadQueryPlan",
    "SoftFrameQueryPlan",
    "SoftObservationLayout",
    "SoftObservationPlan",
    "SoftReducedStateQueryPlan",
    "SoftRobotObservation",
    "SoftSensorPlan",
    "SoftSensorState",
    "SoftStrainQueryPlan",
    "SoftTendonObservationState",
    "SoftTendonQueryPlan",
    "SoftTwistKind",
]
