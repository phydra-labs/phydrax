#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...integration._deformed_measure import DeformedMeasureState


MechanicalLoadSupport: TypeAlias = Literal["body", "boundary", "discrete"]
MechanicalLoadFrame: TypeAlias = Literal["reference", "current"]
MechanicalLoadConservativity: TypeAlias = Literal["potential", "virtual_work"]
ReferenceLoadField: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
CurrentLoadField: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
FollowerLoadField: TypeAlias = Callable[
    [Array, Array, DeformedMeasureState, "MechanicalLoadState", Any], ArrayLike
]


def _optional_identifier(value: str | None, name: str, /) -> str | None:
    if value is None:
        return None
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty when supplied.")
    return identifier


def _required_identifier(value: str | None, name: str, /) -> str:
    identifier = _optional_identifier(value, name)
    if identifier is None:
        raise ValueError(f"{name} is required.")
    return identifier


def _real_inexact_array(name: str, value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact) or jnp.iscomplexobj(array):
        raise TypeError(f"{name} must be a real inexact array.")
    return array


def _constant_or_field(
    name: str,
    value: ArrayLike | Callable,
    load_id: str | None,
    /,
) -> tuple[Array | Callable, str]:
    if callable(value):
        return value, _required_identifier(load_id, "load_id")
    array = _real_inexact_array(name, value)
    if array.ndim == 0:
        raise ValueError(f"{name} must carry a vector component axis.")
    if not np.all(np.isfinite(np.asarray(array))):
        raise ValueError(f"{name} must be finite.")
    generated = canonical_fingerprint(
        {"kind": name, "value": array_tree_fingerprint(array)}
    )
    identifier = generated if load_id is None else str(load_id)
    if not identifier:
        raise ValueError("load_id must be non-empty.")
    return array, identifier


def _scalar_parameter(state: MechanicalLoadState, dtype, /) -> Array:
    parameter = jnp.asarray(state.parameter, dtype=dtype)
    if parameter.shape != () or jnp.iscomplexobj(parameter):
        raise ValueError("This mechanical load requires one real scalar state parameter.")
    return parameter


def _vector_field(
    field: Array | Callable,
    coordinates: Array,
    state: MechanicalLoadState,
    args: Any,
    /,
) -> Array:
    value = (
        field(coordinates, state.time, args)
        if callable(field)
        else jnp.asarray(field, dtype=coordinates.dtype)
    )
    array = jnp.asarray(value, dtype=coordinates.dtype)
    try:
        return jnp.broadcast_to(array, coordinates.shape)
    except ValueError as error:
        raise ValueError(
            "Mechanical load fields must broadcast to the evaluated coordinate shape."
        ) from error


def _coordinates(
    reference_coordinates: ArrayLike,
    current_coordinates: ArrayLike,
    measure: DeformedMeasureState,
    /,
) -> tuple[Array, Array]:
    if not isinstance(measure, DeformedMeasureState):
        raise TypeError("measure must be a DeformedMeasureState.")
    reference = _real_inexact_array("reference_coordinates", reference_coordinates)
    current = _real_inexact_array("current_coordinates", current_coordinates)
    if reference.shape != current.shape or reference.ndim == 0:
        raise ValueError(
            "Reference and current coordinates must have one identical shape."
        )
    if reference.shape[-1] not in (2, 3):
        raise ValueError(
            "Mechanical loads require two- or three-dimensional coordinates."
        )
    if measure.reference_measure.shape != reference.shape[:-1]:
        raise ValueError(
            "Deformed measure and coordinate point layouts must match exactly."
        )
    return reference, current


def _normal(
    supplied: ArrayLike | None,
    measure: DeformedMeasureState,
    frame: MechanicalLoadFrame,
    coordinates: Array,
    /,
) -> Array:
    normal = measure.normal(frame) if supplied is None else jnp.asarray(supplied)
    normal = jnp.asarray(normal, dtype=coordinates.dtype)
    if normal.shape != coordinates.shape:
        raise ValueError("Surface normals must match the evaluated coordinate layout.")
    norm = jnp.sqrt(jnp.sum(normal * normal, axis=-1))
    return normal / jnp.maximum(norm[..., None], jnp.finfo(normal.dtype).tiny)


class MechanicalLoadSemantics(StrictModule, NonTrainableState):
    """Explicit support, frame, measure, and conservative-routing contract."""

    support: MechanicalLoadSupport = eqx.field(static=True)
    configuration: MechanicalLoadFrame = eqx.field(static=True)
    measure_frame: MechanicalLoadFrame = eqx.field(static=True)
    load_frame: MechanicalLoadFrame = eqx.field(static=True)
    conservativity: MechanicalLoadConservativity = eqx.field(static=True)
    potential_certified: bool = eqx.field(static=True)
    closure_id: str | None = eqx.field(static=True)
    orientation_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        support: MechanicalLoadSupport,
        configuration: MechanicalLoadFrame,
        measure_frame: MechanicalLoadFrame,
        load_frame: MechanicalLoadFrame,
        conservativity: MechanicalLoadConservativity,
        /,
        *,
        potential_certified: bool,
        closure_id: str | None = None,
        orientation_id: str | None = None,
    ):
        if support not in ("body", "boundary", "discrete"):
            raise ValueError("Unknown mechanical load support.")
        if configuration not in ("reference", "current"):
            raise ValueError("Unknown mechanical load configuration.")
        if measure_frame not in ("reference", "current"):
            raise ValueError("Unknown mechanical load measure frame.")
        if load_frame not in ("reference", "current"):
            raise ValueError("Unknown mechanical load vector frame.")
        if conservativity not in ("potential", "virtual_work"):
            raise ValueError("Unknown mechanical load conservative routing.")
        certified = bool(potential_certified)
        if conservativity == "potential" and not certified:
            raise ValueError("Potential routing requires a certified load potential.")
        if conservativity == "virtual_work" and certified:
            raise ValueError("Virtual-work routing cannot claim a certified potential.")
        closure = _optional_identifier(closure_id, "closure_id")
        orientation = _optional_identifier(orientation_id, "orientation_id")
        if (closure is None) != (orientation is None):
            raise ValueError(
                "Closure and orientation evidence must be supplied together."
            )
        if (
            certified
            and support == "boundary"
            and configuration == "current"
            and (closure is None or orientation is None)
        ):
            raise ValueError(
                "Certified current-boundary potentials require closure and orientation IDs."
            )
        if support != "boundary" and (closure is not None or orientation is not None):
            raise ValueError(
                "Only boundary loads may carry closure/orientation evidence."
            )
        self.support = support
        self.configuration = configuration
        self.measure_frame = measure_frame
        self.load_frame = load_frame
        self.conservativity = conservativity
        self.potential_certified = certified
        self.closure_id = closure
        self.orientation_id = orientation


class MechanicalLoadState(StrictModule, NonTrainableState):
    """Immutable committed load/controller input at one evaluation time."""

    time: Array
    parameter: Any
    state_id: str = eqx.field(static=True)
    pressure_history_id: str | None = eqx.field(static=True)
    volume_history_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        parameter: Any = 1.0,
        /,
        *,
        time: ArrayLike = 0.0,
        state_id: str | None = None,
        pressure_history_id: str | None = None,
        volume_history_id: str | None = None,
    ):
        time_ = _real_inexact_array("Mechanical load time", time)
        if time_.shape != () or not bool(jnp.isfinite(time_)):
            raise ValueError("Mechanical load time must be one finite scalar.")
        parameter_ = jnp.asarray(parameter) if np.isscalar(parameter) else parameter
        pressure_history = _optional_identifier(
            pressure_history_id, "pressure_history_id"
        )
        volume_history = _optional_identifier(volume_history_id, "volume_history_id")
        if state_id is None:
            identifier = canonical_fingerprint(
                {
                    "kind": "mechanical-load-state",
                    "time": array_tree_fingerprint(time_),
                    "parameter": array_tree_fingerprint(parameter_),
                    "pressure_history_id": pressure_history,
                    "volume_history_id": volume_history,
                }
            )
        else:
            identifier = str(state_id)
        if not identifier:
            raise ValueError("state_id must be non-empty.")
        self.time = time_
        self.parameter = parameter_
        self.state_id = identifier
        self.pressure_history_id = pressure_history
        self.volume_history_id = volume_history


class MechanicalLoadEvaluation(StrictModule):
    """Evaluated load density with unreduced components and routing evidence."""

    total_force_density: Array
    component_force_densities: tuple[Array, ...]
    component_ids: tuple[str, ...] = eqx.field(static=True)
    potential_density: Array | None
    semantics: MechanicalLoadSemantics
    valid: Array


class AbstractMechanicalLoad(StrictModule, NonTrainableState):
    """Discretization-independent mechanical load law."""

    @property
    @abc.abstractmethod
    def load_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def semantics(self) -> MechanicalLoadSemantics:
        raise NotImplementedError

    @abc.abstractmethod
    def _evaluate(
        self,
        reference_coordinates: Array,
        current_coordinates: Array,
        measure: DeformedMeasureState,
        reference_normal: Array | None,
        current_normal: Array | None,
        state: MechanicalLoadState,
        args: Any,
        /,
    ) -> MechanicalLoadEvaluation:
        raise NotImplementedError

    def evaluate(
        self,
        reference_coordinates: ArrayLike,
        current_coordinates: ArrayLike,
        measure: DeformedMeasureState,
        state: MechanicalLoadState,
        args: Any = None,
        /,
        *,
        reference_normal: ArrayLike | None = None,
        current_normal: ArrayLike | None = None,
    ) -> MechanicalLoadEvaluation:
        if not isinstance(state, MechanicalLoadState):
            raise TypeError("state must be a MechanicalLoadState.")
        reference, current = _coordinates(
            reference_coordinates, current_coordinates, measure
        )
        if self.semantics.support == "boundary" and measure.kind != "surface":
            raise ValueError("Boundary loads require a deformed surface measure.")
        if self.semantics.support == "body" and measure.kind != "volume":
            raise ValueError("Body loads require a deformed volume measure.")
        reference_normal_ = (
            None
            if self.semantics.support != "boundary"
            else _normal(reference_normal, measure, "reference", reference)
        )
        current_normal_ = (
            None
            if self.semantics.support != "boundary"
            else _normal(current_normal, measure, "current", current)
        )
        return self._evaluate(
            reference,
            current,
            measure,
            reference_normal_,
            current_normal_,
            state,
            args,
        )


def _single_evaluation(
    force_density: Array,
    potential_density: Array | None,
    load: AbstractMechanicalLoad,
    measure: DeformedMeasureState,
    /,
    *,
    additional_validity: ArrayLike = True,
) -> MechanicalLoadEvaluation:
    force = jnp.asarray(force_density)
    expected_potential_shape = force.shape[:-1]
    if potential_density is not None:
        potential = jnp.asarray(potential_density, dtype=force.dtype)
        if potential.shape != expected_potential_shape:
            raise ValueError(
                "Mechanical potential density must have one scalar per point."
            )
    else:
        potential = None
    finite = jnp.all(jnp.isfinite(force))
    if potential is not None:
        finite = finite & jnp.all(jnp.isfinite(potential))
    valid = measure.valid & finite & jnp.asarray(additional_validity, dtype=bool)
    return MechanicalLoadEvaluation(
        total_force_density=force,
        component_force_densities=(force,),
        component_ids=(load.load_id,),
        potential_density=potential,
        semantics=load.semantics,
        valid=valid,
    )


class ReferenceDeadTraction(AbstractMechanicalLoad):
    """Dead spatial traction per reference boundary measure."""

    traction: Array | ReferenceLoadField
    _load_id: str = eqx.field(static=True)
    _semantics: MechanicalLoadSemantics = eqx.field(static=True)

    def __init__(
        self,
        traction: ArrayLike | ReferenceLoadField,
        /,
        *,
        load_id: str | None = None,
    ):
        field, identifier = _constant_or_field(
            "reference-dead-traction", traction, load_id
        )
        self.traction = field
        self._load_id = identifier
        self._semantics = MechanicalLoadSemantics(
            "boundary",
            "reference",
            "reference",
            "current",
            "potential",
            potential_certified=True,
        )

    @property
    def load_id(self) -> str:
        return self._load_id

    @property
    def semantics(self) -> MechanicalLoadSemantics:
        return self._semantics

    def _evaluate(
        self,
        reference,
        current,
        measure,
        reference_normal,
        current_normal,
        state,
        args,
        /,
    ) -> MechanicalLoadEvaluation:
        del reference_normal, current_normal
        traction = _scalar_parameter(state, current.dtype) * _vector_field(
            self.traction, reference, state, args
        )
        potential = -oe.contract("...i,...i->...", traction, current - reference)
        return _single_evaluation(traction, potential, self, measure)


class ReferenceDeadBodyForce(AbstractMechanicalLoad):
    """Dead spatial body force per reference volume measure."""

    body_force: Array | ReferenceLoadField
    _load_id: str = eqx.field(static=True)
    _semantics: MechanicalLoadSemantics = eqx.field(static=True)

    def __init__(
        self,
        body_force: ArrayLike | ReferenceLoadField,
        /,
        *,
        load_id: str | None = None,
    ):
        field, identifier = _constant_or_field(
            "reference-dead-body-force", body_force, load_id
        )
        self.body_force = field
        self._load_id = identifier
        self._semantics = MechanicalLoadSemantics(
            "body",
            "reference",
            "reference",
            "current",
            "potential",
            potential_certified=True,
        )

    @property
    def load_id(self) -> str:
        return self._load_id

    @property
    def semantics(self) -> MechanicalLoadSemantics:
        return self._semantics

    def _evaluate(
        self,
        reference,
        current,
        measure,
        reference_normal,
        current_normal,
        state,
        args,
        /,
    ) -> MechanicalLoadEvaluation:
        del reference_normal, current_normal
        force = _scalar_parameter(state, current.dtype) * _vector_field(
            self.body_force, reference, state, args
        )
        potential = -oe.contract("...i,...i->...", force, current - reference)
        return _single_evaluation(force, potential, self, measure)


class CurrentBodyForce(AbstractMechanicalLoad):
    """Spatial body force per current volume, routed through virtual work."""

    body_force: Array | CurrentLoadField
    _load_id: str = eqx.field(static=True)
    _semantics: MechanicalLoadSemantics = eqx.field(static=True)

    def __init__(
        self,
        body_force: ArrayLike | CurrentLoadField,
        /,
        *,
        load_id: str | None = None,
    ):
        field, identifier = _constant_or_field("current-body-force", body_force, load_id)
        self.body_force = field
        self._load_id = identifier
        self._semantics = MechanicalLoadSemantics(
            "body",
            "current",
            "current",
            "current",
            "virtual_work",
            potential_certified=False,
        )

    @property
    def load_id(self) -> str:
        return self._load_id

    @property
    def semantics(self) -> MechanicalLoadSemantics:
        return self._semantics

    def _evaluate(
        self,
        reference,
        current,
        measure,
        reference_normal,
        current_normal,
        state,
        args,
        /,
    ) -> MechanicalLoadEvaluation:
        del reference, reference_normal, current_normal
        force = _scalar_parameter(state, current.dtype) * _vector_field(
            self.body_force, current, state, args
        )
        return _single_evaluation(force, None, self, measure)


class CurrentSurfaceTraction(AbstractMechanicalLoad):
    """Spatial traction per current boundary measure, routed through virtual work."""

    traction: Array | CurrentLoadField
    _load_id: str = eqx.field(static=True)
    _semantics: MechanicalLoadSemantics = eqx.field(static=True)

    def __init__(
        self,
        traction: ArrayLike | CurrentLoadField,
        /,
        *,
        load_id: str | None = None,
    ):
        field, identifier = _constant_or_field(
            "current-surface-traction", traction, load_id
        )
        self.traction = field
        self._load_id = identifier
        self._semantics = MechanicalLoadSemantics(
            "boundary",
            "current",
            "current",
            "current",
            "virtual_work",
            potential_certified=False,
        )

    @property
    def load_id(self) -> str:
        return self._load_id

    @property
    def semantics(self) -> MechanicalLoadSemantics:
        return self._semantics

    def _evaluate(
        self,
        reference,
        current,
        measure,
        reference_normal,
        current_normal,
        state,
        args,
        /,
    ) -> MechanicalLoadEvaluation:
        del reference, reference_normal, current_normal
        traction = _scalar_parameter(state, current.dtype) * _vector_field(
            self.traction, current, state, args
        )
        return _single_evaluation(traction, None, self, measure)


class ClosedSurfacePressure(AbstractMechanicalLoad):
    """Constant outward pressure on one certified closed oriented current surface."""

    pressure: Array
    _load_id: str = eqx.field(static=True)
    _semantics: MechanicalLoadSemantics = eqx.field(static=True)

    def __init__(
        self,
        pressure: ArrayLike,
        /,
        *,
        closure_id: str,
        orientation_id: str,
        load_id: str | None = None,
    ):
        value = _real_inexact_array("pressure", pressure)
        if value.shape != () or not bool(jnp.isfinite(value)):
            raise ValueError("Closed-surface pressure must be one finite scalar.")
        generated = canonical_fingerprint(
            {"kind": "closed-surface-pressure", "pressure": array_tree_fingerprint(value)}
        )
        identifier = generated if load_id is None else str(load_id)
        if not identifier:
            raise ValueError("load_id must be non-empty.")
        self.pressure = value
        self._load_id = identifier
        self._semantics = MechanicalLoadSemantics(
            "boundary",
            "current",
            "current",
            "current",
            "potential",
            potential_certified=True,
            closure_id=closure_id,
            orientation_id=orientation_id,
        )

    @property
    def load_id(self) -> str:
        return self._load_id

    @property
    def semantics(self) -> MechanicalLoadSemantics:
        return self._semantics

    def pressure_at_state(
        self,
        state: MechanicalLoadState,
        dtype: Any = None,
        /,
    ) -> Array:
        """Resolve the committed scalar pressure without evaluating geometry."""
        pressure = self.pressure if dtype is None else self.pressure.astype(dtype)
        return pressure * _scalar_parameter(state, pressure.dtype)

    def _evaluate(
        self,
        reference,
        current,
        measure,
        reference_normal,
        current_normal,
        state,
        args,
        /,
    ) -> MechanicalLoadEvaluation:
        del reference, reference_normal, args
        assert current_normal is not None
        pressure = self.pressure_at_state(state, current.dtype)
        volume_density = (
            oe.contract("...i,...i->...", current, current_normal) / current.shape[-1]
        )
        volume = jnp.sum(volume_density * measure.current_measure)
        force = pressure * current_normal
        potential = -pressure * volume_density
        valid = jnp.isfinite(volume) & (volume > 0.0) & jnp.isfinite(pressure)
        return _single_evaluation(
            force, potential, self, measure, additional_validity=valid
        )


class PneumaticPressure(AbstractMechanicalLoad):
    """Closed-surface pressure satisfying p V^k = p_ref V_ref^k."""

    reference_pressure: Array
    reference_volume: float = eqx.field(static=True)
    exponent: float = eqx.field(static=True)
    _load_id: str = eqx.field(static=True)
    _semantics: MechanicalLoadSemantics = eqx.field(static=True)

    def __init__(
        self,
        reference_pressure: ArrayLike,
        reference_volume: float,
        /,
        *,
        exponent: float = 1.0,
        closure_id: str,
        orientation_id: str,
        load_id: str | None = None,
    ):
        pressure = _real_inexact_array("reference_pressure", reference_pressure)
        volume = float(reference_volume)
        exponent_ = float(exponent)
        if pressure.shape != () or not bool(jnp.isfinite(pressure)) or pressure <= 0.0:
            raise ValueError("Reference pneumatic pressure must be one positive scalar.")
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError("reference_volume must be finite and positive.")
        if not np.isfinite(exponent_) or exponent_ < 0.0:
            raise ValueError("Pneumatic exponent must be finite and nonnegative.")
        generated = canonical_fingerprint(
            {
                "kind": "pneumatic-pressure",
                "reference_pressure": array_tree_fingerprint(pressure),
                "reference_volume": volume.hex(),
                "exponent": exponent_.hex(),
            }
        )
        identifier = generated if load_id is None else str(load_id)
        if not identifier:
            raise ValueError("load_id must be non-empty.")
        self.reference_pressure = pressure
        self.reference_volume = volume
        self.exponent = exponent_
        self._load_id = identifier
        self._semantics = MechanicalLoadSemantics(
            "boundary",
            "current",
            "current",
            "current",
            "potential",
            potential_certified=True,
            closure_id=closure_id,
            orientation_id=orientation_id,
        )

    @property
    def load_id(self) -> str:
        return self._load_id

    @property
    def semantics(self) -> MechanicalLoadSemantics:
        return self._semantics

    def pressure_at_volume(
        self,
        volume: ArrayLike,
        state: MechanicalLoadState,
        /,
    ) -> Array:
        """Evaluate the committed pneumatic law at one positive current volume."""
        current_volume = jnp.asarray(volume)
        reference_volume = jnp.asarray(self.reference_volume, dtype=current_volume.dtype)
        reference_pressure = self.reference_pressure.astype(
            current_volume.dtype
        ) * _scalar_parameter(state, current_volume.dtype)
        return reference_pressure * (current_volume / reference_volume) ** (
            -self.exponent
        )

    def potential_at_volume(
        self,
        volume: ArrayLike,
        state: MechanicalLoadState,
        /,
    ) -> Array:
        """Return the pressure potential relative to the reference volume."""
        current_volume = jnp.asarray(volume)
        reference_volume = jnp.asarray(self.reference_volume, dtype=current_volume.dtype)
        reference_pressure = self.reference_pressure.astype(
            current_volume.dtype
        ) * _scalar_parameter(state, current_volume.dtype)
        ratio = current_volume / reference_volume
        if self.exponent == 1.0:
            return -reference_pressure * reference_volume * jnp.log(ratio)
        return (
            reference_pressure
            * reference_volume
            * (ratio ** (1.0 - self.exponent) - 1.0)
            / (self.exponent - 1.0)
        )

    def _evaluate(
        self,
        reference,
        current,
        measure,
        reference_normal,
        current_normal,
        state,
        args,
        /,
    ) -> MechanicalLoadEvaluation:
        del reference, reference_normal, args
        assert current_normal is not None
        volume_density = (
            oe.contract("...i,...i->...", current, current_normal) / current.shape[-1]
        )
        volume = jnp.sum(volume_density * measure.current_measure)
        pressure = self.pressure_at_volume(volume, state)
        potential = self.potential_at_volume(volume, state)
        potential_density = potential * volume_density / volume
        force = pressure * current_normal
        valid = (
            jnp.isfinite(volume)
            & (volume > 0.0)
            & jnp.isfinite(pressure)
            & jnp.isfinite(potential)
        )
        return _single_evaluation(
            force,
            potential_density,
            self,
            measure,
            additional_validity=valid,
        )


class GeneralFollowerLoad(AbstractMechanicalLoad):
    """General state-dependent load density with explicit virtual-work routing."""

    law: FollowerLoadField
    _load_id: str = eqx.field(static=True)
    _semantics: MechanicalLoadSemantics = eqx.field(static=True)

    def __init__(
        self,
        law: FollowerLoadField,
        /,
        *,
        support: MechanicalLoadSupport,
        measure_frame: MechanicalLoadFrame,
        load_frame: MechanicalLoadFrame = "current",
        load_id: str,
    ):
        if not callable(law):
            raise TypeError("Follower load law must be callable.")
        identifier = _required_identifier(load_id, "load_id")
        self.law = law
        self._load_id = identifier
        self._semantics = MechanicalLoadSemantics(
            support,
            "current",
            measure_frame,
            load_frame,
            "virtual_work",
            potential_certified=False,
        )

    @property
    def load_id(self) -> str:
        return self._load_id

    @property
    def semantics(self) -> MechanicalLoadSemantics:
        return self._semantics

    def _evaluate(
        self,
        reference,
        current,
        measure,
        reference_normal,
        current_normal,
        state,
        args,
        /,
    ) -> MechanicalLoadEvaluation:
        del reference_normal, current_normal
        force = jnp.asarray(
            self.law(reference, current, measure, state, args), dtype=current.dtype
        )
        try:
            force = jnp.broadcast_to(force, current.shape)
        except ValueError as error:
            raise ValueError(
                "Follower load density must broadcast to the coordinate shape."
            ) from error
        return _single_evaluation(force, None, self, measure)


class CompositeMechanicalLoad(AbstractMechanicalLoad):
    """Static composition on one support with an unreduced component ledger."""

    loads: tuple[AbstractMechanicalLoad, ...]
    _load_id: str = eqx.field(static=True)
    _semantics: MechanicalLoadSemantics = eqx.field(static=True)

    def __init__(
        self,
        loads: Sequence[AbstractMechanicalLoad],
        /,
        *,
        load_id: str | None = None,
    ):
        loads_ = tuple(loads)
        if not loads_ or any(
            not isinstance(load, AbstractMechanicalLoad) for load in loads_
        ):
            raise TypeError("Composite loads require mechanical load children.")
        first = loads_[0].semantics
        for load in loads_[1:]:
            semantics = load.semantics
            if (
                semantics.support,
                semantics.configuration,
                semantics.measure_frame,
                semantics.load_frame,
            ) != (
                first.support,
                first.configuration,
                first.measure_frame,
                first.load_frame,
            ):
                raise ValueError(
                    "Composite load children must share support, configuration, measure, and frame."
                )
        potential = all(load.semantics.conservativity == "potential" for load in loads_)
        if potential:
            closure_ids = {load.semantics.closure_id for load in loads_}
            orientation_ids = {load.semantics.orientation_id for load in loads_}
            if len(closure_ids) != 1 or len(orientation_ids) != 1:
                raise ValueError(
                    "Composite potential loads require identical closure/orientation evidence."
                )
            closure_id = first.closure_id
            orientation_id = first.orientation_id
        else:
            closure_id = None
            orientation_id = None
        generated = canonical_fingerprint(
            {
                "kind": "composite-mechanical-load",
                "children": [load.load_id for load in loads_],
            }
        )
        identifier = generated if load_id is None else str(load_id)
        if not identifier:
            raise ValueError("load_id must be non-empty.")
        self.loads = loads_
        self._load_id = identifier
        self._semantics = MechanicalLoadSemantics(
            first.support,
            first.configuration,
            first.measure_frame,
            first.load_frame,
            "potential" if potential else "virtual_work",
            potential_certified=potential,
            closure_id=closure_id,
            orientation_id=orientation_id,
        )

    @property
    def load_id(self) -> str:
        return self._load_id

    @property
    def semantics(self) -> MechanicalLoadSemantics:
        return self._semantics

    def _evaluate(
        self,
        reference,
        current,
        measure,
        reference_normal,
        current_normal,
        state,
        args,
        /,
    ) -> MechanicalLoadEvaluation:
        children = tuple(
            load._evaluate(
                reference,
                current,
                measure,
                reference_normal,
                current_normal,
                state,
                args,
            )
            for load in self.loads
        )
        components = tuple(
            component
            for child in children
            for component in child.component_force_densities
        )
        component_ids = tuple(
            component_id for child in children for component_id in child.component_ids
        )
        total = sum(components[1:], start=components[0])
        if self.semantics.conservativity == "potential":
            potentials = tuple(child.potential_density for child in children)
            if any(value is None for value in potentials):
                raise ValueError(
                    "A certified composite child did not provide its potential."
                )
            certified = tuple(value for value in potentials if value is not None)
            potential_density = sum(certified[1:], start=certified[0])
        else:
            potential_density = None
        return MechanicalLoadEvaluation(
            total_force_density=total,
            component_force_densities=components,
            component_ids=component_ids,
            potential_density=potential_density,
            semantics=self.semantics,
            valid=jnp.all(jnp.stack(tuple(child.valid for child in children))),
        )


__all__ = [
    "AbstractMechanicalLoad",
    "ClosedSurfacePressure",
    "CompositeMechanicalLoad",
    "CurrentBodyForce",
    "CurrentSurfaceTraction",
    "GeneralFollowerLoad",
    "MechanicalLoadConservativity",
    "MechanicalLoadEvaluation",
    "MechanicalLoadFrame",
    "MechanicalLoadSemantics",
    "MechanicalLoadState",
    "MechanicalLoadSupport",
    "PneumaticPressure",
    "ReferenceDeadBodyForce",
    "ReferenceDeadTraction",
]
