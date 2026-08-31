#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule


MaxwellFieldKind: TypeAlias = Literal["electric", "magnetic"]


class AbstractPreparedMaxwellObserver(StrictModule):
    """Prepared streaming observer over synchronized physical fields."""

    prepared_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def initialize(self, /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def update(
        self,
        time: Array,
        electric: Array,
        magnetic: Array,
        state: Any,
        /,
    ) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def value(self, state: Any, /) -> Array:
        raise NotImplementedError


class AbstractMaxwellObserverPlan(StrictModule):
    """Plan that binds an observer to electric and magnetic cochain sizes."""

    plan_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def prepare(
        self,
        layout: Any,
        /,
    ) -> AbstractPreparedMaxwellObserver:
        raise NotImplementedError


class FieldProbePlan(AbstractMaxwellObserverPlan):
    """Native or weighted arbitrary-point probe for one field kind."""

    field: MaxwellFieldKind = eqx.field(static=True)
    indices: Array
    weights: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        field: MaxwellFieldKind,
        indices: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
    ):
        if field not in ("electric", "magnetic"):
            raise ValueError("Probe field must be 'electric' or 'magnetic'.")
        indices_ = np.asarray(indices)
        if indices_.ndim == 1:
            indices_ = indices_[:, None]
        if indices_.ndim != 2 or not np.issubdtype(indices_.dtype, np.integer):
            raise TypeError("Probe indices must be an integer (probes, support) array.")
        if indices_.shape[1] <= 0:
            raise ValueError("Every probe requires nonempty support.")
        if weights is None:
            weights_ = np.ones(indices_.shape, dtype=float)
        else:
            weights_ = np.asarray(weights, dtype=float)
        if weights_.shape != indices_.shape or np.any(~np.isfinite(weights_)):
            raise ValueError("Probe weights must be finite and match indices.")
        sums = np.sum(weights_, axis=1)
        if np.any(np.abs(sums) <= np.finfo(float).eps):
            raise ValueError("Probe weights must have nonzero row sums.")
        weights_ = weights_ / sums[:, None]
        self.field = field
        self.indices = jnp.asarray(indices_, dtype=jnp.int32)
        self.weights = jnp.asarray(weights_)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-field-probe",
                "field": field,
                "indices": array_tree_fingerprint(indices_),
                "weights": array_tree_fingerprint(weights_),
            }
        )

    def prepare(
        self,
        layout: Any,
        /,
    ) -> PreparedFieldProbe:
        count = (
            layout.electric_count if self.field == "electric" else layout.magnetic_count
        )
        if bool(jnp.any(self.indices < 0)) or bool(jnp.any(self.indices >= count)):
            raise ValueError("Probe indices are outside the selected cochain field.")
        return PreparedFieldProbe(self)


class PreparedFieldProbe(AbstractPreparedMaxwellObserver):
    field: MaxwellFieldKind = eqx.field(static=True)
    indices: Array
    weights: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: FieldProbePlan, /):
        self.field = plan.field
        self.indices = plan.indices
        self.weights = plan.weights
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-field-probe", "plan": plan.plan_id}
        )

    def initialize(self, /) -> Array:
        return jnp.zeros((self.indices.shape[0],))

    def update(
        self,
        time: Array,
        electric: Array,
        magnetic: Array,
        state: Any,
        /,
    ) -> Array:
        del time, state
        field = electric if self.field == "electric" else magnetic
        return jnp.sum(self.weights.astype(field.dtype) * field[self.indices], axis=1)

    def value(self, state: Any, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != (self.indices.shape[0],):
            raise ValueError("Probe state has wrong shape.")
        return value


class DFTObserverState(StrictModule):
    accumulator: Array
    normalization: Array
    samples: Array


class DFTObserverPlan(AbstractMaxwellObserverPlan):
    """Streaming complex DFT of a prepared probe payload."""

    probe: FieldProbePlan
    angular_frequencies: Array
    start_time: float = eqx.field(static=True)
    stop_time: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        probe: FieldProbePlan,
        angular_frequencies: ArrayLike,
        /,
        *,
        start_time: float = 0.0,
        stop_time: float | None = None,
    ):
        if not isinstance(probe, FieldProbePlan):
            raise TypeError("probe must be a FieldProbePlan.")
        frequencies = jnp.asarray(angular_frequencies, dtype=float)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("angular_frequencies must be a nonempty vector.")
        if bool(jnp.any(~jnp.isfinite(frequencies))) or bool(jnp.any(frequencies < 0.0)):
            raise ValueError("angular_frequencies must be finite and nonnegative.")
        start = float(start_time)
        stop = None if stop_time is None else float(stop_time)
        if not np.isfinite(start) or (
            stop is not None and (not np.isfinite(stop) or stop < start)
        ):
            raise ValueError("DFT start/stop times are invalid.")
        self.probe = probe
        self.angular_frequencies = frequencies
        self.start_time = start
        self.stop_time = stop
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-dft-observer",
                "probe": probe.plan_id,
                "frequencies": array_tree_fingerprint(frequencies),
                "start": start,
                "stop": stop,
            }
        )

    def prepare(
        self,
        layout: Any,
        /,
    ) -> PreparedDFTObserver:
        return PreparedDFTObserver(
            self,
            self.probe.prepare(layout),
        )


class PreparedDFTObserver(AbstractPreparedMaxwellObserver):
    probe: PreparedFieldProbe
    angular_frequencies: Array
    start_time: float = eqx.field(static=True)
    stop_time: float | None = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: DFTObserverPlan, probe: PreparedFieldProbe, /):
        self.probe = probe
        self.angular_frequencies = plan.angular_frequencies
        self.start_time = plan.start_time
        self.stop_time = plan.stop_time
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-maxwell-dft", "plan": plan.plan_id}
        )

    def initialize(self, /) -> DFTObserverState:
        shape = (self.angular_frequencies.size, self.probe.indices.shape[0])
        return DFTObserverState(
            accumulator=jnp.zeros(shape, dtype=complex),
            normalization=jnp.asarray(0.0),
            samples=jnp.asarray(0, dtype=jnp.int32),
        )

    def update(
        self,
        time: Array,
        electric: Array,
        magnetic: Array,
        state: Any,
        /,
    ) -> DFTObserverState:
        if not isinstance(state, DFTObserverState):
            raise TypeError("DFT observer requires DFTObserverState.")
        payload = self.probe.update(time, electric, magnetic, None)
        active = jnp.asarray(time) >= self.start_time
        if self.stop_time is not None:
            active = active & (jnp.asarray(time) <= self.stop_time)
        phase = jnp.exp(-1j * self.angular_frequencies * jnp.asarray(time))
        contribution = phase[:, None] * payload[None, :]
        accumulator = state.accumulator + jnp.where(active, contribution, 0)
        normalization = state.normalization + active.astype(float)
        samples = state.samples + active.astype(jnp.int32)
        return DFTObserverState(accumulator, normalization, samples)

    def value(self, state: Any, /) -> Array:
        if not isinstance(state, DFTObserverState):
            raise TypeError("DFT observer requires DFTObserverState.")
        denominator = jnp.where(state.normalization > 0.0, state.normalization, 1.0)
        return state.accumulator / denominator


class PoyntingFluxPlan(StrictModule):
    """Co-located surface Poynting flux with explicit normals and measures."""

    normals: Array
    measures: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, normals: ArrayLike, measures: ArrayLike, /):
        normals_ = jnp.asarray(normals, dtype=float)
        measures_ = jnp.asarray(measures, dtype=float)
        if normals_.ndim != 2 or normals_.shape[1] != 3:
            raise ValueError("Poynting normals must have shape (points, 3).")
        if measures_.shape != normals_.shape[:1]:
            raise ValueError("Poynting measures must have shape (points,).")
        lengths = jnp.linalg.norm(normals_, axis=1)
        normals_ = (
            eqx.error_if(
                normals_,
                jnp.any(~jnp.isfinite(normals_)) | jnp.any(lengths <= 0.0),
                "Poynting normals must be finite and nonzero.",
            )
            / lengths[:, None]
        )
        measures_ = eqx.error_if(
            measures_,
            jnp.any(~jnp.isfinite(measures_)) | jnp.any(measures_ <= 0.0),
            "Poynting measures must be finite and positive.",
        )
        self.normals = normals_
        self.measures = measures_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "poynting-flux-plan",
                "normals": array_tree_fingerprint(normals_),
                "measures": array_tree_fingerprint(measures_),
            }
        )

    def evaluate(self, electric: ArrayLike, magnetic: ArrayLike, /) -> Array:
        electric_ = jnp.asarray(electric)
        magnetic_ = jnp.asarray(magnetic)
        if electric_.shape != self.normals.shape or magnetic_.shape != self.normals.shape:
            raise ValueError("Poynting fields must have shape (points, 3).")
        flux = jnp.real(
            jnp.sum(jnp.cross(electric_, jnp.conj(magnetic_)) * self.normals, axis=1)
        )
        return jnp.sum(self.measures * flux)


class SynchronizedEnergyObserverPlan(AbstractMaxwellObserverPlan):
    """Synchronized E/H quadratic energy observer with declared weights."""

    electric_weights: Array
    magnetic_weights: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        electric_weights: ArrayLike,
        magnetic_weights: ArrayLike,
        /,
    ):
        electric = jnp.asarray(electric_weights, dtype=float)
        magnetic = jnp.asarray(magnetic_weights, dtype=float)
        if electric.ndim != 1 or magnetic.ndim != 1:
            raise ValueError("Energy observer weights must be vectors.")
        electric = eqx.error_if(
            electric,
            jnp.any(~jnp.isfinite(electric)) | jnp.any(electric <= 0.0),
            "Electric energy weights must be finite and positive.",
        )
        magnetic = eqx.error_if(
            magnetic,
            jnp.any(~jnp.isfinite(magnetic)) | jnp.any(magnetic <= 0.0),
            "Magnetic energy weights must be finite and positive.",
        )
        self.electric_weights = electric
        self.magnetic_weights = magnetic
        self.plan_id = canonical_fingerprint(
            {
                "kind": "synchronized-energy-observer",
                "electric": array_tree_fingerprint(electric),
                "magnetic": array_tree_fingerprint(magnetic),
            }
        )

    def prepare(
        self,
        layout: Any,
        /,
    ) -> PreparedSynchronizedEnergyObserver:
        electric_count, magnetic_count = layout.electric_count, layout.magnetic_count
        if self.electric_weights.shape != (electric_count,):
            raise ValueError("Electric energy weights do not match electric cochains.")
        if self.magnetic_weights.shape != (magnetic_count,):
            raise ValueError("Magnetic energy weights do not match magnetic cochains.")
        return PreparedSynchronizedEnergyObserver(self)


class PreparedSynchronizedEnergyObserver(AbstractPreparedMaxwellObserver):
    electric_weights: Array
    magnetic_weights: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: SynchronizedEnergyObserverPlan, /):
        self.electric_weights = plan.electric_weights
        self.magnetic_weights = plan.magnetic_weights
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-synchronized-energy-observer", "plan": plan.plan_id}
        )

    def initialize(self, /) -> Array:
        return jnp.asarray(0.0)

    def update(
        self,
        time: Array,
        electric: Array,
        magnetic: Array,
        state: Any,
        /,
    ) -> Array:
        del time, state
        return 0.5 * (
            jnp.sum(self.electric_weights * jnp.real(electric * jnp.conj(electric)))
            + jnp.sum(self.magnetic_weights * jnp.real(magnetic * jnp.conj(magnetic)))
        )

    def value(self, state: Any, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != ():
            raise ValueError("Synchronized energy observer state must be scalar.")
        return value


class ModeAmplitudeObserverState(StrictModule):
    accumulator: Array
    samples: Array


class ModeAmplitudeObserverPlan(AbstractMaxwellObserverPlan):
    """Streaming paired E/H modal amplitudes without field-history storage."""

    electric_modes: Array
    magnetic_modes: Array
    angular_frequencies: Array
    direction: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        electric_modes: ArrayLike,
        magnetic_modes: ArrayLike,
        angular_frequencies: ArrayLike,
        /,
        *,
        direction: int = 1,
    ):
        electric = jnp.asarray(electric_modes)
        magnetic = jnp.asarray(magnetic_modes)
        frequencies = jnp.asarray(angular_frequencies, dtype=float)
        if (
            electric.ndim != 2
            or magnetic.ndim != 2
            or electric.shape[1] != magnetic.shape[1]
        ):
            raise ValueError("Paired modal bases must have matching mode axes.")
        if (
            frequencies.ndim != 1
            or frequencies.size == 0
            or bool(jnp.any(frequencies < 0.0))
        ):
            raise ValueError(
                "Modal observer frequencies must be nonempty and nonnegative."
            )
        if direction not in (-1, 1):
            raise ValueError("Modal observer direction must be -1 or +1.")
        self.electric_modes, self.magnetic_modes = electric, magnetic
        self.angular_frequencies, self.direction = frequencies, int(direction)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-mode-amplitude-observer",
                "electric_modes": array_tree_fingerprint(electric),
                "magnetic_modes": array_tree_fingerprint(magnetic),
                "frequencies": array_tree_fingerprint(frequencies),
                "direction": direction,
            }
        )

    def prepare(self, layout: Any, /) -> PreparedModeAmplitudeObserver:
        if self.electric_modes.shape[0] != layout.electric_count:
            raise ValueError(
                "Electric mode traces do not match the retained electric space."
            )
        if self.magnetic_modes.shape[0] != layout.magnetic_count:
            raise ValueError(
                "Magnetic mode traces do not match the retained magnetic space."
            )
        return PreparedModeAmplitudeObserver(self)


class PreparedModeAmplitudeObserver(AbstractPreparedMaxwellObserver):
    electric_modes: Array
    magnetic_modes: Array
    angular_frequencies: Array
    direction: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ModeAmplitudeObserverPlan, /):
        self.electric_modes = plan.electric_modes
        self.magnetic_modes = plan.magnetic_modes
        self.angular_frequencies = plan.angular_frequencies
        self.direction = plan.direction
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-mode-amplitude-observer", "plan": plan.plan_id}
        )

    def initialize(self, /) -> ModeAmplitudeObserverState:
        shape = (self.angular_frequencies.size, self.electric_modes.shape[1])
        return ModeAmplitudeObserverState(
            jnp.zeros(shape, dtype=complex),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def update(
        self,
        time: Array,
        electric: Array,
        magnetic: Array,
        state: Any,
        /,
    ) -> ModeAmplitudeObserverState:
        if not isinstance(state, ModeAmplitudeObserverState):
            raise TypeError("Modal observer requires ModeAmplitudeObserverState.")
        electric_part = jnp.conj(self.electric_modes.T) @ electric
        magnetic_part = jnp.conj(self.magnetic_modes.T) @ magnetic
        amplitude = 0.5 * (electric_part + self.direction * magnetic_part)
        phase = jnp.exp(1j * self.angular_frequencies * time)
        return ModeAmplitudeObserverState(
            state.accumulator + phase[:, None] * amplitude[None, :],
            state.samples + 1,
        )

    def value(self, state: Any, /) -> Array:
        if not isinstance(state, ModeAmplitudeObserverState):
            raise TypeError("Modal observer requires ModeAmplitudeObserverState.")
        return state.accumulator / jnp.maximum(state.samples, 1)


__all__ = [
    "AbstractMaxwellObserverPlan",
    "AbstractPreparedMaxwellObserver",
    "DFTObserverPlan",
    "DFTObserverState",
    "FieldProbePlan",
    "MaxwellFieldKind",
    "ModeAmplitudeObserverPlan",
    "ModeAmplitudeObserverState",
    "PoyntingFluxPlan",
    "PreparedDFTObserver",
    "PreparedFieldProbe",
    "PreparedModeAmplitudeObserver",
    "PreparedSynchronizedEnergyObserver",
    "SynchronizedEnergyObserverPlan",
]
