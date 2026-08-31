#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge


class MaxwellSourceForcing(StrictModule):
    """Paired electric and magnetic current cochains at one sample time."""

    electric_current: Array
    magnetic_current: Array


class AbstractMaxwellSourcePlan(StrictModule):
    """Static source geometry lowered once to retained Maxwell cochains."""

    source_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def prepare(
        self, bridge: StructuredCochainBridge, layout: Any, /
    ) -> PreparedMaxwellSource:
        raise NotImplementedError


class PreparedMaxwellSource(StrictModule):
    """Sparse prepared source with harmonic time dependence and dynamic control."""

    electric_indices: Array
    electric_profile: Array
    magnetic_indices: Array
    magnetic_profile: Array
    electric_count: int = eqx.field(static=True)
    magnetic_count: int = eqx.field(static=True)
    angular_frequency: Array
    phase: Array
    amplitude: Array
    control_key: str | None = eqx.field(static=True)
    envelope: Callable[[Array, Any], ArrayLike] | None = eqx.field(static=True)
    magnetic_closedness_preserving: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        electric_indices: ArrayLike,
        electric_profile: ArrayLike,
        magnetic_indices: ArrayLike,
        magnetic_profile: ArrayLike,
        electric_count: int,
        magnetic_count: int,
        angular_frequency: ArrayLike,
        phase: ArrayLike,
        amplitude: ArrayLike,
        control_key: str | None,
        envelope: Callable[[Array, Any], ArrayLike] | None,
        magnetic_closedness_preserving: bool,
        source_id: str,
        layout_id: str,
    ):
        e_indices = jnp.asarray(electric_indices, dtype=jnp.int32)
        m_indices = jnp.asarray(magnetic_indices, dtype=jnp.int32)
        e_profile = jnp.asarray(electric_profile)
        m_profile = jnp.asarray(magnetic_profile)
        if e_indices.ndim != 1 or m_indices.ndim != 1:
            raise ValueError("Prepared Maxwell source indices must be vectors.")
        if e_profile.shape != e_indices.shape or m_profile.shape != m_indices.shape:
            raise ValueError("Prepared Maxwell source profiles must align with indices.")
        e_count, m_count = int(electric_count), int(magnetic_count)
        if (
            np.any(np.asarray(e_indices) < 0)
            or np.any(np.asarray(e_indices) >= e_count)
            or np.any(np.asarray(m_indices) < 0)
            or np.any(np.asarray(m_indices) >= m_count)
        ):
            raise ValueError("Prepared Maxwell source index lies outside its cochain.")
        frequency = jnp.asarray(angular_frequency)
        phase_ = jnp.asarray(phase)
        amplitude_ = jnp.asarray(amplitude)
        if frequency.shape != () or phase_.shape != () or amplitude_.shape != ():
            raise ValueError("Source frequency, phase, and amplitude must be scalars.")
        if jnp.iscomplexobj(frequency) or jnp.iscomplexobj(phase_):
            raise TypeError("Source frequency and phase must be real.")
        if bool(jnp.any(~jnp.isfinite(frequency))) or bool(frequency < 0.0):
            raise ValueError("Source angular frequency must be finite and nonnegative.")
        if bool(jnp.any(~jnp.isfinite(phase_))) or bool(
            jnp.any(~jnp.isfinite(amplitude_))
        ):
            raise ValueError("Source phase and amplitude must be finite.")
        if control_key is not None and not str(control_key):
            raise ValueError("Source control_key must be nonempty when supplied.")
        if envelope is not None and not callable(envelope):
            raise TypeError("Source envelope must be callable or None.")
        self.electric_indices = e_indices
        self.electric_profile = e_profile
        self.magnetic_indices = m_indices
        self.magnetic_profile = m_profile
        self.electric_count = e_count
        self.magnetic_count = m_count
        self.angular_frequency = frequency
        self.phase = phase_
        self.amplitude = amplitude_
        self.control_key = None if control_key is None else str(control_key)
        self.envelope = envelope
        self.magnetic_closedness_preserving = bool(magnetic_closedness_preserving)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-maxwell-source",
                "source": source_id,
                "layout": layout_id,
                "electric_indices": array_tree_fingerprint(e_indices),
                "magnetic_indices": array_tree_fingerprint(m_indices),
            }
        )

    def _amplitude(self, args: Any, /) -> Array:
        if self.control_key is None:
            return self.amplitude
        if not isinstance(args, dict) or self.control_key not in args:
            raise ValueError(f"Maxwell source requires control {self.control_key!r}.")
        control = jnp.asarray(args[self.control_key])
        if control.shape != ():
            raise ValueError("Maxwell source control amplitude must be scalar.")
        return self.amplitude * control

    def sample(self, time: ArrayLike, args: Any = None, /) -> MaxwellSourceForcing:
        time_ = jnp.asarray(time)
        if time_.shape != ():
            raise ValueError("Maxwell source sample time must be scalar.")
        if self.envelope is None:
            temporal = jnp.exp(-1j * (self.angular_frequency * time_ - self.phase))
        else:
            temporal = jnp.asarray(self.envelope(time_, args))
            if temporal.shape != ():
                raise ValueError("Maxwell source envelope must return a scalar.")
        envelope = self._amplitude(args) * temporal
        dtype = jnp.result_type(envelope, self.electric_profile, self.magnetic_profile)
        electric = (
            jnp.zeros((self.electric_count,), dtype=dtype)
            .at[self.electric_indices]
            .add(envelope * self.electric_profile)
        )
        magnetic = (
            jnp.zeros((self.magnetic_count,), dtype=dtype)
            .at[self.magnetic_indices]
            .add(envelope * self.magnetic_profile)
        )
        return MaxwellSourceForcing(electric, magnetic)


class MaxwellElectricCurrentSourcePlan(AbstractMaxwellSourcePlan, NonTrainableState):
    """Sparse electric-current source with a prepared spatial profile."""

    indices: Array
    profile: Array
    angular_frequency: Array
    phase: Array
    amplitude: Array
    control_key: str | None = eqx.field(static=True)
    envelope: Callable[[Array, Any], ArrayLike] | None = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        indices: ArrayLike,
        profile: ArrayLike,
        /,
        *,
        angular_frequency: ArrayLike = 0.0,
        phase: ArrayLike = 0.0,
        amplitude: ArrayLike = 1.0,
        control_key: str | None = None,
        envelope: Callable[[Array, Any], ArrayLike] | None = None,
    ):
        indices_ = jnp.asarray(indices, dtype=jnp.int32)
        profile_ = jnp.asarray(profile)
        if indices_.ndim != 1 or profile_.shape != indices_.shape:
            raise ValueError("Electric source indices/profile must be aligned vectors.")
        self.indices = indices_
        self.profile = profile_
        self.angular_frequency = jnp.asarray(angular_frequency)
        self.phase = jnp.asarray(phase)
        self.amplitude = jnp.asarray(amplitude)
        self.control_key = control_key
        self.envelope = envelope
        self.source_id = canonical_fingerprint(
            {
                "kind": "maxwell-electric-current-source-plan",
                "indices": array_tree_fingerprint(indices_),
                "profile": array_tree_fingerprint(profile_),
                "control_key": control_key,
                "envelope": None if envelope is None else repr(envelope),
            }
        )

    def prepare(
        self, bridge: StructuredCochainBridge, layout: Any, /
    ) -> PreparedMaxwellSource:
        if layout.layout_id == "":
            raise ValueError("Maxwell source layout identity must be nonempty.")
        return PreparedMaxwellSource(
            electric_indices=self.indices,
            electric_profile=self.profile,
            magnetic_indices=jnp.zeros((0,), dtype=jnp.int32),
            magnetic_profile=jnp.zeros((0,), dtype=self.profile.dtype),
            electric_count=layout.electric_count,
            magnetic_count=layout.magnetic_count,
            angular_frequency=self.angular_frequency,
            phase=self.phase,
            amplitude=self.amplitude,
            control_key=self.control_key,
            envelope=self.envelope,
            magnetic_closedness_preserving=True,
            source_id=self.source_id,
            layout_id=layout.layout_id,
        )


class MaxwellPairedCurrentSourcePlan(AbstractMaxwellSourcePlan, NonTrainableState):
    """Prepared paired J/M source used by discrete Huygens and mode launches."""

    electric_indices: Array
    electric_profile: Array
    magnetic_indices: Array
    magnetic_profile: Array
    angular_frequency: Array
    phase: Array
    amplitude: Array
    control_key: str | None = eqx.field(static=True)
    envelope: Callable[[Array, Any], ArrayLike] | None = eqx.field(static=True)
    magnetic_closedness_preserving: bool = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        electric_indices: ArrayLike,
        electric_profile: ArrayLike,
        magnetic_indices: ArrayLike,
        magnetic_profile: ArrayLike,
        /,
        *,
        angular_frequency: ArrayLike = 0.0,
        phase: ArrayLike = 0.0,
        amplitude: ArrayLike = 1.0,
        control_key: str | None = None,
        envelope: Callable[[Array, Any], ArrayLike] | None = None,
        magnetic_closedness_preserving: bool = False,
        source_id: str | None = None,
    ):
        e_indices = jnp.asarray(electric_indices, dtype=jnp.int32)
        m_indices = jnp.asarray(magnetic_indices, dtype=jnp.int32)
        e_profile = jnp.asarray(electric_profile)
        m_profile = jnp.asarray(magnetic_profile)
        if e_indices.ndim != 1 or m_indices.ndim != 1:
            raise ValueError("Paired source indices must be vectors.")
        if e_profile.shape != e_indices.shape or m_profile.shape != m_indices.shape:
            raise ValueError("Paired source profiles must align with their indices.")
        identifier = source_id or canonical_fingerprint(
            {
                "kind": "maxwell-paired-current-source-plan",
                "electric_indices": array_tree_fingerprint(e_indices),
                "electric_profile": array_tree_fingerprint(e_profile),
                "magnetic_indices": array_tree_fingerprint(m_indices),
                "magnetic_profile": array_tree_fingerprint(m_profile),
            }
        )
        self.electric_indices = e_indices
        self.electric_profile = e_profile
        self.magnetic_indices = m_indices
        self.magnetic_profile = m_profile
        self.angular_frequency = jnp.asarray(angular_frequency)
        self.phase = jnp.asarray(phase)
        self.amplitude = jnp.asarray(amplitude)
        self.control_key = control_key
        self.envelope = envelope
        self.magnetic_closedness_preserving = bool(magnetic_closedness_preserving)
        self.source_id = identifier

    def prepare(
        self, bridge: StructuredCochainBridge, layout: Any, /
    ) -> PreparedMaxwellSource:
        del bridge
        return PreparedMaxwellSource(
            electric_indices=self.electric_indices,
            electric_profile=self.electric_profile,
            magnetic_indices=self.magnetic_indices,
            magnetic_profile=self.magnetic_profile,
            electric_count=layout.electric_count,
            magnetic_count=layout.magnetic_count,
            angular_frequency=self.angular_frequency,
            phase=self.phase,
            amplitude=self.amplitude,
            control_key=self.control_key,
            envelope=self.envelope,
            magnetic_closedness_preserving=self.magnetic_closedness_preserving,
            source_id=self.source_id,
            layout_id=layout.layout_id,
        )


__all__ = [
    "AbstractMaxwellSourcePlan",
    "MaxwellElectricCurrentSourcePlan",
    "MaxwellPairedCurrentSourcePlan",
    "MaxwellSourceForcing",
    "PreparedMaxwellSource",
]
