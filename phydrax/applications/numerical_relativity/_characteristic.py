#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class CharacteristicStatus(IntFlag):
    """Fail-closed status bits for characteristic extraction products."""

    SUCCESS = 0
    NONFINITE = 1
    HYPERSURFACE_NOT_CONVERGED = 2
    NONPHYSICAL_MODE_SET = 4
    DERIVATIVE_INVALID = 8
    ENERGY_BALANCE_INVALID = 16


def _positive_capacity(value: int, name: str, /, *, minimum: int = 1) -> int:
    raw = np.asarray(value)
    if raw.shape != () or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must be one integer.")
    result = int(raw)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _time_derivative(values: Array, times: Array, /) -> Array:
    """Differentiate on a fixed, possibly nonuniform grid.

    Interior nodes use the three-point nonuniform formula. End points use their
    only adjacent interval and are deliberately excluded from derivative
    qualification by ``derivative_mask`` in the product.
    """

    previous_width = times[1:-1] - times[:-2]
    next_width = times[2:] - times[1:-1]
    previous_weight = -next_width / (previous_width * (previous_width + next_width))
    center_weight = (next_width - previous_width) / (previous_width * next_width)
    next_weight = previous_width / (next_width * (previous_width + next_width))
    extra_axes = (1,) * (values.ndim - 1)
    interior = (
        previous_weight.reshape((-1,) + extra_axes) * values[:-2]
        + center_weight.reshape((-1,) + extra_axes) * values[1:-1]
        + next_weight.reshape((-1,) + extra_axes) * values[2:]
    )
    first = (values[1] - values[0]) / (times[1] - times[0])
    last = (values[-1] - values[-2]) / (times[-1] - times[-2])
    return jnp.concatenate((first[None], interior, last[None]), axis=0)


class CharacteristicWorldtubeHistory(StrictModule, NonTrainableState):
    """Completed finite-radius Bondi shear history and radial source.

    ``inverse_radius`` runs strictly from a positive worldtube value to zero at
    future null infinity.  The evolved variable is the asymptotic shear
    coefficient ``q = r J`` and ``radial_source`` is ``dq/d(1/r)``.  This is the
    linearized Bondi-Sachs hypersurface equation; callers may supply nonlinear
    source terms already evaluated from a completed worldtube history.
    """

    retarded_times: Array
    inverse_radius: Array
    mode_l: Array
    mode_m: Array
    worldtube_shear_coefficient: Array
    radial_source: Array
    time_capacity: int = eqx.field(static=True)
    radial_capacity: int = eqx.field(static=True)
    mode_capacity: int = eqx.field(static=True)
    history_id: str = eqx.field(static=True)

    def __init__(
        self,
        retarded_times: ArrayLike,
        inverse_radius: ArrayLike,
        mode_l: ArrayLike,
        mode_m: ArrayLike,
        worldtube_shear_coefficient: ArrayLike,
        radial_source: ArrayLike,
        /,
        *,
        history_name: str = "characteristic-worldtube",
    ):
        times = np.asarray(retarded_times)
        radius = np.asarray(inverse_radius)
        ell = np.asarray(mode_l)
        emm = np.asarray(mode_m)
        boundary = np.asarray(worldtube_shear_coefficient)
        source = np.asarray(radial_source)
        if times.ndim != 1 or times.size < 3:
            raise ValueError("retarded_times must contain at least three nodes.")
        if not np.issubdtype(times.dtype, np.number) or np.iscomplexobj(times):
            raise TypeError("retarded_times must be real numeric data.")
        if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError("retarded_times must be finite and strictly increasing.")
        if radius.ndim != 1 or radius.size < 2:
            raise ValueError("inverse_radius must contain worldtube and scri nodes.")
        if not np.issubdtype(radius.dtype, np.number) or np.iscomplexobj(radius):
            raise TypeError("inverse_radius must be real numeric data.")
        if (
            np.any(~np.isfinite(radius))
            or radius[0] <= 0.0
            or np.any(np.diff(radius) >= 0.0)
            or radius[-1] != 0.0
        ):
            raise ValueError(
                "inverse_radius must decrease strictly from the worldtube to zero."
            )
        if (
            ell.ndim != 1
            or emm.shape != ell.shape
            or ell.size == 0
            or not np.issubdtype(ell.dtype, np.integer)
            or not np.issubdtype(emm.dtype, np.integer)
        ):
            raise TypeError("mode_l and mode_m must be nonempty integer vectors.")
        if np.any(ell < 2) or np.any(np.abs(emm) > ell):
            raise ValueError("Radiative shear modes require l >= 2 and |m| <= l.")
        if (
            len({(int(l_value), int(m_value)) for l_value, m_value in zip(ell, emm)})
            != ell.size
        ):
            raise ValueError("Characteristic mode pairs must be unique.")
        expected_boundary = (times.size, ell.size)
        expected_source = (times.size, radius.size, ell.size)
        if boundary.shape != expected_boundary or source.shape != expected_source:
            raise ValueError(
                "Worldtube shear and radial source do not match time/radius/mode capacities."
            )
        if np.any(~np.isfinite(boundary)) or np.any(~np.isfinite(source)):
            raise ValueError("Characteristic history arrays must be finite.")
        if not isinstance(history_name, str) or not history_name:
            raise ValueError("history_name must be a nonempty string.")

        self.retarded_times = jnp.asarray(times)
        self.inverse_radius = jnp.asarray(radius)
        self.mode_l = jnp.asarray(ell, dtype=jnp.int32)
        self.mode_m = jnp.asarray(emm, dtype=jnp.int32)
        self.worldtube_shear_coefficient = jnp.asarray(boundary)
        self.radial_source = jnp.asarray(source)
        self.time_capacity = times.size
        self.radial_capacity = radius.size
        self.mode_capacity = ell.size
        self.history_id = canonical_fingerprint(
            {
                "kind": "completed-characteristic-worldtube-history",
                "name": history_name,
                "retarded_times": times,
                "inverse_radius": radius,
                "mode_l": ell,
                "mode_m": emm,
                "worldtube_shear_coefficient": boundary,
                "radial_source": source,
            }
        )


class CharacteristicWaveformProduct(StrictModule):
    """Fixed-capacity radiation product at future null infinity."""

    retarded_times: Array
    mode_l: Array
    mode_m: Array
    strain_modes: Array
    news_modes: Array
    psi4_modes: Array
    energy_flux: Array
    radiated_energy: Array
    derivative_mask: Array
    hypersurface_residual: Array
    maximum_hypersurface_residual: Array
    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    history_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


class CharacteristicEvolutionResult(StrictModule):
    """Radial characteristic field together with its scri waveform product."""

    radial_shear_coefficient: Array
    waveform: CharacteristicWaveformProduct
    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    history_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class CharacteristicEvolutionPlan(StrictModule, NonTrainableState):
    """Fixed-capacity linearized Bondi-Sachs hypersurface evolution."""

    time_capacity: int = eqx.field(static=True)
    radial_capacity: int = eqx.field(static=True)
    mode_capacity: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_capacity: int,
        radial_capacity: int,
        mode_capacity: int,
        /,
        *,
        absolute_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-8,
        plan_name: str = "bondi-sachs-characteristic-evolution",
    ):
        time_count = _positive_capacity(time_capacity, "time_capacity", minimum=3)
        radial_count = _positive_capacity(radial_capacity, "radial_capacity", minimum=2)
        mode_count = _positive_capacity(mode_capacity, "mode_capacity")
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if (
            not np.isfinite(absolute)
            or not np.isfinite(relative)
            or absolute <= 0.0
            or relative < 0.0
        ):
            raise ValueError("Characteristic tolerances must be finite and admissible.")
        if not isinstance(plan_name, str) or not plan_name:
            raise ValueError("plan_name must be a nonempty string.")
        self.time_capacity = time_count
        self.radial_capacity = radial_count
        self.mode_capacity = mode_count
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-characteristic-evolution-plan",
                "name": plan_name,
                "time_capacity": time_count,
                "radial_capacity": radial_count,
                "mode_capacity": mode_count,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
            }
        )

    def evolve(
        self, history: CharacteristicWorldtubeHistory, /
    ) -> CharacteristicEvolutionResult:
        if (
            history.time_capacity != self.time_capacity
            or history.radial_capacity != self.radial_capacity
            or history.mode_capacity != self.mode_capacity
        ):
            raise ValueError("Worldtube history does not match prepared capacities.")

        radial_step = history.inverse_radius[1:] - history.inverse_radius[:-1]
        increments = (
            0.5
            * radial_step[None, :, None]
            * (history.radial_source[:, :-1] + history.radial_source[:, 1:])
        )
        radial_tail = history.worldtube_shear_coefficient[:, None, :] + jnp.cumsum(
            increments, axis=1
        )
        radial_field = jnp.concatenate(
            (history.worldtube_shear_coefficient[:, None, :], radial_tail), axis=1
        )
        radial_derivative = jnp.swapaxes(
            _time_derivative(jnp.swapaxes(radial_field, 0, 1), history.inverse_radius),
            0,
            1,
        )
        hypersurface_residual = radial_derivative - history.radial_source
        maximum_residual = jnp.max(jnp.abs(hypersurface_residual))
        source_scale = jnp.max(jnp.abs(history.radial_source))
        converged = maximum_residual <= (
            self.absolute_tolerance + self.relative_tolerance * source_scale
        )

        strain = radial_field[:, -1]
        news = _time_derivative(strain, history.retarded_times)
        psi4 = _time_derivative(news, history.retarded_times)
        mode_power = jnp.real(news * jnp.conj(news))
        energy_flux = ein.contract("tm->t", mode_power) / (16.0 * jnp.pi)
        energy_increment = (
            0.5
            * (energy_flux[1:] + energy_flux[:-1])
            * (history.retarded_times[1:] - history.retarded_times[:-1])
        )
        radiated_energy = jnp.concatenate(
            (jnp.zeros((1,), dtype=energy_flux.dtype), jnp.cumsum(energy_increment))
        )
        derivative_mask = (jnp.arange(self.time_capacity) > 0) & (
            jnp.arange(self.time_capacity) < self.time_capacity - 1
        )

        finite = (
            jnp.all(jnp.isfinite(radial_field))
            & jnp.all(jnp.isfinite(news))
            & jnp.all(jnp.isfinite(psi4))
            & jnp.all(jnp.isfinite(energy_flux))
            & jnp.all(jnp.isfinite(radiated_energy))
        )
        mode_valid = jnp.all(history.mode_l >= 2) & jnp.all(
            jnp.abs(history.mode_m) <= history.mode_l
        )
        energy_valid = jnp.all(energy_flux >= 0.0) & jnp.all(
            radiated_energy[1:] >= radiated_energy[:-1]
        )
        physically_valid = mode_valid & energy_valid
        derivative_valid = (
            finite
            & jnp.all(jnp.isfinite(news[derivative_mask]))
            & jnp.all(jnp.isfinite(psi4[derivative_mask]))
        )
        qualified = finite & converged & physically_valid & derivative_valid

        status = jnp.asarray(int(CharacteristicStatus.SUCCESS), dtype=jnp.int32)
        status = status | jnp.where(
            finite, 0, int(CharacteristicStatus.NONFINITE)
        ).astype(jnp.int32)
        status = status | jnp.where(
            converged, 0, int(CharacteristicStatus.HYPERSURFACE_NOT_CONVERGED)
        ).astype(jnp.int32)
        status = status | jnp.where(
            mode_valid, 0, int(CharacteristicStatus.NONPHYSICAL_MODE_SET)
        ).astype(jnp.int32)
        status = status | jnp.where(
            derivative_valid, 0, int(CharacteristicStatus.DERIVATIVE_INVALID)
        ).astype(jnp.int32)
        status = status | jnp.where(
            energy_valid, 0, int(CharacteristicStatus.ENERGY_BALANCE_INVALID)
        ).astype(jnp.int32)

        product_id = canonical_fingerprint(
            {
                "kind": "characteristic-waveform-product",
                "history": history.history_id,
                "plan": self.plan_id,
            }
        )
        waveform = CharacteristicWaveformProduct(
            history.retarded_times,
            history.mode_l,
            history.mode_m,
            strain,
            news,
            psi4,
            energy_flux,
            radiated_energy,
            derivative_mask,
            hypersurface_residual,
            maximum_residual,
            status,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            history.history_id,
            self.plan_id,
            product_id,
        )
        result_id = canonical_fingerprint(
            {
                "kind": "characteristic-evolution-result",
                "waveform": product_id,
                "radial_capacity": self.radial_capacity,
            }
        )
        return CharacteristicEvolutionResult(
            radial_field,
            waveform,
            status,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            history.history_id,
            self.plan_id,
            result_id,
        )
