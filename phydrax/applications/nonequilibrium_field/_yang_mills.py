#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._keldysh import NonequilibriumStatus


def _quaternion_multiply(left: Array, right: Array, /) -> Array:
    left_scalar, left_vector = left[..., :1], left[..., 1:]
    right_scalar, right_vector = right[..., :1], right[..., 1:]
    scalar = (
        left_scalar * right_scalar
        - contract("...i,...i->...", left_vector, right_vector)[..., None]
    )
    vector = (
        left_scalar * right_vector
        + right_scalar * left_vector
        + jnp.cross(left_vector, right_vector)
    )
    return jnp.concatenate((scalar, vector), axis=-1)


def _quaternion_dagger(value: Array, /) -> Array:
    return jnp.concatenate((value[..., :1], -value[..., 1:]), axis=-1)


def _quaternion_rotate(quaternion: Array, vector: Array, /) -> Array:
    pure = jnp.concatenate((jnp.zeros_like(vector[..., :1]), vector), axis=-1)
    return _quaternion_multiply(
        _quaternion_multiply(quaternion, pure), _quaternion_dagger(quaternion)
    )[..., 1:]


def _su2_exponential(generator: Array, /) -> Array:
    angle = jnp.linalg.norm(generator, axis=-1)
    safe_angle = jnp.where(angle > 1.0e-8, angle, 1.0)
    regular = jnp.sin(safe_angle) / safe_angle
    series = 1.0 - angle**2 / 6.0 + angle**4 / 120.0
    coefficient = jnp.where(angle > 1.0e-8, regular, series)
    return jnp.concatenate(
        (jnp.cos(angle)[..., None], coefficient[..., None] * generator), axis=-1
    )


class ClassicalYangMillsState(StrictModule):
    """Periodic SU(2) links and left electric Lie-algebra coordinates."""

    links: Array
    electric_fields: Array
    step_index: Array
    plan_id: str = eqx.field(static=True)


class YangMillsWardEvidence(StrictModule):
    energy_residual: Array
    gauss_covariance_residual: Array
    link_norm_residual: Array
    finite: Array
    satisfied: Array
    plan_id: str = eqx.field(static=True)


class YangMillsConservationDiagnostics(StrictModule):
    energy: Array
    gauss_residual: Array
    link_norm_residual: Array
    relative_energy_drift: Array
    maximum_gauss_residual: Array
    maximum_link_norm_residual: Array
    finite: Array
    gauss_conserved: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class ClassicalYangMillsResult(StrictModule):
    final_state: ClassicalYangMillsState
    diagnostics: YangMillsConservationDiagnostics
    plan_id: str = eqx.field(static=True)


class ClassicalYangMillsPlan(StrictModule, NonTrainableState):
    """Resource-bounded temporal-gauge SU(2) Kogut--Susskind leapfrog."""

    lattice_shape: tuple[int, ...] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    electric_coupling: float = eqx.field(static=True)
    magnetic_coupling: float = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    maximum_links: int = eqx.field(static=True)
    maximum_history_elements: int = eqx.field(static=True)
    gauss_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lattice_shape: tuple[int, ...],
        spacing: tuple[float, ...],
        /,
        *,
        electric_coupling: float = 1.0,
        magnetic_coupling: float = 1.0,
        time_step: float,
        step_count: int,
        maximum_links: int = 1_000_000,
        maximum_history_elements: int = 10_000_000,
        gauss_tolerance: float = 1.0e-8,
        courant_limit: float = 0.5,
    ):
        shape = tuple(int(value) for value in lattice_shape)
        spacing_ = tuple(float(value) for value in spacing)
        dimension = len(shape)
        electric = float(electric_coupling)
        magnetic = float(magnetic_coupling)
        dt = float(time_step)
        steps = int(step_count)
        link_capacity = int(maximum_links)
        history_capacity = int(maximum_history_elements)
        tolerance = float(gauss_tolerance)
        courant = (
            dt
            * np.sqrt(electric * magnetic)
            * np.sqrt(sum(value**-2 for value in spacing_))
        )
        link_count = int(np.prod(shape, dtype=np.int64)) * dimension
        history_elements = (steps + 1) * 3
        if (
            dimension not in (2, 3)
            or len(spacing_) != dimension
            or any(value < 2 for value in shape)
            or any(not np.isfinite(value) or value <= 0.0 for value in spacing_)
            or not np.isfinite(electric)
            or not np.isfinite(magnetic)
            or electric <= 0.0
            or magnetic <= 0.0
            or not np.isfinite(dt)
            or dt <= 0.0
            or steps <= 0
            or link_count > link_capacity
            or history_elements > history_capacity
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or not np.isfinite(courant_limit)
            or courant_limit <= 0.0
            or courant > courant_limit
        ):
            raise ValueError(
                "Yang--Mills lattice, couplings, timestep, or budget is invalid."
            )
        self.lattice_shape = shape
        self.spacing = spacing_
        self.electric_coupling = electric
        self.magnetic_coupling = magnetic
        self.time_step = dt
        self.step_count = steps
        self.maximum_links = link_capacity
        self.maximum_history_elements = history_capacity
        self.gauss_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "classical-statistical-su2-yang-mills",
                "lattice_shape": shape,
                "spacing": spacing_,
                "electric_coupling": electric,
                "magnetic_coupling": magnetic,
                "time_step": dt,
                "step_count": steps,
                "maximum_links": link_capacity,
                "maximum_history_elements": history_capacity,
                "gauss_tolerance": tolerance,
                "integrator": "lie-group-kick-drift-kick",
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.lattice_shape)

    def vacuum_state(self, /) -> ClassicalYangMillsState:
        links = jnp.zeros(self.lattice_shape + (self.dimension, 4))
        links = links.at[..., 0].set(1.0)
        electric = jnp.zeros(self.lattice_shape + (self.dimension, 3))
        return ClassicalYangMillsState(
            links,
            electric,
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def prepare(
        self, state: ClassicalYangMillsState, /
    ) -> "PreparedClassicalYangMillsEvolution":
        return PreparedClassicalYangMillsEvolution(self, state)


class PreparedClassicalYangMillsEvolution(StrictModule, NonTrainableState):
    __hash__ = object.__hash__

    plan: ClassicalYangMillsPlan
    initial_state: ClassicalYangMillsState
    initial_gauss: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ClassicalYangMillsPlan,
        state: ClassicalYangMillsState,
        /,
    ):
        if not isinstance(plan, ClassicalYangMillsPlan):
            raise TypeError("plan must be ClassicalYangMillsPlan.")
        if (
            not isinstance(state, ClassicalYangMillsState)
            or state.plan_id != plan.plan_id
        ):
            raise ValueError("Initial Yang--Mills state belongs to another plan.")
        expected_links = plan.lattice_shape + (plan.dimension, 4)
        expected_electric = plan.lattice_shape + (plan.dimension, 3)
        links = np.asarray(state.links)
        electric = np.asarray(state.electric_fields)
        norms = np.sum(links**2, axis=-1)
        if (
            links.shape != expected_links
            or electric.shape != expected_electric
            or np.any(~np.isfinite(links))
            or np.any(~np.isfinite(electric))
            or np.max(np.abs(norms - 1.0)) > plan.gauss_tolerance
        ):
            raise ValueError("Initial SU(2) links/electric fields are invalid.")
        initial_gauss = yang_mills_gauss(state)
        if float(jnp.max(jnp.abs(initial_gauss))) > plan.gauss_tolerance:
            raise ValueError("Initial electric fields violate the Gauss constraint.")
        self.plan = plan
        self.initial_state = state
        self.initial_gauss = initial_gauss
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-classical-statistical-yang-mills",
                "plan": plan.plan_id,
                "links": array_tree_fingerprint(links),
                "electric_fields": array_tree_fingerprint(electric),
            }
        )

    def _plaquette_energy(self, links: Array, /) -> Array:
        energy = jnp.asarray(0.0, dtype=links.dtype)
        for first in range(self.plan.dimension):
            for second in range(first + 1, self.plan.dimension):
                first_link = links[..., first, :]
                second_link = links[..., second, :]
                second_forward = jnp.roll(second_link, -1, axis=first)
                first_forward = jnp.roll(first_link, -1, axis=second)
                plaquette = _quaternion_multiply(
                    _quaternion_multiply(first_link, second_forward),
                    _quaternion_multiply(
                        _quaternion_dagger(first_forward),
                        _quaternion_dagger(second_link),
                    ),
                )
                inverse_area_squared = (
                    self.plan.spacing[first] * self.plan.spacing[second]
                ) ** -2
                energy = energy + (
                    self.plan.magnetic_coupling
                    * inverse_area_squared
                    * jnp.sum(1.0 - plaquette[..., 0])
                )
        return energy

    def energy(self, state: ClassicalYangMillsState, /) -> Array:
        electric = 0.5 * self.plan.electric_coupling * jnp.sum(state.electric_fields**2)
        return electric + self._plaquette_energy(state.links)

    def _magnetic_force(self, links: Array, /) -> Array:
        force = jnp.zeros(links.shape[:-1] + (3,), dtype=links.dtype)
        for direction in range(self.plan.dimension):
            link = links[..., direction, :]
            direction_force = jnp.zeros(link.shape[:-1] + (3,), dtype=links.dtype)
            for transverse in range(self.plan.dimension):
                if transverse == direction:
                    continue
                transverse_link = links[..., transverse, :]
                transverse_at_end = jnp.roll(transverse_link, -1, axis=direction)
                link_at_side = jnp.roll(link, -1, axis=transverse)
                forward_staple = _quaternion_multiply(
                    transverse_at_end,
                    _quaternion_multiply(
                        _quaternion_dagger(link_at_side),
                        _quaternion_dagger(transverse_link),
                    ),
                )
                transverse_back_end = jnp.roll(
                    jnp.roll(transverse_link, -1, axis=direction),
                    1,
                    axis=transverse,
                )
                link_back = jnp.roll(link, 1, axis=transverse)
                transverse_back = jnp.roll(transverse_link, 1, axis=transverse)
                backward_staple = _quaternion_multiply(
                    _quaternion_dagger(transverse_back_end),
                    _quaternion_multiply(_quaternion_dagger(link_back), transverse_back),
                )
                forward_loop = _quaternion_multiply(link, forward_staple)
                backward_loop = _quaternion_multiply(link, backward_staple)
                inverse_area_squared = (
                    self.plan.spacing[direction] * self.plan.spacing[transverse]
                ) ** -2
                direction_force = direction_force - (
                    self.plan.magnetic_coupling
                    * inverse_area_squared
                    * (forward_loop[..., 1:] + backward_loop[..., 1:])
                )
            force = force.at[..., direction, :].set(direction_force)
        return force

    def step(self, state: ClassicalYangMillsState, /) -> ClassicalYangMillsState:
        old_force = self._magnetic_force(state.links)
        electric_half = state.electric_fields + 0.5 * self.plan.time_step * old_force
        increment = _su2_exponential(
            self.plan.time_step * self.plan.electric_coupling * electric_half
        )
        links = _quaternion_multiply(increment, state.links)
        new_force = self._magnetic_force(links)
        electric = electric_half + 0.5 * self.plan.time_step * new_force
        return ClassicalYangMillsState(
            links,
            electric,
            state.step_index + 1,
            self.plan.plan_id,
        )

    def run(self, /) -> ClassicalYangMillsResult:
        initial_energy = self.energy(self.initial_state)
        initial_gauss_residual = jnp.max(jnp.abs(self.initial_gauss))
        initial_link_residual = jnp.max(
            jnp.abs(jnp.sum(self.initial_state.links**2, axis=-1) - 1.0)
        )

        def body(state, _):
            advanced = self.step(state)
            energy = self.energy(advanced)
            gauss = jnp.max(jnp.abs(yang_mills_gauss(advanced)))
            link_norm = jnp.max(jnp.abs(jnp.sum(advanced.links**2, axis=-1) - 1.0))
            return advanced, (energy, gauss, link_norm)

        final, histories = jax.lax.scan(
            body, self.initial_state, xs=None, length=self.plan.step_count
        )
        energy = jnp.concatenate((initial_energy[None], histories[0]))
        gauss = jnp.concatenate((initial_gauss_residual[None], histories[1]))
        link_norm = jnp.concatenate((initial_link_residual[None], histories[2]))
        relative_drift = jnp.max(jnp.abs(energy - energy[0])) / jnp.maximum(
            jnp.abs(energy[0]), 1.0
        )
        maximum_gauss = jnp.max(gauss)
        maximum_link = jnp.max(link_norm)
        finite = (
            jnp.all(jnp.isfinite(final.links))
            & jnp.all(jnp.isfinite(final.electric_fields))
            & jnp.all(jnp.isfinite(energy))
            & jnp.all(jnp.isfinite(gauss))
        )
        conserved = finite & (maximum_gauss <= self.plan.gauss_tolerance)
        status = jnp.where(
            conserved,
            int(NonequilibriumStatus.SUCCESS),
            jnp.where(
                finite,
                int(NonequilibriumStatus.CONSTRAINT_VIOLATION),
                int(NonequilibriumStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        diagnostics = YangMillsConservationDiagnostics(
            energy,
            gauss,
            link_norm,
            relative_drift,
            maximum_gauss,
            maximum_link,
            finite,
            conserved,
            status,
            self.plan.plan_id,
        )
        return ClassicalYangMillsResult(final, diagnostics, self.plan.plan_id)


def yang_mills_gauss(state: ClassicalYangMillsState, /) -> Array:
    if not isinstance(state, ClassicalYangMillsState):
        raise TypeError("state must be ClassicalYangMillsState.")
    dimension = state.links.shape[-2]
    residual = jnp.zeros(state.links.shape[:-2] + (3,), dtype=state.links.dtype)
    for direction in range(dimension):
        outgoing = state.electric_fields[..., direction, :]
        previous_link = jnp.roll(state.links[..., direction, :], 1, axis=direction)
        previous_electric = jnp.roll(outgoing, 1, axis=direction)
        incoming = _quaternion_rotate(
            _quaternion_dagger(previous_link), previous_electric
        )
        residual = residual + outgoing - incoming
    return residual


def gauge_transform_yang_mills(
    state: ClassicalYangMillsState, transformations: ArrayLike, /
) -> ClassicalYangMillsState:
    if not isinstance(state, ClassicalYangMillsState):
        raise TypeError("state must be ClassicalYangMillsState.")
    gauge = jnp.asarray(transformations)
    if gauge.shape != state.links.shape[:-2] + (4,):
        raise ValueError("Gauge transformations require one SU(2) quaternion per site.")
    transformed_links = jnp.zeros_like(state.links)
    transformed_electric = jnp.zeros_like(state.electric_fields)
    for direction in range(state.links.shape[-2]):
        end_gauge = jnp.roll(gauge, -1, axis=direction)
        link = _quaternion_multiply(
            _quaternion_multiply(gauge, state.links[..., direction, :]),
            _quaternion_dagger(end_gauge),
        )
        electric = _quaternion_rotate(gauge, state.electric_fields[..., direction, :])
        transformed_links = transformed_links.at[..., direction, :].set(link)
        transformed_electric = transformed_electric.at[..., direction, :].set(electric)
    return ClassicalYangMillsState(
        transformed_links,
        transformed_electric,
        state.step_index,
        state.plan_id,
    )


def yang_mills_ward_evidence(
    prepared: PreparedClassicalYangMillsEvolution,
    state: ClassicalYangMillsState,
    transformations: ArrayLike,
    /,
) -> YangMillsWardEvidence:
    if not isinstance(prepared, PreparedClassicalYangMillsEvolution):
        raise TypeError("prepared must be PreparedClassicalYangMillsEvolution.")
    gauge = jnp.asarray(transformations)
    transformed = gauge_transform_yang_mills(state, gauge)
    original_gauss = yang_mills_gauss(state)
    transformed_gauss = yang_mills_gauss(transformed)
    expected_gauss = _quaternion_rotate(gauge, original_gauss)
    energy_residual = jnp.abs(prepared.energy(transformed) - prepared.energy(state))
    covariance = jnp.max(jnp.abs(transformed_gauss - expected_gauss))
    link_norm = jnp.max(jnp.abs(jnp.sum(transformed.links**2, axis=-1) - 1.0))
    finite = (
        jnp.isfinite(energy_residual) & jnp.isfinite(covariance) & jnp.isfinite(link_norm)
    )
    scale = jnp.maximum(jnp.abs(prepared.energy(state)), 1.0)
    satisfied = (
        finite
        & (energy_residual <= prepared.plan.gauss_tolerance * scale)
        & (covariance <= prepared.plan.gauss_tolerance)
        & (link_norm <= prepared.plan.gauss_tolerance)
    )
    return YangMillsWardEvidence(
        energy_residual,
        covariance,
        link_norm,
        finite,
        satisfied,
        prepared.plan.plan_id,
    )


__all__ = [
    "ClassicalYangMillsPlan",
    "ClassicalYangMillsResult",
    "ClassicalYangMillsState",
    "PreparedClassicalYangMillsEvolution",
    "YangMillsConservationDiagnostics",
    "YangMillsWardEvidence",
    "gauge_transform_yang_mills",
    "yang_mills_gauss",
    "yang_mills_ward_evidence",
]
