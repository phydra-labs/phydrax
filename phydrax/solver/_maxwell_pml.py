#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge


class MaxwellCPMLState(StrictModule):
    electric_memory: Array
    magnetic_memory: Array


class MaxwellCPMLDiagnostics(StrictModule):
    absorbed_power: Array
    maximum_electric_sigma: Array
    maximum_magnetic_sigma: Array
    target_reflection: Array


class MaxwellCPMLQualification(StrictModule, NonTrainableState):
    minimum_undamped_fraction: float = eqx.field(static=True)
    target_reflection: float = eqx.field(static=True)
    corner_axes: int = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)


class MaxwellCPMLPlan(StrictModule, NonTrainableState):
    """Structured convolutional absorbing layer with cochain-aligned profiles."""

    widths: tuple[int, ...] = eqx.field(static=True)
    target_reflection: float = eqx.field(static=True)
    sigma_order: int = eqx.field(static=True)
    kappa_max: float = eqx.field(static=True)
    alpha_max: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        widths: int | Sequence[int],
        /,
        *,
        target_reflection: float = 1e-6,
        sigma_order: int = 3,
        kappa_max: float = 5.0,
        alpha_max: float = 0.05,
    ):
        values = (
            (int(widths),)
            if isinstance(widths, int)
            else tuple(int(value) for value in widths)
        )
        reflection = float(target_reflection)
        order = int(sigma_order)
        kappa = float(kappa_max)
        alpha = float(alpha_max)
        if not values or any(value < 0 for value in values):
            raise ValueError("CPML widths must be nonnegative.")
        if not np.isfinite(reflection) or reflection <= 0.0 or reflection >= 1.0:
            raise ValueError("target_reflection must lie in (0, 1).")
        if order <= 0 or not np.isfinite(kappa) or kappa < 1.0:
            raise ValueError("CPML order/kappa are invalid.")
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError("CPML alpha_max must be finite and nonnegative.")
        self.widths = values
        self.target_reflection = reflection
        self.sigma_order = order
        self.kappa_max = kappa
        self.alpha_max = alpha
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-cpml-plan",
                "widths": list(values),
                "target_reflection": reflection,
                "sigma_order": order,
                "kappa_max": kappa,
                "alpha_max": alpha,
            }
        )

    def prepare(self, bridge: StructuredCochainBridge, /) -> PreparedMaxwellCPML:
        return PreparedMaxwellCPML(self, bridge)


def _profiles(
    coordinates: Array,
    bridge: StructuredCochainBridge,
    widths: tuple[int, ...],
    target_reflection: float,
    order: int,
    kappa_max: float,
    alpha_max: float,
    /,
) -> tuple[Array, Array, Array, Array]:
    sigma_axes = []
    kappa_axes = []
    alpha_axes = []
    depth_axes = []
    for axis, (structured_axis, width) in enumerate(
        zip(bridge.grid.structured_axes, widths, strict=True)
    ):
        if width == 0:
            depth = jnp.zeros(coordinates.shape[0])
        else:
            spacing = jnp.min(structured_axis.interval_widths)
            thickness = width * spacing
            lower = structured_axis.bounds[0]
            upper = structured_axis.bounds[1]
            lower_depth = jnp.clip(
                (lower + thickness - coordinates[:, axis]) / thickness, 0.0, 1.0
            )
            upper_depth = jnp.clip(
                (coordinates[:, axis] - (upper - thickness)) / thickness, 0.0, 1.0
            )
            depth = jnp.maximum(lower_depth, upper_depth)
        sigma_max = -(order + 1.0) * jnp.log(target_reflection) / max(width, 1)
        sigma_axes.append(sigma_max * depth**order)
        kappa_axes.append(1.0 + (kappa_max - 1.0) * depth**order)
        alpha_axes.append(alpha_max * (1.0 - depth) * (depth > 0.0))
        depth_axes.append(depth)
    sigma_stack = jnp.stack(tuple(sigma_axes), axis=0)
    kappa_stack = jnp.stack(tuple(kappa_axes), axis=0)
    alpha_stack = jnp.stack(tuple(alpha_axes), axis=0)
    depth_stack = jnp.stack(tuple(depth_axes), axis=0)
    corner_order = jnp.sum(depth_stack > 0.0, axis=0)
    return sigma_stack, kappa_stack, alpha_stack, corner_order


class PreparedMaxwellCPML(StrictModule):
    electric_sigma: Array
    electric_kappa: Array
    electric_alpha: Array
    magnetic_sigma: Array
    magnetic_kappa: Array
    magnetic_alpha: Array
    qualification: MaxwellCPMLQualification
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MaxwellCPMLPlan, bridge: StructuredCochainBridge, /):
        widths = plan.widths * bridge.dimension if len(plan.widths) == 1 else plan.widths
        if len(widths) != bridge.dimension:
            raise ValueError("CPML requires one width per structured axis.")
        fractions = []
        for axis, (structured_axis, width) in enumerate(
            zip(bridge.grid.structured_axes, widths, strict=True)
        ):
            if width and structured_axis.periodic:
                raise ValueError("Periodic/Bloch axes cannot also carry CPML.")
            count = int(structured_axis.interval_centers.size)
            if 2 * width >= count:
                raise ValueError("CPML leaves no undamped interior.")
            fractions.append((count - 2 * width) / count)
        electric_coordinates = bridge.cochain.coordinates[1]
        magnetic_coordinates = bridge.cochain.coordinates[2]
        if electric_coordinates is None or magnetic_coordinates is None:
            raise RuntimeError("CPML requires cochain coordinates.")
        e_sigma, e_kappa, e_alpha, e_corner = _profiles(
            electric_coordinates,
            bridge,
            widths,
            plan.target_reflection,
            plan.sigma_order,
            plan.kappa_max,
            plan.alpha_max,
        )
        m_sigma, m_kappa, m_alpha, m_corner = _profiles(
            magnetic_coordinates,
            bridge,
            widths,
            plan.target_reflection,
            plan.sigma_order,
            plan.kappa_max,
            plan.alpha_max,
        )
        corner_axes = int(
            max(
                np.max(np.asarray(e_corner), initial=0),
                np.max(np.asarray(m_corner), initial=0),
            )
        )
        minimum_fraction = min(fractions)
        qualification_id = canonical_fingerprint(
            {
                "kind": "maxwell-cpml-qualification",
                "plan": plan.plan_id,
                "bridge": bridge.bridge_id,
                "minimum_undamped_fraction": minimum_fraction,
                "corner_axes": corner_axes,
            }
        )
        self.electric_sigma = e_sigma
        self.electric_kappa = e_kappa
        self.electric_alpha = e_alpha
        self.magnetic_sigma = m_sigma
        self.magnetic_kappa = m_kappa
        self.magnetic_alpha = m_alpha
        self.qualification = MaxwellCPMLQualification(
            minimum_undamped_fraction=minimum_fraction,
            target_reflection=plan.target_reflection,
            corner_axes=corner_axes,
            passed=minimum_fraction > 0.0,
            qualification_id=qualification_id,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-maxwell-cpml",
                "plan": plan.plan_id,
                "bridge": bridge.bridge_id,
                "electric_sigma": array_tree_fingerprint(e_sigma),
                "magnetic_sigma": array_tree_fingerprint(m_sigma),
            }
        )

    def initialize(self, /) -> MaxwellCPMLState:
        return MaxwellCPMLState(
            electric_memory=jnp.zeros_like(self.electric_sigma),
            magnetic_memory=jnp.zeros_like(self.magnetic_sigma),
        )

    @staticmethod
    def _validate_state(
        state: MaxwellCPMLState,
        electric_shape: tuple[int, ...],
        magnetic_shape: tuple[int, ...],
        /,
    ) -> None:
        if not isinstance(state, MaxwellCPMLState):
            raise TypeError("CPML state must be MaxwellCPMLState.")
        if (
            state.electric_memory.shape != electric_shape
            or state.magnetic_memory.shape != magnetic_shape
        ):
            raise ValueError("CPML memory shapes do not match the prepared profiles.")

    @staticmethod
    def _apply(
        forcing: Array,
        memory: Array,
        sigma: Array,
        kappa: Array,
        alpha: Array,
        step_size: Array,
        /,
    ) -> tuple[Array, Array]:
        decay = jnp.exp(-(sigma / kappa + alpha) * step_size)
        denominator = sigma * kappa + alpha * kappa**2
        coefficient = jnp.where(
            denominator > 0.0,
            sigma * (decay - 1.0) / denominator,
            0.0,
        )
        memory_new = decay * memory + coefficient * forcing
        return forcing / kappa + memory_new, memory_new

    def apply_electric(
        self,
        forcing_components: Array,
        state: MaxwellCPMLState,
        step_size: Array,
        /,
    ) -> tuple[Array, MaxwellCPMLState]:
        self._validate_state(
            state,
            self.electric_sigma.shape,
            self.magnetic_sigma.shape,
        )
        forcing = jnp.asarray(forcing_components)
        if forcing.shape != self.electric_sigma.shape:
            raise ValueError(
                "Electric CPML forcing must have one component per axis and cochain."
            )
        value, memory = self._apply(
            forcing,
            state.electric_memory,
            self.electric_sigma,
            self.electric_kappa,
            self.electric_alpha,
            step_size,
        )
        return jnp.sum(value, axis=0), MaxwellCPMLState(
            memory,
            state.magnetic_memory,
        )

    def apply_magnetic(
        self,
        forcing_components: Array,
        state: MaxwellCPMLState,
        step_size: Array,
        /,
    ) -> tuple[Array, MaxwellCPMLState]:
        self._validate_state(
            state,
            self.electric_sigma.shape,
            self.magnetic_sigma.shape,
        )
        forcing = jnp.asarray(forcing_components)
        if forcing.shape != self.magnetic_sigma.shape:
            raise ValueError(
                "Magnetic CPML forcing must have one component per axis and cochain."
            )
        value, memory = self._apply(
            forcing,
            state.magnetic_memory,
            self.magnetic_sigma,
            self.magnetic_kappa,
            self.magnetic_alpha,
            step_size,
        )
        return jnp.sum(value, axis=0), MaxwellCPMLState(
            state.electric_memory,
            memory,
        )

    def electric_rate(
        self,
        forcing_components: Array,
        state: MaxwellCPMLState,
        /,
    ) -> Array:
        forcing = jnp.asarray(forcing_components)
        self._validate_state(
            state,
            self.electric_sigma.shape,
            self.magnetic_sigma.shape,
        )
        if forcing.shape != self.electric_sigma.shape:
            raise ValueError("Electric CPML rate components have the wrong shape.")
        return jnp.sum(
            forcing / self.electric_kappa + state.electric_memory,
            axis=0,
        )

    def magnetic_rate(
        self,
        forcing_components: Array,
        state: MaxwellCPMLState,
        /,
    ) -> Array:
        forcing = jnp.asarray(forcing_components)
        self._validate_state(
            state,
            self.electric_sigma.shape,
            self.magnetic_sigma.shape,
        )
        if forcing.shape != self.magnetic_sigma.shape:
            raise ValueError("Magnetic CPML rate components have the wrong shape.")
        return jnp.sum(
            forcing / self.magnetic_kappa + state.magnetic_memory,
            axis=0,
        )

    def diagnostics(
        self,
        electric: ArrayLike,
        magnetic: ArrayLike,
        electric_metric: ArrayLike,
        magnetic_metric: ArrayLike,
        /,
    ) -> MaxwellCPMLDiagnostics:
        electric_ = jnp.asarray(electric)
        magnetic_ = jnp.asarray(magnetic)
        electric_weight = jnp.asarray(electric_metric)
        magnetic_weight = jnp.asarray(magnetic_metric)
        if (
            electric_weight.ndim != 1
            or magnetic_weight.ndim != 1
            or electric_weight.shape != electric_.shape
            or magnetic_weight.shape != magnetic_.shape
        ):
            raise ValueError(
                "Structured CPML diagnostics require diagonal Hodge metrics."
            )
        electric_attenuation = jnp.sum(
            self.electric_sigma / self.electric_kappa,
            axis=0,
        )
        magnetic_attenuation = jnp.sum(
            self.magnetic_sigma / self.magnetic_kappa,
            axis=0,
        )
        absorbed = jnp.sum(
            electric_weight
            * electric_attenuation
            * jnp.real(electric_ * jnp.conj(electric_))
        ) + jnp.sum(
            magnetic_weight
            * magnetic_attenuation
            * jnp.real(magnetic_ * jnp.conj(magnetic_))
        )
        return MaxwellCPMLDiagnostics(
            absorbed_power=absorbed,
            maximum_electric_sigma=jnp.max(self.electric_sigma),
            maximum_magnetic_sigma=jnp.max(self.magnetic_sigma),
            target_reflection=jnp.asarray(self.qualification.target_reflection),
        )


__all__ = [
    "MaxwellCPMLDiagnostics",
    "MaxwellCPMLPlan",
    "MaxwellCPMLQualification",
    "MaxwellCPMLState",
    "PreparedMaxwellCPML",
]
