#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge


class MaxwellCPMLState(StrictModule):
    """Boundary-packed directional CPML memories."""

    electric_memory: tuple[Array, ...]
    magnetic_memory: tuple[Array, ...]


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
    """Structured convolutional layer compiled by active directional term."""

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
            (int(widths),) if isinstance(widths, int) else tuple(int(v) for v in widths)
        )
        reflection, order = float(target_reflection), int(sigma_order)
        kappa, alpha = float(kappa_max), float(alpha_max)
        if not values or any(value < 0 for value in values):
            raise ValueError("CPML widths must be nonnegative.")
        if not np.isfinite(reflection) or not 0.0 < reflection < 1.0:
            raise ValueError("target_reflection must lie in (0, 1).")
        if order <= 0 or not np.isfinite(kappa) or kappa < 1.0:
            raise ValueError("CPML order/kappa are invalid.")
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError("CPML alpha_max must be finite and nonnegative.")
        self.widths, self.target_reflection = values, reflection
        self.sigma_order, self.kappa_max, self.alpha_max = order, kappa, alpha
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-cpml-plan",
                "widths": values,
                "reflection": reflection,
                "order": order,
                "kappa": kappa,
                "alpha": alpha,
            }
        )

    def prepare(
        self, bridge: StructuredCochainBridge, layout: Any, /
    ) -> PreparedMaxwellCPML:
        return PreparedMaxwellCPML(self, bridge, layout)


class PreparedMaxwellCPMLTerm(StrictModule, NonTrainableState):
    """One packed derivative-axis memory on one output cochain."""

    indices: Array
    sigma: Array
    kappa: Array
    alpha: Array
    axis: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        indices: ArrayLike,
        sigma: ArrayLike,
        kappa: ArrayLike,
        alpha: ArrayLike,
        /,
        *,
        axis: int,
        output_size: int,
        term_id: str,
    ):
        indices_ = jnp.asarray(indices, dtype=jnp.int32)
        sigma_, kappa_, alpha_ = (
            jnp.asarray(sigma),
            jnp.asarray(kappa),
            jnp.asarray(alpha),
        )
        if (
            indices_.ndim != 1
            or sigma_.shape != indices_.shape
            or kappa_.shape != indices_.shape
            or alpha_.shape != indices_.shape
        ):
            raise ValueError("Packed CPML term arrays must be aligned vectors.")
        self.indices, self.sigma, self.kappa, self.alpha = (
            indices_,
            sigma_,
            kappa_,
            alpha_,
        )
        self.axis, self.output_size, self.term_id = (
            int(axis),
            int(output_size),
            str(term_id),
        )


class MaxwellCPMLTermCoefficients(StrictModule):
    inverse_kappa_minus_one: Array
    decay: Array
    memory_coefficient: Array
    term_id: str = eqx.field(static=True)


class MaxwellCPMLCoefficients(StrictModule):
    electric: tuple[MaxwellCPMLTermCoefficients, ...]
    magnetic: tuple[MaxwellCPMLTermCoefficients, ...]
    electric_step: Array
    magnetic_step: Array
    coefficient_id: str = eqx.field(static=True)


def _term_profile(
    bridge: StructuredCochainBridge,
    degree: int,
    axis: int,
    width: int,
    plan: MaxwellCPMLPlan,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    offsets = bridge.orientation_offsets[degree]
    for orientation, shape, offset in zip(
        bridge.orientations[degree],
        bridge.orientation_shapes[degree],
        offsets,
        strict=True,
    ):
        grid = np.indices(shape, dtype=np.int64)
        coordinate = grid[axis]
        extent = shape[axis]
        low = coordinate < width
        high = coordinate >= extent - width
        mask = low | high
        if not np.any(mask):
            continue
        if axis in orientation:
            low_depth = (width - coordinate - 0.5) / max(width, 1)
            high_depth = (coordinate - (extent - width) + 0.5) / max(width, 1)
        else:
            low_depth = (width - coordinate) / max(width, 1)
            high_depth = (coordinate - (extent - 1 - width)) / max(width, 1)
        depth = np.maximum(low_depth, high_depth)
        flat = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
        indices.append(offset + flat[mask])
        depths.append(np.clip(depth[mask], 0.0, 1.0))
    if not indices:
        empty = np.zeros((0,), dtype=float)
        return np.zeros((0,), dtype=np.int32), empty, empty + 1.0, empty
    index = np.concatenate(indices).astype(np.int32)
    depth = np.concatenate(depths)
    sigma_max = -(plan.sigma_order + 1.0) * np.log(plan.target_reflection) / max(width, 1)
    powered = depth**plan.sigma_order
    return (
        index,
        sigma_max * powered,
        1.0 + (plan.kappa_max - 1.0) * powered,
        plan.alpha_max * (1.0 - depth),
    )


def _terms(
    plan: MaxwellCPMLPlan,
    bridge: StructuredCochainBridge,
    degree: int,
    widths: tuple[int, ...],
    kind: str,
    /,
) -> tuple[PreparedMaxwellCPMLTerm, ...]:
    output_size = bridge.cochain.cell_counts[degree]
    output = []
    for axis, width in enumerate(widths):
        if width == 0:
            continue
        index, sigma, kappa, alpha = _term_profile(bridge, degree, axis, width, plan)
        if index.size == 0:
            continue
        term_id = canonical_fingerprint(
            {
                "kind": "maxwell-cpml-term",
                "plan": plan.plan_id,
                "bridge": bridge.bridge_id,
                "degree": degree,
                "axis": axis,
                "field": kind,
                "indices": array_tree_fingerprint(index),
            }
        )
        output.append(
            PreparedMaxwellCPMLTerm(
                index,
                sigma,
                kappa,
                alpha,
                axis=axis,
                output_size=output_size,
                term_id=term_id,
            )
        )
    return tuple(output)


class PreparedMaxwellCPML(StrictModule):
    electric_terms: tuple[PreparedMaxwellCPMLTerm, ...]
    magnetic_terms: tuple[PreparedMaxwellCPMLTerm, ...]
    electric_size: int = eqx.field(static=True)
    magnetic_size: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    qualification: MaxwellCPMLQualification
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: MaxwellCPMLPlan, bridge: StructuredCochainBridge, layout: Any, /
    ):
        widths = plan.widths * bridge.dimension if len(plan.widths) == 1 else plan.widths
        if len(widths) != bridge.dimension:
            raise ValueError("CPML requires one width per structured axis.")
        fractions = []
        for structured_axis, width in zip(
            bridge.grid.structured_axes, widths, strict=True
        ):
            count = int(structured_axis.interval_centers.size)
            if width and structured_axis.periodic:
                raise ValueError("Periodic/Bloch axes cannot also carry CPML.")
            if 2 * width >= count:
                raise ValueError("CPML leaves no undamped interior.")
            fractions.append((count - 2 * width) / count)
        electric_terms = _terms(plan, bridge, layout.electric_degree, widths, "electric")
        magnetic_terms = _terms(plan, bridge, layout.magnetic_degree, widths, "magnetic")
        corner_axes = sum(width > 0 for width in widths)
        minimum_fraction = min(fractions)
        qualification_id = canonical_fingerprint(
            {
                "kind": "maxwell-cpml-qualification",
                "plan": plan.plan_id,
                "bridge": bridge.bridge_id,
                "layout": layout.layout_id,
                "minimum_fraction": minimum_fraction,
                "corner_axes": corner_axes,
            }
        )
        self.electric_terms, self.magnetic_terms = electric_terms, magnetic_terms
        self.electric_size, self.magnetic_size = (
            layout.electric_count,
            layout.magnetic_count,
        )
        self.dimension = bridge.dimension
        self.qualification = MaxwellCPMLQualification(
            minimum_fraction,
            plan.target_reflection,
            corner_axes,
            minimum_fraction > 0.0,
            qualification_id,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-maxwell-cpml",
                "plan": plan.plan_id,
                "bridge": bridge.bridge_id,
                "layout": layout.layout_id,
                "electric_terms": [v.term_id for v in electric_terms],
                "magnetic_terms": [v.term_id for v in magnetic_terms],
            }
        )

    @property
    def state_elements(self) -> int:
        return sum(
            term.indices.size for term in (*self.electric_terms, *self.magnetic_terms)
        )

    def initialize(self, /, *, dtype: Any = float) -> MaxwellCPMLState:
        return MaxwellCPMLState(
            tuple(
                jnp.zeros(term.indices.shape, dtype=dtype) for term in self.electric_terms
            ),
            tuple(
                jnp.zeros(term.indices.shape, dtype=dtype) for term in self.magnetic_terms
            ),
        )

    def validate_state(self, state: MaxwellCPMLState, /) -> None:
        if not isinstance(state, MaxwellCPMLState):
            raise TypeError("CPML state must be MaxwellCPMLState.")
        expected_e = tuple(term.indices.shape for term in self.electric_terms)
        expected_m = tuple(term.indices.shape for term in self.magnetic_terms)
        if (
            tuple(v.shape for v in state.electric_memory) != expected_e
            or tuple(v.shape for v in state.magnetic_memory) != expected_m
        ):
            raise ValueError("CPML memory shapes do not match packed terms.")

    @staticmethod
    def _coefficient(
        term: PreparedMaxwellCPMLTerm, step: Array, /
    ) -> MaxwellCPMLTermCoefficients:
        decay = jnp.exp(-(term.sigma / term.kappa + term.alpha) * step)
        denominator = term.sigma * term.kappa + term.alpha * term.kappa**2
        coefficient = jnp.where(
            denominator > 0.0, term.sigma * (decay - 1.0) / denominator, 0.0
        )
        return MaxwellCPMLTermCoefficients(
            1.0 / term.kappa - 1.0, decay, coefficient, term.term_id
        )

    def bind_coefficients(
        self, electric_step: ArrayLike, magnetic_step: ArrayLike, /
    ) -> MaxwellCPMLCoefficients:
        e_step, m_step = jnp.asarray(electric_step), jnp.asarray(magnetic_step)
        if e_step.shape != () or m_step.shape != ():
            raise ValueError("CPML fixed steps must be scalars.")
        electric = tuple(self._coefficient(term, e_step) for term in self.electric_terms)
        magnetic = tuple(self._coefficient(term, m_step) for term in self.magnetic_terms)
        return MaxwellCPMLCoefficients(
            electric,
            magnetic,
            e_step,
            m_step,
            canonical_fingerprint(
                {
                    "kind": "maxwell-cpml-coefficients",
                    "prepared": self.prepared_id,
                    "electric_step": float(np.asarray(e_step)),
                    "magnetic_step": float(np.asarray(m_step)),
                }
            ),
        )

    @staticmethod
    def _apply_terms(
        forcing: Array,
        memory: tuple[Array, ...],
        terms: tuple[PreparedMaxwellCPMLTerm, ...],
        step_size: Array,
        coefficients: tuple[MaxwellCPMLTermCoefficients, ...] | None,
        /,
    ) -> tuple[Array, tuple[Array, ...]]:
        value = jnp.sum(forcing, axis=0)
        updated = []
        for index, (term, old) in enumerate(zip(terms, memory, strict=True)):
            sample = forcing[term.axis, term.indices]
            fixed = None if coefficients is None else coefficients[index]
            if fixed is None:
                fixed = PreparedMaxwellCPML._coefficient(term, step_size)
            new = fixed.decay * old + fixed.memory_coefficient * sample
            correction = fixed.inverse_kappa_minus_one * sample + new
            value = value.at[term.indices].add(correction)
            updated.append(new)
        return value, tuple(updated)

    def apply_electric(
        self,
        forcing_components: Array,
        state: MaxwellCPMLState,
        step_size: Array,
        /,
        *,
        coefficients: MaxwellCPMLCoefficients | None = None,
    ) -> tuple[Array, MaxwellCPMLState]:
        self.validate_state(state)
        forcing = jnp.asarray(forcing_components)
        if forcing.shape != (self.dimension, self.electric_size):
            raise ValueError("Electric CPML forcing has the wrong directional shape.")
        fixed = None if coefficients is None else coefficients.electric
        value, memory = self._apply_terms(
            forcing,
            state.electric_memory,
            self.electric_terms,
            jnp.asarray(step_size),
            fixed,
        )
        return value, MaxwellCPMLState(memory, state.magnetic_memory)

    def apply_magnetic(
        self,
        forcing_components: Array,
        state: MaxwellCPMLState,
        step_size: Array,
        /,
        *,
        coefficients: MaxwellCPMLCoefficients | None = None,
    ) -> tuple[Array, MaxwellCPMLState]:
        self.validate_state(state)
        forcing = jnp.asarray(forcing_components)
        if forcing.shape != (self.dimension, self.magnetic_size):
            raise ValueError("Magnetic CPML forcing has the wrong directional shape.")
        fixed = None if coefficients is None else coefficients.magnetic
        value, memory = self._apply_terms(
            forcing,
            state.magnetic_memory,
            self.magnetic_terms,
            jnp.asarray(step_size),
            fixed,
        )
        return value, MaxwellCPMLState(state.electric_memory, memory)

    @staticmethod
    def _rate(
        forcing: Array,
        memory: tuple[Array, ...],
        terms: tuple[PreparedMaxwellCPMLTerm, ...],
        /,
    ) -> Array:
        value = jnp.sum(forcing, axis=0)
        for term, saved in zip(terms, memory, strict=True):
            sample = forcing[term.axis, term.indices]
            value = value.at[term.indices].add((1.0 / term.kappa - 1.0) * sample + saved)
        return value

    def electric_rate(
        self, forcing_components: Array, state: MaxwellCPMLState, /
    ) -> Array:
        self.validate_state(state)
        forcing = jnp.asarray(forcing_components)
        if forcing.shape != (self.dimension, self.electric_size):
            raise ValueError("Electric CPML rate components have the wrong shape.")
        return self._rate(forcing, state.electric_memory, self.electric_terms)

    def magnetic_rate(
        self, forcing_components: Array, state: MaxwellCPMLState, /
    ) -> Array:
        self.validate_state(state)
        forcing = jnp.asarray(forcing_components)
        if forcing.shape != (self.dimension, self.magnetic_size):
            raise ValueError("Magnetic CPML rate components have the wrong shape.")
        return self._rate(forcing, state.magnetic_memory, self.magnetic_terms)

    @staticmethod
    def _attenuation(terms: tuple[PreparedMaxwellCPMLTerm, ...], size: int, /) -> Array:
        value = jnp.zeros((size,))
        for term in terms:
            value = value.at[term.indices].add(term.sigma / term.kappa)
        return value

    def diagnostics(
        self,
        electric: ArrayLike,
        magnetic: ArrayLike,
        electric_metric: ArrayLike,
        magnetic_metric: ArrayLike,
        /,
    ) -> MaxwellCPMLDiagnostics:
        electric_, magnetic_ = jnp.asarray(electric), jnp.asarray(magnetic)
        e_metric, m_metric = jnp.asarray(electric_metric), jnp.asarray(magnetic_metric)
        if (
            e_metric.ndim != 1
            or m_metric.ndim != 1
            or e_metric.shape != electric_.shape
            or m_metric.shape != magnetic_.shape
        ):
            raise ValueError(
                "Structured CPML diagnostics require diagonal Hodge metrics."
            )
        e_attenuation = self._attenuation(self.electric_terms, self.electric_size)
        m_attenuation = self._attenuation(self.magnetic_terms, self.magnetic_size)
        absorbed = jnp.sum(
            e_metric * e_attenuation * jnp.real(electric_ * jnp.conj(electric_))
        )
        absorbed += jnp.sum(
            m_metric * m_attenuation * jnp.real(magnetic_ * jnp.conj(magnetic_))
        )
        e_max = (
            jnp.max(jnp.stack(tuple(jnp.max(term.sigma) for term in self.electric_terms)))
            if self.electric_terms
            else jnp.asarray(0.0)
        )
        m_max = (
            jnp.max(jnp.stack(tuple(jnp.max(term.sigma) for term in self.magnetic_terms)))
            if self.magnetic_terms
            else jnp.asarray(0.0)
        )
        return MaxwellCPMLDiagnostics(
            absorbed, e_max, m_max, jnp.asarray(self.qualification.target_reflection)
        )


__all__ = [
    "MaxwellCPMLCoefficients",
    "MaxwellCPMLDiagnostics",
    "MaxwellCPMLPlan",
    "MaxwellCPMLQualification",
    "MaxwellCPMLState",
    "MaxwellCPMLTermCoefficients",
    "PreparedMaxwellCPML",
    "PreparedMaxwellCPMLTerm",
]
