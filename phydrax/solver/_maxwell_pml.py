#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge
from ..typing import checked


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
    ) -> None:
        values = (int(widths),) if isinstance(widths, int) else tuple(widths)
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
        self, bridge: StructuredCochainBridge, layout: Any, wave_speed: ArrayLike, /
    ) -> PreparedMaxwellCPML:
        return PreparedMaxwellCPML(self, bridge, layout, wave_speed)


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
    ) -> None:
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
    """Half-step ``(1/κ − 1, b, a)`` of one term's recursive convolution."""

    inverse_kappa_minus_one: Array
    decay: Array
    memory_coefficient: Array
    term_id: str = eqx.field(static=True)


class MaxwellCPMLCoefficients(StrictModule):
    """Half-step recursion coefficients bound to one fixed leapfrog step."""

    electric: tuple[MaxwellCPMLTermCoefficients, ...]
    magnetic: tuple[MaxwellCPMLTermCoefficients, ...]
    step_size: Array
    coefficient_id: str = eqx.field(static=True)


def _layer_depth(
    coordinate: np.ndarray, cells: int, width: int, staggered: bool, /
) -> np.ndarray:
    """Normalized depth into the two ``width``-cell layers of a ``cells``-cell axis.

    ``staggered`` entries sit at cell centers ``coordinate + ½``, the others at
    nodes ``coordinate``; depth is zero on the interior side of a layer face and
    one on the outer wall.
    """
    scale = max(width, 1)
    offset = 0.5 if staggered else 0.0
    low = (width - coordinate - offset) / scale
    high = (coordinate + offset - (cells - width)) / scale
    return np.clip(np.maximum(low, high), 0.0, 1.0)


def _graded_profile(
    plan: MaxwellCPMLPlan, depth: np.ndarray, thickness: float, wave_speed: float, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Graded ``(σ, κ, α)`` of the stretching ``s = κ + σ/(α − iω)``.

    ``σ_max = (m + 1) c ln(1/R) / (2 d)`` makes the continuum normal-incidence
    round-trip reflection of a layer of physical thickness ``d`` equal to the
    target ``R`` for waves of speed ``c``. The time-domain CPML, the reduced
    Maxwell blocks, and the frequency-domain coordinate stretching share this
    profile.
    """
    sigma_max = (
        -(plan.sigma_order + 1.0)
        * wave_speed
        * np.log(plan.target_reflection)
        / (2.0 * thickness)
    )
    powered = depth**plan.sigma_order
    return (
        sigma_max * powered,
        1.0 + (plan.kappa_max - 1.0) * powered,
        plan.alpha_max * (1.0 - depth),
    )


def _recursion(
    sigma: Array, kappa: Array, alpha: Array, half_step: Array, /
) -> tuple[Array, Array]:
    """Half-step decay ``b`` and gain ``a`` of the CFS recursive convolution.

    ``ψ ← b ψ + a ∂f`` integrates ``ψ̇ = −(σ/κ + α) ψ − (σ/κ²) ∂f`` exactly over
    ``Δt/2`` for a derivative held at the sampled value.
    """
    decay = jnp.exp(-(sigma / kappa + alpha) * half_step)
    denominator = sigma * kappa + alpha * kappa**2
    gain = jnp.where(denominator > 0.0, sigma * (decay - 1.0) / denominator, 0.0)
    return decay, gain


def _stretch(
    sample: Array,
    memory: Array,
    inverse_kappa_minus_one: Array,
    decay: Array,
    gain: Array,
    /,
    *,
    before: bool,
    after: bool,
) -> tuple[Array, Array]:
    """Stretched-derivative correction of one kick and the advanced memory.

    Each memory advances by two half-step recursions per leapfrog step, and a
    kick reads it at the kick's own time level: ``before`` advances it to that
    level first, ``after`` advances it past the kick with the same sample.
    """
    kick = decay * memory + gain * sample if before else memory
    stored = decay * kick + gain * sample if after else kick
    return inverse_kappa_minus_one * sample + kick, stored


def _term_profile(
    bridge: StructuredCochainBridge,
    degree: int,
    axis: int,
    width: int,
    plan: MaxwellCPMLPlan,
    wave_speed: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Packed indices and graded ``(σ, κ, α)`` of one derivative axis."""
    indices: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    widths = np.asarray(bridge.grid.structured_axes[axis].interval_widths)
    for orientation, shape, offset in zip(
        bridge.orientations[degree],
        bridge.orientation_shapes[degree],
        bridge.orientation_offsets[degree],
        strict=True,
    ):
        coordinate = np.indices(shape, dtype=np.int64)[axis]
        depth = _layer_depth(coordinate, widths.size, width, axis in orientation)
        mask = depth > 0.0
        if not np.any(mask):
            continue
        flat = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
        indices.append(offset + flat[mask])
        depths.append(depth[mask])
    if not indices:
        empty = np.zeros((0,), dtype=np.float64)
        return np.zeros((0,), dtype=np.int32), empty, empty + 1.0, empty
    thickness = max(
        float(np.sum(widths[:width])), float(np.sum(widths[widths.size - width :]))
    )
    sigma, kappa, alpha = _graded_profile(
        plan, np.concatenate(depths), thickness, wave_speed
    )
    return np.concatenate(indices).astype(np.int32), sigma, kappa, alpha


def _terms(
    plan: MaxwellCPMLPlan,
    bridge: StructuredCochainBridge,
    degree: int,
    widths: tuple[int, ...],
    kind: str,
    wave_speed: float,
    /,
) -> tuple[PreparedMaxwellCPMLTerm, ...]:
    output_size = bridge.cochain.cell_counts[degree]
    output = []
    for axis, width in enumerate(widths):
        if width == 0:
            continue
        index, sigma, kappa, alpha = _term_profile(
            bridge, degree, axis, width, plan, wave_speed
        )
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
                "wave_speed": wave_speed,
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
    """Prepared CFS-CPML: stretched derivatives ``∂/κ + ψ`` with convolution memories.

    Each memory ``ψ`` of ``ψ̇ = −(σ/κ + α) ψ − (σ/κ²) ∂f`` is stored at the
    integer step ``t_n`` and advances by two exponential half-step recursions
    per leapfrog step, each over one sampled derivative: ``∂H_{n+½}`` twice for
    the electric memory, ``∂E_n`` then ``∂E_{n+1}`` for the magnetic memory.
    Every kick reads its memory at the kick's own time level (``t_n``,
    ``t_{n+½}``, ``t_{n+1}``), so the update is a second-order discretization
    of the stretched Maxwell system whose ``−dW/dt`` is ``electric_rate`` and
    ``magnetic_rate`` evaluated on the stored ``t_n`` memories.
    """

    electric_terms: tuple[PreparedMaxwellCPMLTerm, ...]
    magnetic_terms: tuple[PreparedMaxwellCPMLTerm, ...]
    electric_size: int = eqx.field(static=True)
    magnetic_size: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    qualification: MaxwellCPMLQualification
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MaxwellCPMLPlan,
        bridge: StructuredCochainBridge,
        layout: Any,
        wave_speed: ArrayLike,
        /,
    ) -> None:
        speed = float(np.asarray(wave_speed))
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("CPML reference wave_speed must be finite and positive.")
        widths = plan.widths * bridge.dimension if len(plan.widths) == 1 else plan.widths
        if len(widths) != bridge.dimension:
            raise ValueError("CPML requires one width per structured axis.")
        fractions = []
        for structured_axis, width in zip(
            bridge.grid.structured_axes, widths, strict=True
        ):
            count = structured_axis.interval_centers.size
            if width and structured_axis.periodic:
                raise ValueError("Periodic/Bloch axes cannot also carry CPML.")
            if 2 * width >= count:
                raise ValueError("CPML leaves no undamped interior.")
            fractions.append((count - 2 * width) / count)
        electric_terms = _terms(
            plan, bridge, layout.electric_degree, widths, "electric", speed
        )
        magnetic_terms = _terms(
            plan, bridge, layout.magnetic_degree, widths, "magnetic", speed
        )
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

    def initialize(self, /, *, dtype: Any = jnp.float64) -> MaxwellCPMLState:
        return MaxwellCPMLState(
            tuple(
                jnp.zeros(term.indices.shape, dtype=dtype) for term in self.electric_terms
            ),
            tuple(
                jnp.zeros(term.indices.shape, dtype=dtype) for term in self.magnetic_terms
            ),
        )

    @checked
    def validate_state(self, state: MaxwellCPMLState, /) -> None:
        expected_e = tuple(term.indices.shape for term in self.electric_terms)
        expected_m = tuple(term.indices.shape for term in self.magnetic_terms)
        if (
            tuple(v.shape for v in state.electric_memory) != expected_e
            or tuple(v.shape for v in state.magnetic_memory) != expected_m
        ):
            raise ValueError("CPML memory shapes do not match packed terms.")

    @staticmethod
    def _coefficient(
        term: PreparedMaxwellCPMLTerm, half_step: Array, /
    ) -> MaxwellCPMLTermCoefficients:
        decay, gain = _recursion(term.sigma, term.kappa, term.alpha, half_step)
        return MaxwellCPMLTermCoefficients(
            1.0 / term.kappa - 1.0, decay, gain, term.term_id
        )

    def bind_coefficients(self, step_size: ArrayLike, /) -> MaxwellCPMLCoefficients:
        """Bind the half-step recursion of one fixed leapfrog step ``Δt``."""
        step = jnp.asarray(step_size)
        if step.shape != ():
            raise ValueError("The CPML fixed step must be a scalar.")
        half = 0.5 * step
        return MaxwellCPMLCoefficients(
            tuple(self._coefficient(term, half) for term in self.electric_terms),
            tuple(self._coefficient(term, half) for term in self.magnetic_terms),
            step,
            canonical_fingerprint(
                {
                    "kind": "maxwell-cpml-coefficients",
                    "prepared": self.prepared_id,
                    "step_size": float(np.asarray(step)),
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
        *,
        before: bool,
        after: bool,
    ) -> tuple[Array, tuple[Array, ...]]:
        value = jnp.sum(forcing, axis=0)
        updated = []
        for index, (term, old) in enumerate(zip(terms, memory, strict=True)):
            sample = forcing[term.axis, term.indices]
            fixed = (
                PreparedMaxwellCPML._coefficient(term, 0.5 * step_size)
                if coefficients is None
                else coefficients[index]
            )
            correction, new = _stretch(
                sample,
                old,
                fixed.inverse_kappa_minus_one,
                fixed.decay,
                fixed.memory_coefficient,
                before=before,
                after=after,
            )
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
        """Stretched curl of the ``t_{n+½}`` electric kick of a ``Δt`` step.

        The kick reads the memory advanced half a step to ``t_{n+½}``; the
        stored memory advances the second half with the same ``∂H_{n+½}``.
        """
        self.validate_state(state)
        forcing = jnp.asarray(forcing_components)
        if forcing.shape != (self.dimension, self.electric_size):
            raise ValueError("Electric CPML forcing has the wrong directional shape.")
        value, memory = self._apply_terms(
            forcing,
            state.electric_memory,
            self.electric_terms,
            jnp.asarray(step_size),
            None if coefficients is None else coefficients.electric,
            before=True,
            after=True,
        )
        return value, MaxwellCPMLState(memory, state.magnetic_memory)

    def _magnetic(
        self,
        forcing_components: Array,
        state: MaxwellCPMLState,
        step_size: Array,
        coefficients: MaxwellCPMLCoefficients | None,
        /,
        *,
        before: bool,
        after: bool,
    ) -> tuple[Array, MaxwellCPMLState]:
        self.validate_state(state)
        forcing = jnp.asarray(forcing_components)
        if forcing.shape != (self.dimension, self.magnetic_size):
            raise ValueError("Magnetic CPML forcing has the wrong directional shape.")
        value, memory = self._apply_terms(
            forcing,
            state.magnetic_memory,
            self.magnetic_terms,
            jnp.asarray(step_size),
            None if coefficients is None else coefficients.magnetic,
            before=before,
            after=after,
        )
        return value, MaxwellCPMLState(state.electric_memory, memory)

    def apply_magnetic_start(
        self,
        forcing_components: Array,
        state: MaxwellCPMLState,
        step_size: Array,
        /,
        *,
        coefficients: MaxwellCPMLCoefficients | None = None,
    ) -> tuple[Array, MaxwellCPMLState]:
        """Stretched curl of the opening ``t_n`` magnetic half kick of a ``Δt`` step.

        The kick reads the stored ``t_n`` memory; the memory then advances half a
        step with ``∂E_n``.
        """
        return self._magnetic(
            forcing_components,
            state,
            step_size,
            coefficients,
            before=False,
            after=True,
        )

    def apply_magnetic_end(
        self,
        forcing_components: Array,
        state: MaxwellCPMLState,
        step_size: Array,
        /,
        *,
        coefficients: MaxwellCPMLCoefficients | None = None,
    ) -> tuple[Array, MaxwellCPMLState]:
        """Stretched curl of the closing ``t_{n+1}`` magnetic half kick.

        The memory first advances the second half step with ``∂E_{n+1}``; the
        kick reads that stored ``t_{n+1}`` memory.
        """
        return self._magnetic(
            forcing_components,
            state,
            step_size,
            coefficients,
            before=True,
            after=False,
        )

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
