#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gas-diffusive coarsening of dry foams driven by auction-derived cell pressures.

Every cell is a label with an exact integer volume and incompressible gas; its
pressure is the dual price of the volume-constrained assignment. With potentials
``psi_i = sum_j K_ij * u_j`` normalized so that ``E -> sum sigma_ij |Gamma_ij|``,
the interface condition ``psi_i + p_i = psi_j + p_j`` at a film of curvature
``kappa`` gives ``p_j - p_i = sigma_ij kappa / sqrt(pi)`` for films at rest (first
order in ``sqrt(tau) kappa``), so the physical pressure is ``P_i = -sqrt(pi) p_i``
up to one additive gauge. The qualification tool validates this normalization
against the Laplace law on circles and spheres; prices are determined only within
the assignment's dual interval, so single-step pressures carry a discretization
spread of a few percent when ``tau kappa`` exceeds the grid spacing and degrade
when interfaces pin.

Gas crosses film ``ij`` at rate ``k A_ij (P_i - P_j)`` with permeance ``k`` and
film area ``A_ij``, measured isotropically by the kernel estimate
``A_ij = (sqrt(pi) / sqrt(tau)) m sum_x u_i G(tau) u_j``. Each step prescribes the
next integer volumes ``V + dt dV/dt`` (largest-remainder rounding conserves the
total exactly) and advances the films by one volume-constrained threshold step.
Films move with normal velocity ``mu (sigma kappa - [P])``, so film mobility and
gas permeance act in series with effective coefficient ``k_eff = k mu / (k + mu)``:
an isolated circular cell loses area at ``2 pi sigma k_eff`` and isotropic 2D dry
foams follow von Neumann's law ``dA/dt = (pi sigma k_eff / 3)(n - 6)``; the
quasi-static foam limit is ``mu >> k``.
"""

from __future__ import annotations

from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState, parameter_field, ParameterOwner
from .._validation import finite_real_scalar, positive_integer
from ..typing import Float64, Scalar
from ._contracts import (
    _energy,
    AbstractThresholdHeatKernel,
    LabelFieldState,
    ThresholdDynamicsEvidence,
    ThresholdKernelDecomposition,
    ThresholdPotentials,
)
from ._step import PreparedThresholdDynamics


_CoarseningCarry: TypeAlias = tuple[
    LabelFieldState,
    ThresholdPotentials,
    Array,
    Array,
    "GasDiffusionEvidence",
]


class GasDiffusionEvidence(StrictModule, NonTrainableState):
    """Pressures, film areas, prescribed volumes and the film step of one step.

    ``pressures`` use the gauge of zero mean over active cells; ``film_areas`` is
    the total film area of every cell and ``volume_rates`` its ``dV/dt``.
    """

    pressures: Array
    film_areas: Array
    volume_rates: Array
    target_counts: Array
    step: ThresholdDynamicsEvidence


class GasDiffusionRunResult(StrictModule):
    """Final foam state and per-step coarsening evidence (leading step axis)."""

    state: LabelFieldState
    initial_energy: Array
    initial_pressures: Array
    evidence: GasDiffusionEvidence


def _conserving_round(targets: Array, active: Array, total: int, /) -> Array:
    """Nonnegative integer counts summing to ``total`` (largest remainder)."""
    positive = jnp.where(active, jnp.maximum(targets, 0.0), 0.0)
    scaled = positive * (total / jnp.sum(positive))
    base = jnp.floor(scaled)
    missing = total - jnp.sum(base).astype(jnp.int32)
    order = jnp.argsort(-(scaled - base), stable=True)
    rank = jnp.zeros_like(order).at[order].set(jnp.arange(order.shape[0]))
    return (base + (rank < missing)).astype(jnp.int32)


class GasDiffusionCoarsening(StrictModule, ParameterOwner):
    """Dry-foam coarsening by inter-cell gas diffusion on a dense equal-measure route.

    Requires a prepared system with an exact `LabelVolumeConstraint` (its counts
    are the initial cell volumes) on a dense heat-kernel route with equal site
    measure. ``permeance`` is an inferable parameter leaf.
    """

    __strict_contract__ = True

    prepared: PreparedThresholdDynamics
    permeance: Float64[Scalar] = parameter_field()
    coarsening_id: str = eqx.field(static=True)

    def __init__(self, prepared: PreparedThresholdDynamics, permeance: float, /) -> None:
        if not isinstance(prepared, PreparedThresholdDynamics):
            raise TypeError("prepared must be a PreparedThresholdDynamics.")
        constraint = prepared.plan.volume_constraint
        if constraint is None or not constraint.exact:
            raise ValueError("Gas-diffusive coarsening needs exact cell volumes.")
        if not isinstance(prepared.route, AbstractThresholdHeatKernel):
            raise ValueError("Gas-diffusive coarsening needs a dense heat-kernel route.")
        value = finite_real_scalar(permeance, "permeance")
        if value < 0.0:
            raise ValueError("permeance must be nonnegative.")
        self.prepared = prepared
        self.permeance = jnp.asarray(value, dtype=jnp.float64)
        self.coarsening_id = canonical_fingerprint(
            {"kind": "gas-diffusion-coarsening", "prepared": prepared.prepared_id}
        )

    def film_areas(
        self, state: LabelFieldState, decomposition: ThresholdKernelDecomposition, /
    ) -> Array:
        """Kernel estimate of the ``L x L`` film areas (zero diagonal)."""
        state = self.prepared._validate_state(state)
        return self._film_areas(state, decomposition)

    def _film_areas(
        self, state: LabelFieldState, decomposition: ThresholdKernelDecomposition, /
    ) -> Array:
        prepared = self.prepared
        route = prepared.route
        if not isinstance(route, AbstractThresholdHeatKernel):
            raise RuntimeError("Coarsening lost its dense heat-kernel route.")
        count = prepared.plan.label_count
        dtype = prepared.dtype
        fields = jax.nn.one_hot(state.labels, count, dtype=dtype)
        tau = decomposition.times[0].astype(dtype)
        smoothed, _ = route.smooth(fields, tau)
        flat = fields.reshape((prepared.site_count, count))
        overlap = flat.T @ (
            smoothed.reshape(flat.shape) * prepared.site_measures[:, None]
        )
        areas = jnp.sqrt(jnp.pi) / jnp.sqrt(tau) * overlap
        identity = jnp.eye(count, dtype=jnp.bool_)
        return jnp.where(identity, 0.0, 0.5 * (areas + areas.T))

    def pressures(self, prices: Array, active_labels: Array, /) -> Array:
        """Physical pressures ``-sqrt(pi) p`` with zero mean over active cells."""
        raw = -jnp.sqrt(jnp.pi) * prices
        mean = jnp.sum(jnp.where(active_labels, raw, 0.0)) / jnp.sum(active_labels)
        return jnp.where(active_labels, raw - mean, 0.0)

    def initial_pressures(self, state: LabelFieldState, /) -> Array:
        """Pressures of the current cells from one volume-preserving assignment."""
        state = self.prepared._validate_state(state)
        prepared = self.prepared
        decomposition = prepared.plan.decomposition()
        potentials = prepared._potentials(state, decomposition)
        return self._initial_pressures(state, potentials)

    def _initial_pressures(
        self, state: LabelFieldState, potentials: ThresholdPotentials, /
    ) -> Array:
        prepared = self.prepared
        counts = state.label_counts()
        prices = jnp.zeros((prepared.plan.label_count,), prepared.dtype)
        _, volume, _, _ = prepared._assign(
            state, potentials, prices, counts, (counts, counts)
        )
        if volume is None:
            raise RuntimeError("Coarsening needs the volume-constrained assignment.")
        return self.pressures(volume.prices, state.active_labels)

    def run(self, state: LabelFieldState, steps: int, /) -> GasDiffusionRunResult:
        """``steps`` coarsening steps; the first non-committed film step halts."""
        state = self.prepared._validate_state(state)
        count = positive_integer(steps, "steps")
        prepared = self.prepared
        decomposition = prepared.plan.decomposition()
        potentials = prepared._potentials(state, decomposition)
        initial_energy = _energy(potentials.own, prepared.site_measures)
        pressures = self._initial_pressures(state, potentials)
        total = prepared.site_count
        measure = prepared.site_measures[0]

        def advance(
            current: LabelFieldState, bundle: ThresholdPotentials, pressure: Array
        ) -> tuple[
            LabelFieldState, ThresholdPotentials, Array, Array, GasDiffusionEvidence
        ]:
            areas = self._film_areas(current, decomposition)
            rates = -self.permeance * (
                pressure * jnp.sum(areas, axis=1) - areas @ pressure
            ).astype(prepared.dtype)
            counts = current.label_counts().astype(prepared.dtype)
            targets = _conserving_round(
                counts + prepared.plan.time_step * rates / measure,
                current.active_labels,
                total,
            )
            prices = -pressure / jnp.sqrt(jnp.pi)
            result, bundle_next, prices_next = prepared._advance(
                current, bundle, decomposition, prices, (targets, targets)
            )
            if prices_next is None:
                raise RuntimeError("Coarsening needs auction prices.")
            evidence = GasDiffusionEvidence(
                pressures=pressure,
                film_areas=jnp.sum(areas, axis=1),
                volume_rates=rates,
                target_counts=targets,
                step=result.evidence,
            )
            pressure_next = self.pressures(prices_next, result.state.active_labels)
            return result.state, bundle_next, pressure_next, ~result.committed, evidence

        template = jax.eval_shape(advance, state, potentials, pressures)[-1]
        blank = jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), template)

        def body(
            carry: _CoarseningCarry, _: None
        ) -> tuple[_CoarseningCarry, GasDiffusionEvidence]:
            current, bundle, pressure, halted, last = carry

            def proceed() -> _CoarseningCarry:
                return advance(current, bundle, pressure)

            def hold() -> _CoarseningCarry:
                frozen = eqx.tree_at(
                    lambda evidence: evidence.step.committed,
                    last,
                    jnp.zeros_like(last.step.committed),
                )
                return current, bundle, pressure, halted, frozen

            updated = jax.lax.cond(halted, hold, proceed)
            return updated, updated[-1]

        initial: _CoarseningCarry = (
            state,
            potentials,
            pressures,
            jnp.asarray(False),
            blank,
        )
        final, evidence = jax.lax.scan(body, initial, None, length=count)
        return GasDiffusionRunResult(
            state=final[0],
            initial_energy=initial_energy,
            initial_pressures=pressures,
            evidence=evidence,
        )


__all__ = ["GasDiffusionCoarsening", "GasDiffusionEvidence", "GasDiffusionRunResult"]
