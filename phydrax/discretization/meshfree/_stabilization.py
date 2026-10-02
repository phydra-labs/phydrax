#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked
from .._point_cloud import PreparedPointCloudDiscretization
from .._point_cloud_pde import PointDiffusionOperator


@final
class HyperviscosityEvidence(StrictModule, NonTrainableState):
    spectral_radius_estimate: Array
    hyperviscosity_radius_estimate: Array
    power_iterations: int = eqx.field(static=True)
    scope: Literal["power-iteration-estimate"] = eqx.field(
        static=True, default="power-iteration-estimate"
    )


@final
class HyperviscosityPlan(StrictModule):
    """Explicit -coefficient (-L)^order with dissipative quadrature Laplacian.

    No time integrator is selected: semidiscrete dissipation does not certify
    unconditional stability, and the power estimate is not a spectral bound.
    """

    coefficient: float = eqx.field(static=True)
    order: int = eqx.field(static=True)
    power_iterations: int = eqx.field(static=True)

    def __init__(
        self, coefficient: float, /, *, order: int | float = 2, power_iterations: int = 12
    ) -> None:
        coefficient_ = float(coefficient)
        order_ = int(order)
        iterations = int(power_iterations)
        if not np.isfinite(coefficient_) or coefficient_ < 0:
            raise ValueError("Hyperviscosity coefficient must be finite and nonnegative.")
        if order_ != order or order_ < 1:
            raise ValueError("Hyperviscosity order must be a positive integer.")
        if iterations != power_iterations or iterations < 1:
            raise ValueError("Power iterations must be a positive integer.")
        self.coefficient = coefficient_
        self.order = order_
        self.power_iterations = iterations

    def prepare(
        self,
        discretization: PreparedPointCloudDiscretization,
        /,
    ) -> PreparedHyperviscosity:
        return PreparedHyperviscosity(self, discretization)


@final
class PreparedHyperviscosity(StrictModule, NonTrainableState):
    plan: HyperviscosityPlan
    diffusion: PointDiffusionOperator
    evidence: HyperviscosityEvidence

    @checked
    def __init__(
        self,
        plan: HyperviscosityPlan,
        discretization: PreparedPointCloudDiscretization,
        /,
    ) -> None:
        diffusion = PointDiffusionOperator(discretization, form="dissipative")
        mass = discretization.quadrature_weights
        # A deterministic, nonconstant start, normalized in the physical pairing.
        vector = jnp.sin(jnp.arange(mass.size, dtype=mass.dtype) + 0.5)
        vector = vector - jnp.sum(mass * vector) / jnp.sum(mass)
        tiny = jnp.finfo(mass.dtype).tiny

        def normalize(value: Array) -> Array:
            norm = jnp.sqrt(jnp.sum(mass * value * value))
            return value / jnp.maximum(norm, tiny)

        def step(_: int, value: Array) -> Array:
            return normalize(-diffusion.mv(value))

        vector = jax.lax.fori_loop(0, plan.power_iterations, step, normalize(vector))
        image = -diffusion.mv(vector)
        radius = jnp.maximum(jnp.sum(mass * vector * image), 0.0)
        evidence = HyperviscosityEvidence(
            radius, plan.coefficient * radius**plan.order, plan.power_iterations
        )
        self.plan = plan
        self.diffusion = diffusion
        self.evidence = evidence

    def mv(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        for _ in range(self.plan.order):
            value = -self.diffusion.mv(value)
        return -self.plan.coefficient * value

    def energy_rate(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        mass = self.diffusion.discretization.quadrature_weights.reshape(
            (value.shape[0],) + (1,) * (value.ndim - 1)
        )
        return jnp.real(jnp.vdot(value, mass * self.mv(value)))


__all__ = ["HyperviscosityPlan", "PreparedHyperviscosity", "HyperviscosityEvidence"]
