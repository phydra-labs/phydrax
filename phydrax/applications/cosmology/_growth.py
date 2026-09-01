#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import DifferentialProblem, solve_diffrax
from ._background import FLRWBackground
from ._products import (
    CosmologyProductProvenance,
    ExpansionHistory,
    LagrangianGrowthHistory,
)


def _growth_drift(log_scale: Array, state: Array, background: FLRWBackground) -> Array:
    scale = jnp.exp(log_scale)
    matter_source = 1.5 * background.matter_fraction(scale)
    drag = 2.0 + background.dlog_hubble_dlog_scale(scale)
    first, first_rate, second, second_rate = state
    return jnp.stack(
        (
            first_rate,
            -drag * first_rate + matter_source * first,
            second_rate,
            -drag * second_rate + matter_source * second + matter_source * first**2,
        )
    )


class FLRWGrowthPlan(StrictModule, NonTrainableState):
    """Flat, smooth-dark-energy Newtonian first/second Lagrangian growth."""

    scale_factors: Array
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    matter_era_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale_factors: ArrayLike,
        /,
        *,
        relative_tolerance: float = 1.0e-8,
        absolute_tolerance: float = 1.0e-10,
        maximum_steps: int = 4096,
        matter_era_tolerance: float = 5.0e-2,
    ):
        nodes_host = np.asarray(scale_factors, dtype=float).reshape((-1,))
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        steps = int(maximum_steps)
        era_tolerance = float(matter_era_tolerance)
        if (
            nodes_host.size < 2
            or np.any(~np.isfinite(nodes_host))
            or np.any(nodes_host <= 0.0)
            or np.any(np.diff(nodes_host) <= 0.0)
            or abs(nodes_host[-1] - 1.0) > 1.0e-12
        ):
            raise ValueError(
                "FLRW growth nodes must be finite, positive, increasing, and end at a=1."
            )
        if (
            not np.isfinite(relative)
            or relative <= 0.0
            or not np.isfinite(absolute)
            or absolute <= 0.0
            or steps <= 0
            or not np.isfinite(era_tolerance)
            or era_tolerance <= 0.0
        ):
            raise ValueError("FLRW growth numerical policy is invalid.")
        self.scale_factors = jnp.asarray(nodes_host)
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.maximum_steps = steps
        self.matter_era_tolerance = era_tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "flat-smooth-de-flrw-lagrangian-growth",
                "scale_factors": nodes_host.tolist(),
                "relative_tolerance": relative,
                "absolute_tolerance": absolute,
                "maximum_steps": steps,
                "matter_era_tolerance": era_tolerance,
            }
        )

    def _provenance(self, background: FLRWBackground, /) -> CosmologyProductProvenance:
        return CosmologyProductProvenance(
            producer="phydrax.applications.cosmology.FLRWGrowthPlan",
            producer_version="native",
            model_form_id=background.model_form_id,
            request_id="native-dynamic-background",
            numerical_policy_id=self.plan_id,
            physics_policy_id="flat-smooth-dark-energy-newtonian-growth",
            scale_id=background.scale.scale_id,
            source_kind="native",
            differentiation="native-parameter",
        )

    def expansion_history(self, background: FLRWBackground, /) -> ExpansionHistory:
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        return ExpansionHistory(
            self.scale_factors,
            background.hubble(self.scale_factors),
            background.scale,
            self._provenance(background),
            background.realization,
        )

    def solve(self, background: FLRWBackground, /) -> LagrangianGrowthHistory:
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        start = self.scale_factors[0].astype(background.hubble_constant.dtype)
        start = background.require_flat(start)
        matter = background.matter_fraction(start)
        contamination = background.radiation_fraction(
            start
        ) + background.dark_energy_fraction(start)
        start = eqx.error_if(
            start,
            (matter <= 0.0)
            | (contamination > self.matter_era_tolerance * matter)
            | (background.dark_energy_w0 + background.dark_energy_wa >= 0.0),
            "FLRW growth requires a flat, matter-era start and vanishing early CPL dark energy.",
        )
        initial = jnp.stack(
            (
                start,
                start,
                (3.0 / 7.0) * start**2,
                (6.0 / 7.0) * start**2,
            )
        )
        log_nodes = jnp.log(self.scale_factors).astype(initial.dtype)
        problem = DifferentialProblem(
            _growth_drift,
            initial,
            t0=log_nodes[0],
            t1=log_nodes[-1],
            args=background,
            problem_id=f"flrw-lagrangian-growth:{self.plan_id}",
        )
        solution = solve_diffrax(
            problem,
            save_times=log_nodes,
            rtol=self.relative_tolerance,
            atol=self.absolute_tolerance,
            max_steps=self.maximum_steps,
            throw=False,
            solver_configuration_id=self.plan_id,
        )
        states = eqx.error_if(
            solution.states,
            ~solution.backend_successful | jnp.any(~jnp.isfinite(solution.states)),
            "FLRW Lagrangian growth solve failed.",
        )
        first_normalization = states[-1, 0]
        first = states[:, 0] / first_normalization
        first_derivative = states[:, 1] / first_normalization
        second = states[:, 2] / first_normalization**2
        second_derivative = states[:, 3] / first_normalization**2
        return LagrangianGrowthHistory(
            self.scale_factors.astype(states.dtype),
            first,
            first_derivative / first,
            second,
            second_derivative / second,
            background.scale,
            self._provenance(background),
            background.realization,
        )


__all__ = ["FLRWGrowthPlan"]
