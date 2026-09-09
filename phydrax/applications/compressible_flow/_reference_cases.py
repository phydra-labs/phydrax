#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...nonlinear import (
    Bisection,
    NonlinearTermination,
    scalar_root,
    ScalarRootProblem,
)


class NormalShockReference(StrictModule):
    upstream_mach: Array
    downstream_mach: Array
    density_ratio: Array
    pressure_ratio: Array
    temperature_ratio: Array
    finite: Array
    successful: Array


class NormalShockReferencePlan(StrictModule, NonTrainableState):
    """Calorically-perfect-gas normal-shock relations."""

    gamma: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, gamma: float = 1.4, /):
        gamma_ = float(gamma)
        if not np.isfinite(gamma_) or gamma_ <= 1.0:
            raise ValueError("Normal-shock gamma must be finite and greater than one.")
        self.gamma = gamma_
        self.plan_id = canonical_fingerprint(
            {"kind": "normal-shock-reference", "gamma": gamma_}
        )

    def evaluate(self, upstream_mach: ArrayLike, /) -> NormalShockReference:
        mach = jnp.asarray(upstream_mach)
        mach = eqx.error_if(
            mach,
            jnp.any(~jnp.isfinite(mach) | (mach <= 1.0)),
            "Normal-shock upstream Mach number must exceed one.",
        )
        gamma = self.gamma
        mach_squared = mach * mach
        density_ratio = (
            (gamma + 1.0) * mach_squared / ((gamma - 1.0) * mach_squared + 2.0)
        )
        pressure_ratio = 1.0 + 2.0 * gamma * (mach_squared - 1.0) / (gamma + 1.0)
        downstream_squared = (1.0 + 0.5 * (gamma - 1.0) * mach_squared) / (
            gamma * mach_squared - 0.5 * (gamma - 1.0)
        )
        downstream = jnp.sqrt(downstream_squared)
        temperature_ratio = pressure_ratio / density_ratio
        finite = (
            jnp.all(jnp.isfinite(downstream))
            & jnp.all(jnp.isfinite(density_ratio))
            & jnp.all(jnp.isfinite(pressure_ratio))
            & jnp.all(jnp.isfinite(temperature_ratio))
        )
        return NormalShockReference(
            mach,
            downstream,
            density_ratio,
            pressure_ratio,
            temperature_ratio,
            finite,
            finite & jnp.all(downstream < 1.0),
        )


class ObliqueShockReference(StrictModule):
    shock_angle: Array
    deflection_angle: Array
    upstream_normal_mach: Array
    downstream_mach: Array
    density_ratio: Array
    pressure_ratio: Array
    temperature_ratio: Array
    residual: Array
    finite: Array
    successful: Array
    branch: str = eqx.field(static=True)


class ObliqueShockReferencePlan(StrictModule, NonTrainableState):
    """Attached weak or strong calorically-perfect-gas oblique shock."""

    gamma: float = eqx.field(static=True)
    branch: Literal["weak", "strong"] = eqx.field(static=True)
    scan_points: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gamma: float = 1.4,
        branch: Literal["weak", "strong"] = "weak",
        /,
        *,
        scan_points: int = 2048,
    ):
        gamma_ = float(gamma)
        points = int(scan_points)
        if (
            not np.isfinite(gamma_)
            or gamma_ <= 1.0
            or branch not in ("weak", "strong")
            or points < 128
        ):
            raise ValueError("Oblique-shock reference parameters are invalid.")
        self.gamma = gamma_
        self.branch = branch
        self.scan_points = points
        self.plan_id = canonical_fingerprint(
            {
                "kind": "oblique-shock-reference",
                "gamma": gamma_,
                "branch": branch,
                "scan_points": points,
            }
        )

    def _turning_angle(self, mach: Array, beta: Array, /) -> Array:
        sine = jnp.sin(beta)
        numerator = 2.0 * (mach * mach * sine * sine - 1.0)
        denominator = jnp.tan(beta) * (
            mach * mach * (self.gamma + jnp.cos(2.0 * beta)) + 2.0
        )
        return jnp.arctan(numerator / denominator)

    def evaluate(
        self, upstream_mach: float, deflection_angle: float, /
    ) -> ObliqueShockReference:
        mach = float(upstream_mach)
        theta = float(deflection_angle)
        if (
            not np.isfinite(mach)
            or mach <= 1.0
            or not np.isfinite(theta)
            or theta <= 0.0
            or theta >= 0.5 * np.pi
        ):
            raise ValueError("Attached oblique-shock inputs are invalid.")
        mach_angle = np.arcsin(1.0 / mach)
        epsilon = 32.0 * np.finfo(float).eps
        beta_samples = np.linspace(
            mach_angle + epsilon,
            0.5 * np.pi - epsilon,
            self.scan_points,
        )
        turning = np.asarray(
            self._turning_angle(jnp.asarray(mach), jnp.asarray(beta_samples))
        )
        maximum_index = int(np.argmax(turning))
        maximum_turning = float(turning[maximum_index])
        if theta >= maximum_turning:
            raise ValueError("Requested deflection produces a detached shock.")
        peak = float(beta_samples[maximum_index])
        bracket = (
            (mach_angle + epsilon, peak)
            if self.branch == "weak"
            else (peak, 0.5 * np.pi - epsilon)
        )
        problem = ScalarRootProblem(
            lambda beta, arguments: (
                self._turning_angle(arguments[0], beta) - arguments[1]
            ),
            bracket=bracket,
            problem_id=f"oblique-shock:{self.plan_id}",
        )
        root = scalar_root(
            problem,
            method=Bisection(),
            termination=NonlinearTermination(
                absolute_residual=1.0e-10,
                relative_residual=0.0,
                maximum_steps=100,
                maximum_evaluations=256,
                maximum_linear_iterations=1,
            ),
            args=(jnp.asarray(mach), jnp.asarray(theta)),
        )
        beta = root.root
        normal_mach = jnp.asarray(mach) * jnp.sin(beta)
        normal = NormalShockReferencePlan(self.gamma).evaluate(normal_mach)
        downstream = normal.downstream_mach / jnp.sin(beta - theta)
        finite = normal.finite & jnp.isfinite(beta) & jnp.isfinite(downstream)
        return ObliqueShockReference(
            beta,
            jnp.asarray(theta),
            normal_mach,
            downstream,
            normal.density_ratio,
            normal.pressure_ratio,
            normal.temperature_ratio,
            root.value,
            finite,
            finite & root.successful,
            self.branch,
        )


class PrandtlMeyerReference(StrictModule):
    upstream_mach: Array
    downstream_mach: Array
    upstream_angle: Array
    downstream_angle: Array
    turning_angle: Array
    residual: Array
    finite: Array
    successful: Array


class PrandtlMeyerReferencePlan(StrictModule, NonTrainableState):
    """Calorically-perfect-gas supersonic expansion reference."""

    gamma: float = eqx.field(static=True)
    maximum_mach: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, gamma: float = 1.4, /, *, maximum_mach: float = 100.0):
        gamma_ = float(gamma)
        maximum = float(maximum_mach)
        if (
            not np.isfinite(gamma_)
            or gamma_ <= 1.0
            or not np.isfinite(maximum)
            or maximum <= 1.0
        ):
            raise ValueError("Prandtl-Meyer reference parameters are invalid.")
        self.gamma = gamma_
        self.maximum_mach = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "prandtl-meyer-reference",
                "gamma": gamma_,
                "maximum_mach": maximum,
            }
        )

    def angle(self, mach: ArrayLike, /) -> Array:
        value = jnp.asarray(mach)
        value = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value) | (value <= 1.0)),
            "Prandtl-Meyer Mach number must exceed one.",
        )
        ratio = (self.gamma + 1.0) / (self.gamma - 1.0)
        root = jnp.sqrt(jnp.maximum(value * value - 1.0, 0.0))
        return jnp.sqrt(ratio) * jnp.arctan(root / jnp.sqrt(ratio)) - jnp.arctan(root)

    def evaluate(
        self, upstream_mach: float, turning_angle: float, /
    ) -> PrandtlMeyerReference:
        upstream = float(upstream_mach)
        turning = float(turning_angle)
        if (
            not np.isfinite(upstream)
            or upstream <= 1.0
            or not np.isfinite(turning)
            or turning <= 0.0
            or upstream >= self.maximum_mach
        ):
            raise ValueError("Prandtl-Meyer expansion inputs are invalid.")
        upstream_angle = self.angle(jnp.asarray(upstream))
        target = upstream_angle + turning
        maximum_angle = self.angle(jnp.asarray(self.maximum_mach))
        if float(target) >= float(maximum_angle):
            raise ValueError("Expansion exceeds the configured maximum Mach bracket.")
        problem = ScalarRootProblem(
            lambda mach, target_angle: self.angle(mach) - target_angle,
            bracket=(upstream, self.maximum_mach),
            problem_id=f"prandtl-meyer:{self.plan_id}",
        )
        root = scalar_root(
            problem,
            method=Bisection(),
            termination=NonlinearTermination(
                absolute_residual=1.0e-10,
                relative_residual=0.0,
                maximum_steps=100,
                maximum_evaluations=256,
                maximum_linear_iterations=1,
            ),
            args=target,
        )
        downstream_angle = self.angle(root.root)
        finite = (
            jnp.isfinite(root.root)
            & jnp.isfinite(downstream_angle)
            & jnp.isfinite(root.value)
        )
        return PrandtlMeyerReference(
            jnp.asarray(upstream),
            root.root,
            upstream_angle,
            downstream_angle,
            jnp.asarray(turning),
            root.value,
            finite,
            finite & root.successful,
        )


__all__ = [
    "NormalShockReference",
    "NormalShockReferencePlan",
    "ObliqueShockReference",
    "ObliqueShockReferencePlan",
    "PrandtlMeyerReference",
    "PrandtlMeyerReferencePlan",
]
