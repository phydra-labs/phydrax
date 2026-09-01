#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ._advanced import PedrizzettiRelaxationPlan3D, ReformulatedVPMPlan3D
from ._source import VortexSourceState


class VortexFormulationRate(StrictModule):
    strength_rate: Array
    core_radius_rate: Array | None
    volume_rate: Array | None
    subfilter_energy_rate: Array | None
    finite: Array
    conservation_residual: Array
    formulation_id: str = eqx.field(static=True)
    backend_evidence: object


class AbstractVortexFormulation(StrictModule, NonTrainableState):
    dimension: AbstractAttribute[int]
    requires_dynamic_core: AbstractAttribute[bool]
    formulation_id: AbstractAttribute[str]

    @abc.abstractmethod
    def rate(
        self,
        source: VortexSourceState,
        velocity_gradient: Array,
        diffusion_rate: Array,
        /,
    ) -> VortexFormulationRate:
        """Return formulation-owned rates without topology changes."""


class ClassicVPMFormulation(AbstractVortexFormulation):
    dimension: int = eqx.field(static=True)
    requires_dynamic_core: bool = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /):
        dimension_ = int(dimension)
        if dimension_ not in (2, 3):
            raise ValueError("Classic VPM dimension must be 2 or 3.")
        self.dimension = dimension_
        self.requires_dynamic_core = False
        self.formulation_id = canonical_fingerprint(
            {"kind": "classic-vpm-formulation", "dimension": dimension_}
        )

    def rate(
        self,
        source: VortexSourceState,
        velocity_gradient: Array,
        diffusion_rate: Array,
        /,
    ) -> VortexFormulationRate:
        if (
            source.dimension != self.dimension
            or diffusion_rate.shape != source.strength.shape
        ):
            raise ValueError("Classic VPM source/diffusion shapes are incompatible.")
        if self.dimension == 2:
            stretching = jnp.zeros_like(source.strength)
        else:
            if velocity_gradient.shape != (source.capacity, 3, 3):
                raise ValueError(
                    "Classic 3-D VPM requires one velocity gradient per source."
                )
            stretching = contract(
                "...ij,...j->...i", velocity_gradient, source.safe_strength()
            )
        strength_rate = diffusion_rate + stretching
        finite = jnp.all(jnp.isfinite(strength_rate))
        return VortexFormulationRate(
            strength_rate,
            None,
            None,
            None,
            finite,
            jnp.sum(strength_rate - diffusion_rate, axis=0),
            self.formulation_id,
            None,
        )


class ReformulatedVPMFormulation(AbstractVortexFormulation):
    plan: ReformulatedVPMPlan3D
    dimension: int = eqx.field(static=True)
    requires_dynamic_core: bool = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)

    def __init__(self, f: float = 0.0, g: float = 0.2, /):
        plan = ReformulatedVPMPlan3D(f, g)
        self.plan = plan
        self.dimension = 3
        self.requires_dynamic_core = True
        self.formulation_id = canonical_fingerprint(
            {"kind": "integrated-reformulated-vpm", "plan": plan.formulation_id}
        )

    def rate(
        self,
        source: VortexSourceState,
        velocity_gradient: Array,
        diffusion_rate: Array,
        /,
    ) -> VortexFormulationRate:
        if (
            source.dimension != 3
            or source.core_radius is None
            or velocity_gradient.shape != (source.capacity, 3, 3)
        ):
            raise ValueError(
                "Integrated rVPM requires 3-D source core radii and gradients."
            )
        stretching = contract(
            "...ij,...j->...i", velocity_gradient, source.safe_strength()
        )
        reformulated = self.plan.rate(
            source.safe_strength(), stretching, source.safe_core_radius()
        )
        strength_rate = reformulated.strength_rate + diffusion_rate
        finite = reformulated.finite & jnp.all(jnp.isfinite(strength_rate))
        return VortexFormulationRate(
            strength_rate,
            reformulated.core_radius_rate,
            None,
            None,
            finite,
            reformulated.conservation_residual,
            self.formulation_id,
            reformulated,
        )


class VortexRelaxationSchedule(StrictModule, NonTrainableState):
    relaxation: PedrizzettiRelaxationPlan3D
    every_steps: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, relaxation: PedrizzettiRelaxationPlan3D, every_steps: int, /):
        if (
            not isinstance(relaxation, PedrizzettiRelaxationPlan3D)
            or int(every_steps) <= 0
        ):
            raise ValueError("Relaxation schedule requires a plan and positive cadence.")
        self.relaxation = relaxation
        self.every_steps = int(every_steps)
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "vortex-relaxation-schedule",
                "relaxation": relaxation.relaxation_id,
                "every_steps": self.every_steps,
            }
        )

    def apply(
        self,
        step_index: ArrayLike,
        strength: ArrayLike,
        represented_vorticity: ArrayLike,
        /,
    ):
        index = jnp.asarray(step_index)
        candidate = self.relaxation.apply(strength, represented_vorticity)
        selected = (index % self.every_steps) == 0
        value = jnp.where(selected, candidate.strength, jnp.asarray(strength))
        return value, selected, candidate


class VortexLESEvidence(StrictModule):
    coefficient: Array
    strain_magnitude: Array
    energy_transfer: Array
    dissipative: Array
    filter_id: str = eqx.field(static=True)


class VortexLESPlan(StrictModule, NonTrainableState):
    mode: str = eqx.field(static=True)
    coefficient: float = eqx.field(static=True)
    allow_backscatter: bool = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: str = "constant",
        coefficient: float = 0.1,
        /,
        *,
        allow_backscatter: bool = False,
        filter_id: str = "vortex-core-filter",
    ):
        mode_ = str(mode)
        coefficient_ = float(coefficient)
        if (
            mode_ not in ("constant", "dynamic")
            or not math.isfinite(coefficient_)
            or coefficient_ < 0.0
            or not str(filter_id)
        ):
            raise ValueError("LES mode/coefficient/filter identity is invalid.")
        self.mode = mode_
        self.coefficient = coefficient_
        self.allow_backscatter = bool(allow_backscatter)
        self.filter_id = str(filter_id)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vortex-les-plan",
                "mode": mode_,
                "coefficient": coefficient_,
                "allow_backscatter": self.allow_backscatter,
                "filter_id": self.filter_id,
            }
        )

    def rate(
        self, source: VortexSourceState, velocity_gradient: Array, /
    ) -> tuple[Array, VortexLESEvidence]:
        if (
            source.dimension != 3
            or source.core_radius is None
            or velocity_gradient.shape != (source.capacity, 3, 3)
        ):
            raise ValueError("Vortex LES requires 3-D source cores and gradients.")
        symmetric = 0.5 * (velocity_gradient + jnp.swapaxes(velocity_gradient, -1, -2))
        magnitude = jnp.sqrt(2.0 * jnp.sum(symmetric * symmetric, axis=(-2, -1)))
        if self.mode == "dynamic":
            numerator = jnp.sum(
                source.safe_strength()
                * contract("...ij,...j->...i", symmetric, source.safe_strength()),
                axis=-1,
            )
            denominator = jnp.maximum(
                jnp.sum(source.safe_strength() ** 2, axis=-1) * magnitude,
                jnp.finfo(velocity_gradient.dtype).tiny,
            )
            coefficient = numerator / denominator
            if not self.allow_backscatter:
                coefficient = jnp.maximum(coefficient, 0.0)
        else:
            coefficient = jnp.full(
                (source.capacity,), self.coefficient, dtype=velocity_gradient.dtype
            )
        eddy_rate = (
            -coefficient[:, None]
            * source.safe_core_radius()[:, None] ** 2
            * magnitude[:, None]
            * source.safe_strength()
        )
        energy_transfer = jnp.sum(eddy_rate * source.safe_strength(), axis=-1)
        evidence = VortexLESEvidence(
            coefficient,
            magnitude,
            energy_transfer,
            jnp.all(energy_transfer <= 0.0)
            if not self.allow_backscatter
            else jnp.asarray(True),
            self.filter_id,
        )
        return eddy_rate, evidence


class BaroclinicVorticityRate(StrictModule):
    vorticity_rate: Array
    density_rate: Array
    scalar_rate: Array | None
    finite: Array
    formulation_id: str = eqx.field(static=True)


class BaroclinicVortexFormulation(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    gravity: Array
    formulation_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, gravity: ArrayLike | None = None):
        dimension_ = int(dimension)
        if dimension_ not in (2, 3):
            raise ValueError("Baroclinic formulation dimension must be 2 or 3.")
        gravity_ = (
            jnp.zeros((dimension_,))
            if gravity is None
            else jnp.asarray(gravity, dtype=float)
        )
        if gravity_.shape != (dimension_,):
            raise ValueError("Baroclinic gravity must match dimension.")
        self.dimension = dimension_
        self.gravity = gravity_
        self.formulation_id = canonical_fingerprint(
            {
                "kind": "baroclinic-vortex-formulation",
                "dimension": dimension_,
                "gravity": tuple(float(value) for value in gravity_),
            }
        )

    def rate(
        self,
        density: ArrayLike,
        density_gradient: ArrayLike,
        pressure_gradient: ArrayLike,
        velocity: ArrayLike,
        scalar_gradient: ArrayLike | None = None,
        /,
    ) -> BaroclinicVorticityRate:
        rho = jnp.asarray(density)
        grad_rho = jnp.asarray(density_gradient, dtype=rho.dtype)
        grad_pressure = jnp.asarray(pressure_gradient, dtype=rho.dtype)
        velocity_ = jnp.asarray(velocity, dtype=rho.dtype)
        if (
            rho.ndim != 1
            or grad_rho.shape != (rho.size, self.dimension)
            or grad_pressure.shape != grad_rho.shape
            or velocity_.shape != grad_rho.shape
        ):
            raise ValueError("Baroclinic fields have incompatible shapes.")
        safe_rho = eqx.error_if(
            rho,
            jnp.any(~jnp.isfinite(rho) | (rho <= 0.0)),
            "Density must be finite and positive.",
        )
        if self.dimension == 2:
            source = (
                grad_rho[:, 0] * grad_pressure[:, 1]
                - grad_rho[:, 1] * grad_pressure[:, 0]
            ) / safe_rho**2
            buoyancy = grad_rho[:, 0] * self.gravity[1] - grad_rho[:, 1] * self.gravity[0]
        else:
            source = jnp.cross(grad_rho, grad_pressure) / safe_rho[:, None] ** 2
            buoyancy = (
                jnp.cross(grad_rho, jnp.broadcast_to(self.gravity, grad_rho.shape))
                / safe_rho[:, None]
            )
        density_rate = -jnp.sum(velocity_ * grad_rho, axis=-1)
        scalar_rate = (
            None
            if scalar_gradient is None
            else -jnp.sum(
                velocity_ * jnp.asarray(scalar_gradient, dtype=rho.dtype), axis=-1
            )
        )
        vorticity_rate = source + buoyancy
        finite = jnp.all(jnp.isfinite(vorticity_rate)) & jnp.all(
            jnp.isfinite(density_rate)
        )
        if scalar_rate is not None:
            finite = finite & jnp.all(jnp.isfinite(scalar_rate))
        return BaroclinicVorticityRate(
            vorticity_rate, density_rate, scalar_rate, finite, self.formulation_id
        )


__all__ = [
    "AbstractVortexFormulation",
    "BaroclinicVortexFormulation",
    "BaroclinicVorticityRate",
    "ClassicVPMFormulation",
    "ReformulatedVPMFormulation",
    "VortexFormulationRate",
    "VortexLESEvidence",
    "VortexLESPlan",
    "VortexRelaxationSchedule",
]
