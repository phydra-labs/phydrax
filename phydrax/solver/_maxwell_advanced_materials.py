#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..discretization import CochainDiscretization
from ._maxwell import (
    _apply_hodge_metric,
    _positive_material,
    AbstractMaxwellConstitutivePlan,
    AbstractPreparedMaxwellConstitutive,
    DiagonalMaxwellConstitutivePlan,
    MaxwellCapabilities,
)
from ._maxwell_materials import MatrixMaxwellConstitutivePlan


class KerrPockelsMaxwellConstitutivePlan(AbstractMaxwellConstitutivePlan):
    """Real local D = εE + χ²E² + χ³E³ constitutive law."""

    permittivity: Array
    permeability: Array
    pockels: Array
    kerr: Array
    field_bound: float = eqx.field(static=True)
    newton_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        permittivity: ArrayLike = 1.0,
        permeability: ArrayLike = 1.0,
        pockels: ArrayLike = 0.0,
        kerr: ArrayLike = 0.0,
        field_bound: float,
        newton_steps: int = 12,
        tolerance: float = 1e-10,
    ):
        bound = float(field_bound)
        steps = int(newton_steps)
        tolerance_ = float(tolerance)
        if not np.isfinite(bound) or bound <= 0.0:
            raise ValueError("field_bound must be finite and positive.")
        if steps <= 0 or not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Newton steps/tolerance are invalid.")
        self.permittivity = jnp.asarray(permittivity)
        self.permeability = jnp.asarray(permeability)
        self.pockels = jnp.asarray(pockels)
        self.kerr = jnp.asarray(kerr)
        self.field_bound = bound
        self.newton_steps = steps
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kerr-pockels-maxwell-plan",
                "permittivity": array_tree_fingerprint(self.permittivity),
                "permeability": array_tree_fingerprint(self.permeability),
                "pockels": array_tree_fingerprint(self.pockels),
                "kerr": array_tree_fingerprint(self.kerr),
                "field_bound": bound,
                "newton_steps": steps,
                "tolerance": tolerance_,
            }
        )

    def prepare(
        self,
        cochain: CochainDiscretization,
        /,
    ) -> PreparedKerrPockelsMaxwellConstitutive:
        return PreparedKerrPockelsMaxwellConstitutive(self, cochain)


class PreparedKerrPockelsMaxwellConstitutive(AbstractPreparedMaxwellConstitutive):
    permittivity: Array
    permeability: Array
    pockels: Array
    kerr: Array
    field_bound: float = eqx.field(static=True)
    newton_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    capabilities: MaxwellCapabilities
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: KerrPockelsMaxwellConstitutivePlan,
        cochain: CochainDiscretization,
        /,
    ):
        count = cochain.cell_counts[1]
        epsilon = _positive_material("permittivity", plan.permittivity, count)
        mu = _positive_material("permeability", plan.permeability, cochain.cell_counts[2])
        pockels = jnp.broadcast_to(jnp.asarray(plan.pockels, dtype=float), (count,))
        kerr = jnp.broadcast_to(jnp.asarray(plan.kerr, dtype=float), (count,))
        minimum_tangent = epsilon - 2.0 * jnp.abs(pockels) * plan.field_bound
        minimum_tangent = eqx.error_if(
            minimum_tangent,
            jnp.any(~jnp.isfinite(pockels))
            | jnp.any(~jnp.isfinite(kerr))
            | jnp.any(kerr < 0.0)
            | jnp.any(minimum_tangent <= 0.0),
            "Nonlinear constitutive derivative must remain positive inside field_bound.",
        )
        self.permittivity = epsilon
        self.permeability = mu
        self.pockels = pockels
        self.kerr = kerr
        self.field_bound = plan.field_bound
        self.newton_steps = plan.newton_steps
        self.tolerance = plan.tolerance
        self.capabilities = MaxwellCapabilities(
            lossless=True,
            passive=True,
            nonlinear=True,
            reversible=False,
            structured_only=False,
            frequency_domain=False,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-kerr-pockels-maxwell",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
            }
        )

    def initialize_state(self, /) -> None:
        return None

    def validate_state(self, state: Any, /) -> None:
        if state is not None:
            raise ValueError("Instantaneous nonlinear material state must be None.")

    def electric_displacement(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return (
            self.permittivity * electric
            + self.pockels * electric**2
            + self.kerr * electric**3
        )

    def electric_field(self, displacement: Array, state: Any, /) -> Array:
        self.validate_state(state)
        if jnp.iscomplexobj(displacement):
            raise TypeError("Kerr/Pockels inversion currently requires real fields.")
        initial = displacement / self.permittivity

        def body(_, electric):
            residual = self.electric_displacement(electric, None) - displacement
            derivative = (
                self.permittivity
                + 2.0 * self.pockels * electric
                + 3.0 * self.kerr * electric**2
            )
            return electric - residual / derivative

        electric = jax.lax.fori_loop(0, self.newton_steps, body, initial)
        residual = self.electric_displacement(electric, None) - displacement
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(displacement)))
        return eqx.error_if(
            electric,
            (jnp.max(jnp.abs(residual)) > self.tolerance * scale)
            | jnp.any(jnp.abs(electric) > self.field_bound),
            "Kerr/Pockels constitutive solve did not converge inside field_bound.",
        )

    def magnetic_field(self, flux: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return flux / self.permeability

    def magnetic_flux(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permeability * magnetic

    def electric_conduction(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return jnp.zeros_like(electric)

    def magnetic_conduction(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return jnp.zeros_like(magnetic)

    def dissipated_power(
        self,
        electric: Array,
        magnetic: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        del electric, magnetic, electric_star, magnetic_star
        self.validate_state(state)
        return jnp.asarray(0.0)

    def advance_state(
        self,
        time: Array,
        state: Any,
        displacement: Array,
        magnetic_flux: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> None:
        del time, displacement, magnetic_flux, step_size, args
        self.validate_state(state)

    def energy(
        self,
        displacement: Array,
        magnetic_flux: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        electric_density = (
            0.5 * self.permittivity * electric**2
            + (1.0 / 3.0) * self.pockels * electric**3
            + 0.25 * self.kerr * electric**4
        )
        return jnp.sum(
            _apply_hodge_metric(electric_star, electric_density)
        ) + 0.5 * jnp.real(
            jnp.vdot(
                magnetic,
                _apply_hodge_metric(magnetic_star, magnetic_flux),
            )
        )

    def energy_rate(
        self,
        displacement: Array,
        magnetic_flux: Array,
        displacement_rate: Array,
        magnetic_rate: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        return jnp.real(
            jnp.vdot(
                self.electric_field(displacement, state),
                _apply_hodge_metric(electric_star, displacement_rate),
            )
            + jnp.vdot(
                self.magnetic_field(magnetic_flux, state),
                _apply_hodge_metric(magnetic_star, magnetic_rate),
            )
        )

    def wave_speed_bound(self, /) -> Array:
        tangent_minimum = (
            self.permittivity - 2.0 * jnp.abs(self.pockels) * self.field_bound
        )
        return jnp.sqrt(jnp.max(1.0 / self.permeability) / jnp.min(tangent_minimum))


class ActiveGainMaxwellConstitutivePlan(AbstractMaxwellConstitutivePlan):
    """Explicitly active diagonal gain with optional saturation."""

    permittivity: Array
    permeability: Array
    gain: Array
    saturation_intensity: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gain: ArrayLike,
        /,
        *,
        permittivity: ArrayLike = 1.0,
        permeability: ArrayLike = 1.0,
        saturation_intensity: float = np.inf,
    ):
        saturation = float(saturation_intensity)
        if saturation <= 0.0 or np.isnan(saturation):
            raise ValueError("saturation_intensity must be positive or infinite.")
        gain_ = jnp.asarray(gain, dtype=float)
        gain_ = eqx.error_if(
            gain_,
            jnp.any(~jnp.isfinite(gain_)) | jnp.any(gain_ < 0.0),
            "Gain coefficients must be finite and nonnegative.",
        )
        self.permittivity = jnp.asarray(permittivity)
        self.permeability = jnp.asarray(permeability)
        self.gain = gain_
        self.saturation_intensity = saturation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "active-gain-maxwell-plan",
                "gain": array_tree_fingerprint(gain_),
                "saturation_intensity": saturation,
            }
        )

    def prepare(
        self,
        cochain: CochainDiscretization,
        /,
    ) -> PreparedActiveGainMaxwellConstitutive:
        return PreparedActiveGainMaxwellConstitutive(self, cochain)


class PreparedActiveGainMaxwellConstitutive(AbstractPreparedMaxwellConstitutive):
    permittivity: Array
    permeability: Array
    gain: Array
    saturation_intensity: float = eqx.field(static=True)
    capabilities: MaxwellCapabilities
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ActiveGainMaxwellConstitutivePlan,
        cochain: CochainDiscretization,
        /,
    ):
        self.permittivity = _positive_material(
            "permittivity", plan.permittivity, cochain.cell_counts[1]
        )
        self.permeability = _positive_material(
            "permeability", plan.permeability, cochain.cell_counts[2]
        )
        self.gain = jnp.broadcast_to(plan.gain, (cochain.cell_counts[1],))
        self.saturation_intensity = plan.saturation_intensity
        self.capabilities = MaxwellCapabilities(
            lossless=False,
            passive=False,
            active=True,
            reversible=False,
            structured_only=False,
            frequency_domain=np.isinf(plan.saturation_intensity),
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-active-gain-maxwell",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
            }
        )

    def initialize_state(self, /) -> None:
        return None

    def validate_state(self, state: Any, /) -> None:
        if state is not None:
            raise ValueError("Instantaneous gain state must be None.")

    def electric_field(self, displacement: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return displacement / self.permittivity

    def electric_displacement(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permittivity * electric

    def magnetic_field(self, flux: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return flux / self.permeability

    def magnetic_flux(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permeability * magnetic

    def electric_conduction(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        intensity = jnp.real(electric * jnp.conj(electric))
        denominator = (
            1.0
            if np.isinf(self.saturation_intensity)
            else 1.0 + intensity / self.saturation_intensity
        )
        return -(self.gain / denominator) * electric

    def magnetic_conduction(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return jnp.zeros_like(magnetic)

    def dissipated_power(
        self,
        electric: Array,
        magnetic: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        del magnetic, magnetic_star
        current = self.electric_conduction(electric, state)
        return jnp.real(jnp.vdot(electric, _apply_hodge_metric(electric_star, current)))

    def advance_state(
        self,
        time: Array,
        state: Any,
        displacement: Array,
        magnetic_flux: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> None:
        del time, displacement, magnetic_flux, step_size, args
        self.validate_state(state)

    def energy(
        self,
        displacement: Array,
        magnetic_flux: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        return 0.5 * jnp.real(
            jnp.vdot(electric, _apply_hodge_metric(electric_star, displacement))
            + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_flux))
        )

    def energy_rate(
        self,
        displacement: Array,
        magnetic_flux: Array,
        displacement_rate: Array,
        magnetic_rate: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        return jnp.real(
            jnp.vdot(
                self.electric_field(displacement, state),
                _apply_hodge_metric(electric_star, displacement_rate),
            )
            + jnp.vdot(
                self.magnetic_field(magnetic_flux, state),
                _apply_hodge_metric(magnetic_star, magnetic_rate),
            )
        )

    def wave_speed_bound(self, /) -> Array:
        return jnp.sqrt(jnp.max(1.0 / self.permeability) / jnp.min(self.permittivity))


def gyrotropic_maxwell_constitutive(
    electric_matrix: ArrayLike,
    magnetic_matrix: ArrayLike,
    /,
    *,
    maximum_dense_dofs: int = 4096,
) -> MatrixMaxwellConstitutivePlan:
    return MatrixMaxwellConstitutivePlan(
        electric_matrix,
        magnetic_matrix,
        maximum_dense_dofs=maximum_dense_dofs,
    )


def subpixel_maxwell_constitutive(
    filling_fraction: ArrayLike,
    normal_projection: ArrayLike,
    material_one: ArrayLike,
    material_two: ArrayLike,
    /,
    *,
    permeability: ArrayLike = 1.0,
) -> DiagonalMaxwellConstitutivePlan:
    fraction = jnp.asarray(filling_fraction, dtype=float)
    projection = jnp.asarray(normal_projection, dtype=float)
    first = jnp.asarray(material_one, dtype=float)
    second = jnp.asarray(material_two, dtype=float)
    invalid = (
        jnp.any(~jnp.isfinite(fraction))
        | jnp.any(~jnp.isfinite(projection))
        | jnp.any((fraction < 0.0) | (fraction > 1.0))
        | jnp.any((projection < 0.0) | (projection > 1.0))
        | jnp.any(first <= 0.0)
        | jnp.any(second <= 0.0)
    )
    fraction = eqx.error_if(
        fraction,
        invalid,
        "Subpixel fractions/projections and materials are invalid.",
    )
    arithmetic = fraction * first + (1.0 - fraction) * second
    harmonic_inverse = fraction / first + (1.0 - fraction) / second
    inverse_effective = projection * harmonic_inverse + (1.0 - projection) / arithmetic
    return DiagonalMaxwellConstitutivePlan(
        permittivity=1.0 / inverse_effective,
        permeability=permeability,
    )


def fitted_interface_maxwell_constitutive(
    electric_material_integral: ArrayLike,
    electric_primal_measure: ArrayLike,
    magnetic_material_integral: ArrayLike,
    magnetic_primal_measure: ArrayLike,
    /,
) -> DiagonalMaxwellConstitutivePlan:
    electric_integral = jnp.asarray(electric_material_integral, dtype=float)
    electric_measure = jnp.asarray(electric_primal_measure, dtype=float)
    magnetic_integral = jnp.asarray(magnetic_material_integral, dtype=float)
    magnetic_measure = jnp.asarray(magnetic_primal_measure, dtype=float)
    if electric_integral.shape != electric_measure.shape:
        raise ValueError("Electric fitted material integral/measure shapes must match.")
    if magnetic_integral.shape != magnetic_measure.shape:
        raise ValueError("Magnetic fitted material integral/measure shapes must match.")
    invalid = (
        jnp.any(~jnp.isfinite(electric_integral))
        | jnp.any(~jnp.isfinite(magnetic_integral))
        | jnp.any(~jnp.isfinite(electric_measure))
        | jnp.any(~jnp.isfinite(magnetic_measure))
        | jnp.any(electric_integral <= 0.0)
        | jnp.any(magnetic_integral <= 0.0)
        | jnp.any(electric_measure <= 0.0)
        | jnp.any(magnetic_measure <= 0.0)
    )
    electric_integral = eqx.error_if(
        electric_integral,
        invalid,
        "Fitted interface material integrals/measures must be finite and positive.",
    )
    return DiagonalMaxwellConstitutivePlan(
        permittivity=electric_integral / electric_measure,
        permeability=magnetic_integral / magnetic_measure,
    )


__all__ = [
    "ActiveGainMaxwellConstitutivePlan",
    "KerrPockelsMaxwellConstitutivePlan",
    "PreparedActiveGainMaxwellConstitutive",
    "PreparedKerrPockelsMaxwellConstitutive",
    "fitted_interface_maxwell_constitutive",
    "gyrotropic_maxwell_constitutive",
    "subpixel_maxwell_constitutive",
]
