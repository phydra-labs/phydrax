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
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CochainDiscretization, StructuredCochainBridge
from ..geometry._contracts import CompiledGeometry, GeometryKind
from ._maxwell import (
    _apply_hodge_metric,
    _positive_material,
    AbstractMaxwellConstitutivePlan,
    AbstractPreparedMaxwellConstitutive,
    DiagonalMaxwellConstitutivePlan,
    MaxwellCapabilities,
    MaxwellCochainLayout,
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
        layout: MaxwellCochainLayout,
        /,
    ) -> PreparedKerrPockelsMaxwellConstitutive:
        return PreparedKerrPockelsMaxwellConstitutive(self, cochain, layout)


class PreparedKerrPockelsMaxwellConstitutive(AbstractPreparedMaxwellConstitutive):
    permittivity: Array
    permeability: Array
    pockels: Array
    kerr: Array
    field_bound: float = eqx.field(static=True)
    newton_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    capabilities: MaxwellCapabilities
    layout_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: KerrPockelsMaxwellConstitutivePlan,
        cochain: CochainDiscretization,
        layout: MaxwellCochainLayout,
        /,
    ):
        count = layout.electric_count
        epsilon = _positive_material("permittivity", plan.permittivity, count)
        mu = _positive_material("permeability", plan.permeability, layout.magnetic_count)
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
        self.layout_id = layout.layout_id
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
            linear_time_invariant=False,
            structured_only=False,
            frequency_domain=False,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-kerr-pockels-maxwell",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
                "layout": layout.layout_id,
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
        layout: MaxwellCochainLayout,
        /,
    ) -> PreparedActiveGainMaxwellConstitutive:
        return PreparedActiveGainMaxwellConstitutive(self, cochain, layout)


class PreparedActiveGainMaxwellConstitutive(AbstractPreparedMaxwellConstitutive):
    permittivity: Array
    permeability: Array
    gain: Array
    saturation_intensity: float = eqx.field(static=True)
    capabilities: MaxwellCapabilities
    layout_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ActiveGainMaxwellConstitutivePlan,
        cochain: CochainDiscretization,
        layout: MaxwellCochainLayout,
        /,
    ):
        self.permittivity = _positive_material(
            "permittivity", plan.permittivity, layout.electric_count
        )
        self.permeability = _positive_material(
            "permeability", plan.permeability, layout.magnetic_count
        )
        self.gain = jnp.broadcast_to(plan.gain, (layout.electric_count,))
        self.saturation_intensity = plan.saturation_intensity
        self.layout_id = layout.layout_id
        self.capabilities = MaxwellCapabilities(
            lossless=False,
            passive=False,
            active=True,
            reversible=False,
            structured_only=False,
            linear_time_invariant=np.isinf(plan.saturation_intensity),
            frequency_domain=np.isinf(plan.saturation_intensity),
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-active-gain-maxwell",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
                "layout": layout.layout_id,
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


class MaxwellScalarMaterialAssemblyPolicy(StrictModule, NonTrainableState):
    quadrature_order: int = eqx.field(static=True)
    maximum_samples: int = eqx.field(static=True)

    def __init__(self, *, quadrature_order: int = 3, maximum_samples: int = 10_000_000):
        order, maximum = int(quadrature_order), int(maximum_samples)
        if order < 1 or maximum < 1:
            raise ValueError(
                "Scalar material quadrature order/sample budget must be positive."
            )
        self.quadrature_order, self.maximum_samples = order, maximum


class MaxwellScalarMaterialEvidence(StrictModule, NonTrainableState):
    electric_fractions: Array
    magnetic_fractions: Array
    quadrature_order: int = eqx.field(static=True)
    geometry_certificate: str = eqx.field(static=True)
    bridge_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    material_provenance: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class MaxwellScalarMaterialAssemblyResult(StrictModule):
    constitutive: DiagonalMaxwellConstitutivePlan
    evidence: MaxwellScalarMaterialEvidence


def _cochain_region_fractions(
    geometry: CompiledGeometry,
    bridge: StructuredCochainBridge,
    degree: int,
    policy: MaxwellScalarMaterialAssemblyPolicy,
    /,
) -> Array:
    nodes, weights = np.polynomial.legendre.leggauss(policy.quadrature_order)
    fractions: list[np.ndarray] = []
    sample_total = 0
    dimension = bridge.dimension
    for orientation, shape in zip(
        bridge.orientations[degree],
        bridge.orientation_shapes[degree],
        strict=True,
    ):
        points: list[np.ndarray] = []
        entity_weights: list[np.ndarray] = []
        for index in np.ndindex(*shape):
            axis_points: list[np.ndarray] = []
            axis_weights: list[np.ndarray] = []
            for axis in range(dimension):
                coordinates = np.asarray(
                    bridge.grid.structured_axes[axis].point_coordinates
                )
                if axis in orientation:
                    lower, upper = coordinates[index[axis]], coordinates[index[axis] + 1]
                    axis_points.append(0.5 * ((upper - lower) * nodes + upper + lower))
                    axis_weights.append(0.5 * weights)
                else:
                    axis_points.append(np.asarray([coordinates[index[axis]]]))
                    axis_weights.append(np.asarray([1.0]))
            mesh = np.meshgrid(*axis_points, indexing="ij")
            weight_mesh = np.meshgrid(*axis_weights, indexing="ij")
            points.append(
                np.stack(tuple(value.reshape((-1,)) for value in mesh), axis=-1)
            )
            entity_weights.append(
                np.prod(
                    np.stack(
                        tuple(value.reshape((-1,)) for value in weight_mesh), axis=0
                    ),
                    axis=0,
                )
            )
        sample_total += sum(value.shape[0] for value in points)
        if sample_total > policy.maximum_samples:
            raise ValueError("Scalar material assembly exceeds maximum_samples.")
        point_array = np.concatenate(points, axis=0)
        inside = np.asarray(geometry.contains(jnp.asarray(point_array)), dtype=float)
        counts = tuple(value.shape[0] for value in points)
        starts = np.cumsum((0, *counts))
        values = np.asarray(
            [
                np.sum(inside[starts[i] : starts[i + 1]] * entity_weights[i])
                / np.sum(entity_weights[i])
                for i in range(len(points))
            ]
        )
        fractions.append(values)
    return jnp.asarray(np.concatenate(fractions))


def assemble_scalar_maxwell_material(
    geometry: CompiledGeometry,
    bridge: StructuredCochainBridge,
    layout: MaxwellCochainLayout,
    /,
    *,
    inside_permittivity: ArrayLike,
    outside_permittivity: ArrayLike,
    inside_permeability: ArrayLike = 1.0,
    outside_permeability: ArrayLike = 1.0,
    policy: MaxwellScalarMaterialAssemblyPolicy | None = None,
) -> MaxwellScalarMaterialAssemblyResult:
    """Integrate a static two-phase scalar region over retained primal supports."""

    if (
        not isinstance(geometry, CompiledGeometry)
        or geometry.kind is not GeometryKind.REGION
    ):
        raise TypeError(
            "Scalar Maxwell material assembly requires compiled region geometry."
        )
    if not isinstance(bridge, StructuredCochainBridge) or not isinstance(
        layout, MaxwellCochainLayout
    ):
        raise TypeError("Scalar material assembly requires a structured bridge/layout.")
    if geometry.ambient_dimension != bridge.dimension:
        raise ValueError("Geometry and Maxwell bridge dimensions do not match.")
    if layout.layout_id != MaxwellCochainLayout(bridge, layout.polarization).layout_id:
        raise ValueError("Maxwell material layout belongs to another bridge.")
    policy_ = policy or MaxwellScalarMaterialAssemblyPolicy()
    electric_fraction = _cochain_region_fractions(
        geometry, bridge, layout.electric_degree, policy_
    )
    magnetic_fraction = _cochain_region_fractions(
        geometry, bridge, layout.magnetic_degree, policy_
    )
    inside_e, outside_e = (
        jnp.asarray(inside_permittivity),
        jnp.asarray(outside_permittivity),
    )
    inside_m, outside_m = (
        jnp.asarray(inside_permeability),
        jnp.asarray(outside_permeability),
    )
    scalars = (inside_e, outside_e, inside_m, outside_m)
    if any(value.shape not in ((), (1,)) for value in scalars):
        raise ValueError("First scalar material assembly accepts scalar phase constants.")
    if any(jnp.iscomplexobj(value) for value in scalars):
        raise TypeError("Scalar material assembly requires real phase constants.")
    if any(
        bool(jnp.any(~jnp.isfinite(value))) or bool(jnp.any(value <= 0.0))
        for value in scalars
    ):
        raise ValueError("Scalar material phase constants must be finite and positive.")
    epsilon = outside_e + electric_fraction * (inside_e - outside_e)
    mu = outside_m + magnetic_fraction * (inside_m - outside_m)
    certificate = geometry.field_certificate
    certificate_text = canonical_fingerprint(
        {
            "zero_set": certificate.zero_set_accuracy.value,
            "sign": certificate.sign_reliability.value,
            "distance": certificate.distance_semantics.value,
            "regularity": certificate.regularity.value,
            "provenance": certificate.provenance,
        }
    )
    provenance = canonical_fingerprint(
        {
            "inside_permittivity": array_tree_fingerprint(inside_e),
            "outside_permittivity": array_tree_fingerprint(outside_e),
            "inside_permeability": array_tree_fingerprint(inside_m),
            "outside_permeability": array_tree_fingerprint(outside_m),
        }
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "maxwell-scalar-material-evidence",
            "geometry": certificate_text,
            "bridge": bridge.bridge_id,
            "layout": layout.layout_id,
            "quadrature_order": policy_.quadrature_order,
            "electric_fraction": array_tree_fingerprint(electric_fraction),
            "magnetic_fraction": array_tree_fingerprint(magnetic_fraction),
            "materials": provenance,
        }
    )
    constitutive = DiagonalMaxwellConstitutivePlan(
        permittivity=epsilon,
        permeability=mu,
    )
    evidence = MaxwellScalarMaterialEvidence(
        electric_fraction,
        magnetic_fraction,
        policy_.quadrature_order,
        certificate_text,
        bridge.bridge_id,
        layout.layout_id,
        provenance,
        evidence_id,
    )
    return MaxwellScalarMaterialAssemblyResult(constitutive, evidence)


__all__ = [
    "ActiveGainMaxwellConstitutivePlan",
    "KerrPockelsMaxwellConstitutivePlan",
    "MaxwellScalarMaterialAssemblyPolicy",
    "MaxwellScalarMaterialAssemblyResult",
    "MaxwellScalarMaterialEvidence",
    "PreparedActiveGainMaxwellConstitutive",
    "PreparedKerrPockelsMaxwellConstitutive",
    "assemble_scalar_maxwell_material",
    "fitted_interface_maxwell_constitutive",
    "gyrotropic_maxwell_constitutive",
    "subpixel_maxwell_constitutive",
]
