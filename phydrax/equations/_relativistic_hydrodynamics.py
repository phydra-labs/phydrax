#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Special-relativistic and Valencia general-relativistic hydrodynamics.

Primitive states are ``[rho, eps, v^i]`` where ``v^i`` is the Eulerian
contravariant three-velocity. Conserved states are ``[D, S_i, tau]``. Valencia
states are densitized by ``sqrt(det(gamma))``; SRHD states are the unit-
Cartesian specialization. Momentum is covariant in both formulations.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ..metrix._spacetime_conventions import RelativityConvention
from ._hyperbolic_systems import AbstractConservationSystem
from ._relativistic_eos import AbstractRelativisticEOS, RelativisticEOSState


class RelativisticHydrodynamicsLayout(StrictModule, NonTrainableState):
    """Static primitive/conserved layout and tensor-variance contract."""

    dimension: int = eqx.field(static=True)
    primitive_names: tuple[str, ...] = eqx.field(static=True)
    conserved_names: tuple[str, ...] = eqx.field(static=True)
    densitization: str = eqx.field(static=True)
    velocity_variance: str = eqx.field(static=True)
    momentum_variance: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, densitized: bool):
        dimension_ = int(dimension)
        if dimension_ not in (1, 2, 3):
            raise ValueError("Relativistic hydrodynamics dimension must be 1, 2, or 3.")
        densitization = "sqrt_spatial_determinant" if densitized else "none"
        primitive = (
            "rest_mass_density",
            "specific_internal_energy",
            *(f"eulerian_velocity_contravariant_{axis}" for axis in range(dimension_)),
        )
        conserved = (
            "densitized_rest_mass" if densitized else "rest_mass",
            *(
                f"densitized_momentum_covariant_{axis}"
                if densitized
                else f"momentum_covariant_{axis}"
                for axis in range(dimension_)
            ),
            "densitized_energy_excluding_rest_mass"
            if densitized
            else "energy_excluding_rest_mass",
        )
        self.dimension = dimension_
        self.primitive_names = primitive
        self.conserved_names = conserved
        self.densitization = densitization
        self.velocity_variance = "contravariant-eulerian"
        self.momentum_variance = "covariant-spatial"
        self.layout_id = canonical_fingerprint(
            {
                "kind": "relativistic-hydrodynamics-layout",
                "dimension": dimension_,
                "primitive": list(primitive),
                "conserved": list(conserved),
                "densitization": densitization,
                "velocity_variance": self.velocity_variance,
                "momentum_variance": self.momentum_variance,
            }
        )

    @property
    def component_count(self) -> int:
        return self.dimension + 2

    @property
    def velocity_slice(self) -> slice:
        return slice(2, 2 + self.dimension)

    @property
    def momentum_slice(self) -> slice:
        return slice(1, 1 + self.dimension)

    @property
    def energy_index(self) -> int:
        return self.dimension + 1


class RelativisticFluidEvaluation(StrictModule):
    """Primitive thermodynamics and physical-domain evidence at one batch."""

    primitive: Array
    eos_state: RelativisticEOSState
    velocity_squared: Array
    lorentz_factor: Array
    covariant_velocity: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    system_id: str = eqx.field(static=True)

    @property
    def rest_mass_density(self) -> Array:
        return self.primitive[..., 0]

    @property
    def specific_internal_energy(self) -> Array:
        return self.primitive[..., 1]

    @property
    def velocity(self) -> Array:
        return self.primitive[..., 2:]

    @property
    def pressure(self) -> Array:
        return self.eos_state.pressure

    @property
    def specific_enthalpy(self) -> Array:
        return self.eos_state.specific_enthalpy

    @property
    def sound_speed_squared(self) -> Array:
        return self.eos_state.sound_speed_squared


class ValenciaGeometrySource(StrictModule, NonTrainableState):
    """Spatial ADM derivatives needed by the Valencia volume source.

    ``alpha_gradient[j] = partial_j alpha``,
    ``beta_gradient[j, i] = partial_j beta^i``, and
    ``spatial_metric_gradient[j, i, k] = partial_j gamma_ik``. The base ADM
    exchange intentionally contains no derivatives; this record binds the
    source-only data to that exact geometry snapshot.
    """

    geometry: ADMGridGeometry
    alpha_gradient: Array
    beta_gradient: Array
    spatial_metric_gradient: Array
    source_geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: ADMGridGeometry,
        alpha_gradient: ArrayLike,
        beta_gradient: ArrayLike,
        spatial_metric_gradient: ArrayLike,
        /,
    ):
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be an ADMGridGeometry.")
        alpha = jnp.asarray(alpha_gradient, dtype=geometry.alpha.dtype)
        beta = jnp.asarray(beta_gradient, dtype=geometry.alpha.dtype)
        spatial = jnp.asarray(spatial_metric_gradient, dtype=geometry.alpha.dtype)
        leading = geometry.leading_shape
        if (
            alpha.shape != leading + (3,)
            or beta.shape != leading + (3, 3)
            or spatial.shape != leading + (3, 3, 3)
        ):
            raise ValueError(
                "Valencia source derivatives must have shapes S+(3,), "
                "S+(3,3), and S+(3,3,3)."
            )
        self.geometry = geometry
        self.alpha_gradient = alpha
        self.beta_gradient = beta
        self.spatial_metric_gradient = spatial
        self.source_geometry_id = canonical_fingerprint(
            {
                "kind": "valencia-source-geometry",
                "geometry_lineage": geometry.geometry_lineage_id,
                "derivative_convention": "partial-j-alpha_beta-i_gamma-ik",
            }
        )

    @property
    def finite(self) -> Array:
        return (
            self.geometry.finite
            & jnp.all(jnp.isfinite(self.alpha_gradient), axis=-1)
            & jnp.all(jnp.isfinite(self.beta_gradient), axis=(-2, -1))
            & jnp.all(jnp.isfinite(self.spatial_metric_gradient), axis=(-3, -2, -1))
        )

    @property
    def physically_valid(self) -> Array:
        return self.geometry.physically_valid & self.finite


def _validate_primitive(
    primitive: ArrayLike, layout: RelativisticHydrodynamicsLayout
) -> Array:
    value = jnp.asarray(primitive)
    if value.ndim < 1 or value.shape[-1] != layout.component_count:
        raise ValueError(
            f"Primitive state must end in {layout.component_count} components."
        )
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError("Relativistic primitive states require a real floating dtype.")
    return value


def _validate_conserved(
    conserved: ArrayLike, layout: RelativisticHydrodynamicsLayout
) -> Array:
    value = jnp.asarray(conserved)
    if value.ndim < 1 or value.shape[-1] != layout.component_count:
        raise ValueError(
            f"Conserved state must end in {layout.component_count} components."
        )
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError("Relativistic conserved states require a real floating dtype.")
    return value


def _evaluate_primitive(
    eos: AbstractRelativisticEOS,
    primitive: Array,
    spatial_metric: Array,
    system_id: str,
    /,
) -> RelativisticFluidEvaluation:
    velocity = primitive[..., 2:]
    covariant = ein.contract("...ij,...j->...i", spatial_metric, velocity)
    velocity_squared = ein.contract("...i,...i->...", velocity, covariant)
    safe_margin = jnp.maximum(1.0 - velocity_squared, jnp.finfo(primitive.dtype).tiny)
    lorentz = 1.0 / jnp.sqrt(safe_margin)
    eos_state = eos.evaluate(primitive[..., 0], primitive[..., 1])
    finite = (
        jnp.all(jnp.isfinite(primitive), axis=-1)
        & jnp.all(jnp.isfinite(spatial_metric), axis=(-2, -1))
        & jnp.isfinite(velocity_squared)
        & jnp.isfinite(lorentz)
        & eos_state.finite
    )
    physical = (
        finite
        & (velocity_squared >= 0.0)
        & (velocity_squared < 1.0)
        & (eos_state.pressure > 0.0)
        & (eos_state.specific_enthalpy > 0.0)
        & (eos_state.sound_speed_squared >= 0.0)
        & (eos_state.sound_speed_squared < 1.0)
        & eos_state.physically_valid
    )
    return RelativisticFluidEvaluation(
        primitive,
        eos_state,
        velocity_squared,
        lorentz,
        covariant,
        finite,
        eos_state.converged,
        physical,
        eos_state.qualified & physical,
        eos_state.derivative_valid & physical,
        system_id,
    )


def _conserved_from_evaluation(
    evaluation: RelativisticFluidEvaluation, sqrt_spatial_determinant: Array, /
) -> Array:
    rho_h_w2 = (
        evaluation.rest_mass_density
        * evaluation.specific_enthalpy
        * evaluation.lorentz_factor**2
    )
    mass = evaluation.rest_mass_density * evaluation.lorentz_factor
    momentum = rho_h_w2[..., None] * evaluation.covariant_velocity
    energy = rho_h_w2 - evaluation.pressure - mass
    return sqrt_spatial_determinant[..., None] * jnp.concatenate(
        (mass[..., None], momentum, energy[..., None]), axis=-1
    )


def _primitive_at_pressure(
    eos: AbstractRelativisticEOS,
    undensitized: Array,
    inverse_spatial_metric: Array,
    pressure: Array,
    /,
) -> tuple[Array, Array, Array]:
    mass = undensitized[..., 0]
    momentum = undensitized[..., 1:-1]
    total_energy = undensitized[..., -1] + mass
    momentum_squared = ein.contract(
        "...i,...ij,...j->...", momentum, inverse_spatial_metric, momentum
    )
    q = total_energy + pressure
    safe_q = jnp.maximum(q, jnp.sqrt(jnp.maximum(momentum_squared, 0.0)))
    velocity = (
        ein.contract("...ij,...j->...i", inverse_spatial_metric, momentum)
        / safe_q[..., None]
    )
    velocity_squared = momentum_squared / safe_q**2
    lorentz = 1.0 / jnp.sqrt(jnp.maximum(1.0 - velocity_squared, 1.0e-14))
    density = mass / lorentz
    specific_enthalpy = safe_q / (mass * lorentz)
    specific_internal_energy = specific_enthalpy - 1.0 - pressure / density
    primitive = jnp.concatenate(
        (density[..., None], specific_internal_energy[..., None], velocity), axis=-1
    )
    eos_state = eos.evaluate(density, specific_internal_energy)
    return primitive, eos_state.pressure - pressure, momentum_squared


def _fixed_pressure_recovery(
    eos: AbstractRelativisticEOS,
    conserved: Array,
    inverse_spatial_metric: Array,
    sqrt_spatial_determinant: Array,
    pressure_floor: float,
    /,
) -> Array:
    """Deterministic equation-level pressure recovery for FV interface calls.

    The evidence-rich warm/bracket/atmosphere ladder lives in the solver layer.
    This fixed bracketed route exists so ``SRHDSystem`` satisfies the generic
    conservation-system protocol without importing a solver into equations.
    """

    undensitized = conserved / sqrt_spatial_determinant[..., None]
    mass = undensitized[..., 0]
    momentum = undensitized[..., 1:-1]
    total_energy = undensitized[..., -1] + mass
    momentum_squared = ein.contract(
        "...i,...ij,...j->...", momentum, inverse_spatial_metric, momentum
    )
    momentum_norm = jnp.sqrt(jnp.maximum(momentum_squared, 0.0))
    tiny = jnp.finfo(conserved.dtype).eps * jnp.maximum(jnp.abs(total_energy), 1.0)
    lower = jnp.maximum(
        jnp.asarray(pressure_floor, dtype=conserved.dtype),
        momentum_norm - total_energy + tiny,
    )

    def residual(pressure):
        return _primitive_at_pressure(
            eos, undensitized, inverse_spatial_metric, pressure
        )[1]

    lower_value = residual(lower)
    upper = jnp.maximum(2.0 * lower, jnp.abs(total_energy) + momentum_norm + mass + 1.0)

    def expand(_, current):
        bound, value = current
        candidate = 2.0 * bound
        candidate_value = residual(candidate)
        replace = jnp.signbit(value) == jnp.signbit(lower_value)
        return jnp.where(replace, candidate, bound), jnp.where(
            replace, candidate_value, value
        )

    upper, upper_value = jax.lax.fori_loop(0, 12, expand, (upper, residual(upper)))

    def bisect(_, bracket):
        left, right, left_value, right_value = bracket
        middle = 0.5 * (left + right)
        middle_value = residual(middle)
        replace_right = jnp.signbit(left_value) != jnp.signbit(middle_value)
        return (
            jnp.where(replace_right, left, middle),
            jnp.where(replace_right, middle, right),
            jnp.where(replace_right, left_value, middle_value),
            jnp.where(replace_right, middle_value, right_value),
        )

    lower, upper, lower_value, upper_value = jax.lax.fori_loop(
        0, 64, bisect, (lower, upper, lower_value, upper_value)
    )
    choose_lower = jnp.abs(lower_value) <= jnp.abs(upper_value)
    pressure = jnp.where(choose_lower, lower, upper)
    return _primitive_at_pressure(eos, undensitized, inverse_spatial_metric, pressure)[0]


def _relativistic_bounds(
    evaluation: RelativisticFluidEvaluation,
    inverse_spatial_metric: Array,
    lapse: Array,
    shift: Array,
    covector: Array,
    /,
) -> tuple[Array, Array]:
    velocity = evaluation.velocity
    velocity_squared = evaluation.velocity_squared
    sound_squared = jnp.clip(
        evaluation.sound_speed_squared,
        0.0,
        1.0 - 16.0 * jnp.finfo(velocity.dtype).eps,
    )
    normal_velocity = ein.contract("...i,...i->...", velocity, covector)
    inverse_normal_squared = ein.contract(
        "...i,...ij,...j->...", covector, inverse_spatial_metric, covector
    )
    shift_normal = ein.contract("...i,...i->...", shift, covector)
    denominator = 1.0 - velocity_squared * sound_squared
    radicand = (1.0 - velocity_squared) * (
        inverse_normal_squared * denominator - normal_velocity**2 * (1.0 - sound_squared)
    )
    acoustic = jnp.sqrt(jnp.maximum(sound_squared * radicand, 0.0))
    center = normal_velocity * (1.0 - sound_squared)
    lower = lapse * (center - acoustic) / denominator - shift_normal
    upper = lapse * (center + acoustic) / denominator - shift_normal
    return lower, upper


class SRHDSystem(AbstractConservationSystem):
    """Cartesian special-relativistic hydrodynamics in Valencia variables."""

    eos: AbstractRelativisticEOS
    layout: RelativisticHydrodynamicsLayout
    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)

    def __init__(
        self,
        eos: AbstractRelativisticEOS,
        dimension: int = 1,
        /,
        *,
        density_floor: float = 1.0e-12,
        pressure_floor: float = 1.0e-12,
    ):
        if not isinstance(eos, AbstractRelativisticEOS):
            raise TypeError("eos must be an AbstractRelativisticEOS.")
        if eos.scale.speed_of_light != 1:
            raise ValueError(
                "SRHDSystem requires an EOS declared in geometric c=1 units."
            )
        density_floor_ = float(density_floor)
        pressure_floor_ = float(pressure_floor)
        if (
            not np.isfinite(density_floor_)
            or density_floor_ <= 0.0
            or not np.isfinite(pressure_floor_)
            or pressure_floor_ <= 0.0
        ):
            raise ValueError(
                "SRHD density and pressure floors must be finite and positive."
            )
        layout = RelativisticHydrodynamicsLayout(dimension, densitized=False)
        self.eos = eos
        self.layout = layout
        self.dimension = layout.dimension
        self.component_names = layout.conserved_names
        self.density_floor = density_floor_
        self.pressure_floor = pressure_floor_
        self.system_id = canonical_fingerprint(
            {
                "kind": "special-relativistic-hydrodynamics",
                "eos": eos.eos_id,
                "layout": layout.layout_id,
                "density_floor": density_floor_,
                "pressure_floor": pressure_floor_,
            }
        )

    def primitive_evaluation(
        self, primitive: ArrayLike, /
    ) -> RelativisticFluidEvaluation:
        value = _validate_primitive(primitive, self.layout)
        identity = jnp.broadcast_to(
            jnp.eye(self.dimension, dtype=value.dtype),
            value.shape[:-1] + (self.dimension, self.dimension),
        )
        return _evaluate_primitive(self.eos, value, identity, self.system_id)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        evaluation = self.primitive_evaluation(primitive)
        ones = jnp.ones(evaluation.primitive.shape[:-1], dtype=evaluation.primitive.dtype)
        return _conserved_from_evaluation(evaluation, ones)

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = _validate_conserved(state, self.layout)
        identity = jnp.broadcast_to(
            jnp.eye(self.dimension, dtype=value.dtype),
            value.shape[:-1] + (self.dimension, self.dimension),
        )
        ones = jnp.ones(value.shape[:-1], dtype=value.dtype)
        return _fixed_pressure_recovery(
            self.eos, value, identity, ones, self.pressure_floor
        )

    def pressure(self, state: ArrayLike, /) -> Array:
        primitive = self.conserved_to_primitive(_validate_conserved(state, self.layout))
        return self.eos.evaluate(primitive[..., 0], primitive[..., 1]).pressure

    def primitive_velocity(self, primitive: Array, /) -> Array:
        return _validate_primitive(primitive, self.layout)[
            ..., self.layout.velocity_slice
        ]

    def with_primitive_velocity(self, primitive: Array, velocity: Array, /) -> Array:
        value = _validate_primitive(primitive, self.layout)
        replacement = jnp.asarray(velocity, dtype=value.dtype)
        if replacement.shape != value.shape[:-1] + (self.dimension,):
            raise ValueError("Replacement velocity must match the primitive batch shape.")
        return value.at[..., self.layout.velocity_slice].set(replacement)

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        axis_ = int(axis)
        if not 0 <= axis_ < self.dimension:
            raise ValueError("SRHD flux axis is out of range.")
        value = _validate_conserved(state, self.layout)
        primitive = self.conserved_to_primitive(value)
        evaluation = self.primitive_evaluation(primitive)
        velocity = evaluation.velocity[..., axis_]
        mass_flux = value[..., 0] * velocity
        momentum_flux = value[..., self.layout.momentum_slice] * velocity[..., None]
        momentum_flux = momentum_flux.at[..., axis_].add(evaluation.pressure)
        energy_flux = value[..., -1] * velocity + evaluation.pressure * velocity
        return jnp.concatenate(
            (mass_flux[..., None], momentum_flux, energy_flux[..., None]), axis=-1
        )

    def signal_bounds(
        self, left: Array, right: Array, axis: int, args: Any = None, /
    ) -> tuple[Array, Array]:
        del args
        axis_ = int(axis)
        if not 0 <= axis_ < self.dimension:
            raise ValueError("SRHD signal axis is out of range.")
        left_ = _validate_conserved(left, self.layout)
        right_ = _validate_conserved(right, self.layout)
        batch = jnp.broadcast_shapes(left_.shape[:-1], right_.shape[:-1])
        dtype = jnp.result_type(left_, right_)
        inverse = jnp.broadcast_to(
            jnp.eye(self.dimension, dtype=dtype),
            batch + (self.dimension, self.dimension),
        )
        lapse = jnp.ones(batch, dtype=dtype)
        shift = jnp.zeros(batch + (self.dimension,), dtype=dtype)
        normal = jnp.broadcast_to(
            jax.nn.one_hot(axis_, self.dimension, dtype=dtype),
            batch + (self.dimension,),
        )
        left_bounds = _relativistic_bounds(
            self.primitive_evaluation(self.conserved_to_primitive(left_)),
            inverse,
            lapse,
            shift,
            normal,
        )
        right_bounds = _relativistic_bounds(
            self.primitive_evaluation(self.conserved_to_primitive(right_)),
            inverse,
            lapse,
            shift,
            normal,
        )
        return (
            jnp.minimum(left_bounds[0], right_bounds[0]),
            jnp.maximum(left_bounds[1], right_bounds[1]),
        )

    def max_wave_speed(
        self, left: Array, right: Array, axis: int, args: Any = None, /
    ) -> Array:
        lower, upper = self.signal_bounds(left, right, axis, args)
        return jnp.maximum(jnp.abs(lower), jnp.abs(upper))

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        left_ = _validate_conserved(left, self.layout)
        right_ = _validate_conserved(right, self.layout)
        normal_ = jnp.asarray(normal, dtype=jnp.result_type(left_, right_))
        if normal_.shape != jnp.broadcast_shapes(left_.shape[:-1], right_.shape[:-1]) + (
            self.dimension,
        ):
            raise ValueError("SRHD normal must exactly match the face batch shape.")
        batch = normal_.shape[:-1]
        inverse = jnp.broadcast_to(
            jnp.eye(self.dimension, dtype=normal_.dtype),
            batch + (self.dimension, self.dimension),
        )
        lapse = jnp.ones(batch, dtype=normal_.dtype)
        shift = jnp.zeros_like(normal_)
        left_bounds = _relativistic_bounds(
            self.primitive_evaluation(self.conserved_to_primitive(left_)),
            inverse,
            lapse,
            shift,
            normal_,
        )
        right_bounds = _relativistic_bounds(
            self.primitive_evaluation(self.conserved_to_primitive(right_)),
            inverse,
            lapse,
            shift,
            normal_,
        )
        return (
            jnp.minimum(left_bounds[0], right_bounds[0]),
            jnp.maximum(left_bounds[1], right_bounds[1]),
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        value = _validate_conserved(state, self.layout)
        axis_ = int(axis)
        if not 0 <= axis_ < self.dimension:
            raise ValueError("SRHD reflection axis is out of range.")
        return value.at[..., 1 + axis_].multiply(-1.0)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        value = _validate_conserved(state, self.layout)
        normal_ = jnp.asarray(normal, dtype=value.dtype)
        if normal_.shape != value.shape[:-1] + (self.dimension,):
            raise ValueError("SRHD reflection normal must match the state batch shape.")
        norm_squared = ein.contract("...i,...i->...", normal_, normal_)
        unit = normal_ / jnp.sqrt(norm_squared)[..., None]
        momentum = value[..., self.layout.momentum_slice]
        reflected = (
            momentum
            - 2.0 * ein.contract("...i,...i->...", momentum, unit)[..., None] * unit
        )
        return value.at[..., self.layout.momentum_slice].set(reflected)

    def admissible(self, state: Array, /) -> Array:
        value = _validate_conserved(state, self.layout)
        evaluation = self.primitive_evaluation(self.conserved_to_primitive(value))
        return (
            evaluation.physically_valid
            & (evaluation.rest_mass_density >= self.density_floor)
            & (evaluation.pressure >= self.pressure_floor)
        )


class ValenciaGRHDSystem(StrictModule, NonTrainableState):
    """Valencia GRHD on explicit ADM geometry evaluations.

    This equation object evolves no metric. Every flux, source, recovery, and
    stress-energy projection consumes geometry from the same synchronized ADM
    stage supplied by the caller.
    """

    eos: AbstractRelativisticEOS
    convention: RelativityConvention
    layout: RelativisticHydrodynamicsLayout
    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        eos: AbstractRelativisticEOS,
        dimension: int = 3,
        /,
        *,
        density_floor: float = 1.0e-12,
        pressure_floor: float = 1.0e-12,
        convention: RelativityConvention | None = None,
    ):
        if not isinstance(eos, AbstractRelativisticEOS):
            raise TypeError("eos must be an AbstractRelativisticEOS.")
        if eos.scale.speed_of_light != 1:
            raise ValueError(
                "ValenciaGRHDSystem requires an EOS declared in geometric c=1 units."
            )
        if int(dimension) != 3:
            raise ValueError(
                "ValenciaGRHDSystem uses the three-dimensional ADM exchange layout."
            )
        density_floor_ = float(density_floor)
        pressure_floor_ = float(pressure_floor)
        if (
            not np.isfinite(density_floor_)
            or density_floor_ <= 0.0
            or not np.isfinite(pressure_floor_)
            or pressure_floor_ <= 0.0
        ):
            raise ValueError(
                "GRHD density and pressure floors must be finite and positive."
            )
        convention_ = RelativityConvention() if convention is None else convention
        if not isinstance(convention_, RelativityConvention):
            raise TypeError("convention must be a RelativityConvention or None.")
        layout = RelativisticHydrodynamicsLayout(dimension, densitized=True)
        self.eos = eos
        self.convention = convention_
        self.layout = layout
        self.density_floor = density_floor_
        self.pressure_floor = pressure_floor_
        self.dimension = layout.dimension
        self.component_names = layout.conserved_names
        self.system_id = canonical_fingerprint(
            {
                "kind": "valencia-general-relativistic-hydrodynamics",
                "eos": eos.eos_id,
                "convention": convention_.convention_id,
                "layout": layout.layout_id,
                "density_floor": density_floor_,
                "pressure_floor": pressure_floor_,
                "source": "adm-valencia-eulerian-v1",
            }
        )

    @property
    def component_count(self) -> int:
        return self.layout.component_count

    def primitive_evaluation(
        self, primitive: ArrayLike, geometry: ADMGridGeometry, /
    ) -> RelativisticFluidEvaluation:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be an ADMGridGeometry.")
        if (
            geometry.scale_id != self.eos.scale.scale_id
            or geometry.convention_id != self.convention.convention_id
        ):
            raise ValueError(
                "ADM geometry scale/convention must match the Valencia system."
            )
        value = _validate_primitive(primitive, self.layout)
        if geometry.spatial_metric.shape != value.shape[:-1] + (
            self.dimension,
            self.dimension,
        ):
            raise ValueError("ADM spatial metric must exactly match the primitive batch.")
        return _evaluate_primitive(
            self.eos, value, geometry.spatial_metric, self.system_id
        )

    def primitive_to_conserved(
        self, primitive: ArrayLike, geometry: ADMGridGeometry, /
    ) -> Array:
        evaluation = self.primitive_evaluation(primitive, geometry)
        return _conserved_from_evaluation(evaluation, geometry.sqrt_det_spatial_metric)

    def primitive_from_pressure(
        self,
        conserved: ArrayLike,
        geometry: ADMGridGeometry,
        pressure: ArrayLike,
        /,
    ) -> tuple[Array, Array, Array]:
        value = _validate_conserved(conserved, self.layout)
        undensitized = value / geometry.sqrt_det_spatial_metric[..., None]
        return _primitive_at_pressure(
            self.eos,
            undensitized,
            geometry.inverse_spatial_metric,
            jnp.asarray(pressure, dtype=value.dtype),
        )

    def physical_flux_from_primitive(
        self, primitive: ArrayLike, geometry: ADMGridGeometry, axis: int, /
    ) -> Array:
        axis_ = int(axis)
        if not 0 <= axis_ < self.dimension:
            raise ValueError("GRHD flux axis is out of range.")
        evaluation = self.primitive_evaluation(primitive, geometry)
        conserved = _conserved_from_evaluation(
            evaluation, geometry.sqrt_det_spatial_metric
        )
        transport = (
            geometry.alpha * evaluation.velocity[..., axis_]
            - geometry.beta_contravariant[..., axis_]
        )
        mass_flux = conserved[..., 0] * transport
        momentum_flux = conserved[..., self.layout.momentum_slice] * transport[..., None]
        pressure_term = (
            geometry.alpha * geometry.sqrt_det_spatial_metric * evaluation.pressure
        )
        momentum_flux = momentum_flux.at[..., axis_].add(pressure_term)
        energy_flux = (
            conserved[..., -1] * transport
            + pressure_term * evaluation.velocity[..., axis_]
        )
        return jnp.concatenate(
            (mass_flux[..., None], momentum_flux, energy_flux[..., None]), axis=-1
        )

    def characteristic_bounds_from_primitive(
        self,
        left_primitive: ArrayLike,
        right_primitive: ArrayLike,
        geometry: ADMGridGeometry,
        covector: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        covector_ = jnp.asarray(covector)
        expected = jnp.broadcast_shapes(
            jnp.asarray(left_primitive).shape[:-1],
            jnp.asarray(right_primitive).shape[:-1],
        ) + (self.dimension,)
        if covector_.shape != expected:
            raise ValueError("GRHD characteristic covector must match the face batch.")
        left = _relativistic_bounds(
            self.primitive_evaluation(left_primitive, geometry),
            geometry.inverse_spatial_metric,
            geometry.alpha,
            geometry.beta_contravariant,
            covector_,
        )
        right = _relativistic_bounds(
            self.primitive_evaluation(right_primitive, geometry),
            geometry.inverse_spatial_metric,
            geometry.alpha,
            geometry.beta_contravariant,
            covector_,
        )
        return jnp.minimum(left[0], right[0]), jnp.maximum(left[1], right[1])

    def source_from_primitive(
        self, primitive: ArrayLike, source_geometry: ValenciaGeometrySource, /
    ) -> Array:
        """Return the densitized Valencia geometric source ``[0,S_j,S_tau]``."""

        if not isinstance(source_geometry, ValenciaGeometrySource):
            raise TypeError("source_geometry must be a ValenciaGeometrySource.")
        geometry = source_geometry.geometry
        evaluation = self.primitive_evaluation(primitive, geometry)
        rho_h_w2 = (
            evaluation.rest_mass_density
            * evaluation.specific_enthalpy
            * evaluation.lorentz_factor**2
        )
        eulerian_energy = rho_h_w2 - evaluation.pressure
        momentum_covector = rho_h_w2[..., None] * evaluation.covariant_velocity
        stress_contravariant = (
            rho_h_w2[..., None, None]
            * ein.contract("...i,...j->...ij", evaluation.velocity, evaluation.velocity)
            + evaluation.pressure[..., None, None] * geometry.inverse_spatial_metric
        )
        momentum_source = geometry.sqrt_det_spatial_metric[..., None] * (
            0.5
            * geometry.alpha[..., None]
            * ein.contract(
                "...ik,...jik->...j",
                stress_contravariant,
                source_geometry.spatial_metric_gradient,
            )
            + ein.contract(
                "...i,...ji->...j",
                momentum_covector,
                source_geometry.beta_gradient,
            )
            - eulerian_energy[..., None] * source_geometry.alpha_gradient
        )
        raised_momentum = ein.contract(
            "...ij,...j->...i", geometry.inverse_spatial_metric, momentum_covector
        )
        energy_source = geometry.sqrt_det_spatial_metric * (
            -self.convention.extrinsic_curvature_sign
            * geometry.alpha
            * ein.contract(
                "...ij,...ij->...", stress_contravariant, geometry.extrinsic_curvature
            )
            - ein.contract(
                "...i,...i->...", raised_momentum, source_geometry.alpha_gradient
            )
        )
        return jnp.concatenate(
            (
                jnp.zeros_like(energy_source)[..., None],
                momentum_source,
                energy_source[..., None],
            ),
            axis=-1,
        )

    def stress_energy_projection(
        self, primitive: ArrayLike, geometry: ADMGridGeometry, /
    ) -> StressEnergyProjection:
        """Project a perfect fluid onto the Eulerian ADM normal/frame."""

        evaluation = self.primitive_evaluation(primitive, geometry)
        rho_h_w2 = (
            evaluation.rest_mass_density
            * evaluation.specific_enthalpy
            * evaluation.lorentz_factor**2
        )
        energy = rho_h_w2 - evaluation.pressure
        momentum = rho_h_w2[..., None] * evaluation.covariant_velocity
        stress = (
            rho_h_w2[..., None, None]
            * ein.contract(
                "...i,...j->...ij",
                evaluation.covariant_velocity,
                evaluation.covariant_velocity,
            )
            + evaluation.pressure[..., None, None] * geometry.spatial_metric
        )
        zeros = jnp.zeros_like(energy)
        return StressEnergyProjection(
            energy,
            momentum,
            stress,
            geometry.active,
            evaluation.physically_valid & geometry.physically_valid,
            zeros,
            zeros,
            snapshot_token=geometry.snapshot_token,
            geometry_lineage_id=geometry.geometry_lineage_id,
            convention_id=geometry.convention_id,
            scale_id=geometry.scale_id,
            topology_id=geometry.topology_id,
            projection_id=canonical_fingerprint(
                {
                    "kind": "perfect-fluid-stress-energy-projection",
                    "system": self.system_id,
                    "geometry_lineage": geometry.geometry_lineage_id,
                }
            ),
        )


__all__ = [
    "RelativisticFluidEvaluation",
    "ValenciaGeometrySource",
    "RelativisticHydrodynamicsLayout",
    "SRHDSystem",
    "ValenciaGRHDSystem",
]
