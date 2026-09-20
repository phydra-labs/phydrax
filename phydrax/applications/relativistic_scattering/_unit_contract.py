#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact relativistic units and local-frame bindings.

Four-momenta use contravariant ``(E, c p_x, c p_y, c p_z)`` components.  This
keeps every component in the declared energy unit without silently setting
``c`` or ``hbar`` to one.  Natural-unit arrays are dimensionless arrays obtained
only through an explicit positive reference energy.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein
import phydrax.linalg as linalg

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix import (
    ADMGridGeometry,
    CoordinateChart,
    MetricDomainEvidence,
    MetricInnerProductEvidence,
    OrthonormalTetrad,
    RelativityConvention,
    tetrad_project_vector,
    tetrad_reconstruct_vector,
)
from ...units import derived_unit, UnitDefinition


PhaseSpaceNormalization: TypeAlias = Literal["lorentz-invariant-2E"]
SMatrixNormalization: TypeAlias = Literal["covariant-delta"]
SpinNormalization: TypeAlias = Literal[
    "average-initial-sum-final",
    "sum-all",
    "density-matrix-explicit",
]
ColorNormalization: TypeAlias = Literal[
    "average-initial-sum-final",
    "sum-all",
    "density-matrix-explicit",
]
PolarizationNormalization: TypeAlias = Literal[
    "physical-helicity",
    "covariant-gauge-with-ward-check",
    "density-matrix-explicit",
]
IdenticalParticleNormalization: TypeAlias = Literal[
    "factorial-symmetry",
    "ordered-labeled-final-state",
]

_PHASE_SPACE_NORMALIZATIONS = frozenset(("lorentz-invariant-2E",))
_S_MATRIX_NORMALIZATIONS = frozenset(("covariant-delta",))
_SPIN_NORMALIZATIONS = frozenset(
    ("average-initial-sum-final", "sum-all", "density-matrix-explicit")
)
_COLOR_NORMALIZATIONS = _SPIN_NORMALIZATIONS
_POLARIZATION_NORMALIZATIONS = frozenset(
    (
        "physical-helicity",
        "covariant-gauge-with-ward-check",
        "density-matrix-explicit",
    )
)
_IDENTICAL_PARTICLE_NORMALIZATIONS = frozenset(
    ("factorial-symmetry", "ordered-labeled-final-state")
)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _normalization(value: str, allowed: frozenset[str], name: str, /) -> str:
    normalized = _identifier(value, name)
    if normalized not in allowed:
        raise ValueError(f"Unknown {name}: {normalized!r}.")
    return normalized


def _real_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.result_type(array, jnp.asarray(1.0)))
    return array


def _apply_fraction(value: ArrayLike, factor: Fraction, /) -> Array:
    array = _real_array(value, "Relativistic unit conversion input")
    mantissa, exponent = math.frexp(float(factor))
    scalar = jnp.asarray(mantissa, dtype=array.dtype)
    return jnp.ldexp(array * scalar, exponent)


def _positive_reference_energy(value: ArrayLike, /) -> Array:
    energy = _real_array(value, "reference_energy")
    return eqx.error_if(
        energy,
        jnp.any(~jnp.isfinite(energy) | (energy <= 0.0)),
        "reference_energy must be finite and strictly positive.",
    )


def _nonnegative_rest_mass(value: ArrayLike, /) -> Array:
    mass = _real_array(value, "rest_mass")
    return eqx.error_if(
        mass,
        jnp.any(~jnp.isfinite(mass) | (mass < 0.0)),
        "rest_mass must be finite and non-negative.",
    )


class RelativisticUnitContract(StrictModule, NonTrainableState):
    """Complete unit and normalization identity for relativistic amplitudes.

    ``scale`` owns exact rational values for ``c`` and ``hbar``.  The physical
    momentum unit is the base ``mass*length/time`` unit, while represented
    four-vectors have the energy-unit components ``(E, c p)``.  No operation
    infers a unit system from numeric magnitudes.
    """

    scale: RelativityScaleContract
    convention: RelativityConvention
    metric: Array
    energy_unit: UnitDefinition = eqx.field(static=True)
    momentum_unit: UnitDefinition = eqx.field(static=True)
    four_vector_convention: str = eqx.field(static=True)
    phase_space_normalization: PhaseSpaceNormalization = eqx.field(static=True)
    s_matrix_normalization: SMatrixNormalization = eqx.field(static=True)
    spin_normalization: SpinNormalization = eqx.field(static=True)
    color_normalization: ColorNormalization = eqx.field(static=True)
    polarization_normalization: PolarizationNormalization = eqx.field(static=True)
    identical_particle_normalization: IdenticalParticleNormalization = eqx.field(
        static=True
    )
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        /,
        *,
        phase_space_normalization: PhaseSpaceNormalization = "lorentz-invariant-2E",
        s_matrix_normalization: SMatrixNormalization = "covariant-delta",
        spin_normalization: SpinNormalization = "average-initial-sum-final",
        color_normalization: ColorNormalization = "average-initial-sum-final",
        polarization_normalization: PolarizationNormalization = "physical-helicity",
        identical_particle_normalization: IdenticalParticleNormalization = "factorial-symmetry",
    ):
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        if not isinstance(convention, RelativityConvention):
            raise TypeError("convention must be a RelativityConvention.")
        if not scale.quantum_constants_explicit:
            raise ValueError(
                "Relativistic unit contracts require explicitly declared c and hbar."
            )
        phase_space = _normalization(
            phase_space_normalization,
            _PHASE_SPACE_NORMALIZATIONS,
            "phase-space normalization",
        )
        s_matrix = _normalization(
            s_matrix_normalization,
            _S_MATRIX_NORMALIZATIONS,
            "S-matrix normalization",
        )
        spin = _normalization(
            spin_normalization,
            _SPIN_NORMALIZATIONS,
            "spin normalization",
        )
        color = _normalization(
            color_normalization,
            _COLOR_NORMALIZATIONS,
            "color normalization",
        )
        polarization = _normalization(
            polarization_normalization,
            _POLARIZATION_NORMALIZATIONS,
            "polarization normalization",
        )
        identical = _normalization(
            identical_particle_normalization,
            _IDENTICAL_PARTICLE_NORMALIZATIONS,
            "identical-particle normalization",
        )
        dimensional = scale.dimensional_scale
        momentum = derived_unit(
            (
                f"{dimensional.mass_unit.symbol}*{dimensional.length_unit.symbol}/{dimensional.time_unit.symbol}"
            ),
            (
                (dimensional.mass_unit, 1),
                (dimensional.length_unit, 1),
                (dimensional.time_unit, -1),
            ),
        )
        timelike = 1.0 if convention.metric_signature == "mostly_minus" else -1.0
        metric = jnp.diag(jnp.asarray((timelike, -timelike, -timelike, -timelike)))

        self.scale = scale
        self.convention = convention
        self.metric = metric
        self.energy_unit = scale.energy_unit
        self.momentum_unit = momentum
        self.four_vector_convention = "contravariant-(E,c*p)"
        self.phase_space_normalization = phase_space
        self.s_matrix_normalization = s_matrix
        self.spin_normalization = spin
        self.color_normalization = color
        self.polarization_normalization = polarization
        self.identical_particle_normalization = identical
        self.contract_id = canonical_fingerprint(
            {
                "kind": "relativistic-unit-contract",
                "scale": scale.scale_id,
                "convention": convention.convention_id,
                "four_vector": self.four_vector_convention,
                "energy_unit": self.energy_unit.unit_id,
                "momentum_unit": momentum.unit_id,
                "c": [
                    scale.speed_of_light.numerator,
                    scale.speed_of_light.denominator,
                ],
                "hbar": [
                    scale.reduced_planck_constant.numerator,
                    scale.reduced_planck_constant.denominator,
                ],
                "phase_space_normalization": phase_space,
                "s_matrix_normalization": s_matrix,
                "spin_normalization": spin,
                "color_normalization": color,
                "polarization_normalization": polarization,
                "identical_particle_normalization": identical,
            }
        )

    @property
    def speed_of_light(self) -> Fraction:
        return self.scale.speed_of_light

    @property
    def reduced_planck_constant(self) -> Fraction:
        return self.scale.reduced_planck_constant

    def assemble_four_momentum(
        self,
        energy: ArrayLike,
        spatial_momentum: ArrayLike,
        /,
    ) -> Array:
        """Assemble physical ``energy`` and ``p`` into ``(E, c p)``."""
        energy_ = _real_array(energy, "energy")
        momentum = _real_array(spatial_momentum, "spatial_momentum")
        if momentum.shape[-1:] != (3,):
            raise ValueError("spatial_momentum must have trailing dimension three.")
        leading_shape = jnp.broadcast_shapes(energy_.shape, momentum.shape[:-1])
        energy_ = jnp.broadcast_to(energy_, leading_shape)
        momentum = jnp.broadcast_to(momentum, leading_shape + (3,))
        spatial_energy = _apply_fraction(momentum, self.speed_of_light)
        return jnp.concatenate((energy_[..., None], spatial_energy), axis=-1)

    def split_four_momentum(self, four_momentum: ArrayLike, /) -> tuple[Array, Array]:
        """Invert :meth:`assemble_four_momentum` without setting ``c=1``."""
        value = _real_array(four_momentum, "four_momentum")
        if value.shape[-1:] != (4,):
            raise ValueError("four_momentum must have trailing dimension four.")
        momentum = _apply_fraction(value[..., 1:], 1 / self.speed_of_light)
        return value[..., 0], momentum

    def to_natural_four_momentum(
        self,
        four_momentum: ArrayLike,
        reference_energy: ArrayLike,
        /,
    ) -> Array:
        """Nondimensionalize ``(E,c p)`` by one explicit physical energy."""
        value = _real_array(four_momentum, "four_momentum")
        if value.shape[-1:] != (4,):
            raise ValueError("four_momentum must have trailing dimension four.")
        reference = _positive_reference_energy(reference_energy)
        return value / jnp.expand_dims(reference, axis=-1)

    def from_natural_four_momentum(
        self,
        natural_four_momentum: ArrayLike,
        reference_energy: ArrayLike,
        /,
    ) -> Array:
        """Restore energy-unit ``(E,c p)`` components from natural units."""
        value = _real_array(natural_four_momentum, "natural_four_momentum")
        if value.shape[-1:] != (4,):
            raise ValueError("natural_four_momentum must have trailing dimension four.")
        reference = _positive_reference_energy(reference_energy)
        return value * jnp.expand_dims(reference, axis=-1)

    def mass_to_rest_energy(self, mass: ArrayLike, /) -> Array:
        """Convert mass to ``m c^2`` in the contract energy unit."""
        return _apply_fraction(mass, self.speed_of_light**2)

    def rest_energy_to_mass(self, rest_energy: ArrayLike, /) -> Array:
        """Invert :meth:`mass_to_rest_energy`."""
        return _apply_fraction(rest_energy, 1 / self.speed_of_light**2)

    def wave_number_to_energy(self, wave_number: ArrayLike, /) -> Array:
        """Convert angular wave number to energy using exactly ``hbar*c``."""
        return _apply_fraction(
            wave_number,
            self.reduced_planck_constant * self.speed_of_light,
        )

    def energy_to_wave_number(self, energy: ArrayLike, /) -> Array:
        """Invert :meth:`wave_number_to_energy`."""
        return _apply_fraction(
            energy,
            1 / (self.reduced_planck_constant * self.speed_of_light),
        )

    def angular_frequency_to_energy(self, angular_frequency: ArrayLike, /) -> Array:
        """Convert angular frequency to energy using exactly ``hbar``."""
        return _apply_fraction(angular_frequency, self.reduced_planck_constant)

    def energy_to_angular_frequency(self, energy: ArrayLike, /) -> Array:
        """Invert :meth:`angular_frequency_to_energy`."""
        return _apply_fraction(energy, 1 / self.reduced_planck_constant)

    def lorentz_scalar(
        self,
        left: ArrayLike,
        right: ArrayLike,
        /,
    ) -> Array:
        """Contract two local ``(E,c p)`` four-vectors with the declared metric."""
        left_ = _real_array(left, "left four-vector")
        right_ = _real_array(right, "right four-vector")
        if left_.shape[-1:] != (4,) or right_.shape[-1:] != (4,):
            raise ValueError("Lorentz scalars require trailing four-vector axes.")
        return ein.contract("...i,ij,...j->...", left_, self.metric, right_)

    def mass_shell_residual(
        self,
        four_momentum: ArrayLike,
        rest_mass: ArrayLike,
        /,
    ) -> Array:
        """Return ``P.P - sign_time*(m c^2)^2`` in squared-energy units."""
        value = _real_array(four_momentum, "four_momentum")
        if value.shape[-1:] != (4,):
            raise ValueError("four_momentum must have trailing dimension four.")
        rest_energy = self.mass_to_rest_energy(_nonnegative_rest_mass(rest_mass))
        timelike_sign = self.metric[0, 0]
        return self.lorentz_scalar(value, value) - timelike_sign * rest_energy**2

    def mass_shell_admissible(
        self,
        four_momentum: ArrayLike,
        rest_mass: ArrayLike,
        /,
        *,
        relative_tolerance: float | None = None,
    ) -> Array:
        """Test a future mass shell with a scale-relative, unit-free tolerance.

        Unlike a ``max(1, ...)`` tolerance, this criterion does not change when
        the energy unit changes.  The exact zero four-vector therefore has zero
        absolute tolerance rather than an implicit one-energy-unit allowance.
        """
        value = _real_array(four_momentum, "four_momentum")
        if value.shape[-1:] != (4,):
            raise ValueError("four_momentum must have trailing dimension four.")
        if relative_tolerance is None:
            tolerance = 64.0 * float(jnp.finfo(value.dtype).eps)
        else:
            tolerance = float(relative_tolerance)
            if not math.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError("relative_tolerance must be finite and non-negative.")
        rest_mass_ = _nonnegative_rest_mass(rest_mass)
        rest_energy = self.mass_to_rest_energy(rest_mass_)
        squared_scale = jnp.maximum(
            jnp.max(jnp.abs(value) ** 2, axis=-1),
            jnp.abs(rest_energy) ** 2,
        )
        future_sign = self.convention.future_time_orientation * value[..., 0]
        return (future_sign >= 0.0) & (
            jnp.abs(self.mass_shell_residual(value, rest_mass_))
            <= tolerance * squared_scale
        )

    def invariant_phase_space_weight(self, four_momentum: ArrayLike, /) -> Array:
        """Return the declared ``1 / ((2*pi)^3 2E)`` one-particle weight.

        This multiplies ``d^3(c p)``.  It never clips zero or negative energy;
        such a value is outside the declared future-shell measure and errors.
        """
        value = _real_array(four_momentum, "four_momentum")
        if value.shape[-1:] != (4,):
            raise ValueError("four_momentum must have trailing dimension four.")
        energy = eqx.error_if(
            value[..., 0],
            jnp.any(~jnp.isfinite(value) | (value[..., 0, None] <= 0.0)),
            "Invariant phase space requires finite future-directed four-momenta.",
        )
        return 1.0 / (2.0 * (2.0 * jnp.pi) ** 3 * energy)


def _eulerian_adm_tetrad(
    geometry: ADMGridGeometry,
    units: RelativisticUnitContract,
    /,
    *,
    observer_id: str,
    orientation_id: str,
    coordinate_names: tuple[str, str, str, str],
    tolerance: float | None,
) -> OrthonormalTetrad:
    if (
        not isinstance(coordinate_names, tuple)
        or len(coordinate_names) != 4
        or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in coordinate_names
        )
        or len(set(coordinate_names)) != 4
    ):
        raise ValueError("coordinate_names must contain four distinct canonical labels.")
    tolerance_ = (
        128.0 * float(jnp.finfo(geometry.spatial_metric.dtype).eps)
        if tolerance is None
        else float(tolerance)
    )
    if not math.isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("Eulerian tetrad tolerance must be finite and positive.")
    spatial_scale = jnp.max(
        jnp.abs(geometry.spatial_metric),
        axis=(-2, -1),
    )
    scale_valid = jnp.isfinite(spatial_scale) & (spatial_scale > 0.0)
    safe_scale = jnp.where(scale_valid, spatial_scale, 1.0)
    normalized_spatial_metric = geometry.spatial_metric / safe_scale[..., None, None]
    inverse_sqrt = linalg.hermitian_inverse_sqrt(
        normalized_spatial_metric,
        tolerance=tolerance_,
    )
    square_root = linalg.hermitian_sqrt(
        normalized_spatial_metric,
        tolerance=tolerance_,
    )
    spectral_valid = inverse_sqrt.valid & square_root.valid & scale_valid
    convention = units.convention
    future_sign = convention.future_time_orientation
    final_spatial_sign = convention.spacetime_orientation * future_sign
    row_signs = jnp.asarray(
        (1.0, 1.0, final_spatial_sign),
        dtype=geometry.spatial_metric.dtype,
    )
    root_scale = jnp.sqrt(safe_scale)[..., None, None]
    triad = inverse_sqrt.value / root_scale * row_signs[:, None]
    spatial_dual = square_root.value * root_scale * row_signs[:, None]
    inverse_lapse = 1.0 / geometry.alpha
    normal = future_sign * jnp.concatenate(
        (
            inverse_lapse[..., None],
            -geometry.beta_contravariant * inverse_lapse[..., None],
        ),
        axis=-1,
    )
    spatial_vectors = jnp.concatenate(
        (
            jnp.zeros(geometry.leading_shape + (3, 1), dtype=triad.dtype),
            triad,
        ),
        axis=-1,
    )
    vectors = jnp.concatenate((normal[..., None, :], spatial_vectors), axis=-2)
    time_dual = future_sign * jnp.concatenate(
        (
            geometry.alpha[..., None],
            jnp.zeros(geometry.leading_shape + (3,), dtype=triad.dtype),
        ),
        axis=-1,
    )
    spatial_dual_time = ein.contract(
        "...ai,...i->...a",
        spatial_dual,
        geometry.beta_contravariant,
    )
    spatial_covectors = jnp.concatenate(
        (spatial_dual_time[..., None], spatial_dual),
        axis=-1,
    )
    dual = jnp.concatenate((time_dual[..., None, :], spatial_covectors), axis=-2)

    spatial_sign = units.metric[1, 1]
    temporal_sign = units.metric[0, 0]
    shift_covector = ein.contract(
        "...ij,...j->...i",
        geometry.spatial_metric,
        geometry.beta_contravariant,
    )
    shift_squared = ein.contract(
        "...i,...i->...",
        geometry.beta_contravariant,
        shift_covector,
    )
    metric_time = jnp.concatenate(
        (
            (temporal_sign * geometry.alpha**2 + spatial_sign * shift_squared)[..., None],
            spatial_sign * shift_covector,
        ),
        axis=-1,
    )
    metric_space = jnp.concatenate(
        (
            (spatial_sign * shift_covector)[..., None],
            spatial_sign * geometry.spatial_metric,
        ),
        axis=-1,
    )
    metric = jnp.concatenate((metric_time[..., None, :], metric_space), axis=-2)
    gram = ein.contract("...ai,...ij,...bj->...ab", vectors, metric, vectors)
    chart = CoordinateChart(geometry.chart_id, coordinate_names)
    tetrad_id = canonical_fingerprint(
        {
            "kind": "eulerian-adm-orthonormal-tetrad",
            "chart": geometry.chart_id,
            "topology": geometry.topology_id,
            "geometry_lineage": geometry.geometry_lineage_id,
            "unit_contract": units.contract_id,
            "observer": observer_id,
            "orientation": orientation_id,
            "basis": "symmetric-positive-inverse-square-root",
            "coordinate_names": list(coordinate_names),
            "tolerance": tolerance_,
        }
    )
    margin = jnp.minimum(
        geometry.alpha,
        inverse_sqrt.spectrum.minimum_eigenvalue,
    )
    domain = MetricDomainEvidence.from_margin(
        margin,
        chart=chart,
        domain_id=canonical_fingerprint(
            {"kind": "eulerian-adm-tetrad-domain", "tetrad": tetrad_id}
        ),
        extra_valid=geometry.physically_valid,
    )
    inner_products = MetricInnerProductEvidence(
        gram,
        units.metric,
        domain_valid=domain.physically_valid & spectral_valid,
        derivative_valid=domain.derivative_valid & spectral_valid,
        tolerance=tolerance_,
        evidence_id=canonical_fingerprint(
            {"kind": "eulerian-adm-tetrad-inner-products", "tetrad": tetrad_id}
        ),
    )
    finite = (
        jnp.all(jnp.isfinite(vectors), axis=(-2, -1))
        & jnp.all(jnp.isfinite(dual), axis=(-2, -1))
        & inner_products.finite
    )
    future_directed = convention.future_time_orientation * vectors[..., 0, 0] > 0.0
    physical = (
        finite
        & geometry.physically_valid
        & domain.physically_valid
        & spectral_valid
        & future_directed
    )
    qualified = physical & inner_products.qualified
    orientation = jnp.full(
        geometry.leading_shape,
        convention.spacetime_orientation,
        dtype=vectors.dtype,
    )
    return OrthonormalTetrad(
        vectors,
        dual,
        inner_products,
        orientation,
        jnp.zeros(geometry.leading_shape, dtype=vectors.dtype),
        jnp.maximum(
            -convention.future_time_orientation * vectors[..., 0, 0],
            0.0,
        ),
        finite,
        domain.physically_valid,
        physical,
        qualified,
        domain.derivative_valid & spectral_valid & physical,
        convention=convention,
        domain=domain,
        tetrad_id=tetrad_id,
    )


class LocalRelativisticFramePlan(StrictModule, NonTrainableState):
    """Exact local orthonormal frame bound to one ADM stage snapshot.

    Coordinate/local maps delegate to the package tetrad owner.  Construction
    refuses scale, convention, lane-shape, orientation, or validity mismatch;
    an inactive ADM lane remains masked rather than being treated as evidence.
    """

    geometry: ADMGridGeometry
    tetrad: OrthonormalTetrad
    units: RelativisticUnitContract
    observer_coordinates: Array
    time: Array
    scale_factor: Array
    admissible: Array
    observer_id: str = eqx.field(static=True)
    orientation_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: ADMGridGeometry,
        tetrad: OrthonormalTetrad,
        units: RelativisticUnitContract,
        observer_coordinates: ArrayLike,
        time: ArrayLike,
        scale_factor: ArrayLike,
        /,
        *,
        observer_id: str,
        orientation_id: str,
    ):
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be an ADMGridGeometry.")
        if not isinstance(tetrad, OrthonormalTetrad):
            raise TypeError("tetrad must be an OrthonormalTetrad.")
        if not isinstance(units, RelativisticUnitContract):
            raise TypeError("units must be a RelativisticUnitContract.")
        observer = _identifier(observer_id, "observer_id")
        orientation = _identifier(orientation_id, "orientation_id")
        if geometry.scale_id != units.scale.scale_id:
            raise ValueError("ADM geometry and relativistic units use different scales.")
        if (
            geometry.convention_id != units.convention.convention_id
            or tetrad.convention.convention_id != units.convention.convention_id
        ):
            raise ValueError(
                "ADM geometry, tetrad, and relativistic units use different conventions."
            )
        leading_shape = geometry.leading_shape
        if tetrad.vectors.shape[:-2] != leading_shape:
            raise ValueError("ADM geometry and tetrad lane shapes must match exactly.")
        coordinates = _real_array(observer_coordinates, "observer_coordinates")
        if coordinates.shape == (4,):
            coordinates = jnp.broadcast_to(coordinates, leading_shape + (4,))
        elif coordinates.shape != leading_shape + (4,):
            raise ValueError(
                "observer_coordinates must have shape (4,) or ADM lane shape plus (4,)."
            )
        time_ = _real_array(time, "time")
        scale_ = _real_array(scale_factor, "scale_factor")
        try:
            time_ = jnp.broadcast_to(time_, leading_shape)
            scale_ = jnp.broadcast_to(scale_, leading_shape)
        except ValueError as error:
            raise ValueError(
                "time and scale_factor must broadcast exactly over ADM lanes."
            ) from error
        numeric_valid = (
            jnp.all(jnp.isfinite(coordinates), axis=-1)
            & jnp.isfinite(time_)
            & jnp.isfinite(scale_)
            & (scale_ > 0.0)
        )
        admissible = (
            geometry.active & geometry.physically_valid & tetrad.qualified & numeric_valid
        )
        refusal = ~jnp.any(geometry.active) | jnp.any(geometry.active & ~admissible)
        admissible = eqx.error_if(
            admissible,
            refusal,
            "Active local-frame lanes require valid ADM, tetrad, time, and scale evidence.",
        )

        self.geometry = geometry
        self.tetrad = tetrad
        self.units = units
        self.observer_coordinates = coordinates
        self.time = time_
        self.scale_factor = scale_
        self.admissible = admissible
        self.observer_id = observer
        self.orientation_id = orientation
        self.frame_id = canonical_fingerprint(
            {
                "kind": "local-relativistic-frame-plan",
                "unit_contract": units.contract_id,
                "chart": geometry.chart_id,
                "topology": geometry.topology_id,
                "geometry_lineage": geometry.geometry_lineage_id,
                "tetrad": tetrad.tetrad_id,
                "observer": observer,
                "orientation": orientation,
            }
        )

    @classmethod
    def from_adm(
        cls,
        geometry: ADMGridGeometry,
        units: RelativisticUnitContract,
        observer_coordinates: ArrayLike,
        time: ArrayLike,
        scale_factor: ArrayLike,
        /,
        *,
        observer_id: str,
        orientation_id: str,
        coordinate_names: tuple[str, str, str, str] = ("t", "x", "y", "z"),
        tolerance: float | None = None,
    ) -> LocalRelativisticFramePlan:
        """Build the future Eulerian frame of an arbitrary ADM stage.

        The symmetric positive inverse square root of the spatial metric fixes
        a deterministic oriented triad without a host Cholesky or coordinate
        basis assumption.  All numerical construction and refusal evidence are
        JAX-safe; ``frame_id`` names the static lineage while ``frame_token`` and
        the dynamic fields identify the exact stage realization.
        """
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be an ADMGridGeometry.")
        if not isinstance(units, RelativisticUnitContract):
            raise TypeError("units must be a RelativisticUnitContract.")
        observer = _identifier(observer_id, "observer_id")
        orientation = _identifier(orientation_id, "orientation_id")
        if geometry.scale_id != units.scale.scale_id:
            raise ValueError("ADM geometry and relativistic units use different scales.")
        if geometry.convention_id != units.convention.convention_id:
            raise ValueError(
                "ADM geometry and relativistic units use different conventions."
            )
        tetrad = _eulerian_adm_tetrad(
            geometry,
            units,
            observer_id=observer,
            orientation_id=orientation,
            coordinate_names=coordinate_names,
            tolerance=tolerance,
        )
        return cls(
            geometry,
            tetrad,
            units,
            observer_coordinates,
            time,
            scale_factor,
            observer_id=observer,
            orientation_id=orientation,
        )

    @property
    def frame_token(self) -> Array:
        """Return the authoritative dynamic ADM stage token."""
        return self.geometry.snapshot_token

    def realization_id(self, /) -> str:
        """Content-address this exact materialized frame outside traced execution."""
        return canonical_fingerprint(
            {
                "kind": "local-relativistic-frame-realization",
                "frame": self.frame_id,
                "geometry_snapshot": array_tree_fingerprint(
                    (
                        self.geometry.alpha,
                        self.geometry.beta_contravariant,
                        self.geometry.spatial_metric,
                        self.geometry.inverse_spatial_metric,
                        self.geometry.sqrt_det_spatial_metric,
                        self.geometry.extrinsic_curvature,
                        self.geometry.active,
                        self.geometry.valid,
                        self.geometry.snapshot_token,
                    )
                ),
                "tetrad_snapshot": array_tree_fingerprint(
                    (
                        self.tetrad.vectors,
                        self.tetrad.dual_covectors,
                        self.tetrad.qualified,
                    )
                ),
                "observer_coordinates": array_tree_fingerprint(self.observer_coordinates),
                "time": array_tree_fingerprint(self.time),
                "scale_factor": array_tree_fingerprint(self.scale_factor),
            }
        )

    def coordinate_to_local(self, four_vector: ArrayLike, /) -> Array:
        """Project a coordinate-basis contravariant four-vector to this frame."""
        value = tetrad_project_vector(self.tetrad, four_vector)
        return eqx.error_if(
            value,
            jnp.any(self.geometry.active & ~self.admissible),
            "Coordinate-to-local mapping requires an admissible frame.",
        )

    def local_to_coordinate(self, four_vector: ArrayLike, /) -> Array:
        """Reconstruct a coordinate-basis contravariant four-vector."""
        value = tetrad_reconstruct_vector(self.tetrad, four_vector)
        return eqx.error_if(
            value,
            jnp.any(self.geometry.active & ~self.admissible),
            "Local-to-coordinate mapping requires an admissible frame.",
        )


__all__ = [
    "ColorNormalization",
    "IdenticalParticleNormalization",
    "LocalRelativisticFramePlan",
    "PhaseSpaceNormalization",
    "PolarizationNormalization",
    "RelativisticUnitContract",
    "SMatrixNormalization",
    "SpinNormalization",
]
