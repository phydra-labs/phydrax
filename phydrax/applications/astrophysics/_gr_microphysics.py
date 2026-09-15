#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import (
    derived_unit,
    HERTZ,
    KELVIN,
    KILOGRAM,
    METER,
    RADIAN,
    SECOND,
    TESLA,
    UnitDefinition,
)


_SI_RELATIVITY_SCALE = RelativityScaleContract.si()
_ELECTRON_CHARGE_C = 1.602176634e-19
_ELECTRON_MASS_KG = 9.1093837139e-31
_SPEED_OF_LIGHT_M_S = float(_SI_RELATIVITY_SCALE.speed_of_light)
_BOLTZMANN_CONSTANT_J_K = float(_SI_RELATIVITY_SCALE.boltzmann_constant)
_PLANCK_CONSTANT_J_S = 2.0 * np.pi * float(_SI_RELATIVITY_SCALE.reduced_planck_constant)
_VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
_THETA_PER_KELVIN = _BOLTZMANN_CONSTANT_J_K / (_ELECTRON_MASS_KG * _SPEED_OF_LIGHT_M_S**2)
_CYCLOTRON_HZ_PER_TESLA = _ELECTRON_CHARGE_C / (2.0 * np.pi * _ELECTRON_MASS_KG)
_FARADAY_ROTATION_FACTOR = _ELECTRON_CHARGE_C**3 / (
    8.0 * np.pi**2 * _VACUUM_PERMITTIVITY_F_M * _ELECTRON_MASS_KG**2 * _SPEED_OF_LIGHT_M_S
)
_FARADAY_CONVERSION_FACTOR = _ELECTRON_CHARGE_C**4 / (
    16.0
    * np.pi**3
    * _VACUUM_PERMITTIVITY_F_M
    * _ELECTRON_MASS_KG**3
    * _SPEED_OF_LIGHT_M_S
)
_LOG_K2_LOG10_EDGES = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
_LOG_K2_CHEBYSHEV_COEFFICIENTS = (
    (
        12.20606555855464,
        -2.302595605733206,
        -5.042258578732733e-06,
        -1.7528326725779586e-06,
        -4.742870492885915e-07,
        -1.0469095570748418e-07,
        -1.94774020006306e-08,
        -3.1277869844586712e-09,
        -4.413604438442472e-10,
        -5.547927841488118e-11,
        -6.277578979354226e-12,
        -6.439950525523913e-13,
        -6.078615582230918e-14,
    ),
    (
        7.600197070900404,
        -2.30363059656856,
        -0.0005003392668143887,
        -0.00017320265173912915,
        -4.651812383904501e-05,
        -1.0135420264400576e-05,
        -1.8433842373908956e-06,
        -2.8450611869395863e-07,
        -3.742557547487809e-08,
        -4.141422301010647e-09,
        -3.66091050992432e-10,
        -2.116158295794969e-11,
        3.119404677449119e-13,
    ),
    (
        2.93407688098511,
        -2.392008371130662,
        -0.04049598989829015,
        -0.012719127712850649,
        -0.002905922723163059,
        -0.0004826425533424287,
        -5.385728770547267e-05,
        -2.6262083114149196e-06,
        2.8314556077221673e-07,
        5.0933730071039586e-08,
        -3.7194155767020174e-09,
        -2.0826227158132267e-09,
        -2.15703863493321e-10,
    ),
    (
        -4.034152289817416,
        -5.386680896778239,
        -1.0646291390199631,
        -0.2261840161835721,
        -0.0316773052885899,
        -0.0033074847344752435,
        -0.0003359797862340499,
        -3.246669470308714e-05,
        -1.4826357925031728e-06,
        -3.138408034468569e-08,
        -2.9386108600194292e-08,
        -9.949284224740821e-10,
        6.99257640578423e-10,
    ),
    (
        -44.425383933374235,
        -43.43370059167387,
        -11.665155468413031,
        -2.1863627293349395,
        -0.30865966670090283,
        -0.035229844204541255,
        -0.003350305818289094,
        -0.00027359344457418577,
        -1.968227711047159e-05,
        -1.2442101401184275e-06,
        -7.19901957710932e-08,
        -3.7941203426179766e-09,
        -1.670376305590995e-10,
    ),
    (
        -432.6673190678204,
        -428.4000366675323,
        -116.85043753882353,
        -21.8292621894409,
        -3.0907151265110646,
        -0.351986821795249,
        -0.033506021568085353,
        -0.0027393242693591626,
        -0.00019620556206031447,
        -1.2503715199533323e-05,
        -7.176215416898661e-07,
        -3.745837150295703e-08,
        -1.7933863590233237e-09,
    ),
)
_LOG_K2_FLOAT64_UNIFORM_ERROR = 1.0e-10
_LOG_K2_FLOAT32_UNIFORM_ERROR = 2.0e-4
_MNY96_MINIMUM_TEMPERATURE_K = 3.2e10
_MNY96_MAXIMUM_SHAPE_RELATIVE_ERROR = 0.027


class ThermalSynchrotronUnitContract(StrictModule, NonTrainableState):
    """Exact SI units consumed and produced by thermal synchrotron evaluation."""

    scale: RelativityScaleContract
    number_density_unit: UnitDefinition = eqx.field(static=True)
    temperature_unit: UnitDefinition = eqx.field(static=True)
    magnetic_field_unit: UnitDefinition = eqx.field(static=True)
    frequency_unit: UnitDefinition = eqx.field(static=True)
    emissivity_unit: UnitDefinition = eqx.field(static=True)
    absorption_unit: UnitDefinition = eqx.field(static=True)
    specific_intensity_unit: UnitDefinition = eqx.field(static=True)
    units_id: str = eqx.field(static=True)

    def __init__(self, scale: RelativityScaleContract, /):
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        if scale.scale_id != _SI_RELATIVITY_SCALE.scale_id:
            raise ValueError(
                "Thermal synchrotron SI units require the SI relativity scale."
            )
        number_density = derived_unit("m^-3", ((METER, -3),))
        # j_nu: W m^-3 Hz^-1 sr^-1 = kg m^-1 s^-2 rad^-2.
        emissivity = derived_unit(
            "W/(m^3*Hz*sr)",
            ((KILOGRAM, 1), (METER, -1), (SECOND, -2), (RADIAN, -2)),
        )
        inverse_length = derived_unit("m^-1", ((METER, -1),))
        # I_nu: W m^-2 Hz^-1 sr^-1 = kg s^-1 rad^-2.
        specific_intensity = derived_unit(
            "W/(m^2*Hz*sr)",
            ((KILOGRAM, 1), (SECOND, -1), (RADIAN, -2)),
        )
        self.scale = scale
        self.number_density_unit = number_density
        self.temperature_unit = KELVIN
        self.magnetic_field_unit = TESLA
        self.frequency_unit = HERTZ
        self.emissivity_unit = emissivity
        self.absorption_unit = inverse_length
        self.specific_intensity_unit = specific_intensity
        self.units_id = canonical_fingerprint(
            {
                "kind": "thermal-synchrotron-si-units",
                "scale": scale.scale_id,
                "number_density": number_density.unit_id,
                "temperature": KELVIN.unit_id,
                "magnetic_field": TESLA.unit_id,
                "frequency": HERTZ.unit_id,
                "emissivity": emissivity.unit_id,
                "absorption": inverse_length.unit_id,
                "specific_intensity": specific_intensity.unit_id,
            }
        )


class ThermalSynchrotronReferenceEvidence(StrictModule, NonTrainableState):
    """Published support plus independently validated numerical-function evidence."""

    authors: tuple[str, ...] = eqx.field(static=True)
    title: str = eqx.field(static=True)
    year: int = eqx.field(static=True)
    doi: str = eqx.field(static=True)
    arxiv_id: str = eqx.field(static=True)
    equation: str = eqx.field(static=True)
    minimum_temperature_k: float = eqx.field(static=True)
    maximum_shape_relative_error: float = eqx.field(static=True)
    maximum_error_normalized_frequency: float = eqx.field(static=True)
    k2_log_argument_bounds: tuple[float, float] = eqx.field(static=True)
    k2_float64_uniform_log_error: float = eqx.field(static=True)
    k2_float32_uniform_log_error: float = eqx.field(static=True)
    polarization_status: str = eqx.field(static=True)
    faraday_status: str = eqx.field(static=True)
    source_ledger: tuple[tuple[str, str], ...] = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(self):
        authors = ("Rohan Mahadevan", "Ramesh Narayan", "Insu Yi")
        title = (
            "Harmony in Electrons: Cyclotron and Synchrotron Emission by "
            "Thermal Electrons in a Magnetic Field"
        )
        self.authors = authors
        self.title = title
        self.year = 1996
        self.doi = "10.1086/177422"
        self.arxiv_id = "astro-ph/9601073"
        self.equation = "31"
        self.minimum_temperature_k = _MNY96_MINIMUM_TEMPERATURE_K
        self.maximum_shape_relative_error = _MNY96_MAXIMUM_SHAPE_RELATIVE_ERROR
        self.maximum_error_normalized_frequency = 160.0
        self.k2_log_argument_bounds = (1.0e-3, 1.0e3)
        self.k2_float64_uniform_log_error = _LOG_K2_FLOAT64_UNIFORM_ERROR
        self.k2_float32_uniform_log_error = _LOG_K2_FLOAT32_UNIFORM_ERROR
        self.polarization_status = "unqualified-independent-approximation"
        self.faraday_status = "unqualified-independent-approximation"
        self.source_ledger = (
            (
                "thermal-shape",
                "Mahadevan-Narayan-Yi-1996 DOI:10.1086/177422 Eq.31; "
                "T>=3.2e10 K; maximum relative shape error 0.027",
            ),
            (
                "bessel-k2",
                "independent piecewise Chebyshev log(K2) validation against "
                "scaled float64 reference evaluation on z in [1e-3,1e3]",
            ),
            ("polarization", self.polarization_status),
            ("faraday", self.faraday_status),
        )
        self.reference_id = canonical_fingerprint(
            {
                "kind": "thermal-synchrotron-reference-evidence",
                "authors": authors,
                "title": title,
                "year": 1996,
                "doi": self.doi,
                "arxiv_id": self.arxiv_id,
                "equation": self.equation,
                "minimum_temperature_k": self.minimum_temperature_k,
                "maximum_shape_relative_error": self.maximum_shape_relative_error,
                "maximum_error_normalized_frequency": (
                    self.maximum_error_normalized_frequency
                ),
                "k2_method": "piecewise-degree-12-chebyshev-log-k2",
                "k2_log_argument_bounds": self.k2_log_argument_bounds,
                "k2_float64_uniform_log_error": (self.k2_float64_uniform_log_error),
                "k2_float32_uniform_log_error": (self.k2_float32_uniform_log_error),
                "k2_reference": "scaled-modified-bessel-k2-float64",
                "k2_validation_nodes": 600006,
                "polarization_status": self.polarization_status,
                "faraday_status": self.faraday_status,
                "source_ledger": self.source_ledger,
            }
        )


class ThermalSynchrotronDomain(StrictModule, NonTrainableState):
    """Published ultra-relativistic fit support intersected with numeric support."""

    minimum_temperature_k: float = eqx.field(static=True)
    minimum_theta_e: float = eqx.field(static=True)
    maximum_theta_e: float = eqx.field(static=True)
    minimum_normalized_frequency: float = eqx.field(static=True)
    maximum_normalized_frequency: float = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_temperature_k: float = _MNY96_MINIMUM_TEMPERATURE_K,
        minimum_theta_e: float = 1.0e-3,
        maximum_theta_e: float = 1.0e3,
        minimum_normalized_frequency: float = 1.0e-6,
        maximum_normalized_frequency: float = 1.0e6,
    ):
        values = tuple(
            float(value)
            for value in (
                minimum_temperature_k,
                minimum_theta_e,
                maximum_theta_e,
                minimum_normalized_frequency,
                maximum_normalized_frequency,
            )
        )
        if (
            any(not np.isfinite(value) or value <= 0.0 for value in values)
            or values[2] <= values[1]
            or values[4] <= values[3]
            or values[0] < _MNY96_MINIMUM_TEMPERATURE_K
        ):
            raise ValueError(
                "Thermal synchrotron domain must remain inside published and "
                "numerically validated support."
            )
        (
            self.minimum_temperature_k,
            self.minimum_theta_e,
            self.maximum_theta_e,
            self.minimum_normalized_frequency,
            self.maximum_normalized_frequency,
        ) = values
        self.domain_id = canonical_fingerprint(
            {"kind": "thermal-synchrotron-domain", "bounds": values}
        )


class ThermalSynchrotronEvidence(StrictModule):
    finite: Array
    physically_valid: Array
    in_domain: Array
    k2_approximation_valid: Array
    emission_reference_valid: Array
    polarization_reference_valid: Array
    faraday_reference_valid: Array
    qualified: Array
    emission_derivative_valid: Array
    derivative_valid: Array


class ThermalSynchrotronCoefficients(StrictModule):
    """Polarized coefficients in the projected-field screen basis.

    The Stokes convention is ``(I, Q, U, V)`` with the first local screen axis
    parallel to the projected magnetic field.  Consequently the fitted thermal
    linear emission and dichroism are negative Q, while U and V emission vanish.
    Faraday rotation is signed by the line-of-sight field and conversion is
    positive under this basis convention.
    """

    emission: Array
    absorption_i: Array
    absorption_q: Array
    faraday_rotation: Array
    faraday_conversion: Array
    propagation_matrix: Array
    theta_e: Array
    characteristic_frequency_hz: Array
    normalized_frequency: Array
    evidence: ThermalSynchrotronEvidence
    model_id: str = eqx.field(static=True)


class InvariantSynchrotronCoefficients(StrictModule):
    emission: Array
    propagation_matrix: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    model_id: str = eqx.field(static=True)


def _validated_log_bessel_k2(z: Array, /) -> Array:
    """Evaluate log(K2(z)) by a validated piecewise Chebyshev approximation.

    Six degree-12 panels in log10(z) cover 1e-3 through 1e3.  Against scaled
    modified-Bessel float64 reference evaluation at 600006 uniformly spaced
    panel validation nodes, the maximum absolute log error is below 1e-10 in
    float64 and 2e-4 after float32 coefficient/evaluation rounding.  Callers
    gate qualification on this support.
    """

    log_argument = jnp.log10(z)
    bounded_log_argument = jnp.clip(log_argument, -3.0, 3.0)
    panel = jnp.clip(jnp.floor(bounded_log_argument + 3.0).astype(jnp.int32), 0, 5)
    center = -2.5 + panel.astype(z.dtype)
    coordinate = 2.0 * (bounded_log_argument - center)
    coefficients = jnp.asarray(_LOG_K2_CHEBYSHEV_COEFFICIENTS, dtype=z.dtype)[panel]
    next_value = jnp.zeros_like(coordinate)
    next_next_value = jnp.zeros_like(coordinate)
    for index in range(12, 0, -1):
        value = 2.0 * coordinate * next_value - next_next_value + coefficients[..., index]
        next_next_value = next_value
        next_value = value
    return coordinate * next_value - next_next_value + coefficients[..., 0]


def _thermal_shape_log(normalized_frequency: Array, /) -> Array:
    """Mahadevan, Narayan & Yi (1996), equation 31."""

    x_one_sixth = normalized_frequency ** (1.0 / 6.0)
    x_one_fourth = normalized_frequency**0.25
    x_one_half = jnp.sqrt(normalized_frequency)
    correction = 1.0 + 0.40 / x_one_fourth + 0.5316 / x_one_half
    return (
        jnp.log(4.0505)
        - jnp.log(x_one_sixth)
        + jnp.log(correction)
        - 1.8899 * normalized_frequency ** (1.0 / 3.0)
    )


def _log_planck_frequency(frequency_hz: Array, temperature_k: Array, /) -> Array:
    exponent = (
        _PLANCK_CONSTANT_J_S * frequency_hz / (_BOLTZMANN_CONSTANT_J_K * temperature_k)
    )
    log_denominator = jnp.where(
        exponent > 50.0,
        exponent,
        jnp.log(jnp.expm1(exponent)),
    )
    return (
        jnp.log(2.0)
        + jnp.log(_PLANCK_CONSTANT_J_S)
        - 2.0 * jnp.log(_SPEED_OF_LIGHT_M_S)
        + 3.0 * jnp.log(frequency_hz)
        - log_denominator
    )


def _propagation_matrix(
    absorption_i: Array,
    absorption_q: Array,
    faraday_rotation: Array,
    faraday_conversion: Array,
    /,
) -> Array:
    zero = jnp.zeros_like(absorption_i)
    return jnp.stack(
        (
            jnp.stack((absorption_i, absorption_q, zero, zero), axis=-1),
            jnp.stack((absorption_q, absorption_i, faraday_rotation, zero), axis=-1),
            jnp.stack(
                (zero, -faraday_rotation, absorption_i, faraday_conversion),
                axis=-1,
            ),
            jnp.stack((zero, zero, -faraday_conversion, absorption_i), axis=-1),
        ),
        axis=-2,
    )


class ThermalSynchrotronModel(StrictModule, NonTrainableState):
    """SI thermal synchrotron model with capability-scoped evidence.

    Stokes-I emission uses Mahadevan, Narayan & Yi (1996), equation 31, only
    within its published ultra-relativistic temperature support; its reported
    maximum spectral-shape fit error is 2.7 percent.  Kirchhoff absorption uses
    that emission and the Planck function.  The linear-polarization and Faraday
    forms are independent approximations and are deliberately never reference-
    qualified when active.
    """

    domain: ThermalSynchrotronDomain
    reference: ThermalSynchrotronReferenceEvidence
    units: ThermalSynchrotronUnitContract
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: ThermalSynchrotronDomain | None = None,
        /,
        *,
        scale: RelativityScaleContract | None = None,
    ):
        domain_ = ThermalSynchrotronDomain() if domain is None else domain
        if not isinstance(domain_, ThermalSynchrotronDomain):
            raise TypeError("domain must be a ThermalSynchrotronDomain.")
        scale_ = _SI_RELATIVITY_SCALE if scale is None else scale
        if not isinstance(scale_, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        units = ThermalSynchrotronUnitContract(scale_)
        reference = ThermalSynchrotronReferenceEvidence()
        self.domain = domain_
        self.reference = reference
        self.units = units
        self.model_id = canonical_fingerprint(
            {
                "kind": "thermal-synchrotron-si-model",
                "domain": domain_.domain_id,
                "units": units.units_id,
                "scale": scale_.scale_id,
                "emission_fit": "mahadevan-narayan-yi-1996-equation-31",
                "reference_evidence": reference.reference_id,
                "polarization": reference.polarization_status,
                "faraday": reference.faraday_status,
            }
        )

    def evaluate(
        self,
        electron_number_density_m3: ArrayLike,
        electron_temperature_k: ArrayLike,
        magnetic_field_t: ArrayLike,
        frequency_hz: ArrayLike,
        pitch_cosine: ArrayLike,
        /,
    ) -> ThermalSynchrotronCoefficients:
        density, temperature, magnetic_field, frequency, cosine = jnp.broadcast_arrays(
            jnp.asarray(electron_number_density_m3),
            jnp.asarray(electron_temperature_k),
            jnp.asarray(magnetic_field_t),
            jnp.asarray(frequency_hz),
            jnp.asarray(pitch_cosine),
        )
        dtype = jnp.result_type(density, temperature, magnetic_field, frequency, float)
        density = density.astype(dtype)
        temperature = temperature.astype(dtype)
        magnetic_field = magnetic_field.astype(dtype)
        frequency = frequency.astype(dtype)
        cosine = cosine.astype(dtype)

        input_finite = (
            jnp.isfinite(density)
            & jnp.isfinite(temperature)
            & jnp.isfinite(magnetic_field)
            & jnp.isfinite(frequency)
            & jnp.isfinite(cosine)
        )
        physically_valid = (
            input_finite
            & (density >= 0.0)
            & (temperature > 0.0)
            & (magnetic_field >= 0.0)
            & (frequency > 0.0)
            & (jnp.abs(cosine) <= 1.0)
        )
        safe_density = jnp.where(physically_valid, density, 1.0)
        safe_temperature = jnp.where(physically_valid, temperature, 1.0)
        safe_magnetic_field = jnp.where(physically_valid, magnetic_field, 0.0)
        safe_frequency = jnp.where(physically_valid, frequency, 1.0)
        safe_cosine = jnp.where(physically_valid, cosine, 0.0)

        theta_e = _THETA_PER_KELVIN * safe_temperature
        sine = jnp.sqrt(1.0 - safe_cosine**2)
        perpendicular_field = safe_magnetic_field * sine
        parallel_field = safe_magnetic_field * safe_cosine
        cyclotron_frequency = _CYCLOTRON_HZ_PER_TESLA * safe_magnetic_field
        characteristic_frequency = 1.5 * cyclotron_frequency * theta_e**2
        radiating = (safe_density > 0.0) & (safe_magnetic_field > 0.0)
        safe_characteristic = jnp.where(
            radiating, characteristic_frequency, safe_frequency
        )
        normalized_frequency = jnp.where(
            radiating, safe_frequency / safe_characteristic, 0.0
        )
        fit_x = jnp.where(radiating, normalized_frequency, 1.0)
        k2_argument = 1.0 / theta_e

        log_emissivity = (
            jnp.log(safe_density)
            + 2.0 * jnp.log(_ELECTRON_CHARGE_C)
            + jnp.log(safe_frequency)
            - jnp.log(4.0 * jnp.pi * _VACUUM_PERMITTIVITY_F_M)
            - jnp.log(_SPEED_OF_LIGHT_M_S)
            - 0.5 * jnp.log(3.0)
            + _thermal_shape_log(fit_x)
            - _validated_log_bessel_k2(k2_argument)
        )
        emissivity_i = jnp.where(radiating, jnp.exp(log_emissivity), 0.0)
        x_third = fit_x ** (1.0 / 3.0)
        linear_fraction = 0.5 + 0.25 * x_third / (1.0 + x_third)
        emissivity_q = jnp.where(radiating, -linear_fraction * emissivity_i, 0.0)
        zero = jnp.zeros_like(emissivity_i)
        emission = jnp.stack((emissivity_i, emissivity_q, zero, zero), axis=-1)

        log_planck = _log_planck_frequency(safe_frequency, safe_temperature)
        absorption_i = jnp.where(radiating, jnp.exp(log_emissivity - log_planck), 0.0)
        absorption_q = jnp.where(radiating, -linear_fraction * absorption_i, 0.0)

        rotation_suppression = (1.0 + jnp.log1p(theta_e)) / (1.0 + 2.0 * theta_e**2)
        faraday_rotation = (
            _FARADAY_ROTATION_FACTOR
            * safe_density
            * parallel_field
            / safe_frequency**2
            * rotation_suppression
        )
        conversion_suppression = theta_e / (1.0 + theta_e)
        faraday_conversion = (
            _FARADAY_CONVERSION_FACTOR
            * safe_density
            * perpendicular_field**2
            / safe_frequency**3
            * conversion_suppression
        )
        propagation = _propagation_matrix(
            absorption_i, absorption_q, faraday_rotation, faraday_conversion
        )

        emission_finite = jnp.isfinite(emissivity_i) & jnp.isfinite(absorption_i)
        precision_supported = dtype in (jnp.dtype("float32"), jnp.dtype("float64"))
        k2_approximation_valid = physically_valid & (
            ~radiating
            | (
                precision_supported
                & (k2_argument >= self.reference.k2_log_argument_bounds[0])
                & (k2_argument <= self.reference.k2_log_argument_bounds[1])
            )
        )
        active_in_domain = (
            (temperature >= self.domain.minimum_temperature_k)
            & (theta_e >= self.domain.minimum_theta_e)
            & (theta_e <= self.domain.maximum_theta_e)
            & (normalized_frequency >= self.domain.minimum_normalized_frequency)
            & (normalized_frequency <= self.domain.maximum_normalized_frequency)
        )
        in_domain = physically_valid & (~radiating | active_in_domain)
        emission_reference_valid = (
            physically_valid
            & emission_finite
            & k2_approximation_valid
            & (~radiating | in_domain)
        )
        polarization_active = radiating
        faraday_active = (safe_density > 0.0) & (safe_magnetic_field > 0.0)
        polarization_reference_valid = physically_valid & ~polarization_active
        faraday_reference_valid = physically_valid & ~faraday_active
        qualified = (
            emission_reference_valid
            & polarization_reference_valid
            & faraday_reference_valid
        )
        k2_panel_boundary = jnp.any(
            k2_argument[..., None]
            == jnp.asarray((1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2), dtype=dtype),
            axis=-1,
        )
        emission_derivative_valid = (
            emission_reference_valid
            & (density > 0.0)
            & (magnetic_field > 0.0)
            & (theta_e > self.domain.minimum_theta_e)
            & (theta_e < self.domain.maximum_theta_e)
            & (temperature > self.domain.minimum_temperature_k)
            & (normalized_frequency > self.domain.minimum_normalized_frequency)
            & (normalized_frequency < self.domain.maximum_normalized_frequency)
            & ~k2_panel_boundary
        )
        derivative_valid = qualified & emission_derivative_valid
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        emission_supported = physically_valid & k2_approximation_valid
        emission = jnp.where(emission_supported[..., None], emission, nan)
        absorption_i = jnp.where(emission_supported, absorption_i, nan)
        absorption_q = jnp.where(emission_supported, absorption_q, nan)
        faraday_rotation = jnp.where(physically_valid, faraday_rotation, nan)
        faraday_conversion = jnp.where(physically_valid, faraday_conversion, nan)
        propagation = _propagation_matrix(
            absorption_i, absorption_q, faraday_rotation, faraday_conversion
        )
        theta_output = jnp.where(physically_valid, theta_e, nan)
        characteristic_output = jnp.where(physically_valid, characteristic_frequency, nan)
        normalized_output = jnp.where(physically_valid, normalized_frequency, nan)
        result_finite = (
            jnp.all(jnp.isfinite(emission), axis=-1)
            & jnp.isfinite(absorption_i)
            & jnp.isfinite(absorption_q)
            & jnp.isfinite(faraday_rotation)
            & jnp.isfinite(faraday_conversion)
            & jnp.all(jnp.isfinite(propagation), axis=(-2, -1))
        )
        evidence = ThermalSynchrotronEvidence(
            result_finite,
            physically_valid,
            in_domain,
            k2_approximation_valid,
            emission_reference_valid,
            polarization_reference_valid,
            faraday_reference_valid,
            qualified,
            emission_derivative_valid,
            derivative_valid,
        )
        return ThermalSynchrotronCoefficients(
            emission,
            absorption_i,
            absorption_q,
            faraday_rotation,
            faraday_conversion,
            propagation,
            theta_output,
            characteristic_output,
            normalized_output,
            evidence,
            self.model_id,
        )


def invariant_synchrotron_coefficients(
    coefficients: ThermalSynchrotronCoefficients,
    comoving_frequency_hz: ArrayLike,
    /,
) -> InvariantSynchrotronCoefficients:
    """Convert local physical coefficients to the affine invariant convention.

    This uses ``J = j_nu / nu^2`` and ``K = nu K_nu`` for an affine
    normalization satisfying ``d ell / d lambda = nu``.  A differently normalized
    affine tangent must rescale the returned pair before transfer.
    """

    if not isinstance(coefficients, ThermalSynchrotronCoefficients):
        raise TypeError("coefficients must be ThermalSynchrotronCoefficients.")
    frequency = jnp.asarray(comoving_frequency_hz)
    expected = coefficients.emission.shape[:-1]
    if frequency.shape != expected:
        raise ValueError("Comoving frequency must match coefficient sample shape.")
    frequency_finite = jnp.isfinite(frequency)
    physically_valid = coefficients.evidence.physically_valid & (frequency > 0.0)
    safe_frequency = jnp.where(physically_valid, frequency, 1.0)
    emission = coefficients.emission / safe_frequency[..., None] ** 2
    propagation = coefficients.propagation_matrix * safe_frequency[..., None, None]
    nan = jnp.asarray(jnp.nan, dtype=emission.dtype)
    emission = jnp.where(physically_valid[..., None], emission, nan)
    propagation = jnp.where(physically_valid[..., None, None], propagation, nan)
    finite = (
        coefficients.evidence.finite
        & frequency_finite
        & jnp.all(jnp.isfinite(emission), axis=-1)
        & jnp.all(jnp.isfinite(propagation), axis=(-2, -1))
    )
    qualified = coefficients.evidence.qualified & finite & physically_valid
    derivative_valid = coefficients.evidence.derivative_valid & qualified
    return InvariantSynchrotronCoefficients(
        emission,
        propagation,
        finite,
        physically_valid,
        qualified,
        derivative_valid,
        coefficients.model_id,
    )
