#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import DifferentialProblem
from ._experiment import BatteryRuntimeInputs
from ._properties import (
    ConcentrationTemperaturePropertyLaw,
    ConstantPropertyLaw,
    TabulatedPropertyLaw,
)
from ._results import BatteryModelOutput
from ._spm import _stable_asinh_ratio, _transport, BatteryPropertyLaw, SpmState
from ._spme_marquis2019 import (
    _finite_large_ratio,
    _region_mean,
    Marquis2019SpmeAdapter,
    Marquis2019SpmeInitialCondition,
    Marquis2019SpmeLedger,
    Marquis2019SpmeParameters,
    Marquis2019SpmePlan,
    Marquis2019SpmeState,
    PreparedMarquis2019Spme,
)


_FARADAY_C_MOL = 96485.33212
_GAS_CONSTANT_J_MOL_K = 8.31446261815324
_SOURCE_FORMULATION_ID = "arxiv:2011.01611v3:section-3:eqs-1-10"
_PROPERTY_TYPES = (ConstantPropertyLaw, TabulatedPropertyLaw)
_PropertyLaw: TypeAlias = ConstantPropertyLaw | TabulatedPropertyLaw


def _scalar(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape != () or jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must be one real scalar.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return array


def _property_law(
    value: _PropertyLaw,
    name: str,
    /,
    *,
    coordinate: str,
    coordinate_unit: str,
    value_unit: str,
) -> _PropertyLaw:
    if not isinstance(value, _PROPERTY_TYPES):
        raise TypeError(f"{name} must be ConstantPropertyLaw or TabulatedPropertyLaw.")
    if value.coordinate != coordinate or value.coordinate_unit != coordinate_unit:
        raise ValueError(
            f"{name} must use coordinate {coordinate!r} in {coordinate_unit!r}."
        )
    if value.value_unit != value_unit:
        raise ValueError(f"{name} must use value unit {value_unit!r}.")
    return value


def _bivariate_property_law(
    value: ConcentrationTemperaturePropertyLaw,
    name: str,
    /,
    *,
    value_unit: str,
) -> ConcentrationTemperaturePropertyLaw:
    if not isinstance(value, ConcentrationTemperaturePropertyLaw):
        raise TypeError(f"{name} must be ConcentrationTemperaturePropertyLaw.")
    if value.value_unit != value_unit:
        raise ValueError(f"{name} must use value unit {value_unit!r}.")
    return value


def _electrolyte_thermodynamic_primitive(
    transference_law: ConcentrationTemperaturePropertyLaw,
    thermodynamic_factor_law: _PropertyLaw,
    concentration_mol_m3: Array,
    temperature_k: Array,
    /,
) -> Array:
    """Exact primitive of (1-t_plus(c,T)) chi(c) d(log(c))."""
    concentration = jnp.asarray(concentration_mol_m3)
    nodes = transference_law.concentration_nodes_mol_m3
    transference_at_nodes = transference_law.evaluate(
        nodes,
        temperature_k,
    ).values
    if isinstance(thermodynamic_factor_law, ConstantPropertyLaw):
        thermodynamic_at_nodes = jnp.zeros_like(nodes) + thermodynamic_factor_law.value
    else:
        thermodynamic_at_nodes = thermodynamic_factor_law.evaluate(nodes).values
    left = nodes[:-1]
    right = nodes[1:]
    inverse_width = 1.0 / (right - left)
    transference_slope = (
        transference_at_nodes[1:] - transference_at_nodes[:-1]
    ) * inverse_width
    transference_intercept = transference_at_nodes[:-1] - transference_slope * left
    thermodynamic_slope = (
        thermodynamic_at_nodes[1:] - thermodynamic_at_nodes[:-1]
    ) * inverse_width
    thermodynamic_intercept = thermodynamic_at_nodes[:-1] - thermodynamic_slope * left
    quadratic = -transference_slope * thermodynamic_slope
    linear = (
        1.0 - transference_intercept
    ) * thermodynamic_slope - transference_slope * thermodynamic_intercept
    constant = (1.0 - transference_intercept) * thermodynamic_intercept
    upper = jnp.clip(concentration[..., None], left, right)
    segment_integral = (
        0.5 * quadratic * (upper**2 - left**2)
        + linear * (upper - left)
        + constant * jnp.log(upper / left)
    )
    return jnp.sum(segment_integral, axis=-1)


class BrosaPlanellaTspmeParameters(StrictModule):
    """Dynamic SI properties for Brosa Planella et al. Section 3's base TSPMe.

    Electrolyte diffusivity, conductivity, and transference are bounded
    concentration-temperature surfaces. The thermodynamic-factor law is the full
    ``1 + d f_pm / d c_e`` multiplier and is integrated exactly with transference
    against ``d(log(c_e))``. The reference OCP and its entropic coefficient give
    ``U(theta, T) = U_ref(theta) + (T - T_ref) dU/dT(theta)``.
    """

    spme_parameters: Marquis2019SpmeParameters
    ambient_temperature_k: Array
    volumetric_heat_capacity_j_m3_k: Array
    heat_transfer_coefficient_w_m2_k: Array
    cooling_surface_area_per_volume_m_inv: Array
    battery_length_scale_m: Array
    battery_thermal_conductivity_w_m_k: Array
    typical_discharge_time_s: Array
    typical_electrode_potential_v: Array
    negative_solid_conductivity: BatteryPropertyLaw
    positive_solid_conductivity: BatteryPropertyLaw
    electrolyte_diffusivity: ConcentrationTemperaturePropertyLaw
    electrolyte_conductivity: ConcentrationTemperaturePropertyLaw
    electrolyte_transference_number: ConcentrationTemperaturePropertyLaw
    electrolyte_thermodynamic_factor: _PropertyLaw
    negative_entropic_coefficient: BatteryPropertyLaw
    positive_entropic_coefficient: BatteryPropertyLaw
    electrochemical_property_support_id: str = eqx.field(static=True)
    thermal_property_support_id: str = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        spme_parameters: Marquis2019SpmeParameters,
        /,
        *,
        ambient_temperature_k: ArrayLike,
        volumetric_heat_capacity_j_m3_k: ArrayLike,
        heat_transfer_coefficient_w_m2_k: ArrayLike,
        cooling_surface_area_per_volume_m_inv: ArrayLike,
        battery_length_scale_m: ArrayLike,
        battery_thermal_conductivity_w_m_k: ArrayLike,
        typical_discharge_time_s: ArrayLike,
        typical_electrode_potential_v: ArrayLike,
        negative_solid_conductivity: BatteryPropertyLaw,
        positive_solid_conductivity: BatteryPropertyLaw,
        electrolyte_diffusivity: ConcentrationTemperaturePropertyLaw,
        electrolyte_conductivity: ConcentrationTemperaturePropertyLaw,
        electrolyte_transference_number: ConcentrationTemperaturePropertyLaw,
        electrolyte_thermodynamic_factor: _PropertyLaw,
        negative_entropic_coefficient: BatteryPropertyLaw,
        positive_entropic_coefficient: BatteryPropertyLaw,
    ):
        if not isinstance(spme_parameters, Marquis2019SpmeParameters):
            raise TypeError("spme_parameters must be Marquis2019SpmeParameters.")
        ambient = _scalar(ambient_temperature_k, "ambient_temperature_k")
        heat_capacity = _scalar(
            volumetric_heat_capacity_j_m3_k, "volumetric_heat_capacity_j_m3_k"
        )
        heat_transfer = _scalar(
            heat_transfer_coefficient_w_m2_k,
            "heat_transfer_coefficient_w_m2_k",
        )
        cooling_area = _scalar(
            cooling_surface_area_per_volume_m_inv,
            "cooling_surface_area_per_volume_m_inv",
        )
        battery_length = _scalar(battery_length_scale_m, "battery_length_scale_m")
        thermal_conductivity = _scalar(
            battery_thermal_conductivity_w_m_k,
            "battery_thermal_conductivity_w_m_k",
        )
        discharge_time = _scalar(typical_discharge_time_s, "typical_discharge_time_s")
        electrode_potential = _scalar(
            typical_electrode_potential_v, "typical_electrode_potential_v"
        )
        negative_conductivity = _property_law(
            negative_solid_conductivity,
            "negative_solid_conductivity",
            coordinate="temperature",
            coordinate_unit="K",
            value_unit="S/m",
        )
        positive_conductivity = _property_law(
            positive_solid_conductivity,
            "positive_solid_conductivity",
            coordinate="temperature",
            coordinate_unit="K",
            value_unit="S/m",
        )
        electrolyte_diffusivity_ = _bivariate_property_law(
            electrolyte_diffusivity,
            "electrolyte_diffusivity",
            value_unit="m2/s",
        )
        electrolyte_conductivity_ = _bivariate_property_law(
            electrolyte_conductivity,
            "electrolyte_conductivity",
            value_unit="S/m",
        )
        electrolyte_transference = _bivariate_property_law(
            electrolyte_transference_number,
            "electrolyte_transference_number",
            value_unit="1",
        )
        thermodynamic_factor = _property_law(
            electrolyte_thermodynamic_factor,
            "electrolyte_thermodynamic_factor",
            coordinate="electrolyte_concentration",
            coordinate_unit="mol/m3",
            value_unit="1",
        )
        thermodynamic_support = np.asarray(
            thermodynamic_factor.support_bounds, dtype=float
        )
        if (
            np.any(~np.isfinite(thermodynamic_support))
            or thermodynamic_support[0] <= 0.0
            or (
                isinstance(thermodynamic_factor, TabulatedPropertyLaw)
                and (
                    not np.all(np.asarray(thermodynamic_factor.source_mask, dtype=bool))
                    or np.any(np.asarray(thermodynamic_factor.nodes, dtype=float) <= 0.0)
                )
            )
        ):
            raise ValueError(
                "electrolyte_thermodynamic_factor requires connected positive "
                "concentration support for its exact thermodynamic primitive."
            )
        transference_mask = np.asarray(electrolyte_transference.source_mask, dtype=bool)
        transference_concentration_nodes = np.asarray(
            electrolyte_transference.concentration_nodes_mol_m3, dtype=float
        )
        if not np.all(transference_mask):
            raise ValueError(
                "electrolyte_transference_number requires connected rectangular "
                "support for the exact electrolyte-potential primitive."
            )
        if isinstance(thermodynamic_factor, TabulatedPropertyLaw):
            thermodynamic_nodes = np.asarray(thermodynamic_factor.nodes, dtype=float)
            if (
                thermodynamic_nodes.shape != transference_concentration_nodes.shape
                or np.any(thermodynamic_nodes != transference_concentration_nodes)
            ):
                raise ValueError(
                    "Tabulated electrolyte thermodynamic-factor nodes must equal "
                    "the transference-law concentration nodes."
                )
        elif (
            thermodynamic_support[0] > transference_concentration_nodes[0]
            or thermodynamic_support[1] < transference_concentration_nodes[-1]
        ):
            raise ValueError(
                "Constant electrolyte thermodynamic-factor support must cover "
                "the transference-law concentration axis."
            )
        negative_entropy = _property_law(
            negative_entropic_coefficient,
            "negative_entropic_coefficient",
            coordinate="stoichiometry",
            coordinate_unit="1",
            value_unit="V/K",
        )
        positive_entropy = _property_law(
            positive_entropic_coefficient,
            "positive_entropic_coefficient",
            coordinate="stoichiometry",
            coordinate_unit="1",
            value_unit="V/K",
        )

        positive_scalars = jnp.stack(
            (
                ambient,
                heat_capacity,
                cooling_area,
                battery_length,
                thermal_conductivity,
                discharge_time,
                electrode_potential,
            )
        )
        scalar_valid = (
            jnp.all(jnp.isfinite(positive_scalars) & (positive_scalars > 0.0))
            & jnp.isfinite(heat_transfer)
            & (heat_transfer >= 0.0)
        )
        reference_temperature = spme_parameters.spm_parameters.temperature_k
        temperature_queries = jnp.stack((ambient, reference_temperature))
        temperature_results = (
            spme_parameters.spm_parameters.negative_solid_diffusivity.evaluate(
                temperature_queries
            ),
            spme_parameters.spm_parameters.positive_solid_diffusivity.evaluate(
                temperature_queries
            ),
            spme_parameters.spm_parameters.negative_exchange_current_density.evaluate(
                temperature_queries
            ),
            spme_parameters.spm_parameters.positive_exchange_current_density.evaluate(
                temperature_queries
            ),
            negative_conductivity.evaluate(temperature_queries),
            positive_conductivity.evaluate(temperature_queries),
        )
        temperature_property_valid = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(result.support)
                    & jnp.all(jnp.isfinite(result.values))
                    & jnp.all(result.values > 0.0)
                    for result in temperature_results
                )
            )
        )
        typical_concentration = spme_parameters.typical_electrolyte_concentration_mol_m3
        electrolyte_diffusivity_result = electrolyte_diffusivity_.evaluate(
            typical_concentration, temperature_queries
        )
        electrolyte_conductivity_result = electrolyte_conductivity_.evaluate(
            typical_concentration, temperature_queries
        )
        electrolyte_transference_result = electrolyte_transference.evaluate(
            typical_concentration, temperature_queries
        )
        thermodynamic_factor_result = thermodynamic_factor.evaluate(typical_concentration)
        spm = spme_parameters.spm_parameters
        negative_endpoint_entropy = negative_entropy.evaluate(
            jnp.stack(
                (
                    spm.negative_stoichiometry_at_empty,
                    spm.negative_stoichiometry_at_full,
                )
            )
        )
        positive_endpoint_entropy = positive_entropy.evaluate(
            jnp.stack(
                (
                    spm.positive_stoichiometry_at_full,
                    spm.positive_stoichiometry_at_empty,
                )
            )
        )
        remaining_property_valid = (
            jnp.all(electrolyte_diffusivity_result.support)
            & jnp.all(jnp.isfinite(electrolyte_diffusivity_result.values))
            & jnp.all(electrolyte_diffusivity_result.values > 0.0)
            & jnp.all(electrolyte_conductivity_result.support)
            & jnp.all(jnp.isfinite(electrolyte_conductivity_result.values))
            & jnp.all(electrolyte_conductivity_result.values > 0.0)
            & jnp.all(electrolyte_transference_result.support)
            & jnp.all(jnp.isfinite(electrolyte_transference_result.values))
            & jnp.all(electrolyte_transference_result.values >= 0.0)
            & jnp.all(electrolyte_transference_result.values <= 1.0)
            & thermodynamic_factor_result.support
            & jnp.isfinite(thermodynamic_factor_result.values)
            & (thermodynamic_factor_result.values > 0.0)
            & jnp.all(negative_endpoint_entropy.support)
            & jnp.all(positive_endpoint_entropy.support)
            & jnp.all(jnp.isfinite(negative_endpoint_entropy.values))
            & jnp.all(jnp.isfinite(positive_endpoint_entropy.values))
        )
        ambient = eqx.error_if(
            ambient,
            ~(scalar_valid & temperature_property_valid & remaining_property_valid),
            "TSPMe thermal data and every independently declared property support "
            "must be finite, physical, and cover the reference operating point.",
        )
        electrochemical_support_id = canonical_fingerprint(
            {
                "kind": "battery-tspme-brosa-planella-electrochemical-property-support",
                "spm_parameter_id": spme_parameters.spm_parameters.parameter_id,
                "negative_solid_conductivity_law_id": negative_conductivity.law_id,
                "positive_solid_conductivity_law_id": positive_conductivity.law_id,
                "electrolyte_diffusivity_law_id": electrolyte_diffusivity_.law_id,
                "electrolyte_conductivity_law_id": electrolyte_conductivity_.law_id,
                "electrolyte_thermodynamic_factor_law_id": (thermodynamic_factor.law_id),
                "electrolyte_transference_law_id": (electrolyte_transference.law_id),
                "negative_entropic_law_id": negative_entropy.law_id,
                "positive_entropic_law_id": positive_entropy.law_id,
            }
        )
        thermal_support_id = canonical_fingerprint(
            {
                "kind": "battery-tspme-brosa-planella-thermal-property-support",
                "temperature_law_ids": [
                    negative_conductivity.law_id,
                    positive_conductivity.law_id,
                    electrolyte_diffusivity_.law_id,
                    electrolyte_conductivity_.law_id,
                    electrolyte_transference.law_id,
                ],
                "entropic_law_ids": [negative_entropy.law_id, positive_entropy.law_id],
            }
        )

        self.spme_parameters = spme_parameters
        self.ambient_temperature_k = ambient
        self.volumetric_heat_capacity_j_m3_k = heat_capacity
        self.heat_transfer_coefficient_w_m2_k = heat_transfer
        self.cooling_surface_area_per_volume_m_inv = cooling_area
        self.battery_length_scale_m = battery_length
        self.battery_thermal_conductivity_w_m_k = thermal_conductivity
        self.typical_discharge_time_s = discharge_time
        self.typical_electrode_potential_v = electrode_potential
        self.negative_solid_conductivity = negative_conductivity
        self.positive_solid_conductivity = positive_conductivity
        self.electrolyte_diffusivity = electrolyte_diffusivity_
        self.electrolyte_conductivity = electrolyte_conductivity_
        self.electrolyte_transference_number = electrolyte_transference
        self.electrolyte_thermodynamic_factor = thermodynamic_factor
        self.negative_entropic_coefficient = negative_entropy
        self.positive_entropic_coefficient = positive_entropy
        self.electrochemical_property_support_id = electrochemical_support_id
        self.thermal_property_support_id = thermal_support_id
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "battery-tspme-brosa-planella-parameters",
                "spme_parameter_id": spme_parameters.parameter_id,
                "electrochemical_support_id": electrochemical_support_id,
                "thermal_support_id": thermal_support_id,
                "source_formulation_id": _SOURCE_FORMULATION_ID,
            }
        )


class BrosaPlanellaTspmeInitialCondition(StrictModule):
    """Marquis concentration initial data; Section 3 fixes initial T to ambient."""

    spme_initial_condition: Marquis2019SpmeInitialCondition

    def __init__(self, spme_initial_condition: Marquis2019SpmeInitialCondition, /):
        if not isinstance(spme_initial_condition, Marquis2019SpmeInitialCondition):
            raise TypeError(
                "spme_initial_condition must be Marquis2019SpmeInitialCondition."
            )
        self.spme_initial_condition = spme_initial_condition


class BrosaPlanellaTspmeState(StrictModule):
    """Marquis solid/electrolyte amounts and one homogeneous temperature state."""

    spme_state: Marquis2019SpmeState
    temperature_k: Array

    def __init__(
        self,
        spme_state: Marquis2019SpmeState,
        temperature_k: ArrayLike,
        /,
    ):
        if not isinstance(spme_state, Marquis2019SpmeState):
            raise TypeError("spme_state must be Marquis2019SpmeState.")
        temperature = jnp.asarray(temperature_k)
        leading_shape = spme_state.negative_amount_mol.shape[:-1]
        if temperature.shape != leading_shape:
            raise ValueError(
                "temperature_k must match the leading axes of the Marquis state."
            )
        if jnp.issubdtype(temperature.dtype, jnp.complexfloating):
            raise TypeError("temperature_k must be real-valued.")
        dtype = jnp.result_type(temperature, spme_state.negative_amount_mol, float)
        self.spme_state = Marquis2019SpmeState(
            spme_state.negative_amount_mol.astype(dtype),
            spme_state.positive_amount_mol.astype(dtype),
            spme_state.electrolyte_amount_mol.astype(dtype),
        )
        self.temperature_k = temperature.astype(dtype)


class BrosaPlanellaTspmeLedger(StrictModule):
    """Independent electrochemical, thermal-energy, support, and applicability evidence."""

    electrochemical: Marquis2019SpmeLedger
    initial_thermal_energy_j: Array
    final_thermal_energy_j: Array
    integrated_generated_heat_j: Array
    integrated_boundary_cooling_j: Array
    thermal_energy_balance_residual_j: Array
    maximum_heat_sum_residual_w_m3: Array
    maximum_power_identity_residual_w_m3: Array
    thermal_energy_balanced: Array
    property_support_valid: Array
    electrochemical_applicability_satisfied: Array
    thermal_applicability_satisfied: Array
    applicability_conditions_satisfied: Array
    domain_valid: Array
    finite: Array
    successful: Array


class BrosaPlanellaTspmePlan(StrictModule, NonTrainableState):
    """Static Marquis topology plus independent Section 3 TSPMe evidence policy."""

    spme_plan: Marquis2019SpmePlan
    applicability_small_parameter_threshold: float = eqx.field(static=True)
    conductivity_number_minimum: float = eqx.field(static=True)
    ledger_energy_absolute_tolerance_j: float = eqx.field(static=True)
    ledger_heat_absolute_tolerance_w_m3: float = eqx.field(static=True)
    ledger_relative_tolerance: float = eqx.field(static=True)
    source_formulation_id: str = eqx.field(static=True)
    applicability_support_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        spme_plan: Marquis2019SpmePlan,
        /,
        *,
        applicability_small_parameter_threshold: float = 0.1,
        conductivity_number_minimum: float = 1.0,
        ledger_energy_absolute_tolerance_j: float = 1.0e-4,
        ledger_heat_absolute_tolerance_w_m3: float = 1.0e-8,
        ledger_relative_tolerance: float = 1.0e-5,
    ):
        if not isinstance(spme_plan, Marquis2019SpmePlan):
            raise TypeError("spme_plan must be Marquis2019SpmePlan.")
        small = float(applicability_small_parameter_threshold)
        conductivity_minimum = float(conductivity_number_minimum)
        energy_tolerance = float(ledger_energy_absolute_tolerance_j)
        heat_tolerance = float(ledger_heat_absolute_tolerance_w_m3)
        relative = float(ledger_relative_tolerance)
        values = np.asarray(
            (small, conductivity_minimum, energy_tolerance, heat_tolerance, relative)
        )
        if (
            np.any(~np.isfinite(values))
            or small <= 0.0
            or small >= 1.0
            or conductivity_minimum <= 0.0
            or np.any(values[2:] < 0.0)
        ):
            raise ValueError(
                "TSPMe applicability thresholds must be finite and positive, the "
                "small parameter threshold must lie in (0, 1), and ledger "
                "tolerances must be finite and nonnegative."
            )
        applicability_support_id = canonical_fingerprint(
            {
                "kind": "battery-tspme-brosa-planella-applicability-support",
                "source_formulation_id": _SOURCE_FORMULATION_ID,
                "small_parameter_threshold": small,
                "conductivity_number_minimum": conductivity_minimum,
            }
        )
        self.spme_plan = spme_plan
        self.applicability_small_parameter_threshold = small
        self.conductivity_number_minimum = conductivity_minimum
        self.ledger_energy_absolute_tolerance_j = energy_tolerance
        self.ledger_heat_absolute_tolerance_w_m3 = heat_tolerance
        self.ledger_relative_tolerance = relative
        self.source_formulation_id = _SOURCE_FORMULATION_ID
        self.applicability_support_id = applicability_support_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "battery-tspme-brosa-planella-plan",
                "spme_plan_id": spme_plan.plan_id,
                "applicability_support_id": applicability_support_id,
                "energy_atol_j": energy_tolerance,
                "heat_atol_w_m3": heat_tolerance,
                "ledger_rtol": relative,
            }
        )

    def prepare(self, /) -> "PreparedBrosaPlanellaTspme":
        return PreparedBrosaPlanellaTspme(self)


class PreparedBrosaPlanellaTspme(StrictModule, NonTrainableState):
    """Prepared Marquis meshes retained as nontrainable TSPMe topology."""

    plan: BrosaPlanellaTspmePlan
    spme: PreparedMarquis2019Spme
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: BrosaPlanellaTspmePlan, /):
        if not isinstance(plan, BrosaPlanellaTspmePlan):
            raise TypeError("plan must be BrosaPlanellaTspmePlan.")
        spme = plan.spme_plan.prepare()
        self.plan = plan
        self.spme = spme
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-battery-tspme-brosa-planella",
                "plan_id": plan.plan_id,
                "spme_prepared_id": spme.prepared_id,
            }
        )

    def evaluate(
        self,
        state: BrosaPlanellaTspmeState,
        parameters: BrosaPlanellaTspmeParameters,
        terminal_current_a: ArrayLike,
        /,
    ) -> "_BrosaPlanellaTspmeEvaluation":
        return _evaluate(self, state, parameters, jnp.asarray(terminal_current_a))


class _BrosaPlanellaTspmeEvaluation(StrictModule):
    temperature_k: Array
    ambient_temperature_k: Array
    negative_surface_stoichiometry: Array
    positive_surface_stoichiometry: Array
    negative_surface_concentration_mol_m3: Array
    positive_surface_concentration_mol_m3: Array
    negative_average_concentration_mol_m3: Array
    positive_average_concentration_mol_m3: Array
    negative_electrolyte_mean_mol_m3: Array
    separator_electrolyte_mean_mol_m3: Array
    positive_electrolyte_mean_mol_m3: Array
    negative_solid_diffusivity_m2_s: Array
    positive_solid_diffusivity_m2_s: Array
    electrolyte_diffusivity_m2_s: Array
    electrolyte_transference_number: Array
    negative_solid_conductivity_s_m: Array
    positive_solid_conductivity_s_m: Array
    typical_electrolyte_conductivity_s_m: Array
    typical_electrolyte_thermodynamic_factor: Array
    negative_exchange_current_density_a_m2: Array
    positive_exchange_current_density_a_m2: Array
    particle_ocp_reference_v: Array
    particle_ocp_temperature_correction_v: Array
    particle_ocp_v: Array
    reaction_overpotential_v: Array
    electrolyte_concentration_overpotential_v: Array
    electrolyte_ohmic_loss_v: Array
    solid_ohmic_loss_v: Array
    voltage_v: Array
    passive_current_density_a_m2: Array
    paper_current_density_a_m2: Array
    solid_ohmic_heat_w_m3: Array
    electrolyte_concentration_heat_w_m3: Array
    electrolyte_ohmic_heat_w_m3: Array
    electrolyte_heat_w_m3: Array
    irreversible_reaction_heat_w_m3: Array
    reversible_reaction_heat_w_m3: Array
    generated_heat_w_m3: Array
    boundary_cooling_w_m3: Array
    net_heat_w_m3: Array
    temperature_rate_k_s: Array
    thermal_energy_j: Array
    heat_sum_residual_w_m3: Array
    solid_power_identity_residual_w_m3: Array
    electrolyte_power_identity_residual_w_m3: Array
    reaction_power_identity_residual_w_m3: Array
    small_overpotential_ratio: Array
    negative_solid_conductivity_number: Array
    positive_solid_conductivity_number: Array
    electrolyte_conductivity_number: Array
    biot_number: Array
    internal_conduction_to_cooling_ratio: Array
    internal_thermal_conduction_number: Array
    electrochemical_applicability_satisfied: Array
    thermal_applicability_satisfied: Array
    applicability_conditions_satisfied: Array
    solid_transport_support_valid: Array
    electrolyte_transport_support_valid: Array
    kinetics_support_valid: Array
    conductivity_support_valid: Array
    thermodynamic_factor_support_valid: Array
    ocp_support_valid: Array
    property_support_valid: Array
    domain_valid: Array
    total_solid_lithium_mol: Array
    total_electrolyte_lithium_mol: Array
    particle_transport: object
    electrolyte_transport: object


def _check_prepared(
    plan: BrosaPlanellaTspmePlan,
    prepared_model: PreparedBrosaPlanellaTspme,
    /,
) -> None:
    if not isinstance(prepared_model, PreparedBrosaPlanellaTspme):
        raise TypeError("prepared_model must be PreparedBrosaPlanellaTspme.")
    if prepared_model.plan.plan_id != plan.plan_id:
        raise ValueError("Prepared TSPMe topology does not belong to this adapter.")


def _evaluate_single(
    prepared: PreparedBrosaPlanellaTspme,
    state: BrosaPlanellaTspmeState,
    parameters: BrosaPlanellaTspmeParameters,
    terminal_current_a: Array,
    /,
) -> _BrosaPlanellaTspmeEvaluation:
    temperature = state.temperature_k
    base = parameters.spme_parameters
    base_spm = base.spm_parameters
    spm_parameters = eqx.tree_at(
        lambda value: value.temperature_k,
        base_spm,
        temperature,
    )
    particle_state = SpmState(
        state.spme_state.negative_amount_mol,
        state.spme_state.positive_amount_mol,
    )
    profile, particle_transport = _transport(
        prepared.spme.spm,
        particle_state,
        spm_parameters,
        terminal_current_a,
    )
    through_cell = prepared.spme.through_cell
    reconstruction_metrics = through_cell.metrics(
        negative_thickness_m=spm_parameters.negative_electrode_thickness_m,
        separator_thickness_m=base.separator_thickness_m,
        positive_thickness_m=spm_parameters.positive_electrode_thickness_m,
        negative_porosity=base.negative_electrolyte_porosity,
        separator_porosity=base.separator_electrolyte_porosity,
        positive_porosity=base.positive_electrolyte_porosity,
        bruggeman_coefficient=base.bruggeman_coefficient,
        electrolyte_diffusivity_m2_s=jnp.asarray(1.0),
        electrode_area_m2=spm_parameters.electrode_area_m2,
    )
    safe_electrolyte_amount = jnp.where(
        jnp.isfinite(state.spme_state.electrolyte_amount_mol),
        state.spme_state.electrolyte_amount_mol,
        0.0,
    )
    reconstructed_concentration = jnp.maximum(
        safe_electrolyte_amount / reconstruction_metrics.storage_volume_m3,
        jnp.finfo(safe_electrolyte_amount.dtype).tiny,
    )
    electrolyte_diffusivity = parameters.electrolyte_diffusivity.evaluate(
        reconstructed_concentration, temperature
    )
    electrolyte_transference = parameters.electrolyte_transference_number.evaluate(
        reconstructed_concentration, temperature
    )
    electrolyte_transport = through_cell.evaluate(
        state.spme_state.electrolyte_amount_mol,
        terminal_current_a,
        negative_thickness_m=spm_parameters.negative_electrode_thickness_m,
        separator_thickness_m=base.separator_thickness_m,
        positive_thickness_m=spm_parameters.positive_electrode_thickness_m,
        negative_porosity=base.negative_electrolyte_porosity,
        separator_porosity=base.separator_electrolyte_porosity,
        positive_porosity=base.positive_electrolyte_porosity,
        bruggeman_coefficient=base.bruggeman_coefficient,
        electrolyte_diffusivity_m2_s=electrolyte_diffusivity.values,
        electrode_area_m2=spm_parameters.electrode_area_m2,
        transference_number=electrolyte_transference.values,
        maximum_absolute_current_a=spm_parameters.maximum_absolute_current_a,
    )
    current = jnp.asarray(terminal_current_a)
    safe_current = jnp.where(
        particle_transport.current_valid & profile.domain_valid,
        current,
        0.0,
    )
    passive_current_density = safe_current / spm_parameters.electrode_area_m2
    paper_current_density = -passive_current_density

    negative_surface_concentration = (
        particle_transport.negative.surface_concentration_mol_m3
    )
    positive_surface_concentration = (
        particle_transport.positive.surface_concentration_mol_m3
    )
    negative_surface_stoichiometry = (
        negative_surface_concentration
        / spm_parameters.negative_maximum_concentration_mol_m3
    )
    positive_surface_stoichiometry = (
        positive_surface_concentration
        / spm_parameters.positive_maximum_concentration_mol_m3
    )
    through_cell = prepared.spme.through_cell
    weights = through_cell.reference_cell_widths
    concentration = electrolyte_transport.concentration_mol_m3
    negative_electrolyte_mean = _region_mean(
        concentration, weights, through_cell.negative_mask
    )
    separator_electrolyte_mean = _region_mean(
        concentration, weights, through_cell.separator_mask
    )
    positive_electrolyte_mean = _region_mean(
        concentration, weights, through_cell.positive_mask
    )

    negative_ocp_reference = spm_parameters.negative_open_circuit_potential.evaluate(
        negative_surface_stoichiometry
    )
    positive_ocp_reference = spm_parameters.positive_open_circuit_potential.evaluate(
        positive_surface_stoichiometry
    )
    negative_entropy = parameters.negative_entropic_coefficient.evaluate(
        negative_surface_stoichiometry
    )
    positive_entropy = parameters.positive_entropic_coefficient.evaluate(
        positive_surface_stoichiometry
    )
    reference_temperature = base_spm.temperature_k
    particle_ocp_reference = positive_ocp_reference.values - negative_ocp_reference.values
    particle_ocp_temperature_correction = (temperature - reference_temperature) * (
        positive_entropy.values - negative_entropy.values
    )
    particle_ocp = particle_ocp_reference + particle_ocp_temperature_correction

    typical_concentration = base.typical_electrolyte_concentration_mol_m3
    negative_occupancy = negative_surface_stoichiometry * (
        1.0 - negative_surface_stoichiometry
    )
    positive_occupancy = positive_surface_stoichiometry * (
        1.0 - positive_surface_stoichiometry
    )
    electrolyte_factor = jnp.sqrt(jnp.maximum(concentration / typical_concentration, 0.0))
    negative_exchange_cells = (
        4.0
        * profile.negative_exchange_current_density_scale_a_m2
        * jnp.sqrt(jnp.maximum(negative_occupancy, 0.0))
        * electrolyte_factor
    )
    positive_exchange_cells = (
        4.0
        * profile.positive_exchange_current_density_scale_a_m2
        * jnp.sqrt(jnp.maximum(positive_occupancy, 0.0))
        * electrolyte_factor
    )
    negative_exchange = _region_mean(
        negative_exchange_cells, weights, through_cell.negative_mask
    )
    positive_exchange = _region_mean(
        positive_exchange_cells, weights, through_cell.positive_mask
    )
    negative_denominator = (
        profile.negative_specific_surface_area_m2_m3
        * spm_parameters.negative_electrode_thickness_m
        * negative_exchange_cells
    )
    positive_denominator = (
        profile.positive_specific_surface_area_m2_m3
        * spm_parameters.positive_electrode_thickness_m
        * positive_exchange_cells
    )
    negative_asinh = _stable_asinh_ratio(paper_current_density, negative_denominator)
    positive_asinh = _stable_asinh_ratio(paper_current_density, positive_denominator)
    asinh_sum = _region_mean(
        negative_asinh, weights, through_cell.negative_mask
    ) + _region_mean(positive_asinh, weights, through_cell.positive_mask)
    thermal_voltage_twice = 2.0 * _GAS_CONSTANT_J_MOL_K * temperature / _FARADAY_C_MOL
    reaction_overpotential = -thermal_voltage_twice * asinh_sum

    safe_concentration = jnp.maximum(concentration, jnp.finfo(concentration.dtype).tiny)
    thermodynamic_factor = parameters.electrolyte_thermodynamic_factor.evaluate(
        concentration
    )
    thermodynamic_primitive = _electrolyte_thermodynamic_primitive(
        parameters.electrolyte_transference_number,
        parameters.electrolyte_thermodynamic_factor,
        safe_concentration,
        temperature,
    )
    thermodynamic_primitive_difference = _region_mean(
        thermodynamic_primitive, weights, through_cell.positive_mask
    ) - _region_mean(thermodynamic_primitive, weights, through_cell.negative_mask)
    electrolyte_concentration_overpotential = (
        thermal_voltage_twice * thermodynamic_primitive_difference
    )

    negative_solid_conductivity = parameters.negative_solid_conductivity.evaluate(
        temperature
    )
    positive_solid_conductivity = parameters.positive_solid_conductivity.evaluate(
        temperature
    )
    electrolyte_conductivity_concentration = parameters.electrolyte_conductivity.evaluate(
        concentration, temperature
    )
    electrolyte_conductivity_cells = electrolyte_conductivity_concentration.values
    solid_conductivity_valid = (
        negative_solid_conductivity.support
        & positive_solid_conductivity.support
        & jnp.isfinite(negative_solid_conductivity.values)
        & jnp.isfinite(positive_solid_conductivity.values)
        & (negative_solid_conductivity.values > 0.0)
        & (positive_solid_conductivity.values > 0.0)
    )
    electrolyte_conductivity_valid = (
        jnp.all(electrolyte_conductivity_concentration.support)
        & jnp.all(jnp.isfinite(electrolyte_conductivity_cells))
        & jnp.all(electrolyte_conductivity_cells > 0.0)
    )
    safe_negative_conductivity = jnp.where(
        solid_conductivity_valid,
        negative_solid_conductivity.values,
        1.0,
    )
    safe_positive_conductivity = jnp.where(
        solid_conductivity_valid,
        positive_solid_conductivity.values,
        1.0,
    )
    safe_electrolyte_conductivity = jnp.where(
        electrolyte_conductivity_concentration.support
        & jnp.isfinite(electrolyte_conductivity_cells)
        & (electrolyte_conductivity_cells > 0.0),
        electrolyte_conductivity_cells,
        1.0,
    )
    lengths = jnp.stack(
        (
            spm_parameters.negative_electrode_thickness_m,
            base.separator_thickness_m,
            spm_parameters.positive_electrode_thickness_m,
        )
    )
    total_length = jnp.sum(lengths)
    metrics = through_cell.metrics(
        negative_thickness_m=lengths[0],
        separator_thickness_m=lengths[1],
        positive_thickness_m=lengths[2],
        negative_porosity=base.negative_electrolyte_porosity,
        separator_porosity=base.separator_electrolyte_porosity,
        positive_porosity=base.positive_electrolyte_porosity,
        bruggeman_coefficient=base.bruggeman_coefficient,
        electrolyte_diffusivity_m2_s=electrolyte_diffusivity.values,
        electrode_area_m2=spm_parameters.electrode_area_m2,
    )
    geometry_factor = metrics.porosity**base.bruggeman_coefficient
    left_current_fraction = through_cell.electrolyte_current_fraction_faces[:-1]
    right_current_fraction = through_cell.electrolyte_current_fraction_faces[1:]
    mean_square_current_fraction = (
        left_current_fraction**2
        + left_current_fraction * right_current_fraction
        + right_current_fraction**2
    ) / 3.0
    electrolyte_resistance_area = jnp.sum(
        metrics.cell_widths_m
        * mean_square_current_fraction
        / (safe_electrolyte_conductivity * geometry_factor)
    )
    electrolyte_ohmic_loss = passive_current_density * electrolyte_resistance_area
    solid_resistance_area = (
        lengths[0] / safe_negative_conductivity + lengths[2] / safe_positive_conductivity
    ) / 3.0
    solid_ohmic_loss = passive_current_density * solid_resistance_area
    voltage = (
        particle_ocp
        + reaction_overpotential
        + electrolyte_concentration_overpotential
        + electrolyte_ohmic_loss
        + solid_ohmic_loss
    )

    solid_ohmic_heat = passive_current_density * solid_ohmic_loss / total_length
    electrolyte_concentration_heat = (
        passive_current_density * electrolyte_concentration_overpotential / total_length
    )
    electrolyte_ohmic_heat = (
        passive_current_density * electrolyte_ohmic_loss / total_length
    )
    electrolyte_heat = electrolyte_concentration_heat + electrolyte_ohmic_heat
    irreversible_reaction_heat = (
        passive_current_density * reaction_overpotential / total_length
    )
    negative_peltier = temperature * negative_entropy.values
    positive_peltier = temperature * positive_entropy.values
    reversible_reaction_heat = (
        paper_current_density * (negative_peltier - positive_peltier) / total_length
    )
    generated_heat = (
        solid_ohmic_heat
        + electrolyte_heat
        + irreversible_reaction_heat
        + reversible_reaction_heat
    )
    boundary_cooling = (
        parameters.heat_transfer_coefficient_w_m2_k
        * parameters.cooling_surface_area_per_volume_m_inv
        * (temperature - parameters.ambient_temperature_k)
    )
    net_heat = generated_heat - boundary_cooling
    temperature_rate = net_heat / parameters.volumetric_heat_capacity_j_m3_k
    cell_volume = spm_parameters.electrode_area_m2 * total_length
    thermal_energy = (
        parameters.volumetric_heat_capacity_j_m3_k * cell_volume * temperature
    )

    heat_sum_residual = generated_heat - (
        solid_ohmic_heat
        + electrolyte_concentration_heat
        + electrolyte_ohmic_heat
        + irreversible_reaction_heat
        + reversible_reaction_heat
    )
    solid_power_identity_residual = solid_ohmic_heat - (
        passive_current_density * solid_ohmic_loss / total_length
    )
    electrolyte_power_identity_residual = electrolyte_heat - (
        passive_current_density
        * (electrolyte_concentration_overpotential + electrolyte_ohmic_loss)
        / total_length
    )
    reaction_power_identity_residual = irreversible_reaction_heat - (
        passive_current_density * reaction_overpotential / total_length
    )

    ambient = parameters.ambient_temperature_k
    typical_current_density = (
        spm_parameters.maximum_absolute_current_a / spm_parameters.electrode_area_m2
    )
    ambient_negative_conductivity = parameters.negative_solid_conductivity.evaluate(
        ambient
    )
    ambient_positive_conductivity = parameters.positive_solid_conductivity.evaluate(
        ambient
    )
    current_typical_electrolyte_conductivity = (
        parameters.electrolyte_conductivity.evaluate(
            typical_concentration, temperature
        ).values
    )
    ambient_typical_electrolyte_conductivity = (
        parameters.electrolyte_conductivity.evaluate(
            typical_concentration, ambient
        ).values
    )
    current_typical_electrolyte_diffusivity = parameters.electrolyte_diffusivity.evaluate(
        typical_concentration, temperature
    ).values
    current_typical_transference = parameters.electrolyte_transference_number.evaluate(
        typical_concentration, temperature
    ).values
    potential_ratio = (
        _GAS_CONSTANT_J_MOL_K
        * ambient
        / (_FARADAY_C_MOL * parameters.typical_electrode_potential_v)
    )
    conductivity_scale = (
        _GAS_CONSTANT_J_MOL_K
        * ambient
        / (_FARADAY_C_MOL * total_length * typical_current_density)
    )
    negative_conductivity_number = (
        conductivity_scale * ambient_negative_conductivity.values
    )
    positive_conductivity_number = (
        conductivity_scale * ambient_positive_conductivity.values
    )
    electrolyte_conductivity_number = (
        conductivity_scale * ambient_typical_electrolyte_conductivity
    )
    biot_number = (
        parameters.heat_transfer_coefficient_w_m2_k
        * parameters.battery_length_scale_m
        / parameters.battery_thermal_conductivity_w_m_k
    )
    conduction_to_cooling = _finite_large_ratio(
        parameters.battery_thermal_conductivity_w_m_k,
        parameters.heat_transfer_coefficient_w_m2_k * parameters.battery_length_scale_m,
    )
    internal_conduction_number = (
        parameters.battery_thermal_conductivity_w_m_k
        * parameters.typical_discharge_time_s
        / (
            parameters.battery_length_scale_m**2
            * parameters.volumetric_heat_capacity_j_m3_k
        )
    )
    small_threshold = prepared.plan.applicability_small_parameter_threshold
    conductivity_minimum = prepared.plan.conductivity_number_minimum
    electrochemical_applicability = (
        (potential_ratio <= small_threshold)
        & (negative_conductivity_number >= conductivity_minimum)
        & (positive_conductivity_number >= conductivity_minimum)
        & (electrolyte_conductivity_number >= conductivity_minimum)
    )
    thermal_applicability = (biot_number <= small_threshold) & (
        internal_conduction_number >= 1.0 / small_threshold
    )
    applicability = electrochemical_applicability & thermal_applicability

    negative_diffusivity_support = spm_parameters.negative_solid_diffusivity.evaluate(
        temperature
    )
    positive_diffusivity_support = spm_parameters.positive_solid_diffusivity.evaluate(
        temperature
    )
    negative_kinetics_support = spm_parameters.negative_exchange_current_density.evaluate(
        temperature
    )
    positive_kinetics_support = spm_parameters.positive_exchange_current_density.evaluate(
        temperature
    )
    solid_transport_support = (
        negative_diffusivity_support.support & positive_diffusivity_support.support
    )
    electrolyte_transport_support = (
        jnp.all(electrolyte_diffusivity.support)
        & jnp.all(jnp.isfinite(electrolyte_diffusivity.values))
        & jnp.all(electrolyte_diffusivity.values > 0.0)
        & jnp.all(electrolyte_transference.support)
        & jnp.all(jnp.isfinite(electrolyte_transference.values))
        & jnp.all(electrolyte_transference.values >= 0.0)
        & jnp.all(electrolyte_transference.values <= 1.0)
    )
    kinetics_support = (
        negative_kinetics_support.support & positive_kinetics_support.support
    )
    conductivity_support = solid_conductivity_valid & electrolyte_conductivity_valid
    thermodynamic_factor_support = (
        jnp.all(thermodynamic_factor.support)
        & jnp.all(jnp.isfinite(thermodynamic_factor.values))
        & jnp.all(thermodynamic_factor.values > 0.0)
    )
    ocp_support = (
        negative_ocp_reference.support
        & positive_ocp_reference.support
        & negative_entropy.support
        & positive_entropy.support
    )
    property_support = (
        solid_transport_support
        & electrolyte_transport_support
        & kinetics_support
        & conductivity_support
        & thermodynamic_factor_support
        & ocp_support
    )
    active_current = safe_current != 0.0
    exchange_valid = (
        jnp.isfinite(negative_exchange)
        & jnp.isfinite(positive_exchange)
        & (negative_exchange >= 0.0)
        & (positive_exchange >= 0.0)
        & (~active_current | ((negative_exchange > 0.0) & (positive_exchange > 0.0)))
    )
    scalar_evidence = jnp.stack(
        (
            temperature,
            particle_ocp,
            reaction_overpotential,
            electrolyte_concentration_overpotential,
            electrolyte_ohmic_loss,
            solid_ohmic_loss,
            voltage,
            solid_ohmic_heat,
            electrolyte_concentration_heat,
            electrolyte_ohmic_heat,
            irreversible_reaction_heat,
            reversible_reaction_heat,
            generated_heat,
            boundary_cooling,
            net_heat,
            temperature_rate,
            thermal_energy,
            potential_ratio,
            negative_conductivity_number,
            positive_conductivity_number,
            electrolyte_conductivity_number,
            biot_number,
            conduction_to_cooling,
            internal_conduction_number,
        )
    )
    domain_valid = (
        particle_transport.domain_valid
        & electrolyte_transport.domain_valid
        & metrics.domain_valid
        & property_support
        & exchange_valid
        & jnp.isfinite(temperature)
        & (temperature > 0.0)
        & jnp.all(jnp.isfinite(scalar_evidence))
    )
    return _BrosaPlanellaTspmeEvaluation(
        temperature_k=temperature,
        ambient_temperature_k=parameters.ambient_temperature_k,
        negative_surface_stoichiometry=negative_surface_stoichiometry,
        positive_surface_stoichiometry=positive_surface_stoichiometry,
        negative_surface_concentration_mol_m3=negative_surface_concentration,
        positive_surface_concentration_mol_m3=positive_surface_concentration,
        negative_average_concentration_mol_m3=(
            particle_transport.negative.average_concentration_mol_m3
        ),
        positive_average_concentration_mol_m3=(
            particle_transport.positive.average_concentration_mol_m3
        ),
        negative_electrolyte_mean_mol_m3=negative_electrolyte_mean,
        separator_electrolyte_mean_mol_m3=separator_electrolyte_mean,
        positive_electrolyte_mean_mol_m3=positive_electrolyte_mean,
        negative_solid_diffusivity_m2_s=profile.negative_diffusivity_m2_s,
        positive_solid_diffusivity_m2_s=profile.positive_diffusivity_m2_s,
        electrolyte_diffusivity_m2_s=current_typical_electrolyte_diffusivity,
        negative_solid_conductivity_s_m=negative_solid_conductivity.values,
        positive_solid_conductivity_s_m=positive_solid_conductivity.values,
        typical_electrolyte_conductivity_s_m=(current_typical_electrolyte_conductivity),
        electrolyte_transference_number=current_typical_transference,
        typical_electrolyte_thermodynamic_factor=(
            parameters.electrolyte_thermodynamic_factor.evaluate(
                typical_concentration
            ).values
        ),
        negative_exchange_current_density_a_m2=negative_exchange,
        positive_exchange_current_density_a_m2=positive_exchange,
        particle_ocp_reference_v=particle_ocp_reference,
        particle_ocp_temperature_correction_v=particle_ocp_temperature_correction,
        particle_ocp_v=particle_ocp,
        reaction_overpotential_v=reaction_overpotential,
        electrolyte_concentration_overpotential_v=(
            electrolyte_concentration_overpotential
        ),
        electrolyte_ohmic_loss_v=electrolyte_ohmic_loss,
        solid_ohmic_loss_v=solid_ohmic_loss,
        voltage_v=voltage,
        passive_current_density_a_m2=passive_current_density,
        paper_current_density_a_m2=paper_current_density,
        solid_ohmic_heat_w_m3=solid_ohmic_heat,
        electrolyte_concentration_heat_w_m3=electrolyte_concentration_heat,
        electrolyte_ohmic_heat_w_m3=electrolyte_ohmic_heat,
        electrolyte_heat_w_m3=electrolyte_heat,
        irreversible_reaction_heat_w_m3=irreversible_reaction_heat,
        reversible_reaction_heat_w_m3=reversible_reaction_heat,
        generated_heat_w_m3=generated_heat,
        boundary_cooling_w_m3=boundary_cooling,
        net_heat_w_m3=net_heat,
        temperature_rate_k_s=temperature_rate,
        thermal_energy_j=thermal_energy,
        heat_sum_residual_w_m3=heat_sum_residual,
        solid_power_identity_residual_w_m3=solid_power_identity_residual,
        electrolyte_power_identity_residual_w_m3=(electrolyte_power_identity_residual),
        reaction_power_identity_residual_w_m3=reaction_power_identity_residual,
        small_overpotential_ratio=potential_ratio,
        negative_solid_conductivity_number=negative_conductivity_number,
        positive_solid_conductivity_number=positive_conductivity_number,
        electrolyte_conductivity_number=electrolyte_conductivity_number,
        biot_number=biot_number,
        internal_conduction_to_cooling_ratio=conduction_to_cooling,
        internal_thermal_conduction_number=internal_conduction_number,
        electrochemical_applicability_satisfied=electrochemical_applicability,
        thermal_applicability_satisfied=thermal_applicability,
        applicability_conditions_satisfied=applicability,
        solid_transport_support_valid=solid_transport_support,
        electrolyte_transport_support_valid=electrolyte_transport_support,
        kinetics_support_valid=kinetics_support,
        conductivity_support_valid=conductivity_support,
        thermodynamic_factor_support_valid=thermodynamic_factor_support,
        ocp_support_valid=ocp_support,
        property_support_valid=property_support,
        domain_valid=domain_valid,
        total_solid_lithium_mol=(
            particle_transport.negative.total_amount_mol
            + particle_transport.positive.total_amount_mol
        ),
        total_electrolyte_lithium_mol=electrolyte_transport.total_amount_mol,
        particle_transport=particle_transport,
        electrolyte_transport=electrolyte_transport,
    )


def _evaluate(
    prepared: PreparedBrosaPlanellaTspme,
    state: BrosaPlanellaTspmeState,
    parameters: BrosaPlanellaTspmeParameters,
    terminal_current_a: Array,
    /,
) -> _BrosaPlanellaTspmeEvaluation:
    if not isinstance(state, BrosaPlanellaTspmeState):
        raise TypeError("state must be BrosaPlanellaTspmeState.")
    if not isinstance(parameters, BrosaPlanellaTspmeParameters):
        raise TypeError("parameters must be BrosaPlanellaTspmeParameters.")
    leading_shape = state.temperature_k.shape
    current = jnp.asarray(terminal_current_a)
    if current.shape == ():
        current = jnp.broadcast_to(current, leading_shape)
    elif current.shape != leading_shape:
        raise ValueError("terminal_current_a must be scalar or match state leading axes.")
    if not leading_shape:
        return _evaluate_single(prepared, state, parameters, current)
    negative_count = state.spme_state.negative_amount_mol.shape[-1]
    positive_count = state.spme_state.positive_amount_mol.shape[-1]
    electrolyte_count = state.spme_state.electrolyte_amount_mol.shape[-1]
    flat = jax.vmap(
        lambda negative, positive, electrolyte, temperature, applied_current: (
            _evaluate_single(
                prepared,
                BrosaPlanellaTspmeState(
                    Marquis2019SpmeState(negative, positive, electrolyte), temperature
                ),
                parameters,
                applied_current,
            )
        )
    )(
        state.spme_state.negative_amount_mol.reshape((-1, negative_count)),
        state.spme_state.positive_amount_mol.reshape((-1, positive_count)),
        state.spme_state.electrolyte_amount_mol.reshape((-1, electrolyte_count)),
        state.temperature_k.reshape((-1,)),
        current.reshape((-1,)),
    )
    return jax.tree.map(
        lambda value: value.reshape(leading_shape + value.shape[1:]),
        flat,
    )


class _BrosaPlanellaTspmeVectorField(StrictModule, NonTrainableState):
    prepared: PreparedBrosaPlanellaTspme

    def __call__(
        self,
        time_s: Array,
        state: BrosaPlanellaTspmeState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> BrosaPlanellaTspmeState:
        parameters = runtime_inputs.parameters
        if not isinstance(parameters, BrosaPlanellaTspmeParameters):
            raise TypeError("TSPMe runtime parameters have the wrong type.")
        current = runtime_inputs.current(time_s, state.temperature_k)
        evaluation = _evaluate(self.prepared, state, parameters, current)
        return BrosaPlanellaTspmeState(
            Marquis2019SpmeState(
                evaluation.particle_transport.negative.amount_rate_mol_s,
                evaluation.particle_transport.positive.amount_rate_mol_s,
                evaluation.electrolyte_transport.amount_rate_mol_s,
            ),
            evaluation.temperature_rate_k_s,
        )


_OBSERVABLE_NAMES = (
    "voltage_v",
    "temperature_k",
    "stoichiometry:negative",
    "stoichiometry:positive",
    "negative_surface_concentration_mol_m3",
    "positive_surface_concentration_mol_m3",
    "negative_average_concentration_mol_m3",
    "positive_average_concentration_mol_m3",
    "negative_electrolyte_mean_mol_m3",
    "separator_electrolyte_mean_mol_m3",
    "positive_electrolyte_mean_mol_m3",
    "solid_lithium_mol",
    "electrolyte_lithium_mol",
    "total_lithium_mol",
    "thermal_energy_j",
    "voltage:particle_ocp_reference_v",
    "voltage:particle_ocp_temperature_correction_v",
    "voltage:particle_ocp_v",
    "voltage:reaction_overpotential_v",
    "voltage:electrolyte_concentration_overpotential_v",
    "voltage:electrolyte_ohmic_loss_v",
    "voltage:solid_ohmic_loss_v",
    "heat:solid_ohmic_w_m3",
    "heat:electrolyte_concentration_w_m3",
    "heat:electrolyte_ohmic_w_m3",
    "heat:electrolyte_w_m3",
    "heat:irreversible_reaction_w_m3",
    "heat:reversible_reaction_w_m3",
    "heat:generated_w_m3",
    "heat:boundary_cooling_w_m3",
    "heat:net_w_m3",
    "temperature_rate_k_s",
    "negative_solid_diffusivity_m2_s",
    "positive_solid_diffusivity_m2_s",
    "electrolyte_diffusivity_m2_s",
    "electrolyte_transference_number",
    "negative_solid_conductivity_s_m",
    "positive_solid_conductivity_s_m",
    "electrolyte_conductivity_s_m",
    "electrolyte_thermodynamic_factor",
    "negative_exchange_current_density_a_m2",
    "positive_exchange_current_density_a_m2",
    "passive_current_density_a_m2",
    "paper_current_density_a_m2",
    "applicability:small_overpotential_ratio",
    "applicability:negative_solid_conductivity_number",
    "applicability:positive_solid_conductivity_number",
    "applicability:electrolyte_conductivity_number",
    "applicability:biot_number",
    "applicability:internal_conduction_to_cooling_ratio",
    "applicability:internal_thermal_conduction_number",
    "applicability:electrochemical_satisfied",
    "applicability:thermal_satisfied",
    "applicability:conditions_satisfied",
    "support:solid_transport_valid",
    "support:electrolyte_transport_valid",
    "support:kinetics_valid",
    "support:conductivity_valid",
    "support:thermodynamic_factor_valid",
    "support:ocp_valid",
    "support:properties_valid",
    "identity:heat_sum_residual_w_m3",
    "identity:solid_power_residual_w_m3",
    "identity:electrolyte_power_residual_w_m3",
    "identity:reaction_power_residual_w_m3",
    "absorbed_power_w",
    "current_envelope_margin_a",
)
_OBSERVABLE_UNITS = (
    "V",
    "K",
    "1",
    "1",
    "mol/m3",
    "mol/m3",
    "mol/m3",
    "mol/m3",
    "mol/m3",
    "mol/m3",
    "mol/m3",
    "mol",
    "mol",
    "mol",
    "J",
    "V",
    "V",
    "V",
    "V",
    "V",
    "V",
    "V",
    "W/m3",
    "W/m3",
    "W/m3",
    "W/m3",
    "W/m3",
    "W/m3",
    "W/m3",
    "W/m3",
    "W/m3",
    "K/s",
    "m2/s",
    "m2/s",
    "m2/s",
    "1",
    "S/m",
    "S/m",
    "S/m",
    "1",
    "A/m2",
    "A/m2",
    "A/m2",
    "A/m2",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "W/m3",
    "W/m3",
    "W/m3",
    "W/m3",
    "W",
    "A",
)


class _MarquisSolutionView(StrictModule):
    times: Array
    states: Marquis2019SpmeState
    valid: Array


class BrosaPlanellaTspmeAdapter(StrictModule, NonTrainableState):
    """Native ODE adapter for the unsimplified dimensional Section 3 TSPMe."""

    plan: BrosaPlanellaTspmePlan
    model_id: str = eqx.field(static=True)
    equation_form: Literal["ode"] = eqx.field(static=True)
    observable_names: tuple[str, ...] = eqx.field(static=True)
    observable_units: tuple[str, ...] = eqx.field(static=True)
    source_formulation_id: str = eqx.field(static=True)

    def __init__(self, plan: BrosaPlanellaTspmePlan, /):
        if not isinstance(plan, BrosaPlanellaTspmePlan):
            raise TypeError("plan must be BrosaPlanellaTspmePlan.")
        self.plan = plan
        self.model_id = "battery:tspme:brosa-planella:base-prescribed-current"
        self.equation_form = "ode"
        self.observable_names = _OBSERVABLE_NAMES
        self.observable_units = _OBSERVABLE_UNITS
        self.source_formulation_id = _SOURCE_FORMULATION_ID

    def prepare(self, /) -> PreparedBrosaPlanellaTspme:
        return self.plan.prepare()

    def initial_state(
        self,
        prepared_model: PreparedBrosaPlanellaTspme,
        parameters: BrosaPlanellaTspmeParameters,
        initial_condition: BrosaPlanellaTspmeInitialCondition,
        /,
    ) -> BrosaPlanellaTspmeState:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(parameters, BrosaPlanellaTspmeParameters):
            raise TypeError("parameters must be BrosaPlanellaTspmeParameters.")
        if not isinstance(initial_condition, BrosaPlanellaTspmeInitialCondition):
            raise TypeError(
                "initial_condition must be BrosaPlanellaTspmeInitialCondition."
            )
        spme_state = Marquis2019SpmeAdapter(self.plan.spme_plan).initial_state(
            prepared_model.spme,
            parameters.spme_parameters,
            initial_condition.spme_initial_condition,
        )
        return BrosaPlanellaTspmeState(
            spme_state,
            parameters.ambient_temperature_k,
        )

    def problem(
        self,
        prepared_model: PreparedBrosaPlanellaTspme,
        initial_state: BrosaPlanellaTspmeState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> DifferentialProblem:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(initial_state, BrosaPlanellaTspmeState):
            raise TypeError("initial_state must be BrosaPlanellaTspmeState.")
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        if not isinstance(runtime_inputs.parameters, BrosaPlanellaTspmeParameters):
            raise TypeError("TSPMe runtime parameters have the wrong type.")
        expected_negative = prepared_model.spme.spm.negative_particle.shell_count
        expected_positive = prepared_model.spme.spm.positive_particle.shell_count
        expected_electrolyte = prepared_model.spme.through_cell.cell_count
        state = initial_state.spme_state
        if state.negative_amount_mol.shape != (expected_negative,):
            raise ValueError(
                f"Initial negative TSPMe state must have shape ({expected_negative},)."
            )
        if state.positive_amount_mol.shape != (expected_positive,):
            raise ValueError(
                f"Initial positive TSPMe state must have shape ({expected_positive},)."
            )
        if state.electrolyte_amount_mol.shape != (expected_electrolyte,):
            raise ValueError(
                "Initial electrolyte TSPMe state must have shape "
                f"({expected_electrolyte},)."
            )
        if initial_state.temperature_k.shape != ():
            raise ValueError("Initial TSPMe temperature must be scalar.")
        return DifferentialProblem(
            _BrosaPlanellaTspmeVectorField(prepared_model),
            initial_state,
            t0=runtime_inputs.input_policy.times[0],
            t1=runtime_inputs.input_policy.times[-1],
            args=runtime_inputs,
            problem_id=canonical_fingerprint(
                {
                    "kind": "battery-tspme-brosa-planella-problem",
                    "model_id": self.model_id,
                    "prepared_id": prepared_model.prepared_id,
                    "parameter_id": runtime_inputs.parameters.parameter_id,
                    "protocol_id": runtime_inputs.protocol_id,
                    "source_formulation_id": self.source_formulation_id,
                }
            ),
        )

    def observe(
        self,
        prepared_model: PreparedBrosaPlanellaTspme,
        times_s: Array,
        states: BrosaPlanellaTspmeState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> BatteryModelOutput:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(states, BrosaPlanellaTspmeState):
            raise TypeError("states must be BrosaPlanellaTspmeState.")
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        if not isinstance(runtime_inputs.parameters, BrosaPlanellaTspmeParameters):
            raise TypeError("TSPMe runtime parameters have the wrong type.")
        times = jnp.asarray(times_s)
        expected_negative = times.shape + (
            prepared_model.spme.spm.negative_particle.shell_count,
        )
        expected_positive = times.shape + (
            prepared_model.spme.spm.positive_particle.shell_count,
        )
        expected_electrolyte = times.shape + (
            prepared_model.spme.through_cell.cell_count,
        )
        state = states.spme_state
        if state.negative_amount_mol.shape != expected_negative:
            raise ValueError(
                f"Negative TSPMe observation state must have shape {expected_negative}."
            )
        if state.positive_amount_mol.shape != expected_positive:
            raise ValueError(
                f"Positive TSPMe observation state must have shape {expected_positive}."
            )
        if state.electrolyte_amount_mol.shape != expected_electrolyte:
            raise ValueError(
                "Electrolyte TSPMe observation state must have shape "
                f"{expected_electrolyte}."
            )
        if states.temperature_k.shape != times.shape:
            raise ValueError("TSPMe observation temperature must match times_s shape.")
        current = jax.vmap(runtime_inputs.observed_current)(times.reshape((-1,))).reshape(
            times.shape
        )
        evaluation = _evaluate(prepared_model, states, runtime_inputs.parameters, current)
        total_lithium = (
            evaluation.total_solid_lithium_mol + evaluation.total_electrolyte_lithium_mol
        )
        values = jnp.stack(
            (
                evaluation.voltage_v,
                evaluation.temperature_k,
                evaluation.negative_surface_stoichiometry,
                evaluation.positive_surface_stoichiometry,
                evaluation.negative_surface_concentration_mol_m3,
                evaluation.positive_surface_concentration_mol_m3,
                evaluation.negative_average_concentration_mol_m3,
                evaluation.positive_average_concentration_mol_m3,
                evaluation.negative_electrolyte_mean_mol_m3,
                evaluation.separator_electrolyte_mean_mol_m3,
                evaluation.positive_electrolyte_mean_mol_m3,
                evaluation.total_solid_lithium_mol,
                evaluation.total_electrolyte_lithium_mol,
                total_lithium,
                evaluation.thermal_energy_j,
                evaluation.particle_ocp_reference_v,
                evaluation.particle_ocp_temperature_correction_v,
                evaluation.particle_ocp_v,
                evaluation.reaction_overpotential_v,
                evaluation.electrolyte_concentration_overpotential_v,
                evaluation.electrolyte_ohmic_loss_v,
                evaluation.solid_ohmic_loss_v,
                evaluation.solid_ohmic_heat_w_m3,
                evaluation.electrolyte_concentration_heat_w_m3,
                evaluation.electrolyte_ohmic_heat_w_m3,
                evaluation.electrolyte_heat_w_m3,
                evaluation.irreversible_reaction_heat_w_m3,
                evaluation.reversible_reaction_heat_w_m3,
                evaluation.generated_heat_w_m3,
                evaluation.boundary_cooling_w_m3,
                evaluation.net_heat_w_m3,
                evaluation.temperature_rate_k_s,
                evaluation.negative_solid_diffusivity_m2_s,
                evaluation.positive_solid_diffusivity_m2_s,
                evaluation.electrolyte_diffusivity_m2_s,
                evaluation.electrolyte_transference_number,
                evaluation.negative_solid_conductivity_s_m,
                evaluation.positive_solid_conductivity_s_m,
                evaluation.typical_electrolyte_conductivity_s_m,
                evaluation.typical_electrolyte_thermodynamic_factor,
                evaluation.negative_exchange_current_density_a_m2,
                evaluation.positive_exchange_current_density_a_m2,
                evaluation.passive_current_density_a_m2,
                evaluation.paper_current_density_a_m2,
                evaluation.small_overpotential_ratio,
                evaluation.negative_solid_conductivity_number,
                evaluation.positive_solid_conductivity_number,
                evaluation.electrolyte_conductivity_number,
                evaluation.biot_number,
                evaluation.internal_conduction_to_cooling_ratio,
                evaluation.internal_thermal_conduction_number,
                evaluation.electrochemical_applicability_satisfied.astype(
                    evaluation.voltage_v.dtype
                ),
                evaluation.thermal_applicability_satisfied.astype(
                    evaluation.voltage_v.dtype
                ),
                evaluation.applicability_conditions_satisfied.astype(
                    evaluation.voltage_v.dtype
                ),
                evaluation.solid_transport_support_valid.astype(
                    evaluation.voltage_v.dtype
                ),
                evaluation.electrolyte_transport_support_valid.astype(
                    evaluation.voltage_v.dtype
                ),
                evaluation.kinetics_support_valid.astype(evaluation.voltage_v.dtype),
                evaluation.conductivity_support_valid.astype(evaluation.voltage_v.dtype),
                evaluation.thermodynamic_factor_support_valid.astype(
                    evaluation.voltage_v.dtype
                ),
                evaluation.ocp_support_valid.astype(evaluation.voltage_v.dtype),
                evaluation.property_support_valid.astype(evaluation.voltage_v.dtype),
                evaluation.heat_sum_residual_w_m3,
                evaluation.solid_power_identity_residual_w_m3,
                evaluation.electrolyte_power_identity_residual_w_m3,
                evaluation.reaction_power_identity_residual_w_m3,
                evaluation.voltage_v * current,
                runtime_inputs.parameters.spme_parameters.spm_parameters.maximum_absolute_current_a
                - jnp.abs(current),
            ),
            axis=-1,
        )
        domain_valid = evaluation.domain_valid & jnp.all(jnp.isfinite(values), axis=-1)
        return BatteryModelOutput(values, domain_valid)

    def ledger(
        self,
        prepared_model: PreparedBrosaPlanellaTspme,
        native_solution,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> BrosaPlanellaTspmeLedger:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        if not isinstance(runtime_inputs.parameters, BrosaPlanellaTspmeParameters):
            raise TypeError("TSPMe runtime parameters have the wrong type.")
        states = native_solution.states
        if not isinstance(states, BrosaPlanellaTspmeState):
            raise TypeError("TSPMe native solution states have the wrong type.")
        valid = jnp.asarray(native_solution.valid, dtype=bool)
        times = jnp.asarray(native_solution.times)
        valid_count = jnp.sum(valid.astype(jnp.int32))
        final_index = jnp.maximum(valid_count - 1, 0)
        parameters = runtime_inputs.parameters

        marquis_runtime = BatteryRuntimeInputs(
            parameters.spme_parameters,
            runtime_inputs.input_policy,
            runtime_inputs.stop_thresholds,
            protocol_id=runtime_inputs.protocol_id,
            observation_input_policy=runtime_inputs.observation_input_policy,
        )
        marquis_view = _MarquisSolutionView(
            times,
            states.spme_state,
            valid,
        )
        electrochemical = Marquis2019SpmeAdapter(self.plan.spme_plan).ledger(
            prepared_model.spme,
            marquis_view,
            marquis_runtime,
        )
        currents = jax.vmap(runtime_inputs.observed_current)(times)
        evaluation = _evaluate(prepared_model, states, parameters, currents)
        spm = parameters.spme_parameters.spm_parameters
        total_length = (
            spm.negative_electrode_thickness_m
            + parameters.spme_parameters.separator_thickness_m
            + spm.positive_electrode_thickness_m
        )
        cell_volume = spm.electrode_area_m2 * total_length
        initial_energy = (
            parameters.volumetric_heat_capacity_j_m3_k
            * cell_volume
            * states.temperature_k[0]
        )
        final_energy = (
            parameters.volumetric_heat_capacity_j_m3_k
            * cell_volume
            * states.temperature_k[final_index]
        )
        interval_valid = valid[:-1] & valid[1:]
        interval_duration = jnp.maximum(times[1:] - times[:-1], 0.0)
        interval_midpoints = 0.5 * (times[:-1] + times[1:])
        interval_currents = jax.vmap(
            lambda time: runtime_inputs.current(time, jnp.asarray(0.0))
        )(interval_midpoints)
        left_states = jax.tree.map(lambda value: value[:-1], states)
        right_states = jax.tree.map(lambda value: value[1:], states)
        left_evaluation = _evaluate(
            prepared_model,
            left_states,
            parameters,
            interval_currents,
        )
        right_evaluation = _evaluate(
            prepared_model,
            right_states,
            parameters,
            interval_currents,
        )
        left_generated_power = left_evaluation.generated_heat_w_m3 * cell_volume
        right_generated_power = right_evaluation.generated_heat_w_m3 * cell_volume
        left_boundary_power = left_evaluation.boundary_cooling_w_m3 * cell_volume
        right_boundary_power = right_evaluation.boundary_cooling_w_m3 * cell_volume
        generated_energy = jnp.sum(
            jnp.where(
                interval_valid,
                0.5 * interval_duration * (left_generated_power + right_generated_power),
                0.0,
            )
        )
        boundary_energy = jnp.sum(
            jnp.where(
                interval_valid,
                0.5 * interval_duration * (left_boundary_power + right_boundary_power),
                0.0,
            )
        )
        energy_residual = (
            final_energy - initial_energy - generated_energy + boundary_energy
        )

        def valid_max(values):
            return jnp.max(jnp.where(valid, jnp.abs(values), 0.0))

        maximum_heat_residual = valid_max(evaluation.heat_sum_residual_w_m3)
        maximum_power_residual = jnp.maximum(
            valid_max(evaluation.solid_power_identity_residual_w_m3),
            jnp.maximum(
                valid_max(evaluation.electrolyte_power_identity_residual_w_m3),
                valid_max(evaluation.reaction_power_identity_residual_w_m3),
            ),
        )
        property_support = (valid_count > 0) & jnp.all(
            jnp.where(valid, evaluation.property_support_valid, True)
        )
        electrochemical_applicability = (valid_count > 0) & jnp.all(
            jnp.where(
                valid,
                evaluation.electrochemical_applicability_satisfied,
                True,
            )
        )
        thermal_applicability = (valid_count > 0) & jnp.all(
            jnp.where(valid, evaluation.thermal_applicability_satisfied, True)
        )
        applicability = electrochemical_applicability & thermal_applicability
        domain_valid = (valid_count > 0) & jnp.all(
            jnp.where(valid, evaluation.domain_valid, True)
        )
        finite = (valid_count > 0) & jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        initial_energy,
                        final_energy,
                        generated_energy,
                        boundary_energy,
                        energy_residual,
                        maximum_heat_residual,
                        maximum_power_residual,
                    )
                )
            )
        )
        energy_scale = jnp.maximum(
            jnp.abs(final_energy - initial_energy),
            jnp.abs(generated_energy) + jnp.abs(boundary_energy),
        )
        thermal_balanced = jnp.abs(energy_residual) <= (
            prepared_model.plan.ledger_energy_absolute_tolerance_j
            + prepared_model.plan.ledger_relative_tolerance * energy_scale
        )
        heat_identities = (
            jnp.maximum(maximum_heat_residual, maximum_power_residual)
            <= prepared_model.plan.ledger_heat_absolute_tolerance_w_m3
        )
        successful = (
            electrochemical.successful
            & thermal_balanced
            & heat_identities
            & property_support
            & applicability
            & domain_valid
            & finite
        )
        return BrosaPlanellaTspmeLedger(
            electrochemical=electrochemical,
            initial_thermal_energy_j=initial_energy,
            final_thermal_energy_j=final_energy,
            integrated_generated_heat_j=generated_energy,
            integrated_boundary_cooling_j=boundary_energy,
            thermal_energy_balance_residual_j=energy_residual,
            maximum_heat_sum_residual_w_m3=maximum_heat_residual,
            maximum_power_identity_residual_w_m3=maximum_power_residual,
            thermal_energy_balanced=thermal_balanced,
            property_support_valid=property_support,
            electrochemical_applicability_satisfied=(electrochemical_applicability),
            thermal_applicability_satisfied=thermal_applicability,
            applicability_conditions_satisfied=applicability,
            domain_valid=domain_valid,
            finite=finite,
            successful=successful,
        )


__all__ = [
    "BrosaPlanellaTspmeAdapter",
    "BrosaPlanellaTspmeInitialCondition",
    "BrosaPlanellaTspmeLedger",
    "BrosaPlanellaTspmeParameters",
    "BrosaPlanellaTspmePlan",
    "BrosaPlanellaTspmeState",
    "PreparedBrosaPlanellaTspme",
]
