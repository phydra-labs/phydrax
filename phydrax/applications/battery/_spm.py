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
from ._particle import (
    BatteryParticleEvaluation,
    BatteryParticlePlan,
    PreparedBatteryParticle,
)
from ._properties import ConstantPropertyLaw, TabulatedPropertyLaw
from ._results import BatteryModelOutput


_FARADAY_C_MOL = 96485.33212
_GAS_CONSTANT_J_MOL_K = 8.31446261815324
_PROPERTY_TYPES = (ConstantPropertyLaw, TabulatedPropertyLaw)
BatteryPropertyLaw: TypeAlias = ConstantPropertyLaw | TabulatedPropertyLaw
LimitingElectrode: TypeAlias = Literal["balanced", "negative", "positive"]


def _scalar(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape != () or jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must be one real scalar.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return array


def _property_law(
    value: BatteryPropertyLaw,
    name: str,
    /,
    *,
    coordinate: str,
    value_unit: str,
) -> BatteryPropertyLaw:
    if not isinstance(value, _PROPERTY_TYPES):
        raise TypeError(f"{name} must be a ConstantPropertyLaw or TabulatedPropertyLaw.")
    expected_coordinate_unit = "K" if coordinate == "temperature" else "1"
    if (
        value.coordinate != coordinate
        or value.coordinate_unit != expected_coordinate_unit
    ):
        raise ValueError(
            f"{name} must use the {coordinate!r} coordinate in "
            f"{expected_coordinate_unit!r}."
        )
    if value.value_unit != value_unit:
        raise ValueError(f"{name} must use value unit {value_unit!r}.")
    return value


def _solid_diffusivity_law(
    value: BatteryPropertyLaw,
    name: str,
    /,
) -> BatteryPropertyLaw:
    if not isinstance(value, _PROPERTY_TYPES):
        raise TypeError(f"{name} must be a ConstantPropertyLaw or TabulatedPropertyLaw.")
    coordinate_units = {
        "temperature": "K",
        "stoichiometry": "1",
        "solid_lithium_concentration": "mol/m3",
    }
    if value.coordinate not in coordinate_units:
        raise ValueError(
            f"{name} must use temperature, stoichiometry, or "
            "solid_lithium_concentration coordinates."
        )
    if value.coordinate_unit != coordinate_units[value.coordinate]:
        raise ValueError(
            f"{name} coordinate {value.coordinate!r} must use unit "
            f"{coordinate_units[value.coordinate]!r}."
        )
    if value.value_unit != "m2/s":
        raise ValueError(f"{name} must use value unit 'm2/s'.")
    return value


def _solid_diffusivity_query(
    law: BatteryPropertyLaw,
    temperature_k: Array,
    stoichiometric_endpoints: Array,
    maximum_concentration_mol_m3: Array,
    /,
) -> Array:
    if law.coordinate == "temperature":
        return temperature_k
    if law.coordinate == "stoichiometry":
        return stoichiometric_endpoints
    return stoichiometric_endpoints * maximum_concentration_mol_m3


def _limiting_electrode(value: LimitingElectrode, /) -> LimitingElectrode:
    if value not in ("balanced", "negative", "positive"):
        raise ValueError(
            "limiting_electrode must be 'balanced', 'negative', or 'positive'."
        )
    return value


def _capacity_rule_valid(
    negative_capacity_c: Array,
    positive_capacity_c: Array,
    limiting_electrode: LimitingElectrode,
    relative_tolerance: float,
    /,
) -> Array:
    scale = jnp.maximum(negative_capacity_c, positive_capacity_c)
    balanced = (
        jnp.abs(negative_capacity_c - positive_capacity_c) <= relative_tolerance * scale
    )
    if limiting_electrode == "balanced":
        return balanced
    if limiting_electrode == "negative":
        return negative_capacity_c <= positive_capacity_c + relative_tolerance * scale
    return positive_capacity_c <= negative_capacity_c + relative_tolerance * scale


class SpmParameters(StrictModule):
    """Dynamic SI geometry, material, kinetic, OCP, and support data for one SPM profile."""

    electrode_area_m2: Array
    negative_electrode_thickness_m: Array
    positive_electrode_thickness_m: Array
    negative_active_material_volume_fraction: Array
    positive_active_material_volume_fraction: Array
    negative_particle_radius_m: Array
    positive_particle_radius_m: Array
    negative_maximum_concentration_mol_m3: Array
    positive_maximum_concentration_mol_m3: Array
    temperature_k: Array
    maximum_absolute_current_a: Array
    negative_stoichiometry_at_empty: Array
    negative_stoichiometry_at_full: Array
    positive_stoichiometry_at_empty: Array
    positive_stoichiometry_at_full: Array
    negative_solid_diffusivity: BatteryPropertyLaw
    positive_solid_diffusivity: BatteryPropertyLaw
    negative_exchange_current_density: BatteryPropertyLaw
    positive_exchange_current_density: BatteryPropertyLaw
    negative_open_circuit_potential: BatteryPropertyLaw
    positive_open_circuit_potential: BatteryPropertyLaw
    limiting_electrode: LimitingElectrode = eqx.field(static=True)
    capacity_balance_relative_tolerance: float = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        electrode_area_m2: ArrayLike,
        negative_electrode_thickness_m: ArrayLike,
        positive_electrode_thickness_m: ArrayLike,
        negative_active_material_volume_fraction: ArrayLike,
        positive_active_material_volume_fraction: ArrayLike,
        negative_particle_radius_m: ArrayLike,
        positive_particle_radius_m: ArrayLike,
        negative_maximum_concentration_mol_m3: ArrayLike,
        positive_maximum_concentration_mol_m3: ArrayLike,
        temperature_k: ArrayLike,
        maximum_absolute_current_a: ArrayLike,
        negative_stoichiometry_at_empty: ArrayLike,
        negative_stoichiometry_at_full: ArrayLike,
        positive_stoichiometry_at_empty: ArrayLike,
        positive_stoichiometry_at_full: ArrayLike,
        negative_solid_diffusivity: BatteryPropertyLaw,
        positive_solid_diffusivity: BatteryPropertyLaw,
        negative_exchange_current_density: BatteryPropertyLaw,
        positive_exchange_current_density: BatteryPropertyLaw,
        negative_open_circuit_potential: BatteryPropertyLaw,
        positive_open_circuit_potential: BatteryPropertyLaw,
        limiting_electrode: LimitingElectrode = "balanced",
        capacity_balance_relative_tolerance: float = 1.0e-6,
    ):
        area = _scalar(electrode_area_m2, "electrode_area_m2")
        negative_thickness = _scalar(
            negative_electrode_thickness_m, "negative_electrode_thickness_m"
        )
        positive_thickness = _scalar(
            positive_electrode_thickness_m, "positive_electrode_thickness_m"
        )
        negative_fraction = _scalar(
            negative_active_material_volume_fraction,
            "negative_active_material_volume_fraction",
        )
        positive_fraction = _scalar(
            positive_active_material_volume_fraction,
            "positive_active_material_volume_fraction",
        )
        negative_radius = _scalar(
            negative_particle_radius_m, "negative_particle_radius_m"
        )
        positive_radius = _scalar(
            positive_particle_radius_m, "positive_particle_radius_m"
        )
        negative_cmax = _scalar(
            negative_maximum_concentration_mol_m3,
            "negative_maximum_concentration_mol_m3",
        )
        positive_cmax = _scalar(
            positive_maximum_concentration_mol_m3,
            "positive_maximum_concentration_mol_m3",
        )
        temperature = _scalar(temperature_k, "temperature_k")
        maximum_current = _scalar(
            maximum_absolute_current_a, "maximum_absolute_current_a"
        )
        negative_empty = _scalar(
            negative_stoichiometry_at_empty, "negative_stoichiometry_at_empty"
        )
        negative_full = _scalar(
            negative_stoichiometry_at_full, "negative_stoichiometry_at_full"
        )
        positive_empty = _scalar(
            positive_stoichiometry_at_empty, "positive_stoichiometry_at_empty"
        )
        positive_full = _scalar(
            positive_stoichiometry_at_full, "positive_stoichiometry_at_full"
        )
        negative_diffusivity = _solid_diffusivity_law(
            negative_solid_diffusivity,
            "negative_solid_diffusivity",
        )
        positive_diffusivity = _solid_diffusivity_law(
            positive_solid_diffusivity,
            "positive_solid_diffusivity",
        )
        negative_exchange = _property_law(
            negative_exchange_current_density,
            "negative_exchange_current_density",
            coordinate="temperature",
            value_unit="A/m2",
        )
        positive_exchange = _property_law(
            positive_exchange_current_density,
            "positive_exchange_current_density",
            coordinate="temperature",
            value_unit="A/m2",
        )
        negative_ocp = _property_law(
            negative_open_circuit_potential,
            "negative_open_circuit_potential",
            coordinate="stoichiometry",
            value_unit="V",
        )
        positive_ocp = _property_law(
            positive_open_circuit_potential,
            "positive_open_circuit_potential",
            coordinate="stoichiometry",
            value_unit="V",
        )
        limiting = _limiting_electrode(limiting_electrode)
        tolerance = float(capacity_balance_relative_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "capacity_balance_relative_tolerance must be finite and nonnegative."
            )

        positive_geometry = (
            jnp.isfinite(area)
            & (area > 0.0)
            & jnp.isfinite(negative_thickness)
            & (negative_thickness > 0.0)
            & jnp.isfinite(positive_thickness)
            & (positive_thickness > 0.0)
            & jnp.isfinite(negative_radius)
            & (negative_radius > 0.0)
            & jnp.isfinite(positive_radius)
            & (positive_radius > 0.0)
            & jnp.isfinite(negative_cmax)
            & (negative_cmax > 0.0)
            & jnp.isfinite(positive_cmax)
            & (positive_cmax > 0.0)
            & jnp.isfinite(temperature)
            & (temperature > 0.0)
            & jnp.isfinite(maximum_current)
            & (maximum_current > 0.0)
        )
        fractions_valid = (
            jnp.isfinite(negative_fraction)
            & (negative_fraction > 0.0)
            & (negative_fraction <= 1.0)
            & jnp.isfinite(positive_fraction)
            & (positive_fraction > 0.0)
            & (positive_fraction <= 1.0)
        )
        endpoints_valid = (
            jnp.isfinite(negative_empty)
            & jnp.isfinite(negative_full)
            & jnp.isfinite(positive_empty)
            & jnp.isfinite(positive_full)
            & (negative_empty >= 0.0)
            & (negative_full <= 1.0)
            & (negative_full > negative_empty)
            & (positive_full >= 0.0)
            & (positive_empty <= 1.0)
            & (positive_empty > positive_full)
        )
        negative_solid_volume = area * negative_thickness * negative_fraction
        positive_solid_volume = area * positive_thickness * positive_fraction
        negative_capacity = (
            _FARADAY_C_MOL
            * negative_solid_volume
            * negative_cmax
            * (negative_full - negative_empty)
        )
        positive_capacity = (
            _FARADAY_C_MOL
            * positive_solid_volume
            * positive_cmax
            * (positive_empty - positive_full)
        )
        capacity_valid = _capacity_rule_valid(
            negative_capacity,
            positive_capacity,
            limiting,
            tolerance,
        )

        negative_diffusivity_result = negative_diffusivity.evaluate(
            _solid_diffusivity_query(
                negative_diffusivity,
                temperature,
                jnp.stack((negative_empty, negative_full)),
                negative_cmax,
            )
        )
        positive_diffusivity_result = positive_diffusivity.evaluate(
            _solid_diffusivity_query(
                positive_diffusivity,
                temperature,
                jnp.stack((positive_full, positive_empty)),
                positive_cmax,
            )
        )
        negative_exchange_result = negative_exchange.evaluate(temperature)
        positive_exchange_result = positive_exchange.evaluate(temperature)
        negative_endpoint_ocp = negative_ocp.evaluate(
            jnp.stack((negative_empty, negative_full))
        )
        positive_endpoint_ocp = positive_ocp.evaluate(
            jnp.stack((positive_full, positive_empty))
        )
        properties_valid = (
            jnp.all(
                negative_diffusivity_result.support
                & jnp.isfinite(negative_diffusivity_result.values)
                & (negative_diffusivity_result.values > 0.0)
            )
            & jnp.all(
                positive_diffusivity_result.support
                & jnp.isfinite(positive_diffusivity_result.values)
                & (positive_diffusivity_result.values > 0.0)
            )
            & negative_exchange_result.support
            & positive_exchange_result.support
            & jnp.isfinite(negative_exchange_result.values)
            & (negative_exchange_result.values > 0.0)
            & jnp.isfinite(positive_exchange_result.values)
            & (positive_exchange_result.values > 0.0)
            & jnp.all(negative_endpoint_ocp.support)
            & jnp.all(positive_endpoint_ocp.support)
            & jnp.all(jnp.isfinite(negative_endpoint_ocp.values))
            & jnp.all(jnp.isfinite(positive_endpoint_ocp.values))
        )
        area = eqx.error_if(
            area,
            ~(positive_geometry & fractions_valid & endpoints_valid),
            "SPM geometry, temperature, support current, fractions, and stoichiometric "
            "endpoints must be finite and physically ordered.",
        )
        area = eqx.error_if(
            area,
            ~capacity_valid,
            "SPM electrode capacities are neither balanced nor consistent with the "
            "declared limiting electrode.",
        )
        area = eqx.error_if(
            area,
            ~properties_valid,
            "SPM property laws must support the fixed temperature and operating-window "
            "endpoints with finite positive transport and kinetic values.",
        )

        self.electrode_area_m2 = area
        self.negative_electrode_thickness_m = negative_thickness
        self.positive_electrode_thickness_m = positive_thickness
        self.negative_active_material_volume_fraction = negative_fraction
        self.positive_active_material_volume_fraction = positive_fraction
        self.negative_particle_radius_m = negative_radius
        self.positive_particle_radius_m = positive_radius
        self.negative_maximum_concentration_mol_m3 = negative_cmax
        self.positive_maximum_concentration_mol_m3 = positive_cmax
        self.temperature_k = temperature
        self.maximum_absolute_current_a = maximum_current
        self.negative_stoichiometry_at_empty = negative_empty
        self.negative_stoichiometry_at_full = negative_full
        self.positive_stoichiometry_at_empty = positive_empty
        self.positive_stoichiometry_at_full = positive_full
        self.negative_solid_diffusivity = negative_diffusivity
        self.positive_solid_diffusivity = positive_diffusivity
        self.negative_exchange_current_density = negative_exchange
        self.positive_exchange_current_density = positive_exchange
        self.negative_open_circuit_potential = negative_ocp
        self.positive_open_circuit_potential = positive_ocp
        self.limiting_electrode = limiting
        self.capacity_balance_relative_tolerance = tolerance
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "battery-spm-parameters",
                "limiting_electrode": limiting,
                "capacity_balance_relative_tolerance": tolerance,
                "negative_diffusivity_law_id": negative_diffusivity.law_id,
                "positive_diffusivity_law_id": positive_diffusivity.law_id,
                "negative_exchange_law_id": negative_exchange.law_id,
                "positive_exchange_law_id": positive_exchange.law_id,
                "negative_ocp_law_id": negative_ocp.law_id,
                "positive_ocp_law_id": positive_ocp.law_id,
            }
        )


class SpmInitialCondition(StrictModule):
    """Uniform initial electrode stoichiometries for an SPM run."""

    negative_stoichiometry: Array
    positive_stoichiometry: Array

    def __init__(
        self,
        negative_stoichiometry: ArrayLike,
        positive_stoichiometry: ArrayLike,
        /,
    ):
        negative = _scalar(negative_stoichiometry, "negative_stoichiometry")
        positive = _scalar(positive_stoichiometry, "positive_stoichiometry")
        negative = eqx.error_if(
            negative,
            ~(
                jnp.isfinite(negative)
                & jnp.isfinite(positive)
                & (negative >= 0.0)
                & (negative <= 1.0)
                & (positive >= 0.0)
                & (positive <= 1.0)
            ),
            "Initial SPM stoichiometries must be finite and lie in [0, 1].",
        )
        self.negative_stoichiometry = negative
        self.positive_stoichiometry = positive


class SpmState(StrictModule):
    """Two extensive radial solid-lithium amount arrays."""

    negative_amount_mol: Array
    positive_amount_mol: Array

    def __init__(self, negative_amount_mol: ArrayLike, positive_amount_mol: ArrayLike, /):
        negative = jnp.asarray(negative_amount_mol)
        positive = jnp.asarray(positive_amount_mol)
        if negative.ndim < 1 or positive.ndim < 1:
            raise ValueError("SPM amount states must each contain a radial shell axis.")
        if jnp.issubdtype(negative.dtype, jnp.complexfloating) or jnp.issubdtype(
            positive.dtype, jnp.complexfloating
        ):
            raise TypeError("SPM amount states must be real-valued.")
        dtype = jnp.result_type(negative, positive, float)
        self.negative_amount_mol = negative.astype(dtype)
        self.positive_amount_mol = positive.astype(dtype)


class SpmLedger(StrictModule):
    """Endpoint lithium, charge, and prescribed-current conservation evidence."""

    initial_negative_lithium_mol: Array
    final_negative_lithium_mol: Array
    initial_positive_lithium_mol: Array
    final_positive_lithium_mol: Array
    initial_total_lithium_mol: Array
    final_total_lithium_mol: Array
    lithium_conservation_residual_mol: Array
    charge_conservation_residual_c: Array
    integrated_terminal_charge_c: Array
    negative_current_integral_residual_c: Array
    positive_current_integral_residual_c: Array
    lithium_conserved: Array
    charge_conserved: Array
    current_conserved: Array
    finite: Array
    successful: Array


class PrescribedCurrentSpmPlan(StrictModule, NonTrainableState):
    """Static two-particle topology and conservation tolerances for the isothermal SPM."""

    negative_particle: BatteryParticlePlan
    positive_particle: BatteryParticlePlan
    ledger_amount_absolute_tolerance_mol: float = eqx.field(static=True)
    ledger_charge_absolute_tolerance_c: float = eqx.field(static=True)
    ledger_relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        negative_shell_count: int,
        positive_shell_count: int | None = None,
        /,
        *,
        negative_reference_faces: ArrayLike | None = None,
        positive_reference_faces: ArrayLike | None = None,
        ledger_amount_absolute_tolerance_mol: float = 1.0e-10,
        ledger_charge_absolute_tolerance_c: float = 1.0e-5,
        ledger_relative_tolerance: float = 1.0e-6,
    ):
        positive_count = (
            negative_shell_count if positive_shell_count is None else positive_shell_count
        )
        amount_atol = float(ledger_amount_absolute_tolerance_mol)
        charge_atol = float(ledger_charge_absolute_tolerance_c)
        relative = float(ledger_relative_tolerance)
        if (
            not np.isfinite(amount_atol)
            or amount_atol < 0.0
            or not np.isfinite(charge_atol)
            or charge_atol < 0.0
            or not np.isfinite(relative)
            or relative < 0.0
        ):
            raise ValueError("SPM ledger tolerances must be finite and nonnegative.")
        negative = BatteryParticlePlan(
            negative_shell_count,
            reference_faces=negative_reference_faces,
            particle_id="negative",
        )
        positive = BatteryParticlePlan(
            positive_count,
            reference_faces=positive_reference_faces,
            particle_id="positive",
        )
        self.negative_particle = negative
        self.positive_particle = positive
        self.ledger_amount_absolute_tolerance_mol = amount_atol
        self.ledger_charge_absolute_tolerance_c = charge_atol
        self.ledger_relative_tolerance = relative
        self.plan_id = canonical_fingerprint(
            {
                "kind": "battery-spm-plan",
                "negative_particle": negative.plan_id,
                "positive_particle": positive.plan_id,
                "ledger_amount_atol_mol": amount_atol,
                "ledger_charge_atol_c": charge_atol,
                "ledger_rtol": relative,
            }
        )

    def prepare(self, /) -> "PreparedPrescribedCurrentSpm":
        return PreparedPrescribedCurrentSpm(self)


class PreparedPrescribedCurrentSpm(StrictModule, NonTrainableState):
    """Prepared nontrainable radial topology for dynamic SPM profile parameters."""

    plan: PrescribedCurrentSpmPlan
    negative_particle: PreparedBatteryParticle
    positive_particle: PreparedBatteryParticle
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: PrescribedCurrentSpmPlan, /):
        if not isinstance(plan, PrescribedCurrentSpmPlan):
            raise TypeError("plan must be a PrescribedCurrentSpmPlan.")
        negative = plan.negative_particle.prepare()
        positive = plan.positive_particle.prepare()
        self.plan = plan
        self.negative_particle = negative
        self.positive_particle = positive
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-battery-spm",
                "plan_id": plan.plan_id,
                "negative_particle": negative.prepared_id,
                "positive_particle": positive.prepared_id,
            }
        )


class _SpmProfileEvaluation(StrictModule):
    negative_support_volume_m3: Array
    positive_support_volume_m3: Array
    negative_solid_volume_m3: Array
    positive_solid_volume_m3: Array
    negative_particle_multiplicity: Array
    positive_particle_multiplicity: Array
    negative_active_surface_area_m2: Array
    positive_active_surface_area_m2: Array
    negative_specific_surface_area_m2_m3: Array
    positive_specific_surface_area_m2_m3: Array
    # Mid-window scalar evidence only; transport evaluation owns actual shell values.
    negative_diffusivity_m2_s: Array
    positive_diffusivity_m2_s: Array
    negative_exchange_current_density_scale_a_m2: Array
    positive_exchange_current_density_scale_a_m2: Array
    negative_theoretical_capacity_c: Array
    positive_theoretical_capacity_c: Array
    usable_capacity_c: Array
    negative_capacity_headroom_c: Array
    positive_capacity_headroom_c: Array
    domain_valid: Array


def _profile(parameters: SpmParameters, /) -> _SpmProfileEvaluation:
    negative_support_volume = (
        parameters.electrode_area_m2 * parameters.negative_electrode_thickness_m
    )
    positive_support_volume = (
        parameters.electrode_area_m2 * parameters.positive_electrode_thickness_m
    )
    negative_solid_volume = (
        negative_support_volume * parameters.negative_active_material_volume_fraction
    )
    positive_solid_volume = (
        positive_support_volume * parameters.positive_active_material_volume_fraction
    )
    negative_particle_volume = (
        (4.0 / 3.0) * jnp.pi * parameters.negative_particle_radius_m**3
    )
    positive_particle_volume = (
        (4.0 / 3.0) * jnp.pi * parameters.positive_particle_radius_m**3
    )
    negative_multiplicity = negative_solid_volume / negative_particle_volume
    positive_multiplicity = positive_solid_volume / positive_particle_volume
    negative_active_area = (
        4.0 * jnp.pi * parameters.negative_particle_radius_m**2 * negative_multiplicity
    )
    positive_active_area = (
        4.0 * jnp.pi * parameters.positive_particle_radius_m**2 * positive_multiplicity
    )
    negative_specific_area = negative_active_area / negative_support_volume
    positive_specific_area = positive_active_area / positive_support_volume

    negative_reference_stoichiometry = 0.5 * (
        parameters.negative_stoichiometry_at_empty
        + parameters.negative_stoichiometry_at_full
    )
    positive_reference_stoichiometry = 0.5 * (
        parameters.positive_stoichiometry_at_empty
        + parameters.positive_stoichiometry_at_full
    )
    negative_diffusivity_result = parameters.negative_solid_diffusivity.evaluate(
        _solid_diffusivity_query(
            parameters.negative_solid_diffusivity,
            parameters.temperature_k,
            negative_reference_stoichiometry,
            parameters.negative_maximum_concentration_mol_m3,
        )
    )
    positive_diffusivity_result = parameters.positive_solid_diffusivity.evaluate(
        _solid_diffusivity_query(
            parameters.positive_solid_diffusivity,
            parameters.temperature_k,
            positive_reference_stoichiometry,
            parameters.positive_maximum_concentration_mol_m3,
        )
    )
    negative_exchange_result = parameters.negative_exchange_current_density.evaluate(
        parameters.temperature_k
    )
    positive_exchange_result = parameters.positive_exchange_current_density.evaluate(
        parameters.temperature_k
    )
    negative_capacity = (
        _FARADAY_C_MOL
        * negative_solid_volume
        * parameters.negative_maximum_concentration_mol_m3
        * (
            parameters.negative_stoichiometry_at_full
            - parameters.negative_stoichiometry_at_empty
        )
    )
    positive_capacity = (
        _FARADAY_C_MOL
        * positive_solid_volume
        * parameters.positive_maximum_concentration_mol_m3
        * (
            parameters.positive_stoichiometry_at_empty
            - parameters.positive_stoichiometry_at_full
        )
    )
    usable_capacity = jnp.minimum(negative_capacity, positive_capacity)
    capacity_valid = _capacity_rule_valid(
        negative_capacity,
        positive_capacity,
        parameters.limiting_electrode,
        parameters.capacity_balance_relative_tolerance,
    )
    scalar_values = jnp.stack(
        (
            negative_support_volume,
            positive_support_volume,
            negative_solid_volume,
            positive_solid_volume,
            negative_multiplicity,
            positive_multiplicity,
            negative_active_area,
            positive_active_area,
            negative_specific_area,
            positive_specific_area,
            negative_diffusivity_result.values,
            positive_diffusivity_result.values,
            negative_exchange_result.values,
            positive_exchange_result.values,
            negative_capacity,
            positive_capacity,
            usable_capacity,
        )
    )
    parameter_bounds_valid = (
        jnp.isfinite(parameters.temperature_k)
        & (parameters.temperature_k > 0.0)
        & jnp.isfinite(parameters.maximum_absolute_current_a)
        & (parameters.maximum_absolute_current_a > 0.0)
        & (parameters.negative_active_material_volume_fraction > 0.0)
        & (parameters.negative_active_material_volume_fraction <= 1.0)
        & (parameters.positive_active_material_volume_fraction > 0.0)
        & (parameters.positive_active_material_volume_fraction <= 1.0)
        & (parameters.negative_stoichiometry_at_empty >= 0.0)
        & (parameters.negative_stoichiometry_at_full <= 1.0)
        & (
            parameters.negative_stoichiometry_at_full
            > parameters.negative_stoichiometry_at_empty
        )
        & (parameters.positive_stoichiometry_at_full >= 0.0)
        & (parameters.positive_stoichiometry_at_empty <= 1.0)
        & (
            parameters.positive_stoichiometry_at_empty
            > parameters.positive_stoichiometry_at_full
        )
    )
    domain_valid = (
        jnp.all(jnp.isfinite(scalar_values))
        & jnp.all(scalar_values > 0.0)
        & negative_diffusivity_result.support
        & positive_diffusivity_result.support
        & negative_exchange_result.support
        & positive_exchange_result.support
        & parameter_bounds_valid
        & capacity_valid
    )
    return _SpmProfileEvaluation(
        negative_support_volume,
        positive_support_volume,
        negative_solid_volume,
        positive_solid_volume,
        negative_multiplicity,
        positive_multiplicity,
        negative_active_area,
        positive_active_area,
        negative_specific_area,
        positive_specific_area,
        negative_diffusivity_result.values,
        positive_diffusivity_result.values,
        negative_exchange_result.values,
        positive_exchange_result.values,
        negative_capacity,
        positive_capacity,
        usable_capacity,
        negative_capacity - usable_capacity,
        positive_capacity - usable_capacity,
        domain_valid,
    )


def _shell_diffusivity(
    law: BatteryPropertyLaw,
    temperature_k: Array,
    concentration_mol_m3: Array,
    maximum_concentration_mol_m3: Array,
    /,
) -> tuple[Array, Array]:
    if law.coordinate == "temperature":
        result = law.evaluate(temperature_k)
        values = jnp.broadcast_to(result.values, concentration_mol_m3.shape)
        support = jnp.broadcast_to(result.support, concentration_mol_m3.shape)
    elif law.coordinate == "stoichiometry":
        result = law.evaluate(concentration_mol_m3 / maximum_concentration_mol_m3)
        values = result.values
        support = result.support
    else:
        result = law.evaluate(concentration_mol_m3)
        values = result.values
        support = result.support
    valid = jnp.all(
        support & jnp.isfinite(values) & (values > 0.0),
        axis=-1,
    )
    return values, valid


class _SpmTransportEvaluation(StrictModule):
    negative: BatteryParticleEvaluation
    positive: BatteryParticleEvaluation
    negative_diffusivity_m2_s: Array
    positive_diffusivity_m2_s: Array
    negative_minimum_diffusivity_m2_s: Array
    positive_minimum_diffusivity_m2_s: Array
    negative_outward_flux_mol_m2_s: Array
    positive_outward_flux_mol_m2_s: Array
    current_valid: Array
    domain_valid: Array


def _transport(
    prepared: PreparedPrescribedCurrentSpm,
    state: SpmState,
    parameters: SpmParameters,
    terminal_current_a: Array,
    /,
) -> tuple[_SpmProfileEvaluation, _SpmTransportEvaluation]:
    profile = _profile(parameters)
    current = jnp.asarray(terminal_current_a)
    leading_shape = state.negative_amount_mol.shape[:-1]
    if state.positive_amount_mol.shape[:-1] != leading_shape:
        raise ValueError("Negative and positive SPM amount leading axes must match.")
    if current.shape == ():
        current = jnp.broadcast_to(current, leading_shape)
    elif current.shape != leading_shape:
        raise ValueError("terminal_current_a must be scalar or match state leading axes.")
    negative_concentration = prepared.negative_particle.concentrations(
        state.negative_amount_mol,
        parameters.negative_particle_radius_m,
        profile.negative_particle_multiplicity,
    )
    positive_concentration = prepared.positive_particle.concentrations(
        state.positive_amount_mol,
        parameters.positive_particle_radius_m,
        profile.positive_particle_multiplicity,
    )
    negative_diffusivity, negative_diffusivity_valid = _shell_diffusivity(
        parameters.negative_solid_diffusivity,
        parameters.temperature_k,
        negative_concentration,
        parameters.negative_maximum_concentration_mol_m3,
    )
    positive_diffusivity, positive_diffusivity_valid = _shell_diffusivity(
        parameters.positive_solid_diffusivity,
        parameters.temperature_k,
        positive_concentration,
        parameters.positive_maximum_concentration_mol_m3,
    )
    current_valid = jnp.isfinite(current) & (
        jnp.abs(current) <= parameters.maximum_absolute_current_a
    )
    safe_current = jnp.where(current_valid & profile.domain_valid, current, 0.0)

    # Passive I > 0 charges the cell: positive solid oxidizes and negative solid reduces.
    negative_outward_flux = -safe_current / (
        _FARADAY_C_MOL * profile.negative_active_surface_area_m2
    )
    positive_outward_flux = safe_current / (
        _FARADAY_C_MOL * profile.positive_active_surface_area_m2
    )
    negative = prepared.negative_particle.evaluate(
        state.negative_amount_mol,
        particle_radius_m=parameters.negative_particle_radius_m,
        particle_multiplicity=profile.negative_particle_multiplicity,
        support_volume_m3=profile.negative_support_volume_m3,
        diffusivity_m2_s=negative_diffusivity,
        outward_molar_flux_mol_m2_s=negative_outward_flux,
    )
    positive = prepared.positive_particle.evaluate(
        state.positive_amount_mol,
        particle_radius_m=parameters.positive_particle_radius_m,
        particle_multiplicity=profile.positive_particle_multiplicity,
        support_volume_m3=profile.positive_support_volume_m3,
        diffusivity_m2_s=positive_diffusivity,
        outward_molar_flux_mol_m2_s=positive_outward_flux,
    )
    domain_valid = (
        profile.domain_valid
        & current_valid
        & negative_diffusivity_valid
        & positive_diffusivity_valid
        & negative.domain_valid
        & positive.domain_valid
    )
    return profile, _SpmTransportEvaluation(
        negative,
        positive,
        negative_diffusivity,
        positive_diffusivity,
        jnp.min(negative_diffusivity, axis=-1),
        jnp.min(positive_diffusivity, axis=-1),
        negative_outward_flux,
        positive_outward_flux,
        current_valid,
        domain_valid,
    )


class _SpmVectorField(StrictModule, NonTrainableState):
    prepared: PreparedPrescribedCurrentSpm

    def __call__(
        self,
        time_s: Array,
        state: SpmState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> SpmState:
        parameters = runtime_inputs.parameters
        if not isinstance(parameters, SpmParameters):
            raise TypeError("SPM runtime parameters must be SpmParameters.")
        current = runtime_inputs.current(time_s, state.negative_amount_mol)
        _, transport = _transport(self.prepared, state, parameters, current)
        return SpmState(
            transport.negative.amount_rate_mol_s,
            transport.positive.amount_rate_mol_s,
        )


def _stable_asinh_ratio(numerator: Array, denominator: Array, /) -> Array:
    """Evaluate asinh(numerator / denominator) without forming a huge quotient."""
    absolute_numerator = jnp.abs(numerator)
    safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
    scale = jnp.maximum(absolute_numerator, safe_denominator)
    safe_scale = jnp.where(scale > 0.0, scale, 1.0)
    normalized_numerator = absolute_numerator / safe_scale
    normalized_denominator = safe_denominator / safe_scale
    magnitude = (
        jnp.log(safe_scale)
        - jnp.log(safe_denominator)
        + jnp.log(
            normalized_numerator + jnp.hypot(normalized_numerator, normalized_denominator)
        )
    )
    return jnp.sign(numerator) * magnitude


def _symmetric_overpotential(
    interfacial_current_density_a_m2: Array,
    reference_exchange_current_density_a_m2: Array,
    surface_stoichiometry: Array,
    temperature_k: Array,
    /,
) -> tuple[Array, Array, Array]:
    occupancy = surface_stoichiometry * (1.0 - surface_stoichiometry)
    stoichiometry_valid = (
        jnp.isfinite(surface_stoichiometry)
        & (surface_stoichiometry >= 0.0)
        & (surface_stoichiometry <= 1.0)
    )
    # The factor two normalizes the symmetric occupancy factor to one at theta=1/2.
    exchange_current_density = (
        2.0
        * reference_exchange_current_density_a_m2
        * jnp.sqrt(jnp.maximum(occupancy, 0.0))
    )
    active_current = interfacial_current_density_a_m2 != 0.0
    exchange_valid = (
        jnp.isfinite(exchange_current_density)
        & (exchange_current_density >= 0.0)
        & (~active_current | (exchange_current_density > 0.0))
    )
    inverse_sinh = _stable_asinh_ratio(
        0.5 * interfacial_current_density_a_m2,
        exchange_current_density,
    )
    overpotential = (
        2.0 * _GAS_CONSTANT_J_MOL_K * temperature_k / _FARADAY_C_MOL * inverse_sinh
    )
    overpotential = jnp.where(active_current, overpotential, 0.0)
    valid = stoichiometry_valid & exchange_valid & jnp.isfinite(overpotential)
    return overpotential, exchange_current_density, valid


_OBSERVABLE_NAMES = (
    "voltage_v",
    "temperature_k",
    "stoichiometry:negative",
    "stoichiometry:positive",
    "negative_surface_concentration_mol_m3",
    "positive_surface_concentration_mol_m3",
    "negative_average_concentration_mol_m3",
    "positive_average_concentration_mol_m3",
    "negative_soc",
    "positive_soc",
    "soc_mismatch",
    "negative_ocp_v",
    "positive_ocp_v",
    "negative_overpotential_v",
    "positive_overpotential_v",
    "negative_exchange_current_density_a_m2",
    "positive_exchange_current_density_a_m2",
    "negative_interfacial_flux_mol_m2_s",
    "positive_interfacial_flux_mol_m2_s",
    "negative_specific_surface_area_m2_m3",
    "positive_specific_surface_area_m2_m3",
    "negative_theoretical_capacity_c",
    "positive_theoretical_capacity_c",
    "usable_capacity_c",
    "negative_capacity_headroom_c",
    "positive_capacity_headroom_c",
    "total_lithium_mol",
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
    "1",
    "1",
    "1",
    "V",
    "V",
    "V",
    "V",
    "A/m2",
    "A/m2",
    "mol/(m2*s)",
    "mol/(m2*s)",
    "m2/m3",
    "m2/m3",
    "C",
    "C",
    "C",
    "C",
    "C",
    "mol",
    "W",
    "A",
)


def _check_prepared(
    plan: PrescribedCurrentSpmPlan,
    prepared_model: PreparedPrescribedCurrentSpm,
    /,
) -> None:
    if not isinstance(prepared_model, PreparedPrescribedCurrentSpm):
        raise TypeError("prepared_model must be PreparedPrescribedCurrentSpm.")
    if prepared_model.plan.plan_id != plan.plan_id:
        raise ValueError("Prepared SPM topology does not belong to this adapter.")


class PrescribedCurrentSpmAdapter(StrictModule, NonTrainableState):
    """Isothermal prescribed-current SPM using conservative radial solid transport."""

    plan: PrescribedCurrentSpmPlan
    model_id: str = eqx.field(static=True)
    equation_form: Literal["ode"] = eqx.field(static=True)
    observable_names: tuple[str, ...] = eqx.field(static=True)
    observable_units: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, plan: PrescribedCurrentSpmPlan, /):
        if not isinstance(plan, PrescribedCurrentSpmPlan):
            raise TypeError("plan must be a PrescribedCurrentSpmPlan.")
        self.plan = plan
        self.model_id = "battery:spm:isothermal-prescribed-current"
        self.equation_form = "ode"
        self.observable_names = _OBSERVABLE_NAMES
        self.observable_units = _OBSERVABLE_UNITS

    def prepare(self, /) -> PreparedPrescribedCurrentSpm:
        return self.plan.prepare()

    def initial_state(
        self,
        prepared_model: PreparedPrescribedCurrentSpm,
        parameters: SpmParameters,
        initial_condition: SpmInitialCondition,
        /,
    ) -> SpmState:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(parameters, SpmParameters):
            raise TypeError("parameters must be SpmParameters.")
        if not isinstance(initial_condition, SpmInitialCondition):
            raise TypeError("initial_condition must be SpmInitialCondition.")
        profile = _profile(parameters)
        negative_concentration = (
            initial_condition.negative_stoichiometry
            * parameters.negative_maximum_concentration_mol_m3
        )
        positive_concentration = (
            initial_condition.positive_stoichiometry
            * parameters.positive_maximum_concentration_mol_m3
        )
        negative = prepared_model.negative_particle.initial_amounts(
            negative_concentration,
            parameters.negative_particle_radius_m,
            profile.negative_particle_multiplicity,
        )
        positive = prepared_model.positive_particle.initial_amounts(
            positive_concentration,
            parameters.positive_particle_radius_m,
            profile.positive_particle_multiplicity,
        )
        return SpmState(negative, positive)

    def problem(
        self,
        prepared_model: PreparedPrescribedCurrentSpm,
        initial_state: SpmState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> DifferentialProblem:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(initial_state, SpmState):
            raise TypeError("initial_state must be SpmState.")
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        if not isinstance(runtime_inputs.parameters, SpmParameters):
            raise TypeError("SPM runtime parameters must be SpmParameters.")
        expected_negative = prepared_model.negative_particle.shell_count
        expected_positive = prepared_model.positive_particle.shell_count
        if initial_state.negative_amount_mol.shape != (expected_negative,):
            raise ValueError(
                f"Initial negative SPM state must have shape ({expected_negative},)."
            )
        if initial_state.positive_amount_mol.shape != (expected_positive,):
            raise ValueError(
                f"Initial positive SPM state must have shape ({expected_positive},)."
            )
        return DifferentialProblem(
            _SpmVectorField(prepared_model),
            initial_state,
            t0=runtime_inputs.input_policy.times[0],
            t1=runtime_inputs.input_policy.times[-1],
            args=runtime_inputs,
            problem_id=canonical_fingerprint(
                {
                    "kind": "battery-spm-problem",
                    "model_id": self.model_id,
                    "prepared_id": prepared_model.prepared_id,
                    "parameter_id": runtime_inputs.parameters.parameter_id,
                    "protocol_id": runtime_inputs.protocol_id,
                }
            ),
        )

    def observe(
        self,
        prepared_model: PreparedPrescribedCurrentSpm,
        times_s: Array,
        states: SpmState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> BatteryModelOutput:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        if not isinstance(states, SpmState):
            raise TypeError("states must be SpmState.")
        if not isinstance(runtime_inputs.parameters, SpmParameters):
            raise TypeError("SPM runtime parameters must be SpmParameters.")
        times = jnp.asarray(times_s)
        expected_negative = times.shape + (prepared_model.negative_particle.shell_count,)
        expected_positive = times.shape + (prepared_model.positive_particle.shell_count,)
        if states.negative_amount_mol.shape != expected_negative:
            raise ValueError(
                f"Negative SPM observation state must have shape {expected_negative}."
            )
        if states.positive_amount_mol.shape != expected_positive:
            raise ValueError(
                f"Positive SPM observation state must have shape {expected_positive}."
            )
        flat_current = jax.vmap(runtime_inputs.observed_current)(times.reshape((-1,)))
        current = flat_current.reshape(times.shape)
        parameters = runtime_inputs.parameters
        profile, transport = _transport(prepared_model, states, parameters, current)

        negative_surface_stoichiometry = (
            transport.negative.surface_concentration_mol_m3
            / parameters.negative_maximum_concentration_mol_m3
        )
        positive_surface_stoichiometry = (
            transport.positive.surface_concentration_mol_m3
            / parameters.positive_maximum_concentration_mol_m3
        )
        negative_average_stoichiometry = (
            transport.negative.average_concentration_mol_m3
            / parameters.negative_maximum_concentration_mol_m3
        )
        positive_average_stoichiometry = (
            transport.positive.average_concentration_mol_m3
            / parameters.positive_maximum_concentration_mol_m3
        )
        negative_soc = (
            negative_average_stoichiometry - parameters.negative_stoichiometry_at_empty
        ) / (
            parameters.negative_stoichiometry_at_full
            - parameters.negative_stoichiometry_at_empty
        )
        positive_soc = (
            positive_average_stoichiometry - parameters.positive_stoichiometry_at_empty
        ) / (
            parameters.positive_stoichiometry_at_full
            - parameters.positive_stoichiometry_at_empty
        )
        soc_mismatch = negative_soc - positive_soc

        negative_ocp = parameters.negative_open_circuit_potential.evaluate(
            negative_surface_stoichiometry
        )
        positive_ocp = parameters.positive_open_circuit_potential.evaluate(
            positive_surface_stoichiometry
        )
        negative_window_ocp = parameters.negative_open_circuit_potential.evaluate(
            jnp.stack(
                (
                    parameters.negative_stoichiometry_at_empty,
                    parameters.negative_stoichiometry_at_full,
                )
            )
        )
        positive_window_ocp = parameters.positive_open_circuit_potential.evaluate(
            jnp.stack(
                (
                    parameters.positive_stoichiometry_at_full,
                    parameters.positive_stoichiometry_at_empty,
                )
            )
        )
        negative_interfacial_current_density = (
            _FARADAY_C_MOL * transport.negative_outward_flux_mol_m2_s
        )
        positive_interfacial_current_density = (
            _FARADAY_C_MOL * transport.positive_outward_flux_mol_m2_s
        )
        negative_overpotential, negative_exchange, negative_kinetic_valid = (
            _symmetric_overpotential(
                negative_interfacial_current_density,
                profile.negative_exchange_current_density_scale_a_m2,
                negative_surface_stoichiometry,
                parameters.temperature_k,
            )
        )
        positive_overpotential, positive_exchange, positive_kinetic_valid = (
            _symmetric_overpotential(
                positive_interfacial_current_density,
                profile.positive_exchange_current_density_scale_a_m2,
                positive_surface_stoichiometry,
                parameters.temperature_k,
            )
        )
        voltage = (
            positive_ocp.values
            + positive_overpotential
            - negative_ocp.values
            - negative_overpotential
        )
        total_lithium = (
            transport.negative.total_amount_mol + transport.positive.total_amount_mol
        )
        absorbed_power = voltage * current
        current_margin = parameters.maximum_absolute_current_a - jnp.abs(current)
        concentration_valid = (
            jnp.all(
                transport.negative.concentration_mol_m3
                <= parameters.negative_maximum_concentration_mol_m3,
                axis=-1,
            )
            & (
                transport.negative.surface_concentration_mol_m3
                <= parameters.negative_maximum_concentration_mol_m3
            )
            & jnp.all(
                transport.positive.concentration_mol_m3
                <= parameters.positive_maximum_concentration_mol_m3,
                axis=-1,
            )
            & (
                transport.positive.surface_concentration_mol_m3
                <= parameters.positive_maximum_concentration_mol_m3
            )
        )
        domain_valid = (
            transport.domain_valid
            & concentration_valid
            & negative_ocp.support
            & positive_ocp.support
            & jnp.isfinite(negative_ocp.values)
            & jnp.isfinite(positive_ocp.values)
            & jnp.all(
                negative_window_ocp.support & jnp.isfinite(negative_window_ocp.values)
            )
            & jnp.all(
                positive_window_ocp.support & jnp.isfinite(positive_window_ocp.values)
            )
            & negative_kinetic_valid
            & positive_kinetic_valid
            & jnp.isfinite(negative_soc)
            & jnp.isfinite(positive_soc)
            & jnp.isfinite(voltage)
        )
        values = jnp.stack(
            (
                voltage,
                jnp.zeros_like(voltage) + parameters.temperature_k,
                negative_surface_stoichiometry,
                positive_surface_stoichiometry,
                transport.negative.surface_concentration_mol_m3,
                transport.positive.surface_concentration_mol_m3,
                transport.negative.average_concentration_mol_m3,
                transport.positive.average_concentration_mol_m3,
                negative_soc,
                positive_soc,
                soc_mismatch,
                negative_ocp.values,
                positive_ocp.values,
                negative_overpotential,
                positive_overpotential,
                negative_exchange,
                positive_exchange,
                transport.negative_outward_flux_mol_m2_s,
                transport.positive_outward_flux_mol_m2_s,
                jnp.zeros_like(voltage) + profile.negative_specific_surface_area_m2_m3,
                jnp.zeros_like(voltage) + profile.positive_specific_surface_area_m2_m3,
                jnp.zeros_like(voltage) + profile.negative_theoretical_capacity_c,
                jnp.zeros_like(voltage) + profile.positive_theoretical_capacity_c,
                jnp.zeros_like(voltage) + profile.usable_capacity_c,
                jnp.zeros_like(voltage) + profile.negative_capacity_headroom_c,
                jnp.zeros_like(voltage) + profile.positive_capacity_headroom_c,
                total_lithium,
                absorbed_power,
                current_margin,
            ),
            axis=-1,
        )
        domain_valid = domain_valid & jnp.all(jnp.isfinite(values), axis=-1)
        return BatteryModelOutput(values, domain_valid)

    def ledger(
        self,
        prepared_model: PreparedPrescribedCurrentSpm,
        native_solution,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> SpmLedger:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        if not isinstance(runtime_inputs.parameters, SpmParameters):
            raise TypeError("SPM runtime parameters must be SpmParameters.")
        states = native_solution.states
        if not isinstance(states, SpmState):
            raise TypeError("SPM native solution states must be SpmState.")
        valid = jnp.asarray(native_solution.valid, dtype=bool)
        valid_count = jnp.sum(valid.astype(jnp.int32))
        final_index = jnp.maximum(valid_count - 1, 0)
        initial_negative = jnp.sum(states.negative_amount_mol[0])
        final_negative = jnp.sum(states.negative_amount_mol[final_index])
        initial_positive = jnp.sum(states.positive_amount_mol[0])
        final_positive = jnp.sum(states.positive_amount_mol[final_index])
        initial_total = initial_negative + initial_positive
        final_total = final_negative + final_positive
        lithium_residual = final_total - initial_total
        charge_residual = _FARADAY_C_MOL * lithium_residual

        start_time = native_solution.times[0]
        final_time = native_solution.times[final_index]
        policy_times = runtime_inputs.input_policy.times
        interval_left = jnp.maximum(policy_times[:-1], start_time)
        interval_right = jnp.minimum(policy_times[1:], final_time)
        interval_duration = jnp.maximum(interval_right - interval_left, 0.0)
        integrated_charge = jnp.sum(
            interval_duration * runtime_inputs.input_policy.values[:, 0]
        )
        negative_current_residual = (
            _FARADAY_C_MOL * (final_negative - initial_negative) - integrated_charge
        )
        positive_current_residual = (
            -_FARADAY_C_MOL * (final_positive - initial_positive) - integrated_charge
        )
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        initial_negative,
                        final_negative,
                        initial_positive,
                        final_positive,
                        lithium_residual,
                        charge_residual,
                        integrated_charge,
                        negative_current_residual,
                        positive_current_residual,
                    )
                )
            )
        ) & (valid_count > 0)
        amount_scale = jnp.maximum(jnp.abs(initial_total), jnp.abs(final_total))
        charge_scale = jnp.abs(integrated_charge)
        lithium_conserved = jnp.abs(lithium_residual) <= (
            prepared_model.plan.ledger_amount_absolute_tolerance_mol
            + prepared_model.plan.ledger_relative_tolerance * amount_scale
        )
        charge_conserved = jnp.abs(charge_residual) <= (
            prepared_model.plan.ledger_charge_absolute_tolerance_c
            + prepared_model.plan.ledger_relative_tolerance
            * _FARADAY_C_MOL
            * amount_scale
        )
        current_conserved = jnp.maximum(
            jnp.abs(negative_current_residual), jnp.abs(positive_current_residual)
        ) <= (
            prepared_model.plan.ledger_charge_absolute_tolerance_c
            + prepared_model.plan.ledger_relative_tolerance * charge_scale
        )
        successful = finite & lithium_conserved & charge_conserved & current_conserved
        return SpmLedger(
            initial_negative,
            final_negative,
            initial_positive,
            final_positive,
            initial_total,
            final_total,
            lithium_residual,
            charge_residual,
            integrated_charge,
            negative_current_residual,
            positive_current_residual,
            lithium_conserved,
            charge_conserved,
            current_conserved,
            finite,
            successful,
        )


__all__ = [
    "PrescribedCurrentSpmAdapter",
    "PrescribedCurrentSpmPlan",
    "PreparedPrescribedCurrentSpm",
    "SpmInitialCondition",
    "SpmLedger",
    "SpmParameters",
    "SpmState",
]
