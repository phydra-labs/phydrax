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
from ._spm import _profile, _stable_asinh_ratio
from ._spme_marquis2019 import (
    _evaluate as _evaluate_marquis,
    _OBSERVABLE_NAMES as _MARQUIS_OBSERVABLE_NAMES,
    _OBSERVABLE_UNITS as _MARQUIS_OBSERVABLE_UNITS,
    _region_mean,
    Marquis2019SpmeAdapter,
    Marquis2019SpmeInitialCondition,
    Marquis2019SpmeParameters,
    Marquis2019SpmePlan,
    Marquis2019SpmeState,
    PreparedMarquis2019Spme,
)


_FARADAY_C_MOL = 96485.33212
_GAS_CONSTANT_J_MOL_K = 8.31446261815324
_PROPERTY_TYPES = (ConstantPropertyLaw, TabulatedPropertyLaw)
_PropertyLaw: TypeAlias = ConstantPropertyLaw | TabulatedPropertyLaw


def _property_law(
    value: _PropertyLaw,
    name: str,
    /,
    *,
    value_unit: str,
) -> _PropertyLaw:
    if not isinstance(value, _PROPERTY_TYPES):
        raise TypeError(f"{name} must be ConstantPropertyLaw or TabulatedPropertyLaw.")
    if (
        value.coordinate != "electrolyte_concentration"
        or value.coordinate_unit != "mol/m3"
    ):
        raise ValueError(
            f"{name} must use coordinate 'electrolyte_concentration' in 'mol/m3'."
        )
    if value.value_unit != value_unit:
        raise ValueError(f"{name} must use value unit {value_unit!r}.")
    return value


def _electrolyte_potential_primitive(
    thermodynamic_factor: _PropertyLaw,
    transference_number: ConcentrationTemperaturePropertyLaw,
    concentration_mol_m3: Array,
    temperature_k: Array,
    /,
) -> Array:
    """Exact primitive of (1 - t_plus(c, T)) chi(c) d(log(c))."""
    chi_nodes = (
        thermodynamic_factor.support_bounds
        if isinstance(thermodynamic_factor, ConstantPropertyLaw)
        else thermodynamic_factor.nodes
    )
    nodes = jnp.sort(
        jnp.concatenate(
            (
                transference_number.concentration_nodes_mol_m3,
                chi_nodes,
            )
        )
    )
    left = nodes[:-1]
    right = nodes[1:]
    span = right - left
    safe_span = jnp.where(span > 0.0, span, 1.0)
    middle = 0.5 * (left + right)

    def integrand(coordinate):
        transfer = transference_number.evaluate(coordinate, temperature_k).values
        chi = thermodynamic_factor.evaluate(coordinate).values
        return (1.0 - transfer) * chi

    left_value = integrand(left)
    middle_value = integrand(middle)
    right_value = integrand(right)
    quadratic = 2.0 * (right_value + left_value - 2.0 * middle_value) / safe_span**2
    linear_local = (right_value - left_value - quadratic * safe_span**2) / safe_span
    linear = linear_local - 2.0 * quadratic * left
    logarithmic = quadratic * left**2 - linear_local * left + left_value
    concentration = jnp.asarray(concentration_mol_m3)
    upper = jnp.clip(concentration[..., None], left, right)
    segment = (
        0.5 * quadratic * (upper**2 - left**2)
        + linear * (upper - left)
        + logarithmic * jnp.log(upper / left)
    )
    return jnp.sum(
        jnp.where(span > 0.0, segment, 0.0),
        axis=-1,
    )


def _scalar(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape != () or jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must be one real scalar.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return array


class BrosaPlanellaSpmeSeiParameters(StrictModule):
    """Dynamic SI data for the isothermal SEI-only SPMe+SR specialization."""

    spme_parameters: Marquis2019SpmeParameters
    sei_reaction_rate_m_s: Array
    sei_solvent_concentration_mol_m3: Array
    sei_solvent_diffusivity_m2_s: Array
    sei_transfer_coefficient: Array
    sei_open_circuit_potential_v: Array
    sei_molar_mass_kg_mol: Array
    sei_density_kg_m3: Array
    sei_electron_stoichiometry: Array
    sei_conductivity_s_m: Array
    electrolyte_thermodynamic_factor: _PropertyLaw
    initial_sei_film_thickness_m: Array
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        spme_parameters: Marquis2019SpmeParameters,
        /,
        *,
        sei_reaction_rate_m_s: ArrayLike,
        sei_solvent_concentration_mol_m3: ArrayLike,
        sei_solvent_diffusivity_m2_s: ArrayLike,
        sei_transfer_coefficient: ArrayLike,
        sei_open_circuit_potential_v: ArrayLike,
        sei_molar_mass_kg_mol: ArrayLike,
        sei_density_kg_m3: ArrayLike,
        sei_electron_stoichiometry: ArrayLike,
        sei_conductivity_s_m: ArrayLike,
        electrolyte_thermodynamic_factor: _PropertyLaw | None = None,
        initial_sei_film_thickness_m: ArrayLike = 0.0,
    ):
        if not isinstance(spme_parameters, Marquis2019SpmeParameters):
            raise TypeError("spme_parameters must be Marquis2019SpmeParameters.")
        thermodynamic_factor = (
            ConstantPropertyLaw(
                1.0,
                (1.0e-6, 1.0e7),
                value_bounds=(0.0, jnp.inf),
                quantity="electrolyte-thermodynamic-factor",
                coordinate="electrolyte_concentration",
                value_unit="1",
                coordinate_unit="mol/m3",
                source_id="brosa-planella-spme-sei:ideal-electrolyte",
            )
            if electrolyte_thermodynamic_factor is None
            else _property_law(
                electrolyte_thermodynamic_factor,
                "electrolyte_thermodynamic_factor",
                value_unit="1",
            )
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
        rate = _scalar(sei_reaction_rate_m_s, "sei_reaction_rate_m_s")
        solvent = _scalar(
            sei_solvent_concentration_mol_m3,
            "sei_solvent_concentration_mol_m3",
        )
        diffusivity = _scalar(
            sei_solvent_diffusivity_m2_s, "sei_solvent_diffusivity_m2_s"
        )
        transfer = _scalar(sei_transfer_coefficient, "sei_transfer_coefficient")
        open_circuit = _scalar(
            sei_open_circuit_potential_v, "sei_open_circuit_potential_v"
        )
        molar_mass = _scalar(sei_molar_mass_kg_mol, "sei_molar_mass_kg_mol")
        density = _scalar(sei_density_kg_m3, "sei_density_kg_m3")
        electrons = _scalar(sei_electron_stoichiometry, "sei_electron_stoichiometry")
        conductivity = _scalar(sei_conductivity_s_m, "sei_conductivity_s_m")
        initial_film = _scalar(
            initial_sei_film_thickness_m, "initial_sei_film_thickness_m"
        )
        positive = jnp.stack(
            (solvent, diffusivity, molar_mass, density, electrons, conductivity)
        )
        valid = (
            jnp.isfinite(rate)
            & (rate >= 0.0)
            & jnp.all(jnp.isfinite(positive) & (positive > 0.0))
            & jnp.isfinite(transfer)
            & (transfer > 0.0)
            & (transfer <= 1.0)
            & jnp.isfinite(open_circuit)
            & jnp.isfinite(initial_film)
            & (initial_film >= 0.0)
        )
        rate = eqx.error_if(
            rate,
            ~valid,
            "SEI kinetics, solvent transport, product, film, and stoichiometric "
            "data must be finite and physical.",
        )
        self.spme_parameters = spme_parameters
        self.sei_reaction_rate_m_s = rate
        self.sei_solvent_concentration_mol_m3 = solvent
        self.sei_solvent_diffusivity_m2_s = diffusivity
        self.sei_transfer_coefficient = transfer
        self.sei_open_circuit_potential_v = open_circuit
        self.sei_molar_mass_kg_mol = molar_mass
        self.sei_density_kg_m3 = density
        self.sei_electron_stoichiometry = electrons
        self.sei_conductivity_s_m = conductivity
        self.electrolyte_thermodynamic_factor = thermodynamic_factor
        self.initial_sei_film_thickness_m = initial_film
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "battery-spme-sei-brosa-planella-widanage-parameters",
                "spme_parameter_id": spme_parameters.parameter_id,
                "electrolyte_thermodynamic_factor_law_id": (thermodynamic_factor.law_id),
            }
        )


class BrosaPlanellaSpmeSeiInitialCondition(StrictModule):
    """Uniform particle stoichiometries for a fresh, uniform-porosity electrode."""

    negative_stoichiometry: Array
    positive_stoichiometry: Array

    def __init__(
        self,
        negative_stoichiometry: ArrayLike,
        positive_stoichiometry: ArrayLike,
        /,
    ):
        initial = Marquis2019SpmeInitialCondition(
            negative_stoichiometry, positive_stoichiometry
        )
        self.negative_stoichiometry = initial.negative_stoichiometry
        self.positive_stoichiometry = initial.positive_stoichiometry


class BrosaPlanellaSpmeSeiState(StrictModule):
    """Extensive lithium states and the local negative-electrode porosity field."""

    negative_amount_mol: Array
    positive_amount_mol: Array
    electrolyte_amount_mol: Array
    negative_porosity: Array

    def __init__(
        self,
        negative_amount_mol: ArrayLike,
        positive_amount_mol: ArrayLike,
        electrolyte_amount_mol: ArrayLike,
        negative_porosity: ArrayLike,
        /,
    ):
        base = Marquis2019SpmeState(
            negative_amount_mol,
            positive_amount_mol,
            electrolyte_amount_mol,
        )
        porosity = jnp.asarray(negative_porosity)
        if porosity.ndim < 1:
            raise ValueError("Negative-electrode porosity must contain a cell axis.")
        if jnp.issubdtype(porosity.dtype, jnp.complexfloating):
            raise TypeError("Negative-electrode porosity must be real-valued.")
        leading_shape = base.negative_amount_mol.shape[:-1]
        if porosity.shape[:-1] != leading_shape:
            raise ValueError("Porosity and particle state leading axes must match.")
        dtype = jnp.result_type(
            base.negative_amount_mol,
            base.positive_amount_mol,
            base.electrolyte_amount_mol,
            porosity,
            float,
        )
        self.negative_amount_mol = base.negative_amount_mol.astype(dtype)
        self.positive_amount_mol = base.positive_amount_mol.astype(dtype)
        self.electrolyte_amount_mol = base.electrolyte_amount_mol.astype(dtype)
        self.negative_porosity = porosity.astype(dtype)


class BrosaPlanellaSpmeSeiLedger(StrictModule):
    """Independent endpoint ledgers for lithium, charge, SEI product, and film."""

    initial_negative_solid_lithium_mol: Array
    final_negative_solid_lithium_mol: Array
    initial_positive_solid_lithium_mol: Array
    final_positive_solid_lithium_mol: Array
    initial_electrolyte_lithium_mol: Array
    final_electrolyte_lithium_mol: Array
    initial_sei_lithium_mol: Array
    final_sei_lithium_mol: Array
    initial_sei_product_mol: Array
    final_sei_product_mol: Array
    initial_total_lithium_mol: Array
    final_total_lithium_mol: Array
    initial_film_mass_kg: Array
    final_film_mass_kg: Array
    integrated_terminal_charge_c: Array
    integrated_side_charge_c: Array
    negative_intercalation_charge_c: Array
    total_lithium_conservation_residual_mol: Array
    electrolyte_lithium_conservation_residual_mol: Array
    negative_charge_conservation_residual_c: Array
    positive_charge_conservation_residual_c: Array
    side_charge_product_residual_c: Array
    film_mass_product_residual_kg: Array
    maximum_film_thickness_porosity_residual_m: Array
    maximum_collector_flux_residual_mol_m2_s: Array
    maximum_interface_concentration_jump_mol_m3: Array
    maximum_equation_mapping_residual_mol_s: Array
    maximum_current_split_residual_a_m2: Array
    maximum_rest_current_cancellation_residual_a: Array
    maximum_weak_side_reaction_ratio: Array
    maximum_overpotential_v: Array
    lithium_conserved: Array
    electrolyte_lithium_conserved: Array
    charge_conserved: Array
    product_conserved: Array
    film_conserved: Array
    porosity_conserved: Array
    interfaces_conserved: Array
    current_split_conserved: Array
    rest_current_cancellation_conserved: Array
    weak_side_reaction_condition_satisfied: Array
    small_overpotential_condition_satisfied: Array
    asymptotic_conditions_satisfied: Array
    domain_valid: Array
    finite: Array
    successful: Array


class BrosaPlanellaSpmeSeiPlan(StrictModule, NonTrainableState):
    """Static radial/through-cell topology and SPMe+SR evidence thresholds."""

    marquis_plan: Marquis2019SpmePlan
    weak_side_reaction_threshold: float = eqx.field(static=True)
    small_overpotential_threshold_v: float = eqx.field(static=True)
    ledger_film_mass_absolute_tolerance_kg: float = eqx.field(static=True)
    ledger_porosity_absolute_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        negative_shell_count: int,
        positive_shell_count: int | None = None,
        /,
        *,
        negative_electrolyte_cell_count: int = 8,
        separator_electrolyte_cell_count: int = 6,
        positive_electrolyte_cell_count: int = 8,
        negative_particle_reference_faces: ArrayLike | None = None,
        positive_particle_reference_faces: ArrayLike | None = None,
        negative_electrolyte_reference_faces: ArrayLike | None = None,
        separator_electrolyte_reference_faces: ArrayLike | None = None,
        positive_electrolyte_reference_faces: ArrayLike | None = None,
        asymptotic_small_parameter_threshold: float = 0.1,
        weak_side_reaction_threshold: float = 0.1,
        small_overpotential_threshold_v: float = 0.1,
        ledger_amount_absolute_tolerance_mol: float = 1.0e-10,
        ledger_charge_absolute_tolerance_c: float = 1.0e-5,
        ledger_molar_flux_absolute_tolerance_mol_m2_s: float = 1.0e-12,
        ledger_concentration_absolute_tolerance_mol_m3: float = 1.0e-9,
        ledger_rate_absolute_tolerance_mol_s: float = 1.0e-12,
        ledger_current_density_absolute_tolerance_a_m2: float = 1.0e-8,
        ledger_film_mass_absolute_tolerance_kg: float = 1.0e-12,
        ledger_porosity_absolute_tolerance: float = 1.0e-10,
        ledger_relative_tolerance: float = 1.0e-6,
    ):
        weak = float(weak_side_reaction_threshold)
        overpotential = float(small_overpotential_threshold_v)
        film_mass_atol = float(ledger_film_mass_absolute_tolerance_kg)
        porosity_atol = float(ledger_porosity_absolute_tolerance)
        values = np.asarray((weak, overpotential, film_mass_atol, porosity_atol))
        if (
            np.any(~np.isfinite(values))
            or weak <= 0.0
            or weak >= 1.0
            or overpotential <= 0.0
            or np.any(values[2:] < 0.0)
        ):
            raise ValueError(
                "SEI asymptotic thresholds must be finite and positive, the weak "
                "threshold must be below one, and ledger tolerances nonnegative."
            )
        marquis = Marquis2019SpmePlan(
            negative_shell_count,
            positive_shell_count,
            negative_electrolyte_cell_count=negative_electrolyte_cell_count,
            separator_electrolyte_cell_count=separator_electrolyte_cell_count,
            positive_electrolyte_cell_count=positive_electrolyte_cell_count,
            negative_particle_reference_faces=negative_particle_reference_faces,
            positive_particle_reference_faces=positive_particle_reference_faces,
            negative_electrolyte_reference_faces=negative_electrolyte_reference_faces,
            separator_electrolyte_reference_faces=separator_electrolyte_reference_faces,
            positive_electrolyte_reference_faces=positive_electrolyte_reference_faces,
            asymptotic_small_parameter_threshold=asymptotic_small_parameter_threshold,
            ledger_amount_absolute_tolerance_mol=ledger_amount_absolute_tolerance_mol,
            ledger_charge_absolute_tolerance_c=ledger_charge_absolute_tolerance_c,
            ledger_molar_flux_absolute_tolerance_mol_m2_s=(
                ledger_molar_flux_absolute_tolerance_mol_m2_s
            ),
            ledger_concentration_absolute_tolerance_mol_m3=(
                ledger_concentration_absolute_tolerance_mol_m3
            ),
            ledger_rate_absolute_tolerance_mol_s=ledger_rate_absolute_tolerance_mol_s,
            ledger_current_density_absolute_tolerance_a_m2=(
                ledger_current_density_absolute_tolerance_a_m2
            ),
            ledger_relative_tolerance=ledger_relative_tolerance,
        )
        self.marquis_plan = marquis
        self.weak_side_reaction_threshold = weak
        self.small_overpotential_threshold_v = overpotential
        self.ledger_film_mass_absolute_tolerance_kg = film_mass_atol
        self.ledger_porosity_absolute_tolerance = porosity_atol
        self.plan_id = canonical_fingerprint(
            {
                "kind": "battery-spme-sei-brosa-planella-widanage-plan",
                "marquis_plan_id": marquis.plan_id,
                "weak_side_reaction_threshold": weak,
                "small_overpotential_threshold_v": overpotential,
                "ledger_film_mass_atol_kg": film_mass_atol,
                "ledger_porosity_atol": porosity_atol,
            }
        )

    @property
    def spm_plan(self):
        return self.marquis_plan.spm_plan

    @property
    def negative_region(self):
        return self.marquis_plan.negative_region

    @property
    def separator_region(self):
        return self.marquis_plan.separator_region

    @property
    def positive_region(self):
        return self.marquis_plan.positive_region

    def prepare(self, /) -> "PreparedBrosaPlanellaSpmeSei":
        return PreparedBrosaPlanellaSpmeSei(self)


class PreparedBrosaPlanellaSpmeSei(StrictModule, NonTrainableState):
    """Prepared private candidate; physical SEI data remain runtime leaves."""

    plan: BrosaPlanellaSpmeSeiPlan
    marquis: PreparedMarquis2019Spme
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: BrosaPlanellaSpmeSeiPlan, /):
        if not isinstance(plan, BrosaPlanellaSpmeSeiPlan):
            raise TypeError("plan must be BrosaPlanellaSpmeSeiPlan.")
        marquis = plan.marquis_plan.prepare()
        self.plan = plan
        self.marquis = marquis
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-battery-spme-sei-brosa-planella-widanage",
                "plan_id": plan.plan_id,
                "marquis_prepared_id": marquis.prepared_id,
            }
        )

    @property
    def spm(self):
        return self.marquis.spm

    @property
    def through_cell(self):
        return self.marquis.through_cell

    def evaluate(
        self,
        state: BrosaPlanellaSpmeSeiState,
        parameters: BrosaPlanellaSpmeSeiParameters,
        terminal_current_a: ArrayLike,
        /,
    ) -> "_BrosaPlanellaSpmeSeiEvaluation":
        return _evaluate(self, state, parameters, jnp.asarray(terminal_current_a))


class _LocalPorosityElectrolyteEvaluation(StrictModule):
    concentration_mol_m3: Array
    face_concentration_mol_m3: Array
    diffusive_molar_flux_mol_m2_s: Array
    total_molar_flux_mol_m2_s: Array
    electrolyte_current_density_a_m2: Array
    solid_current_density_a_m2: Array
    source_mol_m3_s: Array
    amount_rate_mol_s: Array
    total_amount_mol: Array
    porosity: Array
    electrolyte_diffusivity_m2_s: Array
    transference_number: Array
    effective_diffusivity_m2_s: Array
    storage_volume_m3: Array
    explicit_dt_limit_s: Array
    collector_flux_residual_mol_m2_s: Array
    negative_separator_concentration_jump_mol_m3: Array
    separator_positive_concentration_jump_mol_m3: Array
    equation_mapping_residual_mol_s: Array
    conservation_residual_mol_s: Array
    current_split_residual_a_m2: Array
    domain_valid: Array


class _BrosaPlanellaSpmeSeiEvaluation(StrictModule):
    voltage_v: Array
    particle_ocp_v: Array
    reaction_overpotential_v: Array
    mean_concentration_overpotential_v: Array
    electrolyte_ohmic_loss_v: Array
    solid_ohmic_loss_v: Array
    film_voltage_correction_v: Array
    negative_surface_stoichiometry: Array
    positive_surface_stoichiometry: Array
    negative_surface_concentration_mol_m3: Array
    positive_surface_concentration_mol_m3: Array
    negative_average_concentration_mol_m3: Array
    positive_average_concentration_mol_m3: Array
    negative_electrolyte_mean_mol_m3: Array
    separator_electrolyte_mean_mol_m3: Array
    positive_electrolyte_mean_mol_m3: Array
    negative_exchange_current_density_a_m2: Array
    positive_exchange_current_density_a_m2: Array
    sei_current_density_a_m3: Array
    electrode_averaged_sei_current_density_a_m3: Array
    side_current_a: Array
    negative_intercalation_current_a: Array
    terminal_current_balance_residual_a: Array
    sei_overpotential_v: Array
    film_thickness_m: Array
    film_resistance_ohm_m2: Array
    negative_porosity: Array
    sei_lithium_amount_mol: Array
    sei_product_amount_mol: Array
    film_mass_kg: Array
    total_solid_lithium_mol: Array
    total_electrolyte_lithium_mol: Array
    total_lithium_mol: Array
    weak_side_reaction_ratio: Array
    maximum_overpotential_v: Array
    weak_side_reaction_condition_satisfied: Array
    small_overpotential_condition_satisfied: Array
    asymptotic_conditions_satisfied: Array
    zero_sei_reduction: Array
    domain_valid: Array
    negative_particle_transport: object
    positive_particle_transport: object
    electrolyte_transport: _LocalPorosityElectrolyteEvaluation
    marquis_evaluation: object


def _check_prepared(
    plan: BrosaPlanellaSpmeSeiPlan,
    prepared_model: PreparedBrosaPlanellaSpmeSei,
    /,
) -> None:
    if not isinstance(prepared_model, PreparedBrosaPlanellaSpmeSei):
        raise TypeError("prepared_model must be PreparedBrosaPlanellaSpmeSei.")
    if prepared_model.plan.plan_id != plan.plan_id:
        raise ValueError("Prepared SPMe+SEI topology does not belong to this adapter.")


def _weighted_mean(values: Array, weights: Array, /) -> Array:
    return jnp.sum(values * weights, axis=-1) / jnp.sum(weights)


def _local_electrolyte_transport(
    prepared: PreparedBrosaPlanellaSpmeSei,
    state: BrosaPlanellaSpmeSeiState,
    parameters: BrosaPlanellaSpmeSeiParameters,
    base_transport,
    zero_sei_reduction: Array,
    /,
) -> _LocalPorosityElectrolyteEvaluation:
    """Apply only the local-porosity specialization to shared through-cell topology."""
    mesh = prepared.through_cell
    marquis = parameters.spme_parameters
    spm = marquis.spm_parameters
    leading_shape = state.electrolyte_amount_mol.shape[:-1]
    negative_count = mesh.negative.cell_count
    separator_count = mesh.separator.cell_count
    positive_count = mesh.positive.cell_count
    separator_porosity = jnp.broadcast_to(
        marquis.separator_electrolyte_porosity,
        leading_shape + (separator_count,),
    )
    positive_porosity = jnp.broadcast_to(
        marquis.positive_electrolyte_porosity,
        leading_shape + (positive_count,),
    )
    porosity = jnp.concatenate(
        (state.negative_porosity, separator_porosity, positive_porosity), axis=-1
    )
    porosity_valid = jnp.all(
        jnp.isfinite(porosity) & (porosity > 0.0) & (porosity <= 1.0), axis=-1
    )
    safe_porosity = jnp.where(
        jnp.isfinite(porosity) & (porosity > 0.0) & (porosity <= 1.0),
        porosity,
        1.0,
    )
    metrics = mesh.metrics(
        negative_thickness_m=spm.negative_electrode_thickness_m,
        separator_thickness_m=marquis.separator_thickness_m,
        positive_thickness_m=spm.positive_electrode_thickness_m,
        negative_porosity=marquis.negative_electrolyte_porosity,
        separator_porosity=marquis.separator_electrolyte_porosity,
        positive_porosity=marquis.positive_electrolyte_porosity,
        bruggeman_coefficient=marquis.bruggeman_coefficient,
        electrolyte_diffusivity_m2_s=marquis.electrolyte_diffusivity.evaluate(
            marquis.typical_electrolyte_concentration_mol_m3,
            spm.temperature_k,
        ).values,
        electrode_area_m2=spm.electrode_area_m2,
    )
    widths = metrics.cell_widths_m
    storage_volume = spm.electrode_area_m2 * safe_porosity * widths
    amounts = jnp.where(
        jnp.isfinite(state.electrolyte_amount_mol),
        state.electrolyte_amount_mol,
        0.0,
    )
    concentration = amounts / storage_volume
    temperature = jnp.zeros_like(concentration) + spm.temperature_k
    diffusivity_result = marquis.electrolyte_diffusivity.evaluate(
        concentration, temperature
    )
    transference_result = marquis.transference_number.evaluate(concentration, temperature)
    diffusivity_valid = jnp.all(diffusivity_result.support, axis=-1) & jnp.all(
        jnp.isfinite(diffusivity_result.values) & (diffusivity_result.values > 0.0),
        axis=-1,
    )
    transference_valid = jnp.all(transference_result.support, axis=-1) & jnp.all(
        jnp.isfinite(transference_result.values)
        & (transference_result.values >= 0.0)
        & (transference_result.values <= 1.0),
        axis=-1,
    )
    safe_diffusivity = jnp.where(
        diffusivity_result.support
        & jnp.isfinite(diffusivity_result.values)
        & (diffusivity_result.values > 0.0),
        diffusivity_result.values,
        1.0,
    )
    safe_transference = jnp.where(
        transference_result.support
        & jnp.isfinite(transference_result.values)
        & (transference_result.values >= 0.0)
        & (transference_result.values <= 1.0),
        transference_result.values,
        0.0,
    )
    effective_diffusivity = (
        safe_porosity**marquis.bruggeman_coefficient * safe_diffusivity
    )

    resistance = (
        0.5 * widths[:-1] / effective_diffusivity[..., :-1]
        + 0.5 * widths[1:] / effective_diffusivity[..., 1:]
    )
    conductance = 1.0 / resistance
    interior_diffusive_flux = -conductance * (
        concentration[..., 1:] - concentration[..., :-1]
    )
    zeros = jnp.zeros(leading_shape + (1,), dtype=concentration.dtype)
    diffusive_flux = jnp.concatenate((zeros, interior_diffusive_flux, zeros), axis=-1)
    left_trace = concentration[..., :-1] - (
        interior_diffusive_flux * 0.5 * widths[:-1] / effective_diffusivity[..., :-1]
    )
    right_trace = concentration[..., 1:] + (
        interior_diffusive_flux * 0.5 * widths[1:] / effective_diffusivity[..., 1:]
    )
    face_concentration = jnp.concatenate(
        (
            concentration[..., :1],
            0.5 * (left_trace + right_trace),
            concentration[..., -1:],
        ),
        axis=-1,
    )
    electrolyte_current = base_transport.electrolyte_current_density_a_m2
    solid_current = base_transport.solid_current_density_a_m2
    transference_faces = jnp.concatenate(
        (
            safe_transference[..., :1],
            0.5 * (safe_transference[..., :-1] + safe_transference[..., 1:]),
            safe_transference[..., -1:],
        ),
        axis=-1,
    )
    migration_flux = transference_faces * electrolyte_current / _FARADAY_C_MOL
    total_flux = diffusive_flux + migration_flux
    paper_current_density = electrolyte_current[..., mesh.negative_separator_face_index]
    full_source = jnp.where(
        mesh.negative_mask,
        paper_current_density[..., None]
        / (_FARADAY_C_MOL * spm.negative_electrode_thickness_m),
        jnp.where(
            mesh.positive_mask,
            -paper_current_density[..., None]
            / (_FARADAY_C_MOL * spm.positive_electrode_thickness_m),
            0.0,
        ),
    )
    source = full_source + (migration_flux[..., :-1] - migration_flux[..., 1:]) / widths
    amount_rate = spm.electrode_area_m2 * (
        diffusive_flux[..., :-1] - diffusive_flux[..., 1:] + source * widths
    )
    full_equation_rate = spm.electrode_area_m2 * (
        total_flux[..., :-1] - total_flux[..., 1:] + full_source * widths
    )
    equation_mapping_residual = jnp.max(
        jnp.abs(amount_rate - full_equation_rate), axis=-1
    )
    conservation_residual = jnp.sum(amount_rate, axis=-1) - spm.electrode_area_m2 * (
        total_flux[..., 0] - total_flux[..., -1] + jnp.sum(full_source * widths, axis=-1)
    )
    face_zeros = jnp.zeros(leading_shape + (1,), dtype=conductance.dtype)
    face_conductance = jnp.concatenate((face_zeros, conductance, face_zeros), axis=-1)
    loss_rate = (face_conductance[..., :-1] + face_conductance[..., 1:]) / (
        safe_porosity * widths
    )
    explicit_dt_limit = jnp.min(
        jnp.where(loss_rate > 0.0, 1.0 / loss_rate, jnp.inf), axis=-1
    )
    ns_face = mesh.negative_separator_face_index
    sp_face = mesh.separator_positive_face_index
    negative_separator_jump = left_trace[..., ns_face - 1] - right_trace[..., ns_face - 1]
    separator_positive_jump = left_trace[..., sp_face - 1] - right_trace[..., sp_face - 1]
    collector_flux_residual = jnp.maximum(
        jnp.abs(total_flux[..., 0]), jnp.abs(total_flux[..., -1])
    )
    current_split_residual = jnp.max(
        jnp.abs(electrolyte_current + solid_current - paper_current_density[..., None]),
        axis=-1,
    )
    state_valid = jnp.all(
        jnp.isfinite(state.electrolyte_amount_mol)
        & (state.electrolyte_amount_mol >= 0.0),
        axis=-1,
    )
    finite = (
        jnp.all(jnp.isfinite(concentration), axis=-1)
        & jnp.all(jnp.isfinite(face_concentration), axis=-1)
        & jnp.all(jnp.isfinite(diffusive_flux), axis=-1)
        & jnp.all(jnp.isfinite(total_flux), axis=-1)
        & jnp.all(jnp.isfinite(amount_rate), axis=-1)
        & jnp.isfinite(explicit_dt_limit)
        & jnp.isfinite(equation_mapping_residual)
        & jnp.isfinite(conservation_residual)
    )
    positive = jnp.all(concentration > 0.0, axis=-1) & jnp.all(
        face_concentration > 0.0, axis=-1
    )
    domain_valid = (
        metrics.domain_valid
        & porosity_valid
        & diffusivity_valid
        & transference_valid
        & state_valid
        & finite
        & positive
    )

    def choose(base_value, local_value):
        condition = zero_sei_reduction
        while condition.ndim < local_value.ndim:
            condition = condition[..., None]
        return jnp.where(condition, base_value, local_value)

    return _LocalPorosityElectrolyteEvaluation(
        choose(base_transport.concentration_mol_m3, concentration),
        choose(base_transport.face_concentration_mol_m3, face_concentration),
        choose(base_transport.diffusive_molar_flux_mol_m2_s, diffusive_flux),
        choose(base_transport.total_molar_flux_mol_m2_s, total_flux),
        electrolyte_current,
        solid_current,
        source,
        choose(base_transport.amount_rate_mol_s, amount_rate),
        jnp.sum(amounts, axis=-1),
        porosity,
        diffusivity_result.values,
        transference_result.values,
        effective_diffusivity,
        storage_volume,
        choose(base_transport.explicit_dt_limit_s, explicit_dt_limit),
        choose(base_transport.collector_flux_residual_mol_m2_s, collector_flux_residual),
        choose(
            base_transport.negative_separator_concentration_jump_mol_m3,
            negative_separator_jump,
        ),
        choose(
            base_transport.separator_positive_concentration_jump_mol_m3,
            separator_positive_jump,
        ),
        choose(base_transport.equation_mapping_residual_mol_s, equation_mapping_residual),
        choose(base_transport.conservation_residual_mol_s, conservation_residual),
        base_transport.current_split_residual_a_m2,
        choose(base_transport.domain_valid, domain_valid),
    )


def _sei_inventory(
    prepared: PreparedBrosaPlanellaSpmeSei,
    negative_porosity: Array,
    parameters: BrosaPlanellaSpmeSeiParameters,
    /,
) -> tuple[Array, Array, Array, Array]:
    marquis = parameters.spme_parameters
    spm = marquis.spm_parameters
    profile = _profile(spm)
    widths = (
        prepared.through_cell.negative.reference_faces[1:]
        - prepared.through_cell.negative.reference_faces[:-1]
    ) * spm.negative_electrode_thickness_m
    film_thickness = (
        parameters.initial_sei_film_thickness_m
        + (marquis.negative_electrolyte_porosity - negative_porosity)
        / profile.negative_specific_surface_area_m2_m3
    )
    film_volume = (
        spm.electrode_area_m2
        * profile.negative_specific_surface_area_m2_m3
        * jnp.sum(film_thickness * widths, axis=-1)
    )
    film_mass = parameters.sei_density_kg_m3 * film_volume
    product = film_mass / parameters.sei_molar_mass_kg_mol
    lithium = parameters.sei_electron_stoichiometry * product
    return lithium, product, film_mass, film_thickness


def _evaluate(
    prepared: PreparedBrosaPlanellaSpmeSei,
    state: BrosaPlanellaSpmeSeiState,
    parameters: BrosaPlanellaSpmeSeiParameters,
    terminal_current_a: Array,
    /,
) -> _BrosaPlanellaSpmeSeiEvaluation:
    if not isinstance(state, BrosaPlanellaSpmeSeiState):
        raise TypeError("state must be BrosaPlanellaSpmeSeiState.")
    if not isinstance(parameters, BrosaPlanellaSpmeSeiParameters):
        raise TypeError("parameters must be BrosaPlanellaSpmeSeiParameters.")
    if state.negative_porosity.shape[-1] != prepared.through_cell.negative.cell_count:
        raise ValueError("Negative porosity does not match the prepared electrode mesh.")
    marquis = parameters.spme_parameters
    spm = marquis.spm_parameters
    base_state = Marquis2019SpmeState(
        state.negative_amount_mol,
        state.positive_amount_mol,
        state.electrolyte_amount_mol,
    )
    base = _evaluate_marquis(
        prepared.marquis,
        base_state,
        marquis,
        terminal_current_a,
    )
    profile = _profile(spm)
    leading_shape = state.negative_amount_mol.shape[:-1]
    current = jnp.asarray(terminal_current_a)
    if current.shape == ():
        current = jnp.broadcast_to(current, leading_shape)
    elif current.shape != leading_shape:
        raise ValueError("terminal_current_a must be scalar or match state leading axes.")
    zero_sei_reduction = (
        (parameters.sei_reaction_rate_m_s == 0.0)
        & (parameters.initial_sei_film_thickness_m == 0.0)
        & jnp.all(
            state.negative_porosity == marquis.negative_electrolyte_porosity,
            axis=-1,
        )
    )
    electrolyte = _local_electrolyte_transport(
        prepared,
        state,
        parameters,
        base.electrolyte_transport,
        zero_sei_reduction,
    )
    negative_count = prepared.through_cell.negative.cell_count
    negative_weights = prepared.through_cell.reference_cell_widths[:negative_count]
    negative_widths_m = negative_weights * spm.negative_electrode_thickness_m
    negative_concentration = electrolyte.concentration_mol_m3[..., :negative_count]
    film_thickness = (
        parameters.initial_sei_film_thickness_m
        + (marquis.negative_electrolyte_porosity - state.negative_porosity)
        / profile.negative_specific_surface_area_m2_m3
    )
    film_resistance = film_thickness / parameters.sei_conductivity_s_m
    passive_current_density = current / spm.electrode_area_m2
    thermal_voltage_twice = (
        2.0 * _GAS_CONSTANT_J_MOL_K * spm.temperature_k / _FARADAY_C_MOL
    )

    preliminary_stoichiometry = base.negative_surface_stoichiometry
    occupancy = preliminary_stoichiometry * (1.0 - preliminary_stoichiometry)
    local_exchange = (
        4.0
        * profile.negative_exchange_current_density_scale_a_m2
        * jnp.sqrt(jnp.maximum(occupancy[..., None], 0.0))
        * jnp.sqrt(
            jnp.maximum(
                negative_concentration / marquis.typical_electrolyte_concentration_mol_m3,
                0.0,
            )
        )
    )
    local_intercalation_overpotential = -thermal_voltage_twice * _stable_asinh_ratio(
        passive_current_density[..., None],
        profile.negative_specific_surface_area_m2_m3
        * spm.negative_electrode_thickness_m
        * local_exchange,
    )
    reference_centers = prepared.through_cell.reference_cell_centers[:negative_count]
    physical_centers = reference_centers * spm.negative_electrode_thickness_m
    safe_negative_porosity = jnp.where(
        jnp.isfinite(state.negative_porosity)
        & (state.negative_porosity > 0.0)
        & (state.negative_porosity <= marquis.negative_electrolyte_porosity),
        state.negative_porosity,
        marquis.negative_electrolyte_porosity,
    )
    concentration = electrolyte.concentration_mol_m3
    conductivity_result = marquis.electrolyte_conductivity.evaluate(
        concentration,
        jnp.zeros_like(concentration) + spm.temperature_k,
    )
    thermodynamic_result = parameters.electrolyte_thermodynamic_factor.evaluate(
        concentration
    )
    conductivity_valid = jnp.all(conductivity_result.support, axis=-1) & jnp.all(
        jnp.isfinite(conductivity_result.values) & (conductivity_result.values > 0.0),
        axis=-1,
    )
    thermodynamic_valid = jnp.all(thermodynamic_result.support, axis=-1) & jnp.all(
        jnp.isfinite(thermodynamic_result.values) & (thermodynamic_result.values > 0.0),
        axis=-1,
    )
    safe_conductivity = jnp.where(
        conductivity_result.support
        & jnp.isfinite(conductivity_result.values)
        & (conductivity_result.values > 0.0),
        conductivity_result.values,
        1.0,
    )
    safe_porosity = jnp.where(
        jnp.isfinite(electrolyte.porosity)
        & (electrolyte.porosity > 0.0)
        & (electrolyte.porosity <= 1.0),
        electrolyte.porosity,
        1.0,
    )
    region_lengths = jnp.stack(
        (
            spm.negative_electrode_thickness_m,
            marquis.separator_thickness_m,
            spm.positive_electrode_thickness_m,
        )
    )
    cell_widths_m = (
        prepared.through_cell.reference_cell_widths
        * region_lengths[prepared.through_cell.cell_region_indices]
    )
    electrolyte_current_centers = 0.5 * (
        electrolyte.electrolyte_current_density_a_m2[..., :-1]
        + electrolyte.electrolyte_current_density_a_m2[..., 1:]
    )
    electrolyte_ohmic_increment = (
        electrolyte_current_centers
        * cell_widths_m
        / (safe_conductivity * safe_porosity**marquis.bruggeman_coefficient)
    )
    electrolyte_ohmic_integral = (
        jnp.cumsum(electrolyte_ohmic_increment, axis=-1)
        - 0.5 * electrolyte_ohmic_increment
    )
    safe_concentration = jnp.where(
        jnp.isfinite(concentration) & (concentration > 0.0),
        concentration,
        marquis.typical_electrolyte_concentration_mol_m3,
    )
    electrolyte_potential_primitive = _electrolyte_potential_primitive(
        parameters.electrolyte_thermodynamic_factor,
        marquis.transference_number,
        safe_concentration,
        spm.temperature_k,
    )
    electrolyte_concentration_potential = thermal_voltage_twice * (
        electrolyte_potential_primitive - electrolyte_potential_primitive[..., :1]
    )
    electrolyte_potential = (
        -electrolyte_ohmic_integral + electrolyte_concentration_potential
    )
    negative_electrolyte_potential = electrolyte_potential[..., :negative_count]
    mean_electrolyte_potential = _weighted_mean(
        negative_electrolyte_potential, negative_weights
    )
    mean_intercalation_overpotential = _weighted_mean(
        local_intercalation_overpotential, negative_weights
    )
    mean_film_thickness = _weighted_mean(film_thickness, negative_weights)
    solid_potential_variation = passive_current_density[..., None] * (
        (2.0 * spm.negative_electrode_thickness_m - physical_centers)
        * physical_centers
        / (
            2.0
            * spm.negative_electrode_thickness_m
            * marquis.negative_solid_conductivity_s_m
        )
        - spm.negative_electrode_thickness_m
        / (3.0 * marquis.negative_solid_conductivity_s_m)
    )
    preliminary_ocp = spm.negative_open_circuit_potential.evaluate(
        preliminary_stoichiometry
    )
    negative_solid_potential = (
        preliminary_ocp.values[..., None]
        + mean_electrolyte_potential[..., None]
        + mean_intercalation_overpotential[..., None]
        + solid_potential_variation
        - passive_current_density[..., None]
        * mean_film_thickness[..., None]
        / (
            spm.negative_electrode_thickness_m
            * profile.negative_specific_surface_area_m2_m3
            * parameters.sei_conductivity_s_m
        )
    )
    sei_overpotential = (
        negative_solid_potential
        - negative_electrolyte_potential
        - parameters.sei_open_circuit_potential_v
        + passive_current_density[..., None]
        * film_thickness
        / (
            spm.negative_electrode_thickness_m
            * profile.negative_specific_surface_area_m2_m3
            * parameters.sei_conductivity_s_m
        )
    )
    solvent_denominator = (
        1.0
        - parameters.sei_reaction_rate_m_s
        * film_thickness
        / parameters.sei_solvent_diffusivity_m2_s
    )
    solvent_domain = jnp.all(
        jnp.isfinite(solvent_denominator) & (solvent_denominator > 0.0), axis=-1
    )
    safe_solvent_denominator = jnp.where(
        jnp.isfinite(solvent_denominator) & (solvent_denominator > 0.0),
        solvent_denominator,
        1.0,
    )
    sei_exchange = (
        _FARADAY_C_MOL
        * parameters.sei_reaction_rate_m_s
        * parameters.sei_solvent_concentration_mol_m3
        / safe_solvent_denominator
    )
    exponent = (
        -parameters.sei_transfer_coefficient
        * _FARADAY_C_MOL
        * sei_overpotential
        / (_GAS_CONSTANT_J_MOL_K * spm.temperature_k)
    )
    dtype = jnp.result_type(exponent, float)
    exponent_limit = jnp.log(jnp.asarray(jnp.finfo(dtype).max, dtype=dtype)) - 4.0
    exponent_domain = jnp.all(
        jnp.isfinite(exponent) & (exponent < exponent_limit), axis=-1
    )
    safe_exponent = jnp.clip(exponent, -exponent_limit, exponent_limit)
    sei_current_density = (
        -profile.negative_specific_surface_area_m2_m3
        * sei_exchange
        * jnp.exp(safe_exponent)
    )
    sei_current_density = jnp.where(
        parameters.sei_reaction_rate_m_s == 0.0,
        jnp.zeros_like(sei_current_density),
        sei_current_density,
    )
    averaged_sei_current_density = _weighted_mean(sei_current_density, negative_weights)
    side_current = -spm.electrode_area_m2 * jnp.sum(
        sei_current_density * negative_widths_m, axis=-1
    )
    negative_intercalation_current = current - side_current
    current_balance_residual = negative_intercalation_current + side_current - current
    intercalation_current_valid = jnp.isfinite(negative_intercalation_current) & (
        jnp.abs(negative_intercalation_current) <= spm.maximum_absolute_current_a
    )
    safe_intercalation_current = jnp.where(
        intercalation_current_valid & profile.domain_valid,
        negative_intercalation_current,
        0.0,
    )
    negative_outward_flux = -safe_intercalation_current / (
        _FARADAY_C_MOL * profile.negative_active_surface_area_m2
    )
    negative_particle = prepared.spm.negative_particle.evaluate(
        state.negative_amount_mol,
        particle_radius_m=spm.negative_particle_radius_m,
        particle_multiplicity=profile.negative_particle_multiplicity,
        support_volume_m3=profile.negative_support_volume_m3,
        diffusivity_m2_s=profile.negative_diffusivity_m2_s,
        outward_molar_flux_mol_m2_s=negative_outward_flux,
    )
    positive_particle = base.particle_transport.positive
    negative_surface_concentration = negative_particle.surface_concentration_mol_m3
    positive_surface_concentration = positive_particle.surface_concentration_mol_m3
    negative_surface_stoichiometry = (
        negative_surface_concentration / spm.negative_maximum_concentration_mol_m3
    )
    positive_surface_stoichiometry = (
        positive_surface_concentration / spm.positive_maximum_concentration_mol_m3
    )
    negative_ocp = spm.negative_open_circuit_potential.evaluate(
        negative_surface_stoichiometry
    )
    positive_ocp = spm.positive_open_circuit_potential.evaluate(
        positive_surface_stoichiometry
    )
    particle_ocp = positive_ocp.values - negative_ocp.values
    weights = prepared.through_cell.reference_cell_widths
    concentration = electrolyte.concentration_mol_m3
    negative_mean = _region_mean(
        concentration, weights, prepared.through_cell.negative_mask
    )
    separator_mean = _region_mean(
        concentration, weights, prepared.through_cell.separator_mask
    )
    positive_mean = _region_mean(
        concentration, weights, prepared.through_cell.positive_mask
    )
    negative_occupancy = negative_surface_stoichiometry * (
        1.0 - negative_surface_stoichiometry
    )
    positive_occupancy = positive_surface_stoichiometry * (
        1.0 - positive_surface_stoichiometry
    )
    typical = marquis.typical_electrolyte_concentration_mol_m3
    positive_start = (
        prepared.through_cell.negative.cell_count
        + prepared.through_cell.separator.cell_count
    )
    positive_concentration = concentration[..., positive_start:]
    positive_weights = weights[positive_start:]
    negative_exchange_local = (
        4.0
        * profile.negative_exchange_current_density_scale_a_m2
        * jnp.sqrt(jnp.maximum(negative_occupancy[..., None], 0.0))
        * jnp.sqrt(jnp.maximum(negative_concentration / typical, 0.0))
    )
    positive_exchange_local = (
        4.0
        * profile.positive_exchange_current_density_scale_a_m2
        * jnp.sqrt(jnp.maximum(positive_occupancy[..., None], 0.0))
        * jnp.sqrt(jnp.maximum(positive_concentration / typical, 0.0))
    )
    negative_exchange = _weighted_mean(negative_exchange_local, negative_weights)
    positive_exchange = _weighted_mean(positive_exchange_local, positive_weights)
    negative_reaction_integrand = _stable_asinh_ratio(
        passive_current_density[..., None],
        profile.negative_specific_surface_area_m2_m3
        * spm.negative_electrode_thickness_m
        * negative_exchange_local,
    )
    positive_reaction_integrand = _stable_asinh_ratio(
        passive_current_density[..., None],
        profile.positive_specific_surface_area_m2_m3
        * spm.positive_electrode_thickness_m
        * positive_exchange_local,
    )
    reaction_overpotential = thermal_voltage_twice * (
        _weighted_mean(negative_reaction_integrand, negative_weights)
        + _weighted_mean(positive_reaction_integrand, positive_weights)
    )
    negative_concentration_potential_mean = _weighted_mean(
        electrolyte_concentration_potential[..., :negative_count],
        negative_weights,
    )
    positive_concentration_potential_mean = _weighted_mean(
        electrolyte_concentration_potential[..., positive_start:],
        positive_weights,
    )
    concentration_overpotential = (
        positive_concentration_potential_mean - negative_concentration_potential_mean
    )
    electrolyte_ohmic_potential = -electrolyte_ohmic_integral
    negative_ohmic_potential_mean = _weighted_mean(
        electrolyte_ohmic_potential[..., :negative_count],
        negative_weights,
    )
    positive_ohmic_potential_mean = _weighted_mean(
        electrolyte_ohmic_potential[..., positive_start:],
        positive_weights,
    )
    electrolyte_ohmic_loss = positive_ohmic_potential_mean - negative_ohmic_potential_mean
    solid_ohmic_loss = (
        passive_current_density
        / 3.0
        * (
            spm.positive_electrode_thickness_m / marquis.positive_solid_conductivity_s_m
            + spm.negative_electrode_thickness_m / marquis.negative_solid_conductivity_s_m
        )
    )
    film_voltage_correction = (
        passive_current_density
        * mean_film_thickness
        / (
            spm.negative_electrode_thickness_m
            * profile.negative_specific_surface_area_m2_m3
            * parameters.sei_conductivity_s_m
        )
    )
    voltage = (
        particle_ocp
        + reaction_overpotential
        + concentration_overpotential
        + electrolyte_ohmic_loss
        + solid_ohmic_loss
        + film_voltage_correction
    )

    def choose_base(base_value, specialized_value):
        return jnp.where(zero_sei_reduction, base_value, specialized_value)

    particle_ocp = choose_base(base.particle_ocp_v, particle_ocp)
    reaction_overpotential = choose_base(
        base.reaction_overpotential_v, reaction_overpotential
    )
    concentration_overpotential = choose_base(
        base.mean_concentration_overpotential_v, concentration_overpotential
    )
    electrolyte_ohmic_loss = choose_base(
        base.electrolyte_ohmic_loss_v, electrolyte_ohmic_loss
    )
    solid_ohmic_loss = choose_base(base.solid_ohmic_loss_v, solid_ohmic_loss)
    film_voltage_correction = jnp.where(zero_sei_reduction, 0.0, film_voltage_correction)
    voltage = choose_base(base.voltage_v, voltage)
    negative_surface_stoichiometry = choose_base(
        base.negative_surface_stoichiometry, negative_surface_stoichiometry
    )
    negative_surface_concentration = choose_base(
        base.negative_surface_concentration_mol_m3,
        negative_surface_concentration,
    )
    negative_mean = choose_base(base.negative_electrolyte_mean_mol_m3, negative_mean)
    separator_mean = choose_base(base.separator_electrolyte_mean_mol_m3, separator_mean)
    positive_mean = choose_base(base.positive_electrolyte_mean_mol_m3, positive_mean)
    negative_exchange = choose_base(
        base.negative_exchange_current_density_a_m2, negative_exchange
    )
    positive_exchange = choose_base(
        base.positive_exchange_current_density_a_m2, positive_exchange
    )

    sei_lithium, sei_product, film_mass, _ = _sei_inventory(
        prepared, state.negative_porosity, parameters
    )
    total_solid = negative_particle.total_amount_mol + positive_particle.total_amount_mol
    total_solid = choose_base(base.total_solid_lithium_mol, total_solid)
    total_electrolyte = electrolyte.total_amount_mol
    total_lithium = total_solid + total_electrolyte + sei_lithium
    weak_ratio = side_current / spm.maximum_absolute_current_a
    maximum_overpotential = jnp.max(
        jnp.abs(
            jnp.stack(
                (
                    reaction_overpotential,
                    concentration_overpotential,
                    electrolyte_ohmic_loss,
                    solid_ohmic_loss,
                    film_voltage_correction,
                ),
                axis=-1,
            )
        ),
        axis=-1,
    )
    weak_condition = weak_ratio <= prepared.plan.weak_side_reaction_threshold
    small_overpotential_condition = (
        maximum_overpotential <= prepared.plan.small_overpotential_threshold_v
    )
    asymptotic_conditions = (
        base.asymptotic_conditions_satisfied
        & weak_condition
        & small_overpotential_condition
    )
    porosity_domain = jnp.all(
        jnp.isfinite(state.negative_porosity)
        & (state.negative_porosity > 0.0)
        & (state.negative_porosity <= marquis.negative_electrolyte_porosity),
        axis=-1,
    )
    film_domain = jnp.all(jnp.isfinite(film_thickness) & (film_thickness >= 0.0), axis=-1)
    side_finite = (
        jnp.all(jnp.isfinite(sei_current_density), axis=-1)
        & jnp.all(jnp.isfinite(sei_overpotential), axis=-1)
        & jnp.isfinite(side_current)
        & jnp.isfinite(film_voltage_correction)
        & jnp.isfinite(sei_lithium)
        & jnp.isfinite(film_mass)
    )
    domain_valid = (
        base.domain_valid
        & profile.domain_valid
        & electrolyte.domain_valid
        & conductivity_valid
        & thermodynamic_valid
        & negative_particle.domain_valid
        & positive_particle.domain_valid
        & intercalation_current_valid
        & preliminary_ocp.support
        & negative_ocp.support
        & positive_ocp.support
        & solvent_domain
        & exponent_domain
        & porosity_domain
        & film_domain
        & side_finite
    )
    return _BrosaPlanellaSpmeSeiEvaluation(
        voltage,
        particle_ocp,
        reaction_overpotential,
        concentration_overpotential,
        electrolyte_ohmic_loss,
        solid_ohmic_loss,
        film_voltage_correction,
        negative_surface_stoichiometry,
        positive_surface_stoichiometry,
        negative_surface_concentration,
        positive_surface_concentration,
        negative_particle.average_concentration_mol_m3,
        positive_particle.average_concentration_mol_m3,
        negative_mean,
        separator_mean,
        positive_mean,
        negative_exchange,
        positive_exchange,
        sei_current_density,
        averaged_sei_current_density,
        side_current,
        negative_intercalation_current,
        current_balance_residual,
        sei_overpotential,
        film_thickness,
        film_resistance,
        state.negative_porosity,
        sei_lithium,
        sei_product,
        film_mass,
        total_solid,
        total_electrolyte,
        total_lithium,
        weak_ratio,
        maximum_overpotential,
        weak_condition,
        small_overpotential_condition,
        asymptotic_conditions,
        zero_sei_reduction,
        domain_valid,
        negative_particle,
        positive_particle,
        electrolyte,
        base,
    )


class _BrosaPlanellaSpmeSeiVectorField(StrictModule, NonTrainableState):
    prepared: PreparedBrosaPlanellaSpmeSei

    def __call__(
        self,
        time_s: Array,
        state: BrosaPlanellaSpmeSeiState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> BrosaPlanellaSpmeSeiState:
        parameters = runtime_inputs.parameters
        if not isinstance(parameters, BrosaPlanellaSpmeSeiParameters):
            raise TypeError("SPMe+SEI runtime parameters have the wrong type.")
        current = runtime_inputs.current(time_s, state.negative_amount_mol)
        evaluation = _evaluate(self.prepared, state, parameters, current)
        porosity_rate = (
            parameters.sei_molar_mass_kg_mol
            / (
                parameters.sei_electron_stoichiometry
                * parameters.sei_density_kg_m3
                * _FARADAY_C_MOL
            )
            * evaluation.sei_current_density_a_m3
        )
        return BrosaPlanellaSpmeSeiState(
            evaluation.negative_particle_transport.amount_rate_mol_s,
            evaluation.positive_particle_transport.amount_rate_mol_s,
            evaluation.electrolyte_transport.amount_rate_mol_s,
            porosity_rate,
        )


_SEI_OBSERVABLE_NAMES = (
    "voltage:sei_film_correction_v",
    "sei_current_density_mean_a_m3",
    "side_current_a",
    "negative_intercalation_current_a",
    "terminal_current_balance_residual_a",
    "negative_porosity_mean",
    "negative_porosity_minimum",
    "sei_film_thickness_mean_m",
    "sei_film_thickness_maximum_m",
    "sei_film_resistance_mean_ohm_m2",
    "sei_lithium_mol",
    "sei_product_mol",
    "sei_film_mass_kg",
    "spme_sr:weak_side_reaction_ratio",
    "spme_sr:maximum_overpotential_v",
    "spme_sr:weak_side_reaction_condition_satisfied",
    "spme_sr:small_overpotential_condition_satisfied",
    "spme_sr:conditions_satisfied",
    "spme_sr:zero_sei_reduction",
)
_SEI_OBSERVABLE_UNITS = (
    "V",
    "A/m3",
    "A",
    "A",
    "A",
    "1",
    "1",
    "m",
    "m",
    "ohm*m2",
    "mol",
    "mol",
    "kg",
    "1",
    "V",
    "1",
    "1",
    "1",
    "1",
)
_OBSERVABLE_NAMES = _MARQUIS_OBSERVABLE_NAMES + _SEI_OBSERVABLE_NAMES
_OBSERVABLE_UNITS = _MARQUIS_OBSERVABLE_UNITS + _SEI_OBSERVABLE_UNITS


class BrosaPlanellaSpmeSeiAdapter(StrictModule, NonTrainableState):
    """Native ODE adapter for the isothermal Brosa Planella--Widanage SPMe+SEI."""

    plan: BrosaPlanellaSpmeSeiPlan
    model_id: str = eqx.field(static=True)
    equation_form: Literal["ode"] = eqx.field(static=True)
    observable_names: tuple[str, ...] = eqx.field(static=True)
    observable_units: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, plan: BrosaPlanellaSpmeSeiPlan, /):
        if not isinstance(plan, BrosaPlanellaSpmeSeiPlan):
            raise TypeError("plan must be BrosaPlanellaSpmeSeiPlan.")
        self.plan = plan
        self.model_id = "battery:spme:sei:brosa-planella-widanage:isothermal"
        self.equation_form = "ode"
        self.observable_names = _OBSERVABLE_NAMES
        self.observable_units = _OBSERVABLE_UNITS

    def prepare(self, /) -> PreparedBrosaPlanellaSpmeSei:
        return self.plan.prepare()

    def initial_state(
        self,
        prepared_model: PreparedBrosaPlanellaSpmeSei,
        parameters: BrosaPlanellaSpmeSeiParameters,
        initial_condition: BrosaPlanellaSpmeSeiInitialCondition,
        /,
    ) -> BrosaPlanellaSpmeSeiState:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(parameters, BrosaPlanellaSpmeSeiParameters):
            raise TypeError("parameters must be BrosaPlanellaSpmeSeiParameters.")
        if not isinstance(initial_condition, BrosaPlanellaSpmeSeiInitialCondition):
            raise TypeError(
                "initial_condition must be BrosaPlanellaSpmeSeiInitialCondition."
            )
        base_adapter = Marquis2019SpmeAdapter(self.plan.marquis_plan)
        base = base_adapter.initial_state(
            prepared_model.marquis,
            parameters.spme_parameters,
            Marquis2019SpmeInitialCondition(
                initial_condition.negative_stoichiometry,
                initial_condition.positive_stoichiometry,
            ),
        )
        negative_porosity = jnp.full(
            (prepared_model.through_cell.negative.cell_count,),
            parameters.spme_parameters.negative_electrolyte_porosity,
            dtype=base.electrolyte_amount_mol.dtype,
        )
        return BrosaPlanellaSpmeSeiState(
            base.negative_amount_mol,
            base.positive_amount_mol,
            base.electrolyte_amount_mol,
            negative_porosity,
        )

    def problem(
        self,
        prepared_model: PreparedBrosaPlanellaSpmeSei,
        initial_state: BrosaPlanellaSpmeSeiState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> DifferentialProblem:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(initial_state, BrosaPlanellaSpmeSeiState):
            raise TypeError("initial_state must be BrosaPlanellaSpmeSeiState.")
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        if not isinstance(runtime_inputs.parameters, BrosaPlanellaSpmeSeiParameters):
            raise TypeError("SPMe+SEI runtime parameters have the wrong type.")
        expected_negative = prepared_model.spm.negative_particle.shell_count
        expected_positive = prepared_model.spm.positive_particle.shell_count
        expected_electrolyte = prepared_model.through_cell.cell_count
        expected_porosity = prepared_model.through_cell.negative.cell_count
        if initial_state.negative_amount_mol.shape != (expected_negative,):
            raise ValueError(
                f"Initial negative SPMe+SEI state must have shape ({expected_negative},)."
            )
        if initial_state.positive_amount_mol.shape != (expected_positive,):
            raise ValueError(
                f"Initial positive SPMe+SEI state must have shape ({expected_positive},)."
            )
        if initial_state.electrolyte_amount_mol.shape != (expected_electrolyte,):
            raise ValueError(
                "Initial electrolyte SPMe+SEI state must have shape "
                f"({expected_electrolyte},)."
            )
        if initial_state.negative_porosity.shape != (expected_porosity,):
            raise ValueError(
                f"Initial negative porosity state must have shape ({expected_porosity},)."
            )
        return DifferentialProblem(
            _BrosaPlanellaSpmeSeiVectorField(prepared_model),
            initial_state,
            t0=runtime_inputs.input_policy.times[0],
            t1=runtime_inputs.input_policy.times[-1],
            args=runtime_inputs,
            problem_id=canonical_fingerprint(
                {
                    "kind": "battery-spme-sei-brosa-planella-widanage-problem",
                    "model_id": self.model_id,
                    "prepared_id": prepared_model.prepared_id,
                    "parameter_id": runtime_inputs.parameters.parameter_id,
                    "protocol_id": runtime_inputs.protocol_id,
                }
            ),
        )

    def observe(
        self,
        prepared_model: PreparedBrosaPlanellaSpmeSei,
        times_s: Array,
        states: BrosaPlanellaSpmeSeiState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> BatteryModelOutput:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(states, BrosaPlanellaSpmeSeiState):
            raise TypeError("states must be BrosaPlanellaSpmeSeiState.")
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        if not isinstance(runtime_inputs.parameters, BrosaPlanellaSpmeSeiParameters):
            raise TypeError("SPMe+SEI runtime parameters have the wrong type.")
        times = jnp.asarray(times_s)
        expected_negative = times.shape + (
            prepared_model.spm.negative_particle.shell_count,
        )
        expected_positive = times.shape + (
            prepared_model.spm.positive_particle.shell_count,
        )
        expected_electrolyte = times.shape + (prepared_model.through_cell.cell_count,)
        expected_porosity = times.shape + (
            prepared_model.through_cell.negative.cell_count,
        )
        if states.negative_amount_mol.shape != expected_negative:
            raise ValueError(
                f"Negative SPMe+SEI observation state must have shape {expected_negative}."
            )
        if states.positive_amount_mol.shape != expected_positive:
            raise ValueError(
                f"Positive SPMe+SEI observation state must have shape {expected_positive}."
            )
        if states.electrolyte_amount_mol.shape != expected_electrolyte:
            raise ValueError(
                "Electrolyte SPMe+SEI observation state must have shape "
                f"{expected_electrolyte}."
            )
        if states.negative_porosity.shape != expected_porosity:
            raise ValueError(
                f"Porosity observation state must have shape {expected_porosity}."
            )
        flat_current = jax.vmap(runtime_inputs.observed_current)(times.reshape((-1,)))
        current = flat_current.reshape(times.shape)
        parameters = runtime_inputs.parameters
        evaluation = _evaluate(prepared_model, states, parameters, current)
        base = evaluation.marquis_evaluation
        electrolyte = evaluation.electrolyte_transport
        ns_face = prepared_model.through_cell.negative_separator_face_index
        sp_face = prepared_model.through_cell.separator_positive_face_index
        values = jnp.stack(
            (
                evaluation.voltage_v,
                jnp.zeros_like(evaluation.voltage_v)
                + parameters.spme_parameters.spm_parameters.temperature_k,
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
                evaluation.total_lithium_mol,
                evaluation.particle_ocp_v,
                evaluation.reaction_overpotential_v,
                evaluation.mean_concentration_overpotential_v,
                evaluation.electrolyte_ohmic_loss_v,
                evaluation.solid_ohmic_loss_v,
                evaluation.negative_exchange_current_density_a_m2,
                evaluation.positive_exchange_current_density_a_m2,
                current / parameters.spme_parameters.spm_parameters.electrode_area_m2,
                -current / parameters.spme_parameters.spm_parameters.electrode_area_m2,
                electrolyte.electrolyte_current_density_a_m2[..., ns_face],
                electrolyte.electrolyte_current_density_a_m2[..., sp_face],
                electrolyte.current_split_residual_a_m2,
                electrolyte.collector_flux_residual_mol_m2_s,
                electrolyte.negative_separator_concentration_jump_mol_m3,
                electrolyte.separator_positive_concentration_jump_mol_m3,
                electrolyte.conservation_residual_mol_s,
                electrolyte.equation_mapping_residual_mol_s,
                electrolyte.explicit_dt_limit_s,
                base.electrolyte_migration_number,
                base.negative_thermal_to_solid_ohmic_ratio,
                base.positive_thermal_to_solid_ohmic_ratio,
                base.thermal_to_electrolyte_ohmic_ratio,
                base.negative_solid_diffusion_number,
                base.positive_solid_diffusion_number,
                base.negative_reaction_number,
                base.positive_reaction_number,
                jnp.zeros_like(evaluation.voltage_v) + base.negative_length_ratio,
                jnp.zeros_like(evaluation.voltage_v) + base.separator_length_ratio,
                jnp.zeros_like(evaluation.voltage_v) + base.positive_length_ratio,
                jnp.zeros_like(evaluation.voltage_v)
                + base.positive_to_negative_maximum_concentration_ratio,
                jnp.zeros_like(evaluation.voltage_v)
                + base.electrolyte_to_negative_maximum_concentration_ratio,
                base.asymptotic_conditions_satisfied.astype(evaluation.voltage_v.dtype),
                base.eq49_applicable.astype(evaluation.voltage_v.dtype),
                base.eq49_electrolyte_error,
                base.eq49_negative_ocp_error,
                base.eq49_positive_ocp_error,
                base.eq49_error_estimate,
                evaluation.voltage_v * current,
                parameters.spme_parameters.spm_parameters.maximum_absolute_current_a
                - jnp.abs(current),
                evaluation.film_voltage_correction_v,
                evaluation.electrode_averaged_sei_current_density_a_m3,
                evaluation.side_current_a,
                evaluation.negative_intercalation_current_a,
                evaluation.terminal_current_balance_residual_a,
                _weighted_mean(
                    evaluation.negative_porosity,
                    prepared_model.through_cell.reference_cell_widths[
                        : prepared_model.through_cell.negative.cell_count
                    ],
                ),
                jnp.min(evaluation.negative_porosity, axis=-1),
                _weighted_mean(
                    evaluation.film_thickness_m,
                    prepared_model.through_cell.reference_cell_widths[
                        : prepared_model.through_cell.negative.cell_count
                    ],
                ),
                jnp.max(evaluation.film_thickness_m, axis=-1),
                _weighted_mean(
                    evaluation.film_resistance_ohm_m2,
                    prepared_model.through_cell.reference_cell_widths[
                        : prepared_model.through_cell.negative.cell_count
                    ],
                ),
                evaluation.sei_lithium_amount_mol,
                evaluation.sei_product_amount_mol,
                evaluation.film_mass_kg,
                evaluation.weak_side_reaction_ratio,
                evaluation.maximum_overpotential_v,
                evaluation.weak_side_reaction_condition_satisfied.astype(
                    evaluation.voltage_v.dtype
                ),
                evaluation.small_overpotential_condition_satisfied.astype(
                    evaluation.voltage_v.dtype
                ),
                evaluation.asymptotic_conditions_satisfied.astype(
                    evaluation.voltage_v.dtype
                ),
                evaluation.zero_sei_reduction.astype(evaluation.voltage_v.dtype),
            ),
            axis=-1,
        )
        domain_valid = evaluation.domain_valid & jnp.all(jnp.isfinite(values), axis=-1)
        return BatteryModelOutput(values, domain_valid)

    def ledger(
        self,
        prepared_model: PreparedBrosaPlanellaSpmeSei,
        native_solution,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> BrosaPlanellaSpmeSeiLedger:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        if not isinstance(runtime_inputs.parameters, BrosaPlanellaSpmeSeiParameters):
            raise TypeError("SPMe+SEI runtime parameters have the wrong type.")
        states = native_solution.states
        if not isinstance(states, BrosaPlanellaSpmeSeiState):
            raise TypeError("SPMe+SEI native solution states have the wrong type.")
        valid = jnp.asarray(native_solution.valid, dtype=bool)
        valid_count = jnp.sum(valid.astype(jnp.int32))
        final_index = jnp.maximum(valid_count - 1, 0)
        parameters = runtime_inputs.parameters

        initial_negative = jnp.sum(states.negative_amount_mol[0])
        final_negative = jnp.sum(states.negative_amount_mol[final_index])
        initial_positive = jnp.sum(states.positive_amount_mol[0])
        final_positive = jnp.sum(states.positive_amount_mol[final_index])
        initial_electrolyte = jnp.sum(states.electrolyte_amount_mol[0])
        final_electrolyte = jnp.sum(states.electrolyte_amount_mol[final_index])
        initial_sei, initial_product, initial_mass, initial_thickness = _sei_inventory(
            prepared_model, states.negative_porosity[0], parameters
        )
        final_sei, final_product, final_mass, final_thickness = _sei_inventory(
            prepared_model, states.negative_porosity[final_index], parameters
        )
        initial_total = (
            initial_negative + initial_positive + initial_electrolyte + initial_sei
        )
        final_total = final_negative + final_positive + final_electrolyte + final_sei
        total_residual = final_total - initial_total
        electrolyte_residual = final_electrolyte - initial_electrolyte
        start_time = native_solution.times[0]
        final_time = native_solution.times[final_index]
        policy_times = runtime_inputs.input_policy.times
        interval_left = jnp.maximum(policy_times[:-1], start_time)
        interval_right = jnp.minimum(policy_times[1:], final_time)
        interval_duration = jnp.maximum(interval_right - interval_left, 0.0)
        terminal_charge = jnp.sum(
            interval_duration * runtime_inputs.input_policy.values[:, 0]
        )
        side_charge = _FARADAY_C_MOL * (final_sei - initial_sei)
        intercalation_charge = terminal_charge - side_charge
        negative_charge_residual = (
            _FARADAY_C_MOL * (final_negative - initial_negative) - intercalation_charge
        )
        positive_charge_residual = (
            -_FARADAY_C_MOL * (final_positive - initial_positive) - terminal_charge
        )
        side_product_residual = side_charge - _FARADAY_C_MOL * (final_sei - initial_sei)
        mass_product_residual = jnp.maximum(
            jnp.abs(initial_mass - parameters.sei_molar_mass_kg_mol * initial_product),
            jnp.abs(final_mass - parameters.sei_molar_mass_kg_mol * final_product),
        )
        specific_area = _profile(
            parameters.spme_parameters.spm_parameters
        ).negative_specific_surface_area_m2_m3
        porosity_thickness_residual = jnp.max(
            jnp.abs(
                (final_thickness - initial_thickness)
                + (states.negative_porosity[final_index] - states.negative_porosity[0])
                / specific_area
            )
        )
        times = native_solution.times
        currents = jax.vmap(runtime_inputs.observed_current)(times)
        evaluation = _evaluate(prepared_model, states, parameters, currents)
        electrolyte = evaluation.electrolyte_transport

        def valid_max(values):
            return jnp.max(jnp.where(valid, jnp.abs(values), 0.0))

        maximum_collector_flux = valid_max(electrolyte.collector_flux_residual_mol_m2_s)
        maximum_interface_jump = jnp.maximum(
            valid_max(electrolyte.negative_separator_concentration_jump_mol_m3),
            valid_max(electrolyte.separator_positive_concentration_jump_mol_m3),
        )
        maximum_mapping_residual = valid_max(electrolyte.equation_mapping_residual_mol_s)
        maximum_current_split = valid_max(electrolyte.current_split_residual_a_m2)
        maximum_rest_cancellation = valid_max(
            evaluation.terminal_current_balance_residual_a
        )
        maximum_weak_ratio = jnp.max(
            jnp.where(valid, evaluation.weak_side_reaction_ratio, 0.0)
        )
        maximum_overpotential = jnp.max(
            jnp.where(valid, evaluation.maximum_overpotential_v, 0.0)
        )
        domain_valid = (valid_count > 0) & jnp.all(
            jnp.where(valid, evaluation.domain_valid, True)
        )
        weak_condition = (valid_count > 0) & jnp.all(
            jnp.where(
                valid,
                evaluation.weak_side_reaction_condition_satisfied,
                True,
            )
        )
        small_overpotential_condition = (valid_count > 0) & jnp.all(
            jnp.where(
                valid,
                evaluation.small_overpotential_condition_satisfied,
                True,
            )
        )
        asymptotic_conditions = (valid_count > 0) & jnp.all(
            jnp.where(valid, evaluation.asymptotic_conditions_satisfied, True)
        )
        scalars = jnp.stack(
            (
                initial_negative,
                final_negative,
                initial_positive,
                final_positive,
                initial_electrolyte,
                final_electrolyte,
                initial_sei,
                final_sei,
                initial_product,
                final_product,
                initial_total,
                final_total,
                initial_mass,
                final_mass,
                terminal_charge,
                side_charge,
                intercalation_charge,
                total_residual,
                electrolyte_residual,
                negative_charge_residual,
                positive_charge_residual,
                side_product_residual,
                mass_product_residual,
                porosity_thickness_residual,
                maximum_collector_flux,
                maximum_interface_jump,
                maximum_mapping_residual,
                maximum_current_split,
                maximum_rest_cancellation,
                maximum_weak_ratio,
                maximum_overpotential,
            )
        )
        finite = (valid_count > 0) & jnp.all(jnp.isfinite(scalars))
        plan = prepared_model.plan
        marquis_plan = plan.marquis_plan
        relative = marquis_plan.ledger_relative_tolerance
        amount_scale = jnp.maximum(jnp.abs(initial_total), jnp.abs(final_total))
        lithium_conserved = jnp.abs(total_residual) <= (
            marquis_plan.ledger_amount_absolute_tolerance_mol + relative * amount_scale
        )
        electrolyte_scale = jnp.maximum(
            jnp.abs(initial_electrolyte), jnp.abs(final_electrolyte)
        )
        electrolyte_conserved = jnp.abs(electrolyte_residual) <= (
            marquis_plan.ledger_amount_absolute_tolerance_mol
            + relative * electrolyte_scale
        )
        charge_scale = jnp.maximum(jnp.abs(terminal_charge), jnp.abs(side_charge))
        charge_conserved = jnp.maximum(
            jnp.abs(negative_charge_residual),
            jnp.abs(positive_charge_residual),
        ) <= (marquis_plan.ledger_charge_absolute_tolerance_c + relative * charge_scale)
        product_conserved = jnp.abs(side_product_residual) <= (
            marquis_plan.ledger_charge_absolute_tolerance_c
            + relative * jnp.abs(side_charge)
        )
        film_conserved = mass_product_residual <= (
            plan.ledger_film_mass_absolute_tolerance_kg
            + relative * jnp.maximum(jnp.abs(initial_mass), jnp.abs(final_mass))
        )
        porosity_conserved = (
            porosity_thickness_residual <= plan.ledger_porosity_absolute_tolerance
        )
        concentration_scale = (
            parameters.spme_parameters.typical_electrolyte_concentration_mol_m3
        )
        interfaces_conserved = (
            (
                maximum_collector_flux
                <= marquis_plan.ledger_molar_flux_absolute_tolerance_mol_m2_s
            )
            & (
                maximum_interface_jump
                <= marquis_plan.ledger_concentration_absolute_tolerance_mol_m3
                + relative * concentration_scale
            )
            & (
                maximum_mapping_residual
                <= marquis_plan.ledger_rate_absolute_tolerance_mol_s
            )
        )
        current_split_conserved = (
            maximum_current_split
            <= marquis_plan.ledger_current_density_absolute_tolerance_a_m2
        )
        rest_cancellation_conserved = (
            maximum_rest_cancellation <= marquis_plan.ledger_charge_absolute_tolerance_c
        )
        successful = (
            finite
            & domain_valid
            & lithium_conserved
            & electrolyte_conserved
            & charge_conserved
            & product_conserved
            & film_conserved
            & porosity_conserved
            & interfaces_conserved
            & current_split_conserved
            & rest_cancellation_conserved
        )
        return BrosaPlanellaSpmeSeiLedger(
            initial_negative,
            final_negative,
            initial_positive,
            final_positive,
            initial_electrolyte,
            final_electrolyte,
            initial_sei,
            final_sei,
            initial_product,
            final_product,
            initial_total,
            final_total,
            initial_mass,
            final_mass,
            terminal_charge,
            side_charge,
            intercalation_charge,
            total_residual,
            electrolyte_residual,
            negative_charge_residual,
            positive_charge_residual,
            side_product_residual,
            mass_product_residual,
            porosity_thickness_residual,
            maximum_collector_flux,
            maximum_interface_jump,
            maximum_mapping_residual,
            maximum_current_split,
            maximum_rest_cancellation,
            maximum_weak_ratio,
            maximum_overpotential,
            lithium_conserved,
            electrolyte_conserved,
            charge_conserved,
            product_conserved,
            film_conserved,
            porosity_conserved,
            interfaces_conserved,
            current_split_conserved,
            rest_cancellation_conserved,
            weak_condition,
            small_overpotential_condition,
            asymptotic_conditions,
            domain_valid,
            finite,
            successful,
        )


__all__ = [
    "BrosaPlanellaSpmeSeiAdapter",
    "BrosaPlanellaSpmeSeiInitialCondition",
    "BrosaPlanellaSpmeSeiLedger",
    "BrosaPlanellaSpmeSeiParameters",
    "BrosaPlanellaSpmeSeiPlan",
    "BrosaPlanellaSpmeSeiState",
    "PreparedBrosaPlanellaSpmeSei",
]
