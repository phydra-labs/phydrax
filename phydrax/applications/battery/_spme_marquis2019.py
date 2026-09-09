#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

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
from ._spm import (
    _stable_asinh_ratio,
    _transport,
    PreparedPrescribedCurrentSpm,
    PrescribedCurrentSpmAdapter,
    PrescribedCurrentSpmPlan,
    SpmInitialCondition,
    SpmParameters,
    SpmState,
)
from ._through_cell import PreparedThroughCellMesh, ThroughCellRegionPlan


_FARADAY_C_MOL = 96485.33212
_GAS_CONSTANT_J_MOL_K = 8.31446261815324


def _scalar(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape != () or jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must be one real scalar.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return array


def _electrolyte_law(
    value: ConcentrationTemperaturePropertyLaw,
    name: str,
    value_unit: str,
    /,
) -> ConcentrationTemperaturePropertyLaw:
    if not isinstance(value, ConcentrationTemperaturePropertyLaw):
        raise TypeError(f"{name} must be ConcentrationTemperaturePropertyLaw.")
    if value.value_unit != value_unit:
        raise ValueError(f"{name} must use value unit {value_unit!r}.")
    return value


class Marquis2019SpmeParameters(StrictModule):
    """Dynamic SI data for the dimensional isothermal Marquis et al. SPMe."""

    spm_parameters: SpmParameters
    separator_thickness_m: Array
    negative_electrolyte_porosity: Array
    separator_electrolyte_porosity: Array
    positive_electrolyte_porosity: Array
    bruggeman_coefficient: Array
    typical_electrolyte_concentration_mol_m3: Array
    electrolyte_diffusivity: ConcentrationTemperaturePropertyLaw
    electrolyte_conductivity: ConcentrationTemperaturePropertyLaw
    transference_number: ConcentrationTemperaturePropertyLaw
    negative_solid_conductivity_s_m: Array
    positive_solid_conductivity_s_m: Array
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        spm_parameters: SpmParameters,
        /,
        *,
        separator_thickness_m: ArrayLike,
        negative_electrolyte_porosity: ArrayLike,
        separator_electrolyte_porosity: ArrayLike,
        positive_electrolyte_porosity: ArrayLike,
        bruggeman_coefficient: ArrayLike,
        typical_electrolyte_concentration_mol_m3: ArrayLike,
        electrolyte_diffusivity: ConcentrationTemperaturePropertyLaw,
        electrolyte_conductivity: ConcentrationTemperaturePropertyLaw,
        transference_number: ConcentrationTemperaturePropertyLaw,
        negative_solid_conductivity_s_m: ArrayLike,
        positive_solid_conductivity_s_m: ArrayLike,
    ):
        if not isinstance(spm_parameters, SpmParameters):
            raise TypeError("spm_parameters must be SpmParameters.")
        separator_thickness = _scalar(separator_thickness_m, "separator_thickness_m")
        negative_porosity = _scalar(
            negative_electrolyte_porosity, "negative_electrolyte_porosity"
        )
        separator_porosity = _scalar(
            separator_electrolyte_porosity, "separator_electrolyte_porosity"
        )
        positive_porosity = _scalar(
            positive_electrolyte_porosity, "positive_electrolyte_porosity"
        )
        bruggeman = _scalar(bruggeman_coefficient, "bruggeman_coefficient")
        typical_concentration = _scalar(
            typical_electrolyte_concentration_mol_m3,
            "typical_electrolyte_concentration_mol_m3",
        )
        diffusivity = _electrolyte_law(
            electrolyte_diffusivity,
            "electrolyte_diffusivity",
            "m2/s",
        )
        conductivity = _electrolyte_law(
            electrolyte_conductivity,
            "electrolyte_conductivity",
            "S/m",
        )
        transference = _electrolyte_law(
            transference_number,
            "transference_number",
            "1",
        )
        negative_conductivity = _scalar(
            negative_solid_conductivity_s_m, "negative_solid_conductivity_s_m"
        )
        positive_conductivity = _scalar(
            positive_solid_conductivity_s_m, "positive_solid_conductivity_s_m"
        )
        porosities = jnp.stack((negative_porosity, separator_porosity, positive_porosity))
        positive_values = jnp.stack(
            (
                separator_thickness,
                bruggeman,
                typical_concentration,
                negative_conductivity,
                positive_conductivity,
            )
        )
        temperature = spm_parameters.temperature_k
        typical_diffusivity = diffusivity.evaluate(typical_concentration, temperature)
        typical_conductivity = conductivity.evaluate(typical_concentration, temperature)
        typical_transference = transference.evaluate(typical_concentration, temperature)
        property_valid = (
            typical_diffusivity.support
            & jnp.isfinite(typical_diffusivity.values)
            & (typical_diffusivity.values > 0.0)
            & typical_conductivity.support
            & jnp.isfinite(typical_conductivity.values)
            & (typical_conductivity.values > 0.0)
            & typical_transference.support
            & jnp.isfinite(typical_transference.values)
            & (typical_transference.values >= 0.0)
            & (typical_transference.values <= 1.0)
        )
        valid = (
            jnp.all(jnp.isfinite(positive_values) & (positive_values > 0.0))
            & jnp.all(jnp.isfinite(porosities) & (porosities > 0.0) & (porosities <= 1.0))
            & property_valid
        )
        separator_thickness = eqx.error_if(
            separator_thickness,
            ~valid,
            "Marquis SPMe geometry, porosity, electrolyte laws, and conductivity "
            "data must support the typical state with finite physical values.",
        )
        self.spm_parameters = spm_parameters
        self.separator_thickness_m = separator_thickness
        self.negative_electrolyte_porosity = negative_porosity
        self.separator_electrolyte_porosity = separator_porosity
        self.positive_electrolyte_porosity = positive_porosity
        self.bruggeman_coefficient = bruggeman
        self.typical_electrolyte_concentration_mol_m3 = typical_concentration
        self.electrolyte_diffusivity = diffusivity
        self.electrolyte_conductivity = conductivity
        self.transference_number = transference
        self.negative_solid_conductivity_s_m = negative_conductivity
        self.positive_solid_conductivity_s_m = positive_conductivity
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "battery-spme-marquis-2019-parameters",
                "spm_parameter_id": spm_parameters.parameter_id,
                "electrolyte_diffusivity_law_id": diffusivity.law_id,
                "electrolyte_conductivity_law_id": conductivity.law_id,
                "transference_number_law_id": transference.law_id,
            }
        )


class Marquis2019SpmeInitialCondition(StrictModule):
    """Uniform solid stoichiometries; electrolyte starts at its typical value."""

    negative_stoichiometry: Array
    positive_stoichiometry: Array

    def __init__(
        self,
        negative_stoichiometry: ArrayLike,
        positive_stoichiometry: ArrayLike,
        /,
    ):
        initial = SpmInitialCondition(negative_stoichiometry, positive_stoichiometry)
        self.negative_stoichiometry = initial.negative_stoichiometry
        self.positive_stoichiometry = initial.positive_stoichiometry


class Marquis2019SpmeState(StrictModule):
    """Two extensive radial solid states and one extensive through-cell state."""

    negative_amount_mol: Array
    positive_amount_mol: Array
    electrolyte_amount_mol: Array

    def __init__(
        self,
        negative_amount_mol: ArrayLike,
        positive_amount_mol: ArrayLike,
        electrolyte_amount_mol: ArrayLike,
        /,
    ):
        particles = SpmState(negative_amount_mol, positive_amount_mol)
        electrolyte = jnp.asarray(electrolyte_amount_mol)
        if electrolyte.ndim < 1:
            raise ValueError("SPMe electrolyte amount must contain a through-cell axis.")
        if jnp.issubdtype(electrolyte.dtype, jnp.complexfloating):
            raise TypeError("SPMe electrolyte amount must be real-valued.")
        dtype = jnp.result_type(
            particles.negative_amount_mol,
            particles.positive_amount_mol,
            electrolyte,
            float,
        )
        self.negative_amount_mol = particles.negative_amount_mol.astype(dtype)
        self.positive_amount_mol = particles.positive_amount_mol.astype(dtype)
        self.electrolyte_amount_mol = electrolyte.astype(dtype)


class Marquis2019SpmeLedger(StrictModule):
    """Endpoint conservation, interface, current-split, and applicability evidence."""

    initial_negative_solid_lithium_mol: Array
    final_negative_solid_lithium_mol: Array
    initial_positive_solid_lithium_mol: Array
    final_positive_solid_lithium_mol: Array
    initial_solid_lithium_mol: Array
    final_solid_lithium_mol: Array
    initial_electrolyte_lithium_mol: Array
    final_electrolyte_lithium_mol: Array
    initial_total_lithium_mol: Array
    final_total_lithium_mol: Array
    solid_lithium_conservation_residual_mol: Array
    electrolyte_lithium_conservation_residual_mol: Array
    total_lithium_conservation_residual_mol: Array
    charge_conservation_residual_c: Array
    integrated_terminal_charge_c: Array
    negative_current_integral_residual_c: Array
    positive_current_integral_residual_c: Array
    maximum_collector_flux_residual_mol_m2_s: Array
    maximum_interface_concentration_jump_mol_m3: Array
    maximum_equation_mapping_residual_mol_s: Array
    maximum_current_split_residual_a_m2: Array
    maximum_eq49_error_estimate: Array
    eq49_applicable: Array
    solid_lithium_conserved: Array
    electrolyte_lithium_conserved: Array
    total_lithium_conserved: Array
    current_conserved: Array
    interfaces_conserved: Array
    current_split_conserved: Array
    asymptotic_conditions_satisfied: Array
    domain_valid: Array
    finite: Array
    successful: Array


class Marquis2019SpmePlan(StrictModule, NonTrainableState):
    """Static radial and three-region FV topology for the canonical 2019 SPMe."""

    spm_plan: PrescribedCurrentSpmPlan
    negative_region: ThroughCellRegionPlan
    separator_region: ThroughCellRegionPlan
    positive_region: ThroughCellRegionPlan
    asymptotic_small_parameter_threshold: float = eqx.field(static=True)
    ledger_amount_absolute_tolerance_mol: float = eqx.field(static=True)
    ledger_charge_absolute_tolerance_c: float = eqx.field(static=True)
    ledger_molar_flux_absolute_tolerance_mol_m2_s: float = eqx.field(static=True)
    ledger_concentration_absolute_tolerance_mol_m3: float = eqx.field(static=True)
    ledger_rate_absolute_tolerance_mol_s: float = eqx.field(static=True)
    ledger_current_density_absolute_tolerance_a_m2: float = eqx.field(static=True)
    ledger_relative_tolerance: float = eqx.field(static=True)
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
        ledger_amount_absolute_tolerance_mol: float = 1.0e-10,
        ledger_charge_absolute_tolerance_c: float = 1.0e-5,
        ledger_molar_flux_absolute_tolerance_mol_m2_s: float = 1.0e-12,
        ledger_concentration_absolute_tolerance_mol_m3: float = 1.0e-9,
        ledger_rate_absolute_tolerance_mol_s: float = 1.0e-12,
        ledger_current_density_absolute_tolerance_a_m2: float = 1.0e-8,
        ledger_relative_tolerance: float = 1.0e-6,
    ):
        positive_count = (
            negative_shell_count if positive_shell_count is None else positive_shell_count
        )
        threshold = float(asymptotic_small_parameter_threshold)
        amount_atol = float(ledger_amount_absolute_tolerance_mol)
        charge_atol = float(ledger_charge_absolute_tolerance_c)
        flux_atol = float(ledger_molar_flux_absolute_tolerance_mol_m2_s)
        concentration_atol = float(ledger_concentration_absolute_tolerance_mol_m3)
        rate_atol = float(ledger_rate_absolute_tolerance_mol_s)
        current_atol = float(ledger_current_density_absolute_tolerance_a_m2)
        relative = float(ledger_relative_tolerance)
        tolerances = np.asarray(
            (
                threshold,
                amount_atol,
                charge_atol,
                flux_atol,
                concentration_atol,
                rate_atol,
                current_atol,
                relative,
            )
        )
        if (
            np.any(~np.isfinite(tolerances))
            or threshold <= 0.0
            or threshold >= 1.0
            or np.any(tolerances[1:] < 0.0)
        ):
            raise ValueError(
                "The asymptotic threshold must lie in (0, 1) and ledger "
                "tolerances must be finite and nonnegative."
            )
        spm_plan = PrescribedCurrentSpmPlan(
            negative_shell_count,
            positive_count,
            negative_reference_faces=negative_particle_reference_faces,
            positive_reference_faces=positive_particle_reference_faces,
            ledger_amount_absolute_tolerance_mol=amount_atol,
            ledger_charge_absolute_tolerance_c=charge_atol,
            ledger_relative_tolerance=relative,
        )
        negative = ThroughCellRegionPlan(
            negative_electrolyte_cell_count,
            region="negative",
            reference_faces=negative_electrolyte_reference_faces,
        )
        separator = ThroughCellRegionPlan(
            separator_electrolyte_cell_count,
            region="separator",
            reference_faces=separator_electrolyte_reference_faces,
        )
        positive = ThroughCellRegionPlan(
            positive_electrolyte_cell_count,
            region="positive",
            reference_faces=positive_electrolyte_reference_faces,
        )
        self.spm_plan = spm_plan
        self.negative_region = negative
        self.separator_region = separator
        self.positive_region = positive
        self.asymptotic_small_parameter_threshold = threshold
        self.ledger_amount_absolute_tolerance_mol = amount_atol
        self.ledger_charge_absolute_tolerance_c = charge_atol
        self.ledger_molar_flux_absolute_tolerance_mol_m2_s = flux_atol
        self.ledger_concentration_absolute_tolerance_mol_m3 = concentration_atol
        self.ledger_rate_absolute_tolerance_mol_s = rate_atol
        self.ledger_current_density_absolute_tolerance_a_m2 = current_atol
        self.ledger_relative_tolerance = relative
        self.plan_id = canonical_fingerprint(
            {
                "kind": "battery-spme-marquis-2019-plan",
                "spm_plan_id": spm_plan.plan_id,
                "regions": [negative.plan_id, separator.plan_id, positive.plan_id],
                "asymptotic_threshold": threshold,
                "ledger_amount_atol_mol": amount_atol,
                "ledger_charge_atol_c": charge_atol,
                "ledger_molar_flux_atol_mol_m2_s": flux_atol,
                "ledger_concentration_atol_mol_m3": concentration_atol,
                "ledger_rate_atol_mol_s": rate_atol,
                "ledger_current_atol_a_m2": current_atol,
                "ledger_rtol": relative,
            }
        )

    def prepare(self, /) -> "PreparedMarquis2019Spme":
        return PreparedMarquis2019Spme(self)


class PreparedMarquis2019Spme(StrictModule, NonTrainableState):
    """Prepared candidate topology; all physical values remain runtime leaves."""

    plan: Marquis2019SpmePlan
    spm: PreparedPrescribedCurrentSpm
    through_cell: PreparedThroughCellMesh
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: Marquis2019SpmePlan, /):
        if not isinstance(plan, Marquis2019SpmePlan):
            raise TypeError("plan must be Marquis2019SpmePlan.")
        spm = plan.spm_plan.prepare()
        through_cell = PreparedThroughCellMesh(
            plan.negative_region,
            plan.separator_region,
            plan.positive_region,
        )
        self.plan = plan
        self.spm = spm
        self.through_cell = through_cell
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-battery-spme-marquis-2019",
                "plan_id": plan.plan_id,
                "spm_prepared_id": spm.prepared_id,
                "through_cell_prepared_id": through_cell.prepared_id,
            }
        )

    def evaluate(
        self,
        state: Marquis2019SpmeState,
        parameters: Marquis2019SpmeParameters,
        terminal_current_a: ArrayLike,
        /,
    ) -> "_Marquis2019SpmeEvaluation":
        return _evaluate(self, state, parameters, jnp.asarray(terminal_current_a))


class _Marquis2019SpmeEvaluation(StrictModule):
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
    particle_ocp_v: Array
    reaction_overpotential_v: Array
    mean_concentration_overpotential_v: Array
    electrolyte_ohmic_loss_v: Array
    solid_ohmic_loss_v: Array
    voltage_v: Array
    passive_current_density_a_m2: Array
    paper_current_density_a_m2: Array
    total_solid_lithium_mol: Array
    total_electrolyte_lithium_mol: Array
    electrolyte_diffusivity_m2_s: Array
    electrolyte_conductivity_s_m: Array
    transference_number_value: Array
    electrolyte_migration_number: Array
    negative_thermal_to_solid_ohmic_ratio: Array
    positive_thermal_to_solid_ohmic_ratio: Array
    thermal_to_electrolyte_ohmic_ratio: Array
    negative_solid_diffusion_number: Array
    positive_solid_diffusion_number: Array
    negative_reaction_number: Array
    positive_reaction_number: Array
    negative_length_ratio: Array
    separator_length_ratio: Array
    positive_length_ratio: Array
    positive_to_negative_maximum_concentration_ratio: Array
    electrolyte_to_negative_maximum_concentration_ratio: Array
    eq49_electrolyte_error: Array
    eq49_negative_ocp_error: Array
    eq49_positive_ocp_error: Array
    eq49_error_estimate: Array
    eq49_applicable: Array
    asymptotic_conditions_satisfied: Array
    domain_valid: Array
    particle_transport: object
    electrolyte_transport: object


def _check_prepared(
    plan: Marquis2019SpmePlan,
    prepared_model: PreparedMarquis2019Spme,
    /,
) -> None:
    if not isinstance(prepared_model, PreparedMarquis2019Spme):
        raise TypeError("prepared_model must be PreparedMarquis2019Spme.")
    if prepared_model.plan.plan_id != plan.plan_id:
        raise ValueError(
            "Prepared Marquis SPMe topology does not belong to this adapter."
        )


def _region_mean(values: Array, weights: Array, mask: Array, /) -> Array:
    region_weights = weights * mask
    return jnp.sum(values * region_weights, axis=-1) / jnp.sum(region_weights)


def _ocp_second_derivative_concentration(
    law,
    stoichiometry: Array,
    maximum_concentration_mol_m3: Array,
    neighborhood_concentration_mol_m3: Array,
    /,
) -> tuple[Array, Array]:
    radius = neighborhood_concentration_mol_m3 / maximum_concentration_mol_m3
    lower = stoichiometry - radius
    upper = stoichiometry + radius
    endpoint_support = law.evaluate(lower).support & law.evaluate(upper).support
    if isinstance(law, ConstantPropertyLaw):
        return jnp.zeros_like(stoichiometry), endpoint_support
    if isinstance(law, TabulatedPropertyLaw):
        slopes = jnp.diff(law.values) / jnp.diff(law.nodes)
        slope_jumps = jnp.abs(jnp.diff(slopes))
        kink_nodes = law.nodes[1:-1]
        active_kinks = (
            law.source_mask[:-2]
            & law.source_mask[1:-1]
            & law.source_mask[2:]
            & (slope_jumps > 0.0)
        )
        crossed_kink = jnp.any(
            (radius[..., None] > 0.0)
            & (lower[..., None] <= kink_nodes)
            & (upper[..., None] >= kink_nodes)
            & active_kinks,
            axis=-1,
        )
        interval_active = law.source_mask[:-1] & law.source_mask[1:]
        overlapped_inactive_interval = jnp.any(
            (radius[..., None] > 0.0)
            & (lower[..., None] <= law.nodes[1:])
            & (upper[..., None] >= law.nodes[:-1])
            & ~interval_active,
            axis=-1,
        )
        neighborhood_valid = (
            endpoint_support & ~crossed_kink & ~overlapped_inactive_interval
        )
        return jnp.zeros_like(stoichiometry), neighborhood_valid

    def scalar_curvature(theta):
        return jax.grad(lambda value: law.evaluate(value, derivative_order=1).values)(
            theta
        )

    flat = stoichiometry.reshape((-1,))
    curvature_theta = jax.vmap(scalar_curvature)(flat).reshape(stoichiometry.shape)
    return (
        curvature_theta / maximum_concentration_mol_m3**2,
        endpoint_support,
    )


def _eq49_ocp_error(
    concentration_scale_c_m3: Array,
    ocp_curvature_v_m6_mol2: Array,
    thermal_energy_j_mol: Array,
    /,
) -> Array:
    # Eq. 49 uses the charge-concentration scale I*L/De [C/m3].
    # Equivalently delta_c = (I*L/De)/F [mol/m3] gives delta_c**2*|U''|*F/(RT).
    return (
        concentration_scale_c_m3**2
        * jnp.abs(ocp_curvature_v_m6_mol2)
        / (_FARADAY_C_MOL * thermal_energy_j_mol)
    )


def _finite_large_ratio(numerator: Array, denominator: Array, /) -> Array:
    dtype = jnp.result_type(numerator, denominator, float)
    maximum = jnp.asarray(jnp.finfo(dtype).max, dtype=dtype)
    floor = jnp.maximum(jnp.abs(numerator) / maximum, jnp.finfo(dtype).tiny)
    return numerator / jnp.maximum(denominator, floor)


def _evaluate(
    prepared: PreparedMarquis2019Spme,
    state: Marquis2019SpmeState,
    parameters: Marquis2019SpmeParameters,
    terminal_current_a: Array,
    /,
) -> _Marquis2019SpmeEvaluation:
    if not isinstance(state, Marquis2019SpmeState):
        raise TypeError("state must be Marquis2019SpmeState.")
    if not isinstance(parameters, Marquis2019SpmeParameters):
        raise TypeError("parameters must be Marquis2019SpmeParameters.")
    spm_parameters = parameters.spm_parameters
    particle_state = SpmState(
        state.negative_amount_mol,
        state.positive_amount_mol,
    )
    profile, particle_transport = _transport(
        prepared.spm,
        particle_state,
        spm_parameters,
        terminal_current_a,
    )
    leading_shape = state.negative_amount_mol.shape[:-1]
    current = jnp.asarray(terminal_current_a)
    if current.shape == ():
        current = jnp.broadcast_to(current, leading_shape)
    elif current.shape != leading_shape:
        raise ValueError("terminal_current_a must be scalar or match state leading axes.")

    typical_electrolyte = parameters.typical_electrolyte_concentration_mol_m3
    temperature = spm_parameters.temperature_k
    typical_diffusivity = parameters.electrolyte_diffusivity.evaluate(
        typical_electrolyte,
        temperature,
    )
    typical_conductivity = parameters.electrolyte_conductivity.evaluate(
        typical_electrolyte,
        temperature,
    )
    typical_transference = parameters.transference_number.evaluate(
        typical_electrolyte,
        temperature,
    )
    electrolyte_property_valid = (
        typical_diffusivity.support
        & jnp.isfinite(typical_diffusivity.values)
        & (typical_diffusivity.values > 0.0)
        & typical_conductivity.support
        & jnp.isfinite(typical_conductivity.values)
        & (typical_conductivity.values > 0.0)
        & typical_transference.support
        & jnp.isfinite(typical_transference.values)
        & (typical_transference.values >= 0.0)
        & (typical_transference.values <= 1.0)
    )
    electrolyte_diffusivity_values = typical_diffusivity.values
    electrolyte_conductivity_values = typical_conductivity.values
    transference_number_values = typical_transference.values
    electrolyte_transport = prepared.through_cell.evaluate(
        state.electrolyte_amount_mol,
        terminal_current_a,
        negative_thickness_m=spm_parameters.negative_electrode_thickness_m,
        separator_thickness_m=parameters.separator_thickness_m,
        positive_thickness_m=spm_parameters.positive_electrode_thickness_m,
        negative_porosity=parameters.negative_electrolyte_porosity,
        separator_porosity=parameters.separator_electrolyte_porosity,
        positive_porosity=parameters.positive_electrolyte_porosity,
        bruggeman_coefficient=parameters.bruggeman_coefficient,
        electrolyte_diffusivity_m2_s=electrolyte_diffusivity_values,
        electrode_area_m2=spm_parameters.electrode_area_m2,
        transference_number=transference_number_values,
        maximum_absolute_current_a=spm_parameters.maximum_absolute_current_a,
    )
    electrolyte_concentration = electrolyte_transport.concentration_mol_m3
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
    weights = prepared.through_cell.reference_cell_widths
    negative_electrolyte_mean = _region_mean(
        electrolyte_concentration,
        weights,
        prepared.through_cell.negative_mask,
    )
    separator_electrolyte_mean = _region_mean(
        electrolyte_concentration,
        weights,
        prepared.through_cell.separator_mask,
    )
    positive_electrolyte_mean = _region_mean(
        electrolyte_concentration,
        weights,
        prepared.through_cell.positive_mask,
    )

    negative_ocp = spm_parameters.negative_open_circuit_potential.evaluate(
        negative_surface_stoichiometry
    )
    positive_ocp = spm_parameters.positive_open_circuit_potential.evaluate(
        positive_surface_stoichiometry
    )
    particle_ocp = positive_ocp.values - negative_ocp.values

    negative_occupancy = negative_surface_stoichiometry * (
        1.0 - negative_surface_stoichiometry
    )
    positive_occupancy = positive_surface_stoichiometry * (
        1.0 - positive_surface_stoichiometry
    )
    typical_electrolyte = parameters.typical_electrolyte_concentration_mol_m3
    negative_electrolyte_factor = _region_mean(
        jnp.sqrt(jnp.maximum(electrolyte_concentration / typical_electrolyte, 0.0)),
        weights,
        prepared.through_cell.negative_mask,
    )
    positive_electrolyte_factor = _region_mean(
        jnp.sqrt(jnp.maximum(electrolyte_concentration / typical_electrolyte, 0.0)),
        weights,
        prepared.through_cell.positive_mask,
    )
    # Marquis uses j = j0 sinh(F eta / (2RT)), whereas the existing SPM uses
    # j = 2 i0 sinh(...). Thus the paper's j0 is exactly twice the SPM i0.
    negative_exchange = (
        4.0
        * profile.negative_exchange_current_density_scale_a_m2
        * jnp.sqrt(jnp.maximum(negative_occupancy, 0.0))
        * negative_electrolyte_factor
    )
    positive_exchange = (
        4.0
        * profile.positive_exchange_current_density_scale_a_m2
        * jnp.sqrt(jnp.maximum(positive_occupancy, 0.0))
        * positive_electrolyte_factor
    )
    negative_reaction_argument_denominator = (
        profile.negative_specific_surface_area_m2_m3
        * negative_exchange
        * spm_parameters.negative_electrode_thickness_m
    )
    positive_reaction_argument_denominator = (
        profile.positive_specific_surface_area_m2_m3
        * positive_exchange
        * spm_parameters.positive_electrode_thickness_m
    )
    thermal_voltage_twice = (
        2.0 * _GAS_CONSTANT_J_MOL_K * spm_parameters.temperature_k / _FARADAY_C_MOL
    )
    reaction_overpotential = thermal_voltage_twice * (
        _stable_asinh_ratio(
            passive_current_density,
            positive_reaction_argument_denominator,
        )
        + _stable_asinh_ratio(
            passive_current_density,
            negative_reaction_argument_denominator,
        )
    )
    mean_concentration_overpotential = (
        thermal_voltage_twice
        / typical_electrolyte
        * (1.0 - transference_number_values)
        * (positive_electrolyte_mean - negative_electrolyte_mean)
    )
    negative_effective_porosity = (
        parameters.negative_electrolyte_porosity**parameters.bruggeman_coefficient
    )
    separator_effective_porosity = (
        parameters.separator_electrolyte_porosity**parameters.bruggeman_coefficient
    )
    positive_effective_porosity = (
        parameters.positive_electrolyte_porosity**parameters.bruggeman_coefficient
    )
    electrolyte_ohmic_loss = (
        passive_current_density
        / electrolyte_conductivity_values
        * (
            spm_parameters.negative_electrode_thickness_m
            / (3.0 * negative_effective_porosity)
            + parameters.separator_thickness_m / separator_effective_porosity
            + spm_parameters.positive_electrode_thickness_m
            / (3.0 * positive_effective_porosity)
        )
    )
    solid_ohmic_loss = (
        passive_current_density
        / 3.0
        * (
            spm_parameters.positive_electrode_thickness_m
            / parameters.positive_solid_conductivity_s_m
            + spm_parameters.negative_electrode_thickness_m
            / parameters.negative_solid_conductivity_s_m
        )
    )
    voltage = (
        particle_ocp
        + reaction_overpotential
        + mean_concentration_overpotential
        + electrolyte_ohmic_loss
        + solid_ohmic_loss
    )

    total_length = (
        spm_parameters.negative_electrode_thickness_m
        + parameters.separator_thickness_m
        + spm_parameters.positive_electrode_thickness_m
    )
    absolute_current_density = jnp.abs(passive_current_density)
    negative_cmax = spm_parameters.negative_maximum_concentration_mol_m3
    positive_cmax = spm_parameters.positive_maximum_concentration_mol_m3
    minimum_electrolyte_diffusivity = electrolyte_diffusivity_values
    minimum_electrolyte_conductivity = electrolyte_conductivity_values
    electrolyte_migration_number = (
        absolute_current_density
        * total_length
        / (minimum_electrolyte_diffusivity * _FARADAY_C_MOL * negative_cmax)
    )
    thermal_energy_mol = _GAS_CONSTANT_J_MOL_K * spm_parameters.temperature_k
    ohmic_scale = _FARADAY_C_MOL * absolute_current_density * total_length
    negative_thermal_to_solid = _finite_large_ratio(
        thermal_energy_mol * parameters.negative_solid_conductivity_s_m,
        ohmic_scale,
    )
    positive_thermal_to_solid = _finite_large_ratio(
        thermal_energy_mol * parameters.positive_solid_conductivity_s_m,
        ohmic_scale,
    )
    thermal_to_electrolyte = _finite_large_ratio(
        thermal_energy_mol * minimum_electrolyte_conductivity,
        ohmic_scale,
    )
    negative_solid_diffusion = (
        spm_parameters.negative_particle_radius_m**2
        * absolute_current_density
        / (
            particle_transport.negative_minimum_diffusivity_m2_s
            * _FARADAY_C_MOL
            * negative_cmax
            * total_length
        )
    )
    positive_solid_diffusion = (
        spm_parameters.positive_particle_radius_m**2
        * absolute_current_density
        / (
            particle_transport.positive_minimum_diffusivity_m2_s
            * _FARADAY_C_MOL
            * negative_cmax
            * total_length
        )
    )
    negative_paper_rate_constant = (
        4.0
        * profile.negative_exchange_current_density_scale_a_m2
        / (negative_cmax * jnp.sqrt(typical_electrolyte))
    )
    positive_paper_rate_constant = (
        4.0
        * profile.positive_exchange_current_density_scale_a_m2
        / (positive_cmax * jnp.sqrt(typical_electrolyte))
    )
    negative_reaction_number = absolute_current_density / (
        negative_paper_rate_constant
        * profile.negative_specific_surface_area_m2_m3
        * jnp.sqrt(typical_electrolyte)
        * negative_cmax
        * total_length
    )
    positive_reaction_number = absolute_current_density / (
        positive_paper_rate_constant
        * profile.positive_specific_surface_area_m2_m3
        * jnp.sqrt(typical_electrolyte)
        * negative_cmax
        * total_length
    )
    negative_length_ratio = spm_parameters.negative_electrode_thickness_m / total_length
    separator_length_ratio = parameters.separator_thickness_m / total_length
    positive_length_ratio = spm_parameters.positive_electrode_thickness_m / total_length
    cmax_ratio = positive_cmax / negative_cmax
    electrolyte_ratio = typical_electrolyte / negative_cmax

    concentration_scale = (
        absolute_current_density * total_length / minimum_electrolyte_diffusivity
    )
    taylor_neighborhood_concentration = concentration_scale / _FARADAY_C_MOL
    (
        negative_ocp_curvature,
        negative_ocp_neighborhood_valid,
    ) = _ocp_second_derivative_concentration(
        spm_parameters.negative_open_circuit_potential,
        negative_surface_stoichiometry,
        negative_cmax,
        taylor_neighborhood_concentration,
    )
    (
        positive_ocp_curvature,
        positive_ocp_neighborhood_valid,
    ) = _ocp_second_derivative_concentration(
        spm_parameters.positive_open_circuit_potential,
        positive_surface_stoichiometry,
        positive_cmax,
        taylor_neighborhood_concentration,
    )
    eq49_electrolyte_error = electrolyte_migration_number**2
    eq49_negative_ocp_error = _eq49_ocp_error(
        concentration_scale,
        negative_ocp_curvature,
        thermal_energy_mol,
    )
    eq49_positive_ocp_error = _eq49_ocp_error(
        concentration_scale,
        positive_ocp_curvature,
        thermal_energy_mol,
    )
    eq49_applicable = negative_ocp_neighborhood_valid & positive_ocp_neighborhood_valid
    eq49_error_estimate = jnp.maximum(
        eq49_electrolyte_error,
        jnp.maximum(eq49_negative_ocp_error, eq49_positive_ocp_error),
    )

    threshold = prepared.plan.asymptotic_small_parameter_threshold
    inverse_threshold = 1.0 / threshold
    ce = electrolyte_migration_number
    table_core = (
        (ce <= threshold)
        & (negative_thermal_to_solid >= inverse_threshold)
        & (positive_thermal_to_solid >= inverse_threshold)
        & (thermal_to_electrolyte >= inverse_threshold)
        & (ce * negative_solid_diffusion <= threshold)
        & (ce * positive_solid_diffusion <= threshold)
        & (ce * negative_reaction_number <= threshold)
        & (ce * positive_reaction_number <= threshold)
    )

    def separated_from_bounds(ratio):
        return (ce <= threshold * ratio) & (ce * ratio <= threshold)

    asymptotic_conditions = (
        table_core
        & separated_from_bounds(negative_length_ratio)
        & separated_from_bounds(separator_length_ratio)
        & separated_from_bounds(positive_length_ratio)
        & separated_from_bounds(cmax_ratio)
        & separated_from_bounds(electrolyte_ratio)
        & eq49_applicable
    )
    active_current = safe_current != 0.0
    exchange_valid = (
        jnp.isfinite(negative_exchange)
        & jnp.isfinite(positive_exchange)
        & (negative_exchange >= 0.0)
        & (positive_exchange >= 0.0)
        & (~active_current | ((negative_exchange > 0.0) & (positive_exchange > 0.0)))
    )
    concentration_valid = (
        jnp.all(
            particle_transport.negative.concentration_mol_m3 <= negative_cmax,
            axis=-1,
        )
        & (negative_surface_concentration <= negative_cmax)
        & jnp.all(
            particle_transport.positive.concentration_mol_m3 <= positive_cmax,
            axis=-1,
        )
        & (positive_surface_concentration <= positive_cmax)
    )
    evidence_values = jnp.stack(
        (
            particle_ocp,
            reaction_overpotential,
            mean_concentration_overpotential,
            electrolyte_ohmic_loss,
            solid_ohmic_loss,
            voltage,
            negative_exchange,
            positive_exchange,
            electrolyte_migration_number,
            negative_thermal_to_solid,
            positive_thermal_to_solid,
            thermal_to_electrolyte,
            negative_solid_diffusion,
            positive_solid_diffusion,
            negative_reaction_number,
            positive_reaction_number,
            eq49_electrolyte_error,
            eq49_negative_ocp_error,
            eq49_positive_ocp_error,
            eq49_error_estimate,
        ),
        axis=-1,
    )
    domain_valid = (
        particle_transport.domain_valid
        & electrolyte_transport.domain_valid
        & electrolyte_property_valid
        & typical_diffusivity.support
        & negative_ocp.support
        & positive_ocp.support
        & jnp.isfinite(negative_ocp.values)
        & jnp.isfinite(positive_ocp.values)
        & exchange_valid
        & concentration_valid
        & jnp.all(jnp.isfinite(evidence_values), axis=-1)
    )
    return _Marquis2019SpmeEvaluation(
        negative_surface_stoichiometry,
        positive_surface_stoichiometry,
        negative_surface_concentration,
        positive_surface_concentration,
        particle_transport.negative.average_concentration_mol_m3,
        particle_transport.positive.average_concentration_mol_m3,
        negative_electrolyte_mean,
        separator_electrolyte_mean,
        positive_electrolyte_mean,
        negative_exchange,
        positive_exchange,
        particle_ocp,
        reaction_overpotential,
        mean_concentration_overpotential,
        electrolyte_ohmic_loss,
        solid_ohmic_loss,
        voltage,
        passive_current_density,
        paper_current_density,
        particle_transport.negative.total_amount_mol
        + particle_transport.positive.total_amount_mol,
        electrolyte_transport.total_amount_mol,
        electrolyte_diffusivity_values,
        electrolyte_conductivity_values,
        transference_number_values,
        electrolyte_migration_number,
        negative_thermal_to_solid,
        positive_thermal_to_solid,
        thermal_to_electrolyte,
        negative_solid_diffusion,
        positive_solid_diffusion,
        negative_reaction_number,
        positive_reaction_number,
        negative_length_ratio,
        separator_length_ratio,
        positive_length_ratio,
        cmax_ratio,
        electrolyte_ratio,
        eq49_electrolyte_error,
        eq49_negative_ocp_error,
        eq49_positive_ocp_error,
        eq49_error_estimate,
        eq49_applicable,
        asymptotic_conditions,
        domain_valid,
        particle_transport,
        electrolyte_transport,
    )


class _Marquis2019SpmeVectorField(StrictModule, NonTrainableState):
    prepared: PreparedMarquis2019Spme

    def __call__(
        self,
        time_s: Array,
        state: Marquis2019SpmeState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> Marquis2019SpmeState:
        parameters = runtime_inputs.parameters
        if not isinstance(parameters, Marquis2019SpmeParameters):
            raise TypeError("Marquis SPMe runtime parameters have the wrong type.")
        current = runtime_inputs.current(time_s, state.negative_amount_mol)
        evaluation = _evaluate(self.prepared, state, parameters, current)
        return Marquis2019SpmeState(
            evaluation.particle_transport.negative.amount_rate_mol_s,
            evaluation.particle_transport.positive.amount_rate_mol_s,
            evaluation.electrolyte_transport.amount_rate_mol_s,
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
    "voltage:particle_ocp_v",
    "voltage:reaction_overpotential_v",
    "voltage:mean_concentration_overpotential_v",
    "voltage:electrolyte_ohmic_loss_v",
    "voltage:solid_ohmic_loss_v",
    "negative_exchange_current_density_a_m2",
    "positive_exchange_current_density_a_m2",
    "passive_current_density_a_m2",
    "paper_current_density_a_m2",
    "electrolyte_current_at_negative_separator_a_m2",
    "electrolyte_current_at_separator_positive_a_m2",
    "current_split_residual_a_m2",
    "collector_molar_flux_residual_mol_m2_s",
    "negative_separator_concentration_jump_mol_m3",
    "separator_positive_concentration_jump_mol_m3",
    "electrolyte_conservation_rate_residual_mol_s",
    "equation_mapping_residual_mol_s",
    "electrolyte_explicit_dt_limit_s",
    "table6:electrolyte_migration_number",
    "table6:negative_thermal_to_solid_ohmic_ratio",
    "table6:positive_thermal_to_solid_ohmic_ratio",
    "table6:thermal_to_electrolyte_ohmic_ratio",
    "table6:negative_solid_diffusion_number",
    "table6:positive_solid_diffusion_number",
    "table6:negative_reaction_number",
    "table6:positive_reaction_number",
    "table6:negative_length_ratio",
    "table6:separator_length_ratio",
    "table6:positive_length_ratio",
    "table6:positive_to_negative_maximum_concentration_ratio",
    "table6:electrolyte_to_negative_maximum_concentration_ratio",
    "table6:conditions_satisfied",
    "eq49:applicable",
    "eq49:electrolyte_error",
    "eq49:negative_ocp_error",
    "eq49:positive_ocp_error",
    "eq49:error_estimate",
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
    "V",
    "V",
    "V",
    "V",
    "V",
    "A/m2",
    "A/m2",
    "A/m2",
    "A/m2",
    "A/m2",
    "A/m2",
    "A/m2",
    "mol/(m2*s)",
    "mol/m3",
    "mol/m3",
    "mol/s",
    "mol/s",
    "s",
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
    "1",
    "1",
    "W",
    "A",
)


class Marquis2019SpmeAdapter(StrictModule, NonTrainableState):
    """ODE adapter for the dimensional canonical Marquis et al. 2019 Eq. 48 SPMe."""

    plan: Marquis2019SpmePlan
    model_id: str = eqx.field(static=True)
    equation_form: Literal["ode"] = eqx.field(static=True)
    observable_names: tuple[str, ...] = eqx.field(static=True)
    observable_units: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, plan: Marquis2019SpmePlan, /):
        if not isinstance(plan, Marquis2019SpmePlan):
            raise TypeError("plan must be Marquis2019SpmePlan.")
        self.plan = plan
        self.model_id = "battery:spme:marquis-2019:isothermal-prescribed-current"
        self.equation_form = "ode"
        self.observable_names = _OBSERVABLE_NAMES
        self.observable_units = _OBSERVABLE_UNITS

    def prepare(self, /) -> PreparedMarquis2019Spme:
        return self.plan.prepare()

    def initial_state(
        self,
        prepared_model: PreparedMarquis2019Spme,
        parameters: Marquis2019SpmeParameters,
        initial_condition: Marquis2019SpmeInitialCondition,
        /,
    ) -> Marquis2019SpmeState:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(parameters, Marquis2019SpmeParameters):
            raise TypeError("parameters must be Marquis2019SpmeParameters.")
        if not isinstance(initial_condition, Marquis2019SpmeInitialCondition):
            raise TypeError("initial_condition must be Marquis2019SpmeInitialCondition.")
        spm_initial = PrescribedCurrentSpmAdapter(self.plan.spm_plan).initial_state(
            prepared_model.spm,
            parameters.spm_parameters,
            SpmInitialCondition(
                initial_condition.negative_stoichiometry,
                initial_condition.positive_stoichiometry,
            ),
        )
        spm_parameters = parameters.spm_parameters
        typical_diffusivity = parameters.electrolyte_diffusivity.evaluate(
            parameters.typical_electrolyte_concentration_mol_m3,
            spm_parameters.temperature_k,
        )
        electrolyte = prepared_model.through_cell.initial_amounts(
            parameters.typical_electrolyte_concentration_mol_m3,
            negative_thickness_m=spm_parameters.negative_electrode_thickness_m,
            separator_thickness_m=parameters.separator_thickness_m,
            positive_thickness_m=spm_parameters.positive_electrode_thickness_m,
            negative_porosity=parameters.negative_electrolyte_porosity,
            separator_porosity=parameters.separator_electrolyte_porosity,
            positive_porosity=parameters.positive_electrolyte_porosity,
            bruggeman_coefficient=parameters.bruggeman_coefficient,
            electrolyte_diffusivity_m2_s=typical_diffusivity.values,
            electrode_area_m2=spm_parameters.electrode_area_m2,
        )
        return Marquis2019SpmeState(
            spm_initial.negative_amount_mol,
            spm_initial.positive_amount_mol,
            electrolyte,
        )

    def problem(
        self,
        prepared_model: PreparedMarquis2019Spme,
        initial_state: Marquis2019SpmeState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> DifferentialProblem:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(initial_state, Marquis2019SpmeState):
            raise TypeError("initial_state must be Marquis2019SpmeState.")
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        if not isinstance(runtime_inputs.parameters, Marquis2019SpmeParameters):
            raise TypeError("Marquis SPMe runtime parameters have the wrong type.")
        expected_negative = prepared_model.spm.negative_particle.shell_count
        expected_positive = prepared_model.spm.positive_particle.shell_count
        expected_electrolyte = prepared_model.through_cell.cell_count
        if initial_state.negative_amount_mol.shape != (expected_negative,):
            raise ValueError(
                f"Initial negative SPMe state must have shape ({expected_negative},)."
            )
        if initial_state.positive_amount_mol.shape != (expected_positive,):
            raise ValueError(
                f"Initial positive SPMe state must have shape ({expected_positive},)."
            )
        if initial_state.electrolyte_amount_mol.shape != (expected_electrolyte,):
            raise ValueError(
                f"Initial electrolyte SPMe state must have shape ({expected_electrolyte},)."
            )
        return DifferentialProblem(
            _Marquis2019SpmeVectorField(prepared_model),
            initial_state,
            t0=runtime_inputs.input_policy.times[0],
            t1=runtime_inputs.input_policy.times[-1],
            args=runtime_inputs,
            problem_id=canonical_fingerprint(
                {
                    "kind": "battery-spme-marquis-2019-problem",
                    "model_id": self.model_id,
                    "prepared_id": prepared_model.prepared_id,
                    "parameter_id": runtime_inputs.parameters.parameter_id,
                    "protocol_id": runtime_inputs.protocol_id,
                }
            ),
        )

    def observe(
        self,
        prepared_model: PreparedMarquis2019Spme,
        times_s: Array,
        states: Marquis2019SpmeState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> BatteryModelOutput:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(states, Marquis2019SpmeState):
            raise TypeError("states must be Marquis2019SpmeState.")
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        if not isinstance(runtime_inputs.parameters, Marquis2019SpmeParameters):
            raise TypeError("Marquis SPMe runtime parameters have the wrong type.")
        times = jnp.asarray(times_s)
        expected_negative = times.shape + (
            prepared_model.spm.negative_particle.shell_count,
        )
        expected_positive = times.shape + (
            prepared_model.spm.positive_particle.shell_count,
        )
        expected_electrolyte = times.shape + (prepared_model.through_cell.cell_count,)
        if states.negative_amount_mol.shape != expected_negative:
            raise ValueError(
                f"Negative SPMe observation state must have shape {expected_negative}."
            )
        if states.positive_amount_mol.shape != expected_positive:
            raise ValueError(
                f"Positive SPMe observation state must have shape {expected_positive}."
            )
        if states.electrolyte_amount_mol.shape != expected_electrolyte:
            raise ValueError(
                f"Electrolyte SPMe observation state must have shape {expected_electrolyte}."
            )
        flat_current = jax.vmap(lambda time: runtime_inputs.observed_current(time))(
            times.reshape((-1,))
        )
        current = flat_current.reshape(times.shape)
        parameters = runtime_inputs.parameters
        evaluation = _evaluate(prepared_model, states, parameters, current)
        electrolyte = evaluation.electrolyte_transport
        ns_face = prepared_model.through_cell.negative_separator_face_index
        sp_face = prepared_model.through_cell.separator_positive_face_index
        total_lithium = (
            evaluation.total_solid_lithium_mol + evaluation.total_electrolyte_lithium_mol
        )
        values = jnp.stack(
            (
                evaluation.voltage_v,
                jnp.zeros_like(evaluation.voltage_v)
                + parameters.spm_parameters.temperature_k,
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
                evaluation.particle_ocp_v,
                evaluation.reaction_overpotential_v,
                evaluation.mean_concentration_overpotential_v,
                evaluation.electrolyte_ohmic_loss_v,
                evaluation.solid_ohmic_loss_v,
                evaluation.negative_exchange_current_density_a_m2,
                evaluation.positive_exchange_current_density_a_m2,
                evaluation.passive_current_density_a_m2,
                evaluation.paper_current_density_a_m2,
                electrolyte.electrolyte_current_density_a_m2[..., ns_face],
                electrolyte.electrolyte_current_density_a_m2[..., sp_face],
                electrolyte.current_split_residual_a_m2,
                electrolyte.collector_flux_residual_mol_m2_s,
                electrolyte.negative_separator_concentration_jump_mol_m3,
                electrolyte.separator_positive_concentration_jump_mol_m3,
                electrolyte.conservation_residual_mol_s,
                electrolyte.equation_mapping_residual_mol_s,
                jnp.zeros_like(evaluation.voltage_v) + electrolyte.explicit_dt_limit_s,
                evaluation.electrolyte_migration_number,
                evaluation.negative_thermal_to_solid_ohmic_ratio,
                evaluation.positive_thermal_to_solid_ohmic_ratio,
                evaluation.thermal_to_electrolyte_ohmic_ratio,
                evaluation.negative_solid_diffusion_number,
                evaluation.positive_solid_diffusion_number,
                evaluation.negative_reaction_number,
                evaluation.positive_reaction_number,
                jnp.zeros_like(evaluation.voltage_v) + evaluation.negative_length_ratio,
                jnp.zeros_like(evaluation.voltage_v) + evaluation.separator_length_ratio,
                jnp.zeros_like(evaluation.voltage_v) + evaluation.positive_length_ratio,
                jnp.zeros_like(evaluation.voltage_v)
                + evaluation.positive_to_negative_maximum_concentration_ratio,
                jnp.zeros_like(evaluation.voltage_v)
                + evaluation.electrolyte_to_negative_maximum_concentration_ratio,
                evaluation.asymptotic_conditions_satisfied.astype(
                    evaluation.voltage_v.dtype
                ),
                evaluation.eq49_applicable.astype(evaluation.voltage_v.dtype),
                evaluation.eq49_electrolyte_error,
                evaluation.eq49_negative_ocp_error,
                evaluation.eq49_positive_ocp_error,
                evaluation.eq49_error_estimate,
                evaluation.voltage_v * current,
                parameters.spm_parameters.maximum_absolute_current_a - jnp.abs(current),
            ),
            axis=-1,
        )
        domain_valid = evaluation.domain_valid & jnp.all(jnp.isfinite(values), axis=-1)
        return BatteryModelOutput(values, domain_valid)

    def ledger(
        self,
        prepared_model: PreparedMarquis2019Spme,
        native_solution,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> Marquis2019SpmeLedger:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        if not isinstance(runtime_inputs.parameters, Marquis2019SpmeParameters):
            raise TypeError("Marquis SPMe runtime parameters have the wrong type.")
        states = native_solution.states
        if not isinstance(states, Marquis2019SpmeState):
            raise TypeError("Marquis SPMe native solution states have the wrong type.")
        valid = jnp.asarray(native_solution.valid, dtype=bool)
        valid_count = jnp.sum(valid.astype(jnp.int32))
        final_index = jnp.maximum(valid_count - 1, 0)

        initial_negative = jnp.sum(states.negative_amount_mol[0])
        final_negative = jnp.sum(states.negative_amount_mol[final_index])
        initial_positive = jnp.sum(states.positive_amount_mol[0])
        final_positive = jnp.sum(states.positive_amount_mol[final_index])
        initial_solid = initial_negative + initial_positive
        final_solid = final_negative + final_positive
        initial_electrolyte = jnp.sum(states.electrolyte_amount_mol[0])
        final_electrolyte = jnp.sum(states.electrolyte_amount_mol[final_index])
        initial_total = initial_solid + initial_electrolyte
        final_total = final_solid + final_electrolyte
        solid_residual = final_solid - initial_solid
        electrolyte_residual = final_electrolyte - initial_electrolyte
        total_residual = final_total - initial_total
        charge_residual = _FARADAY_C_MOL * total_residual

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

        times = native_solution.times
        currents = jax.vmap(runtime_inputs.observed_current)(times)
        evaluation = _evaluate(
            prepared_model, states, runtime_inputs.parameters, currents
        )
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
        maximum_error_estimate = jnp.max(
            jnp.where(valid, evaluation.eq49_error_estimate, 0.0)
        )
        eq49_applicable = (valid_count > 0) & jnp.all(
            jnp.where(valid, evaluation.eq49_applicable, True)
        )
        domain_valid = (valid_count > 0) & jnp.all(
            jnp.where(valid, evaluation.domain_valid, True)
        )
        asymptotic_conditions = (valid_count > 0) & jnp.all(
            jnp.where(valid, evaluation.asymptotic_conditions_satisfied, True)
        )
        finite = (valid_count > 0) & jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        initial_negative,
                        final_negative,
                        initial_positive,
                        final_positive,
                        initial_electrolyte,
                        final_electrolyte,
                        solid_residual,
                        electrolyte_residual,
                        total_residual,
                        charge_residual,
                        integrated_charge,
                        negative_current_residual,
                        positive_current_residual,
                        maximum_collector_flux,
                        maximum_interface_jump,
                        maximum_mapping_residual,
                        maximum_current_split,
                        maximum_error_estimate,
                    )
                )
            )
        )
        amount_scale = jnp.maximum(jnp.abs(initial_total), jnp.abs(final_total))
        solid_scale = jnp.maximum(jnp.abs(initial_solid), jnp.abs(final_solid))
        electrolyte_scale = jnp.maximum(
            jnp.abs(initial_electrolyte), jnp.abs(final_electrolyte)
        )
        amount_atol = prepared_model.plan.ledger_amount_absolute_tolerance_mol
        relative = prepared_model.plan.ledger_relative_tolerance
        solid_conserved = jnp.abs(solid_residual) <= amount_atol + relative * solid_scale
        electrolyte_conserved = (
            jnp.abs(electrolyte_residual) <= amount_atol + relative * electrolyte_scale
        )
        total_conserved = jnp.abs(total_residual) <= amount_atol + relative * amount_scale
        charge_scale = jnp.abs(integrated_charge)
        current_conserved = jnp.maximum(
            jnp.abs(negative_current_residual),
            jnp.abs(positive_current_residual),
        ) <= (
            prepared_model.plan.ledger_charge_absolute_tolerance_c
            + relative * charge_scale
        )
        concentration_scale = (
            runtime_inputs.parameters.typical_electrolyte_concentration_mol_m3
        )
        interfaces_conserved = (
            (
                maximum_collector_flux
                <= prepared_model.plan.ledger_molar_flux_absolute_tolerance_mol_m2_s
            )
            & (
                maximum_interface_jump
                <= prepared_model.plan.ledger_concentration_absolute_tolerance_mol_m3
                + relative * concentration_scale
            )
            & (
                maximum_mapping_residual
                <= prepared_model.plan.ledger_rate_absolute_tolerance_mol_s
            )
        )
        current_split_conserved = (
            maximum_current_split
            <= prepared_model.plan.ledger_current_density_absolute_tolerance_a_m2
        )
        successful = (
            finite
            & domain_valid
            & solid_conserved
            & electrolyte_conserved
            & total_conserved
            & current_conserved
            & interfaces_conserved
            & current_split_conserved
            & eq49_applicable
            & asymptotic_conditions
        )
        return Marquis2019SpmeLedger(
            initial_negative,
            final_negative,
            initial_positive,
            final_positive,
            initial_solid,
            final_solid,
            initial_electrolyte,
            final_electrolyte,
            initial_total,
            final_total,
            solid_residual,
            electrolyte_residual,
            total_residual,
            charge_residual,
            integrated_charge,
            negative_current_residual,
            positive_current_residual,
            maximum_collector_flux,
            maximum_interface_jump,
            maximum_mapping_residual,
            maximum_current_split,
            maximum_error_estimate,
            eq49_applicable,
            solid_conserved,
            electrolyte_conserved,
            total_conserved,
            current_conserved,
            interfaces_conserved,
            current_split_conserved,
            asymptotic_conditions,
            domain_valid,
            finite,
            successful,
        )


def validate_marquis2019_spme_execution(
    prepared_model: PreparedMarquis2019Spme,
    parameters: Marquis2019SpmeParameters,
    initial_condition: Marquis2019SpmeInitialCondition,
    protocol,
    protocol_values,
    /,
) -> None:
    """Concrete model preflight, using Eq.49/Table6 and native property support.

    Check every prescribed amplitude before dispatch, not by detecting a future
    crossing. Inventory feasibility is exact at every current/rest boundary.
    State-dependent surface/property validity remains checked by native outputs
    and the model ledger along the actual trajectory; preflight is not a solve.
    Precision and resource/mesh bounds belong to the authenticated envelope.
    """
    from ._protocol import BatteryProtocolPlan, BatteryProtocolValues

    if not isinstance(prepared_model, PreparedMarquis2019Spme):
        raise TypeError("SPMe preflight requires its prepared native topology.")
    if not isinstance(parameters, Marquis2019SpmeParameters):
        raise TypeError("SPMe preflight requires Marquis2019SpmeParameters.")
    if not isinstance(initial_condition, Marquis2019SpmeInitialCondition):
        raise TypeError("SPMe preflight requires inventory initial conditions.")
    if not isinstance(protocol, BatteryProtocolPlan) or not isinstance(
        protocol_values, BatteryProtocolValues
    ):
        raise TypeError("SPMe preflight requires typed current/rest values.")
    currents = np.asarray(protocol.interval_currents(protocol_values), dtype=float)
    spm = parameters.spm_parameters
    if not np.all(np.isfinite(currents)) or np.any(
        np.abs(currents) > float(spm.maximum_absolute_current_a)
    ):
        raise ValueError("The complete SPMe current schedule exceeds parameter support.")
    state = Marquis2019SpmeAdapter(prepared_model.plan).initial_state(
        prepared_model, parameters, initial_condition
    )
    for current in np.unique(np.r_[0.0, currents]):
        evaluation = prepared_model.evaluate(state, parameters, current)
        if not bool(evaluation.domain_valid):
            raise ValueError(
                "SPMe initial surface concentrations or constitutive properties do not support a prescribed current."
            )
        if not bool(evaluation.eq49_applicable) or not bool(
            evaluation.asymptotic_conditions_satisfied
        ):
            raise ValueError(
                "SPMe current/parameter data do not satisfy canonical Eq.49/Table6 applicability."
            )
    charge_at_boundary = np.r_[
        0.0,
        np.cumsum(currents * np.diff(np.asarray(protocol.boundary_times_s, dtype=float))),
    ]
    negative_inventory = (
        float(jnp.sum(state.negative_amount_mol)) + charge_at_boundary / _FARADAY_C_MOL
    )
    positive_inventory = (
        float(jnp.sum(state.positive_amount_mol)) - charge_at_boundary / _FARADAY_C_MOL
    )
    negative_capacity = float(
        spm.electrode_area_m2
        * spm.negative_electrode_thickness_m
        * spm.negative_active_material_volume_fraction
        * spm.negative_maximum_concentration_mol_m3
    )
    positive_capacity = float(
        spm.electrode_area_m2
        * spm.positive_electrode_thickness_m
        * spm.positive_active_material_volume_fraction
        * spm.positive_maximum_concentration_mol_m3
    )
    if np.any(
        (negative_inventory <= 0) | (negative_inventory >= negative_capacity)
    ) or np.any((positive_inventory <= 0) | (positive_inventory >= positive_capacity)):
        raise ValueError(
            "SPMe prescribed charge/rest schedule exceeds available electrode lithium inventory."
        )


__all__ = [
    "Marquis2019SpmeAdapter",
    "Marquis2019SpmeInitialCondition",
    "Marquis2019SpmeLedger",
    "Marquis2019SpmeParameters",
    "Marquis2019SpmePlan",
    "Marquis2019SpmeState",
    "PreparedMarquis2019Spme",
]
