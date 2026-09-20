#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-quadrature quasiparticle and off-shell dark-sector transport state."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..cosmology._dark_sector_species import DarkSectorSpeciesPlan
from ..cosmology._quantum_dark_kinetics import QuantumKineticState
from ..relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


class OffShellTransportProfile(StrictModule, NonTrainableState):
    support: tuple[str, ...] = eqx.field(static=True)
    refusals: tuple[str, ...] = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)
    rights_id: str = eqx.field(static=True)
    provenance_ids: tuple[str, ...] = eqx.field(static=True)
    production_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    production_ready: bool = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(self, production_evidence_ids: Sequence[str] = (), /):
        evidence = tuple(
            _identifier(value, "production_evidence_id")
            for value in production_evidence_ids
        )
        if len(set(evidence)) != len(evidence):
            raise ValueError("Production evidence identities must be unique.")
        support = (
            "scalar-quasiparticle-retarded-propagator",
            "finite-frequency-spectral-quadrature",
            "first-gradient-compatible-wigner-state",
        )
        refusals = (
            "negative-width",
            "negative-spectral-density",
            "implicit-energy-or-frame-conversion",
            "spectral-clipping-or-silent-sum-rule-renormalization",
        )
        differentiation = "algorithmic-fixed-quadrature-no-positivity-projection"
        rights_id = "phydra-native-no-external-provider"
        provenance_ids = ("retarded-dyson-breit-wigner-transport",)
        self.support = support
        self.refusals = refusals
        self.differentiation = differentiation
        self.rights_id = rights_id
        self.provenance_ids = provenance_ids
        self.production_evidence_ids = evidence
        self.production_ready = bool(evidence)
        self.profile_id = canonical_fingerprint(
            {
                "kind": "off-shell-dark-transport-profile",
                "support": support,
                "refusals": refusals,
                "differentiation": differentiation,
                "rights_id": rights_id,
                "provenance_ids": provenance_ids,
                "production_evidence_ids": evidence,
            }
        )


class OffShellTransportPlan(StrictModule, NonTrainableState):
    quantum_support: QuantumKineticState
    species: tuple[DarkSectorSpeciesPlan, ...]
    energy_nodes: Array
    energy_weights: Array
    momentum_weights: Array
    cell_volumes: Array
    statistics: Array
    units: RelativisticUnitContract
    frame: LocalRelativisticFramePlan
    profile: OffShellTransportProfile
    species_plan_ids: tuple[str, ...] = eqx.field(static=True)
    charge_names: tuple[str, ...] = eqx.field(static=True)
    energy_quadrature_id: str = eqx.field(static=True)
    momentum_quadrature_id: str = eqx.field(static=True)
    spectral_tolerance: float = eqx.field(static=True)
    dyson_tolerance: float = eqx.field(static=True)
    kms_tolerance: float = eqx.field(static=True)
    maximum_spectral_bytes: int = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quantum_support: QuantumKineticState,
        energy_nodes: ArrayLike,
        energy_weights: ArrayLike,
        momentum_weights: ArrayLike,
        cell_volumes: ArrayLike,
        /,
        *,
        energy_quadrature_id: str,
        momentum_quadrature_id: str,
        spectral_tolerance: float = 5.0e-3,
        dyson_tolerance: float = 1.0e-10,
        kms_tolerance: float = 1.0e-9,
        maximum_spectral_bytes: int = 512_000_000,
        production_evidence_ids: Sequence[str] = (),
    ):
        if not isinstance(quantum_support, QuantumKineticState):
            raise TypeError("quantum_support must be a QuantumKineticState.")
        species = tuple(quantum_support.species)
        if not species or any(
            not isinstance(value, DarkSectorSpeciesPlan) for value in species
        ):
            raise TypeError("Quantum support species must use DarkSectorSpeciesPlan.")
        shape = tuple(quantum_support.occupancy.shape)
        if len(shape) != 3 or shape[0] != len(species):
            raise ValueError(
                "Quantum occupancy must have shape (species, cell, momentum)."
            )
        energies = np.asarray(energy_nodes, dtype=np.float64)
        energy_measure = np.asarray(energy_weights, dtype=np.float64)
        momentum_measure = np.asarray(momentum_weights, dtype=np.float64)
        cells = np.asarray(cell_volumes, dtype=np.float64)
        if (
            energies.ndim != 1
            or energies.size < 3
            or np.any(~np.isfinite(energies))
            or np.any(np.diff(energies) <= 0.0)
            or energy_measure.shape != energies.shape
            or np.any(~np.isfinite(energy_measure))
            or np.any(energy_measure <= 0.0)
        ):
            raise ValueError(
                "Energy nodes must be strictly increasing and have positive finite weights."
            )
        if (
            momentum_measure.shape != (shape[2],)
            or np.any(~np.isfinite(momentum_measure))
            or np.any(momentum_measure <= 0.0)
            or cells.shape != (shape[1],)
            or np.any(~np.isfinite(cells))
            or np.any(cells <= 0.0)
        ):
            raise ValueError(
                "Momentum weights and cell volumes must be positive and support-aligned."
            )
        statistics = np.asarray(quantum_support.statistics, dtype=np.int8)
        if statistics.shape != (len(species),) or np.any(
            ~np.isin(statistics, (-1, 0, 1))
        ):
            raise ValueError(
                "Quantum statistics must contain one -1, 0, or +1 mark per species."
            )
        spectral_tol = float(spectral_tolerance)
        dyson_tol = float(dyson_tolerance)
        kms_tol = float(kms_tolerance)
        maximum = int(maximum_spectral_bytes)
        if (
            not np.isfinite(spectral_tol)
            or spectral_tol < 0.0
            or not np.isfinite(dyson_tol)
            or dyson_tol < 0.0
            or not np.isfinite(kms_tol)
            or kms_tol < 0.0
            or maximum < 1
        ):
            raise ValueError("Off-shell tolerances and resource capacity are invalid.")
        required = shape[1] * shape[2] * shape[0] * energies.size * 8 * 8
        if required > maximum:
            raise ValueError(
                f"Off-shell state requires {required} bytes; capacity is {maximum}."
            )
        charge_names = species[0].charge_names
        if any(value.charge_names != charge_names for value in species):
            raise ValueError(
                "All off-shell species must use the same ordered charge names."
            )
        units = quantum_support.units
        frame = quantum_support.frame
        if not isinstance(units, RelativisticUnitContract) or not isinstance(
            frame, LocalRelativisticFramePlan
        ):
            raise TypeError("Quantum support must own Foundation unit/frame contracts.")
        if frame.units.contract_id != units.contract_id:
            raise ValueError(
                "Quantum support frame and unit contract identities disagree."
            )
        expected_mass_unit = units.scale.dimensional_scale.mass_unit.unit_id
        expected_energy_unit = units.energy_unit.unit_id
        if any(
            value.mass_unit != expected_mass_unit
            or value.energy_unit != expected_energy_unit
            for value in species
        ):
            raise ValueError(
                "Off-shell species must bind the exact relativistic mass and energy units."
            )
        frame_realization = frame.realization_id()
        profile = OffShellTransportProfile(production_evidence_ids)
        self.quantum_support = quantum_support
        self.species = species
        self.energy_nodes = jnp.asarray(energies)
        self.energy_weights = jnp.asarray(energy_measure)
        self.momentum_weights = jnp.asarray(momentum_measure)
        self.cell_volumes = jnp.asarray(cells)
        self.statistics = jnp.asarray(statistics)
        self.units = units
        self.frame = frame
        self.profile = profile
        self.species_plan_ids = tuple(value.species_plan_id for value in species)
        self.charge_names = charge_names
        self.energy_quadrature_id = _identifier(
            energy_quadrature_id, "energy_quadrature_id"
        )
        self.momentum_quadrature_id = _identifier(
            momentum_quadrature_id, "momentum_quadrature_id"
        )
        self.spectral_tolerance = spectral_tol
        self.dyson_tolerance = dyson_tol
        self.kms_tolerance = kms_tol
        self.maximum_spectral_bytes = maximum
        self.support_id = quantum_support.support_id
        self.frame_id = frame.frame_id
        self.frame_realization_id = frame_realization
        self.unit_contract_id = units.contract_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-quadrature-off-shell-dark-transport",
                "support": self.support_id,
                "species": self.species_plan_ids,
                "frame": self.frame_id,
                "frame_realization": self.frame_realization_id,
                "units": self.unit_contract_id,
                "energy_nodes": array_tree_fingerprint(energies),
                "energy_weights": array_tree_fingerprint(energy_measure),
                "momentum_weights": array_tree_fingerprint(momentum_measure),
                "cell_volumes": array_tree_fingerprint(cells),
                "statistics": statistics.tolist(),
                "energy_quadrature": self.energy_quadrature_id,
                "momentum_quadrature": self.momentum_quadrature_id,
                "tolerances": (spectral_tol, dyson_tol, kms_tol),
                "maximum_spectral_bytes": maximum,
                "profile": profile.profile_id,
            }
        )

    @property
    def spectral_shape(self) -> tuple[int, ...]:
        species, cells, momenta = self.quantum_support.occupancy.shape
        return (cells, momenta, species, self.energy_nodes.size)


class OffShellEvidence(StrictModule):
    spectral_minimum: Array
    width_minimum: Array
    spectral_sum: Array
    spectral_sum_rule_residual: Array
    dyson_residual: Array
    kms_residual: Array
    finite: Array
    causal: Array
    positive: Array
    sum_rule_satisfied: Array
    dyson_satisfied: Array
    kms_satisfied: Array
    support_respected: Array
    valid: Array
    frame_token: Array
    frame_time: Array
    frame_scale_factor: Array
    frame_realization_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    energy_quadrature_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)


class QuasiparticleOffShellState(StrictModule):
    spectral_function: Array
    occupation: Array
    real_retarded_self_energy: Array
    width: Array
    lesser_self_energy: Array
    greater_self_energy: Array
    pole_energy: Array
    inverse_temperature: Array
    chemical_potentials: Array
    time: Array
    evidence: OffShellEvidence
    support_id: str = eqx.field(static=True)
    energy_quadrature_id: str = eqx.field(static=True)
    momentum_quadrature_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def frame_token(self) -> Array:
        return self.evidence.frame_token

    @property
    def frame_time(self) -> Array:
        return self.evidence.frame_time

    @property
    def frame_scale_factor(self) -> Array:
        return self.evidence.frame_scale_factor

    @property
    def frame_realization_id(self) -> str:
        return self.evidence.frame_realization_id


class OffShellMoments(StrictModule):
    spectral_weight: Array
    spectral_energy_moment: Array
    occupation_number: Array
    energy_moment: Array
    number_density: Array
    charge_density: Array
    total_number: Array
    total_charge: Array
    stress_energy: Array
    finite: Array
    frame_token: Array
    frame_time: Array
    frame_scale_factor: Array
    frame_realization_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)


def off_shell_evidence(
    plan: OffShellTransportPlan,
    spectral_function: ArrayLike,
    occupation: ArrayLike,
    real_retarded_self_energy: ArrayLike,
    width: ArrayLike,
    lesser_self_energy: ArrayLike,
    greater_self_energy: ArrayLike,
    pole_energy: ArrayLike,
    inverse_temperature: ArrayLike,
    chemical_potentials: ArrayLike,
    /,
) -> OffShellEvidence:
    if not isinstance(plan, OffShellTransportPlan):
        raise TypeError("plan must be OffShellTransportPlan.")
    expected = plan.spectral_shape
    arrays = tuple(
        jnp.asarray(value)
        for value in (
            spectral_function,
            occupation,
            real_retarded_self_energy,
            width,
            lesser_self_energy,
            greater_self_energy,
        )
    )
    if any(value.shape != expected for value in arrays):
        raise ValueError(f"Off-shell arrays must all have shape {expected}.")
    if any(jnp.issubdtype(value.dtype, jnp.complexfloating) for value in arrays):
        raise TypeError(
            "Off-shell spectral, occupation, width, and self-energy rate arrays must be real."
        )
    spectral, distribution, real_sigma, gamma, lesser, greater = arrays
    pole = jnp.asarray(pole_energy)
    if jnp.issubdtype(pole.dtype, jnp.complexfloating):
        raise TypeError("pole_energy must be real.")
    species_count = len(plan.species)
    if pole.shape != expected[1:3]:
        raise ValueError("pole_energy must have shape (momentum, species).")
    beta = jnp.asarray(inverse_temperature, dtype=spectral.dtype)
    chemical = jnp.asarray(chemical_potentials, dtype=spectral.dtype)
    if beta.shape != () or chemical.shape != (species_count,):
        raise ValueError(
            "KMS temperature must be scalar with one chemical potential per species."
        )
    omega = plan.energy_nodes[None, None, None, :]
    active = (
        jnp.asarray(plan.quantum_support.spatial_active, dtype=jnp.bool_)[
            :, None, None, None
        ]
        & jnp.asarray(plan.quantum_support.momentum_active, dtype=jnp.bool_)[
            None, :, None, None
        ]
    )
    active = jnp.broadcast_to(active, expected)
    inverse = omega - pole[None, :, :, None] - real_sigma + 1.0j * gamma
    retarded = 1.0 / jnp.where(active, inverse, 1.0 + 0.0j)
    dyson_spectral = -2.0 * jnp.imag(retarded)
    dyson_residual = jnp.max(jnp.abs(spectral - dyson_spectral))
    kms_factor = jnp.exp(beta * (omega - chemical[None, None, :, None]))
    kms_scale = jnp.maximum(1.0, jnp.abs(greater) + jnp.abs(kms_factor * lesser))
    kms_residual = jnp.max(jnp.abs(greater - kms_factor * lesser) / kms_scale)
    spectral_sum = contract("xpsw,w->xps", spectral, plan.energy_weights) / (2.0 * jnp.pi)
    active_phase = active[..., 0]
    sum_residual = jnp.max(jnp.where(active_phase, jnp.abs(spectral_sum - 1.0), 0.0))
    inactive_residual = jnp.max(
        jnp.stack(
            tuple(jnp.max(jnp.abs(jnp.where(active, 0.0, value))) for value in arrays)
        )
    )
    array_finite = jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in arrays))
    finite = (
        jnp.all(array_finite)
        & jnp.all(jnp.isfinite(pole))
        & jnp.isfinite(beta)
        & jnp.all(jnp.isfinite(chemical))
        & jnp.isfinite(sum_residual)
        & jnp.isfinite(dyson_residual)
        & jnp.isfinite(kms_residual)
    )
    minimum_spectral = jnp.min(jnp.where(active, spectral, jnp.inf))
    minimum_width = jnp.min(jnp.where(active, gamma, jnp.inf))
    causal = minimum_width >= 0.0
    fermi = plan.statistics[None, None, :, None] == -1
    positive = (
        (minimum_spectral >= 0.0)
        & jnp.all(distribution >= 0.0)
        & jnp.all((~fermi) | (distribution <= 1.0))
        & jnp.all(lesser >= 0.0)
        & jnp.all(greater >= 0.0)
    )
    support_respected = inactive_residual <= plan.dyson_tolerance
    sum_rule = sum_residual <= plan.spectral_tolerance
    dyson = dyson_residual <= plan.dyson_tolerance
    kms = kms_residual <= plan.kms_tolerance
    valid = finite & causal & positive & support_respected & sum_rule & dyson & kms
    return OffShellEvidence(
        minimum_spectral,
        minimum_width,
        spectral_sum,
        sum_residual,
        dyson_residual,
        kms_residual,
        finite,
        causal,
        positive,
        sum_rule,
        dyson,
        kms,
        support_respected,
        valid,
        plan.frame.frame_token,
        plan.frame.time,
        plan.frame.scale_factor,
        plan.frame_realization_id,
        plan.support_id,
        plan.energy_quadrature_id,
        plan.frame_id,
        plan.unit_contract_id,
    )


def quasiparticle_off_shell_state(
    plan: OffShellTransportPlan,
    spectral_function: ArrayLike,
    occupation: ArrayLike,
    real_retarded_self_energy: ArrayLike,
    width: ArrayLike,
    lesser_self_energy: ArrayLike,
    greater_self_energy: ArrayLike,
    pole_energy: ArrayLike,
    /,
    *,
    inverse_temperature: ArrayLike,
    chemical_potentials: ArrayLike,
    time: ArrayLike,
) -> QuasiparticleOffShellState:
    evidence = off_shell_evidence(
        plan,
        spectral_function,
        occupation,
        real_retarded_self_energy,
        width,
        lesser_self_energy,
        greater_self_energy,
        pole_energy,
        inverse_temperature,
        chemical_potentials,
    )
    structural_valid = (
        evidence.finite & evidence.causal & evidence.positive & evidence.support_respected
    )
    spectral = eqx.error_if(
        jnp.asarray(spectral_function),
        ~structural_valid,
        "Off-shell state violates finiteness, causality, positivity, or support masks.",
    )
    time_ = jnp.asarray(time, dtype=spectral.dtype)
    time_ = eqx.error_if(
        time_, ~jnp.isfinite(time_), "Off-shell state time must be finite."
    )
    return QuasiparticleOffShellState(
        spectral,
        jnp.asarray(occupation),
        jnp.asarray(real_retarded_self_energy),
        jnp.asarray(width),
        jnp.asarray(lesser_self_energy),
        jnp.asarray(greater_self_energy),
        jnp.asarray(pole_energy),
        jnp.asarray(inverse_temperature, dtype=spectral.dtype),
        jnp.asarray(chemical_potentials, dtype=spectral.dtype),
        time_,
        evidence,
        plan.support_id,
        plan.energy_quadrature_id,
        plan.momentum_quadrature_id,
        plan.frame_id,
        plan.unit_contract_id,
        plan.plan_id,
    )


def breit_wigner_off_shell_state(
    plan: OffShellTransportPlan,
    pole_energy: ArrayLike,
    width: ArrayLike,
    real_retarded_self_energy: ArrayLike = 0.0,
    /,
    *,
    inverse_temperature: ArrayLike,
    chemical_potentials: ArrayLike,
    time: ArrayLike = 0.0,
) -> QuasiparticleOffShellState:
    """Construct the unmodified Dyson Breit--Wigner spectral state on the plan grid."""

    if not isinstance(plan, OffShellTransportPlan):
        raise TypeError("plan must be OffShellTransportPlan.")
    pole = jnp.asarray(pole_energy, dtype=plan.energy_nodes.dtype)
    expected_pole = plan.spectral_shape[1:3]
    if pole.shape != expected_pole:
        raise ValueError(f"pole_energy must have shape {expected_pole}.")
    target = plan.spectral_shape
    gamma_input = jnp.asarray(width, dtype=pole.dtype)
    sigma_input = jnp.asarray(real_retarded_self_energy, dtype=pole.dtype)
    if gamma_input.shape == expected_pole:
        gamma_input = gamma_input[None, :, :, None]
    if sigma_input.shape == expected_pole:
        sigma_input = sigma_input[None, :, :, None]
    if jnp.broadcast_shapes(gamma_input.shape, target) != target:
        raise ValueError("width is not broadcastable over off-shell support.")
    if jnp.broadcast_shapes(sigma_input.shape, target) != target:
        raise ValueError(
            "real_retarded_self_energy is not broadcastable over off-shell support."
        )
    gamma = jnp.broadcast_to(gamma_input, target)
    real_sigma = jnp.broadcast_to(sigma_input, target)
    beta = jnp.asarray(inverse_temperature, dtype=pole.dtype)
    chemical = jnp.asarray(chemical_potentials, dtype=pole.dtype)
    if beta.shape != () or chemical.shape != (len(plan.species),):
        raise ValueError(
            "KMS temperature/chemical potentials do not match species support."
        )
    omega = plan.energy_nodes[None, None, None, :]
    offset = omega - pole[None, :, :, None] - real_sigma
    spectral = 2.0 * gamma / (offset * offset + gamma * gamma)
    exponent = beta * (omega - chemical[None, None, :, None])
    eta = plan.statistics[None, None, :, None]
    denominator = jnp.exp(exponent) - eta
    distribution = 1.0 / denominator
    lesser = 2.0 * gamma * distribution
    greater = 2.0 * gamma * (1.0 + eta * distribution)
    active = (
        jnp.asarray(plan.quantum_support.spatial_active, dtype=jnp.bool_)[
            :, None, None, None
        ]
        & jnp.asarray(plan.quantum_support.momentum_active, dtype=jnp.bool_)[
            None, :, None, None
        ]
    )
    spectral = jnp.where(active, spectral, 0.0)
    distribution = jnp.where(active, distribution, 0.0)
    real_sigma = jnp.where(active, real_sigma, 0.0)
    gamma = jnp.where(active, gamma, 0.0)
    lesser = jnp.where(active, lesser, 0.0)
    greater = jnp.where(active, greater, 0.0)
    return quasiparticle_off_shell_state(
        plan,
        spectral,
        distribution,
        real_sigma,
        gamma,
        lesser,
        greater,
        pole,
        inverse_temperature=beta,
        chemical_potentials=chemical,
        time=time,
    )


def off_shell_moments(
    plan: OffShellTransportPlan,
    state: QuasiparticleOffShellState,
    local_four_momenta: ArrayLike,
    /,
) -> OffShellMoments:
    if not isinstance(plan, OffShellTransportPlan):
        raise TypeError("plan must be OffShellTransportPlan.")
    if not isinstance(state, QuasiparticleOffShellState) or state.plan_id != plan.plan_id:
        raise ValueError("Off-shell state belongs to a different transport plan.")
    momentum = jnp.asarray(local_four_momenta)
    expected = (plan.spectral_shape[1], plan.spectral_shape[2], 4)
    if momentum.shape != expected:
        raise ValueError(f"local_four_momenta must have shape {expected}.")
    momentum = eqx.error_if(
        momentum,
        ~jnp.all(jnp.isfinite(momentum)) | jnp.any(momentum[..., 0] <= 0.0),
        "Local four-momenta must be finite with positive energy components.",
    )
    measure = plan.energy_weights / (2.0 * jnp.pi)
    spectral_weight = contract("xpsw,w->xps", state.spectral_function, measure)
    spectral_energy_moment = contract(
        "xpsw,w,w->xps",
        state.spectral_function,
        measure,
        plan.energy_nodes,
    )
    occupation_number = contract(
        "xpsw,xpsw,w->xps",
        state.spectral_function,
        state.occupation,
        measure,
    )
    energy_moment = contract(
        "xpsw,xpsw,w,w->xps",
        state.spectral_function,
        state.occupation,
        measure,
        plan.energy_nodes,
    )
    weighted = occupation_number * plan.momentum_weights[None, :, None]
    number_density = jnp.sum(weighted, axis=(1, 2))
    charges = jnp.stack(tuple(value.charges for value in plan.species), axis=0)
    charge_density = contract("xps,sc->xc", weighted, charges)
    total_number = contract("x,x->", plan.cell_volumes, number_density)
    total_charge = contract("x,xc->c", plan.cell_volumes, charge_density)
    stress_energy = contract(
        "xps,p,ps,psa,psb->xab",
        occupation_number,
        plan.momentum_weights,
        1.0 / momentum[..., 0],
        momentum,
        momentum,
    )
    finite = (
        state.evidence.valid
        & jnp.all(jnp.isfinite(spectral_weight))
        & jnp.all(jnp.isfinite(spectral_energy_moment))
        & jnp.all(jnp.isfinite(occupation_number))
        & jnp.all(jnp.isfinite(energy_moment))
        & jnp.all(jnp.isfinite(stress_energy))
    )
    return OffShellMoments(
        spectral_weight,
        spectral_energy_moment,
        occupation_number,
        energy_moment,
        number_density,
        charge_density,
        total_number,
        total_charge,
        stress_energy,
        finite,
        plan.frame.frame_token,
        plan.frame.time,
        plan.frame.scale_factor,
        plan.frame_realization_id,
        plan.plan_id,
        plan.support_id,
        plan.frame_id,
        plan.unit_contract_id,
    )


__all__ = [
    "OffShellEvidence",
    "OffShellMoments",
    "OffShellTransportPlan",
    "OffShellTransportProfile",
    "QuasiparticleOffShellState",
    "breit_wigner_off_shell_state",
    "off_shell_evidence",
    "off_shell_moments",
    "quasiparticle_off_shell_state",
]
