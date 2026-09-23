#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Energy-integrated coherent multi-terminal electronic transport."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    solve,
)
from ..operators.quantum import (
    elastic_transmission,
    electronic_contact_current_kernels,
    ELECTRONIC_TRANSPORT_CONVENTION,
    PeriodicPrincipalLayerLeadPlan,
    prepare_periodic_lead_embedding,
    retarded_embedding,
    retarded_open_system_point,
)


_BOLTZMANN_CONSTANT_SI = 1.380_649e-23
_PLANCK_CONSTANT_SI = 6.626_070_15e-34


class PeriodicLeadContactPlan(StrictModule, NonTrainableState):
    lead: PeriodicPrincipalLayerLeadPlan
    device_coupling: Array
    contact_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lead: PeriodicPrincipalLayerLeadPlan,
        device_coupling: ArrayLike,
        contact_id: str,
        /,
    ):
        if not isinstance(lead, PeriodicPrincipalLayerLeadPlan):
            raise TypeError("lead must be PeriodicPrincipalLayerLeadPlan.")
        coupling = np.asarray(device_coupling)
        identifier = str(contact_id).strip()
        if (
            coupling.ndim != 2
            or coupling.shape[1] != lead.onsite.shape[0]
            or np.any(~np.isfinite(coupling))
            or not identifier
        ):
            raise ValueError("Lead contact coupling or identity is invalid.")
        self.lead = lead
        self.device_coupling = jnp.asarray(coupling, dtype=jnp.complex128)
        self.contact_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-lead-contact-plan",
                "lead": lead.plan_id,
                "contact_id": identifier,
                "coupling": array_tree_fingerprint(coupling),
            }
        )


class MultiTerminalCoherentProblem(StrictModule, NonTrainableState):
    hamiltonian: Array
    overlap: Array
    contacts: tuple[PeriodicLeadContactPlan, ...]
    energy_grid_joule: Array
    quadrature_weights_joule: Array
    chemical_potentials_joule: Array
    temperatures_kelvin: Array
    particle_continuity_tolerance: float = eqx.field(static=True)
    energy_continuity_tolerance_watt: float = eqx.field(static=True)
    numerical_broadening_joule: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: ArrayLike,
        overlap: ArrayLike,
        contacts: Sequence[PeriodicLeadContactPlan],
        energy_grid_joule: ArrayLike,
        quadrature_weights_joule: ArrayLike,
        chemical_potentials_joule: ArrayLike,
        temperatures_kelvin: ArrayLike,
        /,
        *,
        particle_continuity_tolerance: float = 1.0e-8,
        energy_continuity_tolerance_watt: float = 1.0e-12,
        numerical_broadening_joule: float = 1.0e-12,
    ):
        hamiltonian_ = np.asarray(hamiltonian)
        overlap_ = np.asarray(overlap)
        contacts_ = tuple(contacts)
        energy = np.asarray(energy_grid_joule)
        weights = np.asarray(quadrature_weights_joule)
        chemical = np.asarray(chemical_potentials_joule)
        temperature = np.asarray(temperatures_kelvin)
        eta = float(numerical_broadening_joule)
        particle_tolerance = float(particle_continuity_tolerance)
        energy_tolerance = float(energy_continuity_tolerance_watt)
        if (
            hamiltonian_.ndim != 2
            or hamiltonian_.shape[0] != hamiltonian_.shape[1]
            or overlap_.shape != hamiltonian_.shape
            or np.any(~np.isfinite(hamiltonian_))
            or np.any(~np.isfinite(overlap_))
            or not np.allclose(hamiltonian_, np.conj(hamiltonian_.T))
            or not np.allclose(overlap_, np.conj(overlap_.T))
        ):
            raise ValueError(
                "Device Hamiltonian and overlap must be finite Hermitian matrices."
            )
        if not contacts_ or any(
            not isinstance(value, PeriodicLeadContactPlan) for value in contacts_
        ):
            raise TypeError("contacts must contain PeriodicLeadContactPlan values.")
        if len({value.contact_id for value in contacts_}) != len(contacts_):
            raise ValueError("Multi-terminal contact IDs must be unique.")
        if any(
            value.device_coupling.shape[0] != hamiltonian_.shape[0] for value in contacts_
        ):
            raise ValueError("Every contact coupling must act on the device basis.")
        if (
            energy.ndim != 1
            or energy.size < 1
            or weights.shape != energy.shape
            or chemical.shape != (len(contacts_),)
            or temperature.shape != chemical.shape
            or np.any(~np.isfinite(energy))
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or np.any(~np.isfinite(chemical))
            or np.any(~np.isfinite(temperature))
            or np.any(temperature < 0.0)
            or not isfinite(eta)
            or eta <= 0.0
            or not isfinite(particle_tolerance)
            or particle_tolerance <= 0.0
            or not isfinite(energy_tolerance)
            or energy_tolerance <= 0.0
        ):
            raise ValueError(
                "Transport grid, reservoirs, broadening, or continuity tolerances are invalid."
            )
        self.hamiltonian = jnp.asarray(hamiltonian_, dtype=jnp.complex128)
        self.overlap = jnp.asarray(overlap_, dtype=jnp.complex128)
        self.contacts = contacts_
        self.energy_grid_joule = jnp.asarray(energy, dtype=jnp.float64)
        self.quadrature_weights_joule = jnp.asarray(weights, dtype=jnp.float64)
        self.chemical_potentials_joule = jnp.asarray(chemical, dtype=jnp.float64)
        self.temperatures_kelvin = jnp.asarray(temperature, dtype=jnp.float64)
        self.numerical_broadening_joule = eta
        self.particle_continuity_tolerance = particle_tolerance
        self.energy_continuity_tolerance_watt = energy_tolerance
        self.problem_id = canonical_fingerprint(
            {
                "kind": "multi-terminal-coherent-problem",
                "contacts": tuple(value.plan_id for value in contacts_),
                "arrays": array_tree_fingerprint(
                    {
                        "hamiltonian": hamiltonian_,
                        "overlap": overlap_,
                        "energy": energy,
                        "weights": weights,
                        "chemical": chemical,
                        "temperature": temperature,
                    }
                ),
                "numerical_broadening_joule": eta,
                "particle_continuity_tolerance": particle_tolerance,
                "energy_continuity_tolerance_watt": energy_tolerance,
            }
        )


class MultiTerminalCoherentResult(StrictModule, NonTrainableState):
    transmissions: Array
    particle_current_ampere_per_charge: Array
    charge_current_ampere: Array
    energy_current_watt: Array
    heat_current_watt: Array
    particle_continuity_residual: Array
    energy_continuity_residual: Array
    point_successful: Array
    successful: Array
    problem_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def _fermi(energy: Array, chemical: Array, temperature: Array, /) -> Array:
    zero_temperature = temperature == 0.0
    scale = jnp.where(zero_temperature, 1.0, _BOLTZMANN_CONSTANT_SI * temperature)
    argument = (energy - chemical) / scale
    positive = jnp.exp(-jnp.maximum(argument, 0.0))
    negative = jnp.exp(jnp.minimum(argument, 0.0))
    finite_temperature = jnp.where(
        argument >= 0.0, positive / (1.0 + positive), 1.0 / (1.0 + negative)
    )
    return jnp.where(
        zero_temperature, (energy < chemical).astype(energy.dtype), finite_temperature
    )


def solve_multiterminal_coherent(
    problem: MultiTerminalCoherentProblem,
    /,
) -> MultiTerminalCoherentResult:
    if not isinstance(problem, MultiTerminalCoherentProblem):
        raise TypeError("problem must be MultiTerminalCoherentProblem.")
    contact_count = len(problem.contacts)

    def one_energy(energy):
        spectral_energy = energy + 1.0j * problem.numerical_broadening_joule
        prepared_leads = tuple(
            prepare_periodic_lead_embedding(
                contact.lead,
                energy,
                broadening=problem.numerical_broadening_joule,
            )
            for contact in problem.contacts
        )
        embeddings = tuple(
            retarded_embedding(
                spectral_energy,
                lead.surface_green,
                contact.device_coupling,
            )
            for contact, lead in zip(
                problem.contacts,
                prepared_leads,
                strict=True,
            )
        )
        point = retarded_open_system_point(
            spectral_energy,
            problem.hamiltonian,
            problem.overlap,
            embeddings,
        )
        transmission = jnp.zeros((contact_count, contact_count), dtype=jnp.float64)
        for source in range(contact_count):
            for target in range(contact_count):
                if source != target:
                    transmission = transmission.at[source, target].set(
                        elastic_transmission(
                            point.green,
                            embeddings[source].broadening,
                            embeddings[target].broadening,
                        )
                    )
        occupations = _fermi(
            energy,
            problem.chemical_potentials_joule,
            problem.temperatures_kelvin,
        )
        currents = electronic_contact_current_kernels(
            point,
            occupations,
            problem.chemical_potentials_joule,
        )
        point_successful = jnp.all(point.successful) & jnp.all(
            jnp.stack(tuple(value.successful for value in prepared_leads))
        )
        return (
            transmission,
            currents.particle_current_into_region,
            currents.energy_current_into_region,
            currents.heat_current_into_region,
            point_successful,
        )

    transmissions, particle, energy_current, heat, point_successful = jax.lax.map(
        one_energy,
        problem.energy_grid_joule,
    )
    measure = problem.quadrature_weights_joule[:, None] / _PLANCK_CONSTANT_SI
    integrated_particle = jnp.sum(measure * particle, axis=0)
    integrated_energy = jnp.sum(measure * energy_current, axis=0)
    integrated_heat = jnp.sum(measure * heat, axis=0)
    charge = ELECTRONIC_TRANSPORT_CONVENTION.charge_current(integrated_particle)
    particle_residual = jnp.abs(jnp.sum(integrated_particle))
    energy_residual = jnp.abs(jnp.sum(integrated_energy))
    successful = (
        jnp.all(point_successful)
        & jnp.all(jnp.isfinite(transmissions))
        & jnp.all(jnp.isfinite(charge))
        & (particle_residual <= problem.particle_continuity_tolerance)
        & (energy_residual <= problem.energy_continuity_tolerance_watt)
    )
    result_id = canonical_fingerprint(
        {"kind": "multi-terminal-coherent-result", "problem": problem.problem_id}
    )
    return MultiTerminalCoherentResult(
        transmissions,
        integrated_particle,
        charge,
        integrated_energy,
        integrated_heat,
        particle_residual,
        energy_residual,
        point_successful,
        successful,
        problem.problem_id,
        result_id,
    )


class TransportProbePlan(StrictModule, NonTrainableState):
    probe_indices: tuple[int, ...] = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        probe_indices: Sequence[int],
        /,
        *,
        residual_tolerance: float = 1.0e-10,
    ):
        indices = tuple(int(value) for value in probe_indices)
        tolerance = float(residual_tolerance)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(value < 0 for value in indices)
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Transport probe indices or tolerance are invalid.")
        self.probe_indices = indices
        self.residual_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "transport-probe-plan",
                "probe_indices": indices,
                "residual_tolerance": tolerance,
            }
        )


class TransportProbeResult(StrictModule, NonTrainableState):
    occupations: Array
    chemical_potentials_joule: Array
    probe_current_residual: Array
    linear_solve_status: Array
    successful: Array
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def _landauer_laplacian(transmission: Array, /) -> Array:
    off_diagonal = -transmission
    return off_diagonal.at[jnp.diag_indices(transmission.shape[0])].set(
        jnp.sum(transmission, axis=1)
    )


def _probe_linear_solve(
    laplacian: Array,
    probe_indices: tuple[int, ...],
    fixed_values: Array,
    /,
) -> tuple[Array, Array, Array]:
    count = laplacian.shape[0]
    probes = jnp.asarray(probe_indices, dtype=jnp.int32)
    fixed_mask = jnp.ones((count,), dtype=jnp.bool_).at[probes].set(False)
    fixed = jnp.flatnonzero(fixed_mask, size=count - len(probe_indices))
    matrix = laplacian[probes[:, None], probes[None, :]]
    right = -(laplacian[probes[:, None], fixed[None, :]] @ fixed_values[fixed])
    solved = solve(
        LinearSystem(DenseLinearOperator(matrix)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
    )
    values = fixed_values.at[probes].set(solved.value)
    residual = jnp.max(jnp.abs(laplacian[probes] @ values), initial=0.0)
    return values, residual, solved.status


def _validate_probe_inputs(
    problem: MultiTerminalCoherentProblem,
    coherent: MultiTerminalCoherentResult,
    plan: TransportProbePlan,
    /,
) -> None:
    if not isinstance(problem, MultiTerminalCoherentProblem) or not isinstance(
        coherent, MultiTerminalCoherentResult
    ):
        raise TypeError("problem and coherent have invalid types.")
    if not isinstance(plan, TransportProbePlan):
        raise TypeError("plan must be TransportProbePlan.")
    if coherent.problem_id != problem.problem_id:
        raise ValueError("Probe result and coherent problem identities differ.")
    if any(value >= len(problem.contacts) for value in plan.probe_indices):
        raise ValueError("A probe index is outside the contact roster.")
    if len(plan.probe_indices) >= len(problem.contacts):
        raise ValueError("At least one physical contact must remain fixed.")


def solve_dephasing_probes(
    problem: MultiTerminalCoherentProblem,
    coherent: MultiTerminalCoherentResult,
    plan: TransportProbePlan,
    /,
) -> TransportProbeResult:
    """Solve zero-current occupations independently at every energy."""

    _validate_probe_inputs(problem, coherent, plan)
    physical = jax.vmap(_fermi, in_axes=(0, None, None))(
        problem.energy_grid_joule,
        problem.chemical_potentials_joule,
        problem.temperatures_kelvin,
    )

    def one(transmission, occupation):
        return _probe_linear_solve(
            _landauer_laplacian(transmission),
            plan.probe_indices,
            occupation,
        )

    occupations, residuals, solve_status = jax.vmap(one)(coherent.transmissions, physical)
    residual = jnp.max(residuals)
    successful = (
        coherent.successful
        & jnp.all(solve_status == int(LinearSolveStatus.SUCCESS))
        & jnp.all(
            (occupations >= -plan.residual_tolerance)
            & (occupations <= 1.0 + plan.residual_tolerance)
        )
        & (residual <= plan.residual_tolerance)
    )
    return TransportProbeResult(
        occupations,
        jnp.full_like(problem.chemical_potentials_joule, jnp.nan),
        residual,
        solve_status,
        successful,
        problem.problem_id,
        plan.plan_id,
        canonical_fingerprint(
            {
                "kind": "dephasing-probe-result",
                "problem": problem.problem_id,
                "plan": plan.plan_id,
            }
        ),
    )


def solve_voltage_probes(
    problem: MultiTerminalCoherentProblem,
    coherent: MultiTerminalCoherentResult,
    plan: TransportProbePlan,
    /,
) -> TransportProbeResult:
    """Solve linear-response integrated zero-current probe potentials."""

    _validate_probe_inputs(problem, coherent, plan)
    probe_indices = jnp.asarray(plan.probe_indices, dtype=jnp.int32)
    probe_temperatures = problem.temperatures_kelvin[probe_indices]
    probe_temperatures = eqx.error_if(
        probe_temperatures,
        jnp.any(probe_temperatures <= 0.0),
        "Linear-response voltage probes require positive probe temperatures.",
    )
    temperatures = problem.temperatures_kelvin.at[probe_indices].set(probe_temperatures)
    occupations_at_reference = jax.vmap(_fermi, in_axes=(0, None, None))(
        problem.energy_grid_joule,
        problem.chemical_potentials_joule,
        temperatures,
    )
    positive_temperature = temperatures > 0.0
    safe_temperature = jnp.where(positive_temperature, temperatures, 1.0)
    susceptibility = jnp.where(
        positive_temperature[None, :],
        occupations_at_reference
        * (1.0 - occupations_at_reference)
        / (_BOLTZMANN_CONSTANT_SI * safe_temperature[None, :]),
        0.0,
    )
    laplacians = jax.vmap(_landauer_laplacian)(coherent.transmissions)
    measure = problem.quadrature_weights_joule / _PLANCK_CONSTANT_SI
    base_currents = jnp.sum(
        measure[:, None]
        * jax.vmap(lambda matrix, values: matrix @ values)(
            laplacians, occupations_at_reference
        ),
        axis=0,
    )
    jacobian = jnp.sum(
        measure[:, None, None] * laplacians * susceptibility[:, None, :],
        axis=0,
    )
    probe_matrix = jacobian[probe_indices[:, None], probe_indices[None, :]]
    solved = solve(
        LinearSystem(DenseLinearOperator(probe_matrix)),
        -base_currents[probe_indices],
        policy=LinearSolvePolicy(DenseLU()),
    )
    delta = (
        jnp.zeros_like(problem.chemical_potentials_joule)
        .at[probe_indices]
        .set(solved.value)
    )
    potentials = problem.chemical_potentials_joule + delta
    residual = jnp.max(
        jnp.abs(base_currents[probe_indices] + jacobian[probe_indices] @ delta)
    )
    occupations = jax.vmap(_fermi, in_axes=(0, None, None))(
        problem.energy_grid_joule,
        potentials,
        temperatures,
    )
    successful = (
        coherent.successful
        & solved.successful
        & jnp.all(jnp.isfinite(potentials))
        & (residual <= plan.residual_tolerance)
    )
    return TransportProbeResult(
        occupations,
        potentials,
        residual,
        solved.status,
        successful,
        problem.problem_id,
        plan.plan_id,
        canonical_fingerprint(
            {
                "kind": "voltage-probe-result",
                "problem": problem.problem_id,
                "plan": plan.plan_id,
            }
        ),
    )


__all__ = [
    "MultiTerminalCoherentProblem",
    "MultiTerminalCoherentResult",
    "PeriodicLeadContactPlan",
    "TransportProbePlan",
    "TransportProbeResult",
    "solve_dephasing_probes",
    "solve_multiterminal_coherent",
    "solve_voltage_probes",
]
