# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Coherent zero-frequency noise, quasistatics, and screened finite-frequency response.

Noise is symmetrized, two-sided zero-frequency scattering noise (A²/Hz), with
S=4 kB T G at equilibrium in the usual electrical spectral-density convention.
Quasistatics returns DC conductance and static charge derivatives, NOT a
finite-frequency admittance. Full AC below uses connected-equilibrium Kubo
response of the causal finite lead dilation and self-consistent capacitive
Hartree screening, including lead polarization and displacement. It is not
an interacting SCBA vertex/noise calculation or a DD R+iωS substitution.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .... import linalg as la
from ...._strict import StrictModule
from ....ein import contract
from ....integration import adaptive_interval_callable, AdaptiveQuadraturePlan
from ._basis import _array, HBAR, KB, PLANCK, Q
from ._coherent import bound_states, CoherentDevice
from ._dynamic import _finite_hamiltonian, _hermitian_spectrum
from ._leads import BoundStateOccupation


def _coherent_moments(device, tolerance):
    leads = (device.left, device.right)
    bands = [np.asarray(lead.band()) for lead in leads]
    lo, hi = min(b[0] for b in bands), max(b[1] for b in bands)
    origin = (lo + hi) / 2
    breaks = set(np.linspace((lo - origin) / Q, (hi - origin) / Q, 25)[1:-1])
    for lead in leads:
        for offset in device.transverse.offsets:
            for width in (-10.0, -3.0, 0.0, 3.0, 10.0):
                value = float(
                    (
                        lead.chemical_potential
                        - offset
                        + width * KB * lead.temperature
                        - origin
                    )
                    / Q
                )
                if (lo - origin) / Q < value < (hi - origin) / Q:
                    breaks.add(value)
        for edge in lead.band():
            if lo < float(edge) < hi:
                breaks.add(float((edge - origin) / Q))

    def integrand(x):
        energy = origin + Q * x
        point = device.spectral(energy)
        f = jnp.stack(
            tuple(
                jax.nn.sigmoid(
                    (lead.chemical_potential - energy - device.transverse.offsets)
                    / (KB * lead.temperature)
                )
                for lead in leads
            )
        )
        degeneracy = device.transverse.degeneracies
        t = point.transmission
        thermal = f[0] * (1 - f[0]) + f[1] * (1 - f[1])
        partition = (f[0] - f[1]) ** 2
        noise = jnp.sum(degeneracy * (t * thermal + t * (1 - t) * partition))
        derivative = jnp.sum(
            degeneracy * f[0] * (1 - f[0]) * Q / (KB * leads[0].temperature)
        )
        conductance = t * derivative
        # dN/d(mu/q), per lead, for a uniform internal potential.
        injectivity = jnp.sum(point.injection, axis=0) * Q / (2 * jnp.pi) * derivative
        return jnp.concatenate(
            (
                jnp.asarray([noise, conductance]),
                injectivity,
                jnp.asarray([jnp.where(point.successful, 0.0, jnp.nan)]),
            )
        )

    plan = AdaptiveQuadraturePlan(
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
        max_intervals=device.hamiltonian.resources.max_intervals,
        max_evaluations=device.hamiltonian.resources.max_evaluations,
        breakpoints=tuple(sorted(breaks)),
        collect_partition=True,
        throw=False,
    )
    result = adaptive_interval_callable(
        jax.vmap(integrand),
        jnp.asarray(((lo - origin) / Q, (hi - origin) / Q)),
        plan,
    )
    return result


class CoherentNoiseResult(StrictModule):
    spectrum: Array
    equilibrium_conductance: Array
    fluctuation_dissipation_error: Array
    partition_noise: str = eqx.field(static=True)
    quadrature: object
    successful: Array


def coherent_low_frequency_noise(device, *, tolerance=1e-7):
    """Two-terminal Landauer–Büttiker thermal plus Pauli partition noise.

    Currents have equal-and-opposite cross correlation. Out of equilibrium
    the supplied conductance and FDT error are NaN, rather than an invented
    interacting or finite-frequency noise prediction.
    """
    if not isinstance(device, CoherentDevice):
        raise TypeError("Noise requires CoherentDevice.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    estimate = _coherent_moments(device, tolerance)
    value = estimate.value
    amplitude = 2 * Q**3 / PLANCK * value[0]
    spectrum = amplitude * jnp.asarray([[1.0, -1.0], [-1.0, 1.0]])
    equilibrium = float(device.left.chemical_potential) == float(
        device.right.chemical_potential
    ) and float(device.left.temperature) == float(device.right.temperature)
    conductance = Q**2 / PLANCK * value[1] if equilibrium else jnp.asarray(jnp.nan)
    fdt = (
        jnp.abs(amplitude - 4 * KB * device.left.temperature * conductance)
        / (Q**3 / PLANCK)
        if equilibrium
        else jnp.asarray(jnp.nan)
    )
    fdt_valid = (fdt <= 10 * tolerance) if equilibrium else jnp.asarray(True)
    return CoherentNoiseResult(
        spectrum=spectrum,
        equilibrium_conductance=conductance,
        fluctuation_dissipation_error=fdt,
        partition_noise="coherent elastic zero-frequency; no interacting vertex",
        quadrature=estimate,
        successful=estimate.successful
        & jnp.isfinite(amplitude)
        & (amplitude >= 0)
        & fdt_valid,
    )


class QuantumCapacitance(StrictModule):
    """Physical electrostatic edges from each quantum cell to L, R, gate.

    All entries are F, nonnegative; each cell has positive total capacitance.
    This is a declared local capacitance/Hartree model, not a solved 3D Poisson
    geometry. Static fixed/background charge is already in the operating
    Hamiltonian. The response owns only induced electron charge.
    """

    cell_terminal: Array

    def __init__(self, cell_terminal):
        c = _array(cell_terminal, "cell-to-electrode capacitance")
        if (
            c.ndim != 2
            or c.shape[1] != 3
            or np.any(np.asarray(c) < 0)
            or np.any(np.sum(np.asarray(c), axis=1) <= 0)
        ):
            raise ValueError(
                "Capacitance must have shape (cells,3), nonnegative entries and positive row sums."
            )
        self.cell_terminal = c


class QuasistaticQuantumResponse(StrictModule):
    dc_conductance: Array
    electrode_charge_derivative: Array
    device_charge_derivative: Array
    characteristic_potential: Array
    quadrature: object
    successful: Array
    closure: str = eqx.field(
        static=True,
        default="equilibrium continuum-only uniform-potential quasistatics; not finite-frequency admittance",
    )


def quasistatic_quantum_response(device, terminal_capacitances, *, tolerance=1e-7):
    """Equilibrium DC G and static dQ/dV for one equipotential quantum region.

    A separate, explicitly bounded three-terminal lumped closure. Bound
    populations must not be differentiated as if reservoir injection fixed
    them, so this continuum-only operation rejects resolved bound poles.
    """
    if not isinstance(device, CoherentDevice):
        raise TypeError("Quasistatics requires CoherentDevice.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    if float(device.left.chemical_potential) != float(
        device.right.chemical_potential
    ) or float(device.left.temperature) != float(device.right.temperature):
        raise ValueError("Quasistatic injectivity closure is equilibrium-only.")
    preparation = BoundStateOccupation.equilibrium(device.left, device.right)
    poles = bound_states(device, preparation)
    if poles.energies.size:
        raise ValueError("Continuum quasistatics does not thermalize bound populations.")
    capacitance = _array(terminal_capacitances, "terminal capacitances")
    if (
        capacitance.shape != (3,)
        or np.any(np.asarray(capacitance) < 0)
        or float(jnp.sum(capacitance)) <= 0
    ):
        raise ValueError(
            "Three nonnegative physical capacitances with positive total are required."
        )
    estimate = _coherent_moments(device, tolerance)
    values = estimate.value
    injectivity = jnp.concatenate((values[2:4], jnp.zeros(1))) * Q
    u = (capacitance + injectivity) / (jnp.sum(capacitance) + jnp.sum(injectivity))
    electrode = jnp.diag(capacitance) - capacitance[:, None] * u[None, :]
    device_charge = -jnp.sum(electrode, axis=0)
    dc = (
        jnp.zeros((3, 3))
        .at[:2, :2]
        .set(Q**2 / PLANCK * values[1] * jnp.asarray([[1.0, -1.0], [-1.0, 1.0]]))
    )
    return QuasistaticQuantumResponse(
        dc_conductance=dc,
        electrode_charge_derivative=electrode,
        device_charge_derivative=device_charge,
        characteristic_potential=u,
        quadrature=estimate,
        successful=estimate.successful & poles.successful,
    )


class QuantumFrequencyEvidence(StrictModule):
    gauge_error: Array
    kcl_error: Array
    ward_error: Array
    electrostatic_error: Array
    lead_refinement_error: Array
    adiabatic_refinement_error: Array
    recurrence_tail_bound: Array
    lead_sites: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    eigen_successful: Array
    closure: str = eqx.field(
        static=True,
        default="connected-equilibrium coherent Kubo plus capacitive Hartree; e^(-i z t); no phonon vertex",
    )


class QuantumFrequencyResponse(StrictModule):
    angular_frequency: Array
    adiabatic_rate: Array
    admittance: Array
    particle_admittance: Array
    displacement_admittance: Array
    device_charge_response: Array
    lead_charge_response: Array
    electrode_charge_response: Array
    characteristic_potential: Array
    evidence: QuantumFrequencyEvidence
    successful: Array


def _frequency_once(device, capacitance, omega, rate, sites):
    n, dim = device.hamiltonian.size, device.hamiltonian.size + 2 * sites
    resources = device.hamiltonian.resources
    # Physical transition projectors, not basis-probed dense Jacobians.
    workspace = 16 * (dim * dim * (n + 12) + (n + 2) ** 2 * 6)
    if dim > resources.max_nodes or workspace > resources.workspace_bytes:
        raise ValueError(
            "Full-frequency lead/polarization workspace exceeds QuantumResources."
        )
    h = _finite_hamiltonian(device, sites)
    origin = jnp.mean(jnp.diag(h))
    energies, vectors, eigen_ok = _hermitian_spectrum(
        h - origin * jnp.eye(dim), resources
    )
    energy = energies / Q
    occupations = device.transverse.occupation(
        energies + origin, device.left.chemical_potential, device.left.temperature
    )
    # Observable/projector order is every device cell, left lead, right lead.
    partition = jnp.zeros((n + 2, dim)).at[jnp.arange(n), sites + jnp.arange(n)].set(1.0)
    partition = partition.at[n, :sites].set(1.0).at[n + 1, sites + n :].set(1.0)
    projectors = contract(
        "pi,ia,ib->pab", partition, vectors.conj(), vectors, backend="jax"
    )
    delta = energy[:, None] - energy[None, :]
    z = omega + 1j * rate
    kernel = (occupations[:, None] - occupations[None, :]) / (HBAR * z / Q + delta)
    # q*chi_scaled maps volts to signed electronic charge C.
    charge_chi = Q * contract(
        "pab,ab,qba->pq", projectors, kernel, projectors, backend="jax"
    )
    # Independently derived bond/lead commutator current response. Keeping it
    # separate from iz Qlead makes the Ward identity a genuine observable gate.
    current_chi = (
        -1j
        * Q**2
        / HBAR
        * contract(
            "pab,ab,qba->pq", projectors[-2:], delta * kernel, projectors, backend="jax"
        )
    )
    c = capacitance.cell_terminal
    lead_inputs = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    matrix = (jnp.diag(jnp.sum(c, axis=1)) - charge_chi[:n, :n]) / Q
    rhs = (c + charge_chi[:n, n:] @ lead_inputs) / Q
    linear = la.solve_many(
        la.prepare(
            la.LinearSystem(la.DenseLinearOperator(matrix)), rhs_layout=la.RHSLayout((3,))
        ),
        rhs,
    )
    u = linear.value
    full_drive = jnp.concatenate((u, lead_inputs), axis=0)
    electronic_charge = charge_chi @ full_drive
    electrode = jnp.diag(jnp.sum(c, axis=0)) - c.T @ u
    particle = jnp.zeros((3, 3), dtype=complex).at[:2].set(current_chi @ full_drive)
    displacement = -1j * z * electrode
    admittance = particle + displacement
    scale = Q**2 / HBAR
    gauge = jnp.max(jnp.abs(jnp.sum(admittance, axis=1))) / scale
    kcl = jnp.max(jnp.abs(jnp.sum(admittance, axis=0))) / scale
    ward = jnp.max(jnp.abs(particle[:2] - 1j * z * electronic_charge[n:])) / scale
    electrostatic = jnp.max(jnp.abs(matrix @ u - rhs))
    recurrence = min(
        sites * HBAR / (2 * abs(float(lead.hopping)))
        for lead in (device.left, device.right)
    )
    tail = jnp.exp(-rate * recurrence)
    evidence = QuantumFrequencyEvidence(
        gauge_error=gauge,
        kcl_error=kcl,
        ward_error=ward,
        electrostatic_error=electrostatic,
        lead_refinement_error=jnp.asarray(jnp.inf),
        adiabatic_refinement_error=jnp.asarray(jnp.inf),
        recurrence_tail_bound=tail,
        lead_sites=sites,
        workspace_bytes=workspace,
        eigen_successful=eigen_ok,
    )
    valid = (
        eigen_ok
        & jnp.all(linear.successful)
        & (gauge < 1e-8)
        & (kcl < 1e-8)
        & (ward < 1e-8)
        & (electrostatic < 1e-8)
    )
    return QuantumFrequencyResponse(
        angular_frequency=jnp.asarray(omega),
        adiabatic_rate=jnp.asarray(rate),
        admittance=admittance,
        particle_admittance=particle,
        displacement_admittance=displacement,
        device_charge_response=electronic_charge[:n],
        lead_charge_response=electronic_charge[n:],
        electrode_charge_response=electrode,
        characteristic_potential=u,
        evidence=evidence,
        successful=valid,
    )


def finite_frequency_quantum_response(
    device,
    capacitance,
    angular_frequency,
    *,
    adiabatic_rate,
    lead_sites=32,
    tolerance=2e-2,
):
    """Full coherent screened AC at z=omega+i*adiabatic_rate, with refinements.

    ``adiabatic_rate`` is the inverse switch-on observation window, NOT a
    scattering linewidth. Both doubled leads (at fixed rate) and halved rate
    (at doubled leads) must converge. The exponential recurrence-tail bound
    is independent. The final result reports the actual halved rate used;
    finite-real-frequency qualification requires driving it toward zero as
    lead size increases, rather than fitting it to dissipative measurements.
    Lead electrochemical perturbations and band shifts share -qV. The gate
    has displacement current only. No interacting vertex is silently omitted
    because this API accepts only the coherent device, never SCBA results.
    """
    if not isinstance(device, CoherentDevice) or not isinstance(
        capacitance, QuantumCapacitance
    ):
        raise TypeError(
            "Full-frequency response requires a coherent scalar chain and QuantumCapacitance."
        )
    if float(device.left.chemical_potential) != float(
        device.right.chemical_potential
    ) or float(device.left.temperature) != float(device.right.temperature):
        raise ValueError(
            "Connected-equilibrium Kubo response requires common lead mu and temperature."
        )
    if capacitance.cell_terminal.shape[0] != device.hamiltonian.size:
        raise ValueError("Electrostatic response and Hamiltonian cells must match.")
    if (
        not np.isfinite(angular_frequency)
        or angular_frequency < 0
        or not np.isfinite(adiabatic_rate)
        or adiabatic_rate <= 0
        or not np.isfinite(tolerance)
        or tolerance <= 0
        or isinstance(lead_sites, bool)
        or not isinstance(lead_sites, (int, np.integer))
        or lead_sites < 2
    ):
        raise ValueError(
            "Require nonnegative finite frequency, positive finite rate/tolerance, "
            "and an integer lead size of at least two."
        )
    coarse = _frequency_once(
        device, capacitance, angular_frequency, adiabatic_rate, lead_sites
    )
    doubled = _frequency_once(
        device, capacitance, angular_frequency, adiabatic_rate, 2 * lead_sites
    )
    fine = _frequency_once(
        device, capacitance, angular_frequency, adiabatic_rate / 2, 2 * lead_sites
    )
    scale = Q**2 / HBAR
    error = lambda a, b: jnp.max(jnp.abs(a.admittance - b.admittance)) / scale
    lead_error, rate_error = error(coarse, doubled), error(doubled, fine)
    evidence = eqx.tree_at(
        lambda e: (e.lead_refinement_error, e.adiabatic_refinement_error),
        fine.evidence,
        (lead_error, rate_error),
    )
    valid = (
        coarse.successful
        & doubled.successful
        & fine.successful
        & (lead_error < tolerance)
        & (rate_error < tolerance)
        & (evidence.recurrence_tail_bound < tolerance)
    )
    return eqx.tree_at(lambda r: (r.evidence, r.successful), fine, (evidence, valid))
