# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""History-aware coherent transport by explicit, refinable lead dilation.

The scalar semi-infinite tight-binding leads are truncated in *space*, with
all lead/device coherences retained and unitary evolution of the complete
one-body correlation. Eliminating these sites gives a causal convolution,
not a Lindblad equation. Results are valid only before the reported return
window and after independent lead-size and memory-kernel refinement. This
bounded O((N+2L)^2) storage backend is neither general interacting transient
NEGF nor an atomistic model. Drives are explicitly piecewise constant.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .... import linalg as la
from ...._strict import StrictModule
from ....dynamics import DiscreteEvolution, DiscreteSystem, evolve, StateLayout, TimeGrid
from ....ein import contract
from ....linalg import eigen
from ._basis import _array, HBAR, KB, Q
from ._coherent import CoherentDevice


def _hermitian_spectrum(matrix, resources):
    """Explicit bounded dense *Hamiltonian*, never a dense Green inverse."""
    n = matrix.shape[0]
    op = la.DenseLinearOperator(matrix / Q)
    op = eqx.tree_at(
        lambda x: x.properties,
        op,
        la.OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "construction"}
        ),
    )
    result = eigen.eigensolve(
        eigen.Eigenproblem(op),
        policy=eigen.EigenSolvePolicy(
            eigen.DenseEigh(),
            count=n,
            which="smallest-algebraic",
            tolerance=eigen.EigenTolerancePolicy(
                relative=1e-10, absolute=1e-11, orthogonality=1e-9
            ),
            resources=eigen.EigenResourcePolicy(
                workspace_bytes=resources.workspace_bytes,
                preparation_bytes=resources.workspace_bytes,
                krylov_basis_bytes=resources.workspace_bytes,
            ),
            materialization=la.MaterializationPolicy(
                max_entries=n * n, max_bytes=resources.workspace_bytes
            ),
        ),
    )
    return result.eigenvalues * Q, result.eigenvectors, result.successful


def _finite_hamiltonian(device, sites, device_shift=None, lead_shift=None):
    h = device.hamiltonian
    ds = jnp.zeros(h.size) if device_shift is None else device_shift
    ls = jnp.zeros(2) if lead_shift is None else lead_shift
    diagonal = jnp.concatenate(
        (
            jnp.full(sites, device.left.onsite + ls[0]),
            h.diagonal + ds,
            jnp.full(sites, device.right.onsite + ls[1]),
        )
    )
    off = jnp.concatenate(
        (
            jnp.full(sites - 1, device.left.hopping),
            jnp.asarray([device.left.coupling]),
            h.off_diagonal,
            jnp.asarray([device.right.coupling]),
            jnp.full(sites - 1, device.right.hopping),
        )
    )
    return jnp.diag(diagonal) + jnp.diag(off, 1) + jnp.diag(off, -1)


def _lead_modes(lead, sites):
    k = jnp.pi * jnp.arange(1, sites + 1) / (sites + 1)
    vectors = jnp.sqrt(2 / (sites + 1)) * jnp.sin(
        jnp.arange(1, sites + 1)[:, None] * k[None, :]
    )
    return lead.onsite + 2 * lead.hopping * jnp.cos(k), vectors


def lead_memory_kernel(lead, times, *, sites):
    """Retarded elimination kernel K(t)=V exp(-i Hlead t/hbar) V†/hbar².

    Units s^-2; the amplitude equation contains -integral K(t-s) a(s) ds.
    Uniform lead drives multiply this by exp[-i integral shift(s)ds/hbar].
    Negative lags are exactly zero by causality, not periodically wrapped.
    """
    if isinstance(sites, bool) or not isinstance(sites, (int, np.integer)) or sites < 2:
        raise ValueError("sites must be an integer of at least two.")
    t = _array(times, "memory-kernel times")
    energies, vectors = _lead_modes(lead, sites)
    weights = lead.coupling**2 * vectors[0] ** 2 / HBAR**2
    value = contract(
        "k,...k->...",
        weights,
        jnp.exp(-1j * t[..., None] * energies / HBAR),
        backend="jax",
    )
    return jnp.where(t >= 0, value, 0j)


class QuantumInitialState(StrictModule):
    """Preparation per transverse channel; degeneracy is applied only in observables.

    Partitioned preparation has thermal disconnected leads and an explicit
    device density matrix with eigenvalues in [0,1]. Contacts are switched
    on at t0. Equilibrium preparation is the correlated, connected grand-
    canonical state at common lead mu/T; no initial-slip approximation.
    """

    device_correlation: Array | None
    preparation: str = eqx.field(static=True)

    def __init__(self, *, preparation, device_correlation=None):
        if preparation not in ("partitioned", "equilibrium"):
            raise ValueError("Preparation must be partitioned or equilibrium.")
        if (preparation == "partitioned") != (device_correlation is not None):
            raise ValueError(
                "Only partitioned preparation requires explicit device correlation."
            )
        value = (
            None
            if device_correlation is None
            else jnp.asarray(device_correlation, dtype=complex)
        )
        if value is not None and (
            value.ndim != 3
            or value.shape[-1] != value.shape[-2]
            or not np.all(np.isfinite(np.asarray(value)))
        ):
            raise ValueError(
                "Initial device correlation has shape (transverse modes, cells, cells)."
            )
        if value is not None and not np.allclose(
            np.asarray(value),
            np.asarray(value).conj().swapaxes(-1, -2),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("Initial device correlation must be Hermitian.")
        self.device_correlation, self.preparation = value, preparation


class QuantumPulse(StrictModule):
    """Exact held energy shifts (J), including lead band shifts, on times (s)."""

    times: Array
    device_energy_shifts: Array
    lead_energy_shifts: Array

    def __init__(self, times, device_energy_shifts, lead_energy_shifts):
        t = _array(times, "pulse times")
        d = _array(device_energy_shifts, "device energy shifts")
        l = _array(lead_energy_shifts, "lead energy shifts")
        if t.ndim != 1 or t.size < 2 or np.any(np.diff(np.asarray(t)) <= 0):
            raise ValueError("Pulse times must be strictly increasing.")
        if d.ndim != 2 or d.shape[0] != t.size - 1 or l.shape != (t.size - 1, 2):
            raise ValueError(
                "Each pulse interval requires cell shifts and two lead shifts."
            )
        self.times, self.device_energy_shifts, self.lead_energy_shifts = t, d, l


class QuantumMemoryEvidence(StrictModule):
    lead_sites: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    recurrence_time: Array
    recurrence_valid: Array
    kernel_error: Array
    reservoir_error: Array
    number_error: Array
    unitary_error: Array
    energy_error: Array
    eigen_successful: Array
    preparation: str = eqx.field(static=True)
    closure: str = eqx.field(
        static=True,
        default="unitary finite spatial lead dilation; coherent scalar chain; held drive",
    )


class QuantumTransientResult(StrictModule):
    times: Array
    electron_density: Array
    charge_density: Array
    terminal_currents: Array
    energy_currents: Array
    heat_currents: Array
    device_charge_rate: Array
    transferred_particles: Array
    transferred_energy: Array
    transferred_heat: Array
    stored_energy: Array
    switching_work: Array
    final_correlation: Array
    evidence: QuantumMemoryEvidence
    trajectory: object
    successful: Array


def _initial_correlation(device, initial, sites):
    n, nm = device.hamiltonian.size, device.transverse.offsets.size
    dim = n + 2 * sites
    r = device.hamiltonian.resources
    if initial.preparation == "equilibrium":
        if float(device.left.chemical_potential) != float(
            device.right.chemical_potential
        ) or float(device.left.temperature) != float(device.right.temperature):
            raise ValueError(
                "Correlated equilibrium preparation needs common lead mu and T."
            )
        energy, vectors, ok = _hermitian_spectrum(_finite_hamiltonian(device, sites), r)
        f = jax.nn.sigmoid(
            (
                device.left.chemical_potential
                - energy[None, :]
                - device.transverse.offsets[:, None]
            )
            / (KB * device.left.temperature)
        )
        return contract("ik,mk,jk->mij", vectors, f, vectors.conj(), backend="jax"), ok
    rho = initial.device_correlation
    if rho.shape != (nm, n, n):
        raise ValueError(
            "Initial device correlation does not match the admitted mode/cell populations."
        )
    valid = jnp.asarray(True)
    for mode in rho:
        occupations, _, ok = _hermitian_spectrum(mode * Q, r)
        if np.any(np.asarray(occupations / Q) < -1e-10) or np.any(
            np.asarray(occupations / Q) > 1 + 1e-10
        ):
            raise ValueError(
                "Initial fermion correlation eigenvalues must lie in [0,1]; no clipping is applied."
            )
        valid = valid & ok
    result = (
        jnp.zeros((nm, dim, dim), dtype=complex)
        .at[:, sites : sites + n, sites : sites + n]
        .set(rho)
    )
    for lead, start in ((device.left, 0), (device.right, sites + n)):
        energy, vectors = _lead_modes(lead, sites)
        f = jax.nn.sigmoid(
            (
                lead.chemical_potential
                - energy[None, :]
                - device.transverse.offsets[:, None]
            )
            / (KB * lead.temperature)
        )
        block = contract("ik,mk,jk->mij", vectors, f, vectors, backend="jax")
        result = result.at[:, start : start + sites, start : start + sites].set(block)
    return result, valid


def _transient_once(device, initial, pulse, sites):
    n, nm = device.hamiltonian.size, device.transverse.offsets.size
    dim, steps = n + 2 * sites, pulse.times.size - 1
    resources = device.hamiltonian.resources
    workspace = dim * dim * 16 * (nm * (steps + 5) + 3 * steps + 16)
    if dim > resources.max_nodes or workspace > resources.workspace_bytes:
        raise ValueError(
            "Finite-memory correlation/history exceeds declared quantum resources."
        )
    if steps > resources.max_evaluations:
        raise ValueError("Pulse interval count exceeds declared quantum evaluations.")
    if pulse.device_energy_shifts.shape[1] != n:
        raise ValueError("Pulse device shifts must match the cell basis.")
    rho0, initial_ok = _initial_correlation(device, initial, sites)
    hamiltonians, unitaries, eigen_valid = [], [], []
    for k in range(steps):
        h = _finite_hamiltonian(
            device, sites, pulse.device_energy_shifts[k], pulse.lead_energy_shifts[k]
        )
        # Remove one scalar reference before diagonalization/phase formation.
        origin = jnp.mean(jnp.diag(h))
        values, vectors, ok = _hermitian_spectrum(h - origin * jnp.eye(dim), resources)
        phase = jnp.exp(-1j * values * (pulse.times[k + 1] - pulse.times[k]) / HBAR)
        u = (vectors * phase[None, :]) @ vectors.conj().T
        hamiltonians.append(h)
        unitaries.append(u)
        eigen_valid.append(ok)
    hs, us = jnp.stack(hamiltonians), jnp.stack(unitaries)

    def transition(context, state, args):
        rho = state[0] + 1j * state[1]
        u = args[context.step_index]
        advanced = u[None, :, :] @ rho @ u.conj().T[None, :, :]
        return jnp.stack((jnp.real(advanced), jnp.imag(advanced)))

    layout = StateLayout(
        (2, nm, dim, dim),
        axes=("complex_part", "transverse_mode", "row", "column"),
        layout_id="quantum-full-lead-correlation",
    )
    system = DiscreteSystem(
        transition, state_layout=layout, system_id="unitary-held-quantum-lead-memory"
    )
    # Native evolution uses dimensionless time to avoid SI absolute-step checks.
    trajectory = evolve(
        DiscreteEvolution(system),
        jnp.stack((jnp.real(rho0), jnp.imag(rho0))),
        TimeGrid(
            (pulse.times - pulse.times[0]) * Q / HBAR,
            time_id="quantum-time-in-hbar-per-eV",
        ),
        args=us,
    )
    rhos = trajectory.states[:, 0] + 1j * trajectory.states[:, 1]
    degeneracy = device.transverse.degeneracies
    total_rho = contract("m,tmij->tij", degeneracy, rhos, backend="jax")
    counts = jnp.real(jnp.diagonal(total_rho, axis1=-2, axis2=-1))
    density = counts[:, sites : sites + n] / device.hamiltonian.cell_volumes
    lead_counts = jnp.stack(
        (jnp.sum(counts[:, :sites], axis=-1), jnp.sum(counts[:, sites + n :], axis=-1)),
        axis=-1,
    )
    transfer = -(lead_counts - lead_counts[0])
    obs_h = jnp.concatenate((hs, hs[-1:]), axis=0)
    drho = -1j * (obs_h @ total_rho - total_rho @ obs_h) / HBAR
    lead_rate = jnp.stack(
        (
            jnp.real(jnp.trace(drho[:, :sites, :sites], axis1=-2, axis2=-1)),
            jnp.real(jnp.trace(drho[:, sites + n :, sites + n :], axis1=-2, axis2=-1)),
        ),
        axis=-1,
    )
    particle_current = -lead_rate
    terminal_current = -Q * particle_current
    energy_current, energy_transfer = [], []
    for start in (0, sites + n):
        section = slice(start, start + sites)
        current = -jnp.real(
            contract(
                "tij,tji->t",
                obs_h[:, section, section],
                drho[:, section, section],
                backend="jax",
            )
        )
        # Transverse kinetic energy belongs to the same transferred population.
        mode_drho = -1j * (obs_h[:, None] @ rhos - rhos @ obs_h[:, None]) / HBAR
        mode_rate = jnp.real(
            jnp.trace(mode_drho[:, :, section, section], axis1=-2, axis2=-1)
        )
        current -= contract(
            "m,tm->t", degeneracy * device.transverse.offsets, mode_rate, backend="jax"
        )
        delta = rhos[1:, :, section, section] - rhos[:-1, :, section, section]
        transferred = -jnp.real(
            contract(
                "m,tij,tmji->t", degeneracy, hs[:, section, section], delta, backend="jax"
            )
        )
        transferred -= contract(
            "m,tm->t",
            degeneracy * device.transverse.offsets,
            jnp.real(jnp.trace(delta, axis1=-2, axis2=-1)),
            backend="jax",
        )
        energy_current.append(current)
        energy_transfer.append(transferred)
    energy_current = jnp.stack(energy_current, axis=-1)
    energy_transfer = jnp.stack(energy_transfer, axis=-1)
    mu = jnp.asarray([device.left.chemical_potential, device.right.chemical_potential])
    held_mu = mu + pulse.lead_energy_shifts
    observed_mu = jnp.concatenate((held_mu, held_mu[-1:]), axis=0)
    heat_current = energy_current - observed_mu * particle_current
    heat_transfer = energy_transfer - held_mu * jnp.diff(transfer, axis=0)
    stored = jnp.real(contract("tij,tji->t", obs_h, total_rho, backend="jax"))
    stored += contract(
        "m,tm->t",
        degeneracy * device.transverse.offsets,
        jnp.real(jnp.trace(rhos, axis1=-2, axis2=-1)),
        backend="jax",
    )
    baseline = _finite_hamiltonian(device, sites)
    switching = jnp.concatenate(
        (
            jnp.real(jnp.trace((hs[0] - baseline) @ total_rho[0]))[None],
            jnp.real(
                contract("tij,tji->t", hs[1:] - hs[:-1], total_rho[1:-1], backend="jax")
            ),
            jnp.zeros(1),
        )
    )
    number_error = jnp.max(jnp.abs(jnp.sum(counts, axis=-1) - jnp.sum(counts[0])))
    unitary_error = jnp.max(jnp.abs(us @ us.conj().swapaxes(-1, -2) - jnp.eye(dim)))
    energy_error = (
        jnp.max(jnp.abs(stored - stored[0] - jnp.cumsum(switching) + switching[0])) / Q
    )
    recurrence = min(
        sites * HBAR / (2 * abs(float(lead.hopping)))
        for lead in (device.left, device.right)
    )
    evidence = QuantumMemoryEvidence(
        lead_sites=sites,
        workspace_bytes=workspace,
        recurrence_time=jnp.asarray(recurrence),
        recurrence_valid=(pulse.times[-1] - pulse.times[0] < recurrence),
        kernel_error=jnp.asarray(jnp.inf),
        reservoir_error=jnp.asarray(jnp.inf),
        number_error=number_error,
        unitary_error=unitary_error,
        energy_error=energy_error,
        eigen_successful=initial_ok & jnp.all(jnp.stack(eigen_valid)),
        preparation=initial.preparation,
    )
    valid = (
        evidence.eigen_successful
        & evidence.recurrence_valid
        & (number_error < 1e-8 * dim)
        & (unitary_error < 1e-8)
        & (energy_error < 1e-8 * dim)
    )
    return QuantumTransientResult(
        times=pulse.times,
        electron_density=density,
        charge_density=-Q * density,
        terminal_currents=terminal_current,
        energy_currents=energy_current,
        heat_currents=heat_current,
        device_charge_rate=-Q
        * jnp.real(
            jnp.trace(drho[:, sites : sites + n, sites : sites + n], axis1=-2, axis2=-1)
        ),
        transferred_particles=transfer,
        transferred_energy=energy_transfer,
        transferred_heat=heat_transfer,
        stored_energy=stored,
        switching_work=switching,
        final_correlation=rhos[-1],
        evidence=evidence,
        trajectory=trajectory,
        successful=valid,
    )


def solve_quantum_transient(device, initial, pulse, *, lead_sites=32, tolerance=1e-4):
    """Execute two lead dilations and compare physical observables and memory.

    Currents are charge/energy fluxes *into* the device; they are conduction
    currents, not a self-consistent Maxwell terminal solution. Their sum is
    the returned device_charge_rate. No displacement partition is invented.
    ``stored_energy`` owns the entire finite lead/device Hamiltonian (including
    contacts); switching_work is the exact external work at held-drive jumps.
    Energy/heat transfers are per interval, not approximate current integrals.
    """
    if (
        not isinstance(device, CoherentDevice)
        or not isinstance(initial, QuantumInitialState)
        or not isinstance(pulse, QuantumPulse)
    ):
        raise TypeError(
            "Transient transport needs a coherent device, explicit initial state and QuantumPulse."
        )
    if (
        isinstance(lead_sites, bool)
        or not isinstance(lead_sites, (int, np.integer))
        or lead_sites < 2
    ):
        raise ValueError("lead_sites must be an integer of at least two.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    sites = int(lead_sites)
    coarse = _transient_once(device, initial, pulse, sites)
    fine = _transient_once(device, initial, pulse, 2 * sites)
    count_error = jnp.max(
        jnp.abs(
            (fine.electron_density - coarse.electron_density)
            * device.hamiltonian.cell_volumes
        )
    )
    current_error = (
        jnp.max(jnp.abs(fine.terminal_currents - coarse.terminal_currents)) * HBAR / Q**2
    )
    transfer_error = jnp.max(
        jnp.abs(fine.transferred_particles - coarse.transferred_particles)
    )
    reservoir_error = jnp.maximum(count_error, jnp.maximum(current_error, transfer_error))
    # Dense lag sampling resolves the fastest lead oscillation; recurrence is
    # an independent guard, not inferred solely from a few kernel samples.
    duration = float(pulse.times[-1] - pulse.times[0])
    lag_count = max(
        65,
        int(
            np.ceil(
                duration
                * max(abs(float(device.left.hopping)), abs(float(device.right.hopping)))
                / HBAR
                * 16
            )
        )
        + 1,
    )
    if lag_count > device.hamiltonian.resources.max_evaluations:
        raise ValueError("Memory-kernel refinement sampling exceeds quantum resources.")
    lags = jnp.linspace(0.0, duration, lag_count)
    errors = []
    for lead in (device.left, device.right):
        errors.append(
            jnp.max(
                jnp.abs(
                    lead_memory_kernel(lead, lags, sites=sites)
                    - lead_memory_kernel(lead, lags, sites=2 * sites)
                )
            )
            * HBAR**2
            / lead.coupling**2
        )
    kernel_error = jnp.max(jnp.stack(errors))
    evidence = eqx.tree_at(
        lambda e: (e.kernel_error, e.reservoir_error),
        fine.evidence,
        (kernel_error, reservoir_error),
    )
    valid = (
        coarse.successful
        & fine.successful
        & (kernel_error < tolerance)
        & (reservoir_error < tolerance)
    )
    return eqx.tree_at(lambda r: (r.evidence, r.successful), fine, (evidence, valid))
