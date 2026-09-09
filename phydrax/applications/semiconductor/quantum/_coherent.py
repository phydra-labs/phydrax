# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Coherent stationary two-terminal NEGF with independent spectral evidence.

Only two boundary-source columns are solved. No full Green inverse is formed.
At eta=0 the finite-band leads have exactly compact continuum support; all
real poles outside that support are separately counted and normalized with
lead tails. Adaptive decisions are eager preparation, not differentiable
branches. Fixed-energy complex solves admit native holomorphic derivatives;
conjugated charge/current observables require real-Frechet semantics.
"""

from __future__ import annotations

from itertools import pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .... import linalg as la
from ...._strict import StrictModule
from ....integration import adaptive_interval_callable, AdaptiveQuadraturePlan
from ....nonlinear import Brent, NonlinearTermination, scalar_root, ScalarRootProblem
from ._basis import ChainHamiltonian, KB, PLANCK, Q, selected_eigenpairs, TransverseModes
from ._leads import BoundStateOccupation, SemiInfiniteLead


class SpectralPoint(StrictModule):
    """Per-longitudinal-energy response; sources G[:,{L,R}] have units J^-1.

    injection has shape (cells,2), J^-1. correlation_diagonal is Gn in J^-1
    including all declared transverse occupations. particle_current_kernel
    is dimensionless and positive into the device from each reservoir;
    integrating dE/h gives particles/s. energy_current_kernel is J, includes
    transverse energies, and integrating dE/h gives W. No displacement
    current is included in this stationary backend.
    """

    energy: Array
    green_sources: Array
    self_energies: Array
    broadenings: Array
    injection: Array
    correlation_diagonal: Array
    occupied_energy_diagonal: Array
    transmission: Array
    particle_current_kernel: Array
    energy_current_kernel: Array
    bond_particle_current_kernel: Array
    linear_residual: Array
    causal: Array
    successful: Array


class CoherentDevice(StrictModule):
    hamiltonian: ChainHamiltonian
    left: SemiInfiniteLead
    right: SemiInfiniteLead
    transverse: TransverseModes

    def __init__(self, hamiltonian, left, right, *, transverse=None):
        if (
            not isinstance(hamiltonian, ChainHamiltonian)
            or not isinstance(left, SemiInfiniteLead)
            or not isinstance(right, SemiInfiniteLead)
        ):
            raise TypeError(
                "CoherentDevice requires a cell-basis chain and two analytic semi-infinite leads."
            )
        if (
            len(
                {
                    hamiltonian.energy_reference,
                    left.energy_reference,
                    right.energy_reference,
                }
            )
            != 1
        ):
            raise ValueError(
                "Device and reservoir energies must use one declared reference."
            )
        modes = TransverseModes() if transverse is None else transverse
        if not isinstance(modes, TransverseModes):
            raise TypeError("transverse must be TransverseModes or None.")
        if modes.offsets.size > hamiltonian.resources.max_modes:
            raise ValueError("Transverse modes exceed the resource limit.")
        # Two complex RHS, factors, output and integration channels, never N^2.
        if hamiltonian.size * 512 > hamiltonian.resources.workspace_bytes:
            raise ValueError("Selected-source workspace exceeds the resource limit.")
        self.hamiltonian, self.left, self.right, self.transverse = (
            hamiltonian,
            left,
            right,
            modes,
        )

    def with_potential(self, potential):
        return eqx.tree_at(
            lambda d: d.hamiltonian, self, self.hamiltonian.with_potential(potential)
        )

    def shifted(self, energy):
        return eqx.tree_at(
            lambda d: (d.hamiltonian, d.left, d.right),
            self,
            (
                self.hamiltonian.shifted(energy),
                self.left.shifted(energy),
                self.right.shifted(energy),
            ),
        )

    def retarded_operator(self, energy, *, eta=0.0):
        """Native scaled A=((E+i eta)I-H-Sigma)/q and two scalar embeddings."""
        h = self.hamiltonian
        sigma = jnp.stack(
            (
                self.left.self_energy(energy, eta=eta),
                self.right.self_energy(energy, eta=eta),
            )
        )
        diagonal = (jnp.asarray(energy) + 1j * eta - h.diagonal).at[0].add(-sigma[0]).at[
            -1
        ].add(-sigma[1]) / Q
        hopping = -h.off_diagonal.astype(diagonal.dtype) / Q
        return la.TridiagonalLinearOperator(
            hopping, diagonal, hopping, operator_id="quantum-open-retarded-chain"
        ), sigma

    def spectral(self, energy, *, eta=0.0):
        """Reusable two-column native factorization; eta is numerical only."""
        energy_ = jnp.asarray(energy)
        eta_ = jnp.asarray(eta)
        if energy_.shape != () or eta_.shape != ():
            raise ValueError("spectral accepts one scalar energy and scalar eta.")
        operator, sigma = self.retarded_operator(energy_, eta=eta_)
        n = self.hamiltonian.size
        rhs = (
            jnp.zeros((n, 2), dtype=operator.diagonal.dtype)
            .at[0, 0]
            .set(1)
            .at[-1, 1]
            .set(1)
        )
        budget = self.hamiltonian.resources.workspace_bytes
        policy = la.LinearSolvePolicy(
            materialization=la.MaterializationPolicy(max_entries=1, max_bytes=budget),
            resources=la.SolveResourcePolicy(
                factorization_bytes=budget, workspace_bytes=budget
            ),
        )
        prepared = la.prepare(
            la.LinearSystem(operator), policy, rhs_layout=la.RHSLayout((2,))
        )
        result = la.solve_many(prepared, rhs)
        columns = result.value / Q
        gamma = -2 * jnp.imag(sigma)
        injection = jnp.abs(columns) ** 2 * gamma[None, :]
        f = jnp.stack(
            (
                self.transverse.occupation(
                    energy, self.left.chemical_potential, self.left.temperature
                ),
                self.transverse.occupation(
                    energy, self.right.chemical_potential, self.right.temperature
                ),
            )
        )
        ef = jnp.stack(
            (
                self.transverse.occupied_energy(
                    energy, self.left.chemical_potential, self.left.temperature
                ),
                self.transverse.occupied_energy(
                    energy, self.right.chemical_potential, self.right.temperature
                ),
            )
        )
        correlation = jnp.sum(injection * f[None, :], axis=1)
        occupied_energy = jnp.sum(injection * ef[None, :], axis=1)
        endpoint_spectral = -2 * jnp.imag(jnp.stack((columns[0, 0], columns[-1, 1])))
        endpoint_density = jnp.stack((correlation[0], correlation[-1]))
        endpoint_energy = jnp.stack((occupied_energy[0], occupied_energy[-1]))
        currents = gamma * (f * endpoint_spectral - endpoint_density)
        energy_currents = gamma * (ef * endpoint_spectral - endpoint_energy)
        # Gn_(i+1,i), evaluated from the same two injected source columns.
        adjacent = jnp.sum(
            columns[1:, :] * jnp.conj(columns[:-1, :]) * (gamma * f)[None, :], axis=1
        )
        bonds = -2 * self.hamiltonian.off_diagonal * jnp.imag(adjacent)
        transmission = gamma[0] * gamma[1] * jnp.abs(columns[-1, 0]) ** 2
        residual = jnp.max(jnp.abs(operator.mv_block(result.value) - rhs))
        causal = jnp.all(gamma >= -1e-12 * Q) & jnp.all(endpoint_spectral >= -1e-10 / Q)
        return SpectralPoint(
            energy=jnp.asarray(energy),
            green_sources=columns,
            self_energies=sigma,
            broadenings=gamma,
            injection=injection,
            correlation_diagonal=correlation,
            occupied_energy_diagonal=occupied_energy,
            transmission=transmission,
            particle_current_kernel=currents,
            energy_current_kernel=energy_currents,
            bond_particle_current_kernel=bonds,
            linear_residual=residual,
            causal=causal,
            successful=jnp.all(result.successful)
            & causal
            & (residual < 1e-8)
            & jnp.isfinite(eta_)
            & (eta_ >= 0),
        )


class BoundStates(StrictModule):
    energies: Array
    device_vectors: Array
    device_weights: Array
    electron_counts: Array
    electron_energy: Array
    spectral_weights: Array
    threshold_energy_resolution: Array
    successful: Array
    preparation: str = eqx.field(static=True)


def _real_pencil(device, energy):
    h = device.hamiltonian
    d = (
        h.diagonal.at[0]
        .add(jnp.real(device.left.self_energy(energy)))
        .at[-1]
        .add(jnp.real(device.right.self_energy(energy)))
    )
    return (d - energy) / Q, h.off_diagonal / Q


def _inertia(device, energy):
    """Sturm inertia of the real Schur complement in a lead spectral gap."""
    diagonal, off = _real_pencil(device, energy)
    d, o = np.asarray(diagonal), np.asarray(off)
    tiny = np.finfo(d.dtype).eps * max(1.0, float(np.max(np.abs(d))))
    pivot = float(d[0])
    negative = 0
    for i in range(d.size):
        if abs(pivot) < tiny:
            pivot = -tiny  # well-defined right-continuous Sturm limit, not eta
        negative += pivot < 0
        if i + 1 < d.size:
            pivot = float(d[i + 1]) - float(o[i]) ** 2 / pivot
    return int(negative)


def _determinant(device, energy):
    diagonal, off = _real_pencil(device, energy)

    def step(state, row):
        previous, current = state
        d, t = row
        following = d * current - t * t * previous
        normalization = jnp.maximum(
            jnp.maximum(jnp.abs(current), jnp.abs(following)), 1e-100
        )
        return (current / normalization, following / normalization), None

    (_, determinant), _ = jax.lax.scan(
        step, (jnp.asarray(1.0), diagonal[0]), (diagonal[1:], off)
    )
    return determinant


def _continuum_bands(device):
    bands = sorted(
        (
            tuple(np.asarray(device.left.band(), dtype=float)),
            tuple(np.asarray(device.right.band(), dtype=float)),
        )
    )
    if bands[0][1] >= bands[1][0]:
        return [(bands[0][0], max(bands[0][1], bands[1][1]))]
    return bands


def bound_states(device, occupation=None):
    """Resolve real poles in all common lead gaps by Sturm counting.

    Root isolation and native scalar solves are bounded. The pole vector is
    normalized with I-dSigma/dE, including the actual semi-infinite evanescent
    tail. Contact injection never supplies its occupation. A missing explicit
    preparation raises if a pole is resolved. The returned machine-scaled
    threshold_energy_resolution separates square-integrable poles from
    nonnormalizable band-edge states; integrate_coherent independently bounds
    any unresolved spectral weight with the local completeness sum rule.
    """
    h, resources = device.hamiltonian, device.hamiltonian.resources
    bands = _continuum_bands(device)
    radii = np.zeros(h.size)
    radii[:-1] += np.abs(np.asarray(h.off_diagonal))
    radii[1:] += np.abs(np.asarray(h.off_diagonal))
    radii[0] += abs(float(device.left.coupling))
    radii[-1] += abs(float(device.right.coupling))
    lower = (
        min(
            float(np.min(np.asarray(h.diagonal) - radii)),
            bands[0][0],
            float(
                device.left.onsite - abs(device.left.hopping) - abs(device.left.coupling)
            ),
            float(
                device.right.onsite
                - abs(device.right.hopping)
                - abs(device.right.coupling)
            ),
        )
        - Q
    )
    upper = (
        max(
            float(np.max(np.asarray(h.diagonal) + radii)),
            bands[-1][1],
            float(
                device.left.onsite + abs(device.left.hopping) + abs(device.left.coupling)
            ),
            float(
                device.right.onsite
                + abs(device.right.hopping)
                + abs(device.right.coupling)
            ),
        )
        + Q
    )
    resolution = (
        32
        * np.finfo(np.asarray(h.diagonal).dtype).eps
        * max(Q, *(upper_band - lower_band for lower_band, upper_band in bands))
    )
    gaps = [(lower, bands[0][0] - resolution)]
    gaps.extend((a[1] + resolution, b[0] - resolution) for a, b in pairwise(bands))
    gaps.append((bands[-1][1] + resolution, upper))
    pending = [(a, b, _inertia(device, a), _inertia(device, b)) for a, b in gaps if a < b]
    brackets = []
    subdivisions = 0
    while pending:
        a, b, na, nb = pending.pop()
        if nb < na:
            raise ValueError("Nonmonotone gap inertia: bound-state admission failed.")
        if nb == na:
            continue
        if len(brackets) + nb - na > resources.max_bound_states:
            raise ValueError("Bound-state capacity exhausted; increase max_bound_states.")
        if nb - na == 1:
            brackets.append((a, b))
        else:
            middle = (a + b) / 2
            nm = _inertia(device, middle)
            pending.extend(((a, middle, na, nm), (middle, b, nm, nb)))
            subdivisions += 1
            if subdivisions > resources.max_evaluations or middle == a or middle == b:
                raise ValueError(
                    "Bound-state isolation exhausted its finite resolution budget."
                )
    if brackets and occupation is None:
        raise ValueError(
            "Coherent leads leave bound-state occupation undetermined; supply BoundStateOccupation."
        )
    if occupation is not None and not isinstance(occupation, BoundStateOccupation):
        raise TypeError("Bound occupations require an explicit physical preparation.")
    energies, vectors, successful = [], [], jnp.asarray(True)
    for a, b in sorted(brackets):
        root = scalar_root(
            ScalarRootProblem(
                lambda e, _: _determinant(device, e * Q), bracket=(a / Q, b / Q)
            ),
            method=Brent(),
            termination=NonlinearTermination(
                absolute_residual=1e-11, relative_residual=0.0, maximum_steps=160
            ),
        )
        energy = root.root * Q
        d, _ = _real_pencil(device, energy)
        effective = eqx.tree_at(lambda x: x.diagonal, h, d * Q)
        pair = selected_eigenpairs(effective, 1, which="smallest-magnitude")
        vector = pair.eigenvectors[:, 0]
        derivative_left = jax.grad(lambda e: jnp.real(device.left.self_energy(e)))(energy)
        derivative_right = jax.grad(lambda e: jnp.real(device.right.self_energy(e)))(
            energy
        )
        norm = (
            1
            - derivative_left * jnp.abs(vector[0]) ** 2
            - derivative_right * jnp.abs(vector[-1]) ** 2
        )
        vector = vector / jnp.sqrt(norm)
        energies.append(energy)
        vectors.append(vector)
        successful = (
            successful
            & root.successful
            & pair.successful
            & (norm >= 1)
            & (jnp.abs(pair.eigenvalues[0]) < 1e-8)
        )
    energy_array = (
        jnp.stack(energies) if energies else jnp.zeros((0,), dtype=h.diagonal.dtype)
    )
    vector_array = (
        jnp.stack(vectors, axis=1)
        if vectors
        else jnp.zeros((h.size, 0), dtype=h.diagonal.dtype)
    )
    weights = jnp.abs(vector_array) ** 2
    if occupation is None:
        counts = jnp.zeros(h.size)
        energies_local = jnp.zeros(h.size)
    else:
        counts = jnp.sum(
            weights
            * device.transverse.occupation(
                energy_array, occupation.chemical_potential, occupation.temperature
            )[None, :],
            axis=1,
        )
        energies_local = jnp.sum(
            weights
            * device.transverse.occupied_energy(
                energy_array, occupation.chemical_potential, occupation.temperature
            )[None, :],
            axis=1,
        )
    return BoundStates(
        energies=energy_array,
        device_vectors=vector_array,
        device_weights=jnp.sum(weights, axis=0),
        electron_counts=counts,
        electron_energy=energies_local,
        spectral_weights=jnp.sum(weights, axis=1),
        threshold_energy_resolution=jnp.asarray(resolution),
        successful=successful,
        preparation="no resolved bound poles"
        if occupation is None
        else occupation.preparation,
    )


class CoherentEvidence(StrictModule):
    """Independent dimensionless scaled errors, not empirical qualification.

    Current and bond errors multiply q/h for particles/s; energy error
    multiplies q^2/h for W after subtracting a common energy origin.
    continuum_tail_bound is exactly zero for physical eta=0 finite-band
    leads, not a statement about an unbounded parabolic continuum model.
    """

    quadrature_successful: Array
    quadrature_error: Array
    refinement_error: Array
    spectral_sum_error: Array
    current_conservation_error: Array
    energy_conservation_error: Array
    bond_current_error: Array
    broadening_error: Array
    bound_states_successful: Array
    valid: Array
    continuum_bounds: Array
    numerical_eta: Array
    evaluations: Array
    continuum_tail_bound: Array
    empirical_qualified: bool = eqx.field(static=True, default=False)


class CoherentResult(StrictModule):
    """Stationary lead observables and reference-dependent spectral energy.

    energy_density is the occupied H moment, NOT thermodynamic carrier
    internal energy. It includes -q*psi when present in H and must not be
    added unchanged to electrostatic field energy. energy_currents and
    heat_currents are physical lead fluxes into the device in W.
    """

    electron_density: Array
    charge_density: Array
    energy_density: Array
    particle_currents: Array
    terminal_currents: Array
    energy_currents: Array
    heat_currents: Array
    bond_particle_currents: Array
    bound_states: BoundStates
    evidence: CoherentEvidence
    quadrature: tuple
    successful: Array


def _integral(device, *, eta, tolerance, panels, extra_breakpoints):
    """Integrate scaled positive spectral weights and physical moment kernels."""
    h = device.hamiltonian
    n = h.size
    bands = _continuum_bands(device)
    lo, hi = bands[0][0] / Q, bands[-1][1] / Q
    origin = (lo + hi) / 2
    # Centering the energy moment makes all numerical decisions gauge invariant.
    origin_j = origin * Q
    points = {float(x) for x in np.linspace(lo, hi, panels + 1)[1:-1]}
    points.update(float(v) / Q for band in bands for v in band if lo < v / Q < hi)
    points.update(float(v) / Q for v in extra_breakpoints if lo < float(v) / Q < hi)
    for lead in (device.left, device.right):
        for offset in np.asarray(device.transverse.offsets):
            for thermal_width in (-8.0, 0.0, 8.0):
                edge = (
                    float(
                        lead.chemical_potential
                        - offset
                        + thermal_width * KB * lead.temperature
                    )
                    / Q
                )
                if lo < edge < hi:
                    points.add(edge)
    initial_cells = len(points) + 1
    # Conservative array-workspace admission for simultaneous quadrature
    # nodes, selected complex columns and all retained interval moments.
    estimated_workspace = (
        initial_cells * 21 * n * 1024 + h.resources.max_intervals * (4 * n + 4) * 32
    )
    if estimated_workspace > h.resources.workspace_bytes:
        raise ValueError("Energy quadrature exceeds the declared workspace bound.")
    plan = AdaptiveQuadraturePlan(
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
        max_intervals=h.resources.max_intervals,
        max_evaluations=h.resources.max_evaluations,
        breakpoints=tuple(sorted(p - origin for p in points)),
        collect_partition=True,
        throw=False,
    )

    def one(value):
        point = device.spectral((value + origin) * Q, eta=eta)
        density = point.correlation_diagonal * Q / (2 * jnp.pi)
        energy = (
            point.occupied_energy_diagonal - origin_j * point.correlation_diagonal
        ) / (2 * jnp.pi)
        spectral = jnp.sum(point.injection, axis=1) * Q / (2 * jnp.pi)
        current = point.particle_current_kernel
        energy_current = (point.energy_current_kernel - origin_j * current) / Q
        bad = jnp.where(point.successful, 0.0, jnp.nan)
        return jnp.concatenate(
            (
                density,
                energy,
                spectral,
                current,
                energy_current,
                point.bond_particle_current_kernel,
                jnp.asarray([bad]),
            )
        )

    def supported(value):
        energy = (value + origin) * Q
        active = (eta > 0) | jnp.any(
            jnp.stack([(energy > lower) & (energy < upper) for lower, upper in bands])
        )
        return jax.lax.cond(
            active, one, lambda _: jnp.zeros((4 * n + 4,), dtype=h.diagonal.dtype), value
        )

    estimate = adaptive_interval_callable(
        jax.vmap(supported), jnp.asarray([lo - origin, hi - origin]), plan
    )
    values = jnp.asarray(estimate.value)
    return values, estimate, jnp.asarray([lo * Q, hi * Q]), origin_j


def integrate_coherent(
    device,
    *,
    bound_occupation=None,
    tolerance=1e-6,
    spectral_tolerance=2e-4,
    numerical_eta=0.0,
    initial_panels=16,
    max_refinements=3,
    resonance_energies=(),
):
    """Adaptive continuum + true bound charge, with independent refinement.

    The whole finite lead-band window is integrated: omitted continuum tails
    are exactly zero for the declared eta=0 tight-binding model. User supplied
    resonance seeds supplement, but never replace, spectral completeness.
    A separate refined evaluation and the local spectral sum rule catch
    narrow resonances missed by an embedded quadrature estimate. Finite eta
    is only a convergence experiment against the physical eta=0 result;
    the returned charge and currents always use eta=0.
    """
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0
        or not np.isfinite(spectral_tolerance)
        or spectral_tolerance <= 0
        or not np.isfinite(numerical_eta)
        or numerical_eta < 0
    ):
        raise ValueError(
            "Tolerances must be finite positive and numerical eta finite nonnegative."
        )
    if (
        isinstance(initial_panels, bool)
        or not isinstance(initial_panels, (int, np.integer))
        or initial_panels < 2
        or isinstance(max_refinements, bool)
        or not isinstance(max_refinements, (int, np.integer))
        or max_refinements < 1
    ):
        raise ValueError(
            "initial_panels and max_refinements must be bounded positive integers."
        )
    resonance = np.asarray(resonance_energies)
    if resonance.ndim != 1 or np.any(~np.isfinite(resonance)):
        raise ValueError("resonance_energies must be one finite energy vector.")
    states = bound_states(device, bound_occupation)
    n = device.hamiltonian.size
    r = device.hamiltonian.resources
    previous = None
    estimates = []
    refinement_error = jnp.asarray(jnp.inf)
    values = None
    for level in range(max_refinements + 1):
        panels = initial_panels * 2**level
        if panels + 4 + len(resonance_energies) > r.max_intervals:
            break
        values, estimate, bounds, origin = _integral(
            device,
            eta=0.0,
            tolerance=tolerance / (2**level),
            panels=panels,
            extra_breakpoints=resonance_energies,
        )
        estimates.append(estimate)
        used_panels = panels
        spectral_error = jnp.max(
            jnp.abs(values[2 * n : 3 * n] + states.spectral_weights - 1)
        )
        if previous is not None:
            refinement_error = jnp.max(jnp.abs(values - previous))
            if bool(
                estimate.successful
                & (spectral_error <= spectral_tolerance)
                & (refinement_error <= 4 * tolerance)
            ):
                break
        previous = values
    if values is None or len(estimates) < 2:
        raise ValueError(
            "Resource bounds must permit at least two independent energy refinements."
        )
    broadening_error = jnp.asarray(0.0)
    if numerical_eta > 0:
        for eta in (numerical_eta, numerical_eta / 2):
            regularized, eta_estimate, _, _ = _integral(
                device,
                eta=eta,
                tolerance=tolerance,
                panels=used_panels,
                extra_breakpoints=resonance_energies,
            )
            estimates.append(eta_estimate)
            broadening_error = jnp.maximum(
                broadening_error, jnp.max(jnp.abs(regularized - values))
            )
    counts = values[:n] + states.electron_counts
    energy = values[n : 2 * n] * Q + origin * values[:n] + states.electron_energy
    particle = values[3 * n : 3 * n + 2] * Q / PLANCK
    energy_current = values[3 * n + 2 : 3 * n + 4] * Q * Q / PLANCK + origin * particle
    bonds = values[3 * n + 4 : 4 * n + 3] * Q / PLANCK
    current_error = jnp.abs(jnp.sum(values[3 * n : 3 * n + 2]))
    energy_error = jnp.abs(jnp.sum(values[3 * n + 2 : 3 * n + 4]))
    bond_error = jnp.max(
        jnp.abs(values[3 * n + 4 : 4 * n + 3] - values[3 * n]), initial=0.0
    )
    quadrature_ok = jnp.all(jnp.stack([jnp.asarray(e.successful) for e in estimates]))
    finite = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(counts))
    valid = (
        finite
        & quadrature_ok
        & states.successful
        & (spectral_error <= spectral_tolerance)
        & (refinement_error <= 4 * tolerance)
        & (current_error <= 4 * tolerance)
        & (energy_error <= 4 * tolerance)
        & (bond_error <= 4 * tolerance)
        & (broadening_error <= spectral_tolerance)
    )
    evidence = CoherentEvidence(
        quadrature_successful=quadrature_ok,
        quadrature_error=jnp.max(
            jnp.stack([jnp.asarray(e.error_estimate) for e in estimates])
        ),
        refinement_error=refinement_error,
        spectral_sum_error=spectral_error,
        current_conservation_error=current_error,
        energy_conservation_error=energy_error,
        bond_current_error=bond_error,
        broadening_error=broadening_error,
        bound_states_successful=states.successful,
        valid=valid,
        continuum_bounds=bounds,
        numerical_eta=jnp.asarray(numerical_eta),
        evaluations=jnp.sum(
            jnp.stack([e.diagnostics.num_evaluations for e in estimates])
        ),
        continuum_tail_bound=jnp.asarray(0.0),
    )
    volume = device.hamiltonian.cell_volumes
    return CoherentResult(
        electron_density=counts / volume,
        charge_density=-Q * counts / volume,
        energy_density=energy / volume,
        particle_currents=particle,
        terminal_currents=-Q * particle,
        energy_currents=energy_current,
        heat_currents=energy_current
        - jnp.stack((device.left.chemical_potential, device.right.chemical_potential))
        * particle,
        bond_particle_currents=bonds,
        bound_states=states,
        evidence=evidence,
        quadrature=tuple(estimates),
        successful=valid,
    )
