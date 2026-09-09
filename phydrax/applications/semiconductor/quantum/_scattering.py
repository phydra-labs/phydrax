# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Local optical-phonon Fock SCBA for orthogonal scalar effective-mass chains.

The bath is specified by an actual phonon energy, coupling and temperature.
No numerical eta, absorbing potential, mobility or SRH is added. The static
Hartree/tadpole displacement is excluded (the supplied band Hamiltonian owns
it). Different declared transverse modes scatter independently, not between
valleys. A uniform midpoint energy lattice makes every emission/absorption
pair use exactly the same measure; interpolation of shifted energies would
break this discrete conserving approximation.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._strict import StrictModule
from ....ein import contract
from ....nonlinear import FixedPointIteration, FixedPointProblem, NonlinearTermination
from ._basis import _array, KB, PLANCK, Q
from ._coherent import CoherentDevice


class OpticalPhononBath(StrictModule):
    energy: Array
    coupling: Array
    temperature: Array
    bath_id: str = eqx.field(static=True)

    def __init__(self, energy, coupling, temperature, *, bath_id):
        self.energy = _array(energy, "phonon energy", positive=True)
        self.coupling = _array(coupling, "local phonon coupling")
        self.temperature = _array(temperature, "phonon bath temperature", positive=True)
        if (
            self.energy.shape != ()
            or self.temperature.shape != ()
            or self.coupling.ndim > 1
        ):
            raise ValueError(
                "Phonon energy/temperature are scalars; coupling is scalar or per cell, in J."
            )
        if not isinstance(bath_id, str) or not bath_id.strip():
            raise ValueError("A physical phonon bath identity is required.")
        self.bath_id = bath_id

    @property
    def occupation(self):
        return 1 / jnp.expm1(self.energy / (KB * self.temperature))


class PhononEnergyGrid(StrictModule):
    """Finite midpoint lattice with integer phonon shift, no wraparound.

    ``lower`` and ``upper`` are bin edges in J. Refinement doubles both the
    number of bins and the phonon shift. Outside-grid correlation is zero;
    successful transport additionally requires window and mesh refinement.
    """

    lower: Array
    upper: Array
    points: int = eqx.field(static=True)
    phonon_bins: int = eqx.field(static=True)

    def __init__(self, lower, upper, *, points, phonon_energy):
        lo, hi = float(lower), float(upper)
        if (
            isinstance(points, bool)
            or not isinstance(points, (int, np.integer))
            or points < 8
        ):
            raise ValueError("points must be an integer of at least eight.")
        phonon = np.asarray(phonon_energy)
        if phonon.shape != () or not np.isfinite(phonon) or phonon <= 0:
            raise ValueError("phonon_energy must be one finite positive energy.")
        n = int(points)
        if not np.isfinite(lo + hi) or hi <= lo:
            raise ValueError("The energy window must be increasing and finite.")
        bins = float(phonon) / ((hi - lo) / n)
        shift = round(bins)
        if shift < 1 or shift >= n or abs(bins - shift) > 1e-9 * max(1, bins):
            raise ValueError(
                "The phonon energy must be a nonzero integer number of energy bins."
            )
        self.lower, self.upper = jnp.asarray(lo), jnp.asarray(hi)
        self.points, self.phonon_bins = n, shift

    @property
    def spacing(self):
        return (self.upper - self.lower) / self.points

    @property
    def energies(self):
        return self.lower + (jnp.arange(self.points) + 0.5) * self.spacing

    def refined(self, phonon_energy):
        return PhononEnergyGrid(
            self.lower, self.upper, points=2 * self.points, phonon_energy=phonon_energy
        )

    def expanded(self, phonon_energy):
        width = self.upper - self.lower
        return PhononEnergyGrid(
            self.lower - width / 2,
            self.upper + width / 2,
            points=2 * self.points,
            phonon_energy=phonon_energy,
        )


def _shift(values, bins):
    if bins > 0:  # value(E-phonon)
        return jnp.concatenate((jnp.zeros_like(values[:bins]), values[:-bins]), axis=0)
    bins = -bins
    return jnp.concatenate((values[bins:], jnp.zeros_like(values[:bins])), axis=0)


def _causal_real(gamma, spacing):
    """PV Hilbert transform of positive, piecewise-constant broadening.

    Log integration is exact for the declared bin representation. A row is
    evaluated at a time: O(energy_bins*cells) storage, never a square kernel.
    """
    n = gamma.shape[0]
    edges = (jnp.arange(n + 1) - 0.5) * spacing

    def row(i):
        x = i * spacing
        weights = jnp.log(jnp.abs((x - edges[:-1]) / (x - edges[1:]))) / (2 * jnp.pi)
        return contract("e,en->n", weights, gamma, backend="jax")

    return jax.lax.map(row, jnp.arange(n))


def _chain_correlations(diagonal, hopping, source_in, source_out):
    """Recursive selected diagonal/correlation blocks, O(cells) storage."""

    def sweep(d, t, sin, sout):
        def step(carry, values):
            previous_g, previous_n, previous_p = carry
            di, ti, ni, pi = values
            g = 1 / (di - ti * ti * previous_g)
            nn = jnp.abs(g) ** 2 * (ni + ti * ti * previous_n)
            pp = jnp.abs(g) ** 2 * (pi + ti * ti * previous_p)
            return (g, nn, pp), (g, nn, pp)

        return jax.lax.scan(step, (0j, 0.0, 0.0), (d, t, sin, sout))[1]

    left_t = jnp.concatenate((jnp.zeros(1), hopping))
    right_t = jnp.concatenate((hopping, jnp.zeros(1)))
    gl, nl, pl = sweep(diagonal, left_t, source_in, source_out)
    gr, nr, pr = tuple(
        x[::-1]
        for x in sweep(diagonal[::-1], right_t[::-1], source_in[::-1], source_out[::-1])
    )
    shift_l = lambda a: jnp.concatenate((jnp.zeros(1, dtype=a.dtype), a[:-1]))
    shift_r = lambda a: jnp.concatenate((a[1:], jnp.zeros(1, dtype=a.dtype)))
    g = 1 / (diagonal - left_t**2 * shift_l(gl) - right_t**2 * shift_r(gr))
    nn = jnp.abs(g) ** 2 * (
        source_in + left_t**2 * shift_l(nl) + right_t**2 * shift_r(nr)
    )
    pp = jnp.abs(g) ** 2 * (
        source_out + left_t**2 * shift_l(pl) + right_t**2 * shift_r(pr)
    )
    bond = hopping**2 * (
        nl[:-1] * (-2 * jnp.imag(gr[1:])) - (-2 * jnp.imag(gl[:-1])) * nr[1:]
    )
    bond /= jnp.abs(1 - hopping**2 * gl[:-1] * gr[1:]) ** 2
    return g, nn, pp, bond


class ScatteringEvidence(StrictModule):
    fixed_point_error: Array
    spectral_identity_error: Array
    spectral_sum_error: Array
    collision_particle_error: Array
    terminal_particle_error: Array
    terminal_energy_error: Array
    local_continuity_error: Array
    equilibrium_kms_error: Array
    causal: Array
    mesh_error: Array
    window_error: Array
    workspace_bytes: int = eqx.field(static=True)
    numerical_eta: float = eqx.field(static=True, default=0.0)
    closure: str = eqx.field(
        static=True,
        default="local optical-phonon Fock SCBA; independent transverse modes; no Hartree tadpole",
    )


class ScatteringResult(StrictModule):
    energies: Array
    retarded_self_energy: Array
    lesser_self_energy: Array
    greater_self_energy: Array
    spectral_density: Array
    electron_correlation: Array
    electron_density: Array
    charge_density: Array
    particle_currents: Array
    terminal_currents: Array
    energy_currents: Array
    heat_currents: Array
    bond_particle_currents: Array
    collision_particle_rate: Array
    collision_energy_rate: Array
    phonon_heat: Array
    emission_rate: Array
    absorption_rate: Array
    evidence: ScatteringEvidence
    nonlinear_results: tuple
    successful: Array


def _solve_grid(device, bath, grid, *, tolerance, maximum_steps, damping):
    h = device.hamiltonian
    n, ne = h.size, grid.points
    nm = device.transverse.offsets.size
    workspace = ne * n * nm * 8 * 40 + ne * 8 * 16
    if workspace > h.resources.workspace_bytes or ne * nm > h.resources.max_evaluations:
        raise ValueError("SCBA energy/correlation workspace exceeds QuantumResources.")
    coupling = jnp.broadcast_to(bath.coupling, (n,)) / Q
    if not np.isclose(
        float(grid.spacing * grid.phonon_bins / bath.energy), 1.0, rtol=1e-10
    ):
        raise ValueError("Energy-grid phonon shift does not match this bath.")
    origin = (grid.upper + grid.lower) / 2
    energies = grid.energies
    e = (energies - origin) / Q
    de = grid.spacing / Q
    leads = (device.left, device.right)
    sigma = jnp.stack(tuple(lead.self_energy(energies) / Q for lead in leads), axis=-1)
    gamma_lead = -2 * jnp.imag(sigma)
    diagonal = (e[:, None] - (h.diagonal - origin)[None, :] / Q).astype(complex)
    diagonal = diagonal.at[:, 0].add(-sigma[:, 0]).at[:, -1].add(-sigma[:, 1])
    hopping = h.off_diagonal / Q
    occupation = bath.occupation
    shift = grid.phonon_bins

    def self_energies(nn, pp):
        sin = coupling**2 * (
            occupation * _shift(nn, shift) + (occupation + 1) * _shift(nn, -shift)
        )
        sout = coupling**2 * (
            (occupation + 1) * _shift(pp, shift) + occupation * _shift(pp, -shift)
        )
        return sin, sout

    mode_data, nonlinear_results = [], []
    for offset in device.transverse.offsets:
        f = jnp.stack(
            tuple(
                jax.nn.sigmoid(
                    (lead.chemical_potential - energies - offset)
                    / (KB * lead.temperature)
                )
                for lead in leads
            ),
            axis=-1,
        )
        injected = (
            jnp.zeros((ne, n))
            .at[:, 0]
            .add(gamma_lead[:, 0] * f[:, 0])
            .at[:, -1]
            .add(gamma_lead[:, 1] * f[:, 1])
        )
        extracted = (
            jnp.zeros((ne, n))
            .at[:, 0]
            .add(gamma_lead[:, 0] * (1 - f[:, 0]))
            .at[:, -1]
            .add(gamma_lead[:, 1] * (1 - f[:, 1]))
        )

        def evaluate(state, injected=injected, extracted=extracted):
            sin, sout = state
            broadening = sin + sout
            sr = _causal_real(broadening, de) - 0.5j * broadening
            g, nn, pp, bond = jax.vmap(_chain_correlations, in_axes=(0, None, 0, 0))(
                diagonal - sr, hopping, injected + sin, extracted + sout
            )
            return sr, g, nn, pp, bond

        def mapping(state, args):
            _, _, nn, pp, _ = evaluate(state)
            return jnp.stack(self_energies(nn, pp))

        initial = jnp.zeros((2, ne, n))
        solved = FixedPointIteration(damping=damping).solve(
            FixedPointProblem(mapping, problem_id="optical-phonon-conserving-scba"),
            initial,
            termination=NonlinearTermination(
                absolute_residual=tolerance,
                relative_residual=tolerance,
                maximum_steps=maximum_steps,
            ),
        )
        state = solved.state
        sr, g, nn, pp, bond = evaluate(state)
        sin, sout = state
        spectral = -2 * jnp.imag(g)
        collision = sin * pp - sout * nn
        terminal = gamma_lead * (f * spectral[:, (0, n - 1)] - nn[:, (0, n - 1)])
        # Each upper/lower energy pair is counted once, with identical weights.
        emission = coupling**2 * (occupation + 1) * nn[shift:] * pp[:-shift]
        absorption = coupling**2 * occupation * nn[:-shift] * pp[shift:]
        expected = (
            jax.nn.sigmoid(
                (leads[0].chemical_potential - energies - offset)
                / (KB * leads[0].temperature)
            )[:, None]
            * spectral
        )
        kms = jnp.max(jnp.abs(nn - expected)) / (1 + jnp.max(jnp.abs(spectral)))
        mode_data.append(
            (
                sr,
                sin,
                sout,
                spectral,
                nn,
                pp,
                bond,
                collision,
                terminal,
                jnp.sum(emission, axis=0) * de * Q / PLANCK,
                jnp.sum(absorption, axis=0) * de * Q / PLANCK,
                jnp.max(jnp.abs(mapping(state, None) - state)),
                kms,
            )
        )
        nonlinear_results.append(solved)
    arrays = tuple(jnp.stack(tuple(mode[k] for mode in mode_data)) for k in range(13))
    (
        sr,
        sin,
        sout,
        spectral,
        nn,
        pp,
        bond,
        collision,
        terminal,
        emission,
        absorption,
        fixed_error,
        kms,
    ) = arrays
    degeneracies = device.transverse.degeneracies
    total = lambda value: contract("m,m...->...", degeneracies, value, backend="jax")
    counts = total(jnp.sum(nn, axis=1)) * de / (2 * jnp.pi)
    particles = total(jnp.sum(terminal, axis=1)) * de * Q / PLANCK
    energy = (
        total(
            jnp.sum(
                terminal
                * (energies[None, :, None] + device.transverse.offsets[:, None, None]),
                axis=1,
            )
        )
        * de
        * Q
        / PLANCK
    )
    cparticles = total(jnp.sum(collision, axis=1)) * de * Q / PLANCK
    cenergy = (
        total(
            jnp.sum(
                collision
                * (energies[None, :, None] + device.transverse.offsets[:, None, None]),
                axis=1,
            )
        )
        * de
        * Q
        / PLANCK
    )
    emit, absorb = total(emission), total(absorption)
    heat = bath.energy * (emit - absorb)
    bonds = total(jnp.sum(bond, axis=1)) * de * Q / PLANCK
    source = (
        collision.at[:, :, 0].add(terminal[:, :, 0]).at[:, :, -1].add(terminal[:, :, 1])
    )
    divergence = jnp.pad(bond, ((0, 0), (0, 0), (0, 1))) - jnp.pad(
        bond, ((0, 0), (0, 0), (1, 0))
    )
    rate_scale = Q / PLANCK
    power_scale = Q * rate_scale
    equilibrium = np.array_equal(
        np.asarray(leads[0].chemical_potential), np.asarray(leads[1].chemical_potential)
    ) and all(float(lead.temperature) == float(bath.temperature) for lead in leads)
    evidence = ScatteringEvidence(
        fixed_point_error=jnp.max(fixed_error),
        spectral_identity_error=jnp.max(jnp.abs(spectral - nn - pp))
        / (1 + jnp.max(jnp.abs(spectral))),
        spectral_sum_error=jnp.max(
            jnp.abs(jnp.sum(spectral, axis=1) * de / (2 * jnp.pi) - 1)
        ),
        collision_particle_error=jnp.max(jnp.abs(cparticles)) / rate_scale,
        terminal_particle_error=jnp.abs(jnp.sum(particles)) / rate_scale,
        terminal_energy_error=jnp.abs(jnp.sum(energy) - jnp.sum(heat)) / power_scale,
        local_continuity_error=jnp.max(jnp.abs(source - divergence)),
        equilibrium_kms_error=jnp.max(kms) if equilibrium else jnp.asarray(jnp.nan),
        causal=jnp.all(sin >= -tolerance)
        & jnp.all(sout >= -tolerance)
        & jnp.all(spectral >= -tolerance),
        mesh_error=jnp.asarray(jnp.inf),
        window_error=jnp.asarray(jnp.inf),
        workspace_bytes=workspace,
    )
    equilibrium_gate = (
        (evidence.equilibrium_kms_error < 10 * tolerance)
        if equilibrium
        else jnp.asarray(True)
    )
    valid = (
        jnp.all(jnp.stack(tuple(r.successful for r in nonlinear_results)))
        & evidence.causal
        & (evidence.spectral_identity_error < 10 * tolerance)
        & (evidence.collision_particle_error < 10 * tolerance)
        & (evidence.terminal_particle_error < 10 * tolerance)
        & (evidence.terminal_energy_error < 10 * tolerance)
        & (evidence.local_continuity_error < 100 * tolerance)
        & equilibrium_gate
    )
    return ScatteringResult(
        energies=energies,
        retarded_self_energy=sr * Q,
        lesser_self_energy=1j * sin * Q,
        greater_self_energy=-1j * sout * Q,
        spectral_density=spectral / Q,
        electron_correlation=nn / Q,
        electron_density=counts / h.cell_volumes,
        charge_density=-Q * counts / h.cell_volumes,
        particle_currents=particles,
        terminal_currents=-Q * particles,
        energy_currents=energy,
        heat_currents=energy
        - jnp.asarray([lead.chemical_potential for lead in leads]) * particles,
        bond_particle_currents=bonds,
        collision_particle_rate=cparticles,
        collision_energy_rate=cenergy,
        phonon_heat=heat,
        emission_rate=emit,
        absorption_rate=absorb,
        evidence=evidence,
        nonlinear_results=tuple(nonlinear_results),
        successful=valid,
    )


def solve_phonon_transport(
    device,
    bath,
    grid,
    *,
    tolerance=1e-8,
    observable_tolerance=2e-3,
    maximum_steps=300,
    damping=0.35,
):
    """SCBA plus independent doubled-resolution and doubled-window solves.

    Returned ``successful`` gates nonlinear, charge, energy, causality,
    spectral completeness and both refinements. Exhaustion is not a result
    qualified for use as a transport closure. Grids can be refined by the
    caller for stronger accuracy; no imaginary regulator supplies missing
    bound-state occupation. Singular/underdetermined populations fail.
    """
    if (
        not isinstance(device, CoherentDevice)
        or not isinstance(bath, OpticalPhononBath)
        or not isinstance(grid, PhononEnergyGrid)
    ):
        raise TypeError(
            "SCBA requires CoherentDevice, OpticalPhononBath and PhononEnergyGrid."
        )
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0
        or not np.isfinite(observable_tolerance)
        or observable_tolerance <= 0
    ):
        raise ValueError("SCBA tolerances must be finite and positive.")
    if (
        isinstance(maximum_steps, bool)
        or not isinstance(maximum_steps, (int, np.integer))
        or maximum_steps < 1
    ):
        raise ValueError("maximum_steps must be a positive integer.")
    if not np.isfinite(damping) or not 0 < damping <= 1:
        raise ValueError("damping must lie in (0, 1].")
    options = {
        "tolerance": tolerance,
        "maximum_steps": int(maximum_steps),
        "damping": damping,
    }
    coarse = _solve_grid(device, bath, grid, **options)
    refined = _solve_grid(device, bath, grid.refined(bath.energy), **options)
    wide = _solve_grid(
        device, bath, grid.refined(bath.energy).expanded(bath.energy), **options
    )

    def observables(result):
        return jnp.concatenate(
            (
                result.electron_density * device.hamiltonian.cell_volumes,
                result.particle_currents * PLANCK / Q,
                result.phonon_heat * PLANCK / Q**2,
            )
        )

    error = lambda a, b: jnp.max(
        jnp.abs(observables(a) - observables(b)) / (1 + jnp.abs(observables(b)))
    )
    mesh, window = error(coarse, refined), error(refined, wide)
    evidence = eqx.tree_at(
        lambda e: (e.mesh_error, e.window_error), wide.evidence, (mesh, window)
    )
    valid = (
        coarse.successful
        & refined.successful
        & wide.successful
        & (mesh < observable_tolerance)
        & (window < observable_tolerance)
        & (evidence.spectral_sum_error < observable_tolerance)
    )
    return eqx.tree_at(lambda r: (r.evidence, r.successful), wide, (evidence, valid))
