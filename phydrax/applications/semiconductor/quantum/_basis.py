# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""SI, orthonormal cell basis for connected scalar effective-mass chains.

This backend admits a finite list of transverse subbands, not arbitrary 3D or
atomistic Hamiltonians. ``c.conj() @ c == 1`` represents one longitudinal
particle; ``abs(c)**2 / cell_volumes`` is its volume density. All electronic
energies, including reservoir chemical potentials, share ``energy_reference``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .... import linalg as la
from ...._strict import StrictModule
from ....linalg import eigen
from .._quantities import _text, BOLTZMANN_CONSTANT_SI as KB, ELEMENTARY_CHARGE_SI as Q


PLANCK = 6.62607015e-34
HBAR = PLANCK / (2.0 * np.pi)


def _array(value, name, *, positive=False):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real for the admitted scalar cell basis.")
    host = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(host)) or (positive and np.any(host <= 0)):
        raise ValueError(
            f"{name} must be finite" + (" and positive." if positive else ".")
        )
    array = jnp.asarray(host)
    if array.dtype.itemsize < 8:
        raise ValueError("The physical SI quantum backend requires JAX 64-bit precision.")
    return array


class QuantumResources(StrictModule):
    """Hard structural and execution limits; exhaustion is not convergence."""

    max_nodes: int = eqx.field(static=True)
    max_modes: int = eqx.field(static=True)
    max_bound_states: int = eqx.field(static=True)
    max_evaluations: int = eqx.field(static=True)
    max_intervals: int = eqx.field(static=True)
    max_eigen_steps: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_nodes=4096,
        max_modes=64,
        max_bound_states=32,
        max_evaluations=30000,
        max_intervals=512,
        max_eigen_steps=300,
        workspace_bytes=256 * 1024 * 1024,
    ):
        raw = (
            max_nodes,
            max_modes,
            max_bound_states,
            max_evaluations,
            max_intervals,
            max_eigen_steps,
            workspace_bytes,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
            for value in raw
        ):
            raise ValueError("Quantum resource bounds must be positive integers.")
        values = tuple(int(value) for value in raw)
        (
            self.max_nodes,
            self.max_modes,
            self.max_bound_states,
            self.max_evaluations,
            self.max_intervals,
            self.max_eigen_steps,
            self.workspace_bytes,
        ) = values


class TransverseModes(StrictModule):
    """Complete *declared finite-mode model*, not an inferred transverse DOS.

    Offsets are J, degeneracies include spin and valleys exactly once. Each
    offset shifts both device and leads. Area converts each mode's particle
    count to m^-3; no extra area or spin multiplier is hidden in transport.
    """

    offsets: Array
    degeneracies: Array

    def __init__(self, offsets=(0.0,), degeneracies=(2.0,)):
        offsets_ = _array(offsets, "transverse energies")
        degeneracies_ = _array(degeneracies, "transverse degeneracies", positive=True)
        if (
            offsets_.ndim != 1
            or offsets_.size == 0
            or offsets_.shape != degeneracies_.shape
        ):
            raise ValueError(
                "Transverse energies and degeneracies require equal nonempty vectors."
            )
        self.offsets = offsets_
        self.degeneracies = degeneracies_

    def occupation(self, energy, mu, temperature):
        return jnp.sum(
            self.degeneracies
            * jax.nn.sigmoid(
                (mu - jnp.asarray(energy)[..., None] - self.offsets) / (KB * temperature)
            ),
            axis=-1,
        )

    def occupied_energy(self, energy, mu, temperature):
        total = jnp.asarray(energy)[..., None] + self.offsets
        return jnp.sum(
            total * self.degeneracies * jax.nn.sigmoid((mu - total) / (KB * temperature)),
            axis=-1,
        )


class ChainHamiltonian(StrictModule):
    """Real connected nearest-neighbor H in an orthonormal cell basis (S=I).

    Nonorthogonal device matrices are deliberately not exposed by this
    backend. Use an actual generalized-basis backend rather than dropping S.
    This explicit tridiagonal contract is also the stationary operator seam
    for subsequent physical self-energies and memory kernels.
    """

    diagonal: Array
    off_diagonal: Array
    cell_volumes: Array
    energy_reference: str = eqx.field(static=True)
    resources: QuantumResources

    def __init__(
        self, diagonal, off_diagonal, cell_volumes, *, energy_reference, resources=None
    ):
        d = _array(diagonal, "Hamiltonian diagonal")
        o = _array(off_diagonal, "Hamiltonian hopping")
        v = _array(cell_volumes, "cell volumes", positive=True)
        r = QuantumResources() if resources is None else resources
        if (
            d.ndim != 1
            or not 1 <= d.size <= r.max_nodes
            or o.shape != (d.size - 1,)
            or v.shape != d.shape
        ):
            raise ValueError("Admitted chain storage is n, n-1, n within max_nodes.")
        if np.any(np.asarray(o) == 0):
            raise ValueError(
                "Disconnected chains require separate populations and bound-state ownership."
            )
        self.diagonal, self.off_diagonal, self.cell_volumes = d, o, v
        self.energy_reference = _text(energy_reference, "electronic energy reference")
        self.resources = r

    @property
    def size(self):
        return self.diagonal.size

    def with_potential(self, potential):
        psi = jnp.asarray(potential)
        if psi.shape != self.diagonal.shape:
            raise ValueError("Potential must have one value per cell in V.")
        return eqx.tree_at(lambda h: h.diagonal, self, self.diagonal - Q * psi)

    def shifted(self, energy):
        return eqx.tree_at(lambda h: h.diagonal, self, self.diagonal + energy)

    def operator(self, *, scale=Q, shift=0.0):
        op = la.TridiagonalLinearOperator(
            self.off_diagonal / scale,
            (self.diagonal - shift) / scale,
            self.off_diagonal / scale,
            operator_id="quantum-real-hermitian-chain",
        )
        return eqx.tree_at(
            lambda x: x.properties,
            op,
            la.OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            ),
        )

    def density(self, mode_vectors, occupations):
        return (
            jnp.sum(jnp.abs(mode_vectors) ** 2 * occupations[None, :], axis=1)
            / self.cell_volumes
        )


class EffectiveMass1D(StrictModule):
    """Uniform interior cell centers with homogeneous Dirichlet ghost nodes.

    x is m, mass is kg, band_edge is J, area is m2. Inverse mass on an
    interface is the harmonic mean of the nodal inverse masses, enforcing
    conservative BenDaniel-Duke mass-step matching. No arbitrary mesh claim.
    """

    x: Array
    mass: Array
    spacing: Array
    area: Array
    base_hamiltonian: ChainHamiltonian

    def __init__(self, x, band_edge, mass, *, area, energy_reference, resources=None):
        x_ = _array(x, "cell centers")
        if x_.ndim != 1 or x_.size < 2:
            raise ValueError(
                "EffectiveMass1D requires at least two interior cell centers."
            )
        dx = np.diff(np.asarray(x_))
        if np.any(dx <= 0) or not np.allclose(dx, dx[0], rtol=1e-9, atol=0):
            raise ValueError(
                "Only strictly ordered uniform 1D cell centers are admitted."
            )
        m = _array(np.broadcast_to(mass, x_.shape), "effective masses", positive=True)
        b = _array(np.broadcast_to(band_edge, x_.shape), "band edges")
        a = _array(area, "cross-sectional area", positive=True)
        if a.shape != ():
            raise ValueError("Cross-sectional area must be scalar.")
        internal = HBAR**2 / ((m[:-1] + m[1:]) * dx[0] ** 2)
        links = jnp.concatenate(
            (
                jnp.asarray([HBAR**2 / (2 * m[0] * dx[0] ** 2)]),
                internal,
                jnp.asarray([HBAR**2 / (2 * m[-1] * dx[0] ** 2)]),
            )
        )
        self.x, self.mass, self.spacing, self.area = x_, m, jnp.asarray(dx[0]), a
        self.base_hamiltonian = ChainHamiltonian(
            b + links[:-1] + links[1:],
            -internal,
            jnp.full(x_.shape, dx[0]) * a,
            energy_reference=energy_reference,
            resources=resources,
        )

    def hamiltonian(self, potential):
        return self.base_hamiltonian.with_potential(potential)


class SchrodingerResult(StrictModule):
    """Occupied one-particle spectrum, not a total electrothermal ledger.

    energy_density is the reference-dependent expectation of H (J/m3).
    If H includes -q*psi, do not add this unchanged to Poisson field energy.
    Material thermodynamic internal energy requires a separate consistent
    free-energy/entropy closure.
    """

    energies: Array
    vectors: Array
    occupations: Array
    electron_density: Array
    charge_density: Array
    energy_density: Array
    omitted_particle_bound: Array
    eigen_result: eigen.EigenSolveResult
    successful: Array


def selected_eigenpairs(hamiltonian, count, *, shift=0.0, which="smallest-algebraic"):
    """Native bounded selected Lanczos solve, in numerically scaled joules."""
    n = hamiltonian.size
    r = hamiltonian.resources
    if (
        isinstance(count, bool)
        or not isinstance(count, (int, np.integer))
        or not 1 <= count <= min(n, r.max_modes)
    ):
        raise ValueError(
            "Selected eigenmode count must be an integer within the resource bound."
        )
    subspace = min(n, max(4 * count + 16, 48))
    # Full finite-basis requests use a native block solve; no dense H is formed.
    method = (
        eigen.LOBPCG(block_dimension=count)
        if count == n
        else eigen.RestartedLanczos(
            subspace_dimension=subspace,
            restart_dimension=min(max(2 * count + 4, 8), subspace - 1),
        )
    )
    policy = eigen.EigenSolvePolicy(
        method,
        count=count,
        which=which,
        max_steps=r.max_eigen_steps,
        key=jax.random.key(0),
        tolerance=eigen.EigenTolerancePolicy(
            relative=1e-10, absolute=1e-9, orthogonality=1e-9
        ),
        resources=eigen.EigenResourcePolicy(
            workspace_bytes=r.workspace_bytes,
            krylov_basis_bytes=r.workspace_bytes,
            preparation_bytes=r.workspace_bytes,
        ),
        materialization=la.MaterializationPolicy(
            max_entries=max(1, subspace**2), max_bytes=r.workspace_bytes
        ),
    )
    return eigen.eigensolve(
        eigen.Eigenproblem(hamiltonian.operator(shift=shift)), policy=policy
    )


def _sturm_eigenvalue_lower_bound(hamiltonian, index):
    """Conservative lower endpoint for one ordered tridiagonal eigenvalue."""
    diagonal = hamiltonian.diagonal / Q
    off_diagonal = hamiltonian.off_diagonal / Q
    radius = jnp.zeros_like(diagonal)
    radius = radius.at[:-1].add(jnp.abs(off_diagonal))
    radius = radius.at[1:].add(jnp.abs(off_diagonal))
    lower = jnp.min(diagonal - radius)
    upper = jnp.max(diagonal + radius)
    scale = jnp.maximum(1.0, jnp.maximum(jnp.max(jnp.abs(diagonal)), jnp.max(radius)))
    pivot_floor = 32 * jnp.finfo(diagonal.dtype).eps * scale

    def count_below(value):
        first = diagonal[0] - value
        first_safe = jnp.where(
            jnp.abs(first) > pivot_floor,
            first,
            jnp.where(first < 0, -pivot_floor, pivot_floor),
        )

        def step(carry, data):
            pivot, count = carry
            diagonal_value, coupling = data
            next_pivot = diagonal_value - value - coupling**2 / pivot
            next_safe = jnp.where(
                jnp.abs(next_pivot) > pivot_floor,
                next_pivot,
                jnp.where(next_pivot < 0, -pivot_floor, pivot_floor),
            )
            return (next_safe, count + (next_pivot < 0)), None

        (_, count), _ = jax.lax.scan(
            step,
            (first_safe, jnp.asarray(first < 0, dtype=jnp.int32)),
            (diagonal[1:], off_diagonal),
        )
        return count

    def bisect(_, interval):
        left, right = interval
        middle = 0.5 * (left + right)
        below = count_below(middle)
        return jax.lax.cond(
            below <= index,
            lambda pair: (middle, pair[1]),
            lambda pair: (pair[0], middle),
            (left, right),
        )

    lower, _ = jax.lax.fori_loop(0, 64, bisect, (lower, upper))
    padding = 256 * jnp.finfo(diagonal.dtype).eps * (scale + jnp.abs(lower))
    return (lower - padding) * Q


def solve_schrodinger(
    hamiltonian,
    chemical_potential,
    temperature,
    *,
    count,
    transverse=None,
    omitted_particle_tolerance=1e-8,
):
    """Occupied modes with a conservative finite-basis population bound.

    A tridiagonal Sturm count brackets the first omitted eigenvalue from below.
    Monotone occupation at that certified endpoint bounds every omitted state;
    a Ritz guard value is never presented as a rigorous lower spectral bound.
    This certifies truncation in the declared finite basis, not mesh convergence.
    """
    if (
        isinstance(count, bool)
        or not isinstance(count, (int, np.integer))
        or not 1 <= count <= hamiltonian.size
    ):
        raise ValueError("count must be an in-range integer mode count.")
    if not np.isfinite(omitted_particle_tolerance) or omitted_particle_tolerance < 0:
        raise ValueError("omitted_particle_tolerance must be finite and nonnegative.")
    modes = TransverseModes() if transverse is None else transverse
    if not isinstance(modes, TransverseModes):
        raise TypeError("transverse must be TransverseModes or None.")
    temperature_ = _array(temperature, "reservoir temperature", positive=True)
    chemical_potential_ = _array(chemical_potential, "chemical potential")
    if temperature_.shape != () or chemical_potential_.shape != ():
        raise ValueError("Confinement reservoirs require scalar mu and temperature.")
    if modes.offsets.size > hamiltonian.resources.max_modes:
        raise ValueError("Transverse mode count exceeds the declared resource bound.")
    retained = int(count)
    result = selected_eigenpairs(hamiltonian, retained)
    energies = result.eigenvalues * Q
    vectors = result.eigenvectors
    occupations = modes.occupation(energies, chemical_potential_, temperature_)
    if retained == hamiltonian.size:
        omitted = jnp.asarray(0.0, dtype=energies.dtype)
    else:
        omitted_energy = _sturm_eigenvalue_lower_bound(hamiltonian, retained)
        omitted = (hamiltonian.size - retained) * modes.occupation(
            omitted_energy, chemical_potential_, temperature_
        )
    density = hamiltonian.density(vectors, occupations)
    energy_density = hamiltonian.density(
        vectors,
        modes.occupied_energy(energies, chemical_potential_, temperature_),
    )
    return SchrodingerResult(
        energies=energies,
        vectors=vectors,
        occupations=occupations,
        electron_density=density,
        charge_density=-Q * density,
        energy_density=energy_density,
        omitted_particle_bound=omitted,
        eigen_result=result,
        successful=result.successful & (omitted <= omitted_particle_tolerance),
    )
