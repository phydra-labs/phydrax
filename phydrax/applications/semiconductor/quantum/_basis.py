# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""SI, orthonormal cell basis for connected scalar effective-mass chains.

This backend admits a finite list of transverse subbands, not arbitrary 3D or
atomistic Hamiltonians. ``c.conj() @ c == 1`` represents one longitudinal
particle; ``abs(c)**2 / cell_volumes`` is its volume density. All electronic
energies, including reservoir chemical potentials, share ``energy_reference``.
"""

from __future__ import annotations

from enum import IntEnum
from fractions import Fraction
from functools import partial
from typing import Self, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from .... import linalg as la
from ...._strict import StrictModule
from ....linalg import eigen
from .._quantities import _text, BOLTZMANN_CONSTANT_SI as KB, ELEMENTARY_CHARGE_SI as Q


if TYPE_CHECKING:
    from ._effective_mass_nd import DenseHamiltonianND


PLANCK = 6.62607015e-34
HBAR = PLANCK / (2.0 * np.pi)


def _array(value: npt.ArrayLike, name: str, *, positive: bool = False) -> Array:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real for the admitted scalar cell basis.")
    host = np.asarray(raw, dtype=np.float64)
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
        max_nodes: int = 4096,
        max_modes: int = 64,
        max_bound_states: int = 32,
        max_evaluations: int = 30000,
        max_intervals: int = 512,
        max_eigen_steps: int = 300,
        workspace_bytes: int = 256 * 1024 * 1024,
    ) -> None:
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
        values = tuple(raw)
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

    def __init__(
        self, offsets: npt.ArrayLike = (0.0,), degeneracies: npt.ArrayLike = (2.0,)
    ) -> None:
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

    def occupation(
        self, energy: ArrayLike, mu: ArrayLike, temperature: ArrayLike
    ) -> Array:
        return jnp.sum(
            self.degeneracies
            * jax.nn.sigmoid(
                (mu - jnp.asarray(energy)[..., None] - self.offsets) / (KB * temperature)
            ),
            axis=-1,
        )

    def occupied_energy(
        self, energy: ArrayLike, mu: ArrayLike, temperature: ArrayLike
    ) -> Array:
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
        self,
        diagonal: npt.ArrayLike,
        off_diagonal: npt.ArrayLike,
        cell_volumes: npt.ArrayLike,
        *,
        energy_reference: str,
        resources: QuantumResources | None = None,
    ) -> None:
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
    def size(self) -> int:
        return self.diagonal.size

    def with_potential(self, potential: ArrayLike) -> Self:
        psi = jnp.asarray(potential)
        if psi.shape != self.diagonal.shape:
            raise ValueError("Potential must have one value per cell in V.")
        return eqx.tree_at(lambda h: h.diagonal, self, self.diagonal - Q * psi)

    def shifted(self, energy: ArrayLike) -> Self:
        return eqx.tree_at(lambda h: h.diagonal, self, self.diagonal + energy)

    def operator(
        self, *, scale: ArrayLike = Q, shift: ArrayLike = 0.0
    ) -> la.TridiagonalLinearOperator:
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

    def density(self, mode_vectors: Array, occupations: Array) -> Array:
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

    def __init__(
        self,
        x: npt.ArrayLike,
        band_edge: npt.ArrayLike,
        mass: npt.ArrayLike,
        *,
        area: npt.ArrayLike,
        energy_reference: str,
        resources: QuantumResources | None = None,
    ) -> None:
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

    def hamiltonian(self, potential: ArrayLike) -> ChainHamiltonian:
        return self.base_hamiltonian.with_potential(potential)


class SchrodingerResult(StrictModule):
    """Occupied one-particle spectrum, not a total electrothermal ledger.

    energy_density is the reference-dependent expectation of H (J/m3).
    If H includes -q*psi, do not add this unchanged to Poisson field energy.
    Material thermodynamic internal energy requires a separate consistent
    free-energy/entropy closure. ``omitted_bound_status`` is an
    `OmittedBoundStatus` code; anything but ``CERTIFIED`` makes the solve
    unsuccessful.
    """

    energies: Array
    vectors: Array
    occupations: Array
    electron_density: Array
    charge_density: Array
    energy_density: Array
    omitted_particle_bound: Array
    omitted_bound_status: Array
    eigen_result: eigen.EigenSolveResult
    successful: Array


def selected_eigenpairs(
    hamiltonian: ChainHamiltonian | DenseHamiltonianND,
    count: int,
    *,
    shift: ArrayLike = 0.0,
    which: eigen.EigenTarget = "smallest-algebraic",
) -> eigen.EigenSolveResult:
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


_UNIT_ROUNDOFF = Fraction(1, 2**53)
_STURM_MAXIMUM_SCALE = 2.0**500


class OmittedBoundStatus(IntEnum):
    """Outcome of the omitted-population certificate of `solve_schrodinger`.

    ``OUT_OF_RANGE`` marks a chain with non-finite entries or entries beyond the
    certified 2^500 eV scale; its bound is infinite and the solve unsuccessful.
    """

    CERTIFIED = 0
    OUT_OF_RANGE = 1


def _sturm_scale(
    hamiltonian_diagonal: npt.ArrayLike, hamiltonian_off_diagonal: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Chain entries and Gershgorin radii in eV, and their scale (NaN propagates)."""
    diagonal = np.asarray(hamiltonian_diagonal, dtype=np.float64) / Q
    off_diagonal = np.asarray(hamiltonian_off_diagonal, dtype=np.float64) / Q
    radius = np.zeros_like(diagonal)
    radius[:-1] += np.abs(off_diagonal)
    radius[1:] += np.abs(off_diagonal)
    scale = float(np.max(np.maximum(np.abs(diagonal), radius), initial=1.0))
    return diagonal, off_diagonal, radius, scale


def _sturm_admitted(
    hamiltonian_diagonal: npt.ArrayLike, hamiltonian_off_diagonal: npt.ArrayLike
) -> bool:
    """Whether the chain lies in the certified finite scale of the Sturm bound."""
    scale = _sturm_scale(hamiltonian_diagonal, hamiltonian_off_diagonal)[3]
    return scale <= _STURM_MAXIMUM_SCALE


def _sturm_count_below(
    diagonal: list[float], couplings: list[float], shift: float, pivot_floor: float
) -> int:
    """Negative LDL^T pivots of ``T - shift`` in host binary64 arithmetic."""
    pivot = diagonal[0] - shift
    count = int(pivot < 0)
    for row, coupling in zip(diagonal[1:], couplings, strict=True):
        if abs(pivot) <= pivot_floor:
            pivot = -pivot_floor if pivot < 0 else pivot_floor
        pivot = row - shift - coupling / pivot
        count += pivot < 0
    return count


def _sturm_eigenvalue_lower_bound(
    hamiltonian_diagonal: np.ndarray, hamiltonian_off_diagonal: np.ndarray, index: int
) -> float:
    """Certified binary64 lower bound, in J, on ascending eigenvalue ``index``.

    The count runs on the host in Python binary64 (one round-to-nearest per
    operation, gradual underflow, no contraction), never under XLA. With
    ``u = 2^-53``, ``A = fl(H / Q)``, and a subnormal absolute error at most
    ``2^-1075`` on each product and quotient, the counted signs are exactly the
    LDL^T inertia of a symmetric tridiagonal ``T'`` with, per row,
    ``|a' - a| <= u |a - shift| + (1 + 2u) pivot_floor + 2^-1020`` and
    ``|b' - b| <= 2u |b|``: each pivot absorbs its subtraction rounding and
    floor replacement into ``a'``, and its quotient, square, and the previous
    pivot's rounding into ``b'^2``. Weyl with the row-sum norm gives
    ``|lambda(T') - lambda(A)| <= u |shift| + 68 u scale + 2^-1020``. The
    scaling ``A - H / Q`` adds ``3u scale``; the unverified Gershgorin start
    errs by at most ``4u scale``. ``128 u (|shift| + scale) + 2^-1000``
    dominates both cases, and the conversion to J rounds downward exactly.
    Callers admit the chain first with `_sturm_admitted`.
    """
    from ....geometry._rational import fraction_down

    diagonal, off_diagonal, radius, scale = _sturm_scale(
        hamiltonian_diagonal, hamiltonian_off_diagonal
    )
    pivot_floor = 32 * float(np.finfo(np.float64).eps) * scale
    rows = diagonal.tolist()
    couplings = (off_diagonal * off_diagonal).tolist()
    lower = float(np.min(diagonal - radius))
    upper = float(np.max(diagonal + radius))
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        if _sturm_count_below(rows, couplings, middle, pivot_floor) <= index:
            lower = middle
        else:
            upper = middle
    padding = 128 * _UNIT_ROUNDOFF * (abs(Fraction(lower)) + Fraction(scale))
    return fraction_down((Fraction(lower) - padding - Fraction(1, 2**1000)) * Fraction(Q))


def _omitted_population_upper_bound(
    offsets: np.ndarray,
    degeneracies: np.ndarray,
    omitted_states: int,
    energy: float,
    mu: float,
    temperature: float,
) -> float:
    """Upward-rounded ``omitted_states`` times the transverse-mode occupation."""
    from mpmath.libmp import to_rational

    from ....geometry._rational import fraction_up
    from ....geometry.brep._intersection_curve import interval_context

    context = interval_context(40)
    thermal = Fraction(KB) * Fraction(temperature)
    total = Fraction(0)
    for offset, degeneracy in zip(
        np.asarray(offsets).tolist(), np.asarray(degeneracies).tolist(), strict=True
    ):
        argument = (Fraction(mu) - Fraction(energy) - Fraction(offset)) / thermal
        exponent = -(context.mpf(argument.numerator) / context.mpf(argument.denominator))
        fermi = 1 / (1 + context.exp(exponent))
        total += Fraction(degeneracy) * Fraction(*to_rational(fermi._mpi_[1]))
    return fraction_up(omitted_states * total)


def _omitted_population_host(
    diagonal: np.ndarray,
    off_diagonal: np.ndarray,
    offsets: np.ndarray,
    degeneracies: np.ndarray,
    mu: np.ndarray,
    temperature: np.ndarray,
    *,
    retained: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Host certificate of the omitted population and its status for one chain.

    A chain outside the certified scale returns an infinite bound with
    ``OUT_OF_RANGE`` instead of raising across the callback boundary.
    """
    if not _sturm_admitted(diagonal, off_diagonal):
        return (
            np.asarray(np.inf, dtype=np.float64),
            np.asarray(OmittedBoundStatus.OUT_OF_RANGE, dtype=np.int32),
        )
    energy = _sturm_eigenvalue_lower_bound(diagonal, off_diagonal, retained)
    bound = _omitted_population_upper_bound(
        offsets,
        degeneracies,
        diagonal.shape[0] - retained,
        energy,
        float(mu),
        float(temperature),
    )
    return (
        np.asarray(bound, dtype=np.float64),
        np.asarray(OmittedBoundStatus.CERTIFIED, dtype=np.int32),
    )


def solve_schrodinger(
    hamiltonian: ChainHamiltonian | DenseHamiltonianND,
    chemical_potential: npt.ArrayLike,
    temperature: npt.ArrayLike,
    *,
    count: int | np.integer,
    transverse: TransverseModes | None = None,
    omitted_particle_tolerance: float = 1e-8,
) -> SchrodingerResult:
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
    chain = isinstance(hamiltonian, ChainHamiltonian)
    if chain and retained < hamiltonian.size:
        # Concrete chains refuse here; traced ones report OUT_OF_RANGE instead.
        entries = (hamiltonian.diagonal, hamiltonian.off_diagonal)
        concrete = not any(isinstance(entry, jax_core.Tracer) for entry in entries)
        if concrete and not _sturm_admitted(*entries):
            raise ValueError(
                "Sturm certification requires finite Hamiltonian entries below 2^500 eV."
            )
    result = selected_eigenpairs(hamiltonian, retained)
    energies = result.eigenvalues * Q
    vectors = result.eigenvectors
    occupations = modes.occupation(energies, chemical_potential_, temperature_)
    if retained == hamiltonian.size:
        omitted = jnp.asarray(0.0, dtype=energies.dtype)
        status = jnp.asarray(OmittedBoundStatus.CERTIFIED, dtype=jnp.int32)
    elif chain:
        # The certificate runs on the host in binary64 even under tracing; it
        # gates success and carries no derivative.
        host_inputs = tuple(
            jax.lax.stop_gradient(value)
            for value in (
                hamiltonian.diagonal,
                hamiltonian.off_diagonal,
                modes.offsets,
                modes.degeneracies,
                chemical_potential_,
                temperature_,
            )
        )
        bound, status = jax.lax.stop_gradient(
            jax.pure_callback(
                partial(_omitted_population_host, retained=retained),
                (
                    jax.ShapeDtypeStruct((), jnp.float64),
                    jax.ShapeDtypeStruct((), jnp.int32),
                ),
                *host_inputs,
                vmap_method="sequential",
            )
        )
        omitted = bound.astype(energies.dtype)
    else:
        raise ValueError(
            "DenseHamiltonianND confinement requires the complete declared finite "
            "basis; count must equal hamiltonian.size."
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
        omitted_bound_status=status,
        eigen_result=result,
        successful=result.successful
        & (status == OmittedBoundStatus.CERTIFIED)
        & (omitted <= omitted_particle_tolerance),
    )
