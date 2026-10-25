#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""FourierD3: periodic particle-mesh D3(BJ) with the modified coordination number.

This is the method of Valeeva et al. (arXiv:2607.15103) as implemented by the
inspected NVIDIA nvalchemi-toolkit-ops ``10e12340c39fd12a2375d6301768e72d2cd12868``,
a scientific definition distinct from conventional D3 (:mod:`._dispersion_d3`):

* coordination numbers use the modified counting function of
  :class:`FourierD3CoordinationDefinition` over full directed image routes
  ``0 < r < R_c`` (``R_c`` is the coordination role radius);
* per-atom coefficients ``c_il = sum_p v_{a,p,l} L_p / sum_p L_p`` interpolate
  the signed factors of a :class:`FourierC6Decomposition` of the reference
  species/slot block (``L_p = exp(-4 (CN_i - CN_a^p)^2)``, no threshold), so
  ``C6_ij = sum_l lambda_l c_il c_jl`` with negative ``lambda_l`` retained;
* the dispersion lattice sum has no real-space cutoff:

  ``E = -1/(2V) sum_k sum_{a,b} sum_l lambda_l K_ab(|k|) S_al(k) conj(S_bl(k)) + E_self``

  with ``S_al(k) = sum_{i in a} c_il exp(i k.r_i)``, the ``k = 0`` mode
  included, Hermitian half-spectrum weights from the shared mesh owner and
  the closed-form radial transform ``K_ab`` of
  ``phi_ab(r) = s6 / (r^6 + R0^6) + 3 s8 q_ab / (r^8 + R0^8)``,
  ``q_ab = r4r2_a r4r2_b``, ``R0 = a1 sqrt(3 q_ab) + a2``;
* ``E_self = +1/2 sum_i sum_l lambda_l c_il^2 phi_aa(0)`` cancels the
  zero-separation self pair the reciprocal sum contains. It is part of the
  same scalar closure as the coefficients, so its coordination-number force
  and virial are never lost.

The reciprocal sum runs on a term-owned :class:`PreparedReciprocalMesh`:
species x rank-chunk channel spectra are transformed once per chunk
(``transform``) and reused for every species pair through ``cross_quadratic``
with the pair's damped kernel; rank chunks run in a checkpointed scan and
pairs in a nested checkpointed scan. There is no direct real-space fallback.
Mesh interpolation error, reciprocal truncation (the mesh Nyquist limit), the
C6 reconstruction residual and the modified-CN definition are separate error
sources; none bounds the others.

Derivatives of every order follow from the one scalar closure: positions,
cell strain, damping parameters, the stored signed factors and eigenvalues,
reference coordination numbers, ``r4r2`` and covalent radii. The reference C6
table enters only through the host decomposition, so a changed table needs a
new decomposition rather than a derivative. Differences from the inspected
source kept explicit: the shared mesh's Nyquist convention (``"symmetrized"``
averages the influence over Nyquist sign variants; the source evaluates
``+N/2`` only, which differs for triclinic cells), the declared modulus
(source: ``"continuous-sinc-window"`` with a ``1e-10`` floor; here a vanishing
window refuses) and the series/closed-form crossover of the radial kernel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._d3_parameters import (
    D3BJDamping,
    D3ReferenceTables,
    FourierC6Decomposition,
    FourierD3CoordinationDefinition,
    length_factor,
)
from ._dispersion_d3 import D3Routes, prepared_species_rows, system_scales
from ._neighborhood_bundle import (
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticPairCounting,
)
from ._potential import AtomisticPotentialCapabilities, AtomisticPotentialRequirements
from ._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialContext,
    AtomisticTermEvaluation,
)
from ._reciprocal_mesh import (
    PreparedReciprocalMesh,
    ReciprocalBudget,
    ReciprocalGeometry,
    ReciprocalMeshDensity,
    ReciprocalMeshFrame,
    ReciprocalMeshModulus,
    ReciprocalMeshPlan,
    ReciprocalNyquistPolicy,
    ReciprocalResourceEvidence,
)
from ._system import PreparedAtomisticSystem


#: Dimensionless ``x = |k| R0`` below which the entire shape functions use
#: their Taylor series. Above it the closed forms lose at most a few ulps to
#: cancellation; below it the series remainder after ``SERIES_TERMS`` terms
#: is below ``3 / (SERIES_TERMS + 1)!`` (about ``2e-25`` at ``x = 1``). The two
#: representations of one entire function therefore join at roundoff level.
SERIES_CROSSOVER = 1.0
SERIES_TERMS = 24

_SQRT3 = math.sqrt(3.0)
_SIN_PI_8 = math.sin(math.pi / 8.0)
_COS_PI_8 = math.cos(math.pi / 8.0)
_COMPLEX_ITEMSIZE = 16
_REAL_ITEMSIZE = 8


def _shape6_series() -> tuple[float, ...]:
    """Taylor coefficients of ``N6(x)/x``; ``N6 = e^-x - 2 e^-x/2 cos(pi/3 + sqrt3 x/2)``.

    ``N6(x) = sum_n [(-1)^n - 2 cos(pi (2n + 1) / 3)] x^n / n!`` with an integer
    bracket, so the coefficient of ``x^(n-1)`` in ``N6/x`` is exact up to ``1/n!``.
    """
    values = []
    for n in range(1, SERIES_TERMS + 1):
        bracket = round((-1.0) ** n - 2.0 * math.cos(math.pi * (2 * n + 1) / 3.0))
        values.append(bracket / math.factorial(n))
    return tuple(values)


def _shape8_series() -> tuple[float, ...]:
    """Taylor coefficients of ``N8(x)/x``.

    ``N8(x) = e^(-x sin(pi/8)) cos(pi/4 + x cos(pi/8)) + e^(-x cos(pi/8)) cos(3 pi/4
    + x sin(pi/8)) = Re[e^(i pi/4) e^(x e^(i 5pi/8)) + e^(i 3pi/4) e^(x e^(i 7pi/8))]``.
    """
    values = []
    for n in range(1, SERIES_TERMS + 1):
        bracket = math.cos(math.pi / 4.0 + 5.0 * math.pi * n / 8.0) + math.cos(
            3.0 * math.pi / 4.0 + 7.0 * math.pi * n / 8.0
        )
        # The bracket is 2 cos(pi/2 + 3 pi n/4) cos(pi/4 + pi n/8): exactly zero
        # (a few ulps in floating point) or at least 0.5 in magnitude.
        values.append((0.0 if abs(bracket) < 1.0e-9 else bracket) / math.factorial(n))
    return tuple(values)


SHAPE6_SERIES = _shape6_series()
SHAPE8_SERIES = _shape8_series()


def _horner(coefficients: tuple[float, ...], x: Array, /) -> Array:
    result = jnp.full_like(x, coefficients[-1])
    for value in reversed(coefficients[:-1]):
        result = result * x + value
    return result


def _shape6(x: Array, /) -> Array:
    small = x < SERIES_CROSSOVER
    series = _horner(SHAPE6_SERIES, jnp.where(small, x, 0.0))
    large = jnp.where(small, SERIES_CROSSOVER, x)
    closed = (
        jnp.exp(-large)
        - 2.0 * jnp.exp(-0.5 * large) * jnp.cos(math.pi / 3.0 + 0.5 * _SQRT3 * large)
    ) / large
    return jnp.where(small, series, closed)


def _shape8(x: Array, /) -> Array:
    small = x < SERIES_CROSSOVER
    series = _horner(SHAPE8_SERIES, jnp.where(small, x, 0.0))
    large = jnp.where(small, SERIES_CROSSOVER, x)
    closed = (
        jnp.exp(-large * _SIN_PI_8) * jnp.cos(math.pi / 4.0 + large * _COS_PI_8)
        + jnp.exp(-large * _COS_PI_8) * jnp.cos(3.0 * math.pi / 4.0 + large * _SIN_PI_8)
    ) / large
    return jnp.where(small, series, closed)


def damped_dispersion_fourier_kernel(
    wavenumber: Array, radius: Array, s6: Array, s8: Array, product: Array, /
) -> Array:
    """``K(k) = int phi(r) exp(-i k.r) d^3r`` of the BJ-damped pair potential.

    ``phi(r) = s6 / (r^6 + R0^6) + 3 s8 q / (r^8 + R0^8)`` with ``q = product``;
    ``K = s6 (2 pi^2 / 3 R0^3) N6(x)/x - 3 s8 q (pi^2 / R0^5) N8(x)/x`` and
    ``x = k R0``. All lengths share one unit. The transform is an exponentially
    damped oscillation; ``K(0)`` is the finite integral of ``phi`` over space
    (``N6(x)/x -> 1`` and ``N8(x)/x -> cos(7pi/8) + cos(13pi/8)``), and the
    ``k = 0`` mode is part of the method.
    """
    x = wavenumber * radius
    cube = radius * radius * radius
    return s6 * (2.0 * math.pi**2 / 3.0) / cube * _shape6(x) - 3.0 * s8 * product * (
        math.pi**2
    ) / (cube * radius * radius) * _shape8(x)


@final
class FourierD3PairInfluence(StrictModule):
    """Reciprocal influence of one species pair with dynamic damping leaves.

    Called by the mesh owner with current-cell wavevectors in inverse system
    length; ``table_per_system_length`` converts them to the table length unit.
    """

    radius: Array
    s6: Array
    s8: Array
    product: Array
    table_per_system_length: float = eqx.field(static=True)

    def __call__(self, wave: Array, squared: Array) -> Array:
        del wave
        positive = squared > 0.0
        norm = jnp.sqrt(jnp.where(positive, squared, 1.0))
        wavenumber = jnp.where(positive, norm, 0.0) / self.table_per_system_length
        return damped_dispersion_fourier_kernel(
            wavenumber, self.radius, self.s6, self.s8, self.product
        )


def modified_count(
    distance: Array,
    covalent: Array,
    cutoff: ArrayLike,
    covalent_radius_scale: ArrayLike,
    steepness: ArrayLike,
    epsilon: ArrayLike,
    /,
) -> Array:
    """FourierD3 counting value of routes of length ``distance`` (one length unit)."""
    transition = 0.5 * (covalent / covalent_radius_scale + cutoff)
    beyond = distance > transition
    stabilised = jnp.where(beyond, distance, transition)
    gap = cutoff - stabilised
    offset = stabilised - transition
    steep = steepness + offset * offset / (gap * gap + epsilon)
    return jax.nn.sigmoid(steep * (covalent / distance - 1.0))


def _channel_slice(
    density: ReciprocalMeshDensity, start: Array, count: int, /
) -> ReciprocalMeshDensity:
    return ReciprocalMeshDensity(
        density=jax.lax.dynamic_slice_in_dim(density.density, start, count, axis=0),
        sources_finite=density.sources_finite,
        routes_successful=density.routes_successful,
        successful=density.successful,
        mesh_identity=density.mesh_identity,
    )


@final
class FourierD3Resources(StrictModule, NonTrainableState):
    """Exact static residency of one prepared FourierD3 evaluation.

    ``density_bytes`` is the retained ``case x species x rank_chunk x
    spectrum`` complex block of one chunk, ``kernel_bytes`` one resolved pair
    kernel and ``coefficient_bytes`` the full per-atom rank coefficients and
    their cotangents (``atoms x padded_rank``). Their sum is admitted against
    the mesh budget's retained-byte limit at preparation; the mesh workspace
    is the shared owner's.
    """

    case_count: int = eqx.field(static=True)
    species: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    rank_chunk: int = eqx.field(static=True)
    padded_rank: int = eqx.field(static=True)
    rank_chunks: int = eqx.field(static=True)
    species_pairs: int = eqx.field(static=True)
    atoms: int = eqx.field(static=True)
    spectral_points: int = eqx.field(static=True)
    density_bytes: int = eqx.field(static=True)
    kernel_bytes: int = eqx.field(static=True)
    coefficient_bytes: int = eqx.field(static=True)
    retained_bytes: int = eqx.field(static=True)
    mesh: ReciprocalResourceEvidence


@final
class FourierD3Regularity(StrictModule, NonTrainableState):
    """Regularity of the prepared definition, derived from its formulas.

    * ``transition``: the counting steepness is C1 at ``R_t`` per species pair;
      ``transition_curvature_jump`` is the largest second-derivative jump of
      the steepness ``2 / ((R_c - R_t)^2 + epsilon)`` (table length ``^-2``).
    * ``log_cutoff_step``: the largest natural log of the counting value just
      inside ``R_c``; the value (and slope) step there is ``exp`` of it, small
      but nonzero, so the energy is not exactly C0 or C1 at ``R_c``.
    * ``mesh_coordinate_continuity``: the B-spline assignment of degree ``p``
      makes the mesh energy C^(p-1) in positions.
    * ``kernel_branch``: the radial kernel's series/closed-form join at
      ``SERIES_CROSSOVER`` is exact mathematically and roundoff-level
      numerically.

    PyTree layout: the spline-derived continuity and kernel branch are static
    method facts; the numeric evidence (``transition_curvature_jump``,
    ``log_cutoff_step``) is dynamic float64.
    """

    transition_curvature_jump: Array
    log_cutoff_step: Array
    mesh_coordinate_continuity: int = eqx.field(static=True)
    kernel_branch: str = eqx.field(static=True)


@final
class FourierD3Evidence(StrictModule):
    """Status and components of one FourierD3 evaluation (system energy units).

    ``prepared_identity``/``decomposition_identity`` are the exact prepared and
    C6-decomposition digests as dynamic words (``prepared_id``/
    ``decomposition_id`` decode them on the host); ``coefficient_residual`` is
    the decomposition's dynamic achieved error and ``rank`` its static rank.
    """

    coordination_numbers: Array
    coordination_routes_successful: Array
    geometry_successful: Array
    cell_successful: Array
    reciprocal_successful: Array
    volume: Array
    reciprocal_energy: Array
    self_energy: Array
    method: str = eqx.field(static=True)
    prepared_identity: IdentityWords
    decomposition_identity: IdentityWords
    coefficient_residual: Array
    rank: int = eqx.field(static=True)
    regularity: FourierD3Regularity
    resources: FourierD3Resources

    @property
    def prepared_id(self) -> str:
        """Host decoding of the evaluating term's prepared digest."""
        return self.prepared_identity.fingerprint()

    @property
    def decomposition_id(self) -> str:
        """Host decoding of the C6 decomposition digest."""
        return self.decomposition_identity.fingerprint()


@final
class FourierD3Parts(StrictModule):
    """Reciprocal and self energies (system units) at given coordination numbers."""

    reciprocal_energy: Array
    self_energy: Array
    cell_successful: Array
    reciprocal_successful: Array
    volume: Array


@final
class FourierD3Evaluation(StrictModule):
    energy: Array
    successful: Array
    evidence: FourierD3Evidence


def _count(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


@final
class FourierD3Potential(AbstractAtomisticEnergyTerm, NonTrainableState):
    """FourierD3 periodic dispersion term (modified CN, signed factors, mesh sum).

    ``coordination_cutoff`` (system length) is the full directed coordination
    role radius and the ``R_c`` of the counting function. ``spline_degree``
    (1-5; source spline order = degree + 1, source default order 4) and
    ``modulus`` (source: ``"continuous-sinc-window"``) configure the shared
    mesh. ``rank_chunk`` rank slots of every decomposed species are resident
    per scan step. A decomposition that misses its tolerance is refused.

    PyTree layout: the coordination cutoff (requirement radius), ``rank_chunk``,
    the mesh plan, ``name`` and ``force_group`` are static. ``anchor_index`` (an
    int32 scalar present exactly for the anchor-source frame) and
    ``term_identity`` (exact former ``term_id`` digest) are dynamic.
    """

    tables: D3ReferenceTables
    decomposition: FourierC6Decomposition
    damping: D3BJDamping
    definition: FourierD3CoordinationDefinition
    mesh_plan: ReciprocalMeshPlan
    coordination_cutoff: float = eqx.field(static=True)
    rank_chunk: int = eqx.field(static=True)
    anchor_index: Array | None
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        tables: D3ReferenceTables,
        decomposition: FourierC6Decomposition,
        damping: D3BJDamping,
        grid_shape: Sequence[int],
        /,
        *,
        coordination_cutoff: float,
        definition: FourierD3CoordinationDefinition | None = None,
        spline_degree: int = 3,
        modulus: ReciprocalMeshModulus = "continuous-sinc-window",
        nyquist: ReciprocalNyquistPolicy = "symmetrized",
        frame: ReciprocalMeshFrame = "cell-origin",
        anchor_index: int | None = None,
        rank_chunk: int | None = None,
        budget: ReciprocalBudget | None = None,
        name: str = "fourier-d3",
        force_group: int = 0,
    ) -> None:
        if not isinstance(tables, D3ReferenceTables):
            raise TypeError("tables must be D3ReferenceTables.")
        if not isinstance(decomposition, FourierC6Decomposition):
            raise TypeError("decomposition must be a FourierC6Decomposition.")
        if not isinstance(damping, D3BJDamping):
            raise TypeError("damping must be D3BJDamping.")
        if decomposition.tables_id != tables.tables_id:
            raise ValueError("The C6 decomposition belongs to other reference tables.")
        decomposition.require_admitted()
        coordination = (
            FourierD3CoordinationDefinition() if definition is None else definition
        )
        if not isinstance(coordination, FourierD3CoordinationDefinition):
            raise TypeError("definition must be a FourierD3CoordinationDefinition.")
        chunk = (
            decomposition.rank if rank_chunk is None else _count(rank_chunk, "rank_chunk")
        )
        if chunk > decomposition.rank:
            raise ValueError(
                f"rank_chunk {chunk} exceeds the decomposition rank {decomposition.rank}."
            )
        cutoff = float(coordination_cutoff)
        if not math.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("coordination_cutoff must be finite and positive.")
        mesh = ReciprocalMeshPlan(
            tuple(grid_shape),
            spline_degree=spline_degree,
            modulus=modulus,
            nyquist=nyquist,
            frame=frame,
            channel_chunk=decomposition.species_count * chunk,
            budget=budget,
        )
        if (frame == "anchor-source") != (anchor_index is not None):
            raise ValueError(
                "anchor_index is required exactly for an anchor-source frame."
            )
        identifier = str(name).strip()
        if not identifier:
            raise ValueError("Energy term name must be non-empty.")
        if isinstance(force_group, bool) or not isinstance(force_group, int):
            raise TypeError("force_group must be an integer.")
        if not 0 <= force_group < 32:
            raise ValueError("force_group must be in [0, 32).")
        anchor = None if anchor_index is None else int(anchor_index)
        self.tables = tables
        self.decomposition = decomposition
        self.damping = damping
        self.definition = coordination
        self.mesh_plan = mesh
        self.coordination_cutoff = cutoff
        self.rank_chunk = chunk
        self.anchor_index = (
            None if anchor is None else jnp.asarray(anchor, dtype=jnp.int32)
        )
        self.name = identifier
        self.force_group = force_group
        self.capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True,
            finite_geometry=False,
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            local_energy=False,
        )
        self.requirements = AtomisticPotentialRequirements(
            reciprocal_grid=True,
            neighborhoods=(self.coordination_requirement,),
        )
        self.term_identity = IdentityWords.of(
            {
                "kind": "fourier-d3-potential",
                "tables": tables.tables_id,
                "decomposition": decomposition.decomposition_id,
                "damping": damping.damping_id,
                "definition": coordination.definition_id,
                "mesh": mesh.plan_id,
                "coordination_cutoff": cutoff.hex(),
                "rank_chunk": chunk,
                "anchor_index": anchor,
                "zero_mode": "included",
                "self_energy": "coefficient-dependent-before-cn-chain",
                "kernel_series": [SERIES_CROSSOVER.hex(), SERIES_TERMS],
                "three_body": "excluded",
                "name": identifier,
                "force_group": force_group,
            }
        )

    @property
    def coordination_requirement(self) -> AtomisticNeighborhoodRequirement:
        return AtomisticNeighborhoodRequirement(
            self.name, AtomisticNeighborhoodRole.COORDINATION, self.coordination_cutoff
        )

    def prepare(self, system: PreparedAtomisticSystem, /) -> "PreparedFourierD3Potential":
        return PreparedFourierD3Potential(self, system)


@final
class PreparedFourierD3Potential(AbstractPreparedAtomisticEnergyTerm):
    """System-bound FourierD3: channel rows, species pairs, mesh and residency.

    PyTree layout: the exact unit factors (functions of the static declared
    units only) and the residency are static. ``epsilon`` (table length
    squared) is a dynamic float64 scalar, the regularity evidence is dynamic
    and ``term_identity``/``prepared_identity`` are the exact former
    ``term_id``/``prepared_id`` digests as dynamic words. ``energy`` folds the
    numerical binding of the context system into the term status.
    """

    plan: FourierD3Potential
    system: PreparedAtomisticSystem
    mesh: PreparedReciprocalMesh
    table_rows: Array
    channel_rows: Array
    species_rows: Array
    active: Array
    pair_left: Array
    pair_right: Array
    pair_multiplicity: Array
    length_scale: float = eqx.field(static=True)
    energy_scale: float = eqx.field(static=True)
    a2_scale: float = eqx.field(static=True)
    epsilon: Array
    regularity: FourierD3Regularity
    resources: FourierD3Resources
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self, plan: FourierD3Potential, system: PreparedAtomisticSystem, /
    ) -> None:
        tables = plan.tables
        decomposition = plan.decomposition
        tables.require_use()
        cell = system.cell
        if cell is None or cell.rank != 3 or not cell.fully_periodic:
            raise ValueError("FourierD3 requires a fully periodic rank-3 cell.")
        table_rows, active = prepared_species_rows(tables, system)
        numbers = np.asarray(system.plan.atomic_numbers, dtype=np.int64)
        outside = sorted(
            {int(value) for value in numbers[active]} - set(decomposition.species)
        )
        if outside:
            raise ValueError(
                f"The C6 decomposition does not cover active atomic numbers {outside}."
            )
        channel = {value: index for index, value in enumerate(decomposition.species)}
        channel_rows = np.zeros(numbers.shape, dtype=np.int32)
        channel_rows[active] = [channel[int(value)] for value in numbers[active]]
        species_rows = tables.rows(np.asarray(decomposition.species))
        length, energy = system_scales(tables, system)
        a2_scale = length_factor(plan.damping.a2_unit, tables.length_unit)
        epsilon = plan.definition.epsilon_in(tables.length_unit)
        cutoff = plan.coordination_cutoff * length
        radii = np.asarray(tables.covalent_radii)[species_rows]
        covalent = (radii[:, None] + radii[None, :]).reshape(-1)
        scale = float(tables.covalent_radius_scale)
        if np.any(covalent / scale >= cutoff):
            raise ValueError(
                "Every covered species pair needs R_cov / covalent_radius_scale below "
                "the coordination cutoff (the counting transition must lie inside it)."
            )
        transitions = 0.5 * (covalent / scale + cutoff)
        steps = [
            plan.definition.log_cutoff_step(float(value), cutoff, scale, epsilon)
            for value in covalent
        ]
        mesh = plan.mesh_plan.prepare(
            cell,
            system.particle_discretization(),
            anchor_index=None if plan.anchor_index is None else int(plan.anchor_index),
        )
        species = decomposition.species_count
        rank = decomposition.rank
        chunk = plan.rank_chunk
        chunks = -(-rank // chunk)
        padded = chunks * chunk
        spectral = int(np.prod(mesh.spectral_shape))
        density_bytes = species * chunk * spectral * _COMPLEX_ITEMSIZE
        kernel_bytes = spectral * _REAL_ITEMSIZE
        coefficient_bytes = 2 * system.capacity * padded * _REAL_ITEMSIZE
        retained = density_bytes + kernel_bytes + coefficient_bytes
        plan.mesh_plan.budget.admit(
            retained_bytes=retained, workspace_bytes=mesh.resources.chunk_workspace_bytes
        )
        left, right = np.triu_indices(species)
        self.plan = plan
        self.system = system
        self.mesh = mesh
        self.table_rows = jnp.asarray(table_rows)
        self.channel_rows = jnp.asarray(channel_rows)
        self.species_rows = jnp.asarray(species_rows)
        self.active = jnp.asarray(active)
        self.pair_left = jnp.asarray(left.astype(np.int32))
        self.pair_right = jnp.asarray(right.astype(np.int32))
        self.pair_multiplicity = jnp.asarray(np.where(left == right, 1.0, 2.0))
        self.length_scale = length
        self.energy_scale = energy
        self.a2_scale = a2_scale
        self.epsilon = jnp.asarray(epsilon, dtype=jnp.float64)
        self.regularity = FourierD3Regularity(
            transition_curvature_jump=jnp.asarray(
                float(np.max(2.0 / ((cutoff - transitions) ** 2 + epsilon))),
                dtype=jnp.float64,
            ),
            log_cutoff_step=jnp.asarray(float(max(steps)), dtype=jnp.float64),
            mesh_coordinate_continuity=plan.mesh_plan.spline_degree - 1,
            kernel_branch=(
                f"taylor-{SERIES_TERMS}-terms-below-x-{SERIES_CROSSOVER:g}-closed-form-above"
            ),
        )
        self.resources = FourierD3Resources(
            case_count=1,
            species=species,
            rank=rank,
            rank_chunk=chunk,
            padded_rank=padded,
            rank_chunks=chunks,
            species_pairs=int(left.size),
            atoms=system.capacity,
            spectral_points=spectral,
            density_bytes=density_bytes,
            kernel_bytes=kernel_bytes,
            coefficient_bytes=coefficient_bytes,
            retained_bytes=retained,
            mesh=mesh.resources,
        )
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-fourier-d3-potential",
                "term": plan.term_id,
                "system": system.prepared_id,
                "mesh": mesh.prepared_id,
                "length_scale": float(length).hex(),
                "energy_scale": float(energy).hex(),
            }
        )

    @property
    def method(self) -> str:
        mesh = self.plan.mesh_plan
        return (
            f"fourier-d3-modified-cn-signed-rank-{self.plan.decomposition.rank}-"
            f"particle-mesh-degree-{mesh.spline_degree}-{mesh.modulus}-"
            f"nyquist-{mesh.nyquist}-k0-included"
        )

    def coordination_numbers(self, routes: D3Routes, /) -> tuple[Array, Array, Array]:
        """Modified CN of every atom with route and coincidence status."""
        if routes.directed_multiplicity != 1:
            raise ValueError("Coordination numbers need full directed routes.")
        coordination, endpoints, coincidence = self.row_coordination_numbers(
            routes.displacement,
            routes.counted,
            self.table_rows[routes.receivers],
            self.table_rows[routes.sources],
            self.active[routes.receivers],
            self.active[routes.sources],
            routes.receivers,
            self.system.capacity,
        )
        return coordination, routes.successful & endpoints, coincidence

    def row_coordination_numbers(
        self,
        displacement: Array,
        counted: Array,
        receiver_table_rows: Array,
        source_table_rows: Array,
        receiver_active: Array,
        source_active: Array,
        receivers: Array,
        receiver_count: int,
        /,
    ) -> tuple[Array, Array, Array]:
        """Modified CN of ``receiver_count`` rows from explicit directed routes.

        Each route carries its displacement, whether it is a counted route of
        the full directed coordination relation, the D3 table rows and the
        activity of both endpoints, and its receiver row. Returns the CN, the
        endpoint status (every counted route joins active atoms) and the
        coincidence status (no counted route inside the cutoff has zero
        length). The bound system's rows (`coordination_numbers`) and one
        owner's rows of a distributed domain use this one definition.
        """
        tables = self.plan.tables
        squared = jnp.sum(displacement * displacement, axis=-1)
        cutoff = self.plan.coordination_cutoff
        inside = counted & jax.lax.stop_gradient(squared < cutoff * cutoff)
        positive = squared > 0.0
        valid = inside & positive
        distance = jnp.sqrt(jnp.where(valid, squared, 1.0)) * self.length_scale
        dtype = distance.dtype
        radii = tables.covalent_radii.astype(dtype)
        covalent = radii[receiver_table_rows] + radii[source_table_rows]
        # Uncounted routes take a harmless in-range length so no branch overflows.
        safe = jnp.where(valid, distance, covalent)
        values = modified_count(
            safe,
            covalent,
            cutoff * self.length_scale,
            tables.covalent_radius_scale.astype(dtype),
            self.plan.definition.steepness.astype(dtype),
            self.epsilon.astype(dtype),
        )
        values = jnp.where(valid, values, 0.0)
        coordination = jnp.zeros((receiver_count,), dtype=dtype)
        coordination = coordination.at[receivers].add(values)
        endpoints = jnp.all(~valid | (receiver_active & source_active))
        return coordination, endpoints, jnp.all(~inside | positive)

    def coefficients(self, coordination: Array, /) -> Array:
        """Per-atom signed rank coefficients ``(atoms, padded_rank)``; zero on padding."""
        return self.row_coefficients(coordination, self.channel_rows, self.active)

    def row_coefficients(
        self, coordination: Array, channel_rows: Array, active: Array, /
    ) -> Array:
        """Signed rank coefficients of rows with explicit species channel rows.

        ``(rows, padded_rank)``, zero on inactive rows: the bound system's rows
        (`coefficients`) or one owner's rows of a distributed domain.
        """
        tables = self.plan.tables
        decomposition = self.plan.decomposition
        dtype = coordination.dtype
        reference = tables.reference_coordination.astype(dtype)[self.species_rows]
        valid = tables.reference_valid[self.species_rows]
        exponent = (
            -self.plan.definition.width.astype(dtype)
            * (coordination[:, None] - reference[channel_rows]) ** 2
        )
        mask = valid[channel_rows]
        shift = jax.lax.stop_gradient(
            jnp.max(jnp.where(mask, exponent, -jnp.inf), axis=-1, keepdims=True)
        )
        weight = jnp.where(mask, jnp.exp(jnp.where(mask, exponent - shift, 0.0)), 0.0)
        normalized = weight / jnp.sum(weight, axis=-1, keepdims=True)
        factors = decomposition.factors.astype(dtype)[channel_rows]
        values = contract("np,npl->nl", normalized, factors, backend="jax")
        values = jnp.where(active[:, None], values, 0.0)
        padding = self.resources.padded_rank - decomposition.rank
        return jnp.pad(values, ((0, 0), (0, padding)))

    def _eigenvalues(self, dtype: jnp.dtype, /) -> Array:
        values = self.plan.decomposition.eigenvalues.astype(dtype)
        return jnp.pad(values, (0, self.resources.padded_rank - values.shape[0]))

    def _damping(self, dtype: jnp.dtype, /) -> tuple[Array, Array, Array, Array, Array]:
        damping = self.plan.damping
        r4r2 = self.plan.tables.r4r2.astype(dtype)[self.species_rows]
        return (
            damping.s6.astype(dtype),
            damping.s8.astype(dtype),
            damping.a1.astype(dtype),
            damping.a2.astype(dtype) * self.a2_scale,
            r4r2,
        )

    def self_energy(
        self, coefficients: Array, channel_rows: Array, active: Array, /
    ) -> Array:
        """``+1/2 sum_i sum_l lambda_l c_il^2 phi_aa(0)`` in table energy units.

        ``channel_rows`` are the species rows and ``active`` the activity of the
        rows of ``coefficients``: the bound system rows, or one owner's rows.
        """
        dtype = coefficients.dtype
        s6, s8, a1, a2, r4r2 = self._damping(dtype)
        square = r4r2 * r4r2
        radius = a1 * jnp.sqrt(3.0 * square) + a2
        radius2 = radius * radius
        radius6 = radius2 * radius2 * radius2
        potential = s6 / radius6 + 3.0 * s8 * square / (radius6 * radius2)
        weights = contract(
            "nl,l->n",
            coefficients * coefficients,
            self._eigenvalues(dtype),
            backend="jax",
        )
        return 0.5 * jnp.sum(weights * potential[channel_rows] * active)

    def reciprocal_energy(
        self, positions: Array, geometry: ReciprocalGeometry, coefficients: Array, /
    ) -> tuple[Array, Array]:
        """Mesh reciprocal energy (table units) and its status."""
        dtype = coefficients.dtype
        mesh = self.mesh
        routes = mesh.build(positions, geometry)
        resources = self.resources
        species = resources.species
        chunk = resources.rank_chunk
        atoms = self.system.capacity
        onehot = (
            (self.channel_rows[:, None] == jnp.arange(species)[None, :])
            & self.active[:, None]
        ).astype(dtype)
        coefficient_chunks = jnp.moveaxis(
            coefficients.reshape((atoms, resources.rank_chunks, chunk)), 1, 0
        )
        eigenvalue_chunks = self._eigenvalues(dtype).reshape(
            (resources.rank_chunks, chunk)
        )
        s6, s8, a1, a2, r4r2 = self._damping(dtype)
        product = r4r2[self.pair_left] * r4r2[self.pair_right]
        radius = a1 * jnp.sqrt(3.0 * product) + a2
        volume = geometry.volume.astype(dtype) * self.length_scale**3
        prefactor = -0.5 / volume
        pairs = (
            self.pair_left,
            self.pair_right,
            self.pair_multiplicity.astype(dtype),
            radius,
            product,
        )
        table_per_system = self.length_scale

        def rank_body(
            carry: tuple[Array, Array], inputs: tuple[Array, Array]
        ) -> tuple[tuple[Array, Array], None]:
            chunk_coefficients, chunk_eigenvalues = inputs
            values = contract(
                "ns,nl->nsl", onehot, chunk_coefficients, backend="jax"
            ).reshape((atoms, species * chunk))
            density = mesh.transform(routes, values)

            def pair_body(
                inner: tuple[Array, Array], pair: tuple[Array, ...]
            ) -> tuple[tuple[Array, Array], None]:
                left, right, multiplicity, pair_radius, pair_product = pair
                influence = FourierD3PairInfluence(
                    radius=pair_radius,
                    s6=s6,
                    s8=s8,
                    product=pair_product,
                    table_per_system_length=table_per_system,
                )
                kernel = mesh.resolve_influence(geometry, influence)
                quadratic = mesh.cross_quadratic(
                    _channel_slice(density, left * chunk, chunk),
                    _channel_slice(density, right * chunk, chunk),
                    kernel,
                    zero_mode="include",
                    channel_weights=chunk_eigenvalues * multiplicity * prefactor,
                )
                return (inner[0] + quadratic.value, inner[1] & quadratic.successful), None

            (value, status), _ = jax.lax.scan(
                jax.checkpoint(pair_body),
                (jnp.zeros((), dtype=dtype), density.successful),
                pairs,
            )
            return (carry[0] + value, carry[1] & status), None

        (total, status), _ = jax.lax.scan(
            jax.checkpoint(rank_body),
            (jnp.zeros((), dtype=dtype), routes.successful),
            (coefficient_chunks, eigenvalue_chunks),
        )
        return total, status

    def parts(
        self, positions: Array, cell_vectors: Array, coordination: Array, /
    ) -> FourierD3Parts:
        """Reciprocal and self energies (system units) at the given coordination."""
        dtype = positions.dtype
        coefficients = self.coefficients(coordination.astype(dtype))
        geometry = self.mesh.geometry(jnp.asarray(cell_vectors).astype(dtype))
        reciprocal, status = self.reciprocal_energy(positions, geometry, coefficients)
        return FourierD3Parts(
            reciprocal_energy=reciprocal * self.energy_scale,
            self_energy=self.self_energy(coefficients, self.channel_rows, self.active)
            * self.energy_scale,
            cell_successful=geometry.successful,
            reciprocal_successful=status,
            volume=geometry.volume,
        )

    def evaluate(
        self, positions: Array, cell_vectors: Array, coordination: D3Routes, /
    ) -> FourierD3Evaluation:
        """Total FourierD3 energy from explicit full directed coordination routes."""
        numbers, routes_ok, geometry_ok = self.coordination_numbers(coordination)
        parts = self.parts(positions, cell_vectors, numbers)
        total = parts.reciprocal_energy + parts.self_energy
        successful = (
            routes_ok
            & geometry_ok
            & parts.cell_successful
            & parts.reciprocal_successful
            & jnp.isfinite(total)
        )
        evidence = FourierD3Evidence(
            coordination_numbers=numbers,
            coordination_routes_successful=routes_ok,
            geometry_successful=geometry_ok,
            cell_successful=parts.cell_successful,
            reciprocal_successful=parts.reciprocal_successful,
            volume=parts.volume,
            reciprocal_energy=parts.reciprocal_energy,
            self_energy=parts.self_energy,
            method=self.method,
            prepared_identity=self.prepared_identity,
            decomposition_identity=self.plan.decomposition.decomposition_identity,
            coefficient_residual=self.plan.decomposition.achieved_error,
            rank=self.plan.decomposition.rank,
            regularity=self.regularity,
            resources=self.resources,
        )
        return FourierD3Evaluation(
            energy=jnp.where(successful, total, jnp.nan),
            successful=successful,
            evidence=evidence,
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        # The exact numerical system binding is compared as words and folded
        # into the status: a foreign context system refuses without decoding.
        bound = context.system.prepared_identity.equals(self.system.prepared_identity)
        positions = context.positions
        routes = D3Routes.from_view(
            context.neighborhood(self.plan.coordination_requirement),
            positions,
            context.cell_vectors,
            counting=AtomisticPairCounting.DIRECTED,
        )
        result = self.evaluate(positions, context.cell_vectors, routes)
        successful = result.successful & bound
        return AtomisticTermEvaluation(
            jnp.where(successful, result.energy, jnp.nan),
            jnp.zeros((self.system.capacity,), dtype=result.energy.dtype),
            successful,
            atom_energy_complete=False,
            evidence=(result.evidence,),
        )


__all__ = [
    "damped_dispersion_fourier_kernel",
    "FourierD3Evaluation",
    "FourierD3Evidence",
    "FourierD3PairInfluence",
    "FourierD3Parts",
    "FourierD3Potential",
    "FourierD3Regularity",
    "FourierD3Resources",
    "modified_count",
    "PreparedFourierD3Potential",
    "SERIES_CROSSOVER",
    "SERIES_TERMS",
    "SHAPE6_SERIES",
    "SHAPE8_SERIES",
]
