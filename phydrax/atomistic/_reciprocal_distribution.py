#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owner-local distribution of term-owned reciprocal-space work.

Every function named ``owner_*`` runs INSIDE one mapped owner region
(``DistributedOwnershipPlan.map``: device ``shard_map`` or lane-reference
``vmap`` with the same named axis) on this owner's rows. The scientific
definition (influence, spline weights, modulus, Nyquist convention, Ewald
split, multipole basis, LJ-PME/FourierD3/latent kernels, polarization
operator and solver policy) is the canonical single-device term object; this
module only distributes its work.

Particle mesh (PME, multipoles, LJ-PME, FourierD3)
--------------------------------------------------
1. Owned payload channels are spread with the term's own
   :class:`~phydrax.discretization.TensorBSplineSplatAssignment` routes into a
   bounded local *patch* of the fractional mesh. The patch origin follows the
   owner's declared fractional region of the CURRENT cell widened by the reuse
   certificate's Cartesian reach ``r * ||H^{-1}[:, a]||`` (exact for skew
   cells) and the spline support; its static extent is admitted at
   preparation. Every route is checked against the patch at runtime
   (``patch_covered``), separately from the certificate test
   (``patch_certified``).
2. Patch cells are routed by ``all_to_all`` packets of explicit capacity to
   the owners of the canonical physical FFT layout and summed there: x-planes
   to x-slab owners (``schedule="slab"``), or ``(x, y)``-lines to the pencil
   owners of ``pencil_grid=(px, py)`` (``schedule="pencil"``; owner ``r``
   holds x-block ``r % px`` and y-block ``r // px``).
3. The native distributed transform (``DistributedSpectralExecutionPlan.
   forward_local``) yields this owner's modal block (slab: y-rows; pencil:
   z-slab ``r``). Slabs run on the ownership mesh axis (a lane-reference
   topology for lane ownership); pencils on the owner-region topology
   factoring the same owner axis (``SpectralMeshTopology.owner_region``), so
   the canonical pencil transpose is one all-to-all over the owner axis. The
   term's kernel is evaluated over the full complex spectrum of the block
   (identical to the single-device Hermitian half-spectrum sum, including the
   Nyquist sign average of the ``"symmetrized"`` convention).
4. The owner share is the modal sum of its block; system terms (background,
   boundary dipoles, anchors) use ``owner_system_sum`` and are added once by
   owner 0; per-atom terms (self energies) by the owning owner.
5. Actions use ``inverse_local`` on the kernel-weighted block and the exact
   transpose of the packet route back to the patches;
6. spline gathering at owned atoms is the local adjoint of the spreading.

Energies are ordinary JAX through the collectives: every all-to-all
transposes to the reverse all-to-all, so each contribution's cotangent
returns exactly once.

Explicit modes (Ewald, multipole Ewald, latent Ewald)
-----------------------------------------------------
Partial structure factors of owned atoms are formed per bounded mode chunk
and reduced onto the chunk's mode owner by one ``all_to_all`` (a
reduce-scatter whose transpose is the matching gather); each owner evaluates
the kernel on its sub-chunk.

``"replicated-mesh"`` is an explicitly named bounded comparison strategy: the
deposited mesh is owner-summed and every owner transforms it fully, keeping
only its modal block in its share. It is not a distributed FFT.

Owner-local polarization
------------------------
:func:`solve_owner_polarization` solves ``(alpha^-1 - T_u) u = E_p`` with the
native distributed PCG (:func:`phydrax.linalg._distributed.solve_distributed_pcg`,
collective pairings) under ``jax.lax.custom_linear_solve`` (implicit
stationary sensitivity), warm-started from the two-frame predictor, or with
exactly ``tcg_order`` TCG steps (algorithmic sensitivity) when the canonical
solver declares TCG; TCG is never reported converged. Real space runs over an
owner-blocked directed image relation with a halo column exchange; pair
scales use a separately prepared owner-blocked scaled-pair topology.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from enum import StrEnum
from typing import final, Literal, NamedTuple, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import PartitionSpec
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..backends.distributed import JaxCollectiveProvider
from ..discretization import (
    cardinal_bspline_periodic_symbol,
    lattice_measure,
    PeriodicCell,
)
from ..discretization.particle._distributed import FractionalOwnerPartition
from ..discretization.spatial._distributed_relations import (
    _bucket_ranks,
    _pack,
    DistributedHaloPlan,
    DistributedOwnershipPlan,
)
from ..discretization.spectral._distributed import (
    DistributedSpectralExecutionPlan,
    SpectralMeshTopology,
)
from ..discretization.spectral._precision import SpectralPrecisionPolicy
from ..linalg import LinearSolveStatus
from ..linalg._distributed import (
    DistributedKrylovPolicy,
    DistributedLinearOperator,
    DistributedPairing,
    solve_distributed_pcg,
)
from ..linalg.krylov import KrylovBreakdownStatus
from ..typing import parse
from ._dispersion_fourier_d3 import FourierD3PairInfluence, PreparedFourierD3Potential
from ._electrostatics import (
    _coulomb_influence,
    ewald_background_energy,
    ewald_self_energy,
    ewald_slab_boundary_energy,
    ewald_slab_height,
    ewald_vacuum_boundary_energy,
    PreparedEwaldReferencePotential,
    PreparedParticleMeshEwaldPotential,
)
from ._feature_execution import HaloColumnExchange, owner_system_sum
from ._force_field import (
    ForceFieldTermKind,
    lennard_jones_pme_influence,
    lennard_jones_pme_self_energy,
    PreparedGeneralForceFieldTerm,
)
from ._latent_electrostatics import (
    _orientation,
    latent_ewald_influence,
    PreparedLatentEwald,
)
from ._multipole_basis import (
    CartesianMultipoles,
    co_centered_interaction,
    component_count,
    dipole_multipoles,
    erf_kernel_origin_coefficients,
    GaussianMultipoleWidths,
)
from ._multipole_electrostatics import (
    _channel_values,
    _combined_density,
    _ewald_kernel,
    _mode_coefficients,
    _pair_interaction,
    _PairSites,
    _PeriodicRouteSettings,
    AbstractMultipoleRadialKernel,
    ErfcKernel,
    FiniteMultipoleRoute,
    GaussianEwaldRealKernel,
    MultipolePairScales,
    MultipoleSiteSet,
    PreparedMultipoleElectrostatics,
)
from ._polarization import (
    ExponentialDampedCoulombKernel,
    PolarizationPredictorState,
    PolarizationSolverKind,
    PolarizationStabilityEvidence,
    PolarizationState,
    PreparedPolarizationSolver,
)
from ._reciprocal_mesh import (
    _lattice_geometry,
    _masked_kernel,
    PreparedReciprocalMesh,
    PreparedReciprocalModeSet,
    ReciprocalGeometry,
)


if TYPE_CHECKING:
    # The domain plan imports this module; the topology type is annotation-only.
    from ._domain_state import AtomisticDomainTopology


ReductionMode: TypeAlias = Literal["fast", "deterministic", "compensated"]
AtomisticReciprocalStrategy: TypeAlias = Literal["distributed-fft", "replicated-mesh"]
AtomisticReciprocalSchedule: TypeAlias = Literal["slab", "pencil"]
ReciprocalDefinition: TypeAlias = (
    PreparedParticleMeshEwaldPotential
    | PreparedEwaldReferencePotential
    | PreparedMultipoleElectrostatics
    | PreparedGeneralForceFieldTerm
    | PreparedFourierD3Potential
    | PreparedLatentEwald
)

_REAL_ITEMSIZE = 8
_COMPLEX_ITEMSIZE = 16
_SCALE_COLUMNS = ("direct", "polarization", "mutual")


class AtomisticReciprocalKind(StrEnum):
    """Term family whose reciprocal work is distributed (physics stays term-owned)."""

    PME = "pme"
    EWALD = "ewald"
    MULTIPOLE = "multipole"
    LJ_PME = "lj-pme"
    FOURIER_D3 = "fourier-d3"
    LATENT_EWALD = "latent-ewald"


class ReciprocalDistributionCapacityError(MemoryError):
    """Preparation refusal of a static patch, packet or buffer capacity."""


class ReciprocalSitePayload(NamedTuple):
    """Owner-blocked per-atom inputs of one reciprocal term (rows of this owner).

    * PME / EWALD: ``values`` charges ``(n,)``; ``image_counts`` ``(n, 3)``
      integer lattice images (unwrapped = stored + counts @ H) for the
      vacuum-sphere boundary.
    * LATENT_EWALD: ``values`` latent channels ``(n,)`` or ``(n, C)``.
    * LJ_PME: ``rows`` site type ids ``(n,)``; ``anchor`` ``(n,)`` marks the
      term's anchor site (exactly one row globally) of the anchor-source frame.
    * FOURIER_D3: ``values`` signed rank coefficients ``(n, padded_rank)``;
      ``rows`` species channel rows ``(n,)``.
    * MULTIPOLE: ``moments`` owned :class:`CartesianMultipoles`; ``rows``
      Gaussian width classes ``(n,)`` (Gaussian sites only).
    """

    values: Array | None = None
    moments: CartesianMultipoles | None = None
    rows: Array | None = None
    image_counts: Array | None = None
    anchor: Array | None = None


ChannelFunction: TypeAlias = Callable[[ReciprocalSitePayload], Array]


def _positive_int(value: int, name: str, /) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _nonnegative_float(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _kind_of(definition: ReciprocalDefinition, /) -> AtomisticReciprocalKind:
    match definition:
        case PreparedParticleMeshEwaldPotential():
            return AtomisticReciprocalKind.PME
        case PreparedEwaldReferencePotential():
            return AtomisticReciprocalKind.EWALD
        case PreparedMultipoleElectrostatics():
            return AtomisticReciprocalKind.MULTIPOLE
        case PreparedGeneralForceFieldTerm():
            if definition.plan.kind is not ForceFieldTermKind.LENNARD_JONES_PME:
                raise ValueError(
                    "Only the Lennard-Jones PME force-field term has reciprocal work."
                )
            return AtomisticReciprocalKind.LJ_PME
        case PreparedFourierD3Potential():
            return AtomisticReciprocalKind.FOURIER_D3
        case PreparedLatentEwald():
            return AtomisticReciprocalKind.LATENT_EWALD
    raise TypeError(
        "definition must be a prepared PME, Ewald, multipole, LJ-PME, FourierD3 or "
        "latent Ewald term."
    )


def _definition_mesh(
    definition: ReciprocalDefinition, /
) -> PreparedReciprocalMesh | None:
    match definition:
        case PreparedParticleMeshEwaldPotential():
            return definition.mesh
        case PreparedMultipoleElectrostatics():
            return definition.mesh
        case PreparedGeneralForceFieldTerm():
            return definition.reciprocal_mesh
        case PreparedFourierD3Potential():
            return definition.mesh
    return None


def _definition_modes(
    definition: ReciprocalDefinition, /
) -> PreparedReciprocalModeSet | None:
    match definition:
        case PreparedEwaldReferencePotential():
            return definition.modes
        case PreparedMultipoleElectrostatics():
            return definition.modes
        case PreparedLatentEwald():
            return definition.modes
    return None


def _channel_degree(
    plan: "AtomisticReciprocalDistributionPlan",
    definition: PreparedMultipoleElectrostatics,
    /,
) -> int:
    """Multipole channel degree; the polarization bilinear needs dipole channels."""
    return max(definition.definition.maximum_degree, 1 if plan.bilinear else 0)


def _periodic_route_settings(
    definition: PreparedMultipoleElectrostatics, /
) -> _PeriodicRouteSettings:
    """Ewald split of a periodic multipole route (finite routes have none)."""
    route = definition.plan.route
    if isinstance(route, FiniteMultipoleRoute):
        raise RuntimeError("A finite multipole route has no reciprocal Ewald split.")
    return route.settings


# --- plan ---------------------------------------------------------------------------


@final
class AtomisticReciprocalDistributionPlan(StrictModule, NonTrainableState):
    """Distribution of one term-owned reciprocal definition over domain owners.

    ``definition`` is the canonical prepared term (its mesh/mode set, kernel
    and conventions are used unchanged). ``reuse_reach`` is the Cartesian
    reach of the domain reuse certificate: owned atoms lie in their owner's
    fractional region widened by ``reuse_reach * ||H^{-1}[:, a]||`` in the
    current cell. ``cell_strain_allowance`` bounds the relative growth of
    ``||H^{-1}[:, a]||`` admitted by the static patch extent;
    ``patch_margin_cells`` adds explicit mesh cells. ``schedule`` selects the
    canonical distributed FFT layout of the ``"distributed-fft"`` strategy:
    ``"slab"`` (x-slabs, one owner axis) or ``"pencil"`` with the explicit
    owner grid ``pencil_grid=(px, py)``; owner ``r`` holds the physical
    block ``(r % px, r // px)`` (x-block, y-block) and the modal z-slab ``r``.
    ``packet_planes`` is the per-destination plane capacity of the slab patch
    route and ``packet_lines`` the per-destination ``(x, y)``-line capacity of
    the pencil route (defaults: their exact upper bounds). ``mode_chunk``
    overrides the explicit-mode chunk (rounded up to the owner count).
    ``widths`` are the canonical Gaussian width classes of a Gaussian
    multipole definition (per-atom class ids travel in the payload).
    ``bilinear`` admits the two-set multipole bilinear (polarization).

    PyTree layout: ``plan_identity`` is the exact canonical plan digest
    (``plan_id`` decodes it on the host) carried as dynamic `IdentityWords`, so
    revisions of the definition share one executable. Every other field is
    declared configuration: the term kind, strategy, FFT schedule and grid,
    reuse reach and strain allowance (they size the static patch),
    capacities, byte envelopes and the reduction order. ``scientific_identity``
    (host ``scientific_id``), also dynamic `IdentityWords`, is the
    layout-independent digest of what is computed (term kind, canonical
    definition, Gaussian width classes, bilinear form) without the strategy,
    schedule, owner grid, patch sizing, capacities, byte envelopes or
    reduction order, which only place and order that computation.
    """

    definition: ReciprocalDefinition
    class_widths: Array | None
    kind: AtomisticReciprocalKind = eqx.field(static=True)
    strategy: AtomisticReciprocalStrategy = eqx.field(static=True)
    schedule: AtomisticReciprocalSchedule = eqx.field(static=True)
    pencil_grid: tuple[int, int] | None = eqx.field(static=True)
    reuse_reach: float = eqx.field(static=True)
    cell_strain_allowance: float = eqx.field(static=True)
    patch_margin_cells: int = eqx.field(static=True)
    packet_planes: int | None = eqx.field(static=True)
    packet_lines: int | None = eqx.field(static=True)
    mode_chunk: int | None = eqx.field(static=True)
    bilinear: bool = eqx.field(static=True)
    maximum_fft_bytes: int = eqx.field(static=True)
    maximum_packet_bytes: int = eqx.field(static=True)
    maximum_replicated_bytes: int = eqx.field(static=True)
    reduction_mode: ReductionMode = eqx.field(static=True)
    plan_identity: IdentityWords
    scientific_identity: IdentityWords

    def __init__(
        self,
        definition: ReciprocalDefinition,
        /,
        *,
        reuse_reach: float,
        strategy: AtomisticReciprocalStrategy = "distributed-fft",
        schedule: AtomisticReciprocalSchedule = "slab",
        pencil_grid: tuple[int, int] | None = None,
        cell_strain_allowance: float = 0.0,
        patch_margin_cells: int = 1,
        packet_planes: int | None = None,
        packet_lines: int | None = None,
        mode_chunk: int | None = None,
        widths: GaussianMultipoleWidths | None = None,
        bilinear: bool = False,
        maximum_fft_bytes: int = 2 * 1024**3,
        maximum_packet_bytes: int = 1 << 28,
        maximum_replicated_bytes: int = 1 << 28,
        reduction_mode: ReductionMode = "deterministic",
    ) -> None:
        kind = _kind_of(definition)
        reach = _nonnegative_float(reuse_reach, "reuse_reach")
        strain = _nonnegative_float(cell_strain_allowance, "cell_strain_allowance")
        if isinstance(patch_margin_cells, bool) or int(patch_margin_cells) < 0:
            raise ValueError("patch_margin_cells must be a non-negative integer.")
        strategy = parse(strategy, AtomisticReciprocalStrategy, "strategy")
        reduction_mode = parse(reduction_mode, ReductionMode, "reduction_mode")
        mesh = _definition_mesh(definition)
        modes = _definition_modes(definition)
        if mesh is None and modes is None:
            raise ValueError(
                "The definition has no reciprocal mesh or mode set; a finite route is "
                "a complete pair sum executed by the domain pair relation."
            )
        if mesh is None and strategy == "replicated-mesh":
            raise ValueError("The replicated-mesh strategy applies to particle meshes.")
        if packet_planes is not None:
            _positive_int(packet_planes, "packet_planes")
        if mode_chunk is not None:
            _positive_int(mode_chunk, "mode_chunk")
        if mesh is None and packet_planes is not None:
            raise ValueError("packet_planes applies to particle-mesh routes only.")
        if mesh is not None and mode_chunk is not None:
            raise ValueError("mode_chunk applies to explicit-mode routes only.")
        schedule = parse(schedule, AtomisticReciprocalSchedule, "schedule")
        grid: tuple[int, int] | None = None
        if schedule == "pencil":
            if mesh is None:
                raise ValueError(
                    "The pencil schedule is a particle-mesh FFT layout; explicit "
                    "modes have no distributed transform."
                )
            if strategy != "distributed-fft":
                raise ValueError(
                    "The pencil schedule is a distributed-FFT layout; the replicated "
                    "mesh transforms the full mesh on every owner."
                )
            if packet_planes is not None:
                raise ValueError("packet_planes is the slab route capacity.")
            if pencil_grid is None or len(tuple(pencil_grid)) != 2:
                raise ValueError(
                    "The pencil schedule needs its explicit owner grid (px, py)."
                )
            px, py = tuple(pencil_grid)
            grid = (
                _positive_int(px, "pencil_grid[0]"),
                _positive_int(py, "pencil_grid[1]"),
            )
            if packet_lines is not None:
                _positive_int(packet_lines, "packet_lines")
        else:
            if pencil_grid is not None:
                raise ValueError("pencil_grid belongs to the pencil schedule.")
            if packet_lines is not None:
                raise ValueError("packet_lines is the pencil route capacity.")
        if bilinear and kind is not AtomisticReciprocalKind.MULTIPOLE:
            raise ValueError("The bilinear form is a multipole (polarization) action.")
        class_widths: Array | None = None
        if isinstance(definition, PreparedMultipoleElectrostatics):
            gaussian = definition.definition.site_kind == "gaussian"
            if gaussian != (widths is not None):
                raise ValueError(
                    "Gaussian multipole sites require their canonical width classes; "
                    "point sites take none."
                )
            if widths is not None:
                if not isinstance(widths, GaussianMultipoleWidths):
                    raise TypeError("widths must be GaussianMultipoleWidths.")
                class_widths = jnp.asarray(widths.class_widths)
        elif widths is not None:
            raise ValueError("Width classes belong to Gaussian multipole definitions.")
        if isinstance(definition, PreparedLatentEwald):
            if definition.definition.branch == "finite":
                raise ValueError(
                    "A finite-branch latent Ewald has no reciprocal work: it is a "
                    "complete Gaussian pair sum of the domain pair relation."
                )
        self.definition = definition
        self.class_widths = class_widths
        self.kind = kind
        self.strategy = strategy
        self.schedule = schedule
        self.pencil_grid = grid
        self.reuse_reach = reach
        self.cell_strain_allowance = strain
        self.patch_margin_cells = int(patch_margin_cells)
        self.packet_planes = None if packet_planes is None else int(packet_planes)
        self.packet_lines = None if packet_lines is None else int(packet_lines)
        self.mode_chunk = None if mode_chunk is None else int(mode_chunk)
        self.bilinear = bool(bilinear)
        self.maximum_fft_bytes = _positive_int(maximum_fft_bytes, "maximum_fft_bytes")
        self.maximum_packet_bytes = _positive_int(
            maximum_packet_bytes, "maximum_packet_bytes"
        )
        self.maximum_replicated_bytes = _positive_int(
            maximum_replicated_bytes, "maximum_replicated_bytes"
        )
        self.reduction_mode = reduction_mode
        self.scientific_identity = IdentityWords.of(
            {
                "kind": "atomistic-reciprocal-definition",
                "term_kind": kind.value,
                "definition": definition.prepared_id,
                "widths": None if widths is None else widths.widths_id,
                "bilinear": self.bilinear,
            }
        )
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-reciprocal-distribution-plan",
                "term_kind": kind.value,
                "definition": definition.prepared_id,
                "strategy": strategy,
                "schedule": schedule,
                "pencil_grid": None if grid is None else list(grid),
                "reuse_reach": float(reach).hex(),
                "cell_strain_allowance": float(strain).hex(),
                "patch_margin_cells": self.patch_margin_cells,
                "packet_planes": self.packet_planes,
                "packet_lines": self.packet_lines,
                "mode_chunk": self.mode_chunk,
                "widths": None if widths is None else widths.widths_id,
                "bilinear": self.bilinear,
                "maximum_fft_bytes": self.maximum_fft_bytes,
                "maximum_packet_bytes": self.maximum_packet_bytes,
                "maximum_replicated_bytes": self.maximum_replicated_bytes,
                "reduction_mode": reduction_mode,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical distribution-plan digest."""
        return self.plan_identity.fingerprint()

    @property
    def scientific_id(self) -> str:
        """Host decoding of the layout-independent definition digest."""
        return self.scientific_identity.fingerprint()


# --- prepared layouts --------------------------------------------------------------


@final
class _MeshLayout(StrictModule, NonTrainableState):
    """Host-prepared patch, packet and full-spectrum tables of one mesh.

    ``schedule`` is the transfer layout: ``"slab"`` routes patch x-planes to
    x-slab owners (``block_shape = (K0/P, K1, K2)``), ``"pencil"`` routes
    patch ``(x, y)``-lines to pencil owners (``block_shape =
    (K0/px, K1/py, K2)``, owner ``r`` at grid coordinate ``(r % px, r // px)``).
    ``packet_capacity`` counts planes (slab) or lines (pencil) per
    destination. The modal block of owner ``r`` is rows
    ``[r * modal_rows, (r + 1) * modal_rows)`` of ``modal_axis`` (1 for slabs
    and the replicated mesh, 2 for pencils).
    """

    region_lower: Array
    region_upper: Array
    axis_modes: tuple[Array, Array, Array]
    axis_symbols: tuple[Array, Array, Array]
    axis_nyquist: tuple[Array, Array, Array]
    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    patch_shape: tuple[int, int, int] = eqx.field(static=True)
    schedule: AtomisticReciprocalSchedule = eqx.field(static=True)
    spectral_grid: tuple[int, ...] = eqx.field(static=True)
    block_shape: tuple[int, int, int] = eqx.field(static=True)
    packet_capacity: int = eqx.field(static=True)
    modal_axis: int = eqx.field(static=True)
    modal_rows: int = eqx.field(static=True)
    coordinate_offsets: tuple[float, float, float] = eqx.field(static=True)
    variants: tuple[tuple[bool, bool, bool], ...] = eqx.field(static=True)
    exclude_nyquist: bool = eqx.field(static=True)
    packet_bytes: int = eqx.field(static=True)
    patch_bytes: int = eqx.field(static=True)


@final
class _ModeLayout(StrictModule, NonTrainableState):
    """Owner-divisible explicit-mode chunks of one canonical mode set."""

    modes: Array
    weights: Array
    zero: Array
    chunk: int = eqx.field(static=True)
    chunk_count: int = eqx.field(static=True)
    sub_chunk: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)


def _channel_count(plan: AtomisticReciprocalDistributionPlan, mesh_route: bool, /) -> int:
    definition = plan.definition
    match definition:
        case PreparedParticleMeshEwaldPotential() | PreparedEwaldReferencePotential():
            return 1
        case PreparedLatentEwald():
            return int(definition.definition.channels)
        case PreparedGeneralForceFieldTerm():
            return int(definition.plan.arrays[2].shape[1])
        case PreparedFourierD3Potential():
            return int(definition.resources.species * definition.resources.padded_rank)
        case PreparedMultipoleElectrostatics():
            if not mesh_route:
                return 1
            degree = _channel_degree(plan, definition)
            classes = 1 if plan.class_widths is None else int(plan.class_widths.shape[0])
            return classes * component_count("cartesian-primitive", degree)
    raise RuntimeError(f"Unhandled reciprocal kind {plan.kind!r}.")


def _mesh_layout(
    plan: AtomisticReciprocalDistributionPlan,
    mesh: PreparedReciprocalMesh,
    ownership: DistributedOwnershipPlan,
    partition: FractionalOwnerPartition,
    channels: int,
    /,
) -> _MeshLayout:
    shape = mesh.plan.grid_shape
    owners = ownership.owner_count
    pencil_grid = plan.pencil_grid
    if plan.schedule == "pencil":
        if pencil_grid is None:
            raise RuntimeError("A pencil plan carries its owner grid.")
        px, py = pencil_grid
        if px * py != owners:
            raise ValueError(
                f"pencil_grid {pencil_grid} does not factor the {owners} owners "
                "of the ownership axis."
            )
        if shape[0] % px or shape[1] % py or shape[2] % owners:
            raise ValueError(
                f"Mesh {shape} is not divisible into {px} x-blocks, {py} y-blocks "
                f"and {owners} modal z-slabs."
            )
    elif shape[0] % owners or shape[1] % owners:
        raise ValueError(
            f"Mesh {shape} is not divisible into {owners} x-slabs and y-modal rows."
        )
    if partition.cell.cell_id != mesh.cell.cell_id:
        raise ValueError(
            "The owner partition and the term mesh must use the same periodic cell frame."
        )
    if not mesh.cell.fully_periodic:
        raise ValueError("Particle-mesh distribution requires a fully periodic cell.")
    if partition.owner_count != owners:
        raise ValueError("Partition and ownership owner counts differ.")
    degree = mesh.plan.spline_degree
    counts = np.asarray(partition.counts, dtype=np.float64)
    grid = np.asarray(
        tuple(itertools.product(*(range(count) for count in partition.counts))),
        dtype=np.float64,
    ).reshape((owners, 3))
    lower = grid / counts
    upper = (grid + 1.0) / counts
    inverse = np.linalg.inv(np.asarray(mesh.cell.vectors, dtype=np.float64))
    reach = (
        plan.reuse_reach
        * np.sqrt(np.sum(inverse * inverse, axis=0))
        * (1.0 + plan.cell_strain_allowance)
    )
    width = np.max(upper - lower, axis=0)
    extents = []
    for axis, count in enumerate(shape):
        needed = (
            int(math.floor((width[axis] + 2.0 * reach[axis]) * count))
            + degree
            + 3
            + plan.patch_margin_cells
        )
        extents.append(min(needed, count))
    patch = (extents[0], extents[1], extents[2])
    if pencil_grid is not None:
        block = (shape[0] // pencil_grid[0], shape[1] // pencil_grid[1], shape[2])
        bound = min(patch[0], block[0]) * min(patch[1], block[1])
        requested = plan.packet_lines
        unit, unit_values = "packet_lines", patch[2]
        spectral_grid: tuple[int, ...] = pencil_grid
        modal_axis, modal_rows = 2, shape[2] // owners
    else:
        block = (shape[0] // owners, shape[1], shape[2])
        bound = min(patch[0], block[0])
        requested = plan.packet_planes
        unit, unit_values = "packet_planes", patch[1] * patch[2]
        spectral_grid = (owners,)
        modal_axis, modal_rows = 1, shape[1] // owners
    capacity = bound if requested is None else requested
    if capacity > bound:
        raise ValueError(
            f"{unit}={capacity} exceeds the exact per-destination bound {bound}."
        )
    packet_bytes = 2 * owners * capacity * unit_values * channels * _REAL_ITEMSIZE
    patch_bytes = int(np.prod(patch)) * channels * _REAL_ITEMSIZE
    if plan.strategy == "distributed-fft" and packet_bytes > plan.maximum_packet_bytes:
        raise ReciprocalDistributionCapacityError(
            f"Patch packets need {packet_bytes} bytes (patch {patch}, {unit}="
            f"{capacity} per destination, {channels} channels), exceeding "
            f"maximum_packet_bytes={plan.maximum_packet_bytes}."
        )
    if plan.strategy == "replicated-mesh":
        mode_factor = 1 if plan.reduction_mode == "fast" else owners
        replicated = (
            int(np.prod(shape))
            * channels
            * (_REAL_ITEMSIZE * mode_factor + _COMPLEX_ITEMSIZE)
        )
        if replicated > plan.maximum_replicated_bytes:
            raise ReciprocalDistributionCapacityError(
                f"The replicated mesh needs {replicated} bytes per owner, exceeding "
                f"maximum_replicated_bytes={plan.maximum_replicated_bytes}."
            )
    offsets = tuple(
        float(np.asarray(coordinates, dtype=np.float64)[0])
        for coordinates in mesh.splat.layout.coordinates_by_axis
    )
    modes = tuple(np.rint(np.fft.fftfreq(count) * count) for count in shape)
    if mesh.plan.modulus == "discrete-bspline-modulus":
        symbols = tuple(
            cardinal_bspline_periodic_symbol(degree, count) for count in shape
        )
    else:
        symbols = tuple(
            np.sinc(axis_modes / count) ** (degree + 1)
            for axis_modes, count in zip(modes, shape, strict=True)
        )
    nyquist = tuple(
        (2.0 * np.abs(axis_modes) == count)
        if count % 2 == 0
        else np.zeros((count,), dtype=np.bool_)
        for axis_modes, count in zip(modes, shape, strict=True)
    )
    if mesh.plan.nyquist == "symmetrized":
        even = [axis for axis, count in enumerate(shape) if count % 2 == 0]
        variants = []
        for flips in itertools.product((False, True), repeat=len(even)):
            flag = [False, False, False]
            for axis, flip in zip(even, flips, strict=True):
                flag[axis] = flip
            variants.append((flag[0], flag[1], flag[2]))
        variant_tuple = tuple(variants)
    else:
        variant_tuple = ((False, False, False),)
    return _MeshLayout(
        region_lower=jnp.asarray(lower),
        region_upper=jnp.asarray(upper),
        axis_modes=(
            jnp.asarray(modes[0]),
            jnp.asarray(modes[1]),
            jnp.asarray(modes[2]),
        ),
        axis_symbols=(
            jnp.asarray(symbols[0]),
            jnp.asarray(symbols[1]),
            jnp.asarray(symbols[2]),
        ),
        axis_nyquist=(
            jnp.asarray(nyquist[0]),
            jnp.asarray(nyquist[1]),
            jnp.asarray(nyquist[2]),
        ),
        grid_shape=shape,
        degree=degree,
        channels=channels,
        patch_shape=patch,
        schedule=plan.schedule,
        spectral_grid=spectral_grid,
        block_shape=block,
        packet_capacity=capacity,
        modal_axis=modal_axis,
        modal_rows=modal_rows,
        coordinate_offsets=(offsets[0], offsets[1], offsets[2]),
        variants=variant_tuple,
        exclude_nyquist=mesh.plan.nyquist == "excluded",
        packet_bytes=packet_bytes,
        patch_bytes=patch_bytes,
    )


def _mode_layout(
    plan: AtomisticReciprocalDistributionPlan,
    modes: PreparedReciprocalModeSet,
    owners: int,
    channels: int,
    /,
) -> _ModeLayout:
    base = plan.mode_chunk if plan.mode_chunk is not None else modes.mode_chunk
    chunk = -(-base // owners) * owners
    stored = np.asarray(modes.modes)
    weights = np.asarray(modes.hermitian_weights)
    zero = np.asarray(modes.zero_mode)
    count = stored.shape[0]
    chunks = -(-count // chunk)
    padding = chunks * chunk - count
    stored = np.concatenate((stored, np.zeros((padding, 3), dtype=stored.dtype)))
    weights = np.concatenate((weights, np.zeros((padding,), dtype=weights.dtype)))
    zero = np.concatenate((zero, np.zeros((padding,), dtype=np.bool_)))
    return _ModeLayout(
        modes=jnp.asarray(stored.reshape((chunks, chunk, 3))),
        weights=jnp.asarray(weights.reshape((chunks, chunk))),
        zero=jnp.asarray(zero.reshape((chunks, chunk))),
        chunk=chunk,
        chunk_count=chunks,
        sub_chunk=chunk // owners,
        channels=channels,
    )


@final
class ReciprocalDistributionReport(StrictModule, NonTrainableState):
    """Static resources of one prepared distribution.

    Every resource field is declared layout; ``report_identity`` is the exact
    canonical report digest (it binds the content-addressed distribution plan)
    carried as dynamic `IdentityWords`, decoded by ``report_id`` on the host.

    Particle-mesh transfer layout: ``schedule`` (``"slab"``/``"pencil"``;
    ``None`` for explicit modes), ``spectral_grid`` (``(P,)`` or ``(px, py)``),
    ``owner_coordinates[r]`` the spectral grid coordinate of owner ``r``,
    ``owner_devices[r]`` its ``(process, device id)`` (the lane device for a
    lane reference), the canonical physical/modal layout ids and LOCAL block
    shapes (channels last), and ``collective_order`` the exact execution order
    of the transfer and spectral collectives (forward, then the reverse of
    the action route).
    """

    strategy: str = eqx.field(static=True)
    owner_count: int = eqx.field(static=True)
    lane_reference: bool = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    grid_shape: tuple[int, ...] | None = eqx.field(static=True)
    patch_shape: tuple[int, ...] | None = eqx.field(static=True)
    packet_planes: int | None = eqx.field(static=True)
    packet_lines: int | None = eqx.field(static=True)
    packet_bytes: int = eqx.field(static=True)
    patch_bytes: int = eqx.field(static=True)
    mode_chunk: int | None = eqx.field(static=True)
    mode_chunks: int | None = eqx.field(static=True)
    schedule: str | None = eqx.field(static=True)
    spectral_grid: tuple[int, ...] | None = eqx.field(static=True)
    owner_coordinates: tuple[tuple[int, ...], ...] | None = eqx.field(static=True)
    owner_devices: tuple[tuple[int, int], ...] | None = eqx.field(static=True)
    physical_layout_id: str | None = eqx.field(static=True)
    modal_layout_id: str | None = eqx.field(static=True)
    physical_block_shape: tuple[int, ...] | None = eqx.field(static=True)
    modal_block_shape: tuple[int, ...] | None = eqx.field(static=True)
    collective_order: tuple[str, ...] = eqx.field(static=True)
    spectral_topology_id: str | None = eqx.field(static=True)
    spectral_plan_id: str | None = eqx.field(static=True)
    report_identity: IdentityWords

    @property
    def report_id(self) -> str:
        """Host decoding of the canonical report digest."""
        return self.report_identity.fingerprint()


def _spectral_topology(
    plan: AtomisticReciprocalDistributionPlan,
    ownership: DistributedOwnershipPlan,
    /,
) -> SpectralMeshTopology:
    """Canonical spectral topology bound to the ownership devices and owner axis.

    Slabs use the one-axis group mesh (or its lane reference) directly. A
    pencil grid factors the SAME owner axis: owner ``r`` (device ``r`` of the
    ownership group mesh, or lane ``r``) is spectral coordinate
    ``(r % px, r // px)``, the flattened order of the canonical full-mesh
    pencil all-to-all.
    """
    owners = ownership.owner_count
    group = ownership.execution_group
    axis = ownership.axis_name
    if plan.schedule == "slab":
        if ownership.lane_reference:
            return SpectralMeshTopology.lane_reference(
                owners,
                axis_name=axis,
                device=group.devices[0],
                execution_group_id=group.spec.group_id,
            )
        topology = SpectralMeshTopology.from_execution_group(group)
        if topology.mesh_axis_names != (axis,):
            raise ValueError("The spectral mesh axis must be the ownership axis.")
        return topology
    grid = plan.pencil_grid
    if grid is None:
        raise RuntimeError("A pencil plan carries its owner grid.")
    if ownership.lane_reference:
        devices: tuple[jax.Device, ...] = (group.devices[0],)
    else:
        if tuple(str(name) for name in group.mesh.axis_names) != (axis,):
            raise ValueError("Pencil owners are the ownership axis of its group mesh.")
        devices = tuple(group.mesh.devices.flat)
        processes = tuple(int(device.process_index) for device in devices)
        if len(devices) != owners or processes != tuple(ownership.owner_processes):
            raise ValueError(
                "The ownership group devices do not match its owner-to-process map."
            )
    return SpectralMeshTopology.owner_region(
        grid,
        owner_axis=axis,
        devices=devices,
        lanes=ownership.lane_reference,
        execution_group_id=group.spec.group_id,
    )


def _require_transfer_layout(
    layout: _MeshLayout, spectral: DistributedSpectralExecutionPlan, stored: int, /
) -> None:
    """The particle transfer blocks are exactly the canonical local layouts."""
    topology = spectral.topology
    physical = spectral.physical_layout.local_shape(topology)
    modal = spectral.modal_layout.local_shape(topology)
    modal_shape = list(layout.grid_shape)
    modal_shape[layout.modal_axis] = layout.modal_rows
    expected_physical = (*layout.block_shape, stored)
    expected_modal = (*modal_shape, stored)
    if physical != expected_physical or modal != expected_modal:
        raise RuntimeError(
            f"The {layout.schedule} transfer blocks {expected_physical}/"
            f"{expected_modal} differ from the canonical spectral layouts "
            f"{physical}/{modal}."
        )


class _TransferReport(NamedTuple):
    owner_coordinates: tuple[tuple[int, ...], ...] | None
    owner_devices: tuple[tuple[int, int], ...] | None
    physical_layout_id: str | None
    modal_layout_id: str | None
    physical_block_shape: tuple[int, ...] | None
    modal_block_shape: tuple[int, ...] | None
    collective_order: tuple[str, ...]


def _transfer_report(
    layout: _MeshLayout | None,
    spectral: DistributedSpectralExecutionPlan | None,
    ownership: DistributedOwnershipPlan,
    /,
) -> _TransferReport:
    owners = ownership.owner_count
    if layout is None:
        return _TransferReport(
            None, None, None, None, None, None, ("mode-chunk-reduction:all_to_all",)
        )
    group = ownership.execution_group
    if ownership.lane_reference:
        lane = group.devices[0]
        devices = ((int(lane.process_index), int(lane.id)),) * owners
    else:
        devices = tuple(
            (int(device.process_index), int(device.id))
            for device in group.mesh.devices.flat
        )
    modal_shape = list(layout.grid_shape)
    modal_shape[layout.modal_axis] = layout.modal_rows
    modal_block = (*modal_shape, layout.channels)
    if spectral is None:
        return _TransferReport(
            tuple((owner,) for owner in range(owners)),
            devices,
            None,
            None,
            (*layout.grid_shape, layout.channels),
            modal_block,
            ("mesh-sum:owner_system_sum", "potential-sum:owner_system_sum"),
        )
    if layout.schedule == "pencil":
        px = layout.spectral_grid[0]
        coordinates = tuple((owner % px, owner // px) for owner in range(owners))
        unit = "line"
    else:
        coordinates = tuple((owner,) for owner in range(owners))
        unit = "plane"
    report = spectral.report
    return _TransferReport(
        coordinates,
        devices,
        spectral.physical_layout.layout_id,
        spectral.modal_layout.layout_id,
        (*layout.block_shape, layout.channels),
        modal_block,
        (
            f"patch-{unit}-header:all_to_all",
            "patch-origin:all_gather",
            f"patch-{unit}-packets:all_to_all",
            f"spectral-forward:{report.forward_sequence_id}",
            f"spectral-inverse:{report.inverse_sequence_id}",
            f"patch-{unit}-adjoint:all_to_all",
        ),
    )


@final
class PreparedReciprocalDistribution(StrictModule, NonTrainableState):
    """One reciprocal definition bound to an ownership plan and owner partition.

    ``prepared_identity`` is the exact canonical preparation digest as dynamic
    `IdentityWords` (``prepared_id`` decodes it on the host); axis, owner count
    and local capacity are declared layout.
    """

    plan: AtomisticReciprocalDistributionPlan
    mesh_layout: _MeshLayout | None
    mode_layout: _ModeLayout | None
    spectral: DistributedSpectralExecutionPlan | None
    report: ReciprocalDistributionReport
    axis_name: str = eqx.field(static=True)
    owner_count: int = eqx.field(static=True)
    local_capacity: int = eqx.field(static=True)
    prepared_identity: IdentityWords

    def __init__(
        self,
        plan: AtomisticReciprocalDistributionPlan,
        ownership: DistributedOwnershipPlan,
        partition: FractionalOwnerPartition,
        /,
    ) -> None:
        if not isinstance(plan, AtomisticReciprocalDistributionPlan):
            raise TypeError("plan must be an AtomisticReciprocalDistributionPlan.")
        if not isinstance(ownership, DistributedOwnershipPlan):
            raise TypeError("ownership must be a DistributedOwnershipPlan.")
        if not isinstance(partition, FractionalOwnerPartition):
            raise TypeError("partition must be a FractionalOwnerPartition.")
        owners = ownership.owner_count
        definition = plan.definition
        mesh = _definition_mesh(definition)
        modes = _definition_modes(definition)
        if isinstance(
            definition,
            (PreparedParticleMeshEwaldPotential, PreparedEwaldReferencePotential),
        ):
            boundary = definition.plan.settings.boundary
            if boundary == "slab-yeh-berkowitz" and mesh is not None:
                raise ValueError(
                    "The slab convention has two periodic axes; a particle mesh needs a "
                    "fully periodic cell."
                )
        mesh_layout = mode_layout = None
        spectral = None
        channels = _channel_count(plan, mesh is not None)
        if mesh is not None:
            stored = channels * (2 if plan.bilinear else 1)
            mesh_layout = _mesh_layout(plan, mesh, ownership, partition, stored)
            if plan.strategy == "distributed-fft":
                topology = _spectral_topology(plan, ownership)
                shape = mesh.plan.grid_shape
                spectral = DistributedSpectralExecutionPlan(
                    topology,
                    shape,
                    owner_id="atomistic-reciprocal-distribution",
                    admitted_payload_shapes=tuple(sorted({(channels,), (stored,)})),
                    precision=SpectralPrecisionPolicy(np.float64),
                    schedule=plan.schedule,
                    state_shape=(stored,),
                    transform_scale=math.sqrt(float(np.prod(shape))),
                    maximum_bytes=plan.maximum_fft_bytes,
                )
                _require_transfer_layout(mesh_layout, spectral, stored)
        else:
            if modes is None:
                raise RuntimeError("Reciprocal definition without mesh or modes.")
            mode_layout = _mode_layout(plan, modes, owners, channels)
        self.plan = plan
        self.mesh_layout = mesh_layout
        self.mode_layout = mode_layout
        self.spectral = spectral
        self.axis_name = ownership.axis_name
        self.owner_count = owners
        self.local_capacity = ownership.local_capacity
        transfer = _transfer_report(mesh_layout, spectral, ownership)
        self.report = ReciprocalDistributionReport(
            strategy="explicit-modes" if mesh is None else plan.strategy,
            owner_count=owners,
            lane_reference=ownership.lane_reference,
            channels=channels,
            grid_shape=None if mesh is None else mesh.plan.grid_shape,
            patch_shape=None if mesh_layout is None else mesh_layout.patch_shape,
            packet_planes=None
            if mesh_layout is None or mesh_layout.schedule != "slab"
            else mesh_layout.packet_capacity,
            packet_lines=None
            if mesh_layout is None or mesh_layout.schedule != "pencil"
            else mesh_layout.packet_capacity,
            packet_bytes=0 if mesh_layout is None else mesh_layout.packet_bytes,
            patch_bytes=0 if mesh_layout is None else mesh_layout.patch_bytes,
            mode_chunk=None if mode_layout is None else mode_layout.chunk,
            mode_chunks=None if mode_layout is None else mode_layout.chunk_count,
            schedule=None if mesh_layout is None else mesh_layout.schedule,
            spectral_grid=None if mesh_layout is None else mesh_layout.spectral_grid,
            owner_coordinates=transfer.owner_coordinates,
            owner_devices=transfer.owner_devices,
            physical_layout_id=transfer.physical_layout_id,
            modal_layout_id=transfer.modal_layout_id,
            physical_block_shape=transfer.physical_block_shape,
            modal_block_shape=transfer.modal_block_shape,
            collective_order=transfer.collective_order,
            spectral_topology_id=None
            if spectral is None
            else spectral.topology.topology_id,
            spectral_plan_id=None if spectral is None else spectral.plan_id,
            report_identity=IdentityWords.of(
                {
                    "kind": "reciprocal-distribution-report",
                    "plan": plan.plan_id,
                    "ownership": ownership.plan_id,
                    "partition": partition.partition_id,
                    "spectral": None if spectral is None else spectral.plan_id,
                    "schedule": None if mesh_layout is None else mesh_layout.schedule,
                    "spectral_grid": None
                    if mesh_layout is None
                    else list(mesh_layout.spectral_grid),
                    "owner_coordinates": None
                    if transfer.owner_coordinates is None
                    else [list(value) for value in transfer.owner_coordinates],
                    "owner_devices": None
                    if transfer.owner_devices is None
                    else [list(value) for value in transfer.owner_devices],
                    "physical_layout": transfer.physical_layout_id,
                    "modal_layout": transfer.modal_layout_id,
                    "packet_capacity": None
                    if mesh_layout is None
                    else mesh_layout.packet_capacity,
                    "collective_order": list(transfer.collective_order),
                }
            ),
        )
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-reciprocal-distribution",
                "plan": plan.plan_id,
                "ownership": ownership.plan_id,
                "partition": partition.partition_id,
                "report": self.report.report_id,
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical preparation digest."""
        return self.prepared_identity.fingerprint()

    @property
    def kind(self) -> AtomisticReciprocalKind:
        return self.plan.kind

    def global_term(self, axis_name: str, /) -> "OwnerReciprocalTerm":
        """The C1 ``OwnerGlobalTerm`` epilogue: ``(energy share, successful)``."""
        return OwnerReciprocalTerm(self, axis_name)


def prepare_reciprocal_distribution(
    plan: AtomisticReciprocalDistributionPlan,
    ownership: DistributedOwnershipPlan,
    partition: FractionalOwnerPartition,
    /,
) -> PreparedReciprocalDistribution:
    """Host preparation: patch/packet capacities, spectral plan and mode chunks."""
    return PreparedReciprocalDistribution(plan, ownership, partition)


# --- evidence ----------------------------------------------------------------------


@final
class ReciprocalDistributionEvidence(StrictModule):
    """Owner-local shares with globally reduced status (identical on every owner).

    ``capacity_successful`` (patch coverage, packet capacity) is distinct from
    ``scientific_successful`` (cell, finiteness, coverage/support,
    neutrality, anchors, boundary geometry). ``prepared_identity`` is the
    producing preparation's dynamic identity (``prepared_id`` on the host).
    """

    energy_share: Array
    reciprocal_share: Array
    local_share: Array
    system_share: Array
    patch_certified: Array
    patch_covered: Array
    packets_fit: Array
    maximum_patch_extent: Array
    maximum_packet_load: Array
    cell_successful: Array
    sources_finite: Array
    influence_finite: Array
    coverage_successful: Array
    neutral: Array
    total_charge: Array
    capacity_successful: Array
    scientific_successful: Array
    successful: Array
    kind: str = eqx.field(static=True)
    strategy: str = eqx.field(static=True)
    prepared_identity: IdentityWords

    @property
    def prepared_id(self) -> str:
        """Host decoding of the producing preparation digest."""
        return self.prepared_identity.fingerprint()


class OwnerReciprocalActions(NamedTuple):
    """Exact actions of the term energy on this owner's rows.

    ``payload_gradient`` is ``dE/dpayload`` (the conjugate potentials of the
    inexact payload leaves; ``None`` for integer/boolean leaves) and
    ``forces`` is ``-dE/dx`` with the payload held fixed.
    """

    energy_share: Array
    payload_gradient: ReciprocalSitePayload
    forces: Array
    evidence: ReciprocalDistributionEvidence


# --- collective helpers ------------------------------------------------------------


def _owner(axis_name: str, /) -> Array:
    return jax.lax.axis_index(axis_name).astype(jnp.int32)


def _global_all(flag: ArrayLike, axis_name: str, /) -> Array:
    value = jax.lax.stop_gradient(jnp.all(jnp.asarray(flag))).astype(jnp.int32)
    return JaxCollectiveProvider(axis_name).minimum(value).astype(jnp.bool_)


def _global_max(value: ArrayLike, axis_name: str, /) -> Array:
    return JaxCollectiveProvider(axis_name).maximum(
        jax.lax.stop_gradient(jnp.asarray(value))
    )


def _active_finite(tree: object, mask: Array, /) -> Array:
    """Whether every inexact leaf is finite on its active rows (before sanitizing).

    Row leaves (leading axis = owner rows) are tested on active rows only;
    other inexact leaves (cell vectors) must be finite everywhere.
    """
    flag = jnp.asarray(True)
    rows = jnp.asarray(mask, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        value = jnp.asarray(leaf)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            continue
        finite = jnp.isfinite(value)
        if value.ndim >= 1 and value.shape[0] == rows.shape[0]:
            active = rows.reshape(rows.shape + (1,) * (value.ndim - 1))
            finite = jnp.where(active, finite, True)
        flag = flag & jnp.all(finite)
    return jax.lax.stop_gradient(flag)


# --- geometry ----------------------------------------------------------------------


class _Frame(NamedTuple):
    inverse: Array
    shift: Array
    fractional: Array
    wrapped: Array
    finite: Array
    anchor_successful: Array


def _frame(
    prepared: PreparedReciprocalDistribution,
    mesh: PreparedReciprocalMesh,
    positions: Array,
    mask: Array,
    payload: ReciprocalSitePayload,
    inverse: Array,
    axis_name: str,
    /,
) -> _Frame:
    """Mesh fractional coordinates, including the term's anchor-source frame."""
    dtype = positions.dtype
    fractional = contract(
        "ni,ij->nj", positions - mesh.cell.origin.astype(dtype), inverse.astype(dtype)
    )
    finite = jnp.all(jnp.where(mask[:, None], jnp.isfinite(fractional), True))
    fractional = jnp.where(mask[:, None] & jnp.isfinite(fractional), fractional, 0.0)
    shift = jnp.zeros((3,), dtype=dtype)
    anchor_ok = jnp.asarray(True)
    if mesh.anchor_index is not None:
        if payload.anchor is None:
            raise ValueError("An anchor-source mesh frame needs payload.anchor rows.")
        anchor = jnp.asarray(payload.anchor, dtype=jnp.bool_) & mask
        mode = prepared.plan.reduction_mode
        anchored = owner_system_sum(
            contract("n,nd->d", anchor.astype(dtype), fractional), axis_name, mode
        )
        count = owner_system_sum(jnp.sum(anchor, dtype=jnp.int32), axis_name, mode)
        anchor_ok = count == 1
        shift = 0.5 - anchored
    shifted = fractional + shift
    wrapped = shifted - jax.lax.stop_gradient(jnp.floor(shifted))
    return _Frame(inverse, shift, shifted, wrapped, finite, anchor_ok)


class _Patch(NamedTuple):
    origin: Array
    certified: Array
    required: Array


def _patch_origin(
    prepared: PreparedReciprocalDistribution,
    inverse: Array,
    shift: Array,
    owner: Array,
    /,
) -> _Patch:
    """Owner patch origin from its region, the current cell reach and the spline."""
    layout = prepared.mesh_layout
    if layout is None:
        raise RuntimeError("Patch origins exist only for particle meshes.")
    frozen = jax.lax.stop_gradient(inverse)
    dtype = frozen.dtype
    reach = prepared.plan.reuse_reach * jnp.sqrt(jnp.sum(frozen * frozen, axis=0))
    shape = jnp.asarray(layout.grid_shape, dtype=dtype)
    offset = jnp.asarray(layout.coordinate_offsets, dtype=dtype)
    half = 0.5 * (layout.degree - 1)
    shift_ = jax.lax.stop_gradient(shift).astype(dtype)
    lower = layout.region_lower[owner].astype(dtype) - reach + shift_
    upper = layout.region_upper[owner].astype(dtype) + reach + shift_
    first = jnp.floor((lower - offset) * shape - half)
    last = jnp.floor((upper - offset) * shape - half) + layout.degree
    required = (last - first + 1.0).astype(jnp.int32)
    capacity = jnp.asarray(layout.patch_shape, dtype=jnp.int32)
    counts = jnp.asarray(layout.grid_shape, dtype=jnp.int32)
    certified = jnp.all((required <= capacity) | (capacity == counts))
    origin = jnp.mod(first.astype(jnp.int32), counts)
    return _Patch(origin, certified, required)


class _Deposit(NamedTuple):
    patch: Array
    covered: Array


def _deposit(
    prepared: PreparedReciprocalDistribution,
    mesh: PreparedReciprocalMesh,
    wrapped: Array,
    values: Array,
    mask: Array,
    origin: Array,
    full_mesh: bool,
    /,
) -> _Deposit:
    """Spread owned channels with the term's spline routes into the patch."""
    layout = prepared.mesh_layout
    if layout is None:
        raise RuntimeError("Deposits exist only for particle meshes.")
    splat = mesh.splat
    state = splat.plan.assignment.build(
        splat.layout,
        splat.plan.target.structured_axes,
        splat.axis_bounds,
        jnp.where(mask[:, None], wrapped, 0.0),
        mask,
    )
    indices = jax.lax.stop_gradient(state.indices)
    valid = jax.lax.stop_gradient(state.valid) & mask[:, None]
    k0, k1, k2 = layout.grid_shape
    channels = values.shape[1]
    weights = jnp.where(valid, state.weights.astype(values.dtype), 0.0)
    contributions = weights[:, :, None] * values[:, None, :]
    if full_mesh:
        flat = jnp.where(valid, indices, k0 * k1 * k2)
        target = jnp.zeros((k0 * k1 * k2 + 1, channels), dtype=values.dtype)
        target = target.at[flat.reshape((-1,))].add(contributions.reshape((-1, channels)))
        return _Deposit(target[:-1].reshape((k0, k1, k2, channels)), jnp.asarray(True))
    p0, p1, p2 = layout.patch_shape
    axis = (indices // (k1 * k2), (indices // k2) % k1, indices % k2)
    local = tuple(
        jnp.mod(index - origin[position], count)
        for position, (index, count) in enumerate(
            zip(axis, layout.grid_shape, strict=True)
        )
    )
    inside = (local[0] < p0) & (local[1] < p1) & (local[2] < p2)
    routed = valid & inside
    flat = jnp.where(routed, (local[0] * p1 + local[1]) * p2 + local[2], p0 * p1 * p2)
    target = jnp.zeros((p0 * p1 * p2 + 1, channels), dtype=values.dtype)
    target = target.at[flat.reshape((-1,))].add(contributions.reshape((-1, channels)))
    return _Deposit(target[:-1].reshape((p0, p1, p2, channels)), jnp.all(~valid | inside))


class _Route(NamedTuple):
    destination: Array
    rank: Array
    fits: Array
    local_planes: Array
    received: Array
    y_index: Array
    z_index: Array
    maximum_load: Array


def _route_metadata(
    prepared: PreparedReciprocalDistribution, origin: Array, axis_name: str, /
) -> _Route:
    layout = prepared.mesh_layout
    if layout is None:
        raise RuntimeError("Routes exist only for particle meshes.")
    owners = prepared.owner_count
    provider = JaxCollectiveProvider(axis_name)
    k0, k1, k2 = layout.grid_shape
    p0, p1, p2 = layout.patch_shape
    capacity = layout.packet_capacity
    slab = layout.block_shape[0]
    planes = jnp.mod(origin[0] + jnp.arange(p0, dtype=jnp.int32), k0)
    destination = planes // slab
    rank, loads = _bucket_ranks(destination, jnp.ones((p0,), dtype=jnp.bool_), owners)
    fits = rank < capacity
    header = _pack(planes, destination, rank, fits, owners, capacity, -1)
    received = provider.all_to_all(header, split_axis=0, concat_axis=0)
    origins = provider.all_gather(origin[None, :], axis=0, tiled=True)
    owner = _owner(axis_name)
    local = received - owner * slab
    y_index = jnp.mod(origins[:, 1][:, None] + jnp.arange(p1, dtype=jnp.int32), k1)
    z_index = jnp.mod(origins[:, 2][:, None] + jnp.arange(p2, dtype=jnp.int32), k2)
    return _Route(
        destination,
        rank,
        fits,
        jnp.where(received >= 0, local, 0),
        received >= 0,
        y_index,
        z_index,
        jnp.max(loads),
    )


def _route_forward(
    prepared: PreparedReciprocalDistribution,
    route: _Route,
    patch: Array,
    axis_name: str,
    /,
) -> Array:
    """Sum patch planes into this owner's x-slab ``(slab, K1, K2, C)``."""
    layout = prepared.mesh_layout
    if layout is None:
        raise RuntimeError("Routes exist only for particle meshes.")
    owners = prepared.owner_count
    provider = JaxCollectiveProvider(axis_name)
    _, k1, k2 = layout.grid_shape
    packet = _pack(
        patch,
        route.destination,
        route.rank,
        route.fits,
        owners,
        layout.packet_capacity,
        0.0,
    )
    data = provider.all_to_all(packet, split_axis=0, concat_axis=0)
    data = jnp.where(route.received[:, :, None, None, None], data, 0.0)
    slab = jnp.zeros((layout.block_shape[0], k1, k2, patch.shape[-1]), dtype=patch.dtype)
    return slab.at[
        route.local_planes[:, :, None, None],
        route.y_index[:, None, :, None],
        route.z_index[:, None, None, :],
    ].add(data)


def _route_adjoint(
    prepared: PreparedReciprocalDistribution,
    route: _Route,
    slab: Array,
    axis_name: str,
    /,
) -> Array:
    """Exact transpose of :func:`_route_forward`: slab values back to the patch."""
    layout = prepared.mesh_layout
    if layout is None:
        raise RuntimeError("Routes exist only for particle meshes.")
    provider = JaxCollectiveProvider(axis_name)
    gathered = slab[
        route.local_planes[:, :, None, None],
        route.y_index[:, None, :, None],
        route.z_index[:, None, None, :],
    ]
    gathered = jnp.where(route.received[:, :, None, None, None], gathered, 0.0)
    returned = provider.all_to_all(gathered, split_axis=0, concat_axis=0)
    rows = returned[route.destination, jnp.where(route.fits, route.rank, 0)]
    return jnp.where(route.fits[:, None, None, None], rows, 0.0)


class _LineRoute(NamedTuple):
    """Pencil route of patch ``(x, y)``-lines (each carrying all patch z cells).

    Sender side: ``destination``/``rank``/``fits`` per patch line (row-major
    over the patch x and y extents). Receiver side, per ``(source, slot)``:
    the owned-block ``local_x``/``local_y`` of the line, ``received`` and the
    source patch's global ``z_index``.
    """

    destination: Array
    rank: Array
    fits: Array
    local_x: Array
    local_y: Array
    received: Array
    z_index: Array
    maximum_load: Array


def _line_route_metadata(
    prepared: PreparedReciprocalDistribution, origin: Array, axis_name: str, /
) -> _LineRoute:
    """Destinations of patch lines in the canonical pencil physical layout.

    Line ``(x, y)`` belongs to owner ``x // b0 + px * (y // b1)``: the owner
    whose spectral grid coordinate ``(r % px, r // px)`` holds the x-block and
    y-block of the line.
    """
    layout = prepared.mesh_layout
    if layout is None or layout.schedule != "pencil":
        raise RuntimeError("Line routes exist only for pencil particle meshes.")
    owners = prepared.owner_count
    provider = JaxCollectiveProvider(axis_name)
    k0, k1, k2 = layout.grid_shape
    p0, p1, p2 = layout.patch_shape
    b0, b1, _ = layout.block_shape
    px = layout.spectral_grid[0]
    capacity = layout.packet_capacity
    x = jnp.mod(origin[0] + jnp.arange(p0, dtype=jnp.int32), k0)
    y = jnp.mod(origin[1] + jnp.arange(p1, dtype=jnp.int32), k1)
    lines = jnp.stack(jnp.broadcast_arrays(x[:, None], y[None, :]), axis=-1).reshape(
        (p0 * p1, 2)
    )
    destination = lines[:, 0] // b0 + px * (lines[:, 1] // b1)
    rank, loads = _bucket_ranks(
        destination, jnp.ones((p0 * p1,), dtype=jnp.bool_), owners
    )
    fits = rank < capacity
    header = _pack(lines, destination, rank, fits, owners, capacity, -1)
    received = provider.all_to_all(header, split_axis=0, concat_axis=0)
    origins = provider.all_gather(origin[None, :], axis=0, tiled=True)
    owner = _owner(axis_name)
    valid = received[..., 0] >= 0
    local_x = jnp.where(valid, received[..., 0] - (owner % px) * b0, 0)
    local_y = jnp.where(valid, received[..., 1] - (owner // px) * b1, 0)
    z_index = jnp.mod(origins[:, 2][:, None] + jnp.arange(p2, dtype=jnp.int32), k2)
    return _LineRoute(
        destination, rank, fits, local_x, local_y, valid, z_index, jnp.max(loads)
    )


def _line_route_forward(
    prepared: PreparedReciprocalDistribution,
    route: _LineRoute,
    patch: Array,
    axis_name: str,
    /,
) -> Array:
    """Sum patch lines into this owner's pencil block ``(K0/px, K1/py, K2, C)``."""
    layout = prepared.mesh_layout
    if layout is None:
        raise RuntimeError("Routes exist only for particle meshes.")
    owners = prepared.owner_count
    provider = JaxCollectiveProvider(axis_name)
    p0, p1, p2 = layout.patch_shape
    b0, b1, k2 = layout.block_shape
    channels = patch.shape[-1]
    packet = _pack(
        patch.reshape((p0 * p1, p2, channels)),
        route.destination,
        route.rank,
        route.fits,
        owners,
        layout.packet_capacity,
        0.0,
    )
    data = provider.all_to_all(packet, split_axis=0, concat_axis=0)
    data = jnp.where(route.received[:, :, None, None], data, 0.0)
    block = jnp.zeros((b0, b1, k2, channels), dtype=patch.dtype)
    return block.at[
        route.local_x[:, :, None],
        route.local_y[:, :, None],
        route.z_index[:, None, :],
    ].add(data)


def _line_route_adjoint(
    prepared: PreparedReciprocalDistribution,
    route: _LineRoute,
    block: Array,
    axis_name: str,
    /,
) -> Array:
    """Exact transpose of :func:`_line_route_forward`: block values to the patch."""
    layout = prepared.mesh_layout
    if layout is None:
        raise RuntimeError("Routes exist only for particle meshes.")
    provider = JaxCollectiveProvider(axis_name)
    p0, p1, p2 = layout.patch_shape
    gathered = block[
        route.local_x[:, :, None],
        route.local_y[:, :, None],
        route.z_index[:, None, :],
    ]
    gathered = jnp.where(route.received[:, :, None, None], gathered, 0.0)
    returned = provider.all_to_all(gathered, split_axis=0, concat_axis=0)
    rows = returned[route.destination, jnp.where(route.fits, route.rank, 0)]
    rows = jnp.where(route.fits[:, None, None], rows, 0.0)
    return rows.reshape((p0, p1, p2, block.shape[-1]))


class _Transfer(NamedTuple):
    """One owner's forward particle-to-modal transfer and its action route."""

    block: Array
    route: _Route | _LineRoute
    load: Array


def _transfer_forward(
    prepared: PreparedReciprocalDistribution, origin: Array, patch: Array, axis: str, /
) -> _Transfer:
    """Patch -> canonical physical layout -> native distributed forward FFT."""
    layout = prepared.mesh_layout
    spectral = prepared.spectral
    if layout is None or spectral is None:
        raise RuntimeError("The distributed-FFT strategy has no spectral plan.")
    if layout.schedule == "pencil":
        lines = _line_route_metadata(prepared, origin, axis)
        physical = _line_route_forward(prepared, lines, patch, axis)
        return _Transfer(spectral.forward_local(physical), lines, lines.maximum_load)
    planes = _route_metadata(prepared, origin, axis)
    physical = _route_forward(prepared, planes, patch, axis)
    return _Transfer(spectral.forward_local(physical), planes, planes.maximum_load)


def _transfer_adjoint(
    prepared: PreparedReciprocalDistribution,
    route: _Route | _LineRoute,
    modal: Array,
    dtype: jnp.dtype,
    axis: str,
    /,
) -> Array:
    """Native inverse FFT of a modal block, then the exact reverse patch route."""
    layout = prepared.mesh_layout
    spectral = prepared.spectral
    if layout is None or spectral is None:
        raise RuntimeError("The distributed-FFT strategy has no spectral plan.")
    count = float(np.prod(layout.grid_shape))
    physical = (count * jnp.real(spectral.inverse_local(modal))).astype(dtype)
    if isinstance(route, _LineRoute):
        return _line_route_adjoint(prepared, route, physical, axis)
    return _route_adjoint(prepared, route, physical, axis)


class _Modal(NamedTuple):
    geometry: ReciprocalGeometry
    waves: tuple[Array, ...]
    weight: Array
    zero: Array


def _modal_context(
    prepared: PreparedReciprocalDistribution,
    mesh: PreparedReciprocalMesh,
    vectors: Array,
    owner: Array,
    /,
) -> _Modal:
    """Current-cell wavevectors, Nyquist variants and modulus of this modal block."""
    layout = prepared.mesh_layout
    if layout is None:
        raise RuntimeError("Modal blocks exist only for particle meshes.")
    rows = layout.modal_rows
    start = owner * rows

    def owned(values: Array, axis: int) -> Array:
        if axis != layout.modal_axis:
            return values
        return jax.lax.dynamic_slice(values, (start,), (rows,))

    m0, m1, m2 = (owned(layout.axis_modes[axis], axis) for axis in range(3))
    s0, s1, s2 = (owned(layout.axis_symbols[axis], axis) for axis in range(3))
    n0, n1, n2 = (owned(layout.axis_nyquist[axis], axis) for axis in range(3))
    modes = jnp.stack(jnp.meshgrid(m0, m1, m2, indexing="ij"), axis=-1)
    geometry = _lattice_geometry(mesh.cell, vectors, modes, mesh.maximum_mode)
    dtype = vectors.dtype
    window = s0[:, None, None] * s1[None, :, None] * s2[None, None, :]
    flags = jnp.stack(
        jnp.broadcast_arrays(n0[:, None, None], n1[None, :, None], n2[None, None, :]),
        axis=-1,
    )
    counted = (
        ~jnp.any(flags, axis=-1)
        if layout.exclude_nyquist
        else jnp.ones(window.shape, dtype=jnp.bool_)
    )
    weight = jnp.where(counted, 1.0 / jnp.where(counted, window, 1.0) ** 2, 0.0).astype(
        dtype
    )
    waves = []
    for variant in layout.variants:
        if not any(variant):
            waves.append(geometry.wavevectors.astype(dtype))
            continue
        sign = jnp.where(flags & jnp.asarray(variant), -1.0, 1.0).astype(dtype)
        waves.append(
            contract(
                "...i,ij->...j",
                modes.astype(dtype) * sign,
                geometry.reciprocal_vectors.astype(dtype),
            )
        )
    zero = jnp.all(modes == 0.0, axis=-1)
    return _Modal(geometry, tuple(waves), weight, zero)


# --- kinds: channels and modal energies -------------------------------------------


def _multipole_channels(
    plan: AtomisticReciprocalDistributionPlan,
    moments: CartesianMultipoles,
    mask: Array,
    rows: Array | None,
    degree: int,
    /,
) -> Array:
    """``_channel_values`` layout; Gaussian classes from the host class table."""
    base = _channel_values(MultipoleSiteSet(moments, active=mask), degree, 1)
    if plan.class_widths is None:
        return base
    if rows is None:
        raise ValueError("Gaussian multipole payloads need their width-class rows.")
    classes = jax.nn.one_hot(rows, plan.class_widths.shape[0], dtype=base.dtype)
    return (classes[:, :, None] * base[:, None, :]).reshape(
        (base.shape[0], classes.shape[1] * base.shape[1])
    )


def _scalar_values(
    payload: ReciprocalSitePayload, mask: Array, dtype: jnp.dtype, /
) -> Array:
    if payload.values is None:
        raise ValueError("This reciprocal kind needs payload.values.")
    values = jnp.asarray(payload.values).astype(dtype)
    if values.ndim == 1:
        values = values[:, None]
    return jnp.where(mask[:, None] & jnp.isfinite(values), values, 0.0)


def _mesh_channels(
    prepared: PreparedReciprocalDistribution,
    payload: ReciprocalSitePayload,
    mask: Array,
    dtype: jnp.dtype,
    /,
) -> Array:
    plan = prepared.plan
    definition = plan.definition
    match definition:
        case PreparedParticleMeshEwaldPotential():
            return _scalar_values(payload, mask, dtype)
        case PreparedGeneralForceFieldTerm():
            if payload.rows is None:
                raise ValueError("LJ-PME payloads need site type rows.")
            factors = definition.plan.arrays[2]
            values = factors[jnp.asarray(payload.rows, dtype=jnp.int32)].astype(dtype)
            return values * mask[:, None]
        case PreparedFourierD3Potential():
            if payload.rows is None:
                raise ValueError("FourierD3 payloads need species channel rows.")
            species = definition.resources.species
            coefficients = _scalar_values(payload, mask, dtype)
            onehot = (
                (jnp.asarray(payload.rows)[:, None] == jnp.arange(species)[None, :])
                & mask[:, None]
            ).astype(dtype)
            return contract("ns,nl->nsl", onehot, coefficients).reshape(
                (coefficients.shape[0], -1)
            )
        case PreparedMultipoleElectrostatics():
            if payload.moments is None:
                raise ValueError("Multipole payloads need owned moments.")
            return _multipole_channels(
                plan,
                payload.moments,
                mask,
                payload.rows,
                _channel_degree(plan, definition),
            ).astype(dtype)
    raise RuntimeError(f"{plan.kind} has no mesh channels.")


def _kernel_sum(
    kernel: Array, weight: Array, power: Array, dtype: jnp.dtype, /
) -> tuple[Array, Array]:
    masked, finite = _masked_kernel(kernel, weight > 0.0, dtype)
    return jnp.sum(weight * masked * power), finite


def _mesh_energy(
    prepared: PreparedReciprocalDistribution,
    real: Array,
    imaginary: Array,
    modal: _Modal,
    /,
    *,
    right: tuple[Array, Array] | None = None,
) -> tuple[Array, Array]:
    """Kernel sum over this modal block, averaged over Nyquist sign variants.

    ``right`` gives the second density of the multipole bilinear form.
    """
    plan = prepared.plan
    definition = plan.definition
    dtype = real.dtype
    geometry = modal.geometry
    zero_excluded = jnp.where(modal.zero, 0.0, modal.weight)
    total = jnp.zeros((), dtype=dtype)
    finite = jnp.asarray(True)
    for wave in modal.waves:
        squared = jnp.sum(wave * wave, axis=-1)
        match definition:
            case PreparedParticleMeshEwaldPotential():
                settings = definition.plan.settings
                influence = _coulomb_influence(
                    geometry,
                    settings.alpha,
                    definition.system.plan.units.coulomb_constant,
                )
                power = jnp.sum(real * real + imaginary * imaginary, axis=-1)
                value, ok = _kernel_sum(
                    influence(wave, squared), zero_excluded, power, dtype
                )
            case PreparedGeneralForceFieldTerm():
                alpha = definition.plan.arrays[1].reshape(()).astype(dtype)
                influence = lennard_jones_pme_influence(alpha, geometry.volume)
                power = jnp.sum(real * real + imaginary * imaginary, axis=-1)
                value, ok = _kernel_sum(
                    influence(wave, squared), modal.weight, power, dtype
                )
            case PreparedFourierD3Potential():
                value, ok = _fourier_d3_modal(
                    definition, real, imaginary, wave, squared, modal.weight, geometry
                )
            case PreparedMultipoleElectrostatics():
                value, ok = _multipole_modal(
                    plan,
                    definition,
                    real,
                    imaginary,
                    right,
                    wave,
                    squared,
                    zero_excluded,
                    geometry,
                )
            case _:
                raise RuntimeError(f"{plan.kind} has no particle mesh.")
        total = total + value
        finite = finite & ok
    return total / len(modal.waves), finite


def _fourier_d3_modal(
    definition: PreparedFourierD3Potential,
    real: Array,
    imaginary: Array,
    wave: Array,
    squared: Array,
    weight: Array,
    geometry: ReciprocalGeometry,
    /,
) -> tuple[Array, Array]:
    dtype = real.dtype
    resources = definition.resources
    species, rank = resources.species, resources.padded_rank
    shape = real.shape[:-1] + (species, rank)
    dr, di = real.reshape(shape), imaginary.reshape(shape)
    s6, s8, a1, a2, r4r2 = definition._damping(dtype)
    product = r4r2[definition.pair_left] * r4r2[definition.pair_right]
    radius = a1 * jnp.sqrt(3.0 * product) + a2
    volume = geometry.volume.astype(dtype) * definition.length_scale**3
    prefactor = -0.5 / volume
    eigenvalues = definition._eigenvalues(dtype)

    def pair(
        carry: tuple[Array, Array], inputs: tuple[Array, ...]
    ) -> tuple[tuple[Array, Array], None]:
        left, right, multiplicity, pair_radius, pair_product = inputs
        influence = FourierD3PairInfluence(
            radius=pair_radius,
            s6=s6,
            s8=s8,
            product=pair_product,
            table_per_system_length=definition.length_scale,
        )
        kernel, ok = _masked_kernel(influence(wave, squared), weight > 0.0, dtype)
        cross = jnp.take(dr, left, axis=-2) * jnp.take(dr, right, axis=-2) + jnp.take(
            di, left, axis=-2
        ) * jnp.take(di, right, axis=-2)
        value = (
            multiplicity
            * prefactor
            * jnp.sum(cross * (weight * kernel)[..., None] * eigenvalues)
        )
        return (carry[0] + value, carry[1] & ok), None

    (total, finite), _ = jax.lax.scan(
        jax.checkpoint(pair),
        (jnp.zeros((), dtype=dtype), jnp.asarray(True)),
        (
            definition.pair_left,
            definition.pair_right,
            definition.pair_multiplicity.astype(dtype),
            radius,
            product,
        ),
    )
    return total, finite


def _multipole_modal(
    plan: AtomisticReciprocalDistributionPlan,
    definition: PreparedMultipoleElectrostatics,
    real: Array,
    imaginary: Array,
    right: tuple[Array, Array] | None,
    wave: Array,
    squared: Array,
    weight: Array,
    geometry: ReciprocalGeometry,
    /,
) -> tuple[Array, Array]:
    dtype = real.dtype
    degree = _channel_degree(plan, definition)
    alpha = _periodic_route_settings(definition).alpha
    constant = definition.definition.coulomb_constant
    widths = plan.class_widths

    def combined(dr: Array, di: Array) -> Array:
        density = jnp.moveaxis(dr + 1j * di, -1, 0)
        return _combined_density(density, wave, widths, degree)[0]

    left = combined(real, imaginary)
    other = left if right is None else combined(*right)
    kernel = _ewald_kernel(squared, geometry.volume.astype(dtype), alpha, constant)
    product = jnp.real(left * jnp.conj(other))
    return _kernel_sum(kernel, weight, product, dtype)


# --- explicit modes ----------------------------------------------------------------


def _mode_structure(
    prepared: PreparedReciprocalDistribution,
    positions: Array,
    payload: ReciprocalSitePayload,
    mask: Array,
    wave: Array,
    /,
    *,
    moments: CartesianMultipoles | None = None,
) -> tuple[Array, Array]:
    """Owner-partial ``S(k) = sum_i v_i exp(i k . r_i)`` of one mode chunk."""
    plan = prepared.plan
    definition = plan.definition
    dtype = positions.dtype
    phase = contract("nd,md->nm", positions, wave)
    if isinstance(definition, PreparedMultipoleElectrostatics):
        site = payload.moments if moments is None else moments
        if site is None:
            raise ValueError("Multipole payloads need owned moments.")
        masked = site.masked(mask)
        degree = max(_channel_degree(plan, definition), site.maximum_degree)
        coefficient_real, coefficient_imaginary = _mode_coefficients(masked, wave, degree)
        if plan.class_widths is None:
            width = jnp.zeros((positions.shape[0],), dtype=dtype)
        else:
            if payload.rows is None:
                raise ValueError("Gaussian multipole payloads need width-class rows.")
            width = plan.class_widths.astype(dtype)[payload.rows] ** 2
        envelope = jnp.exp(-0.5 * width[:, None] * jnp.sum(wave * wave, axis=-1)[None, :])
        cosine, sine = jnp.cos(phase), jnp.sin(phase)
        real = jnp.sum(
            envelope * (coefficient_real * cosine - coefficient_imaginary * sine), axis=0
        )
        imaginary = jnp.sum(
            envelope * (coefficient_real * sine + coefficient_imaginary * cosine), axis=0
        )
        return real[:, None], imaginary[:, None]
    values = _scalar_values(payload, mask, dtype)
    return (
        contract("nc,nm->mc", values, jnp.cos(phase)),
        contract("nc,nm->mc", values, jnp.sin(phase)),
    )


def _mode_energy(
    prepared: PreparedReciprocalDistribution,
    real: Array,
    imaginary: Array,
    wave: Array,
    weight: Array,
    geometry: ReciprocalGeometry,
    oriented: Array,
    /,
    *,
    right: tuple[Array, Array] | None = None,
) -> tuple[Array, Array]:
    plan = prepared.plan
    definition = plan.definition
    dtype = real.dtype
    squared = jnp.sum(wave * wave, axis=-1)
    match definition:
        case PreparedEwaldReferencePotential():
            settings = definition.plan.settings
            influence = _coulomb_influence(
                geometry, settings.alpha, definition.system.plan.units.coulomb_constant
            )
            power = jnp.sum(real * real + imaginary * imaginary, axis=-1)
            return _kernel_sum(influence(wave, squared), weight, power, dtype)
        case PreparedLatentEwald():
            influence = latent_ewald_influence(
                definition.definition, geometry.volume, oriented
            )
            power = jnp.sum(real * real + imaginary * imaginary, axis=-1)
            return _kernel_sum(influence(wave, squared), weight, power, dtype)
        case PreparedMultipoleElectrostatics():
            settings = _periodic_route_settings(definition)
            kernel = _ewald_kernel(
                squared,
                geometry.volume.astype(dtype),
                settings.alpha,
                definition.definition.coulomb_constant,
            )
            other_real, other_imaginary = (real, imaginary) if right is None else right
            product = (
                real[:, 0] * other_real[:, 0] + imaginary[:, 0] * other_imaginary[:, 0]
            )
            return _kernel_sum(kernel, weight, product, dtype)
    raise RuntimeError(f"{plan.kind} has no explicit modes.")


# --- reciprocal core ---------------------------------------------------------------


class _ReciprocalResult(NamedTuple):
    value: Array
    position_gradient: Array | None
    payload_gradient: ReciprocalSitePayload | None
    patch_certified: Array
    patch_covered: Array
    packets_fit: Array
    maximum_extent: Array
    maximum_load: Array
    cell_successful: Array
    sources_finite: Array
    influence_finite: Array
    anchor_successful: Array
    support_radius: Array
    volume: Array


def _differentiable(
    payload: ReciprocalSitePayload, /
) -> tuple[ReciprocalSitePayload, ReciprocalSitePayload]:
    return eqx.partition(payload, eqx.is_inexact_array)


def _mesh_reciprocal(
    prepared: PreparedReciprocalDistribution,
    positions: Array,
    payload: ReciprocalSitePayload,
    mask: Array,
    vectors: Array,
    axis_name: str,
    /,
    *,
    actions: bool,
    channels: ChannelFunction | None = None,
    split: int | None = None,
) -> _ReciprocalResult:
    """Steps 1-6 of the particle-mesh route for this owner."""
    plan = prepared.plan
    mesh = _definition_mesh(plan.definition)
    layout = prepared.mesh_layout
    if mesh is None or layout is None:
        raise RuntimeError("No particle mesh is prepared.")
    owner = _owner(axis_name)
    dtype = positions.dtype
    modal = _modal_context(prepared, mesh, vectors, owner)
    inverse = modal.geometry.inverse.astype(dtype)
    full = plan.strategy == "replicated-mesh"
    channel_function = (
        (lambda value: _mesh_channels(prepared, value, mask, dtype))
        if channels is None
        else channels
    )
    frame0 = _frame(prepared, mesh, positions, mask, payload, inverse, axis_name)
    patch_meta = _patch_origin(prepared, inverse, frame0.shift, owner)
    origin = jax.lax.stop_gradient(patch_meta.origin)
    dynamic, static = _differentiable(payload)

    def deposit(position: Array, values: ReciprocalSitePayload) -> tuple[Array, Array]:
        bound = eqx.combine(values, static)
        frame = _frame(prepared, mesh, position, mask, bound, inverse, axis_name)
        result = _deposit(
            prepared, mesh, frame.wrapped, channel_function(bound), mask, origin, full
        )
        return result.patch, result.covered

    patch, covered = deposit(positions, dynamic)
    shape = layout.grid_shape
    count = float(np.prod(shape))
    rows = layout.modal_rows
    route: _Route | _LineRoute | None = None
    if full:
        summed = owner_system_sum(patch, axis_name, plan.reduction_mode)
        spectrum = jnp.fft.fftn(summed.astype(jnp.complex128), axes=(0, 1, 2))
        block = jax.lax.dynamic_slice_in_dim(
            spectrum, owner * rows, rows, axis=layout.modal_axis
        )
        load = jnp.zeros((), dtype=jnp.int32)
        fits = jnp.asarray(True)
    else:
        transfer = _transfer_forward(prepared, origin, patch, axis_name)
        block, route, load = transfer.block, transfer.route, transfer.load
        fits = load <= layout.packet_capacity
    real, imaginary = jnp.real(block).astype(dtype), jnp.imag(block).astype(dtype)

    def energy(dr: Array, di: Array) -> tuple[Array, Array]:
        if split is None:
            return _mesh_energy(prepared, dr, di, modal)
        return _mesh_energy(
            prepared,
            dr[..., :split],
            di[..., :split],
            modal,
            right=(dr[..., split:], di[..., split:]),
        )

    position_gradient = payload_gradient = None
    if actions:
        value, influence_ok = energy(real, imaginary)
        # Actions stay differentiable in positions, payload and cell: their
        # Jacobians are the energy's second derivatives (Hessian/mixed actions).
        gradient_real, gradient_imaginary = jax.grad(
            lambda dr, di: energy(dr, di)[0], argnums=(0, 1)
        )(real, imaginary)
        conjugate = (gradient_real + 1j * gradient_imaginary).astype(block.dtype)
        if full:
            embedded = jnp.zeros(spectrum.shape, dtype=spectrum.dtype)
            embedded = jax.lax.dynamic_update_slice_in_dim(
                embedded, conjugate, owner * rows, axis=layout.modal_axis
            )
            complete = owner_system_sum(embedded, axis_name, plan.reduction_mode)
            potential = count * jnp.real(jnp.fft.ifftn(complete, axes=(0, 1, 2)))
        else:
            if route is None:
                raise RuntimeError("The distributed-FFT strategy has no patch route.")
            potential = _transfer_adjoint(prepared, route, conjugate, dtype, axis_name)
        _, local_vjp = jax.vjp(lambda x, v: deposit(x, v)[0], positions, dynamic)
        position_gradient, payload_dynamic = local_vjp(potential.astype(patch.dtype))
        payload_gradient = payload_dynamic
    else:
        value, influence_ok = energy(real, imaginary)
    extent = _global_max(patch_meta.required, axis_name)
    return _ReciprocalResult(
        value=value,
        position_gradient=position_gradient,
        payload_gradient=payload_gradient,
        patch_certified=_global_all(patch_meta.certified, axis_name),
        patch_covered=_global_all(covered, axis_name),
        packets_fit=_global_all(fits, axis_name),
        maximum_extent=extent,
        maximum_load=_global_max(load, axis_name),
        cell_successful=_global_all(modal.geometry.successful, axis_name),
        sources_finite=_global_all(frame0.finite, axis_name),
        influence_finite=_global_all(influence_ok, axis_name),
        anchor_successful=_global_all(frame0.anchor_successful, axis_name),
        support_radius=modal.geometry.support_radius,
        volume=modal.geometry.volume,
    )


def _mode_reciprocal(
    prepared: PreparedReciprocalDistribution,
    positions: Array,
    payload: ReciprocalSitePayload,
    mask: Array,
    vectors: Array,
    axis_name: str,
    /,
    *,
    actions: bool,
    right: CartesianMultipoles | None = None,
) -> _ReciprocalResult:
    """Explicit-mode route: chunk partials reduced onto their mode owner."""
    plan = prepared.plan
    modes = _definition_modes(plan.definition)
    layout = prepared.mode_layout
    if modes is None or layout is None:
        raise RuntimeError("No explicit mode set is prepared.")
    owner = _owner(axis_name)
    owners = prepared.owner_count
    provider = JaxCollectiveProvider(axis_name)
    dtype = positions.dtype
    flat_modes = layout.modes.reshape((-1, 3))
    geometry = _lattice_geometry(modes.cell, vectors, flat_modes, modes.maximum_mode)
    waves = geometry.wavevectors.astype(dtype).reshape(
        (layout.chunk_count, layout.chunk, 3)
    )
    # Every explicit-mode term (Coulomb, multipole and latent Ewald) excludes k = 0.
    weights = jnp.where(layout.zero, 0.0, layout.weights).astype(dtype)
    oriented = _orientation(vectors) > 0.0
    finite_positions = jnp.all(jnp.where(mask[:, None], jnp.isfinite(positions), True))
    position = jnp.where(mask[:, None] & jnp.isfinite(positions), positions, 0.0)
    dynamic, static = _differentiable(payload)
    sub = layout.sub_chunk

    def partial(x: Array, values: ReciprocalSitePayload, wave: Array) -> Array:
        bound = eqx.combine(values, static)
        left = _mode_structure(prepared, x, bound, mask, wave)
        parts = [left[0], left[1]]
        if right is not None:
            other = _mode_structure(prepared, x, bound, mask, wave, moments=right)
            parts.extend(other)
        return jnp.stack(parts, axis=-1)

    def reduce(stacked: Array) -> Array:
        shaped = stacked.reshape((owners, sub) + stacked.shape[1:])
        return jnp.sum(provider.all_to_all(shaped, split_axis=0, concat_axis=0), axis=0)

    def energy(total: Array, wave: Array, weight: Array) -> tuple[Array, Array]:
        if right is None:
            return _mode_energy(
                prepared, total[..., 0], total[..., 1], wave, weight, geometry, oriented
            )
        return _mode_energy(
            prepared,
            total[..., 0],
            total[..., 1],
            wave,
            weight,
            geometry,
            oriented,
            right=(total[..., 2], total[..., 3]),
        )

    def mine(values: Array) -> Array:
        return jax.lax.dynamic_slice_in_dim(values, owner * sub, sub, axis=0)

    zero_payload = jax.tree.map(jnp.zeros_like, dynamic)

    def body(
        carry: tuple[Array, Array, Array, ReciprocalSitePayload],
        inputs: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array, Array, ReciprocalSitePayload], None]:
        total, finite, gradient, payload_total = carry
        wave, weight = inputs
        stacked = partial(position, dynamic, wave)
        reduced = reduce(stacked)
        value, ok = energy(reduced, mine(wave), mine(weight))
        if actions:
            cotangent = jax.grad(lambda s: energy(s, mine(wave), mine(weight))[0])(
                reduced
            )
            complete = provider.all_gather(cotangent, axis=0, tiled=True)
            _, local_vjp = jax.vjp(lambda x, v: partial(x, v, wave), position, dynamic)
            position_part, payload_part = local_vjp(complete)
            gradient = gradient + position_part
            payload_total = jax.tree.map(jnp.add, payload_total, payload_part)
        return (total + value, finite & ok, gradient, payload_total), None

    (value, influence_ok, position_gradient, payload_gradient), _ = jax.lax.scan(
        jax.checkpoint(body),
        (
            jnp.zeros((), dtype=dtype),
            jnp.asarray(True),
            jnp.zeros_like(position),
            zero_payload,
        ),
        (waves, weights.reshape((layout.chunk_count, layout.chunk))),
    )
    true = jnp.asarray(True)
    return _ReciprocalResult(
        value=value,
        position_gradient=position_gradient if actions else None,
        payload_gradient=payload_gradient if actions else None,
        patch_certified=true,
        patch_covered=true,
        packets_fit=true,
        maximum_extent=jnp.zeros((3,), dtype=jnp.int32),
        maximum_load=jnp.zeros((), dtype=jnp.int32),
        cell_successful=_global_all(geometry.successful, axis_name),
        sources_finite=_global_all(finite_positions, axis_name),
        influence_finite=_global_all(influence_ok, axis_name),
        anchor_successful=true,
        support_radius=geometry.support_radius,
        volume=geometry.volume,
    )


# --- kind assembly: owned and system terms -------------------------------------------


class _Inputs(NamedTuple):
    positions: Array
    mask: Array
    vectors: Array


def _inputs(
    prepared: PreparedReciprocalDistribution,
    positions: ArrayLike,
    atom_mask: ArrayLike,
    cell_vectors: ArrayLike,
    /,
) -> _Inputs:
    position = jnp.asarray(positions)
    if position.shape != (prepared.local_capacity, 3):
        raise ValueError(
            f"Owner positions must have shape ({prepared.local_capacity}, 3)."
        )
    if not jnp.issubdtype(position.dtype, jnp.floating):
        raise TypeError("Owner positions must be floating point.")
    mask = jnp.asarray(atom_mask)
    if mask.shape != (prepared.local_capacity,) or mask.dtype != jnp.bool_:
        raise ValueError("atom_mask must be a boolean owner-row vector.")
    vectors = jnp.asarray(cell_vectors, dtype=position.dtype)
    if vectors.shape != (3, 3):
        raise ValueError("cell_vectors must be the replicated (3, 3) row cell.")
    safe = jnp.where(mask[:, None] & jnp.isfinite(position), position, 0.0)
    return _Inputs(safe, mask, vectors)


class _Terms(NamedTuple):
    local: Array
    system: Array
    neutral: Array
    total_charge: Array
    geometry_successful: Array


def _cell_volume(
    prepared: PreparedReciprocalDistribution, vectors: Array, /
) -> tuple[Array, Array]:
    definition = prepared.plan.definition
    mesh = _definition_mesh(definition)
    if mesh is not None:
        return mesh.cell.measure_with_vectors(vectors)
    modes = _definition_modes(definition)
    if modes is None:
        raise RuntimeError("Reciprocal definition without mesh or modes.")
    if modes.cell is None:
        return lattice_measure(vectors)
    return modes.cell.measure_with_vectors(vectors)


def _coulomb_terms(
    prepared: PreparedReciprocalDistribution,
    definition: PreparedParticleMeshEwaldPotential | PreparedEwaldReferencePotential,
    inputs: _Inputs,
    payload: ReciprocalSitePayload,
    axis_name: str,
    /,
) -> _Terms:
    """Ewald self (owned), background and boundary (system, owner 0)."""
    settings = definition.plan.settings
    constant = definition.system.plan.units.coulomb_constant
    mode = prepared.plan.reduction_mode
    dtype = inputs.positions.dtype
    charges = _scalar_values(payload, inputs.mask, dtype)[:, 0]
    owner = _owner(axis_name)
    volume, measured = _cell_volume(prepared, inputs.vectors)
    local = ewald_self_energy(settings.alpha, constant, charges)
    total_charge = owner_system_sum(jnp.sum(charges), axis_name, mode)
    neutral = jnp.abs(total_charge) <= settings.charge_tolerance
    system = jnp.zeros((), dtype=dtype)
    geometry = measured
    if settings.neutrality == "uniform-background":
        system = system + ewald_background_energy(
            settings.alpha, constant, total_charge, volume
        )
    if settings.boundary == "vacuum-sphere":
        if payload.image_counts is None:
            raise ValueError(
                "The vacuum-sphere boundary needs payload.image_counts (unwrapped "
                "coordinates are stored + image_counts @ H)."
            )
        unwrapped = inputs.positions + contract(
            "na,ad->nd", jnp.asarray(payload.image_counts).astype(dtype), inputs.vectors
        )
        dipole = owner_system_sum(
            contract("n,nd->d", charges, unwrapped), axis_name, mode
        )
        system = system + ewald_vacuum_boundary_energy(constant, dipole, volume)
    elif settings.boundary == "slab-yeh-berkowitz":
        cell = definition.cell
        inverse, solved = cell.inverse_for_vectors_with_status(inputs.vectors)
        height, normal_axis = ewald_slab_height(cell, inputs.vectors, volume)
        fractional = contract(
            "ni,i->n",
            inputs.positions - cell.origin.astype(dtype),
            inverse[:, normal_axis].astype(dtype),
        )
        inside = jnp.all(~inputs.mask | ((fractional >= 0.0) & (fractional < 1.0)))
        coordinate = fractional * height
        provider = JaxCollectiveProvider(axis_name)
        frozen = jax.lax.stop_gradient(coordinate)
        top = provider.maximum(jnp.max(jnp.where(inputs.mask, frozen, -jnp.inf)))
        bottom = provider.minimum(jnp.min(jnp.where(inputs.mask, frozen, jnp.inf)))
        gap = (top - bottom) + settings.real_cutoff < jax.lax.stop_gradient(height)
        moment = owner_system_sum(jnp.sum(charges * coordinate), axis_name, mode)
        spread = owner_system_sum(jnp.sum(charges * coordinate**2), axis_name, mode)
        system = system + ewald_slab_boundary_energy(
            constant, volume, height, total_charge, moment, spread
        )
        geometry = geometry & jnp.all(solved) & _global_all(inside, axis_name) & gap
    neutral_ok = neutral if settings.neutrality == "require-neutral" else True
    return _Terms(
        local,
        jnp.where(owner == 0, system, 0.0),
        jnp.asarray(neutral_ok),
        total_charge,
        _global_all(geometry, axis_name),
    )


def _multipole_parts(
    prepared: PreparedReciprocalDistribution,
    electrostatics: PreparedMultipoleElectrostatics,
    inputs: _Inputs,
    left: CartesianMultipoles,
    right: CartesianMultipoles,
    rows: Array | None,
    left_mask: Array,
    right_mask: Array,
    axis_name: str,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Owned intrasite and owner-0 background of ``B(L, R)``; neutrality status."""
    plan = prepared.plan
    definition = electrostatics.definition
    alpha = _periodic_route_settings(electrostatics).alpha
    constant = definition.coulomb_constant
    dtype = inputs.positions.dtype
    mode = plan.reduction_mode
    left_moments, right_moments = left.masked(left_mask), right.masked(right_mask)
    if plan.class_widths is None:
        width = jnp.zeros((inputs.positions.shape[0],), dtype=dtype)
    else:
        if rows is None:
            raise ValueError("Gaussian multipole sites need width-class rows.")
        width = plan.class_widths.astype(dtype)[rows] ** 2
    spread = 2.0 * width
    smooth = jax.lax.rsqrt(2.0 * spread + 1.0 / alpha**2)
    intrasite = -jnp.sum(
        co_centered_interaction(
            erf_kernel_origin_coefficients(smooth, 2), left_moments, right_moments
        )
    )
    if definition.intrasite == "include":
        intrasite = intrasite + jnp.sum(
            co_centered_interaction(
                erf_kernel_origin_coefficients(jax.lax.rsqrt(2.0 * spread), 2),
                left_moments,
                right_moments,
            )
        )
    left_charge = owner_system_sum(jnp.sum(left_moments.charges), axis_name, mode)
    right_charge = owner_system_sum(jnp.sum(right_moments.charges), axis_name, mode)
    volume, _ = _cell_volume(prepared, inputs.vectors)
    background = (
        -jnp.pi * left_charge * right_charge / (alpha**2 * volume)
        if definition.neutrality == "uniform-background"
        else jnp.zeros((), dtype=dtype)
    )
    owner = _owner(axis_name)
    neutral = (jnp.abs(left_charge) <= definition.charge_tolerance) & (
        jnp.abs(right_charge) <= definition.charge_tolerance
    )
    neutral_ok = (
        neutral if definition.neutrality == "require-neutral" else jnp.asarray(True)
    )
    return (
        constant * intrasite,
        jnp.where(owner == 0, constant * background, 0.0),
        neutral_ok,
        left_charge,
    )


def _kind_terms(
    prepared: PreparedReciprocalDistribution,
    inputs: _Inputs,
    payload: ReciprocalSitePayload,
    axis_name: str,
    /,
) -> _Terms:
    plan = prepared.plan
    definition = plan.definition
    dtype = inputs.positions.dtype
    zero = jnp.zeros((), dtype=dtype)
    true = jnp.asarray(True)
    match definition:
        case PreparedParticleMeshEwaldPotential() | PreparedEwaldReferencePotential():
            return _coulomb_terms(prepared, definition, inputs, payload, axis_name)
        case PreparedLatentEwald():
            latent = definition.definition
            charges = _scalar_values(payload, inputs.mask, dtype)
            removed = (
                -latent.self_coefficient.astype(charges.dtype)
                * jnp.sum(charges * charges)
                if latent.remove_self_interaction
                else zero
            )
            return _Terms(definition.energy_factor * removed, zero, true, zero, true)
        case PreparedGeneralForceFieldTerm():
            if payload.rows is None:
                raise ValueError("LJ-PME payloads need site type rows.")
            c6 = definition.plan.arrays[0]
            alpha = definition.plan.arrays[1].reshape(()).astype(dtype)
            local = lennard_jones_pme_self_energy(
                alpha, c6, jnp.asarray(payload.rows, dtype=jnp.int32), inputs.mask
            )
            return _Terms(local, zero, true, zero, true)
        case PreparedFourierD3Potential():
            if payload.rows is None:
                raise ValueError("FourierD3 payloads need species channel rows.")
            coefficients = _scalar_values(payload, inputs.mask, dtype)
            local = definition.self_energy(
                coefficients, jnp.asarray(payload.rows, dtype=jnp.int32), inputs.mask
            )
            return _Terms(local * definition.energy_scale, zero, true, zero, true)
        case PreparedMultipoleElectrostatics():
            if payload.moments is None:
                raise ValueError("Multipole payloads need owned moments.")
            intrasite, background, neutral, charge = _multipole_parts(
                prepared,
                definition,
                inputs,
                payload.moments,
                payload.moments,
                payload.rows,
                inputs.mask,
                inputs.mask,
                axis_name,
            )
            return _Terms(0.5 * intrasite, 0.5 * background, neutral, charge, true)
    raise RuntimeError(f"Unhandled reciprocal kind {plan.kind!r}.")


def _scaled_inputs(
    prepared: PreparedReciprocalDistribution, inputs: _Inputs, /
) -> _Inputs:
    """Latent Ewald evaluates in its source length unit (exact static factor)."""
    definition = prepared.plan.definition
    if not isinstance(definition, PreparedLatentEwald):
        return inputs
    factor = jnp.asarray(definition.length_factor, dtype=inputs.positions.dtype)
    return _Inputs(inputs.positions * factor, inputs.mask, inputs.vectors * factor)


def _reciprocal_factor(prepared: PreparedReciprocalDistribution, /) -> float:
    definition = prepared.plan.definition
    match definition:
        case PreparedLatentEwald():
            return float(definition.energy_factor)
        case PreparedFourierD3Potential():
            return float(definition.energy_scale)
        case PreparedMultipoleElectrostatics():
            return 0.5
    return 1.0


def _coverage(
    prepared: PreparedReciprocalDistribution,
    support: Array,
    vectors: Array,
    /,
) -> tuple[Array, Array]:
    """Declared mode coverage under the current cell and the latent orientation."""
    plan = prepared.plan
    definition = plan.definition
    support = jax.lax.stop_gradient(support)
    true = jnp.asarray(True)
    match definition:
        case PreparedParticleMeshEwaldPotential() | PreparedEwaldReferencePotential():
            cutoff = definition.plan.settings.reciprocal_cutoff
            return (true if cutoff is None else cutoff <= support), true
        case PreparedMultipoleElectrostatics():
            cutoff = _periodic_route_settings(definition).reciprocal_cutoff
            return (true if cutoff is None else cutoff <= support), true
        case PreparedLatentEwald():
            latent = definition.definition
            oriented = _orientation(vectors) > 0.0
            if latent.mode_support == "complete-sphere":
                return support > latent.reciprocal_cutoff.astype(support.dtype), oriented
            return true, oriented
    return true, true


def _run(
    prepared: PreparedReciprocalDistribution,
    positions: ArrayLike,
    payload: ReciprocalSitePayload,
    atom_mask: ArrayLike,
    cell_vectors: ArrayLike,
    axis_name: str,
    /,
    *,
    actions: bool,
) -> tuple[
    Array, ReciprocalDistributionEvidence, Array | None, ReciprocalSitePayload | None
]:
    if not isinstance(prepared, PreparedReciprocalDistribution):
        raise TypeError("prepared must be a PreparedReciprocalDistribution.")
    if axis_name != prepared.axis_name:
        raise ValueError(
            f"Owner region axis {axis_name!r} is not the ownership axis "
            f"{prepared.axis_name!r}."
        )
    if not isinstance(payload, ReciprocalSitePayload):
        # The C1 runtime passes a model's ``global_payload`` unchanged; for the
        # charge-like kinds that is the owned charge column itself.
        scalar_kinds = (
            AtomisticReciprocalKind.PME,
            AtomisticReciprocalKind.EWALD,
            AtomisticReciprocalKind.LATENT_EWALD,
        )
        if prepared.kind not in scalar_kinds or not isinstance(
            payload, (jax.Array, np.ndarray)
        ):
            raise TypeError(
                "payload must be a ReciprocalSitePayload (or the owned charge array "
                "of a PME, Ewald or latent Ewald term)."
            )
        payload = ReciprocalSitePayload(values=jnp.asarray(payload))
    inputs_finite = _global_all(
        _active_finite((positions, payload), jnp.asarray(atom_mask, dtype=jnp.bool_))
        & jnp.all(jnp.isfinite(jnp.asarray(cell_vectors))),
        axis_name,
    )
    base = _inputs(prepared, positions, atom_mask, cell_vectors)
    inputs = _scaled_inputs(prepared, base)
    factor = _reciprocal_factor(prepared)
    if prepared.mesh_layout is not None:
        reciprocal = _mesh_reciprocal(
            prepared,
            inputs.positions,
            payload,
            inputs.mask,
            inputs.vectors,
            axis_name,
            actions=actions,
        )
    else:
        reciprocal = _mode_reciprocal(
            prepared,
            inputs.positions,
            payload,
            inputs.mask,
            inputs.vectors,
            axis_name,
            actions=actions,
        )
    dynamic, static = _differentiable(payload)

    def terms(position: Array, values: ReciprocalSitePayload) -> tuple[Array, _Terms]:
        bound = eqx.combine(values, static)
        evaluated = _kind_terms(
            prepared, _Inputs(position, base.mask, base.vectors), bound, axis_name
        )
        return evaluated.local + evaluated.system, evaluated

    position_gradient = payload_gradient = None
    if actions:
        term_value, term_vjp, term_parts = jax.vjp(
            terms, base.positions, dynamic, has_aux=True
        )
        term_position, term_payload = term_vjp(jnp.ones_like(term_value))
        definition = prepared.plan.definition
        length = (
            jnp.asarray(definition.length_factor, dtype=base.positions.dtype)
            if isinstance(definition, PreparedLatentEwald)
            else 1.0
        )
        reciprocal_position = reciprocal.position_gradient
        reciprocal_payload = reciprocal.payload_gradient
        if reciprocal_position is None or reciprocal_payload is None:
            raise RuntimeError("Reciprocal actions were not formed.")
        position_gradient = term_position + factor * length * reciprocal_position
        payload_gradient = jax.tree.map(
            lambda direct, mesh: direct + factor * mesh,
            term_payload,
            reciprocal_payload,
        )
    else:
        term_value, term_parts = terms(base.positions, dynamic)
    reciprocal_share = factor * reciprocal.value
    share = reciprocal_share + term_value
    coverage, oriented = _coverage(prepared, reciprocal.support_radius, inputs.vectors)
    capacity = reciprocal.patch_covered & reciprocal.packets_fit
    scientific = (
        reciprocal.cell_successful
        & reciprocal.sources_finite
        & inputs_finite
        & reciprocal.influence_finite
        & reciprocal.anchor_successful
        & _global_all(term_parts.neutral, axis_name)
        & term_parts.geometry_successful
        & _global_all(coverage & oriented, axis_name)
        & _global_all(jnp.isfinite(share), axis_name)
    )
    successful = capacity & scientific
    evidence = ReciprocalDistributionEvidence(
        energy_share=share,
        reciprocal_share=reciprocal_share,
        local_share=term_parts.local,
        system_share=term_parts.system,
        patch_certified=reciprocal.patch_certified,
        patch_covered=reciprocal.patch_covered,
        packets_fit=reciprocal.packets_fit,
        maximum_patch_extent=reciprocal.maximum_extent,
        maximum_packet_load=reciprocal.maximum_load,
        cell_successful=reciprocal.cell_successful,
        sources_finite=reciprocal.sources_finite & inputs_finite,
        influence_finite=reciprocal.influence_finite,
        coverage_successful=_global_all(coverage & oriented, axis_name),
        neutral=_global_all(term_parts.neutral, axis_name),
        total_charge=term_parts.total_charge,
        capacity_successful=capacity,
        scientific_successful=scientific,
        successful=successful,
        kind=prepared.kind.value,
        strategy=prepared.report.strategy,
        prepared_identity=prepared.prepared_identity,
    )
    return share, evidence, position_gradient, payload_gradient


def owner_reciprocal_energy(
    prepared: PreparedReciprocalDistribution,
    positions: ArrayLike,
    payload: ReciprocalSitePayload,
    atom_mask: ArrayLike,
    cell_vectors: ArrayLike,
    axis_name: str,
    /,
) -> tuple[Array, ReciprocalDistributionEvidence]:
    """This owner's share of the term's reciprocal, self and system energy.

    Differentiable in positions, payload and cell through the collectives;
    the owner sum of the shares is the single-device term's reciprocal plus
    self/background/boundary energy (real-space pairs and topology exceptions
    belong to the domain pair relation). Evidence status is global.
    """
    share, evidence, _, _ = _run(
        prepared, positions, payload, atom_mask, cell_vectors, axis_name, actions=False
    )
    return share, evidence


def owner_reciprocal_actions(
    prepared: PreparedReciprocalDistribution,
    positions: ArrayLike,
    payload: ReciprocalSitePayload,
    atom_mask: ArrayLike,
    cell_vectors: ArrayLike,
    axis_name: str,
    /,
) -> OwnerReciprocalActions:
    """Potentials and forces of the term energy at this owner's atoms.

    Mesh routes evaluate the conjugate field with ``inverse_local`` on the
    kernel-weighted modal block, return it through the exact transpose of the
    patch packet route and gather it with the term's spline routes (the local
    adjoint of spreading); explicit modes gather the kernel-weighted structure
    factors of every chunk. Owned and system terms use their exact
    derivatives (system sums transpose through ``owner_system_sum``).
    """
    share, evidence, position_gradient, payload_gradient = _run(
        prepared, positions, payload, atom_mask, cell_vectors, axis_name, actions=True
    )
    if position_gradient is None or payload_gradient is None:
        raise RuntimeError("Reciprocal actions were not formed.")
    return OwnerReciprocalActions(
        energy_share=share,
        payload_gradient=payload_gradient,
        forces=-position_gradient,
        evidence=evidence,
    )


@final
class OwnerReciprocalTerm(StrictModule):
    """C1 ``OwnerGlobalTerm`` epilogue bound to one prepared distribution."""

    prepared: PreparedReciprocalDistribution
    axis_name: str = eqx.field(static=True)

    def __init__(
        self, prepared: PreparedReciprocalDistribution, axis_name: str, /
    ) -> None:
        if not isinstance(prepared, PreparedReciprocalDistribution):
            raise TypeError("prepared must be a PreparedReciprocalDistribution.")
        if axis_name != prepared.axis_name:
            raise ValueError("The global term must run on the ownership axis.")
        self.prepared = prepared
        self.axis_name = axis_name

    def __call__(
        self,
        positions: Array,
        payload: ReciprocalSitePayload,
        atom_mask: Array,
        cell_vectors: Array,
        /,
    ) -> tuple[Array, Array]:
        share, evidence = owner_reciprocal_energy(
            self.prepared, positions, payload, atom_mask, cell_vectors, self.axis_name
        )
        return share, evidence.successful


def owner_multipole_bilinear(
    prepared: PreparedReciprocalDistribution,
    positions: Array,
    left: CartesianMultipoles,
    right: CartesianMultipoles,
    left_mask: Array,
    right_mask: Array,
    rows: Array | None,
    cell_vectors: Array,
    axis_name: str,
    /,
) -> tuple[Array, Array, Array]:
    """Owner share of the reciprocal, intrasite and background parts of ``B(L, R)``.

    Returns ``(share, scientific_status, capacity_status)``; real space is the
    caller's owner relation. Requires a ``bilinear`` multipole plan.
    """
    plan = prepared.plan
    definition = plan.definition
    if not isinstance(definition, PreparedMultipoleElectrostatics) or not plan.bilinear:
        raise ValueError("The bilinear form needs a bilinear multipole distribution.")
    inputs = _inputs(prepared, positions, left_mask | right_mask, cell_vectors)
    dtype = inputs.positions.dtype
    degree = _channel_degree(plan, definition)
    if prepared.mesh_layout is not None:
        split = _channel_count(plan, True)

        def channels(_: ReciprocalSitePayload) -> Array:
            return jnp.concatenate(
                (
                    _multipole_channels(plan, left, left_mask, rows, degree),
                    _multipole_channels(plan, right, right_mask, rows, degree),
                ),
                axis=-1,
            ).astype(dtype)

        reciprocal = _mesh_reciprocal(
            prepared,
            inputs.positions,
            ReciprocalSitePayload(),
            inputs.mask,
            inputs.vectors,
            axis_name,
            actions=False,
            channels=channels,
            split=split,
        )
    else:
        reciprocal = _mode_reciprocal(
            prepared,
            inputs.positions,
            ReciprocalSitePayload(moments=left.masked(left_mask), rows=rows),
            inputs.mask,
            inputs.vectors,
            axis_name,
            actions=False,
            right=right.masked(right_mask),
        )
    intrasite, background, neutral, _ = _multipole_parts(
        prepared, definition, inputs, left, right, rows, left_mask, right_mask, axis_name
    )
    coverage, _ = _coverage(prepared, reciprocal.support_radius, inputs.vectors)
    share = reciprocal.value + intrasite + background
    scientific = (
        reciprocal.cell_successful
        & reciprocal.sources_finite
        & _global_all(
            _active_finite((positions, left), left_mask)
            & _active_finite((positions, right), right_mask)
            & jnp.all(jnp.isfinite(jnp.asarray(cell_vectors))),
            axis_name,
        )
        & reciprocal.influence_finite
        & _global_all(neutral & coverage, axis_name)
    )
    capacity = reciprocal.patch_covered & reciprocal.packets_fit
    return share, scientific, capacity


# --- owner-blocked multipole pair relations ------------------------------------------


class OwnerMultipoleRoutes[Leaf = Array](NamedTuple):
    """Directed routes of one owner (owner-blocked globally; local in a region).

    ``receivers`` index owned rows; ``columns`` index the halo column space
    (owned rows then deduplicated halo rows); the route displacement is the
    domain convention ``d = x_receiver - x_column + shifts @ H`` (ignored for
    minimum-image relations); ``weights`` are the ``direct, polarization,
    mutual`` route factors. ``Leaf`` is the per-field payload: arrays for the
    routes themselves, `PartitionSpec` for their ``ownership.map`` specs.
    """

    receivers: Leaf
    columns: Leaf
    shifts: Leaf
    weights: Leaf
    valid: Leaf
    send_slots: Leaf
    send_valid: Leaf


@final
class OwnerMultipoleRelation(StrictModule, NonTrainableState):
    """Owner-blocked directed multipole pair routes with a halo column exchange.

    A real-space relation lists every directed image route ``receiver <- source``
    of owned receivers; its ``weights`` are per-route scale factors of
    nonperiodic routes (one for periodic real space, where scales act through
    corrections). A ``minimum_image`` relation is the scaled-pair correction
    topology: each listed pair contributes both directed routes at the
    current-cell minimum image with weights ``scale + offset``. Halo refusal is
    capacity evidence.

    The constructor is traceable (no host conversion): it binds an existing
    :class:`DistributedHaloPlan` and its owner-blocked routes, so a domain
    topology rebuilt inside ``jit`` binds through :meth:`from_domain_topology`.
    :meth:`from_routes` and :meth:`scaled_pairs` are host preparations.
    """

    halo: DistributedHaloPlan
    routes: OwnerMultipoleRoutes
    cell: PeriodicCell | None
    minimum_image: bool = eqx.field(static=True)
    route_capacity: int = eqx.field(static=True)
    message_capacity_bytes: int = eqx.field(static=True)
    relation_id: str = eqx.field(static=True)

    def __init__(
        self,
        halo: DistributedHaloPlan,
        routes: OwnerMultipoleRoutes,
        /,
        *,
        minimum_image: bool = False,
        cell: PeriodicCell | None = None,
        message_capacity_bytes: int = 1 << 26,
    ) -> None:
        if not isinstance(halo, DistributedHaloPlan):
            raise TypeError("halo must be a DistributedHaloPlan.")
        if not isinstance(routes, OwnerMultipoleRoutes):
            raise TypeError("routes must be OwnerMultipoleRoutes.")
        ownership = halo.ownership
        total = routes.receivers.shape[0] if routes.receivers.ndim == 1 else -1
        if (
            total < 0
            or total % ownership.owner_count
            or routes.columns.shape != (total,)
            or routes.valid.shape != (total,)
            or routes.shifts.shape != (total, 3)
            or routes.weights.shape != (total, 3)
            or routes.send_slots.shape != halo.send_slots.shape
            or routes.send_valid.shape != halo.send_valid.shape
        ):
            raise ValueError(
                "Relation routes must be owner-blocked vectors of one length."
            )
        if not jnp.issubdtype(routes.receivers.dtype, jnp.integer) or not jnp.issubdtype(
            routes.columns.dtype, jnp.integer
        ):
            raise TypeError("Route receivers and columns must be integer slots.")
        if routes.valid.dtype != jnp.bool_:
            raise TypeError("Route validity must be boolean.")
        if minimum_image and not isinstance(cell, PeriodicCell):
            raise ValueError("Minimum-image corrections need the periodic cell.")
        capacity = _positive_int(message_capacity_bytes, "message_capacity_bytes")
        self.halo = halo
        self.routes = routes
        self.cell = cell
        self.minimum_image = bool(minimum_image)
        self.route_capacity = total // ownership.owner_count
        self.message_capacity_bytes = capacity
        self.relation_id = canonical_fingerprint(
            {
                "kind": "owner-multipole-relation",
                "ownership": ownership.plan_id,
                "routes": int(total),
                "halo_capacity": int(halo.halo_capacity),
                "minimum_image": self.minimum_image,
                "cell": None if cell is None else cell.cell_id,
            }
        )

    @classmethod
    def from_routes(
        cls,
        ownership: DistributedOwnershipPlan,
        receivers: ArrayLike,
        source_owners: ArrayLike,
        source_slots: ArrayLike,
        shifts: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        halo_capacity: int,
        weights: ArrayLike | None = None,
        minimum_image: bool = False,
        cell: PeriodicCell | None = None,
        message_capacity_bytes: int = 1 << 26,
    ) -> "OwnerMultipoleRelation":
        """Host preparation from owner-blocked ``(receiver slot, source owner/slot, n)``."""
        if not isinstance(ownership, DistributedOwnershipPlan):
            raise TypeError("ownership must be a DistributedOwnershipPlan.")
        receiver = np.asarray(receivers, dtype=np.int32)
        owners = np.asarray(source_owners, dtype=np.int32)
        slots = np.asarray(source_slots, dtype=np.int32)
        shift = np.asarray(shifts, dtype=np.float64)
        mask = np.asarray(valid, dtype=np.bool_)
        total = receiver.shape[0]
        if (
            receiver.ndim != 1
            or total % ownership.owner_count
            or owners.shape != receiver.shape
            or slots.shape != receiver.shape
            or mask.shape != receiver.shape
            or shift.shape != (total, 3)
        ):
            raise ValueError(
                "Relation routes must be owner-blocked vectors of one length."
            )
        route_weights = (
            np.ones((total, 3), dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        if route_weights.shape != (total, 3) or not np.all(np.isfinite(route_weights)):
            raise ValueError("Route weights must be finite (routes, 3) factors.")
        if np.any(mask & ((receiver < 0) | (receiver >= ownership.local_capacity))):
            raise ValueError("Route receivers must be owned slots.")
        halo = DistributedHaloPlan(
            ownership, owners, slots, mask, halo_capacity=halo_capacity
        )
        return cls(
            halo,
            OwnerMultipoleRoutes(
                receivers=ownership.place(jnp.asarray(receiver)),
                columns=halo.route_columns,
                shifts=ownership.place(jnp.asarray(shift)),
                weights=ownership.place(jnp.asarray(route_weights)),
                valid=halo.route_valid,
                send_slots=halo.send_slots,
                send_valid=halo.send_valid,
            ),
            minimum_image=minimum_image,
            cell=cell,
            message_capacity_bytes=message_capacity_bytes,
        )

    @classmethod
    def from_domain_topology(
        cls,
        topology: "AtomisticDomainTopology",
        /,
        *,
        weights: ArrayLike | None = None,
        message_capacity_bytes: int = 1 << 26,
    ) -> "OwnerMultipoleRelation":
        """Traceable binding of one halo-strategy domain role topology.

        Reuses the topology's halo plan, receivers, halo columns, integer
        shifts (domain convention ``d = x_r - x_s + n @ H``) and executable
        validity without host conversion. ``weights`` defaults to one per
        route (periodic real space); nonperiodic routes may carry their
        ``direct, polarization, mutual`` pair scales.
        """
        halo = topology.halo
        if halo is None:
            raise ValueError(
                "Multipole real space binds a halo-strategy topology (owned plus halo "
                "columns); a graph-partition topology has no halo plan."
            )
        total = topology.receivers.shape[0]
        route_weights = (
            jnp.ones((total, 3), dtype=jnp.float64)
            if weights is None
            else jnp.asarray(weights, dtype=jnp.float64)
        )
        return cls(
            halo,
            OwnerMultipoleRoutes(
                receivers=topology.receivers,
                columns=topology.columns,
                shifts=topology.shifts,
                weights=route_weights,
                valid=topology.valid,
                send_slots=halo.send_slots,
                send_valid=halo.send_valid,
            ),
            message_capacity_bytes=message_capacity_bytes,
        )

    @classmethod
    def scaled_pairs(
        cls,
        ownership: DistributedOwnershipPlan,
        scales: MultipolePairScales,
        row_owners: ArrayLike,
        row_slots: ArrayLike,
        cell: PeriodicCell,
        /,
        *,
        route_capacity: int,
        halo_capacity: int,
        offset: float = -1.0,
        message_capacity_bytes: int = 1 << 26,
    ) -> "OwnerMultipoleRelation":
        """Owner-blocked correction topology of the canonical sparse pair scales.

        Host preparation (at ingress and after every committed rebuild):
        ``row_owners``/``row_slots`` map every canonical system site to its
        current owner and slot. Both directed routes of each listed pair live
        with their receiver; per-owner route counts above ``route_capacity``
        refuse with :class:`ReciprocalDistributionCapacityError`.
        """
        if not isinstance(scales, MultipolePairScales):
            raise TypeError("scales must be MultipolePairScales.")
        capacity = _positive_int(route_capacity, "route_capacity")
        owner_of = np.asarray(row_owners, dtype=np.int32)
        slot_of = np.asarray(row_slots, dtype=np.int32)
        pairs = np.asarray(scales.pairs, dtype=np.int64).reshape((-1, 2))
        values = np.asarray(scales.values, dtype=np.float64).reshape((pairs.shape[0], -1))
        columns = [scales.column(name) for name in _SCALE_COLUMNS]
        weights = values[:, columns] + float(offset)
        owners = ownership.owner_count
        receiver = np.zeros((owners, capacity), dtype=np.int32)
        source_owner = np.zeros((owners, capacity), dtype=np.int32)
        source_slot = np.zeros((owners, capacity), dtype=np.int32)
        route_weights = np.zeros((owners, capacity, 3), dtype=np.float64)
        valid = np.zeros((owners, capacity), dtype=np.bool_)
        load = np.zeros((owners,), dtype=np.int64)
        for index, (first, second) in enumerate(pairs):
            for left, right in ((first, second), (second, first)):
                owner = int(owner_of[left])
                position = int(load[owner])
                load[owner] += 1
                if position >= capacity:
                    continue
                receiver[owner, position] = slot_of[left]
                source_owner[owner, position] = owner_of[right]
                source_slot[owner, position] = slot_of[right]
                route_weights[owner, position] = weights[index]
                valid[owner, position] = True
        if np.any(load > capacity):
            raise ReciprocalDistributionCapacityError(
                f"Scaled-pair routes need {int(np.max(load))} per owner, exceeding "
                f"route_capacity={capacity}."
            )
        return cls.from_routes(
            ownership,
            receiver.reshape((-1,)),
            source_owner.reshape((-1,)),
            source_slot.reshape((-1,)),
            np.zeros((owners * capacity, 3)),
            valid.reshape((-1,)),
            halo_capacity=halo_capacity,
            weights=route_weights.reshape((-1, 3)),
            minimum_image=True,
            cell=cell,
            message_capacity_bytes=message_capacity_bytes,
        )

    def route_specs(self) -> OwnerMultipoleRoutes[PartitionSpec]:
        """``ownership.map`` in-specs of :attr:`routes` (owner-blocked leading axes)."""
        axis = self.halo.ownership.axis_name
        return OwnerMultipoleRoutes(
            receivers=PartitionSpec(axis),
            columns=PartitionSpec(axis),
            shifts=PartitionSpec(axis, None),
            weights=PartitionSpec(axis, None),
            valid=PartitionSpec(axis),
            send_slots=PartitionSpec(axis, None),
            send_valid=PartitionSpec(axis, None),
        )


class _Sites(NamedTuple):
    moments: CartesianMultipoles
    active: Array
    width_squared: Array
    damping: Array


def _pair_sites(sites: _Sites, index: Array, /) -> _PairSites:
    moments = sites.moments.masked(sites.active)
    return _PairSites(
        charges=moments.charges[index],
        dipoles=moments.dipoles[index],
        second=moments.second_moments[index],
        width_squared=sites.width_squared[index],
        damping=sites.damping[index],
    )


def _route_bilinear(
    relation: OwnerMultipoleRelation,
    routes: OwnerMultipoleRoutes,
    positions: Array,
    vectors: Array,
    left: _Sites,
    right: _Sites,
    kernel: AbstractMultipoleRadialKernel,
    /,
    *,
    cutoff: float | None,
    minimum_distance: float,
    column: int | None,
) -> tuple[Array, Array]:
    """Owner share ``sum_routes w U(L_receiver, R_source; d)`` and its status."""
    exchange = HaloColumnExchange(
        relation.halo,
        routes.send_slots,
        routes.send_valid,
        relation.message_capacity_bytes,
    )
    gathered = exchange.gather(
        {
            "positions": positions,
            "charges": right.moments.charges,
            "dipoles": right.moments.dipoles,
            "second": right.moments.second_moments,
            "width_squared": right.width_squared,
            "damping": right.damping,
            "active": right.active,
        }
    )
    columns = routes.columns
    receivers = routes.receivers
    degree = max(left.moments.maximum_degree, right.moments.maximum_degree)
    order = 2 * degree
    dtype = positions.dtype
    active_columns = gathered["active"]
    right_route = _PairSites(
        charges=jnp.where(active_columns, gathered["charges"], 0.0)[columns],
        dipoles=jnp.where(active_columns[:, None], gathered["dipoles"], 0.0)[columns],
        second=jnp.where(active_columns[:, None, None], gathered["second"], 0.0)[columns],
        width_squared=gathered["width_squared"][columns],
        damping=gathered["damping"][columns],
    )
    left_route = _pair_sites(left, receivers)
    source = gathered["positions"][columns]
    if relation.minimum_image:
        if relation.cell is None:
            raise RuntimeError("Minimum-image relation without a cell.")
        difference = positions[receivers] - source
        displacement = relation.cell.minimum_image_with_vectors(difference, vectors)
        reach = relation.cell.axis_reach_with_vectors(vectors)
        length = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        unique = jnp.all(length[:, None] * reach[None, :] < 0.5, axis=-1)
    else:
        # Domain edge convention (``owner_edge_geometry``): d = x_r - x_s + n @ H.
        displacement = (
            positions[receivers]
            - source
            + contract("ea,ad->ed", routes.shifts.astype(dtype), vectors)
        )
        unique = jnp.ones(receivers.shape, dtype=jnp.bool_)
    squared = jnp.sum(displacement * displacement, axis=-1)
    weight = (
        jnp.ones(receivers.shape, dtype=dtype)
        if column is None
        else routes.weights[:, column].astype(dtype)
    )
    routed = (
        routes.valid & left.active[receivers] & active_columns[columns] & (weight != 0.0)
    )
    if cutoff is not None:
        routed = routed & (squared < cutoff * cutoff)
    positive = squared > minimum_distance * minimum_distance
    safe = jnp.sqrt(jnp.where(routed & positive, squared, 1.0))
    coefficients = kernel.coefficients(safe, left_route, right_route, order)
    value = weight * _pair_interaction(
        coefficients, displacement, left_route, right_route, degree
    )
    total = jnp.sum(jnp.where(routed & positive, value, 0.0))
    status = jnp.all(~routed | (positive & unique))
    return total, status


def _route_row_norms(
    relation: OwnerMultipoleRelation,
    routes: OwnerMultipoleRoutes,
    positions: Array,
    sites: _Sites,
    root_alpha: Array,
    kernel: AbstractMultipoleRadialKernel,
    minimum_distance: float,
    /,
) -> tuple[Array, Array]:
    """Owned rows ``sum_j |w_i w_j s_ij| ||grad grad phi_ij||_2`` (nonperiodic)."""
    exchange = HaloColumnExchange(
        relation.halo,
        routes.send_slots,
        routes.send_valid,
        relation.message_capacity_bytes,
    )
    gathered = exchange.gather(
        {
            "positions": positions,
            "weight": root_alpha,
            "width_squared": sites.width_squared,
            "damping": sites.damping,
        }
    )
    columns, receivers = routes.columns, routes.receivers
    dtype = positions.dtype
    displacement = positions[receivers] - gathered["positions"][columns]
    squared = jnp.sum(displacement * displacement, axis=-1)
    factor = (
        jnp.abs(root_alpha[receivers])
        * jnp.abs(gathered["weight"][columns])
        * jnp.abs(routes.weights[:, _SCALE_COLUMNS.index("mutual")].astype(dtype))
    )
    routed = routes.valid & (factor != 0.0)
    positive = squared > minimum_distance * minimum_distance
    safe = jnp.sqrt(jnp.where(routed & positive, squared, 1.0))
    zeros = jnp.zeros(receivers.shape, dtype=dtype)
    left = _PairSites(
        charges=zeros,
        dipoles=jnp.zeros(receivers.shape + (3,), dtype=dtype),
        second=jnp.zeros(receivers.shape + (3, 3), dtype=dtype),
        width_squared=sites.width_squared[receivers],
        damping=sites.damping[receivers],
    )
    right = _PairSites(
        charges=zeros,
        dipoles=jnp.zeros(receivers.shape + (3,), dtype=dtype),
        second=jnp.zeros(receivers.shape + (3, 3), dtype=dtype),
        width_squared=gathered["width_squared"][columns],
        damping=gathered["damping"][columns],
    )
    coefficients = kernel.coefficients(safe, left, right, 2)
    first, second = coefficients[1], coefficients[2]
    norm = jnp.maximum(jnp.abs(second * safe * safe - first), jnp.abs(first))
    contribution = jnp.where(routed & positive, factor * norm, 0.0)
    contribution = jnp.where(routed & ~positive, jnp.inf, contribution)
    rows = jax.ops.segment_sum(contribution, receivers, positions.shape[0])
    return rows, jnp.all(~routed | positive)


# --- owner-local polarization ----------------------------------------------------------


@final
class OwnerPolarizationPlan(StrictModule, NonTrainableState):
    """Owner-local execution of one canonical prepared polarization solver.

    ``solver`` fixes the operator definition (constant, site kind, damping,
    route, Ewald split, minimum distance), the PCG/TCG policy and tolerances,
    the preconditioner, the predictor and the stability policy. Periodic
    routes distribute their reciprocal bilinear through a ``bilinear``
    :class:`AtomisticReciprocalDistributionPlan` of the solver's own
    electrostatics; the remaining keywords configure it. ``plan_identity`` is
    the exact canonical plan digest as dynamic `IdentityWords` (``plan_id``
    decodes it on the host). ``scientific_identity`` (host ``scientific_id``)
    is its layout-independent dynamic digest: the solver and the reciprocal
    ``scientific_id`` of its periodic bilinear, without that bilinear's
    placement and capacities.
    """

    solver: PreparedPolarizationSolver
    reciprocal: AtomisticReciprocalDistributionPlan | None
    plan_identity: IdentityWords
    scientific_identity: IdentityWords

    def __init__(
        self,
        solver: PreparedPolarizationSolver,
        /,
        *,
        reuse_reach: float,
        strategy: AtomisticReciprocalStrategy = "distributed-fft",
        schedule: AtomisticReciprocalSchedule = "slab",
        pencil_grid: tuple[int, int] | None = None,
        cell_strain_allowance: float = 0.0,
        patch_margin_cells: int = 1,
        packet_planes: int | None = None,
        packet_lines: int | None = None,
        mode_chunk: int | None = None,
        maximum_packet_bytes: int = 1 << 28,
        reduction_mode: ReductionMode = "deterministic",
    ) -> None:
        if not isinstance(solver, PreparedPolarizationSolver):
            raise TypeError("solver must be a PreparedPolarizationSolver.")
        if solver.plan.stability.certificate == "dense-spectrum":
            raise ValueError(
                "The dense-spectrum certificate materialises the global (3N)^2 "
                "operator on one device, which owner-local execution excludes; use "
                "the scaled block-Gershgorin proof (nonperiodic routes) or the "
                "declared assumption with distributed Lanczos detection."
            )
        operator = solver.operator
        reciprocal = None
        if operator.plan.route.periodic:
            reciprocal = AtomisticReciprocalDistributionPlan(
                operator.electrostatics,
                reuse_reach=reuse_reach,
                strategy=strategy,
                schedule=schedule,
                pencil_grid=pencil_grid,
                cell_strain_allowance=cell_strain_allowance,
                patch_margin_cells=patch_margin_cells,
                packet_planes=packet_planes,
                packet_lines=packet_lines,
                mode_chunk=mode_chunk,
                widths=operator.widths,
                bilinear=True,
                maximum_packet_bytes=maximum_packet_bytes,
                reduction_mode=reduction_mode,
            )
        self.solver = solver
        self.reciprocal = reciprocal
        self.scientific_identity = IdentityWords.of(
            {
                "kind": "owner-polarization-definition",
                "solver": solver.prepared_id,
                "reciprocal": None if reciprocal is None else reciprocal.scientific_id,
            }
        )
        self.plan_identity = IdentityWords.of(
            {
                "kind": "owner-polarization-plan",
                "solver": solver.prepared_id,
                "reciprocal": None if reciprocal is None else reciprocal.plan_id,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical owner-polarization plan digest."""
        return self.plan_identity.fingerprint()

    @property
    def scientific_id(self) -> str:
        """Host decoding of the layout-independent solver-definition digest."""
        return self.scientific_identity.fingerprint()


@final
class PreparedOwnerPolarization(StrictModule, NonTrainableState):
    """Owner-local polarization bound to an ownership plan and partition.

    ``prepared_identity`` is the exact canonical preparation digest as dynamic
    `IdentityWords` (``prepared_id`` decodes it on the host). ``layout_id`` is
    the content-free static label of the distributed operator (solver layout,
    ownership and partition) used for the linear-algebra operator and pairing
    labels, so revisions of the solver share one executable.
    """

    plan: OwnerPolarizationPlan
    reciprocal: PreparedReciprocalDistribution | None
    class_widths: Array | None
    axis_name: str = eqx.field(static=True)
    owner_count: int = eqx.field(static=True)
    local_capacity: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    prepared_identity: IdentityWords

    def __init__(
        self,
        plan: OwnerPolarizationPlan,
        ownership: DistributedOwnershipPlan,
        partition: FractionalOwnerPartition,
        /,
    ) -> None:
        if not isinstance(plan, OwnerPolarizationPlan):
            raise TypeError("plan must be an OwnerPolarizationPlan.")
        reciprocal = (
            None
            if plan.reciprocal is None
            else prepare_reciprocal_distribution(plan.reciprocal, ownership, partition)
        )
        widths = plan.solver.operator.widths
        self.plan = plan
        self.reciprocal = reciprocal
        self.class_widths = None if widths is None else jnp.asarray(widths.class_widths)
        self.axis_name = ownership.axis_name
        self.owner_count = ownership.owner_count
        self.local_capacity = ownership.local_capacity
        self.layout_id = canonical_fingerprint(
            {
                "kind": "prepared-owner-polarization-layout",
                "solver": plan.solver.layout_id,
                "ownership": ownership.plan_id,
                "partition": partition.partition_id,
                "reciprocal": reciprocal is not None,
            }
        )
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-owner-polarization",
                "plan": plan.plan_id,
                "ownership": ownership.plan_id,
                "partition": partition.partition_id,
                "reciprocal": None if reciprocal is None else reciprocal.prepared_id,
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical preparation digest."""
        return self.prepared_identity.fingerprint()

    def initial_predictor_state(self, /) -> PolarizationPredictorState:
        """Owner-blocked empty warm-start history ``(owners * 2, local, 3)``.

        Every owner row carries this preparation's identity words
        (``(owners, 8)``), sliced with the history inside an owner region.
        """
        dtype = self.plan.solver.operator.multipoles.dipoles.dtype
        return PolarizationPredictorState(
            jnp.zeros((self.owner_count * 2, self.local_capacity, 3), dtype=dtype),
            jnp.zeros((self.owner_count,), dtype=jnp.int32),
            IdentityWords(
                jnp.tile(self.prepared_identity.words[None], (self.owner_count, 1))
            ),
        )


def prepare_owner_polarization(
    plan: OwnerPolarizationPlan,
    ownership: DistributedOwnershipPlan,
    partition: FractionalOwnerPartition,
    /,
) -> PreparedOwnerPolarization:
    """Host preparation of the owner-local polarization solve."""
    return PreparedOwnerPolarization(plan, ownership, partition)


class OwnerPolarizationSites(NamedTuple):
    """This owner's permanent moments, polarizabilities, damping and width classes."""

    moments: CartesianMultipoles
    polarizabilities: Array
    damping: Array
    width_classes: Array | None = None


class OwnerPolarizationRelations(NamedTuple):
    """Real-space relation (and optional scaled-pair corrections) with local routes."""

    real_space: OwnerMultipoleRelation
    real_routes: OwnerMultipoleRoutes
    corrections: OwnerMultipoleRelation | None = None
    correction_routes: OwnerMultipoleRoutes | None = None


class OwnerPolarizationResult(NamedTuple):
    """Owner share of the polarization functional with global solver evidence.

    ``induced_dipoles``/``permanent_field``/``direct_field`` are owned rows;
    ``state`` and ``stability`` statuses are globally reduced. The share is
    NaN on every owner when the global result is unsuccessful.
    """

    energy_share: Array
    induced_dipoles: Array
    state: PolarizationState
    predictor_state: PolarizationPredictorState
    stability: PolarizationStabilityEvidence
    permanent_field: Array
    direct_field: Array
    initial_dipoles: Array
    breakdown: Array
    successful: Array


class _OwnerBound(NamedTuple):
    positions: Array
    vectors: Array
    permanent: _Sites
    alpha: Array
    polarizable: Array
    mask: Array
    rows: Array | None


def _polarization_form(
    prepared: PreparedOwnerPolarization,
    relations: OwnerPolarizationRelations,
    bound: _OwnerBound,
    left: _Sites,
    right: _Sites,
    column: str,
    axis_name: str,
    /,
) -> tuple[Array, Array, Array]:
    """Owner share of ``B_column(L, R)`` with scientific and capacity status."""
    solver = prepared.plan.solver
    operator = solver.operator
    constant = operator.plan.coulomb_constant
    minimum = operator.plan.minimum_distance
    index = _SCALE_COLUMNS.index(column)
    real, real_routes = relations.real_space, relations.real_routes
    direct = operator._direct_kernel()
    capacity = _global_all(real.halo.evidence.successful, axis_name)
    route = operator.plan.route
    if isinstance(route, FiniteMultipoleRoute):
        value, status = _route_bilinear(
            real,
            real_routes,
            bound.positions,
            bound.vectors,
            left,
            right,
            direct,
            cutoff=None,
            minimum_distance=minimum,
            column=index,
        )
        return constant * value, _global_all(status, axis_name), capacity
    settings = route.settings
    gaussian = operator.plan.site_kind == "gaussian"
    kernel = (
        GaussianEwaldRealKernel(settings.alpha)
        if gaussian
        else ErfcKernel(settings.alpha)
    )
    value, status = _route_bilinear(
        real,
        real_routes,
        bound.positions,
        bound.vectors,
        left,
        right,
        kernel,
        cutoff=settings.real_cutoff,
        minimum_distance=minimum,
        column=None,
    )
    value = constant * value
    if not gaussian and operator.plan.damping == "exponential-cubic":
        damping, damping_ok = _route_bilinear(
            real,
            real_routes,
            bound.positions,
            bound.vectors,
            left,
            right,
            ExponentialDampedCoulombKernel(correction=True),
            cutoff=settings.real_cutoff,
            minimum_distance=minimum,
            column=None,
        )
        value = value + constant * damping
        status = status & damping_ok
    if relations.corrections is not None:
        if relations.correction_routes is None:
            raise ValueError("Correction relations need their local routes.")
        correction, correction_ok = _route_bilinear(
            relations.corrections,
            relations.correction_routes,
            bound.positions,
            bound.vectors,
            left,
            right,
            direct,
            cutoff=None,
            minimum_distance=minimum,
            column=index,
        )
        value = value + constant * correction
        status = status & correction_ok
        capacity = capacity & _global_all(
            relations.corrections.halo.evidence.successful, axis_name
        )
    if prepared.reciprocal is None:
        raise RuntimeError("Periodic polarization lacks its reciprocal distribution.")
    reciprocal, reciprocal_ok, reciprocal_capacity = owner_multipole_bilinear(
        prepared.reciprocal,
        bound.positions,
        left.moments,
        right.moments,
        left.active,
        right.active,
        bound.rows,
        bound.vectors,
        axis_name,
    )
    return (
        value + reciprocal,
        _global_all(status, axis_name) & reciprocal_ok,
        capacity & reciprocal_capacity,
    )


def _induced_sites(bound: _OwnerBound, dipoles: Array, /) -> _Sites:
    return _Sites(
        dipole_multipoles(jnp.where(bound.polarizable[:, None], dipoles, 0.0)),
        bound.polarizable,
        bound.permanent.width_squared,
        bound.permanent.damping,
    )


def _inner(left: Array, right: Array, axis_name: str, mode: ReductionMode, /) -> Array:
    """Differentiable global inner product (exactly-once owner-sum transpose)."""
    return owner_system_sum(jnp.vdot(left, right), axis_name, mode)


def _lanczos_minimum(
    scaled: Callable[[Array], Array],
    mask: Array,
    steps: int,
    seed: int,
    margin: float,
    axis_name: str,
    /,
) -> tuple[Array, Array]:
    """Smallest Ritz value of a symmetric distributed action (full reorthogonalization).

    Inner products are collective; the tridiagonal problem is replicated.
    Returns the Ritz value and whether its residual bound is below
    ``margin * |ritz|`` (or the Krylov space was exhausted).
    """
    pairing_dot = JaxCollectiveProvider(axis_name)
    shape = mask.shape + (3,)
    key = jax.random.fold_in(jax.random.key(seed), _owner(axis_name))
    start = jnp.where(mask[:, None], jax.random.normal(key, shape), 0.0)

    def dot(left: Array, right: Array) -> Array:
        return pairing_dot.sum(jnp.vdot(left, right))

    vector = start / jnp.sqrt(jnp.maximum(dot(start, start), 1e-300))
    basis = [vector]
    alphas, betas = [], []
    previous = jnp.zeros(shape, dtype=start.dtype)
    beta = jnp.zeros((), dtype=start.dtype)
    live = jnp.asarray(True)
    active = []
    for _ in range(steps):
        current = basis[-1]
        product = jnp.where(mask[:, None], scaled(current), 0.0)
        alpha = dot(current, product)
        residual = product - alpha * current - beta * previous
        for prior in basis:
            residual = residual - dot(prior, residual) * prior
        beta_next = jnp.sqrt(jnp.maximum(dot(residual, residual), 0.0))
        active.append(live)
        alphas.append(alpha)
        live = live & (beta_next > 1e-12 * jnp.maximum(jnp.abs(alpha), 1.0))
        betas.append(jnp.where(live, beta_next, 0.0))
        previous = current
        beta = beta_next
        basis.append(jnp.where(live, residual / jnp.where(live, beta_next, 1.0), 0.0))
    used = jnp.stack(active)
    raw = jnp.stack(alphas)
    # Steps after an exhausted (invariant) Krylov space are decoupled with a
    # diagonal above every computed Ritz value, so they never become the minimum.
    ceiling = 2.0 * jnp.max(jnp.where(used, jnp.abs(raw), 0.0)) + 1.0
    diagonal = jnp.where(used, raw, ceiling)
    off = jnp.where(used[1:], jnp.stack(betas[:-1]), 0.0)
    tridiagonal = jnp.diag(diagonal) + jnp.diag(off, 1) + jnp.diag(off, -1)
    values, vectors = jnp.linalg.eigh(tridiagonal)
    ritz = values[0]
    bound = jnp.abs(betas[-1] * vectors[-1, 0])
    converged = ~live | (bound <= margin * jnp.maximum(jnp.abs(ritz), 1e-300))
    return ritz, converged


# One ordered scientific kernel, kept whole: PCG and TCG share the bound
# sites, the collective fields and the evidence; splitting it would reorder
# the owner collectives, owner-sum reductions and the custom_linear_solve
# implicit-derivative boundary on which every owner's status agrees.
def solve_owner_polarization(
    prepared: PreparedOwnerPolarization,
    positions: ArrayLike,
    sites: OwnerPolarizationSites,
    atom_mask: ArrayLike,
    cell_vectors: ArrayLike | None,
    axis_name: str,
    relations: OwnerPolarizationRelations,
    /,
    *,
    predictor_state: PolarizationPredictorState | None = None,
) -> OwnerPolarizationResult:
    """Converged (PCG) or fixed-order (TCG) induced dipoles of the owned sites.

    PCG uses the native distributed PCG with collective pairings inside
    ``custom_linear_solve`` (implicit stationary sensitivity: the energy and
    every derivative of it, including cell/strain, carry the implicit
    induced-dipole response; derivative solves use the solver's
    ``derivative_tolerance``). TCG differentiates its exactly ``tcg_order``
    steps algorithmically and is never reported converged. ``predictor_state``
    is this owner's local slice ``(2, local, 3)`` history (PCG only); a
    history whose identity words name another preparation fails the solve
    (``successful`` false) instead of raising, so the check is traceable.
    """
    if not isinstance(prepared, PreparedOwnerPolarization):
        raise TypeError("prepared must be a PreparedOwnerPolarization.")
    if axis_name != prepared.axis_name:
        raise ValueError("Polarization must run on the ownership axis.")
    solver = prepared.plan.solver
    plan = solver.plan
    operator = solver.operator
    pcg = plan.kind is PolarizationSolverKind.PCG
    position = jnp.asarray(positions)
    count = prepared.local_capacity
    if position.shape != (count, 3) or not jnp.issubdtype(position.dtype, jnp.floating):
        raise ValueError(f"positions must be floating owner rows ({count}, 3).")
    dtype = position.dtype
    mask = jnp.asarray(atom_mask)
    if mask.shape != (count,) or mask.dtype != jnp.bool_:
        raise ValueError("atom_mask must be a boolean owner-row vector.")
    periodic = operator.plan.route.periodic
    if periodic != (cell_vectors is not None):
        raise ValueError("cell_vectors are required exactly for periodic routes.")
    vectors = (
        jnp.eye(3, dtype=dtype)
        if cell_vectors is None
        else jnp.asarray(cell_vectors, dtype=dtype)
    )
    if prepared.class_widths is None:
        if sites.width_classes is not None:
            raise ValueError("Point polarization sites take no width classes.")
        width = jnp.zeros((count,), dtype=dtype)
        rows = None
    else:
        if sites.width_classes is None:
            raise ValueError("Gaussian polarization sites need width classes.")
        rows = jnp.asarray(sites.width_classes, dtype=jnp.int32)
        width = prepared.class_widths.astype(dtype)[rows] ** 2
    alpha = jnp.asarray(sites.polarizabilities, dtype=dtype)
    polarizable = mask & (alpha > 0.0)
    inputs_finite = _global_all(
        _active_finite((position, sites.moments, alpha, sites.damping), mask)
        & jnp.all(jnp.isfinite(vectors)),
        axis_name,
    )
    safe_position = jnp.where(mask[:, None] & jnp.isfinite(position), position, 0.0)
    bound = _OwnerBound(
        safe_position,
        vectors,
        _Sites(sites.moments, mask, width, jnp.asarray(sites.damping, dtype=dtype)),
        alpha,
        polarizable,
        mask,
        rows,
    )
    mode: ReductionMode = (
        "deterministic"
        if prepared.reciprocal is None
        else prepared.reciprocal.plan.reduction_mode
    )
    inverse_alpha = jnp.where(polarizable, 1.0 / jnp.where(alpha > 0.0, alpha, 1.0), 1.0)

    def form(
        state: _OwnerBound, left: _Sites, right: _Sites, column: str
    ) -> tuple[Array, Array, Array]:
        return _polarization_form(
            prepared, relations, state, left, right, column, axis_name
        )

    def field(state: _OwnerBound, column: str) -> tuple[Array, Array, Array]:
        def share(dipoles: Array) -> tuple[Array, tuple[Array, Array]]:
            value, ok, fits = form(
                state, _induced_sites(state, dipoles), state.permanent, column
            )
            return value, (ok, fits)

        gradient, (ok, fits) = jax.grad(share, has_aux=True)(
            jnp.zeros((count, 3), dtype=dtype)
        )
        return jnp.where(state.polarizable[:, None], -gradient, 0.0), ok, fits

    def induced_field(state: _OwnerBound, dipoles: Array) -> Array:
        def share(value: Array) -> Array:
            induced = _induced_sites(state, value)
            return 0.5 * form(state, induced, induced, "mutual")[0]

        return jnp.where(state.polarizable[:, None], -jax.grad(share)(dipoles), 0.0)

    def matvec_for(state: _OwnerBound) -> Callable[[Array], Array]:
        inverse = jnp.where(
            state.polarizable,
            1.0 / jnp.where(state.alpha > 0.0, state.alpha, 1.0),
            1.0,
        )

        def matvec(value: Array) -> Array:
            action = inverse[:, None] * value - induced_field(state, value)
            return jnp.where(state.polarizable[:, None], action, value)

        return matvec

    rhs, rhs_ok, rhs_fits = field(bound, "polarization")
    scaled_pairs = relations.corrections is not None
    if scaled_pairs:
        direct, direct_ok, direct_fits = field(bound, "direct")
    else:
        direct, direct_ok, direct_fits = rhs, rhs_ok, rhs_fits
    fields_ok = rhs_ok & direct_ok
    capacity_ok = rhs_fits & direct_fits
    diagonal = solver.preconditioner.diagonal(alpha, polarizable)
    pairing = DistributedPairing(
        mask, axis_name=axis_name, pairing_id=f"{prepared.layout_id}:pairing"
    )

    def precondition(residual: Array, iteration: Array) -> Array:
        del iteration
        return residual / diagonal[:, None]

    fallback = (
        alpha[:, None] * direct
        if solver.predictor.direct_fallback
        else jnp.zeros_like(direct)
    )
    predictor_bound = jnp.asarray(True)
    if pcg:
        if predictor_state is None:
            history = jnp.zeros((2, count, 3), dtype=dtype)
            valid_count = jnp.zeros((), dtype=jnp.int32)
        else:
            if not isinstance(predictor_state, PolarizationPredictorState):
                raise TypeError("predictor_state must be PolarizationPredictorState.")
            history = jnp.asarray(predictor_state.history, dtype=dtype)
            valid_count = jnp.asarray(predictor_state.valid_count).reshape(())
            if history.shape != (2, count, 3):
                raise ValueError("Owner predictor history must be (2, local, 3).")
            # A history of another preparation refuses the solve numerically.
            predictor_bound = jnp.all(
                predictor_state.prepared_identity.equals(prepared.prepared_identity)
            )
        previous = history[0]
        extrapolated = previous + solver.predictor.history_coefficient * (
            previous - history[1]
        )
        guess = jnp.where(
            valid_count >= 2,
            extrapolated,
            jnp.where(valid_count >= 1, previous, fallback),
        )
    else:
        if predictor_state is not None:
            raise ValueError(
                "TCG uses a deterministic direct-field or zero guess; a warm-start "
                "history would make its functional history dependent."
            )
        history = jnp.zeros((2, count, 3), dtype=dtype)
        valid_count = jnp.zeros((), dtype=jnp.int32)
        guess = fallback
    initial = jnp.where(polarizable[:, None], guess, 0.0)
    if pcg:
        # Only the PCG warm start (history or direct-field fallback) is detached,
        # as in the canonical ``_guess``: the implicit solve owns the derivative.
        # The fixed-order TCG functional differentiates through its start.
        initial = jax.lax.stop_gradient(initial)
    shape = (count, 3)
    matvec = matvec_for(bound)
    if pcg:
        primal_policy = DistributedKrylovPolicy(
            plan.maximum_iterations,
            relative_tolerance=0.0,
            absolute_tolerance=plan.tolerance,
        )
        derivative_policy = DistributedKrylovPolicy(
            plan.maximum_iterations,
            relative_tolerance=0.0,
            absolute_tolerance=plan.derivative_tolerance,
        )

        def run(
            policy: DistributedKrylovPolicy, start: Array | None
        ) -> Callable[..., tuple[Array, tuple[Array, ...]]]:
            def solve(
                action: Callable[[Array], Array], right: Array
            ) -> tuple[Array, tuple[Array, ...]]:
                linear = DistributedLinearOperator(
                    action,
                    action,
                    shape,
                    shape,
                    operator_id=f"{prepared.layout_id}:polarization-operator",
                )
                result = solve_distributed_pcg(
                    linear,
                    right,
                    pairing,
                    policy,
                    initial=start,
                    preconditioner=precondition,
                )
                return result.value, (
                    result.iterations,
                    result.residual_norm,
                    result.converged,
                    result.breakdown,
                )

            return solve

        induced, (iterations, _, converged_flag, breakdown_status) = (
            jax.lax.custom_linear_solve(
                matvec,
                rhs,
                run(primal_policy, initial),
                run(derivative_policy, None),
                symmetric=True,
                has_aux=True,
            )
        )
        # The native PCG reports a KrylovBreakdownStatus code, not a flag:
        # HAPPY is its certified true-residual stop and NONE an unfinished run;
        # every other code is a genuine breakdown (canonical native mapping).
        breakdown = (breakdown_status != int(KrylovBreakdownStatus.NONE)) & (
            breakdown_status != int(KrylovBreakdownStatus.HAPPY)
        )
        converged = converged_flag & ~breakdown
        iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
    else:
        value = initial
        residual = rhs - matvec(value)
        preconditioned = precondition(residual, jnp.zeros((), dtype=jnp.int32))
        direction = preconditioned
        rho = _inner(residual, preconditioned, axis_name, mode)
        breakdown = jnp.asarray(False)
        for _ in range(plan.tcg_order):
            product = matvec(direction)
            curvature = _inner(direction, product, axis_name, mode)
            ok = curvature > 0.0
            breakdown = breakdown | ~ok
            step = jnp.where(ok, rho / jnp.where(ok, curvature, 1.0), 0.0)
            value = value + step * direction
            residual = residual - step * product
            preconditioned = precondition(residual, jnp.zeros((), dtype=jnp.int32))
            rho_next = _inner(residual, preconditioned, axis_name, mode)
            ratio = jnp.where(rho != 0.0, rho_next / jnp.where(rho != 0.0, rho, 1.0), 0.0)
            direction = preconditioned + ratio * direction
            rho = rho_next
        induced = value
        converged = jnp.asarray(False)
        iterations = jnp.asarray(plan.tcg_order, dtype=jnp.int32)
    induced = jnp.where(polarizable[:, None], induced, 0.0)
    frozen_residual = jax.lax.stop_gradient(rhs - matvec(induced))
    residual_norm = jnp.sqrt(
        jnp.maximum(pairing.inner(frozen_residual, frozen_residual), 0.0)
    )
    rhs_norm = jnp.sqrt(
        jnp.maximum(
            pairing.inner(jax.lax.stop_gradient(rhs), jax.lax.stop_gradient(rhs)), 0.0
        )
    )
    relative = residual_norm / jnp.where(rhs_norm > 0.0, rhs_norm, 1.0)
    finite = (
        inputs_finite
        & _global_all(
            jnp.all(jnp.isfinite(induced)) & jnp.all(jnp.isfinite(rhs)), axis_name
        )
        & jnp.isfinite(residual_norm)
    )
    stationary = finite & (residual_norm <= plan.tolerance)
    if pcg:
        status = jnp.where(
            converged,
            int(LinearSolveStatus.SUCCESS),
            jnp.where(
                breakdown,
                int(LinearSolveStatus.BREAKDOWN),
                int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
            ),
        ).astype(jnp.int32)
        method_ok = converged & stationary
        force_valid = method_ok & (residual_norm <= plan.force_tolerance)
    else:
        breakdown = _global_all(breakdown, axis_name)
        status = jnp.where(
            breakdown,
            int(LinearSolveStatus.BREAKDOWN),
            int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
        ).astype(jnp.int32)
        method_ok = ~breakdown | stationary
        force_valid = method_ok
    stability = _owner_stability(
        prepared,
        relations,
        bound,
        jax.lax.stop_gradient(induced),
        jax.lax.stop_gradient(rhs),
        stationary,
        matvec_for(jax.lax.stop_gradient(bound)),
        axis_name,
    )
    successful = (
        finite
        & fields_ok
        & capacity_ok
        & method_ok
        & stability.successful
        & _global_all(predictor_bound, axis_name)
    )
    policy = plan.stability
    state = PolarizationState(
        induced_dipoles=induced,
        residual=residual_norm,
        relative_residual=relative,
        iterations=iterations,
        linear_status=status,
        converged=converged,
        stationary=stationary,
        stability_proven=stability.proven,
        stability_lower_bound=stability.lower_bound,
        force_valid=force_valid & successful,
        finite=finite,
        successful=successful,
        plan_id=solver.result_plan_id,
        solver_kind=plan.kind.value,
        functional=plan.functional,
        derivative_mode=plan.derivative_mode,
        stability_certificate=policy.certificate,
        stability_assumed=policy.assumed,
    )
    induced_set = _induced_sites(bound, induced)
    mutual, mutual_ok, mutual_fits = form(bound, induced_set, induced_set, "mutual")
    permanent, permanent_ok, permanent_fits = form(
        bound, induced_set, bound.permanent, "polarization"
    )
    self_term = 0.5 * contract("n,nd,nd->", inverse_alpha, induced, induced)
    share = self_term + 0.5 * mutual + permanent
    energy_ok = (
        successful
        & mutual_ok
        & permanent_ok
        & mutual_fits
        & permanent_fits
        & _global_all(jnp.isfinite(share), axis_name)
    )
    candidate = jnp.stack((jax.lax.stop_gradient(induced), history[0]))
    next_state = PolarizationPredictorState(
        jnp.where(energy_ok, candidate, history),
        jnp.where(energy_ok, jnp.minimum(valid_count + 1, 2), valid_count).reshape((1,)),
        IdentityWords(prepared.prepared_identity.words[None]),
    )
    return OwnerPolarizationResult(
        energy_share=jnp.where(energy_ok, share, jnp.nan),
        induced_dipoles=induced,
        state=state,
        predictor_state=next_state,
        stability=stability,
        permanent_field=rhs,
        direct_field=direct,
        initial_dipoles=initial,
        breakdown=breakdown,
        successful=energy_ok,
    )


def _owner_stability(
    prepared: PreparedOwnerPolarization,
    relations: OwnerPolarizationRelations,
    bound: _OwnerBound,
    induced: Array,
    rhs: Array,
    stationary: Array,
    matvec: Callable[[Array], Array],
    axis_name: str,
    /,
) -> PolarizationStabilityEvidence:
    """Declared stability certificate plus global indefiniteness detections."""
    solver = prepared.plan.solver
    policy = solver.plan.stability
    operator = solver.operator
    dtype = induced.dtype
    provider = JaxCollectiveProvider(axis_name)
    frozen = jax.lax.stop_gradient(bound)
    nan = jnp.asarray(jnp.nan, dtype=dtype)
    if policy.certificate == "assumed-positive-definite":
        proven, lower, allowance = jnp.asarray(False), nan, nan
    elif policy.certificate == "scaled-block-gershgorin":
        if operator.plan.route.periodic:
            raise ValueError(
                "The block-Gershgorin proof bounds nonperiodic operators only."
            )
        root = jnp.where(
            frozen.polarizable, jnp.sqrt(jnp.maximum(frozen.alpha, 0.0)), 0.0
        )
        rows, distinct = _route_row_norms(
            relations.real_space,
            relations.real_routes,
            frozen.positions,
            frozen.permanent,
            root,
            operator._direct_kernel(),
            operator.plan.minimum_distance,
        )
        radius = operator.plan.coulomb_constant * provider.maximum(
            jnp.max(jnp.where(frozen.polarizable, rows, 0.0))
        )
        lower = 1.0 - radius
        allowance = jnp.asarray(policy.margin, dtype=dtype)
        proven = (
            _global_all(distinct, axis_name) & jnp.isfinite(lower) & (lower > allowance)
        )
    else:
        raise ValueError(
            f"Stability certificate {policy.certificate!r} is not distributed."
        )
    curvature = provider.sum(jnp.vdot(induced, rhs))
    rhs_nonzero = ~_global_all(jnp.all(rhs == 0.0), axis_name)
    detected = stationary & rhs_nonzero & ~(curvature > 0.0)
    probe = solver.plan.curvature_probe
    ritz = nan
    probe_converged = jnp.asarray(False)
    if probe is not None:
        scale = jnp.sqrt(jnp.where(frozen.polarizable, frozen.alpha, 1.0))[:, None]

        def scaled(value: Array) -> Array:
            return scale * matvec(scale * value)

        # One Krylov vector per step without restarts: the probe's subspace
        # dimension bounds the resident basis (at least two for a tridiagonal).
        steps = max(2, min(probe.subspace_dimension, probe.maximum_steps))
        ritz, probe_converged = _lanczos_minimum(
            scaled, frozen.mask, steps, probe.seed, policy.margin, axis_name
        )
        ritz = ritz.astype(dtype)
        detected = detected | (jnp.isfinite(ritz) & (ritz < 0.0))
    admitted = proven if not policy.assumed else jnp.asarray(True)
    return PolarizationStabilityEvidence(
        certificate=policy.certificate,
        assumed=policy.assumed,
        proven=proven,
        lower_bound=lower,
        allowance=allowance,
        solution_curvature=curvature,
        probed=probe is not None,
        minimum_ritz_value=ritz,
        probe_converged=probe_converged,
        indefinite_detected=detected,
        successful=admitted & ~detected,
    )


__all__ = [
    "AtomisticReciprocalDistributionPlan",
    "AtomisticReciprocalKind",
    "AtomisticReciprocalSchedule",
    "AtomisticReciprocalStrategy",
    "OwnerMultipoleRelation",
    "OwnerMultipoleRoutes",
    "OwnerPolarizationPlan",
    "OwnerPolarizationRelations",
    "OwnerPolarizationResult",
    "OwnerPolarizationSites",
    "OwnerReciprocalActions",
    "OwnerReciprocalTerm",
    "PreparedOwnerPolarization",
    "PreparedReciprocalDistribution",
    "ReciprocalDistributionCapacityError",
    "ReciprocalDistributionEvidence",
    "ReciprocalDistributionReport",
    "ReciprocalSitePayload",
    "owner_multipole_bilinear",
    "owner_reciprocal_actions",
    "owner_reciprocal_energy",
    "prepare_owner_polarization",
    "prepare_reciprocal_distribution",
    "solve_owner_polarization",
]
