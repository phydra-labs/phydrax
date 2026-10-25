#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated native long-range electrostatics: point, multipole, polarization.

Every row executes one actual public native ``AtomisticPotentialProgram`` on a
deterministic neutral finite cluster, fully periodic cubic/triclinic cell, or
two-periodic slab, in canonical reduced units (``coulomb_constant = 1``):

* point charges: finite ``DirectCoulombPotential``, periodic
  ``DampedShiftedForceCoulombPotential`` (DSF), ``EwaldReferencePotential`` and
  ``ParticleMeshEwaldPotential`` under the tin-foil boundary and
  ``require-neutral`` convention, plus the admitted ``slab-yeh-berkowitz``
  boundary of both reciprocal methods;
* degree-two point/Gaussian ``MultipoleElectrostaticPotential`` on the finite,
  explicit-mode Ewald and particle-mesh routes (raw traceful Cartesian
  second moments, site-centered origin, intrasite excluded);
* ``InducedDipolePolarizationPotential`` (converged PCG or fixed-order TCG,
  undamped point or Gaussian sites) on the same three routes; the Hamiltonian
  is the induced-dipole functional only (no permanent-permanent energy).

Periodic multipole/polarization rows record their real-space route as an
explicit case axis: ``explicit-image-reference`` (bounded-workspace explicit
image sum, quadratic in atoms) or ``neighborhood`` (the term-owned
``IMAGE_PAIR_SUM`` requirement served by a declared cell-list image source).
Periodic polarization rows also record their stability certificate axis:
``dense-spectrum`` (bounded dense eigensolve, cost included in every
workload) or the explicitly declared ``assumed-positive-definite`` policy.

``learned-charge-{pme,ewald}`` rows connect a native AIMNet2 model (random
source-layout fixture of ``tests/_support/atomistic_model_families.py``) to the
periodic Coulomb term through ``LearnedGraphPotentialTerm`` +
``AtomisticFieldReference`` + ``charge_field`` in one program on a sheared
liquid-density water box in eV/angstrom units.  The Coulomb term declares the
``uniform-background`` convention because AIMNet2 equilibrates the requested
total charge only up to its published ``CHARGE_RESIDUAL``.

Each row separately records host preparation (time and traced host allocation
peak) of the system and of the prepared program with its neighborhood bundle
and term-owned reciprocal binding, then per workload the lowering,
compilation, first and warm synchronized executions, XLA compiler
argument/output/temporary/generated-code bytes, the sampled RSS/device peak of
the warm phase and logical retained bytes:

* ``neighborhood-initialize``: the bundle state at the evaluation point;
* ``energy``: scalar program energy;
* ``efs``: energy, conservative forces and (periodic) tension-positive stress;
  learned rows add the connected-composition controls (model-alone and
  held-charge programs, nonzero charge-response force/stress, independent
  uniform-background Ewald of the predicted charges, native central
  differences of the connected energy);
* ``hessian-vector``: forward-over-reverse force tangent ``-H v`` with a
  native central-difference consistency check (labeled non-independent);
  learned rows add the held-charge tangent chain-rule control;
* ``coefficient-refresh``: revised charges (system data), moments or
  polarizabilities re-prepared on the host (learned rows: revised model
  parameters and, separately, a revised total-charge condition); the existing
  executables are reused only when the static PyTree layout and dynamic avals
  are provably equal, otherwise the required recompilation is measured;
* ``cell-refresh``: an affine cell deformation, pure bundle refresh and
  evaluation through the same executable at the deformed cell;
* ``md-step``: canonical ``AtomisticDynamicsPlan`` velocity-Verlet NVE steps
  of the row's prepared program (every family).  The pure electrostatic
  Hamiltonian has no short-range repulsion; rows are cost measurements of
  complete accepted steps with their solver/capacity/rollback evidence, not
  physical trajectories;
* ``efs-workset`` (opt-in): full-program E/F/S of batch 1/2/... items through
  the public ``ExecutionWorksetPlan`` and
  ``evaluate_execution_worksets_filter_vmap`` under one shared preparation,
  with per-item positions, current cells and neighborhood states on a declared
  item ``LaneLayout``; equal-cell and distinct admitted current-cell items are
  compared with standalone evaluations, with semantic order, padding and
  per-item status;
* ``parameter-gradient`` (opt-in, learned rows): model-parameter gradients of
  an E/F/S loss through the connected program and of the Coulomb term energy
  alone (the charge chain rule into training).

Accuracy evidence is kept separate from cost.  Independent host references are
formed only where the definition is matched: an independently converged
three-dimensional Ewald sum under the same tin-foil/neutral (or uniform
background) convention, converged jointly in energy, forces and analytic
strain derivative (with the same Yeh-Berkowitz embedding correction for
slabs), the same-definition DSF and finite direct pair sums, JAX-autodiff
definitional finite point/Gaussian multipole energies, a dense finite
point-polarization solve, and the independent host multipole/polarization
references of ``tests/_support/atomistic_multipole_reference.py`` for finite
Gaussian polarization and every periodic point/Gaussian multipole and
polarization row.  Periodic multipole and polarization rows additionally keep
the tighter native direct-k comparison, labeled non-independent.  TCG rows
compare a fixed-order functional with a stationary reference: the gap
includes the declared truncation.  Nothing here is a mesh-interpolation
certificate, chemical accuracy or a statement about other hardware.  Refused,
failed and out-of-memory rows and workloads are retained with their exception
type and message.

``main`` captures the canonical benchmark identity first and fails closed when
its source-build prerequisites are absent, unless ``--unattested-probe`` is
given; that record is then explicitly marked unattested with the exact capture
error.  `long_range_cases` and `measure_case` are the reusable entry points of
the same measured kernels.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import gc
import hashlib
import itertools
import math
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from operator import methodcaller
from pathlib import Path
from typing import Any, get_args, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jax import Array
from scipy.special import erfc

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    fitted_scaling_exponent,
    host_allocation_peak,
    process_peak_rss_bytes,
)
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax._identity import IdentityWords
from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.atomistic import (
    AtomisticCellListImageSearch,
    AtomisticCompletePairSource,
    AtomisticConditioning,
    AtomisticConditionLevel,
    AtomisticDynamicsPlan,
    AtomisticDynamicsState,
    AtomisticFieldReference,
    AtomisticInitialStateEvaluation,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodBundleState,
    AtomisticNeighborhoodGeometry,
    AtomisticNeighborhoodSource,
    AtomisticObservable,
    AtomisticObservableRequest,
    AtomisticPairCounting,
    AtomisticPhaseSpaceMeasurePlan,
    AtomisticPotentialEvaluation,
    AtomisticPotentialProgram,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
    AtomisticStepEvaluation,
    AtomisticSystemPlan,
    AtomisticThermodynamicStatePlan,
    AtomisticUnitSystem,
    bind_atomistic_bundle_geometry,
    CartesianMultipoles,
    DampedShiftedForceCoulombPotential,
    DirectCoulombPotential,
    estimate_ewald_splitting,
    estimate_pme_splitting,
    EwaldMultipoleRoute,
    EwaldReferencePotential,
    fft_friendly_mesh_size,
    FiniteMultipoleRoute,
    GaussianMultipoleWidths,
    InducedDipolePolarizationPotential,
    LearnedGraphPotentialTerm,
    MultipoleElectrostaticPotential,
    MultipoleElectrostaticsDefinition,
    MultipoleElectrostaticsPlan,
    MultipoleRealSpace,
    MultipoleSiteKind,
    ParticleMeshEwaldPotential,
    ParticleMeshMultipoleRoute,
    PermanentMultipoleSiteData,
    PolarizationPlan,
    PolarizationSiteKind,
    PolarizationStabilityCertificate,
    PolarizationStabilityPolicy,
    PreparedAtomisticDynamics,
    PreparedAtomisticPotentialProgram,
    PreparedAtomisticSystem,
    PreparedThermodynamicStateTable,
    ReciprocalBudget,
    ReciprocalMeshPlan,
    ReciprocalModeSetPlan,
    ReciprocalResourceEvidence,
    VelocityVerletPlan,
)
from phydrax.discretization import (
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    PeriodicCell,
)
from phydrax.execution import (
    evaluate_execution_worksets_filter_vmap,
    ExecutionWorksetPlan,
    PhaseMemorySampler,
    PoolExecutionSignature,
    PreparedExecutionWorksets,
)
from phydrax.typing import parse
from tests._support.atomistic_model_families import (
    aimnet2_model,
    periodic_execution,
    UNITS as WATER_UNITS,
    water_box,
    WATER_MASSES,
    WATER_NUMBERS,
    WATER_VOLUME,
)
from tests._support.atomistic_multipole_reference import (
    multipole_reference,
    polarization_reference,
)


# ------------------------------------------------------------------ scope

POINT_METHODS: tuple[str, ...] = (
    "direct",
    "dsf",
    "ewald",
    "pme",
    "ewald-slab",
    "pme-slab",
)
_SITES = ("point", "gaussian")
_ROUTES = ("finite", "ewald", "pme")
MULTIPOLE_METHODS: tuple[str, ...] = tuple(
    f"multipole-{site}-{route}" for site in _SITES for route in _ROUTES
)
POLARIZATION_METHODS: tuple[str, ...] = tuple(
    f"polarization-{site}-{solver}-{route}"
    for site in _SITES
    for solver in ("pcg", "tcg")
    for route in _ROUTES
)
LEARNED_METHODS: tuple[str, ...] = ("learned-charge-pme", "learned-charge-ewald")
METHODS: tuple[str, ...] = (
    POINT_METHODS + MULTIPOLE_METHODS + POLARIZATION_METHODS + LEARNED_METHODS
)
DEFAULT_WORKLOADS: tuple[str, ...] = (
    "energy",
    "efs",
    "hessian-vector",
    "coefficient-refresh",
    "cell-refresh",
    "md-step",
)
WORKLOADS: tuple[str, ...] = DEFAULT_WORKLOADS + ("efs-workset", "parameter-gradient")
REAL_SPACE_ROUTES: tuple[MultipoleRealSpace, ...] = get_args(MultipoleRealSpace)
# Periodic polarization admits these certificates; Gershgorin is nonperiodic only.
STABILITY_POLICIES: tuple[PolarizationStabilityCertificate, ...] = (
    "dense-spectrum",
    "assumed-positive-definite",
)
WORKSET_CELLS: tuple[str, ...] = ("equal", "distinct")
_SHAPES: dict[str, np.ndarray] = {
    "cubic": np.eye(3),
    "triclinic": np.asarray([[1.0, 0.0, 0.0], [0.18, 1.0, 0.0], [0.07, -0.11, 1.0]]),
}
# Deterministic symmetric deformation direction of the cell-refresh workload.
_STRAIN_DIRECTION = np.asarray(
    [[1.0, 0.3, -0.2], [0.3, -0.5, 0.1], [-0.2, 0.1, 0.4]], dtype=np.float64
)
_SAMPLE_INTERVAL = 0.005
_SQRT_PI = math.sqrt(math.pi)
_SLAB_NORMAL = 2
_OWNERS = (
    "phydrax/atomistic/_electrostatics.py",
    "phydrax/atomistic/_reciprocal_mesh.py",
    "phydrax/atomistic/_multipole_basis.py",
    "phydrax/atomistic/_multipole_electrostatics.py",
    "phydrax/atomistic/_polarization.py",
    "phydrax/atomistic/_potential_program.py",
    "phydrax/atomistic/_neighborhood_bundle.py",
    "phydrax/atomistic/_cell_dynamics.py",
    "phydrax/atomistic/_dynamics.py",
    "phydrax/atomistic/_system.py",
    "phydrax/_execution_workset.py",
    "phydrax/nn/atomistic/_aimnet2.py",
    "tests/_support/atomistic_model_families.py",
    "tests/_support/atomistic_multipole_reference.py",
)
QUALIFICATION_BOUNDARIES: tuple[str, ...] = (
    "Batching is measured as full-program E/F/S worksets over items that share one "
    "prepared program (one admitted preparation, per-item positions, current cells "
    "and neighborhood states). Programs prepared at different reference cells are "
    "separate preparations; their layout comparison is recorded, never stacked.",
    "Predicted-charge rows use a random source-layout AIMNet2 fixture: they are "
    "connected chain-rule cost/consistency evidence, not trained accuracy, "
    "pretrained-weight or source-provider equivalence. Predicted site dipoles and "
    "second moments are qualified by the typed field-dependency integration tests, "
    "not measured by this driver.",
    "TCG rows are compared with a stationary reference; that gap is the declared "
    "fixed-order approximation, not an algorithmic error.",
    "Mesh rows report errors against independent converged references; they are "
    "not mesh-interpolation certificates. Measurements describe the recorded "
    "devices only.",
    "Distributed reciprocal execution (P11) and host-sync/memory-pattern audits "
    "(tools/audit_host_sync.py, tools/audit_memory_patterns.py) are separate "
    "campaigns and are not run by this driver.",
)


def _family(method: str, /) -> str:
    if method in POINT_METHODS:
        return "point"
    if method in LEARNED_METHODS:
        return "learned"
    return "multipole" if method.startswith("multipole-") else "polarization"


def _route(method: str, /) -> str:
    """Geometry/route of a method: finite, ewald, pme or dsf (and slab forms)."""
    if method in POINT_METHODS:
        return {"direct": "finite", "ewald-slab": "ewald", "pme-slab": "pme"}.get(
            method, method
        )
    return method.rsplit("-", 1)[1]


def _geometry_kind(method: str, /) -> str:
    if _route(method) == "finite":
        return "finite"
    return "slab" if method.endswith("-slab") else "periodic"


# Learned rows use the liquid-water box of the shared model-family fixture.
_WATER_DENSITY = 3.0 / WATER_VOLUME
_WATER_SHAPE = "sheared-water"


@dataclass(frozen=True, slots=True)
class LongRangeCase:
    """One method at one controlling scale; axes a method does not own are None.

    ``real_space`` is the recorded real-space route of periodic multipole and
    polarization rows; ``stability`` the stability certificate of periodic
    polarization rows.
    """

    method: str
    atoms: int
    density: float
    cell_shape: str
    capacity_factor: float | None
    reciprocal_extent: int | None
    mode_chunk: int | None
    grid: int | None
    spline_degree: int | None
    channel_chunk: int | None
    row_block: int | None
    real_space: MultipoleRealSpace | None = None
    stability: PolarizationStabilityCertificate | None = None

    @property
    def family(self) -> str:
        return _family(self.method)

    @property
    def route(self) -> str:
        return _route(self.method)

    @property
    def geometry(self) -> str:
        return _geometry_kind(self.method)

    @property
    def periodic(self) -> bool:
        return self.geometry != "finite"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "family": self.family,
            "route": self.route,
            "geometry": self.geometry,
        }


def long_range_cases(
    methods: Sequence[str],
    atoms: Sequence[int],
    densities: Sequence[float],
    cell_shapes: Sequence[str],
    *,
    capacity_factors: Sequence[float],
    reciprocal_extents: Sequence[int],
    mode_chunks: Sequence[int],
    grids: Sequence[int],
    spline_degrees: Sequence[int],
    channel_chunks: Sequence[int],
    row_blocks: Sequence[int],
    maximum_rows: int,
    real_spaces: Sequence[str] = ("explicit-image-reference",),
    stabilities: Sequence[str] = ("dense-spectrum",),
    learned_molecules: Sequence[int] = (8,),
) -> tuple[LongRangeCase, ...]:
    """Per-method sweep crossing only the axes each method owns, bounded up front.

    Learned rows take their atom count from ``learned_molecules`` (three atoms
    per water) and the fixture's liquid density and sheared cell instead of the
    reduced-unit ``atoms``/``densities``/``cell_shapes`` axes.
    """
    selected_routes: tuple[MultipoleRealSpace, ...] = tuple(
        parse(value, MultipoleRealSpace, "real_space") for value in real_spaces
    )
    selected_policies: tuple[PolarizationStabilityCertificate, ...] = tuple(
        parse(value, PolarizationStabilityCertificate, "stability")
        for value in stabilities
    )
    if any(value not in STABILITY_POLICIES for value in selected_policies):
        raise ValueError(
            f"Periodic polarization stability must be one of {STABILITY_POLICIES!r}."
        )
    rows: list[LongRangeCase] = []
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}.")
        family, route = _family(method), _route(method)
        moment_periodic = family in ("multipole", "polarization") and route != "finite"
        routes: tuple[MultipoleRealSpace | None, ...] = (
            selected_routes if moment_periodic else (None,)
        )
        policies: tuple[PolarizationStabilityCertificate | None, ...] = (
            selected_policies
            if family == "polarization" and route != "finite"
            else (None,)
        )
        blocks = (None,) if family in ("point", "learned") else tuple(row_blocks)
        if route == "ewald":
            reciprocal = [
                (extent, chunk, None, None, None)
                for extent, chunk in itertools.product(reciprocal_extents, mode_chunks)
            ]
        elif route == "pme":
            reciprocal = [
                (None, None, grid, degree, chunk)
                for grid, degree, chunk in itertools.product(
                    grids,
                    spline_degrees,
                    (None,) if family in ("point", "learned") else tuple(channel_chunks),
                )
            ]
        else:
            reciprocal = [(None, None, None, None, None)]
        if family == "learned":
            scales = [
                (3 * int(count), _WATER_DENSITY, _WATER_SHAPE)
                for count in learned_molecules
            ]
        else:
            scales = [
                (int(count), float(density), shape)
                for count, density, shape in itertools.product(
                    atoms, densities, cell_shapes
                )
            ]
        # Axis order of the original point/multipole sweeps is preserved.
        for (count, density, shape), real_space in itertools.product(scales, routes):
            searched = (
                (family == "point" and route != "finite")
                or family == "learned"
                or real_space == "neighborhood"
            )
            capacity = tuple(capacity_factors) if searched else (None,)
            for factor, stability, block, axes in itertools.product(
                capacity, policies, blocks, reciprocal
            ):
                rows.append(
                    LongRangeCase(
                        method,
                        count,
                        density,
                        shape,
                        factor,
                        axes[0],
                        axes[1],
                        axes[2],
                        axes[3],
                        axes[4],
                        block,
                        real_space,
                        stability,
                    )
                )
                if len(rows) > maximum_rows:
                    raise ValueError(
                        f"The requested sweep exceeds --max-rows {maximum_rows}; narrow "
                        "the axes or raise the bound explicitly."
                    )
    return tuple(rows)


# --------------------------------------------------------------- geometry


@dataclass(frozen=True)
class Geometry:
    """Deterministic configuration, coefficients and their revisions.

    Reduced-unit rows carry neutral alternating ions with their moment and
    polarizability data. Learned rows carry a water box: ``atomic_numbers`` and
    ``masses`` are set, ``charges`` are the (zero) system charges the connected
    Coulomb term never reads, and the moment arrays are empty.
    """

    kind: str
    positions: np.ndarray
    cell: np.ndarray | None
    periodic_axes: tuple[bool, bool, bool] | None
    charges: np.ndarray
    dipoles: np.ndarray
    quadrupoles: np.ndarray
    polarizabilities: np.ndarray
    site_classes: np.ndarray
    class_widths: np.ndarray
    revised_charges: np.ndarray
    revised_dipoles: np.ndarray
    revised_quadrupoles: np.ndarray
    revised_polarizabilities: np.ndarray
    strained_positions: np.ndarray
    strained_cell: np.ndarray | None
    occupied_density: float
    geometry_id: str
    atomic_numbers: np.ndarray | None = None
    masses: np.ndarray | None = None

    @property
    def atoms(self) -> int:
        return int(self.positions.shape[0])

    def coefficients(self, revised: bool, /) -> tuple[np.ndarray, ...]:
        if revised:
            return (
                self.revised_charges,
                self.revised_dipoles,
                self.revised_quadrupoles,
                self.revised_polarizabilities,
            )
        return self.charges, self.dipoles, self.quadrupoles, self.polarizabilities

    def point(self, strained: bool, /) -> tuple[np.ndarray, np.ndarray | None]:
        if strained:
            return self.strained_positions, self.strained_cell
        return self.positions, self.cell

    def widths(self) -> np.ndarray:
        return self.class_widths[self.site_classes]

    def summary(self) -> dict[str, Any]:
        positions = self.positions
        separation = positions[:, None, :] - positions[None, :, :]
        if self.cell is not None:
            inverse = np.linalg.inv(self.cell)
            fractional = separation @ inverse
            axes = np.asarray(self.periodic_axes, dtype=np.bool_)
            fractional = np.where(axes, fractional - np.rint(fractional), fractional)
            separation = fractional @ self.cell
        distance = np.linalg.norm(separation, axis=-1)
        distance[np.eye(self.atoms, dtype=np.bool_)] = np.inf
        record: dict[str, Any] = {
            "geometry_id": self.geometry_id,
            "kind": self.kind,
            "atoms": self.atoms,
            "occupied_density": self.occupied_density,
            "minimum_image_separation": float(np.min(distance)),
            "total_charge": float(np.sum(self.charges)),
            "revised_total_charge": float(np.sum(self.revised_charges)),
            "periodic_axes": None
            if self.periodic_axes is None
            else list(self.periodic_axes),
        }
        if self.atomic_numbers is not None:
            record["species"] = "water (O, H, H)"
            record["charge_source"] = (
                "system charges are zero; the connected Coulomb term reads the "
                "predicted AIMNet2 charge field"
            )
        if self.cell is not None:
            record["cell"] = self.cell.tolist()
            record["volume"] = float(abs(np.linalg.det(self.cell)))
            record["strained_cell"] = (
                None if self.strained_cell is None else self.strained_cell.tolist()
            )
        return record


def build_geometry(
    method: str,
    atoms: int,
    density: float,
    cell_shape: str,
    *,
    charge: float,
    jitter: float,
    slab_vacuum: float,
    strain: float,
    gaussian_widths: Sequence[float],
    polarizabilities: Sequence[float],
    seed: int,
) -> Geometry:
    """Jittered lattice ions of alternating sign; exact neutrality by construction."""
    if atoms < 2 or atoms % 2:
        raise ValueError("Neutral alternating ions need an even atom count >= 2.")
    if density <= 0.0 or charge <= 0.0 or not 0.0 <= jitter < 0.5:
        raise ValueError("density and charge must be positive; jitter in [0, 0.5).")
    kind = _geometry_kind(method)
    shape = _SHAPES[cell_shape]
    occupied = np.cbrt(atoms / density / abs(np.linalg.det(shape))) * shape
    rng = np.random.default_rng(seed)
    side = math.ceil(round(atoms ** (1.0 / 3.0), 12))
    side = side + 1 if side**3 < atoms else side
    sites = np.stack(
        np.meshgrid(*(np.arange(side),) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    chosen = np.sort(rng.permutation(len(sites))[:atoms])
    fractional = (sites[chosen] + 0.5 + rng.uniform(-jitter, jitter, (atoms, 3))) / side
    cell: np.ndarray | None
    if kind == "slab":
        if slab_vacuum <= 1.0:
            raise ValueError("slab_vacuum must exceed one.")
        cell = occupied.copy()
        cell[_SLAB_NORMAL] *= slab_vacuum
        fractional[:, _SLAB_NORMAL] /= slab_vacuum
        axes: tuple[bool, bool, bool] | None = (True, True, False)
    elif kind == "periodic":
        cell, axes = occupied.copy(), (True, True, True)
    else:
        cell, axes = None, None
    positions = fractional @ (occupied if cell is None else cell)
    signs = np.where(np.arange(atoms) % 2 == 0, 1.0, -1.0)
    charges = charge * signs
    classes = (signs < 0.0).astype(np.int64)
    dipoles = rng.uniform(-0.1, 0.1, (atoms, 3))
    raw = rng.uniform(-0.05, 0.05, (atoms, 3, 3))
    quadrupoles = raw + np.swapaxes(raw, 1, 2)
    alpha = np.asarray(polarizabilities, dtype=np.float64)[classes]
    revision = np.random.default_rng(seed + 1)
    revised_charges = charges * (1.0 + 0.05 * revision.uniform(-1.0, 1.0, atoms))
    revised_charges -= np.mean(revised_charges)
    revised_dipoles = dipoles + 0.02 * revision.uniform(-1.0, 1.0, (atoms, 3))
    delta = 0.01 * revision.uniform(-1.0, 1.0, (atoms, 3, 3))
    revised_quadrupoles = quadrupoles + delta + np.swapaxes(delta, 1, 2)
    revised_alpha = alpha * (1.0 + 0.05 * revision.uniform(-1.0, 1.0, atoms))
    deformation = np.eye(3) + strain * _STRAIN_DIRECTION
    strained_positions = positions @ deformation.T
    strained_cell = None if cell is None else cell @ deformation.T
    widths = np.asarray(gaussian_widths, dtype=np.float64)
    digest = hashlib.sha256()
    for value in (
        positions,
        np.zeros(0) if cell is None else cell,
        charges,
        dipoles,
        quadrupoles,
        alpha,
        widths,
        revised_charges,
        revised_dipoles,
        revised_quadrupoles,
        revised_alpha,
        strained_positions,
    ):
        digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    digest.update(repr((kind, axes)).encode())
    return Geometry(
        kind=kind,
        positions=positions,
        cell=cell,
        periodic_axes=axes,
        charges=charges,
        dipoles=dipoles,
        quadrupoles=quadrupoles,
        polarizabilities=alpha,
        site_classes=classes,
        class_widths=widths,
        revised_charges=revised_charges,
        revised_dipoles=revised_dipoles,
        revised_quadrupoles=revised_quadrupoles,
        revised_polarizabilities=revised_alpha,
        strained_positions=strained_positions,
        strained_cell=strained_cell,
        occupied_density=float(density),
        geometry_id=digest.hexdigest(),
    )


def build_water_geometry(atoms: int, *, strain: float, seed: int) -> Geometry:
    """Sheared liquid-density water box of the shared model-family fixture."""
    if atoms < 3 or atoms % 3:
        raise ValueError("Learned rows need three atoms per water molecule.")
    cell, positions = water_box(atoms // 3, seed=seed)
    count = positions.shape[0]
    numbers = np.tile(np.asarray(WATER_NUMBERS, dtype=np.int64), count // 3)
    masses = np.tile(np.asarray(WATER_MASSES, dtype=np.float64), count // 3)
    deformation = np.eye(3) + strain * _STRAIN_DIRECTION
    empty = np.zeros((0,))
    zeros = np.zeros((count,))
    digest = hashlib.sha256()
    for value in (positions, cell, numbers, masses):
        digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    digest.update(repr(("learned-water", strain, seed)).encode())
    return Geometry(
        kind="periodic",
        positions=positions,
        cell=cell,
        periodic_axes=(True, True, True),
        charges=zeros,
        dipoles=empty,
        quadrupoles=empty,
        polarizabilities=empty,
        site_classes=np.zeros((count,), dtype=np.int64),
        class_widths=empty,
        revised_charges=zeros,
        revised_dipoles=empty,
        revised_quadrupoles=empty,
        revised_polarizabilities=empty,
        strained_positions=positions @ deformation.T,
        strained_cell=cell @ deformation.T,
        occupied_density=count / abs(float(np.linalg.det(cell))),
        geometry_id=digest.hexdigest(),
        atomic_numbers=numbers,
        masses=masses,
    )


# -------------------------------------------------- independent references


def _translations(extent: Sequence[int], /) -> np.ndarray:
    axes = [np.arange(-int(value), int(value) + 1) for value in extent]
    return (
        np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        .reshape(-1, 3)
        .astype(np.float64)
    )


def _heights(cell: np.ndarray, /) -> np.ndarray:
    volume = abs(np.linalg.det(cell))
    return np.asarray(
        [
            volume / np.linalg.norm(np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3]))
            for axis in range(3)
        ]
    )


Kernel = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def _coulomb_kernel(distance: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    return 1.0 / distance, -1.0 / distance**2


def _erfc_kernel(alpha: float, distance: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    screened = erfc(alpha * distance)
    gaussian = 2.0 * alpha / _SQRT_PI * np.exp(-((alpha * distance) ** 2))
    return screened / distance, -screened / distance**2 - gaussian / distance


def _dsf_kernel(
    alpha: float, cutoff: float, distance: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    value, derivative = _erfc_kernel(alpha, distance)
    shift = (
        math.erfc(alpha * cutoff) / cutoff**2
        + 2.0 * alpha / _SQRT_PI * math.exp(-((alpha * cutoff) ** 2)) / cutoff
    )
    return (
        value - math.erfc(alpha * cutoff) / cutoff + shift * (distance - cutoff),
        derivative + shift,
    )


def _pair_sum(
    positions: np.ndarray,
    charges: np.ndarray,
    cell: np.ndarray | None,
    extent: Sequence[int],
    kernel: Kernel,
    /,
    *,
    cutoff: float | None,
    forces: bool,
    strain: bool = False,
) -> tuple[float, np.ndarray, np.ndarray | None]:
    """``1/2 sum' q_i q_j phi(|x_i - x_j + n H|)`` over a symmetric image cube.

    With ``strain`` the analytic symmetric strain derivative
    ``1/2 sum' q_i q_j phi'(r) r_a r_b / r`` of the same sum is returned.
    """
    products = np.outer(charges, charges)
    shifts = np.zeros((1, 3)) if cell is None else _translations(extent) @ cell
    energy = 0.0
    force = np.zeros_like(positions)
    derivative = np.zeros((3, 3)) if strain else None
    for shift in shifts:
        separation = positions[:, None, :] - positions[None, :, :] + shift
        distance = np.sqrt(np.sum(separation * separation, axis=-1))
        valid = distance > 0.0
        if cutoff is not None:
            valid &= distance < cutoff
        safe = np.where(valid, distance, 1.0)
        value, slope = kernel(safe)
        energy += 0.5 * float(np.sum(np.where(valid, products * value, 0.0)))
        if forces or strain:
            weight = np.where(valid, products * slope / safe, 0.0)
            if forces:
                force -= np.einsum("ij,ijk->ik", weight, separation)
            if derivative is not None:
                derivative += 0.5 * np.einsum(
                    "ij,ija,ijb->ab", weight, separation, separation
                )
    return energy, force, derivative


def _ewald_reciprocal(
    positions: np.ndarray,
    charges: np.ndarray,
    cell: np.ndarray,
    alpha: float,
    extent: Sequence[int],
    /,
    *,
    forces: bool,
    strain: bool = False,
    chunk: int = 2048,
) -> tuple[float, np.ndarray, np.ndarray | None]:
    """Full-space ``(2 pi / V) sum_{m != 0} exp(-k^2/4a^2)/k^2 |S(k)|^2``.

    The analytic strain derivative uses ``V -> V det F`` and ``k -> F^-T k``
    with ``S(k)`` invariant: ``-delta E + (4 pi / V) sum w (1/(4a^2) + 1/k^2)
    k k^T |S|^2``.
    """
    volume = float(abs(np.linalg.det(cell)))
    basis = 2.0 * np.pi * np.linalg.inv(cell).T
    modes = _translations(extent)
    modes = modes[np.any(modes != 0.0, axis=1)]
    energy = 0.0
    force = np.zeros_like(positions)
    derivative = np.zeros((3, 3)) if strain else None
    for start in range(0, len(modes), chunk):
        wave = modes[start : start + chunk] @ basis
        squared = np.sum(wave * wave, axis=1)
        weight = np.exp(-squared / (4.0 * alpha**2)) / squared
        phase = positions @ wave.T
        cosine, sine = np.cos(phase), np.sin(phase)
        real, imaginary = charges @ cosine, charges @ sine
        structure = real**2 + imaginary**2
        energy += 2.0 * np.pi / volume * float(np.sum(weight * structure))
        if forces:
            factor = (sine * real - cosine * imaginary) * weight
            force += 4.0 * np.pi / volume * charges[:, None] * (factor @ wave)
        if derivative is not None:
            scale = structure * weight * (1.0 / (4.0 * alpha**2) + 1.0 / squared)
            derivative += (
                4.0 * np.pi / volume * np.einsum("m,ma,mb->ab", scale, wave, wave)
            )
    if derivative is not None:
        derivative -= energy * np.eye(3)
    return energy, force, derivative


def _slab_correction(
    positions: np.ndarray, charges: np.ndarray, cell: np.ndarray, /
) -> tuple[float, np.ndarray, np.ndarray]:
    """Neutral Yeh-Berkowitz ``2 pi M_z^2 / V`` with ``z = s_c V / |a x b|``.

    ``z = x . n`` with the unit slab normal ``n``; under ``F = I + e`` the
    moment scales as ``1 + n.e.n`` and the volume as ``1 + tr e``, so the
    strain derivative is ``E (2 n n^T - I)``.
    """
    volume = float(abs(np.linalg.det(cell)))
    inverse = np.linalg.inv(cell)
    first, second = (axis for axis in range(3) if axis != _SLAB_NORMAL)
    height = volume / float(np.linalg.norm(np.cross(cell[first], cell[second])))
    gradient = height * inverse[:, _SLAB_NORMAL]
    moment = float(np.sum(charges * (positions @ gradient)))
    energy = 2.0 * np.pi * moment**2 / volume
    force = -4.0 * np.pi / volume * moment * charges[:, None] * gradient[None, :]
    derivative = energy * (2.0 * np.outer(gradient, gradient) - np.eye(3))
    return energy, np.asarray(force), np.asarray(derivative)


def _strain_derivative(
    energy: Callable[[np.ndarray, np.ndarray], float],
    positions: np.ndarray,
    cell: np.ndarray,
    step: float,
    /,
) -> np.ndarray:
    """Central differences of ``E(x (I+e)^T, H (I+e)^T)`` over symmetric strains."""
    derivative = np.zeros((3, 3))
    for first in range(3):
        for second in range(first, 3):
            values = []
            for sign in (1.0, -1.0):
                deformation = np.eye(3)
                deformation[first, second] += sign * step
                if first != second:
                    deformation[second, first] += sign * step
                values.append(energy(positions @ deformation.T, cell @ deformation.T))
            scale = 2.0 * step if first == second else 4.0 * step
            derivative[first, second] = derivative[second, first] = (
                values[0] - values[1]
            ) / scale
    return derivative


def _change(current: np.ndarray, previous: np.ndarray | None, /) -> float | None:
    """Max-abs change between refinement levels relative to ``max(1, max|x|)``."""
    if previous is None:
        return None
    scale = max(1.0, float(np.max(np.abs(current))))
    return float(np.max(np.abs(current - previous))) / scale


def _periodic_reference(
    positions: np.ndarray,
    charges: np.ndarray,
    cell: np.ndarray,
    constant: float,
    /,
    *,
    slab: bool,
    tolerance: float,
    maximum_level: int,
    step: float,
    background: bool = False,
) -> dict[str, Any]:
    """Shell-converged tin-foil Ewald of the full lattice, independent of the term.

    Each refinement level forms the energy, forces and the analytic strain
    derivative; the reference converges only when all three change by at most
    ``tolerance * max(1, scale)`` between consecutive levels.  ``background``
    declares the uniform neutralizing background ``-pi Q^2 / (2 a^2 V)``
    (strain derivative ``-E_bg I``); otherwise the cell must be neutral.  A
    central difference of the reference energy at the final level is kept as
    a consistency record of the analytic strain derivative.
    """
    total = float(np.sum(charges))
    if not background and abs(total) > 1.0e-10:
        raise ValueError("The converged Ewald reference is formed for neutral cells.")
    if background and slab:
        raise ValueError("The slab reference is formed for neutral cells only.")
    alpha = 3.5 / float(np.min(_heights(cell)))
    lengths = np.linalg.norm(cell, axis=1)

    def extents(level: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        cutoff = 2.0 * alpha * (3 + level)
        return (min(1 + level, 2),) * 3, tuple(
            math.ceil(cutoff * length / (2.0 * np.pi)) for length in lengths
        )

    def evaluate(
        x: np.ndarray, vectors: np.ndarray, level: int, derivatives: bool
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        real_extent, reciprocal_extent = extents(level)
        real, real_force, real_strain = _pair_sum(
            x,
            charges,
            vectors,
            real_extent,
            partial(_erfc_kernel, alpha),
            cutoff=None,
            forces=derivatives,
            strain=derivatives,
        )
        reciprocal, reciprocal_force, reciprocal_strain = _ewald_reciprocal(
            x,
            charges,
            vectors,
            alpha,
            reciprocal_extent,
            forces=derivatives,
            strain=derivatives,
        )
        energy: float = real + reciprocal - alpha / _SQRT_PI * float(np.sum(charges**2))
        force: np.ndarray = np.asarray(real_force + reciprocal_force)
        strain: np.ndarray | None = (
            None
            if real_strain is None or reciprocal_strain is None
            else np.asarray(real_strain + reciprocal_strain)
        )
        if background:
            volume = float(abs(np.linalg.det(vectors)))
            neutralizing = -np.pi * total**2 / (2.0 * alpha**2 * volume)
            energy += neutralizing
            if strain is not None:
                strain = strain - neutralizing * np.eye(3)
        if slab:
            correction, correction_force, correction_strain = _slab_correction(
                x, charges, vectors
            )
            energy += correction
            force = force + correction_force
            if strain is not None:
                strain = strain + correction_strain
        return (
            float(constant * energy),
            np.asarray(constant * force),
            None if strain is None else np.asarray(constant * strain),
        )

    history: list[dict[str, Any]] = []
    previous: tuple[float, np.ndarray, np.ndarray] | None = None
    converged = False
    level = 0
    energy, force, derivative = 0.0, np.zeros_like(positions), np.zeros((3, 3))
    for level in range(maximum_level + 1):
        energy, force, strain = evaluate(positions, cell, level, True)
        if strain is None:
            raise RuntimeError("The reference strain derivative was not formed.")
        derivative = strain
        real_extent, reciprocal_extent = extents(level)
        changes = (
            None
            if previous is None
            else (
                abs(energy - previous[0]) / max(1.0, abs(energy)),
                _change(force, previous[1]),
                _change(derivative, previous[2]),
            )
        )
        history.append(
            {
                "level": level,
                "real_extent": list(real_extent),
                "reciprocal_extent": list(reciprocal_extent),
                "energy": energy,
                "energy_change": None if changes is None else changes[0],
                "force_change": None if changes is None else changes[1],
                "strain_derivative_change": None if changes is None else changes[2],
            }
        )
        if changes is not None and all(
            value is not None and value <= tolerance for value in changes
        ):
            converged = True
            break
        previous = (energy, force, derivative)
    central = _strain_derivative(
        lambda x, vectors: evaluate(x, vectors, level, False)[0], positions, cell, step
    )
    return {
        "kind": "converged-ewald-slab-yeh-berkowitz" if slab else "converged-ewald",
        "independent": True,
        "convention": "tin-foil, "
        + ("uniform-background" if background else "require-neutral")
        + ", k=0 excluded"
        + ("; neutral Yeh-Berkowitz embedding correction" if slab else ""),
        "total_charge": total,
        "reference_alpha": alpha,
        "converged": converged,
        "convergence": "energy, forces and analytic strain derivative jointly; "
        "changes relative to max(1, max|value|) between consecutive levels",
        "tolerance": tolerance,
        "history": history,
        "energy": energy,
        "forces": force,
        "strain_derivative": derivative,
        "stress": derivative / abs(np.linalg.det(cell)),
        "stress_method": "analytic strain derivative of the reference sums",
        "strain_central_difference": {
            "step": step,
            "max_absolute_difference": float(np.max(np.abs(central - derivative))),
        },
    }


def _dsf_reference(
    positions: np.ndarray,
    charges: np.ndarray,
    cell: np.ndarray,
    constant: float,
    alpha: float,
    cutoff: float,
    /,
    *,
    step: float,
) -> dict[str, Any]:
    """Same-definition Fennell-Gezelter DSF over every image inside the cutoff."""
    kernel = partial(_dsf_kernel, alpha, cutoff)
    self_term = -float(np.sum(charges**2)) * (
        math.erfc(alpha * cutoff) / (2.0 * cutoff) + alpha / _SQRT_PI
    )

    def evaluate(
        x: np.ndarray, vectors: np.ndarray, forces: bool
    ) -> tuple[float, np.ndarray]:
        pair, force, _ = _pair_sum(
            x, charges, vectors, (1, 1, 1), kernel, cutoff=cutoff, forces=forces
        )
        return constant * (pair + self_term), constant * force

    energy, force = evaluate(positions, cell, True)
    derivative = _strain_derivative(
        lambda x, vectors: evaluate(x, vectors, False)[0], positions, cell, step
    )
    return {
        "kind": "same-definition-dsf",
        "independent": True,
        "convention": "pair sum over all images with r < cutoff plus self term",
        "energy": energy,
        "forces": force,
        "strain_derivative": derivative,
        "stress": derivative / abs(np.linalg.det(cell)),
        "stress_method": f"central finite difference of reference energy, step {step}",
    }


def _direct_reference(
    positions: np.ndarray, charges: np.ndarray, constant: float, /
) -> dict[str, Any]:
    energy, force, _ = _pair_sum(
        positions, charges, None, (0, 0, 0), _coulomb_kernel, cutoff=None, forces=True
    )
    return {
        "kind": "exact-finite-pair-sum",
        "independent": True,
        "energy": constant * energy,
        "forces": constant * force,
        "strain_derivative": None,
        "stress": None,
    }


def _site_kernel(smearing: Array | None, /) -> Callable[[Array], Array]:
    """Point ``1/r`` without a smearing; Gaussian ``erf(r/(sqrt(2) s))/r`` with one."""
    if smearing is None:

        def point(vector: Array) -> Array:
            return 1.0 / jnp.sqrt(jnp.sum(vector * vector))

        return point
    width = smearing

    def gaussian(vector: Array) -> Array:
        distance = jnp.sqrt(jnp.sum(vector * vector))
        return jsp.erf(distance / (jnp.sqrt(2.0) * width)) / distance

    return gaussian


def _source_operator(
    kernel: Callable[[Array], Array],
    charge: Array,
    dipole: Array,
    second: Array,
    /,
) -> Callable[[Array], Array]:
    """``(q - mu.grad + 1/2 Theta:grad grad) K`` of a source in ``d = x - x_j``."""

    def applied(vector: Array) -> Array:
        return (
            charge * kernel(vector)
            - jnp.dot(dipole, jax.grad(kernel)(vector))
            + 0.5 * jnp.sum(second * jax.hessian(kernel)(vector))
        )

    return applied


def _multipole_reference_energy(
    positions: Array,
    charges: Array,
    dipoles: Array,
    seconds: Array,
    widths: Array,
    /,
    *,
    gaussian: bool,
    constant: float,
) -> Array:
    """Definitional finite pair energy ``sum_{i<j} L_i L_j K(|x_i - x_j|)``."""
    left, right = np.triu_indices(positions.shape[0], 1)

    def pair(
        vector: Array,
        qi: Array,
        qj: Array,
        mi: Array,
        mj: Array,
        ti: Array,
        tj: Array,
        smearing: Array,
    ) -> Array:
        kernel = _site_kernel(smearing if gaussian else None)
        source = _source_operator(kernel, qj, mj, tj)
        return (
            qi * source(vector)
            + jnp.dot(mi, jax.grad(source)(vector))
            + 0.5 * jnp.sum(ti * jax.hessian(source)(vector))
        )

    smearing = jnp.sqrt(widths[left] ** 2 + widths[right] ** 2)
    values = jax.vmap(pair)(
        positions[left] - positions[right],
        charges[left],
        charges[right],
        dipoles[left],
        dipoles[right],
        seconds[left],
        seconds[right],
        smearing,
    )
    return constant * jnp.sum(values)


def _polarization_reference_energy(
    positions: Array,
    charges: Array,
    dipoles: Array,
    seconds: Array,
    polarizabilities: Array,
    /,
    *,
    constant: float,
) -> Array:
    """Undamped point stationary functional ``-1/2 mu.E`` of a dense solve."""
    count = positions.shape[0]
    receivers, sources = np.nonzero(~np.eye(count, dtype=np.bool_))
    kernel = _site_kernel(None)

    def field(vector: Array, charge: Array, dipole: Array, second: Array) -> Array:
        return -jax.grad(_source_operator(kernel, charge, dipole, second))(vector)

    separation = positions[receivers] - positions[sources]
    permanent = jax.vmap(field)(
        separation, charges[sources], dipoles[sources], seconds[sources]
    )
    electric = constant * jnp.zeros((count, 3)).at[receivers].add(permanent)
    tensors = jax.vmap(jax.hessian(kernel))(separation)
    operator = (
        jnp.zeros((count, 3, count, 3))
        .at[receivers, :, sources, :]
        .add(-constant * tensors)
    )
    matrix = operator.reshape(3 * count, 3 * count) + jnp.diag(
        jnp.repeat(1.0 / polarizabilities, 3)
    )
    induced = jnp.linalg.solve(matrix, electric.reshape(-1))
    return -0.5 * jnp.dot(induced, electric.reshape(-1))


def _autodiff_reference(
    kind: str,
    energy: Callable[..., Array],
    positions: np.ndarray,
    *values: np.ndarray,
) -> dict[str, Any]:
    value, gradient = jax.jit(jax.value_and_grad(energy))(
        jnp.asarray(positions), *(jnp.asarray(item) for item in values)
    )
    return {
        "kind": kind,
        "independent": True,
        "energy": float(value),
        "forces": -np.asarray(gradient),
        "strain_derivative": None,
        "stress": None,
    }


def _reference_kind(method: str, /) -> str | None:
    """Independent reference of a method's definition; learned rows have none here.

    Learned rows form their uniform-background Ewald reference from the
    runtime predicted charges inside the composition record instead.
    """
    if method in ("ewald", "pme"):
        return "ewald"
    if method in ("ewald-slab", "pme-slab"):
        return "ewald-slab"
    if method in ("direct", "dsf"):
        return method
    if method in LEARNED_METHODS:
        return None
    site = _site_text(method)
    periodic = not method.endswith("-finite")
    if method.startswith("multipole-"):
        return f"multipole-{site}" + ("-periodic" if periodic else "")
    return f"polarization-{site}" + ("-periodic" if periodic else "")


def independent_reference(
    kind: str,
    geometry: Geometry,
    args: argparse.Namespace,
    /,
    *,
    revised: bool,
    strained: bool,
) -> dict[str, Any]:
    """One independent reference of the matched definition, timed on the host."""
    charges, dipoles, seconds, alpha = geometry.coefficients(revised)
    positions, cell = geometry.point(strained)
    constant = AtomisticUnitSystem.reduced().coulomb_constant

    def form() -> dict[str, Any]:
        if kind in ("ewald", "ewald-slab"):
            if cell is None:
                raise ValueError("A periodic reference needs a cell.")
            return _periodic_reference(
                positions,
                charges,
                cell,
                constant,
                slab=kind == "ewald-slab",
                tolerance=args.reference_tolerance,
                maximum_level=args.reference_max_level,
                step=args.fd_step,
            )
        if kind == "dsf":
            if cell is None:
                raise ValueError("The DSF reference here is periodic.")
            return _dsf_reference(
                positions,
                charges,
                cell,
                constant,
                args.dsf_alpha,
                args.real_cutoff,
                step=args.fd_step,
            )
        if kind == "direct":
            return _direct_reference(positions, charges, constant)
        if kind in ("multipole-point", "multipole-gaussian"):
            return _autodiff_reference(
                f"autodiff-definitional-{kind}",
                partial(
                    _multipole_reference_energy,
                    gaussian=kind.endswith("gaussian"),
                    constant=constant,
                ),
                positions,
                charges,
                dipoles,
                seconds,
                geometry.widths(),
            )
        if kind == "polarization-point":
            return _autodiff_reference(
                "dense-undamped-point-polarization",
                partial(_polarization_reference_energy, constant=constant),
                positions,
                charges,
                dipoles,
                seconds,
                alpha,
            )
        if kind in (
            "multipole-point-periodic",
            "multipole-gaussian-periodic",
            "polarization-gaussian",
            "polarization-point-periodic",
            "polarization-gaussian-periodic",
        ):
            return _moment_reference(
                kind,
                geometry,
                args,
                positions=positions,
                cell=cell,
                coefficients=(charges, dipoles, seconds, alpha),
                constant=constant,
            )
        raise ValueError(f"No independent reference kind {kind!r}.")

    reference, seconds_taken = measure_host(form)
    reference["host_seconds"] = seconds_taken
    reference["coefficients"] = "revised" if revised else "base"
    reference["configuration"] = "strained" if strained else "reference"
    return reference


_MOMENT_REFERENCE_ARRAYS = (
    "forces",
    "strain_derivative",
    "induced_dipoles",
    "permanent_field",
)


def _moment_reference(
    kind: str,
    geometry: Geometry,
    args: argparse.Namespace,
    /,
    *,
    positions: np.ndarray,
    cell: np.ndarray | None,
    coefficients: tuple[np.ndarray, ...],
    constant: float,
) -> dict[str, Any]:
    """Independent host multipole/polarization reference of the shared helper.

    Sites are explicitly point (``widths=None``) or Gaussian (per-site sigma);
    periodic rows are tin-foil and neutral (``background=False``); permanent
    multipoles exclude co-centered self interactions, matching the native
    intrasite-excluded definition.  Bounds are explicit CLI resources; an
    exhausted bound raises and is recorded as a refusal, never a value.
    """
    charges, dipoles, seconds, alpha = coefficients
    vectors: np.ndarray | None = None
    if kind.endswith("-periodic"):
        if cell is None:
            raise ValueError("A periodic moment reference needs a cell.")
        vectors = cell
    widths = geometry.widths() if "-gaussian" in kind else None
    resources = {
        "widths": widths,
        "background": False,
        "coulomb_constant": constant,
        "tolerance": args.moment_reference_tolerance,
        "maximum_shells": args.moment_reference_max_shells,
        "mode_chunk": args.moment_reference_mode_chunk,
        "maximum_atoms": args.moment_reference_max_atoms,
    }
    if kind.startswith("multipole-"):
        reference = multipole_reference(
            positions,
            charges,
            dipoles,
            seconds,
            vectors,
            include_coincident_self=False,
            **resources,
        )
    else:
        reference = polarization_reference(
            positions, charges, dipoles, seconds, alpha, vectors, **resources
        )
    reference = dict(reference)
    reference["reference_kind"] = kind
    reference["requested_resources"] = {
        key: value for key, value in resources.items() if key != "widths"
    } | {"sites": "point" if widths is None else "gaussian"}
    derivative = reference.get("strain_derivative")
    if vectors is not None and derivative is not None:
        # The helper's full dE/de keeps lab-frame moments fixed (it need not be
        # symmetric); the native stress is sym(dE/de)/|det H|.
        full = np.asarray(derivative, dtype=np.float64)
        reference["helper_stress"] = reference.get("stress")
        reference["stress"] = 0.5 * (full + full.T) / abs(np.linalg.det(vectors))
        reference["stress_definition"] = "sym(strain_derivative)/|det H|"
        reference["strain_derivative_antisymmetric_max_abs"] = float(
            np.max(np.abs(0.5 * (full - full.T)))
        )
    return reference


# ------------------------------------------------------- native assembly


@dataclass(frozen=True)
class LearnedBinding:
    """Connected learned-charge composition parts of one learned row.

    ``alone`` is the model-only program of the same learned term (also the
    host producer-port probe); ``charge_field`` the typed atom-level CHARGES
    dependency the Coulomb term consumes.
    """

    term: LearnedGraphPotentialTerm
    charge_field: AtomisticFieldReference
    alone: PreparedAtomisticPotentialProgram
    graph_execution: Any
    model_seed: int
    total_charge: float

    @property
    def learned_index(self) -> int:
        return 0

    @property
    def coulomb_index(self) -> int:
        return 1


@dataclass(frozen=True)
class NativeBinding:
    """Prepared program, its system and the host phases that built them."""

    system: PreparedAtomisticSystem
    definition: AtomisticPotentialProgram
    program: PreparedAtomisticPotentialProgram
    phases: dict[str, Any]
    capacities: dict[str, Any]
    learned: LearnedBinding | None = None


def _host_phase(operation: Callable[[], Any], /) -> tuple[Any, dict[str, Any]]:
    """Host time and traced allocation peak (tracing overhead included)."""
    (value, peak), seconds = measure_host(lambda: host_allocation_peak(operation))
    return value, {"seconds": seconds, "host_allocation_peak_bytes": peak}


def _budget(args: argparse.Namespace, /) -> ReciprocalBudget:
    return ReciprocalBudget(
        maximum_grid_points=args.max_grid_points,
        maximum_modes=args.max_modes,
        maximum_chunk_workspace_bytes=args.max_chunk_workspace_bytes,
        maximum_retained_bytes=args.max_retained_bytes,
    )


def _axis_scale(
    case: LongRangeCase, args: argparse.Namespace, /
) -> tuple[float, float, float]:
    """Per-axis resolution factor: the slab normal axis is ``slab_vacuum`` longer."""
    normal = float(args.slab_vacuum) if case.geometry == "slab" else 1.0
    return (1.0, 1.0, normal)


def _extents(case: LongRangeCase, args: argparse.Namespace, /) -> tuple[int, int, int]:
    if case.reciprocal_extent is None:
        raise ValueError(f"{case.method} owns no reciprocal mode extent.")
    extent = case.reciprocal_extent
    first, second, normal = _axis_scale(case, args)
    return (
        math.ceil(extent * first),
        math.ceil(extent * second),
        math.ceil(extent * normal),
    )


def _grid(case: LongRangeCase, args: argparse.Namespace, /) -> tuple[int, int, int]:
    if case.grid is None:
        raise ValueError(f"{case.method} owns no reciprocal mesh.")
    grid = case.grid
    first, second, normal = _axis_scale(case, args)

    def axis(value: float) -> int:
        return grid if value == 1.0 else fft_friendly_mesh_size(grid * value)

    return axis(first), axis(second), axis(normal)


def _mode_chunk(case: LongRangeCase, /) -> int:
    if case.mode_chunk is None:
        raise ValueError(f"{case.method} owns no reciprocal mode chunk.")
    return case.mode_chunk


def _spline_degree(case: LongRangeCase, /) -> int:
    if case.spline_degree is None:
        raise ValueError(f"{case.method} owns no mesh spline degree.")
    return case.spline_degree


def _channel_chunk(case: LongRangeCase, /) -> int:
    if case.channel_chunk is None:
        raise ValueError(f"{case.method} owns no mesh channel chunk.")
    return case.channel_chunk


def _system(geometry: Geometry, charges: np.ndarray | None, /) -> PreparedAtomisticSystem:
    count = geometry.atoms
    cell = (
        None
        if geometry.cell is None
        else PeriodicCell(geometry.cell, periodic_axes=geometry.periodic_axes)
    )
    return AtomisticSystemPlan(
        np.arange(count, dtype=np.int64),
        np.ones(count, dtype=np.int64),
        np.ones(count, dtype=np.float64),
        AtomisticUnitSystem.reduced(),
        atom_type_ids=np.zeros(count, dtype=np.int64),
        charges=charges,
        cell=cell,
        name="long-range-benchmark",
    ).prepare()


def _point_term(case: LongRangeCase, args: argparse.Namespace, /) -> Any:
    boundary = "slab-yeh-berkowitz" if case.geometry == "slab" else "tin-foil"
    match case.route:
        case "finite":
            return DirectCoulombPotential()
        case "dsf":
            return DampedShiftedForceCoulombPotential(args.dsf_alpha, args.real_cutoff)
        case "ewald":
            return EwaldReferencePotential(
                args.alpha,
                args.real_cutoff,
                _extents(case, args),
                boundary=boundary,
                reciprocal_cutoff=args.reciprocal_cutoff,
                mode_chunk=_mode_chunk(case),
                budget=_budget(args),
            )
        case "pme":
            return ParticleMeshEwaldPotential(
                args.alpha,
                args.real_cutoff,
                _grid(case, args),
                spline_degree=_spline_degree(case),
                boundary=boundary,
                reciprocal_cutoff=args.reciprocal_cutoff,
                budget=_budget(args),
            )
    raise ValueError(f"Unknown point route {case.route!r}.")


def _multipole_route(
    case: LongRangeCase, args: argparse.Namespace, /, *, comparison: bool = False
) -> Any:
    block = 32 if case.row_block is None else case.row_block
    budget = _budget(args)
    # The recorded real-space route; the native comparison keeps the reference.
    real_space: MultipoleRealSpace = (
        "explicit-image-reference" if case.real_space is None else case.real_space
    )
    if comparison:
        extent = args.native_comparison_extent
        return EwaldMultipoleRoute(
            args.alpha,
            args.real_cutoff,
            ReciprocalModeSetPlan((extent, extent, extent), budget=budget),
            real_space="explicit-image-reference",
            reciprocal_cutoff=args.reciprocal_cutoff,
            row_block=block,
            maximum_image_count=args.maximum_image_count,
            maximum_workspace_bytes=args.multipole_workspace_bytes,
        )
    match case.route:
        case "finite":
            return FiniteMultipoleRoute(
                row_block=block, maximum_workspace_bytes=args.multipole_workspace_bytes
            )
        case "ewald":
            return EwaldMultipoleRoute(
                args.alpha,
                args.real_cutoff,
                ReciprocalModeSetPlan(
                    _extents(case, args), mode_chunk=_mode_chunk(case), budget=budget
                ),
                real_space=real_space,
                reciprocal_cutoff=args.reciprocal_cutoff,
                row_block=block,
                maximum_image_count=args.maximum_image_count,
                maximum_workspace_bytes=args.multipole_workspace_bytes,
            )
        case "pme":
            return ParticleMeshMultipoleRoute(
                args.alpha,
                args.real_cutoff,
                ReciprocalMeshPlan(
                    _grid(case, args),
                    spline_degree=_spline_degree(case),
                    channel_chunk=_channel_chunk(case),
                    budget=budget,
                ),
                real_space=real_space,
                reciprocal_cutoff=args.reciprocal_cutoff,
                row_block=block,
                maximum_image_count=args.maximum_image_count,
                maximum_workspace_bytes=args.multipole_workspace_bytes,
            )
    raise ValueError(f"Unknown multipole route {case.route!r}.")


def _site_text(method: str, /) -> str:
    """The site field of ``{multipole,polarization}-{site}-...`` method names."""
    return method.split("-")[1]


def _widths(geometry: Geometry, site: str, /) -> GaussianMultipoleWidths | None:
    if site != "gaussian":
        return None
    return GaussianMultipoleWidths(geometry.class_widths, geometry.site_classes)


def _multipole_term(
    case: LongRangeCase,
    geometry: Geometry,
    args: argparse.Namespace,
    /,
    *,
    revised: bool,
    comparison: bool,
) -> Any:
    charges, dipoles, seconds, _ = geometry.coefficients(revised)
    site = parse(_site_text(case.method), MultipoleSiteKind, "multipole site kind")
    plan = MultipoleElectrostaticsPlan(
        MultipoleElectrostaticsDefinition(
            site, 2, coulomb_constant=AtomisticUnitSystem.reduced().coulomb_constant
        ),
        _multipole_route(case, args, comparison=comparison),
    )
    moments = CartesianMultipoles(
        jnp.asarray(charges),
        jnp.asarray(dipoles),
        jnp.asarray(seconds),
        maximum_degree=2,
    )
    return MultipoleElectrostaticPotential(
        plan, moments=moments, widths=_widths(geometry, site)
    )


def _polarization_term(
    case: LongRangeCase,
    geometry: Geometry,
    args: argparse.Namespace,
    /,
    *,
    revised: bool,
    comparison: bool,
) -> Any:
    charges, dipoles, seconds, alpha = geometry.coefficients(revised)
    site = parse(_site_text(case.method), PolarizationSiteKind, "polarization site kind")
    solver = "pcg" if comparison else case.method.split("-")[2]
    # The native comparison keeps the bounded dense reference certificate.
    certificate: PolarizationStabilityCertificate = (
        "dense-spectrum" if comparison or case.stability is None else case.stability
    )
    plan = PolarizationPlan(
        coulomb_constant=AtomisticUnitSystem.reduced().coulomb_constant,
        maximum_iterations=args.polarization_iterations,
        tolerance=min(args.polarization_tolerance, 1.0e-12)
        if comparison
        else args.polarization_tolerance,
        solver_kind=solver,
        tcg_order=args.tcg_order,
        site_kind=site,
        damping="none",
        route=_multipole_route(case, args, comparison=comparison),
        stability=PolarizationStabilityPolicy(certificate) if case.periodic else None,
    )
    data = PermanentMultipoleSiteData(
        charges, dipoles, seconds, alpha, np.ones(geometry.atoms)
    )
    return InducedDipolePolarizationPotential(plan, data, widths=_widths(geometry, site))


def _image_capacities(
    case: LongRangeCase,
    geometry: Geometry,
    args: argparse.Namespace,
    /,
    *,
    cutoff: float,
) -> dict[str, int]:
    """Bounded search resources of one cutoff from the occupied density and factor.

    The image stencil covers ``cutoff + skin`` along every periodic axis (at
    least the 27-image nearest stencil).
    """
    if case.capacity_factor is None:
        raise ValueError(f"{case.method} owns no bounded image search.")
    count = geometry.atoms
    radius = cutoff + args.skin
    images = 27
    if geometry.cell is not None:
        periodic = np.asarray(geometry.periodic_axes, dtype=np.bool_)
        reach = np.where(periodic, np.ceil(radius / _heights(geometry.cell)), 0.0)
        images = max(images, int(np.prod(2 * reach + 1)))
    expected = 4.0 / 3.0 * math.pi * radius**3 * geometry.occupied_density
    degree = min(math.ceil(case.capacity_factor * expected) + 4, images * count)
    slots = (
        args.candidate_slots
        if args.candidate_slots is not None
        else max(
            4096, math.ceil(2.0 * case.capacity_factor * count * 27 * max(1.0, expected))
        )
    )
    return {
        "expected_degree": math.ceil(expected),
        "maximum_degree": degree,
        "maximum_search_edges": count * degree,
        "maximum_candidate_slots": int(slots),
        "maximum_particles_per_cell": count,
        "maximum_images": images,
    }


# Image-source names: point rows keep their historical source identity.
_SOURCE_NAMES = {
    "point": "charge-pairs",
    "multipole": "multipole-image-pairs",
    "polarization": "polarization-image-pairs",
    "learned": "learned-composition-images",
}


def _bundle_plan(
    definition: AtomisticPotentialProgram,
    case: LongRangeCase,
    geometry: Geometry,
    args: argparse.Namespace,
    /,
) -> tuple[AtomisticNeighborhoodBundlePlan, dict[str, Any]]:
    """Explicit resources of exactly the program's own neighborhood requirements.

    Finite rows bind one complete-pair source.  Periodic rows bind one declared
    cell-list image source serving every bounded requirement (search sized by
    the largest cutoff) and one role ladder per requirement sized by its own
    cutoff and counting (directed routes, or half routes of reversal pairs).
    """
    requirements = tuple(definition.requirements.neighborhoods)
    if not requirements:
        return AtomisticNeighborhoodBundlePlan((), (), ()), {"kind": "none"}
    if case.route == "finite":
        (requirement,) = requirements
        pairs = geometry.atoms * (geometry.atoms - 1) // 2
        source = AtomisticCompletePairSource(
            "complete-charge-pairs",
            requirement.entity,
            (requirement,),
            maximum_pairs=pairs,
        )
        return (
            AtomisticNeighborhoodBundlePlan((requirement,), (source,), ()),
            {"kind": "complete-pairs", "maximum_pairs": pairs},
        )
    entities = {requirement.entity for requirement in requirements}
    if len(entities) != 1:
        raise ValueError("One image source serves exactly one searched entity here.")
    cutoffs = [float(requirement.bounded_cutoff()) for requirement in requirements]
    capacities = _image_capacities(case, geometry, args, cutoff=max(cutoffs))
    search = ParticleImageCapacity(
        maximum_particles_per_cell=capacities["maximum_particles_per_cell"],
        maximum_edges=capacities["maximum_search_edges"],
        maximum_degree=capacities["maximum_degree"],
        maximum_images=capacities["maximum_images"],
        maximum_case_edges=capacities["maximum_search_edges"],
    )
    source = AtomisticNeighborhoodSource(
        _SOURCE_NAMES[case.family],
        requirements[0].entity,
        requirements,
        AtomisticCellListImageSearch(
            maximum_candidate_slots=capacities["maximum_candidate_slots"],
            deformation_margin=args.deformation_margin,
        ),
        ParticleImageCapacityLadder((search,)),
        skin=args.skin,
    )
    ladders = []
    roles = []
    for requirement, cutoff in zip(requirements, cutoffs, strict=True):
        degree = _image_capacities(case, geometry, args, cutoff=cutoff)["maximum_degree"]
        directed = requirement.counting is AtomisticPairCounting.DIRECTED
        routes = (
            geometry.atoms * degree
            if directed
            else math.ceil(geometry.atoms * degree / 2)
        )
        ladders.append(
            AtomisticRoleCapacityLadder(
                requirement,
                (AtomisticRoleCapacity(maximum_routes=routes, maximum_degree=degree),),
            )
        )
        roles.append(
            {
                "owner": requirement.owner,
                "role": requirement.role.value,
                "counting": requirement.counting.value,
                "cutoff": cutoff,
                "maximum_role_routes": routes,
                "maximum_degree": degree,
            }
        )
    return (
        AtomisticNeighborhoodBundlePlan(requirements, (source,), tuple(ladders)),
        {
            "kind": "cell-list-image-search",
            "source": _SOURCE_NAMES[case.family],
            "skin": args.skin,
            "deformation_margin": args.deformation_margin,
            **capacities,
            "roles": roles,
        },
    )


def _water_system(geometry: Geometry, /) -> PreparedAtomisticSystem:
    """Prepared water system in eV/angstrom units with its (zero) system charges."""
    if (
        geometry.atomic_numbers is None
        or geometry.masses is None
        or geometry.cell is None
    ):
        raise ValueError("Learned rows need a periodic water geometry.")
    count = geometry.atoms
    return AtomisticSystemPlan(
        np.arange(count, dtype=np.int64),
        geometry.atomic_numbers,
        geometry.masses,
        WATER_UNITS,
        charges=np.asarray(geometry.charges, dtype=np.float64),
        cell=PeriodicCell(geometry.cell),
        name="long-range-learned-charge-benchmark",
    ).prepare()


def _learned_coulomb(
    case: LongRangeCase,
    args: argparse.Namespace,
    /,
    *,
    charge_field: AtomisticFieldReference | None,
) -> Any:
    """Periodic Coulomb of the learned row under the uniform-background convention.

    With ``charge_field=None`` it reads the system charges (the held-charge
    control program); the definition is otherwise identical.
    """
    common: dict[str, Any] = {
        "neutrality": "uniform-background",
        "budget": _budget(args),
        "charge_field": charge_field,
    }
    if case.route == "ewald":
        return EwaldReferencePotential(
            args.learned_alpha,
            args.learned_real_cutoff,
            _extents(case, args),
            mode_chunk=_mode_chunk(case),
            **common,
        )
    if case.route == "pme":
        return ParticleMeshEwaldPotential(
            args.learned_alpha,
            args.learned_real_cutoff,
            _grid(case, args),
            spline_degree=_spline_degree(case),
            **common,
        )
    raise ValueError(f"Unknown learned Coulomb route {case.route!r}.")


def _published_fields(
    program: PreparedAtomisticPotentialProgram,
    positions: Array,
    cell_vectors: Array | None,
) -> tuple[Any, ...]:
    """Fields a program publishes at one point (shape-evaluated for port discovery)."""
    state = program.neighborhood_bundle.initialize(
        _geometry_of(program, positions, cell_vectors)
    )
    return program.energy(positions, state, cell_vectors=cell_vectors)[1].fields


def _learned_graph_execution(geometry: Geometry, args: argparse.Namespace, /) -> Any:
    return periodic_execution(
        degree=args.learned_graph_degree,
        edges=args.learned_graph_degree * geometry.atoms,
        particles_per_cell=32,
        search_cell=geometry.cell,
        search_margin=args.learned_search_margin,
    )


def _prepare_learned(
    case: LongRangeCase,
    geometry: Geometry,
    args: argparse.Namespace,
    /,
    *,
    revised: bool,
    total_charge: float | None,
) -> NativeBinding:
    """AIMNet2 -> typed CHARGES field -> periodic Coulomb in one native program.

    ``revised`` rebuilds the model from the next source-layout inventory seed
    (a model-data refresh); ``total_charge`` overrides the total-charge
    condition bound on the learned term (a condition refresh).  The producer
    port is read from a shape-only evaluation of the model-alone program.
    """
    charge = args.learned_total_charge if total_charge is None else float(total_charge)
    seed = args.learned_seed + (1 if revised else 0)
    system, system_phase = _host_phase(lambda: _water_system(geometry))
    model, model_phase = _host_phase(lambda: aimnet2_model(seed))
    execution = _learned_graph_execution(geometry, args)
    positions = jnp.asarray(geometry.positions)
    vectors = jnp.asarray(geometry.cell)

    def learned() -> tuple[Any, ...]:
        term = LearnedGraphPotentialTerm(
            model,
            name="aimnet2",
            allow_periodic=True,
            conditioning=AtomisticConditioning(
                model.conditioning_spec,
                {"charge": np.asarray([charge])},
                case_count=1,
                atom_capacity=system.capacity,
            ),
            request=AtomisticObservableRequest(
                (AtomisticObservable.CHARGES, AtomisticObservable.CHARGE_RESIDUAL)
            ),
        )
        alone_definition = AtomisticPotentialProgram([term])
        alone_bundle, _ = _bundle_plan(alone_definition, case, geometry, args)
        alone = alone_definition.prepare(
            system, neighborhood_bundle=alone_bundle, graph_execution=execution
        )
        fields = eqx.filter_eval_shape(_published_fields, alone, positions, vectors)
        (published,) = (
            value for value in fields if value.observable is AtomisticObservable.CHARGES
        )
        field = AtomisticFieldReference(
            term.name,
            AtomisticObservable.CHARGES,
            published.port,
            level=AtomisticConditionLevel.ATOM,
            unit=WATER_UNITS.charge_unit,
        )
        definition = AtomisticPotentialProgram(
            [term, _learned_coulomb(case, args, charge_field=field)]
        )
        return term, alone, field, definition

    (term, alone, field, program_definition), definition_phase = _host_phase(learned)
    (bundle_plan, capacities), bundle_phase = _host_phase(
        lambda: _bundle_plan(program_definition, case, geometry, args)
    )
    program, program_phase = _host_phase(
        lambda: program_definition.prepare(
            system, neighborhood_bundle=bundle_plan, graph_execution=execution
        )
    )
    return NativeBinding(
        system=system,
        definition=program_definition,
        program=program,
        phases={
            "system_prepare": system_phase,
            "model_construction": model_phase,
            "term_definition": definition_phase,
            "bundle_plan": bundle_phase,
            "program_prepare": program_phase,
            "retained_program_bytes": _retained(program),
            "learned": {
                "model": "AIMNet2 random source-layout fixture (no pretrained weights)",
                "model_seed": seed,
                "total_charge_condition": charge,
                "graph_execution_id": execution.plan_id,
                "producer_port": field.port.port_id,
                "charge_field": field.reference_id,
                "coulomb": {
                    "route": case.route,
                    "alpha": args.learned_alpha,
                    "real_cutoff": args.learned_real_cutoff,
                    "neutrality": "uniform-background",
                    "boundary": "tin-foil",
                    "units": "eV, angstrom, elementary charge; coulomb_constant "
                    f"{WATER_UNITS.coulomb_constant}",
                },
            },
        },
        capacities=capacities,
        learned=LearnedBinding(term, field, alone, execution, seed, charge),
    )


def prepare_native(
    case: LongRangeCase,
    geometry: Geometry,
    args: argparse.Namespace,
    /,
    *,
    revised: bool = False,
    comparison: bool = False,
    total_charge: float | None = None,
) -> NativeBinding:
    """Construct and prepare the row's actual public program, timing host phases."""
    if case.family == "learned":
        if comparison:
            raise ValueError("Learned rows have no native comparison program.")
        return _prepare_learned(
            case, geometry, args, revised=revised, total_charge=total_charge
        )
    if total_charge is not None:
        raise ValueError("Only learned rows bind a total-charge condition.")
    charges = geometry.coefficients(revised)[0] if case.family == "point" else None
    system, system_phase = _host_phase(lambda: _system(geometry, charges))

    def definition() -> AtomisticPotentialProgram:
        if case.family == "point":
            term = _point_term(case, args)
        elif case.family == "multipole":
            term = _multipole_term(
                case, geometry, args, revised=revised, comparison=comparison
            )
        else:
            term = _polarization_term(
                case, geometry, args, revised=revised, comparison=comparison
            )
        return AtomisticPotentialProgram([term])

    program_definition, definition_phase = _host_phase(definition)
    (bundle_plan, capacities), bundle_phase = _host_phase(
        lambda: _bundle_plan(program_definition, case, geometry, args)
    )
    program, program_phase = _host_phase(
        lambda: program_definition.prepare(system, neighborhood_bundle=bundle_plan)
    )
    return NativeBinding(
        system=system,
        definition=program_definition,
        program=program,
        phases={
            "system_prepare": system_phase,
            "term_definition": definition_phase,
            "bundle_plan": bundle_phase,
            "program_prepare": program_phase,
            "retained_program_bytes": _retained(program),
        },
        capacities=capacities,
    )


# ---------------------------------------------------------------- kernels


def _geometry_of(
    program: PreparedAtomisticPotentialProgram,
    positions: Array,
    cell_vectors: Array | None,
) -> AtomisticNeighborhoodGeometry:
    return bind_atomistic_bundle_geometry(
        program.neighborhood_bundle, program.system, positions, cell_vectors=cell_vectors
    )


def _initialize(
    program: PreparedAtomisticPotentialProgram,
    positions: Array,
    cell_vectors: Array | None,
) -> AtomisticNeighborhoodBundleState:
    return program.neighborhood_bundle.initialize(
        _geometry_of(program, positions, cell_vectors)
    )


def _refresh(
    program: PreparedAtomisticPotentialProgram,
    previous: AtomisticNeighborhoodBundleState,
    positions: Array,
    cell_vectors: Array | None,
) -> AtomisticNeighborhoodBundleState:
    return program.neighborhood_bundle.refresh(
        previous, _geometry_of(program, positions, cell_vectors)
    )


def _energy(
    program: PreparedAtomisticPotentialProgram,
    positions: Array,
    neighborhoods: AtomisticNeighborhoodBundleState,
    cell_vectors: Array | None,
) -> tuple[Array, Array, Array]:
    energy, auxiliary = program.energy(
        positions, neighborhoods, cell_vectors=cell_vectors
    )
    return energy, auxiliary.successful, auxiliary.neighborhood_successful


def _evaluate(
    program: PreparedAtomisticPotentialProgram,
    positions: Array,
    neighborhoods: AtomisticNeighborhoodBundleState,
    cell_vectors: Array | None,
    compute_stress: bool,
) -> AtomisticPotentialEvaluation:
    return program.evaluate(
        positions,
        neighborhoods,
        compute_stress=compute_stress,
        cell_vectors=cell_vectors,
    )


def _hessian_vector(
    program: PreparedAtomisticPotentialProgram,
    positions: Array,
    neighborhoods: AtomisticNeighborhoodBundleState,
    cell_vectors: Array | None,
    direction: Array,
) -> tuple[Array, Array]:
    def forces(values: Array) -> Array:
        return program.evaluate(values, neighborhoods, cell_vectors=cell_vectors).forces

    return jax.jvp(forces, (positions,), (direction,))


def _md_initialize(
    dynamics: PreparedAtomisticDynamics,
    thermodynamic: PreparedThermodynamicStateTable,
    positions: Array,
    velocity: Array,
    key: Array,
    cell_vectors: Array | None,
) -> AtomisticInitialStateEvaluation:
    return dynamics.initialize_state_detailed(
        positions, thermodynamic, velocity=velocity, key=key, cell_vectors=cell_vectors
    )


def _md_step(
    dynamics: PreparedAtomisticDynamics,
    thermodynamic: PreparedThermodynamicStateTable,
    state: AtomisticDynamicsState,
) -> AtomisticStepEvaluation:
    return dynamics.step_detailed(state, thermodynamic)


class WorksetItems(eqx.Module):
    """One shared prepared program and the item lane of positions/current cells.

    Only ``positions`` (and ``cell_vectors`` when periodic) carry the leading
    item axis, declared by an explicit item ``LaneLayout``; the prepared program
    is broadcast within each workset bucket.
    """

    program: PreparedAtomisticPotentialProgram
    positions: Array
    cell_vectors: Array | None


def _workset_item(
    compute_stress: bool,
    signature: PoolExecutionSignature,
    item: WorksetItems,
    key: Array,
    semantic_index: Array,
) -> dict[str, Array]:
    """Complete E/F/S of one item: its own neighborhood state at its own cell."""
    del signature, key, semantic_index
    state = _initialize(item.program, item.positions, item.cell_vectors)
    evaluation = _evaluate(
        item.program, item.positions, state, item.cell_vectors, compute_stress
    )
    values = {
        "energy": evaluation.energy,
        "forces": evaluation.forces,
        "successful": evaluation.successful,
        "neighborhood_successful": evaluation.neighborhood_successful,
        "bundle_successful": state.successful,
        "capacity_failure": state.capacity_failure,
        "scientific_failure": state.scientific_failure,
    }
    if evaluation.stress is not None:
        values["stress"] = evaluation.stress
    return values


def _efs_workset(
    prepared: PreparedExecutionWorksets,
    items: WorksetItems,
    layout: phx.LaneLayout,
    root_key: Array,
    counters: Array,
    compute_stress: bool,
) -> Any:
    return evaluate_execution_worksets_filter_vmap(
        prepared,
        partial(_workset_item, compute_stress),
        items,
        root_key,
        counters,
        layout=layout,
    )


def _parameter_gradient(
    lane: Any,
    state: Any,
    fixed: Any,
    program: PreparedAtomisticPotentialProgram,
    positions: Array,
    neighborhoods: AtomisticNeighborhoodBundleState,
    cell_vectors: Array,
    learned_index: int,
    coulomb_index: int,
) -> tuple[tuple[Array, Any], tuple[Array, Any]]:
    """Model-parameter gradients through the connected program.

    The first is an E/F/S loss of the complete program; the second is the
    Coulomb term energy alone, which depends on the parameters only through
    the predicted charge field (the chain rule into training).
    """

    def bind(values: Any) -> PreparedAtomisticPotentialProgram:
        model = combine_parameters(values, state, fixed)
        return eqx.tree_at(
            lambda value: value.terms[learned_index].plan.potential, program, model
        )

    def loss(values: Any) -> Array:
        evaluation = bind(values).evaluate(
            positions, neighborhoods, compute_stress=True, cell_vectors=cell_vectors
        )
        if evaluation.stress is None:
            raise RuntimeError("The parameter-gradient loss requires the stress.")
        return (
            jnp.sum(evaluation.energy**2)
            + jnp.sum(evaluation.forces**2)
            + jnp.sum(evaluation.stress**2)
        )

    def coulomb(values: Any) -> Array:
        _, auxiliary = bind(values).energy(
            positions, neighborhoods, cell_vectors=cell_vectors
        )
        return auxiliary.term_energies[coulomb_index]

    return jax.value_and_grad(loss)(lane), jax.value_and_grad(coulomb)(lane)


_KERNELS: dict[str, Any] = {
    "neighborhood-initialize": eqx.filter_jit(_initialize),
    "neighborhood-refresh": eqx.filter_jit(_refresh),
    "energy": eqx.filter_jit(_energy),
    "efs": eqx.filter_jit(_evaluate),
    "hessian-vector": eqx.filter_jit(_hessian_vector),
    "md-initialize": eqx.filter_jit(_md_initialize),
    "md-step": eqx.filter_jit(_md_step),
    "efs-workset": eqx.filter_jit(_efs_workset),
    "parameter-gradient": eqx.filter_jit(_parameter_gradient),
}


# ------------------------------------------------------------ measurement


def failure_record(error: BaseException, /) -> dict[str, Any]:
    message = str(error)
    if "RESOURCE_EXHAUSTED" in message or "out of memory" in message.lower():
        status = "out-of-memory"
    elif isinstance(error, (ValueError, TypeError)):
        status = "refused"
    else:
        status = "failed"
    return {
        "status": status,
        "error_type": type(error).__name__,
        "message": message,
        "traceback": traceback.format_exception_only(error),
    }


def _finite(value: Any, /) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if np.all(np.isfinite(array)):
        return array.tolist()
    return {"kind": "nonfinite", "finite_mask": np.isfinite(array).tolist()}


def _scalar(value: Any, /) -> float | dict[str, str]:
    number = float(np.asarray(value, dtype=np.float64).reshape(()))
    return (
        number if math.isfinite(number) else {"kind": "nonfinite", "value": repr(number)}
    )


def _payload(value: Any, depth: int = 0, /) -> Any:
    """JSON evidence of a native record: finite numbers, host-decoded identities."""
    if depth > 6:
        return {"kind": "truncated", "type": type(value).__name__}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return (
            value if math.isfinite(value) else {"kind": "nonfinite", "value": repr(value)}
        )
    if isinstance(value, enum.Enum):
        return _payload(value.value, depth + 1)
    if isinstance(value, IdentityWords):
        return value.fingerprint()
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            return {"kind": "prng-key", "shape": list(value.shape)}
        array = np.asarray(value)
        if array.dtype == np.bool_:
            return (
                array.tolist()
                if array.size <= 64
                else {
                    "shape": list(array.shape),
                    "all": bool(np.all(array)),
                    "any": bool(np.any(array)),
                }
            )
        if array.size <= 64 and array.dtype.kind in "biuf":
            return _finite(array) if array.dtype.kind == "f" else array.tolist()
        finite = np.isfinite(array) if array.dtype.kind == "f" else np.ones(1, bool)
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "all_finite": bool(np.all(finite)),
        }
    if isinstance(value, Mapping):
        return {str(key): _payload(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_payload(item, depth + 1) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__name__,
            **{
                field.name: _payload(getattr(value, field.name), depth + 1)
                for field in dataclasses.fields(value)
            },
        }
    return repr(value)


def _retained(value: Any, /) -> int | dict[str, str]:
    """Logical array payload bytes, or why an object graph could not be sized."""
    try:
        return logical_array_bytes(value)
    except Exception as error:
        return {"kind": "unavailable", "reason": f"{type(error).__name__}: {error}"}


class Executable(Protocol):
    """Equinox AOT executable: callable on the lowered argument layout."""

    compiled: jax.stages.Compiled

    def __call__(self, *arguments: object) -> Any: ...


def _compiled_evidence(executable: Executable, /) -> dict[str, Any]:
    compiled = executable.compiled
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    return asdict(
        compiler_evidence(
            cost,
            compiled.memory_analysis(),
            source="xla",
            unavailable_reason="The selected backend did not expose compiler estimates.",
        )
    )


def _warm(
    name: str, invocation: Callable[[], Any], repeats: int, /
) -> tuple[Any, dict[str, Any]]:
    devices = tuple(device for device in jax.local_devices() if device.platform != "cpu")
    with PhaseMemorySampler(
        name, interval_seconds=_SAMPLE_INTERVAL, devices=devices
    ) as sampler:
        output, warm = measure_repeated(invocation, warmup=1, repeats=repeats)
    evidence = sampler.evidence
    return output, {
        "warm": warm.to_dict(unit="seconds"),
        "sampled_peak_increase_bytes": evidence.sampled_peak_increase_bytes,
        "memory": evidence.to_payload(),
    }


def measure_kernel(
    name: str, arguments: tuple[object, ...], repeats: int, /
) -> tuple[Any, Executable, dict[str, Any]]:
    """Lower, compile, first and warm execution of one native kernel."""
    lowering = _KERNELS[name].lower
    executable, phases = measure_lower_and_compile(
        partial(lowering, *arguments), methodcaller("compile")
    )
    invocation = partial(executable, *arguments)
    output, first = measure_synchronized(invocation)
    output, warm = _warm(name, invocation, repeats)
    return (
        output,
        executable,
        {
            "status": "measured",
            "lowering_seconds": phases.lowering_seconds,
            "compilation_seconds": phases.compilation_seconds,
            "first_seconds": first,
            **warm,
            "compiler": _compiled_evidence(executable),
            "argument_retained_bytes": _retained(arguments),
            "output_retained_bytes": _retained(output),
            "process_peak_rss_bytes": process_peak_rss_bytes(),
        },
    )


def executable_layout(left: tuple[Any, ...], right: tuple[Any, ...], /) -> dict[str, Any]:
    """Whether an AOT executable lowered for ``left`` is valid for ``right``.

    An equinox AOT executable consumes only the array leaves of a new call and
    keeps the static part captured at lowering, so reuse is admitted only when
    the static PyTree part is equal and every dynamic aval matches.
    """
    left_dynamic, left_static = eqx.partition(left, eqx.is_array)
    right_dynamic, right_static = eqx.partition(right, eqx.is_array)
    try:
        static_equal = jax.tree.structure(left_static) == jax.tree.structure(
            right_static
        ) and bool(eqx.tree_equal(left_static, right_static))
        reason = None
    except Exception as error:  # Static leaves without a defined equality.
        static_equal, reason = False, f"{type(error).__name__}: {error}"
    left_leaves, left_tree = jax.tree.flatten(left_dynamic)
    right_leaves, right_tree = jax.tree.flatten(right_dynamic)
    avals_equal = left_tree == right_tree and all(
        np.shape(first) == np.shape(second)
        and jnp.result_type(first) == jnp.result_type(second)
        for first, second in zip(left_leaves, right_leaves, strict=True)
    )
    return {
        "static_equal": static_equal,
        "dynamic_avals_equal": avals_equal,
        "reusable": static_equal and avals_equal,
        "static_comparison_error": reason,
    }


def _reuse_or_recompile(
    name: str,
    executable: Executable,
    lowered_arguments: tuple[object, ...],
    arguments: tuple[object, ...],
    repeats: int,
    /,
) -> tuple[Any, Executable, dict[str, Any]]:
    layout = executable_layout(lowered_arguments, arguments)
    if not layout["reusable"]:
        output, fresh, record = measure_kernel(name, arguments, repeats)
        return output, fresh, {"layout": layout, "executable": "recompiled", **record}
    invocation = partial(executable, *arguments)
    output, first = measure_synchronized(invocation)
    output, warm = _warm(f"{name}-reused", invocation, repeats)
    return (
        output,
        executable,
        {
            "layout": layout,
            "executable": "reused",
            "status": "measured",
            "first_seconds": first,
            **warm,
            "output_retained_bytes": _retained(output),
        },
    )


def _bundle_status(state: Any, /) -> dict[str, Any]:
    return {
        "successful": np.asarray(state.successful).tolist(),
        "capacity_failure": np.asarray(state.capacity_failure).tolist(),
        "scientific_failure": np.asarray(state.scientific_failure).tolist(),
        "capacity_only": np.asarray(state.capacity_only).tolist(),
        "stale": np.asarray(state.stale).tolist(),
        "uninitialized": np.asarray(state.uninitialized).tolist(),
        "bundle_id": state.bundle_id,
        "plan_id": state.plan_id,
        "layout_id": state.layout_id,
    }


def _native_outputs(evaluation: Any, /) -> dict[str, Any]:
    forces = np.asarray(evaluation.forces, dtype=np.float64)
    return {
        "energy": _scalar(evaluation.energy),
        "successful": np.asarray(evaluation.successful).tolist(),
        "neighborhood_successful": np.asarray(
            evaluation.neighborhood_successful
        ).tolist(),
        "max_abs_force": _scalar(np.max(np.abs(forces))),
        "net_force": _finite(np.sum(forces, axis=0)),
        "stress": None if evaluation.stress is None else _finite(evaluation.stress),
        "stress_convention": _payload(evaluation.stress_convention),
        "program_id": evaluation.program_id,
        "term_evidence": _payload(evaluation.term_evidence),
    }


def _errors(evaluation: Any, reference: Mapping[str, Any], /) -> dict[str, Any]:
    """Native minus reference: energy, forces and (where both exist) stress."""
    energy = float(np.asarray(evaluation.energy))
    forces = np.asarray(evaluation.forces, dtype=np.float64)
    if not math.isfinite(energy) or not np.all(np.isfinite(forces)):
        return {"kind": "native-nonfinite"}
    expected = np.asarray(reference["forces"], dtype=np.float64)
    difference = forces - expected
    scale = float(np.sqrt(np.mean(expected**2)))
    record: dict[str, Any] = {
        "reference_energy": reference["energy"],
        "energy_absolute": abs(energy - reference["energy"]),
        "energy_relative": abs(energy - reference["energy"])
        / max(abs(reference["energy"]), np.finfo(np.float64).tiny),
        "force_max_absolute": float(np.max(np.abs(difference))),
        "force_rms_absolute": float(np.sqrt(np.mean(difference**2))),
        "force_rms_relative": float(np.sqrt(np.mean(difference**2)))
        / max(scale, np.finfo(np.float64).tiny),
    }
    stress = evaluation.stress
    if stress is not None and reference.get("stress") is not None:
        native = np.asarray(stress, dtype=np.float64)
        expected_stress = np.asarray(reference["stress"], dtype=np.float64)
        if np.all(np.isfinite(native)):
            record["stress_max_absolute"] = float(
                np.max(np.abs(native - expected_stress))
            )
            record["stress_max_relative"] = record["stress_max_absolute"] / max(
                float(np.max(np.abs(expected_stress))), np.finfo(np.float64).tiny
            )
        else:
            record["stress"] = "native-nonfinite"
    return record


def _reference_summary(reference: Mapping[str, Any], /) -> dict[str, Any]:
    return {
        key: _payload(value)
        for key, value in reference.items()
        if key not in _MOMENT_REFERENCE_ARRAYS
    }


@dataclass
class RowContext:
    """Per-row executables and arguments shared by dependent workloads.

    ``held`` caches the learned row's held-charge control program and the
    predicted charges it binds as system charges.
    """

    native: NativeBinding
    positions: Array
    cell_vectors: Array | None
    neighborhoods: AtomisticNeighborhoodBundleState | None = None
    initialize: tuple[Executable, tuple[object, ...]] | None = None
    efs: tuple[Executable, tuple[object, ...]] | None = None
    held: tuple[NativeBinding, np.ndarray] | None = None


# (reference kind, geometry id, revised coefficients, strained configuration).
ReferenceKey: TypeAlias = tuple[str, str, bool, bool]
ReferenceCache: TypeAlias = dict[ReferenceKey, dict[str, Any]]
Workload: TypeAlias = Callable[
    [
        RowContext,
        LongRangeCase,
        argparse.Namespace,
        dict[str, Any],
        ReferenceCache,
        Geometry,
    ],
    dict[str, Any] | None,
]


def _reference(
    cache: ReferenceCache,
    case: LongRangeCase,
    geometry: Geometry,
    args: argparse.Namespace,
    /,
    *,
    revised: bool,
    strained: bool,
    kind: str | None = None,
) -> dict[str, Any] | None:
    """Cached independent reference, or an explicit reason it is not formed."""
    kind = _reference_kind(case.method) if kind is None else kind
    if kind is None:
        reason = (
            "learned-charge row: the uniform-background Ewald reference of the "
            "runtime predicted charges is recorded in learned_composition"
            if case.family == "learned"
            else "no independent reference of this definition in this driver"
        )
        return {"status": "not-formed", "reason": reason}
    if geometry.atoms > args.reference_max_atoms:
        return {
            "status": "not-computed",
            "reason": f"atoms exceed --reference-max-atoms {args.reference_max_atoms}",
        }
    key = (kind, geometry.geometry_id, revised, strained)
    if key not in cache:
        try:
            cache[key] = {
                "status": "formed",
                **independent_reference(
                    kind, geometry, args, revised=revised, strained=strained
                ),
            }
        except Exception as error:
            cache[key] = failure_record(error)
            history = getattr(error, "history", None)
            if history is not None:
                # Exhausted reference bounds keep their refinement history.
                cache[key]["history"] = _payload(history)
    return cache[key]


def _compare(evaluation: Any, reference: Mapping[str, Any] | None, /) -> dict[str, Any]:
    if reference is None or reference.get("status") != "formed":
        return {"reference": None if reference is None else _payload(reference)}
    return {
        "reference": _reference_summary(reference),
        "errors": _errors(evaluation, reference),
    }


def _arguments(context: RowContext, neighborhoods: Any, stress: bool, /) -> tuple:
    return (
        context.native.program,
        context.positions,
        neighborhoods,
        context.cell_vectors,
        stress,
    )


def _ensure_neighborhoods(
    context: RowContext, args: argparse.Namespace, record: dict[str, Any], /
) -> None:
    if context.neighborhoods is not None:
        return
    arguments = (context.native.program, context.positions, context.cell_vectors)
    state, executable, phase = measure_kernel(
        "neighborhood-initialize", arguments, args.repeats
    )
    phase["bundle"] = _bundle_status(state)
    phase["retained_neighborhood_bytes"] = _retained(state)
    record["neighborhood-initialize"] = phase
    context.neighborhoods = state
    context.initialize = (executable, arguments)


def _ensure_efs(
    context: RowContext,
    case: LongRangeCase,
    args: argparse.Namespace,
    record: dict[str, Any],
    cache: ReferenceCache,
    geometry: Geometry,
    /,
) -> None:
    if context.efs is not None:
        return
    _ensure_neighborhoods(context, args, record)
    arguments = _arguments(context, context.neighborhoods, case.periodic)
    evaluation, executable, phase = measure_kernel("efs", arguments, args.repeats)
    phase["outputs"] = _native_outputs(evaluation)
    phase.update(
        _compare(
            evaluation,
            _reference(cache, case, geometry, args, revised=False, strained=False),
        )
    )
    if case.method == "dsf":
        phase["definition_discrepancy_vs_converged_ewald"] = _compare(
            evaluation,
            _reference(
                cache, case, geometry, args, revised=False, strained=False, kind="ewald"
            ),
        )
    if case.family in ("multipole", "polarization") and case.periodic:
        phase["native_comparison"] = (
            _native_comparison(case, geometry, args, evaluation)
            if args.native_comparison
            else {"status": "not-requested", "reason": "--no-native-comparison"}
        )
    if case.family == "learned":
        try:
            phase["learned_composition"] = _learned_composition(
                context, case, geometry, args, evaluation
            )
        except Exception as error:
            phase["learned_composition"] = failure_record(error)
    if "-tcg-" in case.method:
        phase["reference_note"] = (
            "fixed-order TCG functional versus a converged stationary reference: "
            "the discrepancy includes the declared truncation"
        )
    record["efs"] = phase
    context.efs = (executable, arguments)


def _held_program(
    case: LongRangeCase,
    geometry: Geometry,
    args: argparse.Namespace,
    charges: np.ndarray,
    /,
) -> NativeBinding:
    """The same Coulomb definition reading ``charges`` as fixed system charges."""
    held = dataclasses.replace(geometry, charges=charges, revised_charges=charges)
    system = _water_system(held)
    definition = AtomisticPotentialProgram(
        [_learned_coulomb(case, args, charge_field=None)]
    )
    bundle, capacities = _bundle_plan(definition, case, held, args)
    program = definition.prepare(system, neighborhood_bundle=bundle)
    return NativeBinding(system, definition, program, {}, capacities)


def _observable(evaluation: Any, observable: AtomisticObservable, /) -> Any:
    values = [value for value in evaluation.fields if value.observable is observable]
    if len(values) != 1:
        raise ValueError(f"Expected one published {observable.value} field.")
    return values[0]


def _evaluate_at_point(
    program: PreparedAtomisticPotentialProgram,
    positions: Array,
    cell_vectors: Array | None,
    stress: bool,
    /,
) -> Any:
    state = _KERNELS["neighborhood-initialize"](program, positions, cell_vectors)
    return _KERNELS["efs"](program, positions, state, cell_vectors, stress)


def _learned_composition(
    context: RowContext,
    case: LongRangeCase,
    geometry: Geometry,
    args: argparse.Namespace,
    evaluation: Any,
    /,
) -> dict[str, Any]:
    """Connected AIMNet2 -> Coulomb controls at the row's evaluation point.

    * the model-alone program (same learned term) and the held-charge program
      (same Coulomb definition reading the predicted charges as fixed system
      charges) reproduce the connected term energies;
    * ``connected - alone - held`` forces/stress are the charge-response
      (``dq/dx``, ``dq/dstrain``) contributions, nonzero for a connected chain;
    * an independent uniform-background converged Ewald of the predicted
      charges checks the held program's E/F/S and the connected Coulomb term;
    * native central differences of the connected energy (non-independent)
      check selected connected force components including the chain rule.
    """
    learned = context.native.learned
    cell = geometry.cell
    if learned is None or cell is None or context.cell_vectors is None:
        raise ValueError("The learned composition needs a learned periodic row.")
    if not bool(np.all(np.asarray(evaluation.successful))):
        return {"status": "native-unsuccessful", "outputs": _native_outputs(evaluation)}
    predicted = evaluation.field(learned.charge_field)
    charges = np.asarray(predicted.values, dtype=np.float64).reshape(-1)[: geometry.atoms]
    residual = np.asarray(
        _observable(evaluation, AtomisticObservable.CHARGE_RESIDUAL).values,
        dtype=np.float64,
    ).reshape(-1)
    held, held_phase = _host_phase(lambda: _held_program(case, geometry, args, charges))
    context.held = (held, charges)
    positions, vectors = context.positions, context.cell_vectors
    alone = _evaluate_at_point(learned.alone, positions, vectors, True)
    fixed = _evaluate_at_point(held.program, positions, vectors, True)
    connected_forces = np.asarray(evaluation.forces, dtype=np.float64)
    response = connected_forces - (
        np.asarray(alone.forces, dtype=np.float64)
        + np.asarray(fixed.forces, dtype=np.float64)
    )
    stress_response = None
    if (
        evaluation.stress is not None
        and alone.stress is not None
        and fixed.stress is not None
    ):
        stress_response = np.asarray(evaluation.stress, dtype=np.float64) - (
            np.asarray(alone.stress, dtype=np.float64)
            + np.asarray(fixed.stress, dtype=np.float64)
        )
    terms = np.asarray(evaluation.term_energies, dtype=np.float64)
    reference, reference_seconds = measure_host(
        partial(
            _periodic_reference,
            np.asarray(geometry.positions),
            charges,
            cell,
            WATER_UNITS.coulomb_constant,
            slab=False,
            background=True,
            tolerance=args.reference_tolerance,
            maximum_level=args.reference_max_level,
            step=args.fd_step,
        )
    )
    reference = {"status": "formed", "host_seconds": reference_seconds, **reference}
    program = context.native.program
    differences = []
    for atom, axis in ((0, 0), (geometry.atoms // 2, 1), (geometry.atoms - 1, 2)):
        values = []
        for sign in (1.0, -1.0):
            point = context.positions.at[atom, axis].add(sign * args.fd_step)
            state = _KERNELS["neighborhood-refresh"](
                program, context.neighborhoods, point, vectors
            )
            energy, successful, _ = _KERNELS["energy"](program, point, state, vectors)
            values.append(float(energy) if bool(np.all(successful)) else math.nan)
        central = -(values[0] - values[1]) / (2.0 * args.fd_step)
        differences.append(
            {
                "atom": atom,
                "axis": axis,
                "native_force": float(connected_forces[atom, axis]),
                "central_difference_force": _scalar(central),
                "absolute_difference": _scalar(
                    abs(central - float(connected_forces[atom, axis]))
                ),
            }
        )
    force_scale = float(np.sqrt(np.mean(connected_forces**2)))
    return {
        "status": "formed",
        "charge_source": learned.charge_field.reference_id,
        "per_call_refresh": "every program call re-evaluates the AIMNet2 graph at "
        "the call's positions and current cell and feeds the predicted charges to "
        "the Coulomb term; model parameters and the total-charge condition are "
        "prepared program data (see coefficient-refresh)",
        "predicted_charges": {
            "sum": float(np.sum(charges)),
            "max_abs": float(np.max(np.abs(charges))),
            "published_residual": residual.tolist(),
            "sum_plus_residual": float(np.sum(charges) + np.sum(residual)),
            "convention": "uniform-background: the residual net charge is "
            "neutralized by the declared background, not removed from the charges",
        },
        "model_alone_program": {
            "prepared_id": learned.alone.prepared_id,
            "outputs": _native_outputs(alone),
        },
        "held_charge_program": {
            "preparation": held_phase,
            "prepared_id": held.program.prepared_id,
            "outputs": _native_outputs(fixed),
        },
        "term_energy_consistency": {
            "learned_term_minus_alone": _scalar(
                abs(terms[learned.learned_index] - float(np.asarray(alone.energy)))
            ),
            "coulomb_term_minus_held": _scalar(
                abs(terms[learned.coulomb_index] - float(np.asarray(fixed.energy)))
            ),
        },
        "charge_response": {
            "definition": "connected - model-alone - held-charge",
            "force_max_absolute": _scalar(np.max(np.abs(response))),
            "force_rms_relative_to_connected": _scalar(
                float(np.sqrt(np.mean(response**2)))
                / max(force_scale, np.finfo(np.float64).tiny)
            ),
            "stress_max_absolute": None
            if stress_response is None
            else _scalar(np.max(np.abs(stress_response))),
        },
        "independent_reference": {
            "reference": _reference_summary(reference),
            "held_charge_errors": _errors(fixed, reference),
            "connected_coulomb_term_energy_absolute": _scalar(
                abs(terms[learned.coulomb_index] - reference["energy"])
            ),
        },
        "native_central_difference": {
            "independent": False,
            "step": args.fd_step,
            "components": differences,
        },
    }


def _native_comparison(
    case: LongRangeCase,
    geometry: Geometry,
    args: argparse.Namespace,
    evaluation: Any,
    /,
) -> dict[str, Any]:
    """Tighter native direct-k (converged PCG) evaluation; never independent."""
    try:
        native = prepare_native(case, geometry, args, comparison=True)
        positions = jnp.asarray(geometry.positions)
        vectors = jnp.asarray(geometry.cell)
        state = _KERNELS["neighborhood-initialize"](native.program, positions, vectors)
        comparison = _KERNELS["efs"](native.program, positions, state, vectors, True)
        reference = {
            "status": "formed",
            "kind": f"native-direct-k-extent-{args.native_comparison_extent}"
            + ("-pcg" if case.family == "polarization" else ""),
            "independent": False,
            "energy": float(np.asarray(comparison.energy)),
            "forces": np.asarray(comparison.forces),
            "stress": None
            if comparison.stress is None
            else np.asarray(comparison.stress),
            "successful": np.asarray(comparison.successful).tolist(),
            "scientific_id": native.program.scientific_id,
        }
        return _compare(evaluation, reference)
    except Exception as error:
        return failure_record(error)


def _energy_workload(
    context: RowContext,
    case: LongRangeCase,
    args: argparse.Namespace,
    record: dict[str, Any],
    cache: ReferenceCache,
    geometry: Geometry,
) -> dict[str, Any]:
    _ensure_neighborhoods(context, args, record)
    output, _, phase = measure_kernel(
        "energy",
        (
            context.native.program,
            context.positions,
            context.neighborhoods,
            context.cell_vectors,
        ),
        args.repeats,
    )
    energy, successful, neighborhood = output
    phase["outputs"] = {
        "energy": _scalar(energy),
        "successful": np.asarray(successful).tolist(),
        "neighborhood_successful": np.asarray(neighborhood).tolist(),
    }
    return phase


def _efs_workload(
    context: RowContext,
    case: LongRangeCase,
    args: argparse.Namespace,
    record: dict[str, Any],
    cache: ReferenceCache,
    geometry: Geometry,
) -> None:
    _ensure_efs(context, case, args, record, cache, geometry)


def _hessian_vector_workload(
    context: RowContext,
    case: LongRangeCase,
    args: argparse.Namespace,
    record: dict[str, Any],
    cache: ReferenceCache,
    geometry: Geometry,
) -> dict[str, Any]:
    _ensure_neighborhoods(context, args, record)
    direction = np.random.default_rng(args.seed + 7).normal(size=geometry.positions.shape)
    direction = jnp.asarray(direction / np.linalg.norm(direction))
    output, _, phase = measure_kernel(
        "hessian-vector",
        (
            context.native.program,
            context.positions,
            context.neighborhoods,
            context.cell_vectors,
            direction,
        ),
        args.repeats,
    )
    _, tangent = output
    tangent = np.asarray(tangent, dtype=np.float64)
    step = args.fd_step
    displaced = []
    for sign in (1.0, -1.0):
        position = context.positions + sign * step * direction
        state = _KERNELS["neighborhood-refresh"](
            context.native.program, context.neighborhoods, position, context.cell_vectors
        )
        displaced.append(
            np.asarray(
                _KERNELS["efs"](
                    context.native.program, position, state, context.cell_vectors, False
                ).forces,
                dtype=np.float64,
            )
        )
    central = (displaced[0] - displaced[1]) / (2.0 * step)
    phase["outputs"] = {"tangent_norm": _scalar(np.linalg.norm(tangent))}
    phase["native_central_difference_consistency"] = {
        "independent": False,
        "step": step,
        "max_absolute": _scalar(np.max(np.abs(tangent - central))),
        "relative": _scalar(
            np.linalg.norm(tangent - central)
            / max(np.linalg.norm(central), np.finfo(np.float64).tiny)
        ),
    }
    if context.native.learned is not None:
        try:
            phase["charge_chain_rule"] = _hvp_chain_rule(
                context, case, args, record, cache, geometry, direction, tangent
            )
        except Exception as error:
            phase["charge_chain_rule"] = failure_record(error)
    return phase


def _hvp_chain_rule(
    context: RowContext,
    case: LongRangeCase,
    args: argparse.Namespace,
    record: dict[str, Any],
    cache: ReferenceCache,
    geometry: Geometry,
    direction: Array,
    tangent: np.ndarray,
    /,
) -> dict[str, Any]:
    """Connected minus model-alone and held-charge force tangents along ``v``."""
    _ensure_efs(context, case, args, record, cache, geometry)
    learned = context.native.learned
    if learned is None or context.held is None:
        raise RuntimeError("The held-charge control program was not formed.")
    parts = []
    for program in (learned.alone, context.held[0].program):
        state = _KERNELS["neighborhood-initialize"](
            program, context.positions, context.cell_vectors
        )
        _, part = _KERNELS["hessian-vector"](
            program, context.positions, state, context.cell_vectors, direction
        )
        parts.append(np.asarray(part, dtype=np.float64))
    response = tangent - parts[0] - parts[1]
    return {
        "definition": "connected - model-alone - held-charge force tangents",
        "max_absolute": _scalar(np.max(np.abs(response))),
        "relative_to_connected": _scalar(
            np.linalg.norm(response)
            / max(np.linalg.norm(tangent), np.finfo(np.float64).tiny)
        ),
    }


def _refresh_through(
    context: RowContext,
    case: LongRangeCase,
    args: argparse.Namespace,
    cache: ReferenceCache,
    geometry: Geometry,
    revised: NativeBinding,
    host_seconds: float,
    /,
    *,
    refreshed: str,
    reference_revised: bool,
) -> dict[str, Any]:
    """Evaluate a re-prepared program through the base executables when admitted."""
    base = context.native.program
    phase: dict[str, Any] = {
        "coefficients": refreshed,
        "host_refresh_seconds": host_seconds,
        "revised_preparation": revised.phases,
        "layout_id_equal": base.layout_id == revised.program.layout_id,
        "scientific_id_changed": base.scientific_id != revised.program.scientific_id,
        "prepared_id_changed": base.prepared_id != revised.program.prepared_id,
    }
    if context.initialize is None or context.efs is None:
        raise RuntimeError("The base executables were not formed.")
    executable, lowered = context.initialize
    arguments = (revised.program, context.positions, context.cell_vectors)
    state, _, phase["neighborhood-initialize"] = _reuse_or_recompile(
        "neighborhood-initialize", executable, lowered, arguments, args.repeats
    )
    phase["neighborhood-initialize"]["bundle"] = _bundle_status(state)
    executable, lowered = context.efs
    arguments = (
        revised.program,
        context.positions,
        state,
        context.cell_vectors,
        case.periodic,
    )
    evaluation, _, phase["efs"] = _reuse_or_recompile(
        "efs", executable, lowered, arguments, args.repeats
    )
    phase["efs"]["outputs"] = _native_outputs(evaluation)
    if revised.learned is not None:
        if bool(np.all(np.asarray(evaluation.successful))):
            charges = np.asarray(
                evaluation.field(revised.learned.charge_field).values, dtype=np.float64
            )
            phase["efs"]["predicted_charge_sum"] = float(np.sum(charges))
        return phase
    phase["efs"].update(
        _compare(
            evaluation,
            _reference(
                cache, case, geometry, args, revised=reference_revised, strained=False
            ),
        )
    )
    return phase


def _coefficient_refresh_workload(
    context: RowContext,
    case: LongRangeCase,
    args: argparse.Namespace,
    record: dict[str, Any],
    cache: ReferenceCache,
    geometry: Geometry,
) -> dict[str, Any]:
    _ensure_efs(context, case, args, record, cache, geometry)
    if case.family != "learned":
        revised, host_seconds = measure_host(
            lambda: prepare_native(case, geometry, args, revised=True)
        )
        refreshed = {
            "point": "system charges",
            "multipole": "charges, dipoles and second moments",
            "polarization": "permanent moments and polarizabilities",
        }[case.family]
        return _refresh_through(
            context,
            case,
            args,
            cache,
            geometry,
            revised,
            host_seconds,
            refreshed=refreshed,
            reference_revised=True,
        )
    phase: dict[str, Any] = {}
    refreshes = (
        (
            "model-refresh",
            "AIMNet2 model parameters (next source-layout inventory seed); "
            "geometry and condition unchanged",
            {"revised": True},
        ),
        (
            "condition-refresh",
            f"total-charge condition {args.learned_refresh_charge}; model "
            "parameters and geometry unchanged",
            {"total_charge": args.learned_refresh_charge},
        ),
    )
    for name, refreshed, options in refreshes:
        try:
            revised, host_seconds = measure_host(
                lambda options=options: prepare_native(case, geometry, args, **options)
            )
            phase[name] = _refresh_through(
                context,
                case,
                args,
                cache,
                geometry,
                revised,
                host_seconds,
                refreshed=refreshed,
                reference_revised=False,
            )
        except Exception as error:
            phase[name] = failure_record(error)
    failed = [
        name for name in phase if phase[name].get("status") not in (None, "measured")
    ]
    phase["status"] = "partial" if failed else "measured"
    return phase


def _cell_refresh_workload(
    context: RowContext,
    case: LongRangeCase,
    args: argparse.Namespace,
    record: dict[str, Any],
    cache: ReferenceCache,
    geometry: Geometry,
) -> dict[str, Any]:
    if not case.periodic:
        return {"status": "not-applicable", "reason": "finite geometry has no cell"}
    _ensure_efs(context, case, args, record, cache, geometry)
    positions = jnp.asarray(geometry.strained_positions)
    vectors = jnp.asarray(geometry.strained_cell)
    state, _, phase = measure_kernel(
        "neighborhood-refresh",
        (context.native.program, context.neighborhoods, positions, vectors),
        args.repeats,
    )
    phase["strain"] = args.cell_strain
    phase["bundle"] = _bundle_status(state)
    if context.efs is None:
        raise RuntimeError("The base efs executable was not formed.")
    executable, lowered = context.efs
    arguments = (context.native.program, positions, state, vectors, True)
    evaluation, _, phase["efs"] = _reuse_or_recompile(
        "efs", executable, lowered, arguments, args.repeats
    )
    phase["efs"]["outputs"] = _native_outputs(evaluation)
    phase["efs"].update(
        _compare(
            evaluation,
            _reference(cache, case, geometry, args, revised=False, strained=True),
        )
    )
    return phase


def _md_workload(
    context: RowContext,
    case: LongRangeCase,
    args: argparse.Namespace,
    record: dict[str, Any],
    cache: ReferenceCache,
    geometry: Geometry,
) -> dict[str, Any]:
    system, program = context.native.system, context.native.program
    learned = case.family == "learned"
    step_size = args.learned_md_step_size if learned else args.md_step_size
    dynamics, dynamics_phase = _host_phase(
        lambda: AtomisticDynamicsPlan(
            system, program, VelocityVerletPlan(step_size), strain_response=False
        ).prepare()
    )
    thermodynamic, table_phase = _host_phase(
        lambda: AtomisticThermodynamicStatePlan(
            AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
        ).prepare(dynamics)
    )
    rng = np.random.default_rng(args.seed + 11)
    if learned:
        if geometry.masses is None:
            raise ValueError("Learned MD needs the water masses.")
        # sqrt(k_B T / m) in angstrom/fs: k_B in eV/K, 1 eV/Da = 9.64853321e-3 A^2/fs^2.
        scale = np.sqrt(
            8.617333262e-5 * args.learned_md_temperature / geometry.masses * 9.64853321e-3
        )[:, None]
        velocity = rng.normal(size=geometry.positions.shape) * scale
        velocity -= np.sum(geometry.masses[:, None] * velocity, axis=0) / np.sum(
            geometry.masses
        )
        velocity_definition = (
            f"Maxwell-Boltzmann {args.learned_md_temperature} K (angstrom/fs), "
            "zero total momentum"
        )
    else:
        velocity = rng.normal(
            scale=math.sqrt(args.md_temperature), size=geometry.positions.shape
        )
        velocity -= np.mean(velocity, axis=0)
        velocity_definition = f"normal scale sqrt({args.md_temperature}), zero mean"
    initial, _, initialize_phase = measure_kernel(
        "md-initialize",
        (
            dynamics,
            thermodynamic,
            context.positions,
            jnp.asarray(velocity),
            jax.random.key(args.seed),
            context.cell_vectors,
        ),
        args.repeats,
    )
    initialize_phase["successful"] = np.asarray(initial.successful).tolist()
    initialize_phase["rejection_reasons"] = np.asarray(initial.rejection_reasons).tolist()
    phase: dict[str, Any] = {
        "integrator": f"velocity-verlet-nve dt={step_size}",
        "strain_response": False,
        "hamiltonian_terms": list(program.plan.term_names),
        "initial_velocity": velocity_definition,
        "dynamics_prepare": dynamics_phase,
        "thermodynamic_prepare": table_phase,
        "md-initialize": initialize_phase,
    }
    if case.family == "polarization":
        phase["hamiltonian_note"] = (
            "induced-dipole polarization functional only (no permanent-permanent "
            "energy); every energy call solves from the deterministic initial guess"
        )
    if not bool(np.all(np.asarray(initial.successful))):
        phase["status"] = "initial-state-refused"
        return phase
    start = initial.state
    lowering = _KERNELS["md-step"].lower
    executable, compilation = measure_lower_and_compile(
        partial(lowering, dynamics, thermodynamic, start), methodcaller("compile")
    )
    first, first_seconds = measure_synchronized(
        partial(executable, dynamics, thermodynamic, start)
    )
    holder = {"state": first.accepted_state}
    outcomes: list[tuple[bool, Any, float]] = []
    # Device arrays kept without extra synchronization inside the timed loop.
    indices: list[Array] = []
    rebuilt: list[Array] = []

    def advance() -> Any:
        result = executable(dynamics, thermodynamic, holder["state"])
        holder["state"] = result.accepted_state
        return result

    def chained() -> Any:
        result = advance()
        outcomes.append(
            (
                bool(np.all(np.asarray(result.successful))),
                np.asarray(result.rejection_reasons).tolist(),
                float(np.asarray(result.accepted_state.energy.total_energy)),
            )
        )
        indices.append(result.accepted_state.step_index)
        rebuilt.append(result.diagnostics.neighborhood_rebuilt)
        return result

    _, warm = _warm("md-step", chained, args.md_steps)
    energies = [value for _, _, value in outcomes]
    phase.update(
        {
            "status": "measured",
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
            "first_seconds": first_seconds,
            "first_successful": np.asarray(first.successful).tolist(),
            **warm,
            "compiler": _compiled_evidence(executable),
            "executed_chained_steps": len(outcomes),
            "step_accounting": "after the first call: one untimed warmup step, then "
            "--md-steps timed steps, each advancing from the accepted state",
            "accepted_steps": sum(1 for accepted, _, _ in outcomes if accepted),
            "rejection_reasons": [reasons for _, reasons, _ in outcomes],
            "initial_total_energy": _scalar(start.energy.total_energy),
            "total_energy_drift": _scalar(energies[-1] - float(start.energy.total_energy))
            if energies
            else None,
            "retained_state_bytes": _retained(holder["state"]),
            "accepted_step_indices": [np.asarray(value).tolist() for value in indices],
            "rollback_evidence": "a rejected step keeps the previous accepted state; "
            "its accepted step index does not advance",
            "neighborhood_rebuilt": [np.asarray(value).tolist() for value in rebuilt],
            "first_step_term_evidence": _payload(first.potential.term_evidence),
            "first_step_neighborhood_successful": np.asarray(
                first.potential.neighborhood_successful
            ).tolist(),
        }
    )
    return phase


def _workset_points(
    geometry: Geometry, args: argparse.Namespace, batch: int, cells: str, /
) -> tuple[list[np.ndarray], list[np.ndarray] | None, list[dict[str, Any]]]:
    """Item positions/current cells: item 0 is always the row's base point.

    ``equal`` items share the reference cell with deterministic displacements;
    ``distinct`` items carry admitted affine deformations ``I + s_i D`` with
    ``s_i`` spread over ``[0, --cell-strain]`` (the cell-refresh envelope).
    Finite geometries have no cell: their item cells are ``None``.
    """
    cell = geometry.cell
    if cells == "distinct" and cell is None:
        raise ValueError("Distinct current cells need a periodic geometry.")
    rng = np.random.default_rng(args.seed + 17)
    positions: list[np.ndarray] = []
    vectors: list[np.ndarray] | None = None if cell is None else []
    descriptions: list[dict[str, Any]] = []
    for index in range(batch):
        if cells == "equal":
            shift = (
                np.zeros_like(geometry.positions)
                if index == 0
                else args.workset_displacement * rng.normal(size=geometry.positions.shape)
            )
            positions.append(geometry.positions + shift)
            if vectors is not None and cell is not None:
                vectors.append(cell)
            descriptions.append(
                {
                    "cell": "none" if cell is None else "reference",
                    "displacement_rms": float(np.sqrt(np.mean(shift**2))),
                }
            )
        else:
            strain = args.cell_strain * index / max(batch - 1, 1)
            deformation = np.eye(3) + strain * _STRAIN_DIRECTION
            positions.append(geometry.positions @ deformation.T)
            if vectors is not None and cell is not None:
                vectors.append(cell @ deformation.T)
            descriptions.append({"cell": "deformed", "strain": strain})
    return positions, vectors, descriptions


def _measure_workset(
    context: RowContext,
    case: LongRangeCase,
    args: argparse.Namespace,
    geometry: Geometry,
    batch: int,
    cells: str,
    /,
) -> dict[str, Any]:
    """Full-program E/F/S of ``batch`` items through the public workset evaluator."""
    program = context.native.program
    positions, vectors, descriptions = _workset_points(geometry, args, batch, cells)
    requested = [f"{case.method}/{cells}/item-{index:02d}" for index in range(batch)]
    signature = PoolExecutionSignature(
        topology_id=program.layout_id,
        method_id=f"atomistic-long-range-efs:{case.method}",
        precision_id="float64",
        backend_id=jax.default_backend(),
    )
    capacity = (
        batch if args.workset_bucket_capacity is None else args.workset_bucket_capacity
    )
    plan = ExecutionWorksetPlan(
        tuple(requested), (signature,) * batch, bucket_capacity=capacity
    )
    prepared, prepare_phase = _host_phase(plan.prepare)
    # Item-major values follow the plan's canonical semantic order.
    order = [requested.index(value) for value in prepared.plan.semantic_ids]
    stacked_cells = (
        None
        if vectors is None
        else jnp.asarray(np.stack([vectors[index] for index in order]))
    )
    items = WorksetItems(
        program,
        jnp.asarray(np.stack([positions[index] for index in order])),
        stacked_cells,
    )
    # Finite items map positions only; their cells stay None.
    layout = phx.LaneLayout(
        "item", (".positions",) + (() if vectors is None else (".cell_vectors",))
    )
    counters = jnp.zeros((batch,), dtype=jnp.uint32)
    arguments = (
        prepared,
        items,
        layout,
        jax.random.key(args.seed),
        counters,
        case.periodic,
    )
    output, _, phase = measure_kernel("efs-workset", arguments, args.repeats)
    values = output.values
    evidence = output.evidence
    records = []
    for slot, index in enumerate(order):
        point = jnp.asarray(positions[index])
        cell = None if vectors is None else jnp.asarray(vectors[index])
        alone = _evaluate_at_point(program, point, cell, case.periodic)
        energy = float(np.asarray(values["energy"][slot]))
        forces = np.asarray(values["forces"][slot], dtype=np.float64)
        item: dict[str, Any] = {
            "semantic_id": prepared.plan.semantic_ids[slot],
            "requested_index": index,
            **descriptions[index],
            "successful": np.asarray(values["successful"][slot]).tolist(),
            "neighborhood_successful": np.asarray(
                values["neighborhood_successful"][slot]
            ).tolist(),
            "bundle_successful": np.asarray(values["bundle_successful"][slot]).tolist(),
            "capacity_failure": np.asarray(values["capacity_failure"][slot]).tolist(),
            "scientific_failure": np.asarray(values["scientific_failure"][slot]).tolist(),
            "energy": _scalar(energy),
            "standalone_successful": np.asarray(alone.successful).tolist(),
            "standalone_energy_absolute": _scalar(
                abs(energy - float(np.asarray(alone.energy)))
            ),
            "standalone_force_max_absolute": _scalar(
                np.max(np.abs(forces - np.asarray(alone.forces, dtype=np.float64)))
            ),
        }
        if "stress" in values and alone.stress is not None:
            item["standalone_stress_max_absolute"] = _scalar(
                np.max(
                    np.abs(
                        np.asarray(values["stress"][slot], dtype=np.float64)
                        - np.asarray(alone.stress, dtype=np.float64)
                    )
                )
            )
        records.append(item)
    phase["workset"] = {
        "batch": batch,
        "cells": cells,
        "shared": "one prepared program (one admitted preparation) broadcast per "
        "bucket; per-item positions, current cells and neighborhood states",
        "lane_layout": list(layout.mapped_paths),
        "signature_id": signature.signature_id,
        "plan_id": plan.plan_id,
        "prepared_id": prepared.prepared_id,
        "prepare": prepare_phase,
        "requested_ids": requested,
        "semantic_ids": list(prepared.plan.semantic_ids),
        "bucket_capacity": prepared.bucket_capacity,
        "bucket_count": prepared.bucket_count,
        "item_indices": np.asarray(prepared.item_indices).tolist(),
        "valid_mask": np.asarray(prepared.valid_mask).tolist(),
        "active_items": int(np.asarray(evidence.active_item_count)),
        "padded_lanes": int(np.asarray(evidence.padded_lane_count)),
        "finite": bool(np.asarray(evidence.finite)),
        "exact_coverage": bool(np.asarray(evidence.exact_coverage)),
        "successful": bool(np.asarray(evidence.successful)),
        "mode": evidence.mode,
        "next_rng_counters": np.asarray(output.next_rng_counters).tolist(),
    }
    phase["items"] = records
    return phase


def _independent_preparation(
    context: RowContext,
    case: LongRangeCase,
    args: argparse.Namespace,
    geometry: Geometry,
    /,
) -> dict[str, Any]:
    """Whether a program prepared at the deformed reference cell shares the layout."""
    strained = dataclasses.replace(
        geometry, positions=geometry.strained_positions, cell=geometry.strained_cell
    )
    try:
        other = prepare_native(case, strained, args)
    except Exception as error:
        return failure_record(error)
    base = context.native.program
    return {
        "status": "measured",
        "preparation": other.phases,
        "layout_id_equal": base.layout_id == other.program.layout_id,
        "prepared_id_equal": base.prepared_id == other.program.prepared_id,
        "executable_layout": executable_layout((base,), (other.program,)),
        "use": "recorded only; worksets stack items of one shared preparation, "
        "never independently prepared reference-cell programs",
    }


def _efs_workset_workload(
    context: RowContext,
    case: LongRangeCase,
    args: argparse.Namespace,
    record: dict[str, Any],
    cache: ReferenceCache,
    geometry: Geometry,
) -> dict[str, Any]:
    periodic = geometry.cell is not None
    phase: dict[str, Any] = {}
    for batch, cells in itertools.product(args.workset_batches, args.workset_cells):
        label = f"batch-{batch}-{cells}-cells"
        if cells == "distinct" and not periodic:
            phase[label] = {
                "status": "not-applicable",
                "reason": "finite geometry has no cell",
            }
        elif cells == "distinct" and batch < 2:
            phase[label] = {
                "status": "not-applicable",
                "reason": "one item has only the base cell (see the equal-cell batch)",
            }
        else:
            try:
                phase[label] = _measure_workset(
                    context, case, args, geometry, batch, cells
                )
            except Exception as error:
                phase[label] = failure_record(error)
    if periodic and "distinct" in args.workset_cells:
        phase["independent_reference_cell_preparation"] = _independent_preparation(
            context, case, args, geometry
        )
    failed = [
        name
        for name, value in phase.items()
        if value.get("status") not in (None, "measured", "not-applicable")
    ]
    phase["status"] = "partial" if failed else "measured"
    return phase


def _tree_norm(tree: Any, /) -> float | dict[str, str]:
    leaves = [np.asarray(leaf, dtype=np.float64) for leaf in jax.tree.leaves(tree)]
    return _scalar(math.sqrt(sum(float(np.sum(leaf * leaf)) for leaf in leaves)))


def _parameter_gradient_workload(
    context: RowContext,
    case: LongRangeCase,
    args: argparse.Namespace,
    record: dict[str, Any],
    cache: ReferenceCache,
    geometry: Geometry,
) -> dict[str, Any]:
    learned = context.native.learned
    if learned is None or context.cell_vectors is None:
        return {
            "status": "not-applicable",
            "reason": "only learned-charge rows carry trainable model parameters",
        }
    _ensure_neighborhoods(context, args, record)
    lane, state, fixed = partition_parameters(learned.term.potential)
    arguments = (
        lane,
        state,
        fixed,
        context.native.program,
        context.positions,
        context.neighborhoods,
        context.cell_vectors,
        learned.learned_index,
        learned.coulomb_index,
    )
    output, _, phase = measure_kernel("parameter-gradient", arguments, args.repeats)
    (loss, gradient), (coulomb, coulomb_gradient) = output
    phase["outputs"] = {
        "loss": _scalar(loss),
        "loss_definition": "E^2 + sum F^2 + sum S^2 of the connected program",
        "loss_parameter_gradient_norm": _tree_norm(gradient),
        "coulomb_term_energy": _scalar(coulomb),
        "coulomb_parameter_gradient_norm": _tree_norm(coulomb_gradient),
        "parameter_count": int(sum(np.size(leaf) for leaf in jax.tree.leaves(lane))),
    }
    phase["chain_rule"] = (
        "the Coulomb term depends on model parameters only through the predicted "
        "charge field, so its parameter gradient is the charge chain rule into "
        "training"
    )
    return phase


_WORKLOAD_FUNCTIONS: dict[str, Workload] = {
    "energy": _energy_workload,
    "efs": _efs_workload,
    "hessian-vector": _hessian_vector_workload,
    "coefficient-refresh": _coefficient_refresh_workload,
    "cell-refresh": _cell_refresh_workload,
    "md-step": _md_workload,
    "efs-workset": _efs_workset_workload,
    "parameter-gradient": _parameter_gradient_workload,
}


def _reciprocal_resources(program: Any, /) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for leaf in jax.tree_util.tree_leaves(
        program, is_leaf=lambda value: isinstance(value, ReciprocalResourceEvidence)
    ):
        if isinstance(leaf, ReciprocalResourceEvidence):
            found[leaf.evidence_id] = _payload(leaf)
    return list(found.values())


def _identities(native: NativeBinding, /) -> dict[str, Any]:
    program = native.program
    return {
        "program_definition_id": native.definition.program_id,
        "scientific_id": program.scientific_id,
        "prepared_id": program.prepared_id,
        "layout_id": program.layout_id,
        "system_layout_id": program.system.layout_id,
        "terms": [
            {
                "name": term.name,
                "type": type(term).__name__,
                "term_id": term.term_id,
                "prepared_id": term.prepared_id,
                "layout_id": term.layout_id,
            }
            for term in program.terms
        ],
    }


def _estimates(
    case: LongRangeCase, geometry: Geometry, args: argparse.Namespace, /
) -> dict[str, Any]:
    """Native heuristic splitting estimates for the cell; recorded, never used."""
    if case.family != "point" or case.route not in ("ewald", "pme"):
        return {}
    if geometry.cell is None:
        raise ValueError("Splitting estimates need the periodic cell of the row.")
    cell = PeriodicCell(geometry.cell, periodic_axes=geometry.periodic_axes)
    try:
        estimate = (
            estimate_ewald_splitting(cell, args.real_cutoff, args.estimate_tolerance)
            if case.route == "ewald"
            else estimate_pme_splitting(
                cell,
                args.real_cutoff,
                args.estimate_tolerance,
                spline_degree=_spline_degree(case),
            )
        )
        return {"unique_image_radius": cell.unique_image_radius, **_payload(estimate)}
    except Exception as error:
        return failure_record(error)


def measure_case(
    case: LongRangeCase,
    args: argparse.Namespace,
    cache: ReferenceCache,
    /,
) -> dict[str, Any]:
    """Complete row: geometry, preparation, identities and every workload."""
    row: dict[str, Any] = {"case": case.to_dict()}
    try:
        if case.family == "learned":
            geometry = build_water_geometry(
                case.atoms, strain=args.cell_strain, seed=args.seed
            )
        else:
            geometry = build_geometry(
                case.method,
                case.atoms,
                case.density,
                case.cell_shape,
                charge=args.charge,
                jitter=args.jitter,
                slab_vacuum=args.slab_vacuum,
                strain=args.cell_strain,
                gaussian_widths=args.gaussian_widths,
                polarizabilities=args.polarizabilities,
                seed=args.seed,
            )
    except Exception as error:
        row.update(status="geometry-refused", geometry=failure_record(error))
        return row
    row["geometry"] = geometry.summary()
    learned = case.family == "learned"
    row["reciprocal"] = {
        "extents": None if case.reciprocal_extent is None else list(_extents(case, args)),
        "grid_shape": None if case.grid is None else list(_grid(case, args)),
        "alpha": args.learned_alpha if learned else args.alpha,
        "real_cutoff": args.learned_real_cutoff if learned else args.real_cutoff,
        "dsf_alpha": args.dsf_alpha if case.method == "dsf" else None,
        "declared_reciprocal_cutoff": None if learned else args.reciprocal_cutoff,
    }
    if case.family == "multipole" and case.route == "pme":
        classes = 1 if _site_text(case.method) == "point" else len(geometry.class_widths)
        row["reciprocal"]["mesh_channels"] = {
            "channels": 10 * classes,
            "definition": "degree-two Cartesian channels (1 + 3 + 6) per Gaussian "
            "width class",
            "channel_chunk": case.channel_chunk,
        }
    if case.real_space is not None:
        row["real_space"] = {
            "route": case.real_space,
            "resources": "term-owned IMAGE_PAIR_SUM requirement served by the "
            "declared cell-list image source (see neighborhood_resources)"
            if case.real_space == "neighborhood"
            else "bounded explicit image sum (maximum_image_count "
            f"{args.maximum_image_count}, row_block "
            f"{32 if case.row_block is None else case.row_block}); quadratic in atoms",
        }
    if case.stability is not None:
        row["stability"] = {
            "certificate": case.stability,
            "cost": "dense eigensolve of the materialized (3N)^2 operator on every "
            "evaluation, included in every timed workload"
            if case.stability == "dense-spectrum"
            else "no proof; reported stability_assumed=True, detected "
            "indefiniteness still refuses",
        }
    row["estimates"] = _estimates(case, geometry, args) if case.periodic else {}
    try:
        native = prepare_native(case, geometry, args)
    except Exception as error:
        row.update(status="preparation-refused", preparation=failure_record(error))
        return row
    row["preparation"] = native.phases
    row["neighborhood_resources"] = native.capacities
    row["identities"] = _identities(native)
    row["reciprocal_resources"] = _reciprocal_resources(native.program)
    context = RowContext(
        native,
        jnp.asarray(geometry.positions),
        None if geometry.cell is None else jnp.asarray(geometry.cell),
    )
    workloads: dict[str, Any] = {}
    for name in args.workloads:
        try:
            value = _WORKLOAD_FUNCTIONS[name](
                context, case, args, workloads, cache, geometry
            )
        except Exception as error:
            value = failure_record(error)
        if value is not None:
            workloads[name] = value
    row["workloads"] = workloads
    statuses = [
        item.get("status")
        for item in workloads.values()
        if isinstance(item, Mapping)
        and item.get("status") not in ("measured", "not-applicable", None)
    ]
    row["status"] = "measured" if not statuses else "partial"
    return row


def _exponents(rows: Sequence[Mapping[str, Any]], /) -> dict[str, float | None]:
    """Log-log slope of efs compiler temporary bytes against atoms per configuration."""
    groups: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        case = dict(row["case"])
        atoms = case.pop("atoms")
        efs = row.get("workloads", {}).get("efs")
        temporary = (
            None
            if not isinstance(efs, Mapping)
            else efs.get("compiler", {}).get("temporary_bytes")
        )
        key = ",".join(f"{name}={case[name]}" for name in sorted(case))
        if temporary:
            groups.setdefault(key, []).append((atoms, temporary))
    return {
        key: fitted_scaling_exponent([a for a, _ in values], [b for _, b in values])
        if len({a for a, _ in values}) >= 2
        else None
        for key, values in sorted(groups.items())
    }


# ------------------------------------------------------------------ CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add(
        "--methods", nargs="+", choices=METHODS, default=["direct", "dsf", "ewald", "pme"]
    )
    add(
        "--workloads",
        nargs="+",
        choices=WORKLOADS,
        default=list(DEFAULT_WORKLOADS),
        help="efs-workset and parameter-gradient are opt-in selections.",
    )
    add("--atoms", nargs="+", type=int, default=[32, 64])
    add("--densities", nargs="+", type=float, default=[0.05])
    add("--cell-shapes", nargs="+", choices=tuple(_SHAPES), default=["triclinic"])
    add("--capacity-factors", nargs="+", type=float, default=[2.0])
    add("--ewald-extents", nargs="+", type=int, default=[8])
    add("--mode-chunks", nargs="+", type=int, default=[128])
    add("--pme-grids", nargs="+", type=int, default=[24])
    add("--spline-degrees", nargs="+", type=int, default=[4])
    add("--channel-chunks", nargs="+", type=int, default=[8])
    add("--row-blocks", nargs="+", type=int, default=[32])
    add(
        "--multipole-real-space",
        nargs="+",
        choices=REAL_SPACE_ROUTES,
        default=["explicit-image-reference"],
        help="Recorded real-space route axis of periodic multipole/polarization rows.",
    )
    add(
        "--polarization-stability",
        nargs="+",
        choices=STABILITY_POLICIES,
        default=["dense-spectrum"],
        help="Stability certificate axis of periodic polarization rows.",
    )
    add("--learned-molecules", nargs="+", type=int, default=[8])
    add("--max-rows", type=int, default=64)
    add("--alpha", type=float, default=1.0)
    add("--real-cutoff", type=float, default=3.2)
    add("--dsf-alpha", type=float, default=0.3)
    add("--reciprocal-cutoff", type=float, default=None)
    add("--skin", type=float, default=0.4)
    add("--deformation-margin", type=float, default=0.05)
    add("--candidate-slots", type=int, default=None)
    add("--estimate-tolerance", type=float, default=1.0e-6)
    add("--max-grid-points", type=int, default=1 << 24)
    add("--max-modes", type=int, default=1 << 22)
    add("--max-chunk-workspace-bytes", type=int, default=1 << 30)
    add("--max-retained-bytes", type=int, default=1 << 31)
    add("--multipole-workspace-bytes", type=int, default=1 << 28)
    add("--maximum-image-count", type=int, default=4096)
    add("--native-comparison-extent", type=int, default=12)
    add(
        "--no-native-comparison",
        dest="native_comparison",
        action="store_false",
        help="Skip the non-independent native direct-k comparison (recorded).",
    )
    add("--gaussian-widths", nargs=2, type=float, default=[0.4, 0.6])
    add("--polarizabilities", nargs=2, type=float, default=[0.05, 0.08])
    add("--polarization-tolerance", type=float, default=1.0e-10)
    add("--polarization-iterations", type=int, default=100)
    add("--tcg-order", type=int, default=4)
    add("--charge", type=float, default=1.0)
    add("--jitter", type=float, default=0.15)
    add("--slab-vacuum", type=float, default=3.0)
    add("--cell-strain", type=float, default=0.01)
    add("--md-step-size", type=float, default=1.0e-3)
    add("--md-steps", type=int, default=8)
    add("--md-temperature", type=float, default=0.05)
    add("--reference-max-atoms", type=int, default=128)
    add("--reference-tolerance", type=float, default=1.0e-12)
    add("--reference-max-level", type=int, default=6)
    add("--moment-reference-tolerance", type=float, default=1.0e-10)
    add("--moment-reference-max-shells", type=int, default=24)
    add("--moment-reference-mode-chunk", type=int, default=512)
    add("--moment-reference-max-atoms", type=int, default=64)
    add("--fd-step", type=float, default=1.0e-5)
    add("--workset-batches", nargs="+", type=int, default=[1, 2])
    add("--workset-cells", nargs="+", choices=WORKSET_CELLS, default=list(WORKSET_CELLS))
    add(
        "--workset-bucket-capacity",
        type=int,
        default=None,
        help="Fixed workset bucket capacity (default: the batch, no padding).",
    )
    add("--workset-displacement", type=float, default=0.02)
    add("--learned-seed", type=int, default=3)
    add("--learned-alpha", type=float, default=1.3)
    add("--learned-real-cutoff", type=float, default=2.6)
    add("--learned-graph-degree", type=int, default=128)
    add("--learned-search-margin", type=float, default=0.1)
    add("--learned-total-charge", type=float, default=0.0)
    add("--learned-refresh-charge", type=float, default=1.0)
    add("--learned-md-step-size", type=float, default=0.5)
    add("--learned-md-temperature", type=float, default=300.0)
    add("--repeats", type=int, default=3)
    add("--seed", type=int, default=0)
    add(
        "--unattested-probe",
        action="store_true",
        help="Record an explicitly unattested probe when the canonical source/build "
        "identity cannot be captured; the default fails closed.",
    )
    add("--label", required=True)
    add("--output", type=Path, required=True)
    return parser


def _validate(args: argparse.Namespace, /) -> None:
    positive_lists = (
        args.atoms,
        args.densities,
        args.capacity_factors,
        args.ewald_extents,
        args.mode_chunks,
        args.pme_grids,
        args.spline_degrees,
        args.channel_chunks,
        args.row_blocks,
        args.learned_molecules,
        args.workset_batches,
    )
    if any(value <= 0 for values in positive_lists for value in values):
        raise ValueError("Every swept axis value must be positive.")
    if args.repeats < 1 or args.md_steps < 1 or args.max_rows < 1:
        raise ValueError("repeats, md-steps and max-rows must be positive.")
    if args.fd_step <= 0.0 or args.reference_tolerance <= 0.0:
        raise ValueError("fd-step and reference-tolerance must be positive.")
    if (
        args.moment_reference_tolerance <= 0.0
        or args.moment_reference_max_shells < 1
        or args.moment_reference_mode_chunk < 1
        or args.moment_reference_max_atoms < 1
    ):
        raise ValueError("Moment reference bounds must be positive.")
    if args.workset_bucket_capacity is not None and not (
        1 <= args.workset_bucket_capacity <= 64
    ):
        raise ValueError("--workset-bucket-capacity must lie in [1, 64].")
    if max(args.workset_batches) > 64 and args.workset_bucket_capacity is None:
        raise ValueError("Batches above 64 need an explicit --workset-bucket-capacity.")
    if (
        args.learned_alpha <= 0.0
        or args.learned_real_cutoff <= 0.0
        or args.learned_graph_degree < 1
        or args.learned_md_step_size <= 0.0
        or args.learned_md_temperature <= 0.0
        or args.workset_displacement < 0.0
    ):
        raise ValueError("Learned/workset settings must be positive.")


EVIDENCE_FIELDS = (
    "cases",
    "configuration",
    "conventions",
    "environment",
    "exponents",
    "qualification_boundaries",
    "source_sha256",
)


def main() -> None:
    args = _parser().parse_args()
    _validate(args)
    phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
    root = Path(__file__).resolve().parents[1]
    try:
        identity: dict[str, Any] = capture_benchmark_identity(
            root, Path(__file__), EVIDENCE_FIELDS
        ).to_dict()
        attestation: dict[str, Any] = {"status": "attested"}
    except FileNotFoundError as error:
        if not args.unattested_probe:
            raise
        identity = {"status": "unavailable"}
        attestation = {
            "status": "unattested-probe",
            "capture_error": f"{type(error).__name__}: {error}",
        }
    cases = long_range_cases(
        args.methods,
        args.atoms,
        args.densities,
        args.cell_shapes,
        capacity_factors=args.capacity_factors,
        reciprocal_extents=args.ewald_extents,
        mode_chunks=args.mode_chunks,
        grids=args.pme_grids,
        spline_degrees=args.spline_degrees,
        channel_chunks=args.channel_chunks,
        row_blocks=args.row_blocks,
        maximum_rows=args.max_rows,
        real_spaces=args.multipole_real_space,
        stabilities=args.polarization_stability,
        learned_molecules=args.learned_molecules,
    )
    cache: ReferenceCache = {}
    rows = []
    for case in cases:
        rows.append(measure_case(case, args, cache))
        jax.clear_caches()
        gc.collect()
    configuration = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in sorted(vars(args).items())
    }
    record = {
        "benchmark": "native-atomistic-long-range-electrostatics",
        "label": args.label,
        "attestation": attestation,
        "environment": capture_environment().to_dict(),
        "identity": identity,
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in _OWNERS
        },
        "configuration": configuration,
        "conventions": {
            "units": "AtomisticUnitSystem.reduced(); coulomb_constant "
            f"{AtomisticUnitSystem.reduced().coulomb_constant}; learned rows use "
            "eV/angstrom/elementary charge (coulomb_constant "
            f"{WATER_UNITS.coulomb_constant})",
            "precision": "float64 (x64 enabled)",
            "periodic": "tin-foil boundary, require-neutral, k=0 excluded; slab rows "
            "use slab-yeh-berkowitz over the full embedding lattice; learned rows "
            "declare uniform-background for the predicted-charge residual",
            "stress": "tension-positive sym(dE/dstrain)/|det H|",
            "multipoles": "site-centered raw traceful Cartesian second moments, "
            "operator q - mu.grad + 1/2 Theta:grad grad, intrasite excluded; "
            "periodic rows record their real-space route",
            "polarization": "undamped (damping='none') induced dipoles; PCG "
            "converged, TCG fixed order; induced functional only (no permanent-"
            "permanent energy); periodic rows record their stability certificate "
            "(dense-spectrum by default)",
            "worksets": "public ExecutionWorksetPlan + "
            "evaluate_execution_worksets_filter_vmap over one shared preparation; "
            "item lane = positions and current cells",
            "memory": "compiler bytes are XLA estimates; sampled peaks are "
            "observations of the warm phase, never bounds; retained bytes are "
            "logical array payloads",
            "cache_policy": "fresh lowering per row; jax in-memory caches cleared "
            "between rows; persistent cache per the recorded environment",
            "timing": "cases run sequentially in one process on the recorded devices",
        },
        "cases": rows,
        "exponents": {"efs_compiler_temporary_bytes_vs_atoms": _exponents(rows)},
        "qualification_boundaries": list(QUALIFICATION_BOUNDARIES),
    }
    write_json_atomic(args.output, record)
    print(args.output)


__all__ = [
    "DEFAULT_WORKLOADS",
    "EVIDENCE_FIELDS",
    "Geometry",
    "LearnedBinding",
    "LongRangeCase",
    "METHODS",
    "NativeBinding",
    "QUALIFICATION_BOUNDARIES",
    "WORKLOADS",
    "WorksetItems",
    "build_geometry",
    "build_water_geometry",
    "executable_layout",
    "independent_reference",
    "long_range_cases",
    "measure_case",
    "measure_kernel",
    "prepare_native",
]


if __name__ == "__main__":
    sys.exit(main())
