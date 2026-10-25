#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Point-charge electrostatic energy terms with separate scientific identities.

* :class:`DirectCoulombPotential`: finite (nonperiodic) direct Coulomb over the
  supplied pair relation, with alchemical soft-core coupling.
* :class:`ScreenedCoulombPotential`: Yukawa (Debye-Hueckel) screened Coulomb
  ``exp(-kappa r) / r`` with an explicit hard, shifted-potential or
  shifted-force truncation at its cutoff.
* :class:`DampedShiftedForceCoulombPotential`: Fennell-Gezelter damped
  shifted-force (DSF) Coulomb with its self term.
* :class:`EwaldReferencePotential`: periodic Ewald with explicit canonical
  half-space reciprocal modes evaluated in bounded mode chunks.
* :class:`ParticleMeshEwaldPotential`: smooth particle-mesh Ewald on the
  term-owned :class:`~phydrax.atomistic._reciprocal_mesh.PreparedReciprocalMesh`
  with a declared assignment modulus and spline degree 1-5.

Periodic Coulomb always declares a neutrality convention (``require-neutral``
refuses a charged cell; ``uniform-background`` adds ``-pi Q^2 / (2 alpha^2 V)``)
and a macroscopic boundary convention: ``tin-foil`` (conducting), ``vacuum-
sphere`` (``2 pi |M|^2 / (3 V)`` from the explicit unwrapped dipole), or
``slab-yeh-berkowitz``: the three-dimensional sum over the full embedding cell
with the Yeh-Berkowitz dipole correction and the Ballenegger-Arnold-Cerda
charged-slab terms ``(2 pi / V) [M_z^2 - Q sum q z^2 - Q^2 L^2 / 12]`` for a
cell with exactly two periodic axes, ``L = |det H| / |a x b|`` the projected
height and ``z = s_c L`` from the nonperiodic fractional coordinate. Every
``k = 0`` term is excluded from the reciprocal sum and owned by these
conventions.

Charges are either the prepared system charges or an explicit
:class:`~phydrax.atomistic._conditioning.AtomisticFieldReference` to atom-level
charges produced inside the same scalar energy evaluation, so every coordinate,
cell and parameter derivative includes the charge response. Reciprocal
coefficients use the runtime cell. Numerical evidence (charge availability,
neutrality, native cell solve, declared mode coverage, mesh status and energy
components) is returned as a typed record; heuristic parameter estimates are
reported as estimates, never as error certificates.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import PeriodicCell
from ..typing import parse
from ._conditioning import (
    AtomisticConditionLevel,
    AtomisticFieldReference,
    AtomisticObservable,
)
from ._neighborhood_bundle import (
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticPairExclusions,
)
from ._potential import AtomisticPotentialCapabilities, AtomisticPotentialRequirements
from ._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialContext,
    AtomisticTermEvaluation,
)
from ._reciprocal_mesh import (
    fft_friendly_mesh_size,
    PreparedReciprocalMesh,
    PreparedReciprocalModeSet,
    ReciprocalBudget,
    ReciprocalGeometry,
    ReciprocalInfluence,
    ReciprocalMeshModulus,
    ReciprocalMeshPlan,
    ReciprocalModeSetPlan,
    ReciprocalNyquistPolicy,
    ReciprocalQuadratic,
)
from ._sites import AtomisticSiteDomain
from ._system import PreparedAtomisticSystem


ChargeNeutralityPolicy: TypeAlias = Literal["require-neutral", "uniform-background"]
EwaldBoundaryConvention: TypeAlias = Literal[
    "tin-foil", "vacuum-sphere", "slab-yeh-berkowitz"
]
EwaldRealSpaceRole: TypeAlias = Literal["minimum-image", "explicit-image"]
ScreenedTruncation: TypeAlias = Literal["hard", "shifted-potential", "shifted-force"]

_SQRT_PI = math.sqrt(math.pi)
_SYSTEM_CHARGES = "system-charges"


def _electrostatic_identity(kind: str, values: dict) -> IdentityWords:
    return IdentityWords.of({"kind": kind, **values})


def _term_identity(name: str, force_group: int, /) -> tuple[str, int]:
    identifier = str(name).strip()
    group = int(force_group)
    if not identifier or group < 0:
        raise ValueError("Term name must be non-empty and force_group non-negative.")
    return identifier, group


def _finite_positive(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _checked_charge_field(
    reference: AtomisticFieldReference | None, /
) -> AtomisticFieldReference | None:
    """Admit an atom-level scalar charge dependency with an explicit unit."""
    if reference is None:
        return None
    if not isinstance(reference, AtomisticFieldReference):
        raise TypeError("charge_field must be an AtomisticFieldReference or None.")
    if (
        reference.observable is not AtomisticObservable.CHARGES
        or reference.level is not AtomisticConditionLevel.ATOM
        or tuple(reference.port.event_shape) != ()
        or reference.unit is None
    ):
        raise ValueError(
            "charge_field must reference atom-level scalar CHARGES with an explicit unit."
        )
    return reference


def _require_charge_unit(
    reference: AtomisticFieldReference | None, system: PreparedAtomisticSystem, /
) -> None:
    if reference is None:
        return
    unit = reference.unit
    if unit is None or unit.unit_id != system.plan.units.charge_unit.unit_id:
        raise ValueError(
            "charge_field unit must equal the system charge unit of the Coulomb constant."
        )


def _field_dependencies(
    reference: AtomisticFieldReference | None, /
) -> tuple[AtomisticFieldReference, ...]:
    return () if reference is None else (reference,)


def _charge_source(reference: AtomisticFieldReference | None, /) -> str:
    return _SYSTEM_CHARGES if reference is None else reference.reference_id


class _BoundCharges(StrictModule):
    values: Array
    available: Array
    successful: Array


def _bind_charges(
    system: PreparedAtomisticSystem,
    reference: AtomisticFieldReference | None,
    context: AtomisticPotentialContext,
    /,
) -> _BoundCharges:
    """Active-masked charges for this evaluation and their source status.

    Predicted charges come from ``context.field`` inside the energy closure and
    are never detached. Unavailable or failed charges are zeroed only to keep
    the numerical graph finite; the term then reports failure.
    """
    dtype = context.positions.dtype
    if reference is None:
        values = system.plan.charges.astype(dtype)
        available = jnp.asarray(True)
        successful = jnp.asarray(True)
    else:
        field = context.field(reference)
        if field.values.shape != (system.capacity,) or field.available.shape != (1,):
            raise ValueError(
                "Predicted charges must be one atom-level case of the system capacity."
            )
        values = field.values.astype(dtype)
        available = field.available[0]
        successful = field.successful[0]
    usable = system.active_mask & available & successful & jnp.isfinite(values)
    return _BoundCharges(
        values=jnp.where(usable, values, jnp.zeros((), dtype=dtype)),
        available=available,
        successful=successful & jnp.all(~system.active_mask | jnp.isfinite(values)),
    )


class PairCoulombEvidence(StrictModule):
    """Evidence of one finite-range pair Coulomb evaluation.

    ``prepared_identity`` is the evaluated prepared term's dynamic digest
    (``prepared_id`` decodes it on the host).
    """

    method: str = eqx.field(static=True)
    charge_source: str = eqx.field(static=True)
    prepared_identity: IdentityWords
    charges_available: Array
    charges_successful: Array
    geometry_successful: Array
    total_charge: Array
    pair_energy: Array
    self_energy: Array

    @property
    def prepared_id(self) -> str:
        """Host decoding of the evaluated prepared-term digest."""
        return self.prepared_identity.fingerprint()


class PeriodicCoulombEvidence(StrictModule):
    """Evidence of one Ewald-split periodic Coulomb evaluation.

    Status fields are separate: charge source availability/success, neutrality
    under the declared policy, the native runtime cell solve/measure, pair and
    slab geometry, declared reciprocal mode coverage under the current cell
    (``support_radius`` versus ``declared_reciprocal_cutoff``; ``None`` makes
    no coverage claim) and the reciprocal execution status. The energy
    components are reported individually. ``prepared_identity`` is the
    evaluated prepared term's dynamic digest (``prepared_id`` decodes it on
    the host).
    """

    method: str = eqx.field(static=True)
    boundary: str = eqx.field(static=True)
    neutrality: str = eqx.field(static=True)
    real_space: str = eqx.field(static=True)
    charge_source: str = eqx.field(static=True)
    declared_reciprocal_cutoff: float | None = eqx.field(static=True)
    prepared_identity: IdentityWords
    charges_available: Array
    charges_successful: Array
    total_charge: Array
    neutral: Array
    cell_successful: Array
    geometry_successful: Array
    mode_coverage_successful: Array
    support_radius: Array
    reciprocal_successful: Array
    volume: Array
    real_energy: Array
    reciprocal_energy: Array
    self_energy: Array
    exception_energy: Array
    background_energy: Array
    boundary_energy: Array

    @property
    def prepared_id(self) -> str:
        """Host decoding of the evaluated prepared-term digest."""
        return self.prepared_identity.fingerprint()


class CoulombSplittingEstimate(StrictModule, NonTrainableState):
    """Heuristic Ewald splitting parameters; an estimate, never a certificate.

    ``estimated_real_relative_error = erfc(alpha r_c)`` and
    ``estimated_reciprocal_relative_error = exp(-k_c^2 / (4 alpha^2))`` are the
    Gaussian tail factors of the omitted real and reciprocal terms. Mesh
    interpolation error is not bounded by them; measured reference
    discrepancies are separate evidence.
    """

    alpha: float = eqx.field(static=True)
    real_cutoff: float = eqx.field(static=True)
    reciprocal_cutoff: float = eqx.field(static=True)
    reciprocal_extent: tuple[int, int, int] | None = eqx.field(static=True)
    grid_shape: tuple[int, int, int] | None = eqx.field(static=True)
    spline_degree: int | None = eqx.field(static=True)
    target_tolerance: float = eqx.field(static=True)
    estimated_real_relative_error: float = eqx.field(static=True)
    estimated_reciprocal_relative_error: float = eqx.field(static=True)
    estimator: str = eqx.field(static=True)
    certified: bool = eqx.field(static=True)
    estimate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        alpha: float,
        real_cutoff: float,
        reciprocal_cutoff: float,
        reciprocal_extent: tuple[int, int, int] | None,
        grid_shape: tuple[int, int, int] | None,
        spline_degree: int | None,
        target_tolerance: float,
        estimator: str,
    ) -> None:
        self.alpha = float(alpha)
        self.real_cutoff = float(real_cutoff)
        self.reciprocal_cutoff = float(reciprocal_cutoff)
        self.reciprocal_extent = reciprocal_extent
        self.grid_shape = grid_shape
        self.spline_degree = spline_degree
        self.target_tolerance = float(target_tolerance)
        self.estimated_real_relative_error = math.erfc(self.alpha * self.real_cutoff)
        self.estimated_reciprocal_relative_error = math.exp(
            -((self.reciprocal_cutoff / (2.0 * self.alpha)) ** 2)
        )
        self.estimator = str(estimator)
        self.certified = False
        self.estimate_id = canonical_fingerprint(
            {
                "kind": "coulomb-splitting-estimate",
                "alpha": self.alpha.hex(),
                "real_cutoff": self.real_cutoff.hex(),
                "reciprocal_cutoff": self.reciprocal_cutoff.hex(),
                "reciprocal_extent": None
                if reciprocal_extent is None
                else list(reciprocal_extent),
                "grid_shape": None if grid_shape is None else list(grid_shape),
                "spline_degree": spline_degree,
                "target_tolerance": self.target_tolerance.hex(),
                "estimator": self.estimator,
            }
        )


def _splitting(
    cell: PeriodicCell, real_cutoff: float, tolerance: float, /
) -> tuple[float, float, float, np.ndarray]:
    if not isinstance(cell, PeriodicCell) or cell.rank != 3:
        raise ValueError("Splitting estimates require a rank-3 PeriodicCell.")
    cutoff = _finite_positive(real_cutoff, "real_cutoff")
    target = float(tolerance)
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("tolerance must lie strictly between zero and one.")
    log_target = -math.log(target)
    alpha = math.sqrt(log_target) / cutoff
    reciprocal = 2.0 * alpha * math.sqrt(log_target)
    lengths = np.linalg.norm(np.asarray(cell.vectors, dtype=np.float64), axis=1)
    return cutoff, alpha, reciprocal, lengths


def estimate_ewald_splitting(
    cell: PeriodicCell, real_cutoff: float, tolerance: float, /
) -> CoulombSplittingEstimate:
    """Gaussian-tail heuristic ``alpha``, reciprocal cutoff and covering mode extents."""
    cutoff, alpha, reciprocal, lengths = _splitting(cell, real_cutoff, tolerance)
    needed = reciprocal * lengths / (2.0 * math.pi)
    extents = tuple(max(1, math.ceil(value - 1.0)) for value in needed)
    return CoulombSplittingEstimate(
        alpha=alpha,
        real_cutoff=cutoff,
        reciprocal_cutoff=reciprocal,
        reciprocal_extent=(extents[0], extents[1], extents[2]),
        grid_shape=None,
        spline_degree=None,
        target_tolerance=float(tolerance),
        estimator="gaussian-tail-heuristic",
    )


def estimate_pme_splitting(
    cell: PeriodicCell,
    real_cutoff: float,
    tolerance: float,
    /,
    *,
    spline_degree: int = 4,
) -> CoulombSplittingEstimate:
    """Gaussian-tail ``alpha`` plus a covering, FFT-friendly mesh.

    Each axis keeps the reciprocal cutoff inside the Nyquist-excluded support
    and is at least the common ``2 alpha |a_i| / (3 tol^(1/5))`` interpolation
    heuristic; neither is an interpolation-error bound.
    """
    degree = int(spline_degree)
    if not 1 <= degree <= 5:
        raise ValueError("spline_degree must be one to five.")
    cutoff, alpha, reciprocal, lengths = _splitting(cell, real_cutoff, tolerance)
    shape = []
    for length in lengths:
        maximum_mode = max(1, math.ceil(reciprocal * length / (2.0 * math.pi) - 1.0))
        covering = 2 * maximum_mode + 2
        heuristic = 2.0 * alpha * length / (3.0 * float(tolerance) ** 0.2)
        shape.append(
            fft_friendly_mesh_size(max(float(covering), heuristic, 4.0, degree + 1.0))
        )
    return CoulombSplittingEstimate(
        alpha=alpha,
        real_cutoff=cutoff,
        reciprocal_cutoff=reciprocal,
        reciprocal_extent=None,
        grid_shape=(shape[0], shape[1], shape[2]),
        spline_degree=degree,
        target_tolerance=float(tolerance),
        estimator="gaussian-tail-and-mesh-heuristic",
    )


class DirectCoulombPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    """Finite direct Coulomb over every pair of the supplied pair relation.

    The neighborhood must contain every interacting pair; there is no cutoff.
    Pair exception scales and alchemical soft-core coupling apply per pair.
    The canonical term digest is the dynamic ``term_identity``.
    """

    charge_field: AtomisticFieldReference | None
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        *,
        charge_field: AtomisticFieldReference | None = None,
        name: str = "direct-coulomb",
        force_group: int = 0,
    ) -> None:
        identifier, group = _term_identity(name, force_group)
        reference = _checked_charge_field(charge_field)
        self.charge_field = reference
        self.name = identifier
        self.force_group = group
        self.capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True,
            finite_geometry=True,
            local_energy=True,
            local_energy_delta=True,
            dynamic_species=True,
        )
        self.requirements = AtomisticPotentialRequirements(
            neighborhoods=(
                AtomisticNeighborhoodRequirement(
                    identifier,
                    AtomisticNeighborhoodRole.CLASSICAL_MINIMUM_IMAGE_PAIRS,
                    None,
                    exclusions=AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS,
                ),
            ),
            field_dependencies=_field_dependencies(reference),
        )
        self.term_identity = _electrostatic_identity(
            "direct-coulomb-potential",
            {
                "name": identifier,
                "force_group": group,
                "charges": _charge_source(reference),
            },
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedDirectCoulombPotential":
        if system.cell is not None:
            raise ValueError("DirectCoulombPotential is finite and nonperiodic.")
        _require_charge_unit(self.charge_field, system)
        return PreparedDirectCoulombPotential(self, system)


class PreparedDirectCoulombPotential(AbstractPreparedAtomisticEnergyTerm):
    """Direct Coulomb bound to one system; ``prepared_identity`` is dynamic.

    A context of another prepared system fails the evaluation numerically.
    """

    plan: DirectCoulombPotential
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self, plan: DirectCoulombPotential, system: PreparedAtomisticSystem, /
    ) -> None:
        self.plan = plan
        self.system = system
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_identity = _electrostatic_identity(
            "prepared-direct-coulomb",
            {"term": plan.term_id, "system": system.prepared_id},
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        bound = _context_bound(context, self.system)
        charges = _bind_charges(self.system, self.plan.charge_field, context)
        distance = context.pair_distance
        coupling = context.interaction_scales.electrostatic
        alpha = context.interaction_scales.electrostatic_softcore_alpha
        softened = (
            distance * distance
            + alpha
            * jnp.clip(1.0 - coupling, 0.0, 1.0)
            ** context.interaction_scales.softcore_power
        )
        valid_geometry = jnp.all(
            ~context.pair_valid | (distance > 0.0) | ((coupling < 1.0) & (alpha > 0.0))
        )
        safe = jnp.where(context.pair_valid & (softened > 0.0), softened, 1.0)
        q = charges.values
        pair = (
            coupling
            * self.system.plan.units.coulomb_constant
            * q[context.pair_left]
            * q[context.pair_right]
            / jnp.sqrt(safe)
            * context.electrostatic_scales
        )
        pair = jnp.where(context.pair_valid, pair, 0.0)
        atom_energy = _split_pairs(self.system.capacity, context, pair)
        total = jnp.sum(pair)
        successful = (
            bound
            & valid_geometry
            & jnp.all(jnp.isfinite(pair))
            & charges.available
            & charges.successful
        )
        evidence = PairCoulombEvidence(
            method="direct-coulomb",
            charge_source=_charge_source(self.plan.charge_field),
            prepared_identity=self.prepared_identity,
            charges_available=charges.available,
            charges_successful=charges.successful,
            geometry_successful=valid_geometry,
            total_charge=jnp.sum(q),
            pair_energy=total,
            self_energy=jnp.zeros((), dtype=total.dtype),
        )
        return AtomisticTermEvaluation(
            jnp.where(successful, total, jnp.nan),
            atom_energy,
            successful,
            atom_energy_complete=True,
            evidence=(evidence,),
        )


def _context_bound(
    context: AtomisticPotentialContext, system: PreparedAtomisticSystem, /
) -> Array:
    """Whether the context belongs to the term's prepared system (traced).

    Prepared electrostatic artifacts are stale for another system; a mismatch
    fails the term's evaluation.
    """
    return context.system.prepared_identity.equals(system.prepared_identity)


def _split_pairs(
    capacity: int, context: AtomisticPotentialContext, pair: Array, /
) -> Array:
    atom_energy = jnp.zeros((capacity,), dtype=pair.dtype)
    atom_energy = atom_energy.at[context.pair_left].add(0.5 * pair)
    return atom_energy.at[context.pair_right].add(0.5 * pair)


class _PairSum(StrictModule):
    pair: Array
    geometry_successful: Array


def _cutoff_pair_sum(
    context: AtomisticPotentialContext,
    charges: Array,
    cutoff: float,
    kernel: Callable[[Array], Array],
    /,
) -> _PairSum:
    """``sum_{pairs, r < cutoff} s_ij q_i q_j kernel(r)`` over the MIC pair relation."""
    distance = context.pair_distance
    active = context.pair_valid & (distance < cutoff)
    geometry = jnp.all(~context.pair_valid | (distance > 0.0))
    safe = jnp.where(
        active & (distance > 0.0), distance, jnp.asarray(cutoff, distance.dtype)
    )
    values = (
        charges[context.pair_left]
        * charges[context.pair_right]
        * kernel(safe)
        * context.electrostatic_scales
    )
    return _PairSum(jnp.where(active, values, 0.0), geometry)


def _pair_term_capabilities() -> AtomisticPotentialCapabilities:
    return AtomisticPotentialCapabilities(
        conservative_energy=True,
        finite_geometry=True,
        orthorhombic_periodic=True,
        triclinic_periodic=True,
        cell_derivative=True,
        local_energy=True,
        dynamic_species=True,
    )


def _require_pair_cell(system: PreparedAtomisticSystem, cutoff: float, /) -> None:
    if system.cell is not None:
        system.cell.require_unique_image(cutoff)


class ScreenedCoulombPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    """Yukawa screened Coulomb ``k_e q_i q_j exp(-kappa r) / r`` with declared truncation.

    ``inverse_screening_length`` is ``kappa`` in inverse system length. The
    ``hard`` truncation is discontinuous at ``cutoff``; ``shifted-potential``
    subtracts ``phi(r_c)``; ``shifted-force`` also subtracts
    ``phi'(r_c) (r - r_c)``, making energy and force continuous. Finite cells
    and minimum-image periodic cells (cutoff below the unique-image radius)
    are admitted; there is no self term.

    ``inverse_screening_length`` is a dynamic scalar (the term digest keeps
    its exact host value); ``cutoff`` is the resource-defining neighborhood
    requirement cutoff and ``truncation`` a selector, both static. The
    canonical term digest is the dynamic ``term_identity``.
    """

    inverse_screening_length: Array
    cutoff: float = eqx.field(static=True)
    truncation: ScreenedTruncation = eqx.field(static=True)
    charge_field: AtomisticFieldReference | None
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        inverse_screening_length: float,
        cutoff: float,
        /,
        *,
        truncation: ScreenedTruncation = "shifted-force",
        charge_field: AtomisticFieldReference | None = None,
        name: str = "screened-coulomb",
        force_group: int = 0,
    ) -> None:
        kappa = _finite_positive(inverse_screening_length, "inverse_screening_length")
        radius = _finite_positive(cutoff, "cutoff")
        truncation = parse(truncation, ScreenedTruncation, "truncation")
        identifier, group = _term_identity(name, force_group)
        reference = _checked_charge_field(charge_field)
        self.inverse_screening_length = jnp.asarray(kappa)
        self.cutoff = radius
        self.truncation = truncation
        self.charge_field = reference
        self.name = identifier
        self.force_group = group
        self.capabilities = _pair_term_capabilities()
        self.requirements = AtomisticPotentialRequirements(
            neighborhoods=(
                AtomisticNeighborhoodRequirement(
                    identifier,
                    AtomisticNeighborhoodRole.CLASSICAL_MINIMUM_IMAGE_PAIRS,
                    radius,
                    exclusions=AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS,
                ),
            ),
            field_dependencies=_field_dependencies(reference),
        )
        self.term_identity = _electrostatic_identity(
            "screened-coulomb-potential",
            {
                "kernel": "yukawa",
                "inverse_screening_length": float(kappa).hex(),
                "cutoff": float(radius).hex(),
                "truncation": truncation,
                "charges": _charge_source(reference),
                "name": identifier,
                "force_group": group,
            },
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedScreenedCoulombPotential":
        _require_pair_cell(system, self.cutoff)
        _require_charge_unit(self.charge_field, system)
        return PreparedScreenedCoulombPotential(self, system)


class PreparedScreenedCoulombPotential(AbstractPreparedAtomisticEnergyTerm):
    """Screened Coulomb bound to one system; ``prepared_identity`` is dynamic.

    A context of another prepared system fails the evaluation numerically.
    """

    plan: ScreenedCoulombPotential
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self, plan: ScreenedCoulombPotential, system: PreparedAtomisticSystem, /
    ) -> None:
        self.plan = plan
        self.system = system
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_identity = _electrostatic_identity(
            "prepared-screened-coulomb",
            {"term": plan.term_id, "system": system.prepared_id},
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        bound = _context_bound(context, self.system)
        charges = _bind_charges(self.system, self.plan.charge_field, context)
        dtype = context.positions.dtype
        # Shift constants follow the former host formula in the parameter's
        # own precision and enter the kernel rounded to the coordinate dtype,
        # exactly as the former weakly typed Python constants did.
        parameter = self.plan.inverse_screening_length
        cutoff = self.plan.cutoff
        truncation = self.plan.truncation
        boundary = (jnp.exp(-parameter * cutoff) / cutoff).astype(dtype)
        slope = (
            -jnp.exp(-parameter * cutoff) * (parameter * cutoff + 1.0) / cutoff**2
        ).astype(dtype)
        kappa = parameter.astype(dtype)

        def kernel(r: Array) -> Array:
            value = jnp.exp(-kappa * r) / r
            if truncation == "hard":
                return value
            value = value - boundary
            if truncation == "shifted-potential":
                return value
            return value - slope * (r - cutoff)

        pairs = _cutoff_pair_sum(context, charges.values, cutoff, kernel)
        pair = self.system.plan.units.coulomb_constant * pairs.pair
        total = jnp.sum(pair)
        successful = (
            bound
            & pairs.geometry_successful
            & jnp.all(jnp.isfinite(pair))
            & charges.available
            & charges.successful
        )
        evidence = PairCoulombEvidence(
            method=f"screened-coulomb-yukawa-{truncation}",
            charge_source=_charge_source(self.plan.charge_field),
            prepared_identity=self.prepared_identity,
            charges_available=charges.available,
            charges_successful=charges.successful,
            geometry_successful=pairs.geometry_successful,
            total_charge=jnp.sum(charges.values),
            pair_energy=total,
            self_energy=jnp.zeros((), dtype=total.dtype),
        )
        return AtomisticTermEvaluation(
            jnp.where(successful, total, jnp.nan),
            _split_pairs(self.system.capacity, context, pair),
            successful,
            atom_energy_complete=True,
            evidence=(evidence,),
        )


class DampedShiftedForceCoulombPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    """Fennell-Gezelter damped shifted-force Coulomb (J. Chem. Phys. 124, 234104, 2006).

    For ``r < r_c`` each pair contributes
    ``k_e q_i q_j [erfc(a r)/r - erfc(a r_c)/r_c + (erfc(a r_c)/r_c^2 +
    2a/sqrt(pi) exp(-a^2 r_c^2)/r_c)(r - r_c)]`` times its exception scale,
    and every active atom the self term ``-k_e q_i^2 (erfc(a r_c)/(2 r_c) +
    a/sqrt(pi))``. Energy and force vanish continuously at ``r_c``. The pair
    halves plus each self term are a complete atomic decomposition.

    The damping ``alpha`` is a dynamic scalar (the term digest keeps its
    exact host value); ``cutoff`` is the resource-defining neighborhood
    requirement cutoff and stays static. The canonical term digest is the
    dynamic ``term_identity``.
    """

    alpha: Array
    cutoff: float = eqx.field(static=True)
    charge_field: AtomisticFieldReference | None
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        alpha: float,
        cutoff: float,
        /,
        *,
        charge_field: AtomisticFieldReference | None = None,
        name: str = "damped-shifted-force-coulomb",
        force_group: int = 0,
    ) -> None:
        damping = float(alpha)
        if not math.isfinite(damping) or damping < 0.0:
            raise ValueError("DSF alpha must be finite and non-negative.")
        radius = _finite_positive(cutoff, "cutoff")
        identifier, group = _term_identity(name, force_group)
        reference = _checked_charge_field(charge_field)
        self.alpha = jnp.asarray(damping)
        self.cutoff = radius
        self.charge_field = reference
        self.name = identifier
        self.force_group = group
        self.capabilities = _pair_term_capabilities()
        self.requirements = AtomisticPotentialRequirements(
            neighborhoods=(
                AtomisticNeighborhoodRequirement(
                    identifier,
                    AtomisticNeighborhoodRole.CLASSICAL_MINIMUM_IMAGE_PAIRS,
                    radius,
                    exclusions=AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS,
                ),
            ),
            field_dependencies=_field_dependencies(reference),
        )
        self.term_identity = _electrostatic_identity(
            "damped-shifted-force-coulomb-potential",
            {
                "alpha": damping.hex(),
                "cutoff": float(radius).hex(),
                "charges": _charge_source(reference),
                "name": identifier,
                "force_group": group,
            },
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedDampedShiftedForceCoulombPotential":
        _require_pair_cell(system, self.cutoff)
        _require_charge_unit(self.charge_field, system)
        return PreparedDampedShiftedForceCoulombPotential(self, system)


class PreparedDampedShiftedForceCoulombPotential(AbstractPreparedAtomisticEnergyTerm):
    """DSF Coulomb bound to one system; ``prepared_identity`` is dynamic.

    A context of another prepared system fails the evaluation numerically.
    """

    plan: DampedShiftedForceCoulombPotential
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        plan: DampedShiftedForceCoulombPotential,
        system: PreparedAtomisticSystem,
        /,
    ) -> None:
        self.plan = plan
        self.system = system
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_identity = _electrostatic_identity(
            "prepared-damped-shifted-force-coulomb",
            {"term": plan.term_id, "system": system.prepared_id},
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        bound = _context_bound(context, self.system)
        charges = _bind_charges(self.system, self.plan.charge_field, context)
        dtype = context.positions.dtype
        # Shift, slope and self coefficient follow the former host formula in
        # the parameter's own precision and enter the kernel rounded to the
        # coordinate dtype, exactly as the former weakly typed constants did.
        parameter = self.plan.alpha
        cutoff = self.plan.cutoff
        shift_parameter = jsp.erfc(parameter * cutoff) / cutoff
        slope = (
            shift_parameter / cutoff
            + 2.0 * parameter / _SQRT_PI * jnp.exp(-((parameter * cutoff) ** 2)) / cutoff
        ).astype(dtype)
        shift = shift_parameter.astype(dtype)
        alpha = parameter.astype(dtype)

        def kernel(r: Array) -> Array:
            return ewald_real_space_kernel(alpha, r) - shift + slope * (r - cutoff)

        pairs = _cutoff_pair_sum(context, charges.values, cutoff, kernel)
        constant = self.system.plan.units.coulomb_constant
        pair = constant * pairs.pair
        self_coefficient = (
            -constant * (0.5 * shift_parameter + parameter / _SQRT_PI)
        ).astype(dtype)
        self_atoms = self_coefficient * charges.values**2
        atom_energy = _split_pairs(self.system.capacity, context, pair) + self_atoms
        pair_total = jnp.sum(pair)
        self_total = jnp.sum(self_atoms)
        total = pair_total + self_total
        successful = (
            bound
            & pairs.geometry_successful
            & jnp.isfinite(total)
            & charges.available
            & charges.successful
        )
        evidence = PairCoulombEvidence(
            method="damped-shifted-force-coulomb",
            charge_source=_charge_source(self.plan.charge_field),
            prepared_identity=self.prepared_identity,
            charges_available=charges.available,
            charges_successful=charges.successful,
            geometry_successful=pairs.geometry_successful,
            total_charge=jnp.sum(charges.values),
            pair_energy=pair_total,
            self_energy=self_total,
        )
        return AtomisticTermEvaluation(
            jnp.where(successful, total, jnp.nan),
            atom_energy,
            successful,
            atom_energy_complete=True,
            evidence=(evidence,),
        )


class _EwaldSettings(StrictModule, NonTrainableState):
    """Validated scientific conventions shared by Ewald and PME plans.

    Built only by `_ewald_settings`, which validates every value and returns
    the exact host identity payload. ``alpha`` (the Ewald splitting
    parameter) is a dynamic scalar: it defines neither the stored modes nor
    the mesh, so splitting revisions of one layout share an executable.
    ``real_cutoff`` is the resource-defining neighborhood requirement cutoff,
    ``reciprocal_cutoff`` the declared reciprocal coverage envelope and
    ``charge_tolerance`` the admission threshold of the declared neutrality
    policy; they and the convention selectors stay static.
    """

    alpha: Array
    real_cutoff: float = eqx.field(static=True)
    neutrality: ChargeNeutralityPolicy = eqx.field(static=True)
    charge_tolerance: float = eqx.field(static=True)
    boundary: EwaldBoundaryConvention = eqx.field(static=True)
    real_space: EwaldRealSpaceRole = eqx.field(static=True)
    reciprocal_cutoff: float | None = eqx.field(static=True)

    def __init__(
        self,
        alpha: float,
        real_cutoff: float,
        /,
        *,
        neutrality: ChargeNeutralityPolicy,
        charge_tolerance: float,
        boundary: EwaldBoundaryConvention,
        real_space: EwaldRealSpaceRole,
        reciprocal_cutoff: float | None,
    ) -> None:
        self.alpha = jnp.asarray(alpha)
        self.real_cutoff = real_cutoff
        self.neutrality = neutrality
        self.charge_tolerance = charge_tolerance
        self.boundary = boundary
        self.real_space = real_space
        self.reciprocal_cutoff = reciprocal_cutoff

    def admit(self, system: PreparedAtomisticSystem, /) -> PeriodicCell:
        cell = system.cell
        if cell is None or cell.rank != 3 or cell.ambient_dimension != 3:
            raise ValueError("Periodic Coulomb requires a full rank-3 PeriodicCell.")
        periodic = sum(cell.periodic_axes)
        if self.boundary == "slab-yeh-berkowitz":
            if periodic != 2:
                raise ValueError(
                    "The slab convention requires exactly two periodic cell axes."
                )
        elif periodic != 3:
            raise ValueError(
                f"The {self.boundary!r} convention requires a fully periodic cell; use "
                "boundary='slab-yeh-berkowitz' for two periodic axes."
            )
        if self.real_space == "minimum-image":
            cell.require_unique_image(self.real_cutoff)
        return cell

    def requirements(
        self, owner: str, reference: AtomisticFieldReference | None, /
    ) -> AtomisticPotentialRequirements:
        """Minimum-image pairs from the classical context, or the image-pair-sum role."""
        dependencies = _field_dependencies(reference)
        if self.real_space == "minimum-image":
            return AtomisticPotentialRequirements(
                neighborhoods=(
                    AtomisticNeighborhoodRequirement(
                        owner,
                        AtomisticNeighborhoodRole.CLASSICAL_MINIMUM_IMAGE_PAIRS,
                        self.real_cutoff,
                        exclusions=AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS,
                    ),
                ),
                reciprocal_grid=True,
                field_dependencies=dependencies,
            )
        return AtomisticPotentialRequirements(
            reciprocal_grid=True,
            neighborhoods=(self.image_pair_requirement(owner),),
            field_dependencies=dependencies,
        )

    def image_pair_requirement(self, owner: str, /) -> AtomisticNeighborhoodRequirement:
        return AtomisticNeighborhoodRequirement(
            owner,
            AtomisticNeighborhoodRole.IMAGE_PAIR_SUM,
            self.real_cutoff,
            exclusions=AtomisticPairExclusions.NONE,
        )


def _ewald_settings(
    alpha: float,
    real_cutoff: float,
    /,
    *,
    neutrality: ChargeNeutralityPolicy,
    charge_tolerance: float,
    boundary: EwaldBoundaryConvention,
    real_space: str,
    reciprocal_cutoff: float | None,
) -> tuple[_EwaldSettings, dict[str, object]]:
    """Validate periodic Coulomb conventions; return them and their identity payload.

    The payload carries the exact host values, so term digests are
    independent of how ``alpha`` is carried.
    """
    splitting = _finite_positive(alpha, "alpha")
    cutoff = _finite_positive(real_cutoff, "real_cutoff")
    neutrality_ = parse(neutrality, ChargeNeutralityPolicy, "neutrality")
    tolerance = _finite_positive(charge_tolerance, "charge_tolerance")
    boundary_ = parse(boundary, EwaldBoundaryConvention, "boundary")
    real_space_ = parse(real_space, EwaldRealSpaceRole, "real_space")
    reciprocal = (
        None
        if reciprocal_cutoff is None
        else _finite_positive(reciprocal_cutoff, "reciprocal_cutoff")
    )
    if boundary_ == "vacuum-sphere" and neutrality_ != "require-neutral":
        raise ValueError(
            "The vacuum-sphere dipole boundary is origin dependent for a charged "
            "cell; it requires neutrality='require-neutral'."
        )
    identity: dict[str, object] = {
        "alpha": float(splitting).hex(),
        "real_cutoff": float(cutoff).hex(),
        "neutrality": neutrality_,
        "charge_tolerance": float(tolerance).hex(),
        "boundary": boundary_,
        "real_space": real_space_,
        "reciprocal_cutoff": None if reciprocal is None else float(reciprocal).hex(),
        "k_zero": "excluded-owned-by-neutrality-and-boundary",
    }
    settings = _EwaldSettings(
        splitting,
        cutoff,
        neutrality=neutrality_,
        charge_tolerance=tolerance,
        boundary=boundary_,
        real_space=real_space_,
        reciprocal_cutoff=reciprocal,
    )
    return settings, identity


def _periodic_capabilities() -> AtomisticPotentialCapabilities:
    return AtomisticPotentialCapabilities(
        conservative_energy=True,
        finite_geometry=False,
        orthorhombic_periodic=True,
        triclinic_periodic=True,
        cell_derivative=True,
        local_energy=False,
    )


def _minimum_image(
    cell: PeriodicCell, displacement: Array, vectors: Array, inverse: Array, /
) -> Array:
    """Nearest periodic image of each displacement under the runtime lattice."""
    fractional = contract("ni,ij->nj", displacement, inverse, backend="jax")
    central = jax.lax.stop_gradient(jnp.round(fractional))
    central = jnp.where(cell.periodic_mask, central, 0.0)
    shifts = central[:, None, :] + cell.image_shifts.astype(displacement.dtype)
    candidates = displacement[:, None, :] - contract(
        "nsi,ij->nsj", shifts, vectors, backend="jax"
    )
    selected = jax.lax.stop_gradient(
        jnp.argmin(jnp.sum(candidates * candidates, axis=-1), axis=-1)
    )
    return jnp.take_along_axis(candidates, selected[:, None, None], axis=1)[:, 0, :]


class _PeriodicParts(StrictModule):
    real: Array
    self_energy: Array
    exception: Array
    background: Array
    boundary: Array
    total_charge: Array
    neutral: Array
    geometry_successful: Array


def _image_pair_real_space(
    settings: _EwaldSettings,
    owner: str,
    context: AtomisticPotentialContext,
    charges: Array,
    /,
) -> tuple[Array, Array]:
    """``1/2 sum'_{i,j,n} q_i q_j erfc(alpha r)/r`` over the image-pair-sum role.

    Each stored half route represents ``directed_multiplicity`` directed
    routes, including nonzero self images; no exception scale is applied.
    """
    view = context.neighborhood(settings.image_pair_requirement(owner))
    if view.entity not in (
        AtomisticSiteDomain.PHYSICAL_ATOMS,
        AtomisticSiteDomain.DOF_ATOMS,
    ):
        raise ValueError("Ewald image-pair routes must address the system atoms.")
    dtype = context.positions.dtype
    vectors = context.cell_vectors.astype(dtype)
    displacement = view.displacement(context.positions, vectors)
    squared = jnp.sum(displacement * displacement, axis=-1)
    active = view.cutoff_mask(context.positions, vectors)
    positive = squared > 0.0
    geometry = jnp.all(~active | positive)
    distance = jnp.sqrt(jnp.where(active & positive, squared, settings.real_cutoff**2))
    receiver = view.relation.receiver_indices
    source = view.relation.source_indices
    values = ewald_real_space_kernel(
        settings.alpha.astype(dtype),
        distance,
        numerator=charges[receiver] * charges[source],
    )
    weight = 0.5 * view.directed_multiplicity
    real = weight * jnp.sum(jnp.where(active, values, 0.0))
    return real, geometry


def ewald_self_energy(alpha: ArrayLike, constant: float, charges: Array, /) -> Array:
    """Gaussian self energy ``-k alpha / sqrt(pi) sum_i q_i^2`` of the Ewald split."""
    return -constant * alpha / _SQRT_PI * jnp.sum(charges * charges)


def ewald_real_space_kernel(
    alpha: ArrayLike, distance: Array, /, *, numerator: ArrayLike = 1.0
) -> Array:
    """Ewald real-space pair kernel ``(numerator * erfc(alpha r)) / r`` (``r > 0``).

    ``numerator`` (default ``1.0``, bitwise neutral) carries a charge product
    in the canonical multiply-then-divide order of the image-pair sum.
    """
    return (numerator * jsp.erfc(alpha * distance)) / distance


def ewald_exception_kernel(
    alpha: ArrayLike,
    real_cutoff: float,
    distance: Array,
    positive: Array,
    /,
    *,
    explicit_image: bool,
) -> Array:
    """Smeared kernel removed by a topology exception at its minimum image.

    The reciprocal sum holds ``erf(alpha r) / r`` of the exception pair (its
    ``r -> 0`` limit ``2 alpha / sqrt(pi)`` when coincident). Under the
    explicit-image real-space role the unscaled ``erfc(alpha r) / r`` of the
    member inside ``real_cutoff`` is also present and removed with it.
    """
    smeared = jnp.where(
        positive, jsp.erf(alpha * distance) / distance, 2.0 * alpha / _SQRT_PI
    )
    if explicit_image:
        smeared = smeared + jnp.where(
            positive & (distance < real_cutoff),
            ewald_real_space_kernel(alpha, distance),
            0.0,
        )
    return smeared


def ewald_background_energy(
    alpha: ArrayLike, constant: float, total_charge: Array, volume: Array, /
) -> Array:
    """Uniform neutralizing background ``-pi k Q^2 / (2 alpha^2 V)``."""
    return -jnp.pi * constant * total_charge**2 / (2.0 * alpha**2 * volume)


def ewald_vacuum_boundary_energy(
    constant: float, dipole: Array, volume: Array, /
) -> Array:
    """Vacuum-sphere dipole correction ``2 pi k |D|^2 / (3 V)``."""
    return 2.0 * jnp.pi * constant * jnp.sum(dipole * dipole) / (3.0 * volume)


def ewald_slab_height(
    cell: PeriodicCell, vectors: Array, volume: Array, /
) -> tuple[Array, int]:
    """Slab height ``V / |a x b|`` over the two periodic axes and the normal axis."""
    periodic = tuple(index for index, flag in enumerate(cell.periodic_axes) if flag)
    normal_axis = next(index for index, flag in enumerate(cell.periodic_axes) if not flag)
    normal = jnp.cross(vectors[periodic[0]], vectors[periodic[1]])
    area = jnp.sqrt(jnp.sum(normal * normal))
    return volume / area, normal_axis


def ewald_slab_boundary_energy(
    constant: float,
    volume: Array,
    height: Array,
    total_charge: Array,
    moment: Array,
    spread: Array,
    /,
) -> Array:
    """Yeh-Berkowitz slab correction from the normal-axis charge moments."""
    return (
        2.0
        * jnp.pi
        * constant
        / volume
        * (moment**2 - total_charge * spread - total_charge**2 * height**2 / 12.0)
    )


def _periodic_parts(
    settings: _EwaldSettings,
    owner: str,
    system: PreparedAtomisticSystem,
    cell: PeriodicCell,
    context: AtomisticPotentialContext,
    geometry: ReciprocalGeometry,
    charges: Array,
    /,
) -> _PeriodicParts:
    """Real-space, self, exception, background and boundary energies.

    Topology exceptions are corrected on their minimum-image (bonded) member
    only, from the fixed topology exception relation: the reciprocal sum
    contains ``erf(alpha r)/r`` of that member and, under the explicit-image
    role, real space also contains its unscaled ``erfc(alpha r)/r`` when
    ``r < real_cutoff``; the minimum-image pair relation already applies the
    exception scale.
    """
    constant = system.plan.units.coulomb_constant
    dtype = context.positions.dtype
    alpha = settings.alpha.astype(dtype)
    explicit = settings.real_space == "explicit-image"
    if explicit:
        real_sum, pair_geometry = _image_pair_real_space(
            settings, owner, context, charges
        )
    else:
        pairs = _cutoff_pair_sum(
            context,
            charges,
            settings.real_cutoff,
            lambda r: ewald_real_space_kernel(alpha, r),
        )
        real_sum, pair_geometry = jnp.sum(pairs.pair), pairs.geometry_successful
    real = constant * real_sum
    self_energy = ewald_self_energy(alpha, constant, charges)
    exception = jnp.zeros((), dtype=dtype)
    indices = system.topology.exception_indices
    if indices.shape[0]:
        vectors = context.cell_vectors.astype(dtype)
        displacement = _minimum_image(
            cell,
            context.positions[indices[:, 0]] - context.positions[indices[:, 1]],
            vectors,
            geometry.inverse.astype(dtype),
        )
        squared = jnp.sum(displacement * displacement, axis=-1)
        positive = squared > 0.0
        distance = jnp.sqrt(jnp.where(positive, squared, 1.0))
        smeared = ewald_exception_kernel(
            alpha, settings.real_cutoff, distance, positive, explicit_image=explicit
        )
        if explicit:
            pair_geometry = pair_geometry & jnp.all(positive)
        exception = -constant * jnp.sum(
            (1.0 - system.topology.electrostatic_scales.astype(dtype))
            * charges[indices[:, 0]]
            * charges[indices[:, 1]]
            * smeared
        )
    total_charge = jnp.sum(charges)
    neutral = jnp.abs(total_charge) <= settings.charge_tolerance
    volume = geometry.volume
    background = (
        ewald_background_energy(alpha, constant, total_charge, volume)
        if settings.neutrality == "uniform-background"
        else jnp.zeros((), dtype=dtype)
    )
    geometry_ok = pair_geometry
    boundary = jnp.zeros((), dtype=dtype)
    if settings.boundary == "vacuum-sphere":
        dipole = contract("n,nd->d", charges, context.unwrapped_positions, backend="jax")
        boundary = ewald_vacuum_boundary_energy(constant, dipole, volume)
    elif settings.boundary == "slab-yeh-berkowitz":
        vectors = context.cell_vectors.astype(dtype)
        height, normal_axis = ewald_slab_height(cell, vectors, volume)
        fractional = contract(
            "ni,i->n",
            context.positions - cell.origin.astype(dtype),
            geometry.inverse[:, normal_axis].astype(dtype),
            backend="jax",
        )
        active = system.active_mask
        inside = jnp.all(~active | ((fractional >= 0.0) & (fractional < 1.0)))
        height_coordinate = fractional * height
        top = jnp.max(jnp.where(active, height_coordinate, -jnp.inf))
        bottom = jnp.min(jnp.where(active, height_coordinate, jnp.inf))
        gap = (top - bottom) + settings.real_cutoff < height
        geometry_ok = geometry_ok & inside & gap
        moment = jnp.sum(charges * height_coordinate)
        spread = jnp.sum(charges * height_coordinate**2)
        boundary = ewald_slab_boundary_energy(
            constant, volume, height, total_charge, moment, spread
        )
    return _PeriodicParts(
        real=real,
        self_energy=self_energy,
        exception=exception,
        background=background,
        boundary=boundary,
        total_charge=total_charge,
        neutral=neutral,
        geometry_successful=geometry_ok,
    )


def _coulomb_influence(
    geometry: ReciprocalGeometry, alpha: ArrayLike, constant: float, /
) -> ReciprocalInfluence:
    """``2 pi k_e exp(-k^2 / 4 alpha^2) / (V k^2)``, finite at every stored mode."""
    volume = geometry.volume

    def influence(wave: Array, squared: Array) -> Array:
        del wave
        safe = jnp.where(squared > 0.0, squared, 1.0)
        return (
            2.0
            * jnp.pi
            * constant
            / volume
            * jnp.where(squared > 0.0, jnp.exp(-safe / (4.0 * alpha**2)) / safe, 0.0)
        )

    return influence


def _periodic_evaluation(
    *,
    method: str,
    settings: _EwaldSettings,
    charge_field: AtomisticFieldReference | None,
    prepared_identity: IdentityWords,
    bound: Array,
    capacity: int,
    charges: _BoundCharges,
    parts: _PeriodicParts,
    geometry: ReciprocalGeometry,
    reciprocal: ReciprocalQuadratic,
) -> AtomisticTermEvaluation:
    coverage = (
        jnp.asarray(True)
        if settings.reciprocal_cutoff is None
        else settings.reciprocal_cutoff <= geometry.support_radius
    )
    neutral_ok = (
        parts.neutral if settings.neutrality == "require-neutral" else jnp.asarray(True)
    )
    total = (
        parts.real
        + reciprocal.value
        + parts.self_energy
        + parts.exception
        + parts.background
        + parts.boundary
    )
    successful = (
        bound
        & charges.available
        & charges.successful
        & neutral_ok
        & geometry.successful
        & parts.geometry_successful
        & coverage
        & reciprocal.successful
        & jnp.isfinite(total)
    )
    evidence = PeriodicCoulombEvidence(
        method=method,
        boundary=settings.boundary,
        neutrality=settings.neutrality,
        real_space=settings.real_space,
        charge_source=_charge_source(charge_field),
        declared_reciprocal_cutoff=settings.reciprocal_cutoff,
        prepared_identity=prepared_identity,
        charges_available=charges.available,
        charges_successful=charges.successful,
        total_charge=parts.total_charge,
        neutral=parts.neutral,
        cell_successful=geometry.successful,
        geometry_successful=parts.geometry_successful,
        mode_coverage_successful=coverage,
        support_radius=geometry.support_radius,
        reciprocal_successful=reciprocal.successful,
        volume=geometry.volume,
        real_energy=parts.real,
        reciprocal_energy=reciprocal.value,
        self_energy=parts.self_energy,
        exception_energy=parts.exception,
        background_energy=parts.background,
        boundary_energy=parts.boundary,
    )
    return AtomisticTermEvaluation(
        jnp.where(successful, total, jnp.nan),
        jnp.zeros((capacity,), dtype=total.dtype),
        successful,
        atom_energy_complete=False,
        evidence=(evidence,),
    )


def _reciprocal_extents(value: int | Sequence[int], /) -> tuple[int, int, int]:
    if isinstance(value, Sequence):
        items = tuple(int(item) for item in value)
    else:
        items = (int(value),) * 3
    if len(items) != 3 or any(item <= 0 for item in items):
        raise ValueError("reciprocal_extent must be positive (one value or three).")
    return (items[0], items[1], items[2])


class EwaldReferencePotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    """Periodic Ewald Coulomb with explicit integer reciprocal modes.

    Real space sums ``erfc(alpha r) / r`` below ``real_cutoff`` either over the
    classical minimum-image pair relation (``real_space="minimum-image"``; the
    cutoff must stay below the unique-image radius, so there are no self
    images) or over the bundle's explicitly counted image-pair-sum role
    (``"explicit-image"``; every image and nonzero self image inside the
    cutoff, for cells smaller than twice the cutoff), with topology exceptions
    corrected on their minimum-image member. Reciprocal space sums every canonical
    half-space mode with ``|m_i| <= reciprocal_extent[i]`` in chunks of
    ``mode_chunk``; the fixed integer set keeps the energy smooth under cell
    changes. A declared ``reciprocal_cutoff`` is certified against the
    current cell's ``support_radius`` and its failure is reported.

    The splitting ``alpha`` is a dynamic scalar (see `_EwaldSettings`); the
    mode ``reciprocal_extent`` and ``mode_chunk`` define the stored mode
    tables and stay static. The canonical term digest is the dynamic
    ``term_identity``.
    """

    settings: _EwaldSettings
    reciprocal_extent: tuple[int, int, int] = eqx.field(static=True)
    mode_chunk: int = eqx.field(static=True)
    budget: ReciprocalBudget
    charge_field: AtomisticFieldReference | None
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        alpha: float,
        real_cutoff: float,
        reciprocal_extent: int | Sequence[int],
        /,
        *,
        neutrality: ChargeNeutralityPolicy = "require-neutral",
        charge_tolerance: float = 1.0e-10,
        boundary: EwaldBoundaryConvention = "tin-foil",
        real_space: str = "minimum-image",
        reciprocal_cutoff: float | None = None,
        mode_chunk: int = 128,
        budget: ReciprocalBudget | None = None,
        charge_field: AtomisticFieldReference | None = None,
        name: str = "ewald-reference",
        force_group: int = 0,
    ) -> None:
        settings, settings_identity = _ewald_settings(
            alpha,
            real_cutoff,
            neutrality=neutrality,
            charge_tolerance=charge_tolerance,
            boundary=boundary,
            real_space=real_space,
            reciprocal_cutoff=reciprocal_cutoff,
        )
        extents = _reciprocal_extents(reciprocal_extent)
        chunk = int(mode_chunk)
        if chunk <= 0:
            raise ValueError("mode_chunk must be positive.")
        budget_ = ReciprocalBudget() if budget is None else budget
        identifier, group = _term_identity(name, force_group)
        reference = _checked_charge_field(charge_field)
        self.settings = settings
        self.reciprocal_extent = extents
        self.mode_chunk = chunk
        self.budget = budget_
        self.charge_field = reference
        self.name = identifier
        self.force_group = group
        self.capabilities = _periodic_capabilities()
        self.requirements = settings.requirements(identifier, reference)
        self.term_identity = _electrostatic_identity(
            "ewald-reference-potential",
            {
                **settings_identity,
                "reciprocal_extent": list(extents),
                "mode_chunk": chunk,
                "budget": budget_.budget_id,
                "charges": _charge_source(reference),
                "name": identifier,
                "force_group": group,
            },
        )

    @property
    def alpha(self) -> Array:
        return self.settings.alpha

    @property
    def real_cutoff(self) -> float:
        return self.settings.real_cutoff

    @property
    def neutrality(self) -> ChargeNeutralityPolicy:
        return self.settings.neutrality

    @property
    def charge_tolerance(self) -> float:
        return self.settings.charge_tolerance

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedEwaldReferencePotential":
        cell = self.settings.admit(system)
        _require_charge_unit(self.charge_field, system)
        modes = ReciprocalModeSetPlan(
            self.reciprocal_extent, mode_chunk=self.mode_chunk, budget=self.budget
        ).prepare(cell, system.capacity)
        return PreparedEwaldReferencePotential(self, system, modes)


class PreparedEwaldReferencePotential(AbstractPreparedAtomisticEnergyTerm):
    """Ewald Coulomb bound to one system and its mode set.

    The mode set is layout (plan, reference cell, capacity); ``term_identity``
    and ``prepared_identity`` are dynamic. A context of another prepared
    system fails the evaluation numerically.
    """

    plan: EwaldReferencePotential
    system: PreparedAtomisticSystem
    cell: PeriodicCell
    modes: PreparedReciprocalModeSet
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        plan: EwaldReferencePotential,
        system: PreparedAtomisticSystem,
        modes: PreparedReciprocalModeSet,
        /,
    ) -> None:
        cell = system.cell
        if cell is None or modes.cell is None or modes.cell.cell_id != cell.cell_id:
            raise ValueError(
                "Ewald modes must be prepared for the system's periodic cell."
            )
        self.plan = plan
        self.system = system
        self.cell = cell
        self.modes = modes
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_identity = _electrostatic_identity(
            "prepared-ewald-reference",
            {
                "term": plan.term_id,
                "system": system.prepared_id,
                "modes": modes.prepared_id,
            },
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        bound = _context_bound(context, self.system)
        settings = self.plan.settings
        charges = _bind_charges(self.system, self.plan.charge_field, context)
        geometry = self.modes.geometry(
            context.cell_vectors.astype(context.positions.dtype)
        )
        influence = _coulomb_influence(
            geometry,
            settings.alpha.astype(context.positions.dtype),
            self.system.plan.units.coulomb_constant,
        )
        reciprocal = self.modes.quadratic_energy(
            context.positions,
            charges.values[:, None],
            geometry,
            influence,
            zero_mode="exclude",
        )
        parts = _periodic_parts(
            settings,
            self.name,
            self.system,
            self.cell,
            context,
            geometry,
            charges.values,
        )
        return _periodic_evaluation(
            method="ewald-explicit-modes",
            settings=settings,
            charge_field=self.plan.charge_field,
            prepared_identity=self.prepared_identity,
            bound=bound,
            capacity=self.system.capacity,
            charges=charges,
            parts=parts,
            geometry=geometry,
            reciprocal=reciprocal,
        )


class ParticleMeshEwaldPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    """Smooth particle-mesh Ewald Coulomb on a term-owned reciprocal mesh.

    Charges are spread with the native cardinal B-spline of ``spline_degree``
    (1 to 5) onto a fractional mesh of the runtime row cell; the influence is
    ``2 pi k_e exp(-k^2/4 alpha^2) / (V k^2)`` times the inverse squared
    declared ``modulus`` (the exact discrete B-spline symbol by default) on the
    Hermitian half spectrum under the declared ``nyquist`` convention
    (``"symmetrized"`` keeps every real-FFT mode; ``"excluded"`` is a
    reported conjugation-closed truncation). Real space, self, exceptions,
    neutrality and boundary conventions match :class:`EwaldReferencePotential`.

    The splitting ``alpha`` is a dynamic scalar (see `_EwaldSettings`); the
    mesh plan (grid shape, spline degree, modulus, Nyquist convention) is
    layout. The canonical term digest is the dynamic ``term_identity``.
    """

    settings: _EwaldSettings
    mesh_plan: ReciprocalMeshPlan
    charge_field: AtomisticFieldReference | None
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        alpha: float,
        real_cutoff: float,
        grid_shape: Sequence[int],
        /,
        *,
        spline_degree: int = 3,
        modulus: ReciprocalMeshModulus = "discrete-bspline-modulus",
        nyquist: ReciprocalNyquistPolicy = "symmetrized",
        neutrality: ChargeNeutralityPolicy = "require-neutral",
        charge_tolerance: float = 1.0e-10,
        boundary: EwaldBoundaryConvention = "tin-foil",
        real_space: str = "minimum-image",
        reciprocal_cutoff: float | None = None,
        budget: ReciprocalBudget | None = None,
        charge_field: AtomisticFieldReference | None = None,
        name: str = "particle-mesh-ewald",
        force_group: int = 0,
    ) -> None:
        settings, settings_identity = _ewald_settings(
            alpha,
            real_cutoff,
            neutrality=neutrality,
            charge_tolerance=charge_tolerance,
            boundary=boundary,
            real_space=real_space,
            reciprocal_cutoff=reciprocal_cutoff,
        )
        budget_ = ReciprocalBudget() if budget is None else budget
        mesh = ReciprocalMeshPlan(
            tuple(grid_shape),
            spline_degree=spline_degree,
            modulus=modulus,
            nyquist=nyquist,
            channel_chunk=1,
            budget=budget_,
        )
        identifier, group = _term_identity(name, force_group)
        reference = _checked_charge_field(charge_field)
        self.settings = settings
        self.mesh_plan = mesh
        self.charge_field = reference
        self.name = identifier
        self.force_group = group
        self.capabilities = _periodic_capabilities()
        self.requirements = settings.requirements(identifier, reference)
        self.term_identity = _electrostatic_identity(
            "particle-mesh-ewald-potential",
            {
                **settings_identity,
                "mesh": mesh.plan_id,
                "charges": _charge_source(reference),
                "name": identifier,
                "force_group": group,
            },
        )

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return self.mesh_plan.grid_shape

    @property
    def spline_degree(self) -> int:
        return self.mesh_plan.spline_degree

    @property
    def alpha(self) -> Array:
        return self.settings.alpha

    @property
    def real_cutoff(self) -> float:
        return self.settings.real_cutoff

    @property
    def neutrality(self) -> ChargeNeutralityPolicy:
        return self.settings.neutrality

    @property
    def charge_tolerance(self) -> float:
        return self.settings.charge_tolerance

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> PreparedParticleMeshEwaldPotential:
        return PreparedParticleMeshEwaldPotential(self, system)


class PreparedParticleMeshEwaldPotential(AbstractPreparedAtomisticEnergyTerm):
    """PME Coulomb bound to one system and its term-owned mesh.

    The mesh is prepared here from the system's host particle support
    (``system.particle_discretization()``), so it is bound to exactly this
    system's material; its PyTree is layout-only. ``term_identity`` and
    ``prepared_identity`` are dynamic. A context of another prepared system
    fails the evaluation numerically.
    """

    plan: ParticleMeshEwaldPotential
    system: PreparedAtomisticSystem
    cell: PeriodicCell
    mesh: PreparedReciprocalMesh
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        plan: ParticleMeshEwaldPotential,
        system: PreparedAtomisticSystem,
        /,
    ) -> None:
        cell = plan.settings.admit(system)
        _require_charge_unit(plan.charge_field, system)
        mesh = plan.mesh_plan.prepare(cell, system.particle_discretization())
        self.plan = plan
        self.system = system
        self.cell = cell
        self.mesh = mesh
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_identity = _electrostatic_identity(
            "prepared-particle-mesh-ewald",
            {
                "term": plan.term_id,
                "system": system.prepared_id,
                "mesh": mesh.prepared_id,
            },
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        bound = _context_bound(context, self.system)
        settings = self.plan.settings
        charges = _bind_charges(self.system, self.plan.charge_field, context)
        geometry = self.mesh.geometry(
            context.cell_vectors.astype(context.positions.dtype)
        )
        routes = self.mesh.build(context.positions, geometry)
        influence = _coulomb_influence(
            geometry,
            settings.alpha.astype(context.positions.dtype),
            self.system.plan.units.coulomb_constant,
        )
        reciprocal = self.mesh.quadratic_energy(
            routes, charges.values[:, None], geometry, influence, zero_mode="exclude"
        )
        parts = _periodic_parts(
            settings,
            self.name,
            self.system,
            self.cell,
            context,
            geometry,
            charges.values,
        )
        return _periodic_evaluation(
            method=(
                f"particle-mesh-ewald-degree-{self.plan.spline_degree}-"
                f"{self.plan.mesh_plan.modulus}-nyquist-{self.plan.mesh_plan.nyquist}"
            ),
            settings=settings,
            charge_field=self.plan.charge_field,
            prepared_identity=self.prepared_identity,
            bound=bound,
            capacity=self.system.capacity,
            charges=charges,
            parts=parts,
            geometry=geometry,
            reciprocal=reciprocal,
        )


__all__ = [
    "ChargeNeutralityPolicy",
    "CoulombSplittingEstimate",
    "DampedShiftedForceCoulombPotential",
    "DirectCoulombPotential",
    "EwaldBoundaryConvention",
    "EwaldReferencePotential",
    "PairCoulombEvidence",
    "ParticleMeshEwaldPotential",
    "PeriodicCoulombEvidence",
    "ScreenedCoulombPotential",
    "ScreenedTruncation",
    "estimate_ewald_splitting",
    "estimate_pme_splitting",
]
