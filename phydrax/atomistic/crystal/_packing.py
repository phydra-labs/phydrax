#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded native molecular-crystal packing by overlap relief.

`CrystalPackingPlan` binds a `MolecularCrystalInput` (formula unit, conformer
pools, contact distances and formula-unit volume), a fixed or weighted
space-group policy over admitted standard settings, an integer ``Z'`` (the
asymmetric unit holds ``Z'`` formula units at the general position), the
chirality policy, volume/shape/height policies, the relief numerics and the
explicit contact resources.  `PreparedCrystalPacker` holds one stateful
`PreparedCrystalGroupPacker` sampler per admitted setting (their atom counts
differ) and the ordinal-addressed group route; it plugs into the generation
stream and campaign through the shared sampler lane contract.

One attempt is one lane:

1. Ordinal-addressed draws (roles ``conformer``, ``center``, ``orientation``,
   ``volume``, ``cell``) choose a conformer of each ASU molecule, fractional
   centers, uniform proper rotations, a reference volume
   ``V_ref = v Z V_fu`` and an initial invariant metric of volume
   ``oversample * V_ref`` among a bounded number of shape draws; an attempt
   whose draws all violate the height/axis-ratio policy ends ``CELL_INVALID``.
2. Each relief step expands the ASU through the native ASU owner, refreshes
   the native neighborhood bundle (one ``IMAGE_PAIR_SUM`` requirement whose
   cutoff is the largest contact distance) and evaluates the overlap penalty

   ``E = sum_half-routes 1/2 max(0, r_ij - |d_ij|)^2``

   over canonical half image routes, i.e. every unordered pair-image exactly
   once (``1/2`` of the directed image sum).  A route is excluded only when
   both atoms belong to the same molecular copy with zero net lattice shift on
   their whole-molecule branch; different copies and every periodic self-image
   of a molecule count.  Gradients pull back through the expansion to the ASU
   centers, to a rotation vector at each orientation (the torque) and to the
   metric chart coordinates.
3. The rigid step preconditions the force by the molecule mass and the torque
   by the native principal inertia; the largest induced atom displacement is
   capped at ``min(maximum_step, maximum overlap)``; orientations move by the
   native unit-quaternion retraction.  The cell step is the tangent
   (deviatoric) projection of the chart gradient with a metric-strain cap,
   retracted exactly onto the pre-step volume (the projection is volume
   preserving only to first order), followed by compression of a declared
   fraction of the excess volume above ``V_ref`` (the volume never grows); a
   cell update violating the height floor, axis ratio or Cholesky certificate
   is refused and counted.
4. At every ``check_interval`` steps an attempt is accepted when its maximum
   overlap is at most ``overlap_tolerance`` and its volume satisfies
   ``|V - V_ref| <= volume_tolerance * V_ref``; otherwise it ends
   ``STEP_BUDGET_EXHAUSTED`` at ``maximum_steps``.  Neighborhood capacity
   overflow ends ``CAPACITY_REFUSED`` and is never accepted on truncated
   contacts.

The overlap penalty and its cell tensor are not a physical energy or stress
and overlap convergence is not energy convergence: accepted packings are
proposals for native physical relaxation and ranking.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import isfinite
from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._identity import NumericRevision, SemanticProvenance
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...axes import Axis, AxisKey
from ...discretization import (
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    PeriodicCell,
)
from ...discretization.particle import quaternion_rotation_matrix
from ...ein import contract
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...metrix._quaternion_state_geometry import ScalarFirstQuaternionStateGeometry
from ...typing import parse, PRNGKey
from ...units import ANGSTROM, derived_unit, ELECTRONVOLT, ONE
from .._conditioning import AtomisticConditioning
from .._generation import (
    AbstractAtomisticGenerationSink,
    AbstractPreparedAtomisticSampler,
    AbstractProposalFilter,
    AtomisticCellAdmission,
    AtomisticCellBounds,
    AtomisticChemistryAdmission,
    AtomisticFixedRoute,
    AtomisticGeneratedProposal,
    AtomisticGenerationState,
    AtomisticGenerationStreamPlan,
    AtomisticGenerationSupport,
    AtomisticMemoryGenerationSink,
    AtomisticPayloadSpec,
    AtomisticProposalBlock,
    AtomisticProposalStatus,
    AtomisticSamplerFamily,
    AtomisticTopologyAdmission,
    AtomisticWeightedRoute,
    cell_heights,
    PreparedAtomisticGenerationStream,
    role_keys,
)
from .._neighborhood_bundle import (
    AbstractAtomisticImageSearchPolicy,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodBundleState,
    AtomisticNeighborhoodGeometry,
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticNeighborhoodSource,
    AtomisticPairCounting,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
    PreparedAtomisticNeighborhoodBundle,
)
from .._sites import AtomisticSiteDomain
from .._structure_store import (
    AtomisticStoreField,
    AtomisticStoreLevel,
    AtomisticStructureRecord,
    AtomisticStructureStoreSchema,
)
from .._system import AtomisticSystemPlan
from .._topology import MolecularTopologyPlan
from .._units import AtomisticUnitSystem
from ._asu import (
    AsymmetricUnitLayout,
    AsymmetricUnitPlan,
    AsymmetricUnitSite,
    AsymmetricUnitState,
    ChiralityPolicy,
    expand_asymmetric_unit,
    ExpandedCrystal,
)
from ._conformers import ConformerGenerationPlan, ConformerGenerationResult
from ._input import ConformerPool, MolecularCrystalInput
from ._lattice import metric_from_cell_parameters
from ._setting import CrystalSystem, SpaceGroupSetting
from ._wyckoff import site_of_point


CONTACT_OWNER = "crystal-contact-relief"
_SMALL3 = SmallLinearSolvePlan(3)
_ROTATIONS = ScalarFirstQuaternionStateGeometry(convention="spatial")
# A generic point with distinct prime denominators; admission still verifies
# it is the general position of each setting.
_GENERAL_POINT = (Fraction(1, 7), Fraction(2, 11), Fraction(3, 13))
_MAXIMUM_CHART_DIMENSION = 6
_CELL = Axis(AxisKey("atomistic", "cartesian"), 3, ("x", "y", "z"))
_QUATERNION = Axis(AxisKey("crystal", "quaternion"), 4, ("w", "x", "y", "z"))
_CHART = Axis(AxisKey("crystal", "metric-chart"), _MAXIMUM_CHART_DIMENSION)
_SQUARED_LENGTH = derived_unit("angstrom^2", ((ANGSTROM, 2),))
_CUBED_LENGTH = derived_unit("angstrom^3", ((ANGSTROM, 3),))

# Scientific definition of the relief cell step and acceptance, bound into
# `CrystalReliefPolicy.policy_id`.
_CELL_STEP_DEFINITION = (
    "capped-tangent-metric-step;exact-pre-step-volume-retraction;"
    "excess-volume-compression"
)
_VOLUME_ACCEPTANCE_DEFINITION = "two-sided-relative-reference-volume"
# Scientific definition of one attempt bound into the group-packer law identity:
# its ordinal-addressed draw roles, the overlap penalty with its exclusion rule
# and the preconditioned rigid step.  Lanes, the contact-search algorithm and
# its capacities are execution resources and stay outside it.
_ATTEMPT_DEFINITION = (
    "ordinal-roles:conformer,center,orientation,volume,cell;"
    "general-position-asu;half-image-route-quadratic-overlap-penalty;"
    "same-copy-zero-shift-exclusion;mass-inertia-preconditioned-capped-rigid-step;"
    "unit-quaternion-retraction"
)

CrystalRecordStage = Literal["packed", "relaxed"]
_STAGE_CODES = {"packed": 0, "relaxed": 1}


def _positive(value: float, name: str, /) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _fraction_of_one(value: float, name: str, /) -> float:
    number = float(value)
    if not isfinite(number) or not 0.0 <= number < 1.0:
        raise ValueError(f"{name} must lie in [0, 1).")
    return number


def _count(value: int, name: str, /, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


# -- policies ---------------------------------------------------------------------------------


@final
class CrystalSpaceGroupPolicy(StrictModule, NonTrainableState):
    """Fixed or prior-weighted choice among admitted settings.

    Weights are a caller-admitted prior with explicit provenance (for example
    a licensed CSD frequency table); no prior is inferred from group numbers.
    ``filter`` keeps the settings compatible with a target ``Z``, declared
    crystal systems and/or Sohncke-only (proper operations only) admission;
    an empty result refuses.  ``declared_weights`` are the admitted weights (a
    restoration witness); ``policy_id`` binds their normalization ``weights``.
    """

    settings: tuple[SpaceGroupSetting, ...]
    declared_weights: tuple[float, ...] = eqx.field(static=True)
    weights: tuple[float, ...] = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        settings: Sequence[SpaceGroupSetting],
        /,
        *,
        weights: Sequence[float] | None = None,
        provenance: str = "fixed",
    ) -> None:
        members = tuple(settings)
        if not members or any(
            not isinstance(item, SpaceGroupSetting) for item in members
        ):
            raise TypeError("settings must be a non-empty sequence of SpaceGroupSetting.")
        if len({item.setting_id for item in members}) != len(members):
            raise ValueError("Settings must be distinct.")
        if weights is None:
            if len(members) != 1:
                raise ValueError("Several settings require explicit admitted weights.")
            values = (1.0,)
        else:
            values = tuple(float(value) for value in weights)
            if len(values) != len(members) or any(
                not isfinite(value) or value <= 0.0 for value in values
            ):
                raise ValueError(
                    "weights declare one positive finite weight per setting."
                )
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("provenance must be a non-empty statement.")
        if len(members) > 1 and provenance == "fixed":
            raise ValueError("A weighted policy needs the prior's actual provenance.")
        total = sum(values)
        self.settings = members
        self.declared_weights = values
        self.weights = tuple(value / total for value in values)
        self.provenance = provenance.strip()
        self.policy_id = canonical_fingerprint(
            {
                "kind": "crystal-space-group-policy",
                "settings": [item.setting_id for item in members],
                "weights": list(self.weights),
                "provenance": self.provenance,
            }
        )

    @classmethod
    def fixed(cls, setting: SpaceGroupSetting, /) -> CrystalSpaceGroupPolicy:
        return cls((setting,))

    def filter(
        self,
        *,
        z: int | None = None,
        z_prime: int = 1,
        crystal_systems: Sequence[CrystalSystem] | None = None,
        sohncke_only: bool = False,
    ) -> CrystalSpaceGroupPolicy:
        """Settings compatible with ``Z = |G| Z'``, crystal systems and handedness."""
        prime = _count(z_prime, "z_prime")
        systems = None if crystal_systems is None else set(crystal_systems)
        kept = [
            (setting, weight)
            for setting, weight in zip(self.settings, self.weights, strict=True)
            if (z is None or setting.operation_count * prime == int(z))
            and (systems is None or setting.crystal_system in systems)
            and (not sohncke_only or setting.sohncke)
        ]
        if not kept:
            raise ValueError("No admitted setting satisfies the group filter.")
        if len(kept) == 1:
            provenance = (
                "fixed" if len(self.settings) == 1 else self.provenance + "|filtered"
            )
            return CrystalSpaceGroupPolicy(
                (kept[0][0],), weights=(1.0,), provenance=provenance
            )
        return CrystalSpaceGroupPolicy(
            [item for item, _ in kept],
            weights=[weight for _, weight in kept],
            provenance=self.provenance + "|filtered",
        )


@final
class CrystalCellPolicy(StrictModule, NonTrainableState):
    """Volume, shape and height policy of initial and relieved cells.

    ``volume_range`` scales ``Z V_fu`` (``scaled=True``) or is absolute in
    cubed angstrom.  Initial shapes draw log-lengths with ratio at most
    ``maximum_axis_ratio`` and angles uniform in ``angle_range`` degrees, are
    projected onto the setting's invariant metric subspace and rescaled to
    ``oversample * V_ref``; ``shape_draws`` bounds the vectorized draws.
    Relief refuses cell updates below ``minimum_height_floor`` and accepted or
    initial cells need every height at least ``minimum_height``.
    """

    volume_range: tuple[float, float] = eqx.field(static=True)
    scaled: bool = eqx.field(static=True)
    oversample: float = eqx.field(static=True)
    maximum_axis_ratio: float = eqx.field(static=True)
    angle_range: tuple[float, float] = eqx.field(static=True)
    minimum_height: float = eqx.field(static=True)
    minimum_height_floor: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    shape_draws: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        volume_range: tuple[float, float] = (0.9, 1.2),
        scaled: bool = True,
        oversample: float = 1.5,
        maximum_axis_ratio: float = 10.0,
        angle_range: tuple[float, float] = (60.0, 120.0),
        minimum_height: float = 3.8,
        minimum_height_floor: float = 2.5,
        maximum_condition: float = 1.0e3,
        shape_draws: int = 16,
    ) -> None:
        low, high = (_positive(value, "volume_range") for value in volume_range)
        if high < low:
            raise ValueError("volume_range must be ordered.")
        oversample_ = float(oversample)
        if not isfinite(oversample_) or oversample_ < 1.0:
            raise ValueError("oversample must be finite and at least one.")
        ratio = float(maximum_axis_ratio)
        if not isfinite(ratio) or ratio < 1.0:
            raise ValueError("maximum_axis_ratio must be at least one.")
        alpha, beta = (float(value) for value in angle_range)
        if not (0.0 < alpha <= beta < 180.0):
            raise ValueError("angle_range must lie inside (0, 180) degrees.")
        height = _positive(minimum_height, "minimum_height")
        floor = _positive(minimum_height_floor, "minimum_height_floor")
        if floor > height:
            raise ValueError("minimum_height_floor must not exceed minimum_height.")
        condition = float(maximum_condition)
        if not isfinite(condition) or condition < 1.0:
            raise ValueError("maximum_condition must be at least one.")
        self.volume_range = (low, high)
        self.scaled = bool(scaled)
        self.oversample = oversample_
        self.maximum_axis_ratio = ratio
        self.angle_range = (alpha, beta)
        self.minimum_height = height
        self.minimum_height_floor = floor
        self.maximum_condition = condition
        self.shape_draws = _count(shape_draws, "shape_draws")
        self.policy_id = canonical_fingerprint(
            {
                "kind": "crystal-cell-policy",
                "volume_range": [low, high],
                "scaled": self.scaled,
                "oversample": oversample_,
                "maximum_axis_ratio": ratio,
                "angle_range": [alpha, beta],
                "minimum_height": height,
                "minimum_height_floor": floor,
                "maximum_condition": condition,
                "shape_draws": self.shape_draws,
            }
        )


@final
class CrystalReliefPolicy(StrictModule, NonTrainableState):
    """Explicit overlap-relief numerics (lengths in angstrom).

    ``step_scale`` multiplies mass/inertia-preconditioned forces and torques;
    ``maximum_step`` caps induced atom displacement; ``cell_step_scale``
    multiplies the chart gradient; ``maximum_cell_strain`` caps the relative
    metric change per step (``|dG| <= 2 eps |G|``); ``volume_compression`` is
    the fraction of the excess volume above ``V_ref`` removed per step.

    ``policy_id`` binds these numerics and the relief method definition: the
    finite cell step is retracted exactly onto its pre-step volume before
    compression, and acceptance is two-sided, ``|V - V_ref| <= volume_tolerance
    * V_ref``.  Checkpoints, stored records and recipes of a relief whose
    identity lacks these semantics (a first-order-only deviatoric step and an
    upper-only volume check) therefore fail identity admission; re-prepare the
    plan from its constructor arguments and pack again.
    """

    maximum_steps: int = eqx.field(static=True)
    check_interval: int = eqx.field(static=True)
    overlap_tolerance: float = eqx.field(static=True)
    volume_tolerance: float = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    step_scale: float = eqx.field(static=True)
    cell_step_scale: float = eqx.field(static=True)
    maximum_cell_strain: float = eqx.field(static=True)
    volume_compression: float = eqx.field(static=True)
    maximum_rotation: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_steps: int = 500,
        check_interval: int = 10,
        overlap_tolerance: float = 0.05,
        volume_tolerance: float = 0.02,
        maximum_step: float = 0.3,
        step_scale: float = 0.12,
        cell_step_scale: float = 0.01,
        maximum_cell_strain: float = 0.003,
        volume_compression: float = 0.05,
        maximum_rotation: float = 0.5,
    ) -> None:
        self.maximum_steps = _count(maximum_steps, "maximum_steps")
        self.check_interval = _count(check_interval, "check_interval")
        self.overlap_tolerance = _positive(overlap_tolerance, "overlap_tolerance")
        self.volume_tolerance = float(volume_tolerance)
        if not isfinite(self.volume_tolerance) or self.volume_tolerance < 0.0:
            raise ValueError("volume_tolerance must be finite and nonnegative.")
        self.maximum_step = _positive(maximum_step, "maximum_step")
        self.step_scale = _positive(step_scale, "step_scale")
        self.cell_step_scale = float(cell_step_scale)
        if not isfinite(self.cell_step_scale) or self.cell_step_scale < 0.0:
            raise ValueError("cell_step_scale must be finite and nonnegative.")
        self.maximum_cell_strain = _positive(maximum_cell_strain, "maximum_cell_strain")
        self.volume_compression = _fraction_of_one(
            volume_compression, "volume_compression"
        )
        self.maximum_rotation = _positive(maximum_rotation, "maximum_rotation")
        if self.maximum_rotation >= 1.0:
            raise ValueError("maximum_rotation must stay below one radian.")
        self.policy_id = canonical_fingerprint(
            {
                "kind": "crystal-relief-policy",
                "cell_step": _CELL_STEP_DEFINITION,
                "volume_acceptance": _VOLUME_ACCEPTANCE_DEFINITION,
                "maximum_steps": self.maximum_steps,
                "check_interval": self.check_interval,
                "overlap_tolerance": self.overlap_tolerance,
                "volume_tolerance": self.volume_tolerance,
                "maximum_step": self.maximum_step,
                "step_scale": self.step_scale,
                "cell_step_scale": self.cell_step_scale,
                "maximum_cell_strain": self.maximum_cell_strain,
                "volume_compression": self.volume_compression,
                "maximum_rotation": self.maximum_rotation,
            }
        )


@final
class CrystalContactResources(StrictModule, NonTrainableState):
    """Explicit image-search, skin and storage of the contact bundle.

    The search covers every admitted cell because the bundle is prepared on a
    cubic reference whose heights equal the cell policy's height floor; image,
    route and degree storage are declared capacity ladders (``maximum_case_edges``
    per lane).  Overflow is a capacity refusal, never a truncated contact set.
    """

    search: AbstractAtomisticImageSearchPolicy
    search_capacities: tuple[ParticleImageCapacity, ...]
    role_capacities: tuple[AtomisticRoleCapacity, ...]
    skin: float = eqx.field(static=True)
    maximum_image_count: int = eqx.field(static=True)
    resources_id: str = eqx.field(static=True)

    def __init__(
        self,
        search: AbstractAtomisticImageSearchPolicy,
        /,
        *,
        search_capacities: Sequence[ParticleImageCapacity],
        role_capacities: Sequence[AtomisticRoleCapacity],
        skin: float,
        maximum_image_count: int = 4096,
    ) -> None:
        if not isinstance(search, AbstractAtomisticImageSearchPolicy):
            raise TypeError("search must be a native atomistic image search policy.")
        sources = tuple(search_capacities)
        roles = tuple(role_capacities)
        if not sources or not roles:
            raise ValueError("Contact resources declare search and role capacities.")
        self.search = search
        self.search_capacities = sources
        self.role_capacities = roles
        self.skin = _positive(skin, "skin")
        self.maximum_image_count = _count(maximum_image_count, "maximum_image_count")
        self.resources_id = canonical_fingerprint(
            {
                "kind": "crystal-contact-resources",
                "search": search.policy_id,
                "search_capacities": [item.capacity_id for item in sources],
                "role_capacities": [item.capacity_id for item in roles],
                "skin": self.skin,
                "maximum_image_count": self.maximum_image_count,
            }
        )


# -- plan ---------------------------------------------------------------------------------------


def native_conformer_pools(
    plans: Sequence[ConformerGenerationPlan],
    root_key: PRNGKey,
    /,
    *,
    count: int,
    weights: Sequence[Sequence[float] | None] | None = None,
) -> tuple[tuple[ConformerPool, ...], tuple[ConformerGenerationResult, ...]]:
    """Generate native conformer pools for components (source-to-conformer route).

    Each plan generates ordinals ``[0, count)`` from its own semantic address;
    a component without an accepted conformer refuses with its attempt
    evidence rather than packing without that component.
    """
    members = tuple(plans)
    if not members or any(
        not isinstance(plan, ConformerGenerationPlan) for plan in members
    ):
        raise TypeError("plans must be ConformerGenerationPlan values.")
    requested = _count(count, "count")
    pool_weights = (None,) * len(members) if weights is None else tuple(weights)
    if len(pool_weights) != len(members):
        raise ValueError("weights must align with the plans.")
    results = tuple(plan.generate(root_key, count=requested) for plan in members)
    pools = []
    for plan, result, weight in zip(members, results, pool_weights, strict=True):
        if not result.conformers:
            raise ValueError(
                f"Native conformer generation for {plan.component.name!r} accepted no "
                f"conformer (status {result.status!r}); the packing input is incomplete."
            )
        pools.append(result.pool(plan.component, weights=weight))
    return tuple(pools), results


@final
class CrystalPackingPlan(StrictModule, NonTrainableState):
    """Complete scientific and resource definition of a packing campaign.

    ``law_id`` is the scientific packing law: molecular input (components,
    conformer measure, contacts, volume), space-group policy, ``Z'``,
    chirality, cell prior and relief numerics with the attempt definition.
    ``plan_id`` is the complete plan: the law with the contact resources and
    the atom bound.  Resources choose how an attempt executes, never which
    attempt is drawn, so proposals and stored lineages carry the law while
    recipes bind the complete plan.
    """

    molecular_input: MolecularCrystalInput
    groups: CrystalSpaceGroupPolicy
    cell: CrystalCellPolicy
    relief: CrystalReliefPolicy
    resources: CrystalContactResources
    z_prime: int = eqx.field(static=True)
    chirality: ChiralityPolicy = eqx.field(static=True)
    maximum_atoms: int = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        molecular_input: MolecularCrystalInput,
        groups: CrystalSpaceGroupPolicy,
        /,
        *,
        z_prime: int,
        chirality: ChiralityPolicy,
        resources: CrystalContactResources,
        cell: CrystalCellPolicy | None = None,
        relief: CrystalReliefPolicy | None = None,
        maximum_atoms: int = 4096,
    ) -> None:
        if not isinstance(molecular_input, MolecularCrystalInput):
            raise TypeError("molecular_input must be a MolecularCrystalInput.")
        if not isinstance(groups, CrystalSpaceGroupPolicy):
            raise TypeError("groups must be a CrystalSpaceGroupPolicy.")
        if not isinstance(resources, CrystalContactResources):
            raise TypeError("resources must be CrystalContactResources.")
        cell_ = CrystalCellPolicy() if cell is None else cell
        relief_ = CrystalReliefPolicy() if relief is None else relief
        if not isinstance(cell_, CrystalCellPolicy) or not isinstance(
            relief_, CrystalReliefPolicy
        ):
            raise TypeError("cell and relief must be crystal packing policies.")
        prime = _count(z_prime, "z_prime")
        chirality_ = parse(chirality, ChiralityPolicy, "chirality")
        limit = _count(maximum_atoms, "maximum_atoms")
        atoms_per_unit = molecular_input.formula_unit_atom_count
        for setting in groups.settings:
            atoms = setting.operation_count * prime * atoms_per_unit
            if atoms > limit:
                raise ValueError(
                    f"Setting {setting.hall_symbol!r} expands to {atoms} atoms, above "
                    f"maximum_atoms={limit}."
                )
        self.molecular_input = molecular_input
        self.groups = groups
        self.cell = cell_
        self.relief = relief_
        self.resources = resources
        self.z_prime = prime
        self.chirality = chirality_
        self.maximum_atoms = limit
        self.law_id = canonical_fingerprint(
            {
                "kind": "crystal-packing-law",
                "attempt": _ATTEMPT_DEFINITION,
                "input": molecular_input.input_id,
                "groups": groups.policy_id,
                "cell": cell_.policy_id,
                "relief": relief_.policy_id,
                "z_prime": prime,
                "chirality": chirality_,
            }
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "crystal-packing-plan",
                "law": self.law_id,
                "resources": resources.resources_id,
                "maximum_atoms": limit,
            }
        )

    def asymmetric_unit(self, setting: SpaceGroupSetting, /) -> AsymmetricUnitPlan:
        """ASU of ``Z'`` formula units at the general position of ``setting``."""
        site = site_of_point(setting, _GENERAL_POINT)
        if site.special or site.stabilizer_order != 1:
            raise ValueError("The packing reference point is not a general position.")
        sites = [
            AsymmetricUnitSite(pool, site)
            for _ in range(self.z_prime)
            for pool, count in zip(
                self.molecular_input.pools,
                self.molecular_input.formula_counts,
                strict=True,
            )
            for _ in range(count)
        ]
        return AsymmetricUnitPlan(setting, sites, chirality=self.chirality)

    def prepare(self, *, lanes: int) -> PreparedCrystalPacker:
        return PreparedCrystalPacker(self, lanes=_count(lanes, "lanes"))


# -- lane method --------------------------------------------------------------------------------


@final
class CrystalReliefMethod(StrictModule):
    """Per-lane relief continuation (leading lane axis).

    ``status`` holds `AtomisticProposalStatus` bits; ``accepted`` lanes passed
    a check.  ``resources`` is the case-batched accepted contact bundle state.
    """

    centers: Array
    orientations: Array
    coefficients: Array
    conformers: Array
    reference_volume: Array
    steps: Array
    status: Array
    accepted: Array
    maximum_overlap: Array
    overlap_energy: Array
    cell_refusals: Array
    resources: AtomisticNeighborhoodBundleState


def _payload_spec(sites: int, /) -> AtomisticPayloadSpec:
    return AtomisticPayloadSpec(
        (
            ("asu_centers", (sites, 3), "float64"),
            ("asu_orientations", (sites, 4), "float64"),
            ("cell_coefficients", (_MAXIMUM_CHART_DIMENSION,), "float64"),
            ("conformer_indices", (sites,), "int32"),
            ("group_index", (), "int32"),
            ("steps", (), "int32"),
            ("maximum_overlap", (), "float64"),
            ("overlap_energy", (), "float64"),
            ("reference_volume", (), "float64"),
            ("volume", (), "float64"),
            ("cell_refusals", (), "int32"),
        )
    )


def _uniform_quaternions(keys: Array, /) -> Array:
    draws = jax.vmap(lambda key: jr.normal(key, (4,), dtype=jnp.float64))(keys)
    unit = draws / jnp.linalg.norm(draws, axis=-1, keepdims=True)
    return jnp.where(unit[..., :1] < 0.0, -unit, unit)


@final
class PreparedCrystalGroupPacker(AbstractPreparedAtomisticSampler):
    """Stateful relief sampler of one admitted setting (one attempt per lane).

    The compiled sampler holds the numerical `AsymmetricUnitLayout` and
    derived arrays only; the `AsymmetricUnitPlan` with its molecular
    witnesses stays with the host owner `PreparedCrystalPacker`.

    ``sampler_id`` is the member's scientific law (molecular input, ASU with
    its setting, ``Z'`` and chirality, cell prior, relief numerics and the
    attempt definition); proposal identities and ordinal RNG addresses derive
    from it, so they do not depend on the lane count or contact resources.
    Those execution choices are `signature_facts` (the executable layout that
    stage signatures and generation checkpoints bind).  ``group_index`` is the
    member's address in the host family (route index and ``group_index``
    payload used by `PreparedCrystalPacker.packed`), not part of its law.
    """

    layout: AsymmetricUnitLayout
    bundle: PreparedAtomisticNeighborhoodBundle
    requirement: AtomisticNeighborhoodRequirement
    support: AtomisticGenerationSupport
    payload_spec: AtomisticPayloadSpec
    cell_policy: CrystalCellPolicy
    relief: CrystalReliefPolicy
    contact: Array
    atom_contact: Array
    atom_copy: Array
    site_mass: Array
    conformer_moments: Array
    site_atom_mask: Array
    site_logits: Array
    volume_low: float = eqx.field(static=True)
    volume_high: float = eqx.field(static=True)
    group_index: int = eqx.field(static=True)
    lanes: int = eqx.field(static=True)
    slot: int = eqx.field(static=True)
    resources_id: str = eqx.field(static=True)
    sampler_id: str = eqx.field(static=True)
    stateful: bool = eqx.field(static=True)
    idempotent: bool = eqx.field(static=True)

    def __init__(
        self,
        plan: CrystalPackingPlan,
        asu: AsymmetricUnitPlan,
        /,
        *,
        group_index: int,
        lanes: int,
    ) -> None:
        if not isinstance(asu, AsymmetricUnitPlan):
            raise TypeError("asu must be an AsymmetricUnitPlan.")
        if asu.setting.setting_id != plan.groups.settings[group_index].setting_id:
            raise ValueError("asu must be the plan's ASU of setting group_index.")
        molecular = plan.molecular_input
        # Formula-atom row of every (component, local atom): contact radii depend
        # on the component atom, not on which formula-unit copy it belongs to.
        first_row: dict[tuple[int, int], int] = {}
        for row, (component, _, atom) in enumerate(molecular.formula_atom_sources):
            first_row.setdefault((component, atom), row)
        pool_index = {
            pool.component.component_id: index
            for index, pool in enumerate(molecular.pools)
        }
        site_pool = [pool_index[site.pool.component.component_id] for site in asu.sites]
        atom_contact = np.asarray(
            [
                first_row[(site_pool[int(site)], int(local))]
                for site, local in zip(asu.atom_site, asu.atom_local, strict=True)
            ],
            dtype=np.int32,
        )
        count = asu.atom_count
        contact = np.asarray(
            jax.device_get(molecular.contact_distances), dtype=np.float64
        )
        cutoff = float(np.max(contact))
        requirement = AtomisticNeighborhoodRequirement(
            CONTACT_OWNER, AtomisticNeighborhoodRole.IMAGE_PAIR_SUM, cutoff
        )
        resources = plan.resources
        source = AtomisticNeighborhoodSource(
            "crystal-contacts",
            AtomisticSiteDomain.PHYSICAL_ATOMS,
            (requirement,),
            resources.search,
            ParticleImageCapacityLadder(resources.search_capacities),
            skin=resources.skin,
        )
        bundle_plan = AtomisticNeighborhoodBundlePlan(
            (requirement,),
            (source,),
            (AtomisticRoleCapacityLadder(requirement, resources.role_capacities),),
            case_count=lanes,
        )
        species = []
        masses = []
        for site, local in zip(asu.atom_site, asu.atom_local, strict=True):
            component = asu.sites[int(site)].pool.component
            species.append(component.atomic_numbers[int(local)])
            masses.append(component.masses[int(local)])
        bonds = []
        for copy in range(asu.copy_count):
            atoms = np.flatnonzero(asu.atom_copy == copy)
            component = asu.sites[int(asu.copy_site[copy])].pool.component
            for left, right in component.bonds:
                bonds.append((int(atoms[left]), int(atoms[right])))
        # Searched on a cubic reference at the height floor, the prepared image
        # stencil covers every admitted cell (all heights >= the floor).
        reference = PeriodicCell(
            plan.cell.minimum_height_floor * np.eye(3),
            maximum_image_count=resources.maximum_image_count,
        )
        system = AtomisticSystemPlan(
            np.arange(1, count + 1, dtype=np.int64),
            np.asarray(species, dtype=np.int32),
            np.asarray(masses, dtype=np.float64),
            AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
            molecule_ids=np.asarray(asu.atom_copy, dtype=np.int64),
            topology=MolecularTopologyPlan(
                bonds=np.asarray(
                    [[a + 1, b + 1] for a, b in bonds], dtype=np.int64
                ).reshape((-1, 2))
            ),
            cell=reference,
            name="crystal-contact-relief",
        ).prepare()
        bundle = bundle_plan.prepare(system)
        slot = bundle.slot(requirement)
        vocabulary: dict[int, float] = {}
        for number, mass in zip(species, masses, strict=True):
            if vocabulary.setdefault(int(number), float(mass)) != float(mass):
                raise ValueError(
                    "Atoms of one element carry different masses; the support mass table "
                    "needs one mass per element."
                )
        z = asu.operation_count * plan.z_prime
        formula_volume = molecular.formula_unit_volume
        scale = z * formula_volume if plan.cell.scaled else 1.0
        low, high = plan.cell.volume_range
        self.volume_low = low * scale
        self.volume_high = high * scale
        bounds = AtomisticCellBounds(
            minimum_height=plan.cell.minimum_height_floor,
            maximum_condition=plan.cell.maximum_condition,
            minimum_volume=self.volume_low,
            maximum_volume=plan.cell.oversample * self.volume_high,
        )
        names = tuple(sorted(vocabulary))
        support = AtomisticGenerationSupport(
            atom_capacity=count,
            species_vocabulary=names,
            vocabulary_masses=tuple(vocabulary[name] for name in names),
            mass_provenance="molecular-component-masses:" + molecular.input_id,
            chemistry=AtomisticChemistryAdmission.FIXED,
            length_unit=ANGSTROM,
            fixed_species=tuple(int(value) for value in species),
            fixed_mask=(True,) * count,
            topology=AtomisticTopologyAdmission.FIXED,
            fixed_bonds=tuple(bonds),
            total_charge=z * molecular.total_charge,
            cell_admission=AtomisticCellAdmission.VARIABLE_CELL,
            cell_bounds=bounds,
        )
        width = int(asu.conformer_table.shape[1])
        site_atoms = np.zeros((asu.site_count, width), dtype=bool)
        for index, atoms in enumerate(asu.site_atom_count):
            site_atoms[index, : int(atoms)] = True
        moments = np.stack(
            [
                np.asarray(jax.device_get(conformer.principal_moments), dtype=np.float64)
                for site in asu.sites
                for conformer in site.pool.conformers
            ]
        )
        logits = np.full((asu.site_count, int(np.max(asu.site_conformer_count))), -np.inf)
        for index, site in enumerate(asu.sites):
            logits[index, : len(site.pool.weights)] = np.log(
                np.asarray(site.pool.weights)
            )
        self.layout = AsymmetricUnitLayout(asu)
        self.bundle = bundle
        self.requirement = requirement
        self.support = support
        self.payload_spec = _payload_spec(asu.site_count)
        self.cell_policy = plan.cell
        self.relief = plan.relief
        self.contact = jnp.asarray(contact)
        self.atom_contact = jnp.asarray(atom_contact)
        self.atom_copy = jnp.asarray(asu.atom_copy, dtype=jnp.int32)
        self.site_mass = jnp.asarray(
            [sum(site.pool.component.masses) for site in asu.sites], dtype=jnp.float64
        )
        self.conformer_moments = jnp.asarray(moments)
        self.site_atom_mask = jnp.asarray(site_atoms)
        self.site_logits = jnp.asarray(logits)
        self.group_index = int(group_index)
        self.lanes = int(lanes)
        self.slot = int(slot)
        self.resources_id = resources.resources_id
        self.stateful = True
        self.idempotent = False
        self.sampler_id = canonical_fingerprint(
            {
                "kind": "crystal-group-packer",
                "attempt": _ATTEMPT_DEFINITION,
                "input": molecular.input_id,
                "asu": asu.plan_id,
                "z_prime": plan.z_prime,
                "cell": plan.cell.policy_id,
                "relief": plan.relief.policy_id,
            }
        )

    # -- identities -----------------------------------------------------------------

    def signature_facts(self) -> dict[str, str | int]:
        return {
            "kind": "crystal-group-packer",
            "atoms": self.layout.atom_count,
            "sites": self.layout.site_count,
            "operations": self.layout.operation_count,
            "chart_dimension": self.layout.chart.dimension,
            "lanes": self.lanes,
            "group_index": self.group_index,
            "maximum_steps": self.relief.maximum_steps,
            "bundle": self.bundle.plan.plan_id,
            "prepared_bundle": self.bundle.prepared_id,
            "resources": self.resources_id,
            "precision": "float64",
            "layout": f"lanes={self.lanes};atoms={self.layout.atom_count};sites={self.layout.site_count}",
            "derivatives": "none",
        }

    def numeric_revision(self) -> NumericRevision:
        return NumericRevision(
            SemanticProvenance(
                {"kind": "crystal-group-packer", "sampler": self.sampler_id}
            ),
            {
                "conformers": np.asarray(self.layout.conformer_table),
                "contact": np.asarray(jax.device_get(self.contact)),
            },
        )

    # -- geometry -----------------------------------------------------------------------

    def _expand(
        self, centers: Array, orientations: Array, coefficients: Array, conformers: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Lane expansions: positions, fractional, cell, offsets, certificate."""

        def one(
            center: Array, orientation: Array, coefficient: Array, conformer: Array
        ) -> tuple[Array, Array, Array, Array, Array]:
            expansion = expand_asymmetric_unit(
                self.layout,
                AsymmetricUnitState(center, orientation, coefficient, conformer),
            )
            return (
                expansion.positions,
                expansion.fractional,
                expansion.cell.vectors,
                expansion.site_offsets,
                expansion.cell.successful & expansion.offset_solve_successful,
            )

        return jax.vmap(one)(centers, orientations, coefficients, conformers)

    def _geometry(
        self, fractional: Array, vectors: Array, /
    ) -> tuple[AtomisticNeighborhoodGeometry, Array]:
        counts = jax.lax.stop_gradient(jnp.floor(fractional)).astype(jnp.int32)
        wrapped = contract("lni,lij->lnj", fractional - counts, vectors)
        geometry = self.bundle.geometry(
            wrapped, cell_vectors=vectors, image_counts=counts
        )
        return geometry, counts

    def _overlaps(
        self,
        positions: Array,
        vectors: Array,
        counts: Array,
        resources: AtomisticNeighborhoodBundleState,
        running: Array | None = None,
        /,
    ) -> tuple[Array, Array, Array]:
        """Per-lane penalty, maximum overlap and route overlap of the bound routes."""
        view = resources.view(self.slot)
        if (
            view.counting is not AtomisticPairCounting.IMAGE_HALF
            or view.directed_multiplicity != 2
        ):
            raise RuntimeError(
                "Contact relief requires the canonical half image-pair view."
            )
        relation = view.relation
        lanes, atoms = positions.shape[:2]
        flat = positions.reshape((lanes * atoms, 3))
        flat_counts = counts.reshape((lanes * atoms, 3))
        valid = relation.valid
        source = jnp.where(valid, relation.source_indices, 0)
        receiver = jnp.where(valid, relation.receiver_indices, 0)
        case = jnp.where(valid, relation.route_cases, 0)
        # Route shifts address the wrapped geometry; on the unwrapped
        # whole-molecule branch the net shift is n - c[receiver] + c[source].
        shift = relation.image_shifts - flat_counts[receiver] + flat_counts[source]
        local_source = source % atoms
        local_receiver = receiver % atoms
        same_copy = self.atom_copy[local_source] == self.atom_copy[local_receiver]
        intramolecular = same_copy & jnp.all(shift == 0, axis=-1)
        displacement = (
            flat[receiver]
            - flat[source]
            + contract("ei,eij->ej", shift.astype(flat.dtype), vectors[case])
        )
        squared = jnp.sum(displacement * displacement, axis=-1)
        included = valid & ~intramolecular
        if running is not None:
            included = included & running[case]
        distance = jnp.sqrt(jnp.where(included, squared, 1.0))
        radius = self.contact[
            self.atom_contact[local_source], self.atom_contact[local_receiver]
        ]
        overlap = jnp.where(included, jnp.maximum(radius - distance, 0.0), 0.0)
        energy = jax.ops.segment_sum(0.5 * overlap * overlap, case, num_segments=lanes)
        maximum = jax.ops.segment_max(overlap, case, num_segments=lanes)
        return energy, jnp.maximum(maximum, 0.0), overlap

    def _volume(self, coefficients: Array, /) -> Array:
        vectors = self.layout.chart.cell(coefficients).vectors
        volume, _ = cell_heights(vectors)
        return volume

    def _cell_admitted(self, vectors: Array, successful: Array, /) -> Array:
        _, heights = cell_heights(vectors)
        lengths = jnp.linalg.norm(vectors, axis=-1)
        ratio = jnp.max(lengths, axis=-1) / jnp.min(lengths, axis=-1)
        return (
            successful
            & jnp.all(jnp.isfinite(vectors), axis=(-2, -1))
            & jnp.all(heights >= self.cell_policy.minimum_height_floor, axis=-1)
            & (ratio <= self.cell_policy.maximum_axis_ratio)
        )

    # -- sampler hooks -----------------------------------------------------------------

    def vacant_method(self, lanes: int, /) -> CrystalReliefMethod:
        if lanes != self.lanes:
            raise ValueError(
                f"This packer's contact bundle is prepared for {self.lanes} lanes."
            )
        cell = self.bundle.cell
        if cell is None:
            raise RuntimeError(
                "The packer contact bundle must carry its periodic reference cell."
            )
        sites = self.layout.site_count
        zeros = jnp.zeros((lanes,), dtype=jnp.float64)
        reference = jnp.broadcast_to(
            cell.vectors.astype(jnp.float64)[None], (lanes, 3, 3)
        )
        coefficients = jnp.broadcast_to(
            self.layout.chart.coefficients(reference[0] @ reference[0].T),
            (lanes, self.layout.chart.dimension),
        )
        geometry = self.bundle.geometry(
            jnp.zeros((lanes, self.layout.atom_count, 3), dtype=jnp.float64),
            cell_vectors=reference,
            image_counts=jnp.zeros((lanes, self.layout.atom_count, 3), dtype=jnp.int32),
        )
        return CrystalReliefMethod(
            centers=jnp.zeros((lanes, sites, 3), dtype=jnp.float64),
            orientations=jnp.zeros((lanes, sites, 4), dtype=jnp.float64)
            .at[..., 0]
            .set(1.0),
            coefficients=coefficients,
            conformers=jnp.zeros((lanes, sites), dtype=jnp.int32),
            reference_volume=zeros,
            steps=jnp.zeros((lanes,), dtype=jnp.int32),
            status=jnp.zeros((lanes,), dtype=jnp.int32),
            accepted=jnp.zeros((lanes,), dtype=jnp.bool_),
            maximum_overlap=zeros,
            overlap_energy=zeros,
            cell_refusals=jnp.zeros((lanes,), dtype=jnp.int32),
            resources=self.bundle.initialize(
                geometry, cases=jnp.zeros((lanes,), dtype=jnp.bool_)
            ),
        )

    def select_method(
        self, mask: Array, new: CrystalReliefMethod, old: CrystalReliefMethod, /
    ) -> CrystalReliefMethod:
        """Lane selection: arrays by mask, the contact bundle by its case commit."""
        arrays = jax.tree.map(
            lambda left, right: jnp.where(
                jnp.reshape(mask, mask.shape + (1,) * (jnp.ndim(left) - 1)), left, right
            ),
            eqx.tree_at(lambda value: value.resources, new, None),
            eqx.tree_at(lambda value: value.resources, old, None),
        )
        return eqx.tree_at(
            lambda value: value.resources,
            arrays,
            self.bundle.select_cases(mask, new.resources, old.resources),
            is_leaf=lambda value: value is None,
        )

    def _initial_cells(
        self, keys: Array, reference_volume: Array, /
    ) -> tuple[Array, Array]:
        policy = self.cell_policy
        draws = policy.shape_draws
        chart = self.layout.chart

        def lane(key: Array, volume: Array) -> tuple[Array, Array]:
            length_key, angle_key = jr.split(key)
            spread = jnp.log(policy.maximum_axis_ratio)
            logs = jr.uniform(length_key, (draws, 3), dtype=jnp.float64, maxval=spread)
            angles = jr.uniform(
                angle_key,
                (draws, 3),
                dtype=jnp.float64,
                minval=policy.angle_range[0],
                maxval=policy.angle_range[1],
            )

            def candidate(log_lengths: Array, angle: Array) -> tuple[Array, Array]:
                metric = metric_from_cell_parameters(jnp.exp(log_lengths), angle)
                coefficients = chart.coefficients(metric)
                current = self._volume(coefficients)
                target = policy.oversample * volume
                scaled = coefficients * jnp.where(
                    current > 0.0, (target / current) ** (2.0 / 3.0), 1.0
                )
                cell = chart.cell(scaled)
                _, heights = cell_heights(cell.vectors)
                ok = self._cell_admitted(cell.vectors, cell.successful) & jnp.all(
                    heights >= policy.minimum_height
                )
                return scaled, ok

            coefficients, ok = jax.vmap(candidate)(logs, angles)
            first = jnp.argmax(ok)
            return coefficients[first], jnp.any(ok)

        return jax.vmap(lane)(keys, reference_volume)

    def initialize_method(
        self,
        keys: Array,
        ordinals: Array,
        conditioning: AtomisticConditioning | None,
        selected: Array,
        /,
    ) -> CrystalReliefMethod:
        # Every resident lane initializes its relief state (the contact bundle
        # is prepared for the full lane count); `admit` keeps admitted rows.
        del ordinals, conditioning, selected
        lanes = keys.shape[0]
        sites = self.layout.site_count
        counts = jnp.asarray(self.layout.site_conformer_count, dtype=jnp.int32)
        site_logits = self.site_logits
        conformer_keys = role_keys(keys, "conformer")
        conformers = jax.vmap(
            lambda key: jax.vmap(jr.categorical)(jr.split(key, sites), site_logits)
        )(conformer_keys).astype(jnp.int32)
        conformers = jnp.minimum(conformers, counts - 1)
        centers = jax.vmap(lambda key: jr.uniform(key, (sites, 3), dtype=jnp.float64))(
            role_keys(keys, "center")
        )
        orientations = jax.vmap(lambda key: _uniform_quaternions(jr.split(key, sites)))(
            role_keys(keys, "orientation")
        )
        volumes = jax.vmap(
            lambda key: jr.uniform(
                key,
                (),
                dtype=jnp.float64,
                minval=self.volume_low,
                maxval=self.volume_high,
            )
        )(role_keys(keys, "volume"))
        coefficients, cell_ok = self._initial_cells(role_keys(keys, "cell"), volumes)
        _, fractional, vectors, _, certified = self._expand(
            centers, orientations, coefficients, conformers
        )
        geometry, _ = self._geometry(fractional, vectors)
        resources = self.bundle.initialize(geometry)
        status = jnp.where(
            cell_ok & certified, 0, int(AtomisticProposalStatus.CELL_INVALID)
        ).astype(jnp.int32)
        nan = jnp.full((lanes,), jnp.nan, dtype=jnp.float64)
        return CrystalReliefMethod(
            centers=centers,
            orientations=orientations,
            coefficients=coefficients,
            conformers=conformers,
            reference_volume=volumes,
            steps=jnp.zeros((lanes,), dtype=jnp.int32),
            status=status,
            accepted=jnp.zeros((lanes,), dtype=jnp.bool_),
            maximum_overlap=nan,
            overlap_energy=nan,
            cell_refusals=jnp.zeros((lanes,), dtype=jnp.int32),
            resources=resources,
        )

    def _energy_terms(
        self,
        centers: Array,
        rotation: Array,
        coefficients: Array,
        orientations: Array,
        conformers: Array,
        counts: Array,
        resources: AtomisticNeighborhoodBundleState,
        running: Array | None = None,
    ) -> tuple[Array, tuple[Array, Array]]:
        turned = jax.vmap(jax.vmap(_ROTATIONS.retract))(orientations, rotation)
        positions, _, vectors, _, _ = self._expand(
            centers, turned, coefficients, conformers
        )
        energy, maximum, _ = self._overlaps(
            positions, vectors, counts, resources, running
        )
        return jnp.sum(energy), (energy, maximum)

    def _step(
        self, method: CrystalReliefMethod, running: Array, /
    ) -> CrystalReliefMethod:
        relief = self.relief
        live = running & (method.status == 0) & ~method.accepted
        original = method
        reference = self.bundle.cell
        if reference is None:
            raise RuntimeError("Crystal relief requires its periodic reference cell.")
        reference_coefficients = self.layout.chart.coefficients(
            reference.vectors @ reference.vectors.T
        )
        # Vacant and terminal rows are padding, not singular cells to differentiate.
        method = eqx.tree_at(
            lambda value: value.coefficients,
            method,
            jnp.where(live[:, None], method.coefficients, reference_coefficients),
        )
        _, fractional, vectors, offsets, certified = self._expand(
            method.centers, method.orientations, method.coefficients, method.conformers
        )
        geometry, counts = self._geometry(fractional, vectors)
        candidate = self.bundle.refresh(method.resources, geometry, allow_rebuild=live)
        zero_rotation = jnp.zeros_like(method.centers)
        (_, (energy, maximum)), gradients = jax.value_and_grad(
            self._energy_terms, argnums=(0, 1, 2), has_aux=True
        )(
            method.centers,
            zero_rotation,
            method.coefficients,
            method.orientations,
            method.conformers,
            counts,
            candidate,
            live,
        )
        center_gradient, rotation_gradient, cell_gradient = gradients
        volume = jax.vmap(self._volume)(method.coefficients)
        capacity = candidate.capacity_failure
        scientific = candidate.scientific_failure | (~candidate.successful & ~capacity)
        finite = (
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(center_gradient), axis=(-2, -1))
            & jnp.all(jnp.isfinite(rotation_gradient), axis=(-2, -1))
            & jnp.all(jnp.isfinite(cell_gradient), axis=-1)
        )
        check = (method.steps % relief.check_interval) == 0
        accept = (
            check
            & (maximum <= relief.overlap_tolerance)
            & (
                jnp.abs(volume - method.reference_volume)
                <= relief.volume_tolerance * method.reference_volume
            )
            & candidate.successful
            & certified
            & finite
        )
        failure = jnp.where(
            capacity, int(AtomisticProposalStatus.CAPACITY_REFUSED), 0
        ) | jnp.where(
            scientific | ~certified | ~finite,
            int(AtomisticProposalStatus.SAMPLER_FAILURE),
            0,
        )
        moving = live & ~accept & (failure == 0)
        center_gradient = jnp.where(moving[:, None, None], center_gradient, 0.0)
        rotation_gradient = jnp.where(moving[:, None, None], rotation_gradient, 0.0)
        cell_gradient = jnp.where(moving[:, None], cell_gradient, 0.0)
        # Rigid step: mass-preconditioned Cartesian force and principal-inertia
        # preconditioned torque, capped by the largest induced displacement.
        transpose = jnp.swapaxes(vectors, -1, -2)
        sites = method.centers.shape[1]
        cell_rows = jnp.broadcast_to(vectors[:, None], (vectors.shape[0], sites, 3, 3))
        force = -solve_small_linear(_SMALL3, cell_rows, center_gradient[..., None]).value[
            ..., 0
        ]
        translation = relief.step_scale * force / self.site_mass[None, :, None]
        rotation = quaternion_rotation_matrix(method.orientations)
        rows = (
            jnp.asarray(self.layout.site_conformer_offset, dtype=jnp.int32)[None]
            + method.conformers
        )
        moments = self.conformer_moments[rows]
        inverse = jnp.where(
            moments > 1.0e-12 * jnp.max(moments, axis=-1, keepdims=True),
            1.0 / moments,
            0.0,
        )
        body_torque = contract("lsji,lsj->lsi", rotation, -rotation_gradient)
        angular = relief.step_scale * contract(
            "lsij,lsj->lsi", rotation, inverse * body_torque
        )
        induced = translation[:, :, None, :] + jnp.cross(angular[:, :, None, :], offsets)
        induced = jnp.where(self.site_atom_mask[None, :, :, None], induced, 0.0)
        largest = jnp.max(jnp.linalg.norm(induced, axis=-1), axis=(-2, -1))
        cap = jnp.minimum(relief.maximum_step, maximum)
        scale = jnp.where(largest > cap, cap / jnp.maximum(largest, 1.0e-300), 1.0)
        translation = translation * scale[:, None, None]
        angular = angular * scale[:, None, None]
        # Molecules of small extent are not displacement-limited; their rotation
        # step is capped explicitly, far inside the retraction cut locus.
        turn = jnp.linalg.norm(angular, axis=-1, keepdims=True)
        angular = angular * jnp.where(
            turn > relief.maximum_rotation,
            relief.maximum_rotation / jnp.maximum(turn, 1.0e-300),
            1.0,
        )
        moved = solve_small_linear(
            _SMALL3,
            jnp.broadcast_to(transpose[:, None], cell_rows.shape),
            translation[..., None],
        ).value[..., 0]
        centers = method.centers + moved
        centers = centers - jax.lax.stop_gradient(jnp.floor(centers))
        orientations = jax.vmap(jax.vmap(_ROTATIONS.retract))(
            method.orientations, angular
        )
        # Cell step: deviatoric chart descent with a metric-strain cap, retracted
        # onto the pre-step volume, then compression of the excess volume above
        # the reference; never expansion.
        basis = self.layout.chart.basis_metrics(jnp.float64)
        volume_gradient = jax.vmap(jax.grad(self._volume))(method.coefficients)
        step = -relief.cell_step_scale * cell_gradient
        along = jnp.sum(step * volume_gradient, axis=-1) / jnp.maximum(
            jnp.sum(volume_gradient * volume_gradient, axis=-1), 1.0e-300
        )
        step = step - along[:, None] * volume_gradient
        metric = contract("lk,kij->lij", method.coefficients, basis)
        change = contract("lk,kij->lij", step, basis)
        relative = jnp.linalg.norm(change, axis=(-2, -1)) / jnp.linalg.norm(
            metric, axis=(-2, -1)
        )
        limit = 2.0 * relief.maximum_cell_strain
        step = (
            step
            * jnp.where(relative > limit, limit / jnp.maximum(relative, 1.0e-300), 1.0)[
                :, None
            ]
        )
        sheared = method.coefficients + step
        sheared_volume = jax.vmap(self._volume)(sheared)
        # The tangent projection preserves volume only to first order; the
        # metric is linear in the chart coordinates and V = sqrt(det G), so the
        # uniform rescaling by (V_target / V_sheared)^(2/3) retracts the finite
        # deviatoric step to the pre-step volume and applies the compression in
        # one exact factor.  A non-positive or non-finite sheared volume is a
        # non-SPD metric; it stays unscaled so the Cholesky certificate refuses
        # and counts it.
        excess = jnp.maximum(volume - method.reference_volume, 0.0)
        target = volume - relief.volume_compression * excess
        compressed = (
            sheared
            * jnp.where(
                sheared_volume > 0.0, (target / sheared_volume) ** (2.0 / 3.0), 1.0
            )[:, None]
        )
        new_cells = jax.vmap(self.layout.chart.cell)(compressed)
        admitted = self._cell_admitted(new_cells.vectors, new_cells.successful)
        coefficients = jnp.where(admitted[:, None], compressed, method.coefficients)
        frozen = accept | (failure != 0)
        exhausted = ~frozen & (method.steps + 1 >= relief.maximum_steps)
        updated = CrystalReliefMethod(
            centers=centers,
            orientations=orientations,
            coefficients=coefficients,
            conformers=method.conformers,
            reference_volume=method.reference_volume,
            steps=method.steps + 1,
            status=jnp.where(
                exhausted, int(AtomisticProposalStatus.STEP_BUDGET_EXHAUSTED), 0
            ).astype(jnp.int32),
            accepted=jnp.zeros_like(accept),
            maximum_overlap=maximum,
            overlap_energy=energy,
            cell_refusals=method.cell_refusals + (~admitted).astype(jnp.int32),
            resources=candidate,
        )
        held = CrystalReliefMethod(
            centers=method.centers,
            orientations=method.orientations,
            coefficients=method.coefficients,
            conformers=method.conformers,
            reference_volume=method.reference_volume,
            steps=method.steps,
            status=failure.astype(jnp.int32),
            accepted=accept,
            maximum_overlap=maximum,
            overlap_energy=energy,
            cell_refusals=method.cell_refusals,
            resources=candidate,
        )
        result = self.select_method(frozen, held, updated)
        return self.select_method(live, result, original)

    def advance_method(
        self, method: CrystalReliefMethod, running: Array, steps: int, /
    ) -> tuple[CrystalReliefMethod, Array]:
        def body(_: int, current: CrystalReliefMethod) -> CrystalReliefMethod:
            live = running & (current.status == 0) & ~current.accepted
            return jax.lax.cond(
                jnp.any(live),
                lambda: self._step(current, live),
                lambda: current,
            )

        advanced = jax.lax.fori_loop(0, steps, body, method)
        done = advanced.accepted | (advanced.status != 0)
        return advanced, done

    def emit_method(
        self, method: CrystalReliefMethod, ordinals: Array, /
    ) -> AtomisticProposalBlock:
        positions, _, vectors, _, _ = self._expand(
            method.centers, method.orientations, method.coefficients, method.conformers
        )
        lanes = positions.shape[0]
        species = jnp.broadcast_to(
            jnp.asarray(self.support.fixed_species, dtype=jnp.int32), positions.shape[:2]
        )
        padded = jnp.zeros((lanes, _MAXIMUM_CHART_DIMENSION), dtype=jnp.float64)
        padded = padded.at[:, : self.layout.chart.dimension].set(method.coefficients)
        volume = jax.vmap(self._volume)(method.coefficients)
        status = jnp.where(
            method.accepted,
            0,
            jnp.where(
                method.status == 0,
                int(AtomisticProposalStatus.SAMPLER_FAILURE),
                method.status,
            ),
        )
        return AtomisticProposalBlock(
            ordinals,
            positions,
            species,
            jnp.ones(positions.shape[:2], dtype=jnp.bool_),
            vectors,
            status,
            (
                method.centers,
                method.orientations,
                padded,
                method.conformers,
                jnp.full((lanes,), self.group_index, dtype=jnp.int32),
                method.steps,
                method.maximum_overlap,
                method.overlap_energy,
                method.reference_volume,
                volume,
                method.cell_refusals,
            ),
            sampler_id=self.sampler_id,
            payload_names=self.payload_spec.names,
        )


# -- prepared packer, results and store records --------------------------------------------------


@final
class PackedCrystal(StrictModule, NonTrainableState):
    """One terminal packing attempt with its ASU state and expansion.

    ``accepted`` attempts passed the overlap and volume checks; others keep
    their status bits (step budget, capacity, cell or sampler failure).
    """

    proposal_id: str = eqx.field(static=True)
    ordinal: int = eqx.field(static=True)
    group_index: int = eqx.field(static=True)
    status: int = eqx.field(static=True)
    plan: AsymmetricUnitPlan
    state: AsymmetricUnitState
    expansion: ExpandedCrystal
    steps: int = eqx.field(static=True)
    maximum_overlap: float = eqx.field(static=True)
    overlap_energy: float = eqx.field(static=True)
    reference_volume: float = eqx.field(static=True)
    volume: float = eqx.field(static=True)
    cell_refusals: int = eqx.field(static=True)
    z_prime: int = eqx.field(static=True)

    @property
    def accepted(self) -> bool:
        return self.status == 0

    @property
    def z(self) -> int:
        """Formula units per cell, ``|G| Z'``."""
        return self.plan.operation_count * self.z_prime


@final
class CrystalPackingResult(StrictModule, NonTrainableState):
    """Ordinal-ordered packing outcome with explicit shortfall."""

    attempts: tuple[PackedCrystal, ...]
    target: int = eqx.field(static=True)
    attempt_budget: int = eqx.field(static=True)
    stop_reason: str = eqx.field(static=True)
    shortfall: int = eqx.field(static=True)

    @property
    def accepted(self) -> tuple[PackedCrystal, ...]:
        return tuple(item for item in self.attempts if item.accepted)


@final
class PreparedCrystalPacker(StrictModule, NonTrainableState):
    """One stateful group packer per admitted setting and the ordinal group route.

    ``units`` are the host-owned ASU plans of the settings (with their
    molecular witnesses); each compiled member holds only its layout.
    """

    plan: CrystalPackingPlan
    units: tuple[AsymmetricUnitPlan, ...]
    members: tuple[PreparedCrystalGroupPacker, ...]
    lanes: int = eqx.field(static=True)

    def __init__(self, plan: CrystalPackingPlan, /, *, lanes: int) -> None:
        units = tuple(plan.asymmetric_unit(setting) for setting in plan.groups.settings)
        self.plan = plan
        self.lanes = lanes
        self.units = units
        self.members = tuple(
            PreparedCrystalGroupPacker(plan, unit, group_index=index, lanes=lanes)
            for index, unit in enumerate(units)
        )

    @property
    def family(self) -> AtomisticSamplerFamily:
        groups = self.plan.groups
        route = (
            AtomisticFixedRoute()
            if len(self.members) == 1
            else AtomisticWeightedRoute(groups.weights, provenance=groups.provenance)
        )
        return AtomisticSamplerFamily(self.members, route=route)

    def stream(
        self,
        campaign_id: str,
        /,
        *,
        pending_capacity: int | None = None,
        steps_per_transition: int | None = None,
        filters: Sequence[AbstractProposalFilter] = (),
    ) -> AtomisticGenerationStreamPlan:
        """Generation stream of attempts (lanes per member = the prepared lanes)."""
        return AtomisticGenerationStreamPlan(
            self.family,
            campaign_id=campaign_id,
            lanes_per_member=self.lanes,
            pending_capacity=(
                self.lanes * len(self.members)
                if pending_capacity is None
                else pending_capacity
            ),
            steps_per_transition=(
                self.plan.relief.check_interval
                if steps_per_transition is None
                else steps_per_transition
            ),
            filters=filters,
        )

    def packed(self, proposal: AtomisticGeneratedProposal, /) -> PackedCrystal:
        """Re-expand a published proposal from its ASU payload (host)."""
        group = int(proposal.value("group_index"))
        member, unit = self.members[group], self.units[group]
        if member.sampler_id != proposal.sampler_id:
            raise ValueError("The proposal belongs to another packer.")
        dimension = unit.chart.dimension
        state = AsymmetricUnitState(
            jnp.asarray(proposal.value("asu_centers")),
            jnp.asarray(proposal.value("asu_orientations")),
            jnp.asarray(proposal.value("cell_coefficients")[:dimension]),
            jnp.asarray(proposal.value("conformer_indices")),
        )
        expansion = expand_asymmetric_unit(unit, state)
        return PackedCrystal(
            proposal_id=proposal.proposal_id,
            ordinal=proposal.ordinal,
            group_index=member.group_index,
            status=proposal.status,
            plan=unit,
            state=state,
            expansion=expansion,
            steps=int(proposal.value("steps")),
            maximum_overlap=float(proposal.value("maximum_overlap")),
            overlap_energy=float(proposal.value("overlap_energy")),
            reference_volume=float(proposal.value("reference_volume")),
            volume=float(proposal.value("volume")),
            cell_refusals=int(proposal.value("cell_refusals")),
            z_prime=self.plan.z_prime,
        )

    def record_builder(
        self,
        schema: AtomisticStructureStoreSchema,
        proposal: AtomisticGeneratedProposal,
        /,
    ) -> AtomisticStructureRecord:
        """Structure-store record of a published attempt (sink record builder)."""
        packed = self.packed(proposal)
        return crystal_structure_record(
            schema,
            record_id=proposal.proposal_id,
            lineage_id=self.plan.law_id,
            plan=packed.plan,
            positions=np.asarray(jax.device_get(packed.expansion.positions)),
            cell=np.asarray(jax.device_get(packed.expansion.cell.vectors)),
            stage="packed",
            status=packed.status,
            ordinal=packed.ordinal,
            z_prime=packed.z_prime,
            state=packed.state,
            copy_shifts=np.asarray(jax.device_get(packed.expansion.copy_center_shifts)),
            evidence=(
                packed.steps,
                packed.maximum_overlap,
                packed.overlap_energy,
                packed.reference_volume,
            ),
        )

    def pack(
        self,
        root_key: PRNGKey,
        /,
        *,
        campaign_id: str,
        target: int,
        attempt_budget: int,
        sink: AbstractAtomisticGenerationSink | None = None,
        maximum_transitions: int | None = None,
        state: AtomisticGenerationState | None = None,
    ) -> tuple[CrystalPackingResult, AtomisticGenerationState]:
        """Host driver: a bounded stream run returning ordinal-ordered attempts.

        Zero target or budget returns an empty result without drawing.  With
        ``state`` the run continues an earlier continuation exactly (after
        reconciling attempts the sink already holds); the result lists the
        attempts published by this call.  Without ``sink`` the explicit output
        policy is an in-memory sink.  A run that exhausts
        ``maximum_transitions`` before finishing returns a ``RUNNING``
        continuation, never a truncated success.
        """
        runtime: PreparedAtomisticGenerationStream = self.stream(campaign_id).prepare()
        output = (
            AtomisticMemoryGenerationSink("memory:" + runtime.stream_id)
            if sink is None
            else sink
        )
        if state is None:
            current = runtime.start(
                root_key, target=target, attempt_budget=attempt_budget
            )
        else:
            current = runtime.reconcile(state, output)
        relief = self.plan.relief
        limit = (
            (attempt_budget // self.lanes + 2)
            * (relief.maximum_steps // relief.check_interval + 2)
            if maximum_transitions is None
            else maximum_transitions
        )
        published: list[AtomisticGeneratedProposal] = []
        for _ in range(limit + 1):
            ready = runtime.ready(current)
            current, receipt = runtime.publish(current, output)
            if receipt is not None:
                published.extend(ready[: receipt.count])
            if runtime.finished(current):
                break
            current = runtime.transition(current)
        return (
            CrystalPackingResult(
                attempts=tuple(self.packed(item) for item in published),
                target=int(current.cursor.target),
                attempt_budget=int(current.cursor.budget)
                - int(current.cursor.first_ordinal),
                stop_reason=current.stop_reason.name,
                shortfall=runtime.shortfall(current),
            ),
            current,
        )


# This is one ordered persistence declaration, not an execution algorithm.
# Keep field units, roles and expansion provenance adjacent and canonically ordered.
def crystal_structure_schema() -> AtomisticStructureStoreSchema:
    """Typed store schema of packed and relaxed molecular crystals.

    Structure rows carry the cell, stage, status, setting identity, ``Z``,
    ``Z'``, padded metric-chart coordinates, ordinal, relief evidence and an
    optional energy label; ``asu_site`` rows the ASU centers, orientations and
    conformer choices (none for an unconstrained relaxed structure); atom rows
    the unwrapped whole-molecule positions, species and complete expansion
    provenance (ASU site, local atom, component, copy, operation, whether the
    copy is the mirror stereoisomer ``component.enantiomer()`` of its source
    component, and the copy center's lattice shift).
    """
    return AtomisticStructureStoreSchema(
        (
            AtomisticStoreLevel("structure", "structure"),
            AtomisticStoreLevel("atom", "atom", parent="structure"),
            AtomisticStoreLevel("asu_site", "group", parent="structure"),
        ),
        (
            AtomisticStoreField(
                "cell",
                "structure",
                "float64",
                kind="continuous",
                role="geometry",
                convention="row-lattice-vectors",
                axes=(_CELL, _CELL),
                unit=ANGSTROM,
                extensivity="intensive",
            ),
            AtomisticStoreField(
                "stage",
                "structure",
                "int8",
                kind="categorical",
                role="identity",
                convention="crystal-record-stage:0-packed-proposal:1-relaxed",
            ),
            AtomisticStoreField(
                "status",
                "structure",
                "int32",
                kind="categorical",
                role="label",
                convention="atomistic-proposal-status-bits",
            ),
            AtomisticStoreField(
                "international_number",
                "structure",
                "int16",
                kind="categorical",
                role="identity",
                convention="ita-space-group-number",
            ),
            AtomisticStoreField(
                "hall_number",
                "structure",
                "int16",
                kind="categorical",
                role="identity",
                convention="hall-setting-number",
            ),
            AtomisticStoreField(
                "z",
                "structure",
                "int32",
                kind="integer",
                role="identity",
                convention="formula-units-per-cell",
            ),
            AtomisticStoreField(
                "z_prime",
                "structure",
                "int32",
                kind="integer",
                role="identity",
                convention="formula-units-per-asymmetric-unit",
            ),
            AtomisticStoreField(
                "cell_coefficients",
                "structure",
                "float64",
                kind="continuous",
                role="geometry",
                convention="invariant-metric-chart-zero-padded",
                axes=(_CHART,),
                unit=_SQUARED_LENGTH,
                extensivity="intensive",
            ),
            AtomisticStoreField(
                "ordinal",
                "structure",
                "int64",
                kind="integer",
                role="identity",
                convention="generation-attempt-ordinal",
            ),
            AtomisticStoreField(
                "relief_steps",
                "structure",
                "int32",
                kind="integer",
                role="derived",
                convention="overlap-relief-steps",
                availability="optional",
            ),
            AtomisticStoreField(
                "maximum_overlap",
                "structure",
                "float64",
                kind="continuous",
                role="derived",
                convention="largest-contact-overlap",
                unit=ANGSTROM,
                extensivity="intensive",
                availability="optional",
            ),
            AtomisticStoreField(
                "overlap_penalty",
                "structure",
                "float64",
                kind="continuous",
                role="derived",
                convention="half-image-pair-squared-overlap-sum",
                unit=_SQUARED_LENGTH,
                extensivity="extensive",
                availability="optional",
            ),
            AtomisticStoreField(
                "reference_volume",
                "structure",
                "float64",
                kind="continuous",
                role="derived",
                convention="packing-reference-volume",
                unit=_CUBED_LENGTH,
                extensivity="extensive",
                availability="optional",
            ),
            AtomisticStoreField(
                "energy",
                "structure",
                "float64",
                kind="continuous",
                role="label",
                convention="native-program-total-energy",
                unit=ELECTRONVOLT,
                extensivity="extensive",
                availability="optional",
            ),
            AtomisticStoreField(
                "center",
                "asu_site",
                "float64",
                kind="continuous",
                role="geometry",
                convention="fractional-asu-center",
                axes=(_CELL,),
                unit=ONE,
                extensivity="intensive",
            ),
            AtomisticStoreField(
                "orientation",
                "asu_site",
                "float64",
                kind="continuous",
                role="geometry",
                convention="scalar-first-unit-quaternion-spatial",
                axes=(_QUATERNION,),
                unit=ONE,
                extensivity="intensive",
            ),
            AtomisticStoreField(
                "conformer",
                "asu_site",
                "int32",
                kind="categorical",
                role="identity",
                convention="site-conformer-pool-index",
            ),
            AtomisticStoreField(
                "positions",
                "atom",
                "float64",
                kind="continuous",
                role="geometry",
                convention="cartesian-whole-molecule-branch",
                axes=(_CELL,),
                unit=ANGSTROM,
                extensivity="intensive",
            ),
            AtomisticStoreField(
                "species",
                "atom",
                "int16",
                kind="categorical",
                role="identity",
                convention="atomic-number",
            ),
            AtomisticStoreField(
                "source_asu_site",
                "atom",
                "int32",
                kind="categorical",
                role="identity",
                convention="expansion-source-asu-site",
            ),
            AtomisticStoreField(
                "source_atom",
                "atom",
                "int32",
                kind="categorical",
                role="identity",
                convention="expansion-source-component-atom",
            ),
            AtomisticStoreField(
                "source_component",
                "atom",
                "int32",
                kind="categorical",
                role="identity",
                convention="expansion-source-component",
            ),
            AtomisticStoreField(
                "source_copy",
                "atom",
                "int32",
                kind="categorical",
                role="identity",
                convention="expansion-molecule-copy",
            ),
            AtomisticStoreField(
                "source_operation",
                "atom",
                "int32",
                kind="categorical",
                role="identity",
                convention="expansion-setting-operation-index",
            ),
            AtomisticStoreField(
                "source_enantiomer",
                "atom",
                "int8",
                kind="categorical",
                role="identity",
                convention="expansion-copy-stereoisomer:0-source-component:1-enantiomer",
            ),
            AtomisticStoreField(
                "copy_shift",
                "atom",
                "int32",
                kind="integer",
                role="geometry",
                convention="copy-center-lattice-wrap-shift",
                axes=(_CELL,),
            ),
        ),
    )


def crystal_structure_record(
    schema: AtomisticStructureStoreSchema,
    /,
    *,
    record_id: str,
    lineage_id: str,
    plan: AsymmetricUnitPlan,
    positions: np.ndarray,
    cell: np.ndarray,
    stage: CrystalRecordStage,
    status: int,
    ordinal: int,
    z_prime: int,
    state: AsymmetricUnitState | None = None,
    copy_shifts: np.ndarray | None = None,
    evidence: tuple[int, float, float, float] | None = None,
    energy: float | None = None,
) -> AtomisticStructureRecord:
    """One typed packed or relaxed crystal record with complete expansion provenance.

    ``lineage_id`` names the source: the packing law (`CrystalPackingPlan.law_id`)
    for packed attempts, the packed record identity for relaxed structures.  A
    relaxed structure without a symmetry-preserving chart keeps its source
    setting only as provenance and stores no ASU rows.
    """
    stage_ = parse(stage, CrystalRecordStage, "stage")
    count = plan.atom_count
    position = np.asarray(positions, dtype=np.float64)
    vectors = np.asarray(cell, dtype=np.float64)
    if position.shape != (count, 3) or vectors.shape != (3, 3):
        raise ValueError("positions and cell must follow the ASU plan's expanded atoms.")
    species = np.asarray(
        [
            plan.sites[int(site)].pool.component.atomic_numbers[int(local)]
            for site, local in zip(plan.atom_site, plan.atom_local, strict=True)
        ],
        dtype=np.int16,
    )
    shifts = (
        np.zeros((count, 3), dtype=np.int32)
        if copy_shifts is None
        else np.asarray(copy_shifts, dtype=np.int32)[np.asarray(plan.atom_copy)]
    )
    sites = 0 if state is None else plan.site_count
    coefficients = np.zeros((1, _MAXIMUM_CHART_DIMENSION), dtype=np.float64)
    values: dict[str, np.ndarray] = {
        "cell": vectors[None],
        "stage": np.asarray([_STAGE_CODES[stage_]], dtype=np.int8),
        "status": np.asarray([int(status)], dtype=np.int32),
        "international_number": np.asarray(
            [plan.setting.international_number], dtype=np.int16
        ),
        "hall_number": np.asarray([plan.setting.reference_hall_number], dtype=np.int16),
        "z": np.asarray([plan.operation_count * int(z_prime)], dtype=np.int32),
        "z_prime": np.asarray([int(z_prime)], dtype=np.int32),
        "ordinal": np.asarray([int(ordinal)], dtype=np.int64),
        "positions": position,
        "species": species,
        "source_asu_site": np.asarray(plan.atom_site, dtype=np.int32),
        "source_atom": np.asarray(plan.atom_local, dtype=np.int32),
        "source_component": np.asarray(plan.atom_component, dtype=np.int32),
        "source_copy": np.asarray(plan.atom_copy, dtype=np.int32),
        "source_operation": np.asarray(plan.atom_operation, dtype=np.int32),
        "source_enantiomer": np.asarray(
            plan.copy_enantiomer[np.asarray(plan.atom_copy)], dtype=np.int8
        ),
        "copy_shift": shifts,
    }
    if state is not None:
        dimension = plan.chart.dimension
        coefficients[0, :dimension] = np.asarray(jax.device_get(state.cell_coefficients))
        values["center"] = np.asarray(jax.device_get(state.centers), dtype=np.float64)
        values["orientation"] = np.asarray(
            jax.device_get(state.orientations), dtype=np.float64
        )
        values["conformer"] = np.asarray(
            jax.device_get(state.conformer_indices), dtype=np.int32
        )
    else:
        values["center"] = np.zeros((0, 3), dtype=np.float64)
        values["orientation"] = np.zeros((0, 4), dtype=np.float64)
        values["conformer"] = np.zeros((0,), dtype=np.int32)
    values["cell_coefficients"] = coefficients
    if evidence is not None:
        steps, overlap, penalty, reference = evidence
        values["relief_steps"] = np.asarray([int(steps)], dtype=np.int32)
        values["maximum_overlap"] = np.asarray([float(overlap)], dtype=np.float64)
        values["overlap_penalty"] = np.asarray([float(penalty)], dtype=np.float64)
        values["reference_volume"] = np.asarray([float(reference)], dtype=np.float64)
    if energy is not None:
        values["energy"] = np.asarray([float(energy)], dtype=np.float64)
    return AtomisticStructureRecord(
        schema,
        record_id=record_id,
        lineage_id=lineage_id,
        counts={"atom": [count], "asu_site": [sites]},
        stable_ids=np.arange(count, dtype=np.int64),
        values=values,
    )


__all__ = [
    "CONTACT_OWNER",
    "CrystalCellPolicy",
    "CrystalContactResources",
    "CrystalPackingPlan",
    "CrystalPackingResult",
    "CrystalRecordStage",
    "CrystalReliefMethod",
    "CrystalReliefPolicy",
    "CrystalSpaceGroupPolicy",
    "PackedCrystal",
    "PreparedCrystalGroupPacker",
    "PreparedCrystalPacker",
    "crystal_structure_record",
    "crystal_structure_schema",
    "native_conformer_pools",
]
