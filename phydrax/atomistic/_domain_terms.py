#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owner-local classical and long-range terms of a distributed atomistic domain.

A domain term binds one canonical prepared single-device term to the domain:
its neighborhood requirement executes on the domain role topology of the same
requirement (``term_key`` = the term name), its fixed per-atom inputs travel as
owner-blocked continuation payload, field-dependent inputs (predicted charges)
come from the layered model's `owner_fields`, and its arithmetic is the
canonical term's own pair kernel (`lennard_jones_pair_interactions`,
`ewald_real_space_kernel`, `ewald_exception_kernel`,
`lennard_jones_pme_pair_energy`), its own coordination definition
(`PreparedFourierD3Potential.row_coordination_numbers`) or the reciprocal
distribution of its canonical definition (`owner_reciprocal_energy`: PME,
Ewald, LJ-PME and FourierD3 reciprocal plus self energies).

Domain topologies hold every directed image route of owned receivers (both
orientations of each pair, every repeated and nonzero self image). A pair
term counts every unordered pair exactly once from its two directed routes
(weight ``1/2`` each, summed by the two receivers' owners).
``MINIMUM_IMAGE_HALF`` roles certify at every evaluation that no pair has two
images, and no atom a self image, inside the cutoff (image ambiguity is a
scientific failure).

Listed exception and scaled pairs act, as in the canonical terms, on their
CURRENT minimum image independently of any cutoff. They use a dedicated
exception support topology with its own capacities: the replicated canonical
pair table is split statically across owners, its endpoints are resolved
through the bounded stable-ID directory, both directed routes of every pair
are sent to their receivers' owners, and the partner rows are gathered through
a dedicated `DistributedHaloPlan`. Directory, route and halo overflow are
capacity evidence; an exception naming an absent atom is a scientific failure.
"""

from __future__ import annotations

import abc
import math
from typing import final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import PartitionSpec
from jax.tree_util import PyTreeDef

from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..backends.distributed import JaxCollectiveProvider
from ..discretization.particle._pair_state import PARTICLE_PAIR_INTERACTION
from ..discretization.spatial._distributed_relations import (
    _bucket_ranks,
    _pack,
    DistributedHaloPlan,
    DistributedOwnershipPlan,
    DistributedPointLayout,
)
from ..ein import contract
from ._classical import lennard_jones_pair_interactions, PreparedLennardJonesPotential
from ._conditioning import AtomisticObservable
from ._dispersion_fourier_d3 import PreparedFourierD3Potential
from ._domain_state import owner_directory_lookup
from ._electrostatics import (
    _minimum_image,
    ewald_exception_kernel,
    ewald_real_space_kernel,
    PreparedEwaldReferencePotential,
    PreparedParticleMeshEwaldPotential,
)
from ._force_field import (
    ForceFieldTermKind,
    lennard_jones_pme_pair_energy,
    PreparedGeneralForceFieldTerm,
)
from ._neighborhood_bundle import (
    _identity_site_map,
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticPairCounting,
    AtomisticPairExclusions,
    AtomisticSiteDomain,
)
from ._polarization import PreparedInducedDipolePolarizationPotential
from ._potential_program import AtomisticInteractionScaleState
from ._reciprocal_distribution import (
    AtomisticReciprocalDistributionPlan,
    owner_reciprocal_energy,
    OwnerMultipoleRelation,
    OwnerMultipoleRoutes,
    OwnerPolarizationPlan,
    OwnerPolarizationRelations,
    OwnerPolarizationSites,
    PreparedOwnerPolarization,
    PreparedReciprocalDistribution,
    ReciprocalSitePayload,
    solve_owner_polarization,
)
from ._system import PreparedAtomisticSystem


_HALF_COUNTINGS = (
    AtomisticPairCounting.MINIMUM_IMAGE_HALF,
    AtomisticPairCounting.IMAGE_HALF,
)


def _field(term_key: str, name: str, /) -> str:
    return f"term:{term_key}:{name}"


def _active_rows(system: PreparedAtomisticSystem, /) -> np.ndarray:
    return np.flatnonzero(np.asarray(system.active_mask))


def _capacity(name: str, value: int | None, /) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


@final
class AtomisticDomainExceptionTable(StrictModule, NonTrainableState):
    """Replicated canonical pair list of one term and its route weights.

    ``left_ids``/``right_ids`` are stable particle IDs (``left < right``);
    ``weights`` are the term's per-pair factors (Ewald: electrostatic exception
    scale; polarization: ``direct, polarization, mutual`` scales ``- 1``).
    ``capacity`` bounds directory packets, routes per owner and partner
    columns per owner pair.
    """

    left_ids: Array
    right_ids: Array
    weights: Array
    capacity: int = eqx.field(static=True)

    @property
    def pair_count(self) -> int:
        return self.left_ids.shape[0]


def _exception_table(
    system: PreparedAtomisticSystem,
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    weights: np.ndarray,
    capacity: int | None,
    /,
) -> AtomisticDomainExceptionTable | None:
    """The listed pairs whose atoms are both active (domain rows), or ``None``.

    A pair with an inactive atom contributes nothing in the canonical term
    (its charge, moment and polarizability are masked), and inactive atoms are
    not domain rows, so it is omitted rather than reported missing.
    """
    active = np.asarray(system.plan.particle_ids, dtype=np.int64)[
        np.asarray(system.active_mask, dtype=np.bool_)
    ]
    keep = np.isin(left_ids, active) & np.isin(right_ids, active)
    if not np.any(keep):
        return None
    return AtomisticDomainExceptionTable(
        left_ids=jnp.asarray(np.asarray(left_ids, dtype=np.int64)[keep]),
        right_ids=jnp.asarray(np.asarray(right_ids, dtype=np.int64)[keep]),
        weights=jnp.asarray(np.asarray(weights, dtype=np.float64)[keep]),
        capacity=_capacity("exception_capacity", capacity),
    )


class AbstractAtomisticDomainTerm(StrictModule, NonTrainableState):
    """One canonical prepared term executed owner-locally on a domain.

    ``term_key`` is the term's declared ``name`` slot (static layout);
    ``term_identity`` is the exact canonical domain-term digest carried as
    dynamic `IdentityWords` (``term_id`` decodes it on the host), so revisions
    of the bound prepared term share one executable. ``scientific_identity``
    (host ``scientific_id``), also dynamic and computed once at construction,
    is the layout-independent digest of the term's physics: the canonical
    prepared term (system, parameters, conventions) and the
    ``scientific_id`` of its reciprocal or solver definition. Reciprocal
    placement, FFT layout and exception or route capacities only place that
    computation and are excluded.
    """

    term_key: eqx.AbstractVar[str]
    term_identity: eqx.AbstractVar[IdentityWords]
    scientific_identity: eqx.AbstractVar[IdentityWords]
    system: eqx.AbstractVar[PreparedAtomisticSystem]

    @property
    def term_id(self) -> str:
        """Host decoding of the canonical domain-term digest."""
        return self.term_identity.fingerprint()

    @property
    def scientific_id(self) -> str:
        """Host decoding of the layout-independent scientific term digest."""
        return self.scientific_identity.fingerprint()

    @abc.abstractmethod
    def requirements(self) -> tuple[AtomisticNeighborhoodRequirement, ...]:
        """Neighborhood requirements bound to domain roles of ``term_key``."""

    @property
    @abc.abstractmethod
    def field_names(self) -> tuple[str, ...]:
        """Static payload keys of :meth:`atom_fields`."""

    @abc.abstractmethod
    def atom_fields(self) -> dict[str, np.ndarray]:
        """Per-atom fixed inputs in logical (active system row) order, by key."""

    @property
    def model_fields(self) -> tuple[AtomisticObservable, ...]:
        """Layered-model observables (`owner_fields`) this term consumes."""
        return ()

    @property
    def exceptions(self) -> AtomisticDomainExceptionTable | None:
        """The exception support topology table (``None`` without pairs)."""
        return None


@final
class AtomisticDomainLennardJonesTerm(AbstractAtomisticDomainTerm):
    """Canonical Lennard-Jones on its classical minimum-image domain role."""

    prepared: PreparedLennardJonesPotential
    system: PreparedAtomisticSystem
    term_key: str = eqx.field(static=True)
    term_identity: IdentityWords
    scientific_identity: IdentityWords

    def __init__(self, prepared: PreparedLennardJonesPotential, /) -> None:
        if not isinstance(prepared, PreparedLennardJonesPotential):
            raise TypeError("prepared must be a PreparedLennardJonesPotential.")
        self.prepared = prepared
        self.system = prepared.system
        self.term_key = prepared.name
        self.term_identity = IdentityWords.of(
            {"kind": "atomistic-domain-lennard-jones", "term": prepared.prepared_id}
        )
        self.scientific_identity = IdentityWords.of(
            {"kind": "atomistic-domain-lennard-jones", "term": prepared.prepared_id}
        )

    def requirements(self) -> tuple[AtomisticNeighborhoodRequirement, ...]:
        return tuple(self.prepared.requirements.neighborhoods)

    @property
    def field_names(self) -> tuple[str, ...]:
        return (_field(self.term_key, "atom_type_ids"),)

    def atom_fields(self) -> dict[str, np.ndarray]:
        rows = _active_rows(self.system)
        types = np.asarray(self.system.plan.atom_type_ids)[rows].astype(np.int32)
        return {_field(self.term_key, "atom_type_ids"): types}


@final
class AtomisticDomainEwaldTerm(AbstractAtomisticDomainTerm):
    """Canonical PME or Ewald Coulomb: domain real space plus distributed reciprocal.

    Real-space pairs execute on the term's own domain role (minimum-image pairs
    or the image-pair sum declared by the term); topology exceptions on the
    exception support topology; reciprocal, self, background and boundary
    energies through the term's canonical definition distributed by
    ``reciprocal``. Charges are the system's declared charges, or the layered
    model's predicted ``CHARGES`` when the term declares a charge field.
    ``exception_capacity`` is required when the system lists exceptions.
    """

    prepared: PreparedParticleMeshEwaldPotential | PreparedEwaldReferencePotential
    reciprocal_plan: AtomisticReciprocalDistributionPlan
    system: PreparedAtomisticSystem
    exception_table: AtomisticDomainExceptionTable | None
    term_key: str = eqx.field(static=True)
    term_identity: IdentityWords
    scientific_identity: IdentityWords

    def __init__(
        self,
        prepared: PreparedParticleMeshEwaldPotential | PreparedEwaldReferencePotential,
        reciprocal: AtomisticReciprocalDistributionPlan,
        /,
        *,
        exception_capacity: int | None = None,
    ) -> None:
        if not isinstance(
            prepared,
            (PreparedParticleMeshEwaldPotential, PreparedEwaldReferencePotential),
        ):
            raise TypeError("prepared must be a prepared PME or Ewald Coulomb term.")
        if not isinstance(reciprocal, AtomisticReciprocalDistributionPlan):
            raise TypeError("reciprocal must be an AtomisticReciprocalDistributionPlan.")
        if reciprocal.definition.prepared_id != prepared.prepared_id:
            raise ValueError("The reciprocal distribution distributes another term.")
        field = prepared.plan.charge_field
        if field is not None and field.observable is not AtomisticObservable.CHARGES:
            raise ValueError("A domain Ewald charge field must be predicted CHARGES.")
        system = prepared.system
        keys = np.asarray(system.topology.exception_keys, dtype=np.int64)
        table = None
        if keys.shape[0]:
            table = _exception_table(
                system,
                keys[:, 1],
                keys[:, 2],
                np.asarray(system.topology.electrostatic_scales)[:, None],
                exception_capacity,
            )
        self.prepared = prepared
        self.reciprocal_plan = reciprocal
        self.system = system
        self.exception_table = table
        self.term_key = prepared.name
        self.term_identity = IdentityWords.of(
            {
                "kind": "atomistic-domain-ewald",
                "term": prepared.prepared_id,
                "reciprocal": reciprocal.plan_id,
                "exception_capacity": None if table is None else table.capacity,
            }
        )
        self.scientific_identity = IdentityWords.of(
            {
                "kind": "atomistic-domain-ewald",
                "term": prepared.prepared_id,
                "reciprocal": reciprocal.scientific_id,
            }
        )

    @property
    def reciprocal(self) -> AtomisticReciprocalDistributionPlan:
        return self.reciprocal_plan

    @property
    def exceptions(self) -> AtomisticDomainExceptionTable | None:
        return self.exception_table

    @property
    def model_fields(self) -> tuple[AtomisticObservable, ...]:
        if self.prepared.plan.charge_field is None:
            return ()
        return (AtomisticObservable.CHARGES,)

    def requirements(self) -> tuple[AtomisticNeighborhoodRequirement, ...]:
        return tuple(self.prepared.requirements.neighborhoods)

    @property
    def field_names(self) -> tuple[str, ...]:
        if self.model_fields:
            return ()
        return (_field(self.term_key, "charges"),)

    def atom_fields(self) -> dict[str, np.ndarray]:
        if self.model_fields:
            return {}
        rows = _active_rows(self.system)
        charges = np.asarray(self.system.plan.charges)[rows]
        return {_field(self.term_key, "charges"): charges}


@final
class AtomisticDomainLennardJonesPMETerm(AbstractAtomisticDomainTerm):
    """Canonical LJ-PME dispersion: domain real space plus distributed reciprocal.

    The screened real-space pairs and the exception correction (the scaled
    share of the full dispersion the reciprocal sum contains, inside the
    term's pair relation) execute on the term's interaction-site pair role
    with the canonical `lennard_jones_pme_pair_energy`; the reciprocal and
    self energies are the term's canonical mesh definition distributed by
    ``reciprocal`` (site type rows and the anchor-source frame anchor travel
    as owner-blocked payload). Admitted only when the system's interaction
    sites are a proven identity image of its particles: domain rows are
    atoms, so the site pairs are exactly the atom pairs.
    """

    prepared: PreparedGeneralForceFieldTerm
    reciprocal_plan: AtomisticReciprocalDistributionPlan
    system: PreparedAtomisticSystem
    term_key: str = eqx.field(static=True)
    term_identity: IdentityWords
    scientific_identity: IdentityWords

    def __init__(
        self,
        prepared: PreparedGeneralForceFieldTerm,
        reciprocal: AtomisticReciprocalDistributionPlan,
        /,
    ) -> None:
        if (
            not isinstance(prepared, PreparedGeneralForceFieldTerm)
            or prepared.plan.kind is not ForceFieldTermKind.LENNARD_JONES_PME
        ):
            raise TypeError("prepared must be a prepared Lennard-Jones PME term.")
        if not isinstance(reciprocal, AtomisticReciprocalDistributionPlan):
            raise TypeError("reciprocal must be an AtomisticReciprocalDistributionPlan.")
        if reciprocal.definition.prepared_id != prepared.prepared_id:
            raise ValueError("The reciprocal distribution distributes another term.")
        system = prepared.system
        if not _identity_site_map(system, system.particle_discretization()):
            raise ValueError(
                "Domain rows are physical atoms: LJ-PME interaction sites must be a "
                "proven identity image of the system particles (no virtual sites)."
            )
        self.prepared = prepared
        self.reciprocal_plan = reciprocal
        self.system = system
        self.term_key = prepared.name
        self.term_identity = IdentityWords.of(
            {
                "kind": "atomistic-domain-lennard-jones-pme",
                "term": prepared.prepared_id,
                "reciprocal": reciprocal.plan_id,
            }
        )
        self.scientific_identity = IdentityWords.of(
            {
                "kind": "atomistic-domain-lennard-jones-pme",
                "term": prepared.prepared_id,
                "reciprocal": reciprocal.scientific_id,
            }
        )

    @property
    def reciprocal(self) -> AtomisticReciprocalDistributionPlan:
        return self.reciprocal_plan

    def requirements(self) -> tuple[AtomisticNeighborhoodRequirement, ...]:
        return tuple(self.prepared.requirements.neighborhoods)

    @property
    def field_names(self) -> tuple[str, ...]:
        return (_field(self.term_key, "site_type_ids"), _field(self.term_key, "anchor"))

    def atom_fields(self) -> dict[str, np.ndarray]:
        rows = _active_rows(self.system)
        sites = self.system.coordinate_map.plan.sites
        types = np.asarray(sites.site_type_ids)[rows].astype(np.int32)
        anchor = rows == int(np.asarray(self.prepared.site_anchor))
        return {
            _field(self.term_key, "site_type_ids"): types,
            _field(self.term_key, "anchor"): anchor,
        }


@final
class AtomisticDomainFourierD3Term(AbstractAtomisticDomainTerm):
    """Canonical FourierD3: owner-local modified coordination, distributed mesh.

    Modified coordination numbers execute on the term's own full directed
    coordination role (owned receivers, halo or replicated columns) with the
    canonical `PreparedFourierD3Potential.row_coordination_numbers`; per-atom
    signed rank coefficients follow from `row_coefficients`; the reciprocal
    lattice sum and the coefficient-dependent self energy are the term's
    canonical definition distributed by ``reciprocal``. Coordination, payload
    and position cotangents return through the one owner scalar, so the
    coordination-number forces and virial are kept exactly once.
    """

    prepared: PreparedFourierD3Potential
    reciprocal_plan: AtomisticReciprocalDistributionPlan
    system: PreparedAtomisticSystem
    term_key: str = eqx.field(static=True)
    term_identity: IdentityWords
    scientific_identity: IdentityWords

    def __init__(
        self,
        prepared: PreparedFourierD3Potential,
        reciprocal: AtomisticReciprocalDistributionPlan,
        /,
    ) -> None:
        if not isinstance(prepared, PreparedFourierD3Potential):
            raise TypeError("prepared must be a PreparedFourierD3Potential.")
        if not isinstance(reciprocal, AtomisticReciprocalDistributionPlan):
            raise TypeError("reciprocal must be an AtomisticReciprocalDistributionPlan.")
        if reciprocal.definition.prepared_id != prepared.prepared_id:
            raise ValueError("The reciprocal distribution distributes another term.")
        self.prepared = prepared
        self.reciprocal_plan = reciprocal
        self.system = prepared.system
        self.term_key = prepared.name
        self.term_identity = IdentityWords.of(
            {
                "kind": "atomistic-domain-fourier-d3",
                "term": prepared.prepared_id,
                "reciprocal": reciprocal.plan_id,
            }
        )
        self.scientific_identity = IdentityWords.of(
            {
                "kind": "atomistic-domain-fourier-d3",
                "term": prepared.prepared_id,
                "reciprocal": reciprocal.scientific_id,
            }
        )

    @property
    def reciprocal(self) -> AtomisticReciprocalDistributionPlan:
        return self.reciprocal_plan

    def requirements(self) -> tuple[AtomisticNeighborhoodRequirement, ...]:
        return (self.prepared.plan.coordination_requirement,)

    @property
    def field_names(self) -> tuple[str, ...]:
        return (
            _field(self.term_key, "table_rows"),
            _field(self.term_key, "channel_rows"),
            _field(self.term_key, "active"),
        )

    def atom_fields(self) -> dict[str, np.ndarray]:
        rows = _active_rows(self.system)
        prepared = self.prepared
        return {
            _field(self.term_key, "table_rows"): np.asarray(prepared.table_rows)[
                rows
            ].astype(np.int32),
            _field(self.term_key, "channel_rows"): np.asarray(prepared.channel_rows)[
                rows
            ].astype(np.int32),
            _field(self.term_key, "active"): np.asarray(prepared.active)[rows].astype(
                np.bool_
            ),
        }


@final
class AtomisticDomainPolarizationTerm(AbstractAtomisticDomainTerm):
    """Canonical induced-dipole polarization solved owner-locally.

    The term's explicit site data (permanent moments, polarizabilities,
    damping, Gaussian width classes) travel as owner-blocked payload; its
    real-space operator runs on the term's image-pair-sum domain role and its
    reciprocal bilinear through ``plan`` (`OwnerPolarizationPlan` of the same
    canonical solver). Sparse pair scales act on the current minimum image of
    every listed pair through the exception support topology, as the canonical
    correction topology. The solve starts, like the single-device term, from
    the deterministic predictor guess at every evaluation. No layered model
    predicts moments or polarizabilities (`owner_fields` exposes charges
    only), so field-sourced site data has no domain producer and is refused.
    """

    prepared: PreparedInducedDipolePolarizationPotential
    plan: OwnerPolarizationPlan
    system: PreparedAtomisticSystem
    exception_table: AtomisticDomainExceptionTable | None
    moment_tree: PyTreeDef = eqx.field(static=True)
    term_key: str = eqx.field(static=True)
    term_identity: IdentityWords
    scientific_identity: IdentityWords

    def __init__(
        self,
        prepared: PreparedInducedDipolePolarizationPotential,
        plan: OwnerPolarizationPlan,
        /,
        *,
        exception_capacity: int | None = None,
    ) -> None:
        if not isinstance(prepared, PreparedInducedDipolePolarizationPotential):
            raise TypeError(
                "prepared must be a PreparedInducedDipolePolarizationPotential."
            )
        if not isinstance(plan, OwnerPolarizationPlan):
            raise TypeError("plan must be an OwnerPolarizationPlan.")
        if plan.solver.prepared_id != prepared.solver.prepared_id:
            raise ValueError("The owner polarization plan solves another operator.")
        term = prepared.plan
        if term.moment_fields is not None or term.polarizability_field is not None:
            raise ValueError(
                "No layered model predicts moments or polarizabilities; a domain "
                "polarization term uses its explicit site data."
            )
        if not prepared.requirements.neighborhoods:
            raise ValueError(
                "A domain polarization term needs the neighborhood real-space route."
            )
        operator = prepared.solver.operator
        if not operator.plan.route.periodic:
            raise ValueError(
                "A domain is periodic; its polarization term needs a periodic "
                "(Ewald or particle-mesh) multipole route."
            )
        system = prepared.system
        identifiers = np.asarray(system.plan.particle_ids, dtype=np.int64)
        table = None
        if operator.scaling is not None and operator.scaling.scales.pair_count:
            scales = operator.scaling.scales
            pairs = np.asarray(scales.pairs, dtype=np.int64).reshape((-1, 2))
            values = np.asarray(scales.values, dtype=np.float64).reshape(
                (pairs.shape[0], -1)
            )
            columns = [
                scales.column(name) for name in ("direct", "polarization", "mutual")
            ]
            first = identifiers[pairs[:, 0]]
            second = identifiers[pairs[:, 1]]
            table = _exception_table(
                system,
                np.minimum(first, second),
                np.maximum(first, second),
                values[:, columns] - 1.0,
                exception_capacity,
            )
        self.prepared = prepared
        self.plan = plan
        self.system = system
        self.exception_table = table
        self.moment_tree = jax.tree.structure(
            operator.multipoles.moments(jnp.dtype(jnp.float64))
        )
        self.term_key = prepared.name
        self.term_identity = IdentityWords.of(
            {
                "kind": "atomistic-domain-polarization",
                "term": prepared.prepared_id,
                "plan": plan.plan_id,
                "exception_capacity": None if table is None else table.capacity,
            }
        )
        self.scientific_identity = IdentityWords.of(
            {
                "kind": "atomistic-domain-polarization",
                "term": prepared.prepared_id,
                "plan": plan.scientific_id,
            }
        )

    @property
    def exceptions(self) -> AtomisticDomainExceptionTable | None:
        return self.exception_table

    def requirements(self) -> tuple[AtomisticNeighborhoodRequirement, ...]:
        return tuple(self.prepared.requirements.neighborhoods)

    @property
    def field_names(self) -> tuple[str, ...]:
        names = [
            _field(self.term_key, f"moment-{index}")
            for index in range(self.moment_tree.num_leaves)
        ]
        names += [
            _field(self.term_key, name)
            for name in ("polarizabilities", "damping", "active")
        ]
        if self.prepared.solver.operator.widths is not None:
            names.append(_field(self.term_key, "width_classes"))
        return tuple(names)

    def atom_fields(self) -> dict[str, np.ndarray]:
        rows = _active_rows(self.system)
        operator = self.prepared.solver.operator
        moments = jax.device_get(operator.multipoles.moments(jnp.dtype(jnp.float64)))
        fields = {
            _field(self.term_key, f"moment-{index}"): np.asarray(leaf)[rows]
            for index, leaf in enumerate(jax.tree.leaves(moments))
        }
        fields[_field(self.term_key, "polarizabilities")] = np.asarray(
            operator.multipoles.polarizabilities, dtype=np.float64
        )[rows]
        fields[_field(self.term_key, "damping")] = np.asarray(
            operator.multipoles.damping, dtype=np.float64
        )[rows]
        fields[_field(self.term_key, "active")] = np.asarray(
            operator.active_mask, dtype=np.bool_
        )[rows]
        if operator.widths is not None:
            fields[_field(self.term_key, "width_classes")] = np.asarray(
                operator.widths.site_classes, dtype=np.int32
            )[rows]
        return fields


def require_domain_requirement(requirement: AtomisticNeighborhoodRequirement, /) -> None:
    """Admit a requirement whose routes are physical atoms of the domain rows.

    Interaction-site pairs are admitted as a role: the consuming domain term
    (`AtomisticDomainLennardJonesPMETerm`) proves at construction that its
    system's interaction sites are an identity image of the particles, so the
    site pairs are exactly the atom pairs of the domain rows.
    """
    if requirement.cutoff is None:
        raise ValueError(
            "A periodic domain has no complete all-pairs relation (cutoff=None)."
        )
    if requirement.receiver_group is not None:
        raise ValueError("Domain roles do not admit receiver-row subsets.")
    counting = requirement.counting
    if counting is not AtomisticPairCounting.DIRECTED and counting not in _HALF_COUNTINGS:
        raise ValueError(f"{requirement.label} has no domain counting convention.")
    if requirement.exclusions not in (
        AtomisticPairExclusions.NONE,
        AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS,
    ):
        raise ValueError(f"{requirement.label} declares unsupported pair exclusions.")
    site_pairs = (
        requirement.role is AtomisticNeighborhoodRole.INTERACTION_SITE_PAIRS
        and requirement.entity is AtomisticSiteDomain.INTERACTION_SITES
    )
    if not site_pairs and requirement.entity not in (
        AtomisticSiteDomain.PHYSICAL_ATOMS,
        AtomisticSiteDomain.DOF_ATOMS,
    ):
        raise ValueError(
            f"{requirement.label} addresses interaction sites; domain rows are atoms."
        )


@final
class PreparedAtomisticDomainTerm(StrictModule, NonTrainableState):
    """A domain term bound to its role indices and prepared distributed work.

    ``work`` is the prepared reciprocal distribution of an Ewald, LJ-PME or
    FourierD3 term, the prepared owner polarization of a polarization term, or
    ``None``.
    """

    term: AbstractAtomisticDomainTerm
    work: PreparedReciprocalDistribution | PreparedOwnerPolarization | None
    role_indices: tuple[int, ...] = eqx.field(static=True)


# --------------------------------------------------------------------------
# Exception support topology
# --------------------------------------------------------------------------


@final
class AtomisticDomainExceptionTopology(StrictModule, NonTrainableState):
    """Receiver-owned directed routes of every listed pair of one term.

    Owner-blocked ``(owners * capacity)`` routes: ``receivers`` index owned
    rows, ``route_owners``/``route_slots`` the partner's physical row,
    ``rows`` the table row; ``halo`` gathers partner columns. Evidence:
    ``capacity_successful`` (directory, route and halo capacities) and
    ``missing`` (endpoints absent from the completely registered directory:
    a scientific failure; a registration overflow is capacity only).
    """

    receivers: Array
    route_owners: Array
    route_slots: Array
    rows: Array
    valid: Array
    halo: DistributedHaloPlan
    maximum_load: Array
    capacity_successful: Array
    missing: Array


@eqx.filter_jit
def _exception_routes(
    ownership: DistributedOwnershipPlan,
    table: AtomisticDomainExceptionTable,
    stable_ids: Array,
    active: Array,
) -> tuple[Array, ...]:
    owners = ownership.owner_count
    capacity = table.capacity
    count = table.pair_count
    share = math.ceil(count / owners)
    axis = ownership.axis_name

    def body(ids: Array, alive: Array, left: Array, right: Array) -> tuple[Array, ...]:
        provider = JaxCollectiveProvider(axis)
        me = jax.lax.axis_index(axis).astype(jnp.int32)
        rows = me + owners * jnp.arange(share, dtype=jnp.int32)
        row_valid = rows < count
        safe = jnp.clip(rows, 0, count - 1)
        queries = jnp.concatenate((left[safe], right[safe]))
        asked = jnp.concatenate((row_valid, row_valid))
        owner, slot, found, directory_load, missing = owner_directory_lookup(
            ids, alive, queries, asked, owners=owners, capacity=capacity, axis=axis
        )
        pair_found = found[:share] & found[share:]
        # Both directed routes of a pair live with their receivers' owners.
        receiver_owner = jnp.concatenate((owner[:share], owner[share:]))
        receiver_slot = jnp.concatenate((slot[:share], slot[share:]))
        source_owner = jnp.concatenate((owner[share:], owner[:share]))
        source_slot = jnp.concatenate((slot[share:], slot[:share]))
        route_rows = jnp.concatenate((rows, rows))
        valid = jnp.concatenate((row_valid & pair_found, row_valid & pair_found))
        target = jnp.where(valid, receiver_owner, 0)
        rank, loads = _bucket_ranks(target, valid, owners)
        fits = valid & (rank < capacity)

        def route(values: Array, fill: int | bool) -> Array:
            packet = _pack(values, target, rank, fits, owners, capacity, fill)
            return provider.all_to_all(packet, split_axis=0, concat_axis=0).reshape(
                (owners * capacity,)
            )

        received_valid = route(fits, False)
        received = (
            route(receiver_slot, 0),
            route(source_owner, 0),
            route(source_slot, 0),
            route(route_rows, 0),
        )
        # Compact the received routes into this owner's route capacity.
        total = jnp.sum(received_valid, dtype=jnp.int32)
        order = jnp.nonzero(received_valid, size=capacity, fill_value=0)[0]
        kept = jnp.arange(capacity, dtype=jnp.int32) < total
        compacted = tuple(jnp.where(kept, value[order], 0) for value in received)
        load = provider.maximum(
            jnp.maximum(jnp.maximum(directory_load, jnp.max(loads)), total)
        )
        return (*compacted, kept, load[None], missing[None])

    owner = PartitionSpec(axis)
    replicated = PartitionSpec()
    return ownership.map(body, (owner, owner, replicated, replicated), (owner,) * 7)(
        stable_ids, active, table.left_ids, table.right_ids
    )


def domain_exception_topology(
    layout: DistributedPointLayout, table: AtomisticDomainExceptionTable, /
) -> AtomisticDomainExceptionTopology:
    """Bind one term's listed pairs to the current layout (traceable).

    The routes and partner halo are rebuilt from the layout at every call, so
    they are never stale with respect to migration.
    """
    ownership = layout.plan
    receivers, route_owners, route_slots, rows, valid, load, missing = _exception_routes(
        ownership, table, layout.stable_ids, layout.active
    )
    halo = DistributedHaloPlan(
        ownership, route_owners, route_slots, valid, halo_capacity=table.capacity
    )
    maximum = jnp.max(load)
    return AtomisticDomainExceptionTopology(
        receivers=receivers,
        route_owners=route_owners,
        route_slots=route_slots,
        rows=rows,
        valid=valid & halo.route_valid,
        halo=halo,
        maximum_load=maximum,
        capacity_successful=(maximum <= table.capacity) & halo.evidence.successful,
        missing=jnp.max(missing),
    )


# --------------------------------------------------------------------------
# Owner-local term shares
# --------------------------------------------------------------------------


def replicated_halo(halo: DistributedHaloPlan, /) -> DistributedHaloPlan:
    """``halo`` without its owner-blocked arrays, for a replicated region input.

    The owner geometry and the replicated evidence stay; the owner-blocked
    send and route arrays travel separately as owned inputs.
    """
    return eqx.tree_at(
        lambda plan: (
            plan.send_slots,
            plan.send_valid,
            plan.route_columns,
            plan.route_valid,
        ),
        halo,
        (None, None, None, None),
    )


def region_halo(
    halo: DistributedHaloPlan, send_slots: Array, send_valid: Array, /
) -> DistributedHaloPlan:
    """A replicated halo bound to this owner's send slots inside a region."""
    return eqx.tree_at(
        lambda plan: (plan.send_slots, plan.send_valid),
        halo,
        (send_slots, send_valid),
        is_leaf=lambda leaf: leaf is None,
    )


class OwnerTermRoute(NamedTuple):
    """One owner's routes of one role (inside an owner region)."""

    receivers: Array
    columns: Array
    shifts: Array
    valid: Array
    source_ids: Array


class OwnerExceptionRoute(NamedTuple):
    """One owner's exception routes (inside an owner region)."""

    receivers: Array
    columns: Array
    rows: Array
    valid: Array
    send_slots: Array
    send_valid: Array
    capacity_successful: Array
    missing: Array


class OwnerTermResult(NamedTuple):
    """One owner's share of one term with globally reduced status."""

    energy_share: Array
    capacity_successful: Array
    scientific_successful: Array
    ambiguity: Array


def _global_all(flag: Array, axis_name: str, /) -> Array:
    value = jax.lax.stop_gradient(jnp.all(flag)).astype(jnp.int32)
    return JaxCollectiveProvider(axis_name).minimum(value).astype(jnp.bool_)


def _pair_keys(receiver_ids: Array, source_ids: Array, valid: Array, /) -> Array:
    """Canonical stable unordered-pair keys (``ParticlePairKeySpace.keys``)."""
    left = jnp.minimum(receiver_ids, source_ids).astype(jnp.int64)
    right = jnp.maximum(receiver_ids, source_ids).astype(jnp.int64)
    zeros = jnp.zeros_like(left)
    keys = jnp.stack(
        (
            jnp.full_like(left, PARTICLE_PAIR_INTERACTION),
            left,
            right,
            zeros,
            zeros,
        ),
        axis=-1,
    )
    return jnp.where((valid & (left < right))[:, None], keys, -jnp.ones_like(keys))


def _duplicate_images(
    receivers: Array, source_ids: Array, receiver_ids: Array, active: Array, /
) -> Array:
    """Whether an owned receiver sees one source (or itself) twice inside the cutoff."""
    order = jnp.lexsort(
        (
            jnp.where(active, source_ids, 0),
            jnp.where(active, receivers, 0),
            (~active).astype(jnp.int32),
        )
    )
    sorted_receivers = receivers[order]
    sorted_sources = source_ids[order]
    sorted_active = active[order]
    repeated = (
        sorted_active[1:]
        & sorted_active[:-1]
        & (sorted_receivers[1:] == sorted_receivers[:-1])
        & (sorted_sources[1:] == sorted_sources[:-1])
    )
    self_image = active & (source_ids == receiver_ids)
    return jnp.any(repeated) | jnp.any(self_image)


class _PairGeometry(NamedTuple):
    distance: Array
    active: Array
    receiver_ids: Array


def _pair_geometry(
    positions: Array,
    columns_positions: Array,
    route: OwnerTermRoute,
    cell_vectors: Array,
    owned_ids: Array,
    cutoff: float,
    /,
) -> _PairGeometry:
    dtype = positions.dtype
    displacement = (
        positions[route.receivers]
        - columns_positions[route.columns]
        + contract("ea,ad->ed", route.shifts.astype(dtype), cell_vectors.astype(dtype))
    )
    squared = jnp.sum(displacement * displacement, axis=-1)
    active = route.valid & (squared < cutoff * cutoff)
    positive = route.valid & (squared > 0.0)
    safe = jnp.where(positive, squared, cutoff * cutoff)
    # A coincident valid route keeps distance zero so the canonical kernels
    # report their geometry failure; padding routes sit at the cutoff.
    distance = jnp.where(
        positive, jnp.sqrt(safe), jnp.where(route.valid, 0.0, cutoff)
    ).astype(dtype)
    return _PairGeometry(distance, active, owned_ids[route.receivers])


def owner_lennard_jones_share(
    term: AtomisticDomainLennardJonesTerm,
    requirement: AtomisticNeighborhoodRequirement,
    positions: Array,
    columns_positions: Array,
    column_types: Array,
    owned_types: Array,
    route: OwnerTermRoute,
    cell_vectors: Array,
    owned_ids: Array,
    axis_name: str,
    /,
) -> OwnerTermResult:
    """Owner share of the canonical Lennard-Jones pair sum (each pair once)."""
    plan = term.prepared.plan
    cutoff = requirement.bounded_cutoff()
    geometry = _pair_geometry(
        positions, columns_positions, route, cell_vectors, owned_ids, cutoff
    )
    scales = jnp.ones(route.valid.shape, dtype=positions.dtype)
    if requirement.exclusions is AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS:
        keys = _pair_keys(geometry.receiver_ids, route.source_ids, route.valid)
        lj, _ = term.system.topology.pair_scales(keys)
        scales = lj.astype(positions.dtype)
    interaction_scales = AtomisticInteractionScaleState.identity(
        term.system, route.valid.shape[0], positions.dtype
    )
    interaction, geometry_ok = lennard_jones_pair_interactions(
        plan,
        owned_types[route.receivers],
        column_types[route.columns],
        geometry.distance,
        route.valid,
        scales,
        interaction_scales,
    )
    ambiguity = jnp.asarray(False)
    if requirement.counting is AtomisticPairCounting.MINIMUM_IMAGE_HALF:
        ambiguity = _duplicate_images(
            route.receivers, route.source_ids, geometry.receiver_ids, geometry.active
        )
    share = 0.5 * jnp.sum(interaction)
    scientific = _global_all(geometry_ok & ~ambiguity & jnp.isfinite(share), axis_name)
    return OwnerTermResult(
        share, jnp.asarray(True), scientific, ~_global_all(~ambiguity, axis_name)
    )


def owner_ewald_real_share(
    term: AtomisticDomainEwaldTerm,
    requirement: AtomisticNeighborhoodRequirement,
    positions: Array,
    columns_positions: Array,
    column_charges: Array,
    owned_charges: Array,
    route: OwnerTermRoute,
    cell_vectors: Array,
    owned_ids: Array,
    axis_name: str,
    /,
) -> OwnerTermResult:
    """Owner share of the canonical Ewald real-space pair energy (each pair once)."""
    settings = term.prepared.plan.settings
    constant = term.system.plan.units.coulomb_constant
    dtype = positions.dtype
    cutoff = settings.real_cutoff
    explicit = settings.real_space == "explicit-image"
    geometry = _pair_geometry(
        positions, columns_positions, route, cell_vectors, owned_ids, cutoff
    )
    left = owned_charges[route.receivers].astype(dtype)
    right = column_charges[route.columns].astype(dtype)
    positive = geometry.distance > 0.0
    safe_distance = jnp.where(positive, geometry.distance, cutoff)
    if explicit:
        # The image-pair sum: canonical (q q erfc) / r, no exception scale.
        values = ewald_real_space_kernel(
            settings.alpha, safe_distance, numerator=left * right
        )
        ambiguity = jnp.asarray(False)
    else:
        scales = jnp.ones(route.valid.shape, dtype=dtype)
        if requirement.exclusions is AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS:
            keys = _pair_keys(geometry.receiver_ids, route.source_ids, route.valid)
            _, electrostatic = term.system.topology.pair_scales(keys)
            scales = electrostatic.astype(dtype)
        values = (
            left * right * ewald_real_space_kernel(settings.alpha, safe_distance) * scales
        )
        ambiguity = _duplicate_images(
            route.receivers, route.source_ids, geometry.receiver_ids, geometry.active
        )
    real = jnp.where(geometry.active & positive, values, 0.0)
    geometry_ok = jnp.all(~geometry.active | positive)
    share = 0.5 * constant * jnp.sum(real)
    scientific = _global_all(geometry_ok & ~ambiguity & jnp.isfinite(share), axis_name)
    return OwnerTermResult(
        share, jnp.asarray(True), scientific, ~_global_all(~ambiguity, axis_name)
    )


def owner_ewald_exception_share(
    term: AtomisticDomainEwaldTerm,
    positions: Array,
    columns_positions: Array,
    column_charges: Array,
    owned_charges: Array,
    route: OwnerExceptionRoute,
    cell_vectors: Array,
    axis_name: str,
    /,
) -> OwnerTermResult:
    """Owner share of the canonical Ewald topology-exception correction.

    Each listed pair contributes ``-k (1 - s) q_i q_j K(r_mic)`` once, from its
    two receiver-owned directed routes (weight ``1/2`` each), at the current
    minimum image (`_minimum_image` of the canonical term).
    """
    table = term.exception_table
    if table is None:
        raise RuntimeError("The Ewald term lists no exceptions.")
    settings = term.prepared.plan.settings
    system = term.system
    cell = system.cell
    if cell is None:
        raise ValueError("Ewald exception corrections need the system's periodic cell.")
    constant = system.plan.units.coulomb_constant
    dtype = positions.dtype
    explicit = settings.real_space == "explicit-image"
    vectors = cell_vectors.astype(dtype)
    inverse, inverse_ok = cell.inverse_for_vectors_with_status(vectors)
    inverse = inverse.astype(dtype)
    displacement = _minimum_image(
        cell,
        positions[route.receivers] - columns_positions[route.columns],
        vectors,
        inverse,
    )
    squared = jnp.sum(displacement * displacement, axis=-1)
    positive = squared > 0.0
    distance = jnp.sqrt(jnp.where(positive, squared, 1.0))
    smeared = ewald_exception_kernel(
        settings.alpha,
        settings.real_cutoff,
        distance,
        positive,
        explicit_image=explicit,
    )
    scale = table.weights[route.rows, 0].astype(dtype)
    left = owned_charges[route.receivers].astype(dtype)
    right = column_charges[route.columns].astype(dtype)
    exception = -jnp.sum(
        jnp.where(route.valid, (1.0 - scale) * left * right * smeared, 0.0)
    )
    share = 0.5 * constant * exception
    geometry_ok = inverse_ok & (
        jnp.all(~route.valid | positive) if explicit else jnp.asarray(True)
    )
    scientific = _global_all(geometry_ok & jnp.isfinite(share), axis_name) & (
        route.missing == 0
    )
    return OwnerTermResult(
        share, route.capacity_successful, scientific, jnp.asarray(False)
    )


def owner_reciprocal_share(
    prepared: PreparedReciprocalDistribution,
    positions: Array,
    charges: Array,
    image_counts: Array,
    atom_mask: Array,
    cell_vectors: Array,
    axis_name: str,
    /,
) -> OwnerTermResult:
    """Owner share of the term's distributed reciprocal, self and system energy."""
    share, evidence = owner_reciprocal_energy(
        prepared,
        positions,
        ReciprocalSitePayload(values=charges, image_counts=image_counts),
        atom_mask,
        cell_vectors,
        axis_name,
    )
    return OwnerTermResult(
        share,
        jnp.asarray(evidence.capacity_successful, dtype=jnp.bool_),
        jnp.asarray(evidence.scientific_successful, dtype=jnp.bool_),
        jnp.asarray(False),
    )


def owner_lennard_jones_pme_real_share(
    term: AtomisticDomainLennardJonesPMETerm,
    requirement: AtomisticNeighborhoodRequirement,
    positions: Array,
    columns_positions: Array,
    column_types: Array,
    owned_types: Array,
    route: OwnerTermRoute,
    cell_vectors: Array,
    owned_ids: Array,
    axis_name: str,
    /,
) -> OwnerTermResult:
    """Owner share of the canonical LJ-PME real-space pairs (each pair once).

    Pairs inside the cutoff carry the screened dispersion and the exception
    correction of `lennard_jones_pme_pair_energy` with their topology
    Lennard-Jones scales, exactly as the single-device pair relation.
    """
    c6, alpha, _, _ = term.prepared.plan.arrays
    dtype = positions.dtype
    cutoff = requirement.bounded_cutoff()
    geometry = _pair_geometry(
        positions, columns_positions, route, cell_vectors, owned_ids, cutoff
    )
    scales = jnp.ones(route.valid.shape, dtype=dtype)
    if requirement.exclusions is AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS:
        keys = _pair_keys(geometry.receiver_ids, route.source_ids, route.valid)
        lj, _ = term.system.topology.pair_scales(keys)
        scales = lj.astype(dtype)
    positive = geometry.distance > 0.0
    values = lennard_jones_pme_pair_energy(
        c6.astype(dtype),
        alpha.reshape(()).astype(dtype),
        owned_types[route.receivers],
        column_types[route.columns],
        jnp.where(positive, geometry.distance, cutoff),
        scales,
    )
    real = jnp.where(geometry.active & positive, values, 0.0)
    geometry_ok = jnp.all(~geometry.active | positive)
    ambiguity = jnp.asarray(False)
    if requirement.counting is AtomisticPairCounting.MINIMUM_IMAGE_HALF:
        ambiguity = _duplicate_images(
            route.receivers, route.source_ids, geometry.receiver_ids, geometry.active
        )
    share = 0.5 * jnp.sum(real)
    scientific = _global_all(geometry_ok & ~ambiguity & jnp.isfinite(share), axis_name)
    return OwnerTermResult(
        share, jnp.asarray(True), scientific, ~_global_all(~ambiguity, axis_name)
    )


def owner_lennard_jones_pme_reciprocal_share(
    prepared: PreparedReciprocalDistribution,
    positions: Array,
    owned_types: Array,
    anchor: Array,
    atom_mask: Array,
    cell_vectors: Array,
    axis_name: str,
    /,
) -> OwnerTermResult:
    """Owner share of the distributed LJ-PME reciprocal and self energies."""
    share, evidence = owner_reciprocal_energy(
        prepared,
        positions,
        ReciprocalSitePayload(rows=owned_types, anchor=anchor),
        atom_mask,
        cell_vectors,
        axis_name,
    )
    return OwnerTermResult(
        share,
        jnp.asarray(evidence.capacity_successful, dtype=jnp.bool_),
        jnp.asarray(evidence.scientific_successful, dtype=jnp.bool_),
        jnp.asarray(False),
    )


def owner_fourier_d3_share(
    term: AtomisticDomainFourierD3Term,
    prepared: PreparedReciprocalDistribution,
    positions: Array,
    columns_positions: Array,
    column_table_rows: Array,
    column_active: Array,
    owned_table_rows: Array,
    owned_channel_rows: Array,
    owned_active: Array,
    route: OwnerTermRoute,
    image_counts: Array,
    atom_mask: Array,
    cell_vectors: Array,
    axis_name: str,
    /,
) -> OwnerTermResult:
    """Owner share of FourierD3: modified CN, coefficients, distributed mesh.

    The owned receivers' modified coordination numbers are summed over their
    full directed coordination routes (columns from the role exchange); the
    signed rank coefficients and the distributed reciprocal plus self energy
    then form one owner scalar, so the coordination cotangents return to the
    owned and column positions through the exchange transpose exactly once.
    """
    definition = term.prepared
    dtype = positions.dtype
    displacement = (
        positions[route.receivers]
        - columns_positions[route.columns]
        + contract("ea,ad->ed", route.shifts.astype(dtype), cell_vectors.astype(dtype))
    )
    active = atom_mask & owned_active
    coordination, endpoints_ok, coincidence_ok = definition.row_coordination_numbers(
        displacement,
        route.valid,
        owned_table_rows[route.receivers],
        column_table_rows[route.columns],
        active[route.receivers],
        column_active[route.columns],
        route.receivers,
        positions.shape[0],
    )
    coefficients = definition.row_coefficients(coordination, owned_channel_rows, active)
    share, evidence = owner_reciprocal_energy(
        prepared,
        positions,
        ReciprocalSitePayload(
            values=coefficients, rows=owned_channel_rows, image_counts=image_counts
        ),
        active,
        cell_vectors,
        axis_name,
    )
    scientific = jnp.asarray(evidence.scientific_successful, dtype=jnp.bool_) & (
        _global_all(endpoints_ok & coincidence_ok & jnp.isfinite(share), axis_name)
    )
    return OwnerTermResult(
        share,
        jnp.asarray(evidence.capacity_successful, dtype=jnp.bool_),
        scientific,
        jnp.asarray(False),
    )


def owner_polarization_share(
    term: AtomisticDomainPolarizationTerm,
    prepared: PreparedOwnerPolarization,
    positions: Array,
    route: OwnerTermRoute,
    halo: DistributedHaloPlan,
    send_slots: Array,
    send_valid: Array,
    exceptions: OwnerExceptionRoute | None,
    exception_halo: DistributedHaloPlan | None,
    fields: dict[str, Array],
    atom_mask: Array,
    cell_vectors: Array,
    axis_name: str,
    message_capacity_bytes: int,
    /,
) -> OwnerTermResult:
    """Owner share of the converged (or TCG) induced-dipole polarization energy."""
    key = term.term_key
    dtype = positions.dtype
    count = route.valid.shape[0]
    real_routes = OwnerMultipoleRoutes(
        receivers=route.receivers,
        columns=route.columns,
        shifts=route.shifts,
        weights=jnp.ones((count, 3), dtype=dtype),
        valid=route.valid,
        send_slots=send_slots,
        send_valid=send_valid,
    )
    real = OwnerMultipoleRelation(
        region_halo(halo, send_slots, send_valid),
        real_routes,
        message_capacity_bytes=message_capacity_bytes,
    )
    corrections = correction_routes = None
    capacity = jnp.asarray(True)
    missing_ok = jnp.asarray(True)
    if exceptions is not None:
        table = term.exception_table
        if table is None or exception_halo is None:
            raise RuntimeError("Polarization exception routes lack their table.")
        correction_routes = OwnerMultipoleRoutes(
            receivers=exceptions.receivers,
            columns=exceptions.columns,
            shifts=jnp.zeros(
                (exceptions.receivers.shape[0], 3), dtype=route.shifts.dtype
            ),
            weights=jnp.where(
                exceptions.valid[:, None],
                table.weights[exceptions.rows].astype(dtype),
                0.0,
            ),
            valid=exceptions.valid,
            send_slots=exceptions.send_slots,
            send_valid=exceptions.send_valid,
        )
        corrections = OwnerMultipoleRelation(
            region_halo(exception_halo, exceptions.send_slots, exceptions.send_valid),
            correction_routes,
            minimum_image=True,
            cell=term.system.cell,
            message_capacity_bytes=message_capacity_bytes,
        )
        capacity = exceptions.capacity_successful
        missing_ok = exceptions.missing == 0
    moments = jax.tree.unflatten(
        term.moment_tree,
        [
            fields[_field(key, f"moment-{index}")].astype(dtype)
            for index in range(term.moment_tree.num_leaves)
        ],
    )
    sites = OwnerPolarizationSites(
        moments=moments,
        polarizabilities=fields[_field(key, "polarizabilities")].astype(dtype),
        damping=fields[_field(key, "damping")].astype(dtype),
        width_classes=fields.get(_field(key, "width_classes")),
    )
    result = solve_owner_polarization(
        prepared,
        positions,
        sites,
        atom_mask & fields[_field(key, "active")],
        cell_vectors,
        axis_name,
        OwnerPolarizationRelations(
            real_space=real,
            real_routes=real_routes,
            corrections=corrections,
            correction_routes=correction_routes,
        ),
    )
    return OwnerTermResult(
        result.energy_share,
        capacity,
        jnp.asarray(result.successful, dtype=jnp.bool_) & missing_ok,
        jnp.asarray(False),
    )


__all__ = [
    "AbstractAtomisticDomainTerm",
    "AtomisticDomainEwaldTerm",
    "AtomisticDomainExceptionTable",
    "AtomisticDomainExceptionTopology",
    "AtomisticDomainFourierD3Term",
    "AtomisticDomainLennardJonesPMETerm",
    "AtomisticDomainLennardJonesTerm",
    "AtomisticDomainPolarizationTerm",
    "OwnerExceptionRoute",
    "OwnerTermResult",
    "OwnerTermRoute",
    "PreparedAtomisticDomainTerm",
    "domain_exception_topology",
    "owner_ewald_exception_share",
    "owner_ewald_real_share",
    "owner_fourier_d3_share",
    "owner_lennard_jones_pme_real_share",
    "owner_lennard_jones_pme_reciprocal_share",
    "owner_lennard_jones_share",
    "owner_polarization_share",
    "owner_reciprocal_share",
    "region_halo",
    "replicated_halo",
    "require_domain_requirement",
]
