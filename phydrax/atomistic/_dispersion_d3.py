#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conventional two-body DFT-D3(BJ) dispersion with truncated coordination numbers.

The energy of the declared scientific definition is

``E = -1/2 sum'_{i,j,T} S(r) [ s6 C6_ij / (r^6 + R0^6) + s8 C8_ij / (r^8 + R0^8) ]``

over every directed image route ``(i, j, T)`` with ``0 < r = |x_i - x_j + T| <
dispersion cutoff`` (repeated and nonzero self images included), with

* ``C6_ij = C6(CN_i, CN_j)`` interpolated from the reference tables under the
  declared :class:`D3CoordinationPolicy`,
* ``C8_ij = 3 C6_ij r4r2_a r4r2_b`` and ``R0 = a1 sqrt(3 r4r2_a r4r2_b) + a2``,
* ``CN_i`` the conventional logistic count over every full directed image
  route ``0 < r < coordination cutoff``,
* ``S`` either a hard truncation at the dispersion cutoff or the S5 switch
  ``1 - (10 t^3 - 15 t^4 + 6 t^5)`` on ``[switch_on, switch_off]``.

There is no three-body (ATM/C9) term and no zero damping; neither is
substituted. Coordination and dispersion use separate neighborhood roles and
radii: full directed coordination and the explicitly counted image-pair sum
(canonical halves with ``directed_multiplicity == 2``). Evaluation runs in the
table's length/energy units and returns system units through exact host
factors. Every coordinate, cell, table and damping-parameter derivative is the
derivative of this one scalar closure (CN and C6 included); it is valid inside
the regularity domain reported by :class:`D3BJRegularity`.

Source fidelity: the inspected NVIDIA nvalchemi-toolkit-ops
``10e12340c39fd12a2375d6301768e72d2cd12868`` conventional kernel uses one
neighbor list for both coordination and dispersion (equal radii here), the
thresholded interpolation (``"nvalchemi-relative-threshold"``) and the same S5
switch. It silently skips coincident atoms (``r < 1e-12``); here a counted
coincident route is a geometry failure.
"""

from __future__ import annotations

import math
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._d3_parameters import (
    D3BJDamping,
    D3CoordinationPolicy,
    D3ReferenceTables,
    energy_factor,
    length_factor,
)
from ._neighborhood_bundle import (
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticNeighborhoodView,
    AtomisticPairCounting,
)
from ._potential import AtomisticPotentialCapabilities, AtomisticPotentialRequirements
from ._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialContext,
    AtomisticTermEvaluation,
)
from ._sites import AtomisticSiteDomain
from ._system import PreparedAtomisticSystem


def _identity(kind: str, values: dict, /) -> IdentityWords:
    return IdentityWords.of({"kind": kind, **values})


def _term_identity(name: str, force_group: int, /) -> tuple[str, int]:
    identifier = str(name).strip()
    if not identifier:
        raise ValueError("Energy term name must be non-empty.")
    if isinstance(force_group, bool) or not isinstance(force_group, int):
        raise TypeError("force_group must be an integer.")
    if not 0 <= force_group < 32:
        raise ValueError("force_group must be in [0, 32).")
    return identifier, force_group


def _positive(value: float, name: str, /) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _count(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


@final
class D3Routes(StrictModule):
    """Explicit route geometry consumed by D3 and FourierD3 evaluation.

    ``displacement[e] = x[receivers[e]] - x[sources[e]] + T_e`` in system
    length, differentiable in positions and cell. ``counted`` marks valid
    routes inside the role radius (a discrete, gradient-free mask).
    ``directed_multiplicity`` is the number of directed image routes one
    stored route represents: one for full directed sets, two for canonical
    halves. ``successful`` carries the resource/role status of the set.
    """

    receivers: Array
    sources: Array
    displacement: Array
    counted: Array
    successful: Array
    directed_multiplicity: int = eqx.field(static=True)

    def __init__(
        self,
        receivers: Array,
        sources: Array,
        displacement: Array,
        counted: Array,
        successful: Array,
        /,
        *,
        directed_multiplicity: int,
    ) -> None:
        receiver = jnp.asarray(receivers)
        source = jnp.asarray(sources)
        vectors = jnp.asarray(displacement)
        mask = jnp.asarray(counted)
        if receiver.ndim != 1 or source.shape != receiver.shape:
            raise ValueError("Route receivers and sources must be equal 1D arrays.")
        if not jnp.issubdtype(receiver.dtype, jnp.integer) or not jnp.issubdtype(
            source.dtype, jnp.integer
        ):
            raise TypeError("Route endpoints must be integer indices.")
        if vectors.shape != receiver.shape + (3,):
            raise ValueError("Route displacements must have shape (routes, 3).")
        if mask.shape != receiver.shape or mask.dtype != jnp.bool_:
            raise ValueError("counted must be one boolean per route.")
        multiplicity = _count(directed_multiplicity, "directed_multiplicity")
        if multiplicity not in (1, 2):
            raise ValueError("A stored route represents one or two directed routes.")
        self.receivers = receiver
        self.sources = source
        self.displacement = vectors
        self.counted = mask
        self.successful = jnp.asarray(successful, dtype=jnp.bool_)
        self.directed_multiplicity = multiplicity

    @classmethod
    def from_view(
        cls,
        view: AtomisticNeighborhoodView,
        positions: Array,
        cell_vectors: Array,
        /,
        *,
        counting: AtomisticPairCounting,
    ) -> "D3Routes":
        """Bind a resolved role view at the current positions and cell."""
        if view.entity not in (
            AtomisticSiteDomain.PHYSICAL_ATOMS,
            AtomisticSiteDomain.DOF_ATOMS,
        ):
            raise ValueError("D3 routes must address the system atoms.")
        if view.counting is not counting:
            raise ValueError(
                f"D3 requires {counting.value} routes; the view counts {view.counting.value}."
            )
        dtype = positions.dtype
        vectors = jnp.asarray(cell_vectors).astype(dtype)
        return cls(
            view.relation.receiver_indices,
            view.relation.source_indices,
            view.displacement(positions, vectors),
            view.cutoff_mask(positions, vectors),
            jnp.all(view.evidence.successful),
            directed_multiplicity=view.directed_multiplicity,
        )


def _route_lengths(
    routes: D3Routes, cutoff: float, scale: float, /
) -> tuple[Array, Array, Array]:
    """Table-unit lengths, the counted mask inside ``cutoff`` and coincidence status."""
    squared = jnp.sum(routes.displacement * routes.displacement, axis=-1)
    inside = routes.counted & jax.lax.stop_gradient(squared < cutoff * cutoff)
    positive = squared > 0.0
    safe = jnp.where(inside & positive, squared, 1.0)
    return jnp.sqrt(safe) * scale, inside & positive, jnp.all(~inside | positive)


def _endpoints_active(routes: D3Routes, active: Array, counted: Array, /) -> Array:
    return jnp.all(~counted | (active[routes.receivers] & active[routes.sources]))


def s5_switch(distance: Array, switch_on: ArrayLike, switch_off: ArrayLike, /) -> Array:
    """``1 - (10 t^3 - 15 t^4 + 6 t^5)`` on ``[on, off]``; C2 at both joins."""
    width = jnp.subtract(switch_off, switch_on)
    inside = (distance > switch_on) & (distance < switch_off)
    t = jnp.where(inside, (distance - switch_on) / width, 0.0)
    smooth = 1.0 - t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
    return jnp.where(distance <= switch_on, 1.0, jnp.where(inside, smooth, 0.0))


def interpolate_c6(
    policy: D3CoordinationPolicy,
    c6_reference: Array,
    coordination_left: Array,
    coordination_right: Array,
    reference_left: Array,
    reference_right: Array,
    valid_left: Array,
    valid_right: Array,
    /,
) -> Array:
    """Pair C6 of ``(..., R, R)`` reference blocks at the two coordination numbers.

    The log weights are shifted by their stop-gradient maximum over populated
    slot pairs (shift invariance makes this exact). The policy's dynamic
    scalars are evaluated in the coordination dtype.
    """
    dtype = coordination_left.dtype
    width = policy.width.astype(dtype)
    left = -width * (coordination_left[..., :, None] - reference_left) ** 2
    right = -width * (coordination_right[..., :, None] - reference_right) ** 2
    exponent = left[..., :, None] + right[..., None, :]
    populated = valid_left[..., :, None] & valid_right[..., None, :]
    shift = jax.lax.stop_gradient(
        jnp.max(jnp.where(populated, exponent, -jnp.inf), axis=(-2, -1), keepdims=True)
    )
    relative = jnp.where(populated, exponent - shift, 0.0)
    kept = populated
    if policy.threshold is not None:
        kept = kept & jax.lax.stop_gradient(
            relative >= -policy.threshold.astype(relative.dtype)
        )
    weight = jnp.where(kept, jnp.exp(relative), 0.0)
    return jnp.sum(weight * c6_reference, axis=(-2, -1)) / jnp.sum(weight, axis=(-2, -1))


def conventional_count(ratio: Array, steepness: ArrayLike, /) -> Array:
    """Grimme's logistic count ``1 / (1 + exp(-k1 (R_cov / r - 1)))``."""
    return jax.nn.sigmoid(steepness * (ratio - 1.0))


@final
class D3DispersionCutoff(StrictModule, NonTrainableState):
    """Dispersion route radius and its declared truncation.

    ``cutoff`` (system length) is the image-pair-sum role radius. Without a
    switch the energy steps at ``cutoff``. With ``switch_on < switch_off <=
    cutoff`` the S5 switch makes every route energy C2 at both joins (its third
    derivative jumps) and zero beyond ``switch_off``.

    PyTree layout: ``cutoff`` defines the neighborhood requirement and stays
    static; whether a switch is present is the static truncation selector. The
    switch radii are dynamic float64 scalars and ``cutoff_identity`` the exact
    former ``cutoff_id`` digest (decoded by the host property).
    """

    cutoff: float = eqx.field(static=True)
    switch_on: Array | None
    switch_off: Array | None
    cutoff_identity: IdentityWords

    def __init__(
        self,
        cutoff: float,
        /,
        *,
        switch_on: float | None = None,
        switch_off: float | None = None,
    ) -> None:
        radius = _positive(cutoff, "cutoff")
        if (switch_on is None) != (switch_off is None):
            raise ValueError("The S5 switch needs both switch_on and switch_off.")
        on = off = None
        if switch_on is not None and switch_off is not None:
            on = _positive(switch_on, "switch_on")
            off = _positive(switch_off, "switch_off")
            if not on < off <= radius:
                raise ValueError(
                    "The S5 switch requires switch_on < switch_off <= cutoff."
                )
        self.cutoff = radius
        self.switch_on = None if on is None else jnp.asarray(on, dtype=jnp.float64)
        self.switch_off = None if off is None else jnp.asarray(off, dtype=jnp.float64)
        self.cutoff_identity = _identity(
            "d3-dispersion-cutoff",
            {
                "cutoff": float(radius).hex(),
                "truncation": "hard" if on is None else "s5-switch",
                "switch_on": None if on is None else float(on).hex(),
                "switch_off": None if off is None else float(off).hex(),
            },
        )

    @property
    def cutoff_id(self) -> str:
        """Host decoding of the canonical dispersion-cutoff digest."""
        return self.cutoff_identity.fingerprint()

    @property
    def switched(self) -> bool:
        return self.switch_on is not None


@final
class D3BJRegularity(StrictModule, NonTrainableState):
    """Regularity of the prepared definition, derived from its formulas.

    ``coordination_cutoff_step`` is the largest counting value just inside the
    coordination cutoff over covered species pairs: CN (hence C6 and the
    energy) steps there by up to that amount per route. ``dispersion_cutoff``
    names the dispersion join (``"value-step"`` with ``dispersion_cutoff_step``
    the largest reference-C6 pair energy at the cutoff, or ``"c2-s5-switch"``).
    ``interpolation`` is ``"smooth"`` or ``"threshold-steps"`` (with
    ``threshold_relative_weight``). Coordinate, cell and parameter derivatives
    of every order hold away from these sets; coincident atoms are refused.

    PyTree layout: the join/interpolation kinds follow the static selectors;
    the numeric evidence (``coordination_cutoff_step``,
    ``dispersion_cutoff_step``, ``threshold_relative_weight``) is dynamic
    float64.
    """

    coordination_cutoff_step: Array
    dispersion_cutoff: str = eqx.field(static=True)
    dispersion_cutoff_step: Array
    interpolation: str = eqx.field(static=True)
    threshold_relative_weight: Array | None


@final
class D3BJEvidence(StrictModule):
    """Status and components of one D3(BJ) evaluation.

    ``prepared_identity`` is the evaluating term's exact prepared digest as
    dynamic words (``prepared_id`` decodes it on the host).
    """

    coordination_numbers: Array
    coordination_routes_successful: Array
    dispersion_routes_successful: Array
    geometry_successful: Array
    endpoints_successful: Array
    pair_energy: Array
    method: str = eqx.field(static=True)
    prepared_identity: IdentityWords
    regularity: D3BJRegularity

    @property
    def prepared_id(self) -> str:
        """Host decoding of the evaluating term's prepared digest."""
        return self.prepared_identity.fingerprint()


@final
class D3BJEvaluation(StrictModule):
    """Energy, complete per-atom shares, status and evidence of one D3(BJ) evaluation.

    ``energy`` is the system-unit total (NaN unless ``successful``);
    ``atom_energy`` splits every counted route energy equally between its two
    endpoints, so it sums to the total and is complete. ``successful`` combines
    route, endpoint, coincidence and finiteness status; ``evidence`` retains
    its components.
    """

    energy: Array
    atom_energy: Array
    successful: Array
    evidence: D3BJEvidence


@final
class D3BJPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    """Conventional two-body DFT-D3(BJ) energy term (no ATM, no zero damping).

    ``coordination_cutoff`` and ``dispersion.cutoff`` are system lengths of the
    separate full directed coordination and image-pair-sum roles. ``route_chunk``
    bounds the resident ``routes x R x R`` interpolation block; dispersion
    routes run in checkpointed chunks of that size.

    PyTree layout: the requirement cutoffs, ``route_chunk``, ``name`` and
    ``force_group`` are static; ``term_identity`` is the dynamic exact former
    ``term_id`` digest (the base class decodes ``term_id`` on the host).
    """

    tables: D3ReferenceTables
    damping: D3BJDamping
    coordination: D3CoordinationPolicy
    dispersion: D3DispersionCutoff
    coordination_cutoff: float = eqx.field(static=True)
    route_chunk: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        tables: D3ReferenceTables,
        damping: D3BJDamping,
        coordination: D3CoordinationPolicy,
        dispersion: D3DispersionCutoff,
        /,
        *,
        coordination_cutoff: float,
        route_chunk: int = 4096,
        name: str = "d3-bj",
        force_group: int = 0,
    ) -> None:
        if not isinstance(tables, D3ReferenceTables):
            raise TypeError("tables must be D3ReferenceTables.")
        if not isinstance(damping, D3BJDamping):
            raise TypeError("damping must be D3BJDamping.")
        if not isinstance(coordination, D3CoordinationPolicy):
            raise TypeError("coordination must be a D3CoordinationPolicy.")
        if not isinstance(dispersion, D3DispersionCutoff):
            raise TypeError("dispersion must be a D3DispersionCutoff.")
        identifier, group = _term_identity(name, force_group)
        self.tables = tables
        self.damping = damping
        self.coordination = coordination
        self.dispersion = dispersion
        self.coordination_cutoff = _positive(coordination_cutoff, "coordination_cutoff")
        self.route_chunk = _count(route_chunk, "route_chunk")
        self.name = identifier
        self.force_group = group
        self.capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True,
            finite_geometry=True,
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            local_energy=True,
        )
        self.requirements = AtomisticPotentialRequirements(
            neighborhoods=(self.coordination_requirement, self.dispersion_requirement),
        )
        self.term_identity = _identity(
            "d3-bj-two-body-potential",
            {
                "tables": tables.tables_id,
                "damping": damping.damping_id,
                "coordination": coordination.policy_id,
                "dispersion": dispersion.cutoff_id,
                "coordination_cutoff": float(self.coordination_cutoff).hex(),
                "three_body": "excluded",
                "name": identifier,
                "force_group": group,
            },
        )

    @property
    def coordination_requirement(self) -> AtomisticNeighborhoodRequirement:
        return AtomisticNeighborhoodRequirement(
            self.name, AtomisticNeighborhoodRole.COORDINATION, self.coordination_cutoff
        )

    @property
    def dispersion_requirement(self) -> AtomisticNeighborhoodRequirement:
        return AtomisticNeighborhoodRequirement(
            self.name, AtomisticNeighborhoodRole.IMAGE_PAIR_SUM, self.dispersion.cutoff
        )

    def prepare(self, system: PreparedAtomisticSystem, /) -> "PreparedD3BJPotential":
        return PreparedD3BJPotential(self, system)


def prepared_species_rows(
    tables: D3ReferenceTables, system: PreparedAtomisticSystem, /
) -> tuple[np.ndarray, np.ndarray]:
    """Host table rows and activity of the system atoms; refuses uncovered elements.

    Inactive capacity slots are explicit padding (row 0, inactive); an active
    particle that is not an element or whose element has no table row refuses.
    """
    plan = system.plan
    active = np.asarray(system.active_mask, dtype=np.bool_)
    elements = np.asarray(plan.element_mask, dtype=np.bool_)
    numbers = np.asarray(plan.atomic_numbers, dtype=np.int64)
    if np.any(active & ~elements):
        raise ValueError("D3 dispersion cannot evaluate active non-element particles.")
    missing = tables.uncovered(numbers[active])
    if missing:
        raise ValueError(
            f"D3 reference tables do not cover active atomic numbers {list(missing)}."
        )
    rows = np.zeros(numbers.shape, dtype=np.int32)
    if np.any(active):
        rows[active] = tables.rows(numbers[active])
    return rows, active


def system_scales(
    tables: D3ReferenceTables, system: PreparedAtomisticSystem, /
) -> tuple[float, float]:
    """Exact host factors: system length -> table length, table energy -> system energy."""
    scale = system.plan.units.scale
    return (
        length_factor(scale.length_unit, tables.length_unit),
        energy_factor(tables.energy_unit, scale.energy_unit),
    )


def _pair_bound(
    tables: D3ReferenceTables,
    damping: D3BJDamping,
    rows: np.ndarray,
    distance: float,
    a2: float,
    /,
) -> float:
    c6 = np.asarray(tables.c6_reference)[np.ix_(rows, rows)]
    peak = np.max(c6, axis=(2, 3))
    r4r2 = np.asarray(tables.r4r2)[rows]
    product = 3.0 * r4r2[:, None] * r4r2[None, :]
    radius = float(damping.a1) * np.sqrt(product) + a2
    values = peak * (
        abs(float(damping.s6)) / (distance**6 + radius**6)
        + abs(float(damping.s8)) * product / (distance**8 + radius**8)
    )
    return float(np.max(values))


@final
class PreparedD3BJPotential(AbstractPreparedAtomisticEnergyTerm):
    """System-bound D3(BJ): table rows, unit factors and declared regularity.

    PyTree layout: the exact unit factors (functions of the static declared
    units only) stay static; ``term_identity``/``prepared_identity`` are the
    dynamic exact former ``term_id``/``prepared_id`` digests and the
    regularity evidence is dynamic. ``energy`` folds the numerical binding of
    the context system into the term status (never decoded in traced code).
    """

    plan: D3BJPotential
    system: PreparedAtomisticSystem
    rows: Array
    active: Array
    length_scale: float = eqx.field(static=True)
    energy_scale: float = eqx.field(static=True)
    a2_scale: float = eqx.field(static=True)
    regularity: D3BJRegularity
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan: D3BJPotential, system: PreparedAtomisticSystem, /) -> None:
        plan.tables.require_use()
        rows, active = prepared_species_rows(plan.tables, system)
        length, energy = system_scales(plan.tables, system)
        a2_scale = length_factor(plan.damping.a2_unit, plan.tables.length_unit)
        covered = np.unique(rows[active]) if np.any(active) else np.zeros((0,), np.int32)
        steps = [0.0]
        dispersion_step = 0.0
        if covered.size:
            radii = np.asarray(plan.tables.covalent_radii)[covered]
            covalent = radii[:, None] + radii[None, :]
            ratio = covalent / (plan.coordination_cutoff * length)
            steps = (
                (
                    1.0
                    / (1.0 + np.exp(-float(plan.coordination.steepness) * (ratio - 1.0)))
                )
                .reshape(-1)
                .tolist()
            )
            if not plan.dispersion.switched:
                dispersion_step = energy * _pair_bound(
                    plan.tables,
                    plan.damping,
                    covered,
                    plan.dispersion.cutoff * length,
                    float(plan.damping.a2) * a2_scale,
                )
        self.plan = plan
        self.system = system
        self.rows = jnp.asarray(rows)
        self.active = jnp.asarray(active)
        self.length_scale = length
        self.energy_scale = energy
        self.a2_scale = a2_scale
        threshold = plan.coordination.threshold
        self.regularity = D3BJRegularity(
            coordination_cutoff_step=jnp.asarray(float(max(steps)), dtype=jnp.float64),
            dispersion_cutoff="value-step"
            if not plan.dispersion.switched
            else "c2-s5-switch",
            dispersion_cutoff_step=jnp.asarray(float(dispersion_step), dtype=jnp.float64),
            interpolation="smooth" if threshold is None else "threshold-steps",
            threshold_relative_weight=None
            if threshold is None
            else jnp.asarray(math.exp(-float(threshold)), dtype=jnp.float64),
        )
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_identity = _identity(
            "prepared-d3-bj-potential",
            {
                "term": plan.term_id,
                "system": system.prepared_id,
                "length_scale": float(length).hex(),
                "energy_scale": float(energy).hex(),
            },
        )

    @property
    def method(self) -> str:
        cutoff = "s5" if self.plan.dispersion.switched else "hard"
        return (
            f"d3-bj-two-body-conventional-cn-{self.plan.coordination.interpolation}-"
            f"{cutoff}-cutoff"
        )

    def coordination_numbers(self, routes: D3Routes, /) -> tuple[Array, Array, Array]:
        """Conventional CN of every atom with route and coincidence status.

        Requires a full directed route set (``directed_multiplicity == 1``).
        """
        if routes.directed_multiplicity != 1:
            raise ValueError("Coordination numbers need full directed routes.")
        tables = self.plan.tables
        distance, counted, geometry = _route_lengths(
            routes, self.plan.coordination_cutoff, self.length_scale
        )
        dtype = distance.dtype
        radii = tables.covalent_radii.astype(dtype)
        covalent = radii[self.rows[routes.receivers]] + radii[self.rows[routes.sources]]
        values = conventional_count(
            covalent / distance, self.plan.coordination.steepness.astype(dtype)
        )
        values = jnp.where(counted, values, 0.0)
        coordination = jnp.zeros((self.system.capacity,), dtype=dtype)
        coordination = coordination.at[routes.receivers].add(values)
        status = routes.successful & _endpoints_active(routes, self.active, counted)
        return coordination, status, geometry

    def _route_energies(
        self, coordination: Array, routes: D3Routes, distance: Array, counted: Array, /
    ) -> Array:
        tables = self.plan.tables
        damping = self.plan.damping
        policy = self.plan.coordination
        dtype = distance.dtype
        chunk = min(self.plan.route_chunk, max(int(distance.shape[0]), 1))
        count = distance.shape[0]
        padding = -(-count // chunk) * chunk - count

        def padded(value: Array, fill: int | float | bool) -> Array:
            tail = jnp.full((padding,) + value.shape[1:], fill, dtype=value.dtype)
            return jnp.concatenate((value, tail)).reshape((-1, chunk) + value.shape[1:])

        left = self.rows[routes.receivers]
        right = self.rows[routes.sources]
        inputs = (
            padded(left, 0),
            padded(right, 0),
            padded(coordination[routes.receivers], 0.0),
            padded(coordination[routes.sources], 0.0),
            padded(distance, 1.0),
            padded(counted, False),
        )
        c6_table = tables.c6_reference.astype(dtype)
        reference = tables.reference_coordination.astype(dtype)
        valid = tables.reference_valid
        r4r2 = tables.r4r2.astype(dtype)
        s6 = damping.s6.astype(dtype)
        s8 = damping.s8.astype(dtype)
        a1 = damping.a1.astype(dtype)
        a2 = damping.a2.astype(dtype) * self.a2_scale
        switch_on = self.plan.dispersion.switch_on
        switch_off = self.plan.dispersion.switch_off

        def body(chunk_inputs: tuple[Array, ...]) -> Array:
            a, b, cn_a, cn_b, r, mask = chunk_inputs
            c6 = interpolate_c6(
                policy,
                c6_table[a, b],
                cn_a,
                cn_b,
                reference[a],
                reference[b],
                valid[a],
                valid[b],
            )
            product = 3.0 * r4r2[a] * r4r2[b]
            radius = a1 * jnp.sqrt(product) + a2
            r2 = r * r
            r6 = r2 * r2 * r2
            radius2 = radius * radius
            radius6 = radius2 * radius2 * radius2
            energy = -c6 * (
                s6 / (r6 + radius6) + s8 * product / (r6 * r2 + radius6 * radius2)
            )
            if switch_on is not None and switch_off is not None:
                energy = energy * s5_switch(
                    r / self.length_scale,
                    switch_on.astype(dtype),
                    switch_off.astype(dtype),
                )
            return jnp.where(mask, energy, 0.0)

        energies = jax.lax.map(jax.checkpoint(body), inputs)
        return energies.reshape((-1,))[:count]

    def evaluate(self, coordination: D3Routes, dispersion: D3Routes, /) -> D3BJEvaluation:
        """Energy of explicit coordination (full directed) and dispersion routes."""
        numbers, coordination_ok, coordination_geometry = self.coordination_numbers(
            coordination
        )
        distance, counted, geometry = _route_lengths(
            dispersion, self.plan.dispersion.cutoff, self.length_scale
        )
        route = self._route_energies(numbers, dispersion, distance, counted)
        weight = 0.5 * dispersion.directed_multiplicity * self.energy_scale
        pair = weight * route
        total = jnp.sum(pair)
        atom_energy = jnp.zeros((self.system.capacity,), dtype=pair.dtype)
        atom_energy = atom_energy.at[dispersion.receivers].add(0.5 * pair)
        atom_energy = atom_energy.at[dispersion.sources].add(0.5 * pair)
        dispersion_ok = dispersion.successful
        endpoints = _endpoints_active(dispersion, self.active, counted)
        geometry_ok = geometry & coordination_geometry
        successful = (
            coordination_ok
            & dispersion_ok
            & endpoints
            & geometry_ok
            & jnp.isfinite(total)
        )
        evidence = D3BJEvidence(
            coordination_numbers=numbers,
            coordination_routes_successful=coordination_ok,
            dispersion_routes_successful=dispersion_ok,
            geometry_successful=geometry_ok,
            endpoints_successful=endpoints,
            pair_energy=total,
            method=self.method,
            prepared_identity=self.prepared_identity,
            regularity=self.regularity,
        )
        return D3BJEvaluation(
            energy=jnp.where(successful, total, jnp.nan),
            atom_energy=atom_energy,
            successful=successful,
            evidence=evidence,
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        # The exact numerical system binding is compared as words and folded
        # into the status: a foreign context system refuses without decoding.
        bound = context.system.prepared_identity.equals(self.system.prepared_identity)
        positions = context.positions
        vectors = context.cell_vectors
        coordination = D3Routes.from_view(
            context.neighborhood(self.plan.coordination_requirement),
            positions,
            vectors,
            counting=AtomisticPairCounting.DIRECTED,
        )
        dispersion = D3Routes.from_view(
            context.neighborhood(self.plan.dispersion_requirement),
            positions,
            vectors,
            counting=AtomisticPairCounting.IMAGE_HALF,
        )
        result = self.evaluate(coordination, dispersion)
        successful = result.successful & bound
        return AtomisticTermEvaluation(
            jnp.where(successful, result.energy, jnp.nan),
            result.atom_energy,
            successful,
            atom_energy_complete=True,
            evidence=(result.evidence,),
        )


__all__ = [
    "conventional_count",
    "D3BJEvaluation",
    "D3BJEvidence",
    "D3BJPotential",
    "D3BJRegularity",
    "D3DispersionCutoff",
    "D3Routes",
    "interpolate_c6",
    "prepared_species_rows",
    "PreparedD3BJPotential",
    "s5_switch",
    "system_scales",
]
