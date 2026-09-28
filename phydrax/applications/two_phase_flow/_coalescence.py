#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Film drainage and coalescence of resolved bubble and drop pairs.

A resolved pair whose interfaces are held together by a near-contact force ``F``
flattens into a film of radius ``a`` and thickness ``h``; the film drains until
it reaches the rupture thickness ``h_c`` (merge) or contact support is lost
first (release, a bounce). `FilmDrainageCoalescencePlan.advance` integrates one
`FilmContactLedger` of fixed capacity over one step from a
`FilmContactObservation`. The near-contact load and its work are recorded in the
same ledger slot whose drainage they drive: the force ``F`` transmitted by the
near-contact potential is the one load of the drainage law, so the film is never
driven by a second, independently counted contact force.

Axisymmetric geometry (three-dimensional pairs)
-----------------------------------------------
Equivalent radius (Abid & Chesters 1994, eq. [1]):
``1/R_eq = (1/R_1 + 1/R_2)/2``; equal drops give ``R_eq = R`` and a drop at a
free or rigid plane (``R_2 -> inf``) gives ``R_eq = 2 R_1``. The flattened film
carries the Laplace pressure ``2σ/R_eq`` over ``π a²`` (eq. [17]):
``F = π a² (2σ/R_eq)``.

- Immobile interfaces: the Reynolds (1886) plane-parallel squeeze film,
  ``dh/dt = -2F h³/(3π μ_c a⁴)``; with ``a²`` above this is
  ``dh/dt = -8π σ² h³/(3 μ_c F R_eq²)`` and the constant-force drainage time is
  ``t = (3 μ_c F R_eq²/(16π σ²)) (1/h_c² - 1/h_0²)`` (Chesters 1991).
- Partially mobile interfaces: the plane-film model of Chesters (1988, 1991),
  ``-(1/h⁺²) dh⁺/dt⁺ = k`` in ``h⁺ = h/(3a²/(2R_eq))`` and
  ``t⁺ = t/(μ_d R_eq²/(√12 σ a))`` (Abid & Chesters 1994, eqs. [21], [23]);
  ``k = 0.66`` fits the constant-force solutions of Yiantsios & Davis (1990)
  just after flattening. Dimensionally ``dh/dt = -(4√3 k σ/(3 μ_d R_eq a)) h²``.
- Fully mobile interfaces: plug extensional flow of the film. Continuity gives
  ``u = -r ḣ/(2h)``; with a traction-free rim ``σ_rr = 0`` the normal stress is
  ``σ_zz = 3 μ_c ḣ/h``, which balances the Laplace pressure ``-2σ/R_eq``:
  ``dh/dt = -2σ h/(3 μ_c R_eq)`` and ``h = h_0 exp(-2σ t/(3 μ_c R_eq))``.

Rupture thickness from the Hamaker constant ``A`` (Chesters 1991; Abid &
Chesters 1994, eq. [29a]): ``h_c = (A R_eq/(8π σ))^{1/3}``, or a declared
``h_c``.

Planar geometry (two-dimensional simulations, cylinders per unit depth)
-----------------------------------------------------------------------
A cylinder of radius ``R`` has Laplace pressure ``σ/R``, so the flattened slot
of half-length ``a`` carries ``F' = 2a σ/R_eq`` per unit depth with the same
equivalent radius.

- Immobile: the Reynolds slot. The flux ``q = -ḣ x`` through a Poiseuille gap,
  ``q = -(h³/(12 μ_c)) dp/dx``, gives ``p = 6 μ_c (-ḣ)(a² - x²)/h³`` and
  ``F' = -8 μ_c a³ ḣ/h³``; eliminating ``a`` yields
  ``dh/dt = -σ³ h³/(μ_c F'² R_eq³)``.
- Fully mobile: plug extension ``u = -x ḣ/h``; ``σ_xx = 0`` at the rim gives
  ``σ_zz = 4 μ_c ḣ/h = -σ/R_eq``, so ``dh/dt = -σ h/(4 μ_c R_eq)``.
- Partially mobile planar drainage has no primary source here and is refused.
  The rupture thickness must be declared (no planar van der Waals formula is
  claimed).

Every law is ``dh/dt = -C(F, R_eq) h^n`` with ``n = 3`` (immobile), ``2``
(partially mobile) or ``1`` (fully mobile), so ``h(t)`` and the drainage time
have closed forms for a constant force; `advance` applies the closed form
exactly over each step for the piecewise-constant observed load.

Validity (Abid & Chesters 1994, section 5) is reported, never enforced: the
small-slope ratio ``a/R_eq`` (eq. [52], the constant-force form of the gentle
collision requirement ``Ca^{1/3} << 1``, eq. [51], whose capillary number is
defined by a constant approach velocity not available here) and, for partial
mobility, the plug-flow ratio ``(μ_d/μ_c)/(4√(3k) a/h_c)`` (eq. [38]).

Sources (equations only): Abid & Chesters, Int. J. Multiphase Flow 20 (1994)
613, doi:10.1016/0301-9322(94)90033-7; Chesters, Trans. IChemE 69A (1991) 259;
Yiantsios & Davis, J. Fluid Mech. 217 (1990) 547; Reynolds, Phil. Trans. R.
Soc. 177 (1886) 157.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import assert_never, final, Literal, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState, parameter_field
from ..._validation import positive_integer
from ...sparse import align_key_groups, KeyGroupPlan
from ...typing import (
    AnyShape,
    as_array,
    as_host_array,
    Bool,
    Dim,
    Float,
    Float64,
    HostFloat64,
    Int32,
    parse,
    Scalar,
    Scope,
)


type InterfaceMobilityRegime = Literal["immobile", "partially-mobile", "fully-mobile"]
type FilmGeometry = Literal["axisymmetric", "planar"]

_DEFAULT_MOBILITY_COEFFICIENT = 0.66
_MAXIMUM_ID_UPPER_BOUND = 2**31 - 1


class _FilmPairDim(Dim, minimum=1):
    """Candidate pair slots of one film contact ledger or observation."""


class FilmContactStatus(IntEnum):
    """Lifecycle of one film contact slot.

    ``DRAINING`` pairs are in resolved contact with a thinning film. ``RELEASED``
    (contact support lost before rupture), ``MERGED`` (the film reached the
    rupture thickness) and ``FAILED`` (invalid load, radius, work or thickness)
    are the events of the step that set them; such slots are retired to
    ``INACTIVE`` at the start of the next `advance` and become reusable.
    """

    INACTIVE = 0
    DRAINING = 1
    RELEASED = 2
    MERGED = 3
    FAILED = 4


def film_equivalent_radius(first_radius: ArrayLike, second_radius: ArrayLike) -> Array:
    """Equivalent radius ``2/(1/R_1 + 1/R_2)`` (Abid & Chesters 1994, eq. [1]).

    ``second_radius = inf`` models a plane (free or rigid) and gives ``2 R_1``.
    """
    first = jnp.asarray(first_radius)
    second = jnp.asarray(second_radius)
    return 2.0 / (1.0 / first + 1.0 / second)


def _positive_scalar(value: ArrayLike, name: str, /) -> Array:
    """Validate one finite positive coefficient on the host; return a float64 leaf."""
    host = as_host_array(value, HostFloat64[Scalar], name)
    number = float(host)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return jnp.asarray(host, dtype=jnp.float64)


@final
class FilmDrainageValidity(StrictModule, NonTrainableState):
    """Evaluated validity ratios of the flattened-film drainage model.

    Each ratio must be much smaller than one for the model to apply:
    ``slope_ratio = a/R_eq`` (Abid & Chesters 1994, eq. [52]) and, for partial
    mobility only, ``plug_flow_ratio = (μ_d/μ_c) h_c/(4√(3k) a)`` (eq. [38]).
    ``validity_margin`` is the largest applicable ratio.
    """

    __strict_contract__ = True

    film_radius: Float[AnyShape]
    critical_thickness: Float[AnyShape]
    slope_ratio: Float[AnyShape]
    plug_flow_ratio: Float[AnyShape] | None
    validity_margin: Float[AnyShape]


@final
class FilmContactObservation(StrictModule, NonTrainableState):
    """Near-contact state of candidate pairs over one step (capacity ``P``).

    ``first_id < second_id`` are stable bubble identifiers. ``in_contact`` marks
    resolved contact support this step; ``load`` is the interaction force ``F``
    transmitted by the near-contact potential (per unit depth in planar
    geometry), which must be positive while in contact; ``near_contact_work`` is
    the work done by that potential on the pair this step. Slot order carries no
    identity.
    """

    __strict_contract__ = True

    first_id: Int32[_FilmPairDim]
    second_id: Int32[_FilmPairDim]
    valid: Bool[_FilmPairDim]
    in_contact: Bool[_FilmPairDim]
    load: Float[_FilmPairDim]
    equivalent_radius: Float[_FilmPairDim]
    near_contact_work: Float[_FilmPairDim]

    def __init__(
        self,
        *,
        first_id: ArrayLike,
        second_id: ArrayLike,
        valid: ArrayLike,
        in_contact: ArrayLike,
        load: ArrayLike,
        equivalent_radius: ArrayLike,
        near_contact_work: ArrayLike,
    ) -> None:
        scope = Scope()
        first = as_array(first_id, Int32[_FilmPairDim], "first_id", scope=scope)
        second = as_array(second_id, Int32[_FilmPairDim], "second_id", scope=scope)
        valid_mask = as_array(valid, Bool[_FilmPairDim], "valid", scope=scope)
        contact = as_array(in_contact, Bool[_FilmPairDim], "in_contact", scope=scope)
        force = as_array(load, Float[_FilmPairDim], "load", scope=scope)
        radius = as_array(
            equivalent_radius, Float[_FilmPairDim], "equivalent_radius", scope=scope
        )
        work = as_array(
            near_contact_work, Float[_FilmPairDim], "near_contact_work", scope=scope
        )
        if not force.dtype == radius.dtype == work.dtype:
            raise TypeError(
                "load, equivalent_radius and near_contact_work must share one dtype."
            )
        self.first_id = first
        self.second_id = second
        self.valid = valid_mask
        self.in_contact = contact
        self.load = force
        self.equivalent_radius = radius
        self.near_contact_work = work


@final
class FilmContactLedger(StrictModule, NonTrainableState):
    """Fixed-capacity film contact slots keyed by the stable pair ``(first, second)``.

    This one ledger owns both the near-contact load/work bookkeeping and the film
    drainage of each pair: ``load`` is the last observed force and
    ``near_contact_work`` the cumulative work of the near-contact potential on
    the pair, and the same force drives the drainage law, so contact work is
    never counted twice. ``drained_volume`` accumulates ``Δh`` times the
    flattened film area ``π a²`` (per unit depth ``2a`` in planar geometry) at
    each step's load, a proxy for the liquid expelled from the film. Inactive
    slots carry ids ``-1`` and zeros.
    """

    __strict_contract__ = True

    first_id: Int32[_FilmPairDim]
    second_id: Int32[_FilmPairDim]
    status: Int32[_FilmPairDim]
    film_thickness: Float[_FilmPairDim]
    contact_age: Float[_FilmPairDim]
    load: Float[_FilmPairDim]
    equivalent_radius: Float[_FilmPairDim]
    near_contact_work: Float[_FilmPairDim]
    drained_volume: Float[_FilmPairDim]

    def __init__(
        self,
        *,
        first_id: ArrayLike,
        second_id: ArrayLike,
        status: ArrayLike,
        film_thickness: ArrayLike,
        contact_age: ArrayLike,
        load: ArrayLike,
        equivalent_radius: ArrayLike,
        near_contact_work: ArrayLike,
        drained_volume: ArrayLike,
    ) -> None:
        scope = Scope()
        first = as_array(first_id, Int32[_FilmPairDim], "first_id", scope=scope)
        second = as_array(second_id, Int32[_FilmPairDim], "second_id", scope=scope)
        codes = as_array(status, Int32[_FilmPairDim], "status", scope=scope)
        floats = tuple(
            as_array(value, Float[_FilmPairDim], name, scope=scope)
            for name, value in (
                ("film_thickness", film_thickness),
                ("contact_age", contact_age),
                ("load", load),
                ("equivalent_radius", equivalent_radius),
                ("near_contact_work", near_contact_work),
                ("drained_volume", drained_volume),
            )
        )
        if len({value.dtype for value in floats}) != 1:
            raise TypeError("Film contact ledger floating fields must share one dtype.")
        self.first_id = first
        self.second_id = second
        self.status = codes
        (
            self.film_thickness,
            self.contact_age,
            self.load,
            self.equivalent_radius,
            self.near_contact_work,
            self.drained_volume,
        ) = floats

    @classmethod
    def empty(cls, capacity: int, dtype: DTypeLike) -> FilmContactLedger:
        """An all-``INACTIVE`` ledger of ``capacity`` slots with float ``dtype``."""
        slots = positive_integer(capacity, "capacity")
        float_dtype = np.dtype(dtype)
        if not np.issubdtype(float_dtype, np.floating):
            raise TypeError("dtype must be a floating dtype.")
        ids = jnp.full((slots,), -1, dtype=jnp.int32)
        zeros = jnp.zeros((slots,), dtype=float_dtype)
        return cls(
            first_id=ids,
            second_id=ids,
            status=jnp.full((slots,), FilmContactStatus.INACTIVE.value, dtype=jnp.int32),
            film_thickness=zeros,
            contact_age=zeros,
            load=zeros,
            equivalent_radius=zeros,
            near_contact_work=zeros,
            drained_volume=zeros,
        )


@final
class FilmContactUpdate(StrictModule, NonTrainableState):
    """One `advance` of the film contact ledger.

    Per ledger slot: ``merge_proposals`` (the film ruptured this step, at
    ``merge_time`` within the step; ``merge_time`` is zero elsewhere),
    ``release_events`` (contact support lost before rupture), ``failed`` (invalid
    inputs or thickness), ``drained`` (the film drained this step) and
    ``validity`` evaluated at the slot's load and equivalent radius (meaningful
    where ``drained``). Per observation slot: ``observation_slots`` (ledger slot
    that consumed it, ``-1`` otherwise) and ``refused_observations`` (a new
    in-contact pair beyond the free capacity). ``untracked_near_contact_work``
    sums the work of valid observations no slot consumed. ``successful`` requires
    a positive finite step, no capacity overflow, no invalid or duplicate
    observations and no failed slot.
    """

    __strict_contract__ = True

    ledger: FilmContactLedger
    merge_proposals: Bool[_FilmPairDim]
    release_events: Bool[_FilmPairDim]
    failed: Bool[_FilmPairDim]
    merge_time: Float[_FilmPairDim]
    drained: Bool[_FilmPairDim]
    validity: FilmDrainageValidity
    observation_slots: Int32[_FilmPairDim]
    refused_observations: Bool[_FilmPairDim]
    untracked_near_contact_work: Float[Scalar]
    invalid_observations: Int32[Scalar]
    duplicate_observations: Int32[Scalar]
    capacity_overflow: Bool[Scalar]
    successful: Bool[Scalar]
    plan_id: str = eqx.field(static=True)


class _SlotMatch(NamedTuple):
    """Pair-key alignment of observations with ledger slots for one step."""

    observation_index: Array
    observed: Array
    incoming: Array
    observation_slots: Array
    refused: Array
    canonical: Array


@final
class FilmDrainageCoalescencePlan(StrictModule):
    """Film drainage between resolved pairs, closed-form per piecewise-constant load.

    ``continuous_viscosity`` μ_c, ``dispersed_viscosity`` μ_d (partially mobile
    only), ``surface_tension`` σ, the rupture model (exactly one of
    ``hamaker_constant`` A, axisymmetric only, or a declared
    ``critical_thickness``), ``initial_film_thickness`` h_0 and
    ``mobility_coefficient`` k (partially mobile only, default 0.66 from the fit
    of Abid & Chesters 1994 to Yiantsios & Davis 1990) are inferable leaves.
    ``pair_capacity`` fixes the ledger and observation extent; stable bubble ids
    lie in ``[0, id_upper_bound]``. See the module docstring for the laws.
    """

    __strict_contract__ = True

    continuous_viscosity: Float64[Scalar] = parameter_field()
    dispersed_viscosity: Float64[Scalar] | None = parameter_field()
    surface_tension: Float64[Scalar] = parameter_field()
    hamaker_constant: Float64[Scalar] | None = parameter_field()
    declared_critical_thickness: Float64[Scalar] | None = parameter_field()
    initial_film_thickness: Float64[Scalar] = parameter_field()
    mobility_coefficient: Float64[Scalar] | None = parameter_field()
    regime: InterfaceMobilityRegime = eqx.field(static=True)
    geometry: FilmGeometry = eqx.field(static=True)
    pair_capacity: int = eqx.field(static=True)
    id_upper_bound: int = eqx.field(static=True)
    pair_groups: KeyGroupPlan = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        regime: InterfaceMobilityRegime,
        geometry: FilmGeometry,
        pair_capacity: int,
        id_upper_bound: int,
        continuous_viscosity: ArrayLike,
        surface_tension: ArrayLike,
        initial_film_thickness: ArrayLike,
        dispersed_viscosity: ArrayLike | None = None,
        hamaker_constant: ArrayLike | None = None,
        critical_thickness: ArrayLike | None = None,
        mobility_coefficient: ArrayLike | None = None,
    ) -> None:
        regime_ = parse(regime, InterfaceMobilityRegime, "regime")
        geometry_ = parse(geometry, FilmGeometry, "geometry")
        capacity = positive_integer(pair_capacity, "pair_capacity")
        bound = positive_integer(id_upper_bound, "id_upper_bound")
        if bound > _MAXIMUM_ID_UPPER_BOUND:
            raise ValueError("id_upper_bound must fit the int32 bubble identifiers.")
        partial = regime_ == "partially-mobile"
        if partial and geometry_ == "planar":
            raise ValueError(
                "Partially mobile planar film drainage has no supported law; use "
                "the immobile or fully mobile planar regime."
            )
        if (hamaker_constant is None) == (critical_thickness is None):
            raise ValueError(
                "Declare exactly one of hamaker_constant and critical_thickness."
            )
        if hamaker_constant is not None and geometry_ == "planar":
            raise ValueError(
                "Planar films need a declared critical_thickness; the Hamaker "
                "rupture thickness is axisymmetric."
            )
        if partial and dispersed_viscosity is None:
            raise ValueError("Partially mobile drainage needs dispersed_viscosity.")
        if not partial and (
            dispersed_viscosity is not None or mobility_coefficient is not None
        ):
            raise ValueError(
                "dispersed_viscosity and mobility_coefficient only enter the "
                "partially mobile law."
            )
        mu_c = _positive_scalar(continuous_viscosity, "continuous_viscosity")
        sigma = _positive_scalar(surface_tension, "surface_tension")
        initial = _positive_scalar(initial_film_thickness, "initial_film_thickness")
        mu_d = (
            None
            if dispersed_viscosity is None
            else _positive_scalar(dispersed_viscosity, "dispersed_viscosity")
        )
        hamaker = (
            None
            if hamaker_constant is None
            else _positive_scalar(hamaker_constant, "hamaker_constant")
        )
        critical = (
            None
            if critical_thickness is None
            else _positive_scalar(critical_thickness, "critical_thickness")
        )
        if critical is not None and float(critical) >= float(initial):
            raise ValueError(
                "critical_thickness must be smaller than initial_film_thickness."
            )
        mobility = (
            None
            if not partial
            else _positive_scalar(
                _DEFAULT_MOBILITY_COEFFICIENT
                if mobility_coefficient is None
                else mobility_coefficient,
                "mobility_coefficient",
            )
        )
        groups = KeyGroupPlan(capacity, capacity, bound * (bound + 2))
        self.continuous_viscosity = mu_c
        self.dispersed_viscosity = mu_d
        self.surface_tension = sigma
        self.hamaker_constant = hamaker
        self.declared_critical_thickness = critical
        self.initial_film_thickness = initial
        self.mobility_coefficient = mobility
        self.regime = regime_
        self.geometry = geometry_
        self.pair_capacity = capacity
        self.id_upper_bound = bound
        self.pair_groups = groups
        self.plan_id = canonical_fingerprint(
            {
                "kind": "film-drainage-coalescence",
                "regime": regime_,
                "geometry": geometry_,
                "pair_capacity": capacity,
                "id_upper_bound": bound,
                "rupture": "hamaker" if hamaker is not None else "declared",
            }
        )

    def film_radius(self, force: ArrayLike, equivalent_radius: ArrayLike) -> Array:
        """Flattened film radius ``a`` (half-length per unit depth in planar)."""
        load = jnp.asarray(force)
        radius = jnp.asarray(equivalent_radius)
        sigma = self.surface_tension
        match self.geometry:
            case "axisymmetric":
                return jnp.sqrt(load * radius / (2.0 * jnp.pi * sigma))
            case "planar":
                return load * radius / (2.0 * sigma)
            case unreachable:
                assert_never(unreachable)

    def critical_thickness(self, equivalent_radius: ArrayLike) -> Array:
        """Rupture thickness ``(A R_eq/(8π σ))^{1/3}`` or the declared ``h_c``."""
        radius = jnp.asarray(equivalent_radius)
        if self.hamaker_constant is not None:
            return jnp.cbrt(
                self.hamaker_constant * radius / (8.0 * jnp.pi * self.surface_tension)
            )
        if self.declared_critical_thickness is None:
            raise ValueError("The plan declares no rupture thickness model.")
        return jnp.broadcast_to(self.declared_critical_thickness, radius.shape)

    def _rate_coefficient(self, force: Array, equivalent_radius: Array) -> Array:
        """``C`` of ``dh/dt = -C h^n`` at a constant force."""
        sigma = self.surface_tension
        mu_c = self.continuous_viscosity
        radius = equivalent_radius
        match self.regime:
            case "immobile":
                match self.geometry:
                    case "axisymmetric":
                        return 8.0 * jnp.pi * sigma**2 / (3.0 * mu_c * force * radius**2)
                    case "planar":
                        return sigma**3 / (mu_c * force**2 * radius**3)
                    case unreachable:
                        assert_never(unreachable)
            case "partially-mobile":
                if self.dispersed_viscosity is None or self.mobility_coefficient is None:
                    raise ValueError(
                        "Partially mobile drainage needs dispersed_viscosity and "
                        "mobility_coefficient."
                    )
                match self.geometry:
                    case "axisymmetric":
                        film = self.film_radius(force, radius)
                        return (
                            4.0
                            * math.sqrt(3.0)
                            * self.mobility_coefficient
                            * sigma
                            / (3.0 * self.dispersed_viscosity * radius * film)
                        )
                    case "planar":
                        raise ValueError("Partially mobile planar drainage is refused.")
                    case unreachable:
                        assert_never(unreachable)
            case "fully-mobile":
                match self.geometry:
                    case "axisymmetric":
                        return 2.0 * sigma / (3.0 * mu_c * radius)
                    case "planar":
                        return sigma / (4.0 * mu_c * radius)
                    case unreachable:
                        assert_never(unreachable)
            case unreachable:
                assert_never(unreachable)

    def drainage_rate(
        self, thickness: ArrayLike, force: ArrayLike, equivalent_radius: ArrayLike
    ) -> Array:
        """Film thinning rate ``dh/dt`` (negative) at thickness ``h``."""
        h = jnp.asarray(thickness)
        coefficient = self._rate_coefficient(
            jnp.asarray(force), jnp.asarray(equivalent_radius)
        )
        match self.regime:
            case "immobile":
                return -coefficient * h**3
            case "partially-mobile":
                return -coefficient * h**2
            case "fully-mobile":
                return -coefficient * h
            case unreachable:
                assert_never(unreachable)

    def thickness_after(
        self,
        thickness: ArrayLike,
        elapsed: ArrayLike,
        force: ArrayLike,
        equivalent_radius: ArrayLike,
    ) -> Array:
        """Closed-form thickness after ``elapsed`` time at a constant force."""
        h = jnp.asarray(thickness)
        t = jnp.asarray(elapsed)
        coefficient = self._rate_coefficient(
            jnp.asarray(force), jnp.asarray(equivalent_radius)
        )
        match self.regime:
            case "immobile":
                return h / jnp.sqrt(1.0 + 2.0 * coefficient * h**2 * t)
            case "partially-mobile":
                return h / (1.0 + coefficient * h * t)
            case "fully-mobile":
                return h * jnp.exp(-coefficient * t)
            case unreachable:
                assert_never(unreachable)

    def drainage_time(
        self,
        initial: ArrayLike,
        final: ArrayLike,
        force: ArrayLike,
        equivalent_radius: ArrayLike,
    ) -> Array:
        """Closed-form time to thin from ``initial`` to ``final`` at a constant force.

        Negative when ``final`` exceeds ``initial``.
        """
        h0 = jnp.asarray(initial)
        h1 = jnp.asarray(final)
        coefficient = self._rate_coefficient(
            jnp.asarray(force), jnp.asarray(equivalent_radius)
        )
        match self.regime:
            case "immobile":
                return (1.0 / h1**2 - 1.0 / h0**2) / (2.0 * coefficient)
            case "partially-mobile":
                return (1.0 / h1 - 1.0 / h0) / coefficient
            case "fully-mobile":
                return jnp.log(h0 / h1) / coefficient
            case unreachable:
                assert_never(unreachable)

    def validity(
        self, force: ArrayLike, equivalent_radius: ArrayLike
    ) -> FilmDrainageValidity:
        """Evaluate the small-slope and (partial mobility) plug-flow ratios."""
        load = jnp.asarray(force)
        radius = jnp.asarray(equivalent_radius)
        film = self.film_radius(load, radius)
        critical = self.critical_thickness(radius)
        slope = film / radius
        plug = None
        margin = slope
        if self.dispersed_viscosity is not None and self.mobility_coefficient is not None:
            plug = (
                self.dispersed_viscosity
                / self.continuous_viscosity
                * critical
                / (4.0 * jnp.sqrt(3.0 * self.mobility_coefficient) * film)
            )
            margin = jnp.maximum(slope, plug)
        return FilmDrainageValidity(
            film_radius=film,
            critical_thickness=critical,
            slope_ratio=slope,
            plug_flow_ratio=plug,
            validity_margin=margin,
        )

    def _film_area(self, film_radius: Array, /) -> Array:
        """Flattened film area ``π a²`` (per unit depth ``2a`` in planar)."""
        match self.geometry:
            case "axisymmetric":
                return jnp.pi * film_radius**2
            case "planar":
                return 2.0 * film_radius
            case unreachable:
                assert_never(unreachable)

    def _pair_keys(self, first: Array, second: Array, /) -> Array:
        return first.astype(jnp.int64) * (self.id_upper_bound + 1) + second.astype(
            jnp.int64
        )

    def _match(
        self,
        ledger: FilmContactLedger,
        observation: FilmContactObservation,
        draining: Array,
        observation_active: Array,
        /,
    ) -> _SlotMatch:
        """Align observations with draining slots by pair key; place new pairs.

        New in-contact pairs take free slots in ascending slot order, ranked by
        canonical pair-key order, so placement and refusal do not depend on the
        observation slot order. Duplicate keys keep their first observation.
        """
        capacity = self.pair_capacity
        ledger_groups = self.pair_groups.build(
            self._pair_keys(ledger.first_id, ledger.second_id), draining
        )
        observed_groups = self.pair_groups.build(
            self._pair_keys(observation.first_id, observation.second_id),
            observation_active,
        )
        transition = align_key_groups(ledger_groups, observed_groups)

        ledger_group = jnp.clip(ledger_groups.item_group_slots, 0)
        observed_group = transition.previous_to_candidate[ledger_group]
        observed = draining & transition.previous_retained[ledger_group]
        tracked_index = observed_groups.storage_to_logical[
            observed_groups.group_starts[observed_group]
        ]

        item_group = jnp.clip(observed_groups.item_group_slots, 0)
        canonical = observation_active & (
            observed_groups.logical_to_storage == observed_groups.group_starts[item_group]
        )
        tracked = canonical & transition.candidate_retained[item_group]
        tracked_slot = ledger_groups.storage_to_logical[
            ledger_groups.group_starts[transition.candidate_to_previous[item_group]]
        ]

        new = canonical & observation.in_contact & ~tracked
        new_sorted = new[observed_groups.storage_to_logical]
        new_rank = (jnp.cumsum(new_sorted, dtype=jnp.int32) - 1)[
            observed_groups.logical_to_storage
        ]
        free = ~draining
        free_count = jnp.sum(free, dtype=jnp.int32)
        assigned = new & (new_rank < free_count)
        free_slots = jnp.nonzero(free, size=capacity, fill_value=0)[0].astype(jnp.int32)
        new_slot = free_slots[jnp.clip(new_rank, 0, capacity - 1)]

        free_rank = jnp.cumsum(free, dtype=jnp.int32) - 1
        incoming = free & (free_rank < jnp.sum(new, dtype=jnp.int32))
        new_positions = jnp.nonzero(new_sorted, size=capacity, fill_value=0)[0]
        incoming_index = observed_groups.storage_to_logical[
            new_positions[jnp.clip(free_rank, 0, capacity - 1)]
        ]
        observation_slots = jnp.where(
            tracked, tracked_slot, jnp.where(assigned, new_slot, -1)
        ).astype(jnp.int32)
        return _SlotMatch(
            observation_index=jnp.where(draining, tracked_index, incoming_index),
            observed=observed | incoming,
            incoming=incoming,
            observation_slots=observation_slots,
            refused=new & ~assigned,
            canonical=canonical,
        )

    def advance(
        self,
        ledger: FilmContactLedger,
        observation: FilmContactObservation,
        step_size: ArrayLike,
    ) -> FilmContactUpdate:
        """Retire last step's events, match pairs by key and drain over one step.

        Draining pairs still in contact thin exactly under the observed load; a
        pair whose thickness reaches ``h_c`` within the step merges at the
        closed-form crossing time; draining pairs without contact support are
        released; new in-contact pairs start at ``initial_film_thickness`` and
        drain over this step. Fixed shapes, no host synchronization.
        """
        if not isinstance(ledger, FilmContactLedger):
            raise TypeError("ledger must be a FilmContactLedger.")
        if not isinstance(observation, FilmContactObservation):
            raise TypeError("observation must be a FilmContactObservation.")
        capacity = self.pair_capacity
        if ledger.status.shape != (capacity,):
            raise ValueError(f"ledger must have {capacity} slots.")
        if observation.valid.shape != (capacity,):
            raise ValueError(f"observation must have {capacity} slots.")
        dtype = ledger.film_thickness.dtype
        dt = jnp.asarray(step_size, dtype=dtype)
        if dt.shape != ():
            raise ValueError("step_size must be a scalar.")

        draining = ledger.status == FilmContactStatus.DRAINING.value
        ids_ok = (
            (observation.first_id >= 0)
            & (observation.first_id < observation.second_id)
            & (observation.second_id <= self.id_upper_bound)
        )
        observation_active = observation.valid & ids_ok
        slots = self._match(ledger, observation, draining, observation_active)
        index = slots.observation_index
        observed = slots.observed
        incoming = slots.incoming
        in_contact = observed & observation.in_contact[index]
        force = observation.load[index].astype(dtype)
        radius = observation.equivalent_radius[index].astype(dtype)
        work = observation.near_contact_work[index].astype(dtype)

        integrating = in_contact
        lost = draining & ~in_contact
        drain_inputs = (
            jnp.isfinite(force) & (force > 0) & jnp.isfinite(radius) & (radius > 0)
        )
        safe_force = jnp.where(integrating & drain_inputs, force, 1.0)
        safe_radius = jnp.where(integrating & drain_inputs, radius, 1.0)
        start = jnp.where(
            incoming, self.initial_film_thickness.astype(dtype), ledger.film_thickness
        )
        safe_start = jnp.where(integrating, start, 1.0)
        critical = self.critical_thickness(safe_radius).astype(dtype)
        end = self.thickness_after(safe_start, dt, safe_force, safe_radius).astype(dtype)
        crossing = self.drainage_time(safe_start, critical, safe_force, safe_radius)
        failed = (observed & ~jnp.isfinite(work)) | (
            integrating & ~(drain_inputs & jnp.isfinite(end) & jnp.isfinite(crossing))
        )
        merged = integrating & ~failed & (end <= critical)
        drained = integrating & ~failed
        merge_time = jnp.where(merged, jnp.clip(crossing, 0.0, dt), 0.0).astype(dtype)
        thickness = jnp.where(merged, critical, end)
        film = self.film_radius(safe_force, safe_radius)
        area = self._film_area(film)

        occupied = draining | incoming
        status = jnp.where(
            failed,
            FilmContactStatus.FAILED.value,
            jnp.where(
                merged,
                FilmContactStatus.MERGED.value,
                jnp.where(
                    drained,
                    FilmContactStatus.DRAINING.value,
                    jnp.where(
                        lost,
                        FilmContactStatus.RELEASED.value,
                        FilmContactStatus.INACTIVE.value,
                    ),
                ),
            ),
        ).astype(jnp.int32)
        zero = jnp.zeros((capacity,), dtype=dtype)
        age = jnp.where(incoming, 0.0, ledger.contact_age) + jnp.where(
            merged, merge_time, jnp.where(drained, dt, 0.0)
        )
        new_ledger = FilmContactLedger(
            first_id=jnp.where(
                incoming,
                observation.first_id[index],
                jnp.where(draining, ledger.first_id, -1),
            ).astype(jnp.int32),
            second_id=jnp.where(
                incoming,
                observation.second_id[index],
                jnp.where(draining, ledger.second_id, -1),
            ).astype(jnp.int32),
            status=status,
            film_thickness=jnp.where(
                occupied, jnp.where(drained, thickness, start), zero
            ).astype(dtype),
            contact_age=jnp.where(occupied, age, zero).astype(dtype),
            load=jnp.where(
                observed, force, jnp.where(draining, ledger.load, zero)
            ).astype(dtype),
            equivalent_radius=jnp.where(
                observed, radius, jnp.where(draining, ledger.equivalent_radius, zero)
            ).astype(dtype),
            near_contact_work=(
                jnp.where(draining, ledger.near_contact_work, zero)
                + jnp.where(observed, work, zero)
            ).astype(dtype),
            drained_volume=(
                jnp.where(draining, ledger.drained_volume, zero)
                + jnp.where(drained, area * (safe_start - thickness), zero)
            ).astype(dtype),
        )

        consumed = slots.observation_slots >= 0
        untracked = jnp.sum(
            jnp.where(
                observation.valid & ~consumed,
                observation.near_contact_work.astype(dtype),
                0.0,
            )
        )
        invalid = jnp.sum(observation.valid & ~ids_ok, dtype=jnp.int32)
        duplicates = jnp.sum(observation_active & ~slots.canonical, dtype=jnp.int32)
        overflow = jnp.any(slots.refused)
        successful = (
            jnp.isfinite(dt)
            & (dt > 0)
            & ~overflow
            & (invalid == 0)
            & (duplicates == 0)
            & ~jnp.any(failed)
        )
        return FilmContactUpdate(
            ledger=new_ledger,
            merge_proposals=merged,
            release_events=lost & ~failed,
            failed=failed,
            merge_time=merge_time,
            drained=drained,
            validity=self.validity(safe_force, safe_radius),
            observation_slots=slots.observation_slots,
            refused_observations=slots.refused,
            untracked_near_contact_work=untracked,
            invalid_observations=invalid,
            duplicate_observations=duplicates,
            capacity_overflow=overflow,
            successful=successful,
            plan_id=self.plan_id,
        )


__all__ = [
    "FilmContactLedger",
    "FilmContactObservation",
    "FilmContactStatus",
    "FilmContactUpdate",
    "FilmDrainageCoalescencePlan",
    "FilmDrainageValidity",
    "FilmGeometry",
    "InterfaceMobilityRegime",
    "film_equivalent_radius",
]
