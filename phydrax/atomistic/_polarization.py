#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Induced-dipole polarization through native multipole actions and linear solves.

Physical definition
-------------------
Sites carry permanent primitive Cartesian multipoles (``q``, ``mu``, traceful
``Theta`` in the ``1/2 Theta : grad grad`` convention) and isotropic
polarizabilities ``alpha`` in system units (charge^2 length^2 / energy; a
polarizability volume ``alpha_v`` converts as ``alpha = alpha_v / k``).
Induced dipoles ``u`` on polarizable sites (active, ``alpha > 0``) minimise the
Hylleraas functional

``E(u) = 1/2 sum_i u_i . u_i / alpha_i + 1/2 B_u(u, u) + B_p(u, M)``

where ``B_x`` is the multipole bilinear interaction form of
:mod:`~phydrax.atomistic._multipole_electrostatics` with pair scale column
``x`` (``"polarization"`` for permanent-induced, ``"mutual"`` for induced-
induced; ``"direct"`` only defines the direct-field predictor guess). Its
stationarity condition is the linear system ``(alpha^-1 - T_u) u = E_p``.

``"point"`` sites use ``1/R`` kernels; ``damping="exponential-cubic"``
multiplies every bare pair tensor by ``1 - exp(-sqrt(a_i a_j) R^3)`` (the
uniform damping of the previous owner; it is not AMOEBA Thole damping).
``"gaussian"`` sites use the Gaussian ``erf`` kernels and no further damping.
Periodic routes add the Ewald/PME reciprocal and intrasite (``"exclude"``)
terms; damping and pair scales act on the real-space (respectively current
minimum-image) member only, through explicit correction routes.

Numerical methods (distinct identities)
---------------------------------------
* ``PCG``: native :func:`phydrax.linalg.solve` with
  :class:`~phydrax.linalg.PCG`, absolute residual tolerance, warm starts and
  implicit (``"mathematical"``) differentiation. The returned energy is the
  functional at the solution, differentiated with the implicit induced-dipole
  response, i.e. the stationary functional; forces are qualified only when
  the true stationary residual meets ``force_tolerance``.
* ``TCG``: exactly ``tcg_order`` CG steps from a deterministic guess (direct
  field or zero, never history) with algorithmic differentiation. Its energy
  is the TCG functional and its forces are that functional's exact
  derivatives; it is never reported converged or stationary unless its
  measured residual says so.

Both gate on finite actions, native solve status (no non-happy breakdown) and
the declared :class:`PolarizationStabilityPolicy`: a matrix-free scaled
block-Gershgorin proof (nonperiodic), a bounded dense-spectrum reference, or an
explicitly reported positive-definiteness assumption. Without proof or a
declared assumption no state, energy or force is published. ``u . E_p > 0``
and the optional Lanczos Ritz value are detections only: a nonpositive
stationary ``u . E_p`` or a negative Ritz value refuses, positive values prove
nothing (an unstable mode orthogonal to ``E_p`` leaves ``u . E_p > 0``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, TypeAlias, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_finite_float
from ..discretization import ParticleDiscretization, PeriodicCell
from ..linalg import (
    ArraySpace,
    DiagonalPreconditioner,
    DifferentiationPolicy,
    FailurePolicy,
    FunctionLinearOperator,
    LinearDerivativeSolvePolicy,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    MaterializationPolicy,
    OperatorProperties,
    PCG,
    PreconditioningPolicy,
    solve,
    TolerancePolicy,
)
from ..linalg.eigen import (
    DenseEigh,
    Eigenproblem,
    eigensolve,
    EigenSolvePolicy,
    RestartedLanczos,
)
from ..typing import parse
from ._conditioning import (
    AtomisticConditionLevel,
    AtomisticFieldReference,
    AtomisticObservable,
)
from ._multipole_basis import (
    CartesianMultipoles,
    dipole_multipoles,
    GaussianMultipoleWidths,
    unique_second_moments,
    unpack_multipoles,
)
from ._multipole_electrostatics import (
    _inverse_power,
    _PairSites,
    _ScaleColumn,
    _unit_factor,
    AbstractMultipolePairs,
    AbstractMultipoleRadialKernel,
    CoulombKernel,
    EwaldMultipoleRoute,
    ExplicitImagePairs,
    FiniteMultipoleRoute,
    GaussianKernel,
    MultipoleElectrostaticsDefinition,
    MultipoleElectrostaticsPlan,
    MultipoleMomentFields,
    MultipolePairScales,
    MultipoleRoute,
    MultipoleSiteSet,
    NeighborhoodPairs,
    ParticleMeshMultipoleRoute,
    PreparedMultipoleElectrostatics,
    WeightedPairCorrections,
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
from ._system import PreparedAtomisticSystem


_EnumT = TypeVar("_EnumT", bound=StrEnum)
PolarizationSiteKind: TypeAlias = Literal["point", "gaussian"]
PolarizationDamping: TypeAlias = Literal["exponential-cubic", "none"]
_SCALE_COLUMNS = ("direct", "polarization", "mutual")


def _enum_value(value: object, enum_type: type[_EnumT], name: str, /) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{name} must be {enum_type.__name__} or str.")
    matches = tuple(member for member in enum_type if member.value == value)
    if not matches:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{name} must be one of: {choices}.")
    return matches[0]


class PermanentMultipoleSiteData(StrictModule, NonTrainableState):
    """Fixed-capacity primitive Cartesian multipoles and isotropic polarizabilities.

    ``quadrupoles`` are the raw traceful second moments of the
    ``1/2 Theta : grad grad`` convention (no Buckingham factor, trace kept),
    site-centred in the system Cartesian frame. ``polarizabilities`` are in
    system units (charge^2 length^2 / energy). ``damping`` holds the
    exponential-cubic damping parameters ``a_i`` (used only by
    ``damping="exponential-cubic"``).

    Every array is a dynamic leaf; the exact canonical content digest is the
    dynamic ``multipole_identity`` (``multipole_id`` decodes it on the host).
    """

    charges: Array
    dipoles: Array
    quadrupoles: Array
    polarizabilities: Array
    damping: Array
    multipole_identity: IdentityWords

    def __init__(
        self,
        charges: ArrayLike,
        dipoles: ArrayLike,
        quadrupoles: ArrayLike,
        polarizabilities: ArrayLike,
        damping: ArrayLike,
        /,
    ) -> None:
        charge = np.asarray(charges, dtype=np.float64)
        dipole = np.asarray(dipoles, dtype=np.float64)
        quadrupole = np.asarray(quadrupoles, dtype=np.float64)
        polar = np.asarray(polarizabilities, dtype=np.float64)
        damp = np.asarray(damping, dtype=np.float64)
        count = charge.size
        if (
            count == 0
            or charge.shape != (count,)
            or dipole.shape != (count, 3)
            or quadrupole.shape != (count, 3, 3)
            or polar.shape != (count,)
            or damp.shape != (count,)
        ):
            raise ValueError("Multipole arrays have incompatible fixed-capacity shapes.")
        if (
            np.any(~np.isfinite(charge))
            or np.any(~np.isfinite(dipole))
            or np.any(~np.isfinite(quadrupole))
            or np.any(~np.isfinite(polar))
            or np.any(~np.isfinite(damp))
            or np.any(polar < 0.0)
            or np.any(damp <= 0.0)
            or not np.allclose(quadrupole, np.swapaxes(quadrupole, -1, -2))
        ):
            raise ValueError(
                "Multipoles must be finite, quadrupoles symmetric, polarizabilities nonnegative, and damping positive."
            )
        (
            self.charges,
            self.dipoles,
            self.quadrupoles,
            self.polarizabilities,
            self.damping,
        ) = (jnp.asarray(value) for value in (charge, dipole, quadrupole, polar, damp))
        self.multipole_identity = IdentityWords.of(
            {
                "kind": "permanent-multipoles",
                "moments": "site-centred-primitive-half-theta-grad-grad",
                "arrays": array_tree_fingerprint(
                    {
                        "q": charge,
                        "mu": dipole,
                        "Q": quadrupole,
                        "alpha": polar,
                        "damping": damp,
                    }
                ),
            }
        )

    @property
    def multipole_id(self) -> str:
        """Host decoding of the canonical multipole content digest."""
        return self.multipole_identity.fingerprint()

    @property
    def site_capacity(self) -> int:
        return self.charges.shape[0]

    def moments(self, dtype: jnp.dtype | None = None, /) -> CartesianMultipoles:
        """Canonical degree-2 primitive moments."""
        kind = self.charges.dtype if dtype is None else dtype
        return CartesianMultipoles(
            self.charges.astype(kind),
            self.dipoles.astype(kind),
            unique_second_moments(self.quadrupoles.astype(kind)),
            maximum_degree=2,
        )

    def selected(self, indices: ArrayLike, /) -> "PermanentMultipoleSiteData":
        """The site data of ``indices`` (host integer indices)."""
        index = np.asarray(indices)
        if index.ndim != 1 or index.dtype.kind not in "iu" or index.size == 0:
            raise ValueError("indices must be a non-empty integer array.")
        return PermanentMultipoleSiteData(
            np.asarray(self.charges)[index],
            np.asarray(self.dipoles)[index],
            np.asarray(self.quadrupoles)[index],
            np.asarray(self.polarizabilities)[index],
            np.asarray(self.damping)[index],
        )


class PolarizationScaleData(StrictModule, NonTrainableState):
    """Sparse d-, p- and u-field pair scales (every unlisted pair scales by 1).

    ``direct`` scales the permanent field of the direct-field predictor,
    ``polarization`` the permanent field of the right-hand side and
    ``mutual`` the induced-dipole coupling. Each unordered pair is listed once;
    no dense ``N x N`` table exists. The exact pair-scale digest is the dynamic
    ``scales.scales_identity`` (``scale_identity``; ``scale_id`` decodes it on
    the host).
    """

    scales: MultipolePairScales

    def __init__(
        self,
        pairs: ArrayLike,
        direct: ArrayLike,
        polarization: ArrayLike,
        mutual: ArrayLike,
        /,
        *,
        site_capacity: int,
    ) -> None:
        values = tuple(
            np.asarray(value, dtype=np.float64).reshape((-1,))
            for value in (direct, polarization, mutual)
        )
        if any(np.any(value > 1.0) for value in values):
            raise ValueError("Polarization pair scales must lie in [0, 1].")
        scales = MultipolePairScales(
            pairs,
            np.stack(values, axis=-1),
            columns=_SCALE_COLUMNS,
            site_capacity=site_capacity,
        )
        self.scales = scales

    @property
    def scale_identity(self) -> IdentityWords:
        """The exact pair-scale digest words."""
        return self.scales.scales_identity

    @property
    def scale_id(self) -> str:
        """Host decoding of the canonical pair-scale digest."""
        return self.scales.scales_id

    @classmethod
    def unscaled(cls, site_capacity: int, /) -> "PolarizationScaleData":
        empty = np.zeros((0,), dtype=np.float64)
        return cls(
            np.zeros((0, 2), dtype=np.int64),
            empty,
            empty,
            empty,
            site_capacity=site_capacity,
        )

    @property
    def site_capacity(self) -> int:
        return self.scales.site_capacity


class PolarizationSolverKind(StrEnum):
    """Converged stationary PCG versus fixed-order truncated CG."""

    PCG = "pcg"
    TCG = "tcg"


class PolarizationPreconditionerKind(StrEnum):
    """Local diagonal preconditioner choices."""

    IDENTITY = "identity"
    POLARIZABILITY = "polarizability"


class ExponentialDampedCoulombKernel(AbstractMultipoleRadialKernel):
    """``lambda(R) B_n[1/R]`` with ``lambda = 1 - exp(-sqrt(a_i a_j) R^3)``.

    ``correction=True`` supplies ``(lambda - 1) B_n[1/R]``, the short-range
    member added to an Ewald split. Each Cartesian tensor order is scaled by
    the same ``lambda``, which is the declared polarization damping model.
    """

    correction: bool = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(self, *, correction: bool) -> None:
        self.correction = bool(correction)
        self.kernel_id = (
            "exponential-cubic-damping-correction"
            if self.correction
            else "exponential-cubic-damped-coulomb"
        )

    def coefficients(
        self, distance: Array, left: _PairSites, right: _PairSites, order: int, /
    ) -> tuple[Array, ...]:
        exponent = jnp.sqrt(left.damping * right.damping) * distance**3
        factor = -jnp.exp(-exponent) if self.correction else -jnp.expm1(-exponent)
        return tuple(factor * value for value in _inverse_power(distance, order))


class PolarizationOperatorPlan(StrictModule, NonTrainableState):
    """Definition and route of the matrix-free ``alpha^-1 - T`` operator."""

    coulomb_constant: float = eqx.field(static=True)
    site_kind: PolarizationSiteKind = eqx.field(static=True)
    damping: PolarizationDamping = eqx.field(static=True)
    minimum_distance: float = eqx.field(static=True)
    route: MultipoleRoute
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        coulomb_constant: float,
        site_kind: PolarizationSiteKind = "point",
        damping: PolarizationDamping = "exponential-cubic",
        route: MultipoleRoute | None = None,
        minimum_distance: float = 1.0e-8,
    ) -> None:
        constant = positive_finite_float(coulomb_constant, "coulomb_constant")
        kind = parse(site_kind, PolarizationSiteKind, "site_kind")
        damping_ = parse(damping, PolarizationDamping, "damping")
        if kind == "gaussian" and damping_ != "none":
            raise ValueError("Gaussian polarization sites take damping='none'.")
        route_ = FiniteMultipoleRoute() if route is None else route
        if not isinstance(
            route_,
            (FiniteMultipoleRoute, EwaldMultipoleRoute, ParticleMeshMultipoleRoute),
        ):
            raise TypeError("route must be a finite, Ewald or particle-mesh route.")
        distance = positive_finite_float(minimum_distance, "minimum_distance")
        self.coulomb_constant = constant
        self.site_kind = kind
        self.damping = damping_
        self.minimum_distance = distance
        self.route = route_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarization-operator-plan",
                "coulomb_constant": float(constant).hex(),
                "site_kind": kind,
                "damping": damping_,
                "minimum_distance": float(distance).hex(),
                "route": route_.route_id,
            }
        )

    def electrostatics(self) -> MultipoleElectrostaticsPlan:
        return MultipoleElectrostaticsPlan(
            MultipoleElectrostaticsDefinition(
                self.site_kind,
                2,
                coulomb_constant=self.coulomb_constant,
                intrasite="exclude",
                neutrality="uniform-background",
                minimum_distance=self.minimum_distance,
            ),
            self.route,
        )

    def prepare(
        self,
        multipoles: PermanentMultipoleSiteData,
        /,
        *,
        scaling: PolarizationScaleData | None = None,
        active_mask: ArrayLike | None = None,
        widths: GaussianMultipoleWidths | None = None,
        cell: PeriodicCell | None = None,
        particles: ParticleDiscretization | None = None,
    ) -> "PreparedPolarizationOperator":
        if not isinstance(multipoles, PermanentMultipoleSiteData):
            raise TypeError("multipoles must be PermanentMultipoleSiteData.")
        capacity = multipoles.site_capacity
        scale = PolarizationScaleData.unscaled(capacity) if scaling is None else scaling
        if not isinstance(scale, PolarizationScaleData):
            raise TypeError("scaling must be PolarizationScaleData or None.")
        if scale.site_capacity != capacity:
            raise ValueError("Scaling and multipole capacities differ.")
        if active_mask is None:
            active = jnp.ones((capacity,), dtype=jnp.bool_)
            active_id = "all-active"
        else:
            host_active = np.asarray(active_mask)
            if host_active.shape != (capacity,) or host_active.dtype != np.bool_:
                raise ValueError("active_mask must be a boolean array with shape (N,).")
            active = jnp.asarray(host_active)
            active_id = array_tree_fingerprint(host_active)
        return PreparedPolarizationOperator(
            self,
            multipoles,
            scale,
            active,
            active_id,
            widths=widths,
            cell=cell,
            particles=particles,
        )


class PolarizationOperatorResult(StrictModule):
    """Fields, operator action and route status of one application."""

    action: Array
    d_field: Array
    p_field: Array
    u_field: Array
    geometry_successful: Array
    finite: Array
    successful: Array


class _Bound(StrictModule):
    """Per-evaluation bound inputs of one polarization problem."""

    positions: Array
    cell_vectors: Array | None
    pairs: AbstractMultipolePairs | None
    permanent: MultipoleSiteSet
    polarizabilities: Array
    polarizable: Array
    widths: GaussianMultipoleWidths | None
    damping: Array


class PreparedPolarizationOperator(StrictModule, NonTrainableState):
    """Prepared matrix-free polarization operator over native multipole actions.

    PyTree layout: ``site_capacity`` and the content-free ``layout_id`` (plan,
    capacity, listed scaled-pair count, width-class count and electrostatics
    layout) are static; the listed scaled-pair count is the declared shape of
    the pair-scale table. The exact canonical preparation digest (site data,
    scales, active mask, widths, electrostatics) is the dynamic
    ``prepared_identity`` (``prepared_id`` decodes it on the host).
    """

    plan: PolarizationOperatorPlan
    multipoles: PermanentMultipoleSiteData
    scaling: PolarizationScaleData
    active_mask: Array
    widths: GaussianMultipoleWidths | None
    electrostatics: PreparedMultipoleElectrostatics
    corrections: tuple[WeightedPairCorrections, ...]
    site_capacity: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    prepared_identity: IdentityWords

    def __init__(
        self,
        plan: PolarizationOperatorPlan,
        multipoles: PermanentMultipoleSiteData,
        scaling: PolarizationScaleData,
        active_mask: Array,
        active_id: str | dict[str, Any],
        /,
        *,
        widths: GaussianMultipoleWidths | None = None,
        cell: PeriodicCell | None = None,
        particles: ParticleDiscretization | None = None,
    ) -> None:
        capacity = multipoles.site_capacity
        if (plan.site_kind == "gaussian") != (widths is not None):
            raise ValueError(
                "Gaussian polarization requires widths; point sites refuse them."
            )
        if widths is not None and widths.site_capacity != capacity:
            raise ValueError("Gaussian widths do not match the site capacity.")
        electrostatics = plan.electrostatics().prepare(
            capacity, cell=cell, particles=particles
        )
        corrections: tuple[WeightedPairCorrections, ...] = ()
        if plan.route.periodic:
            corrections = tuple(
                WeightedPairCorrections(scaling.scales, column, cell=cell)
                for column in _SCALE_COLUMNS
            )
        self.plan = plan
        self.multipoles = multipoles
        self.scaling = scaling
        self.active_mask = active_mask
        self.widths = widths
        self.electrostatics = electrostatics
        self.corrections = corrections
        self.site_capacity = capacity
        self.layout_id = canonical_fingerprint(
            {
                "kind": "prepared-polarization-operator-layout",
                "plan": plan.plan_id,
                "site_capacity": capacity,
                "scaled_pairs": scaling.scales.pair_count,
                "width_classes": None if widths is None else widths.class_count,
                "electrostatics": electrostatics.layout_id,
            }
        )
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-polarization-operator",
                "plan": plan.plan_id,
                "multipoles": multipoles.multipole_id,
                "scaling": scaling.scale_id,
                "active": active_id,
                "widths": None if widths is None else widths.widths_id,
                "electrostatics": electrostatics.prepared_id,
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical operator preparation digest."""
        return self.prepared_identity.fingerprint()

    def bind(
        self,
        positions: ArrayLike,
        /,
        *,
        cell_vectors: ArrayLike | None = None,
        pairs: AbstractMultipolePairs | None = None,
        moments: CartesianMultipoles | None = None,
        polarizabilities: ArrayLike | None = None,
        active: ArrayLike | None = None,
    ) -> _Bound:
        coordinate = jnp.asarray(positions)
        if coordinate.shape != (self.site_capacity, 3):
            raise ValueError("positions must have the prepared fixed shape (N,3).")
        if not jnp.issubdtype(coordinate.dtype, jnp.floating):
            raise TypeError("positions must be floating point.")
        dtype = coordinate.dtype
        periodic = self.plan.route.periodic
        if periodic and cell_vectors is None:
            raise ValueError("Periodic polarization requires cell_vectors.")
        if not periodic and cell_vectors is not None:
            raise ValueError("cell_vectors require a periodic polarization route.")
        vectors = None if cell_vectors is None else jnp.asarray(cell_vectors, dtype=dtype)
        if vectors is not None and vectors.shape != (3, 3):
            raise ValueError("cell_vectors must have shape (3,3).")
        permanent = self.multipoles.moments(dtype) if moments is None else moments
        if permanent.site_capacity != self.site_capacity:
            raise ValueError("Permanent moments do not match the site capacity.")
        alpha = (
            self.multipoles.polarizabilities.astype(dtype)
            if polarizabilities is None
            else jnp.asarray(polarizabilities, dtype=dtype)
        )
        if alpha.shape != (self.site_capacity,):
            raise ValueError("polarizabilities must have shape (N,).")
        mask = (
            self.active_mask if active is None else jnp.asarray(active, dtype=jnp.bool_)
        )
        damping = self.multipoles.damping.astype(dtype)
        return _Bound(
            positions=coordinate,
            cell_vectors=vectors,
            pairs=pairs,
            permanent=MultipoleSiteSet(
                permanent, active=mask, widths=self.widths, damping=damping
            ),
            polarizabilities=alpha,
            polarizable=mask & (alpha > 0.0),
            widths=self.widths,
            damping=damping,
        )

    def _induced(self, bound: _Bound, dipoles: Array, /) -> MultipoleSiteSet:
        return MultipoleSiteSet(
            dipole_multipoles(dipoles),
            active=bound.polarizable,
            widths=bound.widths,
            damping=bound.damping,
        )

    def _direct_kernel(self) -> AbstractMultipoleRadialKernel:
        if self.plan.site_kind == "gaussian":
            return GaussianKernel()
        if self.plan.damping == "exponential-cubic":
            return ExponentialDampedCoulombKernel(correction=False)
        return CoulombKernel()

    def form(
        self,
        bound: _Bound,
        left: MultipoleSiteSet,
        right: MultipoleSiteSet,
        column: str,
        /,
        *,
        symmetric: bool = False,
    ) -> tuple[Array, Array]:
        """Scaled, damped bilinear interaction ``B_column(L, R)`` and its status."""
        es = self.electrostatics
        constant = self.plan.coulomb_constant
        index = _SCALE_COLUMNS.index(column)
        relation = es.resolve_pairs(bound.pairs)
        minimum = self.plan.minimum_distance
        route = self.plan.route
        if isinstance(route, FiniteMultipoleRoute):
            scales = self.scaling.scales
            real = relation.pair_sum(
                bound.positions,
                None,
                left,
                right,
                self._direct_kernel(),
                cutoff=None,
                minimum_distance=minimum,
                scale=None if scales.pair_count == 0 else _ScaleColumn(scales, index),
                symmetric=symmetric,
            )
            widths_ok = (
                jnp.asarray(True) if bound.widths is None else bound.widths.valid()
            )
            return (
                constant * real.value,
                real.geometry_successful & real.coverage_successful & widths_ok,
            )
        parts = es.bilinear(
            bound.positions,
            left,
            right,
            cell_vectors=bound.cell_vectors,
            pairs=bound.pairs,
            intrasite="exclude",
            symmetric=symmetric,
        )
        value, successful = parts.value, parts.successful
        if self.plan.site_kind == "point" and self.plan.damping == "exponential-cubic":
            damping = relation.pair_sum(
                bound.positions,
                bound.cell_vectors,
                left,
                right,
                ExponentialDampedCoulombKernel(correction=True),
                cutoff=route.settings.real_cutoff,
                minimum_distance=minimum,
                scale=None,
                symmetric=symmetric,
            )
            value = value + constant * damping.value
            successful = successful & damping.geometry_successful
        correction = self.corrections[index].pair_sum(
            bound.positions,
            bound.cell_vectors,
            left,
            right,
            self._direct_kernel(),
            minimum_distance=minimum,
        )
        return (
            value + constant * correction.value,
            successful & correction.geometry_successful,
        )

    def field(self, bound: _Bound, column: str, /) -> tuple[Array, Array]:
        """Permanent field ``-dB_column(u, M)/du`` at polarizable sites."""

        def form(dipoles: Array) -> tuple[Array, Array]:
            return self.form(
                bound, self._induced(bound, dipoles), bound.permanent, column
            )

        gradient, successful = jax.grad(form, has_aux=True)(
            jnp.zeros((self.site_capacity, 3), dtype=bound.positions.dtype)
        )
        return jnp.where(bound.polarizable[:, None], -gradient, 0.0), successful

    def induced_field(self, bound: _Bound, dipoles: Array, /) -> tuple[Array, Array]:
        """Mutual field ``T_u u = -d(B_u(u, u)/2)/du``."""

        def form(value: Array) -> tuple[Array, Array]:
            sites = self._induced(bound, value)
            energy, successful = self.form(bound, sites, sites, "mutual", symmetric=True)
            return 0.5 * energy, successful

        gradient, successful = jax.grad(form, has_aux=True)(dipoles)
        return jnp.where(bound.polarizable[:, None], -gradient, 0.0), successful

    def inverse_polarizability(self, bound: _Bound, /) -> Array:
        alpha = bound.polarizabilities
        return jnp.where(bound.polarizable, 1.0 / jnp.where(alpha > 0.0, alpha, 1.0), 1.0)

    def action(self, bound: _Bound, dipoles: Array, /) -> tuple[Array, Array]:
        """``A u = alpha^-1 u - T_u u`` on polarizable sites, identity elsewhere."""
        u_field, successful = self.induced_field(bound, dipoles)
        action = self.inverse_polarizability(bound)[:, None] * dipoles - u_field
        return jnp.where(bound.polarizable[:, None], action, dipoles), successful

    def functional(self, bound: _Bound, dipoles: Array, /) -> tuple[Array, Array]:
        """The Hylleraas functional ``E(u)`` and its route status."""
        induced = jnp.where(bound.polarizable[:, None], dipoles, 0.0)
        sites = self._induced(bound, induced)
        mutual, mutual_ok = self.form(bound, sites, sites, "mutual", symmetric=True)
        permanent, permanent_ok = self.form(bound, sites, bound.permanent, "polarization")
        self_term = 0.5 * contract(
            "n,nd,nd->", self.inverse_polarizability(bound), induced, induced
        )
        return self_term + 0.5 * mutual + permanent, mutual_ok & permanent_ok

    def apply(
        self,
        positions: ArrayLike,
        induced_dipoles: ArrayLike,
        /,
        *,
        cell_vectors: ArrayLike | None = None,
        pairs: AbstractMultipolePairs | None = None,
    ) -> PolarizationOperatorResult:
        bound = self.bind(positions, cell_vectors=cell_vectors, pairs=pairs)
        induced = jnp.asarray(induced_dipoles, dtype=bound.positions.dtype)
        if induced.shape != (self.site_capacity, 3):
            raise ValueError("induced_dipoles must have shape (N,3).")
        p_field, p_ok = self.field(bound, "polarization")
        d_field, d_ok = (
            (p_field, p_ok)
            if self.scaling.scales.pair_count == 0
            else self.field(bound, "direct")
        )
        u_field, u_ok = self.induced_field(bound, induced)
        action = jnp.where(
            bound.polarizable[:, None],
            self.inverse_polarizability(bound)[:, None] * induced - u_field,
            induced,
        )
        finite = (
            jnp.all(jnp.isfinite(bound.positions))
            & jnp.all(jnp.isfinite(induced))
            & jnp.all(jnp.isfinite(action))
            & jnp.all(jnp.isfinite(d_field))
            & jnp.all(jnp.isfinite(p_field))
            & jnp.all(jnp.isfinite(u_field))
        )
        geometry = p_ok & d_ok & u_ok
        return PolarizationOperatorResult(
            action, d_field, p_field, u_field, geometry, finite, finite & geometry
        )


class PolarizationPreconditionerResult(StrictModule):
    """One preconditioner application and fail-closed evidence."""

    value: Array
    finite: Array
    successful: Array


class PolarizationPreconditionerPlan(StrictModule, NonTrainableState):
    """Identity or local polarizability (Jacobi ``alpha``) preconditioner."""

    kind: PolarizationPreconditionerKind = eqx.field(static=True)
    diagonal_floor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: PolarizationPreconditionerKind | str = (
            PolarizationPreconditionerKind.POLARIZABILITY
        ),
        /,
        *,
        diagonal_floor: float = 1.0e-12,
    ) -> None:
        kind_ = _enum_value(kind, PolarizationPreconditionerKind, "kind")
        floor = positive_finite_float(diagonal_floor, "diagonal_floor")
        self.kind, self.diagonal_floor = kind_, floor
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarization-preconditioner-plan",
                "preconditioner": kind_.value,
                "diagonal_floor": float(floor).hex(),
            }
        )

    def prepare(
        self, operator: PreparedPolarizationOperator, /
    ) -> "PreparedPolarizationPreconditioner":
        if not isinstance(operator, PreparedPolarizationOperator):
            raise TypeError("operator must be PreparedPolarizationOperator.")
        return PreparedPolarizationPreconditioner(self, operator)


class PreparedPolarizationPreconditioner(StrictModule, NonTrainableState):
    """Native diagonal preconditioner bound to one prepared operator.

    Static content-free ``layout_id`` (plan and operator layout) labels the
    native preconditioner; the exact preparation digest is the dynamic
    ``prepared_identity`` (``prepared_id`` decodes it on the host).
    """

    plan: PolarizationPreconditionerPlan
    operator: PreparedPolarizationOperator
    layout_id: str = eqx.field(static=True)
    prepared_identity: IdentityWords

    def __init__(
        self,
        plan: PolarizationPreconditionerPlan,
        operator: PreparedPolarizationOperator,
        /,
    ) -> None:
        self.plan, self.operator = plan, operator
        self.layout_id = canonical_fingerprint(
            {
                "kind": "prepared-polarization-preconditioner-layout",
                "plan": plan.plan_id,
                "operator": operator.layout_id,
            }
        )
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-polarization-preconditioner",
                "plan": plan.plan_id,
                "operator": operator.prepared_id,
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical preconditioner preparation digest."""
        return self.prepared_identity.fingerprint()

    def diagonal(self, polarizabilities: Array, polarizable: Array, /) -> Array:
        """Operator diagonal estimate ``1/alpha`` (identity rows elsewhere)."""
        if self.plan.kind is PolarizationPreconditionerKind.IDENTITY:
            return jnp.ones_like(polarizabilities)
        return jnp.where(
            polarizable,
            1.0 / jnp.maximum(polarizabilities, self.plan.diagonal_floor),
            1.0,
        )

    def native(
        self, polarizabilities: Array, polarizable: Array, /
    ) -> DiagonalPreconditioner:
        count = self.operator.site_capacity
        diagonal = jnp.repeat(self.diagonal(polarizabilities, polarizable), 3)
        return DiagonalPreconditioner(
            diagonal,
            space=ArraySpace((count, 3), dtype=diagonal.dtype),
            positive_definite=True,
            preconditioner_id=f"{self.layout_id}:diagonal",
        )

    def apply(self, residual: ArrayLike, /) -> PolarizationPreconditionerResult:
        value = jnp.asarray(residual)
        if value.shape != (self.operator.site_capacity, 3):
            raise ValueError("residual must have shape (N,3).")
        alpha = self.operator.multipoles.polarizabilities.astype(value.dtype)
        polarizable = self.operator.active_mask & (alpha > 0.0)
        result = self.native(alpha, polarizable).apply(value)
        finite = jnp.all(jnp.isfinite(result))
        return PolarizationPreconditionerResult(
            jnp.where(finite, result, jnp.nan), finite, finite
        )


class PolarizationPredictorPlan(StrictModule, NonTrainableState):
    """Two-frame induced-dipole warm-start history (PCG only)."""

    history_coefficient: float = eqx.field(static=True)
    direct_fallback: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        history_coefficient: float = 1.0,
        direct_fallback: bool = True,
    ) -> None:
        coefficient = float(history_coefficient)
        if not np.isfinite(coefficient) or not 0.0 <= coefficient <= 2.0:
            raise ValueError("history_coefficient must be finite and in [0,2].")
        self.history_coefficient = coefficient
        self.direct_fallback = bool(direct_fallback)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarization-predictor-plan",
                "history_coefficient": coefficient.hex(),
                "direct_fallback": self.direct_fallback,
            }
        )


class PolarizationPredictorState(StrictModule):
    """Fixed-shape two-frame warm-start history.

    ``prepared_identity`` names the producing prepared solver as dynamic words
    (``prepared_id`` decodes it on the host); a solver folds a mismatch into its
    success status instead of decoding it in compiled code.
    """

    history: Array
    valid_count: Array
    prepared_identity: IdentityWords

    @property
    def prepared_id(self) -> str:
        """Host decoding of the producing solver preparation digest."""
        return self.prepared_identity.fingerprint()


class PolarizationCurvatureProbe(StrictModule, NonTrainableState):
    """Bounded Lanczos estimate of the smallest eigenvalue of ``alpha^1/2 A alpha^1/2``.

    The smallest Ritz value is an upper bound of the true minimum: a negative
    value detects an indefinite (polarization-catastrophe) system and refuses
    the result; a positive value is never stability evidence.
    """

    subspace_dimension: int = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    probe_id: str = eqx.field(static=True)

    def __init__(
        self, /, *, subspace_dimension: int = 12, maximum_steps: int = 40, seed: int = 0
    ) -> None:
        subspace = int(subspace_dimension)
        steps = int(maximum_steps)
        if subspace < 2 or steps <= 0 or int(seed) < 0:
            raise ValueError("Curvature probe dimensions, steps and seed are invalid.")
        self.subspace_dimension = subspace
        self.maximum_steps = steps
        self.seed = int(seed)
        self.probe_id = canonical_fingerprint(
            {
                "kind": "polarization-curvature-probe",
                "subspace_dimension": subspace,
                "maximum_steps": steps,
                "seed": self.seed,
            }
        )


PolarizationStabilityCertificate: TypeAlias = Literal[
    "scaled-block-gershgorin", "dense-spectrum", "assumed-positive-definite"
]


class PolarizationStabilityPolicy(StrictModule, NonTrainableState):
    """How positive definiteness of ``A = alpha^-1 - T_u`` is established.

    Stability is shown for the congruent ``S = alpha^1/2 A alpha^1/2`` on
    polarizable sites (identity rows elsewhere), which is positive definite
    exactly when ``A`` is:

    * ``"scaled-block-gershgorin"`` -- matrix-free analytic proof for
      nonperiodic routes. With ``S = I - M`` and 3x3 blocks
      ``M_ij = sqrt(a_i a_j) k s_ij grad grad phi_ij`` of the actual damped or
      Gaussian kernel and mutual scales, the Schur test on the symmetric
      block-norm matrix gives ``lambda_min(S) >= 1 - max_i sum_j ||M_ij||_2``.
      Proven when that lower bound exceeds ``margin``; a smaller (or
      negative) bound means *unproved*, not unstable. ``O(N^2)`` work,
      ``O(row_block N)`` memory.
    * ``"dense-spectrum"`` -- the bounded reference route for any route
      (including periodic Ewald/PME): native ``DenseEigh`` of the actual
      materialised operator ``S`` (``(3N)^2`` entries admitted against
      ``maximum_dense_bytes`` at preparation). Proven when the computed
      ``lambda_min`` exceeds ``max(margin, 64 * 3N * eps) * max|lambda|``,
      the backward-error allowance of a backward-stable eigensolver (the
      constant 64 is the declared numerical assumption of this certificate).
    * ``"assumed-positive-definite"`` -- a deliberate user assumption: the
      result reports ``stability_proven=False`` and ``stability_assumed=True``
      and is never presented as certified. Detected indefiniteness (negative
      solution curvature or negative Lanczos Ritz value) still refuses.
    """

    certificate: PolarizationStabilityCertificate = eqx.field(static=True)
    margin: float = eqx.field(static=True)
    maximum_dense_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        certificate: PolarizationStabilityCertificate,
        /,
        *,
        margin: float = 1.0e-8,
        maximum_dense_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        certificate_ = parse(certificate, PolarizationStabilityCertificate, "certificate")
        margin_ = positive_finite_float(margin, "margin")
        if margin_ >= 1.0:
            raise ValueError("margin must be below one.")
        dense = int(maximum_dense_bytes)
        if dense <= 0:
            raise ValueError("maximum_dense_bytes must be positive.")
        self.certificate = certificate_
        self.margin = margin_
        self.maximum_dense_bytes = dense
        self.policy_id = canonical_fingerprint(
            {
                "kind": "polarization-stability-policy",
                "certificate": certificate_,
                "margin": float(margin_).hex(),
                "maximum_dense_bytes": dense,
            }
        )

    @property
    def assumed(self) -> bool:
        return self.certificate == "assumed-positive-definite"


class PolarizationSolverPlan(StrictModule, NonTrainableState):
    """Native PCG (stationary) or fixed-order TCG solver policy."""

    kind: PolarizationSolverKind = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    tcg_order: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    force_tolerance: float = eqx.field(static=True)
    derivative_tolerance: float = eqx.field(static=True)
    curvature_probe: PolarizationCurvatureProbe | None
    stability: PolarizationStabilityPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: PolarizationSolverKind | str = PolarizationSolverKind.PCG,
        /,
        *,
        maximum_iterations: int = 100,
        tcg_order: int = 3,
        tolerance: float = 1.0e-8,
        force_tolerance: float | None = None,
        derivative_tolerance: float | None = None,
        curvature_probe: PolarizationCurvatureProbe | None = None,
        stability: PolarizationStabilityPolicy | None = None,
    ) -> None:
        kind_ = _enum_value(kind, PolarizationSolverKind, "kind")
        iterations, order = int(maximum_iterations), int(tcg_order)
        tolerance_ = positive_finite_float(tolerance, "tolerance")
        force = (
            tolerance_
            if force_tolerance is None
            else positive_finite_float(force_tolerance, "force_tolerance")
        )
        derivative = (
            tolerance_
            if derivative_tolerance is None
            else positive_finite_float(derivative_tolerance, "derivative_tolerance")
        )
        if iterations <= 0 or order <= 0:
            raise ValueError("maximum_iterations and tcg_order must both be positive.")
        if curvature_probe is not None and not isinstance(
            curvature_probe, PolarizationCurvatureProbe
        ):
            raise TypeError("curvature_probe must be PolarizationCurvatureProbe or None.")
        stability_ = (
            PolarizationStabilityPolicy("scaled-block-gershgorin")
            if stability is None
            else stability
        )
        if not isinstance(stability_, PolarizationStabilityPolicy):
            raise TypeError("stability must be PolarizationStabilityPolicy or None.")
        self.kind = kind_
        self.maximum_iterations = iterations
        self.tcg_order = order
        self.tolerance = tolerance_
        self.force_tolerance = force
        self.derivative_tolerance = derivative
        self.curvature_probe = curvature_probe
        self.stability = stability_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarization-solver-plan",
                "solver": kind_.value,
                "maximum_iterations": iterations,
                "tcg_order": order,
                "tolerance": float(tolerance_).hex(),
                "force_tolerance": float(force).hex(),
                "derivative_tolerance": float(derivative).hex(),
                "curvature_probe": None
                if curvature_probe is None
                else curvature_probe.probe_id,
                "stability": stability_.policy_id,
                "native": "phydrax.linalg.PCG",
            }
        )

    @property
    def functional(self) -> str:
        return (
            "stationary"
            if self.kind is PolarizationSolverKind.PCG
            else f"tcg-order-{self.tcg_order}"
        )

    @property
    def derivative_mode(self) -> str:
        return (
            "implicit-stationary"
            if self.kind is PolarizationSolverKind.PCG
            else "algorithmic-fixed-order"
        )

    def prepare(
        self,
        operator: PreparedPolarizationOperator,
        preconditioner: PreparedPolarizationPreconditioner,
        /,
        *,
        predictor: PolarizationPredictorPlan | None = None,
    ) -> "PreparedPolarizationSolver":
        if not isinstance(operator, PreparedPolarizationOperator) or not isinstance(
            preconditioner, PreparedPolarizationPreconditioner
        ):
            raise TypeError(
                "Solver preparation requires a prepared operator and preconditioner."
            )
        if preconditioner.operator.prepared_id != operator.prepared_id:
            raise ValueError("Preconditioner and operator preparations differ.")
        predictor_ = PolarizationPredictorPlan() if predictor is None else predictor
        if not isinstance(predictor_, PolarizationPredictorPlan):
            raise TypeError("predictor must be PolarizationPredictorPlan or None.")
        return PreparedPolarizationSolver(
            self, operator, preconditioner, predictor_, self.plan_id
        )


class PolarizationPlan(StrictModule, NonTrainableState):
    """Canonical composition of operator, preconditioner, solver and predictor."""

    operator: PolarizationOperatorPlan
    preconditioner: PolarizationPreconditionerPlan
    solver: PolarizationSolverPlan
    predictor: PolarizationPredictorPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        coulomb_constant: float,
        maximum_iterations: int = 100,
        tolerance: float = 1e-8,
        solver_kind: PolarizationSolverKind | str = PolarizationSolverKind.PCG,
        tcg_order: int = 3,
        force_tolerance: float | None = None,
        derivative_tolerance: float | None = None,
        preconditioner_kind: PolarizationPreconditionerKind | str = (
            PolarizationPreconditionerKind.POLARIZABILITY
        ),
        predictor: PolarizationPredictorPlan | None = None,
        site_kind: PolarizationSiteKind = "point",
        damping: PolarizationDamping = "exponential-cubic",
        route: MultipoleRoute | None = None,
        minimum_distance: float = 1.0e-8,
        curvature_probe: PolarizationCurvatureProbe | None = None,
        stability: PolarizationStabilityPolicy | None = None,
    ) -> None:
        operator = PolarizationOperatorPlan(
            coulomb_constant=coulomb_constant,
            site_kind=site_kind,
            damping=damping,
            route=route,
            minimum_distance=minimum_distance,
        )
        if stability is None and operator.route.periodic:
            raise ValueError(
                "Periodic polarization needs an explicit stability policy: the "
                "matrix-free Gershgorin proof covers nonperiodic routes only; choose "
                "'dense-spectrum' (bounded reference) or a declared "
                "'assumed-positive-definite' assumption."
            )
        preconditioner = PolarizationPreconditionerPlan(preconditioner_kind)
        solver = PolarizationSolverPlan(
            solver_kind,
            maximum_iterations=maximum_iterations,
            tcg_order=tcg_order,
            tolerance=tolerance,
            force_tolerance=force_tolerance,
            derivative_tolerance=derivative_tolerance,
            curvature_probe=curvature_probe,
            stability=stability,
        )
        predictor_ = PolarizationPredictorPlan() if predictor is None else predictor
        if not isinstance(predictor_, PolarizationPredictorPlan):
            raise TypeError("predictor must be PolarizationPredictorPlan or None.")
        self.operator = operator
        self.preconditioner = preconditioner
        self.solver = solver
        self.predictor = predictor_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarization-plan",
                "operator": operator.plan_id,
                "preconditioner": preconditioner.plan_id,
                "solver": solver.plan_id,
                "predictor": predictor_.plan_id,
            }
        )

    def prepare(
        self,
        multipoles: PermanentMultipoleSiteData,
        /,
        *,
        scaling: PolarizationScaleData | None = None,
        active_mask: ArrayLike | None = None,
        widths: GaussianMultipoleWidths | None = None,
        cell: PeriodicCell | None = None,
        particles: ParticleDiscretization | None = None,
    ) -> "PreparedPolarizationSolver":
        operator = self.operator.prepare(
            multipoles,
            scaling=scaling,
            active_mask=active_mask,
            widths=widths,
            cell=cell,
            particles=particles,
        )
        preconditioner = self.preconditioner.prepare(operator)
        return PreparedPolarizationSolver(
            self.solver, operator, preconditioner, self.predictor, self.plan_id
        )


class PolarizationState(StrictModule):
    """Induced dipoles with explicit solve, stability and force-validity evidence.

    ``converged`` is the native PCG success status and is always false for
    fixed-order TCG; ``stationary`` reports the measured stationarity residual
    test ``residual <= tolerance`` for either method. ``stability_proven``
    is true only under a successful proof certificate; ``stability_assumed``
    marks the declared assumption policy (never a proof). ``successful`` and
    ``force_valid`` require proven or explicitly assumed stability without
    detected indefiniteness; ``force_valid`` states that forces are exact
    derivatives of ``functional`` (for PCG this also requires
    ``residual <= force_tolerance``).
    """

    induced_dipoles: Array
    residual: Array
    relative_residual: Array
    iterations: Array
    linear_status: Array
    converged: Array
    stationary: Array
    stability_proven: Array
    stability_lower_bound: Array
    force_valid: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    solver_kind: str = eqx.field(static=True)
    functional: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True)
    stability_certificate: str = eqx.field(static=True)
    stability_assumed: bool = eqx.field(static=True)


class PolarizationStabilityEvidence(StrictModule):
    """Stability certificate outcome and separate indefiniteness detections.

    ``lower_bound`` bounds ``lambda_min(alpha^1/2 A alpha^1/2)`` from below
    (Gershgorin) or is the computed minimum eigenvalue (dense spectrum); it is
    NaN under the assumption policy. ``solution_curvature = u . E_p`` and the
    Lanczos Ritz value are detections only: ``indefinite_detected`` refuses,
    their positive values never prove stability.
    """

    certificate: str = eqx.field(static=True)
    assumed: bool = eqx.field(static=True)
    proven: Array
    lower_bound: Array
    allowance: Array
    solution_curvature: Array
    probed: bool = eqx.field(static=True)
    minimum_ritz_value: Array
    probe_converged: Array
    indefinite_detected: Array
    successful: Array


class PolarizationSolveResult(StrictModule):
    """Solver state, next warm start, fields at the solution and status."""

    state: PolarizationState
    predictor_state: PolarizationPredictorState
    operator: PolarizationOperatorResult
    stability: PolarizationStabilityEvidence
    initial_dipoles: Array
    breakdown: Array
    successful: Array


class _Solved(StrictModule):
    induced: Array
    rhs: Array
    direct: Array
    initial: Array
    status: Array
    residual: Array
    relative: Array
    iterations: Array
    fields_successful: Array
    history_bound: Array


class PreparedPolarizationSolver(StrictModule, NonTrainableState):
    """Prepared fixed-capacity polarization solve runtime.

    PyTree layout: static ``result_plan_id`` (the configuration-only
    :class:`PolarizationPlan` digest) and content-free ``layout_id`` (solver,
    predictor and result plans with the preconditioner/operator layouts) label
    the native linear and eigen problems. The exact preparation digest is the
    dynamic ``prepared_identity`` (``prepared_id`` decodes it on the host); a
    warm-start history of another preparation refuses the solve through its
    success status.
    """

    plan: PolarizationSolverPlan
    operator: PreparedPolarizationOperator
    preconditioner: PreparedPolarizationPreconditioner
    predictor: PolarizationPredictorPlan
    result_plan_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    prepared_identity: IdentityWords

    def __init__(
        self,
        plan: PolarizationSolverPlan,
        operator: PreparedPolarizationOperator,
        preconditioner: PreparedPolarizationPreconditioner,
        predictor: PolarizationPredictorPlan,
        result_plan_id: str,
        /,
    ) -> None:
        stability = plan.stability
        if stability.certificate == "scaled-block-gershgorin" and (
            operator.plan.route.periodic
        ):
            raise ValueError(
                "The scaled block-Gershgorin proof covers nonperiodic operators only; "
                "a periodic operator also has reciprocal and intrasite blocks. Use "
                "'dense-spectrum' or a declared 'assumed-positive-definite' policy."
            )
        if stability.certificate == "dense-spectrum":
            dimension = 3 * operator.site_capacity
            matrix_bytes = dimension * dimension * 8
            if matrix_bytes > stability.maximum_dense_bytes:
                raise ValueError(
                    f"Dense stability spectrum needs {matrix_bytes} bytes, exceeding "
                    f"maximum_dense_bytes={stability.maximum_dense_bytes}."
                )
        self.plan, self.operator, self.preconditioner, self.predictor = (
            plan,
            operator,
            preconditioner,
            predictor,
        )
        self.result_plan_id = str(result_plan_id)
        self.layout_id = canonical_fingerprint(
            {
                "kind": "prepared-polarization-solver-layout",
                "plan": plan.plan_id,
                "operator": operator.layout_id,
                "preconditioner": preconditioner.layout_id,
                "predictor": predictor.plan_id,
                "result_plan": self.result_plan_id,
            }
        )
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-polarization-solver",
                "plan": plan.plan_id,
                "operator": operator.prepared_id,
                "preconditioner": preconditioner.prepared_id,
                "predictor": predictor.plan_id,
                "result_plan": self.result_plan_id,
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical solver preparation digest."""
        return self.prepared_identity.fingerprint()

    def initial_predictor_state(self, /) -> PolarizationPredictorState:
        capacity = self.operator.site_capacity
        return PolarizationPredictorState(
            jnp.zeros((2, capacity, 3), dtype=self.operator.multipoles.dipoles.dtype),
            jnp.zeros((), dtype=jnp.int32),
            self.prepared_identity,
        )

    def _require_predictor(
        self, state: PolarizationPredictorState | None, /
    ) -> PolarizationPredictorState:
        if state is None:
            return self.initial_predictor_state()
        if self.plan.kind is PolarizationSolverKind.TCG:
            raise ValueError(
                "TCG uses a deterministic direct-field or zero guess; a warm-start "
                "history would make its functional history dependent."
            )
        if not isinstance(state, PolarizationPredictorState):
            raise TypeError("predictor_state must be PolarizationPredictorState or None.")
        if state.history.shape != (2, self.operator.site_capacity, 3):
            raise ValueError("Predictor history has an incompatible fixed shape.")
        return state

    def _guess(
        self, bound: _Bound, direct: Array, state: PolarizationPredictorState, /
    ) -> Array:
        alpha = bound.polarizabilities[:, None]
        fallback = (
            alpha * direct if self.predictor.direct_fallback else jnp.zeros_like(direct)
        )
        if self.plan.kind is PolarizationSolverKind.TCG:
            return jnp.where(bound.polarizable[:, None], fallback, 0.0)
        history = state.history.astype(direct.dtype)
        previous = history[0]
        extrapolated = previous + self.predictor.history_coefficient * (
            previous - history[1]
        )
        guess = jnp.where(
            state.valid_count >= 2,
            extrapolated,
            jnp.where(state.valid_count >= 1, previous, fallback),
        )
        return jax.lax.stop_gradient(jnp.where(bound.polarizable[:, None], guess, 0.0))

    def _native_solve(self, bound: _Bound, rhs: Array, initial: Array, /) -> _Solved:
        dtype = bound.positions.dtype
        count = self.operator.site_capacity
        space = ArraySpace((count, 3), dtype=dtype)

        def action(value: Array) -> Array:
            return self.operator.action(bound, value)[0]

        operator = FunctionLinearOperator(
            action,
            source=space,
            target=space,
            transpose_action=action,
            # PCG requires a positive-definite claim. It is the stability
            # policy's claim, and every output (state, energy, forces, implicit
            # derivatives) is gated on that policy's proof or declared
            # assumption, so an unproved operator never publishes a result.
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "asserted",
                },
            ),
            operator_id=f"{self.layout_id}:polarization-operator",
        )
        pcg = self.plan.kind is PolarizationSolverKind.PCG
        policy = LinearSolvePolicy(
            PCG(),
            tolerance=TolerancePolicy(
                relative=0.0,
                absolute=self.plan.tolerance if pcg else 0.0,
                max_steps=self.plan.maximum_iterations if pcg else self.plan.tcg_order,
            ),
            preconditioning=PreconditioningPolicy(
                self.preconditioner.native(bound.polarizabilities, bound.polarizable)
            ),
            differentiation=DifferentiationPolicy(
                "mathematical" if pcg else "algorithmic"
            ),
            derivative_solve=LinearDerivativeSolvePolicy(
                relative_tolerance=0.0,
                absolute_tolerance=self.plan.derivative_tolerance,
                maximum_steps=self.plan.maximum_iterations,
            ),
            failure=FailurePolicy("status"),
        )
        result = solve(
            LinearSystem(operator, problem_id=f"{self.layout_id}:polarization"),
            rhs,
            policy=policy,
            initial_guess=initial,
        )
        diagnostics = result.diagnostics
        return _Solved(
            induced=jnp.where(bound.polarizable[:, None], result.value, 0.0),
            rhs=rhs,
            direct=rhs,
            initial=initial,
            status=jnp.asarray(result.status, dtype=jnp.int32),
            residual=jnp.asarray(diagnostics.residual_norm, dtype=dtype).reshape(()),
            relative=jnp.asarray(diagnostics.relative_residual, dtype=dtype).reshape(()),
            iterations=jnp.asarray(diagnostics.iterations, dtype=jnp.int32).reshape(()),
            fields_successful=jnp.asarray(True),
            history_bound=jnp.asarray(True),
        )

    def _solve_bound(
        self, bound: _Bound, state: PolarizationPredictorState, /
    ) -> _Solved:
        rhs, rhs_ok = self.operator.field(bound, "polarization")
        direct, direct_ok = (
            (rhs, rhs_ok)
            if self.operator.scaling.scales.pair_count == 0
            else self.operator.field(bound, "direct")
        )
        initial = self._guess(bound, direct, state)
        solved = self._native_solve(bound, rhs, initial)
        return _Solved(
            induced=solved.induced,
            rhs=rhs,
            direct=direct,
            initial=initial,
            status=solved.status,
            residual=solved.residual,
            relative=solved.relative,
            iterations=solved.iterations,
            fields_successful=rhs_ok & direct_ok,
            history_bound=state.prepared_identity.equals(self.prepared_identity),
        )

    def _scaled_operator(self, frozen: _Bound, name: str, /) -> FunctionLinearOperator:
        """``S = D A D`` with ``D = alpha^1/2`` on polarizable sites, 1 elsewhere."""
        dtype = frozen.positions.dtype
        count = self.operator.site_capacity
        space = ArraySpace((count, 3), dtype=dtype)
        scale = jnp.sqrt(jnp.where(frozen.polarizable, frozen.polarizabilities, 1.0))[
            :, None
        ]

        def scaled(value: Array) -> Array:
            return scale * self.operator.action(frozen, scale * value)[0]

        return FunctionLinearOperator(
            scaled,
            source=space,
            target=space,
            transpose_action=scaled,
            properties=OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            ),
            operator_id=f"{self.layout_id}:{name}",
        )

    def _certificate(self, frozen: _Bound, /) -> tuple[Array, Array, Array]:
        """``(proven, lower_bound, allowance)`` of the declared stability policy."""
        dtype = frozen.positions.dtype
        policy = self.plan.stability
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        widths_ok = jnp.asarray(True) if frozen.widths is None else frozen.widths.valid()
        if policy.certificate == "assumed-positive-definite":
            return jnp.asarray(False), nan, nan
        if policy.certificate == "scaled-block-gershgorin":
            operator = self.operator
            relation = operator.electrostatics.resolve_pairs(frozen.pairs)
            if not isinstance(relation, ExplicitImagePairs):
                raise ValueError(
                    "The Gershgorin proof requires the nonperiodic reference relation."
                )
            weights = jnp.where(
                frozen.polarizable,
                jnp.sqrt(jnp.maximum(frozen.polarizabilities, 0.0)),
                0.0,
            )
            scales = operator.scaling.scales
            rows, distinct = relation.row_tensor_norm_sums(
                frozen.positions,
                operator._induced(
                    frozen, jnp.zeros((operator.site_capacity, 3), dtype=dtype)
                ),
                weights,
                operator._direct_kernel(),
                minimum_distance=operator.plan.minimum_distance,
                scale=None
                if scales.pair_count == 0
                else _ScaleColumn(scales, _SCALE_COLUMNS.index("mutual")),
            )
            radius = operator.plan.coulomb_constant * jnp.max(
                jnp.where(frozen.polarizable, rows, 0.0)
            )
            lower = 1.0 - radius
            allowance = jnp.asarray(policy.margin, dtype=dtype)
            proven = distinct & widths_ok & jnp.isfinite(lower) & (lower > allowance)
            return proven, lower, allowance
        dimension = 3 * self.operator.site_capacity
        spectrum = eigensolve(
            Eigenproblem(
                self._scaled_operator(frozen, "stability-spectrum"),
                problem_id=f"{self.layout_id}:stability-spectrum",
            ),
            policy=EigenSolvePolicy(
                DenseEigh(),
                count=dimension,
                which="smallest-algebraic",
                materialization=MaterializationPolicy(
                    max_entries=dimension * dimension,
                    max_bytes=policy.maximum_dense_bytes,
                    max_workspace_bytes=8 * policy.maximum_dense_bytes,
                ),
            ),
        )
        values = jnp.real(jnp.asarray(spectrum.eigenvalues)).reshape((-1,)).astype(dtype)
        smallest = jnp.min(values)
        allowance = jnp.maximum(
            jnp.asarray(policy.margin, dtype=dtype),
            64.0 * dimension * jnp.finfo(dtype).eps,
        ) * jnp.max(jnp.abs(values))
        proven = (
            jnp.all(spectrum.successful)
            & widths_ok
            & jnp.all(jnp.isfinite(values))
            & (smallest > allowance)
        )
        return proven, smallest, allowance

    def _stability(
        self, bound: _Bound, solved: _Solved, stationary: Array, /
    ) -> PolarizationStabilityEvidence:
        dtype = bound.positions.dtype
        frozen = jax.lax.stop_gradient(bound)
        policy = self.plan.stability
        proven, lower, allowance = self._certificate(frozen)
        curvature = contract("nd,nd->", solved.induced, solved.rhs)
        rhs_nonzero = jnp.any(solved.rhs != 0.0)
        # For positive definite A, u . E_p = E_p A^-1 E_p > 0 at the solution:
        # a nonpositive value at a stationary solution detects indefiniteness.
        detected = stationary & rhs_nonzero & ~(curvature > 0.0)
        probe = self.plan.curvature_probe
        ritz = jnp.asarray(jnp.nan, dtype=dtype)
        probe_converged = jnp.asarray(False)
        if probe is not None:
            count = self.operator.site_capacity
            subspace = min(probe.subspace_dimension, 3 * count - 1)
            solved_probe = eigensolve(
                Eigenproblem(
                    self._scaled_operator(frozen, "curvature-probe"),
                    problem_id=f"{self.layout_id}:curvature-probe",
                ),
                policy=EigenSolvePolicy(
                    RestartedLanczos(
                        subspace_dimension=subspace,
                        restart_dimension=max(1, min(2, subspace - 1)),
                    ),
                    count=1,
                    which="smallest-algebraic",
                    max_steps=probe.maximum_steps,
                    key=jnp.asarray((0, probe.seed), dtype=jnp.uint32),
                ),
            )
            ritz = jnp.real(jnp.asarray(solved_probe.eigenvalues)).reshape((-1,))[0]
            ritz = ritz.astype(dtype)
            probe_converged = jnp.all(solved_probe.successful)
            # A Ritz value is a Rayleigh quotient, an upper bound of lambda_min:
            # negative proves indefiniteness; positive proves nothing.
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

    def _state(
        self, bound: _Bound, solved: _Solved, /
    ) -> tuple[PolarizationState, Array, PolarizationStabilityEvidence]:
        status = solved.status
        pcg = self.plan.kind is PolarizationSolverKind.PCG
        finite = (
            jnp.all(jnp.isfinite(solved.induced))
            & jnp.isfinite(solved.residual)
            & jnp.all(jnp.isfinite(solved.rhs))
        )
        stationary = finite & (solved.residual <= self.plan.tolerance)
        breakdown = status == int(LinearSolveStatus.BREAKDOWN)
        if pcg:
            converged = status == int(LinearSolveStatus.SUCCESS)
            method_ok = converged & stationary
            force_valid = method_ok & (solved.residual <= self.plan.force_tolerance)
        else:
            converged = jnp.asarray(False)
            exhausted = breakdown & stationary
            method_ok = (
                (status == int(LinearSolveStatus.SUCCESS))
                | (status == int(LinearSolveStatus.MAXIMUM_STEPS_REACHED))
                | exhausted
            )
            force_valid = method_ok
        stability = self._stability(bound, solved, stationary)
        successful = (
            finite
            & solved.fields_successful
            & solved.history_bound
            & method_ok
            & stability.successful
        )
        state = PolarizationState(
            induced_dipoles=solved.induced,
            residual=solved.residual,
            relative_residual=solved.relative,
            iterations=solved.iterations,
            linear_status=status,
            converged=converged,
            stationary=stationary,
            stability_proven=stability.proven,
            stability_lower_bound=stability.lower_bound,
            force_valid=force_valid & successful,
            finite=finite,
            successful=successful,
            plan_id=self.result_plan_id,
            solver_kind=self.plan.kind.value,
            functional=self.plan.functional,
            derivative_mode=self.plan.derivative_mode,
            stability_certificate=stability.certificate,
            stability_assumed=stability.assumed,
        )
        return state, breakdown, stability

    def solve(
        self,
        positions: ArrayLike,
        /,
        *,
        predictor_state: PolarizationPredictorState | None = None,
        cell_vectors: ArrayLike | None = None,
        pairs: AbstractMultipolePairs | None = None,
    ) -> PolarizationSolveResult:
        state = self._require_predictor(predictor_state)
        bound = self.operator.bind(positions, cell_vectors=cell_vectors, pairs=pairs)
        solved = self._solve_bound(bound, state)
        polarization_state, breakdown, stability = self._state(bound, solved)
        u_field, u_ok = self.operator.induced_field(bound, solved.induced)
        action = jnp.where(
            bound.polarizable[:, None],
            self.operator.inverse_polarizability(bound)[:, None] * solved.induced
            - u_field,
            solved.induced,
        )
        fields_finite = (
            jnp.all(jnp.isfinite(action))
            & jnp.all(jnp.isfinite(solved.rhs))
            & jnp.all(jnp.isfinite(solved.direct))
        )
        operator = PolarizationOperatorResult(
            action,
            solved.direct,
            solved.rhs,
            u_field,
            solved.fields_successful & u_ok,
            fields_finite,
            fields_finite & solved.fields_successful & u_ok,
        )
        successful = polarization_state.successful & operator.successful
        candidate = jnp.stack(
            (solved.induced, state.history[0].astype(solved.induced.dtype))
        )
        next_state = PolarizationPredictorState(
            jnp.where(successful, candidate, state.history.astype(candidate.dtype)),
            jnp.where(
                successful, jnp.minimum(state.valid_count + 1, 2), state.valid_count
            ),
            self.prepared_identity,
        )
        return PolarizationSolveResult(
            polarization_state,
            next_state,
            operator,
            stability,
            solved.initial,
            breakdown,
            successful,
        )


class PolarizationDifferentiationEvidence(StrictModule):
    """Which functional was differentiated, by which mode, and its gates.

    ``gate_tolerance`` is the residual gate of the differentiated quantity:
    the force tolerance for energies, the derivative-solve tolerance for
    induced-dipole responses.
    """

    mode: str = eqx.field(static=True)
    functional: str = eqx.field(static=True)
    residual: Array
    gate_tolerance: Array
    stationary: Array
    stability_proven: Array
    stability_assumed: bool = eqx.field(static=True)
    stability_successful: Array
    route_successful: Array
    finite: Array
    successful: Array


class PolarizationEvaluation(StrictModule):
    """Fail-closed energy, forces, cell gradient, solver state and evidence."""

    energy: Array
    forces: Array
    cell_gradient: Array | None
    state: PolarizationState
    predictor_state: PolarizationPredictorState
    successful: Array
    evidence: PolarizationDifferentiationEvidence


def prepared_polarization_energy(
    prepared: PreparedPolarizationSolver,
    positions: ArrayLike,
    /,
    *,
    predictor_state: PolarizationPredictorState | None = None,
    cell_vectors: ArrayLike | None = None,
    pairs: AbstractMultipolePairs | None = None,
) -> tuple[Array, PolarizationSolveResult]:
    """The declared functional at the solution and the solve result.

    The induced dipoles are not detached: PCG differentiates them implicitly
    (stationary functional), TCG algorithmically (TCG functional).
    """
    if not isinstance(prepared, PreparedPolarizationSolver):
        raise TypeError("prepared must be PreparedPolarizationSolver.")
    result = prepared.solve(
        positions,
        predictor_state=predictor_state,
        cell_vectors=cell_vectors,
        pairs=pairs,
    )
    bound = prepared.operator.bind(positions, cell_vectors=cell_vectors, pairs=pairs)
    energy, route_ok = prepared.operator.functional(bound, result.state.induced_dipoles)
    successful = result.successful & route_ok & jnp.isfinite(energy)
    return jnp.where(successful, energy, jnp.nan), result


def evaluate_prepared_polarization(
    prepared: PreparedPolarizationSolver,
    positions: ArrayLike,
    /,
    *,
    predictor_state: PolarizationPredictorState | None = None,
    cell_vectors: ArrayLike | None = None,
    pairs: AbstractMultipolePairs | None = None,
) -> PolarizationEvaluation:
    """Energy, forces and (periodic) cell gradient of the declared functional."""
    if not isinstance(prepared, PreparedPolarizationSolver):
        raise TypeError("prepared must be PreparedPolarizationSolver.")
    coordinate = jnp.asarray(positions)
    periodic = cell_vectors is not None
    vectors = (
        None
        if cell_vectors is None
        else jnp.asarray(cell_vectors, dtype=coordinate.dtype)
    )

    def objective(
        value: Array, cell: Array | None
    ) -> tuple[Array, PolarizationSolveResult]:
        return prepared_polarization_energy(
            prepared,
            value,
            predictor_state=predictor_state,
            cell_vectors=cell,
            pairs=pairs,
        )

    if periodic:
        (energy, result), (position_gradient, cell_gradient) = jax.value_and_grad(
            objective, argnums=(0, 1), has_aux=True
        )(coordinate, vectors)
    else:
        (energy, result), position_gradient = jax.value_and_grad(
            lambda value: objective(value, None), has_aux=True
        )(coordinate)
        cell_gradient = None
    finite = (
        jnp.isfinite(energy)
        & jnp.all(jnp.isfinite(position_gradient))
        & (True if cell_gradient is None else jnp.all(jnp.isfinite(cell_gradient)))
    )
    state = result.state
    successful = (
        state.successful & state.force_valid & result.operator.successful & finite
    )
    evidence = PolarizationDifferentiationEvidence(
        mode=state.derivative_mode,
        functional=state.functional,
        residual=state.residual,
        gate_tolerance=jnp.asarray(prepared.plan.force_tolerance, dtype=energy.dtype),
        stationary=state.stationary,
        stability_proven=state.stability_proven,
        stability_assumed=state.stability_assumed,
        stability_successful=result.stability.successful,
        route_successful=result.operator.successful,
        finite=finite,
        successful=successful,
    )
    return PolarizationEvaluation(
        energy=jnp.where(successful, energy, jnp.nan),
        forces=jnp.where(successful, -position_gradient, jnp.nan),
        cell_gradient=None
        if cell_gradient is None
        else jnp.where(successful, cell_gradient, jnp.nan),
        state=state,
        predictor_state=result.predictor_state,
        successful=successful,
        evidence=evidence,
    )


class PolarizationJVPResult(StrictModule):
    """Induced-dipole response to a coordinate tangent and its evidence."""

    primal: Array
    tangent: Array
    state: PolarizationState
    evidence: PolarizationDifferentiationEvidence
    successful: Array


def evaluate_implicit_polarization_jvp(
    prepared: PreparedPolarizationSolver,
    positions: ArrayLike,
    tangent: ArrayLike,
    /,
    *,
    cell_vectors: ArrayLike | None = None,
    pairs: AbstractMultipolePairs | None = None,
) -> PolarizationJVPResult:
    """``d u / d x [tangent]``: native implicit tangent solve (PCG) or algorithmic (TCG)."""
    if not isinstance(prepared, PreparedPolarizationSolver):
        raise TypeError("prepared must be PreparedPolarizationSolver.")
    coordinate = jnp.asarray(positions)
    direction = jnp.asarray(tangent, dtype=coordinate.dtype)
    if direction.shape != coordinate.shape:
        raise ValueError("tangent must have the same shape as positions.")

    def induced(value: Array) -> tuple[Array, PolarizationSolveResult]:
        result = prepared.solve(value, cell_vectors=cell_vectors, pairs=pairs)
        return result.state.induced_dipoles, result

    primal, derivative, result = jax.jvp(
        induced, (coordinate,), (direction,), has_aux=True
    )
    state = result.state
    finite = jnp.all(jnp.isfinite(derivative)) & jnp.all(jnp.isfinite(primal))
    successful = state.successful & finite
    evidence = PolarizationDifferentiationEvidence(
        mode=state.derivative_mode,
        functional=state.functional,
        residual=state.residual,
        gate_tolerance=jnp.asarray(
            prepared.plan.derivative_tolerance, dtype=coordinate.dtype
        ),
        stationary=state.stationary,
        stability_proven=state.stability_proven,
        stability_assumed=state.stability_assumed,
        stability_successful=result.stability.successful,
        route_successful=result.operator.successful,
        finite=finite,
        successful=successful,
    )
    return PolarizationJVPResult(
        primal, jnp.where(successful, derivative, jnp.nan), state, evidence, successful
    )


# --- potential-program term -------------------------------------------------------


def _checked_polarizability_field(
    reference: AtomisticFieldReference | None, /
) -> AtomisticFieldReference | None:
    if reference is None:
        return None
    if not isinstance(reference, AtomisticFieldReference):
        raise TypeError(
            "polarizability_field must be an AtomisticFieldReference or None."
        )
    if reference.observable is AtomisticObservable.ATOMIC_POLARIZABILITY:
        raise ValueError(
            "ATOMIC_POLARIZABILITY is a reference-scale ratio with no proven absolute "
            "physical unit; it cannot define physical induced dipoles without an "
            "explicit calibrated contract."
        )
    if (
        reference.observable is not AtomisticObservable.POLARIZABILITY
        or reference.level is not AtomisticConditionLevel.ATOM
        or tuple(reference.port.event_shape) != ()
        or reference.unit is None
    ):
        raise ValueError(
            "polarizability_field must reference atom-level scalar POLARIZABILITY "
            "with an explicit physical unit."
        )
    return reference


def _polarizability_factor(
    reference: AtomisticFieldReference, system: PreparedAtomisticSystem, /
) -> float:
    from ..units import derived_unit

    units = system.plan.units
    expected = derived_unit(
        f"{units.charge_unit.symbol}^2*{units.scale.length_unit.symbol}^2/"
        f"{units.scale.energy_unit.symbol}",
        (
            (units.charge_unit, 2),
            (units.scale.length_unit, 2),
            (units.scale.energy_unit, -1),
        ),
    )
    unit = reference.unit
    if unit is None or unit.dimension != expected.dimension:
        raise ValueError(
            "The polarizability field unit must have dimension charge^2 length^2 / energy."
        )
    if unit.reference_system_id != expected.reference_system_id:
        raise ValueError(
            "The polarizability field unit belongs to another reference system."
        )
    return float(unit.scale_to_reference / expected.scale_to_reference)


class PolarizationTermEvidence(StrictModule):
    """Compact evidence of one polarization term evaluation.

    ``prepared_identity`` is the producing prepared term's exact digest as
    dynamic words (``prepared_id`` decodes it on the host).
    """

    functional: str = eqx.field(static=True)
    derivative_mode: str = eqx.field(static=True)
    moment_source: str = eqx.field(static=True)
    polarizability_source: str = eqx.field(static=True)
    prepared_identity: IdentityWords
    inputs_successful: Array
    residual: Array
    iterations: Array
    linear_status: Array
    stationary: Array
    stability_certificate: str = eqx.field(static=True)
    stability_assumed: bool = eqx.field(static=True)
    stability_proven: Array
    stability_lower_bound: Array
    force_valid: Array
    successful: Array

    @property
    def prepared_id(self) -> str:
        """Host decoding of the producing prepared-term digest."""
        return self.prepared_identity.fingerprint()


class InducedDipolePolarizationPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    """Induced-dipole polarization functional as a potential-program term.

    Permanent moments are explicit site data or typed predicted fields;
    polarizabilities are explicit site data or a typed ``POLARIZABILITY`` field
    with an explicit physical unit (``ATOMIC_POLARIZABILITY`` reference ratios
    are refused). The solve runs inside the scalar energy with a deterministic
    guess, so program derivatives are implicit (PCG) or algorithmic (TCG).

    The exact canonical term digest is the dynamic ``term_identity``
    (``term_id`` decodes it on the host); site data, pair scales and widths are
    dynamic leaves with dynamic content identities.
    """

    plan: PolarizationPlan
    multipoles: PermanentMultipoleSiteData
    scaling: PolarizationScaleData | None
    widths: GaussianMultipoleWidths | None
    moment_fields: MultipoleMomentFields | None
    polarizability_field: AtomisticFieldReference | None
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        plan: PolarizationPlan,
        multipoles: PermanentMultipoleSiteData,
        /,
        *,
        scaling: PolarizationScaleData | None = None,
        widths: GaussianMultipoleWidths | None = None,
        moment_fields: MultipoleMomentFields | None = None,
        polarizability_field: AtomisticFieldReference | None = None,
        name: str = "induced-dipole-polarization",
        force_group: int = 0,
    ) -> None:
        if not isinstance(plan, PolarizationPlan):
            raise TypeError("plan must be a PolarizationPlan.")
        if not isinstance(multipoles, PermanentMultipoleSiteData):
            raise TypeError("multipoles must be PermanentMultipoleSiteData.")
        if moment_fields is not None and not isinstance(
            moment_fields, MultipoleMomentFields
        ):
            raise TypeError("moment_fields must be MultipoleMomentFields or None.")
        reference = _checked_polarizability_field(polarizability_field)
        identifier = str(name).strip()
        group = int(force_group)
        if not identifier or group < 0:
            raise ValueError("Term name must be non-empty and force_group non-negative.")
        route = plan.operator.route
        periodic = route.periodic
        dependencies = tuple(
            item
            for item in (
                *(() if moment_fields is None else moment_fields.references),
                reference,
            )
            if item is not None
        )
        neighborhood = (
            not isinstance(route, FiniteMultipoleRoute)
            and route.settings.real_space == "neighborhood"
        )
        self.plan = plan
        self.multipoles = multipoles
        self.scaling = scaling
        self.widths = widths
        self.moment_fields = moment_fields
        self.polarizability_field = reference
        self.name = identifier
        self.force_group = group
        self.capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True,
            finite_geometry=not periodic,
            orthorhombic_periodic=periodic,
            triclinic_periodic=periodic,
            cell_derivative=periodic,
            local_energy=False,
        )
        if neighborhood:
            self.requirements = AtomisticPotentialRequirements(
                reciprocal_grid=periodic,
                neighborhoods=(
                    AtomisticNeighborhoodRequirement(
                        identifier,
                        AtomisticNeighborhoodRole.IMAGE_PAIR_SUM,
                        route.settings.real_cutoff,
                        exclusions=AtomisticPairExclusions.NONE,
                    ),
                ),
                field_dependencies=dependencies,
            )
        else:
            self.requirements = AtomisticPotentialRequirements(
                reciprocal_grid=periodic, field_dependencies=dependencies
            )
        self.term_identity = IdentityWords.of(
            {
                "kind": "induced-dipole-polarization-potential",
                "plan": plan.plan_id,
                "multipoles": multipoles.multipole_id,
                "scaling": None if scaling is None else scaling.scale_id,
                "widths": None if widths is None else widths.widths_id,
                "moment_fields": None
                if moment_fields is None
                else moment_fields.fields_id,
                "polarizability": None if reference is None else reference.reference_id,
                "name": identifier,
                "force_group": group,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedInducedDipolePolarizationPotential":
        return PreparedInducedDipolePolarizationPotential(self, system)


class PreparedInducedDipolePolarizationPotential(AbstractPreparedAtomisticEnergyTerm):
    """Prepared induced-dipole polarization term bound to one atomistic system.

    PyTree layout: static ``name``, ``force_group`` and the declared field
    ``moment_factors``/``polarizability_factor`` (fixed by the declared field
    units and the system unit system, never by material content).
    ``term_identity`` and ``prepared_identity`` are exact canonical digests as
    dynamic words; a context of another prepared system yields a NaN,
    unsuccessful evaluation instead of a traced identity decode. Particle-mesh
    support is the host ``system.particle_discretization()``.
    """

    plan: InducedDipolePolarizationPotential
    system: PreparedAtomisticSystem
    solver: PreparedPolarizationSolver
    moment_factors: tuple[float, float, float] = eqx.field(static=True)
    polarizability_factor: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self, plan: InducedDipolePolarizationPotential, system: PreparedAtomisticSystem, /
    ) -> None:
        if plan.multipoles.site_capacity != system.capacity:
            raise ValueError("Polarization site data does not match the system capacity.")
        if plan.plan.operator.coulomb_constant != system.plan.units.coulomb_constant:
            raise ValueError(
                "The polarization Coulomb constant must equal the system unit constant."
            )
        route = plan.plan.operator.route
        solver = plan.plan.prepare(
            plan.multipoles,
            scaling=plan.scaling,
            active_mask=np.asarray(system.active_mask),
            widths=plan.widths,
            cell=None if isinstance(route, FiniteMultipoleRoute) else system.cell,
            particles=system.particle_discretization()
            if isinstance(route, ParticleMeshMultipoleRoute)
            else None,
        )
        factors = (1.0, 1.0, 1.0)
        fields = plan.moment_fields
        if fields is not None:
            factors = (
                _unit_factor(fields.charges, system, 0),
                _unit_factor(fields.dipoles, system, 1),
                _unit_factor(fields.second_moments, system, 2),
            )
        polar = (
            1.0
            if plan.polarizability_field is None
            else _polarizability_factor(plan.polarizability_field, system)
        )
        self.plan = plan
        self.system = system
        self.solver = solver
        self.moment_factors = factors
        self.polarizability_factor = polar
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-induced-dipole-polarization-potential",
                "term": plan.term_id,
                "system": system.prepared_id,
                "solver": solver.prepared_id,
            }
        )

    def _moments(
        self, context: AtomisticPotentialContext, /
    ) -> tuple[CartesianMultipoles | None, Array, str]:
        fields = self.plan.moment_fields
        if fields is None:
            return None, jnp.asarray(True), "explicit-site-data"
        dtype = context.positions.dtype
        capacity = self.system.capacity
        successful = jnp.asarray(True)
        parts: list[Array] = []
        spherical = fields.basis != "cartesian-primitive"
        for index, (reference, width) in enumerate(
            (
                (fields.charges, 1),
                (fields.dipoles, 3),
                (fields.second_moments, 5 if spherical else 6),
            )
        ):
            if reference is None:
                raise ValueError(
                    "Polarization moment fields must supply every degree explicitly."
                )
            value = context.field(reference)
            if value.available.shape != (1,) or value.values.shape[0] != capacity:
                raise ValueError("Predicted moments must be one atom-level case.")
            values = value.values.astype(dtype).reshape((capacity, width))
            usable = jnp.isfinite(values) & value.available[0] & value.successful[0]
            parts.append(jnp.where(usable, self.moment_factors[index] * values, 0.0))
            successful = successful & value.available[0] & value.successful[0]
        moments = unpack_multipoles(
            jnp.concatenate(parts, axis=-1), fields.basis, 2, trace=fields.trace
        )
        return moments, successful, fields.fields_id

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        bound_system = context.system.prepared_identity.equals(
            self.system.prepared_identity
        )
        dtype = context.positions.dtype
        moments, moments_ok, moment_source = self._moments(context)
        polarizabilities = None
        polar_ok = jnp.asarray(True)
        polar_source = "explicit-site-data"
        if self.plan.polarizability_field is not None:
            field = context.field(self.plan.polarizability_field)
            if field.available.shape != (1,) or field.values.shape != (
                self.system.capacity,
            ):
                raise ValueError(
                    "Predicted polarizabilities must be one atom-level case."
                )
            values = self.polarizability_factor * field.values.astype(dtype)
            polar_ok = (
                field.available[0]
                & field.successful[0]
                & jnp.all(
                    ~self.system.active_mask | (jnp.isfinite(values) & (values >= 0.0))
                )
            )
            polarizabilities = jnp.where(
                jnp.isfinite(values) & polar_ok, jnp.maximum(values, 0.0), 0.0
            )
            polar_source = self.plan.polarizability_field.reference_id
        operator = self.solver.operator
        route = self.plan.plan.operator.route
        pairs = None
        requirement = operator.electrostatics.requirement(self.name)
        if requirement is not None:
            pairs = NeighborhoodPairs(
                context.neighborhood(requirement), self.system.capacity
            )
        bound = operator.bind(
            context.positions,
            cell_vectors=context.cell_vectors if route.periodic else None,
            pairs=pairs,
            moments=moments,
            polarizabilities=polarizabilities,
        )
        state = self.solver.initial_predictor_state()
        solved = self.solver._solve_bound(bound, state)
        polarization_state, _, _ = self.solver._state(bound, solved)
        energy, route_ok = operator.functional(bound, polarization_state.induced_dipoles)
        inputs_ok = moments_ok & polar_ok
        successful = (
            bound_system
            & polarization_state.successful
            & polarization_state.force_valid
            & route_ok
            & inputs_ok
            & jnp.isfinite(energy)
        )
        evidence = PolarizationTermEvidence(
            functional=polarization_state.functional,
            derivative_mode=polarization_state.derivative_mode,
            moment_source=moment_source,
            polarizability_source=polar_source,
            prepared_identity=self.prepared_identity,
            inputs_successful=inputs_ok,
            residual=polarization_state.residual,
            iterations=polarization_state.iterations,
            linear_status=polarization_state.linear_status,
            stationary=polarization_state.stationary,
            stability_certificate=polarization_state.stability_certificate,
            stability_assumed=polarization_state.stability_assumed,
            stability_proven=polarization_state.stability_proven,
            stability_lower_bound=polarization_state.stability_lower_bound,
            force_valid=polarization_state.force_valid,
            successful=successful,
        )
        return AtomisticTermEvaluation(
            jnp.where(successful, energy, jnp.nan),
            jnp.zeros((self.system.capacity,), dtype=energy.dtype),
            successful,
            atom_energy_complete=False,
            evidence=(evidence,),
        )


class ImplicitSolventPlan(StrictModule, NonTrainableState):
    model: str = eqx.field(static=True)
    solvent_dielectric: float = eqx.field(static=True)
    solute_dielectric: float = eqx.field(static=True)
    surface_tension: float = eqx.field(static=True)
    kirkwood_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: str = "gb",
        /,
        *,
        solvent_dielectric: float = 78.5,
        solute_dielectric: float = 1.0,
        surface_tension: float = 0.005,
        kirkwood_factor: float = 2.455,
    ) -> None:
        if (
            model not in ("gb", "gk")
            or min(solvent_dielectric, solute_dielectric, kirkwood_factor) <= 0
            or not np.isfinite(
                [solvent_dielectric, solute_dielectric, surface_tension, kirkwood_factor]
            ).all()
            or surface_tension < 0
        ):
            raise ValueError("Implicit-solvent parameters are invalid.")
        (
            self.model,
            self.solvent_dielectric,
            self.solute_dielectric,
            self.surface_tension,
            self.kirkwood_factor,
        ) = (
            model,
            float(solvent_dielectric),
            float(solute_dielectric),
            float(surface_tension),
            float(kirkwood_factor),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "implicit-solvent",
                "model": model,
                "solvent_dielectric": self.solvent_dielectric,
                "solute_dielectric": self.solute_dielectric,
                "surface_tension": self.surface_tension,
                "kirkwood_factor": self.kirkwood_factor,
            }
        )

    def energy(
        self,
        positions: ArrayLike,
        charges: ArrayLike,
        radii: ArrayLike,
        coulomb_constant: float,
        /,
    ) -> Array:
        coordinate, charge, radius = (
            jnp.asarray(positions),
            jnp.asarray(charges),
            jnp.asarray(radii),
        )
        count = coordinate.shape[0]
        if (
            coordinate.shape != (count, 3)
            or charge.shape != (count,)
            or radius.shape != (count,)
            or not np.isfinite(coulomb_constant)
            or float(coulomb_constant) <= 0.0
        ):
            raise ValueError("Implicit-solvent arrays or Coulomb constant are invalid.")
        valid = (
            jnp.all(jnp.isfinite(coordinate))
            & jnp.all(jnp.isfinite(charge))
            & jnp.all(jnp.isfinite(radius) & (radius > 0.0))
        )
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        displacement = coordinate[:, None, :] - coordinate[None, :, :]
        distance2 = jnp.sum(displacement**2, axis=-1)
        radius_product = safe_radius[:, None] * safe_radius[None, :]
        denominator = 4.0 if self.model == "gb" else self.kirkwood_factor
        effective_distance = jnp.sqrt(
            distance2
            + radius_product * jnp.exp(-distance2 / (denominator * radius_product))
        )
        dielectric_factor = 1.0 / self.solvent_dielectric - 1.0 / self.solute_dielectric
        polar = (
            0.5
            * coulomb_constant
            * dielectric_factor
            * jnp.sum(charge[:, None] * charge[None, :] / effective_distance)
        )
        area = 4.0 * jnp.pi * jnp.sum(safe_radius**2)
        energy = polar + self.surface_tension * area
        return jnp.where(valid & jnp.isfinite(energy), energy, jnp.nan)


__all__ = [
    "ExponentialDampedCoulombKernel",
    "ImplicitSolventPlan",
    "InducedDipolePolarizationPotential",
    "PermanentMultipoleSiteData",
    "PolarizationStabilityCertificate",
    "PolarizationStabilityEvidence",
    "PolarizationStabilityPolicy",
    "PolarizationCurvatureProbe",
    "PolarizationDamping",
    "PolarizationDifferentiationEvidence",
    "PolarizationEvaluation",
    "PolarizationJVPResult",
    "PolarizationOperatorPlan",
    "PolarizationOperatorResult",
    "PolarizationPlan",
    "PolarizationPreconditionerKind",
    "PolarizationPreconditionerPlan",
    "PolarizationPreconditionerResult",
    "PolarizationPredictorPlan",
    "PolarizationPredictorState",
    "PolarizationScaleData",
    "PolarizationSiteKind",
    "PolarizationSolveResult",
    "PolarizationSolverKind",
    "PolarizationSolverPlan",
    "PolarizationState",
    "PolarizationTermEvidence",
    "PreparedInducedDipolePolarizationPotential",
    "PreparedPolarizationOperator",
    "PreparedPolarizationPreconditioner",
    "PreparedPolarizationSolver",
    "evaluate_implicit_polarization_jvp",
    "evaluate_prepared_polarization",
    "prepared_polarization_energy",
]
