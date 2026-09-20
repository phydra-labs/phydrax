#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit apparent-horizon search, certification, and conservative tracking."""

from __future__ import annotations

from enum import IntEnum
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix._spacetime_conventions import RelativityConvention
from ._mots import MOTSSolvePlan, MOTSSolveResult
from ._surfaces import (
    SphericalSpectralSurface,
    SphericalSurfacePlan,
    SurfaceGeometryEvidence,
)


class ApparentHorizonSearchStatus(IntEnum):
    INVALID = 0
    INCOMPLETE = 1
    NO_SURFACE = 2
    UNIQUE_SURFACE = 3
    MULTIPLE_SURFACES = 4
    NON_NESTED_SURFACES = 5


class HorizonTrackingStatus(IntEnum):
    ACCEPTED_NEEDS_RECERTIFICATION = 0
    INACTIVE = 1
    NONMONOTONE_TIME = 2
    MOTS_FAILED = 3
    GEOMETRY_INVALID = 4
    TRACKING_JUMP = 5


class MOTSCandidateBatch(StrictModule):
    """Fixed-capacity numerical evidence for all search seeds."""

    coefficients: Array
    centers: Array
    radius_samples: Array
    residual_norm: Array
    iterations: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    stable: Array
    status: Array
    active: Array
    excluded: Array
    plan_id: str = eqx.field(static=True)


class OutermostSearchEvidence(StrictModule):
    representative: Array
    pairwise_equivalent: Array
    pairwise_contains: Array
    outermost_mask: Array
    found_count: Array
    outermost_count: Array
    outermost_index: Array
    search_complete: Array
    attempts_resolved: Array
    no_surface_certified: Array
    outermost_certified: Array
    finite: Array


class ApparentHorizonResult(StrictModule):
    """An apparent-horizon result only when ``certified`` is true."""

    surface: SphericalSpectralSurface
    coordinate_geometry: SurfaceGeometryEvidence
    candidates: MOTSCandidateBatch
    search: OutermostSearchEvidence
    finite: Array
    found: Array
    certified: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class HorizonGeometryEvidence(StrictModule):
    area: Array
    areal_radius: Array
    irreducible_mass: Array
    angular_momentum: Array
    christodoulou_mass: Array
    dimensionless_spin: Array
    axial_tangency_error: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    convention_id: str = eqx.field(static=True)


class KerrHorizonReference(StrictModule):
    horizon_radius: Array
    area: Array
    irreducible_mass: Array
    angular_momentum: Array
    christodoulou_mass: Array
    dimensionless_spin: Array
    finite: Array
    physically_valid: Array


class ApparentHorizonSearchPlan(StrictModule, NonTrainableState):
    """Multi-start MOTS search with fixed capacity and explicit completeness input.

    A converged MOTS is retained as a candidate only.  Certification additionally
    requires complete search evidence, resolved attempts, a unique enclosing stable
    representative, and finite nested surfaces.
    """

    mots_plan: MOTSSolvePlan
    seed_radii: Array
    active: Array
    equivalence_tolerance: float = eqx.field(static=True)
    nesting_tolerance: float = eqx.field(static=True)
    candidate_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mots_plan: MOTSSolvePlan,
        seed_radii: ArrayLike,
        /,
        *,
        active: ArrayLike | None = None,
        equivalence_tolerance: float = 1.0e-5,
        nesting_tolerance: float = 1.0e-6,
    ):
        if not isinstance(mots_plan, MOTSSolvePlan):
            raise TypeError("mots_plan must be a MOTSSolvePlan.")
        radii = np.asarray(seed_radii, dtype=np.float64).reshape((-1,))
        active_ = (
            np.ones(radii.shape, dtype=np.bool_)
            if active is None
            else np.asarray(active, dtype=np.bool_)
        )
        equivalence = float(equivalence_tolerance)
        nesting = float(nesting_tolerance)
        if (
            radii.size < 1
            or active_.shape != radii.shape
            or not np.any(active_)
            or np.any(~np.isfinite(radii))
            or np.any(radii <= 0.0)
            or equivalence <= 0.0
            or nesting <= 0.0
        ):
            raise ValueError("Apparent-horizon seed capacity or tolerances are invalid.")
        self.mots_plan = mots_plan
        self.seed_radii = jnp.asarray(radii)
        self.active = jnp.asarray(active_)
        self.equivalence_tolerance = equivalence
        self.nesting_tolerance = nesting
        self.candidate_capacity = radii.size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-apparent-horizon-search",
                "mots": mots_plan.plan_id,
                "seed_radii": array_tree_fingerprint(radii),
                "active": array_tree_fingerprint(active_),
                "equivalence_tolerance": equivalence,
                "nesting_tolerance": nesting,
            }
        )

    @property
    def surface_plan(self) -> SphericalSurfacePlan:
        return self.mots_plan.surface_plan

    def search(
        self,
        expansion_operator: Callable[[SphericalSpectralSurface], ArrayLike],
        /,
        *,
        center: ArrayLike = (0.0, 0.0, 0.0),
        excluded: ArrayLike | None = None,
        search_complete: ArrayLike = False,
    ) -> ApparentHorizonResult:
        """Run every fixed seed and apply explicit outermost certification evidence.

        ``excluded[i]`` means independent search logic certified that the region
        represented by seed ``i`` contains no MOTS.  A failed nonlinear solve is
        never interpreted as exclusion.
        """
        center_ = jnp.asarray(center, dtype=self.seed_radii.dtype)
        if center_.shape != (3,):
            raise ValueError("Search center must have shape (3,).")
        excluded_ = (
            jnp.zeros((self.candidate_capacity,), dtype=jnp.bool_)
            if excluded is None
            else jnp.asarray(excluded, dtype=jnp.bool_)
        )
        if excluded_.shape != (self.candidate_capacity,):
            raise ValueError("Exclusion evidence must match candidate capacity.")

        results = []
        for index in range(self.candidate_capacity):
            seed_radius = jnp.where(
                self.active[index], self.seed_radii[index], self.seed_radii[0]
            )
            initial = self.surface_plan.constant(seed_radius, center=center_)
            results.append(self.mots_plan.solve(initial, expansion_operator))

        coefficients = jnp.stack(tuple(result.surface.coefficients for result in results))
        centers = jnp.stack(tuple(result.surface.center for result in results))
        radii = jnp.stack(
            tuple(self.surface_plan.radius(result.surface) for result in results)
        )
        residual_norm = jnp.stack(tuple(result.residual_norm for result in results))
        iterations = jnp.stack(tuple(result.iterations for result in results))
        finite = jnp.stack(tuple(result.finite for result in results))
        converged = jnp.stack(tuple(result.converged for result in results))
        physically_valid = jnp.stack(tuple(result.physically_valid for result in results))
        qualified = jnp.stack(tuple(result.qualified for result in results))
        derivative_valid = jnp.stack(tuple(result.derivative_valid for result in results))
        stable = jnp.stack(tuple(result.stability.stable for result in results))
        statuses = jnp.stack(tuple(result.status for result in results))
        qualified = self.active & qualified
        effective_excluded = self.active & excluded_ & ~qualified
        batch = MOTSCandidateBatch(
            coefficients,
            centers,
            radii,
            residual_norm,
            iterations,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            stable,
            statuses,
            self.active,
            effective_excluded,
            self.plan_id,
        )

        flat = radii.reshape((self.candidate_capacity, -1))
        difference = jnp.max(jnp.abs(flat[:, None, :] - flat[None, :, :]), axis=-1)
        maximum_radius = jnp.max(jnp.abs(flat), axis=-1)
        scale = jnp.maximum(
            jnp.maximum(maximum_radius[:, None], maximum_radius[None, :]),
            1.0,
        )
        equivalent = difference <= self.equivalence_tolerance * scale
        indices = jnp.arange(self.candidate_capacity)
        earlier = indices[None, :] < indices[:, None]
        representative = qualified & ~jnp.any(
            earlier & qualified[None, :] & equivalent, axis=1
        )
        containment_margin = self.nesting_tolerance * scale[..., None]
        contains = jnp.all(
            flat[:, None, :] >= flat[None, :, :] - containment_margin, axis=-1
        )
        contains_every_representative = jnp.all(
            ~representative[None, :] | contains, axis=1
        )
        outermost_mask = representative & contains_every_representative
        found_count = jnp.sum(representative.astype(jnp.int32))
        outermost_count = jnp.sum(outermost_mask.astype(jnp.int32))
        mean_radius = jnp.mean(flat, axis=-1)
        outermost_index = jnp.argmax(
            jnp.where(outermost_mask, mean_radius, -jnp.inf)
        ).astype(jnp.int32)
        complete = jnp.asarray(search_complete, dtype=jnp.bool_).reshape(())
        attempts_resolved = jnp.all(~self.active | qualified | effective_excluded)
        search_finite = jnp.all(~self.active | finite | effective_excluded)
        no_surface_certified = (
            complete
            & attempts_resolved
            & search_finite
            & (found_count == 0)
            & jnp.all(~self.active | effective_excluded)
        )
        selected_stable = stable[outermost_index]
        outermost_certified = (
            complete
            & attempts_resolved
            & search_finite
            & (outermost_count == 1)
            & selected_stable
        )
        search_evidence = OutermostSearchEvidence(
            representative,
            equivalent,
            contains,
            outermost_mask,
            found_count,
            outermost_count,
            outermost_index,
            complete,
            attempts_resolved,
            no_surface_certified,
            outermost_certified,
            search_finite,
        )
        selected = SphericalSpectralSurface(
            coefficients[outermost_index],
            centers[outermost_index],
            self.surface_plan.plan_id,
        )
        geometry = self.surface_plan.geometry(selected)
        found = found_count > 0
        certified = outermost_certified
        status = jnp.select(
            (
                ~search_finite,
                ~complete | ~attempts_resolved,
                found_count == 0,
                outermost_count != 1,
                found_count == 1,
            ),
            (
                int(ApparentHorizonSearchStatus.INVALID),
                int(ApparentHorizonSearchStatus.INCOMPLETE),
                int(ApparentHorizonSearchStatus.NO_SURFACE),
                int(ApparentHorizonSearchStatus.NON_NESTED_SURFACES),
                int(ApparentHorizonSearchStatus.UNIQUE_SURFACE),
            ),
            default=int(ApparentHorizonSearchStatus.MULTIPLE_SURFACES),
        ).astype(jnp.int32)
        return ApparentHorizonResult(
            selected,
            geometry,
            batch,
            search_evidence,
            search_finite,
            found,
            certified,
            certified & geometry.physically_valid,
            certified & geometry.derivative_valid & derivative_valid[outermost_index],
            status,
            self.plan_id,
        )


class HorizonTrackingState(StrictModule):
    surface: SphericalSpectralSurface
    area: Array
    time: Array
    step_index: Array
    active: Array
    certified: Array
    plan_id: str = eqx.field(static=True)


class HorizonTrackingStep(StrictModule):
    candidate_mots: MOTSSolveResult
    candidate_geometry: SurfaceGeometryEvidence
    state: HorizonTrackingState
    accepted: Array
    needs_recertification: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class HorizonTrackerPlan(StrictModule, NonTrainableState):
    """Predict from the last surface and gate jumps; never certify tracked MOTSs."""

    mots_plan: MOTSSolvePlan
    maximum_relative_area_change: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mots_plan: MOTSSolvePlan,
        /,
        *,
        maximum_relative_area_change: float = 0.25,
    ):
        if not isinstance(mots_plan, MOTSSolvePlan):
            raise TypeError("mots_plan must be a MOTSSolvePlan.")
        maximum_change = float(maximum_relative_area_change)
        if maximum_change <= 0.0:
            raise ValueError("maximum_relative_area_change must be positive.")
        self.mots_plan = mots_plan
        self.maximum_relative_area_change = maximum_change
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mots-horizon-tracker",
                "mots": mots_plan.plan_id,
                "maximum_relative_area_change": maximum_change,
            }
        )

    def initialize(
        self, horizon: ApparentHorizonResult, time: ArrayLike = 0.0, /
    ) -> HorizonTrackingState:
        time_ = jnp.asarray(time).reshape(())
        active = horizon.certified & horizon.coordinate_geometry.physically_valid
        return HorizonTrackingState(
            horizon.surface,
            horizon.coordinate_geometry.area,
            time_,
            jnp.asarray(0, dtype=jnp.int32),
            active,
            horizon.certified,
            self.plan_id,
        )

    def advance(
        self,
        previous: HorizonTrackingState,
        time: ArrayLike,
        expansion_operator: Callable[[SphericalSpectralSurface], ArrayLike],
        /,
        *,
        spatial_metric: ArrayLike | None = None,
    ) -> HorizonTrackingStep:
        if previous.plan_id != self.plan_id:
            raise ValueError("Tracking state and tracker plan identities differ.")
        time_ = jnp.asarray(time, dtype=previous.time.dtype).reshape(())
        candidate = self.mots_plan.solve(previous.surface, expansion_operator)
        geometry = self.mots_plan.surface_plan.geometry(candidate.surface, spatial_metric)
        relative_area_change = jnp.abs(geometry.area - previous.area) / jnp.maximum(
            jnp.abs(previous.area), jnp.finfo(geometry.area.dtype).tiny
        )
        monotone_time = time_ > previous.time
        jump_valid = relative_area_change <= self.maximum_relative_area_change
        accepted = (
            previous.active
            & monotone_time
            & candidate.qualified
            & geometry.physically_valid
            & jump_valid
        )
        coefficients = jnp.where(
            accepted, candidate.surface.coefficients, previous.surface.coefficients
        )
        center = jnp.where(accepted, candidate.surface.center, previous.surface.center)
        surface = SphericalSpectralSurface(
            coefficients, center, self.mots_plan.surface_plan.plan_id
        )
        state = HorizonTrackingState(
            surface,
            jnp.where(accepted, geometry.area, previous.area),
            jnp.where(accepted, time_, previous.time),
            previous.step_index + accepted.astype(jnp.int32),
            previous.active,
            jnp.where(accepted, False, previous.certified),
            self.plan_id,
        )
        status = jnp.select(
            (
                ~previous.active,
                ~monotone_time,
                ~candidate.qualified,
                ~geometry.physically_valid,
                ~jump_valid,
            ),
            (
                int(HorizonTrackingStatus.INACTIVE),
                int(HorizonTrackingStatus.NONMONOTONE_TIME),
                int(HorizonTrackingStatus.MOTS_FAILED),
                int(HorizonTrackingStatus.GEOMETRY_INVALID),
                int(HorizonTrackingStatus.TRACKING_JUMP),
            ),
            default=int(HorizonTrackingStatus.ACCEPTED_NEEDS_RECERTIFICATION),
        ).astype(jnp.int32)
        finite = candidate.finite & geometry.finite
        physically_valid = candidate.physically_valid & geometry.physically_valid
        return HorizonTrackingStep(
            candidate,
            geometry,
            state,
            accepted,
            accepted,
            finite,
            physically_valid,
            accepted,
            accepted & candidate.derivative_valid & geometry.derivative_valid,
            status,
            self.plan_id,
        )


def quasilocal_horizon_geometry(
    geometry: SurfaceGeometryEvidence,
    extrinsic_curvature: ArrayLike,
    axial_vector: ArrayLike,
    /,
    *,
    convention: RelativityConvention | None = None,
    tangency_tolerance: float = 1.0e-6,
) -> HorizonGeometryEvidence:
    """Area, irreducible mass, and isolated-horizon angular momentum evidence."""
    convention_ = RelativityConvention() if convention is None else convention
    if not isinstance(convention_, RelativityConvention):
        raise TypeError("convention must be a RelativityConvention or None.")
    curvature = jnp.asarray(extrinsic_curvature)
    axial = jnp.asarray(axial_vector, dtype=curvature.dtype)
    if curvature.shape != geometry.radius.shape + (3, 3):
        raise ValueError("Extrinsic-curvature samples do not match surface geometry.")
    if axial.shape != geometry.radius.shape + (3,):
        raise ValueError("Axial-vector samples do not match surface geometry.")
    integrand = ein.contract(
        "...i,...ij,...j->...", geometry.outward_normal, curvature, axial
    )
    sign_to_standard_k = -float(convention_.extrinsic_curvature_sign)
    angular_momentum = (
        sign_to_standard_k * jnp.sum(geometry.area_weights * integrand) / (8.0 * jnp.pi)
    )
    area = geometry.area
    irreducible_mass = jnp.sqrt(jnp.maximum(area, 0.0) / (16.0 * jnp.pi))
    safe_irreducible_mass = jnp.where(irreducible_mass > 0.0, irreducible_mass, 1.0)
    christodoulou_mass = jnp.sqrt(
        irreducible_mass**2 + angular_momentum**2 / (4.0 * safe_irreducible_mass**2)
    )
    safe_mass = jnp.where(christodoulou_mass > 0.0, christodoulou_mass, 1.0)
    dimensionless_spin = angular_momentum / safe_mass**2
    tangency = jnp.max(
        jnp.abs(ein.contract("...i,...i->...", geometry.outward_normal_covector, axial))
    )
    finite = (
        geometry.finite
        & jnp.all(jnp.isfinite(curvature))
        & jnp.all(jnp.isfinite(axial))
        & jnp.isfinite(angular_momentum)
        & jnp.isfinite(christodoulou_mass)
    )
    physically_valid = (
        geometry.physically_valid
        & finite
        & (irreducible_mass > 0.0)
        & (tangency <= float(tangency_tolerance))
        & (jnp.abs(dimensionless_spin) <= 1.0 + 1.0e-6)
    )
    return HorizonGeometryEvidence(
        area,
        geometry.areal_radius,
        irreducible_mass,
        angular_momentum,
        christodoulou_mass,
        dimensionless_spin,
        tangency,
        finite,
        physically_valid,
        physically_valid,
        geometry.derivative_valid & finite,
        convention_.convention_id,
    )


def kerr_horizon_reference(
    mass: ArrayLike, spin_parameter: ArrayLike, /
) -> KerrHorizonReference:
    """Analytic Kerr outer-horizon geometry in geometric units, ``J = a M``."""
    mass_ = jnp.asarray(mass)
    spin = jnp.asarray(spin_parameter, dtype=mass_.dtype)
    discriminant = mass_**2 - spin**2
    horizon_radius = mass_ + jnp.sqrt(jnp.maximum(discriminant, 0.0))
    area = 4.0 * jnp.pi * (horizon_radius**2 + spin**2)
    irreducible_mass = jnp.sqrt(jnp.maximum(area, 0.0) / (16.0 * jnp.pi))
    angular_momentum = mass_ * spin
    safe_irreducible_mass = jnp.where(irreducible_mass > 0.0, irreducible_mass, 1.0)
    christodoulou_mass = jnp.sqrt(
        irreducible_mass**2 + angular_momentum**2 / (4.0 * safe_irreducible_mass**2)
    )
    dimensionless_spin = spin / jnp.where(mass_ != 0.0, mass_, 1.0)
    finite = jnp.all(
        jnp.isfinite(
            jnp.stack(
                (
                    mass_,
                    spin,
                    horizon_radius,
                    area,
                    irreducible_mass,
                    angular_momentum,
                    christodoulou_mass,
                    dimensionless_spin,
                )
            )
        )
    )
    physically_valid = finite & (mass_ > 0.0) & (jnp.abs(spin) <= mass_)
    return KerrHorizonReference(
        horizon_radius,
        area,
        irreducible_mass,
        angular_momentum,
        christodoulou_mass,
        dimensionless_spin,
        finite,
        physically_valid,
    )


__all__ = [
    "ApparentHorizonResult",
    "ApparentHorizonSearchPlan",
    "ApparentHorizonSearchStatus",
    "HorizonGeometryEvidence",
    "HorizonTrackerPlan",
    "HorizonTrackingState",
    "HorizonTrackingStatus",
    "HorizonTrackingStep",
    "KerrHorizonReference",
    "MOTSCandidateBatch",
    "OutermostSearchEvidence",
    "kerr_horizon_reference",
    "quasilocal_horizon_geometry",
]
