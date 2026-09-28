#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared step status and evidence for surface thin-film routes."""

from __future__ import annotations

from enum import IntEnum

import jax.numpy as jnp
from jax import Array

from .._strict import StrictModule


class FilmStepStatus(IntEnum):
    """Terminal status of one film step; only ``ACCEPTED`` commits a candidate.

    Resolution order follows the enum order after ``ACCEPTED``: the first
    failed premise is reported.
    """

    ACCEPTED = 0
    INADMISSIBLE_INPUT = 1
    INADMISSIBLE_CONDUCTANCE = 2
    GEOMETRY_MISMATCH = 3
    COURANT_LIMIT = 4
    SOLVE_FAILED = 5
    NONFINITE = 6
    POSITIVITY_VIOLATED = 7
    CAPACITY_EXCEEDED = 8
    NONPOSITIVE_TENSION = 9


def resolve_film_status(*conditions: tuple[IntEnum, Array]) -> Array:
    """Return the first failed status among ``(status, failed)`` pairs.

    Film status enums (``FilmStepStatus``, ``SurfaceMotionStatus``) share the
    code ``0`` for ``ACCEPTED``.
    """
    status = jnp.asarray(FilmStepStatus.ACCEPTED, dtype=jnp.int32)
    for code, failed in reversed(conditions):
        status = jnp.where(failed, jnp.asarray(code, dtype=jnp.int32), status)
    return status


class SurfaceFilmEvidence(StrictModule):
    """Conservation, admissibility, solver and energy evidence of one film step.

    ``liquid_volume_residual_m3`` is the total-content change minus the
    declared boundary exchange; it is roundoff for flux-form steps.
    ``positivity_guaranteed`` holds only when conductances are admissible,
    the boundary policy is admitted, the positive-domain nonlinear solve
    converged, and the committed content is positive. ``dissipation_guaranteed``
    reports whether the discrete energy decay follows from the proven premises
    of the route (for example convex energy with admissible conductances); the
    observed ``energy_change_j`` is always reported separately.
    """

    liquid_volume_residual_m3: Array
    boundary_exchange_m3: Array
    minimum_thickness_m: Array
    rupture_mask: Array
    energy_change_j: Array
    dissipation_guaranteed: Array
    positivity_guaranteed: Array
    conductance_admissible: Array
    nonlinear_status: Array
    nonlinear_iterations: Array
    nonlinear_residual_norm: Array
    converged: Array
    finite: Array
    geometry_revision: Array

    def __init__(
        self,
        *,
        liquid_volume_residual_m3: Array,
        boundary_exchange_m3: Array,
        minimum_thickness_m: Array,
        rupture_mask: Array,
        energy_change_j: Array,
        dissipation_guaranteed: Array,
        positivity_guaranteed: Array,
        conductance_admissible: Array,
        nonlinear_status: Array,
        nonlinear_iterations: Array,
        nonlinear_residual_norm: Array,
        converged: Array,
        finite: Array,
        geometry_revision: Array,
    ) -> None:
        self.liquid_volume_residual_m3 = jnp.asarray(liquid_volume_residual_m3)
        self.boundary_exchange_m3 = jnp.asarray(boundary_exchange_m3)
        self.minimum_thickness_m = jnp.asarray(minimum_thickness_m)
        self.rupture_mask = jnp.asarray(rupture_mask, dtype=jnp.bool_)
        self.energy_change_j = jnp.asarray(energy_change_j)
        self.dissipation_guaranteed = jnp.asarray(dissipation_guaranteed, dtype=jnp.bool_)
        self.positivity_guaranteed = jnp.asarray(positivity_guaranteed, dtype=jnp.bool_)
        self.conductance_admissible = jnp.asarray(conductance_admissible, dtype=jnp.bool_)
        self.nonlinear_status = jnp.asarray(nonlinear_status, dtype=jnp.int32)
        self.nonlinear_iterations = jnp.asarray(nonlinear_iterations, dtype=jnp.int32)
        self.nonlinear_residual_norm = jnp.asarray(nonlinear_residual_norm)
        self.converged = jnp.asarray(converged, dtype=jnp.bool_)
        self.finite = jnp.asarray(finite, dtype=jnp.bool_)
        self.geometry_revision = jnp.asarray(geometry_revision, dtype=jnp.int32)


__all__ = ["FilmStepStatus", "SurfaceFilmEvidence", "resolve_film_status"]
