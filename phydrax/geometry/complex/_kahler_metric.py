#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...linalg import FactorizationPolicy, inverse, OperatorProperties
from ...metrix import KahlerPotentialGeometry
from ._hypersurface import ProjectiveHypersurface
from ._hypersurface_patch import HypersurfacePatchGeometry


class HypersurfaceKahlerEvaluation(StrictModule):
    metric: Array
    inverse_metric: Array
    log_determinant: Array
    target_log_volume: Array
    monge_ampere_residual: Array
    positivity_margin: Array
    potential: Array
    valid: Array
    chart_index: int
    pivot_index: int

    def __init__(
        self,
        *,
        metric: ArrayLike,
        inverse_metric: ArrayLike,
        log_determinant: ArrayLike,
        target_log_volume: ArrayLike,
        monge_ampere_residual: ArrayLike,
        positivity_margin: ArrayLike,
        potential: ArrayLike,
        valid: ArrayLike,
        chart_index: int,
        pivot_index: int,
    ):
        self.metric = jnp.asarray(metric)
        self.inverse_metric = jnp.asarray(inverse_metric)
        self.log_determinant = jnp.asarray(log_determinant)
        self.target_log_volume = jnp.asarray(target_log_volume)
        self.monge_ampere_residual = jnp.asarray(monge_ampere_residual)
        self.positivity_margin = jnp.asarray(positivity_margin)
        self.potential = jnp.asarray(potential)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.chart_index = int(chart_index)
        self.pivot_index = int(pivot_index)


class HypersurfaceKahlerGeometry(StrictModule):
    """Kähler-potential metric evaluated on fixed projective hypersurface patches."""

    hypersurface: ProjectiveHypersurface
    potential_function: Callable[[Array], Array]
    normalization: Array
    positivity_floor: float

    def __init__(
        self,
        hypersurface: ProjectiveHypersurface,
        potential: Callable[[Array], Array],
        /,
        *,
        normalization: ArrayLike = 0.0,
        positivity_floor: float = 1e-8,
    ):
        if not isinstance(hypersurface, ProjectiveHypersurface):
            raise TypeError("hypersurface must be a ProjectiveHypersurface.")
        if not callable(potential):
            raise TypeError("potential must be callable.")
        self.hypersurface = hypersurface
        self.potential_function = potential
        self.normalization = jnp.asarray(normalization).reshape(())
        self.positivity_floor = float(positivity_floor)

    def evaluate(
        self,
        homogeneous_point: ArrayLike,
        /,
        *,
        chart_index: int | None = None,
        pivot_index: int | None = None,
    ) -> HypersurfaceKahlerEvaluation:
        point = jnp.asarray(homogeneous_point)
        patch = HypersurfacePatchGeometry(self.hypersurface).evaluate(
            point, chart_index=chart_index, pivot_index=pivot_index
        )
        owner = patch.chart_index
        convention = self.hypersurface.atlas.conventions[owner]

        def local_potential(affine_coordinates: Array) -> Array:
            homogeneous = self.hypersurface.homogeneous_coordinates(
                owner, affine_coordinates
            )
            return jnp.asarray(self.potential_function(homogeneous)).reshape(())

        local_geometry = KahlerPotentialGeometry(
            self.hypersurface.atlas.metric(owner), convention, local_potential
        )
        ambient_metric = local_geometry.metric()(patch.affine_coordinates)
        induced = patch.tangent_basis.T @ ambient_metric @ patch.tangent_basis
        inverse_result = inverse(
            induced,
            FactorizationPolicy("cholesky"),
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "asserted",
                },
            ),
        )
        eigenvalues = jnp.linalg.eigvalsh(induced)
        margin = jnp.min(eigenvalues)
        log_determinant = 0.5 * jnp.linalg.slogdet(induced)[1]
        target_log_volume = 2.0 * jnp.log(jnp.abs(patch.residue_coefficient))
        residual = log_determinant - target_log_volume - self.normalization
        valid = (
            patch.valid
            & jnp.all(jnp.isfinite(induced))
            & (margin > self.positivity_floor)
            & inverse_result.successful
            & jnp.isfinite(residual)
        )
        return HypersurfaceKahlerEvaluation(
            metric=induced,
            inverse_metric=inverse_result.value,
            log_determinant=log_determinant,
            target_log_volume=target_log_volume,
            monge_ampere_residual=residual,
            positivity_margin=margin,
            potential=local_potential(patch.affine_coordinates),
            valid=valid,
            chart_index=owner,
            pivot_index=patch.pivot_index,
        )


__all__ = ["HypersurfaceKahlerEvaluation", "HypersurfaceKahlerGeometry"]
