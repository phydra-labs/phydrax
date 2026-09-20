#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._core import AbstractLayerKernel, BoundaryPanelization2D, KernelActionSide


class StokesLayerKernel2D(AbstractLayerKernel):
    """Steady two-dimensional Stokeslet and stresslet kernels."""

    viscosity: float = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(self, viscosity: float, /):
        value = float(viscosity)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("viscosity must be finite and positive.")
        self.viscosity = value
        self._kernel_id = canonical_fingerprint(
            {"kind": "stokes-layer-kernel-2d", "viscosity": value}
        )

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def source_event_shape(self) -> tuple[int, ...]:
        return (2,)

    @property
    def target_event_shape(self) -> tuple[int, ...]:
        return (2,)

    @property
    def action_side(self) -> KernelActionSide:
        return "left"

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    def value(self, target: Array, source: Array, /) -> Array:
        difference = jnp.asarray(target) - jnp.asarray(source)
        radius_squared = jnp.sum(difference * difference)
        radius = jnp.sqrt(radius_squared)
        identity = jnp.eye(2, dtype=difference.dtype)
        return (
            -jnp.log(radius) * identity
            + ein.contract("i,j->ij", difference, difference) / radius_squared
        ) / (4.0 * jnp.pi * self.viscosity)

    def source_normal_derivative(
        self,
        target: Array,
        source: Array,
        source_normal: Array,
        /,
    ) -> Array:
        difference = jnp.asarray(target) - jnp.asarray(source)
        radius_squared = jnp.sum(difference * difference)
        normal_projection = jnp.dot(difference, jnp.asarray(source_normal))
        return (
            -ein.contract("i,j->ij", difference, difference)
            * normal_projection
            / (jnp.pi * radius_squared**2)
        )


class StokesLayerPotential2D(StrictModule, NonTrainableState):
    """Prepared off-surface planar Stokes layer potential."""

    panelization: BoundaryPanelization2D
    kernel: StokesLayerKernel2D
    density: Array
    minimum_clearance: float = eqx.field(static=True)
    kind: Literal["single", "double"] = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        panelization: BoundaryPanelization2D,
        density: ArrayLike,
        /,
        *,
        viscosity: float,
        kind: Literal["single", "double"] = "single",
        minimum_clearance: float,
    ):
        if not isinstance(panelization, BoundaryPanelization2D):
            raise TypeError("panelization must be BoundaryPanelization2D.")
        values = jnp.asarray(density, dtype=jnp.float64)
        if values.shape != (panelization.node_count, 2):
            raise ValueError("density must have shape (node_count, 2).")
        if kind not in ("single", "double"):
            raise ValueError("kind must be 'single' or 'double'.")
        clearance = float(minimum_clearance)
        if not np.isfinite(clearance) or clearance <= 0.0:
            raise ValueError("minimum_clearance must be finite and positive.")
        kernel = StokesLayerKernel2D(viscosity)
        self.panelization = panelization
        self.kernel = kernel
        self.density = values
        self.minimum_clearance = clearance
        self.kind = kind
        self.representation_id = canonical_fingerprint(
            {
                "kind": "stokes-layer-potential-2d",
                "panelization": panelization.panelization_id,
                "kernel": kernel.kernel_id,
                "layer": kind,
                "clearance": clearance,
            }
        )

    def __call__(self, target: ArrayLike, /) -> Array:
        point = jnp.asarray(target, dtype=self.panelization.points.dtype)
        if point.shape != (2,):
            raise ValueError("Stokes target must have shape (2,).")
        distances = jnp.linalg.norm(point[None, :] - self.panelization.points, axis=-1)
        point = eqx.error_if(
            point,
            jnp.min(distances) <= self.minimum_clearance,
            "Target violates the prepared off-surface clearance.",
        )
        if self.kind == "single":
            kernels = jax.vmap(self.kernel.value, in_axes=(None, 0))(
                point, self.panelization.points
            )
        else:
            kernels = jax.vmap(
                self.kernel.source_normal_derivative,
                in_axes=(None, 0, 0),
            )(point, self.panelization.points, self.panelization.normals)
        return ein.contract(
            "nij,nj,n->i",
            kernels,
            self.density,
            self.panelization.weights,
        )

    def evaluate(self, targets: ArrayLike, /) -> Array:
        values = jnp.asarray(targets)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("targets must have shape (target_count, 2).")
        return jax.vmap(self)(values)


__all__ = ["StokesLayerKernel2D", "StokesLayerPotential2D"]
