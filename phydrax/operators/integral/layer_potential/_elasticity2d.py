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


PlaneElasticityReduction = Literal["plane_stress", "plane_strain"]


class ElasticityLayerKernel2D(AbstractLayerKernel):
    """Planar isotropic Kelvin displacement and source-traction kernels."""

    lame_lambda: float = eqx.field(static=True)
    shear_modulus: float = eqx.field(static=True)
    reduction: PlaneElasticityReduction = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        young_modulus: float,
        poisson_ratio: float,
        reduction: PlaneElasticityReduction,
    ):
        young = float(young_modulus)
        poisson = float(poisson_ratio)
        if not np.isfinite(young) or young <= 0.0:
            raise ValueError("young_modulus must be finite and positive.")
        if not np.isfinite(poisson) or not -1.0 < poisson < 0.5:
            raise ValueError("poisson_ratio must lie strictly between -1 and 0.5.")
        if reduction == "plane_strain":
            lame_lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        elif reduction == "plane_stress":
            lame_lambda = young * poisson / (1.0 - poisson**2)
        else:
            raise ValueError("reduction must be 'plane_stress' or 'plane_strain'.")
        shear = young / (2.0 * (1.0 + poisson))
        self.lame_lambda = lame_lambda
        self.shear_modulus = shear
        self.reduction = reduction
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "elasticity-layer-kernel-2d",
                "young_modulus": young,
                "poisson_ratio": poisson,
                "reduction": reduction,
            }
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
        lam = self.lame_lambda
        mu = self.shear_modulus
        denominator = 4.0 * jnp.pi * mu * (lam + 2.0 * mu)
        identity = jnp.eye(2, dtype=difference.dtype)
        return (
            -(lam + 3.0 * mu) * jnp.log(radius) * identity
            + (lam + mu)
            * ein.contract("i,j->ij", difference, difference)
            / radius_squared
        ) / denominator

    def source_normal_derivative(
        self,
        target: Array,
        source: Array,
        source_normal: Array,
        /,
    ) -> Array:
        gradient = jax.jacfwd(lambda point: self.value(target, point))(source)
        divergence = jnp.trace(gradient, axis1=0, axis2=2)
        identity = jnp.eye(2, dtype=gradient.dtype)
        stress = self.lame_lambda * ein.contract(
            "ik,j->ikj", identity, divergence
        ) + self.shear_modulus * (
            jnp.transpose(gradient, (0, 2, 1)) + jnp.transpose(gradient, (2, 0, 1))
        )
        return ein.contract("ikj,k->ij", stress, jnp.asarray(source_normal))


class ElasticityLayerPotential2D(StrictModule, NonTrainableState):
    """Prepared off-surface planar elasticity layer potential."""

    panelization: BoundaryPanelization2D
    kernel: ElasticityLayerKernel2D
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
        young_modulus: float,
        poisson_ratio: float,
        reduction: PlaneElasticityReduction,
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
        kernel = ElasticityLayerKernel2D(
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            reduction=reduction,
        )
        self.panelization = panelization
        self.kernel = kernel
        self.density = values
        self.minimum_clearance = clearance
        self.kind = kind
        self.representation_id = canonical_fingerprint(
            {
                "kind": "elasticity-layer-potential-2d",
                "panelization": panelization.panelization_id,
                "kernel": kernel.kernel_id,
                "layer": kind,
                "clearance": clearance,
            }
        )

    def __call__(self, target: ArrayLike, /) -> Array:
        point = jnp.asarray(target, dtype=self.panelization.points.dtype)
        if point.shape != (2,):
            raise ValueError("Elasticity target must have shape (2,).")
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


__all__ = [
    "ElasticityLayerKernel2D",
    "ElasticityLayerPotential2D",
    "PlaneElasticityReduction",
]
