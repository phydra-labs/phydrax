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

from phydrax.ein import contract
from phydrax.special import kv

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._core import AbstractLayerKernel, BoundaryPanelization2D, KernelActionSide


class ModifiedHelmholtzLayerKernel2D(AbstractLayerKernel):
    """Two-dimensional modified-Helmholtz fundamental solution."""

    decay: float = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(self, decay: float, /):
        value = float(decay)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("decay must be finite and positive.")
        self.decay = value
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "modified-helmholtz-layer-kernel-2d",
                "decay": value,
                "fundamental_solution": "K0(decay*distance)/(2*pi)",
                "normal": "outward-source",
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def source_event_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def target_event_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def action_side(self) -> KernelActionSide:
        return "left"

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    def value(self, target: Array, source: Array, /) -> Array:
        difference = jnp.asarray(target) - jnp.asarray(source)
        radius = jnp.linalg.norm(difference)
        return kv(0.0, self.decay * radius) / (2.0 * jnp.pi)

    def source_normal_derivative(
        self,
        target: Array,
        source: Array,
        source_normal: Array,
        /,
    ) -> Array:
        difference = jnp.asarray(target) - jnp.asarray(source)
        radius = jnp.linalg.norm(difference)
        return (
            self.decay
            * kv(1.0, self.decay * radius)
            * jnp.dot(difference, jnp.asarray(source_normal))
            / (2.0 * jnp.pi * radius)
        )


class ModifiedHelmholtzLayerPotential2D(StrictModule, NonTrainableState):
    """Prepared off-surface modified-Helmholtz layer reconstruction."""

    panelization: BoundaryPanelization2D
    kernel: ModifiedHelmholtzLayerKernel2D
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
        decay: float,
        kind: Literal["single", "double"] = "single",
        minimum_clearance: float,
    ):
        if not isinstance(panelization, BoundaryPanelization2D):
            raise TypeError("panelization must be BoundaryPanelization2D.")
        values = jnp.asarray(density, dtype=jnp.float64)
        if values.shape != (panelization.node_count,):
            raise ValueError("density must contain one scalar per panelization node.")
        if kind not in ("single", "double"):
            raise ValueError("kind must be 'single' or 'double'.")
        clearance = float(minimum_clearance)
        if not np.isfinite(clearance) or clearance <= 0.0:
            raise ValueError("minimum_clearance must be finite and positive.")
        kernel = ModifiedHelmholtzLayerKernel2D(decay)
        self.panelization = panelization
        self.kernel = kernel
        self.density = values
        self.minimum_clearance = clearance
        self.kind = kind
        self.representation_id = canonical_fingerprint(
            {
                "kind": "modified-helmholtz-layer-potential-2d",
                "panelization": panelization.panelization_id,
                "kernel": kernel.kernel_id,
                "layer": kind,
                "clearance": clearance,
            }
        )

    def __call__(self, target: ArrayLike, /) -> Array:
        point = jnp.asarray(target, dtype=self.panelization.points.dtype)
        if point.shape != (2,):
            raise ValueError("Modified-Helmholtz target must have shape (2,).")
        differences = point[None, :] - self.panelization.points
        radii = jnp.linalg.norm(differences, axis=-1)
        point = eqx.error_if(
            point,
            jnp.min(radii) <= self.minimum_clearance,
            "Target violates the prepared off-surface clearance.",
        )
        if self.kind == "single":
            kernels = jax.vmap(self.kernel.value, in_axes=(None, 0))(
                point,
                self.panelization.points,
            )
        else:
            kernels = jax.vmap(
                self.kernel.source_normal_derivative,
                in_axes=(None, 0, 0),
            )(point, self.panelization.points, self.panelization.normals)
        return contract(
            "n,n,n->",
            kernels,
            self.panelization.weights,
            self.density,
            backend="jax",
        )

    def evaluate(self, targets: ArrayLike, /) -> Array:
        values = jnp.asarray(targets)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("targets must have shape (target_count, 2).")
        return jax.vmap(self)(values)


def prepare_modified_helmholtz_reconstruction_2d(
    panelization: BoundaryPanelization2D,
    density: ArrayLike,
    /,
    *,
    decay: float,
    kind: Literal["single", "double"] = "single",
    minimum_clearance: float,
) -> ModifiedHelmholtzLayerPotential2D:
    return ModifiedHelmholtzLayerPotential2D(
        panelization,
        density,
        decay=decay,
        kind=kind,
        minimum_clearance=minimum_clearance,
    )


__all__ = [
    "ModifiedHelmholtzLayerKernel2D",
    "ModifiedHelmholtzLayerPotential2D",
    "prepare_modified_helmholtz_reconstruction_2d",
]
