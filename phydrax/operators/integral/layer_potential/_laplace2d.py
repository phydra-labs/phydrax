#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._fingerprint import canonical_fingerprint
from ...._model import AbstractArrayModel
from ....equations.trefftz._core import (
    TRIAL_SPACE_CERTIFICATE_KEY,
    TrialSpaceCertificate,
)
from ._core import (
    AbstractLayerKernel,
    BoundaryLayerApproximationReport,
    BoundaryPanelization2D,
    KernelActionSide,
    LayerPotentialTargetReport,
)


class LaplaceLayerKernel2D(AbstractLayerKernel):
    """Two-dimensional Laplace fundamental solution with outward source normal."""

    _kernel_id: str

    def __init__(self):
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "laplace-layer-kernel-2d-v1",
                "fundamental_solution": "-log-distance/(2*pi)",
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
        return -jnp.log(radius) / (2.0 * jnp.pi)

    def source_normal_derivative(
        self,
        target: Array,
        source: Array,
        source_normal: Array,
        /,
    ) -> Array:
        difference = jnp.asarray(target) - jnp.asarray(source)
        radius_squared = jnp.sum(difference * difference)
        return jnp.dot(difference, jnp.asarray(source_normal)) / (
            2.0 * jnp.pi * radius_squared
        )


class LaplaceLayerPotential2D(AbstractArrayModel):
    """Finite weighted Laplace layer sum, algebraically harmonic off its sources."""

    panelization: BoundaryPanelization2D
    kernel: LaplaceLayerKernel2D
    density: Array
    kind: Literal["single", "double"] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)
    _certificate: TrialSpaceCertificate
    _approximation: BoundaryLayerApproximationReport

    def __init__(
        self,
        panelization: BoundaryPanelization2D,
        /,
        *,
        kind: Literal["single", "double"] = "double",
        density: ArrayLike | None = None,
    ):
        if not isinstance(panelization, BoundaryPanelization2D):
            raise TypeError("panelization must be BoundaryPanelization2D.")
        if kind not in ("single", "double"):
            raise ValueError("Laplace layer kind must be 'single' or 'double'.")
        density_ = (
            jnp.zeros((panelization.node_count,), dtype=float)
            if density is None
            else jnp.asarray(density, dtype=float)
        )
        if density_.shape != (panelization.node_count,):
            raise ValueError("Layer density must contain one scalar per source node.")
        kernel = LaplaceLayerKernel2D()
        representation_id = canonical_fingerprint(
            {
                "kind": "discrete-laplace-layer-potential-2d-v1",
                "kernel_id": kernel.kernel_id,
                "panelization_id": panelization.panelization_id,
                "layer_kind": kind,
            }
        )
        self.panelization = panelization
        self.kernel = kernel
        self.density = density_
        self.kind = kind
        self.in_size = 2
        self.out_size = "scalar"
        self._certificate = TrialSpaceCertificate(
            equation_family="laplace",
            ambient_dimension=2,
            construction=f"finite-{kind}-layer-kernel-sum",
            normalization_id="physical-euclidean-coordinates",
            basis_id=representation_id,
            rank=panelization.node_count,
            assumptions=(
                "two-dimensional Euclidean Laplacian",
                "targets lie outside the source singular support",
                f"layer-kernel:{kernel.kernel_id}",
            ),
            construction_residual=0.0,
            construction_tolerance=0.0,
            validity_region="off-singular-support",
            singular_support_id=panelization.source_support_id,
        )
        self._approximation = BoundaryLayerApproximationReport(
            panelization=panelization,
            kernel_id=kernel.kernel_id,
            density_space="quadrature-node-values",
            trace_policy="off-surface-gauss-legendre",
        )

    def with_density(self, density: ArrayLike, /) -> "LaplaceLayerPotential2D":
        values = jnp.asarray(density, dtype=self.density.dtype)
        if values.shape != self.density.shape:
            raise ValueError("Replacement density must preserve source-node shape.")
        return eqx.tree_at(lambda potential: potential.density, self, values)

    def _kernels(self, target: Array, /) -> Array:
        differences = target[None, :] - self.panelization.points
        squared_distances = jnp.sum(differences * differences, axis=-1)
        target = eqx.error_if(
            target,
            jnp.any(squared_distances == 0.0),
            "Layer-potential target intersects its singular support.",
        )
        if self.kind == "single":
            return jax.vmap(self.kernel.value, in_axes=(None, 0))(
                target,
                self.panelization.points,
            )
        return jax.vmap(
            self.kernel.source_normal_derivative,
            in_axes=(None, 0, 0),
        )(
            target,
            self.panelization.points,
            self.panelization.normals,
        )

    def __call__(self, target: Array, /, *, key=None) -> Array:
        del key
        value = jnp.asarray(target, dtype=float)
        if value.shape != (2,):
            raise ValueError(f"Laplace layer target must have shape (2,); got {value.shape}.")
        return contract(
            "n,n,n->",
            self._kernels(value),
            self.panelization.weights,
            self.density,
        )

    def evaluate_with_report(
        self,
        targets: ArrayLike,
        /,
        *,
        target_side: Literal["interior", "exterior"],
        accuracy_clearance: float = 0.0,
    ) -> tuple[Array, LayerPotentialTargetReport]:
        values = jnp.asarray(targets, dtype=float)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        report = LayerPotentialTargetReport(
            values,
            self.panelization,
            target_side=target_side,
            accuracy_clearance=accuracy_clearance,
        )
        output = jax.vmap(self)(values)
        return (output[0] if single else output), report

    def approximation_report(self) -> BoundaryLayerApproximationReport:
        return self._approximation

    def model_metadata(self) -> Mapping[str, Any]:
        return {TRIAL_SPACE_CERTIFICATE_KEY: self._certificate}


def double_layer_principal_value_matrix(
    panelization: BoundaryPanelization2D,
    /,
) -> Array:
    """Nyström principal-value matrix with local removable-diagonal limits."""

    if not isinstance(panelization, BoundaryPanelization2D):
        raise TypeError("panelization must be BoundaryPanelization2D.")
    kernel = LaplaceLayerKernel2D()
    targets = panelization.points
    sources = panelization.points
    normals = panelization.normals
    differences = targets[:, None, :] - sources[None, :, :]
    squared = jnp.sum(differences * differences, axis=-1)
    safe_squared = jnp.where(squared == 0.0, 1.0, squared)
    values = contract("tni,ni->tn", differences, normals) / (
        2.0 * jnp.pi * safe_squared
    )

    step = 1.0 / (
        panelization.panels_per_chart * panelization.quadrature_order * 1_000_000.0
    )
    reference = panelization.references[:, 0]
    direction = jnp.where(reference + step < 1.0, 1.0, -1.0)
    shifted_reference = (reference + direction * step)[:, None]
    shifted = panelization.atlas.frame(
        panelization.chart_indices,
        shifted_reference,
    )
    diagonal = jax.vmap(kernel.source_normal_derivative)(
        targets,
        shifted.origin,
        shifted.normal,
    )
    indices = jnp.arange(panelization.node_count)
    values = values.at[indices, indices].set(diagonal)
    return values * panelization.weights[None, :]


__all__ = [
    "double_layer_principal_value_matrix",
    "LaplaceLayerKernel2D",
    "LaplaceLayerPotential2D",
]
