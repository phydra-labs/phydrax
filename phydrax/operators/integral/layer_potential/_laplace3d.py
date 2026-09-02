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

from phydrax.ein import contract

from ...._fingerprint import canonical_fingerprint
from ...._model import AbstractArrayModel
from ....equations.trefftz._core import (
    TRIAL_SPACE_CERTIFICATE_KEY,
    TrialSpaceCertificate,
)
from ._core import LayerDiscretizationReport
from ._surface3d import SurfacePanelization3D, SurfaceTargetReport3D


class LaplaceLayerKernel3D(eqx.Module):
    """Three-dimensional Laplace fundamental solution."""

    _kernel_id: str = eqx.field(static=True)

    def __init__(self):
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "laplace-layer-kernel-3d-v1",
                "fundamental_solution": "1/(4*pi*r)",
                "normal": "outward-source",
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    def value(self, target: Array, source: Array, /) -> Array:
        radius = jnp.linalg.norm(jnp.asarray(target) - jnp.asarray(source))
        return 1.0 / (4.0 * jnp.pi * radius)

    def source_normal_derivative(
        self,
        target: Array,
        source: Array,
        source_normal: Array,
        /,
    ) -> Array:
        difference = jnp.asarray(target) - jnp.asarray(source)
        radius_squared = jnp.sum(difference * difference)
        radius = jnp.sqrt(radius_squared)
        return jnp.dot(difference, jnp.asarray(source_normal)) / (
            4.0 * jnp.pi * radius_squared * radius
        )


class LaplaceLayerPotential3D(AbstractArrayModel):
    """Finite weighted 3D Laplace layer sum, exact off its support."""

    panelization: SurfacePanelization3D
    kernel: LaplaceLayerKernel3D
    density: Array
    kind: Literal["single", "double"] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)
    _certificate: TrialSpaceCertificate
    _discretization: LayerDiscretizationReport
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        panelization: SurfacePanelization3D,
        /,
        *,
        kind: Literal["single", "double"] = "single",
        density: ArrayLike | None = None,
    ):
        if not isinstance(panelization, SurfacePanelization3D):
            raise TypeError("panelization must be SurfacePanelization3D.")
        if kind not in ("single", "double"):
            raise ValueError("3D Laplace layer kind must be 'single' or 'double'.")
        density_ = (
            jnp.zeros((panelization.node_count,), dtype=float)
            if density is None
            else jnp.asarray(density, dtype=float)
        )
        if density_.shape != (panelization.node_count,):
            raise ValueError("Layer density must contain one value per surface node.")
        kernel = LaplaceLayerKernel3D()
        representation_id = canonical_fingerprint(
            {
                "kind": "discrete-laplace-layer-potential-3d-v1",
                "kernel_id": kernel.kernel_id,
                "panelization_id": panelization.panelization_id,
                "layer_kind": kind,
            }
        )
        self.panelization = panelization
        self.kernel = kernel
        self.density = density_
        self.kind = kind
        self.in_size = 3
        self.out_size = "scalar"
        self.representation_id = representation_id
        self._certificate = TrialSpaceCertificate(
            equation_family="laplace",
            ambient_dimension=3,
            construction=f"finite-{kind}-layer-kernel-sum-3d",
            normalization_id="physical-euclidean-coordinates",
            basis_id=representation_id,
            rank=panelization.node_count,
            assumptions=(
                "three-dimensional Euclidean Laplacian",
                "targets lie outside the source singular support",
                f"layer-kernel:{kernel.kernel_id}",
            ),
            construction_residual=0.0,
            construction_tolerance=0.0,
            validity_region="off-singular-support",
            singular_support_id=panelization.source_support_id,
        )
        self._discretization = LayerDiscretizationReport(
            panelization=panelization,
            kernel_id=kernel.kernel_id,
            density_space="surface-quadrature-node-values",
            trace_policy="off-surface-reference-triangle",
        )

    def with_density(self, density: ArrayLike, /) -> "LaplaceLayerPotential3D":
        values = jnp.asarray(density, dtype=self.density.dtype)
        if values.shape != self.density.shape:
            raise ValueError("Replacement density must preserve source-node shape.")
        return eqx.tree_at(lambda potential: potential.density, self, values)

    def __call__(self, target: Array, /, *, key=None) -> Array:
        del key
        value = jnp.asarray(target, dtype=float)
        if value.shape != (3,):
            raise ValueError(f"3D layer target must have shape (3,); got {value.shape}.")
        differences = value[None, :] - self.panelization.points
        squared = jnp.sum(differences * differences, axis=-1)
        value = eqx.error_if(
            value,
            jnp.any(squared == 0.0),
            "3D layer target intersects its singular support.",
        )
        if self.kind == "single":
            kernels = jax.vmap(self.kernel.value, in_axes=(None, 0))(
                value,
                self.panelization.points,
            )
        else:
            kernels = jax.vmap(
                self.kernel.source_normal_derivative,
                in_axes=(None, 0, 0),
            )(
                value,
                self.panelization.points,
                self.panelization.normals,
            )
        return contract(
            "n,n,n->",
            kernels,
            self.panelization.weights,
            self.density,
        )

    def _evaluate_direct(self, targets: ArrayLike, /) -> Array:
        values = jnp.asarray(targets, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
            raise ValueError("3D layer targets must have shape (target_count, 3).")
        return jax.vmap(self)(values)

    def discretization_report(self) -> LayerDiscretizationReport:
        return self._discretization

    def model_metadata(self) -> Mapping[str, Any]:
        return {TRIAL_SPACE_CERTIFICATE_KEY: self._certificate}


def evaluate_laplace_layer_3d(
    potential: LaplaceLayerPotential3D,
    targets: ArrayLike,
    /,
    *,
    target_side: Literal["interior", "exterior", "boundary"],
    accuracy_clearance: float = 0.0,
) -> tuple[Array, SurfaceTargetReport3D]:
    """Evaluate an off-surface 3D layer with continuous geometry evidence."""
    if not isinstance(potential, LaplaceLayerPotential3D):
        raise TypeError("potential must be LaplaceLayerPotential3D.")
    values = jnp.asarray(targets, dtype=float)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    report = SurfaceTargetReport3D(
        values,
        potential.panelization,
        target_side=target_side,
        accuracy_clearance=accuracy_clearance,
    )
    if not bool(report.pde_membership_valid):
        raise ValueError("3D direct layer evaluation requires off-surface targets.")
    output = potential._evaluate_direct(values)
    return (output[0] if single else output), report


__all__ = [
    "LaplaceLayerKernel3D",
    "LaplaceLayerPotential3D",
    "evaluate_laplace_layer_3d",
]
