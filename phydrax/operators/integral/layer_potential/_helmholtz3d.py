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
from ....equations.trefftz._core import TRIAL_SPACE_CERTIFICATE_KEY, TrialSpaceCertificate
from ._core import LayerDiscretizationReport
from ._surface3d import SurfacePanelization3D


class HelmholtzLayerKernel3D(eqx.Module):
    """Outgoing three-dimensional Helmholtz fundamental solution."""

    wavenumber: float = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(self, wavenumber: float):
        value = float(wavenumber)
        if not jnp.isfinite(value) or value <= 0.0:
            raise ValueError("Helmholtz wavenumber must be finite and positive.")
        self.wavenumber = value
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "helmholtz-layer-kernel-3d-v1",
                "fundamental_solution": "exp(i*k*r)/(4*pi*r)",
                "normal": "outward-source",
                "radiation": "outgoing",
                "wavenumber": value,
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    def value(self, target: Array, source: Array, /) -> Array:
        difference = jnp.asarray(target) - jnp.asarray(source)
        radius = jnp.linalg.norm(difference)
        return jnp.exp(1j * self.wavenumber * radius) / (4.0 * jnp.pi * radius)

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
        return (
            jnp.exp(1j * self.wavenumber * radius)
            * (1.0 - 1j * self.wavenumber * radius)
            * jnp.dot(difference, jnp.asarray(source_normal))
            / (4.0 * jnp.pi * radius_squared * radius)
        )


class HelmholtzLayerPotential3D(AbstractArrayModel):
    """Finite outgoing 3D Helmholtz layer sum."""

    panelization: SurfacePanelization3D
    kernel: HelmholtzLayerKernel3D
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
        wavenumber: float,
        /,
        *,
        kind: Literal["single", "double"] = "single",
        density: ArrayLike | None = None,
    ):
        if not isinstance(panelization, SurfacePanelization3D):
            raise TypeError("panelization must be SurfacePanelization3D.")
        if kind not in ("single", "double"):
            raise ValueError("3D Helmholtz layer kind must be 'single' or 'double'.")
        density_ = (
            jnp.zeros((panelization.node_count,), dtype=complex)
            if density is None
            else jnp.asarray(density, dtype=complex)
        )
        if density_.shape != (panelization.node_count,):
            raise ValueError("Layer density must match surface node count.")
        kernel = HelmholtzLayerKernel3D(wavenumber)
        representation_id = canonical_fingerprint(
            {
                "kind": "discrete-helmholtz-layer-potential-3d-v1",
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
            equation_family="helmholtz",
            ambient_dimension=3,
            construction=f"finite-{kind}-helmholtz-layer-kernel-sum-3d",
            normalization_id="physical-euclidean-coordinates",
            basis_id=representation_id,
            rank=panelization.node_count,
            assumptions=(
                "three-dimensional outgoing Helmholtz equation",
                "targets lie outside the source singular support",
                f"wavenumber:{kernel.wavenumber}",
            ),
            construction_residual=0.0,
            construction_tolerance=0.0,
            validity_region="off-singular-support",
            singular_support_id=panelization.source_support_id,
        )
        self._discretization = LayerDiscretizationReport(
            panelization=panelization,
            kernel_id=kernel.kernel_id,
            density_space="surface-quadrature-node-values-complex",
            trace_policy="off-surface-reference-triangle",
        )

    def with_density(self, density: ArrayLike, /) -> "HelmholtzLayerPotential3D":
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
            "3D Helmholtz target intersects its singular support.",
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
        return contract("n,n,n->", kernels, self.panelization.weights, self.density)

    def _evaluate_direct(self, targets: ArrayLike, /) -> Array:
        values = jnp.asarray(targets, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
            raise ValueError("3D Helmholtz targets must have shape (target_count, 3).")
        return jax.vmap(self)(values)

    def discretization_report(self) -> LayerDiscretizationReport:
        return self._discretization

    def model_metadata(self) -> Mapping[str, Any]:
        return {TRIAL_SPACE_CERTIFICATE_KEY: self._certificate}


class HelmholtzCombinedField3D(AbstractArrayModel):
    """Outgoing 3D Brakhage--Werner combined field."""

    panelization: SurfacePanelization3D
    kernel: HelmholtzLayerKernel3D
    density: Array
    eta: float = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)
    _certificate: TrialSpaceCertificate
    _discretization: LayerDiscretizationReport
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        panelization: SurfacePanelization3D,
        wavenumber: float,
        density: ArrayLike,
        /,
        *,
        eta: float,
    ):
        if not isinstance(panelization, SurfacePanelization3D):
            raise TypeError("panelization must be SurfacePanelization3D.")
        coupling = float(eta)
        if not jnp.isfinite(coupling) or coupling <= 0.0:
            raise ValueError("eta must be finite and positive.")
        kernel = HelmholtzLayerKernel3D(wavenumber)
        density_ = jnp.asarray(density, dtype=complex)
        if density_.shape != (panelization.node_count,):
            raise ValueError("Combined-field density must match surface node count.")
        representation_id = canonical_fingerprint(
            {
                "kind": "helmholtz-brakhage-werner-field-3d-v1",
                "kernel_id": kernel.kernel_id,
                "panelization_id": panelization.panelization_id,
                "eta": coupling,
            }
        )
        self.panelization = panelization
        self.kernel = kernel
        self.density = density_
        self.eta = coupling
        self.in_size = 3
        self.out_size = "scalar"
        self.representation_id = representation_id
        self._certificate = TrialSpaceCertificate(
            equation_family="helmholtz",
            ambient_dimension=3,
            construction="finite-brakhage-werner-combined-layer-sum-3d",
            normalization_id="physical-euclidean-coordinates",
            basis_id=representation_id,
            rank=panelization.node_count,
            assumptions=(
                "three-dimensional outgoing Helmholtz equation",
                "targets lie outside the source singular support",
                f"wavenumber:{kernel.wavenumber}",
                f"coupling:{coupling}",
                "outgoing-radiation-condition",
            ),
            construction_residual=0.0,
            construction_tolerance=0.0,
            validity_region="off-singular-support",
            singular_support_id=panelization.source_support_id,
        )
        self._discretization = LayerDiscretizationReport(
            panelization=panelization,
            kernel_id=kernel.kernel_id,
            density_space="surface-quadrature-node-values-complex",
            trace_policy="three-dimensional-brakhage-werner-combined",
        )

    def __call__(self, target: Array, /, *, key=None) -> Array:
        del key
        value = jnp.asarray(target, dtype=float)
        differences = value[None, :] - self.panelization.points
        squared = jnp.sum(differences * differences, axis=-1)
        value = eqx.error_if(
            value,
            jnp.any(squared == 0.0),
            "3D combined-field target intersects its singular support.",
        )
        single = jax.vmap(self.kernel.value, in_axes=(None, 0))(
            value,
            self.panelization.points,
        )
        double = jax.vmap(
            self.kernel.source_normal_derivative,
            in_axes=(None, 0, 0),
        )(value, self.panelization.points, self.panelization.normals)
        return contract(
            "n,n,n->",
            double - 1j * self.eta * single,
            self.panelization.weights,
            self.density,
        )

    def _evaluate_direct(self, targets: ArrayLike, /) -> Array:
        values = jnp.asarray(targets, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
            raise ValueError("3D combined-field targets must have shape (target_count, 3).")
        return jax.vmap(self)(values)

    def discretization_report(self) -> LayerDiscretizationReport:
        return self._discretization

    def model_metadata(self) -> Mapping[str, Any]:
        return {TRIAL_SPACE_CERTIFICATE_KEY: self._certificate}


__all__ = [
    "HelmholtzCombinedField3D",
    "HelmholtzLayerKernel3D",
    "HelmholtzLayerPotential3D",
]
