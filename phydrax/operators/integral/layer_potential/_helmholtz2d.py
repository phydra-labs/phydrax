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
from ....special import hankel1
from ._core import AbstractLayerKernel, BoundaryPanelization2D, LayerDiscretizationReport


class HelmholtzLayerKernel2D(AbstractLayerKernel):
    wavenumber: float = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(self, wavenumber: float):
        value = float(wavenumber)
        if not jnp.isfinite(value) or value <= 0.0:
            raise ValueError("Helmholtz wavenumber must be finite and positive.")
        self.wavenumber = value
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "helmholtz-layer-kernel-2d-v1",
                "fundamental_solution": "i*hankel1(0,k*r)/4",
                "normal": "outward-source",
                "radiation": "outgoing",
                "wavenumber": value,
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
    def action_side(self) -> Literal["left"]:
        return "left"

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    def value(self, target: Array, source: Array, /) -> Array:
        difference = jnp.asarray(target) - jnp.asarray(source)
        radius = jnp.linalg.norm(difference)
        return 0.25j * hankel1(0.0, self.wavenumber * radius)

    def source_normal_derivative(
        self,
        target: Array,
        source: Array,
        source_normal: Array,
        /,
    ) -> Array:
        difference = jnp.asarray(target) - jnp.asarray(source)
        radius = jnp.linalg.norm(difference)
        safe_radius = jnp.where(radius == 0.0, 1.0, radius)
        return (
            0.25j
            * self.wavenumber
            * hankel1(1.0, self.wavenumber * safe_radius)
            * jnp.dot(difference, jnp.asarray(source_normal))
            / safe_radius
        )


class HelmholtzLayerPotential2D(AbstractArrayModel):
    """Finite outgoing Helmholtz layer sum, exact off its source support."""

    panelization: BoundaryPanelization2D
    kernel: HelmholtzLayerKernel2D
    density: Array
    kind: Literal["single", "double"] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)
    _certificate: TrialSpaceCertificate
    _discretization: LayerDiscretizationReport
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        panelization: BoundaryPanelization2D,
        wavenumber: float,
        /,
        *,
        kind: Literal["single", "double"] = "single",
        density: ArrayLike | None = None,
    ):
        if not isinstance(panelization, BoundaryPanelization2D):
            raise TypeError("panelization must be BoundaryPanelization2D.")
        if kind not in ("single", "double"):
            raise ValueError("Helmholtz layer kind must be 'single' or 'double'.")
        density_ = (
            jnp.zeros((panelization.node_count,), dtype=complex)
            if density is None
            else jnp.asarray(density, dtype=complex)
        )
        if density_.shape != (panelization.node_count,):
            raise ValueError("Layer density must contain one scalar per source node.")
        kernel = HelmholtzLayerKernel2D(wavenumber)
        representation_id = canonical_fingerprint(
            {
                "kind": "discrete-helmholtz-layer-potential-2d-v1",
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
        self.representation_id = representation_id
        self._certificate = TrialSpaceCertificate(
            equation_family="helmholtz",
            ambient_dimension=2,
            construction=f"finite-{kind}-helmholtz-layer-kernel-sum",
            normalization_id="physical-euclidean-coordinates",
            basis_id=representation_id,
            rank=panelization.node_count,
            assumptions=(
                "two-dimensional outgoing Helmholtz equation",
                "targets lie outside the source singular support",
                f"wavenumber:{kernel.wavenumber}",
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
            density_space="quadrature-node-values",
            trace_policy="off-surface-gauss-legendre",
        )

    def with_density(self, density: ArrayLike, /) -> "HelmholtzLayerPotential2D":
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
            raise ValueError(
                f"Helmholtz layer target must have shape (2,); got {value.shape}."
            )
        return contract(
            "n,n,n->",
            self._kernels(value),
            self.panelization.weights,
            self.density,
        )

    def _evaluate_direct(self, targets: ArrayLike, /) -> Array:
        values = jnp.asarray(targets, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError("Direct layer targets must have shape (target_count, 2).")
        return jax.vmap(self)(values)

    def discretization_report(self) -> LayerDiscretizationReport:
        return self._discretization

    def model_metadata(self) -> Mapping[str, Any]:
        return {TRIAL_SPACE_CERTIFICATE_KEY: self._certificate}


class HelmholtzCombinedField2D(AbstractArrayModel):
    """Brakhage--Werner combined field ``D - i*eta*S``."""

    panelization: BoundaryPanelization2D
    kernel: HelmholtzLayerKernel2D
    density: Array
    eta: float = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)
    _certificate: TrialSpaceCertificate
    _discretization: LayerDiscretizationReport
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        panelization: BoundaryPanelization2D,
        wavenumber: float,
        density: ArrayLike,
        /,
        *,
        eta: float,
    ):
        kernel = HelmholtzLayerKernel2D(wavenumber)
        density_ = jnp.asarray(density, dtype=complex)
        if density_.shape != (panelization.node_count,):
            raise ValueError("Combined-field density must match panel node count.")
        eta_ = float(eta)
        if not jnp.isfinite(eta_) or eta_ <= 0.0:
            raise ValueError("Combined-field coupling eta must be finite and positive.")
        representation_id = canonical_fingerprint(
            {
                "kind": "helmholtz-brakhage-werner-field-2d-v1",
                "kernel_id": kernel.kernel_id,
                "panelization_id": panelization.panelization_id,
                "eta": eta_,
            }
        )
        self.panelization = panelization
        self.kernel = kernel
        self.density = density_
        self.eta = eta_
        self.in_size = 2
        self.out_size = "scalar"
        self.representation_id = representation_id
        self._certificate = TrialSpaceCertificate(
            equation_family="helmholtz",
            ambient_dimension=2,
            construction="finite-brakhage-werner-combined-layer-sum",
            normalization_id="physical-euclidean-coordinates",
            basis_id=representation_id,
            rank=panelization.node_count,
            assumptions=(
                "two-dimensional outgoing Helmholtz equation",
                "targets lie outside the source singular support",
                f"wavenumber:{kernel.wavenumber}",
                f"coupling:{eta_}",
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
            density_space="quadrature-node-values-complex",
            trace_policy="brakhage-werner-combined-field",
        )

    def __call__(self, target: Array, /, *, key=None) -> Array:
        del key
        value = jnp.asarray(target, dtype=float)
        if value.shape != (2,):
            raise ValueError(
                f"Combined-field target must have shape (2,); got {value.shape}."
            )
        differences = value[None, :] - self.panelization.points
        squared = jnp.sum(differences * differences, axis=-1)
        value = eqx.error_if(
            value,
            jnp.any(squared == 0.0),
            "Combined-field target intersects its singular support.",
        )
        single = jax.vmap(self.kernel.value, in_axes=(None, 0))(
            value,
            self.panelization.points,
        )
        double = jax.vmap(
            self.kernel.source_normal_derivative,
            in_axes=(None, 0, 0),
        )(
            value,
            self.panelization.points,
            self.panelization.normals,
        )
        return contract(
            "n,n,n->",
            double - 1j * self.eta * single,
            self.panelization.weights,
            self.density,
        )

    def _evaluate_direct(self, targets: ArrayLike, /) -> Array:
        values = jnp.asarray(targets, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError("Direct layer targets must have shape (target_count, 2).")
        return jax.vmap(self)(values)

    def discretization_report(self) -> LayerDiscretizationReport:
        return self._discretization

    def model_metadata(self) -> Mapping[str, Any]:
        return {TRIAL_SPACE_CERTIFICATE_KEY: self._certificate}


__all__ = [
    "HelmholtzCombinedField2D",
    "HelmholtzLayerKernel2D",
    "HelmholtzLayerPotential2D",
]
