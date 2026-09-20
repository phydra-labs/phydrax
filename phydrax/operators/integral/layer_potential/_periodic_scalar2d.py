#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import PeriodicCell
from ._core import BoundaryPanelization2D


PeriodicScalarEquation2D = Literal["laplace", "helmholtz", "modified_helmholtz"]


class PeriodicScalarSpectralEvidence2D(StrictModule):
    minimum_denominator: Array
    shell_indicator: Array
    finite: Array
    nonresonant: Array
    successful: Array
    mode_count: int = eqx.field(static=True)
    cutoff: int = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)


class PeriodicScalarSpectralKernel2D(StrictModule, NonTrainableState):
    """Finite reciprocal-lattice scalar Green provider with explicit truncation evidence."""

    cell: PeriodicCell
    wave_vectors: Array
    denominators: Array
    evidence: PeriodicScalarSpectralEvidence2D
    equation: PeriodicScalarEquation2D = eqx.field(static=True)
    parameter: float = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        /,
        *,
        equation: PeriodicScalarEquation2D,
        parameter: float = 0.0,
        cutoff: int = 16,
        maximum_modes: int = 100_000,
        resonance_tolerance: float = 1e-10,
    ):
        if not isinstance(cell, PeriodicCell):
            raise TypeError("cell must be a PeriodicCell.")
        if cell.rank != 2 or cell.ambient_dimension != 2:
            raise ValueError("Periodic scalar 2D kernels require a full-rank 2D cell.")
        if equation not in ("laplace", "helmholtz", "modified_helmholtz"):
            raise ValueError("Unknown periodic scalar equation.")
        parameter_ = float(parameter)
        cutoff_ = int(cutoff)
        maximum = int(maximum_modes)
        tolerance = float(resonance_tolerance)
        if cutoff_ <= 0 or maximum <= 0 or not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Periodic spectral resources and tolerance are invalid.")
        if equation != "laplace" and (not np.isfinite(parameter_) or parameter_ <= 0.0):
            raise ValueError(
                "Helmholtz and modified-Helmholtz parameters must be positive."
            )
        integer_modes = np.asarray(
            tuple(product(range(-cutoff_, cutoff_ + 1), repeat=2)),
            dtype=np.int32,
        )
        if equation == "laplace":
            integer_modes = integer_modes[np.any(integer_modes != 0, axis=1)]
        if integer_modes.shape[0] > maximum:
            raise ValueError(
                "Periodic spectral kernel exceeds maximum_modes: "
                f"required {integer_modes.shape[0]}, allowed {maximum}."
            )
        modes = jnp.asarray(integer_modes, dtype=cell.vectors.dtype)
        wave_vectors = modes @ cell.reciprocal_vectors
        squared = jnp.sum(wave_vectors * wave_vectors, axis=-1)
        if equation == "laplace":
            denominators = squared
        elif equation == "helmholtz":
            denominators = squared - parameter_**2
        else:
            denominators = squared + parameter_**2
        minimum = jnp.min(jnp.abs(denominators))
        finite = jnp.all(jnp.isfinite(wave_vectors)) & jnp.all(jnp.isfinite(denominators))
        nonresonant = minimum > tolerance
        shell = jnp.max(jnp.abs(1.0 / denominators[-max(8, 8 * cutoff_) :]))
        provider_id = canonical_fingerprint(
            {
                "kind": "periodic-scalar-spectral-kernel-2d",
                "cell": cell.cell_id,
                "equation": equation,
                "parameter": parameter_,
                "cutoff": cutoff_,
                "maximum_modes": maximum,
                "resonance_tolerance": tolerance,
            }
        )
        self.cell = cell
        self.wave_vectors = wave_vectors
        self.denominators = denominators
        self.evidence = PeriodicScalarSpectralEvidence2D(
            minimum,
            shell,
            finite,
            nonresonant,
            finite & nonresonant,
            integer_modes.shape[0],
            cutoff_,
            provider_id,
        )
        self.equation = equation
        self.parameter = parameter_

    def value(self, target: ArrayLike, source: ArrayLike, /) -> Array:
        difference = jnp.asarray(target) - jnp.asarray(source)
        phase = self.wave_vectors @ difference
        checked = eqx.error_if(
            phase,
            ~self.evidence.successful,
            "Periodic scalar spectral kernel is nonfinite or resonant.",
        )
        return jnp.sum(jnp.cos(checked) / self.denominators) / self.cell.cell_measure

    def source_normal_derivative(
        self,
        target: ArrayLike,
        source: ArrayLike,
        source_normal: ArrayLike,
        /,
    ) -> Array:
        difference = jnp.asarray(target) - jnp.asarray(source)
        phase = self.wave_vectors @ difference
        projection = self.wave_vectors @ jnp.asarray(source_normal)
        phase = eqx.error_if(
            phase,
            ~self.evidence.successful,
            "Periodic scalar spectral kernel is nonfinite or resonant.",
        )
        return (
            jnp.sum(jnp.sin(phase) * projection / self.denominators)
            / self.cell.cell_measure
        )


class PeriodicScalarLayerPotential2D(StrictModule, NonTrainableState):
    panelization: BoundaryPanelization2D
    kernel: PeriodicScalarSpectralKernel2D
    density: Array
    kind: Literal["single", "double"] = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        panelization: BoundaryPanelization2D,
        kernel: PeriodicScalarSpectralKernel2D,
        density: ArrayLike,
        /,
        *,
        kind: Literal["single", "double"] = "single",
    ):
        if not isinstance(panelization, BoundaryPanelization2D):
            raise TypeError("panelization must be BoundaryPanelization2D.")
        if not isinstance(kernel, PeriodicScalarSpectralKernel2D):
            raise TypeError("kernel must be PeriodicScalarSpectralKernel2D.")
        values = jnp.asarray(density, dtype=jnp.float64)
        if values.shape != (panelization.node_count,):
            raise ValueError("density must contain one scalar per source node.")
        if kind not in ("single", "double"):
            raise ValueError("kind must be 'single' or 'double'.")
        self.panelization = panelization
        self.kernel = kernel
        self.density = values
        self.kind = kind
        self.representation_id = canonical_fingerprint(
            {
                "kind": "periodic-scalar-layer-potential-2d",
                "panelization": panelization.panelization_id,
                "provider": kernel.evidence.provider_id,
                "layer": kind,
            }
        )

    def __call__(self, target: ArrayLike, /) -> Array:
        point = jnp.asarray(target, dtype=self.panelization.points.dtype)
        if point.shape != (2,):
            raise ValueError("Periodic scalar target must have shape (2,).")
        if self.kind == "single":
            kernels = jax.vmap(self.kernel.value, in_axes=(None, 0))(
                point, self.panelization.points
            )
        else:
            kernels = jax.vmap(
                self.kernel.source_normal_derivative,
                in_axes=(None, 0, 0),
            )(point, self.panelization.points, self.panelization.normals)
        return jnp.sum(kernels * self.panelization.weights * self.density)


__all__ = [
    "PeriodicScalarEquation2D",
    "PeriodicScalarLayerPotential2D",
    "PeriodicScalarSpectralEvidence2D",
    "PeriodicScalarSpectralKernel2D",
]
