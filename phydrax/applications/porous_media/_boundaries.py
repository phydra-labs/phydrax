#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pressure, integrated liquid mass-rate, and leakage boundary qualification."""

from __future__ import annotations

from collections.abc import Mapping

from jax import Array

from ..._strict import StrictModule
from ...discretization.finite_volume._diffusion_boundary import HybridDiffusionBoundary
from ...discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
)


class PorousBoundaryConditions(StrictModule):
    """Pressure in Pa; outward liquid mass in kg/s; leakage in kg/(Pa s).

    ``leakage_kg_Pa_s`` maps each face to (conductance, reservoir_pressure_Pa).
    A zero outward rate is impermeability. Prescribed inflow is a negative rate.
    Values are area integrated, not flux densities. Boundary pressure uses the
    same gauge reference as the retention and material laws.
    """

    diffusion: HybridDiffusionBoundary

    def __init__(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        /,
        *,
        pressure_Pa: Mapping[int, float] | None = None,
        mass_rate_kg_s: Mapping[int, float] | None = None,
        leakage_kg_Pa_s: Mapping[int, tuple[float, float]] | None = None,
    ) -> None:
        self.diffusion = HybridDiffusionBoundary(
            discretization,
            dirichlet=pressure_Pa,
            neumann=mass_rate_kg_s,
            robin=leakage_kg_Pa_s,
        )

    @property
    def kind(self) -> Array:
        return self.diffusion.kind

    @property
    def value(self) -> Array:
        return self.diffusion.value

    @property
    def conductance(self) -> Array:
        return self.diffusion.conductance

    @property
    def geometry_id(self) -> str:
        return self.diffusion.geometry_id

    def face_residual(self, face_values: Array, outward_sum: Array) -> Array:
        return self.diffusion.face_residual(face_values, outward_sum)

    def impose_dirichlet(self, face_values: Array) -> Array:
        return self.diffusion.impose_dirichlet(face_values)


__all__ = ["PorousBoundaryConditions"]
