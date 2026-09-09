#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....interchange import GeospatialContract
from ._gravity import GravityQuadratureSource


VACUUM_PERMEABILITY_H_M = 4.0e-7 * np.pi


class MagneticMaterial(StrictModule):
    susceptibility: Array
    remanent_magnetization_A_m: Array

    def __init__(
        self,
        susceptibility: ArrayLike,
        remanent_magnetization_A_m: ArrayLike,
        cell_count: int,
        /,
    ):
        raw = jnp.asarray(susceptibility)
        count = int(cell_count)
        if raw.shape in ((), (count,)):
            tensors = jnp.broadcast_to(raw, (count,))[:, None, None] * jnp.eye(3)
        elif raw.shape == (3, 3):
            tensors = jnp.broadcast_to(raw, (count, 3, 3))
        elif raw.shape == (count, 3, 3):
            tensors = raw
        else:
            raise ValueError(
                "Magnetic susceptibility must be scalar/cell scalar or SPD tensor."
            )
        remanent = jnp.broadcast_to(jnp.asarray(remanent_magnetization_A_m), (count, 3))
        symmetric = jnp.max(jnp.abs(tensors - jnp.swapaxes(tensors, -1, -2)))
        tensors = eqx.error_if(
            tensors,
            jnp.any(~jnp.isfinite(tensors))
            | (symmetric > 1e-10)
            | jnp.any(jnp.linalg.eigvalsh(tensors) < 0)
            | jnp.any(~jnp.isfinite(remanent)),
            "Magnetic susceptibility must be finite symmetric positive semidefinite and remanence finite.",
        )
        self.susceptibility = 0.5 * (tensors + jnp.swapaxes(tensors, -1, -2))
        self.remanent_magnetization_A_m = remanent

    def magnetization(self, inducing_field_A_m: ArrayLike, /) -> Array:
        field = jnp.asarray(inducing_field_A_m)
        field = jnp.broadcast_to(field, self.remanent_magnetization_A_m.shape)
        field = eqx.error_if(
            field,
            jnp.any(~jnp.isfinite(field)),
            "Inducing magnetic field must be finite.",
        )
        return (
            ein.contract("cij,cj->ci", self.susceptibility, field)
            + self.remanent_magnetization_A_m
        )


class MagneticResult(StrictModule):
    flux_density_T: Array
    total_field_anomaly_T: Array
    finite: Array


class FreeSpaceMagneticPlan(StrictModule, NonTrainableState):
    """Free-space dipole-volume magnetic field without demagnetization."""

    source: GravityQuadratureSource
    observations_m: Array
    reference_direction: Array
    minimum_separation_m: float = eqx.field(static=True)
    coordinates: GeospatialContract
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: GravityQuadratureSource,
        observations_m: ArrayLike,
        reference_direction: ArrayLike,
        coordinates: GeospatialContract,
        /,
        *,
        minimum_separation_m: float,
    ):
        if not isinstance(source, GravityQuadratureSource):
            raise TypeError(
                "Magnetic plan requires gravity-compatible volume quadrature."
            )
        if not isinstance(coordinates, GeospatialContract):
            raise TypeError("Magnetic observations require GeospatialContract.")
        coordinates.require_cartesian(dimensions=3)
        observations = np.asarray(observations_m, dtype=float)
        reference = np.asarray(reference_direction, dtype=float)
        separation = float(minimum_separation_m)
        if (
            observations.ndim != 2
            or observations.shape[1] != 3
            or observations.shape[0] == 0
            or np.any(~np.isfinite(observations))
            or reference.shape != (3,)
            or np.any(~np.isfinite(reference))
            or not np.isclose(np.sum(reference**2), 1.0, rtol=1e-10, atol=1e-10)
            or not np.isfinite(separation)
            or separation <= 0
        ):
            raise ValueError("Magnetic observations/reference/separation are invalid.")
        distance = np.sqrt(
            np.sum(
                (np.asarray(source.points_m)[None, :, :] - observations[:, None, :]) ** 2,
                axis=-1,
            )
        )
        if np.min(distance) < separation:
            raise ValueError("Magnetic observation enters the near-singular exclusion.")
        self.source, self.observations_m = source, jnp.asarray(observations)
        self.reference_direction = jnp.asarray(reference)
        self.minimum_separation_m, self.coordinates = separation, coordinates
        self.plan_id = canonical_fingerprint(
            {
                "kind": "free-space-magnetic-plan",
                "source": source.source_id,
                "observations_m": observations,
                "reference_direction": reference,
                "minimum_separation_m": separation,
                "coordinates": coordinates.coordinate_id,
                "demagnetization": False,
            }
        )

    def evaluate(
        self,
        material: MagneticMaterial,
        inducing_field_A_m: ArrayLike,
        /,
    ) -> MagneticResult:
        if not isinstance(material, MagneticMaterial):
            raise TypeError("Magnetic evaluation requires MagneticMaterial.")
        if material.remanent_magnetization_A_m.shape[0] != self.source.cell_count:
            raise ValueError("Magnetic material does not match source cells.")
        cell_magnetization = material.magnetization(inducing_field_A_m)
        moments = (
            cell_magnetization[self.source.cell_indices]
            * self.source.volume_weights_m3[:, None]
        )
        displacement = self.observations_m[:, None, :] - self.source.points_m[None, :, :]
        radius = jnp.sqrt(jnp.sum(displacement**2, axis=-1))
        radius = eqx.error_if(
            radius,
            jnp.any(radius < self.minimum_separation_m),
            "Magnetic evaluation entered its near-singular exclusion.",
        )
        direction = displacement / radius[..., None]
        projection = ein.contract("qi,oqi->oq", moments, direction)
        kernel = (3.0 * projection[..., None] * direction - moments[None, :, :]) / radius[
            ..., None
        ] ** 3
        field = (VACUUM_PERMEABILITY_H_M / (4.0 * jnp.pi)) * jnp.sum(kernel, axis=1)
        total = field @ self.reference_direction
        finite = jnp.all(jnp.isfinite(field)) & jnp.all(jnp.isfinite(total))
        return MagneticResult(field, total, finite)

    def require_demagnetization(self) -> None:
        raise ValueError(
            "High-susceptibility demagnetization requires a separately solved self-consistent model."
        )


__all__ = [
    "FreeSpaceMagneticPlan",
    "MagneticMaterial",
    "MagneticResult",
    "VACUUM_PERMEABILITY_H_M",
]
