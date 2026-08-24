#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import ComplexStiefelManifold, faithful_density_from_cholesky
from ._causal_process import CausalProcessTensor, CombLegSpec


class ProcessGaugeReport(StrictModule):
    isometry_residuals: Array
    coordinate_count: Array
    physical_parameter_count: Array
    gauge_dimension: Array
    valid: Array

    def __init__(
        self,
        isometry_residuals: ArrayLike,
        coordinate_count: ArrayLike,
        physical_parameter_count: ArrayLike,
        gauge_dimension: ArrayLike,
        /,
    ):
        self.isometry_residuals = jnp.asarray(isometry_residuals)
        self.coordinate_count = jnp.asarray(coordinate_count)
        self.physical_parameter_count = jnp.asarray(physical_parameter_count)
        self.gauge_dimension = jnp.asarray(gauge_dimension)
        self.valid = (
            jnp.all(jnp.isfinite(self.isometry_residuals))
            & jnp.all(self.isometry_residuals <= 1e-8)
            & (self.coordinate_count > 0)
            & (self.physical_parameter_count > 0)
            & (
                self.coordinate_count
                == self.physical_parameter_count + self.gauge_dimension
            )
        )


class SequentialStinespringProcess(StrictModule):
    spec: CombLegSpec
    initial_factor: Array
    isometries: tuple[Array, ...]
    environment_dimensions: tuple[int, ...]
    process_id: str

    def __init__(
        self,
        spec: CombLegSpec,
        initial_factor: ArrayLike,
        isometries: Sequence[ArrayLike],
        environment_dimensions: Sequence[int],
        /,
        *,
        process_id: str,
    ):
        factor = jnp.asarray(initial_factor)
        composite = spec.system_dimension * spec.memory_dimension
        if factor.shape != (composite, composite):
            raise ValueError("Initial Stinespring density factor shape is invalid.")
        values = tuple(jnp.asarray(value) for value in isometries)
        environments = tuple(int(value) for value in environment_dimensions)
        if len(values) != spec.slot_count or len(environments) != spec.slot_count:
            raise ValueError("One Stinespring isometry/environment is required per slot.")
        for value, environment in zip(values, environments, strict=True):
            if value.shape != (composite * environment, composite):
                raise ValueError("Stinespring isometry shape is invalid.")
            manifold = ComplexStiefelManifold(composite * environment, composite)
            if not bool(manifold.contains(value)):
                raise ValueError("Stinespring matrix does not have orthonormal columns.")
        self.spec = spec
        self.initial_factor = factor
        self.isometries = values
        self.environment_dimensions = environments
        self.process_id = str(process_id)

    def materialize(self) -> CausalProcessTensor:
        composite = self.spec.system_dimension * self.spec.memory_dimension
        channels = tuple(
            value.reshape((environment, composite, composite))
            for value, environment in zip(
                self.isometries, self.environment_dimensions, strict=True
            )
        )
        return CausalProcessTensor(
            self.spec,
            faithful_density_from_cholesky(self.initial_factor),
            channels,
            process_id=self.process_id,
        )

    def gauge_report(self) -> ProcessGaugeReport:
        composite = self.spec.system_dimension * self.spec.memory_dimension
        residuals = jnp.stack(
            [
                jnp.linalg.norm(
                    jnp.conj(value.T) @ value - jnp.eye(composite, dtype=value.dtype)
                )
                for value in self.isometries
            ]
        )
        coordinate_count = 2 * self.initial_factor.size + sum(
            2 * value.size for value in self.isometries
        )
        density_parameters = composite**2 - 1
        channel_parameters = sum(
            2 * composite**2 * environment
            - composite**2
            - environment**2
            for environment in self.environment_dimensions
        )
        temporal_memory_gauge = (self.spec.slot_count + 1) * max(
            0, self.spec.memory_dimension**2 - 1
        )
        physical_parameter_count = (
            density_parameters + channel_parameters - temporal_memory_gauge
        )
        gauge_dimension = coordinate_count - physical_parameter_count
        return ProcessGaugeReport(
            residuals,
            coordinate_count,
            physical_parameter_count,
            gauge_dimension,
        )


__all__ = [
    "ProcessGaugeReport",
    "SequentialStinespringProcess",
]
