#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from ..._precision import (
    precision_itemsize,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    PrecisionResourceAssumptions,
    real_precision_dtype_name,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_SUPPORTED_LBM_DTYPES = frozenset(("float32", "float64"))


class LatticeBoltzmannPrecisionPolicy(StrictModule, NonTrainableState):
    """Qualified population, compute, accumulation, and evidence precision.

    Float32 and float64 homogeneous policies are qualified. Narrow float32
    population storage with float64 arithmetic is available only through the
    explicit ``mixed_storage=True`` contract; half storage is never admitted.
    """

    population_dtype: str = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)
    accumulation_dtype: str = eqx.field(static=True)
    certification_dtype: str = eqx.field(static=True)
    mixed_storage: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        population_dtype: Any = jnp.float64,
        compute_dtype: Any | None = None,
        accumulation_dtype: Any | None = None,
        certification_dtype: Any | None = None,
        mixed_storage: bool = False,
    ):
        population = real_precision_dtype_name(population_dtype)
        compute = (
            population
            if compute_dtype is None
            else real_precision_dtype_name(compute_dtype)
        )
        accumulation = (
            compute
            if accumulation_dtype is None
            else real_precision_dtype_name(accumulation_dtype)
        )
        certification = (
            accumulation
            if certification_dtype is None
            else real_precision_dtype_name(certification_dtype)
        )
        values = (population, compute, accumulation, certification)
        if any(value not in _SUPPORTED_LBM_DTYPES for value in values):
            raise ValueError(
                "LBM precision qualification supports only float32 and float64; half storage is rejected."
            )
        mixed = bool(mixed_storage)
        if population != compute:
            if not mixed:
                raise ValueError(
                    "Mixed LBM storage requires the explicit mixed_storage=True configuration."
                )
            if population != "float32" or compute != "float64":
                raise ValueError(
                    "Qualified mixed LBM storage is float32 population with float64 compute."
                )
        elif mixed:
            raise ValueError(
                "mixed_storage=True requires float32 population and float64 compute."
            )
        if precision_itemsize(compute) < precision_itemsize(population):
            raise ValueError(
                "LBM compute precision cannot be narrower than population storage."
            )
        if precision_itemsize(accumulation) < precision_itemsize(compute):
            raise ValueError(
                "LBM accumulation precision cannot be narrower than compute."
            )
        if precision_itemsize(certification) < precision_itemsize(accumulation):
            raise ValueError(
                "LBM certification precision cannot be narrower than accumulation."
            )
        request = PrecisionRequest(
            "lattice-boltzmann",
            {
                "coefficient": compute,
                "storage": population,
                "compute": compute,
                "accumulation": accumulation,
                "certification": certification,
                "communication": population,
                "checkpoint": population,
                "output": population,
            },
        )
        backend = (
            "phydrax-lattice-boltzmann-mixed-storage"
            if mixed
            else "phydrax-lattice-boltzmann"
        )
        resolution = PrecisionResolution(request, backend, dict(request.requested))
        self.population_dtype = population
        self.compute_dtype = compute
        self.accumulation_dtype = accumulation
        self.certification_dtype = certification
        self.mixed_storage = mixed
        self.policy_id = resolution.resolution_id

    def coefficient(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.compute_dtype))

    def population(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.population_dtype))

    def compute(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.compute_dtype))

    def accumulation(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.accumulation_dtype))

    def certification(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.certification_dtype))

    @property
    def precision_request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "lattice-boltzmann",
            {
                "coefficient": self.compute_dtype,
                "storage": self.population_dtype,
                "compute": self.compute_dtype,
                "accumulation": self.accumulation_dtype,
                "certification": self.certification_dtype,
                "communication": self.population_dtype,
                "checkpoint": self.population_dtype,
                "output": self.population_dtype,
            },
        )

    @property
    def precision_resolution(self) -> PrecisionResolution:
        request = self.precision_request
        backend = (
            "phydrax-lattice-boltzmann-mixed-storage"
            if self.mixed_storage
            else "phydrax-lattice-boltzmann"
        )
        return PrecisionResolution(request, backend, dict(request.requested))

    @property
    def resource_assumptions(self) -> PrecisionResourceAssumptions:
        return PrecisionResourceAssumptions(
            "lattice-boltzmann", dict(self.precision_request.requested)
        )

    def evidence(self) -> PrecisionEvidenceEnvelope:
        resolution = self.precision_resolution
        return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))


__all__ = ["LatticeBoltzmannPrecisionPolicy"]
