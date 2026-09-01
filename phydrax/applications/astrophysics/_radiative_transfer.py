#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._observation_status import AstrophysicsObservationStatus
from ._photometry import ObservationDataProvenance


class OpacityTable(StrictModule, NonTrainableState):
    temperature: Array
    pressure: Array
    frequency: Array
    absorption: Array
    provenance: ObservationDataProvenance
    table_id: str = eqx.field(static=True)

    def __init__(self, temperature, pressure, frequency, absorption, provenance, /):
        axes = tuple(
            np.asarray(value, dtype=float) for value in (temperature, pressure, frequency)
        )
        values = np.asarray(absorption, dtype=float)
        if (
            any(
                axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0.0)
                for axis in axes
            )
            or values.shape != tuple(axis.size for axis in axes)
            or np.any(values < 0.0)
            or np.any(~np.isfinite(values))
        ):
            raise ValueError("Opacity table axes or values are invalid.")
        self.temperature, self.pressure, self.frequency = tuple(
            jnp.asarray(axis) for axis in axes
        )
        self.absorption = jnp.asarray(values)
        self.provenance = provenance
        self.table_id = canonical_fingerprint(
            {
                "kind": "opacity-table",
                "shape": list(values.shape),
                "provenance": provenance.provenance_id,
            }
        )

    def evaluate(
        self, temperature: ArrayLike, pressure: ArrayLike, frequency: ArrayLike, /
    ) -> tuple[Array, Array]:
        queries = tuple(
            jnp.asarray(value).reshape(()) for value in (temperature, pressure, frequency)
        )
        axes = (self.temperature, self.pressure, self.frequency)
        indices = tuple(
            jnp.clip(jnp.searchsorted(axis, query, side="right"), 1, int(axis.size) - 1)
            for axis, query in zip(axes, queries, strict=True)
        )
        fractions = tuple(
            (query - axis[index - 1]) / (axis[index] - axis[index - 1])
            for axis, query, index in zip(axes, queries, indices, strict=True)
        )
        value = jnp.asarray(0.0)
        for mask in range(8):
            index_tuple = tuple(
                indices[dimension] if mask & (1 << dimension) else indices[dimension] - 1
                for dimension in range(3)
            )
            weight = jnp.prod(
                jnp.asarray(
                    tuple(
                        fractions[dimension]
                        if mask & (1 << dimension)
                        else 1.0 - fractions[dimension]
                        for dimension in range(3)
                    )
                )
            )
            value = value + weight * self.absorption[index_tuple]
        support = jnp.all(
            jnp.asarray(
                tuple(
                    (query >= axis[0]) & (query <= axis[-1])
                    for axis, query in zip(axes, queries, strict=True)
                )
            )
        )
        return value, support


class RadiativeTransferResult(StrictModule):
    emergent: Array
    iterations: Array
    residual: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class ScalarRadiativeTransferPlan(StrictModule, NonTrainableState):
    segment_lengths: Array
    scattering_albedo: Array
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        segment_lengths,
        scattering_albedo=0.0,
        /,
        *,
        maximum_iterations=32,
        tolerance=1.0e-10,
        plan_id="scalar-radiative-transfer",
    ):
        lengths = jnp.asarray(segment_lengths)
        if lengths.ndim != 1:
            raise ValueError("Radiative-transfer segment lengths must be a vector.")
        self.segment_lengths = lengths
        self.scattering_albedo = jnp.broadcast_to(
            jnp.asarray(scattering_albedo), lengths.shape
        )
        self.maximum_iterations = int(maximum_iterations)
        self.tolerance = float(tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scalar-radiative-transfer",
                "segments": int(lengths.size),
                "iterations": int(maximum_iterations),
            }
        )

    def evaluate(
        self, emissivity: ArrayLike, extinction: ArrayLike, incident: ArrayLike = 0.0, /
    ) -> RadiativeTransferResult:
        source = jnp.asarray(emissivity)
        opacity = jnp.asarray(extinction)
        if source.shape != self.segment_lengths.shape or opacity.shape != source.shape:
            raise ValueError("Radiative source and extinction must match segments.")

        def formal(effective_source):
            def step(intensity, values):
                ds, j_value, k_value = values
                tau = k_value * ds
                transmission = jnp.exp(-tau)
                increment = jnp.where(
                    jnp.abs(k_value) > 0.0,
                    j_value / k_value * (1.0 - transmission),
                    j_value * ds,
                )
                result = intensity * transmission + increment
                return result, result

            return jax.lax.scan(
                step,
                jnp.asarray(incident),
                (self.segment_lengths, effective_source, opacity),
            )

        def iteration(_, carry):
            previous, converged, first = carry
            emergent, history = formal(
                source + self.scattering_albedo * opacity * previous
            )
            residual = jnp.max(jnp.abs(history - previous))
            now = residual <= self.tolerance
            return history, converged | now, jnp.where((first < 0) & now, _ + 1, first)

        initial = jnp.zeros_like(source)
        history, converged, iterations = jax.lax.fori_loop(
            0,
            self.maximum_iterations,
            iteration,
            (initial, jnp.asarray(False), jnp.asarray(-1, dtype=jnp.int32)),
        )
        emergent, final_history = formal(
            source + self.scattering_albedo * opacity * history
        )
        residual = jnp.max(jnp.abs(final_history - history))
        valid = converged & jnp.all(jnp.isfinite(final_history)) & jnp.all(opacity >= 0.0)
        status = jnp.where(
            valid,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.NONPHYSICAL_MODEL),
        ).astype(jnp.int32)
        return RadiativeTransferResult(
            emergent,
            jnp.where(iterations >= 0, iterations, self.maximum_iterations),
            residual,
            valid,
            status,
            self.plan_id,
        )


class PolarizedRadiativeTransferPlan(StrictModule, NonTrainableState):
    segment_lengths: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, segment_lengths, /, *, plan_id="polarized-radiative-transfer"):
        self.segment_lengths = jnp.asarray(segment_lengths)
        self.plan_id = str(plan_id)

    def evaluate(
        self, emission: ArrayLike, propagation_matrix: ArrayLike, incident: ArrayLike, /
    ) -> RadiativeTransferResult:
        emission_ = jnp.asarray(emission)
        matrix = jnp.asarray(propagation_matrix)
        if emission_.shape != (self.segment_lengths.size, 4) or matrix.shape != (
            self.segment_lengths.size,
            4,
            4,
        ):
            raise ValueError("Polarized transfer arrays have incompatible shapes.")

        def step(stokes, values):
            ds, source, operator = values
            transfer = jsp.linalg.expm(-operator * ds)
            increment = jsp.linalg.solve(operator, (jnp.eye(4) - transfer) @ source)
            result = transfer @ stokes + increment
            return result, result

        emergent, history = jax.lax.scan(
            step, jnp.asarray(incident), (self.segment_lengths, emission_, matrix)
        )
        valid = jnp.all(jnp.isfinite(history))
        status = jnp.where(
            valid,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.NONFINITE_INPUT),
        ).astype(jnp.int32)
        return RadiativeTransferResult(
            emergent, jnp.asarray(1), jnp.asarray(0.0), valid, status, self.plan_id
        )


__all__ = [
    "OpacityTable",
    "PolarizedRadiativeTransferPlan",
    "RadiativeTransferResult",
    "ScalarRadiativeTransferPlan",
]
