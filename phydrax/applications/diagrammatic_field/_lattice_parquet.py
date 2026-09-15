#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Single-band lattice parquet channels over explicit two-particle routing."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...operators.quantum._thermal_green import MatsubaraGreenFunction
from ...operators.quantum._two_particle_green import (
    fermionic_crossing_evidence,
    FermionicTwoParticleChannelConvention,
    MatsubaraTwoParticleGreenFunction,
    two_particle_channel_permutation,
)


class LatticeParquetEvidence(StrictModule, NonTrainableState):
    fixed_point_residual: Array
    channel_residuals: Array
    parquet_identity_residual: Array
    crossing_residual: Array
    routing_residual: Array
    finite: Array
    converged: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class LatticeParquetResult(StrictModule):
    full_vertex: MatsubaraTwoParticleGreenFunction
    reducible_channels: Array
    iterations: Array
    evidence: LatticeParquetEvidence
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class LatticeSchwingerDysonEvidence(StrictModule, NonTrainableState):
    computed_self_energy: Array
    residual: Array
    relative_residual: Array
    finite: Array
    satisfied: Array
    vertex_id: str = eqx.field(static=True)
    green_id: str = eqx.field(static=True)


class LatticeParquetPlan(StrictModule, NonTrainableState):
    """Bounded pointwise Bethe--Salpeter closure in all three routed channels."""

    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    maximum_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_iterations: int = 256,
        tolerance: float = 1.0e-10,
        damping: float = 0.7,
        maximum_elements: int = 4_000_000,
    ):
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        damping_ = float(damping)
        capacity = int(maximum_elements)
        if (
            iterations <= 0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or not np.isfinite(damping_)
            or not 0.0 < damping_ <= 1.0
            or capacity <= 0
        ):
            raise ValueError("Lattice parquet iteration policy is invalid.")
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.damping = damping_
        self.maximum_elements = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "single-band-lattice-parquet-channel-plan",
                "maximum_iterations": iterations,
                "tolerance": tolerance_,
                "damping": damping_,
                "maximum_elements": capacity,
            }
        )

    def prepare(
        self,
        fully_irreducible_vertex: MatsubaraTwoParticleGreenFunction,
        channel_bubbles: ArrayLike,
        /,
    ) -> "PreparedLatticeParquet":
        if not isinstance(fully_irreducible_vertex, MatsubaraTwoParticleGreenFunction):
            raise TypeError(
                "fully_irreducible_vertex must be a MatsubaraTwoParticleGreenFunction."
            )
        if (
            fully_irreducible_vertex.mode_count != 1
            or not fully_irreducible_vertex.connected
        ):
            raise ValueError(
                "Lattice parquet is restricted to a connected single-band vertex."
            )
        if fully_irreducible_vertex.convention.channel != "particle-hole-direct":
            raise ValueError(
                "The lattice parquet storage channel must be particle-hole-direct."
            )
        shape = fully_irreducible_vertex.values.shape[:3]
        bubbles = np.asarray(channel_bubbles)
        if bubbles.shape != (3,) + shape or np.any(~np.isfinite(bubbles)):
            raise ValueError(
                "channel_bubbles must be finite with shape (3, transfer, left, right)."
            )
        required = int(fully_irreducible_vertex.values.size + 8 * bubbles.size)
        if required > self.maximum_elements:
            raise ValueError("Prepared lattice parquet arrays exceed maximum_elements.")
        channels = tuple(
            FermionicTwoParticleChannelConvention(name)
            for name in (
                "particle-hole-direct",
                "particle-hole-crossed",
                "particle-particle",
            )
        )
        gathers = []
        inverses = []
        routing_residual = 0
        for channel in channels:
            permutation = two_particle_channel_permutation(
                fully_irreducible_vertex, channel
            )
            gather = np.asarray(permutation.source_flat_indices).reshape(-1)
            if np.unique(gather).size != gather.size:
                raise ValueError(
                    "Cyclic channel routing must be a bijection on the finite bank."
                )
            gathers.append(gather)
            inverses.append(np.argsort(gather).astype(np.int32))
            routing_residual = max(routing_residual, int(permutation.routing_residual))
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-single-band-lattice-parquet",
                "plan": self.plan_id,
                "vertex": fully_irreducible_vertex.representation_id,
                "bubbles": array_tree_fingerprint(bubbles),
                "gathers": array_tree_fingerprint(np.stack(gathers)),
            }
        )
        return PreparedLatticeParquet(
            self,
            fully_irreducible_vertex,
            jnp.asarray(bubbles),
            jnp.asarray(np.stack(gathers)),
            jnp.asarray(np.stack(inverses)),
            jnp.asarray(routing_residual),
            prepared_id,
        )


class PreparedLatticeParquet(StrictModule, NonTrainableState):
    __hash__ = object.__hash__

    plan: LatticeParquetPlan
    fully_irreducible_vertex: MatsubaraTwoParticleGreenFunction
    channel_bubbles: Array
    channel_gathers: Array
    inverse_gathers: Array
    routing_residual: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan, vertex, bubbles, gathers, inverses, routing_residual, prepared_id, /
    ):
        self.plan = plan
        self.fully_irreducible_vertex = vertex
        self.channel_bubbles = bubbles
        self.channel_gathers = gathers
        self.inverse_gathers = inverses
        self.routing_residual = routing_residual
        self.prepared_id = str(prepared_id)

    def iterate(
        self, initial_channels: ArrayLike | None = None, /
    ) -> LatticeParquetResult:
        shape = self.fully_irreducible_vertex.values.shape[:3]
        bare = self.fully_irreducible_vertex.values[..., 0, 0, 0, 0]
        channels = (
            jnp.zeros((3,) + shape, dtype=bare.dtype)
            if initial_channels is None
            else jnp.asarray(initial_channels)
        )
        if channels.shape != (3,) + shape:
            raise ValueError(
                "initial_channels must have shape (3, transfer, left, right)."
            )
        residuals = jnp.full((3,), jnp.inf)
        iterations = 0
        converged = False
        for index in range(self.plan.maximum_iterations):
            full_direct = bare + jnp.sum(channels, axis=0)
            updated = []
            next_residuals = []
            for channel in range(3):
                gather = self.channel_gathers[channel]
                inverse = self.inverse_gathers[channel]
                full_routed = full_direct.reshape(-1)[gather].reshape(shape)
                current_routed = channels[channel].reshape(-1)[gather].reshape(shape)
                irreducible = full_routed - current_routed
                raw_routed = irreducible * self.channel_bubbles[channel] * full_routed
                raw_direct = raw_routed.reshape(-1)[inverse].reshape(shape)
                candidate = (1.0 - self.plan.damping) * channels[
                    channel
                ] + self.plan.damping * raw_direct
                scale = jnp.maximum(1.0, jnp.max(jnp.abs(candidate)))
                next_residuals.append(
                    jnp.max(jnp.abs(candidate - channels[channel])) / scale
                )
                updated.append(candidate)
            channels = jnp.stack(updated)
            residuals = jnp.stack(next_residuals)
            iterations = index + 1
            if bool(np.asarray(jnp.max(residuals) <= self.plan.tolerance)):
                converged = True
                break
        full_values = (bare + jnp.sum(channels, axis=0))[..., None, None, None, None]
        full = MatsubaraTwoParticleGreenFunction(
            self.fully_irreducible_vertex.convention,
            self.fully_irreducible_vertex.beta,
            self.fully_irreducible_vertex.transfer_labels,
            self.fully_irreducible_vertex.left_labels,
            self.fully_irreducible_vertex.right_labels,
            full_values,
            connected=True,
            fermion_label_minimum=self.fully_irreducible_vertex.routing.fermion_label_minimum,
            fermion_label_count=self.fully_irreducible_vertex.routing.fermion_label_count,
            maximum_elements=self.fully_irreducible_vertex.maximum_elements,
        )
        identity = jnp.max(
            jnp.abs(full_values[..., 0, 0, 0, 0] - bare - jnp.sum(channels, axis=0))
        )
        crossing = fermionic_crossing_evidence(full)
        crossing_residual = jnp.max(
            jnp.stack(
                (
                    crossing.annihilation_exchange_residual,
                    crossing.creation_exchange_residual,
                    crossing.simultaneous_exchange_residual,
                )
            )
        )
        finite = (
            full.finite
            & jnp.all(jnp.isfinite(channels))
            & jnp.all(jnp.isfinite(residuals))
        )
        successful = (
            finite
            & jnp.asarray(converged)
            & (identity <= self.plan.tolerance)
            & crossing.satisfied
            & (self.routing_residual == 0)
        )
        evidence = LatticeParquetEvidence(
            jnp.max(residuals),
            residuals,
            identity,
            crossing_residual,
            self.routing_residual,
            finite,
            jnp.asarray(converged),
            successful,
            self.prepared_id,
        )
        return LatticeParquetResult(
            full,
            channels,
            jnp.asarray(iterations, dtype=jnp.int32),
            evidence,
            self.prepared_id,
            "candidate single-band lattice parquet adapter; no material phase claim",
        )


def lattice_schwinger_dyson_evidence(
    vertex: MatsubaraTwoParticleGreenFunction,
    green: MatsubaraGreenFunction,
    self_energy: MatsubaraGreenFunction,
    /,
    *,
    interaction: float,
    density_per_spin: float,
    tolerance: float = 1.0e-8,
) -> LatticeSchwingerDysonEvidence:
    """Check the local single-band Matsubara Schwinger--Dyson contraction."""
    if (
        not isinstance(vertex, MatsubaraTwoParticleGreenFunction)
        or vertex.mode_count != 1
    ):
        raise TypeError("vertex must be a single-band MatsubaraTwoParticleGreenFunction.")
    if not isinstance(green, MatsubaraGreenFunction) or not isinstance(
        self_energy, MatsubaraGreenFunction
    ):
        raise TypeError("green and self_energy must be MatsubaraGreenFunction values.")
    labels = np.asarray(green.indices)
    if (
        green.statistics != "fermionic"
        or self_energy.statistics != "fermionic"
        or green.beta != vertex.beta
        or self_energy.beta != vertex.beta
        or not np.array_equal(labels, np.asarray(self_energy.indices))
        or not np.array_equal(labels, np.asarray(vertex.left_labels))
        or not np.array_equal(labels, np.asarray(vertex.right_labels))
    ):
        raise ValueError("Schwinger--Dyson Matsubara axes and statistics must match.")
    if green.values.shape != labels.shape or self_energy.values.shape != labels.shape:
        raise ValueError(
            "Schwinger--Dyson control is restricted to scalar one-particle data."
        )
    coupling = float(interaction)
    density = float(density_per_spin)
    tolerance_ = float(tolerance)
    if (
        not np.isfinite(coupling)
        or not np.isfinite(density)
        or not 0.0 <= density <= 1.0
        or not np.isfinite(tolerance_)
        or tolerance_ <= 0.0
    ):
        raise ValueError("Schwinger--Dyson parameters are invalid.")
    minimum = int(np.min(labels))
    count = int(labels.size)
    if not np.array_equal(np.sort(labels), np.arange(minimum, minimum + count)):
        raise ValueError(
            "Schwinger--Dyson contraction requires a contiguous cyclic fermion bank."
        )
    positions = {int(value): index for index, value in enumerate(labels)}
    transfer = np.asarray(vertex.transfer_labels)
    if transfer.size != count or not np.array_equal(
        np.sort(transfer), np.arange(int(np.min(transfer)), int(np.min(transfer)) + count)
    ):
        raise ValueError("Transfer labels must form a contiguous bank of the same size.")
    values = vertex.values[..., 0, 0, 0, 0]
    computed = []
    for left_index, left_label in enumerate(labels):
        contraction = jnp.asarray(0.0 + 0.0j, dtype=values.dtype)
        for transfer_index, transfer_label in enumerate(transfer):
            left_shift = positions[
                int(minimum + ((left_label + transfer_label - minimum) % count))
            ]
            for right_index, right_label in enumerate(labels):
                right_shift = positions[
                    int(minimum + ((right_label + transfer_label - minimum) % count))
                ]
                contraction = (
                    contraction
                    + values[transfer_index, left_index, right_index]
                    * green.values[right_index]
                    * green.values[right_shift]
                    * green.values[left_shift]
                )
        computed.append(
            coupling * density - coupling * contraction / (vertex.beta * vertex.beta)
        )
    computed_array = jnp.stack(computed)
    residual = jnp.max(jnp.abs(computed_array - self_energy.values))
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(self_energy.values)))
    relative = residual / scale
    finite = jnp.all(jnp.isfinite(computed_array)) & jnp.isfinite(residual)
    return LatticeSchwingerDysonEvidence(
        computed_array,
        residual,
        relative,
        finite,
        finite & (relative <= tolerance_),
        vertex.representation_id,
        green.representation_id,
    )


__all__ = [
    "LatticeParquetEvidence",
    "LatticeParquetPlan",
    "LatticeParquetResult",
    "LatticeSchwingerDysonEvidence",
    "PreparedLatticeParquet",
    "lattice_schwinger_dyson_evidence",
]
