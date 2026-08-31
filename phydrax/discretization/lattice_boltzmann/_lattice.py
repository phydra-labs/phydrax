#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class LatticeBoltzmannCapabilityEvidence(StrictModule, NonTrainableState):
    """Immutable evidence for algorithms certified on one velocity set."""

    nearest_neighbor: bool = eqx.field(static=True)
    hydrodynamic_isotropy_order: int = eqx.field(static=True)
    tensor_product: bool = eqx.field(static=True)
    capabilities: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        nearest_neighbor: bool,
        hydrodynamic_isotropy_order: int,
        tensor_product: bool,
        capabilities: Sequence[str],
    ):
        capability_tuple = tuple(sorted(str(value) for value in capabilities))
        if len(set(capability_tuple)) != len(capability_tuple):
            raise ValueError("Lattice capabilities must be unique.")
        self.nearest_neighbor = bool(nearest_neighbor)
        self.hydrodynamic_isotropy_order = int(hydrodynamic_isotropy_order)
        self.tensor_product = bool(tensor_product)
        self.capabilities = capability_tuple
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-capability-evidence",
                "nearest_neighbor": self.nearest_neighbor,
                "hydrodynamic_isotropy_order": self.hydrodynamic_isotropy_order,
                "tensor_product": self.tensor_product,
                "capabilities": capability_tuple,
            }
        )

    def supports(self, capability: str, /) -> bool:
        return str(capability) in self.capabilities

    def require(self, capability: str, /) -> None:
        if not self.supports(capability):
            raise ValueError(
                f"Velocity set is not certified for LBM capability {capability!r}."
            )


class LatticeBoltzmannVelocitySet(StrictModule, NonTrainableState):
    """Certified on-lattice nearest-neighbour quadrature.

    Construction is certification: malformed, non-isotropic, or non-local custom
    sets are rejected rather than being admitted with inferred capabilities.
    """

    velocities: Array
    weights: Array
    opposite: Array
    sound_speed_squared: Array
    capability_evidence: LatticeBoltzmannCapabilityEvidence
    velocity_tuples: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    opposite_indices: tuple[int, ...] = eqx.field(static=True)
    name: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    population_count: int = eqx.field(static=True)
    lattice_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        velocities: ArrayLike,
        weights: ArrayLike,
        opposite: Sequence[int] | ArrayLike,
        /,
        *,
        sound_speed_squared: float = 1.0 / 3.0,
    ):
        name_ = str(name)
        if not name_:
            raise ValueError("Velocity-set name must be non-empty.")
        velocity_host = np.asarray(velocities)
        if velocity_host.ndim != 2 or velocity_host.shape[0] < 2:
            raise ValueError("Lattice velocities must have shape (Q, dimension).")
        if velocity_host.shape[1] not in (2, 3):
            raise ValueError("LBM velocity sets require dimension two or three.")
        if not np.issubdtype(velocity_host.dtype, np.integer):
            if np.any(~np.isfinite(velocity_host)) or np.any(
                velocity_host != np.round(velocity_host)
            ):
                raise ValueError("On-lattice velocities must be finite integers.")
        velocity_host = velocity_host.astype(np.int32)
        q, dimension = velocity_host.shape
        if np.max(np.abs(velocity_host)) > 1:
            raise ValueError(
                "Certified nearest-neighbour velocities may move at most one cell per axis."
            )
        if np.unique(velocity_host, axis=0).shape[0] != q:
            raise ValueError("Lattice velocities must be unique.")
        if np.sum(np.all(velocity_host == 0, axis=1)) != 1:
            raise ValueError("A velocity set must contain exactly one rest direction.")

        weight_host = np.asarray(weights, dtype=np.float64).reshape((-1,))
        opposite_host = np.asarray(opposite, dtype=np.int32).reshape((-1,))
        cs2 = float(sound_speed_squared)
        if weight_host.shape != (q,) or opposite_host.shape != (q,):
            raise ValueError("Weights and opposite indices must each have shape (Q,).")
        if (
            np.any(~np.isfinite(weight_host))
            or np.any(weight_host <= 0.0)
            or not np.isfinite(cs2)
            or cs2 <= 0.0
        ):
            raise ValueError(
                "Lattice weights and sound speed must be finite and positive."
            )
        if not np.array_equal(np.sort(opposite_host), np.arange(q, dtype=np.int32)):
            raise ValueError("Opposite indices must form a permutation.")
        if not np.array_equal(opposite_host[opposite_host], np.arange(q)):
            raise ValueError("Opposite indices must be involutive.")
        if not np.array_equal(velocity_host[opposite_host], -velocity_host):
            raise ValueError("Opposite indices must negate lattice velocities.")
        if not np.allclose(weight_host[opposite_host], weight_host, rtol=0.0, atol=1e-15):
            raise ValueError("Opposite lattice directions must have equal weights.")

        identity = np.eye(dimension, dtype=np.float64)
        first = oe.contract("q,qa->a", weight_host, velocity_host)
        second = oe.contract("q,qa,qb->ab", weight_host, velocity_host, velocity_host)
        third = oe.contract(
            "q,qa,qb,qc->abc",
            weight_host,
            velocity_host,
            velocity_host,
            velocity_host,
        )
        fourth = oe.contract(
            "q,qa,qb,qc,qd->abcd",
            weight_host,
            velocity_host,
            velocity_host,
            velocity_host,
            velocity_host,
        )
        expected_fourth = cs2**2 * (
            oe.contract("ab,cd->abcd", identity, identity)
            + oe.contract("ac,bd->abcd", identity, identity)
            + oe.contract("ad,bc->abcd", identity, identity)
        )
        if not np.isclose(np.sum(weight_host), 1.0, rtol=0.0, atol=1e-14):
            raise ValueError("Lattice weights must sum to one.")
        if not np.allclose(first, 0.0, rtol=0.0, atol=1e-14):
            raise ValueError("Lattice first moments must vanish.")
        if not np.allclose(second, cs2 * identity, rtol=0.0, atol=1e-14):
            raise ValueError("Lattice second moments are not isotropic.")
        if not np.allclose(third, 0.0, rtol=0.0, atol=1e-14):
            raise ValueError("Lattice third moments must vanish.")
        if not np.allclose(fourth, expected_fourth, rtol=0.0, atol=1e-14):
            raise ValueError("Lattice fourth moments are not isotropic.")

        complete_tensor_product = q == 3**dimension and {
            tuple(int(value) for value in row) for row in velocity_host
        } == {
            index
            for index in np.ndindex(*(3,) * dimension)
            for index in (tuple(value - 1 for value in index),)
        }
        expected_tensor_weights = np.prod(
            np.where(velocity_host == 0, 1.0 - cs2, 0.5 * cs2), axis=1
        )
        tensor_product = complete_tensor_product and np.allclose(
            weight_host, expected_tensor_weights, rtol=0.0, atol=1e-14
        )
        capabilities = [
            "athermal-hydrodynamics",
            "bgk",
            "fourth-order-isotropy",
            "guo-forcing",
            "mrt",
            "nearest-neighbor-streaming",
            "regularized-second-order",
            "smagorinsky",
            "trt",
        ]
        if tensor_product:
            capabilities.extend(
                (
                    "central-moment",
                    "cumulant-unforced",
                    "entropic-unforced",
                    "kbc",
                )
            )
        evidence = LatticeBoltzmannCapabilityEvidence(
            nearest_neighbor=True,
            hydrodynamic_isotropy_order=4,
            tensor_product=tensor_product,
            capabilities=capabilities,
        )

        self.velocities = jnp.asarray(velocity_host, dtype=jnp.int32)
        self.weights = jnp.asarray(weight_host, dtype=jnp.float64)
        self.opposite = jnp.asarray(opposite_host, dtype=jnp.int32)
        self.sound_speed_squared = jnp.asarray(cs2, dtype=jnp.float64)
        self.capability_evidence = evidence
        self.velocity_tuples = tuple(
            tuple(int(value) for value in row) for row in velocity_host
        )
        self.opposite_indices = tuple(int(value) for value in opposite_host)
        self.name = name_
        self.dimension = dimension
        self.population_count = q
        self.lattice_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-velocity-set",
                "name": name_,
                "velocities": velocity_host.tolist(),
                "weights": weight_host.tolist(),
                "opposite": opposite_host.tolist(),
                "sound_speed_squared": cs2,
                "capability_evidence": evidence.evidence_id,
            }
        )

    def supports(self, capability: str, /) -> bool:
        return self.capability_evidence.supports(capability)

    def require(self, capability: str, /) -> None:
        self.capability_evidence.require(capability)


def certified_nearest_neighbor_velocity_set(
    name: str,
    velocities: ArrayLike,
    weights: ArrayLike,
    opposite: Sequence[int] | ArrayLike,
    /,
    *,
    sound_speed_squared: float = 1.0 / 3.0,
) -> LatticeBoltzmannVelocitySet:
    """Certify a custom local velocity set or reject it without a partial plan."""

    return LatticeBoltzmannVelocitySet(
        name,
        velocities,
        weights,
        opposite,
        sound_speed_squared=sound_speed_squared,
    )


def D2Q9() -> LatticeBoltzmannVelocitySet:
    """Return the certified standard D2Q9 velocity set."""

    return LatticeBoltzmannVelocitySet(
        "D2Q9",
        (
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
        ),
        (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36),
        (0, 2, 1, 4, 3, 6, 5, 8, 7),
    )


def D3Q19() -> LatticeBoltzmannVelocitySet:
    """Return the certified standard D3Q19 velocity set."""

    return LatticeBoltzmannVelocitySet(
        "D3Q19",
        (
            (0, 0, 0),
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
            (1, 1, 0),
            (-1, -1, 0),
            (1, -1, 0),
            (-1, 1, 0),
            (1, 0, 1),
            (-1, 0, -1),
            (1, 0, -1),
            (-1, 0, 1),
            (0, 1, 1),
            (0, -1, -1),
            (0, 1, -1),
            (0, -1, 1),
        ),
        (
            1 / 3,
            1 / 18,
            1 / 18,
            1 / 18,
            1 / 18,
            1 / 18,
            1 / 18,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
            1 / 36,
        ),
        (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17),
    )


def D3Q27() -> LatticeBoltzmannVelocitySet:
    """Return the certified tensor-product D3Q27 velocity set."""

    velocities = (
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 1, 0),
        (-1, -1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (1, 0, 1),
        (-1, 0, -1),
        (1, 0, -1),
        (-1, 0, 1),
        (0, 1, 1),
        (0, -1, -1),
        (0, 1, -1),
        (0, -1, 1),
        (1, 1, 1),
        (-1, -1, -1),
        (1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, -1, -1),
    )
    weights = tuple(
        8 / 27
        if sum(value != 0 for value in velocity) == 0
        else 2 / 27
        if sum(value != 0 for value in velocity) == 1
        else 1 / 54
        if sum(value != 0 for value in velocity) == 2
        else 1 / 216
        for velocity in velocities
    )
    lookup: dict[tuple[int, int, int], int] = {
        velocity: index for index, velocity in enumerate(velocities)
    }
    opposite = tuple(
        lookup[(-velocity[0], -velocity[1], -velocity[2])] for velocity in velocities
    )
    return LatticeBoltzmannVelocitySet("D3Q27", velocities, weights, opposite)


__all__ = [
    "D2Q9",
    "D3Q19",
    "D3Q27",
    "LatticeBoltzmannCapabilityEvidence",
    "LatticeBoltzmannVelocitySet",
    "certified_nearest_neighbor_velocity_set",
]
