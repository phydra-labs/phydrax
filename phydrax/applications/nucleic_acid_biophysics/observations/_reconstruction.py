# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native distance-CV/least-squares reconstruction with independent chirality."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....atomistic import PreparedAtomisticSystem
from ....atomistic.sampling import CollectiveVariableKind, CollectiveVariablePlan
from ....optim import least_squares, LevenbergMarquardt
from ....qualification import ReferenceArtifactManifest
from ....units import conversion_factor, UnitDefinition


class ChiralityEvaluation(StrictModule):
    signed_volume: Array
    correct: Array
    degenerate: Array


class IntervalReconstructionResult(StrictModule):
    initial_positions: Array
    positions: Array
    optimization: object
    distances: Array
    interval_satisfied: Array
    chirality: ChiralityEvaluation
    restraints_satisfied: Array
    chirality_qualified: Array
    # A local restraint solution is not an all-atom physical qualification.
    reconstruction_id: str = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)


class IntervalDistanceReconstruction(StrictModule):
    """Fixed-support distance intervals and explicit signed tetrahedral volumes.

    Uses the native atomistic distance CV and native optimizer. Intervals alone
    cannot distinguish reflections; an empty chirality support is accepted only
    as an explicitly unqualified distance reconstruction. No atom completion,
    chemistry assignment or force-field generation is performed.
    """

    system: PreparedAtomisticSystem
    variables: tuple
    lower: Array
    upper: Array
    sigma: Array
    weight: Array
    chirality_indices: Array
    chirality_sign: Array
    minimum_volume: Array
    chirality_sigma: Array
    sources: tuple[ReferenceArtifactManifest, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system,
        atom_pairs,
        lower,
        upper,
        standard_deviation,
        *,
        weights,
        length_unit: UnitDefinition,
        sources,
        requested_use,
        chirality_atom_ids=(),
        chirality_sign=(),
        minimum_volume=(),
        chirality_standard_deviation=(),
    ):
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("Reconstruction must consume an existing atomistic support.")
        if system.cell is not None:
            raise ValueError(
                "Reconstruction requires a nonperiodic support; unwrap explicitly into a separate realization."
            )
        sources = tuple(sources)
        if not sources or any(
            not isinstance(source, ReferenceArtifactManifest) for source in sources
        ):
            raise ValueError(
                "Restraints require independent source rights and uncertainty provenance."
            )
        for source in sources:
            source.require_rights(**requested_use)
        factor = float(
            conversion_factor(length_unit, system.plan.units.scale.length_unit)
        )
        pairs = np.asarray(atom_pairs)
        lo, hi, sigma, weight = (
            np.asarray(value, float)
            for value in (lower, upper, standard_deviation, weights)
        )
        if (
            pairs.ndim != 2
            or pairs.shape[1] != 2
            or not np.issubdtype(pairs.dtype, np.integer)
            or pairs.shape[0] == 0
        ):
            raise ValueError("Distance restraints require explicit stable atom-ID pairs.")
        if (
            any(x.shape != (pairs.shape[0],) for x in (lo, hi, sigma, weight))
            or any(np.any(~np.isfinite(x)) for x in (lo, hi, sigma, weight))
            or np.any(lo < 0)
            or np.any(hi < lo)
            or np.any(sigma <= 0)
            or np.any(weight <= 0)
        ):
            raise ValueError(
                "Intervals, measured uncertainty and weights must be finite, positive and aligned."
            )
        ids = np.asarray(system.plan.particle_ids)
        lookup = {
            int(atom): row
            for row, atom in enumerate(ids)
            if bool(system.active_mask[row])
        }
        if any(int(atom) not in lookup for atom in pairs.flat) or np.any(
            pairs[:, 0] == pairs[:, 1]
        ):
            raise ValueError(
                "Distance restraints require distinct active atom identities."
            )
        indices = np.asarray(
            [[lookup[int(a)], lookup[int(b)]] for a, b in pairs], dtype=np.int64
        )
        self.variables = tuple(
            CollectiveVariablePlan(CollectiveVariableKind.DISTANCE, pair).prepare(system)
            for pair in indices
        )
        tetra = np.asarray(chirality_atom_ids, dtype=np.int64).reshape((-1, 4))
        signs, minimum, ch_sigma = (
            np.asarray(x, float)
            for x in (chirality_sign, minimum_volume, chirality_standard_deviation)
        )
        if (
            any(x.shape != (tetra.shape[0],) for x in (signs, minimum, ch_sigma))
            or np.any(~np.isin(signs, (-1, 1)))
            or np.any(~np.isfinite(minimum))
            or np.any(~np.isfinite(ch_sigma))
            or np.any(minimum <= 0)
            or np.any(ch_sigma <= 0)
        ):
            raise ValueError(
                "Chirality needs signs ±1, positive volume margins and volume uncertainties."
            )
        if any(len(set(row)) != 4 for row in tetra) or any(
            int(atom) not in lookup for atom in tetra.flat
        ):
            raise ValueError("Chirality requires four distinct active stable atom IDs.")
        self.chirality_indices = jnp.asarray(
            [[lookup[int(atom)] for atom in row] for row in tetra], dtype=jnp.int64
        ).reshape((-1, 4))
        self.system, self.sources = system, sources
        self.lower, self.upper, self.sigma, self.weight = (
            jnp.asarray(lo * factor),
            jnp.asarray(hi * factor),
            jnp.asarray(sigma * factor),
            jnp.asarray(weight),
        )
        self.chirality_sign, self.minimum_volume, self.chirality_sigma = (
            jnp.asarray(signs),
            jnp.asarray(minimum * factor**3),
            jnp.asarray(ch_sigma * factor**3),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "interval-nucleic-reconstruction",
                "system": system.prepared_id,
                "pairs": pairs.tolist(),
                "lower": lo.tolist(),
                "upper": hi.tolist(),
                "sigma": sigma.tolist(),
                "weights": weight.tolist(),
                "unit": length_unit.unit_id,
                "tetra": tetra.tolist(),
                "signs": signs.tolist(),
                "minimum": minimum.tolist(),
                "chirality_sigma": ch_sigma.tolist(),
                "sources": [s.manifest_id for s in sources],
            }
        )

    def distances(self, positions):
        return jnp.stack(
            tuple(variable.evaluate(positions).value for variable in self.variables)
        )

    def chirality(self, positions):
        points = jnp.asarray(positions)[self.chirality_indices]
        vectors = points[:, 1:] - points[:, 0, None, :]
        volume = (
            jnp.sum(vectors[:, 0] * jnp.cross(vectors[:, 1], vectors[:, 2]), axis=-1) / 6
        )
        signed = self.chirality_sign * volume
        return ChiralityEvaluation(
            signed, signed >= self.minimum_volume, jnp.abs(volume) < self.minimum_volume
        )

    def reconstruct(
        self, initial_positions, *, fixed_mask, termination=None, interval_tolerance=1e-7
    ):
        initial = np.asarray(initial_positions, float)
        fixed = np.asarray(fixed_mask, bool)
        if (
            initial.shape != (self.system.capacity, 3)
            or fixed.shape != initial.shape
            or not np.all(np.isfinite(initial))
        ):
            raise ValueError(
                "Initial coordinates and explicit fixed-coordinate mask must match support."
            )
        if not np.isfinite(interval_tolerance) or interval_tolerance < 0:
            raise ValueError(
                "Interval tolerance must be a finite nonnegative native length."
            )
        # Nonmobile atoms are fixed even when the coordinate-gauge mask omits them.
        fixed = fixed | ~np.asarray(self.system.mobile_mask)[:, None]
        free = np.flatnonzero(~fixed.reshape(-1))
        if free.size == 0:
            raise ValueError("Reconstruction needs at least one mobile coordinate.")
        initial_array = jnp.asarray(initial)

        def realize(values):
            return initial_array.reshape(-1).at[free].set(values).reshape(initial.shape)

        def residual(values, args):
            del args
            positions = realize(values)
            distance = self.distances(positions)
            interval = jnp.minimum(distance - self.lower, 0.0) + jnp.maximum(
                distance - self.upper, 0.0
            )
            chirality = self.chirality(positions)
            return (
                jnp.sqrt(self.weight) * interval / self.sigma,
                jnp.minimum(chirality.signed_volume - self.minimum_volume, 0.0)
                / self.chirality_sigma,
            )

        result = least_squares(
            residual,
            initial_array.reshape(-1)[free],
            method=LevenbergMarquardt(),
            termination=termination,
        )
        positions = realize(result.parameters)
        distance = self.distances(positions)
        satisfied = (distance >= self.lower - interval_tolerance) & (
            distance <= self.upper + interval_tolerance
        )
        chiral = self.chirality(positions)
        identity = canonical_fingerprint(
            {"plan": self.plan_id, "initial": initial.tolist(), "fixed": fixed.tolist()}
        )
        return IntervalReconstructionResult(
            initial_array,
            positions,
            result,
            distance,
            satisfied,
            chiral,
            jnp.all(satisfied),
            jnp.asarray(self.chirality_indices.shape[0] > 0) & jnp.all(chiral.correct),
            identity,
            tuple(source.manifest_id for source in self.sources),
        )


__all__ = [
    "ChiralityEvaluation",
    "IntervalReconstructionResult",
    "IntervalDistanceReconstruction",
]
