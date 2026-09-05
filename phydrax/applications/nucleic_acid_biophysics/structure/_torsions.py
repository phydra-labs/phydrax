# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....atomistic import PreparedAtomisticSystem
from ....atomistic.sampling import CollectiveVariableKind, CollectiveVariablePlan
from ....series import SampledSeries
from .._binding import NucleotideAtomMapping, prepare_nucleotide_binding


class NucleotideTorsionEvaluation(StrictModule):
    values: Array
    valid: Array
    branch_margin: Array


class SugarPseudorotationEvaluation(StrictModule):
    phase: Array
    amplitude: Array
    harmonic_residual: Array
    valid: Array


class NucleotideTorsionProgram(StrictModule, NonTrainableState):
    """Native atomistic torsion CVs, preserving missing sites and strand ends.

    Values: alpha,beta,gamma,delta,epsilon,zeta,chi,nu0,...,nu4, in radians.
    Runtime is nonperiodic; periodic structures must be explicitly unwrapped.
    """

    variables: tuple
    output_indices: tuple[int, ...] = eqx.field(static=True)
    nucleotide_count: int = eqx.field(static=True)
    program_id: str = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(
        static=True,
        default=(
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "chi",
            "nu0",
            "nu1",
            "nu2",
            "nu3",
            "nu4",
        ),
    )

    def __init__(
        self,
        mapping: NucleotideAtomMapping,
        system: PreparedAtomisticSystem,
        *,
        coordinate_mask=None,
        image_policy="nonperiodic",
    ):
        if image_policy not in ("nonperiodic", "unwrapped") or (
            system.cell is not None and image_policy != "unwrapped"
        ):
            raise ValueError("Periodic torsion coordinates must be explicitly unwrapped.")
        binding = prepare_nucleotide_binding(
            mapping, system, coordinate_mask=coordinate_mask
        )
        lookup = {
            (key, name): int(row)
            for key, name, row, available in zip(
                mapping.nucleotide_keys,
                mapping.atom_names,
                np.asarray(binding.atom_indices),
                np.asarray(binding.atom_mask),
                strict=True,
            )
            if available
        }
        previous = {b: a for a, b in mapping.construct.directed_edges}
        following = dict(mapping.construct.directed_edges)
        variables, indices = [], []
        sugar = ("O4'", "C1'", "C2'", "C3'", "C4'")
        for n, (key, base) in enumerate(
            zip(mapping.construct.nucleotide_keys, mapping.construct.bases, strict=True)
        ):
            prev, next_ = previous.get(key), following.get(key)
            recipes = [
                ((prev, "O3'"), (key, "P"), (key, "O5'"), (key, "C5'")),
                tuple((key, a) for a in ("P", "O5'", "C5'", "C4'")),
                tuple((key, a) for a in ("O5'", "C5'", "C4'", "C3'")),
                tuple((key, a) for a in ("C5'", "C4'", "C3'", "O3'")),
                ((key, "C4'"), (key, "C3'"), (key, "O3'"), (next_, "P")),
                ((key, "C3'"), (key, "O3'"), (next_, "P"), (next_, "O5'")),
                tuple(
                    (key, a)
                    for a in (
                        ("O4'", "C1'", "N9", "C4")
                        if base in "AG"
                        else ("O4'", "C1'", "N1", "C2")
                    )
                ),
            ]
            # nu0=C4'-O4'-C1'-C2', then cyclic shifts.
            recipes.extend(
                tuple((key, sugar[(j + k - 1) % 5]) for k in range(4)) for j in range(5)
            )
            for column, recipe in enumerate(recipes):
                if all(atom in lookup for atom in recipe):
                    variables.append(
                        CollectiveVariablePlan(
                            CollectiveVariableKind.TORSION,
                            np.asarray([lookup[a] for a in recipe], dtype=np.int64),
                        ).prepare(system)
                    )
                    indices.append(12 * n + column)
        self.variables, self.output_indices = tuple(variables), tuple(indices)
        self.nucleotide_count = mapping.construct.nucleotide_count
        self.program_id = canonical_fingerprint(
            {
                "binding": binding.binding_id,
                "variables": [variable.prepared_id for variable in variables],
                "outputs": indices,
                "images": image_policy,
            }
        )

    def evaluate(self, positions) -> NucleotideTorsionEvaluation:
        shape = (self.nucleotide_count * 12,)
        values, valid, margin = jnp.zeros(shape), jnp.zeros(shape, bool), jnp.zeros(shape)
        for index, variable in zip(self.output_indices, self.variables, strict=True):
            result = variable.evaluate(positions)
            values = values.at[index].set(jnp.where(result.successful, result.value, 0.0))
            valid = valid.at[index].set(result.successful)
            margin = margin.at[index].set(result.branch_margin)
        return NucleotideTorsionEvaluation(
            values.reshape((-1, 12)), valid.reshape((-1, 12)), margin.reshape((-1, 12))
        )

    def observe_series(self, coordinates: SampledSeries) -> SampledSeries:
        """Preserve series/reset and per-coordinate availability semantics."""
        values = jnp.asarray(coordinates.values)
        if values.ndim < 3 or values.shape[-1] != 3:
            raise ValueError(
                "Torsion observations require sampled Cartesian coordinates."
            )
        prefix = values.shape[:-2]
        evaluated = jax.vmap(self.evaluate)(values.reshape((-1,) + values.shape[-2:]))
        shape = prefix + (self.nucleotide_count, 12)
        valid = evaluated.valid.reshape(prefix + (self.nucleotide_count * 12,))
        if coordinates.value_valid is not None:
            source_valid = jnp.broadcast_to(coordinates.value_valid, values.shape)
            for output, variable in zip(self.output_indices, self.variables, strict=True):
                available = jnp.all(
                    source_valid[..., variable.plan.indices, :], axis=(-2, -1)
                )
                valid = valid.at[..., output].set(valid[..., output] & available)
        return SampledSeries(
            coordinates.support,
            evaluated.values.reshape(shape),
            value_valid=valid.reshape(shape),
            series_id=canonical_fingerprint(
                {"source": coordinates.series_id, "torsions": self.program_id}
            ),
        )

    def observe_pseudorotation_series(self, coordinates: SampledSeries) -> SampledSeries:
        torsions = self.observe_series(coordinates)
        result = sugar_pseudorotation(
            NucleotideTorsionEvaluation(
                torsions.values, torsions.value_valid, jnp.zeros_like(torsions.values)
            )
        )
        values = jnp.stack(
            (result.phase, result.amplitude, result.harmonic_residual), axis=-1
        )
        return SampledSeries(
            coordinates.support,
            values,
            value_valid=jnp.broadcast_to(result.valid[..., None], values.shape),
            series_id=canonical_fingerprint(
                {"source": torsions.series_id, "pseudorotation": "five-ring-Fourier"}
            ),
        )


def sugar_pseudorotation(
    torsions: NucleotideTorsionEvaluation,
) -> SugarPseudorotationEvaluation:
    """Least-squares five-ring Fourier pucker, not an exact noisy-ring AS fit.

    nu_j = amplitude*cos(phase+4*pi*(j-2)/5). Phase is undefined for a
    planar sugar and explicitly masked. Harmonic residual qualifies nonideal
    rings; this descriptor does not assign an empirical conformer state.
    """
    nu = torsions.values[..., 7:12]
    theta = 4 * jnp.pi * (jnp.arange(5) - 2) / 5
    cosine = (2 / 5) * jnp.sum(nu * jnp.cos(theta), axis=-1)
    sine = -(2 / 5) * jnp.sum(nu * jnp.sin(theta), axis=-1)
    square = cosine**2 + sine**2
    amplitude = jnp.sqrt(jnp.where(square > 0, square, 1.0))
    valid = jnp.all(torsions.valid[..., 7:12], axis=-1) & (square > 1e-20)
    phase = jnp.arctan2(jnp.where(valid, sine, 0.0), jnp.where(valid, cosine, 1.0))
    fitted = cosine[..., None] * jnp.cos(theta) - sine[..., None] * jnp.sin(theta)
    residual = jnp.sqrt(jnp.mean((nu - fitted) ** 2, axis=-1))
    return SugarPseudorotationEvaluation(
        phase, jnp.where(valid, amplitude, 0.0), residual, valid
    )


__all__ = [
    "NucleotideTorsionEvaluation",
    "SugarPseudorotationEvaluation",
    "NucleotideTorsionProgram",
    "sugar_pseudorotation",
]
