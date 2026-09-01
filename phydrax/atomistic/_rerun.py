#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from contextlib import ExitStack
from itertools import islice

import equinox as eqx
import jax.numpy as jnp

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import AbstractPreparedParticleNeighborhood
from ._frame import (
    AbstractAtomisticTrajectorySourcePlan,
    AtomisticFrame,
    AtomisticFrameFields,
)
from ._hybrid import evaluate_force_group
from ._potential_program import (
    AtomisticPotentialEvaluation,
    PreparedAtomisticPotentialProgram,
)
from ._reporter import AtomisticReporterPlan
from ._sites import AtomisticSiteDomain


def _chunked_frames(stream, capacity: int, /):
    iterator = iter(stream)
    while True:
        chunk = tuple(islice(iterator, capacity))
        if not chunk:
            return
        yield from chunk


class AtomisticRerunPlan(StrictModule):
    source: AbstractAtomisticTrajectorySourcePlan
    potential: PreparedAtomisticPotentialProgram
    neighborhood: AbstractPreparedParticleNeighborhood
    force_groups: tuple[int, ...] = eqx.field(static=True)
    lambda_values: tuple[float, ...] = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    reporter: AtomisticReporterPlan | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source,
        potential,
        neighborhood,
        /,
        *,
        force_groups=(),
        lambda_values=(1.0,),
        chunk_size: int = 64,
        reporter=None,
    ):
        if not isinstance(source, AbstractAtomisticTrajectorySourcePlan):
            raise TypeError("source must be an atomistic trajectory source plan.")
        if not isinstance(potential, PreparedAtomisticPotentialProgram):
            raise TypeError("potential must be PreparedAtomisticPotentialProgram.")
        if not isinstance(neighborhood, AbstractPreparedParticleNeighborhood):
            raise TypeError("neighborhood must be a prepared particle neighborhood.")
        groups = tuple(int(value) for value in force_groups)
        lambdas = tuple(float(value) for value in lambda_values)
        if (
            not lambdas
            or any(not math.isfinite(value) for value in lambdas)
            or any(value < 0 for value in groups)
        ):
            raise ValueError(
                "Rerun lambdas must be finite and force groups non-negative."
            )
        chunk = int(chunk_size)
        if chunk <= 0:
            raise ValueError("Rerun chunk_size must be positive.")
        if reporter is not None and not isinstance(reporter, AtomisticReporterPlan):
            raise TypeError("reporter must be AtomisticReporterPlan or None.")
        if reporter is not None and reporter.sink.sink_id == source.source_id:
            raise ValueError(
                "Rerun input and reporter output must be different resources."
            )
        self.source = source
        self.potential = potential
        self.neighborhood = neighborhood
        self.force_groups = groups
        self.lambda_values = lambdas
        self.chunk_size = chunk
        self.reporter = reporter
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-rerun",
                "source": source.source_id,
                "potential": potential.prepared_id,
                "neighborhood": neighborhood.prepared_id,
                "groups": list(groups),
                "lambdas": list(lambdas),
                "chunk_size": chunk,
                "reporter": None if reporter is None else reporter.reporter_id,
            }
        )

    def _context_kwargs(self, frame, /):
        cell = self.potential.system.cell
        if frame.cell_vectors is None:
            return {"cell": cell}
        if cell is None:
            raise ValueError("Periodic rerun frame is incompatible with a finite system.")
        vectors = jnp.asarray(frame.cell_vectors)
        return {
            "cell": cell,
            "cell_vectors": vectors,
            "fractional_positions": cell.fractional_with_vectors(
                frame.positions, vectors
            ),
        }

    def _reported_frame(self, frame, evaluations, group_energies, /):
        reporter = self.reporter
        if reporter is None:
            raise RuntimeError("Rerun reporter is not configured.")
        fields = reporter.fields
        if reporter.coordinate_domain is AtomisticSiteDomain.INTERACTION_SITES:
            context_kwargs = self._context_kwargs(frame)
            site_state = self.potential.system.coordinate_map.realize(
                frame.positions,
                cell=context_kwargs["cell"],
                fractional_positions=context_kwargs.get("fractional_positions"),
                cell_vectors=context_kwargs.get("cell_vectors"),
            )
            positions = site_state.positions
            stable_ids = self.potential.system.coordinate_map.plan.sites.site_ids
            velocities = momenta = forces = images = None
        else:
            positions = frame.positions
            stable_ids = frame.stable_ids
            velocities = (
                frame.velocities if fields & AtomisticFrameFields.VELOCITIES else None
            )
            momenta = frame.momenta if fields & AtomisticFrameFields.MOMENTA else None
            forces = (
                evaluations[0].forces if fields & AtomisticFrameFields.FORCES else None
            )
            images = frame.image_counts if fields & AtomisticFrameFields.IMAGES else None
        lambda_energy = jnp.stack(tuple(value.energy for value in evaluations))
        auxiliary = (
            {
                **frame.auxiliary,
                "rerun_lambda_values": jnp.asarray(self.lambda_values),
                "rerun_lambda_energies": lambda_energy,
                "rerun_force_group_energies": jnp.asarray(group_energies),
            }
            if fields & AtomisticFrameFields.AUXILIARY
            else {}
        )
        return AtomisticFrame(
            frame.time,
            frame.step,
            positions,
            stable_ids,
            velocities=velocities,
            momenta=momenta,
            forces=forces,
            cell_vectors=frame.cell_vectors
            if fields & AtomisticFrameFields.CELL
            else None,
            image_counts=images,
            energy=lambda_energy if fields & AtomisticFrameFields.ENERGY else None,
            auxiliary=auxiliary,
            valid=frame.valid
            & jnp.all(jnp.stack(tuple(value.successful for value in evaluations))),
            coordinate_domain=reporter.coordinate_domain,
            system_id=frame.system_id,
            topology_id=frame.topology_id,
            unit_system_id=frame.unit_system_id,
            source_id=f"{frame.source_id}:rerun:{self.plan_id}",
        )

    def run(self, /) -> "AtomisticRerunResult":
        evaluations = []
        group_energies = []
        source_ids = []
        count = 0
        mean = jnp.zeros((len(self.lambda_values),))
        second_moment = jnp.zeros_like(mean)
        minimum = jnp.full_like(mean, jnp.inf)
        maximum = jnp.full_like(mean, -jnp.inf)
        group_mean = jnp.zeros((len(self.lambda_values), len(self.force_groups)))
        with ExitStack() as stack:
            reader = stack.enter_context(self.source.open())
            writer = (
                None
                if self.reporter is None
                else stack.enter_context(self.reporter.sink.open(append=False))
            )
            for frame in _chunked_frames(reader, self.chunk_size):
                if frame.system_id not in (
                    self.potential.system.prepared_id,
                    self.potential.system.plan.system_id,
                ):
                    raise ValueError("Rerun frame belongs to another atomistic system.")
                if frame.coordinate_domain is AtomisticSiteDomain.INTERACTION_SITES:
                    raise ValueError(
                        "Rerun input must contain physical degree-of-freedom atoms."
                    )
                neighborhood = self.neighborhood.build(frame.positions)
                context_kwargs = self._context_kwargs(frame)
                lambda_evaluations = tuple(
                    self.potential.evaluate(
                        frame.positions,
                        neighborhood,
                        alchemical_lambda=value,
                        **context_kwargs,
                    )
                    for value in self.lambda_values
                )
                groups = tuple(
                    tuple(
                        evaluate_force_group(
                            self.potential,
                            group,
                            frame.positions,
                            neighborhood,
                            alchemical_lambda=lambda_value,
                            **context_kwargs,
                        ).energy
                        for group in self.force_groups
                    )
                    for lambda_value in self.lambda_values
                )
                evaluations.append(lambda_evaluations)
                group_energies.append(groups)
                source_ids.append(frame.source_id)
                count += 1
                energy = jnp.stack(tuple(value.energy for value in lambda_evaluations))
                delta = energy - mean
                mean = mean + delta / count
                second_moment = second_moment + delta * (energy - mean)
                minimum = jnp.minimum(minimum, energy)
                maximum = jnp.maximum(maximum, energy)
                if self.force_groups:
                    group_value = jnp.stack(tuple(jnp.stack(row) for row in groups))
                    group_mean = group_mean + (group_value - group_mean) / count
                if (
                    writer is not None
                    and self.reporter is not None
                    and int(frame.step) % self.reporter.stride == 0
                ):
                    writer.write(self._reported_frame(frame, lambda_evaluations, groups))
        successful = count > 0 and all(
            bool(value.successful) for row in evaluations for value in row
        )
        reduction = AtomisticRerunReduction(
            count,
            mean,
            second_moment / max(count - 1, 1),
            minimum,
            maximum,
            group_mean,
            jnp.asarray(successful),
        )
        return AtomisticRerunResult(
            tuple(evaluations),
            tuple(group_energies),
            tuple(source_ids),
            reduction,
            jnp.asarray(successful),
            self.plan_id,
        )


class AtomisticRerunReduction(StrictModule):
    frame_count: int = eqx.field(static=True)
    mean_energies: jnp.ndarray
    energy_variances: jnp.ndarray
    minimum_energies: jnp.ndarray
    maximum_energies: jnp.ndarray
    mean_force_group_energies: jnp.ndarray
    successful: jnp.ndarray


class AtomisticRerunResult(StrictModule):
    evaluations: tuple[tuple[AtomisticPotentialEvaluation, ...], ...]
    force_group_energies: tuple[tuple[tuple[jnp.ndarray, ...], ...], ...]
    source_ids: tuple[str, ...] = eqx.field(static=True)
    reduction: AtomisticRerunReduction
    successful: jnp.ndarray
    plan_id: str = eqx.field(static=True)


__all__ = ["AtomisticRerunPlan", "AtomisticRerunReduction", "AtomisticRerunResult"]
