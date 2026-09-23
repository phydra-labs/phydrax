#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from contextlib import ExitStack
from itertools import islice

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import AbstractPreparedParticleNeighborhood
from ._alchemical import (
    ControlledHamiltonianEvaluation,
    PreparedControlledHamiltonian,
)
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
    potential: PreparedAtomisticPotentialProgram | PreparedControlledHamiltonian
    neighborhood: AbstractPreparedParticleNeighborhood
    force_groups: tuple[int, ...] = eqx.field(static=True)
    state_indices: tuple[int, ...] = eqx.field(static=True)
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
        state_indices=None,
        chunk_size: int = 64,
        reporter=None,
    ):
        if not isinstance(source, AbstractAtomisticTrajectorySourcePlan):
            raise TypeError("source must be an atomistic trajectory source plan.")
        if not isinstance(
            potential,
            (PreparedAtomisticPotentialProgram, PreparedControlledHamiltonian),
        ):
            raise TypeError(
                "potential must be a prepared fixed or controlled Hamiltonian."
            )
        if not isinstance(neighborhood, AbstractPreparedParticleNeighborhood):
            raise TypeError("neighborhood must be a prepared particle neighborhood.")
        groups = tuple(force_groups)
        if any(value < 0 for value in groups):
            raise ValueError("Rerun force groups must be non-negative.")
        if isinstance(potential, PreparedControlledHamiltonian):
            states = (
                tuple(range(potential.plan.schedule.state_count))
                if state_indices is None
                else tuple(state_indices)
            )
            if not states or any(
                value < 0 or value >= potential.plan.schedule.state_count
                for value in states
            ):
                raise ValueError(
                    "Rerun state indices must identify controlled schedule states."
                )
        else:
            if state_indices is not None:
                raise ValueError("state_indices require a PreparedControlledHamiltonian.")
            states = (0,)
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
        self.state_indices = states
        self.chunk_size = chunk
        self.reporter = reporter
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-rerun",
                "source": source.source_id,
                "potential": potential.prepared_id,
                "neighborhood": neighborhood.prepared_id,
                "groups": list(groups),
                "state_indices": list(states),
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
        state_energy = jnp.stack(tuple(value.energy for value in evaluations))
        auxiliary = (
            {
                **frame.auxiliary,
                "rerun_state_indices": jnp.asarray(self.state_indices),
                "rerun_state_energies": state_energy,
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
            energy=state_energy if fields & AtomisticFrameFields.ENERGY else None,
            auxiliary=auxiliary,
            valid=frame.valid
            & jnp.all(jnp.stack(tuple(value.successful for value in evaluations))),
            coordinate_domain=reporter.coordinate_domain,
            system_id=frame.system_id,
            topology_id=frame.topology_id,
            units=frame.units,
            source_id=f"{frame.source_id}:rerun:{self.plan_id}",
        )

    def run(self, /) -> "AtomisticRerunResult":
        evaluations = []
        group_energies = []
        source_ids = []
        count = 0
        source_valid = True
        mean = jnp.zeros((len(self.state_indices),))
        second_moment = jnp.zeros_like(mean)
        minimum = jnp.full_like(mean, jnp.inf)
        maximum = jnp.full_like(mean, -jnp.inf)
        group_mean = jnp.zeros((len(self.state_indices), len(self.force_groups)))
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
                if (
                    frame.topology_id != self.potential.system.topology.topology_id
                    or frame.units.unit_system_id
                    != self.potential.system.plan.units.unit_system_id
                ):
                    raise ValueError(
                        "Rerun frame topology or complete unit system is incompatible."
                    )
                if not np.array_equal(
                    np.asarray(frame.stable_ids),
                    np.asarray(self.potential.system.plan.particle_ids),
                ):
                    raise ValueError(
                        "Rerun frame stable IDs must exactly match system particle order."
                    )
                if frame.coordinate_domain is AtomisticSiteDomain.INTERACTION_SITES:
                    raise ValueError(
                        "Rerun input must contain physical degree-of-freedom atoms."
                    )
                neighborhood = self.neighborhood.build(frame.positions)
                context_kwargs = self._context_kwargs(frame)
                if isinstance(self.potential, PreparedControlledHamiltonian):
                    state_evaluations = tuple(
                        self.potential.evaluate(
                            frame.positions,
                            neighborhood,
                            state_index=value,
                            **context_kwargs,
                        )
                        for value in self.state_indices
                    )
                    groups = tuple(
                        tuple(
                            self.potential.force_group_energy(
                                group,
                                frame.positions,
                                neighborhood,
                                state_index=state_index,
                                **context_kwargs,
                            )[0]
                            for group in self.force_groups
                        )
                        for state_index in self.state_indices
                    )
                else:
                    state_evaluations = (
                        self.potential.evaluate(
                            frame.positions, neighborhood, **context_kwargs
                        ),
                    )
                    groups = (
                        tuple(
                            evaluate_force_group(
                                self.potential,
                                group,
                                frame.positions,
                                neighborhood,
                                **context_kwargs,
                            ).energy
                            for group in self.force_groups
                        ),
                    )
                evaluations.append(state_evaluations)
                group_energies.append(groups)
                source_ids.append(frame.source_id)
                source_valid = source_valid and bool(frame.valid)
                count += 1
                energy = jnp.stack(tuple(value.energy for value in state_evaluations))
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
                    writer.write(self._reported_frame(frame, state_evaluations, groups))
        successful = (
            count > 0
            and source_valid
            and all(bool(value.successful) for row in evaluations for value in row)
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
    evaluations: tuple[
        tuple[AtomisticPotentialEvaluation | ControlledHamiltonianEvaluation, ...], ...
    ]
    force_group_energies: tuple[tuple[tuple[jnp.ndarray, ...], ...], ...]
    source_ids: tuple[str, ...] = eqx.field(static=True)
    reduction: AtomisticRerunReduction
    successful: jnp.ndarray
    plan_id: str = eqx.field(static=True)


__all__ = ["AtomisticRerunPlan", "AtomisticRerunReduction", "AtomisticRerunResult"]
