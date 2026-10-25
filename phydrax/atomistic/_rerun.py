#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from itertools import islice
from typing import NamedTuple, Required, TypedDict

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import PeriodicCell
from ..typing import checked
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
from ._neighborhood_bundle import (
    AtomisticNeighborhoodBundleState,
    PreparedAtomisticNeighborhoodBundle,
)
from ._potential_program import (
    AtomisticPotentialEvaluation,
    PreparedAtomisticPotentialProgram,
)
from ._relaxation import trial_geometry
from ._reporter import AtomisticReporterPlan
from ._sites import AtomisticSiteDomain


_RerunPotential = PreparedAtomisticPotentialProgram | PreparedControlledHamiltonian


class _RerunCellKwargs(TypedDict, total=False):
    cell: Required[PeriodicCell | None]
    cell_vectors: Array
    fractional_positions: Array


class FrameResources(NamedTuple):
    """One frame's bound resources; lattice context is absent for finite frames."""

    potential: _RerunPotential
    neighborhoods: AtomisticNeighborhoodBundleState
    positions: Array
    cell_vectors: Array | None
    image_counts: Array | None
    growths: int
    exhausted: bool


def frame_resources(
    potential: _RerunPotential,
    previous: AtomisticNeighborhoodBundleState | None,
    positions: Array,
    cell_vectors: Array | None,
    /,
) -> FrameResources:
    """Independently valid resources of one frame, with host capacity replay.

    ``previous`` (an accepted state of the same bundle plan) is refreshed under
    the frame geometry; without it the bundle is initialized.  A capacity-only
    failure selects the next declared capacities and replays the same frame;
    an exhausted ladder is reported, never treated as success.
    """
    bundle = potential.neighborhood_bundle
    geometry, wrapped, images = trial_geometry(potential, bundle, positions, cell_vectors)

    def attempt(
        prepared: PreparedAtomisticNeighborhoodBundle, /
    ) -> AtomisticNeighborhoodBundleState:
        if previous is None:
            return prepared.initialize(geometry)
        return prepared.refresh(prepared.rebase(previous), geometry)

    state = attempt(bundle)
    growths = 0
    exhausted = False
    while bool(np.asarray(state.capacity_only)[0]):
        try:
            bundle = bundle.grow_capacities(state)
        except RuntimeError:
            exhausted = True
            break
        state = attempt(bundle)
        growths += 1
    if bundle.prepared_id != potential.neighborhood_bundle.prepared_id:
        potential = potential.with_neighborhood_bundle(bundle)
    if cell_vectors is None or images is None:
        return FrameResources(potential, state, wrapped, None, None, growths, exhausted)
    return FrameResources(
        potential, state, wrapped, cell_vectors, images, growths, exhausted
    )


def _chunked_frames(
    stream: Iterable[AtomisticFrame], capacity: int, /
) -> Iterator[AtomisticFrame]:
    iterator = iter(stream)
    while True:
        chunk = tuple(islice(iterator, capacity))
        if not chunk:
            return
        yield from chunk


class AtomisticRerunPlan(StrictModule):
    """Re-evaluate a trajectory with one prepared fixed or controlled Hamiltonian.

    Resources are the Hamiltonian's own neighborhood bundle: every frame binds
    its own geometry (current cell and integer images) and refreshes the
    previous frame's accepted state, or initializes the first one, so each
    evaluation is at its certified point.  Capacity-only failures replay the
    frame on the declared grown capacities (``capacity_growths`` in the
    result); scientific failures and an exhausted ladder make the frame and
    the run unsuccessful.
    """

    source: AbstractAtomisticTrajectorySourcePlan
    potential: PreparedAtomisticPotentialProgram | PreparedControlledHamiltonian
    force_groups: tuple[int, ...] = eqx.field(static=True)
    state_indices: tuple[int, ...] = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    reporter: AtomisticReporterPlan | None
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: AbstractAtomisticTrajectorySourcePlan,
        potential: PreparedAtomisticPotentialProgram | PreparedControlledHamiltonian,
        /,
        *,
        force_groups: Iterable[int] = (),
        state_indices: Iterable[int] | None = None,
        chunk_size: int = 64,
        reporter: AtomisticReporterPlan | None = None,
    ) -> None:
        if not isinstance(
            potential,
            (PreparedAtomisticPotentialProgram, PreparedControlledHamiltonian),
        ):
            raise TypeError(
                "potential must be a prepared fixed or controlled Hamiltonian."
            )
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
        self.force_groups = groups
        self.state_indices = states
        self.chunk_size = chunk
        self.reporter = reporter
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-rerun",
                "source": source.source_id,
                "potential": potential.prepared_id,
                "groups": list(groups),
                "state_indices": list(states),
                "chunk_size": chunk,
                "reporter": None if reporter is None else reporter.reporter_id,
            }
        )

    def _context_kwargs(self, frame: AtomisticFrame, /) -> _RerunCellKwargs:
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

    def _reported_frame(
        self,
        frame: AtomisticFrame,
        evaluations: tuple[
            AtomisticPotentialEvaluation | ControlledHamiltonianEvaluation, ...
        ],
        group_energies: tuple[tuple[Array, ...], ...],
        /,
    ) -> AtomisticFrame:
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
        potential = self.potential
        accepted: AtomisticNeighborhoodBundleState | None = None
        growths = 0
        exhausted = False
        resources_successful = True
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
                cell = self.potential.system.cell
                if frame.cell_vectors is not None and cell is None:
                    raise ValueError(
                        "Periodic rerun frame is incompatible with a finite system."
                    )
                coordinate = jnp.asarray(
                    frame.positions, dtype=self.potential.system.plan.coordinate_dtype
                )
                vectors = (
                    None
                    if cell is None
                    else cell.vectors.astype(coordinate.dtype)
                    if frame.cell_vectors is None
                    else jnp.asarray(frame.cell_vectors, dtype=coordinate.dtype)
                )
                resources = frame_resources(potential, accepted, coordinate, vectors)
                potential = resources.potential
                growths += resources.growths
                exhausted = exhausted or resources.exhausted
                neighborhoods = resources.neighborhoods
                resources_valid = bool(np.asarray(neighborhoods.successful)[0])
                if resources_valid:
                    accepted = neighborhoods
                if isinstance(potential, PreparedControlledHamiltonian):
                    state_evaluations = tuple(
                        potential.evaluate(
                            resources.positions,
                            neighborhoods,
                            state_index=value,
                            cell_vectors=resources.cell_vectors,
                            image_counts=resources.image_counts,
                        )
                        for value in self.state_indices
                    )
                    groups = tuple(
                        tuple(
                            potential.force_group_energy(
                                group,
                                resources.positions,
                                neighborhoods,
                                state_index=state_index,
                                cell_vectors=resources.cell_vectors,
                                image_counts=resources.image_counts,
                            )[0]
                            for group in self.force_groups
                        )
                        for state_index in self.state_indices
                    )
                else:
                    state_evaluations = (
                        potential.evaluate(
                            resources.positions,
                            neighborhoods,
                            cell_vectors=resources.cell_vectors,
                            image_counts=resources.image_counts,
                        ),
                    )
                    groups = (
                        tuple(
                            evaluate_force_group(
                                potential,
                                group,
                                resources.positions,
                                neighborhoods,
                                cell_vectors=resources.cell_vectors,
                                image_counts=resources.image_counts,
                            ).energy
                            for group in self.force_groups
                        ),
                    )
                resources_successful = resources_successful and resources_valid
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
            and resources_successful
            and not exhausted
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
            growths,
            exhausted,
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
    """Per-frame evaluations, force-group energies and reductions of one rerun.

    ``capacity_growths`` counts host capacity replays on the declared bundle
    ladders and ``capacity_exhausted`` reports a ladder that could not hold a
    frame (the run is then unsuccessful).
    """

    evaluations: tuple[
        tuple[AtomisticPotentialEvaluation | ControlledHamiltonianEvaluation, ...], ...
    ]
    force_group_energies: tuple[tuple[tuple[jnp.ndarray, ...], ...], ...]
    source_ids: tuple[str, ...] = eqx.field(static=True)
    reduction: AtomisticRerunReduction
    successful: jnp.ndarray
    plan_id: str = eqx.field(static=True)
    capacity_growths: int = eqx.field(static=True)
    capacity_exhausted: bool = eqx.field(static=True)


__all__ = ["AtomisticRerunPlan", "AtomisticRerunReduction", "AtomisticRerunResult"]
