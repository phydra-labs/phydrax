#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Binary and background collisions as momentum-stage PIC processes.

The collision operators act on the proper velocity ``u`` of the step's
species. ``m u`` is the relativistic momentum, so their exact pair momentum
invariant carries over; their kinetic-energy invariant is the nonrelativistic
``m |u|^2 / 2`` and the processes are valid for ``|u| << c``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from ...._fingerprint import canonical_fingerprint
from ...._trainable import NonTrainableState
from ....typing import checked, PRNGKey
from .._charge_state import PICSpeciesPlan, PICSpeciesState
from .._process import (
    AbstractPICProcess,
    PICProcessContext,
    PICProcessLedger,
    PICProcessResult,
    PICProcessStage,
    RadiationOwnership,
)
from .._types import PICParticleState
from ._background import BackgroundMCCPlan
from ._coulomb import CoulombCollisionPlan
from ._types import PICCollisionResult


def _collided_species(
    context: PICProcessContext,
    species: int,
    result: PICCollisionResult,
    process_id: str,
    /,
) -> PICProcessResult:
    state = context.species[species]
    updated = PICSpeciesState(
        PICParticleState(state.particles.position, result.accepted_velocity),
        state.population,
        state.charge,
    )
    values = list(context.species)
    values[species] = updated
    dtype = result.energy_defect.dtype
    return PICProcessResult(
        tuple(values),
        PICProcessLedger(
            jnp.sum(result.collided, dtype=jnp.int32),
            jnp.zeros((), dtype=dtype),
            result.momentum_defect,
            result.energy_defect,
            result.successful,
            process_id,
        ),
    )


def _require_key(context: PICProcessContext, /) -> PRNGKey:
    if context.key is None:
        raise ValueError("Stochastic PIC collisions require a derived process key.")
    return context.key


class CoulombCollisionProcess(AbstractPICProcess, NonTrainableState):
    """`CoulombCollisionPlan` applied to one species at the momentum stage."""

    plan: CoulombCollisionPlan
    species: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    stage: PICProcessStage = eqx.field(static=True)
    stochastic: bool = eqx.field(static=True)
    radiation_ownership: RadiationOwnership | None = eqx.field(static=True)
    species_indices: tuple[int, ...] = eqx.field(static=True)

    @checked
    def __init__(self, plan: CoulombCollisionPlan, species: int, /) -> None:
        index = int(species)
        self.plan = plan
        self.species = index
        self.stage = "momentum"
        self.stochastic = True
        self.radiation_ownership = None
        self.species_indices = (index,)
        self.process_id = canonical_fingerprint(
            {
                "kind": "pic-coulomb-collision-process",
                "plan": plan.plan_id,
                "species": index,
            }
        )

    def apply(
        self,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        /,
    ) -> PICProcessResult:
        del species
        key = _require_key(context)
        state = context.species[self.species]
        result = self.plan.collide(
            state.particles.proper_velocity,
            state.population.mass,
            state.population.active,
            state.population.incarnation,
            key,
            context.step_size,
        )
        return _collided_species(context, self.species, result, self.process_id)


class BackgroundCollisionProcess(AbstractPICProcess, NonTrainableState):
    """`BackgroundMCCPlan` applied to one species at the momentum stage."""

    plan: BackgroundMCCPlan
    species: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    stage: PICProcessStage = eqx.field(static=True)
    stochastic: bool = eqx.field(static=True)
    radiation_ownership: RadiationOwnership | None = eqx.field(static=True)
    species_indices: tuple[int, ...] = eqx.field(static=True)

    @checked
    def __init__(self, plan: BackgroundMCCPlan, species: int, /) -> None:
        index = int(species)
        self.plan = plan
        self.species = index
        self.stage = "momentum"
        self.stochastic = True
        self.radiation_ownership = None
        self.species_indices = (index,)
        self.process_id = canonical_fingerprint(
            {
                "kind": "pic-background-collision-process",
                "plan": plan.plan_id,
                "species": index,
            }
        )

    def apply(
        self,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        /,
    ) -> PICProcessResult:
        del species
        key = _require_key(context)
        state = context.species[self.species]
        result = self.plan.collide(
            state.particles.proper_velocity,
            state.population.mass,
            state.population.active,
            key,
            context.step_size,
        )
        return _collided_species(context, self.species, result, self.process_id)


__all__ = ["BackgroundCollisionProcess", "CoulombCollisionProcess"]
