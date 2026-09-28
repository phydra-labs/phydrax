#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Particle processes and recorders composed into a PIC step.

A process acts on the species states of one step at a declared stage:

- ``"momentum"`` processes run after the push and before the drift; they may
  change only proper velocities (collisions, radiation reaction);
- ``"population"`` processes run after the field advance on the end-of-step
  species; they may create/deactivate particles and change charge numbers
  (ionization, pair creation) but must preserve the deposited charge
  pointwise, which the PIC runtime verifies by redeposition. A process that
  declares ``redistributes_charge`` (particle merging/splitting) must instead
  conserve total charge; the runtime then Gauss-projects the field onto the
  redeposited charge through the solver's `PICGaussProjection` capability.

Processes that dissipate or emit radiation declare a `RadiationOwnership`
claim; a coupled run refuses overlapping claims at construction.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Literal, TypeAlias

import equinox as eqx
from jax import Array

from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ...typing import PRNGKey
from ._charge_state import PICSpeciesPlan, PICSpeciesState


RadiationOwnership: TypeAlias = Literal[
    "resolved-field", "subgrid-reaction", "diagnostic-only"
]
PICProcessStage: TypeAlias = Literal["momentum", "population"]


class PICProcessContext(StrictModule):
    """Step data visible to a process.

    ``electric``/``magnetic`` are the total fields gathered for the push at the
    step-start positions of each species; ``key`` is the process's own
    stateless substream for this step, or ``None`` for deterministic processes.

    For processes declaring ``requires_field_derivatives`` the runtime also
    supplies, per species, the spatial gradients ``∂F_i/∂x_j`` (``[N, 3, 3]``,
    order-one gather plus exact external-field derivatives) and the time
    derivatives ``∂_t F`` (``[N, 3]``) of the same total fields; grid-field
    time derivatives come from the staggered field history (the previous
    accepted field state gathered at the current positions).
    ``grid_cutoff_frequency`` is the highest angular frequency the field
    discretization resolves, ``min(π c / min Δx, π / Δt)``; ``None`` outside a
    field grid.
    """

    __strict_contract__ = True

    species: tuple[PICSpeciesState, ...]
    electric: tuple[Array, ...]
    magnetic: tuple[Array, ...]
    time: Array
    step_size: Array
    step_index: Array
    key: PRNGKey | None
    electric_gradient: tuple[Array, ...] | None = None
    magnetic_gradient: tuple[Array, ...] | None = None
    electric_rate: tuple[Array, ...] | None = None
    magnetic_rate: tuple[Array, ...] | None = None
    grid_cutoff_frequency: Array | None = None


class PICProcessRadiation(StrictModule):
    """Energy a process hands to radiation the field does not resolve.

    ``radiated_energy`` is this step's total over all bound particles.
    ``maximum_critical_frequency`` is the largest critical angular frequency of
    the emitting particles; ``minimum_scale_separation`` the smallest ratio of
    an emitting particle's critical frequency to the grid cutoff (``inf``
    without a grid). ``scale_separated`` is false when an emitting particle
    radiates at frequencies the grid resolves, so a subgrid claim would
    double-count field-resolved radiation.
    """

    radiated_energy: Array
    maximum_critical_frequency: Array
    minimum_scale_separation: Array
    scale_separated: Array


class PICProcessLedger(StrictModule):
    """Conservation ledger reported by one process application.

    ``radiation`` is present exactly for processes that emit radiation.
    """

    event_count: Array
    charge_defect: Array
    momentum_defect: Array
    energy_defect: Array
    successful: Array
    process_id: str = eqx.field(static=True)
    radiation: PICProcessRadiation | None = None


class PICProcessResult(StrictModule):
    """Accepted species, conservation ledger, and optional process evidence."""

    species: tuple[PICSpeciesState, ...]
    ledger: PICProcessLedger
    evidence: Any = None


class AbstractPICProcess(StrictModule):
    """One particle process bound to explicit species indices."""

    process_id: eqx.AbstractVar[str]
    stage: eqx.AbstractVar[PICProcessStage]
    stochastic: eqx.AbstractVar[bool]
    radiation_ownership: eqx.AbstractVar[RadiationOwnership | None]
    species_indices: eqx.AbstractVar[tuple[int, ...]]
    # Population processes that move charge between grid locations while
    # conserving its total (resampling) set this; they require a field solver
    # implementing `PICGaussProjection`.
    redistributes_charge: ClassVar[bool] = False

    @property
    def requires_field_derivatives(self) -> bool:
        """Whether the force depends on field derivatives (full Landau–Lifshitz).

        The runtime then keeps a staggered field history and supplies field
        gradients and time derivatives in the momentum-stage context.
        """
        return False

    def validate_run(
        self,
        species: tuple[PICSpeciesPlan, ...],
        relativity: RelativityScaleContract,
        /,
    ) -> None:
        """Refuse, at run construction, species or pusher units it cannot act on.

        Processes without such constraints accept every run.
        """
        del species, relativity

    @abc.abstractmethod
    def apply(
        self,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        /,
    ) -> PICProcessResult:
        """Return accepted species states (unchanged on failure) and the ledger."""
        raise NotImplementedError


class AbstractPICRecorder(StrictModule):
    """Diagnostic recorder updated on every accepted PIC step.

    Recorders observe; they own no radiation energy and never alter the run.
    """

    recorder_id: eqx.AbstractVar[str]

    def validate_run(
        self,
        species: tuple[PICSpeciesPlan, ...],
        relativity: RelativityScaleContract,
        /,
    ) -> None:
        """Refuse, at run construction, species or pusher units it cannot observe.

        Recorders without such constraints accept every run.
        """
        del species, relativity

    def shift_frame(self, state: Any, axis: int, distance: float, /) -> Any:
        """Return ``state`` after the particle frame moved by ``distance`` along ``axis``.

        A moving window translates window-local positions by ``−distance``;
        recorders whose data depend on positions accumulate the offset so that
        they keep observing in the fixed frame. Other recorders return ``state``.
        """
        del axis, distance
        return state

    @abc.abstractmethod
    def initialize(
        self,
        species: tuple[PICSpeciesState, ...],
        time: Array,
        /,
    ) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def record(
        self,
        state: Any,
        species: tuple[PICSpeciesState, ...],
        time: Array,
        step_index: Array,
        /,
    ) -> Any:
        raise NotImplementedError


__all__ = [
    "AbstractPICProcess",
    "AbstractPICRecorder",
    "PICProcessContext",
    "PICProcessLedger",
    "PICProcessRadiation",
    "PICProcessResult",
    "PICProcessStage",
    "RadiationOwnership",
]
