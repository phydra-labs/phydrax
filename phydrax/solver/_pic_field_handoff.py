#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mid-run PIC state handoff between the Yee cochain solver and staggered PSATD.

On a periodic uniform Cartesian grid the degree-1 and degree-2 cochains of
`CochainMaxwellPICFieldSolver` are the edge circulations ``E_a Δ_a`` and the
oriented face fluxes ``B_c Δ_a Δ_b`` at exactly the Yee positions where the
staggered PSATD grid stores ``E_a`` (half a cell along ``a``) and ``B_c`` (half
a cell along the other two axes); both solvers hold the node charge density.
The bridge's circulation/flux maps are therefore an exact, invertible state
conversion.

That conversion preserves Gauss's law identically only when both solvers use
the same discrete divergence. The order-2 staggered PSATD symbols
``D± = i[k]e^{±ikΔ/2}`` with ``[k] = 2 sin(kΔ/2)/Δ`` are exactly the Yee
forward/backward node differences. Infinite-order and higher even-order
stencils have a different ``[k]``, and collocated grids store every component
at the nodes, so a Yee-consistent field violates their Gauss law at
``O((kΔ)²)``; restoring it takes a Poisson projection that changes the physical
field, and those targets are refused. Preservation carries a violated
constraint over unchanged, so the evidence separately certifies that the
converted state satisfies Gauss's law, periodic neutrality, and ``∇·B = 0``
within the target plan's ``constraint_tolerance``.

Both runtimes hold ``E`` and ``B`` at the same integer time: the cochain update
is the symmetric half-kick/kick/half-kick leapfrog, so its ``B`` is the mean of
the Yee half-step fluxes. The cochain energy ledger uses the leapfrog-modified
field energy (plain energy minus ``(Δt²/8)⟨dE, ⋆μ⁻¹dE⟩``) and PSATD's the
plain energy; the handoff reports that term, which is the exact jump of the
ledger's field energy across the handoff.
"""

from __future__ import annotations

from typing import Any, assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._cochain_pic_field import CochainMaxwellPICFieldSolver
from ._electromagnetic_pic import (
    ElectromagneticPICPlan,
    ElectromagneticPICState,
    PICFieldHistory,
)
from ._maxwell import (
    CompatibleMaxwellState,
    MaxwellAuxiliaryState,
    MaxwellPrimaryState,
    PreparedDiagonalMaxwellConstitutive,
)
from ._pic_current_source import PreparedPICMaxwellCurrentSource
from .maxwell.spectral import PreparedSpectralMaxwell, SpectralMaxwellState


PICFieldHandoffRoute: TypeAlias = Literal["cochain-to-spectral", "spectral-to-cochain"]

# Roundoff allowance of the conversion: the maps are exact up to one scaling
# per value, and the evidence sums and FFTs add a few ulps of their operands.
_ROUNDOFF = 64.0 * float(np.finfo(np.float64).eps)


class PICFieldHandoffEvidence(StrictModule, NonTrainableState):
    """Constraint, energy, and charge evidence of one PIC field handoff.

    The ``*_electric_constraint`` fields compare one Gauss residual on both
    sides, the mean-free ``−δD − (ρ − ρ̄)`` of the cochain solver and
    ``ε∇⁻·E − (ρ − ρ̄)`` of PSATD (whose derivative cannot see the mean
    ``ρ̄``), relative to the Gauss operand scale
    ``max|ρ| + ε Σ_a 2 max|E_a|/Δ_a``; the net charge is reported by the
    total charges. Magnetic residuals are ``∇·B`` (the cochain's ``d(B)`` per
    cell volume, PSATD's ``∇⁺·B``) relative to ``Σ_a 2 max|B_a|/Δ_a``. Field
    energies are the plain synchronized ``½∫(εE² + B²/μ)``;
    ``leapfrog_energy_correction`` is the Yee half-kick term the cochain ledger
    subtracts from it, so the PIC ledger's field energy jumps by exactly
    ``+correction`` from cochain to PSATD and ``−correction`` back. Charges are
    ``Σρ ΔV``.

    ``constraint_satisfied`` states that the converted state passes the target
    plan's own absolute per-step constraint check: the target solver's
    max-norm Gauss residual including the mean charge (cochain ``−δD − ρ``,
    PSATD ``∇⁻·E − ρ/ε``, so a non-neutral periodic state fails in either
    direction) and its ``∇·B`` residual (cochain ``d(B)``, PSATD ``∇⁺·B``)
    are within ``target.constraint_tolerance``.

    ``successful`` requires finite converted fields, a positive finite step,
    ``constraint_satisfied``, target relative residuals within roundoff of the
    source's, and field energy and total charge preserved to roundoff.
    """

    route: PICFieldHandoffRoute = eqx.field(static=True)
    source_solver_id: str = eqx.field(static=True)
    target_solver_id: str = eqx.field(static=True)
    source_electric_constraint: Array
    target_electric_constraint: Array
    source_magnetic_constraint: Array
    target_magnetic_constraint: Array
    source_field_energy: Array
    target_field_energy: Array
    energy_relative_difference: Array
    leapfrog_energy_correction: Array
    source_total_charge: Array
    target_total_charge: Array
    constraint_satisfied: Array
    successful: Array


class PICFieldHandoffResult(StrictModule):
    """PIC state converted for the target plan, and its handoff evidence."""

    state: ElectromagneticPICState
    evidence: PICFieldHandoffEvidence


class _HandoffPair(NamedTuple):
    route: PICFieldHandoffRoute
    cochain: CochainMaxwellPICFieldSolver
    spectral: PreparedSpectralMaxwell


class _SolverEvidence(NamedTuple):
    electric: Array
    magnetic: Array
    energy: Array
    charge: Array
    # Absolute max-norm Gauss (mean included) and ∇·B residuals, as the
    # solver's own PIC step certifies them against constraint_tolerance.
    absolute_electric: Array
    absolute_magnetic: Array


def _run_declaration(plan: ElectromagneticPICPlan, /) -> dict[str, Any]:
    """Everything outside the field solver that owns or shapes PIC state."""
    key = plan.random_key
    return {
        "species": [value.plan_id for value in plan.species],
        "processes": [value.process_id for value in plan.processes],
        "boundaries": None if plan.boundaries is None else plan.boundaries.plan_id,
        "recorders": [value.recorder_id for value in plan.recorders],
        "filters": [value.filter_id for value in plan.filters],
        "cherenkov_guards": [value.guard_id for value in plan.cherenkov_guards],
        "external_fields": [value.source_id for value in plan.external_fields],
        "ownership": plan.ownership,
        "precision": plan.precision.policy_id,
        "pusher": plan.pusher.plan_id,
        "random_key": None
        if key is None
        else np.asarray(jax.random.key_data(key)).tolist(),
    }


def _require_same_run(
    source: ElectromagneticPICPlan, target: ElectromagneticPICPlan, /
) -> None:
    first, second = _run_declaration(source), _run_declaration(target)
    differing = sorted(name for name in first if first[name] != second[name])
    if differing:
        raise ValueError(
            "PIC field handoff passes species, process, boundary, and recorder "
            "state through unchanged, so both plans must declare the same run; "
            f"they differ in: {', '.join(differing)}."
        )


def _require_spectral_target(solver: PreparedSpectralMaxwell, /) -> None:
    plan = solver.plan
    if plan.grid != "staggered":
        raise ValueError(
            "PIC field handoff requires grid='staggered': the collocated grid "
            "stores E and B at the nodes, and moving Yee components there changes "
            "the field and breaks its Gauss law."
        )
    if plan.stencil_order != 2:
        raise ValueError(
            "PIC field handoff requires stencil='finite-order', stencil_order=2: "
            "only that PSATD divergence equals the Yee node difference; other "
            "stencils would need a Gauss projection that changes the field."
        )
    if plan.variant != "standard":
        raise ValueError(
            "PIC field handoff requires the standard PSATD variant: Galilean "
            "variants hold grid-frame positions and averaged fields."
        )
    if plan.pml is not None or plan.antennas or plan.observers:
        raise ValueError(
            "PIC field handoff refuses PSATD PML split fields and absorber charge, "
            "antenna sheet charges, and Huygens observer accumulations: the "
            "cochain solver holds no equivalent memory."
        )


def _require_cochain_source(
    solver: CochainMaxwellPICFieldSolver, spectral: PreparedSpectralMaxwell, /
) -> None:
    maxwell = solver.maxwell
    plan = maxwell.plan
    if maxwell.layout.electric_degree != 1 or maxwell.layout.magnetic_degree != 2:
        raise ValueError("PIC field handoff requires full 3-D edge E and face B.")
    if plan.boundaries or plan.pml is not None or plan.observers:
        raise ValueError(
            "PIC field handoff refuses Maxwell boundaries, CPML memory, and "
            "observer accumulations: PSATD holds no equivalent state."
        )
    if plan.harmonic_constraint is not None:
        raise ValueError(
            "PIC field handoff refuses a harmonic flux constraint, which PSATD "
            "does not declare."
        )
    if len(maxwell.sources) != 1 or not isinstance(
        maxwell.sources[0], PreparedPICMaxwellCurrentSource
    ):
        raise ValueError(
            "PIC field handoff requires the PIC current as the only Maxwell source."
        )
    constitutive = maxwell.constitutive
    if not isinstance(constitutive, PreparedDiagonalMaxwellConstitutive):
        raise ValueError(
            "PIC field handoff requires a lossless stateless diagonal medium: "
            "conductive, dispersive, and plasma media carry loss or memory PSATD "
            "does not hold."
        )
    permittivity = np.asarray(constitutive.permittivity)
    permeability = np.asarray(constitutive.permeability)
    if np.any(permittivity != spectral.plan.permittivity) or np.any(
        permeability != spectral.plan.permeability
    ):
        raise ValueError(
            "PIC field handoff requires the homogeneous medium of the PSATD plan."
        )


def _admissible_pair(
    source: ElectromagneticPICPlan, target: ElectromagneticPICPlan, /
) -> _HandoffPair:
    if not isinstance(source, ElectromagneticPICPlan) or not isinstance(
        target, ElectromagneticPICPlan
    ):
        raise TypeError("source and target must be ElectromagneticPICPlan.")
    solvers = (source.solver, target.solver)
    match solvers:
        case (
            CochainMaxwellPICFieldSolver() as cochain,
            PreparedSpectralMaxwell() as spectral,
        ):
            pair = _HandoffPair("cochain-to-spectral", cochain, spectral)
        case (
            PreparedSpectralMaxwell() as spectral,
            CochainMaxwellPICFieldSolver() as cochain,
        ):
            pair = _HandoffPair("spectral-to-cochain", cochain, spectral)
        case _:
            raise TypeError(
                "PIC field handoff is defined between CochainMaxwellPICFieldSolver "
                "and PreparedSpectralMaxwell only; wrapped (distributed) and other "
                "field solvers are not converted."
            )
    _require_spectral_target(pair.spectral)
    _require_cochain_source(pair.cochain, pair.spectral)
    _require_same_run(source, target)
    if pair.cochain.bridge.bridge_id != pair.spectral.plan.bridge.bridge_id:
        raise ValueError("PIC field handoff requires one shared grid bridge.")
    if [value.prepared_id for value in pair.cochain.transfers] != [
        value.prepared_id for value in pair.spectral.transfers
    ] or [value.plan_id for value in pair.cochain.currents] != [
        value.plan_id for value in pair.spectral.currents
    ]:
        raise ValueError(
            "PIC field handoff requires the same particle transfers and current "
            "plans, so the field charge stays the particles' deposit."
        )
    return pair


def _to_spectral(pair: _HandoffPair, field: Any, /) -> SpectralMaxwellState:
    if not isinstance(field, CompatibleMaxwellState):
        raise TypeError("The cochain PIC field must be CompatibleMaxwellState.")
    maxwell = pair.cochain.maxwell
    bridge = pair.cochain.bridge
    return SpectralMaxwellState(
        electric=jnp.stack(
            bridge.unpack_edge_circulation(maxwell.electric_field(field)), axis=-1
        ),
        magnetic=jnp.stack(
            bridge.unpack_face_flux(maxwell.magnetic_flux(field)), axis=-1
        ),
        charge=bridge.unpack(0, field.primary.charge)[0],
        averaged_electric=None,
        averaged_magnetic=None,
        electric_split=None,
        magnetic_split=None,
        absorber_charge=None,
        absorber_magnetic_charge=None,
        antenna_charge=None,
        antenna_magnetic_charge=None,
        observations=(),
    )


def _to_cochain(pair: _HandoffPair, field: Any, /) -> CompatibleMaxwellState:
    if not isinstance(field, SpectralMaxwellState):
        raise TypeError("The spectral PIC field must be SpectralMaxwellState.")
    maxwell = pair.cochain.maxwell
    bridge = pair.cochain.bridge
    constitutive = maxwell.constitutive
    material = constitutive.initialize_state()
    electric, magnetic = field.electric, field.magnetic
    circulation = bridge.pack_edge_circulation(
        (electric[..., 0], electric[..., 1], electric[..., 2])
    )
    flux = bridge.pack_face_flux((magnetic[..., 0], magnetic[..., 1], magnetic[..., 2]))
    declared = jnp.zeros((maxwell.magnetic_charge_count,), dtype=flux.dtype)
    return CompatibleMaxwellState(
        MaxwellPrimaryState(
            constitutive.electric_displacement(circulation, material),
            flux,
            bridge.pack(0, (field.charge,)),
        ),
        MaxwellAuxiliaryState(material, None, declared, declared),
        (),
    )


def _convert_field(pair: _HandoffPair, field: Any, /) -> Any:
    match pair.route:
        case "cochain-to-spectral":
            return _to_spectral(pair, field)
        case "spectral-to-cochain":
            return _to_cochain(pair, field)
        case _:
            assert_never(pair.route)


def _convert_charge(pair: _HandoffPair, charge: Array, /) -> Array:
    bridge = pair.cochain.bridge
    match pair.route:
        case "cochain-to-spectral":
            return bridge.unpack(0, charge)[0]
        case "spectral-to-cochain":
            return bridge.pack(0, (charge,))
        case _:
            assert_never(pair.route)


def _maximum(residual: Array, /) -> Array:
    return jnp.max(jnp.abs(residual), initial=0.0)


def _relative(residual: Array, scale: Array, /) -> Array:
    maximum = _maximum(residual)
    return maximum / jnp.maximum(scale, jnp.finfo(maximum.dtype).tiny)


def _scales(pair: _HandoffPair, field: SpectralMaxwellState, /) -> tuple[Array, Array]:
    """Gauss operand scales ``max|ρ| + εΣ2max|E_a|/Δ_a`` and ``Σ2max|B_a|/Δ_a``."""
    inverse = 2.0 / jnp.asarray(pair.spectral.plan.spacing)
    electric = jnp.sum(inverse * jnp.max(jnp.abs(field.electric), axis=(0, 1, 2)))
    magnetic = jnp.sum(inverse * jnp.max(jnp.abs(field.magnetic), axis=(0, 1, 2)))
    charge = jnp.max(jnp.abs(field.charge))
    return charge + pair.spectral.plan.permittivity * electric, magnetic


def _spectral_evidence(
    pair: _HandoffPair, field: SpectralMaxwellState, /
) -> _SolverEvidence:
    """PSATD's own ``∇⁻·E`` and ``∇⁺·B`` through its prepared transform."""
    solver = pair.spectral
    operators = solver.spectral.operators
    coefficients = solver.transform.forward(
        jnp.concatenate((field.electric, field.magnetic), axis=-1)
    )
    divergence = solver.transform.inverse(
        jnp.stack(
            (
                operators.electric_divergence(coefficients[..., 0:3]),
                operators.magnetic_divergence(coefficients[..., 3:6]),
            ),
            axis=-1,
        )
    )
    # Staggered grids resolve every nonzero mode; only the mean is invisible.
    gauss = solver.plan.permittivity * divergence[..., 0] - (
        field.charge - jnp.mean(field.charge)
    )
    electric_scale, magnetic_scale = _scales(pair, field)
    volume = float(np.prod(solver.plan.spacing))
    return _SolverEvidence(
        _relative(gauss, electric_scale),
        _relative(divergence[..., 1], magnetic_scale),
        solver.field_energy(field),
        jnp.sum(field.charge) * volume,
        _maximum(divergence[..., 0] - field.charge / solver.plan.permittivity),
        _maximum(divergence[..., 1]),
    )


def _cochain_evidence(
    pair: _HandoffPair, field: CompatibleMaxwellState, scales: tuple[Array, Array], /
) -> _SolverEvidence:
    maxwell = pair.cochain.maxwell
    volume = float(np.prod(pair.spectral.plan.spacing))
    gauss = maxwell.electric_constraint(field)
    magnetic = maxwell.magnetic_constraint(field)
    # On the periodic uniform grid every vertex has the same dual volume and
    # δD sums to zero, so subtracting the mean removes exactly ρ̄.
    return _SolverEvidence(
        _relative(gauss - jnp.mean(gauss), scales[0]),
        _relative(magnetic / volume, scales[1]),
        maxwell.energy(field),
        jnp.sum(field.primary.charge) * volume,
        _maximum(gauss),
        _maximum(magnetic),
    )


def _evidence(
    pair: _HandoffPair,
    cochain: CompatibleMaxwellState,
    spectral: SpectralMaxwellState,
    step_size: Array,
    tolerance: float,
    /,
) -> PICFieldHandoffEvidence:
    spectral_values = _spectral_evidence(pair, spectral)
    cochain_values = _cochain_evidence(pair, cochain, _scales(pair, spectral))
    match pair.route:
        case "cochain-to-spectral":
            source, target = cochain_values, spectral_values
            ids = (pair.cochain.solver_id, pair.spectral.solver_id)
        case "spectral-to-cochain":
            source, target = spectral_values, cochain_values
            ids = (pair.spectral.solver_id, pair.cochain.solver_id)
        case _:
            assert_never(pair.route)
    maxwell = pair.cochain.maxwell
    correction = maxwell.energy(cochain) - maxwell.leapfrog_energy(cochain, step_size)
    tiny = jnp.finfo(source.energy.dtype).tiny
    energy_difference = jnp.abs(target.energy - source.energy) / jnp.maximum(
        source.energy, tiny
    )
    unsigned = jnp.sum(jnp.abs(spectral.charge)) * float(
        np.prod(pair.spectral.plan.spacing)
    )
    finite = (
        jnp.all(jnp.isfinite(spectral.electric))
        & jnp.all(jnp.isfinite(spectral.magnetic))
        & jnp.all(jnp.isfinite(spectral.charge))
    )
    constraint_satisfied = (target.absolute_electric <= tolerance) & (
        target.absolute_magnetic <= tolerance
    )
    successful = (
        finite
        & jnp.isfinite(step_size)
        & (step_size > 0.0)
        & constraint_satisfied
        & (target.electric <= source.electric + _ROUNDOFF)
        & (target.magnetic <= source.magnetic + _ROUNDOFF)
        & (energy_difference <= _ROUNDOFF)
        & (jnp.abs(target.charge - source.charge) <= _ROUNDOFF * unsigned)
    )
    return PICFieldHandoffEvidence(
        route=pair.route,
        source_solver_id=ids[0],
        target_solver_id=ids[1],
        source_electric_constraint=source.electric,
        target_electric_constraint=target.electric,
        source_magnetic_constraint=source.magnetic,
        target_magnetic_constraint=target.magnetic,
        source_field_energy=source.energy,
        target_field_energy=target.energy,
        energy_relative_difference=energy_difference,
        leapfrog_energy_correction=correction,
        source_total_charge=source.charge,
        target_total_charge=target.charge,
        constraint_satisfied=constraint_satisfied,
        successful=successful,
    )


def hand_off_pic_state(
    source: ElectromagneticPICPlan,
    target: ElectromagneticPICPlan,
    state: ElectromagneticPICState,
    step_size: ArrayLike,
    /,
) -> PICFieldHandoffResult:
    """Convert a PIC state of ``source`` into a state of ``target`` mid-run.

    Admitted only between `CochainMaxwellPICFieldSolver` and
    `PreparedSpectralMaxwell` (either direction) on one periodic uniform grid
    with the same transfers, a standard ``grid="staggered"``,
    ``stencil="finite-order"``, ``stencil_order=2`` PSATD without PML,
    antennas, or observers, a cochain Maxwell whose only source is the PIC
    current in the PSATD plan's homogeneous stateless medium, and PIC plans
    declaring the same species, processes, boundaries, recorders, filters,
    guards, external fields, ownership, precision, pusher, and key. Everything
    else is refused before any array is touched: those configurations would
    need a Gauss projection or would drop solver memory.

    The field, the previous field of ``field_history``, and ``wall_charge``
    are converted; species, boundary, recorder, process states, and the clock
    pass through. The plan checks read host metadata; the array conversion
    and evidence are traceable. ``step_size`` is the run's step, which sets
    the reported leapfrog energy correction.
    """
    pair = _admissible_pair(source, target)
    if not isinstance(state, ElectromagneticPICState):
        raise TypeError("state must be ElectromagneticPICState.")
    dt = jnp.asarray(step_size, dtype=jnp.float64)
    if dt.shape != ():
        raise ValueError("step_size must be a scalar.")
    field = _convert_field(pair, state.field)
    history = state.field_history
    converted = ElectromagneticPICState(
        state.species,
        field,
        state.boundaries,
        _convert_charge(pair, state.wall_charge),
        state.recorders,
        state.time,
        state.accepted_step,
        state.status,
        None
        if history is None
        else PICFieldHistory(_convert_field(pair, history.field), history.time),
        state.processes,
    )
    tolerance = target.constraint_tolerance
    match pair.route:
        case "cochain-to-spectral":
            evidence = _evidence(pair, state.field, field, dt, tolerance)
        case "spectral-to-cochain":
            evidence = _evidence(pair, field, state.field, dt, tolerance)
        case _:
            assert_never(pair.route)
    return PICFieldHandoffResult(converted, evidence)


__all__ = [
    "PICFieldHandoffEvidence",
    "PICFieldHandoffResult",
    "PICFieldHandoffRoute",
    "hand_off_pic_state",
]
