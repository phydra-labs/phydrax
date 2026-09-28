#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prescribed moving charges in compatible Maxwell media: runtime contracts.

References are independent of the runtime: the deposited endpoint charge of
the transfer (Gauss's law against the prescribed charge), exact zero fields of
a charge at rest (coincident-neutral start), and step-end energy against the
work ``−∫ E·J`` and the losses reported by the Maxwell runtime.
"""

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


D = phx.discretization
PIC = phx.discretization.pic
mx = phx.solver.maxwell

_COUNT = 10
_LENGTH = 1.0


def _bridge(
    count: int = _COUNT, periodic: tuple[bool, bool, bool] = (False, False, False)
) -> Any:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(count, periodic=value) for value in periodic),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0] * 3, [_LENGTH] * 3]))
    return D.StructuredCochainBridge(grid)


def _current(bridge: Any, capacity: int) -> Any:
    particles = D.ParticleSetPlan(
        jnp.arange(capacity), jnp.ones((capacity,)), ambient_dimension=3
    ).prepare()
    charged = D.ChargedParticlePlan(jnp.ones((capacity,)), "prescribed").prepare(
        particles
    )
    return PIC.ChargeConservingCurrentPlan(
        PIC.PICParticleCochainTransferPlan(bridge).prepare(charged)
    )


def _stable_step(bridge: Any, **options: Any) -> float:
    return float(phx.solver.CompatibleMaxwellPlan(bridge, **options).prepare().stable_dt)


def _plan(
    bridge: Any,
    path: Any,
    charge: Sequence[float],
    samples: int,
    /,
    *,
    step_count: int | None = None,
    observers: Sequence[Any] = (),
    fraction: float = 0.9,
    **options: Any,
) -> Any:
    """Sample ``path(t) -> positions[P, 3]`` on the Maxwell step grid."""
    step = fraction * _stable_step(bridge, **options)
    times = step * np.arange(samples)
    positions = np.stack([path(time) for time in times])
    current = _current(bridge, positions.shape[1])
    trajectory = mx.PrescribedChargeTrajectory(times, positions)
    source = mx.PrescribedChargeCurrentSourcePlan(
        trajectory, current, step_count=step_count
    )
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge, sources=(source,), observers=tuple(observers), **options
    ).prepare()
    return mx.PrescribedChargeMaxwellPlan(
        maxwell, current, trajectory, np.asarray(charge), step_count=step_count
    )


def _orbit(time: float) -> np.ndarray:
    phase = 2.0 * np.pi * time / 1.2
    return np.asarray(
        [[0.5 + 0.12 * np.cos(phase), 0.5 + 0.12 * np.sin(phase), 0.5 + 0.03 * time]]
    )


def _deposited(plan: Any, positions: np.ndarray) -> np.ndarray:
    transfer = plan.current_plan.transfer
    return np.asarray(
        transfer.deposit_macrocharge(transfer.build(positions), plan.charge).cochain
    )


def test_charge_at_rest_keeps_identically_zero_fields() -> None:
    bridge = _bridge()
    plan = _plan(bridge, lambda time: np.asarray([[0.43, 0.51, 0.47]]), [1.0], 30)
    result = mx.solve_prescribed_charge_maxwell(plan)
    primary = result.final_state.primary
    # The compensating charge coincides with the created charge: no Coulomb field.
    assert np.all(np.asarray(primary.electric_displacement) == 0.0)
    assert np.all(np.asarray(primary.magnetic_flux) == 0.0)
    assert np.all(np.asarray(primary.charge) == 0.0)
    assert np.all(np.asarray(result.prescribed_charge) == 0.0)
    assert np.all(np.asarray(result.field_energy) == 0.0)
    assert bool(result.evidence.successful)


def test_maxwell_charge_follows_prescribed_charge_and_freezes() -> None:
    bridge = _bridge()
    samples = 40
    moving = _plan(bridge, _orbit, [1.5], samples)
    frozen = _plan(bridge, _orbit, [1.5], samples, step_count=samples + 24)
    first = mx.solve_prescribed_charge_maxwell(moving)
    second = mx.solve_prescribed_charge_maxwell(frozen)
    last = np.asarray(moving.trajectory.positions[-1])
    expected = _deposited(moving, last) - _deposited(moving, _orbit(0.0))
    scale = np.max(np.abs(expected))

    np.testing.assert_allclose(np.asarray(first.final_positions), last, atol=1e-15)
    np.testing.assert_allclose(
        np.asarray(first.final_state.primary.charge), expected, atol=1e-11 * scale
    )
    # Frozen steps carry no current: the charge is preserved and does no work.
    np.testing.assert_allclose(
        np.asarray(second.final_state.primary.charge), expected, atol=1e-11 * scale
    )
    np.testing.assert_array_equal(np.asarray(second.source_work[samples - 1 :]), 0.0)
    for result in (first, second):
        evidence = result.evidence
        assert bool(evidence.successful)
        assert float(evidence.maximum_continuity_defect) < 1e-12
        assert float(evidence.maximum_constraint_defect) < 1e-12
        assert float(evidence.maximum_gauss_defect) < 1e-10 * scale
        assert float(evidence.maximum_support_leak) == 0.0
        assert evidence.free_vertex_count == (_COUNT + 1) ** 3


def _lossy() -> dict[str, Any]:
    return {
        "constitutive": mx.ConductiveMaxwellConstitutivePlan(
            permittivity=2.0, electric_conductivity=1.5
        )
    }


def _lorentz() -> dict[str, Any]:
    poles = mx.MaxwellLorentzPoles(
        jnp.asarray([8.0]), jnp.asarray([0.5]), jnp.asarray([40.0])
    )
    return {"constitutive": mx.LorentzDrudeMaxwellConstitutivePlan(poles)}


def _negative_index() -> dict[str, Any]:
    electric = mx.MaxwellLorentzPoles(
        jnp.asarray([6.0]), jnp.asarray([0.2]), jnp.asarray([60.0])
    )
    magnetic = mx.MaxwellLorentzPoles(
        jnp.asarray([6.0]), jnp.asarray([0.2]), jnp.asarray([60.0])
    )
    return {
        "constitutive": mx.LorentzDrudeMaxwellConstitutivePlan(
            electric, magnetic_poles=magnetic
        )
    }


def _plasma() -> dict[str, Any]:
    return {
        "constitutive": mx.MagnetizedColdPlasmaMaxwellConstitutivePlan(
            jnp.asarray([7.0]),
            jnp.asarray([[0.0, 0.0, 5.0]]),
            collision_frequency=0.3,
        )
    }


def _heterogeneous() -> dict[str, Any]:
    layout = mx.MaxwellCochainLayout(_bridge())
    edges = np.arange(layout.electric_count)
    permittivity = np.where(edges % 3 == 0, 2.5, 1.0)
    return {"constitutive": mx.DiagonalMaxwellConstitutivePlan(permittivity=permittivity)}


_LOSSLESS = {
    "vacuum": dict,
    "heterogeneous": _heterogeneous,
    "pec": lambda: {"boundaries": (mx.MaxwellBoundaryPlan("pec"),)},
}
_LOSSY = {
    "lossy": _lossy,
    "impedance": lambda: {
        "boundaries": (mx.MaxwellBoundaryPlan("impedance", admittance=1.0),)
    },
    "lorentz-drude": _lorentz,
    "negative-index": _negative_index,
    "magnetized-plasma": _plasma,
}
_DURATION = 2.9


def _ledger_run(medium: dict[str, Any], fraction: float) -> Any:
    bridge = _bridge()
    step = fraction * _stable_step(bridge, **medium)
    samples = int(_DURATION / step) + 1
    return mx.solve_prescribed_charge_maxwell(
        _plan(bridge, _orbit, [1.0], samples, fraction=fraction, **medium)
    )


def _constraints_hold(evidence: Any) -> None:
    failures = int(evidence.status) & ~int(mx.PrescribedChargeStatus.LEDGER_OPEN)
    assert failures == 0, failures
    assert float(evidence.maximum_continuity_defect) < 1e-12
    assert float(evidence.maximum_constraint_defect) < 1e-10
    assert float(evidence.maximum_magnetic_defect) < 1e-10


@pytest.mark.parametrize("medium", list(_LOSSLESS), ids=list(_LOSSLESS))
def test_lossless_power_ledger_closes_to_roundoff(medium: str) -> None:
    result = _ledger_run(_LOSSLESS[medium](), 0.9)
    _constraints_hold(result.evidence)
    assert bool(result.evidence.successful)
    assert np.sum(np.abs(np.asarray(result.source_work))) > 0.0
    # The leapfrog energy exchanges exactly with −Δt⟨Ē, ⋆J⟩ (PEC included).
    assert float(result.evidence.relative_ledger_defect) < 1e-12


@pytest.mark.parametrize("medium", list(_LOSSY), ids=list(_LOSSY))
def test_dissipative_power_ledger_closes_under_step_refinement(medium: str) -> None:
    coarse, fine = (_ledger_run(_LOSSY[medium](), value) for value in (0.2, 0.1))
    for result in (coarse, fine):
        _constraints_hold(result.evidence)
        work = np.sum(np.abs(np.asarray(result.source_work)))
        assert np.sum(np.asarray(result.dissipated_energy)) > 1e-3 * work
    defects = [float(value.evidence.relative_ledger_defect) for value in (coarse, fine)]
    # Midpoint conduction, boundary-consistent media and trapezoidal losses form a
    # second-order pair: halving Δt divides the ledger defect by ≈4.
    assert 3.6 < defects[0] / defects[1] < 4.4, defects
    assert defects[1] < 2e-3, defects


def test_boundary_exit_freezes_the_charge_on_the_crossed_face() -> None:
    bridge = _bridge()
    speed = 1.8

    def path(time: float) -> np.ndarray:
        return np.asarray([[0.8 + speed * time, 0.47, 0.52]])

    plan = _plan(bridge, path, [1.0], 40)
    result = mx.solve_prescribed_charge_maxwell(plan)
    evidence = result.evidence
    times = np.asarray(plan.trajectory.times)
    crossing = int(np.argmax(0.8 + speed * times > 1.0)) - 1

    assert bool(evidence.successful)
    assert int(evidence.status) & int(mx.PrescribedChargeStatus.BOUNDARY_EXIT)
    assert bool(evidence.exited[0]) and int(evidence.exit_step[0]) == crossing
    np.testing.assert_allclose(
        np.asarray(result.final_positions), [[1.0, 0.47, 0.52]], atol=1e-15
    )
    expected = _deposited(plan, np.asarray([[1.0, 0.47, 0.52]])) - _deposited(
        plan, path(0.0)
    )
    np.testing.assert_allclose(
        np.asarray(result.final_state.primary.charge),
        expected,
        atol=1e-11 * np.max(np.abs(expected)),
    )


def test_refusals() -> None:
    bridge = _bridge()
    step = 0.9 * _stable_step(bridge)
    times = step * np.arange(8)
    positions = np.stack([_orbit(float(time)) for time in times])
    trajectory = mx.PrescribedChargeTrajectory(times, positions)
    current = _current(bridge, 1)
    source = mx.PrescribedChargeCurrentSourcePlan(trajectory, current)
    maxwell = phx.solver.CompatibleMaxwellPlan(bridge, sources=(source,)).prepare()
    with pytest.raises(ValueError, match="uniform step grid"):
        mx.PrescribedChargeTrajectory(times * (1.0 + 0.1 * np.arange(8)), positions)
    with pytest.raises(ValueError, match="particle capacity"):
        mx.PrescribedChargeCurrentSourcePlan(trajectory, _current(bridge, 2))
    with pytest.raises(ValueError, match="start inside"):
        mx.PrescribedChargeCurrentSourcePlan(
            mx.PrescribedChargeTrajectory(times, positions + 2.0), current
        )
    with pytest.raises(ValueError, match="exactly one"):
        mx.PrescribedChargeMaxwellPlan(
            phx.solver.CompatibleMaxwellPlan(bridge).prepare(),
            current,
            trajectory,
            np.ones(1),
        )
    with pytest.raises(ValueError, match="another trajectory"):
        mx.PrescribedChargeMaxwellPlan(
            maxwell, current, trajectory, np.ones(1), step_count=9
        )
    slow = mx.PrescribedChargeTrajectory(4.0 * times, positions)
    slow_source = mx.PrescribedChargeCurrentSourcePlan(slow, current)
    with pytest.raises(ValueError, match="stable step"):
        mx.PrescribedChargeMaxwellPlan(
            phx.solver.CompatibleMaxwellPlan(bridge, sources=(slow_source,)).prepare(),
            current,
            slow,
            np.ones(1),
        )
    with pytest.raises(ValueError, match="one finite value"):
        mx.PrescribedChargeMaxwellPlan(maxwell, current, trajectory, np.ones(2))


def test_huygens_box_certifies_the_prescribed_path_support() -> None:
    bridge = _bridge(12)
    acquisition = mx.MaxwellSpectralAcquisition(
        jnp.asarray([20.0]), sign="positive", measure="time-integral"
    )
    exterior = mx.HomogeneousMaxwellExterior()
    enclosing = mx.MaxwellHuygensBoxPlan(
        bridge, (2, 2, 2), (10, 10, 10), acquisition, exterior
    )
    crossed = mx.MaxwellHuygensBoxPlan(
        bridge, (5, 5, 5), (7, 7, 7), acquisition, exterior
    )
    plan = _plan(bridge, _orbit, [1.0], 12, observers=(enclosing,))
    assert bool(mx.solve_prescribed_charge_maxwell(plan).evidence.successful)
    with pytest.raises(ValueError, match="prescribed charge path"):
        _plan(bridge, _orbit, [1.0], 12, observers=(crossed,))


def test_cpml_absorber_charge_stays_inside_the_layer() -> None:
    bridge = _bridge()
    width = 3
    plan = _plan(bridge, _orbit, [1.0], 60, pml=mx.MaxwellCPMLPlan(width))
    result = mx.solve_prescribed_charge_maxwell(plan)
    _constraints_hold(result.evidence)
    auxiliary = result.final_state.auxiliary
    # Only the prescribed electric current drives this run: no source magnetic charge.
    assert np.all(np.asarray(auxiliary.magnetic_charge) == 0.0)
    absorber = np.asarray(auxiliary.absorber_magnetic_charge).reshape((_COUNT,) * 3)
    # Cells adjacent to a layer face belong to the absorber support.
    margin = width + 1
    interior = absorber[margin:-margin, margin:-margin, margin:-margin]
    assert np.max(np.abs(absorber)) > 0.0
    assert np.all(interior == 0.0)


def test_cpml_power_ledger_converges_at_second_order() -> None:
    # The CPML absorbs about half of the source work here (the orbit's near field
    # reaches the layer). Every kick reads its convolution memory at its own time
    # level, so the trapezoidal CPML loss closes the ledger at O(Δt²) at fixed h.
    runs = [_ledger_run({"pml": mx.MaxwellCPMLPlan(3)}, value) for value in (0.45, 0.225)]
    for result in runs:
        _constraints_hold(result.evidence)
        work = np.sum(np.abs(np.asarray(result.source_work)))
        assert np.sum(np.asarray(result.dissipated_energy)) > 0.3 * work
    coarse, fine = (float(value.evidence.relative_ledger_defect) for value in runs)
    assert fine < 5e-3
    assert 1.8 < np.log2(coarse / fine) < 2.2
