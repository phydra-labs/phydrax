#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mid-run PIC state handoff between the Yee cochain solver and staggered PSATD.

A cold electron plasma (``ω_p² = 16``, ``c = ε = μ = 1``) on a periodic
``8 × 4 × 4`` grid carries a Langmuir displacement along ``x`` and a standing
transverse wave seeded by ``B_z = b cos(kx)``. It runs on the cochain solver,
is handed off to order-2 staggered PSATD, and continues; references are the
geometric Yee layout evaluated with NumPy and an uninterrupted cochain run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


D = phx.discretization
PIC = D.pic
S = phx.solver
sp = S.maxwell.spectral

_COUNTS = (8, 4, 4)
_H = 0.125
_K = 2.0 * np.pi / (_COUNTS[0] * _H)
_PLASMA_FREQUENCY_SQUARED = 16.0
_STEP = 0.04
_STEPS = 20
_MAGNETIC = 0.3
_DISPLACEMENT = 0.02
_PER_CELL = 2


def _bridge(periodic_x: bool = True, cells_x: int = _COUNTS[0]) -> Any:
    grid = D.TensorGridPlan(
        (
            D.UniformCellAxisSpec(cells_x, periodic=periodic_x),
            D.UniformCellAxisSpec(_COUNTS[1], periodic=True),
            D.UniformCellAxisSpec(_COUNTS[2], periodic=True),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [cells_x * _H, 4 * _H, 4 * _H]]))
    return D.StructuredCochainBridge(grid)


def _lattice() -> np.ndarray:
    ix, iy, iz, ip = np.meshgrid(
        *(np.arange(n) for n in _COUNTS), np.arange(_PER_CELL), indexing="ij"
    )
    return np.stack(
        ((ix + (ip + 0.5) / _PER_CELL) * _H, (iy + 0.5) * _H, (iz + 0.5) * _H), -1
    ).reshape(-1, 3)


_LATTICE = _lattice()
_COUNT = _LATTICE.shape[0]
_WEIGHT = _PLASMA_FREQUENCY_SQUARED * float(np.prod(_COUNTS)) * _H**3 / _COUNT


def _species(maximum_charge_number: int = 1) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    species, charged = [], []
    for offset, specific, name, mass in (
        (0, -1.0, "electrons", _WEIGHT),
        (10**6, 1.0 / 1836.0, "ions", 1836.0 * _WEIGHT),
    ):
        support = D.ParticleSetPlan(
            jnp.arange(offset, offset + _COUNT),
            mass * jnp.ones((_COUNT,)),
            ambient_dimension=3,
        ).prepare()
        charged.append(
            D.ChargedParticlePlan(specific * mass * jnp.ones((_COUNT,)), name).prepare(
                support
            )
        )
        species.append(
            PIC.PICSpeciesPlan(
                D.ParticlePopulationPlan(support),
                PIC.PICChargeModelPlan(
                    specific,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=maximum_charge_number,
                    initial_charge_number=1,
                ),
            )
        )
    return tuple(species), tuple(charged)


def _transfers(
    bridge: Any, charged: tuple[Any, ...], order: PIC.PICShapeOrder = 2
) -> tuple[Any, Any]:
    plan = PIC.PICParticleCochainTransferPlan(bridge, shape_order=order)
    transfers = tuple(plan.prepare(value) for value in charged)
    return transfers, tuple(PIC.ChargeConservingCurrentPlan(value) for value in transfers)


def _cochain(
    bridge: Any, transfers: tuple[Any, ...], currents: tuple[Any, ...], **options: Any
) -> Any:
    maxwell = S.CompatibleMaxwellPlan(
        bridge, sources=(S.PICMaxwellCurrentSourcePlan(),), **options
    ).prepare()
    boundary = (
        S.CochainElectrostaticBoundaryPlan.periodic(bridge)
        if all(axis.periodic for axis in bridge.grid.structured_axes)
        else S.CochainElectrostaticBoundaryPlan.dirichlet(bridge)
    )
    constitutive = maxwell.constitutive
    permittivity = constitutive.electric_displacement(
        jnp.ones((maxwell.layout.electric_count,)), constitutive.initialize_state()
    )
    electrostatic = S.CochainElectrostaticPlan(
        bridge, boundary, permittivity=permittivity
    )
    return S.CochainMaxwellPICFieldSolver(maxwell, electrostatic, transfers, currents)


def _spectral(
    bridge: Any, transfers: tuple[Any, ...], currents: tuple[Any, ...], **options: Any
) -> Any:
    declared: dict[str, Any] = {
        "grid": "staggered",
        "stencil": "finite-order",
        "stencil_order": 2,
    }
    return sp.SpectralMaxwellPlan(bridge, **(declared | options)).prepare(
        transfers, currents
    )


@eqx.filter_jit
def _run(plan: Any, state: Any, steps: int) -> tuple[Any, Any]:
    def body(current: Any, _: None) -> tuple[Any, Any]:
        result = plan.step_detailed(current, _STEP)
        return result.accepted_state, result.diagnostics

    return jax.lax.scan(body, state, None, length=steps)


class _Case(NamedTuple):
    bridge: Any
    species: tuple[Any, ...]
    charged: tuple[Any, ...]
    transfers: tuple[Any, ...]
    currents: tuple[Any, ...]
    source: Any
    target: Any
    initial: Any
    middle: Any
    before: Any
    handoff: Any


@pytest.fixture(scope="module")
def case() -> _Case:
    bridge = _bridge()
    species, charged = _species()
    transfers, currents = _transfers(bridge, charged)
    source = S.ElectromagneticPICPlan(
        _cochain(bridge, transfers, currents), species=species
    )
    target = S.ElectromagneticPICPlan(
        _spectral(bridge, transfers, currents), species=species
    )
    electrons = _LATTICE.copy()
    electrons[:, 0] += _DISPLACEMENT * np.sin(_K * _LATTICE[:, 0])
    # B_z sits on the z-faces, whose centers are at x = (i + ½)Δ.
    centers = (np.arange(_COUNTS[0]) + 0.5) * _H
    magnetic_z = np.broadcast_to(
        (_MAGNETIC * np.cos(_K * centers))[:, None, None], _COUNTS
    )
    zero = np.zeros(_COUNTS)
    velocity = np.zeros((_COUNT, 3))
    initial = source.initialize(
        (electrons, _LATTICE),
        (velocity, velocity),
        _STEP,
        magnetic=bridge.pack_face_flux((zero, zero, magnetic_z)),
    )
    middle, before = _run(source, initial, _STEPS)
    handoff = S.hand_off_pic_state(source, target, middle, _STEP)
    return _Case(
        bridge,
        species,
        charged,
        transfers,
        currents,
        source,
        target,
        initial,
        middle,
        before,
        handoff,
    )


class _Continuation(NamedTuple):
    spectral: Any
    after: Any
    cochain: Any


@pytest.fixture(scope="module")
def continuation(case: _Case) -> _Continuation:
    spectral, after = _run(case.target, case.handoff.state, _STEPS)
    cochain, _ = _run(case.source, case.middle, _STEPS)
    return _Continuation(spectral, after, cochain)


# -- independent Yee geometry --------------------------------------------------


def _yee_fields(bridge: Any, field: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical ``E``/``B``/``ρ`` from the cochains by edge length and face area.

    Oriented blocks follow the bridge's declared orientations; an ``(a, b)``
    face carries ``ε_abc B_c Δ_a Δ_b`` for its normal axis ``c``.
    """
    shape = _COUNTS
    size = int(np.prod(shape))
    displacement = np.asarray(field.primary.electric_displacement)
    flux = np.asarray(field.primary.magnetic_flux)
    electric = np.zeros((*shape, 3))
    for index, (axis,) in enumerate(bridge.orientations[1]):
        electric[..., axis] = (
            displacement[index * size : (index + 1) * size].reshape(shape) / _H
        )
    magnetic = np.zeros((*shape, 3))
    for index, (first, second) in enumerate(bridge.orientations[2]):
        (normal,) = {0, 1, 2} - {first, second}
        sign = np.linalg.det(np.eye(3)[[first, second, normal]])
        magnetic[..., normal] = (
            sign * flux[index * size : (index + 1) * size].reshape(shape) / (_H * _H)
        )
    return electric, magnetic, np.asarray(field.primary.charge).reshape(shape)


def _backward_divergence(values: np.ndarray) -> np.ndarray:
    return sum(
        (values[..., axis] - np.roll(values[..., axis], 1, axis=axis)) / _H
        for axis in range(3)
    )


def _forward_divergence(values: np.ndarray) -> np.ndarray:
    """``∇⁺·B`` at cell centers from the face-centered components."""
    return sum(
        (np.roll(values[..., axis], -1, axis=axis) - values[..., axis]) / _H
        for axis in range(3)
    )


def _forward_curl(values: np.ndarray) -> np.ndarray:
    def forward(component: int, axis: int) -> np.ndarray:
        return (
            np.roll(values[..., component], -1, axis=axis) - values[..., component]
        ) / _H

    return np.stack(
        (
            forward(2, 1) - forward(1, 2),
            forward(0, 2) - forward(2, 0),
            forward(1, 0) - forward(0, 1),
        ),
        axis=-1,
    )


def test_handoff_places_yee_components_at_staggered_psatd_positions(
    case: _Case,
) -> None:
    electric, magnetic, charge = _yee_fields(case.bridge, case.middle.field)
    field = case.handoff.state.field
    evidence = case.handoff.evidence
    np.testing.assert_allclose(np.asarray(field.electric), electric, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(np.asarray(field.magnetic), magnetic, rtol=1e-15, atol=0.0)
    np.testing.assert_array_equal(np.asarray(field.charge), charge)
    scale = np.max(np.abs(charge))
    # Yee's backward node difference is the Gauss law both solvers share.
    gauss = np.max(np.abs(_backward_divergence(electric) - charge))
    assert gauss <= 1e-12 * scale
    assert bool(evidence.successful)
    assert bool(evidence.constraint_satisfied)
    assert evidence.route == "cochain-to-spectral"
    assert evidence.source_solver_id == case.source.solver.solver_id
    assert evidence.target_solver_id == case.target.solver.solver_id
    assert float(evidence.target_electric_constraint) <= 1e-13
    assert float(evidence.target_magnetic_constraint) <= 1e-14
    volume = _H**3
    energy = 0.5 * volume * (np.sum(electric**2) + np.sum(magnetic**2))
    np.testing.assert_allclose(float(evidence.source_field_energy), energy, rtol=1e-13)
    np.testing.assert_allclose(float(evidence.target_field_energy), energy, rtol=1e-13)
    # Yee's modified energy subtracts (Δt²/8)∫|∇×E|²/μ.
    correction = _STEP**2 / 8.0 * volume * np.sum(_forward_curl(electric) ** 2)
    np.testing.assert_allclose(
        float(evidence.leapfrog_energy_correction), correction, rtol=1e-12
    )
    assert float(evidence.leapfrog_energy_correction) > 1e-4 * energy


def test_prescribed_magnetic_field_lands_on_its_face_centers(case: _Case) -> None:
    handoff = S.hand_off_pic_state(case.source, case.target, case.initial, _STEP)
    magnetic = np.asarray(handoff.state.field.magnetic)
    centers = (np.arange(_COUNTS[0]) + 0.5) * _H
    expected = np.broadcast_to((_MAGNETIC * np.cos(_K * centers))[:, None, None], _COUNTS)
    np.testing.assert_allclose(magnetic[..., 2], expected, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(magnetic[..., :2], 0.0)
    assert bool(handoff.evidence.successful)


def test_particles_gather_the_same_fields_after_handoff(case: _Case) -> None:
    cochain, spectral = case.source.solver, case.target.solver
    for index, species in enumerate(case.middle.species):
        position = species.particles.position
        active = species.population.active
        before = cochain.gather_fields(index, position, active, case.middle.field)
        after = spectral.gather_fields(index, position, active, case.handoff.state.field)
        for old, new in zip(before, after, strict=True):
            np.testing.assert_allclose(np.asarray(new), np.asarray(old), atol=1e-14)
    source = case.source.synchronized_energy(case.middle, _STEP)
    target = case.target.synchronized_energy(case.handoff.state, _STEP)
    np.testing.assert_allclose(
        float(target.particle_kinetic), float(source.particle_kinetic), rtol=1e-13
    )
    np.testing.assert_allclose(
        float(target.total - source.total),
        float(case.handoff.evidence.leapfrog_energy_correction),
        rtol=1e-9,
    )


def test_psatd_continuation_keeps_gauss_and_a_continuous_ledger(
    case: _Case, continuation: _Continuation
) -> None:
    after = continuation.after
    assert bool(jnp.all(case.before.successful))
    assert bool(jnp.all(after.successful))
    scale = float(jnp.max(jnp.abs(case.handoff.state.field.charge)))
    assert float(jnp.max(after.electric_constraint)) <= 1e-12 * scale
    assert float(jnp.max(after.magnetic_constraint)) <= 1e-12
    # Field energy switches from Yee's modified to PSATD's plain energy: the
    # ledger total jumps by exactly the reported half-kick term.
    jump = float(after.energy.previous_total[0] - case.before.energy.total[-1])
    np.testing.assert_allclose(
        jump, float(case.handoff.evidence.leapfrog_energy_correction), rtol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(after.energy.previous_total[1:]),
        np.asarray(after.energy.total[:-1]),
        rtol=1e-14,
    )


def _modes(state: Any) -> tuple[complex, complex]:
    """Electron ``x``-displacement and ``u_y`` Fourier modes along ``x``."""
    electrons = state.species[0].particles
    phase = np.exp(-1j * _K * _LATTICE[:, 0])
    position = np.asarray(electrons.position)[:, 0]
    displacement = (position - _LATTICE[:, 0] + 0.5 * _COUNTS[0] * _H) % (
        _COUNTS[0] * _H
    ) - 0.5 * _COUNTS[0] * _H
    velocity = np.asarray(electrons.proper_velocity)[:, 1]
    return complex(np.mean(displacement * phase)), complex(np.mean(velocity * phase))


def test_plasma_modes_continue_across_the_handoff(
    continuation: _Continuation,
) -> None:
    handed, reference = _modes(continuation.spectral), _modes(continuation.cochain)
    # After the handoff only the transverse time integration differs: Yee's
    # leapfrog advances a mode of frequency Ω with phase error ≈ Ω³Δt²/24 per
    # unit time against PSATD's exact exponential, Ω² = ω_p² + (c[k])².
    symbol = 2.0 * np.sin(0.5 * _K * _H) / _H
    frequency = np.sqrt(_PLASMA_FREQUENCY_SQUARED + symbol**2)
    phase_error = frequency**3 * _STEP**2 / 24.0 * _STEPS * _STEP
    for value, expected in zip(handed, reference, strict=True):
        assert abs(value - expected) <= 2.0 * phase_error * abs(expected)


def test_spectral_state_hands_back_to_the_cochain_solver(
    case: _Case, continuation: _Continuation
) -> None:
    roundtrip = S.hand_off_pic_state(case.target, case.source, case.handoff.state, _STEP)
    for new, old in zip(
        jax.tree.leaves(roundtrip.state.field.primary),
        jax.tree.leaves(case.middle.field.primary),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(new), np.asarray(old), rtol=1e-15, atol=0)
    back = S.hand_off_pic_state(case.target, case.source, continuation.spectral, _STEP)
    evidence = back.evidence
    assert evidence.route == "spectral-to-cochain"
    assert bool(evidence.successful)
    assert float(evidence.target_electric_constraint) <= 1e-13
    np.testing.assert_allclose(
        float(evidence.target_field_energy),
        float(evidence.source_field_energy),
        rtol=1e-14,
    )
    _, diagnostics = _run(case.source, back.state, _STEPS)
    assert bool(jnp.all(diagnostics.successful))


def _kicked_edge(state: Any) -> Any:
    """One edge circulation raised by 0.5 at fixed charge: Gauss's law broken."""
    primary = state.field.primary
    return eqx.tree_at(
        lambda value: value.field.primary.electric_displacement,
        state,
        primary.electric_displacement.at[0].add(0.5),
    )


def _kicked_face(state: Any) -> Any:
    """One face flux raised by 0.5: ``∇·B = 0`` broken."""
    primary = state.field.primary
    return eqx.tree_at(
        lambda value: value.field.primary.magnetic_flux,
        state,
        primary.magnetic_flux.at[0].add(0.5),
    )


@pytest.mark.parametrize(
    ("kick", "reason"),
    [
        (_kicked_edge, PIC.PICRejectionReason.GAUSS),
        (_kicked_face, PIC.PICRejectionReason.MAGNETIC),
    ],
    ids=["gauss", "divergence-b"],
)
def test_constraint_violating_state_is_not_handed_off_successfully(
    case: _Case, kick: Callable[[Any], Any], reason: PIC.PICRejectionReason
) -> None:
    violated = kick(case.middle)
    electric, magnetic, charge = _yee_fields(case.bridge, violated.field)
    tolerance = case.target.constraint_tolerance
    # Independent Yee residuals: the kick breaks one constraint far beyond the
    # plans' absolute tolerance.
    gauss = np.max(np.abs(_backward_divergence(electric) - charge))
    divergence = np.max(np.abs(_forward_divergence(magnetic)))
    assert max(gauss, divergence) > 1e6 * tolerance
    forward = S.hand_off_pic_state(case.source, case.target, violated, _STEP)
    evidence = forward.evidence
    # The exact conversion preserves the violation (the untouched constraint
    # stays at roundoff), so relative preservation alone cannot tell this
    # state from a constraint-consistent one.
    np.testing.assert_allclose(
        float(evidence.target_electric_constraint),
        float(evidence.source_electric_constraint),
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        float(evidence.target_magnetic_constraint),
        float(evidence.source_magnetic_constraint),
        rtol=1e-12,
        atol=1e-13,
    )
    assert not bool(evidence.constraint_satisfied)
    assert not bool(evidence.successful)
    back = S.hand_off_pic_state(case.target, case.source, forward.state, _STEP)
    assert not bool(back.evidence.constraint_satisfied)
    assert not bool(back.evidence.successful)
    # The target plan itself rejects its next step from this state.
    _, diagnostics = _run(case.target, forward.state, 1)
    assert not bool(diagnostics.successful[0])
    assert int(diagnostics.rejection_reason[0]) & int(reason)


def test_non_neutral_periodic_state_fails_gauss_in_both_directions(
    case: _Case,
) -> None:
    spectral = case.handoff.state
    charged = eqx.tree_at(
        lambda value: value.field.charge, spectral, spectral.field.charge + 1.0
    )
    net = float(np.sum(np.asarray(charged.field.charge))) * _H**3
    assert net > 0.2
    to_cochain = S.hand_off_pic_state(case.target, case.source, charged, _STEP)
    to_spectral = S.hand_off_pic_state(case.source, case.target, to_cochain.state, _STEP)
    for evidence in (to_cochain.evidence, to_spectral.evidence):
        # A net charge on a periodic box violates the integral Gauss law,
        # whichever solver holds the state.
        assert not bool(evidence.constraint_satisfied)
        assert not bool(evidence.successful)
        np.testing.assert_allclose(float(evidence.source_total_charge), net, rtol=1e-12)
        np.testing.assert_allclose(float(evidence.target_total_charge), net, rtol=1e-12)
        # Both sides compare the same mean-free local Gauss residual.
        assert float(evidence.source_electric_constraint) <= 1e-13
        assert float(evidence.target_electric_constraint) <= 1e-13


# -- refusals ------------------------------------------------------------------


def _spectral_target(**options: Any) -> Callable[[_Case], tuple[Any, Any]]:
    def build(case: _Case) -> tuple[Any, Any]:
        solver = _spectral(case.bridge, case.transfers, case.currents, **options)
        return case.source, S.ElectromagneticPICPlan(solver, species=case.species)

    return build


def _cochain_source(periodic: bool, **options: Any) -> Callable[[_Case], tuple[Any, Any]]:
    def build(case: _Case) -> tuple[Any, Any]:
        bridge = _bridge(periodic, _COUNTS[0] if periodic else 12)
        transfers, currents = _transfers(bridge, case.charged)
        solver = _cochain(bridge, transfers, currents, **options)
        return S.ElectromagneticPICPlan(solver, species=case.species), case.target

    return build


def _other_grid(case: _Case) -> tuple[Any, Any]:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(n, periodic=True) for n in _COUNTS),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [n * 0.25 for n in _COUNTS]]))
    bridge = D.StructuredCochainBridge(grid)
    transfers, currents = _transfers(bridge, case.charged)
    return case.source, S.ElectromagneticPICPlan(
        _spectral(bridge, transfers, currents), species=case.species
    )


def _other_transfers(case: _Case) -> tuple[Any, Any]:
    transfers, currents = _transfers(case.bridge, case.charged, order=1)
    return case.source, S.ElectromagneticPICPlan(
        _spectral(case.bridge, transfers, currents), species=case.species
    )


def _other_species(case: _Case) -> tuple[Any, Any]:
    species, _ = _species(maximum_charge_number=2)
    return case.source, S.ElectromagneticPICPlan(case.target.solver, species=species)


def _other_key(case: _Case) -> tuple[Any, Any]:
    return case.source, S.ElectromagneticPICPlan(
        case.target.solver, species=case.species, key=jax.random.key(0)
    )


def _same_solver(case: _Case) -> tuple[Any, Any]:
    return case.source, case.source


@pytest.mark.parametrize(
    ("build", "error", "match"),
    [
        (
            _spectral_target(stencil="infinite-order", stencil_order=None),
            ValueError,
            "stencil_order=2",
        ),
        (_spectral_target(stencil_order=4), ValueError, "stencil_order=2"),
        (_spectral_target(grid="collocated"), ValueError, "grid='staggered'"),
        (
            _spectral_target(
                variant="galilean",
                galilean_velocity=(0.1, 0.0, 0.0),
                charge_conservation="update-with-rho",
            ),
            ValueError,
            "standard PSATD variant",
        ),
        (
            _spectral_target(absorber="psatd-pml", pml=sp.SpectralPMLPlan((2, 0, 0))),
            ValueError,
            "PML",
        ),
        (_spectral_target(permittivity=2.0), ValueError, "homogeneous medium"),
        (
            _cochain_source(
                True,
                constitutive=S.maxwell.ConductiveMaxwellConstitutivePlan(
                    electric_conductivity=0.5
                ),
            ),
            ValueError,
            "lossless stateless diagonal medium",
        ),
        (
            _cochain_source(False, pml=S.maxwell.MaxwellCPMLPlan((2, 0, 0))),
            ValueError,
            "CPML memory",
        ),
        (_other_grid, ValueError, "shared grid bridge"),
        (_other_transfers, ValueError, "same particle transfers"),
        (_other_species, ValueError, "species"),
        (_other_key, ValueError, "random_key"),
        (_same_solver, TypeError, "defined between"),
    ],
    ids=[
        "infinite-order-psatd",
        "fourth-order-psatd",
        "collocated-psatd",
        "galilean-psatd",
        "psatd-pml",
        "psatd-medium",
        "conductive-cochain",
        "cpml-cochain",
        "other-grid",
        "other-transfers",
        "other-species",
        "other-key",
        "unsupported-solver-pair",
    ],
)
def test_inadmissible_handoffs_are_refused(
    case: _Case,
    build: Callable[[_Case], tuple[Any, Any]],
    error: type[Exception],
    match: str,
) -> None:
    source, target = build(case)
    with pytest.raises(error, match=match):
        S.hand_off_pic_state(source, target, case.middle, _STEP)


def test_state_of_another_solver_is_refused(case: _Case) -> None:
    with pytest.raises(TypeError, match="CompatibleMaxwellState"):
        S.hand_off_pic_state(case.source, case.target, case.handoff.state, _STEP)
