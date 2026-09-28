#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Contracts of the Cartesian PSATD/Galilean spectral PIC field solver."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


sp = phx.solver.maxwell.spectral
D = phx.discretization


def _bridge(counts: tuple[int, int, int], spacing: float) -> Any:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(n, periodic=True) for n in counts),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [n * spacing for n in counts]]))
    return D.StructuredCochainBridge(grid)


def _species(
    bridge: Any,
    positions: np.ndarray,
    *,
    shape_order: D.pic.PICShapeOrder = 2,
    weight: float = 1.0,
    ion_mass_ratio: float = 100.0,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    """Electrons (q/m = −1) and ions (q/m = +1/ratio) of equal macrocharge ``weight``."""
    count = positions.shape[0]
    species, charged = [], []
    for offset, specific, name, mass in (
        (0, -1.0, "electrons", weight),
        (10**6, 1.0 / ion_mass_ratio, "ions", ion_mass_ratio * weight),
    ):
        support = D.ParticleSetPlan(
            jnp.arange(offset, offset + count),
            mass * jnp.ones((count,)),
            ambient_dimension=3,
        ).prepare()
        charged.append(
            D.ChargedParticlePlan(specific * mass * jnp.ones((count,)), name).prepare(
                support
            )
        )
        species.append(
            D.pic.PICSpeciesPlan(
                D.ParticlePopulationPlan(support),
                D.pic.PICChargeModelPlan(
                    specific,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    transfer = D.pic.PICParticleCochainTransferPlan(bridge, shape_order=shape_order)
    transfers = tuple(transfer.prepare(value) for value in charged)
    currents = tuple(D.pic.ChargeConservingCurrentPlan(value) for value in transfers)
    return tuple(species), transfers, currents


def _plane_wave(solver: Any, mode: int, time: float) -> tuple[np.ndarray, np.ndarray]:
    """``E_y = cos(kx − ωt)`` and its discrete ``B_z`` for the solver's own symbol."""
    plan = solver.plan
    counts = plan.counts
    k = 2.0 * np.pi * mode / (counts[0] * plan.spacing[0])
    omega = float(solver.dispersion_frequency(np.asarray([[k, 0.0, 0.0]]), 0.1)[0])
    x = np.arange(counts[0]) * plan.spacing[0]
    shift = 0.5 * plan.spacing[0] if plan.grid == "staggered" else 0.0
    electric = np.zeros((*counts, 3))
    magnetic = np.zeros((*counts, 3))
    electric[..., 1] = np.cos(k * x - omega * time)[:, None, None]
    # c = 1: |B| = |[k]|E/ω = E, sampled at the B_z location.
    magnetic[..., 2] = np.cos(k * (x + shift) - omega * time)[:, None, None]
    return electric, magnetic


def _with_fields(solver: Any, electric: np.ndarray, magnetic: np.ndarray) -> Any:
    field = solver.field_with_charge(jnp.zeros(solver.plan.counts))
    split = (
        None
        if field.electric_split is None
        else jnp.asarray(electric)[..., :, None] * jnp.eye(3)
    )
    split_b = (
        None
        if field.magnetic_split is None
        else jnp.asarray(magnetic)[..., :, None] * jnp.eye(3)
    )
    return eqx.tree_at(
        lambda s: (s.electric, s.magnetic, s.electric_split, s.magnetic_split),
        field,
        (jnp.asarray(electric), jnp.asarray(magnetic), split, split_b),
        is_leaf=lambda value: value is None,
    )


def _vacuum_run(solver: Any, field: Any, step: float, steps: int) -> tuple[Any, Array]:
    counts = solver.plan.counts
    intervals = solver.plan.current_intervals
    source = sp.SpectralMaxwellSource(
        jnp.zeros((intervals, *counts, 3)), jnp.zeros((intervals, *counts))
    )

    def body(value: Any, _: None) -> tuple[Any, Array]:
        result = solver.advance(jnp.asarray(0.0), value, source, jnp.asarray(step))
        return result.field, result.diagnostics.absorbed_energy

    return eqx.filter_jit(lambda value: jax.lax.scan(body, value, None, length=steps))(
        field
    )


# -- vacuum dispersion ----------------------------------------------------------------


@pytest.mark.parametrize("grid", ["collocated", "staggered"])
def test_infinite_order_vacuum_dispersion_is_exact(grid: sp.SpectralGrid) -> None:
    solver = sp.SpectralMaxwellPlan(_bridge((16, 4, 4), 0.125), grid=grid).prepare()
    k = 2.0 * np.pi * 5 / 2.0
    assert float(
        solver.dispersion_frequency(np.asarray([[k, 0.0, 0.0]]), 0.01)[0]
    ) == pytest.approx(k, rel=1e-14)
    step, steps = 0.9 * float(solver.stable_step), 25
    field = _with_fields(solver, *_plane_wave(solver, 5, 0.0))
    final, _ = _vacuum_run(solver, field, step, steps)
    # Independent reference: the continuum wave cos(k(x − ct)).
    x = np.arange(16) * 0.125
    np.testing.assert_allclose(
        np.asarray(final.electric[:, 0, 0, 1]),
        np.cos(k * (x - step * steps)),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("grid", "order", "symbol"),
    [
        ("staggered", 2, lambda k, h: 2.0 * np.sin(0.5 * k * h) / h),
        (
            "collocated",
            4,
            lambda k, h: (8.0 * np.sin(k * h) - np.sin(2.0 * k * h)) / (6.0 * h),
        ),
    ],
    ids=["staggered-order-2", "collocated-order-4"],
)
def test_finite_order_vacuum_dispersion_follows_modified_wavenumber(
    grid: sp.SpectralGrid, order: int, symbol: Any
) -> None:
    h = 0.125
    solver = sp.SpectralMaxwellPlan(
        _bridge((16, 4, 4), h),
        grid=grid,
        stencil="finite-order",
        stencil_order=order,
        charge_conservation="update-with-rho",
    ).prepare()
    k = 2.0 * np.pi * 5 / 2.0
    omega = symbol(k, h)
    step, steps = 0.9 * float(solver.stable_step), 25
    field = _with_fields(solver, *_plane_wave(solver, 5, 0.0))
    final, _ = _vacuum_run(solver, field, step, steps)
    x = np.arange(16) * h
    np.testing.assert_allclose(
        np.asarray(final.electric[:, 0, 0, 1]),
        np.cos(k * x - omega * step * steps),
        atol=1e-12,
    )
    # The finite stencil is dispersive: the continuum phase is measurably wrong.
    assert (
        np.max(
            np.abs(
                np.asarray(final.electric[:, 0, 0, 1]) - np.cos(k * (x - step * steps))
            )
        )
        > 1e-2
    )


# -- Gauss law per charge-conservation mode -----------------------------------------


_GAUSS_CASES: dict[str, dict[str, Any]] = {
    "standard-correction-collocated": {},
    "standard-correction-staggered": {"grid": "staggered"},
    "standard-vay-staggered": {
        "charge_conservation": "vay-deposition",
        "grid": "staggered",
    },
    "standard-vay-order-4": {
        "charge_conservation": "vay-deposition",
        "grid": "staggered",
        "stencil": "finite-order",
        "stencil_order": 4,
    },
    "standard-rho": {"charge_conservation": "update-with-rho"},
    "linear-j-rho": {
        "charge_conservation": "update-with-rho",
        "time_dependency": "linear-j",
    },
    "multi-j-rho": {
        "charge_conservation": "update-with-rho",
        "time_dependency": "multi-j",
        "current_substeps": 3,
        "grid": "staggered",
    },
    "galilean-correction": {"variant": "galilean", "galilean_velocity": (0.3, 0.0, 0.1)},
    "galilean-rho": {
        "variant": "galilean",
        "galilean_velocity": (0.3, 0.0, 0.1),
        "charge_conservation": "update-with-rho",
    },
    "averaged-galilean-multi-j": {
        "variant": "averaged-galilean",
        "galilean_velocity": (0.3, 0.1, 0.0),
        "charge_conservation": "update-with-rho",
        "time_dependency": "multi-j",
        "current_substeps": 2,
        "grid": "staggered",
    },
}


@pytest.mark.parametrize("options", list(_GAUSS_CASES.values()), ids=list(_GAUSS_CASES))
def test_gauss_law_holds_to_roundoff_per_conservation_mode(
    options: dict[str, Any],
) -> None:
    bridge = _bridge((8, 8, 7), 0.125)
    rng = np.random.default_rng(3)
    count = 24
    positions = rng.uniform(0.0, 1.0, (count, 3)) * np.asarray([1.0, 1.0, 0.875])
    species, transfers, currents = _species(bridge, positions)
    solver = sp.SpectralMaxwellPlan(bridge, **options).prepare(transfers, currents)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=species)
    # Independent deposit↔Gauss pairing: advancing with the deposited current lands
    # the solver's Gauss charge on the deposited end charge.
    assert pic.pairing_defect < 1e-12
    step = 0.5 * float(solver.stable_step)
    velocity = rng.normal(0.0, 0.2, (count, 3))
    state = pic.initialize((positions, positions), (velocity, np.zeros((count, 3))), step)
    advance = eqx.filter_jit(lambda value: pic.step_detailed(value, step))
    for _ in range(3):
        result = advance(state)
        assert bool(result.successful)
        charge = np.asarray(result.accepted_state.field.charge)
        scale = np.max(np.abs(charge))
        assert scale > 0.0
        assert float(result.diagnostics.electric_constraint) < 1e-11 * scale
        assert float(result.diagnostics.magnetic_constraint) < 1e-11 * scale
        state = result.accepted_state


# -- absorber, decomposition, and far field ------------------------------------------


def _pulse(solver: Any, center: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Right-moving Gaussian ``E_y = B_z`` pulse along x (c = 1)."""
    plan = solver.plan
    x = np.arange(plan.counts[0]) * plan.spacing[0]
    shift = 0.5 * plan.spacing[0] if plan.grid == "staggered" else 0.0
    electric = np.zeros((*plan.counts, 3))
    magnetic = np.zeros((*plan.counts, 3))
    electric[..., 1] = np.exp(-(((x - center) / width) ** 2))[:, None, None]
    magnetic[..., 2] = np.exp(-(((x + shift - center) / width) ** 2))[:, None, None]
    return electric, magnetic


def test_psatd_pml_absorbs_a_normally_incident_pulse() -> None:
    h, cells = 0.1, 16
    bridge = _bridge((96, 4, 4), h)
    pml = sp.SpectralPMLPlan((cells, 0, 0), reflection=1e-6, profile_power=3.0)
    absorbing = sp.SpectralMaxwellPlan(
        bridge, grid="staggered", absorber="psatd-pml", pml=pml
    ).prepare()
    periodic = sp.SpectralMaxwellPlan(bridge, grid="staggered").prepare()
    step = 0.5 * float(absorbing.stable_step)
    # The pulse crosses the interior, enters the layer, and any reflection from the
    # graded layer has returned into the interior by the end of the run.
    steps = int((48 + 3 * cells) * h / step) + 1
    pulse = _pulse(absorbing, 48 * h, 4 * h)
    initial = float(absorbing.field_energy(_with_fields(absorbing, *pulse)))
    final, absorbed = _vacuum_run(absorbing, _with_fields(absorbing, *pulse), step, steps)
    remaining = float(absorbing.field_energy(final)) / initial
    # Continuum target |R|² = 1e-12 for the declared layer; discretization adds little.
    assert remaining < 1e-9
    np.testing.assert_allclose(
        float(np.sum(absorbed)) / initial, 1.0 - remaining, atol=1e-9
    )
    lossless, zero = _vacuum_run(periodic, _with_fields(periodic, *pulse), step, steps)
    assert float(periodic.field_energy(lossless)) / initial == pytest.approx(
        1.0, abs=1e-12
    )
    assert float(np.sum(zero)) == 0.0


def _spectral_divergence(electric: np.ndarray, spacing: float, grid: str) -> np.ndarray:
    """Independent infinite-order ``∇⁻·E`` at the nodes (NumPy FFT per axis)."""
    result = np.zeros(electric.shape[:3])
    for axis in range(3):
        count = electric.shape[axis]
        k = 2.0 * np.pi * np.fft.fftfreq(count, d=spacing)
        if grid == "staggered":
            # E_a sits half a cell up; the derivative lands back on the nodes.
            symbol = 1j * k * np.exp(-0.5j * k * spacing)
        else:
            symbol = np.where(np.abs(k) < np.pi / spacing, 1j * k, 0.0)
        shape = [1, 1, 1]
        shape[axis] = count
        result += np.real(
            np.fft.ifft(
                np.fft.fft(electric[..., axis], axis=axis) * symbol.reshape(shape),
                axis=axis,
            )
        )
    return result


def _fourth_order_divergence(electric: np.ndarray, spacing: float) -> np.ndarray:
    """Independent staggered fourth-order ``∇⁻·E`` (weights 9/8, −1/24)."""
    result = np.zeros(electric.shape[:3])
    for axis in range(3):
        field = electric[..., axis]
        result += (
            9.0 / 8.0 * (field - np.roll(field, 1, axis))
            - (np.roll(field, -1, axis) - np.roll(field, 2, axis)) / 24.0
        ) / spacing
    return result


_PML_GAUSS_CASES: dict[str, dict[str, Any]] = {
    "staggered-infinite-order": {"grid": "staggered"},
    "collocated-infinite-order": {"grid": "collocated"},
    "staggered-order-4": {
        "grid": "staggered",
        "stencil": "finite-order",
        "stencil_order": 4,
        "charge_conservation": "vay-deposition",
    },
}


@pytest.mark.parametrize(
    "options", list(_PML_GAUSS_CASES.values()), ids=list(_PML_GAUSS_CASES)
)
def test_psatd_pml_keeps_gauss_law_outside_its_layers_over_long_runs(
    options: dict[str, Any],
) -> None:
    # A static dipole whose Coulomb field fills the layers: the split-field
    # damping keeps creating divergence there for the whole run.
    h, counts = 0.1, (24, 12, 12)
    solver = sp.SpectralMaxwellPlan(
        _bridge(counts, h),
        absorber="psatd-pml",
        pml=sp.SpectralPMLPlan((4, 4, 4)),
        **options,
    ).prepare()
    rho = np.zeros(counts)
    rho[10, 6, 6], rho[14, 6, 6] = 1.0, -1.0
    field, neutral = solver.initialize_field(jnp.asarray(rho))
    assert bool(neutral)
    step = 0.5 * float(solver.stable_step)
    source = sp.SpectralMaxwellSource(jnp.zeros((1, *counts, 3)), jnp.zeros((1, *counts)))

    def body(value: Any, _: None) -> tuple[Any, tuple[Array, Array]]:
        result = solver.advance(jnp.asarray(0.0), value, source, jnp.asarray(step))
        diagnostics = result.diagnostics
        return result.field, (
            diagnostics.electric_constraint,
            diagnostics.magnetic_constraint,
        )

    final, (electric, magnetic) = eqx.filter_jit(
        lambda value: jax.lax.scan(body, value, None, length=300)
    )(field)
    support = np.asarray(solver.absorber_support)
    absorber = np.asarray(final.absorber_charge)
    divergence = (
        _fourth_order_divergence(np.asarray(final.electric), h)
        if options.get("stencil") == "finite-order"
        else _spectral_divergence(np.asarray(final.electric), h, options["grid"])
    )
    residual = divergence - rho
    # The layers created absorber charge comparable to the dipole itself ...
    assert np.max(np.abs(absorber)) > 1e-3
    assert np.max(np.abs(residual[support])) > 1e-3
    # ... confined to the declared support, where the certificate books it: the
    # independent interior Gauss residual stays at roundoff.
    assert np.max(np.abs(absorber[~support])) < 1e-15
    assert np.max(np.abs(residual[~support])) < 1e-13
    assert float(np.max(electric)) < 1e-13
    assert float(np.max(magnetic)) < 1e-13


def test_psatd_pml_absorbs_an_oblique_three_dimensional_pulse() -> None:
    h, counts = 0.1, (32, 32, 32)
    solver = sp.SpectralMaxwellPlan(
        _bridge(counts, h),
        grid="staggered",
        absorber="psatd-pml",
        pml=sp.SpectralPMLPlan((6, 6, 6), reflection=1e-6, profile_power=3.0),
    ).prepare()
    # E = ∇ × (0, 0, g) of an off-center Gaussian g: a divergence-free burst
    # that meets every layer obliquely.
    x = np.arange(32) * h
    g = np.exp(
        -((x[:, None, None] - 1.6) ** 2 + (x[None, :, None] - 1.6) ** 2) / 0.09
        - (x[None, None, :] - 1.2) ** 2 / 0.09
    )
    electric = np.stack(
        (np.gradient(g, h, axis=1), -np.gradient(g, h, axis=0), np.zeros_like(g)), -1
    )
    zero = solver.field_with_charge(jnp.zeros(counts))
    field = solver.project_gauss(
        _with_fields(solver, electric, np.zeros_like(electric)), zero.charge
    ).field
    initial = float(solver.field_energy(field))
    step = 0.5 * float(solver.stable_step)
    steps = int(4.0 / step)
    final, absorbed = _vacuum_run(solver, field, step, steps)
    remaining = float(solver.field_energy(final)) / initial
    residual = _spectral_divergence(np.asarray(final.electric), h, "staggered")
    support = np.asarray(solver.absorber_support)
    # The burst has crossed the layers; what remains is reflection plus the static
    # field of the layer charge (the split-field PML is not divergence-preserving).
    assert remaining < 3e-5
    np.testing.assert_allclose(
        float(np.sum(absorbed)) / initial, 1.0 - remaining, rtol=0, atol=1e-12
    )
    assert np.max(np.abs(residual[~support])) < 1e-12 * np.max(np.abs(electric)) / h


def test_pic_with_psatd_pml_is_accepted_over_a_long_run() -> None:
    h, counts = 0.125, (16, 16, 16)
    bridge = _bridge(counts, h)
    rng = np.random.default_rng(3)
    count = 16
    positions = 0.75 + rng.uniform(0.0, 0.5, (count, 3))
    species, transfers, currents = _species(bridge, positions)
    solver = sp.SpectralMaxwellPlan(
        bridge, grid="staggered", absorber="psatd-pml", pml=sp.SpectralPMLPlan((4, 4, 4))
    ).prepare(transfers, currents)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=species)
    step = 0.5 * float(solver.stable_step)
    velocity = rng.normal(0.0, 0.1, (count, 3))
    state = pic.initialize((positions, positions), (velocity, np.zeros((count, 3))), step)

    def body(value: Any, _: None) -> tuple[Any, tuple[Array, Array, Array]]:
        result = pic.step_detailed(value, step)
        return result.accepted_state, (
            result.successful,
            result.diagnostics.electric_constraint,
            result.accepted_state.field.absorber_charge,
        )

    _, (successful, constraint, absorber) = eqx.filter_jit(
        lambda value: jax.lax.scan(body, value, None, length=200)
    )(state)
    # The particles' fields reach the layers, which book absorber charge; the
    # runtime accepts every step with its Gauss law at roundoff.
    assert bool(jnp.all(successful))
    assert float(jnp.max(jnp.abs(absorber[-1]))) > 1e-6
    assert float(jnp.max(constraint)) < 1e-12 / h**3


def test_local_guarded_equals_global_within_stencil_truncation() -> None:
    h = 0.1
    bridge = _bridge((32, 16, 8), h)
    options: dict[str, Any] = {
        "grid": "staggered",
        "stencil": "finite-order",
        "stencil_order": 8,
        "charge_conservation": "vay-deposition",
    }
    global_solver = sp.SpectralMaxwellPlan(bridge, **options).prepare()
    pulse = _pulse(global_solver, 16 * h, 2 * h)
    step = 0.5 * float(global_solver.stable_step)
    reference, _ = _vacuum_run(
        global_solver, _with_fields(global_solver, *pulse), step, 20
    )
    errors = []
    for guard in (4, 6, 8):
        local = sp.SpectralMaxwellPlan(
            bridge,
            decomposition="local-guarded",
            subdomains=(4, 2, 1),
            guard_cells=(guard, guard, guard),
            **options,
        ).prepare()
        value, _ = _vacuum_run(local, _with_fields(local, *pulse), step, 20)
        errors.append(
            float(jnp.max(jnp.abs(value.electric - reference.electric)))
            / float(jnp.max(jnp.abs(reference.electric)))
        )
    # Truncating the finite-order propagator's tails at the guards: small and
    # decreasing with guard width.
    assert errors[0] < 1e-2
    assert errors[1] < 0.5 * errors[0]
    assert errors[2] < 0.5 * errors[1]


_LENGTH = 2.0
_COUNT = 36
_OMEGA0 = 4.0 * np.pi
_TAU = 0.15
_T0 = 3.0 * _TAU
_STOP = 1.6
_OMEGAS = np.asarray([0.8 * _OMEGA0, _OMEGA0, 1.2 * _OMEGA0])


def _envelope(time: Any) -> Any:
    return jnp.exp(-(((time - _T0) / _TAU) ** 2)) * jnp.sin(_OMEGA0 * (time - _T0))


def _cochain_envelope(time: Any, args: Any) -> Any:
    del args
    return jnp.exp(-(((time - _T0) / _TAU) ** 2)) * jnp.sin(_OMEGA0 * (time - _T0))


def _dipole_far_field(spectral: bool) -> tuple[np.ndarray, np.ndarray]:
    """Far-field ``F_θ`` of a unit z edge current at the domain center.

    The same Hertzian source (edge current ``j = J·h`` with moment ``j h²``) and
    Huygens box drive the spectral solver and the compatible cochain solver.
    """
    mx = phx.solver.maxwell
    h = _LENGTH / _COUNT
    bridge = _bridge((_COUNT, _COUNT, _COUNT), h)
    center = _COUNT // 2
    half = int(0.3 / h)
    acquisition = mx.MaxwellSpectralAcquisition(
        jnp.asarray(_OMEGAS), sign="positive", measure="time-integral", stop_time=_STOP
    )
    exterior = mx.HomogeneousMaxwellExterior()
    theta = np.linspace(0.2, np.pi - 0.2, 9)
    directions = np.stack(
        (np.sin(theta) * np.cos(0.3), np.sin(theta) * np.sin(0.3), np.cos(theta)), axis=-1
    )
    far_field = mx.MaxwellFarFieldPlan(directions, jnp.asarray([0.0, 0.0, 1.0]), exterior)
    if spectral:
        box = sp.SpectralHuygensBoxPlan(
            (center - half,) * 3, (center + half,) * 3, acquisition, exterior
        )
        solver = sp.SpectralMaxwellPlan(
            bridge, grid="staggered", observers=(box,)
        ).prepare()
        step = 0.9 * float(solver.stable_step)
        steps = int(np.ceil(_STOP / step))
        current = np.zeros((_COUNT, _COUNT, _COUNT, 3))
        current[center, center, center, 2] = 1.0 / h
        change = np.zeros((_COUNT, _COUNT, _COUNT))
        change[center, center, center] = -step / h**2
        change[center, center, center + 1] = step / h**2

        def body(field: Any, index: Array) -> tuple[Any, None]:
            time = index * step
            amplitude = _envelope(time + 0.5 * step)
            source = sp.SpectralMaxwellSource(
                (amplitude * current)[None], (amplitude * change)[None]
            )
            return solver.advance(time, field, source, jnp.asarray(step)).field, None

        final, _ = eqx.filter_jit(
            lambda value: jax.lax.scan(body, value, jnp.arange(steps))
        )(solver.field_with_charge(jnp.zeros((_COUNT, _COUNT, _COUNT))))
        phasors = solver.huygens_phasors(final)[0]
    else:
        edge_shape = bridge.orientation_shapes[1][2]
        edge = bridge.orientation_offsets[1][2] + int(
            np.ravel_multi_index((center, center, center), edge_shape)
        )
        runtime = phx.solver.CompatibleMaxwellPlan(
            bridge,
            observers=(
                mx.MaxwellHuygensBoxPlan(
                    bridge,
                    (center - half,) * 3,
                    (center + half,) * 3,
                    acquisition,
                    exterior,
                ),
            ),
            sources=(
                mx.MaxwellElectricCurrentSourcePlan(
                    jnp.asarray([edge]),
                    jnp.asarray([1.0]),
                    envelope=_cochain_envelope,
                ),
            ),
        ).prepare()
        step = 0.9 * float(runtime.stable_dt)
        steps = int(np.ceil(_STOP / step))
        result = mx.solve_compatible_maxwell(
            runtime, runtime.initialize(), 0.0, step, steps
        )
        (sampler,) = (
            observer
            for observer in runtime.observers
            if isinstance(observer, mx.PreparedMaxwellHuygensBox)
        )
        phasors = sampler.surface_phasors(result.final_state.observations[0])
    spectrum = np.asarray(far_field.evaluate(phasors).field_spectrum[..., 0])
    moment = (
        np.exp(1j * _OMEGAS * _T0)
        * (_TAU * np.sqrt(np.pi) / 2j)
        * (
            np.exp(-(_TAU**2) * (_OMEGAS + _OMEGA0) ** 2 / 4.0)
            - np.exp(-(_TAU**2) * (_OMEGAS - _OMEGA0) ** 2 / 4.0)
        )
        * h**2
    )
    position = np.asarray([center, center, center + 0.5]) * h
    exact = (
        -1j
        * _OMEGAS[:, None]
        * moment[:, None]
        * np.sin(theta)[None]
        * np.exp(-1j * _OMEGAS[:, None] * (directions @ position)[None])
        / (4.0 * np.pi)
    )
    return spectrum, exact


def test_dipole_far_field_matches_cochain_huygens_result() -> None:
    spectral, exact = _dipole_far_field(True)
    cochain, _ = _dipole_far_field(False)
    scale = np.linalg.norm(exact, axis=1)
    spectral_error = np.linalg.norm(spectral - exact, axis=1) / scale
    # Analytic Hertzian dipole (Jackson 9.4) at 11, 9, and 7.5 cells per wavelength:
    # Yee-native surface samples leave the O((kh)²) time-DFT and face-edge error.
    assert np.all(spectral_error < np.asarray([0.015, 0.02, 0.03]))
    # Both solvers carry O((kh)²) surface-sampling and source-staggering error.
    difference = np.linalg.norm(spectral - cochain, axis=1) / scale
    assert np.all(difference < np.asarray([0.06, 0.08, 0.12]))


# -- restart and compatibility ---------------------------------------------------------


def test_restart_round_trip_continues_bitwise_and_refuses_other_solvers() -> None:
    bridge = _bridge((8, 8, 7), 0.125)
    rng = np.random.default_rng(5)
    count = 16
    positions = rng.uniform(0.0, 1.0, (count, 3)) * np.asarray([1.0, 1.0, 0.875])
    species, transfers, currents = _species(bridge, positions)
    options: dict[str, Any] = {
        "variant": "averaged-galilean",
        "galilean_velocity": (0.2, 0.0, 0.0),
        "charge_conservation": "update-with-rho",
    }
    solver = sp.SpectralMaxwellPlan(bridge, **options).prepare(transfers, currents)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=species)
    step = 0.4 * float(solver.stable_step)
    velocity = rng.normal(0.0, 0.1, (count, 3))
    state = pic.initialize((positions, positions), (velocity, np.zeros((count, 3))), step)
    advance = eqx.filter_jit(lambda value: pic.step_detailed(value, step).accepted_state)
    state = advance(state)
    restored = pic.restore(pic.checkpoint(state))
    for left, right in zip(
        jax.tree.leaves(advance(state)), jax.tree.leaves(advance(restored)), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    other = sp.SpectralMaxwellPlan(bridge, grid="staggered").prepare(transfers, currents)
    with pytest.raises(ValueError, match="another plan"):
        other.restore_component(solver.restart_component(state.field))


_REFUSALS: dict[str, tuple[dict[str, Any], str]] = {
    "correction-linear-j": (
        {"time_dependency": "linear-j"},
        "spectral-correction requires constant-j",
    ),
    "correction-local": (
        {
            "stencil": "finite-order",
            "stencil_order": 4,
            "decomposition": "local-guarded",
            "subdomains": (2, 2, 1),
            "guard_cells": (2, 2, 2),
        },
        "local-guarded is refused",
    ),
    "vay-galilean": (
        {
            "variant": "galilean",
            "galilean_velocity": (0.5, 0.0, 0.0),
            "charge_conservation": "vay-deposition",
        },
        "standard constant-j PSATD only",
    ),
    "vay-multi-j": (
        {
            "charge_conservation": "vay-deposition",
            "time_dependency": "multi-j",
            "current_substeps": 2,
            "grid": "staggered",
        },
        "standard constant-j PSATD only",
    ),
    "vay-collocated-even": (
        {"charge_conservation": "vay-deposition"},
        "odd spectral extents",
    ),
    "galilean-pml": (
        {
            "variant": "galilean",
            "galilean_velocity": (0.5, 0.0, 0.0),
            "charge_conservation": "update-with-rho",
            "absorber": "psatd-pml",
            "pml": sp.SpectralPMLPlan((2, 0, 0)),
        },
        "Galilean coordinates with a PML are refused",
    ),
    "local-infinite-order": (
        {
            "charge_conservation": "update-with-rho",
            "decomposition": "local-guarded",
            "subdomains": (2, 2, 1),
            "guard_cells": (2, 2, 2),
        },
        "requires finite-order stencils",
    ),
    "averaged-linear-j": (
        {
            "variant": "averaged-galilean",
            "galilean_velocity": (0.5, 0.0, 0.0),
            "charge_conservation": "update-with-rho",
            "time_dependency": "linear-j",
        },
        "piecewise-constant currents",
    ),
    "galilean-without-velocity": ({"variant": "galilean"}, "require galilean_velocity"),
    "superluminal-galilean": (
        {"variant": "galilean", "galilean_velocity": (1.0, 0.0, 0.0)},
        "subluminal",
    ),
    "multi-j-without-substeps": (
        {"time_dependency": "multi-j", "charge_conservation": "update-with-rho"},
        "current_substeps",
    ),
    "odd-stencil-order": (
        {"stencil": "finite-order", "stencil_order": 3},
        "even stencil_order",
    ),
    "pml-without-plan": ({"absorber": "psatd-pml"}, "SpectralPMLPlan"),
    "thin-infinite-order-pml": (
        {"absorber": "psatd-pml", "pml": sp.SpectralPMLPlan((1, 0, 0))},
        "at least two cells",
    ),
}


@pytest.mark.parametrize(
    ("options", "message"), list(_REFUSALS.values()), ids=list(_REFUSALS)
)
def test_incompatible_configurations_are_refused_at_construction(
    options: dict[str, Any], message: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        sp.SpectralMaxwellPlan(_bridge((8, 8, 8), 0.125), **options)


def test_galilean_huygens_sampling_is_refused() -> None:
    mx = phx.solver.maxwell
    box = sp.SpectralHuygensBoxPlan(
        (2, 2, 2),
        (6, 6, 6),
        mx.MaxwellSpectralAcquisition(
            jnp.asarray([1.0]), sign="positive", measure="time-integral"
        ),
        mx.HomogeneousMaxwellExterior(),
    )
    with pytest.raises(ValueError, match="standard"):
        sp.SpectralMaxwellPlan(
            _bridge((8, 8, 8), 0.125),
            variant="galilean",
            galilean_velocity=(0.5, 0.0, 0.0),
            charge_conservation="update-with-rho",
            observers=(box,),
        )


def _huygens_box(lower: int, upper: int) -> Any:
    mx = phx.solver.maxwell
    return sp.SpectralHuygensBoxPlan(
        (lower,) * 3,
        (upper,) * 3,
        mx.MaxwellSpectralAcquisition(
            jnp.asarray([1.0]), sign="positive", measure="time-integral"
        ),
        mx.HomogeneousMaxwellExterior(),
    )


def test_collocated_huygens_sampling_is_refused() -> None:
    # The collocated half-cell current centering spreads every deposited edge
    # current along its axis, so no Huygens surface is current-free.
    with pytest.raises(ValueError, match="grid='staggered'"):
        sp.SpectralMaxwellPlan(
            _bridge((16, 16, 16), 0.125), observers=(_huygens_box(4, 12),)
        )


def test_huygens_box_must_clear_the_pml_by_the_normal_stencil() -> None:
    bridge = _bridge((16, 16, 16), 0.125)
    options: dict[str, Any] = {
        "grid": "staggered",
        "absorber": "psatd-pml",
        "pml": sp.SpectralPMLPlan((3, 3, 3)),
    }
    with pytest.raises(ValueError, match="two cells clear"):
        sp.SpectralMaxwellPlan(bridge, observers=(_huygens_box(4, 12),), **options)
    sp.SpectralMaxwellPlan(bridge, observers=(_huygens_box(5, 11),), **options)


# -- numerical Cherenkov instability ----------------------------------------------------


def _drifting_plasma_high_k_energy(
    gamma: float, variant: str
) -> tuple[np.ndarray, np.ndarray, Any, float, float]:
    """Neutral cold plasma drifting along x at Lorentz factor ``gamma``.

    Electrons are jittered around a lattice of immobile-mass ions; both drift
    together. Returns times, monitored high-|k| energy, the solver, drift speed,
    and lab plasma frequency (``ω_p²/γ = 4`` in units c = 1).
    """
    h = 0.3868
    counts = (24, 3, 12)
    bridge = _bridge(counts, h)
    speed = float(np.sqrt(1.0 - 1.0 / gamma**2))
    plasma2 = 4.0 * gamma
    ix, iy, iz = np.meshgrid(*(np.arange(n) for n in counts), indexing="ij")
    ions = np.stack(((ix + 0.5) * h, (iy + 0.5) * h, (iz + 0.5) * h), axis=-1).reshape(
        -1, 3
    )
    electrons = ions.copy()
    rng = np.random.default_rng(1)
    electrons[:, (0, 2)] += rng.uniform(-0.05 * h, 0.05 * h, (ions.shape[0], 2))
    weight = plasma2 * float(np.prod(counts)) * h**3 / ions.shape[0]
    species, transfers, currents = _species(
        bridge, ions, weight=weight, ion_mass_ratio=1836.0
    )
    options: dict[str, Any] = {}
    if variant != "standard":
        options: dict[str, Any] = {
            "variant": variant,
            "galilean_velocity": (speed, 0.0, 0.0),
            "charge_conservation": "update-with-rho",
        }
    solver = sp.SpectralMaxwellPlan(bridge, **options).prepare(transfers, currents)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=species)
    monitor = sp.SpectralNCIMonitorPlan(solver, high_fraction=0.3)
    step, steps = 0.45 * h, 80
    velocity = np.zeros(ions.shape)
    velocity[:, 0] = speed
    state = eqx.filter_jit(
        lambda: pic.initialize((electrons, ions), (velocity, velocity), step)
    )()

    def body(value: Any, _: None) -> tuple[Any, tuple[Array, Array]]:
        result = pic.step_detailed(value, step)
        return result.accepted_state, (
            monitor.sample(result.accepted_state.field).high_energy,
            result.successful,
        )

    _, (energy, successful) = eqx.filter_jit(
        lambda value: jax.lax.scan(body, value, None, length=steps)
    )(state)
    assert bool(jnp.all(successful))
    return (
        step * (np.arange(steps) + 1.0),
        np.asarray(energy),
        solver,
        speed,
        np.sqrt(plasma2),
    )


@pytest.mark.parametrize("gamma", [10.0, 100.0])
def test_galilean_psatd_suppresses_nci_that_standard_psatd_grows_as_predicted(
    gamma: float,
) -> None:
    times, standard, solver, speed, plasma = _drifting_plasma_high_k_energy(
        gamma, "standard"
    )
    _, galilean, _, _, _ = _drifting_plasma_high_k_energy(gamma, "galilean")
    reference = sp.godfrey_vay_growth_rate(
        solver,
        drift_axis=0,
        transverse_axis=2,
        drift_speed=speed,
        plasma_frequency=plasma,
        step_size=0.45 * 0.3868,
    )
    predicted = float(reference.maximum_growth_rate)
    assert predicted > 0.2
    window = {"start": 4.5, "stop": 11.0}
    grown = float(sp.SpectralNCIMonitorPlan.fit(times, standard, **window).rate)
    held = float(sp.SpectralNCIMonitorPlan.fit(times, galilean, **window).rate)
    # Linear Godfrey–Vay growth of the standard scheme; the resonant peak is
    # sampled by the box's discrete modes, so agreement is within tens of percent.
    assert 0.5 * predicted < grown < 1.5 * predicted
    assert held < 0.15 * predicted
    stop = np.searchsorted(times, 11.0)
    start = np.searchsorted(times, 4.5)
    assert standard[stop] / standard[start] > 20.0
    assert galilean[stop] / galilean[start] < 3.0
