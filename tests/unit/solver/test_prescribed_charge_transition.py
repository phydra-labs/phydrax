#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transition radiation of prescribed charges in the time-domain Maxwell runtime.

References, all independent of the Maxwell runtime (code units ε₀ = μ₀ = c = 1):

* Method of images for a perfect conductor (Jackson, *Classical
  Electrodynamics*, 3rd ed., §2.1 and §13.7): the field of a charge in front of a
  plane conductor equals the vacuum field of the charge plus its mirror image of
  opposite charge. On the Yee lattice the reflection about a node plane maps the
  lattice onto itself, the Whitney deposit of the mirrored path is the mirrored
  deposit, and the odd image pair has identically zero tangential ``E`` on that
  plane, so the front half-space of a staircased PEC sheet equals the vacuum
  image-pair run to roundoff.
* Ginzburg–Frank transition radiation of a charge crossing a perfect conductor
  at normal incidence (Ginzburg and Frank, J. Phys. USSR 9, 353 (1945);
  Ginzburg and Tsytovich, *Transition Radiation and Transition Scattering*, 1990,
  §2.1), backward hemisphere:
  ``d²W/(dω dΩ) = q²β² sin²θ / (4π³ ε₀ c (1 − β² cos²θ)²)``.
* The A1 trajectory far-field spectrum (Liénard–Wiechert, segment-exact route)
  of the same sampled image-pair chords.

Workflow design. The Huygens transform of a finite acquisition window equals
the radiated spectrum only when the surface fields vanish at both window
edges; a charge that starts far from the plate would leave its static
compensating charge and that charge's image in front of the plate, whose
window-edge term ``Ẽ_static e^{iωT}/(iω)`` is as large as the transition
radiation. The charge therefore emerges from the plate at rest (the
coincident-neutral compensator then sits in the conductor), accelerates
smoothly away (erf velocity ramp of width ``σ``), turns around smoothly, and
approaches at constant ``β`` along −z, crossing the sheet on a step time and
freezing behind it. The image pair starts and ends coincident on the plate, so
every static field cancels. The smooth ramps radiate with the Gaussian
spectral weight ``exp(−(ωσ(1 − β cos θ))²/4)`` of an erf velocity step
(≤ 2 % of the transition-radiation amplitude at ``ω = 6``, far less at the
angles that carry the hemisphere energy), so only the sharp crossing survives
there; at ``ω ≤ 4`` they are not negligible and those frequencies are only
compared with A1, which contains them. The matched-interface null uses
magnetic-field probes: the final static state has ``H = 0`` so no window-edge
term appears, and at a lateral distance ``ρ ≥ 5h`` the bound field of the
slow charge is suppressed by ``K₁(ωρ/(βγc)) ~ e^{−8}``.
"""

import functools
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf

import phydrax as phx
from phydrax import ElectromagneticScaleContract
from phydrax.discretization.pic import PIC_CODE_RELATIVITY
from phydrax.electromagnetics import (
    ChargedTrajectory,
    RadiationObserverPlan,
    TrajectoryRadiationPlan,
    TrajectoryRadiationStatus,
)
from phydrax.units import CHARGE, UnitDefinition


D = phx.discretization
PIC = phx.discretization.pic
mx = phx.solver.maxwell

_CELLS = (26, 26, 38)
_SPACING = 0.1
_CPML = 7
# Node plane of the PEC sheet / dielectric interface (domain center in z) and
# the node line x = y = 13 h the charge travels on (no transverse spreading).
_PLANE = 19
_LINE = 13
_BETA = 0.35
_RAMP = 1.0
_DURATION = 12.0
_HUYGENS_OMEGAS = np.asarray([3.0, 4.0, 6.0, 7.5])
_PROBE_OMEGAS = np.asarray([4.0, 7.5])
_NULL_OMEGA = 6.0
_Z_AXIS = np.asarray([0.0, 0.0, 1.0])


@functools.cache
def _bridge() -> Any:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(count) for count in _CELLS),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0] * 3, [count * _SPACING for count in _CELLS]]))
    return D.StructuredCochainBridge(grid)


@functools.cache
def _step() -> float:
    cpml = mx.MaxwellCPMLPlan(_CPML)
    return 0.9 * float(
        phx.solver.CompatibleMaxwellPlan(_bridge(), pml=cpml).prepare().stable_dt
    )


def _step_count() -> int:
    return round(_DURATION / _step())


def _plane_height() -> float:
    return _PLANE * _SPACING


def _line_position() -> float:
    return _LINE * _SPACING


def _ramp_integral(time: np.ndarray) -> np.ndarray:
    """``∫_{−∞}^t ½(1 + erf(s/σ)) ds`` of the erf velocity step."""
    return 0.5 * (
        time * (1.0 + erf(time / _RAMP))
        + _RAMP / np.sqrt(np.pi) * np.exp(-((time / _RAMP) ** 2))
    )


def _crossing_step() -> int:
    return round(7.5 * _RAMP / _step())


def _plate_heights(samples: int) -> np.ndarray:
    """Emerge at rest, turn around, and cross the plate at ``−β`` on a step time.

    ``v_z = β[Φ(t − t_a) − 2Φ(t − t_b)]`` with ``t_b = (t_a + t_c)/2`` returns
    to the plate at ``t_c`` with velocity ``−β`` (erf tails below 1e-3).
    """
    time = _step() * np.arange(samples)
    start = 2.5 * _RAMP
    turn = 0.5 * (start + _crossing_step() * _step())

    def shape(value: np.ndarray) -> np.ndarray:
        return _ramp_integral(value - start) - 2.0 * _ramp_integral(value - turn)

    return _plane_height() + _BETA * (shape(time) - shape(np.zeros((1,))))


def _on_line(heights: np.ndarray) -> np.ndarray:
    line = np.full_like(heights, _line_position())
    return np.stack((line, line, heights), axis=-1)


def _image_pair_positions() -> np.ndarray:
    heights = np.maximum(_plate_heights(_crossing_step() + 2), _plane_height())
    return np.stack(
        (_on_line(heights), _on_line(2.0 * _plane_height() - heights)), axis=1
    )


def _current(count: int) -> Any:
    particles = D.ParticleSetPlan(
        jnp.arange(count), jnp.ones((count,)), ambient_dimension=3
    ).prepare()
    charged = D.ChargedParticlePlan(jnp.ones((count,)), "prescribed").prepare(particles)
    return PIC.ChargeConservingCurrentPlan(
        PIC.PICParticleCochainTransferPlan(_bridge()).prepare(charged)
    )


def _solve(
    positions: np.ndarray, charge: np.ndarray, observers: tuple[Any, ...], **options: Any
) -> tuple[Any, Any]:
    times = _step() * np.arange(positions.shape[0])
    current = _current(positions.shape[1])
    trajectory = mx.PrescribedChargeTrajectory(times, positions)
    steps = _step_count()
    source = mx.PrescribedChargeCurrentSourcePlan(trajectory, current, step_count=steps)
    prepared = phx.solver.CompatibleMaxwellPlan(
        _bridge(),
        sources=(source,),
        observers=observers,
        pml=mx.MaxwellCPMLPlan(_CPML),
        **options,
    ).prepare()
    plan = mx.PrescribedChargeMaxwellPlan(
        prepared, current, trajectory, charge, step_count=steps
    )
    result = mx.solve_prescribed_charge_maxwell(plan)
    # CPML leaves an open O(1e-2) power-ledger residual owned by the runtime;
    # every other status bit (deposit, continuity, Gauss, magnetic, support,
    # finiteness, exit) must be clear.
    status = int(result.evidence.status) & ~int(mx.PrescribedChargeStatus.LEDGER_OPEN)
    assert status == 0, int(result.evidence.status)
    return prepared, result


def _edge_indices(bridge: Any, selected: Any) -> np.ndarray:
    """Packed electric indices of the edges whose ``(axis, i, j, k)`` pass ``selected``."""
    indices = []
    for axis in range(3):
        shape = bridge.orientation_shapes[1][axis]
        nodes = np.indices(shape).reshape(3, -1)
        chosen = np.flatnonzero(selected(axis, nodes))
        indices.append(bridge.orientation_offsets[1][axis] + chosen)
    return np.concatenate(indices)


def _front_edges() -> np.ndarray:
    """Every edge of the closed front half-space ``k ≥ plane`` (incl. CPML)."""
    return _edge_indices(_bridge(), lambda axis, nodes: nodes[2] >= _PLANE)


def _sheet_mask() -> np.ndarray:
    bridge = _bridge()
    count = sum(int(np.prod(shape)) for shape in bridge.orientation_shapes[1])
    mask = np.zeros((count,), dtype=np.bool_)
    sheet = _edge_indices(bridge, lambda axis, nodes: (axis < 2) & (nodes[2] == _PLANE))
    mask[sheet] = True
    return mask


def _acquisition(frequencies: np.ndarray) -> Any:
    return mx.MaxwellSpectralAcquisition(
        jnp.asarray(frequencies), sign="positive", measure="time-integral"
    )


def _front_probe() -> Any:
    return mx.DFTObserverPlan(
        mx.FieldProbePlan("electric", _front_edges()), _acquisition(_PROBE_OMEGAS)
    )


def _huygens_box() -> Any:
    # Five cells around the path transversally, four beyond the ±6.8 h
    # excursion along z, one cell inside the CPML.
    return mx.MaxwellHuygensBoxPlan(
        _bridge(),
        (8, 8, 8),
        (18, 18, 30),
        _acquisition(_HUYGENS_OMEGAS),
        mx.HomogeneousMaxwellExterior(),
    )


@pytest.fixture(scope="module")
def image_pair() -> tuple[Any, Any]:
    return _solve(
        _image_pair_positions(),
        np.asarray([1.0, -1.0]),
        (_front_probe(), _huygens_box()),
    )


@pytest.fixture(scope="module")
def pec_plate() -> tuple[Any, Any]:
    # After the crossing the charge moves on behind the sheet for eight steps
    # and freezes there; nothing behind the sheet reaches the front half-space.
    heights = _plate_heights(_crossing_step() + 9)
    return _solve(
        _on_line(heights)[:, None],
        np.asarray([1.0]),
        (_front_probe(),),
        boundaries=(mx.MaxwellBoundaryPlan("pec", support=_sheet_mask()),),
    )


def test_pec_sheet_front_half_space_equals_vacuum_image_pair(
    pec_plate: tuple[Any, Any], image_pair: tuple[Any, Any]
) -> None:
    pec_maxwell, pec = pec_plate
    image_maxwell, image = image_pair
    pec_spectrum = np.asarray(pec.observations[0])
    image_spectrum = np.asarray(image.observations[0])
    front = _front_edges()
    pec_final = np.asarray(pec_maxwell.electric_field(pec.final_state))[front]
    image_final = np.asarray(image_maxwell.electric_field(image.final_state))[front]

    assert np.max(np.abs(image_spectrum)) > 0.1
    # Exact discrete mirror symmetry: agreement to accumulated roundoff.
    np.testing.assert_allclose(
        pec_spectrum,
        image_spectrum,
        rtol=0.0,
        atol=1e-12 * np.max(np.abs(image_spectrum)),
    )
    np.testing.assert_allclose(
        pec_final, image_final, rtol=0.0, atol=1e-12 * np.max(np.abs(image_final))
    )


def _code_scale() -> ElectromagneticScaleContract:
    return ElectromagneticScaleContract.code_units(
        PIC_CODE_RELATIVITY.dimensional_scale,
        UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
        gravitational_constant=1,
        speed_of_light=1,
        reduced_planck_constant=1,
        boltzmann_constant=1,
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=1,
        constant_set_id="prescribed-charge-transition-test",
    )


def _hemisphere(order: int = 12, azimuths: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre in ``cos θ ∈ (0, 1)`` times uniform azimuths (backward side)."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    cosine, weight = 0.5 * (nodes + 1.0), 0.5 * weights
    phi = 2.0 * np.pi * (np.arange(azimuths) + 0.5) / azimuths
    cos_theta, phi_grid = np.meshgrid(cosine, phi, indexing="ij")
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    directions = np.stack(
        (sin_theta * np.cos(phi_grid), sin_theta * np.sin(phi_grid), cos_theta),
        axis=-1,
    ).reshape(-1, 3)
    quadrature = (
        weight[:, None] * np.full((1, azimuths), 2.0 * np.pi / azimuths)
    ).reshape(-1)
    return directions, quadrature


def _maxwell_far_field(image_pair: tuple[Any, Any], directions: np.ndarray) -> Any:
    maxwell, result = image_pair
    box = next(
        observer
        for observer in maxwell.observers
        if isinstance(observer, mx.PreparedMaxwellHuygensBox)
    )
    phasors = box.surface_phasors(result.final_state.observations[1])
    return mx.MaxwellFarFieldPlan(
        directions, jnp.asarray(_Z_AXIS), mx.HomogeneousMaxwellExterior()
    ).evaluate(phasors)


def _image_pair_trajectory() -> ChargedTrajectory:
    """The Maxwell chords with a rest segment before the start and after the stop.

    The segment-exact route uses the midpoint of the node proper velocities as
    each segment's velocity; nodes follow ``u_{j+1} = 2U_j − u_j`` from rest so
    every segment carries exactly its chord proper velocity ``U_j``.
    """
    samples = _image_pair_positions()
    positions = np.concatenate((samples[:1], samples, samples[-1:]))
    times = _step() * (np.arange(positions.shape[0]) - 1.0)
    chord = np.diff(positions, axis=0) / _step()
    proper_chord = chord / np.sqrt(1.0 - np.sum(chord**2, axis=-1))[..., None]
    proper = np.zeros_like(positions)
    for index in range(proper_chord.shape[0]):
        proper[index + 1] = 2.0 * proper_chord[index] - proper[index]
    return ChargedTrajectory(
        times,
        positions,
        proper,
        np.asarray([1.0, -1.0]),
        np.ones((2,)),
        np.ones(positions.shape[:2], dtype=np.bool_),
        (np.zeros((2,), dtype=np.uint32), np.arange(2, dtype=np.uint32)),
    )


def _cartesian(spectrum: Any, first: Any, second: Any) -> np.ndarray:
    return np.einsum(
        "fdc,cdx->fdx",
        np.asarray(spectrum),
        np.stack((np.asarray(first), np.asarray(second))),
    )


def test_image_pair_far_field_equals_trajectory_radiation(
    image_pair: tuple[Any, Any],
) -> None:
    directions, _ = _hemisphere()
    observers = RadiationObserverPlan(directions, _Z_AXIS)
    reference = (
        TrajectoryRadiationPlan(
            _code_scale(),
            observers,
            _HUYGENS_OMEGAS,
            coherence="coherent",
            route="segment-exact",
        )
        .prepare()
        .evaluate(_image_pair_trajectory())
    )
    far = _maxwell_far_field(image_pair, directions)
    exact = _cartesian(
        reference.field_spectrum, observers.basis_first, observers.basis_second
    )
    computed = _cartesian(far.field_spectrum, far.theta_basis, far.phi_basis)
    error = np.linalg.norm((computed - exact).reshape(4, -1), axis=1) / np.linalg.norm(
        exact.reshape(4, -1), axis=1
    )

    # The sharp stop on the plate is the transition radiation itself; A1 flags
    # only that its amplitude jump is not resolved by the step.
    assert int(reference.evidence.status) == int(
        TrajectoryRadiationStatus.UNRESOLVED_AMPLITUDE
    )
    # Yee phase error ~ (kh)²/24 accumulated over 5–10 cells to the surface:
    # 21, 16, 10.5, and 8.4 cells per wavelength (measured 2.7, 3.4, 2.3, 8.4 %).
    assert np.all(error < np.asarray([0.05, 0.06, 0.05, 0.12])), error


def test_image_pair_backward_energy_matches_ginzburg_frank(
    image_pair: tuple[Any, Any],
) -> None:
    directions, quadrature = _hemisphere()
    far = _maxwell_far_field(image_pair, directions)
    energy = np.asarray(far.spectral_energy)[list(_HUYGENS_OMEGAS).index(6.0)]
    cos_theta = directions[:, 2]
    ginzburg_frank = (
        _BETA**2
        * (1.0 - cos_theta**2)
        / (4.0 * np.pi**3 * (1.0 - _BETA**2 * cos_theta**2) ** 2)
    )

    # ω = 6: 10.5 cells per wavelength and ramp weight ≤ exp(−3.8) (measured
    # hemisphere ratio 0.999, pattern error 0.7 %).
    np.testing.assert_allclose(
        energy @ quadrature, ginzburg_frank @ quadrature, rtol=0.03
    )
    assert np.linalg.norm(energy - ginzburg_frank) < 0.03 * np.linalg.norm(ginzburg_frank)


def _interface_heights() -> np.ndarray:
    """Smooth start at rest in medium 1, cross at ``−β`` at ``t = 5σ``, smooth stop.

    ``v_z = −β[Φ(t − 2.5σ) − Φ(t − 7.5σ)]``; both ramps sit 2.5σ from the
    crossing, so the charge crosses at constant speed.
    """
    time = _step() * np.arange(round(10.0 * _RAMP / _step()) + 1)
    start, stop, crossing = 2.5 * _RAMP, 7.5 * _RAMP, 5.0 * _RAMP

    def shape(value: np.ndarray) -> np.ndarray:
        return _ramp_integral(value - start) - _ramp_integral(value - stop)

    height = _plane_height() + _BETA * (
        shape(np.asarray([crossing])) - shape(np.zeros((1,)))
    )
    return height - _BETA * (shape(time) - shape(np.zeros((1,))))


def _interface_permittivity(lower: float) -> np.ndarray:
    """Medium 1 (ε = 1) above the node plane, medium ``lower`` below it.

    Tangential edges on the interface carry the arithmetic mean, the standard
    staircased interface permittivity.
    """
    bridge = _bridge()
    heights = []
    for axis in range(3):
        nodes = np.indices(bridge.orientation_shapes[1][axis]).reshape(3, -1)
        heights.append(nodes[2] + (0.5 if axis == 2 else 0.0))
    height = np.concatenate(heights)
    return np.where(
        height < _PLANE, lower, np.where(height > _PLANE, 1.0, 0.5 * (1.0 + lower))
    )


def _lateral_faces() -> np.ndarray:
    """Magnetic faces in medium 1 at ``ρ ≥ 5h`` from the path, outside the CPML."""
    bridge = _bridge()
    centers = []
    for normal, shape in zip((2, 1, 0), bridge.orientation_shapes[2], strict=True):
        nodes = np.indices(shape).reshape(3, -1).T.astype(np.float64)
        nodes[:, [axis for axis in range(3) if axis != normal]] += 0.5
        centers.append(nodes)
    center = np.concatenate(centers)
    radius = np.hypot(center[:, 0] - _LINE, center[:, 1] - _LINE)
    inside = np.all(
        (center > _CPML + 1) & (center < np.asarray(_CELLS) - _CPML - 1), axis=1
    )
    return np.flatnonzero(inside & (center[:, 2] > _PLANE + 1) & (radius >= 5.0))


@pytest.fixture(scope="module")
def interface_spectra() -> dict[float, np.ndarray]:
    probe = mx.DFTObserverPlan(
        mx.FieldProbePlan("magnetic", _lateral_faces()),
        _acquisition(np.asarray([_NULL_OMEGA])),
    )
    positions = _on_line(_interface_heights())[:, None]
    return {
        lower: np.asarray(
            _solve(
                positions,
                np.asarray([1.0]),
                (probe,),
                constitutive=mx.DiagonalMaxwellConstitutivePlan(
                    permittivity=_interface_permittivity(lower)
                ),
            )[1].observations[0]
        )
        for lower in (1.0, 4.0)
    }


def test_matched_interface_emits_no_transition_radiation(
    interface_spectra: dict[float, np.ndarray],
) -> None:
    matched, mismatched = (
        float(np.sum(np.abs(interface_spectra[lower]) ** 2)) for lower in (1.0, 4.0)
    )

    # Ginzburg–Frank transition radiation scales as (ε₁ − ε₂)² at small contrast
    # and vanishes for matched media: only the uniform-motion field, its
    # suppressed ramps, and lattice emission of the moving charge remain
    # (measured matched/mismatched energy 1.4 % at ω = 6).
    assert mismatched > 0.0
    assert matched < 0.03 * mismatched
