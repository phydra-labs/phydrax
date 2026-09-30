#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import itertools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


mx = phx.solver.maxwell

_LENGTH = 2.0
_OMEGA0 = 4.0 * np.pi
_TAU = 0.15
_T0 = 3.0 * _TAU
_STOP = 1.6
_OMEGAS = np.asarray([0.8 * _OMEGA0, _OMEGA0, 1.2 * _OMEGA0])


def _bridge(count: int, dimension: int = 3) -> Any:
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(count) for _ in range(dimension)),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.asarray([[0.0] * dimension, [_LENGTH] * dimension]))
    return phx.discretization.StructuredCochainBridge(grid)


def _huygens_acquisition(frequencies: Any, **window: float) -> Any:
    return mx.MaxwellSpectralAcquisition(
        jnp.asarray(frequencies), sign="positive", measure="time-integral", **window
    )


def _dipole_envelope(time: Any, args: Any) -> Any:
    del args
    return jnp.exp(-(((time - _T0) / _TAU) ** 2)) * jnp.sin(_OMEGA0 * (time - _T0))


def _dipole_moment(omega: np.ndarray, spacing: float) -> np.ndarray:
    """Transient spectrum ``∫ I ℓ e^{iωt} dt`` of the unit edge-current source.

    An edge current cochain ``j`` is ``J·ℓ``; the current through the dual face is
    ``J h²`` so the moment ``I ℓ`` equals ``j h²`` on a uniform grid.
    """
    envelope = (_TAU * np.sqrt(np.pi) / 2j) * (
        np.exp(-(_TAU**2) * (omega + _OMEGA0) ** 2 / 4.0)
        - np.exp(-(_TAU**2) * (omega - _OMEGA0) ** 2 / 4.0)
    )
    return np.exp(1j * omega * _T0) * envelope * spacing**2


def _sphere(order: int = 16, azimuths: int = 32) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    phi = 2.0 * np.pi * (np.arange(azimuths) + 0.5) / azimuths
    cos_theta, phi_grid = np.meshgrid(nodes, phi, indexing="ij")
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    directions = np.stack(
        (sin_theta * np.cos(phi_grid), sin_theta * np.sin(phi_grid), cos_theta), axis=-1
    ).reshape(-1, 3)
    quadrature = (
        weights[:, None] * np.full((1, azimuths), 2.0 * np.pi / azimuths)
    ).reshape(-1)
    return directions, quadrature


def _dipole_run(count: int) -> dict[str, Any]:
    """Hertzian z-dipole at the domain center observed by two nested Huygens boxes."""
    bridge = _bridge(count)
    spacing = _LENGTH / count
    center = count // 2
    edge_shape = bridge.orientation_shapes[1][2]
    edge = bridge.orientation_offsets[1][2] + int(
        np.ravel_multi_index((center, center, center), edge_shape)
    )
    position = np.asarray([center, center, center + 0.5]) * spacing
    half = int(0.3 / spacing)
    inner = ((center - half,) * 3, (center + half,) * 3)
    outer = (
        tuple(value - 1 for value in inner[0]),
        tuple(value + 2 for value in inner[1]),
    )
    acquisition = _huygens_acquisition(_OMEGAS, stop_time=_STOP)
    exterior = mx.HomogeneousMaxwellExterior()
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        observers=(
            mx.MaxwellHuygensBoxPlan(bridge, *inner, acquisition, exterior),
            mx.MaxwellHuygensBoxPlan(bridge, *outer, acquisition, exterior),
        ),
        sources=(
            mx.MaxwellElectricCurrentSourcePlan(
                jnp.asarray([edge]), jnp.asarray([1.0]), envelope=_dipole_envelope
            ),
        ),
    ).prepare()
    step = 0.9 * float(runtime.stable_dt)
    steps = int(np.ceil(_STOP / step))
    result = mx.solve_compatible_maxwell(runtime, runtime.initialize(), 0.0, step, steps)
    directions, quadrature = _sphere()
    far_field = mx.MaxwellFarFieldPlan(directions, jnp.asarray([0.0, 0.0, 1.0]), exterior)
    samplers = tuple(
        observer
        for observer in runtime.observers
        if isinstance(observer, mx.PreparedMaxwellHuygensBox)
    )
    phasors = tuple(
        sampler.surface_phasors(value)
        for sampler, value in zip(samplers, result.final_state.observations, strict=True)
    )
    return {
        "spacing": spacing,
        "position": position,
        "directions": directions,
        "quadrature": quadrature,
        "phasors": phasors,
        "far": tuple(far_field.evaluate(value) for value in phasors),
    }


@pytest.fixture(scope="module")
def dipole_runs() -> dict[int, dict[str, Any]]:
    return {count: _dipole_run(count) for count in (24, 36)}


def _analytic_theta(run: dict[str, Any]) -> np.ndarray:
    """``F_θ = −iωμ m sin θ e^{−ik r̂·r₀}/(4π)`` for every acquired frequency."""
    directions = run["directions"]
    sin_theta = np.sqrt(1.0 - directions[:, 2] ** 2)
    moment = _dipole_moment(_OMEGAS, run["spacing"])
    phase = np.exp(-1j * _OMEGAS[:, None] * (directions @ run["position"])[None])
    return (
        -1j * _OMEGAS[:, None] * moment[:, None] * sin_theta[None] * phase / (4 * np.pi)
    )


def _theta_errors(run: dict[str, Any]) -> np.ndarray:
    exact = _analytic_theta(run)
    computed = np.asarray(run["far"][0].field_spectrum[..., 0])
    return np.linalg.norm(computed - exact, axis=1) / np.linalg.norm(exact, axis=1)


def _total_exact(run: dict[str, Any]) -> np.ndarray:
    moment = _dipole_moment(_OMEGAS, run["spacing"])
    return _OMEGAS**2 * np.abs(moment) ** 2 / (6.0 * np.pi**2)


def test_hertzian_dipole_pattern_polarization_and_absolute_energy(
    dipole_runs: dict[int, dict[str, Any]],
) -> None:
    run = dipole_runs[36]
    far = run["far"][0]
    # Error grows with kh: 11, 9, and 7.5 cells per wavelength at N = 36.
    assert np.all(_theta_errors(run) < np.asarray([0.05, 0.06, 0.08]))
    spectrum = np.asarray(far.field_spectrum)
    assert np.all(
        np.linalg.norm(spectrum[..., 1], axis=1)
        < 0.01 * np.linalg.norm(spectrum[..., 0], axis=1)
    )
    stokes = np.asarray(far.stokes)
    intensity = stokes[..., 0]
    np.testing.assert_allclose(
        stokes[..., 1], intensity, rtol=0.0, atol=1e-3 * intensity.max()
    )
    assert np.all(np.abs(stokes[..., 2:]) < 0.02 * intensity.max())
    sin_squared = 1.0 - run["directions"][:, 2] ** 2
    energy = np.asarray(far.spectral_energy)
    total = energy @ run["quadrature"]
    pattern = 3.0 * sin_squared / (8.0 * np.pi)
    pattern_error = np.linalg.norm(energy / total[:, None] - pattern, axis=1)
    assert np.all(pattern_error < np.asarray([0.01, 0.04, 0.1]) * np.linalg.norm(pattern))
    np.testing.assert_allclose(total, _total_exact(run), rtol=0.03)


def test_hertzian_dipole_far_field_converges_under_grid_refinement(
    dipole_runs: dict[int, dict[str, Any]],
) -> None:
    coarse = _theta_errors(dipole_runs[24])
    fine = _theta_errors(dipole_runs[36])
    assert np.all(fine < coarse)
    assert np.sum(fine) < 0.6 * np.sum(coarse)


def test_nested_huygens_boxes_agree(dipole_runs: dict[int, dict[str, Any]]) -> None:
    run = dipole_runs[36]
    inner, outer = (np.asarray(value.field_spectrum) for value in run["far"])
    difference = np.linalg.norm((outer - inner).reshape(3, -1), axis=1)
    assert np.all(difference < 0.1 * np.linalg.norm(inner.reshape(3, -1), axis=1))
    inner_energy, outer_energy = (
        np.asarray(mx.spectral_poynting_energy(value)) for value in run["phasors"]
    )
    np.testing.assert_allclose(outer_energy, inner_energy, rtol=0.02)


def test_surface_poynting_energy_equals_far_field_energy(
    dipole_runs: dict[int, dict[str, Any]],
) -> None:
    run = dipole_runs[36]
    surface = np.asarray(mx.spectral_poynting_energy(run["phasors"][0]))
    sphere = np.asarray(run["far"][0].spectral_energy) @ run["quadrature"]
    np.testing.assert_allclose(surface, sphere, rtol=0.04)
    patches = np.asarray(run["phasors"][0].patches)
    by_face = [
        float(
            mx.spectral_poynting_energy(run["phasors"][0], jnp.asarray(patches == face))[
                1
            ]
        )
        for face in range(6)
    ]
    # A z-dipole radiates outward through every box face, symmetrically in x and y.
    assert min(by_face) > 0.0
    np.testing.assert_allclose(by_face[:4], np.mean(by_face[:4]), rtol=0.05)


def _uniform_cochains(bridge: Any, electric: np.ndarray, magnetic: np.ndarray) -> Any:
    edge_shapes = bridge.orientation_shapes[1]
    face_shapes = bridge.orientation_shapes[2]
    circulation = bridge.pack_edge_circulation(
        tuple(jnp.full(shape, electric[axis]) for axis, shape in enumerate(edge_shapes))
    )
    flux = bridge.pack_face_flux(
        (
            jnp.full(face_shapes[2], magnetic[0]),
            jnp.full(face_shapes[1], magnetic[1]),
            jnp.full(face_shapes[0], magnetic[2]),
        )
    )
    return circulation, flux


def _tangential(normals: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return vector[None] - (normals @ vector)[:, None] * normals


def test_box_reproduces_uniform_tangential_fields_at_zero_frequency() -> None:
    bridge = _bridge(10)
    layout = mx.MaxwellCochainLayout(bridge.cochain, "full_3d")
    sampler = mx.MaxwellHuygensBoxPlan(
        bridge,
        (2, 3, 4),
        (7, 8, 6),
        _huygens_acquisition([0.0]),
        mx.HomogeneousMaxwellExterior(),
    ).prepare(layout)
    electric = np.asarray([0.3, -0.7, 1.1])
    magnetic = np.asarray([-0.4, 0.9, 0.25])
    circulation, flux = _uniform_cochains(bridge, electric, magnetic)
    state = sampler.initialize()
    for time in (0.0, 0.5, 1.25):
        state = sampler.update(jnp.asarray(time), circulation, flux, state)
    phasors = sampler.surface_phasors(state)
    normals = np.asarray(phasors.normals)
    np.testing.assert_allclose(
        np.asarray(phasors.electric[0]), 1.25 * _tangential(normals, electric), atol=1e-13
    )
    np.testing.assert_allclose(
        np.asarray(phasors.magnetic[0]), 1.25 * _tangential(normals, magnetic), atol=1e-13
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(phasors.measures)), 2 * (10 + 10 + 25) * 0.04
    )


def test_oblique_plane_wave_has_zero_net_flux_and_analytic_inflow() -> None:
    bridge = _bridge(16)
    layout = mx.MaxwellCochainLayout(bridge.cochain, "full_3d")
    tau, center = 0.3, 2.0
    direction = np.asarray([2.0, 1.0, 0.5]) / np.linalg.norm([2.0, 1.0, 0.5])
    polarization = np.cross(direction, [0.0, 0.0, 1.0])
    polarization /= np.linalg.norm(polarization)
    circulation, flux = _uniform_cochains(
        bridge, polarization, np.cross(direction, polarization)
    )
    edge_delay = jnp.asarray(np.asarray(bridge.cochain.coordinates[1]) @ direction)
    face_delay = jnp.asarray(np.asarray(bridge.cochain.coordinates[2]) @ direction)
    omegas = np.asarray([0.0, 3.0, 6.0])
    sampler = mx.MaxwellHuygensBoxPlan(
        bridge,
        (4, 4, 4),
        (12, 11, 10),
        _huygens_acquisition(omegas),
        mx.HomogeneousMaxwellExterior(),
    ).prepare(layout)

    def pulse(time: Any) -> Any:
        return jnp.exp(-(((time - center) / tau) ** 2))

    def body(state: Any, time: Any) -> tuple[Any, None]:
        return (
            sampler.update(
                time,
                pulse(time - edge_delay) * circulation,
                pulse(time - face_delay) * flux,
                state,
            ),
            None,
        )

    state, _ = jax.lax.scan(body, sampler.initialize(), jnp.arange(0.0, 6.0, 0.01))
    phasors = sampler.surface_phasors(state)
    inflow_mask = np.asarray(phasors.normals) @ direction < 0.0
    inflow = np.asarray(mx.spectral_poynting_energy(phasors, jnp.asarray(inflow_mask)))
    net = np.asarray(mx.spectral_poynting_energy(phasors))
    assert np.all(np.abs(net) < 1e-12 * np.abs(inflow))
    spectrum = tau * np.sqrt(np.pi) * np.exp(-(omegas**2) * tau**2 / 4.0)
    projected_area = np.sum(
        np.asarray(phasors.measures)
        * np.abs(np.asarray(phasors.normals) @ direction)
        * inflow_mask
    )
    expected = -(spectrum**2) / np.pi * projected_area
    np.testing.assert_allclose(inflow[0], expected[0], rtol=1e-12)
    np.testing.assert_allclose(inflow, expected, rtol=0.06)


def test_trapezoid_time_integral_matches_gaussian_transform() -> None:
    tau, center = 0.4, 3.0
    omegas = np.asarray([0.0, 1.5, 4.0])
    times = np.arange(0.0, 6.0 + 1e-12, 0.05)
    payload = np.exp(-(((times - center) / tau) ** 2))
    transform = tau * np.sqrt(np.pi) * np.exp(-(omegas**2) * tau**2 / 4.0)
    for sign, exponent in (("positive", 1.0), ("negative", -1.0)):
        acquisition = mx.MaxwellSpectralAcquisition(
            jnp.asarray(omegas), sign=sign, measure="time-integral"
        )
        state = acquisition.initialize((1,))
        for time, value in zip(times, payload, strict=True):
            state = acquisition.accumulate(state, jnp.asarray(time), jnp.asarray([value]))
        np.testing.assert_allclose(
            np.asarray(acquisition.value(state))[:, 0],
            transform * np.exp(exponent * 1j * omegas * center),
            rtol=1e-10,
            atol=1e-12,
        )
    windowed = mx.MaxwellSpectralAcquisition(
        jnp.asarray([0.0]),
        sign="positive",
        measure="time-integral",
        start_time=1.0,
        stop_time=2.5,
    )
    state = windowed.initialize(())
    for time in times:
        state = windowed.accumulate(state, jnp.asarray(time), jnp.asarray(1.0))
    np.testing.assert_allclose(np.asarray(windowed.value(state)), [1.5], rtol=1e-12)


def test_sample_mean_reproduces_windowed_dft_mean() -> None:
    bridge = _bridge(3)
    layout = mx.MaxwellCochainLayout(bridge.cochain, "full_3d")
    omegas = np.asarray([0.0, 1.0, 2.5])
    indices = np.asarray([0, 5, 17], dtype=np.int64)
    observer = mx.DFTObserverPlan(
        mx.FieldProbePlan("electric", jnp.asarray(indices)),
        mx.MaxwellSpectralAcquisition(
            jnp.asarray(omegas),
            sign="negative",
            measure="sample-mean",
            start_time=0.2,
            stop_time=0.7,
        ),
    ).prepare(layout)
    rng = np.random.default_rng(3)
    times = np.linspace(0.0, 1.0, 11)
    fields = rng.standard_normal((times.size, layout.electric_count))
    magnetic = jnp.zeros((layout.magnetic_count,))
    state = observer.initialize()
    for time, field in zip(times, fields, strict=True):
        state = observer.update(jnp.asarray(time), jnp.asarray(field), magnetic, state)
    active = (times >= 0.2) & (times <= 0.7)
    expected = np.mean(
        np.exp(-1j * omegas[None, :, None] * times[active, None, None])
        * np.take(fields[active], indices, axis=1)[:, None, :],
        axis=0,
    )
    np.testing.assert_allclose(np.asarray(observer.value(state)), expected, rtol=1e-13)


def test_huygens_box_refuses_inadmissible_configurations() -> None:
    bridge = _bridge(12)
    acquisition = _huygens_acquisition([1.0])
    exterior = mx.HomogeneousMaxwellExterior()
    box = mx.MaxwellHuygensBoxPlan(bridge, (3, 3, 3), (9, 9, 9), acquisition, exterior)

    for measure, sign in (("sample-mean", "positive"), ("time-integral", "negative")):
        with pytest.raises(ValueError):
            mx.MaxwellHuygensBoxPlan(
                bridge,
                (3, 3, 3),
                (9, 9, 9),
                mx.MaxwellSpectralAcquisition(
                    jnp.asarray([1.0]), sign=sign, measure=measure
                ),
                exterior,
            )
    with pytest.raises(ValueError):
        mx.MaxwellHuygensBoxPlan(bridge, (0, 3, 3), (9, 9, 9), acquisition, exterior)
    with pytest.raises(ValueError):
        mx.MaxwellHuygensBoxPlan(_bridge(12, 2), (3, 3), (9, 9), acquisition, exterior)

    def prepare(**options: Any) -> Any:
        return phx.solver.CompatibleMaxwellPlan(
            bridge, observers=(box,), **options
        ).prepare()

    counts = mx.MaxwellCochainLayout(bridge.cochain, "full_3d")
    with pytest.raises(ValueError):
        prepare(pml=mx.MaxwellCPMLPlan(4))
    with pytest.raises(ValueError):
        prepare(constitutive=mx.DiagonalMaxwellConstitutivePlan(permittivity=2.0))
    with pytest.raises(ValueError):
        prepare(
            constitutive=mx.ConductiveMaxwellConstitutivePlan(
                electric_conductivity=jnp.full((counts.electric_count,), 0.1)
            )
        )
    with pytest.raises(ValueError):
        prepare(
            constitutive=mx.drude_maxwell_constitutive(
                jnp.asarray([1.0]), jnp.asarray([0.1])
            )
        )
    edge = bridge.orientation_offsets[1][0] + int(
        np.ravel_multi_index((4, 3, 5), bridge.orientation_shapes[1][0])
    )
    with pytest.raises(ValueError):
        prepare(
            sources=(
                mx.MaxwellElectricCurrentSourcePlan(
                    jnp.asarray([edge]), jnp.asarray([1.0]), angular_frequency=1.0
                ),
            )
        )
    # A lossless conductive law with the declared exterior and a source strictly
    # inside the box are admissible.
    inside = bridge.orientation_offsets[1][2] + int(
        np.ravel_multi_index((6, 6, 6), bridge.orientation_shapes[1][2])
    )
    runtime = prepare(
        constitutive=mx.ConductiveMaxwellConstitutivePlan(),
        sources=(
            mx.MaxwellElectricCurrentSourcePlan(
                jnp.asarray([inside]), jnp.asarray([1.0]), angular_frequency=1.0
            ),
        ),
    )
    sampler = runtime.observers[0]
    assert isinstance(sampler, mx.PreparedMaxwellHuygensBox)
    assert sampler.surface_count == 6 * 36
    with pytest.raises(ValueError, match="different exterior"):
        mx.MaxwellFarFieldPlan(
            jnp.asarray([[1.0, 0.0, 0.0]]),
            jnp.asarray([0.0, 0.0, 1.0]),
            mx.HomogeneousMaxwellExterior(permittivity=2.0),
        ).evaluate(sampler.surface_phasors(runtime.initialize().observations[0]))
    with pytest.raises(ValueError, match="parallel to the reference axis"):
        mx.MaxwellFarFieldPlan(
            jnp.asarray([[0.0, 0.0, 1.0]]), jnp.asarray([0.0, 0.0, 1.0]), exterior
        )


def _kuhn_mesh(count: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    """Positively oriented Kuhn tetrahedra of a unit cube lattice with cell owners."""
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )

    def vertex(i: int, j: int, k: int) -> int:
        return (i * (count + 1) + j) * (count + 1) + k

    cells: list[list[int]] = []
    owners: list[tuple[int, int, int]] = []
    for i, j, k in itertools.product(range(count), repeat=3):
        corners = [
            vertex(i + (bit >> 2 & 1), j + (bit >> 1 & 1), k + (bit & 1))
            for bit in range(8)
        ]
        for first, second in itertools.permutations((4, 2, 1), 2):
            cell = [corners[0], corners[first], corners[first | second], corners[7]]
            local = points[cell]
            if np.linalg.det((local[1:] - local[0]).T) < 0.0:
                cell[0], cell[1] = cell[1], cell[0]
            cells.append(cell)
            owners.append((i, j, k))
    return points, np.asarray(cells), owners


def _closed_surface(
    points: np.ndarray, cells: np.ndarray, owners: list[tuple[int, int, int]]
) -> np.ndarray:
    """Outward-oriented boundary of the central lattice cell (interior mesh faces)."""
    counts: dict[tuple[int, ...], int] = {}
    for cell, owner in zip(cells, owners, strict=True):
        if owner == (1, 1, 1):
            for face in itertools.combinations(sorted(int(value) for value in cell), 3):
                counts[face] = counts.get(face, 0) + 1
    faces = []
    for face, multiplicity in counts.items():
        if multiplicity == 1:
            corners = points[list(face)]
            normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
            if normal @ (corners.mean(axis=0) - 0.5) < 0.0:
                face = (face[0], face[2], face[1])
            faces.append(face)
    return np.asarray(faces)


def test_tetrahedral_surface_whitney_reconstruction_and_refusals() -> None:
    points, cells, owners = _kuhn_mesh(3)
    faces = _closed_surface(points, cells, owners)
    mesh = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("tetrahedra", "tetrahedron", cells),)
    )
    complex_ = phx.discretization.FiniteElementDeRhamComplex(
        mesh, family="trimmed", order=1
    )
    connectivity = phx.discretization.tetrahedral_connectivity(cells, points.shape[0])
    edges = np.asarray(connectivity.edges)
    mesh_faces = np.asarray(connectivity.faces)
    electric = np.asarray([0.3, -0.7, 1.1])
    flux_density = np.asarray([-0.4, 0.9, 0.25])
    circulation = (points[edges[:, 1]] - points[edges[:, 0]]) @ electric
    flux = (
        0.5
        * np.cross(
            points[mesh_faces[:, 1]] - points[mesh_faces[:, 0]],
            points[mesh_faces[:, 2]] - points[mesh_faces[:, 0]],
        )
        @ flux_density
    )
    runtime = mx.UnstructuredMaxwellPlan(
        complex_,
        mx.FiniteElementMaxwellConstitutivePlan(inverse_permeability=0.5),
        spectral_upper_bound=1.0,
        courant_factor=0.9,
    ).prepare()
    acquisition = _huygens_acquisition([0.0], stop_time=2.0)
    sampler = mx.MaxwellHuygensSurfacePlan(
        complex_, faces, acquisition, mx.HomogeneousMaxwellExterior(permeability=2.0)
    ).prepare(runtime)
    magnetic = runtime.constitutive.magnetic_field(
        jnp.asarray(flux), runtime.constitutive.initialize_state()
    )
    state = sampler.initialize()
    for time in (0.0, 2.0):
        state = sampler.update(
            jnp.asarray(time), jnp.asarray(circulation), magnetic, state
        )
    phasors = sampler.surface_phasors(state)
    normals = np.asarray(phasors.normals)
    np.testing.assert_allclose(np.sum(np.asarray(phasors.measures)), 6.0 / 9.0)
    np.testing.assert_allclose(
        np.asarray(phasors.electric[0]), 2.0 * _tangential(normals, electric), atol=1e-12
    )
    # Material weighting lives in the constitutive map: H = B / μ, with μ = 2.
    np.testing.assert_allclose(
        np.asarray(phasors.magnetic[0]),
        _tangential(normals, flux_density),
        atol=1e-12,
    )

    with pytest.raises(ValueError, match="not closed"):
        mx.MaxwellHuygensSurfacePlan(
            complex_,
            faces[:-1],
            acquisition,
            mx.HomogeneousMaxwellExterior(permeability=2.0),
        ).prepare(runtime)
    flipped = faces.copy()
    flipped[0] = flipped[0, [0, 2, 1]]
    with pytest.raises(ValueError, match="consistently oriented"):
        mx.MaxwellHuygensSurfacePlan(
            complex_,
            flipped,
            acquisition,
            mx.HomogeneousMaxwellExterior(permeability=2.0),
        ).prepare(runtime)
    with pytest.raises(ValueError, match="exterior"):
        mx.MaxwellHuygensSurfacePlan(
            complex_, faces, acquisition, mx.HomogeneousMaxwellExterior()
        ).prepare(runtime)
    current = np.zeros(edges.shape[0])
    current[np.asarray(sampler.geometry.electric_indices)[0]] = 1.0
    with pytest.raises(eqx.EquinoxRuntimeError, match="drives the Huygens surface"):
        jax.block_until_ready(
            sampler.update(
                jnp.asarray(1.0),
                jnp.asarray(circulation),
                magnetic,
                sampler.initialize(),
                electric_current=jnp.asarray(current),
            ).accumulator
        )
