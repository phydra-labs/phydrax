#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


mx = phx.solver.maxwell


def _bridge(
    shape: tuple[int, ...],
    upper: tuple[float, ...],
    *,
    periodic: tuple[bool, ...] | None = None,
) -> Any:
    flags = (False,) * len(shape) if periodic is None else periodic
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=flag)
            for count, flag in zip(shape, flags, strict=True)
        ),
        axis_names=tuple("xyz"[: len(shape)]),
    ).prepare(jnp.asarray([[0.0] * len(shape), list(upper)]))
    return phx.discretization.StructuredCochainBridge(grid)


def _steady_phasors(
    material: Any,
    displacement: Any,
    flux: Any,
    omega: float,
    step: float,
    steps: int,
    record: Any,
) -> tuple[Array, Array]:
    """Drive the executed ADE with D(t) = Re(D₀ e^{-iωt}) using the leapfrog half steps."""
    half = jnp.asarray(0.5 * step)

    def advance(state: Any, index: Any) -> tuple[Any, Any]:
        start = displacement * jnp.cos(omega * index * step)
        end = displacement * jnp.cos(omega * (index + 1) * step)
        state = material.advance_state(half, state, start, flux, half, None)
        state = material.advance_state(half, state, end, flux, half, None)
        return state, record(state, end)

    _, samples = jax.lax.scan(advance, material.initialize_state(), jnp.arange(steps))
    # Twenty whole periods cancel the conjugate image and any static offset.
    window = int(round(20 * 2.0 * np.pi / omega / step))
    times = (jnp.arange(steps) + 1) * step
    weights = jnp.exp(1j * omega * times[-window:])
    response, drive = samples
    return (
        jnp.tensordot(weights, response[-window:], axes=(0, 0)),
        jnp.tensordot(weights, drive[-window:], axes=(0, 0)),
    )


def test_lorentz_frequency_response_equals_ade_transfer_function() -> None:
    bridge = _bridge((2, 2), (1.0, 1.0))
    layout = mx.MaxwellCochainLayout(bridge, "tez")
    material = mx.LorentzDrudeMaxwellConstitutivePlan(
        mx.MaxwellLorentzPoles([1.2, 0.0], [0.4, 0.6], [0.9, 0.5]),
        permittivity_infinity=2.0,
    ).prepare(bridge.cochain, layout)
    omega = 1.1
    step = 2.0 * np.pi / omega / 3000
    displacement = jnp.ones((layout.electric_count,))
    polarization, electric = _steady_phasors(
        material,
        displacement,
        jnp.zeros((layout.magnetic_count,)),
        omega,
        step,
        3000 * 60,
        lambda state, drive: (
            jnp.sum(state.polarization, axis=0),
            material.electric_field(drive, state),
        ),
    )
    expected = material.frequency_response(omega).permittivity - 2.0
    independent = 0.9 / (1.2**2 - omega**2 - 0.4j * omega) + 0.5 / (
        -(omega**2) - 0.6j * omega
    )
    np.testing.assert_allclose(expected, independent, rtol=1e-12)
    np.testing.assert_allclose(polarization / electric, expected, rtol=2e-4)


def test_magnetized_plasma_frequency_response_equals_ade_transfer_function() -> None:
    bridge = _bridge((2, 2, 2), (1.0, 1.0, 1.0))
    layout = mx.MaxwellCochainLayout(bridge)
    material = mx.MagnetizedColdPlasmaMaxwellConstitutivePlan(
        jnp.asarray([1.0, 0.4]),
        jnp.asarray([[0.3, 0.0, -0.8], [0.0, 0.1, 0.2]]),
        collision_frequency=jnp.asarray([0.5, 0.3]),
    ).prepare(bridge.cochain, layout)
    omega = 1.3
    step = 2.0 * np.pi / omega / 3000
    electric = bridge.pack_edge_circulation(
        tuple(
            jnp.full(shape, value)
            for shape, value in zip(
                bridge.orientation_shapes[1], (1.0, -0.4, 0.7), strict=True
            )
        )
    )
    vertex = 13
    current, field = _steady_phasors(
        material,
        electric,
        jnp.zeros((layout.magnetic_count,)),
        omega,
        step,
        3000 * 40,
        lambda state, drive: (
            jnp.sum(state.current, axis=0)[:, vertex],
            material.coupling.vertex_field(material.electric_field(drive, state))[
                :, vertex
            ],
        ),
    )
    response = material.frequency_response(omega)
    np.testing.assert_allclose(current, response.conductivity[vertex] @ field, rtol=2e-4)
    # Independent cold-plasma conductivity: σ = ε₀ωₚ² ((ν - iω)I + Ω n̂×)⁻¹.
    reference = np.zeros((3, 3), dtype=np.complex128)
    for plasma, cyclotron, collision in (
        (1.0, (0.3, 0.0, -0.8), 0.5),
        (0.4, (0.0, 0.1, 0.2), 0.3),
    ):
        c = np.asarray(cyclotron)
        cross = np.asarray([[0.0, -c[2], c[1]], [c[2], 0.0, -c[0]], [-c[1], c[0], 0.0]])
        reference += plasma**2 * np.linalg.inv(
            (collision - 1j * omega) * np.eye(3) + cross
        )
    np.testing.assert_allclose(response.conductivity[vertex], reference, rtol=1e-12)


def test_frequency_operator_applies_dispersive_constitutive_response() -> None:
    bridge = _bridge((3, 3), (1.0, 1.0))
    layout = mx.MaxwellCochainLayout(bridge, "tez")
    material = mx.LorentzDrudeMaxwellConstitutivePlan(
        mx.MaxwellLorentzPoles([1.0], [0.2], [0.7]),
        magnetic_poles=mx.MaxwellLorentzPoles([2.0], [0.1], [0.3]),
    ).prepare(bridge.cochain, layout)
    omega = 1.4
    operator = mx.FrequencyMaxwellOperator(bridge.cochain, layout, material, omega)
    field = jnp.cos(jnp.arange(layout.electric_count, dtype=jnp.float64))
    permittivity = 1.0 + 0.7 / (1.0 - omega**2 - 0.2j * omega)
    permeability = 1.0 + 0.3 / (4.0 - omega**2 - 0.1j * omega)
    expected = (
        bridge.codifferential(2, bridge.exterior_derivative(1, field) / permeability)
        - omega**2 * permittivity * field
    )
    np.testing.assert_allclose(operator.mv(field), expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="lossless nondispersive"):
        operator.eigensystem(1)


@pytest.mark.parametrize(
    ("shape", "periodic", "polarization", "widths", "boundaries"),
    [
        ((5, 8), (True, False), "tez", (0, 2), ("pec",)),
        ((3, 4, 5), (False, False, False), "full_3d", (1, 1, 1), ()),
    ],
    ids=["tez-cpml-pec", "full-3d-cpml"],
)
def test_direct_route_traces_the_exact_curl_curl_pattern(
    shape: tuple[int, ...],
    periodic: tuple[bool, ...],
    polarization: str,
    widths: tuple[int, ...],
    boundaries: tuple[str, ...],
) -> None:
    bridge = _bridge(shape, (1.0,) * len(shape), periodic=periodic)
    layout = mx.MaxwellCochainLayout(bridge, polarization)  # ty: ignore[invalid-argument-type]
    material = mx.DiagonalMaxwellConstitutivePlan(permittivity=2.0).prepare(
        bridge.cochain, layout
    )
    operator = mx.FrequencyMaxwellOperator(
        bridge,
        layout,
        material,
        3.0,
        stretching=mx.MaxwellCPMLPlan(widths, target_reflection=1e-6),
        boundaries=tuple(mx.MaxwellBoundaryPlan(kind) for kind in boundaries),  # ty: ignore[invalid-argument-type]
    )
    size = operator.size
    # Independent structure: free edges couple through a shared face of the
    # incidence d (dᵀd) plus the diagonal; conductor rows are the identity.
    incidence = bridge.cochain.topology.incidences[1].exterior_derivative().relation
    curl = np.zeros((layout.magnetic_count, size), dtype=np.int64)
    curl[np.asarray(incidence.target_indices), np.asarray(incidence.source_indices)] = 1
    expected = (curl.T @ curl) != 0
    conductor = np.asarray(operator.conductor)
    expected[conductor, :] = False
    expected[:, conductor] = False
    expected[np.arange(size), np.arange(size)] = True

    relation = operator.sparse_coloring().pattern.relation
    traced = np.zeros((size, size), dtype=np.bool_)
    traced[np.asarray(relation.target_indices), np.asarray(relation.source_indices)] = (
        True
    )
    assert np.array_equal(traced, expected)
    source = jnp.cos(jnp.arange(size, dtype=jnp.float64)).astype(jnp.complex128)
    direct = operator.solve(source, method="direct")
    assert bool(direct.converged)
    np.testing.assert_allclose(
        operator.mv(direct.electric), source, rtol=1e-10, atol=1e-10
    )


def _stretched_line(width: int, target: float) -> tuple[Any, Any, Any, Any, float]:
    cells, length = 200, 20.0
    bridge = _bridge((cells, 2), (length, 0.2), periodic=(False, True))
    runtime = phx.solver.CompatibleMaxwellPlan(bridge, polarization="tmz").prepare()
    omega = np.pi
    operator = mx.FrequencyMaxwellOperator(
        bridge,
        runtime.layout,
        runtime.constitutive,
        omega,
        stretching=mx.MaxwellCPMLPlan((width, 0), target_reflection=target),
    )
    (nodes,) = bridge.unpack(0, jnp.arange(operator.size))
    current = jnp.zeros((operator.size,), dtype=jnp.complex128).at[nodes[60, :]].set(1.0)
    source = 1j * omega * current
    solved = operator.solve(source, tolerance=1e-12, restart=200, maxiter=4000)
    assert bool(solved.converged)
    return operator, solved, source, nodes, length / cells


def test_stretched_coordinates_reflect_below_target() -> None:
    operator, solved, _, nodes, spacing = _stretched_line(15, 1e-4)
    omega = float(operator.angular_frequency)
    # Discrete Yee wavenumber of the unstretched interior.
    wavenumber = 2.0 / spacing * np.arcsin(0.5 * omega * spacing)
    samples = np.arange(70, 170)
    line = np.asarray(solved.electric)[np.asarray(nodes)[samples, 0]]
    basis = np.stack(
        (
            np.exp(1j * wavenumber * samples * spacing),
            np.exp(-1j * wavenumber * samples * spacing),
        ),
        axis=1,
    )
    (outgoing, reflected), *_ = np.linalg.lstsq(basis, line, rcond=None)
    assert abs(reflected) / abs(outgoing) < 1e-4
    with pytest.raises(ValueError, match="non-Hermitian"):
        operator.eigensystem(1)


def test_stretched_absorbed_power_closes_the_source_ledger() -> None:
    operator, solved, source, _, _ = _stretched_line(15, 1e-4)
    ledger = operator.power_ledger(solved.electric, source)
    assert float(ledger.source_power) > 0.0
    np.testing.assert_allclose(ledger.electric_material, 0.0, atol=1e-14)
    np.testing.assert_allclose(ledger.magnetic_material, 0.0, atol=1e-14)
    np.testing.assert_allclose(ledger.absorbed_power, ledger.source_power, rtol=1e-9)
    assert float(ledger.relative_residual) < 1e-9


def test_nonlinear_laws_have_no_frequency_response() -> None:
    bridge = _bridge((2, 2), (1.0, 1.0))
    layout = mx.MaxwellCochainLayout(bridge, "tez")
    kerr = mx.KerrPockelsMaxwellConstitutivePlan(
        permittivity=1.0, kerr=0.1, field_bound=1.0
    ).prepare(bridge.cochain, layout)
    with pytest.raises(ValueError, match="no linear frequency response"):
        kerr.frequency_response(1.0)
