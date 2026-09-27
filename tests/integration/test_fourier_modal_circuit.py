from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pytest

from phydrax.linalg import DenseLinearOperator
from phydrax.solver.maxwell.fourier_modal import (
    fourier_modal_scattering_component,
    FrequencyMaxwellMaterial,
    HomogeneousPortModes,
    MaxwellPortScatteringOperator,
    PortScatteringDiagnostics,
    PreparedFourierModalMaxwell,
)


def _prepared(*, grazing: Any = False, exterior_permittivity: Any = 1.0) -> Any:
    size = 2
    identity = jnp.eye(size, dtype=jnp.complex128)
    modes_left = HomogeneousPortModes(
        identity,
        identity,
        jnp.ones((size,)),
        jnp.ones((size,)),
        jnp.ones((size,), dtype="bool"),
        jnp.zeros((size,), dtype="bool"),
        jnp.asarray([grazing, False]),
        ("h0:te", "h0:tm"),
        "left",
    )
    modes_right = HomogeneousPortModes(
        identity,
        identity,
        jnp.ones((size,)),
        jnp.ones((size,)),
        jnp.ones((size,), dtype="bool"),
        jnp.zeros((size,), dtype="bool"),
        jnp.zeros((size,), dtype="bool"),
        ("h0:te", "h0:tm"),
        "right",
    )
    diagnostics = PortScatteringDiagnostics(
        jnp.asarray(0.0), jnp.asarray(True), jnp.asarray(True)
    )
    scattering = MaxwellPortScatteringOperator(
        DenseLinearOperator(jnp.asarray([[11.0, 0.0], [0.0, 12.0]])),
        DenseLinearOperator(jnp.asarray([[21.0, 0.0], [0.0, 22.0]])),
        DenseLinearOperator(jnp.asarray([[31.0, 0.0], [0.0, 32.0]])),
        DenseLinearOperator(jnp.asarray([[41.0, 0.0], [0.0, 42.0]])),
        modes_left,
        modes_right,
        diagnostics,
    )
    exterior = FrequencyMaxwellMaterial(
        exterior_permittivity,
        material_id="shared-exterior-id",
    )
    problem = SimpleNamespace(
        problem_id="artificial-fourier",
        bloch_wavevector=jnp.asarray([0.0, 0.0]),
        angular_frequency=jnp.asarray(2.0),
        superstrate=SimpleNamespace(
            material=exterior,
            reference_distance=jnp.asarray(1.0),
        ),
        substrate=SimpleNamespace(
            material=exterior,
            reference_distance=jnp.asarray(2.0),
        ),
    )
    return PreparedFourierModalMaxwell(
        # ty: ignore[invalid-argument-type]
        problem,
        # ty: ignore[invalid-argument-type]
        SimpleNamespace(),
        (),
        (),
        # ty: ignore[invalid-argument-type]
        SimpleNamespace(),
        modes_left,
        modes_right,
        scattering,
        scattering,
        jnp.ones((size,), dtype=jnp.complex128),
        jnp.ones((size,), dtype=jnp.complex128),
        jnp.ones((size,), dtype=jnp.complex128),
        jnp.ones((size,), dtype=jnp.complex128),
        jnp.asarray(1.0),
        0,
        "artificial-prepared",
    )


def test_fourier_modal_contracts() -> None:
    component = fourier_modal_scattering_component(
        _prepared(), left_modes=("h0:te",), right_modes=("h0:te",)
    )
    assert jnp.array_equal(
        component.evaluate(jnp.asarray(2.0)).matrix,
        jnp.asarray([[31.0, 41.0], [11.0, 21.0]]),
    )
    assert tuple(port.port_id for port in component.ports) == ("left", "right")
    assert tuple(port.coordinate_ids for port in component.ports) == (
        ("h0:te",),
        ("h0:te",),
    )
    # ty: ignore[unresolved-attribute]
    assert float(component.ports[0].references[0].reference_plane) == -1.0
    # ty: ignore[unresolved-attribute]
    assert float(component.ports[1].references[0].reference_plane) == 3.0
    with pytest.raises(ValueError, match="nongrazing"):
        fourier_modal_scattering_component(
            _prepared(grazing=True), left_modes=("h0:te",), right_modes=("h0:te",)
        )
    first = fourier_modal_scattering_component(
        _prepared(exterior_permittivity=1.0),
        left_modes=("h0:te",),
        right_modes=("h0:te",),
    )
    second = fourier_modal_scattering_component(
        _prepared(exterior_permittivity=2.0),
        left_modes=("h0:te",),
        right_modes=("h0:te",),
    )
    # ty: ignore[unresolved-attribute]
    assert first.ports[0].references[0].basis_id != second.ports[0].references[0].basis_id
