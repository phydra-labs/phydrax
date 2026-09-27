from typing import Any

import jax.numpy as jnp
import pytest

from phydrax.geometry import MeshRegion
from phydrax.operators.integral.layer_potential._galerkin3d import (
    LaplaceSingleLayerDP0GalerkinPolicy3D,
)
from phydrax.operators.integral.layer_potential._scalar_calderon3d import (
    prepare_scalar_calderon_dp0_3d,
    ScalarKernelFamily3D,
)
from phydrax.operators.integral.layer_potential._scalar_formulations3d import (
    scalar_exterior_dirichlet_formulation_3d,
    scalar_helmholtz_cfie_formulation_3d,
    scalar_interior_dirichlet_formulation_3d,
    scalar_interior_neumann_formulation_3d,
    scalar_robin_mixed_formulation_3d,
)
from phydrax.solver._scalar_boundary3d import (
    solve_helmholtz_boundary_3d,
    solve_laplace_boundary_3d,
)


_VERTICES = jnp.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
_FACES = jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32)


def _policy() -> Any:
    return LaplaceSingleLayerDP0GalerkinPolicy3D(
        regular_order=3,
        singular_order=3,
        near_order=3,
        near_ratio=1.0,
        absolute_tolerance=5.0e-2,
        relative_tolerance=5.0e-2,
        target_block_size=3,
        source_block_size=2,
    )


def _calderon(kernel: Any = None) -> Any:
    return prepare_scalar_calderon_dp0_3d(
        MeshRegion(_VERTICES, _FACES), kernel=kernel, policy=_policy()
    )


def test_scalar_boundary3d_scenario_1() -> None:
    calderon = _calderon()
    formulation = scalar_interior_dirichlet_formulation_3d(
        calderon, representation="double-layer"
    )
    data = jnp.ones((calderon.face_count,), dtype=calderon.space.dtype)
    result = solve_laplace_boundary_3d(formulation, data)

    assert bool(result.valid)
    assert result.metadata.pde == "-Delta(u)=0"
    assert result.metadata.side == "interior"
    assert result.metadata.double_layer_jump == -0.5
    assert result.metadata.continuum_certified is False
    # ty: ignore[invalid-argument-type]
    assert jnp.allclose(result.boundary_dirichlet, data, rtol=2.0e-4, atol=2.0e-4)
    assert jnp.allclose(result.solution, -1.0, rtol=2.0e-1, atol=2.0e-1)
    assert result.boundary_neumann is None
    assert result.potential is not None
    calderon = _calderon()
    formulation = scalar_interior_neumann_formulation_3d(calderon)
    areas = calderon.face_areas
    flux = jnp.asarray([1.0, -areas[0] / areas[1], 0.0, 0.0], dtype=areas.dtype)
    result = solve_laplace_boundary_3d(formulation, flux)

    assert bool(result.valid)
    assert "zero-on-each-closed-component" in (result.metadata.compatibility_requirement)
    assert "area-mean" in result.metadata.gauge
    assert jnp.allclose(result.compatibility_residual, 0.0, atol=1.0e-12)
    assert jnp.allclose(result.gauge_residual, 0.0, atol=2.0e-5)
    # ty: ignore[invalid-argument-type]
    assert jnp.array_equal(result.boundary_neumann, flux)

    incompatible = jnp.ones((calderon.face_count,), dtype=calderon.space.dtype)
    with pytest.raises(ValueError, match="zero-flux compatibility"):
        solve_laplace_boundary_3d(formulation, incompatible)
    calderon = _calderon(ScalarKernelFamily3D.outgoing_helmholtz(0.4))
    raw = scalar_exterior_dirichlet_formulation_3d(
        calderon, representation="double-layer"
    )
    assert "singular-at-interior-Neumann" in raw.metadata.resonance_risk

    cfie = scalar_helmholtz_cfie_formulation_3d(calderon, eta=0.7)
    data = jnp.ones((calderon.face_count,), dtype=calderon.space.dtype)
    result = solve_helmholtz_boundary_3d(cfie, data)

    assert bool(result.valid)
    assert result.metadata.formulation_name == ("exterior-Dirichlet-Brakhage-Werner-CFIE")
    assert result.metadata.coupling_parameter == 0.7
    assert "removes-the-standard-raw" in result.metadata.resonance_risk
    assert result.metadata.single_layer_neumann_jump == -0.5
    # ty: ignore[invalid-argument-type]
    assert jnp.allclose(result.boundary_dirichlet, data, rtol=3.0e-4, atol=3.0e-4)
    assert result.potential is not None
    # ty: ignore[unresolved-attribute]
    assert result.potential.eta == 0.7


def test_robin_mixed_trace_solve_and_pure_neumann_route_failure() -> None:
    calderon = _calderon()
    alpha = jnp.asarray([1.0, 0.0, 1.0, 0.5], dtype=calderon.space.dtype)
    beta = jnp.asarray([0.2, 1.0, 0.3, 0.4], dtype=calderon.space.dtype)
    formulation = scalar_robin_mixed_formulation_3d(
        calderon, alpha, beta, side="interior"
    )
    data = jnp.asarray([0.1, -0.2, 0.3, 0.4], dtype=calderon.space.dtype)
    result = solve_laplace_boundary_3d(formulation, data)

    # ty: ignore[unsupported-operator]
    reconstructed = alpha * result.boundary_dirichlet + beta * result.boundary_neumann
    assert bool(result.valid)
    assert jnp.allclose(reconstructed, data, rtol=5.0e-4, atol=5.0e-4)
    assert result.metadata.boundary_condition == "Robin-or-facewise-mixed"

    with pytest.raises(ValueError, match="compatibility and component-gauge"):
        scalar_robin_mixed_formulation_3d(
            calderon,
            jnp.zeros((4,), dtype=calderon.space.dtype),
            jnp.ones((4,), dtype=calderon.space.dtype),
            side="interior",
        )
