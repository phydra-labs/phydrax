#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.lattice_field._qcd_recipes import build_su3_gauge_geometry
from phydrax.backends.lattice import (
    create_lattice_provider,
    grid_availability,
    LatticeKernelCapabilityError,
    NativeJaxLatticeProvider,
)
from phydrax.discretization._lattice_boundary import LatticeBoundaryPhasePlan
from phydrax.discretization._topology import TensorTopology
from phydrax.operators.path_integral._lattice_fermion import (
    CloverWilsonDiracOperator,
    WilsonDiracOperator,
)


@pytest.fixture(scope="module")
def provider() -> Any:
    availability = grid_availability()
    if not availability.available:
        pytest.skip(availability.reason)
    return create_lattice_provider("gpt-grid")


@pytest.fixture(params=[np.complex64, np.complex128])
def problem(request: pytest.FixtureRequest) -> tuple[Any, np.ndarray, np.ndarray]:
    dtype = request.param
    with jax.enable_x64():
        boundary = LatticeBoundaryPhasePlan(
            TensorTopology(("x", "y", "z", "t"), (4, 4, 4, 4), periodic=(True,) * 4),
            jnp.asarray((1, 1, 1, -1), dtype=dtype),
        )
        geometry = build_su3_gauge_geometry(boundary)
        rng = np.random.default_rng(743)
        raw = rng.normal(size=(geometry.link_space.num_edges, 3, 3)) + 1j * rng.normal(
            size=(geometry.link_space.num_edges, 3, 3)
        )
        links, _ = np.linalg.qr(raw)
        links[:, :, 0] /= np.linalg.det(links)[:, None]
        links = links.astype(dtype)
        operator = WilsonDiracOperator(
            boundary,
            geometry.representation,
            geometry.transport,
            links,
            mass=0.3,
            wilson_parameter=0.8,
            lattice_spacing=1.2,
            dtype=dtype,
        )
        field = (rng.normal(size=(256, 4, 3)) + 1j * rng.normal(size=(256, 4, 3))).astype(
            dtype
        )
    return operator, links[np.asarray(geometry.transport.forward_edges)], field


def test_grid_gauge_action_and_masked_force(
    provider: Any, problem: tuple[Any, np.ndarray, np.ndarray]
) -> None:
    _, links, _ = problem
    native = NativeJaxLatticeProvider()
    geometry = {"lattice_shape": (4, 4, 4, 4), "periodic": (True,) * 4}
    action_mask = np.ones((256, 6), dtype=np.bool_)
    force_mask = np.ones((256, 4), dtype=np.bool_)
    force_mask[::3, 2] = False
    tolerance = 3e-5 if links.dtype == np.dtype(np.complex64) else 2e-11
    with jax.enable_x64():
        np.testing.assert_allclose(
            provider.gauge_action(links, 5.4, action_mask, **geometry),
            native.gauge_action(links, 5.4, action_mask, **geometry),
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            provider.gauge_force(links, 5.4, force_mask, **geometry),
            native.gauge_force(links, 5.4, force_mask, **geometry),
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("clover", [False, True])
def test_grid_dirac_and_block(
    provider: Any,
    problem: tuple[Any, np.ndarray, np.ndarray],
    adjoint: bool,
    clover: bool,
) -> None:
    operator, _, field = problem
    native = NativeJaxLatticeProvider()
    tolerance = 3e-5 if field.dtype == np.dtype(np.complex64) else 2e-11
    with jax.enable_x64():
        if clover:
            rng = np.random.default_rng(71)
            raw = rng.normal(size=(256, 12, 12)) + 1j * rng.normal(size=(256, 12, 12))
            hermitian = 0.003 * (raw + raw.conj().swapaxes(-1, -2))
            gamma5 = np.kron(np.asarray(operator.gamma5), np.eye(3))
            term = (gamma5 @ hermitian).reshape((256, 4, 3, 4, 3)).astype(field.dtype)
            operator = CloverWilsonDiracOperator(operator, term)
        external_action = provider.clover_dslash if clover else provider.wilson_dslash
        native_action = native.clover_dslash if clover else native.wilson_dslash
        np.testing.assert_allclose(
            external_action(operator, field, adjoint=adjoint),
            native_action(operator, field, adjoint=adjoint),
            rtol=tolerance,
            atol=tolerance,
        )
        block = np.stack((field, 0.2j * field), axis=-1)
        np.testing.assert_allclose(
            provider.block_dslash(operator, block, adjoint=adjoint),
            native.block_dslash(operator, block, adjoint=adjoint),
            rtol=tolerance,
            atol=tolerance,
        )


def test_grid_refuses_compiled_host_interchange(
    provider: Any, problem: tuple[Any, np.ndarray, np.ndarray]
) -> None:
    operator, _, field = problem
    with pytest.raises(LatticeKernelCapabilityError, match="eager CPU host interchange"):
        jax.jit(lambda values: provider.wilson_dslash(operator, values))(field)


def test_grid_refuses_partial_action_ownership(
    provider: Any, problem: tuple[Any, np.ndarray, np.ndarray]
) -> None:
    _, links, _ = problem
    mask = np.ones((256, 6), dtype=np.bool_)
    mask[0, 0] = False
    with pytest.raises(LatticeKernelCapabilityError, match="fully owned"):
        provider.gauge_action(
            links, 5.4, mask, lattice_shape=(4, 4, 4, 4), periodic=(True,) * 4
        )
