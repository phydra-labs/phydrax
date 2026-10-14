"""Consumer-visible comparisons for every admitted lattice provider."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.lattice_field._qcd_recipes import build_su3_gauge_geometry
from phydrax.backends.lattice import (
    create_lattice_provider,
    grid_availability,
    pyquda_availability,
    qlat_availability,
)
from phydrax.discretization._lattice_boundary import LatticeBoundaryPhasePlan
from phydrax.discretization._topology import TensorTopology
from phydrax.operators.path_integral._lattice_fermion import (
    CloverWilsonDiracOperator,
    WilsonDiracOperator,
)


@pytest.mark.parametrize("selector", ["pyquda", "gpt-grid", "qlat"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("extent", [4, 8])
def test_external_lattice_provider_contract(
    selector: Any, dtype: Any, extent: int
) -> None:
    probe = {
        "pyquda": pyquda_availability,
        "gpt-grid": grid_availability,
        "qlat": qlat_availability,
    }[selector]
    evidence = probe()
    if not evidence.available:
        pytest.skip(evidence.reason)
    external = create_lattice_provider(selector)
    if np.dtype(dtype).name not in external.capabilities.dtypes:
        pytest.skip("dtype is outside the provider's canonical capability boundary")
    native = create_lattice_provider("jax-native-lattice")
    shape = (extent,) * 4
    sites = extent**4
    rng = np.random.default_rng(1701)
    with jax.enable_x64():
        boundary = LatticeBoundaryPhasePlan(
            TensorTopology(("x", "y", "z", "t"), shape, periodic=(True,) * 4),
            jnp.asarray((1, np.exp(0.2j), 1, -1), dtype=dtype),
        )
        geometry = build_su3_gauge_geometry(boundary)
        raw = rng.normal(size=(geometry.link_space.num_edges, 3, 3)) + 1j * rng.normal(
            size=(geometry.link_space.num_edges, 3, 3)
        )
        graph_links, _ = np.linalg.qr(raw)
        graph_links[:, :, 0] /= np.linalg.det(graph_links)[:, None]
        graph_links = graph_links.astype(dtype)
        operator = WilsonDiracOperator(
            boundary,
            geometry.representation,
            geometry.transport,
            graph_links,
            mass=0.3,
            lattice_spacing=1.2,
            dtype=dtype,
        )
        links = jnp.asarray(graph_links[np.asarray(geometry.transport.forward_edges)])
        spinor = jnp.asarray(
            (rng.normal(size=(sites, 4, 3)) + 1j * rng.normal(size=(sites, 4, 3))).astype(
                dtype
            )
        )
        action_mask = jnp.ones((sites, 6), dtype=jnp.bool_)
        force_mask = jnp.ones((sites, 4), dtype=jnp.bool_).at[::3, 2].set(False)
        kwargs = {"lattice_shape": shape, "periodic": (True,) * 4}
        tolerance = 8e-5 if dtype == np.complex64 else 4e-11

        def compare(actual: Any, expected: Any) -> None:
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
            assert actual.dtype == expected.dtype

        for name, mask in (("gauge_action", action_mask), ("gauge_force", force_mask)):
            action = getattr(external, name)
            reference = getattr(native, name)
            compare(
                action(links, 5.7, mask, **kwargs), reference(links, 5.7, mask, **kwargs)
            )
            if "jit" in external.capabilities.transformations:
                compare(
                    jax.jit(lambda u: action(u, 5.7, mask, **kwargs))(links),
                    reference(links, 5.7, mask, **kwargs),
                )
        if external.capabilities.supports("fermion.wilson_dslash"):
            raw_clover = rng.normal(size=(sites, 12, 12)) + 1j * rng.normal(
                size=(sites, 12, 12)
            )
            hermitian = 0.002 * (raw_clover + raw_clover.conj().swapaxes(-1, -2))
            term = (
                (np.kron(np.asarray(operator.gamma5), np.eye(3)) @ hermitian)
                .reshape((sites, 4, 3, 4, 3))
                .astype(dtype)
            )
            clover = CloverWilsonDiracOperator(operator, term)
            for op, name in ((operator, "wilson_dslash"), (clover, "clover_dslash")):
                if not external.capabilities.supports("fermion." + name):
                    continue
                action = getattr(external, name)
                reference = getattr(native, name)
                for adjoint in (False, True):
                    compare(
                        action(op, spinor, adjoint=adjoint),
                        reference(op, spinor, adjoint=adjoint),
                    )
                    if "jit" in external.capabilities.transformations:
                        compare(
                            jax.jit(lambda x: action(op, x, adjoint=adjoint))(spinor),
                            reference(op, spinor, adjoint=adjoint),
                        )
                    if external.capabilities.supports("fermion.block_dslash"):
                        block = jnp.stack((spinor, 0.3j * spinor), axis=-1)
                        compare(
                            external.block_dslash(op, block, adjoint=adjoint),
                            native.block_dslash(op, block, adjoint=adjoint),
                        )
                        if "jit" in external.capabilities.transformations:
                            compare(
                                jax.jit(
                                    lambda x: external.block_dslash(
                                        op, x, adjoint=adjoint
                                    )
                                )(block),
                                native.block_dslash(op, block, adjoint=adjoint),
                            )
