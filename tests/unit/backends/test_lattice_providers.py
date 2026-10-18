"""Consumer-visible comparisons for every admitted lattice provider."""

import json
import os
import subprocess
import sys
from collections.abc import Callable
from math import prod
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.applications.lattice_field._qcd_recipes import build_su3_gauge_geometry
from phydrax.backends.lattice import (
    create_lattice_provider,
    grid_availability,
    LatticeKernelCapabilityError,
    LatticeKernelProvider,
    qlat_availability,
    quda_availability,
)
from phydrax.discretization._lattice_boundary import LatticeBoundaryPhasePlan
from phydrax.discretization._topology import TensorTopology
from phydrax.operators.path_integral._lattice_fermion import (
    canonical_euclidean_gamma_matrices,
    CloverWilsonDiracOperator,
    WilsonDiracOperator,
)


@pytest.mark.parametrize(
    "selector",
    [pytest.param("quda", marks=pytest.mark.cuda_native), "gpt-grid", "qlat"],
)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("extent", [4, 8])
def test_external_lattice_provider_contract(
    selector: Any, dtype: Any, extent: int
) -> None:
    probe = {
        "quda": quda_availability,
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
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
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
            if selector == "quda":
                assert actual.devices() == links.devices()

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


def _quda_provider() -> LatticeKernelProvider:
    evidence = quda_availability()
    if not evidence.available:
        pytest.skip(evidence.reason)
    return create_lattice_provider("quda")


def _wilson_case(
    shape: tuple[int, ...],
    dtype: Any,
    *,
    seed: int = 719,
    wilson_parameter: float = 1.0,
    custom_gamma: bool = False,
) -> tuple[WilsonDiracOperator, jax.Array, jax.Array]:
    rng = np.random.default_rng(seed)
    boundary = LatticeBoundaryPhasePlan(
        TensorTopology(("t", "z", "y", "x"), shape, periodic=(True,) * 4),
        jnp.asarray((np.exp(0.3j), -1, 1, np.exp(-0.2j)), dtype=dtype),
    )
    geometry = build_su3_gauge_geometry(boundary)
    raw = rng.normal(size=(geometry.link_space.num_edges, 3, 3)) + 1j * rng.normal(
        size=(geometry.link_space.num_edges, 3, 3)
    )
    graph_links, _ = np.linalg.qr(raw)
    graph_links[:, :, 0] /= np.linalg.det(graph_links)[:, None]
    gamma, gamma5 = canonical_euclidean_gamma_matrices(4, dtype=dtype)
    if custom_gamma:
        basis, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        gamma = jnp.asarray(basis @ np.asarray(gamma) @ basis.conj().T, dtype=dtype)
        gamma5 = jnp.asarray(basis @ np.asarray(gamma5) @ basis.conj().T, dtype=dtype)
    operator = WilsonDiracOperator(
        boundary,
        geometry.representation,
        geometry.transport,
        graph_links.astype(dtype),
        mass=0.3,
        lattice_spacing=1.2,
        wilson_parameter=wilson_parameter,
        gamma_matrices=gamma,
        gamma5=gamma5,
        dtype=dtype,
    )
    spinor = jnp.asarray(
        rng.normal(size=(prod(shape), 4, 3)) + 1j * rng.normal(size=(prod(shape), 4, 3)),
        dtype=dtype,
    )
    return operator, spinor, operator.links[operator.transport.forward_edges]


@pytest.mark.cuda_native
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("adjoint", [False, True], ids=("forward", "dagger"))
def test_quda_custom_gamma_jit_with_dynamic_operator(dtype: Any, adjoint: bool) -> None:
    external = _quda_provider()
    native = create_lattice_provider("jax-native-lattice")
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        operator, spinor, _ = _wilson_case((2, 4, 2, 6), dtype, custom_gamma=True)
        block = jnp.stack((spinor, 0.7j * spinor, spinor.conj()), axis=-1)
        compiled = eqx.filter_jit(
            lambda op, x: external.block_dslash(op, x, adjoint=adjoint)
        )
        actual = compiled(operator, block)
        expected = native.block_dslash(operator, block, adjoint=adjoint)
        tolerance = 8e-5 if dtype == np.complex64 else 4e-11
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
        assert actual.dtype == expected.dtype


@pytest.mark.cuda_native
@pytest.mark.parametrize("operation", ["gauge_action", "gauge_force", "wilson_dslash"])
@pytest.mark.parametrize("compiled", [False, True], ids=("eager", "jit"))
def test_quda_alternating_gauges_are_exact(operation: str, compiled: bool) -> None:
    external = _quda_provider()
    native = create_lattice_provider("jax-native-lattice")
    shape = (2, 4, 2, 6)
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        first, spinor, links = _wilson_case(shape, np.complex128)
        second, _, other_links = _wilson_case(shape, np.complex128, seed=730)
        if operation == "wilson_dslash":
            wilson_action = (
                eqx.filter_jit(external.wilson_dslash)
                if compiled
                else external.wilson_dslash
            )
            initial = wilson_action(first, spinor)
            other = wilson_action(second, spinor)
            repeated = wilson_action(first, spinor)
            expected = native.wilson_dslash(second, spinor)
        else:
            mask = jnp.ones(
                (prod(shape), 6 if operation == "gauge_action" else 4),
                dtype=jnp.bool_,
            )
            method = (
                external.gauge_action
                if operation == "gauge_action"
                else external.gauge_force
            )
            reference = (
                native.gauge_action if operation == "gauge_action" else native.gauge_force
            )

            def evaluate(values: jax.Array) -> jax.Array:
                return method(
                    values, 5.7, mask, lattice_shape=shape, periodic=(True,) * 4
                )

            gauge_action = jax.jit(evaluate) if compiled else evaluate
            initial = gauge_action(links)
            other = gauge_action(other_links)
            repeated = gauge_action(links)
            expected = reference(
                other_links, 5.7, mask, lattice_shape=shape, periodic=(True,) * 4
            )
        np.testing.assert_array_equal(repeated, initial)
        np.testing.assert_allclose(other, expected, atol=4e-11, rtol=4e-11)
        assert not np.allclose(initial, other, atol=1e-10, rtol=1e-10)


@pytest.mark.cuda_native
@pytest.mark.parametrize("compiled", [False, True], ids=("eager", "jit"))
def test_quda_action_refuses_partial_ownership(compiled: bool) -> None:
    provider = _quda_provider()
    shape = (2,) * 4
    _, _, links = _wilson_case(shape, np.complex64)
    mask = jnp.ones((16, 6), dtype=jnp.bool_).at[0, 0].set(False)

    def evaluate(values: jax.Array, ownership: jax.Array) -> jax.Array:
        return provider.gauge_action(
            values, 5.7, ownership, lattice_shape=shape, periodic=(True,) * 4
        )

    action = jax.jit(evaluate) if compiled else evaluate
    with pytest.raises(jax.errors.JaxRuntimeError, match="fully owned"):
        action(links, mask).block_until_ready()


@pytest.mark.cuda_native
def test_quda_refuses_multiple_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _quda_provider()
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    evidence = quda_availability()
    assert not evidence.available
    assert "single JAX process" in evidence.reason
    with pytest.raises(LatticeKernelCapabilityError, match="single JAX process"):
        provider.gauge_action(
            jnp.broadcast_to(jnp.eye(3, dtype=jnp.complex64), (16, 4, 3, 3)),
            1.0,
            jnp.ones((16, 6), dtype=jnp.bool_),
            lattice_shape=(2,) * 4,
            periodic=(True,) * 4,
        )


@pytest.mark.cuda_native
@pytest.mark.parametrize("constraint", ["odd", "nonperiodic", "wilson-r"])
def test_quda_refuses_unsupported_geometry(constraint: str) -> None:
    provider = _quda_provider()
    if constraint == "wilson-r":
        operator, spinor, _ = _wilson_case((2,) * 4, np.complex64, wilson_parameter=0.7)
        with pytest.raises(LatticeKernelCapabilityError, match="r=1"):
            provider.wilson_dslash(operator, spinor)
        return
    shape = (3, 2, 2, 2) if constraint == "odd" else (2,) * 4
    periodic = (False, True, True, True) if constraint == "nonperiodic" else (True,) * 4
    with pytest.raises(
        LatticeKernelCapabilityError, match="periodic four-dimensional even"
    ):
        provider.gauge_action(
            jnp.broadcast_to(jnp.eye(3, dtype=jnp.complex64), (prod(shape), 4, 3, 3)),
            1.0,
            jnp.ones((prod(shape), 6), dtype=jnp.bool_),
            lattice_shape=shape,
            periodic=periodic,
        )


def _quda_subprocess(
    source: str, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[3]
    qualifying = "PHYDRAX_CUDA_QUALIFICATION_GPUS" in environment
    environment = {
        key: value for key, value in environment.items() if key != "LD_LIBRARY_PATH"
    }
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path.cwd() if qualifying else root,
        env={
            **environment,
            "PYTHONPATH": (
                environment.get("PYTHONPATH", str(root)) if qualifying else str(root)
            ),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode == 77:
        pytest.skip(completed.stdout.strip())
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


_QUDA_SUBPROCESS_SETUP = """
import sys
import jax
import jax.numpy as jnp
from phydrax.backends.lattice import create_lattice_provider, quda_availability
evidence = quda_availability()
if not evidence.available:
    print(evidence.reason, flush=True)
    sys.exit(77)
provider = create_lattice_provider("quda")
links = jnp.broadcast_to(jnp.eye(3, dtype=jnp.complex64), (16, 4, 3, 3))
mask = jnp.ones((16, 6), dtype=jnp.bool_)
def action(u: jax.Array, m: jax.Array) -> jax.Array:
    return provider.gauge_action(u, 1.0, m, lattice_shape=(2,)*4, periodic=(True,)*4)
"""


@pytest.mark.cuda_native
def test_quda_independent_process_on_each_selected_device(
    record_property: Callable[[str, object], None],
) -> None:
    _quda_provider()
    visible = jax.devices("gpu")
    count = int(os.environ.get("PHYDRAX_CUDA_QUALIFICATION_GPUS", len(visible)))
    assert 0 < count <= len(visible), (
        "Required QUDA qualification devices are unavailable"
    )
    covered = []
    for index, device in enumerate(visible[:count]):
        completed = _quda_subprocess(
            f"DEVICE_INDEX = {index}\n"
            + """
import json
import jax
import numpy as np
from phydrax._jax_config import default_device
from tests.unit.backends.test_lattice_providers import test_external_lattice_provider_contract
device = jax.devices("gpu")[DEVICE_INDEX]
# One QUDA context per child process; never imply multi-device QUDA support.
with default_device(device):
    for dtype in (np.complex64, np.complex128):
        test_external_lattice_provider_contract("quda", dtype, 4)
jax.effects_barrier()
print("QUDA_DEVICE " + json.dumps(device.id), flush=True)
""",
            dict(os.environ),
        )
        reports = [
            json.loads(line.removeprefix("QUDA_DEVICE "))
            for line in completed.stdout.splitlines()
            if line.startswith("QUDA_DEVICE ")
        ]
        assert reports == [device.id], completed.stdout + completed.stderr
        covered.extend(reports)
    record_property("cuda_native_devices", json.dumps(covered))


@pytest.mark.cuda_native
def test_quda_refuses_second_device() -> None:
    if os.environ.get("PHYDRAX_CUDA_QUALIFICATION_GPUS") == "1":
        pytest.skip("QUDA second-device refusal requires two selected CUDA devices")
    _quda_subprocess(
        _QUDA_SUBPROCESS_SETUP
        + """
devices = jax.devices("gpu")
if len(devices) < 2:
    print("QUDA second-device refusal requires two CUDA devices", flush=True)
    sys.exit(77)
first = jax.jit(action)
second = jax.jit(action)
first(jax.device_put(links, devices[0]), jax.device_put(mask, devices[0])).block_until_ready()
try:
    second(jax.device_put(links, devices[1]), jax.device_put(mask, devices[1])).block_until_ready()
except jax.errors.JaxRuntimeError as error:
    assert "device" in str(error).lower() and ("one" in str(error).lower() or "second" in str(error).lower()), str(error)
else:
    raise AssertionError("QUDA accepted a second device")
""",
        dict(os.environ),
    )


@pytest.mark.cuda_native
@pytest.mark.parametrize("user_path", [False, True], ids=("default", "user"))
def test_quda_resource_path_subprocess(tmp_path: Path, user_path: bool) -> None:
    environment = dict(os.environ)
    environment["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    environment.pop("QUDA_RESOURCE_PATH", None)
    if user_path:
        resource = tmp_path / "user-quda"
        resource.mkdir()
        environment["QUDA_RESOURCE_PATH"] = str(resource)
    _quda_subprocess(
        _QUDA_SUBPROCESS_SETUP
        + """
import ctypes
import os
from pathlib import Path
from phydrax._cuda_native import load_cuda_component
component = load_cuda_component("quda")
requested = os.environ.get("QUDA_RESOURCE_PATH")
action(links, mask).block_until_ready()
libc = ctypes.CDLL(None)
libc.getenv.argtypes = [ctypes.c_char_p]
libc.getenv.restype = ctypes.c_char_p
actual = Path(libc.getenv(b"QUDA_RESOURCE_PATH").decode())
if requested is not None:
    assert actual == Path(requested), (actual, requested)
else:
    # One directory per QUDA build variant (architectures, compiler version).
    release = Path(os.environ["XDG_CACHE_HOME"]) / "phydrax" / "quda" / component.build_hash
    assert actual.parent == release, (actual, release)
    assert "gpu_arch" in actual.name and "cuda_version" in actual.name, actual
assert actual.is_dir(), actual
""",
        environment,
    )
