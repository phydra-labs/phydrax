import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.typing as pt


class NodeDim(pt.Dim):
    pass


def test_as_array_converts_nested_host_sequences_once_to_the_exact_dtype():
    values = pt.as_array([[1, 2], [3, 4]], pt.Float64[NodeDim, NodeDim], "matrix")

    assert isinstance(values, jax.Array)
    assert values.dtype == jnp.float64
    np.testing.assert_array_equal(values, [[1.0, 2.0], [3.0, 4.0]])


def test_as_array_refuses_non_numeric_and_ragged_inputs():
    for text in ("1.0", b"1.0", {"a": 1.0}, ["a", "b"]):
        with pytest.raises(TypeError):
            pt.as_array(text, pt.Float64[NodeDim], "values")
    with pytest.raises(ValueError):
        pt.as_array([[1.0, 2.0], [3.0]], pt.Float64[NodeDim, pt.AnyDim], "ragged")


def test_as_array_applies_the_casting_policy():
    with pytest.raises(TypeError):
        pt.as_array(np.asarray([1.0 + 1.0j]), pt.Float64[NodeDim], "complex")
    with pytest.raises(TypeError):
        pt.as_array(np.asarray([1.5]), pt.Int32[NodeDim], "float", casting="safe")
    flags = pt.as_array([True, False], pt.Float64[NodeDim], "flags")
    assert flags.dtype == jnp.float64


def test_category_forms_keep_the_input_dtype_and_never_promote():
    single = pt.as_array(np.zeros((2,), dtype=np.float32), pt.Float[NodeDim], "single")

    assert single.dtype == jnp.float32
    with pytest.raises(TypeError):
        pt.as_array(np.zeros((2,), dtype=np.int32), pt.Float[NodeDim], "integers")


def test_as_array_keeps_device_values_on_device():
    device = jnp.arange(3.0)
    with jax.transfer_guard_device_to_host("disallow"):
        same = pt.as_array(device, pt.Float64[NodeDim], "device")
        cast = pt.as_array(device.astype(jnp.float32), pt.Float64[NodeDim], "device")
    assert same is device
    assert cast.dtype == jnp.float64


def test_as_host_array_transfers_explicitly_and_refuses_tracers():
    host = pt.as_host_array(jnp.arange(3.0), pt.HostFloat64[NodeDim], "host")

    assert isinstance(host, np.ndarray) and host.dtype == np.float64

    def traced(values):
        return pt.as_host_array(values, pt.HostFloat64[NodeDim], "traced")

    with pytest.raises(TypeError):
        jax.jit(traced)(jnp.arange(3.0))


def test_conversion_forms_must_match_their_backend():
    with pytest.raises(TypeError):
        pt.as_array([1.0], pt.HostFloat64[NodeDim], "wrong")
    with pytest.raises(TypeError):
        pt.as_host_array([1.0], pt.Float64[NodeDim], "wrong")
    with pytest.raises(TypeError):
        pt.as_array([1.0], pt.Size[NodeDim], "wrong")
