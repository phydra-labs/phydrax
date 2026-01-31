#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.nn.models import ComplexOutputModel, MLP, SeparableMLP


def test_complex_output_single_model_vector():
    # Single model with out_size = 2k
    model = MLP(in_size=3, out_size=6, width_size=8, depth=2)
    wrapped = ComplexOutputModel(model)

    assert wrapped.in_size == 3
    assert wrapped.out_size == "complex_3"

    x = jnp.ones((3,))
    y = wrapped(x)

    assert jnp.iscomplexobj(y)
    assert y.shape == (3,)


def test_complex_output_two_models_vector():
    # Two models each with out_size = k
    m_real = MLP(in_size=4, out_size=3, width_size=8, depth=2)
    m_imag = MLP(in_size=4, out_size=3, width_size=8, depth=2)
    wrapped = ComplexOutputModel((m_real, m_imag))

    assert wrapped.in_size == 4
    assert wrapped.out_size == "complex_3"

    x = jnp.ones((4,))
    y = wrapped(x)

    assert jnp.iscomplexobj(y)
    assert y.shape == (3,)


def test_complex_output_two_models_scalar():
    # Two models with scalar outputs -> complex scalar
    m_real = MLP(in_size=3, out_size="scalar", width_size=8, depth=2)
    m_imag = MLP(in_size=3, out_size="scalar", width_size=8, depth=2)
    wrapped = ComplexOutputModel((m_real, m_imag))

    assert wrapped.in_size == 3
    assert wrapped.out_size == "complex_scalar"

    x = jnp.ones((3,))
    y = wrapped(x)
    assert jnp.iscomplexobj(y)
    assert y.shape == ()

    # Batched via vmap
    xb = jnp.ones((5, 3))
    yb = jnp.vectorize(lambda xi: wrapped(xi), signature="(d)->()")(xb)
    assert jnp.iscomplexobj(yb)
    assert yb.shape == (5,)


def test_complex_output_single_model_k_scalar():
    # Single model with out_size = 2 (k == 1 -> scalar)
    model = MLP(in_size=3, out_size=2, width_size=8, depth=2)
    wrapped = ComplexOutputModel(model)

    assert wrapped.in_size == 3
    assert wrapped.out_size == "complex_scalar"

    x = jnp.ones((3,))
    y = wrapped(x)
    assert jnp.iscomplexobj(y)
    # Squeezed to scalar
    assert y.shape == ()

    # Batched via vmap
    xb = jnp.ones((5, 3))
    yb = jnp.vectorize(lambda xi: wrapped(xi), signature="(d)->()")(xb)
    assert jnp.iscomplexobj(yb)
    assert yb.shape == (5,)


def test_complex_output_errors():
    # Single model with odd out_size
    bad = MLP(in_size=2, out_size=5, width_size=8, depth=2)
    with pytest.raises(ValueError):
        ComplexOutputModel(bad)

    # Two models with mismatched out_size
    m1 = MLP(in_size=2, out_size=2, width_size=8, depth=2)
    m2 = MLP(in_size=2, out_size=3, width_size=8, depth=2)
    with pytest.raises(ValueError):
        ComplexOutputModel((m1, m2))

    # Two models with mismatched in_size
    m3 = MLP(in_size=2, out_size=2, width_size=8, depth=2)
    m4 = MLP(in_size=3, out_size=2, width_size=8, depth=2)
    with pytest.raises(ValueError):
        ComplexOutputModel((m3, m4))


def test_complex_output_separable_single_model_tuple():
    # Separable single model with out_size = 2k
    model = SeparableMLP(in_size=2, out_size=4, width_size=8, depth=2)
    wrapped = ComplexOutputModel(model)

    assert wrapped.in_size == 2
    assert wrapped.out_size == "complex_2"

    x = (jnp.linspace(0.0, 1.0, 5), jnp.linspace(0.0, 1.0, 6))
    y = wrapped(x)

    assert jnp.iscomplexobj(y)
    assert y.shape == (5, 6, 2)


def test_complex_output_separable_two_models_tuple():
    # Two separable models with k = 2
    mr = SeparableMLP(in_size=2, out_size=2, width_size=8, depth=2)
    mi = SeparableMLP(in_size=2, out_size=2, width_size=8, depth=2)
    wrapped = ComplexOutputModel((mr, mi))

    assert wrapped.in_size == 2
    assert wrapped.out_size == "complex_2"

    x = (jnp.linspace(0.0, 1.0, 4), jnp.linspace(0.0, 2.0, 3))
    y = wrapped(x)

    assert jnp.iscomplexobj(y)
    assert y.shape == (4, 3, 2)
