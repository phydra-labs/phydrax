import jax
import jax.numpy as jnp
import pytest

from phydrax.nn.models import ComplexOutputModel, MLP, SeparableMLP


def test_complex_output_contracts() -> None:
    for scan in (False, True):
        single_vector = ComplexOutputModel(
            MLP(in_size=3, out_size=6, width_size=8, depth=2, scan=scan)
        )
        paired_vector = ComplexOutputModel(
            (
                MLP(in_size=4, out_size=3, width_size=8, depth=2, scan=scan),
                MLP(in_size=4, out_size=3, width_size=8, depth=2, scan=scan),
            )
        )
        paired_scalar = ComplexOutputModel(
            (
                MLP(in_size=3, out_size="scalar", width_size=8, depth=2, scan=scan),
                MLP(in_size=3, out_size="scalar", width_size=8, depth=2, scan=scan),
            )
        )
        single_scalar = ComplexOutputModel(
            MLP(in_size=3, out_size=2, width_size=8, depth=2, scan=scan)
        )
        cases = (
            (single_vector, jnp.ones((3,)), "complex_3", (3,)),
            (paired_vector, jnp.ones((4,)), "complex_3", (3,)),
            (paired_scalar, jnp.ones((3,)), "complex_scalar", ()),
            (single_scalar, jnp.ones((3,)), "complex_scalar", ()),
        )
        for model, value, out_size, expected_shape in cases:
            output = model(value)
            assert model.out_size == out_size, (scan, out_size)
            assert jnp.iscomplexobj(output), (scan, out_size)
            assert output.shape == expected_shape, (scan, out_size)

        batch = jnp.ones((5, 3))
        for model in (paired_scalar, single_scalar):
            output = jax.vmap(model)(batch)
            assert jnp.iscomplexobj(output)
            assert output.shape == (5,)
    for scan in (False, True):
        with pytest.raises(ValueError):
            ComplexOutputModel(
                MLP(in_size=2, out_size=5, width_size=8, depth=2, scan=scan)
            )
        with pytest.raises(ValueError):
            ComplexOutputModel(
                (
                    MLP(in_size=2, out_size=2, width_size=8, depth=2, scan=scan),
                    MLP(in_size=2, out_size=3, width_size=8, depth=2, scan=scan),
                )
            )
        with pytest.raises(ValueError):
            ComplexOutputModel(
                (
                    MLP(in_size=2, out_size=2, width_size=8, depth=2, scan=scan),
                    MLP(in_size=3, out_size=2, width_size=8, depth=2, scan=scan),
                )
            )
    for scan in (False, True):
        single = ComplexOutputModel(
            SeparableMLP(
                in_size=2,
                out_size=4,
                width_size=8,
                depth=2,
                scan=scan,
            )
        )
        paired = ComplexOutputModel(
            (
                SeparableMLP(
                    in_size=2,
                    out_size=2,
                    width_size=8,
                    depth=2,
                    scan=scan,
                ),
                SeparableMLP(
                    in_size=2,
                    out_size=2,
                    width_size=8,
                    depth=2,
                    scan=scan,
                ),
            )
        )
        single_output = single((jnp.linspace(0.0, 1.0, 5), jnp.linspace(0.0, 1.0, 6)))
        paired_output = paired((jnp.linspace(0.0, 1.0, 4), jnp.linspace(0.0, 2.0, 3)))
        assert single.out_size == paired.out_size == "complex_2"
        assert jnp.iscomplexobj(single_output)
        assert jnp.iscomplexobj(paired_output)
        assert single_output.shape == (5, 6, 2)
        assert paired_output.shape == (4, 3, 2)
