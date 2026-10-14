import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._numerics._checkpointed_scan import checkpointed_scan, CheckpointedScanMode


@pytest.mark.parametrize("length", [1, 3], ids=["one-step", "loop"])
def test_scan_refuses_a_body_that_changes_the_carry_dtype(length: int) -> None:
    def body(carry: jax.Array, row: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (carry + row).astype(jnp.float32), row

    with pytest.raises(TypeError):
        checkpointed_scan(
            body,
            jnp.zeros((2,), jnp.float64),
            jnp.ones((length, 2), jnp.float64),
            length=length,
            mode="full",
        )


@pytest.mark.parametrize("length", [1, 3], ids=["one-step", "loop"])
@pytest.mark.parametrize(
    ("mode", "block_size"),
    [("full", None), ("step", None), ("block", 1), ("block", 2)],
    ids=["full", "step", "unit-blocks", "blocks-with-remainder"],
)
def test_scan_matches_the_native_stacked_scan(
    mode: CheckpointedScanMode, block_size: int | None, length: int
) -> None:
    def body(carry: jax.Array, row: jax.Array) -> tuple[jax.Array, jax.Array]:
        carry = carry * 0.5 + jnp.sin(row)
        return carry, carry * row

    initial = jnp.asarray([0.25, -1.0], jnp.float64)
    rows = jnp.asarray([[0.5, 2.0], [1.5, -0.5], [0.0, 3.0]], jnp.float64)[:length]
    expected = jax.lax.scan(body, initial, rows)
    actual = checkpointed_scan(
        body, initial, rows, length=length, mode=mode, block_size=block_size
    )
    assert actual[1].shape == (length, 2)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=1.0e-15, atol=0.0)
