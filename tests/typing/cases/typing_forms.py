"""`phydrax.typing` forms are their base types to static checkers."""

from __future__ import annotations

from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from typing_extensions import assert_never, assert_type

import phydrax.typing as pt


class ComponentDim(pt.Dim, minimum=1):
    """Number of chemical components."""


Basis: TypeAlias = Literal["nodal", "modal"]


def masses(
    values: pt.Float64[ComponentDim],
    count: pt.Size[ComponentDim],
    names: pt.Identifiers[ComponentDim],
    key: pt.PRNGKey,
    host: pt.HostFloat64[ComponentDim],
) -> pt.Float64[pt.Scalar]:
    assert_type(values, jax.Array)
    assert_type(count, int)
    assert_type(names, tuple[str, ...])
    assert_type(key, jax.Array)
    assert_type(host, npt.NDArray[np.float64])
    return jnp.sum(values)


def dispatch(basis: Basis) -> int:
    match basis:
        case "nodal":
            return 0
        case "modal":
            return 1
        case _:
            assert_never(basis)


def incomplete_dispatch(basis: Basis) -> int:
    match basis:
        case "nodal":
            return 0
        case _:
            assert_never(basis)  # ty: ignore[type-assertion-failure]


assert_type(pt.parse("nodal", Basis, "basis"), Literal["nodal", "modal"])
assert_type(pt.parse(3, pt.Size[ComponentDim], "count"), int)
assert_type(pt.as_array([1.0, 2.0], pt.Float64[ComponentDim], "values"), jax.Array)
assert_type(
    pt.as_host_array(((1.0, 2.0), (3.0, 4.0)), pt.HostFloat64[pt.AnyDim, pt.AnyDim], "m"),
    npt.NDArray[np.float64],
)

host_single: pt.HostFloat64[ComponentDim] = np.zeros(3, dtype=np.float32)  # ty: ignore[invalid-assignment]
masses(jnp.zeros(2), "2", ("a", "b"), jax.random.key(0), np.zeros(2))  # ty: ignore[invalid-argument-type]
pt.as_array("1.0", pt.Float64[ComponentDim], "text")  # ty: ignore[invalid-argument-type]
pt.as_array({"a": 1.0}, pt.Float64[ComponentDim], "mapping")  # ty: ignore[invalid-argument-type]
