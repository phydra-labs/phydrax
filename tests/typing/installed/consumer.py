"""Consumer view of an installed Phydrax wheel.

Checked by `tools/check_installed_typing.py` from outside the repository, so every
`phydrax` import resolves to the installed distribution. Deliberately invalid
lines carry `# ty: ignore[<rule>]`; unused suppressions are errors.
"""

from typing import assert_type, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import phydrax as phx
import phydrax.typing as pt
from phydrax.equations import ChemicalComponentCatalog
from phydrax.precision import precision_dtype_name, ScalarPrecisionDType


class ComponentDim(pt.Dim, minimum=1):
    """Number of chemical components."""


def tensor_forms(
    masses: pt.Float64[ComponentDim],
    table: pt.HostFloat64[pt.AnyDim, ComponentDim],
    count: pt.Size[ComponentDim],
    names: pt.Identifiers[ComponentDim],
    key: pt.PRNGKey,
) -> None:
    assert_type(masses, jax.Array)
    assert_type(table, npt.NDArray[np.float64])
    assert_type(count, int)
    assert_type(names, tuple[str, ...])
    assert_type(key, jax.Array)


def boundaries() -> None:
    basis = pt.parse("dense", Literal["dense", "sparse"], "basis")
    assert_type(basis, Literal["dense", "sparse"])
    values = pt.as_array([1.0, 2.0], pt.Float64[ComponentDim], "values")
    assert_type(values, jax.Array)
    host = pt.as_host_array(values, pt.HostFloat64[ComponentDim], "host")
    assert_type(host, npt.NDArray[np.float64])
    assert_type(precision_dtype_name("float32"), ScalarPrecisionDType)


def constructors() -> None:
    catalog = ChemicalComponentCatalog(
        ("H2",), np.asarray((2.016,)), ("H",), np.asarray(((2,),))
    )
    assert_type(catalog, ChemicalComponentCatalog)
    assert_type(catalog.molar_masses, jax.Array)
    ChemicalComponentCatalog(("H2",))  # ty: ignore[missing-argument]
    ChemicalComponentCatalog(
        ("H2",),
        jnp.ones((1,)),
        ("H",),
        np.ones((1, 1), dtype=np.int32),
        weights=1,  # ty: ignore[unknown-argument]
    )
    phx.typing.validate(catalog)
