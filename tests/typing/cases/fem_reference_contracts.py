from __future__ import annotations

from typing import assert_type

import jax.numpy as jnp
from jax import Array

from phydrax.discretization import FiniteElementSpec, lagrange_element
from phydrax.discretization.fem import ReferenceNodalFamily
from phydrax.discretization.fem._form_basis_types import AbstractReferenceFormBasis
from phydrax.discretization.fem._reference_types import (
    AbstractFiniteElementTabulationSource,
    FiniteElementTabulator,
)


family = ReferenceNodalFamily("quadrilateral", 2)
source: AbstractFiniteElementTabulationSource = family
points = jnp.asarray(((0.2, 0.3),), dtype=jnp.float64)
assert_type(source.tabulate(points), tuple[Array, Array])
tabulator = FiniteElementTabulator(source)
assert_type(tabulator.source, AbstractFiniteElementTabulationSource)
element = lagrange_element("triangle", 1)
assert_type(element.form_basis, AbstractReferenceFormBasis | None)
assert_type(FiniteElementSpec.from_stored_fields(element), FiniteElementSpec)
FiniteElementTabulator(object())  # ty: ignore[invalid-argument-type]
