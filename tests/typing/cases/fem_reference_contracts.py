from __future__ import annotations

from typing import assert_type

import jax.numpy as jnp
from jax import Array

from phydrax.discretization import FiniteElementSpec, lagrange_element
from phydrax.discretization.fem import ReferenceNodalFamily
from phydrax.discretization import CoordinateEnclosureBudget, FiniteElementDiscretization
from phydrax.discretization._local_variational import PreparedLocalRegion
from phydrax.equations import (
    compile_finite_element_problem,
    CompiledFiniteElementProblem,
    FiniteElementForm,
)
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


def explicit_preparation_budget(
    form: FiniteElementForm,
    space: FiniteElementDiscretization,
    budget: CoordinateEnclosureBudget,
) -> None:
    assert_type(
        compile_finite_element_problem(form, space, budget=budget),
        CompiledFiniteElementProblem,
    )
    assert_type(
        CompiledFiniteElementProblem(form, space, budget=budget),
        CompiledFiniteElementProblem,
    )
    assert_type(
        space.prepare_local_regions(
            space.cell_domain, field_names=("u",), maximum_derivative_order=1,
            kernel_mode="dense", budget=budget,
        ),
        tuple[PreparedLocalRegion, ...],
    )
