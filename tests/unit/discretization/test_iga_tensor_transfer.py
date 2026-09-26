#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._identity import NumericRevision
from phydrax.discretization._spaces import DiscreteFieldSpace, TensorDofLayout
from phydrax.discretization.iga import BSplineGrid
from phydrax.discretization.iga._basis import TensorSplineBasisSpec
from phydrax.discretization.iga._transfer import prepare_tensor_transfer
from phydrax.linalg import ArraySpace


def _field(basis: TensorSplineBasisSpec) -> DiscreteFieldSpace:
    return DiscreteFieldSpace(
        "u",
        "iga-transfer-support",
        TensorDofLayout(basis.axis_names, basis.control_shape, layout_id=basis.layout_id),
        ArraySpace(basis.control_shape, dtype=jnp.float64),
        representation="basis_coefficient",
        conformity="H1",
    )


@pytest.mark.parametrize(
    "target_knots, target_degree",
    [
        ((0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0), 2),
        ((0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0), 3),
    ],
    ids=["knot-refinement", "degree-elevation"],
)
def test_exact_tensor_transfer_reproduces_linear_spline(
    target_knots: Any, target_degree: Any
) -> None:
    source_grid = BSplineGrid(jnp.asarray((0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0)), 2)
    target_grid = BSplineGrid(jnp.asarray(target_knots), target_degree)
    source_basis = TensorSplineBasisSpec((source_grid,))
    target_basis = TensorSplineBasisSpec((target_grid,))
    plan = prepare_tensor_transfer(
        source_basis,
        target_basis,
        _field(source_basis),
        _field(target_basis),
        source_plan_id="coarse",
        target_plan_id="fine",
        source_revision=NumericRevision("coarse", {"knots": source_grid.knots}),
        target_revision=NumericRevision("fine", {"knots": target_grid.knots}),
        transfer_class="exact",
    )

    # Greville coefficients represent f(x) = x in every B-spline space.
    source_linear = jnp.asarray(source_grid.greville_abscissae)
    target_linear = np.asarray(target_grid.greville_abscissae)
    np.testing.assert_allclose(plan.apply(source_linear), target_linear, atol=1e-12)
    payload = jnp.stack((jnp.ones_like(source_linear), source_linear), axis=-1)
    np.testing.assert_allclose(
        plan.apply_payload(payload),
        np.stack((np.ones_like(target_linear), target_linear), axis=-1),
        atol=1e-12,
    )
    assert plan.evidence.transfer_class == "exact"
    assert plan.field_transfer.properties.constant_preserving is True
    assert plan.evidence.pointwise_residual < 1e-12
