# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Named additive IMEX tableaux against the classical additive RK order conditions.

The conditions are evaluated with host NumPy from the published matrices alone
(Kennedy and Carpenter, Appl. Numer. Math. 44, 2003, Sec. 3): every pairing of
the explicit (E) and implicit (I) weights, abscissae and matrices must satisfy
the one-part conditions up to the declared order.
"""

from __future__ import annotations

import numpy as np
import pytest

from phydrax.solver.advanced import additive_imex_tableau, AdditiveIMEXScheme


def _conditions(scheme: AdditiveIMEXScheme) -> dict[int, float]:
    tableau = additive_imex_tableau(scheme)
    matrices = {
        "E": np.asarray(tableau.explicit_matrix),
        "I": np.asarray(tableau.implicit_matrix),
    }
    weights = {
        "E": np.asarray(tableau.explicit_weights),
        "I": np.asarray(tableau.weights),
    }
    nodes = {name: np.sum(matrix, axis=1) for name, matrix in matrices.items()}
    defects = {1: 0.0, 2: 0.0, 3: 0.0}
    for x in "EI":
        defects[1] = max(defects[1], abs(np.sum(weights[x]) - 1.0))
        for y in "EI":
            defects[2] = max(defects[2], abs(weights[x] @ nodes[y] - 0.5))
            for z in "EI":
                defects[3] = max(
                    defects[3],
                    abs(weights[x] @ (nodes[y] * nodes[z]) - 1.0 / 3.0),
                    abs(weights[x] @ (matrices[y] @ nodes[z]) - 1.0 / 6.0),
                )
    return defects


@pytest.mark.parametrize(
    ("scheme", "order"),
    [
        pytest.param("forward-backward-euler", 1, id="forward-backward-euler"),
        pytest.param("ars-222", 2, id="ars-222"),
        pytest.param("ssp2-222", 2, id="ssp2-222"),
        pytest.param("ars-443", 3, id="ars-443"),
    ],
)
def test_named_imex_scheme_satisfies_exactly_its_declared_order(
    scheme: AdditiveIMEXScheme, order: int
) -> None:
    defects = _conditions(scheme)
    for satisfied in range(1, order + 1):
        assert defects[satisfied] < 1e-14, (scheme, satisfied, defects)
    if order < 3:
        assert defects[order + 1] > 1e-3, (scheme, defects)


@pytest.mark.parametrize("scheme", ["forward-backward-euler", "ars-222", "ars-443"])
def test_ars_family_is_stiffly_accurate_with_shared_stage_times(
    scheme: AdditiveIMEXScheme,
) -> None:
    tableau = additive_imex_tableau(scheme)
    explicit = np.asarray(tableau.explicit_matrix)
    implicit = np.asarray(tableau.implicit_matrix)
    assert tableau.stiffly_accurate
    assert tableau.implicit_parts[0] is None
    np.testing.assert_allclose(np.sum(explicit, axis=1), tableau.nodes, atol=1e-15)
    np.testing.assert_allclose(np.sum(implicit, axis=1), tableau.nodes, atol=1e-15)
    assert all(implicit[stage, stage] > 0 for stage in range(1, implicit.shape[0]))


def test_unknown_scheme_is_refused() -> None:
    with pytest.raises(ValueError, match="scheme"):
        additive_imex_tableau("ars-232")  # ty: ignore[invalid-argument-type]
