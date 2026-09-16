#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite SU(2) recoupling and BF/Ponzano–Regge identity controls."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...tensor_network import (
    su2_clebsch_gordan,
    su2_fusion,
    su2_pentagon_residual,
    su2_recoupling_matrix,
    su2_wigner_6j,
)


class SU2BFIdentityEvidence(StrictModule):
    maximum_clebsch_orthogonality_residual: Array
    maximum_recoupling_unitarity_residual: Array
    maximum_tetrahedral_symmetry_residual: Array
    maximum_pentagon_residual: Array
    finite: Array
    accepted: Array
    maximum_twice_spin: int = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def assess_su2_bf_identities(
    maximum_twice_spin: int = 3,
    /,
    *,
    tolerance: float = 1e-11,
) -> SU2BFIdentityEvidence:
    maximum = int(maximum_twice_spin)
    if maximum < 1 or not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("SU2 BF identity range/tolerance is invalid.")
    clebsch_residual = 0.0
    recoupling_residual = 0.0
    symmetry_residual = 0.0
    pentagon_residual = 0.0
    for left in range(maximum + 1):
        for right in range(maximum + 1):
            for output in su2_fusion(left, right):
                if output > 2 * maximum:
                    continue
                table = np.asarray(su2_clebsch_gordan(left, right, output))
                columns = table.reshape((-1, output + 1))
                gram = columns.T @ columns
                clebsch_residual = max(
                    clebsch_residual,
                    float(np.linalg.norm(gram - np.eye(output + 1))),
                )
    for first in range(maximum + 1):
        for second in range(maximum + 1):
            for third in range(maximum + 1):
                reachable = {
                    total
                    for intermediate in su2_fusion(first, second)
                    for total in su2_fusion(intermediate, third)
                }
                for total in reachable:
                    left_channels, right_channels, matrix = su2_recoupling_matrix(
                        first, second, third, total
                    )
                    if not left_channels or not right_channels:
                        continue
                    matrix_host = np.asarray(matrix)
                    identity = np.eye(matrix_host.shape[0])
                    recoupling_residual = max(
                        recoupling_residual,
                        float(np.linalg.norm(matrix_host @ matrix_host.T - identity)),
                    )
    candidates = (
        (1, 1, 0, 1, 1, 0),
        (1, 1, 2, 1, 1, 2),
        (2, 2, 2, 2, 2, 2),
        (1, 2, 1, 2, 1, 2),
    )
    for a, b, c, d, e, f in candidates:
        value = su2_wigner_6j(a, b, c, d, e, f)
        swap_columns = su2_wigner_6j(b, a, c, e, d, f)
        exchange_rows = su2_wigner_6j(d, e, c, a, b, f)
        symmetry_residual = max(
            symmetry_residual,
            abs(value - swap_columns),
            abs(value - exchange_rows),
        )
    pentagon_cases = (
        (1, 1, 1, 1, 0),
        (1, 1, 1, 1, 2),
        (2, 1, 1, 2, 0),
        (2, 2, 2, 2, 2),
    )
    for case in pentagon_cases:
        pentagon_residual = max(pentagon_residual, float(su2_pentagon_residual(*case)))
    values = jnp.asarray(
        (
            clebsch_residual,
            recoupling_residual,
            symmetry_residual,
            pentagon_residual,
        )
    )
    finite = jnp.all(jnp.isfinite(values))
    accepted = finite & jnp.all(values <= float(tolerance))
    return SU2BFIdentityEvidence(
        maximum_clebsch_orthogonality_residual=values[0],
        maximum_recoupling_unitarity_residual=values[1],
        maximum_tetrahedral_symmetry_residual=values[2],
        maximum_pentagon_residual=values[3],
        finite=finite,
        accepted=accepted,
        maximum_twice_spin=maximum,
        claim="finite-su2-bf-recoupling-and-ponzano-regge-identity-control",
    )


__all__ = ["SU2BFIdentityEvidence", "assess_su2_bf_identities"]
