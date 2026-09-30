"""A relative Whitney one-form solve with its auxiliary scalar potential."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax.discretization import CellMesh
from phydrax.discretization.fem import FiniteElementDeRhamComplex
from phydrax.linalg import harmonic_subspace, hodge_laplacian
from phydrax.solver import HodgeLaplacePlan


def square_complex(n: int = 6) -> FiniteElementDeRhamComplex:
    coordinates = np.asarray(
        [(i / n, j / n) for j in range(n + 1) for i in range(n + 1)], dtype=np.float64
    )
    triangles: list[tuple[int, int, int]] = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b, c, d = a + 1, a + n + 1, a + n + 2
            triangles.extend(((a, b, d), (a, d, c)))
    return FiniteElementDeRhamComplex(
        CellMesh.from_triangles(coordinates, np.asarray(triangles, dtype=np.int32)),
        family="trimmed",
        order=1,
    )


def main() -> None:
    realization = square_complex()
    hilbert = realization.hilbert_complex(boundary="relative")
    harmonic = harmonic_subspace(hilbert, 1, expected_dimension=0)
    phi = jnp.sin(jnp.arange(hilbert.space(0).size, dtype=jnp.float64))
    expected = hilbert.differential(0).mv(phi)
    source = hodge_laplacian(hilbert, 1).mv(expected)
    plan = HodgeLaplacePlan(
        realization, 1, boundary="relative", formulation="mixed", harmonic=harmonic
    )
    result = plan.solve(source)
    if not bool(result.successful):
        raise RuntimeError(
            f"Native mixed solve failed: status={result.solve_result.status}"
        )
    error = jnp.sqrt(hilbert.space(1).inner(result.u - expected, result.u - expected))
    print(
        f"Whitney mixed Hodge solve: error={float(error):.3e}, residual={float(result.residual_norm):.3e}, harmonic={float(result.harmonic_defect):.3e}"
    )


if __name__ == "__main__":
    main()
