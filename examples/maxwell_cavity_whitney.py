"""PEC unit-cube Whitney cavity, with gradient and harmonic kernels excluded."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from phydrax.discretization import CellMesh
from phydrax.discretization.fem import FiniteElementDeRhamComplex
from phydrax.solver import maxwell_cavity_modes


def cube_complex(n: int = 3) -> FiniteElementDeRhamComplex:
    coordinates = np.asarray(
        [
            (i / n, j / n, k / n)
            for k in range(n + 1)
            for j in range(n + 1)
            for i in range(n + 1)
        ],
        dtype=np.float64,
    )

    def vertex(i: int, j: int, k: int) -> int:
        return (k * (n + 1) + j) * (n + 1) + i

    tetrahedra: list[tuple[int, int, int, int]] = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                for axes in permutations(range(3)):
                    position = [i, j, k]
                    nodes = [vertex(*position)]
                    for axis in axes:
                        position[axis] += 1
                        nodes.append(vertex(*position))
                    inversions = sum(
                        axes[first] > axes[second]
                        for first in range(3)
                        for second in range(first + 1, 3)
                    )
                    if inversions % 2:
                        nodes[0], nodes[1] = nodes[1], nodes[0]
                    tetrahedra.append((nodes[0], nodes[1], nodes[2], nodes[3]))
    return FiniteElementDeRhamComplex(
        CellMesh.from_tetrahedra(coordinates, np.asarray(tetrahedra, dtype=np.int32)),
        family="trimmed",
        order=1,
    )


def main() -> None:
    realization = cube_complex()
    result = maxwell_cavity_modes(realization, count=3)
    if not bool(result.successful):
        raise RuntimeError(f"Native cavity eigensolve failed: status={result.status}")
    reference = 2.0 * np.pi**2
    values = np.asarray(result.eigenvalues)
    print(f"Lowest squared frequencies: {values}")
    print(
        f"Continuum first cluster: {reference:.8f}; relative errors: {np.abs(values / reference - 1.0)}"
    )
    print(f"Generalized residuals: {np.asarray(result.relative_residuals)}")


if __name__ == "__main__":
    main()
