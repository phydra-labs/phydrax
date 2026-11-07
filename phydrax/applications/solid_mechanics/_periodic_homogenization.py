#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic cell problems: effective conductivity and linear elasticity.

Each cell problem solves for a periodic P1 fluctuation ``w`` on the vertex
orbits of a periodic tetrahedral quotient (`PeriodicSimplexSpace`) under unit
macroscopic loads, each a `PeriodicHomogenizationCase`:

* conductivity: ``div(k (G + grad w)) = 0`` for the three unit gradients ``G``,
  flux ``q = k (G + grad w)``, effective ``<q> = K G``;
* elasticity: ``div(C : (E + sym grad w)) = 0`` for the six unit Voigt strains
  ``E`` (engineering shear), stress ``sigma = C : eps``, effective
  ``<sigma> = C_eff E`` in Voigt form.

Averages ``<.>`` are over the whole periodic cell measure ``|det(vectors)|``.
Unmeshed space (the pores of a lattice region mesh) carries zero flux or
stress. The stiffness is assembled once on the fixed quotient pattern, factored
once by the supernodal sparse Cholesky of `phydrax.linalg`, and solved for all
cases as one multi-column right-hand side. The constant (rigid translation)
null space of each connected quotient component is fixed explicitly by pinning
its lowest orbit DOF to zero; effective tensors are gauge invariant.

Evidence per case is the Hill–Mandel macrohomogeneity audit
(`hill_mandel_evidence`): the cell-averaged microscopic power equals the
macroscopic power, and the periodic equilibrium residual of the root vanishes.
The effective tensor is computed twice, from averaged fluxes and from the
energy form ``<eps_I : C : eps_J>``; their agreement and the tensor symmetry are
reported. Cubic symmetry is measured (``cubic_symmetry_defect``), never imposed.

Every array path is JAX-traceable in the material, the representative
coordinates and the lattice vectors, so design sensitivities differentiate
through the factored operator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellMesh
from ...discretization.fem._periodic_cell import (
    PeriodicSimplexOperatorPattern,
    PeriodicSimplexSpace,
    simplex_gradients,
)
from ...ein import contract
from ...meshing._periodic_implicit_realization import PeriodicImplicitRealization
from ...typing import parse
from ._topology_design import (
    hill_mandel_evidence,
    HillMandelEvidence,
    PeriodicHomogenizationCase,
)


PeriodicCellPhysics = Literal["conductivity", "elasticity"]
MaterialSpec = ArrayLike | Mapping[str, ArrayLike]

_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_VOIGT_NAMES = ("xx", "yy", "zz", "yz", "xz", "xy")


def _unit_cases(
    physics: PeriodicCellPhysics, /
) -> tuple[PeriodicHomogenizationCase, ...]:
    if physics == "conductivity":
        return tuple(
            PeriodicHomogenizationCase(
                jnp.eye(3, dtype=jnp.float64)[axis], case_id=f"gradient-{name}"
            )
            for axis, name in enumerate("xyz")
        )
    cases = []
    for index, (row, column) in enumerate(_VOIGT_PAIRS):
        strain = np.zeros((3, 3), dtype=np.float64)
        # Unit engineering strain: a shear case has eps_ij = eps_ji = 1/2.
        value = 1.0 if row == column else 0.5
        strain[row, column] = value
        strain[column, row] = value
        cases.append(
            PeriodicHomogenizationCase(
                jnp.asarray(strain), case_id=f"strain-{_VOIGT_NAMES[index]}"
            )
        )
    return tuple(cases)


def _strain_operator(gradients: Array, /) -> Array:
    """Voigt engineering-strain matrices ``(cells, 6, 12)`` of P1 displacements."""
    gx, gy, gz = gradients[..., 0], gradients[..., 1], gradients[..., 2]
    zero = jnp.zeros_like(gx)
    rows = (
        (gx, zero, zero),
        (zero, gy, zero),
        (zero, zero, gz),
        (zero, gz, gy),
        (gz, zero, gx),
        (gy, gx, zero),
    )
    # (cells, 6, nodes, 3) ordered node-major as the operator pattern requires.
    blocks = jnp.stack([jnp.stack(row, axis=-1) for row in rows], axis=1)
    return blocks.reshape(gradients.shape[0], 6, 12)


def _voigt_tensor(voigt: Array, /, *, engineering: bool) -> Array:
    """Voigt vectors ``(..., 6)`` to symmetric tensors ``(..., 3, 3)``."""
    shear = 0.5 if engineering else 1.0
    xx, yy, zz, yz, xz, xy = (voigt[..., index] for index in range(6))
    return jnp.stack(
        (
            jnp.stack((xx, shear * xy, shear * xz), axis=-1),
            jnp.stack((shear * xy, yy, shear * yz), axis=-1),
            jnp.stack((shear * xz, shear * yz, zz), axis=-1),
        ),
        axis=-2,
    )


def isotropic_elastic_stiffness(young: ArrayLike, poisson: ArrayLike, /) -> Array:
    """Voigt (engineering shear) stiffness ``(..., 6, 6)`` of an isotropic solid."""
    modulus = jnp.asarray(young, dtype=jnp.float64)
    ratio = jnp.asarray(poisson, dtype=jnp.float64)
    lame = modulus * ratio / ((1.0 + ratio) * (1.0 - 2.0 * ratio))
    shear = modulus / (2.0 * (1.0 + ratio))
    normal = jnp.full((3, 3), 1.0)[..., :, :] * lame[..., None, None] + 2.0 * shear[
        ..., None, None
    ] * jnp.eye(3)
    zeros = jnp.zeros(modulus.shape + (3, 3), dtype=jnp.float64)
    lower = shear[..., None, None] * jnp.eye(3)
    return jnp.concatenate(
        (
            jnp.concatenate((normal, zeros), axis=-1),
            jnp.concatenate((zeros, lower), axis=-1),
        ),
        axis=-2,
    )


def _cubic_projection(tensor: Array, physics: PeriodicCellPhysics, /) -> Array:
    if physics == "conductivity":
        return jnp.trace(tensor) / 3.0 * jnp.eye(3, dtype=tensor.dtype)
    normal = jnp.mean(jnp.diagonal(tensor)[:3])
    coupling = jnp.mean(
        jnp.stack(
            (
                tensor[0, 1],
                tensor[0, 2],
                tensor[1, 2],
                tensor[1, 0],
                tensor[2, 0],
                tensor[2, 1],
            )
        )
    )
    shear = jnp.mean(jnp.diagonal(tensor)[3:])
    upper = coupling * jnp.ones((3, 3)) + (normal - coupling) * jnp.eye(3)
    zeros = jnp.zeros((3, 3), dtype=tensor.dtype)
    return jnp.block([[upper, zeros], [zeros, shear * jnp.eye(3)]])


@final
class PeriodicHomogenizationResult(StrictModule):
    """Effective tensor of one periodic cell with its per-case evidence.

    ``effective_tensor`` is the averaged-flux tensor: ``(3, 3)`` conductivity or
    ``(6, 6)`` Voigt stiffness (engineering shear), column ``J`` the cell
    average of the flux/stress of unit case ``J``. ``energy_tensor`` is the
    energy form ``<eps_I . C eps_J>``; ``energy_defect`` and ``symmetry_defect``
    are their Frobenius disagreement and asymmetry relative to the Voigt
    (volume-averaged material) tensor, which bounds the effective one.
    ``macroscopic_flux`` holds each case's averaged flux vector or stress
    tensor. ``fluctuation`` is ``(cases, dofs, components)`` with the gauge DOF
    of each component pinned at zero. ``volume_fraction`` is the meshed measure
    over the cell measure.
    """

    cases: tuple[PeriodicHomogenizationCase, ...]
    effective_tensor: Array
    energy_tensor: Array
    macroscopic_flux: Array
    fluctuation: Array
    hill_mandel: tuple[HillMandelEvidence, ...]
    solve_status: Array
    factorization_status: Array
    energy_defect: Array
    symmetry_defect: Array
    meshed_measure: Array
    cell_measure: Array
    physics: PeriodicCellPhysics = eqx.field(static=True)

    @property
    def volume_fraction(self) -> Array:
        return self.meshed_measure / self.cell_measure

    @property
    def cubic_symmetry_defect(self) -> Array:
        """Relative Frobenius distance to the cubic (isotropic for rank 2) form.

        Meaningful only when the lattice axes are the cubic axes; a measured
        consequence, never a constraint of the solve.
        """
        tensor = self.effective_tensor
        projected = _cubic_projection(tensor, self.physics)
        return jnp.linalg.norm(tensor - projected) / jnp.linalg.norm(tensor)

    @property
    def accepted(self) -> Array:
        """Every solve succeeded and every Hill–Mandel audit accepted."""
        audits = jnp.stack([evidence.accepted for evidence in self.hill_mandel])
        return (
            jnp.all(audits)
            & jnp.all(self.solve_status == 0)
            & jnp.all(jnp.isfinite(self.effective_tensor))
        )


@final
class PeriodicCellProblem(StrictModule, NonTrainableState):
    """Prepared conductivity or elasticity cell problem on one periodic quotient mesh.

    Preparation builds the `PeriodicSimplexSpace`, the fixed free-coordinate
    pattern and its symbolic Cholesky plan once; `solve` refreshes numeric
    values only. Elasticity refuses a quotient component that winds along fewer
    than two lattice directions: its periodic rigid rotations would be further
    null modes that pinning one DOF does not remove.
    """

    space: PeriodicSimplexSpace
    pattern: PeriodicSimplexOperatorPattern
    cases: tuple[PeriodicHomogenizationCase, ...]
    physics: PeriodicCellPhysics = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh | PeriodicSimplexSpace,
        /,
        *,
        physics: PeriodicCellPhysics,
    ) -> None:
        selected = parse(physics, PeriodicCellPhysics, "physics")
        space = (
            mesh if isinstance(mesh, PeriodicSimplexSpace) else PeriodicSimplexSpace(mesh)
        )
        if selected == "elasticity" and any(
            rank < 2 for rank in space.component_winding_ranks
        ):
            raise ValueError(
                "Periodic elasticity needs every quotient component to wind along at "
                "least two lattice directions; components winding along "
                f"{space.component_winding_ranks} admit periodic rigid rotations."
            )
        self.space = space
        self.pattern = PeriodicSimplexOperatorPattern(
            space, components=1 if selected == "conductivity" else 3
        )
        self.cases = _unit_cases(selected)
        self.physics = selected

    def _cell_material(self, material: MaterialSpec, /) -> Array:
        space = self.space
        if isinstance(material, Mapping):
            if set(material) != set(space.block_names):
                raise ValueError(
                    f"Block materials must name exactly the blocks {space.block_names}."
                )
            values = jnp.stack(
                [
                    jnp.asarray(material[name], dtype=jnp.float64)
                    for name in space.block_names
                ]
            )
            return self._cell_material(values[space.cell_blocks])
        value = jnp.asarray(material, dtype=jnp.float64)
        count = space.cell_count
        if self.physics == "conductivity":
            if value.ndim == 0 or value.shape == (count,):
                value = jnp.broadcast_to(value, (count,))[:, None, None] * jnp.eye(3)
            elif value.shape == (3, 3):
                value = jnp.broadcast_to(value, (count, 3, 3))
            elif value.shape != (count, 3, 3):
                raise ValueError(
                    "Conductivity must be a scalar, (cells,), (3, 3), (cells, 3, 3) or "
                    "a block mapping of those."
                )
            return value
        if value.shape == (6, 6):
            value = jnp.broadcast_to(value, (count, 6, 6))
        elif value.shape != (count, 6, 6):
            raise ValueError(
                "Elastic stiffness must be Voigt (6, 6), (cells, 6, 6) or a block mapping."
            )
        return value

    def solve(
        self,
        material: MaterialSpec,
        /,
        *,
        representative_coordinates: ArrayLike | None = None,
        lattice_vectors: ArrayLike | None = None,
    ) -> PeriodicHomogenizationResult:
        """Factor once and solve every unit case for the effective tensor.

        ``material`` is per-cell or per-block (a mapping from block name)
        conductivity or Voigt stiffness. ``representative_coordinates``
        ``(dofs, 3)`` and ``lattice_vectors`` ``(3, 3)`` default to the mesh;
        design realizations pass moved values with the same topology.
        """
        space, pattern = self.space, self.pattern
        tensor = self._cell_material(material)
        corners = space.cell_coordinates(representative_coordinates, lattice_vectors)
        measure = space.cell_measure(lattice_vectors)
        volumes, gradients = simplex_gradients(corners)
        volumes = eqx.error_if(
            volumes,
            jnp.any(~(volumes > 0.0)),
            "Periodic cell problems need positively oriented cells.",
        )
        if self.physics == "conductivity":
            strain_operator = jnp.swapaxes(gradients, -1, -2)  # (cells, 3, 4)
            loads = jnp.stack([case.macroscopic_strain for case in self.cases])
        else:
            strain_operator = _strain_operator(gradients)  # (cells, 6, 12)
            loads = jnp.eye(6, dtype=jnp.float64)
        element_matrices = volumes[:, None, None] * contract(
            "cki,ckl,clj->cij", strain_operator, tensor, strain_operator
        )
        operator = pattern.operator(element_matrices)
        # f_e = -vol B^T D E for every unit case: (cases, cells, local).
        macro_flux = contract("ckl,nl->nck", tensor, loads)
        element_loads = -volumes[None, :, None] * contract(
            "cki,nck->nci", strain_operator, macro_flux
        )
        rhs = pattern.load(element_loads)
        factorization = pattern.factor(operator)
        solution = factorization.solve(rhs)
        free = solution.value
        residual = jax.vmap(operator.mv, in_axes=1, out_axes=1)(free) - rhs
        scale = pattern.load(jnp.abs(element_loads))
        relative_residual = residual / jnp.maximum(
            jnp.linalg.norm(scale, axis=0, keepdims=True), jnp.finfo(jnp.float64).tiny
        )
        fluctuation = pattern.expand(free)
        local = pattern.gather(fluctuation)  # (cases, cells, local)
        strains = loads[:, None, :] + contract("cki,nci->nck", strain_operator, local)
        fluxes = contract("ckl,ncl->nck", tensor, strains)
        meshed = jnp.sum(volumes)
        average = contract("c,nck->nk", volumes, fluxes) / measure
        energy = contract("c,ick,jck->ij", volumes, strains, fluxes) / measure
        effective = average.T
        void = jnp.maximum(measure - meshed, measure * jnp.finfo(jnp.float64).eps)
        weights = jnp.concatenate((volumes, void[None]))
        audits = []
        for index, case in enumerate(self.cases):
            if self.physics == "conductivity":
                micro_flux, micro_strain = fluxes[index], strains[index]
                macro_flux_case, macro_strain = average[index], case.macroscopic_strain
            else:
                micro_flux = _voigt_tensor(fluxes[index], engineering=False)
                micro_strain = _voigt_tensor(strains[index], engineering=True)
                macro_flux_case = _voigt_tensor(average[index], engineering=False)
                macro_strain = case.macroscopic_strain
            # The unmeshed complement carries zero flux and therefore zero power.
            pad = jnp.zeros((1,) + micro_flux.shape[1:], dtype=micro_flux.dtype)
            audits.append(
                hill_mandel_evidence(
                    jnp.concatenate((micro_flux, pad)),
                    jnp.concatenate((micro_strain, pad)),
                    weights,
                    macro_flux_case,
                    macro_strain,
                    relative_residual[:, index],
                )
            )
        norm = jnp.linalg.norm(contract("c,ckl->kl", volumes, tensor) / measure)
        macroscopic = (
            average
            if self.physics == "conductivity"
            else _voigt_tensor(average, engineering=False)
        )
        return PeriodicHomogenizationResult(
            cases=self.cases,
            effective_tensor=effective,
            energy_tensor=energy,
            macroscopic_flux=macroscopic,
            fluctuation=fluctuation,
            hill_mandel=tuple(audits),
            solve_status=solution.status,
            factorization_status=factorization.status,
            energy_defect=jnp.linalg.norm(effective - energy) / norm,
            symmetry_defect=jnp.linalg.norm(effective - effective.T) / norm,
            meshed_measure=meshed,
            cell_measure=measure,
            physics=self.physics,
        )

    def solve_realization(
        self, material: MaterialSpec, realization: PeriodicImplicitRealization, /
    ) -> PeriodicHomogenizationResult:
        """Solve on a fixed-topology design realization of this problem's mesh.

        Derivatives with respect to the design state flow through the realized
        coordinates and lattice; a realization with a topology event carries
        poisoned (NaN) derivatives, so its sensitivities are never consumed.
        """
        if not isinstance(realization, PeriodicImplicitRealization):
            raise TypeError("realization must be a PeriodicImplicitRealization.")
        if realization.space_id != self.space.space_id:
            raise ValueError("The realization belongs to a different periodic mesh.")
        return self.solve(
            material,
            representative_coordinates=realization.representative_coordinates,
            lattice_vectors=realization.lattice_vectors,
        )


__all__ = [
    "isotropic_elastic_stiffness",
    "PeriodicCellPhysics",
    "PeriodicCellProblem",
    "PeriodicHomogenizationResult",
]
