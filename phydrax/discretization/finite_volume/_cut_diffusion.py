#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Matrix-free scalar diffusion on mapped multivalued cut-component graphs."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    DiagonalPairing,
    FunctionLinearOperator,
    KernelCertificate,
    LinearSolvePolicy,
    LinearSubspace,
    LinearSystem,
    NullspacePolicy,
    OperatorProperties,
    prepare,
    ProjectedPCG,
    solve,
    TolerancePolicy,
)
from ..amr._cut_complex import MultivaluedCutCellComplex


class MultivaluedCutCellDiffusionPlan(StrictModule, NonTrainableState):
    """Conservative scalar graph Laplacian with component-wise nullspaces."""

    complex: MultivaluedCutCellComplex
    diffusivity: Array
    internal_owner: Array
    internal_neighbour: Array
    internal_conductance: Array
    boundary_owner: Array
    boundary_conductance: Array
    boundary_face_indices: Array
    dirichlet_boundary_mask: Array
    nullspace_components: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex_: MultivaluedCutCellComplex,
        diffusivity: ArrayLike,
        /,
        *,
        dirichlet_face_indices: Sequence[int] = (),
    ):
        if not isinstance(complex_, MultivaluedCutCellComplex):
            raise TypeError("Cut-cell diffusion requires MultivaluedCutCellComplex.")
        cell_count = complex_.component_count
        coefficient = np.asarray(diffusivity, dtype=float)
        if coefficient.shape == ():
            coefficient = np.full((cell_count,), float(coefficient), dtype=float)
        if coefficient.shape != (cell_count,) or np.any(
            ~np.isfinite(coefficient) | (coefficient <= 0.0)
        ):
            raise ValueError("Cut-cell diffusivity must be positive per component.")
        face_active = np.asarray(complex_.face_active, dtype=bool)
        owner = np.asarray(complex_.face_owner_components, dtype=np.int32)[face_active]
        neighbour = np.asarray(complex_.face_neighbour_components, dtype=np.int32)[
            face_active
        ]
        centers = np.asarray(complex_.component_centers, dtype=float)[:cell_count]
        face_centers = np.asarray(complex_.face_centers, dtype=float)[face_active]
        area = np.asarray(complex_.face_area_vectors, dtype=float)[face_active]
        measure = np.asarray(complex_.face_measures, dtype=float)[face_active]
        normal = area / measure[:, None]
        internal = neighbour >= 0
        internal_owner = owner[internal]
        internal_neighbour = neighbour[internal]
        center_displacement = centers[internal_neighbour] - centers[internal_owner]
        distance = np.abs(np.sum(center_displacement * normal[internal], axis=-1))
        tolerance = (
            256.0
            * np.finfo(float).eps
            * np.maximum(1.0, np.linalg.norm(center_displacement, axis=-1))
        )
        if np.any(~np.isfinite(distance) | (distance <= tolerance)):
            raise ValueError("Cut-cell diffusion requires positive normal distances.")
        left = coefficient[internal_owner]
        right = coefficient[internal_neighbour]
        harmonic = 2.0 * left * right / (left + right)
        internal_conductance = measure[internal] * harmonic / distance

        boundary = ~internal
        boundary_owner = owner[boundary]
        boundary_distance = 2.0 * np.abs(
            np.sum(
                (face_centers[boundary] - centers[boundary_owner]) * normal[boundary],
                axis=-1,
            )
        )
        boundary_scale = np.maximum(
            1.0,
            np.linalg.norm(face_centers[boundary] - centers[boundary_owner], axis=-1),
        )
        boundary_tolerance = 256.0 * np.finfo(float).eps * boundary_scale
        if np.any(
            ~np.isfinite(boundary_distance) | (boundary_distance <= boundary_tolerance)
        ):
            raise ValueError("Cut-cell boundary diffusion distances must be positive.")
        boundary_conductance = (
            measure[boundary] * coefficient[boundary_owner] / boundary_distance
        )
        boundary_global = np.flatnonzero(face_active)[boundary].astype(np.int32)
        dirichlet_ids = np.asarray(
            tuple(int(value) for value in dirichlet_face_indices), dtype=np.int32
        )
        if (
            dirichlet_ids.ndim != 1
            or np.any(dirichlet_ids < 0)
            or np.any(dirichlet_ids >= complex_.face_capacity)
            or np.unique(dirichlet_ids).size != dirichlet_ids.size
            or np.any(~np.asarray(complex_.face_active)[dirichlet_ids])
            or np.any(np.asarray(complex_.face_neighbour_components)[dirichlet_ids] >= 0)
        ):
            raise ValueError("Dirichlet cut faces must be unique active boundary faces.")
        dirichlet_mask = np.isin(boundary_global, dirichlet_ids)

        parent = list(range(cell_count))

        def root(index: int) -> int:
            value = int(index)
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        for left_cell, right_cell in zip(internal_owner, internal_neighbour, strict=True):
            left_root = root(int(left_cell))
            right_root = root(int(right_cell))
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)
        groups: dict[int, list[int]] = {}
        for cell in range(cell_count):
            groups.setdefault(root(cell), []).append(cell)
        anchored = {root(int(cell)) for cell in boundary_owner[dirichlet_mask]}
        nullspace = tuple(
            tuple(groups[key]) for key in sorted(groups) if key not in anchored
        )

        self.complex = complex_
        self.diffusivity = jnp.asarray(coefficient)
        self.internal_owner = jnp.asarray(internal_owner)
        self.internal_neighbour = jnp.asarray(internal_neighbour)
        self.internal_conductance = jnp.asarray(internal_conductance)
        self.boundary_owner = jnp.asarray(boundary_owner)
        self.boundary_conductance = jnp.asarray(boundary_conductance)
        self.boundary_face_indices = jnp.asarray(boundary_global)
        self.dirichlet_boundary_mask = jnp.asarray(dirichlet_mask)
        self.nullspace_components = nullspace
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multivalued-cut-cell-diffusion",
                "complex": complex_.topology_id,
                "geometry": complex_.geometry_id,
                "diffusivity": array_tree_fingerprint(coefficient),
                "internal_owner": array_tree_fingerprint(internal_owner),
                "internal_neighbour": array_tree_fingerprint(internal_neighbour),
                "internal_conductance": array_tree_fingerprint(internal_conductance),
                "boundary_owner": array_tree_fingerprint(boundary_owner),
                "boundary_conductance": array_tree_fingerprint(boundary_conductance),
                "dirichlet": array_tree_fingerprint(dirichlet_mask),
                "nullspace": nullspace,
            }
        )

    @property
    def cell_count(self) -> int:
        return self.complex.component_count

    @property
    def volumes(self) -> Array:
        return self.complex.component_volumes[: self.cell_count]

    def apply(self, values: ArrayLike, /, *, shift: ArrayLike = 0.0) -> Array:
        value = jnp.asarray(values)
        shift_ = jnp.asarray(shift, dtype=value.dtype)
        if value.shape != (self.cell_count,) or shift_.shape != ():
            raise ValueError("Cut-cell diffusion values and shift have invalid shapes.")
        difference = value[self.internal_owner] - value[self.internal_neighbour]
        flux = self.internal_conductance.astype(value.dtype) * difference
        content = jnp.zeros_like(value)
        content = content.at[self.internal_owner].add(flux)
        content = content.at[self.internal_neighbour].add(-flux)
        boundary = (
            self.boundary_conductance.astype(value.dtype)
            * value[self.boundary_owner]
            * self.dirichlet_boundary_mask.astype(value.dtype)
        )
        content = content.at[self.boundary_owner].add(boundary)
        return content / self.volumes.astype(value.dtype) + shift_ * value

    def right_hand_side(
        self,
        source: ArrayLike,
        /,
        *,
        dirichlet_values: ArrayLike | None = None,
        outward_neumann_flux: ArrayLike | None = None,
    ) -> Array:
        source_ = jnp.asarray(source)
        if source_.shape != (self.cell_count,):
            raise ValueError("Cut-cell diffusion source must contain one value per cell.")
        boundary_count = int(self.boundary_owner.size)
        dirichlet = (
            jnp.zeros((boundary_count,), dtype=source_.dtype)
            if dirichlet_values is None
            else jnp.asarray(dirichlet_values, dtype=source_.dtype)
        )
        neumann = (
            jnp.zeros((boundary_count,), dtype=source_.dtype)
            if outward_neumann_flux is None
            else jnp.asarray(outward_neumann_flux, dtype=source_.dtype)
        )
        if dirichlet.shape != (boundary_count,) or neumann.shape != (boundary_count,):
            raise ValueError("Cut-cell boundary data must match boundary-face count.")
        dirichlet_content = (
            self.boundary_conductance.astype(source_.dtype)
            * dirichlet
            * self.dirichlet_boundary_mask.astype(source_.dtype)
        )
        face_measure = self.complex.face_measures[self.boundary_face_indices].astype(
            source_.dtype
        )
        neumann_content = (
            -face_measure
            * neumann
            * (~self.dirichlet_boundary_mask).astype(source_.dtype)
        )
        content = jnp.zeros_like(source_)
        content = content.at[self.boundary_owner].add(dirichlet_content + neumann_content)
        return source_ + content / self.volumes.astype(source_.dtype)

    def solve(
        self,
        right_hand_side: ArrayLike,
        /,
        *,
        shift: float = 0.0,
        initial_guess: ArrayLike | None = None,
        relative_tolerance: float = 1.0e-9,
        absolute_tolerance: float = 1.0e-11,
        maximum_iterations: int = 1000,
    ):
        rhs = jnp.asarray(right_hand_side)
        if rhs.shape != (self.cell_count,):
            raise ValueError("Cut-cell diffusion right-hand side has invalid shape.")
        shift_ = float(shift)
        if not np.isfinite(shift_) or shift_ < 0.0:
            raise ValueError("Cut-cell diffusion shift must be finite and nonnegative.")
        space = ArraySpace(
            (self.cell_count,),
            dtype=rhs.dtype,
            pairing=DiagonalPairing(self.volumes.astype(rhs.dtype)),
        )
        positive_definite = shift_ > 0.0 or not self.nullspace_components
        properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=positive_definite,
            positive_semidefinite=not positive_definite,
            evidence={
                "self_adjoint": "construction",
                (
                    "positive_definite" if positive_definite else "positive_semidefinite"
                ): "construction",
            },
        )
        operator = FunctionLinearOperator(
            lambda value: self.apply(value, shift=shift_),
            source=space,
            target=space,
            properties=properties,
            operator_id=canonical_fingerprint(
                {
                    "kind": "multivalued-cut-cell-diffusion-operator",
                    "plan": self.plan_id,
                    "shift": shift_,
                }
            ),
        )
        nullspace_policy = None
        method = None
        if not positive_definite:
            basis = jnp.zeros(
                (self.cell_count, len(self.nullspace_components)), dtype=rhs.dtype
            )
            for column, component in enumerate(self.nullspace_components):
                basis = basis.at[jnp.asarray(component), column].set(1.0)
            subspace = LinearSubspace(space, basis)
            certificate = KernelCertificate(
                operator,
                subspace,
                complete=True,
                evidence="construction",
            )
            nullspace_policy = NullspacePolicy(certificate=certificate)
            method = ProjectedPCG()
        problem = LinearSystem(operator, nullspace_policy=nullspace_policy)
        policy = LinearSolvePolicy(
            method,
            tolerance=TolerancePolicy(
                relative=float(relative_tolerance),
                absolute=float(absolute_tolerance),
                max_steps=int(maximum_iterations),
            ),
        )
        prepared = prepare(problem, policy)
        guess = (
            jnp.zeros_like(rhs) if initial_guess is None else jnp.asarray(initial_guess)
        )
        if guess.shape != rhs.shape:
            raise ValueError("Cut-cell diffusion initial guess has invalid shape.")
        return solve(prepared, rhs, initial_guess=guess)


__all__ = ["MultivaluedCutCellDiffusionPlan"]
