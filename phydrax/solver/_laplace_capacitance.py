#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..discretization import EntitySelection
from ..linalg import (
    DifferentiationPolicy,
    FailurePolicy,
    FGMRES,
    JacobiPreconditionerBuilder,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    PreconditioningPolicy,
    prepare as prepare_linear,
    solve,
)
from ..operators.integral.layer_potential import (
    LaplaceLayerPotential3D,
    LaplaceSingleLayerDP0AssemblyReport3D,
    LaplaceSingleLayerDP0Galerkin3D,
)


class LaplaceCapacitanceResult3D(StrictModule):
    """Unit-voltage conductor responses and their Maxwell capacitance matrix."""

    layer_density: Array
    capacitance: Array
    linear_results: tuple[LinearSolveResult, ...]
    potentials: tuple[LaplaceLayerPotential3D, ...]
    assembly_report: LaplaceSingleLayerDP0AssemblyReport3D
    permittivity: Array
    capacitance_reciprocity_defect: Array
    valid: Array
    conductor_names: tuple[str, ...] = eqx.field(static=True)
    conductor_selection_ids: tuple[str, ...] = eqx.field(static=True)

    @property
    def surface_charge_density(self) -> Array:
        return self.permittivity * self.layer_density


def _conductors(
    galerkin: LaplaceSingleLayerDP0Galerkin3D,
    conductors: Mapping[str, EntitySelection],
    /,
) -> tuple[tuple[str, ...], tuple[EntitySelection, ...], np.ndarray]:
    if not isinstance(conductors, Mapping):
        raise TypeError(
            "[conductor-selection] conductors must be a mapping of names to "
            "EntitySelection values."
        )
    items = tuple(
        sorted((str(name), selection) for name, selection in conductors.items())
    )
    if not items or any(not name for name, _ in items):
        raise ValueError(
            "[conductor-selection] conductors must contain one or more non-empty names."
        )
    names = tuple(name for name, _ in items)
    if len(set(names)) != len(names):
        raise ValueError("[conductor-selection] Conductor names must be unique.")
    selections = tuple(selection for _, selection in items)
    if not all(isinstance(selection, EntitySelection) for selection in selections):
        raise TypeError(
            "[conductor-selection] Every conductor value must be an EntitySelection."
        )
    masks = []
    for selection in selections:
        if selection.entity_set_id != galerkin.surface_entities.entity_set_id:
            raise ValueError(
                "[geometry] Conductor selection does not match the prepared surface."
            )
        mask = np.asarray(selection.mask, dtype=bool)
        if mask.shape != (galerkin.face_count,) or not np.any(mask):
            raise ValueError(
                "[conductor-selection] Every conductor must select at least one "
                "surface face."
            )
        masks.append(mask)
    matrix = np.stack(masks)
    ownership = np.sum(matrix, axis=0)
    if np.any(ownership != 1):
        raise ValueError(
            "[conductor-selection] Conductor selections must be disjoint and cover "
            "every surface face."
        )
    components = np.asarray(galerkin.face_component_ids, dtype=np.int32)
    for component in range(galerkin.component_count):
        component_faces = components == component
        owners = np.flatnonzero(np.any(matrix[:, component_faces], axis=1))
        if owners.size != 1 or not np.all(matrix[owners[0], component_faces]):
            raise ValueError(
                "[geometry] Each connected surface component must belong to exactly "
                "one conductor."
            )
    return names, selections, matrix


def _default_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        FGMRES(restart=30, stagnation_iterations=30),
        preconditioning=PreconditioningPolicy(JacobiPreconditionerBuilder()),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )


def solve_laplace_capacitance_3d(
    galerkin: LaplaceSingleLayerDP0Galerkin3D,
    conductors: Mapping[str, EntitySelection],
    /,
    *,
    permittivity: ArrayLike = 1.0,
    linear: LinearSolvePolicy | None = None,
) -> LaplaceCapacitanceResult3D:
    """Solve unit-voltage 3D conductor problems and integrate their charges."""
    if not isinstance(galerkin, LaplaceSingleLayerDP0Galerkin3D):
        raise TypeError("galerkin must be LaplaceSingleLayerDP0Galerkin3D.")
    if not bool(galerkin.assembly_report.accuracy_supported):
        raise ValueError(
            "[quadrature] Galerkin assembly does not support the requested accuracy."
        )
    names, selections, masks_host = _conductors(galerkin, conductors)
    epsilon = jnp.asarray(permittivity, dtype=galerkin.face_areas.dtype)
    if epsilon.shape != () or not bool(jnp.isfinite(epsilon) & (epsilon > 0.0)):
        raise ValueError(
            "[permittivity] permittivity must be one finite positive scalar."
        )
    policy = _default_policy() if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be LinearSolvePolicy or None.")
    if policy.differentiation.mode != "none":
        raise ValueError(
            "[differentiation] Laplace capacitance solves currently require "
            "differentiation mode 'none'."
        )

    problem = LinearSystem(
        galerkin.strong_operator,
        problem_id="laplace-single-layer-capacitance-3d",
    )
    masks = jnp.asarray(masks_host, dtype=galerkin.face_areas.dtype)
    prepared_linear = prepare_linear(problem, policy)
    linear_results = tuple(
        solve(prepared_linear, masks[index]) for index in range(len(names))
    )
    layer_density = jnp.stack(
        tuple(jnp.asarray(result.value) for result in linear_results),
        axis=1,
    )
    potentials = tuple(
        galerkin.potential(layer_density[:, index]) for index in range(len(names))
    )
    surface_charge = epsilon * layer_density
    capacitance = oe.contract(
        "if,f,fj->ij",
        masks,
        galerkin.face_areas,
        surface_charge,
        backend="jax",
    )
    scale = jnp.maximum(jnp.max(jnp.abs(capacitance)), jnp.finfo(capacitance.dtype).tiny)
    reciprocity = jnp.max(jnp.abs(capacitance - capacitance.T)) / scale
    linear_valid = jnp.all(
        jnp.stack(
            tuple(
                result.successful & result.diagnostics.finite for result in linear_results
            )
        )
    )
    finite = (
        jnp.all(jnp.isfinite(layer_density))
        & jnp.all(jnp.isfinite(surface_charge))
        & jnp.all(jnp.isfinite(capacitance))
        & jnp.isfinite(reciprocity)
    )
    valid = galerkin.assembly_report.accuracy_supported & linear_valid & finite
    return LaplaceCapacitanceResult3D(
        layer_density=layer_density,
        capacitance=capacitance,
        linear_results=linear_results,
        potentials=potentials,
        assembly_report=galerkin.assembly_report,
        permittivity=epsilon,
        capacitance_reciprocity_defect=reciprocity,
        valid=valid,
        conductor_names=names,
        conductor_selection_ids=tuple(selection.selection_id for selection in selections),
    )


__all__ = ["LaplaceCapacitanceResult3D", "solve_laplace_capacitance_3d"]
