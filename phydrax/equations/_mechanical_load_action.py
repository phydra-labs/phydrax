#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


if TYPE_CHECKING:
    from ..applications.solid_mechanics._loads import (
        AbstractMechanicalLoad,
        MechanicalLoadEvaluation,
        MechanicalLoadState,
    )
    from ..integration._deformed_measure import DeformedMeasurePlan, DeformedMeasureState
    from ..nn.parameters import ParameterSubspace
    from ..solver import PreparedFieldEquilibrium


NeuralCoordinateTrace = Callable[[Mapping[str, Any], Array, Any], ArrayLike]


class MechanicalLoadActionEvaluation(StrictModule):
    """Assembled external force/residual with dynamic measure evidence."""

    external_force: Array
    residual: Array
    load: Any
    measure: DeformedMeasureState
    valid: Array
    action_id: str = eqx.field(static=True)


class MechanicalLoadAction(StrictModule, NonTrainableState):
    """Prepared finite-element action for a state-dependent mechanical load.

    Reference gradients are physical gradients with respect to reference
    coordinates. The action reconstructs ``F = dx/dX`` at quadrature points,
    evaluates the deformed measure inside differentiation, and scatters external
    forces through the exact transpose of the interpolation. Its tangent is the
    Jacobian of the unsymmetrized residual; follower contributions are never
    projected onto a self-adjoint operator.
    """

    load: Any
    reference_coordinates: Array
    gathers: Array
    basis_values: Array
    reference_gradients: Array
    measure_plan: DeformedMeasurePlan
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        load: AbstractMechanicalLoad,
        reference_coordinates: ArrayLike,
        gathers: ArrayLike,
        basis_values: ArrayLike,
        reference_gradients: ArrayLike,
        measure_plan: DeformedMeasurePlan,
        /,
        *,
        action_id: str | None = None,
    ):
        from ..applications.solid_mechanics._loads import AbstractMechanicalLoad
        from ..integration._deformed_measure import DeformedMeasurePlan

        if not isinstance(load, AbstractMechanicalLoad):
            raise TypeError("load must implement AbstractMechanicalLoad.")
        if not isinstance(measure_plan, DeformedMeasurePlan):
            raise TypeError("measure_plan must be a DeformedMeasurePlan.")
        reference = jnp.asarray(reference_coordinates)
        routes = jnp.asarray(gathers, dtype=jnp.int32)
        basis = jnp.asarray(basis_values)
        gradients = jnp.asarray(reference_gradients)
        if (
            not jnp.issubdtype(reference.dtype, jnp.inexact)
            or jnp.iscomplexobj(reference)
            or reference.ndim != 2
            or reference.shape[-1] not in (2, 3)
        ):
            raise ValueError("Reference coordinates must have shape (node, 2|3).")
        if routes.ndim != 2 or routes.shape[1] == 0:
            raise ValueError("Finite-element gathers must have shape (cell, local_dof).")
        routes_host = np.asarray(routes)
        if np.any(routes_host < 0) or np.any(routes_host >= reference.shape[0]):
            raise ValueError("Finite-element gathers contain an out-of-range node.")
        if basis.ndim == 2:
            if basis.shape[1] != routes.shape[1]:
                raise ValueError("Basis and gather local dimensions do not match.")
            basis = jnp.broadcast_to(basis, (routes.shape[0],) + basis.shape)
        elif basis.ndim == 3:
            if basis.shape[0] != routes.shape[0] or basis.shape[2] != routes.shape[1]:
                raise ValueError("Cellwise basis values do not match the gathers.")
        else:
            raise ValueError(
                "Basis values must have shape (point, local) or (cell, point, local)."
            )
        expected_gradients = (
            routes.shape[0],
            basis.shape[1],
            routes.shape[1],
            reference.shape[-1],
        )
        if gradients.shape != expected_gradients:
            raise ValueError(
                f"reference_gradients must have shape {expected_gradients}; got {gradients.shape}."
            )
        if (
            not jnp.issubdtype(basis.dtype, jnp.inexact)
            or not jnp.issubdtype(gradients.dtype, jnp.inexact)
            or jnp.iscomplexobj(basis)
            or jnp.iscomplexobj(gradients)
        ):
            raise TypeError(
                "Basis values and reference gradients must be real inexact arrays."
            )
        quadrature_shape = basis.shape[:2]
        try:
            jnp.broadcast_to(measure_plan.reference_measure, quadrature_shape)
        except ValueError as error:
            raise ValueError(
                "The deformed-measure plan does not match the action quadrature layout."
            ) from error
        if measure_plan.kind == "surface":
            assert measure_plan.reference_normal is not None
            try:
                jnp.broadcast_to(
                    measure_plan.reference_normal,
                    quadrature_shape + (reference.shape[-1],),
                )
            except ValueError as error:
                raise ValueError(
                    "Reference surface normals do not match the action quadrature layout."
                ) from error
        if load.semantics.support == "body" and measure_plan.kind != "volume":
            raise ValueError("A body load action requires a volume measure plan.")
        if load.semantics.support == "boundary" and measure_plan.kind != "surface":
            raise ValueError("A boundary load action requires a surface measure plan.")
        if load.semantics.support == "discrete":
            raise ValueError("Discrete loads do not use finite-element measure actions.")
        generated = canonical_fingerprint(
            {
                "kind": "mechanical-load-action",
                "load_id": load.load_id,
                "measure_plan_id": measure_plan.plan_id,
                "reference_coordinates": array_tree_fingerprint(reference),
                "gathers": array_tree_fingerprint(routes),
                "basis_values": array_tree_fingerprint(basis),
                "reference_gradients": array_tree_fingerprint(gradients),
            }
        )
        identifier = generated if action_id is None else str(action_id)
        if not identifier:
            raise ValueError("action_id must be non-empty.")
        self.load = load
        self.reference_coordinates = reference
        self.gathers = routes
        self.basis_values = basis
        self.reference_gradients = gradients
        self.measure_plan = measure_plan
        self.action_id = identifier

    def _quadrature_state(
        self,
        current_coordinates: ArrayLike,
        state: MechanicalLoadState,
        args: Any,
        /,
    ) -> tuple[MechanicalLoadEvaluation, DeformedMeasureState]:
        current = jnp.asarray(current_coordinates, dtype=self.reference_coordinates.dtype)
        if current.shape != self.reference_coordinates.shape:
            raise ValueError(
                "Current coordinates must preserve the reference node layout."
            )
        reference_local = self.reference_coordinates[self.gathers]
        current_local = current[self.gathers]
        reference_points = oe.contract("cqi,cia->cqa", self.basis_values, reference_local)
        current_points = oe.contract("cqi,cia->cqa", self.basis_values, current_local)
        deformation_gradient = oe.contract(
            "cqir,cis->cqsr", self.reference_gradients, current_local
        )
        measure = self.measure_plan.evaluate(deformation_gradient)
        evaluation = self.load.evaluate(
            reference_points,
            current_points,
            measure,
            state,
            args,
        )
        return evaluation, measure

    def evaluate(
        self,
        current_coordinates: ArrayLike,
        state: MechanicalLoadState,
        args: Any = None,
        /,
    ) -> MechanicalLoadActionEvaluation:
        evaluation, measure = self._quadrature_state(current_coordinates, state, args)
        weight = measure.measure(evaluation.semantics.measure_frame)
        local_force = oe.contract(
            "cqi,cq,cqa->cia",
            self.basis_values,
            weight,
            evaluation.total_force_density,
        )
        external = jnp.zeros_like(self.reference_coordinates)
        external = external.at[self.gathers.reshape((-1,))].add(
            local_force.reshape((-1, self.reference_coordinates.shape[-1]))
        )
        residual = -external
        finite = jnp.all(jnp.isfinite(external)) & jnp.all(jnp.isfinite(residual))
        return MechanicalLoadActionEvaluation(
            external_force=external,
            residual=residual,
            load=evaluation,
            measure=measure,
            valid=evaluation.valid & finite,
            action_id=self.action_id,
        )

    def residual(
        self,
        current_coordinates: ArrayLike,
        state: MechanicalLoadState,
        args: Any = None,
        /,
    ) -> Array:
        """Return the external contribution under the total-residual sign convention."""
        return self.evaluate(current_coordinates, state, args).residual

    def tangent(
        self,
        current_coordinates: ArrayLike,
        state: MechanicalLoadState,
        args: Any = None,
        /,
    ) -> Array:
        """Differentiate the state-dependent residual without symmetrization."""
        current = jnp.asarray(current_coordinates, dtype=self.reference_coordinates.dtype)
        if current.shape != self.reference_coordinates.shape:
            raise ValueError(
                "Current coordinates must preserve the reference node layout."
            )
        size = current.size

        def flattened_residual(flattened):
            coordinates = flattened.reshape(current.shape)
            return self.residual(coordinates, state, args).reshape((size,))

        return jax.jacfwd(flattened_residual)(current.reshape((size,)))

    def potential(
        self,
        current_coordinates: ArrayLike,
        state: MechanicalLoadState,
        args: Any = None,
        /,
    ) -> Array:
        """Integrate only a load law carrying certified potential semantics."""
        if self.load.semantics.conservativity != "potential":
            raise ValueError("This load is nonconservative and must use virtual work.")
        evaluation, measure = self._quadrature_state(current_coordinates, state, args)
        if evaluation.potential_density is None:
            raise ValueError(
                "A certified mechanical load did not provide potential density."
            )
        weight = measure.measure(evaluation.semantics.measure_frame)
        return jnp.sum(weight * evaluation.potential_density)

    def prepare_neural_virtual_work(
        self,
        functions: Mapping[str, Any],
        coordinate_trace: NeuralCoordinateTrace,
        parameter_subspace: ParameterSubspace,
        state: MechanicalLoadState,
        /,
        *,
        trace_id: str,
        problem_id: str = "neural-mechanical-load",
    ) -> PreparedFieldEquilibrium:
        """Prepare the canonical parameter-space root from physical load virtual work."""
        from ..nn.parameters import ParameterSubspace
        from ..solver import prepare_virtual_work_equilibrium

        if not callable(coordinate_trace):
            raise TypeError("coordinate_trace must be callable.")
        if not isinstance(parameter_subspace, ParameterSubspace):
            raise TypeError("parameter_subspace must be a ParameterSubspace.")
        trace_identifier = str(trace_id)
        if not trace_identifier:
            raise ValueError("trace_id must be non-empty.")
        realization = _NeuralMechanicalLoadRealization(
            self,
            coordinate_trace,
            state,
            trace_identifier,
        )
        return prepare_virtual_work_equilibrium(
            functions,
            _mechanical_load_field_jet,
            _mechanical_load_virtual_work,
            parameter_subspace,
            realization,
            realization_id=self.action_id,
            provenance_id=trace_identifier,
            problem_id=problem_id,
        )


class _NeuralMechanicalLoadRealization(StrictModule, NonTrainableState):
    action: MechanicalLoadAction
    coordinate_trace: NeuralCoordinateTrace
    state: Any
    trace_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: MechanicalLoadAction,
        coordinate_trace: NeuralCoordinateTrace,
        state: MechanicalLoadState,
        trace_id: str,
        /,
    ):
        self.action = action
        self.coordinate_trace = coordinate_trace
        self.state = state
        self.trace_id = trace_id

    def field_jet(self, functions: Mapping[str, Any], args: Any, /) -> Array:
        coordinates = jnp.asarray(
            self.coordinate_trace(functions, self.action.reference_coordinates, args),
            dtype=self.action.reference_coordinates.dtype,
        )
        if coordinates.shape != self.action.reference_coordinates.shape:
            raise ValueError(
                "Neural coordinate trace changed the finite-element node layout."
            )
        return coordinates

    def virtual_work(
        self,
        functions: Mapping[str, Any],
        coordinates: Array,
        args: Any,
        /,
    ) -> Array:
        del functions
        return self.action.residual(coordinates, self.state, args)


def _mechanical_load_field_jet(
    functions: Mapping[str, Any],
    realization: _NeuralMechanicalLoadRealization,
    args: Any,
    /,
) -> PyTree[Array]:
    return realization.field_jet(functions, args)


def _mechanical_load_virtual_work(
    functions: Mapping[str, Any],
    jets: PyTree[Array],
    realization: _NeuralMechanicalLoadRealization,
    args: Any,
    /,
) -> PyTree[Array]:
    return realization.virtual_work(functions, jets, args)


__all__ = [
    "MechanicalLoadAction",
    "MechanicalLoadActionEvaluation",
    "NeuralCoordinateTrace",
]
