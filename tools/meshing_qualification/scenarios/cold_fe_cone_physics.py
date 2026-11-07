"""Operation-scoped native FE programs for independent physical cone banks."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._strict import StrictModule
from phydrax.applications.solid_mechanics import (
    FiniteElementDynamicsExecutionContext,
    FiniteElementDynamicsState,
    FiniteStrainJ2MPMConstitutivePlan,
    FiniteStrainJ2Parameters,
)
from phydrax.equations.fem import FiniteElementAuxiliaryEvaluation


class _ConePlasticity(StrictModule):
    material: Any
    parameters: Any
    basis_gradients: Any
    cell_dofs: Any
    site_keys: tuple[str, ...] = eqx.field(static=True)

    def response(self, deformation: Any, dynamics: Any) -> Any:
        if not isinstance(dynamics, FiniteElementDynamicsExecutionContext):
            raise TypeError("Cone plasticity requires a dynamics execution context.")
        if dynamics.materials is None:
            raise ValueError("Cone plasticity requires committed material sites.")
        states = {item.site_id.key: item for item in dynamics.materials.states}
        if set(states) != set(self.site_keys):
            raise ValueError(
                "Cone material sites do not match the FE cell quadrature sites."
            )
        history = jnp.stack(tuple(states[key].committed for key in self.site_keys))[
            :, None, :
        ]
        return self.material.evaluate(
            deformation,
            history,
            jnp.ones(deformation.shape[:-2], dtype=deformation.dtype),
            self.parameters,
            dynamics.end_time,
            dynamics.step_size,
        )

    def residual(
        self,
        values: Any,
        gradients: Any,
        points: Any,
        weights: Any,
        test_basis: Any,
        test_gradients: Any,
        context: Any,
    ) -> Any:
        del values, points, test_basis
        # Affine P1 gradients are constant, so centroid history supplies the
        # stress at every assembly quadrature point in this cell.
        deformation = (
            jnp.eye(3, dtype=gradients[0].dtype)[None, None] + gradients[0][:, :1]
        )
        response = self.response(deformation, context.user_args)
        return phx.ein.contract(
            "cq,cqib,cqab->cia",
            weights,
            test_gradients,
            jnp.broadcast_to(response.first_piola, weights.shape + (3, 3)),
        )

    def evaluate_displacement(self, displacement: Any, dynamics: Any) -> Any:
        deformation = jnp.eye(3, dtype=displacement.dtype)[None, None] + jnp.einsum(
            "cia,cqib->cqab", displacement[self.cell_dofs], self.basis_gradients[:, :1]
        )
        return self.response(deformation, dynamics)

    def trials(self, materials: Any, response: Any) -> Any:
        return materials.with_trials(
            {
                key: response.trial_state[index, 0]
                for index, key in enumerate(self.site_keys)
            }
        )

    def auxiliary(self, displacement: Any, context: Any) -> Any:
        dynamics = context.user_args
        response = self.evaluate_displacement(displacement, dynamics)
        return FiniteElementAuxiliaryEvaluation(
            self.trials(dynamics.materials, response),
            successful=jnp.all(response.successful),
            admissible=jnp.all(response.admissible),
        )

    def update(
        self,
        displacement: Any,
        velocity: Any,
        acceleration: Any,
        time: Any,
        step_size: Any,
        previous: Any,
        args: Any,
    ) -> Any:
        del velocity, acceleration
        if previous is None:
            raise ValueError("Cone plasticity lost its committed material.")
        dynamics = FiniteElementDynamicsExecutionContext(
            previous, time - step_size, time, step_size, args
        )
        return self.trials(previous, self.evaluate_displacement(displacement, dynamics))


class ConePhysicalWorkset:
    """Reusable programs owned by one qualification operation, never global."""

    def __init__(self) -> None:
        self._programs: dict[tuple[Any, ...], tuple[Any, ...]] = {}

    def advance(
        self,
        vertices: Any,
        displacement: Any,
        velocity: Any,
        acceleration: Any,
        material_state: Any,
        time: float,
        step: int,
        state_version: int,
        step_size: float,
        /,
        *,
        pending_increment: Any = None,
    ) -> Any:
        single = np.asarray(vertices).ndim == 2
        points = np.asarray(vertices, dtype=np.float64)
        if single:
            points = points[None, ...]
            material_state = (material_state,)
            displacement, velocity, acceleration = (
                np.asarray(value)[None, ...]
                for value in (displacement, velocity, acceleration)
            )
            if pending_increment is not None:
                pending_increment = np.asarray(pending_increment)[None, ...]
        states = tuple(material_state)
        if (
            points.ndim != 3
            or points.shape[1:] != (4, 3)
            or len(states) != len(points)
            or not len(points)
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(
                "Cone dynamics requires a nonempty bank of finite tetrahedra and material sites."
            )
        orientations = np.linalg.det(
            np.transpose(points[:, 1:] - points[:, :1], (0, 2, 1))
        )
        if np.any(~np.isfinite(orientations)) or np.any(orientations == 0):
            raise ValueError("Cone dynamics requires nondegenerate tetrahedra.")
        order = np.tile(np.arange(4), (len(points), 1))
        order[orientations < 0] = (0, 2, 1, 3)
        inverse_order = np.argsort(order, axis=1)
        cells = np.asarray([state.site_id.cell_id for state in states], dtype=np.int64)
        expected = tuple(
            phx.equations.MaterialSiteId.for_cell_quadrature(
                int(cell),
                "centroid",
                domain_id="periodic-cone",
                quadrature_rule_id="tetra-centroid",
            ).key
            for cell in cells
        )
        if (
            len(set(map(int, cells))) != len(cells)
            or tuple(s.site_id.key for s in states) != expected
        ):
            raise ValueError(
                "Cone material identities must identify each actual cell centroid exactly once."
            )
        points = np.take_along_axis(points, order[..., None], axis=1)
        key = (tuple(expected), tuple(s.layout_id for s in states), points.tobytes())
        materials = phx.equations.MaterialTransaction(states)
        program = self._programs.get(key)
        if program is None:
            mesh = phx.discretization.CellMesh.from_tetrahedra(
                jnp.asarray(points.reshape(-1, 3)),
                jnp.asarray(np.arange(4 * len(points), dtype=np.int32).reshape(-1, 4)),
                vertex_global_ids=np.arange(4 * len(points), dtype=np.int64),
                cell_global_ids=cells,
            )
            if not np.array_equal(np.asarray(mesh.blocks[0].global_ids), cells):
                raise ValueError(
                    "Cone FE cell rows no longer match global material-site identities."
                )
            discretization = phx.discretization.FiniteElementPlan(
                mesh,
                phx.discretization.FiniteElementFieldSpec(
                    "u",
                    phx.discretization.lagrange_element("tetrahedron", 1),
                    component_shape=(3,),
                ),
            ).prepare()
            gradients = discretization.block_geometries[0][0].physical_gradients
            dofs = discretization.dof_maps[0].cell_dofs[0]
            if gradients.shape[1] < 1 or gradients.shape[0] != len(cells):
                raise ValueError(
                    "Cone FE geometry must provide quadrature gradients for each cell."
                )
            plasticity = _ConePlasticity(
                FiniteStrainJ2MPMConstitutivePlan(),
                FiniteStrainJ2Parameters(0.2, 0.8, 0.002, 0.02),
                gradients,
                dofs,
                expected,
            )
            form = phx.equations.FiniteElementForm(
                "periodic-cone-j2",
                "u",
                (
                    phx.equations.CellResidualAction(
                        "u", ("u",), plasticity.residual, action_id="cone-j2-stress"
                    ),
                ),
                auxiliary_evaluator=plasticity.auxiliary,
                auxiliary_id="cone-j2-trial",
            )
            compiled = phx.equations.compile_finite_element_problem(form, discretization)
            program = (compiled, plasticity, dofs)
            self._programs[key] = program
        compiled, plasticity, dofs = program[:3]

        def to_dofs(value: Any) -> Any:
            value = np.asarray(value)
            if value.shape != (len(points), 4, 3):
                raise ValueError("Cone kinematics must have shape (cells, 4, 3).")
            ordered = np.take_along_axis(value, order[..., None], axis=1)
            return (
                jnp.zeros((4 * len(points), 3), dtype=jnp.asarray(ordered).dtype)
                .at[dofs]
                .set(jnp.asarray(ordered))
            )

        state = FiniteElementDynamicsState(
            to_dofs(displacement),
            to_dofs(velocity),
            to_dofs(acceleration),
            materials=materials,
            time=time,
            step=step,
            state_version=state_version,
        )
        if len(program) == 3:
            mechanics = phx.applications.solid_mechanics
            plan = mechanics.prepare_finite_element_dynamics(
                compiled,
                state,
                step_size,
                nonlinear_method=phx.nonlinear.NewtonKrylov(
                    linear_policy=phx.linalg.LinearSolvePolicy(
                        phx.linalg.GMRES(restart=16),
                        tolerance=phx.linalg.TolerancePolicy(
                            relative=1e-8, absolute=1e-14, max_steps=64
                        ),
                    )
                ),
                nonlinear_termination=phx.nonlinear.NonlinearTermination(
                    absolute_residual=1e-11, relative_residual=1e-11, maximum_steps=30
                ),
                material_update=plasticity.update,
                material_update_id="cone-j2-trial",
            )
            program = (compiled, plasticity, dofs, plan)
            self._programs[key] = program
        plan = program[3]
        mechanics = phx.applications.solid_mechanics
        result = mechanics.solve_finite_element_dynamics_step(
            mechanics.prepare_finite_element_dynamics_step(plan, state, step_size)
        )
        if not bool(result.accepted):
            raise RuntimeError(
                f"Native cone Newmark/J2 step was rejected: status={int(result.nonlinear.status)}, residual={float(result.nonlinear.diagnostics.final_residual_norm)}."
            )
        accepted = result.promote()

        def from_dofs(value: Any) -> Any:
            result = jnp.take_along_axis(
                value[dofs], jnp.asarray(inverse_order)[..., None], axis=1
            )
            return result[0] if single else result

        output = FiniteElementDynamicsState(
            from_dofs(accepted.displacement),
            from_dofs(accepted.velocity),
            from_dofs(accepted.acceleration),
            materials=accepted.materials,
            time=accepted.time,
            step=accepted.step,
            state_version=accepted.state_version,
        )
        if pending_increment is None:
            return output
        dynamics = FiniteElementDynamicsExecutionContext(
            accepted.materials,
            accepted.time - step_size,
            accepted.time,
            jnp.asarray(step_size),
            None,
        )
        response = plasticity.evaluate_displacement(
            accepted.displacement + to_dofs(pending_increment), dynamics
        )
        if not bool(jnp.all(response.successful)) or not bool(
            jnp.all(response.admissible)
        ):
            raise RuntimeError("Cone pending J2 trial is not admissible.")
        trials = plasticity.trials(accepted.materials, response)
        return output, trials.state(expected[0]).trial if single else trials
