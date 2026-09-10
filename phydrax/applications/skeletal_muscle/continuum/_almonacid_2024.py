#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native idealized Almonacid 2024 muscle–aponeurosis continuum.

Executable identity: flexodeal 0698e3d, repository-default parameters, not the
SIAM material table. The sole force owner is the three-field weak residual.
Continuous Q2 displacement is tied at the muscle/aponeurosis interfaces; each
cell owns four total-degree-one monomial pressure and dilation coefficients.
No tendon, full-MTA, anatomical, or biological validation claim is made.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax._interpolation import linear_interpolate

from ...._fingerprint import canonical_fingerprint
from ...._identity import (
    ExecutableSignature,
    NumericRevision,
    SemanticProvenance,
    strict_module_payload,
)
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....ein import contract
from ....nonlinear import (
    implicit_root_result,
    NewtonKrylov,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ....operators.mechanics import finite_strain_kinematics
from ._almonacid_2024_geometry import (
    Almonacid2024Geometry,
    prepare_geometry,
    PreparedAlmonacid2024Geometry,
)
from ._almonacid_2024_material import (
    Almonacid2024MaterialParameters,
    almonacid_2024_material_response,
)
from ._almonacid_2024_preconditioning import (
    Almonacid2024SymbolicPreconditioning,
    almonacid_2024_linear_policy,
    almonacid_2024_prepare_symbolic,
    almonacid_2024_setup_operator,
)


_SOURCE_COMMIT = "0698e3d87d7261c81437d410dda161fe3b8efcbf"
_FORCE_OWNER = "almonacid-2024-muscle-aponeurosis-three-field-continuum"


def _scalar(value, name):
    result = jnp.asarray(value)
    if result.shape != () or jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must be a real scalar.")
    return (
        result
        if jnp.issubdtype(result.dtype, jnp.floating)
        else result.astype(jnp.float64)
    )


class Almonacid2024MuscleAponeurosisParameters(StrictModule):
    """Required dynamic material/density values; never a biological preset."""

    muscle: Almonacid2024MaterialParameters
    aponeurosis: Almonacid2024MaterialParameters
    density_kg_per_m3: Array

    def __init__(self, muscle, aponeurosis, density_kg_per_m3, /):
        if not isinstance(muscle, Almonacid2024MaterialParameters) or not isinstance(
            aponeurosis, Almonacid2024MaterialParameters
        ):
            raise TypeError("Both tissues require Almonacid2024MaterialParameters.")
        self.muscle, self.aponeurosis = muscle, aponeurosis
        self.density_kg_per_m3 = _scalar(density_kg_per_m3, "density_kg_per_m3")

    def values(self):
        return jnp.concatenate(
            tuple(jnp.ravel(x) for x in jax.tree_util.tree_leaves(self))
        )


class _ControlClock(StrictModule, NonTrainableState):
    time_s: Array
    successful: Array


class Almonacid2024Control(StrictModule):
    """Trainable load amplitudes on a fixed, identified control clock."""

    clock: _ControlClock
    activation: Array
    engineering_strain: Array
    source_id: str = eqx.field(static=True)

    def __init__(
        self, time_s, activation, engineering_strain, /, *, source_id, successful=True
    ):
        time = _scalar(time_s, "time_s")
        self.activation = _scalar(activation, "activation")
        self.engineering_strain = _scalar(engineering_strain, "engineering_strain")
        success = jnp.asarray(successful, dtype=jnp.bool_)
        if success.shape != () or not str(source_id).strip():
            raise ValueError(
                "Control requires scalar success and a nonempty source identity."
            )
        self.clock = _ControlClock(time, success)
        self.source_id = str(source_id)

    @property
    def time_s(self):
        return self.clock.time_s

    @property
    def successful(self):
        return self.clock.successful


class _HistoryTimeGrid(StrictModule, NonTrainableState):
    activation_time_s: Array
    strain_time_s: Array


class Almonacid2024InputHistory(StrictModule):
    """Source tabular linear interpolation, constant after the last knot.

    The C++ implementation has undefined behavior before its first knot. This
    route explicitly rejects that interval rather than inventing a prehistory.
    """

    time_grid: _HistoryTimeGrid
    activation: Array
    engineering_strain: Array
    source_id: str = eqx.field(static=True)
    source_provenance: str = eqx.field(static=True)

    def __init__(self, activation_table, strain_table, /, *, source_id):
        tables = [
            np.asarray(table, dtype=float) for table in (activation_table, strain_table)
        ]
        for table in tables:
            if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] < 2:
                raise ValueError("History tables must have shape (n >= 2, 2).")
            if not np.all(np.isfinite(table)) or np.any(np.diff(table[:, 0]) <= 0):
                raise ValueError("History knots must be finite and strictly increasing.")
        if not str(source_id).strip():
            raise ValueError("History requires a nonempty source identity.")
        self.time_grid = _HistoryTimeGrid(
            jnp.asarray(tables[0][:, 0]), jnp.asarray(tables[1][:, 0])
        )
        self.activation = jnp.asarray(tables[0][:, 1])
        self.engineering_strain = jnp.asarray(tables[1][:, 1])
        self.source_provenance = str(source_id)
        # Bind the loaded reference history, not merely a filename/revision label.
        # Optimized amplitudes remain numeric leaves of this identified family.
        self.source_id = canonical_fingerprint(
            {
                "source": self.source_provenance,
                "activation": tables[0].tolist(),
                "engineering_strain": tables[1].tolist(),
            }
        )

    @property
    def activation_time_s(self):
        return self.time_grid.activation_time_s

    @property
    def strain_time_s(self):
        return self.time_grid.strain_time_s

    def sample(self, time_s):
        time = _scalar(time_s, "time_s")
        valid = (
            jnp.isfinite(time)
            & (time >= self.activation_time_s[0])
            & (time >= self.strain_time_s[0])
        )
        return Almonacid2024Control(
            time,
            linear_interpolate(self.activation_time_s, self.activation, time).values,
            linear_interpolate(self.strain_time_s, self.engineering_strain, time).values,
            source_id=self.source_id,
            successful=valid,
        )


class Almonacid2024State(StrictModule, NonTrainableState):
    displacement_m: Array
    velocity_m_per_s: Array
    pressure_coefficients_Pa: Array
    dilation_coefficients: Array
    deformation_gradient: Array
    reference_velocity_gradient_per_s: Array
    normalized_strain_rate: Array
    time_s: Array
    activation: Array
    engineering_strain: Array
    accepted_steps: Array
    passive_energy_J: Array
    kinetic_energy_J: Array
    boundary_work_J: Array
    active_work_J: Array
    energy_residual_J: Array
    accepted_parameter_values: Array


class Almonacid2024MixedSpaceStatus(StrictModule, NonTrainableState):
    """Discrete space declaration and measured weak constraints, not an LBB proof."""

    pressure_weak_residual_m3: Array
    dilation_weak_residual_J: Array
    pointwise_volume_mismatch: Array
    pressure_jump_l2_Pa: Array
    displacement_degree: int = eqx.field(static=True, default=2)
    scalar_modes_per_cell: int = eqx.field(static=True, default=4)
    scalar_conformity: str = eqx.field(
        static=True, default="cell-discontinuous-total-degree-one"
    )
    quadrature_points_per_axis: int = eqx.field(static=True, default=5)
    pressure_gauge: str = eqx.field(
        static=True, default="none;volumetric-dilation-equation-fixes-pressure"
    )
    stabilization: str = eqx.field(static=True, default="none")
    inf_sup_qualified: bool = eqx.field(static=True, default=False)


class Almonacid2024Diagnostics(StrictModule, NonTrainableState):
    reaction_fixed_N: Array
    reaction_pulling_N: Array
    source_boundary_force_N: Array
    free_force_residual_N: Array
    interface_displacement_jump_l2_m: Array
    interface_traction_jump_l2_Pa: Array
    interface_power_defect_W: Array
    interface_pressure_jump_Pa: Array
    boundary_work_increment_J: Array
    active_work_increment_J: Array
    backward_euler_kinetic_dissipation_J: Array
    work_energy_residual_J: Array
    source_stress_displacement_contraction_J: Array
    current_volume_m3: Array
    mixed_space: Almonacid2024MixedSpaceStatus
    admissible: Array
    force_owner: str = eqx.field(static=True, default=_FORCE_OWNER)


class Almonacid2024Candidate(StrictModule, NonTrainableState):
    """Uncommitted solve evidence bound to the complete originating preparation."""

    origin: PreparedAlmonacid2024MuscleAponeurosis
    proposed: Almonacid2024State
    nonlinear_result: NonlinearResult
    diagnostics: Almonacid2024Diagnostics
    input_valid: Array

    @property
    def successful(self):
        finite = jnp.all(
            jnp.stack(
                [
                    jnp.all(jnp.isfinite(x))
                    for x in jax.tree_util.tree_leaves(self.proposed)
                ]
            )
        )
        return (
            self.input_valid
            & self.nonlinear_result.successful
            & self.diagnostics.admissible
            & finite
        )

    def commit(self, current_prepared, /, *, accept=True):
        """Atomically select this solve or the unchanged current preparation.

        State, parameter leaves, geometry/QP arrays and all static solver policy
        must still match the proposal's origin. No caller-supplied receipt is
        trusted as an accepted state or an acceptance flag.
        """
        if not isinstance(current_prepared, PreparedAlmonacid2024MuscleAponeurosis):
            raise TypeError(
                "current_prepared must be PreparedAlmonacid2024MuscleAponeurosis."
            )
        if jax.tree_util.tree_structure(current_prepared) != jax.tree_util.tree_structure(
            self.origin
        ):
            raise ValueError(
                "Candidate belongs to a foreign preparation or static policy."
            )
        outer = jnp.asarray(accept, dtype=jnp.bool_)
        if outer.shape != ():
            raise ValueError("accept must be scalar.")
        # Tracer implementation types may differ under JVP; values/dtypes may not.
        matches = eqx.tree_equal(current_prepared, self.origin)
        current = eqx.error_if(
            current_prepared,
            ~jnp.asarray(matches),
            "Stale continuum state, changed parameters, or changed frozen geometry/policy.",
        )
        accepted = self.successful & outer
        state = jax.tree_util.tree_map(
            lambda a, b: jnp.where(accepted, a, b), self.proposed, current.state
        )
        return eqx.tree_at(lambda x: x.state, current, state)


class Almonacid2024MuscleAponeurosisPlan(StrictModule, NonTrainableState):
    geometry: Almonacid2024Geometry
    control_source_id: str = eqx.field(static=True)
    dynamic: bool = eqx.field(static=True)
    pulling_face_id: int = eqx.field(static=True)
    stress_scale_Pa: float = eqx.field(static=True)
    method: NewtonKrylov
    termination: NonlinearTermination
    provenance: SemanticProvenance
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: Almonacid2024Geometry,
        /,
        *,
        control_source_id,
        dynamic=True,
        pulling_face_id=1,
        stress_scale_Pa=2e5,
        method=None,
        termination=None,
    ):
        if not isinstance(geometry, Almonacid2024Geometry):
            raise TypeError("geometry must be Almonacid2024Geometry.")
        if not str(control_source_id).strip() or pulling_face_id not in range(1, 8):
            raise ValueError(
                "A source identity and source pulling face in 1..7 are required."
            )
        if not np.isfinite(stress_scale_Pa) or stress_scale_Pa <= 0:
            raise ValueError("stress_scale_Pa must be finite and positive.")
        self.geometry, self.control_source_id = geometry, str(control_source_id)
        self.dynamic, self.pulling_face_id = bool(dynamic), int(pulling_face_id)
        self.stress_scale_Pa = float(stress_scale_Pa)
        self.method = (
            NewtonKrylov(linear_policy=almonacid_2024_linear_policy())
            if method is None
            else method
        )
        if (
            not isinstance(self.method, NewtonKrylov)
            or self.method.jacobian_policy.mode != "autodiff"
        ):
            raise ValueError("Source production solve requires matrix-free NewtonKrylov.")
        self.termination = (
            NonlinearTermination(
                absolute_residual=1e-8, relative_residual=1e-8, maximum_steps=100
            )
            if termination is None
            else termination
        )
        if not isinstance(self.termination, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination.")
        self.provenance = SemanticProvenance(
            {
                "kind": _FORCE_OWNER,
                "geometry": geometry,
                "control_source": self.control_source_id,
                "dynamic": self.dynamic,
                "pulling_face": self.pulling_face_id,
                "stress_scale_Pa": self.stress_scale_Pa,
                "method": strict_module_payload(self.method),
                "termination": strict_module_payload(self.termination),
                "source_case": "repository-default-equations;not-SIAM-table",
                "spaces": "Q2-vector x DGPM1-four-modes x DGPM1-four-modes",
                "quadrature": "QGauss(5)^3",
                "boundary": "face0-clamped;pulling-x-strain;all-other-tractions-zero",
                "force_owner": "three-field-residual",
                "tissue_ids": (1, 2),
                "time_scheme": "backward-Euler-displacement-velocity",
                "scope": "idealized-muscle-aponeurosis;no-tendon;no-biological-validation",
            },
            resource_ids={
                "source_commit": _SOURCE_COMMIT,
                "publication": "doi:10.1137/22M1506985",
            },
        )
        self.plan_id = self.provenance.semantic_id

    def prepare(self, parameters: Almonacid2024MuscleAponeurosisParameters, /):
        if not isinstance(parameters, Almonacid2024MuscleAponeurosisParameters):
            raise TypeError(
                "parameters must be Almonacid2024MuscleAponeurosisParameters."
            )
        if not jax.config.x64_enabled:
            raise ValueError("The source-matched continuum requires JAX float64 enabled.")
        geometry = prepare_geometry(self.geometry, self.pulling_face_id)
        dtype = parameters.density_kg_per_m3.dtype
        if dtype != jnp.float64 or any(
            x.dtype != dtype for x in jax.tree_util.tree_leaves(parameters)
        ):
            raise ValueError("All source continuum parameters must use float64.")
        signature = ExecutableSignature(
            shapes={
                "displacement": (geometry.displacement_dof_count, 3),
                "cell_scalars": (geometry.cell_count, 4),
            },
            dtypes={"parameters": dtype},
            topology_ids={"fem": geometry.discretization.prepared_id},
            algorithm_facts={
                "owner": _FORCE_OWNER,
                "jacobian": "native-matrix-free",
                "root_derivative": "implicit-root",
            },
        )
        prepared_id = canonical_fingerprint(
            {"plan": self.plan_id, "signature": signature.signature_id}
        )
        zero = jnp.asarray(0.0, dtype=dtype)
        u = jnp.zeros((geometry.displacement_dof_count, 3), dtype=dtype)
        p = jnp.zeros((geometry.cell_count, 4), dtype=dtype)
        dilation = p.at[:, 0].set(1.0)
        F = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (geometry.cell_count, 125, 3, 3))
        state = Almonacid2024State(
            u,
            u,
            p,
            dilation,
            F,
            jnp.zeros_like(F),
            jnp.zeros(F.shape[:2], dtype=dtype),
            zero,
            zero,
            zero,
            jnp.asarray(0, dtype=jnp.uint32),
            zero,
            zero,
            zero,
            zero,
            zero,
            parameters.values(),
        )
        prepared = PreparedAlmonacid2024MuscleAponeurosis(
            self,
            parameters,
            geometry,
            state,
            signature,
            prepared_id,
            almonacid_2024_prepare_symbolic(geometry),
        )
        response = prepared.material_response(
            u, p, dilation, zero, jnp.asarray(1.0, dtype=dtype)
        )
        if not bool(jnp.all(response.admissible)) or not bool(
            jnp.isfinite(parameters.density_kg_per_m3)
            & (parameters.density_kg_per_m3 > 0)
        ):
            raise ValueError("Initial tissue material branch or density is inadmissible.")
        initial_energy = jnp.sum(
            response.passive_energy_density_J_per_m3 * geometry.weights_m3
        )
        return eqx.tree_at(lambda x: x.state.passive_energy_J, prepared, initial_energy)


class PreparedAlmonacid2024MuscleAponeurosis(StrictModule):
    plan: Almonacid2024MuscleAponeurosisPlan
    parameters: Almonacid2024MuscleAponeurosisParameters
    geometry: PreparedAlmonacid2024Geometry
    state: Almonacid2024State
    signature: ExecutableSignature
    prepared_id: str = eqx.field(static=True)
    linear_symbolic: Almonacid2024SymbolicPreconditioning

    def numeric_revision(self):
        arrays = tuple(
            x for x in jax.tree_util.tree_leaves(self.geometry) if eqx.is_array(x)
        )
        return NumericRevision(
            self.plan.provenance,
            {
                "parameters": self.parameters,
                "state": self.state,
                "geometry_arrays": arrays,
                "linear_symbolic": self.linear_symbolic,
            },
        )

    def _trajectory_parameters(self):
        return eqx.error_if(
            self.parameters,
            (self.state.accepted_steps > 0)
            & jnp.any(self.parameters.values() != self.state.accepted_parameter_values),
            "Accepted trajectory parameters are fixed; replay or reprepare after changing the law or density.",
        )

    def _tissue_parameters(self, tissue_ids):
        parameters = self._trajectory_parameters()
        return jax.tree_util.tree_map(
            lambda m, a: jnp.where(
                (tissue_ids == 1).reshape(tissue_ids.shape + (1,) * m.ndim), m, a
            ),
            parameters.muscle,
            parameters.aponeurosis,
        )

    def _response(
        self, F, F_previous, pressure, dilation, activation, dt, directions, tissues
    ):
        parameters = self._tissue_parameters(tissues)

        def cell(p, f, fp, pr, di, dr, ti):
            return jax.vmap(
                lambda ff, ffp, pp, jj: almonacid_2024_material_response(
                    p, ff, ffp, pp, jj, activation, dt, dr, ti, dynamic=self.plan.dynamic
                )
            )(f, fp, pr, di)

        return jax.vmap(cell)(
            parameters, F, F_previous, pressure, dilation, directions, tissues
        )

    def deformation(self, displacement_m):
        return jnp.eye(3, dtype=displacement_m.dtype) + contract(
            "cai,cqaJ->cqiJ",
            displacement_m[self.geometry.displacement_dofs],
            self.geometry.gradients,
        )

    def material_response(
        self,
        displacement_m,
        pressure_coefficients_Pa,
        dilation_coefficients,
        activation,
        dt_s,
    ):
        g = self.geometry
        F = self.deformation(displacement_m)
        F_previous = (
            self.state.deformation_gradient
            if self.plan.dynamic
            else jnp.broadcast_to(jnp.eye(3, dtype=F.dtype), F.shape)
        )
        return self._response(
            F,
            F_previous,
            contract("qa,ca->cq", g.scalar_basis, pressure_coefficients_Pa),
            contract("qa,ca->cq", g.scalar_basis, dilation_coefficients),
            activation,
            dt_s,
            g.reference_directions,
            g.tissue_ids,
        )

    def quadrature_fields(self, dt_s, /):
        """Accepted raw fields in source column conventions, retaining step rates.

        The source's ``orientation`` is F*a0 (not a unit direction), and both
        strain-rate columns are normalized by the maximum fibre strain rate.
        """
        g, s = self.geometry, self.state
        dt = _scalar(dt_s, "dt_s")
        previous = s.deformation_gradient - dt * s.reference_velocity_gradient_per_s
        pressure = contract("qa,ca->cq", g.scalar_basis, s.pressure_coefficients_Pa)
        dilation = contract("qa,ca->cq", g.scalar_basis, s.dilation_coefficients)
        response = self._response(
            s.deformation_gradient,
            previous,
            pressure,
            dilation,
            s.activation,
            dt,
            g.reference_directions,
            g.tissue_ids,
        )
        u = contract("qa,cai->cqi", g.basis, s.displacement_m[g.displacement_dofs])
        v = contract("qa,cai->cqi", g.basis, s.velocity_m_per_s[g.displacement_dofs])
        orientation = contract(
            "cqiJ,cJ->cqi", s.deformation_gradient, g.reference_directions
        )
        spatial_rate = contract(
            "cqiJ,cqJk->cqik",
            s.reference_velocity_gradient_per_s,
            finite_strain_kinematics(s.deformation_gradient).inverse_deformation_gradient,
        )
        maximum_rate = jnp.where(
            g.tissue_ids == 1,
            self.parameters.muscle.maximum_strain_rate_per_s,
            self.parameters.aponeurosis.maximum_strain_rate_per_s,
        )
        total_rate = response.volume_ratio ** (1 / 3) * (
            response.normalized_strain_rate
            + jnp.trace(spatial_rate, axis1=-2, axis2=-1)
            * response.isochoric_fiber_stretch
            / (3 * maximum_rate[:, None])
        )
        return {
            "points_m": g.points_m,
            "weights_m3": g.weights_m3,
            "deformation_gradient": s.deformation_gradient,
            "displacement_m": u,
            "velocity_m_per_s": v,
            "pressure_Pa": pressure,
            "dilation": dilation,
            "orientation": orientation,
            "normalized_total_strain_rate": total_rate,
            "response": response,
        }

    def residual(
        self,
        displacement_m,
        pressure_coefficients_Pa,
        dilation_coefficients,
        control: Almonacid2024Control,
    ):
        """Physical unconstrained residual: N, m³, J in its three blocks.

        No penalty, interface spring, fibre force, or pressure smoothing is added.
        Dirichlet reaction rows remain available to the work/reaction owner.
        """
        g = self.geometry
        dt = control.time_s - self.state.time_s
        response = self.material_response(
            displacement_m,
            pressure_coefficients_Pa,
            dilation_coefficients,
            control.activation,
            dt,
        )
        local = contract(
            "cqiJ,cqaJ,cq->cai", response.first_piola_Pa, g.gradients, g.weights_m3
        )
        if self.plan.dynamic:
            acceleration = (
                displacement_m
                - self.state.displacement_m
                - dt * self.state.velocity_m_per_s
            ) / (dt * dt)
            acceleration_qp = contract(
                "qa,cai->cqi", g.basis, acceleration[g.displacement_dofs]
            )
            local = local + self.parameters.density_kg_per_m3 * contract(
                "qa,cqi,cq->cai", g.basis, acceleration_qp, g.weights_m3
            )
        ru = jnp.zeros_like(displacement_m).at[g.displacement_dofs].add(local)
        rp = contract(
            "qa,cq,cq->ca", g.scalar_basis, response.volume_constraint, g.weights_m3
        )
        rj = contract(
            "qa,cq,cq->ca", g.scalar_basis, response.dilation_residual_Pa, g.weights_m3
        )
        return (ru, rp, rj), response

    def _displacement(self, free_scaled, strain):
        g = self.geometry
        u = (
            jnp.zeros_like(self.state.displacement_m)
            .at[g.free_dofs]
            .set(free_scaled * self.plan.geometry.muscle_length_m)
        )
        return u.at[g.pulling_dofs, 0].set(strain * self.plan.geometry.muscle_length_m)

    def _root_residual(self, coordinates, control):
        u, p, j = coordinates
        physical = self.residual(
            self._displacement(u, control.engineering_strain),
            p * self.plan.stress_scale_Pa,
            j,
            control,
        )[0]
        length, stress = self.plan.geometry.muscle_length_m, self.plan.stress_scale_Pa
        # Fixed reference row equilibration; no state-dependent residual modification.
        cell_volume = jnp.sum(self.geometry.weights_m3, axis=1)
        nodal = contract(
            "qa,qa,cq->ca",
            self.geometry.basis,
            self.geometry.basis,
            self.geometry.weights_m3,
        )
        nodal_volume = (
            jnp.zeros((self.geometry.displacement_dof_count,), dtype=u.dtype)
            .at[self.geometry.displacement_dofs]
            .add(nodal)
        )
        return (
            physical[0][self.geometry.free_dofs]
            * length
            / (stress * nodal_volume[self.geometry.free_dofs, None]),
            physical[1] / cell_volume[:, None],
            physical[2] / (stress * cell_volume[:, None]),
        )

    def propose(self, control: Almonacid2024Control, /):
        if any(
            value.dtype != self.state.time_s.dtype
            for value in jax.tree_util.tree_leaves(self.parameters)
        ):
            raise ValueError("Parameters cannot change the prepared numeric dtype.")
        parameters = self._trajectory_parameters()
        prepared = eqx.tree_at(lambda value: value.parameters, self, parameters)
        if not isinstance(control, Almonacid2024Control):
            raise TypeError("control must be Almonacid2024Control.")
        if control.source_id != prepared.plan.control_source_id:
            raise ValueError("Control belongs to a foreign input source.")
        if any(
            value.dtype != prepared.state.time_s.dtype
            for value in (
                control.time_s,
                control.activation,
                control.engineering_strain,
            )
        ):
            raise ValueError("Control cannot change the prepared numeric dtype.")
        dt = control.time_s - prepared.state.time_s
        input_valid = (
            control.successful
            & jnp.isfinite(dt)
            & (dt > 0)
            & jnp.isfinite(control.activation)
            & jnp.isfinite(control.engineering_strain)
        )
        density_valid = jnp.isfinite(prepared.parameters.density_kg_per_m3) & (
            prepared.parameters.density_kg_per_m3 > 0
        )
        problem = NonlinearSystemProblem(
            lambda coordinates, forcing: prepared._root_residual(coordinates, forcing),
            validity=lambda _state, _residual, _auxiliary, _forcing: (
                input_valid & density_valid
            ),
            linear_setup=lambda coordinates, forcing: almonacid_2024_setup_operator(
                prepared, coordinates, forcing
            ),
            problem_id=f"{prepared.prepared_id}:three-field",
        )
        initial = (
            prepared.state.displacement_m[prepared.geometry.free_dofs]
            / prepared.plan.geometry.muscle_length_m,
            prepared.state.pressure_coefficients_Pa / prepared.plan.stress_scale_Pa,
            prepared.state.dilation_coefficients,
        )
        result = implicit_root_result(
            problem,
            initial,
            method=prepared.plan.method,
            termination=prepared.plan.termination,
            args=control,
        )
        free_displacement, pressure, dilation = result.state
        displacement = prepared._displacement(
            free_displacement, control.engineering_strain
        )
        pressure = pressure * prepared.plan.stress_scale_Pa
        residual, response = prepared.residual(displacement, pressure, dilation, control)
        deformation = prepared.deformation(displacement)
        velocity = (
            (displacement - prepared.state.displacement_m) / dt
            if prepared.plan.dynamic
            else jnp.zeros_like(displacement)
        )
        diagnostics = prepared._diagnostics(
            displacement,
            velocity,
            pressure,
            dilation,
            control,
            residual,
            response,
        )
        passive = jnp.sum(
            response.passive_energy_density_J_per_m3 * prepared.geometry.weights_m3
        )
        quadrature_velocity = contract(
            "qa,cai->cqi",
            prepared.geometry.basis,
            velocity[prepared.geometry.displacement_dofs],
        )
        kinetic = (
            0.5
            * prepared.parameters.density_kg_per_m3
            * jnp.sum(
                quadrature_velocity
                * quadrature_velocity
                * prepared.geometry.weights_m3[..., None]
            )
        )
        rate_reference = (
            prepared.state.deformation_gradient
            if prepared.plan.dynamic
            else jnp.eye(3, dtype=deformation.dtype)
        )
        proposed = Almonacid2024State(
            displacement,
            velocity,
            pressure,
            dilation,
            deformation,
            (deformation - rate_reference) / dt,
            response.normalized_strain_rate,
            control.time_s,
            control.activation,
            control.engineering_strain,
            prepared.state.accepted_steps + jnp.asarray(1, dtype=jnp.uint32),
            passive,
            kinetic,
            prepared.state.boundary_work_J + diagnostics.boundary_work_increment_J,
            prepared.state.active_work_J + diagnostics.active_work_increment_J,
            prepared.state.energy_residual_J + diagnostics.work_energy_residual_J,
            prepared.parameters.values(),
        )
        return Almonacid2024Candidate(
            prepared,
            proposed,
            result,
            diagnostics,
            input_valid & density_valid,
        )

    def _trace_fields(self, u, p, j, control, *, neighbour=False):
        g, t = self.geometry, self.geometry.traces
        cells = (
            jnp.where(t.neighbour_cells >= 0, t.neighbour_cells, t.cells)
            if neighbour
            else t.cells
        )
        basis = t.neighbour_basis if neighbour else t.basis
        grad = t.neighbour_gradients if neighbour else t.gradients
        scalar = t.neighbour_scalar_basis if neighbour else t.scalar_basis
        displacement = contract("fqa,fai->fqi", basis, u[g.displacement_dofs[cells]])
        F = jnp.eye(3, dtype=u.dtype) + contract(
            "fai,fqaJ->fqiJ", u[g.displacement_dofs[cells]], grad
        )
        previous = jnp.eye(3, dtype=u.dtype) + contract(
            "fai,fqaJ->fqiJ", self.state.displacement_m[g.displacement_dofs[cells]], grad
        )
        if not self.plan.dynamic:
            previous = jnp.broadcast_to(jnp.eye(3, dtype=u.dtype), F.shape)
        pressure = contract("fqa,fa->fq", scalar, p[cells])
        dilation = contract("fqa,fa->fq", scalar, j[cells])
        response = self._response(
            F,
            previous,
            pressure,
            dilation,
            control.activation,
            control.time_s - self.state.time_s,
            g.reference_directions[cells],
            g.tissue_ids[cells],
        )
        return displacement, pressure, response

    def _diagnostics(self, u, velocity, p, j, control, residual, response):
        g, t = self.geometry, self.geometry.traces
        dt = control.time_s - self.state.time_s
        owner_u, owner_p, owner = self._trace_fields(u, p, j, control)
        neighbour_u, neighbour_p, neighbour = self._trace_fields(
            u, p, j, control, neighbour=True
        )
        interface = t.neighbour_cells >= 0
        weights = t.weights_m2 * interface[:, None]
        area = jnp.sum(weights)
        jump_u = owner_u - neighbour_u
        jump_t = contract(
            "fqiJ,fqJ->fqi", owner.first_piola_Pa - neighbour.first_piola_Pa, t.normals
        )
        jump_p = owner_p - neighbour_p
        step_velocity = (u - self.state.displacement_m) / dt
        vface = contract(
            "fqa,fai->fqi", t.basis, step_velocity[g.displacement_dofs[t.cells]]
        )
        interface_power = jnp.sum(jump_t * vface * weights[..., None])
        source_P = response.first_piola_Pa[t.cells[:, None], t.nearest_volume_qp]
        source_traction = contract("fqiJ,fqJ->fqi", source_P, t.normals)
        source_force = (
            jnp.zeros((8, 3), dtype=u.dtype)
            .at[jnp.maximum(t.boundary_ids, 0)]
            .add(
                jnp.sum(source_traction * t.weights_m2[..., None], axis=1)
                * (t.boundary_ids >= 0)[:, None]
            )
        )
        delta_u = u - self.state.displacement_m
        boundary_work = jnp.sum(
            residual[0][g.fixed_dofs] * delta_u[g.fixed_dofs]
        ) + jnp.sum(residual[0][g.pulling_dofs] * delta_u[g.pulling_dofs])
        delta_F = self.deformation(u) - self.state.deformation_gradient
        # P_active : delta_F = tau_active : (delta_F F^-1), no active stored energy.
        inverse_F = finite_strain_kinematics(
            self.deformation(u)
        ).inverse_deformation_gradient
        active_P = contract("cqiK,cqJK->cqiJ", response.kirchhoff_active_Pa, inverse_F)
        active_work = jnp.sum(active_P * delta_F * g.weights_m3[..., None, None])
        vqp = contract("qa,cai->cqi", g.basis, velocity[g.displacement_dofs])
        dvqp = contract(
            "qa,cai->cqi",
            g.basis,
            (velocity - self.state.velocity_m_per_s)[g.displacement_dofs],
        )
        kinetic = (
            0.5
            * self.parameters.density_kg_per_m3
            * jnp.sum(vqp * vqp * g.weights_m3[..., None])
        )
        dissipation = (
            0.5
            * self.parameters.density_kg_per_m3
            * jnp.sum(dvqp * dvqp * g.weights_m3[..., None])
        )
        passive = jnp.sum(response.passive_energy_density_J_per_m3 * g.weights_m3)

        # Before the first accepted step, parameters remain optimizable. Rebind
        # the reference energy to that law rather than subtracting a stale cache.
        def initial_passive_energy(_):
            previous = self.material_response(
                self.state.displacement_m,
                self.state.pressure_coefficients_Pa,
                self.state.dilation_coefficients,
                self.state.activation,
                dt,
            )
            return jnp.sum(previous.passive_energy_density_J_per_m3 * g.weights_m3)

        previous_passive = jax.lax.cond(
            self.state.accepted_steps == 0,
            initial_passive_energy,
            lambda _: self.state.passive_energy_J,
            operand=None,
        )
        energy_residual = (
            kinetic
            - self.state.kinetic_energy_J
            + passive
            - previous_passive
            + active_work
            - boundary_work
        )
        source_contraction = jnp.sum(
            response.first_piola_Pa
            * (self.deformation(u) - jnp.eye(3, dtype=u.dtype))
            * g.weights_m3[..., None, None]
        )
        pjump_l2 = jnp.sqrt(jnp.sum(jump_p * jump_p * weights) / area)
        mixed = Almonacid2024MixedSpaceStatus(
            jnp.linalg.norm(residual[1]),
            jnp.linalg.norm(residual[2]),
            jnp.sqrt(
                jnp.sum(response.volume_constraint**2 * g.weights_m3)
                / jnp.sum(g.weights_m3)
            ),
            pjump_l2,
        )
        admissible = (
            jnp.all(response.admissible)
            & jnp.all(owner.admissible)
            & jnp.all(jnp.where(interface[:, None], neighbour.admissible, True))
        )
        return Almonacid2024Diagnostics(
            jnp.sum(residual[0][g.fixed_dofs], axis=0),
            jnp.sum(residual[0][g.pulling_dofs], axis=0),
            source_force,
            jnp.linalg.norm(residual[0][g.free_dofs]),
            jnp.sqrt(jnp.sum(jump_u * jump_u * weights[..., None]) / area),
            jnp.sqrt(jnp.sum(jump_t * jump_t * weights[..., None]) / area),
            interface_power,
            jnp.where(interface[:, None], jump_p, 0.0),
            boundary_work,
            active_work,
            dissipation,
            energy_residual,
            source_contraction,
            jnp.sum(response.volume_ratio * g.weights_m3),
            mixed,
            admissible,
        )


def _verify_repository_inputs(directory):
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Pinned repository inputs require their immutable manifest.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("commit") != _SOURCE_COMMIT:
        raise ValueError(
            "Repository input manifest belongs to a different source revision."
        )
    for name in (
        "parameters.prm",
        "control_points_activation.dat",
        "control_points_strain.dat",
    ):
        path = directory / name
        entry = manifest["upstream_files"][name]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Pinned repository input is missing or indirect: {name}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Pinned repository input content mismatch: {name}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Pinned repository input content mismatch: {name}")


def almonacid_2024_repository_case(
    input_directory, /, *, refinement=None, protocol="repository-default-fields"
):
    """Load pinned executable input files explicitly; these are not a biological preset.

    ``fixed-end-activation`` replaces strain by zero; ``passive-cyclic`` replaces
    activation by zero. Both are declared derived numerical controls, not papers'
    experimental observations. Files are read only during host preparation.
    """
    directory = Path(input_directory)
    _verify_repository_inputs(directory)
    values = {}
    for line in (directory / "parameters.prm").read_text().splitlines():
        if line.strip().startswith("set "):
            key, value = line.strip()[4:].split("=", 1)
            values[key.strip()] = value.split("#", 1)[0].strip()

    def number(name):
        return float(values[name])

    if number("Polynomial degree") != 2 or number("Quadrature order") != 5:
        raise ValueError("Source case requires Q2/DGPM1 and QGauss(5).")
    geometry = Almonacid2024Geometry(
        muscle_length_m=number("Muscle length"),
        aponeurosis_length_m=number("Aponeurosis length"),
        aponeurosis_height_m=number("Aponeurosis height"),
        muscle_width_m=number("Muscle width"),
        pennation_angle_rad=number("Pennation angle") * np.pi / 180,
        refinement=int(number("Global refinement")) if refinement is None else refinement,
    )
    muscle = Almonacid2024MaterialParameters(
        maximum_fiber_stress_Pa=number("Sigma naught muscle"),
        bulk_modulus_Pa=number("Bulk modulus muscle"),
        maximum_base_stress_Pa=number("Sigma naught base material"),
        base_scale=number("Muscle base material factor"),
        base_coefficients=jnp.array(
            [number(f"Muscle base material constant {k}") for k in range(1, 4)]
        ),
        maximum_strain_rate_per_s=number("Max strain rate"),
        fat_bulk_modulus_Pa=number("Bulk modulus fat"),
        fat_scale=number("Fat factor"),
        fat_c1_Pa=number("Fat constant 1"),
        fat_fraction=number("Fat fraction"),
    )
    apo = Almonacid2024MaterialParameters(
        maximum_fiber_stress_Pa=number("Sigma naught aponeurosis"),
        bulk_modulus_Pa=number("Bulk modulus aponeurosis"),
        maximum_base_stress_Pa=number("Sigma naught aponeurosis base material"),
        base_scale=number("Aponeurosis base material factor"),
        base_coefficients=jnp.array(
            [number(f"Aponeurosis base material constant {k}") for k in range(1, 4)]
        ),
        maximum_strain_rate_per_s=number("Max strain rate"),
        fat_bulk_modulus_Pa=0.0,
        fat_scale=0.0,
        fat_c1_Pa=0.0,
        fat_fraction=0.0,
    )
    activation = np.loadtxt(directory / "control_points_activation.dat")
    strain = np.loadtxt(directory / "control_points_strain.dat")
    if protocol == "fixed-end-activation":
        strain[:, 1] = 0.0
    elif protocol == "passive-cyclic":
        activation[:, 1] = 0.0
    elif protocol != "repository-default-fields":
        raise ValueError("Unknown source numerical protocol.")
    source_id = f"flexodeal:{protocol}"
    history = Almonacid2024InputHistory(activation, strain, source_id=source_id)
    simulation = values["Type of simulation"]
    if simulation not in ("dynamic", "quasi-static"):
        raise ValueError("Unsupported source simulation mode.")
    mechanical_solver = "condensed-jax-cpu" if jax.default_backend() == "cpu" else None
    method = NewtonKrylov(linear_policy=almonacid_2024_linear_policy(mechanical_solver))
    plan = Almonacid2024MuscleAponeurosisPlan(
        geometry,
        control_source_id=history.source_id,
        dynamic=simulation == "dynamic",
        pulling_face_id=int(number("Pulling face ID")),
        method=method,
    )
    parameters = Almonacid2024MuscleAponeurosisParameters(
        muscle, apo, number("Muscle density")
    )
    return plan, parameters, history, number("Time step size"), number("End time")


__all__ = [
    "Almonacid2024Geometry",
    "Almonacid2024MuscleAponeurosisParameters",
    "Almonacid2024Control",
    "Almonacid2024InputHistory",
    "Almonacid2024State",
    "Almonacid2024MixedSpaceStatus",
    "Almonacid2024Diagnostics",
    "Almonacid2024Candidate",
    "Almonacid2024MuscleAponeurosisPlan",
    "PreparedAlmonacid2024MuscleAponeurosis",
    "almonacid_2024_repository_case",
]
