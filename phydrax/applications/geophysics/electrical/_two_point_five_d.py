#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la
from phydrax import ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import CellMesh, PolygonalConnectivity
from ....geometry.simplicial import AffineSimplexMap
from ....units import AMPERE, convert_value, derived_unit, METER, UnitDefinition


LINE_CURRENT_UNIT = derived_unit("A/m", ((AMPERE, 1), (METER, -1)))


class InvariantElectricalSurvey(StrictModule, NonTrainableState):
    positions_m: Array
    currents: Array
    receiver_weights: Array
    source_indices: Array
    current_kind: str = eqx.field(static=True)
    survey_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions: ArrayLike,
        currents: ArrayLike,
        receiver_weights: ArrayLike,
        source_indices: ArrayLike,
        /,
        *,
        current_kind: str,
        length_unit: UnitDefinition = METER,
        current_unit: UnitDefinition | None = None,
    ):
        if current_kind not in ("line-current", "point-current"):
            raise ValueError(
                "Invariant electrical current kind must be line-current or point-current."
            )
        expected_unit = LINE_CURRENT_UNIT if current_kind == "line-current" else AMPERE
        selected_unit = expected_unit if current_unit is None else current_unit
        positions_ = np.asarray(
            convert_value(positions, source=length_unit, target=METER), dtype=float
        )
        currents_ = np.asarray(
            convert_value(currents, source=selected_unit, target=expected_unit),
            dtype=float,
        )
        receivers = np.asarray(receiver_weights, dtype=float)
        indices = np.asarray(source_indices)
        if positions_.ndim != 2 or positions_.shape[1] != 3 or positions_.shape[0] < 4:
            raise ValueError(
                "Invariant electrical survey requires four or more XYZ electrodes."
            )
        if (
            currents_.ndim != 2
            or currents_.shape[1] != positions_.shape[0]
            or currents_.shape[0] == 0
        ):
            raise ValueError("Invariant survey current matrix has wrong shape.")
        if (
            receivers.ndim != 2
            or receivers.shape[1] != positions_.shape[0]
            or receivers.shape[0] == 0
        ):
            raise ValueError("Invariant survey receiver matrix has wrong shape.")
        if indices.shape != (receivers.shape[0],) or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError("Invariant survey source indices have wrong shape/type.")
        if (
            np.any(~np.isfinite(positions_))
            or np.any(~np.isfinite(currents_))
            or np.any(~np.isfinite(receivers))
            or np.any(indices < 0)
            or np.any(indices >= currents_.shape[0])
        ):
            raise ValueError("Invariant survey arrays must be finite and indices valid.")
        tolerance = 64 * np.finfo(float).eps
        if np.any(
            np.abs(np.sum(currents_, axis=1))
            > tolerance * np.sum(np.abs(currents_), axis=1)
        ):
            raise ValueError("Invariant current patterns must balance.")
        if np.any(
            np.abs(np.sum(receivers, axis=1))
            > tolerance * np.sum(np.abs(receivers), axis=1)
        ):
            raise ValueError(
                "Invariant receiver combinations must reject gauge potential."
            )
        self.positions_m = jnp.asarray(positions_)
        self.currents = jnp.asarray(currents_)
        self.receiver_weights = jnp.asarray(receivers)
        self.source_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.current_kind = current_kind
        self.survey_id = canonical_fingerprint(
            {
                "kind": "invariant-electrical-survey",
                "positions_m": positions_,
                "currents": currents_,
                "receiver_weights": receivers,
                "source_indices": indices,
                "current_kind": current_kind,
            }
        )


class PreparedInvariantElectricalGeometry(StrictModule, NonTrainableState):
    mesh: CellMesh
    cells: Array
    gradients: Array
    areas: Array
    point_cells: Array
    barycentric: Array
    space: la.ArraySpace
    gauge_space: la.ArraySpace
    gauge: Array
    geometry_id: str = eqx.field(static=True)

    def __init__(self, mesh: CellMesh, positions_m: ArrayLike, /):
        if (
            not isinstance(mesh, CellMesh)
            or mesh.topological_dimension != 2
            or mesh.ambient_dimension != 2
        ):
            raise ValueError(
                "Invariant electrical geometry requires a two-dimensional mesh."
            )
        if any(block.cell_kind != "triangle" for block in mesh.blocks):
            raise ValueError("Invariant electrical geometry supports triangles only.")
        connectivity = mesh.connectivity
        if not isinstance(connectivity, PolygonalConnectivity):
            raise TypeError(
                "Invariant electrical geometry requires polygonal connectivity."
            )
        cells = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks]
        )
        coordinates = np.asarray(mesh.coordinates, dtype=float)
        simplex = AffineSimplexMap(jnp.asarray(coordinates[cells]))
        if not bool(jnp.all(simplex.evidence.successful)):
            raise ValueError(
                "Invariant electrical triangles must be finite and nondegenerate."
            )
        gradients = np.asarray(simplex.barycentric_gradients)
        points = np.asarray(positions_m, dtype=float)[:, (0, 2)]
        point_cells: list[int] = []
        barycentric: list[np.ndarray] = []
        for point in points:
            weights = np.asarray(simplex.barycentric(jnp.asarray(point)))
            contained = np.all(weights >= -1e-10, axis=-1) & np.all(
                weights <= 1.0 + 1e-10, axis=-1
            )
            candidates = np.flatnonzero(contained)
            if not candidates.size:
                raise ValueError(
                    "Invariant electrode lies outside the two-dimensional mesh."
                )
            # Boundary points use the first canonical cell; interpolation is continuous.
            selected = int(candidates[0])
            point_cells.append(selected)
            barycentric.append(weights[selected])
        node_count = coordinates.shape[0]
        gauge = np.ones(node_count)
        gauge /= np.sqrt(node_count)
        self.mesh, self.cells = mesh, jnp.asarray(cells)
        self.gradients = jnp.asarray(gradients)
        self.areas = simplex.evidence.measure
        self.point_cells = jnp.asarray(point_cells, dtype=jnp.int32)
        self.barycentric = jnp.asarray(barycentric)
        self.space = la.ArraySpace((node_count,), dtype=coordinates.dtype)
        self.gauge_space = la.ArraySpace((node_count + 1,), dtype=coordinates.dtype)
        self.gauge = jnp.asarray(gauge)
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "invariant-electrical-geometry",
                "mesh": mesh.mesh_id,
                "points": points,
            }
        )

    def interpolation(self, nodal: Array) -> Array:
        nodes = self.cells[self.point_cells]
        return ein.contract("ei,ei->e", self.barycentric, nodal[nodes])

    def load(self, strengths: Array) -> Array:
        nodes = self.cells[self.point_cells]
        return (
            jnp.zeros((self.space.size,), dtype=strengths.dtype)
            .at[nodes]
            .add(self.barycentric * strengths[:, None])
        )

    def conductivity(self, value: ArrayLike) -> tuple[Array, Array]:
        raw = jnp.asarray(value)
        count = self.cells.shape[0]
        if raw.shape in ((), (count,)):
            scalar = jnp.broadcast_to(raw, (count,))
            xz = scalar[:, None, None] * jnp.eye(2)
            along = scalar
        elif raw.shape in ((3,), (count, 3)):
            diagonal = jnp.broadcast_to(raw, (count, 3))
            xz = jnp.zeros((count, 2, 2), dtype=raw.dtype)
            xz = xz.at[:, 0, 0].set(diagonal[:, 0])
            xz = xz.at[:, 1, 1].set(diagonal[:, 2])
            along = diagonal[:, 1]
        else:
            raise ValueError(
                "Invariant conductivity must be scalar/cell scalar or aligned XYZ diagonal."
            )
        xz = eqx.error_if(
            xz,
            jnp.any(~jnp.isfinite(xz))
            | jnp.any(jnp.diagonal(xz, axis1=-2, axis2=-1) <= 0)
            | jnp.any(~jnp.isfinite(along))
            | jnp.any(along <= 0),
            "Invariant conductivity components must be finite and positive.",
        )
        return xz, along

    def operator(
        self,
        conductivity: ArrayLike,
        wavenumber: ArrayLike,
        *,
        positive_definite: bool,
        complex_values: bool,
    ) -> la.FunctionLinearOperator:
        xz, along = self.conductivity(conductivity)
        wave = jnp.asarray(wavenumber)
        wave = eqx.error_if(
            wave,
            ~jnp.isfinite(wave) | (wave < 0),
            "Invariant wavenumber must be finite and nonnegative.",
        )
        local_stiffness = self.areas[:, None, None] * ein.contract(
            "cvi,cij,cwj->cvw", self.gradients, xz, self.gradients
        )
        reference_mass = (
            jnp.asarray(((2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0))) / 12.0
        )
        local_mass = self.areas[:, None, None] * along[:, None, None] * reference_mass

        def action(values):
            local = values[self.cells]
            result = ein.contract(
                "cij,cj->ci", local_stiffness + wave**2 * local_mass, local
            )
            return jnp.zeros_like(values).at[self.cells].add(result)

        space = (
            la.ArraySpace((self.space.size,), dtype=jnp.complex128)
            if complex_values
            else self.space
        )
        return la.FunctionLinearOperator(
            action,
            source=space,
            target=space,
            properties=la.OperatorProperties(
                self_adjoint=True,
                positive_definite=positive_definite,
                positive_semidefinite=not positive_definite,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite"
                    if positive_definite
                    else "positive_semidefinite": "construction",
                },
            ),
        )


class LineCurrentDCPlan(StrictModule, NonTrainableState):
    geometry: PreparedInvariantElectricalGeometry
    survey: InvariantElectricalSurvey
    policy: la.LinearSolvePolicy

    def __init__(self, mesh: CellMesh, survey: InvariantElectricalSurvey, /):
        if (
            not isinstance(survey, InvariantElectricalSurvey)
            or survey.current_kind != "line-current"
        ):
            raise TypeError(
                "Line-current DC requires line-current InvariantElectricalSurvey."
            )
        if np.any(np.asarray(survey.positions_m)[:, 1] != 0):
            raise ValueError(
                "Two-dimensional line-current positions must lie in invariant y=0 plane."
            )
        self.geometry, self.survey = (
            PreparedInvariantElectricalGeometry(mesh, survey.positions_m),
            survey,
        )
        self.policy = la.LinearSolvePolicy(
            la.MINRES(),
            tolerance=la.TolerancePolicy(relative=1e-9, absolute=1e-11, max_steps=2000),
            failure=la.FailurePolicy("status"),
        )

    def predict(self, conductivity: ArrayLike, /) -> Array:
        operator = self.geometry.operator(
            conductivity, 0.0, positive_definite=False, complex_values=False
        )
        nodes = self.geometry.space.size

        def kkt(values):
            return jnp.concatenate(
                (
                    operator.mv(values[:nodes]) + values[-1] * self.geometry.gauge,
                    jnp.vdot(self.geometry.gauge, values[:nodes])[None],
                )
            )

        kkt_operator = la.FunctionLinearOperator(
            kkt,
            source=self.geometry.gauge_space,
            target=self.geometry.gauge_space,
            properties=la.OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            ),
        )
        fields = []
        for current in self.survey.currents:
            rhs = jnp.concatenate((self.geometry.load(current), jnp.zeros(1)))
            solved = la.solve(la.LinearSystem(kkt_operator), rhs, policy=self.policy)
            fields.append(
                eqx.error_if(
                    solved.value[:nodes],
                    ~solved.successful,
                    "Line-current DC solve failed.",
                )
            )
        electrode = jnp.stack([self.geometry.interpolation(field) for field in fields])
        return ein.contract(
            "me,me->m",
            self.survey.receiver_weights,
            electrode[self.survey.source_indices],
        )


class TwoPointFiveDDCPlan(StrictModule, NonTrainableState):
    geometry: PreparedInvariantElectricalGeometry
    survey: InvariantElectricalSurvey
    wavenumbers_m_inverse: Array
    quadrature_weights_m_inverse: Array
    policy: la.LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        survey: InvariantElectricalSurvey,
        wavenumbers_m_inverse: ArrayLike,
        quadrature_weights_m_inverse: ArrayLike,
        /,
    ):
        if (
            not isinstance(survey, InvariantElectricalSurvey)
            or survey.current_kind != "point-current"
        ):
            raise TypeError("2.5D DC requires point-current InvariantElectricalSurvey.")
        waves = np.asarray(wavenumbers_m_inverse, dtype=float)
        weights = np.asarray(quadrature_weights_m_inverse, dtype=float)
        if (
            waves.ndim != 1
            or waves.size < 2
            or weights.shape != waves.shape
            or np.any(~np.isfinite(waves))
            or np.any(waves <= 0)
            or np.any(np.diff(waves) <= 0)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError(
                "2.5D wavenumbers/weights must be positive finite ordered vectors."
            )
        self.geometry, self.survey = (
            PreparedInvariantElectricalGeometry(mesh, survey.positions_m),
            survey,
        )
        self.wavenumbers_m_inverse, self.quadrature_weights_m_inverse = (
            jnp.asarray(waves),
            jnp.asarray(weights),
        )
        self.policy = la.LinearSolvePolicy(
            la.GMRES(restart=40, stagnation_iterations=40),
            tolerance=la.TolerancePolicy(relative=1e-9, absolute=1e-11, max_steps=2000),
            failure=la.FailurePolicy("status"),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-point-five-d-dc",
                "geometry": self.geometry.geometry_id,
                "survey": survey.survey_id,
                "wavenumbers": waves,
                "weights": weights,
            }
        )

    @classmethod
    def gauss_legendre(
        cls,
        mesh: CellMesh,
        survey: InvariantElectricalSurvey,
        /,
        *,
        wavenumber_max_m_inverse: float,
        count: int,
    ) -> TwoPointFiveDDCPlan:
        maximum, count_ = float(wavenumber_max_m_inverse), int(count)
        if not np.isfinite(maximum) or maximum <= 0 or count_ < 2:
            raise ValueError(
                "2.5D quadrature maximum/count must be positive and count >=2."
            )
        nodes, weights = np.polynomial.legendre.leggauss(count_)
        return cls(mesh, survey, 0.5 * maximum * (nodes + 1), 0.5 * maximum * weights)

    def predict(self, conductivity: ArrayLike, /) -> Array:
        responses = []
        for wave, weight in zip(
            self.wavenumbers_m_inverse,
            self.quadrature_weights_m_inverse,
            strict=True,
        ):
            operator = self.geometry.operator(
                conductivity, wave, positive_definite=True, complex_values=True
            )
            source_fields = []
            phase_source = jnp.exp(-1j * wave * self.survey.positions_m[:, 1])
            for current in self.survey.currents:
                load = self.geometry.load(current * phase_source)
                solved = la.solve(la.LinearSystem(operator), load, policy=self.policy)
                source_fields.append(
                    eqx.error_if(
                        solved.value, ~solved.successful, "2.5D wavenumber solve failed."
                    )
                )
            fields = jnp.stack(source_fields)
            electrode = jnp.stack(
                [self.geometry.interpolation(field) for field in fields]
            )
            phase_receiver = jnp.exp(1j * wave * self.survey.positions_m[:, 1])
            measurement = ein.contract(
                "me,me->m",
                self.survey.receiver_weights * phase_receiver[None, :],
                electrode[self.survey.source_indices],
            )
            responses.append(weight * jnp.real(measurement))
        return jnp.sum(jnp.stack(responses), axis=0) / jnp.pi


__all__ = [
    "InvariantElectricalSurvey",
    "LINE_CURRENT_UNIT",
    "LineCurrentDCPlan",
    "PreparedInvariantElectricalGeometry",
    "TwoPointFiveDDCPlan",
]
