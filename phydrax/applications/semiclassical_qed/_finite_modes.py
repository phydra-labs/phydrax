#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import PreparedRealCoordinateTree
from ...solver._differential import DifferentialProblem
from ._homogeneous import _qed_state_coordinates


_SIGMA = jnp.asarray(
    (
        ((0.0, 1.0), (1.0, 0.0)),
        ((0.0, -1j), (1j, 0.0)),
        ((1.0, 0.0), (0.0, -1.0)),
    ),
    dtype=complex,
)
_ZERO_2 = jnp.zeros((2, 2), dtype=complex)
_ALPHA = jnp.stack(
    tuple(
        jnp.concatenate(
            (
                jnp.concatenate((_ZERO_2, sigma), axis=1),
                jnp.concatenate((sigma, _ZERO_2), axis=1),
            ),
            axis=0,
        )
        for sigma in _SIGMA
    )
)
_BETA = jnp.diag(jnp.asarray((1.0, 1.0, -1.0, -1.0), dtype=complex))
_SIGMA_1 = _SIGMA[0]
_SIGMA_3 = _SIGMA[2]


class HomogeneousSpinorQED3DState(StrictModule):
    """Spatially homogeneous 3+1 gauge coordinates and two occupied spin modes."""

    vector_potential: Array
    electric_field: Array
    mode_spinors: Array

    def __init__(
        self,
        vector_potential: ArrayLike,
        electric_field: ArrayLike,
        mode_spinors: ArrayLike,
        /,
    ):
        potential = jnp.asarray(vector_potential)
        field = jnp.asarray(electric_field)
        modes = jnp.asarray(mode_spinors, dtype=complex)
        if potential.shape != (3,) or field.shape != (3,):
            raise ValueError("3+1 homogeneous gauge fields must have shape (3,).")
        if modes.ndim != 3 or modes.shape[1:] != (2, 4):
            raise ValueError("3+1 mode_spinors must have shape (modes, 2, 4).")
        self.vector_potential = potential
        self.electric_field = field
        self.mode_spinors = modes


class HomogeneousSpinorQED3DPlan(StrictModule, NonTrainableState):
    """Finite three-momentum 3+1 Dirac-Maxwell generalization."""

    momenta: Array
    quadrature_weights: Array
    charge: Array
    mass: Array
    external_current: Any
    maximum_modes: int = eqx.field(static=True)
    external_source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        momenta: ArrayLike,
        quadrature_weights: ArrayLike,
        /,
        *,
        charge: ArrayLike,
        mass: ArrayLike,
        external_current: Any | None = None,
        external_source_id: str | None = None,
        maximum_modes: int = 2**17,
    ):
        momentum = np.asarray(momenta, dtype=float)
        weights = np.asarray(quadrature_weights, dtype=float)
        q = np.asarray(charge, dtype=float)
        m = np.asarray(mass, dtype=float)
        if (
            momentum.ndim != 2
            or momentum.shape[1] != 3
            or momentum.shape[0] < 1
            or momentum.shape[0] > int(maximum_modes)
            or not np.isfinite(momentum).all()
        ):
            raise ValueError(
                "3+1 momenta must have finite shape (modes, 3) within capacity."
            )
        if weights.shape != (momentum.shape[0],) or not np.all(
            np.isfinite(weights) & (weights > 0.0)
        ):
            raise ValueError("3+1 quadrature weights must be positive and mode-aligned.")
        if q.shape != () or not np.isfinite(q):
            raise ValueError("charge must be finite scalar.")
        if m.shape != () or not np.isfinite(m) or float(m) < 0.0:
            raise ValueError("mass must be finite and non-negative.")
        if external_current is None:
            source = ZeroExternalCurrent3D()
            if external_source_id is not None:
                raise ValueError("The zero 3-vector source has a canonical identity.")
            source_id = source.source_id
        else:
            if not callable(external_current):
                raise TypeError("external_current must be callable or None.")
            if (
                external_source_id is None
                or not isinstance(external_source_id, str)
                or not external_source_id.strip()
            ):
                raise ValueError("Opaque 3-vector sources require external_source_id.")
            source = external_current
            source_id = external_source_id.strip()
        self.momenta = jnp.asarray(momentum)
        self.quadrature_weights = jnp.asarray(weights)
        self.charge = jnp.asarray(q)
        self.mass = jnp.asarray(m)
        self.external_current = source
        self.maximum_modes = int(maximum_modes)
        self.external_source_id = source_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "homogeneous-spinor-semiclassical-qed-3p1",
                "momenta": array_tree_fingerprint(momentum),
                "weights": array_tree_fingerprint(weights),
                "charge": array_tree_fingerprint(q),
                "mass": array_tree_fingerprint(m),
                "source": source_id,
            }
        )

    def kinetic_momenta(self, vector_potential: ArrayLike, /) -> Array:
        return self.momenta - self.charge * jnp.asarray(vector_potential)[..., None, :]

    def hamiltonians(self, vector_potential: ArrayLike, /) -> Array:
        kinetic = self.kinetic_momenta(vector_potential)
        return ein.contract("...ka,aij->...kij", kinetic, _ALPHA) + self.mass * _BETA

    def raw_current(self, modes: ArrayLike, /) -> Array:
        spinors = jnp.asarray(modes)
        per_mode = self.charge * jnp.real(
            ein.contract("...ksi,aij,...ksj->...ka", jnp.conj(spinors), _ALPHA, spinors)
        )
        return jnp.sum(self.quadrature_weights[..., None] * per_mode, axis=-2)

    def adiabatic_current(self, vector_potential: ArrayLike, /) -> Array:
        kinetic = self.kinetic_momenta(vector_potential)
        omega = jnp.sqrt(jnp.sum(kinetic * kinetic, axis=-1) + self.mass * self.mass)
        direction = jnp.where(omega[..., None] > 0.0, kinetic / omega[..., None], 0.0)
        return jnp.sum(
            self.quadrature_weights[..., None] * (-2.0 * self.charge * direction),
            axis=-2,
        )

    def renormalized_current(
        self, vector_potential: ArrayLike, modes: ArrayLike, /
    ) -> Array:
        return self.raw_current(modes) - self.adiabatic_current(vector_potential)

    def prepare(
        self,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        vector_potential: ArrayLike = (0.0, 0.0, 0.0),
        electric_field: ArrayLike = (0.0, 0.0, 0.0),
        mode_spinors: ArrayLike | None = None,
    ) -> PreparedHomogeneousSpinorQED3D:
        modes = (
            negative_energy_spinor_modes_3d(
                self.momenta,
                mass=self.mass,
                charge=self.charge,
                vector_potential=vector_potential,
            )
            if mode_spinors is None
            else jnp.asarray(mode_spinors)
        )
        state = HomogeneousSpinorQED3DState(vector_potential, electric_field, modes)
        if state.mode_spinors.shape[0] != self.momenta.shape[0]:
            raise ValueError("3+1 mode count does not match the plan.")
        vector_field = HomogeneousSpinorQED3DVectorField(self)
        state_coordinates = _qed_state_coordinates(state)
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-homogeneous-spinor-qed-3p1",
                "plan": self.plan_id,
                "initial": array_tree_fingerprint(state),
                "t0": array_tree_fingerprint(np.asarray(t0)),
                "t1": array_tree_fingerprint(np.asarray(t1)),
            }
        )
        problem = DifferentialProblem(
            vector_field,
            state,
            t0=t0,
            t1=t1,
            problem_id=canonical_fingerprint(
                {"kind": "homogeneous-spinor-qed-3p1-problem", "prepared": prepared_id}
            ),
        )
        return PreparedHomogeneousSpinorQED3D(
            self, state, problem, state_coordinates, prepared_id
        )


class ZeroExternalCurrent3D(StrictModule, NonTrainableState):
    source_id: str = eqx.field(static=True)

    def __init__(self):
        self.source_id = canonical_fingerprint({"kind": "zero-external-current-3-vector"})

    def __call__(self, time: ArrayLike, /) -> Array:
        del time
        return jnp.zeros((3,), dtype=float)


class HomogeneousSpinorQED3DVectorField(StrictModule):
    plan: HomogeneousSpinorQED3DPlan

    def __call__(
        self,
        time: Array,
        state: HomogeneousSpinorQED3DState,
        args: Any,
        /,
    ) -> HomogeneousSpinorQED3DState:
        del args
        hamiltonian = self.plan.hamiltonians(state.vector_potential)
        mode_rate = -1j * ein.contract("kij,ksj->ksi", hamiltonian, state.mode_spinors)
        current = self.plan.renormalized_current(
            state.vector_potential, state.mode_spinors
        )
        source = jnp.asarray(self.plan.external_current(time))
        return HomogeneousSpinorQED3DState(
            -state.electric_field,
            -(current + source),
            mode_rate,
        )


class PreparedHomogeneousSpinorQED3D(StrictModule, NonTrainableState):
    plan: HomogeneousSpinorQED3DPlan
    initial_state: HomogeneousSpinorQED3DState
    problem: DifferentialProblem
    state_coordinates: PreparedRealCoordinateTree
    prepared_id: str = eqx.field(static=True)


def negative_energy_spinor_modes_3d(
    momenta: ArrayLike,
    /,
    *,
    mass: ArrayLike,
    charge: ArrayLike = 0.0,
    vector_potential: ArrayLike = (0.0, 0.0, 0.0),
) -> Array:
    """Two orthonormal occupied eigenmodes for every finite 3-momentum."""

    momentum = jnp.asarray(momenta)
    kinetic = momentum - jnp.asarray(charge) * jnp.asarray(vector_potential)
    hamiltonians = (
        ein.contract("ka,aij->kij", kinetic, _ALPHA) + jnp.asarray(mass) * _BETA
    )
    _, vectors = jnp.linalg.eigh(hamiltonians)
    return jnp.swapaxes(vectors[..., :, :2], -1, -2)


class FiniteSpatialSpinorQEDState(StrictModule):
    """Finite spatial-grid temporal-gauge state in 1+1 dimensions."""

    vector_potential: Array
    electric_field: Array
    mode_spinors: Array

    def __init__(
        self,
        vector_potential: ArrayLike,
        electric_field: ArrayLike,
        mode_spinors: ArrayLike,
        /,
    ):
        potential = jnp.asarray(vector_potential)
        field = jnp.asarray(electric_field)
        modes = jnp.asarray(mode_spinors, dtype=complex)
        if potential.ndim != 1 or field.shape != potential.shape:
            raise ValueError("Finite spatial gauge fields must be matching vectors.")
        if modes.ndim != 3 or modes.shape[1:] != (potential.size, 2):
            raise ValueError("Finite spatial modes must have shape (modes, points, 2).")
        self.vector_potential = potential
        self.electric_field = field
        self.mode_spinors = modes


class FiniteSpatialSpinorQEDPlan(StrictModule, NonTrainableState):
    """Finite 1+1 spatial Dirac-Maxwell semidiscretization on a supplied derivative."""

    derivative: Array
    spatial_weights: Array
    canonical_momenta: Array
    mode_weights: Array
    charge: Array
    mass: Array
    external_current: Any
    volume: float = eqx.field(static=True)
    external_source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        derivative: ArrayLike,
        spatial_weights: ArrayLike,
        canonical_momenta: ArrayLike,
        mode_weights: ArrayLike,
        /,
        *,
        charge: ArrayLike,
        mass: ArrayLike,
        external_current: Any | None = None,
        external_source_id: str | None = None,
        skew_adjoint_tolerance: float = 1e-10,
        maximum_points: int = 8192,
        maximum_modes: int = 2**16,
    ):
        differential = np.asarray(derivative, dtype=float)
        weights = np.asarray(spatial_weights, dtype=float)
        momenta = np.asarray(canonical_momenta, dtype=float)
        mode_mass = np.asarray(mode_weights, dtype=float)
        q = np.asarray(charge, dtype=float)
        m = np.asarray(mass, dtype=float)
        tolerance = float(skew_adjoint_tolerance)
        if (
            differential.ndim != 2
            or differential.shape[0] != differential.shape[1]
            or differential.shape[0] < 2
            or differential.shape[0] > int(maximum_points)
            or not np.isfinite(differential).all()
        ):
            raise ValueError("derivative must be one finite square spatial matrix.")
        if weights.shape != (differential.shape[0],) or not np.all(
            np.isfinite(weights) & (weights > 0.0)
        ):
            raise ValueError("spatial_weights must be positive and point-aligned.")
        if (
            momenta.ndim != 1
            or momenta.size < 1
            or momenta.size > int(maximum_modes)
            or not np.isfinite(momenta).all()
            or mode_mass.shape != momenta.shape
            or not np.all(np.isfinite(mode_mass) & (mode_mass > 0.0))
        ):
            raise ValueError("Finite modes and weights are invalid or exceed capacity.")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("skew_adjoint_tolerance must be finite and non-negative.")
        weighted_defect = (
            weights[:, None] * differential + differential.T * weights[None, :]
        )
        if np.max(np.abs(weighted_defect)) > tolerance:
            raise ValueError("derivative must be skew-adjoint in the spatial quadrature.")
        if (
            q.shape != ()
            or not np.isfinite(q)
            or m.shape != ()
            or not np.isfinite(m)
            or float(m) < 0.0
        ):
            raise ValueError("charge and non-negative mass must be finite scalars.")
        if external_current is None:
            source = ZeroSpatialExternalCurrent(differential.shape[0])
            if external_source_id is not None:
                raise ValueError("The zero spatial source has a canonical identity.")
            source_id = source.source_id
        else:
            if not callable(external_current):
                raise TypeError("external_current must be callable or None.")
            if (
                external_source_id is None
                or not isinstance(external_source_id, str)
                or not external_source_id.strip()
            ):
                raise ValueError("Opaque spatial currents require external_source_id.")
            source = external_current
            source_id = external_source_id.strip()
        volume = float(weights.sum())
        self.derivative = jnp.asarray(differential)
        self.spatial_weights = jnp.asarray(weights)
        self.canonical_momenta = jnp.asarray(momenta)
        self.mode_weights = jnp.asarray(mode_mass)
        self.charge = jnp.asarray(q)
        self.mass = jnp.asarray(m)
        self.external_current = source
        self.volume = volume
        self.external_source_id = source_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-spatial-spinor-semiclassical-qed-1p1",
                "derivative": array_tree_fingerprint(differential),
                "spatial_weights": array_tree_fingerprint(weights),
                "canonical_momenta": array_tree_fingerprint(momenta),
                "mode_weights": array_tree_fingerprint(mode_mass),
                "charge": array_tree_fingerprint(q),
                "mass": array_tree_fingerprint(m),
                "source": source_id,
            }
        )

    def raw_current(self, modes: ArrayLike, /) -> Array:
        spinors = jnp.asarray(modes)
        per_mode = self.charge * jnp.real(
            ein.contract("...kxi,ij,...kxj->...kx", jnp.conj(spinors), _SIGMA_1, spinors)
        )
        return jnp.sum(self.mode_weights[..., None] * per_mode, axis=-2)

    def adiabatic_current(self, vector_potential: ArrayLike, /) -> Array:
        potential = jnp.asarray(vector_potential)
        kinetic = (
            self.canonical_momenta[..., :, None] - self.charge * potential[..., None, :]
        )
        omega = jnp.sqrt(kinetic * kinetic + self.mass * self.mass)
        ratio = jnp.where(omega > 0.0, kinetic / omega, 0.0)
        return (
            jnp.sum(self.mode_weights[..., None] * (-self.charge * ratio), axis=-2)
            / self.volume
        )

    def prepare(
        self,
        mode_spinors: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        vector_potential: ArrayLike,
        electric_field: ArrayLike,
    ) -> PreparedFiniteSpatialSpinorQED:
        state = FiniteSpatialSpinorQEDState(
            vector_potential, electric_field, mode_spinors
        )
        if state.mode_spinors.shape[0] != self.canonical_momenta.size:
            raise ValueError(
                "Finite spatial mode count does not match canonical_momenta."
            )
        norms = jnp.sum(
            self.spatial_weights[None, :, None] * jnp.abs(state.mode_spinors) ** 2,
            axis=(1, 2),
        )
        normalized_modes = state.mode_spinors / jnp.sqrt(norms[:, None, None])
        state = FiniteSpatialSpinorQEDState(
            state.vector_potential, state.electric_field, normalized_modes
        )
        state_coordinates = _qed_state_coordinates(state)
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-spatial-spinor-qed",
                "plan": self.plan_id,
                "initial": array_tree_fingerprint(state),
                "t0": array_tree_fingerprint(np.asarray(t0)),
                "t1": array_tree_fingerprint(np.asarray(t1)),
            }
        )
        problem = DifferentialProblem(
            FiniteSpatialSpinorQEDVectorField(self),
            state,
            t0=t0,
            t1=t1,
            problem_id=canonical_fingerprint(
                {"kind": "finite-spatial-spinor-qed-problem", "prepared": prepared_id}
            ),
        )
        return PreparedFiniteSpatialSpinorQED(
            self, state, problem, state_coordinates, prepared_id
        )


class ZeroSpatialExternalCurrent(StrictModule, NonTrainableState):
    point_count: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(self, point_count: int, /):
        count = int(point_count)
        if count < 1:
            raise ValueError("point_count must be positive.")
        self.point_count = count
        self.source_id = canonical_fingerprint(
            {"kind": "zero-spatial-external-current", "point_count": count}
        )

    def __call__(self, time: ArrayLike, /) -> Array:
        del time
        return jnp.zeros((self.point_count,), dtype=float)


class FiniteSpatialSpinorQEDVectorField(StrictModule):
    plan: FiniteSpatialSpinorQEDPlan

    def __call__(
        self,
        time: Array,
        state: FiniteSpatialSpinorQEDState,
        args: Any,
        /,
    ) -> FiniteSpatialSpinorQEDState:
        del args
        derivative_modes = ein.contract(
            "xy,kyi->kxi", self.plan.derivative, state.mode_spinors
        )
        kinetic_rate = -ein.contract("ij,kxj->kxi", _SIGMA_1, derivative_modes)
        local_hamiltonian = (
            -self.plan.charge * state.vector_potential[:, None, None] * _SIGMA_1
            + self.plan.mass * _SIGMA_3
        )
        local_rate = -1j * ein.contract(
            "xij,kxj->kxi", local_hamiltonian, state.mode_spinors
        )
        mode_rate = kinetic_rate + local_rate
        current = self.plan.raw_current(state.mode_spinors) - self.plan.adiabatic_current(
            state.vector_potential
        )
        source = jnp.asarray(self.plan.external_current(time))
        return FiniteSpatialSpinorQEDState(
            -state.electric_field,
            -(current + source),
            mode_rate,
        )


class PreparedFiniteSpatialSpinorQED(StrictModule, NonTrainableState):
    plan: FiniteSpatialSpinorQEDPlan
    initial_state: FiniteSpatialSpinorQEDState
    problem: DifferentialProblem
    state_coordinates: PreparedRealCoordinateTree
    prepared_id: str = eqx.field(static=True)

    def gauss_residual(
        self, state: FiniteSpatialSpinorQEDState, /, *, reference_charge: ArrayLike = 0.0
    ) -> Array:
        if not isinstance(state, FiniteSpatialSpinorQEDState):
            raise TypeError("state must be FiniteSpatialSpinorQEDState.")
        density = self.plan.charge * jnp.sum(
            self.plan.mode_weights[:, None]
            * jnp.sum(jnp.abs(state.mode_spinors) ** 2, axis=-1),
            axis=0,
        )
        return self.plan.derivative @ state.electric_field - (
            density - jnp.asarray(reference_charge)
        )


__all__ = [
    "FiniteSpatialSpinorQEDPlan",
    "FiniteSpatialSpinorQEDState",
    "FiniteSpatialSpinorQEDVectorField",
    "HomogeneousSpinorQED3DPlan",
    "HomogeneousSpinorQED3DState",
    "HomogeneousSpinorQED3DVectorField",
    "PreparedFiniteSpatialSpinorQED",
    "PreparedHomogeneousSpinorQED3D",
    "ZeroExternalCurrent3D",
    "ZeroSpatialExternalCurrent",
    "negative_energy_spinor_modes_3d",
]
