#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Material, wire, plan, status and evidence contracts of quasi-static foams."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import parameter_field, ParameterOwner
from ..._validation import positive_finite_float, positive_integer
from ...geometry.multiregion_surface import (
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
)
from ...interfacial_transport import InterfaceTensionMatrix
from ...typing import Bool, checked, Dim, Float, Float64, Identifier, Int32, parse, Scalar


FoamEquilibriumMethod: TypeAlias = Literal["sqp", "augmented_lagrangian"]
FoamEquilibriumRoute: TypeAlias = Literal[
    "sqp-exact-hessian",
    "augmented-lagrangian-trust-region",
    "unconstrained-trust-region",
]


class FoamDerivativeUnavailableError(RuntimeError):
    """Fixed-topology equilibrium derivatives were requested without regular KKT evidence."""


class FoamEquilibriumStatus(IntEnum):
    """Outcome of one quasi-static foam equilibrium solve."""

    CONVERGED = 0
    OPTIMIZER_FAILED = 1
    VOLUME_RESIDUAL_EXCEEDED = 2
    DEPENDENT_TARGETS_INCONSISTENT = 3
    NONFINITE = 4


class FoamKKTStatus(IntEnum):
    """Regularity of the equilibrium KKT matrix ``[[H, J^T], [J, 0]]``.

    ``REGULAR``: full rank, inertia ``(n, m, 0)`` (a strict local minimum on
    the prepared physical quotient with independent constraints) and condition
    number within the plan bound. ``INDEFINITE``: full rank but wrong inertia
    (a constrained saddle, e.g. the unstable catenoid branch).
    ``NOT_EVALUATED``: the dense evidence exceeded the plan's dense-dimension
    resource bound.
    """

    REGULAR = 0
    RANK_DEFICIENT = 1
    INDEFINITE = 2
    ILL_CONDITIONED = 3
    NOT_EVALUATED = 4


class _WireDim(Dim, minimum=1):
    """Constrained wire vertices."""


@final
class FoamWireConstraints(StrictModule, ParameterOwner):
    """Wire frames: prescribed coordinates of selected vertices.

    ``fixed_components[w]`` selects which Cartesian components of wire vertex
    ``vertex_global_ids[w]`` are held at ``positions[w]`` (all three for a
    pinned wire point; one for a vertex sliding on an axis-aligned plane).
    Positions are inferable parameters, so equilibrium derivatives with respect
    to the frame geometry (for example ring separation) are available.
    """

    __strict_contract__ = True

    positions: Float64[_WireDim, Literal[3]] = parameter_field()
    vertex_global_ids: tuple[int, ...] = eqx.field(static=True)
    fixed_components: tuple[tuple[bool, bool, bool], ...] = eqx.field(static=True)
    constraint_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        vertex_global_ids: Sequence[int],
        positions: ArrayLike,
        /,
        *,
        fixed_components: ArrayLike | None = None,
    ) -> None:
        ids = tuple(int(value) for value in vertex_global_ids)
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
            raise ValueError("vertex_global_ids must be unique nonnegative ids.")
        points = np.asarray(positions, dtype=np.float64)
        if points.shape != (len(ids), 3) or not np.all(np.isfinite(points)):
            raise ValueError("positions must be finite with shape (wire_vertices, 3).")
        mask = (
            np.ones((len(ids), 3), dtype=np.bool_)
            if fixed_components is None
            else np.asarray(fixed_components)
        )
        if mask.shape != (len(ids), 3) or mask.dtype != np.bool_:
            raise ValueError("fixed_components must be boolean (wire_vertices, 3).")
        if not np.all(np.any(mask, axis=1)):
            raise ValueError("Every wire vertex must fix at least one component.")
        order = np.argsort(np.asarray(ids), kind="stable")
        self.positions = jnp.asarray(points[order])
        self.vertex_global_ids = tuple(ids[index] for index in order)
        self.fixed_components = tuple(
            (bool(row[0]), bool(row[1]), bool(row[2])) for row in mask[order]
        )
        self.constraint_id = canonical_fingerprint(
            {
                "kind": "foam-wire-constraints",
                "vertex_global_ids": list(self.vertex_global_ids),
                "fixed_components": [list(row) for row in self.fixed_components],
            }
        )


@final
class FoamMaterialPlan(StrictModule):
    """Effective pair tensions and optional wire frames of one foam.

    ``tensions`` carries ``gamma_ij`` for every region label of the surface
    (matched by stable region identifier). A soap film between two gas cells
    has effective tension ``2 sigma``; use `FoamMaterialPlan.soap_film`.
    """

    __strict_contract__ = True

    tensions: InterfaceTensionMatrix
    wires: FoamWireConstraints | None
    material_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        tensions: InterfaceTensionMatrix,
        /,
        *,
        wires: FoamWireConstraints | None = None,
    ) -> None:
        if wires is not None and not isinstance(wires, FoamWireConstraints):
            raise TypeError("wires must be FoamWireConstraints or None.")
        host = np.asarray(tensions.values, dtype=np.float64)
        if not np.any(host > 0.0):
            raise ValueError("A foam needs at least one positive pair tension.")
        self.tensions = tensions
        self.wires = wires
        self.material_id = canonical_fingerprint(
            {
                "kind": "foam-material-plan",
                "tensions": tensions.structure_id,
                "tension_values": array_tree_fingerprint(host),
                "wires": None if wires is None else wires.constraint_id,
            }
        )

    @classmethod
    def soap_film(
        cls,
        region_ids: Sequence[str],
        surface_tension: float,
        /,
        *,
        wires: FoamWireConstraints | None = None,
    ) -> FoamMaterialPlan:
        """Soap-film foam: every film carries ``2 sigma`` (two liquid-gas surfaces)."""
        sigma = positive_finite_float(surface_tension, "surface_tension")
        return cls(
            InterfaceTensionMatrix(region_ids, 2.0 * sigma, structure="uniform"),
            wires=wires,
        )

    @checked
    def face_tension_indices(
        self, topology: MultiRegionSurfaceTopology, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tension-matrix label indices of the left and right label of every face slot.

        Refuses a surface carrying a region the tension matrix does not name
        (for example the children of a region split before the material is
        extended). Padding face slots map to the first label.
        """
        missing = sorted(set(topology.region_ids) - set(self.tensions.label_ids))
        if missing:
            raise ValueError(f"The tension matrix lacks surface regions {missing}.")
        index = np.asarray(
            [self.tensions.label_index(region) for region in topology.region_ids],
            dtype=np.int64,
        )
        labels = np.maximum(np.asarray(topology.face_labels, dtype=np.int64), 0)
        return index[labels[:, 0]], index[labels[:, 1]]


@final
class FoamEquilibriumPlan(StrictModule):
    """Solver route, tolerances and resource bounds of quasi-static equilibrium.

    ``method="augmented_lagrangian"`` (default) uses the native Powell–Hestenes
    method with a matrix-free Newton trust-region inner solver, which is robust
    to the indefinite tangential curvature of irregular meshes away from
    equilibrium; ``"sqp"`` uses native dense SQP with the exact Lagrangian
    Hessian and converges quadratically near a regular equilibrium (its convex
    QP subproblems fail on indefinite iterates). Open films without volume or
    geometric gauge constraints use the native unconstrained Newton trust
    region. Interior manifold vertices of open boundary-to-boundary films use
    the tangential reparameterization quotient directly;
    ``tangential_gauge_tolerance`` is the largest sheet-wide point-to-plane
    defect relative to sheet diameter for a finite-cell separating film to use
    the same quotient at its flat limit.
    ``optimality_tolerance`` is the nondimensional KKT tolerance;
    ``volume_tolerance`` bounds the accepted relative volume residual;
    ``maximum_dense_dimension`` bounds dense SQP and the dense KKT evidence
    (larger systems report ``NOT_EVALUATED`` evidence and refuse the dense SQP
    route at preparation).
    """

    __strict_contract__ = True

    method: FoamEquilibriumMethod = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    optimality_tolerance: float = eqx.field(static=True)
    volume_tolerance: float = eqx.field(static=True)
    tangential_gauge_tolerance: float = eqx.field(static=True)
    maximum_dense_dimension: int = eqx.field(static=True)
    kkt_relative_tolerance: float = eqx.field(static=True)
    maximum_kkt_condition: float = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        *,
        method: FoamEquilibriumMethod = "augmented_lagrangian",
        maximum_steps: int = 200,
        optimality_tolerance: float = 1.0e-8,
        volume_tolerance: float = 1.0e-7,
        tangential_gauge_tolerance: float = 1.0e-2,
        maximum_dense_dimension: int = 3000,
        kkt_relative_tolerance: float = 1.0e-10,
        maximum_kkt_condition: float = 1.0e12,
    ) -> None:
        method_ = parse(method, FoamEquilibriumMethod, "method")
        steps = positive_integer(maximum_steps, "maximum_steps")
        optimality = positive_finite_float(optimality_tolerance, "optimality_tolerance")
        volume = positive_finite_float(volume_tolerance, "volume_tolerance")
        gauge_tolerance = float(
            positive_finite_float(
                tangential_gauge_tolerance, "tangential_gauge_tolerance"
            )
        )
        if gauge_tolerance >= 0.25:
            raise ValueError("tangential_gauge_tolerance must be below 0.25.")
        dense = positive_integer(maximum_dense_dimension, "maximum_dense_dimension")
        relative = positive_finite_float(kkt_relative_tolerance, "kkt_relative_tolerance")
        condition = positive_finite_float(maximum_kkt_condition, "maximum_kkt_condition")
        if relative >= 1.0 or condition <= 1.0:
            raise ValueError("KKT tolerances must satisfy 0 < relative < 1 < condition.")
        self.method = method_
        self.maximum_steps = steps
        self.optimality_tolerance = optimality
        self.volume_tolerance = volume
        self.tangential_gauge_tolerance = gauge_tolerance
        self.maximum_dense_dimension = dense
        self.kkt_relative_tolerance = relative
        self.maximum_kkt_condition = condition
        self.plan_id = canonical_fingerprint(
            {
                "kind": "foam-equilibrium-plan",
                "method": method_,
                "maximum_steps": steps,
                "optimality_tolerance": optimality,
                "volume_tolerance": volume,
                "tangential_gauge_tolerance": gauge_tolerance.hex(),
                "maximum_dense_dimension": dense,
                "kkt_relative_tolerance": relative,
                "maximum_kkt_condition": condition,
            }
        )


class _PressureDim(Dim, minimum=2):
    """Region slots of one foam."""


class _FreeDim(Dim, minimum=1):
    """Free (unconstrained) vertex coordinates."""


class _MultiplierDim(Dim):
    """Equality multipliers (volume rows then gauge rows)."""


@final
class FoamEquilibriumEvidence(StrictModule):
    """Convergence, constraint, KKT and junction evidence of one equilibrium.

    Residuals are nondimensional: stationarity in units of ``gamma_ref ell``
    per unit ``ell`` displacement, volumes relative to ``ell^3``. The virial
    residual ``(3 sum_r p_r V_r - 2 E) / (2 E)`` vanishes exactly at any
    discrete equilibrium of a free foam (NaN when wires carry load).
    ``kkt_*`` fields describe the dense matrix ``[[H, J^T], [J, 0]]`` over free
    coordinates and all equality rows (volumes, rigid-motion gauge and the
    deterministic open-/flat-film tangential quotient basis).
    """

    __strict_contract__ = True

    status: Int32[Scalar]
    optimizer_status: Int32[Scalar]
    optimizer_successful: Bool[Scalar]
    iterations: Int32[Scalar]
    stationarity_residual: Float[Scalar]
    volume_residual: Float[Scalar]
    dependent_volume_residual: Float[Scalar]
    gauge_multiplier_norm: Float[Scalar]
    virial_residual: Float[Scalar]
    kkt_status: Int32[Scalar]
    kkt_rank: Int32[Scalar]
    kkt_positive: Int32[Scalar]
    kkt_negative: Int32[Scalar]
    kkt_zero: Int32[Scalar]
    kkt_condition: Float[Scalar]
    kkt_minimum_absolute_eigenvalue: Float[Scalar]
    second_order_sufficient: Bool[Scalar]
    derivative_available: Bool[Scalar]
    tension_admissible: Bool[Scalar]
    junction_minimum_angle: Float[Scalar]
    junction_maximum_angle: Float[Scalar]
    finite: Bool[Scalar]
    route: FoamEquilibriumRoute = eqx.field(static=True)
    primal_dimension: int = eqx.field(static=True)
    constraint_dimension: int = eqx.field(static=True)
    constrained_region_ids: tuple[str, ...] = eqx.field(static=True)
    pressure_reference_region_ids: tuple[str, ...] = eqx.field(static=True)
    rigid_motion_gauge: bool = eqx.field(static=True)
    tangential_gauge_dimension: int = eqx.field(static=True)
    prepared_id: Identifier = eqx.field(static=True)


@final
class FoamEquilibriumResult(StrictModule):
    """Equilibrium state, region pressures and evidence.

    ``pressures[r]`` is the pressure of region ``r`` relative to the declared
    gauge: boundary labels (the ambient) are the zero reference, and a finite
    region listed in ``evidence.pressure_reference_region_ids`` is the zero
    reference of the regions whose total volume is fixed together with it.
    ``free_coordinates`` and ``multipliers`` are the native optimizer vectors
    (nondimensional) used for warm starts and implicit derivatives.
    """

    __strict_contract__ = True

    state: MultiRegionSurfaceState
    pressures: Float[_PressureDim]
    region_volumes: Float[_PressureDim]
    energy: Float[Scalar]
    free_coordinates: Float[_FreeDim]
    multipliers: Float[_MultiplierDim]
    evidence: FoamEquilibriumEvidence

    @property
    def successful(self) -> bool:
        """Host decision: converged with satisfied volume constraints."""
        return int(self.evidence.status) == FoamEquilibriumStatus.CONVERGED


__all__ = [
    "FoamDerivativeUnavailableError",
    "FoamEquilibriumEvidence",
    "FoamEquilibriumMethod",
    "FoamEquilibriumPlan",
    "FoamEquilibriumResult",
    "FoamEquilibriumRoute",
    "FoamEquilibriumStatus",
    "FoamKKTStatus",
    "FoamMaterialPlan",
    "FoamWireConstraints",
]
