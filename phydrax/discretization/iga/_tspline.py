#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import comb, isfinite
from typing import cast, NamedTuple, overload

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from ..._dtype_names import inexact_result_type
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation._tspline import (
    ASTSRefinement,
    certify_tspline_geometry,
    LocalExtractedBernsteinRealization,
    TMesh2D,
    TSplineGeometryCertificate,
    TSplineRationalJet,
)
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...geometry.brep._tspline import TSplineSurfacePatch
from ...linalg import (
    ArraySpace,
    BlockSpace,
    ConstraintMap,
    inverse_small_linear,
    SmallLinearSolvePlan,
)
from ...linalg._spaces import _coordinate_dtype
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ...typing import checked, ConvertibleToArray
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._integration_domain import IntegrationDomain
from .._lifecycle import AbstractDiscretizationPlan, validate_prepared_metadata
from .._local_variational import (
    AbstractPreparedLocalDiscretization,
    LocalFieldBinding,
    LocalGeometryActions,
    LocalMetricResult,
    LocalReferenceActions,
    LocalVariationalCapabilities,
    LocalVariationalOffer,
    PreparedLocalRegion,
)
from .._spaces import DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from .._topology import (
    CellComplexTopology,
    EntitySelection,
    EntitySet,
    OrientedIncidence,
)
from ..fem._precision import FiniteElementPrecisionPolicy
from ._basis import IsogeometricQuadraturePolicy
from ._cut import (
    build_trimmed_surface_quadrature,
    CutQuadratureRule,
    UVTrimLoop,
    UVTrimmedSurface,
)


def _fraction_payload(values: Sequence[Fraction], /) -> list[list[int]]:
    return [[value.numerator, value.denominator] for value in values]


def _exact_rows(
    values: Sequence[Sequence[Fraction]], /
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(Fraction(value) for value in row) for row in values)


def _exact_bank_from_numeric(
    control_points: np.ndarray, weights: np.ndarray, /
) -> tuple[tuple[Fraction, ...], ...]:
    exact_weights = tuple(Fraction.from_float(float(value)) for value in weights)
    return tuple(
        tuple(weight * Fraction.from_float(float(coordinate)) for coordinate in point)
        + (weight,)
        for point, weight in zip(control_points, exact_weights, strict=True)
    )


def _numeric_bank_id(control_points: object, weights: object, /) -> str:
    points = np.asarray(control_points)
    raw_weights = np.asarray(weights)
    return canonical_fingerprint(
        {
            "kind": "asts2d-numeric-geometry-bank",
            "control_points": array_tree_fingerprint(points),
            "raw_weights": array_tree_fingerprint(raw_weights),
        }
    )


def _source_numeric_bank_id(surface: TSplineSurfacePatch, /) -> str:
    return surface.source_numeric_bank_id


def _realize_bound_source_bank(
    surface: TSplineSurfacePatch,
    control_points: ArrayLike,
    weights: ArrayLike,
    source_numeric_bank_id: str,
    source_authority_id: str,
    /,
) -> tuple[Array, Array]:
    if source_authority_id != surface.source_authority_id:
        raise ValueError(
            "Numeric source leaves do not bind the retained exact CAD source authority."
        )
    points_input = jnp.asarray(control_points)
    weights_input = jnp.asarray(weights)
    if jnp.issubdtype(points_input.dtype, jnp.complexfloating) or jnp.issubdtype(
        weights_input.dtype, jnp.complexfloating
    ):
        raise TypeError("T-spline source parameters must be real.")
    points = jnp.asarray(points_input, dtype=jnp.float64)
    raw_weights = jnp.asarray(weights_input, dtype=jnp.float64)
    source_shape = (surface.source_mesh.coefficient_count, 3)
    if points.shape != source_shape or raw_weights.shape != source_shape[:1]:
        raise ValueError(
            "Parameter banks must match the exact source mesh and 3D carrier."
        )
    points = eqx.error_if(
        points, jnp.any(~jnp.isfinite(points)), "T-spline source points must be finite."
    )
    raw_weights = eqx.error_if(
        raw_weights,
        jnp.any(~jnp.isfinite(raw_weights)) | jnp.any(raw_weights <= 0.0),
        "T-spline source weights must be finite and positive.",
    )
    if (
        not isinstance(points, jax_core.Tracer)
        and not isinstance(raw_weights, jax_core.Tracer)
        and _numeric_bank_id(points, raw_weights) != source_numeric_bank_id
    ):
        raise ValueError(
            "Concrete parameter changes require rebind_numeric_bank and a fresh exact source revision."
        )
    for transfer in surface.transfer_plans:
        points, raw_weights = transfer.apply_homogeneous_geometry(points, raw_weights)
    return points, raw_weights


def _geometry_certificate_state_id(
    control_points: Sequence[Sequence[Fraction]],
    weights: Sequence[Fraction],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "asts2d-exact-binary-rational-geometry-bank",
            "control_points": [_fraction_payload(row) for row in control_points],
            "weights": _fraction_payload(weights),
        }
    )


def _numeric_geometry_certificate_state_id(
    control_points: object, weights: object, /
) -> str:
    points = np.asarray(control_points)
    raw_weights = np.asarray(weights)
    exact_points = tuple(
        tuple(Fraction.from_float(float(value)) for value in row) for row in points
    )
    exact_weights = tuple(Fraction.from_float(float(value)) for value in raw_weights)
    return _geometry_certificate_state_id(exact_points, exact_weights)


class TSplineGeometryState(StrictModule):
    """IGA view of one authoritative CAD T-spline source and its numeric realization."""

    surface_patch: TSplineSurfacePatch
    revision: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    numeric_bank_id: str = eqx.field(static=True)
    source_numeric_bank_id: str = eqx.field(static=True)
    geometry_state_id: str = eqx.field(static=True)
    maximum_conversion_error: float = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: TMesh2D | TSplineSurfacePatch,
        control_points: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        anchor_ids: Sequence[int] | None = None,
        /,
        *,
        revision: str = "0",
        exact_homogeneous_control_points: Sequence[Sequence[Fraction]] | None = None,
    ) -> None:
        revision_ = str(revision).strip()
        if not revision_:
            raise ValueError("revision must be non-empty.")
        if isinstance(mesh, TSplineSurfacePatch):
            if (
                control_points is not None
                or weights is not None
                or anchor_ids is not None
                or exact_homogeneous_control_points is not None
            ):
                raise ValueError(
                    "A TSplineSurfacePatch already owns its source and exact coefficient banks."
                )
            surface = mesh
        else:
            if not isinstance(mesh, TMesh2D):
                raise TypeError("mesh must be TMesh2D or TSplineSurfacePatch.")
            if control_points is None or weights is None or anchor_ids is None:
                raise ValueError(
                    "Authoring geometry requires control points, raw weights, and persistent anchor IDs."
                )
            ids = tuple(int(value) for value in anchor_ids)
            mesh_ids = tuple(anchor.anchor_id for anchor in mesh.anchors)
            if len(ids) != len(set(ids)) or set(ids) != set(mesh_ids):
                raise ValueError(
                    "Geometry anchor IDs must exactly cover the T-mesh anchors."
                )
            points_in = jnp.asarray(control_points)
            weights_in = jnp.asarray(weights)
            if points_in.ndim != 2 or points_in.shape[0] != len(ids):
                raise ValueError(
                    "Control points must have one row per explicit anchor ID."
                )
            if weights_in.shape != (len(ids),) or points_in.shape[1] != 3:
                raise ValueError(
                    "A CAD-backed T-spline surface requires three coordinates and one weight per anchor."
                )
            if jnp.issubdtype(points_in.dtype, jnp.complexfloating) or jnp.issubdtype(
                weights_in.dtype, jnp.complexfloating
            ):
                raise TypeError("T-spline geometry values must be real.")
            order = tuple(ids.index(anchor_id) for anchor_id in mesh_ids)
            points = points_in[jnp.asarray(order, dtype=jnp.int32)]
            raw_weights = weights_in[jnp.asarray(order, dtype=jnp.int32)]
            dtype = inexact_result_type(points, raw_weights)
            points = points.astype(dtype)
            raw_weights = raw_weights.astype(dtype)
            points = eqx.error_if(
                points,
                jnp.any(~jnp.isfinite(points)),
                "T-spline control points must be finite.",
            )
            raw_weights = eqx.error_if(
                raw_weights,
                jnp.any(~jnp.isfinite(raw_weights)) | jnp.any(raw_weights <= 0.0),
                "T-spline raw weights must be finite and strictly positive.",
            )
            points_host = np.asarray(points)
            weights_host = np.asarray(raw_weights)
            if exact_homogeneous_control_points is None:
                exact = _exact_bank_from_numeric(points_host, weights_host)
            else:
                external = _exact_rows(exact_homogeneous_control_points)
                if len(external) != len(ids) or any(len(row) != 4 for row in external):
                    raise ValueError(
                        "Exact homogeneous controls must match the source 3D geometry bank."
                    )
                exact = tuple(external[index] for index in order)
                if any(row[-1] <= 0 for row in exact):
                    raise ValueError(
                        "Exact homogeneous weights must be strictly positive."
                    )
            exact_weights = tuple(row[-1] for row in exact)
            exact_points = tuple(
                tuple(value / row[-1] for value in row[:-1]) for row in exact
            )
            surface = TSplineSurfacePatch(
                mesh,
                points_host,
                weights_host,
                exact_control_points=exact_points,
                exact_weights=exact_weights,
            )

        source_numeric_bank_id = _source_numeric_bank_id(surface)
        target_points, target_weights = surface._current_geometry()
        points_host = np.asarray(target_points)
        weights_host = np.asarray(target_weights)
        target_mesh = surface.mesh
        anchor_ids_ = tuple(anchor.anchor_id for anchor in target_mesh.anchors)
        layout_id = canonical_fingerprint(
            {
                "kind": "asts2d-cad-geometry-layout",
                "mesh": target_mesh.mesh_id,
                "anchor_ids": list(anchor_ids_),
                "ambient_dimension": points_host.shape[1],
            }
        )
        numeric_bank_id = _numeric_bank_id(points_host, weights_host)
        maximum_error = surface.maximum_homogeneous_realization_error
        state_id = canonical_fingerprint(
            {
                "kind": "asts2d-cad-geometry-state",
                "source_definition_id": surface.source_definition_id,
                "source_numeric_bank_id": source_numeric_bank_id,
                "source_mesh": surface.source_mesh.mesh_id,
                "target_mesh": target_mesh.mesh_id,
                "transfer_plans": [plan.plan_id for plan in surface.transfer_plans],
                "exact_source_control_points": [
                    _fraction_payload(row) for row in surface.exact_source_control_points
                ],
                "exact_source_weights": _fraction_payload(surface.exact_source_weights),
                "exact_target_homogeneous": [
                    _fraction_payload(row) for row in surface.exact_target_homogeneous
                ],
                "numeric_bank_id": numeric_bank_id,
                "numeric_control_points": array_tree_fingerprint(points_host),
                "numeric_raw_weights": array_tree_fingerprint(weights_host),
                "maximum_homogeneous_realization_error": maximum_error,
                "geometry_layout": layout_id,
                "revision": revision_,
            }
        )
        self.surface_patch = surface
        self.revision = revision_
        self.geometry_layout_id = layout_id
        self.numeric_bank_id = numeric_bank_id
        self.source_numeric_bank_id = source_numeric_bank_id
        self.geometry_state_id = state_id
        self.maximum_conversion_error = maximum_error

    @property
    def mesh(self) -> TMesh2D:
        return self.surface_patch.mesh

    @property
    def anchor_ids(self) -> tuple[int, ...]:
        return tuple(anchor.anchor_id for anchor in self.mesh.anchors)

    @property
    def source_definition_id(self) -> str:
        """Return this target basis and exact transfer-chain identity."""
        return self.surface_patch.source_definition_id

    @property
    def source_authority_id(self) -> str:
        """Return the exact root CAD authority shared by its refinements."""
        return self.surface_patch.source_authority_id

    @property
    def control_points(self) -> Array:
        return self.geometry_bank()[0]

    @property
    def weights(self) -> Array:
        return self.geometry_bank()[1]

    def geometry_bank(self, /) -> tuple[Array, Array]:
        return self.realize_parameter_bank(
            self.surface_patch.control_points,
            self.surface_patch.weights,
            source_authority_id=self.surface_patch.source_authority_id,
        )

    def realize_parameter_bank(
        self,
        control_points: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        source_authority_id: str,
    ) -> tuple[Array, Array]:
        """Realize root control/weight inputs under this exact source authority."""
        return _realize_bound_source_bank(
            self.surface_patch,
            control_points,
            weights,
            self.source_numeric_bank_id,
            source_authority_id,
        )

    def rebind_numeric_bank(
        self,
        control_points: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        revision: str,
    ) -> TSplineGeometryState:
        """Author a fresh exact source map for concrete same-topology numeric updates."""
        candidate = TSplineGeometryState(
            self.mesh,
            control_points,
            weights,
            self.anchor_ids,
            revision=revision,
        )
        if (
            candidate.source_definition_id != self.source_definition_id
            and candidate.revision == self.revision
        ):
            raise ValueError("Changed exact source data requires a new source revision.")
        return candidate

    @property
    def exact_homogeneous_control_points(self) -> tuple[tuple[Fraction, ...], ...]:
        return self.surface_patch.exact_target_homogeneous

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def exact_weights(self) -> tuple[Fraction, ...]:
        return tuple(row[-1] for row in self.exact_homogeneous_control_points)

    @property
    def exact_control_points(self) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(
            tuple(value / row[-1] for value in row[:-1])
            for row in self.exact_homogeneous_control_points
        )

    def prolong(self, refinement: ASTSRefinement, /) -> TSplineGeometryState:
        if refinement.source_mesh.mesh_id != self.mesh.mesh_id:
            raise ValueError("Refinement relation does not start at this geometry basis.")
        source = self.surface_patch
        target_patch = TSplineSurfacePatch(
            refinement.refined_mesh,
            source.control_points,
            source.weights,
            exact_control_points=source.exact_source_control_points,
            exact_weights=source.exact_source_weights,
            source_mesh=source.source_mesh,
            transfer_plans=(*source.transfer_plans, refinement.transfer_plan),
            source_basis_certificate=source.source_basis_certificate,
            basis_certificate=refinement.certificate,
            transfer_meshes=(*source.transfer_meshes, refinement.refined_mesh),
            transfer_basis_certificates=(
                *source.transfer_basis_certificates,
                refinement.certificate,
            ),
        )
        return TSplineGeometryState(
            target_patch,
            revision=f"{self.revision}:refined:{refinement.refinement_id}",
        )


class TSplineRuntimeData(StrictModule):
    """Fixed-topology runtime bound to one CAD source definition and numeric bank."""

    surface_patch: TSplineSurfacePatch
    exact_geometry_certificate: TSplineGeometryCertificate = eqx.field(static=True)
    geometry_certificate: TSplineGeometryCertificate = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    source_definition_id: str = eqx.field(static=True)
    source_numeric_bank_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    numeric_bank_id: str = eqx.field(static=True)
    geometry_state_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: TSplineGeometryState,
        /,
        *,
        numeric_version: str,
        exact_geometry_certificate: TSplineGeometryCertificate | None = None,
        geometry_certificate: TSplineGeometryCertificate | None = None,
    ) -> None:
        version = str(numeric_version).strip()
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        realization = geometry.surface_patch.realization
        basis = realization.basis_certificate
        control_points, weights = geometry.geometry_bank()
        exact_id = _geometry_certificate_state_id(
            geometry.exact_control_points, geometry.exact_weights
        )
        numeric_id = _numeric_geometry_certificate_state_id(control_points, weights)
        computed_exact_certificate = certify_tspline_geometry(
            realization,
            np.asarray(geometry.exact_control_points, dtype=object),
            np.asarray(geometry.exact_weights, dtype=object),
        )
        if exact_geometry_certificate is not None and (
            not isinstance(exact_geometry_certificate, TSplineGeometryCertificate)
            or exact_geometry_certificate.qualification_payload()
            != computed_exact_certificate.qualification_payload()
        ):
            raise ValueError(
                "Retained exact geometry evidence does not match recomputed source data."
            )
        exact_certificate = computed_exact_certificate
        computed_numeric_certificate = certify_tspline_geometry(
            realization, np.asarray(control_points), np.asarray(weights)
        )
        if geometry_certificate is not None and (
            not isinstance(geometry_certificate, TSplineGeometryCertificate)
            or geometry_certificate.qualification_payload()
            != computed_numeric_certificate.qualification_payload()
        ):
            raise ValueError(
                "Retained numeric geometry evidence does not match recomputed source data."
            )
        numeric_certificate = computed_numeric_certificate
        for certificate, expected_state_id in (
            (exact_certificate, exact_id),
            (numeric_certificate, numeric_id),
        ):
            if (
                not isinstance(certificate, TSplineGeometryCertificate)
                or certificate.mesh_id != geometry.mesh.mesh_id
                or certificate.basis_certificate_id != basis.certificate_id
                or certificate.numeric_state_id != expected_state_id
                or not certificate.passed
            ):
                raise ValueError(
                    "TSplineRuntimeData requires valid exact-source and numeric map certificates."
                )
        self.surface_patch = geometry.surface_patch
        self.exact_geometry_certificate = exact_certificate
        self.geometry_certificate = numeric_certificate
        self.mesh_id = geometry.mesh.mesh_id
        self.geometry_layout_id = geometry.geometry_layout_id
        self.source_definition_id = geometry.source_definition_id
        self.source_numeric_bank_id = geometry.source_numeric_bank_id
        self.source_revision = geometry.revision
        self.numeric_version = version
        self.numeric_bank_id = geometry.numeric_bank_id
        self.geometry_state_id = geometry.geometry_state_id
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "asts2d-cad-runtime-data",
                "source_definition_id": geometry.source_definition_id,
                "source_numeric_bank_id": geometry.source_numeric_bank_id,
                "geometry_state": geometry.geometry_state_id,
                "exact_geometry_certificate": exact_certificate.certificate_id,
                "numeric_geometry_certificate": numeric_certificate.certificate_id,
                "numeric_version": version,
            }
        )

    def geometry_bank(self, /) -> tuple[Array, Array]:
        return self.realize_parameter_bank(
            self.surface_patch.control_points,
            self.surface_patch.weights,
            source_authority_id=self.surface_patch.source_authority_id,
        )

    def realize_parameter_bank(
        self,
        control_points: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        source_authority_id: str,
    ) -> tuple[Array, Array]:
        """Realize a design perturbation under this root source authority."""
        return _realize_bound_source_bank(
            self.surface_patch,
            control_points,
            weights,
            self.source_numeric_bank_id,
            source_authority_id,
        )

    @property
    def source_authority_id(self) -> str:
        """Return the exact root CAD authority bound to this runtime."""
        return self.surface_patch.source_authority_id

    @property
    def control_points(self) -> Array:
        return self.geometry_bank()[0]

    @property
    def weights(self) -> Array:
        return self.geometry_bank()[1]

    @property
    def exact_weights(self) -> tuple[Fraction, ...]:
        return tuple(row[-1] for row in self.exact_homogeneous_control_points)

    @property
    def exact_control_points(self) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(
            tuple(value / row[-1] for value in row[:-1])
            for row in self.exact_homogeneous_control_points
        )

    @property
    def exact_homogeneous_control_points(self) -> tuple[tuple[Fraction, ...], ...]:
        return self.surface_patch.exact_target_homogeneous


class TSplineQueryRoute(StrictModule, NonTrainableState):
    """Host-validated trim membership and persistent cell routes for JAX queries."""

    parameters: Array
    cell_ids: Array
    prepared_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    trim_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: ArrayLike,
        cell_ids: ArrayLike,
        /,
        *,
        prepared_id: str,
        mesh_id: str,
        trim_id: str,
    ) -> None:
        points_input = np.asarray(parameters)
        if not np.issubdtype(points_input.dtype, np.number) or np.iscomplexobj(
            points_input
        ):
            raise TypeError("ASTS query parameters must be real numeric coordinates.")
        points = np.asarray(points_input, dtype=np.float64)
        cells = np.asarray(cell_ids)
        if points.ndim != 2 or points.shape[1] != 2 or np.any(~np.isfinite(points)):
            raise ValueError(
                "ASTS query parameters must be finite with shape (point, 2)."
            )
        if cells.shape != points.shape[:1] or not np.issubdtype(cells.dtype, np.integer):
            raise ValueError(
                "ASTS query routes require one integer persistent cell ID per point."
            )
        if cells.dtype.kind == "u" and np.any(cells > np.iinfo(np.int64).max):
            raise ValueError(
                "ASTS query cell IDs must fit the persistent signed 64-bit domain."
            )
        identifiers = tuple(
            str(value).strip() for value in (prepared_id, mesh_id, trim_id)
        )
        if any(not value for value in identifiers):
            raise ValueError("ASTS query route identity fields must be non-empty.")
        cells = np.asarray(cells, dtype=np.int64)
        route_id = canonical_fingerprint(
            {
                "kind": "asts2d-fixed-query-route",
                "prepared": identifiers[0],
                "mesh": identifiers[1],
                "trim": identifiers[2],
                "parameters": array_tree_fingerprint(points),
                "cell_ids": array_tree_fingerprint(cells),
            }
        )
        self.parameters = jnp.asarray(points)
        self.cell_ids = jnp.asarray(cells)
        self.prepared_id, self.mesh_id, self.trim_id = identifiers
        self.route_id = route_id


class TSplineH1FieldSpec(StrictModule):
    """Scalar H1 field on the ASTS basis with its independent raw weight bank."""

    name: str = eqx.field(static=True)
    weights: Array | None
    weights_from_geometry: bool = eqx.field(static=True)
    exact_weights: tuple[Fraction, ...] | None = eqx.field(static=True)
    maximum_conversion_error: float = eqx.field(static=True)
    field_spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        weights: ArrayLike | None = None,
        /,
        *,
        weights_from_geometry: bool = False,
        exact_weights: Sequence[Fraction] | None = None,
        anchor_count: int | None = None,
    ) -> None:
        name_ = str(name).strip()
        dynamic = bool(weights_from_geometry)
        if not name_:
            raise ValueError("T-spline field name must be non-empty.")
        if dynamic and weights is not None:
            raise ValueError("Geometry-owned field weights cannot also be supplied.")
        if dynamic and exact_weights is not None:
            raise ValueError("Geometry-owned field weights have no independent bank.")
        weights_ = None if weights is None else jnp.asarray(weights)
        conversion_error = 0.0
        if weights_ is not None:
            if weights_.ndim != 1 or jnp.issubdtype(weights_.dtype, jnp.complexfloating):
                raise ValueError("Scalar T-spline field weights must be a real vector.")
            if anchor_count is not None and weights_.shape != (int(anchor_count),):
                raise ValueError(
                    "Field weights must contain one value per T-spline anchor."
                )
            weights_ = eqx.error_if(
                weights_,
                jnp.any(~jnp.isfinite(weights_)) | jnp.any(weights_ <= 0.0),
                "Scalar T-spline field weights must be finite and positive.",
            )
            conversion_error = 0.0
        exact = None
        if weights_ is not None:
            if exact_weights is None:
                host = np.asarray(weights_)
                exact = tuple(Fraction.from_float(float(value)) for value in host)
            else:
                exact = tuple(Fraction(value) for value in exact_weights)
                if len(exact) != weights_.shape[0] or any(value <= 0 for value in exact):
                    raise ValueError(
                        "Exact field weights must be positive and basis-aligned."
                    )
                errors = tuple(
                    abs(Fraction.from_float(float(value)) - exact_value)
                    for value, exact_value in zip(
                        np.asarray(weights_), exact, strict=True
                    )
                )
                maximum = max(errors, default=Fraction(0))
                conversion_error = float(maximum)
                if not isfinite(conversion_error):
                    raise ValueError(
                        "Exact field weights have no finite numeric realization."
                    )
                if maximum and Fraction(conversion_error) < maximum:
                    conversion_error = float(np.nextafter(conversion_error, np.inf))
        self.name = name_
        self.weights = weights_
        self.weights_from_geometry = dynamic
        self.exact_weights = exact
        self.maximum_conversion_error = conversion_error
        self.field_spec_id = canonical_fingerprint(
            {
                "kind": "asts2d-scalar-h1-field",
                "name": name_,
                "weight_kind": "geometry"
                if dynamic
                else "polynomial"
                if exact is None
                else "rational",
                "exact_weights": None if exact is None else _fraction_payload(exact),
                "numeric_weights": None
                if weights_ is None
                else array_tree_fingerprint(np.asarray(weights_)),
                "maximum_conversion_error": conversion_error,
            }
        )


class TSplineReferenceActions(LocalReferenceActions):
    """Fixed-route sparse ASTS basis values, jets, and exact algebraic transposes."""

    polynomial_values: Array
    polynomial_gradients: Array
    polynomial_hessians: Array
    coefficient_rows: Array
    field_weights: Array | None
    unit_weights: Array
    field_weights_from_geometry: bool = eqx.field(static=True)
    query_shape: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    local_width: int = eqx.field(static=True)
    global_coefficient_count: int = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    kernel_modes: tuple[str, ...] = eqx.field(static=True)
    is_trace: bool = eqx.field(static=True)

    def __init__(
        self,
        polynomial_values: ArrayLike,
        polynomial_gradients: ArrayLike,
        polynomial_hessians: ArrayLike,
        coefficient_rows: ArrayLike,
        field_weights: ArrayLike | None,
        /,
        *,
        field_weights_from_geometry: bool,
        query_shape: Sequence[int],
        topology_id: str,
        geometry_layout_id: str,
        structural_id: str,
        global_coefficient_count: int,
        maximum_derivative_order: int,
        is_trace: bool,
    ) -> None:
        values = jnp.asarray(polynomial_values)
        gradients = jnp.asarray(polynomial_gradients)
        hessians = jnp.asarray(polynomial_hessians)
        rows = jnp.asarray(coefficient_rows, dtype=jnp.int32)
        weights = None if field_weights is None else jnp.asarray(field_weights)
        shape = tuple(int(value) for value in query_shape)
        if (
            values.ndim != 3
            or gradients.shape != values.shape + (2,)
            or hessians.shape != values.shape + (2, 2)
            or rows.shape != (values.shape[0], values.shape[2])
            or values.shape[1] != int(np.prod(shape))
        ):
            raise ValueError("ASTS local basis arrays have incompatible fixed routes.")
        if bool(field_weights_from_geometry) and weights is not None:
            raise ValueError(
                "Geometry-owned field weights cannot be static action leaves."
            )
        if weights is not None and (
            weights.ndim != 1
            or weights.shape != (int(global_coefficient_count),)
            or np.iscomplexobj(np.asarray(weights))
            or np.any(~np.isfinite(np.asarray(weights)))
            or np.any(np.asarray(weights) <= 0.0)
        ):
            raise ValueError(
                "Rational field weights must be a finite positive basis vector."
            )
        self.polynomial_values = values
        self.polynomial_gradients = gradients
        self.polynomial_hessians = hessians
        self.coefficient_rows = rows
        self.unit_weights = jnp.ones(rows.shape, dtype=values.dtype)
        self.field_weights = weights
        self.field_weights_from_geometry = bool(field_weights_from_geometry)
        self.query_shape = shape
        self.topology_id = str(topology_id)
        self.geometry_layout_id = str(geometry_layout_id)
        self.realization_id = "asts2d-local-extracted-bernstein"
        self.local_width = values.shape[2]
        self.point_count = values.shape[1]
        self.global_coefficient_count = int(global_coefficient_count)
        self.maximum_derivative_order = int(maximum_derivative_order)
        self.kernel_modes = ("dense",)
        self.is_trace = bool(is_trace)
        self.action_id = canonical_fingerprint(
            {
                "kind": "asts2d-reference-actions",
                "structural": str(structural_id),
                "topology": self.topology_id,
                "geometry_layout": self.geometry_layout_id,
                "coefficient_rows": array_tree_fingerprint(np.asarray(rows)),
                "polynomial_values": array_tree_fingerprint(np.asarray(values)),
                "field_weights_from_geometry": self.field_weights_from_geometry,
                "field_weights": None
                if weights is None
                else array_tree_fingerprint(np.asarray(weights)),
                "maximum_derivative_order": self.maximum_derivative_order,
                "trace": self.is_trace,
            }
        )

    def _weights(self, runtime: object, /) -> Array:
        if self.field_weights_from_geometry:
            if not isinstance(runtime, TSplineRuntimeData):
                raise TypeError("ASTS rational actions require TSplineRuntimeData.")
            values = runtime.weights
        elif self.field_weights is not None:
            values = self.field_weights
        else:
            return self.unit_weights
        if values.shape != (self.global_coefficient_count,):
            raise ValueError("ASTS rational weights must match the global anchor bank.")
        local = values[self.coefficient_rows]
        return eqx.error_if(
            local,
            jnp.any(~jnp.isfinite(local)) | jnp.any(local <= 0.0),
            "ASTS field weights must be finite and positive.",
        )

    def _basis_jets(self, runtime: object, /) -> tuple[Array, Array, Array]:
        weights = self._weights(runtime)
        weighted_values = self.polynomial_values * weights[:, None, :]
        denominator = jnp.sum(weighted_values, axis=-1)
        denominator = eqx.error_if(
            denominator,
            jnp.any(~jnp.isfinite(denominator)) | jnp.any(denominator <= 0.0),
            "ASTS rational field denominator must be finite and positive.",
        )
        rational_values = weighted_values / denominator[..., None]
        empty = jnp.empty((0,), dtype=rational_values.dtype)
        if self.maximum_derivative_order == 0:
            return rational_values, empty, empty
        weighted_gradients = self.polynomial_gradients * weights[:, None, :, None]
        denominator_gradients = jnp.sum(weighted_gradients, axis=2)
        rational_gradients = (
            weighted_gradients
            - rational_values[..., None] * denominator_gradients[:, :, None, :]
        ) / denominator[:, :, None, None]
        if self.maximum_derivative_order == 1:
            return rational_values, rational_gradients, empty
        weighted_hessians = self.polynomial_hessians * weights[:, None, :, None, None]
        denominator_hessians = jnp.sum(weighted_hessians, axis=2)
        rational_hessians = (
            weighted_hessians
            - rational_values[..., None, None] * denominator_hessians[:, :, None, :, :]
            - rational_gradients[..., :, None]
            * denominator_gradients[:, :, None, None, :]
            - rational_gradients[..., None, :]
            * denominator_gradients[:, :, None, :, None]
        ) / denominator[:, :, None, None, None]
        return rational_values, rational_gradients, rational_hessians

    def realize_reference_actions(self, runtime: object, /) -> TSplineReferenceActions:
        if not isinstance(runtime, TSplineRuntimeData):
            raise TypeError("ASTS actions require TSplineRuntimeData.")
        if (
            runtime.mesh_id != self.topology_id
            or runtime.geometry_layout_id != self.geometry_layout_id
        ):
            raise ValueError("ASTS runtime does not match the fixed basis realization.")
        self._basis_jets(runtime)
        return self

    def interpolate(self, runtime: object, local_coefficients: ArrayLike, /) -> Array:
        coefficients = jnp.asarray(local_coefficients)
        if coefficients.shape[:2] != (self.polynomial_values.shape[0], self.local_width):
            raise ValueError("ASTS local coefficients do not match their cell gathers.")
        return contract("eql,el...->eq...", self._basis_jets(runtime)[0], coefficients)

    def interpolate_transpose(self, runtime: object, values: ArrayLike, /) -> Array:
        values_ = jnp.asarray(values)
        if values_.shape[:2] != (self.polynomial_values.shape[0], self.point_count):
            raise ValueError("ASTS point values do not match their fixed route.")
        return contract("eql,eq...->el...", self._basis_jets(runtime)[0], values_)

    def reference_gradient(
        self, runtime: object, local_coefficients: ArrayLike, /
    ) -> Array:
        if self.maximum_derivative_order < 1:
            raise ValueError(
                "ASTS reference gradients were not prepared for this region."
            )
        coefficients = jnp.asarray(local_coefficients)
        return contract("eqlr,el...->eq...r", self._basis_jets(runtime)[1], coefficients)

    def reference_gradient_transpose(
        self, runtime: object, gradients: ArrayLike, /
    ) -> Array:
        if self.maximum_derivative_order < 1:
            raise ValueError(
                "ASTS reference gradients were not prepared for this region."
            )
        gradients_ = jnp.asarray(gradients)
        return contract("eqlr,eq...r->el...", self._basis_jets(runtime)[1], gradients_)

    def reference_hessian(
        self, runtime: object, local_coefficients: ArrayLike, /
    ) -> Array:
        if self.maximum_derivative_order < 2:
            raise ValueError("ASTS reference Hessians were not prepared for this region.")
        coefficients = jnp.asarray(local_coefficients)
        return contract(
            "eqlrs,el...->eq...rs", self._basis_jets(runtime)[2], coefficients
        )

    def reference_hessian_transpose(
        self, runtime: object, hessians: ArrayLike, /
    ) -> Array:
        if self.maximum_derivative_order < 2:
            raise ValueError("ASTS reference Hessians were not prepared for this region.")
        hessians_ = jnp.asarray(hessians)
        return contract("eqlrs,eq...rs->el...", self._basis_jets(runtime)[2], hessians_)

    def trace(self, runtime: object, local_coefficients: ArrayLike, /) -> Array:
        if not self.is_trace:
            raise ValueError("ASTS trace actions require a trim-boundary region.")
        return self.interpolate(runtime, local_coefficients)

    def trace_transpose(self, runtime: object, values: ArrayLike, /) -> Array:
        if not self.is_trace:
            raise ValueError("ASTS trace actions require a trim-boundary region.")
        return self.interpolate_transpose(runtime, values)


class TSplineGeometryActions(LocalGeometryActions):
    """Differentiable surface metric actions for one fixed ASTS query route."""

    realization: LocalExtractedBernsteinRealization
    points: Array
    parameter_weights: Array
    cell_ids: Array
    tangents: Array | None
    conormals: Array | None
    runtime_layout_id: str = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    entity_count: int = eqx.field(static=True)
    domain_kind: str = eqx.field(static=True)
    linear_solve_plan: SmallLinearSolvePlan = eqx.field(static=True)

    def __init__(
        self,
        realization: LocalExtractedBernsteinRealization,
        points: ArrayLike,
        parameter_weights: ArrayLike,
        cell_ids: ArrayLike,
        /,
        *,
        runtime_layout_id: str,
        domain_kind: str,
        structural_id: str,
        tangents: ArrayLike | None = None,
        conormals: ArrayLike | None = None,
    ) -> None:
        points_ = jnp.asarray(points, dtype=jnp.float64)
        weights_ = jnp.asarray(parameter_weights, dtype=jnp.float64)
        cells = jnp.asarray(cell_ids, dtype=jnp.int64)
        if points_.ndim != 3 or points_.shape[-1] != 2:
            raise ValueError("ASTS geometry points must have shape (entity, point, 2).")
        if weights_.shape != points_.shape[:2] or cells.shape != (points_.shape[0],):
            raise ValueError(
                "ASTS surface geometry routes do not match their point arrays."
            )
        tangent_ = None if tangents is None else jnp.asarray(tangents, dtype=jnp.float64)
        conormal_ = (
            None if conormals is None else jnp.asarray(conormals, dtype=jnp.float64)
        )
        if (tangent_ is None) != (conormal_ is None):
            raise ValueError("Boundary geometry requires both tangents and UV conormals.")
        if tangent_ is not None and (
            tangent_.shape != (points_.shape[0], 2)
            or conormal_ is None
            or conormal_.shape != tangent_.shape
        ):
            raise ValueError("ASTS boundary tangent/conormal routes are malformed.")
        kind = str(domain_kind)
        if kind not in ("cell", "exterior_facet"):
            raise ValueError("ASTS geometry actions support cells and trim facets.")
        self.realization = realization
        self.points = points_
        self.parameter_weights = weights_
        self.cell_ids = cells
        self.tangents = tangent_
        self.conormals = conormal_
        self.runtime_layout_id = str(runtime_layout_id)
        self.entity_count = points_.shape[0]
        self.linear_solve_plan = SmallLinearSolvePlan(2)
        self.domain_kind = kind
        self.action_id = canonical_fingerprint(
            {
                "kind": "asts2d-geometry-actions",
                "structural": str(structural_id),
                "realization": realization.realization_id,
                "runtime_layout": self.runtime_layout_id,
                "domain_kind": kind,
                "points": array_tree_fingerprint(np.asarray(points_)),
                "parameter_weights": array_tree_fingerprint(np.asarray(weights_)),
                "cell_ids": tuple(int(value) for value in np.asarray(cells)),
                "tangents": None
                if tangent_ is None
                else array_tree_fingerprint(np.asarray(tangent_)),
                "conormals": None
                if conormal_ is None
                else array_tree_fingerprint(np.asarray(conormal_)),
            }
        )

    def realize(self, runtime: object, /) -> LocalMetricResult:
        if not isinstance(runtime, TSplineRuntimeData):
            raise TypeError("ASTS geometry actions require TSplineRuntimeData.")
        if runtime.geometry_layout_id != self.runtime_layout_id:
            raise ValueError("ASTS geometry runtime does not match the prepared layout.")
        entity_count, point_count = self.points.shape[:2]
        flat_points = self.points.reshape((-1, 2))
        flat_cells = jnp.repeat(self.cell_ids, point_count)
        control_points, raw_weights = runtime.geometry_bank()
        jet = self.realization.evaluate_rational(
            flat_points,
            control_points,
            raw_weights,
            cell_ids=flat_cells,
            maximum_order=1,
        )
        ambient = control_points.shape[1]
        if jet.gradients is None:
            raise ValueError(
                "ASTS geometry actions require first-order rational gradients."
            )
        points = jet.values.reshape((entity_count, point_count, ambient))
        jacobian = jet.gradients.reshape((entity_count, point_count, ambient, 2))
        metric = contract("eqdi,eqdj->eqij", jacobian, jacobian)
        inverse_result = inverse_small_linear(self.linear_solve_plan, metric)
        metric_inverse = eqx.error_if(
            inverse_result.value,
            jnp.any(~inverse_result.successful)
            | jnp.any(inverse_result.determinant <= 0.0),
            "ASTS physical surface metric is singular or non-positive.",
        )
        inverse_jacobian = contract("eqij,eqdj->eqid", metric_inverse, jacobian)
        surface_scale = jnp.sqrt(inverse_result.determinant)
        normals = None
        if self.domain_kind == "cell":
            physical_weights = self.parameter_weights * surface_scale
        else:
            if self.tangents is None or self.conormals is None:
                raise ValueError(
                    "ASTS trim-boundary geometry is missing its oriented frame."
                )
            tangent_jacobian = contract("eqdr,er->eqd", jacobian, self.tangents)
            tangent_scale = jnp.linalg.norm(tangent_jacobian, axis=-1)
            conormal = contract(
                "eqdr,eqrs,es->eqd",
                jacobian,
                metric_inverse,
                self.conormals,
            )
            conormal_scale = jnp.linalg.norm(conormal, axis=-1)
            safe_scale = jnp.where(conormal_scale > 0.0, conormal_scale, 1.0)
            normals = conormal / safe_scale[..., None]
            physical_weights = self.parameter_weights * tangent_scale
        valid = jnp.all(inverse_result.successful, axis=1)
        valid = valid & jnp.all(jnp.isfinite(physical_weights), axis=1)
        return LocalMetricResult(
            points,
            physical_weights,
            jacobian,
            inverse_jacobian,
            normals=normals,
            valid=valid,
        )


def _cell_complex_topology(
    mesh: TMesh2D,
) -> tuple[CellComplexTopology, EntitySet, EntitySet, EntitySet]:
    vertices = EntitySet(
        "tmesh-vertices",
        0,
        tuple(vertex.vertex_id for vertex in mesh.vertices),
        entity_set_id=canonical_fingerprint(
            {"kind": "asts2d-vertex-entities", "mesh": mesh.mesh_id}
        ),
    )
    edges = EntitySet(
        "tmesh-edges",
        1,
        tuple(edge.edge_id for edge in mesh.edges),
        entity_set_id=canonical_fingerprint(
            {"kind": "asts2d-edge-entities", "mesh": mesh.mesh_id}
        ),
    )
    cells = EntitySet(
        "tmesh-cells",
        2,
        tuple(cell.cell_id for cell in mesh.cells),
        entity_set_id=canonical_fingerprint(
            {"kind": "asts2d-cell-entities", "mesh": mesh.mesh_id}
        ),
    )
    edge_source: list[int] = []
    edge_target: list[int] = []
    edge_signs: list[int] = []
    for edge in mesh.edges:
        first, second = edge.vertex_ids
        edge_source.extend((first, second))
        edge_target.extend((edge.edge_id, edge.edge_id))
        edge_signs.extend((-1, 1))
    vertex_edge = EdgeRelation(
        edge_source,
        edge_target,
        source_size=len(mesh.vertices),
        target_size=len(mesh.edges),
    )
    boundary_vertices = OrientedIncidence(
        1, vertices, edges, vertex_edge, np.asarray(edge_signs, dtype=np.int8)
    )
    edge_rows = {edge.edge_id: row for row, edge in enumerate(mesh.edges)}
    face_source: list[int] = []
    face_target: list[int] = []
    face_signs: list[int] = []
    for cell_row, cell in enumerate(mesh.cells):
        u0, u1, v0, v1 = cell.parameter_bounds
        for edge_id in cell.edge_ids:
            edge = mesh.edges[edge_rows[edge_id]]
            first = mesh.vertices[edge.vertex_ids[0]].parameter
            if edge.axis == "u":
                sign = 1 if first[1] == v0 else -1
            else:
                sign = 1 if first[0] == u1 else -1
            face_source.append(edge_rows[edge_id])
            face_target.append(cell_row)
            face_signs.append(sign)
    edge_cell = EdgeRelation(
        face_source,
        face_target,
        source_size=len(mesh.edges),
        target_size=len(mesh.cells),
    )
    boundary_cells = OrientedIncidence(
        2, edges, cells, edge_cell, np.asarray(face_signs, dtype=np.int8)
    )
    topology = CellComplexTopology(
        (vertices, edges, cells),
        (boundary_vertices, boundary_cells),
        topology_id=mesh.mesh_id,
    )
    anchors = EntitySet(
        "tmesh-anchors",
        0,
        tuple(anchor.anchor_id for anchor in mesh.anchors),
        entity_set_id=canonical_fingerprint(
            {"kind": "asts2d-anchor-entities", "mesh": mesh.mesh_id}
        ),
    )
    return topology, cells, edges, anchors


def _identity_parameter_map(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64)


def _identity_parameter_jacobian(points: np.ndarray) -> np.ndarray:
    return np.broadcast_to(np.eye(2), (points.shape[0], 2, 2))


def _trim_segments(
    mesh: TMesh2D,
    realization: LocalExtractedBernsteinRealization,
    trim: UVTrimmedSurface,
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...],
]:
    pieces: dict[
        tuple[str, int, Fraction, Fraction],
        tuple[int, tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    ] = {}
    loops = (trim.outer,) + trim.holes
    for loop in loops:
        vertices = np.asarray(loop.vertices)
        for edge_index, raw_start in enumerate(vertices):
            raw_end = vertices[(edge_index + 1) % vertices.shape[0]]
            start = tuple(Fraction.from_float(float(value)) for value in raw_start)
            end = tuple(Fraction.from_float(float(value)) for value in raw_end)
            direction = (end[0] - start[0], end[1] - start[1])
            for piece_row, extraction in enumerate(realization.extractions):
                u0, u1, v0, v1 = extraction.parameter_bounds
                lower = (u0, v0)
                upper = (u1, v1)
                first, last = Fraction(0), Fraction(1)
                intersects = True
                for axis in range(2):
                    if direction[axis] == 0:
                        if start[axis] < lower[axis] or start[axis] > upper[axis]:
                            intersects = False
                            break
                        continue
                    enter = (lower[axis] - start[axis]) / direction[axis]
                    leave = (upper[axis] - start[axis]) / direction[axis]
                    first = max(first, min(enter, leave))
                    last = min(last, max(enter, leave))
                    if first >= last:
                        intersects = False
                        break
                if not intersects:
                    continue
                segment_start = (
                    start[0] + first * direction[0],
                    start[1] + first * direction[1],
                )
                segment_end = (
                    start[0] + last * direction[0],
                    start[1] + last * direction[1],
                )
                key = (loop.loop_id, edge_index, first, last)
                candidate = (piece_row, segment_start, segment_end)
                previous = pieces.get(key)
                if previous is None or candidate[0] < previous[0]:
                    pieces[key] = candidate
    ordered = tuple(sorted(pieces.items(), key=lambda item: item[0]))
    exact_segments = tuple((item[1][1], item[1][2]) for item in ordered)
    piece_rows = np.asarray([item[1][0] for item in ordered], dtype=np.int32)
    segment_values = np.asarray(
        [
            [[float(value) for value in start], [float(value) for value in end]]
            for start, end in exact_segments
        ],
        dtype=np.float64,
    )
    return segment_values, piece_rows, exact_segments


class _CellCutBoundaryData(NamedTuple):
    cut_rule: CutQuadratureRule
    points: np.ndarray
    weights: np.ndarray
    cell_rows: np.ndarray
    tangents: np.ndarray
    conormals: np.ndarray
    segments: np.ndarray
    segment_piece_rows: np.ndarray
    exact_segments: tuple[
        tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...
    ]


def _cell_cut_quadrature(
    mesh: TMesh2D,
    realization: LocalExtractedBernsteinRealization,
    trim: UVTrimmedSurface,
    quadrature: IsogeometricQuadraturePolicy,
    /,
) -> tuple[np.ndarray, np.ndarray, _CellCutBoundaryData]:
    bounds = np.asarray(
        [
            [
                [
                    float(extraction.parameter_bounds[0]),
                    float(extraction.parameter_bounds[1]),
                ],
                [
                    float(extraction.parameter_bounds[2]),
                    float(extraction.parameter_bounds[3]),
                ],
            ]
            for extraction in realization.extractions
        ],
        dtype=np.float64,
    )
    order = max(quadrature.count_for_axis(axis, 2) for axis in range(2))
    rule = build_trimmed_surface_quadrature(
        trim,
        bounds,
        geometry_map=_identity_parameter_map,
        geometry_jacobian=_identity_parameter_jacobian,
        order=order,
        require_tolerance=True,
    )
    if not rule.certificate.tolerance_satisfied:
        raise ValueError(
            "Trimmed ASTS parameter quadrature did not meet its error estimate."
        )
    cells = len(mesh.cells)
    piece_cell_rows = np.asarray(realization.piece_cell_rows, dtype=np.int32)
    volume_piece_rows = np.asarray(rule.volume_cell_ids, dtype=np.int32)
    volume_cell_rows = piece_cell_rows[volume_piece_rows]
    grouped_points = [
        np.asarray(rule.volume_uv_points)[volume_cell_rows == row] for row in range(cells)
    ]
    grouped_weights = [
        np.asarray(rule.volume_weights)[volume_cell_rows == row] for row in range(cells)
    ]
    max_points = max((values.shape[0] for values in grouped_points), default=0)
    if max_points == 0:
        raise ValueError("The trim domain has no quadrature support in the T-mesh cells.")
    points = np.zeros((cells, max_points, 2), dtype=np.float64)
    weights = np.zeros((cells, max_points), dtype=np.float64)
    for row, (point_values, weight_values) in enumerate(
        zip(grouped_points, grouped_weights, strict=True)
    ):
        if point_values.size == 0:
            u0, u1, v0, v1 = mesh.cells[row].parameter_bounds
            points[row, :, :] = ((float(u0 + u1) / 2.0), (float(v0 + v1) / 2.0))
            continue
        points[row, : point_values.shape[0]] = point_values
        weights[row, : weight_values.shape[0]] = weight_values
        if point_values.shape[0] < max_points:
            points[row, point_values.shape[0] :] = point_values[-1]
    boundary_points = np.asarray(rule.boundary_uv_points)
    boundary_weights = np.asarray(rule.boundary_weights)
    boundary_normals = np.asarray(rule.boundary_normals_uv)
    boundary_piece_rows = np.asarray(rule.boundary_cell_ids, dtype=np.int32)
    boundary_cells = piece_cell_rows[boundary_piece_rows]
    if boundary_points.shape[0] == 0:
        raise ValueError("A trimmed ASTS surface requires a nonempty physical boundary.")
    boundary_tangents = np.stack(
        (-boundary_normals[:, 1], boundary_normals[:, 0]), axis=-1
    )
    boundary_segments, segment_piece_rows, exact_segments = _trim_segments(
        mesh, realization, trim
    )
    return (
        points,
        weights,
        _CellCutBoundaryData(
            rule,
            boundary_points[:, None, :],
            boundary_weights[:, None],
            boundary_cells,
            boundary_tangents,
            boundary_normals,
            boundary_segments,
            segment_piece_rows,
            exact_segments,
        ),
    )


class TSplineIsogeometricPlan(AbstractDiscretizationPlan):
    """Qualified scalar H1 IGA on a topology-derived bicubic ASTS surface."""

    geometry: TSplineGeometryState
    fields: tuple[TSplineH1FieldSpec, ...]
    trim: UVTrimmedSurface
    quadrature_policy: IsogeometricQuadraturePolicy
    precision_policy: FiniteElementPrecisionPolicy
    exact_geometry_certificate: TSplineGeometryCertificate
    geometry_certificate: TSplineGeometryCertificate
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        geometry: TSplineGeometryState | TSplineSurfacePatch,
        fields: TSplineH1FieldSpec | Sequence[TSplineH1FieldSpec] | None = None,
        /,
        *,
        field_name: str = "u",
        trim: UVTrimmedSurface | None = None,
        quadrature_policy: IsogeometricQuadraturePolicy,
        precision_policy: FiniteElementPrecisionPolicy | None = None,
    ) -> None:
        if isinstance(geometry, TSplineSurfacePatch):
            geometry = TSplineGeometryState(geometry)
        if not isinstance(geometry, TSplineGeometryState):
            raise TypeError(
                "geometry must be TSplineGeometryState or TSplineSurfacePatch."
            )
        control_points, raw_weights = geometry.geometry_bank()
        if _numeric_bank_id(control_points, raw_weights) != geometry.numeric_bank_id:
            raise ValueError(
                "Numeric geometry leaves must match the bound CAD source definition."
            )
        mesh = geometry.mesh
        realization = geometry.surface_patch.realization
        basis_certificate = realization.basis_certificate
        if not basis_certificate.passed:
            raise ValueError(
                "IGA admission requires a passing ASTS basis and continuity certificate."
            )
        exact_weights = np.asarray(geometry.exact_weights, dtype=object)
        exact_points = np.asarray(geometry.exact_control_points, dtype=object)
        exact_certificate = certify_tspline_geometry(
            realization, exact_points, exact_weights
        )
        numeric_certificate = certify_tspline_geometry(
            realization, np.asarray(control_points), np.asarray(raw_weights)
        )
        if not exact_certificate.passed or not numeric_certificate.passed:
            reasons = (
                *exact_certificate.failure_reasons,
                *numeric_certificate.failure_reasons,
            )
            raise ValueError(
                "IGA admission requires exact-source and numeric physical-map certificates: "
                + "; ".join(reasons)
            )
        trim_ = (
            trim
            if trim is not None
            else UVTrimmedSurface(
                UVTrimLoop(
                    (
                        (
                            float(mesh.parameter_domain[0]),
                            float(mesh.parameter_domain[2]),
                        ),
                        (
                            float(mesh.parameter_domain[1]),
                            float(mesh.parameter_domain[2]),
                        ),
                        (
                            float(mesh.parameter_domain[1]),
                            float(mesh.parameter_domain[3]),
                        ),
                        (
                            float(mesh.parameter_domain[0]),
                            float(mesh.parameter_domain[3]),
                        ),
                    )
                )
            )
        )
        if not isinstance(trim_, UVTrimmedSurface):
            raise TypeError("trim must be UVTrimmedSurface or None.")
        bounds = mesh.parameter_domain
        trim_vertices = np.concatenate(
            [np.asarray(trim_.outer.vertices)]
            + [np.asarray(loop.vertices) for loop in trim_.holes],
            axis=0,
        )
        if (
            np.any(trim_vertices[:, 0] < float(bounds[0]))
            or np.any(trim_vertices[:, 0] > float(bounds[1]))
            or np.any(trim_vertices[:, 1] < float(bounds[2]))
            or np.any(trim_vertices[:, 1] > float(bounds[3]))
        ):
            raise ValueError("Trim loops must lie inside the T-mesh parameter domain.")
        if fields is None:
            field_values = (
                TSplineH1FieldSpec(
                    field_name,
                    weights_from_geometry=True,
                    anchor_count=mesh.coefficient_count,
                ),
            )
        elif isinstance(fields, TSplineH1FieldSpec):
            field_values = (fields,)
        else:
            field_values = tuple(fields)
        if not field_values or not all(
            isinstance(field, TSplineH1FieldSpec) for field in field_values
        ):
            raise TypeError("fields must contain TSplineH1FieldSpec values.")
        if len({field.name for field in field_values}) != len(field_values):
            raise ValueError("T-spline field names must be unique.")
        if any(
            field.weights is not None and field.weights.shape != (mesh.coefficient_count,)
            for field in field_values
        ):
            raise ValueError("T-spline field weight banks must match the active anchors.")
        precision = (
            FiniteElementPrecisionPolicy()
            if precision_policy is None
            else precision_policy
        )
        if not isinstance(precision, FiniteElementPrecisionPolicy):
            raise TypeError(
                "precision_policy must be FiniteElementPrecisionPolicy or None."
            )
        key = DiscretizationKey(
            "isogeometric",
            DiscretizationRole.PHYSICAL,
            domain_labels=("u", "v"),
        )
        capabilities = (
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.TRACE,
            DiscretizationCapability.BOUNDARY_INTEGRAL,
            DiscretizationCapability.VARIATIONAL_ASSEMBLY,
            DiscretizationCapability.MATRIX_FREE,
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        )
        plan_id = canonical_fingerprint(
            {
                "kind": "asts2d-isogeometric-plan",
                "mesh": mesh.mesh_id,
                "geometry_layout": geometry.geometry_layout_id,
                "source_definition_id": geometry.source_definition_id,
                "source_revision": geometry.revision,
                "basis_certificate": basis_certificate.certificate_id,
                "exact_source_control_points": [
                    _fraction_payload(row)
                    for row in geometry.surface_patch.exact_source_control_points
                ],
                "exact_source_weights": _fraction_payload(
                    geometry.surface_patch.exact_source_weights
                ),
                "exact_target_homogeneous": [
                    _fraction_payload(row)
                    for row in geometry.surface_patch.exact_target_homogeneous
                ],
                "numeric_bank_id": geometry.numeric_bank_id,
                "maximum_homogeneous_realization_error": geometry.maximum_conversion_error,
                "fields": [field.field_spec_id for field in field_values],
                "trim": trim_.surface_id,
                "quadrature": quadrature_policy.policy_id,
                "precision": precision.policy_id,
            }
        )
        self.geometry = geometry
        self.fields = field_values
        self.trim = trim_
        self.quadrature_policy = quadrature_policy
        self.precision_policy = precision
        self.exact_geometry_certificate = exact_certificate
        self.geometry_certificate = numeric_certificate
        self.key = key
        self.capabilities = capabilities
        self.plan_id = plan_id

    def prepare(
        self, /, *, numeric_version: str = "0"
    ) -> PreparedTSplineIsogeometricDiscretization:
        return PreparedTSplineIsogeometricDiscretization(
            self, numeric_version=numeric_version
        )

    def refine(
        self, marked_cell_ids: Sequence[int], /, *, tolerance: float = 5.0e-11
    ) -> TSplineIsogeometricRefinement:
        relation = self.geometry.mesh.refine(marked_cell_ids, tolerance=tolerance)
        target_geometry = self.geometry.prolong(relation)
        target_fields: list[TSplineH1FieldSpec] = []
        for field in self.fields:
            if field.weights_from_geometry or field.weights is None:
                target_fields.append(field)
                continue
            exact_target = relation.transfer_plan.apply_exact_scalars(
                field.exact_weights or ()
            )
            numeric_target = relation.transfer_plan.apply(field.weights)
            target_fields.append(
                TSplineH1FieldSpec(
                    field.name,
                    numeric_target,
                    exact_weights=exact_target,
                    anchor_count=relation.refined_mesh.coefficient_count,
                )
            )
        target_plan = TSplineIsogeometricPlan(
            target_geometry,
            tuple(target_fields),
            trim=self.trim,
            quadrature_policy=self.quadrature_policy,
            precision_policy=self.precision_policy,
        )
        return TSplineIsogeometricRefinement(self, target_plan, relation)


class PreparedTSplineIsogeometricDiscretization(AbstractPreparedLocalDiscretization):
    """Fixed-route local IGA assembly over native T-mesh cells and trim pieces."""

    basis_realization: LocalExtractedBernsteinRealization
    fields: tuple[TSplineH1FieldSpec, ...]
    trim: UVTrimmedSurface
    quadrature_policy: IsogeometricQuadraturePolicy
    precision_policy: FiniteElementPrecisionPolicy
    default_runtime: TSplineRuntimeData
    exact_geometry_certificate: TSplineGeometryCertificate
    geometry_certificate: TSplineGeometryCertificate
    field_cell_gathers: tuple[Array, ...]
    cell_points: Array
    cell_parameter_weights: Array
    boundary_points: Array
    boundary_parameter_weights: Array
    boundary_cell_rows: Array
    boundary_tangents: Array
    boundary_conormals: Array
    boundary_segments: Array
    boundary_segment_piece_rows: Array
    boundary_exact_segments: tuple[
        tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...
    ] = eqx.field(static=True)
    boundary_domain: IntegrationDomain
    cell_domain: IntegrationDomain
    bindings: tuple[LocalFieldBinding, ...]
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    block_space: BlockSpace
    measures: tuple
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport
    cell_kind: str = eqx.field(static=True)

    @checked
    def __init__(
        self, plan: TSplineIsogeometricPlan, /, *, numeric_version: str = "0"
    ) -> None:
        version = str(numeric_version).strip()
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        mesh = plan.geometry.mesh
        realization = plan.geometry.surface_patch.realization
        topology, cell_entities, _, anchor_entities = _cell_complex_topology(mesh)
        runtime = TSplineRuntimeData(
            plan.geometry,
            numeric_version=version,
            exact_geometry_certificate=plan.exact_geometry_certificate,
            geometry_certificate=plan.geometry_certificate,
        )
        support = DiscreteSupport(
            topology,
            plan.geometry.ambient_dimension,
            plan.geometry.geometry_layout_id,
        )
        cell_domain = IntegrationDomain(
            "cell",
            np.arange(mesh.cell_count, dtype=np.int32),
            support.support_id,
            cell_entities.entity_set_id,
            owner_cells=np.arange(mesh.cell_count, dtype=np.int32),
        )
        cell_points, cell_weights, boundary_data = _cell_cut_quadrature(
            mesh, realization, plan.trim, plan.quadrature_policy
        )
        (
            cut_rule,
            boundary_points,
            boundary_weights,
            boundary_cell_rows,
            boundary_tangents,
            boundary_conormals,
            boundary_segments,
            boundary_segment_piece_rows,
            boundary_exact_segments,
        ) = boundary_data
        boundary_set_id = canonical_fingerprint(
            {
                "kind": "asts2d-trim-boundary-quadrature-entities",
                "mesh": mesh.mesh_id,
                "trim": plan.trim.surface_id,
                "cut_quadrature": cut_rule.quadrature_id,
            }
        )
        boundary_domain = IntegrationDomain(
            "exterior_facet",
            np.arange(boundary_points.shape[0], dtype=np.int32),
            support.support_id,
            boundary_set_id,
            owner_cells=boundary_cell_rows,
            neighbor_cells=np.full(boundary_cell_rows.shape, -1, dtype=np.int32),
            owner_local_entities=np.zeros(boundary_cell_rows.shape, dtype=np.int32),
        )
        layout = EntityDofLayout(
            anchor_entities.entity_set_id,
            mesh.coefficient_count,
            mesh.coefficient_count,
            layout_id=canonical_fingerprint(
                {"kind": "asts2d-anchor-dof-layout", "mesh": mesh.mesh_id}
            ),
        )
        field_spaces = tuple(
            DiscreteFieldSpace(
                field.name,
                support.support_id,
                layout,
                ArraySpace(
                    (mesh.coefficient_count,),
                    dtype=plan.precision_policy.evaluation_dtype,
                ),
                representation="basis_coefficient",
                conformity="H1",
                reconstruction_id=canonical_fingerprint(
                    {"kind": "asts2d-h1-reconstruction", "field": field.field_spec_id}
                ),
                trace_space_id=canonical_fingerprint(
                    {"kind": "asts2d-h1-trace", "field": field.field_spec_id}
                ),
            )
            for field in plan.fields
        )
        bindings = tuple(
            LocalFieldBinding(
                field.name,
                space,
                component_shape=(),
                public_shape=(mesh.coefficient_count,),
                execution_shape=(mesh.coefficient_count,),
                local_width=realization.local_width,
                layout_id=layout.layout_id,
            )
            for field, space in zip(plan.fields, field_spaces, strict=True)
        )
        resources = {
            "tmesh_cells": mesh.cell_count,
            "tmesh_edges": len(mesh.edges),
            "active_anchors": mesh.coefficient_count,
            "trim_volume_quadrature_points": int(
                np.asarray(cut_rule.volume_weights).size
            ),
            "trim_boundary_quadrature_points": int(
                np.asarray(cut_rule.boundary_weights).size
            ),
            "fields": len(plan.fields),
        }
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                "basis topology, exact knot windows, and sparse cell extraction are shared with CAD",
                "basis independence, Bernstein identities, and complete C2 traces are certified separately",
                "physical denominator, regularity, and global embedding use exact Bernstein bounds",
                "trim volume and oriented boundary quadrature use the prepared UV trim loops",
                "geometry and field weights remain raw; rational field transfer is independently weighted",
            ),
            resource_counts=resources,
        )
        spaces, measures, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=support,
            field_spaces=field_spaces,
            measures=(),
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        gathers = tuple(
            jnp.asarray(realization.cell_coefficient_rows) for _ in plan.fields
        )
        self.basis_realization = realization
        self.fields = plan.fields
        self.trim = plan.trim
        self.quadrature_policy = plan.quadrature_policy
        self.precision_policy = plan.precision_policy
        self.default_runtime = runtime
        self.exact_geometry_certificate = plan.exact_geometry_certificate
        self.geometry_certificate = plan.geometry_certificate
        self.field_cell_gathers = gathers
        self.cell_points = jnp.asarray(cell_points)
        self.cell_parameter_weights = jnp.asarray(cell_weights)
        self.boundary_points = jnp.asarray(boundary_points)
        self.boundary_parameter_weights = jnp.asarray(boundary_weights)
        self.boundary_cell_rows = jnp.asarray(boundary_cell_rows, dtype=jnp.int32)
        self.boundary_tangents = jnp.asarray(boundary_tangents)
        self.boundary_conormals = jnp.asarray(boundary_conormals)
        self.boundary_segments = jnp.asarray(boundary_segments)
        self.boundary_segment_piece_rows = jnp.asarray(
            boundary_segment_piece_rows, dtype=jnp.int32
        )
        self.boundary_exact_segments = boundary_exact_segments
        self.boundary_domain = boundary_domain
        self.cell_domain = cell_domain
        self.bindings = bindings
        self.key = plan.key
        self.support = support
        self.field_spaces = spaces
        self.block_space = BlockSpace(
            tuple(space.vector_space for space in spaces),
            names=tuple(space.name for space in spaces),
        )
        self.measures = measures
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.numeric_version = version
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-asts2d-isogeometric-discretization",
                "plan": plan.plan_id,
                "topology": topology.topology_id,
                "geometry_layout": plan.geometry.geometry_layout_id,
                "runtime": runtime.runtime_id,
                "exact_geometry_certificate": self.exact_geometry_certificate.certificate_id,
                "numeric_geometry_certificate": self.geometry_certificate.certificate_id,
                "numeric_version": version,
                "cut_quadrature": cut_rule.quadrature_id,
            }
        )
        self.cell_kind = "quadrilateral"

    @property
    def cell_gathers(self) -> Array:
        return self.field_cell_gathers[0]

    @property
    def precision_evidence(self) -> PrecisionEvidenceEnvelope:
        return self.precision_policy.evidence()

    def _field_index(self, name: str, /) -> int:
        for index, field in enumerate(self.fields):
            if field.name == str(name):
                return index
        raise KeyError(f"Unknown T-spline H1 field {name!r}.")

    def local_field_binding(self, name: str, /) -> LocalFieldBinding:
        return self.bindings[self._field_index(name)]

    def local_variational_capabilities(self, /) -> LocalVariationalCapabilities:
        semantics = (
            "exact_interpolation_transpose",
            "exact_gradient_transpose",
            "exact_hessian_transpose",
            "exact_trace_transpose",
            "runtime_rational_weights",
        )
        return LocalVariationalCapabilities(
            "asts2d-isogeometric-local-provider",
            (
                LocalVariationalOffer(
                    "prepared-local",
                    ("cell",),
                    ("diffusion", "tensor-diffusion", "mass", "source", "functional"),
                    ("value", "grad"),
                    ("value", "gradient"),
                    ("dense",),
                    ("matrix_free",),
                    ("asts2d-local-extracted-bernstein",),
                    automatic_kernel_mode="dense",
                    automatic_operator_realization="matrix_free",
                    automatic_reference_realization_id="asts2d-local-extracted-bernstein",
                    action_semantics=semantics,
                ),
                LocalVariationalOffer(
                    "prepared-local",
                    ("exterior_facet",),
                    ("boundary-load", "functional"),
                    ("value",),
                    ("value", "trace"),
                    ("dense",),
                    ("matrix_free",),
                    ("asts2d-local-extracted-bernstein",),
                    automatic_kernel_mode="dense",
                    automatic_operator_realization="matrix_free",
                    automatic_reference_realization_id="asts2d-local-extracted-bernstein",
                    action_semantics=semantics,
                ),
            ),
        )

    def validate_local_runtime(self, runtime: object, /) -> None:
        if not isinstance(runtime, TSplineRuntimeData):
            raise TypeError("ASTS execution requires TSplineRuntimeData.")
        if runtime.mesh_id != self.default_runtime.mesh_id or (
            runtime.geometry_layout_id != self.default_runtime.geometry_layout_id
        ):
            raise ValueError(
                "ASTS runtime does not match the prepared topology and layout."
            )
        if (
            not runtime.source_definition_id
            or runtime.source_definition_id != runtime.surface_patch.source_definition_id
        ):
            raise ValueError(
                "ASTS runtime source identity does not bind its CAD carrier."
            )
        control_points, weights = runtime.geometry_bank()
        expected_shape = (self.basis_realization.global_coefficient_count, 3)
        if control_points.shape != expected_shape or weights.shape != expected_shape[:1]:
            raise ValueError(
                "ASTS runtime geometry banks do not match the prepared layout."
            )
        exact_weights = runtime.exact_weights
        exact_points = runtime.exact_control_points
        basis_certificate_id = self.basis_realization.basis_certificate.certificate_id
        if (
            not runtime.exact_geometry_certificate.passed
            or not runtime.geometry_certificate.passed
            or runtime.exact_geometry_certificate.mesh_id != runtime.mesh_id
            or runtime.geometry_certificate.mesh_id != runtime.mesh_id
            or runtime.exact_geometry_certificate.basis_certificate_id
            != basis_certificate_id
            or runtime.geometry_certificate.basis_certificate_id != basis_certificate_id
            or runtime.exact_geometry_certificate.numeric_state_id
            != _geometry_certificate_state_id(exact_points, exact_weights)
        ):
            raise ValueError(
                "ASTS runtime map certificates do not bind its source state."
            )
        if not isinstance(control_points, jax_core.Tracer) and not isinstance(
            weights, jax_core.Tracer
        ):
            current_bank_id = _numeric_bank_id(control_points, weights)
            if (
                _numeric_geometry_certificate_state_id(control_points, weights)
                != runtime.geometry_certificate.numeric_state_id
            ):
                raise ValueError(
                    "ASTS numeric map certificate does not bind its raw bank."
                )
            if current_bank_id != runtime.numeric_bank_id:
                raise ValueError(
                    "Changed geometry leaves require an explicitly re-authored CAD source revision."
                )

    def prepare_runtime(
        self,
        geometry: TSplineGeometryState
        | TSplineSurfacePatch
        | TSplineRuntimeData
        | None = None,
        /,
        *,
        numeric_version: str,
    ) -> TSplineRuntimeData:
        if geometry is None:
            state = TSplineGeometryState(
                self.default_runtime.surface_patch,
                revision=self.default_runtime.source_revision,
            )
        elif isinstance(geometry, TSplineRuntimeData):
            self.validate_local_runtime(geometry)
            state = TSplineGeometryState(
                geometry.surface_patch, revision=geometry.source_revision
            )
        elif isinstance(geometry, TSplineSurfacePatch):
            state = TSplineGeometryState(geometry)
        elif isinstance(geometry, TSplineGeometryState):
            state = geometry
        else:
            raise TypeError(
                "geometry must be TSplineGeometryState, TSplineSurfacePatch, "
                "TSplineRuntimeData, or None."
            )
        if state.mesh.mesh_id != self.default_runtime.mesh_id:
            raise ValueError("Runtime geometry uses another T-mesh topology.")
        if (
            state.source_definition_id != self.default_runtime.source_definition_id
            and state.revision == self.default_runtime.source_revision
        ):
            raise ValueError("Changed exact geometry requires a new source revision.")
        control_points, weights = state.geometry_bank()
        if _numeric_bank_id(control_points, weights) != state.numeric_bank_id:
            raise ValueError(
                "Changed geometry leaves require a newly authored CAD source revision."
            )
        realization = state.surface_patch.realization
        exact_certificate = certify_tspline_geometry(
            realization,
            np.asarray(state.exact_control_points, dtype=object),
            np.asarray(state.exact_weights, dtype=object),
        )
        numeric_certificate = certify_tspline_geometry(
            realization, np.asarray(control_points), np.asarray(weights)
        )
        if not exact_certificate.passed or not numeric_certificate.passed:
            raise ValueError(
                "Runtime geometry failed exact or numeric physical-map certification."
            )
        result = TSplineRuntimeData(
            state,
            numeric_version=numeric_version,
            exact_geometry_certificate=exact_certificate,
            geometry_certificate=numeric_certificate,
        )
        self.validate_local_runtime(result)
        return result

    def integration_domain(
        self, kind: str, selection: EntitySelection | None = None, /
    ) -> IntegrationDomain:
        match str(kind):
            case "cell":
                base = self.cell_domain
            case "exterior_facet":
                base = self.boundary_domain
            case "interior_facet":
                raise ValueError(
                    "ASTS H1 currently admits cell and trim-boundary integration only."
                )
            case _:
                raise ValueError(
                    "ASTS integration domains are 'cell' or 'exterior_facet'."
                )
        if selection is None:
            return base
        if (
            not isinstance(selection, EntitySelection)
            or selection.entity_set_id != base.entity_set_id
        ):
            raise ValueError(
                "Entity selection does not match the ASTS integration domain."
            )
        rows = np.flatnonzero(
            np.asarray(selection.mask, dtype=np.bool_)[np.asarray(base.entity_indices)]
        )
        return IntegrationDomain(
            base.kind,
            np.asarray(base.entity_indices)[rows],
            base.support_id,
            base.entity_set_id,
            owner_cells=np.asarray(base.owner_cells)[rows],
            neighbor_cells=np.asarray(base.neighbor_cells)[rows],
            owner_local_entities=np.asarray(base.owner_local_entities)[rows],
            neighbor_local_entities=np.asarray(base.neighbor_local_entities)[rows],
            selection_id=selection.selection_id,
        )

    def _reference_actions(
        self,
        field: TSplineH1FieldSpec,
        cell_rows: np.ndarray,
        points: np.ndarray,
        /,
        *,
        structural_id: str,
        is_trace: bool,
        maximum_derivative_order: int,
    ) -> TSplineReferenceActions:
        width = self.basis_realization.local_width
        entity_count, point_count = points.shape[:2]
        values = np.zeros((entity_count, point_count, width), dtype=np.float64)
        gradients = np.zeros((entity_count, point_count, width, 2), dtype=np.float64)
        hessians = np.zeros((entity_count, point_count, width, 2, 2), dtype=np.float64)
        global_rows = np.asarray(self.basis_realization.cell_coefficient_rows)
        extractions = self.basis_realization.extractions
        parent_cell_ids = np.asarray(self.basis_realization.cell_ids)[cell_rows]
        routed_piece_rows = np.asarray(
            self.basis_realization._route_cells(
                jnp.asarray(points.reshape((-1, 2))),
                jnp.asarray(np.repeat(parent_cell_ids, point_count)),
            ),
            dtype=np.int32,
        ).reshape((entity_count, point_count))
        derivative_requests = ((1, 0), (0, 1)) if maximum_derivative_order >= 1 else ()
        hessian_requests = (
            ((2, 0), (1, 1), (0, 2)) if maximum_derivative_order >= 2 else ()
        )
        for entity, cell_row in enumerate(cell_rows):
            support_rows = self.basis_realization.cell_support_rows[int(cell_row)]
            support_columns = {
                coefficient_row: column
                for column, coefficient_row in enumerate(support_rows)
            }
            for piece_row in np.unique(routed_piece_rows[entity]):
                point_rows = np.flatnonzero(routed_piece_rows[entity] == piece_row)
                extraction = extractions[int(piece_row)]
                local_columns = np.asarray(
                    [support_columns[row] for row in extraction.coefficient_row_ids],
                    dtype=np.int32,
                )
                routed_points = points[entity, point_rows]
                index = np.ix_(
                    point_rows,
                    local_columns,
                    np.arange(2),
                )
                values[entity][np.ix_(point_rows, local_columns)] = np.asarray(
                    extraction.evaluate(routed_points)
                )
                if derivative_requests:
                    local_gradients = np.stack(
                        tuple(
                            np.asarray(
                                extraction.evaluate(routed_points, derivative=derivative)
                            )
                            for derivative in derivative_requests
                        ),
                        axis=-1,
                    )
                    gradients[entity][index] = local_gradients
                if hessian_requests:
                    local_hessians = np.zeros(
                        (point_rows.size, extraction.local_width, 2, 2),
                        dtype=np.float64,
                    )
                    local_hessians[:, :, 0, 0] = np.asarray(
                        extraction.evaluate(routed_points, derivative=(2, 0))
                    )
                    local_hessians[:, :, 0, 1] = np.asarray(
                        extraction.evaluate(routed_points, derivative=(1, 1))
                    )
                    local_hessians[:, :, 1, 0] = local_hessians[:, :, 0, 1]
                    local_hessians[:, :, 1, 1] = np.asarray(
                        extraction.evaluate(routed_points, derivative=(0, 2))
                    )
                    hessian_index = np.ix_(
                        point_rows,
                        local_columns,
                        np.arange(2),
                        np.arange(2),
                    )
                    hessians[entity][hessian_index] = local_hessians
        gather_rows = global_rows[cell_rows]
        return TSplineReferenceActions(
            values,
            gradients,
            hessians,
            gather_rows,
            field.weights,
            field_weights_from_geometry=field.weights_from_geometry,
            query_shape=(point_count,),
            topology_id=self.default_runtime.mesh_id,
            geometry_layout_id=self.default_runtime.geometry_layout_id,
            structural_id=structural_id,
            global_coefficient_count=self.basis_realization.global_coefficient_count,
            maximum_derivative_order=maximum_derivative_order,
            is_trace=is_trace,
        )

    @checked
    def prepare_local_regions(
        self,
        domain: IntegrationDomain,
        /,
        *,
        field_names: tuple[str, ...],
        maximum_derivative_order: int,
        kernel_mode: str,
    ) -> tuple[PreparedLocalRegion, ...]:
        if domain.support_id != self.support.support_id:
            raise ValueError("ASTS integration domain belongs to another support.")
        if str(kernel_mode) != "dense":
            raise ValueError(
                "ASTS local actions require the dense cell-local realization."
            )
        order = int(maximum_derivative_order)
        if order < 0 or order > 1:
            raise ValueError(
                "ASTS scalar surface H1 actions support derivative orders zero and one."
            )
        names = tuple(str(name) for name in field_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("ASTS local regions require unique field names.")
        field_indices = tuple(self._field_index(name) for name in names)
        selected = np.asarray(domain.entity_indices, dtype=np.int32)
        if domain.kind == "cell":
            if domain.entity_set_id != self.cell_domain.entity_set_id:
                raise ValueError("ASTS cell domain has an incompatible entity set.")
            points = np.asarray(self.cell_points)[selected]
            parameter_weights = np.asarray(self.cell_parameter_weights)[selected]
            cell_ids = np.asarray(self.basis_realization.cell_ids)[selected]
            references = tuple(
                self._reference_actions(
                    self.fields[index],
                    selected,
                    points,
                    structural_id=domain.domain_id,
                    is_trace=False,
                    maximum_derivative_order=order,
                )
                for index in field_indices
            )
            geometry = TSplineGeometryActions(
                self.basis_realization,
                points,
                parameter_weights,
                cell_ids,
                runtime_layout_id=self.default_runtime.geometry_layout_id,
                domain_kind="cell",
                structural_id=domain.domain_id,
            )
            return (
                PreparedLocalRegion(
                    domain,
                    names,
                    tuple(
                        self.field_cell_gathers[index][selected]
                        for index in field_indices
                    ),
                    references,
                    geometry,
                    block_name="asts2d-patch",
                    cell_kind="quadrilateral",
                ),
            )
        if (
            domain.kind != "exterior_facet"
            or domain.entity_set_id != self.boundary_domain.entity_set_id
        ):
            raise ValueError(
                "ASTS facet domain has an incompatible trim-boundary entity set."
            )
        owners = np.asarray(self.boundary_domain.owner_cells, dtype=np.int32)[selected]
        points = np.asarray(self.boundary_points)[selected]
        parameter_weights = np.asarray(self.boundary_parameter_weights)[selected]
        cell_ids = np.asarray(self.basis_realization.cell_ids)[owners]
        references = tuple(
            self._reference_actions(
                self.fields[index],
                owners,
                points,
                structural_id=domain.domain_id,
                is_trace=True,
                maximum_derivative_order=order,
            )
            for index in field_indices
        )
        geometry = TSplineGeometryActions(
            self.basis_realization,
            points,
            parameter_weights,
            cell_ids,
            runtime_layout_id=self.default_runtime.geometry_layout_id,
            domain_kind="exterior_facet",
            structural_id=domain.domain_id,
            tangents=np.asarray(self.boundary_tangents)[selected],
            conormals=np.asarray(self.boundary_conormals)[selected],
        )
        group_domain = IntegrationDomain(
            "exterior_facet",
            selected,
            domain.support_id,
            domain.entity_set_id,
            owner_cells=owners,
            neighbor_cells=np.full(owners.shape, -1, dtype=np.int32),
            owner_local_entities=np.zeros(owners.shape, dtype=np.int32),
            selection_id=domain.selection_id,
        )
        return (
            PreparedLocalRegion(
                group_domain,
                names,
                tuple(self.field_cell_gathers[index][owners] for index in field_indices),
                references,
                geometry,
                block_name="asts2d-trim-boundary",
                cell_kind="oriented-curve",
                owner_cells=owners,
                owner_local_entities=np.zeros(owners.shape, dtype=np.int32),
            ),
        )

    def prepare_query_route(
        self,
        points: ConvertibleToArray,
        /,
        *,
        cell_ids: ConvertibleToArray | None = None,
    ) -> TSplineQueryRoute:
        """Validate finite trimmed-domain points and freeze their T-cell routes."""
        query_input = np.asarray(points)
        if not np.issubdtype(query_input.dtype, np.number) or np.iscomplexobj(
            query_input
        ):
            raise TypeError("ASTS query parameters must be real numeric coordinates.")
        query = np.asarray(query_input, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 2 or np.any(~np.isfinite(query)):
            raise ValueError(
                "ASTS query parameters must be finite with shape (point, 2)."
            )
        if np.any(np.asarray(self.trim.classify(query)) < 0):
            raise ValueError("ASTS query lies outside the trimmed source surface.")
        query_array = jnp.asarray(query)
        if cell_ids is None:
            rows = np.asarray(self.basis_realization._route_cells(query_array, None))
            routes = np.asarray(self.basis_realization.piece_cell_ids)[rows]
        else:
            routes = np.asarray(cell_ids)
            if routes.shape != query.shape[:1] or not np.issubdtype(
                routes.dtype, np.integer
            ):
                raise ValueError(
                    "cell_ids must contain one integer ID per parameter point."
                )
            if routes.dtype.kind == "u" and np.any(routes > np.iinfo(np.int64).max):
                raise ValueError("ASTS query cell IDs must fit signed 64-bit storage.")
            routes = np.asarray(routes, dtype=np.int64)
            np.asarray(
                self.basis_realization._route_cells(query_array, jnp.asarray(routes))
            )
        return TSplineQueryRoute(
            query,
            routes,
            prepared_id=self.prepared_id,
            mesh_id=self.basis_realization.mesh.mesh_id,
            trim_id=self.trim.surface_id,
        )

    def _validate_query_route(self, route: TSplineQueryRoute, /) -> None:
        if not isinstance(route, TSplineQueryRoute):
            raise TypeError("Fixed T-spline queries require a TSplineQueryRoute.")
        if (
            route.prepared_id != self.prepared_id
            or route.mesh_id != self.basis_realization.mesh.mesh_id
            or route.trim_id != self.trim.surface_id
        ):
            raise ValueError(
                "ASTS query route does not match this prepared trimmed surface."
            )

    def route_parameter_points(self, points: ConvertibleToArray, /) -> Array:
        """Return trim-validated persistent cell IDs for a convenience query."""
        return self.prepare_query_route(points).cell_ids

    def contains_parameter_points(self, points: ConvertibleToArray, /) -> Array:
        return self.trim.classify(points) >= 0

    def evaluate_geometry_route(
        self,
        route: TSplineQueryRoute,
        /,
        *,
        runtime: TSplineRuntimeData | None = None,
        maximum_order: int = 2,
    ) -> TSplineRationalJet:
        self._validate_query_route(route)
        runtime_ = self.default_runtime if runtime is None else runtime
        self.validate_local_runtime(runtime_)
        control_points, raw_weights = runtime_.geometry_bank()
        return self.basis_realization.evaluate_rational(
            route.parameters,
            control_points,
            raw_weights,
            cell_ids=route.cell_ids,
            maximum_order=maximum_order,
        )

    def evaluate_field_route(
        self,
        field_name: str,
        route: TSplineQueryRoute,
        coefficients: ConvertibleToArray,
        /,
        *,
        runtime: TSplineRuntimeData | None = None,
    ) -> Array:
        self._validate_query_route(route)
        index = self._field_index(field_name)
        field = self.fields[index]
        runtime_ = self.default_runtime if runtime is None else runtime
        self.validate_local_runtime(runtime_)
        values = jnp.asarray(coefficients)
        if values.shape != (self.basis_realization.global_coefficient_count,):
            raise ValueError(
                "ASTS scalar field coefficients must match the anchor count."
            )
        if field.weights_from_geometry:
            weights = runtime_.geometry_bank()[1]
        elif field.weights is not None:
            weights = field.weights
        else:
            weights = jnp.ones((self.basis_realization.global_coefficient_count,))
        denominator = self.basis_realization.evaluate(
            route.parameters, weights, cell_ids=route.cell_ids
        )
        numerator = self.basis_realization.evaluate(
            route.parameters, weights * values, cell_ids=route.cell_ids
        )
        return numerator / denominator

    def evaluate_geometry(
        self,
        points: ConvertibleToArray,
        /,
        *,
        cell_ids: ConvertibleToArray | None = None,
        runtime: TSplineRuntimeData | None = None,
        maximum_order: int = 2,
    ) -> TSplineRationalJet:
        route = self.prepare_query_route(points, cell_ids=cell_ids)
        return self.evaluate_geometry_route(
            route, runtime=runtime, maximum_order=maximum_order
        )

    def evaluate_field(
        self,
        field_name: str,
        points: ConvertibleToArray,
        coefficients: ConvertibleToArray,
        /,
        *,
        cell_ids: ConvertibleToArray | None = None,
        runtime: TSplineRuntimeData | None = None,
    ) -> Array:
        route = self.prepare_query_route(points, cell_ids=cell_ids)
        return self.evaluate_field_route(field_name, route, coefficients, runtime=runtime)

    def homogeneous_trace_constraint(self, field_name: str, /) -> ConstraintMap:
        index = self._field_index(field_name)
        field_space = self.field_spaces[index]
        full_space = field_space.vector_space
        constrained: set[int] = set()
        for (start, end), piece_row in zip(
            self.boundary_exact_segments,
            np.asarray(self.boundary_segment_piece_rows),
            strict=True,
        ):
            extraction = self.basis_realization.extractions[int(piece_row)]
            u0, u1, v0, v1 = extraction.parameter_bounds
            for local_row, coefficient_row in enumerate(extraction.coefficient_row_ids):
                coefficients = extraction.exact_extraction_operator[local_row]
                for numerator in range(7):
                    ratio = Fraction(numerator, 6)
                    u = start[0] + ratio * (end[0] - start[0])
                    v = start[1] + ratio * (end[1] - start[1])
                    tu = (u - u0) / (u1 - u0)
                    tv = (v - v0) / (v1 - v0)
                    bu = tuple(
                        Fraction(comb(3, axis)) * tu**axis * (1 - tu) ** (3 - axis)
                        for axis in range(4)
                    )
                    bv = tuple(
                        Fraction(comb(3, axis)) * tv**axis * (1 - tv) ** (3 - axis)
                        for axis in range(4)
                    )
                    value = sum(
                        (
                            coefficients[4 * u_axis + v_axis] * bu[u_axis] * bv[v_axis]
                            for u_axis in range(4)
                            for v_axis in range(4)
                        ),
                        Fraction(0),
                    )
                    if value != 0:
                        constrained.add(int(coefficient_row))
                        break
        free = np.asarray(
            [
                row
                for row in range(self.basis_realization.global_coefficient_count)
                if row not in constrained
            ],
            dtype=np.int32,
        )
        reduced = ArraySpace((free.size,), dtype=self.precision_policy.evaluation_dtype)
        relation = EdgeRelation(
            np.arange(free.size, dtype=np.int32),
            free,
            source_size=free.size,
            target_size=self.basis_realization.global_coefficient_count,
        )
        operator = SparseCoordinateOperator(
            relation,
            jnp.ones((free.size,), dtype=_coordinate_dtype(full_space)),
            source=reduced,
            target=full_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "asts2d-trim-boundary-prolongation",
                    "prepared": self.prepared_id,
                    "field": field_space.field_space_id,
                    "free_indices": tuple(int(value) for value in free),
                }
            ),
        )
        return ConstraintMap(
            full_space,
            reduced,
            operator,
            constraint_id=canonical_fingerprint(
                {
                    "kind": "asts2d-trim-boundary-constraint",
                    "prepared": self.prepared_id,
                    "field": field_space.field_space_id,
                }
            ),
        )

    def reconstruct(
        self,
        field_name: str,
        coefficients: ArrayLike,
        parameter_points: ArrayLike,
        /,
        *,
        cell_ids: ArrayLike | None = None,
        runtime: TSplineRuntimeData | None = None,
    ) -> Array:
        return self.evaluate_field(
            field_name,
            parameter_points,
            coefficients,
            cell_ids=cell_ids,
            runtime=runtime,
        )


class TSplineIsogeometricRefinement(StrictModule, NonTrainableState):
    """Atomic source-to-target ASTS basis, geometry, and field correspondence."""

    source_plan: TSplineIsogeometricPlan
    target_plan: TSplineIsogeometricPlan
    basis_refinement: ASTSRefinement
    refinement_id: str = eqx.field(static=True)
    exact_geometry_equivalence_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_plan: TSplineIsogeometricPlan,
        target_plan: TSplineIsogeometricPlan,
        basis_refinement: ASTSRefinement,
        /,
    ) -> None:
        if (
            basis_refinement.source_mesh.mesh_id != source_plan.geometry.mesh.mesh_id
            or basis_refinement.refined_mesh.mesh_id != target_plan.geometry.mesh.mesh_id
        ):
            raise ValueError(
                "IGA refinement geometry and basis relation identities disagree."
            )
        equivalence = canonical_fingerprint(
            {
                "kind": "asts2d-exact-geometry-equivalence",
                "source_state": source_plan.geometry.geometry_state_id,
                "target_state": target_plan.geometry.geometry_state_id,
                "transfer_plan": basis_refinement.transfer_plan.plan_id,
                "source_certificate": source_plan.exact_geometry_certificate.certificate_id,
                "target_certificate": target_plan.exact_geometry_certificate.certificate_id,
            }
        )
        self.source_plan = source_plan
        self.target_plan = target_plan
        self.basis_refinement = basis_refinement
        self.refinement_id = basis_refinement.refinement_id
        self.exact_geometry_equivalence_id = equivalence

    def prolong_field(
        self, field_name: str, coefficients: ArrayLike, /
    ) -> tuple[Array, Array | None]:
        name = str(field_name)
        source_field = next(
            (field for field in self.source_plan.fields if field.name == name), None
        )
        if source_field is None:
            raise KeyError(f"Unknown T-spline H1 field {name!r}.")
        if source_field.weights_from_geometry:
            return self.basis_refinement.transfer_plan.apply_rational_field(
                coefficients, self.source_plan.geometry.weights
            )
        if source_field.weights is None:
            return self.basis_refinement.transfer_plan.apply(coefficients), None
        return self.basis_refinement.transfer_plan.apply_rational_field(
            coefficients, source_field.weights
        )

    @overload
    def prolong_exact_field(
        self,
        field_name: str,
        coefficients: Sequence[Fraction],
        /,
    ) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...] | None]: ...

    @overload
    def prolong_exact_field(
        self,
        field_name: str,
        coefficients: Sequence[Sequence[Fraction]],
        /,
    ) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...] | None]: ...

    def prolong_exact_field(
        self,
        field_name: str,
        coefficients: Sequence[Sequence[Fraction] | Fraction],
        /,
    ) -> tuple[
        tuple[Fraction, ...] | tuple[tuple[Fraction, ...], ...],
        tuple[Fraction, ...] | None,
    ]:
        name = str(field_name)
        field = next(
            (field for field in self.source_plan.fields if field.name == name), None
        )
        if field is None:
            raise KeyError(f"Unknown T-spline H1 field {name!r}.")
        vector_coefficients = bool(coefficients) and isinstance(coefficients[0], Sequence)
        if vector_coefficients:
            coefficients_ = cast(Sequence[Sequence[Fraction]], coefficients)
            if field.weights_from_geometry:
                return self.basis_refinement.transfer_plan.apply_exact_rational_field(
                    coefficients_, self.source_plan.geometry.exact_weights
                )
            if field.exact_weights is None:
                return self.basis_refinement.transfer_plan.apply_exact_payload(
                    coefficients_
                ), None
            return self.basis_refinement.transfer_plan.apply_exact_rational_field(
                coefficients_, field.exact_weights
            )
        coefficients_ = cast(Sequence[Fraction], coefficients)
        if field.weights_from_geometry:
            return self.basis_refinement.transfer_plan.apply_exact_rational_field(
                coefficients_, self.source_plan.geometry.exact_weights
            )
        if field.exact_weights is None:
            return self.basis_refinement.transfer_plan.apply_exact_payload(
                coefficients_
            ), None
        return self.basis_refinement.transfer_plan.apply_exact_rational_field(
            coefficients_, field.exact_weights
        )

    def refresh_geometry(
        self, source_geometry: TSplineGeometryState, /
    ) -> TSplineGeometryState:
        """Recompose target dynamic geometry from an updated source bank through exact P."""
        if source_geometry.mesh.mesh_id != self.basis_refinement.source_mesh.mesh_id:
            raise ValueError("Updated source geometry uses a different ASTS basis.")
        return source_geometry.prolong(self.basis_refinement)


__all__ = [
    "PreparedTSplineIsogeometricDiscretization",
    "TSplineGeometryState",
    "TSplineH1FieldSpec",
    "TSplineIsogeometricPlan",
    "TSplineIsogeometricRefinement",
    "TSplineQueryRoute",
    "TSplineReferenceActions",
    "TSplineRuntimeData",
]
