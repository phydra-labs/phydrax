#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._interpolation._bspline import bspline_jet_stencil
from ..._interpolation._bspline_grid import BSplineGrid
from ..._interpolation._rational_spline import RationalSplineJet
from ..._interpolation._tensor_bspline import TensorBSplineJetPlan
from ...linalg import ArraySpace, BlockSpace, ConstraintMap
from ...sparse import EdgeRelation, SparseCoordinateOperator
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
    LocalVariationalCapabilities,
    LocalVariationalOffer,
    PreparedLocalRegion,
)
from .._spaces import DiscreteFieldSpace, TensorDofLayout
from .._support import DiscreteSupport
from .._topology import EntitySelection
from ..fem._precision import FiniteElementPrecisionPolicy
from ._actions import IsogeometricGeometryActions, IsogeometricReferenceActions
from ._basis import (
    IsogeometricFieldSpec,
    IsogeometricQuadraturePolicy,
    TensorSplineBasisSpec,
)
from ._geometry import (
    IsogeometricGeometryEvidence,
    IsogeometricH1QualificationPolicy,
    IsogeometricRuntimeData,
    NURBSGeometryState,
)
from ._topology import SplineSpanTopology


def _cell_gathers(basis: TensorSplineBasisSpec, /) -> np.ndarray:
    spans = tuple(np.asarray(axis.span_indices) for axis in basis.axes)
    routes = []
    for span_row in np.ndindex(basis.span_shape):
        active = tuple(int(spans[a][row]) for a, row in enumerate(span_row))
        local = []
        for offset in np.ndindex(tuple(degree + 1 for degree in basis.degrees)):
            control = tuple(
                span - degree + shift
                for span, degree, shift in zip(active, basis.degrees, offset, strict=True)
            )
            local.append(np.ravel_multi_index(control, basis.control_shape))
        routes.append(local)
    return np.asarray(routes, dtype=np.int32)


def _facet_routes(
    basis: TensorSplineBasisSpec, /
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int, int, int], ...]]:
    owners: list[int] = []
    local_entities: list[int] = []
    groups = []
    start = 0
    for axis in range(basis.parametric_dimension):
        tangential_shape = basis.span_shape[:axis] + basis.span_shape[axis + 1 :]
        for side in (-1, 1):
            for tangential in np.ndindex(tangential_shape):
                cell = list(tangential)
                cell.insert(axis, 0 if side < 0 else basis.span_shape[axis] - 1)
                owners.append(int(np.ravel_multi_index(tuple(cell), basis.span_shape)))
                local_entities.append(2 * axis + int(side > 0))
            stop = len(owners)
            groups.append((axis, side, start, stop))
            start = stop
    return (
        np.asarray(owners, dtype=np.int32),
        np.asarray(local_entities, dtype=np.int32),
        tuple(groups),
    )


def _query_configuration(
    basis: TensorSplineBasisSpec,
    quadrature: IsogeometricQuadraturePolicy,
    /,
    *,
    overlay_breaks: tuple[tuple[float, ...], ...] | None = None,
    facet_axis: int = -1,
    facet_side: int = 0,
) -> tuple[
    TensorBSplineJetPlan, tuple[int, ...], tuple[int, ...], tuple[int, ...], Array
]:
    stencils, axis_weights, entity_axes, point_axes = [], [], [], []
    entity_shape, point_shape, cursor = [], [], 0
    dimension = basis.parametric_dimension
    zero = (0,) * dimension
    multi_indices = [
        derivative for derivative in np.ndindex((3,) * dimension) if sum(derivative) <= 2
    ]
    for axis_index, axis in enumerate(basis.axes):
        if axis_index == facet_axis:
            endpoint = axis.parameter_interval[0 if facet_side < 0 else 1]
            queries = jnp.asarray([endpoint], dtype=axis.knots.dtype)
            spans = jnp.asarray(
                [np.asarray(axis.span_indices)[0 if facet_side < 0 else -1]],
                dtype=jnp.int32,
            )
            weights = jnp.ones((1,), dtype=axis.knots.dtype)
            point_axes.append(cursor)
            point_shape.append(1)
        else:
            count = quadrature.count_for_axis(axis_index, dimension)
            nodes, rule_weights = np.polynomial.legendre.leggauss(count)
            bounds = (
                np.asarray(axis.span_bounds)
                if overlay_breaks is None
                else np.stack(
                    (overlay_breaks[axis_index][:-1], overlay_breaks[axis_index][1:]),
                    axis=-1,
                )
            )
            half = 0.5 * (bounds[:, 1] - bounds[:, 0])
            queries = jnp.asarray(
                0.5 * (bounds[:, 0] + bounds[:, 1])[:, None] + half[:, None] * nodes
            )
            weights = jnp.asarray(half[:, None] * rule_weights)
            spans = jnp.asarray(
                np.searchsorted(np.asarray(axis.knots), np.asarray(queries), side="right")
                - 1,
                dtype=jnp.int32,
            )
            entity_axes.append(cursor)
            point_axes.append(cursor + 1)
            entity_shape.append(bounds.shape[0])
            point_shape.append(count)
        stencils.append(
            bspline_jet_stencil(
                axis.knots, queries, degree=axis.degree, maximum_order=2, spans=spans
            )
        )
        axis_weights.append(weights)
        cursor += queries.ndim
    tensor = TensorBSplineJetPlan(tuple(stencils), multi_indices=tuple(multi_indices))
    raw_weights = jnp.ones(tensor.query_shape, dtype=axis_weights[0].dtype)
    cursor = 0
    for weight in axis_weights:
        width = weight.ndim
        raw_weights = raw_weights * weight.reshape(
            (1,) * cursor
            + tuple(weight.shape)
            + (1,) * (len(tensor.query_shape) - cursor - width)
        )
        cursor += width
    permutation = tuple(entity_axes + point_axes)
    entity_shape_, point_shape_ = tuple(entity_shape) or (1,), tuple(point_shape) or (1,)
    return (
        tensor,
        permutation,
        entity_shape_,
        point_shape_,
        jnp.transpose(raw_weights, permutation).reshape(
            (prod(entity_shape_), prod(point_shape_))
        ),
    )


def _common_overlay_breaks(
    geometry_basis: TensorSplineBasisSpec,
    fields: Sequence[IsogeometricFieldSpec],
    /,
) -> tuple[tuple[float, ...], ...]:
    """Union positive-span boundaries for a common field/geometry integration mesh."""
    result = []
    for axis_index, geometry_axis in enumerate(geometry_basis.axes):
        lower, upper = geometry_axis.parameter_interval
        values = [np.asarray(geometry_axis.span_bounds).reshape((-1,))]
        for field in fields:
            axis = field.basis.axes[axis_index]
            if axis.parameter_interval != (lower, upper):
                raise ValueError(
                    "Geometry and field bases must share each parameter interval."
                )
            values.append(np.asarray(axis.span_bounds).reshape((-1,)))
        breaks = np.unique(np.concatenate(values))
        breaks = breaks[(breaks >= lower) & (breaks <= upper)]
        if breaks.size < 2 or np.any(np.diff(breaks) <= 0.0):
            raise ValueError("IGA common integration overlay has invalid span breaks.")
        result.append(tuple(float(value) for value in breaks))
    return tuple(result)


def _overlay_gathers(
    basis: TensorSplineBasisSpec, overlay_breaks: tuple[tuple[float, ...], ...], /
) -> np.ndarray:
    routes = []
    for overlay_cell in np.ndindex(tuple(len(values) - 1 for values in overlay_breaks)):
        controls = []
        for axis, row in zip(basis.axes, overlay_cell, strict=True):
            midpoint = 0.5 * (
                overlay_breaks[len(controls)][row]
                + overlay_breaks[len(controls)][row + 1]
            )
            span = int(
                np.searchsorted(np.asarray(axis.knots), midpoint, side="right") - 1
            )
            controls.append((span, axis.degree))
        routes.append(
            [
                np.ravel_multi_index(
                    tuple(
                        span - degree + shift
                        for (span, degree), shift in zip(controls, offset, strict=True)
                    ),
                    basis.control_shape,
                )
                for offset in np.ndindex(tuple(degree + 1 for _, degree in controls))
            ]
        )
    return np.asarray(routes, dtype=np.int32)


def _subset_domain(base: IntegrationDomain, rows: np.ndarray, /) -> IntegrationDomain:
    return IntegrationDomain(
        base.kind,
        np.asarray(base.entity_indices)[rows],
        base.support_id,
        base.entity_set_id,
        owner_cells=np.asarray(base.owner_cells)[rows],
        neighbour_cells=np.asarray(base.neighbour_cells)[rows],
        owner_local_entities=np.asarray(base.owner_local_entities)[rows],
        neighbour_local_entities=np.asarray(base.neighbour_local_entities)[rows],
        selection_id=base.selection_id,
    )


class IsogeometricPlan(AbstractDiscretizationPlan):
    """Structural S1 plan for one fixed-topology NURBS patch."""

    basis: TensorSplineBasisSpec
    topology: SplineSpanTopology
    geometry: NURBSGeometryState
    fields: tuple[IsogeometricFieldSpec, ...]
    quadrature_policy: IsogeometricQuadraturePolicy
    precision_policy: FiniteElementPrecisionPolicy
    qualification_policy: IsogeometricH1QualificationPolicy
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: TensorSplineBasisSpec,
        geometry: NURBSGeometryState,
        fields: IsogeometricFieldSpec | Sequence[IsogeometricFieldSpec],
        /,
        *,
        quadrature_policy: IsogeometricQuadraturePolicy,
        precision_policy: FiniteElementPrecisionPolicy | None = None,
        qualification_policy: IsogeometricH1QualificationPolicy | None = None,
    ):
        if not isinstance(basis, TensorSplineBasisSpec):
            raise TypeError("basis must be a TensorSplineBasisSpec.")
        if not isinstance(geometry, NURBSGeometryState):
            raise TypeError("geometry must be a NURBSGeometryState.")
        if geometry.control_shape != basis.control_shape:
            raise ValueError("IGA geometry and basis control shapes must match exactly.")
        field_values = (
            (fields,) if isinstance(fields, IsogeometricFieldSpec) else tuple(fields)
        )
        if not field_values or not all(
            isinstance(field, IsogeometricFieldSpec) for field in field_values
        ):
            raise TypeError("fields must contain IsogeometricFieldSpec values.")
        if len({field.name for field in field_values}) != len(field_values):
            raise ValueError("IGA field names must be unique.")
        if any(
            field.basis.parametric_dimension != basis.parametric_dimension
            or field.basis.axis_names != basis.axis_names
            for field in field_values
        ):
            raise ValueError(
                "IGA field bases must have the geometry parameter dimension and axis names."
            )
        if any(
            field.weights_from_geometry and field.basis.layout_id != basis.layout_id
            for field in field_values
        ):
            raise ValueError(
                "Geometry-owned field weights require the geometry tensor layout."
            )
        if not isinstance(quadrature_policy, IsogeometricQuadraturePolicy):
            raise TypeError("quadrature_policy must be explicit for S1 IGA.")
        precision = (
            FiniteElementPrecisionPolicy()
            if precision_policy is None
            else precision_policy
        )
        qualification = (
            IsogeometricH1QualificationPolicy()
            if qualification_policy is None
            else qualification_policy
        )
        if not isinstance(precision, FiniteElementPrecisionPolicy):
            raise TypeError(
                "precision_policy must be FiniteElementPrecisionPolicy or None."
            )
        if not isinstance(qualification, IsogeometricH1QualificationPolicy):
            raise TypeError("qualification_policy must be an IGA H1 policy or None.")
        self.basis = basis
        self.topology = SplineSpanTopology(basis)
        self.geometry = geometry
        self.fields = field_values
        self.quadrature_policy = quadrature_policy
        self.precision_policy = precision
        self.qualification_policy = qualification
        self.key = DiscretizationKey(
            "isogeometric", DiscretizationRole.PHYSICAL, domain_labels=basis.axis_names
        )
        self.capabilities = (
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.TRACE,
            DiscretizationCapability.BOUNDARY_INTEGRAL,
            DiscretizationCapability.VARIATIONAL_ASSEMBLY,
            DiscretizationCapability.MATRIX_FREE,
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        )
        geometry_layout = canonical_fingerprint(
            {
                "kind": "isogeometric-geometry-layout",
                "basis_layout": basis.layout_id,
                "control_shape": list(basis.control_shape),
                "ambient_dimension": geometry.ambient_dimension,
            }
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "isogeometric-plan",
                "basis": basis.basis_id,
                "topology": self.topology.topology_id,
                "fields": [field.field_spec_id for field in field_values],
                "quadrature": quadrature_policy.policy_id,
                "precision": precision.policy_id,
                "layout": basis.layout_id,
                "geometry_layout": geometry_layout,
                "qualification": qualification.policy_id,
            }
        )

    @classmethod
    def isoparametric(
        cls,
        grids: BSplineGrid | Sequence[BSplineGrid],
        geometry: NURBSGeometryState,
        /,
        *,
        field_name: str = "u",
        axis_names: Sequence[str] | None = None,
        quadrature_policy: IsogeometricQuadraturePolicy,
        precision_policy: FiniteElementPrecisionPolicy | None = None,
        qualification_policy: IsogeometricH1QualificationPolicy | None = None,
    ):
        grid_values = (grids,) if isinstance(grids, BSplineGrid) else tuple(grids)
        basis = TensorSplineBasisSpec(grid_values, axis_names=axis_names)
        return cls(
            basis,
            geometry,
            IsogeometricFieldSpec(
                field_name,
                basis,
                weights_from_geometry=True,
            ),
            quadrature_policy=quadrature_policy,
            precision_policy=precision_policy,
            qualification_policy=qualification_policy,
        )

    def prepare(self, /, *, numeric_version: str = "0"):
        return PreparedIsogeometricDiscretization(self, numeric_version=numeric_version)


class PreparedIsogeometricDiscretization(AbstractPreparedLocalDiscretization):
    """Prepared aligned-span, runtime-rational S1 IGA discretization."""

    basis: TensorSplineBasisSpec
    fields: tuple[IsogeometricFieldSpec, ...]
    quadrature_policy: IsogeometricQuadraturePolicy
    precision_policy: FiniteElementPrecisionPolicy
    qualification_policy: IsogeometricH1QualificationPolicy
    default_runtime: IsogeometricRuntimeData
    default_geometry_evidence: IsogeometricGeometryEvidence
    field_cell_gathers: tuple[Array, ...]
    facet_owners: Array
    facet_local_entities: Array
    facet_groups: tuple[tuple[int, int, int, int], ...] = eqx.field(static=True)
    overlay_breaks: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    cell_domain: IntegrationDomain
    exterior_facet_domain: IntegrationDomain
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

    def __init__(self, plan: IsogeometricPlan, /, *, numeric_version: str = "0"):
        if not isinstance(plan, IsogeometricPlan):
            raise TypeError("plan must be an IsogeometricPlan.")
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        basis = plan.basis
        topology = plan.topology
        runtime = IsogeometricRuntimeData(
            basis,
            plan.geometry,
            topology_id=topology.topology_id,
            numeric_version=version,
        )
        support = DiscreteSupport(
            topology.tensor_topology,
            plan.geometry.ambient_dimension,
            runtime.geometry_layout_id,
        )
        field_spaces = tuple(
            DiscreteFieldSpace(
                field.name,
                support.support_id,
                TensorDofLayout(
                    field.basis.axis_names,
                    field.basis.control_shape,
                    layout_id=field.basis.layout_id,
                ),
                ArraySpace(
                    field.basis.control_shape + field.component_shape,
                    dtype=plan.precision_policy.evaluation_dtype,
                ),
                representation="basis_coefficient",
                conformity="H1",
                reconstruction_id=canonical_fingerprint(
                    {"kind": "isogeometric-reconstruction", "field": field.field_spec_id}
                ),
                trace_space_id=canonical_fingerprint(
                    {"kind": "isogeometric-trace", "field": field.field_spec_id}
                ),
            )
            for field in plan.fields
        )
        bindings = tuple(
            LocalFieldBinding(
                field.name,
                space,
                component_shape=field.component_shape,
                public_shape=field.basis.control_shape + field.component_shape,
                execution_shape=(field.basis.coefficient_count,) + field.component_shape,
                local_width=field.basis.local_coefficient_count,
                layout_id=field.basis.layout_id,
            )
            for field, space in zip(plan.fields, field_spaces, strict=True)
        )
        overlay_breaks = _common_overlay_breaks(basis, plan.fields)
        cell_gathers = tuple(
            jnp.asarray(_overlay_gathers(field.basis, overlay_breaks))
            for field in plan.fields
        )
        facet_owners, facet_local, facet_groups = _facet_routes(basis)
        cell_domain = IntegrationDomain(
            "cell",
            np.arange(prod(len(values) - 1 for values in overlay_breaks), dtype=np.int32),
            support.support_id,
            canonical_fingerprint(
                {
                    "kind": "isogeometric-common-integration-overlay",
                    "geometry": basis.basis_id,
                    "breaks": [list(values) for values in overlay_breaks],
                }
            ),
            owner_cells=np.arange(
                prod(len(values) - 1 for values in overlay_breaks), dtype=np.int32
            ),
        )
        exterior_domain = IntegrationDomain(
            "exterior_facet",
            np.arange(facet_owners.size, dtype=np.int32),
            support.support_id,
            canonical_fingerprint(
                {"kind": "isogeometric-exterior-facets", "basis": basis.basis_id}
            ),
            owner_cells=facet_owners,
            neighbour_cells=np.full(facet_owners.shape, -1, dtype=np.int32),
            owner_local_entities=facet_local,
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                "spline axes are fixed, clamped, and nonperiodic",
                "field bases may differ from geometry and use a common span overlay",
                "fields are scalar or vector H1 with polynomial or owned rational weights",
                "NURBS denominator, rank, and orientation are checked",
            ),
            resource_counts={
                "geometry_control_coefficients": basis.coefficient_count,
                "integration_cells": int(cell_domain.entity_indices.size),
                "exterior_facets": int(facet_owners.size),
                "fields": len(plan.fields),
            },
        )
        spaces, measures, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=support,
            field_spaces=field_spaces,
            measures=(),
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        self.basis = basis
        self.fields = plan.fields
        self.quadrature_policy = plan.quadrature_policy
        self.field_cell_gathers = cell_gathers
        self.overlay_breaks = overlay_breaks
        self.precision_policy = plan.precision_policy
        self.qualification_policy = plan.qualification_policy
        self.default_runtime = runtime
        self.facet_owners = jnp.asarray(facet_owners)
        self.facet_local_entities = jnp.asarray(facet_local)
        self.facet_groups = facet_groups
        self.cell_domain = cell_domain
        self.exterior_facet_domain = exterior_domain
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
                "kind": "prepared-isogeometric-discretization",
                "plan": plan.plan_id,
                "topology": topology.topology_id,
                "geometry_layout": runtime.geometry_layout_id,
                "numeric_version": version,
            }
        )
        region = self.prepare_local_regions(
            self.cell_domain,
            field_names=(self.field_spaces[0].name,),
            maximum_derivative_order=1,
            kernel_mode="sum_factorized",
        )[0]
        geometry_actions = region.geometry_actions
        if not isinstance(geometry_actions, IsogeometricGeometryActions):
            raise TypeError("IGA regions require isogeometric geometry actions.")
        evidence = geometry_actions.evidence(self.default_runtime)
        self.default_geometry_evidence = self.qualification_policy.check(evidence)
        self.default_geometry_evidence.minimum_rank_ratio.block_until_ready()

    @property
    def cell_gathers(self) -> Array:
        """Primary-field gathers retained for the single-field public workflow."""
        return self.field_cell_gathers[0]

    @property
    def precision_evidence(self):
        return self.precision_policy.evidence()

    @property
    def cell_kind(self) -> str:
        return ("interval", "quadrilateral", "hexahedron")[
            self.basis.parametric_dimension - 1
        ]

    def _field_index(self, field_name: str, /) -> int:
        name = str(field_name)
        for index, field in enumerate(self.field_spaces):
            if field.name == name:
                return index
        raise KeyError(f"Unknown isogeometric field {name!r}.")

    def local_variational_capabilities(self, /) -> LocalVariationalCapabilities:
        semantics = (
            "exact_interpolation_transpose",
            "exact_gradient_transpose",
            "exact_hessian_transpose",
            "exact_trace_transpose",
            "runtime_rational_weights",
        )
        return LocalVariationalCapabilities(
            "isogeometric-local-provider",
            (
                LocalVariationalOffer(
                    "prepared-local",
                    ("cell",),
                    (
                        "diffusion",
                        "tensor-diffusion",
                        "mass",
                        "source",
                        "functional",
                    ),
                    ("value", "grad", "hessian"),
                    ("value", "gradient", "hessian"),
                    ("sum_factorized",),
                    ("matrix_free",),
                    ("isogeometric-direct-tensor",),
                    automatic_kernel_mode="sum_factorized",
                    automatic_operator_realization="matrix_free",
                    automatic_reference_realization_id="isogeometric-direct-tensor",
                    action_semantics=semantics,
                ),
                LocalVariationalOffer(
                    "prepared-local",
                    ("exterior_facet",),
                    ("boundary-load", "functional"),
                    ("value", "normal-trace"),
                    ("value", "trace"),
                    ("sum_factorized",),
                    ("matrix_free",),
                    ("isogeometric-direct-tensor",),
                    automatic_kernel_mode="sum_factorized",
                    automatic_operator_realization="matrix_free",
                    automatic_reference_realization_id="isogeometric-direct-tensor",
                    action_semantics=semantics,
                ),
            ),
        )

    def local_field_binding(self, name: str, /) -> LocalFieldBinding:
        return self.bindings[self._field_index(name)]

    def validate_local_runtime(self, runtime: object, /) -> None:
        if not isinstance(runtime, IsogeometricRuntimeData):
            raise TypeError("IGA execution requires IsogeometricRuntimeData.")
        if (
            runtime.topology_id != self.default_runtime.topology_id
            or runtime.geometry_layout_id != self.default_runtime.geometry_layout_id
            or runtime.control_points.shape != self.default_runtime.control_points.shape
            or runtime.weights.shape != self.default_runtime.weights.shape
        ):
            raise ValueError(
                "IGA runtime does not match the prepared topology and layout."
            )

    def prepare_runtime(
        self, geometry: NURBSGeometryState | None = None, /, *, numeric_version: str
    ) -> IsogeometricRuntimeData:
        state = (
            NURBSGeometryState(
                self.default_runtime.control_points, self.default_runtime.weights
            )
            if geometry is None
            else geometry
        )
        result = IsogeometricRuntimeData(
            self.basis,
            state,
            topology_id=self.default_runtime.topology_id,
            numeric_version=numeric_version,
        )
        self.validate_local_runtime(result)
        return result

    def integration_domain(
        self, kind: str, selection: EntitySelection | None = None, /
    ) -> IntegrationDomain:
        kind_ = str(kind)
        if kind_ == "cell":
            base = self.cell_domain
        elif kind_ == "exterior_facet":
            base = self.exterior_facet_domain
        else:
            raise ValueError("S1 IGA supports only cell and exterior-facet domains.")
        if selection is None:
            return base
        if not isinstance(selection, EntitySelection):
            raise TypeError("selection must be EntitySelection or None.")
        if selection.entity_set_id != base.entity_set_id:
            raise ValueError("IGA selection does not match the requested domain.")
        rows = np.flatnonzero(
            np.asarray(selection.mask, dtype=bool)[np.asarray(base.entity_indices)]
        )
        return _subset_domain(base, rows)

    def prepare_local_regions(
        self,
        domain: IntegrationDomain,
        /,
        *,
        field_names: tuple[str, ...],
        maximum_derivative_order: int,
        kernel_mode: str,
    ) -> tuple[PreparedLocalRegion, ...]:
        if not isinstance(domain, IntegrationDomain):
            raise TypeError("domain must be an IntegrationDomain.")
        if domain.support_id != self.support.support_id:
            raise ValueError("IGA integration domain belongs to another support.")
        if str(kernel_mode) != "sum_factorized":
            raise ValueError("S1 IGA supports only sum_factorized local kernels.")
        order = int(maximum_derivative_order)
        if order < 0 or order > 2:
            raise ValueError("IGA supports local derivative orders zero through two.")
        names = tuple(str(name) for name in field_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("IGA local regions require unique non-empty field names.")
        for name in names:
            self._field_index(name)
        parameter_scales = jnp.asarray(
            [
                axis.parameter_interval[1] - axis.parameter_interval[0]
                for axis in self.basis.axes
            ]
        )
        if domain.kind == "cell":
            if domain.entity_set_id != self.cell_domain.entity_set_id:
                raise ValueError("IGA cell domain has an incompatible entity set.")
            rows = np.asarray(domain.entity_indices, dtype=np.int32)
            geometry_tensor, permutation, entity_shape, point_shape, reference_weights = (
                _query_configuration(
                    self.basis,
                    self.quadrature_policy,
                    overlay_breaks=self.overlay_breaks,
                )
            )
            references = tuple(
                IsogeometricReferenceActions(
                    _query_configuration(
                        self.fields[self._field_index(name)].basis,
                        self.quadrature_policy,
                        overlay_breaks=self.overlay_breaks,
                    )[0],
                    self.fields[self._field_index(name)].weights,
                    rows,
                    permutation,
                    entity_shape,
                    point_shape,
                    topology_id=self.default_runtime.topology_id,
                    geometry_layout_id=self.default_runtime.geometry_layout_id,
                    maximum_derivative_order=order,
                    structural_id=domain.domain_id,
                    is_trace=False,
                    field_weights_from_geometry=self.fields[
                        self._field_index(name)
                    ].weights_from_geometry,
                )
                for name in names
            )
            geometry = IsogeometricGeometryActions(
                geometry_tensor,
                rows,
                reference_weights[rows],
                permutation,
                entity_shape,
                point_shape,
                parameter_scales,
                self.qualification_policy,
                topology_id=self.default_runtime.topology_id,
                runtime_layout_id=self.default_runtime.geometry_layout_id,
                domain_kind="cell",
                structural_id=domain.domain_id,
            )
            return (
                PreparedLocalRegion(
                    domain,
                    names,
                    tuple(
                        self.field_cell_gathers[self._field_index(name)][rows]
                        for name in names
                    ),
                    references,
                    geometry,
                    block_name="patch",
                    cell_kind=self.cell_kind,
                ),
            )
        if domain.kind != "exterior_facet":
            raise ValueError("S1 IGA supports only cell and exterior-facet regions.")
        if domain.entity_set_id != self.exterior_facet_domain.entity_set_id:
            raise ValueError("IGA facet domain has an incompatible entity set.")
        selected = np.asarray(domain.entity_indices, dtype=np.int32)
        regions = []
        for axis, side, start, stop in self.facet_groups:
            active = selected[(selected >= start) & (selected < stop)]
            if active.size == 0:
                continue
            local_rows = active - start
            group_domain = IntegrationDomain(
                "exterior_facet",
                active,
                domain.support_id,
                domain.entity_set_id,
                owner_cells=np.asarray(self.exterior_facet_domain.owner_cells)[active],
                neighbour_cells=np.full(active.shape, -1, dtype=np.int32),
                owner_local_entities=np.asarray(
                    self.exterior_facet_domain.owner_local_entities
                )[active],
                selection_id=domain.selection_id,
            )
            tensor, permutation, entity_shape, point_shape, reference_weights = (
                _query_configuration(
                    self.basis,
                    self.quadrature_policy,
                    facet_axis=axis,
                    facet_side=side,
                )
            )
            references = tuple(
                IsogeometricReferenceActions(
                    tensor,
                    self.fields[self._field_index(name)].weights,
                    local_rows,
                    permutation,
                    entity_shape,
                    point_shape,
                    topology_id=self.default_runtime.topology_id,
                    geometry_layout_id=self.default_runtime.geometry_layout_id,
                    maximum_derivative_order=order,
                    structural_id=group_domain.domain_id,
                    is_trace=True,
                    field_weights_from_geometry=self.fields[
                        self._field_index(name)
                    ].weights_from_geometry,
                )
                for name in names
            )
            geometry = IsogeometricGeometryActions(
                tensor,
                local_rows,
                reference_weights[local_rows],
                permutation,
                entity_shape,
                point_shape,
                parameter_scales,
                self.qualification_policy,
                topology_id=self.default_runtime.topology_id,
                runtime_layout_id=self.default_runtime.geometry_layout_id,
                domain_kind="exterior_facet",
                structural_id=group_domain.domain_id,
                facet_axis=axis,
                facet_side=side,
            )
            owners = np.asarray(group_domain.owner_cells, dtype=np.int32)
            side_name = "lower" if side < 0 else "upper"
            regions.append(
                PreparedLocalRegion(
                    group_domain,
                    names,
                    tuple(
                        self.field_cell_gathers[self._field_index(name)][owners]
                        for name in names
                    ),
                    references,
                    geometry,
                    block_name=f"patch-boundary-{self.basis.axes[axis].name}-{side_name}",
                    cell_kind=self.cell_kind,
                )
            )
        return tuple(regions)

    def reconstruct(
        self,
        field_name: str,
        coefficients: ArrayLike,
        axis_points: Sequence[ArrayLike],
        /,
        *,
        runtime: IsogeometricRuntimeData | None = None,
    ) -> Array:
        field_index = self._field_index(field_name)
        field = self.fields[field_index]
        values = self.field_spaces[field_index].vector_space.validate(coefficients)
        queries = tuple(jnp.asarray(points) for points in axis_points)
        if len(queries) != field.basis.parametric_dimension or any(
            points.ndim != 1 or points.size == 0 for points in queries
        ):
            raise ValueError("IGA reconstruction requires one nonempty rank-1 axis grid.")
        stencils = tuple(
            bspline_jet_stencil(axis.knots, points, degree=axis.degree, maximum_order=0)
            for axis, points in zip(field.basis.axes, queries, strict=True)
        )
        tensor = TensorBSplineJetPlan(
            stencils, multi_indices=((0,) * field.basis.parametric_dimension,)
        )
        realized = self.default_runtime if runtime is None else runtime
        self.validate_local_runtime(realized)
        weights = (
            realized.weights
            if field.weights_from_geometry
            else (
                jnp.ones(field.basis.control_shape)
                if field.weights is None
                else field.weights
            )
        )
        return RationalSplineJet(tensor, weights).apply(values)

    def geometry_evidence(
        self, runtime: IsogeometricRuntimeData | None = None, /
    ) -> IsogeometricGeometryEvidence:
        realized = self.default_runtime if runtime is None else runtime
        self.validate_local_runtime(realized)
        region = self.prepare_local_regions(
            self.cell_domain,
            field_names=(self.field_spaces[0].name,),
            maximum_derivative_order=1,
            kernel_mode="sum_factorized",
        )[0]
        geometry_actions = region.geometry_actions
        if not isinstance(geometry_actions, IsogeometricGeometryActions):
            raise TypeError("IGA regions require isogeometric geometry actions.")
        return self.qualification_policy.check(geometry_actions.evidence(realized))

    def homogeneous_trace_constraint(self, field_name: str, /) -> ConstraintMap:
        field_index = self._field_index(field_name)
        field = self.fields[field_index]
        full_space = self.field_spaces[field_index].vector_space
        boundary = np.zeros(field.basis.control_shape, dtype=bool)
        for axis in range(field.basis.parametric_dimension):
            lower: list[slice | int] = [slice(None)] * field.basis.parametric_dimension
            upper: list[slice | int] = [slice(None)] * field.basis.parametric_dimension
            lower[axis], upper[axis] = 0, -1
            boundary[tuple(lower)] = True
            boundary[tuple(upper)] = True
        component_count = prod(field.component_shape) or 1
        free_sites = np.flatnonzero(~boundary.reshape((-1,))).astype(np.int32)
        free = (
            free_sites[:, None] * component_count
            + np.arange(component_count, dtype=np.int32)[None, :]
        ).reshape((-1,))
        reduced_space = ArraySpace(
            (free.size,), dtype=self.precision_policy.evaluation_dtype
        )
        relation = EdgeRelation(
            np.arange(free.size, dtype=np.int32),
            free,
            source_size=free.size,
            target_size=full_space.size,
        )
        operator = SparseCoordinateOperator(
            relation,
            jnp.ones((free.size,), dtype=full_space.dtype),
            source=reduced_space,
            target=full_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "isogeometric-homogeneous-trace-prolongation",
                    "prepared": self.prepared_id,
                    "field": self.field_spaces[field_index].field_space_id,
                    "free_indices": tuple(int(value) for value in free),
                }
            ),
        )
        return ConstraintMap(
            full_space,
            reduced_space,
            operator,
            constraint_id=canonical_fingerprint(
                {
                    "kind": "isogeometric-homogeneous-trace-constraint",
                    "prepared": self.prepared_id,
                    "field": self.field_spaces[field_index].field_space_id,
                }
            ),
        )


__all__ = ["IsogeometricPlan", "PreparedIsogeometricDiscretization"]
