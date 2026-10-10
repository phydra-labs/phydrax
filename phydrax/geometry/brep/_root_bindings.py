#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact vertex definitions and verified aliases of source intersection roots."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model import register_artifact_value
from ..._strict import StrictModule
from ...typing import ConvertibleToArray, Float64, parse
from .._atlas import AbstractTrimCurve
from .._interval_enclosure import interval_add
from ._correspondence import certify_curve_surface, prove_curve_correspondence
from ._intersection import (
    BranchRootEndpoint,
    CurveSurfaceIntersectionRoot,
    IntersectionCurvePointRoot,
    NativePeriodEndpoint,
    RootEndpoint,
    TrimIntersectionRoot,
    TrimRootEndpoint,
    TripleSurfaceIntersectionRoot,
)
from ._intersection_curve import (
    _trim_source_jet,
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionPCurve,
    SurfaceRegion,
)
from ._patches import AbstractCurve, AbstractSurfacePatch
from ._placed import _validate_pose, PlacedSurface, source_transform_bounds


def _same_surface(first: AbstractSurfacePatch, second: AbstractSurfacePatch, /) -> bool:
    from ._model import _carrier_payload

    return canonical_fingerprint(_carrier_payload(first)) == canonical_fingerprint(
        _carrier_payload(second)
    )


class BRepRootSupport(StrictModule):
    """A face's exact UV root mapped through its authoritative surface definition."""

    patch: AbstractSurfacePatch
    root: TrimIntersectionRoot
    root_id: str = eqx.field(static=True)

    def __init__(self, patch: AbstractSurfacePatch, root: TrimIntersectionRoot) -> None:
        from ._model import _carrier_payload

        if not isinstance(patch, AbstractSurfacePatch) or not isinstance(
            root, TrimIntersectionRoot
        ):
            raise TypeError(
                "A vertex support requires a native surface and certified trim root."
            )
        patch.validate_parameter_box(root.point_enclosure())
        self.patch, self.root = patch, root
        self.root_id = canonical_fingerprint(
            {
                "kind": "brep-root-support",
                "surface": _carrier_payload(patch),
                "trim_root": root.root_id,
            }
        )

    def point_enclosure(self) -> np.ndarray:
        return self.patch.bounding_box(self.root.point_enclosure())


def _support_is_spatial_alias(
    support: BRepRootSupport, spatial: CurveSurfaceIntersectionRoot, /
) -> bool:
    """Prove the UV root solves the SAME unique spatial source system.

    Containment in its root box is only the last premise: zero whole-edge
    source correspondence and the coupled branch's other generating surface
    establish that the mapped UV root is a solution of that spatial system.
    Merely overlapping point boxes does not establish this implication.
    """
    for edge_operand, branch_operand in (("first", "second"), ("second", "first")):
        edge = TrimRootEndpoint(support.root, edge_operand)
        branch = TrimRootEndpoint(support.root, branch_operand)
        pcurve = edge.carrier
        intersection = branch.carrier
        if not isinstance(pcurve, (AbstractCurve, IntersectionPCurve)) or not isinstance(
            intersection, IntersectionPCurve
        ):
            continue
        generating = (
            intersection.curve.first.patch
            if intersection.side == "first"
            else intersection.curve.second.patch
        )
        opposite = (
            intersection.curve.second.patch
            if intersection.side == "first"
            else intersection.curve.first.patch
        )
        if not _same_surface(generating, support.patch) or not _same_surface(
            opposite, spatial.surface.patch
        ):
            continue
        first, last = edge.parameter_enclosure()
        witness = certify_curve_surface(
            spatial.curve,
            pcurve,
            support.patch,
            support.root.point_enclosure(),
            first,
            last,
            point=np.zeros((3,), dtype=np.float64),
            tolerance=0.0,
            endpoint_roots=(edge, edge),
        )
        if not witness.complete or witness.deviation_bound != 0.0:
            continue
        tau_lower, tau_upper = branch.parameter_enclosure()
        other_side = "second" if intersection.side == "first" else "first"
        uv = intersection.curve.p_curve(other_side).enclosure(tau_lower, tau_upper)
        lower, upper = np.concatenate(([first], uv[0])), np.concatenate(([last], uv[1]))
        if np.all(lower >= spatial.parameter_lower) and np.all(
            upper <= spatial.parameter_upper
        ):
            return True
    return False


def _support_has_generating_face(
    support: BRepRootSupport,
    lifts: tuple[BRepCurveSurfaceLift, ...],
    /,
) -> bool:
    """Both trim operands are p-curves of the support face.

    An intersection p-curve names its generating surface; a native p-curve
    needs a certified whole-edge lift onto the support face.
    """
    from ._model import _carrier_payload

    for source in (support.root.first, support.root.second):
        while isinstance(source, CurveTrimSegment):
            source = source.curve
        if isinstance(source, IntersectionPCurve):
            patch = (
                source.curve.first.patch
                if source.side == "first"
                else source.curve.second.patch
            )
            if not _same_surface(patch, support.patch):
                return False
            continue
        source_id = canonical_fingerprint(_carrier_payload(source))
        if not any(
            _same_surface(lift.surface.patch, support.patch)
            and canonical_fingerprint(_carrier_payload(lift.pcurve)) == source_id
            for lift in lifts
        ):
            return False
    return True


class BRepCurveSurfaceLift(StrictModule):
    """An exact source edge parameter mapped onto one declared supporting surface."""

    curve: AbstractCurve | IntersectionCurve
    pcurve: AbstractCurve | IntersectionPCurve
    surface: SurfaceRegion
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    lift_id: str = eqx.field(static=True)

    def __init__(
        self,
        curve: AbstractCurve | IntersectionCurve,
        pcurve: AbstractCurve | IntersectionPCurve,
        surface: SurfaceRegion,
        first: float,
        last: float,
        /,
    ) -> None:
        from ._model import _carrier_payload

        if (
            not isinstance(curve, (AbstractCurve, IntersectionCurve))
            or not isinstance(pcurve, (AbstractCurve, IntersectionPCurve))
            or not isinstance(surface, SurfaceRegion)
        ):
            raise TypeError(
                "A source lift requires exact native curve, pcurve and surface definitions."
            )
        first_, last_ = curve.validate_range(first, last)
        pcurve.validate_range(first_, last_)
        witness = certify_curve_surface(
            curve,
            pcurve,
            surface.patch,
            surface.parameter_box,
            first_,
            last_,
            point=np.zeros((3,), dtype=np.float64),
            tolerance=0.0,
        )
        if not witness.complete or witness.deviation_bound != 0.0:
            raise ValueError(
                "The source lift has no zero-error whole-edge correspondence certificate."
            )
        self.curve, self.pcurve, self.surface = curve, pcurve, surface
        self.first, self.last = first_, last_
        self.lift_id = canonical_fingerprint(
            {
                "kind": "brep-exact-source-lift",
                "curve": _carrier_payload(curve),
                "pcurve": _carrier_payload(pcurve),
                "surface": _carrier_payload(surface),
                "range": (first_, last_),
            }
        )


def _spatial_root_has_joint_lifts(
    root: CurveSurfaceIntersectionRoot,
    joint: TripleSurfaceIntersectionRoot,
    lifts: tuple[BRepCurveSurfaceLift, ...],
    /,
) -> bool:
    from ._model import _carrier_payload

    source_id = canonical_fingerprint(_carrier_payload(root.curve))
    first, last = float(root.parameter_lower[0]), float(root.parameter_upper[0])
    for index, region in enumerate((joint.first, joint.second, joint.third)):
        target_lower = joint.parameter_lower[2 * index : 2 * index + 2]
        target_upper = joint.parameter_upper[2 * index : 2 * index + 2]
        images = []
        if _same_surface(root.surface.patch, region.patch):
            images.append(np.stack((root.parameter_lower[1:], root.parameter_upper[1:])))
        for lift in lifts:
            if (
                _same_surface(lift.surface.patch, region.patch)
                and canonical_fingerprint(_carrier_payload(lift.curve)) == source_id
                and lift.first <= first <= last <= lift.last
            ):
                images.append(lift.pcurve.bounding_box(first, last))
        if not any(
            np.all(image[0] >= target_lower) and np.all(image[1] <= target_upper)
            for image in images
        ):
            return False
    return True


def interval_jacobian_full_rank(lower: np.ndarray, upper: np.ndarray, /) -> bool:
    """Certify full column rank of every matrix in a ``(3, 2)`` interval hull.

    A preconditioned coordinate-minor contraction proves the projected map,
    hence the surface map, injective on any convex (or glued convex) chart
    whose derivative lies in the hull, by the mean-value form.
    """
    from ._intersection import _approximate_inverses, _point_times_interval

    magnitude = 0.5 * (lower + upper)
    for rows in ((0, 1), (0, 2), (1, 2)):
        matrix = magnitude[list(rows)]
        determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
        if not np.isfinite(determinant) or determinant == 0.0:
            continue
        preconditioner = _approximate_inverses(matrix[None])
        product_lower, product_upper = _point_times_interval(
            preconditioner,
            lower[list(rows)][None],
            upper[list(rows)][None],
        )
        residual_lower = np.nextafter(np.eye(2)[None] - product_upper, -np.inf)
        residual_upper = np.nextafter(np.eye(2)[None] - product_lower, np.inf)
        row_sum = np.zeros((1, 2), dtype=np.float64)
        absolute = np.maximum(np.abs(residual_lower), np.abs(residual_upper))
        for column in range(2):
            row_sum = np.nextafter(row_sum + absolute[:, :, column], np.inf)
        if np.max(row_sum) < 1.0:
            return True
    return False


def surface_chart_regular(patch: AbstractSurfacePatch, bounds: np.ndarray, /) -> bool:
    """Certify full column rank throughout a closed native source chart box."""
    lower, upper = patch.derivative_bounds(bounds, order=1)
    return interval_jacobian_full_rank(lower, upper)


def _native_root_operand(
    source: AbstractTrimCurve | AbstractCurve,
    /,
) -> tuple[AbstractCurve, Fraction, Fraction] | None:
    """Unwrap segments without rounding their exact scalar expression."""
    scale, offset = Fraction(1), Fraction(0)
    while isinstance(source, CurveTrimSegment):
        first, last = Fraction(source.first), Fraction(source.last)
        local_scale = first - last if source.reversed else last - first
        local_offset = last if source.reversed else first
        scale, offset = local_scale * scale, local_scale * offset + local_offset
        source = source.curve
    return (source, scale, offset) if isinstance(source, AbstractCurve) else None


def certify_native_root_alias(
    primary: BRepRootSupport,
    alias: BRepRootSupport,
    lifts: tuple[BRepCurveSurfaceLift, ...],
    /,
) -> bool:
    """Prove two native roots solve the same unique planar source system.

    Both physical carriers must correspond exactly, not merely one common
    edge. Transported alias parameters must lie in the primary uniqueness
    box. A planar primary chart supplies global injectivity; both source
    charts must remain regular on their root boxes, so a pole/apex collapse
    cannot borrow an unrelated planar vertex identity.
    """
    from ._model import _carrier_payload
    from ._patches import PlanePatch

    if (
        not isinstance(primary.patch, PlanePatch)
        or not surface_chart_regular(primary.patch, primary.root.point_enclosure())
        or not surface_chart_regular(alias.patch, alias.root.point_enclosure())
    ):
        return False

    def operand_lifts(
        support: BRepRootSupport,
        source: AbstractTrimCurve,
        /,
    ) -> (
        tuple[tuple[AbstractCurve, Fraction, Fraction], tuple[BRepCurveSurfaceLift, ...]]
        | None
    ):
        operand = _native_root_operand(source)
        if operand is None:
            return None
        identity = canonical_fingerprint(_carrier_payload(operand[0]))
        selected = tuple(
            lift
            for lift in lifts
            if _same_surface(lift.surface.patch, support.patch)
            and canonical_fingerprint(_carrier_payload(lift.pcurve)) == identity
        )
        return operand, selected

    first = tuple(
        operand_lifts(primary, source)
        for source in (primary.root.first, primary.root.second)
    )
    second = tuple(
        operand_lifts(alias, source) for source in (alias.root.first, alias.root.second)
    )
    for order in ((0, 1), (1, 0)):
        matched = True
        for index, opposite in enumerate(order):
            target, source = first[index], second[opposite]
            if target is None or source is None:
                return False
            (_, scale, offset), target_lifts = target
            (_, source_scale, source_offset), source_lifts = source
            found = False
            for target_lift in target_lifts:
                for source_lift in source_lifts:
                    mapping = prove_curve_correspondence(
                        target_lift.curve, source_lift.curve
                    )
                    if mapping is None or not isinstance(mapping.offset, Fraction):
                        continue
                    lo, hi = (
                        alias.root.parameter_lower[opposite],
                        alias.root.parameter_upper[opposite],
                    )
                    source_interval = sorted(
                        (
                            source_scale * Fraction(float(lo)) + source_offset,
                            source_scale * Fraction(float(hi)) + source_offset,
                        )
                    )
                    if not (
                        Fraction(source_lift.first) <= source_interval[0]
                        and source_interval[1] <= Fraction(source_lift.last)
                    ):
                        continue
                    spatial = sorted(
                        mapping.scale * value + mapping.offset
                        for value in source_interval
                    )
                    if not (
                        Fraction(target_lift.first) <= spatial[0]
                        and spatial[1] <= Fraction(target_lift.last)
                    ):
                        continue
                    parameters = sorted((value - offset) / scale for value in spatial)
                    if Fraction(float(primary.root.parameter_lower[index])) <= parameters[
                        0
                    ] and parameters[1] <= Fraction(
                        float(primary.root.parameter_upper[index])
                    ):
                        found = True
                        break
                if found:
                    break
            if not found:
                matched = False
                break
        if matched:
            return True
    return False


class BRepVertexRoot(StrictModule):
    """An exact vertex root; alternative face supports require a source alias proof."""

    primary: (
        BRepRootSupport
        | CurveSurfaceIntersectionRoot
        | TripleSurfaceIntersectionRoot
        | IntersectionCurvePointRoot
        | BRepPlacedVertex
    )
    aliases: tuple[BRepRootSupport, ...]
    spatial_root: CurveSurfaceIntersectionRoot | None
    joint_root: TripleSurfaceIntersectionRoot | None
    source_edge_lifts: tuple[BRepCurveSurfaceLift, ...]
    root_id: str = eqx.field(static=True)

    def __init__(
        self,
        primary: BRepRootSupport
        | CurveSurfaceIntersectionRoot
        | TripleSurfaceIntersectionRoot
        | IntersectionCurvePointRoot
        | BRepPlacedVertex,
        /,
        *,
        aliases: tuple[BRepRootSupport, ...] = (),
        spatial_root: CurveSurfaceIntersectionRoot | None = None,
        joint_root: TripleSurfaceIntersectionRoot | None = None,
        source_edge_lifts: tuple[BRepCurveSurfaceLift, ...] = (),
    ) -> None:
        if not isinstance(
            primary,
            (
                BRepRootSupport,
                CurveSurfaceIntersectionRoot,
                TripleSurfaceIntersectionRoot,
                IntersectionCurvePointRoot,
                BRepPlacedVertex,
            ),
        ):
            raise TypeError(
                "A vertex root requires a canonical certified source definition."
            )
        if not isinstance(aliases, tuple) or any(
            not isinstance(alias, BRepRootSupport) for alias in aliases
        ):
            raise TypeError("Vertex aliases must be native face-root supports.")
        if spatial_root is not None and not isinstance(
            spatial_root, CurveSurfaceIntersectionRoot
        ):
            raise TypeError("spatial_root must be a canonical spatial root or None.")
        if not isinstance(source_edge_lifts, tuple) or any(
            not isinstance(lift, BRepCurveSurfaceLift) for lift in source_edge_lifts
        ):
            raise TypeError(
                "source_edge_lifts must contain certified BRepCurveSurfaceLift values."
            )
        if isinstance(primary, BRepPlacedVertex) and (
            aliases
            or spatial_root is not None
            or joint_root is not None
            or source_edge_lifts
        ):
            raise ValueError(
                "A placed vertex retains its complete original source root; it cannot borrow target-space aliases."
            )
        spatial = (
            primary if isinstance(primary, CurveSurfaceIntersectionRoot) else spatial_root
        )
        if isinstance(primary, IntersectionCurvePointRoot) and (
            aliases
            or spatial_root is not None
            or joint_root is not None
            or source_edge_lifts
        ):
            raise ValueError(
                "A fixed branch point cannot borrow unrelated intersection root aliases."
            )
        if (
            isinstance(primary, CurveSurfaceIntersectionRoot)
            and spatial_root is not None
            and primary.root_id != spatial_root.root_id
        ):
            raise ValueError(
                "A vertex cannot claim two independent spatial source roots."
            )
        if joint_root is not None and not isinstance(
            joint_root, TripleSurfaceIntersectionRoot
        ):
            raise TypeError("joint_root must be a canonical joint source root or None.")
        joint = (
            primary if isinstance(primary, TripleSurfaceIntersectionRoot) else joint_root
        )
        if (
            isinstance(primary, TripleSurfaceIntersectionRoot)
            and joint_root is not None
            and primary.root_id != joint_root.root_id
        ):
            raise ValueError("A vertex cannot claim two independent joint source roots.")
        if joint is not None and isinstance(
            primary, (BRepRootSupport, CurveSurfaceIntersectionRoot)
        ):
            source = primary.root if isinstance(primary, BRepRootSupport) else primary
            joint_primary = joint.certifies_root(source)
            if (
                not joint_primary
                and spatial is not None
                and _spatial_root_has_joint_lifts(spatial, joint, source_edge_lifts)
            ):
                joint_primary = isinstance(
                    primary, CurveSurfaceIntersectionRoot
                ) or _support_is_spatial_alias(primary, spatial)
            if not joint_primary:
                raise ValueError(
                    "The primary vertex root has no common joint source-system certificate."
                )
        if (
            joint is not None
            and spatial is not None
            and not (
                joint.certifies_root(spatial)
                or _spatial_root_has_joint_lifts(spatial, joint, source_edge_lifts)
            )
        ):
            raise ValueError(
                "The spatial vertex root has no common joint source-system certificate."
            )
        supports = (
            (primary, *aliases) if isinstance(primary, BRepRootSupport) else aliases
        )
        for support in supports:
            if (
                isinstance(primary, BRepRootSupport)
                and support.root_id == primary.root_id
            ):
                continue
            if (
                joint is not None
                and _support_has_generating_face(support, source_edge_lifts)
                and joint.certifies_root(
                    support.root, source_edge_lifts=source_edge_lifts
                )
            ):
                continue
            if isinstance(primary, BRepRootSupport) and certify_native_root_alias(
                primary, support, source_edge_lifts
            ):
                continue
            if spatial is None or not _support_is_spatial_alias(support, spatial):
                raise ValueError(
                    "The vertex UV alias has no verified common spatial source-root correspondence."
                )
        self.primary, self.aliases, self.spatial_root, self.joint_root = (
            primary,
            aliases,
            spatial,
            joint,
        )
        self.source_edge_lifts = source_edge_lifts
        self.root_id = (
            primary.point_id
            if isinstance(primary, IntersectionCurvePointRoot)
            else (
                joint.root_id
                if joint is not None
                else (primary.root_id if spatial is None else spatial.root_id)
            )
        )

    @classmethod
    def from_curve_point(
        cls, curve: IntersectionCurve, parameter: float = 0.0, /
    ) -> BRepVertexRoot:
        """Define a vertex by a fixed, certified implicit branch parameter atom."""
        return cls(IntersectionCurvePointRoot(curve, parameter))

    def point_enclosure(self) -> np.ndarray:
        return (
            self.primary.point_enclosure()
            if self.joint_root is None
            else self.joint_root.point_enclosure()
        )

    def evaluate(self) -> tuple[np.ndarray, float, bool]:
        if isinstance(
            self.primary,
            (
                CurveSurfaceIntersectionRoot,
                TripleSurfaceIntersectionRoot,
                IntersectionCurvePointRoot,
                BRepPlacedVertex,
            ),
        ):
            return self.primary.evaluate()
        uv, _, certified = self.primary.root.evaluate()
        point = np.asarray(
            self.primary.patch.evaluate(jnp.asarray(uv, dtype=jnp.float64))
        )
        bound = float(
            np.max(np.nextafter(np.abs(self.point_enclosure() - point), np.inf))
        )
        return point, bound, certified

    def supports_uv(
        self, patch: AbstractSurfacePatch, root: TrimIntersectionRoot, /
    ) -> bool:
        if isinstance(self.primary, BRepPlacedVertex):
            if not isinstance(patch, PlacedSurface) or not self.primary.matches_pose(
                patch
            ):
                return False
            source = self.primary.source_root
            return source is not None and source.supports_uv(patch.definition, root)
        supports = (
            (self.primary, *self.aliases)
            if isinstance(self.primary, BRepRootSupport)
            else self.aliases
        )
        return any(
            support.root.root_id == root.root_id and _same_surface(support.patch, patch)
            for support in supports
        )

    def supports_endpoint(self, endpoint: RootEndpoint, /) -> bool:
        if isinstance(endpoint, NativePeriodEndpoint):
            return False
        if isinstance(self.primary, BRepPlacedVertex):
            return (
                self.primary.source_root is not None
                and self.primary.source_root.supports_endpoint(endpoint)
            )
        if isinstance(self.primary, IntersectionCurvePointRoot) and isinstance(
            endpoint.root, IntersectionCurvePointRoot
        ):
            return self.primary.point_id == endpoint.root.point_id
        if (
            self.joint_root is not None
            and isinstance(
                endpoint.root,
                (
                    TrimIntersectionRoot,
                    CurveSurfaceIntersectionRoot,
                    TripleSurfaceIntersectionRoot,
                ),
            )
            and self.joint_root.certifies_root(
                endpoint.root, source_edge_lifts=self.source_edge_lifts
            )
        ):
            return True
        if isinstance(endpoint.root, CurveSurfaceIntersectionRoot):
            return (
                self.spatial_root is not None
                and self.spatial_root.root_id == endpoint.root.root_id
            )
        supports = (
            (self.primary, *self.aliases)
            if isinstance(self.primary, BRepRootSupport)
            else self.aliases
        )
        return any(support.root.root_id == endpoint.root.root_id for support in supports)

    def same_uv_endpoint(
        self,
        patch: AbstractSurfacePatch,
        first_curve: AbstractCurve | IntersectionPCurve,
        first: RootEndpoint,
        second_curve: AbstractCurve | IntersectionPCurve,
        second: RootEndpoint,
        /,
    ) -> bool:
        """Common spatial root plus certified local chart injectivity proves UV equality."""
        if isinstance(first, NativePeriodEndpoint) or isinstance(
            second, NativePeriodEndpoint
        ):
            return False
        if isinstance(self.primary, BRepPlacedVertex):
            if not isinstance(patch, PlacedSurface) or not self.primary.matches_pose(
                patch
            ):
                return False
            source = self.primary.source_root
            return source is not None and source.same_uv_endpoint(
                patch.definition, first_curve, first, second_curve, second
            )

        boxes = []
        for curve, endpoint in ((first_curve, first), (second_curve, second)):
            if not self.supports_endpoint(endpoint):
                return False
            if isinstance(endpoint, BranchRootEndpoint):
                branch = endpoint.curve
                columns = (
                    0
                    if _same_surface(branch.first.patch, patch)
                    else (2 if _same_surface(branch.second.patch, patch) else -1)
                )
                if columns < 0:
                    return False
                source_image = endpoint._source_image(maximum_steps=16)
                boxes.append(source_image[:, columns : columns + 2])
                continue
            if isinstance(endpoint.root, TrimIntersectionRoot):
                if self.supports_uv(patch, endpoint.root):
                    boxes.append(endpoint.root.point_enclosure())
                elif (
                    isinstance(curve, IntersectionPCurve)
                    and not curve.reversed
                    and isinstance(endpoint.carrier, IntersectionPCurve)
                    and curve.curve.branch_id == endpoint.carrier.curve.branch_id
                    and _same_surface(
                        curve.curve.first.patch
                        if curve.side == "first"
                        else curve.curve.second.patch,
                        patch,
                    )
                ):
                    # The scalar atom belongs to both coupled p-curves, even
                    # when the UV junction was authored on the opposite face.
                    lower, upper = endpoint.parameter_enclosure()
                    boxes.append(np.stack(_trim_source_jet(curve, lower, upper, 0)))
                else:
                    return False
            else:
                lower, upper = endpoint.parameter_enclosure()
                box = (
                    curve.enclosure(lower, upper, endpoint_roots=(endpoint, endpoint))
                    if isinstance(curve, IntersectionPCurve)
                    else curve.enclosure(lower, upper)
                    if isinstance(curve, AbstractTrimCurve)
                    else curve.bounding_box(lower, upper)
                )
                witness = certify_curve_surface(
                    endpoint.root.curve,
                    curve,
                    patch,
                    box,
                    lower,
                    upper,
                    point=np.zeros((3,), dtype=np.float64),
                    tolerance=0.0,
                    endpoint_roots=(endpoint, endpoint),
                )
                if not witness.complete or witness.deviation_bound != 0.0:
                    return False
                boxes.append(box)
        bounds = np.stack(
            (np.minimum(boxes[0][0], boxes[1][0]), np.maximum(boxes[0][1], boxes[1][1]))
        )
        # Local rank alone does not distinguish two sheets of a periodic
        # surface. A common-root UV proof must remain in one injective native
        # angular chart, not span a full deck transformation.
        for axis, period in enumerate(patch.periods):
            if period is not None and bounds[1, axis] - bounds[0, axis] >= np.nextafter(
                period, -np.inf
            ):
                return False
        return surface_chart_regular(patch, bounds)


class BRepPlacedVertex(StrictModule):
    """Exact authored pose of a literal source point or its complete root graph."""

    __strict_contract__ = True
    source_point: Float64[Literal[3]]
    rotation: Float64[Literal[3], Literal[3]]
    translation: Float64[Literal[3]]
    source_root: BRepVertexRoot | None
    source_entity_id: str = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_point: ConvertibleToArray,
        rotation: ConvertibleToArray,
        translation: ConvertibleToArray,
        source_entity_id: str,
        /,
        *,
        source_root: BRepVertexRoot | None = None,
    ) -> None:
        point = parse(
            jnp.asarray(source_point, dtype=jnp.float64),
            Float64[Literal[3]],
            "source_point",
        )
        if not np.all(np.isfinite(np.asarray(point))):
            raise ValueError("A placed source vertex requires a finite literal point.")
        if not isinstance(source_entity_id, str) or not source_entity_id:
            raise ValueError(
                "A placed vertex requires its explicit qualified source entity identity."
            )
        if source_root is not None and not isinstance(source_root, BRepVertexRoot):
            raise TypeError(
                "source_root must preserve the complete canonical vertex root graph."
            )
        rotation_, translation_ = _validate_pose(rotation, translation)
        self.source_point, self.rotation, self.translation = (
            point,
            rotation_,
            translation_,
        )
        self.source_root, self.source_entity_id = source_root, source_entity_id
        self.root_id = canonical_fingerprint(
            {
                "kind": "exact-placed-source-vertex",
                "source_entity": source_entity_id,
                "source_root": None if source_root is None else source_root.root_id,
                "source_expression": array_tree_fingerprint(
                    (point, rotation_, translation_)
                ),
            }
        )

    def matches_pose(self, patch: PlacedSurface, /) -> bool:
        return bool(
            np.array_equal(
                np.asarray(self.rotation).view(np.uint64),
                np.asarray(patch.rotation).view(np.uint64),
            )
            and np.array_equal(
                np.asarray(self.translation).view(np.uint64),
                np.asarray(patch.translation).view(np.uint64),
            )
        )

    def point_enclosure(self) -> np.ndarray:
        source = np.asarray(self.source_point)
        box = (
            np.stack((source, source))
            if self.source_root is None
            else self.source_root.point_enclosure()
        )
        bounds = source_transform_bounds(np.asarray(self.rotation), box[0], box[1])
        shift = np.asarray(self.translation)
        return np.stack(interval_add(bounds, (shift, shift)))

    def evaluate(self) -> tuple[np.ndarray, float, bool]:
        if self.source_root is None:
            point, certified = np.asarray(self.source_point), True
        else:
            point, _, certified = self.source_root.evaluate()
        value = np.asarray(self.rotation) @ point + np.asarray(self.translation)
        error = float(
            np.nextafter(
                np.linalg.norm(np.max(np.abs(self.point_enclosure() - value), axis=0)),
                np.inf,
            )
        )
        return value, error, certified


register_artifact_value(
    "phydrax.geometry.brep:BRepPlacedVertex",
    BRepPlacedVertex,
)


__all__ = ["BRepRootSupport", "BRepVertexRoot"]
