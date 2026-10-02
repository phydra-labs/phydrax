#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Translation identifications of opposite Cartesian faces.

A `PeriodicIdentification` glues the lower face ``x[c] = a`` of one interval or box
coordinate to its upper face ``x[c] = b`` by the translation ``x -> x + (b - a) e_c``.
The domain itself remains the closed fundamental box; its raw ``Boundary()`` keeps
its meaning. The identification owns the seam roles: the lower face is the source
and the upper face is the target of every field relation across the seam.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked, parse, PRNGKey
from .._base import AbstractGeometry
from .._components import DomainComponent
from .._domain import Domain
from .._function import DomainFunction
from .._scalar import ScalarInterval
from .._selection import (
    Boundary,
    CoordinateFace,
    CoordinateFaceSide,
    FixedEnd,
    FixedStart,
    Selection,
)


if TYPE_CHECKING:
    from ...discretization import AxisDomain
    from ._cover import PairedSupport


_ValueT = TypeVar("_ValueT")


class _IdentityCoordinate(StrictModule, NonTrainableState):
    def __init__(self) -> None:
        pass

    def __call__(
        self, value: _ValueT, /, *, key: PRNGKey | None = None, **kwargs: object
    ) -> _ValueT:
        del key, kwargs
        return value


class _FaceProjection(StrictModule, NonTrainableState):
    """Replace one vector coordinate component by a fixed face value."""

    component: int = eqx.field(static=True)
    value: float = eqx.field(static=True)

    def __init__(self, component: int, value: float, /) -> None:
        self.component = component
        self.value = value

    def __call__(
        self,
        coordinate: ArrayLike,
        /,
        *,
        key: PRNGKey | None = None,
        **kwargs: object,
    ) -> Array:
        del key, kwargs
        value = jnp.asarray(coordinate, dtype=jnp.float64)
        return value.at[..., self.component].set(
            jnp.asarray(self.value, dtype=value.dtype)
        )


class PeriodicIdentification(StrictModule, NonTrainableState):
    """Translation identification of the two faces of one Cartesian coordinate.

    ``label`` names a `ScalarInterval` factor (``component=None``) or an
    `Interval1d`/`HyperRectangle` factor with coordinate ``component`` (an
    `Interval1d` defaults to component ``0``). The identified coordinate keeps its
    `ValuePort` identity; equal bounds on different labels or components are
    distinct identifications.

    **Arguments:**

    - `domain`: Fundamental domain carrying the identified coordinate.
    - `label`: Coordinate label of the identified factor.
    - `component`: Vector coordinate component for geometry factors.
    - `identification_id`: Optional explicit identity; by default the identity is
      content-addressed from the domain labels, coordinate, and bounds.
    """

    domain: Domain
    label: str = eqx.field(static=True)
    component: int | None = eqx.field(static=True)
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    identification_id: str = eqx.field(static=True)
    revision: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: Domain,
        label: str,
        /,
        *,
        component: int | None = None,
        identification_id: str | None = None,
    ) -> None:
        if not isinstance(label, str) or label not in domain.labels:
            raise KeyError(f"Label {label!r} is not a coordinate of {domain.labels!r}.")
        factor = domain.factor(label)
        if isinstance(factor, ScalarInterval):
            if component is not None:
                raise ValueError("Scalar interval identifications take component=None.")
            lower = float(factor.fixed("start"))
            upper = float(factor.fixed("end"))
            component_ = None
        elif isinstance(factor, AbstractGeometry):
            bounds_lower, bounds_upper = factor.coordinate_face_bounds()
            dimension = factor.spatial_dim
            if component is None:
                if dimension != 1:
                    raise ValueError(
                        f"Identification of the {dimension}-dimensional coordinate "
                        f"{label!r} requires an explicit component."
                    )
                component_ = 0
            else:
                if isinstance(component, bool) or not isinstance(component, int):
                    raise TypeError("component must be an integer or None.")
                if not 0 <= component < dimension:
                    raise ValueError(
                        f"component {component} is outside coordinate {label!r} of "
                        f"dimension {dimension}."
                    )
                component_ = component
            lower = float(bounds_lower[component_])
            upper = float(bounds_upper[component_])
        else:
            raise TypeError(
                "Periodic identifications support ScalarInterval, Interval1d, and "
                f"HyperRectangle factors, not {type(factor).__name__}."
            )
        if not (math.isfinite(lower) and math.isfinite(upper) and upper > lower):
            raise ValueError("A periodic coordinate requires finite increasing bounds.")
        if not math.isfinite(upper - lower):
            raise ValueError(
                "A periodic coordinate requires a representable finite period."
            )
        if identification_id is not None and (
            not isinstance(identification_id, str) or not identification_id.strip()
        ):
            raise ValueError("identification_id must be a non-empty string or None.")
        content = {
            "kind": "periodic-identification",
            "factors": [
                {
                    "type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "labels": value.labels,
                }
                for value in domain.joint_factors
            ],
            "domain": array_tree_fingerprint(domain),
            "label": label,
            "component": component_,
            "lower": lower,
            "upper": upper,
        }
        self.domain = domain
        self.label = label
        self.component = component_
        self.lower = lower
        self.upper = upper
        self.identification_id = (
            canonical_fingerprint(content)
            if identification_id is None
            else identification_id.strip()
        )
        self.revision = canonical_fingerprint(
            {"content": content, "identification_id": self.identification_id}
        )

    @property
    def period(self) -> float:
        """Translation length ``upper - lower``."""
        return self.upper - self.lower

    @property
    def vector_coordinate(self) -> bool:
        return self.component is not None and isinstance(
            self.domain.factor(self.label), AbstractGeometry
        )

    def face_selection(self, side: CoordinateFaceSide, /) -> Selection:
        """Return the selection of the ``"lower"`` (source) or ``"upper"`` face."""
        side_ = parse(side, CoordinateFaceSide, "side")
        if self.component is None or not self.vector_coordinate:
            return FixedStart() if side_ == "lower" else FixedEnd()
        return CoordinateFace(self.component, side_)

    def face(self, side: CoordinateFaceSide, /) -> DomainComponent:
        """Return one identified face as a component of the fundamental domain."""
        return self.domain.component({self.label: self.face_selection(side)})

    def face_value(self, side: CoordinateFaceSide, /) -> float:
        side_ = parse(side, CoordinateFaceSide, "side")
        return self.lower if side_ == "lower" else self.upper

    def face_map(self, side: CoordinateFaceSide, /) -> dict[str, DomainFunction]:
        """Coordinate map projecting every point onto one identified face."""
        value = self.face_value(side)
        maps: dict[str, DomainFunction] = {}
        for label in self.domain.labels:
            if label != self.label:
                maps[label] = self.domain.Function(label)(_IdentityCoordinate())
            elif self.vector_coordinate and self.component is not None:
                maps[label] = self.domain.Function(label)(
                    _FaceProjection(self.component, value)
                )
            else:
                maps[label] = DomainFunction(
                    domain=self.domain,
                    deps=(),
                    func=jnp.asarray(value, dtype=jnp.float64),
                )
        return maps

    def shift(self) -> Array:
        """Coordinate displacement from the lower (source) to the upper face."""
        if self.vector_coordinate and self.component is not None:
            factor = self.domain.factor(self.label)
            if not isinstance(factor, AbstractGeometry):
                raise RuntimeError("A vector identification lost its geometry factor.")
            return (
                self.period
                * jnp.eye(factor.spatial_dim, dtype=jnp.float64)[self.component]
            )
        return jnp.asarray(self.period, dtype=jnp.float64)

    def normal(self) -> DomainFunction:
        """Constant seam direction ``+e_c`` from the source face to the target face."""
        value = (
            self.shift() / self.period
            if self.vector_coordinate
            else jnp.asarray(1.0, dtype=jnp.float64)
        )
        return DomainFunction(domain=self.domain, deps=(), func=value)

    def pairing(
        self,
        /,
        *,
        pairing_id: str | None = None,
        patch_id: str = "fundamental-domain",
    ) -> PairedSupport:
        """Return the self-seam of the fundamental domain as one `PairedSupport`.

        The common support is the lower face. The right side is the source map
        onto the lower face and the left side is the target map onto the upper
        face, following the Cartesian cover traversal convention.
        """
        from ._cover import PairedSupport

        return PairedSupport(
            self.face("lower"),
            self.face_map("upper"),
            self.face_map("lower"),
            pairing_id=(
                f"periodic-seam:{self.identification_id}"
                if pairing_id is None
                else pairing_id
            ),
            left_patch_id=patch_id,
            right_patch_id=patch_id,
            normal=self.normal(),
            topology="periodic-interface",
            identification=self,
        )

    def axis_domain(self) -> AxisDomain:
        """Return the matching periodic numerical axis descriptor."""
        from ...discretization import AxisDomain

        return AxisDomain.periodic(self.lower, self.upper)


def physical_boundary(
    domain: Domain,
    identifications: Sequence[PeriodicIdentification],
    /,
) -> tuple[DomainComponent, ...]:
    """Return the unidentified boundary strata of a fundamental domain.

    Identified faces are seams, not walls. Every other face of a Cartesian factor
    is returned as one exact `CoordinateFace`/`FixedStart`/`FixedEnd` component;
    a geometry factor without identifications contributes its whole
    ``Boundary()``. A fully periodic domain has no physical boundary and yields an
    empty tuple.
    """
    # Keep this module-level native callable unwrapped: callable_payload owns its
    # code/global identity and does not admit a checked wrapper's captured state.
    if not isinstance(domain, Domain):
        raise TypeError("domain must be a Domain.")
    resolved = tuple(identifications)
    if any(not isinstance(value, PeriodicIdentification) for value in resolved):
        raise TypeError("identifications must contain PeriodicIdentification values.")
    identified: set[tuple[str, int | None]] = set()
    for value in resolved:
        if not value.domain.same_support(domain):
            raise ValueError("Every identification must live on the requested domain.")
        key = (value.label, value.component)
        if key in identified:
            raise ValueError(f"Coordinate {key!r} is identified more than once.")
        identified.add(key)
    components: list[DomainComponent] = []
    for label in domain.labels:
        factor = domain.factor(label)
        selections: list[Selection] = []
        if isinstance(factor, ScalarInterval):
            if (label, None) not in identified:
                selections.extend((FixedStart(), FixedEnd()))
        elif isinstance(factor, AbstractGeometry):
            axes = tuple(component for name, component in identified if name == label)
            if not axes:
                selections.append(Boundary())
            else:
                factor.coordinate_face_bounds()
                for axis in range(factor.spatial_dim):
                    if axis not in axes:
                        selections.extend(
                            (CoordinateFace(axis, "lower"), CoordinateFace(axis, "upper"))
                        )
        components.extend(
            domain.component({label: selection}) for selection in selections
        )
    return tuple(components)


__all__ = ["PeriodicIdentification", "physical_boundary"]
