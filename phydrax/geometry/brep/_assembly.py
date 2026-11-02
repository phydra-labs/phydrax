#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import isfinite
from numbers import Integral, Real

import numpy as np

from ...discretization._exact_algebraic_geometry import ExactScalar, RadicalElement


type _ExactInput = Fraction | int | float | np.integer | np.floating


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _scalar(value: _ExactInput, name: str, /) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, Integral, Real)):
        raise TypeError(f"{name} must be a finite real scalar.")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return Fraction.from_float(number)


def _determinant3(matrix: tuple[tuple[Fraction, ...], ...], /) -> Fraction:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


@dataclass(frozen=True, slots=True)
class BRepAffinePlacement:
    """Invertible exact affine map from one assembly definition into its parent."""

    matrix: tuple[tuple[Fraction, ...], ...]

    def __post_init__(self) -> None:
        rows = tuple(
            tuple(_scalar(value, "placement coefficient") for value in row)
            for row in self.matrix
        )
        if len(rows) != 4 or any(len(row) != 4 for row in rows):
            raise ValueError("A BRepAffinePlacement matrix must have shape (4, 4).")
        if rows[3] != (Fraction(0), Fraction(0), Fraction(0), Fraction(1)):
            raise ValueError(
                "A BRepAffinePlacement must have an affine homogeneous last row."
            )
        if _determinant3(tuple(row[:3] for row in rows[:3])) == 0:
            raise ValueError("A BRepAffinePlacement must be nonsingular.")
        object.__setattr__(self, "matrix", rows)

    @classmethod
    def identity(cls) -> BRepAffinePlacement:
        return cls(
            (
                (Fraction(1), Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
            )
        )

    @classmethod
    def from_matrix(
        cls, values: Sequence[Sequence[_ExactInput]], /
    ) -> BRepAffinePlacement:
        return cls(
            tuple(
                tuple(_scalar(value, "placement coefficient") for value in row)
                for row in values
            )
        )

    def compose(self, inner: BRepAffinePlacement, /) -> BRepAffinePlacement:
        """Compose ``self`` after ``inner`` without rounding exact coefficients."""
        return BRepAffinePlacement(
            tuple(
                tuple(
                    sum(
                        (self.matrix[row][k] * inner.matrix[k][column] for k in range(4)),
                        Fraction(0),
                    )
                    for column in range(4)
                )
                for row in range(4)
            )
        )

    def apply(
        self, point: Sequence[ExactScalar | _ExactInput], /
    ) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
        if len(point) != 3:
            raise ValueError("An affine placement point must have three coordinates.")
        coordinates = tuple(
            value
            if isinstance(value, (Fraction, RadicalElement))
            else _scalar(value, "point coordinate")
            for value in point
        )

        def component(row: int, /) -> ExactScalar:
            return sum(
                (self.matrix[row][column] * coordinates[column] for column in range(3)),
                self.matrix[row][3],
            )

        return (component(0), component(1), component(2))

    def identity_payload(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        return tuple(
            tuple((value.numerator, value.denominator) for value in row)
            for row in self.matrix
        )


@dataclass(frozen=True, slots=True)
class BRepPlacementEvidence:
    """Outward source-pose and occurrence displacement bounds in model units."""

    rotation_deviation_bound: float
    model_extent: float
    point_deviation_bound: float
    translation_deviation_bound: float = 0.0
    entrywise_deviation_bounds: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        raw_values = (
            self.rotation_deviation_bound,
            self.model_extent,
            self.point_deviation_bound,
            self.translation_deviation_bound,
        )
        if any(isinstance(value, (bool, np.bool_)) for value in raw_values):
            raise TypeError("BRepPlacementEvidence bounds must be real scalar values.")
        values = tuple(float(value) for value in raw_values)
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "BRepPlacementEvidence bounds must be finite and nonnegative."
            )
        rotation, extent, point, translation = values
        if Fraction(point) < Fraction(rotation) * Fraction(extent) + Fraction(
            translation
        ):
            raise ValueError(
                "BRepPlacementEvidence.point_deviation_bound must enclose rotation and translation error."
            )
        object.__setattr__(self, "rotation_deviation_bound", rotation)
        object.__setattr__(self, "model_extent", extent)
        object.__setattr__(self, "point_deviation_bound", point)
        object.__setattr__(self, "translation_deviation_bound", translation)
        if self.entrywise_deviation_bounds is not None:
            raw_rows = tuple(tuple(row) for row in self.entrywise_deviation_bounds)
            if any(
                isinstance(value, (bool, np.bool_)) for row in raw_rows for value in row
            ):
                raise TypeError(
                    "BRepPlacementEvidence.entrywise_deviation_bounds cannot contain boolean values."
                )
            rows = tuple(tuple(float(value) for value in row) for row in raw_rows)
            if len(rows) != 3 or any(
                len(row) != 4 or any(not isfinite(value) or value < 0.0 for value in row)
                for row in rows
            ):
                raise ValueError(
                    "BRepPlacementEvidence.entrywise_deviation_bounds must be finite nonnegative 3x4 values."
                )
            rotation_squared = sum(
                (Fraction(value) ** 2 for row in rows for value in row[:3]),
                Fraction(0),
            )
            translation_squared = sum(
                (Fraction(row[3]) ** 2 for row in rows), Fraction(0)
            )
            if (
                Fraction(rotation) ** 2 < rotation_squared
                or Fraction(translation) ** 2 < translation_squared
            ):
                raise ValueError(
                    "BRepPlacementEvidence scalar errors must enclose its entrywise bounds."
                )
            object.__setattr__(self, "entrywise_deviation_bounds", rows)

    def payload(self) -> dict[str, object]:
        return {
            "rotation_deviation_bound": self.rotation_deviation_bound,
            "model_extent": self.model_extent,
            "point_deviation_bound": self.point_deviation_bound,
            "translation_deviation_bound": self.translation_deviation_bound,
            "entrywise_deviation_bounds": self.entrywise_deviation_bounds,
        }


def _fraction_upper(value: Fraction, /) -> float:
    rounded = float(value)
    if not isfinite(rounded):
        raise ValueError("A placement error bound is not finitely representable.")
    if Fraction.from_float(rounded) < value:
        rounded = float(np.nextafter(rounded, np.inf))
    return rounded


def _fraction_norm_upper(values: tuple[Fraction, ...], /) -> Fraction:
    squared = sum((value * value for value in values), Fraction(0))
    rounded = float(np.sqrt(float(squared)))
    if not isfinite(rounded):
        raise ValueError("An exact placement norm is not finitely representable.")
    while Fraction.from_float(rounded) ** 2 < squared:
        rounded = float(np.nextafter(rounded, np.inf))
    return Fraction.from_float(rounded)


def _rotation_norm(placement: BRepAffinePlacement, /) -> Fraction:
    return _fraction_norm_upper(
        tuple(placement.matrix[row][column] for row in range(3) for column in range(3))
    )


def _translation_norm(placement: BRepAffinePlacement, /) -> Fraction:
    return _fraction_norm_upper(tuple(placement.matrix[row][3] for row in range(3)))


def _compose_placement_evidence(
    outer: BRepAffinePlacement,
    outer_evidence: BRepPlacementEvidence | None,
    inner: BRepAffinePlacement,
    inner_evidence: BRepPlacementEvidence | None,
    /,
) -> BRepPlacementEvidence | None:
    if outer_evidence is None and inner_evidence is None:
        return None
    outer_rotation_error = Fraction(
        0.0 if outer_evidence is None else outer_evidence.rotation_deviation_bound
    )
    inner_rotation_error = Fraction(
        0.0 if inner_evidence is None else inner_evidence.rotation_deviation_bound
    )
    outer_translation_error = Fraction(
        0.0 if outer_evidence is None else outer_evidence.translation_deviation_bound
    )
    inner_translation_error = Fraction(
        0.0 if inner_evidence is None else inner_evidence.translation_deviation_bound
    )
    outer_rotation_norm = _rotation_norm(outer)
    inner_rotation_norm = _rotation_norm(inner)
    rotation_error = (
        outer_rotation_norm * inner_rotation_error
        + outer_rotation_error * inner_rotation_norm
        + outer_rotation_error * inner_rotation_error
    )
    translation_error = (
        outer_rotation_norm * inner_translation_error
        + outer_rotation_error * _translation_norm(inner)
        + outer_rotation_error * inner_translation_error
        + outer_translation_error
    )
    translation_bound = _fraction_upper(translation_error)
    return BRepPlacementEvidence(
        _fraction_upper(rotation_error),
        0.0,
        translation_bound,
        translation_bound,
    )


@dataclass(frozen=True, slots=True)
class BRepAssemblyUse:
    """One named use of a solid or child definition with a local exact placement."""

    use_id: str
    placement: BRepAffinePlacement = field(default_factory=BRepAffinePlacement.identity)
    solid_index: int | None = None
    definition_id: str | None = None
    placement_evidence: BRepPlacementEvidence | None = None

    def __post_init__(self) -> None:
        use_id = _identifier(self.use_id, "use_id")
        if (self.solid_index is None) == (self.definition_id is None):
            raise ValueError(
                "An assembly use must target exactly one solid or child definition."
            )
        if self.solid_index is not None and (
            isinstance(self.solid_index, bool)
            or not isinstance(self.solid_index, int)
            or self.solid_index < 0
        ):
            raise ValueError("An assembly solid_index must be a nonnegative integer.")
        definition_id = (
            None
            if self.definition_id is None
            else _identifier(self.definition_id, "definition_id")
        )
        if not isinstance(self.placement, BRepAffinePlacement):
            raise TypeError("An assembly use placement must be a BRepAffinePlacement.")
        if self.placement_evidence is not None and not isinstance(
            self.placement_evidence, BRepPlacementEvidence
        ):
            raise TypeError("placement_evidence must be BRepPlacementEvidence or None.")
        object.__setattr__(self, "use_id", use_id)
        object.__setattr__(self, "definition_id", definition_id)

    def identity_payload(self) -> dict[str, object]:
        return {
            "use_id": self.use_id,
            "solid_index": self.solid_index,
            "definition_id": self.definition_id,
            "placement": self.placement.identity_payload(),
            **(
                {"placement_evidence": self.placement_evidence.payload()}
                if self.placement_evidence is not None
                else {}
            ),
        }


@dataclass(frozen=True, slots=True)
class BRepAssemblyDefinition:
    """Reusable source assembly definition with ordered named child uses."""

    definition_id: str
    uses: tuple[BRepAssemblyUse, ...]

    def __post_init__(self) -> None:
        definition_id = _identifier(self.definition_id, "definition_id")
        uses = tuple(self.uses)
        if any(not isinstance(use, BRepAssemblyUse) for use in uses):
            raise TypeError("Assembly definition uses must be BRepAssemblyUse values.")
        if len({use.use_id for use in uses}) != len(uses):
            raise ValueError("Assembly use IDs must be unique within a definition.")
        object.__setattr__(self, "definition_id", definition_id)
        object.__setattr__(self, "uses", uses)

    def identity_payload(self) -> dict[str, object]:
        return {
            "definition_id": self.definition_id,
            "uses": tuple(use.identity_payload() for use in self.uses),
        }


@dataclass(frozen=True, slots=True)
class BRepAssemblyOccurrence:
    """One lazily yielded leaf solid occurrence and its composed exact placement."""

    path: tuple[str, ...]
    solid_index: int
    placement: BRepAffinePlacement
    placement_evidence: BRepPlacementEvidence | None = None


@dataclass(frozen=True, slots=True)
class BRepAssemblyGraph:
    """Shared assembly definition DAG with exact per-use placements."""

    roots: tuple[BRepAssemblyUse, ...]
    definitions: tuple[BRepAssemblyDefinition, ...]
    solid_count: int
    occurrence_evidence: tuple[tuple[tuple[str, ...], BRepPlacementEvidence], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.solid_count, bool) or not isinstance(self.solid_count, int):
            raise TypeError("solid_count must be an integer.")
        if self.solid_count < 0:
            raise ValueError("solid_count must be nonnegative.")
        roots = tuple(self.roots)
        definitions = tuple(self.definitions)
        if any(not isinstance(use, BRepAssemblyUse) for use in roots):
            raise TypeError("Assembly roots must be BRepAssemblyUse values.")
        if not roots and self.solid_count != 0:
            raise ValueError("An assembly graph with solids requires root uses.")
        if any(not isinstance(value, BRepAssemblyDefinition) for value in definitions):
            raise TypeError("Assembly definitions must be BRepAssemblyDefinition values.")
        definition_map = {value.definition_id: value for value in definitions}
        if len(definition_map) != len(definitions):
            raise ValueError("Assembly definition IDs must be unique.")
        root_ids: set[str] = set()
        for use in roots:
            self._validate_use(use, definition_map, root_ids)
        for definition in definitions:
            use_ids: set[str] = set()
            for use in definition.uses:
                self._validate_use(use, definition_map, use_ids)
        self._require_acyclic(definitions, definition_map)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "definitions", definitions)
        evidence_rows = tuple(self.occurrence_evidence)
        for row in evidence_rows:
            if (
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], tuple)
                or not row[0]
                or any(not isinstance(part, str) or not part for part in row[0])
                or not isinstance(row[1], BRepPlacementEvidence)
            ):
                raise ValueError(
                    "Assembly occurrence evidence requires named paths and placement bounds."
                )
        if len({path for path, _ in evidence_rows}) != len(evidence_rows):
            raise ValueError("Assembly occurrence evidence paths must be unique.")
        object.__setattr__(self, "occurrence_evidence", tuple(sorted(evidence_rows)))

    def _validate_use(
        self,
        use: BRepAssemblyUse,
        definition_map: dict[str, BRepAssemblyDefinition],
        use_ids: set[str],
        /,
    ) -> None:
        if use.use_id in use_ids:
            raise ValueError("Assembly use IDs must be unique among siblings.")
        use_ids.add(use.use_id)
        if use.solid_index is not None and use.solid_index >= self.solid_count:
            raise ValueError("An assembly use references an absent solid.")
        if use.definition_id is not None and use.definition_id not in definition_map:
            raise ValueError("An assembly use references an absent definition.")

    @staticmethod
    def _require_acyclic(
        definitions: tuple[BRepAssemblyDefinition, ...],
        definition_map: dict[str, BRepAssemblyDefinition],
        /,
    ) -> None:
        state: dict[str, int] = {}
        for definition in definitions:
            if state.get(definition.definition_id, 0) != 0:
                continue
            stack: list[tuple[str, bool]] = [(definition.definition_id, False)]
            while stack:
                definition_id, exiting = stack.pop()
                if exiting:
                    state[definition_id] = 2
                    continue
                status = state.get(definition_id, 0)
                if status == 2:
                    continue
                if status == 1:
                    raise ValueError("Assembly definition relationships must be acyclic.")
                state[definition_id] = 1
                stack.append((definition_id, True))
                current = definition_map[definition_id]
                for use in reversed(current.uses):
                    child = use.definition_id
                    if child is None:
                        continue
                    child_state = state.get(child, 0)
                    if child_state == 1:
                        raise ValueError(
                            "Assembly definition relationships must be acyclic."
                        )
                    if child_state == 0:
                        stack.append((child, False))

    @classmethod
    def single_solids(cls, solid_count: int, /) -> BRepAssemblyGraph:
        if isinstance(solid_count, bool) or not isinstance(solid_count, int):
            raise TypeError("solid_count must be an integer.")
        if solid_count < 0:
            raise ValueError("solid_count must be nonnegative.")
        roots = tuple(
            BRepAssemblyUse(f"solid-{index}", solid_index=index)
            for index in range(solid_count)
        )
        return cls(roots, (), solid_count)

    def with_occurrence_placement(
        self,
        path: tuple[str, ...],
        placement: BRepAffinePlacement,
        /,
        *,
        evidence: BRepPlacementEvidence | None = None,
    ) -> BRepAssemblyGraph:
        """Copy the selected occurrence ancestry and replace its local placement."""
        if (
            not isinstance(path, tuple)
            or not path
            or any(not isinstance(part, str) or not part for part in path)
        ):
            raise ValueError("An occurrence path must contain named uses.")
        if not isinstance(placement, BRepAffinePlacement):
            raise TypeError("placement must be a BRepAffinePlacement.")
        if evidence is not None and not isinstance(evidence, BRepPlacementEvidence):
            raise TypeError("evidence must be BRepPlacementEvidence or None.")
        definitions = {item.definition_id: item for item in self.definitions}
        uses = self.roots
        ancestry: list[tuple[BRepAssemblyUse, BRepAssemblyDefinition]] = []
        selected: BRepAssemblyUse | None = None
        for index, part in enumerate(path):
            selected = next((use for use in uses if use.use_id == part), None)
            if selected is None:
                raise ValueError("The assembly occurrence path does not exist.")
            if index + 1 == len(path):
                break
            if selected.definition_id is None:
                raise ValueError("The assembly occurrence path extends beyond a solid.")
            definition = definitions[selected.definition_id]
            ancestry.append((selected, definition))
            uses = definition.uses
        if selected is None:
            raise ValueError("The assembly occurrence path does not exist.")
        replacement = replace(selected, placement=placement, placement_evidence=evidence)
        added: list[BRepAssemblyDefinition] = []
        reserved = set(definitions)
        for parent, definition in reversed(ancestry):
            base = f"{definition.definition_id}-placement"
            name = base
            suffix = 1
            while name in reserved:
                name = f"{base}-{suffix}"
                suffix += 1
            reserved.add(name)
            children = tuple(
                replacement if use.use_id == replacement.use_id else use
                for use in definition.uses
            )
            added.append(BRepAssemblyDefinition(name, children))
            replacement = replace(parent, definition_id=name)
        roots = tuple(
            replacement if use.use_id == replacement.use_id else use for use in self.roots
        )
        all_definitions = (*self.definitions, *added)
        by_id = {definition.definition_id: definition for definition in all_definitions}

        def reachable(root_uses: tuple[BRepAssemblyUse, ...], /) -> set[str]:
            pending = [
                use.definition_id for use in root_uses if use.definition_id is not None
            ]
            seen: set[str] = set()
            while pending:
                identifier = pending.pop()
                if identifier in seen:
                    continue
                seen.add(identifier)
                pending.extend(
                    use.definition_id
                    for use in by_id[identifier].uses
                    if use.definition_id is not None
                )
            return seen

        removed = reachable(self.roots) - reachable(roots)
        retained_definitions = tuple(
            definition
            for definition in all_definitions
            if definition.definition_id not in removed
        )
        retained_evidence = tuple(
            (occurrence, value)
            for occurrence, value in self.occurrence_evidence
            if occurrence[: len(path)] != path
        )
        return BRepAssemblyGraph(
            roots, retained_definitions, self.solid_count, retained_evidence
        )

    def iter_occurrences(self, /) -> Iterator[BRepAssemblyOccurrence]:
        """Yield placed solid leaves and occurrence-bound placement certificates."""
        definition_map = {value.definition_id: value for value in self.definitions}
        certified = dict(self.occurrence_evidence)
        identity = BRepAffinePlacement.identity()
        stack: list[
            tuple[
                BRepAssemblyUse,
                tuple[str, ...],
                BRepAffinePlacement,
                BRepPlacementEvidence | None,
            ]
        ] = [(use, (), identity, None) for use in reversed(self.roots)]
        while stack:
            use, parent_path, parent_placement, parent_evidence = stack.pop()
            path = (*parent_path, use.use_id)
            placement = parent_placement.compose(use.placement)
            evidence = _compose_placement_evidence(
                parent_placement,
                parent_evidence,
                use.placement,
                use.placement_evidence,
            )
            if use.solid_index is not None:
                yield BRepAssemblyOccurrence(
                    path,
                    use.solid_index,
                    placement,
                    certified.get(path, evidence),
                )
                continue
            definition_id = use.definition_id
            if definition_id is None:
                raise RuntimeError("Validated assembly use lost its target definition.")
            definition = definition_map[definition_id]
            stack.extend(
                (child, path, placement, evidence) for child in reversed(definition.uses)
            )

    def identity_payload(self) -> dict[str, object]:
        return {
            "solid_count": self.solid_count,
            "roots": tuple(use.identity_payload() for use in self.roots),
            "definitions": tuple(
                value.identity_payload()
                for value in sorted(self.definitions, key=lambda item: item.definition_id)
            ),
            **(
                {
                    "occurrence_evidence": tuple(
                        (path, evidence.payload())
                        for path, evidence in self.occurrence_evidence
                    )
                }
                if self.occurrence_evidence
                else {}
            ),
        }


__all__ = [
    "BRepAffinePlacement",
    "BRepPlacementEvidence",
    "BRepAssemblyDefinition",
    "BRepAssemblyGraph",
    "BRepAssemblyOccurrence",
    "BRepAssemblyUse",
]
