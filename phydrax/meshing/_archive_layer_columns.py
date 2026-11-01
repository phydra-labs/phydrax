#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Column storage for exact layer records; scientific restoration stays with owners."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np

from .._archive_columns import (
    ExactTensorArchiveRecipe,
    integer_column_values,
    IntegerRowsArchiveRecipe,
    StaticColumnArchiveRecipe,
    TextColumnArchiveRecipe,
    TextRowsArchiveRecipe,
)
from .._array_archive import array_payload_digest
from .._fingerprint import canonical_fingerprint


# Only these record fields are projected. All other fields, including scientific
# identities, remain in the ordinary registered model recipe.
_COLUMNS = {
    "LayerColumnProjectionIdentity": {
        "front_row_identities": "text",
        "source_column_ids": "integer_rows",
        "source_column_identities": "text_rows",
        "source_column_root_vertex_ids": "integer_rows",
    },
    "LayerCurvingProjectionTrace": {
        "target_reference_points": "exact",
        "source_piece_ids": "integer_rows",
        **dict.fromkeys(
            (
                "requested_entity_ids",
                "returned_entity_ids",
                "returned_source_ids",
                "returned_source_revisions",
                "returned_frame_ids",
                "failures",
            ),
            "text",
        ),
        **dict.fromkeys(
            ("requested_occurrence_paths", "returned_occurrence_paths"), "text_rows"
        ),
    },
    "LayerCurvingImplicitRootTrace": {
        "target_reference_points": "exact",
        "source_cell_keys": "integer_tensor",
        "root_edge_keys": "integer_tensor",
        "statuses": "text",
        "chart_evidence": "implicit",
    },
    "LayerCurvingMappedChartTrace": {
        **dict.fromkeys(
            ("target_reference_points", "root_reference_points", "source_parameters"),
            "exact",
        ),
        **dict.fromkeys(("source_chart_ids", "source_entity_ids"), "text"),
        "occurrence_paths": "text_rows",
    },
    "LayerCurvingSourceEvidence": {"implicit_chart_evidence": "implicit"},
    "SourceBoundaryChartWitness": {
        "target_reference_rational_tokens": "integer_tensor",
        "source_parameter_rational_tokens": "integer_tensor",
        **dict.fromkeys(("source_chart_ids", "source_entity_ids"), "text"),
        "occurrence_paths": "text_rows",
    },
    "SourceFidelityCertificate": {
        "source_cell_global_ids": "integer_tensor",
        "source_patch_ids": "integer_tensor",
        "source_occurrence_paths": "text_rows",
    },
    "NestedReferenceWitnesses": {"fine_reference_exact_vertices": "exact"},
    "ReferenceCellMapWitnesses": {
        "source_reference_exact_vertices": "exact",
        "source_reference_chart_arguments": "expressions",
    },
    "_ExactCoordinateBankMap": {"exact_coordinates": "exact"},
    "CellGeometrySpec": {"exact_coordinate_bank": "exact"},
    "PerCellRestrictedCellGeometryElement": {
        "target_cell_ids": "integer_tensor",
        "source_cell_ids": "integer_tensor",
        "chart_coefficients": "rational_terms",
    },
}


def _record_types() -> dict[str, type]:
    from ..discretization._cell_geometry import (
        CellGeometrySpec,
        PerCellRestrictedCellGeometryElement,
    )
    from ..discretization._nested_reference_types import (
        NestedReferenceWitnesses,
        ReferenceCellMapWitnesses,
    )
    from ..geometry._source_boundary_records import SourceBoundaryChartWitness
    from ..geometry._source_fidelity_certificate import SourceFidelityCertificate
    from ._curving import _ExactCoordinateBankMap
    from ._layer_curving_records import (
        LayerColumnProjectionIdentity,
        LayerCurvingImplicitRootTrace,
        LayerCurvingMappedChartTrace,
        LayerCurvingProjectionTrace,
        LayerCurvingSourceEvidence,
    )

    return {
        owner.__name__: owner
        for owner in (
            CellGeometrySpec,
            PerCellRestrictedCellGeometryElement,
            NestedReferenceWitnesses,
            ReferenceCellMapWitnesses,
            SourceFidelityCertificate,
            SourceBoundaryChartWitness,
            _ExactCoordinateBankMap,
            LayerColumnProjectionIdentity,
            LayerCurvingImplicitRootTrace,
            LayerCurvingMappedChartTrace,
            LayerCurvingProjectionTrace,
            LayerCurvingSourceEvidence,
        )
    }


def _encode_column(value: Any, kind: str) -> Any:
    if value is None:
        return None
    if type(value) in (
        TextColumnArchiveRecipe,
        TextRowsArchiveRecipe,
        IntegerRowsArchiveRecipe,
        ExactTensorArchiveRecipe,
        StaticColumnArchiveRecipe,
        ImplicitChartRowsArchiveRecipe,
        ExactExpressionRowsArchiveRecipe,
    ):
        return value
    if kind == "text":
        return TextColumnArchiveRecipe.encode(value)
    if kind == "text_rows":
        return TextRowsArchiveRecipe.encode(value)
    if kind == "integer_rows":
        return IntegerRowsArchiveRecipe.encode(value)
    if kind == "integer_tensor":
        return ExactTensorArchiveRecipe.encode(value, rational=False)
    if kind == "exact":
        # Rational coordinates use exact decimal text. Algebraic coordinates
        # retain their registered exact scalar representation, deduplicated by row.
        try:
            return ExactTensorArchiveRecipe.encode(value, rational=True)
        except (TypeError, ValueError):
            return StaticColumnArchiveRecipe.encode(value)
    if kind == "implicit":
        return ImplicitChartRowsArchiveRecipe.encode(value)
    if kind in ("expressions", "rational_terms"):
        return ExactExpressionRowsArchiveRecipe.encode(
            tuple(
                tuple(
                    ("rational", *item)
                    if kind == "rational_terms"
                    else _expression_tokens(item)
                    for item in row
                )
                for row in value
            )
        )
    raise ValueError("Unknown layer column kind.")


def _decode_column(value: Any, kind: str) -> Any:
    if value is None:
        return None
    allowed = {
        "text": (TextColumnArchiveRecipe,),
        "text_rows": (TextRowsArchiveRecipe,),
        "integer_rows": (IntegerRowsArchiveRecipe,),
        "integer_tensor": (ExactTensorArchiveRecipe,),
        "exact": (ExactTensorArchiveRecipe, StaticColumnArchiveRecipe),
        "implicit": (ImplicitChartRowsArchiveRecipe,),
        "expressions": (ExactExpressionRowsArchiveRecipe,),
        "rational_terms": (ExactExpressionRowsArchiveRecipe,),
    }[kind]
    if type(value) not in allowed:
        raise TypeError("Layer record column has an unexpected recipe type.")
    if isinstance(value, ExactTensorArchiveRecipe) and value.rational != (
        kind == "exact"
    ):
        raise ValueError("Layer record column has an unexpected exact scalar type.")
    decoded = value.decode()
    if kind == "expressions":
        return tuple(
            tuple(_expression_from_tokens(item) for item in row) for row in decoded
        )
    if kind == "rational_terms":
        if any(item[0] != "rational" for row in decoded for item in row):
            raise ValueError("Restricted cell columns must retain rational term pairs.")
        return tuple(tuple((item[1], item[2]) for item in row) for row in decoded)
    return decoded


def _expression_tokens(value: Any) -> tuple[Any, ...]:
    from ..discretization._coordinate_enclosure import RationalPolynomial

    if type(value) is RationalPolynomial:
        return (
            "rational",
            tuple(sorted(value.numerator.items())),
            tuple(sorted(value.denominator.items())),
        )
    if type(value) not in (dict, MappingProxyType):
        raise TypeError("Exact chart columns require canonical polynomial mappings.")
    return (
        "mapping" if type(value) is dict else "mappingproxy",
        tuple(sorted(value.items())),
    )


def _expression_from_tokens(value: tuple[Any, ...]) -> Any:
    from ..discretization._coordinate_enclosure import RationalPolynomial

    if not isinstance(value, tuple) or not value:
        raise TypeError("Exact chart expression column must hold tagged tuples.")
    if value[0] == "rational" and len(value) == 3:
        return RationalPolynomial(dict(value[1]), dict(value[2]))
    if value[0] in ("mapping", "mappingproxy") and len(value) == 2:
        terms = dict(value[1])
        return terms if value[0] == "mapping" else MappingProxyType(terms)
    raise ValueError("Exact chart expression column has an invalid field set.")


class ExactExpressionRowsArchiveRecipe(NamedTuple):
    """Deduplicated expression rows with array-backed exact polynomial terms."""

    kinds: np.ndarray
    polynomial_ends: np.ndarray
    exponents: IntegerRowsArchiveRecipe
    coefficients: ExactTensorArchiveRecipe | StaticColumnArchiveRecipe
    rows: IntegerRowsArchiveRecipe

    @classmethod
    def encode(
        cls, rows: tuple[tuple[Any, ...], ...]
    ) -> ExactExpressionRowsArchiveRecipe:
        column = StaticColumnArchiveRecipe.encode(
            tuple(item for row in rows for item in row)
        )
        kinds = {"mapping": 0, "mappingproxy": 1, "rational": 2}
        polynomials = tuple(
            polynomial
            for item in column.table
            for polynomial in (item[1], item[2] if item[0] == "rational" else ())
        )
        terms = tuple(term for polynomial in polynomials for term in polynomial)
        return cls(
            np.asarray([kinds[item[0]] for item in column.table], dtype=np.int8),
            np.cumsum(np.asarray([len(value) for value in polynomials], np.int64)),
            IntegerRowsArchiveRecipe.encode(tuple(term[0] for term in terms)),
            _encode_column(tuple(term[1] for term in terms), "exact"),
            IntegerRowsArchiveRecipe(
                column.rows,
                np.cumsum(np.asarray([len(row) for row in rows], np.int64)),
            ),
        )

    def decode(self) -> tuple[tuple[Any, ...], ...]:
        kinds = np.asarray(self.kinds)
        if kinds.ndim != 1 or kinds.dtype != np.int8 or np.any((kinds < 0) | (kinds > 2)):
            raise TypeError("Exact expression kinds must be a canonical int8 vector.")
        if (
            type(self.exponents) is not IntegerRowsArchiveRecipe
            or type(self.rows) is not IntegerRowsArchiveRecipe
        ):
            raise TypeError(
                "Exact expression terms and rows require integer row recipes."
            )
        ends = integer_column_values(self.polynomial_ends)
        exponents = self.exponents.decode()
        coefficients = _decode_column(self.coefficients, "exact")
        if (
            len(ends) != 2 * len(kinds)
            or len(exponents) != len(coefficients)
            or (not ends and exponents)
            or (
                ends
                and (
                    ends[0] < 0
                    or any(last < first for first, last in zip(ends, ends[1:]))
                    or ends[-1] != len(exponents)
                )
            )
        ):
            raise ValueError("Exact expression columns must partition aligned terms.")
        terms = tuple(zip(exponents, coefficients, strict=True))
        polynomials = tuple(
            terms[start:end] for start, end in zip((0, *ends)[:-1], ends, strict=True)
        )
        tags = ("mapping", "mappingproxy", "rational")
        table = []
        for index, kind in enumerate(kinds.tolist()):
            numerator, denominator = polynomials[2 * index : 2 * index + 2]
            if kind != 2 and denominator:
                raise ValueError(
                    "Polynomial expression columns cannot carry denominators."
                )
            for polynomial in (numerator, denominator):
                if tuple(sorted(dict(polynomial).items())) != polynomial:
                    raise ValueError(
                        "Expression terms must have unique sorted exponents."
                    )
            table.append(
                (tags[kind], numerator, denominator)
                if kind == 2
                else (tags[kind], numerator)
            )
        rows = self.rows.decode()
        # Reuse the canonical first-occurrence/index admission of static columns.
        StaticColumnArchiveRecipe(tuple(table), self.rows.values).decode()
        return tuple(tuple(table[index] for index in row) for row in rows)


def _implicit_columns_digest(
    integers: ExactTensorArchiveRecipe,
    bounds: np.ndarray,
    statuses: TextColumnArchiveRecipe,
    details: TextColumnArchiveRecipe,
) -> str:
    return canonical_fingerprint(
        (
            integers.shape,
            integers.rational,
            tuple(
                array_payload_digest(value)
                for value in (
                    integers.text.data,
                    integers.text.ends,
                    bounds,
                    statuses.data,
                    statuses.ends,
                    details.data,
                    details.ends,
                )
            ),
        )
    )


class ImplicitChartRowsArchiveRecipe(NamedTuple):
    """Primitive root evidence transposed without changing its owner field set."""

    integers: ExactTensorArchiveRecipe
    bounds: np.ndarray
    statuses: TextColumnArchiveRecipe
    details: TextColumnArchiveRecipe
    content_digest: str

    @classmethod
    def encode(cls, rows: tuple[Any, ...]) -> ImplicitChartRowsArchiveRecipe:
        from ..geometry.implicit._source_charts import ImplicitRootChartEvidence

        rows = tuple(
            ImplicitRootChartEvidence(**row) if type(row) is dict else row for row in rows
        )
        if any(type(row) is not ImplicitRootChartEvidence for row in rows):
            raise TypeError("Implicit chart columns require primitive owner records.")
        integers = tuple(
            (
                row.row,
                row.target_cell_global_id,
                row.target_entity_global_id,
                row.target_local_entity_index,
                *row.source_cell_key,
                *(
                    item
                    for edge in row.root_edge_keys
                    for point in edge
                    for item in point
                ),
            )
            for row in rows
        )
        bounds = np.asarray(
            tuple(
                (
                    row.field_residual_bound,
                    row.target_to_root_lower_bound,
                    row.target_to_root_bound,
                    *row.source_normal_deviation_bounds,
                    row.chart_jacobian_lower,
                    row.chart_injectivity_margin,
                    row.source_gradient_lower,
                )
                for row in rows
            ),
            dtype=np.float64,
        ).reshape((-1, 8))
        integer_rows = ExactTensorArchiveRecipe.encode(integers, rational=False)
        statuses = TextColumnArchiveRecipe.encode(tuple(row.status for row in rows))
        details = TextColumnArchiveRecipe.encode(tuple(row.detail for row in rows))
        return cls(
            integer_rows,
            bounds,
            statuses,
            details,
            _implicit_columns_digest(integer_rows, bounds, statuses, details),
        )

    def decode(self) -> tuple[dict[str, Any], ...]:
        if type(self.integers) is not ExactTensorArchiveRecipe or self.integers.rational:
            raise TypeError("Implicit chart integer column requires exact integers.")
        if (
            type(self.statuses) is not TextColumnArchiveRecipe
            or type(self.details) is not TextColumnArchiveRecipe
            or type(self.integers.text) is not TextColumnArchiveRecipe
        ):
            raise TypeError("Implicit chart text columns require exact text recipes.")
        # Scientific trace IDs intentionally omit some primitive diagnostics.
        # Bind every stored column before replay without changing those IDs or
        # treating this archive integrity check as mathematical certification.
        if self.content_digest != _implicit_columns_digest(
            self.integers, self.bounds, self.statuses, self.details
        ):
            raise ValueError("Implicit chart column content digest does not reproduce.")
        integers = self.integers.decode()
        bounds = np.asarray(self.bounds)
        statuses, details = self.statuses.decode(), self.details.decode()
        if (
            bounds.dtype != np.float64
            or bounds.shape != (len(integers), 8)
            or len(statuses) != len(integers)
            or len(details) != len(integers)
            or any(len(row) != 26 for row in integers)
        ):
            raise ValueError("Implicit chart columns must have aligned primitive rows.")
        return tuple(
            dict(
                row=ids[0],
                target_cell_global_id=ids[1],
                target_entity_global_id=ids[2],
                target_local_entity_index=ids[3],
                source_cell_key=ids[4:8],
                root_edge_keys=tuple(
                    tuple(
                        tuple(ids[8 + edge * 6 + point * 3 : 11 + edge * 6 + point * 3])
                        for point in range(2)
                    )
                    for edge in range(3)
                ),
                status=status,
                field_residual_bound=float(values[0]),
                target_to_root_lower_bound=float(values[1]),
                target_to_root_bound=float(values[2]),
                source_normal_deviation_bounds=(float(values[3]), float(values[4])),
                chart_jacobian_lower=float(values[5]),
                chart_injectivity_margin=float(values[6]),
                source_gradient_lower=float(values[7]),
                detail=detail,
            )
            for ids, values, status, detail in zip(
                integers, bounds, statuses, details, strict=True
            )
        )


def _restore_local_record(
    owner: type, names: tuple[str, ...], fields: dict[str, Any]
) -> Any | None:
    """Check local constructor identities before any source or proof replay."""
    name = owner.__name__
    if name == "LayerColumnProjectionIdentity":
        identity = "identity_id"
        rebuilt = owner(*(fields[key] for key in names if key != identity))
    elif name == "SourceBoundaryChartWitness":
        identity = "witness_id"
        rebuilt = owner(
            **{key: value for key, value in fields.items() if key != identity}
        )
    elif name == "LayerCurvingMappedChartTrace":
        identity = "trace_id"
        rebuilt = owner(*(fields[key] for key in names if key != identity))
    elif name == "LayerCurvingProjectionTrace":
        identity = "trace_id"
        keywords = {
            key: fields[key]
            for key in (
                "normals",
                "tangents",
                "chart_witness_rows",
                "target_reference_points",
            )
        }
        rebuilt = owner(
            *(fields[key] for key in names if key != identity and key not in keywords),
            **keywords,
        )
    elif name == "LayerCurvingImplicitRootTrace":
        from ..geometry.implicit._source_charts import ImplicitRootChartEvidence

        identity = "trace_id"
        constructor_fields = {
            **fields,
            "chart_evidence": tuple(
                ImplicitRootChartEvidence(**row) for row in fields["chart_evidence"]
            ),
        }
        rebuilt = owner(*(constructor_fields[key] for key in names if key != identity))
    else:
        return None
    if getattr(rebuilt, identity) != fields[identity]:
        raise ValueError("Layer archive record constructor identity does not reproduce.")
    # The implicit owner callback deliberately consumes primitive mappings,
    # not a live chart-evidence object as mathematical authority.
    return None if name == "LayerCurvingImplicitRootTrace" else rebuilt


class LayerRecordArchiveRecipe(NamedTuple):
    """Exact stored fields of a whitelisted layer record with row columns."""

    record_type: str
    fields: dict[str, Any]

    def restore(self) -> Any:
        from .._model._structure import _restore_dataclass

        owner = _record_types().get(self.record_type)
        if owner is None:
            raise TypeError("Unknown layer archive record type.")
        names = (
            tuple(field.name for field in dataclasses.fields(owner))
            if dataclasses.is_dataclass(owner)
            else getattr(owner, "_fields")
        )
        if type(self.fields) is not dict or set(self.fields) != set(names):
            raise ValueError("Layer archive record must retain its exact field set.")
        columns = _COLUMNS[self.record_type]
        fields = {
            name: _decode_column(value, columns[name]) if name in columns else value
            for name, value in self.fields.items()
        }
        rebuilt = _restore_local_record(owner, names, fields)
        if rebuilt is not None:
            return rebuilt
        # This restores stored fields, not scientific authority. The existing
        # owner restoration callback still reconstructs and checks identities,
        # source queries, coordinate maps and all mathematical evidence.
        return (
            _restore_dataclass(owner, fields)
            if dataclasses.is_dataclass(owner)
            else owner(**fields)
        )


def is_layer_archive_record(value: Any) -> bool:
    name = type(value).__name__
    return name in _COLUMNS and type(value) is _record_types()[name]


def layer_record_archive_recipe(value: Any) -> LayerRecordArchiveRecipe | None:
    name = type(value).__name__
    if not is_layer_archive_record(value):
        return None
    names = (
        tuple(field.name for field in dataclasses.fields(value))
        if dataclasses.is_dataclass(value)
        else value._fields
    )
    columns = _COLUMNS[name]
    return LayerRecordArchiveRecipe(
        name,
        {
            field: _encode_column(object.__getattribute__(value, field), columns[field])
            if field in columns
            else object.__getattribute__(value, field)
            for field in names
        },
    )
