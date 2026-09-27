from typing import Any, Literal

import pytest

import phydrax.typing as pt


class ComponentDim(pt.Dim, minimum=1):
    pass


class BatchDims(pt.VariadicDim):
    pass


def test_dimensions_are_nominal_tokens_with_validated_minimums() -> None:
    assert ComponentDim.minimum == 1
    assert BatchDims.minimum == 0
    with pytest.raises(TypeError):
        ComponentDim()
    with pytest.raises(TypeError):
        # ty: ignore[invalid-argument-type]
        class FractionalDim(pt.Dim, minimum=1.5):
            pass

    with pytest.raises(ValueError):

        class NegativeDim(pt.Dim, minimum=-1):
            pass


def test_dimension_naming_is_a_convention_not_a_runtime_rule() -> None:
    class components(pt.Dim):
        pass

    assert pt.parse(3, pt.Size[components], "count") == 3

type Mode = Literal["direct", "iterative"]


def test_pep695_alias_preserves_literal_contract() -> None:
    assert pt.parse("direct", Mode, "mode") == "direct"
    with pytest.raises(ValueError, match="mode"):
        pt.parse("invalid", Mode, "mode")


@pytest.mark.parametrize("token", [pt.AnyDim, pt.AnyShape, pt.Scalar])
def test_shape_tokens_cannot_be_instantiated_or_extended(token: Any) -> None:
    with pytest.raises(TypeError):
        token()
    with pytest.raises(TypeError):
        type("Extended", (token,), {})


@pytest.mark.parametrize(
    "form",
    [
        pt.Float64,
        pt.Size,
        pt.Identifiers,
        # ty: ignore[invalid-type-form]
        pt.Float64[3],
        pt.Float64[Literal[-1]],
        pt.Float64[Literal[1, 2]],
        pt.Float64[BatchDims, BatchDims],
        pt.Float64[pt.AnyShape, ComponentDim],
        pt.Float64[pt.Scalar, ComponentDim],
        pt.Float64[pt.Broadcast[BatchDims]],
        pt.Size[BatchDims],
        pt.Like[pt.Float64[ComponentDim]],
        list[pt.Float64[ComponentDim]],
        tuple[pt.Float64[ComponentDim], ...],
        pt.Float64[ComponentDim] | int,
    ],
)
def test_malformed_or_unsupported_forms_are_refused(form: Any) -> None:
    with pytest.raises(TypeError):
        pt.parse(None, form, "value")
