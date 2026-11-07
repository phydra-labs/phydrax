"""Ownership and validation checks for the implicit contour C ABI."""

import ctypes
from typing import Protocol

import numpy as np
import pytest

from phydrax._meshcore import load_meshcore
from phydrax.geometry.implicit._adaptive_native_polygon import contours


class _ContourKernel(Protocol):
    def __call__(self, *args: int | float | None) -> int: ...


@pytest.fixture
def kernel() -> _ContourKernel:
    library = load_meshcore()
    function = library._library["phx_mc_implicit_contours"]
    p = ctypes.c_void_p
    function.restype = ctypes.c_int32
    function.argtypes = (
        ctypes.c_int64,
        ctypes.c_int64,
        p,
        p,
        p,
        p,
        p,
        p,
        p,
        ctypes.c_double,
        p,
        p,
        p,
        p,
    )
    return function


def test_native_rejects_null_and_unaddressable_buffers(kernel: _ContourKernel) -> None:
    null = None
    assert (
        kernel(
            1, 0, null, null, null, null, null, null, null, 0.0, null, null, null, null
        )
        != 0
    )
    assert (
        kernel(
            2**63 - 1,
            0,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            0.0,
            null,
            null,
            null,
            null,
        )
        != 0
    )


@pytest.mark.parametrize(
    "bad",
    (
        "out_of_range",
        "duplicate_root",
        "duplicate_vertex",
        "missing_crossing",
        "extra_crossing",
        "bad_sign",
    ),
)
def test_native_rejects_bad_indices_without_modifying_outputs(
    kernel: _ContourKernel, bad: str
) -> None:
    signs = np.array([[1, -1, -1, -1, -1, -1, -1, -1]], dtype=np.int8)
    values = signs.astype(np.float64)
    boxes = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
    roots = np.full((1, 12), -1, dtype=np.int64)
    roots[0, [0, 3, 8]] = [0, 1, 2]
    vertices = np.arange(3, dtype=np.int64)
    if bad == "out_of_range":
        roots[0, 8] = 3
    elif bad == "duplicate_root":
        roots[0, 8] = 0
    elif bad == "duplicate_vertex":
        vertices[2] = 0
    elif bad == "missing_crossing":
        roots[0, 8] = -1
    elif bad == "extra_crossing":
        roots[0, 1] = 2
    else:
        signs[0, 1] = 0
    points = np.zeros((3, 3), dtype=np.float64)
    status = np.zeros(3, dtype=np.int64)
    handled = np.full(1, 7, dtype=np.uint8)
    counts = np.full(1, 7, dtype=np.int8)
    triangles = np.full((1, 10, 3), 7, dtype=np.int64)
    issues = np.full(1, 7, dtype=np.int64)
    inputs = (signs, values, boxes, roots, vertices, points, status)
    outputs = (handled, counts, triangles, issues)
    rc = kernel(
        1, 3, *(x.ctypes.data for x in inputs), 0.0, *(x.ctypes.data for x in outputs)
    )
    assert rc != 0
    for output in outputs:
        assert np.all(output == 7)


def test_native_wrapper_rejects_wrong_shapes(kernel: _ContourKernel) -> None:
    with pytest.raises(ValueError, match="shapes"):
        contours(
            np.ones((1, 8), np.int8),
            np.ones((1, 7)),
            np.zeros((1, 2, 3)),
            np.full((1, 12), -1, np.int64),
            np.empty(0, np.int64),
            np.empty((0, 3)),
            np.empty(0, np.int64),
            np.empty(0),
        )


@pytest.mark.parametrize(
    "fault",
    (
        "null_signs",
        "null_values",
        "null_boxes",
        "null_roots",
        "null_vertices",
        "null_points",
        "null_root_issues",
        "null_handled",
        "null_counts",
        "null_triangles",
        "null_issues",
        "negative_count",
        "negative_root_count",
        "oversized_count",
        "oversized_root_count",
        "bad_sign",
        "bad_root_high",
        "bad_root_negative",
        "missing_crossing",
        "extra_crossing",
        "duplicate_root",
        "duplicate_vertex",
        "negative_vertex",
        "negative_minimum_area",
        "nonfinite_minimum_area",
    ),
)
def test_extended_abi_rejects_malformed_without_writing(
    kernel: _ContourKernel, fault: str
) -> None:
    signs = np.tile(np.asarray((-1, 1, 1, -1, -1, 1, 1, -1), dtype=np.int8), (2, 1))
    values = signs.astype(np.float64)
    boxes = np.tile(np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (2, 1, 1))
    roots = np.full((2, 12), -1, dtype=np.int64)
    roots[:, [0, 2, 4, 6]] = np.arange(4, dtype=np.int64)
    vertices = np.arange(4, dtype=np.int64)
    points = np.zeros((4, 3), dtype=np.float64)
    root_issues = np.zeros(4, dtype=np.int64)
    handled = np.full(2, 165, dtype=np.uint8)
    counts = np.full(2, 37, dtype=np.int8)
    triangles = np.full((2, 10, 3), 123456789, dtype=np.int64)
    issues = np.full(2, 987654321, dtype=np.int64)
    outputs = (handled, counts, triangles, issues)
    before = tuple(a.copy() for a in outputs)
    count, root_count, minimum_area = 2, 4, 0.0
    if fault == "negative_count":
        count = -1
    elif fault == "negative_root_count":
        root_count = -1
    elif fault == "oversized_count":
        count = np.iinfo(np.int64).max
    elif fault == "oversized_root_count":
        root_count = np.iinfo(np.int64).max
    elif fault == "bad_sign":
        signs[1, 0] = 0
    elif fault == "bad_root_high":
        roots[1, 0] = root_count
    elif fault == "bad_root_negative":
        roots[1, 0] = -2
    elif fault == "missing_crossing":
        roots[1, 0] = -1
    elif fault == "extra_crossing":
        roots[1, 1] = 0
    elif fault == "duplicate_root":
        roots[1, 2] = roots[1, 0]
    elif fault == "duplicate_vertex":
        vertices[2] = vertices[0]
    elif fault == "negative_vertex":
        vertices[2] = -1
    elif fault == "negative_minimum_area":
        minimum_area = -1.0
    elif fault == "nonfinite_minimum_area":
        minimum_area = float("nan")
    names = (
        "signs",
        "values",
        "boxes",
        "roots",
        "vertices",
        "points",
        "root_issues",
        "handled",
        "counts",
        "triangles",
        "issues",
    )
    arrays = (signs, values, boxes, roots, vertices, points, root_issues, *outputs)
    pointers = [
        None if fault == "null_" + name else array.ctypes.data
        for name, array in zip(names, arrays, strict=True)
    ]
    status = kernel(count, root_count, *pointers[:7], minimum_area, *pointers[7:])
    assert status == 1, fault
    for actual, expected in zip(outputs, before, strict=True):
        np.testing.assert_array_equal(actual, expected, err_msg=fault)
