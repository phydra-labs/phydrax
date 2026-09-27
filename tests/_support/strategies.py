from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import SearchStrategy


def finite_arrays(
    *,
    dtype: npt.DTypeLike,
    shapes: Sequence[tuple[int, ...]],
) -> SearchStrategy[npt.NDArray[np.generic]]:
    """Generate finite host arrays over a small, explicitly declared shape set."""
    normalized_dtype = np.dtype(dtype)
    elements = hnp.from_dtype(
        normalized_dtype,
        allow_nan=False,
        allow_infinity=False,
    )
    return st.sampled_from(tuple(shapes)).flatmap(
        lambda shape: hnp.arrays(normalized_dtype, shape, elements=elements)
    )


def integer_arrays(
    *,
    dtype: npt.DTypeLike,
    shapes: Sequence[tuple[int, ...]],
    minimum: int,
    maximum: int,
) -> SearchStrategy[npt.NDArray[np.generic]]:
    """Generate bounded host integer arrays over declared shapes."""
    normalized_dtype = np.dtype(dtype)
    elements = st.integers(min_value=minimum, max_value=maximum)
    return st.sampled_from(tuple(shapes)).flatmap(
        lambda shape: hnp.arrays(normalized_dtype, shape, elements=elements)
    )
