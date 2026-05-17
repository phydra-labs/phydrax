#
#  Copyright 2026 PHYDRA, Inc. All rights reserved.
#

from typing import cast

import jax.numpy as jnp
import polars as pl
import pytest
from jaxtyping import Array

from phydrax.data_utils import CSVReader
from phydrax.data_utils._utils import _is_numeric_dtype


def test_csv_reader_reads_numeric_columns(tmp_path):
    path = tmp_path / "numeric.csv"
    path.write_text("x,y,z\n1.0,4.0,7.0\n2.0,5.0,8.0\n3.0,6.0,9.0\n")

    reader = CSVReader(path)
    assert len(reader) == 3
    assert reader.columns == ["x", "y", "z"]
    assert jnp.allclose(cast(Array, reader["x"]), jnp.asarray([1.0, 2.0, 3.0]))
    xy = reader[["x", "y"]]
    assert not isinstance(xy, dict)
    assert jnp.allclose(
        xy,
        jnp.asarray([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
    )
    assert jnp.allclose(
        reader.to_array(),
        jnp.asarray([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]]),
    )


def test_csv_reader_reads_mixed_columns(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text("x,y,name\n1.0,4.0,alpha\n2.0,5.0,beta\n3.0,6.0,gamma\n")

    reader = CSVReader(path)
    assert reader.columns == ["x", "y", "name"]
    assert jnp.allclose(cast(Array, reader["x"]), jnp.asarray([1.0, 2.0, 3.0]))
    assert reader["name"] == ["alpha", "beta", "gamma"]

    selected = reader[["x", "name"]]
    assert isinstance(selected, dict)
    assert selected["name"] == ["alpha", "beta", "gamma"]
    assert jnp.allclose(cast(Array, selected["x"]), jnp.asarray([1.0, 2.0, 3.0]))

    with pytest.raises(TypeError, match="Cannot convert non-numeric CSV data"):
        reader.to_array()

    data = reader.to_dict()
    assert set(data) == {"x", "y", "name"}
    assert jnp.allclose(cast(Array, data["y"]), jnp.asarray([4.0, 5.0, 6.0]))
    assert data["name"] == ["alpha", "beta", "gamma"]


def test_csv_reader_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        CSVReader("missing.csv")


def test_csv_reader_forwards_polars_read_csv_kwargs(tmp_path):
    path = tmp_path / "no_header.csv"
    path.write_text("1,2,3\n4,5,6\n")

    reader = CSVReader(path, has_header=False)
    assert reader.columns == ["column_1", "column_2", "column_3"]

    named = CSVReader(path, has_header=False, new_columns=["a", "b", "c"])
    assert named.columns == ["a", "b", "c"]
    values = named[["a", "b", "c"]]
    assert not isinstance(values, dict)
    assert jnp.allclose(values, jnp.asarray([[1, 2, 3], [4, 5, 6]]))


def test_is_numeric_dtype_for_polars_series_and_frame():
    numeric_frame = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    mixed_frame = pl.DataFrame({"a": [1, 2, 3], "name": ["a", "b", "c"]})

    assert _is_numeric_dtype(numeric_frame) is True
    assert _is_numeric_dtype(mixed_frame) is False
    assert _is_numeric_dtype(numeric_frame.get_column("a")) is True
    assert _is_numeric_dtype(mixed_frame.get_column("name")) is False
