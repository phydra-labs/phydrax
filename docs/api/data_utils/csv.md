# CSV Reader

`CSVReader` is a thin Polars-backed CSV helper. Numeric columns are returned as
JAX arrays; non-numeric columns are returned as Python lists.

::: phydrax.data_utils.CSVReader
    options:
        members:
            - __init__
            - __getitem__
            - columns
            - to_array
            - to_dict
