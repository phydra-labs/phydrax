# Neural networks

Phydrax models are Equinox modules with explicit `in_size` / `out_size` semantics and
support for structured inputs used in product-domain factorization.

!!! note
    Key notes:

    - Most models are pointwise: use `jax.vmap` for batching.
    - `out_size="scalar"` indicates scalar outputs (typically shape `()`).
    - Structured models accept tuple inputs like `(x1, x2, ..., xd)`.
    - Neural operator architectures (DeepONet/FNO) are intentionally minimal reference
      implementations; extend them for production features and scaling needs.
