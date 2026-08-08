# Neural networks

Phydrax models are Equinox modules with explicit `in_size` / `out_size` semantics and
support for structured inputs used in product-domain factorization.

Public ownership is explicit: `phydrax.nn.models` contains pointwise, separable,
and process models; `phydrax.nn.layers` contains reusable layers and Fourier
feature embeddings; and `phydrax.nn.operator` contains neural-operator data,
engines, architectures, layers, adapters, and training workflows.

!!! note
    Key notes:

    - Most models are pointwise: use `jax.vmap` for batching.
    - `out_size="scalar"` indicates scalar outputs (typically shape `()`).
    - Structured models accept tuple inputs like `(x1, x2, ..., xd)`.
    - Models may contribute parameter-space penalties through `model.add_model_loss(...)`
      or a custom `__loss__` method; `FunctionalSolver` adds these to the train objective.
    - Neural operators use explicit source/query samples, coordinates, quadrature,
      masks, and case axes; see [Architectures](architectures.md).
