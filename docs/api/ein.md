# Einstein operations

`phydrax.ein` is the package-wide substrate for optimized tensor contractions and
static JAX axis transformations.

- `contract` is the exact `opt_einsum.contract` function object. Its compact and
  interleaved equations, optimizer controls, backend selection, return types,
  and errors are preserved.
- `rearrange`, `reduce`, and `repeat` compile named unary patterns into a bounded
  sequence of JAX reshape, transpose, reduction, and broadcast primitives.

See the [Einstein operations guide](../guides_ein.md) for the pattern grammar,
static-shape contract, and examples.

## Public API

### `contract`

```python
contract(
    subscripts,
    *operands,
    out=None,
    use_blas=True,
    optimize=True,
    memory_limit=None,
    backend="auto",
    **kwargs,
)
```

This is the exact `opt_einsum.contract` function object rather than a wrapper.
See the
[opt-einsum contract reference](https://optimized-einsum.readthedocs.io/en/stable/reference/generated/opt_einsum.contract.html)
for compact/interleaved notation, path optimizers, and backend-specific
arguments.

::: phydrax.ein.rearrange
    options:
      show_root_heading: true
      show_source: false

::: phydrax.ein.reduce
    options:
      show_root_heading: true
      show_source: false

::: phydrax.ein.repeat
    options:
      show_root_heading: true
      show_source: false

## Deliberate boundaries

The namespace does not provide named contraction strings, contraction-plan
objects, packing, runtime axis metadata, or sharding annotations. Structured
resource-bounded contraction planning remains in
[`phydrax.tensor_network`](tensor_network.md).
