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

Installing cuTENSOR or cuTensorNet does not alter this function object,
register a provider backend here, change its optimizer, or intercept calls.
Explicit optional execution is selected at the canonical owner:
`phydrax.tensor_network.prepare_contraction(plan, operands, execution=...)`,
with `ContractionExecutionPolicy("cutensor")` or
`ContractionExecutionPolicy("cutensornet")`. Omitting the execution request
selects the native provider.

Path selection is separate:
`plan_contraction(..., path=...)` imports a path, while
`plan_contraction(..., path_planner=ContractionPathPlanner(...))` requests bounded
shape-only cuTensorNet planning. These are mutually exclusive with an explicit
native `optimizer`. A different executor on one plan is a same-path comparison;
a newly planned path is an algorithmic change. No speedup is implied.

The [tensor platform guide](../guides_tensor_platform.md#explicit-local-executors-and-path-planning)
documents exact request fields, precision and transformation limits, external
device-byte bounds versus unknown opaque host storage, and a complete runnable
example. The [tensor-network API](tensor_network.md) owns these objects; they
are not exports of `phydrax.ein`.

Prepared sliced calls require
`prepare_sliced_contraction(prepared, sliced_plan)` followed by
`execute_sliced_contraction(binding)`, never a plan plus unbound operands.
Optional executors intend fp32/fp64/c64/c128 admission, not blanket hardware
qualification, and require equal multiply/accumulation dtypes. Native-owned
optional-provider AD rules admit derivatives through order two; host
preparation/checkpoint/distribution is excluded. Charge, symmetry, and
TRG/HOTRG retain their native owners without implicit densification.

Source contracts:
[`ein`'s direct import](https://github.com/phydra-labs/phydrax/blob/main/phydrax/ein/__init__.py),
[prepared planning/execution](https://github.com/phydra-labs/phydrax/blob/main/phydrax/tensor_network/_contraction.py),
[policy/planner definitions](https://github.com/phydra-labs/phydrax/blob/main/phydrax/tensor_network/_contraction_provider.py),
[provider admission](https://github.com/phydra-labs/phydrax/blob/main/phydrax/tensor_network/_cutensor.py), and
[explicit sliced binding](https://github.com/phydra-labs/phydrax/blob/main/phydrax/tensor_network/_slicing.py).

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
