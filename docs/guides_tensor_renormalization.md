# Tensor renormalization

Phydrax provides fixed-rank TRG and HOTRG for one rank-four tensor repeated on an
infinite square lattice. The initial scope is dense bosonic partition tensors.
It estimates the thermodynamic-limit logarithmic partition function per original
site; it does not claim a global truncation-error bound.

## Representation

`UniformSquareTensor` uses the ordered axes `(up, right, down, left)`. Opposite
bond dimensions must match, while vertical and horizontal dimensions may differ.
The representation stores one tensor and its precision/structure identities; it
does not attach contraction labels to array values.

`build_uniform_pair_partition_tensor` lowers real Hermitian positive-semidefinite
vertical and horizontal pair-weight matrices into this representation. An optional
nonnegative site-weight vector supplies one local factor per physical state. The
builder reports Hermiticity, minimum-eigenvalue, square-root reconstruction,
precision, finiteness, and acceptance evidence.

For a homogeneous Ising model with spins `s = (-1, 1)`, the pair matrix is
`Q[s,t] = exp(beta * s * t)`.

## Plan, prepare, refresh, execute

Choose the algorithm explicitly:

```python
import jax.numpy as jnp
import phydrax as phx

spins = jnp.asarray((-1.0, 1.0), dtype=jnp.float64)
beta = 0.44068679350977147
pair_weight = jnp.exp(beta * spins[:, None] * spins[None, :])

tn = phx.tensor_network
built = tn.build_uniform_pair_partition_tensor(pair_weight)
problem = tn.TensorRenormalizationProblem(built.tensor)
policy = tn.TensorRenormalizationPolicy(
    tn.HOTRGMethod(first_direction="vertical"),
    maximum_bond_dimension=8,
    steps=10,
)
plan = tn.plan_tensor_renormalization(problem, policy)
prepared = tn.prepare_tensor_renormalization(problem, plan)
result = tn.run_tensor_renormalization(prepared)
```

`TRGMethod` performs two checkerboard matrix splittings and contracts the four
resulting rank-three factors. `HOTRGMethod` alternates vertical and horizontal
blocking, builds one higher-order density matrix for the paired transverse bonds,
and uses one common isometry on both sides of the coarse tensor. Applying the same
basis on both sides preserves the untruncated periodic closure.

Planning fixes every stage shape, retained rank, contraction schedule, history
size, factorization size, estimated work, and admitted peak workspace. Shapes may
change between stages according to deterministic dimension algebra; they never
depend on singular values. `refresh_tensor_renormalization` accepts new numerical
values only when the tensor structure and problem identity remain unchanged.

## Normalization and terminal correction

The input and every coarse tensor are normalized by their maximum absolute entry.
If scale `c_k` is introduced after coarse step `k`, its contribution per original
site is `log(c_k) / 2^k`. The final normalized tensor is closed on a one-site
periodic cell, and its contribution is divided by `2^steps`.

The terminal scalar must be finite and positive-real within the policy tolerance.
A non-positive terminal value returns `TERMINAL_PARTITION_INVALID`. Complex
partition tensors are refused during problem construction rather than assigned
an implicit logarithm branch.

`TensorRenormalizationResult` separates:

- accumulated normalization contribution;
- terminal correction and terminal value;
- final logarithmic partition density;
- final coarse tensor;
- per-stage retained ranks and discarded squared weights;
- finite history and terminal checks;
- status, provenance, precision, and admitted resources.

A unit-bond tensor is the only route marked exact. For larger bonds, even zero
local discarded weight does not turn a finite number of coarse steps into a
global error certificate.

## Precision and failures

`TensorNetworkPrecisionPolicy` independently controls storage, contraction,
factorization, accumulation, decision, and output roles. Coarse contractions use
`phydrax.ein`; fixed-rank matrix splits and HOTRG Hermitian spectra use native
Phydrax tensor/linalg substrates.

Planning refuses tensor, factorization, workspace, contraction, and history limits
before numerical allocation. Execution distinguishes non-finite input, zero
normalization, failed factorization, non-finite intermediate values, and invalid
terminal partition values. None is accepted.

## Qualification and benchmark

The deterministic qualification compares TRG and HOTRG against the Onsager square-
lattice Ising free energy at high, intermediate, critical, and low temperatures,
then checks the q=2 Potts mapping, a unit-bond exact result, non-finite refusal,
and pre-execution resource refusal:

```text
python tools/tensor_renormalization_qualification.py
```

The benchmark records lowering, compilation, synchronized execution, compiler
cost/memory evidence, logical bytes, planned peak workspace, truncation totals,
and critical-Ising error:

```text
python benchmarks/tensor_renormalization.py --repeats 3
```

A directly runnable example is available at
`examples/ising_tensor_renormalization.py`.

## Current boundary

The initial implementation does not provide GILT-TNR, adaptive rank selection,
Abelian or graded coarse graining, fermionic statistics, multi-device placement,
differentiation through truncation, arbitrary periodic unit cells, or a general
complex partition-function logarithm. Those are separate capabilities requiring
independent qualification.
