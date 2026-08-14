# Composite systems and a Bell state

This recipe constructs the Bell state

$$
|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt2},
$$

forms its density operator, reduces each subsystem, and evaluates local and correlated
Pauli observables.

```python
import jax.numpy as jnp
import phydrax as phx

time = phx.domain.TimeInterval(0.0, 1.0)
zero = time.Function()(jnp.asarray([1.0, 0.0], dtype=complex))
one = time.Function()(jnp.asarray([0.0, 1.0], dtype=complex))
sigma_x = time.Function()(jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
sigma_z = time.Function()(jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex))

zero_zero = phx.operators.tensor_product(zero, zero)
one_one = phx.operators.tensor_product(one, one)
bell = (zero_zero + one_one) / jnp.sqrt(2.0)


@time.Function()
def bell_factor():
    return bell.func()[:, None]


rho = phx.operators.density_from_factor(bell_factor)
rho_a = phx.operators.partial_trace(
    rho,
    subsystem_dims=(2, 2),
    trace_out=1,
)
rho_b = phx.operators.partial_trace(
    rho,
    subsystem_dims=(2, 2),
    trace_out=0,
)

expected_reduction = 0.5 * jnp.eye(2)
assert jnp.allclose(rho_a.func(), expected_reduction)
assert jnp.allclose(rho_b.func(), expected_reduction)
assert jnp.allclose(jnp.trace(rho.func()), 1.0)
assert jnp.all(jnp.linalg.eigvalsh(rho_a.func()) >= -1e-12)

maximally_mixed = time.Function()(expected_reduction)
assert jnp.allclose(phx.operators.purity(rho).func(), 1.0)
assert jnp.allclose(phx.operators.purity(rho_a).func(), 0.5)
assert jnp.allclose(phx.operators.von_neumann_entropy(rho_a).func(), 1.0)
assert jnp.allclose(
    phx.operators.density_fidelity(rho_a, maximally_mixed).func(),
    1.0,
)
assert jnp.allclose(
    phx.operators.trace_distance(rho_a, maximally_mixed).func(),
    0.0,
)

x_on_a = phx.operators.embed_operator(
    sigma_x,
    subsystem=0,
    subsystem_dims=(2, 2),
)
z_on_b = phx.operators.embed_operator(
    sigma_z,
    subsystem=1,
    subsystem_dims=(2, 2),
)
zz = phx.operators.tensor_product(sigma_z, sigma_z)

assert jnp.allclose(phx.operators.density_expectation(rho, x_on_a).func(), 0.0)
assert jnp.allclose(phx.operators.density_expectation(rho, z_on_b).func(), 0.0)
assert jnp.allclose(phx.operators.density_expectation(rho, zz).func(), 1.0)
```

`subsystem_dims` is deliberately explicit. A four-dimensional value could represent
one four-level system or two two-level systems; array shape alone cannot determine the
factorization. Untraced subsystems retain their original order. Tracing every
subsystem returns the scalar total trace, while `trace_out=()` leaves the matrix
unchanged.

The global Bell density is pure, while either reduced qubit has purity $1/2$ and
entropy one bit. `density_fidelity` uses the squared Uhlmann convention, and
`trace_distance` includes the conventional factor of $1/2$.

`tensor_product` accepts either all vector-valued or all square-matrix-valued factors.
Mixing vectors and matrices in one call is rejected. Use `embed_operator` to place a
local square operator on one subsystem while identities are inserted on the others.
