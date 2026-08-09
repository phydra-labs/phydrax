# General affine connections

`AbstractAffineConnection` is the minimal contract for connection coefficients in a
coordinate chart. `LeviCivitaConnection` is the torsion-free, metric-compatible
specialization derived from a metric; `CallableAffineConnection` permits torsion and
nonmetricity.

Connection coefficients are not tensors. `pullback_affine_connection` implements the
full inhomogeneous coordinate-transformation law, including the second derivative of
the coordinate map. `connection_transformation_residual` exposes that law as a
numerical diagnostic.

::: phydrax.metrix.AbstractAffineConnection

::: phydrax.metrix.CallableAffineConnection

::: phydrax.metrix.LeviCivitaConnection

::: phydrax.metrix.pullback_affine_connection

::: phydrax.metrix.connection_transformation_residual

::: phydrax.metrix.torsion_tensor

::: phydrax.metrix.nonmetricity_tensor

::: phydrax.metrix.affine_covariant_derivative

::: phydrax.metrix.connection_covariant_hessian

::: phydrax.metrix.connection_divergence

::: phydrax.metrix.connection_riemann_tensor

::: phydrax.metrix.connection_ricci_tensor
