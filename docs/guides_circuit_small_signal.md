# Circuit small-signal and descriptor analysis

`linearize_circuit` differentiates the prepared implicit circuit residual with respect to state and state rate at one operating point. It returns the shared `phydrax.dynamics.LinearDescriptorSystem` convention

```text
E xdot = A x + B u
y      = C x + D u
```

For the repository-wide `exp(-i omega t)` convention, `descriptor_frequency_response` solves

```text
(-i omega E - A) x = B u
```

through native linear solve policies. The circuit adapter currently exposes identity residual forcing and full state observation; physical port-specific input/output maps can be composed explicitly.

Descriptor pole analysis uses the native general generalized-eigenvalue substrate. Singular mass matrices are allowed when the pencil is regular. Infinite algebraic modes, unstable finite poles, solve residuals, condition estimates, and nonfinite outputs remain explicit evidence.

Small-signal derivatives are valid only for fixed topology, state layout, operating branch, and rank. A fold, event, pole-selection change, or irregular pencil invalidates the ordinary implicit derivative.
