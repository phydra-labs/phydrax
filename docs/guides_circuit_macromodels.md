# Causal circuit macromodels

A `SampledScatteringModel` is an explicitly in-band interpolator. It does not imply causality, stability, passivity, or a time-domain realization.

`fit_rational_matrix` fits a shared set of finite stable poles to a MIMO frequency response with native weighted least squares. The model is

```text
H(s) = D + s P + sum_j R_j / (s - p_j)
```

Poles must remain finite in the stable half-plane. Fit evidence reports the physical residual, maximum point error, native linear status, finiteness, and acceptance against the declared tolerance.

`realize_rational_model` produces a shared `LinearDescriptorSystem` for proper models. A nonzero proportional term must be handled explicitly and is not silently approximated. `RationalScatteringComponent` exposes the fitted model in frequency-domain circuit composition.

`reduce_rational_model` ranks poles by residue norm and reports discarded strength. This general truncation does **not** claim passivity preservation. Passivity-preserving reduction requires a separately certified structured realization; the result therefore exposes `passivity_preserved=False` rather than inferring a claim from sampled data.

`audit_rational_scattering` is a finite-frequency bounded-real audit and explicitly
sets `certified=False`. `passive_descriptor_system` instead constructs a positive-real
descriptor from a positive energy matrix, skew interconnection, positive-semidefinite
dissipation/feedthrough, and conjugate input/output maps. Its certificate is by
construction and the constructor rejects any failed structural condition.
