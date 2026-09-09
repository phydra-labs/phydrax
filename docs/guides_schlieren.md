# Schlieren and background-oriented Schlieren

Schlieren detector values, angular deflection, apparent image displacement, and inferred density are separate quantities.

```text
field gradients → path-integrated deflection → detector signal or BOS displacement
```

## Straight-ray deflection

`SchlierenDeflectionPlan` binds one ray per flattened image sample, fixed segment lengths, an orthonormal detector transverse basis, quantity contracts, and an optional `GladstoneDaleRelation`.

The input is sampled refractive-index gradient or density gradient in the declared
ray reference frame. With a Gladstone–Dale relation, coefficient,
density-gradient unit, and path-length unit must close dimensionlessly; exact
unit scales are applied. Output deflection uses radians and a two-component
detector frame.

The initial method uses fixed straight rays and bounded segment quadrature. It reports finite input, ray support coverage, a small-angle validity check, and success. It does not claim nonlinear ray bending, caustics, diffraction, or full-wave Schlieren.

## Knife-edge image formation

`KnifeEdgeSchlierenPlan` applies an explicit cutoff direction and calibrated contrast gain to a reference `ImageAsset`. Its result is a predicted detector quantity compatible with the reference image. It is not relabelled as density.

## Background-oriented Schlieren

`BackgroundOrientedSchlierenPlan` maps angular deflection to row/column pixel displacement and nonperiodically warps a reference background. Warp boundaries remain invalid. The displacement and predicted detector image are both retained.

Density recovery is an inverse problem over these forward predictions. Any recovered density asset must carry inferred-stage lineage and parent measurement identities.
