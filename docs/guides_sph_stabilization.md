# SPH correction and stabilization

`SPHFirstOrderGradientCorrectionPlan` constructs the local kernel moment matrix
and solves its batched systems through `phydrax.linalg`. Results include the
correction action, condition estimate, residual, and success mask; failed local
systems are never silently accepted.

`AntuonoDeltaSPHDiffusionPlan` extends continuity density with corrected density
gradients. `MolteniColagrossiDensityDiffusionPlan` is the explicitly named cheap
variant and does not claim the same long-time hydrostatic behavior. Density
variance rate and correction status are observable.

`MonaghanArtificialViscosityPlan` supports approaching-only, always-active, and
smooth-approach policies. Pair forces are equal and opposite; pair kinetic power,
dissipation, positive-power defect, and active-pair count are reported. In
barotropic WCSPH it is an energy sink because no thermal state exists.

`ShepardDensityRenormalizationTransform` applies the normalized density estimate
on a fixed accepted-step schedule. It reports application and correction norm,
rejects excessive corrections, and never hides itself inside the drift.
