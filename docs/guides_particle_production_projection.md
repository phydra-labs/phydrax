# Balanced multiphase and projection qualification

`corrected_phase_interface_geometry` produces phase color, corrected normals,
interface delta, and confidence. `ContinuumSurfaceStressPlan` constructs a
reciprocal continuum surface-stress force. `BalancedInterfaceForcePlan` combines
this with the volume-based pressure interaction and optional contact-angle plan.
Production qualification uses pressure-jump convergence and parasitic-current
reduction, not action--reaction alone.

`assemble_iisph_operator` constructs a small-system authority for the matrix-free
pressure action. Operator diagnostics report symmetry, diagonal, row sums,
quadratic forms, and finiteness. `ProductionProjectedSolvePlan` exposes projected
iteration residual and pressure complementarity. Warm-start rescaling and
frozen-active-set tangents are explicit.

DFSPH exposes factor oracles, dimensionless density/divergence constraints,
multiplier timestep rescaling, and selected projected-iteration acceleration.
Execution success remains distinct from production constraint satisfaction.
