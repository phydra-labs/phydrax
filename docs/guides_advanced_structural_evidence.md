# Advanced structural evidence

This layer extends member-network verification with catenaries, contact,
section-orientation fields, connection mechanics, warping beams, fiber
plasticity, local/distortional buckling, collapse events, construction-order
search, standards adapters, reliability, calibration, and immutable evidence
graphs.

It does not replace the fidelity boundaries of force-density or member-network
results. Every advanced result records its model, assumptions, and applicability.

## Generalized coordinates and orientation

`GeneralizedDOFLayout` combines named shaped channels such as warping,
cross-section modes, slip, and contact multipliers. Each channel has its own
constraint map and state geometry.

`SectionOrientationField` records frames and their source: explicit director,
CAD frame, surface normal, joint frame, Bishop transport, or optimization.
`parallel_transport_orientations` detects antiparallel tangent singularities and
closed-loop holonomy. Branching joints require explicit connection frames.

## Connections and supports

`LinearConnectionSpringBlock` and `NonlinearMomentRotationBlock` add
translational/rotational connection energy. Gap and friction support responses
retain complementarity and cone residuals. Connection stiffness, releases,
offsets, and support behavior must be explicit before effective-length or frame
stability claims are made.

## Elastic catenaries and contact

`solve_elastic_catenary` solves the extensible distributed-load cable equations
in material coordinates. The catenary state retains endpoint forces, sampled
centerline, strained length, sag, minimum tension, energy, regime, nested root
evidence, and validity.

Regimes distinguish slack, catenary, near-straight, near-vertical, and
zero-distributed-load behavior. A straight truss element is not silently used as
a catenary.

Node/plane and cable/saddle contact retain gap, traction, friction cone,
stick/slip, wrap angle, and capstan-limit evidence. General mixed
complementarity uses the native variational-inequality and semismooth Newton
runtime.

## Warping beams and bracing

`WarpingBeamSection` extends a physical beam section with shear-center offsets,
warping constant, and monosymmetry data. `evaluate_warping_beam` returns axial,
bending, Saint-Venant torsion, warping energy, bimoment, and load-height
coupling. `evaluate_bracing` adds lateral, torsional, and warping restraint.

This supports nonuniform torsion and LTB model construction. It does not infer
brace stiffness, load height, or warping restraint.

## Fiber plasticity and collapse

`FiberSectionGeometry` maps fibers to bilinear uniaxial materials.
`evaluate_fiber_section` performs a consistent return map and returns fiber
stress, tangent, section resultants, yielded/fractured fibers, plastic
dissipation, and transactional history. Accepted increments commit; rejected
increments discard trial state.

`classify_collapse` identifies limit points, branch points, tangent instability,
plastic mechanisms, strain/rotation limits, fracture, contact transitions, or
unbounded response. Numerical solver failure is not a collapse event.

`newmark_step` supplies an energy-reporting average-acceleration dynamic step for
qualified mass, damping, tangent, and force matrices.

## Thin-walled local buckling

`ThinWalledSection` stores plate midlines, thickness, material mapping, free
edges, and closed cells. `compute_gbt_modes` builds a cross-section deformation
basis. `solve_finite_strip_buckling` sweeps longitudinal half-wavelengths and
reports critical stress, wavelength, governing plate, local/distortional/global
family, interaction margin, and GBT evidence.

`compare_shell_submodel` records beam/strip/shell agreement and boundary-transfer
residuals. Disagreement is model-form evidence, not an error to suppress.

## Construction-order optimization

`PrecedenceSpace` stores a construction DAG, exclusivity, simultaneous groups,
and resource limits. `branch_and_bound` performs deterministic best-bound search
with explored/pruned counts and a certified gap. The member-network adapter
accepts structural prefix feasibility, lower-bound, and complete-objective
callbacks.

Heuristic bounds may guide approximate search but must not certify optimality.

## Standards adapters

`GenericLimitStateStandard` provides user-supplied load combinations and
resistance factors with organization, edition, jurisdiction, clause, case,
applicability, demand, capacity, and utilization evidence. Proprietary tables
remain external.

`direct_analysis_inputs` retains explicit notional loads, stiffness reduction,
and imperfection assumptions. Effective lengths are not inferred from graph
topology.

## Reliability and calibration

`StructuralRandomModel` maps correlated Gaussian variables into physical
parameters. Monte Carlo and FORM evaluate scalar structural limit states. FORM
is valid only on a smooth branch; active-set switches and bifurcations require
directional or sampling evidence.

`StructuralCalibrationProblem` combines priors, structural observation models,
measurement covariance, and model discrepancy. MAP calibration returns posterior
Hessian/covariance and identifiability evidence.

## Evidence graphs and twins

`EvidenceGraph` stores dependencies, model fidelity, status, missing inputs, and
acquisition actions. Actions can be ranked by information gain per cost.
`StructuralTwinSnapshot` creates immutable design, as-built, calibrated, and
service-state ancestry.

## Scientific boundaries

- No design-code verdict without an edition, clause, applicability, and data.
- No LTB claim without warping, bracing, orientation, and load-height inputs.
- No local buckling claim from beam properties alone.
- No unique smooth cable/contact derivative at a switch.
- No collapse inferred from solver failure.
- No committed plastic state from a rejected increment.
- No heuristic sequence labeled globally optimal.
- No reliability result without an uncertainty and discrepancy model.

Runnable workflows:

- `examples/advanced_catenary_contact.py`;
- `examples/advanced_warping_local_buckling.py`;
- `examples/advanced_sequence_reliability.py`.
