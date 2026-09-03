# Cardiovascular anatomy foundations

The cardiovascular anatomy layer turns a fixed affine tetrahedral mesh into audited boundary semantics, harmonic profile coordinates, ventricular material frames, and differentiable chamber volumes. It keeps topology preparation outside differentiated kernels and carries stable identities and numerical evidence into mechanics and electrophysiology.

The geometry kernel uses millimetres. Harmonic coordinates are dimensionless. Angles passed to the ventricular microstructure plan are degrees and committed angle fields are radians.

## Boundary roles are semantic, not numeric

A boundary role is a caller-owned name associated with exterior face indices. There is deliberately no global integer label table. A profile states only the relationships required by a case:

- required roles;
- roles that must each be one edge-connected component;
- pairs whose nodal closures must be disjoint;
- pairs that must share one edge-connected closure component; and
- whether every exterior face must be owned.

Face interiors always have disjoint ownership. Closure relationships are checked separately because adjacent anatomical patches normally share edges while opposite Dirichlet patches must not share P1 nodes.

```python
import numpy as np
from phydrax.applications.cardiovascular import anatomy

profile = anatomy.CardiacBoundaryProfile(
    "lv-coordinate-profile",
    required_roles=(
        "lv-endocardium",
        "epicardium",
        "apex",
        "base",
        "anterior",
        "posterior",
    ),
    connected_roles=(
        "lv-endocardium",
        "epicardium",
        "apex",
        "base",
    ),
    disjoint_closure_pairs=(
        ("lv-endocardium", "epicardium"),
        ("apex", "base"),
    ),
    shared_closure_pairs=(
        ("lv-endocardium", "base"),
        ("epicardium", "base"),
    ),
    exhaustive=True,
)

roles = anatomy.CardiacBoundaryRoles(
    tetrahedral_mesh,
    {
        "lv-endocardium": endocardial_face_indices,
        "epicardium": epicardial_face_indices,
        "apex": apical_face_indices,
        "base": basal_face_indices,
        "anterior": anterior_face_indices,
        "posterior": posterior_face_indices,
    },
    profile=profile,
)
assert bool(roles.evidence.successful)
```

`left_ventricular_boundary_profile()` is the minimal endocardium/epicardium/base LV-wall foundation. Passing an explicit `apex` adds a separate apical cap and the closure relations needed by the LV coordinate recipe. All semantic names are parameters, so case vocabularies remain local.

The qualified record exposes `face_indices(name)`, `vertex_mask(name)`, and `vertex_indices(name)`. The latter two represent the complete P1 nodal closure of the role. Mesh vertex and cell global IDs remain the authoritative stable entity identities.

### Chamber profiles and coordinate recipes

Three additional factories encode useful structural contracts without assigning numeric labels:

- `biventricular_boundary_profile()` requires separate LV and RV endocardia, epicardium, and distinct apical/basal caps. It validates the expected cavity/epicardial closure separation and each cap's shared rings.
- `atrial_boundary_profile()` requires separate LA and RA endocardia, atrial epicardium, mitral and tricuspid caps, and any explicitly listed left/right venous opening caps.
- `whole_heart_boundary_profile()` combines four endocardia, epicardium, four named valve/outflow caps, and caller-listed pulmonary-vein and vena-cava caps. Relationships are stated between named patches, never inferred from integer tags.

The matching coordinate-spec factories are recipes for harmonic boundary pairs:

```python
lv_specs = anatomy.left_ventricular_coordinate_specs(
    endocardium="lv-endo",
    epicardium="heart-epi",
    apex="lv-apical-cut",
    base="lv-basal-cut",
    roles=roles,  # optional: validates that every recipe role is present
)

biv_specs = anatomy.biventricular_coordinate_specs()
# lv-transmural: lv-endocardium -> epicardium
# rv-transmural: rv-endocardium -> epicardium
# biventricular-apicobasal: apex -> base
# lv-rv-separation: lv-endocardium -> rv-endocardium

atrial_specs = anatomy.atrial_coordinate_specs()
# la-transmural and ra-transmural, each against atrial-epicardium
```

These bundles are not a universal ventricular-coordinate or atrial-coordinate system. In particular, the biventricular recipe does not invent a circumferential seam, a rotational coordinate, or a septal blending law. The atrial recipe does not guess appendage/roof landmarks and therefore provides only chamber-specific transmural fields. Applications needing those coordinates must add case-qualified, non-overlapping landmark patches and explicit `HarmonicCoordinateSpec` values.

## Harmonic P1 coordinates

Each `HarmonicCoordinateSpec` supplies a field name, two semantic role names, and two finite endpoint values. Every connected mesh component must touch both roles, and the two nodal closures must be disjoint. Preparation uses the native affine P1 finite-element discretization to assemble the stiffness operator; solving uses the native linear algebra layer.

```python
coordinate_plan = anatomy.HarmonicCoordinatePlan(
    tetrahedral_mesh,
    roles,
    (
        anatomy.HarmonicCoordinateSpec(
            "transmural", "lv-endocardium", "epicardium"
        ),
        anatomy.HarmonicCoordinateSpec("longitudinal", "apex", "base"),
    ),
)
prepared_coordinates = coordinate_plan.prepare(numeric_version="case-geometry-7")
candidate_coordinates = prepared_coordinates.solve()

# Inspect before commit when operating a qualification or recovery workflow.
assert bool(candidate_coordinates.evidence.all_successful)
coordinates = candidate_coordinates.commit()
```

The committed fixed-shape record contains:

- `nodal_values[coordinate, node]`;
- exact affine cell-centre `cell_values[coordinate, cell]`;
- exact affine `cell_gradients[coordinate, cell, xyz]`;
- strong Dirichlet masks; and
- solver status, free residual, boundary error, maximum-principle violation, finite flags, and success flags.

`nodal(name)`, `cell(name)`, and `gradient(name)` avoid hard-coding coordinate positions. A failed solve remains an inspectable candidate, but `commit()` fails closed.

## Ventricular lines and full material frames

The rule-based ventricular construction uses two coordinate gradients. Let the normalized transmural direction be `t`. The longitudinal direction is the normalized exact projection of the longitudinal gradient onto the plane orthogonal to `t`. The circumferential direction is `cross(longitudinal, t)`.

At transmural fraction `u`, the helix angle is the exact linear rule

```
angle(u) = angle_endocardium + u * (angle_epicardium - angle_endocardium)
```

and the fiber direction is

```
fiber = cos(angle) * circumferential + sin(angle) * longitudinal
```

The sheet direction is transmural and the sheet-normal direction is `cross(fiber, sheet)`. Thus the ordered `(fiber, sheet, sheet-normal)` columns are right-handed. The physical fiber line is also stored as the structure tensor `fiber ⊗ fiber`; reversing the longitudinal gauge reverses the representative fiber vector but leaves this tensor unchanged.

```python
micro_plan = anatomy.VentricularMicrostructurePlan(
    "transmural",
    "longitudinal",
    transmural_endocardium=0.0,
    transmural_epicardium=1.0,
    helix_endocardium_degrees=60.0,
    helix_epicardium_degrees=-60.0,
    gradient_tolerance=0.0,
)
micro_candidate = micro_plan.prepare(coordinates).build()
assert bool(micro_candidate.evidence.all_successful)
microstructure = micro_candidate.commit()

frame = microstructure.material_frame
fiber = frame.fiber
sheet = frame.sheet
sheet_normal = frame.sheet_normal
physical_fiber_tensor = microstructure.fiber_structure_tensor
```

A zero transmural gradient or a longitudinal gradient parallel to it is a genuine degeneracy. The implementation does not add an epsilon, perturb a vector, or select an arbitrary axis. It marks the cell invalid, writes NaNs into its frame, reports the raw gradient norms, and refuses commit. The evidence also reports cellwise orthonormality error, frame determinant, tensor symmetry error, coordinate-range validity, and finite status.

## Closed oriented chamber surfaces

A chamber surface plan accepts a fixed triangle topology and reference coordinates. Preparation:

1. rejects duplicate or degenerate triangles;
2. requires exactly two incident faces at every edge;
3. requires one connected orientable component;
4. propagates consistent local orientation; and
5. selects the global orientation with positive enclosed reference volume.

Input face order and winding are canonicalized, so stable plan and surface IDs do not depend on those incidental choices. `from_boundary_roles()` is available when an explicitly selected set of role facets is already closed. An open endocardial wall needs a dedicated cavity cap; do not pass a wall-base annulus that does not close exactly one cavity.

```python
surface_plan = anatomy.ChamberSurfacePlan(
    "lv-cavity",
    cavity_vertices_mm,
    cavity_triangles,
    vertex_global_ids=cavity_vertex_ids,
)
surface = surface_plan.prepare()

volume_candidate = surface.evaluate(current_cavity_vertices_mm)
assert bool(volume_candidate.evidence.successful)
volume = volume_candidate.commit()

volume_mm3 = volume.volume
dvolume_dx = volume.coordinate_derivative
```

The signed volume is evaluated with a translated reference origin to limit cancellation. The analytic derivative is accumulated for every input vertex and remains differentiable through JAX. Evidence includes per-face signed contributions, minimum double area, oriented-area closure residual, explicit translation-volume error, the norm of the derivative under rigid translation, finite status, and positive-orientation status. A reflected or collapsed chamber produces an unsuccessful candidate and cannot be committed.

The orientation selected during `prepare()` is held fixed during `evaluate()`. This is intentional: topology and discrete orientation are preparation-time decisions, while coordinate motion, volume, and its derivative form the fixed-topology differentiation boundary.

## Qualification and performance

Run the deterministic manufactured cases with:

```text
python tools/cardiovascular_geometry_qualification.py --subdivisions 2
```

The first case is a tetrahedral affine LV-wall slab. It proves linear harmonic reproduction, exact cell gradients, the analytic helix frame, physical fiber-tensor sign invariance, and rejection of parallel-gradient degeneracy. The second case is a closed tetrahedral LV cavity. It proves closure, outward orientation, analytic volume, agreement of the supplied derivative with automatic differentiation, rigid-translation invariance, and rejection of reflection.

Geometry preparation and evaluation costs can be measured independently with:

```text
python benchmarks/cardiovascular_geometry.py --subdivisions 2 4 --repeats 5
```

The benchmark reports role qualification, FEM preparation, coordinate solve, microstructure construction, and chamber volume-plus-derivative timings without weakening any evidence checks.
