# Liquid crystals

PHYDRAX evolves nematic order in a compact symmetric-traceless Q-tensor basis. A
three-dimensional orientational tensor has five independent state components; a
two-dimensional tensor has two. Spatial and orientational dimension are independent,
so a two-dimensional domain may retain a three-dimensional Q tensor.

## Thermodynamic closure

`LandauDeGennesClosure` evaluates bulk, one-constant elastic, chiral, and electric
energy; the symmetric-traceless molecular field; distortion and electric stress;
scalar order; and structural residuals. Symmetry and zero trace follow from the basis
rather than runtime projection or eigenvalue clipping.

`LandauDeGennesParameters` exposes general bulk coefficients, positive elastic
constant, chiral wave number, and dielectric anisotropy. `NematicTensorBasis` provides
orthonormal encode/decode/project operations and a stable identity for checkpoints and
coupling.

## Relaxation, flow, and anchoring

`PreparedNematicDynamics` binds the closure to an existing prepared finite-difference
discretization. It computes component-wise gradients and Laplacians, molecular
relaxation, conservative velocity advection, and Beris-Edwards flow alignment. Passive
zero-flow steps additionally require non-increasing free energy.

`PreparedNematicSemiImplicitStepPlan` treats one-constant elastic relaxation through
the certified shifted periodic FD Laplacian solve while keeping local bulk and
anchoring terms explicit.

`NematicAnchoringPlan` supports fixed, homeotropic, and planar-degenerate surface
energy. Boundary masks and normals are prepared data; inactive points contribute no
energy or molecular field.

`MACNematicCouplingPlan` differentiates passive plus active stress to a cell body force.
Passive and active power remain separate, so active systems never inherit a passive
free-energy-dissipation claim.

## Electrolytic nematics

`ElectrolyticNematicClosure` composes nematic, ionic, anisotropic dielectric, and
ion-order coupling energy. It returns one Q molecular field, one ionic electrochemical
potential, one charge density, one permittivity tensor, and one total stress. The
closure rejects a non-positive dielectric tensor margin rather than clipping order.
