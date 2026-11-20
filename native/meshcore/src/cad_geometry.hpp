// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// L0 geometry of a CAD model: binary64 values and derivative jets of every
// surface, 3D curve and p-curve of a phx_mc_cad_model. Evaluation is a
// floating-point proposal; certificates bound it separately.
//
// Every source entity is a node of one geometry graph (children precede
// parents). Straight-line kinds are generated template programs executed by
// the point executors of interval_program.hpp with the entity's coefficients
// as seeds; splines (B-spline, T-spline), placements, offsets, swept and
// mapped kinds, and implicit surface-surface intersections are native nodes.
// A node expands itself in its own parameters as a truncated Taylor
// polynomial (cad_jet.hpp); a parent substitutes its inputs, so derivatives of
// any composition follow by polynomial composition.
#pragma once

#include <cstdint>
#include <vector>

#include "cad_jet.hpp"
#include "interval_program.hpp"
#include "phydrax_meshcore.h"

namespace phx::mc::cad {

// One generated straight-line program (cad_geometry_programs.inc).
struct ProgramData {
  const int64_t* instructions;
  int64_t instruction_count;
  const int64_t* indices;
  int64_t index_count;
  const double* parameters;
  int64_t parameter_count;
  const double* seeds;
  int64_t workspace_size;
  const int64_t* coefficients;  // workspace entry of every coefficient, in order
  const int64_t* outputs;
  int64_t output_count;
};

// A straight-line kind: the value program; the tangent program (value and
// first partials, input after input) whose jets carry third derivatives; and
// the Hessian program (value, first partials, then second partials over pairs
// a <= b) whose jets carry fourth derivatives.
struct TemplateDefinition {
  const char* name;
  int32_t input_dimension;
  int32_t output_dimension;
  int32_t coefficient_count;
  ProgramData value;
  ProgramData tangent;
  ProgramData hessian;
};

struct Node;
// Caller-owned scratch of one evaluating thread (program workspace and jets).
struct EvaluationWorkspace {
  PointWorkspace program;
  std::vector<double> jets;
  const Node* span_node = nullptr;
  int64_t span_chart = -1;
};

// Exact-source coefficient enclosures use the model's node_reals layout.
// The point path continues to use node_reals itself. Programs and temporary
// endpoints are retained by the calling thread across interval evaluations.
struct IntervalEvaluationWorkspace {
  const double* coefficient_lower = nullptr;
  const double* coefficient_upper = nullptr;
  int64_t coefficient_count = 0;
  const ProgramData* bound_program[3] = {};
  PreparedIntervalProgram program[3];
  std::vector<double> lower, upper;
  std::vector<double> taylor_lower, taylor_upper;
  bool direct_template = false;
};

// Closed-form surface evaluators (decision D3) that stand in for a node's
// generic expansion: the five quadric templates and the revolution of a line
// profile. They evaluate the source's own formula, so they agree with the
// template programs within a few ulp of the coordinate scale.
enum class ClosedForm : int32_t { kNone, kPlane, kCylinder, kCone, kSphere, kTorus, kLineRevolution };

// kPrograms keeps every node on its generic route (template programs and
// Taylor composition): the reference of the closed forms in tests and
// measurements.
enum class Evaluators { kClosedForms, kPrograms };

struct Node {
  int32_t kind = -1;
  int32_t inputs = 0;
  int32_t outputs = 0;
  const double* reals = nullptr;
  int64_t real_count = 0;
  const int64_t* integers = nullptr;
  int64_t integer_count = 0;
  // Template nodes: the definition and its seeded workspace templates.
  const TemplateDefinition* definition = nullptr;
  std::vector<double> value_seeds, tangent_seeds, hessian_seeds;
  // T-spline nodes: sorted distinct piece breaks and, per grid cell, the
  // first piece covering it.
  std::vector<double> u_breaks, v_breaks;
  std::vector<int32_t> cell_pieces;
  // Closed-form nodes: the kind and its frame (see expand_closed).
  ClosedForm closed = ClosedForm::kNone;
  double closed_data[24] = {};
};

class CadGeometry {
 public:
  // Validates the model's geometry graph and entity tables; nothing is kept
  // but views into the caller's arrays, which must outlive this object.
  int32_t bind(const phx_mc_cad_model& model, Evaluators evaluators = Evaluators::kClosedForms);
  const phx_mc_cad_model& model() const noexcept { return model_; }

  int64_t surface_count() const noexcept { return model_.surface_count; }
  int64_t curve_count() const noexcept { return model_.curve_count; }
  int64_t use_count() const noexcept { return model_.use_count; }
  int32_t surface_node(int64_t face) const noexcept { return model_.surface_nodes[face]; }
  int32_t curve_node(int64_t curve) const noexcept { return model_.curve_nodes[curve]; }
  int32_t pcurve_node(int64_t use) const noexcept { return model_.use_pcurve_nodes[use]; }
  const Node& node(int32_t index) const noexcept { return nodes_[static_cast<std::size_t>(index)]; }

  // Append representation breakpoints mapped to s in (0,1), where the
  // node parameter is first + s * (last - first). All distinct spline knots
  // are retained, regardless of multiplicity: each starts a polynomial span.
  // degree is raised to the largest encountered spline degree. Endpoints are
  // the topological vertices and are supplied separately by the caller.
  // Crossings of adjacent surface knot or piece lines are not enumerated:
  // they only change the sampled face-curvature size integrand. In particular,
  // a C0 surface crease is the face stage's responsibility.
  void curve_breakpoints(int32_t index, double first, double last,
                         std::vector<double>& points, int& degree) const;

  // Evaluate the exact parameter on the chart selected by span_interior,
  // which must be strictly inside a representation span; a breakpoint returns
  // false rather than choosing either side. Curve composites are followed.
  // Surface knot/piece crossings are not curve representation spans.
  bool jets_in_span(int32_t node, double parameter, double span_interior, int order,
                    double* result, EvaluationWorkspace& workspace) const;

  // Derivatives of `node` at `point` (its input coordinates) through total
  // order `order` (0 to 2): for n = 0..order the derivatives
  // d^n / (du^(n-j) dv^j), j = 0..n (one per order for one input), each an
  // output vector.
  // False when the point lies outside the node's evaluation domain.
  bool jets(int32_t node, const double* point, int order, double* result,
            EvaluationWorkspace& workspace) const;

  // Outward enclosure of derivatives in the jets layout, over an input box.
  // The workspace supplies exact-source coefficient bounds for node_reals.
  // Unsupported nodes or unbounded results return PHX_MC_UNRESOLVED.
  int32_t interval_jets(int32_t node, const double* input_lower, const double* input_upper,
                        int order, double* output_lower, double* output_upper,
                        IntervalEvaluationWorkspace& workspace) const;

  // The local Taylor expansion of `node` at `point` in its own parameters.
  bool expand(int32_t node, const double* point, int order, Taylor* out,
              EvaluationWorkspace& workspace) const;
  // The node composed with input expansions (constant terms = the point).
  bool substitute(int32_t node, const Taylor* inputs, const Truncation& truncation, Taylor* out,
                  EvaluationWorkspace& workspace) const;

 private:
  bool expand_template(const Node& node, const double* point, int order, Taylor* out,
                       EvaluationWorkspace& workspace) const;
  bool expand_closed(const Node& node, const double* point, int order, Taylor* out) const;
  void bind_closed_form(Node& node) const;
  bool expand_bspline_curve(const Node& node, double t, int order, int span_override,
                            Taylor* out) const;
  bool expand_bspline_surface(const Node& node, const double* point, int order,
                              Taylor* out) const;
  bool expand_tspline_surface(const Node& node, const double* point, int order,
                              Taylor* out) const;
  bool expand_offset_surface(const Node& node, const double* point, int order, Taylor* out,
                             EvaluationWorkspace& workspace) const;
  bool expand_offset_curve(const Node& node, double t, int order, Taylor* out,
                           EvaluationWorkspace& workspace) const;
  bool expand_intersection(const Node& node, double t, int order, Taylor* out,
                           EvaluationWorkspace& workspace) const;
  int32_t validate_node(int32_t index);

  phx_mc_cad_model model_{};
  std::vector<Node> nodes_;
};

// Derived surface quantities from a second-order surface jet (value, Su, Sv,
// Suu, Suv, Svv): first fundamental form [E F; F G], unit normal Su x Sv
// normalized, second fundamental form [L M; M N] for that normal, and
// principal curvatures k1 <= k2 (eigenvalues of I^-1 II). False when Su x Sv
// vanishes.
struct SurfaceDifferential {
  double first_form[3];
  double normal[3];
  double second_form[3];
  double curvatures[2];
};
bool surface_differential(const double* jet, SurfaceDifferential& out) noexcept;

// Sizing metric of a face point in parameter coordinates (uu, uv, vv): the
// isotropic 1/h^2 intersected with the fidelity tensor |II| / (4 theta tol).
// A chord of length L in direction d sags kappa_n(d) L^2 / 8; the chart
// kernel accepts circumradius <= 1/sqrt(2) metric units, so edges reach
// sqrt(2) units, and sizes sqrt(4 theta tol / |k_i|) along the principal
// directions keep every edge's sag within theta * tol. Eigenvalues relative
// to I are lambda_i = max(1/h^2, |k_i| / (4 theta tol)), i.e.
// M = lambda_1 I + s (II - k_1 I) with the spectral projector of k_2 and
// s = (lambda_2 - lambda_1) / (k_2 - k_1), bounded by the curvature
// coefficient and so stable as k_1 -> k_2. A non-binding tolerance gives
// exactly I / h^2. `lambdas` (optional) receives lambda_1, lambda_2.
void fidelity_metric(const SurfaceDifferential& differential, double h, double theta,
                     double tolerance, double metric[3], double* lambdas) noexcept;

// Row of the named straight-line kind in the generated template table, or -1.
int32_t template_index(const char* name) noexcept;

// Speed |C'| and curvature |C' x C''| / |C'|^3 of a second-order 3D curve jet.
bool curve_differential(const double* jet, double* speed, double* curvature) noexcept;

}  // namespace phx::mc::cad
