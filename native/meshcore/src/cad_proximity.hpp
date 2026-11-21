// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Proximity size field of the CAD solid route (ProximitySizeControl).
//
// A control scopes two disjoint face sets, source and target. Its size at a
// point p of a scoped face is the local gap over elements_per_gap, clamped to
// [minimum_size, maximum_size]; the local gap is the sum of p's distances to
// the source and to the target faces (Gmsh's and the sampled route's
// definition), so it is the distance to the other scope's faces. Off the
// scoped faces a control bounds nothing.
//
// The other scope's faces are taken as a proxy: a triangulation of them
// (the face stage's own output at the request), each triangle with a bound
// on its deviation from its face. The true distance from p is then at least
// the proxy distance minus that bound, and the field uses that lower bound,
// so its size never exceeds the control's size at the true gap. The bound
// (proxy_deviation) is the sag of the triangle's circumcircle (radius R)
// on the osculating sphere of the largest principal curvature kappa at its
// corners, kappa R^2 / (1 + sqrt(1 - kappa^2 R^2)), capped by the face's
// tolerance (the face stage's own deviation bound): zero on a plane, exact
// on a sphere. With opposite_normals_only the gap counts only when the
// nearest proxy triangle of the other scope faces p (outward normals with a
// negative dot product, p's normal from its own nearest proxy triangle);
// otherwise the gap is infinite and only maximum_size bounds the size.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace phx::mc::cad {

struct ProximityControl {
  std::vector<int32_t> source_faces;
  std::vector<int32_t> target_faces;
  int32_t elements_per_gap = 1;
  double minimum_size = 0.0;   // 0: none
  double maximum_size = 0.0;   // 0: none
  bool opposite_normals_only = true;
};

// The deviation bound of a proxy triangle with circumradius R on a face
// whose largest principal curvature at the triangle's corners is kappa, at
// most the face's tolerance (see above).
double proxy_deviation(double circumradius, double kappa, double tolerance);

class ProximityField {
 public:
  // Proxy triangles (outward oriented) with their face rows and deviation
  // bounds. Returns a PHX_MC status: INVALID_INPUT for a face row out of
  // range, overlapping scopes or a scope without proxy triangles.
  int32_t build(const std::vector<ProximityControl>& controls,
                const std::vector<std::array<double, 3>>& points,
                const std::vector<std::array<int32_t, 3>>& triangles,
                const std::vector<int32_t>& triangle_faces,
                const std::vector<double>& deviations, int64_t face_count);
  bool empty() const noexcept { return controls_.empty(); }
  // The smallest control size at point p of face `face`, +inf when no
  // control scopes the face. Zero when a gap bound reaches zero and the
  // control has no minimum_size (the caller refuses it).
  double size(int32_t face, const double* p) const;

 private:
  struct Node {
    double low[3], high[3];
    int32_t left = -1, right = -1, begin = 0, end = 0;
  };
  struct Set {
    std::vector<std::array<double, 9>> triangles;
    std::vector<std::array<double, 3>> normals;
    std::vector<double> tolerances;
    std::vector<int32_t> order;
    std::vector<Node> nodes;
    int32_t build_node(int32_t begin, int32_t end);
    // Nearest triangle: its row and the distance (best stays when none is nearer).
    void nearest(int32_t node, const double* p, double& best, int32_t& row) const;
  };
  struct Entry {
    ProximityControl control;
    Set source, target;
  };
  std::vector<Entry> controls_;
  // Per face: (control, side) rows; side 0 source, 1 target.
  std::vector<std::vector<std::array<int32_t, 2>>> face_rows_;
};

}  // namespace phx::mc::cad
