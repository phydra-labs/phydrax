// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// A closed triangle mesh as the source of a surface remesh.
//
// MeshSource indexes the source triangles in a uniform bucket grid and answers
// nearest-point queries (closest point, its triangle and barycentric
// coordinates). Normals at a point interpolate angle-weighted vertex normals;
// curvatures interpolate per-vertex estimates of the two principal
// curvatures, the largest and smallest normal curvature along the vertex's
// ring edges (Euler's theorem). MeshSourceOracle presents it
// to the adapter (surface_adapt.hpp): construction only, nothing proved.
//
// triangle_bound states an upper bound on the distance from every point of a
// closed triangle T to the source. Points are given planar coordinates along
// a unit direction d (T's own normal, or the source's interpolated normal
// nearest T, for a T rougher than the source); each triangle then maps
// affinely onto the planar triangle of its corners' coordinates (exact
// binary64 values). The patch is the source triangles reached from the
// nearest one across edges whose planar triangles meet T's; its boundary
// therefore misses T's planar triangle, and when it winds once around it
// (exact orient2d) every planar point y of T's triangle is the image of a
// patch point: x = X(y) in T and Y(y) in a patch triangle t give
// d(x, source) <= |X(y) - Y(y)|, convex over the overlap of the two planar
// triangles, hence at most its value at the overlap's vertices
// (overlap_gap_bound, with certified rounding pads). The bound is O(h^2) for
// a triangle of size h over a smooth source. MeshSourceCertifier presents
// it to the certify driver (surface_certify.hpp).
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "surface_adapt.hpp"
#include "surface_certify.hpp"

namespace phx::mc::remesh {

enum class Projection : uint8_t { kOwnNormal, kSourceNormal };

using Planar = std::array<double, 2>;

// An upper bound on |X(y) - Y(y)| over the overlap of two planar triangles,
// X and Y the affine maps sending each planar triangle's corners to its
// spatial corners: the largest value at the overlap's vertices (corners
// inside the other triangle and proper edge crossings, decided by the exact
// orient2d), each crossing evaluated at a point within a certified offset of
// the exact one (its parameter enclosed from the orientation error bounds),
// plus the maps' slopes times that offset and a first-order bound on the
// rounding of the evaluation, rounded up by a relative 2^-40. 0 when `other`
// is degenerate (it overlaps no area) or the overlap is empty; +inf when
// `own` is degenerate. Coordinates must satisfy coordinate_in_domain.
double overlap_gap_bound(const std::array<Planar, 3>& own, const std::array<Point, 3>& own_space,
                         const std::array<Planar, 3>& other,
                         const std::array<Point, 3>& other_space);

class MeshSource {
 public:
  // PHX_MC_OK, PHX_MC_INVALID_INPUT for a surface that is not a closed oriented
  // 2-manifold or has a degenerate triangle, PHX_MC_NONFINITE_INPUT.
  int32_t bind(std::vector<Point> points, std::vector<Triangle> triangles);

  struct Nearest {
    int64_t triangle = -1;
    Point point{0, 0, 0};
    std::array<double, 3> barycentric{0, 0, 0};
    double distance = 0;
  };
  // The closest point of the source to `position`; triangle -1 when the
  // source is empty.
  Nearest nearest(const Point& position) const;
  // Interpolated unit normal, and the (largest, smallest) principal
  // curvature estimates, at a nearest point.
  Point normal(const Nearest& at) const;
  std::array<double, 2> curvatures(const Nearest& at) const;
  // Upper bound on the distance from the closed triangle (a, b, c) to the
  // source, projecting along `projection`; +inf when the triangle projects
  // degenerately, its patch exceeds 65,536 triangles, or the patch boundary
  // does not wind once around it.
  double triangle_bound(const Point& a, const Point& b, const Point& c,
                        Projection projection) const;

  const std::vector<Point>& points() const noexcept { return points_; }
  const std::vector<Triangle>& triangles() const noexcept { return triangles_; }

 private:
  std::array<int64_t, 3> cell_of(const Point& p) const;
  int64_t flat(const std::array<int64_t, 3>& c) const {
    return (c[0] * shape_[1] + c[1]) * shape_[2] + c[2];
  }
  std::vector<Point> points_;
  std::vector<Triangle> triangles_;
  std::vector<std::array<int64_t, 3>> neighbors_;  // across edge (f[k], f[k + 1])
  std::vector<Point> face_normals_, vertex_normals_;
  std::vector<std::array<double, 4>> balls_;  // per triangle: centroid, radius
  std::vector<std::array<double, 2>> vertex_curvatures_;  // (largest, smallest)
  double scale_ = 0;  // largest coordinate magnitude, for rounding margins
  Point origin_{0, 0, 0};
  double cell_ = 1;
  std::array<int64_t, 3> shape_{1, 1, 1};
  std::vector<int64_t> start_, items_;
  // Query scratch (single-threaded): triangle stamps for nearest, patch
  // membership and patch tests; vertex stamps with projected (u, v, height).
  mutable std::vector<int64_t> visited_, included_, tested_, vertex_stamps_;
  mutable std::vector<std::array<double, 3>> projected_;
  mutable std::vector<int64_t> patch_;
  mutable int64_t stamp_ = 0;
};

class MeshSourceOracle final : public SurfaceOracle {
 public:
  explicit MeshSourceOracle(const MeshSource& source) : source_(source) {}
  bool project(const Point& position, SurfacePoint& out) const override;
  void project_all(const std::vector<Point>& positions, std::vector<SurfacePoint>& out,
                   std::vector<uint8_t>& found) const override;
  void distances(const std::vector<Point>& positions, std::vector<double>& out) const override;

 private:
  const MeshSource& source_;
};

// States triangle_bound when it is within `tolerance`, +inf otherwise.
class MeshSourceCertifier final : public TriangleCertifier {
 public:
  MeshSourceCertifier(const MeshSource& source, double tolerance)
      : source_(source), tolerance_(tolerance) {}
  double bound(const Point& a, const Point& b, const Point& c) const override {
    const double bound = source_.triangle_bound(a, b, c, Projection::kOwnNormal);
    return bound <= tolerance_ ? bound : std::numeric_limits<double>::infinity();
  }

 private:
  const MeshSource& source_;
  double tolerance_;
};

}  // namespace phx::mc::remesh
