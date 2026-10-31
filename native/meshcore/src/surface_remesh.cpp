// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Source-agnostic certified surface remeshing; contract in surface_remesh.hpp.
#include "surface_remesh.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>

#include "geometry3.hpp"
#include "intersections.hpp"
#include "phydrax_meshcore.h"

namespace phx::mc::remesh {
namespace {
using V = Point;
using F = Triangle;
using phx::mc::geometry::cross;
using phx::mc::geometry::difference;
using phx::mc::geometry::dot;
using phx::mc::geometry::next_up;
using phx::mc::geometry::point_in_domain;

constexpr uint32_t kScoreLevels = 1u << 20;

V add(const V& a, const V& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V mul(const V& a, double t) { return {a[0] * t, a[1] * t, a[2] * t}; }
double length(const V& a) { return std::hypot(a[0], a[1], a[2]); }

uint64_t cell_key(int64_t i, int64_t j, int64_t k) {
  constexpr int64_t bias = int64_t{1} << 20;
  return (static_cast<uint64_t>(i + bias) << 42) ^ (static_cast<uint64_t>(j + bias) << 21) ^
         static_cast<uint64_t>(k + bias);
}

class Remesher {
 public:
  Remesher(const TriangleAdmission& admission, const RemeshTargets& targets)
      : predicate(admission), targets(targets), h(targets.edge_length) {}

  const TriangleAdmission& predicate;
  const RemeshTargets& targets;
  double h = 0;
  std::vector<V> points;
  std::vector<VertexEvidence> evidence;  // certified admission evidence per vertex
  std::vector<F> triangles;
  std::vector<uint8_t> alive, stuck;
  std::vector<uint32_t> scores;
  std::vector<std::vector<int64_t>> star;
  std::unordered_map<uint64_t, std::vector<int64_t>> buckets;
  std::set<std::pair<uint32_t, int64_t>> queue;
  int64_t splits = 0, collapses = 0, flips = 0, relocations = 0;

  bool admitted(const F& t) const {
    const size_t a = static_cast<size_t>(t[0]), b = static_cast<size_t>(t[1]),
                 c = static_cast<size_t>(t[2]);
    return predicate.admits(points[a], points[b], points[c],
                            {evidence[a], evidence[b], evidence[c]});
  }
  // Shape and size factor of a triangle in [0, 1].
  double shape_size(const V& a, const V& b, const V& c) const {
    std::array<double, 3> lengths{length(difference(b, c)), length(difference(c, a)),
                                  length(difference(a, b))};
    std::sort(lengths.begin(), lengths.end());
    const double twice_area = length(cross(difference(b, a), difference(c, a)));
    if (!(twice_area > 0) || !(lengths[0] > 0)) return 0;
    const double sine = std::min(1.0, twice_area / (lengths[1] * lengths[2]));
    const double minimum_angle = std::asin(sine) * 180.0 / std::acos(-1.0);
    const double shape = std::min(1.0, minimum_angle / targets.minimum_angle_degrees);
    const double size = std::min(1.0, (4.0 / 3.0) * h / lengths[2]);
    return shape * size;
  }
  V normal(const F& t) const {
    const V& a = points[static_cast<size_t>(t[0])];
    return cross(difference(points[static_cast<size_t>(t[1])], a),
                 difference(points[static_cast<size_t>(t[2])], a));
  }
  // The triangle holding directed edge q -> p (the neighbor across p -> q) in
  // the state where `removed` (sorted IDs) are dead and `created` are live.
  std::optional<F> across(int64_t p, int64_t q, const std::vector<F>& created,
                          const std::vector<int64_t>& removed) {
    const auto holds = [&](const F& t) {
      for (int k = 0; k < 3; ++k)
        if (t[k] == q && t[(k + 1) % 3] == p) return true;
      return false;
    };
    for (const F& t : created)
      if (holds(t)) return t;
    for (int64_t id : live_star(p)) {
      if (std::binary_search(removed.begin(), removed.end(), id)) continue;
      if (holds(triangles[static_cast<size_t>(id)])) return triangles[static_cast<size_t>(id)];
    }
    return std::nullopt;
  }
  // Quantized score of `t` in that state: shape and size times the fold
  // factor, the minimum over its edges of min(1, (180 - turn) / (180 -
  // maximum turn)), turn being the angle between the unit normals of `t` and
  // its edge neighbor (a degenerate normal on either side leaves the edge to
  // the degenerate triangle's own score).
  uint32_t score(const F& t, const std::vector<F>& created, const std::vector<int64_t>& removed) {
    const double value = shape_size(points[static_cast<size_t>(t[0])],
                                    points[static_cast<size_t>(t[1])],
                                    points[static_cast<size_t>(t[2])]);
    if (!(value > 0)) return 0;
    const V own = normal(t);
    double fold = 1;
    for (int k = 0; k < 3; ++k) {
      const auto neighbor = across(t[k], t[(k + 1) % 3], created, removed);
      if (!neighbor) continue;
      const V other = normal(*neighbor);
      if (!(length(other) > 0)) continue;
      const double turn = std::atan2(length(cross(own, other)), dot(own.data(), other.data())) *
                          180.0 / std::acos(-1.0);
      fold = std::min(fold, (180.0 - turn) / (180.0 - targets.maximum_normal_turn_degrees));
    }
    return static_cast<uint32_t>(std::clamp(std::floor(value * std::max(fold, 0.0) * kScoreLevels),
                                            0.0, static_cast<double>(kScoreLevels)));
  }
  // Rescores a live triangle in the current state and requeues it.
  void rescore(int64_t id) {
    const auto slot = static_cast<size_t>(id);
    queue.erase({scores[slot], id});
    scores[slot] = score(triangles[slot], {}, {});
    if (scores[slot] < kScoreLevels && !stuck[slot]) queue.emplace(scores[slot], id);
  }
  std::array<int64_t, 6> bucket_range(const F& t) const {
    V lo = points[static_cast<size_t>(t[0])], hi = lo;
    for (int64_t v : t)
      for (int d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], points[static_cast<size_t>(v)][d]);
        hi[d] = std::max(hi[d], points[static_cast<size_t>(v)][d]);
      }
    std::array<int64_t, 6> range{};
    for (int d = 0; d < 3; ++d) {
      range[d] = static_cast<int64_t>(std::floor(lo[d] / h)) - 1;
      range[3 + d] = static_cast<int64_t>(std::floor(hi[d] / h)) + 1;
    }
    return range;
  }
  // Registers a triangle in the buckets and stars; its score depends on its
  // edge neighbors, so callers rescore once the edit's triangles are all in.
  void insert(int64_t id) {
    const F& t = triangles[static_cast<size_t>(id)];
    const auto range = bucket_range(t);
    for (int64_t i = range[0]; i <= range[3]; ++i)
      for (int64_t j = range[1]; j <= range[4]; ++j)
        for (int64_t k = range[2]; k <= range[5]; ++k) buckets[cell_key(i, j, k)].push_back(id);
    for (int64_t v : t) star[static_cast<size_t>(v)].push_back(id);
  }
  int64_t add_triangle(const F& t) {
    const int64_t id = static_cast<int64_t>(triangles.size());
    triangles.push_back(t);
    alive.push_back(1);
    stuck.push_back(0);
    scores.push_back(0);
    insert(id);
    return id;
  }
  void kill(int64_t id) {
    alive[static_cast<size_t>(id)] = 0;
    queue.erase({scores[static_cast<size_t>(id)], id});
  }
  std::vector<int64_t> live_star(int64_t v) {
    auto& incident = star[static_cast<size_t>(v)];
    incident.erase(std::remove_if(incident.begin(), incident.end(),
                                  [&](int64_t id) { return !alive[static_cast<size_t>(id)]; }),
                   incident.end());
    return incident;
  }
  std::vector<int64_t> edge_triangles(int64_t a, int64_t b) {
    std::vector<int64_t> found;
    for (int64_t id : live_star(a)) {
      const F& t = triangles[static_cast<size_t>(id)];
      if (t[0] == b || t[1] == b || t[2] == b) found.push_back(id);
    }
    return found;
  }
  std::vector<int64_t> neighbors(int64_t v) {
    std::vector<int64_t> result;
    for (int64_t id : live_star(v))
      for (int64_t w : triangles[static_cast<size_t>(id)])
        if (w != v) result.push_back(w);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }
  V project(const V& x) const { return predicate.project(x); }


  struct Edit {
    std::vector<int64_t> removed;
    std::vector<std::pair<int64_t, V>> moved;
    bool adds_vertex = false;
    V added{};
    int64_t dropped = -1;
    std::vector<F> created;
  };

  static std::array<int64_t, 3> sorted(const F& t) {
    std::array<int64_t, 3> key{t[0], t[1], t[2]};
    std::sort(key.begin(), key.end());
    return key;
  }

  // Stages the edit's positions, checks every admission condition, and either
  // applies it or restores the previous positions. `kImprove` edits must raise
  // every created score above the replaced patch's worst score; `kPreserve`
  // edits (regularization) must not lower it.
  enum class Admission { kImprove, kPreserve };
  bool apply_if_admissible(const Edit& edit, Admission admission = Admission::kImprove) {
    if (edit.created.empty() || edit.removed.empty()) return false;
    uint32_t removed_minimum = kScoreLevels;
    V reference{0, 0, 0};
    std::vector<std::array<int64_t, 3>> removed_keys;
    for (int64_t id : edit.removed) {
      if (!alive[static_cast<size_t>(id)]) return false;
      removed_minimum = std::min(removed_minimum, scores[static_cast<size_t>(id)]);
      const F& t = triangles[static_cast<size_t>(id)];
      reference = add(reference, cross(difference(points[static_cast<size_t>(t[1])],
                                                  points[static_cast<size_t>(t[0])]),
                                       difference(points[static_cast<size_t>(t[2])],
                                                  points[static_cast<size_t>(t[0])])));
      removed_keys.push_back(sorted(t));
    }
    std::sort(removed_keys.begin(), removed_keys.end());
    std::vector<int64_t> removed_ids = edit.removed;
    std::sort(removed_ids.begin(), removed_ids.end());
    std::vector<std::pair<int64_t, V>> saved;
    std::vector<VertexEvidence> saved_evidence;
    for (const auto& [vertex, position] : edit.moved) {
      saved.emplace_back(vertex, points[static_cast<size_t>(vertex)]);
      saved_evidence.push_back(evidence[static_cast<size_t>(vertex)]);
    }
    bool pushed = false;
    auto restore = [&]() {
      for (size_t i = 0; i < saved.size(); ++i) {
        points[static_cast<size_t>(saved[i].first)] = saved[i].second;
        evidence[static_cast<size_t>(saved[i].first)] = saved_evidence[i];
      }
      if (pushed) {
        points.pop_back();
        evidence.pop_back();
        star.pop_back();
      }
      return false;
    };
    for (const auto& [vertex, position] : edit.moved) {
      if (!point_in_domain(position)) return restore();
      points[static_cast<size_t>(vertex)] = position;
      evidence[static_cast<size_t>(vertex)] = predicate.evaluate(position);
    }
    if (edit.adds_vertex) {
      points.push_back(edit.added);
      evidence.push_back(predicate.evaluate(edit.added));
      star.emplace_back();
      pushed = true;
      if (!point_in_domain(edit.added)) return restore();
    }
    std::vector<std::array<int64_t, 3>> created_keys;
    for (const F& t : edit.created) {
      if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) return restore();
      const V& a = points[static_cast<size_t>(t[0])];
      const V& b = points[static_cast<size_t>(t[1])];
      const V& c = points[static_cast<size_t>(t[2])];
      if (phx::mc::geometry::collinear(a, b, c, 0)) return restore();
      const uint32_t created_score = score(t, edit.created, removed_ids);
      if (admission == Admission::kImprove ? created_score <= removed_minimum
                                           : created_score < removed_minimum)
        return restore();
      const V normal = cross(difference(b, a), difference(c, a));
      if (!(dot(normal.data(), reference.data()) > 0)) return restore();
      if (!admitted(t)) return restore();
      created_keys.push_back(sorted(t));
    }
    std::sort(created_keys.begin(), created_keys.end());
    if (std::adjacent_find(created_keys.begin(), created_keys.end()) != created_keys.end())
      return restore();
    // A created face may not duplicate a surviving face.
    for (const F& t : edit.created)
      for (int64_t id : live_star(t[0])) {
        if (std::binary_search(removed_ids.begin(), removed_ids.end(), id)) continue;
        if (sorted(triangles[static_cast<size_t>(id)]) == sorted(t)) return restore();
      }
    // Surviving edge neighbors of the created patch change only their fold
    // factor. One that does not keep its score is held to the created
    // triangles' bound, so in the multiset argument it is a replaced element
    // whose successor still exceeds the patch's worst score.
    std::vector<int64_t> ring;
    for (const F& t : edit.created)
      for (int k = 0; k < 3; ++k) {
        const int64_t p = t[k], q = t[(k + 1) % 3];
        for (int64_t id : live_star(p)) {
          if (std::binary_search(removed_ids.begin(), removed_ids.end(), id)) continue;
          const F& u = triangles[static_cast<size_t>(id)];
          for (int j = 0; j < 3; ++j)
            if (u[j] == q && u[(j + 1) % 3] == p) ring.push_back(id);
        }
      }
    std::sort(ring.begin(), ring.end());
    ring.erase(std::unique(ring.begin(), ring.end()), ring.end());
    for (int64_t id : ring) {
      const uint32_t ring_score =
          score(triangles[static_cast<size_t>(id)], edit.created, removed_ids);
      if (ring_score >= scores[static_cast<size_t>(id)]) continue;
      if (admission == Admission::kImprove ? ring_score <= removed_minimum
                                           : ring_score < removed_minimum)
        return restore();
    }
    auto legal = [](const phx::mc::Contact& contact) {
      return contact.kind == PHX_MC_DISJOINT || contact.kind == PHX_MC_SHARED_VERTEX ||
             contact.kind == PHX_MC_SHARED_EDGE;
    };
    std::vector<int64_t> candidates;
    for (size_t first = 0; first < edit.created.size(); ++first) {
      const F& t = edit.created[first];
      const double* t_points[3] = {points[static_cast<size_t>(t[0])].data(),
                                   points[static_cast<size_t>(t[1])].data(),
                                   points[static_cast<size_t>(t[2])].data()};
      for (size_t second = first + 1; second < edit.created.size(); ++second) {
        const F& u = edit.created[second];
        const double* u_points[3] = {points[static_cast<size_t>(u[0])].data(),
                                     points[static_cast<size_t>(u[1])].data(),
                                     points[static_cast<size_t>(u[2])].data()};
        phx::mc::Contact contact;
        if (phx::mc::intersect_triangles(t_points, u_points, t.data(), u.data(), contact) !=
                PHX_MC_OK ||
            !legal(contact))
          return restore();
      }
      candidates.clear();
      const auto range = bucket_range(t);
      for (int64_t i = range[0]; i <= range[3]; ++i)
        for (int64_t j = range[1]; j <= range[4]; ++j)
          for (int64_t k = range[2]; k <= range[5]; ++k) {
            const auto found = buckets.find(cell_key(i, j, k));
            if (found == buckets.end()) continue;
            for (int64_t id : found->second)
              if (alive[static_cast<size_t>(id)] &&
                  !std::binary_search(removed_ids.begin(), removed_ids.end(), id))
                candidates.push_back(id);
          }
      std::sort(candidates.begin(), candidates.end());
      candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
      for (int64_t id : candidates) {
        const F& u = triangles[static_cast<size_t>(id)];
        const double* u_points[3] = {points[static_cast<size_t>(u[0])].data(),
                                     points[static_cast<size_t>(u[1])].data(),
                                     points[static_cast<size_t>(u[2])].data()};
        phx::mc::Contact contact;
        if (phx::mc::intersect_triangles(t_points, u_points, t.data(), u.data(), contact) !=
                PHX_MC_OK ||
            !legal(contact))
          return restore();
      }
    }
    for (int64_t id : edit.removed) kill(id);
    if (edit.dropped >= 0) star[static_cast<size_t>(edit.dropped)].clear();
    std::vector<int64_t> touched;
    std::vector<int64_t> created_ids;
    for (const F& t : edit.created) {
      created_ids.push_back(add_triangle(t));
      touched.insert(touched.end(), t.begin(), t.end());
    }
    for (int64_t id : created_ids) rescore(id);
    for (int64_t id : ring) rescore(id);
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (int64_t v : touched)
      for (int64_t id : live_star(v))
        if (stuck[static_cast<size_t>(id)]) {
          stuck[static_cast<size_t>(id)] = 0;
          if (scores[static_cast<size_t>(id)] < kScoreLevels)
            queue.emplace(scores[static_cast<size_t>(id)], id);
        }
    return true;
  }

  bool split(int64_t a, int64_t b, const V& position) {
    const auto incident = edge_triangles(a, b);
    if (incident.size() != 2) return false;
    Edit edit;
    edit.removed = incident;
    edit.adds_vertex = true;
    edit.added = position;
    const int64_t m = static_cast<int64_t>(points.size());
    for (int64_t id : incident) {
      const F& t = triangles[static_cast<size_t>(id)];
      for (int i = 0; i < 3; ++i) {
        const int64_t p = t[i], q = t[(i + 1) % 3], r = t[(i + 2) % 3];
        if ((p == a && q == b) || (p == b && q == a)) {
          edit.created.push_back({p, m, r});
          edit.created.push_back({m, q, r});
        }
      }
    }
    if (!apply_if_admissible(edit)) return false;
    ++splits;
    return true;
  }

  // `kPreserve` flips are valence-driven: admitted only when they strictly
  // lower the squared deviation from valence six over the four patch vertices.
  bool flip(int64_t a, int64_t b, Admission admission = Admission::kImprove) {
    const auto incident = edge_triangles(a, b);
    if (incident.size() != 2) return false;
    int64_t c = -1, d = -1;
    for (int64_t id : incident) {
      const F& t = triangles[static_cast<size_t>(id)];
      for (int i = 0; i < 3; ++i) {
        if (t[i] == a && t[(i + 1) % 3] == b) c = t[(i + 2) % 3];
        if (t[i] == b && t[(i + 1) % 3] == a) d = t[(i + 2) % 3];
      }
    }
    if (c < 0 || d < 0 || c == d || !edge_triangles(c, d).empty()) return false;
    if (admission == Admission::kPreserve) {
      const int64_t va = static_cast<int64_t>(neighbors(a).size()),
                    vb = static_cast<int64_t>(neighbors(b).size()),
                    vc = static_cast<int64_t>(neighbors(c).size()),
                    vd = static_cast<int64_t>(neighbors(d).size());
      auto deviation = [](int64_t valence) { return (valence - 6) * (valence - 6); };
      if (va <= 3 || vb <= 3) return false;
      if (deviation(va - 1) + deviation(vb - 1) + deviation(vc + 1) + deviation(vd + 1) >=
          deviation(va) + deviation(vb) + deviation(vc) + deviation(vd))
        return false;
    }
    Edit edit;
    edit.removed = incident;
    edit.created = {{c, a, d}, {c, d, b}};
    if (!apply_if_admissible(edit, admission)) return false;
    ++flips;
    return true;
  }

  // Collapses u into v and places v at `position`.
  bool collapse(int64_t u, int64_t v, const V& position,
                Admission admission = Admission::kImprove) {
    const auto incident = edge_triangles(u, v);
    if (incident.size() != 2) return false;
    std::vector<int64_t> opposite;
    for (int64_t id : incident)
      for (int64_t w : triangles[static_cast<size_t>(id)])
        if (w != u && w != v) opposite.push_back(w);
    std::sort(opposite.begin(), opposite.end());
    const auto u_neighbors = neighbors(u), v_neighbors = neighbors(v);
    std::vector<int64_t> common;
    std::set_intersection(u_neighbors.begin(), u_neighbors.end(), v_neighbors.begin(),
                          v_neighbors.end(), std::back_inserter(common));
    if (common != opposite || opposite.size() != 2) return false;
    for (int64_t w : opposite)
      if (neighbors(w).size() <= 3) return false;
    Edit edit;
    std::vector<int64_t> patch = live_star(u);
    for (int64_t id : live_star(v)) patch.push_back(id);
    std::sort(patch.begin(), patch.end());
    patch.erase(std::unique(patch.begin(), patch.end()), patch.end());
    edit.removed = patch;
    for (int64_t id : patch) {
      F t = triangles[static_cast<size_t>(id)];
      const bool has_u = t[0] == u || t[1] == u || t[2] == u;
      const bool has_v = t[0] == v || t[1] == v || t[2] == v;
      if (has_u && has_v) continue;
      for (int64_t& w : t)
        if (w == u) w = v;
      edit.created.push_back(t);
    }
    edit.moved.emplace_back(v, position);
    edit.dropped = u;
    if (!apply_if_admissible(edit, admission)) return false;
    ++collapses;
    return true;
  }

  bool relocate(int64_t v, const V& position) {
    if (position == points[static_cast<size_t>(v)]) return false;
    Edit edit;
    edit.removed = live_star(v);
    for (int64_t id : edit.removed) edit.created.push_back(triangles[static_cast<size_t>(id)]);
    edit.moved.emplace_back(v, position);
    if (!apply_if_admissible(edit)) return false;
    ++relocations;
    return true;
  }

  V star_centroid(int64_t v) {
    V sum{0, 0, 0};
    double weight = 0;
    for (int64_t id : live_star(v)) {
      const F& t = triangles[static_cast<size_t>(id)];
      const V& a = points[static_cast<size_t>(t[0])];
      const V& b = points[static_cast<size_t>(t[1])];
      const V& c = points[static_cast<size_t>(t[2])];
      const double area = length(cross(difference(b, a), difference(c, a)));
      sum = add(sum, mul(add(add(a, b), c), area / 3));
      weight += area;
    }
    return weight > 0 ? mul(sum, 1 / weight) : points[static_cast<size_t>(v)];
  }

  bool improve(int64_t id) {
    const F t = triangles[static_cast<size_t>(id)];
    std::array<std::pair<double, int>, 3> edges;  // (length, local edge opposite corner k)
    for (int k = 0; k < 3; ++k)
      edges[k] = {length(difference(points[static_cast<size_t>(t[(k + 1) % 3])],
                                    points[static_cast<size_t>(t[(k + 2) % 3])])),
                  k};
    std::sort(edges.begin(), edges.end());
    auto endpoints = [&](int k) {
      return std::pair<int64_t, int64_t>{t[(k + 1) % 3], t[(k + 2) % 3]};
    };
    auto midpoint = [&](int64_t a, int64_t b) {
      return mul(add(points[static_cast<size_t>(a)], points[static_cast<size_t>(b)]), 0.5);
    };
    const auto [la, lb] = endpoints(edges[2].second);
    if (edges[2].first > (4.0 / 3.0) * h) {
      const V middle = midpoint(la, lb);
      if (split(la, lb, project(middle)) || split(la, lb, middle)) return true;
    }
    // The longest edge is opposite the largest angle.
    for (int rank = 2; rank >= 0; --rank) {
      const auto [a, b] = endpoints(edges[rank].second);
      if (flip(a, b)) return true;
    }
    const auto [sa, sb] = endpoints(edges[0].second);
    if (collapse(sa, sb, project(midpoint(sa, sb))) ||
        collapse(sa, sb, points[static_cast<size_t>(sb)]) ||
        collapse(sb, sa, points[static_cast<size_t>(sa)]))
      return true;
    for (int64_t v : t) {
      const V centroid = star_centroid(v);
      if (relocate(v, project(centroid)) || relocate(v, centroid) ||
          relocate(v, project(points[static_cast<size_t>(v)])))
        return true;
    }
    const V middle = midpoint(la, lb);
    return split(la, lb, project(middle)) || split(la, lb, middle);
  }

  // Worst-first quality pass; terminates because every admitted edit strictly
  // decreases the multiset of quantized scores.
  void improve_worst_first() {
    for (size_t id = 0; id < triangles.size(); ++id)
      if (alive[id] && stuck[id]) {
        stuck[id] = 0;
        if (scores[id] < kScoreLevels) queue.emplace(scores[id], static_cast<int64_t>(id));
      }
    while (!queue.empty()) {
      const int64_t id = queue.begin()->second;
      queue.erase(queue.begin());
      if (!alive[static_cast<size_t>(id)]) continue;
      if (!improve(id)) stuck[static_cast<size_t>(id)] = 1;
    }
  }

  // Short-edge collapses (below 4/5 h) and valence-driven flips under the same
  // certified admission, never lowering a patch's worst score. Every collapse
  // removes a vertex and every flip keeps the vertex count while strictly
  // lowering the integer valence energy, so the pair (vertex count, valence
  // energy) decreases lexicographically and the sweeps terminate.
  void regularize() {
    const double short_edge = 0.8 * h;
    bool changed = true;
    while (changed) {
      changed = false;
      std::vector<std::pair<int64_t, int64_t>> edges;
      for (size_t id = 0; id < triangles.size(); ++id) {
        if (!alive[id]) continue;
        const F& t = triangles[id];
        for (int k = 0; k < 3; ++k)
          edges.emplace_back(std::min(t[k], t[(k + 1) % 3]), std::max(t[k], t[(k + 1) % 3]));
      }
      std::sort(edges.begin(), edges.end());
      edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
      for (const auto& [a, b] : edges) {
        if (edge_triangles(a, b).size() != 2) continue;
        const V& pa = points[static_cast<size_t>(a)];
        const V& pb = points[static_cast<size_t>(b)];
        if (length(difference(pa, pb)) < short_edge) {
          const V middle = mul(add(pa, pb), 0.5);
          const V keep_a = pa, keep_b = pb;
          if (collapse(a, b, project(middle), Admission::kPreserve) ||
              collapse(a, b, keep_b, Admission::kPreserve) ||
              collapse(b, a, keep_a, Admission::kPreserve)) {
            changed = true;
            continue;
          }
        }
        if (flip(a, b, Admission::kPreserve)) changed = true;
      }
    }
  }

  void run(const std::vector<V>& input_points, const std::vector<F>& input_triangles) {
    points = input_points;
    evidence.resize(points.size());
    for (size_t v = 0; v < points.size(); ++v) evidence[v] = predicate.evaluate(points[v]);
    star.assign(points.size(), {});
    for (const F& t : input_triangles) add_triangle(t);
    for (size_t id = 0; id < triangles.size(); ++id) rescore(static_cast<int64_t>(id));
    improve_worst_first();
    regularize();
    improve_worst_first();
  }

  int64_t unmet() const {
    int64_t count = 0;
    for (size_t id = 0; id < triangles.size(); ++id)
      count += alive[id] && scores[id] < kScoreLevels;
    return count;
  }

  // Surviving triangles over the vertices they use, in first-use order.
  void compact(std::vector<V>& output_points, std::vector<F>& output_triangles) const {
    std::vector<int64_t> remap(points.size(), -1);
    output_points.clear();
    output_triangles.clear();
    for (size_t id = 0; id < triangles.size(); ++id) {
      if (!alive[id]) continue;
      F t = triangles[id];
      for (int64_t& v : t) {
        if (remap[static_cast<size_t>(v)] < 0) {
          remap[static_cast<size_t>(v)] = static_cast<int64_t>(output_points.size());
          output_points.push_back(points[static_cast<size_t>(v)]);
        }
        v = remap[static_cast<size_t>(v)];
      }
      output_triangles.push_back(t);
    }
  }
};
}  // namespace

double certified_free_radius(const Point& a, const Point& b, const Point& c,
                           double roundoff) {
  const double ab = length(difference(b, a)), bc = length(difference(c, b)),
               ca = length(difference(a, c));
  const V normal = cross(difference(b, a), difference(c, a));
  const double twice_area = length(normal);
  if (!(twice_area > 0) || !std::isfinite(twice_area))
    return std::numeric_limits<double>::infinity();
  std::array<double, 3> sides{bc, ca, ab};  // opposite a, b, c
  std::array<double, 3> squares{bc * bc, ca * ca, ab * ab};
  int obtuse = -1;
  for (int k = 0; k < 3; ++k)
    if (squares[k] > squares[(k + 1) % 3] + squares[(k + 2) % 3]) obtuse = k;
  double radius;
  if (obtuse < 0) {
    radius = ab * bc * ca / (2 * twice_area);
  } else {
    // Obtuse at corner k: the bisectors of its two sides meet the longest
    // side at distance side / (2 cos(adjacent angle)) from that side's ends.
    radius = 0;
    for (int k = 0; k < 3; ++k) {
      if (k == obtuse) continue;
      const int other = 3 - k - obtuse;
      // Angle at corner k is between the sides opposite `obtuse` and `other`.
      const double cosine = (squares[obtuse] + squares[other] - squares[k]) /
                            (2 * sides[obtuse] * sides[other]);
      if (!(cosine > 0)) return std::numeric_limits<double>::infinity();
      radius = std::max(radius, sides[other] / (2 * cosine));
    }
  }
  return next_up(next_up(radius * (1 + 64 * std::numeric_limits<double>::epsilon())) +
                 roundoff);
}

int32_t remesh_surface(std::vector<Point>& points, std::vector<Triangle>& triangles,
                       const TriangleAdmission& admission, const RemeshTargets& targets,
                       RemeshCounters& counters) {
  if (!(targets.edge_length > 0) || !std::isfinite(targets.edge_length) ||
      !(targets.minimum_angle_degrees > 0) || !(targets.minimum_angle_degrees < 60) ||
      !(targets.maximum_normal_turn_degrees > 0) || !(targets.maximum_normal_turn_degrees < 180))
    return PHX_MC_INVALID_ARGUMENT;
  if (triangles.empty()) return PHX_MC_INVALID_ARGUMENT;
  for (const Triangle& t : triangles)
    for (int64_t v : t)
      if (v < 0 || v >= static_cast<int64_t>(points.size())) return PHX_MC_INVALID_ARGUMENT;
  Remesher remesher(admission, targets);
  remesher.run(points, triangles);
  counters.splits = remesher.splits;
  counters.collapses = remesher.collapses;
  counters.flips = remesher.flips;
  counters.relocations = remesher.relocations;
  counters.unmet = remesher.unmet();
  remesher.compact(points, triangles);
  if (triangles.empty()) return PHX_MC_INTERNAL_ERROR;
  return PHX_MC_OK;
}

}  // namespace phx::mc::remesh
