// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Included inside arrangement.cpp after reference_tetrahedron_clip.inc.
// Exact pooled clipping of positive tetrahedra in one coordinate frame.

struct TetrahedraArrangementBanks {
  std::vector<std::uint32_t> vertex_words;
  std::vector<std::int64_t> vertex_word_offsets;
  std::vector<std::int8_t> vertex_signs;
  std::vector<std::int32_t> vertex_input_index;
  std::vector<std::int64_t> cell_face_offsets;
  std::vector<std::int64_t> face_offsets;
  std::vector<std::int32_t> face_vertices;
  std::vector<std::int32_t> face_planes;
};

bool tetrahedra_arrangement_arguments_valid(const phx_mc_arrange_tetrahedra_exact_request& r) {
  if (r.point_count < 0 || r.plane_count < 0 || r.tetrahedron_count < 0 || r.job_count < 0 ||
      r.cut_count < 0 || r.point_word_count < 0 || r.plane_word_count < 0 ||
      r.point_count > std::numeric_limits<std::int32_t>::max() ||
      r.plane_count > std::numeric_limits<std::int32_t>::max() ||
      r.tetrahedron_count > std::numeric_limits<std::int32_t>::max() ||
      r.job_count > std::numeric_limits<std::int32_t>::max() ||
      !addressable(r.point_count, 4, sizeof(std::int64_t)) ||
      !addressable(r.plane_count, 4, sizeof(std::int64_t)) ||
      !addressable(4 * r.point_count + 1, 1, sizeof(std::int64_t)) ||
      !addressable(4 * r.plane_count + 1, 1, sizeof(std::int64_t)) ||
      !addressable(r.point_word_count, 1, sizeof(std::uint32_t)) ||
      !addressable(r.plane_word_count, 1, sizeof(std::uint32_t)) ||
      !addressable(r.tetrahedron_count, 4, sizeof(std::int32_t)) ||
      !addressable(r.job_count, 1, sizeof(std::int32_t)) ||
      !addressable(r.job_count + 1, 1, sizeof(std::int64_t)) ||
      !addressable(r.cut_count, 1, sizeof(std::int32_t))) {
    return false;
  }
  if (r.point_offsets == nullptr || r.plane_offsets == nullptr || r.cut_offsets == nullptr ||
      (r.point_count > 0 && r.point_signs == nullptr) ||
      (r.plane_count > 0 && r.plane_signs == nullptr) ||
      (r.point_word_count > 0 && r.point_words == nullptr) ||
      (r.plane_word_count > 0 && r.plane_words == nullptr) ||
      (r.tetrahedron_count > 0 &&
       (r.tetrahedra == nullptr || r.tetrahedron_planes == nullptr)) ||
      (r.job_count > 0 && r.parents == nullptr) ||
      (r.cut_count > 0 && r.cut_planes == nullptr)) {
    return false;
  }
  return r.point_offsets[0] == 0 && r.point_offsets[4 * r.point_count] == r.point_word_count &&
         r.plane_offsets[0] == 0 && r.plane_offsets[4 * r.plane_count] == r.plane_word_count &&
         r.cut_offsets[0] == 0 && r.cut_offsets[r.job_count] == r.cut_count;
}

class TetrahedraArrangement {
 public:
  explicit TetrahedraArrangement(const phx_mc_arrange_tetrahedra_exact_request& request)
      : in_(request) {}

  int32_t run(TetrahedraArrangementBanks& banks) {
    int32_t status = validate_topology();
    if (status != PHX_MC_OK) return status;
    status = decode_planes();
    if (status != PHX_MC_OK) return status;
    status = decode_points();
    if (status != PHX_MC_OK) return status;
    status = validate_tetrahedra();
    if (status != PHX_MC_OK) return status;
    status = execute_jobs(banks);
    if (status != PHX_MC_OK) return status;
    return publish(banks);
  }

 private:
  using Point = std::array<Dyadic, 4>;
  using Plane = std::array<Dyadic, 4>;
  using Key = std::array<std::uint64_t, 3>;

  struct Cell {
    std::vector<std::int32_t> points;
    std::vector<ExactReferenceFace3> faces;
  };

  struct EdgeKey {
    std::int32_t first;
    std::int32_t second;
    std::int32_t plane;
    friend bool operator<(const EdgeKey& a, const EdgeKey& b) {
      return std::tie(a.first, a.second, a.plane) < std::tie(b.first, b.second, b.plane);
    }
  };

  static std::int32_t plane_row(std::int32_t reference) {
    return reference > 0 ? reference - 1 : -(reference + 1);
  }

  bool plane_reference_valid(std::int64_t reference) const {
    return reference != 0 && reference <= in_.plane_count && reference >= -in_.plane_count;
  }

  int32_t validate_scalars(std::int64_t rows, std::int64_t word_count,
                           const std::uint32_t* words, const std::int64_t* offsets,
                           const std::int8_t* signs) const {
    for (std::int64_t scalar = 0; scalar < 4 * rows; ++scalar) {
      const std::int64_t begin = offsets[scalar], end = offsets[scalar + 1];
      if (begin < 0 || end < begin || end > word_count || signs[scalar] < -1 || signs[scalar] > 1 ||
          ((begin == end) != (signs[scalar] == 0)) || (end > begin && words[end - 1] == 0)) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    return PHX_MC_OK;
  }

  int32_t validate_topology() const {
    int32_t status = validate_scalars(in_.point_count, in_.point_word_count, in_.point_words,
                                      in_.point_offsets, in_.point_signs);
    if (status != PHX_MC_OK) return status;
    status = validate_scalars(in_.plane_count, in_.plane_word_count, in_.plane_words,
                              in_.plane_offsets, in_.plane_signs);
    if (status != PHX_MC_OK) return status;
    for (std::int64_t tetrahedron = 0; tetrahedron < in_.tetrahedron_count; ++tetrahedron) {
      for (int corner = 0; corner < 4; ++corner) {
        const std::int64_t point = in_.tetrahedra[4 * tetrahedron + corner];
        const std::int32_t face = in_.tetrahedron_planes[4 * tetrahedron + corner];
        if (point < 0 || point >= in_.point_count || !plane_reference_valid(face))
          return PHX_MC_INVALID_INPUT;
        for (int other = 0; other < corner; ++other)
          if (in_.tetrahedra[4 * tetrahedron + other] == point) return PHX_MC_INVALID_INPUT;
      }
    }
    for (std::int64_t job = 0; job < in_.job_count; ++job) {
      const std::int64_t parent = in_.parents[job];
      if (parent < 0) {
        if (-parent - 1 >= in_.tetrahedron_count) return PHX_MC_INVALID_INPUT;
      } else if (parent >= job) {
        return PHX_MC_INVALID_INPUT;
      }
      const std::int64_t begin = in_.cut_offsets[job], end = in_.cut_offsets[job + 1];
      if (begin < 0 || end < begin || end > in_.cut_count) return PHX_MC_INVALID_INPUT;
      for (std::int64_t cut = begin; cut < end; ++cut)
        if (!plane_reference_valid(in_.cut_planes[cut])) return PHX_MC_INVALID_INPUT;
    }
    return PHX_MC_OK;
  }

  Dyadic decode(const std::uint32_t* words, const std::int64_t* offsets,
                const std::int8_t* signs, std::int64_t scalar) const {
    const std::int64_t begin = offsets[scalar], end = offsets[scalar + 1];
    return Dyadic::integer(words == nullptr ? nullptr : words + begin,
                           static_cast<std::size_t>(end - begin), signs[scalar]);
  }

  int32_t decode_planes() {
    planes_.resize(static_cast<std::size_t>(in_.plane_count));
    for (std::int64_t plane = 0; plane < in_.plane_count; ++plane) {
      Plane& row = planes_[static_cast<std::size_t>(plane)];
      for (int axis = 0; axis < 4; ++axis)
        row[static_cast<std::size_t>(axis)] =
            decode(in_.plane_words, in_.plane_offsets, in_.plane_signs, 4 * plane + axis);
      int leading = 0;
      for (int axis = 0; axis < 3 && leading == 0; ++axis) leading = row[axis].sign();
      if (leading <= 0) return PHX_MC_INVALID_INPUT;
    }
    return PHX_MC_OK;
  }

  int32_t intern(Point point, std::int32_t input_index, std::int32_t& id) {
    if (point[3].sign() <= 0) return PHX_MC_INVALID_INPUT;
    Key key{};
    for (int axis = 0; axis < 3; ++axis) {
      const double value = point[axis].sign() == 0
                               ? 0.0
                               : exact::rounded_quotient(point[axis], point[3]);
      key[static_cast<std::size_t>(axis)] = std::bit_cast<std::uint64_t>(value == 0.0 ? 0.0 : value);
    }
    auto& bucket = welds_[key];
    for (std::int32_t candidate : bucket) {
      const Point& other = points_[static_cast<std::size_t>(candidate)];
      bool equal = true;
      for (int axis = 0; axis < 3 && equal; ++axis)
        equal = (point[axis] * other[3] - other[axis] * point[3]).sign() == 0;
      if (equal) {
        id = candidate;
        return PHX_MC_OK;
      }
    }
    if (points_.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
      return PHX_MC_INDEX_RANGE_ERROR;
    id = static_cast<std::int32_t>(points_.size());
    points_.push_back(std::move(point));
    input_index_.push_back(input_index);
    bucket.push_back(id);
    return PHX_MC_OK;
  }

  int32_t decode_points() {
    point_vertex_.resize(static_cast<std::size_t>(in_.point_count));
    for (std::int64_t index = 0; index < in_.point_count; ++index) {
      Point point;
      for (int axis = 0; axis < 4; ++axis)
        point[axis] = decode(in_.point_words, in_.point_offsets, in_.point_signs, 4 * index + axis);
      if (point[3].sign() <= 0) return PHX_MC_INVALID_INPUT;
      std::int32_t id = -1;
      const int32_t status = intern(std::move(point), static_cast<std::int32_t>(index), id);
      if (status != PHX_MC_OK) return status;
      point_vertex_[static_cast<std::size_t>(index)] = id;
    }
    return PHX_MC_OK;
  }

  int side(std::int32_t plane, std::int32_t point) {
    const auto key = std::pair{plane, point};
    const auto found = sides_.find(key);
    if (found != sides_.end()) return found->second;
    const int value = reference_plane_side(planes_[static_cast<std::size_t>(plane)],
                                           points_[static_cast<std::size_t>(point)]);
    sides_.emplace(key, value);
    return value;
  }

  int32_t validate_tetrahedra() {
    for (std::int64_t tetrahedron = 0; tetrahedron < in_.tetrahedron_count; ++tetrahedron) {
      std::array<std::int32_t, 4> corners{};
      for (int corner = 0; corner < 4; ++corner)
        corners[static_cast<std::size_t>(corner)] = point_vertex_[static_cast<std::size_t>(
            in_.tetrahedra[4 * tetrahedron + corner])];
      const Point& h0 = points_[static_cast<std::size_t>(corners[0])];
      const Point& h1 = points_[static_cast<std::size_t>(corners[1])];
      const Point& h2 = points_[static_cast<std::size_t>(corners[2])];
      const Point& h3 = points_[static_cast<std::size_t>(corners[3])];
      const Dyadic determinant = h0[0] * reference_minor3(h1, h2, h3, 1, 2, 3) -
                                 h0[1] * reference_minor3(h1, h2, h3, 0, 2, 3) +
                                 h0[2] * reference_minor3(h1, h2, h3, 0, 1, 3) -
                                 h0[3] * reference_minor3(h1, h2, h3, 0, 1, 2);
      if (determinant.sign() >= 0) return PHX_MC_INVALID_INPUT;
      for (int face = 0; face < 4; ++face) {
        const std::int32_t reference = in_.tetrahedron_planes[4 * tetrahedron + face];
        const int orientation = reference > 0 ? 1 : -1;
        for (int corner = 0; corner < 4; ++corner) {
          const int value = orientation * side(plane_row(reference), corners[corner]);
          if (value != (corner == face ? -1 : 0)) return PHX_MC_INVALID_INPUT;
        }
      }
    }
    return PHX_MC_OK;
  }

  Cell initial_cell(std::int32_t parent) const {
    const std::int64_t tetrahedron = -static_cast<std::int64_t>(parent) - 1;
    Cell cell;
    for (int corner = 0; corner < 4; ++corner) {
      const auto point = in_.tetrahedra[4 * tetrahedron + corner];
      cell.points.push_back(point_vertex_[static_cast<std::size_t>(point)]);
    }
    for (int face_index = 0; face_index < 4; ++face_index) {
      const int* order = topology::kTetFaceOrder[face_index];
      ExactReferenceFace3 face;
      face.plane = in_.tetrahedron_planes[4 * tetrahedron + face_index];
      face.vertices = {order[0], order[2], order[1]};
      cell.faces.push_back(std::move(face));
    }
    return cell;
  }

  Cell job_cell(std::size_t job, const TetrahedraArrangementBanks& banks) const {
    Cell cell;
    const std::int64_t vertex_begin = job_vertex_offsets_[job];
    cell.points.assign(job_vertices_.begin() + vertex_begin,
                       job_vertices_.begin() + job_vertex_offsets_[job + 1]);
    const std::int64_t face_begin = banks.cell_face_offsets[job];
    const std::int64_t face_end = banks.cell_face_offsets[job + 1];
    for (std::int64_t row = face_begin; row < face_end; ++row) {
      ExactReferenceFace3 face;
      face.plane = banks.face_planes[static_cast<std::size_t>(row)];
      const std::int64_t begin = banks.face_offsets[static_cast<std::size_t>(row)];
      const std::int64_t end = banks.face_offsets[static_cast<std::size_t>(row + 1)];
      face.vertices.assign(banks.face_vertices.begin() + begin, banks.face_vertices.begin() + end);
      cell.faces.push_back(std::move(face));
    }
    return cell;
  }

  int32_t crossing_point(const Cell& cell, std::int32_t first, std::int32_t second,
                         std::int32_t reference, const std::vector<std::vector<std::int32_t>>& facets,
                         std::int32_t& result) {
    const std::int32_t plane = plane_row(reference);
    const std::int32_t a = cell.points[static_cast<std::size_t>(first)];
    const std::int32_t b = cell.points[static_cast<std::size_t>(second)];
    const EdgeKey edge{std::min(a, b), std::max(a, b), plane};
    const auto known = edges_.find(edge);
    if (known != edges_.end()) {
      result = known->second;
      return PHX_MC_OK;
    }
    const auto& first_faces = facets[static_cast<std::size_t>(first)];
    const auto& second_faces = facets[static_cast<std::size_t>(second)];
    std::vector<std::int32_t> support_planes;
    std::set_intersection(first_faces.begin(), first_faces.end(), second_faces.begin(),
                          second_faces.end(), std::back_inserter(support_planes));
    std::vector<std::int32_t> rows;
    for (std::int32_t face : support_planes) {
      const std::int32_t row = plane_row(cell.faces[static_cast<std::size_t>(face)].plane);
      if (row != plane && (rows.empty() || rows.back() != row)) rows.push_back(row);
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    std::int32_t support_a = -1, support_b = -1;
    Dyadic denominator;
    for (std::size_t i = 0; i < rows.size() && support_a < 0; ++i) {
      for (std::size_t j = i + 1; j < rows.size(); ++j) {
        const auto key = std::pair{std::min(rows[i], rows[j]), std::max(rows[i], rows[j])};
        const Dyadic candidate = reference_minor3(planes_[static_cast<std::size_t>(key.first)],
                                                  planes_[static_cast<std::size_t>(key.second)],
                                                  planes_[static_cast<std::size_t>(plane)], 0, 1, 2);
        if (candidate.sign() != 0) {
          support_a = key.first;
          support_b = key.second;
          denominator = candidate;
          break;
        }
      }
    }
    if (support_a < 0) return PHX_MC_INTERNAL_ERROR;
    const auto pair_key = std::pair{support_a, support_b};
    auto pair = pair_minors_.find(pair_key);
    if (pair == pair_minors_.end()) {
      pair = pair_minors_.emplace(
          pair_key, reference_pair_minors(planes_[static_cast<std::size_t>(support_a)],
                                          planes_[static_cast<std::size_t>(support_b)])).first;
    }
    std::array<std::int32_t, 3> triple{support_a, support_b, plane};
    std::sort(triple.begin(), triple.end());
    const auto known_triple = triples_.find(triple);
    if (known_triple != triples_.end()) {
      result = known_triple->second;
      edges_.emplace(edge, result);
      return PHX_MC_OK;
    }
    const Point point = reference_pair_point(pair->second, planes_[static_cast<std::size_t>(plane)],
                                             std::move(denominator));
    const int32_t status = intern(point, -1, result);
    if (status != PHX_MC_OK) return status;
    triples_.emplace(triple, result);
    edges_.emplace(edge, result);
    return PHX_MC_OK;
  }

  int32_t section(std::int32_t reference, Cell& cell, Cell& next,
                  const std::vector<int>* known_sides = nullptr) {
    const std::int32_t plane = plane_row(reference);
    const int orientation = reference > 0 ? 1 : -1;
    std::vector<int> sides(cell.points.size());
    if (known_sides != nullptr) {
      sides = *known_sides;
    } else {
      for (std::size_t vertex = 0; vertex < cell.points.size(); ++vertex)
        sides[vertex] = orientation * side(plane, cell.points[vertex]);
    }
    bool outside = false, inside = false;
    for (int value : sides) {
      outside = outside || value > 0;
      inside = inside || value < 0;
    }
    if (!outside) return PHX_MC_OK;
    if (!inside) return PHX_MC_DEGENERATE_INPUT;

    std::vector<std::vector<std::int32_t>> facets(cell.points.size());
    for (std::size_t face = 0; face < cell.faces.size(); ++face)
      for (std::int32_t vertex : cell.faces[face].vertices)
        facets[static_cast<std::size_t>(vertex)].push_back(static_cast<std::int32_t>(face));
    next.points.clear();
    next.faces.clear();
    std::vector<std::int32_t> retained(cell.points.size(), -1);
    std::vector<uint8_t> on_plane;
    std::map<std::pair<std::int32_t, std::int32_t>, std::int32_t> local_crossings;
    auto keep = [&](std::int32_t old_vertex) -> std::int32_t {
      std::int32_t& local = retained[static_cast<std::size_t>(old_vertex)];
      if (local >= 0) return local;
      if (next.points.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::length_error("Tetrahedron cell vertex ids exceed int32 range.");
      local = static_cast<std::int32_t>(next.points.size());
      next.points.push_back(cell.points[static_cast<std::size_t>(old_vertex)]);
      on_plane.push_back(static_cast<uint8_t>(sides[static_cast<std::size_t>(old_vertex)] == 0));
      return local;
    };
    auto crossing = [&](std::int32_t first, std::int32_t second) -> std::int32_t {
      if (sides[static_cast<std::size_t>(first)] == 0) return keep(first);
      if (sides[static_cast<std::size_t>(second)] == 0) return keep(second);
      const auto local_edge = std::pair{std::min(first, second), std::max(first, second)};
      const auto known = local_crossings.find(local_edge);
      if (known != local_crossings.end()) return known->second;
      std::int32_t point = -1;
      const int32_t status = crossing_point(cell, first, second, reference, facets, point);
      if (status != PHX_MC_OK) throw Refusal{status, -1};
      if (next.points.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::length_error("Tetrahedron cell vertex ids exceed int32 range.");
      const std::int32_t local = static_cast<std::int32_t>(next.points.size());
      next.points.push_back(point);
      on_plane.push_back(1);
      local_crossings.emplace(local_edge, local);
      return local;
    };
    auto is_on_plane = [&](std::int32_t vertex) {
      return on_plane[static_cast<std::size_t>(vertex)] != 0;
    };
    if (!reference_section(cell.faces, sides, keep, crossing, is_on_plane,
                           [&] { return next.points.size(); }, reference, next.faces)) {
      return PHX_MC_INTERNAL_ERROR;
    }
    if (next.faces.size() < 4) return PHX_MC_DEGENERATE_INPUT;
    std::swap(cell, next);
    return PHX_MC_OK;
  }

  bool same_tetrahedron_faces(std::int32_t first, std::int32_t second) const {
    std::array<bool, 4> used{};
    for (int slot = 0; slot < 4; ++slot) {
      const std::int32_t reference = in_.tetrahedron_planes[4 * first + slot];
      int match = 0;
      while (match < 4 &&
             (used[match] || in_.tetrahedron_planes[4 * second + match] != reference)) {
        ++match;
      }
      if (match == 4) return false;
      used[match] = true;
    }
    return true;
  }

  std::pair<Dyadic, Dyadic> tetrahedron_volume(std::int32_t tetrahedron) const {
    const std::int32_t* ids = in_.tetrahedra + 4 * static_cast<std::int64_t>(tetrahedron);
    const Point& origin = points_[static_cast<std::size_t>(point_vertex_[ids[0]])];
    std::array<std::array<Dyadic, 3>, 3> edges;
    for (int vertex = 1; vertex < 4; ++vertex) {
      const Point& point = points_[static_cast<std::size_t>(point_vertex_[ids[vertex]])];
      for (int axis = 0; axis < 3; ++axis) {
        edges[vertex - 1][axis] = point[axis] * origin[3] - origin[axis] * point[3];
      }
    }
    Dyadic numerator = exact::det3(
        edges[0][0], edges[0][1], edges[0][2], edges[1][0], edges[1][1], edges[1][2],
        edges[2][0], edges[2][1], edges[2][2]);
    Dyadic denominator = origin[3] * origin[3] * origin[3];
    for (int vertex = 1; vertex < 4; ++vertex) {
      denominator = denominator * points_[static_cast<std::size_t>(point_vertex_[ids[vertex]])][3];
    }
    if (numerator.sign() < 0) numerator.negate();
    if (denominator.sign() < 0) denominator.negate();
    return {std::move(numerator), std::move(denominator)};
  }

  int compare_tetrahedron_volumes(std::int32_t first, std::int32_t second) const {
    const auto a = tetrahedron_volume(first);
    const auto b = tetrahedron_volume(second);
    return (a.first * b.second - b.first * a.second).sign();
  }

  std::int32_t cut_tetrahedron(std::int64_t job) const {
    const std::int32_t parent = in_.parents[job];
    if (parent >= 0) return -1;
    const std::int64_t begin = in_.cut_offsets[job];
    const std::int64_t end = in_.cut_offsets[job + 1];
    for (std::int32_t tetrahedron = 0; tetrahedron < in_.tetrahedron_count; ++tetrahedron) {
      bool complete = true;
      for (int slot = 0; slot < 4 && complete; ++slot) {
        const std::int32_t reference = in_.tetrahedron_planes[4 * tetrahedron + slot];
        bool present = false;
        for (std::int64_t cut = begin; cut < end; ++cut) {
          present = present || in_.cut_planes[cut] == reference;
        }
        complete = present;
      }
      if (complete) return tetrahedron;
    }
    return -1;
  }

  void order_declared_intersection(Cell& cell, std::int64_t job) const {
    const std::int32_t parent = in_.parents[job];
    if (parent >= 0) return;
    const std::int32_t parent_tet = static_cast<std::int32_t>(-static_cast<std::int64_t>(parent) - 1);
    const std::int32_t other_tet = cut_tetrahedron(job);
    if (other_tet < 0) return;

    // The smaller operand supplies stable face order. Equal geometric operands
    // use the first declaration; equal-volume distinct operands retain parent order.
    std::int32_t first = parent_tet;
    if (same_tetrahedron_faces(parent_tet, other_tet)) {
      first = std::min(parent_tet, other_tet);
    } else if (compare_tetrahedron_volumes(parent_tet, other_tet) > 0) {
      first = other_tet;
    }
    const std::int32_t second = first == parent_tet ? other_tet : parent_tet;
    const auto priority = [&](std::int32_t reference) {
      for (int slot = 0; slot < 4; ++slot) {
        if (in_.tetrahedron_planes[4 * first + slot] == reference) return int64_t{slot};
      }
      for (int slot = 0; slot < 4; ++slot) {
        if (in_.tetrahedron_planes[4 * second + slot] == reference) return int64_t{4 + slot};
      }
      return int64_t{8} + static_cast<int64_t>(reference) -
             std::numeric_limits<std::int32_t>::min();
    };
    std::sort(cell.faces.begin(), cell.faces.end(), [&](const ExactReferenceFace3& a,
                                                        const ExactReferenceFace3& b) {
      return priority(a.plane) < priority(b.plane);
    });
  }
  int32_t commit(bool empty, const Cell& cell, TetrahedraArrangementBanks& banks) {
    if (!empty) {
      std::vector<int32_t> rows(cell.points.size(), -1);
      for (const auto& face : cell.faces)
        for (int32_t vertex : face.vertices) rows[static_cast<std::size_t>(vertex)] = 0;
      int32_t count = 0;
      for (std::size_t vertex = 0; vertex < cell.points.size(); ++vertex) {
        if (rows[vertex] < 0) continue;
        if (count == std::numeric_limits<int32_t>::max()) return PHX_MC_INDEX_RANGE_ERROR;
        rows[vertex] = count++;
        job_vertices_.push_back(cell.points[vertex]);
      }
      for (const auto& face : cell.faces) {
        banks.face_planes.push_back(face.plane);
        for (int32_t vertex : face.vertices)
          banks.face_vertices.push_back(rows[static_cast<std::size_t>(vertex)]);
        banks.face_offsets.push_back(static_cast<std::int64_t>(banks.face_vertices.size()));
      }
    }
    job_vertex_offsets_.push_back(static_cast<std::int64_t>(job_vertices_.size()));
    banks.cell_face_offsets.push_back(static_cast<std::int64_t>(banks.face_planes.size()));
    return PHX_MC_OK;
  }

  int32_t execute_jobs(TetrahedraArrangementBanks& banks) {
    banks.cell_face_offsets = {0};
    banks.face_offsets = {0};
    job_vertex_offsets_ = {0};
    Cell cell, next;
    for (std::int64_t job = 0; job < in_.job_count; ++job) {
      const std::int32_t parent = in_.parents[job];
      bool empty = false;
      if (parent >= 0 && banks.cell_face_offsets[static_cast<std::size_t>(parent)] ==
                             banks.cell_face_offsets[static_cast<std::size_t>(parent) + 1]) {
        empty = true;
      } else {
        cell = parent < 0 ? initial_cell(parent)
                          : job_cell(static_cast<std::size_t>(parent), banks);
        const std::int64_t begin = in_.cut_offsets[job], end = in_.cut_offsets[job + 1];
        std::vector<std::int32_t> active;
        std::vector<int> first_sides;
        for (std::int64_t cut = begin; cut < end; ++cut) {
          const std::int32_t reference = in_.cut_planes[cut];
          const std::int32_t plane = plane_row(reference);
          const int orientation = reference > 0 ? 1 : -1;
          std::vector<int> sides(cell.points.size());
          bool outside = false, inside = false;
          for (std::size_t vertex = 0; vertex < cell.points.size(); ++vertex) {
            sides[vertex] = orientation * side(plane, cell.points[vertex]);
            outside = outside || sides[vertex] > 0;
            inside = inside || sides[vertex] < 0;
          }
          if (!inside) {
            empty = true;
            break;
          }
          if (outside) {
            if (active.empty()) first_sides = sides;
            active.push_back(reference);
          }
        }
        for (std::size_t index = 0; index < active.size() && !empty; ++index) {
          const int32_t status = section(active[index], cell, next,
                                         index == 0 ? &first_sides : nullptr);
          if (status == PHX_MC_DEGENERATE_INPUT) empty = true;
          else if (status != PHX_MC_OK) return status;
        }
      }
      if (!empty) order_declared_intersection(cell, job);
      const int32_t status = commit(empty, cell, banks);
      if (status != PHX_MC_OK) return status;
    }
    return PHX_MC_OK;
  }

  int32_t append_scalar(const Dyadic& value, TetrahedraArrangementBanks& banks) const {
    if (value.sign() != 0) {
      if (value.exponent() < 0) return PHX_MC_INTERNAL_ERROR;
      const Limbs limbs = shift_left(value.magnitude(), value.exponent());
      banks.vertex_words.insert(banks.vertex_words.end(), limbs.begin(), limbs.end());
    }
    banks.vertex_signs.push_back(static_cast<std::int8_t>(value.sign()));
    banks.vertex_word_offsets.push_back(static_cast<std::int64_t>(banks.vertex_words.size()));
    return PHX_MC_OK;
  }

  int32_t publish(TetrahedraArrangementBanks& banks) {
    std::vector<std::int32_t> output(points_.size(), -1);
    std::vector<std::int32_t> order;
    const std::size_t jobs = static_cast<std::size_t>(in_.job_count);
    for (std::size_t job = 0; job < jobs; ++job) {
      const std::int64_t vertex_begin = job_vertex_offsets_[job];
      for (std::int64_t face = banks.cell_face_offsets[job];
           face < banks.cell_face_offsets[job + 1]; ++face) {
        const auto begin = banks.face_vertices.begin() + banks.face_offsets[static_cast<std::size_t>(face)];
        const auto end = banks.face_vertices.begin() + banks.face_offsets[static_cast<std::size_t>(face) + 1];
        for (auto entry = begin; entry != end; ++entry) {
          const std::int32_t point = job_vertices_[static_cast<std::size_t>(vertex_begin + *entry)];
          std::int32_t& id = output[static_cast<std::size_t>(point)];
          if (id < 0) {
            if (order.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
              return PHX_MC_INDEX_RANGE_ERROR;
            id = static_cast<std::int32_t>(order.size());
            order.push_back(point);
          }
          *entry = id;
        }
        std::rotate(begin, std::min_element(begin, end), end);
      }
    }
    banks.vertex_word_offsets = {0};
    banks.vertex_signs.reserve(4 * order.size());
    banks.vertex_input_index.reserve(order.size());
    for (std::int32_t point : order) {
      for (int axis = 0; axis < 4; ++axis) {
        const int32_t status = append_scalar(points_[static_cast<std::size_t>(point)][axis], banks);
        if (status != PHX_MC_OK) return status;
      }
      banks.vertex_input_index.push_back(input_index_[static_cast<std::size_t>(point)]);
    }
    return PHX_MC_OK;
  }

  const phx_mc_arrange_tetrahedra_exact_request& in_;
  std::vector<Plane> planes_;
  std::vector<Point> points_;
  std::vector<std::int32_t> input_index_;
  std::vector<std::int32_t> point_vertex_;
  std::map<Key, std::vector<std::int32_t>> welds_;
  std::map<std::pair<std::int32_t, std::int32_t>, int> sides_;
  std::map<EdgeKey, std::int32_t> edges_;
  std::map<std::array<std::int32_t, 3>, std::int32_t> triples_;
  std::map<std::pair<std::int32_t, std::int32_t>, std::array<Dyadic, 6>> pair_minors_;
  std::vector<std::int64_t> job_vertex_offsets_;
  std::vector<std::int32_t> job_vertices_;
};

int32_t arrange_tetrahedra_exact(const phx_mc_arrange_tetrahedra_exact_request* request,
                                 TetrahedraArrangementBanks& banks) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  if (request->point_count > std::numeric_limits<std::int32_t>::max() ||
      request->plane_count > std::numeric_limits<std::int32_t>::max() ||
      request->tetrahedron_count > std::numeric_limits<std::int32_t>::max() ||
      request->job_count > std::numeric_limits<std::int32_t>::max()) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  if (!tetrahedra_arrangement_arguments_valid(*request)) return PHX_MC_INVALID_ARGUMENT;
  try {
    TetrahedraArrangement arrangement(*request);
    return arrangement.run(banks);
  } catch (const Refusal& refusal) {
    return refusal.status;
  } catch (const std::length_error&) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
}
