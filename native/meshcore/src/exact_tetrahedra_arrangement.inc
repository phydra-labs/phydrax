// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Included inside arrangement.cpp after reference_tetrahedron_clip.inc.
//
// One exact geometry pool per call for a deterministic clipping DAG of
// positive tetrahedra in a single coordinate frame. Plane equations are
// decoded once. Exact plane sides are cached per (plane, welded point), and a
// strict edge crossing per (edge endpoints, plane) and per unordered plane
// triple, so a repeated query performs no exact arithmetic: only its table
// probes are charged. A new crossing is the Cramer point of the edge's two
// facet planes and the cut, as in the reference clipper, evaluated from the
// facet pair's six 2x2 minors (prepared once per call) as three-term
// combinations of the cut row: the same exact integers, fewer products. Its
// nonzero normal determinant proves the three planes independent. Points are
// welded by exact projective equality: correctly rounded binary64
// coordinates only select candidates, homogeneous cross products decide, and
// distinct exact points sharing a rounding stay distinct. Pool ids, plane rows and
// loop rows are numeric identities only; scientific labels stay with the
// caller.
//
// Each job screens its cuts against its operand's immutable vertices with
// pooled exact sides, then copies only a surviving operand's bounded
// topology (pooled point ids and outward face loops over local rows) and
// applies the splitting cuts with the shared reference_section engine. The
// inherited chain holds at most 16 halfspaces, so a cell has at most 16
// facets, 28 vertices and 84 loop entries. Facets lie on pairwise distinct
// geometric planes, and the facets containing both ends of an edge are
// exactly its two facets: they are the independent support pair of every
// strict crossing. Completed jobs and pooled exact points are immutable;
// children copy indices, never coordinates.
//
// A job whose four cuts are, as a set, exactly the outward signed face
// references of a declared initial tetrahedron B (the smallest such index)
// is its parent P intersect B; equal plane references are only numeric
// geometry reuse. Such a job also screens P's facets against B's corners:
// either direction may prove no measure. B inside P publishes B whole, also
// when P equals B; P inside B publishes P whole; otherwise B is clipped by
// P's splitting facets only when they are strictly fewer than B's faces
// splitting P, else P is clipped as any job. The chain bound still holds:
// B gains at most P's facets, which lie in P's chain.

struct TetrahedraArrangementBanks {
  NativeVector<std::uint32_t> vertex_words;
  NativeVector<std::int64_t> vertex_word_offsets;
  NativeVector<std::int8_t> vertex_signs;
  NativeVector<std::int32_t> vertex_input_index;
  NativeVector<std::int64_t> cell_face_offsets;
  NativeVector<std::int64_t> face_offsets;
  NativeVector<std::int32_t> face_vertices;
  NativeVector<std::int32_t> face_planes;
  std::array<std::int64_t, PHX_MC_TETRAHEDRA_ARRANGEMENT_COUNTERS> counters{};
};

// Always-published failure/resource evidence of one call.
struct TetrahedraArrangementRecord {
  std::int64_t work = 0;
  std::int64_t queries = 0;
  std::int64_t completed = 0;
  std::int64_t category = 0;
  std::int64_t row = -1;
};

enum TetrahedraArrangementFailure : std::int64_t {
  kArrangementNoFailure = 0,
  kArrangementPointFailure = 1,
  kArrangementPlaneFailure = 2,
  kArrangementTetrahedronFailure = 3,
  kArrangementJobFailure = 4,
  kArrangementCutFailure = 5,
  kArrangementExecutionFailure = 6,
  kArrangementPublicationFailure = 7,
};

enum TetrahedraArrangementCounter : int {
  kArrangementPropagatedEmpty = 0,
  kArrangementSections = 1,
  kArrangementSideEvaluations = 2,
  kArrangementSideReuses = 3,
  kArrangementConstructions = 4,
  kArrangementConstructionReuses = 5,
  kArrangementWelds = 6,
  kArrangementPoolVertices = 7,
};
static_assert(PHX_MC_TETRAHEDRA_ARRANGEMENT_COUNTERS == 8);
static_assert(PHX_MC_TETRAHEDRA_ARRANGEMENT_EVIDENCE == 6);
static_assert(PHX_MC_TETRAHEDRA_ARRANGEMENT_SIZES == 5);

// Work is admitted against the call's own limit and the inherited native
// scope before it is recorded; refused work is never recorded.
struct ArrangementWork {
  std::int64_t limit;
  std::int64_t* total;
  void operator()(std::int64_t operations) const {
    if (operations < 0 || operations > limit - *total)
      throw ExecutionRefusal{PHX_MC_CAPACITY_EXCEEDED};
    native_execution_charge(operations);
    *total += operations;
  }
};

// Trivially copyable rows on the bounded native pool without value
// initialization: a slot is read only after the table's occupancy bit says
// it was written.
template <class T>
class UninitializedRows {
  static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>);

 public:
  UninitializedRows() = default;
  explicit UninitializedRows(std::size_t count)
      : data_(count == 0 ? nullptr : allocator_.allocate(count)), count_(count) {}
  ~UninitializedRows() {
    if (data_ != nullptr) allocator_.deallocate(data_, count_);
  }
  UninitializedRows(const UninitializedRows&) = delete;
  UninitializedRows& operator=(const UninitializedRows&) = delete;
  void swap(UninitializedRows& other) noexcept {
    std::swap(allocator_, other.allocator_);
    std::swap(data_, other.data_);
    std::swap(count_, other.count_);
  }
  T& operator[](std::size_t index) { return data_[index]; }
  const T& operator[](std::size_t index) const { return data_[index]; }
  std::size_t size() const { return count_; }

 private:
  NativeAllocator<T> allocator_;
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

// Open-addressed table of exact integer keys (plane rows, pool ids, binary64
// bit patterns) on the bounded native pool. Every probe is charged, so cache
// hits poll the work/deadline owner. A miss remembers its empty slot, so the
// following insert of that key does not probe again unless the table grows
// in between. Slots are tracked by an occupancy bitset: growth zeroes one
// word per 64 slots, scans the old words and re-probes each moved entry, and
// charges exactly those operations before allocating; key and value payloads
// are never initialized wholesale.
class ExactKeyTable {
 public:
  using Key = std::array<std::uint64_t, 3>;
  static constexpr std::size_t kNoSlot = std::numeric_limits<std::size_t>::max();

  struct Probe {
    std::int32_t* value = nullptr;
    std::size_t slot = kNoSlot;
  };

  Probe find(const Key& key, const ArrangementWork& charge) {
    if (capacity_ == 0) return {};
    const std::size_t mask = capacity_ - 1;
    for (std::size_t slot = hash(key) & mask;; slot = (slot + 1) & mask) {
      charge(1);
      if (!occupied(slot)) return {nullptr, slot};
      if (keys_[slot] == key) return {&values_[slot], slot};
    }
  }

  // Inserts an absent key after its missed `probe`; `value` is nonnegative.
  void insert(const Probe& probe, const Key& key, std::int32_t value,
              const ArrangementWork& charge) {
    std::size_t slot = probe.slot;
    if (slot != kNoSlot && (slot >= capacity_ || occupied(slot))) slot = kNoSlot;
    if (2 * (used_ + 1) > capacity_) {
      grow(capacity_for(used_ + 1), charge);
      slot = kNoSlot;
    }
    if (slot == kNoSlot) slot = empty_slot(key, charge);
    place(slot, key, value);
    ++used_;
  }

  // One charged allocation for a justified number of entries, instead of
  // repeated doubling up to it; never shrinks.
  void reserve(std::size_t entries, const ArrangementWork& charge) {
    if (entries == 0) return;
    const std::size_t capacity = capacity_for(entries);
    if (capacity > capacity_) grow(capacity, charge);
  }

 private:
  bool occupied(std::size_t slot) const { return (occupied_[slot >> 6] >> (slot & 63)) & 1u; }

  void place(std::size_t slot, const Key& key, std::int32_t value) {
    keys_[slot] = key;
    values_[slot] = value;
    occupied_[slot >> 6] |= std::uint64_t{1} << (slot & 63);
  }

  std::size_t empty_slot(const Key& key, const ArrangementWork& charge) const {
    const std::size_t mask = capacity_ - 1;
    std::size_t slot = hash(key) & mask;
    charge(1);
    while (occupied(slot)) {
      charge(1);
      slot = (slot + 1) & mask;
    }
    return slot;
  }

  // Power of two (at least 64) holding `entries` at load at most one half.
  std::size_t capacity_for(std::size_t entries) const {
    std::size_t capacity = capacity_ == 0 ? 64 : capacity_;
    while (2 * entries > capacity) capacity *= 2;
    return capacity;
  }

  static std::size_t hash(const Key& key) {
    std::uint64_t value = 0x9E3779B97F4A7C15ULL;
    for (const std::uint64_t word : key) {
      value ^= word + 0x9E3779B97F4A7C15ULL + (value << 6) + (value >> 2);
      value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
      value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
      value ^= value >> 31;
    }
    return static_cast<std::size_t>(value);
  }

  void grow(std::size_t capacity, const ArrangementWork& charge) {
    // Zero the new occupancy words and scan the old ones, before allocation.
    const auto words = static_cast<std::int64_t>((capacity + 63) / 64);
    const auto old_words = static_cast<std::int64_t>(occupied_.size());
    charge(words + old_words);
    ExactKeyTable next;
    UninitializedRows<Key> keys(capacity);
    UninitializedRows<std::int32_t> values(capacity);
    next.keys_.swap(keys);
    next.values_.swap(values);
    next.occupied_.assign(static_cast<std::size_t>(words), 0);
    next.capacity_ = capacity;
    for (std::size_t word = 0; word < occupied_.size(); ++word) {
      for (std::uint64_t bits = occupied_[word]; bits != 0; bits &= bits - 1) {
        const std::size_t old = 64 * word + static_cast<std::size_t>(std::countr_zero(bits));
        // Each moved entry re-probes the new table; every probe is charged.
        next.place(next.empty_slot(keys_[old], charge), keys_[old], values_[old]);
      }
    }
    keys_.swap(next.keys_);
    values_.swap(next.values_);
    occupied_.swap(next.occupied_);
    capacity_ = capacity;
  }

  UninitializedRows<Key> keys_;
  UninitializedRows<std::int32_t> values_;
  NativeVector<std::uint64_t> occupied_;
  std::size_t capacity_ = 0;
  std::size_t used_ = 0;
};

bool tetrahedra_arrangement_arguments_valid(const phx_mc_arrange_tetrahedra_exact_request& r) {
  constexpr std::int64_t kRows = std::numeric_limits<std::int32_t>::max();
  if (r.point_count < 0 || r.plane_count < 0 || r.tetrahedron_count < 0 || r.job_count < 0 ||
      r.cut_count < 0 || r.point_word_count < 0 || r.plane_word_count < 0 ||
      r.point_count > kRows || r.plane_count > kRows || r.tetrahedron_count > kRows ||
      r.job_count > kRows || r.scratch_limit <= 0 || r.work_limit <= 0 ||
      !addressable(r.point_word_count, 1, sizeof(std::uint32_t)) ||
      !addressable(r.plane_word_count, 1, sizeof(std::uint32_t)) ||
      !addressable(r.tetrahedron_count, 4, sizeof(std::int32_t)) ||
      !addressable(r.cut_count, 1, sizeof(std::int32_t)))
    return false;
  if (r.point_offsets == nullptr || r.plane_offsets == nullptr || r.cut_offsets == nullptr ||
      (r.point_count > 0 && r.point_signs == nullptr) ||
      (r.plane_count > 0 && r.plane_signs == nullptr) ||
      (r.point_word_count > 0 && r.point_words == nullptr) ||
      (r.plane_word_count > 0 && r.plane_words == nullptr) ||
      (r.tetrahedron_count > 0 && (r.tetrahedra == nullptr || r.tetrahedron_planes == nullptr)) ||
      (r.job_count > 0 && r.parents == nullptr) || (r.cut_count > 0 && r.cut_planes == nullptr))
    return false;
  return r.point_offsets[0] == 0 && r.point_offsets[4 * r.point_count] == r.point_word_count &&
         r.plane_offsets[0] == 0 && r.plane_offsets[4 * r.plane_count] == r.plane_word_count &&
         r.cut_offsets[0] == 0 && r.cut_offsets[r.job_count] == r.cut_count;
}

class TetrahedraArrangement {
 public:
  TetrahedraArrangement(const phx_mc_arrange_tetrahedra_exact_request& request,
                        TetrahedraArrangementRecord& record)
      : in_(request), record_(record), charge_{request.work_limit, &record.work} {}
  TetrahedraArrangement(const TetrahedraArrangement&) = delete;
  TetrahedraArrangement& operator=(const TetrahedraArrangement&) = delete;

  // Validates every input, executes every job and fills `banks`. Throws on
  // refusal; the record then names the failing category and row.
  void run(TetrahedraArrangementBanks& banks) {
    validate_topology();
    decode_planes();
    decode_points();
    validate_tetrahedra();
    execute_jobs(banks);
    publish(banks);
  }

 private:
  using Point = std::array<Dyadic, 4>;
  using Key = ExactKeyTable::Key;
  static constexpr int kCellVertices = kReferenceTetraMaximumVertices;
  static constexpr int kCellFaces = kReferenceTetraMaximumPlanes;

  struct Cell {
    std::array<std::int32_t, kCellVertices> points{};
    int used = 0;
    std::array<ExactReferenceFace3, kCellFaces> faces{};
    int face_used = 0;
  };

  [[noreturn]] void refuse(std::int64_t category, std::int64_t row) {
    record_.category = category;
    record_.row = row;
    throw Refusal{PHX_MC_INVALID_INPUT, row};
  }
  [[noreturn]] static void defect() { throw Refusal{PHX_MC_INTERNAL_ERROR, -1}; }

  static std::int32_t plane_row(std::int32_t reference) {
    return reference > 0 ? reference - 1 : -(reference + 1);
  }
  bool plane_reference_valid(std::int64_t reference) const {
    return reference != 0 && reference <= in_.plane_count && -reference <= in_.plane_count;
  }

  // ------------------------------------------------ structure (no arithmetic)

  void validate_scalars(std::int64_t rows, std::int64_t word_count, const std::uint32_t* words,
                        const std::int64_t* offsets, const std::int8_t* signs,
                        std::int64_t category) {
    record_.category = category;
    for (std::int64_t scalar = 0; scalar < 4 * rows; ++scalar) {
      record_.row = scalar / 4;
      const std::int64_t begin = offsets[scalar], end = offsets[scalar + 1];
      const std::int8_t sign = signs[scalar];
      if (begin < 0 || end < begin || end > word_count || sign < -1 || sign > 1 ||
          (end == begin) != (sign == 0) || (end > begin && words[end - 1] == 0))
        refuse(category, scalar / 4);
    }
  }

  void validate_topology() {
    const auto& r = in_;
    validate_scalars(r.point_count, r.point_word_count, r.point_words, r.point_offsets,
                     r.point_signs, kArrangementPointFailure);
    validate_scalars(r.plane_count, r.plane_word_count, r.plane_words, r.plane_offsets,
                     r.plane_signs, kArrangementPlaneFailure);
    record_.category = kArrangementTetrahedronFailure;
    for (std::int64_t tetrahedron = 0; tetrahedron < r.tetrahedron_count; ++tetrahedron) {
      record_.row = tetrahedron;
      for (int corner = 0; corner < 4; ++corner) {
        const std::int64_t point = r.tetrahedra[4 * tetrahedron + corner];
        if (point < 0 || point >= r.point_count ||
            !plane_reference_valid(r.tetrahedron_planes[4 * tetrahedron + corner]))
          refuse(kArrangementTetrahedronFailure, tetrahedron);
        for (int other = 0; other < corner; ++other)
          if (r.tetrahedra[4 * tetrahedron + other] == point)
            refuse(kArrangementTetrahedronFailure, tetrahedron);
      }
    }
    // Inherited chain length per job: 4 base faces plus every cut along the
    // immutable parent chain, bounded by the reference cell capacity.
    NativeVector<std::uint8_t> chain(static_cast<std::size_t>(r.job_count), 0);
    for (std::int64_t job = 0; job < r.job_count; ++job) {
      record_.category = kArrangementJobFailure;
      record_.row = job;
      const std::int64_t parent = r.parents[job];
      std::int64_t base = 4;
      if (parent < 0) {
        if (-parent - 1 >= r.tetrahedron_count) refuse(kArrangementJobFailure, job);
      } else {
        if (parent >= job) refuse(kArrangementJobFailure, job);
        base = chain[static_cast<std::size_t>(parent)];
      }
      const std::int64_t begin = r.cut_offsets[job], end = r.cut_offsets[job + 1];
      if (begin < 0 || end < begin || end > r.cut_count) refuse(kArrangementCutFailure, job);
      if (end - begin > kReferenceTetraMaximumPlanes - base) refuse(kArrangementJobFailure, job);
      record_.category = kArrangementCutFailure;
      for (std::int64_t cut = begin; cut < end; ++cut)
        if (!plane_reference_valid(r.cut_planes[cut])) refuse(kArrangementCutFailure, job);
      chain[static_cast<std::size_t>(job)] = static_cast<std::uint8_t>(base + (end - begin));
    }
  }

  // ------------------------------------------------------- exact pool owner

  Dyadic decode(const std::uint32_t* words, const std::int64_t* offsets,
                const std::int8_t* signs, std::int64_t scalar) {
    const std::int64_t begin = offsets[scalar], end = offsets[scalar + 1];
    charge_(end - begin);
    return Dyadic::integer(words == nullptr ? nullptr : words + begin,
                           static_cast<std::size_t>(end - begin), signs[scalar]);
  }

  void decode_planes() {
    record_.category = kArrangementPlaneFailure;
    planes_.resize(static_cast<std::size_t>(in_.plane_count));
    for (std::int64_t plane = 0; plane < in_.plane_count; ++plane) {
      record_.row = plane;
      auto& row = planes_[static_cast<std::size_t>(plane)];
      for (int axis = 0; axis < 4; ++axis)
        row[axis] = decode(in_.plane_words, in_.plane_offsets, in_.plane_signs, 4 * plane + axis);
      // A geometric plane has a nonzero normal; canonical rows lead positive.
      int leading = 0;
      for (int axis = 0; axis < 3 && leading == 0; ++axis) leading = row[axis].sign();
      if (leading <= 0) refuse(kArrangementPlaneFailure, plane);
    }
  }

  void decode_points() {
    record_.category = kArrangementPointFailure;
    point_vertex_.resize(static_cast<std::size_t>(in_.point_count));
    // Every input point is interned: one charged allocation for that count.
    welds_.reserve(static_cast<std::size_t>(in_.point_count), charge_);
    for (std::int64_t index = 0; index < in_.point_count; ++index) {
      record_.row = index;
      Point point;
      for (int axis = 0; axis < 4; ++axis)
        point[axis] = decode(in_.point_words, in_.point_offsets, in_.point_signs, 4 * index + axis);
      if (point[3].sign() <= 0) refuse(kArrangementPointFailure, index);
      point_vertex_[static_cast<std::size_t>(index)] =
          intern(std::move(point), static_cast<std::int32_t>(index));
    }
  }

  // Exact projective equality of two W > 0 points.
  bool same_point(const Point& a, const Point& b) {
    charge_(9);
    for (int axis = 0; axis < 3; ++axis)
      if ((a[axis] * b[3] - b[axis] * a[3]).sign() != 0) return false;
    return true;
  }

  // The stable pool id of an exact point: an earlier equal point keeps its
  // id (and its smallest input index); otherwise a new immutable id.
  std::int32_t intern(Point&& point, std::int32_t input) {
    Key key{};
    for (int axis = 0; axis < 3; ++axis) {
      double value = 0.0;
      if (point[axis].sign() != 0) {
        std::int64_t residuals = 0;
        value = exact::rounded_quotient(point[axis], point[3], {&residuals});
        charge_(2 * residuals);
      }
      key[static_cast<std::size_t>(axis)] = std::bit_cast<std::uint64_t>(value + 0.0);
    }
    const auto bucket = welds_.find(key, charge_);
    for (std::int32_t id = bucket.value == nullptr ? -1 : *bucket.value; id >= 0;
         id = weld_next_[static_cast<std::size_t>(id)]) {
      if (same_point(points_[static_cast<std::size_t>(id)], point)) {
        if (input < 0) ++counters_[kArrangementWelds];
        return id;
      }
    }
    if (points_.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
      throw ExecutionRefusal{PHX_MC_CAPACITY_EXCEEDED};
    const auto id = static_cast<std::int32_t>(points_.size());
    points_.push_back(std::move(point));
    input_index_.push_back(input);
    weld_next_.push_back(bucket.value == nullptr ? -1 : *bucket.value);
    if (bucket.value != nullptr) {
      *bucket.value = id;
    } else {
      welds_.insert(bucket, key, id, charge_);
    }
    return id;
  }

  // Exact sign of canonical plane `plane` at pool point `point`, once.
  int side(std::int32_t plane, std::int32_t point) {
    const Key key{static_cast<std::uint64_t>(plane), static_cast<std::uint64_t>(point), 0};
    const auto cached = sides_.find(key, charge_);
    if (cached.value != nullptr) {
      ++counters_[kArrangementSideReuses];
      return *cached.value - 1;
    }
    const int value = reference_plane_side(planes_[static_cast<std::size_t>(plane)],
                                           points_[static_cast<std::size_t>(point)], charge_);
    ++counters_[kArrangementSideEvaluations];
    sides_.insert(cached, key, value + 1, charge_);
    return value;
  }

  void validate_tetrahedra() {
    record_.category = kArrangementTetrahedronFailure;
    for (std::int64_t tetrahedron = 0; tetrahedron < in_.tetrahedron_count; ++tetrahedron) {
      record_.row = tetrahedron;
      std::array<std::int32_t, 4> corners{};
      for (int corner = 0; corner < 4; ++corner)
        corners[static_cast<std::size_t>(corner)] = point_vertex_[static_cast<std::size_t>(
            in_.tetrahedra[4 * tetrahedron + corner])];
      const Point& h0 = points_[static_cast<std::size_t>(corners[0])];
      const Point& h1 = points_[static_cast<std::size_t>(corners[1])];
      const Point& h2 = points_[static_cast<std::size_t>(corners[2])];
      const Point& h3 = points_[static_cast<std::size_t>(corners[3])];
      // det[h0; h1; h2; h3] = -W0 W1 W2 W3 det(p1-p0, p2-p0, p3-p0).
      charge_(4 * 18 + 7);
      const Dyadic determinant = h0[0] * reference_minor3(h1, h2, h3, 1, 2, 3) -
                                 h0[1] * reference_minor3(h1, h2, h3, 0, 2, 3) +
                                 h0[2] * reference_minor3(h1, h2, h3, 0, 1, 3) -
                                 h0[3] * reference_minor3(h1, h2, h3, 0, 1, 2);
      if (determinant.sign() >= 0) refuse(kArrangementTetrahedronFailure, tetrahedron);
      for (int face = 0; face < 4; ++face) {
        const std::int32_t reference = in_.tetrahedron_planes[4 * tetrahedron + face];
        const int orientation = reference > 0 ? 1 : -1;
        for (int corner = 0; corner < 4; ++corner) {
          const int value = orientation * side(plane_row(reference),
                                               corners[static_cast<std::size_t>(corner)]);
          if (value != (corner == face ? -1 : 0))
            refuse(kArrangementTetrahedronFailure, tetrahedron);
        }
      }
    }
  }

  // ------------------------------------------------------------------ jobs

  void load(std::int32_t parent, const TetrahedraArrangementBanks& banks, Cell& cell) const {
    if (parent < 0) {
      const std::int64_t tetrahedron = -static_cast<std::int64_t>(parent) - 1;
      for (int corner = 0; corner < 4; ++corner) {
        cell.points[static_cast<std::size_t>(corner)] = point_vertex_[static_cast<std::size_t>(
            in_.tetrahedra[4 * tetrahedron + corner])];
        // Slot k is the face opposite corner k; reversing the last two
        // entries of the positive face order makes the loop outward.
        const int* order = topology::kTetFaceOrder[corner];
        auto& face = cell.faces[static_cast<std::size_t>(corner)];
        face.plane = in_.tetrahedron_planes[4 * tetrahedron + corner];
        face.count = 3;
        face.vertices[0] = order[0];
        face.vertices[1] = order[2];
        face.vertices[2] = order[1];
      }
      cell.used = 4;
      cell.face_used = 4;
      return;
    }
    const auto job = static_cast<std::size_t>(parent);
    const std::int64_t vertex_begin = job_vertex_offsets_[job];
    cell.used = static_cast<int>(job_vertex_offsets_[job + 1] - vertex_begin);
    for (int vertex = 0; vertex < cell.used; ++vertex)
      cell.points[static_cast<std::size_t>(vertex)] =
          job_vertices_[static_cast<std::size_t>(vertex_begin + vertex)];
    const std::int64_t face_begin = banks.cell_face_offsets[job];
    cell.face_used = static_cast<int>(banks.cell_face_offsets[job + 1] - face_begin);
    for (int face = 0; face < cell.face_used; ++face) {
      const auto row = static_cast<std::size_t>(face_begin + face);
      auto& loop = cell.faces[static_cast<std::size_t>(face)];
      const std::int64_t begin = banks.face_offsets[row], end = banks.face_offsets[row + 1];
      loop.plane = banks.face_planes[row];
      loop.count = static_cast<int>(end - begin);
      for (std::int64_t entry = begin; entry < end; ++entry)
        loop.vertices[static_cast<std::size_t>(entry - begin)] =
            banks.face_vertices[static_cast<std::size_t>(entry)];
    }
  }

  // Strict crossing of edge (first, second) of facet `face` with canonical
  // plane `plane`, from the pool or as the Cramer point of the edge's two
  // facets. Facet incidence of the current cell is built once, on first need.
  std::int32_t crossing_point(const Cell& cell, int face, std::int32_t first,
                              std::int32_t second, std::int32_t plane,
                              std::array<std::uint16_t, kCellVertices>& facets,
                              bool& facets_ready) {
    const std::int32_t a = cell.points[static_cast<std::size_t>(first)];
    const std::int32_t b = cell.points[static_cast<std::size_t>(second)];
    const Key edge{static_cast<std::uint64_t>(std::min(a, b)),
                   static_cast<std::uint64_t>(std::max(a, b)), static_cast<std::uint64_t>(plane)};
    const auto edge_probe = edges_.find(edge, charge_);
    if (edge_probe.value != nullptr) {
      ++counters_[kArrangementConstructionReuses];
      return *edge_probe.value;
    }
    if (!facets_ready) {
      for (int index = 0; index < cell.face_used; ++index) {
        const auto& loop = cell.faces[static_cast<std::size_t>(index)];
        for (int slot = 0; slot < loop.count; ++slot) {
          charge_(1);
          facets[static_cast<std::size_t>(loop.vertices[static_cast<std::size_t>(slot)])] |=
              static_cast<std::uint16_t>(1u << index);
        }
      }
      facets_ready = true;
    }
    // Both ends lie on exactly the edge's two facets: this one and `other`.
    const auto common = static_cast<std::uint16_t>(
        facets[static_cast<std::size_t>(first)] & facets[static_cast<std::size_t>(second)] &
        ~(1u << face));
    if (common == 0 || (common & (common - 1)) != 0) defect();
    const int other = std::countr_zero(common);
    std::array<std::int32_t, 3> support{
        plane_row(cell.faces[static_cast<std::size_t>(face)].plane),
        plane_row(cell.faces[static_cast<std::size_t>(other)].plane), plane};
    if (support[0] == support[1] || support[0] == plane || support[1] == plane) defect();
    std::array<std::int32_t, 3> sorted = support;
    std::sort(sorted.begin(), sorted.end());
    const Key triple{static_cast<std::uint64_t>(sorted[0]), static_cast<std::uint64_t>(sorted[1]),
                     static_cast<std::uint64_t>(sorted[2])};
    const auto triple_probe = triples_.find(triple, charge_);
    if (triple_probe.value != nullptr) {
      const std::int32_t id = *triple_probe.value;
      ++counters_[kArrangementConstructionReuses];
      edges_.insert(edge_probe, edge, id, charge_);
      return id;
    }
    // The edge's facet pair minors are prepared once per call (6 products of
    // differences); each cut then needs only three-term combinations.
    const auto& pair = pair_minors(std::min(support[0], support[1]),
                                   std::max(support[0], support[1]));
    const Point& c = planes_[static_cast<std::size_t>(plane)];
    Dyadic denominator = reference_pair_denominator(pair, c, charge_);
    if (denominator.sign() == 0) defect();
    Point point = reference_pair_point(pair, c, std::move(denominator), charge_);
    ++counters_[kArrangementConstructions];
    const std::int32_t id = intern(std::move(point), -1);
    triples_.insert(triple_probe, triple, id, charge_);
    edges_.insert(edge_probe, edge, id, charge_);
    return id;
  }

  // Pair minors of canonical rows low < high, computed once per call.
  const std::array<Dyadic, 6>& pair_minors(std::int32_t low, std::int32_t high) {
    const Key key{static_cast<std::uint64_t>(low), static_cast<std::uint64_t>(high), 0};
    const auto probe = pairs_.find(key, charge_);
    if (probe.value != nullptr) return pair_minors_[static_cast<std::size_t>(*probe.value)];
    if (pair_minors_.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
      throw ExecutionRefusal{PHX_MC_CAPACITY_EXCEEDED};
    pair_minors_.push_back(reference_pair_minors(planes_[static_cast<std::size_t>(low)],
                                                 planes_[static_cast<std::size_t>(high)], charge_));
    pairs_.insert(probe, key, static_cast<std::int32_t>(pair_minors_.size() - 1), charge_);
    return pair_minors_.back();
  }

  using Sides = std::array<int, kCellVertices>;

  // Signed exact sides of one cut at pool points `points[0..used)`; true when
  // some point is strictly outside and some strictly inside.
  bool cut_sides(std::int32_t reference, const std::int32_t* points, int used, Sides& sides,
                 bool& inside) {
    const std::int32_t plane = plane_row(reference);
    const int orientation = reference > 0 ? 1 : -1;
    bool outside = false;
    inside = false;
    for (int vertex = 0; vertex < used; ++vertex) {
      const auto row = static_cast<std::size_t>(vertex);
      sides[row] = orientation * side(plane, points[row]);
      outside |= sides[row] > 0;
      inside |= sides[row] < 0;
    }
    return outside && inside;
  }

  // Applies one signed cut. False: the cell is empty or meets the closed
  // halfspace only in its boundary (no published measure). `known` holds
  // this cut's sides at the current vertices when the caller already has them.
  bool section(std::int32_t reference, Cell& cell, Cell& next, const Sides* known = nullptr) {
    const std::int32_t plane = plane_row(reference);
    Sides sides{};
    if (known != nullptr) {
      sides = *known;
    } else {
      bool inside = false;
      // No strict outside: unchanged; no strict inside: no measure remains.
      if (!cut_sides(reference, cell.points.data(), cell.used, sides, inside)) return inside;
    }
    ++counters_[kArrangementSections];
    std::array<std::int32_t, kCellVertices> retained;
    retained.fill(-1);
    std::array<std::uint8_t, kCellVertices> on_plane{};
    std::array<std::uint16_t, kCellVertices> facets{};
    bool facets_ready = false;
    struct EdgeCrossing { std::int32_t first, second, vertex; };
    std::array<EdgeCrossing, 3 * kReferenceTetraMaximumPlanes - 6> crossings;
    int crossing_count = 0, next_used = 0, next_face_used = 0;
    auto keep = [&](std::int32_t old_vertex) -> std::int32_t {
      const auto old_row = static_cast<std::size_t>(old_vertex);
      if (retained[old_row] >= 0) return retained[old_row];
      if (next_used == kCellVertices) defect();
      retained[old_row] = next_used;
      next.points[static_cast<std::size_t>(next_used)] = cell.points[old_row];
      on_plane[static_cast<std::size_t>(next_used)] = sides[old_row] == 0;
      return static_cast<std::int32_t>(next_used++);
    };
    auto crossing = [&](int face, std::int32_t first, std::int32_t second) -> std::int32_t {
      if (sides[static_cast<std::size_t>(first)] == 0) return keep(first);
      if (sides[static_cast<std::size_t>(second)] == 0) return keep(second);
      const std::int32_t low = std::min(first, second), high = std::max(first, second);
      for (int row = 0; row < crossing_count; ++row) {
        charge_(1);
        const auto& known = crossings[static_cast<std::size_t>(row)];
        if (known.first == low && known.second == high) return known.vertex;
      }
      if (next_used == kCellVertices || crossing_count == static_cast<int>(crossings.size()))
        defect();
      const std::int32_t point =
          crossing_point(cell, face, first, second, plane, facets, facets_ready);
      const auto row = next_used++;
      next.points[static_cast<std::size_t>(row)] = point;
      on_plane[static_cast<std::size_t>(row)] = 1;
      crossings[static_cast<std::size_t>(crossing_count++)] = {low, high, row};
      return row;
    };
    if (!reference_section(
            cell.faces, cell.face_used, sides, keep, crossing,
            [&](std::int32_t vertex) { return on_plane[static_cast<std::size_t>(vertex)] != 0; },
            next_used, kCellFaces, reference, next.faces, next_face_used, charge_))
      defect();
    next.used = next_used;
    next.face_used = next_face_used;
    std::swap(cell, next);
    return true;
  }

  // Appends one completed job; only facet-used rows are retained.
  void commit(bool empty, const Cell& cell, TetrahedraArrangementBanks& banks) {
    if (!empty) {
      if (cell.face_used < 4) defect();
      std::array<std::int32_t, kCellVertices> rows;
      rows.fill(-1);
      for (int face = 0; face < cell.face_used; ++face) {
        const auto& loop = cell.faces[static_cast<std::size_t>(face)];
        for (int slot = 0; slot < loop.count; ++slot)
          rows[static_cast<std::size_t>(loop.vertices[static_cast<std::size_t>(slot)])] = 0;
      }
      std::int32_t count = 0;
      for (int vertex = 0; vertex < cell.used; ++vertex) {
        const auto row = static_cast<std::size_t>(vertex);
        if (rows[row] < 0) continue;
        rows[row] = count++;
        job_vertices_.push_back(cell.points[row]);
      }
      for (int face = 0; face < cell.face_used; ++face) {
        const auto& loop = cell.faces[static_cast<std::size_t>(face)];
        banks.face_planes.push_back(loop.plane);
        for (int slot = 0; slot < loop.count; ++slot)
          banks.face_vertices.push_back(
              rows[static_cast<std::size_t>(loop.vertices[static_cast<std::size_t>(slot)])]);
        banks.face_offsets.push_back(static_cast<std::int64_t>(banks.face_vertices.size()));
      }
    }
    job_vertex_offsets_.push_back(static_cast<std::int64_t>(job_vertices_.size()));
    banks.cell_face_offsets.push_back(static_cast<std::int64_t>(banks.face_planes.size()));
  }

  using Planes = std::array<std::int32_t, kCellFaces>;

  // Pool vertices and signed facet planes of one job operand, read in place:
  // an initial tetrahedron (its corners resolved into `corners`) or a
  // completed nonempty job, whose loaded rows keep this vertex order.
  struct Operand {
    const std::int32_t* points;
    int used;
    const std::int32_t* planes;
    int faces;
  };

  Operand operand(std::int32_t parent, const TetrahedraArrangementBanks& banks,
                  std::array<std::int32_t, 4>& corners) const {
    if (parent < 0) {
      const std::int64_t tetrahedron = -static_cast<std::int64_t>(parent) - 1;
      for (int corner = 0; corner < 4; ++corner)
        corners[static_cast<std::size_t>(corner)] = point_vertex_[static_cast<std::size_t>(
            in_.tetrahedra[4 * tetrahedron + corner])];
      return {corners.data(), 4, in_.tetrahedron_planes + 4 * tetrahedron, 4};
    }
    const auto job = static_cast<std::size_t>(parent);
    const std::int64_t vertex_begin = job_vertex_offsets_[job];
    const std::int64_t face_begin = banks.cell_face_offsets[job];
    return {job_vertices_.data() + vertex_begin,
            static_cast<int>(job_vertex_offsets_[job + 1] - vertex_begin),
            banks.face_planes.data() + face_begin,
            static_cast<int>(banks.cell_face_offsets[job + 1] - face_begin)};
  }

  // Screens signed planes against immutable pool points with pooled exact
  // sides, before any copy or construction. False: some plane is
  // nonnegative at every point, so no measure remains. Otherwise `active`
  // lists, in plane order, those strictly splitting the points (a plane
  // nonpositive everywhere is redundant for every later section) and
  // `first` holds the first one's sides. The scan stops once `enough` split.
  bool screen(const std::int32_t* planes, int count, const std::int32_t* points, int used,
              int enough, Planes& active, int& active_count, Sides& first) {
    active_count = 0;
    Sides later;
    for (int index = 0; index < count && active_count < enough; ++index) {
      Sides& sides = active_count == 0 ? first : later;
      bool inside = false;
      const bool splits = cut_sides(planes[index], points, used, sides, inside);
      if (!inside) return false;
      if (splits) active[static_cast<std::size_t>(active_count++)] = planes[index];
    }
    return true;
  }

  // Sections the loaded `cell` by screened planes in order; false when no
  // measure remains.
  bool clip(Cell& cell, Cell& next, const Planes& active, int count, const Sides& first) {
    for (int index = 0; index < count; ++index)
      if (!section(active[static_cast<std::size_t>(index)], cell, next,
                   index == 0 ? &first : nullptr))
        return false;
    return true;
  }

  // Canonical exact key of four signed plane references: sorted by five
  // charged compare-exchanges, then packed losslessly.
  Key plane_set(const std::int32_t* references) {
    static constexpr std::array<std::pair<int, int>, 5> kNetwork{
        {{0, 1}, {2, 3}, {0, 2}, {1, 3}, {1, 2}}};
    std::array<std::int32_t, 4> sorted{references[0], references[1], references[2],
                                       references[3]};
    charge_(static_cast<std::int64_t>(kNetwork.size()));
    for (const auto& [low, high] : kNetwork)
      if (sorted[static_cast<std::size_t>(high)] < sorted[static_cast<std::size_t>(low)])
        std::swap(sorted[static_cast<std::size_t>(low)], sorted[static_cast<std::size_t>(high)]);
    const auto pack = [](std::int32_t high, std::int32_t low) {
      return (std::uint64_t{static_cast<std::uint32_t>(high)} << 32) |
             std::uint64_t{static_cast<std::uint32_t>(low)};
    };
    return {pack(sorted[0], sorted[1]), pack(sorted[2], sorted[3]), 0};
  }

  // The smallest declared initial tetrahedron whose outward signed face
  // references are exactly the four `cuts` as a set, or -1. Validated equal
  // sets bound one exact tetrahedron. The index is built on first need with
  // one charged allocation and a charged key and probe per tetrahedron.
  std::int32_t declared_tetrahedron(const std::int32_t* cuts) {
    if (!tetrahedra_indexed_) {
      tetrahedron_sets_.reserve(static_cast<std::size_t>(in_.tetrahedron_count), charge_);
      for (std::int64_t tetrahedron = 0; tetrahedron < in_.tetrahedron_count; ++tetrahedron) {
        const Key key = plane_set(in_.tetrahedron_planes + 4 * tetrahedron);
        const auto probe = tetrahedron_sets_.find(key, charge_);
        if (probe.value == nullptr)
          tetrahedron_sets_.insert(probe, key, static_cast<std::int32_t>(tetrahedron), charge_);
      }
      tetrahedra_indexed_ = true;
    }
    const auto probe = tetrahedron_sets_.find(plane_set(cuts), charge_);
    return probe.value == nullptr ? -1 : *probe.value;
  }

  void execute_jobs(TetrahedraArrangementBanks& banks) {
    const auto jobs = static_cast<std::size_t>(in_.job_count);
    banks.cell_face_offsets.reserve(jobs + 1);
    banks.cell_face_offsets.push_back(0);
    banks.face_offsets.push_back(0);
    job_vertex_offsets_.reserve(jobs + 1);
    job_vertex_offsets_.push_back(0);
    Cell cell, next;
    for (std::int64_t job = 0; job < in_.job_count; ++job) {
      record_.category = kArrangementExecutionFailure;
      record_.row = job;
      native_execution_primitive_query();
      ++record_.queries;
      charge_(1);
      const std::int32_t parent = in_.parents[job];
      bool empty = false;
      if (parent >= 0 && banks.cell_face_offsets[static_cast<std::size_t>(parent) + 1] ==
                             banks.cell_face_offsets[static_cast<std::size_t>(parent)]) {
        ++counters_[kArrangementPropagatedEmpty];
        empty = true;
      } else {
        std::array<std::int32_t, 4> corners;
        const Operand own = operand(parent, banks, corners);
        const std::int32_t* cuts = in_.cut_planes + in_.cut_offsets[job];
        const auto count = static_cast<int>(in_.cut_offsets[job + 1] - in_.cut_offsets[job]);
        Planes active, reverse;
        Sides first{}, reverse_first{};
        int active_count = 0, reverse_count = 0;
        std::int32_t clipped = parent;
        const Planes* planes = &active;
        const Sides* sides = &first;
        empty = !screen(cuts, count, own.points, own.used, kCellFaces, active, active_count, first);
        const std::int32_t other = !empty && count == 4 ? declared_tetrahedron(cuts) : -1;
        if (other >= 0) {
          // The parent's facets at B's corners. When no cut splits the
          // parent it lies inside B, so no facet can separate B and the
          // first splitting facet already rules out B inside the parent.
          std::array<std::int32_t, 4> other_corners;
          const std::int32_t declared = -other - 1;
          const Operand tetrahedron = operand(declared, banks, other_corners);
          empty = !screen(own.planes, own.faces, tetrahedron.points, tetrahedron.used,
                          active_count == 0 ? 1 : kCellFaces, reverse, reverse_count,
                          reverse_first);
          // B inside the parent is B whole, even when both are equal;
          // otherwise B is clipped only by strictly fewer splitting facets.
          if (!empty && (reverse_count == 0 || reverse_count < active_count)) {
            clipped = declared;
            planes = &reverse;
            sides = &reverse_first;
            active_count = reverse_count;
          }
        }
        if (!empty) {
          load(clipped, banks, cell);
          empty = !clip(cell, next, *planes, active_count, *sides);
        }
      }
      commit(empty, cell, banks);
      ++record_.completed;
    }
  }

  // ----------------------------------------------------------- publication

  void append_scalar(const Dyadic& value, TetrahedraArrangementBanks& banks) {
    if (value.sign() != 0) {
      if (value.exponent() < 0) defect();
      const Limbs limbs = shift_left(value.magnitude(), value.exponent());
      charge_(static_cast<std::int64_t>(limbs.size()));
      banks.vertex_words.insert(banks.vertex_words.end(), limbs.begin(), limbs.end());
    }
    banks.vertex_signs.push_back(static_cast<std::int8_t>(value.sign()));
    banks.vertex_word_offsets.push_back(static_cast<std::int64_t>(banks.vertex_words.size()));
  }

  // Output vertices are numbered by first use (job, face, loop order); each
  // loop then starts at its smallest output vertex.
  void publish(TetrahedraArrangementBanks& banks) {
    record_.category = kArrangementPublicationFailure;
    NativeVector<std::int32_t> output(points_.size(), -1);
    NativeVector<std::int32_t> order;
    for (std::int64_t job = 0; job < in_.job_count; ++job) {
      record_.row = job;
      const auto row = static_cast<std::size_t>(job);
      const std::int64_t vertex_begin = job_vertex_offsets_[row];
      for (std::int64_t face = banks.cell_face_offsets[row]; face < banks.cell_face_offsets[row + 1];
           ++face) {
        const auto begin = banks.face_vertices.begin() + banks.face_offsets[static_cast<std::size_t>(face)];
        const auto end = banks.face_vertices.begin() + banks.face_offsets[static_cast<std::size_t>(face) + 1];
        for (auto entry = begin; entry != end; ++entry) {
          charge_(1);
          const std::int32_t point = job_vertices_[static_cast<std::size_t>(vertex_begin + *entry)];
          auto& id = output[static_cast<std::size_t>(point)];
          if (id < 0) {
            id = static_cast<std::int32_t>(order.size());
            order.push_back(point);
          }
          *entry = id;
        }
        std::rotate(begin, std::min_element(begin, end), end);
      }
    }
    record_.row = -1;
    banks.vertex_word_offsets.reserve(4 * order.size() + 1);
    banks.vertex_signs.reserve(4 * order.size());
    banks.vertex_input_index.reserve(order.size());
    banks.vertex_word_offsets.push_back(0);
    for (const std::int32_t point : order) {
      const Point& value = points_[static_cast<std::size_t>(point)];
      for (int axis = 0; axis < 4; ++axis) append_scalar(value[static_cast<std::size_t>(axis)], banks);
      banks.vertex_input_index.push_back(input_index_[static_cast<std::size_t>(point)]);
    }
    counters_[kArrangementPoolVertices] = static_cast<std::int64_t>(points_.size());
    banks.counters = counters_;
  }

  const phx_mc_arrange_tetrahedra_exact_request& in_;
  TetrahedraArrangementRecord& record_;
  ArrangementWork charge_;
  NativeVector<Point> planes_;
  NativeDeque<Point> points_;
  NativeVector<std::int32_t> input_index_;
  NativeVector<std::int32_t> weld_next_;
  NativeVector<std::int32_t> point_vertex_;
  ExactKeyTable welds_, sides_, edges_, triples_, pairs_, tetrahedron_sets_;
  bool tetrahedra_indexed_ = false;
  NativeDeque<std::array<Dyadic, 6>> pair_minors_;
  NativeVector<std::int64_t> job_vertex_offsets_;
  NativeVector<std::int32_t> job_vertices_;
  std::array<std::int64_t, PHX_MC_TETRAHEDRA_ARRANGEMENT_COUNTERS> counters_{};
};

// Runs one call inside its own scratch window. `publish` receives the banks
// only after full success and allocates the caller-owned handle inside the
// same window; the evidence row is written on every path.
template <class Publish>
int32_t arrange_tetrahedra_exact(const phx_mc_arrange_tetrahedra_exact_request* request,
                                 Publish&& publish) {
  if (request == nullptr || request->evidence == nullptr || request->result == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  const auto& r = *request;
  std::fill(r.evidence, r.evidence + PHX_MC_TETRAHEDRA_ARRANGEMENT_EVIDENCE, 0);
  r.evidence[5] = -1;
  *r.result = nullptr;
  if (!tetrahedra_arrangement_arguments_valid(r)) return PHX_MC_INVALID_ARGUMENT;
  TetrahedraArrangementRecord record;
  int32_t status = PHX_MC_OK;
  std::uint64_t peak = 0;
  {
    MemoryBudgetWindow memory(static_cast<std::size_t>(r.scratch_limit));
    MemoryScope memory_scope(memory.owner());
    try {
      TetrahedraArrangementBanks banks;
      {
        TetrahedraArrangement arrangement(r, record);
        arrangement.run(banks);
      }
      record.category = kArrangementPublicationFailure;
      record.row = -1;
      publish(std::move(banks));
      record.category = kArrangementNoFailure;
    } catch (const Refusal& refusal) {
      status = refusal.status;
    } catch (const ExecutionRefusal& refusal) {
      status = refusal.status;
    } catch (const std::bad_alloc&) {
      status = native_execution_status(PHX_MC_CAPACITY_EXCEEDED);
    } catch (...) {
      status = PHX_MC_INTERNAL_ERROR;
    }
    peak = memory.evidence()[2];
  }
  r.evidence[0] = record.work;
  r.evidence[1] = record.queries;
  r.evidence[2] = static_cast<std::int64_t>(
      std::min<std::uint64_t>(peak, static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())));
  r.evidence[3] = record.completed;
  r.evidence[4] = status == PHX_MC_OK ? kArrangementNoFailure : record.category;
  r.evidence[5] = status == PHX_MC_OK ? -1 : record.row;
  return status;
}
