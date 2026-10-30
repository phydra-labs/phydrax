// Copyright © 2026 Phydra, Inc. All rights reserved.
// Deterministic multilevel k-way partitioning of a symmetric weighted CSR graph.
//
// Coarsening uses seeded heavy-edge matching; recursive bisection is projected
// to the input graph before exact capacity and cut comparison. Refinement combines
// overload shedding, weighted packing, greedy/FM moves, pair exchanges, and
// restricted multiway V-cycles. Every tie is resolved from canonical input order
// and fixed integer ranks, so output is deterministic across worker configurations.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "phydrax_meshcore.h"

namespace {

constexpr int64_t kWeightLimit = int64_t{1} << 53;
constexpr int64_t kNoMove = std::numeric_limits<int64_t>::min();
// Initial k-way partitions of the coarsest graph, restricted V-cycles after
// the first uncoarsening, grown trials per coarsest bisection, and the vertex
// count each bisection coarsens its subgraph to.
constexpr int32_t kInitialPartitions = 8;
constexpr int32_t kVCycles = 2;
constexpr int32_t kBisectionSeeds = 8;
constexpr int64_t kBisectionCoarsenTo = 64;


struct GraphView {
  int32_t n;
  const int64_t* xadj;
  const int32_t* adj;
  const int64_t* ew;
  const int64_t* vw;
};

struct OwnedGraph {
  std::vector<int64_t> xadj;
  std::vector<int32_t> adj;
  std::vector<int64_t> ew;
  std::vector<int64_t> vw;

  GraphView view() const {
    return {static_cast<int32_t>(vw.size()), xadj.data(), adj.data(), ew.data(), vw.data()};
  }
};

struct Balance {
  int32_t k;
  const int64_t* capacity;
  bool require_nonempty;
};

struct Partition {
  std::vector<int32_t> part;
  std::vector<int64_t> weight;
  std::vector<int64_t> count;

  void assign(const GraphView& g, int32_t k) {
    weight.assign(k, 0);
    count.assign(k, 0);
    for (int32_t v = 0; v < g.n; ++v) {
      weight[part[v]] += g.vw[v];
      ++count[part[v]];
    }
  }

  void move(const GraphView& g, int32_t v, int32_t to) {
    const int32_t from = part[v];
    weight[from] -= g.vw[v];
    --count[from];
    weight[to] += g.vw[v];
    ++count[to];
    part[v] = to;
  }
};

// SplitMix64 finalizer: a fixed integer bijection, identical on every platform.
uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

uint32_t rank_of(uint64_t seed, int32_t v) {
  return static_cast<uint32_t>(mix(mix(seed) ^ static_cast<uint32_t>(v)));
}

std::vector<uint32_t> ranks(int32_t n, uint64_t seed) {
  std::vector<uint32_t> rank(n);
  for (int32_t v = 0; v < n; ++v) {
    rank[v] = rank_of(seed, v);
  }
  return rank;
}

int64_t total_vertex_weight(const GraphView& g) {
  int64_t total = 0;
  for (int32_t v = 0; v < g.n; ++v) {
    total += g.vw[v];
  }
  return total;
}

// ------------------------------------------------------------------ coarsening

struct Coarsening {
  int64_t max_weight;
  uint64_t seed;
  // Matching stays inside these parts when set.
  const int32_t* part;

  bool compatible(const GraphView& g, int32_t u, int32_t v) const {
    return g.vw[u] + g.vw[v] <= max_weight && (part == nullptr || part[u] == part[v]);
  }
};

// Pairs still unmatched vertices that share a neighbor: every hub pairs its
// unmatched neighbors in row order.
void match_two_hop(const GraphView& g, const Coarsening& c, std::vector<int32_t>& mate) {
  for (int32_t hub = 0; hub < g.n; ++hub) {
    int32_t pending = -1;
    for (int64_t e = g.xadj[hub]; e < g.xadj[hub + 1]; ++e) {
      const int32_t u = g.adj[e];
      if (mate[u] != u) {
        continue;
      }
      if (pending >= 0 && c.compatible(g, pending, u)) {
        mate[pending] = u;
        mate[u] = pending;
        pending = -1;
      } else {
        pending = u;
      }
    }
  }
}

// Heavy-edge matching; returns the coarse vertex count and fills cmap.
int32_t match(const GraphView& g, const Coarsening& c, std::vector<int32_t>& cmap) {
  const int32_t n = g.n;
  // Low-degree vertices choose first; seeded ranks break index stripes.
  std::vector<std::pair<uint64_t, int32_t>> order(n);
  for (int32_t v = 0; v < n; ++v) {
    const auto degree = static_cast<uint64_t>(g.xadj[v + 1] - g.xadj[v]);
    order[v] = {(degree << 32) | rank_of(c.seed, v), v};
  }
  std::sort(order.begin(), order.end());
  std::vector<int32_t> mate(n, -1);
  int64_t unmatched = 0;
  for (const auto& entry : order) {
    const int32_t u = entry.second;
    if (mate[u] != -1) {
      continue;
    }
    int32_t best = -1;
    int64_t best_edge = -1;
    int64_t best_merged = 0;
    for (int64_t e = g.xadj[u]; e < g.xadj[u + 1]; ++e) {
      const int32_t v = g.adj[e];
      if (mate[v] != -1 || !c.compatible(g, u, v)) {
        continue;
      }
      const int64_t merged = g.vw[u] + g.vw[v];
      if (g.ew[e] > best_edge ||
          (g.ew[e] == best_edge &&
           (merged < best_merged ||
            (merged == best_merged &&
             (best < 0 || rank_of(c.seed, v) < rank_of(c.seed, best) ||
              (rank_of(c.seed, v) == rank_of(c.seed, best) && v < best)))))) {
        best = v;
        best_edge = g.ew[e];
        best_merged = merged;
      }
    }
    if (best >= 0) {
      mate[u] = best;
      mate[best] = u;
    } else {
      mate[u] = u;
      ++unmatched;
    }
  }
  if (10 * unmatched > n) {
    match_two_hop(g, c, mate);
  }
  // Isolated vertices have no edge to match along; pair them in index order so
  // disconnected singletons also coarsen.
  int32_t pending = -1;
  for (int32_t v = 0; v < n; ++v) {
    if (g.xadj[v + 1] != g.xadj[v] || mate[v] != v) {
      continue;
    }
    if (pending >= 0 && c.compatible(g, pending, v)) {
      mate[pending] = v;
      mate[v] = pending;
      pending = -1;
    } else {
      pending = v;
    }
  }
  cmap.assign(n, -1);
  int32_t coarse = 0;
  for (int32_t v = 0; v < n; ++v) {
    if (mate[v] >= v) {
      cmap[v] = coarse;
      cmap[mate[v]] = coarse;
      ++coarse;
    }
  }
  return coarse;
}

OwnedGraph contract(const GraphView& g, const std::vector<int32_t>& cmap, int32_t coarse) {
  std::vector<int32_t> first(coarse, -1);
  std::vector<int32_t> second(coarse, -1);
  for (int32_t v = 0; v < g.n; ++v) {
    (first[cmap[v]] < 0 ? first[cmap[v]] : second[cmap[v]]) = v;
  }
  OwnedGraph out;
  out.xadj.assign(static_cast<std::size_t>(coarse) + 1, 0);
  out.vw.assign(coarse, 0);
  out.adj.reserve(static_cast<std::size_t>(g.xadj[g.n]));
  out.ew.reserve(static_cast<std::size_t>(g.xadj[g.n]));
  std::vector<int64_t> slot(coarse, -1);
  std::vector<std::pair<int32_t, int64_t>> row;
  for (int32_t c = 0; c < coarse; ++c) {
    row.clear();
    for (const int32_t u : {first[c], second[c]}) {
      if (u < 0) {
        continue;
      }
      out.vw[c] += g.vw[u];
      for (int64_t e = g.xadj[u]; e < g.xadj[u + 1]; ++e) {
        const int32_t target = cmap[g.adj[e]];
        if (target == c) {
          continue;
        }
        if (slot[target] < 0) {
          slot[target] = static_cast<int64_t>(row.size());
          row.emplace_back(target, g.ew[e]);
        } else {
          row[slot[target]].second += g.ew[e];
        }
      }
    }
    std::sort(row.begin(), row.end());
    for (const auto& [target, weight] : row) {
      slot[target] = -1;
      out.adj.push_back(target);
      out.ew.push_back(weight);
    }
    out.xadj[c + 1] = static_cast<int64_t>(out.adj.size());
  }
  return out;
}

// Coarse graphs of one multilevel cycle; maps[i] sends level i to level i + 1
// and level 0 is the caller's graph.
struct Hierarchy {
  std::vector<OwnedGraph> graphs;
  std::vector<std::vector<int32_t>> maps;

  std::size_t depth() const { return graphs.size(); }

  GraphView level(const GraphView& input, std::size_t index) const {
    return index == 0 ? input : graphs[index - 1].view();
  }
};

// Contracts matchings until at most `coarsen_to` vertices remain or a level
// shrinks by less than 5%.  With `part` set, matching never crosses parts and
// `part` is replaced by its projection onto the coarsest level.
Hierarchy coarsen(const GraphView& input, int64_t coarsen_to, int64_t max_weight, uint64_t seed,
                  std::vector<int32_t>* part) {
  Hierarchy h;
  GraphView current = input;
  while (current.n > coarsen_to) {
    std::vector<int32_t> cmap;
    const Coarsening rule{max_weight, mix(seed + h.depth()),
                          part == nullptr ? nullptr : part->data()};
    const int32_t coarse = match(current, rule, cmap);
    if (int64_t{20} * coarse > int64_t{19} * current.n) {
      break;
    }
    h.graphs.push_back(contract(current, cmap, coarse));
    if (part != nullptr) {
      std::vector<int32_t> projected(coarse);
      for (int32_t v = 0; v < current.n; ++v) {
        projected[cmap[v]] = (*part)[v];
      }
      *part = std::move(projected);
    }
    h.maps.push_back(std::move(cmap));
    current = h.graphs.back().view();
  }
  return h;
}

// Exact part capacities and their per-level relaxation: coarse levels add
// their heaviest merged vertex so refinement can move at coarse granularity;
// the input level is exact.
struct CapacityTable {
  int32_t k;
  const int64_t* exact;
  bool require_nonempty;
  int64_t max_weight;
  std::vector<int64_t> relaxed;

  Balance at(const GraphView& g, bool coarse) {
    int64_t slack = 0;
    for (int32_t v = 0; coarse && v < g.n; ++v) {
      slack = std::max(slack, std::min(g.vw[v], max_weight));
    }
    relaxed.resize(k);
    for (int32_t q = 0; q < k; ++q) {
      relaxed[q] = exact[q] + slack;
    }
    return {k, relaxed.data(), require_nonempty};
  }
};

// ------------------------------------------------------------- refinement

// Accumulates weighted connectivity of one vertex to every adjacent part; a
// zero-weight edge still makes its endpoint part adjacent.
struct Connectivity {
  std::vector<int64_t> to_part;
  std::vector<char> marked;
  std::vector<int32_t> touched;

  explicit Connectivity(int32_t k) : to_part(k, 0), marked(k, 0) {}

  void gather(const GraphView& g, const Partition& p, int32_t v) {
    for (int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      const int32_t q = p.part[g.adj[e]];
      if (!marked[q]) {
        marked[q] = 1;
        touched.push_back(q);
      }
      to_part[q] += g.ew[e];
    }
  }

  void clear() {
    for (const int32_t q : touched) {
      to_part[q] = 0;
      marked[q] = 0;
    }
    touched.clear();
  }
};

bool may_leave(const Partition& p, const Balance& b, int32_t from) {
  return !b.require_nonempty || p.count[from] > 1;
}

int64_t room(const Partition& p, const Balance& b, int32_t q) {
  return b.capacity[q] - p.weight[q];
}

bool fits(const GraphView& g, const Partition& p, const Balance& b, int32_t v, int32_t to) {
  return g.vw[v] <= room(p, b, to);
}

struct Move {
  int32_t to = -1;
  int64_t gain = kNoMove;
};

// Best adjacent move by gain, remaining room, then part index. Optimistic
// heap keys omit capacity/nonempty restrictions; committed moves enforce both.
Move best_adjacent_move(const GraphView& g, const Partition& p, const Balance& b,
                        Connectivity& conn, int32_t v,
                        bool enforce_capacity = true) {
  Move best;
  const int32_t from = p.part[v];
  if (enforce_capacity && !may_leave(p, b, from)) {
    return best;
  }
  conn.gather(g, p, v);
  const int64_t internal = conn.to_part[from];
  int64_t best_room = 0;
  for (const int32_t q : conn.touched) {
    if (q == from || (enforce_capacity && !fits(g, p, b, v, q))) {
      continue;
    }
    const int64_t gain = conn.to_part[q] - internal;
    const int64_t space = room(p, b, q);
    if (gain > best.gain || (gain == best.gain && (space > best_room ||
                                                   (space == best_room && q < best.to)))) {
      best = {q, gain};
      best_room = space;
    }
  }
  conn.clear();
  return best;
}

struct Candidate {
  int64_t gain;
  int32_t v;
};

// Moves vertices out of parts above capacity: adjacent destinations by gain
// first, otherwise the part with the most room.  A part whose remaining
// vertices fit nowhere stays overloaded and is reported by the caller.
void shed_overload(const GraphView& g, Partition& p, const Balance& b) {
  std::vector<char> overloaded(b.k, 0);
  bool any = false;
  for (int32_t q = 0; q < b.k; ++q) {
    overloaded[q] = p.weight[q] > b.capacity[q];
    any = any || overloaded[q];
  }
  if (!any) {
    return;
  }
  std::vector<std::vector<int32_t>> members(b.k);
  for (int32_t v = 0; v < g.n; ++v) {
    if (overloaded[p.part[v]]) {
      members[p.part[v]].push_back(v);
    }
  }
  Connectivity conn(b.k);
  std::vector<Candidate> candidates;
  for (int32_t from = 0; from < b.k; ++from) {
    if (!overloaded[from]) {
      continue;
    }
    candidates.clear();
    for (const int32_t v : members[from]) {
      conn.gather(g, p, v);
      int64_t external = std::numeric_limits<int64_t>::min();
      for (const int32_t q : conn.touched) {
        if (q != from) {
          external = std::max(external, conn.to_part[q]);
        }
      }
      const int64_t internal = conn.to_part[from];
      conn.clear();
      candidates.push_back(
          {external == std::numeric_limits<int64_t>::min() ? -internal : external - internal,
           v});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& c) {
      return a.gain > c.gain || (a.gain == c.gain && a.v < c.v);
    });
    for (const Candidate& candidate : candidates) {
      if (p.weight[from] <= b.capacity[from] || !may_leave(p, b, from)) {
        break;
      }
      const int32_t v = candidate.v;
      Move move = best_adjacent_move(g, p, b, conn, v);
      if (move.to < 0) {
        int64_t most = -1;
        for (int32_t q = 0; q < b.k; ++q) {
          if (q != from && fits(g, p, b, v, q) && room(p, b, q) > most) {
            most = room(p, b, q);
            move.to = q;
          }
        }
      }
      if (move.to >= 0) {
        p.move(g, v, move.to);
      }
    }
  }
}

// Weighted/disconnected packing repair: at most 16 light representatives per
// part and 16 rounds. Destination capacity stays exact; source overload must
// strictly decrease, so multiple exchanges can remove a packing obstruction.
void repair_packing(const GraphView& g, Partition& p, const Balance& b) {
  bool any = false;
  for (int32_t q = 0; q < b.k; ++q) {
    any = any || p.weight[q] > b.capacity[q];
  }
  if (!any) {
    return;
  }
  std::vector<int32_t> order(g.n);
  for (int32_t v = 0; v < g.n; ++v) {
    order[v] = v;
  }
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t c) {
    return g.vw[a] != g.vw[c] ? g.vw[a] < g.vw[c] : a < c;
  });
  for (int32_t round = 0; round < 16; ++round) {
    std::vector<std::vector<int32_t>> representatives(b.k);
    for (const int32_t v : order) {
      auto& row = representatives[p.part[v]];
      if (row.size() < 16) {
        row.push_back(v);
      }
    }
    bool changed = false;
    for (const int32_t v : order) {
      const int32_t from = p.part[v];
      if (p.weight[from] <= b.capacity[from]) {
        continue;
      }
      bool repaired = false;
      for (int32_t to = 0; to < b.k && !repaired; ++to) {
        for (const int32_t u : representatives[to]) {
          if (to == from || p.part[u] != to) {
            continue;
          }
          const int64_t difference = g.vw[u] - g.vw[v];
          if (difference < 0 && p.weight[to] - difference <= b.capacity[to]) {
            p.weight[from] += difference;
            p.weight[to] -= difference;
            std::swap(p.part[v], p.part[u]);
            changed = repaired = true;
            break;
          }
        }
      }
    }
    if (!changed) {
      break;
    }
    shed_overload(g, p, b);
  }
}

// Greedy boundary pass in rank order: a vertex moves to its best adjacent part
// when that lowers the cut or, at equal cut, leaves the destination with more
// room than the source had (so equal-cut moves never oscillate).  Returns the
// number of moves.
int64_t greedy_pass(const GraphView& g, Partition& p, const Balance& b,
                    const std::vector<int32_t>& order) {
  Connectivity conn(b.k);
  int64_t moved = 0;
  for (const int32_t v : order) {
    const Move move = best_adjacent_move(g, p, b, conn, v);
    if (move.to < 0) {
      continue;
    }
    if (move.gain > 0 ||
        (move.gain == 0 && room(p, b, move.to) - g.vw[v] > room(p, b, p.part[v]))) {
      p.move(g, v, move.to);
      ++moved;
    }
  }
  return moved;
}

// Addressable max-heap of vertex gains; ties pop the smaller seeded rank, then
// the smaller vertex.
class GainHeap {
 public:
  GainHeap(int32_t n, const std::vector<uint32_t>& rank)
      : position_(n, -1), key_(n, 0), rank_(rank) {}

  bool empty() const { return heap_.empty(); }
  int32_t top() const { return heap_.front(); }
  int64_t key(int32_t v) const { return key_[v]; }
  bool contains(int32_t v) const { return position_[v] >= 0; }

  void set(int32_t v, int64_t key) {
    if (!contains(v)) {
      key_[v] = key;
      heap_.push_back(v);
      position_[v] = static_cast<int64_t>(heap_.size()) - 1;
      up(heap_.size() - 1);
      return;
    }
    const int64_t old = key_[v];
    key_[v] = key;
    if (key > old) {
      up(static_cast<std::size_t>(position_[v]));
    } else {
      down(static_cast<std::size_t>(position_[v]));
    }
  }

  void erase(int32_t v) {
    const auto index = static_cast<std::size_t>(position_[v]);
    const int32_t last = heap_.back();
    heap_.pop_back();
    position_[v] = -1;
    if (last == v) {
      return;
    }
    place(index, last);
    up(index);
    down(static_cast<std::size_t>(position_[last]));
  }

 private:
  bool before(int32_t a, int32_t b) const {
    if (key_[a] != key_[b]) {
      return key_[a] > key_[b];
    }
    return rank_[a] != rank_[b] ? rank_[a] < rank_[b] : a < b;
  }

  void place(std::size_t index, int32_t v) {
    heap_[index] = v;
    position_[v] = static_cast<int64_t>(index);
  }

  void up(std::size_t index) {
    const int32_t v = heap_[index];
    while (index > 0 && before(v, heap_[(index - 1) / 2])) {
      place(index, heap_[(index - 1) / 2]);
      index = (index - 1) / 2;
    }
    place(index, v);
  }

  void down(std::size_t index) {
    const int32_t v = heap_[index];
    for (;;) {
      std::size_t child = 2 * index + 1;
      if (child >= heap_.size()) {
        break;
      }
      if (child + 1 < heap_.size() && before(heap_[child + 1], heap_[child])) {
        ++child;
      }
      if (!before(heap_[child], v)) {
        break;
      }
      place(index, heap_[child]);
      index = child;
    }
    place(index, v);
  }

  std::vector<int32_t> heap_;
  std::vector<int64_t> position_;
  std::vector<int64_t> key_;
  const std::vector<uint32_t>& rank_;
};

// Boundary FM with optimistic gain keys and lazy feasibility rechecks.
// Negative-gain moves are allowed; each vertex moves once, and the suffix
// after the least-cut prefix is rolled back after a bounded stall.
bool fm_pass(const GraphView& g, Partition& p, const Balance& b,
             const std::vector<uint32_t>& rank) {
  const int32_t n = g.n;
  Connectivity conn(b.k);
  GainHeap heap(n, rank);
  std::vector<char> locked(n, 0);
  for (int32_t v = 0; v < n; ++v) {
    const Move move = best_adjacent_move(g, p, b, conn, v, false);
    if (move.to >= 0) {
      heap.set(v, move.gain);
    }
  }
  const int64_t stall_limit = std::clamp<int64_t>(n / 20, 32, 1024);
  std::vector<std::pair<int32_t, int32_t>> moves;
  int64_t delta = 0;
  int64_t best_delta = 0;
  std::size_t best_length = 0;
  int64_t stall = 0;
  std::vector<int32_t> blocked;
  while (!heap.empty() && stall < stall_limit) {
    const int32_t v = heap.top();
    const int64_t key = heap.key(v);
    if (locked[v]) {
      heap.erase(v);
      continue;
    }
    heap.erase(v);
    // Optimistic keys include capacity-blocked moves. Reconsider them whenever
    // another move releases room, even when the vertices are not neighbors.
    const Move move = best_adjacent_move(g, p, b, conn, v);
    if (move.to < 0) {
      blocked.push_back(v);
      continue;
    }
    if (move.gain != key) {
      heap.set(v, move.gain);
      continue;
    }
    moves.emplace_back(v, p.part[v]);
    p.move(g, v, move.to);
    locked[v] = 1;
    delta -= move.gain;
    if (delta < best_delta) {
      best_delta = delta;
      best_length = moves.size();
      stall = 0;
    } else {
      ++stall;
    }
    for (const int32_t waiting : blocked) {
      if (locked[waiting]) {
        continue;
      }
      const Move next = best_adjacent_move(g, p, b, conn, waiting, false);
      if (next.to >= 0) {
        heap.set(waiting, next.gain);
      }
    }
    blocked.clear();
    for (int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      const int32_t u = g.adj[e];
      if (locked[u]) {
        continue;
      }
      const Move next = best_adjacent_move(g, p, b, conn, u, false);
      if (next.to >= 0) {
        heap.set(u, next.gain);
      } else if (heap.contains(u)) {
        heap.erase(u);
      }
    }
  }
  for (std::size_t index = moves.size(); index > best_length; --index) {
    p.move(g, moves[index - 1].first, moves[index - 1].second);
  }
  return best_delta < 0;
}

// Exact-capacity exchanges cross barriers that no single-vertex FM move can
// cross. Each vertex participates once per pass; at most 64 distinct one/two
// hop or adjacent-part boundary representatives are considered. Counts never change.
int64_t exchange_pass(const GraphView& g, Partition& p, const Balance& b,
                      const std::vector<int32_t>& order) {
  Connectivity left(b.k), right(b.k);
  std::vector<char> locked(g.n, 0);
  std::vector<int32_t> seen(g.n, -1), candidates;
  candidates.reserve(64);
  std::vector<std::vector<int32_t>> representatives(b.k);
  for (const int32_t v : order) {
    auto& row = representatives[p.part[v]];
    if (row.size() >= 8) {
      continue;
    }
    for (int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      if (p.part[g.adj[e]] != p.part[v]) {
        row.push_back(v);
        break;
      }
    }
  }
  int64_t moved = 0;
  for (const int32_t v : order) {
    if (locked[v]) {
      continue;
    }
    const int32_t from = p.part[v];
    candidates.clear();
    const auto add = [&](int32_t u) {
      if (u != v && !locked[u] && p.part[u] != from && seen[u] != v &&
          candidates.size() < 64) {
        seen[u] = v;
        candidates.push_back(u);
      }
    };
    for (int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      add(g.adj[e]);
    }
    for (int64_t e = g.xadj[v]; e < g.xadj[v + 1] && candidates.size() < 64; ++e) {
      const int32_t hub = g.adj[e];
      for (int64_t f = g.xadj[hub]; f < g.xadj[hub + 1] && candidates.size() < 64; ++f) {
        add(g.adj[f]);
      }
    }
    left.gather(g, p, v);
    for (const int32_t to : left.touched) {
      if (to == from) {
        continue;
      }
      for (const int32_t u : representatives[to]) {
        if (p.part[u] == to) {
          add(u);
        }
      }
    }
    if (candidates.empty()) {
      left.clear();
      continue;
    }
    int32_t best = -1;
    int64_t best_gain = 0;
    for (const int32_t u : candidates) {
      const int32_t to = p.part[u];
      const int64_t difference = g.vw[u] - g.vw[v];
      if (p.weight[from] + difference > b.capacity[from] ||
          p.weight[to] - difference > b.capacity[to]) {
        continue;
      }
      right.gather(g, p, u);
      int64_t mutual = 0;
      for (int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
        if (g.adj[e] == u) {
          mutual = g.ew[e];
          break;
        }
      }
      const int64_t gain = left.to_part[to] - left.to_part[from] +
                           right.to_part[from] - right.to_part[to] - 2 * mutual;
      right.clear();
      if (gain > best_gain || (gain == best_gain && gain > 0 && u < best)) {
        best = u;
        best_gain = gain;
      }
    }
    left.clear();
    if (best >= 0) {
      const int32_t to = p.part[best];
      const int64_t difference = g.vw[best] - g.vw[v];
      p.weight[from] += difference;
      p.weight[to] -= difference;
      std::swap(p.part[v], p.part[best]);
      locked[v] = locked[best] = 1;
      moved += 2;
    }
  }
  return moved;
}

void refine(const GraphView& g, Partition& p, const Balance& b, int32_t passes, uint64_t seed) {
  shed_overload(g, p, b);
  repair_packing(g, p, b);
  const std::vector<uint32_t> rank = ranks(g.n, seed);
  std::vector<int32_t> order(g.n);
  for (int32_t v = 0; v < g.n; ++v) {
    order[v] = v;
  }
  std::sort(order.begin(), order.end(), [&rank](int32_t a, int32_t c) {
    return rank[a] != rank[c] ? rank[a] < rank[c] : a < c;
  });
  for (int32_t pass = 0; pass < passes; ++pass) {
    const int64_t greedy = greedy_pass(g, p, b, order);
    const bool improved = fm_pass(g, p, b, rank);
    const int64_t exchanged = exchange_pass(g, p, b, order);
    if (!improved && greedy == 0 && exchanged == 0) {
      break;
    }
  }
}

int64_t cut_weight(const GraphView& g, const std::vector<int32_t>& part) {
  int64_t cut = 0;
  for (int32_t v = 0; v < g.n; ++v) {
    for (int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      if (part[g.adj[e]] != part[v]) {
        cut += g.ew[e];
      }
    }
  }
  return cut / 2;
}

int64_t overload(const Partition& p, const Balance& b) {
  int64_t excess = 0;
  for (int32_t q = 0; q < b.k; ++q) {
    excess += std::max<int64_t>(0, p.weight[q] - b.capacity[q]);
  }
  return excess;
}

// Whether `candidate` has less overload, then less cut, than `incumbent`.
bool better(const GraphView& g, const Partition& candidate, const Partition& incumbent,
            const Balance& b) {
  const int64_t a = overload(candidate, b);
  const int64_t c = overload(incumbent, b);
  return a < c || (a == c && cut_weight(g, candidate.part) <
                                cut_weight(g, incumbent.part));
}

// Projects `p` from the coarsest level of `h` to its input, refining each level.
void uncoarsen(const GraphView& input, const Hierarchy& h, Partition& p, CapacityTable& caps,
               int32_t passes, uint64_t seed) {
  for (std::size_t level = h.depth(); level > 0; --level) {
    const GraphView fine = h.level(input, level - 1);
    const std::vector<int32_t>& cmap = h.maps[level - 1];
    std::vector<int32_t> projected(fine.n);
    for (int32_t v = 0; v < fine.n; ++v) {
      projected[v] = p.part[cmap[v]];
    }
    p.part = std::move(projected);
    p.assign(fine, caps.k);
    refine(fine, p, caps.at(fine, level > 1), passes, mix(seed ^ level));
  }
}

// --------------------------------------------------------- initial partition

// Last vertex reached by a breadth-first sweep of the component of `source`.
int32_t farthest(const GraphView& g, int32_t source) {
  std::vector<char> seen(g.n, 0);
  std::vector<int32_t> queue;
  queue.reserve(g.n);
  seen[source] = 1;
  queue.push_back(source);
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const int32_t v = queue[head];
    for (int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      if (!seen[g.adj[e]]) {
        seen[g.adj[e]] = 1;
        queue.push_back(g.adj[e]);
      }
    }
  }
  return queue.back();
}

// Greedy graph growing of side 0 from `seed` towards `goal` vertex weight;
// exhausted components continue from the lowest unassigned index.
std::vector<int32_t> grow(const GraphView& g, int32_t seed, int64_t goal) {
  std::vector<int32_t> side(g.n, 1);
  std::vector<int64_t> degree(g.n, 0);
  std::vector<int64_t> to_grown(g.n, 0);
  for (int32_t v = 0; v < g.n; ++v) {
    for (int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      degree[v] += g.ew[e];
    }
  }
  const std::vector<uint32_t> rank = ranks(g.n, static_cast<uint64_t>(seed));
  GainHeap frontier(g.n, rank);
  int64_t grown = 0;
  int32_t scan = 0;
  int32_t next = seed;
  while (grown < goal) {
    if (next < 0) {
      if (!frontier.empty()) {
        next = frontier.top();
      } else {
        while (scan < g.n && side[scan] == 0) {
          ++scan;
        }
        if (scan == g.n) {
          break;
        }
        next = scan;
      }
    }
    // Stop when the overshoot would exceed the remaining deficit.
    if (grown > 0 && grown + g.vw[next] - goal > goal - grown) {
      break;
    }
    if (frontier.contains(next)) {
      frontier.erase(next);
    }
    side[next] = 0;
    grown += g.vw[next];
    for (int64_t e = g.xadj[next]; e < g.xadj[next + 1]; ++e) {
      const int32_t u = g.adj[e];
      if (side[u] == 0) {
        continue;
      }
      to_grown[u] += g.ew[e];
      frontier.set(u, 2 * to_grown[u] - degree[u]);
    }
    next = -1;
  }
  return side;
}

// Best of several grown two-way splits of the coarsest bisection graph by
// (overload, cut).  Seeds: both ends of a pseudo-diameter of the component of
// a seeded vertex, then seeded vertices.
Partition grow_bisection(const GraphView& g, const int64_t goal[2], const Balance& balance,
                         int32_t passes, uint64_t seed) {
  const int32_t trials = std::min<int32_t>(g.n, kBisectionSeeds);
  const auto pick = [&](int32_t trial) {
    return static_cast<int32_t>(mix(seed + static_cast<uint64_t>(trial)) %
                                static_cast<uint64_t>(g.n));
  };
  const int32_t end = farthest(g, farthest(g, pick(0)));
  const int32_t start = farthest(g, end);
  Partition best;
  for (int32_t trial = 0; trial < trials; ++trial) {
    Partition p;
    p.part = grow(g, trial == 0 ? end : trial == 1 ? start : pick(trial), goal[0]);
    p.assign(g, 2);
    if (balance.require_nonempty && (p.count[0] == 0 || p.count[1] == 0)) {
      const int32_t empty = p.count[0] == 0 ? 0 : 1;
      int32_t lightest = 0;
      for (int32_t v = 1; v < g.n; ++v) {
        if (g.vw[v] < g.vw[lightest]) {
          lightest = v;
        }
      }
      p.move(g, lightest, empty);
    }
    refine(g, p, balance, passes, mix(seed ^ (static_cast<uint64_t>(trial) << 32)));
    if (best.part.empty() || better(g, p, best, balance)) {
      best = std::move(p);
    }
  }
  return best;
}

// Multilevel two-way split of g with side goals and capacities.
std::vector<int32_t> bisect(const GraphView& g, const int64_t goal[2], const int64_t capacity[2],
                            bool require_nonempty, int32_t passes, uint64_t seed) {
  const int64_t max_weight =
      std::max<int64_t>(1, (3 * (goal[0] + goal[1])) / (2 * kBisectionCoarsenTo));
  const Hierarchy h = coarsen(g, kBisectionCoarsenTo, max_weight, seed, nullptr);
  CapacityTable caps{2, capacity, require_nonempty && g.n >= 2, max_weight, {}};
  const GraphView coarsest = h.level(g, h.depth());
  Partition p =
      grow_bisection(coarsest, goal, caps.at(coarsest, h.depth() > 0), passes, seed);
  uncoarsen(g, h, p, caps, passes, seed);
  return std::move(p.part);
}

OwnedGraph induced(const GraphView& g, const std::vector<int32_t>& side, int32_t which,
                   std::vector<int32_t>& local, std::vector<int32_t>& members) {
  members.clear();
  for (int32_t v = 0; v < g.n; ++v) {
    if (side[v] == which) {
      local[v] = static_cast<int32_t>(members.size());
      members.push_back(v);
    }
  }
  OwnedGraph out;
  out.xadj.assign(members.size() + 1, 0);
  out.vw.reserve(members.size());
  for (std::size_t i = 0; i < members.size(); ++i) {
    const int32_t v = members[i];
    out.vw.push_back(g.vw[v]);
    for (int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      if (side[g.adj[e]] == which) {
        out.adj.push_back(local[g.adj[e]]);
        out.ew.push_back(g.ew[e]);
      }
    }
    out.xadj[i + 1] = static_cast<int64_t>(out.adj.size());
  }
  return out;
}

// No part can contain more than the total admitted graph weight, so larger
// capacity sums are equivalent and saturate at the maximum graph total.
int64_t sum(const int64_t* values, int32_t first, int32_t count) {
  int64_t total = 0;
  for (int32_t q = first; q < first + count; ++q) {
    total += std::min(values[q], kWeightLimit - total);
  }
  return total;
}

// Scales `amount * numerator / denominator` down, saturating above any graph
// weight that this implementation admits.
int64_t scaled(int64_t amount, int64_t numerator, int64_t denominator) {
  const double value = std::floor(static_cast<double>(amount) *
                                  static_cast<double>(numerator) /
                                  static_cast<double>(denominator));
  return value >= static_cast<double>(kWeightLimit) ? kWeightLimit
                                                     : static_cast<int64_t>(value);
}

struct PartTable {
  const int64_t* target;
  const int64_t* capacity;
  bool require_nonempty;
  int32_t passes;
};

void recursive_bisection(const GraphView& g, const std::vector<int32_t>& ids, int32_t first,
                         int32_t count, const PartTable& table, uint64_t seed,
                         std::vector<int32_t>& out) {
  if (count == 1 || g.n == 0) {
    for (int32_t v = 0; v < g.n; ++v) {
      out[ids[v]] = first;
    }
    return;
  }
  const int32_t left = count / 2;
  const int64_t total = total_vertex_weight(g);
  const int64_t target_left = sum(table.target, first, left);
  const int64_t target_both = sum(table.target, first, count);
  const int64_t goal_left = target_both > 0 ? scaled(total, target_left, target_both)
                                            : scaled(total, left, count);
  const int64_t goal[2] = {goal_left, total - goal_left};
  // Side capacities keep the tolerance ratio of their parts' capacities.
  int64_t capacity[2];
  for (int side = 0; side < 2; ++side) {
    const int32_t start = side == 0 ? first : first + left;
    const int32_t size = side == 0 ? left : count - left;
    const int64_t target = sum(table.target, start, size);
    const int64_t space = sum(table.capacity, start, size);
    capacity[side] = std::max(goal[side], target > 0 ? scaled(goal[side], space, target) : space);
  }
  const std::vector<int32_t> side = bisect(g, goal, capacity, table.require_nonempty,
                                           table.passes, mix(seed ^ static_cast<uint32_t>(first)));
  std::vector<int32_t> local(g.n, -1);
  std::vector<int32_t> members;
  for (int which = 0; which < 2; ++which) {
    const OwnedGraph sub = induced(g, side, which, local, members);
    std::vector<int32_t> sub_ids(members.size());
    for (std::size_t i = 0; i < members.size(); ++i) {
      sub_ids[i] = ids[members[i]];
    }
    recursive_bisection(sub.view(), sub_ids, which == 0 ? first : first + left,
                        which == 0 ? left : count - left, table, mix(seed + 1 + which), out);
  }
}

// Gives every empty part one vertex from the part holding the most vertices:
// the vertex that fits and loses the least internal edge weight, then the
// lightest, then the smallest index.
void fill_empty_parts(const GraphView& g, Partition& p, const Balance& b) {
  Connectivity conn(b.k);
  for (int32_t q = 0; q < b.k; ++q) {
    if (p.count[q] > 0) {
      continue;
    }
    int32_t pick = -1;
    bool pick_fits = false;
    int64_t pick_internal = 0;
    for (int32_t v = 0; v < g.n; ++v) {
      if (p.count[p.part[v]] <= 1) {
        continue;
      }
      conn.gather(g, p, v);
      const int64_t internal = conn.to_part[p.part[v]] - conn.to_part[q];
      conn.clear();
      const bool fit = g.vw[v] <= b.capacity[q];
      if (pick < 0 || (fit && !pick_fits) ||
          (fit == pick_fits &&
           (internal < pick_internal ||
            (internal == pick_internal && g.vw[v] < g.vw[pick])))) {
        pick = v;
        pick_fits = fit;
        pick_internal = internal;
      }
    }
    if (pick < 0) {
      throw std::logic_error("Required nonempty partition has no donor.");
    }
    p.move(g, pick, q);
  }
}

// A seeded recursive bisection, refined on the coarsest graph. Selection
// between starts belongs at the input graph: coarse cuts are not a fine-level
// quality oracle, and coarse overload may disappear during projection.
Partition initial_partition(const GraphView& g, const int64_t* targets, const Balance& balance,
                            int32_t passes, uint64_t seed) {
  const PartTable table{targets, balance.capacity, balance.require_nonempty, passes};
  std::vector<int32_t> ids(g.n);
  for (int32_t v = 0; v < g.n; ++v) {
    ids[v] = v;
  }
  Partition p;
  p.part.assign(g.n, 0);
  recursive_bisection(g, ids, 0, balance.k, table, seed, p.part);
  p.assign(g, balance.k);
  if (balance.require_nonempty) {
    fill_empty_parts(g, p, balance);
  }
  refine(g, p, balance, passes, seed);
  return p;
}

// Re-coarsens inside the current parts, so the coarsest graph carries `p`
// exactly, refines back to the input, and keeps the result when it is no
// worse by (overload, cut) under the exact capacities.
void vcycle(const GraphView& input, Partition& p, CapacityTable& caps, int64_t coarsen_to,
            int32_t passes, uint64_t seed) {
  std::vector<int32_t> coarse_part = p.part;
  const Hierarchy h = coarsen(input, coarsen_to, caps.max_weight, seed, &coarse_part);
  const GraphView coarsest = h.level(input, h.depth());
  Partition q;
  q.part = std::move(coarse_part);
  q.assign(coarsest, caps.k);
  refine(coarsest, q, caps.at(coarsest, h.depth() > 0), passes, seed);
  uncoarsen(input, h, q, caps, passes, seed);
  if (better(input, q, p, caps.at(input, false))) {
    p = std::move(q);
  }
}

struct PartitionRegion {
  OwnedGraph graph;
  std::vector<int32_t> members;
};

PartitionRegion pair_graph(const GraphView& g, const Partition& p, int32_t a, int32_t b,
                     std::vector<int32_t>& local) {
  PartitionRegion out;
  out.members.reserve(static_cast<std::size_t>(p.count[a] + p.count[b]));
  int64_t entries = 0;
  for (int32_t v = 0; v < g.n; ++v) {
    if (p.part[v] == a || p.part[v] == b) {
      local[v] = static_cast<int32_t>(out.members.size());
      out.members.push_back(v);
      entries += g.xadj[v + 1] - g.xadj[v];
    }
  }
  out.graph.xadj.reserve(out.members.size() + 1);
  out.graph.vw.reserve(out.members.size());
  out.graph.adj.reserve(static_cast<std::size_t>(entries));
  out.graph.ew.reserve(static_cast<std::size_t>(entries));
  out.graph.xadj.push_back(0);
  for (const int32_t v : out.members) {
    out.graph.vw.push_back(g.vw[v]);
    for (int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      const int32_t u = g.adj[e];
      if (p.part[u] == a || p.part[u] == b) {
        out.graph.adj.push_back(local[u]);
        out.graph.ew.push_back(g.ew[e]);
      }
    }
    out.graph.xadj.push_back(static_cast<int64_t>(out.graph.adj.size()));
  }
  return out;
}

// This is the exact global cut change: membership of the selected-part union
// is fixed, so every external edge stays crossing. A bijective relabeling of
// local parts also leaves the cut unchanged.
int64_t region_delta(const Partition& p, const PartitionRegion& pair,
                   const std::vector<int32_t>& side) {
  const GraphView sub = pair.graph.view();
  int64_t delta = 0;
  for (int32_t v = 0; v < sub.n; ++v) {
    for (int64_t e = sub.xadj[v]; e < sub.xadj[v + 1]; ++e) {
      const int32_t u = sub.adj[e];
      if (u < v) {
        continue;
      }
      const bool before = p.part[pair.members[v]] != p.part[pair.members[u]];
      delta += sub.ew[e] * (int64_t{side[v] != side[u]} - int64_t{before});
    }
  }
  return delta;
}

// Multi-vertex exchanges escape exact-capacity single/swap-move barriers.
// Three rounds, two seeded bisections per adjacent part pair; every proposal
// passes full-graph cut and exact-capacity/nonempty checks before publication.
void refine_pairs(const GraphView& g, Partition& p, const Balance& balance,
                  int32_t passes) {
  if (passes == 0 || balance.k < 2) {
    return;
  }
  std::vector<int32_t> local(g.n, -1);
  for (int32_t round = 0; round < std::min<int32_t>(passes, 3); ++round) {
    std::vector<std::pair<int32_t, int32_t>> pairs;
    for (int32_t v = 0; v < g.n; ++v) {
      for (int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
        const int32_t a = p.part[v], b = p.part[g.adj[e]];
        if (a < b) {
          pairs.emplace_back(a, b);
        }
      }
    }
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    bool changed = false;
    for (const auto& [a, b] : pairs) {
      const PartitionRegion pair = pair_graph(g, p, a, b, local);
      const int64_t goal[2] = {p.weight[a], p.weight[b]};
      const int64_t capacity[2] = {balance.capacity[a], balance.capacity[b]};
      for (int32_t trial = 0; trial < 2; ++trial) {
        const uint64_t seed = mix((static_cast<uint64_t>(a) << 32) ^
                                  static_cast<uint32_t>(b) ^
                                  mix(static_cast<uint64_t>(2 * round + trial)));
        const auto side = bisect(pair.graph.view(), goal, capacity,
                                 balance.require_nonempty, passes, seed);
        int64_t weight[2] = {0, 0}, count[2] = {0, 0};
        for (std::size_t v = 0; v < side.size(); ++v) {
          weight[side[v]] += pair.graph.vw[v];
          ++count[side[v]];
        }
        int32_t best_flip = -1;
        const int64_t delta = region_delta(p, pair, side);
        if (delta >= 0) {
          continue;
        }
        for (int32_t flip = 0; flip < 2; ++flip) {
          if (weight[flip] > capacity[0] || weight[1 ^ flip] > capacity[1] ||
              (balance.require_nonempty && (count[0] == 0 || count[1] == 0))) {
            continue;
          }
          best_flip = flip;
          break;
        }
        if (best_flip < 0) {
          continue;
        }
        p.weight[a] = weight[best_flip];
        p.weight[b] = weight[1 ^ best_flip];
        p.count[a] = count[best_flip];
        p.count[b] = count[1 ^ best_flip];
        for (std::size_t v = 0; v < side.size(); ++v) {
          const int32_t next = (side[v] ^ best_flip) == 0 ? a : b;
          p.part[pair.members[v]] = next;
        }
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
  }
}

void partition_multilevel(const GraphView& input, int32_t k, const int64_t* targets,
                          const int64_t* capacities, bool nonempty, int32_t passes,
                          int32_t starts, int32_t cycles, uint64_t seed_offset,
                          Partition& p) {
  const int64_t total = total_vertex_weight(input);
  int64_t log_parts = 1;
  while ((int64_t{1} << log_parts) < k) {
    ++log_parts;
  }
  const int64_t coarsen_to = std::max<int64_t>(int64_t{30} * k, input.n / (40 * log_parts));
  // Matching never merges beyond 1.5 times the mean coarsest vertex weight.
  const int64_t max_weight = std::max<int64_t>(1, (3 * total) / (2 * coarsen_to));
  CapacityTable caps{k, capacities, nonempty, max_weight, {}};
  for (int32_t trial = 0; trial < starts; ++trial) {
    const uint64_t seed = mix(seed_offset + 0x1000 + static_cast<uint64_t>(trial));
    const Hierarchy h = coarsen(input, coarsen_to, max_weight, seed, nullptr);
    const GraphView coarsest = h.level(input, h.depth());
    Partition q = initial_partition(coarsest, targets, caps.at(coarsest, h.depth() > 0),
                                    passes, seed);
    uncoarsen(input, h, q, caps, passes, seed);
    if (p.part.empty() || better(input, q, p, caps.at(input, false))) {
      p = std::move(q);
    }
  }
  for (int32_t cycle = 0; cycle < cycles; ++cycle) {
    vcycle(input, p, caps, coarsen_to, passes,
           mix(seed_offset + 0x2000 + static_cast<uint64_t>(cycle)));
  }
}

struct PartBoundary {
  std::vector<int64_t> offsets;
  std::vector<int32_t> neighbors;
  std::vector<int64_t> weights;
};

PartBoundary part_boundary(const GraphView& g, const Partition& p, int32_t k) {
  using Edge = std::pair<std::pair<int32_t, int32_t>, int64_t>;
  std::vector<Edge> edges;
  edges.reserve(static_cast<std::size_t>(g.xadj[g.n]));
  for (int32_t v = 0; v < g.n; ++v) {
    for (int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      const int32_t a = p.part[v], b = p.part[g.adj[e]];
      if (a != b) {
        edges.push_back({{a, b}, g.ew[e]});
      }
    }
  }
  std::sort(edges.begin(), edges.end());
  PartBoundary out;
  out.offsets.assign(static_cast<std::size_t>(k) + 1, 0);
  out.neighbors.reserve(edges.size());
  out.weights.reserve(edges.size());
  for (std::size_t first = 0; first < edges.size();) {
    std::size_t end = first;
    int64_t weight = 0;
    while (end < edges.size() && edges[end].first == edges[first].first) {
      weight += edges[end++].second;
    }
    ++out.offsets[edges[first].first.first + 1];
    out.neighbors.push_back(edges[first].first.second);
    out.weights.push_back(weight);
    first = end;
  }
  for (int32_t q = 0; q < k; ++q) {
    out.offsets[q + 1] += out.offsets[q];
  }
  return out;
}

std::vector<int32_t> grow_region(const PartBoundary& boundary, int32_t k, int32_t seed,
                                 uint64_t rank_seed) {
  std::vector<int64_t> connection(k, 0);
  std::vector<char> selected(k, 0);
  const auto rank = ranks(k, rank_seed);
  std::vector<int32_t> group;
  group.reserve(8);
  int32_t next = seed;
  while (next >= 0 && group.size() < 8) {
    group.push_back(next);
    selected[next] = 1;
    for (int64_t e = boundary.offsets[next]; e < boundary.offsets[next + 1]; ++e) {
      const int32_t q = boundary.neighbors[e];
      if (!selected[q]) {
        connection[q] += boundary.weights[e];
      }
    }
    next = -1;
    for (int32_t q = 0; q < k; ++q) {
      if (!selected[q] && connection[q] > 0 &&
          (next < 0 || connection[q] > connection[next] ||
           (connection[q] == connection[next] &&
            (rank[q] < rank[next] || (rank[q] == rank[next] && q < next))))) {
        next = q;
      }
    }
  }
  std::sort(group.begin(), group.end());
  return group;
}

// Two bounded multiway rounds, at most k unique regions of eight adjacent
// parts per round. Repartitioning only the fixed union keeps external cut
// edges invariant; an exact cut decrease and all fine capacities are required.
void refine_regions(const GraphView& g, Partition& p, const Balance& balance,
                    int32_t passes) {
  const int32_t k = balance.k;
  if (passes == 0 || k < 3) {
    return;
  }
  std::vector<int32_t> included(g.n), local(g.n, -1);
  std::vector<char> selected(k);
  for (int32_t round = 0; round < std::min<int32_t>(passes, 2); ++round) {
    const PartBoundary boundary = part_boundary(g, p, k);
    std::vector<std::vector<int32_t>> groups;
    groups.reserve(k);
    for (int32_t seed = 0; seed < k; ++seed) {
      if (p.count[seed] > 0) {
        auto group = grow_region(boundary, k, seed,
                                 mix(0x3000 + (static_cast<uint64_t>(round) << 32) + seed));
        if (group.size() > 2) {
          groups.push_back(std::move(group));
        }
      }
    }
    std::sort(groups.begin(), groups.end());
    groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
    bool changed = false;
    for (const auto& group : groups) {
      std::fill(selected.begin(), selected.end(), 0);
      std::vector<int64_t> target, capacity;
      target.reserve(group.size());
      capacity.reserve(group.size());
      uint64_t seed = mix(0x4000 + static_cast<uint64_t>(round));
      for (const int32_t q : group) {
        selected[q] = 1;
        target.push_back(p.weight[q]);
        capacity.push_back(balance.capacity[q]);
        seed = mix(seed ^ static_cast<uint32_t>(q));
      }
      for (int32_t v = 0; v < g.n; ++v) {
        included[v] = selected[p.part[v]] ? 0 : 1;
      }
      PartitionRegion region;
      region.graph = induced(g, included, 0, local, region.members);
      const int32_t count = static_cast<int32_t>(group.size());
      const Balance sub_balance{count, capacity.data(), balance.require_nonempty};
      Partition candidate;
      partition_multilevel(region.graph.view(), count, target.data(), capacity.data(),
                           balance.require_nonempty, passes, 2, 1, seed, candidate);
      refine_pairs(region.graph.view(), candidate, sub_balance, passes);
      if (overload(candidate, sub_balance) != 0 ||
          (balance.require_nonempty &&
           std::find(candidate.count.begin(), candidate.count.end(), 0) != candidate.count.end()) ||
          region_delta(p, region, candidate.part) >= 0) {
        continue;
      }
      for (std::size_t q = 0; q < group.size(); ++q) {
        p.weight[group[q]] = candidate.weight[q];
        p.count[group[q]] = candidate.count[q];
      }
      for (std::size_t v = 0; v < region.members.size(); ++v) {
        const int32_t next = group[candidate.part[v]];
        p.part[region.members[v]] = next;
      }
      changed = true;
    }
    if (!changed) {
      break;
    }
  }
}

void partition_connected(const GraphView& input, int32_t k, const int64_t* targets,
                         const int64_t* capacities, bool nonempty, int32_t passes, int32_t* out) {
  Partition p;
  partition_multilevel(input, k, targets, capacities, nonempty, passes,
                       kInitialPartitions, kVCycles, 0, p);
  refine_pairs(input, p, {k, capacities, nonempty}, passes);
  refine_regions(input, p, {k, capacities, nonempty}, passes);
  std::copy(p.part.begin(), p.part.end(), out);
}

struct Components {
  std::vector<int32_t> label;
  std::vector<int32_t> size;
  std::vector<int64_t> weight;
};

Components components(const GraphView& g) {
  Components out;
  out.label.assign(g.n, -1);
  std::vector<int32_t> queue;
  queue.reserve(g.n);
  for (int32_t seed = 0; seed < g.n; ++seed) {
    if (out.label[seed] >= 0) {
      continue;
    }
    const int32_t id = static_cast<int32_t>(out.size.size());
    queue.clear();
    queue.push_back(seed);
    out.label[seed] = id;
    int64_t weight = 0;
    for (std::size_t head = 0; head < queue.size(); ++head) {
      const int32_t v = queue[head];
      weight += g.vw[v];
      for (int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
        const int32_t u = g.adj[e];
        if (out.label[u] < 0) {
          out.label[u] = id;
          queue.push_back(u);
        }
      }
    }
    out.size.push_back(static_cast<int32_t>(queue.size()));
    out.weight.push_back(weight);
  }
  return out;
}

// Stable contiguous part allocation minimizing total target mismatch, subject
// to each component's exact aggregate capacity and nonempty vertex count.
std::vector<int32_t> component_parts(const Components& c, int32_t k,
                                      const int64_t* targets, const int64_t* capacities,
                                      bool nonempty) {
  const int32_t count = static_cast<int32_t>(c.size.size());
  if (count <= 1 || count >= k) {
    return {};
  }
  const std::size_t stride = static_cast<std::size_t>(k) + 1;
  std::vector<int32_t> predecessor(static_cast<std::size_t>(count) * stride, -1);
  const int64_t infinity = std::numeric_limits<int64_t>::max();
  std::vector<int64_t> previous(stride, infinity), next(stride, infinity), prefix(stride, 0);
  for (int32_t q = 0; q < k; ++q) {
    prefix[q + 1] = prefix[q] + targets[q];
  }
  previous[0] = 0;
  for (int32_t component = 0; component < count; ++component) {
    std::fill(next.begin(), next.end(), infinity);
    for (int32_t first = component; first < k; ++first) {
      if (previous[first] == infinity) {
        continue;
      }
      int64_t capacity = 0;
      const int32_t last = std::min(k - (count - component - 1),
                                   nonempty ? first + std::min(k - first, c.size[component]) : k);
      for (int32_t end = first + 1; end <= last; ++end) {
        // Only the threshold matters; saturation avoids capacity-sum overflow.
        capacity += std::min(capacities[end - 1], c.weight[component] - capacity);
        if (capacity < c.weight[component]) {
          continue;
        }
        const int64_t difference = prefix[end] - prefix[first] - c.weight[component];
        const int64_t cost = previous[first] + (difference < 0 ? -difference : difference);
        if (cost < next[end]) {
          next[end] = cost;
          predecessor[static_cast<std::size_t>(component) * stride + end] = first;
        }
      }
    }
    previous.swap(next);
  }
  if (previous[k] == infinity) {
    return {};
  }
  std::vector<int32_t> boundaries(static_cast<std::size_t>(count) + 1);
  boundaries[count] = k;
  for (int32_t component = count - 1; component >= 0; --component) {
    boundaries[component] =
        predecessor[static_cast<std::size_t>(component) * stride + boundaries[component + 1]];
  }
  return boundaries;
}

void partition_kway(const GraphView& input, int32_t k, const int64_t* targets,
                    const int64_t* capacities, bool nonempty, int32_t passes, int32_t* out) {
  const Components c = components(input);
  // A zero-cut one-part-per-component solution exists iff sorted weights
  // fit sorted capacities; this avoids quadratic allocation state.
  if (c.size.size() == static_cast<std::size_t>(k)) {
    std::vector<int32_t> component_order(k), part_order(k);
    for (int32_t q = 0; q < k; ++q) {
      component_order[q] = part_order[q] = q;
    }
    std::sort(component_order.begin(), component_order.end(), [&](int32_t a, int32_t b) {
      return c.weight[a] != c.weight[b] ? c.weight[a] < c.weight[b] : a < b;
    });
    std::sort(part_order.begin(), part_order.end(), [&](int32_t a, int32_t b) {
      return capacities[a] != capacities[b] ? capacities[a] < capacities[b] : a < b;
    });
    bool feasible = true;
    std::vector<int32_t> owner(k);
    for (int32_t q = 0; q < k; ++q) {
      feasible = feasible && c.weight[component_order[q]] <= capacities[part_order[q]];
      owner[component_order[q]] = part_order[q];
    }
    if (feasible) {
      for (int32_t v = 0; v < input.n; ++v) {
        out[v] = owner[c.label[v]];
      }
      return;
    }
  }
  const auto allocation = component_parts(c, k, targets, capacities, nonempty);
  Partition separated;
  if (!allocation.empty()) {
    separated.part.resize(input.n);
    std::vector<int32_t> local(input.n, -1), members;
    for (std::size_t component = 0; component < c.size.size(); ++component) {
      const int32_t first = allocation[component];
      const int32_t count = allocation[component + 1] - first;
      const OwnedGraph sub = induced(input, c.label, static_cast<int32_t>(component),
                                     local, members);
      std::vector<int32_t> owners(members.size());
      if (count == 1) {
        std::fill(owners.begin(), owners.end(), 0);
      } else {
        partition_connected(sub.view(), count, targets + first, capacities + first,
                            nonempty, passes, owners.data());
      }
      for (std::size_t v = 0; v < members.size(); ++v) {
        separated.part[members[v]] = first + owners[v];
      }
    }
    separated.assign(input, k);
    const Balance exact{k, capacities, nonempty};
    if (overload(separated, exact) == 0) {
      std::copy(separated.part.begin(), separated.part.end(), out);
      return;
    }
  }
  if (separated.part.empty()) {
    partition_connected(input, k, targets, capacities, nonempty, passes, out);
    return;
  }
  Partition whole;
  whole.part.resize(input.n);
  partition_connected(input, k, targets, capacities, nonempty, passes, whole.part.data());
  whole.assign(input, k);
  const auto& winner = better(input, separated, whole, {k, capacities, nonempty})
                           ? separated.part
                           : whole.part;
  std::copy(winner.begin(), winner.end(), out);
}

// ----------------------------------------------------------------- validation

int32_t validate_graph(int64_t n, int64_t edge_count, const int64_t* xadj, const int32_t* adj,
                       const int64_t* ew, const int64_t* vw) {
  if (!phx::mc::addressable(n + 1, 1, sizeof(int64_t)) ||
      !phx::mc::addressable(edge_count, 1, sizeof(int32_t)) ||
      !phx::mc::addressable(edge_count, 1, sizeof(int64_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (xadj == nullptr || vw == nullptr) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (xadj[0] != 0) {
    return PHX_MC_INVALID_INPUT;
  }
  for (int64_t v = 0; v < n; ++v) {
    if (xadj[v] < 0 || xadj[v + 1] < xadj[v] || xadj[v + 1] > edge_count) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  if (xadj[n] != edge_count) {
    return PHX_MC_INVALID_INPUT;
  }
  if (edge_count > 0 && (adj == nullptr || ew == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  int64_t vertex_total = 0;
  for (int64_t v = 0; v < n; ++v) {
    if (vw[v] < 0 || vw[v] > kWeightLimit - vertex_total) {
      return PHX_MC_INVALID_INPUT;
    }
    vertex_total += vw[v];
  }
  if (vertex_total == 0) {
    return PHX_MC_INVALID_INPUT;
  }
  int64_t edge_total = 0;
  for (int64_t v = 0; v < n; ++v) {
    for (int64_t e = xadj[v]; e < xadj[v + 1]; ++e) {
      const int32_t u = adj[e];
      // Rows are strictly increasing: no self loop, duplicate or out-of-range entry.
      if (u < 0 || u >= n || u == v || (e > xadj[v] && adj[e - 1] >= u) || ew[e] < 0 ||
          ew[e] > kWeightLimit - edge_total) {
        return PHX_MC_INVALID_INPUT;
      }
      edge_total += ew[e];
      const int32_t* row = adj + xadj[u];
      const int32_t* end = adj + xadj[u + 1];
      const int32_t* mirror = std::lower_bound(row, end, static_cast<int32_t>(v));
      if (mirror == end || *mirror != v || ew[xadj[u] + (mirror - row)] != ew[e]) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  return PHX_MC_OK;
}

int32_t validate_parts(int64_t n, int32_t k, const int64_t* target, const int64_t* capacity,
                       const int64_t* vw, int32_t require_nonempty) {
  if (require_nonempty != 0 && k > n) {
    return PHX_MC_INVALID_INPUT;
  }
  int64_t vertex_total = 0;
  for (int64_t v = 0; v < n; ++v) {
    vertex_total += vw[v];
  }
  int64_t target_total = 0;
  for (int32_t q = 0; q < k; ++q) {
    if (target[q] < 0 || capacity[q] < target[q] || capacity[q] > kWeightLimit ||
        target[q] > kWeightLimit - target_total) {
      return PHX_MC_INVALID_INPUT;
    }
    target_total += target[q];
  }
  return target_total == vertex_total ? PHX_MC_OK : PHX_MC_INVALID_INPUT;
}

}  // namespace

extern "C" {

int32_t phx_mc_graph_partition(int64_t vertex_count, int64_t edge_count, const int64_t* offsets,
                               const int32_t* neighbors, const int64_t* edge_weights,
                               const int64_t* vertex_weights, int32_t part_count,
                               const int64_t* part_targets, const int64_t* part_capacities,
                               int32_t require_nonempty, int32_t refinement_passes,
                               int32_t* parts) {
  return phx::mc::guarded([&]() -> int32_t {
    if (vertex_count >= std::numeric_limits<int32_t>::max())
      return PHX_MC_INDEX_RANGE_ERROR;
    if (vertex_count <= 0 || edge_count < 0 || part_count <= 0 || refinement_passes < 0 ||
        offsets == nullptr || vertex_weights == nullptr || part_targets == nullptr ||
        part_capacities == nullptr || parts == nullptr ||
        !phx::mc::addressable(vertex_count + 1, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(vertex_count, 1, sizeof(int64_t)) ||
        !phx::mc::addressable(vertex_count, 1, sizeof(int32_t)) ||
        !phx::mc::addressable(part_count, 1, sizeof(int64_t))) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    int32_t status =
        validate_graph(vertex_count, edge_count, offsets, neighbors, edge_weights, vertex_weights);
    if (status == PHX_MC_OK) {
      status = validate_parts(vertex_count, part_count, part_targets, part_capacities,
                              vertex_weights, require_nonempty);
    }
    if (status != PHX_MC_OK) {
      return status;
    }
    const GraphView input{static_cast<int32_t>(vertex_count), offsets, neighbors, edge_weights,
                          vertex_weights};
    partition_kway(input, part_count, part_targets, part_capacities, require_nonempty != 0,
                   refinement_passes, parts);
    return PHX_MC_OK;
  });
}

}  // extern "C"
