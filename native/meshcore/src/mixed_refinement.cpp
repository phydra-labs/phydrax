//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Dyadic, orientation-preserving same-family templates. No allocation is needed:
// callers query the bounded counts, then supply storage. Every child is an affine
// image in the parent reference, including apex-preserving pyramid restrictions.
#include <algorithm>
#include <cstdint>
#include <cstring>

#include "capi_guard.hpp"
namespace phx::mc {
int32_t quad_template_counts(int32_t axial, int32_t index, int32_t* counts);
int32_t quad_template(int32_t axial, int32_t index, int64_t capacity, double* references);
}
namespace {
constexpr double tet[4][3] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
constexpr double hex[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
constexpr double prism[6][3] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,0,1},{0,1,1}};
constexpr double pyramid[5][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0.5,0.5,1}};
const double* vertices(int32_t kind) {
  switch (kind) {
    case 0: return &tet[0][0];
    case 1: return &hex[0][0];
    case 2: return &prism[0][0];
    case 3: return &pyramid[0][0];
    default: return nullptr;
  }
}
int32_t arity(int32_t kind) {
  switch (kind) { case 0: return 4; case 1: return 8; case 2: return 6; case 3: return 5; default: return 0; }
}
int32_t template_count(int32_t kind, int32_t axial) {
  switch (kind) { case 0: return 11; case 1: return axial ? 8 : 4; case 2: return axial ? 10 : 5; case 3: return 4; default: return 0; }
}
int32_t child_count(int32_t kind, int32_t axial, int32_t index) {
  if (index == 0) return 1;
  if (kind == 0) return index >= 8 ? 8 : index == 7 ? 4 : 2;
  if (kind == 1) { constexpr int masks[] = {3,1,2,7,4,5,6}; int mask = masks[index-1]; return 1 << ((mask&1) + ((mask>>1)&1) + ((mask>>2)&1)); }
  if (kind == 3) return index == 1 ? 4 : 2;
  if (kind == 2) {
    if (axial && index == 9) return 2;
    int base = axial ? (index-1)/2 : index-1;
    int intervals = axial && index%2 == 0 ? 2 : 1;
    return (base == 0 ? 4 : 2) * intervals;
  }
  return 0;
}
}

extern "C" int32_t phx_mc_mixed_template_counts(int32_t kind, int32_t axial, int32_t index, int32_t* counts) {
  return phx::mc::guarded([&]() -> int32_t {
  if (kind == 4) return phx::mc::quad_template_counts(axial, index, counts);
  if (!counts || axial < 0 || axial > 1 || template_count(kind, axial) == 0 || index < 0 || index >= template_count(kind, axial)) return PHX_MC_INVALID_ARGUMENT;
  counts[0] = template_count(kind, axial);
  counts[1] = child_count(kind, axial, index);
  counts[2] = arity(kind);
  return PHX_MC_OK;
  });
}

extern "C" int32_t phx_mc_mixed_template(int32_t kind, int32_t axial, int32_t index, int64_t capacity, double* references) {
  return phx::mc::guarded([&]() -> int32_t {
  if (kind == 4) return phx::mc::quad_template(axial, index, capacity, references);
  int32_t counts[3];
  if (phx_mc_mixed_template_counts(kind, axial, index, counts) != PHX_MC_OK || !references || capacity < 0) return PHX_MC_INVALID_ARGUMENT;
  const int64_t entries = int64_t{counts[1]} * counts[2] * 3;
  if (capacity < entries) return PHX_MC_CAPACITY_EXCEEDED;
  const double* source = vertices(kind);
  const int width = counts[2] * 3;
  if (index == 0) { std::memcpy(references, source, width * sizeof(double)); return PHX_MC_OK; }
  if (kind == 0) {
    if (index >= 8) {
      // Four corner tetrahedra and a four-tetrahedron partition of the central
      // octahedron. All three diagonals have identical red-refined face traces;
      // the owning topology selector chooses their scientific midpoint identity.
      constexpr double points[10][3] = {
        {0,0,0},{1,0,0},{0,1,0},{0,0,1},
        {0.5,0,0},{0,0.5,0},{0,0,0.5},
        {0.5,0.5,0},{0.5,0,0.5},{0,0.5,0.5}
      };
      constexpr int corner_children[4][4] = {
        {0,4,5,6},{4,1,7,8},{5,7,2,9},{6,8,9,3}
      };
      // Diagonal endpoints followed by the positively oriented equatorial ring.
      constexpr int octahedra[3][6] = {
        {4,9,5,6,8,7},{5,8,4,7,9,6},{6,7,4,5,9,8}
      };
      const auto& central = octahedra[index-8];
      for (int child=0; child<8; ++child)
        for (int vertex=0; vertex<4; ++vertex) {
          const int point = child < 4 ? corner_children[child][vertex] :
            vertex < 2 ? central[vertex] : central[2+(child-4+vertex-2)%4];
          std::copy(points[point], points[point]+3, references+child*width+3*vertex);
        }
      return PHX_MC_OK;
    }
    if (index == 7) {
      for (int child=0; child<4; ++child) {
        std::memcpy(references+child*width, source, width*sizeof(double));
        for (int axis=0; axis<3; ++axis) references[child*width+3*child+axis]=0.25;
      }
      return PHX_MC_OK;
    }
    constexpr int edges[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    const int a = edges[index-1][0], b = edges[index-1][1];
    std::memcpy(references, source, width * sizeof(double));
    std::memcpy(references + width, source, width * sizeof(double));
    for (int axis=0; axis<3; ++axis) {
      references[3*b+axis] = (source[3*a+axis] + source[3*b+axis])/2;
      references[width+3*a+axis] = references[3*b+axis];
    }
    return PHX_MC_OK;
  }
  if (kind == 1 || kind == 3) {
    constexpr int masks[] = {3,1,2,7,4,5,6};
    const int mask = masks[index-1];
    int child=0;
    for (int x=0; x<(mask&1 ? 2 : 1); ++x)
      for (int y=0; y<(mask&2 ? 2 : 1); ++y)
        for (int z=0; z<(mask&4 ? 2 : 1); ++z) {
          const int start[3] = {x,y,z};
          for (int vertex=0; vertex<counts[2]; ++vertex)
            for (int axis=0; axis<3; ++axis) {
              const double scale = mask&(1<<axis) ? 0.5 : 1;
              references[child*width+3*vertex+axis] = source[3*vertex+axis]*scale + start[axis]*scale;
            }
          if (kind == 3) std::copy(source+12, source+15, references+child*width+12);
          ++child;
        }
    return PHX_MC_OK;
  }
  if (axial && index == 9) {
    for (int child=0; child<2; ++child) for (int vertex=0; vertex<6; ++vertex) for (int axis=0; axis<3; ++axis)
      references[child*width+3*vertex+axis] = axis == 2 ? source[3*vertex+axis]/2 + child*0.5 : source[3*vertex+axis];
    return PHX_MC_OK;
  }
  const int base = axial ? (index-1)/2 : index-1;
  const int intervals = axial && index%2 == 0 ? 2 : 1;
  const double red[4][3][2] = {{{0,0},{0.5,0},{0,0.5}},{{0.5,0},{1,0},{0.5,0.5}},{{0,0.5},{0.5,0.5},{0,1}},{{0.5,0},{0.5,0.5},{0,0.5}}};
  double triangles[4][3][2];
  int n=4;
  if (base == 0) std::memcpy(triangles, red, sizeof(red));
  else {
    constexpr int edges[3][2] = {{0,1},{1,2},{0,2}};
    const int a=edges[base-1][0], b=edges[base-1][1]; n=2;
    for (int child=0; child<2; ++child) for (int vertex=0; vertex<3; ++vertex) for (int axis=0; axis<2; ++axis)
      triangles[child][vertex][axis] = (child == 0 && vertex == b) || (child == 1 && vertex == a) ? (prism[a][axis]+prism[b][axis])/2 : prism[vertex][axis];
  }
  int child=0;
  for (int interval=0; interval<intervals; ++interval) for (int triangle=0; triangle<n; ++triangle) {
    for (int vertex=0; vertex<6; ++vertex) {
      references[child*width+3*vertex] = triangles[triangle][vertex%3][0];
      references[child*width+3*vertex+1] = triangles[triangle][vertex%3][1];
      references[child*width+3*vertex+2] = (interval + (vertex/3))/static_cast<double>(intervals);
    }
    ++child;
  }
  return PHX_MC_OK;
  });
}
