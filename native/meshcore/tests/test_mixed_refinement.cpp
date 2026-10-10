//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {
double determinant(const double* points, const std::array<int,3>& axes) {
  double a[3], b[3], c[3];
  for (int axis=0; axis<3; ++axis) {
    a[axis]=points[3*axes[0]+axis]-points[axis];
    b[axis]=points[3*axes[1]+axis]-points[axis];
    c[axis]=points[3*axes[2]+axis]-points[axis];
  }
  return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]);
}
void template_partition(int32_t kind, int32_t axial) {
  int32_t counts[3];
  PHX_CHECK(phx_mc_mixed_template_counts(kind, axial, 0, counts)==PHX_MC_OK);
  const int templates=counts[0], corners=counts[2];
  std::vector<double> parent(3*corners);
  PHX_CHECK(phx_mc_mixed_template(kind, axial, 0, parent.size(), parent.data())==PHX_MC_OK);
  const std::array<int,3> axes=kind==0 || kind==2 ? std::array<int,3>{1,2,3} : std::array<int,3>{1,3,4};
  const double reference=determinant(parent.data(), axes);
  for (int index=1; index<templates; ++index) {
    PHX_CHECK(phx_mc_mixed_template_counts(kind, axial, index, counts)==PHX_MC_OK);
    std::vector<double> children(3*corners*counts[1]);
    PHX_CHECK(phx_mc_mixed_template(kind, axial, index, children.size(), children.data())==PHX_MC_OK);
    double volume=0;
    for (int child=0; child<counts[1]; ++child) {
      const double* points=children.data()+3*corners*child;
      const double ratio=determinant(points, axes)/reference;
      PHX_CHECK(ratio>0);
      volume+=ratio;
      for (int vertex=0; vertex<corners; ++vertex) {
        const double x=points[3*vertex], y=points[3*vertex+1], z=points[3*vertex+2];
        PHX_CHECK(x>=0 && y>=0 && z>=0 && x<=1 && y<=1 && z<=1);
        if (kind==0) PHX_CHECK(x+y+z<=1);
        if (kind==2) PHX_CHECK(x+y<=1);
        if (kind==3) PHX_CHECK(x>=z/2 && y>=z/2 && x<=1-z/2 && y<=1-z/2);
      }
    }
    PHX_CHECK(volume==1.0);
  }
}
void capacity_rollback() {
  std::array<double,12> storage;
  storage.fill(42);
  PHX_CHECK(phx_mc_mixed_template(0,0,1,12,storage.data())==PHX_MC_CAPACITY_EXCEEDED);
  for (double value:storage) PHX_CHECK(value==42);
  int32_t counts[3];
  PHX_CHECK(phx_mc_mixed_template_counts(5,0,0,counts)==PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_mixed_template_counts(1,2,0,counts)==PHX_MC_INVALID_ARGUMENT);
}
}
void quad_partitions() {
  int32_t counts[3];
  PHX_CHECK(phx_mc_mixed_template_counts(4,0,0,counts)==PHX_MC_OK);
  PHX_CHECK(counts[0]==4 && counts[2]==4);
  for (int index=1; index<4; ++index) {
    PHX_CHECK(phx_mc_mixed_template_counts(4,0,index,counts)==PHX_MC_OK);
    std::vector<double> points(counts[1]*8);
    PHX_CHECK(phx_mc_mixed_template(4,0,index,points.size(),points.data())==PHX_MC_OK);
    double area=0;
    for (int child=0; child<counts[1]; ++child) {
      const double* p=points.data()+8*child;
      double determinant=(p[2]-p[0])*(p[7]-p[1])-(p[3]-p[1])*(p[6]-p[0]);
      PHX_CHECK(determinant>0);
      area+=determinant;
    }
    PHX_CHECK(area==1);
  }
}

void tetra_face_preserving_partition() {
  int32_t counts[3];
  PHX_CHECK(phx_mc_mixed_template_counts(0,0,7,counts)==PHX_MC_OK);
  PHX_CHECK(counts[1]==4 && counts[2]==4);
  std::array<double,48> children;
  PHX_CHECK(phx_mc_mixed_template(0,0,7,children.size(),children.data())==PHX_MC_OK);
  constexpr double parent[12]={0,0,0,1,0,0,0,1,0,0,0,1};
  double volume=0;
  for (int child=0; child<4; ++child) {
    const double* points=children.data()+12*child;
    const double ratio=determinant(points,{1,2,3});
    PHX_CHECK(ratio==0.25);
    volume+=ratio;
    for (int vertex=0; vertex<4; ++vertex)
      for (int axis=0; axis<3; ++axis)
        PHX_CHECK(points[3*vertex+axis]==(vertex==child ? 0.25 : parent[3*vertex+axis]));
  }
  PHX_CHECK(volume==1);
}

void tetra_red_partition() {
  using Point=std::array<double,3>;
  using Triangle=std::array<Point,3>;
  constexpr Point points[4]={Point{0,0,0},Point{1,0,0},Point{0,1,0},Point{0,0,1}};
  constexpr int faces[4][3]={{0,2,1},{0,1,3},{0,3,2},{1,2,3}};
  auto canonical=[](Triangle triangle) {
    std::sort(triangle.begin(),triangle.end());
    return triangle;
  };
  auto midpoint=[](const Point& first,const Point& second) {
    return Point{(first[0]+second[0])/2,(first[1]+second[1])/2,(first[2]+second[2])/2};
  };
  auto on_face=[](const Point& point,int face) {
    return face==0 ? point[2]==0 : face==1 ? point[1]==0 :
      face==2 ? point[0]==0 : point[0]+point[1]+point[2]==1;
  };
  for (int index=8; index<11; ++index) {
    int32_t counts[3];
    PHX_CHECK(phx_mc_mixed_template_counts(0,0,index,counts)==PHX_MC_OK);
    PHX_CHECK(counts[0]==11 && counts[1]==8 && counts[2]==4);
    std::array<double,96> children;
    children.fill(42);
    PHX_CHECK(phx_mc_mixed_template(0,0,index,children.size()-1,children.data())==PHX_MC_CAPACITY_EXCEEDED);
    for (double value:children) PHX_CHECK(value==42);
    PHX_CHECK(phx_mc_mixed_template(0,0,index,children.size(),children.data())==PHX_MC_OK);
    for (int child=0; child<8; ++child)
      PHX_CHECK(determinant(children.data()+12*child,{1,2,3})==0.125);
    for (int face=0; face<4; ++face) {
      const Point a=points[faces[face][0]],b=points[faces[face][1]],c=points[faces[face][2]];
      const Point ab=midpoint(a,b),bc=midpoint(b,c),ca=midpoint(c,a);
      std::array<Triangle,4> expected={
        canonical(Triangle{a,ab,ca}),canonical(Triangle{ab,b,bc}),
        canonical(Triangle{ca,bc,c}),canonical(Triangle{ab,bc,ca})};
      std::array<Triangle,4> actual;
      int found=0;
      for (int child=0; child<8; ++child)
        for (const auto& local:faces) {
          Triangle triangle;
          for (int vertex=0; vertex<3; ++vertex)
            for (int axis=0; axis<3; ++axis)
              triangle[vertex][axis]=children[12*child+3*local[vertex]+axis];
          if (on_face(triangle[0],face) && on_face(triangle[1],face) && on_face(triangle[2],face)) {
            PHX_CHECK(found<4);
            actual[found++]=canonical(triangle);
          }
        }
      PHX_CHECK(found==4);
      std::sort(expected.begin(),expected.end());
      std::sort(actual.begin(),actual.end());
      PHX_CHECK(actual==expected);
    }
  }
}

int main() {
  template_partition(0,0);
  template_partition(1,0);
  template_partition(1,1);
  template_partition(2,0);
  template_partition(2,1);
  template_partition(3,0);
  quad_partitions();
  tetra_face_preserving_partition();
  tetra_red_partition();
  capacity_rollback();
  std::puts("Mixed same-family template coverage and bounded rollback checks passed.");
  return 0;
}
