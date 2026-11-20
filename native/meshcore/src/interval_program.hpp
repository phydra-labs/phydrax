// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <cstdint>
#include <vector>

namespace phx::mc {

// Owns validated bytecode and constant banks; no mutable views escape. Workspace
// contents are supplied anew for each execution and retain public semantics.
class PreparedIntervalProgram {
 public:
  PreparedIntervalProgram() = default;
  PreparedIntervalProgram(const PreparedIntervalProgram&) = delete;
  PreparedIntervalProgram& operator=(const PreparedIntervalProgram&) = delete;
  PreparedIntervalProgram(PreparedIntervalProgram&&) = delete;
  PreparedIntervalProgram& operator=(PreparedIntervalProgram&&) = delete;
  int32_t bind(const int64_t* instructions, int64_t instruction_count,
               const int64_t* indices, int64_t index_count,
               const double* parameters, int64_t parameter_count, int64_t workspace_size);
  int32_t execute(double* lower, double* upper, int64_t workspace_size) const;
  const std::vector<int64_t>& instructions() const noexcept { return instructions_; }
  const std::vector<int64_t>& indices() const noexcept { return indices_; }
  const std::vector<double>& parameters() const noexcept { return parameters_; }
  void reset() noexcept { workspace_size_ = -1; }

 private:
  std::vector<int64_t> instructions_, indices_;
  std::vector<double> parameters_;
  int64_t workspace_size_ = -1;
};

// Whether phx_mc_interval_program_batch executes every opcode of a program
// (valid bytecode assumed): the elementwise subset, without the reciprocal.
bool batch_admissible(const int64_t* instructions, int64_t instruction_count) noexcept;

// Straight-line program executed at points in binary64 (no rounding control):
// floating-point proposals that bound nothing. `seeds` is the complete
// workspace template (seed values at their entries); the first
// `input_dimension` entries are overwritten by each point's coordinates.
struct ProgramView {
  const int64_t* instructions;
  int64_t instruction_count;
  const int64_t* indices;
  const double* seeds;
  int64_t workspace_size;
  int64_t input_dimension;
  const int64_t* outputs;
  int64_t output_count;
};

// Caller-owned execution memory of the point executors. It grows to the
// largest block a caller has executed and is reused without allocation.
struct PointWorkspace {
  std::vector<double> values;
  std::vector<double> scratch;
};

// Validates a point program once: the bytecode contract of
// phx_mc_program_evaluate. Returns a PHX_MC status.
int32_t validate_point_program(const int64_t* instructions, int64_t instruction_count,
                               const int64_t* indices, int64_t index_count,
                               const double* parameters, int64_t parameter_count,
                               int64_t workspace_size, int64_t input_dimension,
                               const int64_t* outputs, int64_t output_count);

// Components of a second-order jet in `variables` (1 to 3) inputs: value,
// gradient, then the upper triangle of the Hessian row by row.
inline int64_t jet_components(int64_t variables) noexcept {
  return 1 + variables + variables * (variables + 1) / 2;
}

// Values of a validated program at `count` points (row-major, input_dimension
// each) into result (count, output_count).
void evaluate_point(const ProgramView& program, const double* points, int64_t count,
                    double* result, PointWorkspace& workspace);

// Second-order jets of a validated program with 1 to 3 inputs into result
// (count, output_count, jet_components(input_dimension)).
void evaluate_jet(const ProgramView& program, const double* points, int64_t count,
                  double* result, PointWorkspace& workspace);

}  // namespace phx::mc
