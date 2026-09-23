#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# Data-only worker: JSON numbers and integer exponents are lowered without eval/include.

using HomotopyContinuation
using JSON3
import HomotopyContinuation.ModelKit: Expression, Variable

const MAX_VARIABLE_COUNT = 4_096
const MAX_EQUATION_COUNT = 4_096
const MAX_TERM_COUNT = 1_000_000
const MAX_EXPONENT_ENTRIES = 10_000_000
const MAX_STORAGE_BYTES = 256 * 1024 * 1024

const INPUT_KEYS = Set([
    "protocol_id",
    "request_id",
    "support_id",
    "system_id",
    "environment_id",
    "policy_id",
    "homotopy_continuation_uuid",
    "homotopy_continuation_version",
    "start_system",
    "path_capacity",
    "seed",
    "maximum_variable_count",
    "maximum_equation_count",
    "maximum_term_count",
    "maximum_exponent_entries",
    "maximum_storage_bytes",
    "equation_count",
    "variable_count",
    "equation_indices",
    "exponents",
    "coefficients",
])
const PATH_STATUSES = (
    "regular_endpoint",
    "singular_endpoint_candidate",
    "at_infinity",
    "excess_solution",
    "tracking_failed",
    "invalid_endpoint",
)

function require_integer(value, name; minimum = 0, maximum = typemax(Int))
    value isa Integer && !(value isa Bool) || error("$(name) must be an integer")
    minimum <= value <= maximum || error("$(name) is outside its bounds")
    return Int(value)
end

function checked_product(left, right, maximum, name)
    left >= 0 && right >= 0 || error("$(name) factors must be nonnegative")
    left == 0 && return 0
    right <= div(maximum, left) || error("$(name) exceeds its bound")
    return left * right
end

function finite_number(value, name)
    value isa Real || error("$(name) must be a real JSON number")
    result = Float64(value)
    isfinite(result) || error("$(name) must be finite")
    return result
end

function parse_input(path)
    filesize(path) <= 64 * 1024 * 1024 || error("input exceeds the worker byte bound")
    record = JSON3.read(read(path, String), Dict{String,Any})
    Set(keys(record)) == INPUT_KEYS || error("input fields do not match the protocol")
    record["protocol_id"] == "phydrax.homotopy-continuation" ||
        error("unsupported homotopy-continuation protocol")
    maximum_variable_count = require_integer(
        record["maximum_variable_count"],
        "maximum_variable_count";
        minimum = 1,
        maximum = MAX_VARIABLE_COUNT,
    )
    maximum_equation_count = require_integer(
        record["maximum_equation_count"],
        "maximum_equation_count";
        minimum = 1,
        maximum = MAX_EQUATION_COUNT,
    )
    maximum_term_count = require_integer(
        record["maximum_term_count"],
        "maximum_term_count";
        minimum = 1,
        maximum = MAX_TERM_COUNT,
    )
    maximum_exponent_entries = require_integer(
        record["maximum_exponent_entries"],
        "maximum_exponent_entries";
        minimum = 1,
        maximum = MAX_EXPONENT_ENTRIES,
    )
    maximum_storage_bytes = require_integer(
        record["maximum_storage_bytes"],
        "maximum_storage_bytes";
        minimum = 1,
        maximum = MAX_STORAGE_BYTES,
    )
    equation_count = require_integer(
        record["equation_count"],
        "equation_count";
        minimum = 1,
        maximum = maximum_equation_count,
    )
    variable_count = require_integer(
        record["variable_count"],
        "variable_count";
        minimum = 1,
        maximum = maximum_variable_count,
    )
    path_capacity = require_integer(
        record["path_capacity"],
        "path_capacity";
        minimum = 1,
        maximum = 10^8,
    )
    seed = require_integer(record["seed"], "seed"; maximum = typemax(UInt32))
    start_system = record["start_system"]
    start_system isa AbstractString || error("start_system must be a string")
    start_system in ("total-degree", "polyhedral") || error("unsupported start system")
    rows = record["equation_indices"]
    powers = record["exponents"]
    coefficients = record["coefficients"]
    rows isa AbstractVector || error("equation_indices must be an array")
    powers isa AbstractVector || error("exponents must be an array")
    coefficients isa AbstractVector || error("coefficients must be an array")
    length(rows) == length(powers) == length(coefficients) ||
        error("term arrays differ in length")
    length(rows) <= maximum_term_count || error("term count exceeds its bound")
    exponent_entries = checked_product(
        length(rows),
        variable_count,
        maximum_exponent_entries,
        "exponent entry count",
    )
    estimated_storage = checked_product(
        exponent_entries,
        sizeof(Int),
        maximum_storage_bytes,
        "exponent storage",
    )
    fixed_storage = variable_count * 64 + equation_count * 64 + length(rows) * 40
    fixed_storage <= maximum_storage_bytes - estimated_storage ||
        error("estimated worker storage exceeds its bound")
    parsed_rows = Int[]
    parsed_powers = Vector{Int}[]
    parsed_coefficients = ComplexF64[]
    sizehint!(parsed_rows, length(rows))
    sizehint!(parsed_powers, length(rows))
    sizehint!(parsed_coefficients, length(rows))
    for term in eachindex(rows)
        row = require_integer(
            rows[term],
            "equation index";
            maximum = equation_count - 1,
        )
        exponent = powers[term]
        exponent isa AbstractVector || error("each exponent must be an array")
        length(exponent) == variable_count ||
            error("exponent width differs from variable_count")
        parsed_exponent = Int[]
        sizehint!(parsed_exponent, variable_count)
        for value in exponent
            push!(parsed_exponent, require_integer(value, "exponent"))
        end
        coefficient = coefficients[term]
        coefficient isa AbstractVector ||
            error("each coefficient must be [real, imaginary]")
        length(coefficient) == 2 ||
            error("each coefficient must have two components")
        push!(parsed_rows, row)
        push!(parsed_powers, parsed_exponent)
        push!(
            parsed_coefficients,
            complex(
                finite_number(coefficient[1], "coefficient real part"),
                finite_number(coefficient[2], "coefficient imaginary part"),
            ),
        )
    end
    record["equation_count"] = equation_count
    record["variable_count"] = variable_count
    record["path_capacity"] = path_capacity
    record["seed"] = seed
    record["equation_indices"] = parsed_rows
    record["exponents"] = parsed_powers
    record["coefficients"] = parsed_coefficients
    return record
end

function build_system(record)
    variable_count = record["variable_count"]
    equation_count = record["equation_count"]
    variables = [Variable(:x, index) for index in 1:variable_count]
    expressions = [Expression(0) for _ in 1:equation_count]
    rows = record["equation_indices"]
    powers = record["exponents"]
    coefficients = record["coefficients"]
    for term in eachindex(rows)
        monomial = Expression(coefficients[term])
        for variable in 1:variable_count
            exponent = powers[term][variable]
            if exponent != 0
                monomial *= variables[variable]^exponent
            end
        end
        equation = rows[term] + 1
        expressions[equation] += monomial
    end
    return System(expressions; variables = variables)
end


function output_template(record, status, start_count, paths)
    counts = Dict(status => 0 for status in PATH_STATUSES)
    for path in paths
        counts[path["status"]] += 1
    end
    return Dict(
        "protocol_id" => record["protocol_id"],
        "request_id" => record["request_id"],
        "support_id" => record["support_id"],
        "system_id" => record["system_id"],
        "environment_id" => record["environment_id"],
        "policy_id" => record["policy_id"],
        "homotopy_continuation_uuid" => string(Base.PkgId(HomotopyContinuation).uuid),
        "homotopy_continuation_version" => string(Base.pkgversion(HomotopyContinuation)),
        "start_system" => record["start_system"],
        "seed" => record["seed"],
        "execution_status" => status,
        "start_count" => start_count,
        "tracked_path_count" => length(paths),
        "counts" => counts,
        "paths" => paths,
    )
end

nullable_nonnegative(value) = isfinite(value) && value >= 0 ? Float64(value) : nothing

function path_record(result, index, variable_count)
    return_code = string(result.return_code)
    status = if HomotopyContinuation.is_success(result)
        if length(result.solution) != variable_count || !all(isfinite, result.solution)
            "invalid_endpoint"
        elseif HomotopyContinuation.is_singular(result)
            "singular_endpoint_candidate"
        else
            "regular_endpoint"
        end
    elseif HomotopyContinuation.is_at_infinity(result)
        "at_infinity"
    elseif HomotopyContinuation.is_excess_solution(result)
        "excess_solution"
    else
        "tracking_failed"
    end
    endpoint = if status in ("regular_endpoint", "singular_endpoint_candidate")
        [[Float64(real(value)), Float64(imag(value))] for value in result.solution]
    else
        nothing
    end
    return Dict(
        "path_index" => index,
        "return_code" => return_code,
        "status" => status,
        "endpoint" => endpoint,
        "provider_residual_norm" => nullable_nonnegative(result.residual),
        "condition_number" => nullable_nonnegative(result.condition_jacobian),
    )
end

function main(input_path, output_path)
    record = parse_input(input_path)
    string(Base.PkgId(HomotopyContinuation).uuid) == record["homotopy_continuation_uuid"] ||
        error("HomotopyContinuation UUID differs from the requested environment")
    string(Base.pkgversion(HomotopyContinuation)) == record["homotopy_continuation_version"] ||
        error("HomotopyContinuation version differs from the requested environment")
    system = build_system(record)
    start_symbol = record["start_system"] == "total-degree" ? :total_degree : :polyhedral
    start_count = HomotopyContinuation.paths_to_track(
        system;
        start_system = start_symbol,
    )
    if start_count > record["path_capacity"]
        output = output_template(record, "path_capacity_exceeded", start_count, Dict{String,Any}[])
        open(output_path, "w") do io
            JSON3.write(io, output)
            write(io, '\n')
        end
        return
    end
    # The same symbol and system used for the bounded forecast are used for tracking.
    result = HomotopyContinuation.solve(
        system;
        start_system = start_symbol,
        seed = UInt32(record["seed"]),
        show_progress = false,
    )
    raw_paths = HomotopyContinuation.path_results(result)
    length(raw_paths) == start_count ||
        error("provider did not return one result per start path")
    path_numbers = [HomotopyContinuation.path_number(path) for path in raw_paths]
    all(number isa Integer for number in path_numbers) ||
        error("provider path result omitted its source path number")
    sort!(raw_paths; by=path -> Int(HomotopyContinuation.path_number(path)))
    sorted_numbers = [
        Int(HomotopyContinuation.path_number(path)) for path in raw_paths
    ]
    sorted_numbers == collect(1:start_count) ||
        error("provider path numbers do not exactly cover start paths")
    paths = [
        path_record(path, sorted_numbers[index] - 1, record["variable_count"])
        for (index, path) in enumerate(raw_paths)
    ]
    output = output_template(record, "complete", start_count, paths)
    open(output_path, "w") do io
        JSON3.write(io, output)
        write(io, '\n')
    end
end

length(ARGS) == 2 || error("usage: worker input.json output.json")
main(ARGS[1], ARGS[2])
