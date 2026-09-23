# Copyright © 2026 PHYDRA, Inc. All rights reserved.
# Fixed data-only worker for the canonical phydrax.homotopy-geometry protocol.

using HomotopyContinuation
using JSON3
using LinearAlgebra
using SHA

const PROTOCOL = "phydrax.homotopy-geometry"
const MAX_REQUEST_BYTES = 32 * 1024 * 1024
const HARD_MAX_DIMENSION = 4_096
const HARD_MAX_ENTITY_COUNT = 1_000_000
const HARD_MAX_TRACKING_COUNT = 2_000_000
const HARD_MAX_TERM_COUNT = 1_000_000
const HARD_MAX_VALUE_COUNT = 10_000_000
const HARD_MAX_STORAGE_BYTES = 256 * 1024 * 1024
const MAX_DIMENSION = Ref(HARD_MAX_DIMENSION)
const MAX_ENTITY_COUNT = Ref(HARD_MAX_ENTITY_COUNT)
const MAX_TERM_COUNT = Ref(HARD_MAX_TERM_COUNT)
const MAX_VALUE_COUNT = Ref(HARD_MAX_VALUE_COUNT)
const MAX_STORAGE_BYTES = Ref(HARD_MAX_STORAGE_BYTES)
const TRACKING_REMAINING = Ref(0)
const ACTIVE_SEED = Ref(UInt32(0))

function require_integer(value, name; minimum=0, maximum=typemax(Int))
    value isa Integer && !(value isa Bool) || error("$(name) must be an integer.")
    minimum <= value <= maximum || error("$(name) is outside its bounds.")
    return Int(value)
end

function checked_product(left, right, maximum, name)
    left >= 0 && right >= 0 || error("$(name) factors must be nonnegative.")
    left == 0 && return 0
    right <= div(maximum, left) || error("$(name) exceeds its bound.")
    return left * right
end

function tracking_allowance(requested)
    requested >= 0 || error("Requested tracking count must be nonnegative.")
    allowed = min(requested, TRACKING_REMAINING[])
    TRACKING_REMAINING[] -= allowed
    return allowed
end

function finite_scalar(value, name)
    value isa Real && !(value isa Bool) || error("$(name) must be a real number.")
    result = Float64(value)
    isfinite(result) || error("$(name) must be finite.")
    return result
end

function complex_scalar(value)
    value isa AbstractVector && length(value) == 2 ||
        error("Complex scalars must contain two components.")
    return ComplexF64(
        finite_scalar(value[1], "complex real component"),
        finite_scalar(value[2], "complex imaginary component"),
    )
end

function complex_vector(record)
    exact_keys(record, ["shape", "values"], "complex vector")
    shape_record = record["shape"]
    shape_record isa AbstractVector || error("Complex vector shape must be an array.")
    length(shape_record) == 1 || error("Complex vector shape must have rank one.")
    length_ = require_integer(
        shape_record[1],
        "complex vector length";
        maximum=MAX_ENTITY_COUNT[],
    )
    values_record = record["values"]
    values_record isa AbstractVector || error("Complex vector values must be an array.")
    length(values_record) == length_ ||
        error("Complex vector value count disagrees with shape.")
    checked_product(length_, 16, MAX_STORAGE_BYTES[], "complex vector bytes")
    return ComplexF64[complex_scalar(value) for value in values_record]
end

function complex_matrix(record)
    exact_keys(record, ["shape", "values"], "complex matrix")
    shape_record = record["shape"]
    shape_record isa AbstractVector || error("Complex matrix shape must be an array.")
    length(shape_record) == 2 || error("Complex matrix shape must have rank two.")
    rows = require_integer(
        shape_record[1],
        "complex matrix rows";
        maximum=MAX_ENTITY_COUNT[],
    )
    columns = require_integer(
        shape_record[2],
        "complex matrix columns";
        maximum=MAX_DIMENSION[],
    )
    count = checked_product(rows, columns, MAX_VALUE_COUNT[], "complex matrix values")
    checked_product(count, 16, MAX_STORAGE_BYTES[], "complex matrix bytes")
    values_record = record["values"]
    values_record isa AbstractVector || error("Complex matrix values must be an array.")
    length(values_record) == count ||
        error("Complex matrix value count disagrees with shape.")
    values = ComplexF64[complex_scalar(value) for value in values_record]
    return permutedims(reshape(values, columns, rows))
end

function complex_rows(record)
    matrix = complex_matrix(record)
    return [collect(matrix[row, :]) for row in axes(matrix, 1)]
end

function rows_matrix(rows, width)
    require_integer(width, "complex row width"; maximum=MAX_DIMENSION[])
    length(rows) <= MAX_ENTITY_COUNT[] || error("Complex row count exceeds its bound.")
    checked_product(length(rows), width, MAX_VALUE_COUNT[], "complex row values")
    isempty(rows) && return zeros(ComplexF64, 0, width)
    all(length(row) == width for row in rows) || error("Complex row widths disagree.")
    return reduce(vcat, (permutedims(row) for row in rows))
end

wire_scalar(value) = [real(value), imag(value)]
wire_vector(values) = Dict(
    "shape" => [length(values)],
    "values" => [wire_scalar(value) for value in values],
)
wire_matrix(values) = Dict(
    "shape" => [size(values, 1), size(values, 2)],
    "values" => [
        wire_scalar(values[row, column])
        for row in axes(values, 1)
        for column in axes(values, 2)
    ],
)

function exact_keys(record, expected, owner)
    observed = Set(String.(keys(record)))
    required = Set(expected)
    observed == required || error("$(owner) fields differ: observed=$(sort!(collect(observed))), expected=$(sort!(collect(required)))")
end

function polynomial_system(system_record)
    exact_keys(
        system_record,
        ["variable_count", "equation_count", "equation_indices", "exponents", "coefficients"],
        "system",
    )
    variable_count = require_integer(
        system_record["variable_count"],
        "variable_count";
        minimum=1,
        maximum=MAX_DIMENSION[],
    )
    equation_count = require_integer(
        system_record["equation_count"],
        "equation_count";
        minimum=1,
        maximum=MAX_DIMENSION[],
    )
    raw_indices = system_record["equation_indices"]
    raw_exponents = system_record["exponents"]
    raw_indices isa AbstractVector || error("equation_indices must be an array.")
    raw_exponents isa AbstractVector || error("exponents must be an array.")
    term_count = length(raw_indices)
    term_count <= MAX_TERM_COUNT[] || error("Sparse system term count exceeds its bound.")
    length(raw_exponents) == term_count ||
        error("Sparse system term arrays do not align.")
    checked_product(
        term_count,
        variable_count,
        MAX_VALUE_COUNT[],
        "sparse exponent entries",
    )
    equation_indices = Int[]
    exponents = Vector{Int}[]
    sizehint!(equation_indices, term_count)
    sizehint!(exponents, term_count)
    for term in eachindex(raw_indices)
        push!(
            equation_indices,
            require_integer(
                raw_indices[term],
                "equation index";
                maximum=equation_count - 1,
            ),
        )
        row = raw_exponents[term]
        row isa AbstractVector && length(row) == variable_count ||
            error("Sparse exponent width differs from variable_count.")
        parsed = Int[]
        sizehint!(parsed, variable_count)
        for value in row
            push!(parsed, require_integer(value, "exponent"))
        end
        push!(exponents, parsed)
    end
    coefficients = complex_vector(system_record["coefficients"])
    length(coefficients) == term_count ||
        error("Sparse system coefficient count does not align.")
    estimated_bytes =
        variable_count * 64 +
        equation_count * 64 +
        term_count * 40 +
        term_count * variable_count * sizeof(Int)
    estimated_bytes <= MAX_STORAGE_BYTES[] ||
        error("Sparse polynomial system exceeds its storage bound.")
    variables = [
        HomotopyContinuation.ModelKit.Variable(Symbol("x$(position)"))
        for position in 1:variable_count
    ]
    equations = [zero(ComplexF64) * variables[1] for _ in 1:equation_count]
    for term in eachindex(coefficients)
        monomial = coefficients[term]
        for variable in 1:variable_count
            monomial *= variables[variable]^exponents[term][variable]
        end
        equations[equation_indices[term] + 1] += monomial
    end
    return variables, equations
end

function affine_equations(variables, matrix, offset)
    size(matrix, 1) == length(offset) || error("Affine slice rows and offsets disagree.")
    size(matrix, 2) == length(variables) || error("Affine slice ambient dimension disagrees.")
    return [sum(matrix[row, column] * variables[column] for column in eachindex(variables)) + offset[row] for row in axes(matrix, 1)]
end

function residual_norm(equations, variables, point)
    values = HomotopyContinuation.evaluate(equations, variables => point)
    return norm(ComplexF64.(values), Inf)
end

function endpoint_slots(result, expected, width)
    raw_paths = HomotopyContinuation.path_results(result)
    length(raw_paths) <= expected ||
        error("Provider returned more path results than the endpoint capacity.")
    slots = Vector{Union{Nothing, Vector{ComplexF64}}}(undef, expected)
    fill!(slots, nothing)
    observed = Set{Int}()
    for path in raw_paths
        number = HomotopyContinuation.path_number(path)
        number isa Integer || error("Provider path result has no source path number.")
        index = Int(number)
        1 <= index <= expected || error("Provider path number is out of range.")
        index in observed && error("Provider returned a duplicate path number.")
        push!(observed, index)
        if HomotopyContinuation.is_success(path)
            point = ComplexF64.(path.solution)
            if length(point) == width && all(isfinite, point)
                slots[index] = point
            end
        end
    end
    return slots
end

function successful_points(slots)
    return Vector{ComplexF64}[point for point in slots if point !== nothing]
end

function solve_slice(equations, variables, matrix, offset, expected)
    allowed = tracking_allowance(expected)
    if allowed < expected
        TRACKING_REMAINING[] += allowed
        slots = Vector{Union{Nothing, Vector{ComplexF64}}}(undef, expected)
        fill!(slots, nothing)
        return slots, 0
    end
    system = System(
        vcat(equations, affine_equations(variables, matrix, offset));
        variables=variables,
    )
    result = solve(system; seed=ACTIVE_SEED[], show_progress=false)
    return endpoint_slots(result, expected, length(variables)), expected
end

function transport_slice(
    equations,
    variables,
    starts,
    source_matrix,
    source_offset,
    target_matrix,
    target_offset,
)
    size(source_matrix) == size(target_matrix) ||
        error("Transport slice matrices disagree.")
    length(source_offset) == length(target_offset) ||
        error("Transport slice offsets disagree.")
    expected = length(starts)
    attempted = tracking_allowance(expected)
    slots = Vector{Union{Nothing, Vector{ComplexF64}}}(undef, expected)
    fill!(slots, nothing)
    attempted == 0 && return slots, attempted
    parameter = HomotopyContinuation.ModelKit.Variable(:phydrax_t)
    source = affine_equations(variables, source_matrix, source_offset)
    target = affine_equations(variables, target_matrix, target_offset)
    slices = [
        (1 - parameter) * source[row] + parameter * target[row]
        for row in eachindex(source)
    ]
    system = System(
        vcat(equations, slices);
        variables=variables,
        parameters=[parameter],
    )
    result = solve(
        system,
        starts[1:attempted];
        start_parameters=[0.0],
        target_parameters=[1.0],
        seed=ACTIVE_SEED[],
        show_progress=false,
    )
    tracked = endpoint_slots(result, attempted, length(variables))
    slots[1:attempted] = tracked
    return slots, attempted
end

function nearest_permutation(starts, endpoints)
    length(starts) == length(endpoints) ||
        error("A completed loop changed the endpoint count.")
    remaining = Set(eachindex(starts))
    permutation = zeros(Int, length(starts))
    for source in eachindex(endpoints)
        target = argmin([
            candidate in remaining ? norm(endpoints[source] - starts[candidate]) : Inf
            for candidate in eachindex(starts)
        ])
        isfinite(norm(endpoints[source] - starts[target])) ||
            error("Loop endpoint matching failed.")
        permutation[source] = target - 1
        delete!(remaining, target)
    end
    return permutation
end

function requested_paths(request)
    paths = request["paths"]
    paths isa AbstractVector || error("paths must be an array.")
    length(paths) <= MAX_ENTITY_COUNT[] || error("Path inventory exceeds its bound.")
    specifications = Vector{Dict{String, Any}}()
    sizehint!(specifications, length(paths))
    observed = Set{String}()
    for path in paths
        exact_keys(path, ["path_id", "batch_id", "source_index"], "path")
        path_id = String(path["path_id"])
        batch_id = String(path["batch_id"])
        !isempty(path_id) && !isempty(batch_id) || error("Path identities must be nonempty.")
        path_id in observed && error("Path identities must be unique.")
        push!(observed, path_id)
        push!(
            specifications,
            Dict(
                "path_id" => path_id,
                "batch_id" => batch_id,
                "source_index" => require_integer(path["source_index"], "source_index"),
            ),
        )
    end
    return specifications
end

function path_records(
    specifications,
    slots,
    attempted;
    residuals=Union{Nothing, Float64}[],
    targets=Int[],
)
    length(slots) == length(specifications) ||
        error("Endpoint slots differ from the requested path inventory.")
    0 <= attempted <= length(specifications) || error("Invalid attempted path count.")
    isempty(residuals) || length(residuals) == length(specifications) ||
        error("Residual inventory differs from requested paths.")
    isempty(targets) || length(targets) == length(specifications) ||
        error("Target inventory differs from requested paths.")
    records = Vector{Dict{String, Any}}()
    sizehint!(records, length(specifications))
    compact_target = 0
    for (position, specification) in enumerate(specifications)
        point = slots[position]
        if point !== nothing
            residual = isempty(residuals) ? 0.0 : residuals[position]
            residual isa Real && isfinite(residual) && residual >= 0 ||
                error("Successful path residual is invalid.")
            target = isempty(targets) ? compact_target : targets[position]
            compact_target += 1
            push!(
                records,
                Dict(
                    "path_id" => specification["path_id"],
                    "batch_id" => specification["batch_id"],
                    "source_index" => specification["source_index"],
                    "target_index" => target,
                    "status" => "success",
                    "residual_norm" => Float64(residual),
                    "diagnostic" => "",
                ),
            )
        else
            not_attempted = position > attempted
            push!(
                records,
                Dict(
                    "path_id" => specification["path_id"],
                    "batch_id" => specification["batch_id"],
                    "source_index" => specification["source_index"],
                    "target_index" => nothing,
                    "status" => not_attempted ? "not-attempted" : "tracking-failed",
                    "residual_norm" => nothing,
                    "diagnostic" => not_attempted ?
                        "tracking capacity exhausted before this path" :
                        "provider returned no finite endpoint for this path",
                ),
            )
        end
    end
    return records
end

function witness_result(equations, variables, payload, specifications; source_points=nothing)
    dimension = require_integer(
        payload["dimension"],
        "dimension";
        maximum=length(variables),
    )
    target_matrix = complex_matrix(payload["slice_matrix"])
    target_offset = complex_vector(payload["slice_offset"])
    if source_points !== nothing
        length(source_points) == length(specifications) ||
            error("Source points differ from requested path inventory.")
    end
    slots, attempted = if source_points === nothing
        solve_slice(
            equations,
            variables,
            target_matrix,
            target_offset,
            length(specifications),
        )
    else
        transport_slice(
            equations,
            variables,
            source_points,
            complex_matrix(payload["source_slice_matrix"]),
            complex_vector(payload["source_slice_offset"]),
            target_matrix,
            target_offset,
        )
    end
    residual_slots = Union{Nothing, Float64}[
        point === nothing ? nothing : residual_norm(equations, variables, point)
        for point in slots
    ]
    endpoints = successful_points(slots)
    residuals = Float64[value for value in residual_slots if value !== nothing]
    records = path_records(
        specifications,
        slots,
        attempted;
        residuals=residual_slots,
    )
    result = Dict(
        "dimension" => dimension,
        "slice_matrix" => wire_matrix(target_matrix),
        "slice_offset" => wire_vector(target_offset),
        "points" => wire_matrix(rows_matrix(endpoints, length(variables))),
        "residual_norms" => residuals,
    )
    return result, records
end

function trace_result(equations, variables, payload, specifications)
    all_starts = complex_rows(payload["source_points"])
    raw_indices = payload["point_indices"]
    raw_indices isa AbstractVector || error("point_indices must be an array.")
    selected_indices = [
        require_integer(
            value,
            "point index";
            maximum=length(all_starts) - 1,
        ) + 1
        for value in raw_indices
    ]
    starts = all_starts[selected_indices]
    matrix = complex_matrix(payload["source_slice_matrix"])
    source_offset = complex_vector(payload["source_slice_offset"])
    sample_offsets = complex_rows(payload["sample_offsets"])
    traces = zeros(ComplexF64, length(sample_offsets), length(variables))
    complete_samples = true
    records = Vector{Dict{String, Any}}()
    for (sample, offset) in enumerate(sample_offsets)
        selected_specs = [
            specification
            for specification in specifications
            if specification["batch_id"] == "trace:$(sample - 1)"
        ]
        length(selected_specs) == length(starts) ||
            error("Trace path batch differs from selected source points.")
        slots, attempted = transport_slice(
            equations,
            variables,
            starts,
            matrix,
            source_offset,
            matrix,
            offset,
        )
        residual_slots = Union{Nothing, Float64}[
            point === nothing ? nothing : residual_norm(equations, variables, point)
            for point in slots
        ]
        targets = [specification["source_index"] for specification in selected_specs]
        append!(
            records,
            path_records(
                selected_specs,
                slots,
                attempted;
                residuals=residual_slots,
                targets=targets,
            ),
        )
        endpoints = successful_points(slots)
        if length(endpoints) != length(starts)
            complete_samples = false
            continue
        end
        traces[sample, :] = sum(endpoints)
    end
    parameters = [
        finite_scalar(value, "sample parameter")
        for value in payload["sample_parameters"]
    ]
    length(parameters) == length(sample_offsets) ||
        error("Trace sample parameters and offsets disagree.")
    design = hcat(ones(length(parameters)), parameters)
    fit = design * (design \ traces)
    fit_residual = complete_samples ? norm(traces - fit, Inf) : floatmax(Float64)
    tolerance = finite_scalar(payload["tolerance"], "trace tolerance")
    tolerance >= 0 || error("Trace tolerance must be nonnegative.")
    result = Dict(
        "witness_set_id" => String(payload["witness_set_id"]),
        "point_indices" => [value - 1 for value in selected_indices],
        "sample_parameters" => parameters,
        "trace_values" => wire_matrix(traces),
        "affine_fit_residual" => fit_residual,
        "tolerance" => tolerance,
        "passed" => complete_samples && fit_residual <= tolerance,
    )
    return result, records
end

function monodromy_result(equations, variables, payload, specifications)
    starts = complex_rows(payload["source_points"])
    source_matrix = complex_matrix(payload["slice_matrix"])
    source_offset = complex_vector(payload["slice_offset"])
    completed = Vector{Dict{String, Any}}()
    records = Vector{Dict{String, Any}}()
    attempted_loop_ids = String[]
    for loop in payload["loops"]
        loop_id = String(loop["loop_id"])
        push!(attempted_loop_ids, loop_id)
        selected = [
            specification
            for specification in specifications
            if specification["batch_id"] == loop_id
        ]
        length(selected) == length(starts) ||
            error("Monodromy path batch differs from source points.")
        midpoint_matrix = complex_matrix(loop["midpoint_matrix"])
        midpoint_offset = complex_vector(loop["midpoint_offset"])
        outbound_slots, outbound_attempted = transport_slice(
            equations,
            variables,
            starts,
            source_matrix,
            source_offset,
            midpoint_matrix,
            midpoint_offset,
        )
        outbound = successful_points(outbound_slots)
        if length(outbound) != length(starts)
            failed = Vector{Union{Nothing, Vector{ComplexF64}}}(undef, length(starts))
            fill!(failed, nothing)
            append!(records, path_records(selected, failed, outbound_attempted))
            continue
        end
        returned_slots, returned_attempted = transport_slice(
            equations,
            variables,
            outbound,
            midpoint_matrix,
            midpoint_offset,
            source_matrix,
            source_offset,
        )
        returned = successful_points(returned_slots)
        if length(returned) != length(starts)
            failed = Vector{Union{Nothing, Vector{ComplexF64}}}(undef, length(starts))
            fill!(failed, nothing)
            append!(records, path_records(selected, failed, returned_attempted))
            continue
        end
        permutation = nearest_permutation(starts, returned)
        residual_slots = Union{Nothing, Float64}[
            residual_norm(equations, variables, point)
            for point in returned
        ]
        append!(
            records,
            path_records(
                selected,
                returned_slots,
                returned_attempted;
                residuals=residual_slots,
                targets=permutation,
            ),
        )
        push!(
            completed,
            Dict("loop_id" => loop_id, "permutation" => permutation),
        )
    end
    result = Dict(
        "witness_set_id" => String(payload["witness_set_id"]),
        "point_count" => length(starts),
        "attempted_loop_ids" => attempted_loop_ids,
        "completed" => completed,
    )
    return result, records
end

function image_result(equations, variables, payload, specifications)
    target_count = require_integer(
        payload["map_equation_count"],
        "map_equation_count";
        minimum=1,
        maximum=MAX_DIMENSION[],
    )
    image_variables = [
        HomotopyContinuation.ModelKit.Variable(Symbol("y$(position)"))
        for position in 1:target_count
    ]
    map_record = Dict(
        "variable_count" => length(variables),
        "equation_count" => target_count,
        "equation_indices" => payload["map_equation_indices"],
        "exponents" => payload["map_exponents"],
        "coefficients" => payload["map_coefficients"],
    )
    _, map_equations = polynomial_system(map_record)
    graph = [
        image_variables[position] - map_equations[position]
        for position in eachindex(image_variables)
    ]
    source_matrix = complex_matrix(payload["source_slice_matrix"])
    source_offset = complex_vector(payload["source_slice_offset"])
    image_matrix = complex_matrix(payload["image_slice_matrix"])
    image_offset = complex_vector(payload["image_slice_offset"])
    combined_variables = vcat(variables, image_variables)
    length(combined_variables) <= MAX_DIMENSION[] ||
        error("Combined image system dimension exceeds its bound.")
    combined_equations = vcat(
        equations,
        graph,
        affine_equations(variables, source_matrix, source_offset),
        affine_equations(image_variables, image_matrix, image_offset),
    )
    expected = length(specifications)
    allowed = tracking_allowance(expected)
    slots = Vector{Union{Nothing, Vector{ComplexF64}}}(undef, expected)
    fill!(slots, nothing)
    attempted = 0
    if allowed == expected
        system = System(combined_equations; variables=combined_variables)
        solved = solve(system; seed=ACTIVE_SEED[], show_progress=false)
        slots = endpoint_slots(solved, expected, length(combined_variables))
        attempted = expected
    else
        TRACKING_REMAINING[] += allowed
    end
    endpoints = successful_points(slots)
    source_points = [point[1:length(variables)] for point in endpoints]
    image_points = [point[length(variables)+1:end] for point in endpoints]
    residual_slots = Union{Nothing, Float64}[
        point === nothing ?
            nothing :
            residual_norm(combined_equations, combined_variables, point)
        for point in slots
    ]
    residuals = Float64[value for value in residual_slots if value !== nothing]
    records = path_records(
        specifications,
        slots,
        attempted;
        residuals=residual_slots,
    )
    image_degree = length(unique(Tuple(point) for point in image_points))
    result = Dict(
        "source_system_id" => String(payload["source_system_id"]),
        "map_id" => String(payload["map_id"]),
        "source_dimension" => require_integer(
            payload["source_dimension"],
            "source_dimension";
            maximum=length(variables),
        ),
        "image_dimension" => require_integer(
            payload["image_dimension"],
            "image_dimension";
            maximum=target_count,
        ),
        "source_slice_matrix" => wire_matrix(source_matrix),
        "source_slice_offset" => wire_vector(source_offset),
        "image_slice_matrix" => wire_matrix(image_matrix),
        "image_slice_offset" => wire_vector(image_offset),
        "source_points" => wire_matrix(rows_matrix(source_points, length(variables))),
        "image_points" => wire_matrix(rows_matrix(image_points, length(image_variables))),
        "residual_norms" => residuals,
        "image_degree" => image_degree,
    )
    return result, records
end

function membership_result(equations, variables, payload, specifications)
    queries = complex_rows(payload["query_points"])
    tolerance = finite_scalar(payload["tolerance"], "membership tolerance")
    tolerance >= 0 || error("Membership tolerance must be nonnegative.")
    residuals = [residual_norm(equations, variables, point) for point in queries]
    member_witness_set_ids = [String[] for _ in queries]
    records = Vector{Dict{String, Any}}()
    for (query_index, query) in enumerate(queries)
        for witness in payload["witness_sets"]
            witness_id = String(witness["witness_set_id"])
            starts = complex_rows(witness["points"])
            matrix = complex_matrix(witness["slice_matrix"])
            source_offset = complex_vector(witness["slice_offset"])
            target_offset = -matrix * query
            batch = "membership:$(query_index - 1):$(witness_id)"
            selected = [
                specification
                for specification in specifications
                if specification["batch_id"] == batch
            ]
            length(selected) == length(starts) ||
                error("Membership path batch differs from witness points.")
            slots, attempted = transport_slice(
                equations,
                variables,
                starts,
                matrix,
                source_offset,
                matrix,
                target_offset,
            )
            residual_slots = Union{Nothing, Float64}[
                point === nothing ? nothing : residual_norm(equations, variables, point)
                for point in slots
            ]
            targets = [specification["source_index"] for specification in selected]
            append!(
                records,
                path_records(
                    selected,
                    slots,
                    attempted;
                    residuals=residual_slots,
                    targets=targets,
                ),
            )
            endpoints = successful_points(slots)
            if length(endpoints) == length(starts) && any(
                norm(endpoint - query, Inf) <= tolerance for endpoint in endpoints
            ) && residuals[query_index] <= tolerance
                push!(member_witness_set_ids[query_index], witness_id)
            end
        end
    end
    result = Dict(
        "query_points" => wire_matrix(rows_matrix(queries, length(variables))),
        "member_witness_set_ids" => member_witness_set_ids,
        "residual_norms" => residuals,
        "tolerance" => tolerance,
        "claim" => "numerical-witness-transport-membership-not-exact-ideal-membership",
    )
    return result, records
end

function verify_runtime(runtime)
    exact_keys(
        runtime,
        [
            "project_sha256",
            "manifest_sha256",
            "homotopy_continuation_uuid",
            "homotopy_continuation_version",
            "worker_sha256",
        ],
        "runtime",
    )
    observed = Dict(
        "project_sha256" => bytes2hex(sha256(read("julia-project/Project.toml"))),
        "manifest_sha256" => bytes2hex(sha256(read("julia-project/Manifest.toml"))),
        "homotopy_continuation_uuid" => string(Base.PkgId(HomotopyContinuation).uuid),
        "homotopy_continuation_version" => string(Base.pkgversion(HomotopyContinuation)),
        "worker_sha256" => bytes2hex(sha256(read(abspath(PROGRAM_FILE)))),
    )
    for (name, value) in observed
        String(runtime[name]) == value || error(
            "Pinned runtime identity mismatch for $(name)."
        )
    end
    return nothing
end

function configure_policy(policy)
    exact_keys(
        policy,
        [
            "path_capacity",
            "tracking_capacity",
            "loop_capacity",
            "stage_capacity",
            "seed",
            "maximum_dimension",
            "maximum_entity_count",
            "maximum_term_count",
            "maximum_value_count",
            "maximum_storage_bytes",
        ],
        "policy",
    )
    path_capacity = require_integer(
        policy["path_capacity"],
        "path_capacity";
        minimum=1,
        maximum=HARD_MAX_ENTITY_COUNT,
    )
    tracking_capacity = require_integer(
        policy["tracking_capacity"],
        "tracking_capacity";
        minimum=1,
        maximum=HARD_MAX_TRACKING_COUNT,
    )
    require_integer(
        policy["loop_capacity"],
        "loop_capacity";
        minimum=1,
        maximum=HARD_MAX_ENTITY_COUNT,
    )
    require_integer(
        policy["stage_capacity"],
        "stage_capacity";
        minimum=1,
        maximum=HARD_MAX_ENTITY_COUNT,
    )
    seed = require_integer(policy["seed"], "seed"; maximum=typemax(UInt32))
    MAX_DIMENSION[] = require_integer(
        policy["maximum_dimension"],
        "maximum_dimension";
        minimum=1,
        maximum=HARD_MAX_DIMENSION,
    )
    MAX_ENTITY_COUNT[] = require_integer(
        policy["maximum_entity_count"],
        "maximum_entity_count";
        minimum=1,
        maximum=HARD_MAX_ENTITY_COUNT,
    )
    MAX_TERM_COUNT[] = require_integer(
        policy["maximum_term_count"],
        "maximum_term_count";
        minimum=1,
        maximum=HARD_MAX_TERM_COUNT,
    )
    MAX_VALUE_COUNT[] = require_integer(
        policy["maximum_value_count"],
        "maximum_value_count";
        minimum=1,
        maximum=HARD_MAX_VALUE_COUNT,
    )
    MAX_STORAGE_BYTES[] = require_integer(
        policy["maximum_storage_bytes"],
        "maximum_storage_bytes";
        minimum=1,
        maximum=HARD_MAX_STORAGE_BYTES,
    )
    TRACKING_REMAINING[] = tracking_capacity
    ACTIVE_SEED[] = UInt32(seed)
    return path_capacity
end

function execute(request)
    exact_keys(
        request,
        ["protocol", "request_id", "provider_id", "environment_id", "worker_sha256", "runtime", "operation", "system_id", "support_id", "policy", "system", "payload", "paths"],
        "request",
    )
    String(request["protocol"]) == PROTOCOL ||
        error("Unsupported homotopy-geometry protocol.")
    verify_runtime(request["runtime"])
    path_capacity = configure_policy(request["policy"])
    specifications = requested_paths(request)
    length(specifications) <= path_capacity ||
        error("Requested paths exceed policy capacity.")
    variables, equations = polynomial_system(request["system"])
    operation = String(request["operation"])
    payload = request["payload"]
    result, records = if operation == "generic-slice"
        witness_result(equations, variables, payload, specifications)
    elseif operation == "witness-transport"
        source_points = complex_rows(payload["source_points"])
        witness_result(equations, variables, payload, specifications; source_points=source_points)
    elseif operation == "trace-test"
        trace_result(equations, variables, payload, specifications)
    elseif operation == "monodromy"
        monodromy_result(equations, variables, payload, specifications)
    elseif operation == "regeneration-stage"
        raw_indices = payload["equation_indices"]
        raw_indices isa AbstractVector ||
            error("Regeneration equation_indices must be an array.")
        selected = [
            equations[
                require_integer(
                    equation,
                    "regeneration equation index";
                    maximum=length(equations) - 1,
                ) + 1
            ]
            for equation in raw_indices
        ]
        witness_result(selected, variables, payload, specifications)
    elseif operation == "image-degree"
        image_result(equations, variables, payload, specifications)
    elseif operation == "membership"
        membership_result(equations, variables, payload, specifications)
    else
        error("Unsupported homotopy-geometry operation $(operation).")
    end
    verify_runtime(request["runtime"])
    length(records) == length(specifications) ||
        error("Worker path records do not exactly cover requested paths.")
    Set(record["path_id"] for record in records) ==
        Set(specification["path_id"] for specification in specifications) ||
        error("Worker path record identities differ from requested paths.")
    budget_exhausted = any(record["status"] == "not-attempted" for record in records)
    failed = any(record["status"] != "success" for record in records)
    status = budget_exhausted ? "budget-exhausted" : failed ? "partial-path-failure" : operation == "trace-test" && !Bool(result["passed"]) ? "trace-test-failed" : "success"
    return Dict(
        "protocol" => PROTOCOL,
        "request_id" => String(request["request_id"]),
        "provider_id" => String(request["provider_id"]),
        "environment_id" => String(request["environment_id"]),
        "worker_sha256" => String(request["worker_sha256"]),
        "operation" => operation,
        "system_id" => String(request["system_id"]),
        "support_id" => String(request["support_id"]),
        "status" => status,
        "budget_exhausted" => budget_exhausted,
        "seed" => Int(ACTIVE_SEED[]),
        "paths" => records,
        "result" => result,
    )
end

length(ARGS) == 2 || error("Expected request and result file arguments.")
filesize(ARGS[1]) <= MAX_REQUEST_BYTES ||
    error("Homotopy-geometry request exceeds its byte bound.")
request_text = read(ARGS[1], String)
request_text == strip(request_text) ||
    error("Homotopy-geometry request has leading or trailing bytes.")
request = JSON3.read(request_text)
result = execute(request)
open(ARGS[2], "w") do stream
    JSON3.write(stream, result)
end
