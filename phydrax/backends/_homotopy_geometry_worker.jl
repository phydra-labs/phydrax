# Copyright © 2026 PHYDRA, Inc. All rights reserved.
# Fixed data-only worker for the canonical phydrax.homotopy-geometry protocol.

using HomotopyContinuation
using JSON3
using LinearAlgebra
using SHA

const PROTOCOL = "phydrax.homotopy-geometry"

complex_scalar(value) = ComplexF64(Float64(value[1]), Float64(value[2]))

function complex_vector(record)
    exact_keys(record, ["shape", "values"], "complex vector")
    shape = Int.(record["shape"])
    length(shape) == 1 || error("Complex vector shape must have rank one.")
    values = ComplexF64[complex_scalar(value) for value in record["values"]]
    length(values) == shape[1] || error("Complex vector value count disagrees with shape.")
    return values
end

function complex_matrix(record)
    exact_keys(record, ["shape", "values"], "complex matrix")
    shape = Int.(record["shape"])
    length(shape) == 2 || error("Complex matrix shape must have rank two.")
    values = ComplexF64[complex_scalar(value) for value in record["values"]]
    length(values) == shape[1] * shape[2] || error("Complex matrix value count disagrees with shape.")
    return permutedims(reshape(values, shape[2], shape[1]))
end

function complex_rows(record)
    matrix = complex_matrix(record)
    return [collect(matrix[row, :]) for row in axes(matrix, 1)]
end

function rows_matrix(rows, width)
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
    variable_count = Int(system_record["variable_count"])
    equation_count = Int(system_record["equation_count"])
    variables = [HomotopyContinuation.ModelKit.Variable(Symbol("x$(position)")) for position in 1:variable_count]
    equations = [zero(ComplexF64) * variables[1] for _ in 1:equation_count]
    equation_indices = Int.(system_record["equation_indices"])
    exponents = system_record["exponents"]
    coefficients = complex_vector(system_record["coefficients"])
    length(equation_indices) == length(exponents) == length(coefficients) || error("Sparse system term arrays do not align.")
    for term in eachindex(coefficients)
        monomial = coefficients[term]
        for variable in 1:variable_count
            monomial *= variables[variable]^Int(exponents[term][variable])
        end
        equation = equation_indices[term] + 1
        1 <= equation <= equation_count || error("Sparse equation index is out of bounds.")
        equations[equation] += monomial
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

function finite_solutions(result)
    return [ComplexF64.(solution) for solution in solutions(result; only_nonsingular=false, only_finite=true)]
end

function solve_slice(equations, variables, matrix, offset)
    system = System(vcat(equations, affine_equations(variables, matrix, offset)); variables=variables)
    return finite_solutions(solve(system; show_progress=false))
end

function transport_slice(equations, variables, starts, source_matrix, source_offset, target_matrix, target_offset)
    size(source_matrix) == size(target_matrix) || error("Transport slice matrices disagree.")
    length(source_offset) == length(target_offset) || error("Transport slice offsets disagree.")
    parameter = HomotopyContinuation.ModelKit.Variable(:phydrax_t)
    source = affine_equations(variables, source_matrix, source_offset)
    target = affine_equations(variables, target_matrix, target_offset)
    slices = [(1 - parameter) * source[row] + parameter * target[row] for row in eachindex(source)]
    system = System(vcat(equations, slices); variables=variables, parameters=[parameter])
    result = solve(
        system,
        starts;
        start_parameters=[0.0],
        target_parameters=[1.0],
        show_progress=false,
    )
    return finite_solutions(result)
end

function nearest_permutation(starts, endpoints)
    length(starts) == length(endpoints) || error("A completed loop changed the endpoint count.")
    remaining = Set(eachindex(starts))
    permutation = zeros(Int, length(starts))
    for source in eachindex(endpoints)
        target = argmin([candidate in remaining ? norm(endpoints[source] - starts[candidate]) : Inf for candidate in eachindex(starts)])
        isfinite(norm(endpoints[source] - starts[target])) || error("Loop endpoint matching failed.")
        permutation[source] = target - 1
        delete!(remaining, target)
    end
    return permutation
end

function requested_paths(request)
    return [Dict(
        "path_id" => String(path["path_id"]),
        "batch_id" => String(path["batch_id"]),
        "source_index" => Int(path["source_index"]),
    ) for path in request["paths"]]
end

function path_records(specifications, endpoints; residuals=Float64[], targets=Int[], failure_status="tracking-failed")
    records = Vector{Dict{String, Any}}()
    for (position, specification) in enumerate(specifications)
        if position <= length(endpoints)
            target = isempty(targets) ? position - 1 : targets[position]
            residual = isempty(residuals) ? 0.0 : residuals[position]
            push!(records, Dict(
                "path_id" => specification["path_id"],
                "batch_id" => specification["batch_id"],
                "source_index" => specification["source_index"],
                "target_index" => target,
                "status" => "success",
                "residual_norm" => residual,
                "diagnostic" => "",
            ))
        else
            push!(records, Dict(
                "path_id" => specification["path_id"],
                "batch_id" => specification["batch_id"],
                "source_index" => specification["source_index"],
                "target_index" => nothing,
                "status" => failure_status,
                "residual_norm" => nothing,
                "diagnostic" => "provider returned no finite endpoint for this path",
            ))
        end
    end
    if length(endpoints) > length(specifications) && !isempty(records)
        records[end]["target_index"] = nothing
        records[end]["status"] = "invalid-endpoint"
        records[end]["residual_norm"] = nothing
        records[end]["diagnostic"] = "provider returned more finite endpoints than requested paths"
    end
    return records
end

function witness_result(equations, variables, payload, specifications; source_points=nothing)
    dimension = Int(payload["dimension"])
    target_matrix = complex_matrix(payload["slice_matrix"])
    target_offset = complex_vector(payload["slice_offset"])
    endpoints = if source_points === nothing
        solve_slice(equations, variables, target_matrix, target_offset)
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
    residuals = [residual_norm(equations, variables, endpoint) for endpoint in endpoints]
    records = path_records(specifications, endpoints; residuals=residuals)
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
    selected_indices = Int.(payload["point_indices"]) .+ 1
    starts = all_starts[selected_indices]
    matrix = complex_matrix(payload["source_slice_matrix"])
    source_offset = complex_vector(payload["source_slice_offset"])
    sample_offsets = complex_rows(payload["sample_offsets"])
    traces = zeros(ComplexF64, length(sample_offsets), length(variables))
    complete_samples = true
    records = Vector{Dict{String, Any}}()
    for (sample, offset) in enumerate(sample_offsets)
        endpoints = transport_slice(
            equations,
            variables,
            starts,
            matrix,
            source_offset,
            matrix,
            offset,
        )
        selected_specs = [
            specification
            for specification in specifications
            if specification["batch_id"] == "trace:$(sample - 1)"
        ]
        endpoint_residuals = [
            residual_norm(equations, variables, point) for point in endpoints
        ]
        targets = [specification["source_index"] for specification in selected_specs]
        append!(
            records,
            path_records(
                selected_specs,
                endpoints;
                residuals=endpoint_residuals,
                targets=targets,
            ),
        )
        if length(endpoints) != length(starts)
            complete_samples = false
            continue
        end
        traces[sample, :] = sum(endpoints)
    end
    parameters = Float64.(payload["sample_parameters"])
    design = hcat(ones(length(parameters)), parameters)
    fit = design * (design \ traces)
    fit_residual = complete_samples ? norm(traces - fit, Inf) : floatmax(Float64)
    result = Dict(
        "witness_set_id" => String(payload["witness_set_id"]),
        "point_indices" => Int.(payload["point_indices"]),
        "sample_parameters" => parameters,
        "trace_values" => wire_matrix(traces),
        "affine_fit_residual" => fit_residual,
        "tolerance" => Float64(payload["tolerance"]),
        "passed" => complete_samples && fit_residual <= Float64(payload["tolerance"]),
    )
    return result, records
end

function monodromy_result(equations, variables, payload, specifications)
    starts = complex_rows(payload["source_points"])
    source_matrix = complex_matrix(payload["slice_matrix"])
    source_offset = complex_vector(payload["slice_offset"])
    completed = Vector{Dict{String, Any}}()
    endpoint_by_batch = Dict{String, Vector{Vector{ComplexF64}}}()
    for loop in payload["loops"]
        loop_id = String(loop["loop_id"])
        midpoint_matrix = complex_matrix(loop["midpoint_matrix"])
        midpoint_offset = complex_vector(loop["midpoint_offset"])
        outbound = transport_slice(equations, variables, starts, source_matrix, source_offset, midpoint_matrix, midpoint_offset)
        if length(outbound) != length(starts)
            endpoint_by_batch[loop_id] = outbound
            continue
        end
        returned = transport_slice(equations, variables, outbound, midpoint_matrix, midpoint_offset, source_matrix, source_offset)
        endpoint_by_batch[loop_id] = returned
        length(returned) == length(starts) || continue
        push!(completed, Dict("loop_id" => loop_id, "permutation" => nearest_permutation(starts, returned)))
    end
    records = Vector{Dict{String, Any}}()
    for loop in payload["loops"]
        loop_id = String(loop["loop_id"])
        selected = [specification for specification in specifications if specification["batch_id"] == loop_id]
        endpoints = get(endpoint_by_batch, loop_id, Vector{Vector{ComplexF64}}())
        targets = length(endpoints) == length(starts) ? nearest_permutation(starts, endpoints) : Int[]
        append!(records, path_records(selected, endpoints; residuals=[residual_norm(equations, variables, point) for point in endpoints], targets=targets))
    end
    result = Dict(
        "witness_set_id" => String(payload["witness_set_id"]),
        "point_count" => length(starts),
        "attempted_loop_ids" => [String(loop["loop_id"]) for loop in payload["loops"]],
        "completed" => completed,
    )
    return result, records
end

function image_result(equations, variables, payload, specifications)
    target_count = Int(payload["map_equation_count"])
    image_variables = [HomotopyContinuation.ModelKit.Variable(Symbol("y$(position)")) for position in 1:target_count]
    map_record = Dict(
        "variable_count" => length(variables),
        "equation_count" => target_count,
        "equation_indices" => payload["map_equation_indices"],
        "exponents" => payload["map_exponents"],
        "coefficients" => payload["map_coefficients"],
    )
    _, map_equations = polynomial_system(map_record)
    graph = [image_variables[position] - map_equations[position] for position in eachindex(image_variables)]
    source_matrix = complex_matrix(payload["source_slice_matrix"])
    source_offset = complex_vector(payload["source_slice_offset"])
    image_matrix = complex_matrix(payload["image_slice_matrix"])
    image_offset = complex_vector(payload["image_slice_offset"])
    combined_variables = vcat(variables, image_variables)
    combined_equations = vcat(
        equations,
        graph,
        affine_equations(variables, source_matrix, source_offset),
        affine_equations(image_variables, image_matrix, image_offset),
    )
    endpoints = finite_solutions(solve(System(combined_equations; variables=combined_variables); show_progress=false))
    source_points = [point[1:length(variables)] for point in endpoints]
    image_points = [point[length(variables)+1:end] for point in endpoints]
    residuals = [residual_norm(combined_equations, combined_variables, endpoint) for endpoint in endpoints]
    records = path_records(specifications, endpoints; residuals=residuals)
    image_degree = length(unique(Tuple(point) for point in image_points))
    result = Dict(
        "source_system_id" => String(payload["source_system_id"]),
        "map_id" => String(payload["map_id"]),
        "source_dimension" => Int(payload["source_dimension"]),
        "image_dimension" => Int(payload["image_dimension"]),
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
    tolerance = Float64(payload["tolerance"])
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
            endpoints = transport_slice(
                equations,
                variables,
                starts,
                matrix,
                source_offset,
                matrix,
                target_offset,
            )
            batch = "membership:$(query_index - 1):$(witness_id)"
            selected = [
                specification
                for specification in specifications
                if specification["batch_id"] == batch
            ]
            endpoint_residuals = [
                residual_norm(equations, variables, point) for point in endpoints
            ]
            targets = [specification["source_index"] for specification in selected]
            append!(
                records,
                path_records(
                    selected,
                    endpoints;
                    residuals=endpoint_residuals,
                    targets=targets,
                ),
            )
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

function execute(request)
    exact_keys(
        request,
        ["protocol", "request_id", "provider_id", "environment_id", "worker_sha256", "runtime", "operation", "system_id", "support_id", "policy", "system", "payload", "paths"],
        "request",
    )
    String(request["protocol"]) == PROTOCOL || error("Unsupported homotopy-geometry protocol.")
    verify_runtime(request["runtime"])
    specifications = requested_paths(request)
    length(specifications) <= Int(request["policy"]["path_capacity"]) || error("Requested paths exceed policy capacity.")
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
        selected = [equations[Int(equation) + 1] for equation in payload["equation_indices"]]
        witness_result(selected, variables, payload, specifications)
    elseif operation == "image-degree"
        image_result(equations, variables, payload, specifications)
    elseif operation == "membership"
        membership_result(equations, variables, payload, specifications)
    else
        error("Unsupported homotopy-geometry operation $(operation).")
    end
    verify_runtime(request["runtime"])
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
        "paths" => records,
        "result" => result,
    )
end

length(ARGS) == 2 || error("Expected request and result file arguments.")
request_bytes = read(ARGS[1])
request = JSON3.read(request_bytes)
result = execute(request)
open(ARGS[2], "w") do stream
    JSON3.write(stream, result)
end
