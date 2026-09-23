-- Copyright © 2026 PHYDRA, Inc. All rights reserved.
--
-- Fixed Phydrax exact-polynomial worker.  The process receives one JSON object on
-- standard input and emits one JSON object on standard output.  It has no general
-- expression entry point and its operation inventory is closed below.

jsonDecode := value(?? Core#"private dictionary"#"fromJSON0");
if jsonDecode === null then error "this pinned Macaulay2 build has no core JSON decoder";
request := jsonDecode get stdio;

requiredKeys := sort {
    "coefficients", "domain", "environment_id", "equation_count",
    "equation_indices", "executable_sha256", "exponents",
    "installation_inventory", "operation", "operation_args", "plan_id",
    "provider_id", "provider_version", "request_id", "resource_limits",
    "support_id", "system_id", "variables", "worker_sha256"
    };
if class request =!= HashTable or sort keys request =!= requiredKeys then
    error "request field inventory mismatch";

jsonEncode := x -> (
    if instance(x, String) then format x
    else if instance(x, Boolean) or instance(x, ZZ) then toString x
    else if x === null then "null"
    else if instance(x, List) then
        "[" | demark(",", apply(x, jsonEncode)) | "]"
    else if class x === HashTable then (
        kk := sort keys x;
        "{" | demark(",", apply(kk, k -> format k | ":" | jsonEncode(x#k))) | "}")
    else error "worker attempted to emit a non-JSON datum");

digits := {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
parseInteger := text -> (
    if not instance(text, String) or #text === 0 then error "invalid integer coefficient";
    chars := characters text;
    negative := first chars === "-";
    if negative then chars = drop(chars, 1);
    if #chars === 0 then error "invalid integer coefficient";
    answer := 0;
    scan(chars, c -> (
        digit := position(digits, d -> d === c);
        if digit === null then error "invalid integer coefficient";
        answer = 10 * answer + digit));
    if negative then -answer else answer);

parseRational := (ring, text) -> (
    chars := characters text;
    slash := position(chars, c -> c === "/");
    if slash === null then promote(parseInteger text, ring)
    else (
        if any(drop(chars, slash + 1), c -> c === "/") then
            error "invalid rational coefficient";
        numerator := parseInteger concatenate take(chars, slash);
        denominator := parseInteger concatenate drop(chars, slash + 1);
        if denominator <= 0 then error "invalid rational denominator";
        promote(numerator, ring) / promote(denominator, ring)));

domain := request#"domain";
if class domain =!= HashTable or not domain#?"kind" then
    error "invalid coefficient domain";
kind := domain#"kind";
coefficientRing := (
    if kind === "ZZ" then ZZ
    else if kind === "QQ" then QQ
    else if kind === "GF" then (
        if sort keys domain =!= {"kind", "modulus"} then error "invalid finite field";
        modulus := domain#"modulus";
        if not instance(modulus, ZZ) or modulus < 2 or modulus > 2147483647
            or not isPrime modulus then
            error "invalid finite-field modulus";
        ZZ/modulus)
    else error "unsupported coefficient domain");
if (kind === "ZZ" or kind === "QQ") and sort keys domain =!= {"kind"} then
    error "invalid characteristic-zero domain";

operation := request#"operation";
allowedOperations := {
    "discriminant_univariate", "eliminate", "groebner_basis",
    "normal_form", "resultant_univariate"
    };
if not member(operation, allowedOperations) then error "unsupported exact operation";

if not instance(request#"variables", List) then
    error "worker variables must be an array";
variableCount := #(request#"variables");
if request#"variables" =!= apply(variableCount, i -> "x" | toString i) then
    error "worker variables must be canonically numbered";

arguments := request#"operation_args";
if class arguments =!= HashTable then error "operation arguments must be an object";
argumentKeys := sort keys arguments;
expectedArgumentKeys := (
    if operation === "groebner_basis" then {"monomial_order"}
    else if operation === "normal_form" then {"monomial_order", "polynomial"}
    else if operation === "eliminate" then {"variable_indices"}
    else if operation === "resultant_univariate" then {"equation_indices", "variable_index"}
    else {"equation_index", "variable_index"});
if argumentKeys =!= sort expectedArgumentKeys then
    error "operation argument field inventory mismatch";
if operation === "normal_form" then (
    dividendKeys := sort keys arguments#"polynomial";
    if dividendKeys =!= sort {
        "coefficients", "domain", "equation_count", "equation_indices",
        "exponents", "support_id", "system_id", "variable_count"
        } then error "normal form polynomial field inventory mismatch");
if (operation === "resultant_univariate" or operation === "discriminant_univariate")
    and arguments#"variable_index" =!= 0 then error "invalid univariate variable";
eliminated := if operation === "eliminate" then arguments#"variable_indices" else {};
kept := select(toList(0 .. variableCount - 1), i -> not member(i, eliminated));
permutation := join(eliminated, kept);
if operation === "eliminate" and (
    #eliminated === 0 or #kept === 0 or sort unique eliminated =!= eliminated) then
    error "invalid elimination variable inventory";

orderName := (
    if operation === "groebner_basis" or operation === "normal_form"
    then arguments#"monomial_order" else "grevlex");
if not member(orderName, {"grevlex", "lex"}) then error "unsupported monomial order";
monomialOrder := (
    if operation === "eliminate" then Eliminate(#eliminated)
    else if orderName === "lex" then Lex else GRevLex);

limits := request#"resource_limits";
limitKeys := {
    "maximum_equation_count", "maximum_exponent_entries",
    "maximum_storage_bytes", "maximum_term_count", "maximum_variable_count"
    };
if class limits =!= HashTable or sort keys limits =!= sort limitKeys then
    error "resource limit field inventory mismatch";
positiveLimit := (name, hard) -> (
    value := limits#name;
    if not instance(value, ZZ) or value <= 0 or value > hard then
        error("invalid resource limit " | name);
    value);
maximumVariableCount := positiveLimit("maximum_variable_count", 4096);
maximumEquationCount := positiveLimit("maximum_equation_count", 100000);
maximumTermCount := positiveLimit("maximum_term_count", 1000000);
maximumExponentEntries := positiveLimit("maximum_exponent_entries", 10000000);
maximumStorageBytes := positiveLimit("maximum_storage_bytes", 268435456);
if variableCount > maximumVariableCount then
    error "variable count exceeds its resource bound";

resourceRecords := if operation === "normal_form"
    then {request, arguments#"polynomial"} else {request};
totalTerms := 0;
totalExponentEntries := 0;
estimatedStorage := variableCount * 64;
scan(resourceRecords, record -> (
    eqCount := record#"equation_count";
    eqIndices := record#"equation_indices";
    exponentRows := record#"exponents";
    coefficientRows := record#"coefficients";
    if not instance(eqCount, ZZ) or eqCount <= 0
        or eqCount > maximumEquationCount then
        error "equation count exceeds its resource bound";
    if not instance(eqIndices, List) or not instance(exponentRows, List)
        or not instance(coefficientRows, List)
        or #eqIndices =!= #exponentRows or #eqIndices =!= #coefficientRows then
        error "invalid sparse polynomial arrays";
    totalTerms = totalTerms + #eqIndices;
    if totalTerms > maximumTermCount then
        error "term count exceeds its resource bound";
    totalExponentEntries = totalExponentEntries + #eqIndices * variableCount;
    if totalExponentEntries > maximumExponentEntries then
        error "exponent entry count exceeds its resource bound";
    estimatedStorage = estimatedStorage + eqCount * 64
        + #eqIndices * 40 + #eqIndices * variableCount * 8;
    if estimatedStorage > maximumStorageBytes then
        error "estimated polynomial storage exceeds its resource bound";
    scan(#eqIndices, termIndex -> (
        equationIndex := eqIndices#termIndex;
        row := exponentRows#termIndex;
        if not instance(equationIndex, ZZ)
            or equationIndex < 0 or equationIndex >= eqCount then
            error "sparse equation index is out of range";
        if not instance(row, List) or #row =!= variableCount
            or any(row, exponent -> not instance(exponent, ZZ) or exponent < 0) then
            error "invalid sparse exponent row";
        if not instance(coefficientRows#termIndex, String) then
            error "exact coefficients must be strings")));
    ));
R := coefficientRing[Variables => variableCount, MonomialOrder => monomialOrder];

coefficientFromText := text -> (
    if kind === "QQ" then parseRational(R, text)
    else promote(parseInteger text, R));

makePolynomials := record -> (
    eqCount := record#"equation_count";
    eqIndices := record#"equation_indices";
    exponentRows := record#"exponents";
    coefficientRows := record#"coefficients";
    if #eqIndices =!= #exponentRows or #eqIndices =!= #coefficientRows then
        error "inconsistent sparse polynomial arrays";
    result := new MutableList from (eqCount : 0_R);
    scan(#eqIndices, termIndex -> (
        equationIndex := eqIndices#termIndex;
        exponents := exponentRows#termIndex;
        if #exponents =!= variableCount then error "invalid exponent width";
        permuted := apply(permutation, i -> exponents#i);
        monomial := product(variableCount, i -> R_i^(permuted#i));
        result#equationIndex = result#equationIndex
            + coefficientFromText(coefficientRows#termIndex) * monomial));
    toList result);

inputPolynomials := makePolynomials request;
I := ideal inputPolynomials;

univariateResultant := (f, g, x) -> (
    df := degree(x, f);
    dg := degree(x, g);
    if f == 0 or g == 0 then error "resultant inputs must be nonzero";
    if df === 0 and dg === 0 then 1_R
    else (
        size := df + dg;
        entries := table(size, size, (row, column) -> (
            if column < dg then (
                coefficientDegreeF := row - column;
                if coefficientDegreeF < 0 or coefficientDegreeF > df then
                    0_R else coefficient(x^coefficientDegreeF, f))
            else (
                coefficientDegreeG := row - (column - dg);
                if coefficientDegreeG < 0 or coefficientDegreeG > dg then
                    0_R else coefficient(x^coefficientDegreeG, g))));
        det matrix entries));
outputPolynomials := (
    if operation === "groebner_basis" then
        flatten entries gens gb I
    else if operation === "normal_form" then (
        dividendRecord := arguments#"polynomial";
        dividends := makePolynomials dividendRecord;
        if #dividends =!= 1 then error "normal form requires one dividend";
        {first dividends % gb I})
    else if operation === "eliminate" then (
        basis := flatten entries gens gb I;
        select(basis, f -> all(eliminated, i -> degree(R_(position(permutation, j -> j === i)), f) === 0)))
    else if operation === "resultant_univariate" then (
        equations := arguments#"equation_indices";
        if variableCount =!= 1 or #equations =!= 2 then error "invalid univariate resultant";
        {univariateResultant(inputPolynomials#(equations#0), inputPolynomials#(equations#1), R_0)})
    else (
        equation := arguments#"equation_index";
        if variableCount =!= 1 then error "invalid univariate discriminant";
        f := inputPolynomials#equation;
        degreeF := degree(R_0, f);
        signedResultant := (-1)^(binomial(degreeF, 2))
            * univariateResultant(f, diff(R_0, f), R_0);
        {signedResultant / leadCoefficient f}));
if #outputPolynomials === 0 then outputPolynomials = {0_R};

termRows := {};
scan(#outputPolynomials, equationIndex -> (
    polynomial := outputPolynomials#equationIndex;
    if polynomial == 0 then (
        width := if operation === "eliminate" then #kept else variableCount;
        termRows = append(termRows, {equationIndex, toList(width : 0), "0"}))
    else scan(terms polynomial, term -> (
        exponentRow := first exponents term;
        visibleExponents := if operation === "eliminate" then
            apply(#kept, i -> exponentRow#(#eliminated + i))
            else exponentRow;
        scalar := lift(leadCoefficient term, coefficientRing);
        termRows = append(termRows, {equationIndex, visibleExponents, toString scalar})))));
termRows = sort termRows;

polynomials := hashTable {
    "variable_indices" => if operation === "eliminate" then kept else toList(0 .. variableCount - 1),
    "equation_count" => #outputPolynomials,
    "equation_indices" => apply(termRows, row -> row#0),
    "exponents" => apply(termRows, row -> row#1),
    "coefficients" => apply(termRows, row -> row#2)
    };
provenance := hashTable {
    "provider" => "Macaulay2",
    "provider_version" => request#"provider_version",
    "executable_sha256" => request#"executable_sha256",
    "worker_sha256" => request#"worker_sha256",
    "environment_id" => request#"environment_id",
    "external_exact_claim" => true
    };
response := hashTable {
    "request_id" => request#"request_id",
    "plan_id" => request#"plan_id",
    "system_id" => request#"system_id",
    "support_id" => request#"support_id",
    "domain" => domain,
    "operation" => operation,
    "environment_id" => request#"environment_id",
    "provider_id" => request#"provider_id",
    "provider_version" => request#"provider_version",
    "executable_sha256" => request#"executable_sha256",
    "worker_sha256" => request#"worker_sha256",
    "status" => "success",
    "diagnostic" => "",
    "claim" => "exact_claimed_by_external_provider",
    "polynomials" => polynomials,
    "provenance" => provenance
    };
print jsonEncode response;
