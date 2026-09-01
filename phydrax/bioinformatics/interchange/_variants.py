#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded VCF-like host interchange for germline small variants."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import IntEnum

import jax.numpy as jnp

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..genomics import (
    decode_variant_alleles,
    genotype_likelihoods_from_gl,
    genotype_likelihoods_from_pl,
    GenotypeLikelihoods,
    GenotypeStateSpace,
    normalize_small_variant,
    SmallVariantSite,
    VariantNormalizationResult,
)


class VariantInterchangeStatus(IntEnum):
    """Machine-readable VCF-like interchange outcome."""

    OK = 0
    CAPACITY_EXCEEDED = 1
    INVALID_HEADER = 2
    INVALID_RECORD = 3
    INVALID_SAMPLE = 4
    UNSUPPORTED_ALLELE = 5


VCF_PARSE_CONTRACT = BioinformaticsMethodContract(
    "bounded_vcf_text_parse",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Exact parsing of the implemented VCF-like germline small-variant text subset."
    ),
    truncation_statement="Records, samples, alleles, and FORMAT values are never truncated.",
    capacity_semantics=(
        "Text is preflighted against record, sample, and allele capacities; overflow "
        "returns no partial records and CAPACITY_EXCEEDED."
    ),
    assumptions=(
        "VCF 4.x tab-delimited text",
        "Germline non-symbolic small variants",
        "One-based positions at the host boundary",
    ),
    nondifferentiable_outputs=("header", "records", "status"),
)

VCF_WRITE_CONTRACT = BioinformaticsMethodContract(
    "bounded_vcf_text_write",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.STRUCTURED,
    conditioning_statement="Deterministic serialization of validated host VCF-like records.",
    truncation_statement="Records, samples, alleles, and FORMAT values are never truncated.",
    capacity_semantics=(
        "Record capacity is preflighted; overflow returns an empty payload and "
        "CAPACITY_EXCEEDED."
    ),
    assumptions=("Germline non-symbolic small variants",),
    nondifferentiable_outputs=("text", "status"),
)


@dataclass(frozen=True, slots=True)
class VCFSample:
    """Typed host representation of one VCF sample column."""

    genotype: tuple[int | None, ...] | None = None
    phased: bool = False
    genotype_quality: float | None = None
    depth: int | None = None
    allele_depths: tuple[int, ...] | None = None
    genotype_likelihoods: tuple[float, ...] | None = None
    phred_likelihoods: tuple[int, ...] | None = None
    phase_set: int | None = None
    extra_fields: tuple[tuple[str, str], ...] = ()

    @property
    def called(self) -> bool:
        return self.genotype is not None and all(
            allele is not None for allele in self.genotype
        )

    @property
    def no_call(self) -> bool:
        return self.genotype is not None and not self.called


@dataclass(frozen=True, slots=True)
class VCFRecord:
    """VCF-like host record; strings remain outside scientific PyTrees."""

    contig: str
    position: int
    identifier: str | None
    reference: str
    alternates: tuple[str, ...]
    quality: float | None = None
    filters: tuple[str, ...] = ()
    info: tuple[tuple[str, str | bool], ...] = ()
    format_keys: tuple[str, ...] = ()
    samples: tuple[VCFSample, ...] = ()


@dataclass(frozen=True, slots=True)
class VCFHeader:
    """Host VCF header and ordered sample names."""

    meta_lines: tuple[str, ...] = ("##fileformat=VCFv4.3",)
    sample_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VariantInterchangeEvidence:
    record_count: int
    sample_count: int
    error_line: int
    required_capacity: int


@dataclass(frozen=True, slots=True)
class VCFParseResult:
    header: VCFHeader
    records: tuple[VCFRecord, ...]
    valid: bool
    status: VariantInterchangeStatus
    evidence: VariantInterchangeEvidence
    method_contract: BioinformaticsMethodContract = field(
        default_factory=lambda: VCF_PARSE_CONTRACT
    )


@dataclass(frozen=True, slots=True)
class VCFWriteResult:
    text: str
    valid: bool
    status: VariantInterchangeStatus
    evidence: VariantInterchangeEvidence
    method_contract: BioinformaticsMethodContract = field(
        default_factory=lambda: VCF_WRITE_CONTRACT
    )


_INTEGER = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$")
_SYMBOLIC = re.compile(r"[\[\]<>*]")
_STANDARD_FORMAT_KEYS = frozenset(("GT", "GQ", "DP", "AD", "GL", "PL", "PS"))


def _valid_int(value: str, *, nonnegative: bool = False) -> bool:
    return bool(_INTEGER.fullmatch(value)) and (not nonnegative or int(value) >= 0)


def _valid_float(value: str, *, nonnegative: bool = False) -> bool:
    if not _FLOAT.fullmatch(value):
        return False
    parsed = float(value)
    return math.isfinite(parsed) and (not nonnegative or parsed >= 0.0)


def _invalid_parse(
    header: VCFHeader,
    status: VariantInterchangeStatus,
    *,
    record_count: int,
    sample_count: int,
    error_line: int,
    required_capacity: int,
) -> VCFParseResult:
    return VCFParseResult(
        header,
        (),
        False,
        status,
        VariantInterchangeEvidence(
            record_count, sample_count, error_line, required_capacity
        ),
    )


def _parse_gt(
    value: str, allele_count: int
) -> tuple[tuple[int | None, ...], bool] | None:
    if "/" in value and "|" in value:
        return None
    separator = "|" if "|" in value else "/"
    tokens = value.split(separator) if separator in value else [value]
    if not tokens:
        return None
    genotype: list[int | None] = []
    for token in tokens:
        if token == ".":
            genotype.append(None)
        elif _valid_int(token, nonnegative=True) and int(token) < allele_count:
            genotype.append(int(token))
        else:
            return None
    return tuple(genotype), separator == "|"


def _parse_sample(
    format_keys: tuple[str, ...],
    text: str,
    allele_count: int,
) -> VCFSample | None:
    values = text.split(":")
    if len(values) > len(format_keys):
        return None
    values.extend(["."] * (len(format_keys) - len(values)))
    fields = dict(zip(format_keys, values, strict=True))

    genotype: tuple[int | None, ...] | None = None
    phased = False
    if "GT" in fields and fields["GT"] != ".":
        parsed_gt = _parse_gt(fields["GT"], allele_count)
        if parsed_gt is None:
            return None
        genotype, phased = parsed_gt
    elif "GT" in fields:
        genotype = (None,)

    gq_text = fields.get("GQ", ".")
    if gq_text != "." and not _valid_float(gq_text, nonnegative=True):
        return None
    genotype_quality = None if gq_text == "." else float(gq_text)

    depth_text = fields.get("DP", ".")
    if depth_text != "." and not _valid_int(depth_text, nonnegative=True):
        return None
    depth = None if depth_text == "." else int(depth_text)

    allele_depths: tuple[int, ...] | None = None
    ad_text = fields.get("AD", ".")
    if ad_text != ".":
        ad_values = ad_text.split(",")
        if len(ad_values) != allele_count or not all(
            _valid_int(value, nonnegative=True) for value in ad_values
        ):
            return None
        allele_depths = tuple(int(value) for value in ad_values)

    genotype_likelihoods: tuple[float, ...] | None = None
    gl_text = fields.get("GL", ".")
    if gl_text != ".":
        gl_values = gl_text.split(",")
        if not all(_valid_float(value) for value in gl_values):
            return None
        genotype_likelihoods = tuple(float(value) for value in gl_values)

    phred_likelihoods: tuple[int, ...] | None = None
    pl_text = fields.get("PL", ".")
    if pl_text != ".":
        pl_values = pl_text.split(",")
        if not all(_valid_int(value, nonnegative=True) for value in pl_values):
            return None
        phred_likelihoods = tuple(int(value) for value in pl_values)

    if genotype is not None and len(genotype) > 1:
        expected = math.comb(allele_count + len(genotype) - 1, len(genotype))
        if genotype_likelihoods is not None and len(genotype_likelihoods) != expected:
            return None
        if phred_likelihoods is not None and len(phred_likelihoods) != expected:
            return None

    phase_text = fields.get("PS", ".")
    if phase_text != "." and not _valid_int(phase_text, nonnegative=True):
        return None
    phase_set = None if phase_text == "." else int(phase_text)
    extras = tuple(
        (key, value)
        for key, value in zip(format_keys, values, strict=True)
        if key not in _STANDARD_FORMAT_KEYS
    )
    return VCFSample(
        genotype,
        phased,
        genotype_quality,
        depth,
        allele_depths,
        genotype_likelihoods,
        phred_likelihoods,
        phase_set,
        extras,
    )


def parse_vcf(
    text: str,
    /,
    *,
    max_records: int,
    max_samples: int,
    max_alleles: int,
) -> VCFParseResult:
    """Parse a bounded VCF-like germline small-variant text payload exactly."""
    record_capacity = int(max_records)
    sample_capacity = int(max_samples)
    allele_capacity = int(max_alleles)
    if record_capacity < 1 or sample_capacity < 0 or allele_capacity < 2:
        raise ValueError("VCF capacities must be valid and max_alleles at least two.")
    lines = str(text).splitlines()
    meta_lines = tuple(line for line in lines if line.startswith("##"))
    column_indices = [
        index for index, line in enumerate(lines) if line.startswith("#CHROM\t")
    ]
    empty_header = VCFHeader(meta_lines, ())
    if len(column_indices) != 1:
        return _invalid_parse(
            empty_header,
            VariantInterchangeStatus.INVALID_HEADER,
            record_count=0,
            sample_count=0,
            error_line=0,
            required_capacity=0,
        )
    header_index = column_indices[0]
    columns = lines[header_index].split("\t")
    if columns[:8] != [
        "#CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
    ]:
        return _invalid_parse(
            empty_header,
            VariantInterchangeStatus.INVALID_HEADER,
            record_count=0,
            sample_count=0,
            error_line=header_index + 1,
            required_capacity=0,
        )
    if len(columns) > 8 and columns[8] != "FORMAT":
        return _invalid_parse(
            empty_header,
            VariantInterchangeStatus.INVALID_HEADER,
            record_count=0,
            sample_count=0,
            error_line=header_index + 1,
            required_capacity=0,
        )
    sample_names = tuple(columns[9:]) if len(columns) > 8 else ()
    header = VCFHeader(meta_lines, sample_names)
    data_lines = [
        (index, line)
        for index, line in enumerate(lines)
        if index > header_index and line and not line.startswith("#")
    ]
    if len(data_lines) > record_capacity or len(sample_names) > sample_capacity:
        return _invalid_parse(
            header,
            VariantInterchangeStatus.CAPACITY_EXCEEDED,
            record_count=len(data_lines),
            sample_count=len(sample_names),
            error_line=0,
            required_capacity=max(len(data_lines), len(sample_names)),
        )

    parsed_records: list[VCFRecord] = []
    for line_index, line in data_lines:
        fields = line.split("\t")
        expected_columns = 8 if not sample_names else 9 + len(sample_names)
        if len(fields) != expected_columns:
            return _invalid_parse(
                header,
                VariantInterchangeStatus.INVALID_RECORD,
                record_count=len(data_lines),
                sample_count=len(sample_names),
                error_line=line_index + 1,
                required_capacity=0,
            )
        contig, position_text, identifier_text, reference, alt_text = fields[:5]
        if (
            not contig
            or not _valid_int(position_text, nonnegative=True)
            or int(position_text) < 1
            or not reference
            or alt_text in ("", ".")
        ):
            return _invalid_parse(
                header,
                VariantInterchangeStatus.INVALID_RECORD,
                record_count=len(data_lines),
                sample_count=len(sample_names),
                error_line=line_index + 1,
                required_capacity=0,
            )
        alternates = tuple(alt_text.split(","))
        if 1 + len(alternates) > allele_capacity:
            return _invalid_parse(
                header,
                VariantInterchangeStatus.CAPACITY_EXCEEDED,
                record_count=len(data_lines),
                sample_count=len(sample_names),
                error_line=line_index + 1,
                required_capacity=1 + len(alternates),
            )
        if (
            _SYMBOLIC.search(reference)
            or any(_SYMBOLIC.search(allele) for allele in alternates)
            or any(not allele for allele in alternates)
        ):
            return _invalid_parse(
                header,
                VariantInterchangeStatus.UNSUPPORTED_ALLELE,
                record_count=len(data_lines),
                sample_count=len(sample_names),
                error_line=line_index + 1,
                required_capacity=0,
            )
        if len(set(alternates)) != len(alternates) or reference in alternates:
            return _invalid_parse(
                header,
                VariantInterchangeStatus.INVALID_RECORD,
                record_count=len(data_lines),
                sample_count=len(sample_names),
                error_line=line_index + 1,
                required_capacity=0,
            )
        quality_text, filter_text, info_text = fields[5:8]
        if quality_text != "." and not _valid_float(quality_text, nonnegative=True):
            return _invalid_parse(
                header,
                VariantInterchangeStatus.INVALID_RECORD,
                record_count=len(data_lines),
                sample_count=len(sample_names),
                error_line=line_index + 1,
                required_capacity=0,
            )
        quality = None if quality_text == "." else float(quality_text)
        filters = () if filter_text == "." else tuple(filter_text.split(";"))
        info_items: list[tuple[str, str | bool]] = []
        if info_text != ".":
            for item in info_text.split(";"):
                if not item:
                    return _invalid_parse(
                        header,
                        VariantInterchangeStatus.INVALID_RECORD,
                        record_count=len(data_lines),
                        sample_count=len(sample_names),
                        error_line=line_index + 1,
                        required_capacity=0,
                    )
                if "=" in item:
                    key, value = item.split("=", 1)
                    if not key or not value:
                        return _invalid_parse(
                            header,
                            VariantInterchangeStatus.INVALID_RECORD,
                            record_count=len(data_lines),
                            sample_count=len(sample_names),
                            error_line=line_index + 1,
                            required_capacity=0,
                        )
                    info_items.append((key, value))
                else:
                    info_items.append((item, True))

        format_keys: tuple[str, ...] = ()
        samples: tuple[VCFSample, ...] = ()
        if sample_names:
            format_keys = tuple(fields[8].split(":"))
            if not format_keys or len(set(format_keys)) != len(format_keys):
                return _invalid_parse(
                    header,
                    VariantInterchangeStatus.INVALID_SAMPLE,
                    record_count=len(data_lines),
                    sample_count=len(sample_names),
                    error_line=line_index + 1,
                    required_capacity=0,
                )
            parsed_samples = tuple(
                _parse_sample(format_keys, sample_text, 1 + len(alternates))
                for sample_text in fields[9:]
            )
            if any(sample is None for sample in parsed_samples):
                return _invalid_parse(
                    header,
                    VariantInterchangeStatus.INVALID_SAMPLE,
                    record_count=len(data_lines),
                    sample_count=len(sample_names),
                    error_line=line_index + 1,
                    required_capacity=0,
                )
            samples = tuple(sample for sample in parsed_samples if sample is not None)
        parsed_records.append(
            VCFRecord(
                contig,
                int(position_text),
                None if identifier_text == "." else identifier_text,
                reference.upper(),
                tuple(allele.upper() for allele in alternates),
                quality,
                filters,
                tuple(info_items),
                format_keys,
                samples,
            )
        )
    return VCFParseResult(
        header,
        tuple(parsed_records),
        True,
        VariantInterchangeStatus.OK,
        VariantInterchangeEvidence(
            len(parsed_records), len(sample_names), 0, len(parsed_records)
        ),
    )


def _format_float(value: float) -> str:
    return format(float(value), ".12g")


def _format_sample(sample: VCFSample, format_keys: tuple[str, ...]) -> str:
    extras = dict(sample.extra_fields)
    values: list[str] = []
    for key in format_keys:
        if key == "GT":
            if sample.genotype is None:
                value = "."
            else:
                separator = "|" if sample.phased else "/"
                value = separator.join(
                    "." if allele is None else str(allele) for allele in sample.genotype
                )
        elif key == "GQ":
            value = (
                "."
                if sample.genotype_quality is None
                else _format_float(sample.genotype_quality)
            )
        elif key == "DP":
            value = "." if sample.depth is None else str(sample.depth)
        elif key == "AD":
            value = (
                "."
                if sample.allele_depths is None
                else ",".join(map(str, sample.allele_depths))
            )
        elif key == "GL":
            value = (
                "."
                if sample.genotype_likelihoods is None
                else ",".join(_format_float(item) for item in sample.genotype_likelihoods)
            )
        elif key == "PL":
            value = (
                "."
                if sample.phred_likelihoods is None
                else ",".join(map(str, sample.phred_likelihoods))
            )
        elif key == "PS":
            value = "." if sample.phase_set is None else str(sample.phase_set)
        else:
            value = extras.get(key, ".")
        values.append(value)
    return ":".join(values)


def _record_valid(record: VCFRecord, sample_count: int) -> bool:
    if (
        not record.contig
        or record.position < 1
        or not record.reference
        or not record.alternates
        or len(record.samples) != sample_count
        or (sample_count > 0 and not record.format_keys)
        or len(set(record.format_keys)) != len(record.format_keys)
        or len(set(record.alternates)) != len(record.alternates)
        or record.reference in record.alternates
    ):
        return False
    allele_count = 1 + len(record.alternates)
    for sample in record.samples:
        if sample.genotype is not None and any(
            allele is not None and (allele < 0 or allele >= allele_count)
            for allele in sample.genotype
        ):
            return False
        if sample.allele_depths is not None and len(sample.allele_depths) != allele_count:
            return False
        if sample.depth is not None and sample.depth < 0:
            return False
        if sample.genotype_quality is not None and (
            not math.isfinite(sample.genotype_quality) or sample.genotype_quality < 0.0
        ):
            return False
        if sample.genotype is not None and len(sample.genotype) > 1:
            expected = math.comb(
                allele_count + len(sample.genotype) - 1, len(sample.genotype)
            )
            if (
                sample.genotype_likelihoods is not None
                and len(sample.genotype_likelihoods) != expected
            ):
                return False
            if (
                sample.phred_likelihoods is not None
                and len(sample.phred_likelihoods) != expected
            ):
                return False
    return True


def write_vcf(
    header: VCFHeader,
    records: tuple[VCFRecord, ...],
    /,
    *,
    max_records: int,
) -> VCFWriteResult:
    """Serialize validated VCF-like host records without partial bounded output."""
    capacity = int(max_records)
    if capacity < 1:
        raise ValueError("max_records must be positive.")
    sample_count = len(header.sample_names)
    if len(records) > capacity:
        return VCFWriteResult(
            "",
            False,
            VariantInterchangeStatus.CAPACITY_EXCEEDED,
            VariantInterchangeEvidence(len(records), sample_count, 0, len(records)),
        )
    if any(not _record_valid(record, sample_count) for record in records):
        return VCFWriteResult(
            "",
            False,
            VariantInterchangeStatus.INVALID_RECORD,
            VariantInterchangeEvidence(len(records), sample_count, 0, 0),
        )
    lines = list(header.meta_lines)
    columns = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
    if header.sample_names:
        columns.extend(("FORMAT", *header.sample_names))
    lines.append("\t".join(columns))
    for record in records:
        info = (
            ";".join(
                key if value is True else f"{key}={value}" for key, value in record.info
            )
            or "."
        )
        fields = [
            record.contig,
            str(record.position),
            "." if record.identifier is None else record.identifier,
            record.reference,
            ",".join(record.alternates),
            "." if record.quality is None else _format_float(record.quality),
            "." if not record.filters else ";".join(record.filters),
            info,
        ]
        if header.sample_names:
            fields.append(":".join(record.format_keys))
            fields.extend(
                _format_sample(sample, record.format_keys) for sample in record.samples
            )
        lines.append("\t".join(fields))
    return VCFWriteResult(
        "\n".join(lines) + "\n",
        True,
        VariantInterchangeStatus.OK,
        VariantInterchangeEvidence(len(records), sample_count, 0, len(records)),
    )


def vcf_record_to_small_variant(
    record: VCFRecord,
    reference_sequence: str,
    /,
    *,
    reference_index: int,
    contig_index: int,
    max_alleles: int,
    max_allele_length: int,
) -> VariantNormalizationResult:
    """Cross the host/core boundary and exactly normalize a VCF-like record."""
    return normalize_small_variant(
        reference_sequence,
        record.position - 1,
        record.reference,
        record.alternates,
        reference_index=reference_index,
        contig_index=contig_index,
        max_alleles=max_alleles,
        max_allele_length=max_allele_length,
    )


def vcf_record_from_small_variant(
    site: SmallVariantSite,
    contig: str,
    /,
    *,
    identifier: str | None = None,
    quality: float | None = None,
    filters: tuple[str, ...] = (),
    info: tuple[tuple[str, str | bool], ...] = (),
    format_keys: tuple[str, ...] = (),
    samples: tuple[VCFSample, ...] = (),
) -> VCFRecord:
    """Decode one normalized scientific site into a one-based host record."""
    alleles = decode_variant_alleles(site)
    if len(alleles) < 2:
        raise ValueError(
            "A VCF record requires one reference and at least one alternate."
        )
    return VCFRecord(
        str(contig),
        int(site.position) + 1,
        identifier,
        alleles[0],
        alleles[1:],
        quality,
        filters,
        info,
        format_keys,
        samples,
    )


def vcf_sample_likelihoods(
    sample: VCFSample,
    state_space: GenotypeStateSpace,
    /,
) -> GenotypeLikelihoods:
    """Convert typed GL or PL fields to the scientific natural-log scale."""
    depth = 0 if sample.depth is None else sample.depth
    if sample.genotype_likelihoods is not None:
        values = jnp.asarray(sample.genotype_likelihoods, dtype=jnp.float32)
        return genotype_likelihoods_from_gl(values, state_space, depth=depth)
    if sample.phred_likelihoods is not None:
        values = jnp.asarray(sample.phred_likelihoods, dtype=jnp.int32)
        return genotype_likelihoods_from_pl(values, state_space, depth=depth)
    raise ValueError("The VCF sample has neither GL nor PL likelihoods.")


__all__ = [
    "VCFHeader",
    "VCFParseResult",
    "VCFRecord",
    "VCFSample",
    "VCFWriteResult",
    "VCF_PARSE_CONTRACT",
    "VCF_WRITE_CONTRACT",
    "VariantInterchangeEvidence",
    "VariantInterchangeStatus",
    "parse_vcf",
    "vcf_record_from_small_variant",
    "vcf_record_to_small_variant",
    "vcf_sample_likelihoods",
    "write_vcf",
]
