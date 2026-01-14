"""
Design analysis tools for Solar-Flare agents.

These tools support the Design Review Agent in validating designs
against hardware constraints and cross-referencing standards.
"""

from typing import List, Dict, Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ConstraintValidationInput(BaseModel):
    """Input for validating hardware constraints."""

    cpu_overhead_percent: Optional[float] = Field(
        default=None,
        description="Calculated CPU overhead percentage",
    )
    bandwidth_mbps: Optional[float] = Field(
        default=None,
        description="Calculated bandwidth in MB/s",
    )
    mailbox_payload_bytes: Optional[int] = Field(
        default=None,
        description="Mailbox payload size in bytes",
    )
    dma_burst_bytes: Optional[int] = Field(
        default=None,
        description="DMA burst size in bytes",
    )
    timestamp_bits: Optional[int] = Field(
        default=None,
        description="Timestamp bit width",
    )


class CrossReferenceInput(BaseModel):
    """Input for cross-referencing requirements against standards."""

    artifact_types: List[str] = Field(
        description="List of artifact types present in the design"
    )
    requirement_type: str = Field(
        description="Standard to check against: 'iso_26262' or 'aspice'"
    )


class CPUOverheadInput(BaseModel):
    """Input for CPU overhead calculation."""

    isr_cycles: int = Field(description="Cycles spent in ISR per log entry")
    log_rate_hz: int = Field(description="Log entries per second")
    core_frequency_mhz: int = Field(description="Core clock frequency in MHz")
    additional_overhead_cycles: int = Field(
        default=0,
        description="Additional cycles for buffer management, DMA setup, etc.",
    )


class MemoryLayoutInput(BaseModel):
    """Input for memory layout generation."""

    ring_buffer_size_kb: int = Field(description="Ring buffer size per core in KB")
    num_cores: int = Field(description="Number of cores with local buffers")
    log_entry_size_bytes: int = Field(
        default=64,
        description="Size of each log entry in bytes",
    )
    alignment_bytes: int = Field(
        default=64,
        description="Memory alignment requirement in bytes",
    )
    descriptor_queue_entries: int = Field(
        default=16,
        description="Number of DMA descriptor queue entries",
    )


# Define required elements for each standard
ISO_26262_REQUIREMENTS = [
    "safety_goals",
    "functional_safety_requirements",
    "technical_safety_requirements",
    "hardware_software_interface",
    "safety_mechanisms",
    "fmea_analysis",
    "fta_analysis",
    "verification_plan",
    "traceability_matrix",
    "safety_case",
]

ASPICE_REQUIREMENTS = [
    "requirements_specification",
    "architectural_design",
    "detailed_design",
    "unit_tests",
    "integration_tests",
    "qualification_tests",
    "configuration_management",
    "traceability",
    "review_records",
    "change_management",
]


@tool(args_schema=ConstraintValidationInput)
def validate_hardware_constraints(
    cpu_overhead_percent: Optional[float] = None,
    bandwidth_mbps: Optional[float] = None,
    mailbox_payload_bytes: Optional[int] = None,
    dma_burst_bytes: Optional[int] = None,
    timestamp_bits: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Validate design specifications against mandatory hardware constraints.

    Checks compliance with Solar-Flare's non-negotiable constraints:
    - CPU overhead: ≤3% per core
    - Bandwidth: ≤10 MB/s aggregate
    - Mailbox payload: 64 bytes
    - DMA burst: 64 KB
    - Timestamp: 64-bit

    Args:
        cpu_overhead_percent: Calculated CPU overhead percentage
        bandwidth_mbps: Calculated bandwidth in MB/s
        mailbox_payload_bytes: Mailbox payload size in bytes
        dma_burst_bytes: DMA burst size in bytes
        timestamp_bits: Timestamp bit width

    Returns:
        Validation report with pass/fail status for each constraint

    Example:
        >>> result = validate_hardware_constraints(cpu_overhead_percent=2.5, bandwidth_mbps=8.0)
        >>> print(f"Valid: {result['valid']}")
    """
    # Define constraint limits
    MAX_CPU_OVERHEAD = 3.0
    MAX_BANDWIDTH = 10.0
    EXPECTED_MAILBOX = 64
    EXPECTED_DMA_BURST = 65536  # 64 KB
    EXPECTED_TIMESTAMP = 64

    results = {
        "valid": True,
        "violations": [],
        "warnings": [],
        "checks": {},
    }

    # Check CPU overhead
    if cpu_overhead_percent is not None:
        check_passed = cpu_overhead_percent <= MAX_CPU_OVERHEAD
        results["checks"]["cpu_overhead"] = {
            "value": cpu_overhead_percent,
            "limit": MAX_CPU_OVERHEAD,
            "unit": "%",
            "passed": check_passed,
        }
        if not check_passed:
            results["valid"] = False
            results["violations"].append(
                f"CPU overhead {cpu_overhead_percent}% exceeds limit of {MAX_CPU_OVERHEAD}%"
            )

    # Check bandwidth
    if bandwidth_mbps is not None:
        check_passed = bandwidth_mbps <= MAX_BANDWIDTH
        results["checks"]["bandwidth"] = {
            "value": bandwidth_mbps,
            "limit": MAX_BANDWIDTH,
            "unit": "MB/s",
            "passed": check_passed,
        }
        if not check_passed:
            results["valid"] = False
            results["violations"].append(
                f"Bandwidth {bandwidth_mbps} MB/s exceeds limit of {MAX_BANDWIDTH} MB/s"
            )

    # Check mailbox payload
    if mailbox_payload_bytes is not None:
        if mailbox_payload_bytes != EXPECTED_MAILBOX:
            results["warnings"].append(
                f"Mailbox payload {mailbox_payload_bytes} bytes differs from "
                f"expected {EXPECTED_MAILBOX} bytes"
            )
        results["checks"]["mailbox_payload"] = {
            "value": mailbox_payload_bytes,
            "expected": EXPECTED_MAILBOX,
            "unit": "bytes",
            "passed": mailbox_payload_bytes == EXPECTED_MAILBOX,
        }

    # Check DMA burst size
    if dma_burst_bytes is not None:
        if dma_burst_bytes != EXPECTED_DMA_BURST:
            results["warnings"].append(
                f"DMA burst {dma_burst_bytes} bytes differs from "
                f"expected {EXPECTED_DMA_BURST} bytes (64 KB)"
            )
        results["checks"]["dma_burst"] = {
            "value": dma_burst_bytes,
            "expected": EXPECTED_DMA_BURST,
            "unit": "bytes",
            "passed": dma_burst_bytes == EXPECTED_DMA_BURST,
        }

    # Check timestamp bits
    if timestamp_bits is not None:
        if timestamp_bits != EXPECTED_TIMESTAMP:
            results["warnings"].append(
                f"Timestamp {timestamp_bits}-bit differs from "
                f"expected {EXPECTED_TIMESTAMP}-bit"
            )
        results["checks"]["timestamp_bits"] = {
            "value": timestamp_bits,
            "expected": EXPECTED_TIMESTAMP,
            "unit": "bits",
            "passed": timestamp_bits == EXPECTED_TIMESTAMP,
        }

    return results


@tool(args_schema=CrossReferenceInput)
def cross_reference_requirements(
    artifact_types: List[str],
    requirement_type: str,
) -> Dict[str, Any]:
    """
    Cross-reference design artifacts against ISO 26262 or ASPICE requirements.

    Identifies:
    - Missing required elements
    - Coverage percentage
    - Gaps with severity ratings

    Args:
        artifact_types: List of artifact types present (e.g., ["architectural_design", "unit_tests"])
        requirement_type: Standard to check: "iso_26262" or "aspice"

    Returns:
        Gap analysis report with missing coverage

    Example:
        >>> result = cross_reference_requirements(
        ...     artifact_types=["requirements_specification", "architectural_design"],
        ...     requirement_type="aspice"
        ... )
        >>> print(f"Coverage: {result['coverage_percent']}%")
    """
    if requirement_type == "iso_26262":
        requirements = ISO_26262_REQUIREMENTS
        high_priority = {"safety_goals", "safety_mechanisms", "traceability_matrix"}
    elif requirement_type == "aspice":
        requirements = ASPICE_REQUIREMENTS
        high_priority = {"requirements_specification", "traceability", "configuration_management"}
    else:
        return {
            "error": f"Unknown requirement type: {requirement_type}. Use 'iso_26262' or 'aspice'."
        }

    # Normalize artifact types for comparison
    artifact_set = set(a.lower().replace(" ", "_").replace("-", "_") for a in artifact_types)

    covered = []
    missing = []

    for req in requirements:
        if req in artifact_set:
            covered.append(req)
        else:
            missing.append(req)

    # Generate gaps with severity
    gaps = []
    for m in missing:
        severity = "high" if m in high_priority else "medium"
        gaps.append(
            {
                "requirement": m,
                "severity": severity,
                "description": f"Missing {requirement_type.upper()} required element: {m.replace('_', ' ').title()}",
            }
        )

    # Sort gaps by severity
    gaps.sort(key=lambda x: (0 if x["severity"] == "high" else 1, x["requirement"]))

    coverage_percent = (len(covered) / len(requirements)) * 100 if requirements else 0

    return {
        "requirement_type": requirement_type,
        "total_requirements": len(requirements),
        "covered_count": len(covered),
        "missing_count": len(missing),
        "coverage_percent": round(coverage_percent, 1),
        "covered": covered,
        "missing": missing,
        "gaps": gaps,
    }


@tool(args_schema=CPUOverheadInput)
def calculate_cpu_overhead(
    isr_cycles: int,
    log_rate_hz: int,
    core_frequency_mhz: int,
    additional_overhead_cycles: int = 0,
) -> Dict[str, Any]:
    """
    Calculate CPU overhead percentage for logging operations.

    Computes the percentage of CPU time consumed by logging based on
    ISR execution cycles, log rate, and core frequency.

    Args:
        isr_cycles: Cycles spent in ISR per log entry
        log_rate_hz: Log entries per second
        core_frequency_mhz: Core clock frequency in MHz
        additional_overhead_cycles: Extra cycles for buffer mgmt, DMA setup, etc.

    Returns:
        CPU overhead calculation with compliance status

    Example:
        >>> result = calculate_cpu_overhead(
        ...     isr_cycles=500,
        ...     log_rate_hz=10000,
        ...     core_frequency_mhz=400
        ... )
        >>> print(f"Overhead: {result['cpu_overhead_percent']}%")
    """
    MAX_CPU_OVERHEAD = 3.0

    # Convert MHz to Hz
    core_frequency_hz = core_frequency_mhz * 1_000_000

    # Total cycles per second for logging
    total_cycles = isr_cycles + additional_overhead_cycles
    cycles_per_second = total_cycles * log_rate_hz

    # Calculate percentage
    cpu_overhead_percent = (cycles_per_second / core_frequency_hz) * 100

    # Check compliance
    compliant = cpu_overhead_percent <= MAX_CPU_OVERHEAD

    return {
        "isr_cycles": isr_cycles,
        "additional_overhead_cycles": additional_overhead_cycles,
        "total_cycles_per_log": total_cycles,
        "log_rate_hz": log_rate_hz,
        "core_frequency_mhz": core_frequency_mhz,
        "cycles_per_second": cycles_per_second,
        "cpu_overhead_percent": round(cpu_overhead_percent, 4),
        "max_allowed_percent": MAX_CPU_OVERHEAD,
        "compliant": compliant,
        "margin_percent": round(MAX_CPU_OVERHEAD - cpu_overhead_percent, 4),
    }


@tool(args_schema=MemoryLayoutInput)
def generate_memory_layout(
    ring_buffer_size_kb: int,
    num_cores: int,
    log_entry_size_bytes: int = 64,
    alignment_bytes: int = 64,
    descriptor_queue_entries: int = 16,
) -> Dict[str, Any]:
    """
    Generate memory layout specification for multi-core logging system.

    Creates a memory map showing ring buffer placement, descriptor queues,
    and total memory requirements for the logging subsystem.

    Args:
        ring_buffer_size_kb: Ring buffer size per core in KB
        num_cores: Number of cores with local ring buffers
        log_entry_size_bytes: Size of each log entry (default: 64 bytes)
        alignment_bytes: Memory alignment requirement (default: 64 bytes)
        descriptor_queue_entries: DMA descriptor queue entries (default: 16)

    Returns:
        Memory layout specification with addresses and sizes

    Example:
        >>> layout = generate_memory_layout(ring_buffer_size_kb=64, num_cores=4)
        >>> print(f"Total memory: {layout['total_memory_kb']} KB")
    """
    # Constants
    DESCRIPTOR_SIZE = 32  # Bytes per DMA descriptor

    # Calculate sizes
    ring_buffer_size = ring_buffer_size_kb * 1024
    entries_per_buffer = ring_buffer_size // log_entry_size_bytes
    descriptor_queue_size = descriptor_queue_entries * DESCRIPTOR_SIZE

    # Align sizes
    def align_up(size: int, alignment: int) -> int:
        return ((size + alignment - 1) // alignment) * alignment

    aligned_ring_buffer = align_up(ring_buffer_size, alignment_bytes)
    aligned_descriptor_queue = align_up(descriptor_queue_size, alignment_bytes)

    # Generate per-core layout
    cores = []
    current_offset = 0

    for core_id in range(num_cores):
        core_layout = {
            "core_id": core_id,
            "ring_buffer": {
                "offset": current_offset,
                "size": aligned_ring_buffer,
                "entries": entries_per_buffer,
                "entry_size": log_entry_size_bytes,
            },
        }
        current_offset += aligned_ring_buffer

        core_layout["descriptor_queue"] = {
            "offset": current_offset,
            "size": aligned_descriptor_queue,
            "entries": descriptor_queue_entries,
            "entry_size": DESCRIPTOR_SIZE,
        }
        current_offset += aligned_descriptor_queue

        cores.append(core_layout)

    total_memory = current_offset
    total_memory_kb = total_memory // 1024

    return {
        "configuration": {
            "num_cores": num_cores,
            "ring_buffer_size_kb": ring_buffer_size_kb,
            "log_entry_size_bytes": log_entry_size_bytes,
            "alignment_bytes": alignment_bytes,
            "descriptor_queue_entries": descriptor_queue_entries,
        },
        "per_core_memory": {
            "ring_buffer_bytes": aligned_ring_buffer,
            "descriptor_queue_bytes": aligned_descriptor_queue,
            "total_bytes": aligned_ring_buffer + aligned_descriptor_queue,
        },
        "cores": cores,
        "total_memory_bytes": total_memory,
        "total_memory_kb": total_memory_kb,
        "total_memory_mb": round(total_memory_kb / 1024, 2),
    }
