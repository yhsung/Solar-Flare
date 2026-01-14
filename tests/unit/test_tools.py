"""
Unit tests for Solar-Flare tools.
"""

import pytest
from solar_flare.tools.analysis_tools import (
    validate_hardware_constraints,
    cross_reference_requirements,
    calculate_cpu_overhead,
    generate_memory_layout,
)


class TestValidateHardwareConstraints:
    """Tests for the validate_hardware_constraints tool."""

    def test_valid_constraints(self):
        """Test that valid constraints pass validation."""
        result = validate_hardware_constraints.invoke(
            {
                "cpu_overhead_percent": 2.5,
                "bandwidth_mbps": 8.0,
            }
        )

        assert result["valid"] is True
        assert len(result["violations"]) == 0
        assert result["checks"]["cpu_overhead"]["passed"] is True
        assert result["checks"]["bandwidth"]["passed"] is True

    def test_cpu_overhead_violation(self):
        """Test that CPU overhead exceeding 3% is flagged."""
        result = validate_hardware_constraints.invoke(
            {
                "cpu_overhead_percent": 4.5,
                "bandwidth_mbps": 5.0,
            }
        )

        assert result["valid"] is False
        assert len(result["violations"]) == 1
        assert "CPU overhead" in result["violations"][0]
        assert result["checks"]["cpu_overhead"]["passed"] is False

    def test_bandwidth_violation(self):
        """Test that bandwidth exceeding 10 MB/s is flagged."""
        result = validate_hardware_constraints.invoke(
            {
                "cpu_overhead_percent": 2.0,
                "bandwidth_mbps": 12.0,
            }
        )

        assert result["valid"] is False
        assert len(result["violations"]) == 1
        assert "Bandwidth" in result["violations"][0]

    def test_multiple_violations(self):
        """Test that multiple violations are all reported."""
        result = validate_hardware_constraints.invoke(
            {
                "cpu_overhead_percent": 5.0,
                "bandwidth_mbps": 15.0,
            }
        )

        assert result["valid"] is False
        assert len(result["violations"]) == 2

    def test_mailbox_warning(self):
        """Test that non-standard mailbox size generates warning."""
        result = validate_hardware_constraints.invoke(
            {
                "mailbox_payload_bytes": 128,
            }
        )

        assert result["valid"] is True  # Warnings don't fail validation
        assert len(result["warnings"]) == 1
        assert "Mailbox" in result["warnings"][0]

    def test_boundary_values(self):
        """Test exact boundary values (3% and 10 MB/s)."""
        result = validate_hardware_constraints.invoke(
            {
                "cpu_overhead_percent": 3.0,
                "bandwidth_mbps": 10.0,
            }
        )

        assert result["valid"] is True
        assert result["checks"]["cpu_overhead"]["passed"] is True
        assert result["checks"]["bandwidth"]["passed"] is True


class TestCrossReferenceRequirements:
    """Tests for the cross_reference_requirements tool."""

    def test_iso_26262_full_coverage(self):
        """Test ISO 26262 requirements with full coverage."""
        result = cross_reference_requirements.invoke(
            {
                "artifact_types": [
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
                ],
                "requirement_type": "iso_26262",
            }
        )

        assert result["coverage_percent"] == 100.0
        assert result["missing_count"] == 0
        assert len(result["gaps"]) == 0

    def test_iso_26262_partial_coverage(self):
        """Test ISO 26262 requirements with gaps."""
        result = cross_reference_requirements.invoke(
            {
                "artifact_types": [
                    "safety_goals",
                    "technical_safety_requirements",
                ],
                "requirement_type": "iso_26262",
            }
        )

        assert result["coverage_percent"] < 100
        assert result["missing_count"] > 0
        assert len(result["gaps"]) > 0

    def test_aspice_coverage(self):
        """Test ASPICE requirements coverage."""
        result = cross_reference_requirements.invoke(
            {
                "artifact_types": [
                    "requirements_specification",
                    "architectural_design",
                    "detailed_design",
                ],
                "requirement_type": "aspice",
            }
        )

        assert result["requirement_type"] == "aspice"
        assert result["covered_count"] == 3
        assert result["total_requirements"] == 10  # Total ASPICE requirements

    def test_gap_severity(self):
        """Test that high-priority gaps are marked correctly."""
        result = cross_reference_requirements.invoke(
            {
                "artifact_types": ["detailed_design"],
                "requirement_type": "iso_26262",
            }
        )

        # safety_goals is high priority and should be missing
        high_severity_gaps = [g for g in result["gaps"] if g["severity"] == "high"]
        assert len(high_severity_gaps) > 0

    def test_invalid_requirement_type(self):
        """Test handling of invalid requirement type."""
        result = cross_reference_requirements.invoke(
            {
                "artifact_types": ["some_artifact"],
                "requirement_type": "invalid_type",
            }
        )

        assert "error" in result


class TestCalculateCPUOverhead:
    """Tests for the calculate_cpu_overhead tool."""

    def test_compliant_overhead(self):
        """Test calculation that meets constraints."""
        result = calculate_cpu_overhead.invoke(
            {
                "isr_cycles": 200,
                "log_rate_hz": 10000,
                "core_frequency_mhz": 400,
            }
        )

        assert result["compliant"] is True
        assert result["cpu_overhead_percent"] < 3.0
        assert result["margin_percent"] > 0

    def test_non_compliant_overhead(self):
        """Test calculation that exceeds constraints."""
        result = calculate_cpu_overhead.invoke(
            {
                "isr_cycles": 5000,
                "log_rate_hz": 50000,
                "core_frequency_mhz": 100,
            }
        )

        assert result["compliant"] is False
        assert result["cpu_overhead_percent"] > 3.0
        assert result["margin_percent"] < 0

    def test_with_additional_overhead(self):
        """Test calculation with additional overhead cycles."""
        result = calculate_cpu_overhead.invoke(
            {
                "isr_cycles": 200,
                "log_rate_hz": 10000,
                "core_frequency_mhz": 400,
                "additional_overhead_cycles": 100,
            }
        )

        assert result["total_cycles_per_log"] == 300
        assert result["additional_overhead_cycles"] == 100


class TestGenerateMemoryLayout:
    """Tests for the generate_memory_layout tool."""

    def test_single_core_layout(self):
        """Test memory layout for single core."""
        result = generate_memory_layout.invoke(
            {
                "ring_buffer_size_kb": 64,
                "num_cores": 1,
            }
        )

        assert result["configuration"]["num_cores"] == 1
        assert len(result["cores"]) == 1
        assert result["total_memory_kb"] > 64  # Buffer + descriptors

    def test_multi_core_layout(self):
        """Test memory layout for multiple cores."""
        result = generate_memory_layout.invoke(
            {
                "ring_buffer_size_kb": 64,
                "num_cores": 4,
            }
        )

        assert len(result["cores"]) == 4
        # Each core should have non-overlapping offsets
        offsets = [core["ring_buffer"]["offset"] for core in result["cores"]]
        assert len(set(offsets)) == 4  # All unique

    def test_alignment(self):
        """Test that memory is properly aligned."""
        result = generate_memory_layout.invoke(
            {
                "ring_buffer_size_kb": 64,
                "num_cores": 2,
                "alignment_bytes": 64,
            }
        )

        for core in result["cores"]:
            # All offsets should be 64-byte aligned
            assert core["ring_buffer"]["offset"] % 64 == 0
            assert core["descriptor_queue"]["offset"] % 64 == 0

    def test_total_memory_calculation(self):
        """Test total memory is correctly calculated."""
        result = generate_memory_layout.invoke(
            {
                "ring_buffer_size_kb": 64,
                "num_cores": 4,
                "log_entry_size_bytes": 64,
            }
        )

        # Total should be num_cores * (buffer + descriptors)
        per_core = result["per_core_memory"]["total_bytes"]
        expected_total = per_core * 4
        assert result["total_memory_bytes"] == expected_total
