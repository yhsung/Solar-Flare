"""
Unit tests for the markdown export module.

Tests the formatting and export functionality for agent results.
"""

import json
import tempfile
from pathlib import Path

import pytest

from solar_flare.graph.state import WorkerResult, DesignReviewResult
from solar_flare.markdown_export import (
    format_worker_result_markdown,
    format_design_review_markdown,
    format_workflow_summary,
    export_workflow_results,
    _format_agent_name,
)


class TestFormatAgentName:
    """Tests for agent name formatting."""

    def test_formats_underscores_as_spaces(self):
        assert _format_agent_name("iso_26262_analyzer") == "Iso 26262 Analyzer"

    def test_title_case(self):
        assert _format_agent_name("embedded_designer") == "Embedded Designer"


class TestFormatWorkerResultMarkdown:
    """Tests for WorkerResult markdown formatting."""

    def test_basic_result(self):
        result = WorkerResult(
            agent_name="test_agent",
            task_type="analysis",
            status="success",
            confidence_score=0.85,
        )
        markdown = format_worker_result_markdown(result)

        assert "# Test Agent Results" in markdown
        assert "**Task Type:** analysis" in markdown
        assert "**Status:** success ✓" in markdown
        assert "**Confidence:** 85%" in markdown

    def test_result_with_findings(self):
        result = WorkerResult(
            agent_name="test_agent",
            task_type="analysis",
            status="success",
            findings=[
                {
                    "type": "compliance_gap",
                    "severity": "high",
                    "description": "Missing safety documentation",
                    "recommendation": "Add safety analysis",
                }
            ],
            confidence_score=0.75,
        )
        markdown = format_worker_result_markdown(result)

        assert "## Findings" in markdown
        assert "### Finding 1: [HIGH] compliance_gap" in markdown
        assert "Missing safety documentation" in markdown
        assert "**Recommendation:** Add safety analysis" in markdown

    def test_result_with_recommendations(self):
        result = WorkerResult(
            agent_name="test_agent",
            task_type="design",
            status="success",
            recommendations=["First rec", "Second rec"],
            confidence_score=0.9,
        )
        markdown = format_worker_result_markdown(result)

        assert "## Recommendations" in markdown
        assert "1. First rec" in markdown
        assert "2. Second rec" in markdown

    def test_result_with_artifacts(self):
        result = WorkerResult(
            agent_name="test_agent",
            task_type="design",
            status="success",
            artifacts={
                "design_doc": "Some design content",
                "config": {"key": "value"},
            },
            confidence_score=0.8,
        )
        markdown = format_worker_result_markdown(result)

        assert "## Artifacts" in markdown
        assert "### design_doc" in markdown
        assert "### config" in markdown
        assert '"key": "value"' in markdown

    def test_failed_status_emoji(self):
        result = WorkerResult(
            agent_name="test_agent",
            task_type="analysis",
            status="failed",
            confidence_score=0.0,
        )
        markdown = format_worker_result_markdown(result)
        assert "**Status:** failed ✗" in markdown

    def test_partial_status_emoji(self):
        result = WorkerResult(
            agent_name="test_agent",
            task_type="analysis",
            status="partial",
            confidence_score=0.5,
        )
        markdown = format_worker_result_markdown(result)
        assert "**Status:** partial ⚠" in markdown


class TestFormatDesignReviewMarkdown:
    """Tests for DesignReviewResult markdown formatting."""

    def test_basic_review(self):
        review = DesignReviewResult(
            overall_status="approved",
            completeness_score=85.0,
        )
        markdown = format_design_review_markdown(review)

        assert "# Design Review Results" in markdown
        assert "**Overall Status:** Approved ✓" in markdown
        assert "**Completeness Score:** 85%" in markdown
        assert "Completeness:" in markdown  # Progress bar

    def test_review_with_gaps(self):
        review = DesignReviewResult(
            overall_status="needs_revision",
            completeness_score=60.0,
            gaps_identified=[
                {"severity": "high", "description": "Missing error handling"},
            ],
        )
        markdown = format_design_review_markdown(review)

        assert "## Gaps Identified" in markdown
        assert "### Gap 1: [HIGH]" in markdown
        assert "Missing error handling" in markdown

    def test_review_with_violations(self):
        review = DesignReviewResult(
            overall_status="rejected",
            completeness_score=30.0,
            constraint_violations=[
                {
                    "constraint": "CPU overhead",
                    "expected": "≤3%",
                    "actual": "5%",
                    "severity": "critical",
                }
            ],
        )
        markdown = format_design_review_markdown(review)

        assert "## Constraint Violations" in markdown
        assert "| CPU overhead | ≤3% | 5% | critical |" in markdown


class TestWorkerResultToMarkdown:
    """Tests for WorkerResult.to_markdown() method."""

    def test_to_markdown_method(self):
        result = WorkerResult(
            agent_name="test_agent",
            task_type="analysis",
            status="success",
            confidence_score=0.9,
        )
        markdown = result.to_markdown()

        assert "# Test Agent Results" in markdown
        assert "**Status:** success ✓" in markdown


class TestDesignReviewToMarkdown:
    """Tests for DesignReviewResult.to_markdown() method."""

    def test_to_markdown_method(self):
        review = DesignReviewResult(
            overall_status="approved",
            completeness_score=95.0,
        )
        markdown = review.to_markdown()

        assert "# Design Review Results" in markdown
        assert "**Overall Status:** Approved ✓" in markdown


class TestExportWorkflowResults:
    """Tests for export_workflow_results function."""

    def test_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_dir"
            
            state = {
                "current_phase": "complete",
                "iteration_count": 3,
                "worker_results": [],
                "hardware_constraints": None,
                "design_review": None,
                "final_response": None,
                "errors": [],
            }
            
            files = export_workflow_results(state, output_dir)
            
            assert output_dir.exists()
            assert len(files) >= 1  # At least summary

    def test_creates_summary_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "current_phase": "complete",
                "iteration_count": 2,
                "worker_results": [],
                "hardware_constraints": None,
                "design_review": None,
                "final_response": None,
                "errors": [],
            }
            
            files = export_workflow_results(state, tmpdir, include_summary=True)
            
            summary_file = Path(tmpdir) / "00_summary.md"
            assert summary_file.exists()
            assert summary_file in files

    def test_creates_worker_result_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = WorkerResult(
                agent_name="iso_26262_analyzer",
                task_type="analysis",
                status="success",
                confidence_score=0.9,
            )
            result2 = WorkerResult(
                agent_name="embedded_designer",
                task_type="design",
                status="success",
                confidence_score=0.85,
            )
            
            state = {
                "current_phase": "complete",
                "iteration_count": 2,
                "worker_results": [result1, result2],
                "hardware_constraints": None,
                "design_review": None,
                "final_response": None,
                "errors": [],
            }
            
            files = export_workflow_results(state, tmpdir)
            
            assert (Path(tmpdir) / "01_iso_26262_analyzer.md").exists()
            assert (Path(tmpdir) / "02_embedded_designer.md").exists()

    def test_creates_final_response_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "current_phase": "complete",
                "iteration_count": 1,
                "worker_results": [],
                "hardware_constraints": None,
                "design_review": None,
                "final_response": "This is the final response.",
                "errors": [],
            }
            
            files = export_workflow_results(state, tmpdir)
            
            # Check that final response file exists
            response_files = [f for f in files if "final_response" in str(f)]
            assert len(response_files) == 1
            
            content = response_files[0].read_text()
            assert "This is the final response." in content

    def test_skip_summary_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "current_phase": "complete",
                "iteration_count": 1,
                "worker_results": [],
                "hardware_constraints": None,
                "design_review": None,
                "final_response": None,
                "errors": [],
            }
            
            files = export_workflow_results(state, tmpdir, include_summary=False)
            
            summary_file = Path(tmpdir) / "00_summary.md"
            assert not summary_file.exists()


class TestFormatWorkflowSummary:
    """Tests for format_workflow_summary function."""

    def test_includes_phase_and_iterations(self):
        state = {
            "current_phase": "complete",
            "iteration_count": 5,
            "worker_results": [],
            "hardware_constraints": None,
            "design_review": None,
            "errors": [],
        }
        
        markdown = format_workflow_summary(state)
        
        assert "**Phase:** complete" in markdown
        assert "**Iterations:** 5" in markdown

    def test_includes_agents_table(self):
        result = WorkerResult(
            agent_name="test_agent",
            task_type="analysis",
            status="success",
            confidence_score=0.9,
        )
        
        state = {
            "current_phase": "complete",
            "iteration_count": 1,
            "worker_results": [result],
            "hardware_constraints": None,
            "design_review": None,
            "errors": [],
        }
        
        markdown = format_workflow_summary(state)
        
        assert "## Agents Executed" in markdown
        assert "| Test Agent |" in markdown
