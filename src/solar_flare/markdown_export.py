"""
Markdown export functionality for Solar-Flare workflow results.

This module provides utilities for exporting agent results and workflow
state to well-formatted markdown documents.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from solar_flare.graph.state import (
    AgentState,
    WorkerResult,
    DesignReviewResult,
    HardwareConstraints,
)


def format_worker_result_markdown(result: WorkerResult, index: int = 0) -> str:
    """
    Format a single WorkerResult as a markdown document.

    Args:
        result: The worker result to format
        index: Optional index for ordering (used in filenames)

    Returns:
        Formatted markdown string
    """
    # Map status to emoji
    status_emoji = {
        "success": "✓",
        "partial": "⚠",
        "failed": "✗",
    }
    emoji = status_emoji.get(result.status, "•")

    lines = [
        f"# {_format_agent_name(result.agent_name)} Results",
        "",
        f"**Task Type:** {result.task_type}  ",
        f"**Status:** {result.status} {emoji}  ",
        f"**Confidence:** {result.confidence_score:.0%}",
        "",
    ]

    # Findings section
    if result.findings:
        lines.append("## Findings")
        lines.append("")
        for i, finding in enumerate(result.findings, 1):
            severity = finding.get("severity", "medium").upper()
            finding_type = finding.get("type", "observation")
            description = finding.get("description", "")
            
            lines.append(f"### Finding {i}: [{severity}] {finding_type}")
            lines.append("")
            lines.append(description)
            lines.append("")
            
            if "recommendation" in finding:
                lines.append(f"**Recommendation:** {finding['recommendation']}")
                lines.append("")

    # Recommendations section
    if result.recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(result.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    # Artifacts section
    if result.artifacts:
        lines.append("## Artifacts")
        lines.append("")
        for name, content in result.artifacts.items():
            lines.append(f"### {name}")
            lines.append("")
            if isinstance(content, str):
                # If it's a long string, wrap in code block
                if len(content) > 200 or "\n" in content:
                    lines.append("```")
                    lines.append(content)
                    lines.append("```")
                else:
                    lines.append(content)
            elif isinstance(content, dict):
                lines.append("```json")
                import json
                lines.append(json.dumps(content, indent=2))
                lines.append("```")
            else:
                lines.append(f"`{content}`")
            lines.append("")

    return "\n".join(lines)


def format_design_review_markdown(review: DesignReviewResult) -> str:
    """
    Format a DesignReviewResult as a markdown document.

    Args:
        review: The design review result to format (dict or DesignReviewResult)

    Returns:
        Formatted markdown string
    """
    # Handle dict input (from serialization)
    if isinstance(review, dict):
        overall_status = review.get('overall_status', 'unknown')
        completeness_score = review.get('completeness_score', 0)
        gaps_identified = review.get('gaps_identified', [])
        cross_reference_issues = review.get('cross_reference_issues', [])
        constraint_violations = review.get('constraint_violations', [])
        recommendations = review.get('recommendations', [])
    else:
        overall_status = review.overall_status
        completeness_score = review.completeness_score
        gaps_identified = review.gaps_identified
        cross_reference_issues = review.cross_reference_issues
        constraint_violations = review.constraint_violations
        recommendations = review.recommendations

    # Map status to emoji
    status_emoji = {
        "approved": "✓",
        "needs_revision": "⚠",
        "rejected": "✗",
    }
    emoji = status_emoji.get(overall_status, "•")

    lines = [
        "# Design Review Results",
        "",
        f"**Overall Status:** {overall_status.replace('_', ' ').title()} {emoji}  ",
        f"**Completeness Score:** {completeness_score:.0f}%",
        "",
    ]

    # Progress bar visualization
    filled = int(completeness_score / 10)
    empty = 10 - filled
    progress = "█" * filled + "░" * empty
    lines.append(f"```")
    lines.append(f"Completeness: [{progress}] {completeness_score:.0f}%")
    lines.append(f"```")
    lines.append("")


    # Gaps identified
    if gaps_identified:
        lines.append("## Gaps Identified")
        lines.append("")
        for i, gap in enumerate(gaps_identified, 1):
            severity = gap.get("severity", "medium").upper()
            description = gap.get("description", "")
            lines.append(f"### Gap {i}: [{severity}]")
            lines.append("")
            lines.append(description)
            lines.append("")

    # Cross-reference issues
    if cross_reference_issues:
        lines.append("## Cross-Reference Issues")
        lines.append("")
        for issue in cross_reference_issues:
            standard = issue.get("standard", "Unknown")
            description = issue.get("description", "")
            lines.append(f"- **{standard}**: {description}")
        lines.append("")

    # Constraint violations
    if constraint_violations:
        lines.append("## Constraint Violations")
        lines.append("")
        lines.append("| Constraint | Expected | Actual | Severity |")
        lines.append("|------------|----------|--------|----------|")
        for violation in constraint_violations:
            constraint = violation.get("constraint", "Unknown")
            expected = violation.get("expected", "N/A")
            actual = violation.get("actual", "N/A")
            severity = violation.get("severity", "medium")
            lines.append(f"| {constraint} | {expected} | {actual} | {severity} |")
        lines.append("")

    # Recommendations
    if recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    return "\n".join(lines)


def format_workflow_summary(state: AgentState) -> str:
    """
    Generate a summary markdown of the entire workflow execution.

    Args:
        state: The complete workflow state

    Returns:
        Formatted markdown summary
    """
    lines = [
        "# Solar-Flare Workflow Summary",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Phase:** {state.get('current_phase', 'unknown')}  ",
        f"**Iterations:** {state.get('iteration_count', 0)}",
        "",
    ]

    # Hardware constraints
    constraints = state.get("hardware_constraints")
    if constraints:
        lines.append("## Hardware Constraints")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Mailbox Payload | {constraints.mailbox_payload_bytes} bytes |")
        lines.append(f"| DMA Burst | {constraints.dma_burst_bytes // 1024} KB |")
        lines.append(f"| Timer Resolution | {constraints.timestamp_resolution_ns} ns |")
        lines.append(f"| Timestamp Bits | {constraints.timestamp_bits} |")
        lines.append(f"| Max CPU Overhead | {constraints.max_cpu_overhead_percent}% |")
        lines.append(f"| Max Bandwidth | {constraints.max_bandwidth_mbps} MB/s |")
        lines.append("")

    # Agents executed
    worker_results = state.get("worker_results", [])
    if worker_results:
        lines.append("## Agents Executed")
        lines.append("")
        lines.append("| Agent | Task Type | Status | Confidence |")
        lines.append("|-------|-----------|--------|------------|")
        for result in worker_results:
            status_emoji = {"success": "✓", "partial": "⚠", "failed": "✗"}
            emoji = status_emoji.get(result.status, "•")
            lines.append(
                f"| {_format_agent_name(result.agent_name)} | "
                f"{result.task_type} | {result.status} {emoji} | "
                f"{result.confidence_score:.0%} |"
            )
        lines.append("")

    # Design review summary
    review = state.get("design_review")
    if review:
        # Handle dict or object
        if isinstance(review, dict):
            overall_status = review.get('overall_status', 'unknown')
            completeness_score = review.get('completeness_score', 0)
            gaps_count = len(review.get('gaps_identified', []))
            violations_count = len(review.get('constraint_violations', []))
        else:
            overall_status = review.overall_status
            completeness_score = review.completeness_score
            gaps_count = len(review.gaps_identified)
            violations_count = len(review.constraint_violations)
        
        lines.append("## Design Review Summary")
        lines.append("")
        lines.append(f"- **Status:** {overall_status.replace('_', ' ').title()}")
        lines.append(f"- **Completeness:** {completeness_score:.0f}%")
        lines.append(f"- **Gaps:** {gaps_count}")
        lines.append(f"- **Violations:** {violations_count}")
        lines.append("")

    # Errors
    errors = state.get("errors", [])
    if errors:
        lines.append("## Errors")
        lines.append("")
        for error in errors:
            lines.append(f"- {error.get('message', str(error))}")
        lines.append("")

    # Table of contents for generated files
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- [Summary](./00_summary.md)")
    for i, result in enumerate(worker_results, 1):
        filename = f"{i:02d}_{result.agent_name}.md"
        lines.append(f"- [{_format_agent_name(result.agent_name)}](./{filename})")
    if state.get("final_response"):
        lines.append(f"- [Final Response](./{len(worker_results) + 1:02d}_final_response.md)")
    lines.append("")

    return "\n".join(lines)


def format_final_response_markdown(response: str, state: AgentState) -> str:
    """
    Format the final synthesized response as a markdown document.

    Args:
        response: The final response text
        state: The workflow state for context

    Returns:
        Formatted markdown string
    """
    lines = [
        "# Final Response",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        response,
    ]

    return "\n".join(lines)


def export_workflow_results(
    state: AgentState,
    output_dir: Union[str, Path],
    include_summary: bool = True,
) -> List[Path]:
    """
    Export all agent results to markdown files.

    Creates a directory with numbered markdown files for each agent's
    results, plus an optional summary file.

    Args:
        state: The complete workflow state
        output_dir: Directory to write files to (created if not exists)
        include_summary: Whether to include a summary file

    Returns:
        List of paths to created files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    created_files = []

    # Write summary
    if include_summary:
        summary_path = output_path / "00_summary.md"
        summary_content = format_workflow_summary(state)
        summary_path.write_text(summary_content, encoding="utf-8")
        created_files.append(summary_path)

    # Write worker results
    worker_results = state.get("worker_results", [])
    for i, result in enumerate(worker_results, 1):
        filename = f"{i:02d}_{result.agent_name}.md"
        file_path = output_path / filename
        content = format_worker_result_markdown(result, i)
        file_path.write_text(content, encoding="utf-8")
        created_files.append(file_path)

    # Write design review if present
    review = state.get("design_review")
    if review:
        review_index = len(worker_results) + 1
        review_path = output_path / f"{review_index:02d}_design_review.md"
        review_content = format_design_review_markdown(review)
        review_path.write_text(review_content, encoding="utf-8")
        created_files.append(review_path)

    # Write final response if present
    final_response = state.get("final_response")
    if final_response:
        response_index = len(worker_results) + (2 if review else 1)
        response_path = output_path / f"{response_index:02d}_final_response.md"
        response_content = format_final_response_markdown(final_response, state)
        response_path.write_text(response_content, encoding="utf-8")
        created_files.append(response_path)

    return created_files


def _format_agent_name(name: str) -> str:
    """Convert agent_name to human-readable format."""
    return name.replace("_", " ").title()
