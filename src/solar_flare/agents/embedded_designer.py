"""
Embedded Architecture Designer Agent.

Specialized worker for detailed embedded system architecture design
and implementation guidance for logging service components.
"""

from typing import List, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage

from solar_flare.agents.base import BaseWorkerAgent, AgentRegistry
from solar_flare.graph.state import WorkerResult, HardwareConstraints
from solar_flare.prompts.embedded import (
    EMBEDDED_DESIGNER_SYSTEM_PROMPT,
    EMBEDDED_DESIGN_PROMPT,
    EMBEDDED_OUTPUT_FORMAT,
)


@AgentRegistry.register("embedded_designer")
class EmbeddedDesignerAgent(BaseWorkerAgent):
    """
    Expert embedded systems architect for logging service design.

    Creates detailed implementations meeting strict performance and
    safety requirements for automotive applications.

    Capabilities:
    - Ring buffer design with lock-free algorithms
    - DMA controller configuration and descriptor chains
    - ISR design with latency analysis
    - Performance analysis (CPU overhead, bandwidth)
    - Memory layout generation
    - Reference implementations in C/pseudocode
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        hardware_constraints: HardwareConstraints,
    ):
        super().__init__(
            llm=llm,
            tools=tools,
            agent_name="embedded_designer",
            hardware_constraints=hardware_constraints,
        )

    @property
    def system_prompt(self) -> str:
        return EMBEDDED_DESIGNER_SYSTEM_PROMPT

    def _build_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("system", self.format_constraints_reminder()),
                ("system", EMBEDDED_OUTPUT_FORMAT),
                MessagesPlaceholder(variable_name="messages"),
                ("human", EMBEDDED_DESIGN_PROMPT),
            ]
        )

    async def execute(
        self,
        task_context: Dict[str, Any],
        messages: List[BaseMessage],
    ) -> WorkerResult:
        """
        Execute embedded architecture design.

        Args:
            task_context: Contains 'components', 'platform', 'rtos', 'asil_level'
            messages: Conversation history for context

        Returns:
            WorkerResult with design artifacts and performance analysis
        """
        # Extract context with defaults
        components = task_context.get("components", ["ring_buffer"])
        platform = task_context.get("platform", "ARM Cortex-R5")
        rtos = task_context.get("rtos", "FreeRTOS")
        asil_level = task_context.get("asil_level", "ASIL_B")
        requirements = task_context.get("requirements", "")

        # Build requirements string if not provided
        if not requirements:
            requirements = self._build_default_requirements(components)

        # Execute the design chain
        prompt = self._prompt_template
        chain = prompt | self.llm

        try:
            response = await chain.ainvoke(
                {
                    "messages": messages,
                    "component": ", ".join(components),
                    "platform": platform,
                    "rtos": rtos,
                    "asil_level": asil_level,
                    "requirements": requirements,
                }
            )

            # Generate performance metrics
            performance = self._calculate_performance_estimates(components)

            # Parse response for findings
            findings = self._extract_design_findings(response.content, performance)

            return WorkerResult(
                agent_name=self.agent_name,
                task_type="embedded_architecture_design",
                status="success",
                findings=findings,
                recommendations=self._generate_recommendations(components, asil_level),
                artifacts={
                    "components": components,
                    "platform": platform,
                    "rtos": rtos,
                    "performance": performance,
                    "full_design": response.content,
                },
                confidence_score=0.9,
            )

        except Exception as e:
            return self._create_error_result(
                f"Embedded design failed: {str(e)}",
                "embedded_architecture_design",
            )

    def _build_default_requirements(self, components: List[str]) -> str:
        """Build default requirements based on components."""
        reqs = []

        if "ring_buffer" in components:
            reqs.extend(
                [
                    "Lock-free ring buffer with atomic operations",
                    "Support for OVERWRITE and STOP overflow policies",
                    "Cache-line aligned entries for performance",
                ]
            )

        if "dma" in components or "dma_controller" in components:
            reqs.extend(
                [
                    "Descriptor chain-based DMA transfers",
                    "64KB burst transfers",
                    "Error handling for transfer failures",
                ]
            )

        if "isr" in components or "interrupt" in components:
            reqs.extend(
                [
                    "Minimal ISR execution time",
                    "Deferred processing for non-critical work",
                    "Priority configuration guidance",
                ]
            )

        return "\n".join(f"- {req}" for req in reqs)

    def _calculate_performance_estimates(self, components: List[str]) -> Dict[str, Any]:
        """
        Calculate estimated performance metrics.

        These are example estimates - actual values depend on specific design.
        """
        # Base estimates (conservative)
        isr_cycles = 200  # Cycles per ISR invocation
        buffer_cycles = 50  # Cycles per buffer operation
        dma_setup_cycles = 100  # Cycles for DMA setup

        total_cycles_per_log = isr_cycles + buffer_cycles + dma_setup_cycles

        # Assuming 400 MHz core and 10,000 logs/sec
        core_freq_mhz = 400
        log_rate_hz = 10000

        cycles_per_second = total_cycles_per_log * log_rate_hz
        cpu_overhead = (cycles_per_second / (core_freq_mhz * 1_000_000)) * 100

        # Bandwidth: 64 bytes per log at 10,000 logs/sec = 640 KB/s = 0.625 MB/s
        bandwidth_mbps = (64 * log_rate_hz) / (1024 * 1024)

        return {
            "isr_cycles": isr_cycles,
            "buffer_cycles": buffer_cycles,
            "dma_setup_cycles": dma_setup_cycles,
            "total_cycles_per_log": total_cycles_per_log,
            "log_rate_hz": log_rate_hz,
            "core_frequency_mhz": core_freq_mhz,
            "cpu_overhead_percent": round(cpu_overhead, 4),
            "bandwidth_mbps": round(bandwidth_mbps, 4),
            "compliant": cpu_overhead <= 3.0 and bandwidth_mbps <= 10.0,
        }

    def _extract_design_findings(
        self, response: str, performance: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract findings from the design response."""
        findings = []

        # Performance compliance finding
        if performance["compliant"]:
            findings.append(
                {
                    "type": "performance",
                    "severity": "low",
                    "description": f"Design meets performance constraints: "
                    f"{performance['cpu_overhead_percent']}% CPU, "
                    f"{performance['bandwidth_mbps']} MB/s bandwidth",
                }
            )
        else:
            findings.append(
                {
                    "type": "performance",
                    "severity": "critical",
                    "description": f"Design may exceed performance limits: "
                    f"{performance['cpu_overhead_percent']}% CPU (limit 3%), "
                    f"{performance['bandwidth_mbps']} MB/s (limit 10 MB/s)",
                }
            )

        # Design pattern findings
        findings.append(
            {
                "type": "design_pattern",
                "severity": "low",
                "description": "Lock-free ring buffer pattern recommended for multi-core logging",
            }
        )

        return findings

    def _generate_recommendations(
        self, components: List[str], asil_level: str
    ) -> List[str]:
        """Generate recommendations based on design context."""
        recs = [
            "Validate performance calculations with actual hardware measurements",
            "Implement comprehensive error handling for all failure modes",
            "Add diagnostic hooks for runtime monitoring",
        ]

        if asil_level in ["ASIL_C", "ASIL_D"]:
            recs.extend(
                [
                    "Add CRC protection for log entry integrity",
                    "Implement redundant timestamp validation",
                    "Consider dual-channel logging for critical data",
                ]
            )

        if "dma" in str(components).lower():
            recs.append("Ensure proper cache maintenance for DMA buffers")
            recs.append("Implement DMA transfer timeout monitoring")

        return recs
