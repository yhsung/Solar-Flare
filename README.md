# Solar-Flare

A LangChain-based multi-agent system for ISO 26262/ASPICE compliant logging service design in automotive embedded systems.

## Overview

Solar-Flare provides an agentic workflow for designing safety-critical logging services for automotive embedded systems. It coordinates specialized agents to ensure compliance with:

- **ISO 26262** - Functional safety standard for automotive systems (ASIL levels QM to D)
- **ASPICE** - Automotive SPICE process capability assessment (levels 0-5)

The system enforces strict hardware constraints typical of automotive MCUs:
- Interrupt-driven Mailbox (64-byte payload)
- DMA-based data movement (64 KB per burst, zero-copy)
- Global Hardware Timer synchronization (1ns resolution, 64-bit timestamps)
- Performance budgets (≤3% CPU per core, ≤10 MB/s aggregate)
- Per-core Ring Buffers with configurable overflow policies

## Installation

### Prerequisites

- Python 3.10 or higher
- LLM provider (choose one):
  - Cloud: API keys for OpenAI or Anthropic
  - Local: Ollama or LM Studio installed
- Tavily API key for web search (optional, for research capabilities)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-org/Solar-Flare.git
cd Solar-Flare

# Install in development mode
pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```bash
# Cloud LLM Providers (choose one)
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...

# Local LLM Providers (alternative to cloud)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
# or
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model

# LLM Settings
LLM_TEMPERATURE=0.3  # 0.0=deterministic, 1.0=creative

# Optional: For web search capabilities
TAVILY_API_KEY=tvly-...

# Optional: LangSmith for tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
```

## Quick Start

### Basic Usage

```python
import asyncio
from langchain_openai import ChatOpenAI
from solar_flare import run_workflow

async def main():
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # Define your design request
    request = """
    Design a lock-free ring buffer implementation for ASIL-D compliance
    on a Cortex-R5 multi-core system using AUTOSAR OS.
    """

    # Run the workflow
    result = await run_workflow(
        llm=llm,
        user_message=request,
        session_id="my-design-session"
    )

    # Access results
    print(result["final_response"])

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Different LLM Providers

Solar-Flare supports multiple LLM providers through a unified factory function:

```python
from solar_flare import create_llm, LLMProvider

# OpenAI (default)
llm = create_llm()

# OpenAI with specific model
llm = create_llm(provider="openai", model="gpt-4o", temperature=0.3)

# Anthropic Claude
llm = create_llm(provider="anthropic", model="claude-3-5-sonnet-20241022")

# Ollama (local)
llm = create_llm(provider="ollama", model="llama3.1")

# LM Studio (OpenAI-compatible local server)
llm = create_llm(
    provider="lmstudio",
    base_url="http://localhost:1234/v1",
    model="local-model"
)
```

#### Direct Provider Instantiation

You can also use LangChain provider classes directly:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

# OpenAI
llm_openai = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Anthropic Claude
llm_claude = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)

# Ollama (local)
llm_ollama = ChatOllama(model="llama3.1", base_url="http://localhost:11434")
```

#### Local LLM Setup

**Ollama:**
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model: `ollama pull llama3.1`
3. Use with Solar-Flare:
   ```python
   llm = create_llm(provider="ollama", model="llama3.1")
   ```

**LM Studio:**
1. Download LM Studio from [lmstudio.ai](https://lmstudio.ai)
2. Load a model and start the local server
3. Use with Solar-Flare:
   ```python
   llm = create_llm(provider="lmstudio", base_url="http://localhost:1234/v1")
   ```

## Architecture

### Multi-Agent Workflow

Solar-Flare uses a LangGraph-based workflow with 5 specialized agents:

```
                    ┌─────────────────────┐
                    │   Orchestrator      │
                    │   (Routing &        │
                    │    Coordination)    │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
        ┌───────▼──────┐ ┌────▼─────┐ ┌────▼─────┐
        │ ISO 26262    │ │ Embedded │ │ ASPICE   │
        │ Analyzer     │ │ Designer │ │ Assessor │
        └───────┬──────┘ └────┬─────┘ └────┬─────┘
                │              │              │
                └──────────────┼──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Design Reviewer   │
                    │   (Cross-check &    │
                    │    Validation)      │
                    └─────────────────────┘
```

### Agent Capabilities

| Agent | Purpose |
|-------|---------|
| **Orchestrator** | Analyzes requests, routes to appropriate workers, synthesizes results |
| **ISO 26262 Analyzer** | Validates designs against ISO 26262 functional safety requirements |
| **Embedded Designer** | Generates architecture and implementation designs |
| **ASPICE Assessor** | Evaluates process compliance and capability levels |
| **Design Reviewer** | Cross-validates designs, identifies gaps and violations |

## Hardware Constraints

The system enforces mandatory hardware constraints that all designs must validate:

| Constraint | Default Value | Description |
|------------|---------------|-------------|
| Mailbox Payload | 64 bytes | Control signaling and descriptors |
| DMA Burst | 64 KB | Zero-copy data movement |
| Timer Resolution | 1 ns | Global hardware system timer |
| Timestamp Width | 64 bits | Log entry timestamp field |
| CPU Overhead | ≤3% | Per-core logging overhead |
| Bandwidth | ≤10 MB/s | Aggregate logging bandwidth |

### Custom Constraints

```python
from solar_flare import HardwareConstraints, compile_workflow

custom_constraints = HardwareConstraints(
    mailbox_payload_bytes=128,
    dma_burst_bytes=32768,  # 32 KB
    max_cpu_overhead_percent=2.0,
    max_bandwidth_mbps=5.0,
)

app = compile_workflow(llm, hardware_constraints=custom_constraints)
```

## Usage Examples

### Example 1: Design Request

```python
from solar_flare import run_workflow

result = await run_workflow(
    llm=llm,
    user_message="Design a DMA-based logging transport for Infineon AURIX TC397",
    session_id="dma-design"
)

# Check which agents were invoked
for worker in result["worker_results"]:
    print(f"{worker.agent_name}: {worker.status}")
```

### Example 2: Compliance Analysis

```python
result = await run_workflow(
    llm=llm,
    user_message="""
    Analyze this ring buffer design for ISO 26262 ASIL-D compliance:
    - Lock-free using atomic indices
    - DMA transfer on overflow
    - Interrupt-driven notification
    """,
    session_id="compliance-check"
)

# Get compliance findings
for worker in result["worker_results"]:
    if "ISO" in worker.agent_name:
        for finding in worker.findings:
            print(f"[{finding['severity']}] {finding['description']}")
```

### Example 3: Design Review

```python
result = await run_workflow(
    llm=llm,
    user_message="Review the logging subsystem design for gaps",
    session_id="design-review"
)

# Access review results
review = result["design_review"]
print(f"Status: {review.overall_status}")
print(f"Completeness: {review.completeness_score}%")

for gap in review.gaps_identified:
    print(f"  - {gap['description']}")
```

### Example 4: Interactive Session

```python
from solar_flare import compile_workflow
from langchain_core.messages import HumanMessage

app = compile_workflow(llm, enable_checkpointing=True)
config = {"configurable": {"thread_id": "interactive-session"}}

# First message
state = await app.ainvoke({
    "messages": [HumanMessage(content="Design log entry format")],
    "hardware_constraints": HardwareConstraints(),
}, config)

# Follow-up (resumes with context)
state = await app.ainvoke({
    **state,
    "messages": state["messages"] + [HumanMessage(content="Add variable-length support")],
}, config)
```

## Running Examples

The project includes several examples:

```bash
# Basic usage examples
python examples/basic_usage.py

# Advanced features (streaming, multi-session, etc.)
python examples/advanced_usage.py
```

### Example 5: Markdown Export

Export workflow results to markdown files for documentation:

```python
from solar_flare import run_workflow, export_workflow_results

result = await run_workflow(
    llm=llm,
    user_message="Design logging buffer",
    output_dir="./output/my_design"  # Auto-exports to markdown
)

# Or manually export
from solar_flare import export_workflow_results
export_workflow_results(result, "./output/my_design")
```

Generated files:
- `00_summary.md` - Workflow summary
- `01_<agent_name>.md` - Per-agent results
- `design_review.md` - Design review findings

### Example 6: Multi-Turn Requirements with Session Persistence

Maintain session history across multiple runs:

```python
from solar_flare import (
    load_session, save_session, create_session,
    append_iteration, merge_requirements
)

# Define requirements
requirements = [
    {"id": "REQ-001", "title": "Ring Buffer", "asil_level": "ASIL-D"},
    {"id": "REQ-002", "title": "DMA Transport", "asil_level": "ASIL-D"},
]

# Load existing or create new session
session = load_session("./output") or create_session("my-session", requirements)

# Merge any new requirements (skips duplicates)
merge_requirements(session, new_requirements)

# Run iteration and save
result = await run_workflow(llm=llm, user_message=message, output_dir="./output/iter_1")
append_iteration(session, message, result.get("worker_results", []))
save_session(session, "./output")
```

Session files:
- `session.json` - Full session state
- `session_summary.md` - Human-readable summary
- `traceability_matrix.md` - Requirements trace

### Example 7: Import Requirements from Redmine/Jira

Fetch requirements directly from your issue tracker:

```python
from solar_flare import (
    load_requirements_from_redmine,
    load_requirements_from_jira,
    create_session,
)

# From Redmine (filter by tracker type)
requirements = load_requirements_from_redmine(
    project="logging-service",
    tracker="Requirement",  # Filter by tracker name
)

# From Jira (filter with JQL)
requirements = load_requirements_from_jira(
    project="LOG",
    jql="type = Requirement AND status != Done",
)

# Use with session
session = create_session("my-session", requirements=requirements)
```

**Installation:**
```bash
pip install solar-flare[redmine]  # For Redmine
pip install solar-flare[jira]     # For Jira
```

**Configuration (.env):**
```bash
# Redmine
REDMINE_URL=https://redmine.example.com
REDMINE_API_KEY=your-api-key

# Jira  
JIRA_URL=https://company.atlassian.net
JIRA_USERNAME=email@example.com
JIRA_API_TOKEN=your-api-token
```

## Testing


### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# With coverage
pytest --cov=solar_flare
```

### Test Requirements

Some integration tests require API keys. Mock tests can run without credentials:

```bash
# Run tests that don't require API keys
pytest -m "not integration"
```

## API Reference

### Core Functions

#### `run_workflow()`

```python
async def run_workflow(
    llm: BaseChatModel,
    user_message: str,
    hardware_constraints: Optional[HardwareConstraints] = None,
    session_id: str = "default",
    max_iterations: int = 10,
    output_dir: Optional[str] = None,  # Auto-export to markdown
) -> Dict[str, Any]
```

Convenience function to run a complete workflow execution. If `output_dir` is provided, results are automatically exported to markdown.

#### `compile_workflow()`

```python
def compile_workflow(
    llm: BaseChatModel,
    hardware_constraints: Optional[HardwareConstraints] = None,
    enable_checkpointing: bool = True,
)
```

Compile the workflow for repeated execution with state persistence.

### State Schema

#### `AgentState`

The main state object that flows through all workflow nodes:

- `messages`: Conversation history
- `hardware_constraints`: Hardware constraints reference
- `worker_results`: Results from completed workers
- `design_review`: Design review assessment
- `final_response`: Synthesized final response
- `iteration_count`: Current workflow iteration
- `current_phase`: Current workflow phase

#### `HardwareConstraints`

Immutable hardware constraints for all designs:

```python
constraints = HardwareConstraints(
    mailbox_payload_bytes=64,
    dma_burst_bytes=65536,
    timestamp_resolution_ns=1,
    timestamp_bits=64,
    max_cpu_overhead_percent=3.0,
    max_bandwidth_mbps=10.0,
)
```

## RAG (Retrieval-Augmented Generation)

Solar-Flare includes built-in RAG capabilities for retrieving relevant ISO 26262 and ASPICE standards during design and analysis.

**Quick Start:**

```python
from solar_flare.memory import create_default_vector_store

# Create vector store with pre-loaded standards
store = create_default_vector_store()

# Search for relevant content
results = store.search("ASIL D diagnostic coverage requirements", top_k=3)
```

See [RAG_GUIDE.md](docs/RAG_GUIDE.md) for detailed documentation on:
- Document preparation and ingestion
- Vector store configuration
- Integration with agents
- Best practices and troubleshooting
- **Multilingual support** (Mandarin/Chinese, cross-language retrieval)

## Development

```
Solar-Flare/
├── src/solar_flare/
│   ├── agents/          # Agent implementations
│   ├── graph/           # LangGraph workflow and state
│   ├── memory/          # Vector store, checkpointing, conversation
│   ├── prompts/         # Agent prompt templates
│   ├── tools/           # Tool integrations
│   └── utils/           # Utilities
├── examples/            # Usage examples
├── tests/               # Test suite
├── docs/                # Documentation (RAG guide, status reports)
└── pyproject.toml       # Project configuration
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/
```

## License

Apache License 2.0 - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/your-org/Solar-Flare).

## Acknowledgments

Solar-Flare is built on top of:
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent workflow orchestration
- [Pydantic](https://github.com/pydantic/pydantic) - Data validation
