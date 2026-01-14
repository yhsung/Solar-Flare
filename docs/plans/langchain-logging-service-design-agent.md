# LangChain Logging Service Design Agent - Implementation Plan

## Overview

Convert the Solar-Flare project's configuration-driven multi-agent system into a production-ready LangChain Python SDK implementation using LangGraph for orchestration. Add a new Design Review Agent for cross-validation and gap analysis.

---

## 1. Current State Analysis

### Existing Architecture

| Component | Location | Purpose |
|-----------|----------|---------|
| Main Orchestrator | [agents/AGENTS.md](../../agents/AGENTS.md) | Coordinates workers, enforces constraints |
| ISO 26262 Analyzer | [agents/sub-agents/iso_26262_compliance_analyzer/](../../agents/sub-agents/iso_26262_compliance_analyzer/) | Safety compliance analysis |
| Embedded Designer | [agents/sub-agents/embbed_architecture_designer/](../../agents/sub-agents/embbed_architecture_designer/) | Architecture & implementation |
| ASPICE Assessor | [agents/sub-agents/aspice_process_assessor/](../../agents/sub-agents/aspice_process_assessor/) | Process compliance |

### Mandatory Hardware Constraints (Immutable)

```
- Transport: Interrupt-driven Mailbox (64-byte payload)
- Data Movement: DMA (64 KB per burst)
- Synchronization: Global HW Timer (1ns resolution, 64-bit timestamp)
- Performance: ≤3% CPU per core, ≤10 MB/s aggregate bandwidth
- Memory: Per-core local fixed-size Ring Buffer
- Overflow: LOG_POLICY_OVERWRITE / LOG_POLICY_STOP
```

---

## 2. Missing Parts Identified

### Critical Gaps

| Gap | Priority | Description |
|-----|----------|-------------|
| No Python Code | Critical | Everything is config/prompts only |
| No Tool Implementations | Critical | Tools referenced but not implemented |
| No State Management | High | No multi-turn conversation support |
| No Design Review Agent | High | Missing cross-validation capability |
| No Testing Framework | High | No unit/integration tests |
| No RAG for Standards | Medium | No vector store for ISO 26262/ASPICE docs |
| No Observability | Medium | No tracing/logging |

### Additional Tools Needed

- `calculate_cpu_overhead` - Validate CPU budget compliance
- `generate_memory_layout` - Create memory maps for designs
- `build_traceability_matrix` - Link requirements → design → tests
- `generate_fmea` - Create failure mode analysis tables
- `validate_hardware_constraints` - Check designs against limits
- `cross_reference_requirements` - Gap analysis vs standards

---

## 3. Proposed File Structure

```
Solar-Flare/
├── agents/                          # Keep existing configs
├── docs/
│   └── plans/                       # This plan
├── src/
│   └── solar_flare/
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       ├── constants.py             # Hardware constraints, ASIL levels
│       │
│       ├── agents/                  # Agent implementations
│       │   ├── __init__.py
│       │   ├── base.py              # Base agent class
│       │   ├── orchestrator.py      # Main supervisor
│       │   ├── iso_26262_analyzer.py
│       │   ├── embedded_designer.py
│       │   ├── aspice_assessor.py
│       │   └── design_reviewer.py   # NEW
│       │
│       ├── tools/                   # Tool implementations
│       │   ├── __init__.py
│       │   ├── web_search.py        # Tavily integration
│       │   ├── url_reader.py        # URL content fetching
│       │   ├── github_tools.py      # GitHub file/directory
│       │   └── analysis_tools.py    # NEW: Design analysis
│       │
│       ├── graph/                   # LangGraph definitions
│       │   ├── __init__.py
│       │   ├── state.py             # State schemas
│       │   ├── nodes.py             # Graph nodes
│       │   ├── edges.py             # Conditional routing
│       │   └── workflow.py          # Main workflow
│       │
│       ├── prompts/                 # Prompt templates
│       │   ├── orchestrator.py
│       │   ├── iso_26262.py
│       │   ├── embedded.py
│       │   ├── aspice.py
│       │   └── design_review.py
│       │
│       └── memory/                  # Memory & persistence
│           ├── conversation.py      # Conversation history
│           ├── checkpointer.py      # State persistence
│           └── vector_store.py      # RAG for standards
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
│
├── examples/
├── pyproject.toml
└── README.md
```

---

## 4. Core Implementation Components

### 4.1 State Schema

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    current_request: Optional[DesignRequest]
    hardware_constraints: HardwareConstraints
    orchestrator_decision: Optional[OrchestratorDecision]
    worker_results: Annotated[List[WorkerResult], add]
    design_review: Optional[DesignReviewResult]
    final_response: Optional[str]
    iteration_count: int
    max_iterations: int
    current_phase: Literal["understanding", "delegating", "executing", "reviewing", "synthesizing", "complete"]
```

### 4.2 LangGraph Workflow Structure

```
START
  │
  ▼
[understand_request] ──► Orchestrator analyzes request
  │
  ▼
[route_to_agent] ──► Conditional routing
  │
  ├──► [iso_26262_analyzer] ──┐
  ├──► [embedded_designer] ───┤
  ├──► [aspice_assessor] ─────┤
  └──► [design_reviewer] ─────┘
                              │
                              ▼
                        [re_route] ◄── Loop until complete
                              │
                              ▼
                    [synthesize_response]
                              │
                              ▼
                            END
```

### 4.3 Design Review Agent (NEW)

Purpose: Cross-validate designs against ISO 26262 and ASPICE requirements

**Capabilities:**
1. Hardware constraint validation (CPU, bandwidth, memory)
2. ISO 26262 coverage analysis (safety goals, FSR/TSR, traceability)
3. ASPICE work product completeness check
4. Gap identification with severity ratings

**Output:**
- Overall status: APPROVED / NEEDS_REVISION / REJECTED
- Completeness score (0-100%)
- Prioritized gaps list
- Cross-reference issues
- Specific recommendations

---

## 5. Implementation Phases

### Phase 1: Foundation
**Files to create:**
- `src/solar_flare/graph/state.py`
- `src/solar_flare/tools/web_search.py`
- `src/solar_flare/tools/url_reader.py`
- `src/solar_flare/tools/github_tools.py`
- `src/solar_flare/agents/base.py`
- `tests/unit/test_tools.py`

### Phase 2: Agent Migration
**Files to create:**
- `src/solar_flare/agents/orchestrator.py`
- `src/solar_flare/agents/iso_26262_analyzer.py`
- `src/solar_flare/agents/embedded_designer.py`
- `src/solar_flare/agents/aspice_assessor.py`
- `src/solar_flare/prompts/*.py`

### Phase 3: LangGraph Workflow
**Files to create:**
- `src/solar_flare/graph/workflow.py`
- `src/solar_flare/graph/nodes.py`
- `src/solar_flare/graph/edges.py`
- `tests/integration/test_workflow.py`

### Phase 4: Design Review Agent
**Files to create:**
- `src/solar_flare/agents/design_reviewer.py`
- `src/solar_flare/tools/analysis_tools.py`
- `tests/unit/test_design_reviewer.py`

### Phase 5: Memory & Persistence
**Files to create:**
- `src/solar_flare/memory/conversation.py`
- `src/solar_flare/memory/checkpointer.py`
- `src/solar_flare/memory/vector_store.py`

### Phase 6: Evaluation & Documentation
**Files to create:**
- `tests/evaluation/evaluator.py`
- `tests/evaluation/test_cases.json`
- `docs/*.md`
- `examples/*.py`

---

## 6. Key Dependencies

```toml
[project]
dependencies = [
    "langchain>=0.3.0",
    "langgraph>=0.2.0",
    "langchain-openai>=0.2.0",
    "langchain-anthropic>=0.2.0",
    "langchain-community>=0.3.0",
    "langsmith>=0.1.0",
    "tavily-python>=0.3.0",
    "faiss-cpu>=1.7.0",
    "chromadb>=0.4.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
]
```

---

## 7. Verification Plan

### Unit Tests
- Test each tool returns expected format
- Test constraint validation catches violations
- Test cross-reference identifies gaps

### Integration Tests
- Test full workflow completes for design requests
- Test multi-turn conversation maintains context
- Test iteration limit is respected

### Evaluation Criteria
- **Constraint Compliance**: All designs validate against hardware limits
- **Safety Coverage**: ISO 26262 requirements addressed for given ASIL
- **Traceability**: Bidirectional links maintained
- **Completeness**: All required artifacts present

### Manual Testing
1. Run `examples/full_workflow.py` with sample design request
2. Verify orchestrator delegates to correct workers
3. Verify design reviewer identifies intentionally introduced gaps
4. Verify final synthesis prioritizes safety over performance

---

## 8. Critical Files Reference

| File | Purpose |
|------|---------|
| [agents/AGENTS.md](../../agents/AGENTS.md) | Orchestrator prompt - port delegation logic |
| [agents/sub-agents/embbed_architecture_designer/AGENTS.md](../../agents/sub-agents/embbed_architecture_designer/AGENTS.md) | Design principles & output formats |
| [agents/sub-agents/iso_26262_compliance_analyzer/AGENTS.md](../../agents/sub-agents/iso_26262_compliance_analyzer/AGENTS.md) | ASIL analysis framework |
| [agents/sub-agents/aspice_process_assessor/AGENTS.md](../../agents/sub-agents/aspice_process_assessor/AGENTS.md) | Capability level assessment |
| [agents/tools.json](../../agents/tools.json) | Tool configuration structure |

---

## 9. Next Steps

1. Initialize Python package structure with `pyproject.toml`
2. Implement state schemas in `src/solar_flare/graph/state.py`
3. Implement the 4 core tools
4. Port orchestrator agent with delegation logic
5. Build LangGraph workflow
6. Add Design Review Agent
7. Set up testing framework
8. Create usage examples
