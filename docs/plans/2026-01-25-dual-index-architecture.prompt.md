# Execution Prompt: Dual Index Architecture

> **Copy this prompt to a new Claude Code session in UltraRAG directory**

---

## Context

You are implementing the Dual Index Architecture for UltraRAG. This separates Personal Notes from Research Content with different convergence profiles for each.

**Plan file:** `docs/plans/2026-01-25-dual-index-architecture.md`

**What's already done:**
- `IndexProfile` enum and `ConvergenceConfig` dataclass exist in `research_mode.py`
- `ResearchRetriever` accepts `index_profile` and `convergence_config` parameters
- Profiles: PERSONAL (aggressive), RESEARCH (conservative), BALANCED (default)

## Your Task

Execute the 10-task plan using **subagent-driven development**:

1. **Read the plan first:** `docs/plans/2026-01-25-dual-index-architecture.md`
2. **For each task:**
   - Create a subagent to implement that specific task
   - Review the subagent's work before proceeding
   - Commit after each task passes
3. **Do NOT skip tasks or combine them**

## Execution Command

```
Use the Skill tool: Skill("superpowers:subagent-driven-development")
```

Then follow its workflow for each task in the plan.

## Key Files to Understand First

Before starting, read these files to understand the architecture:

```
config.py          - Configuration models (add IndexConfig here)
settings_store.py  - Persistent settings in LanceDB
loader.py          - Document loading (add path filtering)
vector_store.py    - Vector DB operations (add partition support)
federated_query.py - Multi-index querying (IndexSource pattern)
research_mode.py   - Convergence detection (already has profiles)
app.py             - Streamlit UI (add scope selector)
main.py            - CLI commands (add index-partitions)
```

## Success Criteria

After all tasks complete:
- [ ] `python -c "from config import IndexConfig; print('OK')"` passes
- [ ] `python -c "from index_manager import IndexManager; print('OK')"` passes
- [ ] `pytest tests/test_partitioned_index.py -v` passes
- [ ] Streamlit UI shows "Index Partitions" in settings
- [ ] CLI `index-partitions` command works

## Notes

- Each task has exact code snippets in the plan - use them
- TDD: Write tests first where applicable
- Commit after each task with descriptive message
- If a task fails, fix before moving to next
