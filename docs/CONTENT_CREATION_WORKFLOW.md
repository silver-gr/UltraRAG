# Content Creation Workflow

A step-by-step guide to creating well-researched, fact-checked content using the full tooling ecosystem.

**Input:** A subject/topic
**Output:** Publication-ready content backed by 45M+ words of personal knowledge + web research + multi-model validation

---

## Phase 1: Define & Refine (2 min)

**Tool:** Prompt Engineer (`/prompt-context-engineer`)

Before researching, sharpen the topic into a clear angle:
- What's the thesis or question?
- Who's the audience?
- What format? (article, thread, essay, guide)

**Why first:** A vague topic leads to unfocused research. Prompt Engineer's anti-pattern scanner catches ambiguity early.

```
/prompt-context-engineer "I want to write about [topic]. Help me refine the angle, identify the core thesis, and suggest 3-5 key questions the content should answer."
```

**Output:** Clear thesis + key questions to guide research.

---

## Phase 2: Internal Knowledge Sweep (3-5 min)

**Tool:** UltraRAG Research Mode

Search your own knowledge base first. You've accumulated 45M words — use them before going external.

```bash
# Deep research mode — iterates until confidence threshold
python -m cli research --topic "[refined topic]" --depth deep

# Or via web UI: enable Research Mode checkbox
```

**What this searches:**
- Obsidian vault (5M words of notes)
- AI conversations (5M words from ChatGPT/Claude/Gemini)
- Books (7M words, 100+ titles)
- TheSource saved items (28M words via federated retrieval)

**Output:** Synthesized answer with citations from your personal knowledge base. Save the output — this is your foundation.

---

## Phase 3: Cross-Reference & Discover Angles (2-3 min)

**Tool:** theSource (topic clustering + semantic search)

UltraRAG found relevant chunks. Now use theSource to discover adjacent themes you might have missed:

1. **Semantic search** — find saved articles/videos on the topic
   ```
   MCP: semantic_search(query="[topic]", limit=20)
   ```

2. **Topic clustering** — see which clusters your topic falls into, explore neighboring clusters for unexpected connections
   ```
   GET /api/v1/clusters  (or via theSource dashboard)
   ```

3. **Cross-source context** — for any high-relevance item, pull its full context
   ```
   MCP: get_item_context(item_id="...")
   → Returns: Obsidian mentions, conversation mentions, similar items
   ```

**Output:** Broader context map — related themes, saved resources you forgot about, potential angles from adjacent clusters.

---

## Phase 4: External Research (5-10 min)

**Tool:** Research skills (`/research-content` or `/research-tech` or `/research-market`)

Fill gaps that your internal knowledge didn't cover. Pick the right skill:

| Content Type | Skill | What It Adds |
|-------------|-------|-------------|
| Blog/essay/personal dev | `/research-content` | Trends, hooks, quotes, competitor content |
| Technical topic | `/research-tech` | Official docs, benchmarks, code examples |
| Market/business angle | `/research-market` | Competitors, SWOT, market sizing |
| Batch of sub-topics | `/orchestrate-research` | Per-item reports + interaction matrix |

```
/research-content "[topic] — focus on [specific gaps from Phase 2]"
```

**Output:** 3-5 structured files saved to Obsidian vault (`Research/{date}-{topic}/`).

---

## Phase 5: Draft (10-30 min)

Write the first draft using everything gathered:

- **Phase 1** gave you the angle and structure
- **Phase 2** gave you personal knowledge + citations
- **Phase 3** gave you adjacent themes and forgotten resources
- **Phase 4** filled external gaps

Draft in your preferred editor. Reference specific sources by name — you'll verify them next.

---

## Phase 6: Fact-Check (5-10 min)

**Tool:** Fact-check pipeline (`/fact-check`)

Save your draft as an Obsidian note, then run the verification pipeline:

```
/fact-check "[path-to-draft-note]"
```

**What it does (5 phases):**
1. Extracts claims by type (statistical, causal, mechanism, etc.)
2. Verifies each via 3x Tavily + 2x WebSearch (triangulation)
3. Scores sources on 100-point scale (6 tiers)
4. Synthesizes weighted verdicts per claim
5. Generates verification report

**Verdicts:** Verified / Likely True / Uncertain / Likely False / Debunked

**Output:** `Resources/Research/YYYY-MM-DD-{topic}-Fact-Check.md` — fix any Uncertain/False claims before proceeding.

---

## Phase 7: Multi-Model Validation (3-5 min)

**Tool:** LLM Council (localhost:9007)

Final quality gate. Submit the revised draft to 8 frontier models for deliberation:

```
POST /api/conversations/{id}/message
{
  "content": "Review this article for accuracy, clarity, completeness, and persuasiveness. Identify any weak arguments, missing perspectives, or factual concerns:\n\n[draft]",
  "rubric": "Score on: factual accuracy, argument strength, readability, originality, actionability"
}
```

**3-stage process:**
1. All 8 models review independently (Claude, Gemini, GPT, Kimi, GLM)
2. Ranking models evaluate and rank all responses
3. Chairman synthesizes the final consensus review

**Output:** Ranked feedback from 8 models + synthesis. Address consensus issues.

---

## Phase 8: Polish & Publish

Apply LLM Council feedback, do a final pass, publish.

---

## Quick Reference: When to Use What

| Need | Tool | Time |
|------|------|------|
| "What do I already know about X?" | UltraRAG `--depth deep` | 3-5 min |
| "What have I saved about X?" | theSource semantic search | 1-2 min |
| "What themes connect to X?" | theSource topic clustering | 1-2 min |
| "What's the latest on X?" | `/research-content` or `/research-tech` | 5-10 min |
| "Is this claim true?" | `/fact-check` | 5-10 min |
| "Is this draft good?" | LLM Council | 3-5 min |
| "How should I frame this prompt?" | `/prompt-context-engineer` | 2 min |
| "Research 20 sub-topics at once" | `/orchestrate-research` | 10-15 min |

## Total Time Estimate

| Phase | Time |
|-------|------|
| 1. Define | 2 min |
| 2. Internal research | 3-5 min |
| 3. Cross-reference | 2-3 min |
| 4. External research | 5-10 min |
| 5. Draft | 10-30 min |
| 6. Fact-check | 5-10 min |
| 7. LLM validation | 3-5 min |
| 8. Polish | 5-10 min |
| **Total** | **35-75 min** |

From topic to fact-checked, multi-model-validated, publication-ready content — backed by a 45M-word personal knowledge base and 8 frontier AI models.
