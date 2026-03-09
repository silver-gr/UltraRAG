# Productization Prompts — Self-Help Knowledge Platform

## Prompt 1: Gemini 3 Pro (Deep Think)

**Settings:** `temperature: 1.0` | `thinking_level: high`

```xml
<context>
I have a 45-million-word personal knowledge base spanning:
- 5M words: Obsidian vault (personal notes, research, ideas — self-help, psychology, productivity, habits, neuroscience)
- 5M words: AI conversation history (ChatGPT, Claude, Gemini — years of dialogue on these topics)
- 28M words: Saved content (articles, videos, Reddit threads, GitHub repos, bookmarks — curated over years)
- 7M words: Books (full PDFs and EPUBs — growing to ~50M words with 700-800 more books)

This is indexed in a RAG system (UltraRAG) with semantic search, reranking, hybrid retrieval, iterative research mode, and fact-checking capabilities.

I want to productize this into a self-help / personal development platform. My current thinking:

1. CONTENT GENERATION: The RAG system generates articles, insights, and guides from the source material. A fact-check pipeline validates claims against source documents.

2. COMMUNITY: 2-4 AI agents manage a community — moderating, surfacing content, engaging members.

3. PERSONALIZATION: Content is generated per-user based on their interests and stage, but once generated, it becomes available to other users it's relevant for. This creates a growing library of verified, personalized content.

4. COMMUNITY EVALUATION: Users vote (thumbs up/down) and comment on extracted facts from published material. This creates a human-in-the-loop quality signal.

5. RECOMMENDATIONS: A recommendation engine matches users to existing generated content and suggests new content to generate.

6. MY ROLE: Initially I planned to review all generated content daily. Now I want to minimize my involvement — the system should be mostly autonomous with quality controls.

My domain expertise is genuine — this isn't scraped content, it's years of curated learning in self-help, psychology, productivity, neuroscience, and personal development.
</context>

<task>
Design the product architecture for this platform. Address each of these:

A. CONTENT PIPELINE: How should content flow from RAG retrieval → generation → fact-check → publication → user delivery? What content formats work best (micro-insights, long articles, daily digests, Q&A)?

B. PERSONALIZATION MODEL: How do you build a user model from minimal initial input that improves over time? How do you balance "what the user wants" vs "what the user needs" in self-help specifically?

C. AGENT ARCHITECTURE: What should each of the 2-4 community agents own? How do they coordinate? What decisions require human escalation?

D. CONTENT REUSE ECONOMICS: When content is generated for User A, how do you determine it's suitable for Users B, C, D? What's the taxonomy/tagging system? How do you avoid the "content mill" feeling?

E. RECOMMENDATION ENGINE: What signals drive recommendations? How does it differ from Netflix/Spotify-style recommendations for self-help content specifically? What's the cold-start strategy?

F. QUALITY FLYWHEEL: How do community votes, fact-check scores, my manual reviews, and usage data combine into a single quality signal? How does this improve the system over time?

G. MONETIZATION: What pricing model fits? Freemium tiers? What's the defensible moat beyond "I have the data"?

H. MVP SCOPE: What's the smallest version that validates the concept with real users? What do you cut?
</task>

<constraints>
- Focus on architecturally sound decisions, not implementation details
- Call out where my current thinking has gaps or risks
- Distinguish between "build first" and "build later" components
- Be direct about what won't work or is overcomplicated
- If any of my assumptions are wrong, say so
- Output should be structured with clear sections matching A-H above
- For each section: the recommendation, the reasoning, and the key risk
</constraints>
```

---

## Prompt 2: GPT 5.2 Pro

**Settings:** `reasoning_profile: "deep"`

```markdown
## Context

45M-word personal knowledge base (Obsidian notes, AI conversations, saved articles, books) in self-help/psychology/productivity. Indexed in a RAG system with semantic search, reranking, and fact-checking.

I want to build a platform that:
- Generates personalized self-help content from this knowledge base
- Uses 2-4 AI agents to manage a community
- Fact-checks generated content against sources
- Lets users vote on extracted facts (quality signal)
- Makes content generated for one user available to matching users
- Recommends content via a recommendation engine

## Task

Design the product architecture. For each component, give me: the design, the key risk, and what to build first vs later.

1. Content pipeline (RAG → generation → fact-check → publish → deliver)
2. Personalization (user modeling, cold start, "want vs need" in self-help)
3. Agent roles and coordination (what each agent owns, escalation rules)
4. Content reuse (when User A's content fits User B, taxonomy, avoiding content-mill feel)
5. Recommendation engine (signals, cold start, how self-help differs from entertainment recs)
6. Quality flywheel (votes + fact-checks + usage + manual review → single quality signal)
7. Monetization and moat
8. MVP scope (smallest thing that validates with real users)

Be direct. Challenge my assumptions where they're wrong.
```

---

## Model Selection Rationale

| Aspect | Gemini 3 Pro | GPT 5.2 Pro |
|--------|-------------|-------------|
| Prompt style | Rich context, structured XML, constraints last | Minimal, direct, trust the model |
| Thinking | `thinking_level: high` for deep strategic analysis | `reasoning_profile: "deep"` for multi-step planning |
| Strength here | Better with large structured context, research-grade analysis | Better with concise strategic thinking, architectural decisions |
| Expected output | More exhaustive, will explore edge cases | More decisive, will cut to recommendations faster |

**Recommendation:** Run both. Gemini will give you depth and edge cases. GPT 5.2 will give you decisive architecture. Synthesize.

## Anti-Pattern Check

| Code | Check | Status |
|------|-------|--------|
| AP-1 | Over-engineering | Clean — Gemini prompt is detailed because the problem is complex; GPT prompt is minimal |
| AP-2 | Explicit CoT | Clean — no "step by step" or prescribed reasoning |
| AP-3 | Excessive few-shot | Clean — zero-shot for both (strategic task, no format examples needed) |
| AP-4 | Conversational fluff | Clean — no "please", "kindly", "I hope" |
| AP-5 | Gemini temp != 1.0 | Clean — specified 1.0 |
| AP-6 | "Think" sensitivity | Clean — no "think carefully" |
| AP-7 | Verbose tool descriptions | N/A — no tools |
| AP-8 | Prescribed tool sequence | N/A |
| AP-9 | Over-prompting GPT-5 | Clean — GPT prompt is intentionally minimal |
| AP-10 | Missing model params | Clean — both have params specified |

## Usage Tips

1. **Run Gemini first** — its output will be longer and more exploratory. Use it to surface risks and edge cases you hadn't considered.

2. **Run GPT 5.2 second** — feed it a condensed version of Gemini's output as additional context if you want it to build on those insights. Or run independently for a fresh perspective.

3. **For follow-up depth on any section**, ask the same model to expand just that section. Don't re-send the full prompt.

4. **The "want vs need" question** in section B is the hardest product problem here. Neither model will fully solve it — that's where your domain expertise matters. Push back on their answers.
