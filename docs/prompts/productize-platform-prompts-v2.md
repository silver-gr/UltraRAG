# Productization Prompts v2 -- Self-Help Knowledge Platform

## Prompt 1: Gemini 3 Pro (Deep Think)

**Settings:** `temperature: 1.0` | `thinking_level: high`

**System instruction (set in API/UI, not in user message):**
```
You are a product strategist who has built and scaled consumer knowledge platforms. You think in build sequences, not feature lists. You default to the simplest architecture that validates learning. When you see overcomplicated plans from solo builders, you cut scope aggressively.
```

**User message:**
```xml
<context>
I am a solo developer with deep domain expertise in self-help, psychology, productivity, neuroscience, and personal development. I have spent years curating a 45-million-word personal knowledge base:

- 5M words: Obsidian vault (personal notes, research synthesis, interconnected ideas)
- 5M words: AI conversation history (ChatGPT, Claude, Gemini -- years of deep dialogue on these domains)
- 28M words: Saved content via "TheSource" (articles, YouTube transcripts, Reddit threads, GitHub repos, bookmarks -- curated, not scraped)
- 7M words: Books in PDF/EPUB (growing to ~50M words with 700-800 more titles)

This is indexed in UltraRAG, a production RAG system I built, with:
- Semantic search + BM25 hybrid retrieval with Reciprocal Rank Fusion
- Voyage AI embeddings + reranking
- HyDE and multi-query expansion
- Iterative research mode with convergence detection
- Federated query across vault, conversations, saved items, and books
- Fact-check pipeline that validates generated claims against source documents
- Bilingual support (English + Greek)

I want to productize this into a self-help / personal development platform.

WHAT EXISTS IN THIS SPACE:
- Blinkist/Shortform: Book summaries, no personalization, no community, no original synthesis
- Headway/Knowt: Learning apps, no deep knowledge base backing them
- Huberman Lab AI: Single-creator RAG, no community layer
- Substack/newsletters: Manual content creation, no personalization
- None of these combine: curated multi-source knowledge base + RAG generation + fact-checking + community evaluation + personalization

MY CURRENT THINKING (challenge this):
1. CONTENT GENERATION: RAG generates articles/insights/guides from source material. Fact-check pipeline validates claims against sources before publication.
2. COMMUNITY: 2-4 AI agents manage a community -- moderating, surfacing content, engaging members.
3. PERSONALIZATION: Content generated per-user based on interests and journey stage, then made available to other matching users. Growing library of verified, personalized content.
4. COMMUNITY EVALUATION: Users vote (thumbs up/down) and comment on extracted facts from published material. Human-in-the-loop quality signal.
5. RECOMMENDATIONS: Engine matches users to existing generated content and suggests new content to generate.
6. MY ROLE: System should be mostly autonomous with quality controls. I review a sample, not everything.

TARGET USER: Greek-speaking professionals (25-40) interested in self-improvement who read English content but prefer consuming synthesized material in Greek. They follow creators like Andrew Huberman, Ali Abdaal, Atomic Habits-style content. They want depth, not listicles.

BUILDER REALITY: Solo developer. Strong Python/RAG/ML skills. No frontend team. Limited budget for infrastructure. Need to validate before investing months of build time.
</context>

<task>
Design the product architecture for this platform. For each section, provide: the design decision, the reasoning, and the key risk.

A. CONTENT PIPELINE: Flow from RAG retrieval to generation to fact-check to publication to user delivery. Which content formats create the most value with least production cost? (micro-insights, long articles, daily digests, Q&A, threads)

B. PERSONALIZATION MODEL: How to build a user model from minimal initial input that improves over time. The hardest question: in self-help specifically, how do you balance "what the user wants to hear" vs "what the user needs to hear"? My assumption is that recommending only comfortable content creates a filter bubble that defeats the purpose of personal development. Challenge this assumption.

C. AGENT ARCHITECTURE: What should each agent own? How do they coordinate? What requires human escalation? My assumption is 2-4 agents. Challenge whether this is too many or too few, and whether "community management" is even the right job for agents at MVP stage.

D. CONTENT REUSE ECONOMICS: When content generated for User A fits User B. Taxonomy/tagging system. How to avoid the "content mill" feeling while scaling. My assumption is that personalized generation + reuse creates a flywheel. Challenge whether users will trust "AI-generated self-help content" at all.

E. RECOMMENDATION ENGINE: What signals drive recommendations. How self-help recs differ from entertainment recs (Netflix/Spotify). Cold-start strategy. My assumption is that explicit preference capture (onboarding quiz) + implicit signals (reading time, votes, saves) is sufficient. Challenge this.

F. QUALITY FLYWHEEL: How community votes, fact-check scores, my manual reviews, and usage data combine into a single quality score. How this improves the system over time.

G. MONETIZATION: Pricing model. What's the defensible moat beyond "I have the data"? My assumption is the moat is: curated knowledge base + domain expertise + community trust + content quality flywheel. Challenge whether this is actually defensible.

H. MVP SCOPE: The smallest version that validates the core value proposition with real users. What to cut ruthlessly. What single metric proves the concept works.
</task>

<constraints>
- I am a solo developer. Any architecture that requires a team to maintain is wrong.
- Do not design for scale I don't have. Design for 50-200 early users first.
- Distinguish clearly between "build in week 1-2" and "build after validation"
- Be direct about which of my assumptions are wrong or overcomplicated
- If community agents are premature for MVP, say so -- I can take it
- Output structured with clear sections A-H
- For each section: recommendation, reasoning, key risk, build priority (MVP vs later)
</constraints>
```

---

## Prompt 2: GPT 5.2 Pro

**Settings:** `reasoning_profile: "deep"`

**System message:**
```
Product architect for solo-developer consumer apps. You cut scope aggressively and optimize for validation speed. You've seen solo builders overcomplicate platforms that never launch.
```

**User message:**
```markdown
## Context

Solo developer. 45M-word curated knowledge base in self-help/psychology/productivity (Obsidian notes, AI conversations, saved articles/videos, books). Indexed in a production RAG system with semantic search, reranking, fact-checking, and iterative research.

Target: Greek-speaking professionals (25-40) into self-improvement. They consume English content but want synthesized material in Greek. They want depth, not listicles.

Competitive gap: Blinkist = summaries without personalization. Huberman AI = single-source RAG. Newsletters = manual creation. Nothing combines curated multi-source RAG + fact-checking + community evaluation + personalization.

I want to productize this. My plan has 6 components:
1. RAG content pipeline (generate + fact-check + publish)
2. 2-4 AI community agents (moderate, engage, surface content)
3. Per-user personalization (generate for one, reuse for matching users)
4. Community voting on extracted facts (quality signal)
5. Recommendation engine (match users to content)
6. Mostly autonomous -- I review samples, not everything

## Assumptions to Challenge

- "2-4 agents managing community" might be premature for MVP
- "Personalized generation + reuse = flywheel" might not work if users don't trust AI self-help content
- "Want vs need" balance in self-help recs is the hardest unsolved problem
- My moat might not be as defensible as I think
- This might be too many components for a solo developer to validate

## Task

Design the architecture. For each of these 8 areas, give me: the design, key risk, and what to build first vs later.

1. Content pipeline (formats, flow, fact-check integration)
2. Personalization (user model, cold start, want-vs-need tension)
3. Agent architecture (roles, coordination, whether agents belong in MVP)
4. Content reuse (taxonomy, matching, avoiding content-mill feel)
5. Recommendation engine (signals, cold start, how self-help differs from entertainment)
6. Quality flywheel (votes + fact-checks + usage → single quality score)
7. Monetization and moat
8. MVP scope (smallest validating version, single success metric)

Cut anything that doesn't need to exist for a solo dev validating with 50-200 users. Be ruthless.
```

---

## Model Selection Rationale

| Aspect | Gemini 3 Pro (Deep Think) | GPT 5.2 Pro |
|--------|--------------------------|-------------|
| Prompt style | Rich context, XML, persona, constraints last | Minimal, direct, system message for role |
| Thinking | `thinking_level: high` -- exhaustive analysis | `reasoning_profile: "deep"` -- decisive architecture |
| Key strength | Will explore edge cases, surface risks you missed, go deep on each section | Will cut to decisions faster, be more opinionated about what to kill |
| Expected output | Comprehensive analysis with nuanced tradeoffs | Sharp architectural recommendations with clear build sequence |
| Context approach | Full competitive landscape, assumptions spelled out | Compressed context, assumptions listed separately for direct challenge |

**Run order:**
1. **Gemini first** -- its deep think mode will produce exhaustive analysis. Read for risks and edge cases you hadn't considered.
2. **GPT 5.2 second** -- either fresh (independent perspective) or feed it a 3-paragraph summary of Gemini's key insights as additional context.
3. **Synthesize** -- Gemini's depth + GPT's decisiveness = your actual plan.

## Anti-Pattern Check

| Code | Check | Status |
|------|-------|--------|
| AP-1 | Over-engineering | Clean -- Gemini prompt is detailed because problem is complex; GPT prompt is minimal |
| AP-2 | Explicit CoT | Clean -- no "step by step" or prescribed reasoning |
| AP-3 | Excessive few-shot | Clean -- zero-shot (strategic task, no format examples needed) |
| AP-4 | Conversational fluff | Clean -- no "please", "kindly", "I hope" |
| AP-5 | Gemini temp != 1.0 | Clean -- 1.0 specified |
| AP-6 | "Think" sensitivity | Clean -- no "think carefully" |
| AP-7 | Verbose tool descriptions | N/A |
| AP-8 | Prescribed tool sequence | N/A |
| AP-9 | Over-prompting GPT-5 | Clean -- GPT prompt is intentionally compressed |
| AP-10 | Missing model params | Clean -- both configured |

## Key Changes from v1

| Change | Why |
|--------|-----|
| Added builder reality (solo dev, 50-200 users) | Without this, models design for teams/scale you don't have |
| Added competitive landscape | Models can now assess moat and differentiation accurately |
| Added target user persona | "Greek professionals 25-40" changes every architectural decision |
| Added system instruction with persona (Gemini) | Gemini takes personas seriously -- "product strategist who cuts scope" anchors output |
| Added system message with role (GPT 5.2) | GPT-5 strongly prioritizes system messages |
| Named specific assumptions to challenge | "Challenge my assumptions" is vague; naming them gives models specific targets |
| Added bilingual context | Content generation in Greek from English sources is a non-trivial architectural consideration |
| Added MVP constraint (50-200 users) | Forces models to design for validation, not scale |
| Added "single success metric" to MVP section | Forces both models to commit to what "working" means |

## Usage Tips

1. **Copy the system instruction/message separately** -- don't paste it into the user message. Use the platform's system instruction field (Gemini AI Studio) or system message role (OpenAI API/Playground).

2. **The "want vs need" question** (section B) is the hardest product problem. Neither model will fully solve it -- that's where your domain expertise matters. Push back on their answers. Ask follow-ups like: "How does the system know when a user is avoiding difficult growth topics vs genuinely not interested?"

3. **For follow-up depth**, ask the same model to expand one section. Don't re-send the full prompt.

4. **The trust question** (section D) is the second hardest problem. AI-generated self-help content has a credibility problem. Both models should address this, but if they don't go deep enough, ask: "What specific mechanisms make users trust that this content is grounded in real expertise and not hallucinated?"

5. **After both outputs**, the synthesis question is: "What did Gemini surface as a risk that GPT dismissed, and vice versa?" The disagreements are where the real decisions live.
