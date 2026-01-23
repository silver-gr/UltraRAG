"""Content Research — Multi-source federated search with configurable weights."""
import streamlit as st

from content_research_engine import (
    ResearchResult,
    weighted_federated_search,
    deduplicate_results,
)
from research_storage import (
    save_results,
    load_results,
    get_history,
    clear_history,
    export_markdown,
    export_json,
    EXPORTS_DIR,
)

st.set_page_config(page_title="Content Research", layout="wide")
st.title("Content Research")
st.caption("Federated search across Vault, AI Conversations, and The Source")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: Source Configuration
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Source Configuration")

    vault_enabled = st.checkbox("Vault", value=True)
    vault_weight = st.slider("Vault Weight", 0.0, 2.0, 1.0, 0.1,
                             disabled=not vault_enabled)

    conv_enabled = st.checkbox("Conversations", value=True)
    conv_weight = st.slider("Conversations Weight", 0.0, 2.0, 0.7, 0.1,
                            disabled=not conv_enabled)

    source_enabled = st.checkbox("The Source", value=True)
    source_weight = st.slider("The Source Weight", 0.0, 2.0, 0.5, 0.1,
                              disabled=not source_enabled)

    st.divider()
    st.subheader("Retrieval Settings")
    top_k = st.number_input("Top-K per source", min_value=5, max_value=100,
                            value=30, step=5)
    threshold = st.slider("Score Threshold", 0.0, 1.0, 0.4, 0.05)
    dedup_sim = st.slider("Dedup Similarity", 0.5, 1.0, 0.85, 0.05)

    # ──────────────────────────────────────────────────────────────────────────
    # Sidebar: Query History
    # ──────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Query History")
    history = get_history()

    if history:
        for entry in history[:20]:
            label = entry["query"][:30]
            if len(entry["query"]) > 30:
                label += "..."
            sublabel = (
                f"{entry['timestamp'][:10]} | "
                f"{entry['stats']['total_results']} results"
            )
            if st.button(label, key=f"hist_{entry['id']}", help=sublabel):
                stored = load_results(entry["id"])
                if stored:
                    # Reconstruct ResearchResult objects from stored dicts
                    st.session_state.results = [
                        ResearchResult(**{k: v for k, v in r.items()
                                         if k != "embedding"})
                        for r in stored["results"]
                    ]
                    st.session_state.dedup_log = stored.get("dedup_log", [])
                    st.session_state.query_settings = stored["settings"]
                    st.session_state.result_id = entry["id"]
                    st.rerun()

        if st.button("Clear History", type="secondary"):
            clear_history()
            st.rerun()
    else:
        st.caption("No queries yet.")

# ──────────────────────────────────────────────────────────────────────────────
# Main: Query Input
# ──────────────────────────────────────────────────────────────────────────────
query = st.text_input("Topic", placeholder="e.g. ADHD tips and tricks")

if st.button("Search", type="primary") and query:
    # Build weights from enabled sources
    weights = {}
    if vault_enabled:
        weights["vault"] = vault_weight
    if conv_enabled:
        weights["conversations"] = conv_weight
    if source_enabled:
        weights["saved_items"] = source_weight

    if not weights:
        st.error("Enable at least one source.")
    else:
        with st.spinner("Searching across sources..."):
            results = weighted_federated_search(
                query=query,
                weights=weights,
                top_k=top_k,
                threshold=threshold,
            )

        with st.spinner("Deduplicating..."):
            results, dedup_log = deduplicate_results(results, dedup_sim)

        # Store query settings
        query_settings = {
            "query": query,
            "weights": weights,
            "top_k": top_k,
            "threshold": threshold,
            "dedup_similarity": dedup_sim,
        }

        # Save internally
        result_id = save_results(query_settings, results, dedup_log)

        # Session state
        st.session_state.results = results
        st.session_state.dedup_log = dedup_log
        st.session_state.query_settings = query_settings
        st.session_state.result_id = result_id

        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Results Display
# ──────────────────────────────────────────────────────────────────────────────
if "results" in st.session_state and st.session_state.results:
    results = st.session_state.results
    dedup_log = st.session_state.get("dedup_log", [])
    query_settings = st.session_state.get("query_settings", {})

    # Stats bar
    sources_present = sorted(set(r.source if isinstance(r, ResearchResult)
                                 else r.get("source", "?") for r in results))
    source_counts = {}
    for r in results:
        src = r.source if isinstance(r, ResearchResult) else r.get("source", "?")
        source_counts[src] = source_counts.get(src, 0) + 1

    cols = st.columns([2, 1, 1, 1])
    with cols[0]:
        st.markdown(f"**{len(results)} results** for: _{query_settings.get('query', '')}_")
    with cols[1]:
        if dedup_log:
            st.caption(f"{len(dedup_log)} duplicates removed")
    with cols[2]:
        pass
    with cols[3]:
        pass

    # Export buttons
    exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
    with exp_col1:
        if st.button("Export Markdown"):
            path = export_markdown(st.session_state.result_id)
            if path:
                st.success(f"Saved: {path.name}")
            else:
                st.error("Export failed")
    with exp_col2:
        if st.button("Export JSON"):
            path = export_json(st.session_state.result_id)
            if path:
                st.success(f"Saved: {path.name}")
            else:
                st.error("Export failed")
    with exp_col3:
        st.caption(f"Export dir: `{EXPORTS_DIR}`")

    st.divider()

    # Tabbed results
    source_labels = {
        "vault": "Vault",
        "conversations": "Conversations",
        "saved_items": "The Source",
    }
    tab_names = ["All"] + [
        f"{source_labels.get(s, s)} ({source_counts.get(s, 0)})"
        for s in sources_present
    ]
    tabs = st.tabs(tab_names)

    def _display_result(r):
        """Display a single research result in an expander."""
        if isinstance(r, ResearchResult):
            title = r.title
            score = r.weighted_score
            raw = r.raw_score
            source = r.source
            chunk = r.chunk
            metadata = r.metadata
        else:
            title = r.get("title", "Untitled")
            score = r.get("weighted_score", 0)
            raw = r.get("raw_score", 0)
            source = r.get("source", "?")
            chunk = r.get("chunk", "")
            metadata = r.get("metadata", {})

        source_badge = source_labels.get(source, source)
        with st.expander(f"[{score:.3f}] {title}  `{source_badge}`"):
            # Chunk text
            display_chunk = chunk[:800]
            if len(chunk) > 800:
                display_chunk += "..."
            st.markdown(f"> {display_chunk}")

            # Scores
            st.caption(
                f"Source: {source_badge} | "
                f"Raw: {raw:.3f} | Weighted: {score:.3f}"
            )

            # Metadata
            if metadata:
                meta_parts = []
                for k, v in metadata.items():
                    if v is None:
                        continue
                    # Convert numpy/pandas types to native Python
                    import numpy as np
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (np.integer,)):
                        v = int(v)
                    elif isinstance(v, (np.floating,)):
                        v = float(v)
                    # Skip empty values
                    if isinstance(v, (list, tuple)):
                        if len(v) == 0:
                            continue
                        v = ", ".join(str(x) for x in list(v)[:5])
                    elif isinstance(v, str) and v == "":
                        continue
                    meta_parts.append(f"**{k}:** {v}")
                if meta_parts:
                    st.markdown(" | ".join(meta_parts))

    # Tab: All
    with tabs[0]:
        for r in results:
            _display_result(r)

    # Per-source tabs
    for i, source in enumerate(sources_present):
        with tabs[i + 1]:
            source_results = [
                r for r in results
                if (r.source if isinstance(r, ResearchResult)
                    else r.get("source")) == source
            ]
            for r in source_results:
                _display_result(r)

    # Dedup details (collapsible)
    if dedup_log:
        with st.expander(f"Dedup Details ({len(dedup_log)} removed)"):
            for entry in dedup_log:
                st.caption(
                    f"Removed: \"{entry.get('removed_title', '?')}\" "
                    f"({entry.get('removed_source', '?')}) — "
                    f"Similar to: \"{entry.get('kept_title', '?')}\" "
                    f"({entry.get('kept_source', '?')}) — "
                    f"Similarity: {entry.get('similarity', 0):.3f}"
                )

elif "results" in st.session_state and not st.session_state.results:
    st.info("No results found. Try adjusting weights or lowering the threshold.")
