"""Content Research — multi-source federated search with per-user storage."""

from __future__ import annotations

import streamlit as st

from auth import require_auth, logout
from content_research_engine import (
    ResearchResult,
    weighted_federated_search,
    deduplicate_results,
)
from rate_limit import run_with_limits, QueryCooldownError, SystemBusyError
from research_storage import (
    save_results,
    load_results,
    get_history,
    clear_history,
    export_markdown,
    export_json,
    get_exports_dir,
)
from ui_theme import inject_theme

st.set_page_config(
    page_title="Content Research",
    page_icon="/app/static/favicon-32.png",
    layout="wide",
)
inject_theme()
ctx = require_auth()

st.title("Content Research")
st.caption("Federated search across Vault, AI Conversations, and The Source")

with st.sidebar:
    st.caption(f"Logged in as: **{ctx.username}** ({ctx.role})")
    if st.button("Logout", use_container_width=True):
        logout()
        st.rerun()

    st.divider()
    st.subheader("Source Configuration")

    vault_enabled = st.checkbox("Vault", value=True)
    vault_weight = st.slider("Vault Weight", 0.0, 2.0, 1.0, 0.1, disabled=not vault_enabled)

    conv_enabled = st.checkbox("Conversations", value=True)
    conv_weight = st.slider("Conversations Weight", 0.0, 2.0, 0.7, 0.1, disabled=not conv_enabled)

    source_enabled = st.checkbox("The Source", value=True)
    source_weight = st.slider("The Source Weight", 0.0, 2.0, 0.5, 0.1, disabled=not source_enabled)

    st.divider()
    st.subheader("Retrieval Settings")
    top_k = st.number_input("Top-K per source", min_value=5, max_value=100, value=30, step=5)
    threshold = st.slider("Score Threshold", 0.0, 1.0, 0.4, 0.05)
    dedup_sim = st.slider("Dedup Similarity", 0.5, 1.0, 0.85, 0.05)

    st.divider()
    st.subheader("Query History")
    history = get_history(ctx.username)

    if history:
        for entry in history[:20]:
            label = entry["query"][:30]
            if len(entry["query"]) > 30:
                label += "..."
            sublabel = f"{entry['timestamp'][:10]} | {entry['stats']['total_results']} results"
            if st.button(label, key=f"hist_{entry['id']}", help=sublabel):
                stored = load_results(entry["id"], username=ctx.username)
                if stored:
                    st.session_state.results = [
                        ResearchResult(**{k: v for k, v in result.items() if k != "embedding"})
                        for result in stored["results"]
                    ]
                    st.session_state.dedup_log = stored.get("dedup_log", [])
                    st.session_state.query_settings = stored["settings"]
                    st.session_state.result_id = entry["id"]
                    st.rerun()

        if st.button("Clear History", type="secondary"):
            clear_history(ctx.username)
            st.rerun()
    else:
        st.caption("No queries yet.")

query = st.text_input("Topic", placeholder="e.g. ADHD tips and tricks")

if st.button("Search", type="primary") and query:
    weights: dict[str, float] = {}
    if vault_enabled:
        weights["vault"] = vault_weight
    if conv_enabled:
        weights["conversations"] = conv_weight
    if source_enabled:
        weights["saved_items"] = source_weight

    if not weights:
        st.error("Enable at least one source.")
    else:
        try:
            with run_with_limits(ctx, "content_research_search"):
                with st.spinner("Searching across sources..."):
                    results = weighted_federated_search(
                        query=query,
                        weights=weights,
                        top_k=top_k,
                        threshold=threshold,
                    )

                with st.spinner("Deduplicating..."):
                    results, dedup_log = deduplicate_results(results, dedup_sim)

                query_settings = {
                    "query": query,
                    "weights": weights,
                    "top_k": top_k,
                    "threshold": threshold,
                    "dedup_similarity": dedup_sim,
                }

                result_id = save_results(query_settings, results, dedup_log, username=ctx.username)

            st.session_state.results = results
            st.session_state.dedup_log = dedup_log
            st.session_state.query_settings = query_settings
            st.session_state.result_id = result_id
            st.rerun()
        except QueryCooldownError as exc:
            st.warning(str(exc))
        except SystemBusyError as exc:
            st.error(str(exc))

if "results" in st.session_state and st.session_state.results:
    results = st.session_state.results
    dedup_log = st.session_state.get("dedup_log", [])
    query_settings = st.session_state.get("query_settings", {})

    sources_present = sorted(set(r.source if hasattr(r, "source") else r.get("source", "?") for r in results))
    source_counts: dict[str, int] = {}
    for result in results:
        source = result.source if hasattr(result, "source") else result.get("source", "?")
        source_counts[source] = source_counts.get(source, 0) + 1

    cols = st.columns([2, 1, 1, 1])
    with cols[0]:
        st.markdown(f"**{len(results)} results** for: _{query_settings.get('query', '')}_")
    with cols[1]:
        if dedup_log:
            st.caption(f"{len(dedup_log)} duplicates removed")

    exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
    with exp_col1:
        if st.button("Export Markdown"):
            try:
                with run_with_limits(ctx, "export_markdown"):
                    path = export_markdown(st.session_state.result_id, username=ctx.username)
                if path:
                    st.success(f"Saved: {path.name}")
                else:
                    st.error("Export failed")
            except QueryCooldownError as exc:
                st.warning(str(exc))
            except SystemBusyError as exc:
                st.error(str(exc))
    with exp_col2:
        if st.button("Export JSON"):
            try:
                with run_with_limits(ctx, "export_json"):
                    path = export_json(st.session_state.result_id, username=ctx.username)
                if path:
                    st.success(f"Saved: {path.name}")
                else:
                    st.error("Export failed")
            except QueryCooldownError as exc:
                st.warning(str(exc))
            except SystemBusyError as exc:
                st.error(str(exc))
    with exp_col3:
        st.caption(f"Export dir: `{get_exports_dir(ctx.username)}`")

    st.divider()

    source_labels = {
        "vault": "Vault",
        "conversations": "Conversations",
        "saved_items": "The Source",
    }
    tab_names = ["All"] + [f"{source_labels.get(source, source)} ({source_counts.get(source, 0)})" for source in sources_present]
    tabs = st.tabs(tab_names)

    def _display_result(result_obj):
        if hasattr(result_obj, "source"):
            title = result_obj.title
            score = result_obj.weighted_score
            raw = result_obj.raw_score
            source = result_obj.source
            chunk = result_obj.chunk
            metadata = result_obj.metadata
        else:
            title = result_obj.get("title", "Untitled")
            score = result_obj.get("weighted_score", 0)
            raw = result_obj.get("raw_score", 0)
            source = result_obj.get("source", "?")
            chunk = result_obj.get("chunk", "")
            metadata = result_obj.get("metadata", {})

        source_badge = source_labels.get(source, source)
        with st.expander(f"[{score:.3f}] {title}  `{source_badge}`"):
            display_chunk = chunk[:800]
            if len(chunk) > 800:
                display_chunk += "..."
            st.markdown(f"> {display_chunk}")
            st.caption(f"Source: {source_badge} | Raw: {raw:.3f} | Weighted: {score:.3f}")

            if metadata:
                meta_parts = []
                for key, value in metadata.items():
                    if value is None:
                        continue
                    import numpy as np

                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    elif isinstance(value, (np.integer,)):
                        value = int(value)
                    elif isinstance(value, (np.floating,)):
                        value = float(value)

                    if isinstance(value, (list, tuple)):
                        if len(value) == 0:
                            continue
                        value = ", ".join(str(x) for x in list(value)[:5])
                    elif isinstance(value, str) and value == "":
                        continue

                    meta_parts.append(f"**{key}:** {value}")

                if meta_parts:
                    st.markdown(" | ".join(meta_parts))

    with tabs[0]:
        for result in results:
            _display_result(result)

    for idx, source in enumerate(sources_present):
        with tabs[idx + 1]:
            source_results = [
                result
                for result in results
                if (result.source if hasattr(result, "source") else result.get("source")) == source
            ]
            for result in source_results:
                _display_result(result)

    if dedup_log:
        with st.expander(f"Dedup Details ({len(dedup_log)} removed)"):
            for entry in dedup_log:
                st.caption(
                    f"Removed: \"{entry.get('removed_title', '?')}\" "
                    f"({entry.get('removed_source', '?')}) - "
                    f"Similar to: \"{entry.get('kept_title', '?')}\" "
                    f"({entry.get('kept_source', '?')}) - "
                    f"Similarity: {entry.get('similarity', 0):.3f}"
                )

elif "results" in st.session_state and not st.session_state.results:
    st.info("No results found. Try adjusting weights or lowering the threshold.")
