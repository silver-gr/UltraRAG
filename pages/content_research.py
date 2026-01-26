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

# Premium UI Design System (shared with main app)
st.markdown("""
<style>
/* ============================================
   ULTRARAG PREMIUM DESIGN SYSTEM
   Glass morphism + Modern dark theme
   ============================================ */

/* === ROOT VARIABLES === */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --accent-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
    --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    --text-primary: #ffffff;
    --text-secondary: rgba(255, 255, 255, 0.7);
    --text-muted: rgba(255, 255, 255, 0.5);
    --border-radius: 12px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* === GLOBAL STYLES === */
.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
}

/* === MAIN CONTENT AREA === */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* === TYPOGRAPHY === */
.main h1 {
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
}

h2, h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.stMarkdown p {
    color: var(--text-secondary) !important;
}

/* === SIDEBAR STYLING === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 15, 35, 0.95) 0%, rgba(26, 26, 46, 0.95) 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem !important;
}

.stSidebar h1, .stSidebar h2, .stSidebar h3 {
    color: #a78bfa !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.5px;
    font-weight: 600 !important;
}

.stSidebar hr {
    border-color: var(--glass-border) !important;
    margin: 1rem 0 !important;
}

/* === GLASS CARDS === */
[data-testid="stExpander"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--border-radius) !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: var(--glass-shadow) !important;
    transition: var(--transition) !important;
}

[data-testid="stExpander"]:hover {
    border-color: rgba(102, 126, 234, 0.4) !important;
}

/* === BUTTONS === */
button[kind="primary"] {
    background: var(--primary-gradient) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    min-height: 2.5rem !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
}

button[kind="secondary"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    backdrop-filter: blur(10px) !important;
    transition: var(--transition) !important;
}

button[kind="secondary"]:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(102, 126, 234, 0.5) !important;
}

/* === INPUT FIELDS === */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    backdrop-filter: blur(10px) !important;
    transition: var(--transition) !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: var(--text-muted) !important;
}

/* === SLIDERS === */
[data-testid="stSlider"] > div > div > div {
    background: var(--primary-gradient) !important;
}

/* === CHECKBOXES === */
div[data-testid="stCheckbox"] label {
    color: var(--text-secondary) !important;
}

/* === NUMBER INPUT === */
[data-testid="stNumberInput"] input {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* === ALERTS === */
[data-testid="stAlert"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--border-radius) !important;
}

/* === METRICS === */
[data-testid="stMetric"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--border-radius) !important;
    padding: 1rem !important;
}

[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
}

/* === TABS === */
[data-testid="stTabs"] button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-secondary) !important;
    transition: var(--transition) !important;
}

[data-testid="stTabs"] button:hover {
    color: var(--text-primary) !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom-color: #667eea !important;
}

/* === SCROLLBAR === */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(102, 126, 234, 0.5);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(102, 126, 234, 0.7);
}

/* === LINKS === */
a {
    color: #667eea !important;
    text-decoration: none !important;
    transition: var(--transition) !important;
}

a:hover {
    color: #764ba2 !important;
}

/* === DIVIDERS === */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--glass-border) !important;
    margin: 1.5rem 0 !important;
}

/* === DATAFRAMES === */
[data-testid="stDataFrame"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--border-radius) !important;
}

/* === ANIMATIONS === */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

[data-testid="stExpander"],
[data-testid="stAlert"],
[data-testid="stMetric"] {
    animation: fadeIn 0.3s ease-out;
}

/* === SIDEBAR LOGO GLOW === */
.stSidebar img {
    filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.3));
    transition: var(--transition);
}

.stSidebar img:hover {
    filter: drop-shadow(0 0 30px rgba(102, 126, 234, 0.5));
}
</style>
""", unsafe_allow_html=True)

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
    sources_present = sorted(set(r.source if hasattr(r, 'source')
                                 else r.get("source", "?") for r in results))
    source_counts = {}
    for r in results:
        src = r.source if hasattr(r, 'source') else r.get("source", "?")
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
        if hasattr(r, 'source'):
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
                if (r.source if hasattr(r, 'source')
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
