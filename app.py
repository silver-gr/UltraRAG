"""Simple web interface for UltraRAG using Streamlit."""
import os
import re
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from pathlib import Path
from datetime import datetime
from main import UltraRAG
from config import load_config
from vector_store import index_exists
from temporal_filter import get_all_presets, DateFilterPreset


def clean_excerpt_for_display(text: str, max_chars: int = 500) -> str:
    """Clean and truncate excerpt for display.

    - Removes markdown headings (# ## ###)
    - Truncates at word boundary
    - Preserves other markdown formatting (bold, italic, lists)
    """
    # Remove markdown headings (lines starting with #)
    text = re.sub(r'^#{1,6}\s+.*$', '', text, flags=re.MULTILINE)

    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    # Truncate at word boundary if too long
    if len(text) > max_chars:
        # Find the last space before max_chars
        truncate_at = text.rfind(' ', 0, max_chars)
        if truncate_at == -1:
            truncate_at = max_chars
        text = text[:truncate_at] + "..."

    return text


def linkify_citations(text: str) -> str:
    """Convert [1], [2], etc. citations to clickable anchor links.

    Converts patterns like [1], [12], [1][2][3] to HTML links that scroll
    to the corresponding source section.
    """
    # Pattern matches [N] where N is one or more digits
    def replace_citation(match):
        num = match.group(1)
        return f'<a href="#source-{num}" style="color: #1e90ff; text-decoration: none;">[{num}]</a>'

    return re.sub(r'\[(\d+)\]', replace_citation, text)


# Page config
st.set_page_config(
    page_title="UltraRAG - Obsidian Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = None
if 'indexed' not in st.session_state:
    st.session_state.indexed = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'conversations_indexed' not in st.session_state:
    st.session_state.conversations_indexed = False
if 'date_filter' not in st.session_state:
    st.session_state.date_filter = "all_time"


def main():
    st.title("🧠 UltraRAG - Obsidian Knowledge Assistant")
    st.markdown("World-class RAG system for your personal knowledge base")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check if .env exists
        if not Path(".env").exists():
            st.error("⚠️ .env file not found!")
            st.info("Copy .env.example to .env and configure your settings.")
            return
        
        # Check for existing index
        try:
            config = load_config()
            has_existing_index = index_exists(config.vector_db)
        except Exception:
            has_existing_index = False

        # Show appropriate buttons based on state
        if not st.session_state.rag:
            if has_existing_index:
                st.success("📦 Existing index found!")
                if st.button("🚀 Load Existing Index", type="primary"):
                    with st.spinner("Loading RAG system and existing index..."):
                        try:
                            st.session_state.rag = UltraRAG()
                            if st.session_state.rag.load_existing_index():
                                st.session_state.indexed = True
                                st.success("✅ Index loaded!")
                                st.rerun()
                            else:
                                st.error("Failed to load index")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

                if st.button("🔄 Create New Index"):
                    with st.spinner("Initializing RAG system..."):
                        try:
                            st.session_state.rag = UltraRAG()
                            st.success("✅ System initialized!")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
            else:
                if st.button("🚀 Initialize System", type="primary"):
                    with st.spinner("Initializing RAG system..."):
                        try:
                            st.session_state.rag = UltraRAG()
                            st.success("✅ System initialized!")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

        # Index button (only show if initialized but not indexed)
        if st.session_state.rag and not st.session_state.indexed:
            if st.button("📚 Index Vault"):
                with st.spinner("Indexing vault (this may take several minutes)..."):
                    try:
                        st.session_state.rag.index_vault()
                        st.session_state.indexed = True
                        st.success("✅ Vault indexed!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        # Conversations section
        if st.session_state.rag and st.session_state.indexed:
            st.divider()
            st.subheader("💬 AI Conversations")

            # Check for conversations config
            config = st.session_state.rag.config
            has_conv_path = config.conversations.path and config.conversations.path.exists()
            has_conv_index = st.session_state.rag.conversations_index_exists()

            if has_conv_index or st.session_state.conversations_indexed:
                st.success("🟢 Conversations indexed")
                st.session_state.conversations_indexed = True

                # Load if not already
                if st.session_state.rag.conversations_index is None:
                    st.session_state.rag.load_conversations_index()
                    st.session_state.rag._setup_federated_engine()

            elif has_conv_path:
                st.info(f"📁 Found: {config.conversations.path}")
                if st.button("📚 Index Conversations"):
                    with st.spinner("Indexing AI conversations..."):
                        try:
                            st.session_state.rag.index_conversations(force_reindex=False, interactive=False)
                            st.session_state.conversations_indexed = True
                            st.success("✅ Conversations indexed!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
            else:
                st.info("Set CONVERSATIONS_PATH in .env")

        # RAPTOR section
        if st.session_state.rag and st.session_state.indexed:
            st.divider()
            st.subheader("🌳 RAPTOR Summaries")

            config = st.session_state.rag.config
            has_raptor = st.session_state.rag.raptor_index_exists()

            if has_raptor:
                st.success("🟢 RAPTOR index ready")
                stats = st.session_state.rag.get_raptor_stats()
                st.caption(f"Mode: {stats.get('default_mode', 'collapsed')} | Nodes: {stats.get('node_count', 'unknown')}")

                # Load if not already
                if st.session_state.rag.raptor_manager is None:
                    st.session_state.rag.load_raptor_index()

            elif config.raptor.enabled:
                st.info("RAPTOR enabled but not indexed")
                if st.button("🌳 Build RAPTOR Index"):
                    with st.spinner("Building RAPTOR hierarchical summaries (this may take several minutes)..."):
                        try:
                            st.session_state.rag.index_raptor(force_reindex=False, interactive=False)
                            st.success("✅ RAPTOR index built!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
            else:
                st.info("Set ENABLE_RAPTOR=true in .env")

        # System status
        st.divider()
        st.subheader("📊 System Status")

        if st.session_state.rag:
            config = st.session_state.rag.config
            st.metric("Embedding Model", config.embedding.model)
            st.metric("Vector DB", config.vector_db.db_type.upper())
            st.metric("LLM", config.llm.model)

            if st.session_state.indexed:
                if st.session_state.conversations_indexed:
                    st.success("🟢 Federated search ready")
                else:
                    st.success("🟢 Vault ready")
            else:
                st.warning("🟡 Not indexed yet")
        else:
            st.info("System not initialized")

        # Date filter section (only show when indexed)
        if st.session_state.indexed:
            st.divider()
            st.subheader("📅 Date Filter")

            # Get preset options
            preset_options = get_all_presets()
            preset_labels = [label for label, _ in preset_options]
            preset_values = [value for _, value in preset_options]

            # Find current index
            current_idx = 0
            if st.session_state.date_filter in preset_values:
                current_idx = preset_values.index(st.session_state.date_filter)

            selected_label = st.selectbox(
                "Filter by date:",
                options=preset_labels,
                index=current_idx,
                help="Filter search results to documents created/modified within the selected time period"
            )

            # Update session state with selected preset
            selected_idx = preset_labels.index(selected_label)
            st.session_state.date_filter = preset_values[selected_idx]

            # Show active filter info
            if st.session_state.date_filter != "all_time":
                st.info(f"🕐 Showing: {selected_label}")
        
        # Query history
        if st.session_state.history:
            st.divider()
            st.subheader("📜 Recent Queries")
            for i, query in enumerate(reversed(st.session_state.history[-5:]), 1):
                st.text(f"{i}. {query[:50]}...")
    
    # Main content area
    if not st.session_state.indexed:
        st.info("👈 Initialize the system and index your vault to get started")
        
        # Show features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 Smart Retrieval")
            st.write("- Semantic search")
            st.write("- Wikilink traversal")
            st.write("- Temporal filtering")
        
        with col2:
            st.subheader("🧠 Advanced AI")
            st.write("- Gemini 3 Flash")
            st.write("- Thinking mode")
            st.write("- Context-aware")
        
        with col3:
            st.subheader("📊 High Quality")
            st.write("- Voyage embeddings")
            st.write("- Smart reranking")
            st.write("- Source citations")
    
    else:
        # Query interface
        st.subheader("💭 Ask a Question")

        query = st.text_input(
            "What would you like to know from your vault?",
            placeholder="e.g., What are my notes about machine learning?",
            key="query_input",
            max_chars=10000  # Security: Limit query length
        )

        col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 2, 1, 1, 1])
        with col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            search_type = st.radio(
                "Search type:",
                ["Full Answer", "Find Notes Only"],
                horizontal=True,
                label_visibility="collapsed"
            )
        with col3:
            # Search scope (only show if conversations indexed)
            if st.session_state.conversations_indexed:
                search_scope = st.radio(
                    "Search scope:",
                    ["📓 Vault Only", "💬 Conversations", "🔀 Both"],
                    horizontal=True,
                    label_visibility="collapsed",
                    index=2  # Default to "Both"
                )
            else:
                search_scope = "📓 Vault Only"
        with col4:
            # Research mode toggle
            research_mode = st.checkbox(
                "🔬 Research",
                help="Enable multi-step iterative retrieval (3-5x slower, higher accuracy)",
                value=False
            )
        with col5:
            # RAPTOR mode toggle (only show if RAPTOR index exists)
            has_raptor = st.session_state.rag.raptor_index_exists() if st.session_state.rag else False
            if has_raptor:
                raptor_mode = st.checkbox(
                    "🌳 RAPTOR",
                    help="Use hierarchical summaries for better multi-document reasoning",
                    value=False
                )
            else:
                raptor_mode = False
        with col6:
            # Max sources dropdown
            max_sources_options = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]
            max_sources = st.selectbox(
                "Max sources",
                options=max_sources_options,
                index=0,  # Default to 10
                help="Maximum number of sources to use for synthesis"
            )

        # Security: Validate query input
        if search_button and query:
            # Validate non-empty query after stripping whitespace
            if not query.strip():
                st.error("Please enter a valid query (non-empty).")
            elif len(query) > 10000:
                st.error("Query is too long. Please limit to 10,000 characters.")
            else:
                st.session_state.history.append(query)

                # Show appropriate spinner message
                if raptor_mode:
                    spinner_message = "Searching RAPTOR hierarchical summaries..."
                elif research_mode:
                    spinner_message = "Researching knowledge base (this may take 30-60 seconds)..."
                else:
                    spinner_message = "Searching knowledge base..."

                start_time = time.time()

                with st.spinner(spinner_message):
                    try:
                        # Get current date filter from session state
                        date_filter = st.session_state.date_filter

                        if search_type == "Full Answer":
                            # Determine which query method to use
                            if raptor_mode:
                                # RAPTOR mode uses hierarchical summaries
                                result = st.session_state.rag.query_raptor(query, max_sources=max_sources)
                            elif research_mode:
                                # Research mode always uses vault (most comprehensive mode)
                                result = st.session_state.rag.query_research(query, max_sources=max_sources, date_filter=date_filter)
                            elif search_scope == "📓 Vault Only":
                                result = st.session_state.rag.query(query, max_sources=max_sources, date_filter=date_filter)
                            elif search_scope == "💬 Conversations":
                                result = st.session_state.rag.query_conversations_only(query, max_sources=max_sources, date_filter=date_filter)
                            else:  # Both
                                result = st.session_state.rag.query_federated(query, max_sources=max_sources, date_filter=date_filter)

                            # Calculate execution time and stats
                            exec_time = time.time() - start_time
                            total_sources = result.get('total_sources', len(result['sources']))
                            displayed_sources = min(max_sources, total_sources)
                            word_count = len(result['answer'].split())

                            # Display execution summary
                            st.markdown(
                                f"**Retrieved {total_sources} notes** and used **{displayed_sources} sources** "
                                f"for **{word_count:,} words** in **{exec_time:.1f} seconds**"
                            )

                            # Display answer with clickable citation links
                            st.markdown("### 📝 Answer")
                            answer_with_links = linkify_citations(result['answer'])
                            st.markdown(answer_with_links, unsafe_allow_html=True)

                            # Show research summary for research mode
                            if research_mode and 'research_summary' in result:
                                with st.expander("🔬 Research Details", expanded=False):
                                    st.text(result['research_summary'])

                            # Show source summary for federated queries
                            if search_scope == "🔀 Both" and 'source_summary' in result:
                                summary = result.get('source_summary', {})
                                if summary:
                                    by_type = summary.get('by_type', {})
                                    vault_count = by_type.get('vault', 0)
                                    conv_count = by_type.get('conversations', 0)
                                    st.info(f"📊 Sources: {vault_count} from vault, {conv_count} from conversations")

                            # Display sources with type indicators (use max_sources setting)
                            total_sources = result.get('total_sources', len(result['sources']))
                            st.markdown(f"### 📚 Sources ({min(max_sources, total_sources)} of {total_sources})")
                            for source in result['sources'][:max_sources]:
                                source_type = source.get('source_type', 'vault')
                                type_icon = "📓" if source_type == 'vault' else "💬"
                                # Add anchor ID for citation linking
                                st.markdown(f'<div id="source-{source["rank"]}"></div>', unsafe_allow_html=True)
                                with st.expander(
                                    f"**{source['rank']}. {type_icon} {source['title']}** (score: {source['score']:.3f})"
                                ):
                                    st.caption(f"📁 {source['file']} • {source_type.title()}")
                                    # Render markdown with cleaned excerpt
                                    cleaned = clean_excerpt_for_display(source['excerpt'])
                                    st.markdown(cleaned)

                        else:
                            # Just retrieve relevant notes
                            notes = st.session_state.rag.search_notes(query, top_k=max_sources, date_filter=date_filter)

                            st.markdown(f"### 📚 Relevant Notes ({len(notes)} found)")
                            for note in notes:
                                source_type = note.get('source_type', 'vault')
                                type_icon = "📓" if source_type == 'vault' else "💬"
                                with st.expander(
                                    f"**{note['rank']}. {type_icon} {note['title']}** (score: {note['score']:.3f})"
                                ):
                                    st.caption(f"📁 {note['file']} • {source_type.title()}")
                                    cleaned = clean_excerpt_for_display(note['excerpt'])
                                    st.markdown(cleaned)

                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        # Example queries
        with st.expander("💡 Example Queries"):
            st.markdown("""
            - What are my thoughts on [topic]?
            - Show me all notes related to [project]
            - What connections exist between [concept A] and [concept B]?
            - Summarize my notes tagged with #important
            - What did I write about [topic] in the last month?
            """)


if __name__ == "__main__":
    main()
