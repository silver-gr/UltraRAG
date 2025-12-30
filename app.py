"""Simple web interface for UltraRAG using Streamlit."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from pathlib import Path
from main import UltraRAG
from config import load_config
from vector_store import index_exists

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

        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
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

        # Security: Validate query input
        if search_button and query:
            # Validate non-empty query after stripping whitespace
            if not query.strip():
                st.error("Please enter a valid query (non-empty).")
            elif len(query) > 10000:
                st.error("Query is too long. Please limit to 10,000 characters.")
            else:
                st.session_state.history.append(query)

                # Show research mode warning
                spinner_message = "Researching knowledge base (this may take 30-60 seconds)..." if research_mode else "Searching knowledge base..."

                with st.spinner(spinner_message):
                    try:
                        if search_type == "Full Answer":
                            # Determine which query method to use
                            if research_mode:
                                # Research mode always uses vault (most comprehensive mode)
                                result = st.session_state.rag.query_research(query)
                            elif search_scope == "📓 Vault Only":
                                result = st.session_state.rag.query(query)
                            elif search_scope == "💬 Conversations":
                                result = st.session_state.rag.query_conversations_only(query)
                            else:  # Both
                                result = st.session_state.rag.query_federated(query)

                            # Display answer
                            st.markdown("### 📝 Answer")
                            st.markdown(result['answer'])

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

                            # Display sources with type indicators
                            st.markdown("### 📚 Sources")
                            for source in result['sources'][:10]:
                                source_type = source.get('source_type', 'vault')
                                type_icon = "📓" if source_type == 'vault' else "💬"
                                with st.expander(
                                    f"**{source['rank']}. {type_icon} {source['title']}** (score: {source['score']:.3f})"
                                ):
                                    st.text(f"File: {source['file']}")
                                    st.text(f"Type: {source_type.title()}")
                                    st.text(source['excerpt'])

                        else:
                            # Just retrieve relevant notes
                            notes = st.session_state.rag.search_notes(query, top_k=20)

                            st.markdown("### 📚 Relevant Notes")
                            for note in notes:
                                source_type = note.get('source_type', 'vault')
                                type_icon = "📓" if source_type == 'vault' else "💬"
                                with st.expander(
                                    f"**{note['rank']}. {type_icon} {note['title']}** (score: {note['score']:.3f})"
                                ):
                                    st.text(f"File: {note['file']}")
                                    st.text(note['excerpt'])

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
