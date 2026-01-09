"""Simple web interface for UltraRAG using Streamlit."""
import os
import re
import time
import json
import uuid
from urllib.parse import quote

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from pathlib import Path
from datetime import datetime
from main import UltraRAG
from config import load_config
from vector_store import index_exists
from temporal_filter import get_all_presets, DateFilterPreset

# Get Obsidian vault name from environment for clickable links
OBSIDIAN_VAULT_NAME = os.getenv("OBSIDIAN_VAULT_NAME", "")

# Auto-load existing index on startup (skip manual button click)
AUTOLOAD_INDEX = os.getenv("AUTOLOAD_INDEX", "true").lower() == "true"

# Persistent query history file
QUERY_HISTORY_FILE = Path("data/query_history.json")

# Cache invalidation file - touch this to force RAG reload
CACHE_INVALIDATION_FILE = Path("data/.cache_invalid")


def get_cache_key():
    """Get cache invalidation key based on file mtime."""
    if CACHE_INVALIDATION_FILE.exists():
        return int(CACHE_INVALIDATION_FILE.stat().st_mtime)
    return 0


@st.cache_resource
def get_cached_rag(_cache_key: int):
    """Load and cache the RAG system to avoid re-initialization on every interaction.

    The cache is invalidated when:
    - The app is restarted
    - data/.cache_invalid file is modified (touch data/.cache_invalid)
    - st.cache_resource.clear() is called

    Args:
        _cache_key: Cache invalidation key (underscore prefix = not hashed, but triggers reload on change)
    """
    try:
        if not Path(".env").exists():
            return None, False

        config = load_config()
        if not index_exists(config.vector_db):
            return None, False

        # Initialize and load
        rag = UltraRAG()
        if rag.load_existing_index():
            return rag, True
        return None, False
    except Exception as e:
        print(f"Cache load failed: {e}")
        return None, False


def load_query_history() -> list:
    """Load persistent query history from disk."""
    if not QUERY_HISTORY_FILE.exists():
        return []
    try:
        with open(QUERY_HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('queries', [])
    except (json.JSONDecodeError, IOError):
        return []


def save_query_to_history(query: str, result: dict) -> None:
    """Append query + result to persistent history."""
    # Load existing history
    history = load_query_history()

    # Create new entry
    entry = {
        'id': f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'answer': result.get('answer', ''),
        'sources': result.get('sources', []),
        'source_summary': result.get('source_summary', {}),
        'research_summary': result.get('research_summary', '')
    }

    # Append and save
    history.append(entry)

    # Ensure data directory exists
    QUERY_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(QUERY_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump({'queries': history}, f, ensure_ascii=False, indent=2)


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


def extract_citation_numbers(text: str) -> set:
    """Extract all citation numbers from text, handling multiple formats.

    Handles:
    - [1] - single number
    - [1, 2, 3] - comma-separated numbers in single brackets
    - [1][2][3] - consecutive single-number brackets

    Returns:
        Set of unique citation numbers found
    """
    numbers = set()

    # Pattern 1: Single number brackets [1], [23]
    for match in re.finditer(r'\[(\d+)\]', text):
        numbers.add(int(match.group(1)))

    # Pattern 2: Comma-separated numbers [1, 2, 3] or [1,2,3]
    for match in re.finditer(r'\[(\d+(?:\s*,\s*\d+)+)\]', text):
        # Extract all numbers from the comma-separated list
        nums_str = match.group(1)
        for num in re.findall(r'\d+', nums_str):
            numbers.add(int(num))

    return numbers


def linkify_citations(text: str) -> str:
    """Convert citation markers to clickable anchor links.

    Handles both [1] and [1, 2, 3] formats.
    """
    # First, handle comma-separated citations [1, 2, 3] -> [1][2][3] with links
    def replace_comma_citation(match):
        nums_str = match.group(1)
        nums = [n.strip() for n in nums_str.split(',')]
        links = [f'<a href="#source-{n}" style="color: #1e90ff; text-decoration: none;">[{n}]</a>' for n in nums]
        return ''.join(links)

    text = re.sub(r'\[(\d+(?:\s*,\s*\d+)+)\]', replace_comma_citation, text)

    # Then handle single citations [1] -> [1] with link
    def replace_single_citation(match):
        num = match.group(1)
        return f'<a href="#source-{num}" style="color: #1e90ff; text-decoration: none;">[{num}]</a>'

    text = re.sub(r'\[(\d+)\]', replace_single_citation, text)

    return text


def strip_citations(text: str) -> str:
    """Remove all citation markers for clean copy.

    Handles both [1] and [1, 2, 3] formats.
    """
    # Remove comma-separated citations [1, 2, 3]
    text = re.sub(r'\[\d+(?:\s*,\s*\d+)+\]', '', text)
    # Remove single citations [1]
    text = re.sub(r'\[\d+\]', '', text)
    # Clean up multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def format_with_wikilink_footnotes(text: str, source_map: dict) -> str:
    """Keep citations inline but add a Sources footer with Obsidian wikilinks.

    Args:
        text: The answer text with citations (handles both [1] and [1, 2, 3] formats)
        source_map: Dict mapping citation number to note title

    Returns:
        Text with citations preserved and a Sources section with [[Note Title]] wikilinks
    """
    # Extract all unique citation numbers from both formats
    used_citations = sorted(extract_citation_numbers(text))

    if not used_citations:
        return text

    # Build the sources footer (only for citations that have mappings)
    sources_lines = ["\n\n---\n**Sources:**"]
    for num in used_citations:
        if num in source_map:
            sources_lines.append(f"[{num}] [[{source_map[num]}]]")

    return text + "\n".join(sources_lines)


def make_obsidian_link(file_path: str, vault_name: str = None) -> str:
    """Generate an Obsidian URI link for a file.

    Args:
        file_path: Path to the file (e.g., "Archive/Notes/My Note.md")
        vault_name: Obsidian vault name (if None, uses OBSIDIAN_VAULT_NAME env var)

    Returns:
        Obsidian URI string (e.g., "obsidian://open?vault=My%20Vault&file=Archive%2FNotes%2FMy%20Note")
        Returns empty string if vault_name is not configured.
    """
    vault = vault_name or OBSIDIAN_VAULT_NAME
    if not vault:
        return ""

    # Remove .md extension if present
    if file_path.endswith('.md'):
        file_path = file_path[:-3]

    # URL encode vault name and file path
    encoded_vault = quote(vault, safe='')
    encoded_file = quote(file_path, safe='')

    return f"obsidian://open?vault={encoded_vault}&file={encoded_file}"


def render_file_link(file_path: str, source_type: str = "vault") -> str:
    """Render file path as clickable Obsidian link if vault name is configured.

    Args:
        file_path: Path to the file
        source_type: Type of source ("vault" or "conversations")

    Returns:
        HTML string with clickable link or plain text
    """
    obsidian_url = make_obsidian_link(file_path) if source_type == "vault" else ""

    if obsidian_url:
        # Clickable link that opens in Obsidian
        return f'📁 <a href="{obsidian_url}" target="_blank" style="color: #1e90ff;">{file_path}</a> • {source_type.title()}'
    else:
        # Plain text fallback
        return f"📁 {file_path} • {source_type.title()}"


def render_copy_buttons(clean_text: str, linked_text: str):
    """Render copy options for clean and linked versions of the answer.

    Uses Streamlit's native st.code() which has a built-in copy button,
    wrapped in expanders for clean UI.
    """
    st.markdown("---")
    st.markdown("**📋 Copy Answer**")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📋 Clean Text (no citations)", expanded=False):
            st.code(clean_text, language=None)

    with col2:
        with st.expander("🔗 With Wikilinks", expanded=False):
            st.code(linked_text, language=None)


# Page config
st.set_page_config(
    page_title="UltraRAG - Obsidian Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# PWA (Progressive Web App) - inject into parent document head
# components.html runs in sandboxed iframe, so we use parent.document to escape
import streamlit.components.v1 as components
components.html("""
    <script>
        try {
            // Access parent document (Streamlit's main window)
            const parentDoc = window.parent.document;
            const head = parentDoc.head || parentDoc.getElementsByTagName('head')[0];

            // Only inject once (check for existing manifest)
            if (!parentDoc.querySelector('link[rel="manifest"]')) {
                // Manifest
                const manifest = parentDoc.createElement('link');
                manifest.rel = 'manifest';
                manifest.href = '/app/static/manifest.json';
                head.appendChild(manifest);

                // Theme color
                const theme = parentDoc.createElement('meta');
                theme.name = 'theme-color';
                theme.content = '#ff4b4b';
                head.appendChild(theme);

                console.log('PWA manifest injected into parent document');
            }

            // Register service worker in parent window context
            // Use inline blob to avoid MIME type issues with Streamlit static serving
            const swCode = `
                self.addEventListener('install', (event) => { self.skipWaiting(); });
                self.addEventListener('activate', (event) => { event.waitUntil(clients.claim()); });
                self.addEventListener('fetch', (event) => { event.respondWith(fetch(event.request)); });
            `;

            if ('serviceWorker' in window.parent.navigator) {
                const blob = new Blob([swCode], { type: 'application/javascript' });
                const swUrl = URL.createObjectURL(blob);
                window.parent.navigator.serviceWorker.register(swUrl, { scope: '/' })
                    .then(reg => console.log('SW registered:', reg.scope))
                    .catch(err => console.log('SW registration failed:', err));
            }
        } catch (e) {
            console.log('PWA injection failed (cross-origin):', e);
        }
    </script>
""", height=0)

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
if 'autoload_attempted' not in st.session_state:
    st.session_state.autoload_attempted = False
if 'loaded_history_result' not in st.session_state:
    st.session_state.loaded_history_result = None
if 'pending_result' not in st.session_state:
    st.session_state.pending_result = None  # Stores result for display after rerun
if 'history_updated' not in st.session_state:
    st.session_state.history_updated = False  # Flag to prevent double rerun

# Auto-load existing index on startup using cache (if AUTOLOAD_INDEX=true)
# Cache persists across reruns - only reloads on app restart or cache invalidation
# Invalidate with: touch data/.cache_invalid
if AUTOLOAD_INDEX and st.session_state.rag is None:
    cache_key = get_cache_key()
    cached_rag, is_indexed = get_cached_rag(cache_key)
    if cached_rag is not None:
        st.session_state.rag = cached_rag
        st.session_state.indexed = is_indexed
        st.session_state.autoload_attempted = True


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

        # System status (compact)
        st.divider()
        if st.session_state.rag:
            config = st.session_state.rag.config
            # Truncate model names for compact display
            emb_short = config.embedding.model.replace('voyage-', '').replace('-lite', 'L')
            llm_short = config.llm.model[:12] + '...' if len(config.llm.model) > 12 else config.llm.model
            st.caption(f"📊 {emb_short} | {config.vector_db.db_type.upper()} | {llm_short}")

            if st.session_state.indexed:
                auto_note = " (auto)" if st.session_state.autoload_attempted else ""
                if st.session_state.conversations_indexed:
                    st.success(f"🟢 Federated ready{auto_note}")
                else:
                    st.success(f"🟢 Vault ready{auto_note}")
            else:
                st.warning("🟡 Not indexed")
        else:
            st.info("Not initialized")

        # Date filter (compact, only show when indexed)
        if st.session_state.indexed:
            preset_options = get_all_presets()
            preset_labels = [label for label, _ in preset_options]
            preset_values = [value for _, value in preset_options]

            current_idx = 0
            if st.session_state.date_filter in preset_values:
                current_idx = preset_values.index(st.session_state.date_filter)

            selected_label = st.selectbox(
                "📅 Date Filter",
                options=preset_labels,
                index=current_idx,
                help="Filter results by date"
            )
            selected_idx = preset_labels.index(selected_label)
            st.session_state.date_filter = preset_values[selected_idx]
        
        # Query history (persistent, clickable)
        st.divider()
        st.subheader("📜 Query History")
        persistent_history = load_query_history()
        if persistent_history:
            # Show most recent 20 queries (reversed for newest first)
            for entry in reversed(persistent_history[-20:]):
                # Parse timestamp for display
                try:
                    ts = datetime.fromisoformat(entry['timestamp'])
                    time_str = ts.strftime("%b %d, %I:%M %p")
                except (KeyError, ValueError):
                    time_str = "Unknown"

                # Truncate query for button label
                query_short = entry['query'][:40] + '...' if len(entry['query']) > 40 else entry['query']
                button_label = f"[{time_str}] {query_short}"

                if st.button(button_label, key=f"hist_{entry.get('id', entry['timestamp'])}", use_container_width=True):
                    st.session_state.loaded_history_result = entry
                    st.rerun()
        else:
            st.caption("No queries yet")
    
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

        # Display pending result (from rerun after save)
        if st.session_state.pending_result and not search_button:
            pending = st.session_state.pending_result
            result = pending['result']
            exec_time = pending['exec_time']
            search_scope = pending.get('search_scope', '📓 Vault Only')
            research_mode = pending.get('research_mode', False)

            # Clear pending result after retrieving
            st.session_state.pending_result = None

            # Display execution summary
            total_sources = len(result['sources'])
            word_count = len(result['answer'].split())
            st.markdown(
                f"**Synthesized from {total_sources} sources** "
                f"→ **{word_count:,} words** in **{exec_time:.1f}s** "
                f"• *saved to history*"
            )

            # Display answer with clickable citation links
            st.markdown("### 📝 Answer")
            answer_with_links = linkify_citations(result['answer'])
            st.markdown(answer_with_links, unsafe_allow_html=True)

            # Build source map for wikilink replacement
            source_map = {
                source['rank']: source['title']
                for source in result['sources']
            }

            # Generate copy versions
            clean_text = strip_citations(result['answer'])
            linked_text = format_with_wikilink_footnotes(result['answer'], source_map)
            render_copy_buttons(clean_text, linked_text)

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

            # Display all sources used in synthesis
            st.markdown(f"### 📚 Sources ({len(result['sources'])})")
            for source in result['sources']:
                source_type = source.get('source_type', 'vault')
                type_icon = "📓" if source_type == 'vault' else "💬"
                # Anchor for clickable citation
                st.markdown(f'<div id="source-{source["rank"]}"></div>', unsafe_allow_html=True)
                with st.expander(
                    f"**{source['rank']}. {type_icon} {source['title']}** (score: {source.get('score', 0):.3f})"
                ):
                    file_link_html = render_file_link(source['file'], source_type)
                    st.markdown(f'<small style="color: gray;">{file_link_html}</small>', unsafe_allow_html=True)
                    cleaned = clean_excerpt_for_display(source.get('excerpt', ''))
                    st.markdown(cleaned)

        # Display loaded history result (when user clicks a history item)
        elif st.session_state.loaded_history_result and not search_button:
            entry = st.session_state.loaded_history_result

            # Parse timestamp for display
            try:
                ts = datetime.fromisoformat(entry['timestamp'])
                time_str = ts.strftime("%B %d, %Y at %I:%M %p")
            except (KeyError, ValueError):
                time_str = "Unknown time"

            # Clear button
            if st.button("✖ Clear History View", type="secondary"):
                st.session_state.loaded_history_result = None
                st.rerun()

            st.markdown(f"*Query from {time_str}:*")
            st.markdown(f"**{entry['query']}**")

            if entry.get('answer'):
                total_sources = len(entry.get('sources', []))
                word_count = len(entry['answer'].split())
                st.markdown(f"**{total_sources} sources** → **{word_count:,} words**")

                # Display answer with clickable citations
                st.markdown("### 📝 Answer")
                answer_with_links = linkify_citations(entry['answer'])
                st.markdown(answer_with_links, unsafe_allow_html=True)

                # Build source map for wikilinks
                source_map = {
                    source['rank']: source['title']
                    for source in entry.get('sources', [])
                }

                # Generate copy versions
                clean_text = strip_citations(entry['answer'])
                linked_text = format_with_wikilink_footnotes(entry['answer'], source_map)

                # Render copy buttons
                render_copy_buttons(clean_text, linked_text)

                # Display sources
                sources = entry.get('sources', [])
                if sources:
                    st.markdown(f"### 📚 Sources ({len(sources)})")
                    for source in sources:
                        source_type = source.get('source_type', 'vault')
                        type_icon = "📓" if source_type == 'vault' else "💬"
                        st.markdown(f'<div id="source-{source["rank"]}"></div>', unsafe_allow_html=True)
                        with st.expander(
                            f"**{source['rank']}. {type_icon} {source['title']}** (score: {source.get('score', 0):.3f})"
                        ):
                            file_link_html = render_file_link(source['file'], source_type)
                            st.markdown(f'<small style="color: gray;">{file_link_html}</small>', unsafe_allow_html=True)
                            cleaned = clean_excerpt_for_display(source.get('excerpt', ''))
                            st.markdown(cleaned)
            else:
                st.info("No answer stored for this query.")

        # Security: Validate query input
        elif search_button and query:
            # Clear history view when starting new search
            st.session_state.loaded_history_result = None

            # Validate non-empty query after stripping whitespace
            if not query.strip():
                st.error("Please enter a valid query (non-empty).")
            elif len(query) > 10000:
                st.error("Query is too long. Please limit to 10,000 characters.")
            else:

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
                            # Note: We pass max_sources=None to get ALL sources for proper citation matching
                            # The dropdown controls synthesis depth via retrieval config, not display limiting
                            if raptor_mode:
                                # RAPTOR mode uses hierarchical summaries
                                result = st.session_state.rag.query_raptor(query, max_sources=None)
                            elif research_mode:
                                # Research mode uses ALL retrieved sources for synthesis
                                result = st.session_state.rag.query_research(query, max_sources=None, date_filter=date_filter)
                            elif search_scope == "📓 Vault Only":
                                result = st.session_state.rag.query(query, max_sources=None, date_filter=date_filter)
                            elif search_scope == "💬 Conversations":
                                result = st.session_state.rag.query_conversations_only(query, max_sources=None, date_filter=date_filter)
                            else:  # Both
                                result = st.session_state.rag.query_federated(query, max_sources=None, date_filter=date_filter)

                            # Calculate execution time and stats
                            exec_time = time.time() - start_time
                            total_sources = len(result['sources'])
                            word_count = len(result['answer'].split())

                            # Save to persistent history
                            save_query_to_history(query, result)

                            # Store result for potential rerun and trigger sidebar update
                            if not st.session_state.history_updated:
                                st.session_state.pending_result = {
                                    'result': result,
                                    'query': query,
                                    'exec_time': exec_time,
                                    'search_scope': search_scope,
                                    'research_mode': research_mode
                                }
                                st.session_state.history_updated = True
                                st.rerun()  # Rerun to update sidebar with new history

                            # Reset flag for next query
                            st.session_state.history_updated = False

                            # Display execution summary with history confirmation
                            st.markdown(
                                f"**Synthesized from {total_sources} sources** "
                                f"→ **{word_count:,} words** in **{exec_time:.1f}s** "
                                f"• *saved to history*"
                            )

                            # Display answer with clickable citation links
                            st.markdown("### 📝 Answer")
                            answer_with_links = linkify_citations(result['answer'])
                            st.markdown(answer_with_links, unsafe_allow_html=True)

                            # Build source map for wikilink replacement (rank -> title)
                            # Include ALL sources since LLM may cite beyond displayed count
                            source_map = {
                                source['rank']: source['title']
                                for source in result['sources']
                            }

                            # Generate copy versions
                            clean_text = strip_citations(result['answer'])
                            linked_text = format_with_wikilink_footnotes(result['answer'], source_map)

                            # Render copy buttons
                            render_copy_buttons(clean_text, linked_text)

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

                            # Display all sources used in synthesis
                            st.markdown(f"### 📚 Sources ({len(result['sources'])})")
                            for source in result['sources']:
                                source_type = source.get('source_type', 'vault')
                                type_icon = "📓" if source_type == 'vault' else "💬"
                                # Add anchor ID for citation linking
                                st.markdown(f'<div id="source-{source["rank"]}"></div>', unsafe_allow_html=True)
                                with st.expander(
                                    f"**{source['rank']}. {type_icon} {source['title']}** (score: {source['score']:.3f})"
                                ):
                                    # Render file path as clickable Obsidian link
                                    file_link_html = render_file_link(source['file'], source_type)
                                    st.markdown(f'<small style="color: gray;">{file_link_html}</small>', unsafe_allow_html=True)
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
                                    # Render file path as clickable Obsidian link
                                    file_link_html = render_file_link(note['file'], source_type)
                                    st.markdown(f'<small style="color: gray;">{file_link_html}</small>', unsafe_allow_html=True)
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
