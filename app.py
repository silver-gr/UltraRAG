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
from vector_store import index_exists, delete_from_index
from temporal_filter import get_all_presets, DateFilterPreset
from settings_store import get_exclusions, add_exclusion, remove_exclusion
from exclusion_matcher import ExclusionMatcher, preview_exclusions
from llm_token_tracker import get_llm_tracker

# Get Obsidian vault name from environment for clickable links
OBSIDIAN_VAULT_NAME = os.getenv("OBSIDIAN_VAULT_NAME", "")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format (e.g., '2m 47s' or '45s')."""
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        return f"{seconds:.1f}s"


def extract_cited_numbers(text: str) -> set[int]:
    """Extract all citation numbers from text (e.g., [1], [23], [145])."""
    # Match [N] patterns where N is a number
    matches = re.findall(r'\[(\d+)\]', text)
    return set(int(m) for m in matches)


def filter_and_renumber_citations(answer: str, sources: list) -> tuple[str, list]:
    """Filter sources to only cited ones and renumber citations sequentially.

    Args:
        answer: The answer text with citations like [1], [7], [263]
        sources: Full list of sources with 'rank' field

    Returns:
        Tuple of (remapped_answer, filtered_sources) where citations are 1-N
    """
    # Extract all cited source numbers from the answer
    cited_numbers = extract_cited_numbers(answer)

    if not cited_numbers:
        return answer, sources

    # Create mapping from old rank to new sequential number
    # Sort cited numbers to maintain order
    sorted_cited = sorted(cited_numbers)
    old_to_new = {old: new for new, old in enumerate(sorted_cited, 1)}

    # Remap citations in the answer text
    # Must replace largest numbers first to avoid [1] replacing part of [12]
    remapped_answer = answer
    for old_num in sorted(cited_numbers, reverse=True):
        new_num = old_to_new[old_num]
        remapped_answer = remapped_answer.replace(f'[{old_num}]', f'[§{new_num}§]')

    # Replace temporary markers with final numbers
    remapped_answer = re.sub(r'\[§(\d+)§\]', r'[\1]', remapped_answer)

    # Filter and renumber sources
    # Create a lookup by rank
    sources_by_rank = {s['rank']: s for s in sources}

    filtered_sources = []
    for old_rank in sorted_cited:
        if old_rank in sources_by_rank:
            source = sources_by_rank[old_rank].copy()
            source['rank'] = old_to_new[old_rank]  # Renumber
            source['original_rank'] = old_rank  # Keep original for reference
            filtered_sources.append(source)

    return remapped_answer, filtered_sources


# Auto-load existing index on startup (skip manual button click)
AUTOLOAD_INDEX = os.getenv("AUTOLOAD_INDEX", "true").lower() == "true"

# Currency display settings (for cost display in euros)
CURRENCY_EXCHANGE_RATE = float(os.getenv("CURRENCY_EXCHANGE_RATE", "0.8633"))  # 1 USD = X EUR
VAT_RATE = float(os.getenv("VAT_RATE", "0.24"))  # VAT rate (0.24 = 24%)

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
        'research_summary': result.get('research_summary', ''),
        'gap_analyses': result.get('gap_analyses', []),
        'gap_analyses_markdown': result.get('gap_analyses_markdown', '')
    }

    # Append and save
    history.append(entry)

    # Ensure data directory exists
    QUERY_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(QUERY_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump({'queries': history}, f, ensure_ascii=False, indent=2)


def clean_excerpt_for_display(text: str, max_chars: int = 1500) -> str:
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
    page_icon="40px.png",
    layout="wide"
)

# Custom CSS for premium UI design
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
/* Main title only gets gradient (not sidebar headers with emojis) */
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

/* Sidebar headers - keep emoji colors visible */
.stSidebar h1, .stSidebar h2, .stSidebar h3 {
    color: #a78bfa !important;  /* Soft purple */
    font-size: 1.1rem !important;
    letter-spacing: 0.5px;
    font-weight: 600 !important;
}

/* Sidebar dividers */
.stSidebar hr {
    border-color: var(--glass-border) !important;
    margin: 1rem 0 !important;
}

/* === GLASS CARDS (Expanders, Containers) === */
[data-testid="stExpander"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--border-radius) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    box-shadow: var(--glass-shadow) !important;
    overflow: hidden !important;
    transition: var(--transition) !important;
}

[data-testid="stExpander"]:hover {
    border-color: rgba(102, 126, 234, 0.4) !important;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15) !important;
}

[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* === BUTTONS === */
/* Primary button - gradient */
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

/* Secondary button - glass */
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

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: var(--text-muted) !important;
}

/* === SELECT BOXES === */
[data-testid="stSelectbox"] > div > div {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    backdrop-filter: blur(10px) !important;
}

div[data-testid="stSelectbox"] {
    min-width: 70px !important;
}

div[data-testid="stSelectbox"] label {
    display: none !important;
}

/* === RADIO BUTTONS === */
div[data-testid="stRadio"] > div {
    gap: 0.5rem !important;
}

div[data-testid="stRadio"] label {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.8rem !important;
    font-size: 0.85rem !important;
    transition: var(--transition) !important;
}

div[data-testid="stRadio"] label:hover {
    background: rgba(255, 255, 255, 0.1) !important;
}

div[data-testid="stRadio"] label[data-checked="true"] {
    background: var(--primary-gradient) !important;
    border-color: transparent !important;
}

/* === CHECKBOXES === */
div[data-testid="stCheckbox"] label {
    white-space: nowrap !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
}

div[data-testid="stCheckbox"] label span[data-checked="true"] {
    background: var(--primary-gradient) !important;
}

/* === SUCCESS/ERROR/INFO ALERTS === */
[data-testid="stAlert"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--border-radius) !important;
    backdrop-filter: blur(10px) !important;
}

/* Success alert */
div[data-baseweb="notification"][kind="positive"] {
    background: rgba(17, 153, 142, 0.15) !important;
    border-left: 4px solid #38ef7d !important;
}

/* Error alert */
div[data-baseweb="notification"][kind="negative"] {
    background: rgba(245, 87, 108, 0.15) !important;
    border-left: 4px solid #f5576c !important;
}

/* Info alert */
div[data-baseweb="notification"][kind="info"] {
    background: rgba(102, 126, 234, 0.15) !important;
    border-left: 4px solid #667eea !important;
}

/* === METRICS === */
[data-testid="stMetric"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--border-radius) !important;
    padding: 1rem !important;
    backdrop-filter: blur(10px) !important;
}

[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
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

/* === CODE BLOCKS === */
[data-testid="stCode"] {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
}

/* === SPINNER === */
[data-testid="stSpinner"] > div {
    border-top-color: #667eea !important;
}

/* === SEARCH AREA === */
div[data-testid="stHorizontalBlock"] {
    gap: 0.5rem;
    align-items: center;
}

/* === QUERY HISTORY (Sidebar) === */
.stSidebar [data-testid="stCaptionContainer"] {
    margin-bottom: -0.5rem !important;
    margin-top: 0.5rem !important;
    color: var(--text-muted) !important;
}

.stSidebar button[kind="secondary"] {
    text-align: left !important;
    padding: 0.5rem 0.75rem !important;
    line-height: 1.4 !important;
    white-space: normal !important;
    height: auto !important;
    min-height: unset !important;
    font-size: 0.8rem !important;
    margin-bottom: 0.25rem !important;
}

/* Sidebar section spacing */
.stSidebar [data-testid="stVerticalBlockBorderWrapper"] {
    margin-bottom: 0 !important;
}

/* === SCROLLBAR STYLING === */
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

/* === DATAFRAMES/TABLES === */
[data-testid="stDataFrame"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--border-radius) !important;
    overflow: hidden !important;
}

/* === PROGRESS BAR === */
[data-testid="stProgress"] > div > div {
    background: var(--primary-gradient) !important;
}

/* === TOOLTIP === */
[data-testid="stTooltipContent"] {
    background: rgba(15, 15, 35, 0.95) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    backdrop-filter: blur(10px) !important;
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

/* === LOGO GLOW EFFECT === */
.stSidebar img {
    filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.3));
    transition: var(--transition);
}

.stSidebar img:hover {
    filter: drop-shadow(0 0 30px rgba(102, 126, 234, 0.5));
}
</style>
""", unsafe_allow_html=True)

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


@st.dialog("⚙️ Settings", width="large")
def settings_dialog():
    """Settings dialog for file exclusions and other configuration."""
    config = load_config()
    db_path = str(config.vector_db.lancedb_path)
    vault_path = config.vault_path

    st.subheader("📁 File Exclusions")
    st.caption("Exclude files or folders from indexing using patterns")

    # Pattern input section
    col1, col2 = st.columns([3, 1])
    with col1:
        pattern = st.text_input(
            "Add exclusion pattern",
            placeholder="Archive/** or *.excalidraw.md or ^Daily.*2023",
            key="exclusion_pattern_input"
        )
    with col2:
        pattern_type = st.selectbox(
            "Type",
            ["glob", "exact", "regex"],
            key="exclusion_type_select",
            help="glob: wildcards like *, **\nexact: exact path match\nregex: regular expression"
        )

    # Pattern help
    with st.expander("📖 Pattern Examples"):
        st.markdown("""
        | Type | Pattern | Matches |
        |------|---------|---------|
        | glob | `Archive/**` | All files in Archive/ |
        | glob | `*.excalidraw.md` | All Excalidraw files |
        | glob | `**/drafts/*` | Any 'drafts' folder |
        | exact | `Projects/old-project.md` | Specific file |
        | regex | `^Daily.*2023` | Daily notes from 2023 |
        """)

    # Preview and Add buttons
    col_preview, col_add = st.columns(2)

    with col_preview:
        if st.button("👁️ Preview Matches", disabled=not pattern):
            if pattern:
                preview = preview_exclusions(
                    [{"pattern": pattern, "type": pattern_type}],
                    vault_path
                )
                if preview["excluded_count"] > 0:
                    st.info(f"Would exclude **{preview['excluded_count']}** of {preview['total_files']} files")
                    # Show first 10 matches
                    with st.expander(f"Matching files ({min(10, preview['excluded_count'])} of {preview['excluded_count']})"):
                        for path, _ in preview["excluded_files"][:10]:
                            st.text(f"  📄 {path}")
                        if preview["excluded_count"] > 10:
                            st.caption(f"... and {preview['excluded_count'] - 10} more")
                else:
                    st.warning("No files match this pattern")

    with col_add:
        if st.button("➕ Add Pattern", type="primary", disabled=not pattern):
            if pattern:
                try:
                    # Add to settings
                    add_exclusion(db_path, pattern, pattern_type)

                    # Get files to remove from index
                    preview = preview_exclusions(
                        [{"pattern": pattern, "type": pattern_type}],
                        vault_path
                    )

                    if preview["excluded_count"] > 0:
                        # Remove from index - convert relative paths to absolute
                        # LanceDB stores absolute paths, preview returns relative
                        file_paths = [
                            str(vault_path / p) for p, _ in preview["excluded_files"]
                        ]
                        deleted = delete_from_index(db_path, file_paths)
                        st.success(f"✅ Added exclusion. Removed {deleted} chunks from index.")
                    else:
                        st.success("✅ Added exclusion pattern.")

                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # Current exclusions list
    st.subheader("Current Exclusions")
    exclusions = get_exclusions(db_path)

    if not exclusions:
        st.info("No exclusion patterns configured")
    else:
        # Count total excluded files
        all_patterns = [{"pattern": e["pattern"], "type": e["type"]} for e in exclusions]
        total_preview = preview_exclusions(all_patterns, vault_path)
        st.caption(f"💡 Total: {total_preview['excluded_count']} files excluded from indexing")

        # Display each exclusion with delete button
        for exc in exclusions:
            col_pattern, col_type, col_delete = st.columns([4, 1, 0.5])

            with col_pattern:
                # Preview count for this specific pattern
                single_preview = preview_exclusions(
                    [{"pattern": exc["pattern"], "type": exc["type"]}],
                    vault_path
                )
                st.text(f"📁 {exc['pattern']}")
                st.caption(f"Matches {single_preview['excluded_count']} files")

            with col_type:
                st.caption(exc["type"])

            with col_delete:
                if st.button("🗑️", key=f"del_{exc['pattern']}_{exc['type']}", help="Remove pattern"):
                    try:
                        remove_exclusion(db_path, exc["pattern"], exc["type"])
                        st.success(f"Removed: {exc['pattern']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")


@st.dialog("LLM Token Usage & Costs", width="large")
def llm_usage_dialog():
    """Dialog showing daily LLM token usage and costs in EUR with VAT."""
    tracker = get_llm_tracker()
    stats = tracker.get_total_stats()

    # Convert costs to EUR
    total_cost_eur = stats['total_cost'] * CURRENCY_EXCHANGE_RATE
    total_vat = total_cost_eur * VAT_RATE
    total_with_vat = total_cost_eur + total_vat

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Tokens", f"{stats['total_tokens']:,}")
    with col2:
        st.metric("Cost (EUR)", f"€{total_cost_eur:.4f}")
    with col3:
        st.metric("VAT (24%)", f"€{total_vat:.4f}")
    with col4:
        st.metric("Total + VAT", f"€{total_with_vat:.4f}")
    with col5:
        st.metric("Days", stats['days_tracked'])

    st.divider()

    # Daily breakdown table with EUR and VAT
    st.subheader("Daily Breakdown")

    # Get raw daily usage and convert to EUR with VAT
    daily_usage = tracker.get_all_daily_usage()[:30]

    if daily_usage:
        import pandas as pd

        # Build table data with EUR conversion and VAT
        table_data = []
        for usage in daily_usage:
            cost_eur = usage.total_cost * CURRENCY_EXCHANGE_RATE
            vat = cost_eur * VAT_RATE
            total_vat = cost_eur + vat

            table_data.append({
                'Date': usage.date,
                'Input Tokens': f"{usage.input_tokens:,}",
                'Output Tokens': f"{usage.output_tokens:,}",
                'Requests': usage.requests,
                'Cost €': f"€{cost_eur:.4f}",
                'VAT €': f"€{vat:.4f}",
                'Total+VAT': f"€{total_vat:.4f}",
            })

        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Input Tokens": st.column_config.TextColumn("Input", width="small"),
                "Output Tokens": st.column_config.TextColumn("Output", width="small"),
                "Requests": st.column_config.NumberColumn("Req", width="small"),
                "Cost €": st.column_config.TextColumn("Cost €", width="small"),
                "VAT €": st.column_config.TextColumn("VAT €", width="small"),
                "Total+VAT": st.column_config.TextColumn("Total", width="small"),
            }
        )
    else:
        st.info("No usage data recorded yet. Token tracking starts when you make queries.")

    # Pricing info with EUR
    with st.expander("Gemini Pricing Reference (EUR)"):
        # Convert USD prices to EUR
        flash_in = 0.50 * CURRENCY_EXCHANGE_RATE
        flash_out = 3.00 * CURRENCY_EXCHANGE_RATE
        pro_in = 2.00 * CURRENCY_EXCHANGE_RATE
        pro_out = 12.00 * CURRENCY_EXCHANGE_RATE
        flash25_in = 0.30 * CURRENCY_EXCHANGE_RATE
        flash25_out = 2.50 * CURRENCY_EXCHANGE_RATE

        st.markdown(f"""
        | Model | Input (€/1M tokens) | Output (€/1M tokens) |
        |-------|---------------------|----------------------|
        | gemini-3-flash-preview | €{flash_in:.2f} | €{flash_out:.2f} |
        | gemini-3-pro-preview | €{pro_in:.2f} | €{pro_out:.2f} |
        | gemini-2.5-flash | €{flash25_in:.2f} | €{flash25_out:.2f} |
        | gemini-flash-latest | €{flash25_in:.2f} | €{flash25_out:.2f} |

        *Exchange rate: 1 USD = {CURRENCY_EXCHANGE_RATE} EUR | VAT: {int(VAT_RATE * 100)}%*

        *Token counts are estimated when actual counts are unavailable.*
        """)

    # Reset button (with confirmation)
    st.divider()
    col_reset, col_spacer = st.columns([1, 3])
    with col_reset:
        if st.button("Reset Usage Data", type="secondary"):
            st.session_state.confirm_reset_llm_usage = True

    if st.session_state.get("confirm_reset_llm_usage", False):
        st.warning("Are you sure? This will delete all usage history.")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Yes, Reset", type="primary"):
                tracker.reset_usage(confirm=True)
                st.session_state.confirm_reset_llm_usage = False
                st.success("Usage data reset!")
                st.rerun()
        with col_no:
            if st.button("Cancel"):
                st.session_state.confirm_reset_llm_usage = False
                st.rerun()


def main():
    st.title("🧠 UltraRAG - Obsidian Knowledge Assistant")
    st.markdown("World-class RAG system for your personal knowledge base")
    
    # Sidebar for configuration
    with st.sidebar:
        # Logo
        st.markdown(
            """
            <div style='text-align: center; padding: 10px 0 20px 0;'>
                <img src='app/static/icon-512.png' style='width: 175px; height: 175px; border-radius: 12px;'>
            </div>
            """,
            unsafe_allow_html=True
        )

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

        # Settings button (shows exclusions and other configuration)
        if st.session_state.indexed:
            st.divider()
            col_settings, col_usage = st.columns(2)
            with col_settings:
                if st.button("⚙️ Settings", use_container_width=True):
                    settings_dialog()
            with col_usage:
                if st.button("💰 LLM Costs", use_container_width=True):
                    llm_usage_dialog()

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
            # Show most recent 15 queries (reversed for newest first)
            for entry in reversed(persistent_history[-15:]):
                # Parse timestamp for display
                try:
                    ts = datetime.fromisoformat(entry['timestamp'])
                    time_str = ts.strftime("%b %d, %I:%M %p")
                except (KeyError, ValueError):
                    time_str = "Unknown"

                # Show datetime as small caption, query as button
                # Truncate query at 80 chars (more room now without inline datetime)
                query_short = entry['query'][:80] + '...' if len(entry['query']) > 80 else entry['query']

                # Use container for visual grouping
                with st.container():
                    st.caption(f"🕐 {time_str}")
                    if st.button(query_short, key=f"hist_{entry.get('id', entry['timestamp'])}", use_container_width=True):
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

        # Search controls - Row 1: Button + Search type
        row1_col1, row1_col2 = st.columns([1, 4])
        with row1_col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with row1_col2:
            search_type = st.radio(
                "Search type:",
                ["Full Answer", "Find Notes Only"],
                horizontal=True,
                label_visibility="collapsed"
            )

        # Search controls - Row 2: Scope + Options
        has_raptor = st.session_state.rag.raptor_index_exists() if st.session_state.rag else False

        if st.session_state.conversations_indexed:
            # With conversations: Scope + Research + RAPTOR + Max sources
            row2_col1, row2_col2, row2_col3, row2_col4 = st.columns([3, 1, 1, 1])
            with row2_col1:
                search_scope = st.radio(
                    "Search scope:",
                    ["📓 Vault", "💬 Convos", "🔀 Both"],
                    horizontal=True,
                    label_visibility="collapsed",
                    index=2  # Default to "Both"
                )
                # Map back to full names for consistency
                scope_map = {"📓 Vault": "📓 Vault Only", "💬 Convos": "💬 Conversations", "🔀 Both": "🔀 Both"}
                search_scope = scope_map.get(search_scope, search_scope)
        else:
            # Without conversations: Research + RAPTOR + Max sources
            search_scope = "📓 Vault Only"
            row2_col2, row2_col3, row2_col4 = st.columns([1, 1, 1])

        with row2_col2:
            research_mode = st.checkbox(
                "🔬 Research",
                help="Multi-step retrieval with gap analysis. Use @all prefix for exhaustive search.",
                value=False
            )
        with row2_col3:
            if has_raptor:
                raptor_mode = st.checkbox(
                    "🌳 RAPTOR",
                    help="Use hierarchical summaries for multi-document reasoning",
                    value=False
                )
            else:
                raptor_mode = False
                st.empty()  # Placeholder to maintain layout
        with row2_col4:
            max_sources_options = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]
            max_sources = st.selectbox(
                "Sources",
                options=max_sources_options,
                index=0,  # Default to 10
                help="Maximum sources for synthesis",
                label_visibility="collapsed"
            )

        # Display pending result (from rerun after save)
        if st.session_state.pending_result and not search_button:
            pending = st.session_state.pending_result
            result = pending['result']
            exec_time = pending['exec_time']
            search_scope = pending.get('search_scope', '📓 Vault Only')
            research_mode = pending.get('research_mode', False)
            tokens_used = pending.get('tokens_used', 0)

            # Clear pending result after retrieving
            st.session_state.pending_result = None

            # Filter to cited sources only and renumber citations sequentially
            original_source_count = len(result['sources'])
            remapped_answer, filtered_sources = filter_and_renumber_citations(
                result['answer'], result['sources']
            )
            cited_count = len(filtered_sources)

            # Display execution summary
            word_count = len(remapped_answer.split())
            time_str = format_duration(exec_time)
            tokens_str = f" • **{tokens_used:,} tokens**" if tokens_used > 0 else ""
            sources_str = f"**Cited {cited_count} of {original_source_count} sources**" if research_mode else f"**{cited_count} sources**"
            st.markdown(
                f"{sources_str} "
                f"→ **{word_count:,} words** in **{time_str}**{tokens_str} "
                f"• *saved to history*"
            )

            # Display answer with clickable citation links (using remapped answer)
            st.markdown("### 📝 Answer")
            answer_with_links = linkify_citations(remapped_answer)
            st.markdown(answer_with_links, unsafe_allow_html=True)

            # Build source map for wikilink replacement (using filtered sources)
            source_map = {
                source['rank']: source['title']
                for source in filtered_sources
            }

            # Generate copy versions (using remapped answer)
            clean_text = strip_citations(remapped_answer)
            linked_text = format_with_wikilink_footnotes(remapped_answer, source_map)
            render_copy_buttons(clean_text, linked_text)

            # Show research summary for research mode
            if research_mode and 'research_summary' in result:
                with st.expander("🔬 Research Details", expanded=False):
                    st.text(result['research_summary'])

            # Show gap analyses for research mode (valuable insights from iterative retrieval)
            if research_mode and result.get('gap_analyses_markdown'):
                with st.expander("🧠 Gap Analysis Insights", expanded=False):
                    st.markdown(result['gap_analyses_markdown'])

            # Show source summary for federated queries
            if search_scope == "🔀 Both" and 'source_summary' in result:
                summary = result.get('source_summary', {})
                if summary:
                    by_type = summary.get('by_type', {})
                    vault_count = by_type.get('vault', 0)
                    conv_count = by_type.get('conversations', 0)
                    st.info(f"📊 Sources: {vault_count} from vault, {conv_count} from conversations")

            # Display only cited sources (filtered and renumbered)
            st.markdown(f"### 📚 Sources ({cited_count})")
            for source in filtered_sources:
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
                # Filter to cited sources only and renumber citations sequentially
                original_sources = entry.get('sources', [])
                remapped_answer, filtered_sources = filter_and_renumber_citations(
                    entry['answer'], original_sources
                )
                cited_count = len(filtered_sources)
                word_count = len(remapped_answer.split())
                st.markdown(f"**{cited_count} sources** → **{word_count:,} words**")

                # Display answer with clickable citations (using remapped answer)
                st.markdown("### 📝 Answer")
                answer_with_links = linkify_citations(remapped_answer)
                st.markdown(answer_with_links, unsafe_allow_html=True)

                # Build source map for wikilinks (using filtered sources)
                source_map = {
                    source['rank']: source['title']
                    for source in filtered_sources
                }

                # Generate copy versions (using remapped answer)
                clean_text = strip_citations(remapped_answer)
                linked_text = format_with_wikilink_footnotes(remapped_answer, source_map)

                # Render copy buttons
                render_copy_buttons(clean_text, linked_text)

                # Show gap analyses if available (from research mode)
                if entry.get('gap_analyses_markdown'):
                    with st.expander("🧠 Gap Analysis Insights", expanded=False):
                        st.markdown(entry['gap_analyses_markdown'])

                # Display only cited sources (filtered and renumbered)
                if filtered_sources:
                    st.markdown(f"### 📚 Sources ({cited_count})")
                    for source in filtered_sources:
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

                # Capture token usage before query
                llm_tracker = get_llm_tracker()
                tokens_before = llm_tracker.get_today_totals()
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

                            # Calculate tokens used for this query
                            tokens_after = llm_tracker.get_today_totals()
                            tokens_used = (tokens_after[0] - tokens_before[0]) + (tokens_after[1] - tokens_before[1])

                            # Save to persistent history
                            save_query_to_history(query, result)

                            # Store result for potential rerun and trigger sidebar update
                            if not st.session_state.history_updated:
                                st.session_state.pending_result = {
                                    'result': result,
                                    'query': query,
                                    'exec_time': exec_time,
                                    'search_scope': search_scope,
                                    'research_mode': research_mode,
                                    'tokens_used': tokens_used
                                }
                                st.session_state.history_updated = True
                                st.rerun()  # Rerun to update sidebar with new history

                            # Reset flag for next query
                            st.session_state.history_updated = False

                            # Filter to cited sources only and renumber citations sequentially
                            original_source_count = len(result['sources'])
                            remapped_answer, filtered_sources = filter_and_renumber_citations(
                                result['answer'], result['sources']
                            )
                            cited_count = len(filtered_sources)

                            # Display execution summary with history confirmation
                            time_str = format_duration(exec_time)
                            tokens_str = f" • **{tokens_used:,} tokens**" if tokens_used > 0 else ""
                            sources_str = f"**Cited {cited_count} of {original_source_count} sources**" if research_mode else f"**{cited_count} sources**"
                            st.markdown(
                                f"{sources_str} "
                                f"→ **{word_count:,} words** in **{time_str}**{tokens_str} "
                                f"• *saved to history*"
                            )

                            # Display answer with clickable citation links (using remapped answer)
                            st.markdown("### 📝 Answer")
                            answer_with_links = linkify_citations(remapped_answer)
                            st.markdown(answer_with_links, unsafe_allow_html=True)

                            # Build source map for wikilink replacement (using filtered sources)
                            source_map = {
                                source['rank']: source['title']
                                for source in filtered_sources
                            }

                            # Generate copy versions (using remapped answer)
                            clean_text = strip_citations(remapped_answer)
                            linked_text = format_with_wikilink_footnotes(remapped_answer, source_map)

                            # Render copy buttons
                            render_copy_buttons(clean_text, linked_text)

                            # Show research summary for research mode
                            if research_mode and 'research_summary' in result:
                                with st.expander("🔬 Research Details", expanded=False):
                                    st.text(result['research_summary'])

                            # Show gap analyses for research mode (valuable insights from iterative retrieval)
                            if research_mode and result.get('gap_analyses_markdown'):
                                with st.expander("🧠 Gap Analysis Insights", expanded=False):
                                    st.markdown(result['gap_analyses_markdown'])

                            # Show source summary for federated queries
                            if search_scope == "🔀 Both" and 'source_summary' in result:
                                summary = result.get('source_summary', {})
                                if summary:
                                    by_type = summary.get('by_type', {})
                                    vault_count = by_type.get('vault', 0)
                                    conv_count = by_type.get('conversations', 0)
                                    st.info(f"📊 Sources: {vault_count} from vault, {conv_count} from conversations")

                            # Display only cited sources (filtered and renumbered)
                            st.markdown(f"### 📚 Sources ({cited_count})")
                            for source in filtered_sources:
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
