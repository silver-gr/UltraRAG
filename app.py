"""Simple web interface for UltraRAG using Streamlit."""
import os
import re
import time
import json
import uuid
from html import escape
from urllib.parse import quote, urlparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from pathlib import Path
from datetime import datetime
from main import UltraRAG
from config import load_config
from models import BookFilter, _normalize_category
from vector_store import index_exists, delete_from_index
from temporal_filter import get_all_presets, DateFilterPreset
from settings_store import get_exclusions, add_exclusion, remove_exclusion
from exclusion_matcher import ExclusionMatcher, preview_exclusions
from llm_token_tracker import get_llm_tracker
from auth import require_auth, is_admin, require_admin, logout
from rate_limit import run_with_limits, QueryCooldownError, SystemBusyError
from user_storage import user_query_history_path
from ui_theme import inject_theme

# Get Obsidian vault name from environment for clickable links
OBSIDIAN_VAULT_NAME = os.getenv("OBSIDIAN_VAULT_NAME", "")

# Source type icons used consistently across all source display locations
SOURCE_ICONS = {
    "vault": "📓",
    "conversations": "💬",
    "books": "📚",
    "saved_items": "🔖",
}


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format (e.g., '2m 47s' or '45s')."""
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        return f"{seconds:.1f}s"


def extract_cited_numbers(text: str) -> set[int]:
    """Extract all citation numbers from text, handling multiple formats.

    Handles:
    - [1] - single number brackets
    - [1, 2, 3] - comma-separated numbers in single brackets
    - [1][2][3] - consecutive single-number brackets
    """
    numbers = set()

    # Pattern 1: Single number brackets [1], [23]
    for match in re.finditer(r'\[(\d+)\]', text):
        numbers.add(int(match.group(1)))

    # Pattern 2: Comma-separated numbers [1, 2, 3] or [1,2,3]
    for match in re.finditer(r'\[(\d+(?:\s*,\s*\d+)+)\]', text):
        for num in re.findall(r'\d+', match.group(1)):
            numbers.add(int(num))

    return numbers


def filter_and_renumber_citations(answer: str, sources: list) -> tuple[str, list]:
    """Filter sources to only cited ones and renumber citations sequentially.

    Args:
        answer: The answer text with citations like [1], [7], [263] or [1, 7, 263]
        sources: Full list of sources with 'rank' field

    Returns:
        Tuple of (remapped_answer, filtered_sources) where citations are 1-N
    """
    # Extract all cited source numbers from the answer (handles both [N] and [N, M, ...])
    cited_numbers = extract_cited_numbers(answer)

    if not cited_numbers:
        return answer, sources

    # Create mapping from old rank to new sequential number
    # Sort cited numbers to maintain order
    sorted_cited = sorted(cited_numbers)
    old_to_new = {old: new for new, old in enumerate(sorted_cited, 1)}

    # Remap citations in the answer text
    remapped_answer = answer

    # Step 1: Replace comma-separated citations [1, 7, 31] → [§1§, §2§, §4§]
    def remap_comma_citation(match):
        nums_str = match.group(1)
        nums = [n.strip() for n in nums_str.split(',')]
        remapped = []
        for n in nums:
            old_num = int(n)
            if old_num in old_to_new:
                remapped.append(f'§{old_to_new[old_num]}§')
            else:
                remapped.append(f'§{n}§')
        return '[' + ', '.join(remapped) + ']'

    remapped_answer = re.sub(
        r'\[(\d+(?:\s*,\s*\d+)+)\]', remap_comma_citation, remapped_answer
    )

    # Step 2: Replace single citations [31] → [§4§] (largest first to avoid [1] replacing part of [12])
    for old_num in sorted(cited_numbers, reverse=True):
        if old_num in old_to_new:
            new_num = old_to_new[old_num]
            remapped_answer = remapped_answer.replace(f'[{old_num}]', f'[§{new_num}§]')

    # Step 3: Replace temporary markers with final numbers
    remapped_answer = re.sub(r'§(\d+)§', r'\1', remapped_answer)

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

LEGACY_QUERY_HISTORY_FILE = Path("data/query_history.json")
MIGRATED_QUERY_HISTORY_FILE = Path("data/query_history.migrated.json")

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
        if not index_exists(config.vector_db, table_name=config.vector_db.vault_table):
            return None, False

        # Initialize and load
        rag = UltraRAG()
        if rag.load_existing_index():
            return rag, True
        return None, False
    except Exception as e:
        print(f"Cache load failed: {e}")
        return None, False


def load_query_history(username: str) -> list:
    """Load persistent query history from disk."""
    history_file = user_query_history_path(username)
    if not history_file.exists():
        return []
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('queries', [])
    except (json.JSONDecodeError, IOError):
        return []


def save_query_to_history(username: str, query: str, result: dict) -> None:
    """Append query + result to persistent history."""
    # Load existing history
    history = load_query_history(username)

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

    history_file = user_query_history_path(username)
    history_file.parent.mkdir(parents=True, exist_ok=True)

    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump({'queries': history}, f, ensure_ascii=False, indent=2)


def migrate_legacy_query_history(username: str, admin_user: bool) -> None:
    """Migrate one-time global query history into admin's per-user history."""
    if not admin_user or not LEGACY_QUERY_HISTORY_FILE.exists():
        return

    target_file = user_query_history_path(username)
    if target_file.exists():
        return

    try:
        data = json.loads(LEGACY_QUERY_HISTORY_FILE.read_text(encoding="utf-8"))
        queries = data.get("queries", []) if isinstance(data, dict) else []
        if queries:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(
                json.dumps({"queries": queries}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if MIGRATED_QUERY_HISTORY_FILE.exists():
            LEGACY_QUERY_HISTORY_FILE.unlink()
        else:
            LEGACY_QUERY_HISTORY_FILE.rename(MIGRATED_QUERY_HISTORY_FILE)
    except Exception:
        # Keep legacy file untouched if migration fails.
        return


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


def render_file_link(file_path: str, source_type: str = "vault", url: str = None, display_label: str = None) -> str:
    """Render file path as clickable link appropriate to the source type.

    Args:
        file_path: Path to the file (or item_id for saved_items)
        source_type: Type of source ("vault", "conversations", "books", "saved_items")
        url: External URL (for saved_items)
        display_label: Human-readable label (e.g. domain for saved_items)

    Returns:
        HTML string with clickable link or plain text
    """
    file_path = str(file_path or "")

    def safe_external_url(raw_url: str) -> str:
        """Only allow http/https URLs in rendered anchors."""
        if not raw_url:
            return ""
        candidate = raw_url.strip()
        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"}:
            return ""
        return escape(candidate, quote=True)

    if source_type == "vault":
        obsidian_url = make_obsidian_link(file_path)
        safe_file = escape(file_path)
        if obsidian_url:
            return f'📁 <a href="{obsidian_url}" target="_blank" style="color: #1e90ff;">{safe_file}</a> • Vault'
        return f"📁 {safe_file} • Vault"
    elif source_type == "saved_items":
        label = display_label or file_path
        safe_label = escape(str(label))
        safe_url = safe_external_url(url)
        if safe_url:
            return f'🔗 <a href="{safe_url}" target="_blank" rel="noopener noreferrer" style="color: #1e90ff;">{safe_label}</a> • TheSource'
        return f"🔗 {safe_label} • TheSource"
    elif source_type == "books":
        label = display_label or file_path
        return f"📖 {escape(str(label))} • Books"
    elif source_type == "conversations":
        return f"💬 {escape(file_path)} • Conversations"
    else:
        return f"📁 {escape(file_path)} • {source_type.title()}"


def render_stats_pills(cited_count: int, original_count: int, word_count: int,
                       time_str: str, tokens_used: int = 0, research_mode: bool = False,
                       saved: bool = True) -> str:
    """Render execution statistics as styled pill badges."""
    if research_mode:
        sources_text = f"Αναφορές {cited_count}/{original_count}"
    else:
        sources_text = str(cited_count)

    pills = [
        f'<span class="stats-pill pill-sources"><span class="pill-icon">&#x1F4DA;</span> <span class="pill-value">{sources_text}</span> πηγές</span>',
        f'<span class="stats-pill pill-words"><span class="pill-icon">&#x270F;</span> <span class="pill-value">{word_count:,}</span> λέξεις</span>',
        f'<span class="stats-pill pill-time"><span class="pill-icon">&#x23F1;</span> <span class="pill-value">{time_str}</span></span>',
    ]

    if tokens_used > 0:
        pills.append(
            f'<span class="stats-pill pill-tokens"><span class="pill-icon">&#x26A1;</span> <span class="pill-value">{tokens_used:,}</span> tokens</span>'
        )

    return f'<div class="stats-row">{"".join(pills)}</div>'


def render_section_header(icon: str, title: str, count: str = "", icon_class: str = "icon-answer") -> str:
    """Render a styled section header with icon."""
    count_html = f' <span class="section-count">{count}</span>' if count else ""
    return f'''<div class="section-header">
        <div class="section-icon {icon_class}">{icon}</div>
        <span class="section-title">{title}</span>{count_html}
    </div>'''


def render_copy_buttons(clean_text: str, linked_text: str):
    """Render copy options for clean and linked versions of the answer.

    Uses Streamlit's native st.code() which has a built-in copy button,
    wrapped in expanders for clean UI.
    """
    st.markdown("---")
    st.markdown("**📋 Αντιγραφή απάντησης**")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📋 Καθαρό κείμενο (χωρίς παραπομπές)", expanded=False):
            st.code(clean_text, language=None)

    with col2:
        with st.expander("🔗 Με Wikilinks", expanded=False):
            st.code(linked_text, language=None)


# Page config
st.set_page_config(
    page_title="UltraRAG - Βοηθός Γνώσης Obsidian",
    page_icon="/app/static/favicon-32.png",
    layout="wide"
)
inject_theme()

# Custom CSS for premium UI design
st.markdown("""
<style>
/* ============================================
   ULTRARAG DESIGN SYSTEM v2
   Refined dark theme · Inter + JetBrains Mono
   Glass morphism · Swiss-inspired minimalism
   ============================================ */

/* === FONTS === */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* === DESIGN TOKENS === */
:root {
    /* Brand */
    --brand: #667eea;
    --brand-light: #818cf8;
    --brand-dark: #4f46e5;
    --brand-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --accent: #10B981;
    --accent-light: #34D399;
    --accent-gradient: linear-gradient(135deg, #10B981 0%, #34D399 100%);
    --danger: #f43f5e;
    --warning: #f59e0b;

    /* Surfaces */
    --bg-base: #020617;
    --bg-surface: #0f172a;
    --bg-elevated: #1e293b;
    --bg-glass: rgba(255, 255, 255, 0.03);
    --bg-glass-hover: rgba(255, 255, 255, 0.06);
    --bg-glass-active: rgba(255, 255, 255, 0.08);

    /* Borders */
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-default: rgba(255, 255, 255, 0.1);
    --border-hover: rgba(102, 126, 234, 0.3);
    --border-focus: rgba(102, 126, 234, 0.5);

    /* Text */
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;

    /* Shadows */
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.35);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.4);
    --shadow-glow: 0 0 24px rgba(102, 126, 234, 0.12);
    --shadow-glow-hover: 0 0 32px rgba(102, 126, 234, 0.2);

    /* Radius */
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-xl: 20px;

    /* Transitions */
    --ease: cubic-bezier(0.16, 1, 0.3, 1);
    --duration-fast: 150ms;
    --duration-normal: 250ms;
}

/* === GLOBAL === */
.stApp {
    background: linear-gradient(160deg, #020617 0%, #0f172a 40%, #0c1222 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* === TYPOGRAPHY === */
h1, h2, h3, h4, h5, h6, p, label, button, input, textarea, select, a,
.stMarkdown, [data-testid="stTextInput"], [data-testid="stTextArea"],
[data-testid="stSelectbox"], [data-testid="stRadio"],
[data-testid="stCheckbox"], [data-testid="stMetric"],
[data-testid="stAlert"], [data-testid="stCaption"],
[data-testid="stExpander"] summary > span {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

code, pre, [data-testid="stCode"], .stCode {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}

.main h1 {
    background: var(--brand-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

h2, h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}

.stMarkdown p {
    color: var(--text-secondary) !important;
    line-height: 1.65 !important;
}

/* === SIDEBAR === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #0a0f1e 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.75rem !important;
}

.stSidebar h1, .stSidebar h2, .stSidebar h3 {
    color: var(--text-secondary) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

.stSidebar hr {
    border-color: var(--border-subtle) !important;
    margin: 0.75rem 0 !important;
}

/* Sidebar status badges */
.stSidebar [data-testid="stAlert"] {
    padding: 0.5rem 0.75rem !important;
    font-size: 0.8rem !important;
}

/* === GLASS CARDS === */
[data-testid="stExpander"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    box-shadow: var(--shadow-sm) !important;
    overflow: hidden !important;
    transition: all var(--duration-normal) var(--ease) !important;
}

[data-testid="stExpander"]:hover {
    border-color: var(--border-hover) !important;
    box-shadow: var(--shadow-md), var(--shadow-glow) !important;
    background: var(--bg-glass-hover) !important;
}

[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* === BUTTONS === */
button[kind="primary"] {
    background: var(--brand-gradient) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1.5rem !important;
    min-height: 2.5rem !important;
    transition: all var(--duration-normal) var(--ease) !important;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
    letter-spacing: 0.01em !important;
}

button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4) !important;
}

button[kind="primary"]:active {
    transform: translateY(0) !important;
}

button[kind="secondary"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-size: 0.875rem !important;
    backdrop-filter: blur(8px) !important;
    transition: all var(--duration-normal) var(--ease) !important;
}

button[kind="secondary"]:hover {
    background: var(--bg-glass-hover) !important;
    border-color: var(--border-hover) !important;
}

/* === INPUT FIELDS === */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    backdrop-filter: blur(8px) !important;
    transition: all var(--duration-normal) var(--ease) !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--brand) !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15), var(--shadow-glow) !important;
    background: var(--bg-glass-hover) !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: var(--text-muted) !important;
}

/* === SELECT BOXES === */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    backdrop-filter: blur(8px) !important;
}

div[data-testid="stSelectbox"] {
    min-width: 70px !important;
}

div[data-testid="stSelectbox"] label {
    display: none !important;
}

/* === RADIO BUTTONS === */
div[data-testid="stRadio"] > div {
    gap: 0.4rem !important;
}

div[data-testid="stRadio"] label {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.35rem 0.75rem !important;
    font-size: 0.82rem !important;
    transition: all var(--duration-fast) var(--ease) !important;
    cursor: pointer !important;
}

div[data-testid="stRadio"] label:hover {
    background: var(--bg-glass-hover) !important;
    border-color: var(--border-hover) !important;
}

div[data-testid="stRadio"] label[data-checked="true"] {
    background: var(--brand-gradient) !important;
    border-color: transparent !important;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.25) !important;
}

/* === CHECKBOXES === */
div[data-testid="stCheckbox"] label {
    white-space: nowrap !important;
    font-size: 0.82rem !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
}

div[data-testid="stCheckbox"] label span[data-checked="true"] {
    background: var(--brand-gradient) !important;
}

/* === ALERTS === */
[data-testid="stAlert"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    backdrop-filter: blur(8px) !important;
}

div[data-baseweb="notification"][kind="positive"] {
    background: rgba(16, 185, 129, 0.08) !important;
    border-left: 3px solid var(--accent) !important;
}

div[data-baseweb="notification"][kind="negative"] {
    background: rgba(244, 63, 94, 0.08) !important;
    border-left: 3px solid var(--danger) !important;
}

div[data-baseweb="notification"][kind="info"] {
    background: rgba(102, 126, 234, 0.08) !important;
    border-left: 3px solid var(--brand) !important;
}

/* === METRICS === */
[data-testid="stMetric"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem !important;
    backdrop-filter: blur(8px) !important;
    transition: all var(--duration-normal) var(--ease) !important;
}

[data-testid="stMetric"]:hover {
    border-color: var(--border-hover) !important;
}

[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    background: var(--brand-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* === TABS === */
[data-testid="stTabs"] button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    transition: all var(--duration-fast) var(--ease) !important;
}

[data-testid="stTabs"] button:hover {
    color: var(--text-secondary) !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom-color: var(--brand) !important;
}

/* === CODE BLOCKS === */
[data-testid="stCode"] {
    background: rgba(0, 0, 0, 0.4) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
}

/* === SPINNER === */
[data-testid="stSpinner"] > div {
    border-top-color: var(--brand) !important;
}

/* === SEARCH AREA === */
div[data-testid="stHorizontalBlock"] {
    gap: 0.5rem;
    align-items: center;
}

/* === QUERY HISTORY (Sidebar) === */
.stSidebar [data-testid="stCaptionContainer"] {
    margin-bottom: -0.5rem !important;
    margin-top: 0.3rem !important;
    color: var(--text-muted) !important;
    font-size: 0.7rem !important;
}

.stSidebar button[kind="secondary"] {
    text-align: left !important;
    padding: 0.4rem 0.65rem !important;
    line-height: 1.35 !important;
    white-space: normal !important;
    height: auto !important;
    min-height: unset !important;
    font-size: 0.75rem !important;
    margin-bottom: 0.15rem !important;
    border-radius: var(--radius-sm) !important;
}

.stSidebar [data-testid="stVerticalBlockBorderWrapper"] {
    margin-bottom: 0 !important;
}

/* === SCROLLBAR === */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.2);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(148, 163, 184, 0.35);
}

/* === LINKS === */
a {
    color: var(--brand-light) !important;
    text-decoration: none !important;
    transition: color var(--duration-fast) var(--ease) !important;
}

a:hover {
    color: var(--accent-light) !important;
}

/* === DIVIDERS === */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--border-subtle) !important;
    margin: 1.25rem 0 !important;
}

/* === DATAFRAMES === */
[data-testid="stDataFrame"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
}

/* === PROGRESS BAR === */
[data-testid="stProgress"] > div > div {
    background: var(--brand-gradient) !important;
    border-radius: 4px !important;
}

/* === TOOLTIP === */
[data-testid="stTooltipContent"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    backdrop-filter: blur(12px) !important;
}

/* === CUSTOM COMPONENTS === */

/* --- Stats Pill Badges --- */
.stats-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 0.75rem 0 1rem;
}

.stats-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    border: 1px solid var(--border-default);
    background: var(--bg-glass);
    color: var(--text-secondary);
    backdrop-filter: blur(8px);
}

.stats-pill .pill-value {
    color: var(--text-primary);
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
}

.stats-pill.pill-sources { border-color: rgba(102, 126, 234, 0.25); }
.stats-pill.pill-words { border-color: rgba(16, 185, 129, 0.25); }
.stats-pill.pill-time { border-color: rgba(245, 158, 11, 0.25); }
.stats-pill.pill-tokens { border-color: rgba(168, 85, 247, 0.25); }

.stats-pill.pill-sources .pill-icon { color: var(--brand-light); }
.stats-pill.pill-words .pill-icon { color: var(--accent-light); }
.stats-pill.pill-time .pill-icon { color: var(--warning); }
.stats-pill.pill-tokens .pill-icon { color: #a855f7; }

/* --- Answer Glass Card --- */
.answer-card {
    background: var(--bg-glass);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-xl);
    padding: 1.75rem 2rem;
    margin: 1rem 0 1.5rem;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}

.answer-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--brand-gradient);
}

.answer-card h3 {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* --- Feature Cards (Landing Page) --- */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 1.5rem 0;
}

.feature-card {
    background: var(--bg-glass);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    transition: all var(--duration-normal) var(--ease);
    position: relative;
    overflow: hidden;
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    opacity: 0;
    transition: opacity var(--duration-normal) var(--ease);
}

.feature-card:hover {
    border-color: var(--border-hover);
    background: var(--bg-glass-hover);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md), var(--shadow-glow);
}

.feature-card:hover::before {
    opacity: 1;
}

.feature-card:nth-child(1)::before { background: var(--brand-gradient); }
.feature-card:nth-child(2)::before { background: var(--accent-gradient); }
.feature-card:nth-child(3)::before { background: linear-gradient(135deg, #f59e0b, #f97316); }

.feature-card .feature-icon {
    font-size: 1.75rem;
    margin-bottom: 0.75rem;
    display: block;
}

.feature-card .feature-title {
    color: var(--text-primary);
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.5rem;
    letter-spacing: -0.01em;
}

.feature-card .feature-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.feature-card .feature-list li {
    color: var(--text-secondary);
    font-size: 0.85rem;
    padding: 3px 0;
    padding-left: 16px;
    position: relative;
    line-height: 1.5;
}

.feature-card .feature-list li::before {
    content: '';
    position: absolute;
    left: 0;
    top: 11px;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--text-muted);
}

.feature-card:nth-child(1) .feature-list li::before { background: var(--brand-light); }
.feature-card:nth-child(2) .feature-list li::before { background: var(--accent-light); }
.feature-card:nth-child(3) .feature-list li::before { background: #f59e0b; }

/* --- Section Header --- */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.25rem 0 0.75rem;
}

.section-header .section-icon {
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    font-size: 0.9rem;
}

.section-header .section-icon.icon-answer {
    background: rgba(102, 126, 234, 0.12);
}

.section-header .section-icon.icon-sources {
    background: rgba(16, 185, 129, 0.12);
}

.section-header .section-title {
    color: var(--text-primary);
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: -0.01em;
}

.section-header .section-count {
    color: var(--text-muted);
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}

/* --- Hero Subtitle --- */
.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 400;
    margin-top: -0.5rem;
    margin-bottom: 1rem;
    letter-spacing: 0.01em;
}

/* --- Sidebar Logo --- */
.stSidebar img {
    filter: drop-shadow(0 0 16px rgba(102, 126, 234, 0.2));
    transition: filter var(--duration-normal) var(--ease);
}

.stSidebar img:hover {
    filter: drop-shadow(0 0 24px rgba(102, 126, 234, 0.35));
}

/* --- Empty State --- */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--text-muted);
}

.empty-state .empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}

.empty-state .empty-title {
    color: var(--text-secondary);
    font-size: 1.1rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

.empty-state .empty-desc {
    font-size: 0.9rem;
    max-width: 400px;
    margin: 0 auto;
    line-height: 1.6;
}

/* === ANIMATIONS === */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

[data-testid="stExpander"],
[data-testid="stAlert"],
[data-testid="stMetric"] {
    animation: fadeInUp 0.35s var(--ease) both;
}

/* Stagger animation for expanders */
[data-testid="stExpander"]:nth-child(2) { animation-delay: 0.05s; }
[data-testid="stExpander"]:nth-child(3) { animation-delay: 0.1s; }
[data-testid="stExpander"]:nth-child(4) { animation-delay: 0.15s; }
[data-testid="stExpander"]:nth-child(5) { animation-delay: 0.2s; }

/* Respect reduced motion */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* === RESPONSIVE === */
@media (max-width: 768px) {
    .feature-grid {
        grid-template-columns: 1fr;
    }

    .answer-card {
        padding: 1.25rem;
    }

    .stats-row {
        gap: 6px;
    }

    .stats-pill {
        font-size: 0.72rem;
        padding: 4px 10px;
    }
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

CURRENT_AUTH_CTX = require_auth()
migrate_legacy_query_history(CURRENT_AUTH_CTX.username, is_admin(CURRENT_AUTH_CTX))

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


@st.dialog("⚙️ Ρυθμίσεις", width="large")
def settings_dialog(ctx):
    """Settings dialog for file exclusions and other configuration."""
    require_admin(ctx)
    config = load_config()
    db_path = str(config.vector_db.lancedb_path)
    vault_path = config.vault_path

    st.subheader("📁 Εξαιρέσεις αρχείων")
    st.caption("Εξαίρεση αρχείων ή φακέλων από την ευρετηρίαση με μοτίβα")

    # Pattern input section
    col1, col2 = st.columns([3, 1])
    with col1:
        pattern = st.text_input(
            "Προσθήκη μοτίβου εξαίρεσης",
            placeholder="Archive/** or *.excalidraw.md or ^Daily.*2023",
            key="exclusion_pattern_input"
        )
    with col2:
        pattern_type = st.selectbox(
            "Τύπος",
            ["glob", "exact", "regex"],
            key="exclusion_type_select",
            help="glob: μπαλαντέρ όπως *, **\nexact: ακριβές μονοπάτι\nregex: κανονική έκφραση"
        )

    # Pattern help
    with st.expander("📖 Παραδείγματα μοτίβων"):
        st.markdown("""
        | Τύπος | Μοτίβο | Αντιστοιχεί |
        |-------|--------|-------------|
        | glob | `Archive/**` | Όλα τα αρχεία στο Archive/ |
        | glob | `*.excalidraw.md` | Όλα τα αρχεία Excalidraw |
        | glob | `**/drafts/*` | Οποιοσδήποτε φάκελος 'drafts' |
        | exact | `Projects/old-project.md` | Συγκεκριμένο αρχείο |
        | regex | `^Daily.*2023` | Ημερήσιες σημειώσεις από 2023 |
        """)

    # Preview and Add buttons
    col_preview, col_add = st.columns(2)

    with col_preview:
        if st.button("👁️ Προεπισκόπηση", disabled=not pattern):
            if pattern:
                preview = preview_exclusions(
                    [{"pattern": pattern, "type": pattern_type}],
                    vault_path
                )
                if preview["excluded_count"] > 0:
                    st.info(f"Θα εξαιρεθούν **{preview['excluded_count']}** από {preview['total_files']} αρχεία")
                    # Show first 10 matches
                    with st.expander(f"Αρχεία ({min(10, preview['excluded_count'])} από {preview['excluded_count']})"):
                        for path, _ in preview["excluded_files"][:10]:
                            st.text(f"  📄 {path}")
                        if preview["excluded_count"] > 10:
                            st.caption(f"... and {preview['excluded_count'] - 10} more")
                else:
                    st.warning("Κανένα αρχείο δεν αντιστοιχεί")

    with col_add:
        if st.button("➕ Προσθήκη", type="primary", disabled=not pattern):
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
                        with run_with_limits(ctx, "delete_from_index"):
                            deleted = delete_from_index(db_path, file_paths)
                        st.success(f"✅ Προστέθηκε εξαίρεση. Αφαιρέθηκαν {deleted} τμήματα από το ευρετήριο.")
                    else:
                        st.success("✅ Προστέθηκε μοτίβο εξαίρεσης.")

                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # Current exclusions list
    st.subheader("Τρέχουσες εξαιρέσεις")
    exclusions = get_exclusions(db_path)

    if not exclusions:
        st.info("Δεν υπάρχουν μοτίβα εξαίρεσης")
    else:
        # Count total excluded files
        all_patterns = [{"pattern": e["pattern"], "type": e["type"]} for e in exclusions]
        total_preview = preview_exclusions(all_patterns, vault_path)
        st.caption(f"💡 Σύνολο: {total_preview['excluded_count']} αρχεία εξαιρούνται από την ευρετηρίαση")

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
                st.caption(f"Αντιστοιχεί σε {single_preview['excluded_count']} αρχεία")

            with col_type:
                st.caption(exc["type"])

            with col_delete:
                if st.button("🗑️", key=f"del_{exc['pattern']}_{exc['type']}", help="Αφαίρεση μοτίβου"):
                    try:
                        remove_exclusion(db_path, exc["pattern"], exc["type"])
                        st.success(f"Αφαιρέθηκε: {exc['pattern']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")


@st.dialog("Χρήση & Κόστος LLM", width="large")
def llm_usage_dialog(ctx):
    """Dialog showing daily LLM token usage and costs in EUR with VAT."""
    require_admin(ctx)
    tracker = get_llm_tracker()
    stats = tracker.get_total_stats()

    # Convert costs to EUR
    total_cost_eur = stats['total_cost'] * CURRENCY_EXCHANGE_RATE
    total_vat = total_cost_eur * VAT_RATE
    total_with_vat = total_cost_eur + total_vat

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Συνολικά tokens", f"{stats['total_tokens']:,}")
    with col2:
        st.metric("Κόστος (EUR)", f"€{total_cost_eur:.4f}")
    with col3:
        st.metric("ΦΠΑ (24%)", f"€{total_vat:.4f}")
    with col4:
        st.metric("Σύνολο + ΦΠΑ", f"€{total_with_vat:.4f}")
    with col5:
        st.metric("Ημέρες", stats['days_tracked'])

    st.divider()

    # Daily breakdown table with EUR and VAT
    st.subheader("Ημερήσια ανάλυση")

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
                "Date": st.column_config.TextColumn("Ημ/νία", width="small"),
                "Input Tokens": st.column_config.TextColumn("Input", width="small"),
                "Output Tokens": st.column_config.TextColumn("Output", width="small"),
                "Requests": st.column_config.NumberColumn("Req", width="small"),
                "Cost €": st.column_config.TextColumn("Cost €", width="small"),
                "VAT €": st.column_config.TextColumn("ΦΠΑ €", width="small"),
                "Total+VAT": st.column_config.TextColumn("Σύνολο", width="small"),
            }
        )
    else:
        st.info("Δεν υπάρχουν δεδομένα χρήσης ακόμα. Η καταγραφή ξεκινά με το πρώτο ερώτημα.")

    # Pricing info with EUR
    with st.expander("Τιμοκατάλογος Gemini (EUR)"):
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

        *Ισοτιμία: 1 USD = {CURRENCY_EXCHANGE_RATE} EUR | VAT: {int(VAT_RATE * 100)}%*

        *Τα tokens είναι εκτιμήσεις όταν δεν υπάρχουν ακριβή δεδομένα.*
        """)

    # Reset button (with confirmation)
    st.divider()
    col_reset, col_spacer = st.columns([1, 3])
    with col_reset:
        if st.button("Επαναφορά δεδομένων", type="secondary"):
            st.session_state.confirm_reset_llm_usage = True

    if st.session_state.get("confirm_reset_llm_usage", False):
        st.warning("Σίγουρα; Θα διαγραφεί όλο το ιστορικό χρήσης.")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Ναι, επαναφορά", type="primary"):
                with run_with_limits(ctx, "reset_llm_usage"):
                    tracker.reset_usage(confirm=True)
                st.session_state.confirm_reset_llm_usage = False
                st.success("Τα δεδομένα επαναφέρθηκαν!")
                st.rerun()
        with col_no:
            if st.button("Ακύρωση"):
                st.session_state.confirm_reset_llm_usage = False
                st.rerun()


def main():
    ctx = CURRENT_AUTH_CTX
    admin_mode = is_admin(ctx)

    st.title("UltraRAG")
    st.markdown('<p class="hero-subtitle">Ανάκτηση γνώσης από το Obsidian vault σου</p>', unsafe_allow_html=True)
    
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

        st.caption(f"Συνδεδεμένος ως **{ctx.username}** ({ctx.role})")
        if st.button("Αποσύνδεση", use_container_width=True):
            logout()
            st.rerun()

        # Check if .env exists
        if not Path(".env").exists():
            st.error("⚠️ Το αρχείο .env δεν βρέθηκε!")
            st.info("Αντέγραψε το .env.example σε .env και ρύθμισε τις παραμέτρους.")
            return
        
        # Check for existing index
        try:
            config = load_config()
            has_existing_index = index_exists(config.vector_db, table_name=config.vector_db.vault_table)
        except Exception:
            has_existing_index = False

        # Auto-load additional indexes for all users when already built.
        if st.session_state.rag and st.session_state.indexed:
            config = st.session_state.rag.config
            has_conv_index = st.session_state.rag.conversations_index_exists()
            if has_conv_index or st.session_state.conversations_indexed:
                st.session_state.conversations_indexed = True
                if st.session_state.rag.conversations_index is None:
                    st.session_state.rag.load_conversations_index()
                    st.session_state.rag._setup_federated_engine()

            if config.books.enabled and st.session_state.rag.books_index_exists():
                if getattr(st.session_state.rag, "books_index", None) is None:
                    st.session_state.rag.load_books_index()
                    st.session_state.rag._setup_federated_engine()

            if st.session_state.rag.raptor_index_exists() and st.session_state.rag.raptor_manager is None:
                st.session_state.rag.load_raptor_index()

        # Admin-only mutating controls
        if admin_mode:
            with st.expander("Διαχείριση", expanded=True):
                # Show appropriate buttons based on state
                if not st.session_state.rag:
                    if has_existing_index:
                        st.success("📦 Βρέθηκε υπάρχον ευρετήριο!")
                        if st.button("🚀 Φόρτωση ευρετηρίου", type="primary"):
                            with st.spinner("Φόρτωση συστήματος και ευρετηρίου..."):
                                try:
                                    with run_with_limits(ctx, "load_existing_index"):
                                        st.session_state.rag = UltraRAG()
                                        if st.session_state.rag.load_existing_index():
                                            st.session_state.indexed = True
                                            st.success("✅ Το ευρετήριο φορτώθηκε!")
                                            st.rerun()
                                        else:
                                            st.error("Αποτυχία φόρτωσης ευρετηρίου")
                                except SystemBusyError as e:
                                    st.error(str(e))
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")

                        if st.button("🔄 Νέο ευρετήριο"):
                            with st.spinner("Αρχικοποίηση συστήματος..."):
                                try:
                                    with run_with_limits(ctx, "init_system"):
                                        st.session_state.rag = UltraRAG()
                                    st.success("✅ Το σύστημα αρχικοποιήθηκε!")
                                except SystemBusyError as e:
                                    st.error(str(e))
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                    else:
                        if st.button("🚀 Αρχικοποίηση", type="primary"):
                            with st.spinner("Αρχικοποίηση συστήματος..."):
                                try:
                                    with run_with_limits(ctx, "init_system"):
                                        st.session_state.rag = UltraRAG()
                                    st.success("✅ Το σύστημα αρχικοποιήθηκε!")
                                except SystemBusyError as e:
                                    st.error(str(e))
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")

                # Index button (only show if initialized but not indexed)
                if st.session_state.rag and not st.session_state.indexed:
                    if st.button("📚 Ευρετηρίαση vault"):
                        with st.spinner("Ευρετηρίαση vault (μπορεί να πάρει μερικά λεπτά)..."):
                            try:
                                with run_with_limits(ctx, "index_vault"):
                                    st.session_state.rag.index_vault()
                                st.session_state.indexed = True
                                st.success("✅ Το vault ευρετηριάστηκε!")
                                st.balloons()
                            except SystemBusyError as e:
                                st.error(str(e))
                            except Exception as e:
                                st.error(f"❌ Error: {e}")

                # Conversations section
                if st.session_state.rag and st.session_state.indexed:
                    st.divider()
                    st.subheader("Συνομιλίες AI")

                    config = st.session_state.rag.config
                    has_conv_path = config.conversations.path and config.conversations.path.exists()
                    has_conv_index = st.session_state.rag.conversations_index_exists()

                    if has_conv_index or st.session_state.conversations_indexed:
                        st.success("🟢 Συνομιλίες ευρετηριασμένες")
                        st.session_state.conversations_indexed = True

                    elif has_conv_path:
                        st.info(f"📁 Found: {config.conversations.path}")
                        if st.button("📚 Ευρετηρίαση συνομιλιών"):
                            with st.spinner("Ευρετηρίαση συνομιλιών AI..."):
                                try:
                                    with run_with_limits(ctx, "index_conversations"):
                                        st.session_state.rag.index_conversations(force_reindex=False, interactive=False)
                                    st.session_state.conversations_indexed = True
                                    st.success("✅ Οι συνομιλίες ευρετηριάστηκαν!")
                                    st.rerun()
                                except SystemBusyError as e:
                                    st.error(str(e))
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                    else:
                        st.info("Ρύθμισε CONVERSATIONS_PATH στο .env")

                # Books section
                if st.session_state.rag and st.session_state.indexed:
                    config = st.session_state.rag.config
                    if config.books.enabled:
                        st.divider()
                        st.subheader("Βιβλία")
                        has_books_idx = st.session_state.rag.books_index_exists()
                        books_in_mem = getattr(st.session_state.rag, 'books_index', None) is not None
                        if has_books_idx:
                            n_nodes = len(getattr(st.session_state.rag, 'books_nodes', None) or [])
                            if books_in_mem:
                                st.success(f"🟢 Βιβλία έτοιμα ({n_nodes:,} τμήματα)")
                            else:
                                st.info("📚 Βιβλία ευρετηριασμένα — φόρτωση κατ' απαίτηση")
                        else:
                            st.info("Ρύθμισε BOOKS_PATH στο .env")

                # RAPTOR section
                if st.session_state.rag and st.session_state.indexed:
                    st.divider()
                    st.subheader("Περιλήψεις RAPTOR")

                    config = st.session_state.rag.config
                    has_raptor = st.session_state.rag.raptor_index_exists()

                    if has_raptor:
                        st.success("🟢 Ευρετήριο RAPTOR έτοιμο")
                        stats = st.session_state.rag.get_raptor_stats()
                        st.caption(f"Mode: {stats.get('default_mode', 'collapsed')} | Nodes: {stats.get('node_count', 'unknown')}")
                    elif config.raptor.enabled:
                        st.info("RAPTOR ενεργοποιημένο αλλά μη ευρετηριασμένο")
                        if st.button("🌳 Δημιουργία ευρετηρίου RAPTOR"):
                            with st.spinner("Δημιουργία ιεραρχικών περιλήψεων RAPTOR (μπορεί να πάρει λεπτά)..."):
                                try:
                                    with run_with_limits(ctx, "index_raptor"):
                                        st.session_state.rag.index_raptor(force_reindex=False, interactive=False)
                                    st.success("✅ Το ευρετήριο RAPTOR δημιουργήθηκε!")
                                    st.rerun()
                                except SystemBusyError as e:
                                    st.error(str(e))
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                    else:
                        st.info("Ρύθμισε ENABLE_RAPTOR=true στο .env")

                if st.session_state.indexed:
                    st.divider()
                    col_settings, col_usage = st.columns(2)
                    with col_settings:
                        if st.button("⚙️ Ρυθμίσεις", use_container_width=True):
                            settings_dialog(ctx)
                    with col_usage:
                        if st.button("💰 Κόστος LLM", use_container_width=True):
                            llm_usage_dialog(ctx)

        # System status (compact)
        st.divider()
        if st.session_state.rag:
            config = st.session_state.rag.config
            # Truncate model names for compact display
            emb_short = config.embedding.model.replace('voyage-', '').replace('-lite', 'L')
            llm_short = config.llm.model[:12] + '...' if len(config.llm.model) > 12 else config.llm.model
            st.caption(f"{emb_short} / {config.vector_db.db_type.upper()} / {llm_short}")

            if st.session_state.indexed:
                auto_note = " (αυτόματο)" if st.session_state.autoload_attempted else ""
                if st.session_state.conversations_indexed:
                    st.success(f"🟢 Ομοσπονδιακή αναζήτηση έτοιμη{auto_note}")
                else:
                    st.success(f"🟢 Vault έτοιμο{auto_note}")
            else:
                st.warning("🟡 Χωρίς ευρετήριο")
        else:
            st.info("Μη αρχικοποιημένο")

        # Date filter (compact, only show when indexed)
        if st.session_state.indexed:
            preset_options = get_all_presets()
            preset_labels = [label for label, _ in preset_options]
            preset_values = [value for _, value in preset_options]

            current_idx = 0
            if st.session_state.date_filter in preset_values:
                current_idx = preset_values.index(st.session_state.date_filter)

            selected_label = st.selectbox(
                "📅 Φίλτρο ημερομηνίας",
                options=preset_labels,
                index=current_idx,
                help="Φιλτράρισμα κατά ημερομηνία"
            )
            selected_idx = preset_labels.index(selected_label)
            st.session_state.date_filter = preset_values[selected_idx]
        
        # Query history (persistent, clickable)
        st.divider()
        st.subheader("Ιστορικό")
        persistent_history = load_query_history(ctx.username)
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
                    st.caption(time_str)
                    if st.button(query_short, key=f"hist_{entry.get('id', entry['timestamp'])}", use_container_width=True):
                        st.session_state.loaded_history_result = entry
                        st.rerun()
        else:
            st.caption("Δεν υπάρχουν ερωτήματα")
    
    # Main content area
    if not st.session_state.indexed:
        if not admin_mode:
            st.warning("Το ευρετήριο γνώσης δεν είναι έτοιμο. Ζήτησε από τον διαχειριστή να αρχικοποιήσει το σύστημα.")
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">&#x1F50D;</div>
            <div class="empty-title">Αρχικοποίησε το vault σου για να ξεκινήσεις</div>
            <div class="empty-desc">Χρησιμοποίησε την πλαϊνή μπάρα για φόρτωση ευρετηρίου ή δημιουργία νέου.</div>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        st.markdown("""
        <div class="feature-grid">
            <div class="feature-card">
                <span class="feature-icon">&#x1F3AF;</span>
                <div class="feature-title">Έξυπνη ανάκτηση</div>
                <ul class="feature-list">
                    <li>Σημασιολογική αναζήτηση με επανακατάταξη</li>
                    <li>Πλοήγηση γράφου Wikilinks</li>
                    <li>Χρονικό φιλτράρισμα</li>
                </ul>
            </div>
            <div class="feature-card">
                <span class="feature-icon">&#x2728;</span>
                <div class="feature-title">Προηγμένη AI</div>
                <ul class="feature-list">
                    <li>Gemini 3 Flash με context caching</li>
                    <li>Λειτουργία έρευνας με ανάλυση κενών</li>
                    <li>Αυτοδιορθούμενη ανάκτηση</li>
                </ul>
            </div>
            <div class="feature-card">
                <span class="feature-icon">&#x1F4CA;</span>
                <div class="feature-title">Υψηλή ποιότητα</div>
                <ul class="feature-list">
                    <li>Voyage AI embeddings</li>
                    <li>Cross-encoder επανακατάταξη</li>
                    <li>Ενσωματωμένες παραπομπές</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Query interface
        st.subheader("Κάνε μια ερώτηση")

        query = st.text_input(
            "Τι θα ήθελες να μάθεις από το vault σου;",
            placeholder="π.χ., Τι σημειώσεις έχω για μηχανική μάθηση;",
            key="query_input",
            max_chars=10000  # Security: Limit query length
        )

        # Search controls - Row 1: Button + Search type (conditionally shown)
        row1_col1, row1_col2 = st.columns([1, 4])
        with row1_col1:
            search_button = st.button("Αναζήτηση", type="primary", use_container_width=True)

        # Search controls - Row 2: Source checkboxes + Options
        has_raptor = st.session_state.rag.raptor_index_exists() if st.session_state.rag else False
        has_books = st.session_state.rag and getattr(st.session_state.rag, 'books_index', None) is not None
        has_convos = st.session_state.conversations_indexed
        has_saved = st.session_state.rag and getattr(st.session_state.rag, 'saved_items_retriever', None) is not None

        # Source checkboxes row — always shown
        src_col1, src_col2, src_col3, src_col4, row2_col2, row2_col3, row2_col4 = st.columns([1, 1, 1, 1, 1, 1, 1])
        with src_col1:
            use_vault = st.checkbox("📓 Vault", value=True, key="src_vault")
        with src_col2:
            use_convos = st.checkbox("💬 Συνομιλίες", value=has_convos, key="src_convos", disabled=not has_convos)
        with src_col3:
            use_books = st.checkbox("📚 Βιβλία", value=has_books, key="src_books", disabled=not has_books)
        with src_col4:
            use_saved = st.checkbox("🔖 Αποθηκευμένα", value=has_saved, key="src_saved", disabled=not has_saved)

        # Build source_filter — gate availability to prevent stale disabled-checkbox state
        source_filter = []
        if use_vault:
            source_filter.append("vault")
        if use_convos and has_convos:
            source_filter.append("conversations")
        if use_books and has_books:
            source_filter.append("books")
        if use_saved and has_saved:
            source_filter.append("saved_items")

        if not source_filter:
            st.warning("⚠️ Επίλεξε τουλάχιστον μία πηγή.")
        has_selected_sources = bool(source_filter)

        # "Find Notes Only" only makes sense for vault-only
        vault_only = source_filter == ["vault"]
        with row1_col2:
            if vault_only:
                search_type = st.radio(
                    "Τύπος αναζήτησης:",
                    ["Πλήρης απάντηση", "Μόνο σημειώσεις"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
            else:
                search_type = "Πλήρης απάντηση"
                st.empty()

        # search_scope kept for backwards-compat (history display key)
        search_scope = "checkboxes"

        with row2_col2:
            research_mode = st.checkbox(
                "🔬 Έρευνα",
                help="Πολυβηματική ανάκτηση με ανάλυση κενών. Χρήση @all για εξαντλητική αναζήτηση.",
                value=False
            )
        with row2_col3:
            if has_raptor:
                raptor_mode = st.checkbox(
                    "🌳 RAPTOR",
                    help="Ιεραρχικές περιλήψεις για πολυεγγραφική ανάλυση",
                    value=False
                )
            else:
                raptor_mode = False
                st.empty()  # Placeholder to maintain layout
        with row2_col4:
            max_sources_options = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]
            max_sources = st.selectbox(
                "Πηγές",
                options=max_sources_options,
                index=0,  # Default to 10
                help="Μέγιστος αριθμός πηγών για σύνθεση",
                label_visibility="collapsed"
            )

        # Book category/author filters (only show when books are loaded)
        book_filter = None
        has_books = (
            st.session_state.rag
            and getattr(st.session_state.rag, 'books_index', None) is not None
        )
        if has_books:
            categories = st.session_state.rag.get_book_categories()
            if categories:
                with st.expander("Φίλτρα βιβλίων", expanded=False):
                    cat_options = [f"{c['name']} ({c['count']})" for c in categories]
                    selected_cats = st.multiselect(
                        "Κατηγορίες",
                        options=cat_options,
                        key="book_cat_filter",
                        help="Φιλτράρισμα βιβλίων κατά κατηγορία",
                    )
                    # Extract category names from "name (count)" display format
                    selected_cat_names = [
                        _normalize_category(c.rsplit(" (", 1)[0])
                        for c in selected_cats
                    ] if selected_cats else None

                    # Author filter
                    author_input = st.text_input(
                        "Φίλτρο συγγραφέα",
                        placeholder="e.g. Cal Newport, James Clear",
                        key="book_author_filter",
                        help="Ονόματα συγγραφέων χωρισμένα με κόμμα",
                    )
                    selected_authors = (
                        [a.strip() for a in author_input.split(",") if a.strip()]
                        if author_input else None
                    )

                    if selected_cat_names or selected_authors:
                        book_filter = BookFilter(
                            categories=selected_cat_names,
                            authors=selected_authors,
                        )
                        st.info(f"Filters active: {book_filter.to_lance_where()}")
                        if st.button("Καθαρισμός φίλτρων"):
                            st.session_state["book_cat_filter"] = []
                            st.session_state["book_author_filter"] = ""
                            st.rerun()

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

            # Display execution stats as pill badges
            word_count = len(remapped_answer.split())
            time_str = format_duration(exec_time)
            st.markdown(render_stats_pills(
                cited_count, original_source_count, word_count, time_str,
                tokens_used, research_mode
            ), unsafe_allow_html=True)

            # Answer section
            st.markdown(render_section_header("&#x1F4DD;", "Απάντηση", icon_class="icon-answer"), unsafe_allow_html=True)
            answer_with_links = linkify_citations(remapped_answer)
            st.markdown(f'<div class="answer-card">{answer_with_links}</div>', unsafe_allow_html=True)

            # Build source map and copy buttons
            source_map = {
                source['rank']: source['title']
                for source in filtered_sources
            }
            clean_text = strip_citations(remapped_answer)
            linked_text = format_with_wikilink_footnotes(remapped_answer, source_map)
            render_copy_buttons(clean_text, linked_text)

            # Research details
            if research_mode and 'research_summary' in result:
                with st.expander("Λεπτομέρειες έρευνας", expanded=False):
                    st.text(result['research_summary'])

            if research_mode and result.get('gap_analyses_markdown'):
                with st.expander("Ανάλυση κενών", expanded=False):
                    st.markdown(result['gap_analyses_markdown'])

            # Federated source summary (show whenever multi-source result available)
            if 'source_summary' in result:
                by_type = result['source_summary'].get('by_type', {})
                parts = []
                for stype, icon in [("vault", "📓"), ("conversations", "💬"), ("books", "📚"), ("saved_items", "🔖")]:
                    count = by_type.get(stype, 0)
                    if count:
                        label = "saved" if stype == "saved_items" else stype
                        parts.append(f"{icon} {count} {label}")
                if parts:
                    st.info(f"Sources: {', '.join(parts)}")

            # Sources section
            st.markdown(render_section_header("&#x1F4DA;", "Πηγές", str(cited_count), "icon-sources"), unsafe_allow_html=True)
            for source in filtered_sources:
                source_type = source.get('source_type', 'vault')
                type_icon = SOURCE_ICONS.get(source_type, "📄")
                # Anchor for clickable citation
                st.markdown(f'<div id="source-{source["rank"]}"></div>', unsafe_allow_html=True)
                with st.expander(
                    f"**{source['rank']}. {type_icon} {source['title']}** (score: {source.get('score', 0):.3f})"
                ):
                    file_link_html = render_file_link(source['file'], source_type, url=source.get('url'), display_label=source.get('display_label'))
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
            if st.button("✖ Κλείσιμο ιστορικού", type="secondary"):
                st.session_state.loaded_history_result = None
                st.rerun()

            st.markdown(f"*Ερώτημα από {time_str}:*")
            st.markdown(f"**{entry['query']}**")

            if entry.get('answer'):
                # Filter to cited sources only and renumber citations sequentially
                original_sources = entry.get('sources', [])
                remapped_answer, filtered_sources = filter_and_renumber_citations(
                    entry['answer'], original_sources
                )
                cited_count = len(filtered_sources)
                word_count = len(remapped_answer.split())
                st.markdown(render_stats_pills(
                    cited_count, len(original_sources), word_count, "από ιστορικό",
                    research_mode=False
                ), unsafe_allow_html=True)

                # Answer section
                st.markdown(render_section_header("&#x1F4DD;", "Απάντηση", icon_class="icon-answer"), unsafe_allow_html=True)
                answer_with_links = linkify_citations(remapped_answer)
                st.markdown(f'<div class="answer-card">{answer_with_links}</div>', unsafe_allow_html=True)

                # Copy buttons
                source_map = {
                    source['rank']: source['title']
                    for source in filtered_sources
                }
                clean_text = strip_citations(remapped_answer)
                linked_text = format_with_wikilink_footnotes(remapped_answer, source_map)
                render_copy_buttons(clean_text, linked_text)

                # Gap analyses
                if entry.get('gap_analyses_markdown'):
                    with st.expander("Ανάλυση κενών", expanded=False):
                        st.markdown(entry['gap_analyses_markdown'])

                # Sources section
                if filtered_sources:
                    st.markdown(render_section_header("&#x1F4DA;", "Πηγές", str(cited_count), "icon-sources"), unsafe_allow_html=True)
                    for source in filtered_sources:
                        source_type = source.get('source_type', 'vault')
                        type_icon = SOURCE_ICONS.get(source_type, "📄")
                        st.markdown(f'<div id="source-{source["rank"]}"></div>', unsafe_allow_html=True)
                        with st.expander(
                            f"**{source['rank']}. {type_icon} {source['title']}** (score: {source.get('score', 0):.3f})"
                        ):
                            file_link_html = render_file_link(source['file'], source_type, url=source.get('url'), display_label=source.get('display_label'))
                            st.markdown(f'<small style="color: gray;">{file_link_html}</small>', unsafe_allow_html=True)
                            cleaned = clean_excerpt_for_display(source.get('excerpt', ''))
                            st.markdown(cleaned)
            else:
                st.info("Δεν υπάρχει αποθηκευμένη απάντηση.")

        # Security: Validate query input
        elif search_button and query:
            # Clear history view when starting new search
            st.session_state.loaded_history_result = None

            # Validate source selection and query text
            if not has_selected_sources:
                st.error("Επίλεξε τουλάχιστον μία πηγή πριν την αναζήτηση.")
            elif not query.strip():
                st.error("Εισάγετε ένα έγκυρο ερώτημα.")
            elif len(query) > 10000:
                st.error("Το ερώτημα είναι πολύ μεγάλο. Μέγιστο 10.000 χαρακτήρες.")
            else:

                # Show appropriate spinner message
                if raptor_mode:
                    spinner_message = "Αναζήτηση στις ιεραρχικές περιλήψεις RAPTOR..."
                elif research_mode:
                    spinner_message = "Έρευνα στη βάση γνώσης (μπορεί να πάρει 30-60 δευτερόλεπτα)..."
                else:
                    spinner_message = "Αναζήτηση στη βάση γνώσης..."

                # Capture token usage before query
                llm_tracker = get_llm_tracker()
                tokens_before = llm_tracker.get_today_totals()
                start_time = time.time()

                with st.spinner(spinner_message):
                    try:
                        with run_with_limits(ctx, "search_query"):
                            # Get current date filter from session state
                            date_filter = st.session_state.date_filter

                            if search_type == "Πλήρης απάντηση":
                                # Determine which query method to use based on source checkboxes.
                                # We pass max_sources=None to get ALL sources for proper citation matching.
                                if raptor_mode:
                                    # RAPTOR mode uses hierarchical summaries
                                    result = st.session_state.rag.query_raptor(query, max_sources=None)
                                elif research_mode:
                                    # Research mode uses ALL retrieved sources for synthesis
                                    result = st.session_state.rag.query_research(query, max_sources=None, date_filter=date_filter, source_filter=source_filter)
                                elif source_filter == ["vault"]:
                                    # Vault-only: use direct query engine (no federated overhead)
                                    result = st.session_state.rag.query(query, max_sources=None, date_filter=date_filter)
                                else:
                                    # Multi-source: federated query with active source_filter
                                    result = st.session_state.rag.query_federated(
                                        query,
                                        source_filter=source_filter,
                                        max_sources=None,
                                        date_filter=date_filter,
                                        book_filter=book_filter,
                                    )

                                # Calculate execution time and stats
                                exec_time = time.time() - start_time
                                word_count = len(result['answer'].split())

                                # Calculate tokens used for this query
                                tokens_after = llm_tracker.get_today_totals()
                                tokens_used = (tokens_after[0] - tokens_before[0]) + (tokens_after[1] - tokens_before[1])

                                # Save to persistent history
                                save_query_to_history(ctx.username, query, result)

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

                                # Stats pills
                                time_str = format_duration(exec_time)
                                st.markdown(render_stats_pills(
                                    cited_count, original_source_count, word_count, time_str,
                                    tokens_used, research_mode
                                ), unsafe_allow_html=True)

                                # Answer section
                                st.markdown(render_section_header("&#x1F4DD;", "Απάντηση", icon_class="icon-answer"), unsafe_allow_html=True)
                                answer_with_links = linkify_citations(remapped_answer)
                                st.markdown(f'<div class="answer-card">{answer_with_links}</div>', unsafe_allow_html=True)

                                # Copy buttons
                                source_map = {
                                    source['rank']: source['title']
                                    for source in filtered_sources
                                }
                                clean_text = strip_citations(remapped_answer)
                                linked_text = format_with_wikilink_footnotes(remapped_answer, source_map)
                                render_copy_buttons(clean_text, linked_text)

                                # Research details
                                if research_mode and 'research_summary' in result:
                                    with st.expander("Λεπτομέρειες έρευνας", expanded=False):
                                        st.text(result['research_summary'])

                                if research_mode and result.get('gap_analyses_markdown'):
                                    with st.expander("Ανάλυση κενών", expanded=False):
                                        st.markdown(result['gap_analyses_markdown'])

                                # Federated source summary (show whenever multi-source result available)
                                if 'source_summary' in result:
                                    by_type = result['source_summary'].get('by_type', {})
                                    parts = []
                                    for stype, icon in [("vault", "📓"), ("conversations", "💬"), ("books", "📚"), ("saved_items", "🔖")]:
                                        count = by_type.get(stype, 0)
                                        if count:
                                            label = "saved" if stype == "saved_items" else stype
                                            parts.append(f"{icon} {count} {label}")
                                    if parts:
                                        st.info(f"Sources: {', '.join(parts)}")

                                # Sources section
                                st.markdown(render_section_header("&#x1F4DA;", "Πηγές", str(cited_count), "icon-sources"), unsafe_allow_html=True)
                                for source in filtered_sources:
                                    source_type = source.get('source_type', 'vault')
                                    type_icon = SOURCE_ICONS.get(source_type, "📄")
                                    st.markdown(f'<div id="source-{source["rank"]}"></div>', unsafe_allow_html=True)
                                    with st.expander(
                                        f"**{source['rank']}. {type_icon} {source['title']}** — {source['score']:.3f}"
                                    ):
                                        file_link_html = render_file_link(source['file'], source_type, url=source.get('url'), display_label=source.get('display_label'))
                                        st.markdown(f'<small style="color: #64748b;">{file_link_html}</small>', unsafe_allow_html=True)
                                        cleaned = clean_excerpt_for_display(source['excerpt'])
                                        st.markdown(cleaned)
                            else:
                                # Just retrieve relevant notes
                                notes = st.session_state.rag.search_notes(query, top_k=max_sources, date_filter=date_filter)

                                st.markdown(render_section_header("&#x1F4DA;", "Σχετικές σημειώσεις", str(len(notes)), "icon-sources"), unsafe_allow_html=True)
                                for note in notes:
                                    source_type = note.get('source_type', 'vault')
                                    with st.expander(
                                        f"**{note['rank']}. {note['title']}** — {note['score']:.3f}"
                                    ):
                                        file_link_html = render_file_link(note['file'], source_type, url=note.get('url'), display_label=note.get('display_label'))
                                        st.markdown(f'<small style="color: #64748b;">{file_link_html}</small>', unsafe_allow_html=True)
                                        cleaned = clean_excerpt_for_display(note['excerpt'])
                                        st.markdown(cleaned)

                    except QueryCooldownError as e:
                        st.warning(str(e))
                    except SystemBusyError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        # Example queries
        with st.expander("Παραδείγματα ερωτημάτων"):
            st.markdown("""
            - Ποιες είναι οι σκέψεις μου για [θέμα];
            - Δείξε μου σημειώσεις σχετικές με [project]
            - Τι συνδέσεις υπάρχουν μεταξύ [concept A] και [concept B];
            - Σύνοψη σημειώσεων με ετικέτα #important
            - Τι έγραψα για [θέμα] τον τελευταίο μήνα;
            """)


if __name__ == "__main__":
    main()
