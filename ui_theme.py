"""Shared Streamlit theme injection for UltraRAG pages."""
from __future__ import annotations

import streamlit as st


def inject_theme() -> None:
    """Inject shared CSS tokens and global component styling."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --brand: #667eea;
    --brand-light: #818cf8;
    --brand-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --accent: #10B981;
    --bg-base: #020617;
    --bg-surface: #0f172a;
    --bg-glass: rgba(255, 255, 255, 0.03);
    --bg-glass-hover: rgba(255, 255, 255, 0.06);
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-default: rgba(255, 255, 255, 0.1);
    --border-hover: rgba(102, 126, 234, 0.3);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.35);
    --shadow-glow: 0 0 24px rgba(102, 126, 234, 0.12);
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --ease: cubic-bezier(0.16, 1, 0.3, 1);
}

.stApp {
    background: linear-gradient(160deg, #020617 0%, #0f172a 40%, #0c1222 100%) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

.main .block-container { padding-top: 1.5rem !important; max-width: 1200px !important; }

h1, h2, h3, h4, h5, h6, p, label, button, input, textarea, select, a,
.stMarkdown, [data-testid="stTextInput"], [data-testid="stTextArea"],
[data-testid="stSelectbox"], [data-testid="stRadio"],
[data-testid="stCheckbox"], [data-testid="stMetric"],
[data-testid="stAlert"], [data-testid="stCaption"],
[data-testid="stExpander"] summary > span {
    font-family: 'Inter', -apple-system, sans-serif !important;
}
code, pre { font-family: 'JetBrains Mono', monospace !important; }

.main h1 {
    background: var(--brand-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
h2, h3 { color: var(--text-primary) !important; font-weight: 600 !important; }
.stMarkdown p { color: var(--text-secondary) !important; line-height: 1.65 !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #0a0f1e 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
.stSidebar h1, .stSidebar h2, .stSidebar h3 {
    color: var(--text-secondary) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}
.stSidebar hr { border-color: var(--border-subtle) !important; margin: 0.75rem 0 !important; }

[data-testid="stExpander"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 250ms var(--ease) !important;
}
[data-testid="stExpander"]:hover {
    border-color: var(--border-hover) !important;
    box-shadow: var(--shadow-md), var(--shadow-glow) !important;
}

button[kind="primary"] {
    background: var(--brand-gradient) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 250ms var(--ease) !important;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
}
button[kind="primary"]:hover { transform: translateY(-1px) !important; }

button[kind="secondary"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    transition: all 250ms var(--ease) !important;
}
button[kind="secondary"]:hover { background: var(--bg-glass-hover) !important; border-color: var(--border-hover) !important; }

[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 250ms var(--ease) !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--brand) !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-muted) !important; }

[data-testid="stSlider"] > div > div > div { background: var(--brand-gradient) !important; }
div[data-testid="stCheckbox"] label { color: var(--text-secondary) !important; }

[data-testid="stAlert"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
}
[data-testid="stMetric"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem !important;
}
[data-testid="stMetric"] label { color: var(--text-muted) !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; font-size: 0.7rem !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    background: var(--brand-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stTabs"] button { background: transparent !important; border: none !important; border-bottom: 2px solid transparent !important; color: var(--text-muted) !important; font-weight: 500 !important; }
[data-testid="stTabs"] button:hover { color: var(--text-secondary) !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--text-primary) !important; border-bottom-color: var(--brand) !important; }

[data-testid="stDataFrame"] { background: var(--bg-glass) !important; border: 1px solid var(--border-default) !important; border-radius: var(--radius-lg) !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 184, 0.35); }

a { color: var(--brand-light) !important; text-decoration: none !important; }
a:hover { color: #34D399 !important; }
hr { border: none !important; height: 1px !important; background: var(--border-subtle) !important; margin: 1.25rem 0 !important; }

@keyframes fadeInUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
[data-testid="stExpander"], [data-testid="stAlert"], [data-testid="stMetric"] { animation: fadeInUp 0.35s var(--ease) both; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}
</style>
        """,
        unsafe_allow_html=True,
    )
