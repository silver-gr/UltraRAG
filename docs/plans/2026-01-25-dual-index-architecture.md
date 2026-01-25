# Dual Index Architecture Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Enable separate indexes for Personal Notes vs Research Content with per-index convergence profiles and query scope selection.

**Architecture:** Extend the existing federated query system to support multiple vault partitions (personal/research) with distinct LanceDB tables, convergence configs, and UI controls. Reuse existing `IndexSource` pattern from `federated_query.py`.

**Tech Stack:** Python, LanceDB, Streamlit, Pydantic config models

---

## Overview

Currently UltraRAG has:
- Single vault index (`vectors` table)
- AI Conversations index (`conversations` table)
- Federated query across both

This plan adds:
- **Personal index** (`vault_personal` table) — Personal notes with aggressive convergence
- **Research index** (`vault_research` table) — Research content with conservative convergence
- UI to choose query scope: Personal Only, Research Only, Both
- Per-index configuration in settings

---

## Task 1: Add IndexConfig to config.py

**Files:**
- Modify: `config.py:60-70`

**Step 1: Add IndexConfig model**

Add after `VectorDBConfig` class:

```python
class IndexConfig(BaseModel):
    """Configuration for a single index partition."""
    name: str  # e.g., "personal", "research"
    table_name: str  # LanceDB table name
    source_paths: list[str] = Field(default_factory=list)  # Paths within vault to include
    exclude_paths: list[str] = Field(default_factory=list)  # Paths to exclude
    convergence_profile: str = Field(default="balanced")  # "personal", "research", "balanced"
    weight: float = Field(default=1.0)  # Query weight
    enabled: bool = Field(default=True)

    @field_validator('convergence_profile')
    @classmethod
    def validate_profile(cls, v: str) -> str:
        valid = ["personal", "research", "balanced"]
        if v.lower() not in valid:
            raise ValueError(f"convergence_profile must be one of {valid}")
        return v.lower()
```

**Step 2: Add to VectorDBConfig**

```python
class VectorDBConfig(BaseModel):
    """Vector database configuration."""
    db_type: str = Field(default="lancedb")
    lancedb_path: Path = Field(default=Path("./data/lancedb"))
    conversations_table: str = Field(default="conversations")
    vault_table: str = Field(default="vectors")  # Legacy single-index table
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="obsidian_notes")

    # NEW: Multi-index partitions
    indexes: list[IndexConfig] = Field(default_factory=lambda: [
        IndexConfig(
            name="personal",
            table_name="vault_personal",
            convergence_profile="personal",
            weight=1.0
        ),
        IndexConfig(
            name="research",
            table_name="vault_research",
            convergence_profile="research",
            weight=1.0
        )
    ])
    use_partitioned_indexes: bool = Field(default=False)  # Feature flag
```

**Step 3: Verify**

```bash
python -c "from config import IndexConfig, VectorDBConfig; print('OK')"
```

**Step 4: Commit**

```bash
git add config.py
git commit -m "feat(config): add IndexConfig for multi-index partitions"
```

---

## Task 2: Add settings_store functions for index config

**Files:**
- Modify: `settings_store.py`

**Step 1: Add index config constants and functions**

Add after exclusions functions:

```python
# =============================================================================
# Index configuration storage
# =============================================================================

INDEX_CONFIG_KEY = "index_partitions"


def get_index_configs(db_path: str | Path) -> list[dict]:
    """Get all index partition configurations.

    Returns:
        List of index config dicts
    """
    store = SettingsStore(db_path)
    data = store.get(INDEX_CONFIG_KEY, {"indexes": []})
    return data.get("indexes", [])


def set_index_configs(db_path: str | Path, indexes: list[dict]) -> None:
    """Set index partition configurations.

    Args:
        db_path: Path to LanceDB database
        indexes: List of index config dicts
    """
    store = SettingsStore(db_path)
    store.set(INDEX_CONFIG_KEY, {
        "indexes": indexes,
        "updated_at": datetime.now(timezone.utc).isoformat()
    })
    logger.info(f"Saved {len(indexes)} index configurations")


def get_index_config(db_path: str | Path, name: str) -> dict | None:
    """Get a specific index configuration by name.

    Args:
        db_path: Path to LanceDB database
        name: Index name (e.g., "personal", "research")

    Returns:
        Index config dict or None if not found
    """
    configs = get_index_configs(db_path)
    for config in configs:
        if config.get("name") == name:
            return config
    return None


def update_index_config(db_path: str | Path, name: str, updates: dict) -> bool:
    """Update a specific index configuration.

    Args:
        db_path: Path to LanceDB database
        name: Index name to update
        updates: Dict of fields to update

    Returns:
        True if updated, False if index not found
    """
    configs = get_index_configs(db_path)
    for i, config in enumerate(configs):
        if config.get("name") == name:
            configs[i].update(updates)
            set_index_configs(db_path, configs)
            return True
    return False
```

**Step 2: Verify**

```bash
python -c "from settings_store import get_index_configs, set_index_configs; print('OK')"
```

**Step 3: Commit**

```bash
git add settings_store.py
git commit -m "feat(settings): add index partition config storage"
```

---

## Task 3: Update loader.py for path-based filtering

**Files:**
- Modify: `loader.py`

**Step 1: Add filter_by_paths parameter to ObsidianLoader**

Find the `__init__` method and add parameters:

```python
def __init__(
    self,
    vault_path: str | Path,
    config: Optional[EmbeddingConfig] = None,
    include_patterns: list[str] | None = None,  # NEW: Include only these paths
    exclude_patterns: list[str] | None = None,  # NEW: Exclude these paths
    max_workers: int = 4
):
    """Initialize loader.

    Args:
        vault_path: Path to Obsidian vault
        config: Embedding configuration
        include_patterns: Glob patterns for paths to include (None = all)
        exclude_patterns: Glob patterns for paths to exclude
        max_workers: Worker threads for parallel loading
    """
    self.vault_path = Path(vault_path)
    self.config = config or EmbeddingConfig()
    self.include_patterns = include_patterns
    self.exclude_patterns = exclude_patterns or []
    self.max_workers = max_workers
    # ... rest of init
```

**Step 2: Add path filtering to load method**

In the `load` method, add filtering logic:

```python
def _should_include_file(self, file_path: Path) -> bool:
    """Check if file should be included based on path patterns."""
    rel_path = str(file_path.relative_to(self.vault_path))

    # Check include patterns (if specified, must match one)
    if self.include_patterns:
        import fnmatch
        matched = any(fnmatch.fnmatch(rel_path, p) for p in self.include_patterns)
        if not matched:
            return False

    # Check exclude patterns
    if self.exclude_patterns:
        import fnmatch
        excluded = any(fnmatch.fnmatch(rel_path, p) for p in self.exclude_patterns)
        if excluded:
            return False

    return True
```

**Step 3: Verify**

```bash
python -c "from loader import ObsidianLoader; print('OK')"
```

**Step 4: Commit**

```bash
git add loader.py
git commit -m "feat(loader): add include/exclude path filtering"
```

---

## Task 4: Create index_manager.py for multi-index operations

**Files:**
- Create: `index_manager.py`

**Step 1: Create the index manager module**

```python
"""Multi-index management for UltraRAG.

Handles creation, loading, and querying of partitioned indexes
(personal vs research content).
"""
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex

from config import RAGConfig, IndexConfig
from settings_store import get_index_configs, set_index_configs, get_index_config
from research_mode import IndexProfile, ConvergenceConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadedIndex:
    """Represents a loaded index partition."""
    name: str
    config: IndexConfig
    index: VectorStoreIndex
    nodes: List[Any]  # For BM25
    convergence_config: ConvergenceConfig
    wikilink_graph: Optional[Dict[str, List[str]]] = None


class IndexManager:
    """Manages multiple index partitions."""

    def __init__(self, rag_config: RAGConfig):
        """Initialize index manager.

        Args:
            rag_config: RAG configuration
        """
        self.config = rag_config
        self.db_path = rag_config.vector_db.lancedb_path
        self.indexes: Dict[str, LoadedIndex] = {}

    def get_convergence_config(self, profile_name: str) -> ConvergenceConfig:
        """Get convergence config for a profile name."""
        profile_map = {
            "personal": IndexProfile.PERSONAL,
            "research": IndexProfile.RESEARCH,
            "balanced": IndexProfile.BALANCED
        }
        profile = profile_map.get(profile_name, IndexProfile.BALANCED)
        return ConvergenceConfig.for_profile(profile)

    def get_index_configs_from_db(self) -> List[dict]:
        """Get index configurations from settings store."""
        configs = get_index_configs(str(self.db_path))
        if not configs:
            # Return defaults if not configured
            return [
                {
                    "name": "personal",
                    "table_name": "vault_personal",
                    "source_paths": [],
                    "exclude_paths": [],
                    "convergence_profile": "personal",
                    "weight": 1.0,
                    "enabled": True
                },
                {
                    "name": "research",
                    "table_name": "vault_research",
                    "source_paths": [],
                    "exclude_paths": [],
                    "convergence_profile": "research",
                    "weight": 1.0,
                    "enabled": True
                }
            ]
        return configs

    def save_index_configs(self, configs: List[dict]) -> None:
        """Save index configurations to settings store."""
        set_index_configs(str(self.db_path), configs)

    def load_index(self, name: str) -> Optional[LoadedIndex]:
        """Load a specific index partition.

        Args:
            name: Index name (e.g., "personal", "research")

        Returns:
            LoadedIndex or None if not found/disabled
        """
        config_dict = get_index_config(str(self.db_path), name)
        if not config_dict:
            logger.warning(f"Index config not found: {name}")
            return None

        if not config_dict.get("enabled", True):
            logger.info(f"Index disabled: {name}")
            return None

        # TODO: Implement actual index loading from LanceDB table
        # This will be connected in Task 6
        logger.info(f"Index manager ready for: {name}")
        return None

    def get_active_indexes(self) -> List[str]:
        """Get list of active (enabled) index names."""
        configs = self.get_index_configs_from_db()
        return [c["name"] for c in configs if c.get("enabled", True)]
```

**Step 2: Verify**

```bash
python -c "from index_manager import IndexManager; print('OK')"
```

**Step 3: Commit**

```bash
git add index_manager.py
git commit -m "feat: add IndexManager for multi-index partitions"
```

---

## Task 5: Add UI controls for index selection

**Files:**
- Modify: `app.py`

**Step 1: Add index scope selector to sidebar**

Find the sidebar section and add after the search scope toggle:

```python
# Index scope selection (if partitioned indexes enabled)
if st.session_state.get('partitioned_indexes_enabled', False):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Index Scope")

    index_scope = st.sidebar.radio(
        "Query from:",
        options=["All", "Personal Only", "Research Only"],
        index=0,
        help="Choose which index partition to search"
    )

    # Map to source filter
    scope_map = {
        "All": None,  # Query all
        "Personal Only": ["vault_personal"],
        "Research Only": ["vault_research"]
    }
    st.session_state.index_scope_filter = scope_map[index_scope]
```

**Step 2: Add index configuration expander**

```python
# Index Configuration (in Settings section)
with st.sidebar.expander("📂 Index Partitions", expanded=False):
    st.markdown("Configure personal vs research content")

    # Toggle for partitioned indexes
    use_partitioned = st.checkbox(
        "Enable partitioned indexes",
        value=st.session_state.get('partitioned_indexes_enabled', False),
        help="Separate personal notes from research content"
    )
    st.session_state.partitioned_indexes_enabled = use_partitioned

    if use_partitioned:
        st.markdown("**Personal Index:**")
        personal_paths = st.text_area(
            "Include paths (one per line)",
            value="Areas/Personal/**\nAreas/Journal/**",
            height=80,
            key="personal_paths"
        )

        st.markdown("**Research Index:**")
        research_paths = st.text_area(
            "Include paths (one per line)",
            value="Resources/**\nAreas/Research/**",
            height=80,
            key="research_paths"
        )

        if st.button("Save Partition Config"):
            # Save to settings store
            from settings_store import set_index_configs
            configs = [
                {
                    "name": "personal",
                    "table_name": "vault_personal",
                    "source_paths": [p.strip() for p in personal_paths.split('\n') if p.strip()],
                    "convergence_profile": "personal",
                    "weight": 1.0,
                    "enabled": True
                },
                {
                    "name": "research",
                    "table_name": "vault_research",
                    "source_paths": [p.strip() for p in research_paths.split('\n') if p.strip()],
                    "convergence_profile": "research",
                    "weight": 1.0,
                    "enabled": True
                }
            ]
            set_index_configs(str(config.vector_db.lancedb_path), configs)
            st.success("Partition config saved!")
            st.info("Re-index required to apply changes")
```

**Step 3: Verify**

Run Streamlit and check UI renders:

```bash
streamlit run app.py
```

**Step 4: Commit**

```bash
git add app.py
git commit -m "feat(ui): add index partition configuration and scope selection"
```

---

## Task 6: Integrate with vector_store.py for table creation

**Files:**
- Modify: `vector_store.py`

**Step 1: Add get_partitioned_vector_store function**

```python
def get_partitioned_vector_store(
    config: VectorDBConfig,
    table_name: str,
    mode: str = "append"
) -> Any:
    """Get vector store for a specific partition table.

    Args:
        config: Vector DB configuration
        table_name: Name of the LanceDB table (e.g., "vault_personal")
        mode: "append" or "overwrite"

    Returns:
        LanceDBVectorStore instance
    """
    from llama_index.vector_stores.lancedb import LanceDBVectorStore

    vector_store = LanceDBVectorStore(
        uri=str(config.lancedb_path),
        table_name=table_name,
        mode=mode
    )

    logger.info(f"Initialized partitioned vector store: {table_name} (mode={mode})")
    return vector_store


def partitioned_index_exists(config: VectorDBConfig, table_name: str) -> bool:
    """Check if a specific partition table exists.

    Args:
        config: Vector DB configuration
        table_name: Table name to check

    Returns:
        True if table exists
    """
    import lancedb

    if not config.lancedb_path.exists():
        return False

    try:
        db = lancedb.connect(str(config.lancedb_path))
        return table_name in db.table_names()
    except Exception:
        return False
```

**Step 2: Verify**

```bash
python -c "from vector_store import get_partitioned_vector_store, partitioned_index_exists; print('OK')"
```

**Step 3: Commit**

```bash
git add vector_store.py
git commit -m "feat(vector_store): add partitioned table support"
```

---

## Task 7: Update main.py for partitioned indexing

**Files:**
- Modify: `main.py`

**Step 1: Add partition-aware indexing command**

Add new CLI command handling:

```python
# In the command loop, add:
elif cmd == "index-partitions" or cmd == "ip":
    print("\n📂 Partitioned Indexing")
    print("=" * 40)

    from index_manager import IndexManager
    from settings_store import get_index_configs

    manager = IndexManager(self.config)
    configs = manager.get_index_configs_from_db()

    if not configs:
        print("No partition configs found. Configure in UI first.")
        continue

    for idx_config in configs:
        if not idx_config.get("enabled", True):
            print(f"Skipping disabled index: {idx_config['name']}")
            continue

        print(f"\nIndexing: {idx_config['name']}")
        print(f"  Table: {idx_config['table_name']}")
        print(f"  Paths: {idx_config.get('source_paths', ['all'])}")

        # Create loader with path filtering
        loader = ObsidianLoader(
            self.config.vault_path,
            self.config.embedding,
            include_patterns=idx_config.get("source_paths") or None,
            exclude_patterns=idx_config.get("exclude_paths") or None
        )

        # Load and index
        nodes = loader.load()
        print(f"  Loaded {len(nodes)} documents")

        # Create index in specific table
        from vector_store import get_partitioned_vector_store
        vector_store = get_partitioned_vector_store(
            self.config.vector_db,
            idx_config['table_name'],
            mode="overwrite"
        )

        # Build index
        from llama_index.core import VectorStoreIndex, StorageContext
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )

        print(f"  ✓ Indexed {len(nodes)} nodes to {idx_config['table_name']}")

    print("\n✓ Partitioned indexing complete!")
```

**Step 2: Update help command**

```python
# Add to help text:
"index-partitions (ip)  - Index vault by partitions (personal/research)"
```

**Step 3: Verify**

```bash
python main.py
# Then type: help
# Should show new command
```

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat(cli): add partitioned indexing command"
```

---

## Task 8: Connect research mode to use partition-specific convergence

**Files:**
- Modify: `query_engine.py` or wherever ResearchRetriever is instantiated

**Step 1: Pass convergence config based on active index**

When creating ResearchRetriever, determine profile from active scope:

```python
# In the query building code:
from research_mode import IndexProfile, ConvergenceConfig

# Determine profile based on query scope
scope_filter = st.session_state.get('index_scope_filter')
if scope_filter == ["vault_personal"]:
    profile = IndexProfile.PERSONAL
elif scope_filter == ["vault_research"]:
    profile = IndexProfile.RESEARCH
else:
    profile = IndexProfile.BALANCED

# Create research retriever with appropriate config
research_retriever = ResearchRetriever(
    base_retriever=base_retriever,
    llm=llm,
    index_profile=profile,  # Uses the new parameter
    enable_research=config.retrieval.enable_research_mode
)
```

**Step 2: Verify**

Test with different scope selections in UI.

**Step 3: Commit**

```bash
git add query_engine.py app.py
git commit -m "feat: connect research mode convergence to index partition profile"
```

---

## Task 9: Write integration tests

**Files:**
- Create: `tests/test_partitioned_index.py`

**Step 1: Create test file**

```python
"""Tests for partitioned index functionality."""
import pytest
from pathlib import Path
import tempfile
import shutil

from config import IndexConfig, VectorDBConfig
from settings_store import (
    get_index_configs,
    set_index_configs,
    get_index_config,
    update_index_config
)
from research_mode import IndexProfile, ConvergenceConfig


class TestIndexConfig:
    """Test IndexConfig model."""

    def test_default_values(self):
        config = IndexConfig(name="test", table_name="test_table")
        assert config.convergence_profile == "balanced"
        assert config.weight == 1.0
        assert config.enabled is True

    def test_invalid_profile_raises(self):
        with pytest.raises(ValueError):
            IndexConfig(
                name="test",
                table_name="test",
                convergence_profile="invalid"
            )


class TestConvergenceProfiles:
    """Test convergence config profiles."""

    def test_personal_profile_more_aggressive(self):
        personal = ConvergenceConfig.for_profile(IndexProfile.PERSONAL)
        research = ConvergenceConfig.for_profile(IndexProfile.RESEARCH)

        # Personal should stop earlier (higher threshold)
        assert personal.info_gain_threshold > research.info_gain_threshold
        # Personal should have fewer iterations
        assert personal.max_iterations <= research.max_iterations

    def test_research_profile_more_tolerant(self):
        research = ConvergenceConfig.for_profile(IndexProfile.RESEARCH)
        personal = ConvergenceConfig.for_profile(IndexProfile.PERSONAL)

        # Research should tolerate more redundancy
        assert research.redundancy_threshold > personal.redundancy_threshold


class TestSettingsStorage:
    """Test index config persistence."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary LanceDB directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_save_and_load_configs(self, temp_db):
        configs = [
            {"name": "personal", "table_name": "vault_personal", "enabled": True},
            {"name": "research", "table_name": "vault_research", "enabled": True}
        ]

        set_index_configs(temp_db, configs)
        loaded = get_index_configs(temp_db)

        assert len(loaded) == 2
        assert loaded[0]["name"] == "personal"

    def test_get_specific_config(self, temp_db):
        configs = [
            {"name": "personal", "table_name": "vault_personal"},
            {"name": "research", "table_name": "vault_research"}
        ]
        set_index_configs(temp_db, configs)

        personal = get_index_config(temp_db, "personal")
        assert personal["table_name"] == "vault_personal"

        missing = get_index_config(temp_db, "nonexistent")
        assert missing is None
```

**Step 2: Run tests**

```bash
pytest tests/test_partitioned_index.py -v
```

**Step 3: Commit**

```bash
git add tests/test_partitioned_index.py
git commit -m "test: add partitioned index integration tests"
```

---

## Task 10: Update documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/SESSION_2026-01-25_TOKEN_TRACKING_AND_CONVERGENCE.md`

**Step 1: Add section to CLAUDE.md**

```markdown
## Partitioned Indexes (Personal vs Research)

Enable separate indexes for different content types:

```bash
# In .env or via UI Settings
USE_PARTITIONED_INDEXES=true
```

### Configuration

In Streamlit UI: Settings → Index Partitions

- **Personal Index:** Quick answers, aggressive convergence (10% threshold)
- **Research Index:** Comprehensive coverage, conservative convergence (5% threshold)

### CLI Commands

```bash
python main.py
# Then type:
index-partitions  # or 'ip' - Index by partitions
```

### Query Scope

Use the "Index Scope" selector in sidebar:
- All: Query both indexes
- Personal Only: Quick lookups
- Research Only: Deep research
```

**Step 2: Update session report**

Add to the session report doc.

**Step 3: Commit**

```bash
git add CLAUDE.md docs/
git commit -m "docs: add partitioned index documentation"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add IndexConfig to config.py | config.py |
| 2 | Add settings_store functions | settings_store.py |
| 3 | Add path filtering to loader | loader.py |
| 4 | Create IndexManager | index_manager.py (new) |
| 5 | Add UI controls | app.py |
| 6 | Add partitioned table support | vector_store.py |
| 7 | Add CLI indexing command | main.py |
| 8 | Connect convergence to partitions | query_engine.py, app.py |
| 9 | Write integration tests | tests/test_partitioned_index.py (new) |
| 10 | Update documentation | CLAUDE.md, docs/ |

**Total commits:** 10
**Estimated new code:** ~500 lines
