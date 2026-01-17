# File Exclusions

Exclude specific files or folders from indexing using pattern matching. Exclusions are persisted in LanceDB and applied automatically during indexing.

## Quick Start

1. Open the Streamlit web interface: `streamlit run app.py`
2. Click **⚙️ Settings** in the sidebar (appears after index is loaded)
3. Add exclusion patterns
4. Preview matches before applying
5. Patterns are saved and applied immediately

## Pattern Types

### Glob Patterns (Recommended)
Shell-style wildcards for matching file paths.

| Pattern | Matches |
|---------|---------|
| `Archive/**` | All files in Archive/ folder |
| `*.excalidraw.md` | All Excalidraw diagram files |
| `**/drafts/*` | Any file in any 'drafts' folder |
| `**/*.canvas` | All canvas files anywhere |
| `Templates/*.md` | Markdown files in Templates/ only |

### Exact Paths
Match a specific file or folder path exactly.

| Pattern | Matches |
|---------|---------|
| `Projects/old-project.md` | Only that specific file |
| `Archive/2023` | Only files in that exact path |

### Regular Expressions
Full regex support for complex patterns.

| Pattern | Matches |
|---------|---------|
| `^Daily.*2023` | Daily notes from 2023 |
| `.*\\.excalidraw\\.md$` | Excalidraw files (regex) |
| `^(Archive\|Trash)/` | Archive or Trash folders |

## How It Works

### Storage
Exclusion patterns are stored in a `settings` table in LanceDB:
```
data/lancedb/settings/
```

The settings table uses a generic key-value structure:
- **key**: `"file_exclusions"`
- **value**: JSON array of patterns with metadata

### Live Removal
When you add an exclusion pattern:
1. Pattern is saved to settings
2. Matching files are immediately deleted from the vector index
3. Docstore cache is invalidated
4. Next query will not return excluded files

### Indexing Integration
Exclusions are applied at multiple points:
- **Initial indexing**: Files matching patterns are skipped
- **Wikilink graph**: Excluded files don't appear in the graph
- **RAPTOR indexing**: Excluded files are not summarized

## CLI Usage

The exclusion system can also be used programmatically:

```python
from settings_store import get_exclusions, add_exclusion, remove_exclusion
from exclusion_matcher import ExclusionMatcher, preview_exclusions
from config import load_config

config = load_config()
db_path = str(config.vector_db.lancedb_path)

# Add an exclusion
add_exclusion(db_path, "Archive/**", "glob")

# List all exclusions
exclusions = get_exclusions(db_path)
for exc in exclusions:
    print(f"{exc['pattern']} ({exc['type']})")

# Preview what would be excluded
preview = preview_exclusions(
    [{"pattern": "*.excalidraw.md", "type": "glob"}],
    config.vault_path
)
print(f"Would exclude {preview['excluded_count']} files")

# Remove an exclusion
remove_exclusion(db_path, "Archive/**", "glob")
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  settings_store │────>│ exclusion_matcher│────>│    loader.py    │
│    (LanceDB)    │     │  (pattern match) │     │  (load_vault)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                               ┌──────────────────────────┘
                               ▼
                        ┌─────────────────┐
                        │   vector_store  │
                        │ (delete_from_   │
                        │     index)      │
                        └─────────────────┘
```

### Key Files
- `settings_store.py` - Generic settings persistence with exclusion helpers
- `exclusion_matcher.py` - Pattern matching engine (exact, glob, regex)
- `vector_store.py` - `delete_from_index()` for live removal
- `app.py` - Settings dialog UI

## Common Use Cases

### Exclude Archive/Old Content
```
Archive/**
Trash/**
_Archive/**
```

### Exclude Diagram Files
```
*.excalidraw.md
*.canvas
```

### Exclude Date-Based Content
```
Daily Notes/2023-*
Daily Notes/2022-*
```
Or with regex:
```
^Daily Notes/202[0-3]
```

### Exclude Templates
```
Templates/**
_templates/**
```

### Exclude Specific Projects
```
Projects/deprecated-project/**
Work/old-client/**
```

## Re-including Files

To re-include previously excluded files:

1. Remove the exclusion pattern from Settings
2. Re-index the vault to add the files back

Note: Removing an exclusion does NOT automatically re-add files to the index. A re-index is required.

## Troubleshooting

### Pattern Not Matching Expected Files
- Use **Preview Matches** before adding to verify
- Check path separators (use `/` not `\`)
- Glob patterns are case-sensitive
- Paths are relative to vault root

### Files Still Appearing in Results
- Check if pattern was saved (view Current Exclusions)
- Verify the cache was invalidated
- Try a fresh query

### Regex Errors
- Escape special characters: `\.` for literal dot
- Test regex separately before adding
- Check for unbalanced parentheses/brackets
