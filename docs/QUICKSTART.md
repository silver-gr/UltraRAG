# UltraRAG Quick Start Guide

## 🎯 Goal
Get your Obsidian RAG system running in under 10 minutes.

## 📋 Prerequisites
- Python 3.10 or higher
- Your Obsidian vault
- At least one API key:
  - **Recommended**: Voyage AI (for embeddings + reranking) + Google (for Gemini)
  - **Free option**: Google API only (Gemini for both embeddings and LLM)

## 🚀 Installation (5 minutes)

### Step 1: Run setup script
```bash
cd /Users/silver/Projects/UltraRAG
chmod +x setup.sh
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Create a .env configuration file

### Step 2: Configure your system
Edit the `.env` file:

```bash
# REQUIRED: Set your vault path
OBSIDIAN_VAULT_PATH=/Users/your-name/Documents/ObsidianVault

# REQUIRED: Add API keys
VOYAGE_API_KEY=pa-xxx...  # Get from https://www.voyageai.com/
GOOGLE_API_KEY=AIza...    # Get from https://makersuite.google.com/

# OPTIONAL: Customize settings (defaults are optimized)
EMBEDDING_MODEL=voyage-3-large
CHUNK_SIZE=512
TOP_K=75
```

### Step 3: Get API Keys (2 minutes)

#### Voyage AI (Recommended for best quality)
1. Go to https://www.voyageai.com/
2. Sign up for free account
3. Get API key from dashboard
4. Free tier: $200 credits (~5-10M tokens)

#### Google AI Studio (Required for LLM)
1. Go to https://makersuite.google.com/
2. Sign in with Google account
3. Create API key
4. Free tier: Generous limits

## 🎮 Usage

### Option 1: Command Line Interface
```bash
source venv/bin/activate
python main.py
```

Follow the prompts to:
1. Index your vault (one-time, 10-30 minutes)
2. Start querying your notes

### Option 2: Web Interface (Recommended)
```bash
source venv/bin/activate
streamlit run app.py
```

This opens a web UI at http://localhost:8501

## 💡 First Queries to Try

Once indexed, try these queries:

```
What are my main areas of interest?
```

```
Summarize my notes about [your favorite topic]
```

```
What connections exist between [concept A] and [concept B]?
```

```
Show me notes tagged with #important
```

## 🎛️ Configuration Options

### Budget-Conscious Setup (Free)
```bash
# Use Gemini for everything (free tier)
EMBEDDING_MODEL=gemini-embed
GOOGLE_API_KEY=your_key
# No VOYAGE_API_KEY needed
```

### Maximum Quality Setup (Recommended)
```bash
# Best embeddings + reranking
EMBEDDING_MODEL=voyage-3-large
VOYAGE_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

### Self-Hosted Setup (No API costs)
```bash
# Requires GPU with 16-32GB VRAM
EMBEDDING_MODEL=qwen3-8b
# No API keys needed
```

## 📊 What to Expect

### Indexing
- **Time**: 10-30 minutes for 1,650 notes
- **Progress**: Shows progress bar
- **One-time**: Only needed once, then incremental

### Queries
- **Simple queries**: <1 second
- **Complex queries**: <3 seconds
- **Quality**: 85-95% retrieval accuracy

### Costs (with API models)
- **Initial indexing**: $13-40 one-time
- **Monthly usage**: $5-20 (moderate use)
- **Per query**: $0.001-0.01

## 🐛 Troubleshooting

### "OBSIDIAN_VAULT_PATH not found"
→ Edit `.env` and set the correct path to your vault

### "API key not found"
→ Make sure you added your keys to `.env` file

### "Out of memory"
→ Reduce `CHUNK_SIZE` in `.env` to 256

### Slow indexing
→ Normal for first time! 10-30 min is expected

### Poor results
→ Try adjusting `TOP_K` (increase to 100)
→ Enable reranking (automatic with Voyage API key)

## 📚 Next Steps

1. **Try different queries** - Explore your knowledge base
2. **Read the README.md** - Learn about advanced features
3. **Customize settings** - Tune for your specific needs
4. **Phase 2 features** - Graph search, temporal filtering (coming soon)

## 🆘 Need Help?

- Check `README.md` for detailed documentation
- Review `compass_artifact_*.md` for the full strategy
- Common issues: See "Troubleshooting" in README.md

---

**You're all set! Start exploring your knowledge base with AI-powered search.** 🎉
