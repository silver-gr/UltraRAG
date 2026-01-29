# LangChain Agents Reference Guide

This document provides a quick reference for building ReAct-style agents with LangChain,
based on the LangSmith onboarding example. Useful for future UltraRAG extensions.

## Prerequisites

### Install Dependencies

```bash
pip install --pre -U langchain langchain-openai
```

> **Note**: The `--pre` flag installs pre-release versions for latest features.

### Environment Variables

```bash
# LangSmith tracing (required for observability)
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com  # EU endpoint
export LANGSMITH_API_KEY=lsv2_pt_your_key_here
export LANGSMITH_PROJECT="your-project-name"

# LLM provider (choose one)
export OPENAI_API_KEY=<your-openai-api-key>
# or
export GOOGLE_API_KEY=<your-google-api-key>  # For Gemini
```

## Basic ReAct Agent Example

A ReAct (Reasoning + Acting) agent that can:
1. Use tools to gather information
2. Reason about the results
3. Decide next actions

### Simple Tool Definition

```python
from langchain.agents import create_agent

# Define a simple tool
def get_weather(city: str) -> str:
    """Get weather for a given city.

    This docstring becomes the tool description that the LLM sees.
    Make it clear and descriptive!
    """
    # In production, this would call a real weather API
    return f"It's always sunny in {city}!"

# Create the agent
agent = create_agent(
    model="openai:gpt-5-mini",  # Model identifier
    tools=[get_weather],        # List of callable tools
    system_prompt="You are a helpful assistant.",
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
)
```

### What Happens Under the Hood

1. **User sends message** → Agent receives query
2. **LLM reasons** → Decides to use `get_weather` tool
3. **Tool execution** → Calls `get_weather("San Francisco")`
4. **LLM synthesizes** → Generates final response with tool output
5. **Trace logged** → All steps visible in LangSmith dashboard

## Advanced: Multiple Tools

```python
from langchain.agents import create_agent
from langchain_community.tools import TavilySearchResults

# Define custom tools
def search_vault(query: str) -> str:
    """Search the Obsidian vault for relevant notes.

    Use this when you need information from the user's knowledge base.
    """
    # Integration point with UltraRAG!
    from ultrarag import search
    results = search(query, top_k=5)
    return "\n".join([r.text for r in results])

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Use this for any calculations.
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# Web search tool (requires Tavily API key)
web_search = TavilySearchResults(max_results=3)

# Create agent with multiple tools
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_vault, calculate, web_search],
    system_prompt="""You are a research assistant with access to:
    - A personal knowledge vault (search_vault)
    - Web search (tavily_search_results)
    - Calculator (calculate)

    Always search the vault first before using web search.
    Cite your sources.""",
)
```

## Using with Google Gemini

```python
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Create agent with Gemini
agent = create_agent(
    model="google:gemini-3-flash",  # or "google:gemini-3-pro"
    tools=[search_vault, web_search],
    system_prompt="You are a helpful research assistant.",
)

# Run with streaming
async for chunk in agent.astream(
    {"messages": [{"role": "user", "content": "Research quantum computing"}]}
):
    print(chunk, end="", flush=True)
```

## Integration Ideas for UltraRAG

### 1. Research Agent

```python
# Agent that iteratively researches a topic
def create_research_agent():
    tools = [
        search_vault,      # Search Obsidian notes
        web_search,        # Search the web
        save_to_vault,     # Save findings as new notes
    ]

    return create_agent(
        model="google:gemini-3-flash",
        tools=tools,
        system_prompt="""You are a research agent. For each query:
        1. First search the vault for existing knowledge
        2. Identify gaps in the information
        3. Use web search to fill gaps
        4. Synthesize findings into a comprehensive answer
        5. Optionally save new learnings to the vault""",
    )
```

### 2. Note-Taking Agent

```python
# Agent that helps organize and link notes
def create_note_agent():
    tools = [
        search_vault,
        find_related_notes,
        create_note,
        add_wikilink,
    ]

    return create_agent(
        model="google:gemini-3-flash",
        tools=tools,
        system_prompt="""You are a note-taking assistant. Help users:
        - Find connections between notes
        - Suggest related topics
        - Create new notes with proper tags and links
        - Maintain consistency in the knowledge base""",
    )
```

## Tracing and Debugging

All agent runs are automatically traced in LangSmith. View:

1. **Trajectory**: Step-by-step execution path
2. **Tool calls**: Which tools were called and with what arguments
3. **LLM inputs/outputs**: Full prompts and responses
4. **Latency**: Time spent on each step
5. **Token usage**: Cost tracking

### Access Traces Programmatically

```python
from langsmith import Client

client = Client()

# Get recent runs
runs = client.list_runs(
    project_name="ultrarag",
    execution_order=1,
    limit=10
)

for run in runs:
    print(f"Run: {run.name}")
    print(f"Status: {run.status}")
    print(f"Duration: {run.end_time - run.start_time}")
    print(f"Tools used: {[child.name for child in run.child_runs]}")
    print("---")
```

## Best Practices

### 1. Tool Docstrings Matter

The LLM uses docstrings to understand when to use each tool:

```python
# Good - clear and specific
def search_vault(query: str) -> str:
    """Search the personal knowledge vault for notes matching the query.

    Use this tool when you need information that might be in the user's
    existing notes, such as personal preferences, past decisions, or
    domain-specific knowledge they've collected.

    Args:
        query: Natural language search query

    Returns:
        Relevant text excerpts from matching notes
    """

# Bad - vague
def search(q: str) -> str:
    """Search stuff."""
```

### 2. System Prompts

Be explicit about:
- Tool usage priorities
- Output format expectations
- When NOT to use certain tools

### 3. Error Handling

```python
def robust_tool(input: str) -> str:
    """Tool with proper error handling."""
    try:
        result = do_something(input)
        return result
    except SpecificError as e:
        return f"Could not process: {e}. Try rephrasing your request."
    except Exception as e:
        # Log but don't expose internal errors
        logger.error(f"Tool error: {e}")
        return "An error occurred. Please try again."
```

## Resources

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangChain Agents Guide](https://python.langchain.com/docs/concepts/agents/)
- [Tool Calling](https://python.langchain.com/docs/concepts/tool_calling/)
- [LangGraph (Advanced)](https://langchain-ai.github.io/langgraph/) - For complex agent workflows

---

*This reference was created during UltraRAG LangSmith integration (January 2026)*
