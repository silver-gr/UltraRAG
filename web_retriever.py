"""Web search retriever using Tavily API with optional Crawl4AI deep extraction."""
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import os
import logging
import asyncio

from llama_index.core.schema import NodeWithScore, TextNode

if TYPE_CHECKING:
    from crawl4ai import AsyncWebCrawler

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """Result from a web search."""
    title: str
    url: str
    content: str
    score: float


class WebRetriever:
    """Web search retriever using Tavily API.

    Provides web search capabilities that return results compatible with
    LlamaIndex NodeWithScore format, enabling integration with the federated
    retrieval system.

    Example:
        >>> retriever = WebRetriever()
        >>> nodes = retriever.retrieve("latest Python features", max_results=5)
        >>> for node in nodes:
        ...     print(f"{node.node.metadata['title']}: {node.score}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        deep_mode: bool = False
    ):
        """Initialize the web retriever.

        Args:
            api_key: Tavily API key. If not provided, reads from TAVILY_API_KEY env var.
            max_results: Default number of results to return.
            deep_mode: If True, use Crawl4AI to fetch full page content instead of snippets.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.max_results = max_results
        self.deep_mode = deep_mode
        self.client = None
        self._crawler: Optional["AsyncWebCrawler"] = None
        self._crawl4ai_available: Optional[bool] = None

        if not self.api_key:
            logger.warning("TAVILY_API_KEY not found. Web search will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            try:
                from tavily import TavilyClient
                self.client = TavilyClient(api_key=self.api_key)
                logger.info("Tavily web search initialized successfully")
            except ImportError:
                logger.warning(
                    "tavily-python not installed. "
                    "Run: pip install tavily-python"
                )
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Tavily client: {e}")
                self.enabled = False

    def search(self, query: str, max_results: Optional[int] = None) -> List[WebSearchResult]:
        """Execute web search and return results.

        Args:
            query: Search query string.
            max_results: Number of results to return. Defaults to self.max_results.

        Returns:
            List of WebSearchResult objects.
        """
        if not self.enabled:
            logger.debug("Web search disabled, returning empty results")
            return []

        num_results = max_results or self.max_results

        try:
            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=num_results
            )

            results = []
            for item in response.get("results", []):
                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.5)
                ))

            logger.info(f"Web search returned {len(results)} results for: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def retrieve(self, query: str, max_results: Optional[int] = None) -> List[NodeWithScore]:
        """Retrieve web results as LlamaIndex NodeWithScore objects.

        This method is compatible with LlamaIndex retrievers and can be used
        in federated retrieval pipelines.

        Args:
            query: Search query string.
            max_results: Number of results to return. Defaults to self.max_results.

        Returns:
            List of NodeWithScore objects with web search results.
        """
        search_results = self.search(query, max_results)

        nodes = []
        for result in search_results:
            node = TextNode(
                text=result.content,
                metadata={
                    "title": result.title,
                    "url": result.url,
                    "source_type": "web",
                    "retrieval_source": "tavily"
                }
            )
            nodes.append(NodeWithScore(node=node, score=result.score))

        return nodes

    def is_available(self) -> bool:
        """Check if web search is available.

        Returns:
            True if Tavily API is configured and client is initialized.
        """
        return self.enabled and self.client is not None

    def _check_crawl4ai_available(self) -> bool:
        """Check if Crawl4AI is available for import.

        Returns:
            True if crawl4ai can be imported.
        """
        if self._crawl4ai_available is None:
            try:
                import crawl4ai  # noqa: F401
                self._crawl4ai_available = True
                logger.debug("Crawl4AI is available for deep extraction")
            except ImportError:
                self._crawl4ai_available = False
                logger.warning(
                    "crawl4ai not installed. Deep extraction disabled. "
                    "Run: pip install crawl4ai"
                )
        return self._crawl4ai_available

    async def _ensure_crawler(self) -> Optional["AsyncWebCrawler"]:
        """Lazily initialize the Crawl4AI crawler.

        Returns:
            AsyncWebCrawler instance or None if unavailable.
        """
        if self._crawler is not None:
            return self._crawler

        if not self._check_crawl4ai_available():
            return None

        try:
            from crawl4ai import AsyncWebCrawler
            self._crawler = AsyncWebCrawler()
            await self._crawler.__aenter__()
            logger.info("Crawl4AI crawler initialized successfully")
            return self._crawler
        except Exception as e:
            logger.error(f"Failed to initialize Crawl4AI crawler: {e}")
            return None

    async def _fetch_content_with_crawl4ai(self, url: str) -> Optional[str]:
        """Fetch full page content using Crawl4AI.

        Args:
            url: URL to fetch content from.

        Returns:
            Markdown content or None if fetch failed.
        """
        crawler = await self._ensure_crawler()
        if crawler is None:
            return None

        try:
            result = await crawler.arun(url=url)
            # crawl4ai 0.8.0+: result.markdown.fit_markdown
            md = result.markdown
            content = getattr(md, 'fit_markdown', None) or getattr(md, 'raw_markdown', str(md))
            if content and len(content.strip()) > 0:
                logger.debug(f"Crawl4AI extracted {len(content)} chars from {url}")
                return content
            else:
                logger.warning(f"Crawl4AI returned empty content for {url}")
                return None
        except Exception as e:
            logger.warning(f"Crawl4AI failed for {url}: {e}")
            return None

    async def retrieve_deep_async(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> List[NodeWithScore]:
        """Retrieve web results with full page content extraction.

        Uses Tavily to find URLs, then Crawl4AI to fetch full markdown content
        from each URL. Falls back to Tavily snippets if extraction fails.

        Args:
            query: Search query string.
            max_results: Number of results to return. Defaults to self.max_results.

        Returns:
            List of NodeWithScore objects with full page content.
        """
        search_results = self.search(query, max_results)

        if not search_results:
            return []

        nodes = []
        for result in search_results:
            # Try deep extraction with Crawl4AI
            full_content = await self._fetch_content_with_crawl4ai(result.url)

            if full_content:
                # Use full extracted content
                content = full_content
                retrieval_source = "tavily+crawl4ai"
            else:
                # Fall back to Tavily snippet
                content = result.content
                retrieval_source = "tavily"
                logger.debug(f"Using Tavily snippet for {result.url}")

            node = TextNode(
                text=content,
                metadata={
                    "title": result.title,
                    "url": result.url,
                    "source_type": "web",
                    "retrieval_source": retrieval_source,
                    "deep_extracted": full_content is not None
                }
            )
            nodes.append(NodeWithScore(node=node, score=result.score))

        deep_count = sum(1 for n in nodes if n.node.metadata.get("deep_extracted"))
        logger.info(
            f"Deep retrieval: {deep_count}/{len(nodes)} pages fully extracted"
        )

        return nodes

    def retrieve_deep(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> List[NodeWithScore]:
        """Synchronous wrapper for retrieve_deep_async.

        Uses Tavily to find URLs, then Crawl4AI to fetch full markdown content
        from each URL. Falls back to Tavily snippets if extraction fails.

        Args:
            query: Search query string.
            max_results: Number of results to return. Defaults to self.max_results.

        Returns:
            List of NodeWithScore objects with full page content.
        """
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            # We're in an async context, need to use nest_asyncio or run in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.retrieve_deep_async(query, max_results)
                )
                return future.result()
        except RuntimeError:
            # No running loop, we can use asyncio.run directly
            return asyncio.run(self.retrieve_deep_async(query, max_results))

    async def close(self) -> None:
        """Close the Crawl4AI crawler if initialized."""
        if self._crawler is not None:
            try:
                await self._crawler.__aexit__(None, None, None)
                logger.debug("Crawl4AI crawler closed")
            except Exception as e:
                logger.warning(f"Error closing Crawl4AI crawler: {e}")
            finally:
                self._crawler = None
