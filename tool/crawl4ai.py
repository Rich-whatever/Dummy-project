from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import (
    BM25ContentFilter,
    PruningContentFilter,
)
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from langchain_core.tools import tool

RAW_LIMIT = 20000
DEAD_LIMIT = 40000


def _process_crawl_result(result, url: str) -> str:
    """
    Decide whether raw markdown or filtered markdown is safe to return.
    Never returns bare "" — always surfaces a diagnostic on failure.
    """
    # Crawl-level failure: report it instead of silently returning ""
    if not getattr(result, "success", True):
        error_message = getattr(result, "error_message", "") or "unknown error"
        return f"Crawl failed for {url}: {error_message}"

    raw = result.markdown.raw_markdown or ""
    fit = result.markdown.fit_markdown or ""

    # Raw content is already reasonable
    if len(raw) <= RAW_LIMIT:
        if raw:
            return raw
    # Raw too large, use filtered version
    elif len(fit) <= DEAD_LIMIT:
        return fit
    else:
        return (
            f"The webpage at {url} is too large to process reliably "
            "Try another source or a more specific page."
        )

    # Both raw and fit are empty — surface diagnostics so the worker
    # can see WHY instead of a bare "(no output)".
    html_len = len(getattr(result, "html", "") or "")
    error_message = getattr(result, "error_message", "") or "none"
    return (
        f"Webpage at {url} rendered no extractable content "
        f"(success={getattr(result, 'success', '?')}, html_bytes={html_len}, "
        f"error_message={error_message}). The page is likely JavaScript-rendered "
        "and the snapshot was taken before content appeared. Retry, or use "
        "browser_automation instead."
    )


@tool
async def fetch_webpage(url: str) -> str:
    """
    Fetch a webpage and return cleaned markdown content.
    Use for general webpage reading.
    """

    markdown_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter()
    )

    config = CrawlerRunConfig(
        markdown_generator=markdown_generator,
        word_count_threshold=20,
        cache_mode=CacheMode.BYPASS,  # never serve (or store) a cached failure
        wait_until="load",            # render JS but bounded (networkidle can hang)
        page_timeout=25000,           # cap total page-load wait
        wait_for="body",
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
            config=config
        )

    return _process_crawl_result(result, url)


@tool
async def fetch_relevant_webpage(url: str, query: str) -> str:
    """
    Fetch a webpage and keep content relevant to a query.
    Use for finding specific information from url source.
    """

    markdown_generator = DefaultMarkdownGenerator(
        content_filter=BM25ContentFilter(
            user_query=query
        )
    )

    config = CrawlerRunConfig(
        markdown_generator=markdown_generator,
        word_count_threshold=20,
        cache_mode=CacheMode.BYPASS,  # never serve (or store) a cached failure
        wait_until="load",            # render JS but bounded (networkidle can hang)
        page_timeout=25000,           # cap total page-load wait
        wait_for="body",
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
            config=config
        )

    return _process_crawl_result(result, url)
